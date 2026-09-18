"""Tests for src/morphology.py — morphological similarity, vector cache,
NBLAST wrapper, on-demand fetch, and homolog enrichment."""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import morphology as morph  # noqa: E402
import roi_screening as rois_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic neuron builders
# ---------------------------------------------------------------------------

def make_neuron(points, parents, radius=1.0):
    """Build a navis TreeNeuron from (x, y, z) points and parent indices
    (-1 = root). Coordinates are in microns."""
    import navis
    n = len(points)
    nodes = pd.DataFrame({
        "node_id": np.arange(n, dtype=np.int64),
        "parent_id": np.array([p if p >= 0 else -1 for p in parents], dtype=np.int64),
        "x": [p[0] for p in points],
        "y": [p[1] for p in points],
        "z": [p[2] for p in points],
        "radius": [radius] * n,
        "type": ["0"] * n,
    })
    return navis.TreeNeuron(nodes)


def line_neuron(length=4, step=1.0):
    pts = [(i * step, 0.0, 0.0) for i in range(length)]
    return make_neuron(pts, [-1] + list(range(length - 1)))


def y_neuron():
    return make_neuron([(0, 0, 0), (1, 0, 0), (2, 1, 0), (2, -1, 0)], [-1, 0, 1, 1])


def bushy_y_neuron():
    """A Y with long arms (richer geometry for NBLAST discrimination)."""
    pts = [(0, 0, 0), (1, 0, 0)]
    for i in range(1, 6):
        pts.append((1 + i, i * 0.4, 0))
        pts.append((1 + i, -i * 0.4, 0))
    # arm points attach alternately to the stem node (1)
    parents = [-1, 0] + [1] * 10
    return make_neuron(pts, parents)


def translated(neuron, shift=(100.0, 50.0, -30.0)):
    """A translated copy of a neuron (identical morphology, different position)."""
    nodes = neuron.nodes.copy()
    for i, axis in enumerate("xyz"):
        nodes[axis] = nodes[axis] + shift[i]
    return type(neuron)(nodes)


def write_skeleton(tmp_path, dataset, body_id, neuron):
    """Persist a neuron to a tmp skeleton cache."""
    folder = tmp_path / "cache" / morph._dataset_folder(dataset) / "skeletons"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{body_id}.pkl", "wb") as f:
        pickle.dump(neuron, f)
    # Raw skeleton fixtures use the shared raw namespace; the legacy
    # find_similar/raw_skeletons path is covered separately as a migration
    # fallback.
    raw_folder = (tmp_path / "cache" / morph._dataset_folder(dataset)
                  / "skeletons" / "raw_skeletons")
    raw_folder.mkdir(parents=True, exist_ok=True)
    with open(raw_folder / f"{body_id}.pkl", "wb") as f:
        pickle.dump(neuron, f)
    return folder / f"{body_id}.pkl"


def write_raw_find_similar_skeleton(tmp_path, dataset, body_id, neuron,
                                    legacy=False):
    """Persist a raw TreeNeuron to the shared or legacy raw cache."""
    folder = (tmp_path / "cache" / morph._dataset_folder(dataset)
              / ("find_similar" if legacy else "skeletons")
              / "raw_skeletons")
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{body_id}.pkl", "wb") as f:
        pickle.dump(neuron, f)
    return folder / f"{body_id}.pkl"


def write_neuron_index(tmp_path, dataset, rows):
    """Persist a bodyId->type/instance index to the tmp cache."""
    folder = tmp_path / "neuron_indexes" / morph._dataset_folder(dataset)
    folder.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["bodyId", "type", "instance"])
    df["bodyId"] = df["bodyId"].astype(np.int64)
    df.to_parquet(folder / "neuron_index.parquet", index=False)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestMorphometrics:
    def test_line_neuron_values(self):
        m = morph.compute_morphometrics(line_neuron())
        assert m["cable_length"] == pytest.approx(3.0)
        assert m["n_nodes"] == 4.0
        assert m["n_branch"] == 0.0
        assert m["n_leaf"] == 1.0
        assert m["n_root"] == 1.0
        assert m["n_primary"] == 1.0
        assert m["bbox_x"] == pytest.approx(3.0)
        assert m["bbox_diagonal"] == pytest.approx(3.0)
        assert m["tortuosity"] == pytest.approx(1.0)
        assert m["strahler_max"] == 1.0

    def test_y_branch_values(self):
        m = morph.compute_morphometrics(y_neuron())
        assert m["n_branch"] == 1.0
        assert m["n_leaf"] == 2.0
        assert m["n_root"] == 1.0
        assert m["strahler_max"] == 2.0
        # orders: branch node 2, root inherits 2, tips 1 -> [2, 2, 1, 1]
        assert m["strahler_mean"] == pytest.approx(1.5)
        # path 1+sqrt(2) over straight sqrt(5) for each tip
        expected = (1 + np.sqrt(2)) / np.sqrt(5)
        assert m["tortuosity"] == pytest.approx(expected, rel=1e-3)

    def test_all_features_finite_and_constant_dimension(self):
        for neuron in (line_neuron(), y_neuron(), translated(line_neuron())):
            m = morph.compute_morphometrics(neuron)
            assert set(m.keys()) == set(morph.MORPHOMETRIC_FEATURES)
            assert all(np.isfinite(v) for v in m.values())

    def test_translated_copy_identical_features(self):
        a = morph.compute_morphometrics(line_neuron())
        b = morph.compute_morphometrics(translated(line_neuron()))
        for k in morph.MORPHOMETRIC_FEATURES:
            assert a[k] == pytest.approx(b[k]), k

    def test_soma_radius_read(self):
        import navis
        neuron = line_neuron()
        neuron.nodes.loc[0, "radius"] = 5.0
        neuron.soma = np.array([0])
        m = morph.compute_morphometrics(neuron)
        assert m["soma_radius"] == 5.0


class TestPersistenceVector:
    def test_shape_and_stability(self):
        pv = morph.compute_persistence_vector(line_neuron())
        assert pv.shape == (100,)
        assert np.allclose(pv, morph.compute_persistence_vector(line_neuron()))

    def test_distinct_geometries(self):
        pv_line = morph.compute_persistence_vector(line_neuron())
        pv_y = morph.compute_persistence_vector(y_neuron())
        assert not np.allclose(pv_line, pv_y)

    def test_translated_copy_same_vector(self):
        a = morph.compute_persistence_vector(line_neuron())
        b = morph.compute_persistence_vector(translated(line_neuron()))
        assert np.allclose(a, b)

    def test_full_vector_dimension(self):
        _, vec = morph.vectorize_neuron(line_neuron())
        assert vec.shape == (morph.VECTOR_DIM,)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_cosine_self_is_one(self):
        v = np.random.default_rng(0).normal(size=50)
        assert morph.cosine_similarity_matrix(v, v.reshape(1, -1))[0] == pytest.approx(1.0)

    def test_pairwise_cosine_matches_single_query_helper(self):
        rng = np.random.default_rng(10)
        X = rng.normal(size=(9, 17))
        expected = np.vstack([
            morph.cosine_similarity_matrix(row, X)
            for row in X
        ])
        actual = morph.pairwise_similarity_matrix(X, "cosine")
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_pearson_self_is_one(self):
        v = np.random.default_rng(1).normal(size=50)
        assert morph.pearson_similarity_matrix(v, v.reshape(1, -1))[0] == pytest.approx(1.0)

    def test_rank_corresponds_to_distance(self):
        rng = np.random.default_rng(2)
        q = rng.normal(size=30)
        near = q + rng.normal(scale=0.1, size=30)
        far = rng.normal(size=30)
        cos = morph.cosine_similarity_matrix(q, np.vstack([far, near]))
        assert cos[1] > cos[0]


# ---------------------------------------------------------------------------
# Vector cache
# ---------------------------------------------------------------------------

class TestSkeletonVectorCache:
    def _setup(self, tmp_path, dataset="test:v1", n_line=2, n_y=1):
        write_skeleton(tmp_path, dataset, 101, line_neuron())
        write_skeleton(tmp_path, dataset, 102, line_neuron(length=6))
        write_skeleton(tmp_path, dataset, 103, y_neuron())
        write_neuron_index(tmp_path, dataset, [
            (101, "LINE", "LINE_1"), (102, "LINE", "LINE_2"), (103, "Y", "Y_1"),
        ])
        return morph.SkeletonVectorCache(dataset, project_root=str(tmp_path), n_workers=2, verbose=False)

    def test_build_load_roundtrip(self, tmp_path):
        cache = self._setup(tmp_path)
        stats = cache.build()
        assert stats["rows"] == 3 and stats["new"] == 3 and stats["fetched"] == 0
        assert cache.parquet_path.exists() and cache.meta_path.exists()

        data = cache.load()
        assert len(data["bodyIds"]) == 3
        assert data["X"].shape == (3, morph.VECTOR_DIM)
        assert data["types"] == ["LINE", "LINE", "Y"]
        assert data["instances"] == ["LINE_1", "LINE_2", "Y_1"]

        # z-score standardization across the population (constant features
        # keep std 0; all others are standardized to unit variance)
        assert np.allclose(data["X"].mean(axis=0), 0.0, atol=1e-9)
        stds = data["X"].std(axis=0)
        assert np.allclose(stds[stds > 0], 1.0, atol=1e-9)

    def test_incremental_rebuild_skips_existing(self, tmp_path):
        cache = self._setup(tmp_path)
        cache.build()
        stats2 = cache.build()
        assert stats2["new"] == 0 and stats2["rows"] == 3
        # adding a skeleton only vectorizes the new one
        write_skeleton(tmp_path, "test:v1", 104, line_neuron(length=3))
        stats3 = cache.build()
        assert stats3["new"] == 1 and stats3["rows"] == 4

    def test_ensure_autobuilds(self, tmp_path):
        cache = self._setup(tmp_path)
        assert cache.cache_exists() is False
        cache.ensure()
        assert cache.cache_exists() is True

    def test_coverage(self, tmp_path):
        cache = self._setup(tmp_path)
        cov = cache.coverage()
        assert cov["skeletons"] == 3 and cov["vectors"] == 0
        cache.build()
        assert cache.coverage()["vectors"] == 3

    def test_vectors_for_missing_computes_from_skeleton(self, tmp_path):
        cache = self._setup(tmp_path)
        cache.build()
        X, mask, _ = cache.vectors_for([101, 103, 999])
        assert mask.tolist() == [True, True, False]
        assert np.isnan(X[2]).all()
        assert np.isfinite(X[0]).all()

    def test_vectors_for_without_built_cache(self, tmp_path):
        cache = self._setup(tmp_path)
        # no parquet yet: still resolves from existing skeleton files
        X, mask, _ = cache.vectors_for([101, 102])
        assert mask.all() and np.isfinite(X).all()

    def test_fetch_missing_extends_cache(self, tmp_path, monkeypatch):
        cache = self._setup(tmp_path)
        # neuron 999 exists in the index but has no skeleton yet
        write_neuron_index(tmp_path, "test:v1", [
            (101, "LINE", "LINE_1"), (102, "LINE", "LINE_2"),
            (103, "Y", "Y_1"), (999, "NEW", "NEW_1"),
        ])
        fetched = {}

        def fake_fetch(dataset, body_id, project_root=None):
            fetched[body_id] = True
            neuron = line_neuron(length=5)
            folder = (Path(project_root) / "cache"
                      / morph._dataset_folder(dataset) / "skeletons")
            folder.mkdir(parents=True, exist_ok=True)
            with open(folder / f"{body_id}.pkl", "wb") as f:
                pickle.dump(neuron, f)
            return neuron

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", fake_fetch)
        stats = cache.build(fetch_missing=2)
        assert fetched == {999: True}
        assert stats["fetched"] == 1 and stats["rows"] == 4

    def test_shared_raw_cache_isolated_from_simp90(self, tmp_path):
        """The shared raw cache never treats the visualization simp90 pickle as raw."""
        folder = tmp_path / "cache" / morph._dataset_folder("np:v1") / "skeletons"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "101.pkl", "wb") as handle:
            pickle.dump(line_neuron(length=20), handle)
        (folder / ".level").write_text("simp90\n")

        raw_cache = morph.find_similar_raw_cache(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        assert raw_cache.find_skeleton_file(101) is None
        _, mask, _ = raw_cache.vectors_for([101], compute_missing=True)
        assert mask.tolist() == [False]

        write_raw_find_similar_skeleton(tmp_path, "np:v1", 101,
                                        line_neuron(length=20))
        stats = raw_cache.build()
        assert stats["rows"] == 1
        assert raw_cache.load()["dataset_rep"] == "skeleton"

    def test_shared_raw_cache_reads_legacy_find_similar_path(self, tmp_path):
        write_raw_find_similar_skeleton(
            tmp_path, "np:v1", 101, line_neuron(length=20), legacy=True
        )
        cache = morph.find_similar_raw_cache(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        assert cache.find_skeleton_file(101).parent.name == "raw_skeletons"
        assert type(cache.load_skeleton(101)).__name__ == "TreeNeuron"
        assert (cache.skeleton_dir / "101.swc.zst").exists()

    def test_raw_batch_persists_raw_skeleton_but_vector_only_is_independent(
            self, tmp_path, monkeypatch):
        def fake_fetch(batch_ids, **kwargs):
            out = []
            for body_id in batch_ids:
                neuron = line_neuron(length=12)
                neuron.id = int(body_id)
                out.append(neuron)
            return out

        monkeypatch.setattr(
            morph, "_fetch_neuprint_batch_with_progress", fake_fetch)
        raw_cache = morph.find_similar_raw_cache(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        fetched = morph.fetch_skeletons_on_demand_batch(
            "np:v1", [101], project_root=str(tmp_path), persist=False,
            level=morph.VECTOR_BASIS_RAW, raw_cache=raw_cache,
            vector_cache=raw_cache, client=object(),
        )
        assert 101 in fetched
        assert not (raw_cache.skeleton_dir / "101.pkl").exists()
        assert 101 in set(raw_cache.load()["bodyIds"].tolist())

        fetched = morph.fetch_skeletons_on_demand_batch(
            "np:v1", [102], project_root=str(tmp_path), persist=True,
            level=morph.VECTOR_BASIS_RAW, raw_cache=raw_cache,
            vector_cache=raw_cache, client=object(),
        )
        assert 102 in fetched
        cached_path = raw_cache.find_skeleton_file(102)
        assert cached_path is not None
        assert type(raw_cache.load_skeleton(102)).__name__ == "TreeNeuron"

    def test_flywire_batch_uses_mesh_fetcher_and_caller_cache_root(
            self, tmp_path, monkeypatch):
        """FlyWire batches stay mesh-native and honor the caller's root."""
        import cave_data_fetcher as cave

        calls = {}

        class FakeCaveFetcher:
            def __init__(self, *args, **kwargs):
                calls["project_root"] = kwargs.get("project_root")

            def fetch_fafb_meshes(self, body_ids, **kwargs):
                calls["body_ids"] = body_ids
                calls["use_cache"] = kwargs["use_cache"]
                mesh = cube_mesh()
                mesh.id = body_ids[0]
                return [mesh]

        monkeypatch.setattr(cave, "CAVEDataFetcher", FakeCaveFetcher)
        result = morph.fetch_skeletons_on_demand_batch(
            "flywire_FAFB_v783", [42], project_root=str(tmp_path),
            persist=False,
        )

        import navis
        assert isinstance(result["42"], navis.MeshNeuron)
        assert calls == {
            "project_root": str(tmp_path),
            "body_ids": [42],
            "use_cache": False,
        }
        assert not (tmp_path / "cache").exists()


# ---------------------------------------------------------------------------
# On-demand fetch
# ---------------------------------------------------------------------------

class TestFetchOnDemand:
    def test_batch_fetch_uses_bounded_requests_and_one_cache_phase(
            self, tmp_path, monkeypatch):
        calls = []

        def fake_fetch(batch_ids, **kwargs):
            calls.append((list(batch_ids), kwargs.get("max_threads")))
            out = []
            for body_id in batch_ids:
                neuron = line_neuron(length=8)
                neuron.id = int(body_id)
                out.append(neuron)
            return out

        # The NeuPrint batch seam is the vendored per-neuron progress
        # fetcher; parallelism lives inside it (verified separately).
        monkeypatch.setattr(
            morph, "_fetch_neuprint_batch_with_progress", fake_fetch)
        result = morph.fetch_skeletons_on_demand_batch(
            "test:v1", [101, 102, 101, 103],
            project_root=str(tmp_path), persist=False,
            batch_size=2, max_threads=3,
            client=object(),
        )

        assert list(result) == [101, 102, 103]
        assert [ids for ids, _ in calls] == [[101, 102], [103]]
        assert [threads for _, threads in calls] == [2, 1]
        assert not list((tmp_path / "cache" / "test_v1" / "skeletons")
                        .glob("*.pkl"))

    def test_persist_worker_runs_concurrently_with_fetch(
            self, tmp_path, monkeypatch):
        """Simplification + cache writing run on a standalone thread: the
        fetch loop submits the next batch without waiting for the previous
        batch's persist to finish (benchmark: fetch ~0.53s/skeleton vs
        simplify+write ~0.04s and vectorize ~0.11s, so the CPU tail must
        overlap the network phase instead of blocking it)."""
        import threading

        persist_started = threading.Event()
        release_persist = threading.Event()
        fetch_calls = {"n": 0}

        def fake_fetch(batch_ids, **kwargs):
            fetch_calls["n"] += 1
            if fetch_calls["n"] == 2:
                # Batch 2 must be fetched while batch 1's persist is still
                # running: the persist runs on the standalone worker thread.
                assert persist_started.wait(5), (
                    "persist did not start while batch 2 was fetching")
                release_persist.set()
            out = []
            for body_id in batch_ids:
                neuron = line_neuron(length=8)
                neuron.id = int(body_id)
                out.append(neuron)
            return out

        monkeypatch.setattr(
            morph, "_fetch_neuprint_batch_with_progress", fake_fetch)
        raw_cache = morph.find_similar_raw_cache(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        real_persist = raw_cache.persist_skeletons

        def slow_persist(neurons, **kwargs):
            persist_started.set()
            release_persist.wait(5)
            return real_persist(neurons, **kwargs)

        raw_cache.persist_skeletons = slow_persist

        result = morph.fetch_skeletons_on_demand_batch(
            "np:v1", [101, 102, 103, 104], project_root=str(tmp_path),
            persist=True, batch_size=2, max_threads=2,
            raw_cache=raw_cache, vector_cache=raw_cache, client=object(),
        )
        assert list(result) == [101, 102, 103, 104]
        # the worker joined before returning: every fetched skeleton is on
        # disk (simplified .swc.zst) and the vector cache has all rows
        assert fetch_calls["n"] == 2
        assert all(
            raw_cache.find_skeleton_file(bid) is not None for bid in (101, 102, 103, 104)
        )
        assert set(raw_cache.load()["bodyIds"].tolist()) == {101, 102, 103, 104}

    def test_pipeline_stages_temp_skeletons_and_cleans_after_persist(
            self, tmp_path, monkeypatch):
        """Fetched raw skeletons land in ``skeletons/_temp_cache`` (crash-
        resume) and the staging entries are removed once the simplified
        files are persisted."""
        raw_cache = morph.find_similar_raw_cache(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        temp_dir = raw_cache.temp_cache_dir()

        def fake_fetch(batch_ids, **kwargs):
            out = []
            for body_id in batch_ids:
                neuron = line_neuron(length=8)
                neuron.id = int(body_id)
                out.append(neuron)
            return out

        monkeypatch.setattr(
            morph, "_fetch_neuprint_batch_with_progress", fake_fetch)
        result = morph.fetch_skeletons_on_demand_batch(
            "np:v1", [101, 102], project_root=str(tmp_path), persist=True,
            batch_size=1, max_threads=2,
            raw_cache=raw_cache, vector_cache=raw_cache, client=object(),
        )
        assert list(result) == [101, 102]
        # staging cleaned after persist; final simplified files exist
        assert not temp_dir.exists() or not list(temp_dir.glob("*.swc.zst"))
        assert raw_cache.find_skeleton_file(101) is not None
        assert raw_cache.find_skeleton_file(102) is not None

    def test_crash_resume_reprocesses_temp_without_refetch(
            self, tmp_path, monkeypatch):
        """A leftover _temp_cache entry from a crashed run is loaded from
        disk (never re-fetched), vectorized, persisted, and cleaned up."""
        raw_cache = morph.find_similar_raw_cache(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        neuron = line_neuron(length=10)
        neuron.id = 101
        raw_cache.write_temp_skeleton(101, neuron)

        fetch_calls = {"n": 0}

        def fake_fetch(batch_ids, **kwargs):
            fetch_calls["n"] += 1
            return []

        monkeypatch.setattr(
            morph, "_fetch_neuprint_batch_with_progress", fake_fetch)
        result = morph.fetch_skeletons_on_demand_batch(
            "np:v1", [101], project_root=str(tmp_path), persist=True,
            raw_cache=raw_cache, vector_cache=raw_cache, client=object(),
        )
        assert list(result) == [101]
        assert fetch_calls["n"] == 0  # served from temp, no network fetch
        assert raw_cache.find_skeleton_file(101) is not None  # final persisted
        assert not (raw_cache.temp_cache_dir() / "101.swc.zst").exists()
        assert 101 in set(raw_cache.load()["bodyIds"].tolist())

    def test_crash_duplicate_vector_rows_dedupe_at_read_and_merge(
            self, tmp_path, monkeypatch):
        """Crash between 'vector appended' and 'skeleton persisted' (the
        accepted duplicate window): the resume pass re-appends the same
        neuron's vector; first-wins dedupe keeps exactly one row - at
        load() AND after the merge checkpoint."""
        raw_cache = morph.find_similar_raw_cache(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        neuron = line_neuron(length=12)
        neuron.id = 101
        # crash state: vector row already appended, temp file present,
        # final skeleton absent
        _, vec = morph.vectorize_neuron(neuron)
        raw_cache.append_vectors([(101, vec, "skeleton")])
        raw_cache.write_temp_skeleton(101, neuron)

        monkeypatch.setattr(
            morph, "_fetch_neuprint_batch_with_progress",
            lambda *a, **k: [],
        )
        result = morph.fetch_skeletons_on_demand_batch(
            "np:v1", [101], project_root=str(tmp_path), persist=True,
            raw_cache=raw_cache, vector_cache=raw_cache, client=object(),
        )
        assert list(result) == [101]
        data = raw_cache.load()
        assert data is not None
        assert sum(1 for b in data["bodyIds"].tolist() if b == 101) == 1
        # after the merge checkpoint the main file is deduped too
        raw_cache._merge_pending()
        data2 = raw_cache.load()
        assert sum(1 for b in data2["bodyIds"].tolist() if b == 101) == 1
        assert not raw_cache.pending_path.exists()

    def test_cancel_drains_workers_and_cleans_temp(
            self, tmp_path, monkeypatch):
        """Cancel stops new batches; the fetched batch is still staged,
        persisted, vectorized, and its temp entries removed (resume-safe)."""
        import threading

        raw_cache = morph.find_similar_raw_cache(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        cancel = threading.Event()

        def fake_fetch(batch_ids, **kwargs):
            out = []
            for body_id in batch_ids:
                neuron = line_neuron(length=8)
                neuron.id = int(body_id)
                out.append(neuron)
            cancel.set()  # cancel after the first batch
            return out

        monkeypatch.setattr(
            morph, "_fetch_neuprint_batch_with_progress", fake_fetch)
        result = morph.fetch_skeletons_on_demand_batch(
            "np:v1", [101, 102, 103], project_root=str(tmp_path),
            persist=True, batch_size=2, max_threads=2, cancel_event=cancel,
            raw_cache=raw_cache, vector_cache=raw_cache, client=object(),
        )
        assert list(result) == [101, 102]  # batch 2 never submitted
        assert raw_cache.find_skeleton_file(101) is not None
        assert raw_cache.find_skeleton_file(102) is not None
        assert not (raw_cache.temp_cache_dir() / "101.swc.zst").exists()
        assert not (raw_cache.temp_cache_dir() / "102.swc.zst").exists()

    def test_vendored_batch_reports_per_neuron_progress_and_parallelism(
            self, tmp_path, monkeypatch):
        """The vendored NeuPrint batch loop fetches with ``max_threads``
        concurrent SWC requests (not serially) and calls ``on_neuron`` for
        every completed skeleton - the per-neuron cadence that drives the
        Settings pull UI."""
        import threading
        import time
        import navis.interfaces.neuprint as neu

        monkeypatch.setattr(neu, "NeuronCriteria", lambda **kwargs: None)
        meta = pd.DataFrame({
            "bodyId": list(range(1, 13)),
            "instance": [None] * 12,
            "size": [1000] * 12,
            "status": ["Traced"] * 12,
            "somaLocation": [None] * 12,
            "somaRadius": [None] * 12,
        })
        monkeypatch.setattr(
            neu, "fetch_neurons",
            lambda *a, **k: (meta.copy(), pd.DataFrame()),
        )

        class FakeClient:
            meta = {"voxelSize": [8, 8, 8], "voxelUnits": "nm"}

            def __init__(self):
                self.lock = threading.Lock()
                self.active = 0
                self.max_active = 0

            def fetch_skeleton(self, bodyid, format="pandas", heal=False):
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.02)
                with self.lock:
                    self.active -= 1
                return pd.DataFrame({
                    "node_id": [1, 2, 3], "parent_id": [-1, 1, 2],
                    "x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0],
                    "z": [0.0, 0.0, 0.0], "radius": [1.0, 1.0, 1.0],
                })

        client = FakeClient()
        per_neuron = []
        started = time.perf_counter()
        neurons = morph._fetch_neuprint_batch_with_progress(
            list(range(1, 13)), client=client, max_threads=6,
            on_neuron=lambda done, total: per_neuron.append((done, total)),
        )
        elapsed = time.perf_counter() - started

        assert len(neurons) == 12
        # One callback per completed skeleton, in completion order.
        assert len(per_neuron) == 12
        assert per_neuron[-1] == (12, 12)
        # All six workers were in flight at once: the batch is parallel.
        assert client.max_active == 6, client.max_active
        # Parallel wall time << serial (12 x 0.02 s = 0.24 s).
        assert elapsed < 0.24 * 0.9, elapsed

    def test_legacy_simp90_request_writes_and_reads_raw_swc(
            self, tmp_path, monkeypatch):
        """The compatibility level does not change the raw cache contract."""
        calls = []

        def fake_fetch(batch_ids, **kwargs):
            calls.append(list(batch_ids))
            out = []
            for body_id in batch_ids:
                neuron = line_neuron(length=12)
                neuron.id = int(body_id)
                out.append(neuron)
            return out

        monkeypatch.setattr(
            morph, "_fetch_neuprint_batch_with_progress", fake_fetch)
        raw_cache = morph.find_similar_raw_cache(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        first = morph.fetch_skeletons_on_demand_batch(
            "np:v1", [101], project_root=str(tmp_path), persist=True,
            level=morph.VECTOR_BASIS_SIMP90, raw_cache=raw_cache,
            vector_cache=raw_cache, client=object(),
        )
        assert 101 in first
        skeleton_dir = (tmp_path / "cache" / morph._dataset_folder("np:v1")
                        / "skeletons")
        assert (skeleton_dir / "raw_skeletons" / "101.swc.zst").exists()
        assert not (skeleton_dir / "101.pkl").exists()
        assert not (skeleton_dir / ".level").exists()

        monkeypatch.setattr(
            morph, "_fetch_neuprint_batch_with_progress",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("raw SWC cache was not reused")
            ),
        )
        second = morph.fetch_skeletons_on_demand_batch(
            "np:v1", [101], project_root=str(tmp_path), persist=True,
            level=morph.VECTOR_BASIS_SIMP90, raw_cache=raw_cache,
            vector_cache=raw_cache, client=object(),
        )
        assert 101 in second
        assert calls == [[101]]

    def test_neuprint_fetch_persists_raw_swc_and_reuses_it(self, tmp_path, monkeypatch):
        """Raw requests persist portable SWC and reuse that shared cache."""
        neuron = line_neuron()
        monkeypatch.setattr(morph, "_fetch_neuprint_skeleton",
                            lambda dataset, bid: neuron)
        pkl = write_skeleton(tmp_path, "test:v1", 101, None)  # placeholder path
        pkl.unlink()  # ensure missing

        nrn_raw = morph.fetch_skeleton_on_demand(
            "test:v1", 101, project_root=str(tmp_path), level="raw"
        )
        assert nrn_raw is not None
        raw_path = (tmp_path / "cache" / morph._dataset_folder("test:v1")
                    / "skeletons" / "raw_skeletons" / "101.swc.zst")
        assert raw_path.exists()

        # A RAW request reuses the shared compressed cache.
        calls = {"n": 0}

        def counting_fetch(dataset, bid):
            calls["n"] += 1
            return neuron

        monkeypatch.setattr(morph, "_fetch_neuprint_skeleton", counting_fetch)
        again_raw = morph.fetch_skeleton_on_demand(
            "test:v1", 101, project_root=str(tmp_path), level="raw"
        )
        assert calls["n"] == 0
        assert again_raw is not None

    def test_persist_false_does_not_write_cache(self, tmp_path, monkeypatch):
        """Transient fetches (persist=False) must not create skeleton files."""
        monkeypatch.setattr(morph, "_fetch_neuprint_skeleton",
                            lambda d, b: line_neuron())
        skel_dir = tmp_path / "cache" / morph._dataset_folder("np:v1") / "skeletons"
        skel_dir.mkdir(parents=True)
        neuron = morph.fetch_skeleton_on_demand(
            "np:v1", 42, project_root=str(tmp_path), persist=False
        )
        assert neuron is not None
        assert not list(skel_dir.glob("*.pkl"))
        # persist=True (default) writes the raw SWC file for reuse
        neuron2 = morph.fetch_skeleton_on_demand(
            "np:v1", 43, project_root=str(tmp_path)
        )
        assert neuron2 is not None
        assert (skel_dir / "raw_skeletons" / "43.swc.zst").exists()

    def test_cave_fetch_used_for_flywire(self, monkeypatch, tmp_path):
        used = {}

        def fake_cave(dataset, bid, **kwargs):
            used["dataset"] = dataset
            used["kwargs"] = kwargs
            return cube_mesh()

        monkeypatch.setattr(morph, "_fetch_cave_mesh", fake_cave)
        monkeypatch.setattr(morph, "_fetch_neuprint_skeleton",
                            lambda d, b: (_ for _ in ()).throw(AssertionError("neuprint used")))
        nrn = morph.fetch_skeleton_on_demand(
            "flywire_FAFB_v783", 42, project_root=str(tmp_path)
        )
        assert used["dataset"] == "flywire_FAFB_v783"
        assert used["kwargs"]["use_cache"] is True
        import navis
        assert isinstance(nrn, navis.MeshNeuron)


# ---------------------------------------------------------------------------
# MorphologyComparer
# ---------------------------------------------------------------------------

class TestMorphologyComparer:
    def _setup(self, tmp_path, monkeypatch, dataset="test:v1"):
        write_skeleton(tmp_path, dataset, 101, line_neuron(length=20))
        write_skeleton(tmp_path, dataset, 102, translated(line_neuron(length=20)))
        write_skeleton(tmp_path, dataset, 103, bushy_y_neuron())
        write_raw_find_similar_skeleton(tmp_path, dataset, 101, line_neuron(length=20))
        write_raw_find_similar_skeleton(tmp_path, dataset, 102, translated(line_neuron(length=20)))
        write_raw_find_similar_skeleton(tmp_path, dataset, 103, bushy_y_neuron())
        write_neuron_index(tmp_path, dataset, [
            (101, "LINE", "LINE_1"), (102, "LINE", "LINE_2"), (103, "Y", "Y_1"),
        ])

        def fake_getNeurons(required, dataset_name, **kwargs):
            rows = {
                101: ("LINE", "LINE_1"),
                102: ("LINE", "LINE_2"),
                103: ("Y", "Y_1"),
            }
            if isinstance(required, (list, tuple)):
                b = [int(r) for r in required]
            else:
                b = [int(required)]
            df = pd.DataFrame({
                "bodyId": b,
                "type": [rows[x][0] for x in b],
                "instance": [rows[x][1] for x in b],
            })
            # real getNeurons returns (neuron_df, roi_count_df, auto_name, criteria)
            return df, pd.DataFrame(), "auto", None

        monkeypatch.setattr(morph, "getNeurons", fake_getNeurons)
        return tmp_path

    def _make_comparer(self, root, **kwargs):
        params = dict(
            dataset="test:v1", level="bodyid", method="vector_v2", expand_top_types=0, metric="cosine",
            output_dir=str(root / "out"),
            project_root=str(root), verbose=False, candidate_source="cache",
        )
        params.update(kwargs)
        return morph.MorphologyComparer(**params)

    def test_vector_search_ranks_translated_copy_first(self, tmp_path, monkeypatch):
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(root, query=101)
        res = comparer.find_similar()
        assert not res.empty
        assert res.iloc[0]["target_bodyId"] == 102  # translated copy = identical morphology
        assert 101 not in res["target_bodyId"].tolist()  # self excluded
        assert res.iloc[0]["similarity"] >= res.iloc[1]["similarity"]

    def test_vector_search_type_level_includes_intra_reference(self, tmp_path, monkeypatch):
        """Type queries must include the query type itself as the intra-type
        reference row (rank 1, is_intra_type=True) plus its intra-type
        similarity value."""
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(root, query=101, level="type")
        res = comparer.find_similar()
        assert not res.empty
        # The query type (LINE) is present as the intra-type reference row.
        intra = res[res["is_intra_type"]]
        assert len(intra) == 1
        assert intra.iloc[0]["target_type"] == "LINE"
        assert intra.iloc[0]["rank"] == 1
        # 2 LINE members, one translated: V2 spatial block makes the intra
        # reference position-sensitive (not exactly 1.0).
        assert intra.iloc[0]["intra_type_similarity"] > 0.3
        # Inter-type rows exclude the query type and rank after it.
        inter = res[~res["is_intra_type"]]
        assert inter.iloc[0]["target_type"] == "Y"
        assert inter.iloc[0]["rank"] == 2
        assert "LINE" not in inter["target_type"].tolist()

    def test_vector_search_bodyid_carries_intra_columns(self, tmp_path, monkeypatch):
        """bodyId queries must tag same-type rows (is_same_type) and report
        the query type's intra-type similarity."""
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(root, query=101)
        res = comparer.find_similar()
        assert {"is_same_type", "intra_type_similarity"}.issubset(res.columns)
        # 102 is the same type (LINE) and morphologically identical -> top hit
        assert res.iloc[0]["target_bodyId"] == 102
        assert res.iloc[0]["is_same_type"] == True
        # LINE has 2 members, one translated: under the V2 spatial block the
        # intra reference is position-sensitive (lateral normalization and
        # the overlap term both apply), never exactly 1.0.
        assert res.iloc[0]["intra_type_similarity"] > 0.3
        # the Y neuron is a different type
        row_y = res[res["target_bodyId"] == 103]
        assert row_y.iloc[0]["is_same_type"] == False

    def test_intra_type_similarity_values(self, tmp_path, monkeypatch):
        root = self._setup(tmp_path, monkeypatch)
        cache = morph.SkeletonVectorCache("test:v1", project_root=str(root), verbose=False)
        cache.build()
        data = cache.load()
        comparer = self._make_comparer(root, query=101)
        # two identical LINE members -> 1.0; single-member type -> 1.0;
        # unknown/empty types -> NaN
        assert comparer._intra_type_similarity(
            "LINE", data["bodyIds"], data["types"], data["X"]
        ) == pytest.approx(1.0, abs=1e-6)
        assert comparer._intra_type_similarity(
            "Y", data["bodyIds"], data["types"], data["X"]
        ) == 1.0
        assert np.isnan(comparer._intra_type_similarity(
            "ZZZ", data["bodyIds"], data["types"], data["X"]))
        assert np.isnan(comparer._intra_type_similarity(
            "", data["bodyIds"], data["types"], data["X"]))

    def test_nblast_type_level_includes_intra_reference(self, tmp_path, monkeypatch):
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(
            root, query=101, level="type", method="nblast",
            candidate_cap=10, n_workers=1,
        )
        res = comparer.find_similar()
        assert not res.empty
        intra = res[res["is_intra_type"]]
        assert len(intra) == 1
        assert intra.iloc[0]["target_type"] == "LINE"
        assert intra.iloc[0]["rank"] == 1
        # identical members -> both the NBLAST mean and the vector intra ~1
        assert intra.iloc[0]["similarity"] > 0.9
        assert intra.iloc[0]["intra_type_similarity"] == pytest.approx(1.0, abs=1e-6)
        # inter-type rows rank after the intra reference
        inter = res[~res["is_intra_type"]]
        assert inter.iloc[0]["target_type"] == "Y"
        assert inter.iloc[0]["rank"] == 2

    def test_nblast_method_and_outputs(self, tmp_path, monkeypatch):
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(
            root, query=101, method="nblast", candidate_cap=10, n_workers=1,
        )
        res = comparer.find_similar()
        assert not res.empty
        assert res.iloc[0]["target_bodyId"] == 102  # translated copy scores highest
        assert res.iloc[0]["similarity"] >= res.iloc[1]["similarity"]
        # dotprops must never be persisted to disk
        assert not list((root / "cache" / "test_v1").rglob("*.dp")) 
        # results saved with the similar-morphology_ prefix
        run_dirs = [p for p in (root / "out").iterdir() if p.is_dir()]
        assert len(run_dirs) == 1
        assert run_dirs[0].name.startswith("similar-morphology_")
        assert (run_dirs[0] / "results.csv").exists()
        assert (run_dirs[0] / "README.txt").exists()

    def test_invalid_parameters(self, tmp_path):
        with pytest.raises(ValueError):
            morph.MorphologyComparer(level="bad", project_root=str(tmp_path), verbose=False)
        with pytest.raises(ValueError):
            morph.MorphologyComparer(method="bad", project_root=str(tmp_path), verbose=False)
        with pytest.raises(ValueError):
            morph.MorphologyComparer(metric="bad", project_root=str(tmp_path), verbose=False)
        with pytest.raises(ValueError, match="visualize_by"):
            morph.MorphologyComparer(query=1, visualize_by="bad",
                                     project_root=str(tmp_path), verbose=False)


    def test_auto_level_resolution(self, tmp_path, monkeypatch):
        """level='auto': type queries -> type-to-type, bodyId queries ->
        bodyId-to-bodyId, mixed/multi-type lists -> bodyId."""
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(root, query=101, level="auto")

        comparer.query = "LINE"
        qdf = pd.DataFrame({"bodyId": [101, 102], "type": ["LINE", "LINE"],
                            "instance": ["L1", "L2"]})
        assert comparer._resolve_level(qdf) == "type"

        comparer.query = 101
        qdf1 = pd.DataFrame({"bodyId": [101], "type": ["LINE"], "instance": ["L1"]})
        assert comparer._resolve_level(qdf1) == "bodyid"

        comparer.query = [101, 102]
        assert comparer._resolve_level(qdf) == "bodyid"  # all-numeric list

        comparer.query = "SMP"  # single-member type still a type query
        qdf2 = pd.DataFrame({"bodyId": [999], "type": ["SMP"], "instance": ["S"]})
        assert comparer._resolve_level(qdf2) == "type"

        comparer.query = ["LINE", "Y"]  # multi-type list -> bodyId rows
        qdf3 = pd.DataFrame({"bodyId": [101, 103], "type": ["LINE", "Y"],
                             "instance": ["L1", "Y1"]})
        assert comparer._resolve_level(qdf3) == "bodyid"

    def test_auto_level_end_to_end(self, tmp_path, monkeypatch):
        root = self._setup(tmp_path, monkeypatch)

        def fake_getNeurons(required, dataset_name, **kwargs):
            rows = {101: ("LINE", "L1"), 102: ("LINE", "L2"), 103: ("Y", "Y1")}
            if isinstance(required, str) and not required.isdigit():
                b = [bid for bid, (t, _) in rows.items() if t == required]
            else:
                b = [int(r) for r in (required if isinstance(required, (list, tuple))
                                      else [required])]
            df = pd.DataFrame({"bodyId": b, "type": [rows[x][0] for x in b],
                               "instance": [rows[x][1] for x in b]})
            return df, pd.DataFrame(), "auto", None

        monkeypatch.setattr(morph, "getNeurons", fake_getNeurons)
        # type query -> type-to-type rows with the intra reference
        comparer = self._make_comparer(root, query="LINE", level="auto")
        res = comparer.find_similar()
        assert "is_intra_type" in res.columns
        assert res.iloc[0]["target_type"] == "LINE"
        # bodyId query -> bodyId-to-bodyId rows
        comparer2 = self._make_comparer(root, query=101, level="auto")
        res2 = comparer2.find_similar()
        assert "target_bodyId" in res2.columns
        assert "is_same_type" in res2.columns

    def test_type_summary_written_for_bodyid_level(self, tmp_path, monkeypatch):
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(root, query=101, level="bodyid")
        comparer.find_similar()
        run_dir = Path(comparer.output_folder)
        # results.csv is bodyId-level; type_summary.csv the type rows
        res = pd.read_csv(run_dir / "results.csv")
        assert "target_bodyId" in res.columns
        summary = pd.read_csv(run_dir / "type_summary.csv")
        assert {"target_type", "similarity", "n_bodyids", "is_intra_type",
                "intra_type_similarity"}.issubset(summary.columns)
        # the query type (LINE, both members) is the intra reference first
        assert summary.iloc[0]["target_type"] == "LINE"
        assert summary.iloc[0]["is_intra_type"] == True  # noqa: E712
        assert summary.iloc[0]["n_bodyids"] == 2
        assert summary.iloc[1]["target_type"] == "Y"
        assert summary.iloc[1]["is_intra_type"] == False  # noqa: E712

    def test_type_summary_written_for_type_level(self, tmp_path, monkeypatch):
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(root, query=101, level="type")
        comparer.find_similar()
        run_dir = Path(comparer.output_folder)
        summary = pd.read_csv(run_dir / "type_summary.csv")
        assert {"target_type", "similarity", "n_bodyids", "is_intra_type"}.issubset(summary.columns)
        assert summary.iloc[0]["target_type"] == "LINE"
        assert summary.iloc[0]["is_intra_type"] == True  # noqa: E712

    def test_type_search_writes_bodyid_results_csv(self, tmp_path, monkeypatch):
        """results.csv is ALWAYS bodyId-level: a type query's results.csv
        holds ranked bodyId rows, including ordered intra-type pairs when
        multiple query members are resolved, while type_summary.csv holds
        the type rows."""
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(root, query=101, level="type")
        comparer.find_similar()
        run_dir = Path(comparer.output_folder)
        res = pd.read_csv(run_dir / "results.csv")
        assert "target_bodyId" in res.columns
        # This scalar fixture resolves one member, so there is no intra pair.
        assert 101 not in res["target_bodyId"].tolist()
        assert res["target_type"].tolist() == ["LINE", "Y"]
        summary = pd.read_csv(run_dir / "type_summary.csv")
        assert "target_type" in summary.columns
        assert summary.iloc[0]["is_intra_type"] == True  # noqa: E712
        assert summary.iloc[0]["target_type"] == "LINE"


class TestVisualizeTopResults:
    """3D skeleton visualization of the top-N found types (NB-style)."""

    class FakeVisualizer:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            # nested class: its own name is not in method scope
            type(self).instances.append(self)

        def plot_neurons(self):
            pass

    def _setup(self, tmp_path, monkeypatch, **kwargs):
        root = TestMorphologyComparer()._setup(tmp_path, monkeypatch)
        params = dict(query=101, dataset="test:v1",
                      output_dir=str(root / "out"), project_root=str(root),
                      verbose=False, candidate_source="cache",
                      visualize_top_n=3, visualize_by="type")
        params.update(kwargs)
        self.FakeVisualizer.instances = []
        monkeypatch.setattr(morph, "_import_visualizer",
                            lambda: self.FakeVisualizer)
        return morph.MorphologyComparer(**params)

    def test_disabled_by_default(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch, visualize_top_n=0)
        res = comparer.find_similar()
        assert not res.empty
        assert self.FakeVisualizer.instances == []

    def test_type_mode_groups_result_members(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch)
        res = comparer.find_similar()
        assert not res.empty
        assert len(self.FakeVisualizer.instances) == 1
        vs = self.FakeVisualizer.instances[0]
        # The query is rendered as the first (reference) layer; the requested
        # top-N count applies only to result types.
        assert vs.kwargs["neuron_layers"] == [[101], [102], [103]]
        assert vs.kwargs["custom_layer_names"] == [
            "query_101_x1", "r1_LINE_x1", "r2_Y_x1"
        ]
        assert vs.kwargs["legend_mode"] == "layer"
        assert vs.kwargs["skip_synapse"] is True
        assert vs.kwargs["show_fig"] is False
        assert vs.kwargs["saveas"] == morph._dataset_folder("test:v1")
        assert vs.kwargs["skeleton_mesh_simplification"] == 0.95

    def test_type_level_excludes_intra_reference_row(self, tmp_path, monkeypatch):
        """The intra-type reference row (rank 1) must never be rendered."""
        comparer = self._setup(tmp_path, monkeypatch, query=101, level="type")
        res = comparer.find_similar()
        assert not res.empty
        assert len(self.FakeVisualizer.instances) == 1
        vs = self.FakeVisualizer.instances[0]
        # The query is the first layer; the LINE intra reference is excluded,
        # so only the Y row remains, with its members from the vector cache.
        assert vs.kwargs["neuron_layers"] == [[101], [103]]
        assert vs.kwargs["custom_layer_names"] == ["query_101_x1", "r1_Y_x1"]

    def test_bodyid_mode_one_layer_per_row(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch, visualize_by="bodyId")
        comparer.find_similar()
        assert len(self.FakeVisualizer.instances) == 1
        vs = self.FakeVisualizer.instances[0]
        assert vs.kwargs["neuron_layers"] == [[101], [102], [103]]
        assert vs.kwargs["custom_layer_names"] == [
            "query_101_x1", "r1_LINE_102", "r2_Y_103"
        ]
        assert vs.kwargs["legend_mode"] == "single"

    def test_query_rows_do_not_consume_bodyid_top_n(self, tmp_path, monkeypatch):
        comparer = self._setup(
            tmp_path, monkeypatch, visualize_by="bodyId", visualize_top_n=1
        )
        results = pd.DataFrame({
            "target_bodyId": [101, 102, 103],
            "target_type": ["LINE", "LINE", "Y"],
            "similarity": [1.0, 0.9, 0.8],
        })
        query_df = pd.DataFrame({"bodyId": [101], "type": ["LINE"]})

        comparer._visualize_top_results(results, query_df=query_df)

        vs = self.FakeVisualizer.instances[0]
        # The query reference layer never consumes the top-N budget: the
        # query row (101) is excluded from the results, leaving one layer.
        assert vs.kwargs["neuron_layers"] == [[101], [102]]
        assert vs.kwargs["custom_layer_names"] == ["query_101_x1", "r1_LINE_102"]

    def test_visualization_settings_are_forwarded(self, tmp_path, monkeypatch):
        comparer = self._setup(
            tmp_path,
            monkeypatch,
            visualization_settings={
                "brain_mesh": "none",
                "skeleton_mode": "line",
                "background_color": "black",
                "show_fig": True,
                "export_views": True,
            },
        )
        comparer.find_similar()
        vs = self.FakeVisualizer.instances[0]
        assert vs.kwargs["brain_mesh"] == "none"
        assert vs.kwargs["skeleton_mode"] == "line"
        assert vs.kwargs["background_color"] == "black"
        assert vs.kwargs["show_fig"] is True
        assert vs.kwargs["export_views"] is True

    def test_failure_never_breaks_search(self, tmp_path, monkeypatch):
        def boom(self):
            raise RuntimeError("render failed")
        self.FakeVisualizer.plot_neurons = boom
        comparer = self._setup(tmp_path, monkeypatch)
        res = comparer.find_similar()  # must not raise
        assert not res.empty

    def test_visualizer_unavailable_skips(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch)
        monkeypatch.setattr(morph, "_import_visualizer", lambda: None)
        res = comparer.find_similar()
        assert not res.empty
        assert self.FakeVisualizer.instances == []


# ---------------------------------------------------------------------------
# Mesh-based features (FlyWire bulk caches)
# ---------------------------------------------------------------------------

def cube_mesh(size=1.0):
    """A unit cube as a navis MeshNeuron (12 triangles)."""
    import navis
    v = np.array([
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
        (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
    ], dtype=float) * size
    faces = np.array([
        [0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6],
        [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],
        [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5],
    ], dtype=np.int64)
    return navis.MeshNeuron({"vertices": v, "faces": faces})


def translated_mesh(mesh, shift=(100.0, 50.0, -30.0)):
    vertices = np.asarray(mesh.vertices, dtype=float) + np.array(shift)
    return type(mesh)({"vertices": vertices, "faces": np.asarray(mesh.faces)})


class TestMeshFeatures:
    def test_cube_morphometrics(self):
        m = morph.compute_mesh_morphometrics(cube_mesh())
        assert m["n_nodes"] == 8.0
        assert m["n_branch"] == 12.0
        assert m["bbox_x"] == pytest.approx(1.0)
        assert m["bbox_diagonal"] == pytest.approx(np.sqrt(3))
        assert m["leaf_density"] == pytest.approx(6.0 / 3.0)  # area / bbox_diag^2
        assert m["cable_length"] > 0

    def test_histogram_shape_and_normalization(self):
        hist = morph.compute_spatial_histogram(cube_mesh())
        assert hist.shape == (100,)
        assert hist.sum() == pytest.approx(1.0)

    def test_mesh_vector_schema_and_translation_invariance(self):
        v1 = morph.vectorize_neuron(cube_mesh())[1]
        v2 = morph.vectorize_neuron(translated_mesh(cube_mesh()))[1]
        assert v1.shape == (morph.VECTOR_DIM,)
        assert np.isfinite(v1).all()
        assert np.allclose(v1, v2)  # bbox-normalized histogram is translation-invariant

    def test_mesh_and_skeleton_share_schema(self):
        v_skel = morph.vectorize_neuron(line_neuron())[1]
        v_mesh = morph.vectorize_neuron(cube_mesh())[1]
        assert v_skel.shape == v_mesh.shape == (morph.VECTOR_DIM,)

    def test_unsupported_neuron_raises(self):
        class Bad:
            pass
        with pytest.raises(ValueError, match="Unsupported neuron type"):
            morph.vectorize_neuron(Bad())


# ---------------------------------------------------------------------------
# Nested skeleton folders (FlyWire bulk caches)
# ---------------------------------------------------------------------------

class TestNestedDiscovery:
    def test_build_and_coverage_find_nested_files(self, tmp_path):
        dataset = "fw:v1"
        folder = (tmp_path / "cache" / morph._dataset_folder(dataset)
                  / "skeletons" / "bulk_v1")
        folder.mkdir(parents=True)
        for bid, nrn in ((301, line_neuron()), (302, y_neuron())):
            with open(folder / f"{bid}.pkl", "wb") as f:
                pickle.dump(nrn, f)
        cache = morph.SkeletonVectorCache(dataset, project_root=str(tmp_path), verbose=False)
        assert cache.coverage()["skeletons"] == 2
        stats = cache.build()
        assert stats["rows"] == 2
        X, mask, _ = cache.vectors_for([301, 302])
        assert mask.all() and np.isfinite(X).all()

    def test_vectors_for_resolves_nested(self, tmp_path):
        dataset = "fw:v1"
        folder = (tmp_path / "cache" / morph._dataset_folder(dataset)
                  / "skeletons" / "bulk_v1")
        folder.mkdir(parents=True)
        with open(folder / "301.pkl", "wb") as f:
            pickle.dump(line_neuron(), f)
        cache = morph.SkeletonVectorCache(dataset, project_root=str(tmp_path), verbose=False)
        X, mask, _ = cache.vectors_for([301, 999])
        assert mask.tolist() == [True, False]

    def test_fetch_reuses_nested_raw_swc_file(self, tmp_path, monkeypatch):
        """A compatibility-level request reuses nested raw SWC."""
        dataset = "fw:v1"
        folder = (tmp_path / "cache" / morph._dataset_folder(dataset)
                  / "skeletons" / "raw_skeletons" / "bulk_v1")
        folder.mkdir(parents=True)
        morph._write_compressed_swc(folder / "301.swc.gz", line_neuron())
        monkeypatch.setattr(morph, "_fetch_neuprint_skeleton",
                            lambda d, b: (_ for _ in ()).throw(AssertionError("fetcher used")))
        nrn = morph.fetch_skeleton_on_demand(dataset, 301, project_root=str(tmp_path),
                                             level="simp90")
        assert nrn is not None


# ---------------------------------------------------------------------------
# Vectorization-level guards (vector_basis / .level marker)
# ---------------------------------------------------------------------------

class TestLevelGuards:
    def test_folder_level_defaults_to_raw(self, tmp_path):
        dataset = "test:v1"
        write_skeleton(tmp_path, dataset, 101, line_neuron())
        assert morph._skeleton_folder_level(dataset, str(tmp_path)) == "raw"

    def test_folder_level_reads_marker(self, tmp_path):
        dataset = "test:v1"
        folder = tmp_path / "cache" / morph._dataset_folder(dataset) / "skeletons"
        folder.mkdir(parents=True)
        (folder / ".level").write_text("simp90\n")
        assert morph._skeleton_folder_level(dataset, str(tmp_path)) == "simp90"
        # unknown values are treated as raw (never guessed)
        (folder / ".level").write_text("weird\n")
        assert morph._skeleton_folder_level(dataset, str(tmp_path)) == "raw"

    def test_marker_is_idempotent(self, tmp_path):
        dataset = "test:v1"
        morph._write_skeleton_level_marker(dataset, str(tmp_path))
        morph._write_skeleton_level_marker(dataset, str(tmp_path))
        marker = (tmp_path / "cache" / morph._dataset_folder(dataset)
                  / "skeletons" / ".level")
        assert marker.read_text().strip() == "simp90"

    def test_downsample_for_cache_reduces_nodes(self):
        import navis
        pts = [(i * 1.0, 0.0, 0.0) for i in range(200)]
        nrn = make_neuron(pts, [-1] + list(range(199)))
        out = morph._downsample_for_cache(nrn)
        assert isinstance(out, navis.TreeNeuron)
        assert len(out.nodes) <= len(nrn.nodes)
        # deterministic
        out2 = morph._downsample_for_cache(nrn)
        assert len(out.nodes) == len(out2.nodes)

    def test_downsample_ignores_bogus_multi_node_soma(self):
        """navis' radius>=1 soma detection flags every neuprint node (nm
        radii); a whole-neuron "soma" must not freeze the cache level at
        full resolution."""
        pts = [(i * 1.0, 0.0, 0.0) for i in range(200)]
        nrn = make_neuron(pts, [-1] + list(range(199)), radius=10.0)
        # reproduce the bogus detection: every node qualifies as soma
        assert nrn.soma is not None and len(nrn.soma) > 1
        out = morph._downsample_for_cache(nrn)
        # strictly reduced: the bogus soma is treated as no soma
        assert len(out.nodes) < len(nrn.nodes)
        # the original neuron is untouched (caller keeps vectorizing raw)
        assert nrn.soma is not None and len(nrn.soma) > 1

    def test_vectors_for_skips_simp90_files_when_basis_raw(self, tmp_path):
        """On-disk simplified files are never vectorized into a raw cache."""
        cache = TestSkeletonVectorCache()._setup(tmp_path)
        cache.build()
        # mark the folder simplified (post-cleanup NeuPrint state)
        morph._write_skeleton_level_marker("test:v1", str(tmp_path))
        # a NEW simplified file appears on disk: must NOT be vectorized
        folder = tmp_path / "cache" / morph._dataset_folder("test:v1") / "skeletons"
        with open(folder / "999.pkl", "wb") as f:
            pickle.dump(line_neuron(), f)
        X, mask, reps = cache.vectors_for([101, 999], compute_missing=True)
        assert mask.tolist() == [True, False]  # 999 stays unscorable
        assert np.isnan(X[1]).all()
        assert reps[1] == ""
        # the cache basis was recorded as raw
        assert (cache._load_meta() or {}).get("vector_basis") == "raw"

    def test_vectors_for_computes_from_raw_files(self, tmp_path):
        """Raw folder level + raw basis: missing rows still compute (legacy)."""
        cache = TestSkeletonVectorCache()._setup(tmp_path)
        cache.build()
        folder = tmp_path / "cache" / morph._dataset_folder("test:v1") / "skeletons"
        with open(folder / "999.pkl", "wb") as f:
            pickle.dump(line_neuron(), f)
        X, mask, _ = cache.vectors_for([101, 999], compute_missing=True)
        assert mask.tolist() == [True, True]
        assert np.isfinite(X[1]).all()

    def test_build_skips_files_at_wrong_level(self, tmp_path):
        """build() must not vectorize on-disk files whose level != basis."""
        cache = TestSkeletonVectorCache()._setup(tmp_path)
        cache.build()
        morph._write_skeleton_level_marker("test:v1", str(tmp_path))
        # new simp90 file: build() skips it entirely
        folder = tmp_path / "cache" / morph._dataset_folder("test:v1") / "skeletons"
        with open(folder / "999.pkl", "wb") as f:
            pickle.dump(line_neuron(), f)
        stats = cache.build()
        assert stats["new"] == 0 and stats["rows"] == 3

    def test_append_vectors_rejects_different_basis(self, tmp_path):
        cache = TestSkeletonVectorCache()._setup(tmp_path)
        cache.build()
        _, vec = morph.vectorize_neuron(line_neuron())
        n = cache.append_vectors([(999, vec, "skeleton")],
                                 vector_basis="simp90")
        assert n == 0  # basis mismatch: never mixed
        # same basis appends fine
        n = cache.append_vectors([(999, vec, "skeleton")],
                                 vector_basis="raw")
        assert n == 1

    def test_discover_ignores_temp_cache(self, tmp_path):
        """Transient crash-resume staging files are never treated as cached
        skeletons (the legacy cache's skeleton_dir contains _temp_cache)."""
        cache = TestSkeletonVectorCache()._setup(tmp_path)
        temp = cache.temp_cache_dir()
        temp.mkdir(parents=True)
        (temp / "999.swc.zst").write_bytes(b"x")
        discovered = cache._discover_skeleton_files()
        assert all("_temp_cache" not in p for p in discovered)
        assert not any("999" in p for p in discovered)

    def test_build_folds_pending_and_clears_it(self, tmp_path):
        """A full rebuild folds appended-but-unmerged pending rows into the
        new main parquet and clears the staging file."""
        cache = TestSkeletonVectorCache()._setup(tmp_path)
        cache.build()
        _, vec = morph.vectorize_neuron(line_neuron())
        cache.append_vectors([(999, vec, "skeleton")])
        assert cache.pending_path.exists()
        stats = cache.build()
        assert stats["rows"] == 4  # 101, 102, 103 + folded 999
        assert not cache.pending_path.exists()
        assert 999 in set(cache.load()["bodyIds"].tolist())

    def test_population_stats_falls_back_to_cache_meta(self, tmp_path):
        """simp90 on-disk files -> population stats come from the raw vector
        cache meta (no raw skeleton sample exists any more)."""
        cache = TestSkeletonVectorCache()._setup(tmp_path)
        cache.build()
        morph._write_skeleton_level_marker("test:v1", str(tmp_path))
        mu, sd = morph.population_stats("test:v1", str(tmp_path))
        meta = cache._load_meta()
        assert mu is not None and sd is not None
        assert np.allclose(mu, meta["mean"])
        assert np.allclose(sd, meta["std"])

    def test_population_stats_computes_from_raw_files(self, tmp_path, monkeypatch):
        """Raw folder: sample-based stats still computed from on-disk files."""
        TestPopulationStats()._patch_sequential(monkeypatch)
        cache = TestSkeletonVectorCache()._setup(tmp_path)
        cache.build()
        mu, sd = morph.population_stats("test:v1", str(tmp_path))
        assert mu is not None and sd is not None
        assert mu.shape == (morph.VECTOR_DIM,)


# ---------------------------------------------------------------------------
# Shared simplify + compress pipeline (`.swc.zst`, recorded level)
# ---------------------------------------------------------------------------

class TestSimplificationPipeline:
    """The shared simplify + compress cache pipeline: level validation,
    header recording, re-leveling on load, vector-before-simplify order."""

    def _raw_cache(self, tmp_path, dataset="np:v1", raw_format="swc.zst"):
        return morph.SkeletonVectorCache(
            dataset, project_root=str(tmp_path), verbose=False,
            raw_only=True, raw_format=raw_format)

    def test_level_validation(self):
        assert morph._simplification_factor(90) == 10
        assert morph._simplification_factor(50) == 2
        assert morph._simplification_factor(0) == 1
        for bad in (91, 100, -1, "50x", None):
            with pytest.raises(ValueError):
                morph._simplification_factor(bad)

    def test_header_roundtrip_and_stored_level(self, tmp_path):
        """simplification=50 writes a .swc.zst with a recorded header and
        roughly half the raw nodes."""
        nrn = line_neuron(length=200)
        raw = len(nrn.nodes)
        cache = self._raw_cache(tmp_path)
        assert cache.persist_skeletons({101: nrn}, simplification=50) == 1
        path = cache.skeleton_dir / "101.swc.zst"
        assert path.exists()
        import zstandard as zstd
        with open(path, "rb") as handle:
            with zstd.ZstdDecompressor().stream_reader(handle) as reader:
                content = reader.read()
        assert b"# DROCAT simpl: 50" in content
        loaded = morph._load_cached_skeleton_file(path)
        assert loaded._drocat_simplification == 50
        assert 0.3 * raw <= len(loaded.nodes) <= 0.7 * raw

    def test_default_simplification_is_raw(self, tmp_path):
        """persist_skeletons defaults to raw (level 0): the cache stores
        raw skeletons; simplification is a visualization-time concern."""
        nrn = line_neuron(length=200)
        cache = self._raw_cache(tmp_path)
        cache.persist_skeletons({101: nrn})  # default raw
        loaded = morph._load_cached_skeleton_file(
            cache.skeleton_dir / "101.swc.zst")
        assert (loaded._drocat_simplification or 0) == 0
        assert len(loaded.nodes) == len(nrn.nodes)
        # an explicitly requested coarse level still stores that level
        cache.persist_skeletons({102: nrn}, simplification=90)
        loaded90 = morph._load_cached_skeleton_file(
            cache.skeleton_dir / "102.swc.zst")
        assert (loaded90._drocat_simplification or 0) == 90

    def test_raw_level_zero_records_header(self, tmp_path):
        nrn = line_neuron(length=200)
        cache = self._raw_cache(tmp_path)
        cache.persist_skeletons({101: nrn}, simplification=0)
        loaded = morph._load_cached_skeleton_file(
            cache.skeleton_dir / "101.swc.zst")
        assert loaded._drocat_simplification == 0
        assert len(loaded.nodes) == len(nrn.nodes)
        manifest = json.loads(
            (cache.skeleton_dir / "raw_skeleton_manifest.json").read_text()
        )
        assert manifest["cache_schema_version"] == morph.RAW_SKELETON_CACHE_VERSION
        assert manifest["vector_basis"] == morph.VECTOR_BASIS_RAW
        assert manifest["files"]["101"]["simplification"] == 0
        assert manifest["files"]["101"]["representation"] == "skeleton"

    def test_relevel_stored50_to_target90(self, tmp_path):
        """Stored 50% -> target 90% re-simplifies by factor 5 (keeps 20% of
        the remaining nodes = "simplify by 80%")."""
        nrn = line_neuron(length=200)
        raw_nodes = len(nrn.nodes)
        cache = self._raw_cache(tmp_path)
        cache.persist_skeletons({101: nrn}, simplification=50)
        path = cache.skeleton_dir / "101.swc.zst"
        stored = morph._load_cached_skeleton_file(path)
        assert 0.25 * raw_nodes <= len(stored.nodes) <= 0.75 * raw_nodes
        releveled = morph._load_cached_skeleton_file(
            path, target_simplification=90)
        assert releveled._drocat_simplification == 50  # source level kept
        # factor (100-50)/(100-90) = 5: ~20% of the stored nodes remain
        assert len(releveled.nodes) < len(stored.nodes)
        assert len(releveled.nodes) == pytest.approx(
            len(stored.nodes) / 5, rel=0.5)

    def test_relevel_never_upsamples(self, tmp_path):
        """Target <= stored: detail cannot be restored -> file used as-is."""
        nrn = line_neuron(length=200)
        cache = self._raw_cache(tmp_path)
        cache.persist_skeletons({101: nrn}, simplification=90)
        path = cache.skeleton_dir / "101.swc.zst"
        stored = morph._load_cached_skeleton_file(path)
        reloaded = morph._load_cached_skeleton_file(
            path, target_simplification=50)
        assert len(reloaded.nodes) == len(stored.nodes)

    def test_legacy_headerless_swc_gz_reads_as_raw(self, tmp_path):
        cache = self._raw_cache(tmp_path)
        path = cache.skeleton_dir / "202.swc.gz"
        morph._write_compressed_swc(path, line_neuron(length=100))
        loaded = morph._load_cached_skeleton_file(path)
        assert loaded._drocat_simplification == 0

    def test_vectors_before_simplification_order(self, tmp_path, monkeypatch):
        """fetch_skeleton_on_demand vectorizes the RAW neuron before the
        simplified file is written; the vector cache is standalone."""
        events = []
        nrn = line_neuron(length=200)
        nrn.id = 101
        monkeypatch.setattr(morph, "_fetch_neuprint_skeleton",
                            lambda d, b: nrn)

        real_vectorize = morph.cache_fetched_skeleton_vectors

        def recording_vectorize(dataset, neurons, **kwargs):
            events.append(("vectorize", int(next(iter(neurons)))))
            return real_vectorize(dataset, neurons, **kwargs)

        monkeypatch.setattr(morph, "cache_fetched_skeleton_vectors",
                            recording_vectorize)

        real_downsample = morph._downsample_for_cache

        def recording_downsample(neuron, downsampling_factor=10):
            events.append(("simplify", int(downsampling_factor)))
            return real_downsample(neuron, downsampling_factor)

        monkeypatch.setattr(morph, "_downsample_for_cache",
                            recording_downsample)

        out = morph.fetch_skeleton_on_demand(
            "np:v1", 101, project_root=str(tmp_path), persist=True)
        assert out is not None
        kinds = [e[0] for e in events]
        # Raw-basis design: the raw neuron is vectorized and NO cache-time
        # simplification runs at all (simplification is render-time only).
        assert "vectorize" in kinds
        assert "simplify" not in kinds
        # standalone vector cache holds the raw-basis row
        cache = morph.find_similar_raw_cache(
            "np:v1", project_root=str(tmp_path), verbose=False)
        data = cache.load()
        assert data is not None
        assert 101 in set(data["bodyIds"].tolist())

    def test_simplified_files_never_vectorized_into_raw_basis(self, tmp_path):
        """A simplified on-disk file must not feed the raw-basis vector
        cache (vectors come from fetch-time raw vectorization only)."""
        nrn = line_neuron(length=200)
        cache = self._raw_cache(tmp_path, dataset="np:v1")
        cache.persist_skeletons({101: nrn}, simplification=90)
        X, mask, reps = cache.vectors_for([101], compute_missing=True)
        assert mask.tolist() == [False]
        assert np.isnan(X[0]).all()
        assert reps[0] == ""

    def test_fetch_rejects_none_simplification(self, tmp_path, monkeypatch):
        """None is never accepted on the fetch entry points: 0 = raw is the
        explicit escape (persist_skeletons keeps None as migration mode)."""
        monkeypatch.setattr(morph, "_fetch_neuprint_skeleton",
                            lambda d, b: line_neuron())
        with pytest.raises(ValueError):
            morph.fetch_skeleton_on_demand(
                "np:v1", 101, project_root=str(tmp_path),
                simplification=None)
        with pytest.raises(ValueError):
            morph.fetch_skeletons_on_demand_batch(
                "np:v1", [101], project_root=str(tmp_path),
                simplification=None)

    def test_download_all_simplification_levels(self, tmp_path, monkeypatch):
        """Download All writes one recorded level per run and reports it."""
        write_neuron_index(tmp_path, "np:v1", [
            (101, "T", "T_1"), (201, "T", "T_2"),
        ])
        cache_dir = tmp_path / "cache" / "np_v1"
        cache_dir.mkdir(parents=True)
        (cache_dir / "connections.parquet").touch()
        fetched = []

        def fake_fetch(dataset, bid, project_root=None, persist=True, **kwargs):
            fetched.append(bid)
            neuron = line_neuron(length=200)
            neuron.id = int(bid)
            return neuron

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", fake_fetch)
        summary = morph.download_all_skeletons(
            "np:v1", project_root=str(tmp_path), max_workers=2,
            simplification=50, verbose=False)
        assert summary["fetched"] == 2
        assert summary["simplification"] == 50
        raw_dir = cache_dir / "skeletons" / "raw_skeletons"
        for bid in (101, 201):
            loaded = morph._load_cached_skeleton_file(
                raw_dir / f"{bid}.swc.zst")
            assert loaded._drocat_simplification == 50

        # a second run at a different level never re-fetches cached ids
        summary0 = morph.download_all_skeletons(
            "np:v1", project_root=str(tmp_path), max_workers=2,
            simplification=0, verbose=False)
        assert summary0["fetched"] == 0
        assert summary0["skipped_existing"] == 2
        assert summary0["simplification"] == 0

        # extend the population; a fresh raw-level run writes level-0 files
        # with every node
        write_neuron_index(tmp_path, "np:v1", [
            (101, "T", "T_1"), (201, "T", "T_2"), (301, "Z", "Z_1"),
        ])
        summary1 = morph.download_all_skeletons(
            "np:v1", project_root=str(tmp_path), max_workers=2,
            simplification=0, verbose=False)
        assert summary1["fetched"] == 1 and 301 in fetched
        raw301 = morph._load_cached_skeleton_file(raw_dir / "301.swc.zst")
        assert raw301._drocat_simplification == 0
        assert len(raw301.nodes) == len(line_neuron(length=200).nodes)

    def test_download_all_rejects_invalid_level(self, tmp_path, monkeypatch):
        write_neuron_index(tmp_path, "np:v1", [(101, "T", "T_1")])
        cache_dir = tmp_path / "cache" / "np_v1"
        cache_dir.mkdir(parents=True)
        (cache_dir / "connections.parquet").touch()
        with pytest.raises(ValueError):
            morph.download_all_skeletons(
                "np:v1", project_root=str(tmp_path), simplification=95,
                verbose=False)

    def test_mixed_cache_resume_counts_both_suffixes(self, tmp_path):
        """Raw .swc.gz, simp50 .swc.zst and simp90 .swc.zst all load and all
        count as cached for resume."""
        cache = self._raw_cache(tmp_path, dataset="np:v1")
        cache.persist_skeletons({101: line_neuron(length=200)},
                                simplification=90)
        cache.persist_skeletons({201: line_neuron(length=200)},
                                simplification=50)
        morph._write_compressed_swc(cache.skeleton_dir / "301.swc.gz",
                                    line_neuron(length=200))
        files = cache._discover_skeleton_files()
        assert {morph._skeleton_body_id(p) for p in files} == {101, 201, 301}
        for bid in (101, 201, 301):
            assert cache.load_skeleton(bid) is not None

    def test_lazy_migration_preserves_stored_level(self, tmp_path):
        """A legacy .swc.gz loaded with a target level migrates the AS-STORED
        content (level 0) to .swc.zst; only the returned neuron is re-leveled
        to the target."""
        cache = self._raw_cache(tmp_path, dataset="np:v1")
        nrn = line_neuron(length=200)
        legacy_dir = (tmp_path / "cache" / "np_v1" / "find_similar"
                      / "raw_skeletons")
        legacy_dir.mkdir(parents=True)
        morph._write_compressed_swc(legacy_dir / "101.swc.gz", nrn)
        loaded = cache.load_skeleton(101, simplification=90)
        assert loaded is not None
        assert len(loaded.nodes) < 0.2 * len(nrn.nodes)  # target level
        migrated = cache.skeleton_dir / "101.swc.zst"
        assert migrated.exists()
        migrated_neuron = morph._load_cached_skeleton_file(migrated)
        # the migrated file holds the RAW content tagged level 0
        assert migrated_neuron._drocat_simplification == 0
        assert len(migrated_neuron.nodes) == len(nrn.nodes)


# ---------------------------------------------------------------------------
# FlyWire isolation: the simplification pipeline is NeuPrint-only
# ---------------------------------------------------------------------------

class TestFlywireIsolation:
    """The shared simplify + compress pipeline must never leak into the
    FAFB/BANC local-release workflow: no re-leveling of meshes, no simplified raw
    skeletons, no .swc.zst skeleton artifacts from FAFB fetches."""

    def test_relevel_never_touches_meshes(self, tmp_path, monkeypatch):
        mesh = cube_mesh()
        # identity: _relevel_for_target returns the mesh untouched
        assert morph._relevel_for_target(mesh, 0, 90) is mesh
        assert morph._relevel_for_target(mesh, 50, 90) is mesh

        # load_skeleton must not even attempt a downsample on a mesh
        def forbidden_downsample(neuron, downsampling_factor=10):
            raise AssertionError("mesh must never be downsampled")

        monkeypatch.setattr(morph, "_downsample_for_cache",
                            forbidden_downsample)
        cache = morph.SkeletonVectorCache(
            "x:v1", project_root=str(tmp_path), verbose=False)
        path = cache.skeleton_dir / "101.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(mesh, f)
        loaded = cache.load_skeleton(101, simplification=90)
        import navis
        assert isinstance(loaded, navis.MeshNeuron)
        assert len(loaded.vertices) == len(mesh.vertices)

    def test_flywire_fetch_never_writes_raw_skeleton_files(
            self, tmp_path, monkeypatch):
        """A FlyWire fetch returns a MeshNeuron and persists nothing into
        the NeuPrint raw-skeleton namespace (.swc.zst / .swc.gz)."""
        monkeypatch.setattr(morph, "_fetch_cave_mesh",
                            lambda d, b, project_root=None, use_cache=True,
                            soma_pos=None: cube_mesh())
        nrn = morph.fetch_skeleton_on_demand(
            "flywire_FAFB_v783", 42, project_root=str(tmp_path),
            persist=True)
        import navis
        assert isinstance(nrn, navis.MeshNeuron)
        raw_dir = (tmp_path / "cache" / "flywire_FAFB_v783"
                   / "skeletons" / "raw_skeletons")
        if raw_dir.exists():
            assert not list(raw_dir.rglob("*.swc.zst"))
            assert not list(raw_dir.rglob("*.swc.gz"))

    def test_flywire_batch_loads_mesh_with_simplification_zero(
            self, tmp_path, monkeypatch):
        """The batch forces simplification=0 on FlyWire: cached meshes are
        requested raw and never re-leveled."""
        class FakeMeshCache:
            mesh_only = True

            def __init__(self):
                self.calls = []

            def load_skeleton(self, bid, simplification=None):
                self.calls.append((bid, simplification))
                mesh = cube_mesh()
                mesh.id = int(bid)
                return mesh

        fake = FakeMeshCache()
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("cache hit must not fetch")))
        out = morph.fetch_skeletons_on_demand_batch(
            "flywire_FAFB_v783", [101], project_root=str(tmp_path),
            persist=True, raw_cache=fake, vector_cache=fake,
        )
        # FlyWire canonical ids are strings; the load request is raw (0)
        assert "101" in out
        assert fake.calls == [("101", 0)]

    def test_flywire_download_all_disabled_with_instruction(
            self, tmp_path, monkeypatch):
        """download_all_skeletons on FlyWire raises the explicit manual
        Codex instruction; nothing is fetched or persisted."""
        TestFafbBundleLocalSources()._write_bundle(tmp_path, [2, 3])
        write_neuron_index(tmp_path, "flywire_FAFB_v783", [
            (1, "T", "T_1"), (2, "T", "T_2"), (3, "T", "T_3"),
        ])
        cache_dir = tmp_path / "cache" / "flywire_FAFB_v783"
        cache_dir.mkdir(parents=True)
        (cache_dir / "connections.parquet").touch()
        monkeypatch.setattr(
            morph, "fetch_skeleton_on_demand",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("FlyWire bulk download must not fetch")))
        with pytest.raises(morph.FlyWireSkeletonAccessError) as excinfo:
            morph.download_all_skeletons(
                "flywire_FAFB_v783", project_root=str(tmp_path),
                max_workers=2, verbose=False, simplification=50)
        message = str(excinfo.value)
        assert "disabled for FAFB" in message
        assert "https://codex.flywire.ai/api/download?dataset=fafb" in message
        assert "sk_lod1_783_healed.zip" in message
        assert "FAFB_file_converter" in message
        assert not (cache_dir / "meshes").exists()
        assert not (cache_dir / "skeletons" / "raw_skeletons").exists()

        # BANC pulls are allowed: skeletons fetch per-neuron from the
        # public release bucket. Without local tables there is nothing to
        # enumerate, so the pull completes with an empty summary (hermetic:
        # no network access happens).
        summary = morph.download_all_skeletons(
            "banc_v888", project_root=str(tmp_path),
            max_workers=2, verbose=False)
        assert summary["total"] == 0
        assert summary["fetched"] == 0
        assert summary["representation"] == "skeleton"


# ---------------------------------------------------------------------------
# Candidate source & profile-first search
# ---------------------------------------------------------------------------

class TestCandidateSource:
    def test_auto_resolution(self):
        c = morph.MorphologyComparer(query=1, dataset="hemibrain:v1.2.1",
                                     project_root="/tmp", verbose=False)
        assert c._resolved_candidate_source() == "profile"
        c2 = morph.MorphologyComparer(query=1, dataset="flywire_FAFB_v783",
                                      project_root="/tmp", verbose=False)
        assert c2._resolved_candidate_source() == "cache"
        c3 = morph.MorphologyComparer(query=1, dataset="male-cns:v0.9",
                                      candidate_source="cache",
                                      project_root="/tmp", verbose=False)
        assert c3._resolved_candidate_source() == "cache"

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="candidate_source"):
            morph.MorphologyComparer(query=1, dataset="x", candidate_source="bad",
                                     project_root="/tmp", verbose=False)

    def test_size_knobs_are_visualize_top_n_and_candidate_cap(self):
        """The only size knobs across candidate source modes are the
        visualize top-N and the candidate cap (deprecated top-N/expansion
        parameters are gone)."""
        c = morph.MorphologyComparer(query=1, dataset="x",
                                     visualize_top_n=7, candidate_cap=250,
                                     project_root="/tmp", verbose=False)
        assert c.visualize_top_n == 7
        assert c.candidate_cap == 250
        # NBLAST prefilter and per-type sampling derive from the same cap
        assert c.candidate_cap == 250
        import inspect
        sig = inspect.signature(morph.MorphologyComparer.__init__)
        for gone in ("top_n", "nblast_prefilter", "n_per_type",
                     "fetch_top_n", "fetch_missing", "candidate_expansion",
                     "max_pool_per_type"):
            assert gone not in sig.parameters, gone


def write_roi_dataset(tmp_path, dataset, counts, metadata_rois=None,
                      neuron_df=None):
    """Minimal local ROI dataset: roi-count CSV, neuron table, sidecar.

    ``counts``: {bodyId: {"post": {roi: n}, "pre": {roi: n},
    "type": str}}. Neuron-table pre/post totals are the ROI sums, so the
    sidecar's primary list is a true partition. ``neuron_df`` overrides the
    neuron table (to give true totals when counts include hierarchical
    parent ROIs).
    """
    folder = morph._dataset_folder(dataset)
    base = tmp_path / "datasets" / folder
    base.mkdir(parents=True, exist_ok=True)
    rows, neuron_rows = [], []
    for bid, blocks in counts.items():
        pre_map, post_map = blocks.get("pre", {}), blocks.get("post", {})
        for roi in set(pre_map) | set(post_map):
            rows.append({"bodyId": bid, "roi": roi,
                         "pre": pre_map.get(roi, 0),
                         "post": post_map.get(roi, 0)})
        neuron_rows.append({"bodyId": bid,
                            "type": blocks.get("type", ""),
                            "instance": blocks.get("instance", ""),
                            "pre": sum(pre_map.values()),
                            "post": sum(post_map.values())})
    pd.DataFrame(rows).to_csv(base / f"{folder}_allneurons_roi_count_df.csv",
                              index=False)
    if neuron_df is not None:
        neuron_rows = neuron_df
    pd.DataFrame(neuron_rows).to_csv(
        base / f"{folder}_allneurons_neuron_df.csv", index=False)
    rois_list = (metadata_rois if metadata_rois is not None
                 else sorted({r for c in counts.values()
                              for r in list(c.get("pre", {}))
                              + list(c.get("post", {}))}))
    meta = {"dataset": dataset, "source": "neuprint",
            "roi_coverage": {"roi_list": rois_list,
                             "roi_count": len(rois_list),
                             "neuron_counts_per_roi": {}}}
    (base / f"{folder}_metadata.json").write_text(json.dumps(meta))
    return base


def roi_fixture_counts():
    """Query 101 (right), its contralateral twin 201, a same-hemisphere
    decoy 202 and an unrelated midline neuron 204."""
    return {
        101: {"post": {"A(R)": 10, "M": 5}, "pre": {"A(R)": 8, "M": 4},
              "type": "T", "instance": "T_R"},
        201: {"post": {"A(L)": 10, "M": 5}, "pre": {"A(L)": 8, "M": 4},
              "type": "T2", "instance": "T2_L"},
        202: {"post": {"A(R)": 10, "M": 50}, "pre": {"A(R)": 8, "M": 40},
              "type": "Y", "instance": "Y_R"},
        204: {"post": {"M": 3}, "pre": {"M": 3},
              "type": "Z", "instance": "Z_1"},
    }


class TestRoiCandidateSource:
    def test_auto_resolves_roi_when_data_present(self, tmp_path):
        write_roi_dataset(tmp_path, "np:v1", roi_fixture_counts())
        c = morph.MorphologyComparer(query=101, dataset="np:v1",
                                     project_root=str(tmp_path), verbose=False)
        assert c._resolved_candidate_source() == "roi"

    def test_auto_stays_profile_when_backfill_fails(self, tmp_path,
                                                    monkeypatch):
        # ROI table exists but no sidecar: without a successful backfill the
        # auto source must not promise ROI screening.
        write_roi_dataset(tmp_path, "np:v1", roi_fixture_counts())
        (tmp_path / "datasets" / "np_v1" / "np_v1_metadata.json").unlink()
        monkeypatch.setattr(morph, "backfill_dataset_metadata",
                            lambda *a, **k: None)
        c = morph.MorphologyComparer(query=101, dataset="np:v1",
                                     project_root=str(tmp_path), verbose=False)
        assert c._resolved_candidate_source() == "profile"

    def test_auto_without_roi_table_stays_profile(self, tmp_path):
        c = morph.MorphologyComparer(query=1, dataset="hemibrain:v1.2.1",
                                     project_root=str(tmp_path), verbose=False)
        assert c._resolved_candidate_source() == "profile"


class TestRoiProfileFirst:
    """ROI-screened pipeline: primary-ROI cosine candidates -> top types ->
    all members -> morphology. Mirrors TestProfileFirst's fixtures."""

    def _setup(self, tmp_path, monkeypatch, query=101, **kwargs):
        write_roi_dataset(tmp_path, "np:v1", roi_fixture_counts())
        write_skeleton(tmp_path, "np:v1", query, line_neuron(length=20))
        write_skeleton(tmp_path, "np:v1", 201, line_neuron(length=20))
        write_skeleton(tmp_path, "np:v1", 202, bushy_y_neuron())
        params = dict(
            query=query, dataset="np:v1", level="bodyid", method="vector_v2", expand_top_types=0,
            candidate_source="roi",
            output_dir=str(tmp_path / "out"), project_root=str(tmp_path),
            verbose=False,
        )
        params.update(kwargs)
        comparer = morph.MorphologyComparer(**params)
        monkeypatch.setattr(comparer, "_resolve_query", lambda: pd.DataFrame({
            "bodyId": [query], "type": ["T"], "instance": ["T_1"],
        }))
        return comparer

    def _fake_fetch(self, monkeypatch):
        fetched = []

        def fake_fetch(dataset, bid, project_root=None, persist=True):
            fetched.append(bid)
            return line_neuron(length=25)

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", fake_fetch)
        return fetched

    def test_roi_source_ranks_twin_and_carries_roi_scores(self, tmp_path,
                                                          monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch)
        fetched = self._fake_fetch(monkeypatch)
        res = comparer.find_similar()
        assert not res.empty
        # pool: every typed neuron; only 204 has no cached skeleton
        assert fetched == [204]
        # the mirrored twin is the ROI screen's top candidate
        roi_scores = res.set_index("target_bodyId")["roi_similarity"]
        assert roi_scores.index[0] == res.iloc[0]["target_bodyId"]
        assert 201 in roi_scores.index
        assert roi_scores[201] == pytest.approx(1.0, abs=1e-4)
        # screen evidence columns and source label
        assert res["roi_similarity"].notna().all()
        assert res["profile_similarity"].isna().all()
        assert (res["candidate_source"] == "roi").all()
        assert 101 not in res["target_bodyId"].tolist()

    def test_combined_merges_both_screens(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch,
                               candidate_source="combined")
        self._fake_fetch(monkeypatch)
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: pd.DataFrame({
                                "target_bodyId": [201, 204],
                                "shared_count": [9, 3],
                                "profile_similarity": [1.0, 0.3],
                                "target_type": ["T2", "Z"],
                            }))
        res = comparer.find_similar()
        assert not res.empty
        by_id = res.set_index("target_bodyId")
        # connectivity-only evidence: 201/204 have profile scores
        assert by_id.loc[201, "profile_similarity"] == pytest.approx(1.0)
        assert by_id.loc[204, "profile_similarity"] == pytest.approx(0.3)
        # ROI-only evidence: 202 was not a connectivity candidate
        assert pd.isna(by_id.loc[202, "profile_similarity"])
        assert np.isfinite(by_id.loc[202, "roi_similarity"])
        assert (res["candidate_source"] == "combined").all()
        # candidate types came from BOTH screens (Y only via ROI)
        readme = Path(comparer.output_folder, "README.txt").read_text()
        assert "candidate_source: combined" in readme

    def test_auto_roi_falls_back_to_profile_in_run(self, tmp_path,
                                                   monkeypatch):
        """A hierarchical sidecar passes the cheap availability probe but
        fails the partition validation at build: an auto run must degrade to
        the connection-cache screen, an explicit roi run must raise."""
        counts = roi_fixture_counts()
        # parent ROI 'A' re-counts the hemispheric synapses (hierarchy)
        for bid, parent_post, parent_pre in ((101, 10, 8), (201, 10, 8),
                                             (202, 10, 8)):
            counts[bid]["post"]["A"] = parent_post
            counts[bid]["pre"]["A"] = parent_pre
        neuron_df = [
            {"bodyId": 101, "type": "T", "instance": "T_R", "pre": 8,
             "post": 15},
            {"bodyId": 201, "type": "T2", "instance": "T2_L", "pre": 8,
             "post": 15},
            {"bodyId": 202, "type": "Y", "instance": "Y_R", "pre": 48,
             "post": 60},
            {"bodyId": 204, "type": "Z", "instance": "Z_1", "pre": 3,
             "post": 3},
        ]
        write_roi_dataset(tmp_path, "np:v1", counts,
                          metadata_rois=["A", "A(L)", "A(R)", "M"],
                          neuron_df=neuron_df)
        write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        write_skeleton(tmp_path, "np:v1", 201, line_neuron(length=20))
        params = dict(
            query=101, dataset="np:v1", level="bodyid", method="vector_v2", expand_top_types=0,
            output_dir=str(tmp_path / "out"), project_root=str(tmp_path),
            verbose=False,
        )
        monkeypatch.setattr(morph, "backfill_dataset_metadata",
                            lambda *a, **k: None)
        # the store's own recovery path must not reach the network either
        monkeypatch.setattr(rois_mod, "backfill_dataset_metadata",
                            lambda *a, **k: None)
        self._fake_fetch(monkeypatch)

        auto = morph.MorphologyComparer(candidate_source="auto", **params)
        monkeypatch.setattr(auto, "_resolve_query", lambda: pd.DataFrame({
            "bodyId": [101], "type": ["T"], "instance": ["T_1"],
        }))
        monkeypatch.setattr(auto, "_connection_cache_candidates",
                            lambda q: pd.DataFrame({
                                "target_bodyId": [201],
                                "shared_count": [9],
                                "profile_similarity": [1.0],
                                "target_type": ["T2"],
                            }))
        res = auto.find_similar()
        assert not res.empty
        assert (res["candidate_source"] == "profile").all()
        assert res["profile_similarity"].notna().all()

        explicit = morph.MorphologyComparer(candidate_source="roi", **params)
        monkeypatch.setattr(explicit, "_resolve_query", lambda: pd.DataFrame({
            "bodyId": [101], "type": ["T"], "instance": ["T_1"],
        }))
        with pytest.raises(morph.RoiScreeningUnavailable):
            explicit.find_similar()


class TestFlyWireNblast:
    """NBLAST on FAFB: it has real skeleton sources (the
    healed skeleton bundle + CAVE fallback), so NBLAST is not blanket-
    blocked; dotprops prefer the bundle's skeletons over the local mesh
    pickle cache."""

    def test_fafb_nblast_not_blanket_blocked(self, tmp_path, monkeypatch):
        """Regression: the old guard rejected every FlyWire NBLAST run with
        'cache contains meshes' even though FAFB skeleton access was
        validated as ready. The run must proceed past that point."""
        monkeypatch.setattr(morph, "require_flywire_skeleton_access",
                            lambda *a, **k: {"ready": True})
        c = morph.MorphologyComparer(query="aMe12", dataset="flywire_FAFB_v783",
                                     method="nblast", project_root=str(tmp_path),
                                     verbose=False)

        class Sentinel(Exception):
            pass

        def boom():
            raise Sentinel()

        monkeypatch.setattr(c, "_resolve_query", boom)
        with pytest.raises(Sentinel):
            c.find_similar()

    def test_dotprops_prefer_healed_bundle_skeletons(self, tmp_path, monkeypatch):
        """FlyWire dotprops follow the shared pipeline: the healed bundle
        serves the skeletons, ids missing everywhere go to the token-gated
        CAVE skeletonization fallback (never the generic on-demand fetch),
        and every returned source is a TreeNeuron."""
        import fafb_utils

        class _Bundle:
            def __init__(self, texts):
                self._texts = texts
                self.closed = False

            def get(self, bid):
                return self._texts.get(int(bid))

            def close(self):
                self.closed = True

        def _swc():
            lines = ["# SWC skeleton"]
            for i in range(1, 31):
                parent = -1 if i == 1 else i - 1
                lines.append(f"{i} 1 {i} 0 0 1.0 {parent}")
            return "\n".join(lines)

        monkeypatch.setattr(morph, "_fafb_bundle",
                            lambda ds, root: _Bundle({42: _swc(),
                                                      43: _swc()}))
        # The extrusion test is exercised separately; keep this test focused
        # on the local-bundle -> CAVE fallback flow.
        monkeypatch.setattr(fafb_utils, "flag_extrusions",
                            lambda *a, **k: [])
        c = morph.MorphologyComparer(query=42, dataset="flywire_FAFB_v783",
                                     method="nblast", project_root=str(tmp_path),
                                     verbose=False)
        cave_ids = []

        def fake_cave(dataset, body_ids, project_root=None, log=None,
                      denoise_twigs=None):
            cave_ids.extend(int(b) for b in body_ids)
            return {}

        monkeypatch.setattr(morph, "_flywire_cave_skeletons", fake_cave)

        def no_fetch(*a, **k):
            raise AssertionError(
                "generic on-demand fetch must not serve FlyWire")

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", no_fetch)
        dps = c._dotprops_for_ids([42, 43, 44])
        # dotprops come back keyed by the dataset-canonical id
        k42, k43, k44 = (c._body_id(b) for b in (42, 43, 44))
        # bundle skeletons served as trees without any online fetch
        assert dps[k42] is not None and dps[k43] is not None
        # ids missing from the bundle go to the CAVE skeletonization fallback
        assert dps[k44] is None
        assert cave_ids == [44]

    def test_nblast_pairwise_scores_match_nblaster(self, tmp_path):
        """The in-process pair scoring produces exactly the NBlaster
        (navis.nblast) forward normalized scores."""
        import navis
        from navis.nbl.nblast_funcs import NBlaster

        c = morph.MorphologyComparer(query=101, dataset="np:v1",
                                     method="nblast", project_root=str(tmp_path),
                                     verbose=False)
        q_dp = navis.make_dotprops(line_neuron(length=20) / 1000.0, k=20)
        cand_dps = {
            202: navis.make_dotprops(bushy_y_neuron() / 1000.0, k=20),
            203: navis.make_dotprops(line_neuron(length=30) / 1000.0, k=20),
        }
        scores = c._nblast_pairwise({101: q_dp}, cand_dps)
        nb = NBlaster(use_alpha=False, normalized=True, progress=False)
        qi = nb.append(q_dp, self_hit=nb.calc_self_hit(q_dp))
        for bid, t_dp in cand_dps.items():
            ti = nb.append(t_dp, self_hit=nb.calc_self_hit(t_dp))
            expected = float(nb.single_query_target(qi, ti, scores='forward'))
            assert scores[bid] == pytest.approx(expected, abs=1e-9)

    def test_fafb_pipeline_replaces_flagged_extrusions(self, tmp_path,
                                                       monkeypatch):
        """With the extrusion check enabled (the default), bundle skeletons
        run through the cached check; flagged neurons are REPLACED through
        the CAVE skeletonization fallback and the repair status is
        recorded."""
        import fafb_utils

        class _Bundle:
            def __init__(self, texts):
                self._texts = texts

            def get(self, bid):
                return self._texts.get(int(bid))

            def close(self):
                pass

        def _swc():
            lines = ["# SWC skeleton"]
            for i in range(1, 31):
                parent = -1 if i == 1 else i - 1
                lines.append(f"{i} 1 {i} 0 0 1.0 {parent}")
            return "\n".join(lines)

        monkeypatch.setattr(morph, "_fafb_bundle",
                            lambda ds, root: _Bundle({42: _swc(),
                                                      43: _swc()}))
        monkeypatch.setattr(fafb_utils, "flag_extrusions",
                            lambda *a, **k: [42])
        statuses = {}
        monkeypatch.setattr(fafb_utils, "set_extrusion_repair_status",
                            lambda root, folder, s: statuses.update(s))
        c = morph.MorphologyComparer(query=42, dataset="flywire_FAFB_v783",
                                     method="nblast", project_root=str(tmp_path),
                                     verbose=False, check_extrusions=True)
        replacement = line_neuron()
        monkeypatch.setattr(morph, "_flywire_cave_skeletons",
                            lambda dataset, body_ids, project_root=None,
                            log=None, denoise_twigs=None: (
                                cave_ids.extend(int(b) for b in body_ids),
                                {42: replacement})[1])
        cave_ids: list = []
        loaded = c._load_fafb_skeletons([42, 43])
        assert loaded[42] is replacement      # flagged -> CAVE replacement
        assert 43 in loaded                   # clean bundle tree kept
        assert cave_ids == [42]
        assert statuses == {42: "api_repaired"}

    def test_fafb_pipeline_skips_extrusions_when_opted_out(self, tmp_path,
                                                           monkeypatch):
        """``check_extrusions=False`` opts the loader out of the detector:
        bundle skeletons are served unchecked and no CAVE refetch happens."""
        import fafb_utils

        class _Bundle:
            def __init__(self, texts):
                self._texts = texts

            def get(self, bid):
                return self._texts.get(int(bid))

            def close(self):
                pass

        def _swc():
            lines = ["# SWC skeleton"]
            for i in range(1, 31):
                parent = -1 if i == 1 else i - 1
                lines.append(f"{i} 1 {i} 0 0 1.0 {parent}")
            return "\n".join(lines)

        monkeypatch.setattr(morph, "_fafb_bundle",
                            lambda ds, root: _Bundle({42: _swc(),
                                                      43: _swc()}))

        def no_check(*a, **k):
            raise AssertionError(
                "extrusion check must not run when explicitly opted out")

        monkeypatch.setattr(fafb_utils, "flag_extrusions", no_check)
        c = morph.MorphologyComparer(query=42, dataset="flywire_FAFB_v783",
                                     method="nblast", project_root=str(tmp_path),
                                     verbose=False, check_extrusions=False)
        assert c.check_extrusions is False

        def no_cave(*a, **k):
            raise AssertionError("CAVE must not run without flagged neurons")

        monkeypatch.setattr(morph, "_flywire_cave_skeletons", no_cave)
        loaded = c._load_fafb_skeletons([42, 43])
        # both bundle skeletons served raw, no extrusion-driven CAVE refetch
        assert 42 in loaded and 43 in loaded

    @staticmethod
    def _zip_bundle(texts):
        class _Bundle:
            def get(self, bid):
                return texts.get(int(bid))

            def close(self):
                pass
        return _Bundle()

    @staticmethod
    def _line_swc(n=30):
        lines = ["# SWC skeleton"]
        for i in range(1, n + 1):
            parent = -1 if i == 1 else i - 1
            lines.append(f"{i} 1 {i} 0 0 1.0 {parent}")
        return "\n".join(lines)

    def test_fafb_pipeline_persists_local_fix_and_serves_next_run(
            self, tmp_path, monkeypatch):
        """CAVE unavailable + a safe local prune: the pruned fix is persisted
        to the ``extrusion_fixes`` store with status ``local_fallback``, and
        the next run serves it from the repair caches before ever loading
        the zip tree."""
        import fafb_utils
        from cave_data_fetcher import CAVEDataFetcher

        monkeypatch.setattr(
            morph, "_fafb_bundle",
            lambda ds, root: self._zip_bundle({42: self._line_swc()}))
        monkeypatch.setattr(fafb_utils, "flag_extrusions",
                            lambda *a, **k: [42])
        monkeypatch.setattr(morph, "_flywire_cave_skeletons",
                            lambda *a, **k: {})  # CAVE down
        pruned = line_neuron()
        monkeypatch.setattr(
            fafb_utils, "repair_extruded_skeleton",
            lambda n: (pruned, {"repaired": True, "removed_nodes": 3}))

        first = morph.load_local_release_skeletons(
            "flywire_FAFB_v783", [42], project_root=str(tmp_path))
        assert first[42] is pruned

        fix_fetcher = CAVEDataFetcher(
            dataset="flywire_FAFB_v783", cave_token="",
            project_root=str(tmp_path), cache_enabled=True, verbose=False)
        assert fix_fetcher.load_cached_fix_skeleton(42) is not None
        statuses = fafb_utils.load_extrusion_repair_status(
            str(tmp_path), "flywire_FAFB_v783")
        assert statuses.get("42") == "local_fallback"

        # second run: the zip tree must never be read again
        def no_zip(*a, **k):
            raise AssertionError("repaired body must not reload the zip")

        monkeypatch.setattr(morph, "_fafb_bundle", no_zip)
        second = morph.load_local_release_skeletons(
            "flywire_FAFB_v783", [42], project_root=str(tmp_path))
        assert second[42]._drocat_source == "local_extrusion_fix"

    def test_fafb_pipeline_api_failed_retries_cave_next_run(
            self, tmp_path, monkeypatch):
        """``api_failed`` stays retryable: the next run re-attempts CAVE and
        records ``api_repaired`` when the replacement arrives."""
        import fafb_utils

        monkeypatch.setattr(
            morph, "_fafb_bundle",
            lambda ds, root: self._zip_bundle({42: self._line_swc()}))
        monkeypatch.setattr(fafb_utils, "flag_extrusions",
                            lambda *a, **k: [42])
        monkeypatch.setattr(
            fafb_utils, "repair_extruded_skeleton",
            lambda n: (n, {"repaired": False}))  # no safe local cut
        monkeypatch.setattr(morph, "_flywire_cave_skeletons",
                            lambda *a, **k: {})  # CAVE down on run 1
        first = morph.load_local_release_skeletons(
            "flywire_FAFB_v783", [42], project_root=str(tmp_path))
        statuses = fafb_utils.load_extrusion_repair_status(
            str(tmp_path), "flywire_FAFB_v783")
        assert statuses.get("42") == "api_failed"
        assert first[42].nodes is not None  # extruded tree served as-is

        replacement = line_neuron()
        monkeypatch.setattr(
            morph, "_flywire_cave_skeletons",
            lambda dataset, body_ids, project_root=None, log=None,
            denoise_twigs=None: {int(b): replacement for b in body_ids})
        second = morph.load_local_release_skeletons(
            "flywire_FAFB_v783", [42], project_root=str(tmp_path))
        assert second[42] is replacement
        statuses = fafb_utils.load_extrusion_repair_status(
            str(tmp_path), "flywire_FAFB_v783")
        assert statuses.get("42") == "api_repaired"


class TestNblastDotpropsRouting:
    """The NBLAST dotprops fetch dispatch must match the v2 missing-fetch
    gate: only FAFB resolves through the healed-bundle loader; BANC follows
    the NeuPrint batch fetch (its public SWC chain), never the token-gated
    CAVE pipeline whose fallback returns nothing for a BANC dataset."""

    @staticmethod
    def _comparer(dataset, tmp_path):
        return morph.MorphologyComparer(
            query=42, dataset=dataset, method="nblast",
            project_root=str(tmp_path), verbose=False)

    @staticmethod
    def _patch_raw_cache(monkeypatch, tmp_path, dataset):
        """Pin the raw-cache instance so persistence can be observed."""
        raw_cache = morph.find_similar_raw_cache(
            dataset, project_root=str(tmp_path), n_workers=1, verbose=False)
        persist_calls = []
        real_persist = raw_cache.persist_skeletons

        def spy_persist(mapping):
            persist_calls.append(sorted(int(b) for b in mapping))
            return real_persist(mapping)

        monkeypatch.setattr(raw_cache, "persist_skeletons", spy_persist)
        monkeypatch.setattr(morph, "find_similar_raw_cache",
                            lambda *a, **k: raw_cache)
        return persist_calls

    def test_dotprops_banc_routes_to_batch_fetcher(self, tmp_path,
                                                   monkeypatch):
        c = self._comparer("banc_v888", tmp_path)
        persist_calls = self._patch_raw_cache(monkeypatch, tmp_path,
                                              "banc_v888")
        fetch_calls = []

        def fake_fetch(dataset, body_ids, **kwargs):
            fetch_calls.append((dataset, list(body_ids)))
            return {44: line_neuron(length=20)}

        monkeypatch.setattr(morph, "fetch_skeletons_on_demand_batch",
                            fake_fetch)

        def no_fafb_loader(body_ids, check_extrusions=None):
            raise AssertionError("FAFB pipeline must not serve BANC dotprops")

        monkeypatch.setattr(c, "_load_fafb_skeletons", no_fafb_loader)
        dps = c._dotprops_for_ids([44])
        assert fetch_calls == [("banc_v888", [44])]
        assert dps[c._body_id(44)] is not None
        # the batch fetch owns persistence for BANC (fetch_banc_swc caches
        # its own level-0 raw entry); the manual re-persist must not churn
        # the same files behind it
        assert persist_calls == []

    def test_dotprops_fafb_keeps_fafb_pipeline(self, tmp_path, monkeypatch):
        c = self._comparer("flywire_FAFB_v783", tmp_path)
        self._patch_raw_cache(monkeypatch, tmp_path, "flywire_FAFB_v783")
        loader_calls = []

        def fake_loader(body_ids, check_extrusions=None):
            loader_calls.append(list(body_ids))
            return {}

        monkeypatch.setattr(c, "_load_fafb_skeletons", fake_loader)

        def no_generic_fetch(*a, **k):
            raise AssertionError(
                "generic batch fetcher must not serve FAFB dotprops")

        monkeypatch.setattr(morph, "fetch_skeletons_on_demand_batch",
                            no_generic_fetch)
        dps = c._dotprops_for_ids([42, 43])
        assert loader_calls == [[42, 43]]
        assert dps[c._body_id(42)] is None and dps[c._body_id(43)] is None

    def test_dotprops_neuprint_keeps_batch_fetcher(self, tmp_path,
                                                   monkeypatch):
        c = self._comparer("np:v1", tmp_path)
        self._patch_raw_cache(monkeypatch, tmp_path, "np:v1")
        fetch_calls = []

        def fake_fetch(dataset, body_ids, **kwargs):
            fetch_calls.append((dataset, list(body_ids)))
            return {}

        monkeypatch.setattr(morph, "fetch_skeletons_on_demand_batch",
                            fake_fetch)

        def no_fafb_loader(body_ids, check_extrusions=None):
            raise AssertionError(
                "FAFB loader must not serve NeuPrint dotprops")

        monkeypatch.setattr(c, "_load_fafb_skeletons", no_fafb_loader)
        c._dotprops_for_ids([42, 43])
        assert fetch_calls == [("np:v1", [42, 43])]

    def test_banc_find_similar_guard_intact(self, tmp_path):
        """The BANC deferral guard must survive the routing fix: this change
        is plumbing correctness, not the product decision to enable BANC
        similarity."""
        c = self._comparer("banc_v888", tmp_path)
        with pytest.raises(morph.FlyWireSkeletonAccessError):
            c.find_similar()


class TestTypeReevaluationFetch:
    """The V2 expansion pass fetches per dataset family: FAFB loads through
    the healed-bundle loader (the generic batch fetcher is mesh-native and
    its meshes are dropped by the representation guard), NeuPrint keeps the
    batched SWC fetch."""

    @staticmethod
    def _comparer(dataset, tmp_path):
        query = "aMe4" if dataset.startswith("flywire") else 101
        return morph.MorphologyComparer(
            query=query, dataset=dataset, level="type", method="vector_v2",
            expand_top_types=2, expand_per_type=5,
            project_root=str(tmp_path), verbose=False)

    @staticmethod
    def _expansion_kwargs(cache):
        return dict(
            rows=[],
            pre_type_df=pd.DataFrame(
                [{"target_type": "T1", "is_intra_type": False,
                  "similarity": 0.5}]),
            query_df=pd.DataFrame([{"bodyId": 101, "type": "Q"}]),
            query_ids=[101],
            X_q=np.zeros((1, 4)), mask_q=np.array([True]),
            id_to_type={101: "Q", 42: "T1", 43: "T1"},
            id_to_instance={},
            intra=0.5, source="profile",
            cache=cache, cache_data=None,
            mu=np.zeros(4), sd=np.ones(4),
            scored_ids={101}, pool_len=1, q_rep="skeleton",
        )

    def test_expansion_fafb_uses_bundle_loader(self, tmp_path, monkeypatch):
        import fafb_utils

        c = self._comparer("flywire_FAFB_v783", tmp_path)
        loader_calls = []

        def fake_loader(body_ids, check_extrusions=None):
            loader_calls.append((list(body_ids), check_extrusions))
            return {}

        monkeypatch.setattr(c, "_load_fafb_skeletons", fake_loader)

        def no_generic_fetch(*a, **k):
            raise AssertionError(
                "generic batch fetcher must not serve the FAFB expansion")

        monkeypatch.setattr(morph, "fetch_skeletons_on_demand_batch",
                            no_generic_fetch)
        cache = morph.SkeletonVectorCache(
            "flywire_FAFB_v783", project_root=str(tmp_path), raw_only=True)
        c._expand_top_type_rows(**self._expansion_kwargs(cache))
        # both unscored members routed through the FAFB loader, with the
        # comparer's default extrusion policy (check off)
        assert loader_calls == [([42, 43], None)]

    def test_expansion_neuprint_keeps_batch_fetcher(self, tmp_path,
                                                    monkeypatch):
        c = self._comparer("np:v1", tmp_path)
        fetch_calls = []

        def fake_fetch(dataset, body_ids, **kwargs):
            fetch_calls.append((dataset, list(body_ids)))
            return {}

        monkeypatch.setattr(morph, "fetch_skeletons_on_demand_batch",
                            fake_fetch)

        def no_fafb_loader(body_ids, check_extrusions=None):
            raise AssertionError(
                "FAFB loader must not serve the NeuPrint expansion")

        monkeypatch.setattr(c, "_load_fafb_skeletons", no_fafb_loader)
        cache = morph.SkeletonVectorCache(
            "np:v1", project_root=str(tmp_path), raw_only=True)
        c._expand_top_type_rows(**self._expansion_kwargs(cache))
        assert fetch_calls == [("np:v1", [42, 43])]


class TestTypeRenderMemberCap:
    """Type-level render layers sample members per type up to
    TYPE_RENDER_MEMBER_CAP; the resolver tops the vector cache up from the
    full neuron-table type map (so a layer cannot show fewer members than
    the results report) and returns the uncapped total so the renderer can
    warn about truncated layers in-page."""

    def _setup(self, tmp_path, cached_ids):
        folder = morph._dataset_folder("test:v1")
        table_dir = tmp_path / "datasets" / folder
        table_dir.mkdir(parents=True)
        ids = list(range(101, 131))  # 30 neurons
        types = ["T1"] * 25 + ["T2"] * 5
        pd.DataFrame({
            "bodyId": ids,
            "type": types,
            "instance": [f"{t}_1" for t in types],
        }).to_parquet(table_dir / f"{folder}_allneurons_neuron_df.parquet")

        cache = morph.SkeletonVectorCache(
            "test:v1", project_root=str(tmp_path))
        vec = np.zeros(len(cache._feature_columns()), dtype=float)
        # the vector cache only ever learned a sparse subset of the types
        cache.append_vectors([(bid, vec, "skeleton") for bid in cached_ids])
        return morph.MorphologyComparer(
            query=101, dataset="test:v1", level="type",
            method="vector_v2", project_root=str(tmp_path), verbose=False)

    def test_sparse_cache_topped_up_from_type_map(self, tmp_path):
        """Regression: aMe10 rendered 1 neuron while type_summary.csv said
        n_bodyids=3 / coverage=1.0 — the vector cache held one row of the
        type and the neuron-table top-up only fired on zero members."""
        c = self._setup(tmp_path, cached_ids=[101, 126])
        members, total = c._type_members_from_cache("T1")
        # cache member first, then the type map completes the membership
        assert members[0] == 101
        assert total == 25
        assert len(members) == morph.TYPE_RENDER_MEMBER_CAP == 20
        assert set(members) <= set(range(101, 126))
        # a type within the cap resolves completely
        members_t2, total_t2 = c._type_members_from_cache("T2")
        assert total_t2 == 5
        assert members_t2 == [126] + list(range(127, 131))


class TestExtrusionCheck:
    """The shared FAFB extrusion check (fafb_utils): spike detection,
    cached batch results and the parallel/serial flag_extrusions path."""

    def _neuron(self, spike=False):
        """A small curved neuron, SWC-loaded like the healed-bundle input.

        A straight DataFrame-constructed TreeNeuron makes navis' tube
        orientation ill-defined; going through the same SWC reader the
        pipeline uses produces a well-formed tube mesh.
        """
        import io

        import navis

        n = 30
        t = np.arange(n) * 2.0
        coords = np.column_stack(
            [t, 5 * np.sin(t / 5), 5 * np.cos(t / 5)])
        if spike:
            coords[-1, 0] += 60.0  # single long jump = extrusion spike
        lines = ["# SWC skeleton"]
        for i in range(n):
            parent = -1 if i == 0 else i  # 1-based parent of node i+1
            lines.append(
                f"{i + 1} 1 {coords[i, 0]} {coords[i, 1]} "
                f"{coords[i, 2]} 0.25 {parent}"
            )
        nrn = navis.read_swc(io.StringIO("\n".join(lines)))
        nrn.units = "nm"
        return nrn

    def _smooth(self):
        return self._neuron(spike=False)

    def _spiky(self):
        return self._neuron(spike=True)

    def test_detect_extrusion_flags_spike_only(self):
        """A single long jump edge exceeds the 50,000 nm spike threshold;
        a smooth neuron stays unflagged."""
        from fafb_utils import detect_extrusion

        assert detect_extrusion(self._smooth()) is False
        assert detect_extrusion(self._spiky()) is True

    def test_flag_extrusions_batch_and_cache(self, tmp_path):
        """flag_extrusions returns flagged ids, persists the per-neuron
        results and serves repeat calls from the cache (parallel batch with
        a serial fallback)."""
        import fafb_utils

        root = str(tmp_path)
        skeletons = {1: self._smooth(), 2: self._spiky(), 3: self._smooth()}
        flagged = fafb_utils.flag_extrusions(
            root, "flywire_FAFB_v783", skeletons, verbose=False, n_workers=2)
        assert sorted(flagged) == [2]
        assert fafb_utils.extrusion_check_cache_path(
            root, "flywire_FAFB_v783").exists()
        # repeat call: served from the cache (same answer, no re-analysis)
        flagged2 = fafb_utils.flag_extrusions(
            root, "flywire_FAFB_v783", skeletons, verbose=False, n_workers=2)
        assert sorted(flagged2) == [2]
        # cached spikes are returned even for a single-neuron batch
        assert sorted(fafb_utils.flag_extrusions(
            root, "flywire_FAFB_v783", {2: self._spiky()},
            verbose=False, n_workers=1)) == [2]


class TestFafbBundleLocalSources:
    """FAFB v783 local-first behavior: the healed bundle feeds both the
    full skeleton download (skip already-available ids) and the full
    vector-cache build (skeleton vectors instead of mesh pickles)."""

    def _swc(self, n=30):
        t = np.arange(n) * 2.0
        coords = np.column_stack(
            [t, 5 * np.sin(t / 5), 5 * np.cos(t / 5)])
        lines = ["# SWC skeleton"]
        for i in range(n):
            parent = -1 if i == 0 else i
            lines.append(
                f"{i + 1} 1 {coords[i, 0]} {coords[i, 1]} "
                f"{coords[i, 2]} 0.25 {parent}"
            )
        return "\n".join(lines)

    def _write_bundle(self, tmp_path, entries):
        import zipfile
        zip_path = (tmp_path / "datasets" / "flywire_FAFB_v783"
                    / "sk_lod1_783_healed.zip")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w") as z:
            for bid in entries:
                z.writestr(f"{bid}.swc", self._swc())
        return zip_path

    def test_download_all_skeletons_disabled_for_fafb(self, tmp_path,
                                                      monkeypatch):
        """Download All Skeletons is disabled for FAFB: the
        call raises the explicit manual-download instruction (mirroring the
        file converter) instead of fetching meshes."""
        self._write_bundle(tmp_path, [2, 3])
        write_neuron_index(tmp_path, "flywire_FAFB_v783", [
            (1, "T", "T_1"), (2, "T", "T_2"), (3, "T", "T_3"),
        ])
        cache_dir = tmp_path / "cache" / "flywire_FAFB_v783"
        cache_dir.mkdir(parents=True)
        (cache_dir / "connections.parquet").touch()

        def forbidden_fetch(dataset, bid, project_root=None, persist=True,
                            **kwargs):
            raise AssertionError("FlyWire bulk download must not fetch")

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", forbidden_fetch)
        with pytest.raises(morph.FlyWireSkeletonAccessError) as excinfo:
            morph.download_all_skeletons(
                "flywire_FAFB_v783", project_root=str(tmp_path),
                max_workers=2, verbose=False,
            )
        message = str(excinfo.value)
        assert "disabled for FAFB" in message
        assert "sk_lod1_783_healed.zip" in message
        assert "https://codex.flywire.ai/api/download?dataset=fafb" in message
        assert "FAFB_file_converter" in message
        # nothing was fetched or persisted
        assert not (cache_dir / "meshes").exists()

    def test_build_uses_healed_bundle_for_fafb(self, tmp_path):
        """Full vector-cache build on FAFB vectorizes the healed bundle
        skeletons (rep=skeleton), not the mesh pickle cache."""
        self._write_bundle(tmp_path, [1, 2, 3])
        cache = morph.SkeletonVectorCache(
            "flywire_FAFB_v783", project_root=str(tmp_path),
            n_workers=1, verbose=False,
        )
        stats = cache.build()
        assert stats["new"] == 3 and stats["rows"] == 3
        data = cache.load()
        assert set(data["bodyIds"].tolist()) == {"1", "2", "3"}
        assert data["dataset_rep"] == "skeleton"
        assert (cache._load_meta() or {}).get("rep") == "skeleton"
        assert cache.coverage() == {"skeletons": 3, "vectors": 3}

    def test_build_replaces_legacy_mesh_cache_for_fafb(self, tmp_path):
        """A mesh-based (legacy, rep-less) FAFB vector cache is rebuilt
        from the healed bundle instead of being extended."""
        self._write_bundle(tmp_path, [1, 2])
        cache = morph.SkeletonVectorCache(
            "flywire_FAFB_v783", project_root=str(tmp_path),
            n_workers=1, verbose=False,
        )
        cache.morph_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"bodyId": [999], "cable_length": [1.0]}) \
            .to_parquet(cache.parquet_path, index=False)
        stats = cache.build()
        assert stats["rows"] == 2
        data = cache.load()
        assert set(data["bodyIds"].tolist()) == {"1", "2"}
        assert data["dataset_rep"] == "skeleton"

    def test_v2_build_fetch_missing_uses_loader_not_meshes(
            self, tmp_path, monkeypatch):
        """FAFB V2 build(fetch_missing>0) vectorizes skeletons from the local
        loader; the mesh-returning batch fetcher is never used."""
        cache = morph.find_similar_dataset_cache_v2(
            "flywire_FAFB_v783", project_root=str(tmp_path), verbose=False)
        cache.skeleton_dir.mkdir(parents=True, exist_ok=True)
        # neuron index drives the missing set
        write_neuron_index(tmp_path, "flywire_FAFB_v783", [
            (720575940000000001, "T", "T_1"),
        ])
        (tmp_path / "cache" / "flywire_FAFB_v783").mkdir(
            parents=True, exist_ok=True)
        (tmp_path / "cache" / "flywire_FAFB_v783"
         / "connections.parquet").touch()

        loader_calls = []

        def fake_loader(dataset, body_ids, project_root=None, log=None,
                        check_extrusions=False):
            loader_calls.append((dataset, list(body_ids), check_extrusions))
            return {int(b): line_neuron(length=20) for b in body_ids}

        def must_not_batch(*a, **k):
            raise AssertionError(
                "FAFB V2 build must not call the mesh batch fetcher")

        monkeypatch.setattr(morph, "_resolve_fafb_skeleton_trees", fake_loader)
        monkeypatch.setattr(morph, "fetch_skeletons_on_demand_batch",
                            must_not_batch)
        stats = cache.build(fetch_missing=5)
        assert loader_calls and loader_calls[0][0] == "flywire_FAFB_v783"
        assert loader_calls[0][2] is False   # build stays offline
        assert stats["rows"] >= 1
        data = cache.load()
        assert data is not None and data["dataset_rep"] == "skeleton"

    def test_cache_direct_query_fetch_uses_loader_for_fafb(
            self, tmp_path, monkeypatch):
        """A FAFB cache-direct search resolves query skeletons through the
        local loader, not the mesh batch fetcher."""
        # A local healed zip satisfies the readiness gate; the query and cache
        # plumbing are stubbed so the test isolates the fetch routing.
        self._write_bundle(tmp_path, [720575940000000001])
        cache = morph.find_similar_dataset_cache_v2(
            "flywire_FAFB_v783", project_root=str(tmp_path), verbose=False)
        comparer = morph.MorphologyComparer(
            query=720575940000000001, dataset="flywire_FAFB_v783",
            method="vector_v2", project_root=str(tmp_path), verbose=False,
            candidate_source="cache", expand_top_types=0)
        monkeypatch.setattr(morph, "find_similar_dataset_cache_v2",
                            lambda *a, **k: cache)
        monkeypatch.setattr(cache, "ensure", lambda **k: None)
        monkeypatch.setattr(cache, "load", lambda: None)
        monkeypatch.setattr(
            comparer, "_resolve_query",
            lambda: pd.DataFrame({"bodyId": ["720575940000000001"],
                                  "type": [""], "instance": [""]}))

        loader_calls = []

        def fake_loader(dataset, body_ids, project_root=None, log=None,
                        check_extrusions=False):
            loader_calls.append(list(body_ids))
            return {}

        def must_not_batch(*a, **k):
            raise AssertionError("FAFB query fetch must use the loader")

        monkeypatch.setattr(morph, "_resolve_fafb_skeleton_trees", fake_loader)
        monkeypatch.setattr(morph, "fetch_skeletons_on_demand_batch",
                            must_not_batch)
        try:
            comparer.find_similar()
        except Exception:
            # The empty cache cannot score, but the fetch routing must have
            # happened before any scoring failure.
            pass
        assert loader_calls, "loader was not consulted for the FAFB query"
        assert "720575940000000001" in [str(b) for b in loader_calls[0]]


class TestBatchFetchDispatch:
    """fetch_skeletons_on_demand_batch delegates to independent per-source
    fetchers; the shared dispatcher must not carry source-specific logic."""

    def test_banc_routes_to_banc_fetcher_not_others(self, tmp_path,
                                                     monkeypatch):
        calls = {}

        def fake_banc(dataset, missing, root, persist, requested, loaded,
                      progress_callback, cancel_event):
            calls["banc"] = (dataset, list(missing))
            return {bid: line_neuron() for bid in missing}

        def must_not(*a, **k):
            raise AssertionError("wrong source fetcher chosen")

        monkeypatch.setattr(morph, "_fetch_banc_skeleton_batch", fake_banc)
        monkeypatch.setattr(morph, "_fetch_neuprint_skeleton_batch", must_not)
        monkeypatch.setattr(morph, "_fetch_fafb_mesh_batch", must_not)
        # avoid touching the real raw cache
        monkeypatch.setattr(morph, "find_similar_raw_cache",
                            lambda *a, **k: None)
        out = morph.fetch_skeletons_on_demand_batch(
            "banc_v888", [1001, 1002], project_root=str(tmp_path))
        assert calls["banc"][0] == "banc_v888"
        assert [str(b) for b in calls["banc"][1]] == ["1001", "1002"]
        assert {str(b) for b in out} == {"1001", "1002"}

    def test_fafb_routes_to_mesh_fetcher(self, tmp_path, monkeypatch):
        calls = {}

        def fake_mesh(dataset, missing, root, persist, soma_positions,
                      cancel_event):
            calls["mesh"] = (dataset, list(missing))
            return {}

        def must_not(*a, **k):
            raise AssertionError("wrong source fetcher chosen")

        monkeypatch.setattr(morph, "_fetch_fafb_mesh_batch", fake_mesh)
        monkeypatch.setattr(morph, "_fetch_neuprint_skeleton_batch", must_not)
        monkeypatch.setattr(morph, "_fetch_banc_skeleton_batch", must_not)
        morph.fetch_skeletons_on_demand_batch(
            "flywire_FAFB_v783", [5], project_root=str(tmp_path))
        assert calls["mesh"][0] == "flywire_FAFB_v783"

    def test_neuprint_routes_to_neuprint_fetcher(self, tmp_path, monkeypatch):
        calls = {}

        def fake_np(dataset, missing, root, persist, requested, loaded,
                    raw_cache, client, batch_size, max_threads,
                    progress_callback, cancel_event, temp_normalized,
                    simplification):
            calls["np"] = (dataset, list(missing))
            return {bid: line_neuron() for bid in missing}

        def must_not(*a, **k):
            raise AssertionError("wrong source fetcher chosen")

        monkeypatch.setattr(morph, "_fetch_neuprint_skeleton_batch", fake_np)
        monkeypatch.setattr(morph, "_fetch_banc_skeleton_batch", must_not)
        monkeypatch.setattr(morph, "_fetch_fafb_mesh_batch", must_not)
        monkeypatch.setattr(morph, "find_similar_raw_cache",
                            lambda *a, **k: None)
        out = morph.fetch_skeletons_on_demand_batch(
            "np:v1", [7], project_root=str(tmp_path))
        assert calls["np"] == ("np:v1", [7])
        assert set(out) == {7}


class TestProfileFirst:
    """Connection-profile pipeline: connection-cache candidates ->
    top candidate_cap rows -> morphology."""

    def _setup(self, tmp_path, monkeypatch, query=101, **kwargs):
        write_skeleton(tmp_path, "np:v1", query, line_neuron(length=20))
        write_skeleton(tmp_path, "np:v1", 201, line_neuron(length=20))  # identical
        write_skeleton(tmp_path, "np:v1", 202, bushy_y_neuron())
        write_raw_find_similar_skeleton(tmp_path, "np:v1", query,
                                        line_neuron(length=20))
        write_raw_find_similar_skeleton(tmp_path, "np:v1", 201,
                                        line_neuron(length=20))
        write_raw_find_similar_skeleton(tmp_path, "np:v1", 202,
                                        bushy_y_neuron())

        params = dict(
            query=query, dataset="np:v1", level="bodyid", method="vector_v2", expand_top_types=0,
            candidate_source="profile",
            output_dir=str(tmp_path / "out"), project_root=str(tmp_path),
            verbose=False,
        )
        params.update(kwargs)
        comparer = morph.MorphologyComparer(**params)
        monkeypatch.setattr(comparer, "_resolve_query", lambda: pd.DataFrame({
            "bodyId": [query], "type": ["T"], "instance": ["T_1"],
        }))
        return comparer

    def _candidates(self, rows):
        """Build a fake _connection_cache_candidates result."""
        return pd.DataFrame({
            "target_bodyId": [r[0] for r in rows],
            "shared_count": [r[1] for r in rows],
            "profile_similarity": [r[2] for r in rows],
            "target_type": [r[3] for r in rows],
        })

    def _index(self, tmp_path, entries):
        write_neuron_index(tmp_path, "np:v1", entries)

    def test_pool_is_screened_candidates_and_persists_raw_fetch(
            self, tmp_path, monkeypatch):
        """The sorted screen list's top candidate_cap rows are the pool;
        missing members are fetched and written to the shared raw cache."""
        comparer = self._setup(tmp_path, monkeypatch,
                               cache_fetched_skeletons=False)
        self._index(tmp_path, [
            (101, "T", "T_1"), (201, "T", "T_2"),
            (202, "Y", "Y_1"), (204, "Z", "Z_1"),
        ])
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: self._candidates([
                                (201, 9, 1.0, "T"), (202, 5, 0.6, "Y"),
                                (204, 3, 0.3, "Z"),
                            ]))
        fetched = []

        def fake_fetch(dataset, bid, project_root=None, persist=True):
            fetched.append(bid)
            assert persist is True
            return line_neuron(length=25)

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", fake_fetch)
        res = comparer.find_similar()
        assert not res.empty
        # pool = T {101, 201} + Y {202} + Z {204}; only 204 is missing
        assert fetched == [204]
        raw_dir = (tmp_path / "cache" / morph._dataset_folder("np:v1")
                   / "skeletons" / "raw_skeletons")
        assert (raw_dir / "204.swc.zst").exists()
        # query neuron excluded from its own results
        assert 101 not in res["target_bodyId"].tolist()
        # identical morphology ranks first; all pool members are ranked
        assert res.iloc[0]["target_bodyId"] == 201
        assert {"profile_similarity", "target_type"} <= set(res.columns)
        assert set(res["target_bodyId"]) == {201, 202, 204}

    def test_raw_skeleton_persistence_is_always_enabled(self, tmp_path, monkeypatch):
        """The legacy cache_fetched_skeletons flag cannot disable raw SWC writes."""
        comparer = self._setup(tmp_path, monkeypatch,
                               cache_fetched_skeletons=False)
        self._index(tmp_path, [
            (101, "T", "T_1"), (201, "T", "T_2"),
            (202, "Y", "Y_1"), (204, "Z", "Z_1"),
        ])
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: self._candidates([
                                (201, 9, 1.0, "T"), (202, 5, 0.6, "Y"),
                                (204, 3, 0.3, "Z"),
                            ]))
        fetched = []

        def fake_fetch(dataset, bid, project_root=None, persist=True):
            fetched.append((bid, persist))
            return line_neuron(length=25)

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", fake_fetch)
        res = comparer.find_similar()
        assert not res.empty
        # pool = T {101, 201} + Y {202} + Z {204}; only 204 is missing, and
        # with the option on the fetch requests a permanent cache write.
        assert fetched == [(204, True)]
        # The compatibility flag is ignored by the shared raw-cache contract.
        default = self._setup(tmp_path, monkeypatch)
        assert default.cache_fetched_skeletons is True

    def test_candidate_cap_truncates_pool(self, tmp_path, monkeypatch):
        """Only the top candidate_cap rows of the sorted screen list
        enter the pool; lower-ranked candidates are never scored."""
        comparer = self._setup(tmp_path, monkeypatch, candidate_cap=2)
        self._index(tmp_path, [
            (101, "T", "T_1"), (201, "T", "T_2"),
            (202, "Y", "Y_1"), (204, "Z", "Z_1"), (205, "Z", "Z_2"),
        ])
        write_skeleton(tmp_path, "np:v1", 204, line_neuron(length=25))
        write_skeleton(tmp_path, "np:v1", 205, line_neuron(length=25))
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: self._candidates([
                                (201, 9, 1.0, "T"), (202, 5, 0.6, "Y"),
                                (204, 3, 0.3, "Z"), (205, 3, 0.3, "Z"),
                            ]))
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True:
                            line_neuron(length=25))
        res = comparer.find_similar()
        # cap 2 -> only the top-2 candidates (201, 202) are scored
        assert not res.empty
        assert set(res["target_bodyId"]) == {201, 202}
        assert "Z" not in res["target_type"].tolist()
        assert 204 not in res["target_bodyId"].tolist()

    def test_profile_first_bodyid_carries_intra_columns(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch)
        self._index(tmp_path, [
            (101, "T", "T_1"), (201, "T", "T_2"), (202, "Y", "Y_1"),
        ])
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: self._candidates([
                                (201, 9, 1.0, "T"), (202, 5, 0.6, "Y"),
                            ]))
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True:
                            line_neuron(length=25))
        res = comparer.find_similar()
        assert {"is_same_type", "intra_type_similarity"}.issubset(res.columns)
        # 201 is the same type (T) and identical morphology -> top hit
        assert res.iloc[0]["target_bodyId"] == 201
        assert res.iloc[0]["is_same_type"] == True
        # T has 2 identical members -> intra-type similarity ~1
        assert res.iloc[0]["intra_type_similarity"] == pytest.approx(1.0, abs=1e-6)
        assert res.iloc[1]["is_same_type"] == False

    def test_profile_first_type_level_includes_intra_reference(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch)
        self._index(tmp_path, [
            (101, "T", "T_1"), (201, "T", "T_2"), (202, "Y", "Y_1"),
        ])
        comparer.level = "type"
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: self._candidates([
                                (201, 9, 1.0, "T"), (202, 5, 0.6, "Y"),
                            ]))
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True:
                            line_neuron(length=25))
        res = comparer.find_similar()
        assert not res.empty
        intra = res[res["is_intra_type"]]
        assert len(intra) == 1
        assert intra.iloc[0]["target_type"] == "T"
        assert intra.iloc[0]["rank"] == 1
        assert intra.iloc[0]["similarity"] == pytest.approx(1.0, abs=1e-6)
        assert intra.iloc[0]["intra_type_similarity"] == pytest.approx(1.0, abs=1e-6)
        assert intra.iloc[0]["n_bodyids"] == 2
        inter = res[~res["is_intra_type"]]
        assert inter.iloc[0]["target_type"] == "Y"

    def test_type_level_works_without_vector_cache(self, tmp_path, monkeypatch):
        """Type-level profile-first works with skeletons + neuron index but
        NO vector cache (male-cns v1.0 situation)."""
        comparer = self._setup(tmp_path, monkeypatch)
        self._index(tmp_path, [
            (101, "T", "T_1"), (201, "T", "T_2"), (202, "Y", "Y_1"),
        ])
        comparer.level = "type"
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: self._candidates([
                                (201, 9, 1.0, "T"), (202, 5, 0.6, "Y"),
                            ]))
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True:
                            line_neuron(length=25))
        res = comparer.find_similar()
        assert not res.empty
        assert {"target_type", "is_intra_type", "intra_type_similarity"}.issubset(res.columns)
        intra = res[res["is_intra_type"] == True]  # noqa: E712
        assert len(intra) == 1
        assert intra.iloc[0]["target_type"] == "T"
        assert intra.iloc[0]["intra_type_similarity"] == pytest.approx(1.0, abs=1e-6)
        assert (res["target_type"] == "Y").any()

    def test_type_query_exports_all_members_and_intra_pairs(
        self, tmp_path, monkeypatch
    ):
        """A type query must not collapse to query_ids[0] in results.csv.

        The body-level export contains unique intra-type pairs for every
        resolved query member, while the type summary counts the type
        population rather than the pre-fetch cache snapshot.
        """
        comparer = self._setup(tmp_path, monkeypatch)
        # Make the two query members morphologically distinct so a
        # first-bodyId-only implementation cannot pass the type-mean check.
        write_skeleton(tmp_path, "np:v1", 102, bushy_y_neuron())
        self._index(tmp_path, [
            (101, "Q", "Q_1"), (102, "Q", "Q_2"),
            (201, "T", "T_1"), (202, "Y", "Y_1"),
        ])
        comparer.level = "type"
        monkeypatch.setattr(comparer, "_resolve_query", lambda: pd.DataFrame({
            "bodyId": [101, 102], "type": ["Q", "Q"],
            "instance": ["Q_1", "Q_2"],
        }))
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: self._candidates([
                                (201, 9, 1.0, "T"), (202, 5, 0.6, "Y"),
                            ]))
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True:
                            line_neuron(length=25))

        res = comparer.find_similar()
        assert not res.empty
        body = pd.read_csv(Path(comparer.output_folder) / "results.csv")
        endpoints = set(body["source_bodyId"]) | set(body["target_bodyId"])
        assert {101, 102}.issubset(endpoints)
        intra = body[
            (body["target_type"] == "Q")
            & (body["is_same_type"] == True)  # noqa: E712
        ]
        assert set(zip(intra["source_bodyId"], intra["target_bodyId"])) == {
            (101, 102)
        }
        assert intra["intra_type_similarity"].iloc[0] == pytest.approx(
            intra["similarity"].mean(),
            abs=1e-6,
        )

        summary = pd.read_csv(Path(comparer.output_folder) / "type_summary.csv")
        row = summary[summary["target_type"] == "Q"].iloc[0]
        assert row["n_bodyids"] == 2
        assert row["is_intra_type"] == True  # noqa: E712
        target_rows = body[body["target_type"] == "T"]
        assert set(target_rows["source_bodyId"]) == {101, 102}
        target_row = summary[summary["target_type"] == "T"].iloc[0]
        assert target_row["similarity"] == pytest.approx(
            target_rows["similarity"].mean(), abs=1e-6
        )

    def test_type_search_uses_only_connectivity_candidates(self, tmp_path, monkeypatch):
        """Regression: a cached-only type must NOT rank; only the screened
        candidate rows enter the pool (no type -> member expansion)."""
        comparer = self._setup(tmp_path, monkeypatch)
        # 103 (Y) is cached but Y is NOT among the connectivity candidates
        write_skeleton(tmp_path, "np:v1", 103, y_neuron())
        self._index(tmp_path, [
            (101, "T", "T_1"), (201, "T", "T_2"),
            (202, "Y", "Y_1"), (103, "Y", "Y_2"),
        ])
        comparer.level = "type"
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: self._candidates([(201, 9, 1.0, "T")]))
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True:
                            line_neuron(length=25))
        res = comparer.find_similar()
        assert not res.empty
        assert not (res["target_type"] == "Y").any(), \
            "cached-only type leaked into the connectivity-only pool"
        # a Y member entering as its own candidate row joins the pool; other
        # Y members without candidate rows stay out (no type expansion)
        comparer2 = self._setup(tmp_path, monkeypatch)
        comparer2.level = "type"
        monkeypatch.setattr(comparer2, "_connection_cache_candidates",
                            lambda q: self._candidates([
                                (201, 9, 1.0, "T"), (103, 4, 0.4, "Y"),
                            ]))
        res2 = comparer2.find_similar()
        assert (res2["target_type"] == "Y").any(), \
            "candidate row for Y missing from the pool"
        body = pd.read_csv(Path(comparer2.output_folder) / "results.csv")
        assert 103 in body["target_bodyId"].tolist()
        assert 202 not in body["target_bodyId"].tolist(), \
            "non-candidate member leaked into the pool"

    def test_pool_standardization_without_vector_cache(self, tmp_path, monkeypatch):
        """Without a vector cache the pool vectors are z-scored with
        pool-computed stats; scores match the manual z-scored cosine."""
        comparer = self._setup(tmp_path, monkeypatch)
        # Isolate the standardization semantics: no ROI-expansion block.
        comparer._v2_weights["roi"] = 0.0
        self._index(tmp_path, [
            (101, "T", "T_1"), (201, "T", "T_2"), (202, "Y", "Y_1"),
        ])
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: self._candidates([
                                (201, 9, 1.0, "T"), (202, 5, 0.6, "Y"),
                            ]))
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True:
                            line_neuron(length=25))
        res = comparer.find_similar()
        assert not res.empty

        # manual: raw V2 vectors for query + pool, z-scored with pool stats,
        # whitened with the cache's matrix, scored with the same
        # block-weighted V2 similarity incl. the spatial-overlap blend.
        cache = morph.find_similar_dataset_cache_v2(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        data = cache.load()
        rows_by_id = {int(b): i for i, b in enumerate(data["bodyIds"])}
        ids = [101, 201, 202]
        Xr = data["raw"][[rows_by_id[i] for i in ids]]
        mu = Xr.mean(axis=0)
        sd = Xr.std(axis=0)
        sd[sd <= 0] = 1.0
        Xw = morph.apply_whitening(data["whiten"], (Xr - mu) / sd)
        hist = Xr[:, morph.SPATIAL_HIST_SLICE[0]:morph.SPATIAL_HIST_SLICE[1]]
        hist = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1e-9)
        qz = Xw[0]
        for _, row in res.iterrows():
            idx = ids.index(int(row["target_bodyId"]))
            so = {"members": hist[[0]], "centroid": hist[0],
                  "pool": hist[[idx]]}
            expected, _ = morph.v2_similarity_matrix(
                qz, Xw[idx].reshape(1, -1), dict(morph.DEFAULT_V2_BLOCK_WEIGHTS),
                spatial_overlap=so, query_index=0)
            assert row["similarity"] == pytest.approx(expected[0], abs=1e-9),                 row["target_bodyId"]

    def test_profile_first_and_warm_cache_use_the_same_cosine_space(
            self, tmp_path, monkeypatch):
        """A batch fetch must not make the first run mix raw and cached-X rows.

        The fake batch helper mirrors production: it appends raw vectors before
        returning the in-memory skeletons.  The first and warmed-cache runs
        should therefore produce identical cosine scores.
        """
        self._index(tmp_path, [
            (101, "T", "T_1"), (201, "T", "T_2"),
            (202, "Y", "Y_1"), (204, "Z", "Z_1"),
        ])
        calls = []

        def fake_batch(dataset, body_ids, project_root=None, persist=True,
                       level="raw", **kwargs):
            ids = [int(b) for b in body_ids]
            calls.append(ids)
            neurons = {}
            records = []
            for bid in ids:
                neuron = line_neuron(length=25)
                neuron.id = bid
                neurons[bid] = neuron
                _, vector = morph.vectorize_neuron(neuron)
                records.append((bid, vector, "skeleton"))
            morph.SkeletonVectorCache(
                dataset, project_root=project_root, verbose=False
            ).append_vectors(records, vector_basis=morph.VECTOR_BASIS_RAW)
            return neurons

        monkeypatch.setattr(morph, "fetch_skeletons_on_demand_batch", fake_batch)

        def make():
            comparer = self._setup(
                tmp_path, monkeypatch, cache_fetched_skeletons=False,
                metric="cosine",
            )
            monkeypatch.setattr(
                comparer, "_connection_cache_candidates",
                lambda q: self._candidates([
                    (201, 9, 1.0, "T"), (204, 5, 0.6, "Z"),
                ]),
            )
            return comparer

        first = make().find_similar()
        warmed = make().find_similar()
        assert calls == [[204]]
        first_scores = first.set_index("target_bodyId")["similarity"].sort_index()
        warm_scores = warmed.set_index("target_bodyId")["similarity"].sort_index()
        np.testing.assert_allclose(
            first_scores.to_numpy(), warm_scores.to_numpy(),
            rtol=1e-12, atol=1e-12,
        )

    def test_download_all_skeletons_resumable(self, tmp_path, monkeypatch):
        """download_all_skeletons: persist=True, skips existing, respects
        limit and the cancel event."""
        write_neuron_index(tmp_path, "np:v1", [
            (101, "T", "T_1"), (201, "T", "T_2"), (202, "Y", "Y_1"),
        ])
        # The connection cache marker makes the index authoritative for
        # skeleton workflows (a shipped seed alone must not authorize bulk
        # fetches).
        cache_dir = tmp_path / "cache" / "np_v1"
        cache_dir.mkdir(parents=True)
        (cache_dir / "connections.parquet").touch()
        fetched = []
        fetch_levels = []

        def fake_fetch(dataset, bid, project_root=None, persist=True, **kwargs):
            assert persist is True
            fetched.append(bid)
            fetch_levels.append(kwargs.get("level"))
            neuron = line_neuron(length=25)
            neuron.id = int(bid)
            return neuron

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", fake_fetch)
        progress = []

        summary = morph.download_all_skeletons(
            "np:v1", project_root=str(tmp_path), max_workers=2, limit=2,
            progress_callback=lambda c, t, i: progress.append((c, t)),
            verbose=False,
        )
        assert summary["fetched"] == 2 and summary["errors"] == 0
        assert set(fetched) == {101, 201}  # deterministic first two of the index
        assert fetch_levels == [morph.VECTOR_BASIS_RAW] * 2
        raw_dir = cache_dir / "skeletons" / "raw_skeletons"
        assert sorted(path.name for path in raw_dir.glob("*.swc.zst")) == [
            "101.swc.zst", "201.swc.zst",
        ]
        assert not list((cache_dir / "skeletons").glob("*.pkl"))
        assert not (cache_dir / "skeletons" / ".level").exists()
        assert progress and progress[-1][0] == progress[-1][1] == 2
        # resume: the two fetched are now cached; limit=2 fetches the rest
        summary2 = morph.download_all_skeletons(
            "np:v1", project_root=str(tmp_path), max_workers=2, limit=2,
            verbose=False,
        )
        assert summary2["fetched"] == 1 and 202 in fetched
        assert summary2["skipped_existing"] == 2

    def test_fast_pull_uses_raw_swc_cache_not_legacy_pickle(
            self, tmp_path, monkeypatch):
        """The legacy fast label cannot select a pull-time pickle cache."""
        write_neuron_index(tmp_path, "np:v1", [(101, "T", "T_1")])
        cache_dir = tmp_path / "cache" / "np_v1"
        cache_dir.mkdir(parents=True)
        (cache_dir / "connections.parquet").touch()
        skeleton_dir = cache_dir / "skeletons"
        skeleton_dir.mkdir()
        with open(skeleton_dir / "101.pkl", "wb") as f:
            pickle.dump(line_neuron(length=10), f)

        fetched = []

        def fake_fetch(dataset, bid, project_root=None, persist=True, **kwargs):
            fetched.append(bid)
            neuron = line_neuron(length=25)
            neuron.id = int(bid)
            return neuron

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", fake_fetch)
        summary = morph.download_all_skeletons(
            "np:v1", project_root=str(tmp_path), mode="fast", verbose=False
        )

        assert summary["fetched"] == 1
        assert fetched == [101]
        assert summary["mode"] == "raw"
        assert (skeleton_dir / "raw_skeletons" / "101.swc.zst").exists()
        assert not (skeleton_dir / ".level").exists()
        # The ambiguous legacy file is left recoverable but is never used as
        # the new raw pull artifact.
        assert (skeleton_dir / "101.pkl").exists()

    def test_download_all_skeletons_cancel(self, tmp_path, monkeypatch):
        import threading
        write_neuron_index(tmp_path, "np:v1", [
            (101, "T", "T_1"), (201, "T", "T_2"), (202, "Y", "Y_1"),
        ])
        cache_dir = tmp_path / "cache" / "np_v1"
        cache_dir.mkdir(parents=True)
        (cache_dir / "connections.parquet").touch()
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True:
                            line_neuron(length=25))
        ev = threading.Event()
        ev.set()  # pre-cancelled
        summary = morph.download_all_skeletons(
            "np:v1", project_root=str(tmp_path), max_workers=2,
            cancel_event=ev, verbose=False,
        )
        assert summary["cancelled"] is True
        assert summary["fetched"] == 0

    def test_find_homologs_fast_bodyid_reaches_comparison(self, tmp_path, monkeypatch):
        """Regression: the bodyId branch of find_homologs_fast was nested
        inside ``if not is_bodyid:`` (dead code), so bodyId queries returned
        None and profile-first searches always fell back to the vector
        cache. The branch must run and return candidates."""
        from comparison.profile_comparator import HomologFinder
        from comparison.connectivity_profiler import ConnectivityStatus

        class FakeProfile:
            neuron_type = "T"
            upstream_partners = {"A": 10.0, "B": 20.0}
            downstream_partners = {"C": 15.0}
            connectivity_status = ConnectivityStatus.COMPLETE

        conn = pd.DataFrame({
            "bodyId_pre": [1, 2, 3],
            "bodyId_post": [2, 1, 2],
            "weight": [5, 5, 5],
            "type_pre": ["A", "C", "A"],
            "type_post": ["C", "A", "C"],
        })
        finder = HomologFinder(
            source="1", source_dataset="d:v1", target_dataset="d:v1",
            output_dir=str(tmp_path), top_n=5, verbose=False,
            min_shared_partners=1, vector_prune_fraction=1.0,
            vector_prefiltering=False, min_synapse_threshold=1,
            include_untyped_partners=True, use_cache=True,
            ensure_cache_complete=False,
        )
        monkeypatch.setattr(finder, "_load_connection_cache",
                            lambda ds, auto_build=False: conn)
        monkeypatch.setattr(finder, "_prewarm_profile_cache", lambda ds: None)
        monkeypatch.setattr(finder.profiler, "get_profile",
                            lambda bid, ds: FakeProfile())
        monkeypatch.setattr(finder.profiler, "consolidate_profile_cache",
                            lambda ds: None)
        monkeypatch.setattr(finder.profiler, "_get_cached_conn_df",
                            lambda ds: conn)
        monkeypatch.setattr(finder.profiler, "flush_pending_cache_writes",
                            lambda silent=False: None)
        monkeypatch.setattr(finder, "_build_profiles_batch",
                            lambda ids, ds, **kw: {int(b): FakeProfile() for b in ids})
        reached = {}

        def fake_core(**kw):
            reached["source_bodyids"] = kw.get("source_bodyids")
            res = pd.DataFrame({
                "source_bodyId": [1], "target_bodyId": [2],
                "target_type": ["T"], "rank_union": [0.9],
            })
            return (res, pd.DataFrame(), {"none": []}, {"rare_or_uni": []},
                    {}, {}, {})

        monkeypatch.setattr(finder, "_compare_candidates_core", fake_core)
        monkeypatch.setattr(finder, "_save_homolog_results_internal",
                            lambda **kw: None)
        res = finder.find_homologs_fast()
        assert res is not None and len(res) == 1
        assert reached["source_bodyids"] == [1]


# ---------------------------------------------------------------------------
# Homolog enrichment
# ---------------------------------------------------------------------------

class TestConnectionCacheCandidates:
    """Candidate discovery straight from the connections parquet."""

    def _setup(self, tmp_path, monkeypatch, with_roi=True):
        folder = tmp_path / "cache" / morph._dataset_folder("np:v1")
        folder.mkdir(parents=True, exist_ok=True)
        rows = {
            "bodyId_pre": ["2", "3", "4", "5", "1", "1", "2", "3", "3", "6"],
            "bodyId_post": ["1", "1", "1", "1", "2", "3", "6", "6", "7", "2"],
            "weight": [10, 8, 7, 2, 10, 5, 9, 4, 6, 9],
        }
        if with_roi:
            rows["roi"] = ["AL(L)", "AL(R)", "AL(R)", "", "AL(L)", "AL(L)",
                           "AL(L)", "AL(L)", "AL(R)", "AL(L)"]
        rows["cached_date"] = ["x"] * len(rows["bodyId_pre"])
        pd.DataFrame(rows).to_parquet(folder / "connections.parquet", index=False)
        write_neuron_index(tmp_path, "np:v1", [
            (1, "Q", "Q_1"), (2, "A", "A_1"), (3, "B", "B_1"),
            (4, "C", "C_1"), (5, "D", "D_1"), (6, "E", "E_1"), (7, "F", "F_1"),
        ])
        comparer = morph.MorphologyComparer(query=1, dataset="np:v1",
                                            project_root=str(tmp_path), verbose=False)
        monkeypatch.setattr(comparer, "_resolve_query", lambda: pd.DataFrame({
            "bodyId": [1], "type": ["Q"], "instance": ["Q_1"],
        }))
        return comparer

    def test_shared_partner_ranking_and_type_join(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch)
        qdf = comparer._resolve_query()
        res = comparer._connection_cache_candidates(
            qdf, min_weight=3, min_shared_partners=2
        )
        # query 1 partners (w>=3): up {2,3,4}, down {2,3}; 5 excluded (w=2).
        # candidate 6 shares upstream {2,3} + downstream {2} = 3; candidate 7
        # shares upstream {3} = 1 -> below min_shared_partners.
        assert res["target_bodyId"].tolist() == [6]
        assert res["shared_count"].tolist() == [3]
        assert res["profile_similarity"].tolist() == [1.0]
        assert res["target_type"].tolist() == ["E"]

    def test_roi_filter_restricts_candidates(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch)
        qdf = comparer._resolve_query()
        # AL(R) rows only: 3->1, 4->1, 3->7. Query partners {3,4}; shared
        # upstream for 7 = {3} -> 1 shared -> below the minimum.
        res = comparer._connection_cache_candidates(
            qdf, min_weight=3, min_shared_partners=2, roi_filter=["AL(R)"]
        )
        assert res.empty

    def test_dataset_without_roi_column_falls_back_to_all_rows(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch, with_roi=False)
        qdf = comparer._resolve_query()
        res = comparer._connection_cache_candidates(
            qdf, min_weight=3, min_shared_partners=2
        )
        assert res["target_bodyId"].tolist() == [6]
        assert res["shared_count"].tolist() == [3]

    def test_missing_connection_cache_returns_empty(self, tmp_path, monkeypatch):
        comparer = morph.MorphologyComparer(query=1, dataset="np:v1",
                                            project_root=str(tmp_path), verbose=False)
        monkeypatch.setattr(comparer, "_resolve_query", lambda: pd.DataFrame({
            "bodyId": [1], "type": ["Q"], "instance": ["Q_1"],
        }))
        assert comparer._connection_cache_candidates(comparer._resolve_query()).empty


class TestEnrichment:
    def test_enrichment_adds_columns_with_valid_range(self, tmp_path):
        write_skeleton(tmp_path, "src:v1", 101, line_neuron())
        write_skeleton(tmp_path, "src:v1", 102, translated(line_neuron()))
        write_skeleton(tmp_path, "tgt:v1", 201, line_neuron())
        write_skeleton(tmp_path, "tgt:v1", 202, y_neuron())
        results = pd.DataFrame({
            "source_bodyId": [101, 102, 101],
            "target_bodyId": [201, 202, 999],
        })
        enriched = morph.enrich_homolog_results(
            results, "src:v1", "tgt:v1", project_root=str(tmp_path), verbose=False
        )
        assert {"morph_v2_similarity",
                "morph_nblast"}.issubset(enriched.columns)
        assert "morph_cosine" not in enriched.columns
        assert "morph_pearson" not in enriched.columns
        assert "morph_v2_cosine" not in enriched.columns
        assert "morph_v2_pearson" not in enriched.columns
        # 101 vs 201 = identical morphology -> production v2 score ~1
        assert enriched.loc[0, "morph_v2_similarity"] > 0.9
        # missing skeleton -> NaN, row preserved
        assert np.isnan(enriched.loc[2, "morph_v2_similarity"])
        assert len(enriched) == len(results)
        # nblast finite for available skeletons (normalized forward score)
        assert np.isfinite(enriched.loc[0, "morph_nblast"])

    def test_enrichment_skips_without_required_columns(self, tmp_path):
        results = pd.DataFrame({"a": [1], "b": [2]})
        out = morph.enrich_homolog_results(results, "x", "y", project_root=str(tmp_path))
        assert out.equals(results)

    def test_enrichment_empty_df(self, tmp_path):
        out = morph.enrich_homolog_results(pd.DataFrame(), "x", "y", project_root=str(tmp_path))
        assert out.empty


# ---------------------------------------------------------------------------
# Step-progress events ([DROCAT][progress] protocol for the web UI)
# ---------------------------------------------------------------------------

class TestStepProgress:
    """Structured step-progress events emitted by the similarity pipeline."""

    def test_progress_event_format_and_verbose_gating(self, capsys):
        """The event line is '[DROCAT][progress] <step>/<total> <label>' and
        is only emitted when verbose is enabled."""
        c = morph.MorphologyComparer(query=1, dataset="np:v1",
                                     project_root="/tmp", verbose=True)
        c._progress(2, 6, "Discovering candidates (connection cache)")
        out = capsys.readouterr().out
        assert "[DROCAT][progress] 2/6 Discovering candidates (connection cache)" in out

        # Labels are optional; trailing whitespace is stripped.
        c._progress(6, 6, "")
        assert "[DROCAT][progress] 6/6" in capsys.readouterr().out

        # Silent when verbose is off (nothing leaks into quiet runs).
        c2 = morph.MorphologyComparer(query=1, dataset="np:v1",
                                      project_root="/tmp", verbose=False)
        c2._progress(1, 4, "Loading vector cache")
        assert capsys.readouterr().out == ""

    def test_find_similar_emits_ordered_step_events(self, tmp_path, monkeypatch, capsys):
        """A full profile-first run reports steps 1/6..6/6 in pipeline order."""
        write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        write_skeleton(tmp_path, "np:v1", 201, line_neuron(length=20))
        write_skeleton(tmp_path, "np:v1", 202, bushy_y_neuron())
        write_neuron_index(tmp_path, "np:v1", [
            (101, "T", "T_1"), (201, "T", "T_2"), (202, "Y", "Y_1"),
        ])

        comparer = morph.MorphologyComparer(
            query=101, dataset="np:v1", level="bodyid", method="vector_v2", expand_top_types=0,
            candidate_source="profile",
            output_dir=str(tmp_path / "out"), project_root=str(tmp_path),
            verbose=True,
        )
        monkeypatch.setattr(comparer, "_resolve_query", lambda: pd.DataFrame({
            "bodyId": [101], "type": ["T"], "instance": ["T_1"],
        }))
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: pd.DataFrame({
                                "target_bodyId": [201, 202],
                                "shared_count": [9, 5],
                                "profile_similarity": [1.0, 0.6],
                                "target_type": ["T", "Y"],
                            }))

        res = comparer.find_similar()
        assert not res.empty

        out = capsys.readouterr().out
        assert "Step 2/6 — Discovering candidates" in out
        assert "Step 3/6 — Selecting the scoring pool" in out
        assert "Step 4/6 — Loading & vectorizing skeletons" in out
        assert "Profile-first:" in out and "pool neurons" in out
        events = [ln.strip() for ln in out.splitlines()
                  if "[DROCAT][progress]" in ln]
        # visualize_top_n defaults to 0 here, so the final label reflects
        # the visualization-free save phase.
        assert events == [
            "[DROCAT][progress] 1/6 Resolving query neuron",
            "[DROCAT][progress] 2/6 Discovering candidates (connection cache)",
            "[DROCAT][progress] 3/6 Selecting top 2 candidates for scoring",
            "[DROCAT][progress] 4/6 Loading & vectorizing skeletons",
            "[DROCAT][progress] 5/6 Scoring similarity (vector: shape + spatial)",
            "[DROCAT][progress] 6/6 Saving results",
        ]

        # The 'Results saved to:' marker must carry the path ONLY (the UI
        # splits after the marker and checks isdir to find the run folder
        # for streaming output files).
        marker = [ln.strip() for ln in out.splitlines()
                  if "Results saved to:" in ln]
        assert marker, "expected a 'Results saved to:' marker line"
        run_dir = marker[-1].split("Results saved to:", 1)[1].strip()
        assert os.path.isdir(run_dir)
        assert run_dir == comparer.output_folder

    def test_homolog_finder_progress_event(self, capsys):
        """HomologFinder emits the same structured event protocol (plain print
        outside a tqdm context, tqdm.write inside one)."""
        from comparison.profile_comparator import HomologFinder
        finder = HomologFinder(source="a", source_dataset="d:v1",
                               target_dataset="d:v1", verbose=True)
        finder._progress(2, 6, "Building source profiles")
        assert ("[DROCAT][progress] 2/6 Building source profiles"
                in capsys.readouterr().out)

        # Inside a progress bar the event goes through tqdm.write (the UI
        # consumes it the same way regardless of the transport).
        finder._in_progress_bar = True
        finder._progress(4, 6, "Building target profiles")
        assert ("[DROCAT][progress] 4/6 Building target profiles"
                in capsys.readouterr().out)
        finder._in_progress_bar = False

        # Silent when verbose is off.
        quiet = HomologFinder(source="a", source_dataset="d:v1",
                              target_dataset="d:v1", verbose=False)
        quiet._progress(1, 6, "Loading connection data")
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Population standardization statistics
# ---------------------------------------------------------------------------

class TestPopulationStats:
    """population_stats(): stable standardization stats for datasets without
    a vector cache, with version-sibling borrowing for sparse caches."""

    def _patch_sequential(self, monkeypatch):
        """Fork-based parallel vectorization is not deterministic under
        pytest; these tests exercise the statistics logic, not the worker
        pool, so run the vectorization sequentially."""
        monkeypatch.setattr(
            morph.SkeletonVectorCache, "_vectorize_parallel",
            lambda self, files: [_vectorize_one_file(p) for p in files],
        )

    def test_persists_and_reuses(self, tmp_path, monkeypatch):
        self._patch_sequential(monkeypatch)
        write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        write_skeleton(tmp_path, "np:v1", 201, bushy_y_neuron())
        write_skeleton(tmp_path, "np:v1", 202, y_neuron())
        mu, sd = morph.population_stats("np:v1", str(tmp_path))
        assert mu.shape == (morph.VECTOR_DIM,)
        assert sd.shape == (morph.VECTOR_DIM,)
        assert (sd > 0).all()
        stats_file = tmp_path / "cache" / "np_v1" / "morphology" / "population_stats.json"
        assert stats_file.exists()
        # second call loads the persisted file (same arrays)
        mu2, sd2 = morph.population_stats("np:v1", str(tmp_path))
        np.testing.assert_array_equal(mu2, mu)
        np.testing.assert_array_equal(sd2, sd)

    def test_none_without_skeletons(self, tmp_path):
        mu, sd = morph.population_stats("empty:v1", str(tmp_path))
        assert mu is None and sd is None

    def test_borrows_version_sibling_stats(self, tmp_path, monkeypatch):
        """A dataset with too few cached skeletons samples its stats from
        the version sibling's cache (shared reconstruction, e.g. male-cns
        v1.0 <- v0.9), so both datasets share the same statistics."""
        self._patch_sequential(monkeypatch)
        write_skeleton(tmp_path, "mc:v0.9", 1, line_neuron(length=20))
        write_skeleton(tmp_path, "mc:v0.9", 2, bushy_y_neuron())
        write_neuron_index(tmp_path, "mc:v0.9", [(1, "A", "A1"), (2, "B", "B1")])
        write_skeleton(tmp_path, "mc:v1.0", 1, line_neuron(length=20))
        write_neuron_index(tmp_path, "mc:v1.0", [(1, "A", "A1"), (3, "C", "C1")])
        # 50% of the smaller index is shared -> valid sibling

        mu09, sd09 = morph.population_stats("mc:v0.9", str(tmp_path))
        assert mu09 is not None
        mu10, sd10 = morph.population_stats("mc:v1.0", str(tmp_path))
        assert mu10 is not None
        # v1.0 samples the sibling's (larger) cache: identical stats
        np.testing.assert_array_equal(mu10, mu09)
        np.testing.assert_array_equal(sd10, sd09)
        stats_file = (tmp_path / "cache" / "mc_v1_0" / "morphology"
                      / "population_stats.json")
        assert stats_file.exists()

    def test_does_not_borrow_from_unrelated_dataset(self, tmp_path, monkeypatch):
        """Same name prefix but no shared population -> no borrowing."""
        self._patch_sequential(monkeypatch)
        write_skeleton(tmp_path, "mc:v0.9", 1, line_neuron(length=20))
        write_neuron_index(tmp_path, "mc:v0.9", [(1, "A", "A1")])
        # unrelated sibling: no overlapping bodyIds
        write_neuron_index(tmp_path, "mc:v1.0", [(999, "X", "X1")])
        morph.population_stats("mc:v0.9", str(tmp_path))
        mu, sd = morph.population_stats("mc:v1.0", str(tmp_path))
        assert mu is None and sd is None


# ---------------------------------------------------------------------------
# Representation consistency (skeleton vs mesh; one level per comparison)
# ---------------------------------------------------------------------------

class TestRepresentationConsistency:
    """Skeletons and meshes (or two simplification levels) are never mixed
    within one vector cache or one comparison."""

    @staticmethod
    def _write_mesh(tmp_path, dataset, body_id):
        folder = tmp_path / "cache" / morph._dataset_folder(dataset) / "skeletons"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{body_id}.pkl"
        with open(path, "wb") as f:
            pickle.dump(cube_mesh(), f)
        return path

    def test_vectorize_one_file_reports_rep(self, tmp_path):
        p = write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        bid, morph_vals, pv_vals, rep = morph._vectorize_one_file(str(p))
        assert (bid, rep) == (101, "skeleton")
        assert len(morph_vals) == 24 and len(pv_vals) == 100
        mp = self._write_mesh(tmp_path, "np:v1", 901)
        assert morph._vectorize_one_file(str(mp))[3] == "mesh"

    def test_build_skips_foreign_representation(self, tmp_path):
        """A cache holds ONE representation: a mesh pickle beside skeleton
        pickles is skipped at build time."""
        write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        write_skeleton(tmp_path, "np:v1", 201, bushy_y_neuron())
        self._write_mesh(tmp_path, "np:v1", 901)
        cache = morph.SkeletonVectorCache(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        stats = cache.build()
        assert stats["rows"] == 2  # the mesh row never entered the cache
        data = cache.load()
        assert data["dataset_rep"] == "skeleton"
        assert set(data["rep"]) == {"skeleton"}

    def test_vectors_for_skips_foreign_rep_files(self, tmp_path):
        write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        cache = morph.SkeletonVectorCache(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        cache.build()
        # a mesh pickle added after the build is not vectorized into the
        # comparison (different representation than the cache's)
        self._write_mesh(tmp_path, "np:v1", 901)
        X, mask, reps = cache.vectors_for([101, 901], compute_missing=True)
        assert mask.tolist() == [True, False]
        assert reps == ["skeleton", ""]

    def test_profile_first_drops_foreign_rep_pool(self, tmp_path, monkeypatch):
        """A pool neuron cached as a mesh is unscorable against a skeleton
        query: representations are never mixed within one comparison."""
        write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        write_skeleton(tmp_path, "np:v1", 201, line_neuron(length=20))
        self._write_mesh(tmp_path, "np:v1", 301)
        write_neuron_index(tmp_path, "np:v1", [
            (101, "T", "T_1"), (201, "T", "T_2"), (301, "M", "M_1"),
        ])
        comparer = morph.MorphologyComparer(
            query=101, dataset="np:v1", level="bodyid", method="vector_v2", expand_top_types=0,
            candidate_source="profile",
            output_dir=str(tmp_path / "out"), project_root=str(tmp_path),
            verbose=False,
        )
        monkeypatch.setattr(comparer, "_resolve_query", lambda: pd.DataFrame({
            "bodyId": [101], "type": ["T"], "instance": ["T_1"],
        }))
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: pd.DataFrame({
                                "target_bodyId": [201, 301],
                                "shared_count": [9, 5],
                                "profile_similarity": [1.0, 0.6],
                                "target_type": ["T", "M"],
                            }))
        # The shared mesh pickle is intentionally invisible to Find Similar;
        # simulate an unavailable raw online fetch for that candidate.
        monkeypatch.setattr(morph, "fetch_skeletons_on_demand_batch",
                            lambda *args, **kwargs: {})
        res = comparer.find_similar()
        # the skeleton pool neuron is scored; the mesh row stays in the
        # written candidate list with NaN similarity (ranked last)
        assert 201 in res["target_bodyId"].tolist()
        mesh_rows = res[res["target_bodyId"] == 301]
        assert len(mesh_rows) == 1
        assert pd.isna(mesh_rows.iloc[0]["similarity"])
        assert res["similarity"].notna().iloc[0]


# ---------------------------------------------------------------------------
# Vector persistence (always cache the vector, even without the skeleton)
# ---------------------------------------------------------------------------

class TestVectorPersistence:
    """Computed vectors are always persisted into the vector cache: the
    workflow is vector cache -> cached skeleton file -> online fetch, and
    the vector survives even when the original skeleton does not."""

    def test_vectors_for_persists_computed_vectors(self, tmp_path):
        """A vector computed from a cached skeleton file is appended, so a
        later call is a cache hit (no re-vectorization)."""
        write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        cache = morph.SkeletonVectorCache(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        cache.build()
        # a new skeleton file appears after the build
        write_skeleton(tmp_path, "np:v1", 202, bushy_y_neuron())
        X, mask, _ = cache.vectors_for([101, 202], compute_missing=True)
        assert mask.tolist() == [True, True]
        # the computed vector was persisted: a fresh call hits the cache
        data = cache.load()
        assert len(data["bodyIds"]) == 2
        X2, mask2, _ = cache.vectors_for([202], compute_missing=True)
        assert mask2[0]
        # the stored row is the same RAW vector, standardized on load
        idx = int(np.where(data["bodyIds"] == 202)[0][0])
        meta = data["meta"]
        raw_recovered = (data["X"][idx] * np.asarray(meta["std"])
                         + np.asarray(meta["mean"]))
        np.testing.assert_allclose(raw_recovered, X[1], rtol=1e-6)

    def test_profile_first_persists_transient_fetch_vector(self, tmp_path, monkeypatch):
        """A transiently-fetched skeleton (never written to the skeleton
        cache) still persists its VECTOR; the next run reuses the vector
        and skips the online fetch entirely."""
        write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        write_skeleton(tmp_path, "np:v1", 201, line_neuron(length=20))
        write_neuron_index(tmp_path, "np:v1", [
            (101, "T", "T_1"), (201, "T", "T_2"), (204, "Z", "Z_1"),
        ])
        fetched = []

        def fake_fetch(dataset, bid, project_root=None, persist=True):
            fetched.append(bid)
            return line_neuron(length=25)

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", fake_fetch)
        params = dict(
            query=101, dataset="np:v1", level="bodyid", method="vector_v2", expand_top_types=0,
            candidate_source="profile",
            output_dir=str(tmp_path / "out"), project_root=str(tmp_path),
            verbose=False,
        )

        def make():
            c = morph.MorphologyComparer(**params)
            monkeypatch.setattr(c, "_resolve_query", lambda: pd.DataFrame({
                "bodyId": [101], "type": ["T"], "instance": ["T_1"],
            }))
            monkeypatch.setattr(c, "_connection_cache_candidates",
                                lambda q: pd.DataFrame({
                                    "target_bodyId": [201, 204],
                                    "shared_count": [9, 5],
                                    "profile_similarity": [1.0, 0.6],
                                    "target_type": ["T", "Z"],
                                }))
            return c

        # run 1: 204 has no cached skeleton -> transient fetch
        res = make().find_similar()
        assert not res.empty
        assert fetched == [204]
        # the vector was persisted (V2 cache — the fetcher no longer writes
        # legacy V1 rows) even though the raw skeleton was not
        cache = morph.find_similar_dataset_cache_v2(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        data = cache.load()
        assert data is not None
        assert int(204) in set(int(b) for b in data["bodyIds"])
        assert data["dataset_rep"] == "skeleton"
        # the raw skeleton is persisted alongside (raw skeleton caching is
        # unconditional); the vector-reuse guarantee is what matters

        # run 2: the cached vector suffices -> no online fetch at all
        fetched.clear()
        res2 = make().find_similar()
        assert not res2.empty
        assert fetched == []

    def test_append_ignores_duplicate_bodyids(self, tmp_path):
        """Append-only pending rows dedupe at read time: re-appending a
        cached bodyId must never surface a second row (the append returns the
        appended count; the visible cache stays unchanged)."""
        write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        cache = morph.SkeletonVectorCache(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        cache.build()
        before = len(cache.load()["bodyIds"])
        X, _, _ = cache.vectors_for([101], compute_missing=True)
        added = cache.append_vectors([(101, X[0], "skeleton")])
        assert added == 1  # appended to the pending staging file
        # read-time first-wins dedupe: still exactly one visible row
        assert len(cache.load()["bodyIds"]) == before
        # after the merge checkpoint the main file is deduped too
        cache._merge_pending()
        assert len(cache.load()["bodyIds"]) == before
        assert not cache.pending_path.exists()


def test_seed_index_alone_does_not_authorize_skeleton_downloads(tmp_path):
    """A shipped seed index with no local data must not drive skeleton fetches.

    The bundled neuron-index seeds only support search surfaces.  On a fresh
    clone (no prepared tables, no cached connections, no skeletons) the seed
    must not be treated as the authoritative neuron list for bulk downloads,
    so the full dataset pull remains the required first step.
    """
    import pandas as pd

    folder = tmp_path / "neuron_indexes" / morph._dataset_folder("seed-only:v1.0")
    folder.mkdir(parents=True)
    pd.DataFrame(
        {"bodyId": [1, 2, 3], "type": ["A", "A", "B"], "instance": ["A1", "A2", "B1"]}
    ).to_parquet(folder / "neuron_index.parquet", index=False)

    result = morph.download_all_skeletons(
        "seed-only:v1.0", project_root=str(tmp_path), verbose=False
    )
    assert result["total"] == 0
    assert result["fetched"] == 0


def test_seed_index_fallback_active_once_dataset_is_local(tmp_path):
    """With cached skeletons present, the index fallback authorizes fetches again."""
    import pandas as pd

    write_neuron_index(tmp_path, "np:v1", [(101, "A", "A1"), (102, "B", "B1")])
    write_skeleton(tmp_path, "np:v1", 101, line_neuron())

    folder = tmp_path / "neuron_indexes" / morph._dataset_folder("np:v1")
    assert folder.exists()
    assert morph._has_local_dataset_presence("np:v1", tmp_path) is True

    index = morph._load_neuron_type_map("np:v1", project_root=str(tmp_path))
    assert index[0] == {101: "A", 102: "B"}
