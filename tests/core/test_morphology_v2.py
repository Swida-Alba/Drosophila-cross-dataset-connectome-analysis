"""Tests for the V2 spatial vectorization and comparison
(method="vector_v2") in src/morphology.py, plus the RoiProfileStore
expansion-block composer in src/roi_screening.py.

The V2 pipeline is strictly additive: the first 124 dims of a V2 vector are
the V1 vector, and every V1 code path keeps its own cache files and scorer.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import morphology as morph  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic neurons (same convention as test_morphology.py)
# ---------------------------------------------------------------------------

def make_neuron(points, parents, radius=1.0):
    import navis
    n = len(points)
    nodes = pd.DataFrame({
        "node_id": np.arange(n, dtype=np.int64),
        "parent_id": np.array([p if p >= 0 else -1 for p in parents],
                              dtype=np.int64),
        "x": [p[0] for p in points],
        "y": [p[1] for p in points],
        "z": [p[2] for p in points],
        "radius": [radius] * n,
        "type": ["0"] * n,
    })
    return navis.TreeNeuron(nodes)


def blob_neuron(center=(0.0, 0.0, 0.0), n=60, spread=5.0, seed=0):
    """A random bushy arbor around ``center`` (deterministic per seed)."""
    rng = np.random.default_rng(seed)
    pts = [tuple(center)]
    parents = [-1]
    for i in range(n - 1):
        base = rng.integers(0, len(pts))
        base_pt = pts[base]
        step = rng.normal(0, spread / 3.0, size=3)
        pts.append(tuple(np.asarray(base_pt) + step))
        parents.append(int(base))
    return make_neuron(pts, parents)


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

class TestSpatialEllipsoid:
    def test_shape_and_zero_fallback(self):
        out = morph.compute_spatial_ellipsoid(blob_neuron())
        assert out.shape == (morph.SPATIAL_ELLIPSOID_FEATURES.__len__(),)
        assert np.isfinite(out).all()
        assert np.allclose(
            morph.compute_spatial_ellipsoid(make_neuron([(0, 0, 0)], [-1])),
            0.0)

    def test_translation_moves_centroid_keeps_spread(self):
        nrn = blob_neuron(seed=3)
        shift = (100.0, 50.0, -30.0)
        moved = make_neuron(
            [tuple(np.asarray(p) + shift) for p in nrn.nodes[["x", "y", "z"]]
             .to_numpy()],
            nrn.nodes["parent_id"].to_numpy())
        a = morph.compute_spatial_ellipsoid(nrn)
        b = morph.compute_spatial_ellipsoid(moved)
        assert np.allclose(b[:3], a[:3] + np.asarray(shift), atol=1e-6)
        # Spread (sqrt eigenvalues) and anisotropy/flatness are translation
        # invariant; the principal axis is sign-stable either way.
        assert np.allclose(a[3:6], b[3:6], atol=1e-6)
        assert np.allclose(a[6:9], b[6:9], atol=1e-6) \
            or np.allclose(a[6:9], -b[6:9], atol=1e-6)
        assert np.isclose(a[3], a[3]) and a[3] > 0


class TestSpatialHistogram:
    def test_bounds_required(self):
        nrn = blob_neuron(seed=1)
        assert np.allclose(morph.compute_spatial_histogram_abs(nrn), 0.0)

    def test_position_separates_identical_shapes(self):
        bounds = np.array([[0.0, 0.0, 0.0], [200.0, 100.0, 80.0]])
        left = blob_neuron(center=(40, 50, 40), seed=5)
        right = blob_neuron(center=(160, 50, 40), seed=5)
        h_l = morph.compute_spatial_histogram_abs(left, bounds)
        h_r = morph.compute_spatial_histogram_abs(right, bounds)
        assert h_l.shape == (morph.SPATIAL_HIST_DIM,)
        assert np.isclose(np.linalg.norm(h_l), 1.0)   # L1 -> sqrt = unit
        # Same branching pattern, different brain location: different bins.
        assert np.linalg.norm(h_l - h_r) > 0.5
        # Same location, same pattern: identical histogram.
        h_l2 = morph.compute_spatial_histogram_abs(
            blob_neuron(center=(40, 50, 40), seed=5), bounds)
        assert np.allclose(h_l, h_l2)


class TestShapeExtras:
    def test_shape_and_non_skeleton_fallback(self):
        out = morph.compute_shape_extras(blob_neuron(seed=7))
        assert out.shape == (morph.SHAPE_EXTRA_DIM,) == (8,)
        assert np.isfinite(out).all()
        assert out[0] > 0     # make_neuron gives every node radius 1.0
        # non-skeleton inputs (e.g. mesh representations) carry no
        # radius/branch structure -> all zeros
        import types
        out2 = morph.compute_shape_extras(
            types.SimpleNamespace(vertices=np.zeros((10, 3))))
        assert np.allclose(out2, 0.0)

    def test_radius_stats(self):
        # make_neuron gives every node radius=1.0
        out = morph.compute_shape_extras(blob_neuron(seed=7))
        assert np.isclose(out[0], 1.0)            # sx_radius_mean
        assert np.isclose(out[1], 0.0)            # sx_radius_std
        assert np.isclose(out[2], 1.0)            # sx_radius_max
        assert np.isclose(out[3], 1.0)            # sx_radius_leaf_mean
        # no-radius source: all four are zero
        nrn = blob_neuron(seed=7)
        nrn.nodes = nrn.nodes.drop(columns=["radius"])
        out2 = morph.compute_shape_extras(nrn)
        assert np.allclose(out2[:4], 0.0)

    def test_branch_angle_straight_line(self):
        # A straight chain has no branch points -> zero angles.
        pts = [(float(i), 0.0, 0.0) for i in range(6)]
        parents = [-1, 0, 1, 2, 3, 4]
        out = morph.compute_shape_extras(make_neuron(pts, parents))
        assert np.isclose(out[4], 0.0) and np.isclose(out[5], 0.0)

    def test_branch_angle_right_angle(self):
        # One branch point whose two children leave at 90 degrees.
        pts = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (3, 0, 0)]
        parents = [-1, 0, 1, 2, 2]
        out = morph.compute_shape_extras(make_neuron(pts, parents))
        assert np.isclose(out[4], 90.0, atol=1e-6)
        assert np.isclose(out[5], 0.0, atol=1e-6)

    def test_strahler_fractions_sum_within_bounds(self):
        out = morph.compute_shape_extras(blob_neuron(seed=9))
        assert 0.0 <= out[6] <= 1.0 and 0.0 <= out[7] <= 1.0
        assert out[6] + out[7] <= 1.0 + 1e-9


class TestSpatialProfileExtras:
    def test_bounds_required(self):
        assert np.allclose(
            morph.compute_spatial_profile_extras(blob_neuron(seed=1)), 0.0)

    def test_profiles_unit_norm(self):
        bounds = np.array([[0.0, 0.0, 0.0], [200.0, 100.0, 80.0]])
        out = morph.compute_spatial_profile_extras(
            blob_neuron(center=(40, 50, 40), seed=5), bounds)
        assert out.shape == (morph.SPATIAL_PROFILE_DIM,) == (16,)
        # each Hellinger profile is a unit vector
        assert np.isclose(np.linalg.norm(out[:8]), 1.0)
        assert np.isclose(np.linalg.norm(out[8:]), 1.0)

    def test_mirror_symmetric_arbors_same_profiles(self):
        # Radial and midline-distance profiles are |distance|-based, so
        # mirrored arbors (same |x - mid|) produce identical profiles.
        bounds = np.array([[-100.0, 0.0, 0.0], [100.0, 100.0, 80.0]])
        left = blob_neuron(center=(-40, 50, 40), seed=5)
        right = blob_neuron(center=(40, 50, 40), seed=5)
        a = morph.compute_spatial_profile_extras(left, bounds)
        b = morph.compute_spatial_profile_extras(right, bounds)
        assert np.allclose(a, b, atol=1e-9)


class TestVectorizeNeuronV2:
    def test_v1_prefix_preserved(self):
        nrn = blob_neuron(seed=11)
        _, v1 = morph.vectorize_neuron(nrn)
        _, v2 = morph.vectorize_neuron_v2(nrn)
        assert v2.shape == (morph.VECTOR_V2_DIM,)
        assert morph.VECTOR_V2_DIM == 256
        np.testing.assert_allclose(v2[:morph.VECTOR_DIM], v1)

    def test_spatial_block_without_bounds(self):
        nrn = blob_neuron(seed=2)
        _, v2 = morph.vectorize_neuron_v2(nrn, spatial_bounds=None)
        spatial = v2[slice(*morph.V2_SPATIAL_SLICE)]
        # Ellipsoid half populated, histogram half zeros.
        assert np.any(spatial[:len(morph.SPATIAL_ELLIPSOID_FEATURES)] != 0)
        assert np.allclose(
            spatial[len(morph.SPATIAL_ELLIPSOID_FEATURES)
                    + morph.SPATIAL_HIST_DIM:], 0.0)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class TestWhitening:
    def test_whitened_covariance_is_identity(self):
        # Well-conditioned population: every eigenvalue above the relative
        # floor, so truncated ZCA lands on (approximately) identity.
        rng = np.random.default_rng(0)
        mixing = np.eye(16) + 0.1 * rng.normal(size=(16, 16))
        X = rng.normal(size=(2000, 16)) @ mixing
        X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)
        evals = np.linalg.eigvalsh(np.cov(X, rowvar=False))
        assert evals.min() > 1e-2 * evals.max()   # precondition for identity
        W = morph.fit_zca_whitener(X)
        cov = np.cov(morph.apply_whitening(W, X), rowvar=False)
        assert np.allclose(cov, np.eye(16), atol=5e-3)

    def test_whitening_passes_through_noise_directions(self):
        # A near-zero-variance direction must NOT be amplified: the old
        # eps-regularized ZCA scaled it by 1/sqrt(eps), and the shared
        # constant then dominated every cosine (the saturation bug). The
        # truncated whitener leaves such directions at unit gain.
        rng = np.random.default_rng(1)
        X = rng.normal(size=(500, 16)) @ rng.normal(size=(16, 16))
        X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)
        sigma = np.cov(X, rowvar=False)
        evals, evecs = np.linalg.eigh(sigma)
        floor = 1e-2 * evals.max()
        W = morph.fit_zca_whitener(X)
        cov = np.cov(morph.apply_whitening(W, X), rowvar=False)
        for j in range(16):
            var = float(evecs[:, j] @ cov @ evecs[:, j])
            if evals[j] < floor:
                # passed through: variance unchanged (never amplified)
                assert var <= evals[j] * 1.5 + 1e-6
            else:
                # whitened directions land near unit variance
                assert var <= 1.0 + 1e-6
        assert np.all(np.diag(cov) <= 100.0 + 1e-6)   # amplification cap

    def test_small_population_returns_identity(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(10, 8))
        W = morph.fit_zca_whitener((X - X.mean(axis=0)))
        # fit still returns a matrix; the CACHE layer gates on population
        # size, verified through SkeletonVectorCacheV2._whitener below.
        assert W.shape == (8, 8)


class TestV2Similarity:
    def test_identical_neurons_score_one(self):
        nrn = blob_neuron(seed=13)
        bounds = np.array([[-50.0] * 3, [250.0] * 3])
        _, q = morph.vectorize_neuron_v2(nrn, bounds)
        total, blocks = morph.v2_similarity_matrix(
            q, q[None, :], dict(morph.DEFAULT_V2_BLOCK_WEIGHTS))
        assert np.isclose(total[0], 1.0, atol=1e-9)
        # merged default: topology rides inside the spatial block
        assert set(blocks) == {"shape", "spatial"}

    def test_two_block_matches_direct_cosine(self):
        rng = np.random.default_rng(11)
        q = rng.normal(size=morph.VECTOR_V2_DIM)
        row = rng.normal(size=morph.VECTOR_V2_DIM)
        total, blocks = morph.v2_similarity_matrix(
            q, row[None, :], dict(morph.DEFAULT_V2_BLOCK_WEIGHTS))
        s_shape = float(morph.cosine_similarity_matrix(
            q[morph.V2_SHAPE_SLICE[0]:morph.V2_SHAPE_SLICE[1]],
            row[morph.V2_SHAPE_SLICE[0]:morph.V2_SHAPE_SLICE[1]][None, :])[0])
        a, b = morph.V2_SPATIAL_SLICE
        s_spatial = float(morph.cosine_similarity_matrix(
            q[a:b], row[a:b][None, :])[0])
        w = morph.DEFAULT_V2_BLOCK_WEIGHTS
        expected = (w["shape"] * s_shape + w["spatial"] * s_spatial) \
            / (w["shape"] + w["spatial"])
        assert np.isclose(total[0], expected, atol=1e-9)
        assert np.isclose(blocks["spatial"][0], s_spatial, atol=1e-9)
        assert set(blocks) == {"shape", "spatial"}

    def test_shape_extras_carry_shape_block(self):
        rng = np.random.default_rng(3)
        q = rng.normal(size=morph.VECTOR_V2_DIM)
        row = rng.normal(size=morph.VECTOR_V2_DIM)
        # Perturb ONLY the shape-extra dims (124:132): the shape block
        # moves, the spatial block (132:256) does not.
        a, b = morph.VECTOR_DIM, morph.SHAPE_BLOCK_DIM
        row_pert = row.copy()
        row_pert[a:b] += rng.normal(size=b - a)
        _, blocks = morph.v2_similarity_matrix(
            q, row_pert[None, :], dict(morph.DEFAULT_V2_BLOCK_WEIGHTS))
        total_plain, _ = morph.v2_similarity_matrix(
            q, row[None, :], dict(morph.DEFAULT_V2_BLOCK_WEIGHTS))
        total_pert, _ = morph.v2_similarity_matrix(
            q, row_pert[None, :], dict(morph.DEFAULT_V2_BLOCK_WEIGHTS))
        assert not np.isclose(total_plain[0], total_pert[0])
        assert not np.isclose(blocks["shape"][0],
                              morph.cosine_similarity_matrix(
            q[morph.V2_SHAPE_SLICE[0]:morph.V2_SHAPE_SLICE[1]],
            row[morph.V2_SHAPE_SLICE[0]:morph.V2_SHAPE_SLICE[1]][None, :])[0])
        s_spatial = float(morph.cosine_similarity_matrix(
            q[morph.V2_SPATIAL_SLICE[0]:morph.V2_SPATIAL_SLICE[1]],
            row[morph.V2_SPATIAL_SLICE[0]:morph.V2_SPATIAL_SLICE[1]][None, :])[0])
        assert np.isclose(blocks["spatial"][0], s_spatial, atol=1e-9)

    def test_position_mismatch_lowers_v2_not_v1(self):
        bounds = np.array([[0.0, 0.0, 0.0], [200.0, 100.0, 80.0]])
        left = blob_neuron(center=(40, 50, 40), seed=5)
        right = blob_neuron(center=(160, 50, 40), seed=5)
        _, v_l = morph.vectorize_neuron_v2(left, bounds)
        _, v_r = morph.vectorize_neuron_v2(right, bounds)
        # V1 shape prefix: identical branching -> near-identical vectors.
        v1_cos = float(morph.cosine_similarity_matrix(
            v_l[:morph.VECTOR_DIM], v_r[:morph.VECTOR_DIM][None, :])[0])
        total, blocks = morph.v2_similarity_matrix(
            v_l, v_r[None, :], dict(morph.DEFAULT_V2_BLOCK_WEIGHTS))
        assert v1_cos > 0.99
        # V1 sees two identical neurons; V2's spatial block separates them
        # (raw-vector level; production z-scoring widens the gap further).
        assert blocks["shape"][0] > 0.99
        assert blocks["spatial"][0] < 0.9
        assert total[0] < v1_cos - 0.05

    def test_zero_block_renormalizes_weights(self):
        rng = np.random.default_rng(9)
        q = rng.normal(size=morph.VECTOR_V2_DIM)
        row = rng.normal(size=morph.VECTOR_V2_DIM)
        a, b = morph.V2_SPATIAL_SLICE
        row_zero = row.copy()
        row_zero[a:b] = 0.0    # candidate has no spatial evidence at all
        # only shape participates: the total is the shape cosine, not a
        # value dragged toward 0 by the empty block
        total, blocks = morph.v2_similarity_matrix(
            q, row_zero[None, :], dict(morph.DEFAULT_V2_BLOCK_WEIGHTS))
        s_shape = float(morph.cosine_similarity_matrix(
            q[morph.V2_SHAPE_SLICE[0]:morph.V2_SHAPE_SLICE[1]],
            row_zero[morph.V2_SHAPE_SLICE[0]:morph.V2_SHAPE_SLICE[1]][None, :])[0])
        assert np.isclose(total[0], s_shape, atol=1e-9)
        assert np.isclose(blocks["spatial"][0], 0.0, atol=1e-12)

    def test_pairwise_agrees_with_one_to_one(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(5, morph.VECTOR_V2_DIM))
        pair = morph.v2_pairwise_matrix(X, dict(morph.DEFAULT_V2_BLOCK_WEIGHTS))
        one, _ = morph.v2_similarity_matrix(
            X[0], X[1:3], dict(morph.DEFAULT_V2_BLOCK_WEIGHTS))
        assert np.allclose(pair[0, 1:3], one, atol=1e-9)

    def test_lateral_normalization_at_extraction(self):
        # Lateral normalization happens at VECTORIZATION time: a right-
        # hemisphere neuron is reflected onto the left before the spatial
        # features are computed, so bilateral homologs store identical
        # spatial blocks and compare directly (no scoring-time mirroring).
        bounds = np.array([[-100.0, 0.0, 0.0], [100.0, 100.0, 80.0]])
        left = blob_neuron(center=(-40, 50, 40), seed=6)
        pts = left.nodes[["x", "y", "z"]].to_numpy().copy()
        pts[:, 0] *= -1.0                      # exact midline mirror
        right = make_neuron(pts, left.nodes["parent_id"].to_numpy())
        _, v_l = morph.vectorize_neuron_v2(left, bounds, lateral_normalize=True)
        _, v_r = morph.vectorize_neuron_v2(right, bounds, lateral_normalize=True)
        a, b = morph.V2_SPATIAL_SLICE
        np.testing.assert_allclose(v_l[a:b], v_r[a:b], atol=1e-6)
        # Without the flag the two hemispheres stay distinct (fallback).
        _, v_r_plain = morph.vectorize_neuron_v2(right, bounds)
        assert not np.allclose(v_l[a:b], v_r_plain[a:b], atol=1e-6)

    def test_side_midline_helper(self):
        cents = [90.0, 91.0, 89.0, 90.0, 90.0,
                 170.0, 171.0, 169.0, 170.0, 170.0]
        sides = ["left"] * 5 + ["right"] * 5
        out = morph.side_midline(cents, sides)
        assert out is not None
        assert np.isclose(out["midline"], 130.0)
        assert out["reflect_side"] == "right"
        assert (out["n_left"], out["n_right"]) == (5, 5)
        assert morph.side_midline([1.0, 2.0], ["left", "right"]) is None

    def test_explicit_midline_and_reflect_override(self):
        # True midline 130, bbox midpoint 100: the left arbor stays put,
        # the mirrored right arbor is reflected about 130. Explicit
        # midline_x/reflect must recover identical spatial blocks.
        bounds = np.array([[0.0, 0.0, 0.0], [200.0, 100.0, 80.0]])
        left = blob_neuron(center=(90, 50, 40), seed=6)
        pts = left.nodes[["x", "y", "z"]].to_numpy().copy()
        pts[:, 0] = 260.0 - pts[:, 0]               # mirror about x=130
        right = make_neuron(pts, left.nodes["parent_id"].to_numpy())
        _, v_l = morph.vectorize_neuron_v2(
            left, bounds, lateral_normalize=True, midline_x=130.0)
        _, v_r = morph.vectorize_neuron_v2(
            right, bounds, lateral_normalize=True, midline_x=130.0,
            reflect=True)
        a, b = morph.V2_SPATIAL_SLICE
        np.testing.assert_allclose(v_l[a:b], v_r[a:b], atol=1e-6)
        # The same right arbor reflected about the bbox midpoint instead
        # does not align.
        _, v_bbox = morph.vectorize_neuron_v2(
            right, bounds, lateral_normalize=True)
        assert not np.allclose(v_l[a:b], v_bbox[a:b], atol=1e-6)
        # Forcing reflect=False leaves it unreflected.
        _, v_plain = morph.vectorize_neuron_v2(
            right, bounds, lateral_normalize=True, midline_x=130.0,
            reflect=False)
        assert not np.allclose(v_r[a:b], v_plain[a:b], atol=1e-6)

    def test_spatial_overlap_term(self):
        # The mass-overlap component scores the fraction of the query's
        # arbor distribution the candidate actually shares: a candidate
        # with the same distribution shape but less mass in the shared
        # region scores lower than one with full overlap.
        rng = np.random.default_rng(8)
        q = rng.random(96)
        q = q / q.sum()                       # L1 proportions, as stored
        full = q.copy()                       # identical distribution
        partial = np.full(96, q.mean())       # same total mass, spread out
        total, _ = morph.v2_similarity_matrix(
            np.zeros(256), np.zeros((2, 256)), dict(morph.DEFAULT_V2_BLOCK_WEIGHTS),
            spatial_overlap={"members": None, "centroid": q,
                             "pool": np.stack([full, partial])},
            query_index=None)
        # With zero blocks all cosines are 0 -> overlap drives the score.
        assert total[0] > total[1] > 0
        # identical -> spatial score 0.5; the all-zero rows count as valid
        # (production _block_cosine_one semantics), so with equal default
        # weights the renormalizer is shape 0.5 + spatial 0.5.
        w = morph.DEFAULT_V2_BLOCK_WEIGHTS
        assert np.isclose(total[0],
                          w["spatial"] * 0.5 / (w["shape"] + w["spatial"]),
                          atol=1e-9)

# ---------------------------------------------------------------------------
# V2 cache
# ---------------------------------------------------------------------------

class TestSkeletonVectorCacheV2:
    def _cache(self, tmp_path):
        cache = morph.SkeletonVectorCacheV2(
            "hemibrain:v1.2.1", project_root=str(tmp_path), verbose=False)
        return cache

    def test_separate_files_same_skeleton_dir(self, tmp_path):
        v1 = morph.find_similar_raw_cache("hemibrain:v1.2.1",
                                          project_root=str(tmp_path),
                                          verbose=False)
        v2 = self._cache(tmp_path)
        assert v2.skeleton_dir == v1.skeleton_dir
        assert v2.parquet_path != v1.parquet_path
        assert v2.meta_path != v1.meta_path
        assert v2.pending_path != v1.pending_path
        assert v2.parquet_path.name == "skeleton__vectors_v2.parquet"
        assert v2.meta_path.name == "meta_v2.json"

    def test_append_load_roundtrip(self, tmp_path):
        cache = self._cache(tmp_path)
        nrn = blob_neuron(seed=8)
        bounds = np.array([[-50.0] * 3, [250.0] * 3])
        _, vec = morph.vectorize_neuron_v2(nrn, bounds)
        # Simulate a build-written meta (identity standardization) so the
        # appended row round-trips through load() unchanged.
        cache.morph_dir.mkdir(parents=True, exist_ok=True)
        cache._write_meta({"mean": [0.0] * morph.VECTOR_V2_DIM,
                           "std": [1.0] * morph.VECTOR_V2_DIM}, 0)
        assert cache.append_vectors([(42, vec, "skeleton")]) == 1
        data = cache.load()
        assert data is not None and len(data["bodyIds"]) == 1
        assert data["X"].shape == (1, morph.VECTOR_V2_DIM)
        assert data["raw"].shape == (1, morph.VECTOR_V2_DIM)
        np.testing.assert_allclose(data["raw"][0], vec, atol=1e-9)
        meta = cache._load_meta()
        assert meta["version"] == morph.VECTOR_CACHE_V2_VERSION
        assert meta["dataset"] == "hemibrain:v1.2.1"
        assert meta["feature_columns"][0] == "cable_length"
        assert len(meta["feature_columns"]) == morph.VECTOR_V2_DIM
        assert meta["block_weights"] == dict(morph.DEFAULT_V2_BLOCK_WEIGHTS)
        # V1 cache files were never touched.
        assert not (tmp_path / "cache" / "hemibrain_v1_2_1" / "morphology"
                    / "skeleton_vectors.parquet").exists()

    def test_whitener_identity_below_population(self, tmp_path):
        cache = self._cache(tmp_path)
        rng = np.random.default_rng(2)
        W = cache._whitener(rng.normal(size=(10, morph.VECTOR_V2_DIM)))
        assert np.allclose(W, np.eye(morph.VECTOR_V2_DIM))

    def test_whitener_fitted_and_persisted(self, tmp_path):
        cache = self._cache(tmp_path)
        rng = np.random.default_rng(3)
        X = rng.normal(size=(morph.MIN_ROWS_FOR_WHITENING + 10,
                             morph.VECTOR_V2_DIM))
        X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)
        W = cache._whitener(X)
        assert W.shape == (morph.VECTOR_V2_DIM, morph.VECTOR_V2_DIM)
        assert cache.whiten_path.exists()
        with np.load(cache.whiten_path) as data:
            assert int(data["fit_version"]) == morph.WHITEN_FIT_VERSION
            assert np.allclose(W, data["W"])

    def test_spatial_bounds_persisted(self, tmp_path):
        cache = self._cache(tmp_path)
        # No skeletons available: bounds stay None (histogram zeros).
        assert cache.spatial_bounds() is None


# ---------------------------------------------------------------------------
# Comparer wiring
# ---------------------------------------------------------------------------

class TestComparerV2Wiring:
    def test_method_accepts_vector_v2(self):
        c = morph.MorphologyComparer(query="aMe4", dataset="male-cns:v1.0",
                                     method="vector_v2")
        assert c._is_v2
        # topology block removed in schema v4, ROI block removed from
        # scoring (ROI evidence is candidate-selection only): two blocks
        assert c._v2_weights == {"shape": 0.3, "spatial": 0.7}

    def test_unknown_weight_key_ignored(self):
        c = morph.MorphologyComparer(
            query=1, dataset="hemibrain:v1.2.1", method="vector_v2",
            v2_block_weights={"shape": 0.8, "topology": 0.5, "bogus": 9.0})
        # "topology" is no longer a known block key and is dropped
        assert c._v2_weights["shape"] == 0.8
        assert "topology" not in c._v2_weights
        assert "bogus" not in c._v2_weights

    def test_weight_override(self):
        c = morph.MorphologyComparer(
            query=1, dataset="hemibrain:v1.2.1", method="vector_v2",
            v2_block_weights={"shape": 0.8, "bogus": 9.0})
        assert c._v2_weights["shape"] == 0.8
        assert c._v2_weights["spatial"] == 0.7
        assert "bogus" not in c._v2_weights

    def test_expand_params(self):
        c = morph.MorphologyComparer(query=1, dataset="hemibrain:v1.2.1",
                                     method="vector_v2",
                                     expand_top_types=0, expand_per_type=5)
        assert c.expand_top_types == 0 and c.expand_per_type == 5
        with pytest.raises(ValueError):
            morph.MorphologyComparer(query=1, dataset="hemibrain:v1.2.1",
                                     method="vector_v2", expand_per_type=-1)

    def test_vector_v2_is_default(self):
        c = morph.MorphologyComparer(query=1, dataset="hemibrain:v1.2.1")
        assert c.method == "vector_v2"
        assert c._is_v2

    def test_legacy_vector_method_rejected(self):
        with pytest.raises(ValueError):
            morph.MorphologyComparer(query=1, dataset="hemibrain:v1.2.1",
                                     method="vector")

    def test_invalid_method_rejected(self):
        with pytest.raises(ValueError):
            morph.MorphologyComparer(query=1, dataset="hemibrain:v1.2.1",
                                     method="vector3")

    def test_nblast_mode_scores_with_v2_blocks(self):
        # metric is a retired no-op: both methods score through the V2
        # block-weighted scorer (weights renormalize to plain cosine here).
        c = morph.MorphologyComparer(query=1, dataset="hemibrain:v1.2.1",
                                     method="nblast", metric="pearson")
        q = np.array([1.0, 2.0, 3.0])
        m = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        np.testing.assert_allclose(
            c._similarity_matrix(q, m),
            morph.cosine_similarity_matrix(q, m), rtol=1e-9)

    def test_pool_subset_slicing(self):
        c = morph.MorphologyComparer(query=1, dataset="hemibrain:v1.2.1",
                                     method="vector_v2")
        hist = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        c._v2_spatial_overlap = {"members": hist[[0]], "centroid": hist[0],
                                 "pool": hist}
        mask = np.array([True, False, True])
        sliced = c._slice_spatial_overlap(mask)
        assert sliced["pool"].shape == (2, 2)
        np.testing.assert_allclose(sliced["pool"], hist[[0, 2]])
        idx = np.array([0, 2])
        assert c._slice_spatial_overlap(idx)["pool"].shape == (2, 2)
        assert c._slice_spatial_overlap(None)["pool"].shape == (3, 2)


# ---------------------------------------------------------------------------
# Type reevaluation: expansion selection + coverage-aware aggregation
# ---------------------------------------------------------------------------

class TestTypeReevaluation:
    def _comparer(self):
        return morph.MorphologyComparer(query=1, dataset="hemibrain:v1.2.1",
                                        method="vector_v2")

    def _rows(self, scores_by_type):
        """One scored row per member (aMe4 = query type rows excluded)."""
        rows = []
        for t, vals in scores_by_type.items():
            for v in vals:
                rows.append({"target_bodyId": 1000 + len(rows),
                             "target_type": t,
                             "similarity": v,
                             "is_same_type": t == "aMe4"})
        return rows

    def test_expansion_selection_prefers_screen_rank(self):
        id_to_type = {1: "T", 2: "T", 3: "T", 9: "Other", 4: "T"}
        scored = {2}
        # ROI screen ranking: bodyId 3 first, then 1, 4 unscreened.
        roi_rank = {3: (-0.9, 3), 1: (-0.7, 1)}
        out = morph.MorphologyComparer._select_expansion_ids(
            ["T"], id_to_type, scored, roi_rank, per_type_cap=2, total_cap=10)
        assert out == [3, 1]          # screened first, already-scored skipped
        out = morph.MorphologyComparer._select_expansion_ids(
            ["T"], id_to_type, scored, roi_rank, per_type_cap=10, total_cap=10)
        assert out == [3, 1, 4]       # unscreened ids trail, not dropped

    def test_expansion_selection_respects_caps_and_order(self):
        id_to_type = {i: ("A" if i < 10 else "B") for i in range(20)}
        roi_rank = {}
        out = morph.MorphologyComparer._select_expansion_ids(
            ["A", "B"], id_to_type, set(), roi_rank,
            per_type_cap=3, total_cap=5)
        assert len(out) == 5
        assert out.count == out.count  # noqa: PLW0120 — silence linters
        # 3 of A (first-ranked type) then 2 of B within the total cap.
        assert sum(1 for i in out if i < 10) == 3

    def test_coverage_sqrt_weighting(self):
        c = self._comparer()
        rows = self._rows({"Lucky": [0.40], "Solid": [0.30, 0.31, 0.32, 0.33],
                           "Mid": [0.34, 0.38],
                           "Far": [0.10, 0.10, 0.11, 0.12],
                           "Far2": [0.10, 0.12, 0.12, 0.12],
                           "aMe4": [0.9, 0.91]})
        pop = {"Lucky": 100, "Solid": 4, "Mid": 5, "Far": 50, "Far2": 50,
               "aMe4": 19}
        df = c._aggregate_type_rows(rows, query_type="aMe4", intra=0.6,
                                    query_type_count=19,
                                    population_counts=pop)
        by = df.set_index("target_type")
        # Continuous multiplicative weighting: similarity * sqrt(coverage).
        # No thresholds, no exclusion — sparse types are damped, never
        # zeroed, and the raw statistic stays comparable.
        assert np.isclose(by.loc["Solid", "similarity"], 0.315)   # cov 1.0
        assert np.isclose(by.loc["Mid", "similarity"],
                          0.36 * np.sqrt(0.4), atol=1e-9)         # cov 0.4
        assert np.isclose(by.loc["Lucky", "similarity"],
                          0.40 * np.sqrt(0.01), atol=1e-9)        # cov 0.01
        assert by.loc["Solid", "similarity"] > by.loc["Mid", "similarity"] \
            > by.loc["Lucky", "similarity"] > 0
        assert np.isclose(by.loc["Lucky", "similarity_raw"], 0.40)
        assert np.isclose(by.loc["Lucky", "type_coverage"], 0.01)
        # Best member stays visible regardless of aggregation.
        assert np.isclose(by.loc["Lucky", "similarity_max"], 0.40)
        assert np.isclose(by.loc["Solid", "similarity_max"], 0.33)
        # Intra-type reference keeps its score and rank-1 ordering.
        assert df.iloc[0]["target_type"] == "aMe4"
        assert np.isclose(df.iloc[0]["similarity"], 0.6)
        assert "low_coverage" not in df.columns

    def test_coverage_factor_monotone_and_untouched_at_one(self):
        # sqrt curve: continuous, monotone, identity at full coverage.
        assert morph._coverage_factor(1.0) == 1.0
        assert np.isclose(morph._coverage_factor(0.25), 0.5)
        assert 0.0 < morph._coverage_factor(0.01) \
            < morph._coverage_factor(0.2) \
            < morph._coverage_factor(0.5) \
            < morph._coverage_factor(0.8) \
            < morph._coverage_factor(1.0)
        assert morph._coverage_factor(float("nan")) == 1.0

    def test_type_agg_max_uses_best_member(self):
        c = morph.MorphologyComparer(query=1, dataset="hemibrain:v1.2.1",
                                     method="vector_v2", type_agg="max")
        rows = self._rows({"Solid": [0.30, 0.31, 0.32, 0.33],
                           "aMe4": [0.9, 0.91]})
        pop = {"Solid": 4, "aMe4": 19}
        df = c._aggregate_type_rows(rows, query_type="aMe4", intra=0.6,
                                    query_type_count=19,
                                    population_counts=pop)
        by = df.set_index("target_type")
        # Full coverage: the max statistic passes through unscaled.
        assert np.isclose(by.loc["Solid", "similarity_raw"], 0.33)
        assert np.isclose(by.loc["Solid", "similarity"], 0.33)
        assert np.isclose(by.loc["Solid", "similarity_max"], 0.33)

    def test_type_coverage_column_in_bodyid_export(self):
        c = self._comparer()
        rows = self._rows({"T1": [0.3, 0.3], "T2": [0.2]})
        cov = {"T1": 0.5, "T2": 0.25}
        df = c._bodyid_dataframe(rows, query_type="aMe4", type_coverage=cov)
        assert "type_coverage" in df.columns
        assert list(df.columns).index("type_coverage") \
            == list(df.columns).index("similarity") + 1
        got = df.set_index("target_type").type_coverage
        assert np.allclose(got["T1"], 0.5) and np.allclose(got["T2"], 0.25)
        # Absent when no coverage map is supplied (V1 exports unchanged).
        df1 = c._bodyid_dataframe(rows, query_type="aMe4")
        assert "type_coverage" not in df1.columns

    def test_invalid_type_agg_rejected(self):
        with pytest.raises(ValueError):
            morph.MorphologyComparer(query=1, dataset="hemibrain:v1.2.1",
                                     method="vector_v2", type_agg="median")

    def test_no_population_counts_keeps_v1_behavior(self):
        c = self._comparer()
        rows = self._rows({"Lucky": [0.40], "Solid": [0.30, 0.31, 0.32, 0.33]})
        df = c._aggregate_type_rows(rows, query_type="aMe4", intra=float("nan"),
                                    query_type_count=0)
        by = df.set_index("target_type")
        # Plain means: the singleton outranks (historical behavior).
        assert by.loc["Lucky", "similarity"] > by.loc["Solid", "similarity"]
        assert "coverage" not in df.columns
        assert "similarity_raw" not in df.columns

    def test_block_mean_columns_flow_through(self):
        c = self._comparer()
        rows = self._rows({"T1": [0.3, 0.3]})
        for i, r in enumerate(rows):
            r["sim_shape"] = 0.5 + i * 0.1
            r["sim_spatial"] = 0.2
        df = c._aggregate_type_rows(rows, query_type="aMe4", intra=float("nan"))
        assert "sim_shape" in df.columns
        assert np.isclose(df.iloc[0]["sim_shape"], 0.55)
        assert np.isclose(df.iloc[0]["sim_spatial"], 0.2)



# ---------------------------------------------------------------------------
# ROI expansion block
# ---------------------------------------------------------------------------

class TestRoiExpansionRows:
    def test_expansion_rows_hellinger(self, tmp_path, monkeypatch):
        import roi_screening as rois

        def _fake_build(self):
            self.bodyIds = np.array([1, 2, 3], dtype=np.int64)
            self.rois = ["A(L)", "A(R)"]
            self.pre = np.array([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]],
                                dtype=np.float32)
            self.post = np.array([[4.0, 4.0], [8.0, 0.0], [2.0, 2.0]],
                                 dtype=np.float32)
            self._rebuild_index()
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.cache_file, bodyIds=self.bodyIds,
                rois=np.array(self.rois), pre=self.pre, post=self.post,
                fingerprint=np.array("test"))
            # Mirror the real build: normalized views exist, raw counts are
            # only available on disk afterwards.
            self._ensure_normalized()
            return self

        monkeypatch.setattr(rois.RoiProfileStore, "build", _fake_build)
        store = rois.RoiProfileStore(
            "hemibrain:v1.2.1", project_root=str(tmp_path)).ensure()
        found, block = store.expansion_rows([1, 3, 999])
        assert found.tolist() == [1, 3]
        assert block.shape == (2, 4)   # 2 ROIs x (pre, post)
        # Neuron 1: all pre mass in A(L), post split evenly.
        np.testing.assert_allclose(block[0, :2], [1.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(block[0, 2:], [np.sqrt(0.5)] * 2,
                                   atol=1e-9)
        # The screen still works afterwards (state untouched).
        scores = store.screen([1])
        assert len(scores) == 2   # bodyIds 2 and 3


def test_soma_side_map_reads_banc_soma_side(tmp_path):
    """The BANC index calls the column 'Soma side', not 'somaSide'."""
    import polars as pl
    folder = tmp_path / "neuron_indexes" / "banc_v888"
    folder.mkdir(parents=True)
    pl.DataFrame({
        "bodyId": ["1", "2", "3", "4"],
        "Soma side": ["left", "right", None, "left"],
    }).write_parquet(folder / "neuron_index.parquet")
    sides = morph._dataset_soma_side_map("banc_v888", str(tmp_path))
    assert sides[1] == "left"
    assert sides[2] == "right"
    assert sides[3] == ""
    assert sides[4] == "left"


def test_fafb_bundle_bounds_fallback(tmp_path, monkeypatch):
    """A bundle-only FAFB cache gets bounds before fetched rows are vectorized."""
    class _Bundle:
        def ids(self):
            return list(range(1, 7))

        def get(self, body_id):
            return (f"1 0 0 0 0 1 -1\n"
                    f"2 0 {body_id * 10.0} {body_id * 2.0} 5 1 1\n")

        def close(self):
            pass

    monkeypatch.setattr(morph, "_fafb_bundle", lambda *a, **k: _Bundle())
    cache = morph.SkeletonVectorCacheV2(
        "flywire_FAFB_v783", project_root=str(tmp_path), n_workers=1,
        verbose=False, representation="skeleton")
    monkeypatch.setattr(cache, "_discover_skeleton_files", lambda: [])
    bounds = cache.spatial_bounds()
    assert bounds is not None and bounds.shape == (2, 3)
    assert np.all(bounds[0] < bounds[1])
    # 2% padding on every axis (the root node contributes x=0).
    assert np.isclose(bounds[0, 0], -0.02 * 60.0)
    assert np.isclose(bounds[1, 0], 60.0 + 0.02 * 60.0)
