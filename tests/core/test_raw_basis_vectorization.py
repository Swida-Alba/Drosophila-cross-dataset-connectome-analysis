"""Raw-basis vectorization design (plan-raw-basis-vectorization.md).

Contract: the on-disk skeleton cache stores RAW skeletons; simplification
is a visualization-time concern; the V2 vector cache builds from raw trees
only — a simplified file must never seed the vector cache, and the
fetch/persist entry points default to raw (level 0).
"""

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import morphology as morph  # noqa: E402


def _chain(n=60, seed=3):
    rng = np.random.default_rng(seed)
    pts = [(float(i), float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2)))
           for i in range(n)]
    nodes = pd.DataFrame({
        "node_id": np.arange(n, dtype=np.int64),
        "parent_id": np.array([-1] + list(range(n - 1)), dtype=np.int64),
        "x": [p[0] for p in pts],
        "y": [p[1] for p in pts],
        "z": [p[2] for p in pts],
        "radius": [1.0] * n,
        "type": ["0"] * n,
    })
    import navis
    return navis.TreeNeuron(nodes)


BOUNDS = np.array([[0.0, -5.0, -5.0], [80.0, 5.0, 5.0]])


def test_fetch_entry_points_default_to_raw():
    """The fetch/persist entry points default to raw (level 0)."""
    for fn in (morph.fetch_skeleton_on_demand,
               morph.fetch_skeletons_on_demand_batch):
        default = inspect.signature(fn).parameters["simplification"].default
        assert default == 0, fn.__name__
    default = inspect.signature(
        morph.SkeletonVectorCache.persist_skeletons
    ).parameters["simplification"].default
    assert default == 0


def test_v2_file_worker_skips_simplified_file(tmp_path):
    """A simp90-stamped on-disk file must not feed the V2 cache."""
    nrn = _chain()
    simp = tmp_path / "101.swc.zst"
    morph._write_compressed_skeleton(simp, nrn, simplification=90)
    assert morph._vectorize_one_file_v2(str(simp), bounds=BOUNDS) is None

    rawp = tmp_path / "102.swc.zst"
    morph._write_compressed_skeleton(rawp, nrn, simplification=0)
    row = morph._vectorize_one_file_v2(str(rawp), bounds=BOUNDS)
    assert row is not None
    assert row[0] == 102
    assert len(row[1]) == morph.VECTOR_V2_DIM


def test_write_compressed_skeleton_level0_records_honest_stamp(tmp_path):
    """simplification=0 applies no simplification and records the tree's
    own stored level — a simp90 tree is never relabeled as raw."""
    nrn = _chain()
    simp90 = morph._downsample_for_cache(
        nrn, morph._simplification_factor(90))
    simp90._drocat_simplification = 90

    p90 = tmp_path / "90.swc.zst"
    morph._write_compressed_skeleton(p90, simp90, simplification=0)
    reloaded = morph._load_cached_skeleton_file(p90)
    assert (getattr(reloaded, "_drocat_simplification", 0) or 0) == 90

    p0 = tmp_path / "0.swc.zst"
    morph._write_compressed_skeleton(p0, nrn, simplification=0)
    reloaded0 = morph._load_cached_skeleton_file(p0)
    assert (getattr(reloaded0, "_drocat_simplification", 0) or 0) == 0


def test_v2_file_level_gate():
    """The V2 class gate admits raw/absent stamps and rejects simp90."""
    inst = morph.SkeletonVectorCacheV2.__new__(morph.SkeletonVectorCacheV2)
    assert inst._file_level_ok(SimpleNamespace(_drocat_simplification=0))
    assert inst._file_level_ok(SimpleNamespace())
    assert not inst._file_level_ok(
        SimpleNamespace(_drocat_simplification=90))
    assert inst._default_basis() == morph.VECTOR_BASIS_RAW


def test_null_vector_store_signature_tracks_cache_version(tmp_path,
                                                          monkeypatch):
    """A V2 cache-version bump must invalidate existing sidecars."""
    from comparison.morph_cross_dataset import NullVectorStore

    store = NullVectorStore("np:v1", project_root=str(tmp_path))
    before = store._bounds_signature()
    monkeypatch.setattr(morph, "VECTOR_CACHE_V2_VERSION",
                        morph.VECTOR_CACHE_V2_VERSION + 1)
    after = store._bounds_signature()
    assert before != after
