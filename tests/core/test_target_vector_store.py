"""The persisted target-vector store: what it may reuse, and what it must refuse.

The win is measured (0.412 s per neuron of skeleton load + render transform +
vectorize, re-paid on every morph qualification before this existed), so the
risk is reusing geometry that moved — or reusing it altered. Three layers of
invalidation, all tested here: a whole-file signature of render space +
population bounds + the V2 vector-cache version, the file's `vector_dtype`
stamp, and a per-neuron `(mtime_ns, size)` of the backing skeleton. The
hemisphere rides along because a stored vector only lets the scorer skip the
render when the side is answerable without it.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from comparison import morph_cross_dataset as mcd  # noqa: E402
from morphology import VECTOR_V2_DIM as DIM  # noqa: E402  (256 today)


def _vec(seed):
    rng = np.random.default_rng(seed)
    return rng.random(DIM)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A store over a fake dataset whose provenance is a dict the test owns."""
    monkeypatch.setattr(mcd, '_scoring_bounds', lambda ds, root=None: None)
    prov = {}

    def _build():
        st = mcd.TargetVectorStore('np:v1', project_root=str(tmp_path))
        monkeypatch.setattr(
            st, '_provenance',
            lambda bid: prov.get(int(bid), (111, 1000)))
        return st, prov

    def make():
        """A fresh store over the same file — what the NEXT run builds."""
        return _build()

    return make


def test_vectors_and_sides_survive_a_round_trip(store):
    st, _prov = store()
    vectors = {100 + i: _vec(i) for i in range(9)}
    sides = {100 + i: ('left' if i % 2 else 'right') for i in range(9)}
    st.update(vectors, sides)
    assert st.stats['saved'] == 9

    again, _ = store()
    got_v, got_s = again.load()
    assert set(got_v) == set(vectors)
    assert got_s == sides
    # a reused vector IS the vector the run computed: the store may make the
    # next run faster and must never make it disagree
    assert np.array_equal(got_v[103], vectors[103])
    assert got_v[103].dtype == np.float64
    with np.load(st.path, allow_pickle=False) as z:
        assert z['matrix'].dtype == np.float64
        assert str(z['vector_dtype']) == 'float64'


def test_a_small_batch_is_not_worth_a_file(store):
    st, _ = store()
    st.update({1: _vec(1)}, {1: 'left'})
    assert not st.path.exists()


def test_a_moved_skeleton_drops_its_row_and_says_so(store):
    st, prov = store()
    st.update({100 + i: _vec(i) for i in range(9)},
              {100 + i: 'left' for i in range(9)})
    # one neuron was healed underneath the store: its row must not be reused
    prov[104] = (999, 2048)
    again, _ = store()
    got_v, got_s = again.load()
    assert 104 not in got_v
    assert len(got_v) == 8
    assert again.stats['stale_dropped'] == 1
    assert again.stats['loaded'] == 8


def test_a_changed_frame_signature_drops_the_whole_file(store, monkeypatch):
    st, _ = store()
    st.update({100 + i: _vec(i) for i in range(9)},
              {100 + i: 'right' for i in range(9)})
    again, _ = store()
    assert len(again.load()[0]) == 9
    # the population bounds moved: every stored vector describes the old frame
    monkeypatch.setattr(again, '_bounds_signature', lambda: 'other')
    assert again.load()[0] == {}


def test_a_row_without_a_resolvable_skeleton_is_still_usable(store):
    """A FAFB target is served from the release bundle, so no file backs the
    row: it is stored with the unverifiable stamp and trusted on the file
    signature alone, which is what this store did before per-row provenance."""
    st, prov = store()
    for i in range(9):
        prov[100 + i] = (-1, -1)
    st.update({100 + i: _vec(i) for i in range(9)},
              {100 + i: 'left' for i in range(9)})
    again, _ = store()
    got_v, _ = again.load()
    assert len(got_v) == 9
    assert again.stats['stale_dropped'] == 0


def test_a_store_written_before_sides_are_not_half_reused(store, tmp_path):
    """The legacy shape (no `sides`, no provenance) cannot prove a row
    current, so every row is reported stale rather than quietly trusted."""
    st, _ = store()
    st.dir.mkdir(parents=True, exist_ok=True)
    np.savez(st.path,
             bodyIds=np.array([1, 2, 3], dtype=np.int64),
             matrix=np.ones((3, DIM), dtype=np.float64),
             vector_dtype=np.array('float64'),
             space=np.array(st.space),
             bounds_sig=np.array(st._bounds_signature()))
    got_v, got_s = st.load()
    assert got_v == {} and got_s == {}
    assert st.stats['stale_dropped'] == 3


def test_a_reduced_precision_sidecar_is_refused_whole(store, tmp_path):
    """A float32 file was once written here, and the parity gate caught it:
    every score it served moved in the 8th decimal, which is a re-grading
    wearing a cache. The file is refused as a unit, and nothing is loaded."""
    st, _ = store()
    st.dir.mkdir(parents=True, exist_ok=True)
    np.savez(st.path,
             bodyIds=np.array([1, 2, 3], dtype=np.int64),
             matrix=np.ones((3, DIM), dtype=np.float32),
             sides=np.array(['left', 'left', 'left'], dtype='U8'),
             mtimes=np.array([111, 111, 111], dtype=np.int64),
             sizes=np.array([1000, 1000, 1000], dtype=np.int64),
             vector_dtype=np.array('float32'),
             space=np.array(st.space),
             bounds_sig=np.array(st._bounds_signature()))
    got_v, got_s = st.load()
    assert got_v == {} and got_s == {}
    assert st.stats == {'loaded': 0, 'stale_dropped': 0, 'saved': 0}
    # a store that predates the stamp is refused the same way: no dtype
    # claim, no trust
    st.path.unlink()
    np.savez(st.path,
             bodyIds=np.array([1, 2, 3], dtype=np.int64),
             matrix=np.ones((3, DIM), dtype=np.float64),
             sides=np.array(['left'] * 3, dtype='U8'),
             mtimes=np.array([111] * 3, dtype=np.int64),
             sizes=np.array([1000] * 3, dtype=np.int64),
             space=np.array(st.space),
             bounds_sig=np.array(st._bounds_signature()))
    assert st.load()[0] == {}
