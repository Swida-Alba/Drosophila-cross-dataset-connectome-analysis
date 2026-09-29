"""Hermetic tests for the Morphology Comparison backend.

The vector path is exercised through a fake SkeletonVectorCacheV2 (synthetic
256-dim rows, identity whitening) so no dataset cache or network is needed;
the NBLAST path runs with a stubbed NBlaster and fake dotprops. The heatmap
renderers are replaced so the folder contract can be asserted without
VisPath or a browser.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import morphology_comparison as mc  # noqa: E402

DIM = 256


# ----------------------------------------------------------------- fixtures
class _FakeCache:
    """SkeletonVectorCacheV2 stand-in: fixed standardized rows + identity
    whitening; vectors_for is a no-op (everything already cached). Supports
    the fetch contract (`_vectorize_neuron` / `append_vectors`)."""

    def __init__(self, body_ids, X):
        self._ids = list(body_ids)
        self._X = np.asarray(X, dtype=float)
        self.loads = 0
        self.appended: list = []

    def _default_basis(self):
        return "simp90"

    def _vectorize_neuron(self, neuron):
        vec = getattr(neuron, "_vector", None)
        if vec is None:
            raise ValueError("glitchy skeleton")
        return ("skeleton", np.asarray(vec, dtype=float))

    def append_vectors(self, records, vector_basis=None):
        self.appended.extend(records)
        for bid, vec, _rep in records:
            self._ids.append(bid)
            if self._X.size:
                self._X = np.vstack([self._X, np.asarray(vec)[None, :]])
            else:
                self._X = np.asarray(vec)[None, :]

    def load(self):
        if not len(self._ids):
            return None  # models "no cache on disk"
        self.loads += 1
        return {
            "meta": {},
            "df": None,
            "raw": self._X.copy(),
            "X": self._X.copy(),
            "bodyIds": list(self._ids),
            "types": [""] * len(self._ids),
            "instances": [""] * len(self._ids),
            "rep": [""] * len(self._ids),
            "dataset_rep": "skeleton",
            "whiten": np.eye(self._X.shape[1]),
        }

    def vectors_for(self, body_ids, compute_missing=True):
        return (np.zeros((len(body_ids), self._X.shape[1])),
                np.ones(len(body_ids), dtype=bool),
                [""] * len(body_ids))


class _FakeDotprop:
    def __init__(self, bid):
        self.bid = bid


class _FakeNBlaster:
    """Deterministic pairwise scorer: score(a, b) = 1 - d / 10 where d is
    the numeric distance between the two ids (symmetric, self = 1)."""

    def __init__(self, use_alpha=False, normalized=True, progress=False):
        self.handles = []

    def calc_self_hit(self, dp):
        return 1.0

    def append(self, dp, self_hit=None):
        self.handles.append(dp)
        return dp

    def single_query_target(self, a, b, scores="forward"):
        return float(1.0 - abs(a.bid - b.bid) / 10.0)


class _FakeHelper:
    """MorphologyComparer stand-in supplying dotprops for the NBLAST path."""

    def __init__(self, available):
        self._available = set(available)

    def _dotprops_for_ids(self, body_ids, neurons=None, desc=""):
        return {
            int(b): (_FakeDotprop(int(b))
                     if int(b) in self._available else None)
            for b in body_ids
        }


def _install_vector_cache(monkeypatch, body_ids, X):
    cache = _FakeCache(body_ids, X)

    def _factory(dataset, project_root=None, n_workers=8, verbose=True):
        return cache

    monkeypatch.setattr(mc, "find_similar_dataset_cache_v2", _factory)
    return cache


def _install_type_map(monkeypatch, type_map, instance_map=None):
    monkeypatch.setattr(
        mc, "_load_neuron_type_map",
        lambda dataset, project_root=None: (
            type_map, instance_map or {b: f"inst{b}" for b in type_map}))


def _read_matrix(path: Path) -> pd.DataFrame:
    """Read a matrix CSV with str labels (numeric ids would otherwise be
    parsed back as integers)."""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def _lbl(bid, type_name=None):
    """bodyId matrix axis label under the fake maps.

    male-cns (NeuPrint-style) resolves '{bid}_inst{bid}' from the fake
    instance map; FlyWire (local release) resolves '{bid}_{type}' from the
    fake type map (the fixtures carry no side information).
    """
    if type_name is None:
        return f"{bid}_inst{bid}"
    return f"{bid}_{type_name}"


@pytest.fixture
def vector_setup(monkeypatch, tmp_path):
    """Three types with deterministic vector rows:
    aMe12 (2 identical members), aMe10 (2 orthogonal members), PPL1* (1)."""
    ids = [1, 2, 3, 4, 5]
    eye = np.eye(DIM)
    X = np.zeros((5, DIM))
    X[0] = eye[0]
    X[1] = eye[0]           # aMe12 members identical -> cohesion 1.0
    X[2] = eye[1]
    X[3] = eye[2]           # aMe10 members orthogonal -> cohesion 0.0
    X[4] = eye[3]           # PPL1* single member
    _install_vector_cache(monkeypatch, ids, X)
    _install_type_map(monkeypatch, {
        1: "aMe12", 2: "aMe12", 3: "aMe10", 4: "aMe10", 5: "PPL1*",
    })
    return tmp_path


def _comparer(tmp_path, **kw):
    kw.setdefault("generate_heatmaps", False)
    kw.setdefault("verbose", False)
    return mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0",
        query=["aMe12", "aMe10", "PPL1*"],
        output_dir=str(tmp_path),
        **kw,
    )


# ------------------------------------------------------------- vector path
def test_run_writes_folder_contract(vector_setup):
    comparer = _comparer(vector_setup)
    result = comparer.run()
    out = Path(result["output_folder"])

    assert (out / "parameters.json").exists()
    assert (out / "README.txt").exists()
    assert (out / "report.html").exists()
    assert (out / "members.csv").exists()
    type_csv = out / "type_level" / "type_similarity_vector_v2.csv"
    body_csv = out / "bodyid_level" / "bodyid_similarity_vector_v2.csv"
    assert type_csv.exists() and body_csv.exists()

    type_df = _read_matrix(type_csv)
    assert list(type_df.index) == ["aMe12", "aMe10", "PPL1*"]
    assert type_df.shape == (3, 3)
    # Symmetric with unit diagonal (single-member type is trivially 1.0).
    assert np.allclose(type_df.values, type_df.values.T)
    assert type_df.loc["aMe12", "aMe12"] == pytest.approx(1.0)
    assert type_df.loc["PPL1*", "PPL1*"] == pytest.approx(1.0)
    # Identical members score 1; orthogonal blocks score 0.
    assert type_df.loc["aMe12", "aMe10"] == pytest.approx(0.0)
    assert type_df.loc["aMe12", "PPL1*"] == pytest.approx(0.0)

    body_df = _read_matrix(body_csv)
    assert body_df.shape == (5, 5)
    assert body_df.iloc[0, 1] == pytest.approx(1.0)
    assert body_df.iloc[0, 2] == pytest.approx(0.0)
    assert result["rows_compared"] == 3
    assert result["neurons_compared"] == 5


def test_type_level_is_mean_of_cross_member_pairs(vector_setup):
    result = _comparer(vector_setup).run()
    out = Path(result["output_folder"])
    type_df = _read_matrix(
        out / "type_level" / "type_similarity_vector_v2.csv")
    body_df = _read_matrix(
        out / "bodyid_level" / "bodyid_similarity_vector_v2.csv")

    cross = body_df.loc[[_lbl("1"), _lbl("2")], [_lbl("3"), _lbl("4")]].values
    assert type_df.loc["aMe12", "aMe10"] == pytest.approx(cross.mean())
    # Diagonal cohesion = mean over the off-diagonal member pairs.
    assert type_df.loc["aMe10", "aMe10"] == pytest.approx(
        body_df.loc[_lbl("3"), _lbl("4")])


def test_bodyid_query_resolves_to_its_type(vector_setup):
    comparer = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["1", "aMe10"],
        output_dir=str(vector_setup), generate_heatmaps=False, verbose=False)
    result = comparer.run()
    type_df = _read_matrix(
        Path(result["output_folder"]) / "type_level"
        / "type_similarity_vector_v2.csv")
    assert list(type_df.index) == ["aMe12", "aMe10"]


def test_pattern_query_expands_types(vector_setup):
    comparer = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["aMe.*"],
        output_dir=str(vector_setup), generate_heatmaps=False, verbose=False)
    result = comparer.run()
    type_df = _read_matrix(
        Path(result["output_folder"]) / "type_level"
        / "type_similarity_vector_v2.csv")
    assert set(type_df.index) == {"aMe12", "aMe10"}


def test_missing_vector_neuron_is_reported_not_scored(
        vector_setup, monkeypatch):
    ids = [1, 2, 3, 4]
    X = np.zeros((4, DIM))
    X[0] = X[1] = np.eye(DIM)[0]
    X[2] = np.eye(DIM)[1]
    X[3] = np.eye(DIM)[2]
    cache = _FakeCache(ids, X)
    monkeypatch.setattr(
        mc, "find_similar_dataset_cache_v2",
        lambda dataset, project_root=None, n_workers=8, verbose=True: cache)

    # fetch_online=False: the strictly offline mode keeps the neuron
    # reported as `no vector` instead of pulling it from the API.
    comparer = _comparer(vector_setup, fetch_online=False)
    result = comparer.run()
    members = pd.read_csv(Path(result["output_folder"]) / "members.csv")
    row5 = members[members["bodyId"].astype(str) == "5"]
    assert row5["status"].iloc[0] == "no vector"
    # The other neurons still compare; the missing one carries NaN pairs.
    body_df = _read_matrix(
        Path(result["output_folder"]) / "bodyid_level"
        / "bodyid_similarity_vector_v2.csv")
    assert body_df.loc[_lbl("1"), _lbl("2")] == pytest.approx(1.0)
    assert pd.isna(body_df.loc[_lbl("5"), _lbl("1")])


def test_fetch_online_pulls_missing_neurons(vector_setup, monkeypatch):
    """Missing neurons are fetched through the API by default, vectorized
    with the cache's own vectorizer, and appended to the cache."""
    ids = [1, 2, 3, 4]
    X = np.zeros((4, DIM))
    X[0] = X[1] = np.eye(DIM)[0]
    X[2] = np.eye(DIM)[1]
    X[3] = np.eye(DIM)[2]
    cache = _FakeCache(ids, X)
    monkeypatch.setattr(
        mc, "find_similar_dataset_cache_v2",
        lambda dataset, project_root=None, n_workers=8, verbose=True: cache)
    monkeypatch.setattr(mc, "_neuron_rep", lambda n: "skeleton")

    fetch_calls = []

    class _Fetched:
        _vector = np.eye(DIM)[0]  # identical to aMe12's members

    def _fake_fetch(dataset, body_ids, **kw):
        fetch_calls.append((dataset, list(body_ids), kw))
        assert kw.get("raw_cache") is cache
        return {5: _Fetched()}

    monkeypatch.setattr(mc, "fetch_skeletons_on_demand_batch", _fake_fetch)

    result = _comparer(vector_setup).run()  # fetch_online defaults True
    assert fetch_calls and fetch_calls[0][1] == [5]
    assert cache.appended and cache.appended[0][0] == 5

    members = pd.read_csv(Path(result["output_folder"]) / "members.csv")
    row5 = members[members["bodyId"].astype(str) == "5"]
    assert row5["status"].iloc[0] == "compared"
    # The fetched neuron scores against the cached ones (vector_v2 path
    # re-loads the cache after the fetch).
    body_df = _read_matrix(
        Path(result["output_folder"]) / "bodyid_level"
        / "bodyid_similarity_vector_v2.csv")
    assert body_df.loc[_lbl("5"), _lbl("1")] == pytest.approx(1.0)
    assert result["neurons_compared"] == 5


def test_fetch_offline_never_calls_api(vector_setup, monkeypatch):
    ids = [1, 2, 3, 4]
    X = np.zeros((4, DIM))
    X[0] = X[1] = np.eye(DIM)[0]
    X[2] = np.eye(DIM)[1]
    X[3] = np.eye(DIM)[2]
    cache = _FakeCache(ids, X)
    monkeypatch.setattr(
        mc, "find_similar_dataset_cache_v2",
        lambda dataset, project_root=None, n_workers=8, verbose=True: cache)

    def _must_not_fetch(*a, **kw):
        raise AssertionError("API fetch must not run with fetch_online=False")

    monkeypatch.setattr(mc, "fetch_skeletons_on_demand_batch",
                        _must_not_fetch)
    result = _comparer(vector_setup, fetch_online=False).run()
    assert result["neurons_compared"] == 4


def test_flywire_fetch_uses_bundle_loader(monkeypatch, tmp_path):
    """FAFB resolves missing skeletons through the shared
    bundle/CAVE loader, never the NeuPrint batch fetch."""
    ids = [1, 2, 3]
    X = np.zeros((3, DIM))
    X[0] = np.eye(DIM)[0]
    X[1] = np.eye(DIM)[1]
    X[2] = np.eye(DIM)[2]
    cache = _FakeCache(ids, X)
    monkeypatch.setattr(
        mc, "find_similar_dataset_cache_v2",
        lambda dataset, project_root=None, n_workers=8, verbose=True: cache)
    monkeypatch.setattr(mc, "is_fafb_dataset", lambda d: True)
    monkeypatch.setattr(mc, "_neuron_rep", lambda n: "skeleton")

    loader_calls = []

    class _Fetched:
        _vector = np.eye(DIM)[3]

    def _fake_loader(dataset, body_ids, **kw):
        loader_calls.append((dataset, list(body_ids)))
        return {4: _Fetched()}

    monkeypatch.setattr(mc, "load_local_release_skeletons", _fake_loader)

    def _must_not_fetch(*a, **kw):
        raise AssertionError("NeuPrint fetch must not run for FlyWire")

    monkeypatch.setattr(mc, "fetch_skeletons_on_demand_batch",
                        _must_not_fetch)
    _install_type_map(monkeypatch, {1: "aMe12", 2: "aMe10", 3: "aMe5",
                                    4: "aMe13"})

    comparer = mc.MorphologyProfileComparer(
        dataset="flywire_FAFB_v783", query=["aMe12", "aMe10", "aMe13"],
        output_dir=str(tmp_path), generate_heatmaps=False, verbose=False)
    result = comparer.run()
    assert loader_calls and loader_calls[0][1] == [4]
    body_df = _read_matrix(
        Path(result["output_folder"]) / "bodyid_level"
        / "bodyid_similarity_vector_v2.csv")
    assert pd.notna(body_df.loc[_lbl("4", "aMe13"), _lbl("1", "aMe12")])


def test_vector_whitening_applied(monkeypatch, tmp_path):
    """Rows are whitened with the cache's whitener before scoring (a
    uniform rescaling leaves every cosine unchanged)."""

    class _ScaledWhitenCache(_FakeCache):
        def load(self):
            data = super().load()
            data["whiten"] = 2.0 * np.eye(DIM)
            return data

    ids = [1, 2, 3, 4, 5]
    X = np.zeros((5, DIM))
    X[0] = X[1] = np.eye(DIM)[0]
    X[2] = np.eye(DIM)[1]
    X[3] = np.eye(DIM)[2]
    X[4] = np.eye(DIM)[3]
    monkeypatch.setattr(
        mc, "find_similar_dataset_cache_v2",
        lambda dataset, project_root=None, n_workers=8, verbose=True:
        _ScaledWhitenCache(ids, X))
    _install_type_map(monkeypatch, {
        1: "aMe12", 2: "aMe12", 3: "aMe10", 4: "aMe10", 5: "PPL1*",
    })

    result = _comparer(tmp_path).run()
    body_df = _read_matrix(
        Path(result["output_folder"]) / "bodyid_level"
        / "bodyid_similarity_vector_v2.csv")
    # Whitening rescales every row identically -> cosine scores unchanged.
    assert body_df.loc[_lbl("1"), _lbl("2")] == pytest.approx(1.0)
    assert body_df.loc[_lbl("1"), _lbl("3")] == pytest.approx(0.0)


def test_no_vectors_raises(monkeypatch, tmp_path):
    """No cache and no local skeletons (offline mode) -> a clear error."""
    _install_vector_cache(monkeypatch, [], np.zeros((0, DIM)))
    monkeypatch.setattr(mc, "_load_neuron_type_map",
                        lambda d, p=None: ({1: "aMe12", 2: "aMe10"},
                                           {1: "i1", 2: "i2"}))
    comparer = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["aMe12", "aMe10"],
        output_dir=str(tmp_path), generate_heatmaps=False, verbose=False,
        fetch_online=False)
    with pytest.raises(ValueError, match="No morphology vector cache"):
        comparer.run()


# ------------------------------------------------------------- nblast path
def test_nblast_matrix_symmetric_and_capped(monkeypatch, tmp_path):
    _install_type_map(monkeypatch, {
        1: "aMe12", 2: "aMe12", 3: "aMe10", 4: "aMe10",
    })
    monkeypatch.setattr(
        mc, "MorphologyComparer",
        lambda **kw: _FakeHelper([1, 2, 3, 4]))
    import navis.nbl.nblast_funcs as nblast_funcs
    monkeypatch.setattr(nblast_funcs, "NBlaster", _FakeNBlaster)

    comparer = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["aMe12", "aMe10"], method="nblast",
        output_dir=str(tmp_path), generate_heatmaps=False, verbose=False)
    result = comparer.run()
    body_df = _read_matrix(
        Path(result["output_folder"]) / "bodyid_level"
        / "bodyid_similarity_nblast.csv")
    assert np.allclose(body_df.values, body_df.values.T)
    assert body_df.iloc[0, 0] == pytest.approx(1.0)
    # score = 1 - |a - b| / 10
    assert body_df.loc[_lbl("1"), _lbl("3")] == pytest.approx(0.8)
    # Type entry = mean over the 2x2 cross-member block:
    # (1,3)=0.8, (1,4)=0.7, (2,3)=0.9, (2,4)=0.8.
    type_df = _read_matrix(
        Path(result["output_folder"]) / "type_level"
        / "type_similarity_nblast.csv")
    assert type_df.loc["aMe12", "aMe10"] == pytest.approx(0.8)


def test_nblast_warns_past_30_neurons_and_runs(monkeypatch, tmp_path):
    """The NBLAST population bound is a disclosed warning, not a refusal
    (user 2026-09-29): 40 single-member types run to completion with the
    warning in user_warning_notes.txt."""
    type_map = {i: f"T{i}" for i in range(40)}
    _install_nblast(monkeypatch, type_map)
    result = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=[f"T{i}" for i in range(40)],
        method="nblast", output_dir=str(tmp_path),
        generate_heatmaps=False, verbose=False).run()
    body_df = _read_matrix(
        Path(result["output_folder"]) / "bodyid_level"
        / "bodyid_similarity_nblast.csv")
    assert body_df.shape == (40, 40)
    notes = (Path(result["output_folder"])
             / "user_warning_notes.txt").read_text(encoding="utf-8")
    assert "40 neurons is past the 30-neuron comfort bound" in notes


def test_nblast_missing_dotprops_reported(monkeypatch, tmp_path):
    _install_type_map(monkeypatch, {1: "aMe12", 2: "aMe10", 3: "aMe10"})
    # bodyId 1 has no dotprops (no skeleton available).
    monkeypatch.setattr(
        mc, "MorphologyComparer", lambda **kw: _FakeHelper([2, 3]))
    import navis.nbl.nblast_funcs as nblast_funcs
    monkeypatch.setattr(nblast_funcs, "NBlaster", _FakeNBlaster)

    comparer = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["aMe12", "aMe10"], method="nblast",
        output_dir=str(tmp_path), generate_heatmaps=False, verbose=False)
    result = comparer.run()
    members = pd.read_csv(Path(result["output_folder"]) / "members.csv")
    status = dict(zip(members["bodyId"].astype(str), members["status"]))
    assert status["1"] == "no dotprops"
    assert status["2"] == "compared"


def test_nblast_contra_pairs_excluded_from_type_means(monkeypatch, tmp_path):
    _install_type_map(monkeypatch, {1: "aMe12", 2: "aMe10", 3: "aMe10"})
    monkeypatch.setattr(
        mc, "MorphologyComparer", lambda **kw: _FakeHelper([1, 2, 3]))
    import navis.nbl.nblast_funcs as nblast_funcs
    monkeypatch.setattr(nblast_funcs, "NBlaster", _FakeNBlaster)
    # bodyId 3 is on the opposite side from 1 and 2.
    monkeypatch.setattr(
        mc, "_dataset_soma_side_map",
        lambda dataset, project_root=None: {1: "left", 2: "left", 3: "right"})

    comparer = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["aMe12", "aMe10"], method="nblast",
        output_dir=str(tmp_path), generate_heatmaps=False, verbose=False)
    result = comparer.run()
    type_df = _read_matrix(
        Path(result["output_folder"]) / "type_level"
        / "type_similarity_nblast.csv")
    # Only the ipsilateral pair 1x2 (0.9) contributes to the aMe12-aMe10
    # entry; the contra pairs 1x3 (0.8) are excluded.
    assert type_df.loc["aMe12", "aMe10"] == pytest.approx(0.9)
    # ... but the bodyId matrix keeps every pair for inspection.
    body_df = _read_matrix(
        Path(result["output_folder"]) / "bodyid_level"
        / "bodyid_similarity_nblast.csv")
    assert body_df.loc[_lbl("1"), _lbl("3")] == pytest.approx(0.8)


# ------------------------------------------------------------ guards/input
def test_one_type_with_two_neurons_runs(vector_setup):
    """The population gate counts NEURONS, not rows: one type with two members
    is a legitimate comparison whose aggregate cell is that type's cohesion."""
    result = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["aMe12"],
        output_dir=str(vector_setup), generate_heatmaps=False,
        verbose=False).run()
    assert result["rows_compared"] == 1
    aggregate = _read_matrix(
        Path(result["output_folder"]) / "type_level"
        / "type_similarity_vector_v2.csv")
    assert list(aggregate.index) == ["aMe12"]
    # aMe12's two members are identical vectors: cohesion 1.0
    assert aggregate.loc["aMe12", "aMe12"] == pytest.approx(1.0)
    assert _body_labels(result) == [_lbl(1), _lbl(2)]


def test_one_type_at_bodyid_level_runs(vector_setup):
    """A single type at the bodyId level was never the backend's problem: the
    rows are its neurons, so the pairwise matrix is complete either way."""
    result = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["aMe12"], aggregation_level="bodyid",
        output_dir=str(vector_setup), generate_heatmaps=False,
        verbose=False).run()
    assert result["rows_compared"] == 2
    assert _body_labels(result) == [_lbl(1), _lbl(2)]
    assert not (Path(result["output_folder"]) / "type_level").exists()


def test_single_neuron_query_raises(vector_setup):
    comparer = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["PPL1*"],
        output_dir=str(vector_setup), generate_heatmaps=False, verbose=False)
    with pytest.raises(ValueError, match="at least two neurons"):
        comparer.run()


def test_bodyid_cap_of_one_names_the_cap(vector_setup):
    """A refusal the user can act on: the cap, not the query, emptied the run."""
    comparer = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["aMe12"], aggregation_level="bodyid",
        max_members_per_type=1, output_dir=str(vector_setup),
        generate_heatmaps=False, verbose=False)
    with pytest.raises(ValueError, match="Max Members per Type"):
        comparer.run()


def test_empty_query_names_the_query_not_a_min_count(vector_setup):
    comparer = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=[],
        output_dir=str(vector_setup), generate_heatmaps=False, verbose=False)
    with pytest.raises(ValueError, match="needs a query"):
        comparer.run()


def test_unknown_bodyid_raises(vector_setup):
    comparer = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["1", "999999"],
        output_dir=str(vector_setup), generate_heatmaps=False, verbose=False)
    with pytest.raises(ValueError, match="999999"):
        comparer.run()


def test_banc_dataset_warns_instead_of_refusing(monkeypatch):
    """BANC comparison runs with the provisional-scores caveat disclosed
    (user 2026-09-29) — the constructor no longer gates it."""
    monkeypatch.setattr(mc, "is_banc_dataset", lambda d: True)
    comparer = mc.MorphologyProfileComparer(
        dataset="banc_v626", query=["a", "b"], verbose=False)
    assert any("provisional" in note and "BANC" in note
               for note in comparer._resolution_notes)
    # The disclosure rides into the run's notes file with everything else.
    assert comparer._resolution_notes[-1].startswith("BANC morphology")


def test_invalid_method_rejected():
    with pytest.raises(ValueError, match="Invalid method"):
        mc.MorphologyProfileComparer(
            dataset="male-cns:v1.0", query=["a", "b"], method="cosine")


def test_output_dir_defaults_under_project_root(vector_setup, monkeypatch):
    monkeypatch.setattr(mc, "_load_neuron_type_map",
                        lambda d, p=None: ({1: "aMe12", 3: "aMe10"},
                                           {1: "i1", 3: "i2"}))
    cache = _FakeCache([1, 3], np.eye(DIM)[:2])
    monkeypatch.setattr(
        mc, "find_similar_dataset_cache_v2",
        lambda dataset, project_root=None, n_workers=8, verbose=True: cache)
    comparer = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["aMe12", "aMe10"],
        generate_heatmaps=False, verbose=False,
        project_root=str(vector_setup))
    result = comparer.run()
    assert str(vector_setup) in result["output_folder"]
    assert "morphology_comparison" in Path(result["output_folder"]).name


def test_heatmap_fallback_writes_files(vector_setup, monkeypatch):
    """When VisPath is unavailable the kit's plotly fallback renders both
    heatmaps."""
    monkeypatch.setitem(sys.modules, "vispath_pkg", None)
    monkeypatch.setitem(sys.modules, "vispath_pkg.vispath", None)

    result = _comparer(vector_setup, generate_heatmaps=True).run()
    out = Path(result["output_folder"])
    assert (out / "visualization" / "heatmap_type_vector_v2.html").exists()
    assert (out / "visualization" / "heatmap_bodyid_vector_v2.html").exists()
    # The fallback renderer's output is a full interactive heatmap page.
    html = (out / "visualization" /
            "heatmap_bodyid_vector_v2.html").read_text(encoding="utf-8")
    assert "plotly" in html.lower()


def test_completion_marker_printed(vector_setup, capsys):
    comparer = _comparer(vector_setup, verbose=True)
    result = comparer.run()
    captured = capsys.readouterr().out
    assert f"[MorphologyProfileComparer] Output: {result['output_folder']}" \
        in captured


# ------------------------------------------------------- BANC vector fetch
def test_banc_missing_vectors_route_to_public_swc_chain(monkeypatch):
    """BANC must never enter the FAFB/CAVE fetch machinery: the vector
    cache's missing bodies resolve through the shared batch fetch, whose
    BANC branch uses the official public-bucket SWCs (fetch_banc_swc).

    Drives _fetch_missing_vectors on a stub instance because the comparer
    constructor deliberately defers BANC comparison (vector-quality
    validation pending); the routing contract must hold when that lifts.
    """
    from types import SimpleNamespace

    calls = {}

    def _batch(dataset, body_ids, **kw):
        calls["batch"] = (dataset, list(body_ids))
        # skeleton-rep stand-in: _neuron_rep only checks .nodes
        return {bid: SimpleNamespace(nodes=[object()]) for bid in body_ids}

    def _fail_fafb(*_a, **_kw):
        raise AssertionError("BANC must not use the FAFB healed-bundle loader")

    monkeypatch.setattr(mc, "fetch_skeletons_on_demand_batch", _batch)
    monkeypatch.setattr(mc, "load_local_release_skeletons", _fail_fafb)

    stub = SimpleNamespace(
        dataset="banc_v888",
        project_root=Path("."),
        n_workers=2,
        _body_id=lambda b: int(b),
        _log=lambda *a, **k: None,
    )
    cache = SimpleNamespace(
        _vectorize_neuron=lambda neuron: ("raw", np.zeros(DIM)),
        _default_basis=lambda: "raw",
        append_vectors=lambda rows, vector_basis=None: calls.update(
            rows=len(rows)),
    )
    fetched = mc.MorphologyProfileComparer._fetch_missing_vectors(
        stub, cache, [1001, 1002])

    assert fetched == 2
    assert calls["batch"] == ("banc_v888", [1001, 1002])
    assert calls["rows"] == 2


# -------------------------------------------------------------- 3d scene
class _FakeVisualizer:
    """VisualizeSkeleton stand-in that records kwargs and writes a scene."""

    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeVisualizer.instances.append(self)

    def plot_neurons(self):
        out_dir = Path(self.kwargs["output_dir"])
        folder = out_dir / f"plot-3d_{self.kwargs['saveas']}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{self.kwargs['saveas']}.html").write_text(
            "<html>scene</html>", encoding="utf-8")


def test_visualize_members_renders_one_layer_per_type(
        monkeypatch, vector_setup):
    _FakeVisualizer.instances = []
    monkeypatch.setattr(mc, "_import_visualizer", lambda: _FakeVisualizer)
    comparer = _comparer(vector_setup, visualize=True)
    result = comparer.run()

    assert len(_FakeVisualizer.instances) == 1
    kwargs = _FakeVisualizer.instances[0].kwargs
    assert kwargs["neuron_layers"] == [[1, 2], [3, 4], [5]]
    assert kwargs["custom_layer_names"] == [
        "t1_aMe12_x2", "t2_aMe10_x2", "t3_PPL1_x1"]
    # the exported comparison scenes use the interactive tree legend
    assert kwargs["legend_mode"] == "tree"
    assert kwargs["skip_synapse"] is True
    assert kwargs["include_timestamp"] is False
    assert kwargs["export_views"] is False
    assert kwargs["show_fig"] is False
    assert kwargs["output_dir"] == result["output_folder"]

    # The rendered scene is discovered and linked from the report.
    out = Path(result["output_folder"])
    assert (out / "plot-3d_male-cns_v1_0" / "male-cns_v1_0.html").exists()
    report = (out / "report.html").read_text(encoding="utf-8")
    assert "plot-3d_male-cns_v1_0/male-cns_v1_0.html" in report


def test_visualize_disabled_by_default(monkeypatch, vector_setup):
    _FakeVisualizer.instances = []
    monkeypatch.setattr(mc, "_import_visualizer", lambda: _FakeVisualizer)
    _comparer(vector_setup).run()
    assert _FakeVisualizer.instances == []


def test_visualization_failure_does_not_fail_run(monkeypatch, vector_setup):
    def _boom():
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(mc, "_import_visualizer", _boom)
    result = _comparer(vector_setup, visualize=True).run()
    assert result["neurons_compared"] == 5
    out = Path(result["output_folder"])
    assert (out / "report.html").exists()
    assert "3D skeleton visualization" not in (
        out / "report.html").read_text(encoding="utf-8")


def test_visualize_members_capped_per_type(monkeypatch, tmp_path):
    """Type layers sample at most TYPE_RENDER_MEMBER_CAP members."""
    _FakeVisualizer.instances = []
    monkeypatch.setattr(mc, "_import_visualizer", lambda: _FakeVisualizer)
    ids = list(range(1, 31))
    eye = np.eye(DIM)
    X = np.zeros((31, DIM))
    for i in range(31):
        X[i] = eye[i % DIM]
    type_map = {i: "Big" for i in ids}
    type_map[31] = "Tiny"
    _install_vector_cache(monkeypatch, ids + [31], X)
    _install_type_map(monkeypatch, type_map)
    comparer = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["Big", "Tiny"],
        max_members_per_type=50,
        output_dir=str(tmp_path), generate_heatmaps=False, verbose=False,
        visualize=True)
    result = comparer.run()

    kwargs = _FakeVisualizer.instances[0].kwargs
    assert len(kwargs["neuron_layers"]) == 2
    layer = kwargs["neuron_layers"][0]
    assert len(layer) == mc.TYPE_RENDER_MEMBER_CAP
    notes = kwargs["layer_sample_notes"]
    assert notes and "30" in notes[0]
    assert result["neurons_compared"] == 31


# ---------------------------------------------------------------------------
# offline 3D-scene filter (fetch_online=False unifies with the cross mode)
# ---------------------------------------------------------------------------

class TestOfflineRenderFilter:
    def _comparer(self, tmp_path, fetch_online):
        return mc.MorphologyProfileComparer(
            dataset="male-cns:v1.0", query=["aMe12", "aMe10"],
            fetch_online=fetch_online, output_dir=str(tmp_path),
            generate_heatmaps=False, verbose=False, visualize=True,
            project_root=str(tmp_path))

    def _write_raw(self, tmp_path, body_id):
        import pickle

        import navis

        nodes = pd.DataFrame({
            "node_id": [0, 1, 2],
            "parent_id": [-1, 0, 0],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "radius": [1.0, 1.0, 1.0],
        })
        folder = (tmp_path / "cache" / "male-cns_v1_0" / "skeletons"
                  / "raw_skeletons")
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / f"{body_id}.pkl", "wb") as fh:
            pickle.dump(navis.TreeNeuron(nodes), fh)

    def test_online_mode_keeps_everything(self, tmp_path):
        comparer = self._comparer(tmp_path, fetch_online=True)
        members = {"aMe12": [1, 2]}
        filtered, skipped = comparer._offline_render_filter(members)
        assert filtered == members and skipped == []

    def test_offline_mode_drops_uncached_members(self, tmp_path):
        self._write_raw(tmp_path, 1)
        # a legacy simp90-level file is a valid load but not a renderable
        # source (VisualizeSkeleton refuses to mix simplification levels)
        import pickle

        import navis

        nodes = pd.DataFrame({
            "node_id": [0, 1], "parent_id": [-1, 0],
            "x": [0.0, 1.0], "y": [0.0, 0.0], "z": [0.0, 0.0],
            "radius": [1.0, 1.0],
        })
        stale = navis.TreeNeuron(nodes)
        stale._drocat_simplification = 90
        folder = (tmp_path / "cache" / "male-cns_v1_0" / "skeletons"
                  / "raw_skeletons")
        with open(folder / "4.pkl", "wb") as fh:
            pickle.dump(stale, fh)
        comparer = self._comparer(tmp_path, fetch_online=False)
        members = {"aMe12": [1, 2], "aMe10": [3, 4]}
        filtered, skipped = comparer._offline_render_filter(members)
        assert filtered == {"aMe12": [1]}
        assert sorted(skipped) == ["aMe10:3", "aMe10:4", "aMe12:2"]

    def test_offline_mode_all_types_dropped(self, tmp_path):
        comparer = self._comparer(tmp_path, fetch_online=False)
        filtered, skipped = comparer._offline_render_filter(
            {"aMe12": [7, 8]})
        assert filtered == {} and len(skipped) == 2


# ------------------------------------------------- diagonal pairing alignment
def test_type_level_matrix_diagonal_pairs_align(monkeypatch):
    """A member missing from the body matrix must not shift the diagonal
    subscripts (regression: ids[] was filtered but members[a][ii] was not,
    so NBLAST pair exclusion checked the wrong neurons)."""
    import warnings

    cmp = mc.MorphologyProfileComparer(
        dataset="hemibrain:v1.2.1", method="nblast", verbose=False,
        generate_heatmaps=False)
    # 10(L) and 11(R) are contralateral; 12(L) is missing from the matrix.
    monkeypatch.setattr(
        mc, "_dataset_soma_side_map",
        lambda *a, **k: {10: "left", 11: "right", 12: "left"})

    labels = [10, 11, 99]                      # 12 has no matrix row
    body_matrix = np.array([
        [1.0, 0.9, 0.1],
        [0.8, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ])
    members = {"T": [10, 12, 11]}              # middle member uncached

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = cmp._type_level_matrix(body_matrix, labels, members)

    # The only matrix-present pair (10, 11) is contralateral and must be
    # excluded: cohesion is NaN, never the mean of matrix[0,1]/[1,0].
    assert np.isnan(out.loc["T", "T"])


# ------------------------------------------------------------ aggregation levels
def _body_labels(result, method="vector_v2"):
    return list(_read_matrix(
        Path(result["output_folder"]) / "bodyid_level"
        / f"bodyid_similarity_{method}.csv").index)


def test_bodyid_level_rows_are_neurons(vector_setup):
    """Two bodyIds of the SAME type become two rows.

    At the type level they fold into one row and the run refuses, so
    'compare these two neurons' was previously impossible.
    """
    comparer = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["1", "2"],
        aggregation_level="bodyid", output_dir=str(vector_setup),
        generate_heatmaps=False, verbose=False)
    result = comparer.run()
    assert _body_labels(result) == ["1_inst1", "2_inst2"]
    assert result["aggregation_level"] == "bodyid"
    assert result["rows_compared"] == 2


def test_type_level_fold_hint_moves_to_the_log_once_it_runs(vector_setup, capsys):
    """Two bodyIds of ONE type fold into one row at the type level: that is a
    run now rather than a refusal, so the level pointer became a log line."""
    result = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["1", "2"],
        aggregation_level="type", output_dir=str(vector_setup),
        generate_heatmaps=False, verbose=True).run()
    assert result["rows_compared"] == 1
    assert _body_labels(result) == [_lbl(1), _lbl(2)]
    assert "Aggregation Level to 'bodyid'" in capsys.readouterr().out


def test_queried_bodyid_survives_the_member_cap(vector_setup):
    """max_members_per_type may drop a type's other members, never the neuron
    the query named (regression: the sorted prefix evicted it silently)."""
    result = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["2", "aMe10"],
        max_members_per_type=1, output_dir=str(vector_setup),
        generate_heatmaps=False, verbose=False).run()
    labels = _body_labels(result)
    assert "2_inst2" in labels
    assert "1_inst1" not in labels
    members = pd.read_csv(Path(result["output_folder"]) / "members.csv")
    assert 2 in {int(b) for b in members["bodyId"]}


def test_several_queried_bodyids_all_survive_a_cap_of_one(vector_setup):
    """The cap bounds UNPINNED members: two queried neurons of one type both
    stay even when max_members_per_type is 1.

    Regression: `(pinned + rest)[:limit]` truncated the pinned run itself, so
    the second named neuron vanished — the same silent loss this whole level
    exists to prevent, just one layer up.
    """
    result = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["1", "2", "aMe10"],
        max_members_per_type=1, output_dir=str(vector_setup),
        generate_heatmaps=False, verbose=False).run()
    labels = _body_labels(result)
    assert {"1_inst1", "2_inst2"} <= set(labels)
    # The unpinned row is still capped to one member.
    assert "3_inst3" in labels and "4_inst4" not in labels


def test_total_cap_never_drops_a_queried_neuron(vector_setup):
    """max_total_neurons fills with unpinned members only, and yields to the
    named neurons when they alone pass the cap."""
    result = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["2", "4"],
        max_total_neurons=1, output_dir=str(vector_setup),
        generate_heatmaps=False, verbose=False).run()
    assert set(_body_labels(result)) == {"2_inst2", "4_inst4"}
    assert result["rows_compared"] == 2


def test_bodyid_level_suppresses_the_type_matrix(vector_setup, monkeypatch):
    """The bodyId matrix IS the comparison: no aggregate is computed, no empty
    type_level/ directory is left behind, and the report offers no tab for a
    level this run cannot have."""
    monkeypatch.setitem(sys.modules, "vispath_pkg", None)
    monkeypatch.setitem(sys.modules, "vispath_pkg.vispath", None)
    result = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["aMe12", "aMe10"],
        aggregation_level="bodyid", output_dir=str(vector_setup),
        generate_heatmaps=True, verbose=False).run()
    out = Path(result["output_folder"])
    assert not (out / "type_level").exists()
    assert (out / "bodyid_level" / "bodyid_similarity_vector_v2.csv").exists()
    assert not list((out / "visualization").glob("heatmap_type_*"))
    assert list((out / "visualization").glob("heatmap_bodyid_*"))
    report = (out / "report.html").read_text(encoding="utf-8")
    assert "BodyId level" in report
    assert "Type level" not in report
    assert "This level was not computed" not in report


def test_members_type_column_stays_true(vector_setup):
    """`row` carries the matrix row, `type` stays the real type — neither
    column may lie about the other (the defect connectivity's bodyId level has)."""
    result = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["1", "3"],
        aggregation_level="bodyid", output_dir=str(vector_setup),
        generate_heatmaps=False, verbose=False).run()
    members = pd.read_csv(Path(result["output_folder"]) / "members.csv")
    by_id = {str(rec["bodyId"]): (rec["row"], rec["type"])
             for rec in members.to_dict("records")}
    assert by_id["1"] == ("1_inst1", "aMe12")
    assert by_id["3"] == ("3_inst3", "aMe10")


def _write_preset(tmp_path, groups, dataset="male-cns:v1.0"):
    import json

    path = tmp_path / "preset.json"
    path.write_text(json.dumps({"source_mapping": {
        "custom_label": [name for name, _ in groups],
        dataset: [members for _, members in groups],
    }}), encoding="utf-8")
    return str(path)


def test_custom_level_rows_are_groups(vector_setup, tmp_path):
    preset = _write_preset(tmp_path, [
        ("early", ["aMe12"]), ("late", ["aMe10", "PPL1*"])])
    result = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["unused"],
        aggregation_level="custom", custom_mapping_file=preset,
        output_dir=str(tmp_path / "out"), generate_heatmaps=False,
        verbose=False).run()
    out = Path(result["output_folder"])
    aggregate = _read_matrix(out / "group_level" / "group_similarity_vector_v2.csv")
    assert list(aggregate.index) == ["early", "late"]
    # The group matrix is filed under a name that says groups; the type×type
    # name would lie about its axes.
    assert not (out / "type_level").exists()
    members = pd.read_csv(out / "members.csv")
    assert set(members["row"]) == {"early", "late"}
    assert set(members["type"]) == {"aMe12", "aMe10", "PPL1*"}
    import json
    params = json.loads((out / "parameters.json").read_text())
    assert params["aggregation_level"] == "custom"


def test_custom_level_heatmap_is_named_for_groups(vector_setup, monkeypatch):
    monkeypatch.setitem(sys.modules, "vispath_pkg", None)
    monkeypatch.setitem(sys.modules, "vispath_pkg.vispath", None)
    preset = _write_preset(vector_setup, [("g1", ["aMe12"]),
                                          ("g2", ["aMe10"])])
    result = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["unused"], aggregation_level="custom",
        custom_mapping_file=preset, output_dir=str(vector_setup / "hm"),
        generate_heatmaps=True, verbose=False).run()
    viz = Path(result["output_folder"]) / "visualization"
    assert (viz / "heatmap_group_vector_v2.html").exists()
    assert (viz / "heatmap_bodyid_vector_v2.html").exists()
    assert not list(viz.glob("heatmap_type_*"))
    assert "group_level/group_similarity_vector_v2.csv" in (
        Path(result["output_folder"]) / "report.html").read_text(
        encoding="utf-8")


def test_custom_mapping_file_forces_custom_level(vector_setup, tmp_path):
    """A preset given without the level is not silently ignored."""
    comparer = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["aMe12", "aMe10"],
        aggregation_level="type",
        custom_mapping_file=_write_preset(tmp_path, [("g1", ["aMe12"])]),
        verbose=False)
    assert comparer.aggregation_level == "custom"


def test_custom_level_without_a_preset_raises():
    with pytest.raises(ValueError, match="custom_mapping_file"):
        mc.MorphologyProfileComparer(
            dataset="male-cns:v1.0", query=["aMe12", "aMe10"],
            aggregation_level="custom", verbose=False).run()


def test_invalid_aggregation_level_raises():
    """Strict where connectivity is lenient: a level the run did not use must
    never be reported as the run's level."""
    with pytest.raises(ValueError, match="aggregation_level"):
        mc.MorphologyProfileComparer(
            dataset="male-cns:v1.0", query=["aMe12", "aMe10"],
            aggregation_level="suspicious", verbose=False)


def test_aggregation_level_alias_and_empty():
    assert mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["a", "b"],
        aggregation_level="Custom Group", verbose=False).aggregation_level \
        == "custom"
    for empty in (None, "", "   "):
        assert mc.MorphologyProfileComparer(
            dataset="male-cns:v1.0", query=["a", "b"],
            aggregation_level=empty, verbose=False).aggregation_level == "type"


# ------------------------------------------------------ NBLAST population cap
def _install_nblast(monkeypatch, type_map):
    _install_type_map(monkeypatch, type_map)
    monkeypatch.setattr(
        mc, "MorphologyComparer",
        lambda **kw: _FakeHelper(sorted(type_map)))
    import navis.nbl.nblast_funcs as nblast_funcs
    monkeypatch.setattr(nblast_funcs, "NBlaster", _FakeNBlaster)


def test_nblast_cap_below_30_is_honoured_not_refused(monkeypatch, tmp_path):
    """15 neurons with max_total_neurons=10 runs.

    Regression: the gate tested max_total_neurons while printing the 30
    bound, so this raised "capped at 30 total neurons (got 15)" and built no
    dotprops at all.
    """
    type_map = {**{i: "T1" for i in range(1, 14)}, 14: "T2", 15: "T3"}
    _install_nblast(monkeypatch, type_map)
    result = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["T1", "T2", "T3"], method="nblast",
        max_total_neurons=10, output_dir=str(tmp_path),
        generate_heatmaps=False, verbose=False).run()
    body_df = _read_matrix(
        Path(result["output_folder"]) / "bodyid_level"
        / "bodyid_similarity_nblast.csv")
    assert body_df.shape == (10, 10)


def test_nblast_warning_names_the_effective_population(monkeypatch, tmp_path):
    """50 neurons with max_total_neurons=20 scores 20 — under the comfort
    bound, so no cost warning; with the default cap the same query warns
    about the full 50 and runs instead of refusing."""
    type_map = {i: f"T{i}" for i in range(1, 51)}
    _install_nblast(monkeypatch, type_map)
    run = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=[f"T{i}" for i in range(1, 51)],
        method="nblast", max_total_neurons=20, output_dir=str(tmp_path),
        generate_heatmaps=False, verbose=False).run()
    assert _read_matrix(
        Path(run["output_folder"]) / "bodyid_level"
        / "bodyid_similarity_nblast.csv").shape == (20, 20)
    capped_notes = (Path(run["output_folder"])
                    / "user_warning_notes.txt").read_text(encoding="utf-8")
    assert "comfort bound" not in capped_notes

    uncapped = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=[f"T{i}" for i in range(1, 51)],
        method="nblast", output_dir=str(tmp_path),
        generate_heatmaps=False, verbose=False)
    result = uncapped.run()
    assert _read_matrix(
        Path(result["output_folder"]) / "bodyid_level"
        / "bodyid_similarity_nblast.csv").shape == (50, 50)
    notes = (Path(result["output_folder"])
             / "user_warning_notes.txt").read_text(encoding="utf-8")
    assert "50 neurons is past the 30-neuron comfort bound" in notes


# ------------------------------------------- taxonomy + instance query lanes
@pytest.fixture(autouse=True)
def _hermetic_taxonomy_resolver(monkeypatch):
    """No test reads the real allneurons tables: the taxonomy lane defaults
    to a resolver that never matches (so pattern/exact-type tests stay fast
    and hermetic); the taxonomy tests below install the real class against
    a synthetic table."""
    class _NullResolver:
        def __init__(self, *a, **k):
            pass

        def resolve(self, token, dataset):
            return None

    monkeypatch.setattr(mc, "DatasetTaxonomyResolver", _NullResolver)


def _install_taxonomy_table(tmp_path, rows, folder="male-cns_v1_0"):
    """Synthetic allneurons table at the canonical dataset folder — the same
    table _load_neuron_type_map reads, so the resolver's scan finds it."""
    ddir = tmp_path / "datasets" / folder
    ddir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        ddir / f"{folder}_allneurons_neuron_df.csv", index=False)


def _taxonomy_setup(tmp_path, monkeypatch):
    """Two clock types under cell_type 'clock_cluster' plus one non-clock
    type; five cached neurons with deterministic vectors."""
    from comparison.query_resolver import DatasetTaxonomyResolver
    monkeypatch.setattr(mc, "DatasetTaxonomyResolver", DatasetTaxonomyResolver)
    ids = [1, 2, 3, 4, 5]
    eye = np.eye(DIM)
    X = np.zeros((5, DIM))
    X[0] = X[1] = eye[0]            # aMe12 members identical
    X[2] = X[3] = eye[1]            # aMe10 members identical
    X[4] = eye[2]                   # PPL1* single member
    _install_vector_cache(monkeypatch, ids, X)
    _install_type_map(monkeypatch, {
        1: "aMe12", 2: "aMe12", 3: "aMe10", 4: "aMe10", 5: "PPL1*",
    })
    _install_taxonomy_table(tmp_path, [
        {"bodyId": 1, "type": "aMe12", "instance": "inst1",
         "cell_type": "clock_cluster"},
        {"bodyId": 2, "type": "aMe12", "instance": "inst2",
         "cell_type": "clock_cluster"},
        {"bodyId": 3, "type": "aMe10", "instance": "inst3",
         "cell_type": "clock_cluster"},
        {"bodyId": 4, "type": "aMe10", "instance": "inst4",
         "cell_type": "clock_cluster"},
        {"bodyId": 5, "type": "PPL1*", "instance": "inst5",
         "cell_type": "other"},
    ])
    return tmp_path


def test_taxonomy_label_expands_to_member_types(tmp_path, monkeypatch):
    """A cell_type value expands into one row per member type — the
    connectivity comparison's first string lane ('circadian_clock' case)."""
    root = _taxonomy_setup(tmp_path, monkeypatch)
    result = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["clock_cluster"],
        output_dir=str(root / "out"), generate_heatmaps=False, verbose=False,
        project_root=str(root)).run()
    type_df = _read_matrix(
        Path(result["output_folder"]) / "type_level"
        / "type_similarity_vector_v2.csv")
    assert sorted(type_df.index) == ["aMe10", "aMe12"]
    notes = (Path(result["output_folder"])
             / "user_warning_notes.txt").read_text(encoding="utf-8")
    assert "taxonomy label" in notes
    assert "clock_cluster" in notes


def test_taxonomy_label_at_bodyid_level_rows_are_neurons(
        tmp_path, monkeypatch):
    root = _taxonomy_setup(tmp_path, monkeypatch)
    result = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["clock_cluster"],
        aggregation_level="bodyid", output_dir=str(root / "out"),
        generate_heatmaps=False, verbose=False,
        project_root=str(root)).run()
    # Rows follow the resolver's sorted type expansion (aMe10 first).
    assert set(_body_labels(result)) == {
        "1_inst1", "2_inst2", "3_inst3", "4_inst4"}


def test_taxonomy_expansion_respects_member_cap(tmp_path, monkeypatch):
    """Expanded rows go through the same per-type cap machinery (and the cap
    is disclosed in the notes file)."""
    root = _taxonomy_setup(tmp_path, monkeypatch)
    result = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["clock_cluster"],
        max_members_per_type=1, output_dir=str(root / "out"),
        generate_heatmaps=False, verbose=False,
        project_root=str(root)).run()
    members = pd.read_csv(Path(result["output_folder"]) / "members.csv")
    assert set(members["bodyId"].astype(int)) == {1, 3}
    notes = (Path(result["output_folder"])
             / "user_warning_notes.txt").read_text(encoding="utf-8")
    assert "capping" in notes


def test_instance_name_query_pinned_and_folded(vector_setup):
    """An instance name names specific neurons: pinned like bodyId queries
    and folded into their type at the type level."""
    result = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["inst3", "aMe12"],
        max_members_per_type=1, output_dir=str(vector_setup),
        generate_heatmaps=False, verbose=False).run()
    type_df = _read_matrix(
        Path(result["output_folder"]) / "type_level"
        / "type_similarity_vector_v2.csv")
    assert list(type_df.index) == ["aMe10", "aMe12"]
    members = pd.read_csv(Path(result["output_folder"]) / "members.csv")
    by_row = dict(zip(members["row"], members["bodyId"].astype(int)))
    assert by_row == {"aMe10": 3, "aMe12": 1}


def test_instance_names_at_bodyid_level(vector_setup):
    result = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["inst3", "inst4"],
        aggregation_level="bodyid", output_dir=str(vector_setup),
        generate_heatmaps=False, verbose=False).run()
    assert _body_labels(result) == ["3_inst3", "4_inst4"]


def test_unmatched_token_is_disclosed_not_fatal(vector_setup):
    """A token nothing matched degrades to a disclosed note while the rest of
    the query still runs."""
    result = mc.MorphologyProfileComparer(
        dataset="male-cns:v1.0", query=["aMe12", "zzz-nope"],
        output_dir=str(vector_setup), generate_heatmaps=False,
        verbose=False).run()
    assert result["rows_compared"] == 1
    notes = (Path(result["output_folder"])
             / "user_warning_notes.txt").read_text(encoding="utf-8")
    assert "No dataset types matched: zzz-nope" in notes


def test_no_notes_no_warning_file(vector_setup):
    """A run whose query resolved exactly as typed writes no notes file."""
    result = _comparer(vector_setup).run()
    assert not (Path(result["output_folder"])
                / "user_warning_notes.txt").exists()


def test_taxonomy_resolves_for_datasets_outside_the_table_map(
        tmp_path, monkeypatch):
    """The resolver's generic folder fallback: any local dataset resolves,
    not just the five the cross-dataset flow tabulates."""
    from comparison.query_resolver import DatasetTaxonomyResolver
    monkeypatch.setattr(mc, "DatasetTaxonomyResolver", DatasetTaxonomyResolver)
    ids = [1, 2, 3, 4]
    _install_vector_cache(monkeypatch, ids, np.eye(DIM)[:4])
    _install_type_map(monkeypatch, {1: "T1", 2: "T1", 3: "T2", 4: "T2"})
    _install_taxonomy_table(tmp_path, [
        {"bodyId": 1, "type": "T1", "instance": "i1", "cell_type": "clx"},
        {"bodyId": 2, "type": "T1", "instance": "i2", "cell_type": "clx"},
        {"bodyId": 3, "type": "T2", "instance": "i3", "cell_type": "clx"},
        {"bodyId": 4, "type": "T2", "instance": "i4", "cell_type": "clx"},
    ], folder="hemibrain_v1_2_1")
    result = mc.MorphologyProfileComparer(
        dataset="hemibrain:v1.2.1", query=["clx"],
        output_dir=str(tmp_path / "out"), generate_heatmaps=False,
        verbose=False, project_root=str(tmp_path)).run()
    type_df = _read_matrix(
        Path(result["output_folder"]) / "type_level"
        / "type_similarity_vector_v2.csv")
    assert sorted(type_df.index) == ["T1", "T2"]


def test_nblast_dotprops_lookup_uses_canonical_ids(monkeypatch, tmp_path):
    """Regression: the NBLAST path looked dotprops up by int(bodyId), but
    _dotprops_for_ids keys its dict by dataset-canonical ids (strings on
    FlyWire) — every lookup missed, so FAFB NBLAST compared nothing."""
    from types import SimpleNamespace

    class _StrKeyHelper:
        def __init__(self, **kw):
            pass

        def _dotprops_for_ids(self, body_ids, neurons=None, desc=""):
            # Inputs arrive already dataset-canonical; the contract keys the
            # result by the SAME canonical values. The real NBlaster only
            # touches .points for the self-hit, so a stub suffices — pair
            # scores fall back to NaN through the comparer's guard.
            return {
                b: SimpleNamespace(points=np.zeros((10, 3)))
                for b in body_ids
            }

    monkeypatch.setattr(mc, "MorphologyComparer", lambda **kw: _StrKeyHelper())
    monkeypatch.setattr(
        mc, "_canonical_dataset_body_id", lambda dataset, value: f"s{value}")
    import navis.nbl.nblast_funcs as nblast_funcs
    monkeypatch.setattr(nblast_funcs, "NBlaster", _FakeNBlaster)

    def _nblast_matrix():
        cmp = mc.MorphologyProfileComparer(
            dataset="flywire_FAFB_v783", query=["1", "2"], method="nblast",
            verbose=False, generate_heatmaps=False)
        return cmp._nblast_matrix(["1", "2"])

    matrix, kept = _nblast_matrix()
    assert kept == ["1", "2"]
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(1.0)
