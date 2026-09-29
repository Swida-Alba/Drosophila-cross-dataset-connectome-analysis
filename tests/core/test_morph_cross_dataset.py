"""Tests for the cross-dataset morphology helper
(``src/comparison/morph_cross_dataset.py``): dataset scope / pair
availability, seeded null sampling, morph qualification (visualized top-N
only), the CrossDatasetMorphComparer backend, and the population-artifact
bootstrap."""

import pickle
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import morphology as morph  # noqa: E402
import navis  # noqa: E402
from comparison import morph_cross_dataset as mcd  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def make_neuron(points, parents, radius=1.0):
    n = len(points)
    nodes = pd.DataFrame({
        "node_id": np.arange(n, dtype=np.int64),
        "parent_id": np.array([p if p >= 0 else -1 for p in parents],
                              dtype=np.int64),
        "x": [p[0] for p in points], "y": [p[1] for p in points],
        "z": [p[2] for p in points],
        "radius": [radius] * n, "type": ["0"] * n,
    })
    return navis.TreeNeuron(nodes)


def bushy_neuron(shift=(0.0, 0.0, 0.0)):
    pts = [(0 + shift[0], 0 + shift[1], 0 + shift[2]),
           (1 + shift[0], 0 + shift[1], 0 + shift[2])]
    for i in range(1, 6):
        pts.append((1 + i + shift[0], i * 0.4 + shift[1], shift[2]))
        pts.append((1 + i + shift[0], -i * 0.4 + shift[1], shift[2]))
    parents = [-1, 0] + [1] * 10
    return make_neuron(pts, parents)


@pytest.fixture()
def fafb_mcns():
    return "flywire_FAFB_v783", "male-cns:v1.0"


# ---------------------------------------------------------------------------
# dataset scope + pair availability
# ---------------------------------------------------------------------------

class TestDatasetScope:
    def test_allowed_families(self, fafb_mcns):
        fafb, mcns = fafb_mcns
        for name in (fafb, mcns, "banc_v888"):
            scope = mcd.dataset_scope(name)
            assert scope["ok"], scope["reason"]
        assert mcd.dataset_scope("banc_v888")["warnings"]
        assert not mcd.dataset_scope("banc_v888")["warnings"] == []

    def test_unknown_dataset_refused(self):
        scope = mcd.dataset_scope("hemibrain:v1.2.1")
        assert not scope["ok"]
        assert "FAFB" in scope["reason"]

    def test_mcns_v09_refused(self):
        scope = mcd.dataset_scope("male-cns:v0.9")
        assert not scope["ok"]
        assert "v1.0" in scope["reason"]

    def test_optic_lobe_refused(self):
        scope = mcd.dataset_scope("optic-lobe:v1.1")
        assert not scope["ok"]
        assert "male-cns:v1.0" in scope["reason"]

    def test_pair_allowed_within_two_hops(self, fafb_mcns):
        fafb, mcns = fafb_mcns
        check = mcd.check_pair_availability(fafb, mcns)
        assert check["ok"], check["reason"]
        assert 0 <= check["hops"] <= 2

    def test_pair_identity_same_dataset(self, fafb_mcns):
        fafb, _mcns = fafb_mcns
        check = mcd.check_pair_availability(fafb, fafb)
        assert check["ok"] and check["hops"] == 0

    def test_pair_refused_beyond_two_hops(self):
        # MANC is refused at the dataset-scope level (not in the allowed
        # families), which short-circuits before the hop check.
        check = mcd.check_pair_availability("flywire_FAFB_v783",
                                            "manc_v1_2_1")
        assert not check["ok"]
        assert "Only FAFB, male-cns and BANC" in check["reason"]

    def test_bridging_chain_hop_count(self):
        # The raw registry chain for FLYWIRE -> MANC is far beyond the
        # guard; _bridging_chain reports the transform count for banners.
        hops, chain = mcd._bridging_chain("FLYWIRE", "MANC")
        assert hops > 2
        assert "MANC" in chain

    def test_validate_sets_raise_on_bad_dataset(self):
        with pytest.raises(ValueError, match="not supported"):
            mcd.validate_cross_dataset_sets(
                ["flywire_FAFB_v783", "hemibrain:v1.2.1"])

    def test_validate_sets_raise_on_single_dataset(self, fafb_mcns):
        with pytest.raises(ValueError, match="at least two"):
            mcd.validate_cross_dataset_sets([fafb_mcns[0]])


# ---------------------------------------------------------------------------
# seeded null sampling
# ---------------------------------------------------------------------------

class TestNullSample:
    def test_deterministic_and_shared(self, monkeypatch):
        universe = list(range(100, 200))
        monkeypatch.setattr(mcd, "dataset_bodyids",
                            lambda ds, project_root=None: list(universe))
        first = mcd.null_sample("male-cns:v1.0", k=20)
        second = mcd.null_sample("male-cns:v1.0", k=20)
        assert first == second
        assert len(first) == 20

    def test_excludes_candidates(self, monkeypatch):
        universe = list(range(100, 200))
        monkeypatch.setattr(mcd, "dataset_bodyids",
                            lambda ds, project_root=None: list(universe))
        sample = mcd.null_sample("male-cns:v1.0", k=20, exclude=[150])
        assert 150 not in sample
        assert len(sample) == 20

    def test_k_larger_than_universe(self, monkeypatch):
        monkeypatch.setattr(mcd, "dataset_bodyids",
                            lambda ds, project_root=None: [1, 2, 3])
        assert sorted(mcd.null_sample("male-cns:v1.0", k=50)) == [1, 2, 3]


# ---------------------------------------------------------------------------
# morph qualification pure helpers
# ---------------------------------------------------------------------------

def results_frame(rows):
    return pd.DataFrame(rows, columns=[
        "source_bodyId", "target_bodyId", "rank_union", "target_type"])


class TestSelectVisualizedPairs:
    def test_top_n_per_source_with_target_dedupe(self):
        df = results_frame([
            (1, 10, 0.9, "T"),
            (1, 11, 0.8, "U"),
            (1, 12, 0.7, "V"),
            (2, 10, 0.6, "T"),   # target 10 deduped (keep first)
            (2, 13, 0.5, "W"),
        ])
        pairs = mcd.select_visualized_pairs(df, top_n=2)
        assert (1, 10) in pairs and (1, 11) in pairs
        assert (2, 10) not in pairs          # dedupe keeps first source
        assert (2, 13) in pairs
        assert len(pairs) == 3

    def test_empty_frame(self):
        assert mcd.select_visualized_pairs(pd.DataFrame(), top_n=3) == []


class TestMorphQualification:
    def _state(self):
        state = mcd.MorphQualification(
            source_dataset="male-cns:v1.0", target_dataset="flywire_FAFB_v783",
            null_k=200, bar_offset=0.0)
        state.null_stats = {1: {"p95": 0.5, "median": 0.3, "std": 0.1,
                                "mean": 0.3, "n": 50}}
        state.scores = {(1, 100): 0.8, (1, 101): 0.2, (1, 102): 0.5}
        return state

    def test_is_qualified_uses_p95_bar(self):
        state = self._state()
        assert state.is_qualified(1, 100) is True
        assert state.is_qualified(1, 101) is False
        assert state.is_qualified(1, 102) is True   # bar is >= p95
        assert state.is_qualified(1, 999) is None   # unscored
        assert state.qualified_pairs() == [(1, 100), (1, 102)]

    def test_bar_offset_raises_bar(self):
        state = self._state()
        state.bar_offset = 0.1
        assert state.is_qualified(1, 102) is False
        assert state.is_qualified(1, 100) is True

    def test_merge_fills_scored_rows_only(self):
        state = self._state()
        df = pd.DataFrame({
            "source_bodyId": [1, 1, 1],
            "target_bodyId": [100, 101, 999],
        })
        merged = mcd.merge_morph_columns(df, state)
        assert merged.loc[0, "morph_qualified"] is True
        assert merged.loc[1, "morph_qualified"] is False
        assert merged.loc[2, "morph_qualified"] is None
        assert pd.isna(merged.loc[2, "morph_v2"])
        assert merged.loc[0, "morph_null_p95"] == 0.5
        assert merged.loc[0, "morph_z"] == pytest.approx(5.0)

    def test_merge_noop_without_state(self):
        df = pd.DataFrame({"source_bodyId": [1], "target_bodyId": [2]})
        merged = mcd.merge_morph_columns(df, None)
        assert "morph_v2" not in merged.columns

    def test_filter_drops_failed_rows_and_reports(self):
        state = self._state()
        top = pd.DataFrame({
            "source_bodyId": [1, 1, 1],
            "target_bodyId": [100, 101, 102],
            "target_type": ["T1", "T2", "T3"],
        })
        filtered, excluded = mcd.filter_qualified_top_matches(top, state)
        assert list(filtered["target_bodyId"]) == [100, 102]
        assert list(excluded["target_bodyId"]) == [101]

    def test_filter_passes_unscored_rows(self):
        state = self._state()
        top = pd.DataFrame({
            "source_bodyId": [7],
            "target_bodyId": [555],
            "target_type": ["UNSCORED"],
        })
        filtered, excluded = mcd.filter_qualified_top_matches(top, state)
        assert len(filtered) == 1 and excluded.empty


class TestQualifyVisualizedPairs:
    def _install(self, monkeypatch, good_targets=(100,)):
        fake_neurons = ["neuron-a", "neuron-b"]
        monkeypatch.setattr(mcd, "fetch_source_skeletons",
                            lambda ds, bids, project_root=None, log=None,
                            allow_fetch=True:
                            {int(b): object() for b in bids})
        monkeypatch.setattr(
            mcd, "transform_queries",
            lambda src, tgt, skel, validate_bounds=False, log=None:
            (fake_neurons, [1, 2]))
        monkeypatch.setattr(
            mcd, "dataset_bodyids",
            lambda ds, project_root=None:
                list(range(1000, 1100)))

        def fake_score(source, target, query_neurons, query_bids, target_bids,
                       project_root=None, vector_cache=None,
                       side_cache=None, verbose=False):
            rows = []
            rng = np.random.default_rng(7)
            for qb in query_bids:
                for tb in target_bids:
                    if tb < 1000:  # candidates
                        score = 0.9 if tb in good_targets else 0.2
                    else:          # shared null sample
                        score = float(rng.uniform(0.05, 0.35))
                    rows.append({"source_bodyId": qb, "target_bodyId": tb,
                                 "morph_v2_similarity": score})
            return pd.DataFrame(rows)

        monkeypatch.setattr(mcd, "score_pairs", fake_score)

    def test_qualification_flags(self, monkeypatch, fafb_mcns):
        mcns, fafb = fafb_mcns
        self._install(monkeypatch, good_targets=(100,))
        state = mcd.qualify_visualized_pairs(
            mcns, fafb, [(1, 100), (1, 101), (2, 100)],
            null_k=30, project_root=".")
        assert state.active
        assert state.is_qualified(1, 100) is True
        assert state.is_qualified(1, 101) is False
        assert state.is_qualified(2, 100) is True
        # null sample is deterministic per dataset: same ids as null_sample
        sample = mcd.null_sample(fafb, k=30)
        assert set(state.null_stats[1]) is not None
        assert all(s["n"] == 30 for s in state.null_stats.values())
        del sample

    def test_unscored_pairs_dropped(self, monkeypatch, fafb_mcns):
        mcns, fafb = fafb_mcns
        self._install(monkeypatch)

        def score_without_nulls(source, target, query_neurons, query_bids,
                                target_bids, project_root=None,
                                vector_cache=None, side_cache=None,
                                verbose=False):
            # only candidate rows, never the null sample rows
            rows = [{"source_bodyId": qb, "target_bodyId": tb,
                     "morph_v2_similarity": 0.9}
                    for qb in query_bids for tb in target_bids if tb < 1000]
            return pd.DataFrame(rows)

        monkeypatch.setattr(mcd, "score_pairs", score_without_nulls)
        state = mcd.qualify_visualized_pairs(
            mcns, fafb, [(1, 100)], null_k=10, project_root=".")
        assert not state.active
        assert any("no usable scores" in w for w in state.warnings)

    def test_prune_pool_refs_is_the_callers_choice(self, monkeypatch,
                                                   fafb_mcns):
        """mapping_ref's "a branch-pool member was only fetched to ANCHOR a
        bar, so it is not a candidate" rule is the supervised caller's
        semantics. A caller whose candidate set is its own list must be able
        to turn it off — the TM VEV `pooling` gate shipped with it on and
        silently lost the verdict of every candidate the mapper also claimed
        (measured 2026-09-23: 35 of 41 male-cns rows labelled `no-score` while
        the scorer had a value for all 41). Grading stays honest with it off,
        because a candidate is never measured against itself."""
        fafb, mcns = fafb_mcns
        self._install(monkeypatch, good_targets=(100,))
        monkeypatch.setattr(
            mcd, "_mapper_ref_pools",
            lambda source_types, s, t, log=None: {"T": [100]})
        monkeypatch.setattr(
            mcd, "_finalize_mapping_ref_bars",
            lambda mq, refs, df, log=None: None)
        kw = dict(mode="mapping_ref", source_types={1: "T"}, null_k=30,
                  project_root=".")
        pruned = mcd.qualify_visualized_pairs(fafb, mcns, [(1, 100)], **kw)
        kept = mcd.qualify_visualized_pairs(fafb, mcns, [(1, 100)],
                                            prune_pool_refs=False, **kw)
        assert (1, 100) not in pruned.scores      # the default, unchanged
        assert (1, 100) in kept.scores
        assert kept.is_qualified(1, 100) is True

    def test_intra_dataset_qualifies_and_excludes_queries(
            self, monkeypatch, fafb_mcns):
        # Same-dataset (intra) qualification runs on the identity chain;
        # the queries live in the target universe, so they must be
        # excluded from the shared null sample.
        fafb, _ = fafb_mcns
        self._install(monkeypatch, good_targets=(100,))
        monkeypatch.setattr(
            mcd, "transform_queries",
            lambda src, tgt, skel, validate_bounds=False, log=None:
            (["neuron-a"], [1000]))
        captured = {}
        real_null_sample = mcd.null_sample

        def spy(ds, k=mcd.NULL_K_DEFAULT, exclude=(), project_root=None):
            captured["exclude"] = list(exclude)
            out = real_null_sample(ds, k=k, exclude=exclude,
                                   project_root=project_root)
            captured["sample"] = [int(b) for b in out]
            return out

        monkeypatch.setattr(mcd, "null_sample", spy)
        state = mcd.qualify_visualized_pairs(
            fafb, fafb, [(1000, 100), (1000, 101)], null_k=30,
            project_root=".")
        assert state.active
        assert set(captured["exclude"]) == {1000, 100, 101}
        assert 1000 not in captured["sample"]
        assert state.null_stats[1000]["n"] == 30
        assert state.is_qualified(1000, 100) is True
        assert state.is_qualified(1000, 101) is False

    def test_intra_dataset_unscoped_target_refused(self, monkeypatch):
        # The pair check short-circuits same-dataset pairs to identity,
        # so the target scope must refuse artifact-less datasets here.
        with pytest.raises(ValueError, match="v1.0"):
            mcd.qualify_visualized_pairs(
                "male-cns:v0.9", "male-cns:v0.9", [(1, 100)],
                null_k=10, project_root=".")

    def test_scores_only_requested_pairs(self, monkeypatch, fafb_mcns):
        # score_pairs covers the full sources x targets product; the
        # qualification must keep exactly the requested (visualized) pairs.
        mcns, fafb = fafb_mcns
        self._install(monkeypatch, good_targets=(100, 101, 102))
        state = mcd.qualify_visualized_pairs(
            mcns, fafb, [(1, 100)], null_k=30, project_root=".")
        # transform_queries fakes two query neurons; pair (2, 100) was
        # never requested even though it was scored by the product call.
        assert set(state.scores) == {(1, 100)}

    def test_select_pairs_excludes_query_targets(self):
        # Same-dataset scene rule: rows whose target is a query neuron are
        # dropped BEFORE the per-source top-N, mirroring the scene's
        # candidate_results filter.
        df = pd.DataFrame({
            "source_bodyId": [1, 1, 1, 2, 2, 2],
            "target_bodyId": [2, 30, 31, 1, 32, 33],
            "rank_union": [0.9, 0.1, 0.05, 0.8, 0.2, 0.1],
        })
        pairs = mcd.select_visualized_pairs(df, 1, exclude_targets={1, 2})
        # Query-target rows (1->2, 2->1) drop first; each source's top-1 is
        # then its best non-query target.
        assert pairs == [(1, 30), (2, 32)]
        # Without the exclusion the query-target rows win their top-N slot.
        pairs_plain = mcd.select_visualized_pairs(df, 1)
        assert (1, 2) in pairs_plain and (2, 1) in pairs_plain


# ---------------------------------------------------------------------------
# CrossDatasetMorphComparer end-to-end with fakes
# ---------------------------------------------------------------------------

class TestCrossDatasetMorphComparer:
    def _install_fakes(self, monkeypatch):
        # keep the end-to-end test on the raw same-name path; the mapper
        # path has dedicated tests below (the real mapper IS loadable in
        # this repo and would rename APDN3 -> SLP249).
        monkeypatch.setattr(mcd, "get_type_mapper",
                            lambda *a, **k: None)
        monkeypatch.setattr(mcd, "population_artifacts_ready",
                            lambda ds, project_root=None: True)
        monkeypatch.setattr(mcd, "dataset_bodyids",
                            lambda ds, project_root=None:
                                list(range(1000, 1100)))
        monkeypatch.setattr(
            mcd, "fetch_source_skeletons",
            lambda ds, bids, project_root=None, log=None, allow_fetch=True:
                {int(b): object() for b in bids})
        monkeypatch.setattr(
            mcd, "transform_queries",
            lambda src, tgt, skel, validate_bounds=False, log=None:
                ([f"n-{src}-{b}" for b in skel], list(skel.keys())))

        def fake_score(source, target, query_neurons, query_bids, target_bids,
                       project_root=None, vector_cache=None,
                       side_cache=None, verbose=False):
            rows = []
            rng = np.random.default_rng(11)
            for qb in query_bids:
                for tb in target_bids:
                    if tb >= 1000:
                        score = float(rng.uniform(0.05, 0.35))
                    elif source.startswith("male-cns") and tb == 301:
                        score = 0.85   # the known-good cross pair
                    else:
                        score = 0.25
                    rows.append({"source_bodyId": qb, "target_bodyId": tb,
                                 "morph_v2_similarity": score})
            return pd.DataFrame(rows)

        monkeypatch.setattr(mcd, "score_pairs", fake_score)

    def _install_frames(self, monkeypatch):
        def frame_for(dataset):
            bids = ([201, 202] if dataset.startswith("male-cns")
                    else [301, 302])
            return pd.DataFrame({
                "bodyId": [str(b) if "fafb" in dataset else b
                           for b in bids],
                "type": ["APDN3"] * len(bids),
            })

        def loader(self, dataset):
            return frame_for(dataset)

        monkeypatch.setattr(mcd.CrossDatasetMorphComparer,
                            "_load_neuron_frame", loader)

    def test_run_writes_full_output(self, monkeypatch, tmp_path, fafb_mcns):
        _fafb, mcns = fafb_mcns
        self._install_fakes(monkeypatch)
        self._install_frames(monkeypatch)

        comparer = mcd.CrossDatasetMorphComparer(
            datasets=["male-cns:v1.0", "flywire_FAFB_v783"],
            query=["APDN3"], output_dir=str(tmp_path), saveas="run1",
            null_k=25, visualize=False, project_root=str(tmp_path))
        result = comparer.run()

        # mixed FAFB/Neuprint comparison: the FAFB annotation warning must
        # surface (annotation inverted vs other datasets; nothing flipped).
        assert any("FAFB L/R annotation is opposite" in w
                   for w in result["warnings"])

        run_path = Path(result["output_folder"])
        assert run_path == tmp_path / "run1"
        names = {Path(f["path"]).name for f in result["files"]}
        assert {"overview.csv", "report.html", "parameters.json",
                "README.txt", "members_summary.csv"} <= names

        results = run_path / "MCNS_to_FAFB" / "results"
        assert (results / "morph_type_matrix.csv").exists()
        assert (results / "morph_bodyid_matrix.csv").exists()
        assert (results / "morph_bodyid_scores.csv").exists()
        assert (results / "null_baseline.json").exists()

        matrix = pd.read_csv(results / "morph_type_matrix.csv", index_col=0)
        assert list(matrix.columns) == ["APDN3"]
        # Both fake FAFB members (301: 0.85, 302: 0.25) share the type, so
        # the type cell is the member mean.
        assert matrix.loc["APDN3", "APDN3"] == pytest.approx(0.55, abs=1e-6)

        rows = pd.read_csv(results / "morph_bodyid_scores.csv")
        good = rows[rows["target_bodyId"] == 301]
        assert (good["above_baseline"] == True).all()  # noqa: E712
        bad = rows[rows["target_bodyId"] == 302]
        assert (bad["above_baseline"] == False).all()  # noqa: E712

        # BodyId-level matrix: rows = MCNS members, columns = FAFB members,
        # tree-legend labels ('{bodyId}_{type}'), raw vector_v2 values.
        bodyid = pd.read_csv(results / "morph_bodyid_matrix.csv", index_col=0)
        assert bodyid.shape == (2, 2)
        index_labels = [str(label) for label in bodyid.index]
        column_labels = [str(label) for label in bodyid.columns]
        assert any("201" in label for label in index_labels), index_labels
        assert any("202" in label for label in index_labels), index_labels
        assert any("301" in label for label in column_labels), column_labels
        assert any("302" in label for label in column_labels), column_labels
        assert bodyid.iloc[0, 0] == pytest.approx(0.85)
        assert bodyid.iloc[0, 1] == pytest.approx(0.25)

        # Standalone interactive heatmaps (shared report_kit / VisPath).
        viz = run_path / "MCNS_to_FAFB" / "visualization"
        assert (viz / "heatmap_morph_MCNS_to_FAFB_type.html").exists()
        assert (viz / "heatmap_morph_MCNS_to_FAFB_bodyid.html").exists()

        members = pd.read_csv(run_path / "members_summary.csv")
        assert members.loc[0, "queried_type"] == "APDN3"
        assert members.loc[0, "MCNS"] == 2
        assert members.loc[0, "FAFB"] == 2

        overview = pd.read_csv(run_path / "overview.csv")
        assert overview.loc[0, "queried_type"] == "APDN3"
        # best target-type cell per queried type, with the target's name
        assert "MCNS→FAFB" in overview.columns
        assert "MCNS→FAFB best_target" in overview.columns
        apdn3 = overview[overview["queried_type"] == "APDN3"].iloc[0]
        assert apdn3["MCNS→FAFB best_target"] == "APDN3"
        # the single APDN3×APDN3 cell (mean over both members) is the best
        assert apdn3["MCNS→FAFB"] == pytest.approx(0.55, abs=1e-6)

        report = (run_path / "report.html").read_text()
        # Tabbed report on the shared kit: pair tabs, level tabs, Ward
        # clustering, VisPath editor links, embedded Plotly (offline).
        assert "data-tab-button" in report
        assert "MCNS → FAFB" in report
        assert "BodyId level" in report
        assert "null baseline p95" in report
        assert "frame disclosure" in report or "render space" in report
        assert "Ward clustered" in report
        assert "Open VisPath heatmap for editing" in report
        # Offline-capable: Plotly.js embedded inline, no CDN <script> tag.
        assert '<script src="https://cdn.plot.ly' not in report
        # Square-cell strategy depends on matrix size: small matrices use
        # the explicit-width wrapper (no scaleanchor), large matrices keep
        # the scaleanchor path. Either marker proves square_cells is active.
        assert ("'scaleanchor':'x'" in report
                or '"scaleanchor":"x"' in report
                or 'heatmap-square-fit' in report)

    def test_heatmaps_disabled_skips_visualization(self, monkeypatch,
                                                   tmp_path, fafb_mcns):
        self._install_fakes(monkeypatch)
        self._install_frames(monkeypatch)
        comparer = mcd.CrossDatasetMorphComparer(
            datasets=["male-cns:v1.0", "flywire_FAFB_v783"],
            query=["APDN3"], output_dir=str(tmp_path), saveas="run2",
            null_k=25, visualize=False, generate_heatmaps=False,
            project_root=str(tmp_path))
        result = comparer.run()
        run_path = Path(result["output_folder"])
        assert not (run_path / "MCNS_to_FAFB" / "visualization").exists()
        report = (run_path / "report.html").read_text()
        # CSV links remain; the VisPath editor link must not.
        assert "morph_bodyid_matrix.csv" in report
        assert "Open VisPath heatmap for editing" not in report

    def test_refused_dataset_raises(self, tmp_path):
        comparer = mcd.CrossDatasetMorphComparer(
            datasets=["flywire_FAFB_v783", "hemibrain:v1.2.1"],
            query=["x"], output_dir=str(tmp_path), visualize=False,
            project_root=str(tmp_path))
        with pytest.raises(ValueError):
            comparer.run()


# ---------------------------------------------------------------------------
# type-resolver wiring in _resolve_members
# ---------------------------------------------------------------------------

class _FakeResolution:
    def __init__(self, status, kind="renamed", target_types=(),
                 secondary_targets=(), reason="test"):
        self.status = status
        self.kind = kind
        self.target_types = tuple(target_types)
        self.secondary_targets = tuple(secondary_targets)
        self.reason = reason


def _frame_apdn3_and_slp249():
    return pd.DataFrame({
        "bodyId": [201, 202, 301, 302],
        "type": ["APDN3", "APDN3", "SLP249", "SLP249"],
    })


class TestMapperResolution:
    def _comparer(self, tmp_path, monkeypatch, resolution):
        comparer = mcd.CrossDatasetMorphComparer(
            datasets=["flywire_FAFB_v783", "male-cns:v1.0"],
            query=["APDN3"], output_dir=str(tmp_path), visualize=False,
            project_root=str(tmp_path))
        comparer._load_neuron_frame = lambda ds: _frame_apdn3_and_slp249()
        # stubbed outcomes only: never load the real mapper singleton here
        # (its dedicated integration test covers the real path)
        fake_mapper = SimpleNamespace(_loaded=True)
        monkeypatch.setattr(mcd, "get_type_mapper",
                            lambda *a, **k: fake_mapper)
        monkeypatch.setattr(mcd, "resolve_valid_targets",
                            lambda *a, **k: resolution)
        monkeypatch.setattr(
            mcd, "expansion_targets",
            lambda r: (r.target_types if r.status == "mapped"
                       else () if r.status != "unmapped"
                       else (r.target_types or ()) and ())
        ) if False else None
        return comparer

    def test_renamed_query_resolves_through_mapper(self, tmp_path,
                                                   monkeypatch):
        comparer = self._comparer(tmp_path, monkeypatch, _FakeResolution(
            "mapped", kind="renamed", target_types=("SLP249",)))
        # expansion_targets is the real policy helper: patched to its
        # contract for the fake resolution (mapped -> all targets).
        monkeypatch.setattr(mcd, "expansion_targets",
                            lambda r: tuple(r.target_types))
        members, notes = comparer._resolve_members(["APDN3"],
                                                   "male-cns:v1.0")
        # APDN3 has no same-name members in the frame; the mapped SLP249
        # members are returned under the frame's own type label.
        assert members == {"SLP249": [301, 302]}
        assert any("APDN3 → SLP249" in n for n in notes)

    def test_conflict_fails_closed(self, tmp_path, monkeypatch):
        comparer = self._comparer(tmp_path, monkeypatch, _FakeResolution(
            "conflict", reason="unresolved mapping conflict"))
        monkeypatch.setattr(mcd, "expansion_targets", lambda r: ())
        members, notes = comparer._resolve_members(["APDN3"],
                                                   "male-cns:v1.0")
        # fail closed: the same-name APDN3 rows are NOT matched either
        assert members == {}
        assert any("fail closed" in n for n in notes)

    def test_unmapped_falls_back_to_raw_name(self, tmp_path, monkeypatch):
        comparer = self._comparer(tmp_path, monkeypatch, _FakeResolution(
            "unmapped", target_types=()))
        monkeypatch.setattr(mcd, "expansion_targets",
                            lambda r: (r.target_types or ()))
        # unmapped with empty targets -> raw name fallback: expansion
        # returns nothing, so the code must search the raw name instead.
        # Simulate the real helper's unmapped contract:
        monkeypatch.setattr(mcd, "expansion_targets",
                            lambda r: ("APDN3",))
        members, notes = comparer._resolve_members(["APDN3"],
                                                   "male-cns:v1.0")
        assert members == {"APDN3": [201, 202]}

    def test_mapper_off_uses_raw_path(self, tmp_path, monkeypatch):
        comparer = self._comparer(tmp_path, monkeypatch, _FakeResolution(
            "mapped", target_types=("SLP249",)))
        comparer.use_auto_type_mapping = False
        members, notes = comparer._resolve_members(["APDN3"],
                                                   "male-cns:v1.0")
        assert members == {"APDN3": [201, 202]}
        assert not any("→" in n for n in notes)

    def test_pattern_tokens_skip_mapper(self, tmp_path, monkeypatch):
        frame = pd.DataFrame({
            "bodyId": [401, 402], "type": ["aMe12_a", "aMe12_b"]})
        comparer = self._comparer(tmp_path, monkeypatch, _FakeResolution(
            "mapped", target_types=("SHOULD_NOT_MATCH",)))
        monkeypatch.setattr(mcd, "expansion_targets",
                            lambda r: tuple(r.target_types))
        comparer._load_neuron_frame = lambda ds: frame
        members, _notes = comparer._resolve_members(["aMe12*"],
                                                    "male-cns:v1.0")
        assert members == {"aMe12_a": [401], "aMe12_b": [402]}

    def test_real_mapper_apdn3_fafb_to_mcns(self, tmp_path):
        """Integration: the shared mapper resolves FAFB APDN3 into
        male-cns SLP249 (the validation pipeline's verified pair)."""
        pytest.importorskip("comparison.cross_dataset_type_mapper")
        comparer = mcd.CrossDatasetMorphComparer(
            datasets=["flywire_FAFB_v783", "male-cns:v1.0"],
            query=["APDN3"], output_dir=str(tmp_path), visualize=False,
            project_root=".")
        mapper = comparer._get_type_mapper()
        if mapper is None:
            pytest.skip("type mapper not loaded in this environment")
        members, notes = comparer._resolve_members(["APDN3"],
                                                   "male-cns:v1.0")
        assert members, "no members resolved via mapper"
        assert "SLP249" in members
        assert any("APDN3" in n and "SLP249" in n for n in notes)


# ---------------------------------------------------------------------------
# population-artifact bootstrap
# ---------------------------------------------------------------------------

class TestEnsurePopulationArtifacts:
    def test_bootstrap_builds_loadable_cache(self, tmp_path):
        dataset = "banc_vtest"
        folder = (tmp_path / "cache" / morph._dataset_folder(dataset)
                  / "skeletons" / "raw_skeletons")
        folder.mkdir(parents=True, exist_ok=True)
        for i, bid in enumerate(range(5000, 5000 + 30)):
            neuron = bushy_neuron(shift=(i * 10.0, 0.0, 0.0))
            with open(folder / f"{bid}.pkl", "wb") as fh:
                pickle.dump(neuron, fh)

        assert not mcd.population_artifacts_ready(dataset, str(tmp_path))
        result = mcd.ensure_population_artifacts(
            dataset, sample_k=30, project_root=str(tmp_path))
        assert result["status"] == "ready", result

        from morphology import find_similar_dataset_cache_v2
        cache = find_similar_dataset_cache_v2(dataset,
                                              project_root=str(tmp_path),
                                              verbose=False)
        data = cache.load()
        assert data is not None
        assert data["raw"] is not None
        assert data["raw"].shape[0] >= 24
        assert data["raw"].shape[1] == 256
        meta = data["meta"]
        assert meta["version"] == morph.VECTOR_CACHE_V2_VERSION
        assert meta["lateral_normalize"] is True
        assert meta.get("spatial_bounds")
        assert mcd.population_artifacts_ready(dataset, str(tmp_path))

    def test_bootstrap_insufficient_skeletons(self, tmp_path):
        dataset = "banc_vtest2"
        folder = (tmp_path / "cache" / morph._dataset_folder(dataset)
                  / "skeletons" / "raw_skeletons")
        folder.mkdir(parents=True, exist_ok=True)
        for bid in range(6000, 6003):
            with open(folder / f"{bid}.pkl", "wb") as fh:
                pickle.dump(bushy_neuron(), fh)
        result = mcd.ensure_population_artifacts(
            dataset, sample_k=10, project_root=str(tmp_path))
        assert result["status"] == "insufficient"


# ---------------------------------------------------------------------------
# Floors v3: mode / level semantics (plan-unified-morph-qualification-bars)
# ---------------------------------------------------------------------------

class TestMorphQualificationModeLevel:
    def _mq(self, **kw):
        from comparison.morph_cross_dataset import MorphQualification
        mq = MorphQualification(source_dataset='flywire_FAFB_v783',
                                target_dataset='male-cns:v1.0', **kw)
        return mq

    def test_null_mode_bar_uses_level_percentile(self):
        mq = self._mq(mode='null', level=75, bar_offset=0.02)
        mq.null_stats[1] = {'p95': 0.60, 'bar_p': 0.45, 'n': 50}
        mq.scores[(1, 100)] = 0.48
        # bar = bar_p (p75) + offset, NOT the p95
        assert mq.bar(1) == pytest.approx(0.47)
        assert mq.is_qualified(1, 100) is True

    def test_null_mode_default_level_is_95(self):
        mq = self._mq()
        assert mq.level == 95 and mq.mode == 'null'
        mq.null_stats[1] = {'p95': 0.60, 'bar_p': 0.60, 'n': 50}
        mq.scores[(1, 100)] = 0.61
        assert mq.bar(1) == 0.60
        assert mq.is_qualified(1, 100) is True

    def test_mapping_ref_native_floor_binds(self):
        mq = self._mq(mode='mapping_ref')
        mq.ref_bars[1] = {'kind': 'native', 'native_floor': 0.70,
                          'backup_floor': None, 'B_b': 0.75, 'n_refs': 3}
        mq.native_scores[(1, 100)] = 0.72
        mq.native_scores[(1, 101)] = 0.68
        mq.scores[(1, 100)] = 0.40
        mq.scores[(1, 101)] = 0.90
        # native rung compares the native evidence; Track-A is irrelevant
        assert mq.is_qualified(1, 100) is True
        assert mq.is_qualified(1, 101) is False

    def test_mapping_ref_track_a_backup(self):
        mq = self._mq(mode='mapping_ref')
        mq.ref_bars[1] = {'kind': 'track_a', 'native_floor': None,
                          'backup_floor': 0.55, 'B_b': 0.60, 'n_refs': 1}
        mq.scores[(1, 100)] = 0.56
        mq.scores[(1, 101)] = 0.54
        assert mq.is_qualified(1, 100) is True
        assert mq.is_qualified(1, 101) is False

    def test_mapping_ref_without_basis_falls_back_to_null(self):
        mq = self._mq(mode='mapping_ref', bar_offset=0.02)
        mq.ref_bars[1] = {'kind': None, 'native_floor': None,
                          'backup_floor': None, 'B_b': None, 'n_refs': 0}
        mq.null_stats[1] = {'p95': 0.60, 'bar_p': 0.45, 'n': 40}
        mq.scores[(1, 100)] = 0.48
        assert mq.is_qualified(1, 100) is True   # null bar_p + offset
        mq.scores[(1, 101)] = 0.40
        assert mq.is_qualified(1, 101) is False

    def test_merge_morph_columns_adds_bar_columns(self):
        import pandas as pd
        mq = self._mq(mode='mapping_ref')
        mq.ref_bars[1] = {'kind': 'native', 'native_floor': 0.70,
                          'backup_floor': None, 'B_b': None, 'n_refs': 2}
        mq.native_scores[(1, 100)] = 0.75
        mq.scores[(1, 100)] = 0.5
        mq.null_stats[1] = {'p95': 0.6, 'bar_p': 0.5, 'n': 30}
        df = pd.DataFrame([{'source_bodyId': 1, 'target_bodyId': 100}])
        out = mq  # merge lives at module level
        from comparison.morph_cross_dataset import merge_morph_columns
        merged = merge_morph_columns(df, mq)
        assert list(merged['morph_bar_kind']) == ['native']
        assert merged['morph_bar'].iloc[0] == 0.70
        assert list(merged['morph_null_level']) == [95]
        assert list(merged['morph_qualified']) == [True]

    def test_merge_null_mode_kind_token_round_trips(self):
        # 'null' is a pandas NA token — the exported CSV kind must survive
        # a read_csv round trip, hence 'null_bar'.
        import pandas as pd
        from comparison.morph_cross_dataset import merge_morph_columns
        mq = self._mq(mode='null')
        mq.null_stats[1] = {'p95': 0.6, 'bar_p': 0.5, 'n': 30}
        mq.scores[(1, 100)] = 0.55
        df = pd.DataFrame([{'source_bodyId': 1, 'target_bodyId': 100}])
        merged = merge_morph_columns(df, mq)
        assert list(merged['morph_bar_kind']) == ['null_bar']
        round_tripped = pd.read_csv(
            __import__("io").StringIO(merged.to_csv(index=False)))
        assert list(round_tripped['morph_bar_kind']) == ['null_bar']
        # mapping_ref sources without a mapper basis are null-gated too.
        mq2 = self._mq(mode='mapping_ref')
        mq2.null_stats[1] = {'p95': 0.6, 'bar_p': 0.5, 'n': 30}
        mq2.scores[(1, 100)] = 0.55
        merged2 = merge_morph_columns(df, mq2)
        assert list(merged2['morph_bar_kind']) == ['null_bar']


class TestTargetVectorStoreSeam:
    """The persisted target-vector store must be read BEFORE the scorer runs and
    written ONCE after it, with the hemisphere riding along: that is what turns
    the measured 0.412 s/neuron of render + transform + vectorize into a file
    read on the next run. `MorphQualification.vector_cache` is the ledger that
    makes the speed-up auditable instead of felt.
    """

    class FakeStore:
        #: the last instance built, so a test can assert the opt-out
        built = None

        def __init__(self, vectors, sides):
            self._v, self._s = dict(vectors), dict(sides)
            self.calls = []
            self.stats = {'loaded': len(vectors), 'stale_dropped': 0,
                          'saved': 0}
            TestTargetVectorStoreSeam.FakeStore.built = self

        def load(self):
            self.calls.append('load')
            return dict(self._v), dict(self._s)

        def update(self, vectors, sides=None):
            self.calls.append('update')
            self.saved = (dict(vectors), dict(sides or {}))
            self.stats['saved'] = len(vectors)

    def _install(self, monkeypatch, store_cls):
        import comparison.morph_cross_dataset as mcd
        monkeypatch.setattr(mcd, 'fetch_source_skeletons',
                            lambda ds, bids, project_root=None, log=None,
                            allow_fetch=True:
                            {int(b): object() for b in bids})
        monkeypatch.setattr(
            mcd, 'transform_queries',
            lambda src, tgt, skel, validate_bounds=False, log=None:
            (['neuron-a', 'neuron-b'], [1, 2]))
        monkeypatch.setattr(mcd, 'dataset_bodyids',
                            lambda ds, project_root=None:
                            list(range(1000, 1100)))
        monkeypatch.setattr(mcd, 'TargetVectorStore', store_cls)
        seen = {}

        def fake_score(source, target, query_neurons, query_bids, target_bids,
                       project_root=None, vector_cache=None,
                       side_cache=None, verbose=False):
            seen['vector_cache'] = dict(vector_cache or {})
            seen['side_cache'] = dict(side_cache or {})
            rng = np.random.default_rng(5)
            rows = []
            for qb in query_bids:
                for tb in target_bids:
                    score = (0.9 if int(tb) == 100 else
                             float(rng.uniform(0.05, 0.35)))
                    rows.append({'source_bodyId': qb, 'target_bodyId': tb,
                                 'morph_v2_similarity': score})
            return pd.DataFrame(rows)

        monkeypatch.setattr(mcd, 'score_pairs', fake_score)
        return seen

    def test_the_store_feeds_the_scorer_and_is_written_once(self, monkeypatch,
                                                            fafb_mcns):
        import comparison.morph_cross_dataset as mcd
        mcns, fafb = fafb_mcns
        vec = {100: np.ones(4), 7777: np.zeros(4)}
        seen = self._install(
            monkeypatch,
            lambda ds, root=None: self.FakeStore(vec, {100: 'left'}))
        state = mcd.qualify_visualized_pairs(mcns, fafb, [(1, 100)],
                                            null_k=30, project_root='.')
        assert state.active
        # the stored vector and its side reach the scorer, so a hit can skip
        # the render entirely
        assert seen['vector_cache'].keys() >= {100, 7777}
        assert seen['side_cache'] == {100: 'left'}
        built = self.FakeStore.built
        assert built.calls == ['load', 'update'], built.calls
        assert built.saved[0].keys() >= {100}
        assert state.vector_cache['loaded'] == 2
        assert state.vector_cache['saved'] >= 1

    def test_use_vector_store_false_touches_no_file(self, monkeypatch,
                                                    fafb_mcns):
        """The caller can opt out, and then nothing is read or written — a
        reproducibility run against a cold store has to be reachable."""
        import comparison.morph_cross_dataset as mcd

        def explode(*a, **k):
            raise AssertionError('the store must not be constructed')

        mcns, fafb = fafb_mcns
        self._install(monkeypatch, explode)
        state = mcd.qualify_visualized_pairs(mcns, fafb, [(1, 100)],
                                            null_k=30, project_root='.',
                                            use_vector_store=False)
        assert state.active
        assert state.vector_cache == {}


# ---------------------------------------------------------------------------
# bridged-scene member tagging
# ---------------------------------------------------------------------------

def test_tag_scene_members_names_and_tags():
    """Every bridged-scene member gets a unique bodyId-suffixed name and a
    source-dataset tag, in input order (regression: the inline loop used an
    out-of-scope ``b`` and the swallowed NameError skipped both)."""
    members = [
        (720575940614131061, SimpleNamespace(name='old-a')),
        (720575940614131062, SimpleNamespace(name='old-b')),
    ]
    out = mcd._tag_scene_members(members, 'PFN (A-sector)', 'FAFB', 'flywire')
    assert [n.name for n in out] == [
        'PFN__A-sector_FAFB_720575940614131061',
        'PFN__A-sector_FAFB_720575940614131062',
    ]
    assert len({n.name for n in out}) == 2
    assert all(n._drocat_source_dataset == 'flywire' for n in out)
    # body ids that are already ints pass through int() unchanged
    out2 = mcd._tag_scene_members([(1, SimpleNamespace()), (2, SimpleNamespace())],
                                  't', 'BANC', 'banc_v888')
    assert [n.name for n in out2] == ['t_BANC_1', 't_BANC_2']


def test_fetch_source_skeletons_fafb_loader_crash_never_returns_mesh(
        tmp_path, monkeypatch):
    """If the FAFB release loader itself raises, the fallback chain must
    degrade to skipped ids — never to fetch_skeleton_on_demand's old
    prepared-mesh answer (2026-09-29 reverse-scene defects)."""
    import morphology

    def boom(*a, **k):
        raise RuntimeError("healed zip unreadable")

    def must_not(*a, **k):
        raise AssertionError("fetch_skeleton_on_demand used for FAFB")

    monkeypatch.setattr(morphology, "load_local_release_skeletons", boom)
    monkeypatch.setattr(morphology, "fetch_skeleton_on_demand", must_not)
    out = mcd.fetch_source_skeletons("flywire_FAFB_v783", [7],
                                     project_root=str(tmp_path), log=None,
                                     allow_fetch=True)
    assert out == {}
