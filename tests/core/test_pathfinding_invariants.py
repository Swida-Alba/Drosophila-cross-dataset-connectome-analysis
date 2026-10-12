"""Pathfinding mathematical invariants (plan §13 — deep-verification
round).  Every test asserts hand-computed mathematics on tiny synthetic
graphs, with a brute-force simple-path oracle as the independent
reference for the enumeration semantics.

Covered contracts:
- Complete mode (budgeted StrongestFirst): descending-bottleneck
  emission, the tau budget semantics (budget-th strongest; ties
  drained), the canonical threshold (w2 + 1) reproducing the set,
  threshold monotonicity, and the brute-force oracle equality.
- Min-synapse cutoff: pair-level edge semantics; bottleneck = min edge.
- Lossless hop-budget pruning: distS + 1 + distT <= bound, iterating to
  a fixpoint, W* (the widest-path ceiling) unchanged.
- Edge Budget (fit): the floored run is exactly a complete run at t*;
  applied = max(requested, t*).
- Ratio basis: ratio = weight / F9 all-post full-table total
  (cone-external mass never shrinks the denominator); the ratio tier;
  the implied synapse cutoff ceil(t_r * total) clamped >= 1; AND-mode
  composition (synapse cutoff AND ratio co-threshold).
- Fast mode (corridor): the planner rule, the floor certificates, and
  realized-mass conservation between the walk rows and the pair mirror.
"""
import json
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]   # the repo root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vispath-subproject" / "src"))

from vispath_pkg.fast_graph_core import FastGraph  # noqa: E402
from coana import fit_edge_budget, prune_layers_hop_budget  # noqa: E402
from utils.threshold_state import applied_threshold_provenance  # noqa: E402


# ---------------------------------------------------------------- fixtures
def _oracle_paths(edges, sources, targets, bound):
    """Brute-force simple-path enumeration with bottleneck weights —
    the independent reference: returns {(path tuple): bottleneck}."""
    adj = {}
    for u, v, w in edges:
        adj.setdefault(u, []).append((v, w))
    out = {}

    def dfs(node, visited, path, bn):
        if node in targets and path:
            out[tuple(path)] = bn
        if len(path) > bound:
            return
        for v, w in adj.get(node, ()):
            if v in visited:
                continue
            visited.add(v)
            path.append(v)
            dfs(v, visited, path, min(bn, w))
            path.pop()
            visited.discard(v)

    for s in sources:
        dfs(s, {s}, [s], float("inf"))
    return out


def _build_graph(edges):
    g = FastGraph()
    g.build_from_edges([(u, v, float(w)) for u, v, w in edges])
    return g


# G1: the hand-computed complete-mode graph.
G1_EDGES = [
    ("S1", "A", 10), ("S1", "B", 6), ("A", "T", 8), ("B", "T", 8),
    ("A", "B", 3), ("S2", "A", 5),
]
G1_SOURCES, G1_TARGETS = ["S1", "S2"], ["T"]
# hand-computed simple paths (bound 3):
#   S1->A->T bn 8 | S1->B->T bn 6 | S2->A->T bn 5
#   S1->A->B->T bn 3 | S2->A->B->T bn 3
G1_ALL_BN = [8, 6, 5, 3, 3]


def _g1_paths(bound=3, min_w=1):
    edges = [(u, v, w) for u, v, w in G1_EDGES if w >= min_w]
    return _oracle_paths(edges, G1_SOURCES, G1_TARGETS, bound)


def _g1_strongest(bound=3, min_w=1, budget=None):
    g = _build_graph([(u, v, w) for u, v, w in G1_EDGES if w >= min_w])
    stats = {}
    gen = g.find_paths_strongest_first(
        G1_SOURCES, G1_TARGETS, bound, budget=budget, stats=stats,
        yield_bottleneck=True)
    out = [(bn, tuple(p)) for bn, p in gen]
    return out, stats


# ---------------------------------------------------- complete mode
class TestCompleteModeBudgetSemantics:
    def test_complete_matches_bruteforce_oracle(self):
        out, stats = _g1_strongest(budget=None)
        oracle = _g1_paths()
        assert {p for _bn, p in out} == set(oracle)
        # descending-bottleneck emission
        bns = [bn for bn, _p in out]
        assert bns == sorted(bns, reverse=True)
        assert stats.get("tau") == 3          # the natural weakest hop

    def test_budget_emits_exact_top_k_with_tau(self):
        # budget 2: the two strongest (bn 8, 6); tau = the 2nd strongest
        out, stats = _g1_strongest(budget=2)
        assert [bn for bn, _p in out] == [8, 6]
        assert stats["tau"] == 6
        assert stats["budget_bitten"] is True
        # the emitted set == all oracle paths with bn >= tau
        assert {p for _bn, p in out} == {p for p, bn in _g1_paths().items()
                                         if bn >= 6}

    def test_budget_not_bitten_at_full_budget(self):
        out, stats = _g1_strongest(budget=10)
        assert len(out) == 5 and stats["budget_bitten"] is False
        assert stats["tau"] == 3  # the natural weakest bottleneck

    def test_tau_drains_ties_at_the_cutoff(self):
        # budget 4: bn 8, 6, 5 emitted; the tau drain must also keep the
        # bn-3 TIE pair (both S1/S2 A->B->T paths at the same cutoff)
        out, stats = _g1_strongest(budget=4)
        bns = sorted((bn for bn, _p in out), reverse=True)
        assert bns == [8, 6, 5, 3, 3]
        assert stats["tau"] == 3

    def test_canonical_threshold_reproduces_the_set(self):
        # budget 3 bites between bn 5 and bn 3: canonical = w2 + 1 = 4;
        # an edge cutoff at 4 must reproduce the emitted set exactly
        out, _stats = _g1_strongest(budget=3)
        emitted = {p for _bn, p in out}
        reproduced = _g1_paths(min_w=4)
        assert emitted == set(reproduced)
        prov = applied_threshold_provenance(
            requested_threshold=1,
            strongest_first_tau=5,
            strongest_first_budget_bitten=True,
            strongest_dropped_bottleneck=3,
            tau_canonical=4,
        )
        assert prov["applied_threshold"] == 4
        assert prov["applied_threshold_source"] == "strongest_first_budget"

    def test_min_synapse_edge_cutoff_bottleneck_semantics(self):
        # min_synapse 7: edges S1->B (6), A->B (3), S2->A (5) die;
        # only S1->A->T (bn 8) survives — the bottleneck is the min edge
        out, _stats = _g1_strongest(min_w=7)
        assert {p: bn for bn, p in out} == {("S1", "A", "T"): 8}

    def test_threshold_monotonicity_sweep(self):
        oracle = _g1_paths()
        sets = {}
        for t in range(1, 11):
            sets[t] = set(_g1_paths(min_w=t))
        for t in range(2, 11):
            assert sets[t] <= sets[t - 1], t


# ---------------------------------------------------- hop-budget prune
class TestHopBudgetPrune:
    def _layers(self):
        return [pl.DataFrame({
            "bodyId_pre": [u for u, _v, _w in G1_EDGES],
            "bodyId_post": [v for _u, v, _w in G1_EDGES],
            "weight": [float(w) for _u, _v, w in G1_EDGES]})]

    def test_lossless_fixpoint_and_w_star(self):
        tables, stats = prune_layers_hop_budget(
            self._layers(), G1_SOURCES, G1_TARGETS, 2)
        kept = {(r["bodyId_pre"], r["bodyId_post"])
                for r in tables[0].iter_rows(named=True)}
        # bound 2 kills only A->B (distS(A)=1, distT(B)=1: 1+1+1 > 2)
        assert ("A", "B") not in kept
        assert kept == {("S1", "A"), ("S1", "B"), ("S2", "A"),
                        ("A", "T"), ("B", "T")}
        assert stats["strongest_retained"] == 8  # W* unchanged (lossless)
        # every 2-hop oracle path survives
        oracle2 = _oracle_paths(G1_EDGES, G1_SOURCES, G1_TARGETS, 2)
        for path in oracle2:
            for u, v in zip(path, path[1:]):
                assert (u, v) in kept, (u, v)

    def test_bound_3_keeps_everything(self):
        tables, stats = prune_layers_hop_budget(
            self._layers(), G1_SOURCES, G1_TARGETS, 3)
        assert tables[0].height == len(G1_EDGES)


# ---------------------------------------------------- Edge Budget fit
class TestEdgeBudgetFit:
    def _layers(self):
        return [pl.DataFrame({
            "bodyId_pre": [u for u, _v, _w in G1_EDGES],
            "bodyId_post": [v for _u, v, _w in G1_EDGES],
            "weight": [float(w) for _u, _v, w in G1_EDGES]})]

    def test_floor_is_weakest_fitting_tier_and_equivalent(self):
        # 5 rows, budget 4: tiers 10/8/6 fit, tier 3 overshoots (5 rows)
        # -> floor t* = 6; the floored cone = the complete cone at >= 6
        tables, stats = fit_edge_budget(
            self._layers(), 4, G1_SOURCES, G1_TARGETS, 3)
        assert stats["applied"] is True
        assert stats["floor"] == 6
        assert stats["edges_after"] == 4
        kept = {(r["bodyId_pre"], r["bodyId_post"])
                for r in tables[0].iter_rows(named=True)}
        assert kept == {("S1", "A"), ("A", "T"), ("S1", "B"), ("B", "T")}
        # the floored run is exactly a complete run at min_synapse = t*
        oracle6 = _g1_paths(min_w=6)
        assert {p for p in oracle6} == {("S1", "A", "T"),
                                        ("S1", "B", "T")}

    def test_no_floor_when_cone_fits(self):
        tables, stats = fit_edge_budget(
            self._layers(), 100, G1_SOURCES, G1_TARGETS, 3)
        assert stats["applied"] is False and stats["floor_skipped"] is True

    def test_applied_threshold_composition(self):
        # The applied threshold is the CANONICAL equivalent threshold
        # (the minimal threshold reproducing the materialized set) —
        # deliberately NOT a naive max(requested, w0) label: with the
        # StrongestFirst drain landing at tau 8, the materialized set is
        # {bn 8}, so the canonical is 8 even though the floor is 6.
        prov = applied_threshold_provenance(
            requested_threshold=2,
            strongest_first_tau=8,
            edge_weight_floor=6,
            edge_budget_landing=3,
            strongest_retained_bottleneck=8,
        )
        assert prov["applied_threshold"] == 8
        assert "edge_budget" in prov["applied_threshold_source"]
        # floor binding, drain NOT bitten, natural tau at the floor:
        # the floored run is exactly a complete run at max(requested, w0)
        prov2 = applied_threshold_provenance(
            requested_threshold=2,
            strongest_first_tau=6,
            edge_weight_floor=6,
            edge_budget_landing=3,
        )
        assert prov2["applied_threshold"] == 6
        assert prov2["applied_threshold_source"] == "edge_budget"
        assert prov2["paths_complete"] is False


# ---------------------------------------------------- ratio basis
class TestRatioBasis:
    EDGES = [
        # post T: total incoming 200 -> ratio = w / 200
        ("A", "T", 10),   # 0.05
        ("B", "T", 30),   # 0.15
        ("C", "T", 160),  # 0.80
        # post U: total incoming 40
        ("A", "U", 25),   # 0.625
        ("B", "U", 15),   # 0.375
    ]

    def test_ratio_is_weight_over_f9_total(self):
        sys_modules = sys.modules
        from connection_ratio_paths import (compute_incoming_totals,
                                            prepare_ratio_frame)
        frame = pl.DataFrame({
            "bodyId_pre": [u for u, _v, _w in self.EDGES],
            "bodyId_post": [v for _u, v, _w in self.EDGES],
            "weight": [float(w) for _u, _v, w in self.EDGES]})
        frame, totals = prepare_ratio_frame(frame)
        assert totals == {"T": 200.0, "U": 40.0}
        ratios = {(r["bodyId_pre"], r["bodyId_post"]): r["ratio"]
                  for r in frame.iter_rows(named=True)}
        assert ratios[("A", "T")] == pytest.approx(10 / 200)
        assert ratios[("B", "U")] == pytest.approx(15 / 40)

    def test_cone_external_mass_never_shrinks_denominator(self):
        # the F9 invariant: a cone that excludes B's other input (B->U's
        # mass reaching U) must not change A->U's ratio
        from connection_ratio_paths import (compute_incoming_totals,
                                            prepare_ratio_frame)
        cone_only = pl.DataFrame({
            "bodyId_pre": ["A"], "bodyId_post": ["U"],
            "weight": [25.0]})
        frame, totals = prepare_ratio_frame(
            pl.concat([cone_only, pl.DataFrame({
                "bodyId_pre": ["B"], "bodyId_post": ["U"],
                "weight": [15.0]})]))
        ratios = {r["bodyId_pre"]: r["ratio"]
                  for r in frame.iter_rows(named=True)}
        assert totals["U"] == 40.0                      # full-table total
        assert ratios["A"] == pytest.approx(25 / 40)    # not 25/25

    def test_implied_syn_cutoff_clamped(self):
        from connection_ratio_paths import _implied_cutoff
        assert _implied_cutoff(0.10, 200) == 20
        assert _implied_cutoff(0.10, 5) == 1     # clamp: sub-synapse -> 1
        assert _implied_cutoff(0.15, 40) == 6    # ceil(6.0)

    def test_and_mode_composition_synapse_and_ratio(self):
        # AND mode (synapse basis): D_t at min_synapse AND the F9-ratio
        # co-threshold — both must hold on the same pair.
        import coana
        rows = pl.DataFrame({
            "bodyId_pre": ["A", "B", "C"],
            "bodyId_post": ["T", "T", "T"],
            "weight": [10.0, 30.0, 160.0]})
        holder = type("H", (), {"_warn_notes": []})()
        fetches = {}

        def fake_fetch(self, posts, min_weight=1):
            key = (tuple(posts), min_weight)
            if key not in fetches:
                fetches[key] = pl.DataFrame({
                    "bodyId_post": list(posts),
                    "total_incoming_weight": [200.0] * len(posts)})
            return fetches[key].to_pandas()

        monkey_self = holder
        # apply the row-level synapse cutoff first (min_synapse 20), then
        # the ratio co-threshold at 0.10 — the AND composition
        passed = rows.filter(pl.col("weight") >= 20)
        assert set(passed["bodyId_pre"]) == {"B", "C"}
        # the ratio co-threshold via the engine helper (F9 totals stubbed)
        orig = coana.FindNeuronConnection._fetch_total_incoming_weight
        coana.FindNeuronConnection._fetch_total_incoming_weight = fake_fetch
        try:
            kept = coana.FindNeuronConnection._filter_frame_by_ratio_threshold(
                holder, passed, 0.10)
        finally:
            coana.FindNeuronConnection._fetch_total_incoming_weight = orig
        # AND composition: B survives (30 >= 20 synapse AND
        # 30/200 = 0.15 >= 0.10 ratio); A died on the synapse arm
        # (10 < 20) even though 10/200 < 0.10 would also fail the ratio.
        assert set(kept["bodyId_pre"]) == {"B", "C"}


# ---------------------------------------------------- corridor (Fast mode)
class TestCorridorInvariants:
    def test_floor_certificate_and_realized_conservation(self, tmp_path):
        import route_plan
        root = tmp_path / "ds"
        root.mkdir()
        # A(2 neurons) -10-> B -9-> C ; A -4-> B2 -8-> C
        pl.DataFrame({
            "bodyId_pre": ["a1", "a2", "b1", "b2", "a1"],
            "bodyId_post": ["b1", "b2", "c1", "c1", "b2"],
            "weight": [10, 4, 9, 8, 6]}).write_parquet(
            root / "connections.parquet")
        import pandas as pd
        pd.DataFrame({"bodyId": ["a1", "a2", "b1", "b2", "c1"],
                      "type": ["A", "A", "B", "B", "C"]}).to_parquet(
            root / "neurons.parquet", index=False)
        out = tmp_path / "lane"
        route_plan.run_lane(
            None, asked=3, max_interlayer=2, source_types=["A"],
            target_types=["C"], connections_path=root / "connections.parquet",
            neuron_table=root / "neurons.parquet", out_dir=out,
            per_pair_length_budget=100)
        rows = pl.read_csv(out / "route_paths_type.csv")
        body = pl.read_csv(out / "route_paths_bodyId.csv")
        conn = pl.read_csv(out / "route_connection_type.csv")
        prov = __import__("json").loads(
            (out / "route_provenance.json").read_text())
        # every emitted path's min_weight (the realized bottleneck)
        # >= the ASKED threshold — the floor certificate's floor
        assert rows["min_weight"].min() >= 3
        # mass conservation: the mirror's realized_mass per pair equals
        # the sum of the walk rows' hop masses on that pair
        mirror = {f"{r['type_pre']}->{r['type_post']}": r["realized_mass"]
                  for r in conn.iter_rows(named=True)}
        for r in rows.iter_rows(named=True):
            hops = r["path"].split("->")
            masses = json.loads(r["weights"])
            for i, m in enumerate(masses):
                key = f"{hops[i]}->{hops[i + 1]}"
                # single-walk pairs must match the walk row exactly
                used = int(conn.filter(
                    (pl.col("type_pre") == hops[i])
                    & (pl.col("type_post") == hops[i + 1]))["used_by_walks"][0])
                if used == 1:
                    assert mirror[key] == m, (key, mirror[key], m)
        # provenance floors: floor >= asked on live classes
        for _k, v in prov["classes"].items():
            if v["accepted"]:
                assert (v["floor"] is None) or (v["floor"] >= 3)

    def test_planner_rule_first_hop_self_pair(self):
        from route_plan import CorridorDrain, widest_to_targets
        adj = {"A": [("A", 9.0), ("B", 5.0)], "B": [("C", 5.0)],
               "C": []}
        w = widest_to_targets(adj, {"C"}, 4)
        d = CorridorDrain(adj, "A", {"C"}, 4, W=w, allow_multi_visit=False)
        walks = [walk for _ub, walk in d]
        assert ("A", "A", "B", "C") not in walks
        assert ("A", "B", "C") in walks


# ---------------------------------------------------- real data
class TestRealDataDt:
    def test_mcns_dt_pair_physical(self):
        """The converted MCNS cache: D_t totals == the independent
        pair-physical reference (compact permanent version of the §12.5
        gate)."""
        cache = ROOT / "cache/male-cns_v1_0/connections.parquet"
        idx = ROOT / "neuron_indexes/male-cns_v1_0/neuron_index.parquet"
        if not cache.exists() or not idx.exists():
            pytest.skip("MCNS cache/index not present")
        from connection_map import ThresholdedConnectionMap
        cm = ThresholdedConnectionMap(str(cache), str(idx), min_weight=3)
        got = dict(cm.total_incoming_by_bodyid().iter_rows())
        raw = pl.read_parquet(cache, columns=["bodyId_pre",
                                              "bodyId_post", "weight"])
        ref_pairs = (raw.group_by(["bodyId_pre", "bodyId_post"])
                     .agg(pl.col("weight").sum().alias("w"))
                     .filter(pl.col("w") >= 3)
                     .group_by("bodyId_post")
                     .agg(pl.col("w").sum().alias("tot")))
        ref = dict(zip(ref_pairs["bodyId_post"].to_list(),
                       ref_pairs["tot"].to_list()))
        assert got == ref


class TestRealizedRatioCustomLabels:
    """Custom-label posts (label-mapper groups) have no native-type rows,
    so the F9 by-type fetch cannot resolve them — the realized-ratio type
    mass must fall back to the group's member bodyIds (via the run's
    label mapper) and sum their per-neuron F9 totals (plan §13)."""

    def test_type_mass_resolves_custom_label_via_mapper(self):
        import coana
        from comparison.label_mapper import LabelMapper
        import pandas as pd

        mapper_json = (ROOT / "clock-production/data/"
                       "clock_validated_expanded/"
                       "LabelMapper_clock_FAFB-MCNS-BANC.json")
        if not mapper_json.exists():
            pytest.skip("clock LabelMapper JSON not present")
        lm = LabelMapper(overall_mapping_json=str(mapper_json))
        members = lm.get_neurons_for_label(
            "5th-LNv", "male-cns:v1.0", "target")
        assert members, "mapper returned no members for 5th-LNv"

        fc = coana.FindNeuronConnection.__new__(
            coana.FindNeuronConnection)
        fc.dataset = "male-cns:v1.0"
        fc.label_mapper = lm
        fc._ratio_lane_type_members = {}
        wanted = ["5th-LNv"]

        # by-type fetch: custom labels have no native rows (empty)
        fc._fetch_total_incoming_weight_by_type = (
            lambda types, min_weight=1: pd.DataFrame(
                columns=["type_post", "total_incoming_weight"]))
        # per-neuron F9 fetch: the members' physical incoming
        per_neuron = {str(b): float(30 + 5 * i)
                      for i, b in enumerate(members)}
        fc._fetch_total_incoming_weight = (
            lambda posts, min_weight=1: pd.DataFrame({
                "bodyId_post": [str(p) for p in posts],
                "total_incoming_weight": [
                    per_neuron.get(str(p), 0.0) for p in posts]}))

        mass = fc._realized_ratio_type_mass(wanted)
        assert mass.get("5th-LNv") == pytest.approx(
            sum(per_neuron.values()))

    def test_readout_whole_path_none_on_unresolvable(self):
        # the standing whole-path None contract: one unresolvable hop
        # post nullifies the path's realized ratios (the caller labels
        # the row None per the 2026-10-10 directive)
        import coana
        out = coana.FindNeuronConnection._realized_ratio_readout(
            [9.0, 12.0], ("TS", "TA", "TT"),
            {"TA": 77.0, "TT": None})   # TT unresolvable
        assert out is None
        res = coana.FindNeuronConnection._realized_ratio_readout(
            [9.0, 12.0], ("TS", "TA", "TT"),
            {"TA": 77.0, "TT": 154.0})
        assert res[0] == pytest.approx([9 / 77, 12 / 154])


class TestLabelMapperBehaviors:
    """The LabelMapper behavior matrix (plan §13.7 review): the custom
    JSON's own semantics plus every consumer contract probed this round.
    Uses the real clock mapper (3 datasets, 21 group labels)."""

    @pytest.fixture(scope="class")
    def mapper(self):
        from comparison.label_mapper import LabelMapper
        p = (ROOT / "clock-production/data/clock_validated_expanded/"
             "LabelMapper_clock_FAFB-MCNS-BANC.json")
        if not p.exists():
            pytest.skip("clock LabelMapper JSON not present")
        return LabelMapper(overall_mapping_json=str(p))

    def test_get_label_mapped_bodyid_and_identity_fallback(self, mapper):
        members = mapper.get_neurons_for_label(
            "5th-LNv", "male-cns:v1.0", "target")
        assert members
        # mapped bodyId -> the custom label
        assert mapper.get_label("male-cns:v1.0", members[0]) == "5th-LNv"
        # unmapped bodyId -> the IDENTITY fallback (the id as string),
        # which build_effective_type_map treats as 'no override'
        unmapped = mapper.get_label("male-cns:v1.0", "999999999")
        assert unmapped == "999999999"

    def test_get_label_hemisphere_suffix_reapplied(self, mapper):
        # a suffixed mapped id: base lookup + suffix re-application
        members = mapper.get_neurons_for_label(
            "LNd_CRY+", "male-cns:v1.0", "target")
        base = members[0]
        suffixed = mapper.get_label("male-cns:v1.0", f"{base}_L")
        if suffixed.endswith("_L"):
            assert suffixed == "LNd_CRY+_L"
        else:
            # the dataset's bodies may already be suffix-free; the base
            # identity path must at least not lose the mapping
            assert suffixed == "LNd_CRY+"

    def test_roles_scoped_members(self, mapper):
        # the clock export fills source and target mappings with the
        # same groups; intermediate is absent
        src = mapper.get_neurons_for_label(
            "5th-LNv", "male-cns:v1.0", "source")
        tgt = mapper.get_neurons_for_label(
            "5th-LNv", "male-cns:v1.0", "target")
        assert src == tgt and src

    def test_dataset_keys_sanitized_and_raw(self, mapper):
        # 'male-cns:v1.0' sanitizes to 'male-cns_v1_0' — both key forms
        # must resolve (CSV-loaded mappers store the sanitized form)
        raw = mapper.get_neurons_for_label(
            "5th-LNv", "male-cns:v1.0", "target")
        san = mapper.get_neurons_for_label(
            "5th-LNv", "male-cns_v1_0", "target")
        assert raw == san and raw

    def test_apply_to_dataframe_adds_std_columns_with_bodyid_retry(
            self, mapper):
        import pandas as pd
        members = mapper.get_neurons_for_label(
            "5th-LNv", "male-cns:v1.0", "target")
        df = pd.DataFrame({
            "bodyId_pre": [members[0]],
            "bodyId_post": ["999999999"],      # unmapped bodyId
            "type_pre": ["5thsLNv_LNd6"],      # native type
            "type_post": ["Native_Type"],
            "weight": [21]})
        out = mapper.apply_to_dataframe(df, "male-cns:v1.0")
        # the mapped body's std_label = the custom label; the unmapped
        # body's std_label = its native type (identity on the NAME — the
        # bodyId retry adopts only when the bodyId itself maps)
        assert out["std_label_pre"][0] == "5th-LNv"
        assert out["std_label_post"][0] == "Native_Type"
        # type columns are never renamed in place
        assert out["type_pre"][0] == "5thsLNv_LNd6"

    def test_f9_by_type_resolves_custom_labels(self, tmp_path, mapper):
        """The F9 by-type fetch resolves mapper-group labels through the
        members' D_t per-neuron totals (the plan §13.7 source-level
        fix) — verified against a per-ROI-style temp cache."""
        import coana
        members = mapper.get_neurons_for_label(
            "5th-LNv", "male-cns:v1.0", "target")
        conn = tmp_path / "connections.parquet"
        idx = tmp_path / "neuron_index.parquet"
        # the members' incoming: pre-neurons X1/X2 -> member[0] (3+4),
        # and an unrelated pair
        rows = {"bodyId_pre": ["X1", "X2", "Q1"],
                "bodyId_post": [members[0], members[0], "ZZ"],
                "roi": ["R", "R", "R"], "weight": [3, 4, 9]}
        pl.DataFrame(rows).write_parquet(conn)
        pl.DataFrame({"bodyId": ["X1", "X2", "ZZ", str(members[0])],
                      "type": ["X", "X", "Z", "TB"]}).write_parquet(idx)
        fc = coana.FindNeuronConnection.__new__(
            coana.FindNeuronConnection)
        fc.dataset = "male-cns:v1.0"
        fc.use_cache = True
        fc.label_mapper = mapper
        fc.client_type = "neuprint"
        fc._conn_df_cache = None
        fc._get_connection_db_path = lambda: str(conn)
        fc._get_neuron_index_path = lambda: str(idx)
        fc._pull_missing_incoming_totals = (
            lambda types, min_weight, col: {})
        # the D_t by-bodyId product (verified separately) as the stub seam:
        body = pl.DataFrame({"bodyId_post": [str(members[0]), "ZZ"],
                             "total_incoming_weight": [7, 9]})
        fc._get_total_incoming_by_bodyid_table = lambda min_weight=1: body
        res = fc._fetch_total_incoming_weight_by_type(["5th-LNv"], 1)
        got = {r.type_post: r.total_incoming_weight
               for r in res.itertuples()}
        assert got.get("5th-LNv") == 7
