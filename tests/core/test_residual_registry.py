"""Residual registry: the single producer of the label mapper's effective type key.

Contract under test (_plan/issues-2026-10-13-label-mapper-type-space-completeness.md
§8-§9, decisions D1/D4/D5/D7):

* tier 1 — a mapper member takes its settled label; a direct bodyId member
  beats a whole-type/instance expansion whatever the group order;
* tier 2 — a NON-member of a collided native type (partially governed, or the
  name is itself a settled label) takes the terminal ``T*`` key, and enrollment
  is counted BEFORE any star is written (that ordering is what makes the
  disclosure able to fire at all);
* tier 3 — an untouched type keeps its raw name and an untyped neuron keeps its
  bodyId; neither is ever renamed, and no empty native type can collapse into a
  bare ``'*'`` bucket;
* ``population=None`` means residuals are NOT evaluated — membership still
  applies, and the registry says so instead of looking complete.

The structural test at the end is the gate that keeps this the only producer.
"""
from __future__ import annotations

import json
import pathlib
import re

import pandas as pd
import polars as pl
import pytest

from comparison.type_resolver import (
    RESIDUAL_SUFFIX,
    ResidualRegistry,
    build_type_key_map,
    get_residual_registry,
    is_residual_key,
    is_terminal_identity,
    residual_key,
)

DS = "banc_v888"

# 80001 is a member of the partially governed SMP540; 80002-80004 are its
# residuals.  90001/90002 carry the native name of a settled label (the second
# collision class).  70001 is the only neuron of a fully governed type.  60001
# is untouched.  50001/50002 are untyped non-members, 50003 an untyped member.
POPULATION = [
    ("80001", "SMP540", ""), ("80002", "SMP540", ""), ("80003", "SMP540", ""),
    ("80004", "SMP540", ""),
    ("90001", "DN1pD", ""), ("90002", "DN1pD", ""),
    ("70001", "CB3449", ""),
    ("60001", "XYZ", ""),
    ("50001", "", ""), ("50002", "", ""), ("50003", "", "miR"),
]

MAPPING = {
    "format": "drocat_custom_groups/v1",
    "source_mapping": {
        "custom_label": ["DN1pD", "CB3449-LAB", "INSTANCE-LAB"],
        DS: [["80001"], ["CB3449"], ["miR"]],
    },
    "target_mapping": {
        "custom_label": ["DN1pD", "CB3449-LAB", "INSTANCE-LAB"],
        DS: [["80001"], ["CB3449"], ["miR"]],
    },
}


def _population_frame(engine="polars"):
    rows = pd.DataFrame(
        {"bodyId": [b for b, _, _ in POPULATION],
         "type": [t for _, t, _ in POPULATION],
         "instance": [i for _, _, i in POPULATION]},
        dtype=object)
    return pl.from_pandas(rows) if engine == "polars" else rows


@pytest.fixture()
def mapper(tmp_path):
    from comparison.label_mapper import LabelMapper
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(MAPPING), encoding="utf-8")
    return LabelMapper(overall_mapping_json=str(path))


class TestResidualVocabulary:
    """``residual_key`` / ``is_residual_key`` / ``is_terminal_identity``."""

    def test_residual_key_shapes(self):
        assert residual_key("SMP540") == "SMP540*"
        assert residual_key("  SMP540  ") == "SMP540*"
        # An untyped neuron has no shared name to collide with: it must fall
        # through to its own bodyId instead of merging into a bare '*' bucket.
        assert residual_key("") == ""
        assert residual_key(None) == ""
        assert residual_key("   ") == ""

    def test_is_residual_key_is_hemisphere_aware(self):
        assert is_residual_key("SMP540*")
        # The engines append the suffix AFTER the rename (statvis coalesce,
        # coana._node_type_label), so the suffixed spelling is the one that
        # actually reaches the tier-0 guards.
        assert is_residual_key("SMP540*_L")
        assert is_residual_key("SMP540*_R")
        assert is_residual_key("SMP540*_U")
        assert not is_residual_key("SMP540")
        assert not is_residual_key("SMP540_L")
        assert not is_residual_key(RESIDUAL_SUFFIX)   # a bare tag is not a key
        assert not is_residual_key("")
        assert not is_residual_key(None)

    def test_terminal_identity_covers_labels_and_residuals(self):
        labels = frozenset({"DN1pD", "CB3449-LAB"})
        assert is_terminal_identity("DN1pD", labels)
        assert is_terminal_identity("DN1pD_L", labels)      # suffixed base
        assert is_terminal_identity("SMP540*", labels)      # residual
        assert is_terminal_identity("SMP540*_R", labels)    # suffixed residual
        assert not is_terminal_identity("SMP540", labels)
        assert not is_terminal_identity("SMP540", frozenset())


class TestResidualRegistry:
    def test_tier1_membership_tier2_residual_tier3_untouched(self, mapper):
        registry = ResidualRegistry(mapper, DS, _population_frame())
        key = registry.key_for
        assert key("80001") == "DN1pD"                 # direct member
        assert key("70001") == "CB3449-LAB"            # whole-type expansion
        assert key("80002") == "SMP540*"               # partially governed
        assert key("90001") == "DN1pD*"                # settled-label spelling
        assert key("60001") is None                    # untouched: raw name
        assert key("50001") is None                    # untyped: bodyId tier

    def test_instance_expansion_governs_its_whole_group(self, mapper):
        registry = ResidualRegistry(mapper, DS, _population_frame())
        # An instance-keyed group is governed as a whole, so its neurons take
        # the label and no instance group is ever collided.  The untyped
        # neurons the mapper does not know keep their bodyId tier (50001/50002
        # above) instead of merging into one bucket.
        assert registry.key_for("50003") == "INSTANCE-LAB"
        assert registry.key_for("50001") is None
        assert registry.key_for("50002") is None
        assert "SMP540*" in registry.bodyid_map().values()

    def test_direct_member_beats_a_type_expansion_in_any_group_order(
            self, tmp_path):
        # Two groups: one retypes the whole native type, one retypes one of its
        # neurons differently.  The per-neuron judgement must win, and must not
        # depend on the order the board wrote the rows in.  The type is then
        # fully governed, so nothing is left to star-tag.
        for order in ("type_first", "body_first"):
            labels = ["WHOLE-TYPE", "PER-NEURON"]
            members = [["SMP540"], ["80003"]]
            if order == "body_first":
                labels, members = labels[::-1], members[::-1]
            payload = {
                "format": "drocat_custom_groups/v1",
                "source_mapping": {"custom_label": labels, DS: members},
                "target_mapping": {"custom_label": labels, DS: members},
            }
            path = tmp_path / f"two_{order}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            from comparison.label_mapper import LabelMapper
            registry = ResidualRegistry(
                LabelMapper(overall_mapping_json=str(path)), DS,
                _population_frame())
            assert registry.key_for("80003") == "PER-NEURON", order
            assert registry.key_for("80002") == "WHOLE-TYPE", order
            assert registry.collided == {}, order

    def test_population_none_applies_membership_only(self, mapper):
        registry = ResidualRegistry(mapper, DS, None)
        assert registry.population_available is False
        assert registry.key_for("80001") == "DN1pD"    # tier 1 still applies
        assert registry.key_for("80002") is None       # no residual invented
        assert registry.collided == {}
        assert any("NOT evaluated" in line for line in registry.notes())

    def test_foreign_bodyid_is_classified_by_its_frames_native_type(self, mapper):
        # 99999 is in the connection frame but not in the neuron table.  Its
        # same-type cousins are star-tagged, so it must be too — otherwise one
        # native type is split across two keys in one table.
        registry = ResidualRegistry(mapper, DS, _population_frame())
        assert registry.key_for("99999", "SMP540") == "SMP540*"
        assert registry.key_for("99998", "XYZ") is None
        # ...and a name-keyed group governs a neuron the table does not know.
        assert registry.key_for("99997", "CB3449") == "CB3449-LAB"

    def test_collided_census_counts_membership_not_coverage(self, mapper):
        registry = ResidualRegistry(mapper, DS, _population_frame())
        assert registry.collided["SMP540"] == {
            "enrolled": 1, "population": 4, "residuals": 3,
            "key": "SMP540*"}
        assert registry.collided["DN1pD"]["enrolled"] == 0
        assert registry.collided["DN1pD"]["residuals"] == 2
        # A fully governed type and an untouched one are not collided at all.
        assert "CB3449" not in registry.collided
        assert "XYZ" not in registry.collided

    def test_pandas_and_polars_populations_agree(self, mapper):
        polars_map = build_type_key_map(mapper, DS, _population_frame("polars"))
        pandas_map = build_type_key_map(mapper, DS, _population_frame("pandas"))
        assert polars_map == pandas_map

    def test_empty_population_and_empty_mapper_are_harmless(self, mapper):
        assert build_type_key_map(None, DS, _population_frame()) == {}
        # No population means there is nothing to PRECOMPUTE: the flat map the
        # engines join on stays empty and membership is answered per row
        # through key_for.  That is exactly the D5 degradation — tier 1 applies,
        # residuals do not, and notes() says so.
        assert build_type_key_map(mapper, DS, None) == {}
        assert get_residual_registry(mapper, DS, None).key_for("80001") == "DN1pD"
        assert get_residual_registry(
            mapper, DS, None).key_for("99999", "SMP540") is None
        assert build_type_key_map(mapper, DS, pd.DataFrame({"x": [1]})) == {}

    def test_registry_is_cached_per_population_object(self, mapper):
        frame = _population_frame()
        assert get_residual_registry(mapper, DS, frame) is \
            get_residual_registry(mapper, DS, frame)
        assert get_residual_registry(mapper, DS, frame) is not \
            get_residual_registry(mapper, DS, _population_frame())

    def test_metadata_publishes_the_per_dataset_composition(self, mapper):
        meta = get_residual_registry(mapper, DS, _population_frame()).metadata()
        assert meta["dataset"] == DS
        assert meta["population_available"] is True
        assert meta["residual_keys"] == ["DN1pD*", "SMP540*"]
        assert meta["collided"]["SMP540"]["residuals"] == 3
        # D7: the key stays undecorated, so the composition it stands for has
        # to travel with it.
        assert meta["collided"]["SMP540"]["key"] == "SMP540*"

    def test_metadata_names_the_table_the_census_came_from(self, mapper,
                                                           tmp_path):
        """A census is auditable only if it says which file produced it.

        ``population_source``/``population_version`` let a reader re-derive or
        dispute a ``T*`` count against one specific neuron table (a released
        parquet vs a per-dataset csv) — the same record a comparison report
        reads back.
        """
        import statvis as sv
        table_dir = tmp_path / "datasets" / DS
        table_dir.mkdir(parents=True)
        table = table_dir / f"{DS}_allneurons_neuron_df.csv"
        pd.DataFrame({"bodyId": [b for b, _, _ in POPULATION],
                      "type": [t for _, t, _ in POPULATION],
                      "instance": [i for _, _, i in POPULATION]}).to_csv(
                          table, index=False)
        frame, source, version = sv.neuron_population_provenance(
            DS, str(tmp_path))
        assert source == str(table)
        meta = get_residual_registry(mapper, DS, frame, source=source,
                                     version=version).metadata()
        assert meta["registry_version"] == "1"
        assert meta["population_source"] == str(table)
        assert meta["population_version"] == version
        assert meta["population_available"] is True
        assert meta["governed_bodyids"] == 3

    def test_notes_disclose_exactly_when_there_is_something_to_say(self, mapper):
        lines = get_residual_registry(
            mapper, DS, _population_frame()).notes()
        assert len(lines) == 1
        assert lines[0].startswith("- [label mapper residuals]")
        assert "SMP540: 1 enrolled / 4 population (3 residual(s) -> 'SMP540*')" in lines[0]


class TestEnrichmentParity:
    """The delegates are callers of the registry, not second implementations.

    CI runs pytest, not the _plan probe, so the engine-level invariant is
    pinned here: for one population and one mapper, both engines must emit the
    registry's key for every neuron — at bodyId level as well as type level
    (the pandas/polars split is what LM-002 measured).
    """

    def _write_table(self, tmp_path):
        d = tmp_path / "datasets" / DS
        d.mkdir(parents=True)
        pd.DataFrame({"bodyId": [b for b, _, _ in POPULATION],
                      "type": [t for _, t, _ in POPULATION],
                      "instance": [i for _, _, i in POPULATION],
                      "post": [100] * len(POPULATION)}).to_csv(
                          d / f"{DS}_allneurons_neuron_df.csv", index=False)
        return str(tmp_path)

    def _conn(self):
        rows = [{"bodyId_pre": b, "bodyId_post": "80001",
                 "type_pre": t or b, "type_post": "SMP540", "weight": 10}
                for b, t, _i in POPULATION]
        # a frame-only neuron: in the connection table, not in the neuron table
        rows.append({"bodyId_pre": "99999", "bodyId_post": "80001",
                     "type_pre": "SMP540", "type_post": "SMP540",
                     "weight": 3})
        return pd.DataFrame(rows)

    @pytest.mark.parametrize("engine", ["pandas", "polars"])
    def test_engines_emit_the_registry_key(self, mapper, tmp_path, engine):
        import statvis as sv
        script_path = self._write_table(tmp_path)
        frame = self._conn()
        registry = get_residual_registry(
            mapper, DS, _population_frame("pandas"))
        expected = {b: (registry.key_for(b, t or b) or (t or b))
                    for b, t, _i in POPULATION}
        expected["99999"] = "SMP540*"
        bid, typ = (x.to_pandas() if hasattr(x, "to_pandas") else x
                    for x in sv.EnrichConnectionTable(
                        frame.copy(), dataset=DS, script_path=script_path,
                        label_mapper=mapper, engine=engine)[:2])
        for name, df in (("bodyId", bid), ("type", typ)):
            for b, want in expected.items():
                if name == "type" and b not in {r["bodyId_pre"]
                                                for r in frame.to_dict("records")}:
                    continue
                got = df[df["type_pre"].astype(str) == want]
                assert not got.empty, f"{engine} {name}: no row keyed {want!r}"
            if name == "bodyId":
                emitted = dict(zip(bid["bodyId_pre"].astype(str),
                                   bid["type_pre"].astype(str)))
                for b, want in expected.items():
                    assert emitted.get(b) == want, (engine, b,
                                                    emitted.get(b), want)
                # raw provenance travels beside the key (decision D2)
                assert "native_type_pre" in bid.columns
                assert "std_label_pre" not in bid.columns
                native = dict(zip(bid["bodyId_pre"].astype(str),
                                  bid["native_type_pre"].astype(str)))
                assert native["80002"] == "SMP540"
                assert native["70001"] == "CB3449"

    def test_collided_census_is_what_the_engines_emit(self, mapper, tmp_path):
        import statvis as sv
        script_path = self._write_table(tmp_path)
        registry = get_residual_registry(mapper, DS, _population_frame())
        _, typ = (x.to_pandas() if hasattr(x, "to_pandas") else x
                  for x in sv.EnrichConnectionTable(
                      self._conn(), dataset=DS, script_path=script_path,
                      label_mapper=mapper, engine="polars")[:2])
        emitted = {k for k in typ["type_pre"].astype(str)}
        assert registry.collided, "the census must not be empty here"
        for native, info in registry.collided.items():
            assert info["key"] in emitted
            assert native not in emitted or native in registry.labels


class TestOnlyOneProducer:
    """Structural gate: nothing outside the owner may mint a residual key."""

    ALLOWED = {"src/comparison/type_resolver.py"}
    ROOT = pathlib.Path(__file__).resolve().parents[2]

    def test_residual_key_is_emitted_only_by_the_owner(self):
        offenders = []
        for path in sorted(list((self.ROOT / "src").rglob("*.py"))
                           + list((self.ROOT / "ui").rglob("*.py"))):
            if "archive" in path.parts:
                continue
            rel = path.relative_to(self.ROOT).as_posix()
            if rel in self.ALLOWED:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            # ``residual_key(`` CALLS mint a key; importing the predicate
            # (is_residual_key) or naming it in a comment stays legal.
            for match in re.finditer(r"(?<![\w.])residual_key\s*\(", text):
                line = text[:match.start()].count("\n") + 1
                offenders.append(f"{rel}:{line}")
        assert not offenders, (
            "a second producer of the residual tag appeared; the boundary "
            "makes comparison.type_resolver.ResidualRegistry the only one "
            f"(offenders: {offenders})")

    def test_statvis_carries_no_inline_collision_rule(self):
        # The two inline copies are gone: statvis calls the registry, and its
        # remaining ``collided`` mentions live only in explanatory comments.
        text = (self.ROOT / "src" / "statvis.py").read_text(
            encoding="utf-8", errors="replace")
        code = "\n".join(line for line in text.split("\n")
                         if not line.lstrip().startswith("#"))
        assert "governed_by_type" not in code
        assert "mapper_labels" not in code
        assert "bodyid_label_map[neuron_id]" not in code


class TestPartnerCollapseFoldsOnlyGovernedTypes:
    """The similar-connectivity lane may fold a partner type only when the
    mapper governs that type's WHOLE population (plan decision D6).

    A mixed type is left under its raw key rather than folded, because the
    profile stores partners at type level: there is no per-neuron weight here
    with which to attribute part of the mass to the label and part to `T*`.
    Folding anyway — the behaviour this replaces — merged the residual
    population into the settled label, the exact hazard `T*` exists to stop.
    """

    LABELS = frozenset({"DN1pD", "CB3449-LAB", "INSTANCE-LAB"})

    def _comparer(self, population):
        from comparison.profile_comparator import ConnectivityProfileComparer as C
        comparer = C.__new__(C)

        class FakeProfiler:
            def get_bodyids_for_type(self, type_name, dataset):
                return list(population)

        comparer.profiler = FakeProfiler()
        comparer._partner_fold_mixed = {}
        return comparer

    def test_partially_governed_type_is_not_folded_and_is_recorded(self, mapper):
        from comparison.profile_comparator import ConnectivityProfileComparer as C
        comparer = self._comparer(["80001", "80002", "80003"])
        assert C._label_for_type_via_mapper(
            comparer, "SMP540", DS, mapper, self.LABELS) is None
        mixed = comparer._partner_fold_mixed[(DS, "SMP540")]
        assert mixed == {"label": "DN1pD", "residuals": 2, "population": 3}

    def test_fully_governed_type_still_folds(self, mapper):
        from comparison.profile_comparator import ConnectivityProfileComparer as C
        comparer = self._comparer(["80001"])
        assert C._label_for_type_via_mapper(
            comparer, "SMP540", DS, mapper, self.LABELS) == "DN1pD"
        assert comparer._partner_fold_mixed == {}

    def test_members_spanning_two_labels_stay_raw(self, mapper):
        from comparison.profile_comparator import ConnectivityProfileComparer as C
        comparer = self._comparer(["80001", "70001"])
        assert C._label_for_type_via_mapper(
            comparer, "MIXED", DS, mapper,
            self.LABELS | {"CB3449-LAB"}) is None

    def test_residual_keys_are_terminal_in_this_lane(self, mapper):
        # The analyzer's twin already honoured `T*`; this lane re-resolved it
        # through the auto type mapper because its predicate had no residual
        # clause (and none for the suffixed spelling the engines emit).
        from comparison.profile_comparator import ConnectivityProfileComparer as C
        comparer = self._comparer(["80001"])
        comparer._user_label_mapper = mapper
        assert C._user_label_governs(comparer, "DN1pD")
        assert C._user_label_governs(comparer, "SMP540*")
        assert C._user_label_governs(comparer, "SMP540*_L")
        assert not C._user_label_governs(comparer, "SMP540")


class TestDisclosureLands:
    """The census must reach the artifacts, not just exist in memory.

    The 2026-10-08 disclosure never fired for two independent reasons: the
    registry counted enrollment over a map that already held the stars (fixed
    above), and the engine stored it on ``self._residual_registry`` — an
    underscore attribute, which ``_run_export_attributes`` filters out of
    ``all_attributes.json``.
    """

    def _engine(self, mapper, tmp_path):
        import coana
        fc = coana.FindNeuronConnection.__new__(coana.FindNeuronConnection)
        fc.dataset = DS
        fc.script_path = str(tmp_path)
        fc.label_mapper = mapper
        fc._warn_notes = []
        fc.logs = []
        fc._vprint = lambda *a, **k: None
        return fc

    def test_engine_publishes_record_csv_and_note(self, mapper, tmp_path):
        fc = self._engine(mapper, tmp_path)
        registry = get_residual_registry(mapper, DS, _population_frame())
        fc._disclose_residuals(registry, str(tmp_path))

        record = fc.label_mapper_residuals          # public -> all_attributes
        assert record["residual_keys"] == ["DN1pD*", "SMP540*"]
        # 80001 (DN1pD), 70001 (CB3449-LAB), 50003 (INSTANCE-LAB)
        assert record["governed_bodyids"] == 3
        assert record["residual_bodyids"] == 5       # 2x DN1pD* + 3x SMP540*
        assert record["population_available"] is True

        out = tmp_path / "label_mapper_residuals.csv"
        assert out.exists()
        rows = pd.read_csv(out, dtype=str)
        assert list(rows["emitted_key"]) == ["DN1pD*", "SMP540*"]
        assert list(rows["native_type"]) == ["DN1pD", "SMP540"]
        # DN1pD: 2 of 2 unenrolled; SMP540: 3 of 4 unenrolled
        assert list(rows["residuals"]) == ["2", "3"]
        assert list(rows["enrolled"]) == ["0", "1"]

        assert any(str(n).startswith("- [label mapper residuals]")
                   for n in fc._warn_notes)
        # state-owned: a second call replaces, never duplicates
        fc._disclose_residuals(registry, str(tmp_path))
        assert sum(str(n).startswith("- [label mapper residuals]")
                   for n in fc._warn_notes) == 1

    def test_missing_population_is_disclosed_not_silent(self, mapper, tmp_path):
        fc = self._engine(mapper, tmp_path)
        registry = get_residual_registry(mapper, DS, None)
        fc._disclose_residuals(registry, str(tmp_path))
        assert fc.label_mapper_residuals["population_available"] is False
        assert fc.label_mapper_residuals["collided"] == {}
        assert any("NOT evaluated" in str(n) for n in fc._warn_notes)
        rows = pd.read_csv(tmp_path / "label_mapper_residuals.csv", dtype=str)
        assert rows.loc[0, "emitted_key"].startswith("NOT EVALUATED")

    def test_analyzer_reads_the_record_back(self, mapper, tmp_path):
        """The comparison's card reads the delegate's own record, per dataset."""
        from comparison.comparison_analyzer import ComparisonAnalyzer

        folder = tmp_path / "ds_one" / "minsyn_1"
        folder.mkdir(parents=True)
        record = {"dataset": "test:v1", "population_available": True,
                  "mapper_labels": 3, "mapper_members": 3,
                  "governed_bodyids": 3, "residual_bodyids": 5,
                  "collided": {"SMP540": {"enrolled": 1, "population": 4,
                                          "residuals": 3, "key": "SMP540*"}},
                  "residual_keys": ["SMP540*"]}
        (folder / "all_attributes.json").write_text(
            json.dumps({"label_mapper_residuals": record}), encoding="utf-8")

        class Params:
            thresholds = [1]

            def get_dataset_names(self):
                return ["test:v1"]

            def get_thresholds_for_dataset(self, ds):
                return [1]

            def get_dataset_output_path(self, ds, threshold):
                return str(folder)

            def _sanitize_name(self, ds):
                return ds.replace(":", "_").replace(".", "_")

        a = ComparisonAnalyzer.__new__(ComparisonAnalyzer)
        a.parameters = Params()
        a.label_mapper = mapper
        a._log = lambda *args, **kwargs: None
        census = a._label_mapper_residual_census()
        assert census["test:v1"]["collided"]["SMP540"]["residuals"] == 3
        assert a._label_mapper_residuals is census
        # A dataset whose run published nothing is RECOMPUTED, not skipped —
        # and with no neuron table reachable, the honest answer is "residuals
        # were not evaluated", never a silent empty census.
        (folder / "all_attributes.json").write_text("{}", encoding="utf-8")
        recomputed = a._label_mapper_residual_census()
        assert recomputed["test:v1"]["collided"] == {}
        assert recomputed["test:v1"]["population_available"] is False


class TestPolicyLaneStillMaps:
    """Decision 8: the merge-policy mapper is applied at ALIGNMENT time.

    ``raw_results`` are not pre-mapped, so this lane is a legitimate producer
    (native names in, group labels out). The bug it nearly regressed into is
    the opposite case: re-running a label lookup over an already-typed frame,
    which erases `T*`. Both are covered here.
    """

    def _metrics(self):
        from comparison.metrics import ComparisonMetrics
        return ComparisonMetrics.__new__(ComparisonMetrics)

    def test_policy_group_label_is_applied_to_native_names(self):
        from comparison.label_mapper import LabelMapper
        policy = LabelMapper(source_mapping_dict={
                                 "test:v1": [["A"]], "test:v2": [["X"]]},
                             source_labels=["grpA"],
                             target_mapping_dict={
                                 "test:v1": [["B"]], "test:v2": [["Y"]]},
                             target_labels=["grpB"])
        results = {"test:v1": {1: pd.DataFrame(
            {"type_pre": ["A"], "type_post": ["B"], "weight": [7]})}}
        out = self._metrics()._align_results_at_threshold(
            results, ["test:v1"], 1, label_mapper=policy)
        assert "grpA -> grpB" in out.index

    def test_an_emitted_residual_is_never_retyped(self):
        from comparison.label_mapper import LabelMapper
        policy = LabelMapper(source_mapping_dict={"test:v1": [["SMP540"]]},
                             source_labels=["grpA"],
                             target_mapping_dict={"test:v1": [["DN1pD"]]},
                             target_labels=["grpB"])
        typed = pd.DataFrame({"type_pre": ["SMP540*"], "type_post": ["grpB"],
                              "native_type_pre": ["SMP540"], "weight": [4]})
        out = self._metrics()._align_results_at_threshold(
            {"test:v1": {1: typed}}, ["test:v1"], 1, label_mapper=policy)
        # the terminal key survives; only the native name would have folded
        assert "SMP540* -> grpB" in out.index
        assert "grpA -> grpB" not in out.index


class TestStateNoteFamilies:
    """Each state writer replaces ITS OWN notes and never another's.

    Two writers append to the same ``user_warning_notes.txt`` — the tier-0
    label-mapper note and the auto-mapping/merge-policy note — and a real
    three-dataset clock comparison measured the consequence when the replace
    pass treated all state notes as one family: whichever writer ran last
    deleted the other's block.  Losing ``[user label mapper]`` also loses the
    ``[label mapper residuals]`` block, which is the only annotation of a
    cross-dataset ``T*`` row outside the CSVs.
    """

    HEADER = "User warning notes\n==================\n\n"
    TIER0 = ("[user label mapper] A user label mapper is active: 1 label(s).\n"
             "- DN1a\n\n"
             "[label mapper residuals] Non-enrolled neurons are keyed T*\n"
             "- banc_v888: SMP540* = 3/4 neuron(s) (1 enrolled)")
    REFRESHED = ("[user label mapper] A user label mapper is active: 2 labels."
                 "\n- DN1a\n- APDN3\n\n"
                 "[label mapper residuals] refreshed\n"
                 "- banc_v888: SMP540* = 1/4 neuron(s) (3 enrolled)")
    POLICY = ("[merge policy] anchored on flywire_FAFB_v783; 21 group(s)\n\n"
              "[same-name-first] 59 fan-out(s) were SELECTED")

    def _analyzer(self, tmp_path, existing):
        from comparison.comparison_analyzer import ComparisonAnalyzer

        class Params:
            def __init__(self, path):
                self.full_output_path = str(path)

        analyzer = ComparisonAnalyzer.__new__(ComparisonAnalyzer)
        analyzer.parameters = Params(tmp_path)
        analyzer._log = lambda *args, **kwargs: None
        (tmp_path / "user_warning_notes.txt").write_text(
            self.HEADER + existing + "\n", encoding="utf-8")
        return analyzer

    def _text(self, tmp_path):
        return (tmp_path / "user_warning_notes.txt").read_text(
            encoding="utf-8")

    def test_policy_writer_spares_the_tier0_census(self, tmp_path):
        analyzer = self._analyzer(tmp_path, self.TIER0)
        analyzer._replace_state_notes(["[merge policy] new counts"])
        text = self._text(tmp_path)
        assert "[user label mapper]" in text and "SMP540* = 3/4" in text
        assert "new counts" not in text  # replace pass only prunes, never adds

    def test_tier0_writer_spares_the_policy_family(self, tmp_path):
        analyzer = self._analyzer(tmp_path, self.POLICY)
        analyzer._replace_state_notes([self.TIER0])
        text = self._text(tmp_path)
        assert "[merge policy]" in text and "[same-name-first]" in text

    def test_a_reexport_refreshes_the_whole_nested_block(self, tmp_path):
        """Nested headers move with their owner: one copy, current counts."""
        analyzer = self._analyzer(tmp_path, self.TIER0)
        kept = analyzer._replace_state_notes([self.REFRESHED])
        analyzer._append_user_warning_notes(str(tmp_path), kept)
        text = self._text(tmp_path)
        assert text.count("[user label mapper]") == 1
        assert text.count("[label mapper residuals]") == 1
        assert "3/4" not in text and "1/4" in text


class TestVerifierLane:
    """The analyzer's load lane reads, and notices only a real disagreement.

    A name the mapper simply does not govern is normal (most keys in a frame
    are plain native types); the failure worth saying out loud is LM-004 — a
    native type this dataset's registry calls COLLIDED spelled WITHOUT its
    star, i.e. the tag was lost before the frame reached disk. Report it, and
    change nothing.
    """

    def _analyzer(self, mapper, population):
        from comparison.comparison_analyzer import ComparisonAnalyzer

        class Params:
            def _sanitize_name(self, ds):
                return ds.replace(":", "_").replace(".", "_")

        analyzer = ComparisonAnalyzer.__new__(ComparisonAnalyzer)
        analyzer.parameters = Params()
        analyzer.label_mapper = mapper
        analyzer.logged = []
        analyzer._log = lambda msg, *a, **k: analyzer.logged.append(str(msg))
        analyzer._read_local_neuron_frame = (
            lambda safe, with_provenance=False: (
                (population, 'neuron_table.csv', 'token') if with_provenance
                else population))
        return analyzer

    def _frame(self, pre):
        return pd.DataFrame({"bodyId_pre": ["80002"], "bodyId_post": ["80001"],
                             "type_pre": [pre], "type_post": ["CB3449-LAB"],
                             "weight": [5]})

    def test_a_bare_collided_name_is_reported_and_left_alone(self, mapper):
        analyzer = self._analyzer(mapper, _population_frame())
        out = analyzer._verify_label_mapping_in_dataframe(DS, 3,
                                                         self._frame("SMP540"))
        assert list(out["type_pre"]) == ["SMP540"]
        assert any("collided native type(s) by their BARE name" in m
                   for m in analyzer.logged)

    def test_an_ordinary_native_name_is_not_an_accusation(self, mapper):
        analyzer = self._analyzer(mapper, _population_frame())
        analyzer._verify_label_mapping_in_dataframe(DS, 3, self._frame("XYZ"))
        assert analyzer.logged == []

    def test_a_hemisphere_suffixed_bare_name_is_still_caught(self, mapper):
        analyzer = self._analyzer(mapper, _population_frame())
        analyzer._verify_label_mapping_in_dataframe(DS, 3,
                                                    self._frame("SMP540_L"))
        assert any("BARE name" in m for m in analyzer.logged)

    def test_without_a_population_the_lane_makes_no_claim(self, mapper):
        analyzer = self._analyzer(mapper, None)
        analyzer._verify_label_mapping_in_dataframe(DS, 3,
                                                    self._frame("SMP540"))
        assert analyzer.logged == []


class TestRunFolderReattach:
    """A re-export reattaches the mapping the run was produced with.

    ``parameters.json`` serializes scalars only, so ``for_run_folder`` used to
    hand back a mapper-less analyzer: the tier-0 note, the residual census and
    the ``key_class`` rows a re-export is supposed to REFRESH then described a
    run with no user mapper. The run's own ``label_map.json`` is the record —
    but it was not readable until 2026-10-14, because the writer omitted the
    ``format`` tag its own loader (and the grouping board's Load…) requires.
    """

    def _export(self, tmp_path, mapper):
        import types
        from comparison.comparison_analyzer import ComparisonAnalyzer

        analyzer = ComparisonAnalyzer.__new__(ComparisonAnalyzer)
        analyzer.parameters = types.SimpleNamespace(
            get_dataset_names=lambda: [DS], source_neurons=['DN1pD'],
            get_source_groups=lambda: [['DN1pD']], source_labels=None,
            target_neurons=['s-LNv'], get_target_groups=lambda: [['s-LNv']],
            target_labels=None, intermediate_neurons=None,
            auto_type_mapping=False, _auto_type_mapper=None)
        analyzer.label_mapper = mapper
        analyzer._log = lambda *args, **kwargs: None
        path = tmp_path / "label_map.json"
        analyzer._export_label_map(str(path))
        return path

    def test_the_export_round_trips_through_its_own_loader(self, mapper,
                                                          tmp_path):
        from comparison.label_mapper import PRESET_FORMAT_TAG, LabelMapper

        path = self._export(tmp_path, mapper)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["format"] == PRESET_FORMAT_TAG
        back = LabelMapper(overall_mapping_json=str(path))
        assert back.is_empty is False
        assert sorted(back.get_all_std_labels('source')) == \
            sorted(mapper.get_all_std_labels('source'))

    def test_reattach_gives_the_reexport_a_real_mapper(self, mapper, tmp_path):
        from comparison.comparison_parameters import ComparisonParameters

        path = self._export(tmp_path, mapper)
        restored = ComparisonParameters._label_mapper_from_run_folder(
            str(tmp_path))
        assert restored is not None
        assert restored is not mapper
        assert "DN1pD" in restored.get_all_std_labels('source')
        # A native type the mapper only partially governs is NOT a label.
        assert "SMP540" not in restored.get_all_std_labels('source')
        assert path.exists()

    def test_an_empty_export_reattaches_nothing(self, tmp_path):
        from comparison.comparison_parameters import ComparisonParameters

        (tmp_path / "label_map.json").write_text(json.dumps({
            "format": "drocat_custom_groups/v1",
            "source_mapping": {"custom_label": []},
            "target_mapping": {"custom_label": []},
        }), encoding="utf-8")
        assert ComparisonParameters._label_mapper_from_run_folder(
            str(tmp_path)) is None

    def test_a_foreign_json_is_not_a_mapper(self, tmp_path):
        from comparison.comparison_parameters import ComparisonParameters

        (tmp_path / "label_map.json").write_text('{"source_mapping": {}}',
                                                 encoding="utf-8")
        assert ComparisonParameters._label_mapper_from_run_folder(
            str(tmp_path)) is None
