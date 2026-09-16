"""Tests for the per-bridge type-level mapping export
(plan-type-mapper-fine-granularity-export.md).

Live-mapper fixtures (registry is local, no network):
- FAFB APDN3 -> male-cns: 8 prioritized bridge chains across 4 targets,
  4 selected, disjoint refined source pools {4, 2, 4, 2} summing to 12.
- FAFB s-CPDN3A -> male-cns: CB1791->SMP228 branch is 18 vs 12 and is
  never annotated 1-to-1.
- Same-name MCNS aMe12 -> FAFB aMe12: one chain-less branch with
  full-population pools and no fan-out annotation.
- Per-bridge rows / fine CSV round-trips keep FAFB-sized bodyIds exact.
"""

import csv
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.cross_dataset_type_mapper import (  # noqa: E402
    get_type_mapper,
)
from comparison.mapping_visualization import (  # noqa: E402
    PER_BRIDGE_HEADER,
    annotate_branch_records,
    branches_to_per_bridge_rows,
    build_fine_mapping_csv,
    default_branch_pool_fn,
)


@pytest.fixture(scope="module")
def mapper():
    return get_type_mapper()


@pytest.fixture(scope="module")
def pool_fn():
    return default_branch_pool_fn()


def _branches(mapper, pool_fn, source_type, src="flywire_FAFB_v783",
              tgt="male-cns:v1.0"):
    records = mapper.get_mapping_branches(
        source_type, src, tgt, pool_fn=pool_fn)
    annotate_branch_records(records)
    return records


def test_apdn3_branch_structure(mapper, pool_fn):
    records = _branches(mapper, pool_fn, "APDN3")
    assert records, "APDN3 must resolve per-bridge records"
    targets = {r["target_type"] for r in records}
    assert targets == {"CL125", "PLP080", "SLP249", "SLP250"}
    selected = [r for r in records if r["is_selected"]]
    assert len(selected) == 4
    # one selected chain per target, first by chain rank
    for tt in targets:
        sel = [r for r in selected if r["target_type"] == tt]
        assert len(sel) == 1
        best_rank = min(r["chain_rank"] for r in records
                        if r["target_type"] == tt)
        assert sel[0]["chain_rank"] == best_rank


def test_apdn3_refined_pools_disjoint_and_complete(mapper, pool_fn):
    records = _branches(mapper, pool_fn, "APDN3")
    selected = [r for r in records if r["is_selected"]]
    sizes = sorted(len(r["source_body_ids"]) for r in selected)
    assert sizes == [2, 2, 4, 4]
    flat = [int(b) for r in selected for b in r["source_body_ids"]]
    assert len(set(flat)) == 12, "branches must partition the parent pool"
    # refined, not full populations
    for r in selected:
        assert len(r["source_body_ids"]) < 12
        assert r["source_basis"] == "linker rows"
        assert r["linkers"], "selected APDN3 branches carry linkers"


def test_apdn3_annotations_resolved_by_linkers(mapper, pool_fn):
    records = _branches(mapper, pool_fn, "APDN3")
    selected = [r for r in records if r["is_selected"]]
    for r in selected:
        terms = set(r["annotation"].split(";"))
        assert "bifurcation" in terms          # 1 source -> 4 targets
        assert "resolved_by_linkers" in terms  # disjoint + linker evidence
        assert "branches_disjoint" in terms
        assert r["parent_status"] == "evidence_only"
        assert r["parent_relationship"] == "N-to-1"
        assert r["branch_of"] == "APDN3"


def test_s_cpdn3a_branch_is_many_to_many_not_one_to_one(mapper, pool_fn):
    records = _branches(mapper, pool_fn, "s-CPDN3A")
    sel = {r["target_type"]: r for r in records if r["is_selected"]}
    assert set(sel) == {"CB3118", "SMP216", "SMP228", "SMP229"}
    assert len(sel["CB3118"]["source_body_ids"]) == 6
    assert len(sel["SMP216"]["source_body_ids"]) == 4
    assert len(sel["SMP228"]["source_body_ids"]) == 18
    assert len(sel["SMP229"]["source_body_ids"]) == 10
    # within-branch pairing is 18 FAFB vs 12 MCNS: the annotation must not
    # claim 1-to-1 (evidence-based vocabulary only)
    assert "1-to-1" not in sel["SMP228"]["annotation"]


def test_same_name_single_branch(mapper, pool_fn):
    records = mapper.get_mapping_branches(
        "aMe12", "male-cns:v1.0", "flywire_FAFB_v783", pool_fn=pool_fn)
    annotate_branch_records(records)
    assert len(records) == 1
    r = records[0]
    assert r["target_type"] == "aMe12"
    assert r["is_selected"] is True
    # no fan-out: nothing to annotate even when a same-value crosswalk
    # linker carries the branch (male-cns flywireType='aMe12')
    assert r["annotation"] == ""
    assert r["source_body_ids"], "pools resolve for the single branch"


def test_per_bridge_rows_roundtrip(mapper, pool_fn):
    records = _branches(mapper, pool_fn, "APDN3")
    rows = branches_to_per_bridge_rows(records)
    assert len(rows) == len(records)
    for row in rows:
        assert set(row) == set(PER_BRIDGE_HEADER)
    # refined {…} cells parse back exactly; int64-exact for FAFB ids
    sel_rows = [r for r in rows if r["is_selected"]]
    parsed = []
    for row in sel_rows:
        cell = row["source_body_ids"]
        assert cell.startswith("{") and cell.endswith("}")
        parsed.extend(int(x) for x in cell[1:-1].split(",") if x.strip())
    assert len(parsed) == 12 and len(set(parsed)) == 12
    assert all(x > 2 ** 53 for x in parsed)  # FAFB-sized ids stayed exact


def test_fine_mapping_csv_roundtrip(mapper, pool_fn):
    records = _branches(mapper, pool_fn, "APDN3")
    text = build_fine_mapping_csv(records)
    rows = list(csv.DictReader(text.splitlines()))
    # one row per (source bodyId, BRIDGE): the 12 neurons ride 8 chains
    assert len(rows) == sum(len(r["source_body_ids"]) for r in records)
    selected_ids = {int(b) for r in records if r["is_selected"]
                    for b in r["source_body_ids"]}
    sel_rows = [r for r in rows if r["is_selected"] == "True"]
    assert {int(r["source_bodyId"]) for r in sel_rows} == selected_ids
    assert len(sel_rows) == 12
    # same branch record drives both presentations
    by_key = {(int(r["source_bodyId"]), r["target_type"],
               r["chain_rank"]): r for r in rows}
    for r in records:
        for b in r["source_body_ids"]:
            row = by_key[(int(b), r["target_type"], str(r["chain_rank"]))]
            assert row["annotation"] == r["annotation"]


def test_structure_only_export_without_pool_fn(mapper):
    records = mapper.get_mapping_branches(
        "APDN3", "flywire_FAFB_v783", "male-cns:v1.0", pool_fn=None)
    assert records, "structure records exist without pooling"
    for r in records:
        assert r["source_body_ids"] is None
        assert r["status"] == "unpooled"


def test_unmapped_and_conflict_return_empty(mapper, pool_fn):
    assert mapper.get_mapping_branches(
        "TmY3a", "flywire_FAFB_v783", "male-cns:v1.0",
        pool_fn=pool_fn) == []


def test_export_mapping_per_bridge_writes_file(mapper, pool_fn, tmp_path):
    out = tmp_path / "per_bridge.csv"
    n = mapper.export_mapping_per_bridge(
        str(out), source_types=["APDN3", "s-CPDN3A"],
        source_dataset="flywire_FAFB_v783",
        target_datasets=["male-cns:v1.0"], pool_fn=pool_fn)
    assert n > 0
    rows = list(csv.DictReader(out.read_text().splitlines()))
    # support_* trailing columns are additive bridge evidence
    assert rows and set(rows[0]) == set(PER_BRIDGE_HEADER) | {
        'support_votes', 'support_verified', 'support_auto',
        'support_linker_value'}
