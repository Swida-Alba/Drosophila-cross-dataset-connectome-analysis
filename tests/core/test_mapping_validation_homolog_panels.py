"""Tests for the TM VEV homolog panels (user 2026-09-26).

Covers: the ``serialize_topn_union`` payload rule (top-3 rank_union ∪
top-3 jaccard, deduped, chain order), the two new run artifacts
(``forward_matches.csv`` / ``target_matches.csv`` schemas + header
guarantee), the per-bodyId allocation assembly in ``collect_run_data``,
and the two new report tabs (empty state on legacy folders, grouped
rows, the union hover, and the offline morph ✓/✗ re-derivation).
"""

import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.body_id_resolver import (  # noqa: E402
    serialize_topn_union,
    topn_union_rows,
)
from comparison.mapping_validation import (  # noqa: E402
    RUN_FILE_LAYOUT,
    _RUN_CSV_SCHEMAS,
    _write_run_csv,
)
from comparison.mapping_validation_report import (  # noqa: E402
    _homolog_backward_tab,
    _homolog_forward_tab,
    _homolog_morph_cell,
    _topn_hover,
    build_report_document,
    collect_run_data,
)


def _frame(rows):
    return pd.DataFrame(rows)


def _scan_rows():
    # bid 1: jaccard-1 (jac 0.5 / ru rank 3)
    # bid 2: rank_union-1 (ru 0.9 / jac rank 5)
    # bid 3: second on both rankings, and the branch member
    # bid 6: jaccard-3 / ru-4 — in the jaccard window only
    return [
        {"target_bid": 1, "jaccard": 0.5, "rank_union": -0.1,
         "rank_union_rank": 3, "jaccard_rank": 1},
        {"target_bid": 2, "jaccard": 0.1, "rank_union": 0.9,
         "rank_union_rank": 1, "jaccard_rank": 5},
        {"target_bid": 3, "jaccard": 0.4, "rank_union": 0.2,
         "rank_union_rank": 2, "jaccard_rank": 2},
        {"target_bid": 4, "jaccard": 0.05, "rank_union": -0.3,
         "rank_union_rank": 6, "jaccard_rank": 4},
        {"target_bid": 5, "jaccard": 0.02, "rank_union": -0.35,
         "rank_union_rank": 7, "jaccard_rank": 6},
        {"target_bid": 6, "jaccard": 0.3, "rank_union": 0.0,
         "rank_union_rank": 4, "jaccard_rank": 3},
    ]


class TestSerializeTopnUnion:
    def test_the_union_is_the_top3_of_each_rank_column_deduped(self):
        # top-3 jaccard = {1, 3, 6}; top-3 rank_union = {2, 3, 1};
        # the union is {1, 2, 3, 6} — bid 2 is jaccard-invisible but
        # rank_union-1, exactly the hit a chain top-N would hide
        bids = topn_union_rows(_frame(_scan_rows()), k=3)
        assert bids["target_bid"].tolist() == [1, 3, 6, 2]

    def test_records_travel_in_chain_order_with_both_ranks(self):
        payload = serialize_topn_union(
            _frame(_scan_rows()),
            {1: "T1", 2: "T2", 3: "T3", 6: "T6"}, k=3, branch_pool=[3])
        recs = payload.split(";")
        assert [r.split("|")[2] for r in recs] == ["1", "3", "6", "2"]
        # ru_rank|jac_rank|bid|type|ru|jaccard|in_branch
        assert recs[0] == "3|1|1|T1|-0.1000|0.5000|0"
        assert recs[1] == "2|2|3|T3|0.2000|0.4000|1"  # the branch member
        assert recs[3] == "1|5|2|T2|0.9000|0.1000|0"

    def test_rows_that_rank_on_neither_column_are_never_picked(self):
        rows = _scan_rows() + [
            {"target_bid": 9, "jaccard": None, "rank_union": None,
             "rank_union_rank": None, "jaccard_rank": None}]
        bids = topn_union_rows(_frame(rows), k=3)
        assert 9 not in bids["target_bid"].tolist()

    def test_k_bounds_each_window_and_empty_inputs_are_blank(self):
        assert topn_union_rows(_frame(_scan_rows()), k=1)["target_bid"] \
            .tolist() == [1, 2]  # jaccard-1 and rank_union-1
        assert serialize_topn_union(None) == ""
        assert serialize_topn_union(_frame([]), k=3) == ""
        assert serialize_topn_union(_frame(_scan_rows()), k=0) == ""

    def test_a_ranked_row_with_an_unusable_bid_is_skipped_not_fatal(self):
        # self-review fix 2026-09-26: same NaN/None-bid tolerance as
        # serialize_backward_topN — a ranked row without a usable bodyId
        # must never crash the window
        rows = _scan_rows() + [
            {"target_bid": None, "jaccard": 0.9, "rank_union": 0.8,
             "rank_union_rank": 1, "jaccard_rank": 1}]
        bids = topn_union_rows(_frame(rows), k=3)
        assert 1 in bids["target_bid"].tolist()  # clean rows still picked
        assert bids["target_bid"].isna().sum() == 0


def test_the_two_homolog_artifacts_are_registered_and_header_guaranteed(
        tmp_path):
    assert RUN_FILE_LAYOUT["forward_matches.csv"] == "validation"
    assert RUN_FILE_LAYOUT["target_matches.csv"] == "expansion"
    for name in ("forward_matches.csv", "target_matches.csv"):
        assert name in _RUN_CSV_SCHEMAS
    _write_run_csv(tmp_path, "forward_matches.csv", [])
    _write_run_csv(tmp_path, "target_matches.csv", [])
    fwd = tmp_path / "validation" / "forward_matches.csv"
    tgt = tmp_path / "expansion" / "target_matches.csv"
    assert fwd.exists() and tgt.exists()
    with open(fwd, newline="", encoding="utf-8") as f:
        assert f.readline().strip().startswith("source_bodyId,source_type")
    with open(tgt, newline="", encoding="utf-8") as f:
        assert f.readline().strip().startswith(
            "target_bodyId,target_type,pool_category,pool_branches")


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "type-map_dsA_to_dsB_test_20260926_000000"
    rd.mkdir()
    (rd / "parameters.json").write_text(json.dumps({
        "source_dataset": "dsA", "target_dataset": "dsB",
        "query_types": ["q1"],
    }))
    return rd


def _write_csv(path: Path, header: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


_FWD_HEADER = ["source_bodyId", "source_type", "primary_target_bodyId",
               "primary_target_type", "primary_jaccard",
               "primary_rank_union", "primary_in_branch", "forward_topN",
               "n_scanned", "scanned_at", "morph_v2_similarity",
               "morph_pool_ref", "morph_bar_kind"]
_TGT_HEADER = ["target_bodyId", "target_type", "pool_category",
               "pool_branches", "primary_source_bodyId",
               "primary_source_type", "primary_jaccard",
               "primary_rank_union", "primary_in_branch",
               "backward_topN_union", "n_scanned", "scanned_at",
               "morph_v2_similarity", "morph_pool_ref", "morph_bar_kind"]
_PAYLOAD = ("2|1|301|X|-0.1000|0.5000|1;1|2|302|Y|0.2000|0.4000|0")


def test_legacy_folder_renders_the_absent_state(run_dir: Path):
    d = collect_run_data(run_dir)
    assert d["forward_available"] is False
    assert d["target_match_available"] is False
    fwd = _homolog_forward_tab(d)
    assert "forward_matches.csv absent" in fwd
    bwd = _homolog_backward_tab(d)
    assert "target_matches.csv absent" in bwd
    # the full document still assembles over a legacy folder
    html = build_report_document(d)
    assert "Homolog · forward" in html and "Homolog · backward" in html


def test_forward_tab_groups_rows_by_allocation_with_union_hover(
        run_dir: Path):
    _write_csv(run_dir / "forward_matches.csv", _FWD_HEADER, [
        # assigned source with a morph value on the primary pair
        ["101", "A", "301", "X", 0.5, -0.1, True, _PAYLOAD, 40, "run",
         0.7, None, ""],
        # unpaired source without a profile
        ["102", "B", "", "", None, None, "", "", 0, "no_profile",
         None, None, ""],
    ])
    _write_csv(run_dir / "validation_results.csv", [
        "query", "source_type", "target_type", "source_bodyId", "verdict",
        "target_bodyId"],
        [["q1", "A", "X", "101", "verified", "301"]])
    d = collect_run_data(run_dir)
    assert d["forward_available"] is True
    alloc = d["forward_alloc"]["101"]
    assert alloc["verdict"] == "verified"
    html = _homolog_forward_tab(d)
    assert "2 source bodyIds" in html
    # one row per bodyId with the primary match + the union hover payload
    assert "101" in html and "301 · X" in html
    # the payload round-trips parsed (the hover mini-table, not raw)
    assert "<td>301</td>" in html and "<td>302</td>" in html
    assert "<td>-0.1000</td>" in html and "<td>0.4000</td>" in html
    assert "top-3 rank_union ∪ top-3 jaccard" in html
    assert "in a branch pool" in html
    # the no-profile source carries its reason, never a silent dash
    assert "no profile" in html
    assert "mapped · verified" in html


def test_backward_tab_uses_pool_category_and_union_hover(run_dir: Path):
    _write_csv(run_dir / "target_matches.csv", _TGT_HEADER, [
        ["901", "X", "verified", "A→X", "201", "A", 0.4, 0.2, True,
         _PAYLOAD, 12, "run", None, None, ""],
        ["902", "Y", "", "", "", "", None, None, "", "", 0,
         "no_profile", None, None, ""],
    ])
    d = collect_run_data(run_dir)
    html = _homolog_backward_tab(d)
    assert "2 target bodyIds" in html
    assert "verified" in html and "A→X" in html
    assert "201 · A" in html
    assert "<td>301</td>" in html  # union payload parsed into the hover
    assert "unallocated" in html  # the target with no pool category


def test_morph_cell_rederives_the_branch_bar_verdict_offline():
    bars = {"A->X": {"candidate_kind": "track_a_backup",
                     "backup_floor": 0.55, "native_floor": None,
                     "null_bar": None}}
    row = {"source_type": "A", "primary_target_type": "X",
           "morph_v2_similarity": 0.6, "morph_pool_ref": None}
    assert "✓" in _homolog_morph_cell(row, bars)
    row["morph_v2_similarity"] = 0.5
    assert "✗" in _homolog_morph_cell(row, bars)
    # a native bar compares pool_ref, not the Track-A value
    bars_native = {"A->X": {"candidate_kind": "native",
                            "native_floor": 0.45, "backup_floor": None,
                            "null_bar": None}}
    row_native = {"source_type": "A", "primary_target_type": "X",
                  "morph_v2_similarity": 0.1, "morph_pool_ref": 0.5}
    cell = _homolog_morph_cell(row_native, bars_native)
    assert "✓" in cell and "0.5000" in cell
    # an out-map pair grades against its exported null bar
    row_null = {"source_type": "A", "primary_target_type": "Z",
                "morph_v2_similarity": 0.31, "morph_pool_ref": None,
                "morph_bar": 0.3, "morph_bar_kind": "null"}
    assert "✓" in _homolog_morph_cell(row_null, {})
    # nothing scored reads as its own state, never a bare dash
    assert "not scored" in _homolog_morph_cell(
        {"source_type": "A", "primary_target_type": "X"}, {})


def test_the_union_hover_names_the_window_not_the_chain_topn():
    html = _topn_hover(_PAYLOAD, "homolog neighbourhood — top-3 "
                       "rank_union ∪ top-3 jaccard (chain order)")
    assert "top-3 rank_union ∪ top-3 jaccard" in html
    assert "302" in html  # both records render
    assert _topn_hover("") == ""


# ---------------------------------------------------------------------------
# pipeline-side capture + finalize (synthetic profiles, no run)
# ---------------------------------------------------------------------------

from comparison.connectivity_profiler import ConnectivityProfile  # noqa: E402
from comparison.mapping_validation import (  # noqa: E402
    MappingValidator,
    TypePair,
    expanded_vector,
    prep_target_stats,
    scan_source,
)


def _profile(bid, up, dn):
    return ConnectivityProfile(
        neuron_id=bid, dataset='test',
        upstream_partners=dict(up), downstream_partners=dict(dn),
        actual_upstream_count=len(up), actual_downstream_count=len(dn))


def _capture_validator():
    v = MappingValidator.__new__(MappingValidator)
    v.pairs = []
    return v


def test_forward_capture_records_chain_best_union_and_no_profile():
    v = _capture_validator()
    pair = TypePair(source_dataset='dsA', source_type='A',
                    source_pool=[1, 2], target_dataset='dsB',
                    target_type='X', target_pool=[201])
    tgt_vectors = {
        201: expanded_vector(_profile(201, {'A': 9, 'B': 5}, {'X': 4}),
                             None),
        202: expanded_vector(_profile(202, {'B': 3}, {'Y': 2}), None)}
    target_stats = prep_target_stats(tgt_vectors)
    scans = {1: scan_source(
        expanded_vector(_profile(1, {'A': 10, 'B': 8}, {'P': 6}), None),
        target_stats)}
    # bid 2 has no usable profile -> no scan -> a no_profile row
    v._record_forward_matches('A', [pair], scans, {201: 'X', 202: 'Y'})
    rows = v._forward_match_rows
    assert [r['scanned_at'] for r in rows] == ['run', 'no_profile']
    r1 = rows[0]
    assert r1['primary_target_bodyId'] == 201
    assert r1['primary_in_branch'] is True
    assert r1['primary_target_type'] == 'X'
    first_hit = r1['forward_topN'].split(';')[0].split('|')
    assert first_hit[2] == '201' and first_hit[3] == 'X'
    assert first_hit[6] == '1'  # the branch pool member is flagged
    # a second capture of the same source does not duplicate rows
    v._record_forward_matches('A', [pair], scans, {201: 'X', 202: 'Y'})
    assert len(v._forward_match_rows) == 2


def test_morph_pair_index_reads_fills_through_fill_pair():
    v = _capture_validator()
    idx = v._morph_pair_index(
        [{'source_bodyId': 1, 'target_bodyId': 201,
          'morph_v2_similarity': 0.42}],
        [],
        [{'side': 'source', 'bodyId': 1, 'proposal_bodyId': 202,
          'morph_pool_ref': 0.6, 'bar_kind': 'native'}],
        [{'source_bodyId': 1, 'target_bodyId': 203,
          'morph_v2_similarity': 0.2, 'morph_bar': 0.1,
          'morph_bar_kind': 'null'}],
        [])
    assert idx[(1, 201)]['morph_v2_similarity'] == 0.42
    # the fill row's pair is spelled the side it fills
    assert idx[(1, 202)]['morph_pool_ref'] == 0.6
    assert idx[(1, 202)]['morph_bar_kind'] == 'native'
    assert idx[(1, 203)]['morph_bar_kind'] == 'null'


def test_finalize_attaches_morph_joins_to_forward_rows():
    v = _capture_validator()
    v._forward_match_rows = [{
        'source_bodyId': 1, 'source_type': 'A',
        'primary_target_bodyId': 201, 'primary_target_type': 'X',
        'primary_jaccard': 0.5, 'primary_rank_union': -0.1,
        'primary_in_branch': True, 'forward_topN': 'x', 'n_scanned': 3,
        'scanned_at': 'run', 'morph_v2_similarity': None,
        'morph_pool_ref': None, 'morph_bar_kind': ''}]
    idx = v._morph_pair_index(
        [{'source_bodyId': 1, 'target_bodyId': 201,
          'morph_v2_similarity': 0.42}], [], [], [], [])
    rows = v._finalize_forward_match_rows(idx)
    assert rows[0]['morph_v2_similarity'] == 0.42
    assert rows[0]['morph_pool_ref'] is None


def test_appeared_target_bids_spells_fills_through_fill_pair():
    v = _capture_validator()
    v._backward_matches_rows = [{'member_bodyId': 909}]
    bids = v._appeared_target_bids(
        [{'target_bodyId': 901, 'ru_top_target_bodyId': 902}],
        [], [], [],
        [{'side': 'source', 'bodyId': 1, 'proposal_bodyId': 903},
         {'side': 'target', 'bodyId': 905, 'proposal_bodyId': 2}],
        [{'target_bodyId': 904}], [], [], [], [])
    assert bids == [901, 902, 903, 904, 905, 909]
    # a source bodyId never enters the target universe (fill side='target')
    assert 2 not in bids


def test_pooling_engine_targets_enter_the_backward_universe():
    # self-review fix 2026-09-26: a pooling run's own pool is named by no
    # supervised artifact, so the 5e universe must read it directly
    v = _capture_validator()
    v._backward_matches_rows = []
    v._pooling = {'pool': [{'target_bodyId': 950}, {'target_bodyId': 951}]}
    assert v._appeared_target_bids([], [], [], [], [], [], [], [], [],
                                   []) == [950, 951]


def test_deep_window_rows_reach_the_forward_allocation(run_dir: Path):
    # self-review fix 2026-09-26: the `examinees` bin lives in
    # deep_candidates.csv, so its rows must join the allocation
    _write_csv(run_dir / "forward_matches.csv", _FWD_HEADER, [
        ["103", "C", "303", "Z", 0.3, 0.1, True, _PAYLOAD, 20, "run",
         None, None, ""]])
    _write_csv(run_dir / "deep_candidates.csv",
               ["source_bodyId", "category"], [["103", "examinees"]])
    d = collect_run_data(run_dir)
    assert d["forward_alloc"]["103"]["bins"] == ["examinees"]
    html = _homolog_forward_tab(d)
    assert "examinees · C" in html


def test_finalize_target_rows_attach_categories_branches_and_morph():
    v = _capture_validator()
    v._target_match_rows = [{
        'target_bodyId': 901, 'target_type': 'X',
        'primary_source_bodyId': 201, 'primary_source_type': '',
        'primary_jaccard': 0.4, 'primary_rank_union': 0.2,
        'primary_in_branch': True, 'backward_topN_union': 'x',
        'n_scanned': 12, 'scanned_at': 'run'}]
    pool_detail = [{'target_bodyId': 901, 'category': 'verified',
                    'source_type': 'A', 'target_type': 'X'}]
    idx = v._morph_pair_index(
        [{'source_bodyId': 201, 'target_bodyId': 901,
          'morph_v2_similarity': 0.5, 'bar_kind': 'null'}], [], [], [], [])
    rows = v._finalize_target_match_rows(pool_detail, idx)
    assert rows[0]['pool_category'] == 'verified'
    assert rows[0]['pool_branches'] == 'A→X'
    assert rows[0]['primary_source_type'] == ''
    assert rows[0]['morph_v2_similarity'] == 0.5
    assert rows[0]['morph_bar_kind'] == 'null'
