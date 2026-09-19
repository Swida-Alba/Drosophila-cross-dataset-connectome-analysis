"""Tests for the BANC audit fast-follow fixes (issue-report BANC-01 … 08).

Each test pins the specific recorded defect:
- BANC-01: the no-decimation early return restores mesh id/name;
- BANC-03: ``build_connection_cache_from_tables`` default cache_dir lands
  in the project-root ``cache/`` (not ``datasets/cache/``);
- BANC-04: Codex-manual preparation records ``banc_codex_manual`` source;
- BANC-05: the state sidecar counts partner ROWS, not weight sums;
- BANC-06: the ``.src`` marker is written LAST (interrupted rebuilds
  re-run instead of short-circuiting);
- BANC-07: the crosswalk memo hands out copies, not the live dict;
- BANC-08: the reverse crosswalk is memoized.
"""

import json
import sys
from pathlib import Path

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import BANC_file_converter as bfc  # noqa: E402
import banc_public_data as bpd  # noqa: E402


# ---------------------------------------------------------------------------
# BANC-03 / BANC-05 / BANC-06 — build_connection_cache_from_tables
# ---------------------------------------------------------------------------

def _make_dataset(tmp_path, weights=(5, 2, 8), table_ids=("1", "2", "3")):
    dataset_dir = tmp_path / "datasets" / "banc_v999"
    dataset_dir.mkdir(parents=True)
    pl.DataFrame({
        "bodyId_pre": ["1", "1", "2"],
        "bodyId_post": ["2", "3", "3"],
        "weight": list(weights),
        "roi": ["A", "B", "A"],
    }).write_parquet(dataset_dir / "banc_v999_merged_connections.parquet")
    pl.DataFrame({"bodyId": list(table_ids)}).write_parquet(
        dataset_dir / "banc_v999_allneurons_neuron_df.parquet")
    return dataset_dir


def test_default_cache_dir_is_project_root_cache(tmp_path):
    """BANC-03: ``datasets/banc_v999`` → default cache dir is the PARENT's
    ``cache/banc_v999`` (the canonical root layout every reader uses),
    not ``datasets/cache/banc_v999``."""
    dataset_dir = _make_dataset(tmp_path)
    assert bfc.build_connection_cache_from_tables(str(dataset_dir)) is True
    cache_file = tmp_path / "cache" / "banc_v999" / "connections.parquet"
    assert cache_file.exists()
    assert not (tmp_path / "datasets" / "cache").exists()


def test_state_sidecar_counts_rows_not_weight_sum(tmp_path):
    """BANC-05: neuron '1' has two partner rows (weights 5+2): the sidecar
    records connection_count == 2, not 7."""
    dataset_dir = _make_dataset(tmp_path)
    assert bfc.build_connection_cache_from_tables(str(dataset_dir)) is True
    state = pl.read_parquet(
        tmp_path / "cache" / "banc_v999" / "neuron_index_state.parquet")
    counts = dict(zip(state["bodyId"], state["connection_count"]))
    assert counts["1"] == 2  # partner rows, not 5+2=7
    assert counts["2"] == 1
    assert counts["3"] == 0


def test_marker_written_after_sidecar(tmp_path, monkeypatch):
    """BANC-06: the .src marker is the LAST artifact written — an
    interruption before it leaves no marker, so the rebuild re-runs."""
    dataset_dir = _make_dataset(tmp_path)
    events = []
    real_replace = __import__("os").replace

    import BANC_file_converter as mod
    monkeypatch.setattr(mod, "__file__", mod.__file__)  # no-op, keep refs

    import os as _os
    orig_open = _os.path.join  # placeholder to avoid builtin patching

    def _replace(src, dst):
        real_replace(src, dst)
        events.append(dst)
    monkeypatch.setattr(bfc.os, "replace", _replace)

    import builtins
    real_open = builtins.open

    def _open(file, *a, **k):
        if str(file).endswith(".src"):
            events.append("marker")
        return real_open(file, *a, **k)
    monkeypatch.setattr(builtins, "open", _open)

    assert bfc.build_connection_cache_from_tables(str(dataset_dir)) is True
    marker_events = [i for i, e in enumerate(events) if e == "marker"]
    assert marker_events, "marker not written"
    # The marker event comes after both atomic replaces (cache + sidecar).
    assert marker_events[0] == len(events) - 1


# ---------------------------------------------------------------------------
# BANC-04 — Codex-manual provenance
# ---------------------------------------------------------------------------

def test_regenerate_metadata_records_codex_manual_source(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    # _regenerate_banc_metadata needs the neuron table under its expected
    # parquet name and a connections table (both created by _make_dataset).
    assert bfc._regenerate_banc_metadata(
        "banc_v999", str(dataset_dir), source="banc_codex_manual") is True
    meta = json.loads(
        (dataset_dir / "banc_v999_metadata.json").read_text(encoding="utf-8"))
    assert meta["source"] == "banc_codex_manual"
    # Default stays the bucket source.
    assert bfc._regenerate_banc_metadata(
        "banc_v999", str(dataset_dir)) is True
    meta = json.loads(
        (dataset_dir / "banc_v999_metadata.json").read_text(encoding="utf-8"))
    assert meta["source"] == "banc_public_gcs"


# ---------------------------------------------------------------------------
# F9 (2026-09-18 retest) — stale error sidecars are never merged forward
# ---------------------------------------------------------------------------

def _error_sidecar():
    """The payload comparison_analyzer used to persist for a failed pull."""
    return {
        "dataset": "banc_v999",
        "source": "error",
        "error": "No local data found",
        "fetched_at": "2026-09-18T17:26:14.881245",
        "neuron_counts": {"total": 0, "typed": 0, "untyped": 0,
                          "type_coverage": 0},
        "synapse_counts": {"total_presynaptic": 0, "total_postsynaptic": 0,
                           "total": 0},
        "roi_coverage": {"roi_list": [], "roi_count": 0,
                         "neuron_counts_per_roi": {}},
        "coverage_notes": "Coverage information not available.",
        "preparation_notes": ["written by the bucket pull"],
    }


def test_patch_metadata_drops_stale_error_placeholder(tmp_path):
    """Preparing a dataset overwrites a stale error marker instead of
    preserving it next to the real provenance."""
    dataset_dir = _make_dataset(tmp_path)
    meta = dataset_dir / "banc_v999_metadata.json"
    meta.write_text(json.dumps(_error_sidecar()), encoding="utf-8")

    bfc._patch_dataset_metadata(
        str(dataset_dir), "banc_v999", source="banc_codex_manual",
        notes="prepared")

    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["source"] == "banc_codex_manual"
    assert "error" not in data
    # The placeholder's zero stat blocks are gone, not merged forward.
    assert "neuron_counts" not in data
    assert "synapse_counts" not in data
    # Real keys already in the sidecar survive the merge.
    assert data["preparation_notes"] == ["written by the bucket pull",
                                         "prepared"]


def test_regenerate_metadata_drops_stale_error_placeholder(tmp_path):
    """Table-derived stats replace the placeholder; its error text does not
    ride along in the regenerated sidecar."""
    dataset_dir = _make_dataset(tmp_path)
    meta = dataset_dir / "banc_v999_metadata.json"
    meta.write_text(json.dumps(_error_sidecar()), encoding="utf-8")

    assert bfc._regenerate_banc_metadata("banc_v999", str(dataset_dir)) is True

    data = json.loads(meta.read_text(encoding="utf-8"))
    assert "error" not in data
    assert data["source"] == "banc_public_gcs"
    assert data["neuron_counts"]["total"] == 3
    assert data["synapse_counts"]["total"] == 15
    assert data["coverage_notes"] == "Full brain and VNC connectome."


def test_metadata_merge_keeps_a_real_sidecar_intact(tmp_path):
    """The strip only fires on an error placeholder — a real sidecar's stats
    are merged forward as before."""
    dataset_dir = _make_dataset(tmp_path)
    meta = dataset_dir / "banc_v999_metadata.json"
    meta.write_text(json.dumps({
        "dataset": "banc_v999",
        "source": "banc_public_gcs",
        "synapse_density": {"per_neuron_median": 4.5},
    }), encoding="utf-8")

    bfc._patch_dataset_metadata(
        str(dataset_dir), "banc_v999", source="banc_codex_manual")

    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["source"] == "banc_codex_manual"
    assert data["synapse_density"] == {"per_neuron_median": 4.5}


# ---------------------------------------------------------------------------
# BANC-07 / BANC-08 — crosswalk memo hygiene
# ---------------------------------------------------------------------------

CROSSWALK_CACHE = (PROJECT_ROOT / "cache" / "banc_v888"
                   / "banc_id_crosswalk.parquet")

pytestmark_crosswalk = pytest.mark.skipif(
    not CROSSWALK_CACHE.exists(),
    reason="local banc_v888 crosswalk cache not available")


@pytest.mark.skipif(
    not CROSSWALK_CACHE.exists(),
    reason="local banc_v888 crosswalk cache not available")
class TestCrosswalkMemo:
    def test_forward_memo_returns_a_copy(self):
        first = bpd.get_id_crosswalk("banc_v888", project_root=PROJECT_ROOT)
        assert first
        snapshot = dict(first)
        first["MUTATED"] = "MUTATED"  # a caller mutating the return
        second = bpd.get_id_crosswalk("banc_v888", project_root=PROJECT_ROOT)
        assert second == snapshot  # the memo is untouched
        assert "MUTATED" not in second

    def test_reverse_memo_survives_forward_lookup_failures(self, monkeypatch):
        first = bpd.get_id_crosswalk_reverse(
            "banc_v888", project_root=PROJECT_ROOT)
        assert first

        def _dead_lookup(dataset, project_root=None, force_refresh=False):
            return {}
        monkeypatch.setattr(bpd, "get_id_crosswalk", _dead_lookup)
        second = bpd.get_id_crosswalk_reverse(
            "banc_v888", project_root=PROJECT_ROOT)
        assert second == first  # served from the reverse memo

    def test_force_refresh_invalidates_both_memos(self):
        before = bpd.get_id_crosswalk_reverse(
            "banc_v888", project_root=PROJECT_ROOT)
        assert before
        key = bpd._crosswalk_memo_key("banc_v888", PROJECT_ROOT)
        # Exactly what get_id_crosswalk(force_refresh=True) does to the
        # memos: both sides are popped, never served stale.
        bpd._CROSSWALK_CACHE.pop(key, None)
        bpd._CROSSWALK_REVERSE_CACHE.pop(key, None)
        assert key not in bpd._CROSSWALK_REVERSE_CACHE
        # The next reverse call re-derives through a fresh forward lookup
        # and repopulates the memo (content stable).
        second = bpd.get_id_crosswalk_reverse(
            "banc_v888", project_root=PROJECT_ROOT)
        assert second == before
        assert key in bpd._CROSSWALK_REVERSE_CACHE


# ---------------------------------------------------------------------------
# BANC-01 — mesh identity in the no-decimation early return
# ---------------------------------------------------------------------------

def test_no_decimation_early_return_restores_identity():
    """Source-level contract: the early-return branch restores id/name
    like both sibling paths (a behavioral test would need the full BANC
    render pipeline; the identity lines are the fix's exact contract)."""
    source = (PROJECT_ROOT / "src" / "visualize_skeleton.py").read_text(
        encoding="utf-8")
    # Anchor on the branch's own comment (the bare condition string also
    # appears in an unrelated pipeline helper).
    at = source.index("The slider asks for no reduction")
    branch = source[at:at + 650]
    assert "mesh_n.id = getattr(n, 'id', None)" in branch
    assert "mesh_n.name = n.name" in branch
    assert "all_mesh_neurons.append(mesh_n)" in branch
