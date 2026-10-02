#!/usr/bin/env python
"""Tests for FindNetwork — the mutual direct-connection network of the
queried neurons (source == target semantics, FindAllPath backend for
enrichment/hemisphere analysis, network + heatmap only, no Sankey)."""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))

_NETWORK_TYPES = {"S": "TS", "A": "TA", "B": "TB", "X": "TX", "T": "TT"}


def _make_network_fc(monkeypatch, tmp_path, edges, node_ids, **overrides):
    """Build a FindNeuronConnection wired for an offline FindNetwork run.

    Returns (fc, fetch_calls, enrich_inputs, viz_calls, logs). Connection
    fetching is served from the static ``edges`` list (pre, post, weight),
    filtered by min_weight; enrichment and visualization are fakes that
    record their inputs.
    """
    import coana

    edge_set = [(str(u), str(v), w) for (u, v, w) in edges]
    fetch_calls = []

    def fake_fetch(self, upstream_bodyIds, downstream_bodyIds=None,
                   min_weight=None, min_conn_ratio=None,
                   min_traversal_prob=None, **kwargs):
        ups = {str(u) for u in upstream_bodyIds}
        fetch_calls.append(sorted(ups))
        floor = int(min_weight) if min_weight else 1
        rows = [(u, v, w) for (u, v, w) in edge_set if u in ups and w >= floor]
        if not rows:
            return pd.DataFrame(columns=["bodyId_pre", "bodyId_post",
                                         "weight", "type_pre", "type_post"])
        return pd.DataFrame({
            "bodyId_pre": [r[0] for r in rows],
            "bodyId_post": [r[1] for r in rows],
            "weight": [r[2] for r in rows],
            "type_pre": [_NETWORK_TYPES[r[0]] for r in rows],
            "type_post": [_NETWORK_TYPES[r[1]] for r in rows],
        })

    def fake_neurons(self, bodyIds, columns=None, **kwargs):
        ids = [str(b) for b in bodyIds]
        return pd.DataFrame({
            "bodyId": ids,
            "type": [_NETWORK_TYPES.get(b, b) for b in ids],
            "post": [100] * len(ids),
        })

    enrich_inputs = {}

    def fake_enrich(conn, traversal_probability_threshold=None, dataset=None,
                    script_path=None, target_neurons_df=None, label_mapper=None,
                    global_incoming_weights=None, separate_hemispheres=False,
                    global_incoming_body_weights=None, **kwargs):
        enrich_inputs["conn"] = conn
        enrich_inputs["separate_hemispheres"] = separate_hemispheres
        conn_e = conn.copy()
        conn_e["traversal_probability"] = 0.5
        conn_e["connection_ratio"] = 0.5
        conn_t = (conn_e.groupby(["type_pre", "type_post"], as_index=False)["weight"].sum())
        conn_t["traversal_probability"] = 0.5
        conn_t["connection_ratio"] = 0.5
        return conn_e, conn_t, None

    viz_calls = []

    class FakeVisualizePath:
        def __init__(self, *args, **kwargs):
            viz_calls.append(("init", kwargs))
            self.G_network = None

        def visualize(self, **kwargs):
            viz_calls.append(("visualize", kwargs))

    monkeypatch.setattr(coana.FindNeuronConnection,
                        "_fetch_connections_with_cache", fake_fetch)
    monkeypatch.setattr(coana.FindNeuronConnection,
                        "_fetch_neurons_local_or_api", fake_neurons)
    monkeypatch.setattr(coana.FindNeuronConnection,
                        "_fetch_total_incoming_weight_by_type",
                        lambda self, *a, **k: None)
    monkeypatch.setattr(coana.FindNeuronConnection,
                        "_fetch_total_incoming_weight",
                        lambda self, *a, **k: None)
    monkeypatch.setattr(coana.sv, "EnrichConnectionTable", fake_enrich)
    monkeypatch.setattr(coana, "VisualizePath", FakeVisualizePath)

    fc = object.__new__(coana.FindNeuronConnection)
    logs = []
    fc._vprint = lambda msg="", level="full", end="\n", flush=False: logs.append(str(msg))
    fc.source_df = pd.DataFrame({
        "bodyId": list(node_ids),
        "type": [_NETWORK_TYPES[n] for n in node_ids],
    })
    fc.target_df = fc.source_df.copy()
    fc.saveas = ""
    fc.output_dir = str(tmp_path)
    fc.source_fname = "grp"
    fc.target_fname = "grp"
    fc.dataset = "test:v1"
    fc.min_synapse_num = overrides.get("min_synapse_num", 1)
    fc.min_ratio = 0.0
    fc.min_traversal_probability = 0.0
    fc.parameter_dict = {}
    fc.parameter_df = pd.DataFrame()
    fc.label_mapper = None
    fc.separate_hemispheres = overrides.get("separate_hemispheres", False)
    fc.hemisphere_filter = "both"
    fc.symmetry_analysis = overrides.get("symmetry_analysis", False)
    fc.keep_only_hemisphere_conserved_connections = overrides.get(
        "keep_only_hemisphere_conserved_connections", False)
    fc.skip_bodyId = overrides.get("skip_bodyId", True)
    fc.edgeN_limit = 0
    fc.network_layout = "hierarchical"
    fc.showfig = False
    fc.output_format = "csv"
    fc.verbose_mode = "silent"
    fc.script_path = str(PROJECT_ROOT)
    fc._warn_notes = []
    return fc, fetch_calls, enrich_inputs, viz_calls, logs


class TestFindNetwork:
    def test_keeps_only_within_set_edges_and_outputs(self, monkeypatch, tmp_path):
        # X and T are OUTSIDE the query set {S, A, B}: their edges must be
        # dropped; both directions within the set are kept.
        edges = [("S", "A", 10), ("A", "S", 8), ("S", "B", 6),
                 ("S", "X", 50), ("A", "T", 50), ("X", "S", 50)]
        fc, fetch_calls, enrich_inputs, _, logs = _make_network_fc(
            monkeypatch, tmp_path, edges, ["S", "A", "B"])
        fc.FindNetwork()

        assert fetch_calls == [["A", "B", "S"]]  # single fetch of the set
        kept = set(zip(enrich_inputs["conn"]["bodyId_pre"],
                       enrich_inputs["conn"]["bodyId_post"]))
        assert kept == {("S", "A"), ("A", "S"), ("S", "B")}

        base = os.path.basename(fc.network_folder)
        assert base.startswith("find-network_")
        details = os.path.join(fc.network_folder, "data_details")
        assert os.path.exists(os.path.join(details, "connection_type.csv"))
        assert os.path.exists(os.path.join(details, "neurons.csv"))
        assert os.path.exists(os.path.join(fc.network_folder, "parameters.txt"))
        # skip_bodyId=True by default -> no bodyId table
        assert not os.path.exists(os.path.join(details, "connection_info_bodyId.csv"))

    def test_bodyId_table_saved_when_not_skipped(self, monkeypatch, tmp_path):
        edges = [("S", "A", 10)]
        fc, _, _, _, _ = _make_network_fc(
            monkeypatch, tmp_path, edges, ["S", "A"], skip_bodyId=False)
        fc.FindNetwork()
        details = os.path.join(fc.network_folder, "data_details")
        assert os.path.exists(os.path.join(details, "connection_info_bodyId.csv"))

    def test_empty_result_saves_minimal_outputs(self, monkeypatch, tmp_path):
        # No within-set connections at all.
        edges = [("S", "X", 50), ("X", "A", 50)]
        fc, _, enrich_inputs, viz_calls, logs = _make_network_fc(
            monkeypatch, tmp_path, edges, ["S", "A"])
        fc.FindNetwork()

        assert "conn" not in enrich_inputs             # enrichment never ran
        assert not viz_calls                            # no visualization
        assert any("No direct connections found" in m for m in logs)
        # completeness notice pointing to Find Path + Find Reciprocal
        assert any("Find Reciprocal Connections" in m for m in logs)
        details = os.path.join(fc.network_folder, "data_details")
        assert os.path.exists(os.path.join(details, "neurons.csv"))
        assert not os.path.exists(os.path.join(details, "connection_type.csv"))

    def test_visualization_contract_network_heatmap_no_sankey(
            self, monkeypatch, tmp_path):
        edges = [("S", "A", 10), ("A", "S", 8)]
        fc, _, _, viz_calls, _ = _make_network_fc(
            monkeypatch, tmp_path, edges, ["S", "A"])
        fc.FindNetwork()

        inits = [kw for name, kw in viz_calls if name == "init"]
        runs = [kw for name, kw in viz_calls if name == "visualize"]
        assert len(inits) == 1 and len(runs) == 1
        assert inits[0].get("save_data_matrices") is False
        assert runs[0] == {"plot_network": True, "plot_heatmap": True,
                           "plot_Sankey": False}
        # artifacts organized under visualization/ (input CSV kept for
        # reproducibility)
        assert os.path.exists(os.path.join(
            fc.network_folder, "visualization", "visualization_data",
            "network_edges_input.csv"))

    def test_hemisphere_flags_passed_to_enrichment(self, monkeypatch, tmp_path):
        edges = [("S", "A", 10)]
        fc, _, enrich_inputs, _, _ = _make_network_fc(
            monkeypatch, tmp_path, edges, ["S", "A"], separate_hemispheres=True)
        fc.FindNetwork()
        assert enrich_inputs["separate_hemispheres"] is True

    def test_step_progress_events_match_pipeline_phases(self, monkeypatch,
                                                        tmp_path, capsys):
        """FindNetwork emits the 5-step [DROCAT][progress] protocol in
        backend execution order: fetch -> enrich -> save -> visualize
        (step 1 comes from InitializeNeuronInfo before FindNetwork runs)."""
        edges = [("S", "A", 10), ("A", "S", 8)]
        fc, _, _, _, _ = _make_network_fc(
            monkeypatch, tmp_path, edges, ["S", "A"])
        fc.verbose_mode = "full"
        fc.progress_events = True  # opt-in (UI-generated scripts set this)
        fc.FindNetwork()

        events = [ln.strip() for ln in capsys.readouterr().out.splitlines()
                  if "[DROCAT][progress]" in ln]
        assert events == [
            "[DROCAT][progress] 2/5 Fetching mutual direct connections",
            "[DROCAT][progress] 3/5 Enriching and filtering network edges",
            "[DROCAT][progress] 4/5 Saving network data",
            "[DROCAT][progress] 5/5 Building network visualizations",
        ]

    def test_step_progress_silent_mode_suppresses_events(self, monkeypatch,
                                                         tmp_path, capsys):
        """The structured events are opt-in: runs without progress_events
        print none even in full-verbose mode."""
        edges = [("S", "A", 10)]
        fc, _, _, _, _ = _make_network_fc(
            monkeypatch, tmp_path, edges, ["S", "A"])
        fc.verbose_mode = "full"
        fc.FindNetwork()

        out = capsys.readouterr().out
        assert "[DROCAT][progress]" not in out

    def test_step_progress_empty_network_stops_at_fetch(self, monkeypatch,
                                                        tmp_path, capsys):
        """A run with no within-set connections stops after the fetch step."""
        edges = [("S", "X", 50)]
        fc, _, _, _, _ = _make_network_fc(
            monkeypatch, tmp_path, edges, ["S", "A"])
        fc.verbose_mode = "full"
        fc.progress_events = True
        fc.FindNetwork()

        events = [ln.strip() for ln in capsys.readouterr().out.splitlines()
                  if "[DROCAT][progress]" in ln]
        assert events == ["[DROCAT][progress] 2/5 Fetching mutual direct connections"]

    def test_connection_table_byte_reproducible(self, monkeypatch, tmp_path):
        """Determinism pin: two runs of the same query export a
        byte-identical connection table (the shared total-key stable
        sort, _sort_connection_export, must stay wired before the
        writes)."""
        edges = [("S", "A", 10), ("A", "S", 8), ("S", "B", 6),
                 ("A", "B", 7), ("B", "A", 5)]
        contents = []
        for run in ("a", "b"):
            fc, _, _, _, _ = _make_network_fc(
                monkeypatch, tmp_path / run, edges, ["S", "A", "B"])
            fc.FindNetwork()
            path = os.path.join(fc.network_folder, "data_details",
                                "connection_type.csv")
            contents.append(open(path, "rb").read())
        assert contents[0] == contents[1]
