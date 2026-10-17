"""Tests for the 'all_neurons' special query token in pathfinding runs.

The token lets the Complete/Shortest Paths tools fetch all adjacent neurons
at the given connection thresholds:

- 'all_neurons' loads the full (typed) neuron set on that side;
- both sides = 'all_neurons' is rejected;
- the token replaces every other chip in the same query;
- an all-neurons side forces max_interlayer = 0 (direct connections only).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import coana  # noqa: E402


def _make_connection(monkeypatch, source, target, max_interlayer, tmp_path):
    """Build a FindNeuronConnection stub and record getNeurons calls.

    Returns (connection, calls, logs).  getNeurons is faked so the test never
    touches the network or the local dataset files.
    """
    calls = []
    logs = []

    def fake_get_neurons(required, **kwargs):
        calls.append(required)
        return (
            pd.DataFrame({
                "bodyId": ["1", "2", "3"],
                "type": ["A", "B", "C"],
            }),
            pd.DataFrame(),
            "allneurons",
            None,
        )

    monkeypatch.setattr(coana.sv, "getNeurons", fake_get_neurons)

    connection = coana.FindNeuronConnection.__new__(coana.FindNeuronConnection)
    connection._vprint = (
        lambda msg="", level="full", end="\n", flush=False: logs.append(str(msg))
    )
    connection.client_type = "flywire"  # skip the neuprint client setup
    connection.dataset = "male-cns:v1.0"
    connection.sourceNeurons = source
    connection.targetNeurons = target
    connection.max_interlayer = max_interlayer
    connection.verbose_mode = "silent"
    connection.search_columns = "auto"
    connection.label_mapper = None
    connection.custom_mapping_file = None
    connection.separate_hemispheres = False
    connection.hemisphere_filter = "both"
    connection.client_flywire = None
    connection.client_hemibrain = None
    connection.custom_source_group_names = None
    connection.custom_target_group_names = None
    connection.custom_source_name = ""
    connection.custom_target_name = ""
    connection.saveas = ""
    connection.save_folder = None
    connection.output_dir = str(tmp_path)
    connection.folder_prefix = ""
    connection.run_date = "test-run"
    connection.server = "test"
    connection.kwargs_fetch = {}
    connection.min_synapse_num = 3
    connection.min_ratio = 0.0
    connection.min_traversal_probability = 0.0
    connection.aggregate_method = "product"
    connection.filter_by = "bodyId"
    connection.exclude_intra_type_connections = False
    connection.find_reciprocal = False
    connection.keyword_in_path_to_remove = ["None"]
    connection._warn_notes = []
    return connection, calls, logs


class TestAllNeuronsTokenDetection:
    def test_bare_and_list_forms_are_detected(self):
        FNC = coana.FindNeuronConnection
        assert FNC._query_uses_all_neurons("all_neurons")
        assert FNC._query_uses_all_neurons("  ALL_NEURONS ")
        assert FNC._query_uses_all_neurons(["aMe12", "all_neurons"])
        # The token wins even inside a custom-group nested list.
        assert FNC._query_uses_all_neurons([["group_member"], ["all_neurons"]])

    def test_ordinary_queries_are_not_detected(self):
        FNC = coana.FindNeuronConnection
        assert not FNC._query_uses_all_neurons(None)
        assert not FNC._query_uses_all_neurons([])
        assert not FNC._query_uses_all_neurons(["aMe12", "PPL101"])
        assert not FNC._query_uses_all_neurons(12345)
        # A dict filter is a literal type-name search, never the token.
        assert not FNC._query_uses_all_neurons({"contains": "all_neurons"})


class TestAllNeuronsQueryResolution:
    def test_source_token_loads_full_set_and_drops_other_chips(
        self, monkeypatch, tmp_path
    ):
        connection, calls, logs = _make_connection(
            monkeypatch,
            source=["all_neurons", "aMe12"],
            target=["PPL101"],
            max_interlayer=3,
            tmp_path=tmp_path,
        )
        connection.InitializeNeuronInfo()

        # The token replaces every other chip -> empty list (all typed neurons).
        assert calls[0] == []
        assert calls[1] == ["PPL101"]
        assert len(connection.source_df) == 3
        assert any("all_neurons" in line for line in logs)

    def test_target_token_resolves_the_same_way(self, monkeypatch, tmp_path):
        connection, calls, _ = _make_connection(
            monkeypatch,
            source=["aMe12"],
            target=["all_neurons"],
            max_interlayer=2,
            tmp_path=tmp_path,
        )
        connection.InitializeNeuronInfo()

        assert calls[0] == ["aMe12"]
        assert calls[1] == []
        assert len(connection.target_df) == 3

    def test_max_interlayer_forced_to_zero(self, monkeypatch, tmp_path):
        connection, _, logs = _make_connection(
            monkeypatch,
            source=["all_neurons"],
            target=["PPL101"],
            max_interlayer=5,
            tmp_path=tmp_path,
        )
        connection.InitializeNeuronInfo()

        assert connection.max_interlayer == 0
        assert any("max_interlayer=0" in line for line in logs)

    def test_max_interlayer_untouched_without_token(self, monkeypatch, tmp_path):
        connection, calls, _ = _make_connection(
            monkeypatch,
            source=["aMe12"],
            target=["PPL101"],
            max_interlayer=4,
            tmp_path=tmp_path,
        )
        connection.InitializeNeuronInfo()

        assert connection.max_interlayer == 4
        assert calls[0] == ["aMe12"]

    def test_both_sides_all_neurons_raises(self, monkeypatch, tmp_path):
        connection, calls, _ = _make_connection(
            monkeypatch,
            source=["all_neurons"],
            target=["all_neurons"],
            max_interlayer=2,
            tmp_path=tmp_path,
        )
        # Q4 (plan 2026-10-16): a query set that equals its own other side is
        # the SHAPE of the one-box tools (Network passes its single query as
        # both), so the sentence must teach that instead of naming a pairing
        # the reader never chose.
        with pytest.raises(ValueError, match="single-query tool"):
            connection.InitializeNeuronInfo()
        assert calls == []

    def test_two_distinct_sides_keep_the_pairing_message(
            self, monkeypatch, tmp_path):
        """Two boxes, both carrying the token: the reader DID pick both sides,
        so the generic sentence is the honest one."""
        connection, calls, _ = _make_connection(
            monkeypatch,
            source=["all_neurons"],
            target=["aMe12", "all_neurons"],
            max_interlayer=2,
            tmp_path=tmp_path,
        )
        with pytest.raises(ValueError, match="both sides cannot be the full"):
            connection.InitializeNeuronInfo()
        assert calls == []

    def test_case_insensitive_token(self, monkeypatch, tmp_path):
        connection, calls, _ = _make_connection(
            monkeypatch,
            source=["ALL_NEURONS"],
            target=["PPL101"],
            max_interlayer=1,
            tmp_path=tmp_path,
        )
        connection.InitializeNeuronInfo()

        assert calls[0] == []

    def test_run_exports_keep_the_original_token(self, monkeypatch, tmp_path):
        connection, _, _ = _make_connection(
            monkeypatch,
            source=["all_neurons", "aMe12"],
            target=["PPL101"],
            max_interlayer=2,
            tmp_path=tmp_path,
        )
        connection.InitializeNeuronInfo()

        # Provenance keeps the raw user query; execution uses the full set.
        assert connection._requested_query_for_export("source") == [
            "all_neurons",
            "aMe12",
        ]
        assert connection.parameter_dict["resolved source neurons"] == "[]"
        assert connection.parameter_dict["resolved source bodyIds"] == "['1', '2', '3']"
