"""Backend tests for pre/post-site synapse mode and the custom-layer CSV.

Covers:
- ``synapse_mode='pre_post'`` is accepted by the constructor.
- ``_parse_layer_map_csv`` auto-detects the numeric layer base (0 or 1), accepts
  the ``neuron`` column (and the legacy ``id_type_instance`` alias), and builds
  per-neuron color overrides with the alpha override/inherit rule.
- ``_resolve_synapse_color`` / ``_resolve_pre_color`` / ``_resolve_post_color``
  fall back to the layer color and honor per-neuron overrides.
- pre/post site markers use one uniform size based on the mean real connector
  distance, including the one-layer upstream/downstream fallback.
- ``_pre_post_mode_warning_html`` emission.
- ``statvis.build_site_mesh`` returns a Mesh3d trace.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import visualize_skeleton as vs_mod  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402


class FakeClient:
    """Minimal neuprint client stub: construction must not hit the network."""

    def __init__(self, *args, **kwargs):
        self.dataset = kwargs.get("dataset", "hemibrain:v1.2.1")


def build_vs(tmp_path, **overrides):
    defaults = dict(
        dataset="hemibrain:v1.2.1",
        neuron_layers=["aMe12"],
        client=FakeClient(dataset="hemibrain:v1.2.1"),
        verbose=False,
        output_dir=str(tmp_path),
        include_timestamp=False,
        script_path=str(tmp_path),
        data_folder=str(tmp_path),
    )
    defaults.update(overrides)
    vs = VisualizeSkeleton(**defaults)
    vs.client_type = "neuprint"
    return vs


def _noop_getneurons(*args, **kwargs):
    return (pd.DataFrame(), pd.DataFrame(), "x", None)


def _resolved_getneurons(layer_input, *args, **kwargs):
    """Resolve the two names used by the reusable layer-info tests."""
    values = layer_input if isinstance(layer_input, list) else [layer_input]
    values = {str(value) for value in values}
    if "aMe12" in values or "101" in values:
        return (
            pd.DataFrame({
                "bodyId": [101],
                "type": ["aMe12"],
                "instance": ["aMe12_L"],
            }),
            pd.DataFrame(),
            "aMe12",
            None,
        )
    return (
        pd.DataFrame({
            "bodyId": [202],
            "type": ["dn1"],
            "instance": ["dn1_R"],
        }),
        pd.DataFrame(),
        "dn1",
        None,
    )


# ---------------------------------------------------------------------------
# pre_post mode acceptance
# ---------------------------------------------------------------------------

def test_pre_post_mode_accepted(tmp_path):
    vs = build_vs(tmp_path, synapse_mode="pre_post")
    assert vs.synapse_mode == "pre_post"


def test_default_synapse_size_is_one_px(tmp_path):
    vs = build_vs(tmp_path)
    assert vs.synapse_size == 1
    assert vs._synapse_size_px() == 1.0
    assert vs._synapse_size_fold() == pytest.approx(1.0)


def test_pre_post_scatter_field_accepted(tmp_path):
    vs = build_vs(tmp_path, synapse_mode="pre_post", pre_post_scatter=True)
    assert vs.pre_post_scatter is True


def _paired_frame(distances):
    """Build a tiny paired connector frame with known Euclidean distances."""
    rows = []
    for index, distance in enumerate(distances):
        rows.append({
            "bodyId_pre": str(100 + index),
            "bodyId_post": str(200 + index),
            "x_pre": 0.0,
            "y_pre": 0.0,
            "z_pre": 0.0,
            "x_post": float(distance),
            "y_post": 0.0,
            "z_post": 0.0,
        })
    return pd.DataFrame(rows)


def test_pre_post_site_scale_uses_mean_adjacent_distance_and_fold(
    tmp_path, monkeypatch
):
    vs = build_vs(
        tmp_path, synapse_mode="pre_post", synapse_size="2x real",
        brain_mesh="none",
    )
    vs.neuron_layers = ["source", "target"]
    vs.layer_criteria = ["source-criteria", "target-criteria"]
    vs.neuron_dfs = [
        pd.DataFrame({"bodyId": ["100", "101"]}),
        pd.DataFrame({"bodyId": ["200", "201"]}),
    ]
    frame = _paired_frame([10.0, 20.0])
    calls = []

    def fake_fetch(**kwargs):
        calls.append(
            (
                kwargs["source_criteria"],
                kwargs["target_criteria"],
                kwargs["min_total_weight"],
            )
        )
        return frame

    monkeypatch.setattr(vs_mod, "fetch_synapse_connections", fake_fetch)
    monkeypatch.setattr(vs, "_transform_site_df", lambda value: value)

    # Mean distance = 15; a site is centered at its own endpoint, so the
    # full distance is used as the mesh size: 15 * 2x = 30.
    assert vs._pre_post_site_scale() == pytest.approx(30.0)
    assert calls == [("source-criteria", "target-criteria", 0)]
    assert vs._pre_post_real_synapse_distance_sample_count == 2


def test_one_layer_pre_post_size_uses_upstream_and_downstream_pairs(
    tmp_path, monkeypatch
):
    vs = build_vs(
        tmp_path, synapse_mode="pre_post", synapse_size="3x real",
        brain_mesh="none",
    )
    vs.layer_criteria = ["queried-layer"]
    vs.neuron_dfs = [pd.DataFrame({"bodyId": ["100"]})]
    calls = []
    upstream = _paired_frame([30.0])
    upstream["bodyId_pre"] = "300"
    upstream["bodyId_post"] = "100"
    frames = iter([_paired_frame([10.0]), upstream])

    def fake_fetch(**kwargs):
        calls.append(
            (
                kwargs["source_criteria"],
                kwargs["target_criteria"],
                kwargs["min_total_weight"],
            )
        )
        return next(frames)

    monkeypatch.setattr(vs_mod, "fetch_synapse_connections", fake_fetch)
    monkeypatch.setattr(vs, "_transform_site_df", lambda value: value)

    # Mean upstream/downstream distance = 20; full distance * 3x = size 60.
    assert vs._pre_post_site_scale() == pytest.approx(60.0)
    assert calls == [
        ("queried-layer", None, 0),
        (None, "queried-layer", 0),
    ]


def test_pre_post_size_baseline_is_independent_of_display_threshold(
    tmp_path, monkeypatch
):
    """A display cutoff must not shrink the universal pseudo-real site size."""
    calls = []

    def fake_fetch(**kwargs):
        calls.append(kwargs["min_total_weight"])
        return _paired_frame([10.0, 20.0])

    monkeypatch.setattr(vs_mod, "fetch_synapse_connections", fake_fetch)
    scales = []
    for threshold in (3, 30):
        vs = build_vs(
            tmp_path / f"threshold-{threshold}",
            synapse_mode="pre_post",
            synapse_size="3x real",
            min_synapse_num=threshold,
            brain_mesh="none",
        )
        vs.neuron_layers = ["source", "target"]
        vs.layer_criteria = ["source-criteria", "target-criteria"]
        vs.neuron_dfs = [
            pd.DataFrame({"bodyId": ["100", "101"]}),
            pd.DataFrame({"bodyId": ["200", "201"]}),
        ]
        monkeypatch.setattr(vs, "_transform_site_df", lambda value: value)
        scales.append(vs._pre_post_site_scale())

    # Mean distance 15 * 3x = 45 for both display thresholds.
    assert scales == pytest.approx([45.0, 45.0])
    assert calls == [0, 0]


def test_pre_post_plot_passes_one_uniform_size_to_every_site_group(tmp_path):
    vs = build_vs(
        tmp_path, synapse_mode="pre_post", synapse_size="2x real",
        brain_mesh="none",
    )
    vs.neuron_layers = ["one-layer"]
    vs.layer_names = ["one-layer"]
    vs.neuron_dfs = [pd.DataFrame({"bodyId": ["1"]})]
    vs._pre_post_real_synapse_distance = 20.0
    vs._collect_layer_sites = lambda _index: (
        pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0], "neuron_id": ["1"]}),
        pd.DataFrame({"x": [1.0], "y": [0.0], "z": [0.0], "neuron_id": ["1"]}),
    )
    captured_scales = []
    vs._plot_site_group = lambda _df, _site_type, _layer_idx, _name, site_scale: (
        captured_scales.append(site_scale)
    )

    assert vs._plot_pre_post_sites() == 0
    assert captured_scales == [40.0, 40.0]


def test_pre_post_scatter_renders_scatter_markers(tmp_path):
    """With pre_post_scatter enabled, sites render as Scatter3d markers (diamond
    for pre / circle for post) rather than solid meshes."""
    import plotly.graph_objects as go
    vs = build_vs(
        tmp_path, synapse_mode="pre_post", pre_post_scatter=True, backend="plotly",
        brain_mesh="none",  # keep the test deterministic: no template transform
    )
    # build_vs seeds a single layer; the pre/post site renderer loops over
    # ``len(self.neuron_layers)``, so give it two layers (pre from layer 0,
    # post from layer 1).
    vs.neuron_layers = ["aMe12", "dn1"]
    vs.neuron_dfs = [pd.DataFrame({"bodyId": ["1"]}), pd.DataFrame({"bodyId": ["2"]})]
    vs.layer_names = ["aMe12", "dn1"]
    vs.layer_criteria = [None, None]
    vs.fig_3d = go.Figure()
    sites = pd.DataFrame({"x": [1.0, 2.0], "y": [0.0, 0.0], "z": [0.0, 0.0], "neuron_id": ["1", "2"]})

    def fake_collect(idx):
        return (sites.copy(), None) if idx == 0 else (None, sites.copy())

    vs._collect_layer_sites = fake_collect
    rc = vs._plot_pre_post_sites()
    assert rc == 0
    assert vs.fig_3d.data
    assert all(trace.type == "scatter3d" for trace in vs.fig_3d.data)
    symbols = {trace.marker.symbol for trace in vs.fig_3d.data}
    assert symbols == {"diamond", "circle"}
    assert len(vs.fig_3d.layout.sliders) == 1
    assert vs.fig_3d.layout.sliders[0].currentvalue.prefix == "Synapse size: "


@pytest.mark.parametrize(
    ("legend_mode", "expected_names"),
    [
        ("layer", {"L1_pre", "L2_post"}),
        ("type", {"aMe12_pre", "dn1_post"}),
        ("single", {"aMe12_L1_pre", "dn1_L2_post"}),
    ],
)
def test_pre_post_sites_follow_legend_level(
    tmp_path, legend_mode, expected_names
):
    import plotly.graph_objects as go

    vs = build_vs(
        tmp_path,
        synapse_mode="pre_post",
        legend_mode=legend_mode,
        brain_mesh="none",
    )
    vs.neuron_layers = ["aMe12", "dn1"]
    vs.neuron_dfs = [
        pd.DataFrame({"bodyId": ["1"], "type": ["aMe12"]}),
        pd.DataFrame({"bodyId": ["2"], "type": ["dn1"]}),
    ]
    vs.layer_names = ["L1", "L2"]
    vs.layer_criteria = [None, None]
    vs.fig_3d = go.Figure()
    sites = pd.DataFrame({
        "x": [1.0], "y": [0.0], "z": [0.0], "neuron_id": ["1"],
    })
    vs._collect_layer_sites = lambda idx: (
        (sites.copy(), None) if idx == 0 else (None, sites.assign(neuron_id="2"))
    )
    vs._pre_post_real_synapse_distance = 10.0

    assert vs._plot_pre_post_sites() == 0
    shown = {
        trace.name for trace in vs.fig_3d.data
        if trace.showlegend
    }
    assert shown == expected_names
    assert all("(" not in str(name) for name in shown)


def test_pre_post_type_legend_merges_same_type_without_body_suffix(tmp_path):
    import plotly.graph_objects as go

    vs = build_vs(
        tmp_path,
        synapse_mode="pre_post",
        legend_mode="type",
        brain_mesh="none",
    )
    vs.neuron_layers = ["s-LNv"]
    vs.neuron_dfs = [pd.DataFrame({
        "bodyId": [15832, 16461],
        "type": ["s-LNv", "s-LNv"],
    })]
    vs.layer_names = ["L1"]
    vs.layer_criteria = [None]
    vs.fig_3d = go.Figure()
    sites = pd.DataFrame({
        "x": [1.0, 2.0],
        "y": [0.0, 0.0],
        "z": [0.0, 0.0],
        "neuron_id": ["15832.0", "16461.0"],
    })
    vs._collect_layer_sites = lambda _idx: (sites.copy(), None)
    vs._pre_post_real_synapse_distance = 10.0

    assert vs._plot_pre_post_sites() == 0
    shown = {
        trace.name for trace in vs.fig_3d.data
        if trace.showlegend
    }
    assert shown == {"s-LNv_pre"}
    assert all("15832" not in str(trace.name) for trace in vs.fig_3d.data)


# ---------------------------------------------------------------------------
# The tree panel nests a site row one level below its neuron by matching the
# site's ``drocatLegend.ownerItem`` against the leaf's own ``item`` string.
# That key has to be the exact label the panel already uses: the bodyId is not
# a usable substitute, because connector tables serve ``'15832.0'`` where the
# leaf reads ``'15832_s-LNv_L'``.
# ---------------------------------------------------------------------------

_UNSET = object()


def _two_neuron_site_vs(tmp_path, pre_color, post_color, *, custom=True,
                        leaves=_UNSET, extra_site_ids=()):
    """A tree-mode figure with two same-type neurons whose sites are plotted.

    ``pre_color``/``post_color`` may be one color (both neurons share it, so
    their sites merge into ONE trace) or a per-neuron pair.  The two alias maps
    are written the way ``plot_skeleton`` writes them: the group map takes
    every identifier the resolved row carries (bodyId, type, instance, ...),
    while the leaf map takes bodyIds only -- the type alias is shared by both
    neurons and the last one plotted would otherwise own it.
    ``leaves={}`` stands in for sites rendered without a plotted skeleton.
    """
    import plotly.graph_objects as go

    vs = build_vs(
        tmp_path,
        synapse_mode="pre_post",
        legend_mode="tree",
        pre_post_scatter=True,
        brain_mesh="none",
    )
    vs.custom_layer_names = ["G1"] if custom else None
    vs.neuron_layers = ["s-LNv"]
    vs.layer_names = ["s-LNv"]
    vs.layer_criteria = [None]
    vs.neuron_dfs = [pd.DataFrame({
        "bodyId": [15832, 16461],
        "type": ["s-LNv", "s-LNv"],
    })]
    vs.fig_3d = go.Figure()
    body_ids = [15832, 16461]
    leaf_labels = {
        "15832": "15832_s-LNv_L", "16461": "16461_s-LNv_R"}
    if leaves is not _UNSET:
        leaf_labels = leaves
    label_map = {alias: "s-LNv" for bid in body_ids
                 for alias in vs._normalize_neuron_lookup_keys(bid)}
    # The shared type alias resolves to the group for both neurons.
    for alias in vs._normalize_neuron_lookup_keys("s-LNv"):
        label_map[alias] = "s-LNv"
    leaf_map = {alias: leaf_labels.get(str(bid)) for bid in body_ids
                for alias in vs._normalize_neuron_lookup_keys(bid)}
    vs._neuron_legend_labels_by_layer = {0: label_map}
    vs._neuron_tree_leaf_labels_by_layer = {0: leaf_map}
    if isinstance(pre_color, dict):
        vs._neuron_pre_color_overrides = dict(pre_color)
        vs._neuron_post_color_overrides = dict(post_color)
    else:
        vs._neuron_pre_color_overrides = {b: pre_color for b in body_ids}
        vs._neuron_post_color_overrides = {b: post_color for b in body_ids}
    ids = ["15832.0", "15832.0", "16461.0", "16461.0"] + [
        str(value) for value in extra_site_ids]
    sites = pd.DataFrame({
        "x": [float(n) for n in range(len(ids))],
        "y": [0.0] * len(ids),
        "z": [0.0] * len(ids),
        # Float-shaped ids, exactly as a NeuPrint connector frame arrives.
        "neuron_id": ids,
    })
    vs._collect_layer_sites = lambda _idx: (sites.copy(), sites.copy())
    vs._pre_post_real_synapse_distance = 10.0
    assert vs._plot_pre_post_sites() == 0
    return vs


def _site_legends(vs):
    """``[(ownerItem, item), ...]`` for every site trace the figure carries."""
    out = []
    for trace in vs.fig_3d.data:
        lg = (getattr(trace, "meta", None) or {}).get("drocatLegend") or {}
        if lg.get("kind") == "site":
            out.append((lg.get("ownerItem"), lg.get("item"), lg.get("type"),
                        lg.get("group")))
    return out


class TestTreeSiteOwnerLeaf:
    def test_each_site_trace_names_its_owner_leaf(self, tmp_path):
        vs = _two_neuron_site_vs(
            tmp_path, {"15832": "#ff0000", "16461": "#0000ff"},
            {"15832": "#00ff00", "16461": "#ffff00"})
        rows = _site_legends(vs)
        assert len(rows) == 4, "two neurons x pre/post, one trace each"
        assert sorted(owner for owner, *_ in rows) == [
            "15832_s-LNv_L", "15832_s-LNv_L", "16461_s-LNv_R",
            "16461_s-LNv_R"]
        # The row label stays the group-level one it has always been; the new
        # key is the leaf, so the two are independent strings by design.
        assert sorted((owner, item) for owner, item, *_ in rows) == [
            ("15832_s-LNv_L", "s-LNv_post"), ("15832_s-LNv_L", "s-LNv_pre"),
            ("16461_s-LNv_R", "s-LNv_post"), ("16461_s-LNv_R", "s-LNv_pre")]

    def test_merged_sites_of_two_neurons_stay_unnested(self, tmp_path):
        """Same-color neurons share one site trace, so no single leaf owns it.

        This is the case with no field sample (one page in the whole real-data
        cache carries site traces), so it is pinned here rather than by a
        census: dropping the key is what keeps the row at the type level
        instead of losing it.
        """
        vs = _two_neuron_site_vs(tmp_path, "#ff0000", "#00ff00")
        rows = _site_legends(vs)
        assert len(rows) == 2, "pre and post each merged into one trace"
        assert {owner for owner, *_ in rows} == {None}
        # ...and the bodyId stamp agrees: no single owner either.
        for trace in vs.fig_3d.data:
            stamped = (getattr(trace, "meta", None) or {}).get("drocatTrace")
            if stamped and stamped.get("kind") == "site":
                assert stamped.get("owner_body_id") is None

    def test_flat_tree_page_also_gets_the_leaf(self, tmp_path):
        vs = _two_neuron_site_vs(
            tmp_path, {"15832": "#ff0000", "16461": "#0000ff"},
            {"15832": "#00ff00", "16461": "#ffff00"}, custom=False)
        rows = _site_legends(vs)
        assert sorted(owner for owner, *_ in rows) == [
            "15832_s-LNv_L", "15832_s-LNv_L", "16461_s-LNv_R",
            "16461_s-LNv_R"]
        # A flat page files the site under the type-level group and no type.
        assert {(r[2], r[3]) for r in rows} == {(None, "s-LNv")}

    def test_site_without_a_plotted_skeleton_carries_no_key(self, tmp_path):
        """Standalone site rendering keeps today's rows rather than guessing.

        ``ownerItem`` must be absent, not ``None``: the panel reads the key's
        presence to decide whether a row can nest.
        """
        vs = _two_neuron_site_vs(
            tmp_path, {"15832": "#ff0000", "16461": "#0000ff"},
            {"15832": "#00ff00", "16461": "#ffff00"}, leaves={})
        rows = _site_legends(vs)
        assert len(rows) == 4
        assert {owner for owner, *_ in rows} == {None}
        for trace in vs.fig_3d.data:
            lg = (getattr(trace, "meta", None) or {}).get("drocatLegend") or {}
            if lg.get("kind") == "site":
                assert "ownerItem" not in lg
                assert lg.get("item") and lg.get("group")

    def test_an_unplotted_neuron_of_the_same_type_steals_no_leaf(self, tmp_path):
        """The leaf map is bodyId-keyed precisely because the type alias is shared.

        ``plot_skeleton`` registers ``s-LNv`` for every neuron of the layer and
        the last write wins, so a leaf stored under that alias would point at one
        arbitrary neuron.  A site for a neuron whose skeleton was never plotted
        therefore has to fall back to the type-level row rather than nest under
        a neighbour.
        """
        vs = _two_neuron_site_vs(
            tmp_path,
            {"15832": "#ff0000", "16461": "#0000ff", "77777": "#00ffff"},
            {"15832": "#00ff00", "16461": "#ffff00", "77777": "#ff00ff"},
            extra_site_ids=["77777.0"])
        leaf_map = vs._neuron_tree_leaf_labels_by_layer[0]
        assert vs._normalize_neuron_lookup_keys("s-LNv")[0] not in leaf_map
        rows = _site_legends(vs)
        assert {owner for owner, *_ in rows} == {
            "15832_s-LNv_L", "16461_s-LNv_R", None}
        unowned = [trace for trace in vs.fig_3d.data
                   if ((getattr(trace, "meta", None) or {}).get(
                       "drocatLegend") or {}).get("ownerItem") is None
                   and ((getattr(trace, "meta", None) or {}).get(
                       "drocatLegend") or {}).get("kind") == "site"]
        assert len(unowned) == 2, "the third neuron's pre and post rows"
        # Its bodyId stamp survives, so the profile plan can still place it --
        # only the panel lacks a leaf to hang it on.
        assert {str((trace.meta.get("drocatTrace") or {}).get("owner_body_id"))
                for trace in unowned} == {"77777.0"}

    def test_owner_leaf_resolves_through_the_connector_float_form(self, tmp_path):
        """The alias map is keyed by int; the site frame arrives as ``'15832.0'``."""
        vs = _two_neuron_site_vs(
            tmp_path, {"15832": "#ff0000", "16461": "#0000ff"},
            {"15832": "#00ff00", "16461": "#ffff00"})
        assert vs._pre_post_site_owner_leaf("15832.0", 0) == "15832_s-LNv_L"
        assert vs._pre_post_site_owner_leaf(16461, 0) == "16461_s-LNv_R"
        assert vs._pre_post_site_owner_leaf("99999", 0) is None
        # The group-level resolver keeps answering with the group, unchanged.
        assert vs._pre_post_site_owner_identity("15832.0", 0, "s-LNv") == "s-LNv"


def test_pre_post_fallback_baseline_is_scaled_by_requested_fold(tmp_path):
    vs = build_vs(
        tmp_path,
        synapse_mode="pre_post",
        synapse_size="3x real",
        brain_mesh="none",
    )
    vs.neuron_layers = []
    vs._pre_post_real_synapse_distance = None

    assert vs._pre_post_site_scale() == pytest.approx(120.0)
    assert vs._pre_post_real_synapse_distance == pytest.approx(40.0)


def test_plotly_scatter_size_slider_targets_only_synapse_traces(tmp_path):
    import plotly.graph_objects as go

    vs = build_vs(tmp_path, synapse_mode="scatter", synapse_size="3x real")
    vs.fig_3d = go.Figure([
        go.Scatter3d(
            x=[0], y=[0], z=[0], mode="lines", name="skeleton",
        ),
        go.Scatter3d(
            x=[0], y=[0], z=[0], mode="markers", marker={"size": 9},
            meta={
                "drocat_scatter_size_role": "synapse",
                "drocat_scatter_size_factor": 1.0,
            },
        ),
    ])

    vs._add_plotly_synapse_size_slider()

    slider = vs.fig_3d.layout.sliders[0]
    assert len(slider.steps) == 12
    assert slider.steps[slider.active].label == "3 px"
    assert slider.steps[slider.active].args[1] == (1,)
    assert slider.steps[slider.active].args[0]["marker.size"] == [3.0]


def test_pre_post_warning_banner(tmp_path):
    vs = build_vs(tmp_path, synapse_mode="pre_post")
    assert "Pre/post-synaptic site mode." in vs._pre_post_mode_warning_html()
    other = build_vs(tmp_path, synapse_mode="cone")
    assert other._pre_post_mode_warning_html() == ""


# ---------------------------------------------------------------------------
# custom-layer CSV parsing
# ---------------------------------------------------------------------------

class TestCustomLayerCsv:
    def test_zero_based_layers_and_neuron_column(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vs_mod.sv, "getNeurons", _noop_getneurons)
        csv_path = tmp_path / "layers0.csv"
        csv_path.write_text(
            "layer,neuron,color\n"
            "0,aMe12,#ff0000\n"
            "0,aMe13,#00ff00\n"
            "1,dn1,#0000ff\n"
        )
        vs = VisualizeSkeleton(
            dataset="hemibrain:v1.2.1",
            neuron_layers=["x"],  # overridden by the CSV
            layer_map_csv=str(csv_path),
            client=FakeClient(dataset="hemibrain:v1.2.1"),
            verbose=False,
            output_dir=str(tmp_path),
            include_timestamp=False,
            script_path=str(tmp_path),
        )
        assert vs.neuron_layers == [["aMe12", "aMe13"], "dn1"]
        assert vs.custom_layer_names == ["0", "1"]

    def test_one_based_layers_shift_to_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vs_mod.sv, "getNeurons", _noop_getneurons)
        csv_path = tmp_path / "layers1.csv"
        csv_path.write_text(
            "layer,neuron\n"
            "1,aMe12\n"
            "2,dn1\n"
        )
        vs = VisualizeSkeleton(
            dataset="hemibrain:v1.2.1",
            neuron_layers=["x"],
            layer_map_csv=str(csv_path),
            client=FakeClient(dataset="hemibrain:v1.2.1"),
            verbose=False,
            output_dir=str(tmp_path),
            include_timestamp=False,
            script_path=str(tmp_path),
        )
        # 1-based source is shifted to 0-based internally; label keeps the raw value.
        assert vs.neuron_layers == ["aMe12", "dn1"]
        assert vs.custom_layer_names == ["1", "2"]
        # Numeric layer indices group rows but do not become display/folder names.
        assert vs.layer_names != ["1", "2"]

    def test_alpha_override_and_inherit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vs_mod.sv, "getNeurons", _noop_getneurons)
        csv_path = tmp_path / "alpha.csv"
        csv_path.write_text(
            "layer,neuron,color\n"
            "0,rgbn,rgb(255,0,0)\n"
            "0,rgban,rgba(0,255,0,0.5)\n"
        )
        vs = VisualizeSkeleton(
            dataset="hemibrain:v1.2.1",
            neuron_layers=["x"],
            layer_map_csv=str(csv_path),
            client=FakeClient(dataset="hemibrain:v1.2.1"),
            verbose=False,
            output_dir=str(tmp_path),
            include_timestamp=False,
            script_path=str(tmp_path),
        )
        # 'rgb(...)' has no alpha -> inherits the global neuron_alpha (0.2).
        rgb = vs._resolve_neuron_color("rgbn", 0)
        assert "0.2" in str(rgb)
        # 'rgba(...)' carries its own alpha -> overrides the global fallback.
        rgba = vs._resolve_neuron_color("rgban", 0)
        assert "0.5" in str(rgba)

    def test_group_name_labels_are_used_verbatim(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vs_mod.sv, "getNeurons", _noop_getneurons)
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text(
            "layer,id_type_instance\n"
            "DNp_group,DNp01\n"
            "DNp_group,DNp02\n"
            "aMe_group,aMe12\n"
        )
        vs = VisualizeSkeleton(
            dataset="hemibrain:v1.2.1",
            neuron_layers=["x"],
            layer_map_csv=str(csv_path),
            client=FakeClient(dataset="hemibrain:v1.2.1"),
            verbose=False,
            output_dir=str(tmp_path),
            include_timestamp=False,
            script_path=str(tmp_path),
        )
        assert vs.neuron_layers == [["DNp01", "DNp02"], "aMe12"]
        assert vs.custom_layer_names == ["DNp_group", "aMe_group"]


# ---------------------------------------------------------------------------
# per-neuron synapse / pre / post color resolution
# ---------------------------------------------------------------------------

class TestSiteColorResolution:
    def test_resolvers_fall_back_to_layer_color(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vs_mod.sv, "getNeurons", _noop_getneurons)
        vs = build_vs(tmp_path)
        layer_color = vs.neuron_colors[0]
        assert vs._resolve_pre_color("anything", 0) == layer_color
        assert vs._resolve_post_color("anything", 0) == layer_color
        # The synapse color resolver returns a valid rgba string even when the
        # single-layer construction has no gap (synapse_colors is empty).
        resolved = vs._resolve_synapse_color("anything", 0)
        assert isinstance(resolved, str) and resolved.startswith("rgba")

    def test_csv_overrides_pre_and_post_colors(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vs_mod.sv, "getNeurons", _noop_getneurons)
        csv_path = tmp_path / "sites.csv"
        csv_path.write_text(
            "layer,neuron,pre_synaptic_color,post_synaptic_color,synapse_color\n"
            "0,aMe12,rgba(255,0,0,0.8),rgba(0,255,0,0.6),rgba(0,0,255,0.3)\n"
        )
        vs = VisualizeSkeleton(
            dataset="hemibrain:v1.2.1",
            neuron_layers=["x"],
            layer_map_csv=str(csv_path),
            client=FakeClient(dataset="hemibrain:v1.2.1"),
            verbose=False,
            output_dir=str(tmp_path),
            include_timestamp=False,
            script_path=str(tmp_path),
        )
        assert "0.8" in str(vs._resolve_pre_color("aMe12", 0))
        assert "0.6" in str(vs._resolve_post_color("aMe12", 0))
        assert "0.3" in str(vs._resolve_synapse_color("aMe12", 0))
        # Unspecified site type falls back to the layer color.
        assert vs._resolve_pre_color("other", 0) == vs.neuron_colors[0]

    def test_site_colors_use_numeric_aliases_and_layer_entry(self, tmp_path, monkeypatch):
        """Float-like connector IDs and duplicate layer rows keep their colors."""
        monkeypatch.setattr(vs_mod.sv, "getNeurons", _resolved_getneurons)
        csv_path = tmp_path / "per-entry-sites.csv"
        csv_path.write_text(
            "layer,neuron,pre_synaptic_color,post_synaptic_color\n"
            "1,101,#ff0000,#00ff00\n"
            "2,101,#0000ff,#ffff00\n"
        )
        vs = VisualizeSkeleton(
            dataset="hemibrain:v1.2.1",
            neuron_layers=["unused"],
            layer_map_csv=str(csv_path),
            client=FakeClient(dataset="hemibrain:v1.2.1"),
            verbose=False,
            output_dir=str(tmp_path),
            include_timestamp=False,
            script_path=str(tmp_path),
        )

        # NeuPrint connector frames may expose the same body as ``101.0``.
        assert "255, 0, 0" in str(vs._resolve_pre_color("101.0", 0))
        assert "0, 255, 0" in str(vs._resolve_post_color("101.0", 0))
        assert "0, 0, 255" in str(vs._resolve_pre_color("101.0", 1))
        assert "255, 255, 0" in str(vs._resolve_post_color("101.0", 1))

    def test_plot_site_groups_use_the_resolved_entry_colors(self, tmp_path):
        import plotly.graph_objects as go

        vs = build_vs(
            tmp_path,
            synapse_mode="pre_post",
            legend_mode="single",
            pre_post_scatter=True,
            brain_mesh="none",
        )
        vs.neuron_layers = ["one"]
        vs.layer_names = ["L1"]
        vs.neuron_dfs = [pd.DataFrame({"bodyId": [1, 2], "type": ["one", "two"]})]
        vs.layer_criteria = [None]
        vs.fig_3d = go.Figure()
        vs._neuron_pre_color_overrides = {1: "#ff0000", 2: "#0000ff"}
        vs._neuron_post_color_overrides = {1: "#00ff00", 2: "#ffff00"}
        sites = pd.DataFrame({
            "x": [1.0, 2.0], "y": [0.0, 0.0], "z": [0.0, 0.0],
            "neuron_id": [1, 2],
        })
        vs._collect_layer_sites = lambda _idx: (sites.copy(), sites.copy())
        vs._pre_post_real_synapse_distance = 10.0

        assert vs._plot_pre_post_sites() == 0
        colors = [str(trace.marker.color) for trace in vs.fig_3d.data]
        assert set(colors) == {
            "rgba(255, 0, 0, 1.0)", "rgba(0, 0, 255, 1.0)",
            "rgba(0, 255, 0, 1.0)", "rgba(255, 255, 0, 1.0)",
        }


class TestReusableLayerInfo:
    def test_exported_viz_layer_info_is_reusable_and_keeps_colors(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(vs_mod.sv, "getNeurons", _resolved_getneurons)
        csv_path = tmp_path / "custom.csv"
        csv_path.write_text(
            "layer,neuron,color,synapse_color,pre_synaptic_color,post_synaptic_color\n"
            "1,aMe12,rgba(255,0,0,0.8),rgba(0,255,0,0.7),rgba(1,2,3,0.4),rgba(4,5,6,0.3)\n"
            "2,dn1,rgba(0,0,255,0.6),rgba(255,255,0,0.5),rgba(7,8,9,0.2),rgba(10,11,12,0.1)\n"
        )
        first = VisualizeSkeleton(
            dataset="hemibrain:v1.2.1",
            neuron_layers=["unused"],
            layer_map_csv=str(csv_path),
            client=FakeClient(dataset="hemibrain:v1.2.1"),
            verbose=False,
            output_dir=str(tmp_path),
            include_timestamp=False,
            script_path=str(tmp_path),
        )

        # The CSV uses names, but the renderer resolves the fetched neuron by
        # bodyId. The alias bridge must make the custom colors visible there.
        assert "0.8" in str(first._resolve_neuron_color(101, 0))
        assert "0.7" in str(first._resolve_synapse_color(101, 0))
        assert "0.4" in str(first._resolve_pre_color(101, 0))
        assert "0.3" in str(first._resolve_post_color(101, 0))
        assert first.layer_names == ["aMe12", "dn1"]
        assert Path(first.save_folder).name.startswith(
            "plot-3d_HEMI_aMe12_dn1"
        )

        exported = Path(first.save_folder) / "viz_layer_info.csv"
        assert exported.exists()
        exported_df = pd.read_csv(exported)
        assert list(exported_df.columns) == [
            "layer", "neuron", "color", "synapse_color",
            "pre_synaptic_color", "post_synaptic_color",
        ]
        # The reusable export keeps the editor-facing identifiers rather than
        # normalizing them to the resolved body IDs.
        assert exported_df["neuron"].astype(str).tolist() == ["aMe12", "dn1"]
        assert "0.8" in exported_df.loc[0, "color"]
        assert "0.7" in exported_df.loc[0, "synapse_color"]
        assert exported_df["pre_synaptic_color"].isna().all()
        assert exported_df["post_synaptic_color"].isna().all()

        # Feeding the generated file back through the normal layer-map parser
        # must preserve the same effective colors (the CSV-upload path).
        second = VisualizeSkeleton(
            dataset="hemibrain:v1.2.1",
            neuron_layers=["unused"],
            layer_map_csv=str(exported),
            client=FakeClient(dataset="hemibrain:v1.2.1"),
            verbose=False,
            output_dir=str(tmp_path),
            include_timestamp=False,
            script_path=str(tmp_path),
        )
        assert "0.8" in str(second._resolve_neuron_color(101, 0))
        assert "0.7" in str(second._resolve_synapse_color(101, 0))

    @pytest.mark.parametrize(
        "synapse_mode,skip_synapse,active_columns",
        [
            ("scatter", False, {"synapse_color"}),
            ("pre_post", False, {"pre_synaptic_color", "post_synaptic_color"}),
            ("scatter", True, set()),
        ],
    )
    def test_exported_layer_info_has_one_synapse_color_family(
        self, tmp_path, monkeypatch, synapse_mode, skip_synapse, active_columns
    ):
        monkeypatch.setattr(vs_mod.sv, "getNeurons", _resolved_getneurons)
        csv_path = tmp_path / f"input-{synapse_mode}-{skip_synapse}.csv"
        csv_path.write_text(
            "layer,neuron,color,synapse_color,pre_synaptic_color,post_synaptic_color\n"
            "alpha,aMe12,#ff0000,#00ff00,#010203,#040506\n"
            "beta,dn1,#0000ff,#ffff00,#070809,#0a0b0c\n"
        )
        vs = VisualizeSkeleton(
            dataset="hemibrain:v1.2.1",
            neuron_layers=["unused"],
            layer_map_csv=str(csv_path),
            synapse_mode=synapse_mode,
            skip_synapse=skip_synapse,
            client=FakeClient(dataset="hemibrain:v1.2.1"),
            verbose=False,
            output_dir=str(tmp_path / f"out-{synapse_mode}-{skip_synapse}"),
            include_timestamp=False,
            script_path=str(tmp_path),
        )
        exported = pd.read_csv(Path(vs.save_folder) / "viz_layer_info.csv").fillna("")
        assert exported["layer"].tolist() == ["alpha", "beta"]
        assert exported["neuron"].tolist() == ["aMe12", "dn1"]
        for column in ("synapse_color", "pre_synaptic_color", "post_synaptic_color"):
            if column in active_columns:
                assert exported[column].astype(str).str.len().gt(0).all()
            else:
                assert (exported[column].astype(str) == "").all()

    def test_empty_run_still_writes_viz_layer_info_header(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(vs_mod.sv, "getNeurons", _noop_getneurons)
        # A resolved layer with no rows is a valid empty-fetch case and still
        # exercises the unconditional export without violating constructor
        # validation's mesh-only guard for an empty layer list.
        vs = build_vs(tmp_path)
        exported = Path(vs.save_folder) / "viz_layer_info.csv"
        assert exported.exists()
        assert list(pd.read_csv(exported).columns) == [
            "layer", "neuron", "color", "synapse_color",
            "pre_synaptic_color", "post_synaptic_color",
        ]


def test_scatter_size_uses_pixel_value_directly(tmp_path):
    # The Synapse Size control is now a direct pixel size (1-12, default 3).
    vs = build_vs(tmp_path, synapse_mode="scatter", synapse_size="3")
    assert vs.synapse_size == 3.0
    assert vs._synapse_size_px() == 3.0
    assert vs._scatter_synapse_marker_size() == 3.0
    # Mesh modes keep the raw number as a size multiplier (3 = 3x).
    assert vs._synapse_size_fold() == 3.0


def test_individual_profiles_keep_each_owners_pre_post_site_traces(
    tmp_path,
):
    import plotly.graph_objects as go

    vs = build_vs(
        tmp_path,
        synapse_mode="pre_post",
        legend_mode="single",
        brain_mesh="none",
    )
    vs.fig_3d = go.Figure(data=[
        go.Scatter3d(
            x=[0], y=[0], z=[0], name="Neuron_label_1",
            legendgroup="Neuron_label_1", showlegend=True,
        ),
        go.Scatter3d(
            x=[1], y=[1], z=[1], name="Neuron_label_2",
            legendgroup="Neuron_label_2", showlegend=True,
        ),
        go.Scatter3d(
            x=[0.1], y=[0], z=[0], name="Neuron_label_1_pre",
            legendgroup="pre_post:pre:Neuron_label_1", showlegend=True,
        ),
        go.Scatter3d(
            x=[0.2], y=[0], z=[0], name="Neuron_label_1_post",
            legendgroup="pre_post:post:Neuron_label_1", showlegend=True,
        ),
        go.Scatter3d(
            x=[1.1], y=[1], z=[1], name="Neuron_label_2_pre",
            legendgroup="pre_post:pre:Neuron_label_2", showlegend=True,
        ),
        go.Scatter3d(
            x=[1.2], y=[1], z=[1], name="Neuron_label_2_post",
            legendgroup="pre_post:post:Neuron_label_2", showlegend=True,
        ),
    ])

    written = {}

    def capture_html(fig, path, **_kwargs):
        written[Path(path).name] = [
            (trace.name, trace.visible) for trace in fig.data
        ]

    vs._write_plotly_html = capture_html
    output = vs.plot_individuals(
        output_format="html",
        views="front",
        summary_format=None,
        export_method="kaleido",
    )

    assert output is not None
    assert set(written) == {"Neuron_label_1.html", "Neuron_label_2.html"}
    for filename, owner in (
        ("Neuron_label_1.html", "Neuron_label_1"),
        ("Neuron_label_2.html", "Neuron_label_2"),
    ):
        visibility = dict(written[filename])
        assert visibility[owner] is True
        assert visibility[f"{owner}_pre"] is True
        assert visibility[f"{owner}_post"] is True
        other = "Neuron_label_2" if owner == "Neuron_label_1" else "Neuron_label_1"
        assert visibility[other] is False
        assert visibility[f"{other}_pre"] is False
        assert visibility[f"{other}_post"] is False


def test_pre_post_single_legend_order_interleaves_each_owner(tmp_path):
    import plotly.graph_objects as go

    vs = build_vs(
        tmp_path,
        synapse_mode="pre_post",
        legend_mode="single",
        brain_mesh="none",
    )
    vs.neuron_layers = ["one", "two"]
    vs.neuron_dfs = [
        pd.DataFrame({"bodyId": [1], "type": ["one"]}),
        pd.DataFrame({"bodyId": [2], "type": ["two"]}),
    ]
    vs.layer_names = ["L1", "L2"]
    vs.layer_criteria = [None, None]
    vs.fig_3d = go.Figure([
        go.Scatter3d(
            x=[0], y=[0], z=[0], name="Neuron_label_1",
            legendgroup="Neuron_label_1", showlegend=True, legendrank=0,
        ),
        go.Scatter3d(
            x=[0], y=[0], z=[0], name="Neuron_label_2",
            legendgroup="Neuron_label_2", showlegend=True, legendrank=100,
        ),
    ])
    vs._neuron_legend_labels_by_layer = {
        0: {1: "Neuron_label_1", "1": "Neuron_label_1"},
        1: {2: "Neuron_label_2", "2": "Neuron_label_2"},
    }
    vs._neuron_legend_ranks = {
        "Neuron_label_1": 0,
        "Neuron_label_2": 100,
    }
    sites = pd.DataFrame({
        "x": [1.0], "y": [0.0], "z": [0.0], "neuron_id": [1],
    })
    vs._collect_layer_sites = lambda idx: (
        (sites.copy(), sites.copy())
        if idx == 0
        else (sites.assign(neuron_id=2), sites.assign(neuron_id=2))
    )
    vs._pre_post_real_synapse_distance = 10.0

    assert vs._plot_pre_post_sites() == 0
    ordered = [
        trace.name for trace in sorted(
            (trace for trace in vs.fig_3d.data if trace.showlegend),
            key=lambda trace: (trace.legendrank, trace.name),
        )
    ]
    assert ordered == [
        "Neuron_label_1", "Neuron_label_1_pre", "Neuron_label_1_post",
        "Neuron_label_2", "Neuron_label_2_pre", "Neuron_label_2_post",
    ]


def test_selected_solid_shape_and_csv_synapse_color_reach_plotly(
    tmp_path, monkeypatch
):
    """The selected solid shape and per-pre-neuron connector color render."""
    import plotly.graph_objects as go

    monkeypatch.setattr(vs_mod.sv, "getNeurons", _resolved_getneurons)
    csv_path = tmp_path / "shape.csv"
    csv_path.write_text(
        "layer,neuron,synapse_color\n"
        "1,aMe12,#00ff00\n"
        "2,dn1,#0000ff\n"
    )
    vs = VisualizeSkeleton(
        dataset="hemibrain:v1.2.1",
        neuron_layers=["unused"],
        layer_map_csv=str(csv_path),
        client=FakeClient(dataset="hemibrain:v1.2.1"),
        verbose=False,
        output_dir=str(tmp_path),
        include_timestamp=False,
        script_path=str(tmp_path),
        brain_mesh="none",
        synapse_mode="tetrahedron",
    )
    conn = pd.DataFrame({
        "bodyId_pre": [101], "bodyId_post": [202],
        "x_pre": [0.0], "y_pre": [0.0], "z_pre": [0.0],
        "x_post": [0.0], "y_post": [0.0], "z_post": [10.0],
    })
    monkeypatch.setattr(vs_mod, "fetch_synapse_connections", lambda **_: conn.copy())
    vs._load_cached_synapses = lambda source, target: (
        None, {(str(pre), str(post)) for pre in source for post in target}
    )
    vs._save_cached_synapses = lambda *args, **kwargs: None
    vs._save_synapse_data = lambda *args, **kwargs: None
    vs.fig_3d = go.Figure()

    assert vs.plot_synapses() == 0
    mesh = next(trace for trace in vs.fig_3d.data if trace.type == "mesh3d")
    # A tetrahedron has four vertices per synapse; this distinguishes it from
    # the sphere/cone mesh paths and proves the selected shape reached the
    # renderer rather than falling back to scatter.
    assert len(mesh.x) == 4
    assert "0, 255, 0" in str(mesh.color)


# ---------------------------------------------------------------------------
# build_site_mesh
# ---------------------------------------------------------------------------

class TestBuildSiteMesh:
    def test_sphere_and_cone_return_mesh3d(self):
        import numpy as np
        from statvis import build_site_mesh

        coords = np.array([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
        sphere = build_site_mesh(coords, mode="sphere", size=1.0, color="red", opacity=0.5)
        cone = build_site_mesh(coords, mode="cone", size=1.0, color="blue", opacity=0.5)
        assert sphere.type == "mesh3d"
        assert cone.type == "mesh3d"
        # Two instances of the template each produce x/y/z vertex arrays.
        assert len(sphere.x) == 2 * 26  # 2 spheres * (2 poles + 8*3 rings)
        assert len(cone.x) == 2 * (2 + 24)  # 2 cones * (tip + base-center + ring)


# ---------------------------------------------------------------------------
# pre/post cutoff filters the full connection table (FlyWire parquet)
# ---------------------------------------------------------------------------

class TestPrePostCutoff:
    def test_flywire_coordinate_units_use_table_statistics_not_filtered_max(
            self, tmp_path):
        """A low-z selected row must not be voxel-scaled when the table is
        stored in nanometres and another row establishes the table units."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        vs = object.__new__(VisualizeSkeleton)
        vs.dataset = "flywire_FAFB_v783"
        vs.script_path = str(tmp_path)
        vs.min_synapse_num = 0
        vs._vprint = lambda *a, **k: None

        dataset_dir = tmp_path / "datasets" / "flywire_FAFB_v783"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({
            "pre_root_id": [1, 9], "post_root_id": [2, 2],
            "pre_x": [1000.0, 2000.0], "pre_y": [1000.0, 2000.0],
            "pre_z": [5000.0, 50000.0],
            "post_x": [3000.0, 3000.0], "post_y": [3000.0, 3000.0],
            "post_z": [5000.0, 50000.0],
        })
        pq.write_table(
            pa.Table.from_pandas(frame),
            dataset_dir / "flywire_FAFB_v783_synapse_table.parquet",
        )

        out = vs._read_flywire_connection_frame(source_ids={"1"})
        assert out is not None and len(out) == 1
        assert out.iloc[0]["x_pre"] == pytest.approx(1000.0)
        assert out.iloc[0]["z_pre"] == pytest.approx(5000.0)

    def test_flywire_site_frame_filters_below_cutoff(self, tmp_path):
        """Rows whose connection weight is below min_synapse_num are dropped
        before splitting into pre/post sites."""
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        vs = object.__new__(VisualizeSkeleton)
        vs.dataset = "flywire_FAFB_v783"
        vs.script_path = str(tmp_path)
        vs.min_synapse_num = 3
        vs._vprint = lambda *a, **k: None

        dataset_dir = tmp_path / "datasets" / "flywire_FAFB_v783"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "pre_root_id": [1, 1, 2, 9],
            "post_root_id": [9, 9, 9, 1],
            "pre_x": [1000.0, 2000.0, 3000.0, 4000.0],
            "pre_y": [1000.0, 2000.0, 3000.0, 4000.0],
            "pre_z": [50000.0, 50000.0, 50000.0, 50000.0],
            "post_x": [5000.0, 6000.0, 7000.0, 8000.0],
            "post_y": [5000.0, 6000.0, 7000.0, 8000.0],
            "post_z": [50000.0, 50000.0, 50000.0, 50000.0],
            "weight": [5, 2, 4, 6],
        })
        pq.write_table(
            pa.Table.from_pandas(df),
            dataset_dir / "flywire_FAFB_v783_synapse_table.parquet",
        )

        out = vs._read_flywire_site_frame({"1", "2"})
        assert out is not None and not out.empty
        # Only weight >= 3 survive: pre sites for bodyId 1 (w=5) and 2 (w=4),
        # plus one post site for bodyId 1 (w=6). The w=2 row is dropped.
        assert len(out) == 3
        assert (out["role"] == "pre").sum() == 2
        assert (out["role"] == "post").sum() == 1

        # The explicit zero threshold used by pseudo-real sizing gets a
        # separate frame-cache key and can see the otherwise filtered row.
        unfiltered = vs._read_flywire_connection_frame(
            source_ids={"1", "2"}, min_synapse_num=0,
        )
        assert unfiltered is not None and len(unfiltered) == 3
        assert 2 in set(unfiltered["weight"].tolist())


# ---------------------------------------------------------------------------
# pre-site-only BANC tables (no persisted post-site coordinates)
# ---------------------------------------------------------------------------

class TestBancPreOnlyTable:
    def test_post_columns_mirrored_in_memory_and_filtered(self, tmp_path):
        """BANC-derived tables carry pre-site coordinates only; the reader
        synthesizes the post columns from the pre site so markers land on
        the pre-synaptic site, and the syn_count threshold still applies."""
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        vs = object.__new__(VisualizeSkeleton)
        vs.dataset = "banc_v888"
        vs.script_path = str(tmp_path)
        vs.min_synapse_num = 2
        vs._vprint = lambda *a, **k: None

        dataset_dir = tmp_path / "datasets" / "banc_v888"
        dataset_dir.mkdir(parents=True)
        df = pd.DataFrame({
            "pre_root_id": np.array([100, 100], dtype=np.int64),
            "post_root_id": np.array([200, 300], dtype=np.int64),
            "syn_count": [5, 1],
            "x_pre": [1000.0, 1000.0],
            "y_pre": [2000.0, 2000.0],
            "z_pre": [300000.0, 300000.0],
        })
        pq.write_table(
            pa.Table.from_pandas(df),
            dataset_dir / "banc_v888_synapse_table.parquet",
        )

        out = vs._read_flywire_connection_frame(source_ids={"100"})
        assert out is not None and len(out) == 1  # syn_count=1 row filtered
        row = out.iloc[0]
        assert row["bodyId_post"] == "200"
        assert row["x_post"] == pytest.approx(row["x_pre"])
        assert row["y_post"] == pytest.approx(row["y_pre"])
        # nanometre coordinates must not be scaled (z stays 300000)
        assert row["z_post"] == pytest.approx(300000.0)

    def test_missing_pre_columns_still_rejected(self, tmp_path):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        vs = object.__new__(VisualizeSkeleton)
        vs.dataset = "banc_v888"
        vs.script_path = str(tmp_path)
        vs.min_synapse_num = 0
        vs._vprint = lambda *a, **k: None

        dataset_dir = tmp_path / "datasets" / "banc_v888"
        dataset_dir.mkdir(parents=True)
        df = pd.DataFrame({
            "pre_root_id": np.array([1], dtype=np.int64),
            "post_root_id": np.array([2], dtype=np.int64),
            "x_pre": [1.0],
        })
        pq.write_table(
            pa.Table.from_pandas(df),
            dataset_dir / "banc_v888_synapse_table.parquet",
        )
        assert vs._read_flywire_connection_frame() is None


class TestLocalSynapseTableRecovery:
    def _vis(self, tmp_path, dataset):
        vs = object.__new__(VisualizeSkeleton)
        vs.dataset = dataset
        vs.script_path = str(tmp_path)
        vs.min_synapse_num = 0
        messages = []
        vs._vprint = lambda text, *a, **k: messages.append(str(text))
        return vs, messages

    def test_truncated_table_triggers_rebuild_and_warns(self, tmp_path, monkeypatch):
        """A table with no valid footer is rebuilt through the dataset
        preparation entry point instead of silently returning None."""
        import FAFB_file_converter as fafb_mod

        dataset_dir = tmp_path / "datasets" / "flywire_FAFB_v783"
        dataset_dir.mkdir(parents=True)
        table = dataset_dir / "flywire_FAFB_v783_synapse_table.parquet"
        table.write_bytes(b"PAR1 truncated")
        vs, messages = self._vis(tmp_path, "flywire_FAFB_v783")

        calls = []

        def fake_ensure(dataset, ddir, *a, **k):
            calls.append(dataset)
            # Rebuild a minimal valid table in place.
            pd.DataFrame({
                "pre_x": [1000], "pre_y": [2000], "pre_z": [26000],
                "post_x": [1000], "post_y": [2000], "post_z": [26000],
                "pre_root_id_720575940": ["720575940596125868"],
                "post_root_id_720575940": ["720575940596125869"],
            }).to_parquet(table, index=False)
            return True

        monkeypatch.setattr(fafb_mod, "ensure_flywire_data", fake_ensure)
        vs._read_flywire_connection_frame(source_ids={"720575940596125868"})

        assert calls == ["flywire_FAFB_v783"]
        assert any("unreadable" in m for m in messages)

    def test_unreadable_without_source_does_not_raise(self, tmp_path, monkeypatch):
        import FAFB_file_converter as fafb_mod

        dataset_dir = tmp_path / "datasets" / "flywire_FAFB_v783"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "flywire_FAFB_v783_synapse_table.parquet").write_bytes(
            b"PAR1 truncated")
        vs, messages = self._vis(tmp_path, "flywire_FAFB_v783")

        monkeypatch.setattr(
            fafb_mod, "ensure_flywire_data",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no raw source")))

        # Must not raise; the failed recovery is reported and the read
        # returns None.
        assert vs._read_flywire_connection_frame() is None
        assert any("rebuild" in m or "unreadable" in m for m in messages)

    def test_upgrade_skipped_when_marker_present(self, tmp_path):
        """A marked table is neither rebuilt nor re-encoded."""
        vs, _ = self._vis(tmp_path, "flywire_FAFB_v783")
        vs._ensure_local_synapse_table_ready(str(tmp_path / "missing.parquet"))
        # Unreadable + no source: handled, and the once-per-session flag set.
        assert vs._local_synapse_table_checked is True
