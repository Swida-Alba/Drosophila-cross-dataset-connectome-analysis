"""Plan 2026-09-07 Phase E: exported UserGuide pathfinding content.

Verifies that all three pathfinding tool specs (find_path, find_shortest,
inter_dataset) carry the reusable explanation block, that the block renders
in every output format, that the new file patterns match the artifacts the
backend now writes, and that the glossary documents the new metadata keys.
"""

import json

from pathlib import Path

import ui.output_guide as og


def _make_run_folder(tmp_path):
    run = tmp_path / "find-paths-complete_TEST_src_to_tgt_L2w3"
    run.mkdir()
    (run / "parameters.txt").write_text("requested_threshold: 3\n")
    (run / "data_details").mkdir()
    (run / "data_details" / "untyped_dropped_records.csv").write_text("a\n")
    viz = run / "visualization" / "visualization_data"
    viz.mkdir(parents=True)
    (run / "visualization" / "Network_run.html").write_text("x")
    (viz / "run_data_connections.csv").write_text("a\n")
    (viz / "type_paths_visualized.csv").write_text("p\n")
    bviz = run / "bodyId_visualization" / "visualization_data"
    bviz.mkdir(parents=True)
    (run / "bodyId_visualization" / "Network_run.html").write_text("x")
    (run / "bodyId_visualization" / "Heatmap_run.html").write_text("x")
    (run / "bodyId_visualization" / "Sankey_run.html").write_text("x")
    (bviz / "run_data_connections.csv").write_text("a\n")
    (bviz / "bodyId_paths_visualized.csv").write_text("p\n")
    return run


def test_all_three_pathfinding_specs_carry_explanation():
    for tool in ("find_path", "find_shortest", "inter_dataset"):
        spec = og.TOOL_GUIDE_SPECS[tool]
        assert spec.get("explanation"), tool
        headings = [s["heading"] for s in spec["explanation"]]
        assert any("path set was produced" in h for h in headings), tool
        assert any("vocabulary" in h for h in headings), tool


def test_explanation_pipeline_order():
    sections = og.TOOL_GUIDE_SPECS["find_path"]["explanation"]
    pipeline = next(s for s in sections if s.get("pipeline"))["pipeline"]
    text = "\n".join(pipeline)
    assert "requested threshold" in text
    assert "lossless hop/dead-end pruning" in text
    assert "Edge Budget floor w0" in text
    assert "StrongestFirst path-budget ordering and τ" in text
    assert "applied threshold / retained path set" in text
    assert "visualization-only edge limit" in text
    # reader order: threshold before floor before budget before drawing cap
    assert text.index("lossless") < text.index("Edge Budget") \
        < text.index("StrongestFirst") < text.index("visualization-only")


def test_explanation_vocabulary_table():
    sections = og.TOOL_GUIDE_SPECS["find_shortest"]["explanation"]
    table = next(s for s in sections if s.get("table"))["table"]
    flat = " ".join(str(cell) for row in table for cell in row)
    for token in ("W*", "τ", "w2", "w1", "w0", "applied_threshold",
                  "paths_complete", "pruned", "bottleneck"):
        assert token in flat, token


def test_new_file_patterns_match_artifacts(tmp_path):
    run = _make_run_folder(tmp_path)
    content = og.assemble_run_content(run, "find_path", {})
    matched = {entry["pattern"]: entry["matched"]
               for entry in content["entries"] if entry["matched"]}
    assert matched["data_details/untyped_dropped_records.csv"]
    assert matched["visualization/visualization_data/"
                   "type_paths_visualized.csv"]
    assert matched["bodyId_visualization/Network_*.html"]
    assert matched["bodyId_visualization/Heatmap_*.html"]
    assert matched["bodyId_visualization/Sankey_*.html"]
    assert matched["bodyId_visualization/visualization_data/"
                   "*_data_connections.csv"]
    assert matched["bodyId_visualization/visualization_data/"
                   "bodyId_paths_visualized.csv"]


def test_explanation_renders_in_all_formats(tmp_path):
    run = _make_run_folder(tmp_path)
    content = og.assemble_run_content(run, "find_shortest", {})
    assert content["explanation"]

    html = og.render_html(content)
    assert "Pathfinding model" in html
    assert "How the path set was produced" in html
    assert "Drop Untyped Neurons" in html
    assert "StrongestFirst path-budget ordering and τ" in html

    md = og.render_markdown(content)
    assert "## Pathfinding model" in md
    assert "requested threshold" in md
    assert "| W* |" in md
    assert "StrongestFirst path-budget ordering and τ" in md

    txt = og.render_txt(content)
    assert "PATHFINDING MODEL" in txt
    assert "-> requested threshold" in txt
    assert "W*" in txt
    assert "StrongestFirst path-budget ordering and τ" in txt


def test_write_run_guide_contains_model(tmp_path):
    run = _make_run_folder(tmp_path)
    path = og.write_run_guide(run, "find_path", {}, fmt="markdown")
    text = path.read_text(encoding="utf-8")
    assert "Pathfinding model" in text
    assert "applied_threshold" in text
    assert "untyped_dropped_records" in text


def test_glossary_documents_new_keys():
    for key in ("edge_budget_landing", "strongest_retained_bottleneck",
                "strongest_dropped_bottleneck", "applied_threshold_source",
                "strongest_first_budget", "tau_canonical",
                "requested_threshold", "drop_untyped", "untyped_side",
                "untyped_dropped_rows", "untyped_dropped_neurons",
                "edge_budget", "edge_budget_applied",
                "strongest_first_tau"):
        description, _range = og.glossary_entry(key)
        assert description != og.glossary_entry("__missing__")[0], key


def test_dynamic_threshold_columns_use_prefix_glossary(tmp_path):
    run = tmp_path / "inter_dataset_run"
    output = run / "similarity_matrices"
    output.mkdir(parents=True)
    (output / "similarity_query_q1.csv").write_text(
        "query_id,query_label,dataset_1,dataset_2,threshold_banc_v888,"
        "jaccard_similarity\nq1,example,a,b,3,0.5\n",
        encoding="utf-8",
    )

    content = og.assemble_run_content(run, "inter_dataset", {})
    entry = next(
        item for item in content["entries"]
        if item["pattern"] == "similarity_matrices/similarity_query_*.csv"
    )
    assert "threshold_banc_v888" in entry["columns"]
    assert "threshold_" not in entry["columns"]
    metric = next(
        item for item in content["metrics"]
        if item["name"] == "threshold_banc_v888"
    )
    assert "requested-threshold" in metric["description"]


def test_inter_dataset_nested_outputs_point_to_explanation():
    spec = og.TOOL_GUIDE_SPECS["inter_dataset"]
    dataset_entry = next(e for e in spec["files"]
                         if e["pattern"] == "dataset_data/**")
    assert "pathfinding" in dataset_entry["description"]
    assert spec.get("explanation")


# ---------------------------------------------------------------------------
# Applied threshold / bottleneck values in the exported guide
# ---------------------------------------------------------------------------

_PROVENANCE = {
    "requested_threshold": 3,
    "applied_threshold": 8,
    "applied_threshold_source": "strongest_first_budget",
    "strongest_first_budget": 1000000,
    "strongest_first_budget_bitten": True,
    "strongest_first_tau": 12.0,
    "tau_canonical": 8,
    "strongest_dropped_bottleneck": 7.0,
    "edge_budget": 1000000,
    "edge_budget_applied": False,
    "edge_budget_landing": None,
    "edge_weight_floor": None,
    "strongest_retained_bottleneck": 15.0,
    "paths_complete": False,
}


def _add_provenance(run, provenance=None, use_json=True):
    provenance = provenance or dict(_PROVENANCE)
    if use_json:
        (run / "all_attributes.json").write_text(
            json.dumps(provenance), encoding="utf-8")
    else:
        lines = ["Parameters for processing src to tgt:"]
        for key, value in provenance.items():
            if value is None:
                value = "not applied" if "floor" in key else "not reached"
            lines.append(f"{key}: {value}")
        (run / "parameters.txt").write_text("\n".join(lines) + "\n",
                                            encoding="utf-8")


def test_applied_state_read_from_all_attributes(tmp_path):
    run = _make_run_folder(tmp_path)
    _add_provenance(run)
    state = og._read_applied_state(run)
    assert state["applied_threshold"] == 8
    assert state["strongest_first_tau"] == 12.0
    assert state["paths_complete"] is False


def test_applied_state_fallback_parses_parameters_txt(tmp_path):
    run = _make_run_folder(tmp_path)
    _add_provenance(run, use_json=False)
    state = og._read_applied_state(run)
    assert state["applied_threshold"] == 8
    # the legacy 'applied_tau (min path bottleneck)' alias maps to tau
    assert state["strongest_first_tau"] == 12.0
    assert state["edge_weight_floor"] is None
    assert state["paths_complete"] is False


def test_applied_block_renders_under_parameters_in_all_formats(tmp_path):
    run = _make_run_folder(tmp_path)
    _add_provenance(run)
    content = og.assemble_run_content(run, "find_shortest", {})

    assert content["applied"]["applied_threshold"] == 8
    headline = og._applied_headline(content["applied"])
    assert "EXACTLY a complete run at Min Synapse Count = 8" in headline
    assert "strongest_first_budget" in headline

    html = og.render_html(content)
    assert "Applied threshold (this run)" in html
    assert "EXACTLY a complete run at Min Synapse Count = 8" in html
    assert "<code>12</code>" in html          # tau row

    md = og.render_markdown(content)
    assert "## Applied threshold (this run)" in md
    assert "applied threshold (equivalent Min Synapse Count)" in md
    assert "w0 (Edge Budget floor)" in md

    txt = og.render_txt(content)
    assert "APPLIED THRESHOLD (THIS RUN)" in txt
    assert "EXACTLY a complete run at Min Synapse Count = 8" in txt
    assert "w2 (strongest dropped path bottleneck): 7" in txt


def test_vocabulary_table_shows_this_run_values(tmp_path):
    run = _make_run_folder(tmp_path)
    _add_provenance(run)
    content = og.assemble_run_content(run, "find_path", {})

    html = og.render_html(content)
    assert "<th>This run</th>" in html
    # tau row shows this run's landing tau; w2 row shows the dropped value
    assert "<td>τ" in html
    assert "StrongestFirst landing" in html
    assert "<code>7</code>" in html
    # 'pruned' maps onto edge_budget_applied for the run state
    assert "<code>False</code>" in html

    txt = og.render_txt(content)
    assert "This run" in txt
    md = og.render_markdown(content)
    assert "| This run |" in md


def test_no_provenance_omits_applied_block(tmp_path):
    run = _make_run_folder(tmp_path)  # parameters.txt lacks applied_threshold
    content = og.assemble_run_content(run, "find_path", {})
    assert content["applied"] is None
    assert content["applied_by_dataset"] is None

    html = og.render_html(content)
    assert "Applied threshold (this run)" not in html
    assert "<th>This run</th>" not in html
    assert "Pathfinding model" in html  # model section still renders


def test_by_dataset_banner_renders_for_comparison_runs(tmp_path):
    run = tmp_path / "cross-dataset_TEST"
    run.mkdir()
    (run / "effective_thresholds.json").write_text(json.dumps({
        "banner": ["hemibrain: asked [1, 3, 5] -> applied [1, 9] (τ=9)"],
        "datasets": {
            "hemibrain:v1.2.1": {
                "input": [1, 3, 5],
                "effective": [1, 9],
                "skipped": [3, 5],
                "applied_folder": {"3": 9, "5": 9},
                "tau": 9.0,
            },
        },
    }), encoding="utf-8")
    content = og.assemble_run_content(run, "inter_dataset", {})

    assert content["applied_by_dataset"]["hemibrain:v1.2.1"]["tau"] == 9.0
    html = og.render_html(content)
    assert "Per-dataset" in html
    assert "hemibrain:v1.2.1" in html
    assert "3\u21929" in html  # collapsed 3 -> 9 with a literal arrow

    txt = og.render_txt(content)
    assert "asked [1, 3, 5] -> applied [1, 9]" in txt
    md = og.render_markdown(content)
    assert "3->9" in md


def test_by_dataset_provenance_table_shows_applied_and_budget_fields(tmp_path):
    run = tmp_path / "cross-dataset_PROVENANCE"
    run.mkdir()
    run_meta = dict(_PROVENANCE)
    run_meta.update({
        "dataset": "hemibrain:v1.2.1",
        "threshold": 3,
        "tau": 12.0,
        "path_mode": "all",
        "comparison_mode": "path",
        "drop_untyped": True,
    })
    (run / "effective_thresholds.json").write_text(json.dumps({
        "banner": "Pathfinding provenance: hemibrain:v1.2.1",
        "datasets": {
            "hemibrain:v1.2.1": {
                "input": [3],
                "effective": [8],
                "runs": [run_meta],
            },
        },
    }), encoding="utf-8")

    content = og.assemble_run_content(run, "inter_dataset", {})
    html = og.render_html(content)
    assert "<th>Applied</th>" in html
    assert "<th>SF budget</th>" in html
    assert "<th>Edge budget</th>" in html
    assert "<code>8</code>" in html
    assert "<code>12</code>" in html

    text = og.render_txt(content)
    assert "requested 3 -> applied 8" in text
    assert "SF budget 1000000" in text
    assert "edge budget 1000000" in text


def test_combination_manifest_renders_query_identity_and_provenance(tmp_path):
    run = tmp_path / "cross-dataset_COMBINATIONS"
    run.mkdir()
    results = run / "comparison_results"
    results.mkdir()
    rows = [
        {
            "query_id": "q-low", "query_label": "Low density",
            "threshold_mode": "combinations", "dataset": "BANC",
            "requested_threshold": "3", "applied_threshold": "4",
            "applied_threshold_source": "edge_budget",
            "strongest_first_budget": "1000000",
            "strongest_first_budget_bitten": "False",
            "strongest_first_tau": "12", "tau": "12",
            "tau_canonical": "4", "strongest_dropped_bottleneck": "—",
            "strongest_retained_bottleneck": "15",
            "edge_budget": "100", "edge_budget_applied": "True",
            "edge_budget_landing": "3", "edge_weight_floor": "4",
            "paths_complete": "True",
        },
        {
            "query_id": "q-low", "query_label": "Low density",
            "threshold_mode": "combinations", "dataset": "FAFB",
            "requested_threshold": "7", "applied_threshold": "7",
            "applied_threshold_source": "requested",
            "strongest_first_budget": "1000000",
            "strongest_first_budget_bitten": "False",
            "strongest_first_tau": "10", "tau": "10",
            "tau_canonical": "7", "strongest_dropped_bottleneck": "—",
            "strongest_retained_bottleneck": "10",
            "edge_budget": "100", "edge_budget_applied": "False",
            "edge_budget_landing": "—", "edge_weight_floor": "—",
            "paths_complete": "True",
        },
    ]
    columns = list(rows[0])
    text = ",".join(columns) + "\n"
    text += "\n".join(
        ",".join(str(row.get(column, "")) for column in columns)
        for row in rows
    ) + "\n"
    (results / "threshold_combinations.csv").write_text(
        text, encoding="utf-8")

    content = og.assemble_run_content(run, "inter_dataset", {})
    manifest = content["threshold_queries"]
    assert manifest["mode"] == "combinations"
    assert manifest["queries"][0]["thresholds"] == {"BANC": 3, "FAFB": 7}

    html = og.render_html(content)
    assert "Cross-dataset threshold query rows" in html
    assert "q-low" in html
    assert "Edge budget applied" in html
    assert "strongest_first_budget" in html or "SF budget" in html
    assert "w0" in html and "w2" in html and "W*" in html

    md = og.render_markdown(content)
    assert "Cross-dataset threshold query rows:" in md
    assert "BANC=3" in md
    assert "w0 4" in md

    txt = og.render_txt(content)
    assert "Cross-dataset threshold query rows:" in txt
    assert "q-low (Low density)" in txt
    assert "edge budget 100 (applied True)" in txt
    assert "w2 —" in txt


def test_key_params_include_shortest_coverage_knobs():
    """The run guide's Run-parameters table surfaces the coverage
    early-stop knobs when set (audit round 2026-10-04: they were missing
    from the priority list)."""
    from ui.output_guide import _key_params
    rows = _key_params({
        "dataset": "x",
        "shortest_source_coverage": 0.0,
        "shortest_target_coverage": 1.0,
    })
    keys = [k for k, _ in rows]
    assert "shortest_source_coverage" in keys
    assert "shortest_target_coverage" in keys
    # Unset knobs (None) never clutter the table.
    rows2 = _key_params({"dataset": "x",
                         "shortest_source_coverage": None})
    assert "shortest_source_coverage" not in [k for k, _ in rows2]
