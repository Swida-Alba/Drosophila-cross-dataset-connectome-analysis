#!/usr/bin/env python3
"""
Overall "lite" end-to-end suite for DROCAT — 13 scenarios.

Drives the exact UI execution layer (ui.runner.ScriptRunner), the same code
path triggered by clicking "Run" in the web UI, against live NeuPrint /
NeuronBridge / CAVE data with the tokens in config_local.json / config.json.

Scenarios
---------
 1. Complete paths:  aMe12 -> PPL101, L1, reciprocal x hemisphere matrix
 2. Shortest paths:  aMe12 -> PPL101, L1
 3. Find network:    query 'aMe.*'
 4. Cross-dataset:   aMe12 -> PPL101, L1, male-cns:v1.0 / FAFB v783 /
                     BANC v626 / BANC v888 (same reciprocal x hemisphere
                     matrix as scenario 1)
 5. 3D skeleton:     aMe12, SMP238, PPL101 (3 layers, per-neuron colors+alphas)
 6. Net-Viz:         custom edge list a->b:10, b->c:20, a->c:5, d->a:3
 7. Homologs:        aMe12 (male-cns:v1.0) -> flywire_FAFB_v783
 8. Find similar:    morphological + connectivity similar neurons of aMe12
                     (flywire_FAFB_v783)
 9. Profiling:       'aMe.*' in male-cns:v1.0 + flywire_FAFB_v783
10. NB find neurons: SS01015
11. NB find lines:   aMe12
12. NB co-labeling:  SS01015, VT037867, SS46115
13. FlyLight:        lines from scenario 12

Usage (from project root, inside the `drocat` conda env):
    python tests/e2e/run_overall_lite_e2e.py
    python tests/e2e/run_overall_lite_e2e.py --keep-output
    python tests/e2e/run_overall_lite_e2e.py --scenario complete_paths,network
    python tests/e2e/run_overall_lite_e2e.py --variants quick   # 2 variants
                                                                 # for matrix
                                                                 # scenarios

The runner uses one disposable sandbox at /tmp/drocat_lite_e2e. It clears that
sandbox before every run and removes it afterward. Pass --keep-output when
inspecting artifacts after a run.

Each scenario's actual output file tree is recorded in
/tmp/drocat_lite_e2e/file_trees.json — the ground truth used to correct
docs/OUTPUT_FILES.md.
"""

import argparse
import asyncio
import copy
import csv
import io
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ui.runner import ScriptRunner  # noqa: E402

ROOT = Path("/tmp/drocat_lite_e2e")

DATASET_MCNS = "male-cns:v1.0"
DATASET_FAFB = "flywire_FAFB_v783"
DATASET_BANC626 = "banc_v626"
DATASET_BANC888 = "banc_v888"


def reset_output_root() -> None:
    """Remove stale artifacts from the runner's exact, test-owned sandbox."""
    if ROOT.is_symlink():
        raise RuntimeError(f"Refusing to remove symlinked test sandbox: {ROOT}")
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True, exist_ok=True)


def cleanup_output_root() -> None:
    """Remove the runner sandbox after verification, if it still exists."""
    if ROOT.is_symlink():
        raise RuntimeError(f"Refusing to remove symlinked test sandbox: {ROOT}")
    if ROOT.exists():
        shutil.rmtree(ROOT)


def out(name: str) -> str:
    return str(ROOT / name)


# ---------------------------------------------------------------------------
# Scenario parameter builders
# ---------------------------------------------------------------------------

def _pathfinding_base(scenario_dir: str, reciprocal: bool, hemi: bool) -> dict:
    """Complete/Shortest Paths constructor params (mirrors the UI tabs)."""
    return {
        "dataset": DATASET_MCNS,
        "sourceNeurons": ["aMe12"],
        "targetNeurons": ["PPL101"],
        "output_dir": out(scenario_dir),
        "min_synapse_num": 3,
        "min_ratio": 0.0,
        "min_traversal_probability": 0.0,
        "max_interlayer": 1,
        "filter_by": "bodyId",
        "pathfinding": "MemoizedDFS",
        "graph_edge_limit_bodyid": 1000000,
        "visualize_before_reconstruct": False,
        "search_columns": "auto",
        "network_layout": "distributed",
        "use_cache": True,
        "edgeN_limit": 500,
        "output_format": "csv",
        "skip_bodyId": True,
        "showfig": False,
        "custom_source_name": "",
        "custom_target_name": "",
        "keyword_in_path_to_remove": ["None"],
        "cache_only": False,
        "saveas": "",
        "separate_hemispheres": hemi,
        "hemisphere_filter": "both",
        "keep_only_hemisphere_conserved_connections": False,
        "symmetry_analysis": hemi,
        "find_reciprocal": reciprocal,
    }


def complete_paths_specs() -> list:
    """Scenario 1: reciprocal x hemisphere-aware matrix."""
    specs = []
    for reciprocal, hemi in [(False, False), (True, False),
                             (False, True), (True, True)]:
        params = _pathfinding_base("complete_paths", reciprocal, hemi)
        specs.append({
            "variant": f"reciprocal={reciprocal}_hemi={hemi}",
            "tool_name": "find_path",
            "method": "find_all_path",
            "constructor_params": params,
            "method_params": None,
            "timeout": 900,
        })
    return specs


def shortest_paths_specs() -> list:
    """Scenario 2: shortest paths, aMe12 -> PPL101, L1."""
    params = _pathfinding_base("shortest_paths", False, False)
    params.pop("pathfinding")
    params["graph_edge_limit_bodyid"] = 0
    params["keyword_in_path_to_remove"] = []
    return [{
        "variant": "default",
        "tool_name": "find_shortest",
        "method": "find_shortest",
        "constructor_params": params,
        "method_params": None,
        "timeout": 900,
    }]


def network_specs() -> list:
    """Scenario 3: FindNetwork with pattern query 'aMe.*'."""
    query = ["aMe.*"]
    params = {
        "dataset": DATASET_MCNS,
        "sourceNeurons": query,
        "targetNeurons": query,
        "output_dir": out("network"),
        "min_synapse_num": 3,
        "min_ratio": 0.0,
        "min_traversal_probability": 0.0,
        "search_columns": "auto",
        "network_layout": "distributed",
        "use_cache": True,
        "edgeN_limit": 500,
        "output_format": "csv",
        "skip_bodyId": True,
        "custom_source_name": "",
        "cache_only": False,
        "saveas": "",
        "separate_hemispheres": False,
        "hemisphere_filter": "both",
        "keep_only_hemisphere_conserved_connections": False,
        "symmetry_analysis": False,
    }
    return [{
        "variant": "query=aMe.*",
        "tool_name": "find_network",
        "method": "find_network",
        "constructor_params": params,
        "method_params": None,
        "timeout": 600,
    }]


def cross_dataset_specs(variants: str) -> list:
    """Scenario 4: cross-dataset aMe12 -> PPL101, L1, four datasets."""
    matrix = [(False, False), (True, False), (False, True), (True, True)]
    if variants == "quick":
        matrix = [(False, False), (True, True)]
    specs = []
    for reciprocal, hemi in matrix:
        params = {
            "datasets": [DATASET_MCNS, DATASET_FAFB, DATASET_BANC626,
                         DATASET_BANC888],
            "source_neurons": ["aMe12"],
            "target_neurons": ["PPL101"],
            "output_folder": out("cross_dataset"),
            "comparison_mode": "path",
            "path_mode": "all",
            "max_interlayer": 1,
            "thresholds": [3],
            "top_edges": 500,
            "graph_edge_limit_bodyid": 1000000,
            "edgeN_limit": 500,
            "pathfinding": "MemoizedDFS",
            "search_columns": "auto",
            "skip_bodyId": True,
            "cache_only": False,
            "auto_type_mapping": True,
            "_min_ratio": 0.0,
            "_min_prob": 0.0,
            "_output_format": "csv",
            "parallel": True,
            "max_workers": 4,
            "separate_hemispheres": hemi,
            "keep_only_hemisphere_conserved_connections": False,
            "symmetry_analysis": hemi,
            "find_reciprocal": reciprocal,
        }
        specs.append({
            "variant": f"reciprocal={reciprocal}_hemi={hemi}",
            "tool_name": "inter_dataset",
            "method": "run",
            "constructor_params": params,
            "method_params": None,
            "timeout": 2400,
        })
    return specs


def skeleton_specs() -> list:
    """Scenario 5: 3-layer skeleton plot with per-neuron colors and alphas."""
    params = {
        "dataset": DATASET_MCNS,
        "neuron_layers": [["aMe12"], ["SMP238"], ["PPL101"]],
        "search_columns": "auto",
        "hemisphere": "both",
        "custom_layer_names": [],
        "output_dir": out("skeleton_3layer"),
        "skeleton_mode": "tube",
        "brain_mesh": "native",
        "vnc_mesh": False,
        "legend_mode": "type",
        "neuron_alpha": 0.3,
        # Different color AND different alpha per neuron (RGBA tuples).
        "neuron_colors": [
            "rgba(255, 0, 0, 0.3)",
            "rgba(0, 255, 0, 0.6)",
            "rgba(0, 0, 255, 0.9)",
        ],
        "synapse_colors": ["rgba(255, 200, 0, 0.5)", "rgba(0, 255, 255, 0.5)"],
        "background_color": "white",
        "skip_synapse": True,
        "min_synapse_num": 3,
        "synapse_size": "3",
        "synapse_alpha": 0.6,
        "synapse_mode": "cone",
        "mesh_roi": [],
        "mesh_color": (100, 100, 100),
        "mesh_alpha": 0.1,
        "cache_neurons": True,
        "cache_synapses": True,
        "smooth_skeleton": False,
        "show_soma": True,
        "show_connectors": False,
        "export_method": "webdriver",
        "export_scale": 3,
        "export_views": False,
        "show_fig": False,
        "brain_mesh_color": "auto",
        "skeleton_mesh_simplification": None,
    }
    return [{
        "variant": "3layers_rgba",
        "tool_name": "plot3d_skeleton",
        "method": "plot",
        "constructor_params": params,
        "method_params": {},
        "timeout": 1500,
    }]


_EDGE_LIST_CSV = "source,target,weight\na,b,10\nb,c,20\na,c,5\nd,a,3\n"


def net_viz_specs() -> list:
    """Scenario 6: Net-Viz with a custom edge list (a->b:10 ...)."""
    ROOT.mkdir(parents=True, exist_ok=True)
    edge_csv = ROOT / "custom_edge_list.csv"
    edge_csv.write_text(_EDGE_LIST_CSV, encoding="utf-8")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_folder = ROOT / "net_viz" / f"plot-network_custom_edgelist_{timestamp}"
    run_folder.mkdir(parents=True, exist_ok=True)
    params = {
        "path_file": str(edge_csv),
        "output_folder": str(run_folder),
        "source_color": "#4A90E2",
        "intermediate_color": "#50E3C2",
        "target_color": "#B8E986",
        "link_color": "rgba(74,144,226,0.3)",
        "network_layout": "hierarchical",
        "showfig": False,
        "generate_empty_network": False,
    }
    return [{
        "variant": "custom_edgelist",
        "tool_name": "plot_path",
        "method": "plot",
        "constructor_params": params,
        "method_params": None,
        "timeout": 300,
    }]


def homologs_specs() -> list:
    """Scenario 7: homologs aMe12 (male-cns:v1.0) -> flywire_FAFB_v783."""
    params = {
        "source": "aMe12",
        "source_dataset": DATASET_MCNS,
        "target_dataset": DATASET_FAFB,
        "output_dir": out("homologs"),
        "top_n": 10,
        "top_k": 15,
        "top_m": 5,
        "similarity_metric": "rank_union",
        "vector_prefiltering": True,
        "include_untyped_partners": True,
        "visualize_skeleton": False,
        "visualize_top_n": 0,
        "visualization_settings": {},
        "min_synapse_threshold": 3,
        "use_cache": True,
        "saveas": "",
        "use_auto_type_mapping": True,
        "ensure_cache_complete": False,
    }
    return [{
        "variant": "aMe12_mcns_to_fafb",
        "tool_name": "find_homologs",
        "method": "find_homologs_fast",
        "constructor_params": params,
        "method_params": None,
        "timeout": 1500,
    }]


def similar_morphology_specs() -> list:
    """Scenario 8a: morphologically similar neurons of aMe12 in FAFB v783."""
    params = {
        "query": "aMe12",
        "dataset": DATASET_FAFB,
        "level": "auto",
        "method": "vector_v2",
        "metric": "cosine",
        "candidate_cap": 100,
        "candidate_source": "auto",
        "roi_filter": None,
        "visualize_top_n": 0,
        "visualize_by": "type",
        "visualization_settings": {},
        "output_dir": out("similar_morphology"),
        "saveas": "",
        "verbose": True,
        "n_workers": 8,
        "use_cache": True,
        "cache_fetched_skeletons": True,
    }
    return [{
        "variant": "aMe12_fafb_morphology",
        "tool_name": "find_similar_morphology",
        "method": "find_similar",
        "constructor_params": params,
        "method_params": None,
        "timeout": 2400,
    }]


def similar_connectivity_specs() -> list:
    """Scenario 8b: connectivity similar neurons of aMe12 in FAFB v783.

    Runs through the Find Similar backend (HomologFinder) with
    target_dataset == source_dataset, matching the unified
    Connectivity → Find Similar sub-tab."""
    params = {
        "source": ["aMe12"],
        "source_dataset": DATASET_FAFB,
        "target_dataset": DATASET_FAFB,
        "output_dir": out("similar_connectivity"),
        "top_n": 10,
        "top_k": 15,
        "top_m": 5,
        "similarity_metric": "rank_union",
        "vector_prefiltering": True,
        "include_untyped_partners": True,
        "use_cache": True,
        "saveas": "",
        "min_synapse_threshold": 3,
        "use_auto_type_mapping": False,
        "ensure_cache_complete": False,
        "visualize_skeleton": False,
        "visualize_top_n": 0,
        "visualization_settings": {},
        "verbose": True,
    }
    return [{
        "variant": "aMe12_fafb_connectivity",
        "tool_name": "find_homologs",
        "method": "find_homologs_multi",
        "constructor_params": params,
        "method_params": {"use_fast": True},
        "timeout": 1500,
    }]


def profiling_specs() -> list:
    """Scenario 9: profiling 'aMe.*' in male-cns:v1.0 + FAFB v783."""
    params = {
        "query": ["aMe.*"],
        "datasets": [DATASET_MCNS, DATASET_FAFB],
        "output_dir": out("profiling"),
        "top_k": 15,
        "top_m": 5,
        "min_synapse_threshold": 3,
        "direction": "both",
        "generate_heatmaps": True,
        "show_figures": False,
        "verbose": True,
        "use_cache": True,
        "aggregation_level": "type",
        "skip_bodyId_level": "auto",
        "ensure_cache_complete": False,
    }
    return [{
        "variant": "aMe_star_two_datasets",
        "tool_name": "connectivity_profiling",
        "method": "run",
        "constructor_params": params,
        "method_params": None,
        "timeout": 1800,
    }]


def nb_find_neuron_specs() -> list:
    """Scenario 10: NB find neurons for driver line SS01015."""
    return [{
        "variant": "SS01015",
        "tool_name": "nb_find_neuron",
        "method": "find_neurons",
        "constructor_params": {"verbose": True},
        "method_params": {
            "line_names": ["SS01015"],
            "output_dir": out("nb_find_neuron"),
            "match_type": "cds",
            "top_n": 10,
            "visualize_top_n": 0,
            "generate_individual_profiles": None,
            "visualize_by": "type",
            "visualization_settings": {},
            "sort_by": "max_score",
            "pdf_images_per_page": (3, 2),
            "background_color": "white",
        },
        "timeout": 900,
    }]


def nb_find_lines_specs() -> list:
    """Scenario 11: NB find lines for aMe12 (male-cns:v1.0)."""
    return [{
        "variant": "aMe12",
        "tool_name": "nb_find_lines",
        "method": "find_lines",
        "constructor_params": {
            "verbose": True,
            "separate_splitgal4": True,
            "region": "Brain",
            "max_workers": 8,
        },
        "method_params": {
            "queries": "aMe12",
            "dataset": DATASET_MCNS,
            "output_dir": out("nb_find_lines"),
            "match_type": "cds",
            "download_images": None,
            "download_img_for_top_n_lines": None,
            "summary_format": None,
            "sort_by": "max",
            "image_formats": ["png", "jpg"],
            "image_types": ["cdm"],
            "max_download_images_per_line": None,
            "flylight_category": None,
            "simple_mode": True,
            "organize_by_region": False,
            "pdf_images_per_page": (3, 2),
            "summary_background_color": "black",
        },
        "timeout": 900,
    }]


def nb_colabel_specs() -> list:
    """Scenario 12: co-labeling SS01015, VT037867, SS46115."""
    return [{
        "variant": "SS01015_VT037867_SS46115",
        "tool_name": "nb_colabel",
        "method": "colabel",
        "constructor_params": {"verbose": True},
        "method_params": {
            "lines": ["SS01015", "VT037867", "SS46115"],
            "output_dir": out("nb_colabel"),
            "similarity_methods": ["jaccard", "weighted_jaccard"],
            "generate_report": True,
            "visualize": True,
            "visualize_top_n": 0,
            "top_n_neurons": 200,
            "min_score": 20000.0,
            "min_type_avg_score": 10000.0,
            "sort_by": "max_score",
            "background_color": "white",
            "pdf_images_per_page": (3, 2),
            "datasets_to_visualize": "all",
            "visualize_by": "type",
            "visualization_settings": {},
        },
        "timeout": 900,
    }]


def flylight_specs() -> list:
    """Scenario 13: FlyLight download for the lines from scenario 12."""
    return [{
        "variant": "SS01015_VT037867_SS46115",
        "tool_name": "flylight_download",
        "method": "download",
        "constructor_params": {
            "formats": ["png"],
            "image_types": ["mip"],
            "region": "Brain",
            "collection_category": None,
            "max_workers": 4,
            "simple_mode": True,
            "use_boto3": True,
            "include_vt_lines": True,
            "verbose": "pbar",
        },
        "method_params": {
            "line_name": ["SS01015", "VT037867", "SS46115"],
            "output_dir": out("flylight_download"),
            "max_files": 2,
            "flat_structure": False,
            "add_timestamp": True,
            "generate_summary": None,
            "summary_images_per_page": (3, 2),
        },
        "timeout": 600,
    }]


# ---------------------------------------------------------------------------
# Scenario registry (order = run order)
# ---------------------------------------------------------------------------

SCENARIO_BUILDERS = {
    "complete_paths": complete_paths_specs,
    "shortest_paths": shortest_paths_specs,
    "network": network_specs,
    "cross_dataset": cross_dataset_specs,
    "skeleton_3layer": skeleton_specs,
    "net_viz_edgelist": net_viz_specs,
    "homologs": homologs_specs,
    "similar_morphology": similar_morphology_specs,
    "similar_connectivity": similar_connectivity_specs,
    "profiling": profiling_specs,
    "nb_find_neuron": nb_find_neuron_specs,
    "nb_find_lines": nb_find_lines_specs,
    "nb_colabel": nb_colabel_specs,
    "flylight_download": flylight_specs,
}


# ---------------------------------------------------------------------------
# Execution machinery
# ---------------------------------------------------------------------------

def probe_files(directory: str) -> list:
    """List output files with size and (for tabular files) row counts."""
    results = []
    root = Path(directory)
    if not root.exists():
        return results
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        try:
            size = f.stat().st_size
        except OSError:
            continue
        entry = {"name": f.name, "path": str(f), "size": size}
        if f.suffix.lower() in (".csv", ".tsv"):
            try:
                with open(f, "r", encoding="utf-8", errors="replace") as fh:
                    entry["rows"] = sum(1 for _ in fh) - 1
            except OSError:
                pass
        elif f.suffix.lower() in (".xlsx", ".xls"):
            try:
                import openpyxl
                wb = openpyxl.load_workbook(str(f), read_only=True)
                ws = wb.worksheets[0]
                entry["rows"] = max(ws.max_row - 1, 0)
                wb.close()
            except Exception:
                pass
        results.append(entry)
    return results


def relative_tree(directory: str) -> dict:
    """Map relative path -> size for every file under directory."""
    root = Path(directory)
    if not root.exists():
        return {}
    tree = {}
    for f in sorted(root.rglob("*")):
        if f.is_file():
            tree[str(f.relative_to(root))] = f.stat().st_size
    return tree


def spec_output_dir(spec: dict) -> str:
    """Resolve the directory a tool writes into from its UI-style params."""
    for key in ("output_dir", "output_folder"):
        for params in (spec.get("constructor_params") or {},
                       spec.get("method_params") or {}):
            value = params.get(key)
            if value:
                return str(value)
    return out(spec["name"])


async def run_spec(spec: dict, runner: ScriptRunner) -> dict:
    name = spec["name"]
    variant = spec["variant"]
    log_path = ROOT / f"{name}__{variant}.log"
    logs = []

    def log_cb(line: str, stream: str = "stdout"):
        logs.append(line)

    started = time.time()
    result = {"name": name, "variant": variant, "status": "pending",
              "duration_s": 0.0, "error": None}
    try:
        res = await asyncio.wait_for(
            runner.run(
                spec["tool_name"],
                spec["constructor_params"],
                spec["method"],
                method_params=spec.get("method_params"),
                log_callback=log_cb,
                output_dir=spec_output_dir(spec),
            ),
            timeout=spec["timeout"],
        )
        result["status"] = (
            "success"
            if res["returncode"] == 0 and not res["cancelled"]
            else "failed"
        )
        result["returncode"] = res["returncode"]
        result["cancelled"] = res["cancelled"]
        scan_dir = res.get("output_folder") or spec_output_dir(spec)
        result["files"] = probe_files(scan_dir)
        result["tree"] = relative_tree(scan_dir)
    except asyncio.TimeoutError:
        runner.cancel()
        result["status"] = "timeout"
        result["error"] = f"exceeded {spec['timeout']}s"
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["duration_s"] = round(time.time() - started, 1)
        log_path.write_text("\n".join(logs), encoding="utf-8",
                            errors="replace")
        result["log_lines"] = len(logs)
    return result


def build_scenarios(variants: str, only: list = None) -> list:
    """Expand the scenario registry into a flat run list."""
    scenarios = []
    for name, builder in SCENARIO_BUILDERS.items():
        if only and name not in only:
            continue
        # cross_dataset_specs needs the variant budget; others take none.
        specs = builder(variants) if name == "cross_dataset" else builder()
        for spec in specs:
            spec = copy.deepcopy(spec)
            spec["name"] = name
            scenarios.append(spec)
    return scenarios


def reset_scenario_artifacts(names: list) -> None:
    """Remove only the named scenarios' outputs + logs (keeps other results).

    Used when re-running a subset with --scenario so a previous full run is
    not wiped by reset_output_root().
    """
    if not names:
        return
    for name in names:
        target = ROOT / name
        if target.is_symlink():
            raise RuntimeError(f"Refusing to remove symlinked path: {target}")
        if target.exists():
            shutil.rmtree(target)
        for pattern in (f"{name}__*.log", f"{name}_*.log"):
            for stale in ROOT.glob(pattern):
                stale.unlink(missing_ok=True)


async def main(keep_output: bool = False, only: list = None,
               variants: str = "full") -> None:
    if only:
        # Re-running a subset: keep every other scenario's artifacts.
        reset_scenario_artifacts(only)
        ROOT.mkdir(parents=True, exist_ok=True)
    else:
        reset_output_root()
    summary = []
    try:
        scenarios = build_scenarios(variants, only)
        for spec in scenarios:
            runner = ScriptRunner()
            result = await run_spec(spec, runner)
            summary.append(result)
            files = result.get("files", [])
            tree = result.get("tree", {})
            print(
                f"[{result['status'].upper():8s}] {result['name']:24s} "
                f"{result['variant']:28s} {result['duration_s']:7.1f}s "
                f"files={len(files)} tree_entries={len(tree)}"
            )
            if result["error"]:
                print(f"         error: {result['error']}")

        # Merge with any previous run's entries when re-running a subset.
        if only:
            try:
                previous = json.loads((ROOT / "results.json").read_text())
                known = {(r["name"], r["variant"]) for r in summary}
                previous = [r for r in previous
                            if (r.get("name"), r.get("variant")) not in known]
                summary = previous + summary
            except (OSError, ValueError):
                pass

        (ROOT / "results.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        file_trees = {
            r["name"] + "::" + r["variant"]: r.get("tree", {})
            for r in summary
        }
        (ROOT / "file_trees.json").write_text(
            json.dumps(file_trees, indent=2, default=str), encoding="utf-8"
        )

        ok = sum(1 for r in summary if r["status"] == "success")
        print(f"\nSummary: {ok}/{len(summary)} runs succeeded")
        if keep_output:
            print(f"Artifacts retained at {ROOT / 'results.json'}")
            print(f"File trees retained at {ROOT / 'file_trees.json'}")
    finally:
        if not keep_output:
            cleanup_output_root()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="retain /tmp/drocat_lite_e2e artifacts for inspection after the run",
    )
    parser.add_argument(
        "--scenario",
        default="",
        help="comma-separated scenario names to run (default: all)",
    )
    parser.add_argument(
        "--variants",
        choices=["full", "quick"],
        default="full",
        help="'full': reciprocal x hemisphere matrix everywhere; "
             "'quick': 2 matrix variants for cross_dataset",
    )
    args = parser.parse_args()
    only = [n.strip() for n in args.scenario.split(",") if n.strip()] or None
    asyncio.run(main(keep_output=args.keep_output, only=only,
                     variants=args.variants))
