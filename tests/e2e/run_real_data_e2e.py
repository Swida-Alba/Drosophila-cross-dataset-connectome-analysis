#!/usr/bin/env python3
"""
Real-data end-to-end suite for DROCAT.

Drives the exact UI execution layer (ui.runner.ScriptRunner) - the same code
path triggered by clicking "Run" in the web UI - against live NeuPrint /
NeuronBridge data with the tokens in config_local.json / config.json.

Usage (from project root, inside the `drocat` conda env):
    python tests/e2e/run_real_data_e2e.py
    python tests/e2e/run_real_data_e2e.py --keep-output

The runner uses one known disposable sandbox at `/tmp/drocat_e2e`. It clears
that sandbox before every run and removes it afterward. Pass `--keep-output`
when inspecting artifacts after a run.
"""

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ui.runner import ScriptRunner  # noqa: E402

ROOT = Path("/tmp/drocat_e2e")


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


TOOLS = [
    {
        "name": "find_shortest",
        "tool_name": "find_shortest",
        "method": "find_shortest",
        "constructor_params": {
            "dataset": "male-cns:v0.9",
            "sourceNeurons": ["aMe12"],
            "targetNeurons": ["aMe10"],
            "output_dir": out("direct"),
            "min_synapse_num": 3,
            "min_ratio": 0.0,
            "min_traversal_probability": 0.0,
            "filter_by": "bodyId",
            "network_layout": "distributed",
            "use_cache": True,
            "edgeN_limit": 50,
            "output_format": "csv",
            "exclude_intra_type_connections": False,
            "separate_hemispheres": False,
            "keep_only_hemisphere_conserved_connections": False,
            "symmetry_analysis": False,
        },
        "timeout": 600,
    },
    {
        "name": "find_path",
        "tool_name": "find_path",
        "method": "find_all_path",
        "constructor_params": {
            "dataset": "male-cns:v0.9",
            "sourceNeurons": ["aMe12"],
            "targetNeurons": ["aMe10"],
            "output_dir": out("path"),
            "min_synapse_num": 3,
            "min_ratio": 0.0,
            "min_traversal_probability": 0.0,
            "max_interlayer": 2,
            "filter_by": "bodyId",
            "pathfinding": "Bidirectional",
            "network_layout": "distributed",
            "use_cache": True,
            "edgeN_limit": 500,
            "output_format": "csv",
            "skip_bodyId": True,
            "showfig": False,
            "custom_source_name": "",
            "custom_target_name": "",
            "keyword_in_path_to_remove": ["None"],
            "separate_hemispheres": False,
            "keep_only_hemisphere_conserved_connections": False,
            "symmetry_analysis": False,
            "find_reciprocal": False,
        },
        "timeout": 1500,
    },
    {
        "name": "find_network",
        "tool_name": "find_network",
        "method": "find_network",
        "constructor_params": {
            "dataset": "male-cns:v0.9",
            # FindNetwork uses the queried set as both source and target
            # (mutual direct connections == source == target).
            "sourceNeurons": ["aMe12", "aMe10", "aMe9"],
            "targetNeurons": ["aMe12", "aMe10", "aMe9"],
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
        },
        "timeout": 600,
    },
    {
        "name": "connectivity_profiling",
        "tool_name": "connectivity_profiling",
        "method": "run",
        "constructor_params": {
            "query": ["aMe12", "aMe10", "aMe9"],
            "dataset": "male-cns:v0.9",
            "output_dir": out("profiling"),
            "top_k": 15,
            "top_m": 5,
            "direction": "both",
            "generate_heatmaps": True,
            "verbose": True,
            "use_cache": True,
            "ensure_cache_complete": False,
        },
        "timeout": 1200,
    },
    {
        "name": "find_homologs",
        "tool_name": "find_homologs",
        "method": "find_homologs_fast",
        "constructor_params": {
            "source": "aMe12",
            "source_dataset": "male-cns:v0.9",
            "target_dataset": "hemibrain:v1.2.1",
            "output_dir": out("homologs"),
            "top_n": 10,
            "top_k": 15,
            "top_m": 5,
            "similarity_metric": "rank_union",
            "vector_prefiltering": True,
            "include_untyped_partners": True,
            "visualize_skeleton": False,
            "visualize_top_n": 5,
            "ensure_cache_complete": False,
        },
        "timeout": 1200,
    },
    {
        "name": "inter_dataset",
        "tool_name": "inter_dataset",
        "method": "run",
        "constructor_params": {
            "datasets": ["male-cns:v0.9", "hemibrain:v1.2.1"],
            "source_neurons": ["aMe12"],
            "target_neurons": ["aMe10"],
            "output_folder": out("comparison"),
            "comparison_mode": "path",
            "max_interlayer": 1,
            "thresholds": [3],
            "top_edges": 500,
            "pathfinding": "Bidirectional",
            "skip_bodyId": True,
            "cache_only": False,
            "auto_type_mapping": False,
            "separate_hemispheres": False,
            "keep_only_hemisphere_conserved_connections": False,
            "symmetry_analysis": True,
            "find_reciprocal": False,
        },
        "timeout": 1800,
    },
    {
        "name": "nb_find_lines",
        "tool_name": "nb_find_lines",
        "method": "find_lines",
        "constructor_params": {"verbose": True, "separate_splitgal4": True},
        "method_params": {
            "queries": "aMe12",
            "dataset": "male-cns:v0.9",
            "output_dir": out("nb_lines"),
            "match_type": "cds",
            "download_images": None,
            "download_img_for_top_n_lines": None,
            "summary_format": "pdf",
        },
        "timeout": 900,
    },
    {
        "name": "nb_find_neuron",
        "tool_name": "nb_find_neuron",
        "method": "find_neurons",
        "constructor_params": {"verbose": True},
        "method_params": {
            "line_names": ["VT037867"],
            "output_dir": out("nb_neurons"),
            "match_type": "cds",
            "top_n": 10,
            "visualize_top_n": 2,
            "generate_individual_profiles": None,
        },
        "timeout": 900,
    },
    {
        "name": "nb_colabel",
        "tool_name": "nb_colabel",
        "method": "colabel",
        "constructor_params": {"verbose": True},
        "method_params": {
            "lines": ["VT037867", "R10A06"],
            "output_dir": out("nb_colabel"),
            "similarity_methods": ["jaccard", "weighted_jaccard"],
            "generate_report": True,
            "visualize": True,
            "visualize_top_n": 0,
        },
        # Cold-cache colabeling downloads ~45 NeuronBridge CDN images; the
        # 2026-09-01 production run hit 93% when the old 900s cap fired.
        "timeout": 1800,
    },
    {
        "name": "plot3d_skeleton",
        "tool_name": "plot3d_skeleton",
        "method": "plot",
        "constructor_params": {
            "dataset": "male-cns:v0.9",
            "neuron_layers": ["aMe12"],
            "output_dir": out("skeleton"),
            "skeleton_mode": "tube",
            "brain_mesh": "native",
            "neuron_alpha": 0.2,
            "mesh_roi": [],
            "skip_synapse": True,
            "show_fig": False,
            "export_views": False,
            "background_color": "white",
            "export_scale": 3,
        },
        "timeout": 1200,
    },
]


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


def find_path_file() -> str:
    """Locate a FindAllPath result file for the PlotPath tool."""
    candidates = []
    for pattern in [
        "*_allpaths_info.csv",
        "*_allpaths_info.xlsx",
        "*_allpaths_type.csv",
        "*_allpaths_bodyId.csv",
    ]:
        for f in sorted((ROOT / "path").rglob(pattern)):
            candidates.append(f)
    if not candidates:
        return ""
    # Newest first (multiple runs create timestamped subfolders)
    return str(max(candidates, key=lambda f: f.stat().st_mtime))


def spec_output_dir(spec: dict) -> str:
    """Resolve the directory a tool writes into from its UI-style params."""
    for key in ("output_dir", "output_folder"):
        for params in (spec.get("constructor_params", {}), spec.get("method_params", {})):
            value = params.get(key)
            if value:
                return str(value)
    return out(spec["name"])


async def run_tool(spec: dict, runner: ScriptRunner) -> dict:
    name = spec["name"]
    log_path = ROOT / f"{name}.log"
    logs = []

    def log_cb(line: str, stream: str = "stdout"):
        logs.append(line)

    started = time.time()
    result = {"name": name, "status": "pending", "duration_s": 0.0, "error": None}
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
            "success" if res["returncode"] == 0 and not res["cancelled"] else "failed"
        )
        result["returncode"] = res["returncode"]
        result["cancelled"] = res["cancelled"]
        result["files"] = probe_files(spec_output_dir(spec))
    except asyncio.TimeoutError:
        runner.cancel()
        result["status"] = "timeout"
        result["error"] = f"exceeded {spec['timeout']}s"
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["duration_s"] = round(time.time() - started, 1)
        log_path.write_text("\n".join(logs), encoding="utf-8", errors="replace")
        result["log_lines"] = len(logs)
    return result


async def _run_all(keep_output: bool = False) -> None:
    summary = []
    # PlotPath runs after FindAllPath and consumes its output file.
    plot_path_spec = {
        "name": "plot_path",
        "tool_name": "plot_path",
        "method": "plot",
        "constructor_params": {},
        "timeout": 300,
    }

    for spec in TOOLS:
        runner = ScriptRunner()
        result = await run_tool(spec, runner)
        summary.append(result)
        print(
            f"[{result['status'].upper():8s}] {result['name']:24s} "
            f"{result['duration_s']:7.1f}s  "
            f"files={len(result.get('files', []))}"
        )

    # PlotPath: consume FindAllPath output
    path_file = find_path_file()
    if not path_file:
        summary.append(
            {"name": "plot_path", "status": "skipped", "error": "no FindAllPath output file found"}
        )
        print("[SKIPPED ] plot_path (no FindAllPath output)")
    else:
        plot_path_spec["constructor_params"] = {
            "path_file": path_file,
            "output_folder": out("plotpath"),
            "source_color": "#4A90E2",
            "intermediate_color": "#50E3C2",
            "target_color": "#B8E986",
            "link_color": "rgba(74,144,226,0.3)",
            "network_layout": "hierarchical",
            "showfig": False,
        }
        plot_path_spec["output_dir"] = out("plotpath")
        runner = ScriptRunner()
        result = await run_tool(plot_path_spec, runner)
        summary.append(result)
        print(
            f"[{result['status'].upper():8s}] plot_path            "
            f"{result['duration_s']:7.1f}s  "
            f"files={len(result.get('files', []))}"
        )

    (ROOT / "results.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    ok = sum(1 for r in summary if r["status"] == "success")
    print(f"\nSummary: {ok}/{len(summary)} tools succeeded")
    if keep_output:
        print(f"Artifacts retained at {ROOT / 'results.json'}")


async def main(keep_output: bool = False) -> None:
    reset_output_root()
    try:
        await _run_all(keep_output=keep_output)
    finally:
        if not keep_output:
            cleanup_output_root()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="retain /tmp/drocat_e2e artifacts for inspection after the run",
    )
    asyncio.run(main(keep_output=parser.parse_args().keep_output))
