"""
DROCAT UI Runner Engine
Executes tools by generating proper Python scripts with user parameters.
"""

import asyncio
import os
import re
import sys
import tempfile
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, List, Dict, Any

from .config import PROJECT_ROOT, SRC_DIR
from .output_guide import write_run_guide


# The PlotPath tool lives in the vispath subproject, imported as `vispath_pkg`.
# It is not installed into site-packages, so generated scripts must add
# vispath-subproject/src to sys.path themselves.
VISPATH_DIR = PROJECT_ROOT / "vispath-subproject" / "src"


# =============================================================================
# Output stream sanitizer
# =============================================================================

# ANSI escape sequences: CSI (colors, cursor moves, line erases), OSC
# (title/link), and single-character escapes. Stripped so the UI log shows
# clean text instead of raw escape codes.
_ANSI_ESCAPE_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b[@-Z\\-_]"
)

# tqdm-style progress bar line, e.g.
# "Building target profiles:  45%|████▍       | 1328/2972 [00:05<00:06, 252.17it/s]"
# tqdm bars with a total:   desc 100%|████| 1234/2635 [00:12<00:05, 210.0it/s]
# tqdm bars WITHOUT a total (unit counters, e.g. "Processing paths"):
#   desc: 13295222path [00:28, 735242.01path/s]
# Both are single-line refreshes that should update one log line in place.
_PROGRESS_LINE_RE = re.compile(
    r"^\s*(?:"
    r".*?\d+%\|.*\|\s*\d+/\d+\s*\[.*\]"   # with total (n/N + %)
    r"|"
    r".*?:\s*[\d,]+\s*/\s*[\d,]+\s+\([^)]*%\)\s*\[[^\]]*\]"  # LineProgress (n/N + %)
    r"|"
    r".*?:\s*\d+(?:\.\d+)?(?:path|it|file)s?\s*\[[^\]]*\]"  # unit counter
    r")\s*$"
)

# Emitted by generated FindNeuronConnection scripts immediately after
# InitializeNeuronInfo() has resolved the source and target query sets.
_NEURON_MATCH_RE = re.compile(
    r"^\[DROCAT\]\[neuron-match\]\s+source=(\d+)\s+target=(\d+)$"
)


class _OutputSplitter:
    """Incremental splitter turning raw subprocess output into clean log lines.

    Terminal-style output is normalized so the UI log stays readable:

    - ANSI escape sequences (colors, cursor moves, line erases) are stripped.
    - ``\r``-refreshed segments and cursor-up (``\x1b[A``) refreshes are
      yielded as *progress* items so the output panel can update one line
      in place instead of appending a new line per refresh.
    - Newline-terminated tqdm-style progress bars (non-TTY output) are also
      classified as progress.
    - Whitespace-only fragments (tqdm's line-clearing) and trailing spaces
      are dropped.

    ``feed`` consumes decoded chunks and yields ``(text, is_progress)`` pairs
    for every complete output segment; ``flush`` emits the final partial line.
    """

    def __init__(self):
        self._pending = ""

    def feed(self, chunk: str):
        self._pending += chunk
        # A cursor-up escape moves the cursor back to the previous progress
        # line; treat it like a carriage return (refresh separator).
        self._pending = self._pending.replace("\x1b[A", "\r")
        self._pending = _ANSI_ESCAPE_RE.sub("", self._pending)
        while True:
            nl = self._pending.find("\n")
            cr = self._pending.find("\r")
            if nl == -1 and cr == -1:
                break
            if nl != -1 and (cr == -1 or nl < cr):
                segment, self._pending = self._pending.split("\n", 1)
                refreshed = False
            else:
                segment, self._pending = self._pending.split("\r", 1)
                # A "\r\n" pair is a regular line ending (Windows), not an
                # in-place refresh; consume the "\n" so CRLF lines stay
                # ordinary log lines.
                if self._pending.startswith("\n"):
                    self._pending = self._pending[1:]
                    refreshed = False
                else:
                    refreshed = True
            yield from self._classify(segment, refreshed)

        # LineProgress deliberately emits a carriage-return-only update and
        # does not append a newline. If this is the first update (or the
        # update is the only data in the current pipe read), there is no next
        # delimiter yet, so the normal loop above would hold it until the
        # next refresh. Forward complete progress bars immediately so the UI
        # can show progress while the next pathfinding item is expensive.
        if self._pending.strip() and _PROGRESS_LINE_RE.match(self._pending):
            segment, self._pending = self._pending, ""
            yield from self._classify(segment, True)

    def flush(self):
        if self._pending.strip():
            yield from self._classify(self._pending, False)
        self._pending = ""

    @staticmethod
    def _classify(segment, refreshed):
        text = segment.rstrip()
        if not text.strip():
            # tqdm line-clearing residue (spaces) or blank lines.
            return
        is_progress = refreshed or bool(_PROGRESS_LINE_RE.match(text))
        yield (text, is_progress)


# =============================================================================
# Tool Registry: Maps tool names to their import paths and method calls
# =============================================================================
TOOL_REGISTRY: Dict[str, dict] = {
    "find_path": {
        "label": "Complete Paths",
        "import": "from coana import FindNeuronConnection",
        "class": "FindNeuronConnection",
        "var": "fc",
        "init_method": "InitializeNeuronInfo",
        "methods": {
            "find_path": "fc.FindPath()",
            # find_reciprocal defers to the constructor field (the method
            # parameter defaults to None); the old explicit
            # find_reciprocal=fc.find_reciprocal workaround is retired.
            "find_all_path": "fc.FindAllPath(forward_only=True)",
        },
    },
    "find_shortest": {
        "label": "Shortest Paths",
        "import": "from coana import FindNeuronConnection",
        "class": "FindNeuronConnection",
        "var": "fc",
        "init_method": "InitializeNeuronInfo",
        "methods": {
            "find_shortest": "fc.FindShortestPath(forward_only=True)",
        },
    },
    "find_network": {
        "label": "Find Network",
        "import": "from coana import FindNeuronConnection",
        "class": "FindNeuronConnection",
        "var": "fc",
        "init_method": "InitializeNeuronInfo",
        "methods": {
            "find_network": "fc.FindNetwork()",
        },
    },
    "connectivity_profiling": {
        "label": "Connectivity Profiling",
        "import": "from comparison.profile_comparator import ConnectivityProfileComparer",
        "class": "ConnectivityProfileComparer",
        "var": "comparer",
        "init_method": None,
        "methods": {
            "run": "comparer.run()",
        },
    },
    "find_homologs": {
        "label": "Homolog Finding",
        "import": "from comparison.profile_comparator import HomologFinder",
        "class": "HomologFinder",
        "var": "finder",
        "init_method": None,
        "methods": {
            "find_homologs_fast": "finder.find_homologs_fast()",
            "find_homologs": "finder.find_homologs()",
            "find_homologs_multi": "finder.find_homologs_multi(**method_params)",
        },
    },
    "inter_dataset": {
        "label": "Cross-Dataset Comparison",
        "import": "from comparison import ComparisonParameters, ComparisonAnalyzer",
        "class": "ComparisonAnalyzer",
        "var": "analyzer",
        "init_method": None,
        "methods": {
            "run": "analyzer.run()",
        },
        "wrapper": True,  # Needs special wrapper for ComparisonParameters
    },
    "nb_find_lines": {
        "label": "Find Driver Lines",
        "import": "from neuronbridge_finder import NeuronBridgeFinder",
        "class": "NeuronBridgeFinder",
        "var": "finder",
        "init_method": None,
        "methods": {
            "find_lines": "finder.find_lines_batch(**method_params)",
        },
    },
    "nb_find_lines_expanded": {
        "label": "Find Driver Lines (cross-dataset expansion)",
        "import": "from neuronbridge_query_expansion import ExpandedLineFinder",
        "class": "ExpandedLineFinder",
        "var": "finder",
        "init_method": None,
        "methods": {
            "find_lines": "finder.run(**method_params)",
        },
    },
    "nb_find_neuron": {
        "label": "Find EM Neurons",
        "import": "from neuronbridge_finder import NeuronBridgeFinder",
        "class": "NeuronBridgeFinder",
        "var": "finder",
        "init_method": None,
        "methods": {
            "find_neurons": "finder.find_neurons_batch(**method_params)",
        },
    },
    "nb_colabel": {
        "label": "Co-Labeling Analysis",
        "import": "from neuronbridge_finder import NeuronBridgeFinder",
        "class": "NeuronBridgeFinder",
        "var": "finder",
        "init_method": None,
        "methods": {
            "colabel": "finder.analyze_colabeling(**method_params)",
        },
    },
    "flylight_download": {
        "label": "FlyLight Image Download",
        "import": "from flylight_downloader import FlyLightDownloader",
        "class": "FlyLightDownloader",
        "var": "downloader",
        "init_method": None,
        "methods": {
            "download": "downloader.download(**method_params)",
        },
    },
    "plot3d_skeleton": {
        "label": "3D Skeleton Visualization",
        "import": "from visualize_skeleton import VisualizeSkeleton",
        "class": "VisualizeSkeleton",
        "var": "vs",
        "init_method": None,
        "methods": {
            "plot": "vs.plot_neurons()",
        },
    },
    "plot3d_reexport": {
        "label": "Skeleton Re-export",
        # No tool class is instantiated: both re-exporters are module-level
        # functions that read the figure back out of a stored page, so
        # _generate_script dispatches to _generate_skeleton_reexport_script
        # before any of the keys below are used.
        "import": (
            "from visualize_skeleton import (export_individuals_from_html, "
            "export_video_from_html, reexport_output_dir)"
        ),
        "class": None,
        "var": None,
        "init_method": None,
        "methods": {"reexport": ""},
    },
    "plot_path": {
        "label": "Path Network Visualization",
        "import": "from vispath_pkg import VisualizePath",
        "class": "VisualizePath",
        "var": "vp",
        "init_method": None,
        "methods": {
            "plot": "vp.visualize()",
        },
    },
    "find_similar_morphology": {
        "label": "Morphological Similarity",
        "import": "from morphology import MorphologyComparer",
        "class": "MorphologyComparer",
        "var": "comparer",
        "init_method": None,
        "methods": {
            "find_similar": "comparer.find_similar()",
        },
    },
    "morphology_comparison": {
        "label": "Morphology Comparison",
        "import": "from morphology_comparison import MorphologyProfileComparer",
        "class": "MorphologyProfileComparer",
        "var": "comparer",
        "init_method": None,
        "methods": {
            "run": "comparer.run()",
        },
    },
    "morph_cross_dataset": {
        "label": "Cross-Dataset Morphology Comparison",
        "import": "from comparison.morph_cross_dataset import CrossDatasetMorphComparer",
        "class": "CrossDatasetMorphComparer",
        "var": "comparer",
        "init_method": None,
        "methods": {
            "run": "comparer.run()",
        },
    },
    "type_mapping_validation": {
        "label": "Type Validation (TM VEV)",
        "import": ("from comparison.mapping_validation import "
                   "MappingValidationConfig, MappingValidator"),
        "class": None, "var": None, "init_method": None,
        "methods": {"run": ""},  # dispatched to _generate_tmvev_script by name
    },
    # NOTE: the pair report is NOT a registry tool — it is generated
    # automatically after every pathfinding run by
    # _maybe_generate_pair_report (see the post-run hook in run()).
}


def _format_value(value: Any) -> str:
    """Format a Python value for code generation (repr round-trips exactly)."""
    return repr(value)


def _format_params(params: dict) -> str:
    """Format a params dict into constructor argument string."""
    lines = []
    for key, value in params.items():
        lines.append(f"    {key}={_format_value(value)},")
    return "\n".join(lines)


def _maybe_generate_pair_report(scan_dir, log, progress) -> None:
    """Post-run hook: generate the per-pair paths report into *scan_dir*
    when it holds pathfinding outputs (plan-paths-pair-report §7, round 6:
    embedded in the run flow instead of a standalone button). Applies to
    Complete Paths, Shortest Paths, and cross-dataset runs; skips
    everything else silently. Never raises — a report failure must not
    fail the run."""
    try:
        from src.paths_pair_report import (
            generate_paths_pair_report,
            has_paths_tables,
        )
    except ImportError:  # pragma: no cover - direct src/ execution
        try:
            from paths_pair_report import (
                generate_paths_pair_report,
                has_paths_tables,
            )
        except ImportError:
            return
    try:
        if not scan_dir or not has_paths_tables(scan_dir):
            return
        progress("pair-report", "Generating pair report")
        log("Generating paths pair report...", "system")
        generate_paths_pair_report(scan_dir, log=lambda *args, **kwargs: None)
        log("Pair report written: path_report.html", "system")
    except Exception as exc:  # noqa: BLE001 - reporting must never fail the run
        try:
            log(f"Pair report generation skipped: {exc}", "warning")
        except Exception:
            pass


class ScriptRunner:
    """Runs DROCAT tools as subprocesses with real-time output streaming."""

    def __init__(self):
        self.process: Optional[asyncio.subprocess.Process] = None
        self.is_running = False
        self._cancelled = False
        self._run_logs: List[tuple] = []
        self._neuron_match = None

    async def run(
        self,
        tool_name: str,
        constructor_params: dict,
        method_name: str = "run",
        method_params: Optional[dict] = None,
        log_callback: Optional[Callable[[str, str], None]] = None,
        progress_callback: Optional[Callable[[str, str], None]] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Run a tool with given parameters.

        Args:
            tool_name: Key in TOOL_REGISTRY
            constructor_params: Parameters for the class constructor
            method_name: Key in tool's 'methods' dict
            method_params: Parameters for the method call (if any)
            log_callback: Callback(line, stream) for log output
            progress_callback: Callback(phase, label) for coarse lifecycle
                progress. Structured backend progress still travels through
                ``log_callback``.
            output_dir: Expected output directory to scan for results

        Returns:
            dict with 'returncode', 'files', 'duration'
        """
        if tool_name not in TOOL_REGISTRY:
            msg = f"Unknown tool: {tool_name}"
            if log_callback:
                log_callback(msg, "error")
            return {"returncode": -1, "files": [], "duration": 0, "cancelled": False}

        tool = TOOL_REGISTRY[tool_name]
        self.is_running = True
        self._cancelled = False
        self._run_logs = []
        self._neuron_match = None
        start_time = datetime.now()

        def _progress(phase: str, label: str = "") -> None:
            if progress_callback:
                try:
                    progress_callback(phase, label)
                except Exception:
                    # A page may have been closed while the subprocess keeps
                    # unwinding; progress reporting must never fail the run.
                    pass

        def _log(line: str, level: str = "stdout"):
            """Record every log line and forward it to the UI callback."""
            match = _NEURON_MATCH_RE.match(str(line).strip())
            if match:
                source_count = int(match.group(1))
                target_count = int(match.group(2))
                if self._neuron_match is None:
                    self._neuron_match = {
                        "source": source_count,
                        "target": target_count,
                        "any_pair": source_count > 0 and target_count > 0,
                    }
                else:
                    self._neuron_match["source"] = max(
                        self._neuron_match["source"], source_count)
                    self._neuron_match["target"] = max(
                        self._neuron_match["target"], target_count)
                    self._neuron_match["any_pair"] = (
                        self._neuron_match["any_pair"]
                        or (source_count > 0 and target_count > 0)
                    )
                # The generated coana scripts emit this marker immediately
                # after InitializeNeuronInfo().  Only now is it correct to
                # advance initialized tabs from step 1 to their execution
                # step; the subprocess may still be printing initialization
                # messages when it first starts.
                _progress("initialize_complete", "Initialization complete")
                return
            self._run_logs.append((level, line))
            if log_callback:
                log_callback(line, level)

        # Generate the script
        _progress("prepare", "Preparing inputs")
        script_content = self._generate_script(
            tool_name, constructor_params, method_name, method_params
        )

        # Write to temp file
        fd, temp_script = tempfile.mkstemp(suffix=".py", prefix="drocat_run_")
        with os.fdopen(fd, "w") as f:
            f.write(script_content)

        try:
            label = tool.get("label", tool_name)
            _log("", "system")
            _log(f"━━━ ▶ UI FUNCTION: {label}  ({tool_name}) ━━━", "system")
            _log(
                "Output streams live below while the function runs; "
                "generated files appear in Output Files when it finishes.",
                "system",
            )
            _log("", "system")
            # Log params as multi-line for readability
            _log("[DROCAT] Parameters:", "system")
            for k, v in constructor_params.items():
                _log(f"  {k}: {v}", "system")
            if method_params:
                for k, v in method_params.items():
                    _log(f"  {k}: {v}", "system")

            # Get Python executable
            python_exe = sys.executable or "python"

            # Set up environment
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + str(PROJECT_ROOT)
            # Run unbuffered so log lines stream to the UI in real time
            env["PYTHONUNBUFFERED"] = "1"
            # Force UTF-8 for the child's stdout/stderr pipes: on non-UTF-8
            # Windows locales (e.g. zh-CN/GBK) the child would otherwise emit
            # cp936 bytes, which _stream_output decodes as UTF-8, turning any
            # non-ASCII output (tqdm bars, ✓, ∩, …) into replacement chars.
            env["PYTHONIOENCODING"] = "utf-8"

            # Create subprocess.  Keep the tracker on its initialization step
            # until the generated script confirms that initialization ended.
            _progress("initialize", f"Initializing {label}")
            self.process = await asyncio.create_subprocess_exec(
                python_exe,
                "-u",
                temp_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(PROJECT_ROOT),
                env=env,
            )

            # Stream output
            await self._stream_output(_log)

            # Wait for completion
            returncode = await self.process.wait()

            duration = (datetime.now() - start_time).total_seconds()

            if self._cancelled:
                _log("[DROCAT] Execution cancelled by user.", "error")
            elif returncode == 0:
                _log(
                    f"━━━ ■ FINISHED: {tool.get('label', tool_name)} — "
                    f"completed in {duration:.1f}s ━━━",
                    "success",
                )
            else:
                _log(
                    f"━━━ ■ FINISHED: {tool.get('label', tool_name)} — "
                    f"failed with return code {returncode} ━━━",
                    "error",
                )

            # Scan only the folder THIS run actually generated. A run that
            # fails before creating its per-run folder must not fall back to
            # the shared storage root — that would list files from previous
            # runs (e.g. an earlier BANC run while running male-cns).
            _progress("collect", "Collecting output files")
            scan_dir = self._resolve_scan_dir(output_dir)

            # Record the used output root for the Settings → Storage
            # scanner, so every location the user exports to shows up in
            # storage management automatically. Best-effort only.
            if returncode == 0 and not self._cancelled and output_dir:
                try:
                    from .config import (
                        add_storage_scan_root,
                        get_default_output_dir,
                    )
                    if str(Path(output_dir).resolve()) != str(
                            Path(get_default_output_dir()).resolve()):
                        add_storage_scan_root(output_dir)
                except Exception:
                    pass

            # Export the per-run user guide into the run folder (Settings →
            # Run Guide Format; never fails the run itself). Written before
            # the file scan so it appears in the Output Files panel.
            if returncode == 0 and not self._cancelled and scan_dir:
                guide_path = write_run_guide(
                    scan_dir,
                    tool_name,
                    params={**(constructor_params or {}),
                            **(method_params or {})},
                )
                if guide_path:
                    _log(f"Run guide written: {guide_path.name}", "system")

            # Embed the per-pair paths report into pathfinding runs
            # (plan-paths-pair-report §7 round 6: part of the run flow, not
            # a standalone button). Written before the file scan so
            # path_report.html + paths_pair_breakdown/ appear in the Output
            # Files panel like native run outputs. Never fails the run.
            if returncode == 0 and not self._cancelled and scan_dir:
                _maybe_generate_pair_report(
                    scan_dir, _log, _progress)

            files = self._scan_output_files(scan_dir) if scan_dir else []

            return {
                "returncode": returncode,
                "files": files,
                "duration": duration,
                "cancelled": self._cancelled,
                "output_folder": scan_dir,
                "neuron_match": self._neuron_match,
            }

        except Exception as e:
            _log(f"[DROCAT] Error: {str(e)}", "error")
            return {
                "returncode": -1,
                "files": [],
                "duration": 0,
                "cancelled": False,
                "output_folder": None,
                "neuron_match": self._neuron_match,
            }
        finally:
            self.is_running = False
            self.process = None
            # Clean up temp file
            try:
                os.unlink(temp_script)
            except OSError:
                pass

    def _generate_script(
        self,
        tool_name: str,
        constructor_params: dict,
        method_name: str,
        method_params: Optional[dict],
    ) -> str:
        """Generate a Python script that runs the tool with given parameters."""
        tool = TOOL_REGISTRY[tool_name]
        import_stmt = tool["import"]
        class_name = tool["class"]
        var = tool["var"]
        init_method = tool.get("init_method")
        method_call = tool["methods"].get(method_name, "")

        # Handle special wrapper cases
        if tool_name == "inter_dataset":
            return self._generate_inter_dataset_script(constructor_params, method_params)
        elif tool_name in ("nb_find_lines", "nb_find_neuron", "nb_colabel"):
            return self._generate_neuronbridge_script(tool_name, constructor_params, method_params)
        elif tool_name == "flylight_download":
            return self._generate_flylight_script(constructor_params, method_params)
        elif tool_name == "plot3d_skeleton":
            return self._generate_plot3d_script(constructor_params, method_params)
        elif tool_name == "plot3d_reexport":
            return self._generate_skeleton_reexport_script(
                constructor_params, method_params)
        elif tool_name == "type_mapping_validation":
            return self._generate_tmvev_script(
                constructor_params, method_params)

        # Standard script generation
        if tool_name == "plot_path":
            # The standalone Net-Viz run owns its 4-step protocol; nested
            # VisualizePath calls inside other pipelines omit progress_total
            # and stay silent so the outer tool's protocol owns the bar.
            constructor_params = dict(constructor_params)
            constructor_params["progress_total"] = 4
        if tool_name in {"find_path", "find_shortest", "find_network"}:
            # coana step events are opt-in (progress_events defaults False)
            # so nested FindNeuronConnection runs inside other pipelines stay
            # silent; the tab's own run owns the 5-step protocol.
            constructor_params = dict(constructor_params)
            constructor_params["progress_events"] = True
        params_str = _format_params(constructor_params)

        init_call = ""
        if init_method:
            init_call = f"{var}.{init_method}()\n"
            init_call += f'''_source_df = getattr({var}, "source_df", None)
_target_df = getattr({var}, "target_df", None)
print(
    f"[DROCAT][neuron-match] source={{len(_source_df) if _source_df is not None else 0}} "
    f"target={{len(_target_df) if _target_df is not None else 0}}",
    flush=True,
)
'''

        # A tool method string may pass method-level kwargs via
        # ``**method_params`` (e.g. find_homologs_multi(use_fast=...)), so the
        # generated script always defines the variable in scope, defaulting to
        # an empty dict when the caller supplied none.
        mp_value = _format_value(method_params or {})

        script = f'''#!/usr/bin/env python
"""Auto-generated DROCAT runner script for {tool_name}."""
import sys
import warnings
from pathlib import Path

# Add project paths
sys.path.insert(0, r"{SRC_DIR}")
sys.path.insert(0, r"{PROJECT_ROOT}")
sys.path.insert(0, r"{VISPATH_DIR}")

warnings.filterwarnings("ignore")

# Import the tool class
{import_stmt}

# Create instance with user parameters
{var} = {class_name}(
{params_str}
)

# Initialize (if needed)
{init_call}method_params = {mp_value}
# Run the method
{method_call}

print("[DROCAT] Done.")
'''
        return script

    def _generate_plot3d_script(
        self, constructor_params: dict, method_params: Optional[dict]
    ) -> str:
        """Generate script for the 3D skeleton tool, including optional
        individual-profile PDF/PPTX export and rotating-video export.

        The optional export phases extend the run's [DROCAT][progress]
        protocol: ``plot_neurons`` reports its three phases against the
        combined total (passed as ``progress_total``), and the script emits
        one step event per enabled export phase before calling it.
        """
        params_str = _format_params(constructor_params)
        mp = method_params or {}

        export_individuals = bool(mp.get("export_individual_profiles"))
        export_video = bool(mp.get("export_video"))
        total_steps = 3 + int(export_individuals) + int(export_video)
        # _format_params already comma-terminates its last line.
        params_str += f"\n    progress_total={total_steps},"
        # The exported HTML viewers pin their scene axes so legend show/hide
        # cannot rescale the 3D framing; the Skeleton tab sends the switch
        # value, and every other caller keeps the VisualizeSkeleton default
        # (True) spelled out here.
        if "freeze_view" not in constructor_params:
            params_str += f"\n    freeze_view={_format_value(True)},"

        extra_calls = ""
        if export_individuals:
            step = 4
            extra_calls += f"""
print(f"[DROCAT][progress] {step}/{total_steps} Exporting individual profiles", flush=True)
# Export individual profiles (PDF/PPTX)
vs.plot_individuals(
    pdf_images_per_page={_format_value(mp.get('pdf_images_per_page', (3, 2)))},
    views={_format_value(mp.get('views', ['front']))},
    summary_format={_format_value(mp.get('summary_format', ['pdf']))},
    granularity={_format_value(mp.get('granularity', 'legend'))},
)
"""
        if export_video:
            step = 4 + int(export_individuals)
            extra_calls += f"""
print(f"[DROCAT][progress] {step}/{total_steps} Exporting rotating video", flush=True)
# Export rotating video / GIF
vs.export_video(
    fps={_format_value(mp.get('fps', 30))},
    degree_per_frame={_format_value(mp.get('degree_per_frame', 1.0))},
    rotate={_format_value(mp.get('rotate', 'horizontal'))},
    export_gif={_format_value(mp.get('export_gif', True))},
    gif_scale={_format_value(mp.get('gif_scale', 0.2))},
)
"""

        script = f'''#!/usr/bin/env python
"""Auto-generated DROCAT runner script for plot3d_skeleton."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, r"{SRC_DIR}")
sys.path.insert(0, r"{PROJECT_ROOT}")
sys.path.insert(0, r"{VISPATH_DIR}")

warnings.filterwarnings("ignore")

from visualize_skeleton import VisualizeSkeleton

vs = VisualizeSkeleton(
{params_str}
)

vs.plot_neurons()
{extra_calls}
print("[DROCAT] Done.")
'''
        return script

    def _generate_skeleton_reexport_script(
        self, constructor_params: dict, method_params: Optional[dict]
    ) -> str:
        """Generate the script that re-exports from an already-stored page.

        No tool class is constructed: the two module-level re-exporters read
        the figure back out of the page, so the run needs no dataset, cache or
        client. Artifacts land beside the source page in its own
        ``<stem>_reexport/`` folder — announced through the standard output
        marker — and one ``[DROCAT][progress]`` line is emitted per enabled
        phase so the panel's step bar matches the work actually done.
        """
        mp = method_params or {}
        export_profiles = bool(mp.get("export_individual_profiles"))
        export_video = bool(mp.get("export_video"))
        total_steps = int(export_profiles) + int(export_video)

        calls = ""
        step = 0
        if export_profiles:
            step += 1
            calls += f'''print("[DROCAT][progress] {step}/{total_steps} Export individual profiles", flush=True)
_profiles = export_individuals_from_html(
    html_file,
    output_dir=os.path.join(output_dir, "individual_profiles"),
    granularity={_format_value(mp.get("granularity", "legend"))},
    views={_format_value(mp.get("views", ["front"]))},
    scale={_format_value(int(mp.get("export_scale", 3)))},
    auto_crop={_format_value(bool(mp.get("auto_crop", True)))},
)
# The re-exporter reports failure in its result dict instead of raising, so a
# failed phase has to set the exit code or the panel reads it as Completed.
if not _profiles.get("success"):
    print(f"[DROCAT] Profile export failed: {{_profiles.get('error')}}",
          flush=True)
    sys.exit(1)
'''
        if export_video:
            step += 1
            calls += f'''print("[DROCAT][progress] {step}/{total_steps} Export rotating video", flush=True)
_video_rc = export_video_from_html(
    html_file,
    output_dir=output_dir,
    fps={_format_value(int(mp.get("fps", 30)))},
    degree_per_frame={_format_value(mp.get("degree_per_frame", 2.0))},
    rotate={_format_value(mp.get("rotate", "horizontal"))},
    export_gif={_format_value(bool(mp.get("export_gif", True)))},
    gif_scale={_format_value(mp.get("gif_scale", 0.2))},
    auto_crop={_format_value(bool(mp.get("auto_crop", True)))},
    export_method={_format_value(mp.get("export_method", "webdriver"))},
    scale={_format_value(int(mp.get("export_scale", 3)))},
)
if _video_rc:
    sys.exit(_video_rc)
'''

        return f'''#!/usr/bin/env python
"""Auto-generated DROCAT runner script for plot3d_reexport."""
import os
import sys
import warnings

sys.path.insert(0, r"{SRC_DIR}")
sys.path.insert(0, r"{PROJECT_ROOT}")
sys.path.insert(0, r"{VISPATH_DIR}")

warnings.filterwarnings("ignore")

from visualize_skeleton import (
    export_individuals_from_html,
    export_video_from_html,
    reexport_output_dir,
)

# The stored page is the only input; a run folder or a page both resolve to
# the canonical viewer HTML, and its manifest supplies the cameras.
html_file = {_format_value(constructor_params.get("html_file"))}
output_dir = reexport_output_dir(html_file)
os.makedirs(output_dir, exist_ok=True)
print(f"[DROCAT] Output will be saved to: {{output_dir}}", flush=True)
{calls}
print("[DROCAT] Done.")
'''

    def _generate_inter_dataset_script(
        self, constructor_params: dict, method_params: Optional[dict]
    ) -> str:
        """Generate script for InterDatasetComparator."""
        params_str = _format_params(constructor_params)
        script = f'''#!/usr/bin/env python
"""Auto-generated DROCAT runner script for inter-dataset comparison."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, r"{SRC_DIR}")
sys.path.insert(0, r"{PROJECT_ROOT}")
sys.path.insert(0, r"{VISPATH_DIR}")

warnings.filterwarnings("ignore")

from comparison import ComparisonParameters, ComparisonAnalyzer

params = ComparisonParameters(
{params_str}
)

analyzer = ComparisonAnalyzer(params, verbose=True)
analyzer.run_comparison()
analyzer.export_results()

print("[DROCAT] Done.")
'''
        return script

    def _generate_tmvev_script(
        self, constructor_params: dict, method_params: Optional[dict]
    ) -> str:
        """Generate the script that runs the TM VEV pipeline.

        ``MappingValidationConfig`` is a plain dataclass, so ``constructor_params``
        keys must be real field names (None/absent keys are dropped so the
        dataclass defaults stay authoritative) and an unknown key raises TypeError.

        The pipeline emits no ``[DROCAT]`` markers of any kind, so the generated
        script installs a ``log`` wrapper on the validator — the single funnel
        every ``[stage …]``/``[categories]``/``[TMVEV]`` banner passes through. It
        (a) announces the run folder the first time a banner fires (``run_dir`` is
        set before ``[stage 1]``), so the Output Files panel is populated during
        the run and survives a SIGTERM cancel; and (b) maps the verified banner
        literals to ``[DROCAT][progress] i/n label`` step events. The wrapper calls
        through to the real ``log`` (the sole producer of ``notes``/README/report),
        accepts one positional arg, returns None, and never raises.
        """
        cp = {k: v for k, v in (constructor_params or {}).items() if v is not None}

        # Banner literals are matched as exact substrings (never prefixes):
        # [stage 5]/[stage 5d] share a stem, [categories] failed /
        # out-map-expansion no-work lines reuse their success prefix.
        # The (literal, label) table lives in
        # ui.components.page_progress.tmvev_progress_steps so the generated
        # bridge and the results-panel checklist cannot drift apart.
        from .components.page_progress import tmvev_progress_steps

        matches = [
            (lit, i + 1, label)
            for i, (lit, label) in enumerate(tmvev_progress_steps(cp))
        ]

        # A plain (non-f) template keeps every literal ``{``/``}`` in the child
        # code intact; values are spliced in via replace(), so no brace escaping.
        template = '''#!/usr/bin/env python
"""Auto-generated DROCAT runner script for type-mapping validation (TM VEV)."""
import sys
import re
import warnings

sys.path.insert(0, r"@@SRC@@")
sys.path.insert(0, r"@@ROOT@@")
sys.path.insert(0, r"@@VISPATH@@")

warnings.filterwarnings("ignore")

from comparison.mapping_validation import MappingValidationConfig, MappingValidator

_total = @@TOTAL@@
_outmap_re = re.compile(r"out-map expansion: \\d+/\\d+")
_MATCHES = @@MATCHES@@
_state = {"step": 0, "announced": False}


def _announce(rd):
    if rd is None or _state["announced"]:
        return
    _state["announced"] = True
    print(f"[DROCAT] Output will be saved to: {rd}", flush=True)


def _bridge(_orig):
    def _log(msg=''):
        try:
            _orig(msg)
        except Exception:
            pass
        try:
            s = str(msg)
            rd = getattr(validator, "run_dir", None)
            if rd is not None:
                _announce(rd)
            for lit, i, label in _MATCHES:
                if lit in s:
                    if lit == "[TMVEV] out-map expansion: " and not _outmap_re.search(s):
                        continue
                    if i > _state["step"]:
                        _state["step"] = i
                        print(f"[DROCAT][progress] {i}/{_total} {label}", flush=True)
                    break
        except Exception:
            pass
    return _log


cfg = MappingValidationConfig(
@@PARAMS@@
)
validator = MappingValidator(cfg)
validator.log = _bridge(validator.log)
run_dir = validator.run()
_announce(run_dir)
print("[DROCAT] Done.")
'''
        return (
            template
            .replace("@@SRC@@", str(SRC_DIR))
            .replace("@@ROOT@@", str(PROJECT_ROOT))
            .replace("@@VISPATH@@", str(VISPATH_DIR))
            .replace("@@TOTAL@@", str(len(matches)))
            .replace("@@MATCHES@@", repr(matches))
            .replace("@@PARAMS@@", _format_params(cp))
        )

    def _generate_flylight_script(
        self, constructor_params: dict, method_params: Optional[dict]
    ) -> str:
        """Generate script for the FlyLight downloader."""
        init_params_str = _format_params(constructor_params)
        method_params_str = _format_params(method_params or {})

        script = f'''#!/usr/bin/env python
"""Auto-generated DROCAT runner script for flylight_download."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, r"{SRC_DIR}")
sys.path.insert(0, r"{PROJECT_ROOT}")
sys.path.insert(0, r"{VISPATH_DIR}")

warnings.filterwarnings("ignore")

from flylight_downloader import FlyLightDownloader

downloader = FlyLightDownloader(
{init_params_str}
)

downloader.download(
{method_params_str}
)

print("[DROCAT] Done.")
'''
        return script

    def _generate_neuronbridge_script(
        self, tool_name: str, constructor_params: dict, method_params: Optional[dict]
    ) -> str:
        """Generate script for NeuronBridge tools."""
        init_params_str = _format_params(constructor_params)
        method_params_str = _format_params(method_params or {})

        tool = TOOL_REGISTRY[tool_name]
        method_name = "find_lines_batch" if tool_name == "nb_find_lines" else \
                       "find_neurons_batch" if tool_name == "nb_find_neuron" else "analyze_colabeling"

        script = f'''#!/usr/bin/env python
"""Auto-generated DROCAT runner script for {tool_name}."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, r"{SRC_DIR}")
sys.path.insert(0, r"{PROJECT_ROOT}")
sys.path.insert(0, r"{VISPATH_DIR}")

warnings.filterwarnings("ignore")

from neuronbridge_finder import NeuronBridgeFinder

finder = NeuronBridgeFinder(
{init_params_str}
)

finder.{method_name}(
{method_params_str}
)

print("[DROCAT] Done.")
'''
        return script

    async def _stream_output(self, log_callback: Callable[[str, str], None]):
        """
        Stream stdout/stderr to the UI in real time.

        Reads chunks as soon as they arrive (unbuffered subprocess) and emits
        sanitized, complete lines immediately. tqdm-style progress bars and
        carriage-return refreshes are forwarded as 'progress' lines so the
        output panel updates one line in place instead of spamming a new line
        per update (see _OutputSplitter).
        """
        if not self.process:
            return

        async def read_stream(stream, stream_name):
            splitter = _OutputSplitter()
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                for text, is_progress in splitter.feed(
                    chunk.decode("utf-8", errors="replace")
                ):
                    log_callback(text, "progress" if is_progress else stream_name)
            for text, is_progress in splitter.flush():
                log_callback(text, "progress" if is_progress else stream_name)

        # Read both streams concurrently
        await asyncio.gather(
            read_stream(self.process.stdout, "stdout"),
            read_stream(self.process.stderr, "stderr"),
        )

    def _scan_output_files(self, output_dir: str) -> List[dict]:
        """Scan output directory for recently created files.

        Returns EVERY file in the run folder (no count cap): the panel
        mirrors the folder structure, and a cap would make files silently
        disappear from the UI while a run keeps writing (e.g. hundreds of
        downloaded images). The 24h recency filter only guards against
        re-showing files from an older run in the same folder.
        """
        files = []
        output_path = Path(output_dir)
        if not output_path.exists():
            return files

        # Get files modified in the last 24h
        import time
        cutoff = time.time() - 24 * 3600

        for f in output_path.rglob("*"):
            try:
                if f.is_file() and f.stat().st_mtime > cutoff:
                    files.append({
                        "name": f.name,
                        "path": str(f.absolute()),
                        "size": f.stat().st_size,
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    })
            except (OSError, PermissionError):
                continue

        # Sort by modification time (newest first)
        files.sort(key=lambda x: x["modified"], reverse=True)
        return files

    OUTPUT_FOLDER_MARKERS = [
        "Created output folder: ",
        "data will be saved in: ",
        "Output files in: ",
        "Output folder: ",
        "📁 Output folder: ",
        "📁 Output: ",
        # ConnectivityProfileComparer prefixes its final run path with the
        # plain ``Output:`` label.  Keep this marker path-only so the output
        # panel can resolve the current profiling folder from the log.
        "Output: ",
        "Saving results to: ",
        "Results saved to: ",
        "Output will be saved to: ",
    ]

    # Per-run output folder prefixes used by every DROCAT tool. The table
    # lives in src/utils/naming_utils.py (next to the naming scheme it
    # documents) so the Settings → Storage classifier and this scan-dir
    # guard can never drift apart. A directory whose name does not start
    # with one of these is a shared storage root that may contain many
    # runs — it must never be scanned as the current run's folder.
    try:
        from src.utils.naming_utils import (
            RUN_FOLDER_PREFIX_RE as _RUN_FOLDER_PREFIX_RE,
        )
    except ImportError:  # src not importable standalone; keep the local copy
        _RUN_FOLDER_PREFIX_RE = re.compile(
            r"^(find-paths-complete|find-paths-shortest|find-network|cross-dataset|plot-3d|plot-network|"
            r"homologs|similar-morphology|similar-connectivity|similar|profiling|morphology_comparison|morph_cross|NB-find-lines-expanded|NB-find-lines|NB-find-neurons|NB-colabeling|flylight-downloads|flylignt-downloads|"
            r"findpath|findallpath|findshortestpath|findnetwork|finddirect|findhomologs|interdataset|"
            r"plot3d|plotpath|colabel|findlines|findneuron|findsimilar)_"
        )

    def _resolve_scan_dir(self, output_dir: Optional[str] = None) -> Optional[str]:
        """Return the folder whose files belong to the current run.

        The per-run folder announced by the backend wins. Otherwise the
        caller-provided directory is used only when it is itself a per-run
        folder (plot_path pre-creates its own folder before the run). A
        shared storage root holding many runs is never scanned — its files
        belong to previous runs.
        """
        run_folder = self._extract_output_folder(output_dir)
        if run_folder:
            return run_folder
        if output_dir and self._RUN_FOLDER_PREFIX_RE.match(Path(output_dir).name):
            return output_dir
        return None

    def _extract_output_folder(self, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Return the folder the current run actually generated.

        Backend functions log where they create their per-run output folder
        (e.g. "Created output folder: ..."). The last logged path that exists
        and lies under the requested output directory wins, so the results
        panel links to the current run instead of older runs in the same
        storage directory.
        """
        candidates = []
        for _level, line in self._run_logs:
            for marker in self.OUTPUT_FOLDER_MARKERS:
                if marker in line:
                    candidate = line.split(marker, 1)[1].strip().strip("`").strip()
                    if candidate and os.path.isdir(candidate):
                        candidates.append(candidate)
                    break
        if not candidates:
            return None
        if output_dir:
            base = Path(output_dir).expanduser().resolve()
            # Prefer the newest top-level run folder (a direct child of the
            # requested output dir); avoid nested folders like
            # bodyId_visualization that internal tools also log.
            direct_children = [
                c for c in candidates
                if Path(c).expanduser().resolve().parent == base
            ]
            for candidate in reversed(direct_children or candidates):
                resolved = Path(candidate).expanduser().resolve()
                try:
                    resolved.relative_to(base)
                except ValueError:
                    continue
                # Return the logged spelling (important on macOS where
                # /var resolves to /private/var), after validating it safely.
                return candidate
            return None
        return candidates[-1]

    def cancel(self):
        """Cancel the running process."""
        self._cancelled = True
        if self.process:
            try:
                if platform.system() == "Windows":
                    self.process.kill()
                else:
                    import signal
                    self.process.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass


def open_folder(path: str):
    """Open a folder in the system file manager."""
    path = Path(path)
    if not path.exists():
        return

    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", str(path)])
        elif system == "Windows":
            subprocess.run(["explorer", str(path)])
        else:
            subprocess.run(["xdg-open", str(path)])
    except Exception as exc:
        # A silent no-op leaves the user clicking a button that appears
        # broken; say what failed instead.
        from nicegui import ui
        ui.notify(f"Could not open folder {path}: {exc}", type="negative")


def open_file(path: str):
    """Open a file with the default application."""
    path = Path(path)
    if not path.exists():
        return

    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", str(path)])
        elif system == "Windows":
            os.startfile(str(path))
        else:
            subprocess.run(["xdg-open", str(path)])
    except Exception as exc:
        from nicegui import ui
        ui.notify(f"Could not open file {path}: {exc}", type="negative")


# NOTE: the old tkinter-based pick_directory/pick_file helpers were removed.
# They blocked the whole web server until the native dialog was dismissed
# (and could hang when the window never appeared, freezing the app).
# Directory browsing is now an in-browser dialog (dir_browser_dialog in
# ui/components/common.py); file picking uses NiceGUI's ui.upload.
