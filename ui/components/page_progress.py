"""Compact progress tracker shared by all executable tool tabs.

The execution log is useful for detail, but it is a poor place to answer the
simple question "where is this run now?"  This component sits in the output
card directly above the execution log and shows one refreshing current-step
text plus the overall progress bar.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from nicegui import ui

from src.flywire_ids import is_local_connectome_dataset


DEFAULT_PROGRESS_STEPS = (
    "Prepare inputs",
    "Run analysis",
    "Collect output files",
)


# These labels mirror the major phases in each tab's backend entry point. They
# are deliberately user-facing descriptions rather than implementation names.
TOOL_PROGRESS_STEPS: Dict[str, Sequence[str]] = {
    "find_path": (
        "Initialize source and target neurons",
        "Discover connections layer by layer",
        "Enumerate complete paths",
        "Enrich and aggregate path results",
        "Render visualizations and save outputs",
    ),
    "find_shortest": (
        "Initialize source and target neurons",
        "Discover connections until targets are found",
        "Enumerate shortest paths",
        "Enrich and aggregate path results",
        "Render visualizations and save outputs",
    ),
    "find_network": (
        "Initialize queried neurons",
        "Fetch mutual direct connections",
        "Enrich and filter network edges",
        "Save network data",
        "Build network visualizations",
    ),
    "connectivity_profiling": (
        "Resolve queried neurons and datasets",
        "Extract and aggregate connectivity profiles",
        "Compute similarity matrices",
        "Save matrices, profiles, and heatmaps",
    ),
    "find_homologs": (
        "Build source connectivity profiles",
        "Build target connectivity profiles",
        "Compare and score candidates",
        "Save homolog results",
    ),
    "find_similar_morphology": (
        "Resolve query neuron",
        "Discover candidates from connectivity",
        "Select top candidates for the scoring pool",
        "Load and vectorize skeletons",
        "Score morphological similarity",
        "Save results and visualization",
    ),
    "morphology_comparison": (
        "Resolve query neurons and members",
        "Load and vectorize skeletons",
        "Compute similarity matrices",
        "Save matrices, heatmaps, and report",
    ),
    "inter_dataset": (
        "Resolve datasets and thresholds",
        "Run path or edge analyses",
        "Compute cross-dataset metrics",
        "Export reports and result tables",
        "Generate comparison visualizations and HTML report",
    ),
    "nb_find_lines": (
        "Resolve EM neuron queries",
        "Fetch matching driver-line records",
        "Aggregate and rank line matches",
        "Download images and save outputs",
    ),
    "nb_find_neuron": (
        "Resolve driver-line queries",
        "Fetch matching EM neuron records",
        "Aggregate, rank, and save neurons",
        "Visualize top neurons",
    ),
    "nb_colabel": (
        "Resolve driver lines and datasets",
        "Fetch matches and build expression matrix",
        "Build co-labeling matrices",
        "Compute line statistics and save summaries",
        "Generate visualizations and reports",
    ),
    "flylight_download": (
        "Resolve collections and image filters",
        "List and filter FlyLight files",
        "Download selected images",
        "Generate summaries and save metadata",
    ),
    "plot3d_skeleton": (
        "Load skeletons",
        "Load synapses and meshes",
        "Render and save the 3D scene",
        "Export individual profiles and video",
    ),
    "plot_path": (
        "Load and normalize path data",
        "Build the pathway graph",
        "Create heatmap, Sankey, and network views",
        "Save visualization data and files",
    ),
}


METHOD_PROGRESS_STEPS: Dict[Tuple[str, str], Sequence[str]] = {
    # HomologFinder.find_homologs_fast() emits a six-step backend protocol.
    ("find_homologs", "find_homologs_fast"): (
        "Load connection data",
        "Build source profiles",
        "Discover candidate neurons",
        "Build target profiles",
        "Compare and score candidates",
        "Save homolog results",
    ),
    # HomologFinder.find_homologs() shares the four-stage profile workflow.
    ("find_homologs", "find_homologs"): (
        "Build source profiles",
        "Build target profiles",
        "Compare and score candidates",
        "Save homolog results",
    ),
    ("find_similar_morphology", "cache"): (
        "Resolve query neuron",
        "Load vector cache",
        "Score morphological similarity",
        "Save results and visualization",
    ),
    ("find_similar_morphology", "profile"): (
        "Resolve query neuron",
        "Discover candidates from connectivity",
        "Select top candidates for the scoring pool",
        "Load and vectorize skeletons",
        "Score morphological similarity",
        "Save results and visualization",
    ),
    ("find_similar_morphology", "roi"): (
        "Resolve query neuron",
        "Screen candidates by ROI distributions",
        "Select top candidates for the scoring pool",
        "Load and vectorize skeletons",
        "Score morphological similarity",
        "Save results and visualization",
    ),
    ("find_similar_morphology", "combined"): (
        "Resolve query neuron",
        "Screen candidates by ROI + connectivity",
        "Select top candidates for the scoring pool",
        "Load and vectorize skeletons",
        "Score morphological similarity",
        "Save results and visualization",
    ),
}


# These standard coana tools explicitly resolve their source/target neurons in
# a generated initialization call before their analysis method starts.  Their
# first runner phase must therefore remain on step 1 until the neuron-match
# marker confirms initialization has completed.
INITIALIZATION_TOOLS = frozenset({
    "find_path",
    "find_shortest",
    "find_network",
})


def progress_steps_for(
    tool_name: Optional[str],
    method_name: Optional[str] = None,
    context: Optional[dict] = None,
) -> List[str]:
    """Return a mutable, backend-specific checklist for a tool run.

    The step list must mirror the backend's per-branch totals: optional
    pipeline phases (image downloads, visualizations, reports, exports) are
    only included when the matching run parameter enables them.
    """
    context = context or {}
    if tool_name == "find_similar_morphology":
        source = str(context.get("candidate_source", "auto") or "auto").lower()
        if source == "auto":
            dataset = str(context.get("dataset", "") or "").lower()
            # Mirrors the backend's auto resolution: NeuPrint datasets screen
            # by ROI distributions, while FAFB and standalone BANC search the
            # local vector cache. A server-side fallback from roi to
            # profile keeps the same 6-step protocol, so the checklist stays
            # valid either way.
            source = "cache" if is_local_connectome_dataset(dataset) else "roi"
        key = (tool_name, source)
        if key in METHOD_PROGRESS_STEPS:
            return list(METHOD_PROGRESS_STEPS[key])
    if tool_name == "nb_find_lines":
        steps = list(TOOL_PROGRESS_STEPS["nb_find_lines"])
        if not (context.get("download_images") and context.get("output_dir")):
            steps.pop()  # image download + summary phase is optional
        return steps
    if tool_name == "nb_find_neuron":
        steps = list(TOOL_PROGRESS_STEPS["nb_find_neuron"])
        if not (int(context.get("visualize_top_n") or 0) > 0
                and context.get("output_dir")):
            steps.pop()  # 3D visualization phase is optional
        return steps
    if tool_name == "nb_colabel":
        steps = list(TOOL_PROGRESS_STEPS["nb_colabel"])
        wants_visuals = bool(context.get("generate_report")) or (
            int(context.get("visualize_top_n") or 0) > 0
        )
        if not (wants_visuals and context.get("output_dir")):
            steps.pop()  # report + 3D visualization phase is optional
        return steps
    if tool_name == "flylight_download":
        steps = list(TOOL_PROGRESS_STEPS["flylight_download"])
        if not context.get("generate_summary"):
            steps.pop()  # summary generation phase is optional
        return steps
    if tool_name == "plot3d_skeleton":
        steps = [
            "Load skeletons",
            "Load synapses and meshes",
            "Render and save the 3D scene",
        ]
        if context.get("export_individual_profiles"):
            steps.append("Export individual profiles")
        if context.get("export_video"):
            steps.append("Export rotating video")
        return steps
    if method_name:
        key = (tool_name or "", method_name)
        if key in METHOD_PROGRESS_STEPS:
            return list(METHOD_PROGRESS_STEPS[key])
    return list(TOOL_PROGRESS_STEPS.get(tool_name or "", DEFAULT_PROGRESS_STEPS))


class PageProgress:
    """Render and update the compact page-level progress indicator."""

    def __init__(self) -> None:
        self.card = None
        self.container = None
        self.status_label: Optional[ui.badge] = None
        self.progress_label: Optional[ui.label] = None
        self.percent_label: Optional[ui.label] = None
        self.progress_bar = None
        # Kept as a compatibility attribute; no step-card container is
        # rendered anymore.
        self.steps_container = None

        self._steps: List[str] = []
        self._tool_name: Optional[str] = None
        self._method_name: Optional[str] = None
        self._context: dict = {}
        self._backend_mode = False
        self._counter_mode = False
        self._value = 0.0
        self._running = False
        self._active_index: Optional[int] = None
        self._current_step_label = ""
        self._ui_alive = True
        self._status_text = "Ready"
        self._status_color = "grey-5"

    def create(self, compact: bool = False, visible: bool = True) -> "PageProgress":
        """Create the tracker in the current slot.

        ``compact=True`` is used by :class:`OutputPanel` so the tracker stays
        in its original position immediately above the execution log instead
        of becoming a separate card above the workspace. The compact tracker
        relies on the parent output header for the single status badge.
        """
        self._ui_alive = True
        root = ui.column() if compact else ui.card()
        root_classes = "w-full drocat-page-progress gap-2"
        if not compact:
            root_classes += " drocat-card"
        else:
            root_classes += " drocat-page-progress-compact"
        with root.classes(root_classes) as self.container:
            self.card = self.container
            with ui.row().classes(
                "w-full items-center justify-between drocat-page-progress-head"
            ):
                with ui.row().classes("items-center gap-2"):
                    with ui.element("div").classes("drocat-page-progress-mark"):
                        ui.icon("timeline").classes("text-white")
                    ui.label("Run Progress").classes("drocat-card-title")
                if not compact:
                    self.status_label = ui.badge(
                        "Ready", color="grey-5"
                    ).props("outline")

            with ui.row().classes(
                "w-full items-center justify-between gap-3 drocat-page-progress-summary"
            ):
                self.progress_label = ui.label("Ready to run.").classes(
                    "text-caption drocat-muted"
                )
                self.percent_label = ui.label("0%").classes(
                    "text-caption drocat-progress-percent"
                )

            self.progress_bar = ui.linear_progress(
                value=0, show_value=False
            ).classes("w-full drocat-progress-bar drocat-page-progress-bar").style(
                "height: 12px;"
            )

        self._steps = progress_steps_for(None)
        self.reset()
        self.container.set_visibility(visible)
        return self

    @property
    def step_labels(self) -> List[str]:
        """Return the logical step labels (there are no step cards rendered)."""
        return list(self._steps)

    @property
    def current_value(self) -> float:
        """Return the last determinate fraction sent to the progress bar."""
        return self._value

    def detach(self) -> None:
        """Stop sending element updates after the owning page is deleted."""
        self._ui_alive = False

    @staticmethod
    def _element_is_live(element) -> bool:
        if element is None:
            return False
        try:
            client = element.client
            return not client.is_deleted and not element.is_deleted
        except Exception:
            return False

    def _render_steps(self) -> None:
        """Retained as a no-op compatibility hook for callers."""
        return

    def set_status(self, text: str, color: str) -> None:
        self._status_text = text
        self._status_color = color
        if not self._ui_alive or not self._element_is_live(self.status_label):
            return
        if self.status_label is None:
            return
        try:
            self.status_label.text = text
            self.status_label.props(f"color={color}")
        except Exception:
            pass

    def _set_progress(
        self,
        value: float,
        label: str,
        active_index: Optional[int] = None,
        completed: bool = False,
    ) -> None:
        self._value = max(0.0, min(1.0, float(value)))
        self._active_index = None if completed else active_index
        if not self._ui_alive or not self._element_is_live(self.progress_bar):
            return
        try:
            if self.progress_bar is not None:
                # Remove the static boolean prop as well as setting the dynamic
                # value. Leaving both props in place makes Quasar keep rendering
                # the bar as indeterminate even though its value changes.
                self.progress_bar.props(
                    ":indeterminate='false'", remove="indeterminate"
                )
                self.progress_bar.set_value(self._value)
            if self.percent_label is not None:
                self.percent_label.text = f"{round(self._value * 100):.0f}%"
            if self.progress_label is not None:
                self.progress_label.text = label
        except Exception:
            pass

    def reset(self) -> None:
        """Return the page tracker to its idle state."""
        self._tool_name = None
        self._method_name = None
        self._context = {}
        self._backend_mode = False
        self._counter_mode = False
        self._running = False
        self._steps = progress_steps_for(None)
        self._current_step_label = ""
        self.set_status("Ready", "grey-5")
        self._set_progress(0.0, "Ready to run.", active_index=None)

    def start(
        self,
        tool_name: Optional[str] = None,
        method_name: Optional[str] = None,
        context: Optional[dict] = None,
        steps: Optional[Iterable[str]] = None,
    ) -> None:
        """Start a run and show its explicit checklist."""
        self._tool_name = tool_name
        self._method_name = method_name
        self._context = dict(context or {})
        self._backend_mode = False
        self._running = True
        self._steps = list(steps) if steps is not None else progress_steps_for(
            tool_name, method_name=method_name, context=context
        )
        if not self._steps:
            self._steps = list(DEFAULT_PROGRESS_STEPS)
        self.set_status("Running", "blue")
        total = max(1, len(self._steps))
        first = self._steps[0]
        self._current_step_label = first
        self._set_progress(0.0, f"Step 1/{total}: {first}", active_index=0)

    def update_phase(self, phase: str, label: str = "") -> None:
        """Advance the runner lifecycle using the active tool's step map."""
        if not self._running:
            return

        phase = (phase or "").strip().lower()
        if (self._backend_mode or self._counter_mode) and phase not in {
            "complete", "failed"
        }:
            # Structured backend events are more precise than the subprocess
            # lifecycle. Keep their step and fraction instead of regressing to
            # the generic phase position.
            return

        total = max(1, len(self._steps))
        initialization_tool = self._tool_name in INITIALIZATION_TOOLS
        execute_index = min(1, total - 1) if initialization_tool else 0
        final_index = max(0, total - 1)
        phase_positions = {
            # ``prepare`` and ``initialize`` are both pre-analysis work.
            "prepare": (0.0, 0),
            "initialize": (0.0, 0),
            # Initialized tools advance only after the generated
            # InitializeNeuronInfo call emits its completion marker. Tools
            # without that call stay on their first backend step until a
            # structured event or streaming counter provides finer detail.
            "initialize_complete": (execute_index / total, execute_index),
            "execute": (execute_index / total, execute_index),
            # Collection belongs to the final function-specific output step,
            # not the preceding analysis step.
            "collect": (final_index / total, final_index),
            "complete": (1.0, None),
        }
        value, active_index = phase_positions.get(
            phase, (self._value, max(0, min(total - 1, round(self._value * total))))
        )
        # Keep the current text anchored to the inspected backend stage.
        if active_index is not None and active_index < len(self._steps):
            phase_label = self._steps[active_index]
            phase_text = f"Step {active_index + 1}/{total}: {phase_label}"
        else:
            phase_label = label or {
                "prepare": "Preparing inputs",
                "execute": "Running analysis",
                "collect": "Collecting output files",
                "complete": "Run complete",
            }.get(phase, "Running analysis")
            phase_text = phase_label
        self._current_step_label = phase_label
        self._set_progress(
            value,
            phase_text,
            active_index=active_index,
            completed=phase == "complete",
        )

    def update_step(self, step: int, total: int, label: str = "") -> None:
        """Apply a structured ``step/total`` backend progress event."""
        if not self._running:
            # A direct log consumer can receive a backend event before the
            # caller toggles the run state. Preserve the event instead of
            # dropping the only precise current-progress signal.
            self._running = True
        total = max(1, int(total))
        step = max(1, min(int(step), total))

        # Reuse the descriptive checklist when its length matches the backend
        # protocol.  Otherwise create a precise step list from the event total.
        if len(self._steps) != total:
            candidate_steps = progress_steps_for(
                self._tool_name,
                method_name=self._method_name,
                context=self._context,
            )
            if (
                self._tool_name == "find_similar_morphology"
                and total == 4
            ):
                candidate_steps = list(
                    METHOD_PROGRESS_STEPS[("find_similar_morphology", "cache")]
                )
            self._steps = (
                candidate_steps
                if len(candidate_steps) == total
                else [f"Step {index + 1}" for index in range(total)]
            )
        self._backend_mode = True
        if label and 0 <= step - 1 < len(self._steps):
            self._steps[step - 1] = label

        current_step_label = label or self._steps[step - 1]
        self._current_step_label = current_step_label

        current = f"Step {step}/{total}:"
        if label:
            current += f" {label}"
        # The bar reflects COMPLETED steps only: a backend event marks the
        # START of step ``step``, so its fraction is (step - 1) / total.
        # Starting the final step shows (total - 1) / total — never 100%;
        # the bar reaches 100% only when the run reports completion.  The
        # monotonic guard also absorbs branch switches (e.g. the similarity
        # pipeline falling back from a 6-step to a 4-step protocol).
        value = max(self._value, (step - 1) / total)
        self._set_progress(
            value,
            current,
            active_index=step - 1,
            completed=False,
        )

    def update_fraction(self, current: int, total: int, label: str = "") -> None:
        """Advance the current outer step from a streaming item counter.

        Several tools expose a nested ``tqdm``/``LineProgress`` counter
        instead of the structured step protocol used by similarity search.
        Treat that counter as progress within the currently active outer
        step, so the overall bar advances continuously without resetting when
        a new nested counter starts.
        """
        if total <= 0:
            return
        if not self._running:
            self._running = True
        if self._backend_mode:
            # Structured step events are the authoritative source for these
            # pipelines; nested tqdm bars must not make the bar regress.
            return
        self._counter_mode = True

        total_steps = max(1, len(self._steps))
        active_index = self._active_index
        if active_index is None:
            active_index = max(
                0, min(total_steps - 1, int(self._value * total_steps))
            )
        fraction = max(0.0, min(1.0, float(current) / float(total)))
        value = (active_index + fraction) / total_steps
        value = max(self._value, min(1.0, value))

        progress_text = ""
        if label:
            progress_text = f"Step {active_index + 1}/{total_steps}: {label}"
            self._current_step_label = label
        elif self.progress_label is not None:
            progress_text = self.progress_label.text
        if not progress_text:
            progress_text = self._current_step_label or "Running analysis"

        self._set_progress(
            value,
            progress_text,
            active_index=active_index,
        )

    def finish(self, success: bool = True, message: str = "") -> None:
        """Finish while preserving the last live step on failure."""
        self._running = False
        if self._ui_alive and self._element_is_live(self.container):
            try:
                self.container.set_visibility(True)
            except Exception:
                pass
        if success:
            self.set_status("Completed", "green")
            self._set_progress(1.0, "Completed successfully!", completed=True)
        else:
            self.set_status("Failed", "red")
            # The bar stops at the last reported fraction instead of being
            # forced to 100%, while the label identifies the failed stage.
            failed_during = self._current_step_label or "the current step"
            self._set_progress(
                self._value,
                f"Failed during {failed_during}",
                active_index=self._active_index,
            )

    def restore_state(
        self,
        *,
        tool_name: Optional[str],
        method_name: Optional[str],
        context: Optional[dict],
        status: str,
        progress: Optional[dict],
    ) -> None:
        """Restore a persisted run without replaying the whole live run."""
        progress = dict(progress or {})
        self._tool_name = tool_name
        self._method_name = method_name
        self._context = dict(context or {})
        self._steps = progress_steps_for(
            tool_name, method_name=method_name, context=context
        ) or list(DEFAULT_PROGRESS_STEPS)
        self._backend_mode = bool(progress.get("step"))
        self._counter_mode = bool(progress.get("counter_total"))
        value = float(progress.get("value", 0.0) or 0.0)
        if progress.get("step") and progress.get("total"):
            value = max(
                value,
                (max(1, int(progress["step"])) - 1)
                / max(1, int(progress["total"])),
            )
        if progress.get("counter_total") and not progress.get("value"):
            value = max(
                value,
                min(
                    0.99,
                    float(progress.get("counter_current", 0) or 0)
                    / max(1, int(progress["counter_total"])),
                ),
            )
        if str(status).lower() == "completed":
            value = 1.0
        self._value = max(0.0, min(1.0, value))
        self._active_index = progress.get("active_index")
        if self._active_index is not None:
            self._active_index = max(0, int(self._active_index))
        self._current_step_label = str(progress.get("label", "") or "")
        active = str(status).lower() in {"queued", "running"}
        self._running = active
        color = {
            "completed": "green",
            "failed": "red",
            "cancelled": "orange",
            "interrupted": "orange",
            "running": "blue",
            "queued": "blue",
        }.get(str(status).lower(), "grey-5")
        self.set_status(status, color)
        label = self._current_step_label or str(progress.get("phase", "") or "")
        if progress.get("step") and progress.get("total"):
            label = (
                f"Step {int(progress['step'])}/{int(progress['total'])}: "
                f"{label}".rstrip()
            )
        elif not label:
            label = {
                "completed": "Completed successfully!",
                "failed": "Run failed",
                "cancelled": "Execution cancelled",
                "interrupted": "Execution interrupted",
                "running": "Running analysis",
                "queued": "Queued",
            }.get(str(status).lower(), "Ready to run.")
        self._set_progress(
            self._value,
            label,
            active_index=self._active_index,
            completed=str(status).lower() == "completed",
        )
