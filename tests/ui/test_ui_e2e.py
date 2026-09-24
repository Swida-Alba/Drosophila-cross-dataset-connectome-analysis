#!/usr/bin/env python
"""
Comprehensive End-to-End Tests for DROCAT UI (v4.5.0 - rewritten)
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest


# =============================================================================
# Test Configuration Module
# =============================================================================
class TestConfig:
    def test_config_imports(self):
        from ui import config
        assert config is not None

    def test_config_datasets(self):
        from ui.config import DATASETS, NEUPRINT_DATASETS, FLYWIRE_DATASETS
        assert len(DATASETS) >= 5
        assert "male-cns:v0.9" in NEUPRINT_DATASETS
        assert "flywire_FAFB_v783" in FLYWIRE_DATASETS

    def test_config_defaults(self):
        from ui.config import DEFAULTS
        assert DEFAULTS["min_synapse_num"] == 3
        assert DEFAULTS["max_interlayer"] == 2
        # Skeleton viewer knobs keep the backend's own defaults: the exported
        # HTML pins its 3D axes and individual profiles follow legend entries.
        assert DEFAULTS["freeze_view"] is True
        assert DEFAULTS["profile_granularity"] == "legend"

    def test_config_paths(self):
        from ui.config import PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR
        assert PROJECT_ROOT.exists()
        assert SRC_DIR.exists()

    def test_config_app_settings(self):
        from ui.config import APP_TITLE, APP_VERSION, APP_PORT, APP_HOST
        assert APP_VERSION == "4.5.0"
        assert APP_PORT == 8080


# =============================================================================
# Test Runner Engine
# =============================================================================
class TestRunner:
    def test_runner_imports(self):
        from ui.runner import ScriptRunner, TOOL_REGISTRY
        assert len(TOOL_REGISTRY) >= 9

    def test_script_runner_instantiation(self):
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        assert sr.is_running is False

    def test_tool_registry_has_all_tools(self):
        from ui.runner import TOOL_REGISTRY
        expected = {"find_path", "find_shortest", "connectivity_profiling", "find_homologs",
                     "inter_dataset", "nb_find_lines", "nb_find_neuron", "nb_colabel",
                     "plot3d_skeleton", "plot_path"}
        assert expected.issubset(TOOL_REGISTRY.keys())
        for name in expected:
            assert "label" in TOOL_REGISTRY[name], f"missing UI label for {name}"

    def test_standard_script_defines_method_params(self):
        """A standard tool method that uses **method_params (e.g.
        find_homologs_multi) must have method_params defined in the generated
        script, or the call raises NameError at runtime (regression for the
        Homolog Finding tab)."""
        from ui.runner import ScriptRunner, TOOL_REGISTRY
        sr = ScriptRunner()
        script = sr._generate_script(
            "find_homologs",
            {"source": ["s-LNv"], "source_dataset": "male-cns:v1.0"},
            "find_homologs_multi",
            {"use_fast": True},
        )
        assert "method_params = {'use_fast': True}" in script
        assert "finder.find_homologs_multi(**method_params)" in script
        assert "find_homologs_multi" in TOOL_REGISTRY["find_homologs"]["methods"]

    def test_streaming_logs_emit_lines_and_progress_immediately(self):
        """Partial lines and \\r progress must reach the log before the run ends."""
        import asyncio
        import sys
        from ui.runner import ScriptRunner

        async def run() -> list:
            runner = ScriptRunner()
            code = (
                "import sys, time\n"
                "print('start-line', flush=True)\n"
                "for i in range(3):\n"
                "    sys.stdout.write(f'\\rprogress {i}')\n"
                "    sys.stdout.flush()\n"
                "    time.sleep(0.02)\n"
                "print()\n"
                "print('end-line', flush=True)\n"
                # Raw ANSI colors and tqdm-style bars must be sanitized into
                # clean, single progress lines (no per-refresh log spam).
                "print('\\x1b[33mANSI-warning\\x1b[0m', flush=True)\n"
                "sys.stdout.write('Loading data:   0%|          | 0/4 [00:00<?, ?it/s]\\n')\n"
                "sys.stdout.write('Loading data: 100%|██████████| 4/4 [00:00<00:00,  5.22s/it]\\n')\n"
            )
            runner.process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",
                "-c",
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # The splitter decodes subprocess output as UTF-8; on
                # non-UTF-8 Windows locales the child must emit UTF-8 or the
                # tqdm block characters arrive as replacement chars.
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            events = []
            await runner._stream_output(lambda msg, level: events.append((level, msg)))
            await runner.process.wait()
            return events

        events = asyncio.run(run())
        assert ("stdout", "start-line") in events
        assert ("stdout", "end-line") in events
        progress_events = [(lvl, msg) for lvl, msg in events if lvl == "progress"]
        assert any("progress" in msg for _, msg in progress_events), progress_events
        first_progress = next(i for i, (lvl, _) in enumerate(events) if lvl == "progress")
        assert first_progress < events.index(("stdout", "end-line"))
        # ANSI escapes are stripped, and the tqdm bars arrive as progress.
        assert ("stdout", "ANSI-warning") in events, events
        assert ("progress", "Loading data: 100%|██████████| 4/4 [00:00<00:00,  5.22s/it]") in events
        assert ("progress", "Loading data:   0%|          | 0/4 [00:00<?, ?it/s]") in events
        # No whitespace-only or empty log lines leak into the stream.
        assert all(msg.strip() for _lvl, msg in events), events

    def test_output_splitter_cleans_tqdm_refresh_stream(self):
        """\\r-refreshed tqdm bars become progress segments, spaces are dropped."""
        from ui.runner import _OutputSplitter

        splitter = _OutputSplitter()
        parts = list(splitter.feed(
            "Loading data:   0%|          | 0/4 [00:00<?, ?it/s]   \r"
            "Loading data:  25%|██▌       | 1/4 [00:09<00:28,  9.64s/it]  \r"
            "\n"
        ))
        assert parts == [
            ("Loading data:   0%|          | 0/4 [00:00<?, ?it/s]", True),
            ("Loading data:  25%|██▌       | 1/4 [00:09<00:28,  9.64s/it]", True),
        ]

    def test_output_splitter_classifies_newline_tqdm_bars(self):
        """Non-TTY tqdm writes one bar per line; each must be classified progress."""
        from ui.runner import _OutputSplitter

        splitter = _OutputSplitter()
        parts = list(splitter.feed(
            "Building target profiles:   0%|          | 0/2972 [00:00<?, ?it/s]\n"
            "Building target profiles:  50%|█████     | 1488/2972 [00:06<00:05, 265.91it/s]\n"
            "[HomologFinder] Built 2972 target profiles\n"
        ))
        assert parts == [
            ("Building target profiles:   0%|          | 0/2972 [00:00<?, ?it/s]", True),
            ("Building target profiles:  50%|█████     | 1488/2972 [00:06<00:05, 265.91it/s]", True),
            ("[HomologFinder] Built 2972 target profiles", False),
        ]

    def test_output_splitter_strips_ansi_and_drops_blank_lines(self):
        """ANSI codes are removed; whitespace-only lines are dropped."""
        from ui.runner import _OutputSplitter

        splitter = _OutputSplitter()
        parts = list(splitter.feed(
            "\x1b[33m⚠️  Warning: 19 layers but only 10 colors.\x1b[0m\n"
            "   \n"
            "\n"
            "saving figure to \x1b[34m/out/plot.html\x1b[0m...Done (HTML saved)\n"
        ))
        assert parts == [
            ("⚠️  Warning: 19 layers but only 10 colors.", False),
            ("saving figure to /out/plot.html...Done (HTML saved)", False),
        ]

    def test_output_splitter_cursor_up_refreshes_progress(self):
        """Cursor-up escapes (\\x1b[A) act as progress refresh separators."""
        from ui.runner import _OutputSplitter

        splitter = _OutputSplitter()
        parts = list(splitter.feed(
            "Fetching:   0%|          | 0/6 [00:00<?, ?it/s]\x1b[A\n"
            "Fetching:  17%|█▋        | 1/6 [00:00<00:03,  1.32it/s]\x1b[A\n"
        ))
        assert parts == [
            ("Fetching:   0%|          | 0/6 [00:00<?, ?it/s]", True),
            ("Fetching:  17%|█▋        | 1/6 [00:00<00:03,  1.32it/s]", True),
        ]

    def test_output_splitter_flushes_partial_line(self):
        """A trailing partial line is flushed at EOF."""
        from ui.runner import _OutputSplitter

        splitter = _OutputSplitter()
        assert list(splitter.feed("partial line")) == []
        assert list(splitter.flush()) == [("partial line", False)]

    def test_output_splitter_preserves_crlf_plain_lines(self):
        """CRLF line endings must not be mistaken for progress refreshes."""
        from ui.runner import _OutputSplitter

        splitter = _OutputSplitter()
        parts = list(splitter.feed("[HomologFinder] Loaded cache\r\n"))
        assert parts == [("[HomologFinder] Loaded cache", False)]

    def test_output_splitter_classifies_unit_counter_bars(self):
        """tqdm bars WITHOUT a total (e.g. 'Processing paths: 13295222path
        [00:28, ...]') must be progress lines — not stderr warnings with a
        '[WARN] ' prefix — so they update one line in place."""
        from ui.runner import _OutputSplitter

        splitter = _OutputSplitter()
        parts = list(splitter.feed(
            "Processing paths: 0path [00:00, ?path/s]\n"
            "Processing paths: 36950path [00:00, 369481.85path/s]\n"
            "Processing paths: 13369202path [00:28, 736577.89path/s]\n"
            "⚠️  Warning: something real\n"
        ))
        assert parts == [
            ("Processing paths: 0path [00:00, ?path/s]", True),
            ("Processing paths: 36950path [00:00, 369481.85path/s]", True),
            ("Processing paths: 13369202path [00:28, 736577.89path/s]", True),
            ("⚠️  Warning: something real", False),
        ]
        # the in-place refresh groups all three under the same bar name
        from ui.components.output_panel import _progress_bar_name
        assert _progress_bar_name("Processing paths: 13369202path [00:28, ...]") == \
            "Processing paths"

    def test_output_splitter_emits_lineprogress_without_waiting_for_refresh(self):
        """LineProgress's first CR-only update reaches the UI immediately."""
        from ui.runner import _OutputSplitter

        splitter = _OutputSplitter()
        parts = list(splitter.feed(
            "\rL1 BuildMemo: 0/145273 (0.0%) [00:00<?, 0.0it/s]"
        ))
        assert parts == [
            ("L1 BuildMemo: 0/145273 (0.0%) [00:00<?, 0.0it/s]", True)
        ]

    def test_generate_find_path_script(self):
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        script = sr._generate_script("find_path", {"dataset": "male-cns:v0.9"}, "find_all_path", None)
        assert "from coana import FindNeuronConnection" in script
        assert "FindAllPath" in script
        assert "find_reciprocal=fc.find_reciprocal" in script
        assert "male-cns:v0.9" in script

    def test_generate_find_shortest_script(self):
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        script = sr._generate_script("find_shortest", {"dataset": "hemibrain:v1.2.1"}, "find_shortest", None)
        assert "from coana import" in script
        assert "FindShortestPath" in script

    def test_generate_find_network_script(self):
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        script = sr._generate_script("find_network", {"dataset": "hemibrain:v1.2.1"}, "find_network", None)
        assert "from coana import" in script
        assert "FindNetwork()" in script

    def test_generate_inter_dataset_script(self):
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        script = sr._generate_inter_dataset_script({"datasets": ["a", "b"]}, None)
        assert "ComparisonParameters" in script
        assert "ComparisonAnalyzer" in script
        # the run must EXPORT the results (report files, comparison_results/,
        # visualizations) — generate_report() alone only builds the text
        assert "export_results()" in script
        assert "generate_report()" not in script

    def test_generate_inter_dataset_script_passes_path_mode(self):
        """The shortest enumeration mode reaches ComparisonParameters so the
        per-dataset runs use FindShortestPath."""
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        script = sr._generate_inter_dataset_script(
            {"datasets": ["a", "b"], "path_mode": "shortest", "max_interlayer": 0}, None)
        assert "path_mode='shortest'" in script
        assert "max_interlayer=0" in script

    def test_generate_inter_dataset_script_supports_single_dataset_thresholds(self):
        """Script generation preserves a one-dataset threshold-sensitivity run."""
        from ui.runner import ScriptRunner

        script = ScriptRunner()._generate_inter_dataset_script(
            {
                "datasets": ["male-cns:v1.0"],
                "thresholds": [3, 7],
            },
            None,
        )

        assert "datasets=['male-cns:v1.0']" in script
        assert "thresholds=[3, 7]" in script

    def test_inter_dataset_tab_hides_nickname_editor_and_defaults_to_male_cns(self):
        """Cross-dataset nicknames stay script-only and male-cns is preselected."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.inter_dataset import create_inter_dataset_tab

        client = Client(page("/inter-dataset-defaults"))
        with client:
            create_inter_dataset_tab()

        controls = {
            getattr(el, "_props", {}).get("label"): el
            for el in client.elements.values()
            if getattr(el, "_props", {}).get("label")
        }
        dataset_control = controls[
            "Datasets to compare (one dataset with multiple thresholds is also supported)"
        ]
        assert dataset_control.value == ["male-cns:v1.0"]
        assert not any("Nickname" in label for label in controls)
        assert any(
            getattr(el, "text", "") == "Custom Mapping · none"
            for el in client.elements.values()
        )

    def test_inter_dataset_allows_one_dataset_with_multiple_thresholds(self):
        """The UI accepts one dataset while retaining multiple thresholds."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.inter_dataset import create_inter_dataset_tab

        client = Client(page("/inter-dataset-single-thresholds"))
        with client:
            create_inter_dataset_tab()

        controls = {
            getattr(el, "_props", {}).get("label"): el
            for el in client.elements.values()
            if getattr(el, "_props", {}).get("label")
        }
        dataset_control = controls[
            "Datasets to compare (one dataset with multiple thresholds is also supported)"
        ]
        threshold_control = controls["Synapse Thresholds"]
        dataset_control.value = ["male-cns:v1.0"]
        threshold_control.value = [3, 5]

        assert dataset_control.value == ["male-cns:v1.0"]
        assert dataset_control._props.get("multiple") is True
        assert threshold_control.value == [3, 5]

    def test_shortest_tab_removes_polynomial_explanation_and_thresholds_span_row(self):
        """Keep the shortest tab concise and give thresholds a full row."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.find_shortest import create_find_shortest_tab
        from ui.tabs.inter_dataset import create_inter_dataset_tab

        shortest_client = Client(page("/shortest-no-explanation"))
        with shortest_client:
            create_find_shortest_tab()
        shortest_texts = [
            getattr(el, "text", "")
            for el in shortest_client.elements.values()
            if getattr(el, "text", "")
        ]
        assert not any(
            "Only per-pair shortest paths are enumerated" in text
            for text in shortest_texts
        )

        inter_client = Client(page("/interdataset-threshold-full-row"))
        with inter_client:
            create_inter_dataset_tab()
        full_row_controls = [
            el for el in inter_client.elements.values()
            if "drocat-full-row-control" in getattr(el, "_classes", [])
        ]
        assert len(full_row_controls) == 1
        ancestor = full_row_controls[0]
        ancestor_ids = []
        while ancestor is not None:
            ancestor_ids.append(getattr(ancestor, "_props", {}).get("id", ""))
            parent_slot = getattr(ancestor, "parent_slot", None)
            ancestor = parent_slot.parent if parent_slot is not None else None
        assert "card-interdataset-core" in ancestor_ids

    def test_generate_neuronbridge_script(self):
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        script = sr._generate_neuronbridge_script("nb_find_lines", {"verbose": True}, {"queries": ["aMe12"]})
        assert "NeuronBridgeFinder" in script
        assert "find_lines" in script

        colabel_script = sr._generate_script(
            "nb_colabel",
            {"verbose": True},
            "colabel",
            {"lines": ["A", "B"], "datasets_to_visualize": "male-cns:v1.0"},
        )
        assert "datasets_to_visualize" in colabel_script
        assert "male-cns:v1.0" in colabel_script

    def test_generated_scripts_include_vispath(self):
        """The PlotPath tool must resolve vispath_pkg in a fresh subprocess."""
        from ui.runner import ScriptRunner, VISPATH_DIR
        assert VISPATH_DIR.exists(), "vispath-subproject/src missing"
        sr = ScriptRunner()
        script = sr._generate_script("plot_path", {"path_file": "x.csv"}, "plot", None)
        assert str(VISPATH_DIR) in script
        assert "vispath_pkg" in script
        assert "vp.visualize()" in script

    def test_generated_script_supports_empty_network_canvas(self):
        """The network panel can run PlotPath without a path input file."""
        from ui.runner import ScriptRunner

        script = ScriptRunner()._generate_script(
            "plot_path",
            {
                "path_file": None,
                "output_folder": "/tmp/drocat-empty-network",
                "generate_empty_network": True,
            },
            "plot",
            None,
        )
        assert "path_file=None" in script
        assert "generate_empty_network=True" in script
        assert "vp.visualize()" in script

    def test_empty_network_opens_a_new_browser_tab(self, tmp_path, monkeypatch):
        """A visible empty canvas opens as a fresh browser tab after export.

        Both webbrowser entry points are patched: the same-window ``open`` must
        never fire (the canvas opens exactly once, in a new tab), otherwise the
        test would pop a real browser window.
        """
        import webbrowser

        sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))
        from vispath_pkg import VisualizePath

        opened = []
        monkeypatch.setattr(webbrowser, "open", opened.append)
        monkeypatch.setattr(webbrowser, "open_new_tab", opened.append)
        visualizer = VisualizePath(
            path_file=None,
            output_folder=str(tmp_path),
            generate_empty_network=True,
            showfig=True,
            verbose=False,
        )
        output_path = Path(visualizer.generate_empty_network_html())

        assert output_path.exists()
        # Exactly one open: a fresh tab (not the same-window open + tab).
        assert opened == [f"file://{output_path.resolve()}"]

    def test_generate_plot3d_script_includes_optional_exports(self):
        """Plot3D optional profile/video calls must be appended after plot_neurons."""
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        script = sr._generate_script(
            "plot3d_skeleton",
            {"dataset": "male-cns:v0.9", "neuron_layers": ["aMe12"]},
            "plot",
            {
                "export_individual_profiles": True,
                "pdf_images_per_page": (3, 2),
                "views": ["front"],
                "summary_format": ["pdf"],
                "export_video": True,
                "fps": 30,
                "degree_per_frame": 1.0,
                "rotate": "horizontal",
                "export_gif": True,
                "gif_scale": 0.2,
            },
        )
        assert "vs.plot_neurons()" in script
        assert "vs.plot_individuals(" in script
        assert "vs.export_video(" in script
        # The viewer pin and the profile grouping are always spelled out so
        # the generated script states the framing/grouping it relies on.
        assert "freeze_view=" in script
        assert "granularity=" in script
        # Optional export keys must not leak into the VisualizeSkeleton constructor
        assert "export_individual_profiles" not in script.split("VisualizeSkeleton(")[1].split(")")[0]

    def test_generate_plot3d_script_threads_freeze_view_and_granularity(self):
        """freeze_view reaches the VisualizeSkeleton constructor and the
        chosen profile granularity reaches plot_individuals; the tab's values
        win over the generated-script defaults and are emitted once."""
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        script = sr._generate_script(
            "plot3d_skeleton",
            {
                "dataset": "male-cns:v0.9",
                "neuron_layers": ["aMe12"],
                "legend_mode": "tree",
                "freeze_view": False,
            },
            "plot",
            {
                "export_individual_profiles": True,
                "pdf_images_per_page": (3, 2),
                "views": ["front"],
                "summary_format": ["pdf"],
                "granularity": "body",
            },
        )
        constructor = script.split("VisualizeSkeleton(")[1].split(")")[0]
        assert "freeze_view=False" in constructor
        assert script.count("freeze_view=") == 1, "the pin must appear exactly once"
        profiles = script.split("vs.plot_individuals(")[1].split("\n)")[0]
        assert "granularity='body'" in profiles
        assert "granularity" not in constructor
        # The emitted call must stay valid Python (no duplicated keywords).
        compile(script, "generated_plot3d_skeleton.py", "exec")

    def test_homologs_ui_params_match_signature(self):
        """Regression: the UI used to send expand_untyped_2hop, which does not exist."""
        import inspect
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from comparison.profile_comparator import HomologFinder
        params = inspect.signature(HomologFinder.__init__).parameters
        assert "include_untyped_partners" in params
        assert "expand_untyped_2hop" not in params

    def test_similarity_tabs_have_no_loose_search_controls(self):
        """The loose-search knobs (Min Shared Partners, Candidate Prune %)
        belonged to the removed connectivity-similarity mode; neither the
        Connectivity nor the Morphology tab exposes them anymore."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.connectivity import create_connectivity_tab
        from ui.tabs.morphology import create_morphology_tab

        for factory, url in (
            (create_connectivity_tab, "/similar-no-loose-controls"),
            (create_morphology_tab, "/morphology-no-loose-controls"),
        ):
            client = Client(page(url))
            with client:
                factory()
            labels = [
                getattr(el, "_props", {}).get("label")
                for el in client.elements.values()
                if getattr(el, "_props", {}).get("label")
            ]
            assert "Min Shared Partners" not in labels
            assert "Candidate Prune %" not in labels

    def test_homologs_sort_by_defaults_to_jaccard(self):
        """The candidate-ranking control is labeled 'Sort By' and defaults to
        jaccard (the benchmark's best bodyId-level metric — conservative,
        bounded [0,1], never degenerate): all metrics are always computed by
        the shared backend, so the selection only affects the candidate
        ordering (top-N cut)."""
        from nicegui import Client
        from nicegui.page import page
        from ui.config import DEFAULTS
        from ui.tabs.connectivity import create_connectivity_tab

        client = Client(page("/similar-sort-by"))
        with client:
            create_connectivity_tab()
        by_label = {}
        for el in client.elements.values():
            label = getattr(el, "_props", {}).get("label")
            if label == "Sort By":
                by_label[label] = el
        assert "Sort By" in by_label, by_label
        assert by_label["Sort By"].value == DEFAULTS["similarity_metric"]
        assert DEFAULTS["similarity_metric"] == "jaccard"
        assert "Similarity Metric" not in by_label

    def test_morphology_tab_has_find_similar_and_comparison(self):
        """The Morphology tab renders the Find Similar sub-tab (morphological
        search) and the Comparison sub-tab (intra-dataset N×N comparison
        form); the connectivity-similarity mode and its loose knobs are
        gone."""
        from nicegui import Client
        from nicegui.page import page
        from ui.config import DEFAULTS
        from ui.tabs.morphology import create_morphology_tab

        client = Client(page("/morphology-tab-structure"))
        with client:
            create_morphology_tab()

        labels = [
            getattr(el, "_props", {}).get("label")
            for el in client.elements.values()
            if getattr(el, "_props", {}).get("label")
        ]
        texts = [
            getattr(el, "text", "")
            for el in client.elements.values()
            if getattr(el, "text", "")
        ]
        # morphological Find Similar controls (the legacy cosine/Pearson
        # "Metric" selector was removed: vector_v2 scoring is per-block
        # whitened cosine and NBLAST carries its own score)
        for label in ("Query Neuron(s)", "Level",
                      "Candidate Cap", "Candidate Source",
                      "ROI Filter", "Visualize Top N Types / Neurons",
                      "Visualize By"):
            assert label in labels, f"missing morphological control: {label}"
        assert "Download All Skeletons" not in labels
        # removed connection-profile controls
        for gone in ("Min Shared Partners", "Candidate Prune %",
                     "Visualize Top Candidates", "Visualize Top N Candidates",
                     "Top N Candidates", "Query Neuron (type or bodyId)"):
            assert gone not in labels and gone not in texts, gone

        by_label = {
            getattr(el, "_props", {}).get("label"): el
            for el in client.elements.values()
            if getattr(el, "_props", {}).get("label")
        }
        # level auto-follows the query kind (type -> type, bodyId -> bodyId)
        assert by_label["Level"].value == DEFAULTS["morph_level"] == "auto"
        # the two size knobs: candidate cap for the pool, top-N for rendering
        assert by_label["Candidate Cap"].value == DEFAULTS["candidate_cap"] == 500
        # 3D visualization defaults: enabled with 6 top types, grouped by type
        assert by_label["Visualize Top N Types / Neurons"].value == DEFAULTS["morph_visualize_top_n"]
        assert by_label["Visualize By"].value == DEFAULTS["morph_visualize_by"]
        # Analysis panels now default to the fast pipeline (0.90 removal).
        assert by_label["Mesh Simplification"].value == 0.90
        # The Advanced Visualization editor renders as a drocat-card section
        # (same pattern as the Skeleton tab's appearance blocks), one per
        # sub-tab, not a bare inline expander beside the Visualize checkbox.
        advanced_viz = [
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Advanced Visualization"
        ]
        assert len(advanced_viz) == 2  # Find Similar + Comparison sub-tabs
        for expansion in advanced_viz:
            assert "drocat-section-expansion" in getattr(expansion, "_classes", [])
            card = expansion.parent_slot.parent
            assert "drocat-card" in getattr(card, "_classes", [])
            assert str(getattr(card, "_props", {}).get("id", "")).endswith(
                "-advanced-viz"
            )

        # The two sub-tabs are independent, outlined buttons rather than
        # one segmented toggle, so each remains easy to target and read.
        # The Comparison sub-tab covers intra- AND cross-dataset
        # comparison, dispatched by the number of selected datasets.
        mode_buttons = {
            getattr(el, "text", ""): el
            for el in client.elements.values()
            if getattr(el, "text", "") in {"Find Similar", "Comparison"}
        }
        assert set(mode_buttons) == {"Find Similar", "Comparison"}
        assert len(mode_buttons) == 2
        # The per-tab "Build Vector Cache" action row was removed when the
        # cross-dataset comparison sub-tab landed (1d356c7); vector-cache
        # building lives with Find Similar / dataset tooling instead.

        # Comparison sub-tab: real form with its own dataset card, neuron
        # list, method selector, and run button.
        for label in ("Neurons to Compare", "Max Members per Type",
                      "Max Total Neurons", "Advanced Visualization"):
            assert label in labels, f"missing Comparison control: {label}"
        assert labels.count("Method") == 2  # Find Similar + Comparison
        assert "Run Comparison" in texts
        assert "Find Similar Neurons" in texts

        # Cross-dataset mode lives inside the Comparison sub-tab (two or
        # more selected datasets dispatch to the cross-dataset backend).
        assert "Datasets to compare" in labels
        assert "Reference Template (scenes)" in labels
        assert "Null Sample Size" in labels
        # NiceGUI checkboxes carry their label in `_text`, not the `label`
        # prop the collectors above read.
        checkbox_labels = [
            getattr(el, "_text", "")
            for el in client.elements.values()
            if getattr(el, "_text", "")
        ]
        assert "Auto Type Mapping" in checkbox_labels
        assert not any("Coming in a future update" in text for text in texts)

    def test_connectivity_tab_hosts_find_similar_and_comparison(self):
        """The Connectivity tab hosts the Find Homolog (homolog engine) and
        Comparison sub-tabs; the removed connectivity-similarity knobs are
        gone and Sort By keeps its jaccard default."""
        from nicegui import Client
        from nicegui.page import page
        from ui.config import DEFAULTS
        from ui.tabs.connectivity import create_connectivity_tab

        client = Client(page("/connectivity-tab-structure"))
        with client:
            create_connectivity_tab()

        labels = [
            getattr(el, "_props", {}).get("label")
            for el in client.elements.values()
            if getattr(el, "_props", {}).get("label")
        ]
        texts = [
            getattr(el, "text", "")
            for el in client.elements.values()
            if getattr(el, "text", "")
        ]
        # Find Similar (homolog engine) controls
        for label in ("Source Dataset", "Target Dataset",
                      "Source Neuron(s) (type or bodyId)",
                      "Top N Candidates", "Top K Partners", "Min Types (M)",
                      "Sort By", "Min Synapse Threshold",
                      "Auto Type Mapping", "Visualize Top N Candidates",
                      "Qualification bar offset above the null bar"):
            assert label in labels or label in texts, \
                f"missing Find Homolog control: {label}"
        # Comparison controls
        for label in ("Datasets to compare (select one or more)",
                      "Neurons to Compare", "Aggregation Level",
                      "BodyId-Level Computation"):
            assert label in labels, f"missing Comparison control: {label}"
        # removed connectivity-similarity controls
        for gone in ("Min Shared Partners", "Candidate Prune %",
                     "Similarity Metric", "Visualize Top Candidates"):
            assert gone not in labels and gone not in texts, gone

        by_label = {
            getattr(el, "_props", {}).get("label"): el
            for el in client.elements.values()
            if getattr(el, "_props", {}).get("label")
        }
        assert by_label["Top N Candidates"].value == DEFAULTS["top_n"]
        assert by_label["Sort By"].value == DEFAULTS["similarity_metric"]
        assert by_label["Visualize Top N Candidates"].value == 5

        # The two sub-tabs are independent, outlined buttons.
        mode_buttons = {
            getattr(el, "text", ""): el
            for el in client.elements.values()
            if getattr(el, "text", "") in {"Find Homolog", "Comparison"}
        }
        assert set(mode_buttons) == {"Find Homolog", "Comparison"}
        assert len(mode_buttons) == 2

    def test_analysis_visualization_simplification_uses_analysis_default(self):
        """The advanced panel displays the same analysis default that
        the analysis backend will use, without overwriting custom values."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.morphology import create_morphology_tab

        client = Client(page("/similar-visualization-simplification"))
        with client:
            create_morphology_tab()

        default_control = next(
            el for el in client.elements.values()
            if getattr(el, "text", "") == "Default Simplification"
        )
        dataset = None
        for el in client.elements.values():
            if getattr(el, "_props", {}).get("label") != "Dataset":
                continue
            ancestor = el
            while ancestor is not None:
                if getattr(ancestor, "_props", {}).get("id", "") == \
                        "card-morphology-findsimilar-dataset":
                    dataset = el
                    break
                parent_slot = getattr(ancestor, "parent_slot", None)
                ancestor = parent_slot.parent if parent_slot is not None else None
            if dataset is not None:
                break
        assert dataset is not None
        # The Find Similar sub-tab has one advanced visualization editor.
        mesh = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Mesh Simplification"
        )

        # The method defaults are shared across datasets: 0.90 for fast and
        # 0.95 for fine/artistic.
        assert mesh.value == 0.90
        dataset.value = "flywire_FAFB_v783"
        assert mesh.value == 0.90

        default_control.value = False
        mesh.value = 0.75
        dataset.value = "male-cns:v1.0"
        assert mesh.value == 0.75

        default_control.value = True
        assert mesh.value == 0.90

        mode = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Skeleton Mode"
        )
        method = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Simplification Method"
        )
        # Analysis tabs default to line mode with the fast pipeline;
        # tube/fine is an explicit high-quality opt-in.
        assert mode.value == "line"
        assert method.enabled is False
        assert method.value == "fast"
        assert default_control.enabled is False
        assert mesh.enabled is False
        mode.value = "tube"
        assert method.enabled is True
        assert default_control.enabled is True

    def test_similar_tools_generate_runner_scripts(self):
        """The runner generates scripts for both similarity tools."""
        from ui.runner import ScriptRunner, TOOL_REGISTRY

        assert "find_similar_morphology" in TOOL_REGISTRY
        assert "morphology_comparison" in TOOL_REGISTRY
        assert "find_homologs" in TOOL_REGISTRY
        assert "find_similar_profile" not in TOOL_REGISTRY

        sr = ScriptRunner()
        morph_script = sr._generate_script(
            "find_similar_morphology",
            {"query": "aMe12", "dataset": "male-cns:v1.0", "method": "vector_v2",
             "candidate_cap": 500, "visualize_top_n": 6, "visualize_by": "type"},
            "find_similar",
            None,
        )
        assert "from morphology import MorphologyComparer" in morph_script
        assert "comparer.find_similar()" in morph_script
        assert "method='vector_v2'" in morph_script
        assert "candidate_cap=500" in morph_script
        assert "visualize_top_n=6" in morph_script
        assert "visualize_by='type'" in morph_script

        similar_script = sr._generate_script(
            "find_homologs",
            {"source": ["aMe12"], "source_dataset": "male-cns:v1.0",
             "target_dataset": "male-cns:v1.0", "top_n": 100,
             "use_auto_type_mapping": True,
             "visualize_skeleton": True, "visualize_top_n": 5,
             "visualization_settings": {"brain_mesh": "native"}},
            "find_homologs_multi",
            {"use_fast": True},
        )
        assert "from comparison.profile_comparator import HomologFinder" in similar_script
        assert "finder.find_homologs_multi(" in similar_script
        assert "method_params" in similar_script
        assert "use_auto_type_mapping=True" in similar_script
        assert "visualize_skeleton=True" in similar_script
        assert "visualize_top_n=5" in similar_script
        assert "visualization_settings={'brain_mesh': 'native'}" in similar_script

        comparison_script = sr._generate_script(
            "morphology_comparison",
            {"dataset": "male-cns:v1.0", "query": ["aMe12", "aMe10"],
             "method": "vector_v2", "max_members_per_type": 25,
             "max_total_neurons": 200, "output_dir": "/tmp/morph_cmp",
             "generate_heatmaps": True, "show_figures": False},
            "run",
            None,
        )
        assert "from morphology_comparison import MorphologyProfileComparer" \
            in comparison_script
        assert "comparer.run()" in comparison_script
        assert "max_members_per_type=25" in comparison_script
        assert "query=['aMe12', 'aMe10']" in comparison_script

    def test_homologs_empty_saveas_uses_auto_folder(self, tmp_path):
        """UI sends saveas='' when blank; results must land in a per-run
        homologs_ folder instead of being dumped into output_dir."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        import pandas as pd
        from comparison.profile_comparator import HomologFinder

        finder = HomologFinder(
            source="aMe12",
            source_dataset="male-cns:v1.0",
            target_dataset="male-cns:v1.0",
            output_dir=str(tmp_path),
            saveas="",  # The UI passes an empty string when the field is blank
            verbose=False,
        )
        results = pd.DataFrame({
            "source_bodyId": [1, 2],
            "target_bodyId": [3, 4],
            "rank_union": [0.5, 0.4],
        })
        finder._save_homolog_results_internal(
            results_df=results,
            query="aMe12",
            source_dataset="male-cns:v1.0",
            target_dataset="male-cns:v1.0",
            output_dir=str(tmp_path),
            saveas="",
            direction="both",
            include_partner_details=False,
            top_n_details=5,
            params={"query": "aMe12"},
        )
        folders = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(folders) == 1, [p.name for p in tmp_path.iterdir()]
        folder = folders[0]
        assert folder.name.startswith("homologs_MCNS_to_MCNS_aMe12_"), folder.name
        assert (folder / "README.txt").exists()
        assert (folder / "results" / "homolog_results.csv").exists()
        # Nothing is dumped into the output root itself
        assert not (tmp_path / "README.txt").exists()
        assert not (tmp_path / "results").exists()

    def test_homologs_custom_saveas_respected(self, tmp_path):
        """A non-empty saveas must still be used as the folder name."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        import pandas as pd
        from comparison.profile_comparator import HomologFinder

        finder = HomologFinder(
            source="aMe12",
            source_dataset="male-cns:v1.0",
            target_dataset="male-cns:v1.0",
            output_dir=str(tmp_path),
            saveas="my_custom_run",
            verbose=False,
        )
        results = pd.DataFrame({"source_bodyId": [1], "target_bodyId": [3], "rank_union": [0.5]})
        finder._save_homolog_results_internal(
            results_df=results,
            query="aMe12",
            source_dataset="male-cns:v1.0",
            target_dataset="male-cns:v1.0",
            output_dir=str(tmp_path),
            saveas="my_custom_run",
            direction="both",
            include_partner_details=False,
            top_n_details=5,
            params={"query": "aMe12"},
        )
        assert (tmp_path / "my_custom_run" / "README.txt").exists()

    def test_homologs_loose_search_params(self):
        """HomologFinder exposes loose-search knobs: min_shared_partners and
        vector_prune_fraction flow from the constructor into the searches."""
        import inspect
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from comparison.profile_comparator import HomologFinder

        init_params = inspect.signature(HomologFinder.__init__).parameters
        assert "min_shared_partners" in init_params
        assert "vector_prune_fraction" in init_params
        for method in ("find_homologs", "find_homologs_fast"):
            params = inspect.signature(getattr(HomologFinder, method)).parameters
            assert "vector_prune_fraction" in params
        fast_params = inspect.signature(HomologFinder.find_homologs_fast).parameters
        # None default = falls back to the constructor-level setting
        assert fast_params["min_shared_partners"].default is None

        finder = HomologFinder(
            source="aMe12",
            source_dataset="male-cns:v1.0",
            target_dataset="male-cns:v1.0",
            verbose=False,
            min_shared_partners=1,
            vector_prune_fraction=1.0,
        )
        assert finder.min_shared_partners == 1
        assert finder.vector_prune_fraction == 1.0

    def test_homologs_prune_fraction_controls_candidate_pool(self, monkeypatch):
        """_compare_candidates_core keeps only the top-5% of candidates by
        default, but every cosine-positive candidate when
        vector_prune_fraction is 1.0 (loose search)."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from comparison import profile_comparator as pc
        from comparison.profile_comparator import HomologFinder
        from comparison.connectivity_profiler import ConnectivityStatus

        class FakeProfile:
            def __init__(self, partners):
                self.upstream_partners = dict(partners)
                self.downstream_partners = {}
                self.untyped_upstream_bodyids = {}
                self.untyped_downstream_bodyids = {}
                self.untyped_upstream_2hop = {}
                self.untyped_downstream_2hop = {}
                self.connectivity_status = ConnectivityStatus.COMPLETE

        finder = object.__new__(HomologFinder)
        finder.verbose = False
        finder._in_progress_bar = False

        monkeypatch.setattr(
            pc.ProfileComparator, "weighted_cosine_similarity",
            staticmethod(lambda a, b, direction: 0.5),  # every candidate positive
        )
        seen = {"bids": None}

        def fake_batch(source_profile, target_profiles_cache, candidate_map, direction,
                       type_mapper=None):
            seen["bids"] = sorted(candidate_map.keys())
            return []

        monkeypatch.setattr(
            pc.ProfileComparator, "batch_compare_cross_dataset",
            staticmethod(fake_batch),
        )

        target_profiles = {bid: FakeProfile({"A": 5}) for bid in range(2, 22)}
        common = dict(
            source_bodyids=[1],
            source_profiles_cache={1: FakeProfile({"A": 5, "B": 5})},
            source_status_map={1: ConnectivityStatus.COMPLETE},
            target_profiles_cache=target_profiles,
            target_type_lookup={bid: f"T{bid}" for bid in range(2, 22)},
            source_type_lookup={1: "SRC"},
            candidate_map={1: {bid: 1 for bid in range(2, 22)}},  # 20 candidates
            is_cross_dataset=True,
            target_dataset="male-cns:v1.0",
            show_progress=False,
            similarity_metric="rank_union",
            top_n=20,
            include_intra_type=False,
            vector_prefiltering=True,
            type_mapper=None,
        )

        # Default prune: top 5% of 20 = 1 candidate reaches full scoring.
        finder._compare_candidates_core(**common, vector_prune_fraction=0.05)
        assert seen["bids"] == [2]

        # Loose: every cosine-positive candidate reaches full scoring.
        finder._compare_candidates_core(**common, vector_prune_fraction=1.0)
        assert seen["bids"] == list(range(2, 22))

    def test_interdataset_output_name_prefix(self):
        """Cross-dataset runs use the cross-dataset_ folder prefix."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from comparison.comparison_parameters import ComparisonParameters

        params = ComparisonParameters(
            datasets=["male-cns:v0.9", "hemibrain:v1.2.1"],
            source_neurons=["aMe12"],
            target_neurons=["PPL101"],
            output_folder="/tmp/drocat-test",
        )
        assert params.output_name.startswith("cross-dataset_aMe12_to_PPL101_"), params.output_name
        assert "comp_" not in params.output_name
        # Empty-string saveas must fall back to the auto name
        params.saveas = ""
        assert params.output_name.startswith("cross-dataset_aMe12_to_PPL101_"), params.output_name

    def test_scan_output_files_empty(self):
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        assert sr._scan_output_files("/nonexistent") == []

    def test_scan_output_files_with_files(self):
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.csv").write_text("a,b\n1,2")
            files = sr._scan_output_files(tmpdir)
            assert len(files) >= 1

    def test_scan_output_files_returns_all_files_no_cap(self):
        """The panel mirrors the run folder - no 50-file cap may hide files
        (hundreds of images are written by the NB find-lines workflow)."""
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(60):
                (Path(tmpdir) / f"file_{i:03d}.csv").write_text("a,b\n1,2")
            (Path(tmpdir) / "sub" / "nested.txt").parent.mkdir()
            (Path(tmpdir) / "sub" / "nested.txt").write_text("x")
            files = sr._scan_output_files(tmpdir)
            assert len(files) == 61  # every file, nested included

    def test_extract_output_folder_picks_current_run(self):
        """The results panel must link to THIS run's folder, not older ones."""
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            old = Path(tmpdir) / "aMe12_to_aMe10"
            new = Path(tmpdir) / "find-paths-complete_MCNS_aMe12_to_aMe10_L2w3_20260801_120000"
            old.mkdir()
            new.mkdir()
            sr._run_logs = [
                ("stdout", f"data will be saved in: {old}"),
                ("stdout", f"  📁 Created output folder: {new}"),
            ]
            assert sr._extract_output_folder(tmpdir) == str(new)

            sibling = Path(f"{tmpdir}-outside")
            sibling.mkdir()
            try:
                sr._run_logs = [("stdout", f"Created output folder: {sibling}")]
                assert sr._extract_output_folder(tmpdir) is None
            finally:
                sibling.rmdir()

        sr2 = ScriptRunner()
        sr2._run_logs = [("stdout", "nothing here")]
        assert sr2._extract_output_folder("/tmp") is None

    def test_extract_output_folder_accepts_path_only_morphology_marker(self):
        """The morphology backend announces its run folder with a path-only
        'Results saved to:' marker (counts on a separate line); the UI must
        resolve it so output files stream during similar finding."""
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            run = Path(tmpdir) / "similar-morphology_hemibrain_v1_2_1_aMe12_20260801_120000"
            run.mkdir()
            (run / "results.csv").write_text("rank,target_bodyId\n1,201")
            sr._run_logs = [
                ("stdout", f"Results saved to: {run}"),
                ("stdout", "Saved 30 bodyId rows -> results.csv, "
                           "12 type rows -> type_summary.csv"),
            ]
            assert sr._extract_output_folder(tmpdir) == str(run)
            assert sr._resolve_scan_dir(tmpdir) == str(run)

    def test_resolve_scan_dir_failed_run_returns_none(self):
        """A failed run that created no folder must NOT scan the shared root
        (which would surface files from previous runs, e.g. BANC files while
        running male-cns)."""
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "banc_previous_output.csv").write_text("a,b\n1,2")
            sr._run_logs = [
                ("stderr", "[ERROR] MTe07 not found as source in any columns"),
            ]
            assert sr._resolve_scan_dir(tmpdir) is None

    def test_resolve_scan_dir_uses_announced_run_folder(self):
        """The backend-announced per-run folder wins over the storage root."""
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            run = Path(tmpdir) / "find-paths-complete_MCNS_aMe12_to_aMe10_L2w3_20260801_120000"
            run.mkdir()
            sr._run_logs = [("stdout", f"  📁 Created output folder: {run}")]
            assert sr._resolve_scan_dir(tmpdir) == str(run)

    def test_resolve_scan_dir_accepts_direct_run_folder(self):
        """plot_path passes its pre-created plot-network_ folder directly."""
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            run = Path(tmpdir) / "plot-network_my_paths_20260801_120000"
            run.mkdir()
            sr._run_logs = [("stdout", "plotting network...")]
            assert sr._resolve_scan_dir(str(run)) == str(run)

    def test_generate_color_palette_small_n(self):
        """Bokeh categorical palettes have minimum sizes; small n must still work."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from utils.color_utils import generate_color_palette
        assert len(generate_color_palette(1, "category20")) == 1
        assert len(generate_color_palette(2, "category10")) == 2
        assert len(generate_color_palette(3, "cool")) == 3

    def test_palette_catalog_and_sampling(self):
        """The bokeh catalog must include categorical/sequential/diverging
        palettes and sample them evenly for any requested count."""
        from ui.components.palette_picker import (
            assign_palette_colors,
            get_palette_catalog,
            move_color,
            normalize_palette_range,
            palette_slice,
            sample_palette,
        )
        catalog = get_palette_catalog()
        names = [name for name, _ in catalog]
        assert "Category20" in names
        assert "Blues" in names
        assert "Spectral" in names
        assert len(catalog) >= 40
        colors = dict(catalog)["Category20"]
        assert len(sample_palette(colors, 1)) == 1
        assert len(sample_palette(colors, 5)) == 5
        assert len(sample_palette(colors, 50)) == 50
        assert assign_palette_colors(["red", "green", "blue"], 2) == [
            "red", "green",
        ]
        assert assign_palette_colors(["blue", "red", "green"], 5) == [
            "blue", "red", "green", "blue", "red",
        ]
        half = palette_slice(colors, 0, 50)
        assert 0 < len(half) < len(colors)
        assert palette_slice(colors, 100, 100) == colors[-1:]
        assert normalize_palette_range(80, 20) == (80, 81)
        assert normalize_palette_range(-5, 200) == (0, 100)
        assert move_color(["a", "b", "c"], 1, -1) == ["b", "a", "c"]
        assert move_color(["a", "b", "c"], 0, -1) == ["a", "b", "c"]

    def test_palette_catalog_names_embed_their_correct_bokeh_sources(self):
        """Every displayed name must retain its own source palette."""
        import bokeh.palettes as bp

        from ui.components.palette_picker import get_palette_catalog

        catalog = dict(get_palette_catalog())

        # Cover each source family, including the two sizes used by the
        # catalog.  This catches a name accidentally pointing at a nearby
        # sequential or diverging palette.
        expected_sources = {
            "Category20": "Category20",
            "Blues": "Blues256",
            "BuGn": "BuGn9",
            "Viridis": "Viridis256",
            "BrBG": "BrBG11",
            "Spectral": "Spectral11",
        }
        for name, source_name in expected_sources.items():
            source = getattr(bp, source_name)
            if isinstance(source, dict):
                source = source[max(source.keys())]
            assert catalog[name] == list(source)

        # The preview cannot be correct when two displayed names resolve to
        # the same sequence.  Keep this invariant explicit in the test.
        assert len({tuple(colors) for colors in catalog.values()}) == len(catalog)

    def test_palette_picker_preview_uses_the_full_named_palette(self):
        """A long palette preview must include both ends of its source."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.palette_picker import get_palette_catalog, palette_picker

        client = Client(page("/palette-picker-full-preview"))
        with client:
            picker = palette_picker("Palette", value="Blues")

        source = dict(get_palette_catalog())["Blues"]
        assert picker.get_colors() == source
        strips = [
            el for el in client.elements.values()
            if "drocat-palette-strip" in getattr(el, "_classes", [])
        ]
        preview_style = getattr(strips[-1], "_style", {})
        background = preview_style.get("background", "")
        assert source[0] in background
        assert source[-1] in background

    def test_palette_editor_set_palette_and_on_change(self):
        """palette_editor exposes set_palette() (programmatic, no callback)
        and fires on_change only when the user picks a palette in the
        dropdown."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.palette_picker import palette_editor

        changes = []
        client = Client(page("/palette-editor-api"))
        with client:
            editor = palette_editor(
                "Neuron Colors", value="Category10",
                on_change=lambda: changes.append("manual"),
            )
        assert editor.get_value() == "Category10"
        assert editor.get_colors()[0] == "#1f77b4"  # Category10 blue

        # Programmatic switch must not fire on_change
        editor.set_palette("Set3")
        assert editor.get_value() == "Set3"
        assert editor.get_colors()[0] == "#8dd3c7"  # Set3 teal
        assert changes == []

        # A user dropdown pick fires on_change
        palette_select = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Palette"
        )
        palette_select.value = "Dark2"
        assert editor.get_value() == "Dark2"
        assert changes == ["manual"]

    def test_palette_editor_preset_selector_uses_full_width_dropdown(self):
        """The preset palette picker is a full-width name dropdown (the card
        grid and the side preview strip are gone; the reorderable preview
        row shows the selected palette)."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.palette_picker import palette_editor

        client = Client(page("/palette-editor-preset-selector"))
        with client:
            editor = palette_editor("Neuron Colors", value="Category10")

        # No card grid remains in the preset panel.
        assert not any(
            "drocat-palette-card" in getattr(el, "_classes", [])
            for el in client.elements.values()
        )
        palette_select = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Palette"
        )
        assert palette_select.value == "Category10"
        assert "w-full" in getattr(palette_select, "_classes", [])

        # Picking another palette updates the state; the reorderable preview
        # row above shows the selected palette.
        palette_select.value = "Dark2"
        assert editor.get_value() == "Dark2"
        assert editor.get_colors() == editor.get_palette_order()
        assert any(
            "#1b9e77" in getattr(el, "_style", {}).get("background", "")
            for el in client.elements.values()
        )

    def test_palette_dropdowns_embed_strip_previews_in_options(self):
        """Both palette dropdowns embed a color strip beside each option name
        so the palette is visible before it is selected."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.palette_picker import (
            get_palette_catalog,
            palette_editor,
        )

        client = Client(page("/palette-editor-option-strips"))
        with client:
            palette_editor("Neuron Colors", value="Category10")

        catalog = dict(get_palette_catalog())
        selects = [
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") in {"Palette", "Bokeh palette"}
        ]
        assert len(selects) == 2
        for select in selects:
            options = select._props.get("options") or []
            assert options and all(
                isinstance(opt.get("label"), str)
                and "colors" in opt
                and "strip" in opt
                for opt in options
            )
            assert "option" in select.slots
            template = select.slots["option"].template
            assert "drocat-select-palette-strip" in template
            # NiceGUI compiles slot templates with the slot props under the
            # variable ``props``; a template using ``scope`` throws at
            # popup-open time and the dropdown can never expand.
            assert "props.itemProps" in template
            assert "props.opt.strip" in template
            # The name sits left of the strip preview inside each option row.
            assert template.index("props.opt.label") < template.index(
                "drocat-select-palette-strip"
            )
            dark2 = next(opt for opt in options if opt["label"] == "Dark2")
            assert dark2["colors"] == list(catalog["Dark2"])
            # Discrete palettes keep every color as distinct hard-stop
            # segments (no blurred gradient).
            assert "#1b9e77 0.0% 12.5%" in dark2["strip"]
            blues = next(opt for opt in options if opt["label"] == "Blues")
            assert len(blues["colors"]) == 12  # long palettes are sampled
            assert "%" not in blues["strip"]  # continuous: smooth gradient

        # Value semantics are unchanged: the option round-trips the name,
        # and the enriched options survive the programmatic value update.
        preset = next(s for s in selects if s._props.get("label") == "Palette")
        preset.value = "Dark2"
        assert preset.value == "Dark2"
        assert "colors" in preset._props["options"][0]

    def test_palette_editor_drag_reorder_reset_and_range(self):
        """palette_editor exposes ONE interactive preview row: drag & drop
        reorders discrete colors, the range slider slices the palette live,
        and the reset button (beside the preview) restores the original
        order and the full range."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.palette_picker import palette_editor, palette_slice

        changes = []
        client = Client(page("/palette-editor-interactive"))
        with client:
            editor = palette_editor(
                "Neuron Colors", value="Category10",
                on_change=lambda: changes.append("edit"),
            )
        original = list(editor.get_palette_order())
        assert len(original) == 10  # Category10 is discrete

        # ---- drag & drop: move color #0 before color #2 ----
        drop_rows = [
            el for el in client.elements.values()
            if any(
                l.type == "drop" and "clientX" in (l.js_handler or "")
                for l in el._event_listeners.values()
            )
        ]
        assert len(drop_rows) == 1, "expected exactly one horizontal drag target row"
        swatch_area = drop_rows[0]
        drop_listener = next(
            l for l in swatch_area._event_listeners.values() if l.type == "drop"
        )
        swatch_area._handle_event({
            "listener_id": drop_listener.id,
            "args": {"from": 0, "to": 2},
        })
        reordered = editor.get_palette_order()
        assert reordered[0:3] == [original[1], original[0], original[2]]
        assert editor.get_colors() == reordered  # full range = current state
        assert changes == ["edit"]  # manual edit locks the palette

        # ---- range slider: slice the palette live ----
        range_el = next(
            el for el in client.elements.values()
            if getattr(el, "tag", "") == "q-range"
        )
        range_listener = next(
            l for l in range_el._event_listeners.values()
            if l.type == "update:modelValue" and l.args is None
        )
        range_el._handle_event({
            "listener_id": range_listener.id,
            "args": {"min": 20, "max": 60},
        })
        assert editor.get_range() == (20, 60)
        assert editor.get_colors() == palette_slice(reordered, 20, 60)
        assert len(changes) == 2

        # Thumb-position bubbles are replaced by lateral end labels aligned
        # with the track ends; they update live with the range values.
        mono_labels = [
            el for el in client.elements.values()
            if "font-mono" in getattr(el, "_classes", [])
        ]
        assert len(mono_labels) == 2
        assert sorted(el.text for el in mono_labels) == ["20", "60"]
        assert "label-always" not in range_el._props

        # ---- reset restores the original order and the full range ----
        reset_button = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("aria-label") == "Reset palette"
        )
        click_listener = next(iter(reset_button._event_listeners.values()))
        click_listener.handler(None)  # ui.button on_click wraps with an event arg
        assert editor.get_palette_order() == original
        assert editor.get_range() == (0, 100)
        assert editor.get_colors() == original
        assert len(changes) == 3
        assert sorted(el.text for el in mono_labels) == ["0", "100"]

        # ---- the old full-palette preview and reorder editor are gone ----
        texts = [
            getattr(el, "text", "")
            for el in client.elements.values()
            if getattr(el, "text", "")
        ]
        assert "Reorder discrete colors" not in texts
        assert not any("Full palette preview" in text for text in texts)

    def test_palette_editor_custom_colors_drag_reorder(self):
        """Custom colors are added via the shared picker/text field and
        reordered in the horizontal preview row."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.palette_picker import palette_editor

        client = Client(page("/palette-editor-custom-drag"))
        with client:
            editor = palette_editor("Neuron Colors", value="Category10")

        # Switch to the custom-colors mode (setting .value fires the change
        # handlers exactly like the client-side model-value update)
        toggle = next(
            el for el in client.elements.values()
            if getattr(el, "tag", "") == "q-btn-toggle"
        )
        toggle.value = "Custom colors"

        # Add three colors via the format input + Add button.
        color_input = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Color format"
        )
        add_button = next(
            el for el in client.elements.values()
            if getattr(el, "text", "") == "Add color"
        )
        click_listener = next(iter(add_button._event_listeners.values()))
        for hex_color in ("#ff0000", "#00ff00", "#0000ff"):
            color_input.value = hex_color
            click_listener.handler(None)
        assert editor.get_custom_colors() == ["#ff0000", "#00ff00", "#0000ff"]
        assert editor.get_colors() == editor.get_custom_colors()  # preview = current state

        # Drag the first swatch to the third swatch's position (horizontal bar).
        drop_lists = [
            el for el in client.elements.values()
            if any(
                l.type == "drop" and "clientX" in (l.js_handler or "")
                for l in el._event_listeners.values()
            )
        ]
        assert drop_lists, "expected a horizontal drag target list"
        drop_list = drop_lists[-1]
        drop_listener = next(
            l for l in drop_list._event_listeners.values()
            if l.type == "drop" and "clientX" in (l.js_handler or "")
        )
        drop_list._handle_event({
            "listener_id": drop_listener.id,
            "args": {"from": 0, "to": 2},
        })
        assert editor.get_custom_colors() == ["#00ff00", "#ff0000", "#0000ff"]

        # The template palette strip and arrow/reverse controls are gone
        texts = [
            getattr(el, "text", "")
            for el in client.elements.values()
            if getattr(el, "text", "")
        ]
        assert "Reverse list" not in texts
        assert not any("Click the selected palette strip" in t for t in texts)
        # Removal is now attached to each horizontal swatch; no vertical list
        # of custom-color rows remains.
        remove_buttons = [
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("aria-label") == "Remove custom color"
        ]
        assert len(remove_buttons) >= 3
        assert not any(
            "drocat-custom-color-row" in getattr(el, "_classes", [])
            for el in client.elements.values()
        )

    def test_palette_editor_custom_mode_starts_with_empty_palette(self):
        """Custom mode shows no preset fallback: the custom palette stays
        empty until colors are added."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.palette_picker import palette_editor

        client = Client(page("/palette-editor-custom-empty"))
        with client:
            editor = palette_editor("Neuron Colors", value="Category10")

        # Preset mode starts with the full preset palette.
        assert editor.get_colors() == editor.get_palette_order()

        toggle = next(
            el for el in client.elements.values()
            if getattr(el, "tag", "") == "q-btn-toggle"
        )
        toggle.value = "Custom colors"

        # No preset fallback: the custom palette is empty until colors are
        # added, and the preview shows a placeholder instead of the preset.
        assert editor.get_mode() == "custom"
        assert editor.get_custom_colors() == []
        assert editor.get_colors() == []
        assert editor.get_colors_for_count(3) == []
        texts = [
            getattr(el, "text", "")
            for el in client.elements.values()
            if getattr(el, "text", "")
        ]
        assert any(
            "No custom colors yet — use Add color below." in text
            for text in texts
        )

        # Adding the first color fills the palette and the preview follows.
        format_input = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Color format"
        )
        add_button = next(
            el for el in client.elements.values()
            if getattr(el, "text", "") == "Add color"
        )
        format_input.value = "#ff0000"
        next(iter(add_button._event_listeners.values())).handler(None)
        assert editor.get_custom_colors() == ["#ff0000"]
        assert editor.get_colors() == ["#ff0000"]
        assert editor.get_colors_for_count(2) == ["#ff0000", "#ff0000"]

        # Switching back to the preset mode restores the preset palette.
        toggle.value = "Preset palette"
        assert editor.get_colors() == editor.get_palette_order()

    def test_skeleton_run_warns_on_empty_custom_palettes(self):
        """The shared helpers list the palettes whose empty custom mode would
        silently fall back to backend defaults, and emit the UI warning."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.palette_picker import (
            empty_custom_palette_names,
            notify_empty_custom_palettes,
            palette_editor,
        )

        client = Client(page("/skeleton-empty-custom-warning"))
        with client:
            neuron = palette_editor("Neuron Colors", value="Category10")
            synapse = palette_editor("Synapse Colors", value="Dark2")
            roi = palette_editor("ROI Colors", value="Cool", include_auto=True)

        # Switch the neuron editor to Custom colors (empty); the others stay
        # preset, so only the neuron palette is reported.
        toggle = next(
            el for el in client.elements.values()
            if getattr(el, "tag", "") == "q-btn-toggle"
        )
        toggle.value = "Custom colors"

        palettes = (
            (neuron, "Neuron Colors"),
            (synapse, "Synapse Colors"),
            (roi, "ROI Colors"),
        )
        assert empty_custom_palette_names(*palettes) == ["Neuron Colors"]

        # The notify wrapper emits exactly one warning mentioning the empty
        # palette and the default-palette fallback.
        with client:
            notify_empty_custom_palettes(*palettes)
        notifies = [
            options
            for _, kind, options in client.outbox.messages
            if kind == "notify"
        ]
        assert any(
            options.get("type") == "warning"
            and "Neuron Colors custom palette is empty" in options["message"]
            and "falls back to the default palette" in options["message"]
            for options in notifies
        )

        # A custom palette with colors is NOT reported as empty.
        format_input = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Color format"
        )
        add_button = next(
            el for el in client.elements.values()
            if getattr(el, "text", "") == "Add color"
        )
        format_input.value = "#ff0000"
        next(iter(add_button._event_listeners.values())).handler(None)
        assert empty_custom_palette_names(*palettes) == []

    def test_palette_editor_supports_free_form_alpha_overrides(self):
        """Custom entries keep typed alpha and popup alpha only when opted in."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.palette_picker import COLOR_FORMAT_HINT, palette_editor

        client = Client(page("/palette-editor-alpha-overrides"))
        with client:
            editor = palette_editor("Neuron Colors", value="Category10")

        toggle = next(
            el for el in client.elements.values()
            if getattr(el, "tag", "") == "q-btn-toggle"
        )
        toggle.value = "Custom colors"

        format_input = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Color format"
        )
        popup = editor.custom_color_popup
        alpha_input = popup.alpha
        alpha_toggle = popup.apply_alpha
        add_button = next(
            el for el in client.elements.values()
            if getattr(el, "text", "") == "Add color"
        )
        add_handler = next(iter(add_button._event_listeners.values())).handler

        assert "#RRGGBBAA" in COLOR_FORMAT_HINT
        assert "global opacity" in COLOR_FORMAT_HINT
        assert alpha_toggle.value is False

        # Explicit alpha in the typed format is kept as-is.
        format_input.value = "rgba(10, 20, 30, 0.25)"
        add_handler(None)
        # A color without alpha remains eligible for the renderer's global
        # opacity fallback until the popup's Apply alpha switch is checked.
        format_input.value = "#abcdef"
        add_handler(None)
        alpha_toggle.value = True
        alpha_input.value = 0.35
        popup._current = "#ff0000"
        popup._commit()
        add_handler(None)

        assert editor.get_custom_colors() == [
            "rgba(10, 20, 30, 0.25)",
            "#abcdef",
            "rgba(255, 0, 0, 0.35)",
        ]

    def test_palette_editor_bokeh_palette_picker_adds_with_alpha(self):
        """Custom mode picks individual colors from a Bokeh palette via the
        dropdown + swatch grid; the per-entry alpha override still applies."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.palette_picker import palette_editor

        client = Client(page("/palette-editor-bokeh-pick"))
        with client:
            editor = palette_editor("Neuron Colors", value="Category10")

        toggle = next(
            el for el in client.elements.values()
            if getattr(el, "tag", "") == "q-btn-toggle"
        )
        toggle.value = "Custom colors"

        palette_select = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Bokeh palette"
        )
        palette_select.value = "Dark2"

        # The swatch grid re-renders with the Dark2 colors; the first swatch
        # carries a click listener whose captured color is the palette's
        # first entry.
        swatches = [
            el for el in client.elements.values()
            if "drocat-palette-pick-swatch" in getattr(el, "_classes", [])
        ]
        assert len(swatches) == 8  # Dark2 is discrete with 8 colors
        first_swatch = swatches[0]
        assert "#1b9e77" in getattr(first_swatch, "_style", {}).get("background", "")
        next(iter(first_swatch._event_listeners.values())).handler(None)

        # Clicking selects (does not add yet) and syncs the format input.
        assert editor.get_custom_colors() == []
        format_input = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Color format"
        )
        assert format_input.value == "#1b9e77"
        highlighted = [
            el for el in client.elements.values()
            if "drocat-palette-pick-swatch" in getattr(el, "_classes", [])
            and "selected" in getattr(el, "_classes", [])
        ]
        assert len(highlighted) == 1
        assert "#1b9e77" in getattr(highlighted[0], "_style", {}).get("background", "")

        # The popup's explicit alpha applies to the picked color on Add.
        popup = editor.custom_color_popup
        alpha_toggle = popup.apply_alpha
        alpha_input = popup.alpha
        add_button = next(
            el for el in client.elements.values()
            if getattr(el, "text", "") == "Add color"
        )
        assert alpha_toggle.value is False
        alpha_toggle.value = True
        alpha_input.value = 0.5
        popup._current = "#1b9e77"
        popup._commit()
        next(iter(add_button._event_listeners.values())).handler(None)

        assert editor.get_custom_colors() == ["rgba(27, 158, 119, 0.5)"]
        assert editor.get_colors() == editor.get_custom_colors()  # preview = current state

        # The square preview beside the color value reflects the picked
        # color with the alpha override applied.
        preview_square = next(
            el for el in client.elements.values()
            if "drocat-custom-color-preview" in getattr(el, "_classes", [])
        )
        assert "rgba(27, 158, 119, 0.5)" in preview_square._style.get(
            "background", ""
        )

    def test_palette_editor_custom_color_preview_applies_alpha(self):
        """The custom palette preview follows the shared picker result."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.palette_picker import palette_editor

        client = Client(page("/palette-editor-color-preview"))
        with client:
            editor = palette_editor("Neuron Colors", value="Category10")

        toggle = next(
            el for el in client.elements.values()
            if getattr(el, "tag", "") == "q-btn-toggle"
        )
        toggle.value = "Custom colors"

        format_input = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Color format"
        )
        popup = editor.custom_color_popup
        alpha_toggle = popup.apply_alpha
        alpha_input = popup.alpha
        preview_square = next(
            el for el in client.elements.values()
            if "drocat-custom-color-preview" in getattr(el, "_classes", [])
        )

        # The preview follows the typed color without any alpha override.
        format_input.value = "#ff0000"
        assert "#ff0000" in preview_square._style.get("background", "")

        # Committing the popup with Apply alpha embeds the alpha value into the
        # format field and the live preview.
        popup._current = "#ff0000"
        alpha_toggle.value = True
        alpha_input.value = 0.5
        popup._commit()
        assert "rgba(255, 0, 0, 0.5)" in preview_square._style.get("background", "")

        # A second commit with Apply alpha off restores the raw color.
        alpha_input.value = 0.25
        alpha_toggle.value = False
        popup._commit()
        assert "#ff0000" in preview_square._style.get("background", "")

        # An unparseable value leaves the preview transparent (border only).
        format_input.value = "not-a-color"
        assert preview_square._style.get("background", "") == "transparent"

    def test_palette_editor_bokeh_palette_picker_swatch_grid(self):
        """The Bokeh swatch grid follows the selected palette: continuous
        palettes are sampled to 20 swatches and switching palettes clears
        the selected highlight."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.palette_picker import get_palette_catalog, palette_editor

        client = Client(page("/palette-editor-bokeh-grid"))
        with client:
            palette_editor("Neuron Colors", value="Category10")

        toggle = next(
            el for el in client.elements.values()
            if getattr(el, "tag", "") == "q-btn-toggle"
        )
        toggle.value = "Custom colors"

        palette_select = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Bokeh palette"
        )

        def pick_swatches():
            return [
                el for el in client.elements.values()
                if "drocat-palette-pick-swatch" in getattr(el, "_classes", [])
            ]

        # Default grid renders the discrete Category10 palette in full.
        assert len(pick_swatches()) == 10

        # A continuous 256-color palette is sampled to 20 selectable
        # swatches spanning the full gradient (both ends present).
        palette_select.value = "Blues"
        swatches = pick_swatches()
        assert len(swatches) == 20
        blues = dict(get_palette_catalog())["Blues"]
        assert blues[0] in getattr(swatches[0], "_style", {}).get("background", "")
        assert blues[-1] in getattr(swatches[-1], "_style", {}).get("background", "")

        # Selecting a swatch highlights it; switching the palette clears it.
        next(iter(swatches[5]._event_listeners.values())).handler(None)
        highlighted = [
            el for el in pick_swatches()
            if "selected" in getattr(el, "_classes", [])
        ]
        assert len(highlighted) == 1
        palette_select.value = "Dark2"
        assert not any(
            "selected" in getattr(el, "_classes", []) for el in pick_swatches()
        )
        assert len(pick_swatches()) == 8

    def test_palette_editor_long_palette_uses_gradient_preview(self):
        """Long (sequential) palettes render as a gradient strip without
        drag targets; the range slider still slices them."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.palette_picker import (
            palette_editor,
            palette_slice,
            sample_palette,
        )

        client = Client(page("/palette-editor-gradient"))
        with client:
            editor = palette_editor("Neuron Colors", value="Blues")
        assert len(editor.get_palette_order()) > 20
        # no horizontal drag target: the preview is a plain gradient strip
        drop_rows = [
            el for el in client.elements.values()
            if any(
                l.type == "drop" and "clientX" in (l.js_handler or "")
                for l in el._event_listeners.values()
            )
        ]
        assert drop_rows == []
        # a long palette still exposes the range + reset controls
        assert any(
            getattr(el, "tag", "") == "q-range"
            for el in client.elements.values()
        )
        assert any(
            getattr(el, "_props", {}).get("aria-label") == "Reset palette"
            for el in client.elements.values()
        )

        # The range must be rendered as the complete selected gradient, not
        # truncated to the first 32 source colors.
        range_el = next(
            el for el in client.elements.values()
            if getattr(el, "tag", "") == "q-range"
        )
        range_listener = next(
            listener for listener in range_el._event_listeners.values()
            if listener.type == "update:modelValue" and listener.args is None
        )
        range_el._handle_event({
            "listener_id": range_listener.id,
            "args": {"min": 20, "max": 60},
        })
        selected = palette_slice(editor.get_palette_order(), 20, 60)
        assert editor.get_colors() == selected
        assert editor.get_colors_for_count(3) == sample_palette(selected, 3)

        preview_strips = [
            el for el in client.elements.values()
            if "drocat-palette-strip" in getattr(el, "_classes", [])
        ]
        preview_style = getattr(preview_strips[-1], "_style", {})
        assert selected[0] in preview_style.get("background", "")
        assert selected[-1] in preview_style.get("background", "")

    def test_shared_visualization_settings_preserve_continuous_palette_metadata(self):
        from ui.components.skeleton_visualization_settings import (
            SkeletonVisualizationSettings,
        )

        class PaletteField:
            value = None

            def get_colors(self):
                from ui.components.palette_picker import _PaletteSelection

                return _PaletteSelection(
                    ["#000000", "#ffffff"],
                    continuous=True,
                )

        values = SkeletonVisualizationSettings(
            fields={"neuron_colors": PaletteField()}
        ).values()
        assert values["neuron_colors"] == {
            "colors": ["#000000", "#ffffff"],
            "continuous": True,
        }

    def test_roi_mesh_traces_have_independent_legend_entries(self):
        """Each resolved ROI stays separately toggleable in the Plotly legend."""
        import plotly.graph_objects as go

        from visualize_skeleton import VisualizeSkeleton, _configure_roi_mesh_traces

        left_traces = (go.Mesh3d(), go.Scatter3d())
        right_traces = (go.Mesh3d(),)
        _configure_roi_mesh_traces(left_traces, "LH(L)")
        _configure_roi_mesh_traces(right_traces, "LH(R)")

        assert left_traces[0].legendgroup == "roi_mesh:LH(L)"
        assert right_traces[0].legendgroup == "roi_mesh:LH(R)"
        assert left_traces[0].legendgroup != right_traces[0].legendgroup
        assert [trace.showlegend for trace in left_traces] == [True, False]
        assert right_traces[0].showlegend is True
        assert left_traces[0].name == "brain region [LH(L)]"
        assert right_traces[0].name == "brain region [LH(R)]"

        visualizer = object.__new__(VisualizeSkeleton)
        expanded_rois, expanded_colors = visualizer._expand_roi_names_with_colors(
            ["LH", "EB"],
            ["#112233", "#abcdef"],
            available_rois=["LH(L)", "LH(R)", "EB"],
        )
        assert expanded_rois == ["LH(L)", "LH(R)", "EB"]
        assert expanded_colors == ["#112233", "#112233", "#abcdef"]

    def test_dir_browser_is_in_browser_not_tkinter(self, tmp_path):
        """Directory browsing uses the in-browser dialog (server-side folder
        listing), NOT the blocking tkinter dialog that could hang and freeze
        the whole app. The tkinter helpers are gone from the runner."""
        import ui.runner as runner_mod
        assert not hasattr(runner_mod, "pick_directory")
        assert not hasattr(runner_mod, "pick_file")

        from ui.components.common import (
            _list_subdirs,
            _native_directory_picker_sync,
            dir_browser_dialog,
            native_directory_picker,
        )
        assert callable(dir_browser_dialog)
        assert callable(native_directory_picker)
        assert callable(_native_directory_picker_sync)
        # folder listing helper: sorted subdirs only, tolerant of bad paths
        a = tmp_path / "b_dir"; a.mkdir()
        (tmp_path / "a_dir").mkdir()
        (tmp_path / "a_file.txt").write_text("x")
        assert _list_subdirs(str(tmp_path)) == ["a_dir", "b_dir"]
        assert _list_subdirs(str(tmp_path / "nope")) == []
        assert _list_subdirs(str(tmp_path / "a_file.txt")) == []

    def test_native_directory_picker_uses_osascript(self, tmp_path, monkeypatch):
        """macOS uses the native choose-folder panel through osascript
        (isolated behind a platform adapter) so no real desktop dialog opens
        during tests."""
        import ui.components.common as common

        calls = []

        class Completed:
            returncode = 0
            stdout = str(tmp_path) + "\n"
            stderr = ""

        monkeypatch.setattr(common.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(common.shutil, "which", lambda name: "/usr/bin/osascript")

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            return Completed()

        monkeypatch.setattr(common.subprocess, "run", fake_run)
        available, selected = common._native_directory_picker_sync(
            "Choose output", str(tmp_path)
        )
        assert available is True
        assert selected == str(tmp_path)
        assert calls and calls[0][0][0] == "/usr/bin/osascript"
        assert "choose folder" in calls[0][0][-1]

    def test_macos_picker_unavailable_without_osascript(self, tmp_path, monkeypatch):
        """Without osascript, macOS reports the picker as unavailable so
        callers fall back to the in-browser dialog."""
        import ui.components.common as common

        monkeypatch.setattr(common.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(common.shutil, "which", lambda name: None)
        available, selected = common._native_directory_picker_sync(
            "Choose output", str(tmp_path)
        )
        assert available is False
        assert selected is None

    def test_macos_picker_instant_cancel_reports_unavailable(self, tmp_path, monkeypatch):
        """An instant -128 (no dialog could have been shown and dismissed
        that fast) means the panel failed to appear; report unavailable so
        the caller falls back to the in-browser dialog."""
        import ui.components.common as common

        class InstantCancel:
            returncode = 1
            stdout = ""
            stderr = "execution error: User canceled. (-128)"

        monkeypatch.setattr(common.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(common.shutil, "which", lambda name: "/usr/bin/osascript")
        monkeypatch.setattr(
            common.subprocess, "run", lambda *a, **k: InstantCancel()
        )
        available, selected = common._native_directory_picker_sync(
            "Choose output", str(tmp_path)
        )
        assert available is False
        assert selected is None

    def test_macos_picker_slow_cancel_returns_none(self, tmp_path, monkeypatch):
        """A cancel after the dialog was shown (elapsed >= 2s) is a genuine
        user cancel: supported picker, no path, no second dialog."""
        import time
        import ui.components.common as common

        class SlowCancel:
            returncode = 1
            stdout = ""
            stderr = "execution error: User canceled. (-128)"

        def fake_run(command, **kwargs):
            time.sleep(2.5)
            return SlowCancel()

        monkeypatch.setattr(common.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(common.shutil, "which", lambda name: "/usr/bin/osascript")
        monkeypatch.setattr(common.subprocess, "run", fake_run)
        available, selected = common._native_directory_picker_sync(
            "Choose output", str(tmp_path)
        )
        assert available is True
        assert selected is None

    def test_windows_native_directory_picker_uses_folder_browser_dialog(
        self, tmp_path, monkeypatch
    ):
        """Windows uses the STA PowerShell folder dialog adapter."""
        import ui.components.common as common

        calls = []

        class Completed:
            returncode = 0
            stdout = str(tmp_path) + "\n"
            stderr = ""

        powershell = r"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"
        monkeypatch.setattr(common.platform, "system", lambda: "Windows")
        monkeypatch.setattr(
            common.shutil,
            "which",
            lambda name: powershell if name == "powershell" else None,
        )

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            return Completed()

        monkeypatch.setattr(common.subprocess, "run", fake_run)
        available, selected = common._native_directory_picker_sync(
            "Choose output", str(tmp_path)
        )

        assert available is True
        assert selected == str(tmp_path)
        assert calls and calls[0][0][:3] == [powershell, "-NoProfile", "-STA"]
        assert "FolderBrowserDialog" in calls[0][0][-1]
        assert "SelectedPath" in calls[0][0][-1]

    def test_dir_input_uses_one_direct_system_picker_button(self):
        """The output field exposes one fixed folder action that opens the
        desktop picker directly; no second in-app browse layer is rendered."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.find_path import create_find_path_tab

        client = Client(page("/dir-input-browse"))
        with client:
            create_find_path_tab()

        elements = list(client.elements.values())
        icons = [getattr(el, "_props", {}).get("icon") for el in elements]
        folder_buttons = [
            el for el in elements
            if getattr(el, "_props", {}).get("icon") == "folder_open"
        ]
        assert len(folder_buttons) == 1
        assert "drocat-dir-icon-btn" in getattr(folder_buttons[0], "_classes", [])
        assert getattr(folder_buttons[0], "_props", {}).get("round") is True
        assert "folder_tree" not in icons

        # The common neuron-input builder is used by every pathfinding input;
        # all visible Clear actions must remain text-only.
        clear_buttons = [el for el in elements if getattr(el, "text", "") == "Clear"]
        assert clear_buttons
        assert all("icon" not in getattr(el, "_props", {}) for el in clear_buttons)

    def test_pathfinding_tabs_wire_auto_suggest(self):
        """All four pathfinding tabs pass a suggestions provider to their
        neuron inputs: the native popup is replaced by the custom suggestion
        menu (popup-content-class + focus/history wiring) and the filter-mode
        dropdown is labeled 'Match by'."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.find_path import create_find_path_tab
        from ui.tabs.find_shortest import create_find_shortest_tab
        from ui.tabs.inter_dataset import create_inter_dataset_tab
        from ui.tabs.network import create_network_tab

        builders = [
            ("/wire-find-path", create_find_path_tab, 2),
            ("/wire-find-shortest", create_find_shortest_tab, 2),
            ("/wire-network", create_network_tab, 1),
            ("/wire-inter-dataset", create_inter_dataset_tab, 2),
        ]
        for route, builder, expected_viewer_links in builders:
            client = Client(page(route))
            with client:
                builder()
            inputs = [
                el for el in client.elements.values()
                if getattr(el, "chip_input", None) is not None
                and getattr(el, "filter_mode", None) is not None
            ]
            assert inputs, f"{route}: no neuron inputs found"
            for inp in inputs:
                chip = inp.chip_input
                # The suggestion machinery is only wired when a provider was
                # passed — its popup suppression marks the wired inputs.
                assert chip._props.get("popup-content-class") == \
                    "drocat-native-popup-hidden", route
                event_types = [l.type for l in chip._event_listeners.values()]
                assert "focus" in event_types, route  # history-on-focus
                assert sum(t == "input" for t in event_types) >= 2, route
            labels = [
                getattr(el, "_props", {}).get("label")
                for el in client.elements.values()
            ]
            assert "Match by" in labels, route
            viewer_links = [
                el for el in client.elements.values()
                if getattr(el, "text", "") == "See available neurons"
            ]
            assert len(viewer_links) == expected_viewer_links, route
            assert all(
                hasattr(link, "neuron_index_dialog") for link in viewer_links
            )

    def test_shortest_source_type_warning_recommends_small_source_set(self):
        """Warn in-page when Shortest Paths receives multiple source types."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.find_shortest import (
            _has_multiple_source_type_queries,
            create_find_shortest_tab,
        )

        assert _has_multiple_source_type_queries(["L2", "aMe12"], "auto")
        assert _has_multiple_source_type_queries(["L2", "aMe12"], "type")
        assert not _has_multiple_source_type_queries(["L2", "aMe12"], "bodyId")
        assert not _has_multiple_source_type_queries([123, 456], "auto")
        assert not _has_multiple_source_type_queries(
            ["all_neurons", "L2"], "auto"
        )

        client = Client(page("/shortest-source-set-warning"))
        with client:
            create_find_shortest_tab()

        source = next(
            element for element in client.elements.values()
            if getattr(element, "chip_input", None) is not None
            and element.chip_input._props.get("label") == "Source Neurons"
        )
        warning = next(
            element for element in client.elements.values()
            if "small source set" in str(getattr(element, "text", ""))
        )
        assert "bodyId" in warning.text
        assert "hidden" in getattr(warning, "_classes", [])

        value_listener = next(
            listener for listener in source.chip_input._event_listeners.values()
            if listener.type == "update:modelValue"
        )
        source.chip_input.set_value(["L2", "aMe12"])
        source.chip_input._handle_event(
            {"listener_id": value_listener.id, "args": ["L2", "aMe12"]}
        )
        assert "hidden" not in getattr(warning, "_classes", [])

        source.chip_input.set_value(["L2"])
        source.chip_input._handle_event(
            {"listener_id": value_listener.id, "args": ["L2"]}
        )
        assert "hidden" in getattr(warning, "_classes", [])

    def test_inter_dataset_suggestions_are_dataset_aware(self, monkeypatch):
        """The cross-dataset tab's suggestion provider receives the datasets
        selected in the tab and renders dataset-aware gray hints
        (``type · male-cns:v1.0``) in the dropdown rows."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.inter_dataset import create_inter_dataset_tab

        calls = []

        def fake_suggest(text, datasets, search_columns="auto", *, limit=None):
            calls.append((text, list(datasets), search_columns))
            return [("aMe12", "type · male-cns:v1.0 · hemibrain:v1.2.1")]

        monkeypatch.setattr(
            "ui.tabs.inter_dataset.dataset_aware_suggestions", fake_suggest
        )
        client = Client(page("/inter-dataset-aware-suggest"))
        with client:
            create_inter_dataset_tab()

        source = next(
            el for el in client.elements.values()
            if getattr(el, "chip_input", None) is not None
            and el.chip_input._props.get("label") == "Source Neurons"
        )
        suggest_input = next(
            listener for listener in
            source.chip_input._event_listeners.values()
            if listener.type == "input"
        )
        source.chip_input._handle_event(
            {"listener_id": suggest_input.id, "args": "a"}
        )

        # The provider got the typed text plus the tab's selected datasets.
        assert calls and calls[-1][0] == "a"
        assert calls[-1][1] == ["male-cns:v1.0"]  # the tab's default selection
        assert calls[-1][2] == "auto"
        texts = [el.text for el in client.elements.values()
                 if getattr(el, "text", "")]
        assert "type · male-cns:v1.0 · hemibrain:v1.2.1" in texts


# =============================================================================
# Test Dataset Service
# =============================================================================
class TestDatasetService:
    def test_service_imports(self):
        from ui.dataset_service import get_dataset_service, DatasetInfo, folder_to_dataset, dataset_to_folder
        assert callable(get_dataset_service)

    def test_availability_snapshot_persists_and_refresh_overwrites(self, tmp_path):
        """A refresh persists the server dimension used by the next session."""
        from ui.dataset_service import (
            SERVER_AVAILABLE,
            SERVER_UNREACHABLE,
            DatasetService,
        )

        service = DatasetService()
        service._cache_dir = tmp_path / "cache"
        state = {"server": SERVER_UNREACHABLE}

        service._probe_server = lambda dataset: {
            "state": state["server"],
            "checked_at": "2026-09-10T00:00:00+08:00",
            "metadata": {},
        }
        service.refresh_availability(["demo:v1.0"])
        first, first_updated = service.get_cached_availability()
        assert first["demo:v1.0"].server_state == SERVER_UNREACHABLE
        assert first["demo:v1.0"].available is False
        assert first_updated

        state["server"] = SERVER_AVAILABLE
        service.refresh_availability(["demo:v1.0"])
        second, second_updated = service.get_cached_availability()
        assert second["demo:v1.0"].server_state == SERVER_AVAILABLE
        assert second["demo:v1.0"].available is True
        assert second_updated

        # The persisted file holds only the server dimension.
        import json
        payload = json.loads(service.availability_cache_path.read_text())
        assert payload["format"] == service.AVAILABILITY_CACHE_FORMAT
        assert payload["servers"]["demo:v1.0"]["state"] == SERVER_AVAILABLE
        assert "datasets" not in payload

        next_session = DatasetService()
        next_session._cache_dir = service._cache_dir
        persisted, persisted_updated = next_session.get_cached_availability()
        assert persisted["demo:v1.0"].server_state == SERVER_AVAILABLE
        assert persisted["demo:v1.0"].available is True
        assert persisted_updated == second_updated

    def test_targeted_refresh_merges_and_full_refresh_replaces(self, tmp_path):
        """A targeted refresh updates only the probed rows; a full refresh
        (``None``) is authoritative and replaces the server dimension."""
        from ui.dataset_service import DatasetService

        service = DatasetService()
        service._cache_dir = tmp_path / "cache"
        service._datasets_dir = tmp_path / "datasets"
        service._index_dir = tmp_path / "neuron_indexes"
        service._probe_server = lambda ds: {
            "state": "available", "checked_at": "t", "metadata": {}}

        service.refresh_availability(["hemibrain:v1.2.1", "male-cns:v1.0"])
        assert set(service._server_rows) == {"hemibrain:v1.2.1", "male-cns:v1.0"}

        # Targeted refresh of one row preserves the other.
        service._probe_server = lambda ds: {
            "state": "timeout", "checked_at": "t2", "metadata": {}}
        service.refresh_availability(["male-cns:v1.0"])
        assert service._server_rows["male-cns:v1.0"]["state"] == "timeout"
        assert service._server_rows["hemibrain:v1.2.1"]["state"] == "available"

        # A full refresh replaces the whole server dimension.
        service._probe_server = lambda ds: {
            "state": "unreachable", "checked_at": "t3", "metadata": {}}
        service.refresh_availability()
        assert service._server_rows["hemibrain:v1.2.1"]["state"] == "unreachable"
        assert service._server_rows["male-cns:v1.0"]["state"] == "unreachable"

    def test_folder_to_dataset_conversion(self):
        from ui.dataset_service import folder_to_dataset
        assert folder_to_dataset("hemibrain_v1_2_1") == "hemibrain:v1.2.1"
        assert folder_to_dataset("male-cns_v0_9") == "male-cns:v0.9"
        assert folder_to_dataset("flywire_FAFB_v783") == "flywire_FAFB_v783"
        assert folder_to_dataset("optic-lobe_v1_1") == "optic-lobe:v1.1"

    def test_dataset_to_folder_conversion(self):
        from ui.dataset_service import dataset_to_folder
        assert dataset_to_folder("hemibrain:v1.2.1") == "hemibrain_v1_2_1"
        assert dataset_to_folder("male-cns:v0.9") == "male-cns_v0_9"
        assert dataset_to_folder("flywire_FAFB_v783") == "flywire_FAFB_v783"

    def test_flywire_identifier_does_not_match_neuprint_banc(self):
        from ui.dataset_service import is_banc_dataset, is_flywire_dataset
        assert is_flywire_dataset("banc_v626") is False
        assert is_banc_dataset("banc_v626") is True
        assert is_flywire_dataset("flywire_FAFB_v783") is True
        assert is_flywire_dataset("banc:v888") is False

    def test_service_get_token(self):
        from ui.dataset_service import get_dataset_service
        service = get_dataset_service()
        # Token may or may not be present
        token = service.get_token()
        assert token is None or isinstance(token, str)

    def test_settings_token_loader_reads_config_json(self, tmp_path, monkeypatch):
        """_load_tokens reads the config.json tokens section per key."""
        from ui.tabs import settings as settings_module

        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "cfg-np", "cave": ""}}\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)

        assert settings_module._load_tokens() == {"neuprint": "cfg-np"}

    def test_settings_token_loader_ignores_legacy_files(self, tmp_path, monkeypatch):
        """token_info files are deprecated; only config.json is read."""
        from ui.tabs import settings as settings_module

        (tmp_path / "token_info.txt").write_text(
            "NEUPRINT_TOKEN='template-neuprint'\nCAVE_TOKEN='template-cave'\n",
            encoding="utf-8",
        )
        (tmp_path / "token_info_local.txt").write_text(
            "NEUPRINT_TOKEN='local-np'\nCAVE_TOKEN='local-cave'\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)

        assert settings_module._load_tokens() == {}

    def test_settings_token_loader_prefers_config_json(self, tmp_path, monkeypatch):
        """config.json wins per key; config_local.json only fills empties."""
        from ui.tabs import settings as settings_module

        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "cfg-np", "cave": "cfg-cave"}}\n',
            encoding="utf-8",
        )
        (tmp_path / "config_local.json").write_text(
            '{"tokens": {"neuprint": "local-np"}}\n', encoding="utf-8"
        )
        monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)

        assert settings_module._load_tokens() == {
            "neuprint": "cfg-np",
            "cave": "cfg-cave",
        }

    def test_settings_token_loader_config_local_fills_empty(self, tmp_path, monkeypatch):
        """An empty config.json entry falls back to config_local.json."""
        from ui.tabs import settings as settings_module

        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "", "cave": "cfg-cave"}}\n',
            encoding="utf-8",
        )
        (tmp_path / "config_local.json").write_text(
            '{"tokens": {"neuprint": "local-np"}}\n', encoding="utf-8"
        )
        monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)

        assert settings_module._load_tokens() == {
            "neuprint": "local-np",
            "cave": "cfg-cave",
        }

    def test_settings_save_tokens_writes_config_local_and_keeps_envs(self, tmp_path, monkeypatch):
        """Save Tokens writes the gitignored config_local.json and preserves
        the versioned env map; the committed config.json stays untouched."""
        import json as _json
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs import settings as settings_module

        (tmp_path / "config.json").write_text(
            _json.dumps({"tokens": {"neuprint": "committed-np", "cave": ""}, "envs": {"4.5.0": ""}}),
            encoding="utf-8",
        )
        (tmp_path / "config_local.json").write_text(
            _json.dumps({"envs": {"4.5.0": "my-custom-env"}, "tokens": {"neuprint": "", "cave": ""}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)

        client = Client(page("/settings-save-tokens-config-local"))
        with client:
            settings_module.create_settings_tab()

        neuprint_input = next(
            el for el in client.elements.values()
            if (getattr(el, "_props", {}).get("label") == "NeuPrint Token")
        )
        neuprint_input.value = "saved-np"
        save_btn = next(
            el for el in client.elements.values()
            if getattr(el, "text", "") == "Save Tokens"
        )
        next(iter(save_btn._event_listeners.values())).handler(None)

        saved = _json.loads((tmp_path / "config_local.json").read_text(encoding="utf-8"))
        assert saved["tokens"] == {"neuprint": "saved-np", "cave": ""}
        # the versioned env map must survive a token save
        assert saved["envs"] == {"4.5.0": "my-custom-env"}
        # the committed config.json is never rewritten
        committed = _json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert committed["tokens"] == {"neuprint": "committed-np", "cave": ""}

    def test_settings_save_tokens_preserves_envs_from_bom_config(self, tmp_path, monkeypatch):
        """A config_local.json written by Windows PowerShell (BOM + autofilled
        envs) must keep its envs section through a token save: reading it with
        plain utf-8 would fail, silently dropping the section."""
        import json as _json
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs import settings as settings_module

        (tmp_path / "config.json").write_text(
            _json.dumps({"tokens": {"neuprint": "", "cave": ""}, "envs": {"4.5.0": ""}}),
            encoding="utf-8",
        )
        (tmp_path / "config_local.json").write_bytes(
            b"\xef\xbb\xbf"
            + _json.dumps(
                {"tokens": {"neuprint": "", "cave": ""}, "envs": {"4.5.0": "drocat-4.5.0"}}
            ).encode("utf-8")
        )
        monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)

        client = Client(page("/settings-save-tokens-bom"))
        with client:
            settings_module.create_settings_tab()

        neuprint_input = next(
            el for el in client.elements.values()
            if (getattr(el, "_props", {}).get("label") == "NeuPrint Token")
        )
        neuprint_input.value = "bom-saved"
        save_btn = next(
            el for el in client.elements.values()
            if getattr(el, "text", "") == "Save Tokens"
        )
        next(iter(save_btn._event_listeners.values())).handler(None)

        saved = _json.loads((tmp_path / "config_local.json").read_text(encoding="utf-8-sig"))
        assert saved["tokens"] == {"neuprint": "bom-saved", "cave": ""}
        assert saved["envs"] == {"4.5.0": "drocat-4.5.0"}

    def test_settings_dataset_cache_card(self):
        """The Settings tab exposes full-dataset and complete-connection pulls:
        a dataset selector, a force-rebuild option, run/cancel buttons, and a
        progress element - all wired to a DatasetPuller."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.settings import create_settings_tab

        client = Client(page("/settings-dataset-cache"))
        with client:
            create_settings_tab()

        labels = {}
        for el in client.elements.values():
            props = getattr(el, "_props", {})
            label = props.get("label")
            text = getattr(el, "text", None)
            if label:
                labels[label] = el
            if text == "Dataset Cache":
                labels["card_title"] = el

        assert "card_title" in labels
        assert "Dataset" in labels  # dataset selector
        assert labels["Dataset"]._props.get("outlined") is True
        assert "Batch size" in labels
        assert "Parallel workers" in labels
        assert "Cache Simplification" in labels  # skeleton pull level selector
        assert "Skeleton Batch Size" in labels  # skeleton pull batch control
        assert any(
            getattr(el, "text", None) == "Pull Dataset Metadata"
            for el in client.elements.values()
        )
        assert any(
            getattr(el, "text", None) == "Pull Complete Connections"
            for el in client.elements.values()
        )
        assert any(
            getattr(el, "text", None) == "Cancel"
            for el in client.elements.values()
        )
        assert any(
            getattr(el, "text", None) == "Force rebuild (clear broken cache first)"
            for el in client.elements.values()
        )
        # progress element + status label
        assert any(
            getattr(el, "text", None) == "Idle"
            for el in client.elements.values()
        )
        assert not any(
            getattr(el, "_props", {}).get("label") == "Skeleton cache"
            for el in client.elements.values()
        )
        elements = list(client.elements.values())
        download_button = next(
            el for el in elements if getattr(el, "text", None) == "Download All Skeletons"
        )
        dataset_cache_title = labels["card_title"]
        cancel_button = next(
            el for el in elements if getattr(el, "text", None) == "Cancel"
        )
        assert elements.index(dataset_cache_title) < elements.index(download_button)
        # The skeleton pull controls (simplification selector + button) sit
        # below the dataset-pull buttons and progress, grouped with the
        # skeleton progress bar - so Cancel (a dataset-row button) precedes
        # the skeleton download button in the element order.
        assert elements.index(cancel_button) < elements.index(download_button)

    def test_settings_skeleton_pull_blocks_flywire_with_manual_instruction(self):
        """Clicking 'Download All Skeletons' on a FlyWire dataset must not
        start a pull; it shows the manual Codex instruction instead.

        Regression: the guard imported ``is_flywire_dataset`` from
        utils.flywire_readiness, which only defines is_banc/is_fafb, so the
        click crashed with ImportError before showing the instruction.
        """
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs import settings as settings_module

        client = Client(page("/settings-skeleton-flywire-guard"))
        with client:
            settings_module.create_settings_tab()

        # Pick a FlyWire dataset in the shared dataset selector.
        dataset_select = next(
            el for el in client.elements.values()
            if (getattr(el, "_props", {}).get("label") == "Dataset")
        )
        dataset_select.value = "flywire_FAFB_v783"

        skeleton_btn = next(
            el for el in client.elements.values()
            if getattr(el, "text", None) == "Download All Skeletons"
        )
        next(iter(skeleton_btn._event_listeners.values())).handler(None)

        # The guard ran (no ImportError) and displayed the manual instruction.
        result_texts = [
            el.text for el in client.elements.values()
            if getattr(el, "text", "") and el.text.startswith("❌")
        ]
        assert any(
            "sk_lod1_783_healed.zip" in text and "codex.flywire.ai" in text
            for text in result_texts
        ), result_texts
        # The status line switched to the manual-download state (the guard
        # returned before any pull could start).
        assert any(
            getattr(el, "text", "") == "Manual download required (FAFB)"
            for el in client.elements.values()
        )

    def test_settings_cancel_shows_cancelling_hint_until_wind_down_ends(
            self, monkeypatch):
        """After Cancel, the status must show a persistent 'Cancelling...'
        hint (with an indeterminate bar) while the puller is still winding
        down (running + cancel_requested), instead of the frozen progress
        text — for both the connection pull and the skeleton pull."""
        from nicegui import Client
        from nicegui.page import page
        import ui.dataset_pull as dataset_pull_module
        import ui.skeleton_pull as skeleton_pull_module
        from ui.tabs import settings as settings_module

        class FakePuller:
            def __init__(self):
                self._state = {
                    "running": False, "dataset": "male-cns:v1.0",
                    "operation": "connections", "phase": "fetch",
                    "current": 100, "total": 1000, "info": "Fetching...",
                    "done": False, "cancelled": False,
                    "cancel_requested": False, "error": None,
                    "summary": None, "started_at": None,
                    "fetch_started_at": None,
                }

            @property
            def state(self):
                return dict(self._state)

            @property
            def running(self):
                return self._state["running"]

            def start(self, *args, **kwargs):
                return True

            def cancel(self):
                self._state["cancel_requested"] = True

        dataset_fake = FakePuller()
        skeleton_fake = FakePuller()
        monkeypatch.setattr(
            dataset_pull_module, "DatasetPuller", lambda: dataset_fake
        )
        monkeypatch.setattr(
            skeleton_pull_module, "SkeletonPuller", lambda: skeleton_fake
        )

        client = Client(page("/settings-cancel-hint"))
        with client:
            settings_module.create_settings_tab()

        timers = [
            el for el in client.elements.values()
            if type(el).__name__ == "Timer" and getattr(el, "callback", None)
        ]
        assert timers, "refresh timers missing"

        def drive_refresh():
            for timer in timers:
                timer.callback()

        def label_texts():
            return [
                el.text for el in client.elements.values()
                if isinstance(getattr(el, "text", None), str)
            ]

        # Dataset pull winding down after cancel.
        dataset_fake._state["running"] = True
        dataset_fake._state["cancel_requested"] = True
        drive_refresh()
        texts = label_texts()
        assert any(
            "Cancelling complete connections pull" in text and "male-cns:v1.0" in text
            for text in texts
        ), texts
        assert any(
            "stopping after the current batch" in text for text in texts
        ), texts

        # Skeleton pull winding down after cancel.
        dataset_fake._state["running"] = False
        skeleton_fake._state["running"] = True
        skeleton_fake._state["cancel_requested"] = True
        drive_refresh()
        texts = label_texts()
        assert any(
            "Cancelling skeleton pull" in text and "male-cns:v1.0" in text
            for text in texts
        ), texts

        # Wind-down finished: the normal result/status rendering resumes.
        skeleton_fake._state["running"] = False
        skeleton_fake._state["done"] = True
        skeleton_fake._state["cancelled"] = True
        skeleton_fake._state["summary"] = {
            "total": 10, "fetched": 4, "skipped_existing": 0,
            "cancelled": True, "errors": 6,
        }
        drive_refresh()
        texts = label_texts()
        assert any("⏹ Cancelled" in text for text in texts), texts

    def test_settings_shows_availability_timestamp_and_shared_mapping_panel(
        self, tmp_path, monkeypatch
    ):
        """Settings reuses Custom Mapping and renders the persisted refresh time."""
        from nicegui import Client
        from nicegui.page import page
        from ui import dataset_service as dataset_service_module
        from ui.dataset_service import DatasetInfo, DatasetService
        from ui.tabs import settings as settings_module

        service = DatasetService()
        service._cache_dir = tmp_path / "cache"
        # demo:v1.0 is not a real dataset family; stub the server probe so
        # refresh persists a server row for it.
        service._probe_server = lambda dataset: {
            "state": "available",
            "checked_at": "2026-09-10T00:00:00+08:00",
            "metadata": {"server": "test"},
        }
        service.refresh_availability(["demo:v1.0"])
        monkeypatch.setattr(dataset_service_module, "get_dataset_service", lambda: service)
        monkeypatch.setattr(settings_module, "get_dataset_service", lambda: service)

        client = Client(page("/settings-availability-mapping"))
        with client:
            settings_module.create_settings_tab()

        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert any(text.startswith("Last updated at:") for text in texts)
        mapping_labels = [
            getattr(el, "_props", {}).get("label")
            for el in client.elements.values()
        ]
        assert "Saved mappings" not in mapping_labels
        assert "Mapping name" not in mapping_labels
        assert "Description (optional)" not in mapping_labels
        assert not any(
            getattr(el, "text", "") in {"New", "Rename", "Delete", "Set Active"}
            for el in client.elements.values()
        )
        target_dataset_selects = [
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Target datasets"
        ]
        assert len(target_dataset_selects) == 1
        assert target_dataset_selects[0].value == []
        assert target_dataset_selects[0]._props.get("multiple") is True

        selected_dataset = next(iter(target_dataset_selects[0].options))
        target_dataset_selects[0].value = [selected_dataset]
        settings_dialogs = [
            el for el in client.elements.values()
            if hasattr(el, "inline_grouper")
        ]
        assert len(settings_dialogs) == 1
        assert settings_dialogs[0].dataset_selector is target_dataset_selects[0]
        settings_grouper = settings_dialogs[0].inline_grouper
        settings_grouper.handle.add_row(
            "demo", {selected_dataset: ["aMe12"]}
        )
        settings_grouper.resync()
        assert settings_grouper.datasets() == [selected_dataset]
        save_buttons = [
            el for el in client.elements.values()
            if getattr(el, "text", "") == "Save Mapping"
        ]
        assert len(save_buttons) == 1
        assert save_buttons[0]._props.get("outline") is True
        assert any(text == "Custom Mapping · none" for text in texts)
        assert any(text == "Custom Mapping" for text in texts)
        mapping_buttons = [
            el for el in client.elements.values()
            if getattr(el, "text", "") == "Custom Mapping · none"
        ]
        assert len(mapping_buttons) == 1
        assert "drocat-settings-mapping-button" in mapping_buttons[0]._classes

        from ui.config import APP_DOCS_BRANCH, APP_DOCS_URL
        docs_links = [
            el for el in client.elements.values()
            if getattr(el, "text", "") == "Docs"
        ]
        assert len(docs_links) == 1
        assert docs_links[0]._props.get("href") == APP_DOCS_URL
        assert APP_DOCS_BRANCH in APP_DOCS_URL

    def test_settings_auto_suggest_toggle(self, tmp_path, monkeypatch):
        """The Settings tab exposes the input auto-suggestion toggle and
        persists it to the local config (isolated from the real config)."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.settings import create_settings_tab
        import ui.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        client = Client(page("/settings-auto-suggest"))
        with client:
            create_settings_tab()

        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "App Settings" in texts
        checkboxes = [
            el for el in client.elements.values()
            if type(el).__name__ == "Checkbox"
        ]
        suggest_cb = next(
            el for el in checkboxes
            if getattr(el, "text", "") == "Input Auto-Suggestion"
        )
        assert suggest_cb.value is True  # on by default

        # Toggling off persists the setting.
        suggest_cb.value = False
        assert cfg_mod.get_auto_suggest_enabled() is False
        suggest_cb.value = True
        assert cfg_mod.get_auto_suggest_enabled() is True

    def test_settings_default_settings_card_save_and_reset(self, tmp_path, monkeypatch):
        """The Settings tab renders the registry-driven Default Settings card
        and Save/Reset persist the overrides to the local config."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.settings import create_settings_tab
        import ui.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        client = Client(page("/settings-defaults-card"))
        with client:
            create_settings_tab()

        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "Default Settings" in texts

        def by_label(label):
            return next(
                el for el in client.elements.values()
                if getattr(el, "_props", {}).get("label") == label
                or (type(el).__name__ == "Checkbox"
                    and getattr(el, "text", "") == label)
            )

        skip_cb = by_label("Skip BodyId-Level Export")
        min_syn = by_label("Min Synapse Count / Threshold")
        dataset_sel = by_label("Default Dataset")
        assert bool(skip_cb.value) == cfg_mod.DEFAULTS["skip_bodyId"]
        assert int(min_syn.value) == cfg_mod.DEFAULTS["min_synapse_num"]
        assert dataset_sel.value == cfg_mod.DEFAULTS["default_dataset"]

        skip_cb.set_value(not cfg_mod.DEFAULTS["skip_bodyId"])
        min_syn.set_value(9)
        dataset_sel.set_value("hemibrain:v1.2.1")

        save_btn = next(
            el for el in client.elements.values()
            if getattr(el, "text", "") == "Save Defaults"
        )
        next(iter(save_btn._event_listeners.values())).handler(None)
        assert cfg_mod.get_user_default("skip_bodyId") is not cfg_mod.DEFAULTS["skip_bodyId"]
        assert cfg_mod.get_user_default("min_synapse_num") == 9
        assert cfg_mod.get_user_default("default_dataset") == "hemibrain:v1.2.1"

        # The Reset action lives on the same card right after Save Defaults.
        elements = list(client.elements.values())
        save_idx = elements.index(save_btn)
        reset_btn = next(
            el for el in elements[save_idx:]
            if getattr(el, "text", "") == "Reset"
        )
        next(iter(reset_btn._event_listeners.values())).handler(None)
        assert cfg_mod.get_user_defaults() == {}
        assert bool(skip_cb.value) == cfg_mod.DEFAULTS["skip_bodyId"]
        assert int(min_syn.value) == cfg_mod.DEFAULTS["min_synapse_num"]
        assert dataset_sel.value == cfg_mod.DEFAULTS["default_dataset"]

    def test_saved_defaults_wire_into_pathfinding_tabs(self, tmp_path, monkeypatch):
        """Saved user defaults initialize the one-time tab controls: dataset
        selectors, Skip BodyId, Min Synapse Count, and Use Cache."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.find_path import create_find_path_tab
        from ui.tabs.inter_dataset import create_inter_dataset_tab
        import ui.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        cfg_mod.set_user_default("default_dataset", "hemibrain:v1.2.1")
        cfg_mod.set_user_default("skip_bodyId", False)
        cfg_mod.set_user_default("min_synapse_num", 7)
        cfg_mod.set_user_default("use_cache", False)

        client = Client(page("/defaults-find-path"))
        with client:
            create_find_path_tab()

        def labeled(c, label):
            return [
                el for el in c.elements.values()
                if getattr(el, "_props", {}).get("label") == label
                or (type(el).__name__ == "Checkbox"
                    and getattr(el, "text", "") == label)
            ]

        dataset_sels = labeled(client, "Dataset")
        assert dataset_sels
        assert all(sel.value == "hemibrain:v1.2.1" for sel in dataset_sels)
        assert labeled(client, "Skip BodyId in Output")[0].value is False
        assert int(labeled(client, "Min Synapse Count")[0].value) == 7
        assert labeled(client, "Use Cache")[0].value is False

        client2 = Client(page("/defaults-inter-dataset"))
        with client2:
            create_inter_dataset_tab()
        multi = [
            el for el in client2.elements.values()
            if getattr(el, "_props", {}).get("multiple")
            and isinstance(getattr(el, "value", None), list)
        ]
        assert multi
        assert multi[0].value == ["hemibrain:v1.2.1"]

    def test_cache_neurons_override_gates_auto_flip(self, tmp_path, monkeypatch):
        """A saved Cache Neurons override initializes the analysis component
        and disables the pipeline-based auto-flip."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.skeleton_visualization_settings import (
            skeleton_visualization_settings,
        )
        import ui.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")

        # Without an override the auto-flip keeps Cache Neurons on (the
        # built-in default is already True; the flip is the pin guarantee).
        client = Client(page("/cache-flip-auto"))
        with client:
            auto = skeleton_visualization_settings(
                dataset_provider=lambda: "FAFB:FlyWire",
            )
        assert auto.fields["cache_neurons"].value is True

        # A saved override wins and survives the refresh triggered by a
        # skeleton-mode change.
        cfg_mod.set_user_default("cache_neurons", False)
        client2 = Client(page("/cache-flip-pinned"))
        with client2:
            pinned = skeleton_visualization_settings(
                dataset_provider=lambda: "FAFB:FlyWire",
            )
        assert pinned.fields["cache_neurons"].value is False
        pinned.fields["skeleton_mode"].set_value("tube")
        assert pinned.fields["cache_neurons"].value is False

    def test_analysis_opt_in_defaults_stay_off_without_override(self, tmp_path, monkeypatch):
        """Show Figure / Export Views keep their historical opt-in False state
        in analysis tabs unless the user explicitly saved an override; the
        skeleton tab's built-in True must not leak in."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.skeleton_visualization_settings import (
            skeleton_visualization_settings,
        )
        import ui.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")

        client = Client(page("/optin-no-override"))
        with client:
            panel = skeleton_visualization_settings()
        assert panel.fields["show_fig"].value is False
        assert panel.fields["export_views"].value is False

        cfg_mod.set_user_default("show_fig_skeleton", True)
        cfg_mod.set_user_default("export_views", True)
        client2 = Client(page("/optin-override"))
        with client2:
            panel2 = skeleton_visualization_settings()
        assert panel2.fields["show_fig"].value is True
        assert panel2.fields["export_views"].value is True

        # An invalid stored override behaves like no override at all.
        cfg_mod.save_local_config({
            cfg_mod.USER_DEFAULTS_KEY: {
                "show_fig_skeleton": "yes",
                "export_views": 3,
            }
        })
        client3 = Client(page("/optin-invalid"))
        with client3:
            panel3 = skeleton_visualization_settings()
        assert panel3.fields["show_fig"].value is False
        assert panel3.fields["export_views"].value is False

    def test_settings_token_status_is_non_sensitive(self):
        from ui.tabs.settings import _token_status

        assert _token_status("secret-token", "config.json") == "configured (config.json)"
        assert "secret-token" not in _token_status("secret-token", "config.json")
        assert _token_status("secret-token", "env var NEUPRINT_APPLICATION_CREDENTIALS") == (
            "configured (env var NEUPRINT_APPLICATION_CREDENTIALS)"
        )
        assert _token_status("") == "not configured"

    def test_token_sources_env_var_recognition(self, tmp_path, monkeypatch):
        """NEUPRINT_APPLICATION_CREDENTIALS / NEUPRINT_TOKEN / CAVE_TOKEN
        count as configured with an explicit env-var source; config files
        still win per key and empty entries fall back to the env."""
        from ui.tabs import settings as settings_module

        monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("NEUPRINT_APPLICATION_CREDENTIALS", "env-app-creds")
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.delenv("CAVE_TOKEN", raising=False)

        # env only -> configured with the env-var source
        sources = settings_module._token_sources()
        assert sources["neuprint"] == (
            "env-app-creds",
            "env var NEUPRINT_APPLICATION_CREDENTIALS",
        )
        assert sources["cave"] == ("", "")

        # config.json wins over the env var
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "cfg-np", "cave": "cfg-cave"}}\n', encoding="utf-8"
        )
        sources = settings_module._token_sources()
        assert sources["neuprint"] == ("cfg-np", "config.json")
        assert sources["cave"] == ("cfg-cave", "config.json")

        # an empty config.json entry falls back to the env var
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "", "cave": "cfg-cave"}}\n', encoding="utf-8"
        )
        sources = settings_module._token_sources()
        assert sources["neuprint"] == (
            "env-app-creds",
            "env var NEUPRINT_APPLICATION_CREDENTIALS",
        )

        # NEUPRINT_TOKEN is the fallback env var when the application
        # credentials variable is absent
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS")
        monkeypatch.setenv("NEUPRINT_TOKEN", "env-np-token")
        monkeypatch.setenv("CAVE_TOKEN", "env-cave-token")
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "", "cave": ""}}\n', encoding="utf-8"
        )
        sources = settings_module._token_sources()
        assert sources["neuprint"] == ("env-np-token", "env var NEUPRINT_TOKEN")
        assert sources["cave"] == ("env-cave-token", "env var CAVE_TOKEN")

    def test_settings_token_status_shows_source(self, tmp_path, monkeypatch):
        """The Settings tab labels each token with its source: config.json
        or the environment variable."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs import settings as settings_module

        monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "cfg-np", "cave": ""}}\n', encoding="utf-8"
        )
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.setenv("CAVE_TOKEN", "env-cave-token")

        client = Client(page("/settings-token-source"))
        with client:
            settings_module.create_settings_tab()

        statuses = [
            el.text for el in client.elements.values()
            if getattr(el, "text", "") and el.text.startswith("configured (")
        ]
        assert "configured (config.json)" in statuses
        assert "configured (env var CAVE_TOKEN)" in statuses
        assert not any("kept hidden" in s for s in statuses)

    def test_custom_grouping_instruction_link_present(self):
        """Every custom-grouping UI surface links to the LabelMapper guide:
        the Settings 'Custom Type Mappings' card and the mapping selector in
        the tool tabs (Cross-Dataset / Find Path / Find Direct)."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.mapping_editor import MAPPING_GUIDE_URL, mapping_selector

        def guide_links(client):
            return [
                el for el in client.elements.values()
                if getattr(el, "_props", {}).get("href") == MAPPING_GUIDE_URL
            ]

        # Settings card
        client = Client(page("/guide-link-settings"))
        with client:
            from ui.tabs.settings import create_settings_tab
            create_settings_tab()
        links = guide_links(client)
        assert len(links) == 1
        assert links[0]._props.get("target") == "_blank"  # opens in a new tab

        # Selector component (used by Cross-Dataset / Find Path / Find Direct)
        client = Client(page("/guide-link-selector"))
        with client:
            mapping_selector()
        links = guide_links(client)
        assert len(links) == 1
        assert links[0]._props.get("target") == "_blank"

        # Opt-out keeps the layout clean for callers that add their own link
        client = Client(page("/guide-link-selector-off"))
        with client:
            mapping_selector(show_instructions=False)
        assert guide_links(client) == []

    def test_settings_tab_reminds_when_tokens_missing(self, tmp_path, monkeypatch):
        """The Settings tab raises a visible reminder when tokens are missing:
        NeuPrint is required, CAVE is explicitly optional (only FlyWire FAFB
        online fetching), and the banner disappears once both are set."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs import settings as settings_module

        built = []

        def build(neuprint: str, cave: str):
            monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
            monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
            monkeypatch.delenv("CAVE_TOKEN", raising=False)
            (tmp_path / "config.json").write_text(
                '{"tokens": {"neuprint": "' + neuprint + '", "cave": "' + cave + '"}}\n',
                encoding="utf-8",
            )
            monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
            client = Client(page(f"/settings-reminder-{len(built)}"))
            with client:
                settings_module.create_settings_tab()
            built.append(client)
            reminder = next(
                el for el in client.elements.values()
                if getattr(el, "_props", {}).get("id") == "drocat-token-reminder"
            )
            return reminder, reminder.default_slot.children[0]

        # both missing -> prominent reminder
        reminder, text = build("", "")
        assert reminder.visible is True
        assert "No API tokens configured" in text.text
        assert "required for NeuPrint datasets" in text.text
        assert "only needed for FAFB online CAVE fetching" in text.text

        # only CAVE missing -> soft reminder marking it optional
        reminder, text = build("real-neuprint-token", "")
        assert reminder.visible is True
        assert "CAVE token not configured - optional" in text.text
        assert "FAFB online CAVE fetching" in text.text

        # only NeuPrint missing -> required-token reminder
        reminder, text = build("", "real-cave-token")
        assert reminder.visible is True
        assert "NeuPrint token not configured - it is required" in text.text

        # both set -> no reminder
        reminder, text = build("real-neuprint-token", "real-cave-token")
        assert reminder.visible is False

    def test_local_dataset_listing_requires_complete_flywire_conversion(self, tmp_path):
        from ui.dataset_service import DatasetService

        service = DatasetService()
        service._datasets_dir = tmp_path / "datasets"
        dataset_path = service._datasets_dir / "flywire_FAFB_v783"
        dataset_path.mkdir(parents=True)
        (dataset_path / "flywire_FAFB_v783_metadata.json").write_text(
            '{"neuron_counts": {"total": 10}}',
            encoding="utf-8",
        )
        (dataset_path / "flywire_FAFB_v783_allneurons_neuron_df.parquet").touch()

        listed = service.get_local_datasets()
        assert listed[0].local_prepared is False
        assert listed[0].available is False

        (dataset_path / "flywire_FAFB_v783_merged_connections.parquet").touch()
        listed = service.get_local_datasets()
        assert listed[0].local_prepared is True
        assert listed[0].available is True

    def test_local_dataset_listing_ignores_metadata_only_folders(self, tmp_path):
        """F9: ``datasets/<name>/`` must hold real tables to be listed.

        A failed cross-dataset metadata collection used to persist an error
        sidecar into an otherwise empty folder; combined with the bundled
        neuron-index seed the chip rendered it as a half-installed local
        dataset ("metadata: partial").  A metadata-only folder — and an empty
        one — is not data.
        """
        import json
        from ui.dataset_service import DatasetService

        service = DatasetService()
        service._datasets_dir = tmp_path / "datasets"
        service._cache_dir = tmp_path / "cache"
        service._index_dir = tmp_path / "neuron_indexes"

        error_only = service._datasets_dir / "hemibrain_v1_2_1"
        error_only.mkdir(parents=True)
        (error_only / "hemibrain_v1_2_1_metadata.json").write_text(
            json.dumps({
                "dataset": "hemibrain:v1.2.1",
                "source": "error",
                "error": "No NeuPrint token configured",
                "neuron_counts": {"total": 0, "typed": 0},
            }),
            encoding="utf-8",
        )
        # The bundled seed index that used to make the chip read "partial".
        seed = service._index_dir / "hemibrain_v1_2_1"
        seed.mkdir(parents=True)
        (seed / "neuron_index.parquet").touch()
        assert service.probe_metadata("hemibrain:v1.2.1") == "partial"
        (service._datasets_dir / "half_written_download").mkdir()

        assert service.get_local_datasets() == []

        real = service._datasets_dir / "manc_v1_0"
        real.mkdir(parents=True)
        (real / "manc_v1_0_allneurons_neuron_df.parquet").touch()
        (real / "manc_v1_0_metadata.json").write_text(
            json.dumps({"source": "neuprint",
                        "neuron_counts": {"total": 3}}),
            encoding="utf-8",
        )
        listed = service.get_local_datasets()
        assert [info.name for info in listed] == ["manc:v1.0"]
        assert listed[0].metadata["neuron_counts"]["total"] == 3

    def test_flywire_prepared_requires_neurons_and_connections(self, tmp_path):
        from ui.dataset_service import DatasetService

        service = DatasetService()
        service._datasets_dir = tmp_path / "datasets"
        dataset_path = service._datasets_dir / "flywire_FAFB_v783"
        dataset_path.mkdir(parents=True)

        (dataset_path / "flywire_FAFB_v783_allneurons_neuron_df.parquet").touch()
        assert service._check_local_prepared("flywire_FAFB_v783") is False

        (dataset_path / "flywire_FAFB_v783_merged_connections.parquet").touch()
        assert service._check_local_prepared("flywire_FAFB_v783") is True

        neuprint_path = service._datasets_dir / "hemibrain_v1_2_1"
        neuprint_path.mkdir()
        (neuprint_path / "hemibrain_v1_2_1_neuron_df.parquet").touch()
        assert service._check_local_prepared("hemibrain:v1.2.1") is True

    def test_local_neuron_counts_correct_without_metadata(self, tmp_path):
        """Regression: counting a neuron table without a metadata file must
        return the real row count.  The old `pl.len()` + frame-level `.sum()`
        query corrupted the total (158262 rows reported as 3572024164)."""
        pytest.importorskip("polars")
        import polars as pl

        from ui.dataset_service import DatasetService

        service = DatasetService()
        service._datasets_dir = tmp_path / "datasets"
        dataset_path = service._datasets_dir / "banc_v888"
        dataset_path.mkdir(parents=True)
        pl.DataFrame(
            {
                "bodyId": list(range(7)),
                "type": ["a", "b", None, "", "c", "d", "e"],
            }
        ).write_parquet(dataset_path / "banc_v888_allneurons_neuron_df.parquet")

        total, typed = service._load_local_neuron_counts("banc_v888")
        assert total == 7
        assert typed == 5

    def test_local_neuron_counts_accepts_plain_neuron_df_table(self, tmp_path):
        """NeuPrint conversions name their table *_neuron_df.* - it must be
        counted too, otherwise a prepared dataset shows no neuron count."""
        pytest.importorskip("polars")
        import polars as pl

        from ui.dataset_service import DatasetService

        service = DatasetService()
        service._datasets_dir = tmp_path / "datasets"
        dataset_path = service._datasets_dir / "male-cns_v0_9"
        dataset_path.mkdir(parents=True)
        pl.DataFrame({"bodyId": [1, 2, 3, 4]}).write_parquet(
            dataset_path / "male-cns_v0_9_neuron_df.parquet"
        )

        total, typed = service._load_local_neuron_counts("male-cns:v0.9")
        assert total == 4

    def test_flywire_codex_fallback_count(self, tmp_path):
        """A FlyWire dataset without local files still reports its known
        Codex release size instead of showing no neuron number."""
        from ui.dataset_service import DatasetService

        service = DatasetService()
        service._datasets_dir = tmp_path / "datasets"
        service._cache_dir = tmp_path / "cache"

        info = service.check_dataset_availability("flywire_FAFB_v783")
        assert info.available is False
        assert info.neuron_count == service.CODEX_DATASETS["flywire_FAFB_v783"]["neurons"]

    def test_cache_index_fallback_count(self, tmp_path):
        """A dataset that only has a neuron index reports the number
        of indexed neurons from neuron_indexes/<dataset>/neuron_index.parquet."""
        pytest.importorskip("polars")
        import polars as pl

        from ui.dataset_service import DatasetService

        service = DatasetService()
        service._datasets_dir = tmp_path / "datasets"
        service._cache_dir = tmp_path / "cache"
        service._index_dir = tmp_path / "neuron_indexes"
        index_path = service._index_dir / "hemibrain_v1_2_1"
        index_path.mkdir(parents=True)
        pl.DataFrame({"bodyId": [1, 2, 3]}).write_parquet(
            index_path / "neuron_index.parquet"
        )

        total, typed = service._load_cache_neuron_counts("hemibrain:v1.2.1")
        assert total == 3

    def test_fetch_neuprint_counts_without_token(self, monkeypatch, tmp_path):
        """Without a NeuPrint token the count query must short-circuit to
        (0, 0) instead of hitting the network. The developer's real
        config_local.json and shell env are excluded so "without token"
        actually holds."""
        import ui.dataset_service as ds_mod
        from ui.dataset_service import DatasetService

        monkeypatch.setattr(ds_mod, "PROJECT_ROOT", tmp_path)
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)

        service = DatasetService()
        assert service.get_token() is None
        assert service._fetch_neuprint_counts("hemibrain:v1.2.1") == (0, 0)

    def test_settings_guide_matches_converter_layout(self):
        guide = (PROJECT_ROOT / "docs" / "ui_guides" / "settings.html").read_text(
            encoding="utf-8")
        embedded_guide = (PROJECT_ROOT / "ui" / "tabs" / "settings.py").read_text(
            encoding="utf-8")
        assert "datasets/&lt;dataset&gt;/downloads/" in guide
        assert "classification.csv.gz" in guide
        assert "connections_princeton_no_threshold.csv.gz" in guide
        assert "neurons.csv.gz" in guide
        assert "connections_princeton.csv.gz" in guide
        assert "flywire_FAFB_v783_allneurons_neuron_df.parquet" in guide
        assert "flywire_FAFB_v783/flywire_FAFB_v783_allneurons_neuron_df.csv" not in guide
        assert "datasets/flywire_FAFB_v783/downloads/" in embedded_guide
        assert "connections_princeton_no_threshold.csv.gz" in embedded_guide
        assert "banc_v888" in embedded_guide

    def test_dataset_info_dataclass(self):
        from ui.dataset_service import DatasetInfo
        info = DatasetInfo(name="test", source="neuprint")
        assert info.name == "test"
        assert info.available is False
        assert info.neuron_count == 0


# =============================================================================
# Test Tab Modules
# =============================================================================
class TestTabs:
    def test_all_tab_functions_exist(self):
        from ui.tabs import (
            create_find_path_tab, create_find_shortest_tab, create_connectivity_tab,
            create_morphology_tab, create_inter_dataset_tab, create_type_validation_tab,
            create_nb_find_lines_tab, create_nb_find_neuron_tab, create_nb_colabel_tab,
            create_skeleton_tab, create_net_viz_tab, create_network_tab,
            create_visualization_tab, create_settings_tab,
        )
        assert all(callable(f) for f in [
            create_find_path_tab, create_find_shortest_tab, create_connectivity_tab,
            create_morphology_tab, create_inter_dataset_tab, create_type_validation_tab,
            create_nb_find_lines_tab, create_nb_find_neuron_tab, create_nb_colabel_tab,
            create_skeleton_tab, create_net_viz_tab, create_network_tab,
            create_visualization_tab, create_settings_tab,
        ])

    def test_tab_bar_shows_group_labels(self):
        """Layered-card navigation: every group is an independent tinted card
        holding its header above its tabs (no partition, always aligned),
        Settings is a standalone card without a header, and clicking a group
        tab switches the shared panel."""
        from types import SimpleNamespace

        from nicegui import Client
        from nicegui.page import page
        import ui.app as app_module

        client = Client(page("/nav-group-labels"))
        with client:
            app_module.main_page()

        def card_of(el):
            node = el
            while node is not None:
                if "drocat-group-card" in getattr(node, "_classes", []):
                    return node
                slot = getattr(node, "parent_slot", None)
                node = slot.parent if slot else None
            return None

        expected_headers = {
            "Connection": "connection", "Cross-Dataset": "crossdataset",
            "Visualization": "visualization",
            "Similarity": "similarity", "NeuronBridge": "nb",
            "FlyLight": "flylight",
        }
        headers = [
            el for el in client.elements.values()
            if "drocat-group-head" in getattr(el, "_classes", [])
        ]
        assert [el.text for el in headers] == list(expected_headers)
        # Header and tabs share the same tinted card (no partition).
        for el in headers:
            card = card_of(el)
            assert f"drocat-tint-{expected_headers[el.text]}" in card._classes
        # The single NB badge modifier lives on the NeuronBridge header only.
        for el in headers:
            has_badge = "drocat-head-nb" in el._classes
            assert has_badge == (el.text == "NeuronBridge")

        # Every tab button lives inside its group's tinted card.
        tint_by_label = {
            "Complete Paths": "connection", "Shortest Paths": "connection",
            "Network": "connection",
            "Paths": "crossdataset", "Type Validation": "crossdataset",
            "Skeleton": "visualization", "Net-Viz": "visualization",
            "Connectivity": "similarity", "Morphology": "similarity",
            "Find Lines": "nb", "Find Neurons": "nb", "Co-Labeling": "nb",
            "Downloader": "flylight",
        }
        buttons = {
            ((el._props or {}).get("label") or "").replace("\n", " "): el
            for el in client.elements.values()
            if "drocat-group-tab" in getattr(el, "_classes", [])
        }
        assert set(buttons) == set(tint_by_label)
        for label, tint in tint_by_label.items():
            card = card_of(buttons[label])
            assert f"drocat-tint-{tint}" in card._classes, f"{label} in wrong card"

        # Settings is promoted to the header (beside the theme toggle) rather than
        # a left-nav card, so no Settings group-tab button or nav card exists.
        header_settings = [
            el for el in client.elements.values()
            if type(el).__name__ == "Button"
            and "drocat-dark-toggle" in getattr(el, "_classes", [])
            and el._props.get("icon") == "settings"
        ]
        assert header_settings, "Settings button should live in the header beside the theme toggle"

        # Clicking a group tab switches the shared panel and active state.
        panels = next(
            el for el in client.elements.values()
            if type(el).__name__ == "TabPanels"
        )
        assert panels.value == "Complete Paths"
        assert "drocat-active" in buttons["Complete Paths"]._classes
        click = next(
            listener for listener in buttons["Net-Viz"]._event_listeners.values()
            if listener.type == "click"
        )
        click.handler(SimpleNamespace())
        assert panels.value == "Net-Viz"
        assert "drocat-active" in buttons["Net-Viz"]._classes
        assert "drocat-active" not in buttons["Complete Paths"]._classes

    def test_token_warning_dialog_opens_when_neuprint_missing(self, tmp_path, monkeypatch):
        """main_page builds a token warning dialog; without a NeuPrint token
        the one-shot timer opens it, and Open Settings jumps to the panel."""
        from nicegui import Client, ui
        from nicegui.page import page
        import ui.app as app_module
        from ui.tabs import settings as settings_module

        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "", "cave": ""}}\n', encoding="utf-8"
        )
        monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.delenv("CAVE_TOKEN", raising=False)

        client = Client(page("/token-warning-missing"))
        with client:
            app_module.main_page()

        dialog = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("id") == "drocat-token-warning"
        )
        assert dialog.value is False
        assert app_module._neuprint_token_configured() is False

        # the one-shot timer opens the dialog once the page is live
        timer = next(
            el for el in client.elements.values()
            if isinstance(el, ui.timer)
            and el.callback.__name__ == "_maybe_open_token_warning"
        )
        timer.callback()
        assert dialog.value is True

        # Open Settings navigates to the Settings panel and closes
        settings_btn = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("id") == "drocat-token-warning-settings"
        )
        panels = next(
            el for el in client.elements.values()
            if type(el).__name__ == "TabPanels"
        )
        next(iter(settings_btn._event_listeners.values())).handler(None)
        assert panels.value == "Settings"
        assert dialog.value is False

    def test_token_warning_dialog_stays_closed_when_neuprint_configured(self, tmp_path, monkeypatch):
        """With a NeuPrint token in config.json the warning never opens."""
        from nicegui import Client, ui
        from nicegui.page import page
        import ui.app as app_module
        from ui.tabs import settings as settings_module

        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "real-token", "cave": ""}}\n', encoding="utf-8"
        )
        monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.delenv("CAVE_TOKEN", raising=False)

        client = Client(page("/token-warning-configured"))
        with client:
            app_module.main_page()

        assert app_module._neuprint_token_configured() is True
        dialog = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("id") == "drocat-token-warning"
        )
        timer = next(
            el for el in client.elements.values()
            if isinstance(el, ui.timer)
            and el.callback.__name__ == "_maybe_open_token_warning"
        )
        timer.callback()
        assert dialog.value is False

    def test_token_warning_dialog_stays_closed_when_env_var_configured(self, tmp_path, monkeypatch):
        """NEUPRINT_APPLICATION_CREDENTIALS counts as configured: the
        startup warning never opens."""
        from nicegui import Client, ui
        from nicegui.page import page
        import ui.app as app_module
        from ui.tabs import settings as settings_module

        monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("NEUPRINT_APPLICATION_CREDENTIALS", "env-app-creds")
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.delenv("CAVE_TOKEN", raising=False)

        client = Client(page("/token-warning-env"))
        with client:
            app_module.main_page()

        assert app_module._neuprint_token_configured() is True
        dialog = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("id") == "drocat-token-warning"
        )
        timer = next(
            el for el in client.elements.values()
            if isinstance(el, ui.timer)
            and el.callback.__name__ == "_maybe_open_token_warning"
        )
        timer.callback()
        assert dialog.value is False

    def test_flatten_neuron_layers(self):
        """The nested layer model flattens into one neuron per entry for
        the per-neuron palette counts (single-neuron layers are plain
        values, multi-neuron layers are lists)."""
        from ui.tabs.visualization import _flatten_neuron_layers

        assert _flatten_neuron_layers(["aMe12", ["aMe10", "MBON01"], "KC"]) == [
            "aMe12", "aMe10", "MBON01", "KC",
        ]
        assert _flatten_neuron_layers([["a", "b"]]) == ["a", "b"]
        assert _flatten_neuron_layers([]) == []

    def test_skeleton_tab_shares_pathfinding_search_controls(self):
        """The 3D Skeleton tab must expose the same neuron-search controls as
        pathfinding: a filter-mode select (exact/starts-with/contains/...)
        on the neuron input, a Search Columns scope selector, and a
        Hemisphere selector (both / left / right)."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.visualization import create_skeleton_tab

        client = Client(page("/skeleton-search-controls"))
        with client:
            create_skeleton_tab()

        labels = [
            getattr(el, "_props", {}).get("label")
            for el in client.elements.values()
            if getattr(el, "_props", {}).get("label")
        ]
        assert "Search Columns" in labels
        assert "Hemisphere" in labels
        assert "Match by" in labels  # filter-mode select (renamed from 'Filter')

        # The form is split into four visually independent blocks.
        texts = [
            getattr(el, "text", "")
            for el in client.elements.values()
            if getattr(el, "text", "")
        ]
        for header in (
            "General Appearance",
            "Neuron Skeleton Appearance",
            "Synapse Appearance",
            "Brain Region ROIs (independent)",
        ):
            assert header in texts, f"missing block header: {header}"

        ids = [
            getattr(el, "_props", {}).get("id", "")
            for el in client.elements.values()
        ]
        for block_id in (
            "card-skeleton-appearance",
            "card-skeleton-neuron-colors",
            "card-skeleton-synapse-colors",
            "card-skeleton-roi-colors",
        ):
            assert block_id in ids, f"missing block card: {block_id}"

        # The brain outline color lives in General Appearance (Auto checkbox
        # + color input); the ROI card hosts only the palette editor, no
        # swatch-picker row.
        assert "Brain Mesh Color" in labels
        # One color trio per half, because the tab has always sent only
        # brain_mesh_color: the nerve cord -- ticked or only embedded hidden --
        # then kept the auto blue whatever the user picked.
        assert "VNC Mesh Color" in labels
        assert "VNC Opacity" in labels
        assert not any(
            "drocat-swatch-row" in getattr(el, "_classes", [])
            for el in client.elements.values()
        ), "swatch picker rows should be gone from the Skeleton tab"

        # Neuron palette defaults to Category10 (background is white);
        # the synapse palette defaults to Dark2.
        editors = [
            el for el in client.elements.values()
            if callable(getattr(el, "get_colors", None))
        ]
        assert any(el.get_value() == "Category10" for el in editors)
        assert any(el.get_value() == "Dark2" for el in editors)

    def test_skeleton_tab_keeps_roi_panel_mode_independent(self):
        """The Brain Region ROIs panel (its palette editor included) is
        identical in every layer-editor mode: the layer editor defines
        neurons and synapses only, never ROI meshes. Switching to Advanced
        or File upload hides the Neuron/Synapse palettes but must keep the
        ROI palette visible (the run handler consults it in every mode)."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.visualization import create_skeleton_tab

        client = Client(page("/skeleton-roi-mode-independence"))
        with client:
            create_skeleton_tab()

        palettes = {
            "neuron": "card-skeleton-neuron-palette",
            "synapse": "card-skeleton-synapse-palette",
            "roi": "card-skeleton-roi-palette",
        }

        def palette_visible(name: str) -> bool:
            target = next(
                el for el in client.elements.values()
                if getattr(el, "_props", {}).get("id") == palettes[name]
            )
            return bool(target.visible)

        def set_mode(mode: str) -> None:
            btn = next(
                el for el in client.elements.values()
                if type(el).__name__ == "Button" and getattr(el, "text", "") == mode
            )
            click_listener = next(iter(btn._event_listeners.values()))
            click_listener.handler(None)

        set_mode("Advanced")
        assert not palette_visible("neuron")
        assert not palette_visible("synapse")
        assert palette_visible("roi"), "ROI palette must stay visible in Advanced mode"

        set_mode("File upload")
        assert not palette_visible("neuron")
        assert not palette_visible("synapse")
        assert palette_visible("roi"), "ROI palette must stay visible in File upload mode"

        set_mode("Standard")
        assert palette_visible("neuron")
        assert palette_visible("synapse")
        assert palette_visible("roi")

    def test_all_tabs_put_advanced_settings_last_in_own_card(self):
        """Every tool tab keeps its Advanced Settings expansion at the very
        bottom of the form column, wrapped in its own drocat-card (same
        pattern as the 3D Skeleton tab), and styles the header with the
        shared drocat-section-expansion class."""
        from nicegui import Client
        from nicegui.page import page

        builders = [
            ("skeleton", "ui.tabs.visualization", "create_skeleton_tab"),
            ("find_path", "ui.tabs.find_path", "create_find_path_tab"),
            ("find_shortest", "ui.tabs.find_shortest", "create_find_shortest_tab"),
            ("network", "ui.tabs.network", "create_network_tab"),
            ("inter_dataset", "ui.tabs.inter_dataset", "create_inter_dataset_tab"),
            ("connectivity", "ui.tabs.connectivity", "create_connectivity_tab"),
            ("morphology", "ui.tabs.morphology", "create_morphology_tab"),
            ("nb_find_lines", "ui.tabs.nb_find_lines", "create_nb_find_lines_tab"),
            ("nb_find_neuron", "ui.tabs.nb_find_neuron", "create_nb_find_neuron_tab"),
            ("nb_colabel", "ui.tabs.nb_colabel", "create_nb_colabel_tab"),
        ]
        import importlib

        for name, module_name, func_name in builders:
            module = importlib.import_module(module_name)
            builder = getattr(module, func_name)
            client = Client(page(f"/advanced-bottom-{name}"))
            with client:
                builder()

            cards = [
                el for el in client.elements.values()
                if "drocat-card" in getattr(el, "_classes", [])
                and str(getattr(el, "_props", {}).get("id", "")).endswith("-advanced")
            ]
            assert len(cards) == 1, f"{name}: expected exactly one advanced card"
            card = cards[0]

            expansion = next(iter(card), None)
            assert expansion is not None, f"{name}: advanced card is empty"
            assert getattr(expansion, "text", "") == "Advanced Settings"
            assert "drocat-section-expansion" in getattr(expansion, "_classes", []), (
                f"{name}: advanced expansion must use drocat-section-expansion"
            )

            parent = card.parent_slot.parent
            last_child = list(parent)[-1] if list(parent) else None
            assert last_child is card, (
                f"{name}: the advanced card must be the last block of its "
                "form column"
            )

    def test_skeleton_tab_keeps_method_selectable_for_flywire(self):
        """Tube-mode FlyWire/FAFB renders can choose their pipeline."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.visualization import create_skeleton_tab

        client = Client(page("/skeleton-method-dataset-gating"))
        with client:
            create_skeleton_tab()

        dataset = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Dataset"
        )
        method = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Simplification Method"
        )

        assert method.enabled is True
        assert [option["label"] for option in method._props["options"]] == [
            "fast", "fine", "artistic"
        ]
        assert method.value == "fast"
        dataset.value = "flywire_FAFB_v783"
        assert method.enabled is True
        dataset.value = "male-cns:v1.0"
        assert method.enabled is True

    def test_skeleton_tab_disables_simplification_controls_for_line_mode(self):
        """Line rendering bypasses both method and mesh simplification."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.visualization import create_skeleton_tab

        client = Client(page("/skeleton-line-simplification-gating"))
        with client:
            create_skeleton_tab()

        mode = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Skeleton Mode"
        )
        method = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Simplification Method"
        )
        default_control = next(
            el for el in client.elements.values()
            if getattr(el, "text", "") == "Use Default Mesh Simplification"
        )
        mesh = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") ==
            "Mesh Simplification (faces removed)"
        )

        assert method.enabled is True
        assert default_control.enabled is True
        assert mesh.enabled is False
        assert mesh.value == 0.90

        method.value = "fast"
        assert mesh.value == 0.90
        method.value = "fine"
        assert mesh.value == 0.95

        # A user-entered value is not overwritten by method toggles while the
        # default checkbox is disabled.
        default_control.value = False
        mesh.value = 0.75
        method.value = "fast"
        assert mesh.value == 0.75
        default_control.value = True
        assert mesh.value == 0.90

        mode.value = "line"
        assert method.enabled is False
        assert default_control.enabled is False
        assert mesh.enabled is False

        mode.value = "tube"
        assert method.enabled is True
        assert default_control.enabled is True
        method.value = "fast"
        assert mesh.value == 0.90
        method.value = "fine"
        assert mesh.value == 0.95
        default_control.value = False
        assert mesh.enabled is True

    def test_skeleton_tab_export_and_grouping_controls(self):
        """The 3D Skeleton tab exposes the drag-and-drop layer tree editor,
        the individual-profile export controls (collapsed entries above the
        Advanced Settings expansion, which closes the tab), and the
        legend-mode grouping notice."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.visualization import create_skeleton_tab

        client = Client(page("/skeleton-export-controls"))
        with client:
            create_skeleton_tab()

        labels = [
            getattr(el, "_props", {}).get("label")
            for el in client.elements.values()
            if getattr(el, "_props", {}).get("label")
        ]
        texts = [
            getattr(el, "text", "")
            for el in client.elements.values()
            if getattr(el, "text", "")
        ]
        all_text = labels + texts
        assert "Layer Structure" in all_text
        assert "Add Layer" in all_text
        assert "Add Chain" not in all_text  # chain support was removed
        # The tree seeds three layer rows (2nd/3rd may stay empty).
        rows = [
            el for el in client.elements.values()
            if "drocat-layer-row" in getattr(el, "_classes", [])
        ]
        assert len(rows) == 3, "expected three default layer rows"
        ids = [
            getattr(el, "_props", {}).get("id", "")
            for el in client.elements.values()
        ]
        assert "card-skeleton-layers" in ids
        assert "Export Individual Profiles" in all_text
        assert "Summary Format" in all_text

        assert "Export Video / GIF" in texts
        assert "Individual Profiles (PDF / PPTX)" in texts
        # The legend select is wired to the persisted user default; compare
        # against the same source instead of a hardcoded value so the test
        # holds whatever the developer's local settings currently are.
        from ui.config import get_user_default
        legend = [
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Neuron Legend Mode"
        ]
        assert legend and legend[0].value == get_user_default("legend_mode"), \
            "legend mode should follow the user default"
        assert any("Each individual profile follows the Neuron Legend Mode" in t
                   for t in texts), "legend-mode grouping notice missing"
        assert any("Each row is one layer" in t
                   for t in texts), "layer tree hint missing"

        # the profiles controls live in the export card (outside Advanced
        # Settings): the export card and the Advanced Settings card must be
        # independent blocks
        ids = [
            getattr(el, "_props", {}).get("id", "")
            for el in client.elements.values()
        ]
        assert "card-skeleton-export-video" in ids
        assert "card-skeleton-advanced" in ids

    def test_find_path_tab_exposes_path_cap_control(self):
        """The Find All Paths tab exposes the per-source path cap (the
        practical bound for the combinatorial Reconstruct explosion)."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.find_path import create_find_path_tab

        client = Client(page("/findpath-cap-control"))
        with client:
            create_find_path_tab()

        labels = [
            getattr(el, "_props", {}).get("label")
            for el in client.elements.values()
            if getattr(el, "_props", {}).get("label")
        ]
        texts = [
            getattr(el, "text", "")
            for el in client.elements.values()
            if getattr(el, "text", "")
        ]
        all_text = labels + texts
        # Fix C: the lossy Edge Limit – BodyIds control was retired — the
        # StrongestFirst path budget (Max Paths (BodyId)) is the single
        # bounding knob (0 = auto).
        assert "Max Paths (BodyId)" in all_text
        assert "Edge Limit – BodyIds" not in all_text
        max_paths = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Max Paths (BodyId)"
        )
        assert max_paths.value == 0  # auto (StrongestFirst: 1M budget)
        # the type/group edge-limit controls are gone from the UI
        assert "Limit Graph Edges" not in all_text
        assert "Edge Limit – Groups" not in all_text
        assert "Edge Limit – Types/Groups" not in all_text
        assert "Visualize Network Before Reconstruction" not in all_text
        # Complete Paths does not expose the early-visualization checkbox.
        early_viz = [
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("id") == "checkbox-early-viz"
        ]
        assert not early_viz
        # the deep-layer warning label exists (hidden until layers >= 4)
        assert any("Layers ≥ 4" in t for t in texts), "deep-layer warning missing"

    def test_find_lines_tab_top_lines_default_is_30(self):
        """The Find Driver Lines image/visualization top-N defaults to 30
        lines (was 20)."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.nb_find_lines import create_nb_find_lines_tab

        client = Client(page("/nbfindlines-top30"))
        with client:
            create_nb_find_lines_tab()

        labels = [
            getattr(el, "_props", {}).get("label")
            for el in client.elements.values()
            if getattr(el, "_props", {}).get("label")
        ]
        texts = [
            getattr(el, "text", "")
            for el in client.elements.values()
            if getattr(el, "text", "")
        ]
        assert "Top Lines for Images" in labels + texts
        # find the number input by its label and check the default value
        for el in client.elements.values():
            if getattr(el, "_props", {}).get("label") == "Top Lines for Images":
                assert el.value == 30, f"default top lines = {el.value}, expected 30"
                break
        else:
            raise AssertionError("Top Lines for Images control not found")

        # 'From FlyLight' image download is ON by default (matched via its
        # stable id; checkbox labels are client-side slot text)
        flylight = [
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("id") == "checkbox-flylight"
        ]
        assert flylight and flylight[0].value is True, "From FlyLight default must be checked"

    def test_pathfinding_tabs_have_hemisphere_filter_select(self):
        """Find All Paths and Shortest expose the 'Hemisphere' selector
        (both / left / right) next to 'Hemisphere-aware'."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.find_path import create_find_path_tab
        from ui.tabs.find_shortest import create_find_shortest_tab

        for name, builder in [
            ("/hemi-filter-findpath", create_find_path_tab),
            ("/hemi-filter-findshortest", create_find_shortest_tab),
        ]:
            client = Client(page(name))
            with client:
                builder()
            labels = [
                getattr(el, "_props", {}).get("label")
                for el in client.elements.values()
                if getattr(el, "_props", {}).get("label")
            ]
            texts = [
                getattr(el, "text", "")
                for el in client.elements.values()
                if getattr(el, "text", "")
            ]
            assert "Hemisphere-aware" in texts
            assert "Hemisphere" in labels

    def test_hemisphere_filter_is_inline_with_aware_toggle(self):
        """Keep the hemisphere selector beside its enabling checkbox."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.find_path import create_find_path_tab
        from ui.tabs.find_shortest import create_find_shortest_tab
        from ui.tabs.network import create_network_tab

        controls = [
            ("/hemi-inline-findpath", create_find_path_tab, "Hemisphere-aware"),
            ("/hemi-inline-findshortest", create_find_shortest_tab, "Hemisphere-aware"),
            ("/hemi-inline-network", create_network_tab, "Hemisphere-aware"),
        ]
        for name, builder, toggle_label in controls:
            client = Client(page(name))
            with client:
                builder()

            selector = next(
                el for el in client.elements.values()
                if getattr(el, "_props", {}).get("label") == "Hemisphere"
            )
            selector_wrapper = selector.parent_slot.parent
            assert "drocat-inline-select" in selector_wrapper._classes
            assert "w-full" not in selector_wrapper._classes

            toggle = next(
                el for el in client.elements.values()
                if getattr(el, "text", "") == toggle_label
            )
            assert toggle.parent_slot.parent is selector_wrapper.parent_slot.parent

    def test_reciprocal_connections_belong_to_core_parameters(self):
        """Reciprocal enrichment is a core path option, not hemisphere analysis."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.inter_dataset import create_inter_dataset_tab
        from ui.tabs.find_path import create_find_path_tab
        from ui.tabs.find_shortest import create_find_shortest_tab

        for name, builder, core_id, hemisphere_id in [
            ("/reciprocal-core-findpath", create_find_path_tab,
             "card-findpath-core", "card-findpath-hemisphere"),
            ("/reciprocal-core-findshortest", create_find_shortest_tab,
             "card-findshortest-core", "card-findshortest-hemisphere"),
            ("/reciprocal-core-interdataset", create_inter_dataset_tab,
             "card-interdataset-core", "card-interdataset-hemisphere"),
        ]:
            client = Client(page(name))
            with client:
                builder()

            reciprocal = next(
                el for el in client.elements.values()
                if getattr(el, "text", "") == "Find Reciprocal Connections"
            )
            ancestors = []
            node = reciprocal
            while node is not None:
                ancestors.append(getattr(node, "_props", {}).get("id", ""))
                parent_slot = getattr(node, "parent_slot", None)
                node = parent_slot.parent if parent_slot is not None else None

            assert core_id in ancestors
            assert hemisphere_id not in ancestors

    def test_conserved_edges_checkbox_is_on_row_below_symmetry(self):
        """Keep-only-conserved edges gets its own row below symmetry."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.find_path import create_find_path_tab
        from ui.tabs.find_shortest import create_find_shortest_tab
        from ui.tabs.inter_dataset import create_inter_dataset_tab
        from ui.tabs.network import create_network_tab

        controls = [
            ("/conserved-row-findpath", create_find_path_tab),
            ("/conserved-row-findshortest", create_find_shortest_tab),
            ("/conserved-row-network", create_network_tab),
            ("/conserved-row-interdataset", create_inter_dataset_tab),
        ]
        for name, builder in controls:
            client = Client(page(name))
            with client:
                builder()

            symmetry = next(
                el for el in client.elements.values()
                if getattr(el, "text", "") == "Symmetry Analysis"
            )
            conserved = next(
                el for el in client.elements.values()
                if getattr(el, "text", "") == "Keep Only Hemisphere-Conserved Edges"
            )
            aware = next(
                el for el in client.elements.values()
                if getattr(el, "text", "") in (
                    "Hemisphere-aware", "Separate Hemispheres (L/R)"
                )
            )
            aware_row = aware.parent_slot.parent
            symmetry_row = symmetry.parent_slot.parent
            conserved_row = conserved.parent_slot.parent
            assert aware_row is not symmetry_row
            assert symmetry_row is not conserved_row
            assert aware_row is not conserved_row
            assert "items-center" in conserved_row._classes
            elements = list(client.elements.values())
            assert elements.index(aware_row) < elements.index(symmetry_row)
            assert elements.index(conserved_row) > elements.index(symmetry_row)

    def test_path_tabs_start_with_dataset_and_fixed_full_width_neuron_rows(self):
        """Path tabs mirror Cross-Dataset's dataset-first input layout."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.find_path import create_find_path_tab
        from ui.tabs.find_shortest import create_find_shortest_tab
        from ui.tabs.inter_dataset import create_inter_dataset_tab

        controls = [
            ("/layout-findpath", create_find_path_tab,
             "card-findpath-dataset", "card-findpath-neurons", "card-findpath-core"),
            ("/layout-findshortest", create_find_shortest_tab,
             "card-findshortest-dataset", "card-findshortest-neurons", "card-findshortest-core"),
            ("/layout-interdataset", create_inter_dataset_tab,
             "card-interdataset-datasets", "card-interdataset-neurons", "card-interdataset-core"),
        ]
        for name, builder, dataset_id, neurons_id, core_id in controls:
            client = Client(page(name))
            with client:
                builder()

            ids = [
                getattr(el, "_props", {}).get("id", "")
                for el in client.elements.values()
            ]
            assert ids.index(dataset_id) < ids.index(neurons_id) < ids.index(core_id)

            fixed_inputs = [
                el for el in client.elements.values()
                if "drocat-fixed-neuron-input" in getattr(el, "_classes", [])
            ]
            assert len(fixed_inputs) == 2
            for container in fixed_inputs:
                assert container.parent_slot.parent._props.get("id") == neurons_id
                input_row = next(
                    el for el in client.elements.values()
                    if "drocat-neuron-input-row" in getattr(el, "_classes", [])
                    and el.parent_slot.parent is container
                )
                assert "w-full" in input_row._classes
                row_children = [
                    el for el in client.elements.values()
                    if getattr(getattr(el, "parent_slot", None), "parent", None)
                    is input_row
                ]
                assert any(
                    "drocat-neuron-match-filter" in getattr(el, "_classes", [])
                    for el in row_children
                )
                assert any(
                    "drocat-upload-trigger" in getattr(el, "_classes", [])
                    for el in row_children
                )

            output_dirs = [
                el for el in client.elements.values()
                if getattr(el, "_props", {}).get("label") == "Output Directory"
            ]
            assert len(output_dirs) == 1
            ancestor = output_dirs[0]
            ancestor_ids = []
            while ancestor is not None:
                ancestor_ids.append(getattr(ancestor, "_props", {}).get("id", ""))
                parent_slot = getattr(ancestor, "parent_slot", None)
                ancestor = parent_slot.parent if parent_slot is not None else None
            assert dataset_id in ancestor_ids

    def test_network_and_similarity_tabs_use_standalone_dataset_cards(self):
        """Network and similarity tabs keep dataset/output controls together."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.network import create_network_tab
        from ui.tabs.connectivity import create_connectivity_tab
        from ui.tabs.morphology import create_morphology_tab

        controls = [
            ("/layout-network", create_network_tab,
             ("card-network-dataset", "card-network-neurons"), 1),
            ("/layout-connectivity", create_connectivity_tab,
             (
                 "card-connectivity-similar-datasets",
                 "card-connectivity-similar-neurons",
                 "card-connectivity-comparison-datasets",
                 "card-connectivity-comparison-neurons",
             ), 2),
            ("/layout-morphology", create_morphology_tab,
             (
                 "card-morphology-findsimilar-dataset",
                 "card-morphology-findsimilar-neurons",
                 "card-morphology-comparison-dataset",
                 "card-morphology-comparison-neurons",
             ), 2),
        ]
        for name, builder, card_ids, expected_fixed_inputs in controls:
            client = Client(page(name))
            with client:
                builder()

            ids = [
                getattr(el, "_props", {}).get("id", "")
                for el in client.elements.values()
            ]
            for dataset_id, neurons_id in zip(card_ids[::2], card_ids[1::2]):
                assert ids.index(dataset_id) < ids.index(neurons_id)

            fixed_inputs = [
                el for el in client.elements.values()
                if "drocat-fixed-neuron-input" in getattr(el, "_classes", [])
            ]
            assert len(fixed_inputs) == expected_fixed_inputs
            expected_neuron_cards = set(card_ids[1::2])
            for container in fixed_inputs:
                parent_card_id = container.parent_slot.parent._props.get("id")
                assert parent_card_id in expected_neuron_cards

            output_dirs = [
                el for el in client.elements.values()
                if getattr(el, "_props", {}).get("label") == "Output Directory"
            ]
            assert len(output_dirs) == expected_fixed_inputs
            for output_dir in output_dirs:
                ancestor = output_dir
                ancestor_ids = []
                while ancestor is not None:
                    ancestor_ids.append(
                        getattr(ancestor, "_props", {}).get("id", "")
                    )
                    parent_slot = getattr(ancestor, "parent_slot", None)
                    ancestor = parent_slot.parent if parent_slot is not None else None
                dataset_ancestors = [
                    card_id for card_id in card_ids[::2]
                    if card_id in ancestor_ids
                ]
                assert len(dataset_ancestors) == 1

    def test_visualization_tabs_put_dataset_or_output_cards_at_the_top(self):
        """Keep Skeleton and Net-Viz storage controls in their first card."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.visualization import create_skeleton_tab, create_net_viz_tab

        controls = [
            ("/layout-skeleton", create_skeleton_tab,
             "card-skeleton-dataset", "card-3d", "Output Directory"),
            ("/layout-net-viz-output", create_net_viz_tab,
             "card-net-viz-output", "card-net-viz-source", "Path Output Directory"),
        ]
        for name, builder, storage_id, next_card_id, output_label in controls:
            client = Client(page(name))
            with client:
                builder()

            ids = [
                getattr(el, "_props", {}).get("id", "")
                for el in client.elements.values()
            ]
            assert ids.index(storage_id) < ids.index(next_card_id)

            output_dir = next(
                el for el in client.elements.values()
                if getattr(el, "_props", {}).get("label") == output_label
            )
            ancestor = output_dir
            ancestor_ids = []
            while ancestor is not None:
                ancestor_ids.append(
                    getattr(ancestor, "_props", {}).get("id", "")
                )
                parent_slot = getattr(ancestor, "parent_slot", None)
                ancestor = parent_slot.parent if parent_slot is not None else None
            assert storage_id in ancestor_ids

    def test_connectivity_find_similar_uses_single_source_and_target(self):
        """Find Similar exposes one source and one target dataset; Target =
        Source gives the intra-dataset search, so no separate connectivity
        mode exists anymore."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.connectivity import create_connectivity_tab

        client = Client(page("/similar-source-target"))
        with client:
            create_connectivity_tab()

        labels = [
            getattr(el, "_props", {}).get("label")
            for el in client.elements.values()
            if getattr(el, "_props", {}).get("label")
        ]
        assert labels.count("Source Dataset") == 1
        assert labels.count("Target Dataset") == 1
        source = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Source Dataset"
        )
        target = next(
            el for el in client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Target Dataset"
        )
        assert source._props.get("multiple") is not True
        assert target._props.get("multiple") is not True

    def test_restructured_tabs_have_independent_block_cards(self):
        """Tabs reviewed for card separation must expose their logical
        groups as independent cards (source / output / hemisphere /
        image-download / rendering), not one merged card."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.find_path import create_find_path_tab
        from ui.tabs.find_shortest import create_find_shortest_tab
        from ui.tabs.inter_dataset import create_inter_dataset_tab
        from ui.tabs.nb_find_lines import create_nb_find_lines_tab
        from ui.tabs.visualization import create_net_viz_tab

        expected_blocks = [
            ("/blocks-findpath", create_find_path_tab, (
                "card-findpath-dataset", "card-findpath-neurons",
                "card-findpath-core", "card-findpath-output",
                "card-findpath-hemisphere",
            )),
            ("/blocks-findshortest", create_find_shortest_tab, (
                "card-findshortest-dataset", "card-findshortest-neurons",
                "card-findshortest-core", "card-findshortest-output",
                "card-findshortest-hemisphere",
            )),
            ("/blocks-interdataset", create_inter_dataset_tab, (
                "card-interdataset-datasets", "card-interdataset-neurons",
                "card-interdataset-core",
                "card-interdataset-hemisphere",
            )),
            ("/blocks-nbfindlines", create_nb_find_lines_tab, (
                "card-nb-image-download",
            )),
            ("/blocks-net-viz", create_net_viz_tab, (
                "card-net-viz-output", "card-net-viz-source",
                "card-net-viz-rendering",
            )),
        ]
        for name, builder, blocks in expected_blocks:
            client = Client(page(name))
            with client:
                builder()
            ids = [
                getattr(el, "_props", {}).get("id", "")
                for el in client.elements.values()
            ]
            for block_id in blocks:
                assert block_id in ids, f"{name}: missing block card {block_id}"

        # The restructured cards surface their headers without expansion.
        client = Client(page("/blocks-findpath-headers"))
        with client:
            create_find_path_tab()
        texts = [
            getattr(el, "text", "")
            for el in client.elements.values()
            if getattr(el, "text", "")
        ]
        for header in ("Core Parameters", "Output Options", "Hemisphere Analysis"):
            assert header in texts, f"missing visible header: {header}"

        # The Net-Viz tab must not keep the old merged card id.
        client = Client(page("/blocks-net-viz-id"))
        with client:
            create_net_viz_tab()
        ids = [
            getattr(el, "_props", {}).get("id", "")
            for el in client.elements.values()
        ]
        assert "card-network" not in ids

    def test_interdataset_symmetry_off_without_hemispheres(self):
        """Regression: the cross-dataset tab must NOT pass symmetry_analysis=True
        (or keep-hemisphere-conserved) when Hemisphere-aware is unchecked —
        the dependent checkboxes are unchecked AND disabled by default, so a
        greyed-out True never reaches ComparisonParameters."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.inter_dataset import create_inter_dataset_tab

        client = Client(page("/interdataset-symmetry"))
        with client:
            create_inter_dataset_tab()
        by_id = {}
        for el in client.elements.values():
            pid = getattr(el, "_props", {}).get("id")
            if pid:
                by_id[pid] = el
        sep = by_id["checkbox-separate-hemi"]
        sym = by_id["checkbox-symmetry"]
        cons = by_id["checkbox-hemi-conserved"]
        assert any(
            getattr(el, "text", "") == "Hemisphere-aware"
            for el in client.elements.values()
        )
        assert sep.value is False
        # symmetry is unchecked AND disabled while hemispheres are off
        assert sym.value is False, "symmetry must be unchecked without hemispheres"
        assert sym._props.get("disable") is True
        assert cons.value is False
        assert cons._props.get("disable") is True

    def test_interdataset_edge_limits_shared_with_find_path(self):
        """The cross-dataset tab carries the report-row cap and the
        Visualization Edge Limit (shared DEFAULTS, same as Find All
        Paths). Fix C removed the bodyId edge limit entirely: the
        StrongestFirst path budget (Max Paths (BodyId)) is the single
        pathfinding knob."""
        from nicegui import Client
        from nicegui.page import page
        from ui.config import DEFAULTS
        from ui.tabs.inter_dataset import create_inter_dataset_tab

        client = Client(page("/interdataset-edge-limits"))
        with client:
            create_inter_dataset_tab()
        by_label = {}
        for el in client.elements.values():
            label = getattr(el, "_props", {}).get("label")
            if label in (
                "Top Edges in Analysis Reports",
                "Max Paths (BodyId)",
                "Visualization Edge Limit",
            ):
                by_label[label] = el
        assert "Top Edges in Analysis Reports" in by_label, by_label
        assert by_label["Top Edges in Analysis Reports"].value == 500
        # single pathfinding knob: 0 = auto (StrongestFirst 1M budget)
        assert "Max Paths (BodyId)" in by_label, by_label
        assert by_label["Max Paths (BodyId)"].value == DEFAULTS["max_paths_bodyid"]
        labels = [
            getattr(el, "_props", {}).get("label")
            for el in client.elements.values()
            if getattr(el, "_props", {}).get("label")
        ]
        assert "Edge Limit – BodyIds" not in labels
        # the visualization edge limit default follows the shared config
        assert by_label["Visualization Edge Limit"].value == DEFAULTS["edgeN_limit"]
        assert DEFAULTS["edgeN_limit"] == 500

    def test_path_and_shortest_tabs_use_unbounded_max_intermediate_layers(self):
        """Both path tabs expose the same unbounded layer control."""
        from nicegui import Client
        from nicegui.page import page
        from ui.config import DEFAULTS
        from ui.tabs.find_path import create_find_path_tab
        from ui.tabs.find_shortest import create_find_shortest_tab

        shortest_client = Client(page("/findshortest-max-intermediate-layers"))
        with shortest_client:
            create_find_shortest_tab()
        shortest_layers = [
            el for el in shortest_client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Max Intermediate Layers"
        ]

        path_client = Client(page("/findpath-max-intermediate-layers"))
        with path_client:
            create_find_path_tab()
        path_layers = [
            el for el in path_client.elements.values()
            if getattr(el, "_props", {}).get("label") == "Max Intermediate Layers"
        ]

        assert shortest_layers and path_layers, "Max Intermediate Layers input missing"
        assert shortest_layers[0].value == 5, shortest_layers[0].value
        assert path_layers[0].value == DEFAULTS["max_interlayer"], path_layers[0].value
        for layers in (shortest_layers, path_layers):
            props = layers[0]._props
            assert props.get("min") == 0 and props.get("max") is None, props

        for client in (shortest_client, path_client):
            labels = [
                getattr(el, "_props", {}).get("label")
                for el in client.elements.values()
                if getattr(el, "_props", {}).get("label")
            ]
            ids = [
                getattr(el, "_props", {}).get("id")
                for el in client.elements.values()
            ]
            assert "Visualize Network Before Reconstruction" not in labels
            assert "checkbox-early-viz" not in ids

    def test_interdataset_mode_switch_resets_mode_defaults(self):
        """Switching Path Enumeration resets the mode-specific Max Layers
        default (shortest: 8; all: 2) and warns the user their values were
        reset. Fix C: no edge-limit control participates anymore."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.inter_dataset import create_inter_dataset_tab

        client = Client(page("/interdataset-mode-switch"))
        with client:
            create_inter_dataset_tab()
        by_label = {}
        for el in client.elements.values():
            label = getattr(el, "_props", {}).get("label")
            if label in ("Path Enumeration", "Max Intermediate Layers"):
                by_label[label] = el
        mode = by_label["Path Enumeration"]
        layers = by_label["Max Intermediate Layers"]
        # 'all' default
        assert layers.value == 2, layers.value
        mode.value = "shortest"
        assert layers.value == 8, layers.value
        mode.value = "all"
        assert layers.value == 2, layers.value

    def test_path_tabs_uncheck_hemisphere_dependents(self):
        """Find All Paths and Find Shortest UNCHECK (not just disable) the
        hemisphere-dependent options when Hemisphere-aware is off, so a
        greyed-out True never reaches the backend."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.find_path import create_find_path_tab
        from ui.tabs.find_shortest import create_find_shortest_tab

        for name, builder in [("/fp-hemi-uncheck", create_find_path_tab),
                              ("/fs-hemi-uncheck", create_find_shortest_tab)]:
            client = Client(page(name))
            with client:
                builder()
            by_label = {}
            for el in client.elements.values():
                label = (getattr(el, "_props", {}).get("label")
                         or getattr(el, "text", ""))
                if label in ("Hemisphere-aware",
                             "Keep Only Hemisphere-Conserved Edges",
                             "Symmetry Analysis"):
                    by_label[label] = el
            sep = by_label["Hemisphere-aware"]
            keep = by_label["Keep Only Hemisphere-Conserved Edges"]
            sym = by_label["Symmetry Analysis"]
            sep.value = True
            keep.value = True
            sym.value = True
            sep.value = False
            assert keep.value is False, name
            assert sym.value is False, name
            assert keep._props.get("disable") is True, name

    def test_interdataset_path_enumeration_selector(self):
        """The cross-dataset tab exposes a Path Enumeration selector
        (all / shortest). F1 removed the Pathfinding Algorithm selector
        (StrongestFirst is built-in); F1/F3 removed the Edge Limit field in
        favor of the Edge Budget."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.inter_dataset import create_inter_dataset_tab

        client = Client(page("/interdataset-path-mode"))
        with client:
            create_inter_dataset_tab()
        by_label = {}
        for el in client.elements.values():
            label = getattr(el, "_props", {}).get("label")
            if label in ("Path Enumeration", "Pathfinding Algorithm",
                         "Edge Budget", "Max Paths (BodyId)"):
                by_label[label] = el
        assert "Path Enumeration" in by_label, sorted(by_label)
        assert by_label["Path Enumeration"].value == "all"
        # F1: the Pathfinding Algorithm selector is gone entirely
        assert "Pathfinding Algorithm" not in by_label
        # F3: the Edge Budget + Max Paths knobs are present
        assert "Edge Budget" in by_label
        assert "Max Paths (BodyId)" in by_label
        # switching to shortest: the Edge Budget is disabled (never floors)
        by_label["Path Enumeration"].value = "shortest"
        assert by_label["Edge Budget"].enabled is False

    def test_network_tab_is_find_network_with_scope_notice(self):
        """The Network tab (Connection group) hosts FindNetwork: a single
        Query Neurons input, no path controls (max layers / algorithm), the
        limited-scope notice pointing to Find Path + Find Reciprocal, and
        the old placeholder card is gone."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.network import create_network_tab

        client = Client(page("/network-findnetwork"))
        with client:
            create_network_tab()
        labels = [
            getattr(el, "_props", {}).get("label")
            for el in client.elements.values()
            if getattr(el, "_props", {}).get("label")
        ]
        texts = [
            getattr(el, "text", "")
            for el in client.elements.values()
            if getattr(el, "text", "")
        ]
        ids = [
            getattr(el, "_props", {}).get("id", "")
            for el in client.elements.values()
        ]
        assert "Query Neurons" in labels
        assert "Find Network" in " ".join(texts)  # run button label
        # limited-scope notice with the reciprocal alternative
        assert any("Find Reciprocal Connections" in t for t in texts)
        # no irrelevant path controls
        assert "Max Layers" not in labels
        assert not any("Pathfinding" in str(l) for l in labels if l)
        # placeholder removed
        assert "card-network-placeholder" not in ids
        # FindNetwork cards present
        for card_id in ("card-network-core", "card-network-output",
                        "card-network-hemisphere"):
            assert card_id in ids, card_id

    def test_comparison_tab_custom_group_aggregation(self):
        """The Comparison sub-tab offers a custom-group aggregation level:
        selecting it reveals the LabelMapper preset selector that feeds
        custom_mapping_file into ConnectivityProfileComparer."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.connectivity import create_connectivity_tab

        client = Client(page("/comparison-custom-groups"))
        with client:
            create_connectivity_tab()

        elements = list(client.elements.values())
        agg = [
            el for el in elements
            if getattr(el, "_props", {}).get("id") == "select-aggregation"
        ]
        assert agg, "Aggregation Level select missing"
        agg_sel = agg[0]
        assert agg_sel.options == ["type", "bodyid", "custom group"], agg_sel.options

        # the profiling tab supports multiple datasets (cross-dataset mode)
        multi_ds = [
            el for el in elements
            if getattr(el, "_props", {}).get("label") ==
            "Datasets to compare (select one or more)"
        ]
        assert multi_ds, "multi-dataset selector missing"
        assert multi_ds[0]._props.get("multiple") is True
        assert multi_ds[0].value == ["male-cns:v1.0"]

        # the custom-group card is hidden until the level is chosen
        card = [
            el for el in elements
            if getattr(el, "_props", {}).get("id") == "card-custom-group"
        ]
        assert card, "custom-group card missing"
        assert card[0].visible is False
        agg_sel.value = "custom group"
        assert card[0].visible is True
        # the card hosts the Custom Grouping button that opens the panel
        # (the panel hosts the saved-preset selector + inline board)
        texts = [
            getattr(el, "text", "")
            for el in client.elements.values()
        ]
        assert any("Custom Grouping Preset" in t for t in texts)



# =============================================================================
# Test Components
# =============================================================================
class TestComponents:
    def test_common_imports(self):
        from ui.components.common import (
            dataset_selector, neuron_input, number_input, select_input,
            checkbox_input, dir_input, section_header,
            param_grid, open_folder, dataset_status_card,
        )
        assert all(callable(f) for f in [
            dataset_selector, neuron_input, number_input, select_input,
            checkbox_input, dir_input, section_header,
            param_grid, open_folder, dataset_status_card,
        ])

    def test_dataset_selectors_use_outlined_fields(self):
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import dataset_multi_selector, dataset_selector

        client = Client(page("/outlined-dataset-selectors"))
        with client:
            single = dataset_selector(
                label="Source Dataset",
                datasets=["demo:v1.0"],
                show_local_status=False,
            )
            multi = dataset_multi_selector(
                label="Target Datasets",
                default=[],
                datasets=["demo:v1.0", "demo:v2.0"],
                show_local_status=False,
            )

        assert single._props.get("outlined") is True
        assert multi._props.get("outlined") is True

    def test_native_multi_selects_clear_filter_text_after_chip_changes(self):
        """Mouse-picked chips must not leave their search text in the editor."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import (
            chip_list_input,
            dataset_multi_selector,
            multi_select_input,
        )

        client = Client(page("/native-select-clear-test"))
        with client:
            controls = [
                chip_list_input(),
                dataset_multi_selector(
                    default=[],
                    datasets=["demo:v1.0", "demo:v2.0"],
                    show_local_status=False,
                ),
                multi_select_input("Mesh ROIs", ["AME", "ME"]),
            ]
            no_search = multi_select_input(
                "No Search", ["AME", "ME"], with_search=False
            )

        for control in controls:
            listeners = {
                listener.type: listener
                for listener in control._event_listeners.values()
            }
            assert "add" in listeners
            assert "remove" in listeners

            calls = []
            control.run_method = lambda *args, **kwargs: calls.append((args, kwargs))
            listeners["add"].handler(None)
            listeners["remove"].handler(None)
            assert calls == [
                (("updateInputValue", ""), {}),
                (("updateInputValue", ""), {}),
            ]

        assert not any(
            listener.type in {"add", "remove"}
            for listener in no_search._event_listeners.values()
        )

    def test_neuron_list_input_uses_file_upload_not_list_paste(self):
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/neuron-input-upload-controls"))
        with client:
            neuron_list_input(show_upload=True)

        icons = {
            getattr(el, "_props", {}).get("icon")
            for el in client.elements.values()
        }
        texts = [str(getattr(el, "text", "")) for el in client.elements.values()]
        assert "upload_file" in icons
        assert "playlist_add" not in icons
        assert not any("Paste a list" in value for value in texts)

    def test_neuron_upload_accepts_mixed_entries_in_every_supported_format(self):
        from ui.components.common import parse_neuron_upload

        expected = [10001, "DNp01", "DNp01(GF)_R", "VS1", "KC.*"]
        fixture_dir = PROJECT_ROOT / "tests" / "fixtures"
        for suffix in ("csv", "tsv", "xlsx", "xls"):
            path = fixture_dir / f"neuron_upload_mixed.{suffix}"
            assert parse_neuron_upload(path.name, path.read_bytes()) == expected

    def test_neuron_list_input_commits_pending_text_on_blur(self):
        """Leaving a neuron chip field commits text without requiring Enter."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/neuron-input-blur-test"))
        with client:
            container = neuron_list_input(label="Source Neurons")

        listeners = {
            listener.type: listener
            for listener in container.chip_input._event_listeners.values()
        }
        assert listeners["input"].js_handler == "(event) => emit(event?.target?.value ?? '')"
        assert "blur" in listeners

        # Simulate NiceGUI's native input and blur events. The browser's
        # QSelect editor is empty after blur, but the selected chip remains.
        container.chip_input._handle_event({
            "listener_id": listeners["input"].id,
            "args": "PPL1",
        })
        assert container.chip_input.value == []
        container.chip_input._handle_event({
            "listener_id": listeners["blur"].id,
            "args": None,
        })
        assert container.get_value() == ("exact", ["PPL1"])
        # Committed values must live in the option list, otherwise the chip
        # never renders in the browser (model-value is filtered by options).
        assert container.chip_input.options == ["PPL1"]

    def test_neuron_list_input_clearing_text_drops_pending_value(self):
        """Deleting the editor text must not commit the previously typed text."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/neuron-input-clear-test"))
        with client:
            container = neuron_list_input(label="Source Neurons")

        listeners = {
            listener.type: listener
            for listener in container.chip_input._event_listeners.values()
        }

        # Type a value, then clear the field with Backspace (native input '').
        container.chip_input._handle_event({
            "listener_id": listeners["input"].id,
            "args": "PPL1",
        })
        container.chip_input._handle_event({
            "listener_id": listeners["input"].id,
            "args": "",
        })
        # Quasar's blur-reset emits an empty input-value; it must be ignored.
        container.chip_input._handle_event({
            "listener_id": listeners["inputValue"].id,
            "args": "",
        })
        container.chip_input._handle_event({
            "listener_id": listeners["blur"].id,
            "args": None,
        })
        # Nothing was committed: the field was cleared before blur.
        assert container.get_value() == ("exact", [])

    def test_neuron_list_input_max_items_rejects_second_chip(self):
        """A chip input with max_items=1 keeps only the first value."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/neuron-input-max-items-test"))
        with client:
            container = neuron_list_input(label="Source Neuron", max_items=1)

        listeners = {
            listener.type: listener
            for listener in container.chip_input._event_listeners.values()
        }

        # The first value is committed on blur.
        container.chip_input._handle_event({
            "listener_id": listeners["input"].id,
            "args": "PPL1",
        })
        container.chip_input._handle_event({
            "listener_id": listeners["blur"].id,
            "args": None,
        })
        assert container.get_value() == ("exact", ["PPL1"])

        # A second value is rejected once the input is at capacity.
        container.chip_input._handle_event({
            "listener_id": listeners["input"].id,
            "args": "PPL2",
        })
        container.chip_input._handle_event({
            "listener_id": listeners["blur"].id,
            "args": None,
        })
        assert container.get_value() == ("exact", ["PPL1"])

        # Quasar-native additions (Enter) are truncated to the cap as well.
        container.chip_input.value = ["PPL1", "PPL2"]
        container.chip_input._handle_event({
            "listener_id": listeners["update:modelValue"].id,
            "args": ["PPL1", "PPL2"],
        })
        assert container.get_value() == ("exact", ["PPL1"])

    def test_neuron_list_input_initial_values_survive_enter(self):
        """Seeded chips survive get_value(); Enter keeps the whole text as ONE chip."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/neuron-input-initial-test"))
        with client:
            container = neuron_list_input(
                label="Synapse Thresholds",
                initial=[3, 5, 10],
                show_filter=False,
                show_upload=False,
            )
        assert container.get_value() == ("exact", [3, 5, 10])
        assert container.chip_input.options == [3, 5, 10]

        listeners = {
            listener.type: listener
            for listener in container.chip_input._event_listeners.values()
        }

        # Quasar's add-unique adds the whole editor text as a single chip on
        # Enter; commas are legal inside names and must NOT be split.
        container.chip_input.value = [3, 5, 10, "3, 7"]
        container.chip_input._handle_event({
            "listener_id": listeners["update:modelValue"].id,
            "args": [3, 5, 10, "3, 7"],
        })
        assert container.get_value() == ("exact", [3, 5, 10, "3, 7"])
        assert container.chip_input.options == [3, 5, 10, "3, 7"]

    def test_neuron_list_input_collapses_and_double_click_reopens_a_chip(self):
        """Long chip lists stay compact, while double-click restores one
        existing value to the editor for typo correction."""
        from nicegui import Client
        from nicegui.page import page
        from types import SimpleNamespace
        from ui.components.common import neuron_list_input

        client = Client(page("/neuron-input-expand-edit-test"))
        with client:
            container = neuron_list_input(
                label="Source Neurons",
                initial=["aMe1", "aMe2", "aMe3", "aMe4"],
            )

        assert "drocat-chip-list-collapsed" in container.chip_input_anchor._classes
        assert "drocat-chip-list-expanded" not in container.chip_input_anchor._classes
        assert container.expand_button.text == "Expand"
        expand_listener = next(
            listener
            for listener in container.expand_button._event_listeners.values()
            if listener.type == "click"
        )
        expand_listener.handler(SimpleNamespace())
        assert "drocat-chip-list-expanded" in container.chip_input_anchor._classes
        assert container.expand_button.text == "Collapse"
        expand_listener.handler(SimpleNamespace())
        assert "drocat-chip-list-collapsed" in container.chip_input_anchor._classes
        assert container.expand_button.text == "Expand"

        dblclick = next(
            listener
            for listener in container.chip_input._event_listeners.values()
            if listener.type == "dblclick"
        )
        container.chip_input._handle_event({
            "listener_id": dblclick.id,
            "args": "aMe2",
        })
        assert container.chip_input.value == ["aMe1", "aMe3", "aMe4"]

        blur = next(
            listener
            for listener in container.chip_input._event_listeners.values()
            if listener.type == "blur"
        )
        blur.handler(SimpleNamespace(args=None))
        assert container.get_value() == (
            "exact",
            ["aMe1", "aMe3", "aMe4", "aMe2"],
        )

    def test_neuron_list_input_commas_and_spaces_are_not_separators(self):
        """Blur commits the WHOLE editor text as one chip (',' and ' ' preserved)."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/neuron-input-no-split-test"))
        with client:
            container = neuron_list_input(label="Source Neurons")

        listeners = {
            listener.type: listener
            for listener in container.chip_input._event_listeners.values()
        }

        # Comma-containing names (e.g. driver lines) stay one chip.
        container.chip_input._handle_event({
            "listener_id": listeners["input"].id,
            "args": "PPL1, PPL2",
        })
        container.chip_input._handle_event({
            "listener_id": listeners["blur"].id,
            "args": None,
        })
        assert container.get_value() == ("exact", ["PPL1, PPL2"])

        # Space-containing names (e.g. 'A -> B -> C' layers) stay one chip.
        container.chip_input._handle_event({
            "listener_id": listeners["input"].id,
            "args": "A -> B -> C",
        })
        container.chip_input._handle_event({
            "listener_id": listeners["blur"].id,
            "args": None,
        })
        assert container.get_value() == ("exact", ["PPL1, PPL2", "A -> B -> C"])

        # Surrounding whitespace is trimmed; digit-only text still normalizes.
        container.chip_input._handle_event({
            "listener_id": listeners["input"].id,
            "args": "  720575940610453042  ",
        })
        container.chip_input._handle_event({
            "listener_id": listeners["blur"].id,
            "args": None,
        })
        assert container.get_value() == ("exact", ["PPL1, PPL2", "A -> B -> C", 720575940610453042])

    def test_neuron_list_input_suggestions_menu_and_history(self, tmp_path, monkeypatch):
        """With a suggestions provider the input replaces the native popup
        with a custom menu: suggestions appear from the first character, show
        a solid value + gray column hint, commit on click; an empty focused
        field offers the persisted query history."""
        from nicegui import Client
        from nicegui.page import page
        import ui.history_store as hs
        from ui.components.common import neuron_list_input

        monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
        monkeypatch.setattr(
            "ui.config.LOCAL_CONFIG_FILE", tmp_path / "local_config.json"
        )
        hs.record(["aMe12", "aMe10"], now="2026-08-11T10:00:00")
        hs.record(["aMe12"], now="2026-08-11T10:05:00")

        def fake_suggest(text):
            if text == "a":
                return [("aMe12", "type"), ("aMe10", "instance")]
            if text == "ap":
                return [("APL", "type"), ("APL2", "type")]
            return []

        client = Client(page("/neuron-input-suggest-test"))
        with client:
            container = neuron_list_input(
                label="Source Neurons", suggestions=fake_suggest
            )

        chip = container.chip_input
        # The native QSelect popup is suppressed in favor of the custom menu.
        assert chip._props.get("popup-content-class") == "drocat-native-popup-hidden"
        assert chip._props.get("hide-dropdown-icon") is True

        input_listeners = [
            listener for listener in chip._event_listeners.values()
            if listener.type == "input"
        ]
        assert len(input_listeners) == 2  # suggestion + pending-text trackers
        suggest_input = input_listeners[0]
        # The input hosts several ui.menus (paste/upload/suggest); the
        # suggestion menu is the last one created.
        menus = [el for el in client.elements.values()
                 if type(el).__name__ == "Menu"]
        menu = menus[-1]
        # The suggestion menu must not steal focus from the editor (typing
        # would die); NiceGUI 3.15 renders no DOM ids, so it anchors to its
        # private wrapper rather than a shared page target.
        assert menu._props.get("no-focus") is True
        assert menu._props.get("no-refocus") is True
        assert menu._props.get("no-parent-event") is True
        assert menu._props.get("target", "").startswith("#")

        # The cached provider is fast enough to show suggestions immediately
        # from the first character; history is reserved for blank focus.
        chip._handle_event({"listener_id": suggest_input.id, "args": "a"})
        assert menu.value is True
        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "aMe12" in texts and "aMe10" in texts

        # A matching nonblank query opens the custom menu with value + gray
        # hint entries.
        chip._handle_event({"listener_id": suggest_input.id, "args": "ap"})
        assert menu.value is True
        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "APL" in texts and "APL2" in texts and "type" in texts

        # Clicking a suggestion commits it as a chip and HOLDS the query's rows
        # on screen — every row stays listed and the picked one is ticked — so
        # the next entry can be added without retyping (mirrors the standard
        # query box, where a pick does not close the list).
        def _subtree_texts(el):
            """Flatten the element subtree texts (labels live inside a row)."""
            out = [getattr(el, "text", "")]
            for child in el.default_slot.children:
                out.extend(_subtree_texts(child))
            return out

        item = next(
            el for el in client.elements.values()
            if type(el).__name__ == "Item" and "APL" in _subtree_texts(el)
        )
        click = next(l for l in item._event_listeners.values() if l.type == "click")
        from types import SimpleNamespace
        click.handler(SimpleNamespace())
        assert container.get_value() == ("exact", ["APL"])

        assert menu.value is True
        held_texts = [
            el.text for el in client.elements.values() if getattr(el, "text", "")
        ]
        assert "APL2" in held_texts and "Recent" not in held_texts
        assert "type" in held_texts  # held rows keep their category hints
        added = [
            el for el in client.elements.values()
            if type(el).__name__ == "Item"
            and any(c == "drocat-suggest-added" for c in el._classes)
        ]
        assert len(added) == 1 and "APL" in _subtree_texts(added[0])

        # A closing click (focus leaving the field) ends the held query, so the
        # next focus opens the persisted query history instead.
        blur = next(l for l in chip._event_listeners.values() if l.type == "blur")
        chip._handle_event({
            "listener_id": blur.id,
            "args": {"still_inside": False},
        })
        assert menu.value is False
        focus = next(l for l in chip._event_listeners.values() if l.type == "focus")
        chip._handle_event({"listener_id": focus.id, "args": None})
        assert menu.value is True
        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "Recent" in texts
        assert "aMe12" in texts and "aMe10" in texts

        # Typing the first character renders the cached provider's results,
        # without committing it as a chip. History is not mixed into search.
        chip._handle_event({"listener_id": suggest_input.id, "args": "a"})
        assert menu.value is True
        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "aMe12" in texts and "aMe10" in texts
        assert container.get_value() == ("exact", ["APL"])

    def test_neuron_list_history_uses_the_active_dataset_scope(
        self, tmp_path, monkeypatch
    ):
        """History rows follow the same dataset scope as auto-suggestions."""
        from nicegui import Client
        from nicegui.page import page
        import ui.group_history as gh
        import ui.history_store as hs
        from ui.components.common import neuron_list_input

        dataset_a = "male-cns:v0.9"
        dataset_b = "hemibrain:v1.2.1"
        monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
        monkeypatch.setattr(gh, "HISTORY_PATH", tmp_path / "group_history.json")
        hs.record(["ordinary_a"], now="2026-08-11T10:00:00",
                  datasets=[dataset_a])
        hs.record(["ordinary_b"], now="2026-08-11T10:01:00",
                  datasets=[dataset_b])
        gh.record([("custom_a", {dataset_a: ["aMe12"], dataset_b: []})])
        hs.record(["custom_a"], now="2026-08-11T10:02:00",
                  custom_values=["custom_a"], datasets=[dataset_a])
        selected = {"value": dataset_a}

        client = Client(page("/neuron-input-dataset-history"))
        with client:
            box = neuron_list_input(
                label="Source Neurons",
                suggestions=lambda _text: [],
                available_neurons=lambda: selected["value"],
            )

        focus = next(
            listener for listener in box.chip_input._event_listeners.values()
            if listener.type == "focus"
        )

        box.chip_input._handle_event({"listener_id": focus.id, "args": None})
        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "ordinary_a" in texts and "custom_a" in texts
        assert "ordinary_b" not in texts

        box.suggest_menu.close()
        selected["value"] = dataset_b
        box.chip_input._handle_event({"listener_id": focus.id, "args": None})
        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "ordinary_b" in texts
        assert "ordinary_a" not in texts and "custom_a" not in texts

    def test_history_rows_show_dataset_provenance_badges(self, tmp_path, monkeypatch):
        """With ``show_history_datasets`` (cross-dataset tab), history rows
        carry a gray tag only for SELECTED datasets whose local pool actually
        contains the value — a multi-dataset run's recorded selection never
        tags a value with a dataset it does not resolve in; the default
        dropdown shows no dataset tags at all."""
        from nicegui import Client
        from nicegui.page import page
        import ui.history_store as hs
        import ui.type_suggestions as ts
        from ui.components.common import neuron_list_input

        dataset_a = "male-cns:v1.0"
        dataset_b = "hemibrain:v1.2.1"
        monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
        pools = {
            dataset_a: {"type": [("aMe12", "type")]},
            dataset_b: {"type": [("aMe12", "type"), ("KC", "type")]},
        }
        monkeypatch.setattr(
            ts, "get_dataset_pools", lambda dataset: pools.get(str(dataset), {}))
        # Recorded like a multi-dataset run: both chips carry both datasets.
        hs.record(["aMe12", "KC"], now="2026-08-11T10:00:00",
                  datasets=[dataset_a, dataset_b])
        selected = {"value": [dataset_a]}

        client = Client(page("/neuron-history-dataset-badges"))
        with client:
            plain = neuron_list_input(
                label="Source Neurons", suggestions=lambda _text: []
            )
            cross = neuron_list_input(
                label="Target Neurons",
                suggestions=lambda _text: [],
                available_neurons=lambda: selected["value"],
                show_history_datasets=True,
            )

        def subtree_texts(el):
            out = [getattr(el, "text", "")]
            for child in el.default_slot.children:
                out.extend(subtree_texts(child))
            return out

        def focus(box):
            listener = next(
                listener for listener in box.chip_input._event_listeners.values()
                if listener.type == "focus"
            )
            box.chip_input._handle_event({"listener_id": listener.id, "args": None})

        def row_texts(box, value):
            focus(box)
            for child in box.suggest_menu.default_slot.children:
                texts = subtree_texts(child)
                if value in texts:
                    return texts
            raise AssertionError(f"{value} history row not found")

        # Default dropdown: entries are shown without dataset tags.
        assert "male-cns:v1.0" not in row_texts(plain, "aMe12")

        # Cross-dataset dropdown: aMe12 is tagged male-cns only — the
        # recorded hemibrain provenance never becomes a badge it does not
        # resolve in.
        cross_texts = row_texts(cross, "aMe12")
        assert "male-cns:v1.0" in cross_texts
        assert "hemibrain:v1.2.1" not in cross_texts

        # Selecting the other dataset swaps the visible tag.
        cross.suggest_menu.close()
        selected["value"] = [dataset_b]
        cross_texts = row_texts(cross, "aMe12")
        assert "hemibrain:v1.2.1" in cross_texts
        assert "male-cns:v1.0" not in cross_texts

        # KC resolves only in hemibrain: under a male-cns-only selection the
        # row stays (recorded scope still intersects) but shows no badge.
        selected["value"] = [dataset_a]
        cross.suggest_menu.close()
        assert "hemibrain:v1.2.1" not in row_texts(cross, "KC")
        assert "male-cns:v1.0" not in row_texts(cross, "KC")

    def test_history_rows_show_category_hint_like_suggestions(self, tmp_path, monkeypatch):
        """History rows resolve their searched column (type/instance/bodyId)
        from the selected dataset pools, mirroring the auto-suggestion list."""
        from nicegui import Client
        from nicegui.page import page
        import ui.history_store as hs
        import ui.type_suggestions as ts
        from ui.components.common import neuron_list_input

        monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
        monkeypatch.setattr(ts, "get_dataset_pools", lambda dataset: {
            "type": [("aMe12", "type"), ("DN1p", "type")],
            "instance": [("aMe12_L", "instance")],
            "bodyId": [],
        })
        hs.record(["aMe12", "aMe12_L"], now="2026-08-11T10:00:00",
                  datasets=["male-cns:v1.0"])

        client = Client(page("/neuron-history-category-hints"))
        with client:
            box = neuron_list_input(
                label="Source Neurons",
                suggestions=lambda _text: [],
                available_neurons=lambda: "male-cns:v1.0",
            )

        focus = next(
            listener for listener in box.chip_input._event_listeners.values()
            if listener.type == "focus"
        )
        box.chip_input._handle_event({"listener_id": focus.id, "args": None})

        def subtree_texts(el):
            out = [getattr(el, "text", "")]
            for child in el.default_slot.children:
                out.extend(subtree_texts(child))
            return out

        item = next(
            subtree_texts(el)
            for el in client.elements.values()
            if type(el).__name__ == "Item" and "aMe12" in subtree_texts(el)
        )
        assert "type" in item  # the searched column, like suggestion rows
        instance_item = next(
            subtree_texts(el)
            for el in client.elements.values()
            if type(el).__name__ == "Item" and "aMe12_L" in subtree_texts(el)
        )
        assert "instance" in instance_item

    def test_suggestion_menu_arrow_key_navigation_wiring(self):
        """Each suggestion input pairs its anchor with its menu through a
        unique class token, and the shared arrow-key navigation script is
        registered exactly once per client."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/neuron-input-keynav"))
        with client:
            first = neuron_list_input(
                label="Source Neurons", suggestions=lambda _text: []
            )
            second = neuron_list_input(
                label="Target Neurons", suggestions=lambda _text: []
            )

        def anchor_tokens(box):
            return [
                cls for cls in box.chip_input_anchor._classes
                if cls.startswith("drocat-suggest-anchor-")
            ]

        first_tokens = anchor_tokens(first)
        second_tokens = anchor_tokens(second)
        assert len(first_tokens) == 1 and len(second_tokens) == 1
        assert first_tokens != second_tokens
        menu_token_classes = {
            cls
            for classes in (first.suggest_menu._classes,
                            second.suggest_menu._classes)
            for cls in classes
            if cls.startswith("drocat-suggest-menu-")
        }
        assert f"drocat-suggest-menu-{first_tokens[0].rsplit('-', 1)[-1]}" \
            in menu_token_classes
        assert f"drocat-suggest-menu-{second_tokens[0].rsplit('-', 1)[-1]}" \
            in menu_token_classes
        # The document-level handler is shared: registered once per client.
        from ui.components.common import _SUGGEST_KEYNAV_SCRIPT
        assert client._head_html.count(_SUGGEST_KEYNAV_SCRIPT) == 1
        assert "drocat-suggest-active" in client._head_html
        assert "drocat-suggest-header" in client._head_html
        # Enter and Tab both pick the highlighted row; Tab falls back to its
        # native focus move when no row is highlighted.
        assert "'Enter', 'Tab'" in _SUGGEST_KEYNAV_SCRIPT
        assert "event.key === 'Enter' || event.key === 'Tab'" in \
            _SUGGEST_KEYNAV_SCRIPT
        # After a pick the highlight stays on the list: the handler advances it
        # locally and a MutationObserver re-applies it once the server rebuilds
        # the rows (a held query re-renders the same rows, ticked).
        assert "MutationObserver" in _SUGGEST_KEYNAV_SCRIPT
        assert "pending = current + 1" in _SUGGEST_KEYNAV_SCRIPT
        # The pick never wipes the editor text locally: blanking the field
        # while the pick is still on the wire is what reads as the text being
        # cleared before the chip is added. The server clears it in the same
        # pass that adds the chip, and no input event is dispatched, so the
        # in-flight pick cannot be displaced by a rebuilt list.
        assert "clearEditor" not in _SUGGEST_KEYNAV_SCRIPT
        assert "dispatchEvent" not in _SUGGEST_KEYNAV_SCRIPT

    def test_history_body_id_has_instance_hint_and_removable_entry(self, tmp_path, monkeypatch):
        """Blank-input history rows expose the cached body-ID instance and a
        close action that removes only that history value."""
        from nicegui import Client
        from nicegui.page import page
        import ui.history_store as hs
        from ui.components.common import neuron_list_input

        monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
        monkeypatch.setattr(
            "ui.config.LOCAL_CONFIG_FILE", tmp_path / "local_config.json"
        )
        hs.record(["5813", "aMe12"], now="2026-08-11T10:00:00")

        def fake_suggest(text):
            return [("5813", "aMe12_L")] if text == "5813" else []

        client = Client(page("/neuron-history-remove-test"))
        with client:
            container = neuron_list_input(
                label="Source Neurons", suggestions=fake_suggest
            )

        chip = container.chip_input
        focus = next(l for l in chip._event_listeners.values() if l.type == "focus")
        chip._handle_event({"listener_id": focus.id, "args": None})

        def _subtree_texts(el):
            out = [getattr(el, "text", "")]
            for child in el.default_slot.children:
                out.extend(_subtree_texts(child))
            return out

        history_item = next(
            el for el in client.elements.values()
            if type(el).__name__ == "Item" and "5813" in _subtree_texts(el)
        )
        texts = _subtree_texts(history_item)
        assert "aMe12_L" in texts
        close = next(
            el for el in client.elements.values()
            if type(el).__name__ == "Button"
            and getattr(el, "_props", {}).get("icon") == "close"
        )
        click = next(l for l in close._event_listeners.values() if l.type == "click")
        click.handler(None)
        assert "5813" not in hs.recent()

    def test_viewer_selection_keeps_display_name_until_backend_resolution(
        self, monkeypatch
    ):
        """A viewer name cannot be re-resolved through the wrong metadata column."""
        from nicegui import Client, ui
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        callbacks = {}

        def fake_viewer_link(_dataset_getter, **kwargs):
            callbacks.update(kwargs)
            return ui.button("Fake available neurons")

        monkeypatch.setattr(
            "ui.components.neuron_index_viewer.create_neuron_index_viewer_link",
            fake_viewer_link,
        )

        client = Client(page("/neuron-input-viewer-resolution"))
        with client:
            container = neuron_list_input(
                label="Source Neurons",
                available_neurons=lambda: "test:v1.0",
            )

        callbacks["query_selection"](["MTe01a"])
        callbacks["query_resolution"](["100", "200", "300"])
        assert container.chip_input.value == ["MTe01a"]
        # The input surface keeps the readable name.  Resolution is performed
        # later by the shared analysis backend, so logs and run names cannot
        # be replaced by the viewer's body-ID snapshot.
        assert container.get_value() == ("exact", ["MTe01a"])

        callbacks["query_selection"]([])
        callbacks["query_resolution"]([])
        assert container.get_value() == ("exact", [])

        # If the user already has the same display name in the query, a
        # viewer selection must still execute through its verified body IDs;
        # deselecting restores the pre-existing name rather than deleting it.
        container.add_values(["MTe01a"])
        callbacks["query_selection"](["MTe01a"])
        callbacks["query_resolution"](["100", "200"])
        assert container.chip_input.value == ["MTe01a"]
        assert container.get_value() == ("exact", ["MTe01a"])
        callbacks["query_selection"]([])
        callbacks["query_resolution"]([])
        assert container.get_value() == ("exact", ["MTe01a"])

        # The viewer's mirrored-query close button removes the same value
        # from the owning chip input, including a pre-existing display name.
        callbacks["query_remove"]("MTe01a")
        assert container.get_value() == ("exact", [])

    def test_neuron_list_input_suggestions_staged_typing(self, tmp_path, monkeypatch):
        """The rendered menu follows the input one character at a time.

        One character immediately opens dataset suggestions, and every later
        edit replaces or narrows the previous result set.
        """
        from nicegui import Client
        from nicegui.page import page
        import ui.config as cfg_mod
        import ui.history_store as hs
        from ui.components.common import neuron_list_input

        monkeypatch.setattr(cfg_mod, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
        hs.record(["aMe12", "aMe10", "DN1p"], now="2026-08-11T10:00:00")
        calls = []

        def fake_suggest(text):
            calls.append(text)
            return {
                "a": [
                    ("aMe12", "type"), ("aMap", "type"),
                    ("AmbiguousType", "type"),
                ],
                # This entry proves the full provider is not called for a
                # strict continuation while the previous list still matches.
                "aMe12x": [("BackendFallback", "type")],
            }.get(text, [])

        client = Client(page("/neuron-input-suggest-staged"))
        with client:
            container = neuron_list_input(
                label="Source Neurons", suggestions=fake_suggest
            )

        chip = container.chip_input
        input_listeners = [
            listener for listener in chip._event_listeners.values()
            if listener.type == "input"
        ]
        suggest_input = input_listeners[0]
        focus = next(l for l in chip._event_listeners.values() if l.type == "focus")
        menu = [el for el in client.elements.values() if type(el).__name__ == "Menu"][-1]

        # Do not bind the native focusout path: QSelect emits a transient
        # focusout during its first internal editor handoff, which used to
        # close the Recent menu immediately after it opened.
        assert not any(
            listener.type == "focusout"
            for listener in chip._event_listeners.values()
        )

        def texts():
            return [el.text for el in client.elements.values() if getattr(el, "text", "")]

        chip._handle_event({"listener_id": focus.id, "args": None})
        assert menu.value is True

        # The first QSelect click can emit a transient blur while the native
        # editor is taking focus. The browser-side focus check marks that
        # event as internal, so the Recent menu must stay open.
        blur = next(l for l in chip._event_listeners.values() if l.type == "blur")
        assert "activeElement" in (blur.js_handler or "")
        chip._handle_event({
            "listener_id": blur.id,
            "args": {"still_inside": True},
        })
        assert menu.value is True

        # a: the cached provider is rendered immediately, including all
        # matching values returned for the one-character query.
        chip._handle_event({"listener_id": suggest_input.id, "args": "a"})
        assert calls == ["a"]
        assert menu.value is True
        assert "AmbiguousType" in texts()
        assert "aMe12" in texts() and "aMap" in texts() and "DN1p" not in texts()

        # aM: all matching type suggestions appear with their column hints.
        chip._handle_event({"listener_id": suggest_input.id, "args": "aM"})
        assert calls == ["a"]
        assert "aMe12" in texts() and "aMap" in texts()
        assert "type" in texts() and "Recent" not in texts()

        # aMe -> aMe1 -> aMe12: each edit replaces the previous menu.
        chip._handle_event({"listener_id": suggest_input.id, "args": "aMe"})
        assert calls == ["a"]
        assert "aMe12" in texts() and "aMap" not in texts()
        chip._handle_event({"listener_id": suggest_input.id, "args": "aMe1"})
        assert calls == ["a"]
        assert "aMe12" in texts() and "aMap" not in texts()
        chip._handle_event({"listener_id": suggest_input.id, "args": "aMe12"})
        assert calls == ["a"]
        assert "aMe12" in texts() and "aMap" not in texts()

        # Once the reused list has no continuation match, the provider is
        # asked for a fresh staged search.
        chip._handle_event({"listener_id": suggest_input.id, "args": "aMe12x"})
        assert calls == ["a", "aMe12x"]
        assert "BackendFallback" in texts()
        assert container.get_value() == ("exact", [])

        # Clearing the editor starts a new candidate search; the old "aMe12x"
        # result must not be reused if the user begins a different query.
        chip._handle_event({"listener_id": suggest_input.id, "args": ""})
        chip._handle_event({"listener_id": suggest_input.id, "args": "a"})
        assert calls == ["a", "aMe12x", "a"]

    def test_neuron_list_input_suggestions_single_popup_on_focus_change(
        self, tmp_path, monkeypatch
    ):
        """Moving between neuron inputs closes the previous popup."""
        from nicegui import Client
        from nicegui.page import page
        import ui.config as cfg_mod
        import ui.history_store as hs
        from ui.components.common import neuron_list_input

        monkeypatch.setattr(cfg_mod, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
        hs.record(["aMe12"], now="2026-08-11T10:00:00")

        client = Client(page("/neuron-input-suggest-focus-change"))
        with client:
            source = neuron_list_input(
                label="Source Neurons", show_filter=False, show_upload=False,
                suggestions=lambda _text: [("aMe12", "type")],
            )
            target = neuron_list_input(
                label="Target Neurons", show_filter=False, show_upload=False,
                suggestions=lambda _text: [("PPL101", "type")],
            )

        source_focus = next(
            listener for listener in source.chip_input._event_listeners.values()
            if listener.type == "focus"
        )
        target_focus = next(
            listener for listener in target.chip_input._event_listeners.values()
            if listener.type == "focus"
        )

        source.chip_input._handle_event(
            {"listener_id": source_focus.id, "args": None}
        )
        assert source.suggest_menu.value is True
        assert target.suggest_menu.value is False

        # The registry closes/guards the source menu before the target menu
        # opens, even when no source blur event is delivered by QSelect.
        target.chip_input._handle_event(
            {"listener_id": target_focus.id, "args": None}
        )
        assert source.suggest_menu.value is False
        assert target.suggest_menu.value is True

    def test_neuron_list_input_suggestions_quasar_input_value_updates(
        self, tmp_path, monkeypatch
    ):
        """Quasar's input-value event also refreshes each query stage."""
        from nicegui import Client
        from nicegui.page import page
        import ui.config as cfg_mod
        import ui.history_store as hs
        from ui.components.common import neuron_list_input

        monkeypatch.setattr(cfg_mod, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
        hs.record(["aMe12", "aMe10"], now="2026-08-11T10:00:00")
        calls = []

        def fake_suggest(text):
            calls.append(text)
            return {
                "aM": [("aMe12", "type"), ("aMap", "type")],
            }.get(text, [])

        client = Client(page("/neuron-input-suggest-input-value"))
        with client:
            container = neuron_list_input(
                label="Source Neurons", suggestions=fake_suggest
            )

        chip = container.chip_input
        focus = next(l for l in chip._event_listeners.values() if l.type == "focus")
        native_input = next(
            l for l in chip._event_listeners.values()
            if l.type == "input" and l.handler.__name__ == "_on_suggest_input"
        )
        input_value = next(
            l for l in chip._event_listeners.values()
            if l.type == "inputValue"
            and l.handler.__name__ == "_on_suggest_input_value"
        )
        menu = [el for el in client.elements.values() if type(el).__name__ == "Menu"][-1]

        chip._handle_event({"listener_id": focus.id, "args": None})

        # Simulate the native event followed by Quasar's canonical event for
        # the same keystroke: the dedupe guard must refresh exactly once.
        chip._handle_event({"listener_id": native_input.id, "args": "aM"})
        chip._handle_event({"listener_id": input_value.id, "args": "aM"})
        assert calls == ["aM"]
        assert menu.value is True
        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "aMe12" in texts and "aMap" in texts

        # A later input-value event must replace the previous result set.
        chip._handle_event({"listener_id": input_value.id, "args": "aMe"})
        assert calls == ["aM"]
        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "aMe12" in texts and "aMap" not in texts

        # A no-match update also removes the previous rows instead of
        # leaving stale suggestions attached to the next popup.
        chip._handle_event({"listener_id": input_value.id, "args": "zZ"})
        assert calls == ["aM", "zZ"]
        assert menu.value is False
        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "aMe12" not in texts and "aMap" not in texts

    def test_neuron_list_input_suggestions_finish_and_focus(self, tmp_path, monkeypatch):
        """Graceful lifecycle: a finished input (suggestion pick) holds the
        query's rows for the next pick, and a focus change hides the list
        automatically and ends the hold."""
        from nicegui import Client
        from nicegui.page import page
        import ui.config as cfg_mod
        from ui.components.common import neuron_list_input

        monkeypatch.setattr(cfg_mod, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        import ui.history_store as hs
        monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
        hs.record(["aMe12", "aMe10"], now="2026-08-11T10:00:00")

        def fake_suggest(text):
            if text == "ap":
                return [("APL", "type")]
            if text == "apl":
                return [("APL_clock", "type"), ("APL_13", "type")]
            return []

        client = Client(page("/neuron-input-suggest-lifecycle"))
        with client:
            container = neuron_list_input(label="Source Neurons", suggestions=fake_suggest)

        chip = container.chip_input
        input_listeners = [
            listener for listener in chip._event_listeners.values()
            if listener.type == "input"
        ]
        suggest_input = input_listeners[0]
        menus = [el for el in client.elements.values() if type(el).__name__ == "Menu"]
        menu = menus[-1]
        focus = next(l for l in chip._event_listeners.values() if l.type == "focus")

        # Focus the empty field -> Recent list opens.
        chip._handle_event({"listener_id": focus.id, "args": None})
        assert menu.value is True
        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "Recent" in texts

        # Type 2 chars -> suggestions replace the history.
        chip._handle_event({"listener_id": suggest_input.id, "args": "ap"})
        assert menu.value is True
        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "APL" in texts and "Recent" not in texts

        # Every edit replaces the previous result set; it does not leave the
        # old suggestions mixed into the more precise query.
        chip._handle_event({"listener_id": suggest_input.id, "args": "apl"})
        assert menu.value is True
        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "APL_clock" in texts and "APL" not in texts

        # Pick the suggestion -> exactly one chip, and the finished input holds
        # the 'apl' rows (minus the picked one) for the next pick.
        def _subtree_texts(el):
            out = [getattr(el, "text", "")]
            for child in el.default_slot.children:
                out.extend(_subtree_texts(child))
            return out

        item = next(
            el for el in client.elements.values()
            if type(el).__name__ == "Item" and "APL_clock" in _subtree_texts(el)
        )
        click = next(l for l in item._event_listeners.values() if l.type == "click")
        from types import SimpleNamespace
        click.handler(SimpleNamespace())
        assert container.get_value() == ("exact", ["APL_clock"])
        assert menu.value is True  # held on the same query
        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "APL_13" in texts
        assert "Recent" not in texts
        assert "aMe12" not in texts

        # Focus change (no pointer in the menu) hides the list automatically,
        # commits nothing extra, and ends the hold.
        blur = next(l for l in chip._event_listeners.values() if l.type == "blur")
        chip._handle_event({
            "listener_id": blur.id,
            "args": {"still_inside": False},
        })
        assert menu.value is False
        assert container.get_value() == ("exact", ["APL_clock"])

        # Re-focusing now offers the history list, not the old query.
        focus = next(l for l in chip._event_listeners.values() if l.type == "focus")
        chip._handle_event({"listener_id": focus.id, "args": None})
        assert menu.value is True
        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "Recent" in texts and "APL_13" not in texts

        # A blur with pending text commits it like a plain blur (and hides).
        # Real typing fires both input listeners: the suggestion driver and
        # the pending-text tracker.
        chip._handle_event({"listener_id": focus.id, "args": None})
        chip._handle_event({"listener_id": suggest_input.id, "args": "aMe"})
        chip._handle_event({"listener_id": input_listeners[1].id, "args": "aMe"})
        chip._handle_event({"listener_id": blur.id, "args": None})
        assert menu.value is False
        assert container.get_value() == ("exact", ["APL_clock", "aMe"])

    def test_neuron_list_input_history_click_toggle(self, tmp_path, monkeypatch):
        """Click-to-toggle the history list on the focused empty editor.

        The click that focuses the field opens the list once; that same
        click must not immediately hide it again. Every further editor
        click flips the list: hide, show, hide. Clicks that did not land
        on the editor surface (chips, remove buttons, borders) never
        toggle anything.
        """
        from nicegui import Client
        from nicegui.page import page
        import ui.config as cfg_mod
        from ui.components.common import neuron_list_input

        monkeypatch.setattr(cfg_mod, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        import ui.history_store as hs
        monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
        hs.record(["aMe12", "aMe10"], now="2026-08-11T10:00:00")

        client = Client(page("/neuron-input-history-click-toggle"))
        with client:
            container = neuron_list_input(
                label="Source Neurons", suggestions=lambda text: [])

        chip = container.chip_input
        menu = [el for el in client.elements.values() if type(el).__name__ == "Menu"][-1]
        focus_listener = next(
            l for l in chip._event_listeners.values() if l.type == "focus")
        click_listener = next(
            l for l in chip._event_listeners.values() if l.type == "click")

        # The focusing click opens the history...
        chip._handle_event({"listener_id": focus_listener.id, "args": None})
        assert menu.value is True
        # ...and that same click's toggle is swallowed (focus already opened
        # the list — the menu must not flash closed on the opening click).
        chip._handle_event({"listener_id": click_listener.id, "args": True})
        assert menu.value is True
        # Next editor click hides; the next shows again; hide again.
        chip._handle_event({"listener_id": click_listener.id, "args": True})
        assert menu.value is False
        chip._handle_event({"listener_id": click_listener.id, "args": True})
        assert menu.value is True
        chip._handle_event({"listener_id": click_listener.id, "args": True})
        assert menu.value is False
        # A click that did not land on the editor surface never toggles.
        chip._handle_event({"listener_id": click_listener.id, "args": False})
        assert menu.value is False

    def test_neuron_list_input_click_toggle_spares_suggestions(
        self, tmp_path, monkeypatch,
    ):
        """While a suggestion list is showing (typed text), editor clicks
        never hide it — the toggle applies to the history list only."""
        from nicegui import Client
        from nicegui.page import page
        import ui.config as cfg_mod
        from ui.components.common import neuron_list_input

        monkeypatch.setattr(cfg_mod, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        import ui.history_store as hs
        monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
        hs.record(["aMe12", "aMe10"], now="2026-08-11T10:00:00")

        def fake_suggest(text):
            if text == "ap":
                return [("APL", "type")]
            return []

        client = Client(page("/neuron-input-click-toggle-suggestions"))
        with client:
            container = neuron_list_input(label="Source Neurons", suggestions=fake_suggest)

        chip = container.chip_input
        menu = [el for el in client.elements.values() if type(el).__name__ == "Menu"][-1]
        focus_listener = next(
            l for l in chip._event_listeners.values() if l.type == "focus")
        click_listener = next(
            l for l in chip._event_listeners.values() if l.type == "click")
        suggest_input = next(
            l for l in chip._event_listeners.values() if l.type == "input")

        chip._handle_event({"listener_id": focus_listener.id, "args": None})
        assert menu.value is True
        # Typing replaces the history with suggestions.
        chip._handle_event({"listener_id": suggest_input.id, "args": "ap"})
        assert menu.value is True
        # An editor click while suggestions are showing never hides them.
        chip._handle_event({"listener_id": click_listener.id, "args": True})
        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "APL" in texts
        assert menu.value is True

    @staticmethod
    def _ppl_pool(text):
        """One held-list provider: three names behind the 'PPL1' prefix."""
        if str(text).startswith("PPL1"):
            return [("PPL101", "type"), ("PPL102", "type"), ("PPL103", "type")]
        return []

    def _build_held_box(self, tmp_path, monkeypatch, route, provider,
                        show_history=True):
        """A standard query box over a persisted history, plus its listeners.

        ``show_history`` is applied only AFTER the config path is redirected:
        the toggles persist to ``ui/local_config.json``, and writing the
        developer's real settings from a test would silently disable the
        history list for every later run.
        """
        from nicegui import Client
        from nicegui.page import page
        import ui.config as cfg_mod
        from ui.components.common import neuron_list_input

        monkeypatch.setattr(cfg_mod, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        cfg_mod.set_show_history_enabled(show_history)
        import ui.history_store as hs
        monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
        hs.record(["aMe12"], now="2026-08-11T10:00:00")

        client = Client(page(route))
        with client:
            container = neuron_list_input(
                label="Source Neurons", suggestions=provider)
        chip = container.chip_input

        def listener(kind, index=0):
            return [
                l for l in chip._event_listeners.values() if l.type == kind
            ][index]

        return client, container, chip, container.suggest_menu, listener

    def test_neuron_list_input_holds_suggestions_across_consecutive_picks(
        self, tmp_path, monkeypatch
    ):
        """Typing 'PPL1' and picking two rows adds two chips from one list.

        The pick clears the editor text but keeps every row of the query listed
        in the same order, ticking the names that are already in the query
        rather than dropping them, so the list never shifts under the pointer.
        A ticked row toggles: the next click on it removes that chip. The
        candidate pool is reused, so the provider is never asked again for
        a held query.
        """
        from types import SimpleNamespace

        calls = []

        def provider(text):
            calls.append(str(text))
            return TestComponents._ppl_pool(text)

        client, box, chip, menu, listener = self._build_held_box(
            tmp_path, monkeypatch, "/neuron-input-held-suggestions", provider)
        focus, typing, editor_click = (
            listener("focus"), listener("input"), listener("click"))
        wipes = []
        real_run_method = chip.run_method
        chip.run_method = lambda name, *args, **kwargs: (
            wipes.append((name, args)), real_run_method(name, *args, **kwargs))[1]

        def subtree_texts(el):
            out = [getattr(el, "text", "")]
            for child in el.default_slot.children:
                out.extend(subtree_texts(child))
            return out

        def visible():
            return [el.text for el in client.elements.values()
                    if getattr(el, "text", "")]

        def pick(value):
            item = next(
                el for el in client.elements.values()
                if type(el).__name__ == "Item" and value in subtree_texts(el)
            )
            next(l for l in item._event_listeners.values()
                 if l.type == "click").handler(SimpleNamespace())

        chip._handle_event({"listener_id": focus.id, "args": None})
        # The click that caused the focus is swallowed (focus opened the list).
        chip._handle_event({"listener_id": editor_click.id, "args": True})
        chip._handle_event({"listener_id": typing.id, "args": "PPL1"})
        assert menu.value is True
        assert calls == ["PPL1"]

        def ticked():
            """Values whose row carries the already-in-the-query marker."""
            out = []
            for el in client.elements.values():
                if type(el).__name__ != "Item":
                    continue
                if not any(c == "drocat-suggest-added" for c in el._classes):
                    continue
                texts = [t for t in subtree_texts(el) if t]
                out.append(texts[0])
            return sorted(out)

        pick("PPL101")
        assert box.get_value() == ("exact", ["PPL101"])
        assert menu.value is True
        texts = visible()
        assert "PPL101" in texts and "PPL102" in texts and "PPL103" in texts
        assert "Recent" not in texts
        assert ticked() == ["PPL101"]
        # The chip is in the model by the time the wipe is issued, and
        # NiceGUI's outbox always sends element updates ahead of run_method
        # messages: the browser paints the chip first, then empties the field.
        assert chip.value == ["PPL101"]
        assert ("updateInputValue", ("",)) in wipes
        assert calls == ["PPL1"]  # held rows re-render from the cached pool

        # The list stayed put, so the second pick needs no new text.
        pick("PPL102")
        assert box.get_value() == ("exact", ["PPL101", "PPL102"])
        assert ticked() == ["PPL101", "PPL102"]

        # Clicking a ticked row is the reverse of the first click: that chip
        # comes back out and the rows stay listed in place, now unmarked.
        pick("PPL101")
        assert box.get_value() == ("exact", ["PPL102"])
        assert menu.value is True and ticked() == ["PPL102"]
        assert "PPL101" in visible()

        # Picking it a third time re-adds it, so a marked row really is a
        # checkbox rather than a one-way commit.
        pick("PPL101")
        assert box.get_value() == ("exact", ["PPL102", "PPL101"])
        assert ticked() == ["PPL101", "PPL102"]

        # Removing a chip through the field's own x clears its tick the same
        # way; the row was never gone.
        chip.set_value([])
        assert "PPL101" in visible()
        assert ticked() == []

        # A closing click is still the only thing that ends the held query, so
        # the next focus offers the history again.
        blur = listener("blur")
        chip._handle_event({
            "listener_id": blur.id,
            "args": {"still_inside": False},
        })
        assert menu.value is False
        chip._handle_event({"listener_id": focus.id, "args": None})
        assert "Recent" in visible()

    def test_neuron_list_input_editor_click_dismisses_the_held_list(
        self, tmp_path, monkeypatch
    ):
        """A click on the query box closes the held list and resets it.

        The dismissal click does not open the history list in the same
        breath; the next click shows Recent, so the held query never leaks
        back into a fresh round.
        """
        from types import SimpleNamespace

        client, box, chip, menu, listener = self._build_held_box(
            tmp_path, monkeypatch, "/neuron-input-held-dismiss",
            TestComponents._ppl_pool)
        focus, typing, editor_click = (
            listener("focus"), listener("input"), listener("click"))

        def subtree_texts(el):
            out = [getattr(el, "text", "")]
            for child in el.default_slot.children:
                out.extend(subtree_texts(child))
            return out

        chip._handle_event({"listener_id": focus.id, "args": None})
        chip._handle_event({"listener_id": editor_click.id, "args": True})
        chip._handle_event({"listener_id": typing.id, "args": "PPL1"})
        item = next(
            el for el in client.elements.values()
            if type(el).__name__ == "Item" and "PPL101" in subtree_texts(el)
        )
        next(l for l in item._event_listeners.values()
             if l.type == "click").handler(SimpleNamespace())
        assert menu.value is True and "PPL102" in [
            el.text for el in client.elements.values() if getattr(el, "text", "")]

        # Click one: the held list closes and the query is dropped.
        chip._handle_event({"listener_id": editor_click.id, "args": True})
        assert menu.value is False
        # Click two: the ordinary history toggle takes over from there.
        chip._handle_event({"listener_id": editor_click.id, "args": True})
        assert menu.value is True
        texts = [el.text for el in client.elements.values()
                 if getattr(el, "text", "")]
        assert "Recent" in texts and "PPL102" not in texts

    def test_neuron_list_input_history_pick_does_not_hold_suggestions(
        self, tmp_path, monkeypatch
    ):
        """Picking from the Recent list re-offers the Recent list, not a
        leftover type-ahead query."""
        from types import SimpleNamespace

        client, box, chip, menu, listener = self._build_held_box(
            tmp_path, monkeypatch, "/neuron-input-history-pick",
            TestComponents._ppl_pool)
        focus, typing = listener("focus"), listener("input")

        def subtree_texts(el):
            out = [getattr(el, "text", "")]
            for child in el.default_slot.children:
                out.extend(subtree_texts(child))
            return out

        # A blank focused editor shows the history; a pick from it must keep
        # showing the history even though a suggestion provider exists.
        chip._handle_event({"listener_id": focus.id, "args": None})
        item = next(
            el for el in client.elements.values()
            if type(el).__name__ == "Item" and "aMe12" in subtree_texts(el)
        )
        next(l for l in item._event_listeners.values()
             if l.type == "click").handler(SimpleNamespace())
        assert box.get_value() == ("exact", ["aMe12"])
        assert menu.value is True
        texts = [el.text for el in client.elements.values()
                 if getattr(el, "text", "")]
        assert "Recent" in texts and "PPL101" not in texts

        # New typed text starts its own round: the suggestions for it show.
        chip._handle_event({"listener_id": typing.id, "args": "PPL1"})
        texts = [el.text for el in client.elements.values()
                 if getattr(el, "text", "")]
        assert "PPL101" in texts and "Recent" not in texts

    def test_neuron_list_input_history_rows_mark_and_deselect(
        self, tmp_path, monkeypatch
    ):
        """A Recent row that is also a chip ticks, and clicking it drops the chip.

        The history half of the shared contract: the mark comes from the same
        ``marked_rows`` helper the type-ahead rows use, the deselect lands on the
        same model path as the chip's own ``x``, and the row's prune button stays
        a separate action -- which is why a marked row says in its hover text what
        the next click will do.
        """
        from types import SimpleNamespace

        client, box, chip, menu, listener = self._build_held_box(
            tmp_path, monkeypatch, "/neuron-input-history-deselect",
            TestComponents._ppl_pool)
        focus = listener("focus")

        def subtree_texts(el):
            out = [getattr(el, "text", "")]
            for child in el.default_slot.children:
                out.extend(subtree_texts(child))
            return out

        def rows_of(value):
            return [el for el in client.elements.values()
                    if type(el).__name__ == "Item"
                    and value in subtree_texts(el)]

        def ticked():
            return sorted(
                [text for text in subtree_texts(el) if text][0]
                for el in client.elements.values()
                if type(el).__name__ == "Item"
                and any(c == "drocat-suggest-added" for c in el._classes))

        def click_row(el):
            next(listener for listener in el._event_listeners.values()
                 if listener.type == "click").handler(SimpleNamespace())

        def descendants(el):
            for child in el.default_slot.children:
                yield child
                yield from descendants(child)
        chip._handle_event({"listener_id": focus.id, "args": None})
        texts = [el.text for el in client.elements.values()
                 if getattr(el, "text", "")]
        assert "Recent" in texts
        assert ticked() == []  # the field holds nothing yet

        click_row(rows_of("aMe12")[-1])
        assert box.get_value() == ("exact", ["aMe12"])
        assert menu.value is True
        assert ticked() == ["aMe12"]  # the history row carries the mark
        # The tick itself is a shared-class icon, so the CSS tints it the same
        # way the overlay's does (the row-level tint cannot reach .drocat-muted).
        assert any(type(el).__name__ == "Icon"
                   and "drocat-suggest-check" in el._classes
                   for el in descendants(rows_of("aMe12")[-1]))

        click_row(rows_of("aMe12")[-1])
        assert box.get_value() == ("exact", [])
        assert menu.value is True
        assert rows_of("aMe12"), "the row stays listed, unmarked"
        assert ticked() == []

        click_row(rows_of("aMe12")[-1])
        assert box.get_value() == ("exact", ["aMe12"])

        # The prune x on that marked row is still the other action: it drops the
        # history entry and leaves the chip alone. It closes the list on the way,
        # because pruning the last entry leaves nothing to offer.
        button = next(el for el in descendants(rows_of("aMe12")[-1])
                      if type(el).__name__ == "Button")
        # Built into the row and only hidden by CSS, so the reveal rule reaches it.
        assert "drocat-suggest-remove" in button._classes
        next(listener for listener in button._event_listeners.values()
             if listener.type == "click").handler(SimpleNamespace())
        assert box.get_value() == ("exact", ["aMe12"])
        # Pruning the last entry leaves nothing to offer, so the list closes and
        # rebuilds empty rather than leaving the pruned row mounted behind it.
        assert menu.value is False
        assert not rows_of("aMe12")

    def test_neuron_list_input_held_list_survives_history_disabled(
        self, tmp_path, monkeypatch
    ):
        """With the history list switched off, a pick keeps the suggestions
        open instead of closing the menu."""
        from types import SimpleNamespace

        client, box, chip, menu, listener = self._build_held_box(
            tmp_path, monkeypatch, "/neuron-input-held-no-history",
            TestComponents._ppl_pool, show_history=False)
        typing = listener("input")

        def subtree_texts(el):
            out = [getattr(el, "text", "")]
            for child in el.default_slot.children:
                out.extend(subtree_texts(child))
            return out

        chip._handle_event({"listener_id": typing.id, "args": "PPL1"})
        assert menu.value is True
        item = next(
            el for el in client.elements.values()
            if type(el).__name__ == "Item" and "PPL101" in subtree_texts(el)
        )
        next(l for l in item._event_listeners.values()
             if l.type == "click").handler(SimpleNamespace())
        assert box.get_value() == ("exact", ["PPL101"])
        assert menu.value is True
        texts = [el.text for el in client.elements.values()
                 if getattr(el, "text", "")]
        assert "PPL102" in texts

    def test_neuron_list_input_suggestions_settings_toggle(self, tmp_path, monkeypatch):
        """The Settings toggle switches the auto-suggest off/on at runtime."""
        from nicegui import Client
        from nicegui.page import page
        import ui.config as cfg_mod
        from ui.components.common import neuron_list_input

        monkeypatch.setattr(cfg_mod, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        cfg_mod.set_auto_suggest_enabled(False)

        def fake_suggest(text):
            return [("APL", "type")]

        client = Client(page("/neuron-input-suggest-toggle"))
        with client:
            container = neuron_list_input(label="Source Neurons", suggestions=fake_suggest)

        chip = container.chip_input
        input_listeners = [
            listener for listener in chip._event_listeners.values()
            if listener.type == "input"
        ]
        suggest_input = input_listeners[0]
        focus = next(l for l in chip._event_listeners.values() if l.type == "focus")
        menus = [el for el in client.elements.values() if type(el).__name__ == "Menu"]
        menu = menus[-1]

        # Disabled: focus and typing never open the menu.
        chip._handle_event({"listener_id": focus.id, "args": None})
        chip._handle_event({"listener_id": suggest_input.id, "args": "ap"})
        assert menu.value is False
        assert container.get_value() == ("exact", [])

        # Re-enabled: the very next keystroke works again.
        cfg_mod.set_auto_suggest_enabled(True)
        chip._handle_event({"listener_id": suggest_input.id, "args": "ap"})
        assert menu.value is True

        # Turning it off while the popup is visible must not leave a stale
        # menu behind after focus moves away.
        cfg_mod.set_auto_suggest_enabled(False)
        blur = next(l for l in chip._event_listeners.values() if l.type == "blur")
        chip._handle_event({"listener_id": blur.id, "args": None})
        assert menu.value is False

    def test_parse_neuron_upload_text_and_excel(self):
        import asyncio
        import io
        from types import SimpleNamespace
        import pandas as pd
        from ui.components.common import parse_neuron_upload, read_upload_event

        class NiceGUIFile:
            name = "neurons.csv"

            async def read(self):
                return b"type\naMe12\n"

        filename, content = asyncio.run(
            read_upload_event(SimpleNamespace(file=NiceGUIFile()))
        )
        assert (filename, content) == ("neurons.csv", b"type\naMe12\n")

        csv_bytes = b"type,notes\naMe12,first\naMe10,second\n"
        assert parse_neuron_upload("neurons.csv", csv_bytes) == ["aMe12", "aMe10"]
        tsv_bytes = b"bodyId\tlabel\n720575940610453042\tcell\n"
        assert parse_neuron_upload("neurons.tsv", tsv_bytes) == [720575940610453042]

        workbook = io.BytesIO()
        pd.DataFrame({"type": ["PPL101", "DN1p"]}).to_excel(workbook, index=False)
        assert parse_neuron_upload("neurons.xlsx", workbook.getvalue()) == ["PPL101", "DN1p"]

        numeric_workbook = io.BytesIO()
        pd.DataFrame({"bodyId": [42, None]}).to_excel(numeric_workbook, index=False)
        assert parse_neuron_upload("body_ids.xlsx", numeric_workbook.getvalue()) == [42]

    def test_output_panel(self):
        from ui.components.output_panel import OutputPanel
        panel = OutputPanel("Test")
        assert panel.title == "Test"
        assert panel._dom_id.startswith("drocat-results-")
        assert panel._files == []
        assert panel._format_size(500) == "500 B"
        assert panel._format_size(1536) == "1.5 KB"
        assert panel._format_size(1048576) == "1.0 MB"

    def test_output_panel_renders_one_status_badge(self):
        """The runner header owns the only status badge for the panel."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.output_panel import OutputPanel

        client = Client(page("/output-panel-single-status-test"))
        with client:
            panel = OutputPanel("Test")
            panel.create()

        assert panel.status_label is not None
        assert panel.page_progress is not None
        assert panel.page_progress.status_label is None

    def test_output_panel_progress_refreshes_same_bar_in_place(self):
        """Progress lines refresh their own bar; new bars start fresh lines."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.output_panel import OutputPanel

        client = Client(page("/output-panel-progress-test"))
        with client:
            panel = OutputPanel("Test")
            panel.create()

        bar_0 = "Building target profiles:   0%|          | 0/2972 [00:00<?, ?it/s]"
        bar_50 = "Building target profiles:  50%|█████     | 1488/2972 [00:06<00:05, 265.91it/s]"
        bar_other = "Loading data:   0%|          | 0/4 [00:00<?, ?it/s]"

        panel.log(bar_0, "progress")
        panel.log(bar_50, "progress")
        children = panel.log_area.default_slot.children
        assert len(children) == 1
        assert children[0].text == bar_50

        # A different bar starts a new line instead of overwriting the first.
        panel.log(bar_other, "progress")
        assert len(children) == 2
        assert children[1].text == bar_other

        # A normal line ends the refresh chain; the next progress line is new.
        panel.log("[HomologFinder] Built 2972 target profiles", "stdout")
        panel.log("Loading data:  50%|█████     | 2/4 [00:00<00:00,  9.64s/it]", "progress")
        assert len(children) == 4
        assert children[3].text.startswith("Loading data:  50%")

        # Trailing whitespace (tqdm line-clearing) is trimmed; blank lines drop.
        panel.log("   ", "stdout")
        panel.log("done   ", "stdout")
        assert len(children) == 5
        assert children[4].text == "done"

    def test_output_panel_counter_progress_updates_overall_bar(self):
        """tqdm/LineProgress counters advance the active outer step."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.output_panel import OutputPanel

        client = Client(page("/output-panel-counter-progress-test"))
        with client:
            panel = OutputPanel("Test")
            panel.create()

        panel.page_progress.start("find_path")
        panel.page_progress.update_phase("initialize", "Initializing Complete Paths")
        assert panel.page_progress.current_value == pytest.approx(0.0)
        assert panel.progress_label.text == (
            "Step 1/5: Initialize source and target neurons"
        )
        panel.page_progress.update_phase(
            "initialize_complete", "Initialization complete"
        )
        assert panel.page_progress.current_value == pytest.approx(0.2)
        assert panel.progress_label.text == (
            "Step 2/5: Discover connections layer by layer"
        )
        panel.page_progress.update_phase("execute", "Running Complete Paths")
        panel.log("Deriving type-level paths: 0/100 (0.0%) [00:00<?, 0.0it/s]", "progress")
        assert panel.page_progress.current_value == pytest.approx(1 / 5)
        assert panel.progress_label.text == "Step 2/5: Deriving type-level paths"

        panel.log(
            "Deriving type-level paths: 50/100 (50.0%) [00:01<00:01, 50.0it/s]",
            "progress",
        )
        assert panel.page_progress.current_value == pytest.approx(1.5 / 5)

        # The tqdm/LineProgress format is also refreshed in place in the log.
        assert len(panel.log_area.default_slot.children) == 1
        assert panel.log_area.default_slot.children[0].text.startswith(
            "Deriving type-level paths: 50/100"
        )

        stopped_value = panel.page_progress.current_value
        panel.set_running(False)
        assert panel.page_progress.current_value == pytest.approx(stopped_value)
        panel.set_status("Failed", "red")
        assert panel.progress_label.text == "Failed during Deriving type-level paths"

    def test_output_panel_step_progress_events_drive_determinate_bar(self):
        """[DROCAT][progress] events switch the bar to a determinate step
        fraction, set the step label, and never appear in the log.

        The bar reflects COMPLETED steps: a ``k/total`` event marks the
        START of step k, so it shows ``(k-1)/total``.  Even the final step
        event leaves the bar below 100%; only the run's completion status
        fills it.
        """
        from nicegui import Client
        from nicegui.page import page
        from ui.components.output_panel import OutputPanel

        client = Client(page("/output-panel-step-progress-test"))
        with client:
            panel = OutputPanel("Test")
            panel.create()
            # Indeterminate bar, like a live run (set_running(True) needs an
            # event loop for its scrollIntoView JS, so set it directly here).
            panel.progress_bar.props("indeterminate")

        # The event is consumed: bar value + label update, no log line.
        panel.log("[DROCAT][progress] 1/6 Resolving query neuron", "stdout")
        assert panel.progress_bar.value == pytest.approx(0.0)
        assert "indeterminate" not in panel.progress_bar._props
        assert panel.progress_label.text == "Step 1/6: Resolving query neuron"
        assert panel.log_area.default_slot.children == []

        # A later step moves the bar forward and replaces the label.
        panel.log("[DROCAT][progress] 3/6 Expanding candidate types to the scoring pool", "stdout")
        assert panel.progress_bar.value == pytest.approx(2 / 6)
        assert panel.progress_label.text == "Step 3/6: Expanding candidate types to the scoring pool"

        # Starting the final step must NOT show 100%: only completed steps
        # count, so the bar sits at 5/6 until the run completes.
        panel.log("[DROCAT][progress] 6/6 Saving results & visualization", "stdout")
        assert panel.progress_bar.value == pytest.approx(5 / 6)
        assert panel.progress_label.text == "Step 6/6: Saving results & visualization"
        panel.log("regular line", "stdout")
        assert len(panel.log_area.default_slot.children) == 1
        assert panel.log_area.default_slot.children[0].text == "regular line"

        # Ending the run keeps the progress row visible; the caller's final
        # status determines success vs failure.
        panel.set_running(False)
        assert panel.progress_label.text == "Step 6/6: Saving results & visualization"
        assert panel.progress_row.visible is True
        panel.set_status("Completed", "green")
        assert panel.progress_bar.value == pytest.approx(1.0)
        assert panel.progress_label.text == "Completed successfully!"
        assert panel.progress_row.visible is True

    def test_output_panel_places_function_progress_above_execution_log(self):
        """The tracker stays in the output card and uses backend stages."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import tool_page
        from ui.components.output_panel import OutputPanel

        client = Client(page("/page-progress-test"))
        with client:
            _form, results = tool_page("Progress Test")
            with results:
                panel = OutputPanel("Test")
                panel.create()

        assert panel.page_progress is not None
        assert panel.progress_bar._style.get("height") == "12px"
        assert panel.page_progress.progress_bar._style.get("height") == "12px"
        assert panel.page_progress.steps_container is None
        assert not any(
            "drocat-progress-step" in getattr(element, "_classes", [])
            for element in client.elements.values()
        )

        panel.page_progress.start("find_similar_morphology")
        assert panel.page_progress.step_labels == [
            "Resolve query neuron",
            "Screen candidates by ROI distributions",
            "Select top candidates for the scoring pool",
            "Load and vectorize skeletons",
            "Score morphological similarity",
            "Save results and visualization",
        ]
        panel.log(
            "[DROCAT][progress] 3/6 Expanding candidate types",
            "stdout",
        )
        assert panel.page_progress.current_value == pytest.approx(2 / 6)
        assert panel.page_progress.progress_label.text == (
            "Step 3/6: Expanding candidate types"
        )
        assert panel.page_progress.step_labels[2] == "Expanding candidate types"

        # A tool without backend step events still advances through the
        # generic lifecycle phases used by ScriptRunner.
        panel.page_progress.start("find_path")
        panel.page_progress.update_phase("initialize", "Initializing Complete Paths")
        assert panel.page_progress.current_value == pytest.approx(0.0)
        assert panel.page_progress.progress_label.text == (
            "Step 1/5: Initialize source and target neurons"
        )
        panel.page_progress.update_phase("initialize_complete")
        assert panel.page_progress.current_value == pytest.approx(0.2)
        assert panel.page_progress.progress_label.text == (
            "Step 2/5: Discover connections layer by layer"
        )

        panel.page_progress.update_step(2, 5, "Discovering connections")
        failed_value = panel.page_progress.current_value
        panel.page_progress.finish(False)
        assert panel.page_progress.current_value == pytest.approx(failed_value)
        assert panel.page_progress.progress_label.text == (
            "Failed during Discovering connections"
        )

        panel.page_progress.start(
            "find_homologs", method_name="find_homologs"
        )
        assert panel.page_progress.step_labels == [
            "Build source profiles",
            "Build target profiles",
            "Compare and score candidates",
            "Save homolog results",
        ]

        panel.page_progress.start(
            "find_similar_morphology",
            context={"candidate_source": "cache"},
        )
        assert panel.page_progress.step_labels == [
            "Resolve query neuron",
            "Load vector cache",
            "Score morphological similarity",
            "Save results and visualization",
        ]

    def test_progress_step_lists_follow_optional_run_parameters(self):
        """Optional backend phases (image downloads, reports, exports) only
        appear in a tab's step checklist when the matching parameter enables
        them, mirroring the backend's per-branch step totals."""
        from ui.components.page_progress import progress_steps_for

        # Find Driver Lines: no image download -> 3 steps, else 4.
        assert progress_steps_for("nb_find_lines", context={}) == [
            "Resolve EM neuron queries",
            "Fetch matching driver-line records",
            "Aggregate and rank line matches",
        ]
        assert progress_steps_for(
            "nb_find_lines",
            context={"download_images": "flylight", "output_dir": "/tmp/out"},
        ) == [
            "Resolve EM neuron queries",
            "Fetch matching driver-line records",
            "Aggregate and rank line matches",
            "Download images and save outputs",
        ]
        # The backend also gates the optional phase on an output folder.
        assert len(progress_steps_for(
            "nb_find_lines", context={"download_images": "flylight"}
        )) == 3

        # Find EM Neurons: 3D visualization is optional (flag + output dir).
        assert len(progress_steps_for("nb_find_neuron", context={})) == 3
        assert len(progress_steps_for(
            "nb_find_neuron",
            context={"visualize_top_n": 5, "output_dir": "/tmp/out"},
        )) == 4
        assert len(progress_steps_for(
            "nb_find_neuron", context={"visualize_top_n": 5}
        )) == 3

        # Co-labeling: report/3D visualization gate the final phase.
        assert len(progress_steps_for("nb_colabel", context={})) == 4
        assert len(progress_steps_for(
            "nb_colabel",
            context={"generate_report": True, "output_dir": "/tmp/out"},
        )) == 5
        assert len(progress_steps_for(
            "nb_colabel", context={"generate_report": True}
        )) == 4

        # FlyLight download: summary generation is optional.
        assert len(progress_steps_for("flylight_download", context={})) == 3
        assert len(progress_steps_for(
            "flylight_download", context={"generate_summary": "pdf"}
        )) == 4

        # 3D skeleton: export phases extend the three core phases.
        assert progress_steps_for("plot3d_skeleton", context={}) == [
            "Load skeletons",
            "Load synapses and meshes",
            "Render and save the 3D scene",
        ]
        both = progress_steps_for("plot3d_skeleton", context={
            "export_individual_profiles": True,
            "export_video": True,
        })
        assert both == [
            "Load skeletons",
            "Load synapses and meshes",
            "Render and save the 3D scene",
            "Export individual profiles",
            "Export rotating video",
        ]
        profiles_only = progress_steps_for(
            "plot3d_skeleton", context={"export_individual_profiles": True}
        )
        assert profiles_only[-1] == "Export individual profiles"
        assert len(profiles_only) == 4

        # Find Network: backend order is save data -> visualizations.
        assert progress_steps_for("find_network") == [
            "Initialize queried neurons",
            "Fetch mutual direct connections",
            "Enrich and filter network edges",
            "Save network data",
            "Build network visualizations",
        ]

        # Cross-dataset: tables export before visualizations/HTML report.
        assert progress_steps_for("inter_dataset") == [
            "Resolve datasets and thresholds",
            "Run path or edge analyses",
            "Compute cross-dataset metrics",
            "Export reports and result tables",
            "Generate comparison visualizations and HTML report",
        ]

    def test_output_dir_fields_sync_and_persist(self, tmp_path, monkeypatch):
        """Global changes update inherited fields but leave tab overrides
        alone; a forced Settings reset updates every field."""
        import ui.config as cfg
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import dir_input, sync_output_dir_fields

        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")

        client = Client(page("/dir-sync-test"))
        with client:
            field_a = dir_input(scope="tab_a")
            field_b = dir_input(scope="tab_b")

        # The blur/persist handler runs the same helper the test calls now.
        target = tmp_path / "permanent_outputs"
        saved, effective = cfg.set_default_output_dir(str(target), create=False)
        assert saved is True
        # sync what the handler would sync: every other field follows
        sync_output_dir_fields(field_a, effective)
        assert field_b.value == effective
        assert cfg.get_default_output_dir() == effective

        override = tmp_path / "tab_a_override"
        saved, _ = cfg.set_tab_output_dir("tab_a", str(override), create=False)
        assert saved is True
        field_a.value = str(override)
        field_a.classes(
            add="drocat-output-dir-override",
            remove="drocat-output-dir-inherited",
        )

        newer_default = tmp_path / "new_default"
        saved, effective = cfg.set_default_output_dir(
            str(newer_default), create=False
        )
        assert saved is True
        sync_output_dir_fields(field_a, effective)
        assert field_a.value == str(override)
        assert field_b.value == effective

        cfg.clear_tab_output_overrides()
        sync_output_dir_fields(field_b, effective, force=True)
        assert field_a.value == effective
        assert field_b.value == effective

        # A second dir_input built later picks up the persisted default.
        client2 = Client(page("/dir-sync-test-2"))
        with client2:
            field_c = dir_input(scope="tab_c")
        assert field_c.value == effective

    def test_output_panel_log_is_pointer_resizable(self):
        """The execution-log window must be drag-resizable: the wrapper owns
        the CSS resize handle with a definite height range, and the inner log
        fills the wrapper (h-full) plus scrolls, so dragging actually resizes
        the visible console instead of leaving a dead handle."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.output_panel import OutputPanel

        client = Client(page("/output-panel-resize-test"))
        with client:
            panel = OutputPanel("Test")
            panel.create()

        wrapper_style = panel.log_wrapper._style
        assert wrapper_style.get("resize") == "vertical"
        assert wrapper_style.get("overflow") == "hidden"  # required for resize
        assert wrapper_style.get("height") == "400px"     # definite start height
        assert wrapper_style.get("min-height") == "100px"
        assert wrapper_style.get("max-height") == "1800px"

        # The log tracks the wrapper so every drag changes the console size.
        assert "h-full" in panel.log_area._classes
        log_style = panel.log_area._style
        assert log_style.get("overflow-y") == "auto"

        # Streaming still works after the change (sanity for the console).
        panel.log("resizable log line", "stdout")
        assert panel.log_area.default_slot.children[0].text == "resizable log line"

    def test_output_panel_log_scrolls_freely(self):
        """The execution log must not force-jump to the newest line.

        NiceGUI's stock ui.log snaps the viewport to the bottom on every
        pushed line, which makes a streaming log unreadable anywhere but at
        the tail. OutputPanel therefore wires the FreeLog drop-in, which
        keeps the full ui.log API (push/max_lines/clear) while only its
        client-side scroll policy differs.
        """
        from nicegui import Client
        from nicegui.elements.log import Log
        from nicegui.page import page
        from ui.components.free_log import FreeLog
        from ui.components.output_panel import OutputPanel

        client = Client(page("/output-panel-free-scroll-test"))
        with client:
            panel = OutputPanel("Test")
            panel.create()

        assert isinstance(panel.log_area, FreeLog)
        assert isinstance(panel.log_area, Log)  # push/max_lines/callers' API preserved
        assert panel.log_area.max_lines == 500

        # Streaming still works after the swap.
        panel.log("free scroll line", "stdout")
        assert panel.log_area.default_slot.children[0].text == "free scroll line"

    def test_output_panel_log_has_copy_button(self):
        """The execution log must offer a clipboard copy action.

        The button sits in the log's heading row and dispatches a
        client-side copy routine that locates the log through its DOM id,
        so the copied text is exactly what the browser shows.
        """
        from nicegui import Client
        from nicegui.page import page
        from ui.components.output_panel import _COPY_LOG_JS, OutputPanel

        client = Client(page("/output-panel-log-copy-test"))
        with client:
            panel = OutputPanel("Test")
            panel.create()

        assert panel.copy_log_button is not None
        assert panel._log_dom_id.startswith("drocat-exec-log-")
        # The log element carries the DOM id the copy routine resolves.
        assert panel.log_area.props.get("id") == panel._log_dom_id

        # The routine is parameterized for this panel's log element.
        js = _COPY_LOG_JS.replace("__LOG_ID__", panel._log_dom_id)
        assert f"document.getElementById('{panel._log_dom_id}')" in js

        # Copying without a rendered panel is a no-op, not a crash.
        orphan = OutputPanel("Orphan")
        orphan._copy_log_to_clipboard()

    def _collect_panel_texts(self, container):
        """Recursively collect all label texts inside a UI container."""
        texts = []

        def walk(element):
            for child in element.default_slot.children:
                if hasattr(child, "text"):
                    texts.append(child.text)
                walk(child)

        walk(container)
        return texts

    def test_output_panel_streams_new_files_during_run(self, tmp_path):
        """Files created while a run is active appear in the panel immediately."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.output_panel import OutputPanel
        from ui.runner import ScriptRunner

        run_folder = tmp_path / "find-paths-complete_MCNS_aMe12_to_aMe10_L2w3_20260801_120000"
        run_folder.mkdir()
        (run_folder / "connections.csv").write_text("a,b\n1,2")

        client = Client(page("/output-panel-stream-test"))
        with client:
            panel = OutputPanel("Test")
            panel.create()
            runner = ScriptRunner()
            runner.is_running = True
            runner._run_logs = [("stdout", f"  📁 Created output folder: {run_folder}")]
            panel._poll_output_files(runner, str(tmp_path))

        texts = self._collect_panel_texts(panel.files_container)
        assert any("connections.csv" in text for text in texts), texts
        # the folder-structure display shows the run folder, not type categories
        assert not any("Data Tables (CSV)" in text for text in texts), texts

    def test_output_panel_streaming_skips_unknown_run_folder(self, tmp_path):
        """Before the run folder is known, polling must not show stale files."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.output_panel import OutputPanel
        from ui.runner import ScriptRunner

        (tmp_path / "old_run_connections.csv").write_text("a,b\n1,2")

        client = Client(page("/output-panel-stream-skip-test"))
        with client:
            panel = OutputPanel("Test")
            panel.create()
            runner = ScriptRunner()
            runner.is_running = True
            runner._run_logs = [("stdout", "some unrelated log line")]
            panel._poll_output_files(runner, str(tmp_path))

        texts = self._collect_panel_texts(panel.files_container)
        assert not any("old_run_connections.csv" in text for text in texts), texts

    def test_output_panel_show_files_mirrors_folder_structure(self, tmp_path):
        """Output files are grouped by their folder in the run directory
        (root files first, subfolders as nested expansions) - not by file
        type."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.output_panel import OutputPanel

        client = Client(page("/output-panel-tree-test"))
        with client:
            panel = OutputPanel("Test")
            panel.create()

        def file_entry(rel):
            return {"name": Path(rel).name, "path": str(tmp_path / rel), "size": 10,
                    "modified": "2026-08-01T00:00:00"}

        panel.show_files([
            file_entry("summary.csv"),
            file_entry("data_details/params.json"),
            file_entry("data_details/layer_1.csv"),
            file_entry("images/a.png"),
            file_entry("images/nested/b.png"),
        ], str(tmp_path))

        texts = self._collect_panel_texts(panel.files_container)
        # root file is listed directly
        assert any("summary.csv" in text for text in texts), texts
        # subfolders appear as expansions with their file counts
        assert any("data_details  (2)" in text for text in texts), texts
        assert any("images  (2)" in text for text in texts), texts
        # no type-category headers
        assert not any("Data Tables (CSV)" in text for text in texts), texts
        assert not any("Images" in text for text in texts), texts

    def test_output_panel_show_files_preserves_expanded_state(self, tmp_path):
        """Streaming refreshes keep the user's expanded folders open."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.output_panel import OutputPanel

        client = Client(page("/output-panel-expand-test"))
        with client:
            panel = OutputPanel("Test")
            panel.create()

        def file_entry(rel):
            return {"name": Path(rel).name, "path": str(tmp_path / rel), "size": 10,
                    "modified": "2026-08-01T00:00:00"}

        panel.show_files([file_entry("data_details/a.csv")], str(tmp_path))
        assert panel._file_expansions
        for expansion in panel._file_expansions.values():
            expansion.value = True  # user expands a folder

        # A streaming refresh with a new file arrives.
        panel.show_files([
            file_entry("data_details/a.csv"),
            file_entry("data_details/b.csv"),
        ], str(tmp_path))

        assert panel._file_expansions, "folders must be rebuilt"
        assert all(expansion.value for expansion in panel._file_expansions.values()), \
            "expanded folders must stay open after a streaming refresh"
        texts = self._collect_panel_texts(panel.files_container)
        assert any("b.csv" in text for text in texts), texts


# =============================================================================
# Test App Module
# =============================================================================
class TestApp:
    def test_app_imports(self):
        from ui.app import main, main_page
        assert callable(main)
        assert callable(main_page)

    def test_tab_toolbar_is_compact_and_non_scrolling(self):
        """The layered-card navigation must stay compact: group cards and
        tab segments shrink (min-width 0, equal flex) instead of scrolling;
        horizontal scrolling only exists as the small-screen fallback."""
        from ui.app import DROCAT_CSS

        assert ".drocat-nav {" in DROCAT_CSS
        assert ".drocat-group-card {" in DROCAT_CSS
        assert ".drocat-group-tab {" in DROCAT_CSS
        assert "flex: 1 1 0;" in DROCAT_CSS  # equal-width segments per card
        assert "min-width: 0;" in DROCAT_CSS
        # Two-row segments: icon on top, name below, tight line spacing.
        assert "flex-direction: column" in DROCAT_CSS
        # Multi-word names stack one word per line with tight leading.
        # (Quasar renders the button text in .q-btn__content .block here.)
        assert ".drocat-group-tab .q-btn__content .block" in DROCAT_CSS
        assert "white-space: pre-line;" in DROCAT_CSS
        assert "line-height: 1.15;" in DROCAT_CSS
        # The NB badge renders once, on the NeuronBridge group header.
        assert ".drocat-group-head.drocat-head-nb::after" in DROCAT_CSS
        assert ".drocat-tint-nb .drocat-group-tab .q-icon::after" not in DROCAT_CSS
        # Responsive fallback: horizontal scroll only below 700px.
        assert ".drocat-nav { overflow-x: auto; scrollbar-width: none; }" in DROCAT_CSS
        assert ".q-tooltip {" in DROCAT_CSS
        assert "font-size: 14px !important;" in DROCAT_CSS

    def test_settings_never_prefills_saved_tokens_in_browser_dom(self, tmp_path, monkeypatch):
        """Saved secrets remain server-side instead of entering the client DOM."""
        from nicegui import Client
        from nicegui.page import page
        import ui.tabs.settings as settings_module

        secret_neuprint = "neuprint-secret-for-dom-test"
        secret_cave = "cave-secret-for-dom-test"
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "'
            + secret_neuprint
            + '", "cave": "'
            + secret_cave
            + '"}}\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)

        client = Client(page("/settings-dom-test"))
        with client:
            settings_module.create_settings_tab()

        input_props = [
            getattr(element, "_props", {})
            for element in client.elements.values()
            if getattr(element, "_props", {}).get("label")
            in {"NeuPrint Token", "CAVE Token (for FAFB)"}
        ]
        assert len(input_props) == 2
        assert all(props.get("value", "") == "" for props in input_props)
        assert secret_neuprint not in repr(client.elements)
        assert secret_cave not in repr(client.elements)

    def test_guide_html_files_and_links_are_direct(self):
        """Instruction links must point to real local HTML guides with valid
        internal links."""
        import re as _re
        from nicegui.page import page
        from nicegui import Client
        import ui.app as app_module

        p2 = page('/smoke-guide')
        client2 = Client(p2)
        with client2:
            app_module.main_page()
        hrefs = [
            str((getattr(e, '_props', None) or {}).get('href', ''))
            for e in client2.elements.values()
        ]
        guide_links = [h for h in hrefs if h.startswith('docs/ui_guides/')]
        assert 'docs/ui_guides/find_path.md'.replace('.md', '.html') in guide_links
        assert 'docs/ui_guides/input_formats.html' in guide_links
        assert 'docs/ui_guides/README.html' in guide_links

        # Every linked HTML guide must exist and its internal links must resolve
        guides_dir = PROJECT_ROOT / 'docs' / 'ui_guides'
        html_files = sorted(guides_dir.glob('*.html'))
        assert len(html_files) >= 12
        for html_file in html_files:
            text = html_file.read_text(encoding='utf-8')
            assert text.lstrip().startswith('<!doctype html>')
            assert '<title>' in text
            assert 'guide.css' in text
            # Theme support: the guide must load the shared theme resolver
            # so an opened page matches the app's dark/light mode.
            assert '<script src="guide-theme.js"></script>' in text
            for target in _re.findall(r'href="([^"]+\.html)"', text):
                if target == 'guide.css' or target.startswith(('http:', 'https:')):
                    continue
                assert (guides_dir / target).exists(), f'{html_file.name} -> {target}'

        # The theme resolver reads the app's drocat_dark cookie and exposes
        # the resolved mode as <html data-theme>, which guide.css themes.
        # It also injects the app-style top-right toggle: one click flips
        # light/dark, persists to the same cookie, and broadcasts
        # 'drocat-theme' so already-open pages follow.
        theme_js = guides_dir / 'guide-theme.js'
        assert theme_js.exists()
        theme_js_text = theme_js.read_text(encoding='utf-8')
        assert 'drocat_dark' in theme_js_text
        assert 'data-theme' in theme_js_text
        assert 'drocat-guide-theme-toggle' in theme_js_text
        assert 'drocat-theme' in theme_js_text
        guide_css = (guides_dir / 'guide.css').read_text(encoding='utf-8')
        assert ':root[data-theme="dark"]' in guide_css
        assert '.drocat-guide-theme-toggle' in guide_css


# =============================================================================
# Test Installer Scripts
# =============================================================================
class TestInstallerScripts:
    def test_all_scripts_exist(self):
        for f in ["mac_DROCAT.command", "windows_DROCAT.bat",
                  "archive/install/install.sh", "archive/install/install.bat",
                  "archive/install/install.ps1"]:
            assert (PROJECT_ROOT / f).exists(), f"Missing: {f}"

    def test_shell_scripts_executable(self):
        assert os.access(PROJECT_ROOT / "mac_DROCAT.command", os.X_OK)
        assert os.access(PROJECT_ROOT / "archive/install/install.sh", os.X_OK)


# =============================================================================
# Test pyproject.toml
# =============================================================================
class TestPyproject:
    def test_version(self):
        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10
            import tomli as tomllib
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["project"]["version"] == "4.5.0"
        assert "ui" in data["project"]["optional-dependencies"]
        assert "coana" in data["tool"]["setuptools"]["py-modules"]
        assert "neuronbridge_client" in data["tool"]["setuptools"]["py-modules"]


# =============================================================================
# Integration: HTTP Server
# =============================================================================
class TestHTTPServer:
    def test_server_starts_and_responds(self):
        import socket
        import subprocess
        import time
        import urllib.error
        import urllib.request

        with socket.socket() as port_probe:
            port_probe.bind(("127.0.0.1", 0))
            port = port_probe.getsockname()[1]

        child_env = os.environ.copy()
        # NiceGUI enables its own screen-test mode when this pytest variable is
        # inherited, which requires unrelated test-runner configuration.
        child_env.pop("PYTEST_CURRENT_TEST", None)
        child_env.pop("NICEGUI_SCREEN_TEST_PORT", None)
        child_env["DROCAT_UI_PORT"] = str(port)
        child_env["DROCAT_UI_SHOW"] = "0"
        proc = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "ui" / "app.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
            env=child_env,
            text=True,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            # Cold start of ui/app.py (imports plotly/k3d/open3d/nicegui) can
            # take 20+ s on slower machines, so allow a generous deadline.
            deadline = time.monotonic() + 120
            while True:
                if proc.poll() is not None:
                    stdout, stderr = proc.communicate()
                    pytest.fail(
                        "UI server exited before becoming ready.\n"
                        f"stdout:\n{stdout}\nstderr:\n{stderr}"
                    )
                try:
                    with urllib.request.urlopen(url, timeout=5) as response:
                        html = response.read().decode()
                        assert response.status == 200
                        assert "DROCAT" in html
                        assert "drocat-cobalt" in html
                        break
                except (urllib.error.URLError, OSError) as error:
                    # urlopen+read share the 1s socket timeout; a slow first
                    # render of the ~680 KB DROCAT page can exceed it and
                    # raise a bare TimeoutError (an OSError, NOT wrapped in
                    # URLError) — retry until the deadline either way.
                    if time.monotonic() >= deadline:
                        pytest.fail(f"UI server did not become ready: {error}")
                    time.sleep(0.2)
        finally:
            if proc.poll() is None:
                proc.terminate()
            proc.communicate(timeout=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
