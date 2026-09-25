"""Type Validation (TM VEV) tab — bodyId-level validation of a type mapping.

Drives ``src/comparison/mapping_validation.py`` (``MappingValidationConfig`` +
``MappingValidator.run()``) through the runner's bespoke
``type_mapping_validation`` generator. Payload keys are always the *field*
names of the dataclass (the CLI's flag names differ and several are negated),
so the parity between this form and the dataclass is what keeps the tab honest.
"""

import glob
import json
import logging
import os

from nicegui import ui

from ..config import get_user_default
from ..components.common import (
    dataset_selector, neuron_list_input, number_input, checkbox_input,
    dir_input, section_header, param_grid, tool_page, apply_filter_mode,
)
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner, open_file, open_folder
from ..type_suggestions import dataset_aware_suggestions

# `pooling` is a fourth CLI value but NOT a rung of the nested ladder: it is
# outside VALIDATION_MODES/MODE_RANK, so a pooling run reports mode_rank null and
# the three nested bins keep the same meaning it gave them. The dropdown offers
# it because the tab must be able to start the mode, not because it is wider.
MODE_OPTIONS = ["restrictive", "family", "aggressive", "pooling"]
MODE_HINTS = {
    "restrictive": "matched / verified / borderline tiers only — the like-for-like spine.",
    "family": "adds the sibling / candidates / family bins (mapped relatives).",
    "aggressive": "adds the relative tier and the deep-window examinees bin (widest, slowest).",
    "pooling": "not a wider mode: a PARALLEL unsupervised homolog search over "
               "the whole target universe (absolute floors, morphology last), "
               "compared with the mapper afterwards. Writes pooling/ beside the "
               "bins above and changes none of them.",
}

# Field-name -> default fallback, used only if the backend dataclass cannot be
# imported in the UI process; normally the live dataclass defaults win.
_FALLBACK_DEFAULTS = {
    "rank_top_k": 5, "gap_min": 1, "verified_top_n": 2,
    "invader_borderline_max": 3, "matched_ru_min": 0.1,
    "candidate_morph_factor": 0.25, "target_min_weight": 10.0,
    "target_min_partner_types": 2, "suspicious_jaccard_factor": 0.5,
    "target_min_size_ratio": 0.1, "suspicious_ru_margin": 0.02,
    "pool_ref_cap": 6, "pool_ref_floor_margin": 0.05,
    "morph_track_a_offset": 0.05, "morph_suspicious_level": 3,
    "out_map_top_k": 10, "candidate_window": 25, "deep_cap": 10,
    "null_jaccard_max": 0.05, "null_per_source_cap": 5, "null_min_n": 10,
    "null_percentile": 95.0, "candidate_morph_cap": 20,
    "morph_auc_floor": 0.65, "max_scenes": 0, "neuron_alpha": 0.2,
    "backward_top_n": 5, "backward_max_neurons": 300,
    "backward_per_branch_cap": 40,
    "include_untyped_partners": True, "backward_scan_pool_targets": True,
    "scene_selfcheck": False, "verify_suspects": False,
    "pooling_jaccard_floor": 0.10, "pooling_rank_union_floor": 0.0,
    "pooling_window_mult": 2.0,
    "pooling_bar_metric": "either", "pooling_bar_top_n": 3,
    "pooling_max_morph_targets": 0,
}
_RATIO_FLOATS = {
    "matched_ru_min", "candidate_morph_factor", "suspicious_jaccard_factor",
    "target_min_size_ratio", "suspicious_ru_margin", "pool_ref_floor_margin",
    "morph_track_a_offset", "null_jaccard_max", "neuron_alpha", "morph_auc_floor",
}
# Numeric knobs held in the collapsed Advanced card. The backward budget knobs
# live on the Stages card (revealed under Backward evidence) instead.
ADV_NUM = [
    "rank_top_k", "gap_min", "verified_top_n", "invader_borderline_max",
    "matched_ru_min", "candidate_morph_factor", "target_min_weight",
    "target_min_partner_types", "suspicious_jaccard_factor",
    "target_min_size_ratio", "suspicious_ru_margin", "pool_ref_cap",
    "pool_ref_floor_margin", "morph_track_a_offset", "morph_suspicious_level",
    "out_map_top_k", "candidate_window", "deep_cap", "null_jaccard_max",
    "null_per_source_cap", "null_min_n", "null_percentile", "candidate_morph_cap",
]
ADV_BOOL = ["include_untyped_partners", "scene_selfcheck"]


def _field_defaults():
    """Live MappingValidationConfig defaults; fall back to the literal map."""
    try:
        import dataclasses
        from comparison.mapping_validation import MappingValidationConfig as C
        out = {}
        for f in dataclasses.fields(C):
            if f.default is not dataclasses.MISSING:
                out[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:
                out[f.name] = f.default_factory()
        return out
    except Exception:
        return dict(_FALLBACK_DEFAULTS)


def _default(field):
    return _FIELD_DEFAULTS.get(field, _FALLBACK_DEFAULTS.get(field))


def _setting(key, fallback):
    try:
        return get_user_default(key)
    except Exception:
        return fallback


def _bounds(field, value):
    if field in _RATIO_FLOATS:
        return 0.0, 1.0, 0.01
    if field == "null_percentile":
        return 0.0, 100.0, 1.0
    if field == "target_min_weight":
        return 0.0, 100000.0, 0.5
    if isinstance(value, float):
        return 0.0, 1000.0, 0.01
    return 0, 100000, 1


_FIELD_DEFAULTS = _field_defaults()


def _scene_cap_line(run_dir):
    """Return the backend's scenes-cap line from the run's README.txt, or None.

    The stage-4 cap is logged through ``validator.log``, so besides stdout it
    lands in README.txt's run log; the tab surfaces it as a persistent notice
    instead of leaving it buried in the log stream (cap-and-warn, §3.2).
    """
    readme = os.path.join(run_dir, "README.txt")
    if not os.path.isfile(readme):
        return None
    with open(readme, encoding="utf-8", errors="replace") as rh:
        for _line in rh:
            if "[stage 4] scene cap:" in _line:
                return _line.strip()
    return None


def create_type_validation_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("Type Validation Output",
                               state_key="type_mapping_validation")
    source_dataset = None
    target_dataset = None

    form_col, results_col = tool_page(
        "Type Validation",
        "BodyId-level validation of a source→target type mapping: verify branches, "
        "expand candidates, render review scenes. Proposals only — the mapping is "
        "never rewritten.",
        icon="policy",
        doc="type_validation.md",
    )

    def _query_suggest(text):
        ds = [source_dataset.value] if source_dataset is not None and source_dataset.value else []
        # scope pinned to 'auto': coarse cell_type categories resolve only there.
        return dataset_aware_suggestions(text, ds, "auto", limit=None)

    with form_col:
        # --- Datasets & Run ---
        with ui.card().classes("w-full drocat-card").props('id="card-tmvev-datasets"'):
            section_header("Datasets & Run", "storage")
            with param_grid(2):
                source_dataset = dataset_selector(
                    "Source Dataset", default=_setting("default_dataset", None))
                # Target defaults to FAFB (user 2026-09-22) — deliberately NOT
                # the Settings default_target_dataset key, which stays owned by
                # Connectivity → Find Similar. The hint makes the departure
                # visible in the UI.
                target_dataset = dataset_selector(
                    "Target Dataset", default="flywire_FAFB_v783",
                    hint="Defaults to flywire_FAFB_v783. Independent of the "
                         "Settings 'Default Similar-Search Target Dataset' "
                         "(that default belongs to Connectivity → Find "
                         "Similar).")
                run_label = ui.input("Run Label (optional)").classes("w-full drocat-input")
                output_dir = dir_input(scope="type_mapping_validation")
            same_pair_note = ui.label(
                "Source and Target are the same dataset — validation compares a "
                "dataset against itself and usually resolves no cross-dataset branches."
            ).classes("text-caption text-amber-8").set_visibility(False)

            def _sync_pair_note(_e=None):
                same_pair_note.set_visibility(
                    bool(source_dataset.value) and source_dataset.value == target_dataset.value)

            source_dataset.on_value_change(_sync_pair_note)
            target_dataset.on_value_change(_sync_pair_note)
            _sync_pair_note()

        # --- Query ---
        with ui.card().classes("w-full drocat-card").props('id="card-tmvev-query"'):
            section_header("Query", "hub")
            query_input = neuron_list_input(
                label="Source Types / Categories",
                placeholder="e.g. APDN3, s-CPDN3A, or a coarse category like circadian_clock",
                hint="Type one query per chip: a neuron type (APDN3, s-CPDN3A) or a coarse "
                     "cell_type category (e.g. circadian_clock). Empty query is rejected — the "
                     "backend defaults to no queries and would emit an empty run.",
                suggestions=_query_suggest,
                available_neurons=lambda: [source_dataset.value] if source_dataset is not None and source_dataset.value else [],
                show_history_datasets=True,
            ).classes("drocat-fixed-neuron-input")
            # §3.4 informational entry: what the auto-mapper CLAIMS for this
            # pair. It is NOT this run's directional truth (that is the run's
            # own mapping/mapping_export.csv), so label + tooltip say so.
            try:
                from ..components.type_mapping_panel import create_type_mapping_entry
                with ui.row().classes("items-center gap-3 flex-wrap"):
                    tm_preview = create_type_mapping_entry(
                        lambda: [d for d in (source_dataset.value,
                                             target_dataset.value) if d])

                    def _sync_tm_state(_e=None):
                        try:
                            tm_preview.refresh_state()
                        except Exception:
                            pass
                    source_dataset.on_value_change(_sync_tm_state)
                    target_dataset.on_value_change(_sync_tm_state)
                    _sync_tm_state()
                ui.label("What the auto-mapper claims (informational; direction is "
                         "the mapper's derivation, not this run's).").classes(
                    "text-caption opacity-60 w-full")
            except Exception:
                pass

        # --- Validation Mode ---
        with ui.card().classes("w-full drocat-card").props('id="card-tmvev-mode"'):
            section_header("Validation Mode", "tune")
            mode_value = {"value": _setting("tmvev_mode", "restrictive")}
            if mode_value["value"] not in MODE_OPTIONS:
                mode_value["value"] = "restrictive"
            mode_buttons = {}
            with ui.row().classes("w-full items-center justify-between gap-4 px-2 flex-wrap"):
                for _mode in MODE_OPTIONS:
                    _b = ui.button(_mode.capitalize()).props("outline no-caps").classes("w-1/4")
                    _b.style("min-height: 3rem; font-size: 1.0rem; font-weight: 700;")
                    mode_buttons[_mode] = _b
            mode_hint = ui.label("").classes("text-xs opacity-60 w-full")

            def _sync_mode():
                mode_hint.text = MODE_HINTS.get(mode_value["value"], "")
                for _m, _b in mode_buttons.items():
                    _b.props("color=primary" if _m == mode_value["value"] else "color=grey-7")

            def _set_mode(_e=None, *, mode=None):
                if mode:
                    mode_value["value"] = mode
                _sync_mode()
                _sync_stage_visibility()
                if _sync_pooling_visibility:
                    _sync_pooling_visibility()

            for _m, _b in mode_buttons.items():
                _b.on_click(lambda _e, mode=_m: _set_mode(mode=mode))
            _sync_mode()

        # --- Pooling gate (only read when the mode is `pooling`) ---
        _sync_pooling_visibility = None
        with ui.card().classes("w-full drocat-card").props('id="card-tmvev-pooling"') as pooling_card:
            section_header("Pooling gate (unsupervised mode)", "public")
            ui.label(
                "Read only in mode `pooling`: every queried source keeps the "
                "top-N of each chosen metric over the whole target universe, "
                "so no branch pool decides a row. Morphology then qualifies the "
                "findings — it is mandatory here, because each tier is defined "
                "as morph-qualified. The run writes pooling/ beside the nested "
                "bins and joins the mapper afterwards, as advice only."
            ).classes("text-caption opacity-70 w-full")
            pooling_bar_metric = ui.select(
                {'either': 'either metric (the union of both top-N)',
                 'jaccard': 'jaccard only',
                 'rank_union': 'rank_union only'},
                value=str(_default("pooling_bar_metric")), with_input=False,
            ).classes("w-72").tooltip(
                "What decides admission. `either` is each metric's own top-N, "
                "unioned — the same reading the matched tier uses. A merged "
                "best-rank ordering is NOT used: it spends the slots on the two "
                "metrics' rank-1 rows and keeps 79 of the 118 targets the union "
                "keeps 116 of.")
            pooling_bar_top_n = number_input(
                "Bar depth (top-N per metric)",
                int(_default("pooling_bar_top_n")), 1, 100, 1,
                hint="How deep each metric's rank list goes. The default admits "
                     "~1 candidate target per queried source and keeps 0.92 of "
                     "them after morphology on FAFB->male-cns (272 found / 222 "
                     "kept over 242 sources) and 0.86 on FAFB->BANC. Each source "
                     "is capped at 2N rows because rank_union ties (17 rows/"
                     "source at N=5), and the run publishes how many the cap "
                     "cut.")
            pooling_j_floor = number_input(
                "Jaccard floor (advisory flag)",
                float(_default("pooling_jaccard_floor")),
                0.0, 1.0, 0.01,
                hint="No longer a filter: every admitted row is measured "
                     "against it and published as below_jaccard_floor, and "
                     "nothing is removed for missing it. The bar above decides "
                     "admission; raise this only to read the flag differently.")
            pooling_ru_floor = number_input(
                "rank_union floor (advisory flag)",
                float(_default("pooling_rank_union_floor")),
                -1.0, 1.0, 0.01,
                hint="Advisory, like the Jaccard floor, and its default 0 is "
                     "deliberate: it only asks that the union be positive. When "
                     "it was still a filter that sign test starved 119 of 242 "
                     "queried sources, which is why rank_union is never a bar.")
            pooling_window_mult = number_input(
                "Window multiplier (advisory flag)",
                float(_default("pooling_window_mult")),
                0.5, 20.0, 0.5,
                hint="The window the outside_window flag measures against = this "
                     "x the size of the source neuron's own type population. "
                     "Advisory: it removed nothing even when it was a filter.")
            pooling_budget = number_input(
                "Morph budget (scoring units)",
                int(_default("pooling_max_morph_targets")), 0, 100000, 1,
                hint="0 = auto: 3 x the number of queried source neurons, so the "
                     "budget scales with the work instead of with a constant. "
                     "Rows past the budget read morph_gate='not-attempted-cap', "
                     "never blank, and morph.capped names them.")

            def _sync_pooling():
                pooling_card.set_visibility(mode_value["value"] == "pooling")

        _sync_pooling_visibility = _sync_pooling
        _sync_pooling()

        # --- Stages ---
        with ui.card().classes("w-full drocat-card").props('id="card-tmvev-stages"'):
            section_header("Stages", "layers")
            morph_enabled = checkbox_input(
                "Morphology verification", True,
                hint="Track A/B morphology self-calibration (slow; may load skeleton vectors).")
            morph_auc_floor = number_input(
                "Morph AUC floor", _default("morph_auc_floor"), *_bounds("morph_auc_floor", _default("morph_auc_floor")),
                hint="Minimum qualified-area AUC before a branch counts as morph-consistent.")
            visualize = checkbox_input(
                "Render 3D review scenes", True,
                hint="Emit branches_*.html skeletons for matched/verified pairs (capped below).")
            with param_grid(2):
                max_scenes = number_input(
                    "Max scenes (0 = every parent)",
                    int(_setting("tmvev_max_scenes", _default("max_scenes"))),
                    0, 50, 1,
                    hint="0 renders one scene per parent type; a positive value "
                         "caps the scenes and names the parents it dropped.")
                neuron_alpha = number_input(
                    "Neuron alpha", _default("neuron_alpha"), *_bounds("neuron_alpha", _default("neuron_alpha")),
                    hint="Skeleton opacity in review scenes (0-1).")
            backward_enabled = checkbox_input(
                "Backward (reciprocal) homolog evidence", False,
                hint="Reverse-scan target neurons for reciprocal evidence — the slowest stage "
                     "(≈4-7 s × up to hundreds of scans). Default off.")
            with param_grid(3):
                backward_top_n = number_input(
                    "Backward top N", _default("backward_top_n"), 0, 100, 1)
                backward_max_neurons = number_input(
                    "Backward max neurons", _default("backward_max_neurons"), 0, 10000, 1)
                backward_per_branch_cap = number_input(
                    "Backward per-branch cap", _default("backward_per_branch_cap"), 0, 10000, 1)
                backward_scan_pool_targets = checkbox_input(
                    "Backward scans unmatched pool members",
                    _default("backward_scan_pool_targets"),
                    hint="Also reverse-scan each branch pool's UNMATCHED members, so a "
                         "source sees its out-of-branch competitors. Already-mapped members "
                         "are skipped — their forward pair is the symmetric evidence.")
            skip_out_map = checkbox_input(
                "Skip out-map expansion", False,
                hint="Do not scan unclaimed source neurons for extra candidates.")
            verify_suspects = checkbox_input(
                "Verify suspects (advisory)", _default("verify_suspects"),
                hint="Additionally verify excluded/suspect source types (advisory pass).")

            cost_note = ui.label(
                "Cost scales fast: a coarse category, aggressive or pooling "
                "mode, or backward evidence each "
                "multiply runtime. Start with a type-level restrictive pass. Max scenes 0 renders a "
                "page per parent type, so a coarse query pays for all of them — cap it or split "
                "the query."
            ).classes("text-caption text-amber-8").set_visibility(False)

            def _sync_stage_visibility(_e=None):
                morph_auc_floor.set_visibility(morph_enabled.value)
                max_scenes.set_visibility(visualize.value)
                neuron_alpha.set_visibility(visualize.value)
                backward_top_n.set_visibility(backward_enabled.value)
                backward_max_neurons.set_visibility(backward_enabled.value)
                backward_per_branch_cap.set_visibility(backward_enabled.value)
                backward_scan_pool_targets.set_visibility(backward_enabled.value)
                _mode = mode_value["value"]
                _many = len(query_input.get_value()[1] or []) > 1
                cost_note.set_visibility(
                    bool(backward_enabled.value
                         or _mode in ("aggressive", "pooling") or _many))

            for w in (morph_enabled, visualize, backward_enabled, query_input):
                if hasattr(w, "on_value_change"):
                    w.on_value_change(_sync_stage_visibility)
            _sync_stage_visibility()

        # --- Data & Offline ---
        with ui.card().classes("w-full drocat-card").props('id="card-tmvev-data"'):
            section_header("Data & Offline", "cloud_off")
            use_cache = checkbox_input(
                "Use connection cache", bool(_setting("use_cache", True)),
                hint="False disables the disk cache and forces a refetch — it is NOT an offline "
                     "switch (it increases network use).")
            skip_profile_build = checkbox_input(
                "Cache-only profiles (skip online profile build)",
                bool(_default("skip_profile_build")),
                hint="Stay cache-only: do not reach NeuPrint to build target profiles. This is the "
                     "offline control, together with Morphology off + Scenes off for a fully offline run.")

        # --- Advanced (collapsed) ---
        with ui.card().classes("w-full drocat-card").props('id="card-tmvev-advanced"'):
            with ui.expansion("Advanced Settings", icon="settings_suggest").classes(
                    "w-full drocat-section-expansion"):
                ui.label("Unset fields keep the pipeline's own defaults.").classes(
                    "text-caption opacity-60 w-full")
                adv = {}
                with param_grid(3):
                    for field in ADV_NUM:
                        val = _default(field)
                        lo, hi, step = _bounds(field, val)
                        label = field.replace("_", " ").title()
                        adv[field] = (number_input(label, val, lo, hi, step), "num", isinstance(val, float))
                    for field in ADV_BOOL:
                        cb = checkbox_input(field.replace("_", " ").title(), bool(_default(field)))
                        adv[field] = (cb, "bool", False)

    with results_col:
        output_panel.create(run_label="Run Validation", run_icon="policy")
        action_row = ui.row().classes("w-full items-center gap-2")

    async def run_validation():
        _mode, queries = query_input.get_value()
        queries = apply_filter_mode(queries, _mode)
        queries = [str(q).strip() for q in (queries or []) if str(q).strip()]
        if not queries:
            ui.notify("Provide at least one source type or category to validate.",
                      type="warning")
            return
        source = source_dataset.value
        target = target_dataset.value
        if not source or not target:
            ui.notify("Select both a source and a target dataset.", type="warning")
            return
        if source == target:
            ui.notify("Source and target are the same dataset.", type="warning")
            return

        cp = {
            "source_dataset": source,
            "target_dataset": target,
            "query_types": queries,
            "validation_mode": mode_value["value"],
            # the pooling gate: read only in mode `pooling`, sent always so the
            # payload keeps mirroring the dataclass surface rather than the
            # CLI's flag names
            "pooling_jaccard_floor": float(pooling_j_floor.value),
            "pooling_rank_union_floor": float(pooling_ru_floor.value),
            "pooling_window_mult": float(pooling_window_mult.value),
            "pooling_bar_metric": str(pooling_bar_metric.value),
            "pooling_bar_top_n": int(pooling_bar_top_n.value),
            "pooling_max_morph_targets": int(pooling_budget.value),
            "aggressive_expansion": False,
            "pool_widen": False,
            "morph_enabled": bool(morph_enabled.value),
            "visualize": bool(visualize.value),
            "backward_evidence_enabled": bool(backward_enabled.value),
            "skip_out_map_expansion": bool(skip_out_map.value),
            "verify_suspects": bool(verify_suspects.value),
            "skip_profile_build": bool(skip_profile_build.value),
            "use_cache": bool(use_cache.value),
            "output_dir": output_dir.value,
            "run_label": (run_label.value or "").strip() or None,
            "verbose": True,
            "top_k": int(_setting("top_k", 25) or 25),
            "top_m": int(_setting("top_m", 5) or 5),
            "min_synapse_threshold": int(_setting("min_synapse_num", 3) or 3),
        }
        if morph_enabled.value:
            cp["morph_auc_floor"] = float(morph_auc_floor.value)
        if visualize.value:
            cp["max_scenes"] = int(max_scenes.value)
            cp["neuron_alpha"] = float(neuron_alpha.value)
        if backward_enabled.value:
            cp["backward_top_n"] = int(backward_top_n.value)
            cp["backward_max_neurons"] = int(backward_max_neurons.value)
            cp["backward_per_branch_cap"] = int(backward_per_branch_cap.value)
            cp["backward_scan_pool_targets"] = bool(backward_scan_pool_targets.value)
        for field, (widget, kind, is_float) in adv.items():
            if kind == "bool":
                cp[field] = bool(widget.value)
            else:
                cp[field] = float(widget.value) if is_float else int(widget.value)

        output_panel.clear()
        output_panel.set_running(True)
        result = await output_panel.run(runner, "type_mapping_validation", cp, "run",
                                        output_dir=output_dir.value)
        rc = result.get("returncode")
        run_dir = result.get("output_folder") or output_dir.value or ""

        # Post-run provenance banner. An unsupported pair / out-of-scope query is
        # NOT an error — it writes an empty run — so surface "no branches"
        # explicitly. set_coverage.json holds two ROLE-named blocks plus the
        # dataset labels:
        #   {"source_dataset", "target_dataset",
        #    "source": {total_queried, assigned, fill_proposed_only,
        #               unpaired_unproposed, per_type:{...}}, "target": {...}}
        try:
            if rc != 0:
                output_panel.clear_notice()
            else:
                n_branches = -1
                pair_path = os.path.join(run_dir, "validation", "pair_summary.csv")
                if os.path.isfile(pair_path):
                    try:
                        with open(pair_path, encoding="utf-8") as pf:
                            n_branches = sum(1 for _ in pf) - 1  # minus header
                    except Exception:
                        n_branches = -1
                bits = []
                cov_path = os.path.join(run_dir, "set_coverage.json")
                if os.path.isfile(cov_path):
                    with open(cov_path, encoding="utf-8") as fh:
                        cov = json.load(fh)
                    for ds, block in list(cov.items())[:4]:
                        if isinstance(block, dict):
                            total = block.get("total_queried")
                            assigned = block.get("assigned")
                            fill = block.get("fill_proposed_only")
                            unpaired = block.get("unpaired_unproposed")
                            if total is not None and assigned is not None:
                                bits.append(
                                    f"{ds}: {assigned}/{total} assigned"
                                    + (f", {fill} fill-only" if fill is not None else "")
                                    + (f", {unpaired} unpaired" if unpaired is not None else ""))
                        elif isinstance(block, (int, float)):
                            bits.append(f"{ds}={block}")
                if not bits and n_branches <= 0:
                    output_panel.set_notice(
                        "Completed — but NO type pairs / branches were resolved for "
                        "this scope, so the report is empty. The mapping covers none "
                        "of the queried types, or the source→target pair is not "
                        "supported. Try a type present in both datasets.")
                elif bits:
                    output_panel.set_notice("Set coverage — " + "; ".join(bits))
                else:
                    output_panel.set_notice(
                        f"Completed — {n_branches} branch(es) validated.")
        except Exception as exc:
            # This reader includes the ONLY signal for a successful-but-empty
            # run (unsupported dataset pair); a silent drop leaves a bare
            # "Completed". Keep the UI stable but leave a trace.
            logging.debug("set-coverage notice unavailable: %s", exc)

        # Cap-and-warn (§3.2): surface the backend's scenes cap as a
        # persistent notice (user decision 2026-09-19: cap properly and tell
        # the user to split the run).
        if rc == 0 and run_dir and os.path.isdir(run_dir):
            try:
                cap_line = _scene_cap_line(run_dir)
                if cap_line:
                    output_panel.set_notice(cap_line)
            except Exception as exc:
                logging.debug("scene-cap notice unavailable: %s", exc)

        # Record query chips into the shared history (single-dataset scope).
        if rc == 0:
            try:
                from ..history_store import record as _record_history
                try:
                    from ..type_suggestions import datasets_resolving
                    vd = datasets_resolving(queries, [source, target])
                except Exception:
                    vd = {}
                _record_history(queries, datasets=[source], value_datasets=vd)
            except Exception:
                pass

        output_panel.set_running(False)
        output_panel.set_status("Completed" if rc == 0 else "Failed",
                                "green" if rc == 0 else "red")
        output_panel.show_files(result.get("files", []), run_dir)

        # Post-run action row.
        action_row.clear()
        if rc == 0 and run_dir and os.path.isdir(run_dir):
            report = os.path.join(run_dir, "report.html")
            if os.path.isfile(report):
                ui.button("Open report", icon="insert_chart_outlined",
                          on_click=lambda p=report: open_file(p)).props("outline dense")
            ui.button("Open run folder", icon="folder_open",
                      on_click=lambda d=run_dir: open_folder(d)).props("outline dense")
            scenes = sorted(glob.glob(os.path.join(
                run_dir, "visualization", "plot-3d_*", "branches_*.html")),
                key=os.path.getmtime) if os.path.isdir(
                os.path.join(run_dir, "visualization")) else []
            if scenes:
                ui.button("Open latest scene", icon="view_in_ar",
                          on_click=lambda s=scenes[-1]: open_file(s)).props("outline dense")

    output_panel.run_button.on_click(run_validation)
    output_panel.cancel_button.on_click(runner.cancel)
