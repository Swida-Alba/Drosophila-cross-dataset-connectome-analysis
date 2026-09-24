"""Type Validation (TM VEV) tab: nav, runner-bridge, config-parity, settings.

These guard the pieces that are easy to silently break: the banner-literal
progress bridge, the field-name payload (dataclass parity), and the two new
Settings keys landing in both DEFAULTS and DEFAULT_SETTING_SPECS.
"""

import contextlib
import dataclasses
import io
import re
import sys
import types
from pathlib import Path

import pytest


# --------------------------------------------------------------------------
# Nav / exports
# --------------------------------------------------------------------------
def test_type_validation_tab_registered_in_nav():
    import ui.app as app
    assert "Paths" in app._DROCAT_TAB_NAMES
    assert "Type Validation" in app._DROCAT_TAB_NAMES
    assert "Cross-Dataset" not in app._DROCAT_TAB_NAMES
    # legacy cookie migrates to the new label
    assert app._TAB_NAME_ALIASES.get("Cross-Dataset") == "Paths"


def test_type_validation_exported_from_tabs_package():
    from ui.tabs import create_type_validation_tab
    assert callable(create_type_validation_tab)


# --------------------------------------------------------------------------
# Dataclass parity (anti-drift guard)
# --------------------------------------------------------------------------
def test_tab_defaults_match_live_dataclass():
    from ui.tabs import type_validation as tv
    from comparison.mapping_validation import MappingValidationConfig as C
    live = {f.name: f.default for f in dataclasses.fields(C)
            if f.default is not dataclasses.MISSING}
    for field, fallback in tv._FALLBACK_DEFAULTS.items():
        assert field in live, f"{field} is no longer a config field"
        assert live[field] == fallback, (
            f"{field}: tab default {fallback!r} drifted from dataclass "
            f"{live[field]!r} — update the fallback or wire the widget")


def test_tab_advanced_fields_are_real_fields():
    from ui.tabs import type_validation as tv
    from comparison.mapping_validation import MappingValidationConfig as C
    names = {f.name for f in dataclasses.fields(C)}
    for field in (*tv.ADV_NUM, *tv.ADV_BOOL):
        assert field in names, f"{field} is not a MappingValidationConfig field"


def test_validation_mode_options_match_backend():
    from comparison.mapping_validation import (MODE_RANK, POOLING_MODE,
                                               VALIDATION_MODES)
    from ui.tabs import type_validation as tv
    # `pooling` is the one dropdown value that is NOT in the nested enum: it
    # is a parallel mode, so `MODE_RANK` must not learn it (a rank would make
    # every mode_at_least comparison admit it) while the tab still has to be
    # able to start it.
    assert list(VALIDATION_MODES) + [POOLING_MODE] == tv.MODE_OPTIONS
    assert POOLING_MODE not in MODE_RANK
    assert set(tv.MODE_HINTS) == set(tv.MODE_OPTIONS)


def test_pooling_widgets_back_real_fields():
    """Every pooling knob the tab renders is a dataclass field.

    The parity test above already pins the DEFAULTS; this pins the set, so a
    widget cannot be left behind when a knob is renamed.
    """
    from ui.tabs import type_validation as tv
    assert {"pooling_jaccard_floor", "pooling_rank_union_floor",
            "pooling_window_mult", "pooling_bar_metric",
            "pooling_bar_top_n",
            "pooling_max_morph_targets"} <= set(tv._FALLBACK_DEFAULTS)
    # the fit and the sibling join were DELETED (2026-09-23), not hidden: a
    # widget left behind would send a field the dataclass no longer has.
    assert not {k for k in tv._FALLBACK_DEFAULTS
                if 'floor_from_evidence' in k or 'sibling' in k}


# --------------------------------------------------------------------------
# Generated runner script
# --------------------------------------------------------------------------
def _generate(cp):
    from ui.runner import ScriptRunner, TOOL_REGISTRY
    assert "type_mapping_validation" in TOOL_REGISTRY
    return ScriptRunner()._generate_script(
        "type_mapping_validation", cp, "run", None)


def test_generated_script_passes_the_pooling_gate():
    cp = {"source_dataset": "A", "target_dataset": "B", "query_types": ["t"],
          "validation_mode": "pooling", "pooling_jaccard_floor": 0.07,
          "pooling_window_mult": 3.0, "pooling_bar_metric": "jaccard",
          "pooling_bar_top_n": 5,
          "pooling_max_morph_targets": 50}
    s = _generate(cp)
    compile(s, "<gen>", "exec")
    for frag in ("validation_mode='pooling'", "pooling_jaccard_floor=0.07",
                 "pooling_window_mult=3.0", "pooling_bar_metric='jaccard'",
                 "pooling_bar_top_n=5",
                 "pooling_max_morph_targets=50"):
        assert frag in s, frag
    assert 'floor_from_evidence' not in s


def test_generated_script_compiles_and_is_wired():
    cp = {"source_dataset": "A", "target_dataset": "B", "query_types": ["t"],
          "morph_enabled": True, "visualize": True, "verbose": True,
          "run_label": None}
    s = _generate(cp)
    compile(s, "<gen>", "exec")
    assert "MappingValidationConfig(" in s
    assert "Output will be saved to:" in s
    assert "[DROCAT][progress]" in s
    assert "run_label=None" not in s  # None keys are pruned


@pytest.mark.parametrize("flags,expected_total", [
    (dict(morph_enabled=True, visualize=True, backward_evidence_enabled=True,
          skip_out_map_expansion=False), 9),
    (dict(morph_enabled=True, visualize=False, backward_evidence_enabled=True,
          skip_out_map_expansion=False), 8),
    (dict(morph_enabled=False, visualize=False, backward_evidence_enabled=False,
          skip_out_map_expansion=True), 5),
])
def test_generated_step_total_tracks_flags(flags, expected_total):
    cp = {"source_dataset": "A", "target_dataset": "B", "query_types": ["t"],
          **flags}
    m = re.search(r"_total = (\d+)", _generate(cp))
    assert int(m.group(1)) == expected_total


def test_scan_dir_resolves_run_folder(tmp_path):
    from ui.runner import ScriptRunner
    run_dir = tmp_path / "type-map-validation_A_to_B_20200101_000000"
    run_dir.mkdir()
    sr = ScriptRunner()
    sr._run_logs = [("log", f"[DROCAT] Output will be saved to: {run_dir}")]
    assert sr._resolve_scan_dir(str(tmp_path)) == str(run_dir)


# --------------------------------------------------------------------------
# Progress bridge — execute the real generated script against a stub backend
# --------------------------------------------------------------------------
def _run_generated_with_stub(cp, banners):
    """Install a stub comparison.mapping_validation, exec the generated script,
    return captured stdout."""
    script = _generate(cp)

    class _Cfg:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            self.verbose = kw.get("verbose", True)

    class _Validator:
        def __init__(self, cfg):
            self.cfg = cfg
            self.notes = []
            self.run_dir = None

        def log(self, msg=''):
            if self.cfg.verbose:
                print(msg, flush=True)
            self.notes.append(str(msg))

        def run(self):
            self.run_dir = "/tmp/type-map-validation_A_to_B_20200101_000000"
            for b in banners:
                self.log(b)
            self.log(f"done in 3s -> {self.run_dir}")
            return self.run_dir

    stub = types.ModuleType("comparison.mapping_validation")
    stub.MappingValidationConfig = _Cfg
    stub.MappingValidator = _Validator
    saved = sys.modules.get("comparison.mapping_validation")
    sys.modules["comparison.mapping_validation"] = stub
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(script, "<gen>", "exec"), {"__name__": "__gen__"})
    finally:
        if saved is not None:
            sys.modules["comparison.mapping_validation"] = saved
        else:
            del sys.modules["comparison.mapping_validation"]
    return buf.getvalue()


PROGRESS_RE = re.compile(r"^\[DROCAT\]\[progress\] (\d+)/(\d+) (.+)$")


def _full_run_banners():
    return [
        "[stage 1] resolving type pairs A -> B",
        "[stage 2-lite] building target vectors for the suspects pass",  # must NOT match step 2
        "[stage 2] building target expanded-type vectors (B)",
        "[stage 2] source type 1/2: t — scanning",
        "[stage 5] Track-A null bar: p95 = 0.4",                        # must NOT advance step 4
        "[stage 5] morphology verification + self-calibration",
        "[categories] mode=family: matched=3, candidates=1",
        "[categories] failed: ignore me",                               # must NOT advance step 5 twice
        "[stage 5d] backward homolog evidence: 5 distinct member(s)",
        "[TMVEV] out-map expansion: no unclaimed sources",              # no-work twin: must NOT advance
        "[TMVEV] out-map expansion: 25/60 sources scanned",
        "[stage 4] scenes in A template (fafb)",
        "[set coverage] failed: boom",                                  # failure-only: no advance
        "[gap fill levels] failed: boom",
        "[TMVEV] report written: /x/report.html",
    ]


def test_progress_bridge_emits_advancing_steps_and_callthrough():
    cp = {"source_dataset": "A", "target_dataset": "B", "query_types": ["t"],
          "morph_enabled": True, "visualize": True, "backward_evidence_enabled": True,
          "skip_out_map_expansion": False, "verbose": True}
    out = _run_generated_with_stub(cp, _full_run_banners())
    steps = [int(m.group(1)) for m in
             (PROGRESS_RE.match(l) for l in out.splitlines()) if m]
    totals = {int(m.group(2)) for m in
              (PROGRESS_RE.match(l) for l in out.splitlines()) if m}
    assert steps == sorted(steps) and len(set(steps)) == len(steps), steps
    assert max(steps) == 9 and totals == {9}
    assert out.count("Output will be saved to:") == 1
    assert out.index("Output will be saved to:") < out.index("[DROCAT] Done.")
    # call-through: the raw banners still reached stdout (notes preserved)
    assert "[stage 5] Track-A null bar" in out


def test_progress_bridge_collapses_when_scenes_off():
    cp = {"source_dataset": "A", "target_dataset": "B", "query_types": ["t"],
          "morph_enabled": True, "visualize": False, "backward_evidence_enabled": True,
          "skip_out_map_expansion": False, "verbose": True}
    banners = [b for b in _full_run_banners() if "[stage 4]" not in b]
    out = _run_generated_with_stub(cp, banners)
    steps = [int(m.group(1)) for m in
             (PROGRESS_RE.match(l) for l in out.splitlines()) if m]
    assert max(steps) == 8  # scenes step removed


# --------------------------------------------------------------------------
# Tab build + settings round-trip
# --------------------------------------------------------------------------
def test_type_validation_tab_mounts():
    from nicegui import Client
    from nicegui.page import page
    from ui.tabs import create_type_validation_tab
    client = Client(page("/tmvev-build"))
    with client:
        create_type_validation_tab()
    ids = {getattr(e, "_props", {}).get("id") for e in client.elements.values()}
    assert "card-tmvev-datasets" in ids
    assert "card-tmvev-advanced" in ids
    assert any(getattr(e, "_props", {}).get("label") == "Run Validation"
               for e in client.elements.values())


def test_tmvev_settings_round_trip():
    from ui import config
    assert config.DEFAULTS["tmvev_mode"] == "restrictive"
    assert "restrictive" in config.DEFAULT_SETTING_SPECS["tmvev_mode"]["options"]
    spec = config.DEFAULT_SETTING_SPECS["tmvev_max_scenes"]
    assert spec["kind"] == "int"
    # 0 is the "render every parent" setting on the backend, so the UI has to
    # be able to express it: DEFAULTS matching the dataclass (the live check is
    # test_tab_defaults_match_live_dataclass) and a min of 0.
    assert config.DEFAULTS["tmvev_max_scenes"] == 0
    assert spec["min"] == 0
    config.set_user_default("tmvev_mode", "family")
    assert config.get_user_default("tmvev_mode") == "family"
    config.set_user_default("tmvev_max_scenes", 20)
    assert config.get_user_default("tmvev_max_scenes") == 20
    # A stored 0 must survive too, or the override would silently become a cap.
    config.set_user_default("tmvev_max_scenes", 0)
    assert config.get_user_default("tmvev_max_scenes") == 0
    assert config.has_user_default("tmvev_max_scenes") is True
    config.reset_user_defaults()


# --------------------------------------------------------------------------
# Target default, named checklist, cap-and-warn
# --------------------------------------------------------------------------
def test_target_dataset_defaults_to_fafb():
    """Target default is the literal FAFB dataset — deliberately independent
    of the Settings default_target_dataset key (user 2026-09-22), which stays
    owned by Connectivity → Find Similar."""
    from nicegui import Client
    from nicegui.page import page
    from ui.tabs import create_type_validation_tab
    client = Client(page("/tmvev-fafb-default"))
    with client:
        create_type_validation_tab()
    target = [e for e in client.elements.values()
              if type(e).__name__ == "Select"
              and (e._props or {}).get("label") == "Target Dataset"]
    assert len(target) == 1
    assert target[0].value == "flywire_FAFB_v783"


def test_pooling_mounts_a_button_and_a_hidden_gate_card():
    """The tab has to be able to START `pooling` without presenting it as a
    wider rung: four mode buttons, and a gate card that stays hidden until the
    mode is `pooling` (a knob panel visible on a restrictive run would read as
    if the floors gated that run)."""
    from nicegui import Client
    from nicegui.page import page
    from ui.tabs import create_type_validation_tab
    client = Client(page("/tmvev-pooling-mount"))
    with client:
        create_type_validation_tab()
    props = [getattr(e, "_props", {}) or {} for e in client.elements.values()]
    assert "card-tmvev-pooling" in {p.get("id") for p in props}
    # inputs carry their caption as a `label` prop, checkboxes as `.text`
    labels = ({p.get("label") for p in props}
              | {getattr(e, "text", None) for e in client.elements.values()})
    for want in ("Bar depth (top-N per metric)",
                 "Jaccard floor (advisory flag)",
                 "Window multiplier (advisory flag)",
                 "Morph budget (scoring units)"):
        assert want in labels, want
    # the fitted-floor checkbox and the corroboration note were deleted with
    # the features themselves — and so is the morphology checkbox: the gate is
    # MANDATORY in pooling (2026-09-24), so no control offers to turn it off,
    # and no control sends a field the dataclass no longer has.
    assert not [w for w in labels if w and (
        'Morphology as the last gate' in w or 'Fit the floor' in w
        or 'corrobor' in w.lower())]
    assert {"Restrictive", "Family", "Aggressive", "Pooling"} <= {
        p.get("label") for p in props if p.get("label") in
        {"Restrictive", "Family", "Aggressive", "Pooling"}}
    card = [e for e in client.elements.values()
            if (getattr(e, "_props", {}) or {}).get("id")
            == "card-tmvev-pooling"][0]
    assert card.visible is False          # default mode is restrictive


def test_progress_steps_for_names_the_checklist():
    """The results-panel checklist mirrors the generated bridge's step table:
    same labels, same flag-driven collapsing, same totals."""
    from ui.components.page_progress import (
        progress_steps_for, tmvev_progress_steps)
    full = {"source_dataset": "A", "target_dataset": "B",
            "query_types": ["t"], "morph_enabled": True, "visualize": True,
            "backward_evidence_enabled": True, "skip_out_map_expansion": False}
    named = progress_steps_for("type_mapping_validation", context=full)
    assert named[0] == "Resolve type mapping branches"
    assert named[-1] == "Write exports and the run report"
    assert len(named) == 9
    assert len(progress_steps_for(
        "type_mapping_validation", context={**full, "visualize": False})) == 8
    assert len(progress_steps_for(
        "type_mapping_validation",
        context={**full, "morph_enabled": False})) == 8
    assert len(progress_steps_for(
        "type_mapping_validation",
        context={**full, "backward_evidence_enabled": False})) == 8
    assert len(progress_steps_for(
        "type_mapping_validation",
        context={**full, "skip_out_map_expansion": True})) == 8
    # default context: backward evidence off → 8
    assert len(progress_steps_for("type_mapping_validation")) == 8
    # `pooling` adds the ONE stage the mode owns, because for a pooling run
    # that is the long one — and it collapses away in every other mode.
    pool_named = progress_steps_for(
        "type_mapping_validation",
        context={**full, "validation_mode": "pooling"})
    assert len(pool_named) == len(named) + 1
    assert "Scan the unsupervised pool" in pool_named
    assert "Scan the unsupervised pool" not in named
    # single source of truth: every banner literal the checklist is built
    # from is embedded in the generated script's bridge table
    script = _generate(full)
    for literal, _label in tmvev_progress_steps(full):
        assert literal in script
    pool_script = _generate({**full, "validation_mode": "pooling"})
    for literal, _label in tmvev_progress_steps(
            {**full, "validation_mode": "pooling"}):
        assert literal in pool_script


def test_scene_cap_line_read_from_readme(tmp_path):
    from ui.tabs.type_validation import _scene_cap_line
    cap = ("[stage 4] scene cap: rendering 12 of 27 candidate branch scenes; "
           "raise Max Scenes or split the query into separate runs to render "
           "the rest")
    run_dir = tmp_path / "type-map-validation_A_to_B_20200101_000000"
    run_dir.mkdir()
    assert _scene_cap_line(str(run_dir)) is None  # no README yet
    (run_dir / "README.txt").write_text(
        "DROCAT run\nRun log:\n[stage 1] resolving type pairs A -> B\n"
        + cap + "\ndone in 3s\n", encoding="utf-8")
    assert _scene_cap_line(str(run_dir)) == cap
