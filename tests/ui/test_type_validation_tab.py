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
    from comparison.mapping_validation import VALIDATION_MODES
    from ui.tabs import type_validation as tv
    assert list(VALIDATION_MODES) == tv.MODE_OPTIONS


# --------------------------------------------------------------------------
# Generated runner script
# --------------------------------------------------------------------------
def _generate(cp):
    from ui.runner import ScriptRunner, TOOL_REGISTRY
    assert "type_mapping_validation" in TOOL_REGISTRY
    return ScriptRunner()._generate_script(
        "type_mapping_validation", cp, "run", None)


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
    # single source of truth: every banner literal the checklist is built
    # from is embedded in the generated script's bridge table
    script = _generate(full)
    for literal, _label in tmvev_progress_steps(full):
        assert literal in script


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
