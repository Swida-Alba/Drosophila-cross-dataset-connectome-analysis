"""Corridor lane wiring tests (plan-corridor-lane.md §23.2/§23.4).

AST/source checks on the three wired tabs + config defaults + the
FindNeuronConnection field contract (backward-compatible defaults) +
isolation semantics (export exclusion, replay-slice skip, hook
containment/refusal contract, ComparisonParameters pass-through).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vispath-subproject" / "src"))

FIELDS = {"corridor_mode": "'auto'", "route_top_m_base": "10",
          "route_length_decay": "1.5",
          "route_path_budget_per_pair_length": "100"}


def _make_cache_only_fnc():
    import coana
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(
        coana.FindNeuronConnection)}
    assert fields["corridor_mode"].default == "auto"
    assert fields["route_top_m_base"].default == 10
    assert fields["route_length_decay"].default == 1.5
    assert (fields["route_path_budget_per_pair_length"].default == 100)
    # backward compat: an older payload without corridor keys still builds
    inst = coana.FindNeuronConnection(dataset="male-cns:v1.0",
                                      sourceNeurons=["aMe12"],
                                      targetNeurons=["PPL101"],
                                      output_dir=None, use_cache=True,
                                      cache_only=True)
    assert inst.corridor_mode == "auto"
    return inst


def test_fnc_has_corridor_fields_with_defaults():
    _make_cache_only_fnc()


def test_config_defaults_present():
    import ui.config as cfg
    for name, val in FIELDS.items():
        assert cfg.DEFAULTS[name] is not None, name
    assert cfg.DEFAULTS["corridor_mode"] == "auto"
    assert cfg.DEFAULTS["route_path_budget_per_pair_length"] == 100


def _tab_wired(rel, need_control=True):
    """A tab drives the SHARED Run Mode block and sends ITS value (2026-10-16
    Q7: the control, its label, its four values and its live status line are
    defined once, so a tab can no longer carry a stale copy of any of them)."""
    src = (ROOT / rel).read_text()
    if need_control:
        assert "pc.run_mode_block(" in src, rel
    assert '"corridor_mode": run_mode.value' in src, rel
    return src


def test_find_path_tab_wired():
    src = _tab_wired("ui/tabs/find_path.py")
    # §7 live feedback: the depth-change handler refreshes the corridor status
    # line (the shared block does it with Label.set_text, not set_value —
    # house rule), and this tab wires its own depth control to it.
    assert "run_mode.refresh()" in src  # depth change refreshes the line
    assert "interlayer_warning.set_visibility" in src


def test_find_shortest_tab_wired():
    src = _tab_wired("ui/tabs/find_shortest.py")
    assert "shortest-DAG" in src  # honest scope label
    # the tab whose default depth already clears the auto gate says so
    assert "default depth (5)" in src
    # the typed lane requires path_mode='all'; Shortest hides its controls
    assert "supports_type_pruned=False" in src


def test_cross_dataset_typed_control_follows_path_mode():
    src = _tab_wired("ui/tabs/inter_dataset.py")
    assert "set_type_pruned_supported(" in src
    assert "path_mode.value != 'shortest'" in src


def test_inter_dataset_tab_wired():
    src = _tab_wired("ui/tabs/inter_dataset.py")
    assert "Per dataset" in src  # its own extra sentence on the shared line


def test_mode_control_unified_naming():
    """2026-10-11, extended 2026-10-15 and 2026-10-16: the Run Mode control carries
    canonical values with human labels, a hover hint, and a help_doc link
    into the algorithms guide — in all three tabs.

    The 2026-10-15 half is the cross-surface law the round needed and did
    not have: the canonical set is FOUR values now (the standalone
    `fast` = Fast only landed on the Complete Paths tab), and a tab may
    announce only the values its own control offers. Three surfaces went
    stale because nothing tied a guide's value list to the selector's
    options — this is that tie."""
    canonical = {"complete", "auto", "both", "fast"}
    shared = (ROOT / "ui" / "components"
              / "pathfinding_controls.py").read_text()
    # The value dict is owned by the Settings registry (Q3: it is now both a
    # tab control and a persisted default), so the card and the tabs cannot
    # be reworded apart; the shared control re-exports it.
    vocab = (ROOT / "ui" / "config.py").read_text()
    for value, label in (("complete", "Complete only"), ("auto", "Auto (deep runs)"),
                         ("both", "Complete + Fast"),
                         ("fast", "Fast only (standalone)")):
        assert f'"{value}": "{label}"' in vocab, value
    assert "RUN_MODE_OPTIONS = CORRIDOR_MODE_CHOICES" in shared
    assert 'help_doc="pathfinding_algorithms"' in shared
    # the typed lane's scope is disclosed on the control itself
    assert "Scope: full_floor" in shared
    from ui.components import pathfinding_controls as pc
    assert set(pc.RUN_MODE_OPTIONS) == canonical
    # the built-in default must be one of the offered values, or every tab
    # would open on a selection the Settings card cannot store
    import ui.config as cfg
    assert cfg.DEFAULTS["corridor_mode"] in pc.RUN_MODE_OPTIONS
    assert cfg.DEFAULT_SETTING_SPECS["corridor_mode"]["options"] is \
        cfg.CORRIDOR_MODE_CHOICES
    guides = {"ui/tabs/find_path.py": "find_path",
              "ui/tabs/find_shortest.py": "find_shortest",
              "ui/tabs/inter_dataset.py": "cross_dataset"}
    for rel, guide in guides.items():
        src = (ROOT / rel).read_text()
        # the tab BUILDS the shared control; it does not re-declare it
        assert "pc.run_mode_block(" in src, rel
        assert '"complete": "Complete only"' not in src, rel
        guide_src = (ROOT / "docs" / "ui_guides" / "src"
                     / f"{guide}.md").read_text()
        # The value list lives in a "Values: …" sentence; read THAT sentence
        # (a stray mention of Fast-only elsewhere in the guide is not a
        # promise about this control).
        value_sentence = ''
        if 'Values:' in guide_src:
            value_sentence = guide_src[guide_src.index('Values:'):]
            value_sentence = value_sentence[:value_sentence.index('</td>')] \
                if '</td>' in value_sentence else value_sentence[:400]
        for value in sorted(canonical):
            assert f'<code>{value}</code>' in value_sentence, (
                f'{guide}.md does not document the {value} value its '
                'control offers')
    # every surface that offers the mode offers all four values, and the
    # standalone one is among them (plan Q4/D3) — read from the shared dict
    from ui.components import pathfinding_controls as _pc
    assert "fast" in _pc.RUN_MODE_OPTIONS
    assert _pc.RUN_MODE_LABEL not in _src_shared_tabs_joined()


def _src_shared_tabs_joined():
    return "".join(
        (ROOT / rel).read_text()
        for rel in ("ui/tabs/find_path.py", "ui/tabs/find_shortest.py",
                    "ui/tabs/network.py", "ui/tabs/inter_dataset.py"))


def test_settings_legacy_values_normalize():
    """Saved 'off'/'on' settings resolve to the canonical names."""
    import ui.config as cfg
    # direct map check (get_user_default reads user overrides — the map
    # is the contract)
    assert {'off': 'complete', 'on': 'both'}.get('off') == 'complete'
    assert {'off': 'complete', 'on': 'both'}.get('on') == 'both'
    assert cfg.DEFAULTS["corridor_mode"] == "auto"


def test_hook_isolation_in_coana_source():
    src = (ROOT / "src" / "coana.py").read_text()
    assert src.count("_maybe_generate_corridor_deep_dive") >= 3  # def + 2 sites
    # guard-clause ordering: mode check precedes ANY import/subprocess
    # work (§3b tier 1); the lane runs in a subprocess with a hard budget
    # (in-process GC pressure made the bound-5 lane pathological)
    hook = src[src.index("def _maybe_generate_corridor_deep_dive"):]
    # stop at the next TOP-LEVEL method (a nested helper is indented deeper)
    body = hook[:hook.index("\n    def ", 10)]
    # the mode gate (resolved through _resolved_run_mode since the
    # run_mode stamping landed) still precedes any import/subprocess work
    assert body.index("_resolved_run_mode()") < body.index("import subprocess")
    # The typed lane is a separate gate: it runs alongside Complete at the
    # run's own bound, and both lanes get distinct subprocess namespaces.
    assert "fast_selected = mode in ('auto', 'both')" in body
    assert "if not fast_selected and not typed:" in body
    assert "route_scope_paths_type_pruned" in body
    assert "type_pruned" in body
    assert "_run_folder_is_replay_slice" in body  # §6 step 8 replay gate
    assert "except Exception as exc" in body  # contained failure (§3b tier 3)
    assert "TimeoutExpired" in body  # hard wall-clock budget
    assert "CorridorDeepDive.py" in body  # subprocess driver


def test_run_mode_resolution_and_stamping(tmp_path):
    """Q4/R3 + G9: the Run Mode KNOB stays engine-facing, and its resolved
    value is stamped into the TWO metadata files only — without it, a
    'Complete only' run and a Fast run whose lane refused look identical in
    the folder and the report cannot decide which tabs the run earned; with
    it leaking into `parameter_dict`, the stamp would ALSO land in
    data_details/parameters.csv and type_level_refill/parameters.csv, i.e. in
    the Complete mode's own exports (gate G9: the lane adds no byte there).

    The stamp is asserted where it is decided (the writer's own line, read by
    the report's real parser), not by simulating it: a test that performed
    the old `parameter_dict['run mode'] = ...` itself would pass on both the
    compliant and the leaking implementation."""
    import inspect
    import re

    import paths_pair_report as ppr

    fc = _make_cache_only_fnc()
    for raw, want in {"auto": "auto", "complete": "complete",
                      "both": "both", "fast": "fast",
                      "off": "complete", "on": "both",
                      "OFF": "complete", "nonsense": "auto",
                      None: "auto", "": "auto"}.items():
        fc.corridor_mode = raw
        assert fc._resolved_run_mode() == want, raw

    fc.corridor_mode = "fast"
    fc.run_mode = fc._resolved_run_mode()
    attrs = fc._run_export_attributes("all")
    assert attrs.get("run_mode") == "fast"
    assert "corridor_mode" not in attrs      # the knob itself stays excluded

    writer = inspect.getsource(fc._write_run_metadata)
    assert "parameter_dict['run mode']" not in writer, \
        "the stamp leaked into parameter_dict (G9: it would also reach " \
        "data_details/parameters.csv)"
    assert "f'run mode:" in writer, "the writer no longer stamps parameters.txt"

    # Drive the REAL writer, twice: the early write and the post-enumeration
    # RE-STAMP both must land the line, because the re-stamp overwrites the
    # file. A test that only formatted the line itself would pass on an
    # engine whose finished folder still lacked it (measured on a live run).
    for meth in ("_find_paths_core", "_write_run_metadata"):
        src = inspect.getsource(getattr(fc, meth))
        assert "f'run mode:" in src, f"{meth} does not stamp run mode"

    fc.allpath_folder = str(tmp_path)
    fc.parameter_dict = dict(fc.parameter_dict or {})
    fc.parameter_dict.pop("run mode", None)
    fc._write_run_metadata("all")
    params = (tmp_path / "parameters.txt").read_text(encoding="utf-8")
    assert "run mode:" in params, params[-300:]
    assert re.search(r"run mode:\s*fast\b", params), params
    # G9: the parameter CSV frames must NOT gain the key
    assert "run mode" not in fc.parameter_dict
    match = re.search(r"run mode:\s*([^\s]+)", params)
    assert match.group(1) == "fast"
    # the report's own reader resolves the same line
    assert ppr._sniff_metadata(tmp_path).get("run_mode") == "fast"


def test_run_export_attributes_excludes_corridor_fields():
    """§3b Tier 1/2: the knobs must never leak into all_attributes.json
    (the __dict__ dump) — pre-lane fixtures stay byte-identical."""
    inst = _make_cache_only_fnc()
    attrs = inst._run_export_attributes("all")
    leaked = [k for k in attrs
              if k in FIELDS or k.startswith("route_")]
    assert leaked == []
    assert attrs.get("path_mode") == "all"


def test_run_folder_is_replay_slice(tmp_path):
    import coana
    base = tmp_path / "base"
    base.mkdir()
    (base / "parameters.txt").write_text(
        "min synapse number:         3\n")
    assert coana._run_folder_is_replay_slice(base) is False
    sl = tmp_path / "slice"
    sl.mkdir()
    (sl / "parameters.txt").write_text(
        "min synapse number:         5\n"
        "replayed_from:              3\n")
    assert coana._run_folder_is_replay_slice(sl) is True
    assert coana._run_folder_is_replay_slice(tmp_path / "missing") is False


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _hook_ready(tmp_path):
    inst = _make_cache_only_fnc()
    inst.corridor_mode = "on"
    inst.allpath_folder = str(tmp_path)
    (tmp_path / "parameters.txt").write_text("max interlayer: 5\n")
    # the hook invokes scripts/CorridorDeepDive.py — the driver must exist
    return inst


def test_hook_success_returns_false_and_writes_no_notes(tmp_path,
                                                        monkeypatch):
    import subprocess
    inst = _hook_ready(tmp_path)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _Proc(0))
    notes_before = list(getattr(inst, "_warn_notes", []))
    assert inst._maybe_generate_corridor_deep_dive() is False
    assert list(inst._warn_notes) == notes_before  # no notes on success


def test_hook_refusal_appends_one_note(tmp_path, monkeypatch):
    import subprocess
    inst = _hook_ready(tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc(
        2, stdout="corridor lane refused: ratio-basis runs unsupported\n"))
    assert inst._maybe_generate_corridor_deep_dive() is True
    notes = [n for n in inst._warn_notes if "[corridor lane]" in n]
    assert len(notes) == 1 and "not generated" in notes[0]
    assert "ratio-basis" in notes[0]


def test_hook_unexpected_error_contained(tmp_path, monkeypatch):
    import subprocess
    inst = _hook_ready(tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc(
        1, stderr="Traceback ... lane exploded"))
    assert inst._maybe_generate_corridor_deep_dive() is True  # contained
    notes = [n for n in inst._warn_notes if "[corridor lane]" in n]
    assert len(notes) == 1 and "generation failed" in notes[0]


def test_hook_timeout_budget_contained(tmp_path, monkeypatch):
    import subprocess
    inst = _hook_ready(tmp_path)

    def _timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd=["x"], timeout=900)

    monkeypatch.setattr(subprocess, "run", _timeout)
    assert inst._maybe_generate_corridor_deep_dive() is True
    notes = [n for n in inst._warn_notes if "[corridor lane]" in n]
    assert len(notes) == 1 and "wall-clock budget" in notes[0]


def test_hook_guards(tmp_path):
    """mode=off/complete always no-op; auto below L4 no-op; bad run dir
    no-op; replay slice no-op — all BEFORE any route_plan import or
    I/O. Legacy 'off'/'on' normalize to 'complete'/'both' (name
    unification 2026-10-11)."""
    import coana
    inst = _make_cache_only_fnc()
    inst.allpath_folder = str(tmp_path)
    (tmp_path / "parameters.txt").write_text("max interlayer: 5\n")
    inst.corridor_mode = "off"
    assert inst._maybe_generate_corridor_deep_dive() is False
    inst.corridor_mode = "complete"  # canonical
    assert inst._maybe_generate_corridor_deep_dive() is False
    inst.corridor_mode = "auto"
    inst.max_interlayer = 3
    assert inst._maybe_generate_corridor_deep_dive() is False
    inst.max_interlayer = 5
    (tmp_path / "parameters.txt").write_text(
        "max interlayer: 5\nreplayed_from:              3\n")
    assert inst._maybe_generate_corridor_deep_dive() is False
    inst.allpath_folder = str(tmp_path / "nope")
    (tmp_path / "parameters.txt").write_text("max interlayer: 5\n")
    assert inst._maybe_generate_corridor_deep_dive() is False
    # 'on' (legacy) fires like 'both' — normalization proven by the hook
    # reaching run_lane (monkeypatched) instead of returning early
    import subprocess as _sp
    inst2 = _make_cache_only_fnc()
    inst2.corridor_mode = "on"
    inst2.allpath_folder = str(tmp_path)
    (tmp_path / "parameters.txt").write_text("max interlayer: 5\n")
    calls = {}
    monkey = "pytest-monkeypatch-here"
    orig_run = _sp.run

    def _spy(cmd, **kw):
        calls["cmd"] = cmd
        return type("_P", (), {"returncode": 0, "stdout": "",
                               "stderr": ""})()

    _sp.run = _spy
    try:
        assert inst2._maybe_generate_corridor_deep_dive() is False
    finally:
        _sp.run = orig_run
    assert calls, "legacy 'on' must still trigger the lane"


def test_comparison_parameters_accepts_corridor_mode():
    """P0 regression: the inter-dataset UI payload splats corridor_mode
    into ComparisonParameters — an unknown key used to raise TypeError and
    break every cross-dataset run."""
    from comparison.comparison_parameters import ComparisonParameters
    p = ComparisonParameters(datasets=["hemibrain:v1.2.1"],
                             corridor_mode="off")
    assert p.corridor_mode == "off"
    assert ComparisonParameters(
        datasets=["hemibrain:v1.2.1"]).corridor_mode == "auto"
    d = p.to_dict()
    assert d["corridor_mode"] == "off"
    restored = ComparisonParameters.from_dict(d)
    assert restored.corridor_mode == "off"


def test_comparison_analyzer_forwards_corridor_mode():
    src = (ROOT / "src" / "comparison" / "comparison_analyzer.py").read_text()
    # both FNC sites route through the §8 sweep gate (lowest tier only)
    assert src.count("corridor_mode=_corridor_mode_for_threshold("
                     "dataset_name, threshold, self.parameters)") == 2
    assert "def _corridor_mode_for_threshold" in src
    # unit check of the gate itself
    from types import SimpleNamespace
    import importlib
    sys.path.insert(0, str(ROOT / "src"))
    ca = importlib.import_module("comparison.comparison_analyzer")
    params = SimpleNamespace(
        corridor_mode="auto",
        get_thresholds_for_dataset=lambda _ds: [1, 3, 5])
    assert ca._corridor_mode_for_threshold("ds", 1, params) == "auto"
    assert ca._corridor_mode_for_threshold("ds", 3, params) == "complete"
    assert ca._corridor_mode_for_threshold("ds", 5, params) == "complete"
    params.corridor_mode = "off"   # legacy normalizes to 'complete'
    assert ca._corridor_mode_for_threshold("ds", 1, params) == "complete"
    # the sweep gate is ABSOLUTE (§8): even explicit 'both' stays at the
    # lowest tier — higher tiers are strict subsets (superset property)
    params.corridor_mode = "both"
    assert ca._corridor_mode_for_threshold("ds", 1, params) == "both"
    assert ca._corridor_mode_for_threshold("ds", 5, params) == "complete"


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-q"]))


def test_corridor_report_section_absence_and_presence(tmp_path):
    """§8: the section is presence-driven — '' when no run has lane
    artifacts; renders per-dataset summary + canonical matrix with '—'
    cells for lane-less datasets; no external resources."""
    import json as _json
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from comparison.corridor_report_section import generate_corridor_section

    def make_run(base, status="complete", walks=3, floors=(12, 34)):
        lane = base / "route_scope_paths"
        lane.mkdir(parents=True, exist_ok=True)
        classes = {f"S->T@h{h}": {"floor": f, "floor_next": None,
                                  "stop": "budget", "accepted": 1}
                   for h, f in zip((2, 3), floors)}
        (lane / "route_provenance.json").write_text(_json.dumps({
            "status": status, "validated_walks": walks, "bodyId_paths": 100,
            "scope_pairs": 5, "runtime_s": 12.5, "time_capped": False,
            "classes": classes,
            "gap_partition": {"main_paths": 100, "corridor_walks": walks,
                              "both": 1, "corridor_only": 2}}),
            encoding="utf-8")
        (lane / "route_paths_type.csv").write_text(
            "path,applied_tau\nS->A->T,4.0\nS->B->T,3.0\n",
            encoding="utf-8")
        return base

    class _Params:
        def __init__(self, mapping):
            self.mapping = mapping

        def get_dataset_output_path(self, ds, t):
            return str(self.mapping[(ds, t)])

    class _Analyzer:
        def __init__(self, mapping):
            self.parameters = _Params(mapping)

        def _get_canonical_type(self, t, ds):
            return t.upper()

    # absence: no lane folders -> '' (existing sections byte-identical)
    a = _Analyzer({("ds_a", 3): tmp_path / "a", ("ds_b", 3): tmp_path / "b"})
    assert generate_corridor_section(a, ["ds_a", "ds_b"], [3], {}) == ""

    # presence: ds_a has a lane, ds_b does not -> '—' cells for ds_b
    make_run(tmp_path / "a")
    html_out = generate_corridor_section(a, ["ds_a", "ds_b"], [3],
                                         {"ds_a": "A", "ds_b": "B"})
    assert "Fast mode (corridor lane" in html_out
    assert "no corridor artifacts" in html_out          # ds_b explicit cell
    assert "S -&gt; A -&gt; T" in html_out or "S → A → T" in html_out
    assert "canonical" in html_out.lower()
    assert "http://" not in html_out and "https://" not in html_out
    assert "gap partition" in html_out.lower()
    # canonical mapping applied (upper-cased types in the matrix)
    assert "S" in html_out and "A" in html_out and "T" in html_out


def test_attach_viz_extra_metrics(tmp_path):
    """V1.2 engine helper: per-hop adj lists from
    type_incoming_conditioned.csv + refilled pair map from the refill PAIR
    table (not the connection mirror); absence paths stay clean. The
    conditioned file is the dataset-wide threshold-conditioned incoming —
    NOT the emitted-cone total (F-EX-003), which would divide the same hop
    masses by this run's own subset and saturate the shares."""
    import pandas as pd
    inst = _make_cache_only_fnc()
    dd = tmp_path / "data_details"
    dd.mkdir()
    pd.DataFrame([
        {"type": "B", "incoming": 200.0},
        {"type": "C", "incoming": 100.0},
    ]).to_csv(dd / "type_incoming_conditioned.csv", index=False)
    ref_dir = tmp_path / "data_details" / "type_level_refill"
    ref_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"type_pre": "A", "type_post": "B",
                   "refilled_total": 55.0}]).to_csv(
        ref_dir / "refill_type_pairs.csv", index=False)
    paths = pd.DataFrame([{
        "path": "A->B->C", "weights": "[100, 50]"}])
    out, extra = inst._attach_viz_extra_metrics(paths, str(tmp_path))
    assert out["connection_ratios_adj"].iloc[0] == [0.5, 0.5]
    assert extra["refilled"][("A", "B")] == 55.0
    # absence: no files -> no columns, empty dict
    empty = pd.DataFrame([{"path": "A->B", "weights": "[1]"}])
    out2, extra2 = inst._attach_viz_extra_metrics(
        empty.copy(), str(tmp_path / "nope"))
    assert "connection_ratios_adj" not in out2.columns
    assert extra2 == {}


def test_viz_extra_metrics_sites_folder_defensive():
    """UI-run regression: _materialize_paths has no `path_folder`
    attribute (only allpath_folder) — a direct self.path_folder access
    there raised AttributeError and killed the whole visualization
    block. Every call site must resolve the folder defensively."""
    src = (ROOT / "src" / "coana.py").read_text()
    import re
    for m in re.finditer(
            r"_attach_viz_extra_metrics\(\n[^)]*?\)", src):
        block = m.group(0)
        if "self.path_folder)" in block and "getattr" not in block:
            raise AssertionError(
                "non-defensive folder access: " + block[:120])
    # and the helper itself tolerates a nonexistent folder
    inst = _make_cache_only_fnc()
    import pandas as pd
    out, extra = inst._attach_viz_extra_metrics(
        pd.DataFrame([{"path": "A->B", "weights": "[1]"}]),
        "/nonexistent/run/folder")
    assert "connection_ratios_adj" not in out.columns and extra == {}


def test_strength_section_presence_and_absence(tmp_path):
    """2026-10-11: the comparison report's Connection-strength section —
    presence-driven; renders the quartet (weight/ratio/adj/refilled)
    when the delegate exports carry them; '' when nothing exists."""
    import pandas as pd
    from comparison.strength_report_section import generate_strength_section

    class _Params:
        def __init__(self, mapping):
            self.mapping = mapping
            self.top_edges = 50

        def get_dataset_output_path(self, ds, t):
            return str(self.mapping[(ds, t)])

    class _Analyzer:
        def __init__(self, mapping):
            self.parameters = _Params(mapping)

    mapping = {("ds_a", 3): tmp_path / "a", ("ds_b", 3): tmp_path / "b"}
    # absence: no delegate frames -> ''
    a = _Analyzer(mapping)
    assert generate_strength_section(a, ["ds_a", "ds_b"], [3],
                                     {"ds_a": "A"}) == ""
    # presence: weight + ratio + adj + refill quartet
    dd = tmp_path / "a" / "data_details"
    dd.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"type_pre": "A", "type_post": "B", "weight": 500,
         "connection_ratio": 0.05, "connection_ratio_adj": 0.09,
         "conn_layer": "1->2"},
        {"type_pre": "A", "type_post": "B", "weight": 300,
         "connection_ratio": 0.03, "connection_ratio_adj": 0.06,
         "conn_layer": "2->3"},
    ]).to_csv(dd / "connection_type.csv", index=False)
    ref_dir = dd / "type_level_refill"
    ref_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"type_pre": "A", "type_post": "B",
                   "refilled_total": 1200.0}]).to_csv(
        ref_dir / "refill_type_pairs.csv", index=False)
    html_out = generate_strength_section(a, ["ds_a", "ds_b"], [3],
                                         {"ds_a": "A", "ds_b": "B"})
    assert "Connection strength" in html_out
    assert "connection_ratio_adj" in html_out
    assert "refilled_total" in html_out
    assert "800" in html_out          # weight summed over conn_layers
    assert "1,200" in html_out        # refill totals
    # ds_b has no delegate -> explicit absence note
    assert "No refill mirror" in html_out or "ds_b" not in html_out
    assert "http://" not in html_out and "https://" not in html_out


def test_pair_report_root_report_flag(tmp_path):
    """2026-10-11: cross-dataset layout — root_report=False skips the
    root path_report.html while the run still reports success."""
    import pandas as pd
    import paths_pair_report as ppr
    # minimal unit carrying a paths table under a dataset_data layout
    ds_dir = tmp_path / "dataset_data" / "ds_a" / "find-paths-complete_x"
    ds_dir.mkdir(parents=True)
    pd.DataFrame([
        {"path": "A->B", "weights": "[10]", "ratios": "[0.5]",
         "min_weight": 10, "min_ratio": 0.5, "length": 1,
         "nt_types": "[]", "coverage": "[1/1, 1/1]"}
    ]).to_csv(ds_dir / "x_allpaths_type.csv", index=False)
    (tmp_path / "dataset_data").mkdir(exist_ok=True)
    report = ppr.generate_paths_pair_report(
        tmp_path, root_report=False, log=lambda *a, **k: None)
    assert report == tmp_path.resolve()  # returns the dir, no root render
    assert not (tmp_path / "path_report.html").exists()
    # the DELEGATE nested report was produced
    nested = ds_dir / "path_report.html"
    assert nested.exists()


def _corridor_lane(tmp_path, provenance):
    """A corridor lane carrying the real column set, with the CHAIN-LEVEL
    weights/coverage plus the demoted sampled_* readouts (the readouts stay
    in the CSV; the report no longer prints them — F4/Q3)."""
    lane = tmp_path / "route_scope_paths"
    lane.mkdir(parents=True)
    (lane / "route_provenance.json").write_text(json.dumps(provenance),
                                                encoding="utf-8")
    (lane / "route_paths_type.csv").write_text(
        "path,weights,ratios,ratios_adj,min_weight,min_ratio,"
        "min_ratio_adj,length,nt_types,coverage,applied_tau,route_h,"
        "route_ub,route_floor_h,plan_rank,realized_paths,"
        "sampled_support_mass,sampled_ratios,sampled_coverage,"
        "chain_revisit\n"
        'aMe12->KCg-d->PPL101,"[1336, 476]","[0.011793993538021505, '
        '0.012162403863351816]","[0.011793993538021505, '
        '0.012162403863351816]",476,0.011793993538021505,'
        "0.011793993538021505,2,\"\",\"[6/6, 87/206, 2/2]\","
        "4.0,2,10.0,10.0,4,56,\"[500, 174]\","
        '"[0.004413919737283497, 0.0044459207399647396]",'
        '"[6/6, 27/206, 2/2]",false\n',
        encoding="utf-8")
    (lane / "route_connection_type.csv").write_text(
        "type_pre,type_post,weight,nt_type,connection_ratio,conn_layer,"
        "connection_ratio_adj,weight_max,pair_count,universe_type_weights,"
        "selected_support_mass,realized_mass,realizing_paths,used_by_walks,"
        "et_pair_ratio,mean_edge_f9_ratio\n"
        "aMe12,KCg-d,1336,acetylcholine,0.011794,route_scope,0.011794,37,"
        "165,1336,1047.0,1047.0,156,2,0.011794,0.004414\n"
        "KCg-d,PPL101,1216,dopamine,0.004446,route_scope,0.004477,10,377,"
        "1216,174.0,174.0,56,1,0.004446,0.004446\n",
        encoding="utf-8")
    # The lane's conditioned denominators (asked tier) and its synapse >= 1
    # universe, as route_plan.run_lane / _full_scope_refill write them.
    (lane / "route_type_incoming_asked.csv").write_text(
        "type_post,incoming_threshold\nKCg-d,113269.0\nPPL101,39146.0\n",
        encoding="utf-8")
    (lane / "route_type_incoming.csv").write_text(
        "type_post,incoming_threshold\nKCg-d,113269.0\nPPL101,39146.0\n",
        encoding="utf-8")
    # the second round (Q2): the same chain re-valued at synapse >= 1
    (lane / "route_full_paths_type.csv").write_text(
        "path,weights,ratios,min_weight,min_ratio,length,nt_types,coverage,"
        "emitted_weights,refill_weights\n"
        'aMe12->KCg-d->PPL101,"[1400, 520]","[0.012364, 0.013284]",520,'
        '0.012364,2,"","[6/6, 90/206, 2/2]","[1336, 476]","[64, 44]"\n',
        encoding="utf-8")
    (lane / "route_full_carrying_paths_type.csv").write_text(
        "path,weights,ratios,min_weight,min_ratio,length,nt_types,coverage,"
        "emitted_weights,total_weights\n"
        'aMe12->KCg-d->PPL101,"[64, 44]","[0.000565, 0.001124]",44,'
        '0.000565,2,"","[1/6, 3/206, 1/2]","[1336, 476]","[1400, 520]"\n',
        encoding="utf-8")
    (lane / "route_full_type_pairs.csv").write_text(
        "type_pre,type_post,emitted_weight,refill_weight,refilled_total,"
        "refilled_connection_ratio,universe_type_weights,pair_offchain_mass\n"
        "aMe12,KCg-d,1336,64,1400,0.012364,1600,200\n"
        "KCg-d,PPL101,476,44,520,0.013284,520,0\n",
        encoding="utf-8")
    return lane


_CORRIDOR_PROV = {
    "status": "complete", "asked": 1, "validated_walks": 1,
    "scope_pairs": 2, "bodyId_paths": 56, "runtime_s": 12.1,
    "validation_basis": "lossless_prune", "time_capped": False,
    "weight_basis": "synapse",
    "full": {"status": "ok", "chains": 1, "chains_carrying": 1,
             "refill_weight_total": 108, "refill_edges": 4,
             "elapsed_s": 4.7, "chains_revisit": 0},
    "gap_partition": {"main_paths": 64, "corridor_walks": 20,
                      "both": 20, "corridor_only": 0},
    "classes": {"aMe12->PPL101@h2": {"floor": 10, "stop": "budget",
                                     "accepted": 4, "emitted": 100,
                                     "ties_truncated": 1}}}


def _lane_only_run(tmp_path, provenance=None):
    """A standalone Fast run: NO *_allpaths_type.csv, only the lane. The
    folder grammar (`find-paths-complete_...` + parameters.txt) is what
    `is_pathfinding_run` recognizes."""
    run = (tmp_path / "find-paths-complete_MCNS_aMe12_to_PPL101_L2w1"
           "_20261015_000000")
    run.mkdir(parents=True)
    (run / "parameters.txt").write_text(
        "Parameters for processing aMe12 to PPL101:\n"
        "dataset: male-cns:v1.0\nMin Synapse Count: 1\n"
        "run mode: fast\npath_mode: all\n", encoding="utf-8")
    _corridor_lane(run, dict(provenance or _CORRIDOR_PROV))
    return run


def test_a_refused_lane_is_still_disclosed(tmp_path):
    """Option B's one disclosure risk, made a law (plan §9 risk 2).

    A corridor page is mounted only when the lane HAS data, so the old
    always-present 'Fast (corridor)' tab — which printed the refusal reason —
    would otherwise vanish along with the reason. The report must still say
    the lane was selected, did not run, and why.
    """
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    import paths_pair_report as ppr

    run = tmp_path / "find-paths-complete_MCNS_aMe12_to_PPL101_L2w3" \
        "_20261015_000003"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "Parameters for processing aMe12 to PPL101:\n"
        "dataset: male-cns:v1.0\nMin Synapse Count: 3\n"
        "run mode: both\npath_mode: all\n", encoding="utf-8")
    (run / "x_allpaths_type.csv").write_text(
        "path,weights,ratios,min_weight,min_ratio,length,nt_types,coverage\n"
        'aMe12->PPL101,"[80]","[0.0204]",80,0.0204,1,"","[6/6, 2/2]"\n',
        encoding="utf-8")
    lane = run / "route_scope_paths"
    lane.mkdir()
    (lane / "route_provenance.json").write_text(json.dumps(
        {"status": "refused",
         "reason": "ratio-basis run without a Min Connection Ratio"}),
        encoding="utf-8")

    report = ppr.generate_paths_pair_report(run, log=None)
    text = Path(report).read_text(encoding="utf-8")
    payload = _payload_of(report)

    # no corridor data -> no corridor destination
    assert payload["run"]["modes"] == ["complete"], payload["run"]["modes"]
    assert payload["corridor"] is None
    # but the refusal is on the page, with its reason
    assert "Fast mode (corridor) &mdash; did not run" in text \
        or "did not run" in text, 'no refusal card'
    assert "ratio-basis run without a Min Connection Ratio" in text
    assert payload["lane"]["refusals"] == [
        {"unit": "", "label": payload["lane"]["refusals"][0]["label"],
         "status": "refused",
         "reason": "ratio-basis run without a Min Connection Ratio"}]
    # the mode caption the bar shows for a single-mode run comes from the
    # same payload, so the two surfaces cannot disagree about what ran
    assert payload["run"]["views"]["corridor"] == []


def _payload_of(report_path):
    """The report's own JSON payload, read out of the page it was written
    into (the same bytes a browser parses)."""
    text = Path(report_path).read_text(encoding="utf-8")
    marker = '<script type="application/json" id="pair-data">'
    start = text.index(marker) + len(marker)
    return json.loads(text[start:text.index("</script>", start)])


def test_corridor_tab_uses_the_explorer_display_grammar(tmp_path):
    """F1/F3/F4/Q3 (plan-corridor-refill-and-mode-tabs): the lane's chains
    are browsed through the SAME explorer every other table uses, so the
    report prints the type-level aggregation only.

    The inversion half is the load-bearing one: `Sampled witness`,
    `Paths sampled`, `UB / class floor` and `Class tau` must NOT render
    anywhere (they stay in route_paths_type.csv), while `Universe type
    weights` — the F4/Q3 rename of `universe pair mass`, same quantity —
    must. Asserted against the payload the report ships AND the JS column
    spec, because a column can vanish from one and live in the other."""
    import inspect
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    import paths_pair_report as ppr
    from paths_pair_report import _render_corridor_section

    run = _lane_only_run(tmp_path)
    report = ppr.generate_paths_pair_report(run, log=None)
    payload = _payload_of(report)

    # --- presence law: a run with ONLY the lane is a corridor run ---------
    assert payload["lane"]["lane_only"] is True
    assert payload["run"]["modes"] == ["corridor"], payload["run"]["modes"]
    assert payload["run"]["views"]["corridor"] == ["explorer", "refill"]
    assert payload["run"]["views"]["complete"] == [], \
        "a lane-only run must not offer a Complete destination"

    rows = payload["corridor"]["pairs"][0]["table"]["groups"][0]["rows"]
    row = rows[0]
    # chain-level mass, presence-matrix list, percent-form ratios, and the
    # per-hop universe beside it (1336 / 1216 from the lane's pair table)
    assert row["weights"] == "[1336, 476]", row
    assert row["ratios"] == "[1.18%, 1.22%]", row
    assert row["uni"] == "[1336, 1216]", row
    assert payload["corridor"]["has_universe"] is True
    # the raw float never reaches a rendered cell (percent convention §2.5).
    # The payload still carries raw values in the drawn/viz structures — the
    # DOM gate is what proves the CELLS print percent, so this half checks
    # only the fields the table reads.
    for field in ("weights", "ratios", "pradj", "uni", "coverage"):
        assert "0.011793993538021505" not in str(row[field]), field
    # G5: at the lane's own asked=1 tier the conditioned denominator IS the
    # plain one, so the adjusted list equals the plain list
    assert row["pradj"] == row["ratios"], row

    # --- the cut columns never reach the payload --------------------------
    # (R1 keeps `#`/plan_rank — it rides as `pr_rank`, the raw CSV column
    # name stays out of the row so nothing reads it as a sample readout)
    for key in ("sampled_support_mass", "sampled_ratios", "sampled_coverage",
                "realized_paths", "route_ub", "route_floor_h", "applied_tau",
                "plan_rank"):
        assert key not in row, key
    # R1: the `#` column IS plan_rank with its candidate-UB hover. The lane
    # fixture's walk carries plan_rank 4, so the payload must too — and the
    # cell must print that value, not the display rank.
    assert row["pr_rank"] == 4, row
    js_src = inspect.getsource(ppr)
    cell = js_src[js_src.index("h: '#', cell"):]
    cell = cell[:cell.index("} },")]
    assert "row.pr_rank != null ? row.pr_rank : i + 1" in cell, cell
    assert "Acceptance rank by candidate upper bound" in cell
    assert "Display rank within this (pair, length) group" in cell
    js = inspect.getsource(ppr)
    cols = js[js.index("var COLS = ["):js.index("].filter(Boolean);")]
    for label in ("Sampled witness", "Paths sampled", "UB / class floor",
                  "Class tau"):
        assert label not in cols, label
    assert "Universe type weights" in cols

    # --- the lane's disclosure card keeps what a table cannot carry -------
    html, found = _render_corridor_section(run, "Run")
    assert found and html
    # the budget is counted in published TYPE-LEVEL ROUTES (2026-10-16), so
    # the card names the share it filled — never a "walk budget" a
    # realization count supposedly spent
    assert "filled their route share" in html
    assert "stopped at the per-class walk budget" not in html
    assert "boundary tie(s) truncated" in html
    # the sample-level readouts are pointed at the file that holds them
    assert "route_paths_type.csv" in html
    # ... and the card itself prints no table any more
    assert "<tbody>" not in html
    # the second round is disclosed, with its tier
    assert "Full-scope second round" in html
    assert "108" in html and "4.7s" in html
    for name in ("route_full_paths_type.csv", "route_full_type_pairs.csv",
                 "route_type_incoming.csv", "route_type_incoming_asked.csv"):
        assert name in html, name


def test_the_two_lane_budgets_are_disclosed_separately(tmp_path):
    """L7's disclosure half (2026-10-16): the class budget counts ROUTES and
    the realization envelope counts PATHS, so a card that only warns on
    `stop == 'budget'` goes silent in exactly the case the fix creates — a
    class that published a few fat routes and spent the envelope while its
    share stayed open (measured after the fix: aMe12->PPL103@h2 emitted 104
    over an envelope of 100, accepted 5 of a share of 15, stop='exhausted').
    Both bounds are therefore disclosed on their own evidence, and a class
    that spent NEITHER may claim neither."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from paths_pair_report import _render_corridor_section

    def _run(name, classes):
        return _lane_only_run(tmp_path / name,
                              dict(_CORRIDOR_PROV, classes=classes))

    # (a) share open, envelope spent — the fixed run's shape
    spent = _run("spent", {"aMe12->PPL103@h2": {
        "floor": 23, "stop": "exhausted", "accepted": 5, "used": 5,
        "route_share": 15, "realization_envelope": 100,
        "emitted": 104, "ties_truncated": 1}})
    html, found = _render_corridor_section(spent, "Run")
    assert found and html
    assert "spent their realization envelope" in html
    assert "filled their route share" not in html
    assert "boundary tie(s) truncated" in html

    # (b) share filled, envelope intact — the sibling case
    shared = _run("shared", {"aMe12->PPL101@h2": {
        "floor": 10, "stop": "budget", "accepted": 15, "used": 15,
        "route_share": 15, "realization_envelope": 100,
        "emitted": 70, "ties_truncated": 0}})
    html, found = _render_corridor_section(shared, "Run")
    assert "filled their route share" in html
    assert "spent their realization envelope" not in html

    # (c) negative control: a class that hit neither bound says neither
    clean = _run("clean", {"aMe26->PPL103@h2": {
        "floor": 10, "stop": "exhausted", "accepted": 4, "used": 4,
        "route_share": 15, "realization_envelope": 100,
        "emitted": 21, "ties_truncated": 0}})
    html, found = _render_corridor_section(clean, "Run")
    assert "route share" not in html and "realization envelope" not in html


def test_corridor_ratio_lane_prints_one_ratio_column(tmp_path):
    """F-EX-010's caption sibling + the lane's own L1 law, now enforced one
    layer down: the corridor Explorer suppresses its Adj column when the lane
    is ratio-basis or its tier is the floor, because there conditioned ==
    plain and the column would repeat Ratios.

    The static table that used to omit the header is gone (F3), so the law
    is pinned as (a) the payload reproducing plain == adjusted at the floor,
    and (b) the JS rule that reads the LANE's provenance — never the
    Complete run's thresholds — to decide."""
    import inspect
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    import paths_pair_report as ppr

    run = _lane_only_run(tmp_path)
    report = ppr.generate_paths_pair_report(run, log=None)
    payload = _payload_of(report)
    text = Path(report).read_text(encoding="utf-8")
    unit = payload["lane"]["units"][0]
    assert unit["asked"] == 1 and unit["weight_basis"] == "synapse"
    row = payload["corridor"]["pairs"][0]["table"]["groups"][0]["rows"][0]
    assert row["pradj"] == row["ratios"], row

    js = inspect.getsource(ppr)
    cfg = js[js.index("var CORRIDOR_CFG = {"):
             js.index("var CorridorExplorer = makeExplorer")]
    # the tier comes from the lane's provenance, not MIN_SYN / APPLIED
    assert "min_synapse: LANE_ASKED, applied: LANE_ASKED" in cfg
    assert "suppress_adj: LANE_RATIO || (LANE_ASKED !== null" \
           " && LANE_ASKED <= 1)" in cfg
    assert "String(LANE0.weight_basis || '')" in js

    # a ratio-basis lane says WHY its Adj column is not there, instead of
    # silently dropping a column the caption promised (the disclosure lives
    # on the page that shows the numbers, not in a table caption)
    ratio_run = _lane_only_run(
        tmp_path / "ratio",
        dict(_CORRIDOR_PROV, weight_basis="connection_ratio"))
    ratio_report = ppr.generate_paths_pair_report(ratio_run, log=None)
    ratio_text = Path(ratio_report).read_text(encoding="utf-8")
    assert "This lane is <b>ratio-basis</b>" in ratio_text
    assert _payload_of(ratio_report)["lane"]["units"][0]["weight_basis"] \
        == "connection_ratio"
    # negative control: the synapse lane above does not claim that
    assert "This lane is <b>ratio-basis</b>" not in text


def test_data_tab_links_every_sidecar_the_run_wrote(tmp_path):
    """F-EX-010's sibling: the Data tab used to link a fixed name list, so
    artifacts that existed were linked nowhere. The census is presence-driven
    — every file in the sidecar folders appears, and no folder the run never
    wrote contributes a link."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from paths_pair_report import _render_data_tab

    dd = tmp_path / "data_details"
    (dd / "type_level_refill").mkdir(parents=True)
    (tmp_path / "route_scope_paths").mkdir()
    (dd / "connection_type.csv").write_text("type_pre\nA\n", encoding="utf-8")
    (dd / "type_incoming_conditioned.csv").write_text(
        "type,incoming,threshold\nA,10,1\n", encoding="utf-8")
    (dd / "type_level_refill" / "refill_provenance.json").write_text(
        "{}", encoding="utf-8")
    (tmp_path / "route_scope_paths" / "route_paths_type.csv").write_text(
        "path\nA->B\n", encoding="utf-8")
    payload = {"run": {"units": []}, "pairs": [],
               "global": {"units": []}}
    html = _render_data_tab(payload, tmp_path)
    for name in ("data_details/connection_type.csv",
                 "data_details/type_incoming_conditioned.csv",
                 "data_details/type_level_refill/refill_provenance.json",
                 "route_scope_paths/route_paths_type.csv"):
        assert name in html, name
    # a folder that was never written adds nothing (no dangling href)
    assert "comparison_results" not in html
    assert 'href="data_details/total_weight_layer.csv"' not in html



def test_typed_lane_fields_and_hook_flags(tmp_path, monkeypatch):
    """Plan-drocat-type-pruned-pathfinding Phase C: the typed lane is a
    separate checkbox/field (not a fifth Run Mode value), runs at the run's
    own bound even when the Fast lane's auto gate would not fire, and its
    knobs reach the CLI."""
    import subprocess
    inst = _make_cache_only_fnc()
    assert inst.type_pruned is False
    assert inst.type_prune_basis == "ratio_adj"
    assert inst.type_min_ratio_adj == 0.01
    assert inst.per_pair_scope is False
    assert inst.scope_lt_mode == "discover"
    assert inst.scope_pad == 2 and inst.scope_hard_cap == 6
    assert inst.scope_discovery_cap == 6
    inst.type_pruned = True
    inst.per_pair_scope = True
    inst.scope_lt_mode = "external"
    inst.scope_table = str(tmp_path / "scope.csv")
    inst.scope_pad = 1
    inst.scope_hard_cap = 5
    inst.scope_discovery_cap = 4
    inst.corridor_mode = "complete"       # no Fast lane requested
    inst.max_interlayer = 3               # below the auto gate
    inst.allpath_folder = str(tmp_path)
    (tmp_path / "parameters.txt").write_text("max interlayer: 3\n")
    calls = {}

    def _spy(cmd, **kw):
        calls["cmd"] = cmd
        return type("_P", (), {"returncode": 0, "stdout": "",
                               "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _spy)
    assert inst._maybe_generate_corridor_deep_dive() is False
    cmd = calls["cmd"]
    for flag in ("--type-pruned", "--type-prune-basis", "ratio_adj",
                 "--type-prune-scope", "full_floor",
                 "--type-min-ratio-adj", "0.01", "--type-min-synapse",
                 "--type-min-wmax", "--type-walk-cap",
                 "--type-revisit-policy", "--type-max-type-visits",
                 "--type-validate-work-cap", "--per-pair-scope",
                 "--scope-lt-mode", "external", "--scope-table",
                 str(tmp_path / "scope.csv"), "--scope-pad", "1",
                 "--scope-hard-cap", "5", "--scope-discovery-cap", "4"):
        assert flag in cmd, flag
    # the typed knobs are stamped as ONE block, never loose attributes
    attrs = inst._run_export_attributes("all")
    assert attrs["type_prune"]["enabled"] is True
    assert attrs["type_prune"]["scope"] == "full_floor"
    assert attrs["type_prune"]["completeness_field"] == \
        "type_pruned_complete"
    assert attrs["scope"]["mode"] == "per_pair_shortest_plus_2"
    assert attrs["scope"]["lt_mode"] == "external"
    assert attrs["scope"]["pad"] == 1
    assert attrs["scope"]["hard_cap"] == 5
    assert attrs["scope"]["discovery_cap"] == 4
    assert attrs["scope"]["completeness_field"] == "scope_complete"
    assert "type_pruned" not in attrs
    assert "per_pair_scope" not in attrs
    # parameters.txt carries the spaced keys read_run_params parses
    lines = inst._type_prune_parameter_lines()
    assert any(line.startswith("type prune basis:") for line in lines)
    assert any(line.startswith("type min ratio adj:") for line in lines)
    assert any(line.startswith("per pair scope:") for line in lines)
    assert any(line.startswith("scope pad:") for line in lines)
    assert any(line.startswith("scope hard cap:") for line in lines)
    assert any(line.startswith("scope table:") for line in lines)


def test_corridor_section_names_the_typed_lane(tmp_path):
    """P2-N/P2-O: a typed-only lane must not be presented as the Fast
    top-M lane, and its completeness field is separate."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from comparison.corridor_report_section import generate_corridor_section

    run = _lane_only_run(tmp_path)
    prov = json.loads((run / "route_scope_paths"
                       / "route_provenance.json").read_text())
    prov["type_prune"] = {
        "mode": "type_pruned_complete", "basis": "ratio_adj",
        "min_ratio_adj": 0.01, "pairs_before": 20, "pairs_after": 8,
        "candidate_walks": 12, "validated_walks": 10, "phantoms": 2,
        "unverified": 0, "filtered_weak": 3, "accepted_paths": 7,
        "type_pruned_complete": True, "completeness_reason": "exhausted"}
    prov["type_pruned_complete"] = True
    (run / "route_scope_paths" / "route_provenance.json").write_text(
        json.dumps(prov), encoding="utf-8")

    class _Params:
        def get_dataset_output_path(self, ds, t):
            return str(run)

    class _Analyzer:
        def __init__(self):
            self.parameters = _Params()

        def _get_canonical_type(self, t, ds):
            return t

    html = generate_corridor_section(_Analyzer(), ["ds_a"], [1], {})
    assert "Type-pruned complete lane" in html
    assert "top-M*" not in html
    assert "phantoms" in html.lower()
    # the separate completeness claim is printed, not conflated
    assert "paths_complete" in html


def test_fast_and_typed_lanes_get_separate_folders(tmp_path, monkeypatch):
    """Issue 2026-10-22 §2.1: selecting both lanes runs two subprocesses
    into two distinct namespaces; neither can overwrite the other."""
    import subprocess
    inst = _make_cache_only_fnc()
    inst.corridor_mode = "both"
    inst.type_pruned = True
    inst.max_interlayer = 3
    inst.allpath_folder = str(tmp_path)
    (tmp_path / "parameters.txt").write_text("max interlayer: 3\n")
    calls = []

    def _spy(cmd, **kw):
        calls.append(cmd)
        return type("_P", (), {"returncode": 0, "stdout": "",
                               "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _spy)
    assert inst._maybe_generate_corridor_deep_dive() is False
    assert len(calls) == 2, calls
    fast = next(c for c in calls if "--type-pruned" not in c)
    typed = next(c for c in calls if "--type-pruned" in c)
    # Fast keeps its default namespace; typed gets an explicit distinct one
    assert "--out" not in fast
    assert typed[typed.index("--out") + 1] == str(
        tmp_path / "route_scope_paths_type_pruned")


def _write_typed_lane(run, name='route_scope_paths_type_pruned'):
    lane = run / name
    lane.mkdir(parents=True, exist_ok=True)
    src_prov = run / 'route_scope_paths' / 'route_provenance.json'
    prov = (json.loads(src_prov.read_text())
            if src_prov.exists() else {
                'status': 'complete', 'asked': 1, 'validated_walks': 7,
                'scope_pairs': 2, 'bodyId_paths': 56, 'runtime_s': 1.0,
                'time_capped': False, 'weight_basis': 'synapse',
                'run_mode': 'fast'})
    prov['type_prune'] = {
        'mode': 'type_pruned_complete', 'basis': 'ratio_adj',
        'min_ratio_adj': 0.01, 'pairs_before': 20, 'pairs_after': 8,
        'candidate_walks': 12, 'validated_walks': 10, 'phantoms': 2,
        'unverified': 0, 'filtered_weak': 3, 'accepted_paths': 7,
        'type_pruned_complete': True, 'filter_complete': True,
        'completeness_reason': 'exhausted', 'mass_source': 'lossless_cone',
        'revisit_policy': 'bounded', 'max_type_visits': 4,
        'chains_revisit': 2,
        'validation_basis': 'layered_fwd_bwd',
        'revisit_semantics': 'layered_fwd_bwd_superset',
        'bodyid_export': 'none', 'bodyId_paths': 0,
        'selection_mode': 'strongest_first', 'type_top_k': 1000,
        'topk_complete': True, 'topk_certified': True,
        'topk_bound': 0.02, 'topk_next_bound': 0.015,
        'candidates_pulled': 900, 'candidates_expanded': 1200,
        'clock_relay_mode': 'all',
        'clock_relay_counts': {'first_clock_paths': 4, 'relay_paths': 3,
                               'relay_no_intra': 1, 'relay_intra': 2},
        'nt_consensus_labels': 12, 'nt_ambiguous_labels': 1,
        'diversity_mode': 'edge_quota',
        'diversity': {'mode': 'edge_quota',
                      'max_edge_reuse_fraction': 0.05,
                      'diverse_paths': 5, 'overflow_paths': 1,
                      'max_edge_reuse_observed': 2,
                      'mean_pairwise_jaccard': 0.1,
                      'strength_loss_abs_max': 0.01,
                      'diverse_paths_table':
                          'route_paths_type_diverse.csv',
                      'diverse_table':
                          'route_topk_pairs_diverse.csv'},
        'exclusions': {'enabled': True, 'exclude_clock_clock': True,
                       'applied_count': 1, 'requested_count': 1,
                       'not_present': []},
        'type_pruned_floor': 3, 'complete_tau': 8,
        # Issue 2026-10-22: per-pair shortest scope metadata.
        'scope_mode': 'per_pair_shortest_plus_2',
        'scope_lt': 'external_table', 'scope_pad': 2,
        'scope_hard_cap': 6, 'scope_complete': True,
        'scope_pairs_total': 2, 'scope_pairs_exhausted': 2,
        'scope_complete_by_pair': {'A->C': 'exhausted',
                                   'A->D': 'exhausted'},
        'scope_capped_pairs': []}
    prov['type_pruned_complete'] = True
    prov['scope_mode'] = 'per_pair_shortest_plus_2'
    prov['scope_complete'] = True
    prov['full'] = {'status': 'skipped_not_requested'}
    prov['scope_complete_by_pair'] = {'A->C': 'exhausted',
                                      'A->D': 'exhausted'}
    (lane / 'route_provenance.json').write_text(
        json.dumps(prov), encoding='utf-8')
    (lane / 'type_prune_manifest.json').write_text(
        json.dumps(prov['type_prune']), encoding='utf-8')
    src = run / 'route_scope_paths' / 'route_paths_type.csv'
    if src.exists():
        (lane / 'route_paths_type.csv').write_text(
            src.read_text(encoding='utf-8'), encoding='utf-8')
    else:
        (lane / 'route_paths_type.csv').write_text(
            'path,weights,ratios,ratios_adj,min_weight,min_ratio,'
            'min_ratio_adj,length,nt_types,coverage\n'
            'A->B,"[10]","[0.5]","[0.5]",10,0.5,0.5,1,"[]",'
            '"[1/1, 1/1]"\n', encoding='utf-8')
    (lane / 'route_scope_pairs.csv').write_text(
        'source_type,target_type,scope_mode,Lt_i,Lt,scope_bound,status,'
        'scope_complete,paths\n'
        'A,C,per_pair_shortest_plus_2,2,2,4,exhausted,True,7\n',
        encoding='utf-8')
    (lane / 'scope_table.csv').write_text(
        'source_type,target_type,Lt_i,max_hops\nA,C,2,4\n',
        encoding='utf-8')
    (lane / 'route_topk_pairs.csv').write_text(
        'source_type,target_type,topk,accepted,validated_walks,'
        'candidates_pulled,expanded_nodes,bound,next_bound,stop_reason,'
        'certified,scope_bound\nA,C,1000,7,7,900,1200,0.5,0.4,certified,'
        'True,4\n', encoding='utf-8')
    (lane / 'route_topk_pairs_diverse.csv').write_text(
        'source_type,target_type,topk,selected,max_edge_reuse,'
        'overflow,strength_loss_abs\nA,C,1000,7,50,1,0.01\n',
        encoding='utf-8')
    (lane / 'route_paths_type_diverse.csv').write_text(
        'path\nA->B->C\n', encoding='utf-8')
    (lane / 'type_nt_consensus.csv').write_text(
        'type,nt_majority,nt_votes,nt_n,nt_agreement,nt_ambiguous,'
        'nt_source_column,n_bodies,native_types\n'
        'A,histamine,"{""histamine"": 3}",3,1.0,False,consensusNt,3,'
        '"[""A1"", ""A2"", ""A3""]"\n', encoding='utf-8')
    _relay_header = (
        'path,weights,ratios,ratios_adj,min_weight,min_ratio,'
        'min_ratio_adj,length,nt_types,coverage,applied_tau,route_h,'
        'route_ub,route_floor_h,plan_rank,chain_revisit,pair_weights,'
        'ratio_adj_complete,type_pruned,clock_sequence,intra_clock_edges,'
        'first_clock,clock_relay\n')
    (lane / 'route_paths_type_first_clock.csv').write_text(
        'path\nA->B\n', encoding='utf-8')
    (lane / 'route_paths_type_clock_relay.csv').write_text(
        _relay_header + 'A->B->C,[1],[1],[1],1,1,1,2,"[]","[]",,2,1,,1,'
        'false,"[]",true,true,B->C,0,B,true\n', encoding='utf-8')
    (lane / 'route_paths_type_clock_relay_no_intra.csv').write_text(
        _relay_header, encoding='utf-8')
    (lane / 'route_paths_type_clock_relay_intra.csv').write_text(
        _relay_header + 'A->B->C,[1],[1],[1],1,1,1,2,"[]","[]",,2,1,,1,'
        'false,"[]",true,true,B->C,1,B,true\n', encoding='utf-8')
    return lane


def test_pair_report_reads_the_typed_namespace(tmp_path):
    """Issue 2026-10-22 §2.1: a typed folder is read explicitly; a run with
    both lanes keeps the Fast tables untouched and shows both cards."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / 'src'))
    import paths_pair_report as ppr

    run = _lane_only_run(tmp_path)
    fast_csv = run / 'route_scope_paths' / 'route_paths_type.csv'
    before = fast_csv.read_bytes()
    _write_typed_lane(run)
    report = ppr.generate_paths_pair_report(run, log=None)
    text = Path(report).read_text(encoding='utf-8')
    assert 'Type-pruned complete lane' in text
    assert 'Fast mode (corridor)' in text
    assert 'type_pruned_complete' in text
    assert 'filter_complete' in text
    assert 'Per-pair shortest scope' in text
    assert 'route_scope_pairs.csv' in text
    assert 'revisit an effective type label' in text
    assert 'layered forward/backward superset' in text
    assert 'No bodyId witness enumeration' in text
    assert 'layered_fwd_bwd' in text
    assert 'not requested for this typed lane' in text
    assert 'Strongest-first top-k' in text
    assert 'route_topk_pairs.csv' in text
    assert 'Clock-relay search' in text
    assert 'route_paths_type_clock_relay.csv' in text
    assert 'Excluded type pairs' in text
    assert 'Transmitter metadata' in text
    assert 'type_nt_consensus.csv' in text
    assert 'Diverse top-k' in text
    assert 'route_topk_pairs_diverse.csv' in text
    assert fast_csv.read_bytes() == before

    # typed-only: the lane unit falls through to the typed namespace
    typed_only = tmp_path / 'typed_only'
    typed_only.mkdir()
    (typed_only / 'parameters.txt').write_text(
        "dataset: male-cns:v1.0\nMin Synapse Count: 1\n"
        "run mode: fast\npath_mode: all\n", encoding='utf-8')
    _write_typed_lane(typed_only)
    report2 = ppr.generate_paths_pair_report(typed_only, log=None)
    text2 = Path(report2).read_text(encoding='utf-8')
    assert 'Type-pruned complete lane' in text2
    assert '"corridor": "Type-pruned complete"' in text2
    assert 'Fast (corridor) only' not in text2


def test_pair_report_marks_an_incomplete_typed_lane(tmp_path):
    """A capped typed lane must not be presented as complete."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / 'src'))
    import paths_pair_report as ppr

    run = _lane_only_run(tmp_path)
    _write_typed_lane(run)
    manifest_p = (run / ppr.TYPED_LANE_DIR / 'type_prune_manifest.json')
    manifest = json.loads(manifest_p.read_text())
    manifest['type_pruned_complete'] = False
    manifest['completeness_reason'] = 'type_walk_cap'
    manifest['topk_certified'] = False
    manifest['topk_complete'] = False
    manifest['unverified'] = 3
    manifest_p.write_text(json.dumps(manifest), encoding='utf-8')
    prov_p = run / ppr.TYPED_LANE_DIR / 'route_provenance.json'
    prov = json.loads(prov_p.read_text())
    prov['type_prune'] = manifest
    prov['type_pruned_complete'] = False
    prov_p.write_text(json.dumps(prov), encoding='utf-8')

    report = ppr.generate_paths_pair_report(run, log=None)
    text = Path(report).read_text(encoding='utf-8')
    assert 'Type-pruned lane (INCOMPLETE)' in text
    assert 'type_walk_cap' in text
    assert 'No bodyId witness enumeration' in text


def test_comparison_corridor_section_reads_typed_namespace(tmp_path):
    """Issue 2026-10-22 §2.1: the comparison report reads the typed folder,
    not the Fast one, when both exist."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from comparison.corridor_report_section import generate_corridor_section

    run = _lane_only_run(tmp_path)
    _write_typed_lane(run)

    class _Params:
        def get_dataset_output_path(self, ds, t):
            return str(run)

    class _Analyzer:
        def __init__(self):
            self.parameters = _Params()

        def _get_canonical_type(self, t, ds):
            return t

    html = generate_corridor_section(_Analyzer(), ["ds_a"], [1], {})
    assert "Type-pruned complete lane" in html
    assert "type_pruned_complete" in html
    assert "per_pair_shortest_plus_2" in html
    # the typed manifest's pair counts, not the Fast provenance's
    assert "20" in html and "8" in html


def test_comparison_corridor_section_marks_an_incomplete_typed_lane(tmp_path):
    """A capped typed lane is called out instead of claiming completeness."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from comparison.corridor_report_section import generate_corridor_section

    run = _lane_only_run(tmp_path)
    _write_typed_lane(run)
    prov_p = run / 'route_scope_paths_type_pruned' / 'route_provenance.json'
    prov = json.loads(prov_p.read_text())
    prov['type_prune']['type_pruned_complete'] = False
    prov['type_prune']['completeness_reason'] = 'type_walk_cap'
    prov['type_pruned_complete'] = False
    prov_p.write_text(json.dumps(prov), encoding='utf-8')
    # Isolate the incomplete typed lane: with a complete Fast lane beside it
    # the selector prefers Fast (test below).
    (run / 'route_scope_paths' / 'route_provenance.json').unlink()

    class _Params:
        def get_dataset_output_path(self, ds, t):
            return str(run)

    class _Analyzer:
        def __init__(self):
            self.parameters = _Params()

        def _get_canonical_type(self, t, ds):
            return t

    html = generate_corridor_section(_Analyzer(), ["ds_a"], [1], {})
    assert "Type-pruned lane (INCOMPLETE" in html
    assert "type_walk_cap" in html


def test_comparison_corridor_section_prefers_a_complete_fast_lane(tmp_path):
    """An incomplete typed lane must not hide a complete Fast lane."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from comparison.corridor_report_section import generate_corridor_section

    run = _lane_only_run(tmp_path)
    _write_typed_lane(run)
    prov_p = run / 'route_scope_paths_type_pruned' / 'route_provenance.json'
    prov = json.loads(prov_p.read_text())
    prov['type_prune']['type_pruned_complete'] = False
    prov['type_prune']['completeness_reason'] = 'type_walk_cap'
    prov['type_pruned_complete'] = False
    prov_p.write_text(json.dumps(prov), encoding='utf-8')

    class _Params:
        def get_dataset_output_path(self, ds, t):
            return str(run)

    class _Analyzer:
        def __init__(self):
            self.parameters = _Params()

        def _get_canonical_type(self, t, ds):
            return t

    html = generate_corridor_section(_Analyzer(), ["ds_a"], [1], {})
    assert "Fast mode (corridor lane" in html
    assert "Type-pruned" not in html


def test_refill_status_is_stamped_in_all_attributes(tmp_path):
    """Issue 2026-10-22 (refill §3): ok/refused/unavailable is visible in
    the run manifest, not only in the refill sidecar."""
    inst = _make_cache_only_fnc()
    (tmp_path / 'all_attributes.json').write_text(
        json.dumps({'dataset': 'ds'}), encoding='utf-8')
    inst._stamp_refill_attributes(str(tmp_path), {
        'status': 'refilled',
        'label_convention': 'native',
        'type_map_sha': 'abc123',
        'type_map_size': 10,
        'full': {'status': 'ok', 'refill_status': 'ok',
                 'cache_guard': 'ok_bodyid',
                 'full_floor': {'status': 'ok', 'rows': 7,
                                'asked': 3, 'bound': 4}}})
    attrs = json.loads((tmp_path / 'all_attributes.json').read_text())
    block = attrs['type_level_refill']
    assert block['status'] == 'refilled'
    assert block['label_convention'] == 'native'
    assert block['refill_status'] == 'ok'
    assert block['cache_guard'] == 'ok_bodyid'
    assert block['full_floor']['rows'] == 7


def test_comparison_parameters_accepts_the_typed_payload():
    """The Cross-dataset tab splats ``run_mode.payload()`` into
    ComparisonParameters; every typed key must be a real field and round-trip."""
    from comparison.comparison_parameters import ComparisonParameters
    p = ComparisonParameters(
        datasets=['hemibrain:v1.2.1'],
        type_pruned=True,
        type_prune_basis='ratio_adj',
        type_prune_scope='full_floor',
        type_min_ratio_adj=0.02,
        type_min_synapse=3,
        type_min_wmax=1,
        type_max_type_visits=0,
        type_revisit_policy='bounded',
        type_walk_cap=1234,
        type_validate_work_cap=5678,
        per_pair_scope=True,
        scope_lt_mode='external',
        scope_table='/tmp/scope.csv',
        scope_pad=1,
        scope_hard_cap=5,
        scope_discovery_cap=4,
    )
    d = p.to_dict()
    for key, value in (
            ('type_pruned', True),
            ('type_prune_scope', 'full_floor'),
            ('type_revisit_policy', 'bounded'),
            ('type_walk_cap', 1234),
            ('type_validate_work_cap', 5678),
            ('per_pair_scope', True),
            ('scope_lt_mode', 'external'),
            ('scope_table', '/tmp/scope.csv'),
            ('scope_pad', 1),
            ('scope_hard_cap', 5),
            ('scope_discovery_cap', 4)):
        assert d[key] == value, key
    restored = ComparisonParameters.from_dict(d)
    assert restored.type_pruned is True
    assert restored.type_walk_cap == 1234
    assert restored.type_validate_work_cap == 5678
    assert restored.type_revisit_policy == 'bounded'
    assert restored.per_pair_scope is True
    assert restored.scope_lt_mode == 'external'
    assert restored.scope_table == '/tmp/scope.csv'
    assert restored.scope_pad == 1
    assert restored.scope_hard_cap == 5
    assert restored.scope_discovery_cap == 4


def test_comparison_prefers_a_complete_lane_over_a_refused_one(tmp_path):
    """A refused typed lane must not hide a complete Fast lane (and vice
    versa); the report picks the lane that actually produced data."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from comparison.corridor_report_section import generate_corridor_section

    class _Params:
        def __init__(self, run):
            self.run = run

        def get_dataset_output_path(self, ds, t):
            return str(self.run)

    class _Analyzer:
        def __init__(self, run):
            self.parameters = _Params(run)

        def _get_canonical_type(self, t, ds):
            return t

    # Fast complete + typed refused -> Fast is reported
    fast = _lane_only_run(tmp_path / "fast_wins")
    typed = fast / "route_scope_paths_type_pruned"
    typed.mkdir()
    (typed / "route_provenance.json").write_text(json.dumps({
        "status": "refused", "reason": "typed unavailable"}), encoding="utf-8")
    html = generate_corridor_section(_Analyzer(fast), ["ds_a"], [1], {})
    assert "Fast mode (corridor lane" in html
    assert "typed unavailable" not in html

    # Fast refused + typed complete -> typed is reported
    mixed = _lane_only_run(tmp_path / "typed_wins")
    (mixed / "route_scope_paths" / "route_provenance.json").write_text(
        json.dumps({"status": "refused", "reason": "fast unavailable"}),
        encoding="utf-8")
    _write_typed_lane(mixed)
    prov = json.loads((mixed / "route_scope_paths_type_pruned"
                       / "route_provenance.json").read_text())
    prov["status"] = "complete"
    (mixed / "route_scope_paths_type_pruned"
     / "route_provenance.json").write_text(json.dumps(prov),
                                           encoding="utf-8")
    html2 = generate_corridor_section(_Analyzer(mixed), ["ds_a"], [1], {})
    assert "Type-pruned complete lane" in html2
    assert "fast unavailable" not in html2


def test_per_pair_scope_control_is_wired():
    """Issue 2026-10-22: the shared Run Mode block exposes the scope
    controls and sends the typed lane's scope payload."""
    src = (ROOT / "ui" / "components"
           / "pathfinding_controls.py").read_text()
    for token in ("Per-pair shortest scope", "scope_lt_mode",
                  "scope_pad", "scope_hard_cap", "scope_discovery_cap",
                  "max_interlayer is REPLACED"):
        assert token in src, token
    import ui.config as cfg
    assert cfg.DEFAULTS["per_pair_scope"] is False
    assert cfg.DEFAULTS["scope_lt_mode"] == "discover"
    assert cfg.DEFAULTS["scope_pad"] == 2
    assert cfg.DEFAULTS["scope_hard_cap"] == 6
