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
    src = (ROOT / rel).read_text()
    if need_control:
        assert "corridor_mode = select_input(" in src
    assert '"corridor_mode": corridor_mode.value' in src
    return src


def test_find_path_tab_wired():
    src = _tab_wired("ui/tabs/find_path.py")
    # §7 live feedback: the depth-change handler refreshes the corridor
    # status line (Label.set_text, not set_value — house rule)
    assert "corridor_note" in src and ".set_text(_corridor_note_text())" in src
    assert "_refresh_corridor_note()" in src  # depth + mode both refresh


def test_find_shortest_tab_wired():
    src = _tab_wired("ui/tabs/find_shortest.py")
    assert "shortest-DAG" in src  # honest scope label


def test_inter_dataset_tab_wired():
    src = _tab_wired("ui/tabs/inter_dataset.py")
    assert 'get_user_default("corridor_mode")' in src


def test_mode_control_unified_naming():
    """2026-10-11: the Fast Mode control carries canonical values
    complete/auto/both with human labels, a hover hint, and a help_doc
    link into the algorithms guide — in all three tabs."""
    for rel in ("ui/tabs/find_path.py", "ui/tabs/find_shortest.py",
                "ui/tabs/inter_dataset.py"):
        src = (ROOT / rel).read_text()
        assert '"complete": "Complete only"' in src, rel
        assert '"auto": "Auto (deep runs)"' in src, rel
        assert '"both": "Complete + Fast"' in src, rel
        assert 'help_doc="pathfinding_algorithms"' in src, rel


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
    body = hook[:hook.index("def ", 10)]
    assert body.index("corridor_mode") < body.index("import subprocess")
    assert "_run_folder_is_replay_slice" in body  # §6 step 8 replay gate
    assert "except Exception as exc" in body  # contained failure (§3b tier 3)
    assert "TimeoutExpired" in body  # hard wall-clock budget
    assert "CorridorDeepDive.py" in body  # subprocess driver


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
    type_incoming_applied.csv + refilled pair map from the refill PAIR
    table (not the connection mirror); absence paths stay clean."""
    import pandas as pd
    inst = _make_cache_only_fnc()
    dd = tmp_path / "data_details"
    dd.mkdir()
    pd.DataFrame([
        {"type": "B", "incoming": 200.0},
        {"type": "C", "incoming": 100.0},
    ]).to_csv(dd / "type_incoming_applied.csv", index=False)
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
