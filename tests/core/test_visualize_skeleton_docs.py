"""Docs/skill drift tripwire for ``src/visualize_skeleton.py``.

Guards the surfaces that describe the skeleton viewer's newer behaviour
(frozen 3D framing, trace identity + profile granularity, HTML re-export, the
run manifest) against the module: every name and value asserted here is read
from ``inspect.signature`` / ``__doc__`` / the dataclass fields, so a rename or
a changed default has to be reflected in the prose or this test fails. The
re-export half is checked in both directions: every helper the prose
advertises must still exist in the module (and stay registered in the UI), and
the engine the re-export replaced must appear in neither.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import visualize_skeleton as viz  # noqa: E402

VISUALIZE_SKELETON = viz.VisualizeSkeleton

GUIDE = "docs/visualizations/3D_Skeleton_Guide.md"
OUTPUT_FILES = "docs/OUTPUT_FILES.md"
SKILL = "skills/drocat-backend/modules/visualize-skeleton.md"
UI_GUIDE = "docs/ui_guides/skeleton.html"

#: The surfaces that must describe the viewer's legend/framing API.
USAGE_SKILL = "skills/drocat-usage/tabs/visualization.md"
PROSE_FILES = (GUIDE, OUTPUT_FILES, SKILL, USAGE_SKILL)

#: Every Phase-8 prose surface, including the in-app HTML guide.
ALL_DOCS = PROSE_FILES + (UI_GUIDE,)

#: In-app code surfaces the re-export card is wired through.
RUNNER = "ui/runner.py"
REEXPORT_CARD = "ui/components/skeleton_reexport.py"
SKELETON_TAB = "ui/tabs/visualization.py"
UI_CONFIG = "ui/config.py"
SETTINGS_PANEL = "ui/components/skeleton_visualization_settings.py"

#: ``VisualizeSkeleton`` moved out of ``src/coana.py`` at the module split, so
#: this import raises. Nothing readers are sent to today may teach it.
RETIRED_IMPORT = "from coana import VisualizeSkeleton"

CURRENT_VISUALIZE_SKELETON_DOCS = (
    GUIDE, "src/visualize_skeleton.py", "docs/FAFB_INTEGRATION.md",
    "docs/FLYWIRE_USAGE.md", "docs/TROUBLESHOOTING.md",
)

#: Dated November-2024 records, kept as written: each either drops the retired
#: import or names itself a historical record and points at the live guide.
HISTORICAL_VISUALIZE_SKELETON_DOCS = (
    "docs/bugfixes/Brain_Template_Fix_Nov2024.md",
) + tuple(
    f"docs/visualizations/{name}" for name in (
        "VisualizeSkeleton_Implementation_Summary.md",
        "VisualizeSkeleton_Optimizations_Nov2024.md",
        "VisualizeSkeleton_Performance_Optimizations_Nov2024.md",
        "VisualizeSkeleton_Quick_Reference.md",
        "VisualizeSkeleton_Updates_Nov2024.md",
    ))

#: Dated records the parameter sweep must not touch.
DOC_SWEEP_SKIP = ("docs/bugfixes/", "docs/audits/", "archive/", "_plan/",
                  "docs/visualizations/VisualizeSkeleton_")

#: Files that quote a retired token deliberately, to explain what was retired.
LEGACY_VOCAB_NOTES = {
    GUIDE: ("whole", "template"),
    "docs/technical/SKELETON_DATA_PIPELINE.md": ("whole",),
    "docs/core-features/HomologFinding_Guide.md": ("whole", "template"),
}

LEGEND_MODES = ("single", "type", "tree", "layer")

MODULE_FUNCTIONS = ("figure_payload_from_html", "figure_from_plotly_html",
                    "classify_traces")

#: Module-level re-export API, every member of which Phase 8 documents.
REEXPORT_FUNCTIONS = MODULE_FUNCTIONS + (
    "resolve_viewer_page", "read_visualization_manifest", "reexport_output_dir",
    "profile_plan_from_html", "manifest_role_drift",
    "export_individuals_from_html",
    "export_video_from_html",
)

#: Manifest keys ``save_figure()`` writes next to the viewer HTML.
MANIFEST_NAME = "visualization_manifest.json"

#: Suffix of the folder everything re-exported lands in.
REEXPORT_SUFFIX = "_reexport"

#: UI tool the re-export card runs.
REEXPORT_TOOL = "plot3d_reexport"


def _read(rel):
    return (PROJECT_ROOT / rel).read_text(encoding="utf-8")


def _mentions(rel, needle):
    """Case-insensitive substring check, insensitive to prose wording."""
    return needle.lower() in _read(rel).lower()


def _dataclass_field(name):
    return {field.name: field for field in dataclasses.fields(VISUALIZE_SKELETON)}[name]


def _parameter_block(doc, name):
    """One numpydoc ``Parameters`` block, or ``''`` when it is absent."""
    match = re.search(
        rf"(?m)^(\s*){re.escape(name)} : .*?(?=^\1\w[\w_]* : |\Z)", doc, re.S)
    return match.group(0) if match else ''


def test_every_prose_surface_exists():
    for rel in PROSE_FILES:
        assert (PROJECT_ROOT / rel).is_file(), rel


def test_legend_modes_match_the_module_constant():
    assert tuple(viz.LEGEND_MODES) == LEGEND_MODES


def test_ui_legend_modes_mirror_the_module_constant():
    """One option list in the UI, and it is the renderer's set.

    Three panels offer ``legend_mode`` and each used to carry its own literal,
    so a mode added to the viewer could stay unselectable -- or be selectable in
    one panel and rejected by ``validate()``. Quoted mode words are fine in
    hint prose; a bracketed list of them is a second option list.
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import ui.config as ui_config

    assert set(ui_config.LEGEND_MODES) == set(viz.LEGEND_MODES)
    quoted = '|'.join(re.escape(f'"{mode}"') for mode in LEGEND_MODES)
    literals = re.compile(rf"\[[^\]]*(?:{quoted})[^\]]*(?:{quoted})[^\]]*\]")
    for rel in (UI_CONFIG, SKELETON_TAB, SETTINGS_PANEL):
        text = _read(rel)
        assert "LEGEND_MODES" in text, f"{rel}: does not use the shared list"
        if rel != UI_CONFIG:
            assert not literals.search(text), f"{rel} re-literals the option list"


#: Skeleton selects whose options must be a subset of what ``validate()`` takes.
#: Each pairs the Settings key with the module-level source of its legal values.
UI_SELECT_FIELDS = {
    "legend_mode": lambda viz: set(viz.LEGEND_MODES),
    "profile_granularity": lambda viz: set(viz.PROFILE_GRANULARITIES),
    "brain_mesh": lambda viz: set(viz.BRAIN_MESH_SELECTIONS),
    "skeleton_mode": lambda viz: _validated_choices("skeleton_mode"),
}


def test_ui_selects_offer_only_what_the_renderer_accepts():
    """A Settings option the renderer rejects is a click away from a crash.

    The option lists and the ``validate()`` literals live in different files, so
    only an assertion joins them: the same join the prose sweep makes between
    the docs and the code, one layer closer to the user.
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import ui.config as ui_config

    for key, legal in UI_SELECT_FIELDS.items():
        spec = ui_config.DEFAULT_SETTING_SPECS[key]
        assert spec["kind"] == "select", key
        accepted = legal(viz)
        assert accepted, key
        assert set(spec["options"]) == accepted, (
            f"{key}: UI and renderer disagree on "
            f"{sorted(set(spec['options']) ^ accepted)}")
        assert ui_config.DEFAULTS[key] in accepted, f"{key} default is rejected"


def test_legend_modes_documented_everywhere():
    for rel in PROSE_FILES:
        text = _read(rel)
        assert "legend_mode" in text, rel
        for mode in LEGEND_MODES:
            assert mode in text.lower(), f"{rel}: legend_mode={mode!r} undocumented"


def test_tree_mode_documented_as_the_interactive_panel():
    assert _mentions(GUIDE, "Interactive tree legend"), GUIDE
    assert _mentions(SKILL, "tree"), SKILL


def test_tree_hint_names_both_panel_shapes():
    """The two legend_mode hints must show the custom-group level too.

    The panel is type -> bodyId/instance normally and group -> type ->
    bodyId/instance with custom groups; a hint that promises only the first
    makes the second look like a bug rather than the same knob.
    """
    for rel in (UI_CONFIG, SKELETON_TAB, SETTINGS_PANEL):
        text = _read(rel)
        assert "bodyId/instance" in text, f"{rel}: hint says 'neuron'"
        assert "group -> type" in text, f"{rel}: custom group level hidden"


def test_current_docs_do_not_teach_the_retired_import():
    for rel in CURRENT_VISUALIZE_SKELETON_DOCS:
        assert RETIRED_IMPORT not in _read(rel), f"{rel} imports from coana"


def test_historical_records_of_the_retired_import_carry_a_banner():
    for rel in HISTORICAL_VISUALIZE_SKELETON_DOCS:
        text = _read(rel)
        if RETIRED_IMPORT in text:
            assert "Historical record" in text, f"{rel}: no banner"
            assert "3D_Skeleton_Guide.md" in text, f"{rel}: banner is a dead end"


#: Module whose ``validate()`` literals define most of the accepted choices.
SKELETON_MODULE = "src/visualize_skeleton.py"


def _doc_sweep_files():
    """Docs, skills and example scripts a reader may copy code from."""
    out = []
    for pattern in ("docs/**/*.md", "skills/**/*.md", "scripts/**/*.py"):
        for path in PROJECT_ROOT.glob(pattern):
            rel = str(path.relative_to(PROJECT_ROOT))
            if rel.startswith(DOC_SWEEP_SKIP):
                continue
            out.append(path)
    return sorted(out)


def _python_blocks(text, is_py):
    """Executable-looking code: the whole ``.py``, fenced blocks in markdown."""
    if is_py:
        return [text]
    lines, blocks, i = text.splitlines(True), [], 0
    while i < len(lines):
        if lines[i].startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            blocks.append("".join(lines[i + 1:j]))
            i = j + 1
        else:
            i += 1
    return blocks


def _validated_choices(field):
    """The literal list ``validate()`` compares ``self.<field>`` against."""
    match = re.search(rf"self\.{field} not in (\[[^\]]*\])",
                      _read(SKELETON_MODULE))
    return set(ast.literal_eval(match.group(1))) if match else set()


def test_documented_calls_use_parameters_the_code_accepts():
    """Snippets in docs and example scripts must call the API as it now is.

    Nothing here executes a markdown block, so a dropped or renamed parameter,
    or an option value the validator rejects, only surfaces when a reader
    copies it -- the same way the retired ``from coana import`` examples
    survived a whole release. Retired-but-normalized values count too: they
    run, but the UI no longer offers them, so the prose should not teach them.
    """
    targets = {"VisualizeSkeleton": (
        {field.name for field in dataclasses.fields(VISUALIZE_SKELETON)}, "")}
    for name in ("plot_individuals", "export_video"):
        fn = getattr(VISUALIZE_SKELETON, name)
        targets[name] = (set(inspect.signature(fn).parameters),
                         inspect.getsource(fn))
    for name in ("export_individuals_from_html", "export_video_from_html"):
        fn = getattr(viz, name)
        targets[name] = (set(inspect.signature(fn).parameters),
                         inspect.getsource(fn))
    choices = {"legend_mode": set(viz.LEGEND_MODES),
               "granularity": set(viz.PROFILE_GRANULARITIES),
               "brain_mesh": set(viz.BRAIN_MESH_SELECTIONS)}
    for field in ("skeleton_mode", "synapse_mode", "color_mode",
                  "expand_colors", "backend"):
        choices[field] = _validated_choices(field)
        assert choices[field], f"validate() no longer lists {field} inline"

    problems = []
    for path in _doc_sweep_files():
        rel = str(path.relative_to(PROJECT_ROOT))
        exempt = LEGACY_VOCAB_NOTES.get(rel, ())
        for code in _python_blocks(
                path.read_text(encoding="utf-8", errors="replace"),
                path.suffix == ".py"):
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue  # prose snippets and pseudo-code
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.keywords:
                    continue
                called = (getattr(node.func, "id", None)
                          or getattr(node.func, "attr", None))
                if called not in targets:
                    continue
                accepted, body = targets[called]
                for kw in node.keywords:
                    key, value = kw.arg, kw.value
                    if key is None:
                        continue
                    if key not in accepted and f"'{key}'" not in body:
                        problems.append(
                            f"{rel}:{node.lineno} {called}() does not take "
                            f"{key}")
                    if key not in choices or not isinstance(
                            value, ast.Constant) or not isinstance(
                            value.value, str):
                        continue
                    token = value.value
                    if token not in choices[key]:
                        if token in exempt:
                            continue
                        folded = viz.normalize_brain_mesh(token)
                        hint = (f" (use {folded!r})" if folded in choices[key]
                                else "")
                        problems.append(
                            f"{rel}:{kw.value.lineno} {key}={token!r} is not "
                            f"accepted{hint}")
    assert not problems, "\n".join(problems)


#: A prose figure quoted next to a constant name must be that constant's value.
_CAP_MENTION = re.compile(r"MAX_INDIVIDUAL_PROFILES[^0-9\n]{0,40}?(\d+)")


def test_prose_figures_for_the_profile_cap_are_current():
    """``MAX_INDIVIDUAL_PROFILES`` may be named without a number, never a stale one."""
    current = str(viz.MAX_INDIVIDUAL_PROFILES)
    stale = []
    for path in _doc_sweep_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _CAP_MENTION.finditer(text):
            if match.group(1) != current:
                line = text[:match.start()].count("\n") + 1
                stale.append(f"{path.relative_to(PROJECT_ROOT)}:{line} "
                             f"quotes {match.group(1)}, the cap is {current}")
    assert not stale, "\n".join(stale)


def test_freeze_view_is_a_real_field_defaulting_to_true():
    field = _dataclass_field("freeze_view")
    assert field.default is True
    assert field.type in (bool, "bool")


def test_freeze_view_documented_in_docstring_and_prose():
    assert "freeze_view" in inspect.getdoc(VISUALIZE_SKELETON)
    assert "freeze_view" in inspect.getsource(VISUALIZE_SKELETON)
    for rel in (GUIDE, SKILL):
        assert _mentions(rel, "freeze_view"), rel


def test_freeze_view_default_documented_in_prose():
    for rel in (GUIDE, SKILL):
        text = _read(rel)
        assert "freeze_view=True" in text or "default=True" in text.lower(), rel


def test_granularity_levels_match_the_module_constant():
    assert tuple(viz.PROFILE_GRANULARITIES) == ("legend", "layer", "type", "body")


def test_granularity_is_a_plot_individuals_argument():
    params = inspect.signature(VISUALIZE_SKELETON.plot_individuals).parameters
    assert "granularity" in params
    assert params["granularity"].default in viz.PROFILE_GRANULARITIES


def test_granularity_documented_with_every_level():
    doc = inspect.getdoc(VISUALIZE_SKELETON.plot_individuals)
    assert _mentions(GUIDE, "granularity"), GUIDE
    for level in viz.PROFILE_GRANULARITIES:
        assert level in _read(GUIDE).lower(), f"guide: granularity={level!r}"
        assert level in doc.lower(), f"plot_individuals docstring: {level!r}"


def test_trace_identity_tags_documented():
    for tag in ("drocatTrace", "drocatLegend"):
        assert _mentions(SKILL, tag), tag
    for method in ("_stamp_trace_identity", "_stamp_site_identity",
                   "classify_traces"):
        assert hasattr(VISUALIZE_SKELETON, method) or hasattr(viz, method), method


def test_html_reexport_helpers_are_module_attributes():
    for name in MODULE_FUNCTIONS:
        assert callable(getattr(viz, name)), name
        assert _mentions(SKILL, name), f"skill file omits {name}()"
    assert _mentions(GUIDE, "figure_from_plotly_html"), GUIDE
    # plotly's own reader really is missing in the pinned release.
    assert "read_html" in inspect.getdoc(viz.figure_payload_from_html)


def test_every_reexport_entrance_is_documented():
    """Each re-export helper exists in the module *and* is named by the prose."""
    documented = (_read(GUIDE) + _read(SKILL)).lower()
    for name in REEXPORT_FUNCTIONS:
        assert callable(getattr(viz, name)), name
        assert name.lower() in documented, f"guide/skill omit {name}"


def test_reexport_output_dir_naming_matches_the_code():
    source = inspect.getsource(viz.reexport_output_dir)
    assert f"{{stem}}{REEXPORT_SUFFIX}" in source, "destination rule changed"
    assert _mentions(GUIDE, REEXPORT_SUFFIX), GUIDE
    assert _mentions(OUTPUT_FILES, REEXPORT_SUFFIX), OUTPUT_FILES
    assert _mentions(UI_GUIDE, REEXPORT_SUFFIX), UI_GUIDE


def test_video_reexport_engine_is_the_export_method_switch():
    # The card's Frame Method select feeds this argument; the webdriver engine
    # is the session-based one, and kaleido is its automatic fallback.
    params = inspect.signature(viz.export_video_from_html).parameters
    assert "export_method" in params, "export_video_from_html lost export_method"
    assert "timeout" in params and "background_color" in params
    assert callable(getattr(viz, "_render_video_frames_via_session"))
    assert callable(getattr(viz, "_rotation_camera"))
    assert callable(getattr(viz, "_page_z_sign"))
    for rel in PROSE_FILES:
        assert _mentions(rel, "export_method"), rel
    assert _mentions(GUIDE, "_render_video_frames_via_session"), GUIDE
    assert _mentions(UI_GUIDE, "Frame Method"), UI_GUIDE


def test_retired_webdriver_video_exporter_stays_out_of_the_docs():
    # It was a hand-rolled duplicate of the session engine, deleted with the
    # Phase-7 replacement; naming it in prose would advertise a dead API.
    assert not hasattr(viz, "export_video_webdriver")
    for rel in ALL_DOCS:
        assert "export_video_webdriver" not in _read(rel), rel


def test_reexport_tool_is_registered_and_documented():
    assert REEXPORT_TOOL in _read(RUNNER), "plot3d_reexport not registered"
    assert REEXPORT_TOOL in _read(REEXPORT_CARD), "card does not run the tool"
    assert REEXPORT_TOOL in _read(GUIDE), GUIDE
    assert REEXPORT_TOOL in _read(UI_GUIDE), UI_GUIDE


def test_granularity_names_the_shared_constant_and_the_ui_knob():
    config = _read(UI_CONFIG)
    assert "PROFILE_GRANULARITIES" in config, "ui/config.py lost its own constant"
    for level in viz.PROFILE_GRANULARITIES:
        assert f'"{level}"' in config, f"ui/config.py is missing {level!r}"
    assert "profile_granularity" in _read(SKELETON_TAB), \
        "the Skeleton tab does not send granularity"
    assert _mentions(SKILL, "PROFILE_GRANULARITIES"), SKILL
    assert _mentions(GUIDE, "granularity="), GUIDE
    assert _mentions(UI_GUIDE, "Granularity"), UI_GUIDE


def test_freeze_view_ui_entrance_and_fit_behaviour():
    assert "freeze_view" in _read(UI_CONFIG), "no freeze_view setting"
    assert "freeze_view" in _read(SKELETON_TAB), "the tab does not send freeze_view"
    assert _mentions(UI_GUIDE, "Freeze/Fit"), UI_GUIDE
    # Fit has to clear the range, not merely re-enable autorange.
    script = inspect.getsource(VISUALIZE_SKELETON._freeze_view_html)
    assert "'scene.xaxis.range': null" in script, "Fit no longer releases the pin"
    assert "navigator.webdriver" in script, "freeze reached the export path"
    assert _mentions(GUIDE, "Freeze/Fit"), GUIDE
    assert _mentions(UI_GUIDE, "navigator.webdriver"), UI_GUIDE


def test_manifest_api_is_on_the_class_and_documented():
    # ``visualization_manifest`` is a method of VisualizeSkeleton, not a
    # module-level function, unlike the HTML re-export helpers above.
    for name in ("visualization_manifest", "write_visualization_manifest"):
        assert callable(getattr(VISUALIZE_SKELETON, name)), name
        assert _mentions(SKILL, name + "()"), name
    assert _mentions(OUTPUT_FILES, MANIFEST_NAME), OUTPUT_FILES


def test_manifest_keys_documented():
    source = inspect.getsource(VISUALIZE_SKELETON.visualization_manifest)
    text = _read(OUTPUT_FILES)
    for key in ("canonical_page", "legend_mode", "freeze_view", "traces"):
        assert f"'{key}'" in source, f"manifest drops {key}"
        assert key in text, f"OUTPUT_FILES.md omits manifest key {key}"
    # A page list and a copy of the pin the page already bakes have no reader,
    # so neither earns a key.
    for key in ("degraded_pages", "frozen_ranges"):
        assert f"'{key}'" not in source, f"{key} is written for no one"
        assert key not in text, f"OUTPUT_FILES.md still lists {key}"


def test_plot_individuals_alpha_doc_matches_the_dataclass_default():
    default = _dataclass_field("neuron_alpha").default
    expected = f"{default:g}"
    doc = inspect.getdoc(VISUALIZE_SKELETON.plot_individuals)
    block = _parameter_block(doc, "neuron_alpha")
    assert block, "plot_individuals docstring no longer documents neuron_alpha"
    claims = re.findall(r"defaults? to\s+(\d+(?:\.\d+)?)", block, re.IGNORECASE)
    assert claims, f"neuron_alpha block states no default: {block!r}"
    assert set(claims) == {expected}, (
        f"docstring claims default(s) {claims}, the dataclass default is "
        f"{expected}")


def test_viewer_page_names_documented():
    assert _mentions(OUTPUT_FILES, "_simplified.html"), OUTPUT_FILES
    for folder in ("exported_views/", "individual_profiles/", "pics_"):
        assert _mentions(OUTPUT_FILES, folder), f"OUTPUT_FILES.md omits {folder}"
