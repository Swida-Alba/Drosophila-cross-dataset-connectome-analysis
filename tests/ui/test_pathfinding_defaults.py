"""Q3 (plan 2026-10-16): the pathfinding defaults must be REAL defaults.

Threshold Basis, Combine Thresholds and Run Mode were read by the tabs through
`get_user_default`, but carried no entry in `DEFAULT_SETTING_SPECS` — a key
with no spec can never be overridden, so Settings offered a knob that resolved
to a constant and the legacy 'off'/'on' normalization placed after coercion was
unreachable code. Every test here is the leak it closed.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ui.config as cfg

KEYS = ("threshold_basis", "threshold_combination", "corridor_mode")
VOCAB = {
    "threshold_basis": cfg.THRESHOLD_BASIS_CHOICES,
    "threshold_combination": cfg.THRESHOLD_COMBINATION_CHOICES,
    "corridor_mode": cfg.CORRIDOR_MODE_CHOICES,
}


@pytest.fixture
def cfg_file(monkeypatch, tmp_path):
    """A private local_config.json, so nothing touches the developer's."""
    path = tmp_path / "local_config.json"
    monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", path)
    return path


@pytest.mark.parametrize("key", KEYS)
def test_each_key_is_an_offered_setting(key):
    spec = cfg.DEFAULT_SETTING_SPECS.get(key)
    assert spec is not None, f"{key} is read by the tabs but not registered"
    assert spec["kind"] == "select"
    assert spec["group"] in {gid for gid, _ in cfg.DEFAULT_SETTING_GROUPS}
    assert spec["options"] is VOCAB[key]

    # the built-in default must validate against its own spec — a rename on
    # one side alone would silently make every saved override invalid
    assert cfg._coerce_user_default(key, cfg.DEFAULTS[key]) == cfg.DEFAULTS[key]


def test_basis_override_reaches_the_tab(cfg_file):
    assert cfg.set_user_default("threshold_basis", "connection_ratio")
    assert cfg.get_user_default("threshold_basis") == "connection_ratio"
    # and the shared select opens on the matching label
    from ui.components import pathfinding_controls as pc
    assert (pc.THRESHOLD_BASIS_CHOICES[cfg.get_user_default("threshold_basis")]
            in pc.THRESHOLD_BASIS_OPTIONS)


def test_combination_override_reaches_the_tab(cfg_file):
    assert cfg.set_user_default("threshold_combination", "and")
    assert cfg.get_user_default("threshold_combination") == "and"


def test_run_mode_override_round_trips(cfg_file):
    assert cfg.set_user_default("corridor_mode", "both")
    assert cfg.get_user_default("corridor_mode") == "both"
    assert cfg.load_local_config()["user_defaults"]["corridor_mode"] == "both"


def test_legacy_spellings_still_resolve(cfg_file):
    """Stored BEFORE the normalization existed: 'off' meant Complete-only and
    'on' meant Complete+Fast. They must keep working on read…"""
    cfg.save_local_config({"user_defaults": {"corridor_mode": "off"}})
    assert cfg.get_user_default("corridor_mode") == "complete"
    cfg.save_local_config({"user_defaults": {"corridor_mode": "on"}})
    assert cfg.get_user_default("corridor_mode") == "both"
    """…and the WRITE path normalizes too: the old post-coercion map sat after
    the option test, so saving 'off' was rejected and the value never stored."""
    assert cfg.set_user_default("corridor_mode", "off")
    assert cfg.load_local_config()["user_defaults"]["corridor_mode"] == "complete"


def test_an_invalid_override_falls_back(cfg_file):
    cfg.save_local_config({"user_defaults": {"corridor_mode": "sometimes"}})
    assert cfg.get_user_default("corridor_mode") == cfg.DEFAULTS["corridor_mode"]
    assert cfg.has_user_default("corridor_mode") is False


def test_reset_restores_the_builtin(cfg_file):
    cfg.set_user_default("corridor_mode", "fast")
    assert cfg.reset_user_default("corridor_mode")
    assert cfg.get_user_default("corridor_mode") == "auto"


# --- one vocabulary, four surfaces -----------------------------------------

def test_the_tabs_and_settings_share_one_vocabulary():
    from ui.components import pathfinding_controls as pc

    assert pc.RUN_MODE_OPTIONS is cfg.CORRIDOR_MODE_CHOICES
    assert pc.THRESHOLD_BASIS_OPTIONS == list(
        cfg.THRESHOLD_BASIS_CHOICES.values())
    assert pc.THRESHOLD_COMBINATION_OPTIONS == list(
        cfg.THRESHOLD_COMBINATION_CHOICES.values())
    assert cfg.DEFAULTS["corridor_mode"] in pc.RUN_MODE_OPTIONS
    assert pc.weight_basis_of(_w("Connection ratio")) == "connection_ratio"
    assert pc.weight_basis_of(_w("Synapse count")) == "synapse"


def test_the_run_mode_labels_are_not_re_declared_in_the_tabs():
    """L3: a tab that re-typed 'Complete + Fast' would drift from the Settings
    card the moment one side was reworded."""
    root = Path(__file__).resolve().parents[2]
    for rel in ("ui/tabs/find_path.py", "ui/tabs/find_shortest.py",
                "ui/tabs/inter_dataset.py"):
        src = (root / rel).read_text()
        assert '"Complete + Fast"' not in src, rel
        assert "Complete only" not in src, rel


def test_the_settings_card_renders_the_new_selects():
    """The card builds options straight from the spec, so a dict option list
    must be dict-safe in `select_input` (its no-default branch used to take
    `options[0]`, which raises on a dict)."""
    from ui.components.common import select_input
    import inspect

    body = inspect.getsource(select_input)
    assert "isinstance(options, dict)" in body


def test_the_rendered_card_stores_engine_keys():
    """Live-render law: the Default Settings card is built and every one of
    the three new selects holds an ENGINE KEY as its value. A card that
    displayed labels but stored them would write a value `_coerce_user_default`
    rejects, and the saved default would silently never apply."""
    from nicegui import Client
    from nicegui.page import page
    from ui.tabs.settings import create_settings_tab

    client = Client(page("/defaults-card-live"))
    with client:
        create_settings_tab()

    want = {
        "Threshold Basis": cfg.THRESHOLD_BASIS_CHOICES,
        "Combine Thresholds": cfg.THRESHOLD_COMBINATION_CHOICES,
        "Run Mode (Complete / Fast)": cfg.CORRIDOR_MODE_CHOICES,
    }
    found = {}
    for el in client.elements.values():
        label = (getattr(el, "_props", None) or {}).get("label")
        if label in want:
            found[label] = el
    for label, choices in want.items():
        el = found.get(label)
        assert el is not None, f"{label} is not rendered by the card"
        assert set(el.options) == set(choices), label
        assert el.value in choices, (label, el.value, list(choices))
        # what the reader sees is the human label of that stored key
        assert choices[el.value] in list(el.options.values())


def _w(value):
    class _Widget:
        def __init__(self, value):
            self.value = value
    return _Widget(value)


# --- per-pair shortest scope (issue 2026-10-22) ----------------------------

@pytest.mark.parametrize("key", (
        "per_pair_scope", "scope_lt_mode", "scope_pad",
        "scope_hard_cap", "scope_discovery_cap"))
def test_scope_defaults_are_registered(key):
    spec = cfg.DEFAULT_SETTING_SPECS.get(key)
    assert spec is not None, f"{key} has a default but no Settings spec"
    assert spec["group"] == "pathfinding_output"
    assert cfg._coerce_user_default(key, cfg.DEFAULTS[key]) == cfg.DEFAULTS[key]


def test_scope_settings_round_trip(cfg_file):
    assert cfg.set_user_default("scope_lt_mode", "external")
    assert cfg.set_user_default("scope_pad", 1)
    assert cfg.set_user_default("scope_hard_cap", 5)
    assert cfg.get_user_default("scope_lt_mode") == "external"
    assert cfg.get_user_default("scope_pad") == 1
    assert cfg.get_user_default("scope_hard_cap") == 5
