"""Round-13: the runner refuses unknown method keys loudly.

Driving find_path with method_name='run' used to generate an init-only
script that "completed" in seconds without running any analysis — a
silent no-op. The guard must raise; a present-but-empty entry (the
tmvev dispatcher) stays legal.
"""
import asyncio
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ui"))

from ui.runner import ScriptRunner, TOOL_REGISTRY  # noqa: E402


def test_unknown_method_refuses():
    r = ScriptRunner()
    with pytest.raises(ValueError, match="unknown method 'run'"):
        r._generate_script(
            "find_path",
            {"dataset": "flywire_FAFB_v783", "sourceNeurons": ["aMe12"],
             "targetNeurons": ["PPL101"]},
            "run", None)


def test_known_method_still_generates():
    r = ScriptRunner()
    script = r._generate_script(
        "find_path",
        {"dataset": "flywire_FAFB_v783", "sourceNeurons": ["aMe12"],
         "targetNeurons": ["PPL101"]},
        "find_all_path", None)
    assert "FindAllPath" in script


def test_present_but_empty_entry_stays_legal():
    # type_mapping_validation dispatches by tool name; its methods entry
    # is intentionally empty and must not trip the guard.
    r = ScriptRunner()
    tool = TOOL_REGISTRY["type_mapping_validation"]
    assert tool["methods"].get("run") == ""
    script = r._generate_script(
        "type_mapping_validation", {}, "run", None)
    assert isinstance(script, str)
