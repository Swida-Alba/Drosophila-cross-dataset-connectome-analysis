"""Tests for scripts/maintenance/verify_doc_cites.py (I-7, 2026-09-28).

The checker exists because line-number cites in a moving tree decay
within hours: a cite is stale when the path is missing, the line is out
of range, or the citing line's named symbol no longer sits within ±3
lines of the cite.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "maintenance"))

import verify_doc_cites as vdc  # noqa: E402


def test_cites_detect_missing_out_of_range_and_drifted(tmp_path):
    src = tmp_path / "src" / "mod.py"
    src.parent.mkdir(parents=True)
    src.write_text(
        "def kept():\n"      # 1
        "    return 1\n"     # 2
        + "\n" * 3           # 3-5
        + "def tail():\n"    # 6
        "    return 2\n",    # 7
        encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    doc = docs / "a.md"
    doc.write_text(
        'see `src/mod.py:1` with `kept`\n'    # ok: symbol adjacent
        'see `src/mod.py:99`\n'               # line out of range
        'see `src/mod.py:5` with `moved_away`\n'  # in range, symbol gone
        'see `src/gone.py:1`\n'               # missing file
        'see `src/mod.py:2`\n',               # in range, no symbol
        encoding="utf-8")

    cites = vdc.collect_cites(doc)
    assert len(cites) == 5
    verdicts = [vdc.check_cite(tmp_path, doc, raw, line, syms)
                for _, raw, line, syms in cites]
    assert verdicts[0] == "ok"
    assert "out of range" in verdicts[1]
    assert verdicts[2].startswith("symbol")
    assert "missing file" in verdicts[3]
    assert verdicts[4] == "ok"


def test_fenced_code_blocks_and_prose_dates_are_not_cites(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    doc = docs / "b.md"
    doc.write_text(
        "dated prose 2026-09-28:5 does not match, nor does v1.2:3\n"
        "```\n"
        "src/mod.py:1 inside a fence is example text\n"
        "```\n"
        "back `src/mod.py:2` after the fence\n",
        encoding="utf-8")
    cites = vdc.collect_cites(doc)
    assert [(raw, line) for _, raw, line, _ in cites] == [
        ("src/mod.py", 2)]
