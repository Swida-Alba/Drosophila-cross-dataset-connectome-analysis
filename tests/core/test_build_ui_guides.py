"""Tests for the ui_guides markdown → themed-HTML builder.

The builder (scripts/build_ui_guides.py) is the anti-drift gate for the
instruction pages: ``docs/ui_guides/src/<name>.md`` regenerates the sibling
``<name>.html`` in place, and ``--check`` fails CI when a committed page is
stale. These tests exercise the CLI against temporary guide roots
(subprocess — the script is a standalone tool, not an importable package
module).
"""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "build_ui_guides.py"

SRC_PAGE = """---
title: Network
tag: Connectome
---

Build the **mutual** network — equivalent to Direct Connections with
source == target == the query set.

<div class="callout warn">
  <b>Limited scope:</b> raw HTML blocks pass through.
</div>

## Quick start

1. Pick the **dataset**.
2. Click **Find Network**.
"""


def _write_root(tmp_path: Path, body: str = SRC_PAGE) -> Path:
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "find_network.md").write_text(body, encoding="utf-8")
    return tmp_path


def _run(*args, cwd=None):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=cwd or str(PROJECT_ROOT))


def _built(root: Path) -> str:
    return (root / "find_network.html").read_text(encoding="utf-8")


def test_build_renders_chrome_lede_and_raw_html(tmp_path):
    root = _write_root(tmp_path)
    result = _run("--root", str(root))
    assert result.returncode == 0, result.stderr

    html = _built(root)
    # chrome: head links + main wrapper + back-link/tag header
    assert '<link rel="stylesheet" href="guide.css">' in html
    assert '<script src="guide-theme.js"></script>' in html
    assert '<main class="guide">' in html
    assert '<a class="back" href="README.html">' in html
    assert '<span class="tag">Connectome</span>' in html
    assert '<h1>Network</h1>' in html
    assert "<title>Network - DROCAT Instructions</title>" in html
    # lede: first paragraph wrapped, markdown inline rendered
    assert '<p class="lede">Build the <strong>mutual</strong> network' in html
    # raw HTML passthrough: the callout div survives verbatim
    assert '<div class="callout warn">' in html
    # headings/list converted
    assert "<h2>Quick start</h2>" in html
    assert "<li>Pick the <strong>dataset</strong>.</li>" in html


def test_check_passes_when_fresh_and_fails_when_stale(tmp_path):
    root = _write_root(tmp_path)
    assert _run("--root", str(root)).returncode == 0
    assert _run("--root", str(root), "--check").returncode == 0

    # touch the source: committed HTML is now stale
    src = root / "src" / "find_network.md"
    src.write_text(src.read_text(encoding="utf-8") + "\nExtra.\n",
                   encoding="utf-8")
    stale = _run("--root", str(root), "--check")
    assert stale.returncode == 1
    assert "STALE" in stale.stderr

    # a rebuild clears it
    assert _run("--root", str(root)).returncode == 0
    assert _run("--root", str(root), "--check").returncode == 0


def test_missing_page_is_stale_under_check(tmp_path):
    """A src page without a committed HTML sibling is a drift failure —
    otherwise a new guide could ship without its rendered page."""
    root = _write_root(tmp_path)
    stale = _run("--root", str(root), "--check")
    assert stale.returncode == 1
    assert "STALE" in stale.stderr


def test_missing_title_or_tag_refused(tmp_path):
    root = _write_root(tmp_path, body="---\ntitle: Network\n---\n\nBody.\n")
    result = _run("--root", str(root))
    assert result.returncode == 2
    assert "title:" in result.stderr or "tag:" in result.stderr

    root2 = _write_root(tmp_path / "r2", body="no front matter at all\n")
    result2 = _run("--root", str(root2))
    assert result2.returncode == 2


def test_front_matter_continuation_lines_join(tmp_path):
    root = _write_root(tmp_path, body=(
        "---\n"
        "title: A Very Long\n"
        "  Guide Title\n"
        "tag: Connectome\n"
        "---\n\n"
        "Body paragraph.\n"))
    result = _run("--root", str(root))
    assert result.returncode == 0, result.stderr
    assert "<h1>A Very Long Guide Title</h1>" in _built(root)


def test_find_network_page_is_up_to_date():
    """The committed find_network page matches its source (the repo's own
    --check, pinned as a test so local drift is caught without CI)."""
    result = _run("--check")
    assert result.returncode == 0, result.stderr
