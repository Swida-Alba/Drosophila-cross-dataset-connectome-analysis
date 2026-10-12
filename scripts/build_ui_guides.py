#!/usr/bin/env python3
"""Build docs/ui_guides pages from markdown sources (anti-drift gate).

docs/ui_guides holds instruction pages linked from the UI's Information
buttons. Historically they were hand-written HTML (or hand-synced md/html
twins), so the same content was edited twice and drifted. This script makes
``docs/ui_guides/src/<name>.md`` the source of truth and regenerates the
sibling ``docs/ui_guides/<name>.html`` in place — same URLs, same chrome
(``guide.css`` + ``guide-theme.js``), fully offline.

Source contract (``docs/ui_guides/src/<name>.md``)::

    ---
    title: Network
    tag: Connectome
    ---

    <first paragraph becomes the lede>
    <rest of the markdown body; raw HTML blocks (callout divs, tables)
    pass through untouched — the convention the md twins already use>

Required front-matter fields: ``title`` (the page ``<h1>``) and ``tag``
(the header chip). Optional fields: ``subtag`` (a second chip inside the
h1 — the tab a sub-tab guide belongs to), ``tag_class`` (chip class
variant, e.g. ``sys``/``nb``), ``page_title`` (the fuller head
``<title>``; default ``{title} - DROCAT Instructions``), and
``lede: false`` (the first body paragraph is wrapped as ``p.lede``
unless this is set). The front-matter reader is a deliberate 20-liner
(key: value), not YAML.

README.html is intentionally NOT generated: it is a curated editorial
index (panel → guide table with hand-written "Covers" blurbs), not a
per-guide render. Until every guide has a src page it could not even be
fully derived; it stays hand-maintained.

Usage:
    python scripts/build_ui_guides.py            # regenerate in place
    python scripts/build_ui_guides.py --check    # exit 1 when stale (CI)
    python scripts/build_ui_guides.py --root DIR # operate on another root

Exit codes: 0 ok, 1 stale pages found (--check), 2 configuration error
(no sources / missing required front-matter fields).

CI: the static-gates job runs ``--check`` right after
``scripts/maintenance/check_docs_links.py``.
"""
import argparse
import difflib
import sys
from pathlib import Path

import markdown

# Extensions: tables (parameter/option grids); fenced_code and
# attr_list are available for authoring (no current page uses fenced
# fences or {:attr} syntax — kept so new guides can). Raw HTML blocks
# (the callout divs every guide already uses) pass through by default
# — that IS the authoring convention, so md_in_html is deliberately
# not enabled.
_EXTENSIONS = ["tables", "fenced_code", "attr_list"]

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{page_title} - DROCAT Instructions</title>
  <link rel="stylesheet" href="guide.css">
  <script src="guide-theme.js"></script>
</head>
<body>
<main class="guide">
  <header class="guide-top">
    <a class="back" href="README.html">&larr; Back to guides</a>
    <span class="tag{tag_class}">{tag}</span>
  </header>
  <h1>{title}{subtag}</h1>
{body}
</main>
</body>
</html>
"""


def parse_front_matter(text: str) -> tuple:
    """Split ``---`` front-matter from the body.

    Returns ({key: value}, body). Keys are lower-cased; a value continues
    over indented lines. Unknown/missing fields are the caller's problem.
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.split("\n")
    try:
        end = lines.index("---", 1)
    except ValueError:
        return {}, text
    meta: dict = {}
    key = None
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[:1].isspace() and key:
            meta[key] = f"{meta[key]} {line.strip()}"
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        meta[key] = value.strip()
    return meta, "\n".join(lines[end + 1:]).lstrip("\n")


def render_page(meta: dict, body: str) -> str:
    """Render one source page into the themed HTML chrome."""
    title = str(meta.get("title", "")).strip()
    tag = str(meta.get("tag", "")).strip()
    if not title or not tag:
        raise ValueError(
            "front-matter must define 'title:' and 'tag:'")
    # Optional chrome variants the hand pages use: a sub-tag inside the h1
    # (the tab a sub-tab guide belongs to) and a header-tag class variant.
    subtag = str(meta.get("subtag", "")).strip()
    subtag_html = f' <span class="tag">{subtag}</span>' if subtag else ""
    # page_title: the fuller head <title> some hand pages used (the h1
    # carries the short title + subtag); falls back to title.
    page_title = str(meta.get("page_title", "")).strip() or title
    tag_class = str(meta.get("tag_class", "")).strip()
    tag_class_html = f" {tag_class}" if tag_class else ""
    # The first paragraph of the body is the lede: split the SOURCE (not
    # the output) at the first blank line, render the two halves
    # separately, and wrap the lede in the class — regexing rendered HTML
    # would misfire on the raw callout divs several guides open with.
    # `lede: false` opts out (pages that open with a plain paragraph).
    blocks = body.split("\n\n", 1)
    lede_src = blocks[0]
    rest_src = blocks[1] if len(blocks) > 1 else ""
    lede = markdown.markdown(lede_src, extensions=_EXTENSIONS)
    if lede.startswith("<p>") and str(meta.get("lede", "true")).strip().lower() \
            not in ("false", "no", "off"):
        lede = f'<p class="lede">{lede[3:]}'
    body_html = f"{lede}\n"
    if rest_src.strip():
        body_html += markdown.markdown(rest_src, extensions=_EXTENSIONS)
    return _TEMPLATE.format(title=title, page_title=page_title, tag=tag,
                            tag_class=tag_class_html,
                            subtag=subtag_html, body=body_html)


def _targets(root: Path):
    src_dir = root / "src"
    for src in sorted(src_dir.glob("*.md")):
        yield src, root / f"{src.stem}.html"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_root = Path(__file__).resolve().parents[1] / "docs" / "ui_guides"
    parser.add_argument("--root", type=Path, default=default_root,
                        help="guides root (default: docs/ui_guides)")
    parser.add_argument("--check", action="store_true",
                        help="fail when a committed page is stale")
    args = parser.parse_args(argv)

    sources = list(_targets(args.root))
    if not sources:
        print(f"no source pages under {args.root / 'src'}", file=sys.stderr)
        return 2

    stale = 0
    for src, out in sources:
        meta, body = parse_front_matter(src.read_text(encoding="utf-8"))
        try:
            rendered = render_page(meta, body)
        except ValueError as exc:
            print(f"{src}: {exc}", file=sys.stderr)
            return 2
        if args.check:
            current = out.read_text(encoding="utf-8") if out.exists() else ""
            if current != rendered:
                stale += 1
                diff = "\n".join(list(difflib.unified_diff(
                    current.splitlines(), rendered.splitlines(),
                    str(out), f"{out} (regenerated)", lineterm=""))[:40])
                print(f"STALE {out}\n{diff}", file=sys.stderr)
            else:
                print(f"up-to-date {out}")
            continue
        out.write_text(rendered, encoding="utf-8")
        print(f"built {out}")

    if args.check and stale:
        print(f"\n{stale} page(s) stale — run "
              "scripts/build_ui_guides.py and commit", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
