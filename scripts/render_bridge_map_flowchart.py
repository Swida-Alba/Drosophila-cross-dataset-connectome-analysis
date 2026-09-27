"""Render the bidirectional type-mapper bridge map as an editorial flowchart.

Straight-line edition (user 2026-09-27): every connector is a STRAIGHT
double-headed segment, all connections bidirectional, and the layout is
verified programmatically — no segment may clip a node box it does not
connect (the generator exits non-zero on any violation).

Grounding (drift = non-zero exit):
- dataset<->column edges must come from ``BRIDGE_SOURCE_MAP`` (plus two
  explicitly-declared compositional incidences) — and every linker name
  must exist verbatim in its home dataset's neuron-table header;
- column<->column edges must be co-licensed in one ``BRIDGE_STANDARD``
  registry.

The ``fafb_alignment_cell_type`` fallback lane is intentionally NOT drawn
(fallback-only, never a primary route).
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comparison.cross_dataset_type_mapper import (  # noqa: E402
    BRIDGE_SOURCE_MAP,
    BRIDGE_STANDARD,
)

ROOT = Path(__file__).resolve().parents[1]

DATASET_NODES = ["BANC", "FAFB", "MCNS", "HEMI", "MANC"]
COLUMN_HOME = {
    "fafb_cell_type": "BANC",
    "malecns_cell_type": "BANC",
    "manc_cell_type": "BANC",
    "hemibrain_cell_type": "BANC",
    "Alternative Cell Type(s)": "BANC",
    "additional_type(s)": "FAFB",
    "flywireType": "MCNS",
    "hemibrainType": "MCNS",
    "mancType": "MCNS",
}
# home dataset -> the neuron table whose header must carry the column
HOME_TABLE = {
    "BANC": "datasets/banc_v626/banc_v626_allneurons_neuron_df.csv",
    "FAFB": "datasets/flywire_FAFB_v783/flywire_FAFB_v783_allneurons_neuron_df.csv",
    "MCNS": "datasets/male-cns_v1_0/male-cns_v1_0_allneurons_neuron_df.csv",
}
# column -> display label (verbatim metadata column names, request 2)
COLUMN_DISPLAY = {
    "fafb_cell_type": "fafb_cell_type",
    "malecns_cell_type": "malecns_cell_type",
    "manc_cell_type": "manc_cell_type",
    "hemibrain_cell_type": "hemibrain_cell_type",
    "Alternative Cell Type(s)": "Alternative Cell Type(s)",
    "additional_type(s)": "additional_type(s)",
    "flywireType": "flywireType",
    "hemibrainType": "hemibrainType",
    "mancType": "mancType",
}

EDGES = [
    # BANC label lanes (home side) + the aT-token landing (request 1: many
    # FAFB additional_type(s) tokens ARE BANC primaries — 290 of them —
    # the same token-landing structure as the MCNS flywireType lane)
    ("BANC", "fafb_cell_type"), ("BANC", "malecns_cell_type"),
    ("BANC", "manc_cell_type"), ("BANC", "hemibrain_cell_type"),
    ("BANC", "Alternative Cell Type(s)"), ("BANC", "additional_type(s)"),
    # shared-token exchange between the two FlyWire namespaces
    ("fafb_cell_type", "additional_type(s)"),
    ("Alternative Cell Type(s)", "additional_type(s)"),
    ("FAFB", "Alternative Cell Type(s)"), ("FAFB", "additional_type(s)"),
    ("FAFB", "fafb_cell_type"),
    # MCNS crosswalk lanes + the reverse-annotation landing
    ("MCNS", "flywireType"), ("MCNS", "hemibrainType"),
    ("MCNS", "mancType"), ("MCNS", "malecns_cell_type"),
    ("MCNS", "additional_type(s)"), ("FAFB", "flywireType"),
    # the ORIGINAL composed two-linker standard (MCNS --fT--aT-- FAFB)
    ("flywireType", "additional_type(s)"),
    # outer datasets
    ("HEMI", "hemibrainType"), ("HEMI", "hemibrain_cell_type"),
    ("MANC", "mancType"), ("MANC", "manc_cell_type"),
]

# compositional incidences (licensed by chain composition, not one map row)
COMPOSITIONAL_INCIDENCES = {
    # pure-ACT landing class (a FAFB node's first hop carries BANC's ACT;
    # 2,347 endpoint pairs live, ACT tokens are 7,637 FAFB primaries)
    ("FAFB", "Alternative Cell Type(s)"),
    # reverse-annotation landing from the crosswalk home (CL125 -> APDN3)
    ("MCNS", "additional_type(s)"),
    # aT-token landing (request 1, user 2026-09-27): 290 FAFB
    # additional_type(s) tokens ARE BANC primaries — the aT -> type ->
    # BANC chain class, the same token-landing structure as the MCNS
    # flywireType lane; survives curated subsumption only where the
    # curated lane has no same-token bridge (e.g. CB3767)
    ("BANC", "additional_type(s)"),
}

# hand-tuned straight-line layout: node -> (cx, cy); box 180x56
POS = {
    "BANC": (150, 460),
    "fafb_cell_type": (370, 265),
    "Alternative Cell Type(s)": (380, 730),
    "malecns_cell_type": (900, 240),
    "manc_cell_type": (370, 60),
    "hemibrain_cell_type": (180, 900),
    "FAFB": (620, 180),
    "additional_type(s)": (620, 460),
    "MCNS": (1100, 460),
    "flywireType": (880, 620),
    "hemibrainType": (940, 750),
    "mancType": (1200, 690),
    "HEMI": (1330, 760),
    "MANC": (1300, 140),
}
BOX_W, BOX_H = 180, 56


def _rect(node):
    cx, cy = POS[node]
    return (cx - BOX_W / 2, cy - BOX_H / 2, cx + BOX_W / 2, cy + BOX_H / 2)


def _trim(a, b):
    """Trim the a->b center line to the two boxes' boundaries."""
    import math
    ra, rb = _rect(a), _rect(b)
    ax, ay = POS[a]
    bx, by = POS[b]
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return None

    def exit_t(rect):
        # smallest t>0 where the ray exits rect
        cands = []
        if dx:
            for rx in (rect[0], rect[2]):
                t = (rx - ax) / dx
                if t > 0:
                    y = ay + t * dy
                    if rect[1] - 0.5 <= y <= rect[3] + 0.5:
                        cands.append(t)
        if dy:
            for ry in (rect[1], rect[3]):
                t = (ry - ay) / dy
                if t > 0:
                    x = ax + t * dx
                    if rect[0] - 0.5 <= x <= rect[2] + 0.5:
                        cands.append(t)
        return min(cands) if cands else 0.0

    t0, t1 = exit_t(ra), exit_t(rb)
    if t1 <= t0:
        return None
    return (ax + t0 * dx, ay + t0 * dy, ax + t1 * dx, ay + t1 * dy)


def _seg_hits_rect(x1, y1, x2, y2, rect):
    """True when the segment intersects the rectangle (Liang-Barsky)."""
    rx0, ry0, rx1, ry1 = rect
    dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x1 - rx0), (dx, rx1 - x1),
                 (-dy, y1 - ry0), (dy, ry1 - y1)):
        if p == 0:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return False
            t0 = max(t0, t)
        else:
            if t < t0:
                return False
            t1 = min(t1, t)
    return t1 >= t0


def layout_violations() -> list:
    """Straight segments must not clip any box they do not connect."""
    problems = []
    for a, b in EDGES:
        seg = _trim(a, b)
        if seg is None:
            problems.append(f"edge {a}-{b}: no visible straight segment")
            continue
        x1, y1, x2, y2 = seg
        for node in POS:
            if node in (a, b):
                continue
            if _seg_hits_rect(x1, y1, x2, y2, _rect(node)):
                problems.append(
                    f"edge {a}-{b} clips box {node} "
                    f"({x1:.0f},{y1:.0f} -> {x2:.0f},{y2:.0f})")
    return problems


def ground_check() -> list:
    problems = []
    # linker names must exist verbatim in the home dataset's table header
    for column, home in COLUMN_HOME.items():
        table = ROOT / HOME_TABLE[home]
        with open(table) as f:
            header = next(csv.reader(f))
        if column not in header:
            problems.append(
                f"linker name {column!r} not a column of {home} table")
    # dataset<->column incidences must come from BRIDGE_SOURCE_MAP
    licensed = set()
    for (home, column), targets in BRIDGE_SOURCE_MAP.items():
        home_family = None if home == "*" else _family(home)
        for target in targets:
            for edge in ((home_family, column), (column, _family(target))):
                licensed.add(edge)
                licensed.add((edge[1], edge[0]))
    for edge in EDGES:
        a, b = edge
        for da, db in ((a, b), (b, a)):
            if da in DATASET_NODES and db in COLUMN_HOME:
                if (da, db) not in licensed \
                        and (da, db) not in COMPOSITIONAL_INCIDENCES:
                    problems.append(f"unlicensed incidence: {da} - {db}")
    # column<->column edges must be co-licensed in one registry
    for a, b in EDGES:
        if a in COLUMN_HOME and b in COLUMN_HOME:
            composed = any(
                {a, b} <= {col for col, _ in registry}
                for registry in BRIDGE_STANDARD.values())
            if not composed:
                problems.append(f"unlicensed column-column edge: {a} - {b}")
    return problems


def _family(key: str) -> str:
    if key.startswith("banc"):
        return "BANC"
    if key.startswith("male-cns"):
        return "MCNS"
    if key.startswith("flywire"):
        return "FAFB"
    if key.startswith("hemibrain"):
        return "HEMI"
    if key.startswith("manc"):
        return "MANC"
    return key


SUBLABEL = {
    "BANC": "v626 \u00b7 v888 \u2014 release crosswalk",
    "FAFB": "v783",
    "MCNS": "v1.0 \u00b7 v0.9 \u2014 release_alias",
    "HEMI": "v1.2.1",
    "MANC": "v1.0 \u00b7 v1.2.1",
}


def build_html() -> str:
    connectors = []
    for a, b in EDGES:
        seg = _trim(a, b)
        x1, y1, x2, y2 = (round(v) for v in seg)
        connectors.append(
            f'      <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            'stroke="#4f5d75" stroke-width="1" '
            'marker-start="url(#arrow)" marker-end="url(#arrow)"/>')
    nodes = []
    for name, (cx, cy) in POS.items():
        is_dataset = name in DATASET_NODES
        display = name if is_dataset else COLUMN_DISPLAY[name]
        sublabel = SUBLABEL.get(name, f"home: {COLUMN_HOME.get(name, '')}")
        stroke = "#2d3142" if is_dataset else "rgba(45,49,66,0.30)"
        fill = "#ffffff" if is_dataset else "rgba(45,49,66,0.03)"
        x, y = cx - BOX_W / 2, cy - BOX_H / 2
        nodes.append(
            f'      <g>\n'
            f'        <rect x="{x:.0f}" y="{y:.0f}" width="{BOX_W}" '
            f'height="{BOX_H}" rx="6" fill="{fill}" stroke="{stroke}"/>\n'
            f'        <text x="{cx:.0f}" y="{cy - 4:.0f}" text-anchor="middle" '
            f'font-family="\'Geist\', sans-serif" font-size="11.5" '
            f'font-weight="500" fill="#2d3142">{display}</text>\n'
            f'        <text x="{cx:.0f}" y="{cy + 13:.0f}" text-anchor="middle" '
            f'font-family="\'Geist Mono\', monospace" font-size="7.5" '
            f'fill="#4f5d75">{sublabel}</text>\n'
            f'      </g>')
    return HTML.replace("__CONNECTORS__", "\n".join(connectors)).replace(
        "__NODES__", "\n".join(nodes))


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Type-mapper bridge map · every licensed route between datasets</title>
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --color-paper:   #f5f5f5;
      --color-ink:     #2d3142;
      --color-muted:   #4f5d75;
      --font-sans:     'Geist', system-ui, sans-serif;
      --font-serif:    'Instrument Serif', serif;
      --font-mono:     'Geist Mono', ui-monospace, monospace;
    }
    body { font-family: var(--font-sans); background: var(--color-paper);
           color: var(--color-ink); min-height: 100vh;
           display: flex; align-items: center; justify-content: center;
           padding: 3rem 2rem; }
    .frame { max-width: 1400px; width: 100%; }
    .eyebrow { font-family: var(--font-mono); font-size: 0.66rem;
               font-weight: 500; letter-spacing: 0.18em;
               text-transform: uppercase; color: var(--color-muted);
               margin-bottom: 0.5rem; }
    h1 { font-family: var(--font-serif);
         font-size: clamp(1.5rem, 2.4vw + 0.75rem, 2rem); font-weight: 400;
         letter-spacing: -0.02em; line-height: 1.15; margin-bottom: 1.5rem; }
    svg { width: 100%; min-width: 1000px; display: block; }
    .caption { font-size: 0.78rem; color: var(--color-muted);
               line-height: 1.55; max-width: 940px; margin-top: 1rem; }
    .caption code { font-family: var(--font-mono); font-size: 0.72rem; }
  </style>
</head>
<body>
  <div class="frame">
    <p class="eyebrow">Type mapping · Bridge map · Diagram Design</p>
    <h1>Every licensed route between datasets</h1>

    <svg viewBox="0 0 1440 900" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="bm-title bm-desc">
      <title id="bm-title">Type-mapper bridge map</title>
      <desc id="bm-desc">Bidirectional straight-line map of the type
        mapper's licensed bridges: dataset families (BANC, FAFB, MCNS,
        HEMI, MANC) connected through their metadata name-column linkers.
        Every connector is double-headed — each bridge traverses both
        directions. additional_type(s) is the shared-token exchange hub.</desc>
      <defs>
        <marker id="arrow" markerWidth="8" markerHeight="6" refX="7"
                refY="3" orient="auto-start-reverse">
          <polygon points="0 0, 8 3, 0 6" fill="#4f5d75"/>
        </marker>
      </defs>
      <rect width="100%" height="100%" fill="#f5f5f5"/>

      <!-- connectors (straight, before boxes) -->
      __CONNECTORS__

      <!-- dataset + column boxes -->
      __NODES__
    </svg>

    <p class="caption">
      Every connector is a straight double-headed line: each licensed
      bridge traverses both directions. Column nodes sit beside the dataset
      whose tables physically carry them (<em>home</em>); a column touching
      a second dataset is that dataset's <em>landing</em>.
      <code>additional_type(s)</code> is the shared-token exchange hub: FAFB
      annotation tokens frequently <em>are</em> the other datasets' primary
      type names — 290 of them are BANC primaries (the same token-landing
      structure as the MCNS <code>flywireType</code> lane), and BANC's
      <code>Alternative Cell Type(s)</code> carries 7,637 FAFB primary
      names. Not drawn: the universal same-name identity (<code>type</code> —
      any exact name matches across namespaces), the curated-subsumption and
      token-chaining walk rules, and the <code>fafb_alignment_cell_type</code>
      fallback lane (fallback-only, never a primary route). Release bridges
      ride the family nodes: BANC v626 ↔ v888 via
      <code>banc_release_crosswalk</code>, male-cns v0.9 ↔ v1.0 via
      <code>release_alias</code>; MANC columns land both manc releases.
      Grounded in <code>BRIDGE_SOURCE_MAP</code> /
      <code>BRIDGE_STANDARD</code> and the dataset table headers; see
      docs/AUTO_TYPE_MAPPING.md.
    </p>
  </div>
</body>
</html>
"""


def main() -> int:
    problems = ground_check() + layout_violations()
    if problems:
        for problem in problems:
            print(f"BRIDGE MAP DRIFT: {problem}", file=sys.stderr)
        return 1
    root = ROOT
    out = root / "outputs" / "type_mapping" / "bridge_map_flowchart.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    html = build_html()
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html):,} bytes); grounding + layout checks OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
