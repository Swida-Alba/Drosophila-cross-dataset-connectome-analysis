"""Render the bidirectional type-mapper bridge map as an editorial flowchart.

Derives the dataset<->linker-column incidences from the live
``BRIDGE_SOURCE_MAP`` and the column<->column two-linker adjacencies from
``BRIDGE_STANDARD`` co-membership (the licensed compositions), checks the
hand-laid-out diagram against them (drift = non-zero exit), and writes a
self-contained editorial HTML in the Diagram Design system to
``outputs/type_mapping/bridge_map_flowchart.html``.

The diagram is BIDIRECTIONAL: every connector is double-headed, because
every licensed bridge is traversable in both directions (the derivation
walk is direction-symmetric).  A dashed connector means the adjacency is
licensed but not exercised by the current release tables.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comparison.cross_dataset_type_mapper import (  # noqa: E402
    BRIDGE_SOURCE_MAP,
    BRIDGE_STANDARD,
)

# ---------------------------------------------------------------------------
# The diagram's declared edge list — (source, target, style).  Datasets are
# family nodes ("MCNS" covers v1.0/v0.9, "BANC" covers v626/v888); the
# release-internal bridges ride each family node's sublabel.
# ---------------------------------------------------------------------------
DATASET_NODES = ["BANC", "FAFB", "MCNS", "HEMI", "MANC"]
COLUMN_NODES = {
    # column: (display, home-family)
    "fafb_cell_type": ("fafb_cell_type", "BANC"),
    "fafb_alignment_cell_type": ("fafb_alignment_cell_type", "BANC"),
    "malecns_cell_type": ("malecns_cell_type", "BANC"),
    "manc_cell_type": ("manc_cell_type", "BANC"),
    "hemibrain_cell_type": ("hemibrain_cell_type", "BANC"),
    "Alternative Cell Type(s)": ("Alternative Cell Type(s)", "BANC"),
    "additional_type(s)": ("additional_type(s)", "FAFB"),
    "flywireType": ("flywireType", "MCNS"),
    "hemibrainType": ("hemibrainType", "MCNS"),
    "mancType": ("mancType", "MCNS"),
}
EDGE_STYLE = {
    # (source, target) -> "" solid | "dashed" licensed-not-exercised
    ("fafb_alignment_cell_type", "additional_type(s)"): "dashed",
}
EDGES = [
    # BANC home lanes
    ("BANC", "fafb_cell_type"), ("BANC", "fafb_alignment_cell_type"),
    ("BANC", "malecns_cell_type"), ("BANC", "manc_cell_type"),
    ("BANC", "hemibrain_cell_type"), ("BANC", "Alternative Cell Type(s)"),
    # the shared-token exchange between the two FlyWire namespaces
    ("fafb_cell_type", "additional_type(s)"),
    ("fafb_alignment_cell_type", "additional_type(s)"),
    ("Alternative Cell Type(s)", "additional_type(s)"),
    ("FAFB", "Alternative Cell Type(s)"),
    ("FAFB", "additional_type(s)"),
    ("FAFB", "fafb_cell_type"), ("FAFB", "fafb_alignment_cell_type"),
    # MCNS crosswalk lanes + the reverse-annotation landing
    ("MCNS", "flywireType"), ("MCNS", "hemibrainType"),
    ("MCNS", "mancType"), ("MCNS", "malecns_cell_type"),
    ("MCNS", "additional_type(s)"),
    ("FAFB", "flywireType"),
    # outer datasets
    ("HEMI", "hemibrainType"), ("HEMI", "hemibrain_cell_type"),
    ("MANC", "mancType"), ("MANC", "manc_cell_type"),
]


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


# Incidences licensed by COMPOSITION rather than a single map row —
# declared explicitly with their justification so the grounding check can
# tell them apart from drift:
#   - FAFB--ACT: the pure-ACT landing class (a FAFB node's first hop
#     carries BANC's Alternative Cell Type(s); 2,347 endpoint pairs live).
#   - MCNS--aT: the reverse-annotation landing from the crosswalk home
#     (MCNS CL125 --aT--> FAFB APDN3; verified live).
COMPOSITIONAL_INCIDENCES = {
    ("FAFB", "Alternative Cell Type(s)"),
    ("MCNS", "additional_type(s)"),
}


def ground_check() -> list:
    """Compare the declared edges against the live licensing constants."""
    problems = []
    # dataset<->column incidences must come from BRIDGE_SOURCE_MAP
    licensed = set()
    for (home, column), targets in BRIDGE_SOURCE_MAP.items():
        home_family = None if home == "*" else _family(home)
        for target in targets:
            for edge in (
                    (home_family if home_family is None else home_family,
                     column),
                    (column, _family(target))):
                # bidirectional map: license both orientations
                licensed.add(edge)
                licensed.add((edge[1], edge[0]))
    for edge in EDGES:
        a, b = edge
        if a in DATASET_NODES and b in COLUMN_NODES:
            if (a, b) not in licensed and (a, b) not in COMPOSITIONAL_INCIDENCES:
                problems.append(f"unlicensed dataset-column edge: {a} - {b}")
        elif a in COLUMN_NODES and b in DATASET_NODES:
            if (a, b) not in licensed and (a, b) not in COMPOSITIONAL_INCIDENCES:
                problems.append(f"unlicensed column-dataset edge: {a} - {b}")
        elif a in COLUMN_NODES and b in COLUMN_NODES:
            # both columns must be co-licensed in one pair's registry
            composed = any(
                {a, b} <= {col for col, _ in registry}
                for registry in BRIDGE_STANDARD.values())
            if not composed:
                problems.append(f"unlicensed column-column edge: {a} - {b}")
    # and the licensed incidences must not be missing from the diagram
    for a, b in sorted(licensed):
        if a in DATASET_NODES and b in COLUMN_NODES \
                and (a, b) not in EDGES and b in COLUMN_NODES:
            problems.append(f"diagram misses licensed edge: {a} - {b}")
    return problems


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
      --color-accent:  #eb6c36;
      --font-sans:     'Geist', system-ui, sans-serif;
      --font-serif:    'Instrument Serif', serif;
      --font-mono:     'Geist Mono', ui-monospace, monospace;
    }
    body { font-family: var(--font-sans); background: var(--color-paper);
           color: var(--color-ink); min-height: 100vh;
           display: flex; align-items: center; justify-content: center;
           padding: 3rem 2rem; }
    .frame { max-width: 1320px; width: 100%; }
    .eyebrow { font-family: var(--font-mono); font-size: 0.66rem;
               font-weight: 500; letter-spacing: 0.18em;
               text-transform: uppercase; color: var(--color-muted);
               margin-bottom: 0.5rem; }
    h1 { font-family: var(--font-serif);
         font-size: clamp(1.5rem, 2.4vw + 0.75rem, 2rem); font-weight: 400;
         letter-spacing: -0.02em; line-height: 1.15; margin-bottom: 1.5rem; }
    svg { width: 100%; min-width: 1000px; display: block; }
    .caption { font-size: 0.78rem; color: var(--color-muted);
               line-height: 1.55; max-width: 900px; margin-top: 1rem; }
    .caption code { font-family: var(--font-mono); font-size: 0.72rem; }
  </style>
</head>
<body>
  <div class="frame">
    <p class="eyebrow">Type mapping · Bridge map · Diagram Design</p>
    <h1>Every licensed route between datasets</h1>

    <svg viewBox="0 0 1320 900" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="bm-title bm-desc">
      <title id="bm-title">Type-mapper bridge map</title>
      <desc id="bm-desc">Bidirectional map of the type mapper's licensed
        bridges: dataset families (BANC, FAFB, MCNS, HEMI, MANC) connected
        through their name-column linkers. Every connector is
        double-headed — each bridge is traversable in both directions.
        The alignment fallback lane is accented; its dashed edge to
        additional_type(s) is licensed but not exercised by the current
        release tables.</desc>
      <defs>
        <marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3"
                orient="auto-start-reverse">
          <polygon points="0 0, 8 3, 0 6" fill="#4f5d75"/>
        </marker>
        <marker id="arrow-accent" markerWidth="8" markerHeight="6" refX="7"
                refY="3" orient="auto-start-reverse">
          <polygon points="0 0, 8 3, 0 6" fill="#eb6c36"/>
        </marker>
      </defs>
      <rect width="100%" height="100%" fill="#f5f5f5"/>

      <text x="40" y="60" fill="#4f5d75" font-size="8" font-family="'Geist Mono', monospace" letter-spacing="0.18em">BANC LABEL LANES</text>
      <text x="560" y="60" fill="#4f5d75" font-size="8" font-family="'Geist Mono', monospace" letter-spacing="0.18em">SHARED-TOKEN EXCHANGE</text>
      <text x="880" y="60" fill="#4f5d75" font-size="8" font-family="'Geist Mono', monospace" letter-spacing="0.18em">MCNS CROSSWALK LANES</text>

      <!-- connectors (before boxes) -->
      __CONNECTORS__

      <!-- dataset + column boxes -->
      __NODES__
    </svg>

    <p class="caption">
      Every connector is double-headed: each licensed bridge traverses both
      ways. Column nodes sit beside the dataset whose tables physically
      carry them (<em>home</em>); a column touching a second dataset is that
      dataset's <em>landing</em>. The accented
      <code>fafb_alignment_cell_type</code> lane is fallback-only — it fills
      a mapping just where <code>fafb_cell_type</code> has no winner — and
      its dashed edge to <code>additional_type(s)</code> is the licensed
      token-chaining adjacency, unexercised by the current tables. Release
      bridges ride the family nodes: BANC v626 ↔ v888 via
      <code>banc_release_crosswalk</code>, male-cns v0.9 ↔ v1.0 via
      <code>release_alias</code>; MANC columns land both manc releases.
      Not drawn: the universal same-name identity (<code>type</code> — any
      exact name matches across namespaces) and the curated-subsumption and
      token-chaining walk rules, which shape chains, not lanes. Grounded in
      <code>BRIDGE_SOURCE_MAP</code> / <code>BRIDGE_STANDARD</code>; see
      docs/AUTO_TYPE_MAPPING.md.
    </p>
  </div>
</body>
</html>
"""

# hand layout: (cx, cy) per node id; box 170x56
POS = {
    "BANC": (155, 430),
    "fafb_cell_type": (420, 435),
    "fafb_alignment_cell_type": (420, 550),
    "malecns_cell_type": (420, 320),
    "manc_cell_type": (420, 205),
    "hemibrain_cell_type": (420, 90),
    "Alternative Cell Type(s)": (420, 680),
    "FAFB": (710, 400),
    "additional_type(s)": (710, 600),
    "MCNS": (985, 400),
    "flywireType": (985, 120),
    "hemibrainType": (985, 580),
    "mancType": (985, 700),
    "HEMI": (1170, 700),
    "MANC": (1170, 520),
}

# hand-routed connectors: (path-d, style) — style ""|accent|dashed
CONNECTORS = [
    # BANC fan: the two upward lanes leave the TOP edge, four lanes share
    # the right edge at 12px spacing (attach 419/431/443/455)
    ("M 155,402 V 98 Q 155,90 163,90 H 335", ""),
    ("M 170,402 V 213 Q 170,205 178,205 H 335", ""),
    ("M 240,419 H 280 Q 288,419 288,411 V 328 Q 288,320 296,320 H 335", ""),
    ("M 240,431 H 335", ""),
    ("M 240,443 H 260 Q 268,443 268,451 V 542 Q 268,550 276,550 H 335", ""),
    ("M 240,455 H 252 Q 260,455 260,463 V 657 Q 260,665 268,665 H 335", ""),
    # fct -> aT (chained two-linker standard) ; fct -> FAFB landing
    ("M 505,443 H 647 Q 655,443 655,451 V 564 Q 655,572 663,572", ""),
    ("M 505,427 H 602 Q 610,427 610,421 V 421 Q 610,415 618,415 H 625", ""),
    # align -> FAFB landing (solid) ; align -> aT (dashed: the token-
    # chaining adjacency is licensed but unexercised by current tables)
    ("M 505,550 H 622 Q 630,550 630,542 V 436 Q 630,428 638,428", ""),
    ("M 505,565 H 596 Q 604,565 604,573 V 592 Q 604,600 612,600 H 625", "dashed-accent"),
    # ACT -> aT and ACT -> FAFB (pure-ACT landing class)
    ("M 505,680 H 810 Q 818,680 818,672 V 608 Q 818,600 810,600 H 795", ""),
    ("M 505,665 H 602 Q 610,665 610,657 V 436 Q 610,428 618,428", ""),
    # aT <-> FAFB vertical
    ("M 710,572 V 428", ""),
    # aT -> MCNS (reverse-annotation landing) ; mct -> MCNS
    ("M 795,585 H 850 Q 858,585 858,577 V 418 Q 858,410 866,410 H 900", ""),
    ("M 505,320 H 884 Q 892,320 892,328 V 382 Q 892,390 900,390 H 900", ""),
    # FAFB -> fT landing ; fT <-> MCNS
    ("M 740,372 V 128 Q 740,120 748,120 H 900", ""),
    ("M 985,148 V 372", ""),
    # MCNS -> hT/mT ; outer landings
    ("M 985,428 V 552", ""),
    ("M 1070,415 H 1082 Q 1090,415 1090,423 V 692 Q 1090,700 1082,700 H 1070", ""),
    ("M 1070,588 H 1077 Q 1085,588 1085,596 V 692 Q 1085,700 1093,700 H 1085", ""),
    ("M 1070,692 H 1077 Q 1085,692 1085,684 V 528 Q 1085,520 1093,520 H 1085", ""),
    # the two long BANC-column landings, routed around the outside
    ("M 420,62 V 48 Q 420,40 428,40 H 1262 Q 1270,40 1270,48 V 692 Q 1270,700 1262,700 H 1255", ""),
    ("M 505,205 H 1230 Q 1238,205 1238,213 V 512 Q 1238,520 1246,520 H 1255", ""),
]


def build_html() -> str:
    connectors = []
    for d, style in CONNECTORS:
        if style == "accent":
            connectors.append(
                f'      <path d="{d}" fill="none" stroke="#eb6c36" '
                'stroke-width="1" marker-start="url(#arrow-accent)" '
                'marker-end="url(#arrow-accent)"/>')
        elif style == "dashed-accent":
            connectors.append(
                f'      <path d="{d}" fill="none" stroke="#eb6c36" '
                'stroke-width="1" stroke-dasharray="5,4" '
                'marker-start="url(#arrow-accent)" '
                'marker-end="url(#arrow-accent)"/>')
        else:
            connectors.append(
                f'      <path d="{d}" fill="none" stroke="#4f5d75" '
                'stroke-width="1" marker-start="url(#arrow)" '
                'marker-end="url(#arrow)"/>')

    nodes = []
    for name, (cx, cy) in POS.items():
        is_dataset = name in DATASET_NODES
        display = name
        sublabel = ""
        if is_dataset:
            if name == "BANC":
                sublabel = "v626 \u00b7 v888 \u2014 release crosswalk"
            elif name == "MCNS":
                sublabel = "v1.0 \u00b7 v0.9 \u2014 release_alias"
            elif name == "MANC":
                sublabel = "v1.0 \u00b7 v1.2.1"
            elif name == "FAFB":
                sublabel = "v783"
            else:
                sublabel = "v1.2.1"
        else:
            display, home = COLUMN_NODES[name]
            sublabel = f"home: {home}"
        stroke = "#2d3142"
        fill = "#ffffff"
        if not is_dataset:
            fill = "rgba(45,49,66,0.03)"
            stroke = "rgba(45,49,66,0.30)"
        if name == "fafb_alignment_cell_type":
            stroke = "#eb6c36"
            fill = "#ffffff"
        x, y = cx - 85, cy - 28
        nodes.append(
            f'      <g>\n'
            f'        <rect x="{x}" y="{y}" width="170" height="56" rx="6" '
            f'fill="{fill}" stroke="{stroke}"/>\n'
            f'        <text x="{cx}" y="{cy - 4}" text-anchor="middle" '
            f'font-family="\'Geist\', sans-serif" font-size="11.5" '
            f'font-weight="500" fill="#2d3142">{display}</text>\n'
            f'        <text x="{cx}" y="{cy + 13}" text-anchor="middle" '
            f'font-family="\'Geist Mono\', monospace" font-size="7.5" '
            f'fill="#4f5d75">{sublabel}</text>\n'
            f'      </g>')
    html = HTML.replace("__CONNECTORS__", "\n".join(connectors))
    return html.replace("__NODES__", "\n".join(nodes))


def main() -> int:
    problems = ground_check()
    if problems:
        for problem in problems:
            print(f"BRIDGE MAP DRIFT: {problem}", file=sys.stderr)
        return 1
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs" / "type_mapping" / "bridge_map_flowchart.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    html = build_html()
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html):,} bytes); grounding check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
