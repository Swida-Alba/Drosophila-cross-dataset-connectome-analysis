"""Generate the three skeleton-handling flowcharts (FAFB / male-cns / BANC).

Editorial flowcharts per the diagram-design conventions. Vertical decision
spine; branching diamonds get a side box (right) that REJOINS the spine via a
merge dot below the diamond — no dead-end branches.
"""
from pathlib import Path

INK = '#2d3142'; MUTED = '#4f5d75'; ACCENT = '#eb6c36'; PAPER = '#f5f5f5'

W = 920
MID = 300            # spine center
SIDE = 690           # side-branch box center
SIDE_W = 280
GAP_P = 26           # gap after a process box
GAP_D = 30           # gap before the next spine node after a plain diamond
BRANCH_EXTRA = 44    # extra spine space after a branching diamond (merge dot)


class Flow:
    def __init__(self, H):
        self.H = H
        self.E = [f'<rect width="{W}" height="{H}" fill="{PAPER}"/>',
                  f'<rect width="{W}" height="{H}" fill="url(#dots)"/>',
                  f'<text class="rail-label" x="90" y="104">FIRST RUN</text>',
                  f'<text class="rail-label" x="830" y="104" text-anchor="middle">CACHE USED</text>']

    def pill(self, y, w, h, text):
        self.E.append(f'<rect x="{MID-w/2}" y="{y}" width="{w}" height="{h}" rx="{h/2:.0f}" fill="{PAPER}" stroke="{INK}" stroke-width="1"/>')
        self.E.append(f'<text class="node-title" x="{MID}" y="{y+h/2+4}">{text}</text>')

    def proc(self, y, w, h, title, sub='', focal=False, cx=None):
        cx = MID if cx is None else cx
        fill = 'rgba(235,108,54,0.07)' if focal else PAPER
        stroke = ACCENT if focal else 'rgba(45,49,66,0.25)'
        self.E.append(f'<rect x="{cx-w/2}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
        cls = 'node-title focal-text' if focal else 'node-title'
        self.E.append(f'<text class="{cls}" x="{cx}" y="{y+18}">{title}</text>')
        if sub:
            self.E.append(f'<text class="node-sub" x="{cx}" y="{y+34}">{sub}</text>')

    def diamond(self, cy, w, h, lines):
        pts = f'{MID},{cy-h/2} {MID+w/2},{cy} {MID},{cy+h/2} {MID-w/2},{cy}'
        self.E.append(f'<polygon points="{pts}" fill="{PAPER}" stroke="{INK}" stroke-width="1"/>')
        for i, ln in enumerate(lines):
            dy = (i - (len(lines)-1)/2) * 12
            self.E.append(f'<text class="node-title" x="{MID}" y="{cy+dy+4}">{ln}</text>')

    def arrow(self, x1, y1, x2, y2, label='', accent=False):
        color = ACCENT if accent else MUTED
        marker = 'arr-a' if accent else 'arr-m'
        self.E.append(f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{color}" stroke-width="{1.2 if accent else 1}" marker-end="url(#{marker})"/>')
        if label:
            self.E.append(f'<text class="arrow-label" x="{x1+10}" y="{(y1+y2)/2+3}" text-anchor="start">{label}</text>')

    def elbow(self, pts, label='', accent=False):
        color = ACCENT if accent else MUTED
        marker = 'arr-a' if accent else 'arr-m'
        d = f'M {pts[0][0]} {pts[0][1]} ' + ' '.join(f'L {x} {y}' for x, y in pts[1:])
        self.E.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{1.2 if accent else 1}" marker-end="url(#{marker})"/>')
        if label:
            mid = pts[len(pts)//2]
            self.E.append(f'<text class="arrow-label" x="{mid[0]+8}" y="{mid[1]-6}" text-anchor="start">{label}</text>')

    def note(self, y, text):
        self.E.append(f'<text class="note-text" x="{MID}" y="{y}">{text}</text>')

    def merge(self, y):
        self.E.append(f'<circle cx="{MID}" cy="{y}" r="4" fill="{INK}"/>')

    def html(self, title, desc, footer):
        svg = [f'<svg viewBox="0 0 {W} {self.H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="fc-t fc-d">',
               f'<title id="fc-t">{title}</title>', f'<desc id="fc-d">{desc}</desc>',
               '<defs><pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse"><circle cx="11" cy="11" r="0.8" fill="rgba(45,49,66,0.10)"/></pattern>'
               '<marker id="arr-m" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#4f5d75"/></marker>'
               '<marker id="arr-a" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#eb6c36"/></marker></defs>',
               f'<rect width="{W}" height="{self.H}" fill="{PAPER}"/>',
               f'<rect width="{W}" height="{self.H}" fill="url(#dots)"/>']
        svg += self.E
        svg.append('</svg>')
        css = '''
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root { --paper: #f5f5f5; --ink: #2d3142; --muted: #4f5d75; --soft: #7a8399;
    --accent: #eb6c36; --sans: 'Geist', system-ui, sans-serif;
    --serif: 'Instrument Serif', serif; --mono: 'Geist Mono', ui-monospace, monospace; }
  body { min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 3rem 2rem; background: var(--paper); color: var(--ink); font-family: var(--sans); }
  .frame { width: 100%; max-width: 1160px; }
  .eyebrow { margin-bottom: 0.5rem; color: var(--muted); font: 500 0.66rem var(--mono); letter-spacing: 0.18em; text-transform: uppercase; }
  h1 { margin-bottom: 1.5rem; color: var(--ink); font: 400 clamp(1.5rem, 2.4vw + 0.75rem, 2rem)/1.15 var(--serif); letter-spacing: -0.02em; }
  svg { display: block; width: 100%; min-width: 760px; }
  .node-title { fill: var(--ink); font: 600 12px var(--sans); text-anchor: middle; }
  .node-sub { fill: var(--muted); font: 400 9px var(--mono); text-anchor: middle; }
  .arrow-label { fill: var(--ink); font: 600 9px var(--mono); }
  .rail-label { fill: var(--muted); font: 600 10px var(--mono); letter-spacing: 0.14em; text-anchor: middle; }
  .note-text { fill: var(--soft); font: 400 9px var(--mono); text-anchor: middle; }
  footer { margin-top: 1.5rem; color: var(--soft); font: 400 0.72rem var(--mono); }'''
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} · Flowchart</title>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&amp;family=Geist:wght@400;500;600&amp;family=Geist+Mono:wght@400;500;600&amp;display=swap" rel="stylesheet">
<style>{css}</style>
</head>
<body>
  <main class="frame">
    <p class="eyebrow">Flowchart · Diagram Design</p>
    <h1>{title}</h1>
{chr(10).join(svg)}
    <footer>{footer}</footer>
  </main>
</body>
</html>'''


def skeleton_flow(title, desc, footer, steps, H=None):
    """steps: (kind, ...) where kind:
       pill(text) | P(title, sub, focal) | D([lines], label, branch_title, branch_sub)
       | note(text). Branching diamonds get a side box that rejoins the spine."""
    f = Flow(H if H else 400)
    y = 40
    prev = None
    for it in steps:
        k = it[0]
        if k == 'pill':
            f.pill(y, 240, 34, it[1])
            if prev is not None:
                f.arrow(MID, prev, MID, y)
            prev = y + 34; y += 34 + 26
        elif k == 'P':
            _, t, s, focal = it
            h = 56 if s else 40
            f.proc(y, 280, h, t, s, focal)
            if prev is not None:
                f.arrow(MID, prev, MID, y)
            prev = y + h; y += h + GAP_P
        elif k == 'D':
            label = it[2] if len(it) > 2 else ''
            branch = it[3] if len(it) > 3 else None   # (title, sub)
            f.diamond(y + 38, 300, 76, it[1])
            if prev is not None:
                f.arrow(MID, prev, MID, y, label)
            diamond_bottom = y + 76
            if branch:
                bt, bs = branch
                # side box parallel to the diamond (drawn at SIDE, not MID)
                f.proc(y, SIDE_W, 62, bt, bs, cx=SIDE)
                # diamond right tip → side box left edge
                f.arrow(MID + 150, y + 38, SIDE - SIDE_W/2, y + 38, label)
                # side box bottom → down → left → merge into the spine
                rejoin_y = y + 76 + BRANCH_EXTRA / 2
                f.elbow([(SIDE, y + 31), (SIDE, rejoin_y), (MID + 6, rejoin_y)])
                f.merge(rejoin_y)
                prev = rejoin_y
                y = y + 76 + BRANCH_EXTRA
            else:
                prev = diamond_bottom
                y = diamond_bottom + GAP_D
    H_final = y + 60
    f.H = H_final
    f.E[0] = f.E[0].replace('height="400"', f'height="{H_final}"')
    f.E[1] = f.E[1].replace('height="400"', f'height="{H_final}"')
    return f.html(title, desc, footer)


# ================= FAFB =================
fafb = skeleton_flow(
    'FAFB skeleton handling — fetch · cache · simplify · scene',
    'FAFB neuron skeleton flow: raw cache vs healed bundle, CAVE replacement store/API, fast prep, tube mesh, decimation, cross-template bridging, tilt correction.',
    'Tilt correction is data-level (rotates coordinates of mesh + neurons + synapses + ROIs); the FAFB camera has no tilt. Native FAFB skeletons need no MCNS→FAFB transform.',
    [
        ('pill', "FAFB body IDs (layer)"),
        ('D', ['raw skeleton cache hit?', '(raw_skeletons/*.swc.zst)'], 'raw level',
         ('Cache used', 'skip bundle; still check extrusion')),
        ('P', 'Healed bundle → raw cache (lazy)', 'sk_lod1_783_healed.zip (~13.8 GB)', False),
        ('P', 'Extrusion check → CAVE replacement', 'cave_skeletons/ hit or wavefront tree', False),
        ('P', 'Fast prep: nodes → 25% retention', 'topology floor: roots · branch · terminals', True),
        ('P', 'Tube mesh (6 pts)', '', False),
        ('P', 'Face decimation (slider 0.90)', '', False),
        ('D', ['cross-template selection?', "(brain_mesh 'BANC' / 'male-cns')"], 'slider',
         ('Bridge ≤ 2 hops', 'FLYWIRE → BANC / → JRCFIB2022M')),
        ('P', 'Tilt correction (data-level)', '−3° Z / −3° Y · mesh + neurons + synapses + ROIs', False),
        ('pill', "plot (FLYWIRE nm / selected space)"),
        ('note', 'native FAFB: no MCNS→FAFB transform — skeletons are already in FLYWIRE coordinates (skip_transform)'),
        ('note', 'line mode: 90% node reduction · no render-boundary skeletonization · CAVE replacements are pre-skeletonized trees'),
    ])

# ================= male-cns =================
mcns = skeleton_flow(
    'male-cns skeleton handling — fetch · cache · simplify · scene',
    'male-cns neuron skeleton flow: raw cache vs batched NeuPrint fetch, FAFB-format styling (no coordinate change), tube mesh + decimation, affine to nm, cross-template bridging.',
    'Raw cache stores JRCFIB2022Mraw voxels; the nm affine runs once in the layer block. No tilt in the JRCFIB2022M frame.',
    [
        ('pill', "male-cns body IDs (layer)"),
        ('D', ['raw skeleton cache hit?', '(raw_skeletons/*.swc.zst · raw voxels)'], 'raw level',
         ('Cache used', 'skip NeuPrint fetch')),
        ('P', 'Batched parallel NeuPrint fetch', 'token-gated → persist raw (JRCFIB2022Mraw)', False),
        ('P', 'FAFB-format styling', 'smooth · resample · radius — NO coordinate change', False),
        ('P', 'Tube mesh → decimation', 'cache level 0.95 · extra if slider > 0.95', False),
        ('D', ['cross-template selection?', "(brain_mesh 'FAFB' / 'BANC')"], 'slider',
         ('Bridge ≤ 2 hops', 'JRCFIB2022Mraw → FLYWIRE / → BANC')),
        ('P', 'Affine raw → nm (layer block)', 'JRCFIB2022Mraw → JRCFIB2022M — native scene', False),
        ('pill', "plot (JRCFIB2022M nm / selected space)"),
        ('note', 'preprocessing "transform" stage is FAFB-format styling only — the coordinate affine happens once, in the layer block'),
        ('note', 'line mode: 50% node reduction · no tilt (JRCFIB2022M frame)'),
    ])

# ================= BANC =================
banc = skeleton_flow(
    'BANC skeleton handling — fetch · cache · simplify · scene',
    'BANC neuron skeleton flow: raw cache vs the GCS chain (L2 → full → pcg-µm), radius repair + 240 nm normalization, class-dependent decimation, cross-template bridging.',
    'Raw cache entries keep a provenance header (gcs_l2 / gcs_full / pcg_um_x1000) restored on every load. BANC renders natively in BANC nm — no tilt.',
    [
        ('pill', "BANC body IDs (layer)"),
        ('D', ['raw skeleton cache hit?', '(provenance header restores class)'], 'raw level',
         ('Cache used', 'skip GCS download')),
        ('P', 'GCS chain (per-neuron fallback)', 'L2 (nm) → full (nm) → pcg-µm ×1000', False),
        ('P', 'Radius repair → normalize 240 nm', '≤0/NaN → 1 nm · median → 240 nm target', False),
        ('P', 'Tube mesh (6 pts)', '', False),
        ('D', ['source class?', 'full vs l2 / pcg-µm (provenance)'], 'provenance',
         ('L2 / pcg-µm: excess decim', 'asked 0.95 → 50% · ≤0.90 → as-is')),
        ('P', 'full: decimation to slider', '4,000-face full-res floor', False),
        ('D', ['cross-template selection?', "(brain_mesh 'FAFB' / 'male-cns')"], 'slider',
         ('Bridge ≤ 2 hops', 'BANC → JRCFIB2022M / → FLYWIRE')),
        ('pill', "plot (BANC nm / selected space)"),
        ('note', 'no tilt (BANC frame) · line mode: full-res 50%, L2/pcg untouched'),
    ])

Path('docs/visualizations/fafb_skeleton_flow.html').write_text(fafb)
Path('docs/visualizations/malecns_skeleton_flow.html').write_text(mcns)
Path('docs/visualizations/banc_skeleton_flow.html').write_text(banc)
print('three skeleton flowcharts regenerated with rejoining branches')
