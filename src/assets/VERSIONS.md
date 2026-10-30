# Vendored assets — version pins

Manually vendored browser libraries inlined into generated HTML reports
(offline availability, plan-offline-html-exports). Each entry: library,
version, source URL, download date. plotly is NOT vendored — it is read
from the installed plotly.py package at generation time
(`vendored_assets.inline_plotly()`), so it is version-matched to the
figure producer automatically.

| File | Library | Version | Source URL | Downloaded |
|---|---|---|---|---|
| `vis-network.min.js` | vis-network (standalone UMD) | 10.1.2 | https://unpkg.com/vis-network/standalone/umd/vis-network.min.js | 2026-10-02 |
| `cytoscape/cytoscape.min.js` | cytoscape | 3.28.1 + DROCAT arrow-blend patch | https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js | 2026-10-02 |
| `cytoscape/dagre.min.js` | dagre | 0.8.5 | https://unpkg.com/dagre@0.8.5/dist/dagre.min.js | 2026-10-02 |
| `cytoscape/cytoscape-dagre.js` | cytoscape-dagre | 2.5.0 | https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js | 2026-10-02 |
| `cytoscape/layout-base.js` | layout-base | 1.0.2 | https://unpkg.com/layout-base@1.0.2/layout-base.js | 2026-10-02 |
| `cytoscape/cose-base.js` | cose-base | 1.0.3 | https://unpkg.com/cose-base@1.0.3/cose-base.js | 2026-10-02 |
| `cytoscape/cytoscape-cose-bilkent.js` | cytoscape-cose-bilkent | 4.1.0 | https://unpkg.com/cytoscape-cose-bilkent@4.1.0/cytoscape-cose-bilkent.js | 2026-10-02 |
| `cytoscape/cytoscape-fcose.js` | cytoscape-fcose | 2.2.0 | https://unpkg.com/cytoscape-fcose@2.2.0/cytoscape-fcose.js | 2026-10-02 |
| `cytoscape/klay.js` | klayjs | 0.4.1 | https://unpkg.com/klayjs@0.4.1/klay.js | 2026-10-02 |
| `cytoscape/cytoscape-klay.js` | cytoscape-klay | 3.1.4 | https://unpkg.com/cytoscape-klay@3.1.4/cytoscape-klay.js | 2026-10-02 |
| `cytoscape/cytoscape-svg.js` | cytoscape-svg | 0.4.0 | https://unpkg.com/cytoscape-svg@0.4.0/cytoscape-svg.js | 2026-10-02 |

Update procedure: download the new file, update the table, run
`tests/core/test_html_exports_offline.py`, and smoke-render one report of
each affected kind.

## Patched vendor file: `cytoscape/cytoscape.min.js`

The vendored Cytoscape bundle carries two local patches (2026-10-30):

- **Why:** upstream `drawArrowhead` paints an opaque `destination-out`
  silhouette under every filled arrowhead whenever the arrow alpha
  (element opacity × `line-opacity`) is below 1. That cut-out erases
  underlying ink, so overlapping translucent arrowheads never composite.
  Separately, the triangle's line gap (`2 × width × arrow-scale`) is
  shorter than the drawn head (`arrowShapeWidth × getArrowWidth`), so the
  shaft runs several pixels inside the head; with the cut-out gone that
  overlap shows as a darker stripe in a translucent arrow.
- **Change:** the guard `1===o&&"hollow"!==u||` becomes `"hollow"===u&&`,
  so only hollow arrowheads keep the cut-out; and the triangle arrow gets
  `gap:function(e){return t.arrowShapeWidth*t.getArrowWidth(...)}`, so the
  shaft ends exactly at the arrowhead base.
- **Marker:** the file starts with
  `/* DROCAT patch (Cytoscape 3.28.1): blended arrow fill; shaft ends at the arrowhead base; hollow cut-out kept */`.
- **Script:** `scripts/patch_cytoscape_arrow_blend.py` (SHA-256-guarded,
  idempotent; `--check` is the CI gate). Pristine SHA-256
  `92d752b48ea949720675865197fd2a0001c95bc5888545e990af60321712d4c6`;
  patched SHA-256
  `65cfafc30915fda27501126a3531d4f5667dadd3864261841de9923ce6b5a77f`.
- **Upgrading:** re-vendor the new upstream file, then either rework and
  re-run the patch script against the new bundle or verify upstream fixed
  the cut-out and arrow geometry and update/remove the patch. `--check`
  and `tests/vispath/test_cytoscape_arrow_blend_patch.py` must pass.
