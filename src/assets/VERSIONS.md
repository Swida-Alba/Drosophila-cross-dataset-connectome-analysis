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
| `cytoscape/cytoscape.min.js` | cytoscape | 3.28.1 | https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js | 2026-10-02 |
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
