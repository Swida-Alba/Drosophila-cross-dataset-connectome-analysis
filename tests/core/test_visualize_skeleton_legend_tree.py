"""Tests for the collapsible tree legend panel injected into viewer HTML.

``legend_mode='tree'`` renders exactly like ``'type'`` (the native
legend used by static exports) and additionally tags traces with
``drocatLegend`` meta so the exported interactive HTML can embed a
collapsible type -> neuron legend panel built client-side.
"""

from collections import namedtuple
from pathlib import Path
import json
import re
import shutil
import subprocess
import sys

import plotly.graph_objects as go
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import visualize_skeleton  # noqa: E402
from visualize_skeleton import LEGEND_MODES, VisualizeSkeleton  # noqa: E402


def _make_visualizer(legend_mode='tree', fig=None):
    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.background_color = 'white'
    visualizer.legend_mode = legend_mode
    # _legend_tree_html bakes the opacity-mesh table off the live figure.
    visualizer.fig_3d = fig
    return visualizer


def _tagged_figure():
    """A figure shaped like a tree export: tagged neurons, a paired
    synapse legend dummy, and a mesh pinned last."""
    fig = go.Figure()
    for i, (group, item) in enumerate(
        [('Tm3', '101'), ('Tm3', '102'), ('Mi1', '201')]
    ):
        trace = go.Scatter3d(
            x=[0, i + 1], y=[0, 1], z=[0, 1], mode='lines',
            line=dict(color='#1f77b4', width=4),
            name=group, legendgroup=group, showlegend=(i != 1),
        )
        trace.legendrank = i * 100
        trace.meta = {'drocatLegend': {
            'kind': 'neuron', 'group': group, 'item': item,
        }}
        fig.add_trace(trace)

    dummy = go.Scatter3d(
        x=[None], y=[None], z=[None], mode='markers',
        name='synapses 0 -> 1', legendgroup='synapses 0 -> 1',
        showlegend=True, marker=dict(size=10, color='rgb(44,160,44)'),
    )
    fig.add_trace(dummy)

    mesh = go.Mesh3d(
        x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0], j=[1], k=[2],
        color='#c8e6f0', opacity=0.05, name='Brain mesh', showlegend=True,
    )
    mesh.legendrank = visualize_skeleton.BRAIN_MESH_LEGEND_RANK
    fig.add_trace(mesh)
    return fig


def test_legend_modes_include_tree():
    assert 'tree' in LEGEND_MODES
    assert 'type' in LEGEND_MODES


def test_tree_custom_group_detection():
    visualizer = _make_visualizer()
    visualizer.custom_layer_names = []
    assert visualizer._tree_uses_custom_groups() is False
    visualizer.custom_layer_names = ['Group A', 'Group B']
    assert visualizer._tree_uses_custom_groups() is True


def test_tree_neuron_label_neuprint_uses_instance():
    import pandas as pd

    visualizer = _make_visualizer('male-cns:v1.0')
    visualizer.dataset = 'male-cns:v1.0'
    row = pd.Series({'bodyId': 11309, 'instance': 'aMe4_L'})
    assert visualizer._tree_neuron_label('11309', row) == '11309_aMe4_L'
    # no instance -> bare bodyId; no row -> raw neuron id
    assert visualizer._tree_neuron_label(
        '11309', pd.Series({'bodyId': 11309})) == '11309'
    assert visualizer._tree_neuron_label('11309', None) == '11309'


def test_tree_neuron_label_fafb_uses_type_and_hemisphere():
    import pandas as pd

    visualizer = _make_visualizer()
    visualizer.dataset = 'flywire_FAFB_v783'
    row = pd.Series({'bodyId': '7205759406', 'flywireType': 'Tm3',
                     'somaSide': 'L'})
    assert visualizer._tree_neuron_label('x', row) == '7205759406_Tm3_L'
    # hemisphere from the instance suffix when no side column exists
    row2 = pd.Series({'bodyId': '7205759406', 'flywireType': 'Tm3',
                      'instance': 'Tm3_R'})
    assert visualizer._tree_neuron_label('x', row2) == '7205759406_Tm3_R'
    # known type without hemisphere keeps just the type
    row3 = pd.Series({'bodyId': '7205759406', 'flywireType': 'Tm3'})
    assert visualizer._tree_neuron_label('x', row3) == '7205759406_Tm3'


def test_tree_neuron_label_cross_dataset_overlay_prefers_native_type():
    import pandas as pd

    visualizer = _make_visualizer()
    visualizer.dataset = 'flywire_FAFB_v783'
    row = pd.Series({
        'bodyId': 42274,
        'type': 'SMP227',
        'flywireType': 'CB1449,CB2843',
        'somaSide': 'L',
    })
    assert visualizer._tree_neuron_label(
        'x', row, source_dataset='male-cns:v1.0'
    ) == '42274_SMP227_L'


def test_tree_neuron_label_normalizes_legacy_names():
    """Legacy navis neuron names ('{instance} ({bodyId})') normalize to the
    same '{bodyId}_{instance}' shape as properly resolved labels."""
    import pandas as pd

    visualizer = _make_visualizer('male-cns:v1.0')
    # no matching metadata row: parse the legacy name
    assert visualizer._tree_neuron_label(
        'MeVPMe7_L (12805)', None) == '12805_MeVPMe7_L'
    # matching row wins with the proper instance column
    row = pd.Series({'bodyId': 12805, 'instance': 'MeVPMe7_L'})
    assert visualizer._tree_neuron_label(
        'MeVPMe7_L (12805)', row) == '12805_MeVPMe7_L'


def test_tree_hover_body_id_resolution():
    class _Vol:
        id = 720575940647731252  # FAFB-sized: int64-exact only

    assert VisualizeSkeleton._tree_hover_body_id(
        [_Vol()], 0, '7205759406477') == 720575940647731252
    # source_index beyond the list, empty list, un-int-able id, no list
    assert VisualizeSkeleton._tree_hover_body_id([_Vol()], 5, '1') is None
    assert VisualizeSkeleton._tree_hover_body_id([], 0, '1') is None
    assert VisualizeSkeleton._tree_hover_body_id([object()], 0, None) is None
    assert VisualizeSkeleton._tree_hover_body_id(None, 0, '1') is None


def test_tree_leaf_hover_embeds_bodyid_without_renaming_trace():
    """Ordinary tree scenes: the interactive hover reads like the bodyId
    leaf while the trace keeps its type-level name for the native legend
    (static exports never see hoverlabels)."""
    trace = go.Scatter3d(x=[1], y=[2], z=[3], mode='lines', name='Tm3')
    ok = VisualizeSkeleton._apply_tree_leaf_hover(
        trace, '720575940647731252_Tm3_L', 720575940647731252)
    assert ok is True
    assert '<b>720575940647731252_Tm3_L</b>' in trace.hovertemplate
    assert 'bodyId 720575940647731252' in trace.hovertemplate
    # legend identity untouched: hover-only change
    assert trace.name == 'Tm3'
    assert trace.customdata is None
    assert trace.legendgroup is None


def test_tree_leaf_hover_falls_back_to_leaf_label():
    """Without a resolvable bodyId the hover still names the leaf (the
    label itself starts with the bodyId) instead of the type name."""
    trace = go.Scatter3d(x=[1], y=[2], z=[3], mode='lines', name='Tm3')
    ok = VisualizeSkeleton._apply_tree_leaf_hover(
        trace, '11309_aMe4_L', None)
    assert ok is False
    assert trace.hovertemplate == '<b>11309_aMe4_L</b><extra></extra>'
    assert trace.name == 'Tm3'


def test_legend_tree_html_contains_panel_and_markers():
    html = _make_visualizer()._legend_tree_html()
    assert 'drocat-legend-tree' in html
    assert 'drocatLegend' in html  # meta key read client-side
    assert '"meshRankBase": 100000000' in html
    # theme-aware via the body class the theme switch toggles
    assert 'drocat-theme-dark' in html
    # replaces the native legend on injected pages only
    assert '.js-plotly-plot .legend{display:none !important;}' in html
    # banner-aware positioning + webdriver guard
    assert 'drocat-warning-container' in html
    assert 'navigator.webdriver' in html
    # panel sits on the right, below the theme switch; content-width (shrink
    # to fit short labels) capped at 420px / the viewport.
    assert 'position:fixed;right:10px;top:60px;' in html
    assert '"brainMeshRank": 200000000' in html
    assert '"vncMeshRank": 200000001' in html
    assert 'width:fit-content;min-width:180px;' in html
    assert 'max-width:min(420px,calc(100vw - 20px));' in html
    assert 'overflow-y:auto;overflow-x:hidden;box-sizing:border-box;' in html
    assert '.drocat-lt-items{padding-left:16px;min-width:max-content;}' in html
    # manual detector matches Plotly 6.4.0's native doubleClickDelay default
    assert '"doubleClickMs": 300' in html
    assert 'isDoubleClick' in html
    # the exported panel explains the configured timing to users
    assert 'Double-click window: under ' in html
    assert 'CONFIG.doubleClickMs' in html
    # Custom bodyId layers with skeleton + soma traces count one neuron, and
    # every group row stays expandable down to its bodyId-level children —
    # singleton groups included; only bodyId leaf rows are childless.
    assert 'custom: false' in html
    assert 'g.custom = g.custom || !!lg.customGroup;' in html
    assert 'function groupNeuronItemCount(g)' in html
    assert 'var neuronCount = neuronItemCount || (g.indices.length - g.sites.length);' in html
    assert 'attachDirectRow' not in html
    assert 'neuronItemCount === 1' not in html
    assert 'var typeNeuronCount = tt.itemOrder.length;' in html
    # explicit closed/open caret glyphs keep the direction in sync with state
    assert '\\u25B6' in html
    assert r"caret.textContent = open ? '\u25BC' : '\u25B6';" in html
    assert "row.setAttribute('aria-expanded', open ? 'true' : 'false');" in html
    # ROI meshes share an expandable group; brain and VNC remain root rows.
    assert "panel, 'ROI meshes'" in html
    assert 'm.rank < CONFIG.brainMeshRank' in html
    # pre/post site traces get leaf rows; long item lists are scroll-capped
    assert 'addSiteLeaves' in html
    assert "children.length > 10" in html
    assert '.drocat-lt-items.drocat-lt-scroll{max-height:224px;' in html
    assert 'min-width:0;overflow-y:auto;overflow-x:hidden;}' in html
    # horizontal overflow must stay mouse-reachable: the panel mounts its own
    # bar, and the axis is hidden so no engine can add a native bar under it.
    # Deleting the ::-webkit-scrollbar rule this replaced reproduced the
    # doubled bar at the same widths and the same gap, so that rule alone was
    # never enough.
    assert '::-webkit-scrollbar' not in html
    assert ('.drocat-lt-hscroll{position:sticky;bottom:0;left:0;'
            'display:none;height:10px;width:100%;cursor:pointer;'
            'background:rgba(255,255,255,0.92);border-radius:4px;}') in html
    assert ('body.drocat-theme-dark .drocat-lt-hscroll-thumb'
            '{background:rgba(255,255,255,0.4);}') in html
    assert 'function mountHScrollbar(container)' in html
    assert 'function updateHScrollbars()' in html
    assert 'mountHScrollbar(panel);' in html
    assert 'updateHScrollbars();' in html
    # a boot that cannot build the panel puts the native legend back and
    # names the script version, so a legend-less page stays diagnosable
    assert '"treeScriptVersion": "%s"' % (
        visualize_skeleton.LEGEND_TREE_SCRIPT_VERSION) in html
    assert 'tries < CONFIG.bootMaxPolls' in html
    assert "id = 'drocat-legend-fallback'" in html
    assert "display:block !important" in html


def test_write_plotly_html_embeds_tree_only_when_requested(tmp_path):
    visualizer = _make_visualizer()

    on_path = tmp_path / 'on.html'
    visualizer._write_plotly_html(
        _tagged_figure(), str(on_path), legend_tree=True
    )
    on_html = on_path.read_text(encoding='utf-8')
    assert 'drocat-legend-tree' in on_html
    assert 'drocatLegend' in on_html  # meta serialized into the figure

    off_path = tmp_path / 'off.html'
    visualizer._write_plotly_html(_tagged_figure(), str(off_path))
    off_html = off_path.read_text(encoding='utf-8')
    assert 'drocat-legend-tree' not in off_html
    assert 'drocatLegend' in off_html  # meta is harmless without the panel


def test_singleton_custom_group_page_embeds_group_and_bodyid(tmp_path):
    """A single-neuron custom group keeps its bodyId-level data in the
    page: the client-side tree renders every group row expandable down
    to its bodyId leaf, so the figure must carry group and item labels."""
    fig = go.Figure()
    trace = go.Scatter3d(
        x=[0, 1], y=[0, 1], z=[0, 1], mode='lines',
        line=dict(color='#d62728', width=4),
        name='r1_SLP459_x1', legendgroup='r1_SLP459_x1', showlegend=True,
    )
    trace.legendrank = 0
    trace.meta = {'drocatLegend': {
        'kind': 'neuron', 'group': 'r1_SLP459_x1', 'type': 'SLP459',
        'item': '459_SLP459_R', 'color': '#d62728', 'customGroup': True,
    }}
    fig.add_trace(trace)

    visualizer = _make_visualizer()
    page = tmp_path / 'singleton.html'
    visualizer._write_plotly_html(fig, str(page), legend_tree=True)
    html = page.read_text(encoding='utf-8')
    assert 'drocat-legend-tree' in html
    assert 'r1_SLP459_x1' in html
    assert '459_SLP459_R' in html
    # no direct-row flattening: the singleton group renders like any group
    assert 'attachDirectRow' not in html


def test_tree_injection_is_idempotent(tmp_path):
    visualizer = _make_visualizer()
    page = tmp_path / 'idempotent.html'
    visualizer._write_plotly_html(
        _tagged_figure(), str(page), legend_tree=True
    )
    before = page.read_text(encoding='utf-8')
    visualizer._inject_page_extras(str(page), legend_tree=True)
    assert page.read_text(encoding='utf-8') == before


def test_tree_sort_only_when_sortkey_present():
    """The legend panel re-orders a leaf container only when its leaves
    carry an explicit sortKey (validation expansion bins). Other
    custom-group scenes keep insertion order."""
    html = _make_visualizer()._legend_tree_html()
    # the client sorts inside a truthiness guard on the stamped key
    assert 'if (lg.sortKey) {' in html
    assert 'if (!tt.sortKeys || !Object.keys(tt.sortKeys).length) { return; }' \
        in html


# ---------------------------------------------------------------------------
# Boot behaviour, executed: the EMITTED tree script run under node against a
# stub DOM.
#
# The panel is built client-side, and the injected stylesheet hides the native
# plotly legend on these pages *unconditionally*. So a boot that throws
# anywhere before ``panel.style.display = 'block'`` leaves the page with no
# legend of any kind -- which is exactly how the 2026-10-12 12:30 export
# shipped. String assertions cannot see that class of failure, so these tests
# execute the emitted script verbatim (never a hand-written copy) against just
# enough DOM for the panel to build itself.
# ---------------------------------------------------------------------------

_NODE_STUB = r"""
'use strict';
function El(tag) {
  this.tagName = String(tag).toUpperCase();
  this.id = '';
  this.children = [];
  this.parentNode = null;
  this._classes = [];
  this._text = '';
  this.style = {};
  this.attributes = {};
  this.handlers = {};
  this.title = '';
  this.scrollTop = 0;
  this.scrollLeft = 0;
  this.scrollWidth = 0;
  this.clientWidth = 0;
  this.scrollHeight = 0;
  this.clientHeight = 0;
  var self = this;
  this.classList = {
    add: function (c) { if (self._classes.indexOf(c) < 0) { self._classes.push(c); } },
    remove: function (c) {
      var k = self._classes.indexOf(c);
      if (k >= 0) { self._classes.splice(k, 1); }
    },
    contains: function (c) { return self._classes.indexOf(c) >= 0; },
    toggle: function (c, force) {
      var want = (force === undefined) ? !self.classList.contains(c) : !!force;
      if (want) { self.classList.add(c); } else { self.classList.remove(c); }
      return want;
    }
  };
}
Object.defineProperty(El.prototype, 'className', {
  get: function () { return this._classes.join(' '); },
  set: function (v) {
    this._classes = String(v).split(/\s+/).filter(function (c) { return c; });
  }
});
Object.defineProperty(El.prototype, 'textContent', {
  get: function () {
    if (!this.children.length) { return this._text; }
    return this.children.map(function (c) { return c.textContent; }).join('');
  },
  set: function (v) { this._text = String(v); this.children = []; }
});
Object.defineProperty(El.prototype, 'innerHTML', {
  get: function () { return ''; },
  set: function () { this._text = ''; this.children = []; }
});
El.prototype.appendChild = function (child) {
  child.parentNode = this;
  this.children.push(child);
  return child;
};
El.prototype.removeChild = function (child) {
  var k = this.children.indexOf(child);
  if (k >= 0) { this.children.splice(k, 1); }
  child.parentNode = null;
  return child;
};
El.prototype.setAttribute = function (k, v) {
  this.attributes[k] = String(v);
  if (k === 'id') { this.id = String(v); }
};
El.prototype.getAttribute = function (k) {
  return Object.prototype.hasOwnProperty.call(this.attributes, k)
    ? this.attributes[k] : null;
};
El.prototype.addEventListener = function (type, fn) {
  (this.handlers[type] = this.handlers[type] || []).push(fn);
};
El.prototype.removeEventListener = function () {};
El.prototype.getBoundingClientRect = function () {
  return {top: 0, left: 0, right: 100, bottom: 0, width: 100, height: 10};
};
El.prototype._descendants = function (out) {
  out = out || [];
  this.children.forEach(function (c) { out.push(c); c._descendants(out); });
  return out;
};
El.prototype.querySelectorAll = function (selector) {
  var want = String(selector).split('.').filter(function (c) { return c; });
  return this._descendants().filter(function (el) {
    return want.every(function (c) { return el.classList.contains(c); });
  });
};
El.prototype.querySelector = function (selector) {
  var hits = this.querySelectorAll(selector);
  return hits.length ? hits[0] : null;
};
function findByClass(root, cls) {
  return root._descendants().filter(function (el) {
    return el.classList.contains(cls);
  });
}
function fire(el, type, evt) {
  var handlers = (el.handlers[type] || []).slice();
  if (!handlers.length) { throw new Error('no handler: ' + type); }
  handlers.forEach(function (fn) {
    fn(evt || {stopPropagation: function () {}});
  });
  return handlers.length;
}
function rows() { return findByClass(panel, 'drocat-lt-row'); }
function rowLabel(row) {
  var l = findByClass(row, 'drocat-lt-label')[0];
  return l ? l.textContent : '';
}
function rowGlyph(row) {
  var e = findByClass(row, 'drocat-lt-eye')[0];
  return e ? e.textContent : '';
}
function rowOff(row) { return row.classList.contains('drocat-lt-off'); }
function rowState() {
  return rows().map(function (r) {
    return {label: rowLabel(r), glyph: rowGlyph(r), off: rowOff(r)};
  });
}
function eyes() { return findByClass(panel, 'drocat-lt-eye'); }
function rowByLabel(label) {
  var hits = rows().filter(function (r) { return rowLabel(r) === label; });
  return hits.length ? hits[0] : null;
}
function rowEye(row) { return findByClass(row, 'drocat-lt-eye')[0]; }
function meshRowState() {
  return rows().filter(function (r) {
    return r.classList.contains('drocat-lt-mesh-row');
  }).map(function (r) {
    return {label: rowLabel(r), glyph: rowGlyph(r), off: rowOff(r)};
  });
}
function toggle(label) {
  var row = rowByLabel(label);
  if (!row) { throw new Error('no row labelled ' + label); }
  fire(rowEye(row), 'click');
  return {label: label, glyph: rowGlyph(row), off: rowOff(row)};
}
function toggleHeader() {
  fire(eyes()[0], 'click');
  return true;
}

var panel = new El('div');
panel.id = 'drocat-legend-tree';
panel.style.display = 'none';
var DOCUMENT_ELEMENT = new El('html');
var HEAD = new El('head');
var BODY = new El('body');
DOCUMENT_ELEMENT.appendChild(HEAD);
DOCUMENT_ELEMENT.appendChild(BODY);
var PANEL_IN_PAGE = __PANEL__;
var ROOTS = PANEL_IN_PAGE ? [panel, DOCUMENT_ELEMENT] : [DOCUMENT_ELEMENT];
var TIMERS = [];
var LOGS = [];
var RESTYLE_CALLS = [];
var GD = null;
function flushTimers(limit) {
  var rounds = 0;
  while (TIMERS.length && rounds < (limit || 500)) {
    var queued = TIMERS;
    TIMERS = [];
    queued.forEach(function (fn) { fn(); });
    rounds++;
  }
  return rounds;
}
var document = {
  head: HEAD,
  body: BODY,
  documentElement: DOCUMENT_ELEMENT,
  createElement: function (tag) { return new El(tag); },
  getElementById: function (id) {
    for (var i = 0; i < ROOTS.length; i++) {
      var walk = ROOTS[i]._descendants();
      walk.unshift(ROOTS[i]);
      for (var j = 0; j < walk.length; j++) {
        if (walk[j].id === id) { return walk[j]; }
      }
    }
    return null;
  },
  querySelector: function (selector) {
    if (selector === '.js-plotly-plot' || selector === '.plotly-graph-div') {
      return GD;
    }
    for (var i = 0; i < ROOTS.length; i++) {
      var hits = ROOTS[i].querySelectorAll(selector);
      if (hits.length) { return hits[0]; }
    }
    return null;
  }
};
var navigator = {webdriver: false};
var window = {
  Plotly: {
    /* Real plotly writes the update into the listed traces and then emits
       plotly_restyle, which is what re-syncs the panel glyphs. */
    restyle: function (gd, update, indices) {
      RESTYLE_CALLS.push({update: update, indices: (indices || []).slice()});
      (indices || []).forEach(function (i) {
        var tr = gd.data[i];
        if (!tr) { return; }
        Object.keys(update).forEach(function (k) { tr[k] = update[k]; });
      });
      (gd._listeners || []).forEach(function (fn) { fn(); });
      return Promise.resolve(gd);
    }
  },
  addEventListener: function () {},
  removeEventListener: function () {}
};
/* plotly's UMD build publishes window.Plotly as a bare global too, which is
   how the tree's own calls reach it. */
var Plotly = window.Plotly;
function setTimeout(fn) { TIMERS.push(fn); return TIMERS.length; }
function clearTimeout() {}
var console = {
  log: function (s) { process.stdout.write(String(s) + '\n'); },
  info: function () { LOGS.push('info ' + Array.prototype.join.call(arguments, ' ')); },
  warn: function () { LOGS.push('warn ' + Array.prototype.join.call(arguments, ' ')); },
  error: function () { LOGS.push('error ' + Array.prototype.join.call(arguments, ' ')); }
};
GD = {
  data: __TRACES__,
  _fullLayout: {scene: {camera: {eye: {x: 1, y: 1, z: 1},
                                  center: {x: 0, y: 0, z: 0},
                                  up: {x: 0, y: 0, z: 1}}}},
  _listeners: [],
  on: function (evt, fn) { GD._listeners.push(fn); }
};
"""

TreeRun = namedtuple('TreeRun', 'value logs restyle code stderr')


def _page_traces(fig):
    """The trace objects a browser holds in ``gd.data`` for this figure."""
    return json.loads(json.dumps(fig.to_plotly_json()['data']))


def _emitted_tree_script(fig):
    """The tree <script> body exactly as it is written into the page."""
    html = _make_visualizer(fig=fig)._legend_tree_html()
    start = html.index('<script>') + len('<script>')
    return html[start:html.index('</script>', start)]


def run_tree_js(expression, fig=None, *, script=None, prelude='',
                tree_panel=True):
    """Evaluate ``expression`` in node after the emitted tree script boots.

    ``script`` overrides the emitted body so a test can break the page the way
    the field broke it; ``prelude`` runs *before* the script, for globals the
    break needs (a countdown, a missing API). ``tree_panel`` toggles whether
    the page carries the panel div at all.
    """
    if shutil.which('node') is None:
        pytest.skip('node is not installed')
    fig = _tagged_figure() if fig is None else fig
    body = (_emitted_tree_script(fig) if script is None else script)
    stub = (_NODE_STUB
            .replace('__TRACES__', json.dumps(_page_traces(fig)))
            .replace('__PANEL__', 'true' if tree_panel else 'false'))
    program = (
        '(function(){\n'
        + stub
        + '\n' + prelude
        + '\n' + body
        + '\nconsole.log(JSON.stringify({value: (' + expression + '), '
        'logs: LOGS, restyle: RESTYLE_CALLS}));\n'
        '})();'
    )
    out = subprocess.run(['node', '-e', program], capture_output=True,
                         text=True, timeout=120)
    if out.returncode != 0:
        # An escaped throw is a finding in itself; hand the stack to the test.
        return TreeRun(None, [], [], out.returncode, out.stderr)
    payload = json.loads(out.stdout)
    return TreeRun(payload['value'], payload['logs'], payload['restyle'],
                   out.returncode, out.stderr)


def _isDrawn_scope_stripped(script):
    """Re-create the 12:30 defect: a helper that reads ``gd`` without binding
    it, which is what a mid-edit bake produced on the user's page."""
    broken = script.replace(
        '  function isDrawn(i) {\n    var gd = graphDiv();\n',
        '  function isDrawn(i) {\n',
        1,
    )
    assert broken != script, (
        'the boot breaker did not bite: isDrawn no longer binds graphDiv(), '
        'so update _isDrawn_scope_stripped() to break the current script'
    )
    return broken


def _graph_div_flaky(script, fails_first):
    """Make the first ``fails_first`` polls see no plotly div at all."""
    healed = script.replace(
        '  function graphDiv() {\n    return document.querySelector('
        "'.js-plotly-plot')",
        '  function graphDiv() {\n    if (FLAKY-- > 0) { return null; }\n'
        '    return document.querySelector(\'.js-plotly-plot\')',
        1,
    )
    assert healed != script, (
        'the flaky-graphDiv breaker did not bite: update '
        '_graph_div_flaky() to match the current script'
    )
    return healed


def _render_oneshot_broken(script):
    """Throw exactly once inside ``render()``, so the next poll can succeed."""
    broken = script.replace(
        '  function makeEl(tag, className, text) {\n'
        '    var el = document.createElement(tag);\n',
        '  function makeEl(tag, className, text) {\n'
        '    if (ONESHOT-- > 0) { throw new Error(\'stubbed render failure\'); }\n'
        '    var el = document.createElement(tag);\n',
        1,
    )
    assert broken != script, (
        'the one-shot render breaker did not bite: update '
        '_render_oneshot_broken() to match the current script'
    )
    return broken


def _envelope_figure():
    """A BANC-shaped page: one neuron plus both context envelopes.

    The VNC half is embedded but shipped transparent (``visible: true`` with
    ``opacity: 0``) -- that is how a frozen view hides an envelope without
    letting the scene renormalise around it. The row glyph for that half is
    what ``sync()`` must read off the opacity, not off ``visible``.
    """
    fig = go.Figure()
    neuron = go.Scatter3d(
        x=[0, 1], y=[0, 1], z=[0, 1], mode='lines',
        line=dict(color='#1f77b4', width=4),
        name='l-LNv', legendgroup='l-LNv', showlegend=True,
    )
    neuron.legendrank = 0
    neuron.meta = {'drocatLegend': {
        'kind': 'neuron', 'group': 'l-LNv', 'item': 'l-LNv',
        'color': '#1f77b4',
    }}
    fig.add_trace(neuron)
    for rank, name, opacity in (
        (visualize_skeleton.BRAIN_MESH_LEGEND_RANK, 'BANC (brain)', 0.04),
        (visualize_skeleton.VNC_MESH_LEGEND_RANK, 'BANC (VNC)', 0.0),
    ):
        mesh = go.Mesh3d(
            x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0], j=[1], k=[2],
            color='#c8e6f0', opacity=opacity, name=name, showlegend=True,
        )
        mesh.legendrank = rank
        fig.add_trace(mesh)
    return fig


class TestTreeBoot:
    def test_boot_builds_the_panel_and_shows_it(self):
        run = run_tree_js('({display: panel.style.display, '
                          'rows: rowState().length})')
        assert run.code == 0, run.stderr
        assert run.value['display'] == 'block'
        assert run.value['rows'] > 0

    def test_boot_without_plotly_never_arriving_still_leaves_a_legend(self):
        """The poll gives up after ~60 s; giving up silently is the same
        total-legend-loss failure as throwing."""
        script = _graph_div_flaky(_emitted_tree_script(_tagged_figure()), 1000)
        run = run_tree_js(
            '(function(){ flushTimers(1000); return {'
            'display: panel.style.display, '
            'fallback: !!document.getElementById("drocat-legend-fallback"), '
            'error: window.__drocatLegendTreeError || null}; })()',
            fig=_tagged_figure(), script=script, prelude='var FLAKY = 1000;')
        assert run.code == 0, run.stderr
        assert run.value['fallback'] is True
        assert run.value['error']

    def test_broken_boot_restores_the_native_legend(self):
        fig = _tagged_figure()
        broken = _isDrawn_scope_stripped(_emitted_tree_script(fig))
        run = run_tree_js(
            '({display: panel.style.display, '
            'fallback: !!document.getElementById("drocat-legend-fallback"), '
            'fallbackRule: (document.getElementById("drocat-legend-fallback")'
            ' || {textContent: ""}).textContent, '
            'fallbackInBody: BODY.children.indexOf('
            'document.getElementById("drocat-legend-fallback")) >= 0, '
            'error: window.__drocatLegendTreeError || null})',
            fig=fig, script=broken)
        assert run.code == 0, (
            'the boot throw escaped the page: ' + run.stderr)
        assert run.value['display'] == 'none', 'a broken panel stays hidden'
        assert run.value['fallback'] is True, (
            'nothing re-enabled the native legend')
        assert '.js-plotly-plot .legend{display:block !important' \
            in run.value['fallbackRule']
        # the page's own hiding rule ships in the body stylesheet; an override
        # placed before it in document order simply loses.
        assert run.value['fallbackInBody'] is True
        assert run.value['error']
        assert any('[drocat]' in line and 'error' in line for line in run.logs)

    def test_boot_that_recovers_on_a_later_poll_clears_the_fallback(self):
        """A one-shot render failure must not keep the page in fallback."""
        fig = _tagged_figure()
        broken = _render_oneshot_broken(_emitted_tree_script(fig))
        run = run_tree_js(
            '(function(){ flushTimers(10); return {'
            'display: panel.style.display, '
            'rows: rowState().length, '
            'fallback: !!document.getElementById("drocat-legend-fallback")};'
            ' })()',
            fig=fig, script=broken, prelude='var ONESHOT = 1;')
        assert run.code == 0, run.stderr
        assert run.value['display'] == 'block'
        assert run.value['rows'] > 0
        assert run.value['fallback'] is False
        assert any('[drocat]' in line for line in run.logs)


class TestTreeRowGlyphs:
    """The eye glyph must report what the scene actually draws.

    Context envelopes hide through ``opacity: 0`` while keeping
    ``visible: true``, so a row that reads ``visible`` stays solid over a mesh
    the page is not drawing -- the state a user can only discover by clicking.
    """

    def test_boot_marks_the_shipped_transparent_half_hidden(self):
        run = run_tree_js('meshRowState()', fig=_envelope_figure())
        assert run.code == 0, run.stderr
        assert run.value == [
            {'label': 'BANC (brain)', 'glyph': '●', 'off': False},
            {'label': 'BANC (VNC)', 'glyph': '○', 'off': True},
        ]

    def test_hiding_a_mesh_empties_its_own_row(self):
        run = run_tree_js(
            '(function(){'
            ' var brain = toggle("BANC (brain)");'
            ' return {brain: brain, vnc: rowGlyph(rowByLabel("BANC (VNC)")),'
            ' opacity: GD.data[1].opacity,'
            ' visible: (GD.data[1].visible === undefined)'
            ' ? "absent" : GD.data[1].visible};'
            ' })()',
            fig=_envelope_figure())
        assert run.code == 0, run.stderr
        assert run.value['brain'] == {'label': 'BANC (brain)',
                                      'glyph': '○', 'off': True}
        assert run.value['opacity'] == 0
        # toggling an envelope never touches `visible`: the frozen view holds
        # its box only when the trace stays in the scene normalization.
        assert run.value['visible'] == 'absent'
        assert run.restyle[-1] == {'update': {'opacity': 0}, 'indices': [1]}
        assert run.value['vnc'] == '○'

    def test_showing_the_transparent_half_lights_its_row(self):
        run = run_tree_js(
            '(function(){ var vnc = toggle("BANC (VNC)");'
            ' return {vnc: vnc, opacity: GD.data[2].opacity}; })()',
            fig=_envelope_figure())
        assert run.code == 0, run.stderr
        assert run.value['vnc']['glyph'] == '●'
        assert run.value['vnc']['off'] is False
        # the row's own baked show-opacity is restored, not a hardcoded 1.0
        assert run.value['opacity'] == pytest.approx(0.04)

    def test_header_eye_covers_the_transparent_half_too(self):
        run = run_tree_js(
            '(function(){'
            ' toggleHeader();'
            ' var off = {rows: meshRowState(), neuron: GD.data[0].visible,'
            ' brainOpacity: GD.data[1].opacity,'
            ' vncOpacity: GD.data[2].opacity};'
            ' toggleHeader();'
            ' return {off: off, on: meshRowState(),'
            ' onBrain: GD.data[1].opacity, onVnc: GD.data[2].opacity};'
            ' })()',
            fig=_envelope_figure())
        assert run.code == 0, run.stderr
        assert [row['glyph'] for row in run.value['off']['rows']] == ['○', '○']
        assert all(row['off'] for row in run.value['off']['rows'])
        assert run.value['off']['neuron'] is False
        assert run.value['off']['brainOpacity'] == 0
        assert run.value['off']['vncOpacity'] == 0
        # the header eye means *every row*, including the half the page opens
        # without, so a second click brings both envelopes up.
        assert [row['glyph'] for row in run.value['on']] == ['●', '●']
        assert run.value['onBrain'] == pytest.approx(0.04)
        assert run.value['onVnc'] == pytest.approx(0.04)



def _restore_js_function():
    """The WebDriver restore script, callable, so a top-level ``return`` in it
    does not end the harness program."""
    return ('function drocatRestoreLegend() {'
            + visualize_skeleton.REEXPORT_LEGEND_RESTORE_JS
            + '\n}')


_RESTORE_PROBE = (
    '(function(){ drocatRestoreLegend(); drocatRestoreLegend(); return {'
    'styles: BODY.children.length, '
    'rule: (document.getElementById("drocat-reexport-legend")'
    ' || {textContent: ""}).textContent}; })()')


class TestReexportLegendRestore:
    """Issue E: a WebDriver render of a saved tree page shows no legend.

    The panel early-returns on ``navigator.webdriver`` and the page's own
    stylesheet has hidden the native legend, so a screenshot of it comes out
    legend-less -- measured on the real session: 581 dark legend-text pixels
    with the restore, zero without, and every differing pixel inside the
    legend box. The profile exporters are not such a consumer: they ask plotly
    for no legend at all (``showlegend: False`` in the layout), which the
    stylesheet does not override.
    """

    def test_a_tree_page_gets_its_native_legend_back_once(self):
        run = run_tree_js(_RESTORE_PROBE, script=_restore_js_function())
        assert run.code == 0, run.stderr
        assert run.value['styles'] == 1, 'the restore is not idempotent'
        assert ('.js-plotly-plot .legend{display:block !important;}'
                in run.value['rule'])
        assert ('#drocat-legend-tree{display:none !important;}'
                in run.value['rule'])

    def test_a_page_without_the_panel_is_left_alone(self):
        run = run_tree_js(_RESTORE_PROBE, script=_restore_js_function(),
                          tree_panel=False)
        assert run.code == 0, run.stderr
        assert run.value['styles'] == 0
        assert run.value['rule'] == ''


def _specificity(selector):
    """CSS's (ids, classes/attributes/pseudo-classes, elements) triple."""
    ids = classes = elements = 0
    for part in re.split(r'[\s>+~]+', selector.strip()):
        if not part:
            continue
        ids += part.count('#')
        classes += part.count('.') + part.count('[') + part.count(':')
        if re.match(r'^[a-zA-Z]', part):
            elements += 1
    return (ids, classes, elements)


def _block_override_selectors(source):
    """Selectors in ``source`` that force ``display:block !important``."""
    return [selector.strip()
            for selector, declaration
            in re.findall(r"([^'\"{}\n]+)\{([^}\n]*)\}", source)
            if 'display:block !important' in declaration]


class TestLegendOverrideWinsTheCascade:
    """An override of the panel's hiding rule has to win on arithmetic.

    The hide ships as ``!important`` inside the *body* stylesheet, so an equal
    specificity override in the head loses whichever way the cascade is read.
    That is how the first break-glass fallback failed in a live browser: the
    rule was in the DOM, the legend stayed hidden, and every node assertion
    still passed. These tests compare the selectors the page really emits.
    """

    HIDE_SELECTOR = '.js-plotly-plot .legend'

    def test_the_tree_page_does_hide_the_native_legend(self):
        html = _make_visualizer()._legend_tree_html()
        assert self.HIDE_SELECTOR + '{display:none !important;}' in html
        # ... and that stylesheet is in the body, after the head.
        assert html.index('<style>') > 0

    def test_every_legend_reveal_beats_the_rule_it_has_to_override(self):
        sources = {
            'boot fallback': _make_visualizer()._legend_tree_html(),
            're-export restore': visualize_skeleton.REEXPORT_LEGEND_RESTORE_JS,
        }
        for label, source in sources.items():
            overrides = _block_override_selectors(source)
            assert overrides, 'no display:block override in ' + label
            for selector in overrides:
                assert self.HIDE_SELECTOR in selector, selector
                assert _specificity(selector) > _specificity(
                    self.HIDE_SELECTOR), (
                    '%s: %r only ties %r, so the body stylesheet wins'
                    % (label, selector, self.HIDE_SELECTOR))
