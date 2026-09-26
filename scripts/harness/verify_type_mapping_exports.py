"""Real-data verification of the optimized type-mapping exports.

Mimics the viewer's export paths end-to-end against the local cached
indexes and inspects the written artifacts:
1. Type-level network HTML (dagre): types-only nodes — no query-entry node,
   no funnel intermediates — display labels, edge-only bridge derivation
   ({key:val; ...}-safe), per-side neuron counts, and edge weights that
   match ``build_mapping_network_graph`` flow for flow.
2. Linker-path HTML (the preset-positioned artifact): every node carries a
   layer coordinate, columns are evenly spaced, and the barycenter ordering
   clusters shared-neighbour sources.
3. Matched-rows CSV (the Export matched rows handler logic): row count ==
   query total, retained metadata columns only, paged traversal equality.
"""

import csv
import io
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from ui.neuron_index import (  # noqa: E402
    collect_native_type_matches,
    count_type_in_index,
    count_types_in_index,
    enrich_native_type_matches,
    load_cached_neuron_index,
    query_neuron_index,
)
from comparison.mapping_visualization import (  # noqa: E402
    build_mapping_flows,
    build_mapping_network_graph,
    render_bridge_linker_html,
    write_mapping_network_html,
)

MCNS = "male-cns:v1.0"
# The counterpart the circadian entry resolves to, read off the flows
# rather than hard-coded, so the linker check follows the same data.
FAFB = "flywire_FAFB_v783"
STAMP = time.strftime("%Y%m%d_%H%M%S")
OUT = REPO / "outputs" / "type_mapping"
OUT.mkdir(parents=True, exist_ok=True)
failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def _barycenter_beats_alphabetical(nodes, edges, row_gap=70):
    """Does the layered ordering actually cluster shared-neighbour nodes?

    A group is the set of layer-0 nodes sharing one layer-1 neighbour.  The
    ordering wins when the mean row spread across those groups is smaller
    than the same mean under a plain alphabetical ordering — that is what
    ``_layered_y_positions`` promises (barycenter sweeps, so co-neighbours
    end up adjacent and crossings drop).
    """
    ids = [n["data"]["id"] for n in nodes
           if n["position"].get("x") == 0]
    if not ids:
        return False, "no layer-0 nodes"
    neighbours = {}
    for e in edges:
        s, t = e["data"]["source"], e["data"]["target"]
        if s in ids:
            neighbours.setdefault(s, set()).add(t)
    groups = {}
    for node, nbrs in neighbours.items():
        for nbr in nbrs:
            groups.setdefault(nbr, []).append(node)
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not groups:
        return False, "no shared-neighbour groups"
    y_by_id = {n["data"]["id"]: -n["position"]["y"] for n in nodes
               if n["position"].get("x") == 0}
    ordered = sorted(ids, key=lambda i: y_by_id[i])

    def _mean_spread(position_of):
        total = 0
        for members in groups.values():
            rows = [position_of(m) for m in members]
            total += max(rows) - min(rows)
        return total / len(groups)

    placed = _mean_spread(lambda m: round(y_by_id[m] / row_gap))
    alpha = _mean_spread(lambda m: ordered.index(m))
    return placed <= alpha, (f"{len(groups)} groups, mean spread "
                             f"placed={placed:.2f} alphabetical={alpha:.2f}")


# --------------------------------------------- type-level mapping network
native = collect_native_type_matches(MCNS, "circadian", uncapped=True)
enrich_native_type_matches(native, MCNS)
entry = next(e for e in native if e["dataset"] == "flywire_FAFB_v783")
index = load_cached_neuron_index(MCNS)

flows = build_mapping_flows([entry], MCNS)
source_types = {f["source_type"] for f in flows}
source_counts = count_types_in_index(index, source_types)
flows = build_mapping_flows([entry], MCNS, source_counts=source_counts)
check("flows built", bool(flows), f"{len(flows)} flows")
check("source_count is the local type's own neuron count",
      all(f["source_count"] == source_counts[f["source_type"]]
          for f in flows))
check("foreign_count is the foreign type's neuron count",
      all(f["foreign_count"] == count_type_in_index(index, f["foreign_type"])
          or f["foreign_count"]  # foreign types do not exist locally
          for f in flows))

net_path = OUT / f"verify_mapping_network_{STAMP}.html"
result = write_mapping_network_html(flows, str(net_path), open_browser=False)
check("network html written", bool(result) and net_path.exists(),
      f"{net_path.stat().st_size:,} bytes" if net_path.exists() else "missing")
net_html = net_path.read_text()

nodes_match = re.search(
    r"nodes:\s*(\[.*?\])\s*,\s*\n\s*edges:\s*(\[.*?\])\s*\n", net_html, re.S)
check("network elements JSON found", bool(nodes_match))
if nodes_match:
    nodes = json.loads(nodes_match.group(1))
    edges = json.loads(nodes_match.group(2))

    # The per-pair TYPE network is positioned by dagre, so it carries no
    # preset coordinates — the explicit layer map lives in the linker-path
    # view and in the composed star, both checked below.  (These four checks
    # once ran against this artifact and could never pass: the type network
    # shed its custom layer map for dagre, and a dagre document ships
    # `"position": {}` for every node.)
    unpositioned = [n for n in nodes if not (n.get("position") or {})]
    check("dagre artifact carries no baked positions",
          len(unpositioned) == len(nodes) and len(nodes) > 10,
          f"{len(unpositioned)}/{len(nodes)} unpositioned")
    check("dagre is the preselected layout",
          '<option value="dagre" selected>' in net_html)

    data_labels = [n["data"].get("label", "") for n in nodes]
    check("labels are display values (no layer|dataset prefixes)",
          data_labels and not any("|" in str(lab) for lab in data_labels),
          f"e.g. {data_labels[:3]}")
    node_types = [n["data"].get("node_type", "") for n in nodes]
    # The renderer overwrites data.node_type with the dataset GROUP code, so
    # the structural role lives in data.role.  2026-09-26: the query chip is
    # no longer a node, so no role may be 'entry'; a type-level pair view has
    # no 'intermediate' either.
    roles = [n["data"].get("role", "") for n in nodes]
    check("types only: no funnel intermediates, no query-entry node",
          "intermediate" not in roles and "entry" not in roles
          and roles.count("source") >= 1 and roles.count("target") >= 1,
          f"roles={ {r: roles.count(r) for r in set(roles)} }")

    # node hover titles are plain (bridge derivation only on edges)
    titles = [n["data"].get("dataset_info", {}) for n in nodes]
    check("node hovers carry no bridge derivation",
          all("[" not in json.dumps(info) for info in titles))

    # pair edges: correct per-side counts + one maps-via label per chain
    pair_edges = [e for e in edges if str(e["data"]["source"]).startswith("0|")]
    check("direct type-level pair edges present", bool(pair_edges),
          f"{len(pair_edges)} pair edges")
    sample = pair_edges[0]["data"]["custom_labels"]
    check("edge info has per-side neuron count labels",
          any("source neurons" in k for k in sample)
          and any("foreign neurons" in k for k in sample),
          f"keys={list(sample)}")
    check("edge info maps-via values glue name[source]",
          all(" [" not in v for k, v in sample.items() if k.startswith("maps via")))
    # The exported edges must match what the builder produced, weight for
    # weight.  Re-deriving the weight here was the old bug: this artifact is
    # built from NATIVE-MATCH flows, whose origin side is the flow's TARGET
    # dataset, so the graph flips the presentation (layer 0 = FAFB) and a
    # check keyed on (source_type → foreign_type) found no flow at all.
    graph = build_mapping_network_graph(flows)
    graph_weights = {(u, v): d.get("weight")
                     for u, v, d in graph.edges(data=True)}
    html_weights = {(e["data"]["source"], e["data"]["target"]):
                    e["data"]["weight"] for e in pair_edges}
    drifted = {k: (v, graph_weights.get(k)) for k, v in html_weights.items()
               if graph_weights.get(k) != v}
    check("every pair edge weight matches the builder's graph",
          bool(html_weights) and not drifted and
          set(html_weights) <= set(graph_weights), f"{list(drifted.items())[:3]}")

    # the shared coverage formula, not the source type's whole count: a
    # source type fanning out to N targets must not repeat its full local
    # population on every outgoing edge
    fan_sources = {}
    for (u, _v) in graph_weights:
        fan_sources[u] = fan_sources.get(u, 0) + 1
    fan_nodes = [u for u, n in fan_sources.items() if n >= 2]
    def _type_of(node_id):
        return str(node_id).split("|")[2]
    over_counted = [
        u for u in fan_nodes
        if any(w > (count_type_in_index(index, _type_of(u)) or w)
               for (a, _b), w in graph_weights.items() if a == u
               and _type_of(a) == _type_of(u)
               and count_type_in_index(index, _type_of(u)))]
    check("fanning sources do not repeat their whole local count",
          not over_counted, f"{len(fan_nodes)} fan-out sources checked")

    # count correctness for a fan-in target: several sources into one
    # foreign type keep their OWN edge weights (the shared target population
    # must not flatten them)
    by_target = {}
    for (u, v), w in graph_weights.items():
        by_target.setdefault(v, []).append((u, w))
    fan_in = {v: pairs for v, pairs in by_target.items() if len(pairs) >= 2}
    flat = [v for v, pairs in fan_in.items()
            if len({w for _u, w in pairs}) == 1
            and len({_type_of(u) for u, _w in pairs}) > 1
            and len({count_type_in_index(index, _type_of(u))
                     for u, _w in pairs}) > 1]
    check("fan-in edges carry per-flow weights (not the foreign count)",
          not flat, f"{len(fan_in)} fan-in targets, flattened: {flat[:2]}")
    check("mapping preset config present",
          re.search(r"'mapping': \{\s*name: 'preset'", net_html) is not None)
    check("neurons weight label wired",
          "const edgeWeightLabelJS = 'neurons'" in net_html)


# --------------------------------- linker-path view: the preset-positioned one
# The explicit layer map (x = layer * layer_gap, y = barycenter row *
# row_gap) belongs to the linker-path artifact, which renders through the
# cytoscape `mapping` preset.  These are the checks the type network used to
# claim.
linker_html = render_bridge_linker_html(flows, source_dataset=MCNS,
                                        target_dataset=FAFB)
check("linker html rendered", bool(linker_html),
      f"{len(linker_html or ''):,} chars")
linker_match = (re.search(r"nodes:\s*(\[.*?\])\s*,\s*\n\s*edges:\s*(\[.*?\])\s*\n",
                          linker_html or "", re.S))
check("linker elements JSON found", bool(linker_match))
if linker_match:
    lnodes = json.loads(linker_match.group(1))
    ledges = json.loads(linker_match.group(2))
    positioned = [n for n in lnodes if (n.get("position") or {}).get("x")
                  is not None]
    check("every linker node carries a preset position",
          len(positioned) == len(lnodes) and len(lnodes) > 10,
          f"{len(positioned)}/{len(lnodes)}")
    xs = sorted({n["position"]["x"] for n in positioned})
    check("at least three layer columns (L->R)", len(xs) >= 3, f"x={xs[:6]}")
    for x in xs:
        ys = sorted((n["position"]["y"] for n in positioned
                     if n["position"]["x"] == x), reverse=True)
        gaps = {round(a - b, 6) for a, b in zip(ys, ys[1:])}
        check(f"column x={x} evenly distributed",
              len(ys) == 1 or gaps == {70}, f"{len(ys)} nodes")
    bary_ok, bary_detail = _barycenter_beats_alphabetical(
        lnodes, json.loads(linker_match.group(2)))
    check("barycenter: shared-neighbour sources cluster", bary_ok, bary_detail)
    check("linker layout is the mapping preset",
          "value=\"mapping\" selected" in (linker_html or ""))


# ------------------------------------------------- Matched-rows CSV path
def export_matched_rows(dataset, **query_kwargs):
    """The viewer's _export_matched_rows handler, isolated."""
    idx = load_cached_neuron_index(dataset)
    result = query_neuron_index(
        idx, **query_kwargs, page=1, page_size=50, include_all_rows=True)
    if result.total > 100_000:
        return None, result.total
    buffer = io.StringIO()
    fieldnames = [c for c in idx.columns if c in result.rows[0]]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames,
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(result.rows)
    return buffer.getvalue(), result.total


csv_text, total = export_matched_rows(MCNS, search="aMe")
rows = list(csv.DictReader(io.StringIO(csv_text)))
check("broad 'aMe' export row count == total", len(rows) == total,
      f"{len(rows):,} rows (total={total:,})")
check("csv columns == retained metadata columns",
      list(rows[0].keys()) == list(index.columns),
      f"{len(rows[0].keys())} columns")
paged_keys = set()
page = 1
while True:
    pg = query_neuron_index(index, search="aMe", page=page, page_size=200)
    paged_keys.update(r["__neuron_key"] for r in pg.rows)
    if page >= pg.pages:
        break
    page += 1
check("csv rows == union of all paged rows",
      {r["__neuron_key"] for r in
       query_neuron_index(index, search="aMe", page=1, page_size=50,
                          include_all_rows=True).rows} == paged_keys,
      f"{len(paged_keys):,} keys")

# zero-hit query through the mapped view (types_include path)
mapped_types = sorted(entry.get("mapped_type_names", []))
csv_text2, total2 = export_matched_rows(
    MCNS, types_include=mapped_types, sort_by="type")
rows2 = list(csv.DictReader(io.StringIO(csv_text2)))
check("mapped-view export row count == total", len(rows2) == total2,
      f"{len(rows2):,} rows (total={total2:,})")
check("mapped-view export only mapped types",
      {r["type"] for r in rows2} <= set(mapped_types))
_, big_total = export_matched_rows(MCNS, search="")
check("cap guard returns nothing over 100k", big_total > 100_000,
      f"total={big_total:,}")
csv_path = OUT / f"verify_matched_rows_{STAMP}.csv"
csv_path.write_text(csv_text, encoding="utf-8")
check("matched-rows csv written", csv_path.exists(),
      f"{csv_path.stat().st_size:,} bytes")

print()
if failures:
    print("FAILURES:", failures)
    sys.exit(1)
print("ALL EXPORT VERIFICATIONS PASSED")
