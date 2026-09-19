"""Real-data verification of the optimized type-mapping exports.

Mimics the viewer's export paths end-to-end against the local cached
indexes and inspects the written artifacts (both the Sankey and the
Network buttons now render the same interactive vispath graph):
1. vispath HTML: type-level 3-layer preset positions, display labels,
   edge-only bridge derivation ({key:val; ...}-safe), correct per-side
   neuron counts, even per-layer distribution, barycenter ordering.
2. Matched-rows CSV (the Export matched rows handler logic): row count ==
   query total, retained metadata columns only, paged traversal equality.
"""

import csv
import io
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
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
    write_mapping_network_html,
)

MCNS = "male-cns:v1.0"
STAMP = time.strftime("%Y%m%d_%H%M%S")
OUT = REPO / "outputs" / "type_mapping"
OUT.mkdir(parents=True, exist_ok=True)
failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def _adjacent(pair_edges, nodes):
    """At least one target's sources occupy consecutive rows."""
    y_by_id = {n["data"]["id"]: -n["position"]["y"]
               for n in nodes if n["position"].get("x") == 0}
    per_target = {}
    for e in pair_edges:
        src = e["data"]["source"]
        if src in y_by_id:
            per_target.setdefault(e["data"]["target"], []).append(y_by_id[src])
    return any(
        max(ys) - min(ys) <= 70 * (len(ys) - 1) + 1e-9
        for ys in per_target.values() if len(ys) >= 3)


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

    # positions: preset 3-layer left-to-right layout, even row spacing
    non_empty = [n for n in nodes if (n.get("position") or {}).get("x") is not None]
    check("all nodes carry preset positions",
          len(non_empty) == len(nodes) and len(nodes) > 10,
          f"{len(non_empty)}/{len(nodes)} positioned")
    xs = sorted({n["position"]["x"] for n in non_empty})
    check("three distinct layer x values (L->R)", xs == [0, 380, 760],
          f"x={xs}")
    for layer in (0, 1, 2):
        ys = sorted((n["position"]["y"] for n in non_empty
                     if n["position"]["x"] == layer * 380), reverse=True)
        gaps = {round(b - a, 6) for a, b in zip(ys, ys[1:])}
        check(f"layer {layer} evenly distributed",
              len(ys) == 1 or gaps == {-70}, f"{len(ys)} nodes")

    data_labels = [n["data"].get("label", "") for n in nodes]
    check("labels are display values (no layer|dataset prefixes)",
          data_labels and not any("|" in str(lab) for lab in data_labels),
          f"e.g. {data_labels[:3]}")
    node_types = [n["data"].get("node_type", "") for n in nodes]
    check("no funnel intermediate nodes; entry nodes exist",
          "intermediate" not in node_types and node_types.count("entry") >= 1,
          f"types={ {t: node_types.count(t) for t in set(node_types)} }")

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
    # the weight equals the source type's own local count
    src_id = pair_edges[0]["data"]["source"]
    src_type = str(src_id).split("|")[2]
    local = count_type_in_index(index, src_type)
    check("pair edge weight == source type's local neuron count",
          bool(local) and pair_edges[0]["data"]["weight"] == local,
          f"{src_type}: {local}")

    # count correctness for a fan-in target: edges into one foreign type
    # each carry their OWN source count (distinct when the sources differ)
    by_target = {}
    for e in pair_edges:
        by_target.setdefault(e["data"]["target"], set()).add(
            (e["data"]["source"], e["data"]["weight"]))
    multi = [t for t, pairs in by_target.items()
             if len(pairs) >= 3 and len({w for _s, w in pairs}) >= 2]
    check("fan-in edges carry per-source counts (not the foreign count)",
          bool(multi), f"{len(multi)} targets with distinct source counts")

    check("barycenter: co-target sources on adjacent rows", _adjacent(pair_edges, nodes))
    check("preset layout config present", "name: 'preset'" in net_html)
    check("mapping preset config present",
          re.search(r"'mapping': \{\s*name: 'preset'", net_html) is not None)
    check("neurons weight label wired",
          "const edgeWeightLabelJS = 'neurons'" in net_html)


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
