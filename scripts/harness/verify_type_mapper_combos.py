"""Cross-combination real-data verification for the auto type mapper.

Runs the type mapper over every directed dataset combination that is
locally cached (different mapping namespaces only — family versions
share one registry key by design), and asserts the systematic
invariants on every derivation chain:

0. source-map licensing (§9I): every chain passes
   ``bridge_is_valid`` — map-licensed hops, the crosswalk
   endpoint-family rule, and the ping-pong suppression;
1. every chain ENDS in the target namespace at a real target type;
2. no two consecutive standardized linkers are identical (the
   additional_type(s)→additional_type(s) self-loop class, e.g. the
   5th-LNv regression);
3. at most 2 direct (registry) linkers per chain;
4. chain length within the walk depth cap.

Also verifies the 4-char dataset labels (version suffixes on family
collisions), the 5th-LNv linker graph (no self-loop edges), and
renders the four real mapping artifacts (sankey type/linker via the
vispath backend, dagre network, linker paths) plus the valid-bridge
source-map network — written to a temp dir for inspection, never into
the repo.

Exits 1 when any invariant fails.
"""

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

SAMPLE_TOP = 40
SAMPLE_TAIL = 20
MAX_DIRECT_LINKERS = 2
MAX_CHAIN_NODES = 6

failures = []


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"    [FAIL] {msg}")


def sample_types(index, total: int):
    """Deterministic sample: heaviest types first, then a spread tail."""
    frame = index.frame
    if "type" not in frame.columns:
        return []
    counts = (
        frame.group_by("type").len()
        .sort(["len", "type"], descending=[True, False])
        .to_dicts()
    )
    names = [str(row["type"]) for row in counts if row["type"]]
    top = names[:SAMPLE_TOP]
    rest = names[SAMPLE_TOP:]
    if rest and SAMPLE_TAIL:
        step = max(1, len(rest) // SAMPLE_TAIL)
        tail = rest[::step][:SAMPLE_TAIL]
    else:
        tail = []
    return (top + tail)[:total]


def main() -> int:
    from comparison.cross_dataset_type_mapper import (
        CROSSWALK_COLUMNS,
        bridge_is_valid,
        crosswalk_route_licensed,
        get_type_mapper,
        standardize_bridge,
    )
    from ui.neuron_index import (
        datasets_with_cached_indexes,
        load_cached_neuron_index,
    )
    from utils.naming_utils import make_unique_dataset_labels

    datasets = datasets_with_cached_indexes()
    mapper = get_type_mapper()
    if mapper is None or not getattr(mapper, "_loaded", False):
        print("mapper not loaded — nothing to verify")
        return 1
    key_of = {ds: mapper._get_type_mapping_key(ds) for ds in datasets}
    print("datasets:", ", ".join(
        f"{ds} [{key_of[ds]}]" for ds in datasets))

    # ---- version-suffix labels (family collisions) -------------------
    labels = make_unique_dataset_labels(datasets)
    expect = {
        'male-cns:v1.0': 'MCNS_v1_0', 'male-cns:v0.9': 'MCNS_v0_9',
        'banc_v888': 'BANC_v888', 'banc_v626': 'BANC_v626',
        'flywire_FAFB_v783': 'FAFB', 'hemibrain:v1.2.1': 'HEMI',
        'manc:v1.2.3': 'MANC',
    }
    for ds in datasets:
        if ds in expect and labels[datasets.index(ds)] != expect[ds]:
            fail(f"label for {ds}: {labels[datasets.index(ds)]} "
                 f"!= {expect[ds]}")
    print("[ok] dataset labels:", ", ".join(
        f"{l}({ds})" for ds, l in zip(datasets, labels)))

    # ---- combination sweep -------------------------------------------
    pairs = [(s, t) for s in datasets for t in datasets
             if s != t and key_of[s] != key_of[t]]
    print(f"\n== combination sweep: {len(pairs)} directed namespace pairs ==")
    total_chains = 0
    per_combo = []
    for src, tgt in pairs:
        index = load_cached_neuron_index(src)
        types = sample_types(index, SAMPLE_TOP + SAMPLE_TAIL)
        chains_n = 0
        mapped = 0
        t0 = time.time()
        for type_name in types:
            bridges = mapper.get_type_bridges(type_name, src, tgt)
            if not bridges:
                continue
            mapped += 1
            target_key = key_of[tgt]
            for chain in bridges:
                total_chains += 1
                chains_n += 1
                where = f"{src}→{tgt} {type_name}"
                # 0. source-map licensing (§9I)
                if not bridge_is_valid(chain, src, tgt, key_of=key_of.get):
                    fail(f"{where}: chain fails bridge_is_valid")
                for hop in chain[1:]:
                    if (hop.get("column") in CROSSWALK_COLUMNS
                            and not crosswalk_route_licensed(
                                hop["column"], key_of[src], key_of[tgt])):
                        fail(f"{where}: crosswalk hop {hop['column']} "
                             "without endpoint-family licensing")
                # 1. chain ends in the target namespace
                if not chain or chain[-1].get("dataset") != target_key:
                    fail(f"{where}: chain end "
                         f"{chain[-1] if chain else None} not in {target_key}")
                    continue
                # 4. depth cap
                if len(chain) > MAX_CHAIN_NODES:
                    fail(f"{where}: chain too long ({len(chain)} hops)")
                # 2./3. standardized linker invariants
                linkers = [l for l in standardize_bridge(
                    chain, src, tgt) if l.get("kind") == "linker"]
                direct = [l for l in linkers if not l.get("indirect")]
                if len(direct) > MAX_DIRECT_LINKERS:
                    fail(f"{where}: {len(direct)} direct linkers > cap")
                for a, b in zip(linkers, linkers[1:]):
                    if (a["column"], a["value"]) == (b["column"], b["value"]):
                        fail(f"{where}: consecutive identical linkers "
                             f"{a['column']}·{a['value']} (self-loop class)")
        per_combo.append((src, tgt, len(types), mapped, chains_n,
                          time.time() - t0))
        print(f"  {src}→{tgt}: sampled {len(types)}, mapped {mapped}, "
              f"chains {chains_n} ({time.time() - t0:.1f}s)")
    print(f"[ok] sweep done — {total_chains} chains checked")

    # ---- 5th-LNv focus (the self-loop regression) ----------------------
    print("\n== 5th-LNv self-loop focus (MCNS→FAFB) ==")
    from comparison.mapping_visualization import (
        build_bridge_linker_graph,
        build_bridge_texts,
        build_mapping_flows,
    )

    chains = mapper.get_type_bridges("5th-LNv", "male-cns:v1.0",
                                     "flywire_FAFB_v783")
    print(f"  5th-LNv chains: {len(chains)}")
    for chain in chains:
        print("   -", " → ".join(
            f"{h['value']}[{h['column']}]" for h in chain))
    for chain in chains:
        linkers = [l for l in standardize_bridge(
            chain, "male-cns:v1.0", "flywire_FAFB_v783")
            if l.get("kind") == "linker"]
        for a, b in zip(linkers, linkers[1:]):
            if (a["column"], a["value"]) == (b["column"], b["value"]):
                fail("5th-LNv chain carries consecutive identical linkers")
    if not chains:
        fail("5th-LNv lost its MCNS→FAFB mapping")

    # real circadian flows → linker graph must have NO self-loop edges
    from ui.neuron_index import (
        collect_native_type_matches, enrich_native_type_matches,
    )
    native = collect_native_type_matches(
        "male-cns:v1.0", "5th-LNv", uncapped=True)
    enrich_native_type_matches(native, "male-cns:v1.0")
    entry = next((e for e in native if e.get("dataset")
                  == "flywire_FAFB_v783"), None)
    if entry is None:
        fail("no FAFB entry for the 5th-LNv query")
    else:
        flows = build_mapping_flows([entry], "male-cns:v1.0")
        graph = build_bridge_linker_graph(
            flows, source_dataset="male-cns:v1.0",
            target_dataset="flywire_FAFB_v783")
        loops = [(u, v) for u, v in graph.edges() if u == v]
        if loops:
            fail(f"linker graph self-loops: {loops}")
        else:
            print(f"[ok] linker graph clean — {graph.number_of_nodes()} "
                  f"nodes, {graph.number_of_edges()} edges, no self-loops")
        texts = [t for f in flows for t in build_bridge_texts(f["bridges"])]
        print(f"  {len(texts)} bridge texts; sample: {texts[:2]}")

    # ---- artifact renders for a dual-version query ---------------------
    print("\n== artifact renders (MCNS v1.0 query, BANC v888+v626 "
          "foreign) ==")
    import tempfile

    from comparison.mapping_visualization import (
        render_bridge_linker_html,
        render_mapping_network_html,
        render_mapping_sankey_html,
        render_source_map_network_html,
    )

    native = collect_native_type_matches(
        "male-cns:v1.0", "DN1pA", uncapped=True)
    enrich_native_type_matches(native, "male-cns:v1.0")
    banc_entries = [e for e in native
                    if str(e.get("dataset", "")).startswith(("banc_", "flywire_BANC"))]
    if not banc_entries:
        fail("no BANC entries for the DN1pA query")
    else:
        flows = build_mapping_flows(banc_entries, "male-cns:v1.0")
        outdir = Path(tempfile.mkdtemp(prefix="drocat_combos_"))
        rendered = {}
        for name, html in [
            ("sankey_type", render_mapping_sankey_html(
                flows, variant="type")),
            ("sankey_linker", render_mapping_sankey_html(
                flows, variant="linker")),
            ("network", render_mapping_network_html(flows)),
            ("linker_paths", render_bridge_linker_html(
                flows, source_dataset="male-cns:v1.0",
                target_dataset="banc_v888")),
            ("source_map", render_source_map_network_html()),
        ]:
            if not html:
                fail(f"{name} render produced nothing")
                continue
            path = outdir / f"mapping_{name}.html"
            path.write_text(html, encoding="utf-8")
            rendered[name] = path
            print(f"  wrote {path} ({len(html):,} bytes)")
        for name in ("sankey_type", "sankey_linker"):
            html = (rendered.get(name) or Path("x")).read_text("utf-8") \
                if name in rendered else ""
            if html and "Node Colors" not in html:
                fail(f"{name}: shared control panel missing")
        linker = rendered.get("linker_paths")
        if linker:
            html = linker.read_text("utf-8")
            for bad in ("FLYW", "ADDI"):
                if bad in html:
                    fail(f"linker_paths: abbreviated code {bad} present")
            if "flywireType" not in html and "additional_type" not in html:
                fail("linker_paths: no full bridge column names")

    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL COMBINATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
