"""Deep verification of the auto type mapper against raw index columns.

Design (plan §8B): the mapper's derivation graph is re-derived INDEPENDENTLY
from the cached index columns and the two are compared exhaustively:

1. Oracle sweep (registry pair MCNS~FAFB): the expected mapped-type relation
   is recomputed from raw columns only —
     same name        : T in target primaries
     crosswalk cells  : flywireType cell values (comma-split) on MCNS rows
                        typed T, kept when they are FAFB primaries
     annotation table : FAFB primaries whose additional_type(s) cells carry
                        T (or a crosswalk value that names T's FAWIRE types)
   compared against `mapper.get_type_bridges` chain end-values.
2. Hop validation: every reported chain hop is validated against raw data —
   crosswalk hops (cell co-occurrence on the previous type), annotation hops
   ({value, via} co-occurrence on primary rows), same-name hops (value
   equality across namespaces).
3. Pool oracle: source/target bodyId pools recomputed via raw filters and
   compared with `pool_bridge_body_ids`.
4. CSV inspection: matched-entries exports parsed and cross-checked against
   the raw indexes (uniform field counts, neuron counts, bridge columns).

Exit code 1 on any failure.  Run:  python scripts/harness/verify_type_mapper_deep.py
"""

import csv
import io
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import polars as pl  # noqa: E402

from comparison.cross_dataset_type_mapper import (  # noqa: E402
    get_type_mapper,
    preferred_bridge_chain,
    standardize_bridge,
)
from comparison.mapping_visualization import (  # noqa: E402
    build_mapping_flows,
    format_bridge,
)
from ui.neuron_index import (  # noqa: E402
    build_matches_csv,
    count_type_in_index,
    load_cached_neuron_index,
    pool_bridge_body_ids,
)

MCNS = "male-cns:v1.0"
FAFB = "flywire_FAFB_v783"
HEMI = "hemibrain:v1.2.1"
MANC = "manc:v1.2.3"

FAILURES = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}"
          + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def parts(cell):
    if cell is None:
        return set()
    return {p.strip() for p in str(cell).split(",") if p.strip()}


def body_ids(frame):
    return [str(b) for b in frame.select("bodyId").to_series().to_list()
            if b is not None and str(b).strip() not in ("", "nan")]


def cells_contain(index, type_name, column, value):
    """ANY row of type_name carries value in its comma-split cell."""
    frame = index.frame.filter(index.frame["type"] == type_name)
    if not frame.height or column not in frame.columns:
        return False
    cell = pl.col(column).cast(pl.Utf8, strict=False)
    mask = (
        cell.str.split(",")
        .list.eval(pl.element().str.strip_chars() == value)
        .list.any()
    )
    return bool(frame.filter(mask.fill_null(False)).height)


def side_pool(index, type_name, column, value):
    """Raw bodyId pool: rows of type_name, filtered by the linker cell."""
    frame = index.frame.filter(index.frame["type"] == type_name)
    if column and column in frame.columns:
        cell = pl.col(column).cast(pl.Utf8, strict=False)
        mask = (
            cell.str.split(",")
            .list.eval(pl.element().str.strip_chars() == value)
            .list.any()
        )
        frame = frame.filter(mask.fill_null(False))
    return set(body_ids(frame))


def main():
    mapper = get_type_mapper()
    assert mapper.load() is True
    mcns = load_cached_neuron_index(MCNS)
    fafb = load_cached_neuron_index(FAFB)
    hemi = load_cached_neuron_index(HEMI)
    manc = load_cached_neuron_index(MANC)

    fafb_prim = set(mapper._flywire_primaries.get(FAFB, ()))
    hemi_types = set(hemi.frame["type"].drop_nulls().to_list())
    manc_types = set(manc.frame["type"].drop_nulls().to_list())

    # ---- independent raw-column relations --------------------------------
    crosswalk = defaultdict(lambda: defaultdict(set))
    for column, target_names in (("flywireType", fafb_prim),
                                 ("hemibrainType", hemi_types),
                                 ("mancType", manc_types)):
        for t, cell in zip(mcns.frame["type"].to_list(),
                           mcns.frame[column].to_list()):
            if cell is None:
                continue
            vals = parts(cell) & target_names
            if vals:
                crosswalk[str(t)][column] |= vals

    fafb_alt_to_prim = defaultdict(set)
    for p, cell in zip(fafb.frame["type"].to_list(),
                       fafb.frame["additional_type(s)"].to_list()):
        if cell is None:
            continue
        for alt in parts(cell) - {p}:
            fafb_alt_to_prim[alt].add(p)

    # ---- 1. oracle sweep MCNS → FAFB -------------------------------------
    # Independent bounded BFS over the same raw-column relations the mapper
    # documents: same-name hops only from the start fan-out (and as arrival
    # into the target), crosswalk hops from MCNS nodes, annotation hops
    # (both directions) inside FlyWire namespaces, total depth <= 4.
    # Reachable FAFB-primary names == the mapper's chain ends.
    print("== oracle sweep MCNS→FAFB (bounded BFS) over all MCNS types ==")
    from collections import deque

    fafb_annotation_to_prim = defaultdict(set)
    for p, cell in zip(fafb.frame["type"].to_list(),
                       fafb.frame["additional_type(s)"].to_list()):
        if cell is None:
            continue
        for alt in parts(cell) - {p}:
            fafb_annotation_to_prim[alt].add(p)
    fafb_prim_to_annotation = defaultdict(set)
    for alt, prims in fafb_annotation_to_prim.items():
        for p in prims:
            fafb_prim_to_annotation[p].add(alt)

    MAX_HOPS = 4

    ANNOTATION_COLUMNS = ("additional_type(s)",
                          "Alternative Cell Type(s)")

    def oracle_ends(source_type):
        start = (MCNS, source_type)
        seen = {(start, "")}
        queue = deque([(start, 0, "")])
        ends = set()
        while queue:
            (ns, name), hops, prev_column = queue.popleft()
            if hops >= MAX_HOPS:
                continue
            neighbors = []
            # same-name: from the source fan-out (hop 1), and as arrival
            # into the target namespace from anywhere
            if hops == 0:
                for other, names in (
                        (FAFB, fafb_prim), (HEMI, hemi_types),
                        (MANC, manc_types)):
                    if name in names:
                        neighbors.append((other, name, "type"))
            elif ns != FAFB:
                if name in fafb_prim:
                    neighbors.append((FAFB, name, "type"))
            # crosswalk hops: MCNS nodes name FAFB primaries
            if ns == MCNS:
                for v in crosswalk.get(name, {}).get("flywireType", ()):
                    neighbors.append((FAFB, v, "flywireType"))
            # annotation hops inside the FAFB namespace (both directions)
            if ns == FAFB:
                for p in fafb_annotation_to_prim.get(name, ()):
                    neighbors.append((FAFB, p, "additional_type(s)"))
                for a in fafb_prim_to_annotation.get(name, ()):
                    neighbors.append((FAFB, a, "additional_type(s)"))
            for nns, nname, column in neighbors:
                if column in ANNOTATION_COLUMNS \
                        and prev_column in ANNOTATION_COLUMNS:
                    continue  # no annotation-to-annotation chaining
                node = (nns, nname)
                state = (node, column)
                if state in seen:
                    continue
                seen.add(state)
                # FlyWire endpoints must be real primary types — non-primary
                # FAWIRE names remain traversable graph nodes but never ends
                if nns == FAFB and nname in fafb_prim:
                    ends.add(nname)
                queue.append((node, hops + 1, column))
        return ends

    sweep_types = sorted(set(crosswalk) | set(fafb_annotation_to_prim)
                         | (set(mcns.frame["type"].drop_nulls().to_list())
                            & fafb_prim))
    sweep_types = [t for t in sweep_types if t]
    print(f"   types compared: {len(sweep_types)}")
    missing_total = extra_total = compared = 0
    examples = []
    for t in sweep_types:
        expected = oracle_ends(t)
        if not expected:
            continue
        compared += 1
        ends = {c[-1]["value"] for c in mapper.get_type_bridges(t, MCNS, FAFB)
                if c}
        if ends != expected:
            missing = expected - ends
            extra = ends - expected
            missing_total += bool(missing)
            extra_total += bool(extra)
            if len(examples) < 6:
                examples.append((t, sorted(missing)[:4], sorted(extra)[:4]))
    check("oracle sweep: no missing mappings", missing_total == 0,
          f"{compared} types compared, {missing_total} with missing ends")
    check("oracle sweep: no extra mappings", extra_total == 0,
          f"{extra_total} with extra ends")
    for t, missing, extra in examples:
        print(f"    {t}: missing={missing} extra={extra}")

    # ---- 2. hop validation on sampled chains ------------------------------
    print("\n== hop validation ==")
    rng = random.Random(11)
    sample = rng.sample(sweep_types, min(400, len(sweep_types)))
    validated = bad = 0
    bad_examples = []
    for t in sample:
        for chain in mapper.get_type_bridges(t, MCNS, FAFB):
            ok = True
            prev = chain[0]
            if prev["dataset"] != MCNS or prev["column"] != "type" \
                    or prev["value"] != t:
                ok = False
            for hop in chain[1:]:
                column = hop["column"]
                if column == "type":
                    if hop["value"] != prev["value"]:
                        ok = False
                elif column == "flywireType":
                    if not cells_contain(mcns, prev["value"], column,
                                         hop["value"]):
                        ok = False
                else:  # annotation hop: {value, via} co-occurrence
                    if not hop.get("via"):
                        ok = False
                    elif not (cells_contain(fafb, hop["value"],
                                            column, hop["via"])
                              or cells_contain(fafb, hop["via"],
                                               column, hop["value"])):
                        ok = False
                if not ok:
                    break
                prev = hop
            validated += 1
            if not ok:
                bad += 1
                if len(bad_examples) < 3:
                    bad_examples.append(format_bridge(chain))
    check(f"hop validation on {validated} chains", bad == 0,
          "; ".join(bad_examples))

    # ---- 3. pool oracle on sampled chains ---------------------------------
    print("\n== pool oracle ==")
    pool_checked = pool_bad = 0
    for t in sample[:60]:
        for chain in mapper.get_type_bridges(t, MCNS, FAFB)[:2]:
            foreign = chain[-1]["value"]
            linkers = [l for l in standardize_bridge(chain, MCNS, FAFB)
                       if l.get("kind") == "linker"]
            pool = pool_bridge_body_ids(
                MCNS, FAFB, linkers, t, foreign,
                indexes={MCNS: mcns, FAFB: fafb})
            exp_src, exp_tgt = set(), set()
            src_constrained = tgt_constrained = False
            for linker in linkers:
                home = linker.get("home", "")
                if home == MCNS:
                    src_constrained = True
                    exp_src |= side_pool(mcns, t, linker["column"],
                                         linker["value"])
                else:
                    tgt_constrained = True
                    exp_tgt |= side_pool(fafb, foreign, linker["column"],
                                         linker["value"])
            if not src_constrained:
                exp_src = set(body_ids(
                    mcns.frame.filter(mcns.frame["type"] == t)))
            if not tgt_constrained:
                exp_tgt = set(body_ids(
                    fafb.frame.filter(fafb.frame["type"] == foreign)))
            got_src = set(pool["source_body_ids"])
            got_tgt = set(pool["target_body_ids"])
            pool_checked += 1
            if got_src != exp_src or got_tgt != exp_tgt:
                pool_bad += 1
                if pool_bad <= 3:
                    print(f"    {t}→{foreign}: src {len(got_src)}/"
                          f"{len(exp_src)} tgt {len(got_tgt)}/{len(exp_tgt)}")
    check(f"pool oracle on {pool_checked} pools", pool_bad == 0)

    # ---- 4. CSV content inspection ----------------------------------------
    print("\n== CSV inspection ==")
    index_by_ds = {
        ds: load_cached_neuron_index(ds)
        for ds in (MCNS, FAFB, HEMI, MANC, "banc_v888",
                   "banc_v626")
    }
    for query in ("circadian_clock", "APDN3", "DN1pB", "l-LNv"):
        csv_text = build_matches_csv(MCNS, query)
        if not csv_text:
            print(f"    '{query}': no entries (skipped)")
            continue
        rows = list(csv.reader(io.StringIO(csv_text)))
        field_counts = {len(r) for r in rows}
        check(f"entries CSV '{query}' uniform rows",
              len(field_counts) == 1, f"{len(rows) - 1} data rows")
        header = rows[0]
        f_ds, f_ft = header.index("dataset"), header.index("foreign_type")
        f_cnt = header.index("neuron_count")
        bad_count = 0
        for r in rows[1:]:
            if not r[f_ft] or r[f_ds] not in index_by_ds:
                continue
            expected = count_type_in_index(index_by_ds[r[f_ds]], r[f_ft])
            if expected is not None and r[f_cnt] \
                    and int(r[f_cnt]) != expected:
                bad_count += 1
        check(f"entries CSV '{query}' neuron counts match indexes",
              bad_count == 0, f"{bad_count} mismatched")

    # mapped-rows CSV: provenance must be filled for every visible type
    from ui.neuron_index import (collect_native_type_matches,  # noqa: E402
                                 enrich_native_type_matches,
                                 query_neuron_index)
    native = collect_native_type_matches(MCNS, "circadian_clock")
    enrich_native_type_matches(native, MCNS)
    entry = next(e for e in native if e["dataset"] == FAFB)
    mapped = set(entry["mapped_type_names"])
    index = load_cached_neuron_index(MCNS)
    result = query_neuron_index(
        index, types_include=sorted(mapped), page=1, page_size=50,
        include_all_rows=True)
    row_types = {r["type"] for r in result.rows}
    check("every mapped-view row type has provenance data",
          row_types <= mapped, f"{len(row_types)} types on full row set")

    print()
    if FAILURES:
        print("FAILURES:", FAILURES)
        sys.exit(1)
    print("ALL DEEP CHECKS PASSED")


if __name__ == "__main__":
    main()
