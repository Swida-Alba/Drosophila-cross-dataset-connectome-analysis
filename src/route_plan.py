"""Route-plan corridor lane core (plan-corridor-lane.md R1/R2/R3, §23.3).

R1  build_type_pairs      — G_type* (type_pre, type_post, w_sum, w_max, n)
R2  CorridorDrain         — W-guided per-source resumable drain generator
                            (UB-descending completions for all targets/
                            lengths; PLANNER RULE: type-revisiting walks
                            are not planned unless allow_multi_visit —
                            checked at the FIRST hop too)
R3  validate_and_fill     — lazy plan/validate loop: phantom check (layered
                            reachability; exact forward DP for distinct-
                            type walks with a work cap, visited DFS for
                            multi-visit) -> realizations vs per-(pair,h)
                            path budget -> per-class floors/budget stops
run_lane                  — standalone driver: reads the run folder's
                            parameters.txt (spaced-key grammar), reproduces
                            the run's edge universe (filter parity; the
                            shortest-store basis is read vectorized and
                            DEDUPED keep-first by sorted layer), then
                            enumerates PER-WALK subgraphs (§22 share
                            consumption; per-position layered graphs built
                            from the shared _PairEdges cache via
                            FastGraph.build_from_edges) and exports the
                            §23.5 layout: one type row per validated walk
                            (*_allpaths_type.csv column family + additive
                            lane columns), per-path bodyId rows, scope
                            type-pair masses, content-hashed plan.json,
                            provenance with gap partition + per-stage
                            phases_s + store_dedup, README. ONE monotonic
                            clock drives the wall-clock budget (graceful
                            time-capping); an explicit out_dir owns the
                            whole layout, plan.json included.

Pure library: no coana import at module scope; polars-in, plain dicts out.
Heavy imports (coana, type_level_refill, FastGraph) happen inside run_lane.
"""
from __future__ import annotations

import heapq
import time
from collections import defaultdict
from dataclasses import dataclass, field

import polars as pl

UNSET = 0

_HEMI_SUFFIXES = ("_L", "_R", "_U")


class CorridorRefusal(Exception):
    """Deterministic lane refusal (disclosed, never a run failure): the
    run's settings or resolvable inputs make the corridor universe
    non-reproducible (plan §3b tier 3 — the hook turns this into one
    user_warning_notes entry)."""


# ---------------------------------------------------------------- R1
def build_type_pairs(conn: pl.DataFrame, tmap: pl.DataFrame,
                     min_synapse: int, drop_untyped: bool = True
                     ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """conn: bodyId_pre, bodyId_post, weight (Utf8 ids, pair-collapsed).
    tmap: bid, t. Returns (typed_edges, type_pairs). Filter parity is the
    CALLER's duty (run_lane applies the run's own settings first)."""
    conn = conn.filter(pl.col("weight") >= min_synapse)
    typed = (conn.join(tmap, left_on="bodyId_pre", right_on="bid")
                 .rename({"t": "type_pre"})
                 .join(tmap, left_on="bodyId_post", right_on="bid")
                 .rename({"t": "type_post"})
                 .select("bodyId_pre", "bodyId_post", "weight",
                         "type_pre", "type_post"))
    if drop_untyped:
        bad = pl.Series([
            _missing(a) or _missing(b)
            for a, b in zip(typed["type_pre"], typed["type_post"])])
        typed = typed.filter(~bad)
    pairs = (typed.group_by("type_pre", "type_post")
             .agg(pl.col("weight").sum().alias("w_sum"),
                  pl.col("weight").max().alias("w_max"),
                  pl.len().alias("n")))
    return typed, pairs


def _missing(v) -> bool:
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in ("", "nan", "none", "null")


def wmax_map(pairs: pl.DataFrame) -> dict:
    return {(a, b): w for a, b, w in zip(
        pairs["type_pre"], pairs["type_post"], pairs["w_max"])}


# ---------------------------------------------------------------- R2
def widest_to_targets(adj: dict, targets: set, bound: int) -> list[dict]:
    """W[r][x] = widest-path value of the best walk x -> any target <= r hops.
    UB admissibility basis: UB(c) = min over hops of w*; W guides the drain.
    Shared across sources of one run (same target set) — computing it per
    source re-drains the identical DP (§18 measured 3.1 s for 63 pairs)."""
    universe = set(adj)
    for nbrs in adj.values():
        universe.update(v for v, _w in nbrs)
    W = [dict.fromkeys(universe, UNSET) for _ in range(bound + 1)]
    for t in targets:
        if t in W[0]:
            W[0][t] = float("inf")
    for r in range(1, bound + 1):
        prev, cur = W[r - 1], W[r]
        for x, nbrs in adj.items():
            best = UNSET
            for y, w in nbrs:
                py = prev.get(y, UNSET)
                if py:
                    v = w if w < py else py
                    if v > best:
                        best = v
            cur[x] = best
        for t in targets:
            if t in cur:
                cur[t] = float("inf")
    return W


def widest_to_targets_fast(pairs: pl.DataFrame, targets: set, bound: int
                           ) -> list[dict]:
    """Polars relaxation of widest_to_targets over the pair table —
    identical semantics (pure-Python twin kept for small graphs/tests;
    FAFB's 2.8M pairs x bound rounds cost ~20 s in per-edge Python)."""
    universe = set(pairs["type_pre"].unique().to_list()) \
        | set(pairs["type_post"].unique().to_list()) | set(targets)
    targets_pl = pl.DataFrame(
        {"node": sorted(targets)}, schema={"node": pl.Utf8})
    edges = pairs.select("type_pre", "type_post", "w_max")
    W = [{t: (float("inf") if t in targets else UNSET) for t in universe}]
    prev = pl.DataFrame(
        {"y": list(W[0].keys()), "v": list(W[0].values())},
        schema={"y": pl.Utf8, "v": pl.Float64})
    for _r in range(1, bound + 1):
        cand = (edges.join(prev, left_on="type_post", right_on="y")
                .select(pl.col("type_pre"),
                        pl.min_horizontal("w_max", "v").alias("cand"))
                .group_by("type_pre")
                .agg(pl.col("cand").max().alias("v"))
                .rename({"type_pre": "node"}))
        cur_df = (pl.DataFrame(
            {"node": sorted(universe)}, schema={"node": pl.Utf8})
            .join(cand, on="node", how="left")
            .join(targets_pl, on="node", how="left", suffix="_tgt")
            .with_columns(
                pl.when(pl.col("node").is_in(targets_pl["node"]))
                .then(float("inf"))
                .otherwise(pl.col("v").fill_null(UNSET))
                .alias("v"))
            .select("node", "v"))
        W.append(dict(zip(cur_df["node"].to_list(),
                          cur_df["v"].to_list())))
        prev = cur_df.rename({"node": "y"})
    return W


@dataclass
class ClassBudget:
    """Per (pair, h) binding path budget with adaptive walk acceptance.

    mh_prov (§21): the provisional candidate-probe cap
    clamp(M_base * decay**(h-1)) — a class stops pulling candidates once
    probed that many (the binding limit stays the path budget). feasible
    (§22 cost control): False when no length-h walk can exist (h below the
    min type distance) — such a class can never receive candidates, so it
    must not keep the drain alive."""
    limit: int = 100
    used: int = 0
    walks: list = field(default_factory=list)   # accepted (ub, walk)
    floor: float | None = None
    floor_next: float | None = None
    phantoms: int = 0
    unverified: int = 0
    truncated_walks: int = 0
    ties_truncated: int = 0
    # F-PF-006 (audit 2026-10-13): a share break used to set
    # ties_truncated without checking whether an equal-strength route
    # actually existed beyond the share. The flag now splits: CONFIRMED
    # (the next yielded realization's bottleneck equals the class tau)
    # vs POTENTIAL (share reached, next unverified — e.g. a deadline cut
    # the peek). ties_truncated keeps meaning confirmed+potential.
    ties_truncated_confirmed: int = 0
    ties_truncated_potential: int = 0
    layered_rejected: int = 0
    pulled: int = 0
    stop: str = ""
    mh_prov: int | None = None
    feasible: bool = True
    # enumeration-side disclosures (filled by run_lane's per-class pass)
    emitted: int = 0
    off_plan: int = 0
    off_len: int = 0

    def satisfied(self) -> bool:
        if self.used >= self.limit or not self.feasible:
            return True
        return (self.mh_prov is not None
                and self.pulled >= self.mh_prov)


class CorridorDrain:
    """W-guided best-first drain from ONE source type; completions for all
    targets/lengths at once; resumable (generator semantics)."""

    def __init__(self, adj, source, targets, bound, max_pops=1_200_000,
                 W=None, deadline=None, max_pushes=4_000_000,
                 allow_multi_visit=False):
        self.adj, self.source, self.tgt = adj, source, set(targets)
        self.bound, self.max_pops = bound, max_pops
        self.deadline = deadline
        # PLANNER RULE (2026-10-11, user default): a walk that revisits a
        # type is NOT planned — reciprocals are common in the connectome
        # and multi-visit subtrees explode the drain for little gain.
        # (The run's own type paths still contain repeated-type routes —
        # realized by DIFFERENT bodies, per the derivation rule in
        # TYPE_AGGREGATION_AND_BODYID_DISCOVERY.md — and the REFILL full
        # scope re-values those; the corridor simply does not plan new
        # ones.) allow_multi_visit=True restores the old behavior.
        self.allow_multi_visit = allow_multi_visit
        self.multi_visit_pruned = 0
        # PERF (2026-10-10): each pop expands ~avg-degree neighbors onto
        # the heap — at deep bounds over the whole type quotient the heap
        # grows to tens of millions of tuples and the run dies in GC/swap
        # (measured: 700 s+ sys time). The push budget bounds the heap
        # itself; exhausted => 'push_capped' (disclosed like pops).
        self.max_pushes = max_pushes
        self.pushes = 0
        self.push_capped = False
        self.time_capped = False
        # F-PF-007 (audit 2026-10-13): the pop budget used to set
        # exhausted=True with no flag of its own, so a pop-capped drain was
        # indistinguishable from a fully searched graph and hungry classes
        # were labeled 'exhausted'/'dry'. exhausted now means NATURAL heap
        # exhaustion only; a pop-budget stop carries pop_capped and the
        # stop assignment names the cause on every pending class.
        self.pop_capped = False
        self.W = W if W is not None else widest_to_targets(
            adj, self.tgt, bound)
        self.heap = []
        self.counter = 0
        self.pops = 0
        self.exhausted = False
        for v, w in sorted(adj.get(source, []), key=lambda x: -x[1]):
            # PLANNER RULE at the FIRST hop too (2026-10-11, plan F1): the
            # expansion checks look at the walk tuple, but this initial push
            # never did — a self type pair (X->X, present whenever
            # exclude_intra is off) entered as walk (X, X) and every later
            # hop looked new. That is exactly the R8->R8->aMe12->... shape
            # that survived the rule on the real UI run.
            if not self.allow_multi_visit and v == source:
                self.multi_visit_pruned += 1
                continue
            g = self.W[bound - 1].get(v, UNSET)
            if g:
                heapq.heappush(self.heap,
                               (-min(w, g), self.counter, v, 1,
                                (source, v), w))
                self.counter += 1

    def __iter__(self):
        return self

    def __next__(self):
        """Yield (ub, walk) in descending-UB guided order; StopIteration at
        exhaustion/cap."""
        while self.heap:
            if self.pops >= self.max_pops:
                self.pop_capped = True
                raise StopIteration
            if (self.deadline is not None
                    and not self.pops & 4095
                    and time.monotonic() > self.deadline):
                # the wall-clock budget must bite INSIDE the drain: at deep
                # bounds the generator can pop millions of prefixes between
                # two completions, and the caller's per-completion check
                # never regains control (in-hook bound-5 anchor lesson)
                self.time_capped = True
                self.exhausted = True
                raise StopIteration
            if self.pushes >= self.max_pushes:
                self.push_capped = True
                self.exhausted = True
                raise StopIteration
            negp, _, node, hops, walk, ub = heapq.heappop(self.heap)
            self.pops += 1
            prio = -negp
            if prio <= UNSET:
                break
            if node in self.tgt and hops >= 1:
                if hops < self.bound:
                    remaining = self.bound - hops
                    for v2, w2 in self.adj.get(node, []):
                        if (not self.allow_multi_visit
                                and v2 in walk):
                            self.multi_visit_pruned += 1
                            continue
                        g = self.W[remaining - 1].get(v2, UNSET)
                        if g:
                            nub = ub if ub < w2 else w2
                            np_ = nub if nub < g else g
                            heapq.heappush(self.heap,
                                           (-np_, self.counter, v2, hops + 1,
                                            walk + (v2,), nub))
                            self.counter += 1
                            self.pushes += 1
                return ub, walk
            if hops >= self.bound:
                continue
            remaining = self.bound - hops
            for v2, w2 in self.adj.get(node, []):
                if (not self.allow_multi_visit
                        and v2 in walk):
                    self.multi_visit_pruned += 1
                    continue
                g = self.W[remaining - 1].get(v2, UNSET)
                if not g:
                    continue
                nub = ub if ub < w2 else w2
                np_ = nub if nub < g else g
                heapq.heappush(self.heap, (-np_, self.counter, v2, hops + 1,
                                           walk + (v2,), nub))
                self.counter += 1
                self.pushes += 1
        self.exhausted = True
        raise StopIteration


# ---------------------------------------------------------------- R3
def realize_count(hop_edge_dicts, src_bodies, tgt_bodies, budget,
                  work_cap=200_000):
    """Exact layered simple-path count for ONE walk (hop-i edges at layer i).

    work_cap bounds the node-expansion WORK, not just the count: deep
    phantom chains at low thresholds can explode combinatorially before
    yielding the first path (the bound-5 in-hook anchor burned >40 min
    there). Returns (count, truncated): truncated=True means the
    exploration was cut early — a positive count is then a lower bound,
    and a ZERO count proves nothing (the caller must treat it as
    unverified, not phantom).
    """
    count = 0
    work = 0
    truncated = False
    n = len(hop_edge_dicts)

    def dfs(i, body, visited):
        nonlocal count, work, truncated
        if i == n:
            if body in tgt_bodies:
                count += 1
            return
        for v in hop_edge_dicts[i].get(body, ()):
            work += 1
            if work > work_cap:
                truncated = True
                return
            if v in visited:
                continue
            visited.add(v)
            dfs(i + 1, v, visited)
            visited.remove(v)
            if count >= budget or truncated:
                return

    for s in src_bodies:
        if truncated:
            break
        dfs(0, s, {s})
        if count >= budget:
            break
    return count, truncated


def validate_and_fill(drain: CorridorDrain, classes: dict,
                      realize_fn, max_pulls: int = 200_000,
                      deadline: float | None = None,
                      stall_limit: int = 200_000):
    """classes: {(pair, h): ClassBudget}. Consumes the drain lazily; routes
    each completion to its class; phantom-checks; accepts until the class
    budget binds. Returns per-class provenance.

    realize_fn returns (count, truncated). The global stop tracks a
    PENDING set of unsatisfied classes and re-checks it on every state
    transition (acceptance, phantom, probe-cap crossing, skip) — a class
    that satisfies WITHOUT an acceptance (probe cap reached on phantoms)
    must still terminate the drain, or the loop spins to the pop cap (the
    bound-5 in-hook anchor hung >40 min exactly there). ``deadline`` is a
    time.monotonic() cut: on exceed the loop stops and hungry classes are
    disclosed as capped (in-run resource envelope, plan §3c).

    PERF refinement (2026-10-10): ``stall_limit`` bounds FRUITLESS
    completions — completions routed to no hungry class since the last
    acceptance. At deep bounds the UB-descending completion space over
    the whole type quotient is enormous and the drain can pop millions
    of prefixes while every hungry class waits for rare deep candidates
    (the MCNS store-basis profile: ~700 s of drain for a handful of
    acceptances). On a stall the loop stops and hungry classes are
    disclosed as 'stalled' (their corridors may exist beyond the stall —
    unreached ≠ nonexistent, the standing deep-bound caveat)."""
    pulls = 0
    capped = False
    stalled = False
    since_accept = 0
    time_capped = bool(getattr(drain, "time_capped", False))
    pending = {k for k, cb in classes.items() if not cb.satisfied()}
    for ub, walk in drain:
        if not pending:
            # §22 global stop: every class satisfied (full / probe-capped /
            # infeasible) — further drain pops buy nothing
            break
        if deadline is not None and time.monotonic() > deadline:
            capped = time_capped = True
            break
        if since_accept >= stall_limit:
            stalled = True
            break
        if pulls >= max_pulls:
            capped = True
            break
        pulls += 1
        key = ((walk[0], walk[-1]), len(walk) - 1)
        cb = classes.get(key)
        if cb is None:
            since_accept += 1
            continue
        if cb.satisfied():
            # Class already done: the first post-stop candidate's UB is
            # the honest floor witness (plan §5: floor = UB(M+1) — the
            # last ACCEPTED UB can overstate safety by a tie).
            if cb.floor_next is None:
                cb.floor_next = ub
            pending.discard(key)
            continue
        cb.pulled += 1
        since_accept += 1
        n, truncated = realize_fn(walk, cb.limit - cb.used)
        if n == 0:
            if truncated:
                # work-capped before the first path: unverified, NOT a
                # proven phantom — disclosed separately
                cb.unverified += 1
            else:
                cb.phantoms += 1
            if cb.satisfied() and key in pending:
                pending.discard(key)
            continue
        since_accept = 0
        cb.walks.append((ub, walk))
        cb.used += n
        cb.floor = ub
        if truncated:
            cb.truncated_walks += 1
        if cb.used >= cb.limit:
            cb.stop = "budget"
        if cb.satisfied():
            pending.discard(key)
    time_capped = time_capped or bool(getattr(drain, "time_capped", False))
    pop_capped = bool(getattr(drain, "pop_capped", False))
    push_capped = bool(getattr(drain, "push_capped", False))
    for _key, cb in classes.items():
        if cb.stop == "budget":
            continue
        if not cb.feasible:
            cb.stop = "infeasible"
        elif cb.mh_prov is not None and cb.pulled >= cb.mh_prov \
                and cb.used < cb.limit:
            cb.stop = "probe_cap"
        elif stalled:
            # the accept stream stalled while this class stayed hungry —
            # its emptiness is unverified (F-PF-007 family: no absence
            # claims from capped/unverified classes)
            cb.stop = "stalled"
        elif push_capped:
            cb.stop = "push_capped"
        elif pop_capped:
            # F-PF-007: the pop budget died with candidates unsearched —
            # never report this as exhaustion or dryness (the pulled /
            # unpulled distinction lives in the class's pulled counter)
            cb.stop = "pop_capped"
        elif capped or time_capped:
            # a cap fired while this class was still hungry and unpulled —
            # its emptiness is UNVERIFIED, never an absence claim
            # (the old no-pull "dry" here claimed proven absence)
            cb.stop = "capped"
        elif not drain.exhausted:
            cb.stop = "capped" if cb.pulled else "dry"
        elif cb.pulled:
            cb.stop = "exhausted" if cb.used < cb.limit else "budget"
        else:
            cb.stop = "dry"
    return {"drain_pops": drain.pops, "drain_exhausted": drain.exhausted,
            "drain_pop_capped": pop_capped,
            "pulls": pulls, "stalled": stalled,
            "time_capped": time_capped,
            "classes": {f"{k[0][0]}->{k[0][1]}@h{k[1]}": {
                "accepted": len(cb.walks), "used": cb.used,
                "phantoms": cb.phantoms, "unverified": cb.unverified,
                "ties_truncated": cb.ties_truncated,
                "layered_rejected": cb.layered_rejected,
                "pulled": cb.pulled,
                "floor": cb.floor, "floor_next": cb.floor_next,
                "mh_prov": cb.mh_prov, "feasible": cb.feasible,
                "stop": cb.stop}
                for k, cb in classes.items()}}


class _PairEdges:
    """Per-(type pair) edge data over the pruned universe, materialized
    ONCE per pair from ONE canonical sort (plan F4/F5 — the previous
    _HopIndex re-filtered the whole frame per pair, O(pairs x |loss|) on
    the multimillion-row store universe). Two views share the cache:

      hop(a, b)    -> {u: [v, ...]}     dedup'd adjacency (validation)
      tuples(a, b) -> [(u, v, w), ...]  canonical order (enumeration)

    Rows are dedup'd keep-first (canonical (u, v) order makes duplicate
    store rows adjacent, so this is deterministic — plan F8)."""

    def __init__(self, loss: pl.DataFrame):
        self._loss = loss
        self._sorted = None      # canonical (pair, pre, post) row order
        self._bounds: dict = {}  # (a, b) -> (start, end) row slice
        self._cache: dict = {}

    def _ensure_sorted(self):
        if self._sorted is not None:
            return
        df = self._loss.sort(["type_pre", "type_post",
                              "bodyId_pre", "bodyId_post"])
        pres = df["type_pre"].to_list()
        posts = df["type_post"].to_list()
        n = len(pres)
        bounds = {}
        start = 0
        for i in range(1, n + 1):
            if i == n or pres[i] != pres[start] or posts[i] != posts[start]:
                bounds[(pres[start], posts[start])] = (start, i)
                start = i
        self._sorted = df
        self._bounds = bounds

    def _columns(self, a, b):
        key = (a, b)
        rec = self._cache.get(key)
        if rec is None:
            self._ensure_sorted()
            rng = self._bounds.get(key)
            if rng is None:
                rec = ([], [], [])
            else:
                part = self._sorted[rng[0]:rng[1]]
                rec = (part["bodyId_pre"].to_list(),
                       part["bodyId_post"].to_list(),
                       part["weight"].to_list())
            self._cache[key] = rec
        return rec

    def hop(self, a, b) -> dict:
        """u -> dedup'd [v, ...] adjacency (validation view)."""
        us, vs, _ws = self._columns(a, b)
        d = {}
        prev_u = prev_v = None
        for u, v in zip(us, vs):
            if u == prev_u and v == prev_v:
                continue
            d.setdefault(u, []).append(v)
            prev_u, prev_v = u, v
        return d

    def tuples(self, a, b) -> list:
        """Dedup'd (u, v, w) rows in canonical order (enumeration view)."""
        us, vs, ws = self._columns(a, b)
        out = []
        prev = None
        for u, v, w in zip(us, vs, ws):
            if (u, v) == prev:
                continue
            out.append((u, v, w))
            prev = (u, v)
        return out


def _dp_count(hop_dicts, src_bodies, tgt_bodies, work_cap=2_000_000):
    """EXACT layered realization count via forward path counting — no path
    materialization. VALID only when the walk's types are all DISTINCT: a
    body has one type, so distinct types make body-level simplicity
    automatic (a layered walk cannot revisit a body). Multi-visit walks
    must use realize_count's visited DFS (simplicity is not
    DP-enforceable).

    work_cap (2026-10-11, plan F3) bounds the edge RELAXATIONS: deep
    store-basis chains carry subgraphs where one uncapped call burned
    minutes (the 4,471s store-basis profile). A capped pass returns
    (lower_bound, True) — forward sums only grow, so a partial pass
    under-counts but never over-counts; zero + truncated proves nothing
    (the caller treats it as unverified, realize_count's contract)."""
    h = len(hop_dicts)
    f = [defaultdict(int) for _ in range(h + 1)]
    for s in src_bodies:
        f[0][s] = 1
    work = 0
    truncated = False
    for i in range(h):
        d = hop_dicts[i]
        nxt = f[i + 1]
        for u, cnt in f[i].items():
            for v in d.get(u, ()):
                nxt[v] += cnt
                work += 1
                if work > work_cap:
                    truncated = True
                    break
            if truncated:
                break
        if truncated:
            break
    total = 0
    last = f[h]
    for t in tgt_bodies:
        total += last.get(t, 0)
    return total, truncated


def _walk_is_distinct(walk):
    return len(set(walk)) == len(walk)


def _realize_on_loss(pair_edges, walk, src_bodies, tgt_bodies, cap):
    """Layered-reachability realizations of ONE walk via the shared
    per-pair edge cache."""
    hd = [pair_edges.hop(a, b) for a, b in zip(walk, walk[1:])]
    if _walk_is_distinct(walk):
        # distinct types: the DP is exact (simplicity automatic) and
        # proves phantoms in O(chain subgraph) — the DFS's work-cap
        # 'unverified' category disappears for these walks (unless the
        # DP's own work cap fires on a store-basis monster)
        n, truncated = _dp_count(hd, src_bodies, tgt_bodies)
        return (n if n < cap else cap), truncated
    return realize_count(hd, src_bodies, tgt_bodies, cap)


# ------------------------------------------------------- run parameters
def read_run_params(run_dir) -> dict:
    """Parse a run folder's parameters.txt into a plain dict.

    The engine writes SPACED keys ('min synapse number', 'max interlayer',
    'source neurons', ...) with Python list reprs for the query neurons —
    NOT the underscore grammar; the refill's read_run_provenance is the
    precedent. First stamp wins on contradictory re-stamps."""
    import re
    from pathlib import Path

    text = (Path(run_dir) / "parameters.txt").read_text(
        encoding="utf-8", errors="replace")
    out: dict = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z][^:]*):\s*(.*)$", line)
        if not m:
            continue
        k, v = m.group(1).strip(), m.group(2).strip()
        prev = out.get(k)
        if prev in (None, "", v):
            out[k] = v
    return out


def _parse_query_tokens(v):
    """Parse a run's source/target token record into (mode, tokens).

    The engine records dict filters verbatim (e.g. a UI wildcard query
    writes ``{'startswith': ['R7', 'R8']}``) — return ('startswith',
    tokens) for those; plain lists/CSVs return ('equal', tokens)."""
    import ast
    v = (v or "").strip()
    if not v:
        return "equal", []
    try:
        parsed = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, dict):
        tokens = []
        for token_list in parsed.values():
            tokens.extend(str(t).strip() for t in token_list
                          if str(t).strip())
        return "startswith", tokens
    if isinstance(parsed, (list, tuple)):
        return "equal", [str(t).strip() for t in parsed if str(t).strip()]
    return "equal", [t.strip() for t in v.split(",") if t.strip()]


def _parse_list(v):
    import ast
    v = (v or "").strip()
    if not v:
        return []
    try:
        parsed = ast.literal_eval(v)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    return [s.strip() for s in v.split(",") if s.strip()]


def _parse_int(v, default=None):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def _parse_float(v, default=None):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def _parse_bool(v, default=False):
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("true", "yes", "1"):
        return True
    if s in ("false", "no", "0", "none", ""):
        return False
    return default


def _match_type_labels(labels, tokens, body2type=None, prefix=False):
    """Enroll run-query tokens against effective labels. A token matches a
    label on equality, on a hemisphere-suffix strip (separate_hemispheres
    runs enroll both 'X_L' and 'X_R' for token 'X'), as a bodyId, or —
    prefix mode (a recorded dict-filter query) — as a startswith match.

    F-PF-005 (audit 2026-10-13): returns a SECOND set — the tokens that
    resolved as explicit bodyIds. Such a token used to enroll its whole
    TYPE, so an unrequested sibling cell entered the query and could
    displace the requested one; the lane now keeps the explicit bodyId
    membership and restricts the endpoint body sets to it. Label
    equality keeps today's precedence (a token that is also a label
    stays a label)."""
    tokens = [str(t).strip() for t in tokens if str(t).strip()]
    label_set = set(labels)
    matched = set()
    body_ids = set()

    def _enroll(tok):
        if tok in label_set:
            matched.add(tok)
            return
        for lab in labels:
            if (lab.endswith(_HEMI_SUFFIXES)
                    and lab[: lab.rfind("_")] == tok):
                matched.add(lab)
        if body2type and tok in body2type:
            matched.add(body2type[tok])
            body_ids.add(tok)

    for tok in tokens:
        _enroll(tok)
    if prefix:
        # UI wildcard query: {'startswith': ['R7', 'R8']} recorded in
        # parameters.txt — enroll every effective label with the prefix
        for lab in labels:
            if any(lab.startswith(tok) for tok in tokens):
                matched.add(lab)
    return matched, body_ids


# ------------------------------------------------------------- driver
def run_lane(run_dir=None, corridor_mode="both", top_m_base=10,
             length_decay=1.5, per_pair_length_budget=100, fnc=None,
             asked=None, max_interlayer=None, source_types=None,
             target_types=None, connections_path=None, neuron_table=None,
             out_dir=None, dataset=None, trigger=None,
             write_bodyid=True, time_budget_s=900.0,
             allow_multi_visit=False):
    """Standalone corridor lane (plan §23.3/§23.5): G_type* -> shared W ->
    per-source drain -> validate loop -> PER-WALK subgraph enumeration
    (§22 share-consumption) -> route_scope_paths/ exports.

    Returns True on success (lane-namespace outputs only). Raises
    CorridorRefusal (after writing a status='refused' provenance) when the
    run's universe cannot be reproduced. Reads the run folder's
    parameters.txt (spaced keys) when run_dir is given; direct mode
    (run_dir=None) requires explicit asked/max_interlayer/sources/targets.
    ``max_interlayer`` is the run's own knob; the lane's hop bound is
    max_interlayer + 1."""
    import json
    import os as _os
    import sys as _sys
    import time
    from pathlib import Path

    # ONE CLOCK (2026-10-11, plan F2): t0/deadline are time.monotonic().
    # The previous t0 = time.time() mixed an EPOCH-scale deadline with the
    # monotonic checks in CorridorDrain/validate_and_fill/per-path
    # consumption — those could never fire (monotonic is uptime-scale), so
    # the wall-clock budget only bit in the post-validate walk loop. The
    # measured victim: the MCNS store-basis lane burned 4,471s against its
    # 900s budget with the entire overrun inside drain/validate.
    t0 = time.monotonic()
    wall0 = time.time()
    # an EXPLICIT out_dir owns the whole lane layout, plan.json included
    # (2026-10-11 review fix: plan_dir preferred run_dir even when the
    # caller routed outputs elsewhere — scratch/gate regens then polluted
    # the exported run folder's data_details/)
    explicit_out = out_dir is not None
    phases: dict = {}
    _dbg = bool(_os.environ.get("CORRIDOR_DEBUG"))

    def _phase(label):
        phases[label] = round(time.monotonic() - t0, 1)
        if _dbg:
            print(f"[corridor {label}] {phases[label]}s",
                  file=_sys.stderr, flush=True)

    deadline = (t0 + float(time_budget_s)
                if time_budget_s else None)
    repo = Path(__file__).resolve().parents[1]

    # ---- resolve run parameters -------------------------------------
    kv = read_run_params(run_dir) if run_dir else {}
    trigger = trigger or ("in-run" if fnc is not None else
                          "run-folder" if run_dir else "driver-direct")
    out = Path(out_dir) if out_dir else (
        Path(run_dir) / "route_scope_paths" if run_dir else None)
    if out is None:
        raise CorridorRefusal("no output directory (run_dir or out_dir)")

    def _refuse(reason):
        out.mkdir(parents=True, exist_ok=True)
        (out / "route_provenance.json").write_text(json.dumps(
            {"status": "refused", "reason": str(reason),
             "corridor_mode": corridor_mode, "trigger": trigger},
            indent=2, default=str), encoding="utf-8")
        (out / "README.md").write_text(
            "# Corridor lane (route_scope_paths)\n\n"
            f"NOT generated — refused: {reason}\n", encoding="utf-8")
        raise CorridorRefusal(reason)

    asked = int(asked if asked is not None
                else _parse_int(kv.get("min synapse number"), 1) or 1)
    max_il = int(max_interlayer if max_interlayer is not None
                 else _parse_int(kv.get("max interlayer"), 4))
    bound = max_il + 1
    if max_il < 0:
        _refuse(f"max interlayer {max_il} is negative")
    basis = str(kv.get("weight basis", "synapse") or "synapse").lower()
    if basis.startswith("connection_ratio"):
        _refuse("ratio-basis runs are not supported this round "
                "(corridor UBs are synapse-weight bottlenecks)")
    # F-PF-004 (audit 2026-10-13): parse the AND ratio as a FLOAT — the
    # old _parse_int turned a fractional threshold like '0.01' into the
    # default 0 and the dual-threshold refusal never fired, so the lane
    # searched a graph violating the parent's own predicate. A non-empty
    # value that is not a number refuses too (never silently zero).
    _and_raw = str(kv.get("min connection ratio") or "").strip()
    if _and_raw:
        _and_val = _parse_float(_and_raw)
        if _and_val is None:
            _refuse(f"min connection ratio {_and_raw!r} is not a number "
                    "(refusing rather than treating it as zero)")
        if _and_val > 0:
            _refuse("dual-threshold (AND mode) runs cannot be reproduced "
                    "by the synapse-corridor universe")
    separate_hemi = _parse_bool(kv.get("separate hemispheres"), False) \
        if not fnc else bool(getattr(fnc, "separate_hemispheres", False))
    hemi_filter = (str(getattr(fnc, "hemisphere_filter", None)
                       or kv.get("hemisphere filter", "both") or "both")
                   .strip().lower())
    exclude_intra = _parse_bool(kv.get("exclude intra-type connections"),
                                False) \
        if not fnc else bool(getattr(
            fnc, "exclude_intra_type_connections", False))
    drop_untyped = (_parse_bool(kv.get("drop untyped"), True)
                    if not fnc
                    else bool(getattr(fnc, "drop_untyped", True)))
    dataset = str(dataset or kv.get("dataset", "") or
                  (getattr(fnc, "dataset", "") if fnc is not None else "")
                  or "").strip()

    query_prefix_src = query_prefix_tgt = False
    if source_types is not None:
        src_tokens = list(source_types)
    else:
        # _parse_query_tokens returns a MODE STRING ('equal'/'startswith');
        # unpack to the boolean the lane means — the bare assignment made
        # 'equal' truthy and silently ran every plain run-folder query in
        # prefix mode (same unrequested-member family as F-PF-005)
        _src_mode, src_tokens = _parse_query_tokens(
            kv.get("source neurons"))
        query_prefix_src = _src_mode == "startswith"
    if target_types is not None:
        tgt_tokens = list(target_types)
    else:
        _tgt_mode, tgt_tokens = _parse_query_tokens(
            kv.get("target neurons"))
        query_prefix_tgt = _tgt_mode == "startswith"
    if not src_tokens or not tgt_tokens:
        _refuse("source/target neurons could not be read from the run")

    # ---- resolve connection + neuron tables --------------------------
    local_pair = (None, None)
    if fnc is not None and hasattr(fnc, "_local_connectome_table_paths"):
        try:
            local_pair = fnc._local_connectome_table_paths() or (None, None)
        except Exception:
            local_pair = (None, None)
    conn_path = None
    for cand in (connections_path,
                 local_pair[0] if isinstance(local_pair, tuple) else None,
                 (Path(getattr(fnc, "cache_folder", "")) / "connections.parquet"
                  if getattr(fnc, "cache_folder", None) else None)):
        if cand and Path(cand).exists():
            conn_path = Path(cand)
            break
    neuron_path = None
    for cand in (neuron_table,
                 local_pair[1] if isinstance(local_pair, tuple) else None):
        if cand and Path(cand).exists():
            neuron_path = Path(cand)
            break
    if conn_path is None and dataset:
        safe = dataset.replace(":", "_").replace(".", "_")
        cand = repo / "cache" / safe / "connections.parquet"
        if cand.exists():
            conn_path = cand
    if neuron_path is None and dataset:
        safe = dataset.replace(":", "_").replace(".", "_")
        folder = repo / "datasets" / safe
        for name in (f"{safe}_allneurons_neuron_df.parquet",
                     f"{safe}_allneurons_neuron_df.csv"):
            if (folder / name).exists():
                neuron_path = folder / name
                break
    if conn_path is None:
        _refuse("connection table not resolvable (no local release, no "
                "connection cache); pass connections_path explicitly")
    if neuron_path is None:
        _refuse("neuron table not resolvable; pass neuron_table explicitly")

    # ---- P1.3: validation basis resolution (before the universe scan) --
    # Shortest runs validate against the run's own shortest-DAG discovery
    # store when present (min-hop-restricted scope, refill's
    # _load_shortest_store contract: layer_weights = [(label,
    # {(pre,post): w})]); absent/unreadable ⇒ lossless prune fallback with
    # an explicit disclosure (the storeless path keeps working).
    path_mode = str(kv.get("path_mode", "all") or "all").strip().lower()
    store_universe = None
    validation_basis = ("lossless_prune" if path_mode != "shortest"
                        else "lossless_prune_fallback")
    if path_mode == "shortest" and run_dir:
        try:
            # VECTORIZED store read (plan F8): _load_shortest_store's
            # per-row Python dict build over the multimillion-row store
            # cost more wall than every other phase together (1,800s on
            # the MCNS store); this reader dedups deterministically in
            # polars (keep-first by sorted layer, last-wins within a
            # layer — identical semantics) and counts the cross-layer
            # weight conflicts the policy resolves.
            store_universe, store_weight_conflicts = \
                _read_store_universe(Path(run_dir))
        except Exception:
            if _dbg:
                raise  # debug runs surface store-read failures verbatim
            store_universe = None
            store_weight_conflicts = 0
        if store_universe is not None and store_universe.height:
            _phase(f"store-loaded+dedup ({store_universe.height:,} rows)")
            # validation_basis is set to shortest_store_edge_union AFTER the
            # type join succeeds (post-tmap stage below)

    # ---- connection universe: collapse per-roi rows, cast ids once ---
    _scan = (pl.scan_csv(conn_path, infer_schema_length=10000,
                         schema_overrides={"bodyId_pre": pl.Utf8,
                                           "bodyId_post": pl.Utf8})
             if str(conn_path).lower().endswith(".csv")
             else pl.scan_parquet(conn_path))
    conn = (_scan
            .group_by(["bodyId_pre", "bodyId_post"])
            .agg(pl.col("weight").sum().alias("weight"))
            .with_columns(pl.col("bodyId_pre").cast(pl.Utf8),
                          pl.col("bodyId_post").cast(pl.Utf8))
            .collect())
    _phase(f"conn-collapsed ({conn.height:,} rows)")

    # ---- effective type map (run label-chain parity, refill recipe) ---
    try:
        from type_level_refill import (
            build_effective_type_map, _hemisphere_codes)
    except ImportError as exc:  # pragma: no cover - packaging failure
        _refuse(f"type_level_refill unavailable ({exc})")
    frame = None
    neuron_source = str(neuron_path) if neuron_path else None
    if fnc is not None and hasattr(fnc, "_refill_neuron_frame"):
        try:
            frame = fnc._refill_neuron_frame()
        except Exception:
            frame = None
        if frame is not None:
            neuron_source = "engine-resolved (_refill_neuron_frame)"
    if frame is None:
        if neuron_path is None:
            _refuse("neuron table not resolvable; pass neuron_table "
                    "explicitly")
        import pandas as pd
        frame = (pd.read_parquet(neuron_path)
                 if str(neuron_path).lower().endswith(".parquet")
                 else pd.read_csv(neuron_path))
        tcol = next((c for c in frame.columns
                     if c.lower() in ("type", "cell_type")), None)
        if tcol is None or "bodyId" not in frame.columns:
            _refuse(f"neuron table {neuron_path} lacks bodyId/type columns")
        frame = frame.rename(columns={tcol: "type"})
        frame["bodyId"] = frame["bodyId"].astype(str)
    frame = frame.copy()
    frame["bodyId"] = frame["bodyId"].astype(str)
    if hemi_filter in ("left", "right"):
        # fetch-time neuron rule (coana _apply_hemisphere_suffix_to_neuron_df):
        # keep only that side + unlabeled ('U') bodies.
        try:
            codes = _hemisphere_codes(frame)
        except Exception as exc:
            _refuse(f"hemisphere filter not reproducible ({exc})")
        keep_code = hemi_filter[0].upper()
        frame = frame[codes.isin([keep_code, "U"])]
    label_mapper = getattr(fnc, "label_mapper", None) if fnc else None
    if label_mapper is None:
        # subprocess mode: the run's recorded mapping file expands
        # custom-label query tokens (e.g. '5th-LNv') into the type map.
        # F-PF-010 (audit 2026-10-13): a RECORDED map that is absent must
        # refuse — falling back to native labels silently changes the
        # settled group membership, endpoint enrollment and every
        # exported label of the lane.
        _mf = (kv.get("custom mapping file")
               or "").strip()
        if _mf and not Path(_mf).exists():
            _refuse(f"recorded mapping file {_mf!r} does not exist — "
                    "refusing rather than silently falling back to "
                    "native labels")
        if _mf:
            try:
                from comparison.label_mapper import LabelMapper
                label_mapper = LabelMapper(overall_mapping_json=_mf)
            except Exception as _lm_exc:
                _refuse(f"recorded mapping file not loadable "
                        f"({_lm_exc})")
    type_map = build_effective_type_map(
        frame, dataset=dataset or "unknown",
        label_mapper=label_mapper, separate_hemispheres=separate_hemi)
    if not type_map:
        _refuse("effective type map is empty (neuron table unusable)")
    tmap = pl.DataFrame({"bid": list(type_map.keys()),
                         "t": list(type_map.values())},
                        schema={"bid": pl.Utf8, "t": pl.Utf8})
    body2type = dict(type_map)

    src_labels, src_bodies_q = _match_type_labels(
        tmap["t"].to_list(), src_tokens, body2type,
        prefix=query_prefix_src)
    tgt_labels, tgt_bodies_q = _match_type_labels(
        tmap["t"].to_list(), tgt_tokens, body2type,
        prefix=query_prefix_tgt)
    if not src_labels or not tgt_labels:
        _refuse(f"query tokens resolved to no types "
                f"(sources kept {sorted(src_labels)}, "
                f"targets kept {sorted(tgt_labels)})")
    src_types, tgt_types = sorted(src_labels), sorted(tgt_labels)

    # F-PF-005 (audit 2026-10-13): preserve the parent's EXACT endpoint
    # membership. (i) Query tokens that are explicit bodyIds restrict the
    # endpoint body sets to themselves instead of their whole types.
    # (ii) Run-folder mode additionally reads the parent's enrollment
    # CSVs when present — source_neurons.csv and the Checked rows of
    # target_neurons.csv are the parent's resolved enrollment (partially
    # Checked targets included), which token re-expansion cannot
    # reproduce. Both restrictions intersect; the provenance discloses
    # the mode.
    src_enrollment = ("bodyids" if src_bodies_q else
                      "prefix" if query_prefix_src else "types")
    tgt_enrollment = ("bodyids" if tgt_bodies_q else
                      "prefix" if query_prefix_tgt else "types")
    src_keep, tgt_keep = set(src_bodies_q), set(tgt_bodies_q)
    if run_dir:
        try:
            _src_csv = pl.read_csv(
                Path(run_dir) / "source_neurons.csv",
                infer_schema_length=0)
            if "bodyId" in _src_csv.columns:
                src_keep |= set(
                    _src_csv["bodyId"].drop_nulls().cast(pl.Utf8)
                    .str.strip_chars().to_list())
                src_enrollment += "+parent_csv"
        except Exception:
            pass
        try:
            _tgt_csv = pl.read_csv(
                Path(run_dir) / "target_neurons.csv",
                infer_schema_length=0)
            if "bodyId" in _tgt_csv.columns:
                if "Checked" in _tgt_csv.columns:
                    _checked = _tgt_csv.filter(
                        _tgt_csv["Checked"].cast(pl.Utf8)
                        .str.to_lowercase().is_in(["true", "1"]))
                else:
                    _checked = _tgt_csv
                tgt_keep |= set(
                    _checked["bodyId"].drop_nulls().cast(pl.Utf8)
                    .str.strip_chars().to_list())
                tgt_enrollment += "+parent_csv"
        except Exception:
            pass
    # A restriction applies whenever an explicit keep set exists (bodyId
    # tokens, or the parent's enrollment CSVs — which are the parent's
    # resolved query and must win over any token/prefix re-expansion,
    # F-PF-005). Empty keep sets mean "no explicit membership known" —
    # the type-level enrollment stands (audit fixture: token '10' of
    # type {10, 11} must never emit 11's edges).
    _src_restrict = bool(src_keep)
    _tgt_restrict = bool(tgt_keep)

    # ---- G_type* + lossless prune ------------------------------------
    # CANONICAL ROW ORDER (2026-10-11): polars group_by is a parallel
    # hash aggregate — its output order varies run to run, which leaks
    # into FastGraph edge-insertion order and the budgeted tie-drain's
    # tie-breaking (measured: the same run produced 25 vs 27 accepted
    # walks across regens). Sorting the universe canonically makes the
    # enumeration byte-reproducible; tie MEMBERSHIP at a class tau is
    # then stable too.
    if store_universe is not None:
        # post-tmap stage: type columns + asked tier (downstream consumers
        # — the per-pair edge cache and the scope pair lookup — key on
        # type pairs)
        su = (store_universe.filter(pl.col("weight") >= asked)
              .join(tmap.rename({"bid": "bodyId_pre",
                                 "t": "type_pre"}),
                    on="bodyId_pre", how="inner")
              .join(tmap.rename({"bid": "bodyId_post",
                                 "t": "type_post"}),
                    on="bodyId_post", how="inner"))
        if exclude_intra:
            su = su.filter(pl.col("type_pre") != pl.col("type_post"))
        if su.height:
            store_universe = su.sort(["bodyId_pre", "bodyId_post"])
            # F-PF-009 (audit 2026-10-13): the FLATTENED store union does
            # not carry the parent's per-target distance maps — the lane
            # enumerates ordinary bounded routes over the edge union, so
            # a route can be nonminimum-hop for its bodyId pair while the
            # old 'shortest_dag_store' name claimed shortest semantics.
            # Name the scope what it is; enforcing per-endpoint minimum
            # hops is a ruled follow-up (plan ...-audit-fixes.md D3-B).
            validation_basis = "shortest_store_edge_union"
        else:
            store_universe = None  # nothing survives — prune fallback
    if neuron_source and neuron_source.startswith("engine-resolved"):
        # the in-run engine frame has no file stats for the cache
        # signature — hash the REAL neuron table instead (the local
        # release table the engine frame mirrors); a stale string key
        # would make the in-run and CLI signatures disagree
        try:
            _conn_p, _neuron_p = (fnc._local_connectome_table_paths()
                                  or (None, None)) \
                if fnc is not None and hasattr(
                    fnc, '_local_connectome_table_paths') else (None, None)
            if _neuron_p and Path(_neuron_p).exists():
                neuron_source = _neuron_p
        except Exception:
            pass
    typed, pairs = _typed_universe_cached(
        conn_path, neuron_source, conn, tmap, asked, drop_untyped,
        exclude_intra, separate_hemi, hemi_filter, label_mapper, _phase)
    # CANONICAL ROW ORDER (2026-10-11): polars group_by is a parallel
    # hash aggregate — its output order varies run to run, which leaks
    # into FastGraph edge-insertion order and the budgeted tie-drain's
    # tie-breaking (measured: the same run produced 25 vs 27 accepted
    # walks across regens). Sorting the universe canonically makes the
    # enumeration byte-reproducible; tie MEMBERSHIP at a class tau is
    # then stable too.
    typed = typed.sort(["bodyId_pre", "bodyId_post"])
    pairs = pairs.sort(["type_pre", "type_post"])
    _phase(f"type-pairs ({pairs.height:,})")
    if exclude_intra:
        typed = typed.filter(pl.col("type_pre") != pl.col("type_post"))
        pairs = (typed.group_by("type_pre", "type_post")
                 .agg(pl.col("weight").sum().alias("w_sum"),
                      pl.col("weight").max().alias("w_max"),
                      pl.len().alias("n")))
    if typed.height == 0:
        _refuse(f"no typed edges at asked threshold {asked}")
    src_ids = tmap.filter(pl.col("t").is_in(src_types))["bid"].to_list()
    tgt_ids = tmap.filter(pl.col("t").is_in(tgt_types))["bid"].to_list()

    vis_adj, rel, ds_all, _dt_all = _type_adj(pairs, src_types, tgt_types,
                                              bound)

    # Lossless speed-up (§16 finding 1, inverted): the type-level
    # dS+1+dT<=bound filter is implied by the body-level hop-budget rule
    # (type dist <= body dist), so restricting the typed edges to
    # type-routable pairs BEFORE the body prune drops nothing — the
    # 4-pass body prune then runs on a far smaller table (71 s -> ~10 s
    # on the FAFB B4 anchor).
    rel_keys = rel.select(pl.concat_str(
        [pl.col("type_pre"), pl.lit("\x1f"), pl.col("type_post")])
        .alias("k"))
    typed_routed = typed.with_columns(pl.concat_str(
        [pl.col("type_pre"), pl.lit("\x1f"), pl.col("type_post")])
        .alias("k")).join(rel_keys, on="k", how="semi").drop("k")
    coana_mod = __import__("coana")  # also puts vispath on sys.path in
    # embedded contexts — import FastGraph only after this
    if store_universe is not None:
        # P1.3: shortest-DAG discovery store IS the universe (min-hop-
        # restricted explored scope; the engine's own shortest-scope note
        # wording applies). The lossless prune is skipped — the store's
        # per-layer frames are already the run's discovery result.
        loss = store_universe
        _phase(f"store-universe ({loss.height:,} rows)")
    else:
        _phase("pre-prune")
        loss_tables, _stats = coana_mod.prune_layers_hop_budget(
            [typed_routed], src_ids, tgt_ids, bound)
        loss = loss_tables[0]
        _phase(f"lossless-pruned ({loss.height:,} rows)")
    from vispath_pkg.fast_graph_core import FastGraph

    pair_edges = _PairEdges(loss)
    shared_W = widest_to_targets_fast(pairs, set(tgt_types), bound)
    limit = int(per_pair_length_budget)

    # ---- per-type support tables (route_paths_type.csv mirrors the
    # *_allpaths_type.csv column semantics: weights/ratios/min_weight/
    # min_ratio/nt_types/coverage) --------------------------------------
    type_membership = dict(zip(
        *(tmap.group_by("t").agg(pl.len().alias("n")).sort("t")
          .select("t", "n")).to_dict().values()))
    incoming = (conn.group_by("bodyId_post")
                .agg(pl.col("weight").sum().alias("w")))
    incoming = incoming.join(tmap, left_on="bodyId_post", right_on="bid")
    type_incoming = {}
    for t, w in zip(incoming["t"].to_list(), incoming["w"].to_list()):
        type_incoming[t] = type_incoming.get(t, 0) + float(w)
    # C1.1 (plan-corridor-alignment-and-viz-bases §2): the
    # threshold-CONDITIONED denominators come from the lane's own >= asked
    # universe — zero extra scans. `== plain` by construction when
    # asked <= 1 (round-13 "always emitted" parity with the refill lane).
    incoming_adj = (typed.group_by("bodyId_post")
                    .agg(pl.col("weight").sum().alias("w")))
    incoming_adj = incoming_adj.join(tmap, left_on="bodyId_post",
                                    right_on="bid")
    type_incoming_adj = {}
    for t, w in zip(incoming_adj["t"].to_list(),
                    incoming_adj["w"].to_list()):
        type_incoming_adj[t] = type_incoming_adj.get(t, 0) + float(w)
    type_nt = _type_nt_map(frame)

    walk_rows, body_rows = [], []
    provenance_classes, budgets = {}, {}
    scope_pairs = {}      # (a, b) -> {"mass": x, "paths": n, "walks": k}
    time_capped = False
    for s in src_types:
        drain = CorridorDrain(vis_adj, s, set(tgt_types), bound,
                              W=shared_W, deadline=deadline,
                              allow_multi_visit=allow_multi_visit)
        # Feasibility precheck (§22 cost control): a class with h below
        # the NEAREST-source type distance cannot receive candidates (no
        # length-h walk exists) — mark it infeasible so it cannot keep
        # the drain alive. ds_all[t] = min_s d(s,t) is a lower bound on
        # d(s,t), so the rule is exact for single-source runs and
        # conservative (never wrongly infeasible) for multi-source ones.
        classes = {}
        for t in tgt_types:
            d_lb = ds_all.get(t)
            for h in range(1, bound + 1):
                classes[((s, t), h)] = ClassBudget(
                    limit=limit,
                    mh_prov=max(1, round(
                        float(top_m_base) * (float(length_decay) ** (h - 1)))),
                    feasible=(d_lb is not None and h >= d_lb))
        s_bodies = [b for b in src_ids if body2type.get(b) == s
                    and (not _src_restrict or b in src_keep)]
        prov = validate_and_fill(
            drain, classes,
            lambda walk, cap, _s=s: _realize_on_loss(
                pair_edges, walk, s_bodies,
                {b for b in tgt_ids if body2type.get(b) == walk[-1]}, cap),
            max_pulls=drain.max_pops, deadline=deadline)
        time_capped = time_capped or prov.get("time_capped", False)
        _phase(f"validated source {s}")
        provenance_classes.update(prov["classes"])
        budgets.update(classes)

        for t in tgt_types:
            if deadline is not None and time.monotonic() > deadline:
                time_capped = True
                break
            t_bodies = [b for b in tgt_ids if body2type.get(b) == t
                        and (not _tgt_restrict or b in tgt_keep)]
            # per-WALK enumeration (§22): each accepted walk gets the
            # class's REMAINING share in UB-descending acceptance order —
            # a fat walk can consume the whole budget, a thin one leaves
            # room. The walk's own hop-pair subgraph admits (almost) only
            # its own realizations, so the StrongestFirst budget is not
            # starved by other classes' stronger paths (MCNS R1-R6->Tm3:
            # per-pair cutoff=h enumeration emitted 0 of 100 counted h3
            # realizations — the 2-hop tie mass ate every budget).
            for h in range(1, bound + 1):
                cb = classes[((s, t), h)]
                if not cb.walks:
                    continue
                off_len = off_plan = 0
                remaining = cb.limit
                emitted_total = 0
                for rank, (ub, walk) in enumerate(cb.walks, start=1):
                    if remaining <= 0:
                        break
                    if deadline is not None and time.monotonic() > deadline:
                        time_capped = True
                        break
                    # PER-PAIR CACHE + PER-POSITION PLACEMENT (2026-10-11,
                    # plan F5): each hop's rows come from the shared
                    # _PairEdges cache (canonical order, dedup'd) and are
                    # placed ONLY at their layer position. The previous
                    # all-layer build carried h× the necessary edges —
                    # provably dead for distinct walks (a distinct walk's
                    # pair set is a simple path graph, so every reachable
                    # layered path's type sequence IS the walk) — and,
                    # under the multi-visit knob, admitted pair-set
                    # PERMUTATION paths whose masses aggregated under
                    # wrong hop labels. Backward-alive pruning runs on
                    # the tuples with sets PER POSITION (correct for
                    # repeated-type walks too; the old type-keyed dict
                    # was distinct-only) and stays lossless: a simple
                    # path is a walk, so its suffix always satisfies
                    # walk-level reachability.
                    pair_rows = [pair_edges.tuples(walk[i], walk[i + 1])
                                 for i in range(h)]
                    alive = [set() for _ in range(h + 1)]
                    alive[h] = set(t_bodies)
                    for i in range(h - 1, -1, -1):
                        nxt = alive[i + 1]
                        alive[i] = {u for u, v, _w in pair_rows[i]
                                    if v in nxt}
                    kept_rows = []
                    for i in range(h):
                        for u, v, w in pair_rows[i]:
                            if v not in alive[i + 1]:
                                continue
                            if i == 0 and u not in s_bodies:
                                continue
                            kept_rows.append((i, u, v, w))
                    # P1.2 LAYERED (time-expanded) enumeration: nodes
                    # "bodyId\x1f{i}" per layer, edges only (u,i)->(v,i+1)
                    # for the walk's hop-i pair. Simple paths here ARE the
                    # exact length-h layered realizations realize_count
                    # counts; per-position placement additionally forces
                    # the walk's own pair SEQUENCE (permutation reuse of
                    # the walk's pair set is structurally impossible).
                    # raw weights keyed by RAW pair for output assembly.
                    raw_w = {}
                    layered_edges = []
                    for i, u, v, w in kept_rows:
                        raw_w[(u, v)] = w
                        layered_edges.append(
                            (f"{u}\x1f{i}", f"{v}\x1f{i + 1}", w))
                    g = FastGraph()
                    g.build_from_edges(layered_edges)
                    adj_w = g.adj
                    # F-PF-006: ONE path beyond the share (bounded: the
                    # share + 1) so the tie-witness peek below can see
                    # what the share cut — the old budget == share made
                    # the generator stop exactly at the share and every
                    # boundary read as unverifiable.
                    budget = remaining + 1
                    for _attempt in range(1):  # layered: no retry needed
                        st: dict = {}
                        # LAZY consumption with a per-item deadline check:
                        # a tau tie-drain can emit millions of paths inside
                        # one call (w>=1 ties), and list() would block past
                        # the budget (the in-hook bound-5 anchor stalled
                        # >20 min in exactly that drain)
                        paths = []
                        ties_truncated = False
                        ties_next_bn = None
                        _nxt_repeats = False
                        # BODY-SIMPLICITY (F-PF-003, audit 2026-10-13):
                        # the engine's no-repeated-nodes rule operates on
                        # LAYERED ids (body\x1f{i}), so a multi-visit
                        # walk's layered enumeration admitted the same
                        # bodyId at several layers — and the old external
                        # filter rejected those walks AFTER the engine's
                        # budget had already been spent on them, so one
                        # invalid stronger walk could starve a valid
                        # weaker realization (emitted 0 despite a
                        # successful validation). raw_body_simple moves
                        # the simplicity predicate INTO the generator:
                        # every yield is a body-simple path and the
                        # budget/tau/tie-drain semantics apply to exactly
                        # those.
                        multi_visit = len(set(walk)) < len(walk)
                        layered_rejected_here = 0
                        src_l = [f"{b}\x1f0" for b in s_bodies]
                        tgt_l = {f"{b}\x1f{h}" for b in t_bodies}
                        gen = g.find_paths_strongest_first(
                            src_l, tgt_l, h, budget=budget, stats=st,
                            raw_body_simple=multi_visit)
                        for p in gen:
                            if (deadline is not None
                                    and time.monotonic() > deadline
                                    and len(paths) >= 1):
                                time_capped = True
                                break
                            paths.append(p)
                            if len(paths) >= remaining:
                                # §21 heavy-tail ruling: the share
                                # truncates the walk's tie mass. The
                                # generator is LAZY — breaking here stops
                                # the engine's tie-drain before it can
                                # explode (the MCNS h5 layered tau-tie
                                # mass burned 26 min when fully drained).
                                # F-PF-006: ONE extra lazy pull witnesses
                                # whether an equal-strength route was
                                # really cut (confirmed) or not (the
                                # flag then clears); no pull at all
                                # stays potential.
                                ties_truncated = True
                                nxt = next(gen, None)
                                if nxt is not None:
                                    ties_next_bn = min(
                                        adj_w[nxt[i]][nxt[i + 1]]
                                        for i in range(h))
                                    _nxt_repeats = (
                                        len(set(n.split("\x1f")[0]
                                                for n in nxt))
                                        != len(nxt))
                                break
                        kept, class_body_rows = [], []
                        # engine type-path semantics: a hop's mass counts
                        # each DISTINCT realizing bodyId edge ONCE (not once
                        # per path — shared hubs would double-count)
                        hop_edges = [dict() for _ in range(h)]
                        node_seen = [set() for _ in range(h + 1)]
                        for p in paths:
                            raw = [n.split("\x1f")[0] for n in p]
                            kept.append(raw)
                            for i in range(h):
                                hop_edges[i][(raw[i], raw[i + 1])] = \
                                    float(adj_w[p[i]][p[i + 1]])
                                node_seen[i].add(raw[i])
                            node_seen[h].add(raw[h])
                            if write_bodyid:
                                class_body_rows.append({
                                    "path": "->".join(
                                        str(b) for b in raw),
                                    "weights": ",".join(_fmt_mass(
                                        raw_w[(raw[i], raw[i + 1])])
                                        for i in range(h)),
                                    "length": h, "route_h": h,
                                    "applied_tau": None,
                                    "plan_rank": rank})
                        break
                    tau = st.get("tau")
                    if tau is None and kept:
                        # (fallback: with raw_body_simple the engine tau
                        # already describes the body-simple drain; this
                        # only fires when the engine never opened a tau)
                        tau = float(min(
                            min(raw_w[(p[i], p[i + 1])]
                                for i in range(h)) for p in kept))
                    if ties_truncated:
                        # F-PF-006 classification: CONFIRMED when the
                        # peeked next realization ties the class tau;
                        # cleared when it is strictly weaker or the
                        # generator finished under share+1 (the share
                        # cut nothing); POTENTIAL when the generator
                        # bit its own budget or a deadline cut the peek,
                        # or the peeked layered yield is a repeated-body
                        # walk (multi-visit — not a route).
                        _nxt_is_route = (
                            ties_next_bn is not None
                            and not (multi_visit and _nxt_repeats))
                        if _nxt_is_route and tau is not None and \
                                abs(ties_next_bn - tau) <= 1e-9 * max(
                                    1.0, abs(tau)):
                            cb.ties_truncated_confirmed += 1
                        elif ties_next_bn is not None and _nxt_is_route:
                            ties_truncated = False
                        elif time_capped or st.get("budget_bitten"):
                            cb.ties_truncated_potential += 1
                        else:
                            ties_truncated = False
                        if ties_truncated:
                            cb.ties_truncated += 1
                    cb.layered_rejected += layered_rejected_here
                    for r in class_body_rows:
                        r["applied_tau"] = tau
                    # ---- per-WALK type row (§23.5: one row per validated
                    # walk; *_allpaths_type.csv column order + additive) --
                    hop_mass = [sum(e.values()) for e in hop_edges]
                    ratios, ratios_adj = [], []
                    for i in range(h):
                        denom = type_incoming.get(walk[i + 1], 0.0)
                        ratios.append(hop_mass[i] / denom
                                      if denom > 0 else 0.0)
                        denom_adj = type_incoming_adj.get(
                            walk[i + 1], 0.0)
                        ratios_adj.append(hop_mass[i] / denom_adj
                                          if denom_adj > 0 else None)
                    finite_adj = [r for r in ratios_adj
                                  if r is not None]
                    cov = [
                        "{}/{}".format(
                            len(node_seen[i]),
                            type_membership.get(walk[i],
                                                len(node_seen[i])))
                        for i in range(h + 1)]
                    nts = [type_nt.get(walk[i], "unknown")
                           for i in range(h)]
                    if not kept:
                        # validated but UNEMITTED (multi-visit deep-h walk
                        # whose hops compose stronger shorter chains —
                        # disclosed via off_len): keep it in plan.json/
                        # provenance, not as an all-zero data row
                        emitted_total += 0
                    else:
                        walk_rows.append({
                        "path": "->".join(walk),
                        "weights": "[" + ", ".join(
                            _fmt_mass(m) for m in hop_mass) + "]",
                        "ratios": "[" + ", ".join(
                            repr(r) for r in ratios) + "]",
                        "ratios_adj": (
                            "[" + ", ".join(
                                repr(r) for r in ratios_adj) + "]"
                            if len(finite_adj) == len(ratios_adj)
                            and ratios_adj else ""),
                        "min_weight": (int(min(hop_mass))
                                       if hop_mass else None),
                        "min_ratio": min(ratios) if ratios else None,
                        "min_ratio_adj": (min(finite_adj)
                                          if finite_adj else None),
                        "length": h,
                        "nt_types": "[" + ", ".join(
                            f'"{x}"' for x in nts) + "]",
                        "coverage": "[" + ", ".join(cov) + "]",
                        "applied_tau": tau,
                        "route_h": h, "route_ub": float(ub),
                        "route_floor_h": (None if cb.floor is None
                                          else float(cb.floor)),
                        "plan_rank": rank,
                        "realized_paths": len(kept)})
                    for i in range(h) if kept else ():
                        key = (walk[i], walk[i + 1])
                        rec = scope_pairs.setdefault(
                            key, {"edges": {}, "paths": 0, "walks": 0})
                        # merge DISTINCT edges across walks sharing the hop
                        rec["edges"].update(hop_edges[i])
                        rec["paths"] += len(kept)
                        rec["walks"] += 1
                    body_rows.extend(class_body_rows)
                    emitted_total += len(kept)
                    remaining -= len(kept)
                # actual rows appended (tie-drain can exceed the share —
                # engine semantics: every path at the cutoff tau)
                cb.emitted = emitted_total
                cb.off_len = off_len
                cb.off_plan = off_plan

    # ---- exports (§23.5; lane namespace only) -------------------------
    runtime = round(time.time() - wall0, 2)
    for (pair, h), cb in budgets.items():
        entry = provenance_classes.get(f"{pair[0]}->{pair[1]}@h{h}")
        if entry is not None:
            entry["emitted"] = cb.emitted
            entry["off_plan"] = cb.off_plan
            entry["off_len"] = cb.off_len
            entry["layered_rejected"] = cb.layered_rejected
            # F-PF-006: the audit found the tie flags counted during the
            # per-class enumeration but never copied into the serialized
            # snapshot — a hard share cut at the boundary serialized as
            # ties_truncated 0. Copy the FINAL state.
            entry["ties_truncated"] = cb.ties_truncated
            entry["ties_truncated_confirmed"] = \
                cb.ties_truncated_confirmed
            entry["ties_truncated_potential"] = \
                cb.ties_truncated_potential
    out.mkdir(parents=True, exist_ok=True)
    # one row per VALIDATED WALK in *_allpaths_type.csv column order +
    # the additive lane columns (per-path detail lives in the bodyId csv)
    walk_rows.sort(key=lambda r: (r["path"], r["route_h"],
                                  r["plan_rank"]))
    type_schema = {"path": pl.Utf8, "weights": pl.Utf8, "ratios": pl.Utf8,
                   "ratios_adj": pl.Utf8,
                   "min_weight": pl.Int64, "min_ratio": pl.Float64,
                   "min_ratio_adj": pl.Float64,
                   "length": pl.Int64, "nt_types": pl.Utf8,
                   "coverage": pl.Utf8, "applied_tau": pl.Float64,
                   "route_h": pl.Int64, "route_ub": pl.Float64,
                   "route_floor_h": pl.Float64, "plan_rank": pl.Int64,
                   "realized_paths": pl.Int64}
    pl.DataFrame(walk_rows, schema=type_schema).write_csv(
        out / "route_paths_type.csv")
    if write_bodyid:
        pl.DataFrame(body_rows, schema={
            "path": pl.Utf8, "weights": pl.Utf8, "length": pl.Int64,
            "applied_tau": pl.Float64, "route_h": pl.Int64,
            "plan_rank": pl.Int64}).write_csv(
            out / "route_paths_bodyId.csv")
    # route_connection_type.csv (C1.3): mirror-column family aligned with
    # the refill's _TYPE_MIRROR_COLUMNS so mirror-style loaders consume it
    # unchanged, then the corridor-specific tail.
    if scope_pairs:
        pair_lookup = {(a, b): (wsum, wmax, n) for a, b, wsum, wmax, n in
                       zip(pairs["type_pre"], pairs["type_post"],
                           pairs["w_sum"], pairs["w_max"], pairs["n"])}
        conn_rows = []
        for (a, b) in sorted(scope_pairs):
            rec = scope_pairs[(a, b)]
            wsum, wmax, n = pair_lookup.get((a, b), (None, None, None))
            realized = round(sum(rec["edges"].values()), 2)
            denom = type_incoming.get(b, 0.0)
            denom_adj = type_incoming_adj.get(b, 0.0)
            conn_rows.append({
                # _TYPE_MIRROR_COLUMNS family (type_level_refill.py:1758)
                "type_pre": a, "type_post": b,
                "weight": wsum,
                "nt_type": type_nt.get(b, ""),
                "connection_ratio": (round(realized / denom, 6)
                                     if denom > 0 else None),
                "conn_layer": "route_scope",
                "connection_ratio_adj": (round(realized / denom_adj, 6)
                                         if denom_adj > 0 else None),
                # corridor tail
                "weight_max": wmax, "pair_count": n,
                # F-PF-008 (audit 2026-10-13): `weight` is the WHOLE
                # admitted type-pair mass while connection_ratio divides
                # the SELECTED-realization mass — two support sets behind
                # one row. The explicit aliases name each quantity so no
                # reader has to guess which numerator a number uses.
                "universe_pair_mass": wsum,
                "selected_support_mass": realized,
                "realized_mass": realized,
                "realizing_paths": rec["paths"],
                "used_by_walks": rec["walks"]})
        pl.DataFrame(conn_rows).write_csv(
            out / "route_connection_type.csv")
    live = {k: v for k, v in provenance_classes.items()
            if v["accepted"] or v["pulled"]}
    # serialized RoutePlan (§5/§23.5): the validated plan is content-hashed;
    # a later deep-dive can re-derive scope from it without re-draining
    plan = {
        "asked": asked, "bound": bound, "sources": src_types,
        "targets": tgt_types, "knobs": {
            "top_m_base": top_m_base, "length_decay": length_decay,
            "per_pair_length_budget": limit},
        "classes": {k: {"floor": budget.floor,
                        "floor_next": budget.floor_next,
                        "stop": budget.stop, "mh_prov": budget.mh_prov,
                        "walks": [[ub, list(w)] for ub, w in
                                  budget.walks]}
                    for (budget, k) in sorted(
                        ((budgets[((pair, h))],
                          f"{pair[0]}->{pair[1]}@h{h}")
                         for (pair, h) in budgets
                         if budgets[((pair, h))].walks),
                        key=lambda kv: kv[1])},
    }
    import hashlib
    plan_hash = hashlib.md5(
        json.dumps(plan, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    plan["plan_hash"] = plan_hash
    try:
        plan_dir = (out if (explicit_out or not run_dir)
                    else Path(run_dir)) / "data_details" / "route_plan"
        plan_dir.mkdir(parents=True, exist_ok=True)
        (plan_dir / "plan.json").write_text(
            json.dumps(plan, indent=2, default=str), encoding="utf-8")
    except OSError:
        pass  # plan.json is best-effort; the lane's primary exports stand

    # gap partition (§6 step 7): lane walks vs the run's own type paths
    # (main at its applied threshold; refill when it produced records)
    gap = None
    if run_dir:
        try:
            main_csv = next(
                (Path(run_dir)).glob("*_allpaths_type.csv"), None)
            lane_keys = {r["path"] for r in walk_rows}
            if main_csv is not None:
                main_df = pl.read_csv(main_csv)
                main_col = ("path" if "path" in main_df.columns
                            else main_df.columns[0])
                main_keys = set(main_df[main_col].drop_nulls().to_list())
                refill_csv = (Path(run_dir) / "type_level_refill"
                              / "refill_paths_type.csv")
                refill_keys = set()
                if refill_csv.exists():
                    rf = pl.read_csv(refill_csv)
                    rf_col = ("path" if "path" in rf.columns
                              else rf.columns[0])
                    refill_keys = set(rf[rf_col].drop_nulls().to_list())
                gap = {
                    "main_paths": len(main_keys),
                    "corridor_walks": len(lane_keys),
                    "both": len(lane_keys & main_keys),
                    "main_only": len(main_keys - lane_keys),
                    "corridor_only": len(lane_keys - main_keys),
                    "refill_paths": len(refill_keys),
                    "corridor_vs_refill_both": (
                        len(lane_keys & refill_keys)
                        if refill_keys else None),
                    "corridor_vs_refill_only": (
                        len(lane_keys - refill_keys)
                        if refill_keys else None),
                }
        except Exception:
            gap = None
    provenance = {
        "status": "complete", "corridor_mode": corridor_mode,
        "trigger": trigger, "asked": asked, "bound": bound,
        "max_interlayer": max_il, "dataset": dataset,
        "separate_hemispheres": separate_hemi,
        "hemisphere_filter": hemi_filter,
        "exclude_intra_type_connections": exclude_intra,
        "label_mapper": label_mapper is not None,
        # F-PF-005: how the endpoint bodies were enrolled — 'types' /
        # 'prefix' (token expansion), 'bodyids' (explicit bodyId tokens),
        # '+parent_csv' when the run's own enrollment CSVs restricted the
        # endpoints (source_neurons.csv + Checked target_neurons.csv)
        "enrollment": {"sources": src_enrollment, "targets": tgt_enrollment,
                       "source_bodies": len(src_keep) or None,
                       "target_bodies": len(tgt_keep) or None},
        "sources": src_types, "targets": tgt_types,
        "connections": str(conn_path), "neuron_table": neuron_source,
        "knobs": {"top_m_base": top_m_base,
                  "length_decay": length_decay,
                  "per_pair_length_budget": limit,
                  "allow_multi_visit": allow_multi_visit},
        "runtime_s": runtime,
        "phases_s": phases,
        "store_dedup": ("ROI-slice sum per layer; keep-first by sorted "
                        "layer"
                        if validation_basis == "shortest_store_edge_union"
                        else None),
        "store_weight_conflicts": (store_weight_conflicts
                                   if validation_basis
                                   == "shortest_store_edge_union" else None),
        "time_capped": time_capped,
        "time_budget_s": time_budget_s,
        "plan_hash": plan_hash,
        "path_mode": path_mode,
        "validation_basis": validation_basis,
        "validated_walks": len(walk_rows),
        "bodyId_paths": len(body_rows),
        "scope_pairs": len(scope_pairs),
        "gap_partition": gap,
        "classes": dict(sorted(provenance_classes.items())),
        "live_classes": sorted(live.keys()),
    }
    (out / "route_provenance.json").write_text(json.dumps(
        provenance, indent=2, default=str), encoding="utf-8")
    floors = [f"{k}: floor={v['floor']} floor_next={v['floor_next']} "
              f"stop={v['stop']} accepted={v['accepted']} "
              f"emitted={v['emitted']} off_plan={v['off_plan']} "
              f"off_len={v['off_len']}"
              for k, v in sorted(live.items())]
    (out / "README.md").write_text(
        "# Corridor lane (route_scope_paths)\n\n"
        "Standalone route-scoped pass at the ASKED threshold; never merged "
        "into allpaths (plan-corridor-lane.md §23.5).\n\n"
        f"- trigger: {trigger} (mode `{corridor_mode}`, asked {asked}, "
        f"bound {bound})\n"
        f"- validated walks (type rows): {len(walk_rows)} over "
        f"{len(scope_pairs)} scope type-pairs; bodyId paths: "
        f"{len(body_rows)}\n"
        f"- runtime: {runtime}s\n"
        + (f"- validation basis: {validation_basis}"
           + (" (the run's shortest-DAG discovery store, FLATTENED to "
              "an edge union — routes are bounded routes within that "
              "union, NOT guaranteed minimum-hop per bodyId pair; "
              "per-layer ROI slices are summed to the physical weight "
              "and layers deduplicated keep-first by sorted layer; "
              "cross-layer weight disagreements after summation are "
              "disclosed as store_weight_conflicts)"
              if validation_basis == "shortest_store_edge_union" else
              " (lossless hop-budget prune)"
              if validation_basis == "lossless_prune" else
              " (store absent — lossless-prune FALLBACK, not min-hop-"
              "restricted)") + "\n\n")
        + "Per-class disclosure (F-PF-002: floor = lowest accepted walk UB "
        "and floor_next = first skipped candidate's UB are CANDIDATE-BOUND "
        "quantities — optimistic upper bounds from per-hop max edges, not "
        "the strengths of realized/omitted neuron routes; the lane is "
        "upper-bound-guided representative exploration, not a "
        "strongest-actual-route certificate):\n\n"
        + ("- route_connection_type.csv carries TWO support sets, named: "
           "`weight`/`universe_pair_mass` is the WHOLE admitted type-pair "
           "mass; `selected_support_mass`/`realized_mass` is the mass on "
           "the lane's emitted witnesses (the heatmap/network draw the "
           "latter; the ratios divide it). Both are UNIONS of edges over "
           "realizations — a large aggregate never implies one neuron "
           "path carrying all of it.\n\n")
        + ("\n".join(f"- {f}" for f in floors) if floors
           else "- no validated corridors (all classes dry)\n")
        + (f"\n- TIME-CAPPED at {time_budget_s}s — enumeration was cut "
           "early; remaining classes/walks are absent by budget, not by "
           "absence.\n" if time_capped else ""),
        encoding="utf-8")
    _write_mirror_visualizations(out, src_types=src_types,
                                 tgt_types=tgt_types)
    return True


def _write_mirror_visualizations(out, src_types=(), tgt_types=()):
    """V1.4 (plan-corridor-alignment-and-viz-bases §4.4): the mirror
    heatmap + network from route_connection_type.csv — the refill's own
    recipe (coana _vp + create_heatmap/create_network). Best-effort:
    failures leave the CSVs untouched (never fatal to the lane).
    src_types/tgt_types drive the network's node roles (an edge-list
    input has no path-position roles — without them the legend reads
    'Source (0) / Target (0)')."""
    try:
        import pandas as pd
        conn_csv = out / "route_connection_type.csv"
        if not conn_csv.exists():
            return
        frame = pd.read_csv(conn_csv)
        # F-PF-008 (audit 2026-10-13): the route-scoped visualization
        # draws the SELECTED-support mass — a route-support panel, not
        # the whole admitted type-pair universe (the old `weight` column
        # drew 101 synapses for a hop whose emitted witnesses carry 1).
        # The universe stays in the CSV under `weight`/`universe_pair_mass`.
        frame = frame[frame["selected_support_mass"].fillna(0) > 0]
        if frame.empty:
            return
        from vispath_pkg.vispath import VisualizePath
        vp = VisualizePath(
            path_file=frame[["type_pre", "type_post",
                             "selected_support_mass",
                             "connection_ratio", "connection_ratio_adj"]]
            .rename(columns={"type_pre": "source",
                             "type_post": "target",
                             "selected_support_mass": "weight",
                             "connection_ratio": "ratio",
                             "connection_ratio_adj": "ratio_adj"}),
            output_folder=str(out), showfig=False, verbose=False,
            edgeN_limit=500)
        vp.base_filename = "route_type"
        vp.build_network()   # conn_df/G feed the heatmap + network
        # node roles from the query (edge-list inputs carry none)
        try:
            for node, data in vp.G_network.nodes(data=True):
                base = str(node)
                if base in src_types or base.rsplit('_', 1)[0] in src_types:
                    data['node_type'] = 'source'
                elif (base in tgt_types
                        or base.rsplit('_', 1)[0] in tgt_types):
                    data['node_type'] = 'target'
        except Exception:
            pass
        vp.create_heatmap()
        vp.create_network()
    except Exception:
        pass  # best-effort extras (lane CSVs are the deliverable)


def _read_store_universe(run_dir, _phase=None):
    """Vectorized shortest-store reader for the lane's P1.3 validation
    basis (plan F8 + original-data adjudication 2026-10-12): reads
    shortest_discovery_store/connections directly and emits ONE row per
    distinct (u, v) whose weight is the pair's per-layer ROI-SLICE SUM
    (the layer files are verbatim fetches carrying per-ROI slice rows;
    the previous keep-one-row policies sampled a different slice per
    layer and manufactured hundreds of thousands of fake cross-layer
    conflicts). Cross-layer resolution is keep-first over layers in
    SORTED label order — after summation the layers agree exactly
    (measured 0 conflicts on the MCNS store), so the choice is exact;
    any residual disagreement is a real fetch inconsistency and is
    returned as ``conflicts`` for the provenance disclosure. Replaces
    the _load_shortest_store call whose 27M-row Python dict pass cost
    more wall than the rest of the lane together. Returns
    (universe | None, conflicts)."""
    from pathlib import Path
    base = Path(run_dir) / "shortest_discovery_store"
    if not (base / "meta.json").exists():
        return None, 0
    conn = base / "connections"
    merged = conn / "connections_all.parquet"
    frames = []
    has_roi = True
    if merged.exists():
        f = pl.read_parquet(merged)
        layer_col = ("conn_layer" if "conn_layer" in f.columns
                     else "layer")
        if layer_col not in f.columns:
            return None, 0
        has_roi = "roi" in f.columns
        roi_expr = (pl.col("roi").cast(pl.Utf8) if has_roi
                    else pl.lit("").alias("roi"))
        for label in f[layer_col].unique().sort().to_list():
            frames.append(f.filter(pl.col(layer_col) == label).select(
                pl.col("bodyId_pre").cast(pl.Utf8),
                pl.col("bodyId_post").cast(pl.Utf8),
                pl.col("weight").cast(pl.Float64),
                roi_expr,
                pl.lit(str(label)).alias("conn_layer")))
    else:
        for path in sorted(conn.glob("*.parquet")):
            f = pl.read_parquet(path)
            has_roi = has_roi and "roi" in f.columns
            roi_expr = (pl.col("roi").cast(pl.Utf8) if "roi" in f.columns
                        else pl.lit("").alias("roi"))
            frames.append(f.select(
                pl.col("bodyId_pre").cast(pl.Utf8),
                pl.col("bodyId_post").cast(pl.Utf8),
                pl.col("weight").cast(pl.Float64),
                roi_expr,
                pl.lit(str(path.stem)).alias("conn_layer")))
    if not frames:
        return None, 0
    big = pl.concat(frames)
    # dedup refetch multiplicity on the slice-defining fields (identical
    # slice rows recur across layers verbatim), then SUM per layer: the
    # physical per-layer weight. Without an roi column a row is its own
    # slice definition (dedup on (layer, pair, weight)).
    dedup_subset = (["conn_layer", "bodyId_pre", "bodyId_post", "roi",
                     "weight"] if has_roi
                    else ["conn_layer", "bodyId_pre", "bodyId_post",
                          "weight"])
    per_layer = (big.unique(subset=dedup_subset, keep="any")
                 .group_by(["conn_layer", "bodyId_pre", "bodyId_post"])
                 .agg(pl.col("weight").sum().alias("weight")))
    conflicts = int(
        per_layer.group_by(["bodyId_pre", "bodyId_post"])
        .agg(pl.col("weight").n_unique().alias("wv"))
        .filter(pl.col("wv") > 1).select(pl.len()).item())
    universe = (per_layer.sort("conn_layer", maintain_order=True)
                .unique(subset=["bodyId_pre", "bodyId_post"], keep="first",
                        maintain_order=True)
                .select("bodyId_pre", "bodyId_post", "weight", "conn_layer")
                .sort(["bodyId_pre", "bodyId_post"]))
    return universe, conflicts


def _universe_signature(conn_path, neuron_source, asked, drop_untyped,
                        exclude_intra, separate_hemi, hemi_filter,
                        label_mapper) -> str:
    """Fail-closed cache key material (P1.1): file stats of BOTH input
    tables + every filter that shapes the universe + the label-mapper
    digest. Any change ⇒ a different key ⇒ rebuild; nothing stale is
    ever served (cache-survey c2e26c1 discipline; the lane uses direct
    file stats because it can run on release tables where the engine's
    data-generation stamp does not exist)."""
    import hashlib
    import os
    h = hashlib.md5()
    for p in (conn_path, neuron_source):
        try:
            st = os.stat(p)
            h.update(f"{p}|{st.st_mtime_ns}|{st.st_size}".encode())
        except OSError:
            return ""  # unstat-able input ⇒ no caching
    h.update(f"|{asked}|{drop_untyped}|{exclude_intra}|"
             f"{separate_hemi}|{hemi_filter}".encode())
    if label_mapper is not None:
        try:
            import json as _json
            h.update(_json.dumps(
                getattr(label_mapper, "overall_mapping", None)
                or {}, sort_keys=True, default=str).encode())
        except Exception:
            h.update(b"|mapper-unhashable")
    return h.hexdigest()[:12]


def _typed_universe_cached(conn_path, neuron_source, conn, tmap, asked,
                           drop_untyped, exclude_intra, separate_hemi,
                           hemi_filter, label_mapper, _phase):
    """typed/pairs universe with a signature-keyed parquet cache
    (P1.1): cache/route_plan/<dataset_safe>/universe__w<min>__u<0|1>__<sig12>.parquet (typed edges; pairs re-derive in <1 s). Atomic write;
any signature mismatch ⇒ rebuild — fail-closed."""
    import os
    from pathlib import Path as _P
    sig = _universe_signature(conn_path, neuron_source, asked,
                              drop_untyped, exclude_intra, separate_hemi,
                              hemi_filter, label_mapper)
    cache_file = None
    if sig:
        dataset_safe = _P(str(conn_path)).resolve().parent.name or "default"
        cache_dir = _P(__file__).resolve().parents[1] / "cache" \
            / "route_plan" / dataset_safe
        cache_file = cache_dir / (
            f"universe__w{asked}__u{int(drop_untyped)}__{sig}.parquet")
        if cache_file.exists():
            try:
                typed = pl.read_parquet(cache_file)
                _phase(f"cache-hit ({typed.height:,} typed rows)")
                pairs = (typed.group_by("type_pre", "type_post")
                         .agg(pl.col("weight").sum().alias("w_sum"),
                              pl.col("weight").max().alias("w_max"),
                              pl.len().alias("n")))
                return typed, pairs
            except Exception:
                pass  # corrupt cache ⇒ rebuild (fail-closed)
    typed, pairs = build_type_pairs(conn, tmap, asked,
                                    drop_untyped=drop_untyped)
    if cache_file is not None and typed.height:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".tmp.parquet")
            typed.write_parquet(tmp)
            os.replace(tmp, cache_file)  # atomic (c2e26c1 pattern)
        except OSError:
            pass  # cache write is best-effort
    return typed, pairs


def _fmt_mass(value) -> str:
    """Engine mass formatting: ints stay ints, floats keep their value."""
    return str(int(value)) if float(value).is_integer() else str(
        float(value))


def _type_nt_map(frame):
    """Type -> neurotransmitter via the engine's column preference chain
    (coana enrichment: nt_type > consensusNt > predictedNt >
    celltypePredictedNt), most common non-null label per type."""
    import collections
    nt_col = next((c for c in ("nt_type", "consensusNt", "predictedNt",
                               "celltypePredictedNt")
                   if c in getattr(frame, "columns", [])), None)
    counts = collections.defaultdict(collections.Counter)
    if nt_col is not None:
        try:
            types = frame["type"].astype(str)
            nts = frame[nt_col].astype(str).where(
                frame[nt_col].notna(), "")
            for t, nt in zip(types, nts):
                if nt and str(nt).lower() not in ("nan", "none", ""):
                    counts[t][nt] += 1
        except Exception:
            pass
    return {t: c.most_common(1)[0][0] for t, c in counts.items()}


def _type_adj(pairs, src_types, tgt_types, bound):
    ds = type_bfs(pairs, src_types, "fwd", bound)
    dt = type_bfs(pairs, tgt_types, "bwd", bound)
    ds_df = pl.DataFrame({"type_pre": list(ds.keys()),
                          "dS": list(ds.values())})
    dt_df = pl.DataFrame({"type_post": list(dt.keys()),
                          "dT": list(dt.values())})
    rel = (pairs.join(ds_df, on="type_pre", how="inner")
                .join(dt_df, on="type_post", how="inner")
                .filter((pl.col("dS") + 1 + pl.col("dT")) <= bound))
    adj = {}
    for pre, post, w in zip(rel["type_pre"].to_list(),
                            rel["type_post"].to_list(),
                            rel["w_max"].to_list()):
        adj.setdefault(pre, []).append((post, w))
    return adj, rel, ds, dt


def type_bfs(pairs, seeds, direction, max_layers):
    dist = {t: 0 for t in seeds}
    frontier = pl.DataFrame({"t": list(seeds)}, schema={"t": pl.Utf8})
    visited = frontier
    for layer in range(1, max_layers + 1):
        if direction == "fwd":
            nxt = (pairs.join(frontier, left_on="type_pre", right_on="t")
                   .select(pl.col("type_post").alias("t")).unique())
        else:
            nxt = (pairs.join(frontier, left_on="type_post", right_on="t")
                   .select(pl.col("type_pre").alias("t")).unique())
        nxt = nxt.join(visited, on="t", how="anti")
        if nxt.height == 0:
            break
        for t in nxt["t"].to_list():
            dist[t] = layer
        visited = pl.concat([visited, nxt])
        frontier = nxt
    return dist
