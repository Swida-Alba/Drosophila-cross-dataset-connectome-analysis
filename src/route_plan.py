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
                            multi-visit) -> acceptance pays the class's
                            ROUTE share once (existence probe, not a
                            realization count), realization mass is spent
                            later against the class envelope
                            -> per-class floors/budget stops
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
                            lane columns), per-path bodyId rows for the
                            Fast lane, scope type-pair masses (the typed
                            lane uses one layered fwd/bwd pass and writes
                            no bodyId witnesses), content-hashed plan.json,
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

# Namespaces (issue 2026-10-22 §2.1): the Fast lane owns route_scope_paths/;
# the type-pruned lane owns its own folder so neither can overwrite the other.
FAST_LANE_DIR = "route_scope_paths"
TYPED_LANE_DIR = "route_scope_paths_type_pruned"


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
        # P1-H: use the RUN's untyped predicate (it includes the literal
        # `Unknown` and the bodyId fallback), not a bare empty/nan test.
        # The full-floor builder and the engine's loss filter must agree.
        from utils.label_utils import is_untyped_type_label
        bad = pl.Series([
            is_untyped_type_label(a) or is_untyped_type_label(b)
            for a, b in zip(typed["type_pre"], typed["type_post"])])
        typed = typed.filter(~bad)
    pairs = (typed.group_by("type_pre", "type_post")
             .agg(pl.col("weight").sum().alias("w_sum"),
                  pl.col("weight").max().alias("w_max"),
                  pl.len().alias("n")))
    return typed, pairs


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
    """Per (pair, h) class budget: acceptance in ROUTES, materialization in PATHS.

    Two currencies, deliberately separated (2026-10-16; measured on
    find-paths-complete_MCNS_aMe12_etc_to_PPL101_etc_L1w5_20261016_132625):

    * ``route_share`` — how many VALID TYPE-LEVEL ROUTES the class
      publishes. Acceptance used to consume ``limit`` by REALIZATION COUNT,
      so the single walk aMe12->KCg-d->PPL103 (>= 100 bodyId paths) filled
      its class and the lane published ONE route for that pair while a
      candidate at UB 15 waited behind it (accepted=1, used=100, pulled=1,
      stop='budget'). A candidate is now probed for EXISTENCE, so a fat
      corridor cannot starve its siblings.
    * ``limit`` (``--pair-length-budget``) — the class's REALIZATION
      envelope: how many bodyId witness paths the materialization pass may
      enumerate for the class. Every accepted route is still guaranteed one
      witness, so the envelope truncates the strong walks' tie mass and
      never the route list.
    * ``pull_cap`` — candidate pulls allowed (route share + a phantom
      allowance). §21's probe cap was the only pull bound, which let
      phantoms eat the share with no fill-up; now a class that spends its
      pulls on phantoms closes as ``probe_cap`` and one that filled its
      share closes as ``budget``.

    ``count_routes=False`` is the opt-in certify_actual reading: that mode
    certifies the class's top ``limit`` REALIZED paths by actual bottleneck,
    so its share stays counted in paths.
    feasible (§22 cost control): False when no length-h walk can exist (h
    below the min type distance) — such a class can never receive
    candidates, so it must not keep the drain alive."""
    limit: int = 100
    route_share: int | None = None
    pull_cap: int | None = None
    count_routes: bool = True
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
    feasible: bool = True
    # D2-B (certified actual-strength mode, opt-in): the class stays
    # share-satisfied mechanically, but the batch loop keeps witnessing
    # skipped candidates until the first skipped candidate's optimistic
    # bound drops to the weakest ACTUALLY emitted path's bottleneck --
    # then no skipped candidate can beat any exported path and the
    # top-share is certified against actual route strength.
    certify_actual: bool = False
    certified: bool = False
    weakest_emitted: float | None = None
    enum_upto: int = 0
    realized_bns: list = field(default_factory=list)
    pending_records: list = field(default_factory=list)
    # enumeration-side disclosures (filled by run_lane's per-class pass)
    emitted: int = 0
    off_plan: int = 0
    off_len: int = 0

    @property
    def share(self) -> int:
        """The acceptance bound, in its own currency: type-level ROUTES by
        default, realized PATHS for certify_actual (or for a caller that
        passed no route_share, which keeps the old path-share reading)."""
        if not self.count_routes or self.route_share is None:
            return self.limit
        return int(self.route_share)

    def satisfied(self) -> bool:
        if self.certify_actual and self.feasible:
            # D2-B: an OPEN class stays in the drain's pending set so the
            # skip path keeps witnessing skipped candidates' UBs against
            # the running weakest exported bottleneck. The pull cap below
            # still closes the class (a certified-mode drain is bounded by
            # the same budgets).
            if self.used >= self.share or not self.feasible:
                return self.certified
            return (self.pull_cap is not None
                    and self.pulled >= self.pull_cap)
        if self.used >= self.share or not self.feasible:
            return True
        return (self.pull_cap is not None
                and self.pulled >= self.pull_cap)


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
        share_done = cb.used >= cb.share or not cb.feasible
        _cert_open = (cb.certify_actual and cb.feasible
                      and not cb.certified)
        if cb.satisfied() or (share_done and (
                not _cert_open
                or (cb.weakest_emitted is not None
                    and ub < cb.weakest_emitted))):
            # Class already done: the first post-stop candidate's UB is
            # the floor witness (plan §5: floor = UB(M+1) — the last
            # ACCEPTED UB can overstate safety by a tie).
            if cb.floor_next is None:
                cb.floor_next = ub
            if (cb.certify_actual and not cb.certified
                    and cb.weakest_emitted is not None
                    and cb.floor_next <= cb.weakest_emitted):
                # D2-B: the first skipped candidate's optimistic bound has
                # dropped below the weakest exported path's ACTUAL
                # bottleneck -- no skipped candidate can beat any exported
                # path; the top-share is certified against actual route
                # strength. (Certified-open classes with UB >= the witness
                # are ACCEPTED above: their realizations join the pool and
                # may take the share -- the audit's displacement case.)
                cb.certified = True
            if not (cb.certify_actual and share_done
                    and not cb.certified and cb.feasible):
                # D2-B: an open certified-mode class stays pending -- the
                # drain keeps producing the witnesses it needs
                pending.discard(key)
            continue
        cb.pulled += 1
        since_accept += 1
        # ROUTE-COUNTED ACCEPTANCE (2026-10-16): a candidate is probed for
        # EXISTENCE (cap 1 — any realization proves it is no phantom), so a
        # fat corridor's realization mass can no longer consume the class's
        # route share. certify_actual keeps its path-share reading: its
        # certification needs every realization's bottleneck inline.
        n, truncated = realize_fn(
            walk, cb.share if _cert_open
            else (1 if cb.count_routes else cb.share - cb.used))
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
        cb.used += 1 if cb.count_routes else n
        cb.floor = ub
        if truncated:
            cb.truncated_walks += 1
        if cb.used >= cb.share:
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
        elif cb.pull_cap is not None and cb.pulled >= cb.pull_cap \
                and cb.used < cb.share:
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
            cb.stop = "exhausted" if cb.used < cb.share else "budget"
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
                "route_share": cb.share, "pull_cap": cb.pull_cap,
                "realization_envelope": cb.limit,
                "counted": "routes" if cb.count_routes else "paths",
                "feasible": cb.feasible,
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
        self._full_cache: dict = {}  # (a, b) -> {u: [(v, w), ...]}

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
        # The canonical sorted frame is the only view needed after this;
        # release the caller's unsorted frame reference (the caller may
        # also drop its own reference in the typed lane).
        self._loss = None

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

    def full(self, a, b) -> dict:
        """u -> [(v, w), ...] view consumed by the layered chain pass.

        Cached once per pair; the typed lane calls it for every walk, so
        rebuilding/sorting this view per walk was a measured cost (issue
        2026-10-23). Values are sorted by (v, w) for deterministic sums."""
        key = (a, b)
        rec = self._full_cache.get(key)
        if rec is None:
            d = {}
            for u, v, w in self.tuples(a, b):
                d.setdefault(u, []).append((v, float(w)))
            for u in d:
                d[u].sort()
            rec = d
            self._full_cache[key] = rec
            # The typed lane consumes only `full` for this pair; the
            # column lists are now redundant and would otherwise double
            # the retained edge memory.
            self._cache.pop(key, None)
        return rec


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


def _realize_on_loss(pair_edges, walk, src_bodies, tgt_bodies, cap,
                     *, dp_work_cap=2_000_000, dfs_work_cap=200_000):
    """Layered-reachability realizations of ONE walk via the shared
    per-pair edge cache. The two work caps are explicit so a complete
    lane can disclose (and raise) the truncation threshold instead of
    inheriting the top-M lane's defaults (P2-Q)."""
    hd = [pair_edges.hop(a, b) for a, b in zip(walk, walk[1:])]
    if _walk_is_distinct(walk):
        # distinct types: the DP is exact (simplicity automatic) and
        # proves phantoms in O(chain subgraph) — the DFS's work-cap
        # 'unverified' category disappears for these walks (unless the
        # DP's own work cap fires on a store-basis monster)
        n, truncated = _dp_count(hd, src_bodies, tgt_bodies,
                                 work_cap=dp_work_cap)
        return (n if n < cap else cap), truncated
    return realize_count(hd, src_bodies, tgt_bodies, cap,
                         work_cap=dfs_work_cap)


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


# ---------------------------------------------- Phase B: type-pruned lane
def _prune_type_pairs(pairs, type_incoming, type_incoming_adj, *,
                      basis="ratio_adj", min_ratio_adj=0.01,
                      min_synapse=0, min_wmax=0, default_floor=None):
    """Coarse TYPE-LEVEL candidate filter (P0-B/P0-C).

    `pairs` is the full-floor pair table (w_sum/w_max/n), NOT a top-M or
    emitted-chain view. A pair's adjusted ratio is its synapse mass over
    the post type's CONDITIONED incoming denominator — the same
    denominator the realized path readout divides by. This is a COARSE
    candidate filter: pair mass is spread over positions/layers, so a
    strong path can sit behind a diluted pair. The exact path filter is
    applied after bodyId validation on the realized hop mass.

    Returns (kept_pairs, block). Kept pairs carry both pair-level ratios
    so the manifest and the connection table can state which quantity was
    pruned on."""
    import polars as pl
    _applied_mass_floor = (int(min_synapse) if int(min_synapse) > 0
                           else int(default_floor or 1))
    if pairs.height == 0:
        return pairs, {
            "basis": basis, "min_ratio_adj": float(min_ratio_adj),
            "min_synapse": (int(min_synapse) if basis != "synapse"
                            else _applied_mass_floor),
            "min_wmax": int(min_wmax),
            "pairs_before": 0, "pairs_after": 0, "pairs_removed": 0,
            "mass_before": 0.0, "mass_after": 0.0, "mass_removed": 0.0}
    den = pl.DataFrame(
        {"t": list(type_incoming.keys()),
         "w": [float(v) for v in type_incoming.values()]},
        schema={"t": pl.Utf8, "w": pl.Float64})
    den_adj = pl.DataFrame(
        {"t": list(type_incoming_adj.keys()),
         "w": [float(v) for v in type_incoming_adj.values()]},
        schema={"t": pl.Utf8, "w": pl.Float64})
    p = (pairs
         .join(den.rename({"t": "type_post", "w": "_in"}), on="type_post",
               how="left")
         .join(den_adj.rename({"t": "type_post", "w": "_in_adj"}),
               on="type_post", how="left")
         .with_columns(
             pl.when(pl.col("_in") > 0)
             .then(pl.col("w_sum") / pl.col("_in"))
             .otherwise(None).alias("connection_ratio"),
             pl.when(pl.col("_in_adj") > 0)
             .then(pl.col("w_sum") / pl.col("_in_adj"))
             .otherwise(None).alias("connection_ratio_adj"))
         .drop(["_in", "_in_adj"]))
    basis = str(basis).lower()
    if basis == "ratio_adj":
        keep = (pl.col("connection_ratio_adj").is_not_null()
                & (pl.col("connection_ratio_adj") >= float(min_ratio_adj)))
    elif basis == "connection_ratio":
        keep = (pl.col("connection_ratio").is_not_null()
                & (pl.col("connection_ratio") >= float(min_ratio_adj)))
    elif basis == "synapse":
        # the basis itself is a mass floor; an explicit min_synapse wins,
        # otherwise the requested ratio knob is not a synapse count, so the
        # caller's min_synapse is authoritative (default 1 keeps the table
        # non-empty rather than silently pruning everything at 0).
        floor = _applied_mass_floor
        keep = pl.col("w_sum") >= floor
    else:  # guarded by run_lane's refusal; keep the function total
        raise ValueError(f"unknown type-prune basis {basis!r}")
    if int(min_synapse) > 0 and basis != "synapse":
        keep = keep & (pl.col("w_sum") >= int(min_synapse))
    if int(min_wmax) > 0:
        keep = keep & (pl.col("w_max") >= int(min_wmax))
    kept = p.filter(keep).sort(["type_pre", "type_post"])
    mass_before = float(pairs["w_sum"].sum())
    mass_after = float(kept["w_sum"].sum()) if kept.height else 0.0
    return kept, {
        "basis": basis,
        "min_ratio_adj": float(min_ratio_adj),
        "min_synapse": (int(min_synapse) if basis != "synapse"
                        else _applied_mass_floor),
        "min_wmax": int(min_wmax),
        "pairs_before": int(pairs.height),
        "pairs_after": int(kept.height),
        "pairs_removed": int(pairs.height - kept.height),
        "mass_source": "lossless_cone",
        "mass_before_source": "lossless_discovery_cone_pair_mass",
        "denominator_source": "conditioned type incoming (all pres, "
                              "edges >= asked)",
        "mass_before": mass_before,
        "mass_after": mass_after,
        "mass_removed": mass_before - mass_after,
        "kept_mass_share": (mass_after / mass_before
                            if mass_before > 0 else None)}


def _type_adjacency(pairs):
    """Type-level adjacency in the lane's canonical order."""
    adj: dict = {}
    for a, b, w in zip(pairs["type_pre"], pairs["type_post"],
                       pairs["w_max"]):
        adj.setdefault(str(a), []).append((str(b), float(w)))
    for a in adj:
        adj[a].sort(key=lambda x: (-x[1], x[0]))
    return adj


def _type_distance_avoiding(pairs, source, target, targets, max_layers,
                            adj=None):
    """Shortest type-only hop count from *source* to *target* that never
    passes THROUGH another target type (the lane's first-target rule).

    Returns None when no such route exists within *max_layers*. This is a
    discovery aid only; the validated shortest length is proven at bodyId
    level, never taken from this BFS."""
    adj = adj if adj is not None else _type_adjacency(pairs)
    if source == target or source in targets:
        return 0 if source == target else None
    frontier = {source}
    seen = {source}
    for layer in range(1, int(max_layers) + 1):
        nxt = set()
        for u in frontier:
            for v, _w in adj.get(u, ()):
                if v in seen:
                    continue
                if v == target:
                    return layer
                if v in targets:
                    # a first target ends the lane's walk; it is not a
                    # valid intermediate
                    continue
                seen.add(v)
                nxt.add(v)
        if not nxt:
            break
        frontier = nxt
    return None


def _type_walks_exact(pairs, source, target, hops, targets, *,
                      revisit_policy="bounded", max_type_visits=0,
                      max_walks=2_000_000, adj=None):
    """Every type walk of EXACTLY *hops* hops from source to target.

    Uses the same first-target rule and canonical adjacency order as
    ``_enumerate_type_walks``, so a discovered length is a length the
    complete pass can actually enumerate. The discovery pass tries these
    lengths in ascending order; a phantom shortest walk simply yields no
    accepted realization and the pass advances one hop."""
    adj = adj if adj is not None else _type_adjacency(pairs)
    tgt = {str(t) for t in targets}
    source = str(source)
    target = str(target)
    if source == target or source in tgt:
        return [], {"walks": 0, "capped": False}
    _max_visits = (int(max_type_visits) if int(max_type_visits) > 0
                   else int(hops))
    walks: list = []
    capped = False

    def _dfs(node, walk, visits, depth):
        nonlocal capped
        if depth >= hops:
            return
        for nxt, _w in adj.get(node, ()):
            if len(walks) >= max_walks:
                capped = True
                return
            nd = depth + 1
            if nxt in tgt:
                if nxt == target and nd == hops:
                    walks.append(tuple(walk + (nxt,)))
                # FIRST-TARGET RULE: never continue through a target
                continue
            if revisit_policy == "simple":
                if nxt in visits:
                    continue
                nv = dict(visits)
                nv[nxt] = 1
                _dfs(nxt, walk + (nxt,), nv, nd)
            elif revisit_policy == "bounded":
                if visits.get(nxt, 0) >= _max_visits:
                    continue
                nv = dict(visits)
                nv[nxt] = nv.get(nxt, 0) + 1
                _dfs(nxt, walk + (nxt,), nv, nd)
            else:
                raise ValueError(
                    f"unknown revisit policy {revisit_policy!r}")

    _dfs(source, (source,), {source: 1}, 0)
    return walks, {"walks": len(walks), "capped": capped}


def _layered_chain_walk(walk, pair_edges, src_bodies, tgt_bodies,
                        body2type):
    """ONE forward/backward layered pass over a type walk.

    This is the typed lane's validation and readout contract (issue
    2026-10-23): forward body sets from the walk's source type, backward
    sets from its target type, intersect per position, and keep the
    surviving per-hop edges. No strongest-first enumeration, no bodyId
    witness list, no realization envelope. Returns the
    `_layered_chain_structures` dict, or None when the intersection is
    empty (a phantom)."""
    from type_level_refill import _layered_chain_structures

    hop_dicts = {}
    for a, b in zip(walk, walk[1:]):
        d = pair_edges.full(a, b)
        if not d:
            return None
        hop_dicts[(a, b)] = d
    try:
        return _layered_chain_structures(
            tuple(walk), hop_dicts, src_bodies, tgt_bodies,
            body2type, set())
    except Exception:  # noqa: BLE001 - fail closed, never fabricate
        return None


def _layered_min_adj(walk, pair_edges, src_bodies, tgt_bodies, body2type,
                     type_incoming_adj):
    """Discovery readout: (realized, min_ratio_adj).

    ``realized=False`` is a phantom. ``realized=True, min_adj=None`` is a
    validated walk whose adjusted readout is unresolved (a missing
    conditioned denominator), never a phantom."""
    st = _layered_chain_walk(
        walk, pair_edges, src_bodies, tgt_bodies, body2type)
    if st is None:
        return False, None
    masses = [sum(e.values()) for e in st["edges_at"]]
    finite = []
    for i, mass in enumerate(masses):
        denom = type_incoming_adj.get(walk[i + 1], 0.0)
        if denom > 0:
            finite.append(mass / denom)
    return True, (min(finite) if finite else None)


def _layered_row(walk, pair_edges, src_bodies, tgt_bodies, body2type,
                 type_incoming, type_incoming_adj, type_membership,
                 type_nt, pair_lookup):
    """The typed lane's chain-level row from ONE layered pass.

    Returns (row, hop_edges) or None for a phantom. The row carries only
    chain-level masses/ratios/coverage: no sampled witness columns and no
    bodyId rows."""
    st = _layered_chain_walk(
        walk, pair_edges, src_bodies, tgt_bodies, body2type)
    if st is None:
        return None
    h = len(walk) - 1
    chain_edges = st["edges_at"]
    chain_nodes = st["nodes_at"]
    chain_revisit = bool(st["revisit"])
    hop_mass = [sum(e.values()) for e in chain_edges]
    ratios, ratios_adj = [], []
    for i in range(h):
        denom = type_incoming.get(walk[i + 1], 0.0)
        ratios.append(hop_mass[i] / denom if denom > 0 else 0.0)
        denom_adj = type_incoming_adj.get(walk[i + 1], 0.0)
        ratios_adj.append(hop_mass[i] / denom_adj
                          if denom_adj > 0 else None)
    finite_adj = [r for r in ratios_adj if r is not None]
    cov = ["{}/{}".format(
        len(chain_nodes[i]),
        type_membership.get(walk[i], len(chain_nodes[i])))
        for i in range(h + 1)]
    nts = [type_nt.get(walk[i], "unknown") for i in range(h)]
    pair_weights = [pair_lookup.get((walk[i], walk[i + 1]), (None,))[0]
                    for i in range(h)]
    row = {
        "path": "->".join(walk),
        "weights": "[" + ", ".join(_fmt_mass(m) for m in hop_mass) + "]",
        "ratios": "[" + ", ".join(repr(r) for r in ratios) + "]",
        "ratios_adj": "[" + ", ".join(
            'None' if r is None else repr(r)
            for r in ratios_adj) + "]",
        "min_weight": (int(min(hop_mass)) if hop_mass else None),
        "min_ratio": min(ratios) if ratios else None,
        "min_ratio_adj": (min(finite_adj) if finite_adj else None),
        "length": h,
        "nt_types": "[" + ", ".join(f'"{x}"' for x in nts) + "]",
        "coverage": "[" + ", ".join(cov) + "]",
        "chain_revisit": chain_revisit,
        "applied_tau": None,
        "route_h": h,
        "route_ub": min((pair_lookup.get(
            (walk[i], walk[i + 1]), (0.0, 0.0, 0))[1]
            for i in range(h)), default=None),
        "route_floor_h": None,
        "plan_rank": None,
        "pair_weights": "[" + ", ".join(
            'None' if w is None else _fmt_mass(w)
            for w in pair_weights) + "]",
        "ratio_adj_complete": (len(finite_adj) == len(ratios_adj)
                               and bool(ratios_adj)),
        "type_pruned": True,
    }
    return row, chain_edges


def _empty_scope_counts():
    return {"candidate_walks": 0, "validated_walks": 0, "paths": 0,
            "bodyid_paths": 0, "phantoms": 0, "filtered_weak": 0,
            "unverified": 0, "ratio_unresolved": 0}


def _scope_dataset_family(dataset):
    """The per-pair report's Lt_<volume> column this dataset fills."""
    name = str(dataset or "").strip().lower()
    if "banc" in name:
        return "BANC"
    if "male-cns" in name or "mcns" in name:
        return "MCNS"
    if "fafb" in name or "flywire" in name:
        return "FAFB"
    return None


def _discover_scope(*, pairs, pair_edges, src_types, tgt_types, src_ids,
                    tgt_ids, src_keep, tgt_keep, body2type, discovery_cap,
                    scope_pad, hard_cap, min_ratio_adj,
                    type_incoming_adj, revisit_policy, max_type_visits,
                    walk_cap, dataset="", deadline=None, log=None):
    """Pass A: the per-pair validated shortest length.

    For each selected (source_type, target_type) pair, try type walks in
    ascending hop count. A type-shortest walk with no layered bodyId
    realization (a phantom) does NOT set ``Lt_i``; the pass advances until
    the first walk with a non-empty forward/backward intersection that
    clears the exact realized ``min_ratio_adj`` gate. Returns
    (rows_by_pair, bounds_by_pair, block).

    Raw type-BFS distances appear only as evidence
    (``type_shortest_hops``); the stored ``Lt_i`` is the validated length.
    """
    say = log or (lambda *a, **k: None)
    import time as _t

    targets = {str(t) for t in tgt_types}
    src_order = sorted({str(s) for s in src_types})
    tgt_order = sorted(targets)
    s_bodies_by_type, t_bodies_by_type = {}, {}
    for b in src_ids:
        typ = body2type.get(b)
        if typ in targets or typ is None:
            continue
        if not src_keep or b in src_keep:
            s_bodies_by_type.setdefault(typ, []).append(b)
    for b in tgt_ids:
        typ = body2type.get(b)
        if typ is None:
            continue
        if not tgt_keep or b in tgt_keep:
            t_bodies_by_type.setdefault(typ, set()).add(b)

    unbounded_cap = max(1, len({str(x) for x in pairs["type_pre"]}) + 1)
    adj = _type_adjacency(pairs)
    rows: dict = {}
    bounds: dict = {}
    cap_hit = []
    validation_cap_pairs = []
    no_path_pairs = []

    for s in src_order:
        for t in tgt_order:
            pair = (s, t)
            row = {
                "source_type": s, "target_type": t,
                "Lt_i": None, "Lt": None, "Lt_dataset": dataset or None,
                "scope_bound": None, "status": "no_path",
                "hard_capped": False, "discovery_cap_hit": False,
                "type_shortest_hops": None,
                "phantom_attempts": 0,
                "first_phantom_type_path": None,
                "witness_type_path": None,
                **_empty_scope_counts(),
            }
            bounds[pair] = None
            if s in targets:
                rows[pair] = row
                no_path_pairs.append(pair)
                continue
            d_cap = _type_distance_avoiding(
                pairs, s, t, targets, discovery_cap, adj=adj)
            d_any = None
            if d_cap is None:
                d_any = _type_distance_avoiding(
                    pairs, s, t, targets, unbounded_cap, adj=adj)
            row["type_shortest_hops"] = d_cap if d_cap is not None else d_any
            if d_cap is None:
                if d_any is not None and d_any > discovery_cap:
                    row["status"] = "capped"
                    row["discovery_cap_hit"] = True
                    cap_hit.append(pair)
                else:
                    no_path_pairs.append(pair)
                rows[pair] = row
                bounds[pair] = None
                continue

            found = False
            walk_cap_hit = False
            for hops in range(int(d_cap), int(discovery_cap) + 1):
                if deadline is not None and _t.monotonic() > deadline:
                    row["status"] = "capped"
                    cap_hit.append(pair)
                    break
                walks, wstats = _type_walks_exact(
                    pairs, s, t, hops, targets,
                    revisit_policy=revisit_policy,
                    max_type_visits=max_type_visits,
                    max_walks=walk_cap, adj=adj)
                if wstats.get("capped"):
                    row["status"] = "capped"
                    walk_cap_hit = True
                    cap_hit.append(pair)
                    break
                for walk in walks:
                    row["candidate_walks"] += 1
                    s_bodies = s_bodies_by_type.get(s, [])
                    t_bodies = t_bodies_by_type.get(t, set())
                    if not s_bodies or not t_bodies:
                        row["unverified"] += 1
                        continue
                    realized, min_adj = _layered_min_adj(
                        walk, pair_edges, s_bodies, t_bodies, body2type,
                        type_incoming_adj)
                    if not realized:
                        row["phantoms"] += 1
                        row["phantom_attempts"] += 1
                        if row["first_phantom_type_path"] is None:
                            row["first_phantom_type_path"] = "->".join(walk)
                        continue
                    row["validated_walks"] += 1
                    if min_adj is None:
                        row["ratio_unresolved"] += 1
                        continue
                    if min_ratio_adj > 0 and float(min_adj) <= float(
                            min_ratio_adj):
                        row["filtered_weak"] += 1
                        continue
                    bound = int(hops) + int(scope_pad)
                    hard_capped = bool(
                        hard_cap is not None and bound > int(hard_cap))
                    if hard_capped:
                        bound = int(hard_cap)
                    row.update({
                        "Lt_i": int(hops), "Lt": int(hops),
                        "scope_bound": int(bound),
                        "scope_pad_effective": int(bound) - int(hops),
                        "hard_capped": hard_capped,
                        "status": "capped" if hard_capped else "exhausted",
                        "witness_type_path": "->".join(walk),
                    })
                    found = True
                    break
                if found:
                    break
            if found:
                bounds[pair] = row["scope_bound"]
                if row["hard_capped"]:
                    cap_hit.append(pair)
            elif row["status"] == "no_path":
                if (row["unverified"] or row["ratio_unresolved"]):
                    row["status"] = "validation_cap"
                    validation_cap_pairs.append(pair)
                elif walk_cap_hit:
                    row["status"] = "capped"
                else:
                    no_path_pairs.append(pair)
            rows[pair] = row

    block = {
        "scope_mode": "per_pair_shortest_plus_2",
        "scope_lt": "max_local_validated",
        "scope_pad": int(scope_pad),
        "scope_hard_cap": (int(hard_cap) if hard_cap is not None else None),
        "scope_discovery_cap": int(discovery_cap),
        "scope_bound_source": "Lt_plus_pad",
        "scope_pairs_total": len(rows),
        "scope_pairs_with_lt": sum(
            1 for r in rows.values() if r["Lt_i"] is not None),
        "scope_capped_pairs": sorted(
            {f"{s}->{t}" for s, t in cap_hit}),
        "scope_no_path_pairs": sorted(
            {f"{s}->{t}" for s, t in no_path_pairs}),
        "scope_validation_cap_pairs": sorted(
            {f"{s}->{t}" for s, t in validation_cap_pairs}),
        "scope_lt_evidence": "bodyId-validated accepted walk per pair "
                             "(not a raw type-BFS shortest)",
    }
    say(f"  · scope discovery: {block['scope_pairs_with_lt']}/"
        f"{block['scope_pairs_total']} pairs with a validated Lt "
        f"(cap {discovery_cap} hops)")
    return rows, bounds, block


def _scope_int(value):
    """Parse a scope-table integer; blank/None/null/nan -> None."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in ("", "none", "null", "nan", "na"):
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        raise CorridorRefusal(f"scope table value {value!r} is not an int")
    if not number.is_integer():
        raise CorridorRefusal(
            f"scope table value {value!r} is not a whole hop count")
    return int(number)


def _scope_table_value(df, column):
    """One non-empty value from a scope-table column, refusing conflicts."""
    if column is None:
        return None
    vals = set()
    for value in df[column].to_list():
        if value is None:
            continue
        text = str(value).strip()
        if text.lower() in ("", "none", "null", "nan", "na"):
            continue
        vals.add(text)
    if len(vals) > 1:
        raise CorridorRefusal(
            f"scope table column {column!r} carries conflicting values "
            f"{sorted(vals)[:4]}")
    return next(iter(vals)) if vals else None


def _load_scope_table(path, *, src_types, tgt_types, scope_pad, hard_cap,
                      type_map_sha, asked, threshold):
    """Load and validate the external per-pair scope table.

    Required coverage is every selected (source_type, target_type) pair.
    A row supplies ``max_hops`` (or ``scope_bound``/``bound``) and may
    carry ``Lt_i``/``Lt``/``Lt_dataset`` evidence. Rows without a bound are
    unreached pairs (bound None), never silently assigned a global bound.
    Missing/extra/duplicate pairs, a bound below a declared Lt, and stale
    mapping/threshold metadata all refuse the run.
    """
    import hashlib
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise CorridorRefusal(
            f"scope table {str(p)!r} does not exist")
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise CorridorRefusal(
            f"scope table {str(p)!r} is not readable ({exc})")
    if not raw.strip():
        raise CorridorRefusal("scope table has no rows")
    try:
        df = pl.read_csv(p, infer_schema_length=0)
    except Exception as exc:  # noqa: BLE001 - refusal is the contract
        raise CorridorRefusal(
            f"scope table {str(p)!r} is not readable as CSV ({exc})")
    if df.height == 0:
        raise CorridorRefusal("scope table has no rows")
    cols = {str(c).strip().lower(): c for c in df.columns}

    def _col(*names):
        for name in names:
            if name in cols:
                return cols[name]
        return None

    src_c = _col("source_type", "source", "type_pre")
    tgt_c = _col("target_type", "target", "type_post")
    max_c = _col("max_hops", "scope_bound", "bound", "hop_bound")
    lti_c = _col("lt_i", "local_lt")
    lt_c = _col("lt", "common_lt")
    ltd_c = _col("lt_dataset")
    status_c = _col("status", "scope_status")
    vol_cols = {name: _col(name.lower())
                for name in ("Lt_FAFB", "Lt_MCNS", "Lt_BANC")}
    if src_c is None or tgt_c is None:
        raise CorridorRefusal(
            "scope table needs source_type and target_type columns")
    if (max_c is None and lt_c is None
            and not any(c is not None for c in vol_cols.values())):
        raise CorridorRefusal(
            "scope table needs max_hops (or scope_bound/bound), Lt, or "
            "Lt_FAFB/Lt_MCNS/Lt_BANC")

    hash_c = _col("type_map_sha")
    asked_c = _col("asked", "requested_floor")
    thr_c = _col("threshold", "min_ratio_adj")

    def _check_meta(label, column, current, parse=float):
        got = _scope_table_value(df, column)
        if got is None:
            return
        try:
            same = abs(parse(got) - parse(current)) <= 1e-9
        except (TypeError, ValueError):
            same = str(got) == str(current)
        if not same:
            raise CorridorRefusal(
                f"scope table is stale: {label}={got!r} but this run's "
                f"value is {current!r}")

    if _scope_table_value(df, hash_c) is not None:
        got = _scope_table_value(df, hash_c)
        if str(got) != str(type_map_sha):
            raise CorridorRefusal(
                f"scope table is stale: type_map_sha={got!r} but this "
                f"run's effective type map is {type_map_sha!r}")
    _mode_col = _col("scope_mode")
    _mode_val = _scope_table_value(df, _mode_col)
    if _mode_val is not None:
        _mode_val = _mode_val.strip().lower()
    if _mode_val is not None and _mode_val != "per_pair_shortest_plus_2":
        raise CorridorRefusal(
            f"scope table declares scope_mode={_mode_val!r}; this lane "
            "only consumes 'per_pair_shortest_plus_2'")
    _check_meta("asked", asked_c, asked)
    _check_meta("threshold", thr_c, threshold)

    selected = {(str(s), str(t)) for s in src_types for t in tgt_types}
    rows: dict = {}
    bounds: dict = {}
    capped_pairs = []
    for rec in df.iter_rows(named=True):
        s = str(rec[src_c] or "").strip()
        t = str(rec[tgt_c] or "").strip()
        if not s or not t:
            raise CorridorRefusal(
                "scope table has a row with an empty source/target type")
        pair = (s, t)
        if pair in rows:
            raise CorridorRefusal(
                f"scope table has duplicate rows for {s}->{t}")
        max_hops = _scope_int(rec[max_c]) if max_c is not None else None
        lt_i = _scope_int(rec[lti_c]) if lti_c is not None else None
        lt = _scope_int(rec[lt_c]) if lt_c is not None else None
        lt_dataset = (str(rec[ltd_c]).strip()
                      if ltd_c is not None and rec[ltd_c] is not None
                      else None)
        status = (str(rec[status_c]).strip().lower()
                  if status_c is not None and rec[status_c] is not None
                  else None)
        if status is not None and status not in (
                "exhausted", "capped", "no_path", "validation_cap"):
            raise CorridorRefusal(
                f"scope table row {s}->{t} has unknown status {status!r}; "
                "expected exhausted | capped | no_path | validation_cap")
        extra = {}
        for name, col in vol_cols.items():
            if col is not None:
                extra[name] = _scope_int(rec[col])
        _vol_lts = [v for v in extra.values() if v is not None]
        _vol_max = max(_vol_lts) if _vol_lts else None
        if lt is None:
            # A multi-dataset table may carry only the per-volume Lt
            # columns: the common Lt is their maximum (issue §3 item 2).
            if _vol_max is not None:
                lt = _vol_max
        elif _vol_max is not None and _vol_max > int(lt):
            raise CorridorRefusal(
                f"scope table row {s}->{t} declares Lt={lt} below its "
                f"volume maximum {_vol_max}; Lt must be max_i Lt_i")
        if max_hops is None and lt is not None:
            max_hops = int(lt) + int(scope_pad)
        if max_hops is not None and max_hops < 0:
            raise CorridorRefusal(
                f"scope table max_hops for {s}->{t} is negative")
        if max_hops == 0:
            max_hops = None
        if lt_i is not None and int(lt_i) < 1:
            raise CorridorRefusal(
                f"scope table row {s}->{t} has Lt_i={lt_i}; a hop count "
                "must be at least 1")
        if lt is not None and int(lt) < 1:
            raise CorridorRefusal(
                f"scope table row {s}->{t} has Lt={lt}; a hop count "
                "must be at least 1")
        if lt_i is not None and max_hops is None:
            raise CorridorRefusal(
                f"scope table row {s}->{t} declares Lt_i={lt_i} but no "
                "max_hops/scope_bound")
        if lt is not None and max_hops is None:
            raise CorridorRefusal(
                f"scope table row {s}->{t} declares Lt={lt} but no "
                "max_hops/scope_bound")
        if lt_i is not None and max_hops < int(lt_i):
            raise CorridorRefusal(
                f"scope table max_hops={max_hops} for {s}->{t} is below "
                f"its local Lt_i={lt_i}")
        if lt is not None and max_hops < int(lt):
            raise CorridorRefusal(
                f"scope table max_hops={max_hops} for {s}->{t} is below "
                f"its common Lt={lt}")
        hard_capped = False
        if max_hops is not None and hard_cap is not None \
                and max_hops > int(hard_cap):
            max_hops = int(hard_cap)
            hard_capped = True
        if status == "capped":
            hard_capped = True
        # The cap is what makes a pair incomplete when Lt + pad exceeds it,
        # even if the table already clamped max_hops to the cap. The common
        # Lt (when supplied) is the cross-dataset maximum, so it governs.
        _lt_for_cap = lt if lt is not None else lt_i
        if (_lt_for_cap is not None and hard_cap is not None
                and int(_lt_for_cap) + int(scope_pad) > int(hard_cap)):
            hard_capped = True
        if hard_capped:
            capped_pairs.append(pair)
        bounds[pair] = max_hops
        _lt_ref = lt if lt is not None else lt_i
        _eff_pad = (int(max_hops) - int(_lt_ref)
                    if (max_hops is not None and _lt_ref is not None)
                    else None)
        rows[pair] = {
            "source_type": s, "target_type": t,
            "Lt_i": lt_i, "Lt": lt, "Lt_dataset": lt_dataset or None,
            "scope_bound": max_hops,
            "scope_pad_effective": _eff_pad,
            "status": status,
            "hard_capped": hard_capped, "discovery_cap_hit": False,
            "type_shortest_hops": None, "phantom_attempts": 0,
            "first_phantom_type_path": None,
            "witness_type_path": None,
            **_empty_scope_counts(), **extra,
        }
    missing = sorted(selected - set(rows))
    extra_pairs = sorted(set(rows) - selected)
    if missing:
        shown = ", ".join(f"{s}->{t}" for s, t in missing[:6])
        raise CorridorRefusal(
            f"scope table does not cover every selected pair "
            f"({len(missing)} missing: {shown})")
    if extra_pairs:
        shown = ", ".join(f"{s}->{t}" for s, t in extra_pairs[:6])
        raise CorridorRefusal(
            f"scope table has pairs this run did not select "
            f"({len(extra_pairs)} extra: {shown})")
    block = {
        "scope_mode": "per_pair_shortest_plus_2",
        "scope_lt": "external_table",
        "scope_pad": int(scope_pad),
        "scope_hard_cap": (int(hard_cap) if hard_cap is not None else None),
        "scope_bound_source": "external_table",
        "scope_table_sha": hashlib.md5(raw).hexdigest()[:12],
        "scope_table_source": str(p),
        "scope_pairs_total": len(rows),
        "scope_pairs_with_lt": sum(
            1 for r in rows.values() if r["Lt_i"] is not None),
        "scope_capped_pairs": sorted(
            f"{s}->{t}" for s, t in capped_pairs),
        "scope_lt_evidence": "external scope table (Lt/Lt_i supplier)",
    }
    return rows, bounds, block


def _write_scope_table(out, rows, *, scope_pad, hard_cap, type_map_sha,
                       asked, threshold, dataset):
    """Write the reusable Pass-A table and return its sha12."""
    import hashlib
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for pair in sorted(rows):
        row = rows[pair]
        recs.append({
            "source_type": row["source_type"],
            "target_type": row["target_type"],
            "Lt_i": row.get("Lt_i"),
            "Lt": row.get("Lt"),
            "Lt_dataset": row.get("Lt_dataset") or (dataset or ""),
            "max_hops": row.get("scope_bound"),
            "status": row.get("status"),
            "type_map_sha": type_map_sha,
            "asked": asked,
            "threshold": threshold,
            "scope_pad": int(scope_pad),
            "scope_hard_cap": (int(hard_cap) if hard_cap is not None else ""),
            "scope_mode": "per_pair_shortest_plus_2",
            "scope_lt": "max_local_validated",
        })
    path = out / "scope_table.csv"
    pd.DataFrame(recs).to_csv(path, index=False)
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


def _enumerate_type_walks(pairs, sources, targets, bound, *,
                          max_walks=2_000_000, revisit_policy="bounded",
                          max_type_visits=0, log=None, pair_bounds=None):
    """The COMPLETE type-walk set s -> ... -> t over the pruned pair table.

    The first-target rule is structural: a walk is recorded when it reaches
    a target and is NOT expanded through that target, so every walk carries
    exactly one clock token at its final node (for the clock lane). Sources
    that are themselves targets are skipped (a clock at position 0 is not
    an input path).

    ``revisit_policy='bounded'`` (default) permits up to
    ``max_type_visits`` visits of one type; ``'simple'`` forbids repeats (the
    article's complete tables can carry type revisits realized by
    different bodies — P1-I). The default is bounded, disclosed in the
    manifest. Returns (walks, stats); stats['capped'] marks a
    ``max_walks`` stop, which forbids a completeness claim.

    ``pair_bounds`` (per-pair scope) maps each selected (source, target)
    type pair to its hop bound (or None for an unreached pair). When it is
    given, a target is recorded only within ITS pair's bound and a prefix
    is pruned unless at least one target is still reachable within that
    target's bound. With it absent the behaviour is the original global
    ``bound`` pass byte-for-byte (the regression contract)."""
    adj = _type_adjacency(pairs)
    tgt = {str(t) for t in targets}
    srcs = sorted({str(s) for s in sources})
    scoped = pair_bounds is not None
    if scoped:
        pb = { (str(s), str(t)): (None if b is None else int(b))
               for (s, t), b in pair_bounds.items() }
        _positive = [b for b in pb.values() if b is not None]
        max_bound = max(_positive) if _positive else 0
        dist_by_t = {
            t: type_bfs(pairs, {t}, "bwd", max_bound)
            for t in tgt} if pairs.height and max_bound > 0 else {}
        bound = int(max_bound)
    else:
        pb = None
        max_bound = int(bound)
        dist = type_bfs(pairs, tgt, "bwd", bound) if pairs.height else {}
    # bounded: a type may appear at most as often as the hop bound allows
    # (0 = "use the bound"). This is the COMPLETE revisit set for walks up
    # to `bound`; a smaller positive value is a disclosed sub-policy.
    _max_visits = (int(max_type_visits) if int(max_type_visits) > 0
                   else (max(1, int(max_bound)) if scoped
                         else int(max_bound)))
    walks: list = []
    capped = False
    expanded = 0

    def _target_allowed(s, t, hops):
        if not scoped:
            return True
        b = pb.get((s, t))
        return b is not None and int(hops) <= b

    def _prefix_feasible(s, node, depth):
        if not scoped:
            d = dist.get(node)
            return d is not None and d <= int(max_bound) - int(depth)
        for t in tgt:
            b = pb.get((s, t))
            if b is None:
                continue
            d = dist_by_t.get(t, {}).get(node)
            if d is not None and int(depth) + int(d) <= b:
                return True
        return False

    def _dfs(node, walk, visits, remaining, source):
        nonlocal capped, expanded
        for nxt, _w in adj.get(node, ()):
            if len(walks) >= max_walks:
                capped = True
                return
            if nxt in tgt:
                if _target_allowed(source, nxt, len(walk)):
                    walks.append(tuple(walk + (nxt,)))
                # FIRST-TARGET RULE: never continue through a clock
                continue
            if remaining <= 1:
                continue
            depth = len(walk)
            if not _prefix_feasible(source, nxt, depth):
                continue
            if revisit_policy == "simple":
                if nxt in visits:
                    continue
                nv = dict(visits)
                nv[nxt] = 1
                _dfs(nxt, walk + (nxt,), nv, remaining - 1, source)
            elif revisit_policy == "bounded":
                if visits.get(nxt, 0) >= _max_visits:
                    continue
                nv = dict(visits)
                nv[nxt] = nv.get(nxt, 0) + 1
                _dfs(nxt, walk + (nxt,), nv, remaining - 1, source)
            else:
                raise ValueError(
                    f"unknown revisit policy {revisit_policy!r}")
            expanded += 1

    for s in srcs:
        if s in tgt:
            continue
        if scoped and not any(
                b is not None for (ss, _t), b in pb.items() if ss == s):
            continue
        _dfs(s, (s,), {s: 1}, int(max_bound), s)
        if capped:
            break
    stats = {
        "sources": srcs, "targets": sorted(tgt),
        "adj_nodes": len(adj),
        "adj_edges": int(pairs.height),
        "walks": len(walks),
        "expanded_nodes": expanded,
        "capped": capped,
        "max_walks": int(max_walks),
        "revisit_policy": revisit_policy,
        "max_type_visits": (_max_visits
                            if revisit_policy == "bounded" else 1),
        "scope": "per_pair" if scoped else "global",
        "max_bound": int(max_bound),
    }
    return walks, stats


def _type_prune_completeness(*, enum_capped, time_capped, unverified,
                              ratio_unresolved):
    """The typed lane's completeness contract (issue 2026-10-22 §2.2).

    Complete means the pruned graph was exhausted and every candidate got
    a definitive layered forward/backward verdict, including a computable
    realized adjusted ratio for the exact filter. `filtered_weak` is the
    deliberate exact gate and does not break completeness; an unresolved
    ratio does. No witness sample/envelope exists in this lane (issue
    2026-10-23).
    Returns (complete, filter_complete, reason)."""
    filter_complete = ratio_unresolved == 0
    if enum_capped:
        return False, filter_complete, "type_walk_cap"
    if time_capped:
        return False, filter_complete, "time_budget"
    if unverified:
        return False, filter_complete, "unverified"
    if not filter_complete:
        return False, filter_complete, "ratio_unresolved"
    return True, True, "exhausted"


def _run_type_pruned(*, walk_rows, scope_pairs, pairs,
                     pair_edges, src_types, tgt_types, src_ids, tgt_ids,
                     src_keep, tgt_keep, body2type, bound,
                     type_membership, type_incoming, type_incoming_adj,
                     type_nt, min_ratio_adj,
                     walk_cap, revisit_policy,
                     deadline, prune_block,
                     max_type_visits=0, log=None, out_dir=None,
                     pair_bounds=None, scope_seed=None,
                     scope_context=None):
    """The TYPE-PRUNED COMPLETE drain (Phase B steps 3-6).

    Completeness is over the PRUNED TYPE GRAPH: every enumerated walk is
    assigned exactly one terminal state (phantom / unverified / validated
    but exact-filtered / exported), and no walk is hidden by a per-class
    share. Validation is ONE layered forward/backward pass per walk (issue
    2026-10-23); there is no bodyId witness enumeration, no realization
    envelope, and no bodyId CSV. ``type_pruned_complete`` is true only when
    the enumeration exhausted and every walk got a layered verdict. With
    ``pair_bounds`` it is true only when EVERY selected pair also
    exhausted within its own bound; the manifest's
    ``scope_complete_by_pair`` is the per-pair contract."""
    import time
    from pathlib import Path

    say = log or (lambda *a, **k: None)
    started = time.monotonic()
    s_bodies_by_type: dict = {}
    for b in src_ids:
        t = body2type.get(b)
        if t in src_types and (not src_keep or b in src_keep):
            s_bodies_by_type.setdefault(t, []).append(b)
    t_bodies_by_type: dict = {}
    for b in tgt_ids:
        t = body2type.get(b)
        if t in tgt_types and (not tgt_keep or b in tgt_keep):
            t_bodies_by_type.setdefault(t, []).append(b)

    scoped = pair_bounds is not None
    selected_pairs = ([(s, t) for s in sorted({str(x) for x in src_types})
                       for t in sorted({str(x) for x in tgt_types})]
                      if scoped else [])
    scope_stats = ({p: _empty_scope_counts() for p in selected_pairs}
                   if scoped else {})
    walks, enum = _enumerate_type_walks(
        pairs, src_types, set(tgt_types), bound,
        max_walks=walk_cap, revisit_policy=revisit_policy,
        max_type_visits=max_type_visits, log=log,
        pair_bounds=pair_bounds)
    say(f"  · type-pruned complete: {enum['walks']} candidate walks over "
        f"{enum['adj_edges']} kept pairs"
        + (" (CAPPED)" if enum["capped"] else " (exhausted)"))

    pair_lookup = {(str(a), str(b)): (float(ws), float(wm), int(n))
                   for a, b, ws, wm, n in
                   zip(pairs["type_pre"], pairs["type_post"],
                       pairs["w_sum"], pairs["w_max"], pairs["n"])}

    time_capped = False
    validated = phantoms = unverified = 0
    filtered_weak = ratio_unresolved = 0
    accepted = chains_revisit = 0
    for walk in walks:
        if deadline is not None and time.monotonic() > deadline:
            time_capped = True
            break
        _pair_key = (str(walk[0]), str(walk[-1]))
        _stat = scope_stats.get(_pair_key)
        if _stat is not None:
            _stat["candidate_walks"] += 1
        s_bodies = s_bodies_by_type.get(walk[0], [])
        t_bodies = set(t_bodies_by_type.get(walk[-1], []))
        if not s_bodies or not t_bodies:
            unverified += 1
            if _stat is not None:
                _stat["unverified"] += 1
            continue
        got = _layered_row(
            walk, pair_edges, s_bodies, t_bodies, body2type,
            type_incoming, type_incoming_adj, type_membership,
            type_nt, pair_lookup)
        if got is None:
            phantoms += 1
            if _stat is not None:
                _stat["phantoms"] += 1
            continue
        row, hop_edges = got
        validated += 1
        if _stat is not None:
            _stat["validated_walks"] += 1
        if row["min_ratio_adj"] is None:
            # The adjusted readout is incomplete whether or not an exact
            # gate is active. With a gate the row is dropped; without one
            # it is exported with ratio_adj_complete=false, and the run
            # cannot claim filter_complete.
            ratio_unresolved += 1
            if _stat is not None:
                _stat["ratio_unresolved"] += 1
            if min_ratio_adj > 0:
                continue
        elif min_ratio_adj > 0 and float(
                row["min_ratio_adj"]) <= float(min_ratio_adj):
            filtered_weak += 1
            if _stat is not None:
                _stat["filtered_weak"] += 1
            continue
        walk_rows.append(row)
        accepted += 1
        if _stat is not None:
            _stat["paths"] += 1
        if row.get("chain_revisit"):
            chains_revisit += 1
        h = len(walk) - 1
        for i in range(h):
            key = (walk[i], walk[i + 1])
            # Retain only the summed selected mass for the typed lane; the
            # per-edge union of every accepted walk was a large retained
            # structure and is not needed by any export (issue 2026-10-23).
            rec = scope_pairs.setdefault(
                key, {"mass": 0.0, "paths": 0, "walks": 0,
                      "bodyid_paths": 0})
            rec["mass"] += float(sum(hop_edges[i].values()))
            rec["paths"] += 1
            rec["walks"] += 1
    block = dict(prune_block or {})
    block.update({
        "mode": "type_pruned_complete",
        "walk_mode": "complete",
        "revisit_policy": revisit_policy,
        "max_type_visits": enum.get("max_type_visits"),
        "type_walk_cap": int(walk_cap),
        "candidate_walks": enum["walks"],
        "enumeration_capped": bool(enum["capped"]),
        "enumeration": enum,
        "type_self_edges": int((pairs["type_pre"]
                                == pairs["type_post"]).sum()),
        "chains_revisit": chains_revisit,
        "revisit_semantics": "layered_fwd_bwd_superset",
        "revisit_caveat": (
            "a type-revisiting walk's kept edges are a layered "
            "forward/backward superset of body-simple realizations "
            "(a body may occupy two positions); disclosed per row via "
            "chain_revisit"),
        "first_target_rule": ("exactly one target token, at the final "
                              "node; targets are not expanded"),
        "validated_walks": validated,
        "phantoms": phantoms,
        "unverified": unverified,
        "filtered_weak": filtered_weak,
        "ratio_unresolved": ratio_unresolved,
        "accepted_paths": accepted,
        "bodyId_paths": 0,
        "bodyid_export": ("disabled (layered validation; no bodyId "
                          "witness enumeration)"),
        "validation_basis": "layered_fwd_bwd",
        "exact_filter": {"basis": "realized_min_ratio_adj",
                         "threshold": float(min_ratio_adj),
                         "strictly_greater": True},
        "validation_source": "lossless_prune_full_floor",
        "time_capped": bool(time_capped),
        "elapsed_s": round(time.monotonic() - started, 2),
    })
    # Issue 2026-10-22 §2.2: a validated walk whose realized adjusted
    # ratio is unresolved is a path the exact filter could not adjudicate.
    # The lane is not complete if that happened. `filter_complete` names the
    # same fact separately; `filtered_weak` is the DELIBERATE exact gate and
    # does not break completeness.
    complete, filter_complete, completeness_reason = \
        _type_prune_completeness(
            enum_capped=enum["capped"], time_capped=time_capped,
            unverified=unverified, ratio_unresolved=ratio_unresolved)
    scope_complete = None
    scope_complete_by_pair = None
    scope_capped_pairs = []
    scope_no_path_pairs = []
    scope_validation_cap_pairs = []
    scope_exhausted_pairs = []
    if scoped:
        scope_complete_by_pair = {}
        scope_rows_out = []
        for pair in selected_pairs:
            rec = scope_stats.get(pair, _empty_scope_counts())
            seed = (scope_seed or {}).get(pair, {})
            b = (pair_bounds or {}).get(pair)
            seed_status = str(seed.get("status") or "").lower()
            if b is None:
                # No validated path: never assign a global bound. A
                # discovery cap/validation cap is reported as such.
                status = (seed_status if seed_status in
                          ("capped", "validation_cap") else "no_path")
            elif (enum["capped"] or time_capped
                  or bool(seed.get("hard_capped"))
                  or seed_status == "capped"):
                status = "capped"
            elif seed_status == "validation_cap":
                # The supplied scope table itself was derived from
                # incomplete validation evidence. Even if this run's
                # enumeration exhausts at that bound, the scope is not
                # certifiable as complete.
                status = "validation_cap"
            elif rec["unverified"] or rec["ratio_unresolved"]:
                status = "validation_cap"
            elif rec["validated_walks"] > 0:
                status = "exhausted"
            else:
                status = "no_path"
            scope_complete_by_pair[f"{pair[0]}->{pair[1]}"] = status
            if status == "capped":
                scope_capped_pairs.append(f"{pair[0]}->{pair[1]}")
            elif status == "no_path":
                scope_no_path_pairs.append(f"{pair[0]}->{pair[1]}")
            elif status == "validation_cap":
                scope_validation_cap_pairs.append(
                    f"{pair[0]}->{pair[1]}")
            elif status == "exhausted":
                scope_exhausted_pairs.append(f"{pair[0]}->{pair[1]}")
            row = {
                "source_type": pair[0], "target_type": pair[1],
                "scope_mode": "per_pair_shortest_plus_2",
                "Lt_i": seed.get("Lt_i"), "Lt": seed.get("Lt"),
                "Lt_dataset": seed.get("Lt_dataset"),
                "scope_pad": (scope_context or {}).get("scope_pad"),
                "scope_pad_effective": (
                    seed.get("scope_pad_effective")
                    if seed.get("scope_pad_effective") is not None else
                    (int(b) - int(seed["Lt_i"])
                     if (b is not None and seed.get("Lt_i") is not None)
                     else None)),
                "scope_bound": b,
                "hard_capped": bool(seed.get("hard_capped")),
                "status": status,
                "scope_complete": status == "exhausted",
                "paths": rec["paths"],
                "bodyid_paths": rec["bodyid_paths"],
                "candidate_walks": rec["candidate_walks"],
                "validated_walks": rec["validated_walks"],
                "phantoms": rec["phantoms"],
                "filtered_weak": rec["filtered_weak"],
                "ratio_unresolved": rec["ratio_unresolved"],
                "discovery_cap_hit": bool(
                    seed.get("discovery_cap_hit")),
                "phantom_attempts": int(seed.get("phantom_attempts") or 0),
                "first_phantom_type_path": seed.get(
                    "first_phantom_type_path"),
                "witness_type_path": seed.get("witness_type_path"),
                "type_shortest_hops": seed.get("type_shortest_hops"),
                "applied_floor": (scope_context or {}).get(
                    "requested_floor"),
                "threshold": float(min_ratio_adj),
                "type_map_sha": (scope_context or {}).get("type_map_sha"),
                "asked": (scope_context or {}).get("requested_floor"),
            }
            for extra in ("Lt_FAFB", "Lt_MCNS", "Lt_BANC"):
                if extra in seed:
                    row[extra] = seed.get(extra)
            # The per-pair report names one Lt column per volume. DROCAT
            # runs one dataset, so it can fill its OWN volume's column even
            # when the input table did not carry the Lt_<volume> family.
            _family = _scope_dataset_family(
                (scope_context or {}).get("dataset"))
            if _family:
                _local_col = f"Lt_{_family}"
                _lt_i = seed.get("Lt_i")
                _lt_ds = seed.get("Lt_dataset")
                if (row.get(_local_col) is None
                        and _lt_i is not None
                        and (not _lt_ds
                             or _scope_dataset_family(_lt_ds) in
                             (None, _family))):
                    # Only a LOCAL Lt_i fills the run's volume column; the
                    # common Lt may belong to a different dataset.
                    row[_local_col] = _lt_i
            scope_rows_out.append(row)
        scope_complete = bool(
            selected_pairs and all(
                scope_complete_by_pair[f"{s}->{t}"] == "exhausted"
                for s, t in selected_pairs))
        block.update(scope_context or {})
        block.update({
            "scope_complete": bool(scope_complete),
            "scope_complete_by_pair": scope_complete_by_pair,
            "scope_capped_pairs": sorted(scope_capped_pairs),
            "scope_no_path_pairs": sorted(scope_no_path_pairs),
            "scope_validation_cap_pairs": sorted(
                scope_validation_cap_pairs),
            "scope_exhausted_pairs": sorted(scope_exhausted_pairs),
            "scope_pairs_total": len(selected_pairs),
            "scope_pairs_exhausted": len(scope_exhausted_pairs),
            "scope_report_table": "route_scope_pairs.csv",
            "scope_input_table": (
                "scope_table.csv"
                if (out_dir is not None
                    and (Path(out_dir) / "scope_table.csv").exists())
                else (scope_context or {}).get("scope_table_source")),
            "scope_replaces_max_interlayer": True,
        })
        if out_dir is not None:
            import pandas as _pd_scope
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            cols = ["source_type", "target_type", "scope_mode", "Lt_i",
                    "Lt", "Lt_dataset", "Lt_FAFB", "Lt_MCNS", "Lt_BANC",
                    "scope_pad", "scope_pad_effective", "scope_bound",
                    "hard_capped", "status",
                    "scope_complete", "paths", "bodyid_paths",
                    "candidate_walks", "validated_walks", "phantoms",
                    "filtered_weak",
                    "ratio_unresolved", "discovery_cap_hit",
                    "phantom_attempts", "first_phantom_type_path",
                    "witness_type_path",
                    "type_shortest_hops", "applied_floor", "threshold",
                    "type_map_sha", "asked"]
            _pd_scope.DataFrame(
                [{c: r.get(c) for c in cols} for r in scope_rows_out],
                columns=cols).to_csv(
                    out_dir / "route_scope_pairs.csv", index=False)
    if scoped:
        block["type_pruned_complete"] = bool(complete and scope_complete)
        if complete and not scope_complete:
            if scope_capped_pairs:
                completeness_reason = "per_pair_capped"
            elif scope_no_path_pairs:
                completeness_reason = "per_pair_no_path"
            elif scope_validation_cap_pairs:
                completeness_reason = "per_pair_validation_cap"
            else:
                completeness_reason = "per_pair_incomplete"
        block["completeness_scope"] = (
            "per-pair validated-shortest scope: each selected "
            "(source,target) pair is enumerated to min(Lt + pad, hard cap) "
            "hops; every walk is validated by one layered forward/backward "
            "pass; complete means exhausted WITHIN that per-pair scope only "
            "(never 'complete up to max_interlayer')")
    else:
        block["type_pruned_complete"] = bool(complete)
        block["completeness_scope"] = (
            "pruned type graph enumerated + every walk validated by one "
            "layered forward/backward pass over the full-floor body graph; "
            "the exact realized min_ratio_adj gate applied to the "
            "validated set")
    block["filter_complete"] = bool(filter_complete)
    block["completeness_reason"] = completeness_reason
    say(f"  · type-pruned complete: {validated} validated / "
        f"{phantoms} phantoms / {unverified} unverified / "
        f"{filtered_weak} weak-filtered -> {accepted} paths "
        f"(complete={complete}"
        + (f", scope_complete={scope_complete}" if scoped else "")
        + ")")
    return block


# ------------------------------------------------------------- driver
def run_lane(run_dir=None, corridor_mode="both", top_m_base=10,
             length_decay=1.5, per_pair_length_budget=100, fnc=None,
             asked=None, max_interlayer=None, source_types=None,
             target_types=None, connections_path=None, neuron_table=None,
             out_dir=None, dataset=None, trigger=None,
             write_bodyid=True, full_scope=None, time_budget_s=900.0,
             allow_multi_visit=False, certify_actual=False,
             type_pruned=False, type_prune_basis="ratio_adj",
             type_min_ratio_adj=0.01, type_min_synapse=0,
             type_min_wmax=0, type_walk_mode="complete",
             type_walk_cap=2_000_000, type_revisit_policy="bounded",
             type_max_type_visits=0,
             type_validate_work_cap=2_000_000,
             type_prune_scope="full_floor", per_pair_scope=False,
             scope_table=None, scope_pad=2, scope_hard_cap=6,
             scope_lt_mode=None, scope_discovery_cap=6,
             infer_type_pruned=True):
    """Standalone corridor lane (plan §23.3/§23.5): G_type* -> shared W ->
    per-source drain -> validate loop -> PER-WALK subgraph enumeration
    (§22 share-consumption) -> route_scope_paths/ exports.

    Returns True on success (lane-namespace outputs only). Raises
    CorridorRefusal (after writing a status='refused' provenance) when the
    run's universe cannot be reproduced. Reads the run folder's
    parameters.txt (spaced keys) when run_dir is given; direct mode
    (run_dir=None) requires explicit asked/max_interlayer/sources/targets.
    ``max_interlayer`` is the run's own knob; the lane's hop bound is
    max_interlayer + 1.

    ``type_pruned=True`` selects the TYPE-PRUNED COMPLETE lane
    (plan-drocat-type-pruned-pathfinding Phases B/D): prune the type graph
    on ``type_prune_basis`` (pair-level coarse filter), enumerate EVERY
    simple type walk over the pruned graph, and validate each with ONE
    layered forward/backward pass over the full-floor lossless-pruned body
    graph (issue 2026-10-23). No strongest-first witness enumeration, no
    bodyId witness rows, and no realization envelope: a walk is a phantom
    iff the forward ∩ backward intersection is empty, and the kept per-hop
    edges supply the exact ``min_ratio_adj > type_min_ratio_adj`` gate.
    No top-M share and no class quota: every validated walk that clears
    the exact filter is exported. ``type_pruned_complete`` is true only
    when the type-walk enumeration exhausted and every walk got a layered
    verdict. This mode requires path_mode='all' (a shortest-run discovery
    store is not a full-floor bodyId source). It writes no
    ``route_paths_bodyId.csv`` and skips the additive synapse >= 1 second
    round unless ``full_scope=True`` (``--full-scope``); the Fast lane
    keeps its previous second-round default.

    ``per_pair_scope=True`` replaces the single global hop bound with a
    per-(source_type,target_type) scope. ``Lt_i`` is the shortest
    bodyId-validated walk clearing the exact filter (never a raw type-BFS
    shortest: a phantom walk advances the pass), ``bound(pair) = min(Lt +
    scope_pad, scope_hard_cap)``, and a pair with no validated path gets
    ``bound=None`` and is reported unreached. ``scope_lt_mode='discover'``
    computes the scope locally and writes ``scope_table.csv``;
    ``'external'`` reads a supplied table (``scope_table``). The
    per-pair table ``route_scope_pairs.csv`` and the manifest state
    ``scope_complete_by_pair``; ``type_pruned_complete`` is then true only
    WITHIN that per-pair scope. With ``per_pair_scope=False`` the global
    ``max_interlayer + 1`` behaviour is unchanged."""
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
    # P2-S round-trip: a folder stamped by a typed run reproduces the lane
    # on a bare `CorridorDeepDive.py <run_dir>` re-run. Explicit CLI flags
    # still win for the knobs not present in the folder. The in-run FAST
    # subprocess disables this inference: it runs on a typed-stamped folder
    # and must not write typed tables into route_scope_paths/.
    if run_dir and infer_type_pruned and not type_pruned and _parse_bool(
            kv.get("type_pruned"), False):
        type_pruned = True
        type_prune_basis = (kv.get("type prune basis")
                            or type_prune_basis)
        type_prune_scope = (kv.get("type prune scope")
                            or type_prune_scope)
        type_min_ratio_adj = _parse_float(
            kv.get("type min ratio adj"), type_min_ratio_adj)
        type_min_synapse = _parse_int(
            kv.get("type min synapse"), type_min_synapse)
        type_min_wmax = _parse_int(
            kv.get("type min wmax"), type_min_wmax)
        type_walk_mode = (kv.get("type walk mode") or type_walk_mode)
        type_walk_cap = _parse_int(kv.get("type walk cap"), type_walk_cap)
        type_revisit_policy = (kv.get("type revisit policy")
                               or type_revisit_policy)
        type_max_type_visits = _parse_int(
            kv.get("type max type visits"), type_max_type_visits)
        type_validate_work_cap = _parse_int(
            kv.get("type validate work cap"), type_validate_work_cap)
    # A Fast-lane subprocess on a typed-stamped folder passes
    # --no-infer-type-pruned: it must not adopt the per-pair scope stamp
    # either, or a scoped typed run's Fast lane would refuse as typed.
    if (run_dir and infer_type_pruned and not per_pair_scope
            and _parse_bool(kv.get("per pair scope"), False)):
        per_pair_scope = True
        scope_lt_mode = (kv.get("scope lt mode") or scope_lt_mode)
        scope_table = (kv.get("scope table") or scope_table)
        scope_pad = _parse_int(kv.get("scope pad"), scope_pad)
        scope_hard_cap = _parse_int(
            kv.get("scope hard cap"), scope_hard_cap)
        scope_discovery_cap = _parse_int(
            kv.get("scope discovery cap"), scope_discovery_cap)
        if scope_table and not Path(scope_table).is_absolute():
            _cand = Path(run_dir) / scope_table
            if _cand.exists():
                scope_table = str(_cand)
        if scope_table and not Path(scope_table).exists():
            # Portable replay: an external run copies its supplied table
            # into the lane folder. If the originally recorded path is
            # gone, that byte-identical copy is the audit witness.
            _copy = Path(run_dir) / TYPED_LANE_DIR / "scope_table.csv"
            if _copy.exists():
                scope_table = str(_copy)
    # D2-B: a run folder can request certified actual-strength selection
    # via parameters.txt (the CLI passes --certify-actual; the in-run
    # hook keeps the default off)
    certify_actual = certify_actual or _parse_bool(
        kv.get("certify actual"), False)
    trigger = trigger or ("in-run" if fnc is not None else
                          "run-folder" if run_dir else "driver-direct")
    # R3 (plan-corridor-refill-and-mode-tabs): `corridor_mode` is what the
    # CALLER passed; `run_mode` is the USER's resolved selection (the engine
    # stamps it into parameters.txt / all_attributes.json). A reader needs
    # the second one to tell "Complete only was selected" from "the lane
    # was selected and refused" — the report's tab-presence rule keys on it.
    run_mode = str(corridor_mode or "both").lower()
    _rm = getattr(fnc, "_resolved_run_mode", None)
    if callable(_rm):
        try:
            run_mode = _rm()
        except Exception:  # noqa: BLE001 - provenance, never fatal
            pass
    elif kv.get("run mode"):
        run_mode = str(kv["run mode"]).strip().lower() or run_mode
    out = Path(out_dir) if out_dir else (
        Path(run_dir) / (TYPED_LANE_DIR if type_pruned else FAST_LANE_DIR)
        if run_dir else None)
    if out is None:
        raise CorridorRefusal("no output directory (run_dir or out_dir)")

    def _refuse(reason):
        out.mkdir(parents=True, exist_ok=True)
        (out / "route_provenance.json").write_text(json.dumps(
            {"status": "refused", "reason": str(reason),
             "corridor_mode": corridor_mode, "run_mode": run_mode,
             "trigger": trigger},
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
    # ---- per-pair scope configuration -------------------------------
    # The switched-off path keeps the global bound above byte-for-byte.
    scope_mode_resolved = None
    _scope_hard_cap = None
    # The typed lane's additive synapse >= 1 second round is opt-in; when
    # skipped, its full connection/typed frames are released before the
    # enumeration (memory bound, issue 2026-10-23).
    full_scope_requested = ((not type_pruned) if full_scope is None
                            else bool(full_scope))
    if per_pair_scope:
        if not type_pruned:
            _refuse("per-pair scope requires the type-pruned complete lane")
        scope_pad = int(scope_pad if scope_pad is not None else 2)
        if scope_pad < 0:
            _refuse(f"scope pad {scope_pad} is negative")
        scope_hard_cap = int(
            scope_hard_cap if scope_hard_cap is not None else 6)
        if scope_hard_cap < 0:
            _refuse(f"scope hard cap {scope_hard_cap} is negative")
        _scope_hard_cap = (scope_hard_cap if scope_hard_cap > 0 else None)
        scope_discovery_cap = int(
            scope_discovery_cap if scope_discovery_cap is not None else 6)
        if scope_discovery_cap < 1:
            _refuse("scope discovery cap must be at least 1 hop")
        scope_lt_mode = str(
            scope_lt_mode or ("external" if scope_table else "discover")
        ).strip().lower()
        if scope_lt_mode not in ("external", "discover"):
            _refuse(f"scope LT mode {scope_lt_mode!r} is not one of "
                    "external / discover")
        scope_mode_resolved = scope_lt_mode
        if scope_mode_resolved == "external" and not scope_table:
            _refuse("scope LT mode 'external' requires --scope-table")
        if scope_mode_resolved == "discover" and scope_table:
            _refuse("scope LT mode 'discover' computes its own table; "
                    "do not pass --scope-table")
        if scope_mode_resolved == "external":
            scope_table = str(scope_table)
    # P2-P: the Complete run may have raised its applied floor via the Edge
    # Budget. The typed lane rebuilds from the connection source at the
    # ASKED floor, so record both rather than inheriting the floor silently.
    _run_applied_floor = _parse_int(kv.get("edge_weight_floor"))
    if _run_applied_floor is None:
        _run_applied_floor = _parse_int(kv.get("applied_threshold"))
    basis = str(kv.get("weight basis", "synapse") or "synapse").lower()
    ratio_basis = basis.startswith("connection_ratio")
    if type_pruned and ratio_basis:
        # The typed lane's chain mass and pair universe are SYNAPSE units;
        # on a ratio-basis run `typed.weight` is a per-edge ratio, so the
        # lossless cone's pair sums would be ratio sums, not mass. Refuse
        # rather than publish a silently wrong universe (the article lane
        # is the synapse >= 3 lane).
        _refuse("the type-pruned complete lane currently supports the "
                "synapse weight basis only (a ratio-basis run's lossless "
                "cone carries per-edge ratios, not synapse mass)")
    # F-PF-004 (audit 2026-10-13): the AND ratio parses as a FLOAT and a
    # non-numeric value refuses (never silently zero).
    _and_raw = str(kv.get("min connection ratio") or "").strip()
    _and_val = _parse_float(_and_raw) if _and_raw else None
    if _and_raw and _and_val is None:
        _refuse(f"min connection ratio {_and_raw!r} is not a number "
                "(refusing rather than treating it as zero)")
    if ratio_basis:
        # CORRIDOR-ON-RATIO (plan-corridor-ratio-basis §2, the adopted
        # design): E1 admits each edge on its own F9 input share
        # (w / W_v >= t_r, W_v = the post neuron's FULL-table incoming),
        # the drain ranks by the Et type-level ratios, and the
        # realization exports BOTH units (synapse masses + mean
        # realizing-edge ratios). t_r is the run's own min connection
        # ratio; the mass floor (min synapse number) composes in the
        # admission (R3 AND).
        if not _and_raw or _and_val is None or _and_val <= 0:
            _refuse("ratio-basis corridor requires the run's "
                    "'min connection ratio' (t_r) to be positive")
        asked = float(_and_val)
        t_ratio = float(_and_val)
        t_syn_floor = int(_parse_float(
            kv.get("min synapse number"), 1) or 1)
    else:
        t_ratio = None
        if _and_val is not None and _and_val > 0:
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
    if type_pruned:
        # P0-F / P2-P: the typed lane's completeness claim needs a
        # FULL-FLOOR bodyId source. A shortest run's discovery store is the
        # run's min-hop-restricted explored scope, not that source.
        if path_mode == "shortest":
            _refuse("the type-pruned complete lane requires path_mode='all' "
                    "(a shortest-run discovery store is not a full-floor "
                    "bodyId source)")
        if str(type_walk_mode).lower() != "complete":
            _refuse(f"type walk mode {type_walk_mode!r} is not supported; "
                    "only 'complete' can carry a completeness claim")
        if str(type_prune_scope).lower() != "full_floor":
            _refuse(f"type prune scope {type_prune_scope!r} is not "
                    "supported; only 'full_floor' is a full-floor universe")
        if str(type_prune_basis).lower() not in (
                "ratio_adj", "connection_ratio", "synapse"):
            _refuse(f"type prune basis {type_prune_basis!r} is not one of "
                    "ratio_adj / connection_ratio / synapse")
        if certify_actual:
            _refuse("certify_actual is a top-share mode and cannot be "
                    "combined with the type-pruned complete lane")
    store_universe = None
    store_distances = None   # F-PF-009 D3-B: per-target distance maps
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
    # P1-I/P1-K: the run's body-autapse rule. A bodyId self-edge is NOT a
    # type self-edge: the latter (X->X via two different bodies) must stay.
    exclude_autapses = (
        bool(getattr(fnc, "exclude_autapses", False))
        if fnc is not None
        else _parse_bool(kv.get("exclude autapses"), False))
    # The ENGINE's denominators are all-pres: they count the body autapses
    # even when the run's graph rule drops those self-edges. Keep a copy
    # before filtering so the lane's ratios match connection_type.csv.
    denom_conn = conn
    if exclude_autapses:
        conn = conn.filter(pl.col("bodyId_pre") != pl.col("bodyId_post"))
        _phase(f"conn-collapsed+autapses-removed ({conn.height:,} rows)")
    else:
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
    mapping_file_path = None
    if label_mapper is None:
        # subprocess mode: the run's recorded mapping file expands
        # custom-label query tokens (e.g. '5th-LNv') into the type map.
        # F-PF-010 (audit 2026-10-13): a RECORDED map that is absent must
        # refuse — falling back to native labels silently changes the
        # settled group membership, endpoint enrollment and every
        # exported label of the lane.
        _mf = (kv.get("custom mapping file")
               or "").strip()
        mapping_file_path = _mf or None
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
    # P0-D/P1-K: the ONE effective map this lane used, hashed. The label
    # mapper's own serialized map is hashed too when it is live; a recorded
    # mapping file that had to be loaded is covered by type_map_sha.
    import hashlib as _hashlib
    _tmh = _hashlib.md5()
    for _b, _t in sorted(type_map.items()):
        _tmh.update(f'{_b}\x1f{_t}\x1e'.encode('utf-8', 'replace'))
    type_map_sha = _tmh.hexdigest()[:12]
    # Issue 2026-10-22 §2.3: publish a REAL mapping witness. The recorded
    # file's bytes are the identity when there is one; a live mapper is
    # hashed only when it actually carries a map. An empty map is recorded
    # as `None` (the effective `type_map_sha` remains the reliable witness),
    # never as the MD5 of `{}`.
    label_mapper_sha = None
    label_mapper_source = 'none'
    if mapping_file_path:
        try:
            label_mapper_sha = _hashlib.md5(
                Path(mapping_file_path).read_bytes()).hexdigest()[:12]
            label_mapper_source = f'file:{mapping_file_path}'
        except OSError:
            label_mapper_sha = 'unreadable'
            label_mapper_source = f'file-unreadable:{mapping_file_path}'
    elif label_mapper is not None:
        # A live LabelMapper stores its groups in _source/_target/
        # _intermediate_mapping; ``overall_mapping`` does not exist on it.
        # Hashing only the latter made every live map 'live-empty' and hid
        # the grouped-vs-native mapping distinction from provenance.
        _overall = (getattr(label_mapper, 'overall_mapping', None)
                    or _label_mapper_payload(label_mapper) or {})
        if _overall:
            try:
                label_mapper_sha = _hashlib.md5(json.dumps(
                    _overall, sort_keys=True, default=str).encode()
                ).hexdigest()[:12]
                label_mapper_source = 'live'
            except Exception:
                label_mapper_sha = 'unhashable'
                label_mapper_source = 'live-unhashable'
        else:
            label_mapper_source = 'live-empty'

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

    # ---- per-pair scope: the working universe bound -------------------
    # The scope REPLACES the global max_interlayer+1 bound for this lane
    # only. External mode knows its maximum bound before the lossless cone
    # is built; discover mode builds at the discovery cap (a superset of
    # every bound it can later assign).
    scope_seed: dict = {}
    scope_bounds = None
    scope_block: dict = {}
    scope_working_bound = None
    scope_table_sha = None
    scope_table_source = None
    if per_pair_scope:
        if scope_mode_resolved == "external":
            try:
                scope_seed, scope_bounds, scope_block = _load_scope_table(
                    scope_table, src_types=src_types, tgt_types=tgt_types,
                    scope_pad=int(scope_pad),
                    hard_cap=_scope_hard_cap, type_map_sha=type_map_sha,
                    asked=asked, threshold=float(type_min_ratio_adj))
            except CorridorRefusal as exc:
                # Disclose the refusal in the lane namespace, exactly like
                # every other deterministic lane refusal.
                _refuse(str(exc))
            scope_table_sha = scope_block["scope_table_sha"]
            scope_table_source = scope_block["scope_table_source"]
            # Keep the supplied table beside the lane outputs, so the
            # scope is auditable/downloadable after the run (same bytes
            # the sha above commits to).
            try:
                import shutil as _shutil
                _src = Path(scope_table)
                _dst = out / "scope_table.csv"
                if _src.resolve() != _dst.resolve():
                    out.mkdir(parents=True, exist_ok=True)
                    _shutil.copyfile(_src, _dst)
            except OSError:
                pass  # the sha/source path still disclose the input
            _pos = [b for b in scope_bounds.values() if b is not None]
            scope_working_bound = max(_pos) if _pos else 1
        else:
            scope_working_bound = (
                min(int(scope_discovery_cap), int(_scope_hard_cap))
                if _scope_hard_cap is not None
                else int(scope_discovery_cap))
        scope_working_bound = max(1, int(scope_working_bound))
        bound = int(scope_working_bound)

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
            # F-PF-009 D3-B: the per-target distance maps load AFTER
            # tgt_ids exists (below); with them the basis becomes
            # shortest_dag_store_enforced and realization enforces
            # per-endpoint minimum hops; without them the lane stays on
            # the honestly-named edge union.
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
    if ratio_basis:
        # CORRIDOR-ON-RATIO universe (plan-corridor-ratio-basis §2):
        # E1 admits each edge on its own F9 input share
        # (weight / W_post >= t_r, W_post = the post neuron's FULL-table
        # incoming) AND the mass floor (min synapse number, R3 AND);
        # typed carries BOTH units (weight = the per-edge RATIO -- the
        # enumeration space -- and weight_syn = the synapse mass -- the
        # bookkeeping space); the drain's per-pair weight (w_max) is the
        # ET type-level ratio = pair admitted synapse mass over the post
        # TYPE's full-membership incoming.
        # the RAW cache carries per-ROI slice rows: collapse to one row
        # per bodyId pair FIRST -- the F9 ratio is a property of the
        # pair's TOTAL weight, and per-slice admission would admit/drop
        # slices of the same edge inconsistently
        conn_pairs = conn.group_by(["bodyId_pre", "bodyId_post"]).agg(
            pl.col("weight").sum().alias("weight"))
        # The ratio denominator is the engine's all-pres universe; a
        # filtered self-edge must still contribute its incoming mass.
        denom_pairs = denom_conn.group_by(
            ["bodyId_pre", "bodyId_post"]).agg(
                pl.col("weight").sum().alias("weight"))
        W_post = denom_conn.group_by("bodyId_post").agg(
            pl.col("weight").sum().alias("_W"))
        conn_r = (conn_pairs.join(W_post, on="bodyId_post", how="left")
                  .with_columns(
                      (pl.col("weight") / pl.col("_W")).alias("_ratio"))
                  .drop("_W")
                  .filter((pl.col("_ratio") >= t_ratio)
                          & (pl.col("weight") >= t_syn_floor)))
        typed = build_type_pairs(conn_r, tmap, 0,
                                 drop_untyped=drop_untyped)[0]
        # The conditioned denominator is NOT the ratio-gated cone: the
        # engine conditions `connection_ratio_adj` at the MIN-SYNAPSE tier
        # (`thr = max(1, min_synapse_num)`), and a ratio-basis run forces
        # that to 1 — so its adjusted denominator is the plain all-inputs
        # one and adj == plain by construction (gate L1 covers every basis,
        # not only synapse runs). `conn_r` here would cut the ratio gate
        # too, leaving an 8/8 adj != plain cone denominator (measured on
        # L1r0.001). Same universe as the synapse branch below.
        adj_denom_src = (denom_pairs.filter(
            pl.col("weight") >= int(t_syn_floor))
            if int(t_syn_floor) > 1 else denom_pairs)
        typed = typed.join(
            conn_r.select("bodyId_pre", "bodyId_post", "_ratio"),
            on=["bodyId_pre", "bodyId_post"], how="left")
        typed = typed.with_columns(
            pl.col("_ratio").fill_null(0.0).alias("weight"),
            pl.col("weight").alias("weight_syn")).drop("_ratio")
        # Et denominators: the post TYPE's full-membership incoming
        W_type_full = (W_post.join(tmap, left_on="bodyId_post",
                                   right_on="bid")
                       .group_by("t").agg(pl.col("_W").sum().alias("_tw")))
        pairs = (typed.group_by("type_pre", "type_post")
                 .agg(pl.col("weight_syn").sum().alias("w_sum"),
                      pl.len().alias("n"))
                 .join(W_type_full, left_on="type_post", right_on="t",
                       how="left")
                 .with_columns(
                     (pl.col("w_sum") / pl.col("_tw")).alias("w_max"))
                 .drop("_tw"))
        _phase(f"ratio universe ({typed.height:,} admitted edges, "
               f"t_r={t_ratio:g})")
    else:
        typed, pairs = _typed_universe_cached(
            conn_path, neuron_source, conn, tmap, asked, drop_untyped,
            exclude_intra, separate_hemi, hemi_filter, label_mapper,
            _phase, exclude_autapses=exclude_autapses,
            type_map_sha=type_map_sha)
        # Same rule on the synapse basis: all pres, synapse units,
        # conditioned at the lane's own asked floor. At asked <= 1 nothing is
        # cut, so the conditioned denominator IS the plain one and
        # adj == plain by construction (gate L1).
        adj_denom_src = (denom_conn.filter(pl.col("weight") >= int(asked))
                         if int(asked) > 1 else denom_conn)
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
        if ratio_basis:
            # re-derive the Et ratios over the intra-filtered set
            pairs = (pairs.drop("w_max")
                     .join(W_type_full, left_on="type_post", right_on="t",
                           how="left")
                     .with_columns(
                         (pl.col("w_sum") / pl.col("_tw")).alias("w_max"))
                     .drop("_tw"))
    if typed.height == 0:
        _refuse(f"no typed edges at asked threshold {asked}")
    src_ids = tmap.filter(pl.col("t").is_in(src_types))["bid"].to_list()
    tgt_ids = tmap.filter(pl.col("t").is_in(tgt_types))["bid"].to_list()
    # Grouped-source provenance (issue 2026-10-22): an effective source
    # label may fold many native subtypes. Report both the grouped labels
    # and the native types of the enrolled source bodies, so a grouped run
    # is auditable and a partially-folded mapping is visible.
    _src_members = [b for b in src_ids if (not src_keep or b in src_keep)]
    _native_by_bid = {}
    try:
        if "type" in frame.columns:
            _native_by_bid = {
                str(b): str(t) for b, t in zip(
                    frame["bodyId"].astype(str), frame["type"])}
    except Exception:
        _native_by_bid = {}
    from collections import Counter as _Counter
    _src_type_set = set(src_types)
    source_group_counts = dict(sorted(_Counter(
        str(type_map.get(b)) for b in _src_members
        if type_map.get(b) in _src_type_set).items()))
    source_group_counts_all = dict(sorted(_Counter(
        str(type_map.get(b)) for b in src_ids
        if type_map.get(b) in _src_type_set).items()))
    source_unenrolled_counts = {
        k: int(v) - int(source_group_counts.get(k, 0))
        for k, v in source_group_counts_all.items()
        if int(v) - int(source_group_counts.get(k, 0)) > 0}
    source_native_counts = dict(sorted(_Counter(
        _native_by_bid[b] for b in _src_members
        if _native_by_bid.get(b)).items()))

    # ---- per-type support tables (route_paths_type.csv mirrors the
    # *_allpaths_type.csv column semantics: weights/ratios/min_weight/
    # min_ratio/nt_types/coverage) --------------------------------------
    # These live BEFORE the typed-lane prune: the type-prune basis divides
    # pair mass by the same conditioned denominators the realized path
    # readout uses, so a prune and its exact filter cannot disagree.
    type_membership = dict(zip(
        *(tmap.group_by("t").agg(pl.len().alias("n")).sort("t")
          .select("t", "n")).to_dict().values()))
    incoming = (denom_conn.group_by("bodyId_post")
                .agg(pl.col("weight").sum().alias("w")))
    incoming = incoming.join(tmap, left_on="bodyId_post", right_on="bid")
    type_incoming = {}
    for t, w in zip(incoming["t"].to_list(), incoming["w"].to_list()):
        type_incoming[t] = type_incoming.get(t, 0) + float(w)
    # C1.1 (plan-corridor-alignment-and-viz-bases §2) + F-EX-003 D3b/D3c:
    # the threshold-CONDITIONED denominator is the ALL-PRES, SYNAPSE-UNIT
    # incoming of the edges this lane admits at its own asked rule — the
    # same universe `connection_type.csv`'s `connection_ratio_adj` divides
    # by. Two defects fixed here: (i) it used to sum the lane's `typed`
    # frame, which has already dropped untyped pres, so adj != plain even at
    # asked <= 1 (measured on MCNS: PPL101 38,866 conditioned vs 39,137
    # plain, against the code's own "== plain by construction" claim); (ii)
    # on a ratio-basis run `typed.weight` is the per-edge F9 RATIO, so a
    # synapse numerator over it yielded impossible input shares (measured
    # 0.912 and 204.54 on the L1r0.001 run).
    incoming_adj = (adj_denom_src.group_by("bodyId_post")
                    .agg(pl.col("weight").sum().alias("w")))
    incoming_adj = incoming_adj.join(tmap, left_on="bodyId_post",
                                    right_on="bid")
    type_incoming_adj = {}
    for t, w in zip(incoming_adj["t"].to_list(),
                    incoming_adj["w"].to_list()):
        type_incoming_adj[t] = type_incoming_adj.get(t, 0) + float(w)
    # Largest single SYNAPSE edge per admitted type pair, on every weight
    # basis (the ratio branch's `pairs.w_max` is the Et pair RATIO, so it
    # cannot serve a column named `weight_max`). Built AFTER the body prune
    # so the dict does not sit on the prune's peak (issue 2026-10-23).
    _mass = "weight_syn" if ratio_basis else "weight"
    type_nt = _type_nt_map(frame)

    # F-PF-009 D3-B (ruled follow-up, implemented): load the parent's
    # per-target distance maps and ENFORCE per-endpoint minimum hops at
    # realization -- a layered path (s_body -> ... -> t_body, h hops) is
    # kept iff h == dist[t_body][s_body]. With the maps the basis
    # regains true shortest semantics; without them (missing
    # node_distances) the lane falls back to the honestly-named union.
    if store_universe is not None and path_mode == "shortest" and run_dir:
        try:
            from shortest_discovery_store import load_target_distances
            store_distances = load_target_distances(
                Path(run_dir) / 'shortest_discovery_store', tgt_ids)
            store_distances = {
                str(k): {str(n): int(d) for n, d in v.items()}
                for k, v in (store_distances or {}).items()}
            if any(store_distances.values()):
                validation_basis = "shortest_dag_store_enforced"
            else:
                store_distances = None  # maps unavailable -> union
        except Exception:
            store_distances = None

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
    if ratio_basis:
        # the MASS space lookup: per bodyId edge's synapse weight (the
        # enumeration space on the ratio basis is the per-edge ratio)
        syn_w = {(u, v): float(w) for u, v, w in
                 zip(typed_routed["bodyId_pre"].to_list(),
                     typed_routed["bodyId_post"].to_list(),
                     typed_routed["weight_syn"].to_list())}
    else:
        syn_w = None
    # The body prune below allocates several full-size hop frames. The
    # typed lane without its additive second round has already consumed
    # every frame it needs from conn/typed, so release them (and the
    # denominator views) BEFORE the prune — the 2026-10-23 peak was the
    # union of the full universe and the prune's working set.
    if type_pruned and not full_scope_requested:
        del conn, typed
    if type_pruned:
        # The typed lane re-derives its pair table from the pruned loss
        # frame; the full typed pair table is not needed during the prune.
        del pairs
    del denom_conn, adj_denom_src, incoming, incoming_adj
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
            [typed_routed], src_ids, tgt_ids, bound,
            # The typed lane discards the prune stats; the widest-path
            # report costs a dict-of-dicts over the whole cone.
            report_bottleneck_bound=(not type_pruned),
            compute_strongest=(not type_pruned))
        loss = loss_tables[0]
        _phase(f"lossless-pruned ({loss.height:,} rows)")
        del loss_tables, _stats
    wmax_syn = {}
    try:
        _wm = (typed_routed.group_by("type_pre", "type_post")
               .agg(pl.col(_mass).max().alias("_mx")))
        wmax_syn = {(str(r["type_pre"]), str(r["type_post"])):
                    float(r["_mx"]) for r in _wm.to_dicts()}
    except Exception:  # noqa: BLE001 - falls back to the lane's own w_max
        wmax_syn = {}
    del typed_routed, rel, rel_keys
    from vispath_pkg.fast_graph_core import FastGraph

    pair_edges = _PairEdges(loss)
    # ---- Phase B: the TYPE-LEVEL COARSE PRUNE ------------------------
    # The pair table is the candidate filter; `loss` stays UNPRUNED, so
    # validation still sees every bodyId edge of a kept pair (a weak
    # aggregate pair can contain a strong bodyId edge).
    #
    # The pair table is the LOSSLESS DISCOVERY CONE, not the whole dataset
    # cache: `loss` holds exactly the bodyId edges lying on some
    # source->target path within the hop bound. Grouping it gives the
    # full-floor path-relevant type universe the plan asks for (P0-C);
    # pruning the raw 860k-pair FAFB cache instead kept 184k pairs and
    # enumerated walks through types no R7/R8->clock path can use
    # (measured 2026-10-22).
    prune_block = None
    if type_pruned:
        loss_pairs = (loss.group_by(["type_pre", "type_post"])
                      .agg(pl.col("weight").sum().alias("w_sum"),
                           pl.col("weight").max().alias("w_max"),
                           pl.len().alias("n")))
        pairs, prune_block = _prune_type_pairs(
            loss_pairs, type_incoming, type_incoming_adj,
            basis=str(type_prune_basis).lower(),
            min_ratio_adj=float(type_min_ratio_adj),
            min_synapse=int(type_min_synapse),
            min_wmax=int(type_min_wmax), default_floor=int(asked))
        _phase(f"type-pruned ({pairs.height:,} pairs, "
               f"{prune_block['mass_removed']:g} mass removed)")
        if pairs.height == 0 and not per_pair_scope:
            _refuse(f"type prune on {type_prune_basis!r} removed every "
                    f"type pair (basis threshold {type_min_ratio_adj:g}, "
                    f"min synapse {type_min_synapse}, min w_max "
                    f"{type_min_wmax})")
        # A scoped run with an empty cone is a legitimate "no pair is
        # reachable within its scope" result: it completes with the
        # per-pair no_path rows, never silently invents a bound.
    if per_pair_scope:
        # Pass A (discover) or the already-loaded external table. The
        # final per-pair bound is what the complete pass enumerates; the
        # universe above is a superset built at the working bound.
        if scope_mode_resolved == "discover":
            scope_seed, scope_bounds, scope_block = _discover_scope(
                pairs=pairs, pair_edges=pair_edges, src_types=src_types,
                tgt_types=tgt_types, src_ids=src_ids, tgt_ids=tgt_ids,
                src_keep=src_keep, tgt_keep=tgt_keep,
                body2type=body2type,
                discovery_cap=int(scope_working_bound),
                scope_pad=int(scope_pad), hard_cap=_scope_hard_cap,
                min_ratio_adj=float(type_min_ratio_adj),
                type_incoming_adj=type_incoming_adj,
                revisit_policy=str(type_revisit_policy).lower(),
                max_type_visits=int(type_max_type_visits),
                walk_cap=int(type_walk_cap), dataset=dataset or "",
                deadline=deadline,
                log=((lambda *a, **k: print(
                    *a, file=_sys.stderr, flush=True)) if _dbg else None))
            scope_table_sha = _write_scope_table(
                out, scope_seed, scope_pad=int(scope_pad),
                hard_cap=_scope_hard_cap, type_map_sha=type_map_sha,
                asked=asked, threshold=float(type_min_ratio_adj),
                dataset=dataset or "")
            scope_table_source = "discovered"
            scope_block["scope_table_sha"] = scope_table_sha
            scope_block["scope_table_source"] = scope_table_source
        _pos = [b for b in (scope_bounds or {}).values()
                if b is not None]
        bound = max(_pos) if _pos else 1
        scope_block.update({
            "scope_table_sha": scope_table_sha,
            "scope_table_source": scope_table_source,
            "scope_working_bound": int(scope_working_bound),
            "scope_final_max_bound": int(bound),
            "scope_dataset": dataset,
            "scope_requested_floor": asked,
            "scope_threshold": float(type_min_ratio_adj),
        })
        _phase(f"scope ({scope_mode_resolved}, max bound {bound})")
    _loss_height = None
    if type_pruned:
        # The prune has consumed the loss frame; the typed lane reads the
        # canonical sorted view through _PairEdges. Release the duplicate
        # unsorted frame before enumeration (memory bound, 2026-10-23).
        _loss_height = int(loss.height)
        pair_edges._ensure_sorted()
        del loss
    shared_W = (None if type_pruned
                else widest_to_targets_fast(pairs, set(tgt_types), bound))
    limit = int(per_pair_length_budget)

    walk_rows, body_rows = [], []
    provenance_classes, budgets = {}, {}
    scope_pairs = {}      # (a, b) -> {"mass": x, "paths": n, "walks": k}
    time_capped = False

    # ---- D2-B helpers (certified actual-strength mode, opt-in) -------
    def _certified_realize(s, walk, cap):
        """realize_fn for the certified mode: enumerate ONE accepted
        walk's body-simple realizations (per-endpoint
        distance-enforced when the store maps exist) into the class's
        pending pool and update the running share-th strongest realized
        bottleneck -- the LIVE certification witness the validate loop's
        share-skip path checks each skipped candidate's UB against.
        Returns (count, truncated) like the counting realize_fn."""
        nonlocal time_capped
        h = len(walk) - 1
        t = walk[-1]
        t_bodies = {b for b in tgt_ids if body2type.get(b) == t
                    and (not _tgt_restrict or b in tgt_keep)}
        cb = classes[((s, t), h)]
        if deadline is not None and time.monotonic() > deadline:
            time_capped = True
            return 0, True
        pair_rows = [pair_edges.tuples(walk[i], walk[i + 1])
                     for i in range(h)]
        alive = [set() for _ in range(h + 1)]
        alive[h] = set(t_bodies)
        for i in range(h - 1, -1, -1):
            nxt = alive[i + 1]
            alive[i] = {u for u, v, _w in pair_rows[i] if v in nxt}
        kept_rows = []
        for i in range(h):
            for u, v, w in pair_rows[i]:
                if v not in alive[i + 1]:
                    continue
                if i == 0 and u not in s_bodies:
                    continue
                kept_rows.append((i, u, v, w))
        raw_w = {}
        layered_edges = []
        for i, u, v, w in kept_rows:
            raw_w[(u, v)] = w
            layered_edges.append(
                (f"{u}\x1f{i}", f"{v}\x1f{i + 1}", w))
        g = FastGraph()
        g.build_from_edges(layered_edges)
        adj_w = g.adj
        st: dict = {}
        multi_visit = len(set(walk)) < len(walk)
        src_l = [f"{b}\x1f0" for b in s_bodies]
        tgt_l = {f"{b}\x1f{h}" for b in t_bodies}
        kept = []
        hop_edges = [dict() for _ in range(h)]
        hop_ratio = [dict() for _ in range(h)]
        node_seen = [set() for _ in range(h + 1)]
        bns = []
        for p in g.find_paths_strongest_first(
                src_l, tgt_l, h, budget=cap, stats=st,
                raw_body_simple=multi_visit):
            raw = [n.split("\x1f")[0] for n in p]
            if store_distances is not None and store_distances.get(
                    raw[-1], {}).get(raw[0]) != h:
                continue
            bn = min(adj_w[p[i]][p[i + 1]] for i in range(h))
            kept.append(raw)
            bns.append(bn)
            for i in range(h):
                hop_edges[i][(raw[i], raw[i + 1])] = float(
                    syn_w[(raw[i], raw[i + 1])]
                    if ratio_basis else adj_w[p[i]][p[i + 1]])
                hop_ratio[i][(raw[i], raw[i + 1])] = float(
                    adj_w[p[i]][p[i + 1]])
                node_seen[i].add(raw[i])
            node_seen[h].add(raw[h])
        tau = st.get("tau")
        if tau is None and kept:
            tau = float(min(bns))
        cb.realized_bns.extend(bns)
        rank = len(cb.pending_records) + 1
        # the walk's type-level candidate bound: the widest-path product
        # of its hops' max edges (the drain's own UB semantics)
        walk_ub = min(
            (max((w for _u, _v, w in pair_rows[i]), default=0.0)
             for i in range(h)), default=0.0)
        if kept:
            cb.pending_records.append({
                "walk": walk, "rank": rank, "ub": float(walk_ub),
                "tau": tau, "kept": kept, "hop_edges": hop_edges,
                "hop_ratio": hop_ratio,
                "node_seen": node_seen, "raw_w": raw_w,
                "bns": bns})
        else:
            cb.off_len += 1
        if len(cb.realized_bns) >= cb.limit:
            cb.weakest_emitted = sorted(cb.realized_bns,
                                        reverse=True)[cb.limit - 1]
        return len(kept), False

    def _certified_allocate(s, t, h, cb):
        """Allocate the class's share across ALL realized paths by actual
        bottleneck (the true top-share), emit the witness rows, and set
        the certification pair (weakest_emitted vs floor_next)."""
        pool = []
        for ri, rec in enumerate(cb.pending_records):
            for pi, bn in enumerate(rec["bns"]):
                pool.append((bn, ri, pi))
        pool.sort(key=lambda x: -x[0])
        alloc = pool[:cb.limit]
        by_rec = {}
        for bn, ri, pi in alloc:
            by_rec.setdefault(ri, []).append(pi)
        cb.weakest_emitted = (alloc[-1][0] if alloc else None)
        cb.emitted = len(alloc)
        cb.off_plan = sum(1 for ri in range(len(cb.pending_records))
                          if ri not in by_rec)
        if (cb.weakest_emitted is not None and cb.floor_next is not None
                and cb.floor_next <= cb.weakest_emitted):
            cb.certified = True
        tau_share = cb.weakest_emitted
        for ri, rec in enumerate(cb.pending_records):
            pis = sorted(by_rec.get(ri, []))
            if not pis:
                continue
            walk = rec["walk"]
            kept = [rec["kept"][pi] for pi in pis]
            hop_edges = [dict() for _ in range(h)]
            hop_ratio = [dict() for _ in range(h)]
            node_seen = [set() for _ in range(h + 1)]
            for raw in kept:
                for i in range(h):
                    hop_edges[i][(raw[i], raw[i + 1])] = float(
                        syn_w[(raw[i], raw[i + 1])]
                        if ratio_basis else
                        rec["raw_w"][(raw[i], raw[i + 1])])
                    if ratio_basis:
                        hop_ratio[i][(raw[i], raw[i + 1])] = float(
                            rec["raw_w"][(raw[i], raw[i + 1])])
                    node_seen[i].add(raw[i])
                node_seen[h].add(raw[h])
            hop_mass = [sum(e.values()) for e in hop_edges]
            hop_ratio_mass = [sum(e.values()) for e in hop_ratio]
            ratios, ratios_adj = [], []
            for i in range(h):
                if ratio_basis:
                    # CORRIDOR-ON-RATIO readout pin: the realized per-hop
                    # ratio = the MEAN realizing-edge F9 ratio (the run's
                    # type-table convention); the >=t_r conditioning is
                    # vacuous (every admitted edge clears it)
                    _m = (hop_ratio_mass[i] / len(hop_ratio[i])
                          if hop_ratio[i] else None)
                    ratios.append(_m if _m is not None else 0.0)
                    ratios_adj.append(_m)
                    continue
                denom = type_incoming.get(walk[i + 1], 0.0)
                ratios.append(hop_mass[i] / denom if denom > 0 else 0.0)
                denom_adj = type_incoming_adj.get(walk[i + 1], 0.0)
                ratios_adj.append(hop_mass[i] / denom_adj
                                  if denom_adj > 0 else None)
            finite_adj = [r for r in ratios_adj if r is not None]
            cov = ["{}/{}".format(
                len(node_seen[i]),
                type_membership.get(walk[i], len(node_seen[i])))
                for i in range(h + 1)]
            nts = [type_nt.get(walk[i], "unknown") for i in range(h)]
            walk_rows.append({
                "path": "->".join(walk),
                "weights": "[" + ", ".join(
                    _fmt_mass(m) for m in hop_mass) + "]",
                "ratios": "[" + ", ".join(
                    repr(r) for r in ratios) + "]",
                "ratios_adj": "[" + ", ".join(
                    'None' if r is None else repr(r)
                    for r in ratios_adj) + "]" if ratios_adj else "",
                "ratio_adj_complete": (len(finite_adj) == len(ratios_adj)
                                       and bool(ratios_adj)),
                "type_pruned": False,
                "pair_weights": None,
                "min_weight": (int(min(hop_mass)) if hop_mass else None),
                "min_ratio": min(ratios) if ratios else None,
                "min_ratio_adj": (min(finite_adj) if finite_adj else None),
                "length": h,
                "nt_types": "[" + ", ".join(
                    f'"{x}"' for x in nts) + "]",
                "coverage": "[" + ", ".join(cov) + "]",
                "applied_tau": tau_share,
                "route_h": h, "route_ub": float(rec["ub"]),
                "route_floor_h": (None if cb.floor is None
                                  else float(cb.floor)),
                "plan_rank": rec["rank"],
                "realized_paths": len(kept)})
            for i in range(h):
                key = (walk[i], walk[i + 1])
                r2 = scope_pairs.setdefault(
                    key, {"edges": {}, "paths": 0, "walks": 0})
                r2["edges"].update(hop_edges[i])
                r2["paths"] += len(kept)
                r2["walks"] += 1
            for raw in kept:
                body_rows.append({
                    "path": "->".join(str(b) for b in raw),
                    "weights": ",".join(_fmt_mass(
                        rec["raw_w"][(raw[i], raw[i + 1])])
                        for i in range(h)),
                    "length": h, "route_h": h,
                    "applied_tau": tau_share,
                    "plan_rank": rec["rank"]})

    type_block = None
    if type_pruned:
        # The typed lane builds its universe from the LOSSLESS DISCOVERY
        # CONE, never the materialized path store or the refill's emitted
        # chains. The refill's status is still recorded, so a reader knows
        # whether the parallel artifact exists (P0-C/P0-E).
        _refill_status = ('not_available (no refill artifact in this run '
                          'folder)' if run_dir else
                          'not_available (direct mode)')
        if run_dir:
            for _cand in (
                    Path(run_dir) / 'data_details' / 'type_level_refill'
                    / 'refill_provenance.json',
                    Path(run_dir) / 'type_level_refill'
                    / 'refill_provenance.json'):
                if not _cand.exists():
                    continue
                try:
                    _rec = json.loads(_cand.read_text(encoding='utf-8'))
                    _full = _rec.get('full') or {}
                    _refill_status = str(
                        _full.get('refill_status')
                        or _rec.get('status') or 'unknown')
                except Exception:
                    _refill_status = 'unreadable'
                break
        type_block = _run_type_pruned(
            walk_rows=walk_rows, scope_pairs=scope_pairs, pairs=pairs,
            pair_edges=pair_edges,
            src_types=src_types, tgt_types=tgt_types,
            src_ids=src_ids, tgt_ids=tgt_ids,
            src_keep=src_keep, tgt_keep=tgt_keep, body2type=body2type,
            bound=bound, type_membership=type_membership,
            type_incoming=type_incoming,
            type_incoming_adj=type_incoming_adj, type_nt=type_nt,
            min_ratio_adj=float(type_min_ratio_adj),
            walk_cap=int(type_walk_cap),
            revisit_policy=str(type_revisit_policy).lower(),
            max_type_visits=int(type_max_type_visits),
            deadline=deadline, prune_block=prune_block,
            out_dir=out,
            pair_bounds=(scope_bounds if per_pair_scope else None),
            scope_seed=(scope_seed if per_pair_scope else None),
            scope_context=({
                "scope_mode": "per_pair_shortest_plus_2",
                "scope_lt": ("external_table"
                             if scope_mode_resolved == "external"
                             else "max_local_validated"),
                "scope_pad": int(scope_pad),
                "scope_hard_cap": _scope_hard_cap,
                "scope_discovery_cap": (
                    None if scope_mode_resolved == "external"
                    else int(scope_discovery_cap)),
                "scope_bound_source": (
                    "external_table"
                    if scope_mode_resolved == "external"
                    else "Lt_plus_pad"),
                "scope_lt_evidence": scope_block.get("scope_lt_evidence"),
                "scope_table_sha": scope_table_sha,
                "scope_table_source": scope_table_source,
                "scope_working_bound": int(scope_working_bound),
                "scope_final_max_bound": int(bound),
                "scope_dataset": dataset,
                "scope_report_table": "route_scope_pairs.csv",
                "requested_floor": int(asked),
                "type_map_sha": type_map_sha,
                "dataset": dataset,
            } if per_pair_scope else None),
            log=((lambda *a, **k: print(*a, file=_sys.stderr, flush=True))
                 if _dbg else None))
        type_block['refill_used'] = False
        type_block['refill_status'] = _refill_status
        type_block['universe_source'] = (
            'lossless_discovery_cone (run connection source; not the '
            'materialized path store and not the refill emitted chains)')
        type_block['full_floor_rows'] = int(_loss_height or 0)
        type_block['requested_floor'] = int(asked)
        # Issue 2026-10-22 §2.4: name the two floors for what they are.
        # `run_applied_floor` / `typed_lane_floor` stay as compatibility
        # aliases; `complete_tau` is the Complete lane's applied cut and
        # `type_pruned_floor` is this lane's floor.
        type_block['complete_tau'] = (
            int(_run_applied_floor)
            if _run_applied_floor is not None else None)
        type_block['type_pruned_floor'] = int(asked)
        type_block['run_applied_floor'] = type_block['complete_tau']
        type_block['typed_lane_floor'] = type_block['type_pruned_floor']
        type_block['floor_raised_in_run'] = (
            _run_applied_floor is not None
            and int(_run_applied_floor) > int(asked))
        # P1-K: the filter/label/enrollment parity object, recorded beside
        # the walk set so a reader can reproduce the universe.
        type_block['scope'] = str(type_prune_scope).lower()
        type_block['type_prune_scope'] = str(type_prune_scope).lower()
        type_block['drop_untyped'] = bool(drop_untyped)
        if not drop_untyped:
            try:
                from utils.label_utils import is_untyped_type_label
                _nodes = set(pairs["type_pre"].to_list()) | set(
                    pairs["type_post"].to_list())
                type_block['untyped_nodes_kept'] = sum(
                    1 for _t in _nodes if is_untyped_type_label(_t))
            except Exception:
                type_block['untyped_nodes_kept'] = None
        type_block['exclude_intra_type_connections'] = bool(exclude_intra)
        type_block['exclude_autapses'] = bool(exclude_autapses)
        type_block['separate_hemispheres'] = bool(separate_hemi)
        type_block['hemisphere_filter'] = hemi_filter
        type_block['weight_basis'] = basis
        type_block['dataset'] = dataset
        type_block['type_map_sha'] = type_map_sha
        type_block['type_map_size'] = len(type_map)
        type_block['label_mapper_sha'] = label_mapper_sha
        type_block['label_mapper_source'] = label_mapper_source
        type_block['sources'] = src_types
        type_block['targets'] = tgt_types
        type_block['source_bodies_effective'] = len(_src_members)
        type_block['source_group_counts'] = source_group_counts
        type_block['source_group_counts_all'] = source_group_counts_all
        type_block['source_unenrolled_counts'] = source_unenrolled_counts
        type_block['source_native_counts'] = source_native_counts
        type_block['source_group_mapping_sha'] = label_mapper_sha
        type_block['enrollment'] = {
            'sources': src_enrollment, 'targets': tgt_enrollment,
            'source_bodies': len(src_keep) or None,
            'target_bodies': len(tgt_keep) or None,
            'source_bodies_effective': len(_src_members),
            'source_group_counts': source_group_counts,
            'source_group_counts_all': source_group_counts_all,
            'source_unenrolled_counts': source_unenrolled_counts,
            'source_native_counts': source_native_counts,
            'source_group_mapping_sha': label_mapper_sha}
        time_capped = time_capped or type_block.get("time_capped", False)
        if per_pair_scope:
            # The provenance's scope block must carry the FINAL per-pair
            # statuses (the discovery seed only knew hard caps), never a
            # pre-enumeration snapshot.
            scope_block.update({
                k: type_block.get(k) for k in (
                    "scope_complete", "scope_complete_by_pair",
                    "scope_capped_pairs", "scope_no_path_pairs",
                    "scope_validation_cap_pairs",
                    "scope_exhausted_pairs", "scope_pairs_total",
                    "scope_pairs_exhausted", "scope_report_table",
                    "scope_input_table", "scope_bound_source",
                    "scope_replaces_max_interlayer")})
        _phase(f"type-pruned complete ({type_block['accepted_paths']} "
               f"paths, {type_block['candidate_walks']} candidates)")
        # The layered lane no longer needs its per-pair edge cache. Free it
        # before the additive full-scope refill so the peak is not the
        # union of both passes (the loss frame was released after the
        # prune, above).
        del pair_edges

    for s in src_types:
        if type_pruned:
            break
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
                # ROUTE SHARE (§21's M*, now spent on accepted routes) and
                # the candidate-probe bound around it. The pull cap allows
                # one share's worth of phantoms: a phantom-heavy class
                # still terminates, but a run of phantoms can no longer eat
                # the share the routes were asked for (the 2026-10-16
                # finding: the old single cap made phantoms and routes pay
                # from one purse). certify_actual counts its share in
                # realized paths, so it keeps count_routes=False.
                _share = max(1, round(
                    float(top_m_base) * (float(length_decay) ** (h - 1))))
                classes[((s, t), h)] = ClassBudget(
                    limit=limit,
                    route_share=_share,
                    pull_cap=_share * 2,
                    count_routes=not certify_actual,
                    feasible=(d_lb is not None and h >= d_lb),
                    certify_actual=certify_actual)
        s_bodies = [b for b in src_ids if body2type.get(b) == s
                    and (not _src_restrict or b in src_keep)]
        realize_fn = lambda walk, cap, _s=s: _realize_on_loss(  # noqa: E731
            pair_edges, walk, s_bodies,
            {b for b in tgt_ids if body2type.get(b) == walk[-1]}, cap)
        if certify_actual:
            # D2-B CERTIFIED ACTUAL-STRENGTH MODE (opt-in, ruled): the
            # realize_fn enumerates each accepted walk's FULL body-simple
            # realization set inline and keeps the class's running
            # share-th strongest realized bottleneck (weakest_emitted)
            # live, so the validate loop's share-skip path certifies
            # against ACTUAL route strength: candidates past the share
            # are accepted while their UB can still beat the witness and
            # skipped (floor_next recorded) once it cannot -- the
            # exported top-share is then the true strongest actual
            # routes. The share allocation is deferred to the post-drain
            # pass below.
            prov = validate_and_fill(
                drain, classes,
                lambda walk, cap, _s=s: _certified_realize(_s, walk, cap),
                max_pulls=drain.max_pops, deadline=deadline)
            time_capped = time_capped or prov.get("time_capped", False)
            provenance_classes.update(prov["classes"])
            budgets.update(classes)
            for t in tgt_types:
                for h in range(1, bound + 1):
                    cb = classes[((s, t), h)]
                    _certified_allocate(s, t, h, cb)
                    if drain.exhausted:
                        # every candidate was consumed: nothing was
                        # skipped beyond the last witness, so the emitted
                        # set is the complete realization set --
                        # certified trivially
                        cb.certified = True
            _phase(f"certified source {s}")
            continue
        prov = validate_and_fill(
            drain, classes, realize_fn,
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
                    if deadline is not None and time.monotonic() > deadline:
                        time_capped = True
                        break
                    # EVERY accepted route gets at least one witness
                    # (2026-10-16): the class envelope `cb.limit` governs the
                    # tie MASS an enumeration may carry, not whether a route
                    # is published at all. The old `if remaining <= 0: break`
                    # silently dropped every accepted walk after the envelope
                    # was spent — the materialization half of the fat-walk
                    # starvation. A route's mass columns never depended on
                    # this sample anyway (chain mass comes from the layered
                    # filter below, by construction unbudgeted).
                    share = max(1, remaining)
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
                    budget = share + 1
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
                        hop_ratio = [dict() for _ in range(h)]
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
                            if store_distances is not None:
                                # D3-B: per-endpoint minimum-hop
                                # enforcement -- the realization must be
                                # SHORTEST for its own (src, tgt) body
                                # pair, not merely present in the
                                # flattened store union
                                _raw = [n.split("\x1f")[0] for n in p]
                                if store_distances.get(
                                        _raw[-1], {}).get(_raw[0]) != h:
                                    paths.pop()
                                    continue
                            if len(paths) >= share:
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
                                for cand in gen:
                                    _rw = [n.split("\x1f")[0]
                                           for n in cand]
                                    if store_distances is not None                                             and store_distances.get(
                                                _rw[-1], {}).get(
                                                _rw[0]) != h:
                                        continue  # filtered witness
                                    nxt = cand
                                    ties_next_bn = min(
                                        adj_w[nxt[i]][nxt[i + 1]]
                                        for i in range(h))
                                    _nxt_repeats = (
                                        len(set(_rw)) != len(_rw))
                                    break
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
                                # the ratio basis keeps BOTH units: the
                                # mass columns take the SYNAPSE weight,
                                # the realized-ratio readout the edge's
                                # own F9 ratio (the engine space)
                                hop_edges[i][(raw[i], raw[i + 1])] = float(
                                    syn_w[(raw[i], raw[i + 1])]
                                    if ratio_basis
                                    else adj_w[p[i]][p[i + 1]])
                                hop_ratio[i][(raw[i], raw[i + 1])] = float(
                                    adj_w[p[i]][p[i + 1]])
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
                    # ---- CHAIN-LEVEL per-hop mass + coverage (F-EX-001,
                    # Decisions 3) ------------------------------------
                    # The witness union above is a SAMPLE: every realizing
                    # bodyId path is a representative route, but the sum over
                    # a budgeted subset of them (500 of 1336 synapses on the
                    # MCNS aMe12->KCg-d->PPL101 walk, with the class at
                    # stop='budget') is not a mass anyone can read as a
                    # connection weight. The chain's own per-position
                    # restricted mass comes from the REFILL's forward +
                    # backward layered filter — the helper that already
                    # reproduces the complete lane's [1336, 476] element-wise
                    # when the tiers match — run over THIS lane's admitted
                    # graph in SYNAPSE units (`weight_syn` on a ratio run, the
                    # same rule that anchors the refill's path tables). No
                    # enumeration, so the class budget cannot bias it.
                    chain_edges, chain_nodes = hop_edges, node_seen
                    chain_revisit = False
                    try:
                        from type_level_refill import (
                            _layered_chain_structures)
                        _pe = {}
                        for i in range(h):
                            _d = _pe.setdefault((walk[i], walk[i + 1]), {})
                            for u, v, w in pair_rows[i]:
                                _d.setdefault(u, []).append(
                                    (v, float(syn_w[(u, v)])
                                     if ratio_basis else float(w)))
                        _st = _layered_chain_structures(
                            tuple(walk), _pe, s_bodies, t_bodies,
                            body2type, set())
                        if _st is not None:
                            chain_edges = _st['edges_at']
                            chain_nodes = _st['nodes_at']
                            chain_revisit = bool(_st['revisit'])
                    except Exception:  # pragma: no cover - additive lane
                        # The chain readout is additive: a lane that cannot
                        # resolve the shared filter keeps its witness union
                        # rather than publishing nothing.
                        pass
                    hop_mass = [sum(e.values()) for e in chain_edges]
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
                            len(chain_nodes[i]),
                            type_membership.get(walk[i],
                                                len(chain_nodes[i])))
                        for i in range(h + 1)]
                    # The SAMPLED witness readout survives, renamed so it can
                    # never be mistaken for a connection weight, and kept in
                    # the file for the lane's own sampling audits (gate L6a:
                    # it must never exceed the chain mass beside it).
                    samp_mass = [sum(e.values()) for e in hop_edges]
                    samp_cov = [
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
                                'None' if r is None else repr(r)
                                for r in ratios_adj) + "]"
                            if ratios_adj else ""),
                        "ratio_adj_complete": (
                            len(finite_adj) == len(ratios_adj)
                            and bool(ratios_adj)),
                        "type_pruned": False,
                        "pair_weights": None,
                        "min_weight": (int(min(hop_mass))
                                       if hop_mass else None),
                        "min_ratio": min(ratios) if ratios else None,
                        "min_ratio_adj": (min(finite_adj)
                                          if finite_adj else None),
                        "length": h,
                        "nt_types": "[" + ", ".join(
                            f'"{x}"' for x in nts) + "]",
                        "coverage": "[" + ", ".join(cov) + "]",
                        "sampled_support_mass": "[" + ", ".join(
                            _fmt_mass(m) for m in samp_mass) + "]",
                        "sampled_ratios": "[" + ", ".join(
                            repr(m / type_incoming.get(walk[i + 1], 0.0))
                            if type_incoming.get(walk[i + 1], 0.0) > 0
                            else 'None'
                            for i, m in enumerate(samp_mass)) + "]",
                        "sampled_coverage": "[" + ", ".join(
                            samp_cov) + "]",
                        "chain_revisit": chain_revisit,
                        "applied_tau": tau,
                        "route_h": h, "route_ub": float(ub),
                        "route_floor_h": (None if cb.floor is None
                                          else float(cb.floor)),
                        "plan_rank": rank,
                        "realized_paths": len(kept)})
                    for i in range(h) if kept else ():
                        key = (walk[i], walk[i + 1])
                        rec = scope_pairs.setdefault(
                            key, {"edges": {}, "edge_ratios": {},
                                  "paths": 0, "walks": 0})
                        # merge DISTINCT edges across walks sharing the hop
                        rec["edges"].update(hop_edges[i])
                        if ratio_basis:
                            rec["edge_ratios"].update(hop_ratio[i])
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
            entry["certified"] = (cb.certified if certify_actual
                                  else None)
            entry["weakest_emitted"] = (cb.weakest_emitted
                                        if certify_actual else None)
    out.mkdir(parents=True, exist_ok=True)
    if type_block is not None:
        # The layered typed lane writes no bodyId witness rows (issue
        # 2026-10-23). Remove a stale CSV left by an older in-place run so
        # the folder cannot advertise a table this lane no longer produces.
        (out / "route_paths_bodyId.csv").unlink(missing_ok=True)
        # P2-N: the typed lane's own completeness field and manifest live
        # beside the tables, so no reader has to infer them from `paths_complete`
        # (the Complete lane's tau field).
        (out / "type_prune_manifest.json").write_text(
            json.dumps(type_block, indent=2, default=str),
            encoding="utf-8")
        # The KEPT type graph itself (the coarse candidate universe every
        # exported walk was drawn from), with both pair-level ratios.
        pairs.write_csv(out / "route_type_pairs_pruned.csv")
    # one row per VALIDATED WALK in *_allpaths_type.csv column order +
    # the additive lane columns (per-path detail lives in the bodyId csv)
    walk_rows.sort(key=lambda r: (r["path"], r["route_h"],
                                  r["plan_rank"]))
    if type_pruned:
        # Layered contract (issue 2026-10-23): chain-level columns only.
        # No sampled witness columns, no realization envelope/count.
        typed_schema = {
            "path": pl.Utf8, "weights": pl.Utf8, "ratios": pl.Utf8,
            "ratios_adj": pl.Utf8,
            "min_weight": pl.Int64, "min_ratio": pl.Float64,
            "min_ratio_adj": pl.Float64,
            "length": pl.Int64, "nt_types": pl.Utf8,
            "coverage": pl.Utf8, "applied_tau": pl.Float64,
            "route_h": pl.Int64, "route_ub": pl.Float64,
            "route_floor_h": pl.Float64, "plan_rank": pl.Int64,
            "chain_revisit": pl.Boolean,
            "pair_weights": pl.Utf8,
            "ratio_adj_complete": pl.Boolean,
            "type_pruned": pl.Boolean}
        pl.DataFrame(walk_rows, schema=typed_schema).write_csv(
            out / "route_paths_type.csv")
    else:
        for _row in walk_rows:
            _row.setdefault("gate_min_ratio_adj",
                            _row.get("min_ratio_adj"))
            _row.setdefault("exact_gate_basis", "layered_chain")
            _row.setdefault("realization_sample_truncated", False)
        type_schema = {
            "path": pl.Utf8, "weights": pl.Utf8, "ratios": pl.Utf8,
            "ratios_adj": pl.Utf8,
            "min_weight": pl.Int64, "min_ratio": pl.Float64,
            "min_ratio_adj": pl.Float64,
            "gate_min_ratio_adj": pl.Float64,
            "exact_gate_basis": pl.Utf8,
            "realization_sample_truncated": pl.Boolean,
            "length": pl.Int64, "nt_types": pl.Utf8,
            "coverage": pl.Utf8, "applied_tau": pl.Float64,
            "route_h": pl.Int64, "route_ub": pl.Float64,
            "route_floor_h": pl.Float64, "plan_rank": pl.Int64,
            "realized_paths": pl.Int64,
            # Decisions 3: the sampled witness readout, demoted to
            # explicitly-named columns beside the chain-level ones.
            "sampled_support_mass": pl.Utf8,
            "sampled_ratios": pl.Utf8,
            "sampled_coverage": pl.Utf8,
            "chain_revisit": pl.Boolean,
            "pair_weights": pl.Utf8,
            "ratio_adj_complete": pl.Boolean,
            "type_pruned": pl.Boolean}
        pl.DataFrame(walk_rows, schema=type_schema).write_csv(
            out / "route_paths_type.csv")
    if write_bodyid and not type_pruned:
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
            realized = round(
                float(rec.get("mass", sum(rec.get("edges", {}).values()))), 2)
            # F-EX-003 D3b: ONE universe for both flavors, all pres, synapse
            # units — the same denominators `connection_type.csv` uses for
            # `connection_ratio` / `connection_ratio_adj`.
            denom = type_incoming.get(b, 0.0)
            denom_adj = type_incoming_adj.get(b, 0.0)
            _sel_ratios = rec.get("edge_ratios") or {}
            _mean_ratio = (round(sum(_sel_ratios.values())
                                 / len(_sel_ratios), 6)
                           if _sel_ratios else None)
            conn_rows.append({
                # _TYPE_MIRROR_COLUMNS family (type_level_refill.py:1758)
                "type_pre": a, "type_post": b,
                "weight": wsum,
                # F-EX-006: `nt_type` is the PRESYNAPTIC type's
                # transmitter, exactly as `connection_type.csv` publishes it
                # in this same mirror family. It used to read the POST type,
                # so `aMe26→SMP091` said acetylcholine here and gaba there,
                # and every hop into PPL101/PPL103 claimed dopamine.
                "nt_type": type_nt.get(a, ""),
                # F-EX-011 (Decisions 4): the ratio columns are MASS SHARES on
                # every weight basis — a quotient a reader can reproduce from
                # the row's own numbers, like every other connection table.
                # The mean realizing-edge F9 ratio is still published, but
                # under its own name, never in the `connection_ratio` slot.
                "connection_ratio": (
                    round(realized / denom, 6) if denom > 0 else None),
                "conn_layer": "route_scope",
                "connection_ratio_adj": (
                    round(realized / denom_adj, 6)
                    if denom_adj > 0 else None),
                # corridor tail — `weight_max` is the largest single SYNAPSE
                # edge of the pair on every basis (it carried the Et pair
                # RATIO on ratio runs, a unit change under a name that says
                # "max weight"). The Et ratio keeps its own named column.
                "weight_max": wmax_syn.get((a, b), wmax if not ratio_basis
                                           else None),
                "et_pair_ratio": (round(wmax, 6) if ratio_basis
                                  and wmax is not None else None),
                "mean_edge_f9_ratio": _mean_ratio,
                "pair_count": n,
                # F-PF-008 (audit 2026-10-13): `weight` is the WHOLE
                # admitted type-pair mass while connection_ratio divides
                # the SELECTED-realization mass — two support sets behind
                # one row. The explicit aliases name each quantity so no
                # reader has to guess which numerator a number uses.
                # Renamed (plan-corridor-refill-and-mode-tabs F4/Q3) from
                # `universe_pair_mass`: it is the type-pair's WEIGHT
                # UNIVERSE the chain hops are drawn from, and the report
                # column that prints it per hop is `Universe type weights`.
                "universe_type_weights": wsum,
                "selected_support_mass": realized,
                "realized_mass": realized,
                # Fast lane: bodyId paths. Layered typed lane: 0 (no
                # witness rows); its accepted walks live in used_by_walks.
                "realizing_paths": int(rec.get("bodyid_paths", rec["paths"])),
                "used_by_walks": rec["walks"]})
        pl.DataFrame(conn_rows).write_csv(
            out / "route_connection_type.csv")
    # The lane's CONDITIONED denominators (>= asked), published beside the
    # pair table so the report's corridor Adj column stays CHAIN-derived
    # (realized hop mass / this file) at the lane's own tier — the same
    # contract the refill's type_incoming_asked.csv has. Header always,
    # even when the map is empty.
    import pandas as _pd_inc
    _pd_inc.DataFrame(
        [{"type_post": k, "incoming_threshold": float(v)}
         for k, v in sorted(type_incoming_adj.items())],
        columns=["type_post", "incoming_threshold"]).to_csv(
        out / "route_type_incoming_asked.csv", index=False)
    # Q2 (plan-corridor-refill-and-mode-tabs): the lane's SECOND ROUND —
    # the same validated chains re-valued at synapse >= 1 through the
    # refill's own layered filter, so a Fast run carries the same scope
    # pair the Complete mode does. Additive: a failure is DISCLOSED in the
    # provenance (a silent skip here would leave a reader believing the
    # lane had nothing below its tier).
    full_block = None
    # The second round is additive and is NOT part of the layered typed
    # contract (it retains per-chain structures and was the residual peak).
    # The Fast lane keeps its existing default; the typed lane opts in with
    # --full-scope.
    if not full_scope_requested:
        # Remove stale additive artifacts from an older in-place typed run.
        for _stale in (
                "route_full_paths_type.csv",
                "route_full_carrying_paths_type.csv",
                "route_full_type_pairs.csv", "route_type_incoming.csv",
                "route_full_type_network.html",
                "route_full_type_heatmap.html"):
            (out / _stale).unlink(missing_ok=True)
        full_block = {
            "status": "skipped_not_requested", "scope": "synapse >= 1",
            "asked": asked,
            "note": ("the layered typed lane does not run the additive "
                     "synapse >= 1 second round by default; pass "
                     "--full-scope to re-value its chains")}
    else:
        refill_conn = conn
        refill_cut_keys = None
        if type_pruned:
            # The second round only needs edges on the lane's own hop pairs.
            # Release the typed frame BEFORE projecting the connection table
            # (the cut-key set is all the refill needs from it), then
            # release the full table.
            if typed is not None:
                refill_cut_keys = set(zip(typed["bodyId_pre"].to_list(),
                                          typed["bodyId_post"].to_list()))
                typed = None
            _seqs = {tuple(str(p).strip()
                           for p in str(row["path"]).split("->"))
                     for row in walk_rows}
            _hops = sorted({(seq[i], seq[i + 1]) for seq in _seqs
                            for i in range(len(seq) - 1)})
            if _hops:
                _want = pl.DataFrame({
                    "type_pre": [a for a, _b in _hops],
                    "type_post": [b for _a, b in _hops]})
                refill_conn = (
                    conn.rename({"bodyId_pre": "u", "bodyId_post": "v"})
                    .join(tmap.rename({"bid": "u", "t": "tpre"}), on="u",
                          how="inner")
                    .join(tmap.rename({"bid": "v", "t": "tpost"}), on="v",
                          how="inner")
                    .rename({"tpre": "type_pre", "tpost": "type_post"})
                    .join(_want, on=["type_pre", "type_post"], how="inner")
                    .select([pl.col("u").alias("bodyId_pre"),
                             pl.col("v").alias("bodyId_post"), "weight"])
                    .unique(subset=["bodyId_pre", "bodyId_post"]))
            else:
                refill_conn = None
            del conn
        try:
            full_block = _full_scope_refill(
                out, conn=refill_conn, tmap=tmap, body2type=body2type,
                src_ids=src_ids, tgt_ids=tgt_ids, walk_rows=walk_rows,
                typed=typed, cut_keys=refill_cut_keys,
                type_membership=type_membership,
                type_incoming=type_incoming, asked=asked,
                # the lane's own logging convention (_dbg), and the SAME
                # clock: the pass joins the wall-clock budget (plan §4.3)
                log=((lambda *a, **k: print(*a, file=_sys.stderr,
                                            flush=True))
                     if _dbg else None),
                time_budget_s=time_budget_s, t_start=t0, type_nt=type_nt)
        except Exception as exc:  # noqa: BLE001 - the lane's own CSVs stand
            full_block = {"status": "failed", "error": repr(exc)}
            print(f"[corridor] full-scope refill failed: {exc!r}",
                  file=_sys.stderr, flush=True)
    _phase(f"full-scope refill ({(full_block or {}).get('status')})")
    live = {k: v for k, v in provenance_classes.items()
            if v["accepted"] or v["pulled"]}
    # serialized RoutePlan (§5/§23.5): the validated plan is content-hashed;
    # a later deep-dive can re-derive scope from it without re-draining
    plan = {
        "asked": asked, "bound": bound, "sources": src_types,
        "targets": tgt_types, "knobs": {
            "top_m_base": top_m_base, "length_decay": length_decay,
            "per_pair_length_budget": limit,
            "type_pruned": bool(type_pruned),
            "type_prune_basis": (str(type_prune_basis).lower()
                                 if type_pruned else None),
            "type_prune_scope": (str(type_prune_scope).lower()
                                 if type_pruned else None),
            "type_min_ratio_adj": (float(type_min_ratio_adj)
                                   if type_pruned else None),
            "type_min_synapse": (int(type_min_synapse)
                                 if type_pruned else None),
            "type_min_wmax": (int(type_min_wmax)
                              if type_pruned else None),
            "type_walk_mode": (type_walk_mode if type_pruned else None),
            "type_walk_cap": (int(type_walk_cap) if type_pruned else None),
            "type_revisit_policy": (type_revisit_policy
                                    if type_pruned else None),
            "type_max_type_visits": (int(type_max_type_visits)
                                     if type_pruned else None),
            "type_validate_work_cap": (int(type_validate_work_cap)
                                       if type_pruned else None),
            "per_pair_scope": bool(per_pair_scope),
            "scope_lt_mode": (scope_mode_resolved
                              if per_pair_scope else None),
            "scope_pad": (int(scope_pad) if per_pair_scope else None),
            "scope_hard_cap": (_scope_hard_cap if per_pair_scope else None),
            "scope_discovery_cap": (
                int(scope_discovery_cap) if per_pair_scope else None),
            "scope_table_sha": scope_table_sha},
        "type_prune": type_block,
        "scope": (scope_block if per_pair_scope else None),
        "classes": {k: {"floor": budget.floor,
                        "floor_next": budget.floor_next,
                        "stop": budget.stop, "route_share": budget.share,
                        "pull_cap": budget.pull_cap,
                        "realization_envelope": budget.limit,
                        "counted": ("routes" if budget.count_routes
                                    else "paths"),
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
        if type_pruned and not explicit_out:
            # The typed lane owns its own plan.json; the Fast lane's
            # data_details/route_plan must stay untouched.
            plan_dir = out / "data_details" / "route_plan"
        else:
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
        "run_mode": run_mode,
        "trigger": trigger, "asked": asked, "bound": bound,
        # the reader needs the basis to know which columns are duplicates:
        # on a ratio-basis lane the conditioned denominator IS the plain
        # one, so ratios_adj == ratios by construction.
        "weight_basis": basis,
        "max_interlayer": max_il, "dataset": dataset,
        "separate_hemispheres": separate_hemi,
        "hemisphere_filter": hemi_filter,
        "exclude_intra_type_connections": exclude_intra,
        "exclude_autapses": exclude_autapses,
        "label_mapper": label_mapper is not None,
        "label_mapper_sha": label_mapper_sha,
        "label_mapper_source": label_mapper_source,
        "type_map_sha": type_map_sha,
        "type_map_size": len(type_map),
        # F-PF-005: how the endpoint bodies were enrolled — 'types' /
        # 'prefix' (token expansion), 'bodyids' (explicit bodyId tokens),
        # '+parent_csv' when the run's own enrollment CSVs restricted the
        # endpoints (source_neurons.csv + Checked target_neurons.csv)
        "enrollment": {"sources": src_enrollment, "targets": tgt_enrollment,
                       "source_bodies": len(src_keep) or None,
                       "target_bodies": len(tgt_keep) or None,
                       "source_bodies_effective": len(_src_members),
                       "source_group_counts": source_group_counts,
                       "source_group_counts_all": source_group_counts_all,
                       "source_unenrolled_counts": source_unenrolled_counts,
                       "source_native_counts": source_native_counts,
                       "source_group_mapping_sha": label_mapper_sha},
        "sources": src_types, "targets": tgt_types,
        "connections": str(conn_path), "neuron_table": neuron_source,
        "knobs": {"top_m_base": top_m_base,
                  "length_decay": length_decay,
                  "per_pair_length_budget": limit,
                  "allow_multi_visit": allow_multi_visit,
                  "certify_actual": certify_actual},
        "runtime_s": runtime,
        "phases_s": phases,
        "store_dedup": ("ROI-slice sum per layer; keep-first by sorted "
                        "layer"
                        if validation_basis in ("shortest_store_edge_union", "shortest_dag_store_enforced")
                        else None),
        "store_weight_conflicts": (store_weight_conflicts
                                   if (validation_basis.startswith("shortest_dag") or validation_basis == "shortest_store_edge_union") else None),
        "time_capped": time_capped,
        "time_budget_s": time_budget_s,
        "plan_hash": plan_hash,
        "path_mode": path_mode,
        "validation_basis": validation_basis,
        # P2-N/P2-S: the typed lane's identity and completeness are SEPARATE
        # from the Complete lane's `paths_complete`.
        "type_pruned": bool(type_pruned),
        "type_pruned_complete": (type_block.get("type_pruned_complete")
                                 if type_block is not None else None),
        "type_prune": type_block,
        # Per-pair scope contract (issue 2026-10-22): the scope fields are
        # repeated at the top level so a reader never has to open the typed
        # manifest to know which claim `type_pruned_complete` makes.
        "scope": (scope_block if per_pair_scope else None),
        "scope_mode": (scope_block.get("scope_mode")
                       if per_pair_scope else None),
        "scope_lt": (scope_block.get("scope_lt")
                     if per_pair_scope else None),
        "scope_pad": (int(scope_pad) if per_pair_scope else None),
        "scope_table_sha": scope_table_sha,
        "scope_hard_cap": (_scope_hard_cap if per_pair_scope else None),
        "scope_complete": (type_block.get("scope_complete")
                           if (per_pair_scope and type_block) else None),
        "scope_complete_by_pair": (
            type_block.get("scope_complete_by_pair")
            if (per_pair_scope and type_block) else None),
        "scope_capped_pairs": (
            type_block.get("scope_capped_pairs")
            if (per_pair_scope and type_block) else None),
        "scope_no_path_pairs": (
            type_block.get("scope_no_path_pairs")
            if (per_pair_scope and type_block) else None),
        "scope_validation_cap_pairs": (
            type_block.get("scope_validation_cap_pairs")
            if (per_pair_scope and type_block) else None),
        "scope_exhausted_pairs": (
            type_block.get("scope_exhausted_pairs")
            if (per_pair_scope and type_block) else None),
        "scope_report_table": ("route_scope_pairs.csv"
                               if per_pair_scope else None),
        "scope_input_table": (
            type_block.get("scope_input_table")
            if (per_pair_scope and type_block) else None),
        "bound_source": ("per_pair_scope" if per_pair_scope
                         else "max_interlayer"),
        "paths_complete": _parse_bool(kv.get("paths_complete"), None),
        "shortest_enforced": (store_distances is not None
                              if validation_basis.startswith(
                                  "shortest_dag") else None),
        "validated_walks": (
            type_block.get("validated_walks")
            if type_block is not None else len(walk_rows)),
        "bodyId_paths": len(body_rows),
        "scope_pairs": len(scope_pairs),
        "gap_partition": gap,
        # the lane's second round (synapse >= 1) — mirrors the refill's
        # `full` block so one reader knows both scopes' recovery
        "full": full_block,
        "classes": dict(sorted(provenance_classes.items())),
        "live_classes": sorted(live.keys()),
    }
    (out / "route_provenance.json").write_text(json.dumps(
        provenance, indent=2, default=str), encoding="utf-8")
    floors = [f"{k}: floor={v['floor']} floor_next={v['floor_next']} "
              f"stop={v['stop']} accepted={v['accepted']}"
              f"/share={v.get('route_share')} "
              f"pulled={v['pulled']} (cap {v.get('pull_cap')}) "
              f"emitted={v['emitted']}"
              f"/envelope={v.get('realization_envelope')} "
              f"off_plan={v['off_plan']} "
              f"off_len={v['off_len']}"
              for k, v in sorted(live.items())]
    _type_prune_txt = ""
    _scope_txt = ""
    if type_block is not None:
        _scope_phrase = (" within the per-pair scope" if per_pair_scope
                         else "")
        _type_prune_txt = (
            "- TYPE-PRUNED COMPLETE lane (Phase B): the type graph was "
            f"pruned on `{type_block.get('basis')}` at "
            f"{type_block.get('min_ratio_adj')} (pairs "
            f"{type_block.get('pairs_before')} -> "
            f"{type_block.get('pairs_after')}; "
            f"{type_block.get('mass_removed')} synapses removed). "
            f"{type_block.get('candidate_walks')} type walks"
            f"{_scope_phrase} "
            f"(revisit policy `{type_block.get('revisit_policy')}`) were "
            f"enumerated ({type_block.get('enumeration_capped') and 'CAPPED' or 'exhausted'}); "
            f"{type_block.get('validated_walks')} validated by one "
            "layered forward/backward pass, "
            f"{type_block.get('phantoms')} phantoms rejected, "
            f"{type_block.get('unverified')} unverified, "
            f"{type_block.get('filtered_weak')} filtered by the EXACT "
            f"realized min_ratio_adj > "
            f"{(type_block.get('exact_filter') or {}).get('threshold')} "
            f"gate, {type_block.get('accepted_paths')} paths exported. "
            f"`filter_complete={type_block.get('filter_complete')}`; "
            "the type prune is a COARSE candidate filter; the exported "
            "paths carry `pair_weights` (the pruned pair's universe mass) "
            "beside the realized hop masses and adjusted ratios. "
            "Validation basis `layered_fwd_bwd`; label-revisiting walks "
            "carry the disclosed layered-superset masses (`chain_revisit`). "
            "No bodyId witness rows or realization envelope are produced. "
            f"`type_pruned_complete={type_block.get('type_pruned_complete')}` "
            f"({type_block.get('completeness_reason')})"
            + ("; that claim is WITHIN the per-pair scope only, never "
               "'complete up to max_interlayer', and is NOT the Complete "
               "lane's `paths_complete`." if per_pair_scope else
               "; this is NOT the Complete lane's `paths_complete`.")
            + "\n\n")
        if per_pair_scope:
            _scope_txt = (
                "- PER-PAIR SHORTEST SCOPE: each selected "
                "(source_type,target_type) pair carries its own validated "
                "shortest length `Lt_i` (the first bodyId-validated walk "
                "clearing the exact gate; a phantom type-shortest walk "
                "advances the pass). Its bound is "
                f"min(Lt + {int(scope_pad)}, "
                f"{'no hard cap' if _scope_hard_cap is None else int(_scope_hard_cap)}"
                ") hops. "
                f"`scope_lt={type_block.get('scope_lt')}`, "
                f"`scope_table_sha={scope_table_sha}`, "
                f"`scope_complete={type_block.get('scope_complete')}`; "
                "pairs with no validated path are reported unreached with "
                "no bound, never assigned a global bound. Per-pair "
                "statuses (`exhausted | capped | no_path | validation_cap`) "
                "and the Lt/witness evidence are in "
                "`route_scope_pairs.csv`. `scope_table.csv` is "
                + ("the supplied table, copied byte-for-byte (the table's "
                   "Lt/Lt_i is the audit witness; local rediscovery is "
                   "deliberately skipped for external replay). "
                   if scope_mode_resolved == "external" else
                   "the reusable Pass-A table written by this run. ")
                + "This run's `max_interlayer` does not bound the scoped "
                "pairs.\n\n")
        else:
            _scope_txt = ""
    # The mirror pages (or their skip) are decided BEFORE the disclosure text
    # that reports them: the second-round block below names the pages only
    # when they were actually built (self-review 2026-10-15 — the first
    # version read `_viz_outcome` before assigning it).
    _viz_outcome = _write_mirror_visualizations(
        out, src_types=src_types, tgt_types=tgt_types,
        adj_ratio_tier=asked,
        log=((lambda *a, **k: print(*a, file=_sys.stderr, flush=True))
             if _dbg else None))

    # The lane's SECOND ROUND (plan-corridor-refill Q2), disclosed from the
    # block the round itself returned — including when it failed or found
    # nothing, so a reader never has to infer why a file is missing.
    if full_block is None:
        _second_round_txt = (
            "- SECOND ROUND (synapse >= 1): not attempted (no validated "
            "chains to re-value).\n\n")
    elif full_block.get('status') == 'skipped_not_requested':
        _second_round_txt = (
            "- SECOND ROUND (synapse >= 1): NOT REQUESTED for this typed "
            "lane. The layered lane keeps only its chain tables; pass "
            "`--full-scope` if the additive synapse >= 1 re-value is "
            "wanted. The asked-scope tables above are unaffected.\n\n")
    elif full_block.get('status') == 'failed':
        _second_round_txt = (
            f"- SECOND ROUND (synapse >= 1): FAILED — "
            f"{full_block.get('error')}. The asked-scope tables above are "
            "unaffected.\n\n")
    else:
        _second_round_txt = (
            "- SECOND ROUND (synapse >= 1): the same validated chains "
            "re-valued with no tier through the refill's forward+backward "
            "layered filter. The lane has no budget that raises an applied "
            "threshold, so its sub-threshold band is `[1, asked)`. "
            "`route_full_paths_type.csv` (every lane chain, "
            "`emitted_weights` + `refill_weights` == `weights`), "
            "`route_full_carrying_paths_type.csv` (only chains carrying "
            "sub-threshold mass; `emitted_weights` + `weights` == "
            "`total_weights`), `route_full_type_pairs.csv` (per hop pair: "
            "`refilled_total`, its share `refilled_connection_ratio`, "
            "`universe_type_weights` and `pair_offchain_mass` — total + "
            "off-chain == the universe, so every chain mass printed above is "
            "a lower bound of its pair). Denominators: "
            "`route_type_incoming.csv` (synapse >= 1) and "
            "`route_type_incoming_asked.csv` (the lane's asked tier — the "
            "file the report's corridor Adj. column divides by). At this "
            "floor the conditioned denominator IS the plain one, so the "
            ">= 1 tables carry NO adjusted column. "
            # The mirror pages are named only when they were actually built
            # (F3): the outcome line reports the skip, and a status sentence
            # promising pages that were skipped read as a contradiction
            # (self-review 2026-10-15).
            + ("Mirror pages: `route_full_type_network.html` / "
               "`_heatmap.html`, which carry each pair's `refilled_total` "
               "as the network's Refilled metric. "
               if "built" in str(_viz_outcome or "") else
               "Mirror pages: none in this folder \u2014 see the outcome "
               "line below. ")
            + f"Status: {full_block.get('status')}; chains "
            f"{full_block.get('chains')}, carrying "
            f"{full_block.get('chains_carrying')}, "
            f"{full_block.get('refill_weight_total')} synapses recovered in "
            f"{full_block.get('elapsed_s')}s. Cache guard: "
            f"{full_block.get('cache_guard', 'not_checked')} — the >= 1 "
            "all-post incoming of every post type was checked against this "
            "run's own emitted-tier incoming; a smaller stream value REFUSES "
            "the round (an incomplete connection source must not silently "
            "undercount the universe). Time budget: "
            + ("TIME-CAPPED — the chains listed above are the ones the pass "
               "reached before the lane's wall clock ran out; the rest are "
               "absent by budget, not by absence."
               if full_block.get('time_capped') else
               "complete within the lane's wall clock.") + ")\n\n")
    _basis_detail = (
        " (the run's shortest-DAG discovery store, FLATTENED to "
        "an edge union — routes are bounded routes within that "
        "union, NOT guaranteed minimum-hop per bodyId pair; "
        "per-layer ROI slices are summed to the physical weight "
        "and layers deduplicated keep-first by sorted layer; "
        "cross-layer weight disagreements after summation are "
        "disclosed as store_weight_conflicts)"
        if validation_basis in ("shortest_store_edge_union",
                                "shortest_dag_store_enforced") else
        " (lossless hop-budget prune)"
        if validation_basis == "lossless_prune" else
        " (store absent — lossless-prune FALLBACK, not min-hop-"
        "restricted)")
    _basis_line = (
        f"- universe basis: {validation_basis}{_basis_detail}; per-walk "
        f"validation: {type_block.get('validation_basis')}\n\n"
        if type_block is not None else
        f"- validation basis: {validation_basis}{_basis_detail}\n\n")
    (out / "README.md").write_text(
        ("# Type-pruned complete lane" if type_block is not None
         else "# Corridor lane") + f" ({out.name})\n\n"
        "Standalone route-scoped pass at the ASKED threshold; never merged "
        "into allpaths (plan-corridor-lane.md §23.5)."
        + (" The type-pruned lane owns this namespace; the Fast lane's "
           "route_scope_paths/ is never written here." if type_block is not None
           else "") + "\n\n"
        f"- trigger: {trigger} (run mode `{run_mode}`, lane call "
        f"`{corridor_mode}`, asked {asked}, "
        f"bound {bound})\n"
        f"- validated walks (type rows): {len(walk_rows)} over "
        f"{len(scope_pairs)} scope type-pairs; "
        + (f"bodyId paths: {len(body_rows)}\n" if type_block is None else
           "bodyId witness export: disabled (layered forward/backward "
           "validation)\n")
        + f"- runtime: {runtime}s\n"
        + _basis_line
        + ("- RATIO BASIS: the corridor admits each edge on its own F9 "
           "input share (weight / the post neuron's FULL-table incoming "
           ">= t_r = the run's Min Connection Ratio), ranks type pairs by "
           "the Et type-level ratio (pair admitted mass / the post type's "
           "full incoming), and exports BOTH units per hop (synapse "
           "masses + the mean realizing-edge ratio). Every realized hop "
           "clears t_r by construction -- the structural bodyId floor. "
           "The pair table's Et weight is the pair's true connection "
           "strength: on a budget-cut run it exceeds the parent run's "
           "PATH-SUPPORTED connection_ratio by design (that table counts "
           "only mass on the run's emitted paths).\n\n")
        + (("Per-class disclosure (F-PF-002: floor = lowest accepted walk UB "
        "and floor_next = first skipped candidate's UB are CANDIDATE-BOUND "
        "quantities — optimistic upper bounds from per-hop max edges, not "
        "the strengths of realized/omitted neuron routes; the lane is "
        "upper-bound-guided representative exploration, not a "
        "strongest-actual-route certificate):\n\n"
        + "- TWO BUDGETS, TWO CURRENCIES (2026-10-16): the class budget is "
          "counted in VALID TYPE-LEVEL ROUTES (route_share = max(1, round(top_m_base "
          "* length_decay^(h-1))) per source-target type pair × hop length), "
          "and a candidate pays it only once — a corridor whose hops compose "
          "into hundreds of neuron paths no longer buys the whole class with "
          "its realization count. `--pair-length-budget` is the class's "
          "REALIZATION envelope (bodyId witness paths enumerated for the "
          "class), and every accepted route keeps at least one witness, so "
          "the envelope truncates a fat route's tie mass and never the route "
          "list. Phantoms cost candidate pulls, never the route share: the "
          "pull cap (2 × route_share) is what closes a phantom-heavy class, "
          "disclosed as stop=probe_cap, while stop=budget means the class "
          "published its whole share.\n\n")
           if type_block is None else
           "- TYPE-PRUNED BUDGETS: no class share and no realization "
           "envelope. Every enumerated type walk is validated by ONE layered "
           "forward/backward pass, and every walk clearing the exact "
           "realized `min_ratio_adj` gate is exported — the walk list is "
           "complete within the disclosed scope, not a top-M sample.\n\n")
        + ("- route_connection_type.csv carries TWO support sets, named: "
           "`weight`/`universe_type_weights` is the WHOLE admitted type-pair "
           "mass; `selected_support_mass`/`realized_mass` is the mass on "
           "the lane's accepted chains (layered kept edges in the typed "
           "lane; emitted bodyId witnesses in the Fast lane). Both are "
           "UNIONS of edges over realizations — a large aggregate never "
           "implies one neuron path carrying all of it. The typed lane's "
           "`realizing_paths` is 0 (no bodyId rows); use `used_by_walks`.\n\n")
        + _type_prune_txt
        + _scope_txt
        + _second_round_txt
        + (f"- mirror pages (route_type_* / route_full_type_*): "
           f"{_viz_outcome}\n\n" if _viz_outcome else "")
        + ("\n".join(f"- {f}" for f in floors) if floors
           else ("- type-pruned lane: no corridor classes (see "
                 "type_prune_manifest.json)\n" if type_block is not None
                 else "- no validated corridors (all classes dry)\n"))
        + (f"\n- TIME-CAPPED at {time_budget_s}s — enumeration was cut "
           "early; remaining classes/walks are absent by budget, not by "
           "absence.\n" if time_capped else ""),
        encoding="utf-8")
    return True


def _full_scope_refill(out, *, conn, tmap, body2type, src_ids, tgt_ids,
                       walk_rows, typed=None, type_membership, type_incoming,
                       asked, log=None, time_budget_s=None, t_start=None,
                       type_nt=None, cut_keys=None):
    """Q2 (plan-corridor-refill-and-mode-tabs): the lane's SECOND ROUND.

    The SAME chains the lane validated, re-valued at **synapse >= 1**
    through the refill's own forward + backward layered filter, so the Fast
    mode offers the same scope pair the Complete mode has and can be read
    standalone. The lane has no budget that raises an applied threshold, so
    its sub-threshold band is `[1, asked)`: the mass the lane's own
    admission cut left out. `emitted` = the edges the lane admitted at
    `asked` (the cut keys), `refill` = the rest.

    Denominators are the lane's plain all-inputs totals (`type_incoming`,
    summed over the whole collapsed table). At the >= 1 floor the
    conditioned denominator IS the plain one, so the full-scope tables
    carry one ratio column — an Adj column there would duplicate it
    (gate G5, same doctrine as the refill's full scope).

    Returns the lane's `full` provenance block (None when there are no
    chains to re-value)."""
    import time as _t

    say = log or (lambda *a, **k: None)
    started = _t.monotonic()
    chains = []
    seen = set()
    for row in walk_rows:
        seq = tuple(str(p).strip() for p in str(row["path"]).split("->"))
        if len(seq) < 2 or seq in seen:
            continue
        seen.add(seq)
        chains.append(seq)
    if not chains:
        return None

    # -- time budget (plan §4.3): this pass JOINs the lane's clock. It is
    # the last stage, so a lane with no time left skips it explicitly rather
    # than silently dropping the round; a deadline hit mid-way marks the
    # block time_capped so the tables say they are partial.
    deadline = (_t.monotonic() + float(time_budget_s)
                if time_budget_s else None)
    if deadline is not None and t_start is not None:
        deadline = float(t_start) + float(time_budget_s)
    if deadline is not None and _t.monotonic() > deadline:
        say('  · corridor full-scope refill: skipped — the lane\'s time '
            'budget was spent before this round could start')
        return {'status': 'skipped_time_budget', 'chains': len(chains),
                'scope': 'synapse >= 1', 'asked': asked,
                'note': 'the lane ran out of its wall-clock budget before '
                        'the synapse >= 1 re-value; the asked-scope tables '
                        'are the landmark of this run'}

    # -- cache guard (plan §4.2, mirroring the complete mode's full scope):
    # at synapse >= 1 every post type's all-post incoming must be at least
    # the run's own emitted-tier incoming (data_details/type_incoming_
    # emitted.csv). A smaller value means the connection source reads
    # incomplete RELATIVE TO THE RUN, and publishing these tables would
    # silently undercount the universe — refuse, and let run_lane disclose
    # the refusal, exactly as the complete mode raises.
    cache_guard = 'not_checked'
    _run_root = out.parent
    _emitted_csv = _run_root / "data_details" / "type_incoming_emitted.csv"
    if _emitted_csv.exists():
        try:
            import pandas as _pd_guard
            _emitted = _pd_guard.read_csv(_emitted_csv)
            _tcol = next((c for c in ('type', 'type_post', 'type_pre')
                          if c in _emitted.columns), None)
            _icol = next((c for c in ('incoming', 'incoming_weight',
                                      'incoming_threshold')
                          if c in _emitted.columns), None)
            if _tcol and _icol:
                _bad = []
                for _, _r in _emitted.iterrows():
                    _key = str(_r[_tcol])
                    _want = float(_r[_icol] or 0.0)
                    _got = float(type_incoming.get(_key, 0.0) or 0.0)
                    if _want > _got + 1e-6:
                        _bad.append((_key, _want, _got))
                if _bad:
                    return {
                        'status': 'failed', 'scope': 'synapse >= 1',
                        'asked': asked, 'chains': len(chains),
                        'cache_guard': 'FAILED',
                        'error': (
                            'full-scope cache guard FAILED — '
                            f'{len(_bad)} type(s) have a smaller '
                            'stream-derived all-post incoming than the '
                            "run's own emitted-tier incoming (first: "
                            f'{_bad[0][0]}: stream {_bad[0][2]:g} < emitted '
                            f'{_bad[0][1]:g}). The connection source '
                            'appears incomplete relative to the run; '
                            'refusing to write full-scope tables that '
                            'would silently undercount the universe.')}
                cache_guard = 'ok'
        except Exception:  # noqa: BLE001 — the guard is best-effort
            cache_guard = 'not_checked'

    import polars as pl
    hop_pairs = sorted({(seq[i], seq[i + 1])
                        for seq in chains for i in range(len(seq) - 1)})
    want = pl.DataFrame({"type_pre": [a for a, _b in hop_pairs],
                         "type_post": [b for _a, b in hop_pairs]})
    # The >= 1 universe, restricted to the lane's own hop pairs: the same
    # collapsed table the lane already read (no second cache pass), joined
    # to the type map on both ends.
    edges = (conn.rename({"bodyId_pre": "u", "bodyId_post": "v"})
             .join(tmap.rename({"bid": "u", "t": "tpre"}), on="u",
                   how="inner")
             .join(tmap.rename({"bid": "v", "t": "tpost"}), on="v",
                   how="inner")
             .rename({"tpre": "type_pre", "tpost": "type_post"})
             .join(want, on=["type_pre", "type_post"], how="inner")
             .filter(pl.col("weight") >= 1)
             .select(["u", "v", "weight", "type_pre", "type_post"]))
    pair_universe: dict = {}
    for a, b, w in (edges.group_by(["type_pre", "type_post"])
                    .agg(pl.col("weight").sum().alias("w"))
                    .iter_rows()):
        pair_universe[(a, b)] = float(w)
    pair_edges_full: dict = {}
    for u, v, w, a, b in edges.iter_rows():
        pair_edges_full.setdefault((a, b), {}).setdefault(
            u, []).append((v, float(w)))

    # cut = the edges the lane actually admitted at its own tier. The
    # caller may pass the set and release the typed frame first (memory
    # bound, issue 2026-10-23).
    if cut_keys is None:
        cut_keys = set(zip(typed["bodyId_pre"].to_list(),
                           typed["bodyId_post"].to_list())) \
            if typed is not None else set()

    from type_level_refill import (
        _complete_rows_from_structures, _layered_chain_structures,
        _refill_rows_from_structures)
    seq_edges: dict = {}
    seq_nodes: dict = {}
    seq_cls: dict = {}
    chains_revisit = 0
    no_realization = 0
    time_capped_here = False
    for seq in sorted(chains):
        if deadline is not None and _t.monotonic() > deadline:
            time_capped_here = True
            break
        dp = _layered_chain_structures(
            seq, pair_edges_full, src_ids, tgt_ids, body2type, cut_keys)
        if dp is None:
            no_realization += 1
            continue
        seq_edges[seq] = dp["edges_at"]
        seq_nodes[seq] = dp["nodes_at"]
        seq_cls[seq] = dp["edges_cls_at"]
        if dp["revisit"]:
            chains_revisit += 1
    if not seq_cls:
        return {"status": "empty", "chains": len(chains),
                "chains_revisit": chains_revisit,
                "cache_guard": cache_guard,
                "time_capped": time_capped_here,
                "note": "no chain realized at synapse >= 1"}

    carrying = {seq: cls for seq, cls in seq_cls.items()
                if any((c.get("refill") or {}) for c in cls)}
    complete_rows, _stats = _complete_rows_from_structures(
        seq_cls, seq_nodes, type_membership, type_incoming)
    carrying_rows = (
        _refill_rows_from_structures(
            {s: seq_edges[s] for s in carrying},
            {s: seq_nodes[s] for s in carrying},
            type_membership, type_incoming,
            {s: seq_cls[s] for s in carrying})
        if carrying else [])

    # pair table: distinct-edge sums per hop pair, split by class
    pair_acc: dict = {}
    for seq in seq_cls:
        for i, classes in enumerate(seq_cls[seq]):
            key = (seq[i], seq[i + 1])
            acc = pair_acc.setdefault(key, {"emitted": {}, "refill": {}})
            for cls, edges_at in classes.items():
                acc.setdefault(cls, {}).update(edges_at or {})
    pair_rows = []
    for (a, b), acc in sorted(pair_acc.items()):
        emitted = sum(acc.get("emitted", {}).values())
        refill = sum(acc.get("refill", {}).values())
        total = emitted + refill
        denom = float(type_incoming.get(b, 0.0) or 0.0)
        universe = pair_universe.get((a, b), total)
        pair_rows.append({
            "type_pre": a, "type_post": b,
            # the mirror family's transmitter column names the PRESYNAPTIC
            # type (F-EX-006), so the second round's pages can colour edges
            # by NT exactly like the run's own Network page
            "nt_type": (type_nt or {}).get(a, ""),
            "emitted_weight": emitted, "refill_weight": refill,
            "refilled_total": total,
            "refilled_connection_ratio": (round(total / denom, 6)
                                          if denom > 0 else None),
            # the >= 1 universe this hop pair's mass is drawn from — the
            # report's `Universe type weights` column (F4/Q3 rename of
            # `universe_pair_mass`), so every printed hop ratio is
            # reproducible against the whole admitted pair, not just the
            # chain's share of it
            "universe_type_weights": round(universe, 2),
            # mass in this type pair that no lane chain realizes
            "pair_offchain_mass": round(max(universe - total, 0.0), 2)})

    import pandas as _pd
    (_pd.DataFrame(complete_rows, columns=_FULL_PATH_COLUMNS).to_csv(
        out / "route_full_paths_type.csv", index=False))
    (_pd.DataFrame(carrying_rows, columns=_FULL_CARRYING_COLUMNS).to_csv(
        out / "route_full_carrying_paths_type.csv", index=False))
    (_pd.DataFrame(pair_rows, columns=_FULL_PAIR_COLUMNS).to_csv(
        out / "route_full_type_pairs.csv", index=False))
    # the >= 1 denominators, so a reader can reproduce every ratio in the
    # two path tables and the pair table from one file
    (_pd.DataFrame([{"type_post": k, "incoming_threshold": float(v)}
                    for k, v in sorted(type_incoming.items())],
                   columns=["type_post", "incoming_threshold"]).to_csv(
        out / "route_type_incoming.csv", index=False))

    refill_edges = sum(len(c.get("refill") or {}) for cls in
                       seq_cls.values() for c in cls)
    block = {
        "status": "ok" if carrying else "nothing_below_tier",
        "scope": "synapse >= 1",
        "asked": asked,
        "chains": len(seq_cls),
        "chains_carrying": len(carrying),
        "chains_no_realization": no_realization,
        "chains_revisit": chains_revisit,
        # plan §4.2 key parity with the complete mode's full block
        "cache_guard": cache_guard,
        # plan §4.3: the pass shares the lane's clock; a deadline hit here
        # means the tables are PARTIAL, not that the lane's chains lack
        # lower-tier mass
        "time_capped": time_capped_here,
        "refill_edges": refill_edges,
        "refill_weight_total": round(sum(
            sum((c.get("refill") or {}).values())
            for cls in seq_cls.values() for c in cls), 2),
        "emitted_weight_total": round(sum(
            sum((c.get("emitted") or {}).values())
            for cls in seq_cls.values() for c in cls), 2),
        "enumeration": "none (layered per-position filtering reference)",
        "elapsed_s": round(_t.monotonic() - started, 2),
    }
    if time_capped_here:
        say(f"  · corridor full-scope refill: TIME-CAPPED after "
            f"{block['chains']} of {len(chains)} chains — the remaining "
            "chains are absent by budget, not by absence")
    say(f"  · corridor full-scope refill: {block['chains']} chains, "
        f"{block['chains_carrying']} carrying "
        f"{block['refill_weight_total']:g} synapses below the lane's tier "
        f"({block['elapsed_s']}s)")
    _write_mirror_visualizations(
        out, src_types=sorted({s for seq in seq_cls for s in (seq[0],)}),
        tgt_types=sorted({seq[-1] for seq in seq_cls}),
        adj_ratio_tier=1, base_filename="route_full_type",
        conn_csv=out / "route_full_type_pairs.csv",
        weight_column="refilled_total",
        refilled_column="refilled_total")
    return block


_FULL_PATH_COLUMNS = ["path", "weights", "ratios", "min_weight",
                      "min_ratio", "length", "nt_types", "coverage",
                      "emitted_weights", "refill_weights"]
_FULL_CARRYING_COLUMNS = ["path", "weights", "ratios", "min_weight",
                          "min_ratio", "length", "nt_types", "coverage",
                          "emitted_weights", "total_weights"]
_FULL_PAIR_COLUMNS = ["type_pre", "type_post", "nt_type",
                      "emitted_weight",
                      "refill_weight", "refilled_total",
                      "refilled_connection_ratio",
                      "universe_type_weights",
                      "pair_offchain_mass"]


def _write_mirror_visualizations(out, src_types=(), tgt_types=(),
                                 adj_ratio_tier=None,
                                 base_filename="route_type",
                                 conn_csv=None, weight_column=None,
                                 refilled_column=None, log=None):
    """V1.4 (plan-corridor-alignment-and-viz-bases §4.4): the mirror
    heatmap + network from route_connection_type.csv — the refill's own
    recipe (coana _vp + create_heatmap/create_network). Best-effort:
    failures leave the CSVs untouched (never fatal to the lane).
    src_types/tgt_types drive the network's node roles (an edge-list
    input has no path-position roles — without them the legend reads
    'Source (0) / Target (0)').

    Returns a one-line outcome for the lane README: 'built', a
    'skipped — …' reason, 'nothing to draw', or 'FAILED'. Never
    raises."""
    try:
        import pandas as pd
        from pathlib import Path as _P
        # REDUNDANCY (user, 2026-10-15): when the run wrote its OWN
        # visualization/Network_*.html, that page is a strict SUPERSET of
        # this mirror (measured: 38 pairs vs the lane's 26, same style, and
        # — before the NT fix — richer, because it carries the transmitter
        # per edge). The duplicate pages are skipped; the lane's CSVs are
        # untouched and a fast-only run, which has no main visualization,
        # still gets these pages.
        _main_viz = (out.parent / "visualization")
        if _main_viz.is_dir() and list(_main_viz.glob("Network_*.html")):
            return ("skipped \u2014 the run has its own visualization/ "
                    "pages (a superset of this scope; this folder's "
                    "CSVs are the lane's own data)")
        # Defaults: the lane's asked-tier mirror (route_connection_type.csv).
        conn_csv = _P(conn_csv) if conn_csv else (
            out / "route_connection_type.csv")
        weight_column = weight_column or "selected_support_mass"
        if not conn_csv.exists():
            return 'nothing to draw (no connection table)'
        frame = pd.read_csv(conn_csv)
        # F-PF-008 (audit 2026-10-13): the route-scoped visualization
        # draws the SELECTED-support mass — a route-support panel, not
        # the whole admitted type-pair universe (the old `weight` column
        # drew 101 synapses for a hop whose emitted witnesses carry 1).
        # The universe stays in the CSV under
        # `weight`/`universe_type_weights`.
        frame = frame[frame[weight_column].fillna(0) > 0]
        if frame.empty:
            return
        cols = ["type_pre", "type_post", weight_column]
        rename = {weight_column: "weight", "type_pre": "source",
                  "type_post": "target"}
        if "nt_type" in frame.columns:
            cols.append("nt_type")
            rename["nt_type"] = "nt_type"
        for src_col, dst in (("connection_ratio", "ratio"),
                             ("refilled_connection_ratio", "ratio"),
                             ("connection_ratio_adj", "ratio_adj")):
            if src_col in frame.columns and dst not in rename.values():
                cols.append(src_col)
                rename[src_col] = dst
        # F7.1/F7.2 parity with the refill's pages: the lane's second round
        # knows each pair's refilled total, so its network/heatmap carry it
        # as the 'Refilled' metric instead of the (N/A) a page with no
        # metric map shows. The asked mirror has NO refill round, so it
        # stays honestly unavailable there.
        pair_extra = None
        if refilled_column and refilled_column in frame.columns:
            totals = {}
            for _, r in frame.iterrows():
                if pd.notna(r[refilled_column]):
                    totals[(str(r['type_pre']), str(r['type_post']))] = float(
                        r[refilled_column])
            if totals:
                pair_extra = {'refilled': totals}
        from vispath_pkg.vispath import VisualizePath
        vp = VisualizePath(
            path_file=frame[cols].rename(columns=rename),
            output_folder=str(out), showfig=False, verbose=False,
            adj_ratio_threshold=adj_ratio_tier,
            pair_extra_metrics=pair_extra,
            color_edges_by_nt=("nt_type" in frame.columns),
            edgeN_limit=500)
        vp.base_filename = base_filename
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
        return 'built'
    except Exception as exc:  # noqa: BLE001 - best-effort, never silent
        # The lane CSVs are the deliverable, so a viz failure must not
        # break the lane — but a bare `pass` here once hid a NameError for
        # 20 minutes while the HTML pages silently kept their stale bytes.
        # A skipped artifact has to leave a trace a reader can find.
        import sys as _s
        try:
            (out / '.mirror_visualizations_failed').write_text(
                f'mirror heatmap/network generation failed: {exc!r}\n',
                encoding='utf-8')
        except OSError:
            pass
        print(f'[corridor] mirror visualizations failed: {exc!r}',
              file=_s.stderr, flush=True)
        return ('FAILED \u2014 see .mirror_visualizations_failed '
                'beside this README')


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


def _label_mapper_payload(label_mapper):
    """Stable JSON-able mapping content for hashing.

    ``LabelMapper`` stores its groups in ``_source_mapping`` /
    ``_target_mapping`` / ``_intermediate_mapping``; it has no
    ``overall_mapping`` attribute. Reading the latter alone made every
    mapping hash ``{}`` and let a grouped run reuse a universe built
    under the native labels. Returns None when the mapper carries no
    readable mapping content."""
    if label_mapper is None:
        return None
    overall = getattr(label_mapper, "overall_mapping", None)
    if overall:
        return overall
    payload = {}
    for attr in ("_source_mapping", "_target_mapping",
                 "_intermediate_mapping"):
        mapping = getattr(label_mapper, attr, None)
        if not mapping:
            continue
        try:
            normalized = {}
            for label, ds_map in dict(mapping).items():
                normalized[str(label)] = {
                    str(ds): [list(x) if isinstance(x, (list, tuple))
                              else [x] for x in groups]
                    for ds, groups in dict(ds_map).items()}
        except Exception:  # noqa: BLE001 - an unhashable mapper is disclosed
            continue
        if normalized:
            payload[attr] = normalized
    if payload:
        return payload
    # A wrapper mapper may expose only its source files; hash their bytes.
    import hashlib
    for attr in ("_source_file", "_target_file", "_intermediate_file"):
        path = getattr(label_mapper, attr, None)
        if not path:
            continue
        try:
            with open(path, "rb") as _fh:
                payload[attr] = hashlib.md5(_fh.read()).hexdigest()
        except OSError:
            continue
    return payload or None


def _universe_signature(conn_path, neuron_source, asked, drop_untyped,
                        exclude_intra, separate_hemi, hemi_filter,
                        label_mapper, exclude_autapses=False,
                        type_map_sha=None) -> str:
    """Fail-closed cache key material (P1.1): file stats of BOTH input
    tables + every filter that shapes the universe + the EFFECTIVE
    bodyId->type digest (and, as a fallback, the mapper content). Any
    change ⇒ a different key ⇒ rebuild; nothing stale is ever served
    (cache-survey c2e26c1 discipline; the lane uses direct file stats
    because it can run on release tables where the engine's
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
             f"{separate_hemi}|{hemi_filter}|{exclude_autapses}".encode())
    # The cached frame carries effective labels. The digest is the exact
    # map the lane used; without it a grouped mapping silently reuses a
    # native-label universe (MCNS grouped-source bug, 2026-10-22).
    if type_map_sha:
        h.update(f"|tm:{type_map_sha}".encode())
    payload = _label_mapper_payload(label_mapper)
    if payload is not None:
        try:
            import json as _json
            h.update(_json.dumps(
                payload, sort_keys=True, default=str).encode())
        except Exception:
            h.update(b"|mapper-unhashable")
    elif label_mapper is not None:
        h.update(b"|mapper-no-mapping")
    return h.hexdigest()[:12]


def _typed_universe_cached(conn_path, neuron_source, conn, tmap, asked,
                           drop_untyped, exclude_intra, separate_hemi,
                           hemi_filter, label_mapper, _phase,
                           exclude_autapses=False, type_map_sha=None):
    """typed/pairs universe with a signature-keyed parquet cache
    (P1.1): cache/route_plan/<dataset_safe>/universe__w<min>__u<0|1>__a<0|1>__<sig12>.parquet (typed edges; pairs re-derive in <1 s). Atomic write;
any signature mismatch ⇒ rebuild — fail-closed.

    ``type_map_sha`` is the EFFECTIVE bodyId->type digest. It is part of
    the key because the cached `typed` frame carries the effective labels:
    a custom mapping that folds native subtypes into a source group must
    never reuse a universe built under another mapping."""
    import os
    from pathlib import Path as _P
    sig = _universe_signature(conn_path, neuron_source, asked,
                              drop_untyped, exclude_intra, separate_hemi,
                              hemi_filter, label_mapper, exclude_autapses,
                              type_map_sha=type_map_sha)
    cache_file = None
    if sig:
        dataset_safe = _P(str(conn_path)).resolve().parent.name or "default"
        cache_dir = _P(__file__).resolve().parents[1] / "cache" \
            / "route_plan" / dataset_safe
        cache_file = cache_dir / (
            f"universe__w{asked}__u{int(drop_untyped)}__"
            f"a{int(exclude_autapses)}__{sig}.parquet")
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
    from utils.label_utils import canonical_neurotransmitter
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
                canon = canonical_neurotransmitter(nt)
                if canon:
                    counts[t][canon] += 1
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
