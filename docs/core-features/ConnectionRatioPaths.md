# Connection-Ratio Pathfinding Lane (Phase 1, standalone)

A parallel FindAllPath-analog lane in which the **edge weight basis is the
bodyId-level connection ratio** instead of the synapse count. The
threshold, the Edge Budget, and the StrongestFirst ranking all keep their
production semantics — evaluated on ratio values. Plan of record:
`_plan/plan-connection-ratio-pathfinding.md` (§2 semantics, §18
exploration findings).

## Definition (F9, threshold-free)

```
ratio(u → v) = synapses(u → v) / total_incoming(v)
total_incoming(v) = ALL-post incoming synapse mass of v at min_weight=1
                    over the FULL dataset table
traversal_probability = min(1, ratio / 0.3)
```

The denominator is the same F9 definition as the pipeline's
`connection_ratio` readout columns — cone-external inputs never shrink
it. A path's strength is its **ratio bottleneck** (the minimum ratio along
it); paths are emitted strongest-first by the PRODUCTION
`find_paths_strongest_first` engine, so τ tie-drain and determinism are
inherited from the same code path synapse runs use.

## What changes vs a synapse run — and what does not

| Aspect | Behavior |
|---|---|
| Threshold | `min_ratio` (0 < t_r ≤ 1), same `>=` convention |
| Edge Budget | floors at the weakest ratio tier whose hop-closed cone fits the cap (gallop/bisect, ≤8 probes default; a floored lane ≡ an unfloored lane at that tier) |
| StrongestFirst | bottleneck = min ratio; budget drains all ties at τ; default budget 1,000,000 |
| Admission guards | `drop_untyped` DEFAULT ON (untyped neurons never intermediates; an untyped ENROLLED source/target keeps enrollment, loses its edges), `exclude_intra_type` DEFAULT OFF |
| Type-level ratio | **full-type mass recompute (round-9)**: emitted synapse mass ÷ the post TYPE's total all-post incoming mass over its full membership — aggregating every type bodyId avoids single-connection artifacts (a high ratio resting on one low-mass member). The per-pair `type_coverage` column (involved members / total members, n/N) and the per-node coverage lists on the type paths expose low-support pairs |
| W\* | the P1 measured ceiling: max ratio bottleneck over emitted paths |
| Empty cone | STATUS `no_paths` with a disclosure — not a refusal: ratio thresholds compound per hop, so high `t_r` legitimately yields empty cones on hub-targeted enrollments |

Real-data scale note (FAFB v783, aMe12→PPL101, bound 2): the raw cone is
1.78M edges with ratios q50 0.0027 / q90 0.010, and viable thresholds sit
at ~5e-4..2e-3 — two orders below synapse ranges. The
`ConnectionRatio_Filter.md` tables (0.01–0.1) are direct-connection-era
guidance, not pathfinding defaults.

## CLI

```bash
python scripts/ConnectionRatioPaths.py \
    --connections cache/<dataset>/connections.parquet \
    --sources <bodyIds> --targets <bodyIds> \
    --min-ratio 0.0005 --max-interlayer 2 \
    [--neuron-table ... --dataset ... --mapping-file ...] \
    [--edge-budget N --path-budget N --max-probes 8] \
    [--separate-hemispheres --hemisphere-filter left|right] \
    [--exclude-intra-type | --keep-untyped] \
    [--aggregate-method product|average|ratio] \
    [--in-run <find-paths folder> | --out <folder>]
```

`--connections` must be the full-dataset natural-weight table (the same
universe the pipeline's connection cache stores) — it is the denominator
universe. `--in-run` reads the enrollment + run flags READ-ONLY and
writes to a SIBLING folder (`<run>_ratio_L<d>r<t>_<ts>`), never into the
run. Refusals print `REFUSED —` and exit 1 (bad t_r, missing enrollment
ids, missing columns, unloadable mapping file, ratio-basis refill runs).

## Outputs

`ratio_paths_bodyId.csv` (paths with both bases: `ratio_bottleneck`,
`synapse_bottleneck`, per-hop `ratios`), `data_details/ratio_edges_bodyId.csv`
(per on-path edge, both bases + traversal counts),
`data_details/ratio_type_pairs.csv` (mass-recomputed type readouts),
`data_details/ratio_synapse_map.csv` (the ratio→synapse mapping:
`implied_syn_cutoff = max(1, ceil(t_r · total_incoming))` per post — a
sub-synapse cutoff would admit every edge; no-op posts disclosed),
`data_details/ratio_provenance.json` (the 14-key float-unit threshold
provenance mirror + tier/probe traces + guard counts),
`parameters.txt` (frame-compatible keys incl.
`weight basis: connection_ratio (bodyId)`).

## Relation to the frame (Phase 2+)

The pipeline's `min_ratio` parameter and `r{decimal}` folder vocabulary
already exist but are force-disabled for pathfinding (F9: ratio is a
readout). This lane is NOT a re-enabled filter — it is a separate weight
basis (the bottleneck itself becomes the weakest fractional input along
the path). In-pipeline integration (a `weight_basis` field, UI selector,
dual-basis exports, pair-report support) is planned in the plan file's
§12-§16 and needs its own approval round. The type-level refill is
permanently disabled for ratio runs (its semantics are synapse-budget
recovery). Related: `TypeLevelRefill.md`, `ConnectionRatio_Filter.md`,
`PathFinding_Methods.md`.


## Phase 3 (2026-10-03)

The in-pipeline lane now covers everything Phase 1-3 planned:

- **Shortest Paths** runs the ratio lane (threshold applied at the
  target-rooted backward fetch, ratio bottlenecks in the per-pair
  min-hop enumeration, r-suffix folders, float provenance). The
  batched/store discovery path filters at the fetch; the monolithic path
  filters at the shared frame attach.
- **Multi-threshold replay** accepts float ratio tiers
  (`FindAllPathMultiThreshold([0.001, 0.002, 0.005])`): float
  normalization, per-slice `weight_ratio` filters, float canonical-tau
  (the weakest distinct ratio tier above w2 — NOT `int(w2)+1`, which
  would collapse every sub-1 ratio to 1), `L{d}r{t}` slice folders,
  float `replayed_from` gates, and float threshold-collapse discipline.
  Verified against fresh per-threshold runs (set equality incl. the
  canonical collapse folder). In the Cross-Dataset tool, the Replay
  Paths checkbox applies under the ratio basis too.
- **Type-level refill** runs for ratio-basis budget-bitten runs: float
  threshold parsing, ratio-weighted re-enumeration, SYNPASE-mass refill
  records (dual-basis), the table-reproduction anchor intact. Refills
  fire for ordinary and most replay-slice ratio folders; a canonical
  collapse folder whose stamps say complete (while its set is a drained
  subset) refuses honestly at the anchor — stamping those slices with
  the t0 bite state is the one open Phase-3 item.
- **Density capture + auto-mode alignment run on float ratio tiers**
  (implemented 2026-10-04): the density arrays are float64 F9 ratios
  (`weight_axis: per_connection_ratio`, `weight_basis` in the meta),
  `w_start` is the applied ratio threshold (no synapse floor of 3),
  curve grids are the distinct ratio tiers (≤48, deterministically
  thinned), the alignment prober searches the distinct-tier ladder
  instead of the integer grid, and cross-dataset AUTO mode works under
  the ratio basis (bootstrap at the float floor, vertical spine over
  the window intersection, density-matched horizontal rows when the
  bands intersect). Synapse captures are byte-identical to before (all
  casts are value-preserving: integral floats render as ints).
- **The query/export identity layer is float-preserving** (audit round
  2026-10-04): query rows, raw-run jobs, threshold views, the
  `threshold_combinations.csv` manifest (incl. `raw_run_key` in the
  `minratio_{decimal}` grammar), comparison points, and the HTML
  report's provenance/chart cells all carry exact float tiers — an
  earlier `int()` in that chain truncated every sub-1 tier to 0 in the
  exported views while the run itself was correct. Two honest
  degradations remain by design: the union type-coverage
  below-threshold/not-recruited diagnosis (a synapse-weight compare)
  reports "unavailable under the connection-ratio basis" instead of
  guessing, and F7 auto-extension (the integer k × τ ladder) is
  synapse-only.
- Still deferred: ratio-basis shortest-mode replay (shortest is never
  replayable in any basis).
