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
| Type-level ratio | **mass recompute over the pair's INVOLVED posts**: emitted synapse mass ÷ the all-post incoming mass of exactly the post bodyIds receiving this pair's emitted edges — never a fold of bodyId ratios, and never the full-type mass. With every bodyId pair clearing t_r, the mediant inequality (n_i/m_i ≥ k ⇒ Σn_i/Σm_i ≥ k) guarantees the aggregate clears t_r too; a full-type denominator would drag it below via zero-numerator members |
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
