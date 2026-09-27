# Auto Type Mapping — Implementation Report

*Technical reference for the cross-dataset auto type mapping engine as of
2026-09-25 (inline user-date markers record later revisions). Companion to the user-facing `docs/AUTO_TYPE_MAPPING.md`; this
report documents the implementation: components, the bridge-rule algebra,
the derivation walk, resolution precedence, the shared UI backend, pooling,
and the visualization contract.*

## 1. Components and data flow

```
male-cns v1.0/v0.9 neuron_df (release-aware crosswalk tables)
        │  type / flywireType / hemibrainType / mancType cells
        ▼
CrossDatasetTypeMapper                      src/comparison/cross_dataset_type_mapper.py
  ├─ _build_type_mappings()        stored 1-to-1 mappings + N-to-1 / 1-to-N conflicts
  ├─ _load_flywire_type_tables()   per-release FAFB/BANC labels + additional-name tables
  ├─ _apply_banc_label_overlay()   curated BANC per-dataset label votes
  ├─ _apply_banc_release_overlay() root_626 ↔ root_888 type relation
  ├─ _build_mcns_v09_mappings()    native v0.9 + explicit same-name alias
  ├─ _apply_annotation_bridge_overlay()   same-name identity + annotation-bridge pairs
  ├─ get_type_bridges()            derivation chains (the evidence algebra, §3)
  ├─ get_alias_candidates()        per-dataset alias candidates (rename / same name / one-of-N)
  ├─ get_mapping_decision()        source/target-scoped accepted, split, evidence, or conflict state
  └─ get_mapped_type()             stored-mapping lookup (compatibility-only; see §4.1)
        ▼
mapped_type_targets()                       ui/neuron_index.py — THE shared backend (§5)
  ├─ 'See available neurons' viewer mapped view   (enrich_native_type_matches)
  └─ Cross-dataset tab Type Mapping panel summary (type_mapping_panel._compute)
        ▼
mapping_visualization.py                    flows → network / linker graph / Sankey / CSV
```

Every consumer surface resolves a foreign type through
`mapped_type_targets(mapper, foreign_type, foreign_ds, selected_ds)` — the
union of the stored-mapping/alias resolution and the derivation-bridge
ends. One engine, one number.

## 2. Version control — per-release namespaces

Releases are separate mapping namespaces; nothing resolves through another
release's tables:

| release | mapping namespace | tables |
|---|---|---|
| male-cns v1.0 | `male-cns:v1.0` | the crosswalk df |
| male-cns v0.9 | `male-cns:v0.9` (own native table) | v0.9 table; shared names alias explicitly to v1.0 |
| FAFB v783 | `flywire_FAFB_v783` | `flywire_FAFB_v783_allneurons_neuron_df.csv` |
| BANC v626 | `banc_v626` | `banc_v626_allneurons_neuron_df.csv` |
| BANC v888 | `banc_v888` | `banc_v888_allneurons_neuron_df.csv` |

Both BANC releases sit in `DATASET_PRIORITY` (v888 immediately after
v626), so `detect_type_source` auto-detects a v888-only type name as
`banc_v888` instead of falling through to "unknown" — pinned in
`tests/core/test_dataset_identity_collisions.py`.

Consequences (user directive, 2026-09-07):

- A `banc_v888` selection resolves **only** against the v888 tables —
  v626 names ("via banc v626") and v626 bodyId pools are impossible.
- `male-cns:v0.9` never silently borrows the v1.0 table or body IDs. Shared
  primary names use an explicit `release_alias` hop before downstream v1.0
  mapping; v0.9-only names use only their own native `*Type` fallback.
- BANC v626↔v888 has one narrow `banc_release_crosswalk` type bridge when
  the relation is available, plus exact same-name type identity as a safe
  fallback when it is not. Generic BANC↔BANC annotation/transitive paths
  remain blocked.

The v0.9 release probe records 176,379 common bodyIds, 163,130 same-name
typed rows, 1,354 typed-row disagreements, and 11,597 shared primary names.
`_release_alias_diagnostics["release_alias_disagreement"]` keeps the
actionable source-side subset: 29 shared v0.9 names across 71 joined rows,
with a bounded example list. A disagreement is visible to audit/reporting but
does not rewrite the selected release's bodyId pool.

## 3. The bridge-rule algebra (derivation walk)

The derivation walk (`get_type_bridges`) expands a source type through
licensed evidence edges and records every simple path that lands on a real
primary `type` of the target dataset. Every edge is licensed by
`BRIDGE_SOURCE_MAP` (declarative; verified against the tables at load).

### 3.1 Direct bridge forms (plus same-name identity everywhere)

```
HEMI  --hT--  MCNS
MANC  --mT--  MCNS
MCNS  --fT--aT--  FAFB   |  MCNS --fT-- FAFB  |  MCNS --aT-- FAFB
MCNS  --mct--  BANC
FAFB  --fafb_cell_type--  BANC  |  FAFB --aT--ACT-- BANC  |  FAFB --aT--fafb_cell_type-- BANC  |  FAFB --aT--type-- BANC  |  FAFB --ACT-- BANC
HEMI  --hemibrain_cell_type--  BANC
MANC  --manc_cell_type--  BANC
BANC v626  --banc_release_crosswalk--  BANC v888
MCNS v0.9  --release_alias--  MCNS v1.0  (+ v0.9 native *Type fallback, §2)
```

`hT/mT/fT` = hemibrainType/mancType/flywireType cells on the male-cns rows;
`mct` = BANC `malecns_cell_type`; `aT` = FAFB `additional_type(s)`;
`ACT` = BANC `Alternative Cell Type(s)`. The BANC direct bridges also use
`fafb_cell_type`, `hemibrain_cell_type`, and `manc_cell_type` for their named
target namespaces. MCNS `flywireType` does not land in BANC. In
`FAFB --aT--type-- BANC` the trailing `type` hop is the same-name identity
landing: the shared token is itself a BANC primary, so the aT hop stays
inside FAFB and a `type` hop crosses into BANC (FAFB `DN1pD`
--aT 'SMP537'--> BANC `SMP537`). In `FAFB --ACT-- BANC` the two ACT hops
(landing + alt→primary continuation) collapse to one standardized linker
when the via values coincide — read the selected bridge text, not the
column list, for the hop detail.

(2026-09-27 audit, code-verified) One emitted edge never survives: the
male-cns-side annotation-reverse block also offers `MCNS --ACT-- BANC` —
MCNS type names do appear in BANC `Alternative Cell Type(s)` cells (e.g.
`Acc. ti flexor MN` in the cells of BANC `CB0975`/`CB4187`) — but the
MCNS~BANC registry standard admits only `malecns_cell_type` (§3.2), so the
registry scoping strips every ACT chain for that pair. It is a
licensed-column detour the registry refuses, not a bridge, which is why the
diagram does not draw it.

(2026-09-27 audit, probe-verified) All five FAFB↔BANC forms derive on the
real tables. (1) `fafb_cell_type` — the curated label edge lands directly
on the FAFB primary ONLY when the cell token names it exactly; a
rename-resolved winner (BANC v888 `SMP537`'s fct cell `SMP537` → FAFB
`DN1pD`, or BANC `CB1770` → FAFB `s-CPDN3A`) terminates at the raw-token
name node and the walk continues through FAFB's own
`additional_type(s)` edge — §token chaining (user 2026-09-27): the chain
SHOWS the rename (`fct 'SMP537' → aT → DN1pD`) instead of hiding it in
one hop, and both sides pool honest linker rows (3 SMP537 voters / 4 of
8 DN1pD rows). (2) `aT--ACT` — FAFB `4I1` reaches BANC `FB4F_a` through
the non-primary token `FB4I`. (3) `aT--fafb_cell_type` — the chained
route is now the STANDARD shape for every renamed token, walked from
either side (FAFB `DN1pD` --aT 'SMP537'--> token --fct--> BANC
`SMP537`/`SMP539`; the circadian export renders 36 of 44 pairs this
way). (4) `aT--type` — a token that IS a BANC primary lands by same-name
identity; this form survives only for pairs with no curated bridge of
the same token (**curated subsumption**, user 2026-09-27: a one-linker
aT chain is dropped when an fct chain bridges the same token to the
same target — before chaining+subsumption the linker view drew two
PARALLEL one-linker paths on 36 of 44 circadian pairs; today it draws
one sequential chain). The decision/provenance layer stays
winner-resolved throughout (votes `{DN1pD: 3}`, target DN1pD).

**Usage census (2026-09-27, post chaining+subsumption, endpoint pairs
with surviving chains):** v626 — fct 7,472 (selected 7,472), pure-ACT
2,348 (selected 1,878), aT→ACT 416 (selected 194), **aT→type landing
297 (selected 287)**, aT→fct 114 (selected 114), bare echo 115,
alignment lane 18; v888 — fct 7,484, pure-ACT 2,273 (selected 1,803),
aT→ACT 421 (selected 199), aT→type 297 (selected 287), aT→fct 113,
bare 114, alignment 18. All classes reach the decision layer, pooling
(linker-row or full-population bases per hop home) and the mapping
exports.

**(5) `ACT--ACT` (pure-ACT landing)** — the mirror of the aT lane: a
FAFB primary whose OWN name appears in a BANC primary's
`Alternative Cell Type(s)` cell derives `FAFB --ACT 'P'--> [BANC
token P] --ACT--> BANC primary` (the two ACT hops collapse to one
standardized linker when the via values coincide). Legitimate and
heavily applicable: 7,637 of 7,643 v888 ACT tokens are FAFB primaries
(7,637 for v626), the class is the second-most-selected lane
(1,878/1,803 selections), and the two linkers' identical column makes
it invisible in column-list exports — read the selected bridge text
instead. The mirror aT lane is the same structure: 290 FAFB
`additional_type(s)` tokens are BANC primaries (aT→type→BANC).

### 3.2 Connector licenses (`ROUTE_MIDS`)

A chain visits at most ONE intermediate namespace, and only from the
licensed set:

- male-cns is the controlled connector for the neuprint crosswalk families;
- BANC label bridges and the BANC release relation are direct-only;
- **BANC is never a connector** between unrelated endpoint pairs;
- MCNS↔BANC is `malecns_cell_type`, not an fT/FAFB/ACT detour.

This makes the BANC ban structural (a chain cannot even be built through
it) and enforces the two-linker no-flip rule: a flipped `fT/aT` chain
would have to revisit the source namespace, which the route guard forbids.
All bridges are bidirectional — the reverse travel direction reverses the
hop order (the SAME bridge, not a flip).

### 3.3 Preference order — evidence first, same-name LAST

`_name_neighbors` emits evidence edges (crosswalk, annotation,
reverse-crosswalk, cross-namespace landing) **before** same-name
membership edges, so the walk's visited race records the EVIDENCE chain
for a pair. A same-name pair whose crosswalk cell names itself therefore
derives through the pair's registered metadata linker (for example
`malecns_cell_type '<name>'` for MCNS↔BANC), not the
bare name echo:

```
DN1a[FAFB·type] → DN1a[MCNS·flywireType]      (crosswalk-verified)
```

The bare same-name chain is the LAST choice — offered only when no
evidence edge connects the pair (then the UI still says "no metadata
verification — please double check"). The reverse-crosswalk leg no longer
skips the same-name arrival: the verification edge must be walkable.

### 3.4 Arrival gate for registry-less pairs

The post-arrival annotation continuation is a two-linker REGISTRY-standard
privilege (male-cns↔FAFB etc.: crosswalk primary → its annotated
siblings). Registry-less pairs — every BANC pair — stop at the arrival;
without the gate the walk wandered BANC annotation classes
(`APDN3 → R8_unclear → T1`), landing coarse hub names and inflating a
circadian FAFB→BANC run to 3,262 neurons (post-fix: 21/21 types, 242/242
FAFB neurons → 42 v888 targets, 207 unique at the 2026-09-09 fix;
2026-09-27 refresh against the 2026-09-26 pre-alignment-lane baseline
export: the flow table is IDENTICAL row-for-row (44 rows, 21 source
types, 42 target types, 887/230 neurons, same statuses/relationships)
and only the neuron-level union drifted (200 unique today; 39 targets
under the resolver-union definition vs 42 flow-table target types —
different counting bases, both stable) — pre-existing drift from later
mapper rounds and the refreshed bucket, unchanged by the alignment
fallback lane, verified both by stripping the lane in-memory and by the
baseline export diff).

### 3.5 Data hygiene inside the walk

- Untyped labels (`Unknown`, empty, bare numbers) never become bridge
  nodes or targets (the BANC `Unknown` convention hub stayed out).
- FAFB `additional_type(s)` cells listing several names keep only the
  in-use ones (a FAFB primary, a male-cns type name, or a `flywireType`
  cell value) when the cell names any used name — e.g. `AVLP011` keeps
  `aSP8b` and drops the unused `AVLP012`; fully-unused cells stay
  (indistinguishable). The verified `LMTe01, CL125` cell on APDN3 rows
  survives intact.

### 3.6 Terminal evidence hops and the cross-reference rule (2026-09-09)

Two walk guards close the `l-LNv → BM_*` derivation class (a FAFB
circadian type fanning out to BANC's 1,212-neuron `BM_InOm` and friends
with zero row-level support on the reached types):

- **Terminal evidence hops** — `TERMINAL_LINKER_COLUMNS` (the curated
  BANC label columns, `banc_release_crosswalk`, `release_alias`) carry
  the reached type's own row-level evidence. When such a hop ARRIVES in
  the target namespace, the derivation ends there; the walk never
  continues into that namespace's annotation graph. When the hop lands
  in a licensed intermediate instead (BANC `malecns_cell_type 'MDN'` →
  MCNS `MDN` —flywireType→ FAFB `MDN`), the licensed hub leg still
  continues, and an annotation hop directly after a terminal hop is
  refused (`ann_chained`).
- **Cross-reference rule** — the primary→alt emission skips alt tokens
  that are themselves primaries of the same namespace. A primary's
  annotation cell naming another primary is a cross-reference (BANC's
  `Alternative Cell Type(s)` concatenates the other datasets' curated
  labels), not a rename; hopping to it as a name node only fed the
  name-graph wander. Same-name identity stays handled by the membership
  block, and the designed FAFB `aT` → BANC `ACT` two-linker standard is
  untouched (its tokens are not primaries of the landing namespace).
- **Zero-evidence chains are dropped at pooling** — safety net on top of
  the licensing: `ui.neuron_index.chain_is_supported` refuses a chain
  whose target-home linkers ALL pooled zero rows on the reached type
  (a measured zero). Same-name/crosswalk-arrival chains (full-population
  pools) and unmeasured sides pass; the panel and the viewer log the
  drop instead of rendering an unsupported derivation.
- `get_mapping_origins` reports the one-linker descriptor of a 2-hop
  bridge (e.g. `malecns_cell_type via 'MDN'` for a curated label pair)
  instead of dropping it.

## 4. Production resolution precedence

`_build_type_mappings` + the release/BANC overlays +
`_apply_annotation_bridge_overlay` fill the
stored mappings consumed by `get_mapped_type`:

1. **BANC label/release overlays** — direct curated label votes and the
   root relation fill their release-local slots; conflicts are never
   guessed. The `fafb_alignment_cell_type` fallback lane runs inside the
   same overlay with a keep-rule in `record_votes`: it may fill a slot
   ONLY where the curated `fafb_cell_type` pass left it unresolved
   (curated winner → provenance untouched; curated conflict →
   fail-closed, alignment votes recorded as diagnostics only, never as
   `TypeMappingConflict` records). Its mappings carry the tier
   `direct alignment label` and the kind
   `cross-dataset cell type (alignment)` (2026-09-27; 18 types/release,
   all cross-name, e.g. `AVLP614` → `CB1476`).
2. **Crosswalk routes** (male-cns anchored, per release) — win when
   present; several `flywireType` names per male-cns type become 1-to-N
   conflicts, never guesses.
3. **Transitive mappings** — each release's male-cns anchor gains the
   anchor's other targets.
4. **Annotation-bridge overlay** — same-name identity, then the
   annotation bridge (exactly one candidate maps; several become a
   1-to-N conflict with `origin` provenance). BANC↔BANC annotation paths
   are skipped.
5. Exports state the derivation: `mapping_origin` in
   `auto_type_mapping.csv`, `origin` in the conflicts CSV.

`get_mapping_decision(source_type, source_dataset, target_dataset)` is the
policy gate used by UI consumers. It scopes conflicts to the ordered dataset
pair, keeps BANC label-vote conflicts blocked, and distinguishes an accepted
mapping from `valid_split_evidence` and `evidence_only`. A split can therefore
remain inspectable without being mistaken for one canonical target.

### 4.1 The comparison-layer resolver (`comparison/type_resolver.py`)

`comparison/type_resolver.py` is THE shared backend for analysis code. The
panel's validity policy — status/evidence-aware target resolution plus
canonical profile expansion — lives here, in the comparison layer; no
analysis consumer imports a UI module, and the UI
(`mapped_type_targets`) is a thin adapter over the same core.

- `MapperSnapshot` — one load-state record per analysis run (source table,
  version token from the v1.0 CSV, `load_error`, decision cache). Obtained
  once per run and passed through candidate discovery, query mapping, and
  scoring so every step observes the same mapper state.
- `resolve_valid_targets()` — the typed resolution record
  (`TypeResolution`): status vocabulary `mapped | bridged |
  valid_split_evidence | evidence_only | conflict | claimed | unmapped |
  mapper_unavailable` (`claimed` = a stale pre-2026-09-12 claim), ordered
  target list, equivalence key, bridge
  provenance, and scoped conflicts. `bridged` is the single bridge-derived
  target case emitted by `get_mapping_decision(include_bridges=True)`; it
  keeps its own status and is never silently rewritten to curated `mapped`.
- `expansion_targets()` — the analysis policy: mapped/bridged/splits
  contribute their targets, conflicts, evidence-only relations and
  stale `claimed` entries fail closed (claimed → no expansion),
  unmapped types keep the raw name as the explicitly counted
  long-tail fallback.
- `expand_profile_types()` — status-aware profile canonicalization for
  scoring: each canonical key records WHICH mapping status produced it,
  valid splits distribute weight EVENLY across every licensed target
  (`split_policy='even'`, total mass preserved), conflicts are excluded,
  and unmapped types stay raw with a fallback flag.
- `auto_mapping_result_metadata()` (in `profile_comparator.py`) — the
  standard `auto_type_mapping_*` metadata block saved with results:
  requested vs active, source, version, load error, per-status resolution
  counts, raw-fallback flag, and the mapping-policy version.

**Counting unit:** exported per-status counts report
`mapping_resolution_counts_by_status` on a
`mapping_resolution_counts_basis` of `unique_type_resolutions` — one count
per distinct `(source_dataset, type[, target])` resolver input, deduped
across repeated lookups (path rows, edges, shared query items). A second,
distinctly keyed metric `mapping_partner_type_resolutions_by_status`
reports contributor-type *occurrences* inside canonicalized profiles and is
never conflated with the primary counts. Each surface holds ONE mapper
snapshot per run (reset per search) so load state and decision caches are
shared across expansions.

Consumers: homolog finding (candidate expansion, same-type rescue, the
mapping-aware vector prefilter, type-level same-type marking),
connectivity-profile comparison (query mapping, inter-dataset and
explicit matrices), the cross-dataset HTML report (type-count grouping
and canonical row keys), the mapping flow builders
(`resolve_flow_status`), and the comparison analyzer/parameters type
resolution. This lane is separate from the explicit LabelMapper lane
(`std_label_*`, applied at connection extraction) — see 'The two label
lanes' in docs/AUTO_TYPE_MAPPING.md. The
legacy `get_mapped_type()` / `resolve_type_across_datasets()` remain as
compatibility-only wrappers (None for splits/conflicts, no provenance);
new analysis code must use the resolver.

## 5. Shared backend and surface parity

`mapped_type_targets()` resolves one foreign type as the **union** of
(a) the alias/stored resolution (`get_alias_candidates`: rename, same
name, refused N-to-1 members) and (b) the derivation-bridge ends
(`get_type_bridges`). The core union decision lives in
`comparison.type_resolver.resolve_valid_targets` (§4.1); this UI function
is the response-shape adapter, and both UI surfaces consume it:

- **'See available neurons' viewer** — `enrich_native_type_matches`
  annotates every covered foreign type; the mapped-type view lists the
  selected dataset's neurons of the mapped names.
- **Cross-dataset Type Mapping panel** — `_compute` seeds
  `origin_seeded_flows` with the resolved origins, then derives the
  summary from the same backend.

Parity contract (pinned by
`tests/core/test_type_mapper_real_datasets.py::test_panel_and_viewer_share_mapped_type_backend`):
`circadian_clock` resolves to 21 FAFB types / 242 neurons → 40 male-cns
targets / **219 unique neurons on BOTH surfaces**. The panel no longer
sums per-flow counts — shared targets (`s-LNv`, `5thsLNv_LNd6` hit by two
flows, `SMP227` by three) are counted once; the old Σ-with-multiplicity
reported 243.

## 6. BodyId pooling and the pool fix

`pool_bridge_body_ids` pools, per standardized linker, the bodyIds of the
home side's endpoint rows whose linker column carries the linker value;
an unconstrained side pools the FULL endpoint type (crosswalk-arrival and
bare same-name chains name that side's identity without a per-side cell).
For BANC's curated label bridges, `fafb_match` and `malecns_match` are
optional provenance diagnostics only. A locally observed match may be
counted as verified or flagged as a conflict, but it never filters a
type-label bridge and never joins bodyIds across datasets. The releases
also publish `manc_match`, `hemibrain_match`, `fanc_match`, and five
`*_nblast_match` columns; the mapper never reads them —
`_banc_label_match_column` wires only `fafb_match` and `malecns_match`,
the only locally verifiable optional match columns (2026-09-27 audit).
Each endpoint therefore reports its own coverage pool, even when one type covers only a
subset of the other type's population. Multiple linkers union independent
home-side candidates. (2026-09-27 audit) `mancBodyid` is published on the
male-cns crosswalk rows but is unused: no mapper path — and no current code
at all — reads it, nor the neighboring `mancGroup`/`mancSerial` columns.
The MANC↔MCNS mapping is name-only end to end: `mancType` crosswalk edges
for derivation, per-side pool coverage for bodies, and no bodyId join or
match-column verification in either direction.
The virtual `banc_release_crosswalk` linker is the corresponding exception
for BANC v626↔v888: it reads the complete `root_626`↔`root_888` relation,
filters both sides by their requested type, and preserves repeated roots.
For example, the real `L5` bridge reaches 1,655 of 1,683 BANC v888 rows
while retaining 1,651 v626 roots.

Pool dictionaries used by the Type Mapping panel are keyed by
`(source_dataset, target_dataset, source_type, foreign_type)`. This keeps
same-named types and opposite directions independent; renderers retain a
two-part-key fallback for older programmatic callers and tests.

Pool fix: the crosswalk-arrival linker's value is the **cell content**
(`via`) — the reverse leg's hop value is the arrival (male-cns) type
name while the cell carries the foreign token (MCNS rows typed
`5thsLNv_LNd6` carry `flywireType` cells `s-LNv_a,LNd_a`). Matching the
arrival name emptied the target pool and blanked the coverage of rows
like `5th-LNv → 5thsLNv_LNd6`. Post-fix all 44 circadian pairs pool.

`granularity` ("n to m") remains a compatibility summary, while the pool
now carries independent `source_coverage` and `target_coverage` strings
(`covered <pool> of <endpoint type total> (<pct>)` — the shared
`format_coverage` formatter with thousands separators and a one-decimal
share). The historical `coverage` field remains the target-side alias for
callers that still read it, and an unmeasurable side yields `""`.

Per-side pool STATE is explicit since 2026-09-09: `source_basis` /
`target_basis` distinguish `linker rows` (a linker measured a subset),
`full population` (the unconstrained fallback), `release relation
participants`, and `unmeasured` (the side's coverage index was
unavailable — never rendered as a measured zero). The same numbers are
machine-readable as `source_pool_size` / `source_type_total` /
`target_pool_size` / `target_type_total` (totals `None` when
unmeasured), and per-linker `body_ids` are order-deduplicated so card
counts equal pool counts. Renderers go through `format_pool_side`, which
maps the four states to `FB: 1,655 of 1,683 (98.3%)` / `FB: all 1,683` /
`FB: not measured` / `not pooled` so a subset, a tautological full
population, and an unmeasurable side can never look alike. Panel text
and CSV output make clear that these are coverage counts, not
bodyId-to-bodyId pairings. The pool also carries `source_type_body_ids`
/ `target_type_body_ids` — the FULL population of each mapped type in
its own dataset (sorted, independent of the linker-filtered subsets),
exported as the mapping CSV's `source_body_ids` / `target_body_ids`
columns; listed per type per side, never paired across datasets.

### 6.1 Prioritized bridge resolution and coverage scopes

`prioritized_bridge_chains(chains, source_dataset, target_dataset)` provides a
deterministic evidence order without consulting body counts. Direct curated
labels and direct crosswalk evidence outrank the alignment fallback label
(`direct alignment label`, 2026-09-27), which outranks direct auto labels,
annotations, release metadata, indirect routes, and finally a bare
same-name chain. Ties
use the number of direct/indirect linkers, chain length, and hop values. The
older `preferred_bridge_chain()` API is now a compatibility view of the first
ordered candidate.

`resolve_prioritized_bridge_pool()` is the shared bodyId-boundary resolver used
by the Type Mapping panel and mapped viewer. It:

- filters candidates to the requested endpoint type and attempts them in the
  priority order, within an explicit alternative budget;
- selects the first supported chain for legacy edge weights and records every
  unsupported attempt, fallback reason, linker evidence tier, raw token, and
  canonical token;
- retains later supported chains as alternatives and unions their independent
  source/target bodyId pools into `all_valid_*` fields, including overlap IDs
  and per-linker unions; and
- checks source-home and target-home linkers separately while never pairing
  bodyIds across datasets or using coverage to resolve a type conflict.

The selected pool is the stable primary display/weight scope. The all-valid
union is the complete supported-evidence scope. For a fan-out, branch rows are
non-exclusive: an aggregate may report `5/6` selected MCNS bodies and `6/6`
all-valid bodies while the per-branch evidence is `2 + 3 + 3 = 8`. The
coverage layer reports both scopes and overlap rather than presenting eight as
eight distinct source neurons.

## 7. Visualization contract

- **Native source labels in transformed query overlays**: auto type mapping
  uses crosswalk columns such as `flywireType` to resolve/search comparable
  types, but it does not overwrite the source row's primary `type` for the
  query morphology. A cross-dataset `query_transformed_*` overlay therefore
  keeps a source label such as MCNS `SMP227` even when its mapped
  `flywireType` cell contains `CB1449,CB2843`.
- **Tree bodyId presentation**: interactive `legend_mode='tree'` marks
  custom-layer groups explicitly. A custom group with one unique neuron item
  is rendered as a direct bodyId-level row with count `1` and no redundant
  child leaf. Counts are based on unique neuron items, so a skeleton trace
  plus a companion soma mesh still represents one neuron; multi-neuron groups
  retain their expandable rows.
- **Panel result presentation** (user 2026-09-07, refreshed 2026-09-09):
  the Type Mapping panel's results are presentation-refined around the
  FAFB `APDN3` ↔ male-cns example.  `build_type_coverage` derives two
  views over the mapped pairs, rendered as a TOP-LEVEL "Type coverage"
  expansion PER dataset pair — placed directly above its pair card and
  named with the pair (`Type coverage — <source> → <target>
  (bidirectional, 1-to-N / N-to-1)`), scoped to
  that pair's flows: the FORWARD view (one row per queried type: its
  neuron count, the target types, `1-to-N`/`1-to-1`, the queried type's
  TOTAL mapped number and the target-side total as `x of y (pct)`
  bodyIds) and the BACKWARD view (one row per receiving type: the source
  types mapping onto it, `N-to-1` when several converge — three FAFB
  circadian types onto male-cns `SMP227` — with both sides' coverage).
  Both tables name their coverage columns by DATASET (`<dataset> side
  (bodyIds)`; user 2026-09-09: "source/target side" read as flipped in
  the backward view, so the dataset name replaces the side words).  A
  side's
  denominator is the population of the endpoint types involved (types
  are disjoint body sets, so a 1-to-N row's target total is the summed
  population of all mapped target types); each table exposes selected-bridge
  and all-valid-union coverage. Fan-out branch values are marked
  non-exclusive and overlap counts are shown. A side whose pools were
  unmeasurable renders `not measured`, never a fake `0 of n`. The
  per-pair cards label both sides ("12 FAFB → 4 MCNS"), annotate every
  linker in Map used with its own pooled bodyId count (pooling now
  covers both rendered chains), and show per-side BASIS-AWARE pool
  coverage (`FB: 1,655 of 1,683 (98.3%)` / `FB: all 1,683` /
  `FB: not measured` / `not pooled`); long cells wrap within capped
  column widths so every column stays visible.  The summary strip
  splits received vs issued mapped neurons.
- **Mapping graph plots types only** (re-cut 2026-09-26): the downloaded
  Mapping graph and Mapping sankey draw type nodes and nothing else. A
  taxonomy/metadata chip such as FAFB `cell_type · 'circadian_clock'` used to
  be rendered as an `E|<origin_dataset>|<label>` hub linked to its 21 FAFB
  source types, which read as one more mapping endpoint; it is now search
  provenance only, carried by the flow metadata (`origin_dataset`,
  `origin_column`, `origin_value`, `origin_type`, plus the legacy
  `matched_origin` display field) and by the panel tables and CSV columns.
- **Centre-source star** (`_star_component_order`, `_origin_target_shape`):
  when one dataset is the origin and exactly two receive it, the composed
  network renders from explicit preset positions as
  `target 1 | origin | target 2` — the origin keeps role `source` in the middle
  column (layer position must not demote it), and the flank covering more
  origin types goes left, ties broken by the selection order then the dataset
  key. Dagre ranks by edge direction and cannot produce that shape, so
  `render_composed_mapping_html` switches to the preset layout whenever every
  component is a star. The composed Sankey follows the same columns; because
  Plotly ranks Sankey nodes topologically and ignores `node.x` when a link
  contradicts it (measured), the left half's rows are emitted
  `target → origin`, and the artifact states that convention in a floating
  note. One origin into one target is a plain two-column Sankey; more than two
  targets is not drawn (the per-pair `Sankey (type-level)` buttons cover it).
- **Per-pair network orientation** (§14, user 2026-09-09): the
  per-pair `Network (type-level)` export still presents the ORIGIN side in
  layer 0 — a native-match flow (the viewer's expanded search, whose matched
  column lives in the foreign dataset) is flipped so the view reads
  origin types → searched types like the panel's seeded exports. Only the
  entry hub is gone; the orientation rule that was introduced to rank it
  survives because it is what makes the two exports read alike.
- **Mapping exports count neurons** (user 2026-09-26): vispath's Sankey and
  network templates hard-coded the pathway vocabulary, so a mapping Sankey
  hovered "Synapses: 9" and its metric selector read "Synapse Count" even
  though DROCAT already passed `edge_weight_label='neurons'`. The templates now
  honour `metric_option_label`, `sankey_title` and `sankey_label_layers`; the
  mapping views pass `Neuron count`, a `Type mapping Sankey — …` title, and no
  `(L<n>)` hop suffix (whose columns are datasets, not pathway depth). The
  pathway defaults are unchanged, so connectome artifacts render byte-for-byte
  as before.
- **Dataset-wide backward coverage** (§12, user 2026-09-09): the panel's
  backward table upgrades receiving-type rows to the dataset-wide incoming
  scope. `CrossDatasetTypeMapper.incoming_type_names()` discovers the full
  incoming family for one receiving type by walking the name graph
  backwards (the neighbor relation is stored both ways) and verifying every
  candidate with the real forward `get_type_bridges` — exact for the
  audited families, ~0.1 s per receiving type where a full forward sweep
  costs minutes. `build_reverse_type_contexts()` materializes each
  incoming pair's prioritized bridge pool and accumulates the per-receiving
  unions; `build_type_coverage(..., reverse_contexts=...)` merges them so a
  backward row lists the full family (active query marked), labels the
  relationship from the row subject's fan-out (`1-to-N`; a multi-source
  row never reads `1-to-1` — §12.1 user decision), measures coverage with
  the incoming population union as the source denominator, and keeps the
  query-scoped slice on `query_scope_*` fields plus
  `coverage_scope`/`incoming_source_count`/`active_query_sources`/
  `truncated` metadata. For MCNS `SMP227` the incoming families are
  4/4/6 types with source unions 22/26, 23/26 · 19/28, 25/28 · 25/36,
  33/36 selected/all-valid — the context that explains the 6 → 94 forward
  fan-out.
- **Coverage-table layout** (§12.5, user 2026-09-09): the four trailing
  coverage columns shrink to 135/145 px minimums with short
  `CODE · selected` / `CODE · all valid` headers; the full dataset key and
  the selected/all-valid scope explanation ride on each header's tooltip
  (a Quasar `header-cell` slot rendering each column's `tooltip` key), and
  the reclaimed width goes to `Maps to` / `Mapped from` and
  `Coverage interpretation` (480 px caps).
- **One shared per-pair weight** (`pair_flow_weight`, user 2026-09-07):
  the Sankey ribbon, the network pair edge, the linker-path edges and
  the mapping-graph edges all draw the SAME number for a mapped pair —
  per side the pooled bodyId count when pooled, else that side's neuron
  count, collapsed by `min`.  Before, the network duplicated the SOURCE
  type's whole count onto every edge (all edges of a 12-neuron type
  showed "12"), the linker graph fell back foreign-first, and only the
  pooled Sankey agreed — the same pair showed a different number in
  each artifact.
- **Pool hover counts are the UNION across pairs** (user 2026-09-07):
  `_endpoint_pool_counts` uniques the pooled bodyIds per type; an N-to-1
  target pools a different disjoint subset per counterpart (male-cns
  CL125/SLP249/PLP080/SLP250 pool 4/4/2/2 of FAFB APDN3's 12 bodyIds),
  and the old max-across-pairs hovered "pool 4 bodyIds" next to
  "12 neurons" — the union reports the 12 the mapping actually reaches.
- **Sankey edge keys are layer-less** (user 2026-09-07): one pair's two
  derivation chains (direct annotation bridge + crosswalk chain) reach
  the shared band at different hop depths; keyed by layer, `create_sankey`
  drew the shared node pair as PARALLEL ribbons with the weight counted
  twice (PLP080 → APDN3 twice).  Keyed by (source, target), same-pair
  links merge with max — one biological connection, one ribbon.
- **Adjustable edge-label size** (user 2026-09-07): the on-edge weight
  labels were pinned to 9px; the network control panel now carries an
  "Edge Label Size" spinner (`edge_label_font_size`, default 9) wired
  through the undo history, independent of the node-label Font Size.
- **Linker network** (`build_bridge_linker_graph`): one column per
  bridge linker COLUMN, ordered by traversal SIDE (user 2026-09-27 —
  source-home columns first, hub columns between, target-home columns
  last; first appearance orders columns within a side, and the registry
  order only places columns the chains did not) so every sequential
  two-linker bridge wires left-to-right like the MCNS↔FAFB standard
  (`type | flywireType | additional_type(s) | type`; the chained
  FAFB→BANC routes render `type | additional_type(s) | fafb_cell_type |
  type`).
  Layering by per-chain hop order (the old behavior) let a 1-linker
  chain's `additional_type(s)` node share the `flywireType` column and
  let a shared linker node's position be overwritten by the last chain;
  pure registry pre-order (the interim rule) wired the FAFB→BANC
  chained paths backwards (36 right-to-left edges on the circadian
  export) — the side rule removes both classes.
- **Unified header legends**: dataset chips render `CODE: full (count)`
  (`dataset_legend_meta` carries the node count and optional swatch
  override); the dynamic group chip for a code with a static chip is
  skipped (the duplicated `FAFB (65)` / `FAFB: flywire_FAFB_v783` header
  rows are gone); linker columns join the same header row with their
  `LINKER_COLORS` swatch. The Sankey note uses the same chip shape.
- **Export mapping CSV** (user 2026-09-09, replaces the old "Export
  bridges (CSV)"): buttons `Export mapping` (per pair) and
  `Export mapping — all pairs (CSV)`; filenames `mapping_*.csv`.
  `mapping_origin` uses the unified mapper-provenance vocabulary
  (2026-09-27 — `same name` / `same name+evidence` / `cross-dataset
  cell type` / `cross-dataset cell type (alignment)` / `crosswalk` /
  `annotation bridge via <tokens>` / `release relation` / `release
  alias`, derived from the selected chain's linkers by
  `_pair_mapping_origin`; previously the column only ever read
  `mapped`/`same name`).  One
  FIXED column set for every pair — 37 columns, verified against the real
  export: `source_dataset, source_entry, matched_column, source_type,
  target_dataset, target_type, relationship, source_neurons, target_neurons,
  selected_bridge, bridge, bridge_columns, mapping_origin, mapping_status,
  selected_bridge_rank, valid_bridge_count, selected_linker_values,
  selected_linker_canonical_values, unsupported_attempts, source_pool,
  source_total, target_pool, target_total, all_valid_source_pool,
  all_valid_source_total, all_valid_target_pool, all_valid_target_total,
  source_body_ids, target_body_ids, all_valid_source_body_ids,
  all_valid_target_body_ids, pool_coverage, pool_coverage_basis,
  coverage_overlap, coverage_scope, target_out_map,
  target_out_map_body_ids` — so the all-pairs file is a plain
  header + rows concatenation (the old per-pair pivoted
  `bridge-<column>` fields needed a union-of-columns hack to avoid
  ragged rows).  `source_entry` is the value that matched on the source
  side (the type itself when matched via `type`, otherwise the label
  value); `relationship` is derived 1-to-1/1-to-N per source type; the
  numeric pool/total columns make coverage machine-readable while
  `pool_coverage` stays the human-readable `source covered …; target
  …` field.  The `selected_*` / `all_valid_*` pair of scopes is the
  machine-readable form of the same split the panel's coverage columns
  show: the primary bridge chain versus the union of every supported
  chain, with `coverage_overlap` quantifying how much the two share and
  `coverage_scope` naming which basis a cell was measured on.  `source_body_ids` / `target_body_ids` carry the FULL
  per-type populations (every bodyId of the mapped type in its own
  dataset, one brace-wrapped comma-separated list quoted as one CSV field
  per cell — user 2026-09-09, after ';'-joined lists made spreadsheet
  delimiter sniffing split them into pseudo-columns): listed per type per
  side, never a bodyId-to-bodyId
  pairing.  The export never represents bodyId
  pairings. The UI requests the `extended=True` export, which adds the
  selected bridge, mapping status, selected rank, valid-chain count,
  raw/canonical selected linker values, unsupported attempts,
  selected/all-valid pool totals and IDs, coverage overlap, and scope. The
  default 20-column form remains available for existing programmatic callers.
- **Bridge-support pass-through** (mapper boundary, user 2026-09-14):
  the row-based bodyId-level evidence each bridge resolves — BANC label
  votes per source type with curated vs `auto:`-stripped provenance,
  linker value, release root-id pairs — is retained in
  `_bridge_provenance` and now **passed** via
  `get_mapping_support(source_type, source_dataset, target_type,
  target_dataset)`, a `support` key on `get_mapping_decision` (per
  branch for splits), a trailing **`mapping_support`** column in the
  compact `auto_type_mapping.csv` (e.g. `cross-dataset cell type:
  5thsLNv_LNd6=1 (auto 1)` vs `…=2 (auto 2)`), and `support_votes` /
  `support_verified` / `support_auto` / `support_linker_value` columns
  on `auto_type_mapping_per_bridge.csv`.  Broad-vote records (some BANC
  types carry votes from dozens of candidate source types) list the top
  3 candidates by count then `+N more candidates` in the compact
  column; the per-bridge columns keep the full counts.  **Direct
  same-name pairs**
  return an identity record instead — `same name (all bodyIds pooled)`:
  the nomenclature identity is the pair's bodyId-level handling (full
  populations, fully in-map).  **Exception**: a same-name type inside a
  1-to-N/N-to-1 structure (e.g. MCNS `CB2572` → FAFB {`CB2572`,
  `CB2572a`, `CB2572b`}) carries no identity marker — the evidence
  bridge resolves the bodyId-level resolution per branch.  Same-name
  records whose populations differ by ≥10× (ratio < 0.1, from the lazy
  per-dataset `type`-column counts in `_type_population_counts`) carry a
  `population_asymmetry` block — surfaced in `mapping_support` and the
  viewer's same-name candidates as a suggested check for the user
  (full-dataset audit: `local_data/same-name-fidelity-audit.csv`; 118
  flagged pairs of 18,797 — plan-same-name-fidelity-and-three-level-
  coverage.md §7).  The mapper
  never consumes this
  evidence to gate, rank, or verify a mapping — verification is the
  validate-expand-visualize pipeline's role; a tight-mapping filter is
  consumer-side over these surfaces.  Implementation hazard recorded:
  the per-row support is written into the row dicts BEFORE
  `pd.DataFrame`/`sort_values` — assigning a list after sorting would
  scramble supports across reordered rows.
- **~~bodyId-level split columns + companion export~~ REMOVED**
  (added 2026-09-14 per bodyId-plan §3b.1; **removed the same day** by
  the final mapper boundary — connectivity split resolution left the
  type mapper entirely, so `build_bridges_csv` no longer accepts
  `splits` and the `split_*` columns / `mapping_bodyid_assignments_*.csv`
  companion export are gone; the legacy AND extended column orders are
  back to the pre-split contracts).  The panel's bodyId-level export is
  now **row-based**: the pair card's **Export branch bodyIds** button
  downloads `mapping_branch_bodyids_*.csv` — one row per mapped pair
  branch with the bridge-resolved pools (per-side basis
  `linker rows` / `full population`, sizes, and the pool bodyIds
  `source_body_ids` / `target_body_ids` — the CLAIM, not the type's
  population), plus the per-branch out-map
  (`target_out_map` / `target_out_map_body_ids`: the endpoint type's own
  neurons this branch's pool does not reach, so a `full population` basis
  reads 0).  This covers every
  multi-branch group, including `evidence_only` N-to-1 pairs (e.g. FAFB
  `s-CPDN3D` → 6 MCNS branches) that the connectivity-based export
  could never serve.
- **One integrated backend** (Revision 3 of the same plan; RE-SCOPED
  2026-09-14): all
  bodyId-level scoring/primitives moved verbatim to
  `comparison/body_id_resolver.py` (`expanded_vector`,
  `score_one_candidate_fast`, `_SideStats`, `scan_source`,
  `build_target_vectors`, `prep_target_stats`, quality gate,
  caliber/hemisphere loaders); `mapping_validation` imports them back
  (single implementation, no parallel path) and `MappingValidator` runs
  on a `BodyIdResolver` with its benchmark profiler injected.  The
  pool-scoped connectivity API (`mapper.body_id_resolver.assign_bodyids`,
  `derive_split_groups`, `resolve_members_across_datasets`) builds
  on-demand per-neuron profiles ONLY for resolved pool members — no bulk
  dataset profiling, no global scans.  **Boundary: this backend is TM VEV
  verification machinery** — it backs the validation pipeline and the
  panel's informational split view, and is never consumed by the mapper's
  mapping decisions; the mapper's granularity surfaces are row-based
  (bridge pools, `get_mapping_support`, `mapping_support`).

## 8. Testing matrix

| suite | pins |
|---|---|
| `tests/core/test_body_id_resolver.py` | the bodyId-level backend: moved-primitive parity + shim identity, clean 1-to-N split, tie → `low_confidence`, float64-range bodyIds, missing/empty profile flags, explicit pools, `ProfilesUnavailable`, side `require/prefer/off` + `side_unknown`/`side_fallback`, lazy mapper property, `derive_split_groups` (real MCNS→FAFB / MCNS→BANC conflicts), bridge-support accessor (aMe24 1 auto vote vs s-LNv_a 2), decision `support` per branch, `mapping_support` export column, same-name pooled-identity record + the 1-to-N exception (CB2572) |
| `tests/ui/test_type_mapping_panel.py` (row-based) | per-type breakdown gating (single- vs multi-type previews), orphans/renames in the summary, entrance + history behavior in the matrix row below |
| `tests/core/test_type_mapper_source_map.py` | declarative licensing vs the tables, per-pair sweeps |
| `tests/core/test_type_mapper_bridge_rules.py` | the algebra: reverse crosswalk legs, connector licenses, BANC ban, no-flip order, untyped exclusion, label-hop terminality + primary-valued-alt refusal (the `l-LNv → BM_*` regression), the designed `aT`→`ACT` standard, real-data acceptance |
| `tests/core/test_type_mapper_annotation_bridge.py` | overlay precedence, exports, release-name resolution |
| `tests/core/test_type_mapper_real_datasets.py` | circadian parity (panel == viewer, 219 unique), linker layout + header legend chips, direct BANC label routes, normalized-auto MeVPLo2 pools, SMP227 selected/all-valid coverage and overlap, CB1011 conflict blocking, two-linker cap, APDN3 pair weights == Sankey ribbons, APDN3 pool-union hover, Sankey no parallel links, edge-label size control, per-pair network types-only (no circadian query-entry node, FAFB still layer 0 — §14 re-cut 2026-09-26), SMP227 reverse-context incoming families and dataset-wide unions (§12) |
| `tests/core/test_banc_release_and_mcns_version.py` | BANC label votes/verification, `auto:`-stripped label provenance, duplicated root relation, MCNS v0.9 alias/native fallback |
| `tests/core/test_dataset_release_registry.py` | shared recommendation policy and unavailable-release behavior |
| `tests/ui/test_dataset_release_notice.py` | explicit single/multi selector recommendation action and suppression |
| `tests/core/test_type_mapping_composed.py` | mapping-CSV fixed-width contract (`source_dataset`…`pool_coverage_basis`), extended selected/all-valid scope and raw/canonical linker export, `format_coverage` states, `not measured` coverage rows, shared `pair_flow_weight` formula, pool-count union, forward 1-to-N + reverse row-subject `1-to-N` fan-out labeling with dataset-wide incoming contexts (§12), types-only canvases (no query-entry node in the composed graph or either per-pair network, origin side still left), the centre-source star (columns, `source` role kept in the middle, flank by coverage then selection order, chain-is-not-a-star), and the composed Sankey (reversed left half, column list, slipped-type report, ≤2-target gate) |
| `tests/core/test_mapping_export_wording.py` | mapping Sankey/network exports never speak pathway: no 'Synapse'/'Synapse Count', no `(L0)` hop suffix, a mapping-specific title on canvas and in `<title>`, the layout dropdown pre-selects the rendered layout, and the vispath defaults keep a PATHWAY artifact byte-identical |
| `tests/ui/test_type_mapping_panel.py` | entrance enable/disable, global search composition, short coverage headers with dataset-key + selected/all-valid tooltips, dataset-wide backward scope rows, per-pair artifact actions, the suspects badge hover on all three tables (slot bound + per-rival text, collapsed block still rendered), the composed Sankey button beside the graph button, panel-scoped history |
| `tests/ui/test_alias_matches.py` | viewer enrichment, source-scoped conflict rendering, mapped-type view, pool granularity |
| `tests/ui/test_neuron_index_viewer.py` | independent endpoint pools, two-sided support semantics, prioritized fallback after an unsupported chain |

Probes under `local_data/`: `repro_two_flows.py` (surface parity),
`probe_t2_t3.py` (version control + used-name filter),
`probe_circadian_banc888.py` (FAFB→BANC v888 coverage),
`bridge_rules_probe.py` (the algebra sweep).

## 9. Known limits

- `optic-lobe` and `manc:v1.2.3` are not in the male-cns crosswalk table
  and therefore have no verified mappings (same-name only). `manc:v1.2.3`
  additionally carries a latent `BRIDGE_STANDARD` registry row — the
  namespace is in neither `DATASET_PRIORITY` nor the source map's
  `mancType` landings, so that row can never be exercised (2026-09-27
  audit).
- BANC `fafb_alignment_cell_type` is licensed as the fallback-only FAFB
  label lane (2026-09-27 — see §4; never overrides a curated winner and
  never fills a curated conflict), while `fanc_cell_type` remains
  deliberately unlicensed. `banc_public_data`'s `_ALT_TYPE_COLUMNS`
  continues to fold alignment values into the regenerated
  `Alternative Cell Type(s)` passthrough — coexistence is intended: the
  converter pipeline and the mapper lane read the same source column
  independently.
- MCNS v0.9 is intentionally name-aliased to v1.0 only for shared primary
  names. Its body IDs remain native, and v0.9-only fallback labels are lower
  tier than the certified v1.0 crosswalk.
- Multi-candidate evidence is never guessed: 1-to-N conflicts are
  exported for manual adjudication.
