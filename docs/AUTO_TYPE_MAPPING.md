# Auto Type Mapping for Cross-Dataset Comparison

## Overview

When comparing neuron connectivity profiles across different *Drosophila* connectome datasets (e.g., hemibrain vs male-cns, FAFB vs BANC), a fundamental challenge arises: **the same biological neuron types may have different names in different datasets**. 

For example:
- `lLN7` (hemibrain) = `ALIN4` (male-cns, flywire)
- `DNp01` (male-cns) = `DNp01` (flywire) = `DNp01` (hemibrain) ← same name, no issue
- `MTe07` (flywire) = `MeVPLo2` (male-cns)

The **Auto Type Mapping** feature automatically standardizes neuron type names across datasets, enabling meaningful cross-dataset comparisons of connectivity profiles.

## How It Works

### 1. Canonical Type Names

We use **male-cns** as the canonical reference dataset because:
- It represents a complete CNS reconstruction (unlike hemibrain's partial brain)
- It includes the most up-to-date neuron type annotations
- It has extensive type metadata including mappings to other datasets

The `CrossDatasetTypeMapper` class builds mappings from the male-cns `neuron_df`, which contains columns linking each neuron's type across datasets.

### 2. Type Mapping Source

The mapper uses the male-cns neuron DataFrame located at:
`datasets/male-cns_v1_0/male-cns_v1_0_allneurons_neuron_df.csv`

Key columns used:
- `type`: male-cns type name
- `flywireType`: corresponding FAFB v783 type (it is not a BANC bridge)
- `hemibrainType`: corresponding type in hemibrain
- `mancType`: corresponding type in MANC

These crosswalk columns may list several names in one cell, separated by
`,`; each name is used individually (a male-cns type pointing at several
flywire types becomes a 1-to-N conflict instead of a bogus joined name).

#### Mapping labels versus display labels

Crosswalk columns are used to resolve equivalent type names for search and
connectivity comparison. They do not replace the source dataset's native
`type` label in a transformed query visualization. For example, an MCNS row
with `type = SMP227` and `flywireType = CB1449,CB2843` is mapped through the
`flywireType` value for cross-dataset analysis, but the query overlay in a
target FAFB scene remains labeled `SMP227`.

#### Renamed FlyWire types (additional Type(S) columns)

FAFB and BANC releases publish release-specific additional-type columns that
record type renames between releases:

- FAFB v783: `additional_type(s)` in
  `datasets/flywire_FAFB_v783/flywire_FAFB_v783_allneurons_neuron_df.csv`
- BANC v626: `Alternative Cell Type(s)` in
  `datasets/banc_v626/banc_v626_allneurons_neuron_df.csv`
- BANC v888: `Alternative Cell Type(s)` in
  `datasets/banc_v888/banc_v888_allneurons_neuron_df.csv`

When a male-cns `flywireType` value is **no longer a primary `type`** in the
target dataset but appears in that column, the mapping resolves to the
current primary name. Example: male-cns `SLP249` has `flywireType = SLP249`,
but in FAFB v783 those neurons are typed `APDN3` and `SLP249` survives only
in `additional_type(s)` — so the auto mapping resolves
`male-cns SLP249 → FAFB APDN3`.

Resolution rules:

- The old name resolves to the **single** primary type listing it (rename).
- If **several** primary types list the old name (a split, e.g. FAFB
  `AOTU008` → `AOTU008a/b/c/d`), no automatic mapping is made and a
  `TypeMappingConflict` is recorded instead.
- Names that are already a primary type pass through unchanged.
- FAFB and BANC resolve independently: a name FAFB renamed may still be
  primary in BANC (e.g. `MDN` → `DNp50` in FAFB, still `MDN` in BANC), so
  each keeps its own mapping namespace.
- Missing dataset tables only disable the rename resolution; the mapper
  still works from the male-cns crosswalk alone.

#### Curated BANC label bridges

BANC v626 and v888 publish release-local, per-dataset labels. The mapper
reads these columns directly and treats them as the authoritative direct
bridge for the corresponding namespace:

| BANC column | target namespace | verification |
|---|---|---|
| `fafb_cell_type` | FAFB v783 | `fafb_match` when the FAFB table is available |
| `fafb_alignment_cell_type` | FAFB v783 (fallback-only lane) | `fafb_match` diagnostics |
| `malecns_cell_type` | male-cns v1.0 | `malecns_match` when the MCNS table is available |
| `hemibrain_cell_type` | hemibrain v1.2.1 | curated label or known normalized `auto:` label |
| `manc_cell_type` | MANC v1.0/v1.2.1 | curated label or known normalized `auto:` label |

**The alignment fallback lane (2026-09-27).** `fafb_alignment_cell_type`
is a second BANC→FAFB label lane with weaker per-row quality — its labels
agree with the `fafb_match` bodyIds on 96.5% of verifiable rows versus
99.2% for the curated column (whole-release measurement), and it anchors
on the BANC type's own name in the disagreement cases (the ORN glomerular
swaps). It is therefore **fallback-only**: a mapping is created ONLY for
BANC types where the curated `fafb_cell_type` pass yields no winner — a
curated winner keeps its provenance untouched, and a curated conflict
stays fail-closed (the lane's votes are recorded as diagnostics beside
the conflict, never as new conflict records). On the 2026-09-27 tables
the lane fills 18 types per release, every one a genuinely cross-name
mapping (e.g. BANC `AVLP614` → FAFB `CB1476`); ~402 types per release
with conflicting curated votes are correctly left unfilled. Its mappings
carry the evidence tier `direct alignment label` (below crosswalk, above
auto) and the provenance kind `cross-dataset cell type (alignment)`, and
runs surface them in the `[BANC alignment fallback]` block of
`user_warning_notes.txt` with vote and `fafb_match` verification counts.

The releases also publish `manc_match`, `hemibrain_match`, `fanc_match`,
and five `*_nblast_match` columns; the mapper never reads them — only
`fafb_match` and `malecns_match` are wired, the only locally verifiable
optional match columns.

The labels are voted per BANC primary type. A single candidate, or a
candidate with more than half of the votes and at least twice the runner-up,
wins; unresolved splits become a `TypeMappingConflict`. A known match-bodyId
whose target `type` agrees or disagrees with a FAFB/MCNS label never changes
the vote itself: every row's candidate keeps its vote, and the observation
is recorded separately as `verified_votes` (match-bodyId found among the
row's candidates) or `verification_conflicts` (match-bodyId points at a
different type). Those tallies are diagnostics for confidence review — they
do not add or remove votes in the final decision. The
`auto:` prefix is provenance, not a separate type namespace: a normalized
`auto:<name>` token is eligible only when `<name>` resolves to a known target
type. The raw token and its auto evidence tier are retained in bridge
provenance and exports. Unknown `auto:` tokens, `Unknown`, empty labels, and
bare numeric sentinels remain ineligible. Conflicting votes remain rejected
even when one conflicting row uses a known `auto:` token. These bridges are
direct-only: BANC is not introduced as a connector between unrelated endpoint
pairs. A label hop is also a
**derivation endpoint**: once it lands in the target namespace, the chain
ends there — the label cell names the reached type exactly, and continuing
into that namespace's annotation graph would only drift onto unrelated
primaries (see the cross-reference rule below).

`fafb_alignment_cell_type` is licensed only as the fallback label lane
described above — it never overrides a curated label and never fills a
curated conflict. `fanc_cell_type` is intentionally unlicensed
until a FANC namespace and evidence policy are added.

`Alternative Cell Type(s)` remains an intra-BANC annotation column in the
source map — never a substitute for the curated per-dataset label columns;
its cross-dataset reach comes from the mirror landing — a FAFB
primary whose own name appears in a BANC primary's ACT cell derives
`FAFB --ACT--> BANC` chains (7,637 of 7,643 v888 ACT tokens are FAFB
primaries; after the 2026-09-30 post-arrival discipline the lane derives
1,476 FAFB→BANC endpoint pairs on v626 and 1,478 on v888 — mirror
landings, `aT`→`ACT` two-linker bridges, and the backward-reciprocal hub
pairs — the second-most-selected lane after the curated labels).

#### The FAFB ↔ BANC annotation bridge (additional Type(S) ⇄ Alternative Cell Type(s))

The two FlyWire annotation columns also work as a **direct bridge between
the FAFB and BANC namespaces**, without routing through the male-cns
crosswalk. A FAFB `additional_type(s)` token that also appears in a BANC
`Alternative Cell Type(s)` cell connects the types on both sides:

```
FAFB type --(additional_type(s))--> shared token --(Alternative Cell Type(s))--> BANC type
```

Example: FAFB `s-CPDN3A` rows annotate `CB1770`/`CB1791`/`SMP229`, which
are BANC primary types — the bridge derives all three candidates even
though no crosswalk row connects them. (2026-09-27 audit, code-verified)
Because those tokens are themselves BANC primaries, they land by the
same-name identity hop: the chain reads `additional_type(s) → type`
(`s-CPDN3A → CB1770 → CB1770[BANC]`), and the ACT column is not involved
for this example. The two-annotation-column chain above derives when the
shared token is NOT a BANC primary — e.g. FAFB `4I1` reaches BANC
`FB4F_a` through the token `FB4I` (`additional_type(s) → Alternative
Cell Type(s)`).

The same pair space also carries a DIRECT curated bridge: BANC rows have
a `fafb_cell_type` column (see "Curated BANC label bridges" above) whose
chains outrank pure-annotation routes in the derivation order. **Token
chaining (2026-09-27):** the label edge lands directly on the FAFB
primary only when the cell value names it EXACTLY; a rename-resolved
token (cell `SMP537`, FAFB primary `DN1pD`) terminates at the raw-token
node and the chain continues through FAFB's own `additional_type(s)`
edge (`fafb_cell_type 'SMP537' → additional_type(s) 'SMP537' → DN1pD`),
so the rename is shown, never hidden in one hop, and both sides pool
honest linker rows. **Curated subsumption (2026-09-27):** when the
curated chain bridges the SAME token to the SAME target as a one-linker
`additional_type(s)` chain, the annotation chain is dropped as redundant
— annotation-only pairs and the composed `aT`→`ACT` routes keep their
chains.

**Cross-reference rule (within one namespace)**: a primary type's
annotation cell that names *another primary of the same namespace* is a
cross-reference, not a rename — BANC's `Alternative Cell Type(s)`
concatenates the other datasets' curated labels, so such a token usually
describes how a neuron is called ELSEWHERE. The walk therefore never hops
from a primary to another primary via that primary's own annotation cell.
This closes the name-graph wander where one oddly-labeled neuron bridged a
circadian type (`l-LNv`) onto every primary sharing its `BM_InOm` label
(1,212 interommatidial bristle neurons) with zero supporting rows on the
reached types.

**Post-arrival annotation discipline with backward-reciprocity rescue
(2026-09-30)**: on the FAFB↔BANC pair — both directions — once a chain
has arrived at a target-namespace primary by same-name identity, an
annotation hop (`Alternative Cell Type(s)` departing a BANC arrival, or
`additional_type(s)` departing an FAFB arrival) may not depart from it.
An annotation cell names how the reached type's neurons are called
ELSEWHERE; two primaries sharing one such foreign token are co-named
elsewhere, not bridged to each other — using the token as
intra-namespace transitivity glues unrelated primaries. Real failures
this closes: FAFB `R8` arrived at BANC `R8` and its `R7p`/`R8p`/`R8y`
sub-type tokens fanned out onto 26 unrelated BANC ends (`m_NSC_DILP`,
`MBON26`, `Tm3` …); `s-CPDN3A`/`s-CPDN3D` were cross-paired onto
`CB1791`/`CB3612` through the shared `SMP220` token; BANC `T3` arrived
at FAFB `T3` and its `Pm03`/`Pm08` additional names fanned onto
unrelated FAFB ends — and in every case the reverse direction refuses
those pairs; that direction asymmetry is the tell (both directions'
designed annotation forms are 100% backward-reciprocated on real data).
A glue chain whose pair IS corroborated in the reverse direction (some
reverse bridge from the end type reaches the source type) is real
correspondence and survives, flagged `reciprocal` (e.g. FAFB
`DNp17` ↔ BANC `DNpe054`, whose two legs ride INDEPENDENT BANC-side and
FAFB-side annotation cells). Verified effect on real data: FAFB→BANC
loses exactly the 1,144 (v626) / 1,070 (v888) non-reciprocated glue
pairs and keeps the 49/50 reciprocated ones; BANC→FAFB loses 2,527 per
release and keeps 121 reciprocated pairs; every other pair direction is
unchanged. Deliberately NOT applied to other pairs — e.g. MCNS→FAFB,
whose designed-form control is only ~55% reciprocal (the crosswalk is
structurally one-directional), so non-reciprocity proves nothing there
and the same-name+aT continuation stays adjacent to the ratified
registry standard.

**Reverse-label tail-claimant guard (2026-09-30)**: the label lanes are
naturally read BANC-primary → labelled source type; the reverse lookup
(a source type's BANC slot) materializes only when exactly ONE BANC
primary's dominant label IS that token. A 1–2-row type whose only
labelled row carries the token wins that internal election while the
token's real population lives on ANOTHER primary (BANC `CL257` elected
the MCNS `R8y` slot with one row while BANC `R8` holds 107 `R8y`
votes but its own winner is a different token; FAFB `T1` elected BANC
`R8_unclear` with one row against 636 votes elsewhere — 45 refusals of
6,674 reverse materializations on real data, all of this shape). The
ratified population-asymmetry signal (≥10×) refuses such tail
claimants: no reverse slot is written and the provenance discloses the
refusal (`tail_claimant` + `dominant_others`). Forward lanes, same-name
slots, singleton-type label mappings (the release's own curation) and
real conflicts are untouched.

**Route scope — Curated vs Full map (re-ratified 2026-10-01)**: the
mapper's surfaces (`get_type_bridges`, `get_mapping_decision`, the
resolver) carry a `route_scope` keyword defaulting to `'curated'` — the
historical licensed routes, byte-identical. `'full'` switches to a
COMPLETE parallel mode: `licensed_route_mids` licenses **all datasets**
as connectors (minus the two endpoints; male-cns v0.9 deduped against
v1.0 — identical leg output via shared-name delegation; the previously
forbidden BANC routes included), and
`compose_full_map_bridges` composes two-leg routes from curated walks
(leg A into the mid, leg B re-walked per reached mid type). Every
per-hop rule (glue suppression, subsumptions, crosswalk licensing,
registry scoping) applies per leg — the mode widens the ROUTE universe,
never the per-hop rule set. Composed chains flag `transitive_via:
<mid>` on post-mid hops (propagated to standardized linkers, disclosure
texts, and the per-pair `mapping_origin` suffix
`(transitive via <mid>)`), and the combined bridge list stays
DIRECT-FIRST with the budget capping each class separately. Full-mode
decisions merge the transitive-only ends into the target universe with
per-target provenance (fail-closed preserved: a curated 1:1 that gains
additions becomes `valid_split_evidence` over the expanded set; a
curated conflict stays fail-closed with transitive ends disclosed).
Measured yield (200-type samples, 2026-10-01): HEMI→FAFB +88 ends (85
via BANC), MCNS→BANC +108 (78 via FAFB), FAFB→BANC +94 (67 via MCNS),
BANC→FAFB +65, HEMI→MCNS +100, MANC→FAFB +12. The validation pipeline
accepts the scope gated (`MappingValidationConfig.route_scope`,
`--route-scope full`); composed pairs carry `route_basis=composed` +
`via_mid` markers in `mapping_export.csv` / `validation_results.csv`
and the report's advisory accounting (the pair-level vocabulary is
DIRECT/COMPOSED — it names the pair's route, not the run mode).

**BANC joint type labels**: BANC `type` cells may carry comma-joined
labels the release itself left ambiguous (e.g. `TuBu09,TuBu10`,
`PVLP004,PVLP005`; ~130 distinct values per release). These are
release-native target primaries with their own curated evidence — they
are never split into their component names (splitting would claim a
resolution the release did not make) and may legitimately appear as
bridge ends.

Production resolution applies the bridge as an overlay with this
precedence:

1. **Crosswalk route** (male-cns anchored) — wins when present.
2. **Same-name identity** — a type that is a primary in both namespaces
   maps to itself (e.g. FAFB `APDN3` → BANC `APDN3`). Without this, a
   same-name type absent from the crosswalk would resolve to nothing and
   vanish from cross-dataset queries.
3. **Annotation bridge** — exactly one candidate becomes the mapping;
   several candidates become a `1-to-N` conflict (never guessed),
   deduplicated against crosswalk conflicts.

Untyped sentinels (`Unknown`, empty, bare numbers) never become bridge
targets or candidates — consistent with the cross-dataset run's default
untyped-neuron drop.

The exports state how each pair was derived: `auto_type_mapping.csv`
carries a `mapping_origin` column (`crosswalk`, `same name`, or
`annotation bridge via <token>`); `auto_type_mapping_conflicts.csv`
carries an `origin` column with the same distinction; and since the
2026-09-27 unification the per-pair `mapping_*.csv` exports (panel
"Export mapping") carry the SAME vocabulary in their `mapping_origin`
column — `same name` (bare echo), `same name+evidence` (a same-name
pair whose chain carries the verifying relation), `cross-dataset cell
type` (curated label lane), `cross-dataset cell type (alignment)`
(fallback lane), `crosswalk`, `annotation bridge via <tokens>`,
`annotation bridge via <tokens> (reciprocal)` (a glue chain rescued by
the backward-reciprocity check, e.g. `DNp17` → `DNpe054`),
`release relation`, `release alias` — so every surface distinguishes the
label lanes. Bridge pairs whose
endpoints have no male-cns anchor get their own rows. Ambiguity
resolution by neuron counts is deliberately NOT applied — the conflicts
export is the place to adjudicate those by hand.

#### Valid bridge source map (BRIDGE_SOURCE_MAP)

Every derivation bridge the mapper offers is licensed by the declarative
`BRIDGE_SOURCE_MAP` constant in
`src/comparison/cross_dataset_type_mapper.py` — valid bridges are derived
from the source map, never guessed from the data layout. The map licenses
one edge per `(home dataset, name column) -> {landing datasets}`:

| home dataset | column | lands in |
|---|---|---|
| any | `type` | any namespace (same-name identity) |
| male-cns | `flywireType` | FAFB only |
| male-cns | `hemibrainType` | hemibrain only |
| male-cns | `mancType` | manc only (the crosswalk was built against MANC v1.0) |
| BANC v626/v888 | `fafb_cell_type` | FAFB only |
| BANC v626/v888 | `fafb_alignment_cell_type` | FAFB only (fallback lane) |
| BANC v626/v888 | `malecns_cell_type` | male-cns v1.0 only |
| BANC v626/v888 | `hemibrain_cell_type` | hemibrain only |
| BANC v626/v888 | `manc_cell_type` | MANC v1.0/v1.2.1 only |
| FAFB | `additional_type(s)` | FAFB only |
| BANC | `Alternative Cell Type(s)` | BANC only |
| BANC v626/v888 | `banc_release_crosswalk` | the other BANC release only; relation-backed when available, exact same-name fallback otherwise |
| MCNS v0.9/v1.0 | `release_alias` | same-name MCNS release alias only |

On top of the map, one endpoint rule applies: **a crosswalk hop is valid
only when the bridge's endpoints include a namespace the column routes
to**. A `hemibrainType` hop on a male-cns↔BANC bridge describes a third
dataset's naming and is rejected (e.g.
`DN1pA[MCNS·type] → DN1pA[MCNS·hemibrainType] → DN1pA[BANC·type]` is
invalid), while the sanctioned hemibrain↔flywire route through male-cns
(`hemibrain type → hemibrainType → male-cns type → flywireType → flywire
type`) is licensed on both legs. Chains whose standardized linkers contain
consecutive identical `(column, value)` pairs (zero-information ping-pong,
the additional_type(s)→additional_type(s) self-loop class) are also
rejected, and every chain must end at a real primary `type` of the target
dataset.

The map is verified against the data at load time and by
`tests/core/test_type_mapper_source_map.py`, which grounds the declared
columns in the actual dataset tables, sweeps every directed namespace pair
for licensing violations, and pins the known-pair regressions.  At load
time, a declared column missing from a table that IS present aborts the
load (data drift must not silently change what is bridgeable — the run
logs `BRIDGE_SOURCE_MAP: …` and the mapper stays unloaded); an absent
optional table (e.g. no FlyWire side table) only disables its bridges.

An interactive network visualization of the map (datasets, their name
columns, and the licensed edges) is regenerated with:

```bash
python scripts/render_source_map_network.py
# → outputs/type_mapping/source_map_network.html
```

#### The bridge-rule algebra, connectors, and preference order

The derivation walk is governed by a closed algebra — the full
implementation reference lives in
`docs/technical/AUTO_TYPE_MAPPING_IMPLEMENTATION.md`. The rules:

- **Direct bridge forms** — the pair linkers above plus same-name
  identity everywhere.
- **Connectors** (`ROUTE_MIDS`): a derivation chain visits at most ONE
  intermediate namespace, and only a licensed one — male-cns connects the
  neuprint families through its own crosswalks. BANC label bridges are
  direct-only, and **BANC is never a connector**. MCNS↔BANC uses
  `malecns_cell_type`, never `flywireType`; every other unregistered pair is
  direct-only.
- **Bidirectional, no flips** — the reverse travel direction reverses
  the hop order (the SAME bridge, not a flip); a two-linker chain
  cannot reorder its linkers.
- **Evidence first, same-name LAST** — a same-name pair derives through
  its metadata bridge when one exists: male-cns `DN1a` rows'
  `flywireType` cell naming `DN1a` verifies FAFB `DN1a → DN1a` as
  `flywireType 'DN1a'` instead of the bare "same name — no metadata
  verification" echo. The same-name chain is the LAST choice, shown
  only when no evidence edge connects the pair.
- **Registry-less pairs stop at the arrival** — the post-arrival
  annotation continuation is a two-linker registry-standard privilege
  (male-cns↔FAFB etc.); BANC pairs cannot wander their annotation
  classes after landing.
- **Curated label / release / alias hops end the derivation** — when one
  of these linkers lands in the target namespace, nothing may follow it
  (the reached type's rows carry the evidence themselves); when it lands
  in a licensed intermediate (BANC `malecns_cell_type` → male-cns), only
  the licensed hub leg may continue, never an annotation hop.
- **Primary→annotation-primary hops are refused** — a primary's
  annotation cell naming another primary of the SAME namespace is a
  cross-reference (see above), not a rename edge; the derivation never
  hops through it.
- **Zero-evidence chains are dropped at pooling** — as a safety net, a
  chain whose target-side linkers all pool zero rows on the reached type
  never renders (its own coverage would read `0 of n`); the UI logs the
  drop instead of showing an unsupported derivation.
- **Untyped labels** (`Unknown`, empty, bare numbers) never become
  bridge nodes or targets.

#### Per-release namespaces (version control)

BANC v626 and BANC v888 are separate mapping namespaces, each resolving
against its OWN neuron tables — a `banc_v888` selection can never land
v626 names or pool v626 bodyIds. Both releases sit in the mapper's
`DATASET_PRIORITY` walk (v888 right after v626), so a v888-only type name
auto-detects its own namespace instead of falling through to "unknown". They have one narrow exception: the
metadata-backed `root_626`↔`root_888` relation provides a direct
`banc_release_crosswalk` type bridge. It retains duplicate `root_626` rows,
uses `root_888` (never `banc_888_id`), and never falls back to equal numeric
IDs. Generic BANC annotation/transitive paths remain forbidden.

`male-cns:v0.9` also keeps its native table and bodyId space. For a shared
primary name, the mapper emits `v0.9 type → release_alias → v1.0` and then
uses the v1.0 crosswalk for the requested target. A v0.9-only name never
borrows a fabricated v1.0 name; it can use its own `flywireType`,
`hemibrainType`, and `mancType` columns as a lower-tier fallback.

The dataset selector recommends `male-cns:v1.0` when v0.9 is selected. The
notice is advisory and explicit: it does not silently replace the selection,
and it is hidden when the newer release is already selected or unavailable
in the current options. The shared policy lives in
`src/utils/dataset_release_registry.py`; uncertified MANC release candidates
remain non-recommended.

The v0.9/v1.0 comparison is exposed in mapper diagnostics: the real local
tables have 176,379 common bodyIds, 11,597 shared primary names, 1,354
typed-row name disagreements, and a nested `release_alias_disagreement`
record for the 29 shared v0.9 names affecting 71 joined rows. These are audit
signals; they never substitute v1.0 bodyIds for a v0.9 query.

#### One shared backend (viewer ⇄ panel parity)

The 'See available neurons' mapped view and the cross-dataset tab's
Type Mapping panel resolve every type through the same
`mapped_type_targets()` backend (stored/alias resolution ∪
derivation-bridge ends) and count each mapped target's neurons ONCE —
both surfaces report identical target sets and unique neuron counts
(e.g. `circadian_clock`: 21 FAFB types / 242 neurons → 40 male-cns
targets / 219 unique neurons on both).

The Type Mapping panel's **Mapping graph (HTML)** and **Mapping sankey (HTML)**
plot **types only**. The chip that produced them — a taxonomy hit such as FAFB
`cell_type = circadian_clock` — is search provenance, not a mapping endpoint,
so it is not drawn as a node on any canvas (it stays in the panel tables, the
per-pair cards and the CSV exports, where the `matched_origin` /
`source_entry` / `matched_column` columns carry it).

When one dataset is the query's origin and exactly two datasets receive it,
both exports put the origin in the **middle column** with one target on each
flank (`target 1 | origin | target 2`), so each column boundary is exactly one
dataset pair; the flank covering more origin types takes the left. Dagre cannot
do that (it ranks by edge direction), so the network renders from explicit
preset positions, and Plotly's Sankey ignores an `x` that contradicts its link
direction, so the left half's ribbons are drawn `target ← origin` — a drawing
convention only: a type mapping is an equivalence, the artifact says so in a
floating note, and the derivation direction stays on the edge hover. (The
network keeps the derivation direction on BOTH halves, so its arrows point
outward from the centre — measured on the `circadian_clock` star: 44 edges to
the left flank, 43 to the right.) The
composed Sankey exists for one or two targets; a wider selection falls back to
the per-pair **Sankey (type-level)** buttons.

#### BodyId-level granularity (row-based bridge evidence only)

The type mapper's bodyId-level granularity comes **exclusively from
row-based bridge evidence** — never from connectivity similarity:

- **Branch pools**: for a 1-to-N parent (e.g. male-cns `5thsLNv_LNd6` →
  FlyWire {`5th-LNv`, `LNd_CRY+_ITP+`}), each branch carries the
  bodyIds its bridge **linker rows** tie to it
  (`resolve_prioritized_bridge_pool`, basis "linker rows").  Where no
  linker rows exist, the pair keeps its full populations (basis "full
  population").
- **Vote provenance**: every label bridge records how many bodyIds back
  it and with what provenance (`get_mapping_support` / decision
  `support` / the `mapping_support` + `support_*` export columns —
  e.g. `s-LNv_a`: 2 votes vs `aMe24`: 1 auto vote).
- **Direct same-name pairs** — the name must match **exactly** — pool
  all bodyIds of the type across both datasets (`same name (all bodyIds
  pooled)`), except a same-name type inside a 1-to-N — there the
  bridge's linker rows/votes resolve the bodyId-level resolution per
  branch, and under the same-name-first rule (above) the same-name
  candidate is SELECTED with the other candidates demoted to rival
  suspects.  A same-name pair whose populations differ by an order of
  magnitude (ratio < 0.1 — e.g. `TmY18`: 1,367 neurons in male-cns vs 1
  in FlyWire) carries an `extreme population asymmetry … suggested
  check` flag on the support surfaces and in the viewer's same-name
  candidates: a suggested check for the user (the full-dataset audit
  found such pairs are predominantly annotation-coverage asymmetries,
  not naming errors — 118 flagged of 18,797 pairs; see
  `local_data/same-name-fidelity-audit-flagged.csv`).

**The mapper never uses connectivity similarity.**  Per-neuron
connectivity verification of splits — profiling, matching, correlation
gates — is the validate-expand-visualize pipeline's role
(`mapping_validation`; see
`docs/technical/TYPE_MAPPING_VALIDATE_EXPAND_VISUALIZE_PIPELINE.md`),
which owns that machinery end-to-end including any UI for it.

When several types are queried — or one chip resolves to many types (a
taxonomy query such as `circadian_clock`) — the panel's overview gains a
collapsed **Per-type breakdown** expansion: one row per `(matched type ×
target dataset)`, so the mapped counts are specific to the row's target
dataset and never summed across datasets, plus the originating query
chip.  Its **Mapped neurons** cell combines count and type breadth:
`{N}({m} types)` from 2 mapped types up, plain `{N}` for a single mapped
type.  Each row also carries an **Evidence reach** cell (2026-09-27
three-tier readout): all flows' pools toward that row's target dataset,
disclosure ends included — e.g. the `APDN3 → banc_v888` row reads
claim `7` vs reach `12(3 types)` because the same-name-first rivals
LMTe01/LTe71 are evidence, not claims.  Single-type previews look
exactly as before, and the breakdown is view-only.

#### The type mapper's boundary (row-based evidence is carried, not consumed)

The mapper's role ends at the mapping **plus its row-based bodyId-level
evidence**: every bridge carries the bodyIds that back it (BANC label
votes per source type with curated vs `auto:`-transferred provenance —
e.g. `s-LNv_a`: 2 votes, `aMe24`: 1 auto vote; FAFB `additional_type(s)`
linker rows; release root-id pairs) in `_bridge_provenance`.  That
evidence is **passed through, never consumed**:

- `get_mapping_support(source_type, source_dataset, target_type,
  target_dataset)` returns the pair's retained bridge support record;
  `get_mapping_decision` carries it per branch under `support`.
  **Direct same-name pairs** carry their own bodyId-level handling: all
  bodyIds of the type pooled across both datasets (full populations,
  fully in-map — nothing refined, nothing gated), surfaced as
  `same name (all bodyIds pooled)` in the support surfaces.  **One
  exception**: a same-name type that participates in a 1-to-N/N-to-1
  structure (e.g. MCNS `CB2572` → FAFB {`CB2572`, `CB2572a`,
  `CB2572b`}) is resolved by the evidence bridge instead — pooled
  identity would wrongly absorb the other branches' bodyIds, so such a
  pair carries no identity marker; the bridge's linker rows/votes and
  the split structure resolve the bodyId-level resolution.  The compact
  `auto_type_mapping.csv` gains a trailing
  **`mapping_support`** column (e.g. `cross-dataset cell type:
  5thsLNv_LNd6=1 (auto 1)` vs `…=2 (auto 2)` — the exact data that
  distinguishes the `aMe24` weak-vote bridge from `s-LNv_a`), and
  `auto_type_mapping_per_bridge.csv` gains `support_votes`,
  `support_verified`, `support_auto`, `support_linker_value`.
- What the mapper never does with this evidence: gate, rank, or verify
  a mapping with it.  Verification is the validate-expand-visualize
  pipeline's role (bodyId-level connectivity scoring, morphology) — the
  panel's connectivity split view is an informational display of that
  style of evidence, not a mapper decision input.  A downstream
  "tight" mapping policy (only verified/evidence-backed mappings
  treated as valid) is a **consumer-side** filter over these surfaces:
  the bridge support columns for label-vote gating, the validation
  pipeline for full verification.

#### Bidirectional type coverage (the panel's tables)

The panel's per-pair **Type coverage** expansion shows two tables over the
same mapped pairs. Relationship cells follow the row subject's fan-out: a
forward row (queried type → several targets) and a backward row (receiving
type ← several sources) both read **1-to-N** — read the backward rows from
the receiving type back to its sources; a multi-source row never reads
`1-to-1`.

The **Relationship** cardinality is not confined to these tables: the
pair-card **mapped-pairs** table and the collapsed **per-type breakdown**
report the same per-flow cardinality (from `_pair_relationship`, matching
the all-pairs CSV's `relationship`), and both also carry a **Suspects**
column — a `⚠ suspects (N)` badge on a row whose same-name-first selection
demoted rival candidates.  Expanding the row's collapsed **Suspects**
block below the table lists the per-rival evidence (rival name, own 1-to-1
pair, votes, reverse target, rival-pair status, populations) — the facts
`get_same_name_conflict_detail` already returns, rendered once per surface
and never hover-only.  The Type coverage expansion's own title is
data-driven off the forward rows (`1-to-N fan-out` / `N-to-1 fan-in` /
`all 1-to-1`, plus `· ⚠ N suspect pair(s)` when the pair carries suspects),
so the heading matches the tables rather than always claiming fan-out.

Backward rows marked **dataset-wide incoming** list every source type in
the source dataset that maps onto the receiving type (the active query
members marked in `Mapped from`), with the incoming family's selected- and
all-valid-union coverage — the context that explains why one queried type's
few neurons fan out to a large target population (e.g. MCNS `SMP227`: 6
neurons → 94 FAFB neurons across `s-CPDN3B/C/D`, whose incoming families
are 4/4/6 MCNS types). This context is explanatory evidence only; it never
changes which mappings the run accepts.

The four coverage columns carry short `CODE · selected` / `CODE · all
valid` labels; hover a header for the full dataset key and the scope
explanation: **selected** = the first supported bridge chain after
deterministic evidence ordering (the primary chain behind edge weights and
hovers, not a biological adjudication); **all-valid union** = the
deduplicated union of bodyIds from every independently supported candidate
bridge (completeness of supported evidence; branches are not mutually
exclusive). Neither column is a bodyId-to-bodyId correspondence.

#### A reverse lookup is not the inverse of a forward claim

The panel invites "map it back and see if it returns" — but the two
directions do not walk the same lanes, and a forward claim is **not
required to survive reversal**. Forward reaches its target through adopted
same-name pairs and annotation bridges whose vocabulary column exists on
one side only (a FlyWire `type` name the reverse lane does not re-read, or
an `additional Type(S)` hop with no reverse spelling). Measured on
`circadian_clock` ↔ `banc_v888`: the forward side carries the adopted
same-name pair `l-LNv → l-LNv`, yet querying BANC `l-LNv` alone returns
**zero** flows back (FAFB holds 8 rows named `l-LNv` in `type` and none in
the column the reverse direction resolves); forward reaches `CB3767` as an
annotation bridge that has no reverse form at all. Where the reverse
direction does speak it agrees with the target's own curation
(`LMTe01 → APDN3`, `LTe71 → APDN3`, `CB3508 → s-CPDN3D`), so reverse
silence means "this lane does not re-walk", never "the claim is wrong".

The independent check for a claim set is therefore not the mapper's
reverse query but the TM VEV validation pipeline's backward panels
(per-bodyId reverse connectivity scans — `expansion/target_matches.csv`,
the Homolog · backward tab), which score against the whole source dataset
instead of re-walking name lanes.

#### Reading the per-dataset summary table (the preview)

The preview's summary strip has one row per selected dataset (hover any
header for the same explanation):

| Column | Meaning |
| --- | --- |
| Matched types | Search terms that matched type names in this dataset. |
| Neurons | Neurons of the matched types in this dataset — also the issued side: every matched type is issued into the resolution, flowed or orphaned. |
| Mapped neurons | What this dataset RECEIVES: branch-claimed bodyIds (union of the branches' resolved pools — the claim set), shown as `{N}({m} types)` from 2 distinct received types up, plain `{N}` otherwise. Only pairs the decision **adopted** count; a same-name rival it declined, or a valid-split fan-out it did not adopt, is listed in the pair table as a disclosure row and never enters this number — which is what makes it the same claim set the validate-expand-visualize report grades (204 bodyIds for the circadian_clock example since the same-name-first adoption; earlier adopted-only readings were 198/205). A dataset that only issues the query reads 0. |
| Evidence reach (all flows) | **The reach tier (2026-09-27 three-tier readout)**: every flow's pools unioned — the claim set PLUS the disclosure ends the decision declined (a same-name rival, a vote-declined annotation pair), shown in the same `{N}({m} types)` form. On circadian_clock FAFB → banc_v888: claim `198(39 types)` vs reach `205(42 types)`; the 7-body / 3-type difference is the disclosure material (LMTe01/LTe71 — same-name-first rivals of APDN3; CB3767 — fan-out branch not adopted), listed per type with its decline reason in the pair table's "not adopted" rows. The same scope the CSVs' all-valid columns and the coverage tables publish. claim ≤ reach always. |
| Disclosure detail | Not a column — the hover/expanded material behind the two figures above: each declined end's type, body count, and reason (`same-name-first rival — 'X' selected instead` / `fan-out branch not adopted` / `conflict — fail-closed`). Delivered across the mapper→TM VEV interface as the decision's `disclosure_targets` key and rendered by the validation report's "Disclosure evidence" section + `disclosure_evidence.csv` (advisory bin, never the headline counts). |
| Out-map (in-map types) | Neurons of the received (in-map) types in this dataset that the claim set above does **not** reach — the union of those types' own populations minus the branch-claimed bodyIds, so a type pooled on its full population contributes 0. On the circadian_clock FAFB → male-cns example it reads 219 − 204 = **15**. This is the PANEL-side figure and deliberately not the validation pipeline's `family` bin: that one is 11 rows on the same query, because a morph-qualified candidate closes a hole there and nothing does here. Coverage evidence, never a mapping claim. The collapsed per-type breakdown computes its figure **per received target type** (a neuron an adopted branch reaches anywhere in that type counts as claimed) and sums it over the row's accepted types — so it is narrower than the validation pipeline's per-**branch** `family` bin under N-to-1 convergence, and its rows total 17 where the deduped dataset union is 15. |
| Unmapped (orphans) | Matched types here with no realized counterpart in another selected dataset. |

**Claimed versus realized (2026-09-12).** A crosswalk cell can name a
counterpart the target dataset does not actually have (e.g. male-cns
`CB4091` carries `flywireType CB4091`, but FAFB v783 has no such neurons —
863 of the crosswalk's FAFB names are like this, plus hundreds more
toward hemibrain/MANC). Such names are registered as **stale claims** at
mapper build time (`_stale_type_claims`) and are kept out of the target
dataset's namespace (`_dataset_types`), so they can never resurface as
identity aliases or bridge endpoints. `mapper.type_exists(dataset, name)`
answers namespace ground truth post-load; `mapper.is_stale_claim(...)`
reports the registered claims.

The resolver reports a stale claim with the dedicated `claimed` status
instead of `mapped`: the claim stays visible as evidence (its target name
is retained, `get_mapped_type` still returns it), but no equivalence is
licensed and profile expansion follows it nowhere. When the queried name
is itself native in the target namespace, the realized same-name
equivalence wins and the unfulfilled claim is kept in the resolution
reason (e.g. male-cns `Dm8a`, whose crosswalk names the FAFB-nonexistent
`yDm8`, resolves to FAFB's native `Dm8a`).

In the preview, a stale claim therefore never counts toward the target's
*Mapped neurons* — only a target type with ≥1
neuron in that dataset does. The originating dataset's orphan entry
reports the unfulfilled claim, e.g. `male-cns:v1.0 → flywire_FAFB_v783:
CB4091 (20) — auto-mapping claims 'CB4091' but flywire_FAFB_v783 has no
such neurons`. A row with zeros everywhere therefore means "nothing
realized here", consistently with the "No mapping graph" header.

## The two label lanes (explicit vs automatic)

DROCAT standardizes neuron type names through TWO independent lanes. They
never mix implicitly, and the explicit lane always wins:

1. **Explicit LabelMapper lane** — a user-provided `LabelMapper` (UI presets,
   custom mapping files) is applied at connection-extraction time: the
   connection builder writes `std_label_pre`/`std_label_post` and overwrites
   the raw `type` columns BEFORE results are cached (`LabelMapper.
   apply_to_dataframe`). Runs with an explicit mapper are compared under
   those labels; auto mapping does not second-guess them.
2. **Automatic mapper lane** — when no explicit mapper governs the run,
   `CrossDatasetTypeMapper` (male-cns v1.0 tables) resolves names through the
   shared validity resolver (`comparison/type_resolver.py`) at
   comparison/merge time: homolog candidates, path/edge canonical merge keys,
   profile expansion, query mapping, reports.

Shared conventions across both lanes: `label_utils.is_untyped_type_label`
is THE untyped-neuron predicate (pathfinding and comparison must agree on
what counts as untyped), and the auto lane's conflict/split/fallback policy
is the one documented in this guide (fail closed on conflicts; raw long-tail
fallback is counted, never silent).

### The query-anchored merge policy (auto-mode comparison runs)

Comparison runs with auto type mapping additionally build a per-run
**MergePolicy** (`comparison/merge_policy.py`) that makes the merge
granularity *query-anchored*: the merge keys of the aligned frames become
the query chips' group labels instead of the canonical male-cns namespace.

- **Anchor selection (B3)** — the anchor is the first chip's home
  namespace when all chips resolve into a common naming (resolution-based,
  not origin-based); a run whose chips are native in every selected
  dataset (true same-name) uses the shared naming with the canonical
  fallback for everything else; mixed runs emit the
  `[type granularity]` warning and resolve conflicted groups to their
  minimal inseparable leaves (the parent keeps its own whole row).
- **1-to-N granularity (B1)** — a chip on the "1" side merges ALL its
  branches into one row (no vote threshold; weak branches are warned per
  the BANC auto-label block below); a leaf-anchored chip covers only its
  own branch. Auto-only branch seeds are chain-terminal: a branch
  licensed only by auto-transferred labels stays confined to its own
  dataset and never recruits same-named cell types from other datasets
  through downstream identities (the FAFB aMe24 case). A clean target that is itself the "1"-side of a 1-to-N
  toward the chip's namespace is excluded even when the reverse direction
  is a clean rename (e.g. FAFB `5th-LNv` → MCNS `5thsLNv_LNd6`) — the
  parent keeps its own whole row.
- **Fan-in** — a type claimed by two queried parents merges with NEITHER
  (`[merge fan-in]` warning); the custom label mapper overrides. When a
  queried parent chip is one of the claimants, the warning explicitly
  states that the parent's row is PARTIAL by design (it excludes the
  co-queried shared branch, which is presented separately at its own
  minimal level); querying the parent alone merges it.
- **Aggregation rides lane-1 plumbing (Decision 8)** — the policy's
  raw→group-label map is materialized as a *policy-synthesized*
  `LabelMapper` applied through the same
  `apply_to_dataframe` → `std_label_*` lane at alignment time. It is
  separate from the user's mapper, and because it is keyed on raw names
  (which user-governed types no longer carry at that point), user
  mappings win without any precedence code.
- **Evidence surfaces** — the run exports
  `type_resolution_topology.json` (per-group tree with per-branch
  linker-refined pool sizes and vote support), and
  `auto_type_mapping.csv` gains additive trailing `anchor_group` /
  `auto_only` columns (`anchor_group` is tagged only when EVERY
  non-empty endpoint of the row belongs to that same group — a row that
  merely touches a group through one endpoint stays blank). `[merge
  policy]` and `[BANC auto labels]` blocks
  are appended to `user_warning_notes.txt` (the auto-label block opens
  with a per-direction count summary); mappings flagged auto-only
  stay VALID — the custom label mapper is the removal path. Neuron
  counts (`neuron_counts_by_type.csv` and the report's Neuron Counts by
  Type table) are keyed by group labels too, so counts, presence,
  similarity, and networks all agree on group membership; merged rows
  carry their per-dataset raw composition in the `group_members` column
  / table sub-line.

### 3. User Warnings and Double-Check Recommendation

Auto type mapping is applied automatically, so runs surface what it changed
in two places — **please double check them before interpreting
cross-dataset results**:

1. **Console summary**: when source/target neurons are resolved, the run
   prints the auto-mapped names plus explicit `N-to-1` / `1-to-N` warnings,
   and ends with a reminder to double check the automatic mappings.
2. **`user_warning_notes.txt`** in the run folder root (rendered in the run
   guide's Warnings section). It is written when auto type mapping:
   - **expanded** a queried type name to a different name in a target
     dataset (e.g. `SLP249` → `APDN3` in FAFB, `MeVPLo2` → `MTe07`),
   - hit an **N-to-1** mapping (several types share one name across
     datasets — they are *not* merged to avoid wrong aggregation), or
   - hit a **1-to-N** mapping (a type splits into several names in the
     other dataset — no automatic mapping is made).

   Example:

   ```
   - Auto type mapping expanded queried type 'SLP249' to 'APDN3'
     (FlyWire FAFB v783); the queried name may not exist there.
   - N-to-1 type mapping: 'SLP249' is one of 4 types (CL125, PLP080,
     SLP249, SLP250) that all correspond to 'APDN3' in FlyWire FAFB v783;
     they were NOT merged to avoid wrong aggregation.
   - These name mappings were applied automatically - please double check
     them (against the datasets' type annotations or the exported mapping
     files) before interpreting cross-dataset results.
   ```

For cross-dataset comparisons the full applied mapping is additionally
exported to `auto_type_mapping.csv` (and conflicts to
`auto_type_mapping_conflicts.csv`) in the output folder.

3. **See available neurons viewer (expanded search)**: when a search finds
   no rows in the selected dataset, the viewer probes two expansions and
   shows a clearly separated panel:

   - **Type-name matches (native, mapper-free)**: the search text is
     matched as a case-insensitive substring against the `type` column and
     the taxonomy label columns (`class`/`cell_class`/`cell_type`/… ) of
     every other *cached* dataset's index. Matches are name-similar entries
     — **not necessarily the same type** — listed with neuron counts
     (exact matches first, then by count; capped per dataset). This tier
     works even when the auto type mapping knows nothing about the query.
   - **Auto type mapping** (the tier described above): renamed types,
     splits, and N-to-1 groups for the query name.  A `splits into`
     candidate additionally carries the branch pools' vote provenance
     and — when the split-verification view has run — a per-neuron
     verification summary; a `one of N` candidate notes that its parent
     family splits by neuron.

   Every matched foreign type is additionally annotated with what it is
   called in the selected dataset (unique rename, same name, or the members
   of a refused N-to-1 aggregation); types without a counterpart stay
   visible unannotated so the user is led to inspect them in the other
   dataset. Cross-dataset rows are informational only — they are never
   merged into the selected dataset's table or selection (bodyIds from
   different datasets must not be mixed) — and the panel repeats the
   double-check recommendation.

   Each dataset block also has a **"Mapped types"** button (the mapped type
   count is on its tooltip). It runs the equivalent search in the selected
   dataset: the main
   table switches to a *mapped-type view* listing the current dataset's
   neurons of the mapped type names, with a prominent warning banner
   ("automatic mapping — please double check") and a "Back to normal
   search" exit. Three provenance columns are appended per row: the
   **foreign type(s)** that mapped to it, the **map used column** (`type`,
   `additional_type(s) · via '<old name>'`, or the `flywireType`
   crosswalk), and the **matched column** that triggered the expansion
   (e.g. `cell_type · circadian_clock`). Because the view queries the
   *selected* dataset's index only, selection and add-to-query keep
   working normally; any new query leaves the mapped view and returns to
   the regular workflow.

### 4. Standardization Process

When comparing profiles from different datasets:

1. **Query type standardization**: Input type names resolve to canonical names for profile retrieval
2. **Partner type standardization**: When computing similarity, each partner type resolves through the shared validity-aware resolver (`comparison/type_resolver.py`) — licensed renames and valid splits map to canonical names, conflicts are excluded (fail closed), and unmapped types keep their raw name as the counted fallback
3. **Similarity computation**: Jaccard, cosine, and rank correlation are computed on the resolved partner type sets

## Usage

### Enabling Auto Type Mapping in ComparisonAnalyzer

```python
from comparison import ComparisonAnalyzer, ComparisonParameters

params = ComparisonParameters(
    datasets=['male-cns:v0.9', 'flywire_FAFB_v783'],
    source_neurons=['MeVPLo2', 'MeVPaMe1'],
    target_neurons=['aMe.*'],
    auto_type_mapping=True,  # Enable type mapping (default: True)
    # ... other parameters
)

analyzer = ComparisonAnalyzer(params)
results = analyzer.run_comparison()
analyzer.export_results()  # Exports filtered auto_type_mapping.csv
```

### In ConnectivityProfileComparer (Cross-Dataset Mode)

```python
from comparison.profile_comparator import ConnectivityProfileComparer

# Cross-dataset comparison with dict query format
comparer = ConnectivityProfileComparer(
    query={
        'hemibrain:v1.2.1': ['MBON-γ2α\'1', 'MBON-γ5β\'2a', 'PPL1-γ1pedc'],
        'male-cns:v0.9': ['MBON-γ2α\'1', 'MBON-γ5β\'2a', 'PPL1-γ1pedc']
    },
    dataset=None,  # Must be None for cross-dataset (warning if not)
    use_auto_type_mapping=True  # Enable partner type standardization
)

# Generates N×M similarity matrix comparing types across datasets
results = comparer.run()
```

## Same-name policy & query resolution (2026-09 rev 2)

The mapper distinguishes two same-name cases, and the two consumers apply
them differently:

- **Same name + evidence** — a cross-dataset relation (crosswalk cell,
  label column, release alias) resolves the source type back to that very
  target name. This is the strongest pairing: it ranks first, survives the
  derivation union intact, and is labelled `same name+evidence` in the
  per-bridge mapping export (`mapping_origin`).
- **Bare same-name echo** — the name exists in both releases but no
  relation connects it. It stays the LAST choice for merging and keeps the
  plain `same name` label.

For **type-merging surfaces** (presence matrices, canonical rows), a unique
relation that names a *different* counterpart wins over the name identity
(e.g. male-cns `Dm8a` merges with FAFB `yDm8`, not with FAFB's unrelated
`Dm8a` type). For **input-query resolution** the priority flips: a query
token that is a native type of the target dataset resolves to itself
(`same_name_identity`), and the relation only annotates it — `confirmed`,
`contradicted` (identity kept, counter-evidence shown) or `none`. A mapper
`conflict` refuses in both surfaces.

Taxonomy-column values (e.g. FAFB `cell_type=circadian_clock`, a value no
other dataset shares) resolve per dataset by expanding to their member
types, and datasets without the value bridge the concept through member
mapping: each member type is resolved through the auto mapper and the
union is queried (`taxonomy_mapped` records in
`comparison_report_used_data/query_resolution.csv`).

## API Reference

> **New analysis code should use the shared validity-aware resolver
> (`comparison.type_resolver`)**, not the raw mapper methods below. The
> resolver exposes `resolve_valid_targets`, `resolve_one_target`,
> `canonical_merge_key`, `expand_profile_types`, and `resolve_flow_status`,
> and is what the panel, viewer, homolog finding, and profile comparison
> all use. `CrossDatasetTypeMapper.get_mapped_type()` and
> `resolve_type_across_datasets()` are **COMPATIBILITY-ONLY**: they carry no
> mapping status/provenance and return `None` for splits and conflicts.

### CrossDatasetTypeMapper

```python
from comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper

# Create mapper instance
mapper = CrossDatasetTypeMapper(
    workspace_path='/path/to/workspace',  # Optional
    neuron_df_path=None,  # Optional, uses default location if None
    verbose=True,
    # Optional: explicit FAFB/BANC neuron tables for rename resolution;
    # a None value disables it for that namespace
    flywire_neuron_df_paths=None,
    # Optional: explicit MCNS v0.9 table for native release queries
    mcns_v09_neuron_df_path=None,
)

# Load mappings (called automatically when needed)
mapper.load()

# Get equivalent type in target dataset
mapped = mapper.get_mapped_type(
    type_name='lLN7',
    source_dataset='hemibrain:v1.2.1', 
    target_dataset='male-cns:v0.9'
)
# Returns: 'ALIN4'

# Resolve type across multiple datasets
mappings = mapper.resolve_type_across_datasets(
    type_name='MeVPLo2',
    datasets=['male-cns:v0.9', 'flywire_FAFB_v783', 'hemibrain:v1.2.1']
)
# Returns: {'male-cns:v0.9': 'MeVPLo2', 'flywire_FAFB_v783': 'MTe07', ...}

# Get canonical (male-cns) type name
canonical = mapper.get_canonical_type('MTe07', source_dataset='flywire_FAFB_v783')
# Returns: 'MeVPLo2'

# Standardize partner types for cross-dataset comparison
standardized = mapper.standardize_partner_types(
    partner_types={'lLN7': 10.5, 'KC-γm': 5.2},
    source_dataset='hemibrain:v1.2.1'
)
# Returns: {'ALIN4': 10.5, 'KC-γm': 5.2} (lLN7 mapped to male-cns name)

# Get display name with cross-dataset mappings
display = mapper.get_display_name(
    type_name='MeVPLo2',
    datasets=['male-cns:v0.9', 'flywire_FAFB_v783'],
    source_dataset='male-cns:v0.9'
)
# Returns: 'MeVPLo2(MTe07)' if names differ across datasets
```

### Export Methods

```python
# Export mappings (filtered to result types)
mapper.export_mapping(
    output_path='auto_type_mapping.csv',
    filter_types={'MeVPLo2', 'ALIN4', 'DNp01'}  # Only include these types
)

# Export mapping conflicts (1-to-N or N-to-1 relationships)
mapper.export_conflicts(
    output_path='auto_type_mapping_conflicts.csv',
    filter_types={'WED092'}  # Optional filtering
)
```

### Disabling Auto Type Mapping

If you want to compare connectivity profiles without type name standardization (e.g., to see raw differences in naming conventions):

```python
# In ComparisonParameters
params = ComparisonParameters(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    auto_type_mapping=False,  # Disable standardization
    # ...
)

# In ConnectivityProfileComparer
comparer = ConnectivityProfileComparer(
    query={'hemibrain:v1.2.1': [...], 'male-cns:v0.9': [...]},
    use_auto_type_mapping=False  # Disable standardization
)
```

## Implementation Details

### Partner Type Standardization

When computing connectivity profile similarity across datasets, the key insight is that **partner types** must also be standardized. For example:

**Profile A (hemibrain: ALIN4 → lLN7)**:
- Upstream: `PPL1-γ1pedc` (20%), `lLN7` (15%), `KC-γm` (10%), ...
- Downstream: `FB2B` (25%), ...

**Profile B (male-cns: ALIN4)**:
- Upstream: `PPL1-γ1pedc` (18%), `ALIN4` (16%), `KC-γm` (12%), ...
- Downstream: `FB2B` (22%), ...

Without standardization, `lLN7` and `ALIN4` would be treated as different types, reducing similarity. With standardization to male-cns names:

**Profile A (standardized)**:
- Upstream: `PPL1-γ1pedc` (20%), `ALIN4` (15%), `KC-γm` (10%), ...

**Profile B (standardized)**:
- Upstream: `PPL1-γ1pedc` (18%), `ALIN4` (16%), `KC-γm` (12%), ...

Now the Jaccard and cosine similarities correctly identify these as highly similar profiles.

### Supported Datasets

Type mappings are available for:
- `male-cns:v1.0` (active default mapping source)
- `male-cns:v0.9` (retained for compatibility / native release queries)
- `flywire_FAFB_v783`
- `banc_v626`
- `banc_v888`
- `hemibrain:v1.2.1`
- `manc:v1.0` / `manc:v1.2.1`

### Output Files

When running `ComparisonAnalyzer.export_results()` with `auto_type_mapping=True`, two files are generated:

1. **auto_type_mapping.csv**: Type mappings for neurons in results only
   ```csv
   male-cns:v0.9,flywire_FAFB_v783,banc_v626,banc_v888,hemibrain:v1.2.1,manc:v1.0,manc:v1.2.1
   ALIN4,ALIN4,ALIN4,ALIN4,lLN7,,
   DNp01,DNp01,DNp01,DNp01,DNp01,,
   MeVPLo2,MTe07,MTe07,MTe07,,,
   ...
   ```

2. **auto_type_mapping_conflicts.csv**: 1-to-N or N-to-1 mapping conflicts
   ```csv
   source_dataset,source_type,target_dataset,target_types,relationship
   male-cns:v0.9,WED092,flywire_FAFB_v783,"WED092b, WED092c, WED092d",1-to-N
   ```

### Conflict Handling

- **Unresolved 1-to-N mappings**: One source type has several target
  candidates. The candidates and licensed bridge chains remain available as
  explicitly labeled evidence, but no single canonical target is selected and
  automatic mapped-neuron totals do not choose a branch. The resolver reports
  this as `valid_split_evidence` and expands to every licensed branch; the
  compatibility-only `get_mapped_type()` returns `None` for the same case.
  For example, MCNS `SMP227` → FAFB `s-CPDN3B`, `s-CPDN3C`, and
  `s-CPDN3D` is valid split evidence, not a single accepted mapping.
- **N-to-1 mappings**: Multiple source types share one target name. These
  types should NOT be aggregated incorrectly; coverage and artifacts retain
  their independent endpoint populations.
- **BANC label votes**: a unique winning vote may be accepted, including a
  known normalized `auto:` label, but a conflicting vote set stays rejected.
  For example, BANC `CB1011` remains unmapped toward MCNS despite its
  same-name row.

### Same-name-first within a fan-out (rival suspects)

When a fan-out's candidate set contains the **source type's own name** —
exactly, never as a prefix (MCNS `SMP520` → `{SMP520a, SMP520b}` never
fires; `DNp51,DNpe019` never matches a part) — the mapper SELECTS that
same-name candidate instead of leaving the fan-out unresolved:

```
status='mapped', target_type=<same name>, relationship='suspects',
suspects=True, fan_out_candidates=<the OTHER candidates>,
target_types=[<same name>]        # the rivals stay OUT by default
```

- The other candidates become **suspects**: exported one row per rival in
  `auto_type_mapping_suspects.csv` (with `rival_has_own_clean_pair`,
  `rival_pair_status`, `reverse_target`/`backs_source`, votes, populations,
  the pair-level `selection_disposition`, and a ready-made
  custom-label-mapper entry).
- **Dispositions are per path** (plan-samename-first-fanout-resolution
  §0.0): a structural split (crosswalk 1-to-N into a BANC target,
  `valid_split_evidence`) fires broadly; a **terminal BANC vote conflict**
  fires only when **every** rival candidate has its own 1-to-1 pairing
  (the duplication check — otherwise it keeps its ordinary `conflict` status
  and goes to the suspects list for the user to adjudicate); an
  `evidence_only` (N-to-1 convergence view) fan-out is **excluded**.
- The rivals are NOT candidates for the source type, so they are left out
  of `target_types` by default (`mapper.include_suspects_in_targets = True`
  restores `[selection] + rivals` for in-pipeline verification).
- The user's inclusion path for a rival believed to be a buried fact is the
  **custom label mapper** (user mappings win by construction).
- Read the decision via `mapper.same_name_first_fires(...)`, the per-rival
  evidence via `mapper.get_same_name_conflict_detail(...)` /
  `mapper.same_name_suspects_for_source(...)`, and the run-scoped counts
  via `mapper.same_name_first_summary(...)`.
- **Per-rival labels state observations, never verdicts**: the evidence
  record carries `rival_pair_status` = `own_1to1_pair` /
  `no_own_1to1_pair` (does that rival's own name also pair 1-to-1 in this
  direction) plus the boolean `rival_has_own_clean_pair`.  Terms like
  *duplicate*/*(dup)* and *confirmed* belong to the verification pipeline
  (its bodyId-level tag), not here — the mapper never verifies.
- **UI surfaces** (plan-ui-type-mapper-alignment): the cross-dataset tab's
  Type Mapping panel and the "See available neurons" viewer with
  cross-dataset mapping ON display this state — a short marker plus
  **collapsed** details.  In the panel the
  `⚠ suspects (N)` badge rides on the row's own **Suspects** column across
  all three surfaces (the pair-card mapped-pairs table, the forward and
  backward Type coverage tables, and the per-type breakdown).  Hovering the
  badge opens the same-name-first explanation plus one line per rival
  (candidate name, own 1-to-1 pair, curated/auto votes, reverse target,
  rival-pair status, src/tgt neurons); the **collapsed Suspects block** below
  each table holds the identical facts as a table, so the hover is additive
  and never the only route to the evidence.  The viewer keeps its
  per-candidate expander.  A pair that did
  NOT fire explains itself where it surfaces: a held conflict/kept-unmapped
  line naming how many rivals lack their own 1-to-1 pairing.  The viewer's
  same-name annotation distinguishes three cases — curated cross-dataset
  relation, same-name-first selection, bare echo — instead of showing the
  bare-echo caveat for all three.

### Multi-value (comma-joined) `type` cells

Some releases annotate a neuron with several candidate types in ONE `type`
cell, in either of two encodings: plain comma-joined (MCNS `DNp51,DNpe019`,
BANC `LAL173,LAL174`) or parenthesized with an optional variant suffix
(BANC `(PLP191,PLP192)a` — the alternatives are inside the parens; the
suffix is not a candidate name).  In **crosswalk/annotation** cells the
parenthesized group is split with the suffix DISTRIBUTED to each alternative
(`'(AVLP346,AVLP348)a'` -> `AVLP346a`, `AVLP348a`), including the BANC
`auto:`-prefixed form; the `type` column itself is never split.  **BodyId-valued cells are dropped**
(a bodyId is not a type name): `hb1874217622`, `(hb5813083315)`,
`(5901212906)`, the paired `(hb…,hb…)` form and their `auto:`-prefixed
variants resolve to nothing instead of becoming pseudo-types; a NAME carrying
a bodyId annotation (`PS279(hb1499087543)`) keeps its base name.  The raw
cells remain on disk, so bodyId-level provenance is available to the
verification pipeline. The **raw cell stays
the atomic type name** — bodyId→primary cardinality is unchanged and no
mapping decision moves. The oddity is recorded and queryable so it is
visible rather than silent: `mapper.is_multivalue_type(name, dataset)` /
`mapper.multivalue_parts(name, dataset)` / `mapper.multivalue_summary(...)`,
a data-quality line in `user_warning_notes.txt`, a `🧩 multi` marker on the
report grid row, and `multivalue_source` / `source_parts` columns in
`auto_type_mapping_conflicts.csv`. Splitting the cells (registering each
part) would change bodyId→type cardinality and every consumer — that is a
separate, deliberately un-scheduled round
(plan-type-column-multivalue-normalization).

Use `mapper.get_mapping_conflicts(source_dataset, target_dataset,
source_type)` for a direction-scoped conflict, and
`mapper.is_n_to_1_type(type_name, dataset)` for the legacy broad check.

## Performance Considerations

- Type mapping lookups are O(1) hash table operations
- Mapping tables are loaded lazily on first use
- For large-scale batch comparisons, the mapper is passed once to avoid repeated loading
- Partner standardization adds minimal overhead (dict comprehension)

## Troubleshooting

### "Type mapper not loaded" warning

This means the neuron_df file wasn't found. Check:
1. File exists at `datasets/male-cns_v1_0/male-cns_v1_0_allneurons_neuron_df.csv`
   (the active default mapping source is male-cns **v1.0**; the retained
   `datasets/male-cns_v0_9/` table is auxiliary/legacy compatibility data,
   not the active source)
2. The workspace path is correctly configured

A failed load is observable and retryable: the mapper records the reason
on `last_load_error` (also reported by
`comparison.cross_dataset_type_mapper.get_type_mapper_state()`), and any
subsequent mapper call retries the load. Profile-comparison results carry
an `auto_type_mapping_*` metadata block in `parameters.json` (requested
vs active, source, version, load error, per-status resolution counts and
their `mapping_resolution_counts_basis` of `unique_type_resolutions`, the
separate occurrence-basis `mapping_partner_type_resolutions_by_status`
metric, and the raw-fallback flag) so a run that fell back to raw names can
never be mistaken for a valid-mapper run.

### Type not found in mapping

If a type has no cross-dataset mapping:
- The original type name is preserved (identity mapping); this long-tail
  fallback is counted in the result metadata (`raw_fallback_used` and the
  `unmapped` status counts)
- Types with a mapping CONFLICT (for example BANC `CB1011`) fail closed:
  they are excluded from automatic cross-dataset comparison and are never
  compared by raw same-name
- This is expected for types unique to one dataset

### Unexpected similarity scores

If cross-dataset similarity seems too low:
1. Check if key partner types have mappings
2. Consider that some connectivity differences are real biological variation
3. Use `use_auto_type_mapping=False` to see raw (unmapped) comparison

## See Also

- [Cross-Dataset Comparison](./core-features/CrossDatasetComparison_Guide.md) - Overview of cross-dataset analysis
- [Connectivity Profiling](./CONNECTIVITY_PROFILING.md) - Connectivity profile computation

---

## Threshold equivalence across datasets

Standard mode compares all datasets at the SAME threshold (horizontal
comparison); Custom combination mode lets one named query use a different
requested threshold in each dataset. Synapse-count conventions differ strongly between datasets —
the median number of synapses per neuron spans ~6x (BANC v626 ≈ 52 post,
FAFB v783 ≈ 308 post, male-cns v1.0 ≈ 340 post / 490 pre+post) — so "BANC
≥ 3" and "FAFB ≥ 3" do NOT cut the connectomes at comparable sparsities.
This section gives a rough, whole-dataset alignment. The threshold-alignment
files are raw-run density diagnostics; use the query manifest for the actual
per-query comparison rows.

### Criterion

**bodyId-level per-neuron connection-pair density**: the number of distinct
(presynaptic, postsynaptic) body pairs with weight ≥ t, divided by the
dataset's total neuron count. Unweighted — edge presence only, synapse
counts (weights) ignored.

Caveats:

- BANC local downloads are pre-truncated at weight ≥ 3 (thresholds 1–2 are
  no-ops on the pair counts).
- male-cns numbers come from the ~98% coverage connection cache.
- Whole-dataset values are a rough hint only. Real matching is
  query-specific — a given query's best-aligned thresholds can differ from
  the global rule by several units (this is exactly why every run exports
  its own alignment).

### Reference values (whole dataset, pairs per neuron)

The dated, maintained reference table lives in the
[Cross-Dataset Comparison guide](./core-features/CrossDatasetComparison_Guide.md)
(re-baselined 2026-09-07 on the refreshed 2026-09-04 BANC bucket tables).
Use that table for the current values; the headline calibration is:

- **τ = 3**: BANC v626/v888 (46.8/46.1 pairs per neuron) are directly
  comparable with FAFB @3 (47.3); male-cns sits ≈ 1.3x BANC (≈ male-cns @5).
- **τ ≥ 5**: the classic multipliers re-emerge and grow with τ — FAFB ≈ 2x
  BANC at τ=5 rising to ≈ 2.7–2.8x at τ=10; male-cns ≈ 2.7x at τ=5 rising
  to ≈ 3.8–3.9x at τ=10 (BANC @5 ≈ FAFB @8 ≈ male-cns @8).
- The pre-refresh numbers (BANC 23.2/19.2 @3; "FAFB ≈ 2.2–2.3x BANC,
  male-cns ≈ 2.8–3x BANC at every threshold"; "BANC lowest at every
  threshold") described the 2026-08 downloads and **no longer hold** — the
  table refresh roughly doubled BANC's τ=3 density. Do not reuse them.

### Where the per-query alignment comes from

Every cross-dataset run exports threshold-alignment files (spec Feature C):

- `comparison_results/threshold_alignment_best_matches.csv` — a bisection
  prober over each dataset's lowest-threshold extract finds the
  best-matching density threshold in every other dataset
  (extended range, not limited to the typed thresholds). Primary metric:
  edge-count distance `|n_a − n_b| / max(n_a, n_b, 1)`; tolerance ≤ 0.10.
- `comparison_results/threshold_alignment_matrix.csv` (+ heatmap) —
  pairwise metrics over the typed thresholds only.
- `comparison_results/edge_density_per_threshold.csv` and
  `comparison_visualizations/edge_density_threshold_curves.png` — the
  density curves behind the matching (absolute + per-neuron).

### Threshold query modes

Thresholds are edited in **Core Parameters**, not in Advanced Settings.
Standard N-chip input creates same-threshold queries (for example, `N=3` is
run at 3 in every selected dataset). Custom combination mode instead
use a dataset-column table in which each row is one complete query, for
example `BANC=3, FAFB=7, male-cns=8`. Every selected dataset must have a cell;
the table is not an independent threshold schedule for each dataset.

The sorted union of advanced cell values is only the deduplicated raw-run
schedule. Query alignment and similarity keep the row identity and never
substitute a scalar union value. The stable query/dataset join is exported as
`comparison_results/threshold_combinations.csv` and mirrored in the
`queries` block of `effective_thresholds.json`, including requested/applied
threshold, StrongestFirst budget and tau, Edge Budget `w0`/`w1`, `w2`, `W*`,
and `paths_complete`.

### Related run features

- **Duplicate-threshold skipping (Feature G)**: with a path budget, a run
  whose weakest emitted path has bottleneck τ produces the identical set
  for every threshold up to τ; later input thresholds ≤ τ are skipped and
  marked `skipped/duplicate_of` in `threshold_sensitivity.csv` (τ collapse).
- **Replay paths (Feature F)**: in path mode 'all' the path set is
  enumerated once at the lowest threshold; every higher threshold is
  materialized from the bottleneck-annotated path set (identical outputs,
  no re-enumeration). Disable via Advanced Settings ▸ Replay Paths.
