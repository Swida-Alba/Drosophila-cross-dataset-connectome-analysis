"""Per-run HTML report for the TM VEV pipeline (validate·expand·visualize).

Template: ``_plan/tmvev-run-report-template.md`` (DECIDED 2026-09-16).

The report renders ONE self-contained offline ``report.html`` in the run
folder (same conventions as the ``_UserGuide_please_read_me.html``
artifacts: inline CSS, zero external references, readable standalone).
Every rendered term carries a hover definition, each section has an
expandable definitions table, warnings are mirrored to the folder's
``user_warning_notes.txt`` in bracketed-tag style, and ``README.txt``
stays slim directions-only.

The generator reads ONLY the run folder's artifacts (the same files the
README documents), so any past run can be regenerated::

    python -m comparison.mapping_validation_report <run_dir> [--notes]

Section map (template §): §0 hero · §1+§2 Coverage tab · §3 Branches ·
§4 Targets · §5+§6 Fill · §7 Out-map · §8 Morph · §9 Scenes ·
§10+§11+§12 Log/Provenance.
"""

import ast
import collections
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:  # package import (usual pipeline path)
    from . import report_kit
    from .cross_dataset_type_mapper import basis_is_row_evidence
except ImportError:  # script / notebook path
    import report_kit  # type: ignore[no-redef]
    from cross_dataset_type_mapper import basis_is_row_evidence


def split_basis_buckets(basis_map: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Partition a basis-keyed bucket map into (row-evidence, name-asserted).

    Branches bucket under whatever basis they resolved to, so reading only
    the 'linker rows' and 'full population' keys would drop a side resolved
    through the release relation out of BOTH counts and shrink the in-map
    source population the Coverage and Targets tabs publish.
    """
    keys = sorted(basis_map or {})
    row = [k for k in keys if basis_is_row_evidence(k)]
    return row, [k for k in keys if k not in set(row)]

# ---------------------------------------------------------------------------
# glossary: term -> one-line definition (hover layer + per-section tables)
# ---------------------------------------------------------------------------

TERM_DEFS: Dict[str, str] = {
    'map-covered':
        'Count verb (decided D6): the target neuron is in a branch pool '
        'of the validated mapping. The L1 coverage level is still named '
        '"claim" per the approved fidelity plan.',
    'validation mode':
        'Ordered enum restrictive < family < aggressive; modes NEST — '
        'switching mode only admits more neurons, never relabels one. '
        '`pooling` is parallel to that ladder, not its top rung: it is the '
        'unsupervised engine, and a pooling run reports mode `pooling` '
        'with no nested bins.',
    'pooling':
        'The UNSUPERVISED candidate engine (`--mode pooling`): every neuron '
        'the query names is scanned against the WHOLE target universe, each '
        'source then keeps its top-N rows under the admission bar, and '
        'morphology gates those connectivity survivors. Nothing here is read '
        'off a branch pool, so the type mapper is compared with the result '
        'afterwards instead of deciding it.',
    'admission bar':
        'What decides which rows a pooling run keeps: each queried source '
        'takes the top-N of the chosen metric (`pooling_bar_metric` × '
        '`pooling_bar_top_n`), where `either` is the UNION of both metrics\' '
        'own top-N rather than a merged best-rank ordering — a merged order '
        'spends the slots on the two metrics\' rank-1 rows and loses the '
        'candidates the union recovers. Because `rank_union` ties across '
        'hundreds of targets, a rank cut is not a row bound, so each source '
        'holds at most 2 × N rows in the chain order and the number the cap '
        'cut is published. The absolute floors are NOT part of admission: '
        'they are flags (see `jaccard floor`).',
    'jaccard floor':
        'An ADVISORY FLAG on a pooling row, measured against the configured '
        '`pooling_jaccard_floor` and published as `below_jaccard_floor` — '
        'evaluated, exported, and applied to nothing (user 2026-09-24). It '
        'was a filter until a run showed the rank_union floor beside it, set '
        'to its own default 0, was starving 119 of 242 queried sources: a '
        'connectivity threshold says how wide the scan opened, never whether '
        'a pair is a homolog, which is the morphology gate\'s job. The '
        'configured value is still published so the flag is recomputable.',
    'pooling tier':
        'Pooling\'s OWN ladder, assigned per row and first-match down: '
        '`matched` (the row is some metric\'s top-1 for its source AND its '
        'rank_union clears the matched bar), `verified` (top-1 on either '
        'metric), `nominated` (ranks 2..N inside the bar). It shares two NAMES '
        'with the supervised tiers and none of their machinery: pooling has no '
        'branch pool to be a member of, so a pooling `matched` asserts a '
        'connectivity-and-morphology finding under this mode\'s own bar, not '
        'the mapper\'s mutual-best claim. There is no `borderline` and no '
        '`relative` here — the deep window and the type-mate bins belong to '
        'the nested modes.',
    'mapper_cell':
        'Post-hoc comparison of one pooling candidate with the supervised '
        'mapping: `confirmed` sits in a branch\'s refined target pool; '
        '`type_miss` is an unmapped neuron of a type the map does assert; '
        '`type_new` is of a type outside the map; `verified_only` names a '
        'target the supervised path graded verified that this absolute gate '
        'did not admit. No cell is a recall measure — the two engines '
        'admit on different quantities.',
    'morph_gate':
        'Pooling\'s last gate, on the connectivity survivors only. It is a '
        'GATE: a `scored` row below its bar leaves the exported pool and the '
        'scene root, and the count of those refusals is published as '
        '`targets refused`. `scored` carries the numbers of the pair the bar '
        'was applied to: `morph_bar_kind` names the rule, `morph_bar` is its '
        'binding value, and the score is `morph_pool_ref` for a native row '
        'and `morph_similarity` otherwise, so the verdict can always be '
        'recomputed from the row. `shared` means the verdict was made for the '
        'pair named in `verdict_for_pair`, not for this row — a borrowed '
        'number is never this row\'s own measurement. `inactive` means the AUC '
        'gate suspended morphology, `disabled` means the run asked for none '
        '(which pooling refuses outright), and the remaining absences are kept '
        'apart on purpose — `not-attempted-cap` (the morph budget refused to '
        'look; `morph.budget` names the rule that priced it), `no-score` (the '
        'scorer returned no value for the pair, so nothing was measured either '
        'way). Neither is a rejection, and each names a different reason the '
        'row stayed in the pool unscored.',
    'mutual-best (assigned)':
        'A source neuron whose best pool member is itself the target\'s '
        'best source — both sides read on the ordering chain, so the '
        'strict pairing counted as "assigned" is the same relation the '
        'ranked lists show.',
    'ordering chain':
        'How two candidate neurons are compared for "best": jaccard first, '
        'rank_union breaking a jaccard tie, bodyId last (user 2026-09-20, '
        'J1/J3). It decides the published top-1, the mutual-best pairing, a '
        'target\'s best source, gap-fill proposals and every hover list. '
        'The two rank columns are EVIDENCE the verdicts are read off, not '
        'the order: jaccard is a ratio of small integers, so a tie block '
        'shares one rank and would otherwise be decided by row order.',
    'verdict':
        'Tiered rule, read off the POOL rather than off one chosen row, with a '
        'rank-1 claim lifting only the row that PUBLISHES the partner holding '
        'it: verified_strong (ONE pool member is global top-1 under BOTH '
        'rank_union, positive, and jaccard), verified (that same published '
        'member is top-1 under at least one metric), borderline (the pool\'s '
        'best sits inside the top-5 window), unmatched. Quantifying over the '
        'pool keeps a label from moving with the ordering key; tying the '
        'claim to the published partner is what stops a verdict meaning "this '
        'pool contains a winner somewhere" — r17 measured 46 rows on two '
        'half-claims and r18 2 more on an unpublished rank_union win. Any '
        'rank_union top-1 that is not the published target stays visible in '
        'ru_top_target_bodyId, as evidence rather than verdict.',
    'matched':
        'The only ASSERTED tier: the pool member is the mutual best under '
        'both metrics.',
    'verified':
        'Review tier: the published pool member is global top-1 under at '
        'least one of rank_union / jaccard.',
    'borderline':
        'Review tier: the pool\'s best member sits inside the top-5 window.',
    'unmatched':
        'No pool member qualifies under the tier rule — counted as a '
        'miss, never as coverage.',
    'category':
        'Revision 3.12 partition; every in-scope target gets EXACTLY '
        'one, decided in this order: tier (matched > verified > '
        'borderline > unmatched) > sibling > candidates > family > '
        'relative > examinees.',
    'sibling':
        'An in-map target of the query in ANOTHER branch that appears '
        'in this branch\'s expansion — already mapped, never a fill.',
    'candidates':
        'Out-of-map suspect, connectivity-qualified AND morph-qualified — '
        'the restrictive fill. Connectivity is an invader ahead of the '
        'pool best, a gap fire, or (family mode and up) a neuron inside '
        "the candidate-discovery window, the top-rank_top_k of EACH "
        'metric: a pool holding both global rank-1s leaves the invader '
        'window empty, and since rule 5 seeds `relative` from the '
        'candidate types, the bar alone made a strong branch report less '
        'than a weak one (r16 -> r18: relatives 39 rows -> 1).',
    'family':
        'Out-map bodyIds of THIS branch\'s target type (family/aggressive '
        'modes; type-gated, not morph-gated).',
    'relative':
        'Type-mates of candidate types outside the map (family/aggressive '
        'modes).',
    'examinees':
        'The aggressive-only WIDE window: out-of-pool homologs ranked below '
        "the pool best, beyond `rank_top_k` (`candidate_source="
        "'deep_window'). Inside `rank_top_k` the same rows are tagged "
        "'top_window', count as connectivity evidence and land in "
        "`candidates`, so the two bands are also the advisory-vs-fill "
        "boundary. Renamed from 'suspicious' 2026-09-18 - the mapper's "
        'rival-suspects concept now owns that word.',
    'candidate_annotation':
        'Per-bodyId leaf token on every expansion bin, one ordered '
        'value: {T}(out-map) wins, then {T}>{src}, then {T}(no_source), '
        'then untyped; (dup) is a standalone trailing tag.',
    '{T}(out-map)':
        'The TYPE is an in-map type — an unmapped bodyId of a type '
        'already in the map (bodyId-level). The fill material; wins '
        'over the other tokens.',
    '{T}>{src}':
        'The type is NOT in-map but maps backward to a real source '
        'population (type-level).',
    '{T}(no_source)':
        'The type is NOT in-map with no usable backward route — a '
        'source-annotation gap, not a fillable homolog (type-level).',
    'untyped':
        'No type annotation. A leaf token on whatever bin the neuron '
        'earns, never a peer category.',
    'out of scope':
        'A connectivity-qualified suspect that FAILED the morph rule: '
        'in_scope=False, morph_failed=True, category blank, never '
        'rendered — the connectivity-only homolog result.',
    'dup':
        'A non-sibling bodyId labeled in more than one branch — '
        'cross-branch convergence, not double counting.',
    'gap':
        'min(source_pool, target_pool) - matched; fills fire when '
        'gap > 1. Per-branch gaps double-count cross-branch convergence.',
    'hole':
        'An OUT-MAP bodyId of an in-map TYPE — family material no branch '
        'pool claims AND no fill candidate reaches: the overhang still '
        'unexplained. BodyIds inline in the Targets tab.',
    'family material':
        'Out-map bodyIds of in-map types: the mapped types\' populations '
        'minus everything a branch pool map-covers (the `family` bin\'s '
        'population; holes are its candidate-unreached subset).',
    'fill levels':
        'Branch-level fill ranking: high / medium / low (evidence '
        'strength), type_gated (type qualifies, morph does not), advice '
        '(cross-type neighbors, never fills for single-type queries).',
    'restrictive fill':
        'The bodyId-unique candidates bin (gap_fill_dedup.csv, '
        'counts_toward_restrictive_fill=True). Proposals only.',
    'family fill':
        'candidates + family + relative bodyIds '
        '(counts_toward_family_fill=True) — the wider fill surface.',
    'out-map expansion':
        'Top-k typed non-in-map candidates per UNCLAIMED source (a '
        'source with no branch anywhere), morph-checked against the run '
        'null bar — mapper-gap evidence, not fills.',
    'branch bar':
        'Floors v3 admission rule: candidate = native matched+verified '
        'floor (≥2 scored refs) else Track-A backup floor B_b − Δ else '
        'run null p95; suspicious (aggressive) = B_b − k·Δ (null p50 '
        'fallback). Kinds: native / track_a_backup / null.',
    'native floor':
        'Mean pairwise reference similarity (matched+verified pool '
        'refs, native Track B) minus the native margin.',
    'Track-A backup floor':
        'Branch Track-A baseline B_b minus Δ (morph_track_a_offset); '
        'used when the branch has < 2 scored native references.',
    'null bar':
        'p95 of the run\'s jaccard ≤ 0.05 similarity window — the '
        'no-homology backdrop every morph admission is checked against.',
    'AUC gate':
        'Calibration check on the morph rule (verified vs examinee '
        'distributions). Below the 0.65 floor the gate is INACTIVE — '
        'informational only, bars stay binding.',
    'score frames':
        'Track A scores in the TARGET render space; Track B (native '
        'reference) in TARGET native space; scenes render in SOURCE '
        'coordinates — anatomy, never the scoring frame.',
    'pool_ref tier':
        'The native reference set is matched-only when a branch has '
        '≥2 matched neurons; verified joins as an explicitly-flagged '
        'compromise otherwise.',
    'noise gates':
        'Rows moved to noise_filtered_candidates.csv: spatial caliber '
        '< 0.1 × branch pool best (primary), rank_union ≤ 0, jaccard '
        '< 0.5 × pool best, rank_union margin < 0.02 (tie).',
    'scene self-check':
        'Debug checks: every legend leaf\'s rendered geometry matches '
        'its neuron (identity), every expected neuron is plotted and '
        'rendered (population), and under `line` mode every neuron leaf '
        'carries a centerline (geometry census) — the CSV and the '
        'picture cannot disagree.',
    'source-matched':
        'Backward status (advisory, D-B11): this in-branch source is the '
        'column-best source of its own row-best pool target, with pair '
        'rank_union above the matched bar. Informational — targets remain '
        'the validated entities.',
    'source-verified':
        'Backward status (advisory): this source is the column-best '
        'source of some pool target (any metric), or sits in a column '
        'whose top sources are all in-branch. Not an assertion.',
    'source-borderline':
        'Backward status (advisory): the source is not column-top-1 '
        'anywhere, but only a few out-of-branch sources rank above it in '
        'its best column.',
    'source-unmatched':
        'Backward status (advisory): dominated by out-of-branch sources '
        'in every column it appears in, or no ranked rows into the '
        'branch pool. Often a convergence duplicate of an '
        'already-matched/verified target, not a mapping failure.',
    'source-candidates':
        'Backward candidates (advisory, visualized): out-of-branch '
        'sources whose qualified sibling rows point into this branch\'s '
        'pool — convergence information, never a fill. Qualification is '
        'the owning branch\'s bar (v1; target-branch re-evaluation is a '
        'documented refinement).',
    'coverage levels':
        'L1 claim (what the map covers) · L2 provenance (how each '
        'covered neuron was earned) · L3 validation (what was checked) '
        '— reported in every TM VEV report.',
    'linker rows':
        'Row evidence for one side of a pool: a linker column of the chain '
        'supplying that side named the neurons, so the pool is the bridging '
        'subset rather than the whole type population. Each side is supplied '
        'separately, so the source side can be named by a different chain '
        'than the selected (target-side) one — the Branches detail table '
        'shows both.',
    'full population':
        'That side of the pool is the WHOLE type population. Either no '
        'supported bridge chain exists, or the selected chain carries no '
        'linker column on this side (a target-side-only chain, e.g. '
        'FAFB->BANC through the banc_v888 `fafb_cell_type` hop) — the '
        'other side can still be linker-refined. A same-name pair is always '
        'like this on the source side, because FAFB annotates neurons with '
        'OTHER datasets\' names and never with its own `type`: the wide pool '
        'is then the correct terminal answer, not an unresolved gap.',
    'evidence_only':
        'Mapper N-to-1 convergence view: several source types map onto '
        'one target; branches without a supported bridge chain are '
        'dropped rather than validated against full populations.',
    'gap triggered':
        'The branch fired the gap rule. Advisory only — the trigger no '
        'longer gates anything (retired); proposals, never rewrites.',
    'verified_strong':
        'The source ranks the same pool target top-1 under BOTH metrics '
        '(positive rank_union AND jaccard) — the strongest verdict.',
    'forward':
        'The validation direction: source neurons are scanned and the '
        'target-side pool members earn the verdict tiers. Branches, '
        'Targets and Coverage report the forward view.',
    'backward':
        'The interpretive direction: what the receiving target types and '
        'the in-branch sources look like from the other side (column '
        'view of the same pair scores). Informational — never gates.',
    'examinee rows':
        'Non-pool neurons ranked ahead of the best pool member (the '
        'aggressive deep window adds more). Exported in examinees.csv; '
        "renamed from 'suspicious' 2026-09-18.",
    'reciprocal':
        'Stage 5d evidence (advisory): a candidates / family / relative '
        'member — or an UNMATCHED pool member — is scanned BACK in the '
        'source universe — the same '
        'homolog-finding scorer, run on the member\'s own target profile. '
        'Matched / verified / borderline pool members are not scanned: '
        'they are already mapped, and the symmetric forward score is their '
        'evidence. '
        'Connectivity only; morphology is not re-evaluated (candidates '
        'are already morph-qualified, and family / relative members are '
        'morph-similar to the query or to those candidates). It never '
        'moves a neuron between bins and never counts toward a fill. Not '
        'the Backward tab: that one is the type-level mapping direction, '
        'this one is per-neuron reverse connectivity.',
    'reciprocal top-1':
        'The chain-best source-side hit for one member — Jaccard first, '
        'rank_union as the tie-break, bodyId last (user 2026-09-20, J1): '
        'its bodyId, source type, where it lives '
        "(this branch's pool, or elsewhere — out-of-branch, whatever "
        'tier it would belong to) and its jaccard. rank_union travels in '
        'the hover as the confirmation value and never filters a row. '
        'Hover for the full top-N list with both scores and both ranks.',
    'branch-type hit':
        'The reverse hit the reciprocal grade actually rests on — the '
        'best-ranked source of the CLAIMING BRANCH\'S OWN type, with the '
        'rank it reached and which ranking (rank_union or jaccard) placed '
        'it there. It is a different neuron from the top-1 whenever the '
        'branch type wins on jaccard alone, which is why a row can read '
        'high while its top-1 sits elsewhere. Its jaccard is also what '
        'orders the Reciprocal list, so a row with no hit ranks below the '
        'rows that have one.',
    'rank_union':
        'Rank agreement between the two partner-strength vectors over the '
        'UNION of their partner types: each side is average-tie ranked, a '
        'type one neuron lacks is ranked as weight 0, and the score is the '
        'Pearson correlation of the two rank lists. Symmetric by '
        'construction, so a forward and a reverse pass cannot disagree about '
        'one pair. It reads blank when the union holds fewer than 3 types or '
        'one side is constant — no monotone information, not a zero score. '
        'A disjoint pair sits near -0.8, not 0, so 0 is mid-scale rather than '
        '"nothing"; see the shared/union column for what it rests on.',
    'high':
        'Reciprocal grade: a hit of the member\'s OWN branch source type '
        'is the top-1 reverse hit by rank_union or by jaccard — wherever '
        'that hit lives. Pure rank evidence, no score bar; advisory only.',
    'medium':
        'Reciprocal grade: the branch\'s own source type appears within '
        'the top-3 of the rank_union or the jaccard ranking (but is not a '
        'top-1). Advisory only — it never gates anything.',
    'low':
        'Reciprocal grade: scanned, but the branch\'s own source type '
        'ranked outside the top-3 of both rankings (or nothing usable '
        'ranked at all). A graded negative, not an error.',
    'not-checked':
        'Reciprocal label for members the pass did not scan: the pass is '
        'off, the neuron was over the per-run / per-branch caps, or it is '
        'outside the scanned set — an already-mapped (matched / verified / '
        'borderline) pool member, or a bin the pass does not read. '
        'Displayed as an explicit dash so absence is never read as failure.',
    'shared partner types':
        'How many partner types the two compared vectors have in common — '
        'the denominator a rank_union or jaccard is actually built on. '
        'The union (the second number) is what rank_union ranks; the '
        'shared count is the part carrying real evidence, since a type one '
        'side lacks is scored 0.0. Same neurons, ~12+ shared types, is a '
        'different claim from 2.',
    'thin evidence':
        'A reciprocal hit whose two vectors share at most 3 partner types. '
        'rank_union can be high on such a pair simply because there was '
        'almost nothing to rank — an incompletely traced neuron scores well '
        'against another short one. Advisory marker only: it never changes '
        'the reciprocal verdict, a bar, or a fill count.',
    'homolog forward':
        'One row per source bodyId the run touched — assigned, '
        'fill-proposed, out-of-map or unpaired alike — with the target '
        'neuron the connectivity scan matched it to.',
    'homolog backward':
        'One row per target bodyId that appeared in the run, scanned back '
        'against the WHOLE source dataset — the exact mirror of the '
        'forward panel.',
    'primary match':
        'The published chain-best hit of the per-bodyId scan (Jaccard '
        'first, rank_union as the tie-break, bodyId last) — the same '
        'definition the reciprocal top-1 column publishes.',
    'union window':
        'The hover list is the UNION of the top-3 hits by rank_union and '
        'the top-3 by jaccard (deduped, listed in chain order), so a hit '
        'that is rank-1 on one metric while invisible on the other still '
        'shows.',
    'allocation':
        "Every bin/status the run's artifacts give this bodyId — category "
        'bins (sibling/candidates/family/relative/examinees), '
        'fill-proposed, out-of-map, the mapping verdict and the backward '
        'source status — so one row answers "where does this neuron live '
        'today".',
    'not scored':
        'The run exported no morphology for this pair. The homolog panels '
        'display the run\'s own morph exports and never re-score at '
        'report time.',
    'no profile':
        'The neuron had no usable connectivity profile, so its scan was '
        'skipped — silence with a reason, never a negative result.',
}

# per-file column/term notes for §12 expanders
FILE_GLOSSARY: Dict[str, List[str]] = {
    'validation/validation_results.csv': [
        'verdict', 'ordering chain', 'matched', 'verified', 'borderline',
        'unmatched'],
    'validation/examinees.csv': [
        'category', 'sibling', 'candidates', 'family', 'relative',
        'out of scope', 'candidate_annotation', 'reciprocal',
        'shared partner types', 'thin evidence'],
    'expansion/backward_matches.csv': [
        'reciprocal', 'reciprocal top-1', 'branch-type hit', 'high',
        'medium', 'low', 'not-checked', 'shared partner types',
        'thin evidence'],
    'validation/forward_matches.csv': [
        'homolog forward', 'primary match', 'union window', 'allocation',
        'not scored', 'no profile'],
    'expansion/target_matches.csv': [
        'homolog backward', 'primary match', 'union window', 'not scored',
        'no profile'],
    'gap_fill/gap_fill_dedup.csv': [
        'dup', 'restrictive fill', 'family fill', 'branch-type hit',
        'thin evidence'],
    'gap_fill/gap_fill_levels.csv': ['fill levels'],
    'expansion/out_map_expansion.csv': ['out-map expansion', 'null bar'],
    'pooling/pooling_candidates.csv': [
        'pooling', 'ordering chain', 'rank_union', 'candidate_annotation',
        'out of scope', 'morph_gate', 'branch bar', 'mapper_cell',
        'jaccard floor'],
    'pooling/pooling_pool.csv': ['dup', 'mapper_cell', 'morph_gate'],
    'pooling/pooling_sources.csv': ['pooling', 'admission bar', 'morph_gate'],
    'pooling/pooling_cross_validation.json': ['pooling', 'mapper_cell',
                                              'morph_gate'],
    'validation/pair_summary.csv': ['gap', 'verdict'],
    'validation/pool_categories.csv': ['pool_ref tier'],
    'morphology_calibration.json': [
        'branch bar', 'native floor', 'Track-A backup floor', 'null bar',
        'AUC gate', 'score frames'],
    'set_coverage.json': ['map-covered', 'hole', 'family material',
                          'coverage levels'],
}

ARTIFACT_LINES: List[Tuple[str, str]] = [
    ('mapping/mapping_export.csv',
     'branch-level mapping (chains, linker values, bodyId pools)'),
    ('validation/pair_summary.csv', 'per-branch pools / gap / verdicts (§3)'),
    ('validation/pool_categories.csv',
     'per in-map target tier + best evidence'),
    ('validation/validation_results.csv',
     'source×branch verdict rows (§1)'),
    ('validation/examinees.csv',
     'expansion bins, Revision 3.12 categories (§5); renamed from '
     'suspicious_candidates.csv'),
    ('validation/noise_filtered_candidates.csv',
     'dropped rows + noise_reason'),
    ('validation/deep_candidates.csv', 'aggressive-only deep window'),
    ('gap_fill/gap_fill_proposals.csv',
     'proposals, side × fill_class (§6)'),
    ('gap_fill/gap_fill_levels.csv', 'branch-level fill level (§6)'),
    ('gap_fill/gap_fill_dedup.csv', 'bodyId-unique fill (§6)'),
    ('expansion/family_candidates.csv', 'the family bin (§5)'),
    ('expansion/relatives.csv', 'the relative bin (§5)'),
    ('expansion/out_map_expansion.csv',
     'unclaimed-source expansion (§7)'),
    ('expansion/backward_matches.csv',
     'reciprocal homolog evidence per candidates/family/relative member '
     'and per unmatched pool member (§5d, advisory)'),
    ('validation/forward_matches.csv',
     'Homolog · forward: one row per appeared source bodyId — chain-best '
     'target + top-3 rank_union ∪ top-3 jaccard payload (display only)'),
    ('expansion/target_matches.csv',
     'Homolog · backward: one row per appeared target bodyId scanned back '
     'against the whole source dataset (display only)'),
    ('expansion/source_status.csv',
     'backward `source-` status per in-branch source (advisory)'),
    ('expansion/source_candidates.csv',
     'out-of-branch sources pointing into each branch pool, tagged '
     'in-map/out-map (advisory)'),
    ('pooling/pooling_candidates.csv',
     'POOLING mode: every (source, target) pair the admission bar kept, with '
     'the morph verdict and the post-hoc mapper cell; includes the targets the '
     'morphology gate refused'),
    ('pooling/pooling_sources.csv',
     'POOLING mode: one row per QUERIED source — its chain-best finding, that '
     'row\'s tier, how many targets it admitted vs kept, and no_finding when '
     'the bar admitted nothing'),
    ('pooling/pooling_pool.csv',
     'POOLING mode: one row per candidate target neuron, represented by its '
     'chain-best SURVIVING row; `in_pool=False` marks a target the last gate '
     'refused on every row, and `verified_only` rows are the mapper\'s alone'),
    ('pooling/pooling_cross_validation.json',
     'POOLING mode: the unsupervised-vs-mapper comparison, the morph '
     'record including how many targets the bar refused, and the '
     'input fingerprint'),
    ('mapping/same_name_excluded.csv',
     'queried types held/excluded by the same-name-first rule, or '
     'multi-value cells (advisory accounting)'),
    ('mapping/disclosure_evidence.csv',
     'the three-tier disclosure ends — bridge evidence the decision '
     'declined, with decline reason and advisory verification '
     '(2026-09-27; written when any were recorded)'),
    ('mapping/suspects_verification.csv',
     'rival-suspect connectivity verification (opt-in, advisory)'),
    ('set_coverage.json', 'set-level coverage (§1, §4)'),
    ('morphology_calibration.json',
     'branch bars, null bar, AUC gate, score frames (§8)'),
    ('parameters.json', 'full run parameters (§11)'),
    ('pipeline_progress.jsonl', 'stage timeline events (§11)'),
]

TRUE_STR = {'true', '1', 'yes'}
#: Pooling's per-row ladder, ordered best-first.  The report stays standalone
#: (it regenerates any past folder from the folder alone, so it imports no
#: pipeline module); this mirrors
#: `mapping_validation_pooling.POOLING_TIERS`, and
#: `test_pooling_tab_headlines_the_source_axis` pins the two against the
#: vocabulary a real run exports.
POOLING_TIERS = ('matched', 'verified', 'nominated')


# ---------------------------------------------------------------------------
# small readers / parsers
# ---------------------------------------------------------------------------

def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return None


def _run_file(run_dir, name: str) -> Path:
    """Locate one exported artifact inside a run folder.

    The writer files evidence CSVs under category subfolders
    (``validation/`` / ``expansion/`` / ``gap_fill/`` / ``mapping/``) and
    keeps the report, guide and parameter/meta at the root; older runs are
    flat.  File names are unique across the layout, so a one-level search
    resolves both without this module importing the pipeline's layout
    registry — which keeps the report standalone and able to regenerate any
    past run from its folder alone."""
    root = Path(run_dir)
    flat = root / name
    if flat.exists():
        return flat
    try:
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            cand = sub / name
            if cand.exists():
                return cand
    except OSError:
        pass
    return flat


def _run_file_rel(run_dir, name: str) -> str:
    """The run-relative display path of an artifact (``gap_fill/x.csv``)."""
    p = _run_file(run_dir, name)
    try:
        return p.relative_to(Path(run_dir)).as_posix()
    except ValueError:
        return name


def _read_csv_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    try:
        with open(path, newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    except Exception:  # noqa: BLE001
        return []


def _read_examinees(run_dir: Path) -> List[Dict]:
    """The expansion-bin rows: examinees.csv, falling back to the
    pre-rename suspicious_candidates.csv so older run folders keep
    regenerating (rename 2026-09-18)."""
    p = _run_file(run_dir, 'examinees.csv')
    if not p.exists():
        p = _run_file(run_dir, 'suspicious_candidates.csv')
    return _read_csv_rows(p)


def _n_rows(path: Path) -> Optional[int]:
    """Data-row count for the file index; None for JSON / missing."""
    if not path.exists() or path.suffix == '.json':
        return None
    try:
        with open(path, newline='', encoding='utf-8') as f:
            return max(sum(1 for _ in csv.reader(f)) - 1, 0)
    except Exception:  # noqa: BLE001
        return None


def _truthy(v) -> bool:
    return str(v).strip().lower() in TRUE_STR


def _f(v, nd=3) -> str:
    try:
        return f'{float(v):.{nd}f}'
    except (TypeError, ValueError):
        return '—'


def _chain_key(r: Dict):
    """Ascending sort key for the ordering chain of a run-CSV row — Jaccard
    rank, rank_union rank as the tie-break, then the two scores (higher
    first), blank last.  The authority is ``body_id_resolver.chain_key``;
    re-spelled here because this module reads finished runs with no
    pipeline import (it must stay runnable on a folder of CSVs alone)."""
    def rank(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return float('inf')
        return f if f == f else float('inf')

    def score(v):
        f = rank(v)
        return -f if f != float('inf') else float('inf')

    return (rank(r.get('jaccard_rank')), rank(r.get('rank_union_rank')),
            score(r.get('jaccard')), score(r.get('rank_union')))


def _pct(part, whole) -> str:
    try:
        whole = float(whole)
        return f'{float(part) / whole:.1%}' if whole else '—'
    except (TypeError, ZeroDivisionError, ValueError):
        return '—'


def _parse_readme(run_dir: Path) -> Dict:
    """Parse the stable README.txt blocks: run log, self-check lines,
    ``!`` warning lines, and (legacy slim-README runs) the mapper-gap
    evidence block."""
    out = {'log_lines': [], 'selfcheck_pass': [], 'selfcheck_fail': [],
           'bang_lines': [], 'mapper_gap': {}, 'mapper_gap_untyped': 0}
    path = Path(run_dir) / 'README.txt'
    if not path.exists():
        return out
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except Exception:  # noqa: BLE001
        return out
    in_log = in_gap = False
    for ln in lines:
        s = ln.strip()
        if s == 'Run log:':
            in_log, in_gap = True, False
            continue
        if in_log and (s == 'Pair summaries:'
                       or s.startswith('SET-LEVEL COVERAGE')
                       or s.startswith('Mapper-gap evidence')
                       or s.startswith('Morphology calibration:')):
            in_log = False
        if s.startswith('SET-LEVEL COVERAGE'):
            in_gap = False
            continue
        if s.startswith('Mapper-gap evidence'):
            in_log, in_gap = False, True
            continue
        if s.startswith('Morphology calibration:'):
            in_log = in_gap = False
            continue
        if in_gap:
            if not s:            # blank line ends the block
                in_gap = False
                continue
            m = re.match(r'\(untyped\):\s*(\d+) row', s)
            if m:
                out['mapper_gap_untyped'] = int(m.group(1))
                continue
            m = re.match(r'(.+?):\s*(\d+) row', s)
            if m:
                out['mapper_gap'][m.group(1).strip()] = int(m.group(2))
            continue             # prose continuation lines are ignored
        if in_log:
            out['log_lines'].append(ln)
            if s.startswith('! self-check ['):
                out['selfcheck_fail'].append(s)
                out['bang_lines'].append(s)
            elif s.startswith('self-check ['):
                out['selfcheck_pass'].append(s)
            elif s.startswith('!'):
                out['bang_lines'].append(s)
    return out


def _parse_scenes(run_dir: Path) -> List[Dict]:
    """One entry per plot-3d scene folder: type + png + entry html, read
    from the actual file names inside the folder (folder names are
    sanitizer-mangled — e.g. ``LNd_CRY-`` becomes ``LNd_CRY`` — so the
    type comes from the ``branches_*.html`` stem, which is also the key
    the README self-check lines use)."""
    vis = Path(run_dir) / 'visualization'
    if not vis.is_dir():
        return []
    scenes = []
    for folder in sorted(vis.iterdir()):
        if not folder.is_dir() or not folder.name.startswith('plot-3d_'):
            continue
        htmls = sorted(folder.glob('branches_*.html'))
        if not htmls:
            continue
        stem = htmls[0].stem[len('branches_'):]
        png = folder / f'branches_{stem}.png'
        scenes.append({
            'type': stem,
            'png': f'visualization/{folder.name}/{png.name}'
                   if png.exists() else None,
            'html': f'visualization/{folder.name}/{htmls[0].name}',
        })
    return scenes


# ---------------------------------------------------------------------------
# data collection: every section's numbers, artifact-bound
# ---------------------------------------------------------------------------

def collect_run_data(run_dir: Path,
                     mapper_gap_types: Optional[Dict[str, int]] = None,
                     mapper_gap_untyped: int = 0) -> Dict:
    run_dir = Path(run_dir)
    params = _read_json(_run_file(run_dir, 'parameters.json')) or {}
    calib = _read_json(
        _run_file(run_dir, 'morphology_calibration.json')) or {}
    coverage = _read_json(_run_file(run_dir, 'set_coverage.json')) or {}
    progress: List[Dict] = []
    pp = _run_file(run_dir, 'pipeline_progress.jsonl')
    if pp.exists():
        try:
            for ln in pp.read_text(encoding='utf-8').splitlines():
                ln = ln.strip()
                if ln:
                    try:
                        progress.append(json.loads(ln))
                    except ValueError:
                        pass
        except Exception:  # noqa: BLE001
            pass
    readme = _parse_readme(run_dir)

    # mapper-gap evidence: in-memory pass-through → set_coverage.json
    # (new runs) → README block (legacy runs)
    gap = (dict(mapper_gap_types) if mapper_gap_types is not None
           else (coverage.get('mapper_gap') or {}).get('types'))
    if gap is None:
        gap = readme['mapper_gap']
    gap_untyped = (mapper_gap_untyped
                   or int((coverage.get('mapper_gap') or {})
                          .get('untyped_rows') or 0)
                   or readme['mapper_gap_untyped'] or 0)

    val_rows = _read_csv_rows(_run_file(run_dir, 'validation_results.csv'))
    sus_rows = _read_examinees(run_dir)
    dedup_rows = _read_csv_rows(
        _run_file(run_dir, 'gap_fill_dedup.csv'))
    levels_rows = _read_csv_rows(
        _run_file(run_dir, 'gap_fill_levels.csv'))
    prop_rows = _read_csv_rows(
        _run_file(run_dir, 'gap_fill_proposals.csv'))
    pair_rows = _read_csv_rows(_run_file(run_dir, 'pair_summary.csv'))
    # One name for one quantity: the mutual-best 1:1 PAIR count was published as
    # `matched`, the same word the asserted tier carries (verified AND
    # rank_union > matched_ru_min, `mapping_validation.py`), so the column is now
    # `best`.  A run archived before the rename still carries the old header, and
    # this report is documented as regenerable over any run folder — so read it
    # under the new name rather than rendering an empty cell for old runs.
    for _r in pair_rows:
        if 'best' not in _r and 'matched' in _r:
            _r['best'] = _r['matched']
    out_rows = _read_csv_rows(_run_file(run_dir, 'out_map_expansion.csv'))
    fam_rows = _read_csv_rows(
        _run_file(run_dir, 'family_candidates.csv'))
    rel_rows = _read_csv_rows(_run_file(run_dir, 'relatives.csv'))
    same_name_excluded = _read_csv_rows(
        _run_file(run_dir, 'same_name_excluded.csv'))
    suspects_rows = _read_csv_rows(
        _run_file(run_dir, 'suspects_verification.csv'))
    disclosure_rows = _read_csv_rows(
        _run_file(run_dir, 'disclosure_evidence.csv'))

    # -- pooling mode: the unsupervised pool and its post-hoc comparison ----
    pooling_xval = _read_json(
        _run_file(run_dir, 'pooling_cross_validation.json')) or {}
    pooling_pool = _read_csv_rows(_run_file(run_dir, 'pooling_pool.csv'))
    pooling_rows = _read_csv_rows(_run_file(run_dir, 'pooling_candidates.csv'))
    # the mode's own unit: one row per queried source, so a source that found
    # nothing is present with an empty finding rather than missing
    pooling_sources = _read_csv_rows(_run_file(run_dir, 'pooling_sources.csv'))

    # -- stage 5d backward homolog evidence (advisory; connectivity only) --
    backward_rows = _read_csv_rows(_run_file(run_dir, 'backward_matches.csv'))
    _bev_rank = {'low': 1, 'medium': 2, 'high': 3}
    _DEDUP_DISPLAY_RANK = {'matched': 9, 'verified': 8, 'borderline': 7,
                           'unmatched': 6, 'sibling': 5, 'candidates': 4,
                           'family': 3, 'relative': 2, 'examinees': 1}

    def _member_row_key(r: Dict) -> Tuple[int, int]:
        # a bodyId scanned under several branches reads ONCE: the
        # strongest grade wins, ties go to the higher dedup rank (the
        # gap_fill_dedup precedence) so a candidates+family member has
        # one home bin
        return (-_bev_rank.get(
                    str(r.get('backward_evidence') or 'not-checked'), 0),
                -_DEDUP_DISPLAY_RANK.get(
                    str(r.get('member_category') or ''), 0))

    backward_by_bid: Dict[str, Dict] = {}
    rows_by_bid: Dict[str, List[Dict]] = collections.defaultdict(list)
    for r in backward_rows:
        bid = str(r.get('member_bodyId') or '')
        if not bid:
            continue
        rows_by_bid[bid].append(r)
        cur = backward_by_bid.get(bid)
        if cur is None or _member_row_key(r) < _member_row_key(cur):
            backward_by_bid[bid] = r
    backward_bins: Dict[str, Dict[str, int]] = {}
    for bid, best in backward_by_bid.items():
        cat = str(best.get('member_category') or 'unknown')
        lab = str(best.get('backward_evidence') or 'not-checked')
        b = backward_bins.setdefault(cat, {'high': 0, 'medium': 0,
                                           'low': 0, 'not-checked': 0,
                                           'neurons': 0})
        b['neurons'] += 1
        if lab in b:
            b[lab] += 1

    # -- source side -------------------------------------------------------
    scanned_sources = sorted({r.get('source_bodyId', '')
                              for r in val_rows
                              if r.get('source_bodyId')})
    best_verdict: Dict[str, str] = {}
    rank = {'verified_strong': 0, 'verified': 1, 'borderline': 2,
            'unmatched': 3}
    for r in val_rows:
        b = r.get('source_bodyId', '')
        v = r.get('verdict', '')
        if b and (b not in best_verdict
                  or rank.get(v, 9) < rank.get(best_verdict[b], 9)):
            best_verdict[b] = v
    verdict_counts: Dict[str, int] = {}
    for v in best_verdict.values():
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    # -- expansion bins ------------------------------------------------------
    cat_counts: Dict[str, int] = {}
    for r in sus_rows:
        c = r.get('category', '') or 'out of scope'
        cat_counts[c] = cat_counts.get(c, 0) + 1
    n_out_of_scope = sum(1 for r in sus_rows if r.get('category', '') == ''
                         or not _truthy(r.get('in_scope', True)))
    cand_ann: Dict[str, int] = {}
    cand_bin_ids = set()
    sus_cand_by_bid: Dict[str, Dict] = {}
    cand_break = {
        'in_family': {'rows': 0, 'bids': set(), 'types': collections.Counter()},
        'no_source': {'rows': 0, 'bids': set(), 'types': collections.Counter()},
        'backward': {'rows': 0, 'bids': set(), 'to': collections.Counter()},
        'untyped': {'rows': 0, 'bids': set()},
    }
    for r in sus_rows:
        if r.get('category') != 'candidates':
            continue
        bid = r.get('ahead_target_bodyId', '')
        cand_bin_ids.add(bid)
        sus_cand_by_bid.setdefault(bid, r)
        tok = r.get('candidate_annotation', '') or '?'
        cand_ann[tok] = cand_ann.get(tok, 0) + 1
        # structured candidates breakdown (user 2026-09-17): in-family vs
        # no_source orphans vs backward-mapped, by the ordered leaf token
        if tok.endswith('(out-map)'):
            br = cand_break['in_family']
            br['types'][tok.split('(')[0].strip()] += 1
        elif '(no_source)' in tok:
            br = cand_break['no_source']
            br['types'][tok.split('(')[0].strip()] += 1
        elif '>' in tok:
            br = cand_break['backward']
            br['to'][tok.split('>', 1)[1].strip()] += 1
        elif 'untyped' in tok:
            br = cand_break['untyped']
        else:
            br = None
        if br is not None:
            br['rows'] += 1
            br['bids'].add(bid)

    # -- fill (dedup) ---------------------------------------------------------
    dedup_cat: Dict[str, int] = {}
    for r in dedup_rows:
        c = r.get('dedup_category', '')
        dedup_cat[c] = dedup_cat.get(c, 0) + 1
    n_dup = sum(1 for r in dedup_rows if _truthy(r.get('dup')))
    restrictive = [r for r in dedup_rows
                   if _truthy(r.get('counts_toward_restrictive_fill'))]
    family_fill = [r for r in dedup_rows
                   if _truthy(r.get('counts_toward_family_fill'))
                   and not _truthy(r.get('counts_toward_restrictive_fill'))]
    levels_by_bid = {r.get('target_bodyId'): r for r in levels_rows}
    prop_by_bid: Dict[str, Dict] = {}
    for r in prop_rows:
        prop_by_bid.setdefault(r.get('proposal_bodyId', ''), r)
    level_counts: Dict[str, int] = {}
    for r in levels_rows:
        level_counts[r.get('level', '')] = \
            level_counts.get(r.get('level', ''), 0) + 1
    fill_class_counts: Dict[str, int] = {}
    side_counts: Dict[str, int] = {}
    for r in prop_rows:
        fill_class_counts[r.get('fill_class', '')] = \
            fill_class_counts.get(r.get('fill_class', ''), 0) + 1
        side_counts[r.get('side', '')] = \
            side_counts.get(r.get('side', ''), 0) + 1

    # -- out-map ---------------------------------------------------------------
    out_sources = sorted({r.get('source_bodyId', '') for r in out_rows
                          if r.get('source_bodyId')})
    out_morph_pass = sum(1 for r in out_rows
                         if _truthy(r.get('morph_qualified')))
    out_type_counts: Dict[str, int] = {}
    for r in out_rows:
        t = r.get('target_type', '')
        out_type_counts[t] = out_type_counts.get(t, 0) + 1
    per_source_best = []
    for src in out_sources:
        rows = [r for r in out_rows if r.get('source_bodyId') == src]
        if not rows:
            continue

        best = min(rows, key=_chain_key)
        n_q = sum(1 for r in rows if _truthy(r.get('morph_qualified')))
        per_source_best.append({'src': src, 'row': best, 'n_q': n_q,
                                'n_total': len(rows)})

    # -- morphology --------------------------------------------------------------
    branch_bars = calib.get('branch_bars') or {}
    bar_kind_counts: Dict[str, int] = {}
    susp_kind_counts: Dict[str, int] = {}
    for spec in branch_bars.values():
        k = str(spec.get('candidate_kind', '?'))
        bar_kind_counts[k] = bar_kind_counts.get(k, 0) + 1
        k = str(spec.get('suspicious_kind', '?'))
        susp_kind_counts[k] = susp_kind_counts.get(k, 0) + 1
    null_used = any(k.startswith('null')
                    for k in list(bar_kind_counts) + list(susp_kind_counts))

    ss_rows = _read_csv_rows(_run_file(run_dir, 'source_status.csv'))
    ss_counts = collections.Counter(r['status'] for r in ss_rows)
    per_type_status: Dict[str, Dict[str, int]] = {}
    for r in ss_rows:
        per_type_status.setdefault(
            r['source_type'], collections.Counter())[r['status']] += 1

    # -- homolog panels (user 2026-09-26): per-bodyId forward/backward ------
    forward_rows = _read_csv_rows(_run_file(run_dir, 'forward_matches.csv'))
    forward_available = _run_file(run_dir, 'forward_matches.csv').exists()
    target_match_rows = _read_csv_rows(
        _run_file(run_dir, 'target_matches.csv'))
    target_match_available = _run_file(
        run_dir, 'target_matches.csv').exists()
    # the pipeline labels unannotated neurons `untyped` (mapping_validation
    # _type_label); run folders written before that fix carry the profiler's
    # raw literal 'nan' here — normalize at read time so legacy folders
    # render the pipeline's word without re-running
    _HOMOLOG_JUNK_TYPES = frozenset(
        {'', '?', 'nan', 'na', 'n/a', 'none', 'null', 'unknown',
         '<na>', '<null>'})
    for _r in forward_rows + target_match_rows:
        for _f in ('source_type', 'primary_target_type', 'target_type',
                   'primary_source_type'):
            _v = str(_r.get(_f) or '').strip().lower()
            if _v in _HOMOLOG_JUNK_TYPES:
                _r[_f] = 'untyped'
    # the deep window carries the `examinees` bin — its rows must reach the
    # allocation the same way the other expansion bins do
    deep_rows = _read_csv_rows(_run_file(run_dir, 'deep_candidates.csv'))
    # allocation = every bin/status a SOURCE bodyId appears in across the
    # run's artifacts, so one panel row answers "where does this neuron
    # live today" without cross-referencing five tabs
    forward_alloc: Dict[str, Dict] = {}

    def _fwd_alloc(bid) -> Dict:
        return forward_alloc.setdefault(
            str(bid), {'verdict': '', 'bins': [], 'fill': False,
                       'out_map': False, 'source_status': ''})

    for b, v in (best_verdict or {}).items():
        _fwd_alloc(b)['verdict'] = str(v or '')
    for r in (list(sus_rows) + list(deep_rows) + list(fam_rows)
              + list(rel_rows)):
        c = str(r.get('category') or '')
        b = str(r.get('source_bodyId') or '')
        if c and b:
            bins = _fwd_alloc(b)['bins']
            if c not in bins:
                bins.append(c)
    for r in prop_rows:
        if str(r.get('side') or '') != 'source':
            continue
        b = str(r.get('bodyId') or '')
        if b:
            _fwd_alloc(b)['fill'] = True
    for b in out_sources:
        if b:
            _fwd_alloc(b)['out_map'] = True
    for r in ss_rows:
        b = str(r.get('source_bodyId') or '')
        if b:
            _fwd_alloc(b)['source_status'] = str(r.get('status') or '')

    scenes = _parse_scenes(run_dir)
    selfcheck = {'pass': len(readme['selfcheck_pass']),
                 'fail': len(readme['selfcheck_fail'])}

    # mapper-consistent source split: distinct source bodyIds per branch
    # pool basis (row-evidence backed = 'linker rows'; same-name pooled =
    # 'full population'), from the SELECTED branches of mapping_export.
    basis_sources: Dict[str, set] = {}
    basis_branches: Dict[str, int] = {}
    target_basis_by_branch: Dict[tuple, str] = {}
    parent_all: set = set()
    claimed_all: set = set()
    try:
        import ast as _ast
        for r in _read_csv_rows(_run_file(run_dir, 'mapping_export.csv')):
            if not _truthy(r.get('is_selected')):
                continue
            basis = r.get('pool_basis', '?')
            basis_branches[basis] = basis_branches.get(basis, 0) + 1
            target_basis_by_branch[
                (str(r.get('source_type', '')),
                 str(r.get('target_type', '')))] = str(
                     r.get('target_pool_basis', '') or '')
            try:
                ids = _ast.literal_eval(r.get('source_body_ids') or '{}')
                parents = _ast.literal_eval(
                    r.get('parent_source_body_ids') or '{}')
            except (ValueError, SyntaxError):
                continue
            basis_sources.setdefault(basis, set()).update(
                int(b) for b in ids)
            claimed_all.update(int(b) for b in ids)
            parent_all.update(int(b) for b in parents)
    except Exception:  # noqa: BLE001
        basis_sources = {}
    # A run whose pair_summary predates the per-side column still has the
    # target-side basis in mapping_export, so the Branches cell can name both
    # sides for every run, not just new ones.
    for r in pair_rows:
        if not r.get('target_pool_basis'):
            r['target_pool_basis'] = target_basis_by_branch.get(
                (str(r.get('source_type', '')),
                 str(r.get('target_type', ''))), '')
    n_with_branch = (len(set().union(*basis_sources.values()))
                     if basis_sources else None)
    # The source-side claim envelope (§15.3): of every queried source
    # neuron, how many sit in at least one branch pool (claimed and graded
    # there) and how many fall outside all of them — the pool-refinement
    # RESIDUE that `compute_out_map_by_type` hands to the out-map
    # expansion. A residue of 0 over a large parent pool is not "nothing
    # left to find": on an unrefined (wide) source pool every neuron is
    # claimed by construction, so the expansion route never fires.
    src_residue = len(parent_all - claimed_all) if parent_all else None

    # family-material reconciliation: family material is the map-structure
    # remainder (in-map − map-covered) and is reported COMPLETE; the dedup
    # bins split it by fill-accounting precedence (candidates > family), so
    # the note names which members the candidates bin claims.
    fm_ids = {int(b) for b in (coverage.get('target') or {}).get(
        'family_material') or []}
    fm_family = fm_ids & {int(r['target_bodyId']) for r in dedup_rows
                          if r['dedup_category'] == 'family'}
    fm_cand = fm_ids & {int(r['target_bodyId']) for r in dedup_rows
                        if r['dedup_category'] == 'candidates'}
    fm_other = fm_ids - fm_family - fm_cand

    advisories = []
    # Track-A availability: a transient enrichment failure silently
    # degrades the run to native Track-B floors only — surface it loudly
    # instead of letting '—' bars speak for themselves.
    track_a_dead = bool(
        calib and params.get('morph_enabled', True)
        and not calib.get('n_verified_scored')
        and not calib.get('track_a_null_n'))
    if track_a_dead:
        advisories.append('morph Track-A unavailable')
    elif any(k.startswith('null') for k in susp_kind_counts):
        advisories.append('null-sample')
    if selfcheck['fail'] or readme['bang_lines']:
        advisories.append('run warnings')
    n_top_bev = sum(b['high'] + b['medium']
                    for k, b in backward_bins.items()
                    if k in ('candidates', 'family', 'relative'))
    if n_top_bev:
        advisories.append(
            f'reciprocal: {n_top_bev} member(s) rank their own branch '
            'source type top-3')

    return {
        'run_dir': run_dir,
        'params': params,
        'calib': calib,
        'coverage': coverage,
        'progress': progress,
        'readme': readme,
        'mapper_gap': dict(gap or {}),
        'mapper_gap_untyped': gap_untyped,
        'src': coverage.get('source') or {},
        'tgt': coverage.get('target') or {},
        'scanned_sources': scanned_sources,
        'src_parent': len(parent_all) or None,
        'src_claimed': len(claimed_all) or None,
        'src_residue': src_residue,
        'verdict_counts': verdict_counts,
        'cat_counts': cat_counts,
        'n_out_of_scope': n_out_of_scope,
        'cand_ann': cand_ann,
        'cand_break': cand_break,
        'cand_bin_ids': cand_bin_ids,
        'sus_cand_by_bid': sus_cand_by_bid,
        'dedup_cat': dedup_cat,
        'n_dup': n_dup,
        'restrictive': restrictive,
        'family_fill': family_fill,
        'levels_by_bid': levels_by_bid,
        'prop_by_bid': prop_by_bid,
        'level_counts': level_counts,
        'fill_class_counts': fill_class_counts,
        'side_counts': side_counts,
        'pair_rows': pair_rows,
        'same_name_excluded': same_name_excluded,
        'disclosure_evidence': disclosure_rows,
        'suspects_verification': suspects_rows,
        'pooling_xval': pooling_xval,
        'pooling_pool': pooling_pool,
        'pooling_rows': pooling_rows,
        'pooling_sources': pooling_sources,
        'backward_rows': backward_rows,
        'backward_bins': backward_bins,
        'backward_by_bid': backward_by_bid,
        'backward_rows_by_bid': rows_by_bid,
        'forward_rows': forward_rows,
        'forward_available': forward_available,
        'forward_alloc': forward_alloc,
        'target_match_rows': target_match_rows,
        'target_match_available': target_match_available,
        'backward_counters': (coverage.get('target') or {}).get(
            'backward_evidence') or {},
        'out_sources': out_sources,
        'out_morph_pass': out_morph_pass,
        'out_rows_total': len(out_rows),
        'out_type_counts': out_type_counts,
        'per_source_best': per_source_best,
        'fam_rows_n': len(fam_rows),
        'rel_rows_n': len(rel_rows),
        'branch_bars': branch_bars,
        'bar_kind_counts': bar_kind_counts,
        'susp_kind_counts': susp_kind_counts,
        'null_used': null_used,
        'track_a_dead': track_a_dead,
        'basis_sources': basis_sources,
        'basis_branches': basis_branches,
        'n_with_branch': n_with_branch,
        'fm_ids': fm_ids,
        'fm_family': fm_family,
        'fm_cand': fm_cand,
        'fm_other': fm_other,
        'ss_rows': ss_rows,
        'ss_counts': dict(ss_counts),
        'per_type_status': {k: dict(v) for k, v in per_type_status.items()},
        'scenes': scenes,
        'selfcheck': selfcheck,
        'advisories': advisories,
    }


# ---------------------------------------------------------------------------
# warning notes (user_warning_notes.txt, bracketed-tag convention)
# ---------------------------------------------------------------------------

def _reciprocal_warning_line(d: Dict) -> Optional[str]:
    """The stage-5d advisory summary, built once so ``report.html`` and
    ``user_warning_notes.txt`` cannot drift apart."""
    if not d.get('backward_bins'):
        return None
    gap_bins = {k: v for k, v in d['backward_bins'].items()
                if k in _BEV_GAP_BINS}
    if not gap_bins:
        return None
    n_fgn = sum(v['high'] + v['medium'] for v in gap_bins.values())
    n_scan = sum(v['neurons'] for v in gap_bins.values())
    capped = _as_num((d.get('backward_counters') or {}).get('beyond_cap'))
    note = (f'[reciprocal] {n_fgn}/{n_scan} scanned gap-fill member(s) '
            'rank their own branch source type in a top-3 — advisory '
            'provenance to review, never a rejection (Reciprocal tab)')
    if capped:
        note += f'; {int(capped)} member(s) beyond the scan cap are ' \
                'unchecked'
    return note


def _pooling_warning_line(d: Dict) -> Optional[str]:
    """The pooling pass's own caveats, built ONCE so the Pooling tab, the Log
    tab and ``user_warning_notes.txt`` cannot drift apart.

    Emitted only when there is a caveat to state: the last gate measuring less
    than it attempted (a `no-score` target is a missing measurement, and a run
    where most targets are unscored must not read as "the pool was
    morphologically cleared"), the budget dropping targets, the gate failing,
    or the scorer's own warnings — BANC's experimental-skeleton caveat is the
    standing one.
    """
    m = ((d.get('pooling_xval') or {}).get('morph') or {})
    if not m:
        return None
    att = _as_num(m.get('attempted')) or 0
    scored = _as_num(m.get('scored')) or 0
    no_score = _as_num(m.get('no_score')) or 0
    capped = _as_num(m.get('capped')) or 0
    bits = []
    if m.get('error'):
        bits.append(f'the morphology gate itself failed ({m["error"]}); '
                    'every row is unpublished, not rejected')
    if scored < att or no_score:
        bits.append(f'scored {int(scored)}/{int(att)} attempted targets '
                    f'({int(no_score)} `no-score`) — an unscored candidate '
                    'is a missing measurement, never a rejection')
    if capped:
        bits.append(f'{int(capped)} of {int(_as_num(m.get("units")) or capped)} '
                    'scoring units past the morph budget '
                    f'({m.get("budget") or "provenance not published"}) were '
                    'never looked at (`morph_gate=not-attempted-cap`)')
    bits += [str(w) for w in (m.get('warnings') or [])]
    if not bits:
        return None
    return '[pooling] ' + '; '.join(bits)


def collect_warnings(d: Dict) -> List[str]:
    lines = []
    sc = d['selfcheck']
    n_checks = sc['pass'] + sc['fail']
    if n_checks:
        lines.append(f"[self-check] {sc['pass']}/{n_checks} scene "
                     'self-checks passed (leaf geometry matches, every '
                     'expected neuron plotted, line mode serves '
                     'centerlines)')
    elif d['scenes']:
        lines.append('[scenes] rendered without self-check '
                     '(scene_selfcheck off)')
    else:
        lines.append('[scenes] none rendered (visualization skipped or '
                     'no renderable expansion content)')
    for fail in d['readme']['selfcheck_fail']:
        lines.append(f'[self-check FAIL] {fail}')
    for bang in d['readme']['bang_lines']:
        if not bang.startswith('! self-check'):
            lines.append(f'[run-warning] {bang}')
    if d['track_a_dead']:
        lines.append(
            '[morph] Track-A unavailable this run (0 scored rows) — '
            'candidate admission ran on native Track-B floors only; '
            'morph-dependent counts (candidates, out-map passes, AUC, '
            'null bar) are not comparable to a healthy run. Out-of-scope '
            'rows stay in the CSVs for reconciliation')
    elif d['null_used']:
        c = d['calib']
        bar = c.get('track_a_null_bar')
        if bar is not None:
            lines.append(
                '[null-sample] run null bar '
                f"{bar} (p95, n={c.get('track_a_null_n', '?')}, "
                f"{c.get('track_a_null_source', 'jaccard<=0.05 window')}) "
                '— null-kind branch bars are run-sensitive '
                '(skeleton-cache growth changes the scored null subset); '
                'do not compare null-kind bars ACROSS runs (native-floor '
                'branches are unaffected)')
    for t, n in sorted(d['mapper_gap'].items(), key=lambda kv: -kv[1]):
        lines.append(f'[mapper-gap] {t}: {n} row(s) with NO backward '
                     'mapping to the source dataset — consider '
                     'annotating or crosswalking them')
    if d['mapper_gap_untyped']:
        lines.append(f"[mapper-gap] (untyped): {d['mapper_gap_untyped']} "
                     'row(s) with no type annotation at all')
    recip = _reciprocal_warning_line(d)
    if recip:
        lines.append(recip)
    pooling = _pooling_warning_line(d)
    if pooling:
        lines.append(pooling)
    return lines


def append_warning_notes(run_dir: Path, blocks: List[str],
                         log: Optional[Callable] = None) -> None:
    """Append to ``user_warning_notes.txt`` with the shared header
    discipline (several writers append; whoever creates the file writes
    the title; headerless legacy files are healed in place).

    Idempotent for regeneration: blocks whose exact line already exists
    in the file are skipped, so re-running the report writer on the same
    run never duplicates a warning block."""
    if not blocks:
        return
    path = Path(run_dir) / 'user_warning_notes.txt'
    header = 'User warning notes\n==================\n\n'
    try:
        existing = ''
        if path.exists():
            try:
                existing = path.read_text(encoding='utf-8')
            except OSError:
                existing = ''
        if existing.strip():
            # skip lines already recorded (regeneration idempotency)
            have = {ln.strip() for ln in existing.splitlines()}
            blocks = [b for b in blocks if b.strip() not in have]
            if not blocks:
                return
        if not existing.strip():
            path.write_text(header + '\n'.join(blocks) + '\n',
                            encoding='utf-8')
        elif existing.startswith('User warning notes'):
            with open(path, 'a', encoding='utf-8') as f:
                f.write('\n' + '\n'.join(blocks) + '\n')
        else:
            path.write_text(header + existing.rstrip('\n') + '\n'
                            + '\n' + '\n'.join(blocks) + '\n',
                            encoding='utf-8')
    except Exception as exc:  # noqa: BLE001
        if log:
            log(f'    ! could not append user warning notes: {exc}')


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _esc(v) -> str:
    from html import escape
    return escape(str(v))


def _th(label: str, tip: str = '') -> str:
    """A table header whose text carries the standard hover definition.
    The JS hover layer lifts the tip above every card grid and scroll
    wrapper; without JS the inline CSS tooltip still works."""
    if not tip:
        return f'<th>{_esc(label)}</th>'
    return (f"<th><span class='term'>{_esc(label)}"
            f"<span class='tip'><b>{_esc(label)}</b>{_esc(tip)}"
            '</span></span></th>')


def _term(key: str, label: Optional[str] = None) -> str:
    """Hoverable term: dotted underline + CSS-only tooltip."""
    definition = TERM_DEFS.get(key)
    text = label or key
    if not definition:
        return _esc(text)
    return (f"<span class='term'>{_esc(text)}"
            f"<span class='tip'><b>{_esc(text)}</b>"
            f'{_esc(definition)}</span></span>')


def _hover(label_html: str, tip_html: str) -> str:
    """A ``.term`` whose tooltip carries rendered HTML instead of a
    definition string — used where the payload is a table (the top-N
    homolog list), which ``_term`` cannot express.  ``label_html`` is
    markup the caller already escaped."""
    if not tip_html:
        return label_html
    return (f"<span class='term'>{label_html}"
            f"<span class='tip mv-tip-wide'>{tip_html}</span></span>")


def _defs_block(keys: List[str]) -> str:
    rows = ''.join(
        f'<tr><td>{_esc(k)}</td><td>{_esc(TERM_DEFS[k])}</td></tr>'
        for k in keys if k in TERM_DEFS)
    if not rows:
        return ''
    return ('<details class="detail-block"><summary>Definitions for this '
            "section</summary><div style='overflow-x:auto'>"
            "<table class='mv-table mv-defs'><tbody>"
            f'{rows}</tbody></table></div></details>')


def _scroll_viewport(rows_html: List[str], headers: str,
                     visible: int = 50) -> str:
    """ONE table with EVERY row (user 2026-09-18): the container scrolls
    vertically once more than ``visible`` rows are shown — no second
    details table."""
    head = f'<thead><tr>{headers}</tr></thead>'
    max_h = visible * 29
    return ("<div style='overflow:auto;max-height:"
            f"{max_h}px'><table class='mv-table'>{head}"
            f"<tbody>{''.join(rows_html)}</tbody></table></div>")


def _kv_block(title: str, pairs: List[Tuple[str, str]]) -> str:
    rows = ''.join(
        f"<tr><td class='kv-k'>{k}</td><td class='kv-v'>{v}</td></tr>"
        for k, v in pairs)
    return (f"<div class='mv-pre-wrap'><div class='mv-kv-title'>"
            f'{_esc(title)}</div>'
            f"<table class='mv-table mv-kv'><tbody>{rows}</tbody>"
            '</table></div>')


def _section_card(heading: str, summary: str, body: str,
                  term_keys: Optional[List[str]] = None) -> str:
    parts = [f"<h2 class='section-heading'>{_esc(heading)}</h2>"]
    if summary:
        parts.append(f'<p class="section-summary">{summary}</p>')
    parts.append(body)
    if term_keys:
        parts.append(_defs_block(term_keys))
    return "<div class='section-card'>" + ''.join(parts) + '</div>'


def _chip(label: str, value: str) -> str:
    return (f"<div class='meta-chip'><span>{_esc(label)}</span>"
            f'<strong>{value}</strong></div>')


def _empty(msg: str) -> str:
    return f"<div class='heatmap-empty'>{_esc(msg)}</div>"


# ---------------------------------------------------------------------------
# section builders
# ---------------------------------------------------------------------------

def _hero(d: Dict) -> str:
    rd = d['run_dir']
    params = d['params']
    start_ts = next((p.get('ts') for p in d['progress']
                     if p.get('event') == 'run_start'), '')
    elapsed = next((p.get('elapsed_s') for p in d['progress']
                    if p.get('event') == 'run_done'), None)
    elapsed_txt = (f'{elapsed / 60:.0f} min {elapsed % 60:.0f} s'
                   if isinstance(elapsed, (int, float)) else '—')
    queries = ', '.join(params.get('query_types') or [])
    mode = params.get('validation_mode', '—')
    sc = d['selfcheck']
    n_checks = sc['pass'] + sc['fail']
    if n_checks:
        health = f'✅ scene self-checks {sc["pass"]}/{n_checks}'
    elif d['scenes']:
        health = '✅ scenes rendered (self-checks off)'
    else:
        health = 'no scenes'
    if d['advisories']:
        health += ' · ⚠️ ' + ', '.join(d['advisories'])

    scov, tcov = d['src'], d['tgt']
    total = scov.get('total_queried')
    covered = tcov.get('in_branch_pool')
    mapped_set = tcov.get('mapped_target_set')
    if total and covered is not None:
        supervised = (
            f'{_esc(total)} source neurons → <b>{_esc(covered)}</b> '
            f'{_esc(params.get("target_dataset", ""))} neurons '
            f'{_term("map-covered")} (of {_esc(mapped_set)} in-map; '
            f'{_pct(covered, mapped_set)}) · '
            f'{_term("mutual-best (assigned)", "mutual-best")} '
            f'{_esc(scov.get("assigned", "—"))} · fill-proposed '
            f'+{_esc(scov.get("fill_proposed_only", "—"))}')
    else:
        supervised = ('Coverage artifacts absent — the run produced no '
                      'set-level coverage (see the Log tab).')

    # On a pooling run the ladder above is NOT the run's answer — the mode is
    # parallel to it, and its own result lives on the source axis.  Leading
    # with the supervised numbers made the headline quote one mode's levels for
    # another mode's run, two screens above the answer (issue 13), so the
    # headline now names the level it is quoting and puts the pool first.
    headline = supervised
    # §three-tier qualifier (user 2026-09-27): the supervised ladder quotes
    # the CLAIM tier only; the disclosure ends (declined but
    # evidence-reached) ride the same run as advisory material — name both
    # so the headline cannot be read as the full evidence reach.
    _disc_rows = _read_csv_rows(
        _run_file(rd, 'disclosure_evidence.csv'))
    if _disc_rows:
        _disc_types = len({str(r.get('target_type') or '')
                           for r in _disc_rows})
        supervised += (
            f" · <span class='mv-note'>disclosure evidence "
            f"+{_esc(len(_disc_rows))} row(s) / "
            f"{_esc(_disc_types)} declined type(s) — advisory, not in "
            'the claim set (see the Coverage tab)</span>')
        headline = supervised
    px = (d.get('pooling_xval') or {}) if str(mode) == 'pooling' else {}
    psrc = d.get('pooling_sources') or []
    if px and psrc:
        def _pi(r, k):
            return int(_as_num(r.get(k)) or 0)

        kept = sum(1 for r in psrc if _pi(r, 'n_in_pool') > 0)
        n_pool = sum(1 for r in (d.get('pooling_pool') or [])
                     if str(r.get('in_pool') or '').strip().lower()
                     in TRUE_STR)
        headline = (
            f"<b>{_esc(len(psrc))}</b> {_term('pooling', 'queried sources')} → "
            f"{_esc(px.get('seed', {}).get('sources_with_a_candidate', '—'))} "
            f"reached ≥1 candidate · <b>{_esc(kept)}</b> kept one after "
            f"morphology · <b>{_esc(n_pool)}</b> distinct pooled targets "
            f"(pool/source {_f(px.get('pool_per_source'), 2)}) — the "
            'unsupervised pool, which the mapper decides none of'
            f"<br><span class='mv-note'>Supervised ladder, same run: "
            f'{supervised}</span>')

    return (
        '<header class="report-hero">'
        '<div class="report-kicker">DROCAT · TM VEV run report</div>'
        '<h1 class="report-title">Type-mapping validate · expand · '
        'visualize</h1>'
        f'<p class="report-subtitle">{headline}</p>'
        '<div class="report-meta">'
        + _chip('Run', _esc(rd.name))
        + _chip('Started', _esc(start_ts or '—'))
        + _chip('Elapsed', _esc(elapsed_txt))
        + _chip('Query', _esc(queries or '—'))
        + _chip('Source', _esc(params.get('source_dataset', '—')))
        + _chip('Target', _esc(params.get('target_dataset', '—')))
        + _chip('Mode', _term('validation mode', _esc(mode)))
        + _chip('Health', _esc(health))
        + '</div>'
        "<p class='report-note'>Generated "
        f'{datetime.now():%Y-%m-%d %H:%M:%S} · hover any dotted term or '
        'table header for its definition · every number traces to a run '
        'artifact (Log tab lists them) · '
        '<a href="_UserGuide_please_read_me.html">'
        '📘 User guide — this run\'s files and terms, '
        'explained</a></p>'
        '</header>')


def _coverage_tab(d: Dict) -> str:
    scov, tcov = d['src'], d['tgt']
    #: the per-type identity's own term, summed over mapped types only
    _in_map_cand = sum(int((v or {}).get('reached_as_candidates_only') or 0)
                       for v in (tcov.get('per_type') or {}).values())
    cards = []
    if scov and tcov:
        # §1 the three coverage levels
        l1 = _kv_block('L1 CLAIM (forward) — what does the map cover?', [
            ('Source claim envelope',
             ('—' if d.get('src_claimed') is None else
              f"{_esc(d['src_claimed'])} of {_esc(d['src_parent'])} queried "
              f"sources sit in a branch pool; "
              f"{_esc(d.get('src_residue'))} outside every pool — the "
              f"{_term('out-map expansion', 'out-map residue')}, which the "
              f"expansion scans against the full target universe")),
            ('In-map target population',
             f"{_esc(tcov.get('mapped_target_set', '—'))} neurons "
             f'({len(tcov.get("per_type") or {})} target types)'),
            (_term('map-covered', 'Map-covered (branch pools)'),
             f"{_esc(tcov.get('in_branch_pool', '—'))} "
             f'({_pct(tcov.get("in_branch_pool"), tcov.get("mapped_target_set"))})'),
            # The set-level count is NOT the identity's term. `reached_tgt` spans
            # every candidate-reached bodyId whatever its type, while the per-type
            # rows exist only for mapped types — so 27 here ⊃ the 10 that
            # `mapped − in_pool = candidates_only + holes` sums to. Printing the
            # big number between "map-covered" and "holes" invited the reader to
            # subtract, and the arithmetic then failed (219 − 204 = 15 ≠ 27 + 5).
            ('Reached only as candidates',
             _hover(
                 f"{_esc(tcov.get('reached_as_candidates_only', '—'))}",
                 '<b>Two scopes, one number each</b>'
                 "This is every candidate-reached target neuron outside a "
                 "branch pool, INCLUDING types the map does not assert. The "
                 "term the per-type identity uses — "
                 "`mapped_population − in_pool = reached_as_candidates_only + "
                 "holes` — counts only in-map types, and sums to "
                 + _cnt(_in_map_cand) + " here. The difference is out-of-map "
                 "candidate material, which can never be a hole.")),
            (_term('hole', 'Holes (never map-covered)'),
             f"{_esc(tcov.get('holes', '—'))} ▸ Targets tab"),
            (_term('family material'),
             f'{len(tcov.get("family_material") or [])} '
             f'(= {tcov.get("mapped_target_set")} − '
             f'{tcov.get("in_branch_pool")}) ▸ Targets tab'),
        ])
        tiers = {k: d['dedup_cat'].get(k, 0)
                 for k in ('matched', 'verified', 'borderline')}
        l2 = _kv_block(
            'L2 PROVENANCE — how was each of the '
            f'{_esc(tcov.get("in_branch_pool", "…"))} earned?', [
                (f'{_term("matched")} <i>(the only ASSERTED tier)</i>',
                 _esc(tiers['matched'])),
                (f'{_term("verified")} <i>(review tier)</i>',
                 _esc(tiers['verified'])),
                (f'{_term("borderline")} <i>(review tier)</i>',
                 _esc(tiers['borderline'])),
                (_term('sibling', 'Cross-branch convergence'),
                 f"{_esc(d['cat_counts'].get('sibling', 0))} sibling "
                 f'rows; {_esc(d["n_dup"])} dedup rows sit in more than '
                 f'one branch ({_term("dup")}) — convergence, not '
                 'double counting'),
            ])
        sc = d['selfcheck']
        calib = d['calib']
        auc = calib.get('auc')
        if d.get('track_a_dead'):
            auc_txt = ('Track-A unavailable this run (0 scored rows) — '
                       'admission ran on native Track-B floors only; '
                       'morph-dependent counts are not comparable to a '
                       'healthy run (see Morph tab)')
        elif auc is not None:
            auc_state = ('INACTIVE' if not calib.get('calibrated')
                         else 'ACTIVE')
            auc_txt = (
                f'AUC gate {auc_state} ({_f(auc)} vs floor '
                f"{_f(calib.get('auc_floor', 0.65) or 0.65)}) — "
                'informational only; every rendered member passed its '
                'branch bar')
        else:
            auc_txt = 'morphology not run (--no-morphology)'
        l3 = _kv_block('L3 VALIDATION — what was checked?', [
            ('Sources scanned into pools',
             f'{len(d["scanned_sources"])} of '
             f"{_esc(scov.get('total_queried', '—'))} queried"),
            (_term('mutual-best (assigned)', 'Mutual-best (assigned)'),
             f"{_esc(scov.get('assigned', '—'))} sources"),
            ('Morph record', _esc(auc_txt)),
            (_term('scene self-check', 'Scene self-checks'),
             (f'{sc["pass"]}/{sc["pass"] + sc["fail"]} passed'
              + (' ⚠️ failures — see Log tab' if sc['fail'] else ''))
             if (sc['pass'] + sc['fail']) else 'no scenes rendered'),
        ])
        cards.append(_section_card(
            'Coverage — the deliverable (three levels)',
            'L1 claim · L2 provenance · L3 validation, reported in '
            'every TM VEV run.',
            "<div class='metric-grid single'>" + l1 + l2 + l3 + '</div>',
            ['coverage levels', 'matched', 'verified', 'borderline',
             'sibling', 'dup', 'hole', 'family material',
             'scene self-check', 'AUC gate', 'mutual-best (assigned)',
             'map-covered']))



        # §2 source waterfall — mapper-consistent buckets (user
        # 2026-09-17: row-evidence backed + same-name pooled + out-map),
        # with the evidence overlay as the decided flat split.
        assigned_n = scov.get('assigned', 0)
        proposed_n = scov.get('fill_proposed_only', 0)
        unpaired = scov.get('unpaired_unproposed', 0)
        unclaimed = len(d['out_sources'])
        in_pool_residue = unpaired - unclaimed
        basis = d['basis_sources']
        row_bases, wide_bases = split_basis_buckets(basis)
        n_linker = sum(len(basis[b]) for b in row_bases)
        n_full = sum(len(basis[b]) for b in wide_bases)
        b_linker = sum(d['basis_branches'].get(b, 0) for b in row_bases)
        b_full = sum(d['basis_branches'].get(b, 0) for b in wide_bases)
        covered = tcov.get('in_branch_pool', '—')
        total_q = scov.get('total_queried', '—')
        if basis:
            # The out-map bucket derives from COVERAGE (queried − sources
            # with a branch), not from out-map row presence: a sparse
            # typed target universe (BANC) can leave unclaimed sources
            # with zero expansion rows.  Surface that case explicitly.
            out_n = (total_q - d['n_with_branch']
                     if isinstance(total_q, (int, float))
                     and d.get('n_with_branch') is not None
                     else unclaimed)
            out_meaning = ('no branch anywhere → Out-map tab (expansion '
                           'evidence, not fills)')
            if not d['out_rows_total']:
                out_meaning += ' — expansion produced 0 candidate rows'
            wf_rows = [
                f"<tr><td>{_esc(total_q)} queried</td>"
                '<td>the query population</td></tr>',
                f"<tr><td>{_esc(n_linker)} row-evidence backed</td>"
                '<td>bridge-chain branches (linker rows · '
                f'{b_linker} branches) — the mapper\'s row-based '
                'evidence</td></tr>',
                f"<tr><td>{_esc(n_full)} same-name pooled</td>"
                '<td>full-population branches ('
                f'{b_full} branches) — same-name identity, whole type '
                'pooled</td></tr>',
                f"<tr><td>{_esc(out_n)} out-map</td>"
                f'<td>{out_meaning}</td></tr>',
            ]
        else:
            # fixture / legacy fallback: the flat evidence split only
            wf_rows = [
                f"<tr><td>{_esc(scov.get('total_queried', '—'))} queried</td>"
                '<td>the query population</td></tr>',
                f"<tr><td>{_esc(assigned_n)} assigned</td>"
                '<td>paired onto branch pools</td></tr>',
                f'<tr><td>+{_esc(proposed_n)} fill-proposed only</td>'
                '<td>tiered pool members below mutual-best</td></tr>',
                f"<tr><td>{_esc(unpaired)} unpaired without proposal</td>"
                f'<td>= {_esc(in_pool_residue)} in-pool residue + '
                f'{_esc(unclaimed)} unclaimed</td></tr>',
            ]
        wf = ("<div style='overflow-x:auto'><table class='mv-table'>"
              '<thead><tr>'
              + _th('bucket', 'Which side of the mapper-consistent '
                    'accounting the row counts: queried, assigned, '
                    'fill-proposed, or unpaired.')
              + _th('meaning', 'What lands in the bucket and where the '
                    'residue goes.')
              + '</tr></thead>'
              "<tbody>" + ''.join(wf_rows) + '</tbody></table></div>'
              "<p class='mv-note'>Consistent with the type mapper: its "
              'mapped set is the map-covered '
              f'{_esc(covered)} — family material is unmapped in-map '
              'material and stays outside it. Per-source backward '
              'statuses: see the Backward tab (informational only).</p>')
        cards.append(_section_card(
            f"Source side — the {scov.get('total_queried', '…')}, "
            'accounted',
            'Mapper-consistent buckets (row-evidence backed / same-name '
            'pooled / out-map); proposals are evidence, the mapping is '
            'never rewritten.',
            wf, ['mutual-best (assigned)', 'map-covered',
                 'validation mode']))

        # per-type table (21 rows here)
        per = scov.get('per_type') or {}
        rows = []
        for name in sorted(per, key=lambda k: -int(per[k].get('pool')
                                                   or 0)):
            v = per[name]
            rows.append(
                f'<tr><td>{_esc(name)}</td>'
                f'<td>{_esc(v.get("pool", 0))}</td>'
                f'<td>{_esc(v.get("assigned", 0))}</td>'
                f'<td>{_esc(v.get("fill_proposed", 0))}</td>'
                f'<td>{_esc(v.get("unpaired_unproposed", 0))}</td></tr>')
        body = _scroll_viewport(
            rows,
            _th('source type', 'The queried source-side type.')
            + _th('pool', 'Source neurons of this type in the query '
                  'set.')
            + _th('assigned', 'Neurons with a mutual-best pair onto a '
                  'branch pool — the ASSERTED tier.')
            + _th('fill-proposed', 'Neurons carried only as tiered '
                  'fill proposals — evidence, never an assignment.')
            + _th('unpaired', 'No pair and no proposal: in-pool '
                  'residue plus unclaimed neurons.'))
        cards.append(_section_card(
            'Per-type pools (source)',
            'Feeds from set_coverage.json source.per_type.', body))
    else:
        cards.append(_section_card(
            'Coverage — the deliverable', '',
            _empty('set_coverage.json absent — the run did not reach '
                   'the coverage stage.')))
    # Same-name-first consumers (P1/P2/P4): additive accounting card,
    # rendered only when the run has same-name or multivalue content.
    # Counts come from set_coverage.json when present; the exported
    # rows are the fallback (a no-pair run — every queried type held
    # — has no coverage block at all).
    cov = d.get('coverage') or {}
    excl_rows = d.get('same_name_excluded') or []
    snf_pairs = cov.get('same_name_first_pairs') or 0
    snf_held = cov.get('same_name_first_held') or len(
        [r for r in excl_rows
         if r.get('disposition') == 'gated_held'])
    snf_excl = cov.get('same_name_first_excluded') or len(
        [r for r in excl_rows
         if r.get('disposition') == 'excluded_evidence_only'])
    mv_types = cov.get('multivalue_types') or len(
        [r for r in excl_rows
         if r.get('reason') == 'multivalue_cell'])
    mv_tgt = cov.get('multivalue_target_types') or 0
    fullmap_pairs = cov.get('full_map_pairs') or 0
    snf_rows: list = []
    if fullmap_pairs:
        snf_rows.append((
            'Full-map transitive pairs',
            f"{fullmap_pairs} pair(s) across "
            f"{cov.get('full_map_types') or 0} source type(s) reached "
            f"through connector dataset(s) "
            f"{', '.join(cov.get('full_map_mids') or [])} — labeled "
            'route_scope/via_mid in mapping_export.csv '
            '(2026-10-01 full-map mode)'))
    if any((snf_pairs, snf_held, snf_excl, mv_types, mv_tgt)):
        snf_rows.append((
            'Same-name-first selections (fired)',
            f"{snf_pairs} pair(s) across "
            f"{cov.get('same_name_first_types') or 0} source "
            'type(s) — marked ⟡ in Branches; rivals withheld to '
            'auto_type_mapping_suspects.csv'))
        if snf_held or snf_excl:
            snf_rows.append((
                'Held / evidence-only fan-outs',
                f'{snf_held} held + {snf_excl} evidence-only — same_'
                'name_excluded.csv (the types stay excluded)'))
        if mv_types or mv_tgt:
            snf_rows.append((
                'Multi-value type cells',
                f'{mv_types} source cell(s) + {mv_tgt} target '
                'cell(s) — kept atomic, accounted not split'))
    if snf_rows:
        cards.append(_section_card(
            'Route scope, same-name-first & multivalue accounting',
            'Advisory record of how the mapping scope and the mapper\'s '
            'same-name-first rule shaped this run\'s pair set. Never a '
            'gate.',
            _kv_block('', snf_rows),
            ['full-map mode', 'same-name-first', 'suspects',
             'multi-value type cells']))

    # §three-tier readout (user 2026-09-27): the DISCLOSURE tier — ends
    # the mapper decision declined but the derivation evidence reaches.
    # Advisory only: these rows never entered the claim set, the branch
    # pools, or any headline count above.
    disc_rows = d.get('disclosure_evidence') or []
    if disc_rows:
        from collections import Counter as _C
        by_reason = _C(str(r.get('decline_reason') or '?')
                       for r in disc_rows)
        disc_items = [
            (f"{_esc(r.get('source_type', '?'))} → "
             f"{_esc(r.get('target_type', '?'))}",
             f"declined: {_esc(r.get('decline_reason', '?'))}"
             + (f" · verification: {_esc(r.get('verdict'))}"
                if str(r.get('verdict') or '').strip() else
                ' · unverified (run without --verify-suspects)'))
            for r in disc_rows[:12]
        ]
        if len(disc_rows) > 12:
            disc_items.append(
                (f'+{len(disc_rows) - 12} more row(s)', ''))
        cards.append(_section_card(
            'Disclosure evidence (declined, not claimed)',
            'Bridge ends the scoped decision declined — same-name-first '
            'rivals, fan-out branches outside the adopted list, '
            'vote-conflicted candidates. The evidence reaches them; the '
            'claim set does not. Verified with the ordinary machinery '
            'when --verify-suspects ran; never in the headline counts.',
            _kv_block('', disc_items),
            ['three-tier', 'disclosure',
             ', '.join(f'{k}×{v}' for k, v in sorted(by_reason.items()))]))

    return ''.join(cards)


def _branches_tab(d: Dict) -> str:
    entries = []
    # Scene coverage is per PARENT type (one scene renders every branch of
    # one source type), and stage 4 can be capped — so a row's empty SCENE
    # cell must say whether the parent simply has no scene, rather than
    # leaving "no scene" indistinguishable from "nothing to review".
    scene_types = {x['type'] for x in d['scenes'] if x.get('html')}
    parent_types = {s.get('source_type', '?') for s in d['pair_rows']}
    scene_note = ''
    if scene_types and parent_types - scene_types:
        scene_note = (
            f' {len(scene_types)} of {len(parent_types)} parent types have '
            f'a rendered scene; no scene for: '
            f"{_esc(', '.join(sorted(parent_types - scene_types)))} "
            '(Max Scenes caps the render — 0 renders every parent).')
    for s in d['pair_rows']:
        src, tgt = s.get('source_type', '?'), s.get('target_type', '?')
        # same-name-first selections (P1): the pair exists because the
        # mapper fired the same-name rule — marked, never gated.
        snf_marker = ''
        if _truthy(s.get('same_name_first')):
            snf_marker = (
                " <span title='same-name-first selection — the mapper "
                'selected the candidate carrying the source\'s own name; '
                'rivals withheld to auto_type_mapping_suspects.csv'
                f" ({_esc(s.get('same_name_rivals', '') or 'none')})'"
                f'>⟡</span>')
        # branch_bars keys come in two generations: plain 'src->tgt'
        # (v3r7-era) and query-prefixed 'query|src->tgt' (multi-query
        # runs) — accept both.
        bars = d['branch_bars']
        bar = bars.get(f'{src}->{tgt}') or \
            bars.get(f"{s.get('query', '')}|{src}->{tgt}") or {}
        kind = bar.get('candidate_kind')
        floor = bar.get('native_floor') or bar.get('backup_floor')
        if kind == 'null':
            bar_txt = '— (null)'
        elif floor is not None:
            bar_txt = f'{_f(floor)} ({kind})'
        else:
            bar_txt = '—'
        scene = next((x for x in d['scenes'] if x['type'] == src), None)
        if scene and scene['html']:
            scene_cell = f"<a href='{_esc(scene['html'])}'>▶</a>"
        elif scene_types:
            scene_cell = ("<span class='term warn-chip'>—<span class='tip'>"
                          'No scene rendered for this parent type: stage 4 '
                          'renders one scene per parent type and Max Scenes '
                          'can cap it (0 renders every parent). When the cap '
                          'fires the run log names the dropped '
                          'parents.</span></span>')
        else:
            scene_cell = '—'
        hemi = ''
        try:
            h = ast.literal_eval(s.get('hemisphere', '') or '{}')
            if h.get('hemisphere_asymmetry'):
                def _sides(key):
                    c = h.get(key) or {}
                    out = f"L {c.get('L', 0)} / R {c.get('R', 0)}"
                    return out + (f" / ? {c.get('?', 0)}"
                                  if c.get('?') else '')
                hemi = (" <span class='term warn-chip'>⚠<span class='tip'>"
                        '<b>hemisphere asymmetry</b>'
                        f'Source pool {_sides("source_sides")}, target pool '
                        f'{_sides("target_sides")}. An L/R imbalance on '
                        'either side fires the gap rule even at arithmetic '
                        'gap 0 — some neuron has no same-side partner. '
                        'Advisory: it gates nothing and claims no fill.'
                        '</span></span>')
        except (ValueError, SyntaxError):
            pass
        # Mapped = every source on this branch carrying a MAPPING verdict.
        # The mutual-best pair count (`matched` in pair_summary.csv) is a
        # STRICTER subset — a source inside a pair is already counted here —
        # so the two are shown side by side, never added.
        try:
            mapped = sum(int(s.get(k) or 0) for k in
                         ('verdict_verified_strong', 'verdict_verified',
                          'verdict_borderline'))
        except (TypeError, ValueError):
            mapped = None
        try:
            smaller = min(int(s.get('source_pool') or 0),
                          int(s.get('target_pool') or 0))
        except (TypeError, ValueError):
            smaller = 0
        if mapped is None:
            mapped_cell = gap_cell = '—'
        else:
            mapped_cell = _hover(
                f"{mapped}<span class='mv-note'> · pairs "
                f"{_esc(s.get('best', '—'))}</span>",
                '<b>Mapped vs paired</b>'
                f'{mapped} source neurons carry a mapping verdict '
                '(verified_strong / verified / borderline). '
                f"{s.get('best', '—')} of them sit in a mutual-best 1:1 "
                'pair — both sides name each other — which is the stricter '
                'count pair_summary.csv reports as `best`. The paired '
                'sources are inside the mapped total, not additional to it. '
                '<b>`best` is a per-branch count</b>: two branches of one '
                'query can both pair the same neuron, so summing the column '
                'over branches can exceed the run\'s set-level mutual-best '
                '(the headline and set_coverage.json count each neuron once).')
            gap_v = max(0, smaller - mapped)
            gr = f'{(gap_v / smaller):.0%}' if smaller else '—'
            gap_cell = _hover(
                f"{gap_v} ({gr})",
                '<b>Gap</b>'
                f'min(|source pool|, |target pool|) = {smaller} − {mapped} '
                'mapped = '
                f'{gap_v}: the neurons on the smaller side that NO mapping '
                'verdict claims. Clamped at 0. pair_summary.csv still '
                f"reports the stricter pair-based gap ({s.get('gap', '—')}, "
                'smaller − mutual-best pairs), so the two differ wherever '
                'sources are mapped without pairing 1:1.')
        verdicts = ' / '.join(
            str(s.get(k, 0)) for k in
            ('verdict_verified_strong', 'verdict_verified',
             'verdict_borderline', 'verdict_unmatched'))
        susp = (f"{_esc(s.get('suspicious_neurons', 0))} / "
                f"{_esc(s.get('suspicious_noise_filtered', 0))}")
        # Both sides, labelled: after the per-side basis the single
        # `pool_basis` cell describes only the SOURCE side, and a wide side is
        # a structural fact, not a gap — so name each side and let the term
        # carry the reason.
        src_basis = str(s.get('pool_basis', '') or '')
        tgt_basis = str(s.get('target_pool_basis', '') or '')

        def _side(label, basis):
            if not basis:
                return ''          # run predates the per-side column
            if basis == 'full population':
                return f'{label} {_term("full population", basis)}'
            return f'{label} {_esc(basis)}'

        basis_cell = ' · '.join(p for p in (_side('source', src_basis),
                                            _side('target', tgt_basis)) if p)
        triggered = _truthy(s.get('gap_triggered'))
        entries.append({
            'key': (not triggered, str(src), str(tgt)),
            'html': (
                f"<tr><td>{_esc(src)} → {_esc(tgt)}{snf_marker}</td>"
                f"<td>{_esc(s.get('mapping_status', ''))} / "
                f"{basis_cell}</td>"
                f"<td>{_esc(s.get('source_pool', '—'))}/"
                f"{_esc(s.get('source_type_total', '—'))} → "
                f"{_esc(s.get('target_pool', '—'))}/"
                f"{_esc(s.get('target_type_total', '—'))}</td>"
                f"<td>{mapped_cell}</td>"
                f"<td>{gap_cell}</td>"
                f"<td>{'●' if triggered else '–'}{hemi}</td>"
                f'<td>{_esc(verdicts)}</td><td>{susp}</td>'
                f'<td>{_term("branch bar", _esc(bar_txt))}</td>'
                f'<td>{scene_cell}</td></tr>'),
        })
    # Sorted by BRANCH (user 2026-09-18): plain (source, target) order —
    # not gap-triggered-first.
    entries.sort(key=lambda e: (e['key'][1], e['key'][2]))
    rows = [e['html'] for e in entries]
    headers = (
        _th('Branch (source → target)',
            'One row per branch: a (source type, target type) '
            'validation pair of the query. Sorted by branch.')
        + _th('Mapping status / pool basis',
              'Mapper decision status (mapped / evidence_only / …) and how '
              'EACH side of the pool was resolved, labelled source / '
              'target: linker rows = the bridging-evidence subset; full '
              'population = no supported chain named those neurons, so the '
              'whole type population is validated. The two sides are '
              'decided separately — the selected chain can be a '
              'target-side-only hop while another supported chain of the '
              'same endpoint names the source (hover the detail table\'s '
              'source_chain). Runs predating the per-side column show the '
              'source side only.')
        + _th('Pools (source of total → target of total)',
              'Validated pool sizes: source neurons of the source type '
              'total → target neurons of the target type total.')
        + _th('Mapped (M)',
              'Source neurons on this branch carrying a MAPPING verdict: '
              'verified_strong + verified + borderline. The smaller number '
              'beside it is the mutual-best 1:1 pair count (`matched` in '
              'pair_summary.csv) — a stricter subset, since a paired source '
              'is already inside the mapped total. The two are never added.')
        + _th('Gap',
              'min(|source pool|, |target pool|) − Mapped, with its pool '
              'ratio: the neurons on the smaller side that NO mapping verdict '
              'claims. Clamped at 0. Informational — proposals only, the '
              'mapping is never rewritten. pair_summary.csv keeps the '
              'stricter pair-based gap (smaller − mutual-best pairs); hover a '
              'cell for both.')
        + _th('Gap triggered',
              '● = the pipeline gap rule fired on the PAIR-based gap '
              '(advisory; the trigger no longer gates anything), so it can '
              'stay filled where the displayed Gap reads 0. '
              '⚠ = hemisphere asymmetry — hover it for the L/R counts.')
        + _th('Verdicts v★ / v / b / u',
              'Per-branch source-side verdict counts: verified_strong '
              '(v★) / verified (v) / borderline (b) / unmatched (u).')
        + _th('Examinee rows / noise filtered',
              'Examinee rows (non-pool neurons ranked ahead of the '
              'pool) / rows dropped by the noise gates (spatial '
              'caliber, negative rank_union, jaccard sanity, tie '
              'margin).')
        + _th('Morph bar (kind)',
              'The branch morph admission bar and its kind: native '
              'floor / track_a_backup / null (run-sensitive).')
        + _th('Scene', 'Link to the branch 3D scene, when rendered.'))
    table = _scroll_viewport(rows, headers, visible=50) \
        if rows else _empty('pair_summary.csv absent or empty.')
    detail_cols = ('selected_chain', 'source_chain', 'branch_linker_values',
                   'branch_annotation', 'branches_disjoint',
                   'pool_widen_added', 'deep_candidates', 'null_sample')
    det_rows = []
    for s in d['pair_rows']:
        det_rows.append(
            f"<tr><td>{_esc(s.get('source_type'))} → "
            f"{_esc(s.get('target_type'))}</td>"
            + ''.join(f"<td>{_esc(s.get(c, ''))}</td>"
                      for c in detail_cols) + '</tr>')
    detail_tips = {
        'selected_chain': 'The linker chain the mapper selected for '
                          'this branch — the best single derivation, and '
                          'the one that resolves the TARGET pool.',
        'source_chain': 'The chain that supplied the SOURCE pool. Equal to '
                        'selected_chain except where the per-side basis '
                        '(§15.3) found another supported chain of the same '
                        'endpoint that names the source neurons.',
        'branch_linker_values': 'Linker evidence values along the '
                                'selected chain.',
        'branch_annotation': 'Branch annotation text from '
                             'mapping_export.csv.',
        'branches_disjoint': 'Whether this branch pool is disjoint '
                             'from its sibling branches.',
        'pool_widen_added': 'bodyIds added by pool widening (off by '
                            'default).',
        'deep_candidates': 'Aggressive-mode deep-window candidates '
                           '(aggressive_expansion only).',
        'null_sample': 'Null sample size scored for the branch bar.',
    }
    details = _scroll_viewport(
        det_rows,
        _th('branch', 'The branch as source type → target type.')
        + ''.join(_th(c, detail_tips[c]) for c in detail_cols))
    body = table + (
        '<details class="detail-block"><summary>Per-branch chain &amp; '
        'linker details</summary>' + details + '</details>')
    return _section_card(
        f'Branches ({len(d["pair_rows"])})',
        'One row per branch, sorted by branch. All rows render in one '
        'scrollable table (50-row viewport). Hover any column header '
        'for what it measures; verdict columns are per-branch '
        'source-side counts (v★ = verified_strong).' + scene_note,
        body,
        ['branch bar', 'verdict', 'gap', 'validation mode'])


def _targets_tab(d: Dict) -> str:
    tcov = d['tgt']
    per = tcov.get('per_type') or {}
    if per:
        rows = []
        for name in sorted(
                per, key=lambda k: (-int(per[k].get('holes') or 0),
                                    -int(per[k].get('mapped_population')
                                         or 0))):
            v = per[name]
            holes = v.get('hole_body_ids') or []
            tier = (f'{_esc(v.get("in_pool_matched", 0))} / '
                    f'{_esc(v.get("in_pool_verified", 0))} / '
                    f'{_esc(v.get("in_pool_borderline", 0))}')
            rows.append(
                f'<tr><td>{_esc(name)}</td>'
                f'<td>{_esc(v.get("mapped_population", 0))}</td>'
                f'<td>{_esc(v.get("in_pool", 0))}</td>'
                f'<td>{tier}</td>'
                f'<td>{_esc(v.get("reached_as_candidates_only", 0))}</td>'
                f"<td>{_esc(', '.join(str(b) for b in holes))}</td></tr>")
        body = _scroll_viewport(
            rows,
            _th('target type', 'The target-side (male-cns) type.')
            + _th('mapped pop', 'Target neurons annotated with this '
                  'type in the target dataset.')
            + _th('map-covered', 'How many of them the type mapper '
                  'covers (its mapped set).')
            + _th('tier m / v / b', 'In-pool members by validation '
                  'tier: matched / verified / borderline.')
            + _th('cand-only', 'Neurons reached only as fill '
                  'candidates, never in a branch pool.')
            + _th('holes', 'Out-map bodyIds of this in-map type that no '
                  'branch pool claims and no fill candidate reaches '
                  '(family material minus its candidate-closed part); '
                  'bodyIds inline.'))
        summary = (
            f"Mapped target population "
            f"{_esc(tcov.get('mapped_target_set', '—'))} neurons across "
            f'{len(per)} types — holes get their bodyIds inline, '
            'because holes are the actionable output.')
    else:
        body = _empty('set_coverage.json target block absent.')
        summary = 'Target-side coverage not available for this run.'
    fam = tcov.get('family_material') or []
    fam_html = ''
    if fam:
        fam_html = (
            '<details class="detail-block"><summary>'
            f'{_term("family material")} ({len(fam)} bodyIds = '
            f"{_esc(tcov.get('mapped_target_set', '—'))} in-map − "
            f"{_esc(tcov.get('in_branch_pool', '—'))} map-covered; the type "
            'mapper\'s mapped set is the 204-style pool set — these are '
            'unmapped in-map material, never part of it)</summary>'
            "<p class='mv-note'>"
            + _esc(', '.join(str(b) for b in fam)) + '</p></details>')
    gap_html = ''
    if d['mapper_gap'] or d['mapper_gap_untyped']:
        items = [f'{_esc(t)}: {_esc(n)} row(s)'
                 for t, n in sorted(d['mapper_gap'].items(),
                                    key=lambda kv: -kv[1])]
        if d['mapper_gap_untyped']:
            items.append(f'(untyped): {d["mapper_gap_untyped"]} row(s)')
        gap_html = (
            "<div class='mv-callout'><b>Mapper-gap evidence</b> — "
            'target types hit by this run with NO backward mapping to '
            'the source (a source-annotation gap; consider annotating '
            'or crosswalking them):<br>'
            + ' &nbsp; '.join(items) + '</div>')
    return _section_card(
        'Forward — target side: map-coverage, holes, family material',
        summary,
        body + fam_html + gap_html,
        ['map-covered', 'hole', 'family material', '{T}(no_source)',
         'matched', 'verified', 'borderline'])


def _fill_tab(d: Dict) -> str:
    # §5 expansion bins
    cc = d['cat_counts']
    bin_rows = [
        f"<tr><td>{_term('candidates', 'candidates (restrictive fill)')}"
        f'</td><td>{_esc(cc.get("candidates", 0))}</td>'
        f"<td>→ dedup {len(d['restrictive'])} bodyIds (below)</td></tr>",
        f"<tr><td>{_term('sibling', 'sibling (already mapped)')}</td>"
        f"<td>{_esc(cc.get('sibling', 0))}</td>"
        '<td>cross-branch convergence</td></tr>',
        f"<tr><td>{_term('out of scope')}</td>"
        f"<td>{_esc(d['n_out_of_scope'])}</td>"
        '<td>in_scope=False, morph_failed — connectivity-only homolog '
        'result, kept in CSV, never rendered</td></tr>',
        f"<tr><td>{_term('family', 'family bin')}</td>"
        f"<td>{_esc(d['fam_rows_n'])}</td>"
        "<td>→ dedup "
        f"{_esc(d['dedup_cat'].get('family', 0))} bodyIds</td></tr>",
        f"<tr><td>{_term('relative', 'relative bin')}</td>"
        f"<td>{_esc(d['rel_rows_n'])}</td>"
        "<td>→ dedup "
        f"{_esc(d['dedup_cat'].get('relative', 0))} bodyIds</td></tr>",
        f"<tr><td>{_term('examinees', 'examinees bin')}</td>"
        f"<td>{_esc(cc.get('examinees', 0))}</td>"
        '<td>aggressive-only deep window</td></tr>',
    ]
    bins = ("<div style='overflow-x:auto'><table class='mv-table'>"
            '<thead><tr>'
            + _th('bin', 'The Rev 3.12 examinee category the branch '
                  'rows landed in.')
            + _th('branch-level rows', 'Rows in examinees.csv before '
                  'bodyId dedup.')
            + _th('dedup', 'Distinct bodyIds after cross-branch dedup '
                  '(candidates rows outrank family; DEDUP_RANK).')
            + '</tr></thead><tbody>'
            + ''.join(bin_rows) + '</tbody></table></div>')
    cb = d['cand_break']
    n_cross = len([r for r in d['restrictive']
                   if r.get('target_bodyId') not in d['cand_bin_ids']])
    n_cand_rows = sum(v['rows'] for v in cb.values())

    def _types(counter) -> str:
        return ' · '.join(f'{_esc(t)} {_esc(n)}'
                          for t, n in counter.most_common())

    br_rows = [
        ('in family — {T}(out-map)', cb['in_family']['rows'],
         len(cb['in_family']['bids']),
         'unmapped members of in-map types ('
         + _types(cb['in_family']['types']) + ')'),
        ('no_source — orphan types', cb['no_source']['rows'],
         len(cb['no_source']['bids']),
         'no backward route to the source (' +
         _types(cb['no_source']['types']) + ')'),
        ('backward mapped — {T}>{src}', cb['backward']['rows'],
         len(cb['backward']['bids']),
         'to ' + _types(cb['backward']['to'])),
    ]
    if cb['untyped']['rows']:
        br_rows.append(('untyped', cb['untyped']['rows'],
                        len(cb['untyped']['bids']),
                        'no type annotation'))
    br_html_rows = ''.join(
        f'<tr><td>{_esc(label)}</td><td>{rows_n}</td><td>{bids}</td>'
        f'<td>{meaning}</td></tr>'
        for label, rows_n, bids, meaning in br_rows)
    breakdown = (
        f"<p class='mv-note'>Candidates breakdown ({n_cand_rows} "
        'branch-level rows'
        + (f' + {_esc(n_cross)} further bodyIds fill at the candidates '
           'bar without a candidates row in examinees.csv — the query-'
           'level dedup assigns the category; their level/evidence is '
           'per row in the fill table below)'
           if n_cross else ')') + ':</p>'
        "<div style='overflow-x:auto'><table class='mv-table'>"
        '<thead><tr>'
        + _th('class', 'The candidate leaf class: {T}(out-map), '
              'no_source, backward mapped, or untyped.')
        + _th('rows', 'Branch-level candidate rows.')
        + _th('bodyIds', 'Distinct target bodyIds behind those rows.')
        + _th('meaning', 'What the class says about the backward '
              'route, with the type composition.')
        + '</tr></thead><tbody>' + br_html_rows
        + '</tbody></table></div>')
    # family-material reconciliation (user 2026-09-17): the bin's dedup
    # count is NOT the family-material total — the dedup rank puts
    # candidates above family (DEDUP_RANK: candidates 4 > family 3, pinned
    # by test), so a family-material bodyId that also has a candidates row
    # rolls up as `candidates` (listed below); one whose best row is
    # family rolls up as `family`.
    fam_rec = ''
    if d['fm_ids']:
        parts = [f"family-bin dedup {len(d['fm_family'])}"]
        if d['fm_cand']:
            ids = ', '.join(str(b) for b in sorted(d['fm_cand']))
            parts.append(f'candidates-bin {len(d["fm_cand"])} '
                         f'({_esc(ids)} — also family-evidenced; dedup '
                         'precedence candidates > family claims them for '
                         'the fill view, and the complete family list '
                         'stays in the Targets tab)')
        if d['fm_other']:
            ids = ', '.join(str(b) for b in sorted(d['fm_other']))
            parts.append(f'{len(d["fm_other"])} in no expansion bin '
                         f'({_esc(ids)})')
        fam_rec = (
            f"<p class='mv-note'><b>Family material "
            f"{len(d['fm_ids'])} (= {d['tgt'].get('mapped_target_set')} "
            f"in-map − {d['tgt'].get('in_branch_pool')} map-covered) = "
            + ' + '.join(parts) + '.</p>')
    sec5 = _section_card(
        'Expansion bins (Revision 3.12 distribution)',
        'One ordered first-match category per in-scope target per '
        'branch; branch-level rows vs the bodyId-unique dedup.',
        bins + breakdown + fam_rec,
        ['category', 'candidates', 'sibling', 'family', 'relative',
         'examinees', 'out of scope', 'candidate_annotation',
         '{T}(out-map)', '{T}>{src}', '{T}(no_source)', 'untyped', 'dup'])

    # §6 the fill
    lv = d['level_counts']
    lv_txt = ' · '.join(
        f'{k} {_esc(lv.get(k, 0))}'
        for k in ('high', 'medium', 'low', 'type_gated', 'advice'))
    ff_split: Dict[str, int] = {}
    for r in d['family_fill']:
        c = r.get('dedup_category', '?')
        ff_split[c] = ff_split.get(c, 0) + 1
    ff_txt = ' + '.join(f'{k} {v}' for k, v in sorted(ff_split.items()))
    # Gap bins only: the unmatched pool control is graded the same way, and
    # counting it here would state a far larger reciprocal figure than the
    # per-bin breakdown beside it.
    n_bev = sum((d['backward_bins'].get(k) or {}).get('high', 0)
                for k in _BEV_GAP_BINS)
    n_thin = sum(1 for r in d['backward_rows'] if _is_thin(r))
    # D7: the reverse fact splits each gap-fill bin WITHOUT joining the level
    # ladder, so the split reads as its own axis next to the level counts.
    bins = d['backward_bins']
    bev_parts = []
    for c in _BEV_BIN_ORDER:
        b = bins.get(c) or {}
        if not b.get('neurons'):
            continue
        parts = [f'{_esc(_BEV_TEXT[v])} {b[v]}'
                 for v in ('high', 'medium', 'low') if b.get(v)]
        if b.get('not-checked'):
            parts.append(f'not checked {b["not-checked"]}')
        bev_parts.append(f'{_esc(c)} {b["neurons"]} scanned → '
                         + (' · '.join(parts) if parts else 'nothing ranked'))
    summ = _kv_block('Fill accounting', [
        (_term('fill levels', 'Fill by level (branch-level)'), lv_txt),
        (_term('restrictive fill', 'Fill, bodyId-unique (dedup)'),
         f'{len(d["restrictive"])} restrictive · family-fill '
         f'+{len(d["family_fill"])} ({ff_txt})'),
        (_term('reciprocal', 'Reciprocal (stage 5d)'),
         f'{n_bev} gap-fill members rank their own branch source type in the '
         'reverse top-1 — advisory; it never changes a level or a fill'
         + (f' · {n_thin} hit(s) rest on ≤3 shared partner types '
            '(<b>thin</b>)' if n_thin else '')),
        *([(_term('reciprocal', 'Reverse evidence by bin'),
            ' · '.join(bev_parts))] if bev_parts else []),
        ('Proposals exported',
         f'{sum(d["fill_class_counts"].values())} rows ('
         + ' / '.join(f'{k} {v}' for k, v in
                      sorted(d['fill_class_counts'].items()))
         + '; side: '
         + ' / '.join(f'{k} {v}' for k, v in
                      sorted(d['side_counts'].items())) + ')'),
    ])
    rows = []
    for r in d['restrictive']:
        bid = r.get('target_bodyId', '')
        lvl = d['levels_by_bid'].get(bid) or {}
        sus = d['sus_cand_by_bid'].get(bid)
        if sus is not None:
            tok = sus.get('candidate_annotation', '?')
            kind = sus.get('bar_kind', '')
            bar_txt = f'{_f(sus.get("bar_value"))} ({kind})' \
                if kind else _f(sus.get('bar_value'))
            prov = 'expansion row'
        else:
            prop = d['prop_by_bid'].get(bid)
            bar_txt = '—'
            if prop:
                tok = f"{r.get('target_type', '?')}(out-map)"
                prov = ('out-of-pool same-type proposal '
                        f"({prop.get('source_type')}→"
                        f"{prop.get('target_type')}, source verdict "
                        f"{prop.get('source_verdict')})")
            else:
                # I-1 (2026-09-28): most cross-bin fills never had a
                # proposal row — the query-level dedup assigned the bin.
                # Say that, with the levels row as the provenance, and
                # stop borrowing the out-map token.
                tok = f"{r.get('target_type', '?')}(dedup-fill)"
                prov = ('no candidates row in examinees.csv — the '
                        'query-level dedup assigned the bin')
                if lvl.get('evidence'):
                    prov += f" (evidence {lvl.get('evidence')})"
        brec = d['backward_by_bid'].get(str(bid)) or {}
        bev = str(brec.get('backward_evidence') or '')
        rows.append(
            f'<tr><td>{_esc(bid)}</td>'
            f"<td>{_esc(r.get('target_type'))}</td>"
            f'<td>{_esc(tok)}</td><td>{_esc(bar_txt)}</td>'
            f"<td>{_esc(lvl.get('level', '—'))}</td>"
            f"<td>{_esc(lvl.get('dup', r.get('dup', '')))}</td>"
            f'<td>{_bev_badge(bev, _badge_thin(brec))}</td>'
            f'<td>{_esc(prov)}</td></tr>')
    table = _scroll_viewport(
        rows,
        _th('target bodyId', 'The proposed fill neuron (bodyId-unique '
            'across branches).')
        + _th('type', 'Its target-side type.')
        + _th('leaf token', 'Per-bodyId provenance token: '
              '{T}(out-map) / >src / (no_source) / untyped; '
              '{T}(dedup-fill) marks a fill that had no candidates row '
              'in examinees.csv (the dedup assigned the bin).')
        + _th('bar (kind)', 'Branch morph admission bar and its kind: '
              'native / track_a_backup / null.')
        + _th('level', 'Fill level the bodyId lands at: high / '
              'type_gated / advice.')
        + _th('dup', 'Whether another branch also proposes this '
              'bodyId (the dedup keeps one row).')
        + _th('reciprocal', 'Reverse-scan grade of this member: high '
              '/ medium / low — details in the Reciprocal tab.')
        + _th('provenance', 'Which branch proposed it and on what '
              'evidence.')) \
        if rows else _empty('No restrictive fill proposed this run.')
    footer = ("<p class='mv-note'><b>Proposals only</b> — fills are "
              'ranked evidence, not deterministic assignments; '
              '<i>matched</i> is the only asserted tier; the mapping is '
              'never rewritten.</p>')
    sec6 = _section_card(
        'The fill (proposals only — the mapping is never rewritten)',
        'The bodyId-unique restrictive fill; this is the table a '
        'reviewer acts on.',
        summ + table + footer,
        ['restrictive fill', 'family fill', 'fill levels', 'dup',
         '{T}(out-map)', 'reciprocal', 'high', 'medium', 'low',
         'not-checked', 'thin evidence'])
    return sec5 + sec6


def _outmap_tab(d: Dict) -> str:
    if not d['out_rows_total']:
        total_q = d['src'].get('total_queried')
        unclaimed = (total_q - d['n_with_branch']
                     if isinstance(total_q, (int, float))
                     and d.get('n_with_branch') is not None else None)
        detail = (
            f'{unclaimed} source neurons have no branch, but the '
            'expansion produced 0 candidate rows (sparse typed universe '
            'on the target side)' if unclaimed else
            'No out-map expansion rows (every source neuron is claimed '
            'by a branch, or the stage was skipped).')
        return _section_card('Out-map expansion', '', _empty(detail))
    null_bar = _f(d['calib'].get('track_a_null_bar'))
    pct = _pct(d['out_morph_pass'], d['out_rows_total'])
    top = ' · '.join(
        f'{_esc(t)} {_esc(n)}' for t, n in
        sorted(d['out_type_counts'].items(), key=lambda kv: -kv[1]))
    n_with_q = len(d['per_source_best'])
    summ = _kv_block('Out-map expansion summary', [
        ('Sources expanded',
         f'{len(d["out_sources"])} (top-k typed non-in-map candidates '
         f'each = {d["out_rows_total"]} rows)'),
        ('Morph-checked',
         f'vs run null bar {_esc(null_bar)} (p95) → '
         f'{d["out_morph_pass"]}/{d["out_rows_total"]} pass ({pct}); '
         f'{n_with_q}/{len(d["out_sources"])} sources have ≥1 '
         'morph-qualified candidate'),
        ('Top candidate types', top),
    ])
    structural = [t for t in d['out_type_counts']
                  if re.match(r'^R1-R\d$', t)]
    structural_note = ''
    if structural:
        n = sum(d['out_type_counts'][t] for t in structural)
        structural_note = (
            "<div class='mv-callout'><b>"
            f"{_esc(', '.join(structural))} ({n} rows)</b> is the "
            'photoreceptor population — labeled by real crosswalk, '
            'structural, never a homolog claim; its rows are '
            'morph_qualified=False by construction.</div>')
    rows = []
    for item in d['per_source_best']:
        r = item['row']
        q = '✓' if _truthy(r.get('morph_qualified')) else '✗'
        # A mark beside a number the mark was not compared against is the
        # defect #58/#61 exist for: the row now carries its bar, so print it.
        # An archived export has no `morph_bar`, and inventing one there would
        # grade a run that never published its floor.
        bar = _f(r.get('morph_bar'))
        vs = f' vs {bar}' if r.get('morph_bar') not in (None, '') and bar \
            else ''
        rows.append(
            f"<tr><td>{_esc(r.get('source_type'))} {_esc(item['src'])}"
            f'</td>'
            f"<td>{_esc(r.get('target_bodyId'))} "
            f"{_esc(r.get('target_type'))}</td>"
            f"<td>{_f(r.get('rank_union'))}</td>"
            f"<td>{_f(r.get('jaccard'))}</td>"
            f"<td>{_f(r.get('morph_v2_similarity'))}{vs} {q}</td>"
            f"<td>{item['n_q']}/{item['n_total']}</td></tr>")
    table = _scroll_viewport(
        rows,
        _th('source', 'The unclaimed source neuron (type + bodyId).')
        + _th('best candidate', 'Its best-ranked typed non-in-map '
              'target candidate.')
        + _th('rank_union', 'Rank-agreement score of the pair — near '
              '-0.8 means disjoint partner vectors, so 0 is mid-scale, '
              'not nothing.')
        + _th('jaccard', 'Shared-partner jaccard of the pair.')
        + _th('morph ✓/✗', 'morph_v2_similarity against the run null '
              'bar: ✓ passes, ✗ does not. The cell prints the bar it was '
              'graded against (`morph_bar`, `null` kind) so the mark is '
              'recomputable from the row; an export older than that column '
              'prints the score alone.')
        + _th('qualified', "Of the source's top-k candidates, how many "
              'pass the null bar over how many were scored.'))
    return _section_card(
        f'Out-map expansion (the {len(d["out_sources"])} unclaimed '
        'sources)',
        'Top-k typed non-in-map candidates per unclaimed source, '
        'morph-checked against the run null bar — mapper-gap evidence, '
        'not fills.',
        summ + structural_note + table,
        ['out-map expansion', 'null bar', 'untyped'])


def _backward_tab(d: Dict) -> str:
    """Standalone backward panel (plan-backward-source-status.md, D-B4):
    primary = row-evidence linker-supported in-map counts + out-map count;
    interpretation layer = the advisory `source-` status distribution and
    the source-candidates regroup. Informational — the targets remain the
    validated entities."""
    scov, tcov = d['src'], d['tgt']
    basis = d['basis_sources']
    row_bases, wide_bases = split_basis_buckets(basis)
    n_linker = sum(len(basis[b]) for b in row_bases)
    n_full = sum(len(basis[b]) for b in wide_bases)
    row_label = ' / '.join(row_bases) or 'row evidence'
    wide_label = ' / '.join(wide_bases) or 'name assertion'
    total_q = scov.get('total_queried', '—')
    out_n = (total_q - d['n_with_branch']
             if isinstance(total_q, (int, float))
             and d.get('n_with_branch') is not None
             else len(d['out_sources']))
    primary = _kv_block('Source population — where every neuron sits', [
        ('In-map (branch pools)',
         f'{n_linker + n_full} of {_esc(total_q)} — row-evidence backed '
         f'{n_linker} ({_esc(row_label)}) · same-name pooled {n_full} '
         f'({_esc(wide_label)})'),
        ('Out-map (no branch anywhere)', out_n),
    ])
    cards = [_section_card(
        'Backward — source accounting (informational)',
        'Primary counts per D-B4: the sources sit either in branch pools '
        '(swept in by linker row evidence) or out-map. The advisory '
        "`source-` statuses below describe each in-branch source's "
        'column standing — they never gate, never enter the dedup, and '
        'never rewrite the mapping (D-B11).',
        primary, ['map-covered', 'validation mode'])]

    # status distribution + per-type table
    if d['ss_rows']:
        order = ['source-matched', 'source-verified', 'source-borderline',
                 'source-unmatched']
        dist = ' · '.join(
            f'{_term(k, k)} {d["ss_counts"].get(k, 0)}' for k in order)
        dist_block = f"<p class='mv-note'>{dist}</p>"
        pt_rows = []
        for stype in sorted(d['per_type_status'],
                            key=lambda k: -sum(d['per_type_status'][k]
                                               .values())):
            c = d['per_type_status'][stype]
            pt_rows.append(
                f'<tr><td>{_esc(stype)}</td>'
                f"<td>{_esc(c.get('source-matched', 0))}</td>"
                f"<td>{_esc(c.get('source-verified', 0))}</td>"
                f"<td>{_esc(c.get('source-borderline', 0))}</td>"
                f"<td>{_esc(c.get('source-unmatched', 0))}</td></tr>")
        pt_table = _scroll_viewport(
            pt_rows,
            _th('source type', 'The queried source-side type.')
            + ''.join(_th(k, TERM_DEFS[k])
                      for k in ('source-matched', 'source-verified',
                                'source-borderline',
                                'source-unmatched')))
        cards.append(_section_card(
            'Advisory `source-` status distribution',
            'The column view of the same pair scores: is this source the '
            'best source of some pool target (column-top-1), a runner-up '
            'behind in-branch competitors, or dominated by out-of-branch '
            'sources?',
            dist_block + pt_table,
            ['source-matched', 'source-verified', 'source-borderline',
             'source-unmatched']))

    # source-candidates: the exported `source_candidates.csv` rows (the
    # SAME rows the scene renders) — RE-AIMED (plan §10, user option 2):
    # out-of-MAP sources whose best-ranked hits land in a branch pool,
    # morph-qualified against the run null bar (D-B8's mirror of
    # candidate admission).  Legacy folders without the artifact fall
    # back to the sibling-row re-derivation (cross-branch convergence
    # view; those rows are other branches' query neurons by
    # construction).
    sc_export = _read_csv_rows(
        _run_file(Path(d['run_dir']), 'source_candidates.csv'))
    sib = ([r for r in _read_examinees(Path(d['run_dir']))
            if r.get('category') == 'sibling']
           if not sc_export else [])
    if sc_export:
        by_branch: Dict[str, set] = collections.defaultdict(set)
        types_of: Dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter)
        for r in sc_export:
            label = (f"{r.get('branch_source_type', '?')} → "
                     f"{r.get('branch_target_type', '?')}")
            bid = int(r['source_bodyId'])
            by_branch[label].add(bid)
            types_of[label][r.get('source_type') or '?'] += 1
        if by_branch:
            rows = []
            for label in sorted(by_branch,
                                key=lambda k: -len(by_branch[k])):
                tc = ' · '.join(f'{_esc(t)} {_esc(n)}' for t, n in
                                sorted(types_of[label].items(),
                                       key=lambda kv: -kv[1]))
                rows.append(
                    f'<tr><td>{_esc(label)}</td>'
                    f"<td>{len(by_branch[label])}</td>"
                    f'<td>{tc}</td></tr>')
            sc_table = (
                "<div style='overflow-x:auto'><table class='mv-table'>"
                '<thead><tr>'
                + _th('target branch', 'The branch pool the sources '
                      'reach, as source type → target type.')
                + _th('distinct out-of-map sources', 'Source neurons '
                      'claimed by NO branch whose best-ranked scan hits '
                      'land in this pool (null-bar morph-checked).')
                + _th('their types', 'Type composition of those '
                      'sources.')
                + '</tr></thead><tbody>' + ''.join(rows)
                + '</tbody></table></div>')
            cards.append(_section_card(
                'Source-candidates — out-of-map sources reaching this '
                "run's branch pools",
                'The D-B8 backward mirror of candidate admission: sources '
                'claimed by NO branch whose best-ranked scan hits land in '
                'a branch pool AND pass the run null bar (morph). '
                'Advisory — never a fill; inclusion of a candidate goes '
                'through user verification.',
                sc_table,
                ['source-candidates', 'source-candidates (dup)']))
    elif sib:
        tgt_branch = {}
        try:
            import ast as _ast
            for r in _read_csv_rows(
                    _run_file(Path(d['run_dir']), 'mapping_export.csv')):
                if not _truthy(r.get('is_selected')):
                    continue
                try:
                    tids = _ast.literal_eval(
                        r.get('target_body_ids') or '[]')
                except (ValueError, SyntaxError):
                    continue
                for b in tids:
                    tgt_branch[int(b)] = (r['source_type'],
                                          r['target_type'])
        except Exception:  # noqa: BLE001
            tgt_branch = {}
        by_branch: Dict[str, set] = collections.defaultdict(set)
        types_of: Dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter)
        for r in sib:
            t = int(r['ahead_target_bodyId'])
            b = tgt_branch.get(t)
            if b is None:
                continue
            label = f'{b[0]} → {b[1]}'
            by_branch[label].add(int(r['source_bodyId']))
            types_of[label][r.get('source_type') or '?'] += 1
        if by_branch:
            rows = []
            for label in sorted(by_branch,
                                key=lambda k: -len(by_branch[k])):
                tc = ' · '.join(f'{_esc(t)} {_esc(n)}' for t, n in
                                sorted(types_of[label].items(),
                                       key=lambda kv: -kv[1]))
                rows.append(
                    f'<tr><td>{_esc(label)}</td>'
                    f"<td>{len(by_branch[label])}</td>"
                    f'<td>{tc}</td></tr>')
            sc_table = (
                "<div style='overflow-x:auto'><table class='mv-table'>"
                '<thead><tr>'
                + _th('target branch', 'The branch pool the sources '
                      'rank ahead of, as source type → target type.')
                + _th('distinct out-of-branch sources', 'Source neurons '
                      'of OTHER branches that rank ahead of this '
                      "branch's pool members (sibling rows).")
                + _th('their types (sibling rows)', 'Type composition '
                      'of those out-of-branch sources.')
                + '</tr></thead><tbody>' + ''.join(rows)
                + '</tbody></table></div>')
            cards.append(_section_card(
                'Source-candidates — out-of-branch sources connecting '
                "to this run's mapped targets",
                'Out-of-branch sources whose qualified sibling rows point '
                'into a branch pool (connectivity + morph, owning-branch '
                'bar, v1) — by construction other branches\' query '
                'neurons (cross-branch convergence). Advisory — never a '
                'fill.',
                sc_table, ['source-candidates']))

    if not d['ss_rows'] and not sib and not basis:
        return ''.join(cards) or _empty('No backward data.')
    disclaimer = (
        "<p class='mv-note'><b>Informational only</b> — the targets "
        'remain the validated entities. Backward statuses never gate, '
        'never enter the dedup, and never rewrite the mapping (D-B11). '
        'Unpaired sources are typically column runners-up of '
        'already-matched/verified targets (population surplus + N-to-1 '
        'convergence), not mapping failures.</p>')
    cards.append(_section_card('How to read this panel', '', disclaimer,
                               ['coverage levels']))
    return ''.join(cards)


_BEV_TEXT = {'high': 'high', 'medium': 'medium', 'low': 'low'}
_BEV_CSS = {'high': 'bev-high', 'medium': 'bev-medium',
            'low': 'bev-low'}
#: the gap-fill bins the pass exists for, then the unmatched pool members.
#: matched / verified / borderline are NOT scanned (already mapped — the
#: symmetric forward score is their evidence); legacy run folders that
#: still carry those rows fall through to the idx-99 ordering below.
_BEV_BIN_ORDER = ('candidates', 'family', 'relative', 'unmatched')
#: the bins the pass exists for; `unmatched` is the in-map control and is
#: counted per bin for display, but it is not reciprocal *evidence* about a
#: gap, so the headline counts and the warning line stay inside this subset.
_BEV_GAP_BINS = ('candidates', 'family', 'relative')


def _bev_badge(lab: str, thin: bool = False) -> str:
    out = ("<span class='bev "
           f"{_BEV_CSS.get(lab, 'bev-off')}'>"
           f"{_esc(_BEV_TEXT.get(lab, 'not checked'))}</span>")
    if thin:
        out += " <span class='bev bev-thin'>thin</span>"
    return out


def _cnt(v) -> str:
    """A type count as an integer, or an em dash when never scored."""
    n = _as_num(v)
    return '—' if n is None else str(int(n))


def _evidence_cell(r: Dict) -> str:
    """``shared/union`` partner types behind this row's rank_union — the
    denominator a rank similarity is actually built on."""
    return _term('shared partner types',
                 f'{_cnt(r.get("backward_shared_type_count"))}'
                 f'/{_cnt(r.get("backward_union_type_count"))}')


def _is_thin(r: Dict) -> bool:
    return _truthy(r.get('backward_thin_evidence'))


def _own_type_cell(r: Dict) -> str:
    """The reverse hit the GRADE actually rests on: the best-ranked source
    of the claiming branch's own type, with the rank and the ranking that
    placed it there (plan D1c).

    Kept separate from the top-1 cell because the two are different
    neurons whenever the branch's type wins on jaccard but not on
    rank_union — 20 of the 49 high/medium rows in the r15 run."""
    bid = r.get('backward_own_type_rank_source_bodyId')
    if bid in (None, ''):
        return "<span class='missing'>—</span>"
    via = str(r.get('backward_own_type_via') or '')
    rank_col = ('backward_own_type_jaccard_rank' if via == 'jaccard'
                else 'backward_own_type_rank_union_rank')
    rk = _as_num(r.get(rank_col))
    otype = str(r.get('backward_own_type_rank_source_type') or '')
    txt = (f'{_esc(str(bid))} · {_esc(otype or "(untyped)")}'
           f"<span class='mv-note'> · "
           f'{"#" + str(int(rk)) if rk is not None else "#—"}'
           f' by {_esc(via or "—")}'
           f' · jac {_f(r.get("backward_own_type_jaccard"), 4)}'
           f' · ru {_f(r.get("backward_own_type_rank_union"), 4)}'
           '</span>')
    if _truthy(r.get('backward_own_type_thin_evidence')):
        txt += " <span class='bev bev-thin'>thin</span>"
    return txt


def _badge_thin(r: Dict) -> bool:
    """The thin marker belongs on the evidence the grade rests on: the
    own-type hit when there is one, else the top-1 row's base."""
    if r.get('backward_own_type_rank_source_bodyId') not in (None, ''):
        return _truthy(r.get('backward_own_type_thin_evidence'))
    return _is_thin(r)


def _topn_hover(raw, head: str = 'top-N reverse hits (Jaccard order)') -> str:
    """Render a ``backward_topN`` / union-payload cell as the hover's mini
    table.

    Records are ``ru_rank|jac_rank|bid|type|ru|jaccard|in_branch`` and the
    list arrives in chain order (Jaccard first), so BOTH ranks travel: a
    hit can be jaccard 1 while ranking poorly by rank_union, and one rank
    column would hide that.  ``head`` names the list the caller serialized
    (the reciprocal chain top-N vs the homolog union window)."""
    recs = [r for r in str(raw or '').split(';') if r.strip()]
    if not recs:
        return ''
    rows = []
    for rec in recs:
        f = (rec.split('|') + ['', '', '', '', '', '', ''])[:7]
        rows.append(
            f'<tr><td>{_esc(f[1] or "—")}</td>'
            f'<td>{_esc(f[0] or "—")}</td>'
            f'<td>{_esc(f[2] or "—")}</td>'
            f'<td>{_esc(f[3] or "(untyped)")}</td>'
            f'<td>{_esc(f[4] or "—")}</td>'
            f'<td>{_esc(f[5] or "—")}</td>'
            f'<td>{"this branch" if f[6] == "1" else "elsewhere"}</td></tr>')
    return (f'<b>{head}</b>'
            '<table><thead><tr>'
            "<th title='Rank by jaccard; the list is ordered by this.'>"
            '#jac</th>'
            "<th title='Rank by rank_union for the same hit.'>#ru</th>"
            "<th title='Source-side bodyId of the hit.'>source</th>"
            "<th title='Its source type; (untyped) when blank.'>type</th>"
            "<th title='Rank-agreement score of the pair.'>rank_union</th>"
            "<th title='Shared-partner jaccard of the pair.'>jaccard</th>"
            "<th title='Hits inside the claiming branch pool read "
            "this branch; all other hits read elsewhere.'>where</th>"
            '</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def _as_num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _store_stamp(value) -> str:
    """A `_store_identity` mtime (epoch SECONDS, the key `mtime_s`) as a UTC
    minute — what a reader lines up between two runs.

    Reading the wrong key here prints a bare `@ None`, which a unit fixture
    cannot catch unless it carries the generator's real shape (a 2026-09-23
    real-data run caught exactly that).
    """
    sec = _as_num(value)
    if sec is None:
        return ''
    try:
        return datetime.fromtimestamp(sec, tz=timezone.utc).strftime(
            '%Y-%m-%d %H:%M UTC')
    except (OverflowError, OSError, ValueError):
        return ''


def _reciprocal_row_sort(r: Dict) -> Tuple[float, float, float, int]:
    """The reciprocal list's default order: the Jaccard of the
    BRANCH-TYPE HIT, descending.

    Not the top-1's Jaccard: the badge is a statement about the branch's
    own source type, and a `low` neuron's unrelated top-1 can score high —
    so ordering by it floated graded negatives above `high` rows (refines
    plan-tmvev-reciprocal-jaccard-sort-and-parity D2, user 2026-09-20).
    Ties break on that hit's jaccard rank, then on the top-1's jaccard, so
    the `low` rows — which publish no hit and therefore sink together — keep
    a meaningful order among themselves.  bodyId makes the key total.
    """
    j = _as_num(r.get('backward_own_type_jaccard'))
    jr = _as_num(r.get('backward_own_type_jaccard_rank'))
    t1 = _as_num(r.get('backward_jaccard'))
    try:
        bid = int(str(r.get('member_bodyId') or '0'))
    except ValueError:
        bid = 0
    return (-(j if j is not None else -1e9),
            jr if jr is not None else 1e9,
            -(t1 if t1 is not None else -1e9),
            bid)


def _reciprocal_tab(d: Dict) -> str:
    """Stage 5d (plan-tmvev-backward-expansion-evidence): the candidates /
    family / relative members and the unmatched pool members scanned BACK
    against the whole SOURCE universe — the homolog finding run in the
    reverse direction, one display row per neuron (graded per claiming
    branch), ordered by the branch-type hit's Jaccard, the hit the grade
    rests on shown beside the global top-1.  Connectivity only, advisory
    only."""
    rows = d['backward_rows']
    params = d['params']
    bins = d['backward_bins']
    ctr = d['backward_counters']
    if not rows:
        return _section_card(
            'Reciprocal homolog evidence',
            'What each gap-fill member prefers when IT is scanned back '
            'against the whole source universe.',
            _empty('backward_matches.csv absent — the pass was skipped '
                   '(--no-backward-evidence / the UI checkbox) or no '
                   'member of the three bins was scanned.'),
            ['reciprocal', 'not-checked'])

    def _bin_rows():
        out = []
        order = ([b for b in _BEV_BIN_ORDER if b in bins]
                 + sorted(k for k in bins if k not in _BEV_BIN_ORDER))
        for cat in order:
            b = bins[cat]
            label = (cat if cat in _BEV_BIN_ORDER
                     else f'{cat} (in-map control)')
            out.append(
                f'<tr><td>{_esc(label)}</td>'
                f'<td>{b["neurons"]:,}</td>'
                f'<td>{b["high"]:,}</td>'
                f'<td>{b["medium"]:,}</td>'
                f'<td>{b["low"]:,}</td>'
                f'<td>{_pct(b["high"], b["neurons"])}</td></tr>')
        return out
    scanned = _as_num(ctr.get('distinct_scanned'))
    beyond = _as_num(ctr.get('beyond_cap'))
    eligible = ('—' if scanned is None or beyond is None
                else f'{int(scanned + beyond):,}')
    budget = (
        f"<p class='mv-note'>Budget: "
        f"{_esc(ctr.get('distinct_scanned', '—'))} of {eligible} eligible "
        f"members scanned (cap "
        f"{_esc(params.get('backward_max_neurons', '—'))} per run, "
        f"{_esc(params.get('backward_per_branch_cap', '—'))} per branch); "
        'the rest carry no row and read as <b>not checked</b>. '
        'Morphology: not evaluated (connectivity-only pass).</p>')
    below_k = _as_num(ctr.get('source_vectors_below_k'))
    if below_k:
        # sparsity, not staleness: the scorer is symmetric and both vector
        # paths were verified identical, so this only says how much of the
        # scanned universe carries a short partner vector
        budget += (
            f"<p class='mv-note'>Source sparsity: {int(below_k):,} of the "
            "scanned universe neurons hold fewer partner types than this "
            "run's top_k, so their rank_union sits nearer the "
            "disjoint-vector floor (about -0.8). Ranking within a scan is "
            'unaffected; read an absolute rank_union bar against that '
            'floor.</p>')
    cards = [_section_card(
        f'Reciprocal scan — {len(d["backward_by_bid"]):,} neurons',
        f'The {", ".join(_BEV_BIN_ORDER)} bins asked the reverse question '
        'with the same scorer: which source does this member prefer, and '
        'is it the one our branch claims? Matched / verified / borderline '
        'pool members are not scanned — they are already mapped, and the '
        'symmetric forward score is their evidence. Morphology is '
        'deliberately not re-run (candidates are already morph-qualified; '
        'family and relative members are morph-similar to the query or to '
        'those candidates), so this is connectivity evidence and nothing '
        'here moves a neuron between bins or counts toward a fill.',
        _scroll_viewport(
            _bin_rows(),
            _th('bin', 'The bin scanned back: the gap-fill bins '
                'candidates / family / relative, plus unmatched pool '
                'members. Matched / verified / borderline members are '
                'not scanned — the symmetric forward score is their '
                'evidence.')
            + _th('neurons', 'Distinct members of the bin that were '
                  'reverse-scanned.')
            + _th('high', 'Members whose own branch source type is the '
                  'top-1 reverse hit of EITHER ranking — rank_union or '
                  'jaccard; the branch-type hit column names which one.')
            + _th('medium', 'Members whose branch source type reaches '
                  'rank 2 or 3 inside a top-3 window of either ranking, '
                  'and rank 1 in neither.')
            + _th('low', 'Members whose branch source type is outside '
                  'both top-3 windows.')
            + _th('top-3 share', 'high + medium as a share of the '
                  'scanned neurons.'))
        + budget,
        ['reciprocal', 'reciprocal top-1', 'high', 'medium', 'low',
         'not-checked'])]

    # per-neuron rows, grouped by (bin, member type) — the display
    # contract: ONE row per neuron (a member claimed by several branches
    # was scanned once and graded per branch; the strongest branch
    # verdict is shown and every claiming branch is listed), top-1 and the
    # branch-type hit side by side, the full top-N on hover, rows ordered by
    # _reciprocal_row_sort.
    groups: Dict[Tuple[str, str], List[Dict]] = collections.defaultdict(list)
    for best in (d.get('backward_by_bid') or {}).values():
        groups[(str(best.get('member_category') or '?'),
                str(best.get('member_type') or '(untyped)'))].append(best)
    all_rows_by_bid = d.get('backward_rows_by_bid') or {}

    def _branch_rows(bid: str) -> List[Dict]:
        return all_rows_by_bid.get(bid) or []

    def _branch_cell(bid: str, best: Dict) -> str:
        branch_recs = _branch_rows(bid)
        if len(branch_recs) <= 1:
            return (f"{_esc(best.get('branch_source_type') or '?')} → "
                    f"{_esc(best.get('branch_target_type') or '?')}")
        parts = []
        for rr in sorted(branch_recs, key=_member_row_sort):
            parts.append(
                f"{_esc(rr.get('branch_source_type') or '?')} → "
                f"{_esc(rr.get('branch_target_type') or '?')} "
                f"({_esc(str(rr.get('backward_evidence') or '?'))})")
        return ' · '.join(parts)

    def _role_cell(bid: str, best: Dict) -> str:
        cats: List[str] = []
        for rr in _branch_rows(bid):
            c = str(rr.get('member_category') or '?')
            if c not in cats:
                cats.append(c)
        return _esc(', '.join(cats) or str(best.get('scan_role') or '?'))

    def _group_order(key):
        cat, tpe = key
        idx = _BEV_BIN_ORDER.index(cat) if cat in _BEV_BIN_ORDER else 99
        return (idx, cat, -len(groups[key]), tpe)

    _grade_rank = {'low': 1, 'medium': 2, 'high': 3}

    def _member_row_sort(rr):
        # strongest branch verdict first, then branch name for stability
        return (-_grade_rank.get(str(rr.get('backward_evidence') or ''), 0),
                str(rr.get('branch_target_type') or ''),
                str(rr.get('branch_source_type') or ''))

    blocks = []
    for key in sorted(groups, key=_group_order):
        cat, tpe = key
        members = sorted(groups[key], key=_reciprocal_row_sort)
        trs = []
        for r in members:
            lab = str(r.get('backward_evidence') or 'not-checked')
            bid = str(r.get('backward_top1_source_bodyId') or '')
            t1_type = str(r.get('backward_top1_source_type') or '')
            if bid:
                where = ('this branch' if _truthy(
                    r.get('backward_top1_in_branch')) else 'elsewhere')
                # J1: the row IS the Jaccard-best hit now, so its jaccard
                # rank is #1 by construction — the informative number is the
                # score, and rank_union travels in the hover as the
                # confirmation value (J2: never a filter).
                cell = (f'{_esc(bid)} · {_esc(t1_type or "(untyped)")}'
                        f"<span class='mv-note'> · {where} · "
                        f'jac {_esc(_f(r.get("backward_jaccard"), 4))}'
                        '</span>')
            else:
                cell = "<span class='missing'>—</span>"
            cell = _hover(cell, _topn_hover(r.get('backward_topN')))
            n_out = r.get('backward_n_out_of_branch')
            size = r.get('backward_size_filtered')
            bid_key = str(r.get('member_bodyId') or '?')
            trs.append(
                f'<tr><td>{_esc(r.get("member_bodyId") or "?")}</td>'
                f'<td>{_bev_badge(lab, _badge_thin(r))}</td>'
                f'<td>{_own_type_cell(r)}</td>'
                f'<td>{cell}</td>'
                f'<td>{_evidence_cell(r)}</td>'
                f'<td>{_branch_cell(bid_key, r)}</td>'
                f'<td>{_role_cell(bid_key, r)}</td>'
                f'<td>{_esc(n_out if n_out not in ("", None) else "—")}'
                '</td>'
                f'<td>{"size-filtered" if _truthy(size) else "—"}</td>'
                '</tr>')
        blocks.append(
            '<details class="detail-block"><summary>'
            f'{_esc(cat)} · {_esc(tpe)} — {len(members)} neuron'
            f'{"s" if len(members) != 1 else ""}</summary>'
            # one shared colgroup (item 3): every group table used to be
            # auto-layout, so each card sized its columns from its own
            # content and the cards above/below never lined up
            "<div style='overflow-x:auto'>"
            "<table class='mv-table mv-bev-table'>"
            '<colgroup>'
            '<col style="width:8%"><col style="width:7%">'
            '<col style="width:20%"><col style="width:20%">'
            '<col style="width:8%"><col style="width:20%">'
            '<col style="width:7%"><col style="width:5%">'
            '<col style="width:5%">'
            '</colgroup>'
            '<thead><tr>'
            + _th('member bodyId',
                  'The scanned neuron: one reverse scan per bodyId and '
                  'one display row per bodyId, however many branches '
                  'claim it.')
            + _th('reciprocal',
                  'How prominently this member prefers its OWN branch: '
                  'an advisory reverse-scan verdict that labels the row, '
                  'never a gate, never a fill count. high = the branch\'s '
                  'source type is top-1 by rank_union OR jaccard; medium = '
                  'inside a top-3 of either. For a multi-branch member '
                  'this is the strongest branch verdict.')
            + _th('branch-type hit (the grade)',
                  'The reverse hit the badge actually rests on: the '
                  'best-ranked source of the CLAIMING BRANCH\'S OWN type, '
                  'with the rank it reached and which ranking put it there. '
                  'This is a different neuron from the top-1 whenever the '
                  'branch type wins on jaccard but not on rank_union. Its '
                  'jaccard is what orders this list, descending; a row with '
                  'no hit (every low) sinks below the rows that have one.')
            + _th('top-1 source (Jaccard-best; hover: top-N)',
                  'The chain-best reverse hit — Jaccard first, rank_union '
                  'as the tie-break — and whether it sits in the claiming '
                  'branch. Its jaccard score is shown because its rank is #1 '
                  'by construction; rank_union travels in the hover as the '
                  'confirmation value, never as a filter. Hover for the '
                  'full top-N neighbourhood with both scores and both '
                  'ranks, listed in the same chain order.')
            + _th('shared/union types',
                  'How many partner types the TOP-1 row\'s score was '
                  'computed over — the denominator a rank_union or jaccard '
                  'rests on; the thin marker flags the few-shared cases. '
                  'The branch-type hit carries its own marker.')
            + _th('branch',
                  'The claiming branch (source type → target type) whose '
                  'pool was reversed against. A member claimed by several '
                  'branches lists each with its grade; the badge is the '
                  'strongest.')
            + _th('role',
                  'Which bin the member was scanned for: candidates, '
                  'family, relative, or an unmatched pool member. '
                  'Matched / verified / borderline members are not '
                  'scanned — the symmetric forward score is their '
                  'evidence.')
            + _th('out-of-branch ahead',
                  'How many source neurons OUTSIDE the branch pool rank '
                  'above the branch\'s own best source in this column.')
            + _th('caliber',
                  'Size ratio of the top-1 source against the branch '
                  'pool\'s best; below target_min_size_ratio the '
                  'comparison is flagged as an artifact.')
            + f'</tr></thead><tbody>{"".join(trs)}</tbody></table>'
            '</div></details>')
    cards.append(_section_card(
        'Per-neuron evidence, grouped by type',
        'One row per scanned neuron — a neuron claimed by several '
        'branches is graded per branch and shown once, with its strongest '
        'verdict and every claiming branch listed. Rows are ordered by the '
        'Jaccard of the <b>branch-type hit</b>, descending — the evidence '
        'the badge rests on, so a row whose own type never ranks sinks '
        'below the rows that have one. The <b>branch-type hit</b> column is '
        'the reverse hit the badge rests on (the best-ranked source of the '
        'claiming branch\'s own type, with the rank and the ranking that '
        'placed it there); the <b>top-1 source</b> column is the chain-best '
        'hit (Jaccard first, rank_union as the tie-break), and the two differ '
        'whenever the branch '
        'type wins on Jaccard alone. Hover the top-1 for the full top-N '
        'neighbourhood with both scores and both ranks.',
        ''.join(blocks),
        ['reciprocal top-1', 'branch-type hit', 'rank_union',
         'shared partner types', 'thin evidence']))
    return ''.join(cards)


# ---------------------------------------------------------------------------
# homolog panels (user 2026-09-26): per-bodyId forward/backward match sheets
# ---------------------------------------------------------------------------

def _branch_bar_for(bars: Dict, src_type: str, tgt_type: str) -> Dict:
    """The branch bar spec for a pair, accepting both key generations
    (plain ``src->tgt`` and query-prefixed ``query|src->tgt``)."""
    if not bars:
        return {}
    suffix = f'{src_type}->{tgt_type}'
    for key in sorted(bars):
        if key == suffix or key.endswith('|' + suffix):
            return bars.get(key) or {}
    return {}


def _homolog_morph_cell(row: Dict, bars: Dict) -> str:
    """The morph qualification of the row's PRIMARY pair, display-only:
    the values are the run's own exports and the ✓/✗ re-derives
    ``morph_bars.candidate_qualified`` against the branch bar (or the
    exported bar for out-map pairs) — nothing is re-scored here."""
    from comparison.morph_bars import BarSet, candidate_qualified
    sim = _as_num(row.get('morph_v2_similarity'))
    pref = _as_num(row.get('morph_pool_ref'))
    if sim is None and pref is None:
        return "<span class='missing'>not scored</span>"
    src_type = str(row.get('source_type')
                   or row.get('primary_source_type') or '')
    tgt_type = str(row.get('primary_target_type')
                   or row.get('target_type') or '')
    spec = _branch_bar_for(bars, src_type, tgt_type)
    if spec:
        bar = BarSet(
            candidate_kind=str(spec.get('candidate_kind') or 'null'),
            native_floor=_as_num(spec.get('native_floor')),
            backup_floor=_as_num(spec.get('backup_floor')),
            null_bar=_as_num(spec.get('null_bar')),
        )
        ok = candidate_qualified(bar, pref, sim)
        val = pref if bar.candidate_kind == 'native' else sim
        return (f"{'✓' if ok else '✗'} {_f(val, 4)} vs "
                f'{_f(bar.candidate_bar_value(), 4)} '
                f'({_esc(bar.candidate_kind)})')
    bar_val = _as_num(row.get('morph_bar'))
    kind = str(row.get('morph_bar_kind') or '')
    if bar_val is not None and sim is not None:
        return (f"{'✓' if sim >= bar_val else '✗'} {_f(sim, 4)} vs "
                f'{_f(bar_val, 4)} ({_esc(kind or "bar")})')
    shown = sim if sim is not None else pref
    return f"{_f(shown, 4)} <span class='mv-note'>(no bar in this run)</span>"


def _homolog_forward_tab(d: Dict) -> str:
    """Homolog · forward (user 2026-09-26): EVERY source bodyId that
    appeared in the run — assigned, fill-proposed, out-of-map or unpaired
    — with the target neuron the connectivity scan matched it to (the
    published chain-best hit), the morph qualification of that pair when
    the run scored it, and the union of the top-3 rank_union and top-3
    jaccard neighbourhood on hover.  One display row per bodyId in a
    single type-sorted table (user 2026-09-28): the allocation the run's
    artifacts give it stays a per-row column, so this is still the
    worksheet the user allocates candidates / family / relative /
    examinees and all other appeared bodyIds from."""
    if not d.get('forward_available'):
        return _section_card(
            'Homolog · forward',
            'One row per source bodyId the run touched, with its '
            'connectivity match in the target dataset.',
            _empty('forward_matches.csv absent — this run predates the '
                   'homolog panels; re-run the pipeline to populate the '
                   'panel.'),
            ['homolog forward', 'union window', 'not scored'])
    rows = d['forward_rows']
    if not rows:
        return _section_card(
            'Homolog · forward', '',
            _empty('forward_matches.csv is present but empty — the run '
                   'had no source neurons to match.'),
            ['homolog forward'])
    bars = d['branch_bars']
    alloc_by_bid = d['forward_alloc']
    branches_by_src: Dict[str, List[str]] = {}
    for pr in d['pair_rows']:
        st = str(pr.get('source_type') or '')
        tt = str(pr.get('target_type') or '')
        if st and tt:
            lst = branches_by_src.setdefault(st, [])
            if tt not in [x.split(' → ')[-1] for x in lst]:
                lst.append(f'{st} → {tt}')

    def _alloc_label(bid: str) -> str:
        a = alloc_by_bid.get(bid) or {}
        parts = list(a.get('bins') or [])
        if a.get('fill'):
            parts.append('fill-proposed')
        if a.get('out_map'):
            parts.append('out-of-map')
        if parts:
            return '+'.join(parts)
        v = str(a.get('verdict') or '')
        if not v:
            return 'appeared'
        return 'unpaired' if v == 'unmatched' else f'mapped · {v}'

    def _alloc_detail(bid: str) -> str:
        a = alloc_by_bid.get(bid) or {}
        notes = []
        if a.get('verdict'):
            notes.append(f"verdict {a['verdict']}")
        if a.get('source_status'):
            notes.append(str(a['source_status']))
        return (f"<span class='mv-note'> · {_esc(' · '.join(notes))}</span>"
                if notes else '')

    def _row_sort(r):
        j = _as_num(r.get('primary_jaccard'))
        return (-(j if j is not None else -1e9),
                str(r.get('source_bodyId') or ''))

    scanned_counts: Dict[str, int] = collections.Counter(
        str(r.get('scanned_at') or '?') for r in rows)
    n_scored = sum(1 for r in rows
                   if _as_num(r.get('morph_v2_similarity')) is not None
                   or _as_num(r.get('morph_pool_ref')) is not None)
    summary = _section_card(
        f'Homolog · forward — {len(rows):,} source bodyIds',
        'Every source bodyId the run touched, one row each, sorted by '
        'type and then by the primary match\'s Jaccard. The '
        '<b>primary match</b> is the published chain-best hit (Jaccard '
        'first, rank_union tie-break); its hover carries the union of the '
        'top-3 rank_union and top-3 jaccard neighbourhood. Morphology is '
        'the run\'s own export for that pair — displayed, never re-scored '
        'here.',
        "<p class='mv-note'>Scanned: "
        f"{scanned_counts.get('run', 0):,} of {len(rows):,} against the "
        'full target dataset · '
        f"{scanned_counts.get('no_profile', 0):,} without a usable "
        f"profile · {scanned_counts.get('error', 0):,} scan errors · "
        f"{n_scored:,} with a morph value on the primary pair.</p>"
        + _scroll_viewport(
            [f'<tr><td>{_esc(k)}</td><td>{v:,}</td></tr>'
             for k, v in sorted(
                 collections.Counter(_alloc_label(str(r.get(
                     'source_bodyId') or '?')) for r in rows).items(),
                 key=lambda kv: kv[1], reverse=True)],
            _th('allocation', 'The bins/status the bodyId holds across '
                  "the run's artifacts, spelled per row below.")
            + _th('bodyIds', 'Rows in the table below.')),
        ['homolog forward', 'primary match', 'union window', 'allocation',
         'not scored'])

    trs = []
    for r in sorted(rows, key=lambda r: (
            str(r.get('source_type') or 'untyped').lower(),) + _row_sort(r)):
        bid = str(r.get('source_bodyId') or '?')
        label = _alloc_label(bid)
        p_bid = str(r.get('primary_target_bodyId') or '')
        if p_bid:
            where = ('in a branch pool'
                     if _truthy(r.get('primary_in_branch'))
                     else 'outside the branch pools')
            p_type = str(r.get('primary_target_type') or 'untyped')
            cell = (f'{_esc(p_bid)} · {_esc(p_type)}'
                    f"<span class='mv-note'> · {where} · "
                    f'jac {_f(r.get("primary_jaccard"), 4)} · '
                    f'ru {_f(r.get("primary_rank_union"), 4)}'
                    '</span>')
            cell = _hover(cell, _topn_hover(
                r.get('forward_topN'),
                'homolog neighbourhood — top-3 rank_union ∪ top-3 '
                'jaccard (chain order)'))
        else:
            cell = "<span class='missing'>—</span>"
        scanned = str(r.get('scanned_at') or '')
        if scanned == 'run':
            scan_cell = f'{_cnt(r.get("n_scanned"))} targets'
        elif scanned == 'no_profile':
            scan_cell = _term('no profile')
        elif scanned == 'error':
            scan_cell = "<span class='missing'>scan error</span>"
        else:
            scan_cell = _esc(scanned or '—')
        brs = branches_by_src.get(str(r.get('source_type') or ''), [])
        trs.append(
            f'<tr><td>{_esc(bid)}</td>'
            f'<td>{_esc(str(r.get("source_type") or "untyped"))}</td>'
            f'<td>{_esc(label)}{_alloc_detail(bid)}</td>'
            f'<td>{cell}</td>'
            f'<td>{_homolog_morph_cell(r, bars)}</td>'
            f'<td>{_esc(" · ".join(brs) or "—")}</td>'
            f'<td>{scan_cell}</td>'
            f'</tr>')
    table = ("<div style='overflow-x:auto'><table class='mv-table'>"
             '<thead><tr>'
             + _th('source bodyId', 'The source-dataset neuron this run '
                   'mapped from.')
             + _th('type', 'Its source-side type.')
             + _th('allocation', 'The bins/status this bodyId holds across '
                   "the run's artifacts, with the mapping verdict and the "
                   'backward source status as notes.')
             + _th('primary match (hover: top-3 ∪ top-3)',
                   'The chain-best target hit; hover for the union '
                   'neighbourhood with both ranks and both scores.')
             + _th('morph', 'The primary pair\'s morph qualification, '
                   're-derived offline from the run\'s own exports and its '
                   'branch bar — display only.')
             + _th('branches', 'The branches this source type maps through.')
             + _th('scanned', 'How many target neurons the scan ranked, or '
                   'why it did not run.')
             + f'</tr></thead><tbody>{"".join(trs)}</tbody></table></div>')
    return summary + _section_card(
        'Per-bodyId matches',
        'One row per source bodyId in a single table, sorted by type '
        '(case-insensitive) and then by the primary match\'s Jaccard; the '
        'allocation column keeps each bodyId\'s bin assignment on its own '
        'row, so the panel still drives the manual allocation of every '
        'appeared neuron.',
        table,
        ['homolog forward', 'primary match', 'union window', 'allocation',
         'not scored', 'no profile'])


def _homolog_backward_tab(d: Dict) -> str:
    """Homolog · backward (user 2026-09-26): EVERY target bodyId that
    appeared in the run — pool members, expansion-bin members, out-map
    targets, proposal targets — scanned back against the WHOLE source
    dataset (stage 5e), the exact mirror of the forward panel: primary
    source match, the top-3 rank_union ∪ top-3 jaccard union on hover, and
    the morph qualification of the primary pair when the run scored it.
    One display row per bodyId in a single type-sorted table (user
    2026-09-28), plus the queried-type return count: how many target
    bodyIds have a primary source match whose type is one of the query's
    appeared source types, and how many distinct source bodyIds that
    claims back."""
    if not d.get('target_match_available'):
        return _section_card(
            'Homolog · backward',
            'One row per target bodyId the run touched, scanned back '
            'against the whole source dataset.',
            _empty('target_matches.csv absent — this run predates the '
                   'homolog panels; re-run the pipeline to populate the '
                   'panel.'),
            ['homolog backward', 'union window', 'not scored'])
    rows = d['target_match_rows']
    if not rows:
        return _section_card(
            'Homolog · backward', '',
            _empty('target_matches.csv is present but empty — the run had '
                   'no target neurons to scan back.'),
            ['homolog backward'])
    bars = d['branch_bars']
    member_cat_by_bid = {
        bid: str(best.get('member_category') or '')
        for bid, best in (d.get('backward_by_bid') or {}).items()}

    def _alloc_label(r: Dict) -> str:
        cats = [c for c in str(r.get('pool_category') or '').split(';')
                if c]
        mc = member_cat_by_bid.get(str(r.get('target_bodyId') or ''), '')
        if mc and mc not in cats:
            cats.append(mc)
        return '+'.join(cats) if cats else 'unallocated'

    def _row_sort(r):
        j = _as_num(r.get('primary_jaccard'))
        return (-(j if j is not None else -1e9),
                str(r.get('target_bodyId') or ''))

    scanned_counts: Dict[str, int] = collections.Counter(
        str(r.get('scanned_at') or '?') for r in rows)
    n_scored = sum(1 for r in rows
                   if _as_num(r.get('morph_v2_similarity')) is not None
                   or _as_num(r.get('morph_pool_ref')) is not None)

    # Queried-type return (user 2026-09-28, both sides): of the target
    # bodyIds scanned back, how many have a primary source match whose
    # type is one of the query's appeared source types (the type-set in
    # forward_matches.csv), and how many distinct source bodyIds does
    # that claim back.  Read-only arithmetic on the run's own exports.
    q_types = sorted({str(r.get('source_type') or '')
                      for r in (d.get('forward_rows') or [])} - {''})
    ret_body = ''
    if q_types:
        hits = [r for r in rows
                if str(r.get('primary_source_bodyId') or '')
                and str(r.get('primary_source_type') or '') in q_types]
        per_type = collections.Counter(
            str(r.get('primary_source_type')) for r in hits)
        n_src = len({str(r.get('primary_source_bodyId')) for r in hits})
        params = d.get('params') or {}
        src_ds = str(params.get('source_dataset') or 'source-dataset')
        tgt_ds = str(params.get('target_dataset') or 'target-dataset')
        lead = (f'{len(hits):,} of {len(rows):,} {tgt_ds} target bodyIds '
                'have a primary source match whose type is one of the '
                f'{len(q_types)} queried {src_ds} source types — '
                f'{n_src:,} distinct source bodyIds are claimed back.')
        silent = [t for t in q_types if t not in per_type]
        if silent:
            lead += (' No target bodyId lands on: '
                     + _esc(', '.join(silent)) + '.')
        ret_body = ("<p class='mv-note'><b>Queried-type return</b> — "
                    + lead + '</p>'
                    + _scroll_viewport(
                        [f'<tr><td>{_esc(t)}</td><td>{n:,}</td></tr>'
                         for t, n in sorted(per_type.items())],
                        _th('queried source type', 'A type that appeared '
                            'in forward_matches.csv under the run\'s '
                            'query.')
                        + _th('target bodyIds', 'Target neurons whose '
                              'primary source match carries that type.')))

    summary = _section_card(
        f'Homolog · backward — {len(rows):,} target bodyIds',
        'Every target bodyId the run touched, scanned back against the '
        'whole source dataset with the same scorer the reciprocal pass '
        'uses — one row each, sorted by type, the mirror of the forward '
        'panel. The <b>primary source match</b> is the chain-best source; '
        'its hover carries the union of the top-3 rank_union and top-3 '
        'jaccard neighbourhood. Morphology is the run\'s own export for '
        'that pair — displayed, never re-scored here.',
        "<p class='mv-note'>Scanned: "
        f"{scanned_counts.get('run', 0):,} of {len(rows):,} · "
        f"{scanned_counts.get('no_profile', 0):,} without a usable "
        f"profile · {scanned_counts.get('error', 0):,} scan errors · "
        f"{n_scored:,} with a morph value on the primary pair.</p>"
        + _scroll_viewport(
            [f'<tr><td>{_esc(k)}</td><td>{v:,}</td></tr>'
             for k, v in sorted(
                 collections.Counter(_alloc_label(r) for r in rows).items(),
                 key=lambda kv: kv[1], reverse=True)],
            _th('allocation', 'The bins/status the bodyId holds across '
                  "the run's artifacts, spelled per row below.")
            + _th('bodyIds', 'Rows in the table below.'))
        + ret_body,
        ['homolog backward', 'primary match', 'union window', 'allocation',
         'not scored'])

    trs = []
    for r in sorted(rows, key=lambda r: (
            str(r.get('target_type') or 'untyped').lower(),) + _row_sort(r)):
        bid = str(r.get('target_bodyId') or '?')
        label = _alloc_label(r)
        s_bid = str(r.get('primary_source_bodyId') or '')
        if s_bid:
            where = ('in a branch source pool'
                     if _truthy(r.get('primary_in_branch'))
                     else 'outside the branch source pools')
            s_type = str(r.get('primary_source_type') or 'untyped')
            cell = (f'{_esc(s_bid)} · {_esc(s_type)}'
                    f"<span class='mv-note'> · {where} · "
                    f'jac {_f(r.get("primary_jaccard"), 4)} · '
                    f'ru {_f(r.get("primary_rank_union"), 4)}'
                    '</span>')
            cell = _hover(cell, _topn_hover(
                r.get('backward_topN_union'),
                'homolog neighbourhood — top-3 rank_union ∪ top-3 '
                'jaccard (chain order)'))
        else:
            cell = "<span class='missing'>—</span>"
        scanned = str(r.get('scanned_at') or '')
        if scanned == 'run':
            scan_cell = f'{_cnt(r.get("n_scanned"))} sources'
        elif scanned == 'no_profile':
            scan_cell = _term('no profile')
        elif scanned == 'error':
            scan_cell = "<span class='missing'>scan error</span>"
        else:
            scan_cell = _esc(scanned or '—')
        trs.append(
            f'<tr><td>{_esc(bid)}</td>'
            f'<td>{_esc(str(r.get("target_type") or "untyped"))}</td>'
            f'<td>{_esc(label)}</td>'
            f'<td>{_esc(str(r.get("pool_branches") or "—"))}</td>'
            f'<td>{cell}</td>'
            f'<td>{_homolog_morph_cell(r, bars)}</td>'
            f'<td>{scan_cell}</td>'
            f'</tr>')
    table = ("<div style='overflow-x:auto'><table class='mv-table'>"
             '<thead><tr>'
             + _th('target bodyId', 'The target-dataset neuron, whatever '
                   'bin it allocates into.')
             + _th('type', 'Its target-side type.')
             + _th('allocation', 'The pool tier / reverse-scan bin this '
                   'bodyId holds (a neuron claimed by several bins lists '
                   'each).')
             + _th('branches', 'The branches claiming this target.')
             + _th('primary source match (hover: top-3 ∪ top-3)',
                   'The chain-best source of the whole source-dataset scan; '
                   'hover for the union neighbourhood with both ranks and '
                   'both scores.')
             + _th('morph', 'The primary pair\'s morph qualification, '
                   're-derived offline from the run\'s own exports and its '
                   'branch bar — display only.')
             + _th('scanned', 'How many source neurons the scan ranked, or '
                   'why it did not run.')
             + f'</tr></thead><tbody>{"".join(trs)}</tbody></table></div>')
    return summary + _section_card(
        'Per-bodyId matches',
        'One row per target bodyId in a single table, sorted by type '
        "(case-insensitive) and then by the primary source match's "
        'Jaccard — including the matched / verified / borderline pool '
        'members the reciprocal pass deliberately leaves unscanned (their '
        'symmetric forward score is their evidence), so this panel covers '
        'every appeared target without exception.',
        table,
        ['homolog backward', 'primary match', 'union window', 'not scored',
         'no profile'])


def _suspects_tab(d: Dict) -> str:
    """P3 (opt-in): advisory verification of the mapper's rival
    suspects — per rival, the ordinary tier machinery applied to the
    rival's own target pool."""
    rows = d.get('suspects_verification') or []
    if not rows:
        if (d.get('params') or {}).get('verify_suspects'):
            empty = ('suspects_verification.csv has no rows — the pass ran '
                     'and no same-name rival was withheld for the query '
                     'set, so there was nothing to verify.')
        else:
            empty = ('suspects_verification.csv absent — the pass is '
                     'opt-in (--verify-suspects) and was not run.')
        return _section_card(
            'Suspects (same-name rivals)',
            'Connectivity check of the rival candidates the mapper '
            'withheld when the same-name-first rule fired.',
            _empty(empty),
            ['suspects', 'same-name-first'])
    groups: Dict[Tuple[str, str], Dict] = {}
    for r in rows:
        key = (str(r.get('source_type') or '?'),
               str(r.get('target_type') or '?'))
        g = groups.setdefault(key, {
            'verdicts': collections.Counter(), 'rival_of': '',
            'disposition': '', 'clean': False, 'votes': '',
            'pop_src': '', 'pop_tgt': ''})
        g['verdicts'][str(r.get('verdict') or '?')] += 1
        g['rival_of'] = r.get('rival_of') or ''
        g['disposition'] = r.get('disposition') or ''
        if _truthy(r.get('rival_has_own_clean_pair')):
            g['clean'] = True
        if r.get('rival_votes'):
            g['votes'] = str(r.get('rival_votes'))
        if r.get('rival_population_source'):
            g['pop_src'] = str(r.get('rival_population_source'))
        if r.get('rival_population_target'):
            g['pop_tgt'] = str(r.get('rival_population_target'))
    trs = []
    for (src, rival), g in sorted(groups.items()):
        v = g['verdicts']
        dist = ' / '.join(str(v.get(k, 0)) for k in
                          ('verified_strong', 'verified', 'borderline',
                           'unmatched'))
        clean = ('✔ own clean pair' if g['clean']
                 else '✖ no clean own pair')
        pop = (f"{g['pop_src']}→{g['pop_tgt']}"
               if g['pop_src'] or g['pop_tgt'] else '—')
        trs.append(
            f'<tr><td>{_esc(src)}</td>'
            f"<td>{_esc(rival)} <i>(sel: {_esc(g['rival_of'])})</i></td>"
            f"<td>{_esc(g['disposition'])}</td>"
            f'<td>{clean}</td>'
            f"<td>{_esc(g['votes'] or '—')}</td>"
            f'<td>{pop}</td>'
            f'<td>{sum(v.values())}</td>'
            f'<td>{dist}</td></tr>')
    table = (
        "<div style='overflow-x:auto'><table class='mv-table'>"
        '<thead><tr>'
        + _th('source type', 'The source type whose same-name rivals '
              'the mapper withheld.')
        + _th('rival (selection)', 'The rival type and, in italics, '
              'the selection basis that held it back.')
        + _th('disposition', 'How the mapper disposed of the rival '
              '(e.g. gated_held).')
        + _th('mapper evidence', 'Whether the rival itself has a '
              'clean mapped pair of its own.')
        + _th('votes', 'Same-name rival votes recorded by the mapper.')
        + _th('pop src→tgt', "The rival's own population sizes, "
              'source → target.')
        + _th('sources', 'Source neurons behind this rival relation.')
        + _th('verdicts vS / v / b / u', 'Rival-pool validation tier '
              'counts: verified_strong / verified / borderline / '
              'unmatched.')
        + '</tr></thead><tbody>' + ''.join(trs)
        + '</tbody></table></div>')
    n_types = len({k[0] for k in groups})
    return _section_card(
        f'Suspects — {len(groups)} rival relation(s) across '
        f'{n_types} source type(s)',
        'ADVISORY (never a gate): each rival\'s own target pool was '
        'validated with the ordinary tier machinery. A rival that '
        'verifies well is a candidate ANNOTATION, not a mapping — '
        'inclusion goes through the custom label mapper (Decision 8).',
        table,
        ['suspects', 'same-name-first', 'verified', 'borderline',
         'unmatched'])


def _morph_tab(d: Dict) -> str:
    calib = d['calib']
    if not calib:
        return _section_card(
            'Morphology record', '',
            _empty('morphology_calibration.json absent — morphology was '
                   'not run (--no-morphology).'))
    bk = ' · '.join(f'{k} {v}' for k, v in
                    sorted(d['bar_kind_counts'].items()))
    sk = ' · '.join(f'{k} {v}' for k, v in
                    sorted(d['susp_kind_counts'].items()))
    frames = calib.get('score_frame') or {}
    # In a pooling run this record describes the SUPERVISED side of the same
    # run — the branch bars the nested bins used. "AUC gate INACTIVE" must not
    # be read as "morphology was skipped here": pooling refuses a run without
    # morphology outright, and its own record (units scored, budget, refusals)
    # lives on the Pooling tab.
    pooling_note = (
        "<div class='mv-callout'>This is a <b>pooling</b> run: the record "
        'above is the supervised side of the same run. Pooling\'s own '
        'morphology pass is mandatory and its record — units offered, '
        'attempted, scored, refused, and the budget that priced them — is on '
        'the Pooling tab.</div>'
        if str(d['params'].get('validation_mode') or '') == 'pooling' else '')
    summ = _kv_block('Morphology record', [
        (_term('AUC gate'),
         (f"{_f(calib.get('auc'))} — " if calib.get('auc') is not None
          else '')
         + f"{'INACTIVE' if not calib.get('calibrated') else 'ACTIVE'} "
         f"({_esc(calib.get('note', ''))})"),
        (_term('null bar'),
         f"{_f(calib.get('track_a_null_bar'))} = "
         f"{_esc(calib.get('track_a_null_source', 'p95'))}, "
         f"n = {_esc(calib.get('track_a_null_n', '—'))}"),
        (_term('branch bar', 'Branch bars'),
         f"{_esc((calib.get('bar_params') or {}).get('rule', 'floors v3'))}"
         f'<br>candidate bars: {_esc(bk)}<br>suspicious bars: '
         f'{_esc(sk)}'),
        ('Scored',
         f"{_esc(calib.get('n_verified_scored', '—'))} verified-tier + "
         f"{_esc(calib.get('n_suspicious_scored', '—'))} "
         'examinee-bin scorings'),
        (_term('score frames', 'Score frames'),
         f'Track A: {_esc(frames.get("track_a", "target render space"))}'
         f'<br>Track B: {_esc(frames.get("track_b", "target native"))}'
         '<br>Scenes: '
         f'{_esc(frames.get("visualization", "source coordinates"))}'),
    ])
    advisory = ''
    if d.get('track_a_dead'):
        advisory = (
            "<div class='mv-callout mv-warn'>⚠️ <b>Track-A morphology "
            'unavailable this run</b> (0 scored rows) — candidate '
            'admission ran on native Track-B floors only; '
            'morph-dependent counts (candidates, out-map passes, AUC, '
            'null bar) are not comparable to a healthy run. The '
            'connectivity tier and the map-covered claims are '
            'unaffected; out-of-scope rows stay in the CSVs.</div>')
    elif d['null_used'] and calib.get('track_a_null_bar') is not None:
        advisory = (
            "<div class='mv-callout mv-warn'>⚠️ <b>Null-sample "
            "sensitivity:</b> this run's null bar sits on the "
            'skeleton-cache-grown sample; identical earlier runs have '
            'sampled differently (e.g. 0.593/n=72). Null-kind branch '
            'bars are run-sensitive until the deterministic '
            'per-dataset null sample lands — do not compare null-kind '
            'bars ACROSS runs. A native-floor bar is stable across runs '
            'only because one vector-cache read returns one space '
            '(2026-09-25: a cold run graded the neurons it vectorized '
            'itself raw beside standardized pool refs and moved 107 '
            'rows of a real BANC run); '
            '<code>input_fingerprint.morph_stores</code> names the '
            'vector cache this run scored out of (keyed by pass — '
            'supervised / pooling), so compare it before '
            'comparing two runs\' bars.</div>')
    return _section_card(
        'Morphology record',
        'Per-branch bars live in the Branches tab; this section only '
        'aggregates.',
        summ + pooling_note + advisory,
        ['AUC gate', 'null bar', 'branch bar', 'native floor',
         'Track-A backup floor', 'score frames', 'pool_ref tier'])


def _pooling_tab(d: Dict) -> str:
    """`pooling` mode: the UNSUPERVISED pool — every neuron the query names
    scanned against the whole target universe, each source keeping its top-N
    under the admission bar — and its post-hoc comparison with the mapper's
    claims.

    The nested ladder's tabs read a pooling run as empty by construction: no
    branch pool, no branch bar and no mapper claim decides a candidate here,
    so this tab is where a pooling run's result lives.  The comparison cells
    are published with their own honesty rules attached (`reading_notes`) —
    a cell is a set difference, never a recall measure.
    """
    x = d['pooling_xval'] or {}
    pool = d['pooling_pool']
    rows = d['pooling_rows']
    mode = str(d['params'].get('validation_mode') or '')
    if not (x or pool or rows):
        return _section_card(
            'Pooling (unsupervised candidates)', '',
            _empty(
                f'Mode {mode} is a rung of the nested ladder; `pooling` is '
                'PARALLEL to it, so this run performed no unsupervised '
                'scan.' if mode and mode != 'pooling' else
                'No pooling artifact: the seed resolved to 0 neurons or the '
                'pass did not finish. Read the Log tab before taking an '
                'absent file as "nothing found".'))
    cells = x.get('cells') or {}
    seed = x.get('seed') or {}
    gate = x.get('gate') or {}
    morph = x.get('morph') or {}

    error_note = ''
    if x.get('error'):
        error_note = (
            "<div class='mv-callout'><b>The pooling pass raised "
            f"{_esc(x['error'])}</b> — whatever the exports below hold is "
            'what survived that failure, so an empty pool here means a pass '
            'that did not complete, not a universe with no homolog.</div>')

    # The scores and the floors both come from stores and knobs that can move
    # between runs, and the comparison is only comparable across runs that
    # read the same ones — so name them (the §P7a input fingerprint, the same
    # record parameters.json carries).  A part that the run could not record
    # says so: a silently missing git rev would read as "same code" to a
    # reader lining two runs up.
    fp = x.get('input_fingerprint') or {}
    snap = fp.get('mapper_snapshot') if isinstance(fp, dict) else None
    parts = []
    if fp:
        rev = fp.get('git_rev')
        # A dirty worktree means the rev names a DIFFERENT code state than the
        # one that scored, so the run says so beside the rev instead of letting
        # a reader line up two runs that did not share a tree.
        if rev:
            dirty = '-dirty' if fp.get('git_dirty') else ''
            parts.append(f'git {_esc(str(rev)[:8])}{dirty}')
        else:
            parts.append('git rev not recorded')
        if fp.get('scanned_target_universe') is not None:
            parts.append('target universe '
                         f"{_cnt(fp.get('scanned_target_universe'))}")
        if isinstance(snap, dict) and snap.get('bytes'):
            stamp = _store_stamp(snap.get('mtime_s'))
            parts.append(f"mapper snapshot {snap['bytes']} B"
                         + (f' @ {stamp}' if stamp else ''))
        # A native (`morph_pool_ref`) verdict is read out of the target's V2
        # vector cache, which the profile caches above do not cover: name it,
        # or a reader comparing two runs cannot see that the scores came from
        # different stores.  `morph_stores` is keyed by pass (supervised /
        # pooling — both name the same run-baseline store); pre-2026-09-25
        # runs carry the older flat shape.
        _stores = fp.get('morph_stores') or {}
        if 'target' not in _stores and 'source' not in _stores:
            _stores = next(iter(_stores.values()), {}) or {}
        tgt_stores = _stores.get('target') or {}
        vec = tgt_stores.get('vector_cache') or {}
        if isinstance(vec, dict) and vec.get('bytes'):
            parts.append(f"morph vector cache {vec['bytes']} B @ "
                         f"{_store_stamp(vec.get('mtime_s'))}"
                         + (f" · {_cnt(tgt_stores['skeleton_files'])} skeletons"
                            if tgt_stores.get('skeleton_files') else ''))
    comparability = ' · '.join(parts) or (
        'not recorded — this run predates the input fingerprint, so these '
        'cells cannot be lined up with another run\'s')

    bar = x.get('bar') or {}
    # The bar is what admits a row, so it leads the block; the floors follow as
    # what they are now — flags.  An archived pre-bar run has no `bar` record,
    # and printing `? × top-` there would invent an admission rule it never had.
    adm_rows = [
        (_term('admission bar', 'Bar'),
         (f"{_esc(str(bar.get('metric')))} × top-{_cnt(bar.get('top_n'))}"
          + (' — the union of BOTH metrics\' own top-N, not a merged '
             'best-rank ordering' if str(bar.get('metric')) == 'either'
             else ' per source')
          + (f" · row cap {_cnt(bar.get('row_cap_multiple'))} × N, "
             f"{_cnt(bar.get('rows_cut'))} row(s) cut by it"))
         if bar else 'not recorded — this run predates the admission bar, so '
                     'its floors were its admission rule'),
    ]
    gate_rows = adm_rows + [
        (_term('jaccard floor'),
         f"advisory flag only · measured against the configured "
         f"{_f(gate.get('jaccard_floor'), 4)}, nothing removed for missing it"),
        (_term('rank_union', 'rank_union floor'),
         f"advisory flag only · configured {_f(gate.get('rank_union_floor'), 3)}"),
        ('Window',
         f"advisory flag only · ranks ≤ {_f(gate.get('window_mult'), 2)} × the "
         "source type's own queried population"),
        ('Universe', f"{_cnt(x.get('universe_scanned'))} target neurons "
                     f"({len(rows)} candidate pairs kept → {len(pool)} "
                     'distinct candidate targets)'),
        ('Seed', f"{_cnt(seed.get('queried_sources'))} queried sources · "
                 f"{_cnt(seed.get('scanned'))} had a profile · "
                 f"{_cnt(seed.get('sources_with_a_candidate'))} found ≥1 "
                 f"candidate · {_cnt(seed.get('distinct_best_sources'))} "
                 'are some pooled target\'s best source'),
        ('Scored against', comparability),
    ]
    if gate.get('role'):
        # The run's own record states what these numbers are FOR, so the tab
        # shows it rather than leaving a reader to infer that a connectivity
        # floor is a volume knob and not a quality bar.
        gate_rows.insert(len(adm_rows) + 1,
                         ('What they gate', _esc(str(gate['role']))))
    gate_block = _kv_block('Admission — the bar decides, and read this before '
                           'anything was selected', gate_rows)

    cells_block = _kv_block('The post-hoc comparison with the mapping', [
        (_term('mapper_cell', 'confirmed'),
         f"{_cnt(cells.get('confirmed'))} of the pool also sit in a branch's "
         'refined target pool — two engines, named independently, agreeing'),
        (_term('mapper_cell', 'pool_miss'),
         f"{_cnt(cells.get('pool_miss'))} outside every pool: "
         f"type_miss {_cnt(cells.get('type_miss'))} (a neuron of a type the "
         f"map does assert) · type_new {_cnt(cells.get('type_new'))} (a type "
         'outside the map) — the harvest this mode exists to produce'),
        (_term('mapper_cell', 'verified_only'),
         f"{_cnt(cells.get('verified_only'))} target(s) the supervised path "
         'graded matched/verified that this gate did not admit — the two '
         'admit on different quantities, so this is expected and not a '
         'false-positive count'),
    ])

    morph_bits = [f"attempted {_cnt(morph.get('attempted'))}",
                  f"scored {_cnt(morph.get('scored'))}",
                  f"qualified {_cnt(morph.get('qualified'))}"]
    if 'no_score' in morph:
        # the distinction the record exists to make: attempted is what the
        # budget allowed to be looked at, scored is what came back with a
        # value. Collapsing them printed `scored 8` for a run whose rows
        # held 1 `scored` and 7 `no-score`.
        morph_bits.append(f"no-score {_cnt(morph.get('no_score'))}")
    if 'units' in morph:
        # `attempted`/`capped` are a SPLIT of `units`, so the line has to show
        # the whole or the two halves read as a sum that does not add up.
        morph_bits.append(f"of {_cnt(morph.get('units'))} units")
    if morph.get('budget'):
        morph_bits.append(f"budget {morph['budget']}")
    vc = morph.get('vector_cache') or {}
    if vc:
        # The store's own ledger, beside the counts it changed: `reused` is
        # neuron preparation this run did NOT pay for, `computed` what it paid
        # for, `store rows` what the next run starts from.  Those three are
        # different numbers and the old line conflated the first with the last
        # by calling the file size "saved".  `stale` is geometry the store
        # refused to reuse because the skeleton behind it moved.
        _load = _as_num(vc.get('loaded')) or 0
        _tgt = _as_num(vc.get('targets'))
        morph_bits.append(
            f"target vectors reused {_cnt(_load)}"
            + (f" · computed {_cnt(max(_tgt - _load, 0))}"
               if _tgt is not None else '')
            + f" · store rows {_cnt(vc.get('saved'))}"
            + (f" · {_cnt(vc.get('stale_dropped'))} stale"
               if _as_num(vc.get('stale_dropped')) else ''))
    if _as_num(morph.get('capped')):
        morph_bits.append(f"budget-capped {_cnt(morph.get('capped'))}"
                          ' (no look taken, not a rejection)')
    if morph.get('error'):
        morph_bits.append(f"error {_esc(morph['error'])}")
    # Morphology is a GATE here, not an advisory column: a scored candidate
    # below its bar leaves the pool and the scene root.  The count of those
    # refusals is the only trace they keep, so it is named rather than left as
    # an absent row.
    if morph.get('gate_applied') is not None:
        if _truthy(morph.get('gate_applied')):
            gate_word = 'applied'
        elif (_as_num(morph.get('attempted')) or 0) > 0:
            # the pass RAN and graded nothing: not disabled, unapplied
            gate_word = 'unapplied (0 measurements)'
        else:
            gate_word = 'off'
        morph_bits.append(
            'gate ' + gate_word
            + (f", {_cnt(morph.get('dropped_targets'))} target(s) refused "
               'for scoring below the bar'
               if _as_num(morph.get('dropped_targets')) else ''))

    notes = x.get('reading_notes') or []
    notes_block = ('' if not notes else
                   "<div class='mv-callout'><b>How to read these cells</b>"
                   '<ul>'
                   + ''.join(f'<li>{_esc(n)}</li>' for n in notes)
                   + '</ul></div>')

    pool_warn = _pooling_warning_line(d)
    warn_block = ('' if not pool_warn else
                  "<div class='mv-callout mv-warn'>⚠️ "
                  f"{_esc(pool_warn)}</div>")

    # The mode's own axis. `pooling_pool.csv` answers "which targets were
    # found"; this answers the question the run was actually asked — "for each
    # queried neuron, what did it find, and did anything survive". A source
    # that found nothing is a ROW here, never an absence, which is the whole
    # reason the file exists (plan §5).
    srcs = d.get('pooling_sources') or []
    src_block = ''
    if srcs:
        def _n(r, k):
            return int(_as_num(r.get(k)) or 0)

        admitted = [r for r in srcs if _n(r, 'n_admitted') > 0]
        kept = [r for r in srcs if _n(r, 'n_in_pool') > 0]
        lost = [r for r in admitted if _n(r, 'n_in_pool') == 0]
        none = [r for r in srcs
                if str(r.get('no_finding') or '').strip()]
        claimed = [r for r in srcs
                   if str(r.get('source_claimed') or '').strip().lower()
                   in TRUE_STR]
        tiers = {t: sum(1 for r in srcs
                        if str(r.get('tier') or '') == t)
                 for t in POOLING_TIERS}
        src_rows = [
            (_term('pooling', 'Queried sources'),
             f"{_cnt(len(srcs))} · across {_cnt(len({str(r.get('source_type')) for r in srcs}))} "
             'source types'),
            ('Reached ≥1 target',
             f"{_cnt(len(admitted))} of {_cnt(len(srcs))} · "
             f"{_cnt(sum(_n(r, 'n_admitted') for r in srcs))} admitted rows "
             'total'),
            (_term('pooling tier', 'Tier of each source\'s best finding'),
             ' · '.join(f"{t} {_cnt(tiers.get(t, 0))}" for t in POOLING_TIERS)
             + (f" · {_cnt(len(none))} found nothing" if none else '')),
            ('Kept ≥1 after morphology',
             f"{_cnt(len(kept))} of {_cnt(len(srcs))}"
             + (f" · {_cnt(len(lost))} saw every finding refused by the bar"
                if lost else '')),
            ('Claimed by the mapper',
             f"{_cnt(len(claimed))} — post-hoc advice only; the mapper named "
             'none of these rows'),
            ('Pool per source',
             f"{_f(x.get('pool_per_source'), 3)} targets per queried source"
             + (f" · {_esc(str(x.get('pool_size_warning')))}"
                if x.get('pool_size_warning') else
                ' · inside the [0.5, 2.0] band this bar aims for')),
        ]
        src_block = _kv_block('Per source — the mode\'s own axis', src_rows)

    head = (_section_card(
        'Pooling — the unsupervised scan',
        'Every neuron the query names, scanned against the whole target '
        'universe, each source then keeping its top-N under the admission bar '
        'with morphology last; the mapper\'s own answer is joined only '
        'afterwards, so it cannot decide what this list holds. Proposals for '
        'review — the mapping is never rewritten.',
        error_note + gate_block + cells_block
        + _kv_block('Morphology — the last gate', [
            (_term('morph_gate', 'morphology'), ' · '.join(morph_bits)),
        ]) + src_block + warn_block + notes_block,
        ['pooling', 'mapper_cell', 'morph_gate',
         'jaccard floor', 'admission bar', 'pooling tier', 'ordering chain']))

    miss = x.get('pool_miss_by_type') or {}
    miss_tot = sum(int(v or 0) for v in miss.values())
    harvest = ''
    if miss:
        harvest = ("<div class='mv-kv-title'>The harvest, by target "
                   'type</div>' + _scroll_viewport(
            [f"<tr><td>{_esc(t or 'untyped')}</td><td>{_cnt(n)}</td>"
             f"<td>{_pct(n, miss_tot)}</td></tr>"
             for t, n in sorted(miss.items(), key=lambda kv: -(int(
                 kv[1] or 0)))],
            _th('target type', 'A type the pooling pool holds candidates '
                'for that no branch pool claims.')
            + _th('pool_miss neurons', 'How many of that type the gate '
                  'admitted outside every mapper pool.')
            + _th('share', 'Of all pool_miss neurons.')))

    trs = []
    for r in pool:
        mg = str(r.get('morph_gate') or '')
        if mg == 'scored':
            # Name the pair the verdict was actually made from.  A `native`
            # row is graded by its pool reference against the native floor;
            # showing its Track-A number beside that floor made a passing row
            # read as a broken gate (see apply_morph_gate).  A missing verdict
            # is never drawn as a refusal — the pool kept such a row on
            # purpose.
            kind = str(r.get('morph_bar_kind') or 'null_bar')
            deciding = _as_num(r.get('morph_pool_ref')
                               if kind == 'native'
                               else r.get('morph_similarity'))
            q = str(r.get('morph_qualified') or '').strip().lower()
            if deciding is None:
                mcell = (f"<span class='missing'>{_esc(kind)}: no "
                         'evidence for this pair</span>')
            else:
                mark = ('✓' if q in TRUE_STR else
                        '✗' if q in ('false', '0', 'no') else 'no verdict')
                mcell = (f"<span class='mv-note'>{_esc(kind)}</span> "
                         f"{deciding:.3f} vs {_f(r.get('morph_bar'), 3)} "
                         f'{mark}')
        else:
            mcell = (f"<span class='missing'>{_esc(mg or '—')}</span>")
        n_src = _cnt(r.get('n_sources'))
        dup = _as_num(r.get('dup'))
        # A row can be published and NOT pooled: `in_pool=False` is the target
        # the bar refused on every admitting row. It left the scene, not the
        # file, and the table has to say which of the two the reader is looking
        # at — otherwise "N candidate targets" counts rows that are not in the
        # pool, and the ✗ beside them reads as a bug in the gate.
        kept = str(r.get('in_pool') or '').strip().lower() in TRUE_STR
        tiers = str(r.get('tiers') or '').strip()
        refused_n = _as_num(r.get('n_rows_refused'))
        trs.append(
            f"<tr><td>{_esc(r.get('target_bodyId'))} "
            f"<span class='mv-note'>{_esc(r.get('leaf') or '(untyped)')}"
            + (f" · {_esc(tiers)}" if tiers else '')
            + ('' if kept else
               f" <span class='missing'>refused on all "
               f'{_cnt(refused_n)} row(s)</span>')
            + '</span></td>'
            f"<td>{_esc(r.get('best_source_type') or '—')} "
            f"{_esc(r.get('best_source_bodyId'))}</td>"
            f"<td>{_f(r.get('jaccard'), 4)} "
            f"<span class='mv-note'>#{_esc(r.get('jaccard_rank'))} · RU "
            f"{_f(r.get('rank_union'), 3)} · w "
            f"{_esc(r.get('window_size'))}</span></td>"
            f"<td>{n_src}{' (+' + str(int(dup)) + ' other)' if dup else ''}"
            '</td>'
            f'<td>{mcell}</td>'
            f"<td>{_esc(r.get('mapper_cell') or '—')}"
            + (f" <span class='mv-note'>{_esc(r['mapper_verdict'])}</span>"
               if str(r.get('mapper_verdict') or '').strip() else '')
            + '</td></tr>')
    table = _scroll_viewport(
        trs,
        _th('candidate target', 'The pooled target neuron, its shared '
            'leaf token (`(out-map)` / `T>src` / `T(no_source)` / '
            '`untyped`) and the tier set its admitting rows carry. A row '
            'marked `refused on all N row(s)` is published but NOT pooled: '
            'every source that reached this target was refused by the '
            'morphology bar, so it left the scene while its row stayed here '
            'for the count.')
        + _th('best source', 'The chain-best source of the many that reach '
              'this target — jaccard first, rank_union as the tie-break, '
              'bodyId last.')
        + _th('jaccard · rank · RU · window', 'The absolute gate this '
              'candidate passed, plus the window that bounded its ranks.')
        + _th('sources', 'How many queried sources reach this target (the '
              'extra ones are listed in pooling_candidates.csv).')
        + _th('morphology', 'The LAST gate, printed as the pair it actually '
              'graded: `native` compares the candidate\'s similarity to the '
              'source\'s own reference pool (`morph_pool_ref`) against the '
              'native floor, `track_a` and `null_bar` compare the Track-A '
              'pair score against their bar — ✓ at or above. With the gate '
              'on, a scored ✗ row has already left this pool, so the only '
              'absences here (`no-score`, `not-attempted-cap`, "no evidence '
              'for this pair") name a look not taken, never a rejection.')
        + _th('mapper_cell', 'The post-hoc comparison with the mapping, '
              'with the supervised verdict where one exists.')
    ) if trs else _empty(
        'The bar admitted no candidate: with this bar the queried population '
        'has no connectivity finding in the target universe. Raising '
        'pooling_bar_top_n or widening pooling_bar_metric is the knob, not a '
        'different verdict — the floors are flags and were never the knob.')

    n_kept = sum(1 for r in pool
                 if str(r.get('in_pool') or '').strip().lower() in TRUE_STR)
    return head + _section_card(
        f'Pooling pool — {len(pool)} candidate target'
        f'{"s" if len(pool) != 1 else ""} published, {n_kept} kept',
        'One row per target neuron, on the ordering chain; the count that '
        'matters is the second one — a target the morphology bar refused on '
        'every row stays in this table with `in_pool=False` so the refusal is '
        'auditable, and leaves the scene. The full '
        'per-pair rows are in `pooling/pooling_candidates.csv`; the '
        'comparison cells above are set differences, so no row here is a '
        'recall measure.',
        harvest + table,
        ['ordering chain', 'mapper_cell', 'morph_gate'])


def _scene_failures(d: Dict) -> list:
    """Parents whose scene was ATTEMPTED and died, as `[(type, error)]`.

    Two sources, unioned: the run log's `! scene X failed: …` line (which works
    for runs archived before the marker existed), and the folder-level
    `SCENE_FAILED.txt` the renderer now writes. Without this the Scenes tab
    attributed a crash to policy — a male-cns family run showed "16 scenes
    rendered — only types with renderable expansion content get a scene" while
    five more parents had been attempted and lost to a ZeroDivisionError.
    """
    failed: Dict[str, str] = {}
    for ln in (d.get('readme') or {}).get('bang_lines') or []:
        m = re.match(r'!\s*scene (.+?) failed:\s*(.*)', ln)
        if m:
            failed.setdefault(m.group(1).strip(), m.group(2).strip()
                              or 'error not recorded')
    viz = Path(d['run_dir']) / 'visualization' if d.get('run_dir') else None
    if viz and viz.is_dir():
        for marker in sorted(viz.glob('plot-3d_*/SCENE_FAILED.txt')):
            text = marker.read_text(errors='replace')
            m = re.search(r'parent type:\s*(.+)', text)
            e = re.search(r'error:\s*(.+)', text)
            if m:
                failed.setdefault(m.group(1).strip(),
                                  (e.group(1).strip() if e else
                                   'see SCENE_FAILED.txt'))
    return sorted(failed.items())


def _scene_palette_html(d: Dict) -> str:
    """The category colors this run's scenes actually wore, as legend chips.

    A scene's tree legend is the only place the palette was ever visible, so a
    run whose categories were recolored could not be read from its report
    alone. `parameters.json` records the EFFECTIVE map (see
    `scene_styling_record`), which is what this renders — the same
    `COLOR_EDITABLE_CATEGORIES` order the UI editor offers, so the two never
    disagree about which bins are paintable. A changed bin names the default it
    replaced, because "why is matched red?" is the question this answers.
    """
    palette = (d.get('params') or {}).get('scene_category_colors')
    if not isinstance(palette, dict) or not palette:
        return ''
    try:
        from comparison.mapping_validation_visualize import (
            CATEGORY_COLORS, COLOR_EDITABLE_CATEGORIES)
    except Exception:  # noqa: BLE001 - a legend is never worth a failed report
        return ''
    chips = []
    changed = 0
    for cat in COLOR_EDITABLE_CATEGORIES:
        color = str(palette.get(cat) or '')
        if not color:
            continue
        default = CATEGORY_COLORS.get(cat, '')
        moved = color.lower() != str(default).lower()
        # counted from the bins that actually produced a chip, so the headline
        # can never claim a recolor for a category the block does not show
        changed += int(moved)
        note = (f"<span class='mv-palette-was'>was {_esc(default)}</span>"
                if moved else '')
        chips.append(
            "<span class='mv-palette-chip'>"
            f"<span class='mv-palette-swatch' "
            f"style='background:{_esc(color)}'></span>"
            f"<span class='mv-palette-name'>{_esc(cat)}</span>"
            f"<span class='mv-palette-hex'>{_esc(color)}</span>{note}</span>")
    if not chips:
        return ''
    if changed:
        subtitle = (f"<b>{changed}</b> "
                    f"{'category' if changed == 1 else 'categories'} "
                    'recolored from the pipeline defaults')
    else:
        subtitle = 'the pipeline defaults, nothing recolored'
    return (
        "<div class='mv-palette'>"
        "<p class='mv-palette-title'>Scene palette this run wore — "
        + subtitle + "</p><div class='mv-palette-chips'>"
        + ''.join(chips) + "</div>"
        "<p class='mv-note'>Color is a property of the CATEGORY, so one bin "
        'wears one color across every branch and parent scene in the run — '
        'that is what makes two scenes comparable. Adjust per category under '
        'Cross-Dataset › Type Validation › Advanced Visualization.</p></div>')


def _scenes_tab(d: Dict) -> str:
    sc = d['selfcheck']
    statuses: Dict[str, bool] = {}
    for ln in d['readme']['selfcheck_pass']:
        m = re.match(r'self-check \[(.+?)\]:', ln)
        if m:
            statuses[m.group(1)] = True
    for ln in d['readme']['selfcheck_fail']:
        m = re.match(r'!\s*self-check \[(.+?)\]:', ln)
        if m:
            statuses[m.group(1)] = False
    if not d['scenes']:
        return _section_card(
            'Scenes', '',
            _empty('No scenes rendered (visualization skipped or no '
                   'renderable expansion content).'))
    tiles = []
    for s in d['scenes']:
        t = s['type']
        status = ('✓' if statuses.get(t, True) else '⚠ FAILED')
        img = (f"<img src='{_esc(s['png'])}' alt='{_esc(t)} scene'>"
               if s['png'] else "<div class='scene-tile-ph'>no preview"
                                '</div>')
        inner = (img
                 + f"<span class='scene-tile-name'>{_esc(t)}</span>"
                 + f"<span class='scene-tile-status'>{status}</span>")
        if s['html']:
            tiles.append(f"<a class='scene-tile' "
                         f"href='{_esc(s['html'])}'>{inner}</a>")
        else:
            tiles.append(f"<div class='scene-tile'>{inner}</div>")
    failed = _scene_failures(d)
    fail_note = ''
    if failed:
        fail_note = (
            f" <b class='mv-warn'>{len(failed)} attempted and FAILED</b>"
            ' (their folders hold no page; see the Log tab): '
            + _esc(', '.join(f'{t} — {e}' for t, e in failed)))
    body = (
        f"<p class='section-summary'>{len(d['scenes'])} scenes rendered "
        '— types with nothing renderable get no scene (decided: a scene '
        'with no content is noise), and every parent that was attempted '
        'but failed is named below rather than folded into that count.'
        f'{fail_note}</p>'
        f"<div class='scene-grid'>{''.join(tiles)}</div>"
        + _scene_palette_html(d) +
        "<p class='mv-note'>Scenes render in SOURCE coordinates — read "
        'as anatomy, never as the scoring frame (morph tracks score in '
        'TARGET coordinates).</p>')
    return _section_card(
        'Scenes',
        f'Self-check {sc["pass"]}/{sc["pass"] + sc["fail"]}: every '
        "legend leaf's rendered geometry matches its neuron and every "
        'expected neuron is plotted (population + geometry census) — '
        'the CSV and the picture cannot disagree.',
        body, ['scene self-check', 'score frames'])


def _log_tab(d: Dict) -> str:
    # §10 warnings & anomalies — every `!` line verbatim
    bang = d['readme']['bang_lines']
    if bang:
        warn_body = (
            "<p class='section-summary'>Every ``!`` log line is "
            'reproduced VERBATIM — the report never summarizes away a '
            "failure. Warnings are also appended to the folder's "
            'user_warning_notes.txt.</p>'
            "<pre class='mv-log'>"
            + '\n'.join(_esc(ln.strip()) for ln in bang) + '</pre>')
    else:
        sc = d['selfcheck']
        if sc['pass'] + sc['fail']:
            checks = f"Self-checks {sc['pass']}/{sc['pass'] + sc['fail']}."
        else:
            checks = 'No scenes rendered.'
        warn_body = (
            "<p class='mv-note'>This run wrote no <code>!</code> failure "
            f'line. {checks}</p>')
    if d['advisories']:
        warn_body += ("<p class='mv-note'>Advisories carried: "
                      + _esc(', '.join(d['advisories'])) + '.</p>')
    # The advisory summaries written to user_warning_notes.txt are derived
    # from the exports rather than the log, so without this they would live
    # only in a side file a reader never opens.
    recip = _reciprocal_warning_line(d)
    derived = [ln for ln in (recip, _pooling_warning_line(d)) if ln]
    if derived:
        warn_body += ("<p class='mv-note'>Derived advisories (also in "
                      'user_warning_notes.txt):</p>'
                      "<pre class='mv-log'>" + _esc('\n'.join(derived))
                      + '</pre>')
    sec10 = _section_card('Warnings & anomalies', '', warn_body)

    # §11 provenance
    params = d['params']
    order = ['validation_mode', 'top_k', 'top_m', 'min_synapse_threshold',
             'rank_top_k', 'gap_min', 'suspicious_jaccard_factor',
             'morph_enabled', 'morph_auc_floor', 'morph_track_a_offset',
             'morph_suspicious_level', 'suspicious_per_source_cap',
             'scene_selfcheck', 'aggressive_expansion', 'pool_widen',
             'backward_evidence_enabled', 'backward_top_n',
             'backward_max_neurons', 'backward_per_branch_cap',
             'backward_scan_pool_targets', 'skip_backward_pass',
             'pooling_jaccard_floor', 'pooling_rank_union_floor',
             'pooling_window_mult', 'pooling_bar_metric',
             'pooling_bar_top_n',
             'pooling_max_morph_targets']
    pl = ' · '.join(f'{_esc(k)}={_esc(params[k])}'
                    for k in order if k in params)
    timeline: List[str] = []
    stage_open: Dict[str, Tuple[str, str]] = {}
    for p in d['progress']:
        ev = p.get('event')
        if ev == 'stage_start':
            stage_open[str(p.get('stage'))] = (str(p.get('ts', '')),
                                               str(p.get('label') or ''))
        elif ev == 'stage_done':
            st = str(p.get('stage'))
            t0, label = stage_open.pop(st, ('', ''))
            dur = ''
            if t0:
                try:
                    dt = (datetime.fromisoformat(str(p.get('ts')))
                          - datetime.fromisoformat(t0)).total_seconds()
                    dur = f' ({dt:.0f} s)'
                except (ValueError, TypeError):
                    pass
            timeline.append(f'stage {st} {label}'.strip() + dur)
        elif ev == 'scan_progress' and 'done' in p:
            timeline.append(
                f"scan {p.get('done')}/{p.get('total', '?')} "
                f"{p.get('source_type', '')} "
                f"({p.get('scanned', 0)}/{p.get('sources', 0)} sources, "
                f"{p.get('elapsed_s', 0):.0f} s)")
        elif ev == 'profiles_progress' and 'universe' in p:
            timeline.append(
                f"profiles pre-flight universe {p.get('universe', 0):,} "
                f"cached {p.get('cached', 0):,} built "
                f"{p.get('built', 0):,}")
        elif ev == 'out_map_progress':
            if 'done' in p:
                timeline.append(f"out-map expansion {p.get('done')}/"
                                f"{p.get('total', '?')}")
            elif p.get('note'):
                timeline.append(f"out-map {p['note']}")
        elif ev == 'backward_progress':
            if 'done' in p:
                timeline.append(f"reciprocal scan {p.get('done')}/"
                                f"{p.get('total', '?')}"
                                + (f" ({p['note']})"
                                   if p.get('note') else ''))
            elif p.get('note'):
                timeline.append(f"reciprocal {p['note']}")
    elapsed = next((p.get('elapsed_s') for p in d['progress']
                    if p.get('event') == 'run_done'), None)
    if isinstance(elapsed, (int, float)):
        timeline.append(f'total {elapsed:,.1f} s')
    prov = _kv_block('Provenance & reproducibility', [
        ('Parameters (full)', pl or '—'),
        ('Timeline', _esc(' · '.join(timeline) or '—')),
        ('Datasets', f"{_esc(params.get('source_dataset', '—'))} → "
                     f"{_esc(params.get('target_dataset', '—'))}"),
        ('Run folder', f"<code>{_esc(d['run_dir'])}</code>"),
        ('Matrix cross-link', 'none (decided: the run report is '
                              'self-contained)'),
    ])
    sec11 = _section_card(
        'Provenance & reproducibility',
        'The run report is self-contained (decided D5): no matrix '
        'cross-link.', prov)

    # §12 file index + glossary
    idx_rows = []
    for name, blurb in ARTIFACT_LINES:
        # Print the path THIS run actually wrote: a pre-layout folder has no
        # subfolders, and a phantom `expansion/…` line sends the reader
        # hunting for a file that is one level up.
        shown = _run_file_rel(d['run_dir'], Path(name).name)
        n = _n_rows(_run_file(d['run_dir'], Path(name).name))
        idx_rows.append(
            f'<tr><td>{_esc(shown)}</td><td>'
            + ('—' if n is None else f'<span class="missing">{n:,}</span>')
            + f'</td><td>{_esc(blurb)}</td></tr>')
    idx_rows.append(
        f"<tr><td>visualization/*.html</td><td>{len(d['scenes'])}</td>"
        '<td>scenes (Scenes tab)</td></tr>')
    idx_rows.append(
        '<tr><td>README.txt</td><td>—</td><td>slim directions + raw run '
        'log (the analysis lives here in report.html)</td></tr>')
    idx = ("<div style='overflow-x:auto'><table class='mv-table'>"
           '<thead><tr>'
           + _th('artifact', 'The run-folder file, relative to the '
                 'run root.')
           + _th('rows', 'CSV data rows; — marks a file absent from '
                 'this run.')
           + _th('one-liner', 'What the artifact is for.')
           + '</tr></thead><tbody>' + ''.join(idx_rows)
           + '</tbody></table></div>')
    gloss = []
    for fname, terms in FILE_GLOSSARY.items():
        rows_html = ''.join(
            f'<tr><td>{_esc(t)}</td><td>{_esc(TERM_DEFS[t])}</td></tr>'
            for t in terms if t in TERM_DEFS)
        if not rows_html:
            continue
        gloss.append(
            '<details class="detail-block"><summary>'
            f"{_esc(fname)} — columns</summary>"
            "<div style='overflow-x:auto'>"
            "<table class='mv-table mv-defs'><tbody>"
            f'{rows_html}</tbody></table></div></details>')
    sec12 = _section_card(
        'File index + glossary',
        'Row counts computed at write time; the column glossary lives '
        'here and on hover, out of the report front.',
        idx + ''.join(gloss))
    return sec10 + sec11 + sec12


# extra stylesheet on top of report_kit.report_css (kept local so the
# shared kit stays untouched)
_EXTRA_CSS = """<style>
.term { position: relative; border-bottom: 1px dotted #7c8db5;
        cursor: help; }
.term .tip { display: none; position: absolute; z-index: 40; left: 0;
             top: calc(100% + 8px); width: 360px; padding: 10px 12px;
             background: #152238; color: #f4f7fb; font-size: 12px;
             line-height: 1.45; border-radius: 9px;
             box-shadow: 0 10px 26px rgba(21, 34, 56, .35);
             font-weight: 400; }
.term .tip b { display: block; margin-bottom: 3px; }
.term:hover .tip { display: block; }
.term .tip.mv-tip-wide { width: 460px; }
.term .tip.mv-tip-wide table { border-collapse: collapse; width: 100%;
       font-size: 11.5px; margin-top: 4px; }
.term .tip.mv-tip-wide th, .term .tip.mv-tip-wide td {
       border: 0; border-bottom: 1px solid rgba(244, 247, 251, .18);
       padding: 3px 5px; text-align: left; }
.term .tip.mv-tip-wide th { color: #a9bcd9; font-weight: 700;
       text-transform: uppercase; letter-spacing: .03em; }
.bev { display: inline-block; padding: 1px 7px; border-radius: 999px;
       font-size: 11px; font-weight: 800; white-space: nowrap; }
.bev-high { background: #e2f4e6; color: #1f6b32; }
.bev-medium { background: #fdf0d8; color: #8a5a06; }
.bev-low { background: #eceff5; color: #55617a; }
.bev-off { background: #f6f7fa; color: #97a2b5;
           border: 1px dashed #c4ccdb; }
.bev-thin { background: #fdeef1; color: #96324a;
            border: 1px dashed #d9a3b3; }
.mv-table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
.mv-table th, .mv-table td { border: 1px solid var(--line);
    padding: 6px 9px; text-align: left; vertical-align: top; }
.mv-table thead th { background: var(--surface-soft); font-size: 11.5px;
    text-transform: uppercase; letter-spacing: .04em;
    color: var(--muted); }
.mv-table.mv-defs td:first-child { white-space: nowrap;
    font-weight: 700; }
/* the reciprocal per-neuron groups share one colgroup; fixed layout is what
   makes the columns line up across cards, and `anywhere` keeps a 19-digit
   bodyId from forcing its column wider than the plan allows */
.mv-bev-table { table-layout: fixed; }
.mv-bev-table td { overflow-wrap: anywhere; }
.mv-kv { max-width: 980px; }
.mv-kv .kv-k { width: 260px; font-weight: 700; white-space: nowrap; }
.mv-kv-title { font-weight: 800; font-size: 13px; margin-bottom: 6px;
               text-transform: uppercase; letter-spacing: .05em;
               color: var(--accent-dark); }
.mv-pre-wrap { margin-bottom: 14px; }
.mv-note { color: var(--muted); font-size: 12.5px; margin: 10px 0 0; }
.mv-callout { background: var(--accent-soft);
              border: 1px solid #c9d9f5; border-radius: 10px;
              padding: 11px 14px; font-size: 13px; margin: 12px 0; }
.mv-callout.mv-warn { background: #fdf6e3; border-color: #ecd9a0; }
.mv-log { background: var(--surface-soft);
          border: 1px solid var(--line); border-radius: 9px;
          padding: 11px 13px; font-size: 12px; overflow-x: auto;
          white-space: pre-wrap; }
.warn-chip { color: #92600a; font-weight: 800; }
.scene-grid { display: grid;
              grid-template-columns: repeat(auto-fill,
                                            minmax(210px, 1fr));
              gap: 13px; margin: 14px 0; }
.scene-tile { display: block; background: var(--surface-soft);
              border: 1px solid var(--line); border-radius: 11px;
              overflow: hidden; text-align: center; }
.scene-tile img { width: 100%; height: 128px; object-fit: cover;
                  display: block; }
.scene-tile-ph { height: 128px; display: flex; align-items: center;
                 justify-content: center; color: var(--muted);
                 font-size: 12px; }
.scene-tile-name { display: block; font-weight: 750; font-size: 13px;
                   padding: 7px 8px 0; }
.scene-tile-status { display: block; color: var(--success);
                     font-size: 11px; padding: 0 8px 8px; }
.mv-palette { border-top: 1px solid var(--line); margin: 16px 0 0;
              padding-top: 12px; }
.mv-palette-title { font-size: 13px; font-weight: 700; margin: 0 0 8px;
                    color: var(--ink); }
.mv-palette-chips { display: flex; flex-wrap: wrap; gap: 7px; }
.mv-palette-chip { display: inline-flex; align-items: center; gap: 6px;
                   background: var(--surface-soft);
                   border: 1px solid var(--line); border-radius: 999px;
                   padding: 3px 10px 3px 5px; font-size: 12px; }
.mv-palette-swatch { width: 14px; height: 14px; border-radius: 4px;
                     border: 1px solid rgba(11, 31, 58, .22);
                     flex: none; display: inline-block; }
.mv-palette-name { font-weight: 650; }
.mv-palette-hex { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                  color: var(--muted); font-size: 11px; }
.mv-palette-was { color: #92600a; font-size: 11px; }
.missing { color: var(--muted); }
.report-hero .report-subtitle { font-size: 16px; color: var(--ink); }
.report-hero .report-subtitle b { color: var(--accent-dark); }
</style>"""


# ---------------------------------------------------------------------------
# assembly + entry points
# ---------------------------------------------------------------------------

_TABS: List[Tuple[str, str]] = [
    ('coverage', 'Coverage'), ('branches', 'Branches'),
    ('targets', 'Targets'), ('fill', 'Fill'),
    ('reciprocal', 'Reciprocal'), ('outmap', 'Out-map'),
    ('backward', 'Backward'),
    ('homolog_forward', 'Homolog · forward'),
    ('homolog_backward', 'Homolog · backward'),
    ('suspects', 'Suspects'),
    ('morph', 'Morph'), ('pooling', 'Pooling'),
    ('scenes', 'Scenes'), ('log', 'Log'),
]


# A single fixed-position layer that mirrors the hovered term's tip.
# The inline CSS tooltip is clipped by every `overflow-x:auto` table
# wrapper and painted under later section cards (r13 review); a fixed
# layer at z-index 9999 escapes both, and works for <th> terms too.
# Without JS the inline CSS tooltip remains the fallback.
_HOVER_LAYER_CSS = """<style>
#mv-hover-layer { position: fixed; display: none; z-index: 9999;
  pointer-events: none; max-width: 480px; padding: 10px 12px;
  background: #152238; color: #f4f7fb; font-size: 12px;
  line-height: 1.45; border-radius: 9px;
  box-shadow: 0 10px 26px rgba(21, 34, 56, .45); font-weight: 400; }
#mv-hover-layer b { display: block; margin-bottom: 3px; }
#mv-hover-layer table { border-collapse: collapse; width: 100%;
  font-size: 11.5px; margin-top: 4px;
  /* fixed layout + wrapping: the 7-column top-N table carries 19-digit
     bodyIds, and an auto layout let that column push WHERE past the box
     (r16 review screenshot) */
  table-layout: fixed; }
#mv-hover-layer th:nth-child(1), #mv-hover-layer td:nth-child(1) { width: 8%; }
#mv-hover-layer th:nth-child(2), #mv-hover-layer td:nth-child(2) { width: 8%; }
#mv-hover-layer th:nth-child(3), #mv-hover-layer td:nth-child(3) { width: 30%; }
#mv-hover-layer th:nth-child(4), #mv-hover-layer td:nth-child(4) { width: 17%; }
#mv-hover-layer th:nth-child(5), #mv-hover-layer td:nth-child(5) { width: 12%; }
#mv-hover-layer th:nth-child(6), #mv-hover-layer td:nth-child(6) { width: 12%; }
#mv-hover-layer th:nth-child(7), #mv-hover-layer td:nth-child(7) { width: 13%; }
#mv-hover-layer td { overflow-wrap: anywhere; }
#mv-hover-layer th, #mv-hover-layer td { border: 0;
  border-bottom: 1px solid rgba(244, 247, 251, .18); padding: 3px 5px;
  text-align: left; }
#mv-hover-layer th { color: #a9bcd9; font-weight: 700;
  text-transform: uppercase; letter-spacing: .03em; }
html.js .term .tip { display: none !important; }
</style>"""

_HOVER_LAYER_JS = """<script>
(function () {
  document.documentElement.className += ' js';
  var layer = document.createElement('div');
  layer.id = 'mv-hover-layer';
  document.body.appendChild(layer);
  var cur = null;
  function show(term) {
    var tip = term.querySelector('.tip');
    if (!tip) { hide(); return; }
    layer.innerHTML = tip.innerHTML;
    layer.style.display = 'block';
    // a table payload (the top-N neighbourhood) needs more than the prose
    // width — 7 columns of bodyIds and scores at 480px clipped the last one
    var wide = layer.querySelector('table') ? 620 : 480;
    layer.style.width = Math.min(wide, window.innerWidth - 16) + 'px';
    var r = term.getBoundingClientRect();
    var top = r.bottom + 8;
    if (top + layer.offsetHeight > window.innerHeight - 8)
      top = Math.max(8, r.top - layer.offsetHeight - 8);
    layer.style.top = top + 'px';
    layer.style.left = Math.min(Math.max(8, r.left),
                                window.innerWidth - layer.offsetWidth
                                - 8) + 'px';
  }
  function hide() { layer.style.display = 'none'; cur = null; }
  document.addEventListener('mouseover', function (e) {
    var t = e.target.closest ? e.target.closest('.term') : null;
    if (t === cur) return;
    cur = t;
    if (t) show(t); else hide();
  });
  document.addEventListener('scroll', hide, true);
})();
</script>"""


def build_report_document(d: Dict) -> str:
    """Assemble the full self-contained HTML document from collected
    data ``d``."""
    renderers = {
        'coverage': lambda: _coverage_tab(d),
        'branches': lambda: _branches_tab(d),
        'targets': lambda: _targets_tab(d),
        'fill': lambda: _fill_tab(d),
        'reciprocal': lambda: _reciprocal_tab(d),
        'outmap': lambda: _outmap_tab(d),
        'backward': lambda: _backward_tab(d),
        'homolog_forward': lambda: _homolog_forward_tab(d),
        'homolog_backward': lambda: _homolog_backward_tab(d),
        'suspects': lambda: _suspects_tab(d),
        'morph': lambda: _morph_tab(d),
        'pooling': lambda: _pooling_tab(d),
        'scenes': lambda: _scenes_tab(d),
        'log': lambda: _log_tab(d),
    }
    lines = [
        '<!DOCTYPE html>',
        "<html><head><meta charset='utf-8'>",
        "<title>TM VEV run report — "
        f"{_esc(Path(d['run_dir']).name)}</title>",
        report_kit.report_css(),
        _EXTRA_CSS,
        _HOVER_LAYER_CSS,
        '</head><body><main class="report-shell">',
        _hero(d),
        "<div class='tab-list' role='tablist' data-tab-list='tmvev'>",
    ]
    panel_ids = []
    for i, (key, label) in enumerate(_TABS):
        pid = f'tmvev-panel-{i}'
        panel_ids.append((key, pid))
        active = ' active' if i == 0 else ''
        selected = 'true' if i == 0 else 'false'
        lines.append(
            f"<button type='button' class='tab-button{active}' "
            "data-tab-button data-tab-group='tmvev' "
            f"data-tab-target='{pid}' aria-selected='{selected}' "
            f"role='tab'>{_esc(label)}</button>")
    lines.append('</div>')
    for i, (key, pid) in enumerate(panel_ids):
        active = ' active' if i == 0 else ''
        lines.append(
            f"<section id='{pid}' class='tab-panel{active}' "
            "data-tab-panel-group='tmvev' role='tabpanel'>")
        lines.append(renderers[key]())
        lines.append('</section>')
    lines.extend(['</main>', report_kit.report_script(),
                  _HOVER_LAYER_JS, '</body></html>'])
    return '\n'.join(lines)


def collect_and_write(run_dir: Path,
                      mapper_gap_types: Optional[Dict[str, int]] = None,
                      mapper_gap_untyped: int = 0,
                      log: Optional[Callable] = None,
                      with_notes: bool = True,
                      out_path: Optional[Path] = None) -> Optional[Path]:
    """One collection pass → report.html (+ warning notes). Fail-open:
    any error is logged and swallowed — never blocks the pipeline.
    ``out_path`` redirects the report (user 2026-09-28: regenerate a past
    run's report for testing without touching the run folder)."""
    try:
        d = collect_run_data(run_dir, mapper_gap_types, mapper_gap_untyped)
        out = Path(out_path) if out_path else Path(run_dir) / 'report.html'
        if out_path is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_report_document(d), encoding='utf-8')
        if log:
            log(f'[TMVEV] report written: {out}')
        if with_notes:
            append_warning_notes(run_dir, collect_warnings(d), log)
        return out
    except Exception as exc:  # noqa: BLE001
        if log:
            log(f'    ! run report failed: {exc}')
        return None


# pipeline-facing alias
finish_run_outputs = collect_and_write


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description='Regenerate the TM VEV per-run report.html from a '
                    "run folder's artifacts (works for past runs).")
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--out', type=Path, default=None,
                    help='write the report to this path instead of '
                         '<run_dir>/report.html — the run folder, '
                         'including its existing report, stays untouched')
    ap.add_argument('--notes', action='store_true',
                    help='also append the collected warnings to '
                         'user_warning_notes.txt')
    args = ap.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        ap.error(f'not a directory: {run_dir}')
    out = collect_and_write(run_dir, with_notes=args.notes,
                            out_path=args.out)
    return 0 if out is not None else 1


if __name__ == '__main__':
    raise SystemExit(main())
