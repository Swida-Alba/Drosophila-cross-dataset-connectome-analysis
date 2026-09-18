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
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:  # package import (usual pipeline path)
    from . import report_kit
except ImportError:  # script / notebook path
    import report_kit  # type: ignore[no-redef]

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
        'switching mode only admits more neurons, never relabels one.',
    'mutual-best (assigned)':
        'A source neuron whose best pool member is itself the target\'s '
        'best source — the strict pairing counted as "assigned".',
    'verdict':
        'Tiered rule — verified_strong (best pool member is global top-1 '
        'under BOTH rank_union and jaccard), verified (top-1 under one), '
        'borderline (top-5 window), unmatched.',
    'matched':
        'The only ASSERTED tier: the pool member is the mutual best '
        'under both metrics.',
    'verified':
        'Review tier: best pool member is top-1 under exactly one of '
        'rank_union / jaccard.',
    'borderline':
        'Review tier: best pool member sits inside the top-5 window.',
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
        'Out-of-map suspect, connectivity-qualified (invader or gap '
        'fire) AND morph-qualified — the restrictive fill.',
    'family':
        'Out-map bodyIds of THIS branch\'s target type (family/aggressive '
        'modes; type-gated, not morph-gated).',
    'relative':
        'Type-mates of candidate types outside the map (family/aggressive '
        'modes).',
    'examinees':
        'The aggressive-only deep window (out-of-pool homologs below the '
        'pool best). Renamed from \'suspicious\' 2026-09-18 — the '
        'mapper\'s rival-suspects concept now owns that word.',
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
        'An in-map type\'s neuron never claimed by any branch pool — '
        'the actionable coverage gap.',
    'family material':
        'BodyIds of in-map types not map-covered by any branch pool '
        '(superset of holes).',
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
        'Debug check: every legend leaf\'s rendered geometry matches '
        'its neuron — the CSV and the picture cannot disagree.',
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
        'Pool resolved through the bridging evidence: the selected '
        "chain's linker column values refine the full type population "
        'to the neurons actually carried by the chain.',
    'full population':
        'Pool fallback: no supported bridge chain exists, so the WHOLE '
        'type population on each side is validated.',
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
}

# per-file column/term notes for §12 expanders
FILE_GLOSSARY: Dict[str, List[str]] = {
    'validation_results.csv': [
        'verdict', 'matched', 'verified', 'borderline', 'unmatched'],
    'examinees.csv': [
        'category', 'sibling', 'candidates', 'family', 'relative',
        'out of scope', 'candidate_annotation'],
    'gap_fill_dedup.csv': ['dup', 'restrictive fill', 'family fill'],
    'gap_fill_levels.csv': ['fill levels'],
    'out_map_expansion.csv': ['out-map expansion', 'null bar'],
    'pair_summary.csv': ['gap', 'verdict'],
    'pool_categories.csv': ['pool_ref tier'],
    'morphology_calibration.json': [
        'branch bar', 'native floor', 'Track-A backup floor', 'null bar',
        'AUC gate', 'score frames'],
    'set_coverage.json': ['map-covered', 'hole', 'family material',
                          'coverage levels'],
}

ARTIFACT_LINES: List[Tuple[str, str]] = [
    ('mapping_export.csv',
     'branch-level mapping (chains, linker values, bodyId pools)'),
    ('pair_summary.csv', 'per-branch pools / gap / verdicts (§3)'),
    ('pool_categories.csv', 'per in-map target tier + best evidence'),
    ('validation_results.csv', 'source×branch verdict rows (§1)'),
    ('examinees.csv',
     'expansion bins, Revision 3.12 categories (§5); renamed from '
     'suspicious_candidates.csv'),
    ('noise_filtered_candidates.csv', 'dropped rows + noise_reason'),
    ('deep_candidates.csv', 'aggressive-only deep window'),
    ('gap_fill_proposals.csv', 'proposals, side × fill_class (§6)'),
    ('gap_fill_levels.csv', 'branch-level fill level (§6)'),
    ('gap_fill_dedup.csv', 'bodyId-unique fill (§6)'),
    ('family_candidates.csv', 'the family bin (§5)'),
    ('relatives.csv', 'the relative bin (§5)'),
    ('out_map_expansion.csv', 'unclaimed-source expansion (§7)'),
    ('source_status.csv',
     'backward `source-` status per in-branch source (advisory)'),
    ('source_candidates.csv',
     'out-of-branch sources pointing into each branch pool, tagged '
     'in-map/out-map (advisory)'),
    ('same_name_excluded.csv',
     'queried types held/excluded by the same-name-first rule, or '
     'multi-value cells (advisory accounting)'),
    ('suspects_verification.csv',
     'rival-suspect connectivity verification (opt-in, advisory)'),
    ('set_coverage.json', 'set-level coverage (§1, §4)'),
    ('morphology_calibration.json',
     'branch bars, null bar, AUC gate, score frames (§8)'),
    ('parameters.json', 'full run parameters (§11)'),
    ('pipeline_progress.jsonl', 'stage timeline events (§11)'),
]

TRUE_STR = {'true', '1', 'yes'}


# ---------------------------------------------------------------------------
# small readers / parsers
# ---------------------------------------------------------------------------

def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


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
    p = run_dir / 'examinees.csv'
    if not p.exists():
        p = run_dir / 'suspicious_candidates.csv'
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
    params = _read_json(run_dir / 'parameters.json') or {}
    calib = _read_json(run_dir / 'morphology_calibration.json') or {}
    coverage = _read_json(run_dir / 'set_coverage.json') or {}
    progress: List[Dict] = []
    pp = run_dir / 'pipeline_progress.jsonl'
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

    val_rows = _read_csv_rows(run_dir / 'validation_results.csv')
    sus_rows = _read_examinees(run_dir)
    dedup_rows = _read_csv_rows(run_dir / 'gap_fill_dedup.csv')
    levels_rows = _read_csv_rows(run_dir / 'gap_fill_levels.csv')
    prop_rows = _read_csv_rows(run_dir / 'gap_fill_proposals.csv')
    pair_rows = _read_csv_rows(run_dir / 'pair_summary.csv')
    out_rows = _read_csv_rows(run_dir / 'out_map_expansion.csv')
    fam_rows = _read_csv_rows(run_dir / 'family_candidates.csv')
    rel_rows = _read_csv_rows(run_dir / 'relatives.csv')
    same_name_excluded = _read_csv_rows(
        run_dir / 'same_name_excluded.csv')
    suspects_rows = _read_csv_rows(
        run_dir / 'suspects_verification.csv')

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

        def rank_of(r):
            try:
                return float(r.get('rank_union_rank') or 999)
            except ValueError:
                return 999.0

        best = min(rows, key=rank_of)
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

    ss_rows = _read_csv_rows(run_dir / 'source_status.csv')
    ss_counts = collections.Counter(r['status'] for r in ss_rows)
    per_type_status: Dict[str, Dict[str, int]] = {}
    for r in ss_rows:
        per_type_status.setdefault(
            r['source_type'], collections.Counter())[r['status']] += 1

    scenes = _parse_scenes(run_dir)
    selfcheck = {'pass': len(readme['selfcheck_pass']),
                 'fail': len(readme['selfcheck_fail'])}

    # mapper-consistent source split: distinct source bodyIds per branch
    # pool basis (row-evidence backed = 'linker rows'; same-name pooled =
    # 'full population'), from the SELECTED branches of mapping_export.
    basis_sources: Dict[str, set] = {}
    basis_branches: Dict[str, int] = {}
    try:
        import ast as _ast
        for r in _read_csv_rows(run_dir / 'mapping_export.csv'):
            if not _truthy(r.get('is_selected')):
                continue
            basis = r.get('pool_basis', '?')
            basis_branches[basis] = basis_branches.get(basis, 0) + 1
            try:
                ids = _ast.literal_eval(r.get('source_body_ids') or '{}')
            except (ValueError, SyntaxError):
                continue
            basis_sources.setdefault(basis, set()).update(
                int(b) for b in ids)
    except Exception:  # noqa: BLE001
        basis_sources = {}
    n_with_branch = (len(set().union(*basis_sources.values()))
                     if basis_sources else None)

    # family-material reconciliation: family material is the map-structure
    # remainder (in-map − map-covered) and is reported COMPLETE; the dedup
    # bins split it by fill-accounting precedence (candidates > family), so
    # the note names which members the candidates bin claims.
    fm_ids = {int(b) for b in (coverage.get('mcns') or {}).get(
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

    return {
        'run_dir': run_dir,
        'params': params,
        'calib': calib,
        'coverage': coverage,
        'progress': progress,
        'readme': readme,
        'mapper_gap': dict(gap or {}),
        'mapper_gap_untyped': gap_untyped,
        'fafb': coverage.get('fafb') or {},
        'mcns': coverage.get('mcns') or {},
        'scanned_sources': scanned_sources,
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
        'suspects_verification': suspects_rows,
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

def collect_warnings(d: Dict) -> List[str]:
    lines = []
    sc = d['selfcheck']
    n_checks = sc['pass'] + sc['fail']
    if n_checks:
        lines.append(f"[self-check] {sc['pass']}/{n_checks} scene "
                     'self-checks passed (all legend leaves match their '
                     'neuron geometry)')
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


def _term(key: str, label: Optional[str] = None) -> str:
    """Hoverable term: dotted underline + CSS-only tooltip."""
    definition = TERM_DEFS.get(key)
    text = label or key
    if not definition:
        return _esc(text)
    return (f"<span class='term'>{_esc(text)}"
            f"<span class='tip'><b>{_esc(text)}</b>"
            f'{_esc(definition)}</span></span>')


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


def _viewport(rows_html: List[str], headers: str, n: int = 20) -> str:
    """First ``n`` rows in the open table; the rest behind a details
    catch-all (the 20-row viewport convention)."""
    head = f'<thead><tr>{headers}</tr></thead>'
    if len(rows_html) <= n:
        return ("<div style='overflow-x:auto'>"
                f"<table class='mv-table'>{head}"
                f"<tbody>{''.join(rows_html)}</tbody></table></div>")
    open_rows = ''.join(rows_html[:n])
    rest_rows = ''.join(rows_html[n:])
    return (
        "<div style='overflow-x:auto'><table class='mv-table'>"
        f'{head}<tbody>{open_rows}</tbody></table></div>'
        '<details class="detail-block"><summary>Show all '
        f'{len(rows_html)} rows (first {n} above)</summary>'
        "<div style='overflow-x:auto'><table class='mv-table'>"
        f'{head}<tbody>{rest_rows}</tbody></table></div></details>')


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

    f_, m = d['fafb'], d['mcns']
    total = f_.get('total_queried')
    covered = m.get('in_branch_pool')
    mapped_set = m.get('mapped_target_set')
    if total and covered is not None:
        headline = (
            f'{_esc(total)} source neurons → <b>{_esc(covered)}</b> '
            f'{_esc(params.get("target_dataset", ""))} neurons '
            f'{_term("map-covered")} (of {_esc(mapped_set)} in-map; '
            f'{_pct(covered, mapped_set)}) · '
            f'{_term("mutual-best (assigned)", "mutual-best")} '
            f'{_esc(f_.get("assigned", "—"))} · fill-proposed '
            f'+{_esc(f_.get("fill_proposed_only", "—"))}')
    else:
        headline = ('Coverage artifacts absent — the run produced no '
                    'set-level coverage (see the Log tab).')

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
        f'{datetime.now():%Y-%m-%d %H:%M:%S} · hover any dotted term '
        'for its definition · every number traces to a run artifact '
        '(Log tab lists them) · <a href="_UserGuide_please_read_me.html">'
        '📘 User guide — this run\'s files and terms, '
        'explained</a></p>'
        '</header>')


def _coverage_tab(d: Dict) -> str:
    f_, m = d['fafb'], d['mcns']
    cards = []
    if f_ and m:
        # §1 the three coverage levels
        l1 = _kv_block('L1 CLAIM (forward) — what does the map cover?', [
            ('In-map target population',
             f"{_esc(m.get('mapped_target_set', '—'))} neurons "
             f'({len(m.get("per_type") or {})} target types)'),
            (_term('map-covered', 'Map-covered (branch pools)'),
             f"{_esc(m.get('in_branch_pool', '—'))} "
             f'({_pct(m.get("in_branch_pool"), m.get("mapped_target_set"))})'),
            ('Reached only as candidates',
             _esc(m.get('reached_as_candidates_only', '—'))),
            (_term('hole', 'Holes (never map-covered)'),
             f"{_esc(m.get('holes', '—'))} ▸ Targets tab"),
            (_term('family material'),
             f'{len(m.get("family_material") or [])} '
             f'(= {m.get("mapped_target_set")} − '
             f'{m.get("in_branch_pool")}) ▸ Targets tab'),
        ])
        tiers = {k: d['dedup_cat'].get(k, 0)
                 for k in ('matched', 'verified', 'borderline')}
        l2 = _kv_block(
            'L2 PROVENANCE — how was each of the '
            f'{_esc(m.get("in_branch_pool", "…"))} earned?', [
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
             f"{_esc(f_.get('total_queried', '—'))} queried"),
            (_term('mutual-best (assigned)', 'Mutual-best (assigned)'),
             f"{_esc(f_.get('assigned', '—'))} sources"),
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
        assigned_n = f_.get('assigned', 0)
        proposed_n = f_.get('fill_proposed_only', 0)
        unpaired = f_.get('unpaired_unproposed', 0)
        unclaimed = len(d['out_sources'])
        in_pool_residue = unpaired - unclaimed
        basis = d['basis_sources']
        n_linker = len(basis.get('linker rows', set()))
        n_full = len(basis.get('full population', set()))
        b_linker = d['basis_branches'].get('linker rows', 0)
        b_full = d['basis_branches'].get('full population', 0)
        covered = m.get('in_branch_pool', '—')
        total_q = f_.get('total_queried', '—')
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
                f"<tr><td>{_esc(f_.get('total_queried', '—'))} queried</td>"
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
              '<thead><tr><th>bucket</th><th>meaning</th></tr></thead>'
              "<tbody>" + ''.join(wf_rows) + '</tbody></table></div>'
              "<p class='mv-note'>Consistent with the type mapper: its "
              'mapped set is the map-covered '
              f'{_esc(covered)} — family material is unmapped in-map '
              'material and stays outside it. Per-source backward '
              'statuses: see the Backward tab (informational only).</p>')
        cards.append(_section_card(
            f"Source side — the {f_.get('total_queried', '…')}, "
            'accounted',
            'Mapper-consistent buckets (row-evidence backed / same-name '
            'pooled / out-map); proposals are evidence, the mapping is '
            'never rewritten.',
            wf, ['mutual-best (assigned)', 'map-covered',
                 'validation mode']))

        # per-type table (21 rows here)
        per = f_.get('per_type') or {}
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
        body = _viewport(
            rows,
            '<th>source type</th><th>pool</th><th>assigned</th>'
            '<th>fill-proposed</th><th>unpaired</th>')
        cards.append(_section_card(
            'Per-type pools (source)',
            'Feeds from set_coverage.json fafb.per_type.', body))
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
    if any((snf_pairs, snf_held, snf_excl, mv_types, mv_tgt)):
        snf_rows = [
            ('Same-name-first selections (fired)',
             f"{snf_pairs} pair(s) across "
             f"{cov.get('same_name_first_types') or 0} source "
             'type(s) — marked ⟡ in Branches; rivals withheld to '
             'auto_type_mapping_suspects.csv'),
        ]
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
        cards.append(_section_card(
            'Same-name-first & multivalue accounting',
            'Advisory record of how the mapper\'s same-name-first '
            'rule shaped this run\'s pair set. Never a gate.',
            _kv_block('', snf_rows),
            ['same-name-first', 'suspects', 'multi-value type cells']))

    return ''.join(cards)


def _branches_tab(d: Dict) -> str:
    entries = []
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
        scene_cell = (f"<a href='{_esc(scene['html'])}'>▶</a>"
                      if scene and scene['html'] else '—')
        hemi = ''
        try:
            h = ast.literal_eval(s.get('hemisphere', '') or '{}')
            if h.get('hemisphere_asymmetry'):
                hemi = (" <span class='warn-chip' title='Hemisphere "
                        'asymmetry fires the gap even at arithmetic '
                        "gap 0'>⚠</span>")
        except (ValueError, SyntaxError):
            pass
        try:
            gap_ratio = f'{float(s.get("gap_ratio") or 0):.0%}'
        except ValueError:
            gap_ratio = '—'
        verdicts = ' / '.join(
            str(s.get(k, 0)) for k in
            ('verdict_verified_strong', 'verdict_verified',
             'verdict_borderline', 'verdict_unmatched'))
        susp = (f"{_esc(s.get('suspicious_neurons', 0))} / "
                f"{_esc(s.get('suspicious_noise_filtered', 0))}")
        triggered = _truthy(s.get('gap_triggered'))
        entries.append({
            'key': (not triggered, str(src), str(tgt)),
            'html': (
                f"<tr><td>{_esc(src)} → {_esc(tgt)}{snf_marker}</td>"
                f"<td>{_esc(s.get('mapping_status', ''))} / "
                f"{_esc(s.get('pool_basis', ''))}</td>"
                f"<td>{_esc(s.get('source_pool', '—'))}/"
                f"{_esc(s.get('source_type_total', '—'))} → "
                f"{_esc(s.get('target_pool', '—'))}/"
                f"{_esc(s.get('target_type_total', '—'))}</td>"
                f"<td>{_esc(s.get('matched', '—'))}</td>"
                f"<td>{_esc(s.get('gap', '—'))} ({gap_ratio})</td>"
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
        "<th title='One row per branch: a (source type, target type) "
        "validation pair of the query. Sorted by branch.'>Branch "
        "(source → target)</th>"
        "<th title='Mapper decision status (mapped / evidence_only / …) "
        'and how the source pool was resolved: linker rows = the '
        'bridging-evidence subset; full population = no supported '
        "chain, the whole type population.'>Mapping status / pool "
        'basis</th>'
        "<th title='Validated pool sizes: source neurons of the source "
        'type total → target neurons of the target type total.'
        "'>Pools (source of total → target of total)</th>"
        "<th title='Mutual-best (assigned) source–target pairs — the "
        "only ASSERTED tier.'>Matched (M)</th>"
        "<th title='min(|source pool|, |target pool|) − M, with its "
        'pool ratio. Informational — proposals only, the mapping is '
        "never rewritten.'>Gap</th>"
        "<th title='● = the branch fired the gap rule (advisory; the "
        'trigger no longer gates anything); ⚠ marks hemisphere '
        "asymmetry.'>Gap triggered</th>"
        "<th title='Per-branch source-side verdict counts: "
        'verified_strong (v★) / verified (v) / borderline (b) / '
        "unmatched (u).'>Verdicts v★ / v / b / u</th>"
        "<th title='Examinee rows (non-pool neurons ranked ahead of the "
        'pool) / rows dropped by the noise gates (spatial caliber, '
        "negative rank_union, jaccard sanity, tie margin).'>Examinee "
        'rows / noise filtered</th>'
        "<th title='The branch morph admission bar and its kind: native "
        "floor / track_a_backup / null (run-sensitive).'>Morph bar "
        '(kind)</th>'
        "<th title='Link to the branch 3D scene, when rendered."
        "'>Scene</th>")
    table = _scroll_viewport(rows, headers, visible=50) \
        if rows else _empty('pair_summary.csv absent or empty.')
    detail_cols = ('selected_chain', 'branch_linker_values',
                   'branch_annotation', 'branches_disjoint',
                   'pool_widen_added', 'deep_candidates', 'null_sample')
    det_rows = []
    for s in d['pair_rows']:
        det_rows.append(
            f"<tr><td>{_esc(s.get('source_type'))} → "
            f"{_esc(s.get('target_type'))}</td>"
            + ''.join(f"<td>{_esc(s.get(c, ''))}</td>"
                      for c in detail_cols) + '</tr>')
    details = _scroll_viewport(
        det_rows,
        '<th>branch</th>'
        + ''.join(f'<th>{_esc(c)}</th>' for c in detail_cols))
    body = table + (
        '<details class="detail-block"><summary>Per-branch chain &amp; '
        'linker details</summary>' + details + '</details>')
    return _section_card(
        f'Branches ({len(d["pair_rows"])})',
        'One row per branch, sorted by branch. All rows render in one '
        'scrollable table (50-row viewport). Hover any column header '
        'for what it measures; verdict columns are per-branch '
        'source-side counts (v★ = verified_strong).',
        body,
        ['branch bar', 'verdict', 'gap', 'validation mode'])


def _targets_tab(d: Dict) -> str:
    m = d['mcns']
    per = m.get('per_type') or {}
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
        body = _viewport(
            rows,
            '<th>target type</th><th>mapped pop</th>'
            '<th>map-covered</th><th>tier m / v / b</th>'
            '<th>cand-only</th><th>holes</th>')
        summary = (
            f"Mapped target population "
            f"{_esc(m.get('mapped_target_set', '—'))} neurons across "
            f'{len(per)} types — holes get their bodyIds inline, '
            'because holes are the actionable output.')
    else:
        body = _empty('set_coverage.json mcns block absent.')
        summary = 'Target-side coverage not available for this run.'
    fam = m.get('family_material') or []
    fam_html = ''
    if fam:
        fam_html = (
            '<details class="detail-block"><summary>'
            f'{_term("family material")} ({len(fam)} bodyIds = '
            f"{_esc(m.get('mapped_target_set', '—'))} in-map − "
            f"{_esc(m.get('in_branch_pool', '—'))} map-covered; the type "
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
            '<thead><tr><th>bin</th><th>branch-level rows</th>'
            '<th>dedup</th></tr></thead><tbody>'
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
        + (f' + {_esc(n_cross)} cross-bin via out-of-pool same-type '
           'fill proposals; provenance per row below — decided D8)'
           if n_cross else ')') + ':</p>'
        "<div style='overflow-x:auto'><table class='mv-table'>"
        '<thead><tr><th>class</th><th>rows</th><th>bodyIds</th>'
        '<th>meaning</th></tr></thead><tbody>' + br_html_rows
        + '</tbody></table></div>')
    # family-material reconciliation (user 2026-09-17): the bin's dedup
    # count is NOT the family-material total — family outranks candidates
    # in the dedup, so a family-material bodyId with family rows rolls up
    # as `family`; one without any family row (it was candidate-labeled
    # inside its own branch) stays `candidates`.
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
            f"{len(d['fm_ids'])} (= {d['mcns'].get('mapped_target_set')} "
            f"in-map − {d['mcns'].get('in_branch_pool')} map-covered) = "
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
    summ = _kv_block('Fill accounting', [
        (_term('fill levels', 'Fill by level (branch-level)'), lv_txt),
        (_term('restrictive fill', 'Fill, bodyId-unique (dedup)'),
         f'{len(d["restrictive"])} restrictive · family-fill '
         f'+{len(d["family_fill"])} ({ff_txt})'),
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
        sus = d['sus_cand_by_bid'].get(bid)
        if sus is not None:
            tok = sus.get('candidate_annotation', '?')
            kind = sus.get('bar_kind', '')
            bar_txt = f'{_f(sus.get("bar_value"))} ({kind})' \
                if kind else _f(sus.get('bar_value'))
            prov = 'expansion row'
        else:
            prop = d['prop_by_bid'].get(bid)
            tok = f"{r.get('target_type', '?')}(out-map)"
            bar_txt = '—'
            prov = 'out-of-pool same-type proposal'
            if prop:
                prov += (f" ({prop.get('source_type')}→"
                         f"{prop.get('target_type')}, source verdict "
                         f"{prop.get('source_verdict')})")
        lvl = d['levels_by_bid'].get(bid) or {}
        rows.append(
            f'<tr><td>{_esc(bid)}</td>'
            f"<td>{_esc(r.get('target_type'))}</td>"
            f'<td>{_esc(tok)}</td><td>{_esc(bar_txt)}</td>'
            f"<td>{_esc(lvl.get('level', '—'))}</td>"
            f"<td>{_esc(lvl.get('dup', r.get('dup', '')))}</td>"
            f'<td>{_esc(prov)}</td></tr>')
    table = _viewport(
        rows,
        '<th>target bodyId</th><th>type</th><th>leaf token</th>'
        '<th>bar (kind)</th><th>level</th><th>dup</th>'
        '<th>provenance</th>') \
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
         '{T}(out-map)'])
    return sec5 + sec6


def _outmap_tab(d: Dict) -> str:
    if not d['out_rows_total']:
        total_q = d['fafb'].get('total_queried')
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
        sorted(d['out_type_counts'].items(), key=lambda kv: -kv[1])[:6])
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
        rows.append(
            f"<tr><td>{_esc(r.get('source_type'))} {_esc(item['src'])}"
            f'</td>'
            f"<td>{_esc(r.get('target_bodyId'))} "
            f"{_esc(r.get('target_type'))}</td>"
            f"<td>{_f(r.get('rank_union'))}</td>"
            f"<td>{_f(r.get('jaccard'))}</td>"
            f"<td>{_f(r.get('morph_v2_similarity'))} {q}</td>"
            f"<td>{item['n_q']}/{item['n_total']}</td></tr>")
    table = _viewport(
        rows,
        '<th>source</th><th>best candidate</th><th>rank_union</th>'
        '<th>jaccard</th><th>morph ✓/✗</th><th>qualified</th>')
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
    f_, m = d['fafb'], d['mcns']
    basis = d['basis_sources']
    n_linker = len(basis.get('linker rows', set()))
    n_full = len(basis.get('full population', set()))
    total_q = f_.get('total_queried', '—')
    out_n = (total_q - d['n_with_branch']
             if isinstance(total_q, (int, float))
             and d.get('n_with_branch') is not None
             else len(d['out_sources']))
    primary = _kv_block('Source population — where every neuron sits', [
        ('In-map (branch pools)',
         f'{n_linker + n_full} of {_esc(total_q)} — row-evidence backed '
         f'{n_linker} (linker rows) · same-name pooled {n_full} '
         '(full population)'),
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
        pt_table = _viewport(
            pt_rows,
            '<th>source type</th><th>source-matched</th>'
            '<th>source-verified</th><th>source-borderline</th>'
            '<th>source-unmatched</th>')
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
    sc_export = _read_csv_rows(Path(d['run_dir']) / 'source_candidates.csv')
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
                '<thead><tr><th>target branch</th>'
                '<th>distinct out-of-map sources</th>'
                '<th>their types</th></tr>'
                '</thead><tbody>' + ''.join(rows)
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
                    Path(d['run_dir']) / 'mapping_export.csv'):
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
                '<thead><tr><th>target branch</th>'
                '<th>distinct out-of-branch sources</th>'
                '<th>their types (sibling rows)</th></tr>'
                '</thead><tbody>' + ''.join(rows)
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


def _suspects_tab(d: Dict) -> str:
    """P3 (opt-in): advisory verification of the mapper's rival
    suspects — per rival, the ordinary tier machinery applied to the
    rival's own target pool."""
    rows = d.get('suspects_verification') or []
    if not rows:
        return _section_card(
            'Suspects (same-name rivals)',
            'Connectivity check of the rival candidates the mapper '
            'withheld when the same-name-first rule fired.',
            _empty('suspects_verification.csv absent — the pass is '
                   'opt-in (--verify-suspects) and was not run.'),
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
        '<thead><tr><th>source type</th><th>rival (selection)</th>'
        '<th>disposition</th><th>mapper evidence</th><th>votes</th>'
        '<th>pop src→tgt</th>'
        '<th>sources</th><th>verdicts vS / v / b / u</th></tr></thead>'
        '<tbody>' + ''.join(trs) + '</tbody></table></div>')
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
            'bars ACROSS runs (native-floor branches are '
            'unaffected).</div>')
    return _section_card(
        'Morphology record',
        'Per-branch bars live in the Branches tab; this section only '
        'aggregates.',
        summ + advisory,
        ['AUC gate', 'null bar', 'branch bar', 'native floor',
         'Track-A backup floor', 'score frames', 'pool_ref tier'])


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
    body = (
        f"<p class='section-summary'>{len(d['scenes'])} scenes rendered "
        '— only types with renderable expansion content get a scene '
        '(decided): a scene with nothing rendered is noise.</p>'
        f"<div class='scene-grid'>{''.join(tiles)}</div>"
        "<p class='mv-note'>Scenes render in SOURCE coordinates — read "
        'as anatomy, never as the scoring frame (morph tracks score in '
        'TARGET coordinates).</p>')
    return _section_card(
        'Scenes',
        f'Self-check {sc["pass"]}/{sc["pass"] + sc["fail"]}: every '
        "legend leaf's rendered geometry matches its neuron — the CSV "
        'and the picture cannot disagree.',
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
            "<p class='mv-note'>This run: no ``!`` failures. "
            f'{checks}</p>')
    if d['advisories']:
        warn_body += ("<p class='mv-note'>Advisories carried: "
                      + _esc(', '.join(d['advisories'])) + '.</p>')
    sec10 = _section_card('Warnings & anomalies', '', warn_body)

    # §11 provenance
    params = d['params']
    order = ['validation_mode', 'top_k', 'top_m', 'min_synapse_threshold',
             'rank_top_k', 'gap_min', 'suspicious_jaccard_factor',
             'morph_enabled', 'morph_auc_floor', 'morph_track_a_offset',
             'morph_suspicious_level', 'suspicious_per_source_cap',
             'scene_selfcheck', 'aggressive_expansion', 'pool_widen']
    pl = ' · '.join(f'{_esc(k)}={_esc(params[k])}'
                    for k in order if k in params)
    timeline: List[str] = []
    stage_open: Dict[str, str] = {}
    for p in d['progress']:
        ev = p.get('event')
        if ev == 'stage_start':
            stage_open[str(p.get('stage'))] = str(p.get('ts', ''))
        elif ev == 'stage_done':
            st = str(p.get('stage'))
            t0 = stage_open.pop(st, None)
            dur = ''
            if t0:
                try:
                    dt = (datetime.fromisoformat(str(p.get('ts')))
                          - datetime.fromisoformat(t0)).total_seconds()
                    dur = f' ({dt:.0f} s)'
                except (ValueError, TypeError):
                    pass
            timeline.append(f'stage {st}{dur}')
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
        n = _n_rows(d['run_dir'] / name)
        idx_rows.append(
            f'<tr><td>{_esc(name)}</td><td>'
            + ('—' if n is None else f'<span class="missing">{n:,}</span>')
            + f'</td><td>{_esc(blurb)}</td></tr>')
    idx_rows.append(
        f"<tr><td>visualization/*.html</td><td>{len(d['scenes'])}</td>"
        '<td>scenes (Scenes tab)</td></tr>')
    idx_rows.append(
        '<tr><td>README.txt</td><td>—</td><td>slim directions + raw run '
        'log (the analysis lives here in report.html)</td></tr>')
    idx = ("<div style='overflow-x:auto'><table class='mv-table'>"
           '<thead><tr><th>artifact</th><th>rows</th><th>one-liner>'
           '</th></tr></thead><tbody>' + ''.join(idx_rows)
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
.mv-table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
.mv-table th, .mv-table td { border: 1px solid var(--line);
    padding: 6px 9px; text-align: left; vertical-align: top; }
.mv-table thead th { background: var(--surface-soft); font-size: 11.5px;
    text-transform: uppercase; letter-spacing: .04em;
    color: var(--muted); }
.mv-table.mv-defs td:first-child { white-space: nowrap;
    font-weight: 700; }
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
.missing { color: var(--muted); }
.report-hero .report-subtitle { font-size: 16px; color: var(--ink); }
.report-hero .report-subtitle b { color: var(--accent-dark); }
</style>"""


# ---------------------------------------------------------------------------
# assembly + entry points
# ---------------------------------------------------------------------------

_TABS: List[Tuple[str, str]] = [
    ('coverage', 'Coverage'), ('branches', 'Branches'),
    ('targets', 'Targets'), ('fill', 'Fill'), ('outmap', 'Out-map'),
    ('backward', 'Backward'), ('suspects', 'Suspects'),
    ('morph', 'Morph'), ('scenes', 'Scenes'), ('log', 'Log'),
]


def build_report_document(d: Dict) -> str:
    """Assemble the full self-contained HTML document from collected
    data ``d``."""
    renderers = {
        'coverage': lambda: _coverage_tab(d),
        'branches': lambda: _branches_tab(d),
        'targets': lambda: _targets_tab(d),
        'fill': lambda: _fill_tab(d),
        'outmap': lambda: _outmap_tab(d),
        'backward': lambda: _backward_tab(d),
        'suspects': lambda: _suspects_tab(d),
        'morph': lambda: _morph_tab(d),
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
                  '</body></html>'])
    return '\n'.join(lines)


def collect_and_write(run_dir: Path,
                      mapper_gap_types: Optional[Dict[str, int]] = None,
                      mapper_gap_untyped: int = 0,
                      log: Optional[Callable] = None,
                      with_notes: bool = True) -> Optional[Path]:
    """One collection pass → report.html (+ warning notes). Fail-open:
    any error is logged and swallowed — never blocks the pipeline."""
    try:
        d = collect_run_data(run_dir, mapper_gap_types, mapper_gap_untyped)
        out = Path(run_dir) / 'report.html'
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
    ap.add_argument('--notes', action='store_true',
                    help='also append the collected warnings to '
                         'user_warning_notes.txt')
    args = ap.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        ap.error(f'not a directory: {run_dir}')
    out = collect_and_write(run_dir, with_notes=args.notes)
    return 0 if out is not None else 1


if __name__ == '__main__':
    raise SystemExit(main())
