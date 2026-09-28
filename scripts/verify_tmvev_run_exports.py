#!/usr/bin/env python3
"""Audit the files a TM VEV run exports, against the pipeline's own registry.

Usage
    python scripts/verify_tmvev_run_exports.py [PATH ...]

Each PATH is a run folder (`type-map-validation_*`) or a directory holding a
`manifest.tsv` of them (a queue run); with no PATH the newest `tmvev-*`
queue directory under `local_data/` is used. Every check is derived from the run folder itself,
so this is what to point at any TM VEV result — the ladder claims below are
checked per (source, target, query) group and runs of different queries are
never compared to each other.

What it proves, in the order the skill says to read a run:

* **schema** — each CSV's header equals `_RUN_CSV_SCHEMAS` and the file sits
  where `RUN_FILE_LAYOUT` says it sits, so a renamed or relocated export cannot
  pass by being looked for under its old name;
* **the ladder** — `restrictive <= family <= aggressive`: the tier and the pool
  arithmetic are identical, no row changes category as the mode widens, each
  expansion bin only grows, the candidate-window feed of the narrower mode is
  kept by the wider one, holes only close, and ONE null backdrop grades all
  three (each of those earned its keep on 2026-09-24, when four of them failed
  for two reasons now fixed — see the two window/backdrop notes below);
* **the partition** — categories are in the Rev 3.12 enum, `in_scope` and
  `morph_failed` are never both true, the fill counters follow the category,
  and every expansion leaf carries at most one leaf token;
* **the pooling ledger** — a `pooling` run's pool is the connectivity-admitted
  targets with the ones its morphology bar refused marked `in_pool=False`, every
  refusal counted, a bar-driven run's floors published as the advisory flags they
  are (and every flagged row still present), each row's tier in the pooling
  vocabulary, every borrowed verdict naming the pair it came from, one source row
  per queried neuron, and the target-vector store's ledger present;
* **the report** — every tab button has a panel and vice versa, the Pooling tab
  renders, its pool table has the column count it claims, deleted features are
  gone, and no cell leaks `@ None` / `nan`;
* **cost** — every stage's seconds, read from `pipeline_progress.jsonl` rather
  than from wall-clock folklore.

Exit code 0 means no FAIL line was printed.
"""
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from comparison.mapping_validation import (  # noqa: E402
    RUN_FILE_LAYOUT, _RUN_CSV_SCHEMAS)

LADDER = ('restrictive', 'family', 'aggressive')
# Rev 3.12's ordered first-match categories; '' is a connectivity-only row,
# and `examinees` is the aggressive deep-window bin (renamed from `suspicious`
# 2026-09-18, which now means the mapper's rival suspects).
CATEGORIES = {'matched', 'verified', 'borderline', 'unmatched', 'sibling',
              'candidates', 'family', 'relative', 'examinees', ''}
TOKENS = ('(out-map)', '(no_source)', 'untyped')
# pair_summary columns whose value is a COUNT OF A MODE-SPECIFIC BIN, so they
# legitimately differ along the ladder; everything else must not move.
MODE_COLUMNS = ('deep_candidates', 'suspicious_count',
                'suspicious_noise_filtered', 'suspicious_size_filtered',
                'suspicious_tie_filtered', 'candidates', 'family', 'relative',
                'examinees', 'null_sample')

_fails = []
_infos = []
#: Queue folders with no `parameters.json` — a run still in flight, skipped on
#: purpose and named in the report so a subset never reads as a full verdict.
_INCOMPLETE = []


def chk(ok, name, detail=''):
    print(f'{"PASS" if ok else "FAIL":4}  {name}'
          + (f'  [{detail}]' if detail and not ok else ''))
    if not ok:
        _fails.append(name + (f'  [{detail}]' if detail else ''))


def info(name, detail=''):
    _infos.append(f'{name}' + (f': {detail}' if detail else ''))


def as_bool(v):
    return str(v).strip().lower() in ('true', '1', 'yes')


def path_of(run, fname):
    """A run's file in the current layout, falling back to the FLAT pre-
    2026-09-19 layout so an old folder still opens (the pipeline's readers do
    the same, so an audit that cannot read them is not a clean audit)."""
    sub = RUN_FILE_LAYOUT.get(fname, '')
    p = (run / sub / fname) if sub else (run / fname)
    if p.exists() or not sub:
        return p
    flat = run / fname
    return flat if flat.exists() else p


def table(run, fname):
    p = path_of(run, fname)
    if not p.exists():
        return None
    return pd.read_csv(p, dtype=str).fillna('')


# ---------------------------------------------------------------- discovery
def discover(paths):
    """Run folders from argv, unioned with every folder a queue dir holds.

    The manifest is a BONUS, never the authority: it carries the exit code and
    mode label the driver recorded, but a queue whose driver was stopped between
    runs leaves folders with no row at all, and auditing only the manifest would
    hand back a verdict computed on a subset. So every ``type-map-validation_*``
    folder under a queue directory is read, and the manifest is consulted only
    for the rows it actually has.
    """
    runs = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if p.name.startswith('type-map-validation'):
            runs.append((p, None, None, None, None))
            continue
        if not p.is_dir():
            chk(False, f'not a run folder or queue dir: {p}')
            continue
        seen = set()
        man = p / 'manifest.tsv'
        if man.exists():
            for line in man.read_text().splitlines()[1:]:
                f = line.split('\t')
                if len(f) > 6 and f[4] and Path(f[4]).exists():
                    run = Path(f[4])
                    runs.append((run, f[1], f[2], f[5], f[6]))
                    seen.add(run.name)
        for folder in sorted(p.glob('type-map-validation_*')):
            if folder.name in seen:
                continue
            # A folder without parameters.json is a run still in flight (the
            # file is written as the run closes); auditing it would read a half
            # -written export set and report the gap as a defect.
            if path_of(folder, 'parameters.json').exists():
                runs.append((folder, None, None, None, None))
            else:
                _INCOMPLETE.append(folder)
    return runs


def group_key(run):
    """What identifies a comparable run: source, target, and the query."""
    par = json.loads(path_of(run, 'parameters.json').read_text('utf-8'))
    return (par.get('source_dataset'), par.get('target_dataset'),
            tuple(par.get('query_types') or ()))


# ---------------------------------------------------------------- per run
def check_layout(run, tag):
    bad = []
    for fname, cols in _RUN_CSV_SCHEMAS.items():
        p = path_of(run, fname)
        if not p.exists():
            continue
        try:
            head = pd.read_csv(p, nrows=0).columns.tolist()
        except Exception as exc:                          # noqa: BLE001
            bad.append(f'{fname}: unreadable ({exc})')
            continue
        if head != cols:
            bad.append(f"{fname}: +{sorted(set(head) - set(cols))} "
                       f"-{sorted(set(cols) - set(head))}")
    chk(not bad, f'{tag}: CSV schemas match the registry', '; '.join(bad[:3]))
    # A scene folder is created before the figure is written, so a render that
    # dies halfway leaves a directory with `parameters.txt` and the layer CSVs
    # but no HTML — indistinguishable from a success on disk. Either the page and
    # its manifest are there, or a SCENE_FAILED.txt says they are not (found
    # this way: 5 of 21 parent scenes on a male-cns family run were empty
    # folders while the audit still reported a clean run).
    scenes = sorted((run / 'visualization').glob('plot-3d_*')) \
        if (run / 'visualization').is_dir() else []
    broken = []
    for folder in scenes:
        html = list(folder.glob('*.html'))
        marked = (folder / 'SCENE_FAILED.txt').exists()
        if not html and not marked:
            broken.append(folder.name)
        elif html and not (folder / 'visualization_manifest.json').exists():
            broken.append(f'{folder.name} (html, no manifest)')
    chk(not broken, f'{tag}: every scene folder rendered or says it failed',
        f'{len(scenes)} folders; incomplete: {broken[:6]}' if broken
        else f'{len(scenes)} folders')
    for name in ('README.txt', 'report.html', 'parameters.json',
                 'set_coverage.json', 'pipeline_progress.jsonl',
                 '_UserGuide_please_read_me.html'):
        if name == 'set_coverage.json' and not path_of(run, name).exists():
            # check_coverage names the explained-empty case; the layout
            # sweep must not re-fail it
            continue
        if not path_of(run, name).exists():
            chk(False, f'{tag}: missing {name}')
    # the old name must not exist; narrating the rename in prose is fine
    old = list(run.rglob('suspicious_candidates.csv'))
    chk(not old and 'suspicious_candidates.csv' not in RUN_FILE_LAYOUT,
        f'{tag}: suspicious_candidates.csv is gone from the export', str(old))
    bangs = [l for l in path_of(run, 'README.txt').read_text(
        'utf-8', errors='replace').splitlines() if l.startswith('!')]
    if bangs:
        info(f'{tag} README !-lines', f'{len(bangs)} · {bangs[0][:150]}')
    wn = path_of(run, 'user_warning_notes.txt')
    if wn.exists() and wn.read_text().strip():
        info(f'{tag} user warnings',
             wn.read_text()[:200].replace('\n', ' | '))


def check_report(run, tag, mode):
    html = path_of(run, 'report.html').read_text('utf-8', errors='replace')
    chk('@ None' not in html, f'{tag}: report names what it cannot record')
    for gone in ('targets_corroborated', 'floor_from_evidence'):
        chk(gone not in html, f'{tag}: no {gone} in the report')
    leak = [m for m in ('>nan<', '>None<', '>NaN<', 'None%', 'nan%')
            if m in html]
    chk(not leak, f'{tag}: no blank-cell leakage', str(leak))
    # tabs: every button needs a panel and every panel a button — a tab that
    # opens nothing is invisible to someone clicking through the report
    tabs = re.findall(r"data-tab-target='([^']+)'[^>]*>([^<]+)</button>", html)
    panels = re.findall(r"<section id='([^']+)'", html)
    names = [n for _k, n in tabs]
    info(f'{tag} tabs', ' '.join(names))
    chk(sorted(k for k, _n in tabs) == sorted(panels),
        f'{tag}: every tab button has a panel',
        f'{len(tabs)} buttons vs {len(panels)} panels')
    # EVERY run renders the Pooling tab: a supervised run's tab states that
    # pooling is PARALLEL to the ladder, so an absent tab is the defect
    chk('Pooling' in names, f'{tag}: Pooling tab rendered', str(names))
    pid = next((k for k, n in tabs if n == 'Pooling'), '')
    i = html.find(f"<section id='{pid}'")
    j = html.find("<section id='", i + 20)
    panel = html[i:j if j > i else len(html)]
    if mode == 'pooling':
        # the card holds the harvest table first, so pick the pool table by
        # its own header rather than by position
        m = panel.find('Pooling pool —')
        after = panel[m:] if m >= 0 else ''
        tables = re.findall(r'<table.*?</table>', after, re.S)
        tbl = next((t for t in tables if 'best source' in t), '')
        widths = {len(r.split('</td>')) - 1 for r in tbl.split('<tr>')[1:]
                  if '</td>' in r}
        chk(bool(tbl) and widths == {6},
            f'{tag}: Pooling panel renders its 6-cell pool table',
            f'{len(tables)} tables, widths {sorted(widths)}')
        chk('refused' in panel.lower() or 'no evidence' in panel.lower(),
            f'{tag}: Pooling panel speaks the morphology gate')
    else:
        chk('PARALLEL' in panel,
            f'{tag}: Pooling tab says the mode is parallel', panel[:120])


def check_coverage(run, tag):
    cov_path = path_of(run, 'set_coverage.json')
    if not cov_path.exists():
        # An empty run (every queried type fail-closed at stage 1, e.g. a
        # category that does not exist on the source side) writes no
        # coverage artifact at all — the README's own log says so.  That is
        # an explained outcome, not an unauditable run (the POL-1 class:
        # a legitimate run must not fail its auditor).
        notes = path_of(run, 'README.txt')
        explained = notes.exists() and 'no valid type pairs' in notes.read_text(
            'utf-8', errors='replace')
        if explained:
            info(f'{tag}: no set_coverage.json — the run resolved no type '
                 'pairs (README records the fail-closed query)')
            return
    cov = json.loads(cov_path.read_text('utf-8'))
    src, tgt = cov.get('source') or {}, cov.get('target') or {}
    bad = []
    # per type: mapped_population - in_pool == reached_as_candidates_only +
    # holes, and a hole count that does not match its own bodyId list is a
    # report that cannot be audited
    for t, d in (tgt.get('per_type') or {}).items():
        mp, ip = d.get('mapped_population'), d.get('in_pool')
        only, hl = d.get('reached_as_candidates_only'), d.get('holes')
        ids = d.get('hole_body_ids') or []
        if None in (mp, ip, only, hl):
            bad.append(f'{t}: keys missing {sorted(d)}')
            continue
        if int(mp) - int(ip) != int(only) + int(hl):
            bad.append(f'{t}: {mp}-{ip} != {only}+{hl}')
        if int(hl) != len(ids):
            bad.append(f'{t}: holes {hl} != {len(ids)} bodyIds listed')
    chk(not bad, f'{tag}: set_coverage per-type hole arithmetic',
        '; '.join(bad[:3]))
    parts = (src.get('assigned', 0) + src.get('fill_proposed_only', 0)
             + src.get('unpaired_unproposed', 0))
    chk(parts == src.get('total_queried'),
        f'{tag}: source claim partition adds up',
        f'{parts} vs {src.get("total_queried")}')
    lv = table(run, 'gap_fill_levels.csv')
    by = cov.get('gap_fill_by_level') or {}
    if lv is not None and by:
        got = Counter(lv['level'])
        want = {k: v for k, v in by.items() if isinstance(v, int)}
        chk(all(int(got.get(k, 0)) == int(v) for k, v in want.items()),
            f'{tag}: gap_fill_by_level matches the CSV', f'{dict(got)} vs {want}')
    return cov


def check_pooling(run, tag):
    par = json.loads(path_of(run, 'parameters.json').read_text('utf-8'))
    chk(par.get('validation_mode') == 'pooling',
        f'{tag}: parameters name pooling', str(par.get('validation_mode')))
    chk(par.get('mode_rank') in (None, ''),
        f'{tag}: pooling has no mode_rank', str(par.get('mode_rank')))
    cand, pool = table(run, 'pooling_candidates.csv'), \
        table(run, 'pooling_pool.csv')
    xvp = path_of(run, 'pooling_cross_validation.json')
    chk(cand is not None and pool is not None and xvp.exists(),
        f'{tag}: pooling artifacts all exported')
    if cand is None or pool is None or not xvp.exists():
        return
    xv = json.loads(xvp.read_text('utf-8'))
    morph = xv.get('morph') or {}
    for k in ('attempted', 'scored', 'qualified', 'no_score', 'capped',
              'dropped_targets', 'gate_applied'):
        chk(k in morph, f'{tag}: morph ledger carries {k}', str(sorted(morph)))
    # `attempted` is what the budget LOOKED at; `capped` is what it refused to,
    # so the two are a split of `units`, never a sum inside `attempted` (the
    # first version of this check added them and reported 546 vs 400 on a run
    # whose ledger was correct)
    parts = sum(morph.get(k, 0) or 0 for k in ('scored', 'no_score'))
    chk(parts == morph.get('attempted'), f'{tag}: morph ledger adds up',
        f'{parts} vs {morph.get("attempted")}')
    if 'units' in morph:
        chk((morph.get('attempted') or 0) + (morph.get('capped') or 0)
            == morph['units'], f'{tag}: attempted + capped == units',
            f"{morph.get('attempted')}+{morph.get('capped')} vs "
            f"{morph['units']}")
    else:
        info(f'{tag}: run predates morph.units, so the cap cannot be '
             f'reconciled (capped={morph.get("capped")})')
    # A cap that bites must say what priced it: `capped: 146` on a run whose
    # budget came from a constant nobody wrote down is the same unreadable
    # number twice.  The auto rule (3 units per queried source) publishes its
    # own arithmetic, so `attempted + capped` is checkable against it.
    budget = str(morph.get('budget') or '')
    if (morph.get('capped') or 0) > 0 or 'units' in morph:
        chk(bool(budget), f'{tag}: morph budget states its provenance',
            f'budget={budget!r} capped={morph.get("capped")}')
    m = re.search(r'auto: (\d+) x (\d+)', budget)
    if m and 'units' in morph:
        chk(morph['units'] <= int(m.group(1)) * int(m.group(2)),
            f'{tag}: auto budget covers the units it was priced for',
            f'{budget} vs units={morph["units"]}')
    # `gate_applied` is a claim about the PASS (pipeline comment at
    # mapping_validation_pooling.py:947): a pass that raised, or one that
    # graded nothing, did not gate this pool — its tiers are then UNREFUSED
    # rather than passed, so demanding zero claim tiers (the old check)
    # failed every run that recorded itself honestly.  Assert both
    # implications instead.  `scored` alone is "graded": `shared` counts
    # ROWS (a borrowed-verdict row), `scored`/`no_score` count VERDICTS —
    # adding them once priced `graded` in units-of-mixed-kind and skipped
    # the all-shared pass the tripwire existed for.
    scored = int(morph.get('scored') or 0)
    if morph.get('gate_applied') is True:
        chk(scored > 0, f'{tag}: gate_applied means the pass graded '
            f'something', f'scored={scored} shared={morph.get("shared")}')
    if morph.get('gate_applied') is False:
        chk(not morph.get('dropped_targets'),
            f'{tag}: an ungated pass refuses nothing (tiers unrefused, '
            'not absent)',
            f'dropped_targets={morph.get("dropped_targets")}')
    if morph.get('error'):
        chk(morph.get('gate_applied') is not True,
            f'{tag}: a failed morph pass does not claim its gate applied',
            f'error={str(morph["error"])[:90]} gate_applied='
            f'{morph.get("gate_applied")}')
        bang = '\n'.join(
            l for l in path_of(run, 'README.txt').read_text(
                'utf-8', errors='replace').splitlines() if l.startswith('!'))
        chk('pooling' in bang and 'morph' in bang.lower(),
            f'{tag}: the failed pass is a ! line in README.txt',
            bang[:120] or 'no ! lines at all')
    # a refusal is per ROW; a target leaves the pool only when EVERY row that
    # admitted it was refused, because another source may hold its own verdict
    refused_rows = {int(r.target_bodyId) for r in cand.itertuples(index=False)
                    if r.morph_gate in ('scored', 'shared')
                    and not as_bool(r.morph_qualified or 'False')}
    admitted = {int(r.target_bodyId) for r in cand.itertuples(index=False)}
    kept = {int(r.target_bodyId) for r in cand.itertuples(index=False)
            if not (r.morph_gate in ('scored', 'shared')
                    and not as_bool(r.morph_qualified or 'False'))}
    dropped = admitted - kept
    chk(len(dropped) == morph.get('dropped_targets'),
        f'{tag}: dropped_targets counts the refused targets',
        f'{len(dropped)} vs {morph.get("dropped_targets")}')
    chk(refused_rows >= dropped,
        f'{tag}: every dropped target has a refused row')
    seen = {int(b) for b in pool['target_bodyId']}
    in_pool = {int(r.target_bodyId) for r in pool.itertuples(index=False)
               if as_bool(str(r.in_pool or 'True'))}
    chk(seen >= admitted,
        f'{tag}: every admitted target has a pool row',
        str(sorted(admitted - seen)[:5]))
    chk(in_pool == kept, f'{tag}: in_pool == admitted minus refused',
        f'{len(in_pool)} vs {len(kept)}')
    chk(pool['target_bodyId'].is_unique,
        f'{tag}: pool deduped to one row per target')
    # the supervised-only rows: the same file answers "which target does EITHER
    # engine claim", so they must be the JSON set, not a count beside a table
    vo = {int(r.target_bodyId) for r in pool.itertuples(index=False)
          if r.mapper_cell == 'verified_only'}
    chk(vo == set(int(b) for b in (xv.get('body_ids') or {}).get(
        'verified_only', [])),
        f'{tag}: verified_only rows == the JSON set',
        f'{len(vo)} vs {len(xv.get("body_ids", {}).get("verified_only", []))}')
    chk(not (vo & admitted),
        f'{tag}: a verified_only row was never admitted by the bar',
        str(sorted(vo & admitted)[:5]))
    # `pooling_pool.csv` is the complete target picture, so a refused target is
    # IN the file with in_pool=False; the rule is that no refused verdict is
    # still marked as part of the pool
    bad_gate = [str(r.target_bodyId) for r in pool.itertuples(index=False)
                if r.morph_gate == 'scored'
                and not as_bool(r.morph_qualified or 'False')
                and as_bool(str(r.in_pool or 'True'))]
    chk(not bad_gate, f'{tag}: no refused verdict survives in the pool',
        str(bad_gate[:5]))
    # the source axis: the mode's own unit, and its denominator
    src = table(run, 'pooling_sources.csv')
    chk(src is not None and len(src) == int(
        (xv.get('seed') or {}).get('queried_sources') or -1),
        f'{tag}: pooling_sources.csv is one row per queried source',
        f'{0 if src is None else len(src)} vs '
        f'{(xv.get("seed") or {}).get("queried_sources")}')
    if src is not None:
        chk(src['source_bodyId'].is_unique,
            f'{tag}: one pooling source row per bodyId')
        chk(set(src['tier']) <= {'matched', 'verified', 'nominated', ''},
            f'{tag}: source tiers use the pooling vocabulary',
            str(sorted(set(src['tier']))))
    chk(set(cand['tier']) <= {'matched', 'verified', 'nominated', ''},
        f'{tag}: candidate tiers use the pooling vocabulary',
        str(sorted(set(cand['tier']))))
    # a borrowed verdict must name the pair it came from, and its own row must
    # not: this is the difference between a measurement and a quotation
    bad_share = [f'{r.source_bodyId}->{r.target_bodyId}'
                 for r in cand.itertuples(index=False)
                 if (r.morph_gate == 'shared') != bool(r.verdict_for_pair)]
    chk(not bad_share, f'{tag}: every shared verdict names its pair',
        str(bad_share[:5]))
    scored_pairs = {(str(r.source_bodyId), str(r.target_bodyId))
                    for r in cand.itertuples(index=False)
                    if r.morph_gate == 'scored'}
    orphan = [r.verdict_for_pair for r in cand.itertuples(index=False)
              if r.verdict_for_pair
              and tuple(r.verdict_for_pair.split('->')) not in scored_pairs]
    chk(not orphan, f'{tag}: every borrowed verdict points at a scored pair',
        str(orphan[:5]))
    # The bar's own arithmetic, checked against the rows it produced.  These are
    # the four facts the bar can be wrong about without anything else noticing:
    # a `bar_rank` that is not the rank it claims, a source holding more rows
    # than its cap allows, a `supported_by` that disagrees with its flag, and a
    # pool row that represents a target with a verdict the bar refused (which is
    # what the 2026-09-25 BANC run exported before `pool_by_target` was fixed).
    bar = xv.get('bar') or {}
    if bar and {'bar_rank', 'jaccard_rank', 'rank_union_rank'} <= set(cand.columns):
        want = ('min' if str(bar.get('metric')) == 'either'
                else str(bar.get('metric')))
        bad_rank = []
        for r in cand.itertuples(index=False):
            jr, rr = int(float(r.jaccard_rank)), int(float(r.rank_union_rank))
            exp = min(jr, rr) if want == 'min' else (
                jr if want == 'jaccard' else rr)
            if int(float(r.bar_rank)) != exp:
                bad_rank.append(f'{r.source_bodyId}->{r.target_bodyId}: '
                                f'{r.bar_rank} != {exp}')
        chk(not bad_rank, f'{tag}: bar_rank is the rank the metric says',
            f'{len(bad_rank)} rows, e.g. {bad_rank[:3]}')
        cap = int(bar.get('row_cap_multiple') or 0) * int(bar.get('top_n') or 0)
        per_src = cand.groupby('source_bodyId').size()
        over = [f'{s}:{int(n)}' for s, n in per_src.items() if cap and n > cap]
        chk(not over, f'{tag}: no source holds more than the {cap}-row cap',
            str(over[:5]))
    if {'supported_by', 'single_metric_support'} <= set(cand.columns):
        bad_sup = [f'{r.source_bodyId}->{r.target_bodyId}' for r in
                   cand.itertuples(index=False)
                   if bool(str(r.supported_by or '').split('+')[0])
                   and (len([m for m in str(r.supported_by or '').split('+')
                             if m.strip()]) > 1)
                   == as_bool(r.single_metric_support or 'False')]
        chk(not bad_sup,
            f'{tag}: single_metric_support agrees with supported_by',
            f'{len(bad_sup)} rows, e.g. {bad_sup[:3]}')
    tiers_by_target = {}
    refused_by_target = {}
    for r in cand.to_dict('records'):
        t = int(r['target_bodyId'])
        if str(r.get('tier') or '').strip():
            tiers_by_target.setdefault(t, set()).add(str(r['tier']).strip())
        if (str(r.get('morph_gate') or '') in ('scored', 'shared')
                and r.get('morph_qualified') is False
                or str(r.get('morph_gate') or '') in ('scored', 'shared')
                and str(r.get('morph_qualified', '')).lower() in ('false', '0')):
            refused_by_target[t] = refused_by_target.get(t, 0) + 1
    bad_tiers = []
    for r in pool.to_dict('records'):
        t = int(r['target_bodyId'])
        if str(r.get('mapper_cell') or '') == 'verified_only':
            continue
        if set(str(r.get('tiers') or '').split('+')) - {''} != tiers_by_target.get(t, set()):
            bad_tiers.append(f'{t}: {r.get("tiers")} != '
                             f'{sorted(tiers_by_target.get(t, set()))}')
        if int(float(r.get('n_rows_refused') or 0)) != refused_by_target.get(t, 0):
            bad_tiers.append(f'{t}: n_rows_refused {r.get("n_rows_refused")} != '
                             f'{refused_by_target.get(t, 0)}')
    chk(not bad_tiers, f'{tag}: pool rows report their target\'s tier set and '
        'refusal count', f'{len(bad_tiers)}, e.g. {bad_tiers[:3]}')
    contrad = [str(r['target_bodyId']) for r in pool.to_dict('records')
               if str(r.get('in_pool') or '').strip().lower() in ('true', '1', 'yes')
               and str(r.get('morph_gate') or '') in ('scored', 'shared')
               and str(r.get('morph_qualified') or '').strip().lower()
               in ('false', '0')]
    chk(not contrad, f'{tag}: a pooled row never publishes a refused verdict',
        f'{len(contrad)} targets, e.g. {contrad[:3]}')
    # the floors are flags: a flagged row is still in the file
    for col in ('below_jaccard_floor', 'below_rank_union_floor',
                'outside_window'):
        chk(col in cand.columns, f'{tag}: candidates carry the {col} flag')
    flagged = int(pd.to_numeric(
        cand['below_rank_union_floor'].map(as_bool), errors='coerce').fillna(0)
        .sum()) if 'below_rank_union_floor' in cand.columns else 0
    chk(xv.get('floor_flags', {}).get('below_rank_union_floor') == flagged,
        f'{tag}: floor_flags counts the rows, it removes none',
        f'{xv.get("floor_flags", {}).get("below_rank_union_floor")} vs '
        f'{flagged}')
    # no neuron is ever labelled with the placeholder a float leaves behind
    labelled = [str(v) for v in list(cand.get('target_type', []))
                + list(pool.get('target_type', []))
                if str(v).strip() in ('nan', 'NaN', 'NA')]
    chk(not labelled, f'{tag}: no exported type is the string nan',
        str(labelled[:5]))
    pairs = {(r.target_bodyId, r.source_bodyId)
             for r in cand.itertuples(index=False)}
    chk({(r.target_bodyId, r.best_source_bodyId)
         for r in pool.itertuples(index=False)} <= pairs,
        f'{tag}: each pool row names a scanned pair')
    # the verdict has to be recomputable from the row it appears on: the score
    # the bar was applied to depends on the kind, and a published score below a
    # published bar with a ✓ beside it is the bug this check exists for
    score_column_by_kind = {
        'native': 'morph_pool_ref', 'track_a': 'morph_similarity',
        'track_a_backup': 'morph_similarity', 'null_bar': 'morph_similarity',
        'null': 'morph_similarity', '': 'morph_similarity'}
    contradict = []
    for r in cand.itertuples(index=False):
        if r.morph_gate != 'scored':
            continue
        col = score_column_by_kind.get(str(r.morph_bar_kind or ''),
                                       'morph_similarity')
        score = float(getattr(r, col) or 'nan')
        bar = float(r.morph_bar or 'nan')
        if score != score or bar != bar:
            contradict.append(f'{r.target_bodyId}: no {col}/bar')
        elif (score >= bar) != as_bool(r.morph_qualified or 'False'):
            contradict.append(f'{r.target_bodyId}: {score} vs {bar} '
                              f'{r.morph_qualified}')
    chk(not contradict,
        f'{tag}: every scored verdict is recomputable from its own row',
        f'{len(contradict)} rows, e.g. {contradict[:3]}')
    for col in ('jaccard', 'rank_union'):
        vals = pd.to_numeric(cand[col], errors='coerce')
        chk(vals.notna().all(), f'{tag}: {col} parses',
            str(int(vals.isna().sum())))
    gate = xv.get('gate') or {}
    # which contract the run was written under is the run's own fact: a run that
    # publishes `bar` admitted by the bar, and its floors must describe
    # themselves as advisory; an older run gated on the floors and said so.
    # Reading the published schema is not a compatibility shim - it is what the
    # file says about itself.
    if xv.get('bar'):
        chk(gate.get('jaccard_floor') is not None
            and 'advisory' in str(gate.get('role', '')).lower(),
            f'{tag}: a bar-driven run calls its floors advisory', str(gate))
        below = pd.to_numeric(cand['jaccard'], errors='coerce') <= float(
            gate.get('jaccard_floor') or 0)
        flagged = cand['below_jaccard_floor'].map(as_bool)
        chk(bool(below.sum()) == bool(flagged.sum())
            and below.fillna(False).astype(bool).equals(
                flagged.astype(bool)),
            f'{tag}: below_jaccard_floor agrees with the value it names',
            f'{int(below.sum())} below vs {int(flagged.sum())} flagged')
        # the point of the change: a flagged row is still a row
        chk(len(cand) >= int(flagged.sum()),
            f'{tag}: flagged rows are kept, not filtered',
            f'{int(flagged.sum())} of {len(cand)}')
    else:
        chk(gate.get('jaccard_floor') is not None
            and 'volume' in str(gate.get('role', '')).lower(),
            f'{tag}: floor-gated run published its guard-rail', str(gate))
        low = pd.to_numeric(cand['jaccard'], errors='coerce') < float(
            gate.get('jaccard_floor') or 0)
        chk(not bool(low.sum()),
            f'{tag}: every candidate clears the floor it names',
            f'{int(low.sum())} rows below {gate.get("jaccard_floor")}')
    vc = morph.get('vector_cache') or {}
    info(f'{tag} target-vector store', str(vc))
    chk(bool(vc), f'{tag}: the store ledger is published', str(morph)[:120])
    chk('dropped_targets' in ' '.join(xv.get('reading_notes') or []),
        f'{tag}: reading notes explain the gate')
    chk(bool(xv.get('input_fingerprint')),
        f'{tag}: input_fingerprint published', str(xv)[:80])
    # The native score is read out of the TARGET's V2 vector cache, which the
    # profile caches above do not cover, so a fingerprint that stops at them
    # cannot explain a `morph_pool_ref` that moved between two runs of one
    # query — measured 2026-09-25 on FAFB->BANC as 107 of 1266 rows.
    fp = xv.get('input_fingerprint') or {}
    stores = fp.get('morph_stores') or {}
    if stores and 'target' not in stores and 'source' not in stores:
        # keyed by pass (supervised / pooling, 2026-09-25 WIP-D): both keys
        # name the same run-baseline store, so any one of them answers
        jobs = stores
        stores = next(iter(jobs.values()), {}) or {}
    if stores:
        tgt = stores.get('target') or {}
        # Only a pass that actually graded native verdicts owes a target
        # vector-store identity: a Track-A-only pass can publish a target
        # store entry with no counts (falsy values are dropped by the
        # identity writer), and demanding the counts there would fail a
        # healthy run.
        native_rows = sum(
            1 for r in cand.itertuples(index=False)
            if str(r.morph_bar_kind or '') == 'native')
        if native_rows:
            chk(bool((tgt.get('vector_cache') or {}).get('bytes')),
                f'{tag}: the vector cache a native verdict came out of is named',
                str(tgt)[:120])
            chk(bool(tgt.get('skeleton_files')),
                f'{tag}: the skeleton store the pass fetched into is counted',
                str(tgt)[:120])
        else:
            info(f'{tag}: no native verdicts — the target vector store is '
                 'not part of this pass\'s explanation')
    else:
        info(f'{tag}: fingerprint names no morph_stores (a run before '
             '2026-09-25) — a moved morph_pool_ref is unattributable here')


# ---------------------------------------------------------------- ladder
def _rows(run, fname):
    df = table(run, fname)
    return [] if df is None else list(df.itertuples(index=False))


def _feed(run):
    return {(r.source_bodyId, r.ahead_target_bodyId, r.candidate_source)
            for r in _rows(run, 'deep_candidates.csv')}


def _tight(run):
    return {(s, t) for s, t, src in _feed(run) if src == 'top_window'}


def _cats(run):
    return {(r.source_bodyId, r.ahead_target_bodyId): r.category
            for r in _rows(run, 'examinees.csv')}


def check_ladder(runs, label):
    """runs: {mode: run-dir} with all three ladder modes present."""
    r, f, a = (runs['restrictive'], runs['family'], runs['aggressive'])

    def tierkeys(run):
        return {(x.source_bodyId, x.target_bodyId, x.verdict)
                for x in _rows(run, 'validation_results.csv')}
    ks = [tierkeys(runs[m]) for m in LADDER]
    chk(ks[0] == ks[1] == ks[2], f'{label}: tier identical across the ladder',
        f'{[len(x) for x in ks]}')

    # the backdrop bar is what every null-kind admission is measured against,
    # so one dataset pair must yield ONE sample. It did not: the sample
    # excluded the mode's own candidate window, and rows sitting inside the
    # resulting drift changed category as the mode widened
    bars = {}
    for m in LADDER:
        cal = json.loads(path_of(runs[m], 'morphology_calibration.json')
                         .read_text('utf-8'))
        ps = table(runs[m], 'pair_summary.csv')
        bars[m] = (cal.get('track_a_null_bar'),
                   int(pd.to_numeric(ps['null_sample'], errors='coerce')
                       .fillna(0).sum()))
    chk(len({v[0] for v in bars.values()}) == 1,
        f'{label}: one null backdrop across the ladder', str(bars))

    sa, sb, sc = (table(runs[m], 'pair_summary.csv') for m in LADDER)
    keep = [c for c in sa.columns if c not in MODE_COLUMNS]
    chk(sa[keep].equals(sb[keep]) and sb[keep].equals(sc[keep]),
        f'{label}: pair_summary invariant outside the mode-counting columns',
        f'{len(keep)} columns compared')

    cat = {m: _cats(runs[m]) for m in LADDER}
    drift = [(narrow, k, cat[narrow][k], cat[wide][k])
             for narrow, wide in zip(LADDER, LADDER[1:])
             for k in cat[narrow].keys() & cat[wide].keys()
             if cat[narrow][k] != cat[wide][k]]
    chk(not drift, f'{label}: no row changes category as the mode widens',
        str(drift[:3]))
    cands = {m: {k for k, v in cat[m].items() if v == 'candidates'}
             for m in LADDER}
    chk(cands['restrictive'] <= cands['family'] <= cands['aggressive'],
        f'{label}: candidates bin nests R<=F<=A',
        f'{[len(cands[m]) for m in LADDER]}')
    sib = {m: {k for k, v in cat[m].items() if v == 'sibling'}
           for m in LADDER}
    chk(sib['restrictive'] <= sib['family'] <= sib['aggressive'],
        f'{label}: sibling bin nests R<=F<=A',
        f'{[len(sib[m]) for m in LADDER]}')

    # I-3 correction (2026-09-28): the window cut is chain-ordered
    # (order_by_chain + per-band budget), so a displaced-row difference on
    # same-code runs is a SCAN-INPUT drift signal, not a cut bug — measured
    # on the 09-28 matrix: the mapper snapshot grew and one target neuron
    # joined the universe between two same-day cells (102144 -> 102145),
    # which moved four tie-block members and one APDN3 row.  Read each
    # run's input_fingerprint first and downgrade the window-identity
    # checks to advisory when the store moved under the runs.
    def _fingerprint(run):
        p = path_of(run, 'parameters.json')
        try:
            return json.loads(p.read_text('utf-8')).get(
                'input_fingerprint') or {}
        except Exception:
            return {}

    fps = {m: _fingerprint(runs[m]) for m in LADDER}
    store_keys = ('scanned_target_universe', 'target_vectors_built',
                  'profile_cache_source', 'profile_cache_target',
                  'mapper_snapshot')
    drift = [f'{LADDER[0]}->{m}: {k}'
             for m in LADDER[1:]
             for k in store_keys
             if (k in fps[m] or k in fps[LADDER[0]])
             and fps[m].get(k) != fps[LADDER[0]].get(k)]
    store_drift = bool(drift)
    if store_drift:
        uni = {m: fps[m].get('scanned_target_universe') for m in LADDER}
        info(f'{label}: scan-input drift across the ladder',
             ' ; '.join(drift)
             + f' (target universes {uni}) — the store moved under the '
             'runs, so the window-identity check below is advisory')

    # the narrow mode's window rows must survive the wider mode: the two bands
    # used to share one per-source budget, so aggressive spent it on its deep
    # band and never read the second metric
    for narrow, wide in zip(LADDER, LADDER[1:]):
        lost = _tight(runs[narrow]) - _tight(runs[wide])
        msg = (f"{label}: {wide} keeps {narrow}'s window rows",
               f'{len(_tight(runs[narrow]))} vs {len(_tight(runs[wide]))}, '
               f'{len(lost)} displaced (e.g. {sorted(lost)[:2]})')
        if store_drift:
            info(msg[0] + ' (advisory: scan-input drift)', msg[1])
        else:
            chk(not lost, *msg)
    for nm, fname in (('family', 'family_candidates.csv'),
                      ('relative', 'relatives.csv')):
        tg = {m: {r.ahead_target_bodyId for r in _rows(runs[m], fname)}
              for m in LADDER}
        chk(tg['restrictive'] == set() and tg['family'] <= tg['aggressive'],
            f'{label}: {nm} bin empty in restrictive and nests F<=A',
            f'{len(tg["restrictive"])}/{len(tg["family"])}/{len(tg["aggressive"])}'
            f' targets lost: {sorted(tg["family"] - tg["aggressive"])[:4]}')
    deep = {m: _rows(runs[m], 'deep_candidates.csv') for m in LADDER}
    chk(len(deep['restrictive']) == 0,
        f'{label}: restrictive reads no candidate window',
        str(len(deep['restrictive'])))
    kinds = {m: Counter(r.category for r in deep[m]).get('examinees', 0)
             for m in LADDER}
    chk(kinds['family'] == 0 and kinds['aggressive'] > 0,
        f'{label}: the deep-window bin is aggressive-only', str(kinds))

    # the deliverable: a wider mode may close holes, never reopen them
    cov = {m: check_coverage(runs[m], f'{label}/{m}') for m in LADDER}
    holes = [int(cov[m]['target'].get('holes') or 0) for m in LADDER]
    only = [int(cov[m]['target'].get('reached_as_candidates_only') or 0)
            for m in LADDER]
    info(f'{label} coverage along the ladder', f'holes {holes} · only {only}')
    chk(all(holes[i + 1] <= holes[i] for i in range(2))
        and all(only[i + 1] >= only[i] for i in range(2)),
        f'{label}: holes only close along the ladder',
        f'holes {holes}, reached-only {only}')
    for m in LADDER:
        dedup, lv = (table(runs[m], 'gap_fill_dedup.csv'),
                     table(runs[m], 'gap_fill_levels.csv'))
        if dedup is None or lv is None or not len(dedup):
            continue
        chk(dedup['target_bodyId'].is_unique,
            f'{label}/{m}: gap_fill_dedup is one row per bodyId')
        chk(not set(lv['target_bodyId']) - set(dedup['target_bodyId']),
            f'{label}/{m}: every level row is a dedup row',
            str(sorted(set(lv.target_bodyId) - set(dedup.target_bodyId))[:5]))


def check_placeholder_types(run, tag):
    """No identity column of any exported CSV may hold a float placeholder.

    `str(nan)` is the non-empty string 'nan', so a `.get(bid, '?')` on a dict
    built from a pandas column hands an unannotated neuron a TYPE NAMED "nan" —
    which then reads as a real type in a leaf token or a hover.  Measured on the
    2026-09-24 male-cns run: 178 rows of `noise_filtered_candidates.csv` and one
    `gap_fill_proposals.csv` row did exactly that, after the same hole had been
    closed on the pooling side; a check that only looks at the pooling files
    cannot see the supervised ones.
    """
    hits = []
    for path in sorted(run.rglob('*.csv')):
        rel = path.relative_to(run).as_posix()
        if rel.startswith('visualization/'):
            continue
        try:
            # keep_default_na=False is the whole point: pandas reads the
            # literal string "nan" as a missing value by default, so a naive
            # .fillna('') erases exactly the evidence this check is looking for
            # (it passed the run that had 180 of them).
            df = pd.read_csv(path, dtype=str, keep_default_na=False,
                             na_filter=False)
        except Exception:  # noqa: BLE001
            continue
        for col in df.columns:
            if not any(k in col.lower()
                       for k in ('type', 'leaf', 'annotation', 'category')):
                continue
            n = int(df[col].isin(['nan', 'NaN', 'NA', 'None', 'none']).sum())
            if n:
                hits.append(f'{rel}:{col}={n}')
    chk(not hits, f'{tag}: no float placeholder in an identity column',
        str(hits[:6]))


def check_partition(runs, label):
    """Rev 3.12 hygiene on every binned row of every mode."""
    for m, run in runs.items():
        for fname in ('examinees.csv', 'family_candidates.csv',
                      'relatives.csv', 'deep_candidates.csv'):
            df = table(run, fname)
            if df is None or not len(df):
                continue
            bad = sorted(set(df['category']) - CATEGORIES)
            chk(not bad, f'{label}/{m}/{fname}: category vocabulary',
                str(bad[:4]))
            both = sum(1 for r in df.itertuples(index=False)
                       if as_bool(r.in_scope) and as_bool(r.morph_failed))
            chk(not both, f'{label}/{m}/{fname}: in_scope never morph_failed',
                str(both))
            fill_bad = []
            tok_bad = []
            for r in df.itertuples(index=False):
                if (as_bool(getattr(r, 'counts_toward_restrictive_fill', ''))
                        and r.category != 'candidates'):
                    fill_bad.append((r.category, 'restrictive'))
                if (as_bool(getattr(r, 'counts_toward_family_fill', ''))
                        and r.category not in ('candidates', 'family',
                                               'relative')):
                    fill_bad.append((r.category, 'family'))
                ann = str(r.candidate_annotation)
                hits = [t for t in TOKENS if t in ann] + (
                    ['>src'] if '>' in ann and '(out-map)' not in ann
                    and not ann.endswith('untyped') else [])
                if len(hits) > 1 or (r.category == 'family'
                                      and '(out-map)' not in ann):
                    tok_bad.append(ann)
            chk(not fill_bad, f'{label}/{m}/{fname}: fill counters follow the '
                'category', str(fill_bad[:3]))
            chk(not tok_bad, f'{label}/{m}/{fname}: one leaf token per row',
                str(tok_bad[:3]))


def check_summary_rows(run, tag):
    """pair_summary verdict counters must equal the row distribution (TMV-3).

    Stage 5's AUC demotion rewrote row verdicts after `_summary` froze its
    counters, so the Branches tab and the Validation tab of one report
    disagreed (matrix-measured 2026-09-25: banc l-LNv -> l-LNv summary
    strong=2/verified=6 against 8/8 plain `verified` rows).  The resync
    landed with this check; a run that still shows drift carries the bug
    (pre-fix runs re-audited with this script fail it — that is the point).
    """
    ps = table(run, 'pair_summary.csv')
    vr = table(run, 'validation_results.csv')
    if ps is None or vr is None or not len(ps) or not len(vr):
        return
    from collections import Counter
    counts = {}
    for r in vr.itertuples(index=False):
        counts.setdefault(
            (r.query, r.source_type, r.target_type), Counter())[r.verdict] += 1
    bad = []
    for s in ps.itertuples(index=False):
        key = (s.query, s.source_type, s.target_type)
        got = counts.get(key)
        if got is None:
            continue
        for verdict in ('verified_strong', 'verified', 'borderline',
                        'unmatched', 'skipped'):
            want = int(getattr(s, f'verdict_{verdict}', 0) or 0)
            have = got.get(verdict, 0)
            if want != have:
                bad.append((key, verdict, want, have))
    chk(not bad, f'{tag}: pair_summary verdicts equal the row distribution',
        str(bad[:3]))


def check_verdict_bars(runs, label):
    """A published morph verdict must be recomputable from its own row.

    #58 gave pooling's rows their bar; #61 gave the two remaining exports the
    same record. Before that, `out_map_expansion.csv` and
    `source_candidates.csv` printed ✓/✗ while the threshold lived only in
    `morphology_calibration.json` — a reader could not check a single mark.
    An export older than the columns is named, not failed: the file is what it
    is, and the run predates the contract.
    """
    for m, run in runs.items():
        for fname in ('out_map_expansion.csv', 'source_candidates.csv'):
            df = table(run, fname)
            if df is None or not len(df):
                continue
            if 'morph_bar' not in df.columns:
                info(f'{label}/{m}/{fname}: predates the bar columns '
                     f'(#61) — its ✓/✗ cannot be checked from the row')
                continue
            verdicts = df[df.morph_qualified.astype(str).str.len() > 0]
            missing = int((verdicts.morph_bar.astype(str) == '').sum())
            chk(missing == 0, f'{label}/{m}/{fname}: every verdict carries '
                f'its bar', f'{missing} of {len(verdicts)}')
            wrong = []
            for r in verdicts.itertuples(index=False):
                try:
                    got = float(r.morph_v2_similarity) >= float(r.morph_bar)
                except (TypeError, ValueError):
                    continue
                if got != (str(r.morph_qualified).lower() == 'true'):
                    wrong.append((r.target_bodyId, r.morph_v2_similarity,
                                  r.morph_bar, r.morph_qualified))
            chk(not wrong, f'{label}/{m}/{fname}: the verdict recomputes from '
                'score >= bar', str(wrong[:2]))


def cost_table(entries):
    print('\n== cost (pipeline_progress.jsonl, seconds) ==')
    rows, order, labels = {}, [], {}
    for run, _tgt, mode, _exit, _wall in entries:
        p = path_of(run, 'pipeline_progress.jsonl')
        if not p.exists():
            continue
        spans = {}
        for line in p.read_text().splitlines():
            try:
                ev = json.loads(line)
            except Exception:                               # noqa: BLE001
                continue
            if ev.get('event') not in ('stage_start', 'stage_done'):
                continue
            k = str(ev.get('stage'))
            if ev.get('label'):
                labels[k] = ev['label']
            spans.setdefault(k, []).append(ev['ts'])
        cells = {}
        for k, v in spans.items():
            if len(v) >= 2:
                cells[k] = cells.get(k, 0) + int(
                    (datetime.fromisoformat(v[-1])
                     - datetime.fromisoformat(v[0])).total_seconds())
        par = json.loads(path_of(run, 'parameters.json').read_text('utf-8'))
        rows[(str(par.get('target_dataset'))[:10],
              str(mode or par.get('validation_mode')))] = cells
        for k in cells:
            if k not in order:
                order.append(k)
    if not rows:
        return
    print(f'{"run":22}' + ''.join(
        f'{(k + ":" + str(labels.get(k, "")))[:15]:>16}' for k in order)
        + f'{"TOTAL":>9}')
    for key, cells in sorted(rows.items()):
        print(f'{key[0]}/{key[1]:10} ' + ''.join(
            f'{cells.get(k, ""):>16}' for k in order)
            + f'{sum(cells.values()):>9}')


def main(argv=None):
    paths = argv or sys.argv[1:]
    if not paths:
        queues = sorted((REPO / 'local_data').glob('tmvev-*'),
                        key=lambda p: p.stat().st_mtime)
        paths = [str(queues[-1])] if queues else []
        if paths:
            print(f'# no PATH given: newest queue dir {paths[0]}')
    entries = discover(paths)
    if not entries:
        print('nothing to inspect', file=sys.stderr)
        return 2
    groups = {}
    for ent in entries:
        run = ent[0]
        par = json.loads(path_of(run, 'parameters.json').read_text('utf-8'))
        mode = ent[2] or par.get('validation_mode')
        tag = f'{ent[1] or par.get("target_dataset")}/{mode}'
        # A bare run folder has no manifest row, so its exit code was never
        # recorded — that is nothing to assert, not a failure (`str(None)` is
        # how this read as a red `exit 0 [None]` on a single-folder run).
        chk(ent[3] is None or str(ent[3]) == '0', f'{tag}: exit 0',
            'not in a manifest' if ent[3] is None else str(ent[3]))
        check_layout(run, tag)
        check_report(run, tag, mode)
        check_coverage(run, tag)
        check_summary_rows(run, tag)
        check_placeholder_types(run, tag)
        if mode == 'pooling':
            check_pooling(run, tag)
        grp = groups.setdefault(group_key(run), {})
        grp.setdefault('_entries', []).append(ent)
        grp[mode] = run
    for key, runs in groups.items():
        src, tgt, queries = key
        label = f'{src}->{tgt} [{",".join(queries)}]'
        per_mode = {m: r for m, r in runs.items() if m != '_entries'}
        check_partition(per_mode, label)
        check_verdict_bars(per_mode, label)
        if all(m in per_mode for m in LADDER):
            check_ladder(per_mode, label)
        else:
            info(f'{label}: no ladder', f'modes present {sorted(per_mode)}')
    cost_table(entries)
    print('\n== notes ==')
    if _INCOMPLETE:
        print(f'  · {len(_INCOMPLETE)} folder(s) skipped, no parameters.json '
              f'yet (run in flight): '
              + ', '.join(f.name[-17:] for f in _INCOMPLETE))
    for line in _infos:
        print('  ·', line)
    print(f'\n== {len(_fails)} FAIL ==')
    for f in _fails:
        print('  !', f)
    return 1 if _fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
