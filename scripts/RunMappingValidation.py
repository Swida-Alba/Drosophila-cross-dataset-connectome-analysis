#!/usr/bin/env python
"""Run the type-mapping validation pipeline (source -> target).

Plan: _plan/plan-type-mapping-validation-pipeline.md (design locked
2026-09-11).

Examples
--------
    # validate two FAFB types against MCNS (test pair)
    python scripts/RunMappingValidation.py \
        --source flywire_FAFB_v783 --target male-cns:v1.0 \
        --types APDN3,s-CPDN3A --label APDN3_sCPDN3A_test

    # full coarse-category run
    python scripts/RunMappingValidation.py \
        --source flywire_FAFB_v783 --target male-cns:v1.0 \
        --types circadian_clock --label circadian_full

    # review the proposed fill with the advisory reciprocal evidence
    python scripts/RunMappingValidation.py \
        --source flywire_FAFB_v783 --target male-cns:v1.0 \
        --types s-CPDN3C,s-CPDN3D --mode family --backward-evidence

Results land in {--output-dir}/type-map-validation_{SRC}_to_{TGT}_{ts}/
(short dataset nicknames, YYYYMMDD_HHMMSS stamp; --label is recorded in
parameters.json and README.txt, not in the folder name). The evidence
CSVs are grouped into validation/, expansion/, gap_fill/ and mapping/;
the root keeps report.html, the run guide and the parameter/meta files.
"""

import argparse
import json
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / 'src'))

from comparison.mapping_validation import (  # noqa: E402
    MappingValidationConfig,
    MappingValidator,
)


def _json_object(raw, flag):
    """Parse a ``{...}`` CLI value into a dict, naming the flag on failure.

    The two scene-styling flags carry the same dicts the UI's Advanced
    Visualization panel sends, so the CLI stays a complete surface for a run
    someone reproduces from `parameters.json`.
    """
    if raw is None or not str(raw).strip():
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f'{flag}: invalid JSON ({exc})')
    if not isinstance(value, dict):
        raise SystemExit(f'{flag}: expected a JSON object, got '
                         f'{type(value).__name__}')
    return {str(k): v for k, v in value.items()}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Type-mapping validation pipeline '
                    '(bodyId-level, connectivity + morphology)')
    p.add_argument('--source', default='flywire_FAFB_v783',
                   help='source dataset (default: flywire_FAFB_v783)')
    p.add_argument('--target', default='male-cns:v1.0',
                   help='target dataset (default: male-cns:v1.0)')
    p.add_argument('--types', required=True,
                   help='comma-separated source types or coarse '
                        'categories (e.g. APDN3,s-CPDN3A or circadian_clock)')
    p.add_argument('--label', default='run',
                   help='run label recorded in parameters.json and the '
                        'report (the folder name keeps prefix + datasets '
                        '+ timestamp)')
    p.add_argument('--output-dir', default=None,
                   help='base output dir (default: local_data/'
                        'mapping_validation)')
    p.add_argument('--rank-top-k', type=int, default=5,
                   help='borderline window k (default: 5)')
    p.add_argument('--gap-min', type=int, default=1,
                   help='gap trigger: gap > gap_min fires fill review '
                        '(Revision 3.4; default 1)')
    p.add_argument('--verified-top-n', type=int, default=2,
                   help='continuous ordered top-N verified path')
    p.add_argument('--invader-borderline-max', type=int, default=3,
                   help='max invading neurons ahead for borderline')
    p.add_argument('--matched-ru-min', type=float, default=0.1,
                   help='min rank_union for the matched tier')
    p.add_argument('--candidate-morph-factor', type=float, default=0.25,
                   help='candidates: morph >= factor x avg pool morph '
                        '(Revision 3.5 Issue 3c recalibration; default '
                        '0.25)')
    p.add_argument('--target-min-weight', type=float, default=10.0,
                   help='target-side profile-quality gate: scan '
                        'candidates below this total expanded weight '
                        'are excluded (Revision 3.5 Issue 5; default 10)')
    p.add_argument('--target-min-partner-types', type=int, default=2,
                   help='target-side profile-quality gate: minimum '
                        'partner-type count (Revision 3.5 Issue 5; '
                        'default 2)')
    p.add_argument('--suspicious-jaccard-factor', type=float, default=0.5,
                   help='jaccard sanity: a rank_union-ahead row whose '
                        'jaccard < factor x best pool jaccard is filtered '
                        'as noise (Revision 3.5 Issue 5b; default 0.5)')
    p.add_argument('--target-min-size-ratio', type=float, default=0.1,
                   help='PRIMARY noise filter (Rev 3.6): examinee rows '
                        'whose spatial size < ratio x the branch pool '
                        'best are dropped (default 0.1)')
    p.add_argument('--suspicious-ru-margin', type=float, default=0.02,
                   help='tie-margin rule (Rev 3.6): a rank_union-ahead '
                        'row with margin below this over the pool best '
                        'is a numerical tie (default 0.02)')
    p.add_argument('--pool-ref-cap', type=int, default=6,
                   help='Track-B native reference neurons per invader '
                        '(Rev 3.7; default 6)')
    p.add_argument('--pool-ref-floor-margin', type=float, default=0.05,
                   help='Track-B floor = native baseline - margin '
                        '(Rev 3.7; default 0.05)')
    p.add_argument('--morph-track-a-offset', type=float, default=0.05,
                   help='Floors v3: Track-A admission offset Δ — candidate '
                        'bar = pool Track-A baseline - Δ, suspicious bar = '
                        'baseline - k*Δ (default 0.05)')
    p.add_argument('--morph-suspicious-level', type=int, default=3,
                   help='Floors v3: suspicious level k (default 3)')
    p.add_argument('--out-map-top-k', type=int, default=10,
                   help='Plan I: out-map expansion keeps the top-k '
                        'connectivity-ranked candidates per unpaired source '
                        '(default 10)')
    p.add_argument('--skip-out-map-expansion', action='store_true',
                   help='Plan I: skip the out-map expansion stage entirely')
    p.add_argument('--skip-profile-build', action='store_true',
                   help='Plan I: skip the stage-2 target profile pre-flight '
                        '(stay cache-only; fail-closed on a thin cache)')
    p.add_argument('--mode', choices=('restrictive', 'family', 'aggressive',
                                      'pooling'), default='restrictive',
                   help='validation mode: restrictive (default) = tier + '
                        'sibling + candidates; family = adds the '
                        'family/relative bins; aggressive = adds the '
                        'deep-window examinees bin. Those three NEST. '
                        'pooling is a PARALLEL, unsupervised engine: it scans '
                        'the whole queried population against the whole '
                        'target universe, keeping each source\'s top-N rows '
                        'under its own admission bar (the Jaccard/rank-union '
                        'floors only FLAG, never filter), and joins the '
                        'type mapper afterwards as a post-hoc comparison '
                        '(plan-tmvev-pooling-mode.md).')
    p.add_argument('--aggressive-expansion', action='store_true',
                   help='alias for --mode aggressive (Rev 3.12; retained '
                        'for compatibility).')
    p.add_argument('--candidate-window', type=int, default=25,
                   help='deep-window candidate search: out-of-pool '
                        'neurons within this per-metric rank are '
                        'considered (Rev 3.8; default 25)')
    p.add_argument('--deep-cap', type=int, default=10,
                   help='max candidate-window rows kept per source neuron '
                        'and PER BAND — the borderline and deep bands each '
                        'draw on a budget of this size, so a wider mode only '
                        'adds rows (Rev 3.8; default 10)')
    p.add_argument('--pooling-bar-metric', default='either',
                   choices=['either', 'jaccard', 'rank_union'],
                   help='--mode pooling: WHICH metric\'s rank admits a '
                        'candidate. `either` is the union of each metric\'s own '
                        'top-N (the same reading the matched tier uses), not a '
                        'merged best-rank ordering — measured, a merged '
                        'ordering keeps 79 of the 118 targets the union keeps '
                        '116 of (plan-tmvev-pooling-tiers.md §2).')
    p.add_argument('--pooling-bar-top-n', type=int, default=3,
                   help='--mode pooling: how deep each metric\'s rank list goes. '
                        'The default admits ~1.1 targets per queried source on '
                        'FAFB->male-cns (272 for 242) and keeps 116 of the 118 '
                        'targets the old floors found; N>=5 on rank_union opens '
                        'a tie mass (17 rows/source), which the 2N row cap names.')
    p.add_argument('--pooling-jaccard-floor', type=float, default=0.10,
                   help='--mode pooling: ADVISORY flag, not a filter — each '
                        'floor is evaluated against this number and published '
                        'per row (below_jaccard_floor), and removes nothing. '
                        'They were filters until 2026-09-24, when the '
                        'rank_union one alone was found to starve 119 of 242 '
                        'queried sources.')
    p.add_argument('--pooling-rank-union-floor', type=float, default=0.0,
                   help='--mode pooling: advisory rank_union flag (default 0 = '
                        'strictly positive), published per row, never applied')
    p.add_argument('--pooling-window-mult', type=float, default=2.0,
                   help='--mode pooling: the window the outside_window flag '
                        'measures against = this x the size of the source '
                        "neuron's own queried type population. Advisory.")
    p.add_argument('--pooling-max-morph-targets', type=int, default=0,
                   help='--mode pooling: budget for the morph pass, counted in '
                        'SCORING UNITS (one network-bound step each). 0 (default) '
                        'means auto: 3 x the number of queried source neurons, so '
                        'the budget scales with the work asked for. Rows past the '
                        "budget are labelled morph_gate='not-attempted-cap', never "
                        'blank, and `morph.capped` says how many.')
    p.add_argument('--backward-evidence', action='store_true', default=True,
                   help='stage 5d: reverse (target -> source) scans label '
                        'the expansion bins — ADVISORY only, never gates '
                        'nor relabels a row; connectivity only, no '
                        'morphology re-scored (DEFAULT ON since '
                        '2026-09-26; kept for compatibility — passing it '
                        'changes nothing)')
    p.add_argument('--no-backward-evidence', dest='backward_evidence',
                   action='store_false',
                   help='stage 5d opt-out: skip the reciprocal pass (the '
                        'Reciprocal tab then renders its not-run note)')
    p.add_argument('--skip-backward-pass', action='store_true',
                   help='stage 5d: force-skip the backward pass even when '
                        'enabled')
    p.add_argument('--backward-top-n', type=int, default=5,
                   help='stage 5d: reverse hits kept per neuron (the '
                        'report hover label; default 5)')
    p.add_argument('--backward-max-neurons', type=int, default=300,
                   help='stage 5d: hard budget on dataset-scale reverse '
                        'scans per run (default 300)')
    p.add_argument('--backward-per-branch-cap', type=int, default=40,
                   help='stage 5d: max expansion members labeled per '
                        'branch (default 40)')
    p.add_argument('--no-backward-pool-targets', action='store_true',
                   help='stage 5d: do not reverse-scan the unmatched pool '
                        'targets (matched / verified / borderline pool '
                        'members are never scanned — the symmetric '
                        'forward score is their evidence); the source-side '
                        'columns then see fewer out-of-branch competitors')
    p.add_argument('--scene-selfcheck', action='store_true',
                   help='debug: verify each legend leaf geometry against '
                        'its neuron bbox after rendering (Revision 3.5 '
                        'Issue 6c)')
    p.add_argument('--no-morphology', action='store_true',
                   help='skip stage 5 (morphology verification)')
    p.add_argument('--morph-auc-floor', type=float, default=0.65)
    p.add_argument('--skeleton-fetch-workers', type=int, default=8,
                   help='threads the stage-5 skeleton pre-flight fetches '
                        'with (default 8; this is the run\'s only '
                        'network-bound block)')
    p.add_argument('--skeleton-fetch-timeout', type=int, default=120,
                   help='seconds of socket inactivity a skeleton request '
                        'may idle before it is counted as failed '
                        '(0 = no bound)')
    p.add_argument('--no-visualize', action='store_true',
                   help='skip stage 4 (pair scenes)')
    p.add_argument('--max-scenes', type=int, default=0,
                   help='cap on rendered parent scenes, largest pool first '
                        '(default 0 = one scene per parent type; a positive '
                        'value names every dropped parent in the run log)')
    p.add_argument('--neuron-alpha', type=float, default=0.2,
                   help='global neuron opacity in scenes (default: 0.2)')
    p.add_argument('--scene-viz-json', default=None, metavar='JSON',
                   help='scene styling as a JSON object of VisualizeSkeleton '
                        'kwargs, e.g. \'{"skeleton_mode": "tube", '
                        '"background_color": "black", "export_scale": 4}\'. '
                        'What a scene may not be talked out of is ignored: '
                        'the legend tree, the coordinate template, the synapse '
                        'skip and the layer identity')
    p.add_argument('--scene-colors-json', default=None, metavar='JSON',
                   help='per-category scene colors as a JSON object merged '
                        'over CATEGORY_COLORS, e.g. \'{"matched": "#00ff00", '
                        '"pooling": "#123456"}\'. A color the renderer cannot '
                        'parse keeps the pipeline default and logs one line')
    p.add_argument('--suspicious-cap', type=int, default=20,
                   help='max examinee rows kept per source neuron')
    p.add_argument('--verify-suspects', action='store_true',
                   help='opt-in (default OFF): verify each queried '
                        'same-name fan-out\'s rival candidates against '
                        'their own target pools — advisory, writes '
                        'suspects_verification.csv + the Suspects tab')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args(argv)
    if args.mode == 'pooling' and args.no_morphology:
        # Every pooling tier is defined morph-qualified, so a run without the
        # gate would publish connectivity findings under claim-shaped names —
        # the distinction the mode exists to make. Refuse, like the widening
        # flags below, rather than degrade silently.
        p.error('--mode pooling requires morphology: drop --no-morphology, or '
                'run --mode restrictive for a structural pass')
    if args.mode == 'pooling' and args.aggressive_expansion:
        # A CLI contradiction should read as a usage error (exit 2 + the
        # help), not as a traceback from deep inside the config.
        p.error('--mode pooling is the unsupervised engine and does not nest '
                'with --aggressive-expansion')
    if args.mode == 'pooling':
        # ... and it needs a target the morph scorer can score. Without this the
        # run scans for tens of minutes and then records the scorer's
        # "not supported" per row while still publishing `matched` claims
        # (measured on FAFB->hemibrain, 2026-09-25).
        from comparison.mapping_validation import check_pooling_target_supported
        try:
            check_pooling_target_supported(
                argparse.Namespace(source_dataset=args.source,
                                   target_dataset=args.target))
        except ValueError as exc:
            p.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg = MappingValidationConfig(
        source_dataset=args.source,
        target_dataset=args.target,
        query_types=[t for t in args.types.split(',') if t.strip()],
        rank_top_k=args.rank_top_k,
        gap_min=args.gap_min,
        verified_top_n=args.verified_top_n,
        invader_borderline_max=args.invader_borderline_max,
        matched_ru_min=args.matched_ru_min,
        candidate_morph_factor=args.candidate_morph_factor,
        target_min_weight=args.target_min_weight,
        target_min_partner_types=args.target_min_partner_types,
        suspicious_jaccard_factor=args.suspicious_jaccard_factor,
        target_min_size_ratio=args.target_min_size_ratio,
        suspicious_ru_margin=args.suspicious_ru_margin,
        pool_ref_cap=args.pool_ref_cap,
        pool_ref_floor_margin=args.pool_ref_floor_margin,
        morph_track_a_offset=args.morph_track_a_offset,
        morph_suspicious_level=args.morph_suspicious_level,
        skip_profile_build=args.skip_profile_build,
        out_map_top_k=args.out_map_top_k,
        skip_out_map_expansion=args.skip_out_map_expansion,
        validation_mode=args.mode,
        # `--aggressive-expansion` is a legacy alias; `normalize_mode`
        # resolves it to the aggressive mode.  `pool_widen` is retired
        # (Rev 3.12) — family mode no longer widens the validated pool.
        aggressive_expansion=args.aggressive_expansion,
        pool_widen=False,
        candidate_window=args.candidate_window,
        deep_cap=args.deep_cap,
        pooling_jaccard_floor=args.pooling_jaccard_floor,
        pooling_rank_union_floor=args.pooling_rank_union_floor,
        pooling_window_mult=args.pooling_window_mult,
        pooling_bar_metric=args.pooling_bar_metric,
        pooling_bar_top_n=args.pooling_bar_top_n,
        pooling_max_morph_targets=args.pooling_max_morph_targets,
        backward_evidence_enabled=args.backward_evidence,
        skip_backward_pass=args.skip_backward_pass,
        backward_top_n=args.backward_top_n,
        backward_max_neurons=args.backward_max_neurons,
        backward_per_branch_cap=args.backward_per_branch_cap,
        backward_scan_pool_targets=not args.no_backward_pool_targets,
        scene_selfcheck=args.scene_selfcheck,
        morph_enabled=not args.no_morphology,
        morph_auc_floor=args.morph_auc_floor,
        skeleton_fetch_workers=args.skeleton_fetch_workers,
        skeleton_fetch_timeout_s=args.skeleton_fetch_timeout,
        visualize=not args.no_visualize,
        max_scenes=args.max_scenes,
        neuron_alpha=args.neuron_alpha,
        scene_viz=_json_object(args.scene_viz_json, '--scene-viz-json'),
        scene_category_colors=_json_object(
            args.scene_colors_json, '--scene-colors-json'),
        suspicious_per_source_cap=args.suspicious_cap,
        verify_suspects=args.verify_suspects,
        output_dir=args.output_dir,
        run_label=args.label,
        verbose=not args.quiet,
    )
    # F-P2 gate: a local-release target auto-prepares the way every other
    # BANC entry point does; a failed preparation refuses the run instead
    # of resolving type pairs against a universe that does not exist.
    from comparison.mapping_validation import ensure_local_release_data
    if not ensure_local_release_data(args.target):
        print('! [TMVEV] refusing: target dataset '
              f'{args.target!r} has no local data and its preparation '
              'failed or was declined')
        return 2

    validator = MappingValidator(cfg)
    run_dir = validator.run()
    print(f'\nResults: {run_dir}')
    # per-run _UserGuide (same content model the UI runner uses; never
    # fails the run). report.html itself is written by the pipeline's
    # own report writer (mapping_validation_report).
    try:
        from ui.output_guide import write_run_guide
        guide = write_run_guide(
            run_dir, 'type_mapping_validation',
            params={
                'source_dataset': cfg.source_dataset,
                'target_dataset': cfg.target_dataset,
                'query': ', '.join(cfg.query_types),
                'validation_mode': cfg.effective_mode,
                'top_k': cfg.top_k, 'top_m': cfg.top_m,
                'min_synapse_threshold': cfg.min_synapse_threshold,
                'rank_top_k': cfg.rank_top_k,
                'backward_evidence_enabled': cfg.backward_evidence_enabled,
                'backward_max_neurons': cfg.backward_max_neurons,
            })
        if guide:
            print(f'Run guide: {guide.name}')
    except Exception as exc:  # noqa: BLE001
        print(f'(run guide skipped: {exc})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
