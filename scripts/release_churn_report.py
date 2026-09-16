#!/usr/bin/env python
"""
Release-to-release connectivity churn report.

Compares the SAME neurons across two versions of one dataset family
(e.g. male-cns:v0.9 -> male-cns:v1.0) and ranks them by how much their
connectivity changed. Thin orchestration over the existing building
blocks: ConnectivityProfiler profiles + ProfileComparator scoring — the
same backend the Connectivity and Cross-Dataset tabs use.

Example:
    python scripts/release_churn_report.py \
        --old male-cns:v0.9 --new male-cns:v1.0 \
        --bodyids 12211,12517,12737,12740

    python scripts/release_churn_report.py \
        --old male-cns:v0.9 --new male-cns:v1.0 --type aMe12 --top 20

Output: one CSV row per neuron with per-version type/instance identity,
the similarity metrics, and a 'churn' column (1 - jaccard, so higher =
more changed), sorted most-changed first.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd  # noqa: E402

from comparison.connectivity_profiler import ConnectivityProfiler  # noqa: E402
from comparison.profile_comparator import (  # noqa: E402
    ProfileComparator,
    _body_id_instance_name,
)
from utils.naming_utils import make_unique_dataset_labels  # noqa: E402


def _resolve_bodyids(profiler: ConnectivityProfiler, args) -> list:
    """BodyIds to compare: explicit list, or a type resolved in either version."""
    if args.bodyids:
        return [int(b) for b in str(args.bodyids).split(',') if b.strip()]
    if args.type:
        for dataset in (args.old, args.new):
            ids = profiler.get_bodyids_for_type(args.type, dataset) or []
            if ids:
                return [int(b) for b in ids]
        raise SystemExit(f"No bodyIds found for type '{args.type}' in either version")
    raise SystemExit("Provide --bodyids or --type")


def _has_partners(profile) -> bool:
    return bool(profile is not None and (
        profile.upstream_partners or profile.downstream_partners))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--old', required=True,
                        help='Older dataset id (e.g. male-cns:v0.9)')
    parser.add_argument('--new', required=True,
                        help='Newer dataset id (e.g. male-cns:v1.0)')
    parser.add_argument('--bodyids', default=None,
                        help='Comma-separated bodyIds present in both versions')
    parser.add_argument('--type', default=None, dest='type',
                        help='Resolve bodyIds from this type name')
    parser.add_argument('--direction', default='both',
                        choices=['both', 'upstream', 'downstream'])
    parser.add_argument('--top', type=int, default=20,
                        help='Rows to print to the console (default 20)')
    parser.add_argument('--output', default=None,
                        help='Output CSV path (default: local_data/release_churn/…)')
    args = parser.parse_args()

    profiler = ConnectivityProfiler(datasets=[args.old, args.new], verbose=False)
    bodyids = _resolve_bodyids(profiler, args)
    print(f"Comparing {len(bodyids)} neuron(s): {args.old} -> {args.new}")

    rows = []
    for bid in bodyids:
        old_profile = profiler.get_profile(bid, args.old)
        new_profile = profiler.get_profile(bid, args.new)
        old_ok, new_ok = _has_partners(old_profile), _has_partners(new_profile)
        row = {
            'bodyId': bid,
            'old_type': getattr(old_profile, 'neuron_type', None) or '',
            'new_type': getattr(new_profile, 'neuron_type', None) or '',
            'old_instance': _body_id_instance_name(args.old, bid),
            'new_instance': _body_id_instance_name(args.new, bid),
            'same_type': bool(old_profile and new_profile
                              and old_profile.neuron_type
                              and old_profile.neuron_type == new_profile.neuron_type),
            'status': 'ok' if (old_ok and new_ok) else (
                'no data in old' if not old_ok else 'no data in new'),
        }
        if old_ok and new_ok:
            scores = ProfileComparator.combined_score(
                old_profile, new_profile, direction=args.direction)
            row.update({
                'jaccard': scores.get('jaccard'),
                'weighted_jaccard': scores.get('weighted_jaccard'),
                'cosine': scores.get('cosine'),
                'rank_union': scores.get('rank_union'),
                'churn': (1.0 - scores['jaccard']
                          if scores.get('jaccard') is not None
                          and pd.notna(scores.get('jaccard')) else None),
            })
        else:
            row.update({'jaccard': None, 'weighted_jaccard': None,
                        'cosine': None, 'rank_union': None, 'churn': None})
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        'churn', ascending=False, na_position='last').reset_index(drop=True)

    if args.output:
        out_path = Path(args.output)
    else:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Unique 4-char labels, version-suffixed on same-family collisions
        # (churn_MCNS_v0_9_to_MCNS_v1_0_...) — filename-safe, no colons.
        old_label, new_label = make_unique_dataset_labels([args.old, args.new])
        out_path = (Path(__file__).parent.parent / 'local_data' / 'release_churn'
                    / f"churn_{old_label}_to_{new_label}_{stamp}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nMost-changed neurons ({args.old} -> {args.new}, "
          f"direction={args.direction}):")
    show = df.head(args.top)
    if show.empty:
        print("  (no comparable neurons)")
    else:
        print(show.to_string(index=False, max_colwidth=24))
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
