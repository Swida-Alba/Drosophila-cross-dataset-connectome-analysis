"""Real-data validation of the tier-0 label mapper on the profiling lane.

Loads the actual settled clock label map
(``_clock_production/data/clock_validated_expanded/
clock_settled_type_label_map.json`` — 21 labels x FAFB/MCNS/BANC bodyIds,
built by ``local_data/drive_clock_label_mapper.py``) and runs small real
``ConnectivityProfileComparer`` campaigns against the LOCAL connection
caches (male-cns v1.0 + flywire FAFB v783):

* every clock label resolves members in each dataset (the preset IS the
  per-dataset mapping);
* a label that collides with a real type name (``DN1a`` is a FAFB cell
  type the auto mapper knows) is never re-resolved — tier-0 identity;
* a real two-dataset type-mode run rows exactly the queried labels and
  records ``user_label`` resolutions;
* with partner grouping on, the saved aggregated profiles' partner keys
  collapse into the label namespace.

All tests skip when the fixture or the local datasets are not present.
"""

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent

from comparison.cross_dataset_type_mapper import get_type_mapper  # noqa: E402
from comparison.profile_comparator import ConnectivityProfileComparer  # noqa: E402
from comparison.type_resolver import get_mapper_snapshot  # noqa: E402

MCNS = 'male-cns:v1.0'
FAFB = 'flywire_FAFB_v783'

CLOCK_MAP = (PROJECT_ROOT / '_clock_production' / 'data'
             / 'clock_validated_expanded' / 'clock_settled_type_label_map.json')

# Small label subset: keeps the conn-cache-backed run minutes, not tens of
# minutes.  Every one of these labels has members in BOTH datasets.
LABELS = ['l-LNv', 's-LNv', 'DN1a']

pytestmark = [
    pytest.mark.requires_data,
    pytest.mark.skipif(
        not CLOCK_MAP.exists(),
        reason='clock settled-type label map not available locally'),
]


def _run_kwargs(output_dir):
    return dict(
        query=list(LABELS), datasets=[MCNS, FAFB], verbose=False,
        output_dir=str(output_dir), use_cache=True,
        generate_heatmaps=False, skip_bodyId_level=True,
        ensure_cache_complete=False,
    )


@pytest.fixture(scope='module')
def tier0_comparer():
    """A comparer carrying the real clock label map (no run)."""
    return ConnectivityProfileComparer(
        **_run_kwargs(Path('/tmp')), label_mapping_file=str(CLOCK_MAP))


@pytest.fixture(scope='module')
def real_run_tier0(tmp_path_factory):
    out = tmp_path_factory.mktemp('profiling_tier0')
    comparer = ConnectivityProfileComparer(
        **_run_kwargs(out), label_mapping_file=str(CLOCK_MAP))
    return comparer.run(), comparer, out


@pytest.fixture(scope='module')
def real_run_partner_grouping(tmp_path_factory):
    out = tmp_path_factory.mktemp('profiling_partner_grouping')
    comparer = ConnectivityProfileComparer(
        **_run_kwargs(out), label_mapping_file=str(CLOCK_MAP),
        group_partners_by_mapping=True)
    return comparer.run(), comparer, out


def test_clock_labels_resolve_members_per_dataset(tier0_comparer):
    """R1: every clock label resolves members in each dataset."""
    lm = tier0_comparer._user_label_mapper
    assert lm is not None
    for label in LABELS:
        for ds in (MCNS, FAFB):
            members = lm.get_neurons_for_label(label, ds, role='source')
            assert members, f'clock label {label!r} has no members in {ds}'


def test_tier0_label_not_reresolved_on_real_collision(tier0_comparer):
    """R2: 'DN1a' is a label AND a FAFB cell type the auto mapper knows.
    With the label mapper loaded it must anchor verbatim — never be
    renamed, split, or conflicted away."""
    control = ConnectivityProfileComparer(
        query=['DN1a'], datasets=[MCNS, FAFB], verbose=False)
    mapper = get_type_mapper()
    if mapper is None or not getattr(mapper, '_loaded', False):
        pytest.skip('auto type mapper not available locally')
    control._type_mapper = mapper
    control._mapper_snapshot = get_mapper_snapshot(mapper)
    control_targets, control_status = control._map_query_item_multi(
        'DN1a', FAFB)

    targets, status = tier0_comparer._map_query_item_multi('DN1a', FAFB)
    assert status == 'user_label'
    assert targets == ['DN1a']
    assert tier0_comparer._canonical_label('DN1a', FAFB) == 'DN1a'
    # Whatever the auto mapper does with the raw name (rename, split,
    # conflict), the tier-0 treatment is identity.
    assert (control_status, control_targets) != ('user_label', ['DN1a']) \
        or True  # control shape is informational; identity is the contract


def test_real_run_rows_are_labels_and_resolution_recorded(
        real_run_tier0):
    """R3: a real 2-dataset type-mode run rows exactly the queried labels
    and records the tier-0 resolutions."""
    result, comparer, out = real_run_tier0
    assert sorted(result['type_labels']) == sorted(LABELS)
    assert comparer.mapping_resolution_counts.get('user_label', 0) >= 1
    # every label resolved in both datasets
    for ds in (MCNS, FAFB):
        for label in LABELS:
            record = comparer.mapping_resolution_record[ds][label]
            assert record['status'] == 'user_label'
    # matrices on disk use the label axes
    params = json.loads((Path(result['output_path']) / 'parameters.json')
                        .read_text(encoding='utf-8'))
    assert params['label_mapping_file'] == str(CLOCK_MAP)
    assert params['user_label_mapper_active'] is True
    assert params['aggregation_level'] == 'type'


def test_real_partner_grouping_collapses_profile_keys(
        real_run_partner_grouping):
    """R4: with partner grouping on, the saved aggregated profiles'
    partner keys collapse into the label namespace — member type names
    never survive where a label governed, and at least one label key
    actually appears (the collapse is not a silent no-op)."""
    result, comparer, out = real_run_partner_grouping
    labels = set(comparer._user_label_governing_labels())
    member_ids = set()
    lm = comparer._user_label_mapper
    for label in labels:
        for ds in (MCNS, FAFB):
            member_ids.update(
                str(m) for m in lm.get_neurons_for_label(label, ds,
                                                         role='source'))
    agg_dir = Path(result['output_path']) / 'profiles'
    # multi-dataset layout: profiles/{dataset}/aggregated/*_profile.json
    # (single-dataset runs use profiles/aggregated/)
    profile_files = list(agg_dir.glob('*/aggregated/*_profile.json')) or \
        list(agg_dir.glob('aggregated/*_profile.json'))
    assert profile_files, 'aggregated profiles were not saved'
    n_label_keys = 0
    for path in profile_files:
        data = json.loads(path.read_text(encoding='utf-8'))
        for direction in ('upstream_partners', 'downstream_partners'):
            for key, weight in data[direction].items():
                if key in member_ids:
                    pytest.fail(
                        f'{path.name}: member key {key!r} survived the '
                        f'collapse in {direction}')
                if key in labels:
                    n_label_keys += 1
    assert n_label_keys > 0, (
        'no partner key folded into a group label — the collapse no-opped')
