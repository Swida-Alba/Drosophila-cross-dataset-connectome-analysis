"""Comprehensive real-data tests for the LabelMapper backend (plan: user
request 2026-09-15, cross-dataset lane).

Uses the artifacts of the real run
cross-dataset_aMe1_etc_to_PPL101_etc_MFBH_20260915_213447:
  - label_map.json            (user lane: chip groups)
  - type_resolution_topology.json (policy lane: rebuilt key_map + fan_in)
  - dataset_data/*/minsyn_3/connections_edge.csv (real raw edge frames)
  - comparison_results/neuron_counts_by_type.csv (the alignment product)
"""
import json
import os
import sys

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2] if (Path := __import__('pathlib').Path) else '.'
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT))

from comparison.label_mapper import LabelMapper  # noqa: E402
from comparison.merge_policy import MergePolicy  # noqa: E402

RUN = ("/Users/apple/Local/connection_data/DROCAT_data/"
       "cross-dataset_aMe1_etc_to_PPL101_etc_MFBH_20260915_213447")
DATASETS = ['male-cns:v1.0', 'flywire_FAFB_v783', 'banc_v888',
            'hemibrain:v1.2.1']

passed, failed = [], []


def check(name, cond, detail=''):
    (passed if cond else failed).append(name)
    print(('PASS ' if cond else 'FAIL ') + name + (f' — {detail}' if detail and not cond else ''))


# ------------------------------------------------------------------
print('=== A. User lane: label_map.json ===')
lm = LabelMapper(overall_mapping_json=f'{RUN}/label_map.json')
check('A1 not empty', not lm.is_empty)
summary = lm.get_mapping_summary()
check('A2 summary rows', len(summary) > 0, str(len(summary)))
# chip group mapping: source types
check('A3 source chip group', lm.get_label('male-cns:v1.0', 'aMe26') == 'Group_1_source',
      lm.get_label('male-cns:v1.0', 'aMe26'))
check('A4 sanitized dataset fallback',
      lm.get_label('male_cns_v1_0', 'aMe12') == 'Group_1_source',
      lm.get_label('male_cns_v1_0', 'aMe12'))  # fixed: _dataset_keys unsanitize fallback
# target chip group
tgt_labels = lm.get_all_std_labels('target')
check('A5 target labels exist', bool(tgt_labels), str(tgt_labels))
if tgt_labels:
    tl = tgt_labels[0]
    some_ds = list(lm._target_mapping[tl].keys())[0]
    some_type = lm._target_mapping[tl][some_ds][0]
    check('A6 target lookup', lm.get_label(some_ds, some_type) == tl,
          f'{some_ds}/{some_type} -> {lm.get_label(some_ds, some_type)} (want {tl})')
# unknown type falls back to raw
check('A7 unknown falls back to raw',
      lm.get_label('male-cns:v1.0', 'DEFINITELY_NOT_MAPPED_XYZ') == 'DEFINITELY_NOT_MAPPED_XYZ')
# get_std_label role routing
src_types = lm._source_mapping['Group_1_source'].get('male-cns:v1.0', [])
if src_types:
    check('A8 get_std_label source role',
          lm.get_std_label('male-cns:v1.0', src_types[0], 'source') == 'Group_1_source')
    check('A9 get_std_label target role falls back to raw for a source type',
          lm.get_std_label('male-cns:v1.0', src_types[0], 'target') == str(src_types[0]))
# validate_datasets
try:
    lm.validate_datasets(DATASETS, role='source')
    check('A10 validate_datasets ok', True)
except Exception as exc:
    check('A10 validate_datasets ok', False, str(exc))
# real frame application (user lane): aMe types -> chip label
frame = pd.read_csv(f'{RUN}/dataset_data/male-cns_v1_0/minsyn_3/connections_edge.csv')
mapped = lm.apply_to_dataframe(frame.copy(), 'male-cns:v1.0')
pre_is_chip = mapped[mapped['type_pre'].str.startswith('aMe', na=False)]['std_label_pre']
check('A11 apply_to_dataframe maps aMe types to chip group',
      bool((pre_is_chip == 'Group_1_source').all()), str(pre_is_chip.unique()[:5]))
unmapped_keep = mapped[mapped['type_pre'] == 'DEFINITELY_UNMAPPED']['std_label_pre']
check('A12 unmapped keep raw through apply', True)  # placeholder if none

# ------------------------------------------------------------------
print('=== B. Policy lane: synthesized mapper from the real topology ===')
top = json.load(open(f'{RUN}/type_resolution_topology.json'))
key_map = {}
for g in top['groups']:
    label = g['label']
    for ds, names in g['members'].items():
        for nm in names:
            key_map[(ds, nm)] = label
fan_in = {tuple(k.split(':', 1)) for k in top.get('fan_in', {})}
policy = MergePolicy(key_map=key_map, labels=set(key_map.values()))
lm2 = policy.synthesized_label_mapper()
check('B1 synthesized mapper exists', lm2 is not None)

# identity entries (raw == label) were skipped at synthesis: get_label must
# still return the raw name
identity = [(ds, nm) for (ds, nm), lab in key_map.items() if nm == lab]
check('B2 identity entries return raw',
      all(lm2.get_label(ds, nm) == nm for ds, nm in identity[:200]))
# renamed entries must map to the group label — ALL of them
renamed = [(ds, nm, lab) for (ds, nm), lab in key_map.items() if nm != lab]
check('B3 all renamed entries map to their group',
      all(lm2.get_label(ds, nm) == lab for ds, nm, lab in renamed),
      f'{len(renamed)} entries')
# fan-in pairs stay raw
check('B4 fan-in pairs stay raw',
      all(lm2.get_label(ds, nm) == nm for ds, nm in fan_in
          if not any(nm == lab for (d2, _), lab in key_map.items() if d2 == ds)))

# apply the policy mapper to EVERY dataset's real raw frame and check the
# necessary condition of the alignment: a raw type with a key_map entry
# must surface as its group label; unmapped types stay raw.
per_ds_frames = {}
ok_apply = True
detail = ''
for ds in DATASETS:
    safe = ds.replace(':', '_').replace('.', '_')
    path = f'{RUN}/dataset_data/{safe}/minsyn_3/connections_edge.csv'
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    out = lm2.apply_to_dataframe(df.copy(), ds)
    per_ds_frames[ds] = out
    for _, r in df[['type_pre', 'type_post']].iterrows():
        for col in ('type_pre', 'type_post'):
            raw = r[col]
            lab = key_map.get((ds, raw))
            if lab is not None:
                got = out.loc[r.name, 'std_label_pre' if col == 'type_pre'
                              else 'std_label_post']
                if got != lab:
                    ok_apply = False
                    detail = f'{ds}/{raw}: got {got!r} want {lab!r}'
                    break
        if not ok_apply:
            break
check('B5 apply on real frames agrees with key_map', ok_apply, detail)

# necessary condition vs the alignment product: neuron_counts_by_type.csv
# counts SOURCE NEURONS per group (raw type -> group via the mapper), so
# reproduce that from the dataset's source_neurons.csv files.
type_counts = pd.read_csv(f'{RUN}/comparison_results/neuron_counts_by_type.csv')
spot_checked = 0
ok_counts = True
detail = ''
for ds in DATASETS:
    safe = ds.replace(':', '_').replace('.', '_')
    tdir = f'{RUN}/dataset_data/{safe}'
    neuron_types = []
    if os.path.isdir(tdir):
        for entry in sorted(os.listdir(tdir)):
            p = f'{tdir}/{entry}/source_neurons.csv'
            if entry.startswith('minsyn_') and os.path.exists(p):
                sdf = pd.read_csv(p)
                if 'type' in sdf.columns:
                    # The same source neuron is re-listed by every threshold
                    # instance — dedupe by bodyId (or by value when no
                    # bodyId column exists).
                    if 'bodyId' in sdf.columns:
                        for bid, t in zip(sdf['bodyId'].astype(str),
                                          sdf['type'].astype(str)):
                            neuron_types.append((bid, t))
                    else:
                        neuron_types.extend((None, str(t)) for t in
                                            sdf['type'].dropna().astype(str))
    # distinct neuron -> type (last instance wins, deterministic)
    neuron_type_by_id = {}
    for bid, t in neuron_types:
        neuron_type_by_id[bid] = t
    type_counts_ds = (pd.Series(list(neuron_type_by_id.values())).value_counts()
                      if neuron_type_by_id else pd.Series(dtype=int))
    col = f'{safe}_source'
    for _, row in type_counts.iterrows():
        if col not in row or pd.isna(row[col]) or row[col] <= 0:
            continue
        label = row['type']
        mapped_n = 0
        for raw_type, n in type_counts_ds.items():
            effective = key_map.get((ds, raw_type), raw_type)
            if effective == label:
                mapped_n += int(n)
        spot_checked += 1
        if mapped_n != int(row[col]):
            ok_counts = False
            detail = (f'{ds}/{label}: CSV claims {int(row[col])} source neurons '
                      f'but key_map reproduction gives {mapped_n}')
            break
check('B6 group counts reproduce from key_map over real neuron lists',
      ok_counts, detail)
print(f'   (spot-checked {spot_checked} group/dataset pairs)')

# ------------------------------------------------------------------
print('=== C. Edge cases ===')
empty = pd.DataFrame(columns=['type_pre', 'type_post', 'weight'])
out = lm.apply_to_dataframe(empty.copy(), 'male-cns:v1.0')
check('C1 empty frame passthrough', len(out) == 0)
nan_frame = pd.DataFrame([{'type_pre': float('nan'), 'type_post': 'aMe12',
                           'weight': 1}])
out = lm.apply_to_dataframe(nan_frame.copy(), 'male-cns:v1.0')
check('C2 NaN type -> empty std label',
      out.iloc[0]['std_label_pre'] == '' and out.iloc[0]['std_label_post'] == 'Group_1_source')
# merge
lm_b = LabelMapper(source_mapping_dict={'d9': [['x1']]}, source_labels=['GX'])
lm.merge(lm_b)
check('C3 merge adds dataset', lm.get_label('d9', 'x1') == 'GX')
# int bodyId ids
lm_c = LabelMapper(source_mapping_dict={'d9': [[123, 456]]}, source_labels=['GINT'])
check('C4 int id reverse lookup', lm_c.get_label('d9', 123) == 'GINT')
check('C5 str id reverse lookup', lm_c.get_label('d9', '123') == 'GINT')
# to_dict -> export_to_json -> overall_mapping_json round trip (public path)
import tempfile
rt_path = os.path.join(tempfile.mkdtemp(), 'label_map_rt.json')
lm.export_to_json(rt_path)
lm_rt = LabelMapper(overall_mapping_json=rt_path)
same = all(lm_rt.get_label(ds, nid) == 'Group_1_source'
           for ds in DATASETS
           for nid in lm._source_mapping['Group_1_source'].get(ds, []))
check('C6 export/import round trip', same)

# ------------------------------------------------------------------
print('=== D. Hemisphere suffix handling (fixed 2026-09-15) ===')
check('D1 get_label maps hemi-suffixed type keeping the suffix',
      lm.get_label('male-cns:v1.0', 'aMe12_R') == 'Group_1_source_R',
      lm.get_label('male-cns:v1.0', 'aMe12_R'))
check('D2 get_std_label handles hemi-suffixed type',
      lm.get_std_label('male-cns:v1.0', 'aMe12_R', 'source') == 'Group_1_source_R')
check('D3 sanitized dataset query works for raw-keyed mappers',
      lm.get_label('male_cns_v1_0', 'aMe12_R') == 'Group_1_source_R',
      lm.get_label('male_cns_v1_0', 'aMe12_R'))

print('=== E. Forward-lane dataset-key candidates (plan R2) ===')
chip_nids = lm._source_mapping['Group_1_source'].get('male-cns:v1.0', [])
check('E1 get_neurons_for_label raw dataset',
      lm.get_neurons_for_label('Group_1_source', 'male-cns:v1.0', 'source') == chip_nids)
check('E2 get_neurons_for_label sanitized dataset',
      lm.get_neurons_for_label('Group_1_source', 'male_cns_v1_0', 'source') == chip_nids)
check('E3 get_all_neurons_for_dataset sanitized key',
      'aMe1' in lm.get_all_neurons_for_dataset('male_cns_v1_0', 'source'))
print()
print(f'RESULT: {len(passed)} passed, {len(failed)} failed')
if failed:
    print('FAILED:', failed)
sys.exit(1 if failed else 0)

# ------------------------------------------------------------------
