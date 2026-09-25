import pandas as pd
D='local_data/tmvev-matrix54-20260925_1656/type-map-validation_FAFB_to_BANC_20260925_195213/'
v=pd.read_csv(D+'validation/validation_results.csv',dtype=str)
s=pd.read_csv(D+'validation/pair_summary.csv',dtype=str)
print('validation rows',len(v),'| pairs in summary',len(s))
print('verdict values:',v['verdict'].fillna('').value_counts().to_dict())
flag=[c for c in v.columns if 'below_threshold' in c or 'demoted' in c]
print('demotion flag columns:',flag)
# TMV-3: summary counters vs recomputed verdict counts per (query,source_type,target_type)
key=['query','source_type','target_type']
re_c=v.groupby(key+['mapping_status'])['verdict'].apply(
    lambda x:(x=='verified').sum()).rename('recomputed_verified')
s2=s.rename(columns={c:c for c in s.columns})
sc=[c for c in s.columns if 'verified' in c]
print('summary verified-ish cols:',sc)
m=s2.merge(re_c.reset_index(),on=[k for k in key if k in s2.columns and k in v.columns],how='left') if all(k in s2.columns for k in key) else None
if m is not None:
    col=[c for c in sc if 'strong' not in c][0]
    m['recomputed_verified']=m['recomputed_verified'].fillna(0).astype(int)
    m[col]=m[col].fillna(0).astype(int)
    bad=m[m[col]!=m['recomputed_verified']]
    print('TMV-3 mismatched branches: %d/%d'%(len(bad),len(m)))
    for _,r in bad.head(6).iterrows():
        print('   ',r.get('source_type'),'->',r.get('target_type'),
              'summary',r[col],'rows',r['recomputed_verified'])
# the unobtainable-skeleton question
scored=[c for c in v.columns if 'morph' in c]
print('morph cols:',scored)
nos=v[v['morph_v2_similarity'].isna()] if 'morph_v2_similarity' in v.columns else None
if nos is not None:
    print('rows with NO morph score:',len(nos),'| distinct source bodyIds:',
          nos['source_bodyId'].nunique() if 'source_bodyId' in nos.columns else 'n/a')
    print(nos[[c for c in ('source_type','target_type','source_bodyId','verdict','flags') if c in nos.columns]].head(12).to_string(index=False))
