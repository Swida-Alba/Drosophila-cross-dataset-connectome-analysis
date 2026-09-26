import json, pathlib, sys, shutil
sys.path.insert(0,'src'); sys.path.insert(0,'.')
from ui.dataset_service import DatasetService
root = pathlib.Path('.').resolve()
try:
    svc = DatasetService(str(root))
except TypeError:
    svc = DatasetService()

def listing():
    return [str(getattr(i,'dataset',None) or getattr(i,'id',None) or getattr(i,'name',None))
            for i in svc.get_local_datasets()]

base = listing()
print(f'E2 catalog ({len(base)}): {base}')
pe = root/'datasets'/'probe_empty'; pm = root/'datasets'/'probe_meta'
pe.mkdir(exist_ok=True); pm.mkdir(exist_ok=True)
(pm/'probe_meta_metadata.json').write_text(json.dumps({'dataset':'probe_meta','source':'local'}), encoding='utf-8')
after = listing()
print(f'E3 catalog with probes ({len(after)})')
print('  probe_empty listed:', any('probe_empty' in x for x in after))
print('  probe_meta  listed:', any('probe_meta'  in x for x in after))
print('  new entries vs base:', sorted(set(after)-set(base)) or 'none')
for p in (pe,pm): shutil.rmtree(p, ignore_errors=True)
print('  probes removed:', not pe.exists() and not pm.exists())
disk = sorted(p.name for p in (root/'datasets').iterdir() if p.is_dir())
print('  dataset folders on disk:', disk)
