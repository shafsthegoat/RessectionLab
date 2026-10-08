import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('metadata_fetch',ROOT/'metadata_fetch.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
folders=json.loads((ROOT/'request-02-metadata.json').read_text())['data']
train=[3,5,7,11,12,15,16,17,18,21,23,24,25]
for case in train:
 matching=[d for d in folders if d['attributes']['name']==f'Case{case}']; assert len(matching)==1
 url=matching[0]['relationships']['files']['links']['related']['href']+'?page%5Bsize%5D=100'
 try: b=m.fetch(url)
 except ValueError as e:
  print('BUDGET_STOP',type(e).__name__,flush=True); break
 if b:
  assert b['links']['next'] is None,'Unexpected pagination; do not infer absence'
  names=[(d['attributes']['name'],d['attributes'].get('size'),d['attributes'].get('current_version')) for d in b['data'] if '-resection.nii.gz' in d['attributes']['name']]
  print(json.dumps({'case':case,'cavity_files':names}),flush=True)
