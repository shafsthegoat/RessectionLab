"""Copy only compact saved metadata and bind the completed saved-result review."""
from pathlib import Path
import hashlib,json
ROOT=Path.cwd();OUT=ROOT/'build/hbe-v5-fixed-fit-result-independent-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def serial(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def bind(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
summary=json.loads((OUT/'result-summary.json').read_bytes())
audit=json.loads((OUT/'audit-result.json').read_bytes())
assert summary['decision']==audit['decision']=='PASS'
evidence=OUT/'evidence';evidence.mkdir(exist_ok=False)
for key in ('publication','terminal','freeze','continuation_ledger'):
    b=summary[key];p=ROOT/b['path'];assert sha(p)==b['sha256']
    (evidence/(key.replace('_','-')+'.json')).write_bytes(p.read_bytes())
rb=summary['release'];p=ROOT/rb['path'];assert sha(p)==rb['sha256']
(evidence/'root-release.json').write_bytes(p.read_bytes())
review={'schema':'hbe-v5-fixed-fit-heldout-admission-v1','decision':'GO',
    'scope':'accepted saved fixed-fit confirmation and pre-reveal prediction freeze evidence only',
    'release':summary['release'],'source_commit':summary['source_commit'],
    'fixed_fit_sha256':summary['fixed_fit_sha256'],'mu_Pa':summary['mu_Pa'],'scale':summary['scale'],
    **{k:summary[k] for k in ('publication','terminal','freeze','predictions','numerical','continuation_ledger')},
    'fitted_checks':{b:'PASS' for b in ('compression','tension')},
    'states_per_native_run':121,'native_calls':2,'refit_calls':0,
    'measured_member_reads':0,'held_out_member_reads':0,'physical_validation_pass':None,
    'held_out_access_released':False,
    'future_access_requires':'separately reviewed reader and explicit root release bound to this exact freeze',
    'report':bind(OUT/'REPORT.txt'),'audit':bind(OUT/'audit-result.json'),
    'summary':bind(OUT/'result-summary.json'),
    'audit_scope':'saved JSON arithmetic and authenticated producer receipts; no independent raw primitive replay'}
(OUT/'heldout-admission-review.json').write_bytes(serial(review))
names=['REPORT.txt','result-summary.json','audit-result.json','audit-console.txt',
       'audit_saved.py','heldout-admission-review.json','package_review.py']
files=[OUT/n for n in names]+sorted(evidence.iterdir())
index={'schema':'hbe-v5-fixed-fit-result-independent-index-v1','decision':'PASS',
       'files':[{**bind(p),'bytes':p.stat().st_size} for p in files],
       'source_output_bindings':{k:summary[k] for k in ('publication','terminal','freeze','predictions','numerical','continuation_ledger')},
       'full_numerical_json_copied':False,'prediction_arrays_copied':False,'raw_primitives_copied':False,
       'native_calls_by_audit':0,'measured_member_reads_by_audit':0,'physical_validation_pass':None}
(OUT/'source-index.json').write_bytes(serial(index))
for name in ('REPORT.txt','result-summary.json','audit-result.json','heldout-admission-review.json','source-index.json'):
    print(name,sha(OUT/name))
print('packaged_files',len(files),'packaged_bytes',sum(p.stat().st_size for p in files))
