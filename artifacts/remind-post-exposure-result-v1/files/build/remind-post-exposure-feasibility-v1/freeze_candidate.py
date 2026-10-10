"""One-shot metadata/source binding for the reviewed fixed-four search-only run."""
from pathlib import Path
import ast,hashlib,json,subprocess,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
EXPECTED_HEAD='d583008cb1ebc36b4ed34b5f722f3b4f8cc959ec'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==EXPECTED_HEAD
priorbase=ROOT/'build/remind-partial-domain-public-preparation-v1'
prior=json.loads((priorbase/'source-index.json').read_bytes())
paths={ROOT/n for n in prior['files'] if n.startswith('src/resectionlab/')}
paths.add(ROOT/'src/resectionlab/post_exposure.py')
# Retain prior closure and resolve any new local static imports without importing models/data.
pending=list(paths);seen=set()
while pending:
 p=pending.pop()
 if p in seen:continue
 seen.add(p)
 for node in ast.walk(ast.parse(p.read_text())):
  modules=[]
  if isinstance(node,ast.Import):modules=[a.name for a in node.names]
  elif isinstance(node,ast.ImportFrom):
   if node.level:modules=['resectionlab'+('.'+node.module if node.module else '')]
   elif node.module:modules=[node.module]
  for module in modules:
   if module.startswith('resectionlab.'):
    dep=ROOT/'src'/Path(*module.split('.')).with_suffix('.py')
    if dep.is_file() and dep not in paths:paths.add(dep);pending.append(dep)
paths.update(ROOT/n for n in ('build/goal-conditioned-policy-v1/run_contact_owned.py','build/goal-conditioned-policy-v1/darwin_fast_sampler.py','tests/test_post_exposure.py'))
paths.update(HERE/n for n in ('run_owned.py','initial_inventory_worker.py','batch_contract.py','freeze_candidate.py','tests/test_batch_controls.py','controls-attempt-01/stdout.log','controls-attempt-01/receipt.json'))
paths.update((priorbase/'manifests-v2').glob('*.json'))
paths.add(priorbase/'evidence-index.json')
evidence=json.loads((priorbase/'evidence-index.json').read_bytes())
paths.update(Path(evidence[k]['path']) for k in ('cohort','visual'))
for row in evidence['cases']:paths.update(Path(row[k]['path']) for k in ('conversion','saved_review'))
historical=ROOT/'build/balanced-teacher-il64-v1/root-release.json';paths.add(historical)
promotion=ROOT/'build/remind-post-exposure-preparation-v1/promotion-index.json';paths.add(promotion)
promotions=json.loads(promotion.read_bytes())['files']
for row in promotions:assert sha(ROOT/row['canonical_path'])==row['sha256'],row['canonical_path']
index={'scope':'Fixed four TRAIN post-exposure initial inventory plus nominal greedy complete routes and independent replay; zero learning admission',
 'canonical_head':EXPECTED_HEAD,'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},
 'promotions':promotions,'generated_controls':{'passed':23,'seconds':0.88,'patient_reads':0},
 'failure_semantics':'Every fixed row retained; any unresolved route makes batch incomplete without complete-history attestation; genuine zero-legal STOP is a completed negative.'}
save(HERE/'source-index.json',index)
recipe=json.loads(historical.read_bytes())['learning_protocol']['cohort_execution']
sys.path.insert(0,str(HERE));from batch_contract import CAPS,SUBJECTS,CONDITION,EXPOSURE,GREEDY_SECONDS
release={'execution_released':False,'expected_head':None,'subjects':list(SUBJECTS),'caps':CAPS,
 'source_index':ref(HERE/'source-index.json'),'public_index':ref(priorbase/'manifests-v2/public-index.json'),
 'cohort':ref(ROOT/'manifests/experiments/remind-component-cohort-v1.json'),'historical_learning_release':ref(historical),
 'proposal_config':recipe['proposal_config'],'max_steps':recipe['max_steps'],'retention_reference':recipe['retention_mode'],'retention_applied':False,
 'occupancy_condition':CONDITION,'post_exposure_condition':EXPOSURE,'greedy_seconds_per_case':GREEDY_SECONDS,
 'qualification':'Fixed occupancy-aware source-axis0 post-exposure condition: initial inventory and one complete nominal greedy route per case; authoritative sequence and independent replay; no patient/access replacement or fallback',
 'models_or_optimizer_updates':0,'training_admitted':False}
save(HERE/'prepared-release.json',release)
# Root explicitly requested this separate release at the frozen canonical HEAD; no launch here.
released=dict(release,execution_released=True,expected_head=EXPECTED_HEAD)
save(HERE/'root-release.json',released)
print(json.dumps({'source_index_sha256':sha(HERE/'source-index.json'),'prepared_release_sha256':sha(HERE/'prepared-release.json'),'root_release_sha256':sha(HERE/'root-release.json'),'source_bindings':len(paths),'workers_started':0,'patient_arrays_opened':0}))
