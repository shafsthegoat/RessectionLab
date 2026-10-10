"""Bounded, stdlib-only saved JSON audit. No geometry replay or array/weight reads."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'build/post-exposure-IL045-evaluation-diagnostic-v1'
START = time.monotonic()
signal.signal(signal.SIGALRM, lambda *unused: (_ for _ in ()).throw(TimeoutError('60s audit cap')))
signal.alarm(60)
inputs = {}
checks = 0

def need(ok, label):
    global checks
    checks += 1
    if not ok:
        raise AssertionError(label)

def read(path, expected=None):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    need(p.suffix in ('.json', '.py', '.log'), 'saved JSON/source only')
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        s = os.fstat(fd)
        need(stat.S_ISREG(s.st_mode) and 0 <= s.st_size <= 4*1024**2 and (s.st_size > 0 or p.suffix == '.log'), 'bounded regular input')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            data = stream.read(4*1024**2 + 1)
        need(len(data) == s.st_size and len(data) <= 4*1024**2, 'bounded stable size')
    finally:
        os.close(fd)
    digest = hashlib.sha256(data).hexdigest()
    if expected:
        need(digest == expected, 'exact input '+str(p))
    inputs[str(p.relative_to(ROOT))] = {'sha256': digest, 'bytes': len(data)}
    return data

def load(path, expected=None):
    return json.loads(read(path, expected))

def semantic(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')

def dot(a, b):
    return sum(x*y for x, y in zip(a,b))

def norm(a):
    return math.sqrt(dot(a,a))

def sub(a,b):
    return [x-y for x,y in zip(a,b)]

def unit(a):
    n=norm(a)
    return [x/n for x in a]

release_sha = 'fcf09a5b90fffb4f6ca91632097b7979f384c76211f50dd64db3f38c2bd8aad6'
result_sha = '2a6654ccc3855c180c2724cc9eb24d5720d3120a5a61bfd84e9ef416ba7304cc'
release = load(RUN/'root-release.json', release_sha)
index = load(release['source_index']['path'], release['source_index']['sha256'])
result = load(RUN/'attempt-01/result.json', result_sha)
receipt = load(RUN/'attempt-01.supervision/receipt.json')
certificate = load(RUN/'attempt-01/native-replay.json', result['native_replay_sha256'])
metrics = load(RUN/'attempt-01/committed-replay-metrics.json')
derivation = load(RUN/'attempt-01/source/public-task-derivation.json')
saved = load(**{'path':release['inputs']['plan']['path'], 'expected':release['inputs']['plan']['sha256']})
failed_result = load(release['inputs']['failed_result']['path'],release['inputs']['failed_result']['sha256'])
failed_receipt = load(release['inputs']['failed_receipt']['path'],release['inputs']['failed_receipt']['sha256'])
isolation = load(ROOT/'build/independent-identical-axis-repair-v1/saved-pose-isolation.json',
                 '2855bca3e6b33ecf5eae5532038b801b0e935781e2b420e2ce1e9eda710fd88d')
for name in ('batch_contract.py','initial_inventory_worker.py','run_owned.py'):
    read(RUN/name, index['files'][str((RUN/name).relative_to(ROOT))])
read(RUN/'attempt-01.supervision/declaration.json')
read(RUN/'attempt-01.supervision/worker.log')

need(release['execution_released'] is True and release['expected_head']==index['head'], 'release/source join')
need(receipt['source_index']==release['source_index'], 'parent source binding')
need(receipt['release_sha256']==result['release_sha256']==release_sha, 'release joins')
need(receipt['result_sha256']==result_sha and receipt['caps']==release['caps'], 'parent result/caps')
need(receipt['status']=='complete' and receipt['exit_code']==0 and receipt['stop_reason'] is None, 'owned diagnostic complete')
need(receipt['cleanup_errors']==[] and receipt['remaining_owned_pids']==[] and receipt['worker_termination_confirmed'] is True, 'owned clean/reaped')
need(receipt['automatic_retry'] is False, 'one attempt')
need(receipt['elapsed_seconds']<release['caps']['parent_seconds'] and result['elapsed_seconds']<release['caps']['worker_seconds'], 'wall caps')
need(receipt['sampled_peak_rss_bytes']<release['caps']['sampled_rss_bytes'], 'sampled RSS cap')
actual_output_bytes=sum(p.stat().st_size for p in (RUN/'attempt-01').rglob('*') if p.is_file())
need(actual_output_bytes==receipt['output_bytes']<release['caps']['output_bytes'], 'exact output size')
need(result['native_previews']==2184<=release['caps']['native_previews'], 'preview count')
need(result['source_visits']==1 and result['committed_actions']==24, 'fixed one-source 24-action scope')
need(all(result[k]==0 for k in ('policy_forwards','optimizer_updates','checkpoint_loads','teacher_search_calls')), 'no model/refit/search')
need(result['training_admitted'] is False and result['independent_accepted'] is False, 'diagnostic is not route admission')
need(result['checkpoint_sha256']==release['inputs']['checkpoint']['sha256'], 'hash-only checkpoint binding')
need(result['saved_plan_sha256']==release['inputs']['plan']['sha256'], 'unchanged saved plan')
plan=saved['plan']
need(semantic(plan)==saved['plan_seal'], 'plan seal independently recomputed')
need(plan['learning_updates']==64 and plan['parameter_hash']==release['parameter_hash'], 'fixed fitted parameter metadata')
need(len(plan['actions'])==24 and plan['terminal_reason']=='HORIZON' and 'STOP' not in plan['actions'], 'fixed horizon sequence')
identity=lambda rows:[{k:v for k,v in row.items() if k!='outcome_scope'} for row in rows]
need(identity(plan['history'])==identity(metrics['history']), 'complete original/committed physical histories equal')
need(certificate['metrics']==metrics, 'certificate binds complete committed metrics')
need([x['action_id'] for x in metrics['history']]==plan['actions'], 'action order equal')
need(metrics['terminated'] is True and metrics['steps']==24, 'complete native HORIZON')
audit=certificate['independent_geometry']
need(audit['complete_episode'] is True and audit['accepted'] is False, 'complete episode and rejected geometry distinct')
geometry=audit['geometry']
need(geometry==result['independent_geometry'], 'exact negative certificate in result')
need(geometry['failures']==['native_post_exposure_access_failure'] and geometry['action_count']==22, 'exact failed branch')
need(result['first_failing_action_indices_zero_based']==[21], 'first failure index')
row=metrics['history'][21]
need(row['action_id']==geometry['first_failed_action']==isolation['action_id'], 'failure action join')
need(isolation['source_metrics_sha256']==inputs[str((RUN/'attempt-01/committed-replay-metrics.json').relative_to(ROOT))]['sha256'], 'isolation metrics binding')
need(isolation['source_access_sha256']==inputs[str((RUN/'attempt-01/source/public-task-derivation.json').relative_to(ROOT))]['sha256'], 'isolation access binding')
need(isolation['start_access_failure'] is None and isolation['end_access_failure'] is None, 'owner isolation endpoints fit')
need(isolation['independent_motion']['failures']==['continuous_rotating_access_window_check_unsupported'], 'owner isolation nested reason')
need(isolation['axis_exact_equal'] is True and isolation['acos_angle_radians']>1e-10 and isolation['stable_atan2_angle_radians']==0, 'isolated false angle')

all_removed=[tuple(c) for h in metrics['history'] for c in h['removed_indices_native']]
need(len(all_removed)==len(set(all_removed))==51, '51 distinct committed removed cells')
need(math.isclose(sum(h['target_removed_mm3'] for h in metrics['history']),metrics['target_removed_mm3'],rel_tol=0,abs_tol=1e-10), 'target scalar sum')
need(math.isclose(sum(h['reward'] for h in metrics['history']),metrics['total_reward'],rel_tol=0,abs_tol=1e-10), 'reward scalar sum')
need(metrics['normal_removed_mm3']==0 and all(h['normal_removed_mm3']==0 for h in metrics['history']), 'zero outside-target removal only')
prior=sum(len(h['removed_indices_native']) for h in metrics['history'][:21])
need(math.isclose(prior*geometry['source_voxel_volume_mm3'],geometry['claimed_source_tissue_volume_mm3'],rel_tol=0,abs_tol=1e-10), 'accepted prefix volume accounting')

# Independent scalar aperture math, not a collision/coverage replay.
axis=unit(unit(row['axis_unit']))
access=derivation['post_exposure']['access']; normal=unit(access['normal_inward'])
cosine=dot(axis,normal)
endpoint_checks=[]
for label, tip in [('entry',row['entry_mm']),('tip',row['tip_mm'])]:
    depth=dot(sub(tip,access['center_mm']),normal)/cosine
    crossing=[x-depth*a for x,a in zip(tip,axis)]
    footprint=norm(sub(crossing,access['center_mm']))+1.4/cosine
    need(cosine>math.cos(math.radians(45))-1e-9 and -1e-9<=depth<=100+1e-9 and footprint<=6+1e-9, 'scalar endpoint aperture '+label)
    endpoint_checks.append({'endpoint':label,'depth_mm':depth,'conservative_footprint_radius_mm':footprint,'margin_mm':6-footprint})
delta=sub(row['tip_mm'],row['entry_mm']);advance=dot(delta,axis)
axial_residual=norm([d-advance*a for d,a in zip(delta,axis)])
need(axial_residual<1e-9, 'fixed-axis axial displacement')
need(math.isclose(2*norm(sub(row['tip_mm'],row['physical_start_mm'])),row['complete_tool_path_length_mm'],rel_tol=0,abs_tol=1e-10), 'approach plus reverse cost length')

excluded={'checkpoint':release['inputs']['checkpoint'],'full_native_replay':{'path':str((RUN/'attempt-01/native-replay.json').relative_to(ROOT)),**inputs[str((RUN/'attempt-01/native-replay.json').relative_to(ROOT))]},
          'full_committed_metrics':{'path':str((RUN/'attempt-01/committed-replay-metrics.json').relative_to(ROOT)),**inputs[str((RUN/'attempt-01/committed-replay-metrics.json').relative_to(ROOT))]},
          'original_plan':release['inputs']['plan']}
write('input-index.json', {'files':inputs,'excluded_bulk_bindings':excluded,'checkpoint_bytes_not_opened_by_audit':True})
summary={'status':'PASS_saved_diagnostic_consistency_not_route_acceptance','checks':checks,
    'elapsed_seconds':time.monotonic()-START,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    'interpretation':'Identical-axis rotation classification defect isolated; full route remains independently rejected in this retained attempt. No corrected route replay by this audit.',
    'owned_elapsed_seconds':receipt['elapsed_seconds'],'owned_peak_rss_bytes':receipt['sampled_peak_rss_bytes'],
    'native_previews':result['native_previews'],'committed_actions':24,'first_failed_action_index':21,
    'failed_action_id':row['action_id'],'final_target_removed_mm3':metrics['target_removed_mm3'],
    'final_target_removed_cells':51,'normal_removed_mm3':metrics['normal_removed_mm3'],'total_reward':metrics['total_reward'],
    'geometry_accepted':False,'training_admitted':False,'prefix_removed_cells_before_failure':prior,
    'scalar_endpoint_access':endpoint_checks,'scalar_axial_residual_mm':axial_residual,
    'owner_generated_isolation':isolation,'physical_validation_pass':None,
    'scope':'saved JSON and source only; no acquired array, checkpoint, geometry replay, model or optimizer execution; original failed result/receipt untouched'}
need(summary['peak_rss_bytes']<512*1024**2, 'audit observed RSS cap')
summary['checks']=checks
write('audit-result.json',summary)
for rel in ('result.json','source/public-task-derivation.json','source/admitted-public-bindings.json'):
    src=RUN/'attempt-01'/rel
    dst=OUT/('result.json' if rel=='result.json' else Path(rel).name)
    dst.write_bytes(read(src))
for name, src in [('receipt.json',RUN/'attempt-01.supervision/receipt.json'),('declaration.json',RUN/'attempt-01.supervision/declaration.json'),
                  ('worker.log',RUN/'attempt-01.supervision/worker.log'),('root-release.json',RUN/'root-release.json'),
                  ('source-index.json',RUN/'source-index.json'),('batch_contract.py',RUN/'batch_contract.py'),
                  ('initial_inventory_worker.py',RUN/'initial_inventory_worker.py'),('run_owned.py',RUN/'run_owned.py'),
                  ('saved-pose-isolation.json',ROOT/'build/independent-identical-axis-repair-v1/saved-pose-isolation.json')]:
    (OUT/name).write_bytes(read(src))
write('input-index.json', {'files':inputs,'excluded_bulk_bindings':excluded,'checkpoint_bytes_not_opened_by_audit':True})
signal.alarm(0)
print(json.dumps(summary,indent=2))
