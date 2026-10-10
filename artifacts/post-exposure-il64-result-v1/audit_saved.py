"""Bounded stdlib saved-JSON audit/package; never loads patient arrays or weights."""
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
RUN = ROOT / 'build/post-exposure-learning-v1/IL64/attempt-01'
SUP = RUN.with_name(RUN.name + '.supervision')
SUBJECTS = ('ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045')
STEPS = (1, 14, 1, 13)
inputs = {}
checks = 0
started = time.monotonic()

def need(condition, message):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(message)

def close(a, b, name, tolerance=1e-7):
    need(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=tolerance,abs_tol=tolerance), name)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def semantic(value):
    return 'sha256:' + hashlib.sha256(canonical(value)).hexdigest()

def read(path, copy_to=None, text=False):
    path = Path(path)
    need(path.suffix in ('.json', '.txt', '.log'), 'JSON/text only')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        need(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= 4*1024**2, 'bounded regular evidence')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            data = stream.read(4*1024**2+1)
        need(len(data) == info.st_size, 'unchanged evidence size')
    finally:
        os.close(fd)
    key = str(path.relative_to(ROOT))
    digest = hashlib.sha256(data).hexdigest()
    if key in inputs:
        need(inputs[key]['sha256'] == digest, 'repeat evidence read unchanged')
    inputs[key] = {'sha256':digest, 'bytes':len(data)}
    if copy_to:
        destination = OUT/'files'/copy_to
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream: stream.write(data)
    return data.decode() if text else json.loads(data)

def save(name, value):
    with (OUT/name).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')

def keys(rows):
    return {tuple(row) for row in rows}

signal.signal(signal.SIGALRM, lambda *args: (_ for _ in ()).throw(TimeoutError('saved audit 30 seconds')))
signal.alarm(30)
result = read(RUN/'result.json', 'result.json')
receipt = read(SUP/'receipt.json', 'receipt.json')
final = read(SUP/'worker-final.json', 'worker-final.json')
read(SUP/'worker.log', 'worker.log', text=True)
declaration = read(SUP/'declaration.json', 'declaration.json')
release = read(ROOT/'build/post-exposure-learning-v1/IL64/root-release.json', 'root-release.json')
source_index = read(ROOT/release['source_index']['path'], 'source-index.json')
read(ROOT/'artifacts/post-exposure-learning-source-v1/source-index.json')
need(receipt['status'] == result['status'] == final['status'] == 'failed_or_unresolved', 'original failure retained')
need(receipt['exit_code'] == 1 and not receipt['cleanup_errors'] and not receipt['final_owned_pids'] and receipt['worker_termination_confirmed'], 'clean failed child')
need(receipt['result_sha256'] == inputs[str((RUN/'result.json').relative_to(ROOT))]['sha256'] == final['canonical_result_sha256'], 'result binding')
need(receipt['release_sha256'] == inputs['build/post-exposure-learning-v1/IL64/root-release.json']['sha256'], 'release binding')
need(release['source_index'] == receipt['source_index'] == declaration['source_index'], 'source index joins')
need(release['source_index']['sha256'] == inputs[release['source_index']['path']]['sha256'], 'source index digest')
need(receipt['elapsed_seconds'] < receipt['caps']['hard_total_wall_seconds'] and receipt['sampled_peak_rss_bytes'] < receipt['caps']['sampled_owned_tree_rss_bytes'], 'resource caps')
need(result['TRAIN'] == list(SUBJECTS) and result['optimizer_updates'] == {'IL':64,'RL':0}, 'all-four denominator and updates')
need(result['teacher_decisions'] == result['teacher_logit_forwards'] == 29 and result['teacher_trace_reuses'] == 256, 'teacher counts')
need(result['loss_forward_calls'] == 1856 and result['checkpoint_loads'] == 1 and result['search_calls'] == result['private_reference_reads'] == 0 and result['SELECT_EVAL_opened'] is False, 'execution counts')
need(result['selection_readiness'] == {'execution_admitted':False,'ready':False}, 'no completion admission')
reload = read(RUN/'checkpoint-reload.json', 'checkpoint-reload.json')
need(reload['sha256'] == result['checkpoints']['IL']['sha256'] and reload['parameter_hash'] == result['checkpoints']['IL']['parameter_hash'] and reload['completed_updates'] == 64 and reload['exact_parameter_match'], 'saved checkpoint metadata')
dynamics = read(RUN/'training-dynamics.json', 'training-dynamics.json')
need(inputs[str((RUN/'training-dynamics.json').relative_to(ROOT))]['sha256'] == result['training_dynamics_sha256'], 'dynamics digest')
need(len(dynamics['updates']) == 64, '64 dynamics rows')
previous = result['initial_parameter_hash']; updates = []
for i,d in enumerate(dynamics['updates'],1):
    update_path = RUN/'IL'/f'update-{i:02d}'/'update.json'
    update = read(update_path)
    need(d['update_record_sha256'] == inputs[str(update_path.relative_to(ROOT))]['sha256'], 'update digest')
    need(update['completed_updates'] == d['completed_updates'] == i and update['optimizer_updates'] == 1 and update['loss_forward_calls'] == d['loss_forward_calls'] == 29, 'one shared complete update')
    need(update['before_parameter_hash'] == d['before_parameter_hash'] == previous and update['after_parameter_hash'] == d['after_parameter_hash'], 'parameter hash chain')
    need(update['teacher_group_counts'] == {'STOP':4,'motion':25} and update['teacher_group_weights'] == {'STOP':.125,'motion':.02}, 'balanced group weights')
    close(d['pre_update_loss'],update['loss'],'loss dynamics')
    if i > 1: close(d['loss_change_from_previous'],d['pre_update_loss']-dynamics['updates'][i-2]['pre_update_loss'],'loss delta')
    contributions = [read(update_path.parent/s/'gradient-contribution.json') for s in SUBJECTS]
    need([r['steps'] for r in contributions] == list(STEPS) and sum(r['loss_forward_calls'] for r in contributions)==29, 'all-four update contributions')
    close(math.fsum(r['loss'] for r in contributions), update['loss'], 'global weighted loss')
    previous = update['after_parameter_hash'];updates.append({'update':update,'contributions':contributions})
need(previous == reload['parameter_hash'], 'final parameter endpoint')
save('update-summaries.json', updates)
readouts=[]
for subject,count in zip(SUBJECTS,STEPS):
    for step in range(count):
        row=read(RUN/'teacher-readout'/subject/f'state-{step:02d}.json',f'teacher-readout/{subject}/state-{step:02d}.json')
        scores=row['scores']; mask=row['action_mask']; logits=scores['logits']; label=row['action_ids'].index(row['teacher_action'])
        legal=[i for i,v in enumerate(mask) if v]; ranking=sorted(legal,key=lambda i:(-logits[i],i))
        peak=max(logits[i] for i in legal); denominator=math.fsum(math.exp(logits[i]-peak) for i in legal)
        probabilities=[math.exp(logits[i]-peak)/denominator if mask[i] else 0. for i in range(len(mask))]
        need(mask[label] and len(probabilities)==len(scores['probabilities']), 'legal teacher')
        for a,b in zip(probabilities,scores['probabilities']): close(a,b,'saved softmax',2e-6)
        ce=math.log(denominator)+peak-logits[label]
        close(ce,scores['teacher_CE'],'teacher CE',2e-6)
        need(scores['teacher_rank']==ranking.index(label)+1 and scores['greedy_action']==row['action_ids'][ranking[0]],'rank/greedy')
        close(scores['teacher_probability'],probabilities[label],'teacher probability',2e-6)
        readouts.append(row)
stop=[r['scores']['teacher_CE'] for r in readouts if r['teacher_action']=='STOP']
motion=[r['scores']['teacher_CE'] for r in readouts if r['teacher_action']!='STOP']
metrics=read(RUN/'endpoint-teacher-metrics.json','endpoint-teacher-metrics.json')
need(len(stop)==4 and len(motion)==25 and metrics==result['endpoint_teacher_metrics'],'readout endpoint joins')
close(metrics['STOP_mean_NLL'],math.fsum(stop)/4,'STOP mean')
close(metrics['motion_mean_NLL'],math.fsum(motion)/25,'motion mean')
close(metrics['balanced_mean_CE'],.5*math.fsum(stop)/4+.5*math.fsum(motion)/25,'balanced CE')
close(metrics['teacher_action_accuracy'],sum(r['scores']['greedy_action']==r['teacher_action'] for r in readouts)/29,'accuracy')
routes=[]
for subject in SUBJECTS:
    folder=RUN/'TRAIN-greedy'/'IL'/subject
    trace=read(folder/'complete-trace.json'); plan=read(folder/'plan.json')
    m=trace['metrics'];history=m['history'];actions=[r['action_id'] for r in history]
    need(plan['plan_seal']==semantic(plan['plan']) and plan['plan']['history']==history and plan['plan']['actions']==actions,'complete sealed plan')
    need(plan['plan']['parameter_hash']==reload['parameter_hash'] and plan['plan']['learning_updates']==64,'fixed endpoint plan')
    need(m['terminated'] and len(history)==m['steps'] and (actions[-1]=='STOP' or len(history)==24),'complete STOP/horizon')
    removed=set();contacts=set();action_summaries=[]
    for n,(r,decision) in enumerate(zip(history,trace['decisions'])):
        need(decision['action_id']==r['action_id'] and decision['action_mask'][decision['action_ids'].index(r['action_id'])],'legal recorded choice')
        close(decision['reward'],r['reward'],'decision reward')
        cells=keys(r.get('removed_indices_native',[])); contact=keys(r.get('contact_indices_native',[]))
        need(not (removed & cells),'no double removal');removed|=cells;contacts|=contact
        if r['action_id']!='STOP':
            need(cells==set().union(*(keys(q['removed_indices_native']) for q in r['microsteps'])),'micro/macro removal')
            need(contact==set().union(*(keys(q['contact_indices_native']) for q in r['microsteps'])),'micro/macro contact')
        action_summaries.append({k:r.get(k) for k in ('action_id','tool_id','reward','target_removed_mm3','normal_removed_mm3','complete_tool_path_length_mm','geometry_unknowns')} | {'removed_cells':len(cells),'contact_cells':len(contact)})
    for key in ('reward','target_removed_mm3','normal_removed_mm3'):
        close(math.fsum(r.get(key,0.) for r in history),m['total_reward' if key=='reward' else key], 'route totals')
    teacher=read(ROOT/'build/remind-post-exposure-feasibility-v1/attempt-01'/subject/'episode-metrics.json')
    teacher_removed=set().union(*(keys(r.get('removed_indices_native',[])) for r in teacher['history']))
    independent=None
    if subject!='ReMIND-045':
        replay=read(folder/'native-replay.json')
        independent=replay['independent_geometry']; need(independent['accepted'] and independent['complete_episode'],'original independent pass')
        with (OUT/'files'/f'{subject}-independent-geometry.json').open('x') as stream: json.dump(independent,stream,indent=2,sort_keys=True);stream.write('\n')
    else:
        need(not (folder/'native-replay.json').exists(),'original missing rejected certificate retained')
    routes.append({'subject':subject,'terminal_reason':plan['plan']['terminal_reason'],'steps':len(history),'actions':actions,
        'independent_accepted':independent['accepted'] if independent else False,
        'independent_failure_reason':None if independent else 'not persisted by original worker',
        'target_removed_mm3':m['target_removed_mm3'],'outside_target_removed_mm3':m['normal_removed_mm3'],
        'public_return':m['total_reward'],'removed_cells':len(removed),'contact_cells':len(contacts),'retained_contact_cells':len(contacts-removed),
        'teacher_actions':[r['action_id'] for r in teacher['history']], 'teacher_public_return':teacher['total_reward'],
        'teacher_target_removed_mm3':teacher['target_removed_mm3'],'same_removed_cell_union_as_teacher':removed==teacher_removed,
        'plan_seal':plan['plan_seal'],'action_summaries':action_summaries})
need(result['total_policy_forward_calls']==1856+29+sum(r['steps'] for r in routes)==1925,'partial-attempt complete actual forward cost')
need(result['completed_source_visits']==7 and len(result['TRAIN_greedy'])==3,'failed eighth visit not falsely complete')
save('route-summary.json',routes)
save('input-index.json',inputs)
summary={'status':'saved_scalar_audit_pass_original_endpoint_failed','checks':checks,
    'original_045_independent_acceptance':False,'original_045_failure_reason_saved':False,
    'checkpoint':result['checkpoints']['IL'],'checkpoint_bytes_opened':False,
    'loss_first':dynamics['updates'][0]['pre_update_loss'],'loss_last':dynamics['updates'][-1]['pre_update_loss'],
    'gradient_max':max(r['gradient_norm_before_clip'] for r in dynamics['updates']),
    'clipped_updates':sum(r['gradient_exceeds_clip_threshold'] for r in dynamics['updates']),
    'teacher_metrics':metrics,'runtime_seconds':receipt['elapsed_seconds'],'rss_bytes':receipt['sampled_peak_rss_bytes'],
    'actual_forwards':1925,'native_preview_entries':result['native_preview_entries'],
    'all_four_denominator':list(SUBJECTS),'scope':'saved scalar/hash consistency; no mask geometry, tensor equality or new native replay',
    'audit_wall_seconds':time.monotonic()-started,'audit_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
save('audit-result.json',summary)
signal.alarm(0)
print(json.dumps(summary,sort_keys=True))
