"""Terminal RL8 source/JSON evidence only. No weights, arrays or reruns."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'build/obstruction-opening-learning-v1';RUN=BASE/'RL8';OUT=RUN/'attempt-01'
PACKAGE=Path(__file__).resolve().parent/'artifacts/obstruction-opening-learning-rl8-v1'
TRAIN=('ReMIND-008','ReMIND-010','ReMIND-020','ReMIND-025')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    assert path.is_file() and not path.is_symlink() and path.stat().st_size<=2*1024**2
    return json.loads(path.read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
refs=read(Path(__file__).resolve().parent/'review-references.json')
for row in refs['files']:
    p=ROOT/row['source_path'];assert sha(p)==row['sha256']
audit=read(ROOT/'build/obstruction-opening-learning-independent-v1/RL8/audit-result.json')
assert audit['status']=='PASS_saved_RL8_metadata_scalar_audit' and audit['checks']==6132 and audit['input_count']==302
result=read(OUT/'result.json');parent=read(RUN/'attempt-01.supervision/receipt.json');release=read(RUN/'root-release.json')
assert sha(OUT/'result.json')=='60631a2fb4cebda2d6b29475156e487a8b51b4bd1c5cc1d6ca89dd2511e29897'==parent['result_sha256']
assert result['status']=='complete_matched_TRAIN_endpoint' and parent['status']=='complete'
assert not parent['cleanup_errors'] and not parent['final_owned_pids'] and parent['worker_termination_confirmed']
assert result['optimizer_updates']=={'IL':0,'RL':8} and result['checkpoint_loads']==1
assert all(result['TRAIN_greedy'][s]['actions']==['STOP'] and result['TRAIN_greedy'][s]['public_return']==0 for s in TRAIN)
assert not PACKAGE.exists();PACKAGE.mkdir(parents=True);entries=[]
def copy(source,target):
    assert source.is_file() and not source.is_symlink() and source.stat().st_size<=2*1024**2
    assert source.suffix not in ('.psckpt','.npz','.npy')
    raw=source.read_bytes();dest=PACKAGE/target;dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:f.write(raw)
    entries.append({'path':target,'source_path':str(source.relative_to(ROOT)),'bytes':len(raw),'sha256':sha(dest),'kind':'exact_copy'})
for name in ('pilot_contract.py','cohort_worker.py','run_owned.py','freeze-runtime.py','test_generated_worker_flow.py'):
    copy(BASE/name,'source/'+name)
for name in ('receipt.json','console.txt'):copy(BASE/'generated-attempt-01'/name,'generated-control/'+name)
copy(BASE/'PREPARATION.json','source/PREPARATION.json')
for name in ('root-release.json','release-template.json','source-index.json'):copy(RUN/name,'provenance/'+name)
for name in ('receipt.json','worker-final.json','endpoint-control.json','declaration.json'):
    copy(RUN/'attempt-01.supervision'/name,'supervision/'+name)
for name in ('result.json','configuration.json','costs.json','training-dynamics.json','checkpoint-reload.json','teacher-pins.json','endpoint-teacher-metrics.json'):
    copy(OUT/name,'result/'+name)
for subject in TRAIN:
    copy(OUT/(subject+'-context.json'),'contexts/'+subject+'.json')
    for group in ('teachers','TRAIN-greedy/RL'):
        for name in ('plan.json','native-replay.json'):copy(OUT/group/subject/name,group+'/'+subject+'/'+name)
    for step in range(4 if subject=='ReMIND-025' else 1):
        name=f'state-{step:02d}.json';copy(OUT/'teacher-readout'/subject/name,'teacher-readout/'+subject+'/'+name)
episodes=[];update_rows=[]
for update in range(1,9):
    stem=f'RL/update-{update:02d}';update_path=OUT/stem/'update.json';receipt=read(update_path)
    copy(update_path,stem+'/update.json');update_rows.append(receipt)
    for subject in TRAIN:
        name=stem+'/'+subject+'/gradient-contribution.json';path=OUT/name;row=read(path);data=row['rl_decision_diagnostics'];decisions=data['decisions']
        assert data['behavior_parameter_hash']==receipt['before_parameter_hash']
        assert len(decisions)==row['steps']==row['loss_forward_calls']
        assert all(d['step']==i and d['action_id']==row['actions'][i] for i,d in enumerate(decisions))
        copy(path,name)
        episodes.append({'update':update,'subject':subject,'steps':row['steps'],'undiscounted_return':sum(d['reward'] for d in decisions),
            'recorded_return_to_go':row['return'],'before_parameter_hash':data['behavior_parameter_hash'],
            'after_shared_update_parameter_hash':receipt['after_parameter_hash'],'trace_seal':row['trace_seal'],'plan_seal':row['plan_seal'],
            'positive_reward_decisions':sum(d['reward']>0 for d in decisions),
            'positive_reward_negative_advantage_decisions':sum(d['reward']>0 and d['detached_advantage']<0 for d in decisions),
            'negative_reward_decisions':sum(d['reward']<0 for d in decisions),'STOP_decisions':sum(d['action_id']=='STOP' for d in decisions),
            'diagnostics_path':name,'diagnostics_sha256':sha(path)})
assert len(episodes)==32 and sum(e['steps'] for e in episodes)==320
for row in refs['files']:copy(ROOT/row['source_path'],row['package_path'])
copy(Path(__file__).resolve().parent/'review-references.json','provenance/review-references.json')
copy(BASE/'IL64/attempt-01/result.json','matched-IL64/result.json')
copy(BASE/'IL64/attempt-01.supervision/receipt.json','matched-IL64/receipt.json')
copy(Path(__file__),'source/assemble.py')
checkpoint={'scope':'metadata only; ignored checkpoint payload deliberately excluded','source_result_sha256':sha(OUT/'result.json'),
    'source_path':str((OUT/result['checkpoints']['RL']['path']).relative_to(ROOT)),**result['checkpoints']['RL'],
    'initial_parameter_hash':result['initial_parameter_hash'],'learning_protocol_hash':release['learning_protocol_hash'],
    'completed_updates':8,'reload_record':'result/checkpoint-reload.json'}
write(PACKAGE/'checkpoint-metadata.json',checkpoint)
costs=read(OUT/'costs.json');dynamics=read(OUT/'training-dynamics.json')['updates']
counts={'shared_RL_updates':8,'IL_updates':0,'fresh_episodes':32,'loss_decisions':320,'collection_forwards':320,
    'teacher_readout_forwards':7,'greedy_forwards':4,'total_policy_forwards':651,'source_visits':44,
    'cache_reuses':0,'native_previews':costs['native_budget']['native_preview_entries'],'checkpoint_reloads':1,
    'positive_return_episodes':sum(e['recorded_return_to_go']>0 for e in episodes),
    'zero_return_episodes':sum(e['recorded_return_to_go']==0 for e in episodes),
    'negative_return_episodes':sum(e['recorded_return_to_go']<0 for e in episodes),
    'positive_reward_decisions':sum(e['positive_reward_decisions'] for e in episodes),
    'positive_reward_negative_advantage_decisions':sum(e['positive_reward_negative_advantage_decisions'] for e in episodes)}
assert costs['total_policy_forward_calls']==651 and result['completed_source_visits']==44
write(PACKAGE/'episode-summary.json',{'scope':'same-forward saved decision diagnostics, no re-evaluation','counts':counts,'episodes':episodes,
    'credit_interpretation':'Immediate positive reward need not imply positive return-to-go or actor advantage. Counts are descriptive; they do not isolate the cause of the final policy.'})
il=read(BASE/'IL64/attempt-01/result.json');il_parent=read(BASE/'IL64/attempt-01.supervision/receipt.json')
assert il['initial_parameter_hash']==result['initial_parameter_hash']
summary={'version':'obstruction-opening-learning-rl8-v1','status':result['status'],'TRAIN':list(TRAIN),
    'scope':'fixed-four TRAIN scratch RL8 endpoint on explicit S OR T occupancy/obstruction-opening world',
    'SELECT_EVAL_executed':False,'counts':counts,'greedy_routes':result['TRAIN_greedy'],
    'endpoint_teacher_metrics':result['endpoint_teacher_metrics'],'teacher_top1':{'correct':4,'total':7},
    'runtime':{'parent_seconds':parent['elapsed_seconds'],'worker_seconds':result['complete_wall_seconds'],
        'sampled_peak_rss_bytes':parent['sampled_peak_rss_bytes'],'owned_output_bytes':parent['output_bytes'],
        'caps':parent['caps'],'cleanup_errors':parent['cleanup_errors'],'final_owned_pids':parent['final_owned_pids']},
    'source_head':release['expected_head'],'release_sha256':sha(RUN/'root-release.json'),'result_sha256':sha(OUT/'result.json'),
    'checkpoint':checkpoint,'reviews':refs,'independent_audit':{'checks':audit['checks'],'input_count':audit['input_count'],'status':audit['status']},'dynamics':{'updates':len(dynamics),'first_pre_update_loss':dynamics[0]['pre_update_loss'],
        'final_pre_update_loss':dynamics[-1]['pre_update_loss'],'first_gradient_norm':dynamics[0]['gradient_norm_before_clip'],
        'final_gradient_norm':dynamics[-1]['gradient_norm_before_clip'],'updates_exceeding_clip_threshold':sum(d['gradient_exceeds_clip_threshold'] for d in dynamics),
        'trajectory_path':'result/training-dynamics.json'},
    'matched_IL64_comparison':{'same_initial_parameter_hash':il['initial_parameter_hash'],
        'IL_result_sha256':sha(BASE/'IL64/attempt-01/result.json'),'IL025_return':il['TRAIN_greedy']['ReMIND-025']['public_return'],
        'RL025_return':0.,'IL_parent_seconds':il_parent['elapsed_seconds'],'RL_parent_seconds':parent['elapsed_seconds'],
        'equal_compute_claim':False,'different_fixed_update_counts_and_training_objectives':True},
    'omissions':['checkpoint payload','patient arrays/images','repeated inventory/source files','full per-episode native histories'],
    'omitted_evidence_bindings':'independent audit input hashes and saved episode trace/plan seals; all 320 scalar credit rows and8 update receipts retained',
    'interpretation':'Negative scratch-RL endpoint on fixed TRAIN; no held-out, physical/material, clinical, or equal-compute claim'}
write(PACKAGE/'summary.json',summary)
readme='''# Scratch RL8 on the union-occupancy obstruction-opening world

All four reloaded greedy TRAIN plans chose STOP immediately: return 0, supplied-target removal 0 and outside-target removal 0. This is a completed negative learning result, not a successful task solution. Teacher-action top-1 agreement is 4/7 because the four teacher STOP labels rank first; all three motion labels do not.

The fixed endpoint ran eight shared RL updates over 32 freshly collected episodes and 320 decisions, with no imitation update, cached-gradient reuse or new search. Each episode's behavior tensor hash equals its shared update's starting tensor hash. The existing same-forward diagnostics retain reward, return-to-go, value, detached advantage, chosen log probability, entropy and actor/value terms for every decision. There were 29 negative-return episodes, three zero-return episodes and no positive-return episode. Eleven individual decisions had positive immediate reward; four of those had negative advantage. These describe the observed training signal and do not establish an isolated cause of the final STOP policy. The independent terminal diagnosis is included separately.

Complete accounting is 320 collection forwards +320 loss forwards +7 teacher readouts +4 final greedy forwards =651; 44 sequential source visits; 80,356 native previews; one authenticated checkpoint reload. Parent elapsed 987.010701 seconds (worker 983.245890), sampled peak RSS 2,527,870,976 bytes, with no cleanup errors or survivors. The original 41,354,524 output bytes remain in ignored local storage. This package retains the compact scalar records and bindings rather than repeated inventories.

The matched IL64 endpoint used the same model, fresh initial tensors, seed and public world, but a different fixed update count and objective. IL64 matched the positive025 search outcome (+2.7583290722) while this RL8 endpoint stopped there. Their measured parent costs were 161.917943 and 987.010701 seconds respectively; this is not an equal-compute comparison. Both use the explicit unvalidated S OR (T > 0) removable-occupancy assumption, h24/cap120, footprint and obstruction-opening proposals, and global public target context. No SELECT/EVAL or newer partial-domain cohort was executed.

`MANIFEST.json` maps every copied file to its original path/hash. It includes source/tests, release/closure and terminal receipts, results/costs/dynamics, all eight update receipts, all 32 compact episode diagnostics, seven teacher readouts, complete final and teacher plans/native replays, checkpoint metadata, independent audit and terminal diagnosis. Weight payloads and patient arrays are absent. The independent saved audit passed 6,132 checks across 302 input bindings, including the full behavior/update chain and credit arithmetic. Audit limitations remain explicit; recorded tensor hashes are not independent tensor recomputation. Collection/replay cost scopes include nested backward time and must not be added as disjoint costs.

Copied scripts/tests are archival and preserve original relative-path assumptions. Do not run them in-place; reproduction requires the manifest's original layout and a separate owned release. This package neither admits held-out execution nor makes a clinical, anatomical/material, generalization or equal-compute claim.
'''
(PACKAGE/'README.md').write_text(readme)
for name in ('checkpoint-metadata.json','episode-summary.json','summary.json','README.md'):
    p=PACKAGE/name;entries.append({'path':name,'source_path':None,'bytes':p.stat().st_size,'sha256':sha(p),'kind':'derived_from_packaged_evidence'})
manifest={'version':'compact-result-manifest-v1','destination':'artifacts/obstruction-opening-learning-rl8-v1',
    'files':sorted(entries,key=lambda x:x['path']),'file_count':len(entries),'total_bytes':sum(x['bytes'] for x in entries),
    'payloads_excluded':True,'copy_verification':'all copied source/JSON bytes, lengths and SHA256 compared with saved originals'}
write(PACKAGE/'MANIFEST.json',manifest)
for row in entries:
    p=PACKAGE/row['path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
    if row['source_path']:assert p.read_bytes()==(ROOT/row['source_path']).read_bytes()
check={'status':'PASS_exact_saved_package_parity','manifest_sha256':sha(PACKAGE/'MANIFEST.json'),
    'file_count':len(entries),'total_bytes_excluding_manifest':manifest['total_bytes'],
    'total_bytes_including_manifest':manifest['total_bytes']+(PACKAGE/'MANIFEST.json').stat().st_size,
    'checkpoint_or_patient_payloads_read':False,'destination':str(PACKAGE.relative_to(ROOT))}
write(Path(__file__).resolve().parent/'package-check.json',check);print(json.dumps(check,indent=2))
