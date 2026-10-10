"""One fixed public ray after every existing legal root opening; metadata only."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUTPUT='build/union025-clearance-witness-v1/attempt-01'
SUBJECT='ReMIND-025';TRAIN=(SUBJECT,)
CONDITION='cerebrum_plus_supplied_tumor_assumption'
CLOSED={'SELECT':['ReMIND-013','ReMIND-037'],'EVAL':['ReMIND-067']}
WORKER_SECONDS=180;PARENT_SECONDS=210;MEMORY_BYTES=3*1024**3
OUTPUT_BYTES=64*1024**2;SUPERVISION_BYTES=16*1024**2
INPUT_INDEX='build/union025-clearance-witness-v1/input-index.json'
INPUT_SHA='c4e9a35729f74dc40d1e0d4120fda239bdf6bc22fe009d446a10fb6b913b6e90'
EXECUTION={'root_preparations':15,'complete_histories':16,'task_max_steps':24,
    'max_actions_per_branch':3,'candidate_cap':120,'max_native_previews':10000,
    'max_native_transition_calls':92,'max_rollout_calls':46,'max_replay_calls':46,
    'max_fixed_ray_diagnostic_previews':16,'obstruction_cell_limit':4096,
    'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0,'search_calls':0,'threads':1}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def semantic(value):return 'sha256:'+hashlib.sha256(canonical(value).encode()).hexdigest()
def small(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=4*1024**2:
        raise ValueError('Bounded regular JSON required: '+str(path))
    return json.loads(path.read_text())
def physical(row):return {k:row[k] for k in ('tool_id','entry_mm','tip_mm')}
def same_ray(row,ray):return canonical(physical(row))==canonical(physical(ray))

def inputs():
    path=ROOT/INPUT_INDEX
    if sha(path)!=INPUT_SHA:raise ValueError('Input index changed')
    refs=small(path)['files'];records={}
    for key,ref in refs.items():
        p=ROOT/ref['path']
        if sha(p)!=ref['sha256']:raise ValueError('Pinned metadata changed: '+key)
        records[key]=small(p)
    old=records['paired_result'];inventory=records['union_inventory'];plan=records['raw_plan']
    if (old['status']!='complete_paired_TRAIN_search' or records['paired_receipt']['status']!='complete'
            or records['paired_receipt']['result_sha256']!=refs['paired_result']['sha256']
            or old['arms'][7]['subject']!=SUBJECT or old['arms'][7]['condition']!=CONDITION
            or semantic(plan['plan'])!=plan['plan_seal']):raise ValueError('Completed paired evidence required')
    moves=[r for r in plan['plan']['history'] if r['action_id']!='STOP']
    if len(moves)!=1 or moves[0]['reward']<=0 or moves[0]['target_removed_mm3']<=0:
        raise ValueError('Exactly one previously useful fixed ray required')
    ray=physical(moves[0]);matching=[r for r in inventory['emitted'] if same_ray(r,ray)]
    legal=[r for r in inventory['emitted'] if r['feasible']]
    if (len(legal)!=15 or inventory['omitted_count']!=0 or inventory['candidate_cap']!=120
            or len(matching)!=1 or matching[0]['feasible'] is not False
            or matching[0]['reason']!='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE'):
        raise ValueError('Exact union025 root inventory/fixed refusal required')
    if (records['public_manifest']['patient_id']!=SUBJECT or records['public_manifest']['role']!='TRAIN'):
        raise ValueError('Only fixed TRAIN025 source is released')
    return records,refs,ray,legal

def release_template(head,index_sha):
    records,refs,ray,legal=inputs()
    return {'version':'union025-depth-two-fixed-ray-release-v1','status':'pending_root_release',
        'expected_head':head,'output':OUTPUT,'TRAIN':list(TRAIN),'closed_roles':CLOSED,
        'SELECT_EVAL_execution':False,'attempts':1,'automatic_retry':False,
        'parent_seconds':PARENT_SECONDS,'worker_seconds':WORKER_SECONDS,'memory_bytes':MEMORY_BYTES,
        'output_bytes':OUTPUT_BYTES,'supervision_bytes':SUPERVISION_BYTES,'execution_limits':EXECUTION,
        'condition':CONDITION,'condition_admission':records['paired_release']['condition_admission'][CONDITION],
        'fixed_ray':ray,'root_preparations':legal,
        'input_index':{'path':INPUT_INDEX,'sha256':INPUT_SHA},
        'source_index':{'path':'build/union025-clearance-witness-v1/source-index.json','sha256':index_sha},
        'scope':'All15 existing legal root openings, each followed by one previously useful physical ray if emitted and legal, then STOP',
        'selection':'Highest unchanged complete cumulative reward; baseline STOP wins equality',
        'limitations':'Depth-two fixed-ray witness only; no global reachability/optimality or anatomical correctness claim',
        'rejected_previews':'Temporary cuts are diagnostic only and never committed; failed ray is distinct from the completed prep+STOP history',
        'proposal_unavailable':'Native-feasible geometry is not injected when absent or illegal in the current task inventory',
        '010_omitted_rays':'deferred; no source or geometry access in this attempt',
        'private_reference_reads':0,'model_forwards':0,'optimizer_updates':0,'checkpoint_loads':0}

def validate_release(release):
    expected=release_template(release.get('expected_head'),release.get('source_index',{}).get('sha256'))
    expected['status']='released_one_attempt'
    if canonical(release)!=canonical(expected):raise ValueError('Exact one-attempt witness release required')
    for value,n in ((release['expected_head'],40),(release['source_index']['sha256'],64)):
        if not isinstance(value,str) or len(value)!=n or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid source/HEAD binding')

def source_guard(release,release_path,release_sha):
    validate_release(release)
    if sha(release_path)!=release_sha:raise ValueError('Release changed')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    ref=release['source_index']
    if head!=release['expected_head'] or sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('HEAD/source index changed')
    index=small(ROOT/ref['path'])
    if index['head']!=head or index['version']!='union025-clearance-witness-runtime-v1':raise ValueError('Wrong runtime index')
    for section in ('source_files','metadata_files'):
        for relative,digest in index[section].items():
            p=Path(relative)
            if p.is_absolute() or '..' in p.parts or (ROOT/p).is_symlink() or sha(ROOT/p)!=digest:
                raise ValueError('Bound source/metadata changed: '+relative)
    return index

def complete_result(result):
    rows=result.get('branches',[])
    return (result.get('status')=='complete_fixed_ray_witness' and len(rows)==16
        and [r.get('ordinal') for r in rows]==list(range(16))
        and all(r.get('status')=='complete' for r in rows)
        and result.get('source_released') is True and result.get('SELECT_EVAL_opened') is False
        and all(type(result.get(k)) is int and result[k]==0 for k in
            ('policy_forwards','optimizer_updates','checkpoint_loads','private_reference_reads','search_calls')))

def endpoint_control(output,release):
    root=Path(output);result=small(root/'result.json');costs=small(root/'costs.json')
    if not complete_result(result):raise ValueError('Incomplete15-branch plus STOP denominator')
    counts=costs['native_counts'];records,refs,ray,legal=inputs();world=small(root/'world.json')
    if (counts!=result['native_counts'] or set(counts)!={'rollout','replay'}
            or any(type(n) is not int or not 0<=n<=46 for n in counts.values())
            or sum(counts.values())>92 or costs['diagnostic_previews']!=16
            or costs['native_budget']['native_preview_entries']>10000
            or costs['prohibited_calls']!={'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0,'search_calls':0}):
        raise ValueError('Execution accounting differs')
    old=records['union_world']
    for key in ('source_hash','decision_model_hash','initial_observation_hash'):
        if world[key]!=old[key]:raise ValueError('Fixed union world changed')
    if canonical(small(root/'initial-inventory.json'))!=canonical(records['union_inventory']):
        raise ValueError('Original inventory changed')
    rows=[]
    for i,row in enumerate(result['branches']):
        directory=root/f'branch-{i:02d}';plan=small(directory/'plan.json');replay=small(directory/'replay.json')
        body=plan['plan'];audit=replay['independent_geometry'];metrics=replay['metrics']
        actions=body['actions'];expected_first='STOP' if i==0 else legal[i-1]['action_id']
        if (actions[0]!=expected_first or actions[-1]!='STOP' or not 1<=len(actions)<=3
                or (i==0 and actions!=['STOP']) or (i>0 and len(actions)<2)
                or semantic(body)!=plan['plan_seal'] or row['plan_seal']!=plan['plan_seal']
                or row['actions']!=actions or body['source_hash']!=world['source_hash']
                or body['decision_model_hash']!=world['decision_model_hash'] or body['context_hash']!=world['context_hash']
                or body['initial_observation_hash']!=world['initial_observation_hash']
                or body['max_steps']!=24 or body['learning_updates']!=0 or body['parameter_hash'] is not None
                or canonical(body['history'])!=canonical(metrics['history']) or metrics['terminated'] is not True
                or audit['accepted'] is not True or audit['complete_episode'] is not True
                or audit['committed_history_hash']!=semantic(metrics['history'])
                or audit['source_hash']!=world['source_hash'] or audit['decision_model_hash']!=world['decision_model_hash']
                or canonical(row['outcomes'])!=canonical(audit['outcomes'])):
            raise ValueError('Complete branch seal/replay differs')
        diagnostic=small(directory/'fixed-ray-preview.json')
        if not same_ray(diagnostic,ray) or diagnostic['committed'] is not False:
            raise ValueError('Diagnostic ray or temporary-cut scope differs')
        if i==0 and (diagnostic['feasible'] or row['ray_status']!='baseline_blocked'):
            raise ValueError('Baseline fixed ray was not refused')
        if i>0:
            expected='executed' if len(actions)==3 else ('blocked' if not diagnostic['feasible'] else 'proposal_unavailable')
            if row['ray_status']!=expected:raise ValueError('Native feasibility/action availability conflated')
            if len(actions)==3 and not same_ray(body['history'][1],ray):raise ValueError('Executed ray geometry differs')
        rows.append({'ordinal':i,'plan_seal':plan['plan_seal'],'plan_sha256':sha(directory/'plan.json'),
            'replay_sha256':sha(directory/'replay.json'),'diagnostic_sha256':sha(directory/'fixed-ray-preview.json')})
    rewards=[r['outcomes']['total_reward'] for r in result['branches']]
    selected=max(range(16),key=lambda i:rewards[i])
    if rewards[0]!=0 or result['selected_ordinal']!=selected or result['selected_plan_seal']!=result['branches'][selected]['plan_seal']:
        raise ValueError('Actual-return STOP-first selection differs')
    if counts['rollout']!=sum(len(r['actions']) for r in result['branches']) or counts['replay']!=counts['rollout']:
        raise ValueError('Committed and replay action counts differ')
    return {'status':'all15_preparations_and_STOP_bound','branches':rows,'native_counts':counts,
        'selected_ordinal':selected,'selected_return':rewards[selected],
        'fixed_ray_diagnostics':16,'zero_model_work':True,'full_target_denominator_unchanged':True}
