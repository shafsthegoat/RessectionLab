"""Fixed eight-arm TRAIN occupancy search contract; stdlib metadata only."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUTPUT='build/paired-train-occupancy-search-v1/attempt-01'
TRAIN=('ReMIND-008','ReMIND-010','ReMIND-020','ReMIND-025')
CONDITIONS=('raw_cerebrum_baseline','cerebrum_plus_supplied_tumor_assumption')
ARMS=tuple({'subject':s,'condition':c} for s in TRAIN for c in CONDITIONS)
CLOSED={'SELECT':['ReMIND-013','ReMIND-037'],'EVAL':['ReMIND-067']}
WORKER_SECONDS=600;PARENT_SECONDS=660;MEMORY_BYTES=3*1024**3
OUTPUT_BYTES=64*1024**2;SUPERVISION_BYTES=16*1024**2
INPUT_INDEX='build/paired-train-occupancy-search-v1/input-index.json'
INPUT_SHA='2ca8758aa64d17b4490ad4753e79a66b227cdc3992fc085bc8ef187e4fb42a95'
EXECUTION={'arms':8,'max_steps':24,'candidate_cap':120,'search_seconds_per_arm':45,
    'max_native_previews':100000,'max_search_commits':192,'max_execution_replay_commits':384,
    'max_native_transition_calls':576,'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0,
    'threads':1,'search_method':'observed_greedy_search'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def semantic(value):return 'sha256:'+hashlib.sha256(canonical(value).encode()).hexdigest()
def small(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=4*1024**2:
        raise ValueError('Bounded regular JSON required: '+str(path))
    return json.loads(path.read_text())
def inputs():
    path=ROOT/INPUT_INDEX
    if sha(path)!=INPUT_SHA:raise ValueError('Input index changed')
    refs=small(path)['files'];records={}
    for key,ref in refs.items():
        p=ROOT/ref['path']
        if sha(p)!=ref['sha256']:raise ValueError('Pinned input metadata changed: '+key)
        records[key]=small(p)
    recipe=records['recipe'];baseline=records['release'];rows=records['public_index']['cases']
    if (recipe['inputs']['subjects_in_fixed_order']!=list(TRAIN)
            or [r['id'] for r in recipe['conditions']]!=list(CONDITIONS)
            or len(rows)!=4 or {r['patient_id'] for r in rows}!=set(TRAIN)
            or any(r['role']!='TRAIN' for r in rows)
            or baseline['learning_protocol']['cohort_execution']['proposal_config']['max_candidates']!=120
            or baseline['limits']['max_steps']!=24):raise ValueError('Exact paired TRAIN recipe required')
    for subject in TRAIN:
        plan=records['teacher_'+subject]
        if semantic(plan['plan'])!=plan['plan_seal']:raise ValueError('Baseline plan seal changed')
    return records,refs

def condition_admission(condition):
    records,_=inputs();old=records['release']
    limits=json.loads(canonical(old['limits']))
    protocol_hash=old['learning_protocol_hash']
    if condition==CONDITIONS[1]:
        limits.update(max_policy_forwards=0,max_optimizer_updates=0)
        protocol_hash=semantic({'scope':'TRAIN_search_only_occupancy_assumption',
            'recipe_sha256':inputs()[1]['recipe']['sha256'],'condition':condition,'execution':EXECUTION,
            'baseline_public_protocol_hash':old['learning_protocol_hash']})
    elif condition!=CONDITIONS[0]:raise ValueError('Unknown occupancy condition')
    return {'limits':limits,'learning_protocol_hash':protocol_hash,
        'initialization_performed':False,'actual_model_optimizer_checkpoint_calls':0,
        'raw_envelope_note':'historical raw positive admission envelope retained; runner forbids actual model work'}

def release_template(head,index_sha):
    records,refs=inputs()
    return {'version':'paired-TRAIN-occupancy-search-release-v1','status':'pending_root_release',
        'expected_head':head,'output':OUTPUT,'TRAIN':list(TRAIN),'closed_roles':CLOSED,
        'SELECT_EVAL_execution':False,'attempts':1,'automatic_retry':False,
        'parent_seconds':PARENT_SECONDS,'worker_seconds':WORKER_SECONDS,'memory_bytes':MEMORY_BYTES,
        'output_bytes':OUTPUT_BYTES,'supervision_bytes':SUPERVISION_BYTES,
        'execution_limits':EXECUTION,'planned_arms':list(ARMS),
        'condition_admission':{c:condition_admission(c) for c in CONDITIONS},
        'input_index':{'path':INPUT_INDEX,'sha256':INPUT_SHA},
        'source_index':{'path':'build/paired-train-occupancy-search-v1/source-index.json','sha256':index_sha},
        'recipe':refs['recipe'],'scope':'source-label S versus explicit simulated S union T, not corrected anatomy',
        'completion':'all eight planned rows retained; capped/failed searches unresolved; each complete plan independently replayed under its own world',
        'search_limit':'greedy scores all currently emitted legal actions; no negative preparatory moves/global optimum proof',
        'normalization':'raw image identical; support-percentile normalization may change and must be reported',
        'private_reference_reads':0,'model_forwards':0,'optimizer_updates':0,'checkpoint_loads':0}

def validate_release(release):
    expected=release_template(release.get('expected_head'),release.get('source_index',{}).get('sha256'))
    expected['status']='released_one_attempt'
    if canonical(release)!=canonical(expected):raise ValueError('Exact one-attempt paired release required')
    for key,n in (('expected_head',40),):
        v=release[key]
        if not isinstance(v,str) or len(v)!=n or any(c not in '0123456789abcdef' for c in v):raise ValueError('Invalid HEAD binding')
    v=release['source_index']['sha256']
    if not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v):raise ValueError('Invalid source binding')

def source_guard(release,release_path,release_sha):
    validate_release(release)
    if sha(release_path)!=release_sha:raise ValueError('Release changed')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    ref=release['source_index']
    if head!=release['expected_head'] or sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('HEAD/source index changed')
    index=small(ROOT/ref['path'])
    if index['head']!=head or index['version']!='paired-TRAIN-occupancy-search-runtime-v1':raise ValueError('Wrong runtime index')
    for section in ('source_files','metadata_files'):
        for relative,digest in index[section].items():
            p=Path(relative)
            if p.is_absolute() or '..' in p.parts or (ROOT/p).is_symlink() or sha(ROOT/p)!=digest:
                raise ValueError('Bound source/metadata changed: '+relative)
    return index

def complete_result(result):
    rows=result.get('arms',[])
    return (result.get('status') == 'complete_paired_TRAIN_search'
        and [{k:r.get(k) for k in ('subject','condition')} for r in rows]==list(ARMS)
        and all(r.get('status') == 'complete' for r in rows)
        and all(r.get('source_released') is True for r in rows)
        and all(type(result.get(k)) is int and result[k]==0 for k in ('policy_forwards','optimizer_updates','checkpoint_loads','private_reference_reads'))
        and result.get('SELECT_EVAL_opened') is False and result.get('global_resource_failure') is None)

def endpoint_control(output,release):
    root=Path(output);result=small(root/'result.json')
    if not complete_result(result):raise ValueError('Incomplete fixed eight-arm denominator')
    native=result['native_counts'];costs=small(root/'costs.json')
    if (native!=costs['native_counts'] or native['search']>192 or native['rollout']+native['replay']>384
            or sum(native.values())>576 or costs['native_budget']['native_preview_entries']>100000
            or costs['prohibited_calls']!={'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0}):
        raise ValueError('Aggregate declared execution accounting changed')
    rows=[];baseline,_=inputs();paired={}
    for ordinal,row in enumerate(result['arms']):
        directory=root/f'arm-{ordinal:02d}'
        if row['status']!='complete':
            if not row.get('failure'):raise ValueError('Failed arm must retain reason')
            rows.append({'ordinal':ordinal,'status':row['status']});continue
        plan=small(directory/'plan.json');replay=small(directory/'replay.json');search=small(directory/'search.json')
        body=plan['plan'];audit=replay['independent_geometry'];metrics=replay['metrics']
        world=small(directory/'world.json')
        for key in ('source_hash','decision_model_hash','initial_observation_hash','context_hash','common_public_world','normalization','derived_occupancy','supplied_goal_extent'):
            if canonical(row[key])!=canonical(world[key]):raise ValueError('Saved world binding changed')
        if row['condition']==CONDITIONS[0]:
            old=baseline['teacher_'+row['subject']]['plan']
            if any(row[k]!=old[k] for k in ('source_hash','decision_model_hash','initial_observation_hash')):
                raise ValueError('Raw baseline identity changed')
            paired[row['subject']]=row
        else:
            raw=paired[row['subject']]
            if (canonical(row['common_public_world'])!=canonical(raw['common_public_world'])
                    or row['source_hash']==raw['source_hash']
                    or row['pairing']['common_inputs_equal'] is not True
                    or row['pairing']['normalization_equal']!=(canonical(row['normalization'])==canonical(raw['normalization']))):
                raise ValueError('Explicit unchanged-input pairing differs')
        if (semantic(body)!=plan['plan_seal'] or row['plan_seal']!=plan['plan_seal']
                or body['source_hash']!=row['source_hash'] or body['decision_model_hash']!=row['decision_model_hash']
                or body['actions']!=search['actions'] or search['accounting']['complete'] is not True
                or canonical(body['history'])!=canonical(metrics['history']) or metrics['terminated'] is not True
                or audit['accepted'] is not True or audit['complete_episode'] is not True
                or audit['committed_history_hash']!=semantic(metrics['history'])
                or audit['source_hash']!=row['source_hash'] or audit['decision_model_hash']!=row['decision_model_hash']
                or canonical(row['outcomes'])!=canonical(audit['outcomes'])
                or len(body['actions'])>24 or body['learning_updates']!=0 or body['parameter_hash'] is not None):
            raise ValueError('Completed arm seal/replay/outcome join differs')
        rows.append({'ordinal':ordinal,'status':'complete','plan_seal':plan['plan_seal'],
            'plan_sha256':sha(directory/'plan.json'),'replay_sha256':sha(directory/'replay.json'),
            'search_sha256':sha(directory/'search.json')})
    return {'status':'eight_planned_TRAIN_rows_bound','arms':rows,'native_counts':native,
        'zero_model_work':True,'no_anatomical_validation':True,'full_denominator_retained':True}
