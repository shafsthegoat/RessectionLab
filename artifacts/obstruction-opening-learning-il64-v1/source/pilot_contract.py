"""Two serial fixed TRAIN endpoints on the completed union/obstruction world."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
TRAIN=('ReMIND-008','ReMIND-010','ReMIND-020','ReMIND-025')
CLOSED={'SELECT':['ReMIND-013','ReMIND-037'],'EVAL':['ReMIND-067']}
PUBLIC_INDEX='build/remind-planning-qc-v1/public-cohort-v1/public-manifest-index.json'
PUBLIC_SHA='11b5c1ae8f9b2c03695e7eb0768d2cc340db68631f3ddb50ba365910fe0187f7'
COHORT='manifests/experiments/remind-component-cohort-v1.json'
COHORT_SHA='326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05'
OCCUPANCY='cerebrum_plus_supplied_tumor_assumption'
STORAGE='cache_complete_replayed_TRAIN_teacher_traces_v1'
WEIGHTING='balanced_STOP_motion_CE_v1'
CACHE_BYTES=64*1024**2
WORKER_SECONDS=3540;PARENT_SECONDS=3600;MEMORY_BYTES=3*1024**3
OUTPUT_BYTES=128*1024**2;SUPERVISION_BYTES=16*1024**2
METHODS={'IL':{'updates':64,'worker_seconds':900,'parent_seconds':960,'source_visits':8,'loss_forward_cap':448,'policy_forward_cap':551,
    'native_preview_cap':70080,'teacher_trace_reuses':256,'teacher_cache_payload_bytes':CACHE_BYTES},
    'RL':{'updates':8,'worker_seconds':3540,'parent_seconds':3600,'source_visits':44,'loss_forward_cap':768,'policy_forward_cap':1639,
    'native_preview_cap':385440,'teacher_trace_reuses':0,'teacher_cache_payload_bytes':0}}
INPUTS={'result': {'path': 'build/paired-obstruction-opening-search-v2/attempt-02/result.json', 'sha256': 'f9f690056ab5b263f7306f9ec6807fa3dc7629f0290b5077ae9c80d57b777d50'}, 'parent': {'path': 'build/paired-obstruction-opening-search-v2/attempt-02.supervision/receipt.json', 'sha256': '54cef3f9163a1ebfb8d90752d930410e359747b8052051cdae320eb266320a99'}, 'release': {'path': 'build/paired-obstruction-opening-search-v2/root-release.json', 'sha256': '16053b9b7f146b8a5d858496c5225f00f0270c39e1a6f605e968991c3c80434d'}, 'source_index': {'path': 'build/paired-obstruction-opening-search-v2/source-index.json', 'sha256': '688c320d979eaa919a1cd04d3fd57afd1480429a99d5aa1d7d0dde8761b639ce'}, 'initial_release': {'path': 'build/cross-patient-planning-v1/fixed-four-train-eight-updates-v1/root-release.json', 'sha256': 'c1cde8eb92f9df5d85ccb91ed0e591adb1da70e011582f3be116d237fecf72db'}, 'initial_result': {'path': 'build/cross-patient-planning-v1/fixed-four-train-eight-updates-v1/attempt-01/result.json', 'sha256': 'b75bd91d284cacba2c980283424586ca506799919f6850175b6f4eeea4315aeb'}, 'plan_ReMIND-008': {'path': 'build/paired-obstruction-opening-search-v2/attempt-02/arm-01/plan.json', 'sha256': '7cb388c8436db10555ea844e8c625a857cb8e1608bfd1d36d440ca0e01f50304'}, 'replay_ReMIND-008': {'path': 'build/paired-obstruction-opening-search-v2/attempt-02/arm-01/replay.json', 'sha256': '9fc9a9d41434b403504a31823769112df681cab0a2dbcb7dc74bca2affbd18e8'}, 'plan_ReMIND-010': {'path': 'build/paired-obstruction-opening-search-v2/attempt-02/arm-03/plan.json', 'sha256': '45942cb56b20e1bb78f5da48609f7e8a19b1d413f6f148c5ce84f795a45ffbfd'}, 'replay_ReMIND-010': {'path': 'build/paired-obstruction-opening-search-v2/attempt-02/arm-03/replay.json', 'sha256': '8849c6119c28508499c5544ee64da969ef1cbfcd2fe7a1ac597015c78cd2c223'}, 'plan_ReMIND-020': {'path': 'build/paired-obstruction-opening-search-v2/attempt-02/arm-05/plan.json', 'sha256': '496fa90c28f4fec2e48fd5efdde8c855ebeee9fa82858deee9b5776c2a7273d8'}, 'replay_ReMIND-020': {'path': 'build/paired-obstruction-opening-search-v2/attempt-02/arm-05/replay.json', 'sha256': 'c55710b1cb4c2023418aa5c2fb6562e111633895a09351b4a3d46a76481a0500'}, 'plan_ReMIND-025': {'path': 'build/paired-obstruction-opening-search-v2/attempt-02/arm-07/plan.json', 'sha256': 'fa73149b4c2f3afea4a37041ccf83a82e8f67a6415e0c20cf6b5b62d3977f978'}, 'replay_ReMIND-025': {'path': 'build/paired-obstruction-opening-search-v2/attempt-02/arm-07/replay.json', 'sha256': '41e04e7b3dd565e6f50461b6ef203bce510cc0b9d562dad76f178c3eb5c1d776'}}

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def semantic(value):return 'sha256:'+hashlib.sha256(canonical(value).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def output_for(method):
    if method not in METHODS:raise ValueError('Fixed IL64 or RL8 method required')
    return 'build/obstruction-opening-learning-v1/'+method+str(METHODS[method]['updates'])+'/attempt-01'
def small(ref):
    path=ROOT/ref['path']
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=4*1024**2 or sha(path)!=ref['sha256']:
        raise ValueError('Bounded pinned input changed: '+ref['path'])
    return json.loads(path.read_text())

def inputs():
    rows={k:small(v) for k,v in INPUTS.items()}
    result,parent,release=rows['result'],rows['parent'],rows['release']
    if (result['status']!='complete_paired_TRAIN_search' or result['SELECT_EVAL_opened'] is not False
            or parent['status']!='complete' or parent['exit_code']!=0 or parent['cleanup_errors']
            or parent['final_owned_pids'] or parent['worker_termination_confirmed'] is not True
            or parent['result_sha256']!=INPUTS['result']['sha256']
            or parent['release_sha256']!=INPUTS['release']['sha256']
            or parent['source_index']!=release['source_index']):
        raise ValueError('Clean completed paired search required')
    selected=[a for a in result['arms'] if a['condition']=='obstruction_opening']
    if [a['subject'] for a in selected]!=list(TRAIN):raise ValueError('Exact original four TRAIN arms required')
    for arm in selected:
        subject=arm['subject'];saved=rows['plan_'+subject];plan=saved['plan'];replay=rows['replay_'+subject]
        if (arm['status']!='complete' or arm['source_released'] is not True
                or arm['independent_geometry_accepted'] is not True
                or arm['proposal_config']['obstruction_opening'] is not True
                or arm['proposal_config']['tool_footprint_opening'] is not True
                or arm['proposal_config']['max_candidates']!=120
                or arm['derived_occupancy']['condition']!=OCCUPANCY
                or semantic(plan)!=saved['plan_seal'] or saved['plan_seal']!=arm['plan_seal']
                or plan['actions']!=arm['actions'] or plan['max_steps']!=24
                or len(plan['actions'])!=(4 if subject=='ReMIND-025' else 1)
                or plan['actions'][-1]!='STOP' or plan['parameter_hash'] is not None or plan['learning_updates']!=0
                or canonical(replay['metrics']['history'])!=canonical(plan['history'])
                or replay['independent_geometry']['accepted'] is not True):
            raise ValueError('Complete generalized search teacher/world changed: '+subject)
        for field in ('source_hash','decision_model_hash','initial_observation_hash'):
            if plan[field]!=arm[field]:raise ValueError('Generalized search plan/world differs')
    rows['arms']={a['subject']:a for a in selected}
    return rows

def expected_configuration(method):
    data=inputs();protocol=json.loads(canonical(data['initial_release']['learning_protocol']))
    execution=protocol['cohort_execution'];arm=data['arms'][TRAIN[0]]
    protocol['updates_per_method']=METHODS[method]['updates']
    execution.update(proposal_config=arm['proposal_config'],proposal_rule_hash=arm['proposal_rule_hash'],occupancy_condition=OCCUPANCY)
    if method=='IL':execution.update(il_teacher_weighting=WEIGHTING,teacher_observations=STORAGE,teacher_cache_payload_bytes=CACHE_BYTES)
    limits={'max_steps':24,'max_optimizer_updates':128 if method=='IL' else 16,
        'max_native_previews':7297920 if method=='IL' else 3373440,
        'max_policy_forwards':18624 if method=='IL' else 2496,'worker_seconds':METHODS[method]['worker_seconds'],
        'memory_bytes':MEMORY_BYTES,'threads':1,'search':execution['search'],
        'output_bytes':OUTPUT_BYTES,'checkpoint_bytes':8*1024**2}
    return protocol,limits

def canonical_configuration(method):
    from resectionlab.core import thaw_json
    from resectionlab.native_proposals import NominalCavityProposalConfig
    from resectionlab.patient_planning_cohort_spec import sequential_learning_protocol,sequential_limits
    expected,_=expected_configuration(method);execution=expected['cohort_execution']
    options={} if method=='RL' else {'il_teacher_weighting':WEIGHTING,'teacher_observations':STORAGE,'teacher_cache_payload_bytes':CACHE_BYTES}
    protocol=sequential_learning_protocol(updates=METHODS[method]['updates'],max_steps=24,
        search=execution['search'],proposal_config=NominalCavityProposalConfig(**execution['proposal_config']),
        retention_mode=execution['retention_mode'],occupancy_condition=OCCUPANCY,**options)
    limits=sequential_limits(protocol,worker_seconds=METHODS[method]['worker_seconds'],memory_bytes=MEMORY_BYTES,output_bytes=OUTPUT_BYTES)
    return thaw_json(protocol),thaw_json(limits)

def validate_release(release):
    method=release.get('method')
    if method not in METHODS:raise ValueError('Fixed declared method required')
    protocol,limits=expected_configuration(method)
    if (release.get('version')!='matched-union-obstruction-learning-v1'
            or release.get('status')!='released_one_attempt' or release.get('output')!=output_for(method)
            or release.get('TRAIN')!=list(TRAIN) or release.get('closed_roles')!=CLOSED
            or release.get('SELECT_EVAL_execution') is not False or release.get('new_four_training') is not False
            or release.get('attempts')!=1 or type(release.get('attempts')) is not int
            or release.get('automatic_retry') is not False or release.get('inputs')!=INPUTS
            or canonical(release.get('execution_limits'))!=canonical(METHODS[method])
            or canonical(release.get('learning_protocol'))!=canonical(protocol)
            or canonical(release.get('cohort_limits'))!=canonical(limits)
            or release.get('learning_protocol_hash')!=semantic(protocol)
            or canonical(release.get('limits'))!=canonical({k:v for k,v in limits.items() if k not in ('output_bytes','checkpoint_bytes')})
            or type(release.get('parent_seconds')) is not int or type(release.get('supervision_bytes')) is not int
            or release.get('parent_seconds')!=METHODS[method]['parent_seconds'] or release.get('supervision_bytes')!=SUPERVISION_BYTES):
        raise ValueError('Exact separately bounded fixed endpoint release required')
    if method=='RL':
        ref=release.get('matched_IL_completion',{})
        if ref.get('path')!=output_for('IL')+'.supervision/receipt.json':raise ValueError('RL follows the completed matched IL endpoint')
        receipt=small(ref)
        if (receipt['status']!='complete' or receipt['exit_code']!=0 or receipt['cleanup_errors'] or receipt['final_owned_pids']
                or receipt.get('worker_termination_confirmed') is not True):
            raise ValueError('Clean matched IL completion required before RL')
        prior=small({'path':output_for('IL')+'/result.json','sha256':receipt['result_sha256']})
        if not complete_result(prior) or prior['method']!='IL':raise ValueError('Completed matched IL64 required before RL')

def source_guard(release,release_path,release_sha):
    validate_release(release)
    if sha(release_path)!=release_sha:raise ValueError('Release changed')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    if head!=release['expected_head']:raise ValueError('Canonical HEAD changed')
    ref=release['source_index'];index=small(ref)
    if index['head']!=head:raise ValueError('Source index HEAD differs')
    for section in ('source_files','metadata_files'):
        for path,digest in index[section].items():
            p=Path(path)
            if p.is_absolute() or '..' in p.parts or (ROOT/p).is_symlink() or sha(ROOT/p)!=digest:
                raise ValueError('Source/input changed: '+path)
    return index

def complete_result(result):
    method=result.get('method');cfg=METHODS.get(method,{})
    if not cfg:return False
    return (result.get('status')=='complete_matched_TRAIN_endpoint'
        and result.get('optimizer_updates')=={m:cfg['updates'] if m==method else 0 for m in ('IL','RL')}
        and (result.get('loss_forward_calls')==448 if method=='IL' else 32<=result.get('loss_forward_calls',0)<=768)
        and result.get('teacher_logit_forwards')==7 and result.get('checkpoint_loads')==1
        and result.get('completed_source_visits')==cfg['source_visits']
        and result.get('teacher_trace_reuses')==cfg['teacher_trace_reuses']
        and result.get('search_calls')==0 and result.get('SELECT_EVAL_opened') is False
        and result.get('teacher_statuses')=={s:'complete_replayed' for s in TRAIN}
        and set(result.get('checkpoints',{}))=={method} and set(result.get('TRAIN_greedy',{}))==set(TRAIN)
        and all(result['TRAIN_greedy'][s].get('complete') is True for s in TRAIN)
        and result.get('selection_readiness',{}).get('ready') is True
        and result.get('selection_readiness',{}).get('execution_admitted') is False)

def endpoint_control(output,release):
    result=json.loads((Path(output)/'result.json').read_text());data=inputs()
    if not complete_result(result) or result['initial_parameter_hash']!=data['initial_result']['initial_parameter_hash']:
        raise ValueError('Complete endpoint with identical initial tensors required')
    dynamics=Path(output)/'training-dynamics.json'
    if sha(dynamics)!=result['training_dynamics_sha256']:raise ValueError('Dynamics changed')
    if len(json.loads(dynamics.read_text())['updates'])!=METHODS[release['method']]['updates']:
        raise ValueError('Fixed complete update trajectory required')
    matched=None
    if release['method']=='RL':
        receipt=small(release['matched_IL_completion'])
        prior_path=ROOT/output_for('IL')/'result.json'
        if sha(prior_path)!=receipt['result_sha256']:raise ValueError('Matched IL result changed')
        prior=json.loads(prior_path.read_text())
        if not complete_result(prior) or prior['method']!='IL' or prior['initial_parameter_hash']!=result['initial_parameter_hash']:
            raise ValueError('Matched methods initial tensors differ')
        matched={'IL_result_sha256':receipt['result_sha256'],'initial_parameters_equal':True,
            'same_world_teacher_plans':{s:INPUTS['plan_'+s]['sha256'] for s in TRAIN},'equal_compute_claim':False}
    return {'status':'complete_fixed_endpoint','method':release['method'],
        'initial_parameter_hash':result['initial_parameter_hash'],'matched_methods':matched,
        'seven_generalized_teacher_states':True,'checkpoint_reloaded':True,
        'TRAIN_greedy':result['TRAIN_greedy'],'teacher_metrics':result['endpoint_teacher_metrics'],
        'training_dynamics_sha256':result['training_dynamics_sha256'],'heldout_or_clinical_claim':False}
