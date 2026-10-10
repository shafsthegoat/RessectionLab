"""Two serial fixed TRAIN endpoints on the tested partial-domain post-exposure world."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
TRAIN=('ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045')
TEACHER_STEPS={'ReMIND-002':1,'ReMIND-015':14,'ReMIND-018':1,'ReMIND-045':13}
TEACHER_STATES=29
EXPOSURE='public-union-post-exposure-axis0-v1'
CLOSED={'SELECT':['ReMIND-013','ReMIND-037'],'EVAL':['ReMIND-067']}
PUBLIC_INDEX='build/remind-partial-domain-public-preparation-v1/manifests-v2/public-index.json'
PUBLIC_SHA='c5138559e45fd67b44b047e8ea6d800b56b8e63ae3371a5ff5c9d83e60b25fe8'
COHORT='manifests/experiments/remind-component-cohort-v1.json'
COHORT_SHA='326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05'
OCCUPANCY='cerebrum_plus_supplied_tumor_with_preserved_partial_source_domain'
STORAGE='cache_complete_replayed_TRAIN_teacher_traces_v1'
WEIGHTING='balanced_STOP_motion_CE_v1'
CACHE_BYTES=256*1024**2
WORKER_SECONDS=3540;PARENT_SECONDS=3600;MEMORY_BYTES=3*1024**3
OUTPUT_BYTES=128*1024**2;SUPERVISION_BYTES=16*1024**2
METHODS={'IL':{'updates':64,'worker_seconds':900,'parent_seconds':960,'source_visits':8,'loss_forward_cap':1856,'policy_forward_cap':1981,
    'native_preview_cap':70080,'teacher_trace_reuses':256,'teacher_cache_payload_bytes':CACHE_BYTES},
    'RL':{'updates':8,'worker_seconds':3540,'parent_seconds':3600,'source_visits':44,'loss_forward_cap':768,'policy_forward_cap':1661,
    'native_preview_cap':385440,'teacher_trace_reuses':0,'teacher_cache_payload_bytes':0}}
INPUTS={'result': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/result.json', 'sha256': '7a57fd74f58b15d6975be4459fe4f46e6396dfa9e2f3c0fec0ef68cf5e550358'}, 'parent': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01.supervision/receipt.json', 'sha256': '664b7ac101e8c79e512d52f971f424b5e2b1bbeaa7d7b1502c110afb4364f233'}, 'release': {'path': 'build/remind-post-exposure-feasibility-v1/root-release.json', 'sha256': '8315099d95b16a9dacac54a65c4d84d6385e458534b6dab26b73a37555d4af1a'}, 'source_index': {'path': 'build/remind-post-exposure-feasibility-v1/source-index.json', 'sha256': '4ac60917a28368056b1b0d07481a88f447be4116ec677fed1bb31dd2766c7b50'}, 'initial_release': {'path': 'build/cross-patient-planning-v1/fixed-four-train-eight-updates-v1/root-release.json', 'sha256': 'c1cde8eb92f9df5d85ccb91ed0e591adb1da70e011582f3be116d237fecf72db'}, 'plan_ReMIND-002': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-002/greedy-plan.json', 'sha256': '757449e8ac744daee89dd085924a27c8a7a4fc571a1c906994188bf08c4989bc'}, 'metrics_ReMIND-002': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-002/episode-metrics.json', 'sha256': '59a3e11e3bd662063df245ba485ff36b4001c0445c688f572421ea42e1a83789'}, 'replay_ReMIND-002': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-002/independent-episode.json', 'sha256': '40d784c101280b97bc902fc77acdf363e5243b1bfab3cfeeb4feb276713acae8'}, 'plan_ReMIND-015': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-015/greedy-plan.json', 'sha256': '954ec5204482ba6c91867e34e61700470f060c74b702711e57414a77cb2cfcf1'}, 'metrics_ReMIND-015': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-015/episode-metrics.json', 'sha256': '3ff9a551a4cad52e26936c797cdaf0b6923a6ddf5ae894867c8dd517a99359d9'}, 'replay_ReMIND-015': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-015/independent-episode.json', 'sha256': 'bcaf1303133576b7a281eec3915a3088f95306b17ef655041222f85272387547'}, 'plan_ReMIND-018': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-018/greedy-plan.json', 'sha256': '4d0b9a0029293e787a99c276bd6bca9430bc0722c2eaaecd88bb193d4e1e5040'}, 'metrics_ReMIND-018': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-018/episode-metrics.json', 'sha256': '911e0d6060b4e16188a054ff2f736be90f897796db48b22c2b85a3faf16cb012'}, 'replay_ReMIND-018': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-018/independent-episode.json', 'sha256': '20d0f0be9e4aa990136d47e5e2c91c2df140492b130d850fcb641e737bf7e6cb'}, 'plan_ReMIND-045': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-045/greedy-plan.json', 'sha256': 'da7b49d6fde2c97c9f6911f42475afb7a4cc946e8c894e8267cc4a82679c58c1'}, 'metrics_ReMIND-045': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-045/episode-metrics.json', 'sha256': 'fa67b7ee6862222d480d60a570161e7990082e854e0327814191a629a48f6e21'}, 'replay_ReMIND-045': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-045/independent-episode.json', 'sha256': 'a18c03193e4968bf2f2caa44257117c51107091ee0bdfe18083b7f5edd1b0db9'}}

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def semantic(value):return 'sha256:'+hashlib.sha256(canonical(value).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def output_for(method):
    if method not in METHODS:raise ValueError('Fixed IL64 or RL8 method required')
    return 'build/post-exposure-learning-v1/'+method+str(METHODS[method]['updates'])+'/attempt-01'
def small(ref):
    path=ROOT/ref['path']
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=4*1024**2 or sha(path)!=ref['sha256']:
        raise ValueError('Bounded pinned input changed: '+ref['path'])
    return json.loads(path.read_text())

def inputs():
    rows={k:small(v) for k,v in INPUTS.items()}
    result,parent,release=rows['result'],rows['parent'],rows['release']
    if (result['status']!='post_exposure_batch_complete' or result['training_admitted'] is not False
            or result['occupancy_condition']!=OCCUPANCY or result['post_exposure_condition']!=EXPOSURE
            or any(result[k]!=0 for k in ('model_calls','optimizer_calls','checkpoint_loads','source_arrays_written'))
            or parent['status']!='complete' or parent['exit_code']!=0 or parent['cleanup_errors']
            or parent['remaining_owned_pids'] or parent['worker_termination_confirmed'] is not True
            or parent['result_sha256']!=INPUTS['result']['sha256']
            or parent['release_sha256']!=INPUTS['release']['sha256']
            or parent['source_index']!=release['source_index']):
        raise ValueError('Clean completed fixed-four post-exposure measurement required')
    selected=result['cases']
    if [a['patient_id'] for a in selected]!=list(TRAIN):raise ValueError('Exact new four TRAIN cases required')
    config=result['proposal_config']
    if (config.get('obstruction_opening',False) is not False or config['max_candidates']!=120
            or config['tool_footprint_opening'] is not True or len(config['offsets_source_voxels'])!=13):
        raise ValueError('Tested historical13-axis/cap120 proposal world required')
    for arm in selected:
        subject=arm['patient_id'];saved=rows['plan_'+subject];metrics=rows['metrics_'+subject];audit=rows['replay_'+subject]
        if (arm['status']!='completed_fixed_post_exposure_route' or arm['role']!='TRAIN'
                or arm['route_complete'] is not True or arm['independent_episode_accepted'] is not True
                or arm['full_target_preserved'] is not True or arm['source_domain_extended'] is not False
                or saved['actions']!=arm['actions'] or len(saved['actions'])!=TEACHER_STEPS[subject]
                or saved['actions'][-1]!='STOP' or saved['accounting']['complete'] is not True
                or [h['action_id'] for h in metrics['history']]!=saved['actions']
                or metrics['terminated'] is not True or metrics['steps']!=TEACHER_STEPS[subject]
                or audit['accepted'] is not True or audit['complete_episode'] is not True
                or audit['committed_history_hash']!=semantic(metrics['history'])
                or audit['outcomes']!=arm['full_route_outcomes']
                or metrics['source_hash']!=arm['source_hash']
                or metrics['decision_model_hash']!=audit['decision_model_hash']
                or metrics['post_exposure']!=arm['post_exposure_start']):
            raise ValueError('Complete saved greedy teacher/world changed: '+subject)
        # This projection creates no training authority; every trace is recollected
        # and independently replayed under its new explicit learning context.
        saved['plan']={'source_hash':metrics['source_hash'],'decision_model_hash':metrics['decision_model_hash'],
            'initial_observation_hash':arm['initial_observation_hash'],'max_steps':24,'actions':saved['actions'],
            'history':metrics['history'],'terminal_reason':'STOP','parameter_hash':None,'architecture_hash':None,'learning_updates':0}
    rows['arms']={a['patient_id']:a for a in selected}
    return rows

def expected_configuration(method):
    data=inputs();protocol=json.loads(canonical(data['initial_release']['learning_protocol']))
    execution=protocol['cohort_execution'];arm=data['arms'][TRAIN[0]]
    protocol['updates_per_method']=METHODS[method]['updates']
    execution.update(proposal_config=data['result']['proposal_config'],proposal_rule_hash=data['metrics_'+TRAIN[0]]['proposal_rule_hash'],occupancy_condition=OCCUPANCY,post_exposure_condition=EXPOSURE,patient_order=list(TRAIN))
    protocol['version']='fixed-four-TRAIN-post-exposure-learning-v1'
    protocol['scope']='fixed_four_TRAIN_shared_population_pilot'
    execution.update(teacher_source='fixed_saved_greedy_complete_histories_v1',teacher_decisions=29,teacher_steps=[1,14,1,13])
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
        retention_mode=execution['retention_mode'],occupancy_condition=OCCUPANCY,post_exposure_condition=EXPOSURE,**options)
    limits=sequential_limits(protocol,worker_seconds=METHODS[method]['worker_seconds'],memory_bytes=MEMORY_BYTES,output_bytes=OUTPUT_BYTES)
    return thaw_json(protocol),thaw_json(limits)

def validate_release(release):
    method=release.get('method')
    if method not in METHODS:raise ValueError('Fixed declared method required')
    protocol,limits=expected_configuration(method)
    keys={'version','status','method','expected_head','output','TRAIN','closed_roles','SELECT_EVAL_execution',
        'new_four_training','attempts','automatic_retry','parent_seconds','supervision_bytes','cohort_limits',
        'limits','execution_limits','learning_protocol','learning_protocol_hash','inputs','source_index',
        'public_manifest_index','completion','claim'} | ({'matched_IL_completion'} if method=='RL' else set())
    if set(release)!=keys:raise ValueError('Exact endpoint declaration keys required')
    if release['public_manifest_index']!={'path':PUBLIC_INDEX,'sha256':PUBLIC_SHA}:raise ValueError('Exact public manifest index required')
    if (release.get('version')!='matched-post-exposure-learning-v1'
            or release.get('status')!='released_one_attempt' or release.get('output')!=output_for(method)
            or release.get('TRAIN')!=list(TRAIN) or release.get('closed_roles')!=CLOSED
            or release.get('SELECT_EVAL_execution') is not False or release.get('new_four_training') is not True
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
        and result.get('TRAIN')==list(TRAIN) and result.get('post_exposure_condition')==EXPOSURE
        and result.get('fresh_common_initialization_verified') is True
        and result.get('optimizer_updates')=={m:cfg['updates'] if m==method else 0 for m in ('IL','RL')}
        and (result.get('loss_forward_calls')==1856 if method=='IL' else 32<=result.get('loss_forward_calls',0)<=768)
        and result.get('teacher_logit_forwards')==TEACHER_STATES and result.get('checkpoint_loads')==1
        and result.get('completed_source_visits')==cfg['source_visits']
        and result.get('teacher_trace_reuses')==cfg['teacher_trace_reuses']
        and result.get('search_calls')==0 and result.get('private_reference_reads')==0 and result.get('SELECT_EVAL_opened') is False
        and result.get('teacher_decisions')==TEACHER_STATES
        and result.get('teacher_steps')==TEACHER_STEPS
        and 0<=result.get('native_preview_entries',-1)<=cfg['native_preview_cap']
        and result.get('total_policy_forward_calls')==result.get('loss_forward_calls',0)*(1 if method=='IL' else 2)+TEACHER_STATES+sum(r.get('steps',0) for r in result.get('TRAIN_greedy',{}).values())
        and result.get('total_policy_forward_calls',cfg['policy_forward_cap']+1)<=cfg['policy_forward_cap']
        and result.get('teacher_statuses')=={s:'complete_replayed' for s in TRAIN}
        and set(result.get('checkpoints',{}))=={method} and set(result.get('TRAIN_greedy',{}))==set(TRAIN)
        and all(result['TRAIN_greedy'][s].get('complete') is True for s in TRAIN)
        and result.get('selection_readiness',{}).get('ready') is True
        and result.get('selection_readiness',{}).get('execution_admitted') is False)

def endpoint_control(output,release):
    result=json.loads((Path(output)/'result.json').read_text());data=inputs()
    if not complete_result(result):
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
        'twenty_nine_saved_greedy_teacher_states':True,'checkpoint_reloaded':True,
        'TRAIN_greedy':result['TRAIN_greedy'],'teacher_metrics':result['endpoint_teacher_metrics'],
        'training_dynamics_sha256':result['training_dynamics_sha256'],'heldout_or_clinical_claim':False}
