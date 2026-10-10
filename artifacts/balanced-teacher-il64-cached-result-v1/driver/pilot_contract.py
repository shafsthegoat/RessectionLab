"""Owned cached64 parity contract. Metadata only; no model/payload loader."""
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
original_path=ROOT/'build/balanced-teacher-il64-v1/pilot_contract.py'
if hashlib.sha256(original_path.read_bytes()).hexdigest()!='e44e4846981a89c1bd760161b3d3aecb041ed3b786cc6e233b919db0a10b0d5a':
    raise ValueError('Original fixed64 contract changed')
spec=importlib.util.spec_from_file_location('cache_original_fixed64_contract',original_path)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
OUTPUT='build/balanced-teacher-il64-cached-v1/attempt-01'
TRAIN=original.TRAIN
CLOSED=original.CLOSED
PARENT_SECONDS=original.PARENT_SECONDS
MEMORY_BYTES,OUTPUT_BYTES,SUPERVISION_BYTES=original.MEMORY_BYTES,original.OUTPUT_BYTES,original.SUPERVISION_BYTES
COHORT_SHA=original.COHORT_SHA
BASELINE_FILES,BALANCED_FILES=original.BASELINE_FILES,original.BALANCED_FILES
FIXED64_INDEX={'path':'build/balanced-teacher-il64-cached-v1/baseline-metadata-index.json',
    'sha256':'e75b2a8e67284c3fc98b11ca7cd74ac9f72dc607d02b10d04a606846f47ed041'}
PUBLIC_INDEX,PUBLIC_SHA,COHORT=original.PUBLIC_INDEX,original.PUBLIC_SHA,original.COHORT
WORKER_SECONDS=original.WORKER_SECONDS
canonical,sha,semantic=original.canonical,original.sha,original.semantic
STORAGE='cache_complete_replayed_TRAIN_teacher_traces_v1'
CACHE_BYTES=64*1024**2
# Keep forward/optimizer budgets; eight native source visits replace268.
EXECUTION={**original.EXECUTION,'source_visits':8,
    'max_native_previews':8*120*(1+3*24), 'teacher_trace_reuses':256,
    'cached_teacher_readout_rows':5, 'teacher_cache_payload_bytes':CACHE_BYTES}


def recollection_protocol(protocol):
    plain=json.loads(canonical(protocol));execution=plain['cohort_execution']
    if (execution.get('teacher_observations')!=STORAGE
            or type(execution.get('teacher_cache_payload_bytes')) is not int
            or execution['teacher_cache_payload_bytes']!=CACHE_BYTES):
        raise ValueError('Exact prospective cache storage mode required')
    execution['teacher_observations']='recollect_complete_pinned_plan_each_IL_update'
    execution.pop('teacher_cache_payload_bytes')
    return plain


def fixed64_metadata():
    index_path=HERE/'baseline-metadata-index.json'
    if sha(index_path)!='e75b2a8e67284c3fc98b11ca7cd74ac9f72dc607d02b10d04a606846f47ed041':
        raise ValueError('Fixed64 metadata index changed')
    refs=json.loads(index_path.read_text())['files']
    records={key:original.read_pinned(ref) for key,ref in refs.items()}
    result,parent,release=records['result'],records['parent'],records['release']
    if (not original.complete_result(result) or parent['status']!='complete'
            or parent['exit_code']!=0 or parent['cleanup_errors'] or parent['final_owned_pids']
            or parent['worker_termination_confirmed'] is not True
            or parent['result_sha256']!=refs['result']['sha256']
            or parent['release_sha256']!=refs['release']['sha256']
            or parent['source_index']!=release['source_index']):
        raise ValueError('Clean completed fixed64 comparison baseline required')
    return records,refs


def require_baseline(protocol):
    plain=recollection_protocol(protocol)
    baseline=original.require_baseline(plain)
    current,_=fixed64_metadata()
    if canonical(plain)!=canonical(current['release']['learning_protocol']):
        raise ValueError('Only prospective storage mode/allowance may differ from completed64')
    baseline['fixed64']=current
    return baseline


def canonical_configuration():
    from resectionlab.core import thaw_json
    from resectionlab.native_proposals import NominalCavityProposalConfig
    from resectionlab.patient_planning_cohort_spec import sequential_learning_protocol,sequential_limits
    old,_=original.canonical_configuration();execution=old['cohort_execution']
    protocol=sequential_learning_protocol(updates=64,max_steps=24,search=execution['search'],
        proposal_config=NominalCavityProposalConfig(**execution['proposal_config']),
        retention_mode=execution['retention_mode'],il_teacher_weighting=execution['il_teacher_weighting'],
        teacher_observations=STORAGE,teacher_cache_payload_bytes=CACHE_BYTES)
    limits=sequential_limits(protocol,worker_seconds=WORKER_SECONDS,
        memory_bytes=original.MEMORY_BYTES,output_bytes=original.OUTPUT_BYTES)
    return thaw_json(protocol),thaw_json(limits)


def expected_configuration():
    """Metadata-only freeze; actual worker checks canonical constructor equality."""
    records,_=fixed64_metadata()
    original.validate_release(records['release'])
    protocol=json.loads(canonical(records['release']['learning_protocol']))
    protocol['cohort_execution'].update(teacher_observations=STORAGE,
        teacher_cache_payload_bytes=CACHE_BYTES)
    limits=json.loads(canonical(records['release']['cohort_limits']))
    require_baseline(protocol)
    return protocol,limits


def validate_release(release):
    protocol,limits=expected_configuration()
    if (release.get('version')!='balanced-teacher-IL64-cached-release-v1'
            or release.get('status')!='released_one_attempt' or release.get('output')!=OUTPUT
            or release.get('TRAIN')!=list(TRAIN) or release.get('closed_roles')!=CLOSED
            or release.get('SELECT_EVAL_execution') is not False
            or release.get('automatic_retry') is not False
            or type(release.get('attempts')) is not int or release['attempts']!=1
            or type(release.get('parent_seconds')) is not int or release['parent_seconds']!=PARENT_SECONDS
            or type(release.get('supervision_bytes')) is not int or release['supervision_bytes']!=SUPERVISION_BYTES
            or canonical(release.get('execution_limits'))!=canonical(EXECUTION)
            or canonical(release.get('cohort_limits'))!=canonical(limits)
            or canonical(release.get('limits'))!=canonical({k:v for k,v in limits.items()
                if k not in ('output_bytes','checkpoint_bytes')})
            or canonical(release.get('learning_protocol'))!=canonical(protocol)
            or release.get('learning_protocol_hash')!=semantic(protocol)
            or release.get('baseline')!=BASELINE_FILES or release.get('balanced_baseline')!=BALANCED_FILES
            or release.get('fixed64_baseline')!=FIXED64_INDEX
            or release.get('public_manifest_index')!={'path':PUBLIC_INDEX,'sha256':PUBLIC_SHA}):
        raise ValueError('Exact one-attempt cached64 parity release required')
    head=release.get('expected_head');ref=release.get('source_index',{})
    if (not isinstance(head,str) or len(head)!=40 or any(c not in '0123456789abcdef' for c in head)
            or set(ref)!={'path','sha256'} or ref['path']!='build/balanced-teacher-il64-cached-v1/source-index.json'
            or not isinstance(ref['sha256'],str) or len(ref['sha256'])!=64
            or any(c not in '0123456789abcdef' for c in ref['sha256'])):
        raise ValueError('Exact source/head binding required')


def source_guard(release,release_path,release_sha):
    validate_release(release)
    if sha(release_path)!=release_sha:raise ValueError('Release changed')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    if head!=release['expected_head']:raise ValueError('Canonical HEAD changed')
    ref=release['source_index']
    if sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Source index changed')
    index=json.loads((ROOT/ref['path']).read_text())
    if index['head']!=head or index['version']!='balanced-teacher-IL64-cached-runtime-v1':
        raise ValueError('Exact cached runtime index required')
    for section in ('source_files','metadata_files'):
        for relative,expected in index[section].items():
            path=Path(relative)
            if path.is_absolute() or '..' in path.parts or (ROOT/path).is_symlink() or sha(ROOT/path)!=expected:
                raise ValueError('Bound source/metadata changed: '+relative)
    return index


def update_control(output,release,count):
    baseline,refs=fixed64_metadata();require_baseline(release['learning_protocol'])
    rows=[]
    for i in range(1,count+1):
        path=Path(output)/'IL'/f'update-{i:02d}'/'update.json'
        if path.is_symlink() or not path.is_file() or path.stat().st_size>2*1024**2:
            raise ValueError('Bounded complete update record required')
        actual=json.loads(path.read_text());prior=baseline[f'update_{i:02d}']
        excluded={'context_hashes','trace_seals'}
        if canonical({k:v for k,v in actual.items() if k not in excluded})!=canonical({k:v for k,v in prior.items() if k not in excluded}):
            raise ValueError('Cache numerical parity failed at update '+str(i))
        rows.append({'update':i,'actual_sha256':sha(path),'baseline_sha256':refs[f'update_{i:02d}']['sha256'],
            'after_parameter_hash':actual['after_parameter_hash'],'loss':actual['loss']})
    return {'exact_match':True,'updates':rows,'excluded_metadata_fields':sorted(excluded),
        'scope':'all other update fields, including gradient and full parameter hashes, equal completed recollected64'}


def first_eight_update_control(output,release):return update_control(output,release,8)


def complete_result(result):
    return (result.get('status')=='complete_cached_balanced_IL64_TRAIN_only'
        and result.get('optimizer_updates')=={'IL':64,'RL':0}
        and result.get('loss_forward_calls')==320 and result.get('teacher_logit_forwards')==5
        and result.get('checkpoint_loads')==1 and result.get('search_calls')==0
        and result.get('completed_source_visits')==8 and result.get('teacher_trace_reuses')==256
        and result.get('first_eight_update_control',{}).get('exact_match') is True
        and result.get('training_dynamics_updates')==64
        and result.get('SELECT_EVAL_opened') is False
        and result.get('selection_readiness',{}).get('ready') is True
        and result.get('selection_readiness',{}).get('execution_admitted') is False
        and set(result.get('checkpoints',{}))=={'IL'}
        and result.get('teacher_statuses')=={s:'complete_replayed' for s in TRAIN}
        and set(result.get('TRAIN_greedy',{}))==set(TRAIN)
        and all(result['TRAIN_greedy'][s].get('complete') is True for s in TRAIN))


def endpoint_control(output,release):
    prior,_=fixed64_metadata();result=json.loads((Path(output)/'result.json').read_text())
    if not complete_result(result):raise ValueError('Incomplete cached64 endpoint')
    all_updates=update_control(output,release,64)
    if canonical(first_eight_update_control(output,release))!=canonical(result['first_eight_update_control']):
        raise ValueError('Result does not bind the exact first-eight control')
    if result['checkpoints']['IL']['parameter_hash']!=prior['result']['checkpoints']['IL']['parameter_hash']:
        raise ValueError('Final full parameter hash differs')
    if result['initial_parameter_hash']!=prior['result']['initial_parameter_hash']:
        raise ValueError('Initial full parameter hash differs')
    dynamics_path=Path(output)/'training-dynamics.json'
    if (dynamics_path.is_symlink() or not dynamics_path.is_file()
            or dynamics_path.stat().st_size>2*1024**2
            or sha(dynamics_path)!=result['training_dynamics_sha256']):
        raise ValueError('Complete64 loss/gradient dynamics changed')
    dynamics=json.loads(dynamics_path.read_text())
    old_dynamics=prior['dynamics']
    if len(dynamics['updates'])!=64 or canonical({k:v for k,v in dynamics.items() if k!='updates'})!=canonical({k:v for k,v in old_dynamics.items() if k!='updates'}):
        raise ValueError('Complete64 dynamics scope/count changed')
    for row,old,update in zip(dynamics['updates'],old_dynamics['updates'],all_updates['updates']):
        if (row['update_record_sha256']!=update['actual_sha256']
                or canonical({k:v for k,v in row.items() if k!='update_record_sha256'})!=canonical({k:v for k,v in old.items() if k!='update_record_sha256'})):
            raise ValueError('Complete64 dynamics arithmetic/update binding changed')
    cache=cache_control(output,result,release)
    if canonical(result['endpoint_teacher_metrics'])!=canonical(prior['result']['endpoint_teacher_metrics']):
        raise ValueError('Reported teacher endpoint metrics differ')
    for subject in TRAIN:
        plan=json.loads((Path(output)/'TRAIN-greedy/IL'/subject/'plan.json').read_text())
        old=prior['greedy_'+subject]
        if (semantic(plan['plan'])!=plan['plan_seal']
                or canonical({k:v for k,v in plan['plan'].items() if k!='context_hash'})!=canonical({k:v for k,v in old['plan'].items() if k!='context_hash'})):
            raise ValueError('Final complete TRAIN native plan/history differs')
        actual_row=result['TRAIN_greedy'][subject];prior_row=prior['result']['TRAIN_greedy'][subject]
        if (actual_row['plan_seal']!=plan['plan_seal']
                or canonical({k:v for k,v in actual_row.items() if k!='plan_seal'})!=canonical({k:v for k,v in prior_row.items() if k!='plan_seal'})):
            raise ValueError('Reported TRAIN outcome differs from matched complete history')
        for step in range(2 if subject=='ReMIND-025' else 1):
            row=json.loads((Path(output)/'teacher-readout'/subject/f'state-{step:02d}.json').read_text())
            if canonical(row)!=canonical(prior[f'state_{subject}_{step}']):
                raise ValueError('Exact final teacher input/logit readout differs')
    return {'status':'exact_cached_vs_recollected64','updates':all_updates,'cache_control':cache,
        'training_dynamics_sha256':result['training_dynamics_sha256'],
        'same_endpoint_parameter_hash':result['checkpoints']['IL']['parameter_hash'],
        'same_five_readouts':True,'same_four_complete_native_histories':True,
        'checkpoint_file_equality_expected':False,
        'reason':'storage protocol/context metadata differs, tensor values must not',
        'only_change':'teacher observation storage and corresponding source/replay work',
        'heldout_or_clinical_claim':False}


def cache_control(output,result,release):
    """Bind the actual fill/reuse receipt without loading observation arrays."""
    output=Path(output)
    def read(name):
        path=output/name
        if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=2*1024**2:
            raise ValueError('Bounded cache/visit metadata required')
        return json.loads(path.read_text())
    cache=read('teacher-cache.json');final=result.get('teacher_cache',{})
    pins=read('teacher-pins.json');costs=read('costs.json')
    if (cache.get('mode')!=STORAGE or cache.get('complete') is not True
            or cache.get('RL_reuse') is not False
            or cache.get('learning_protocol_hash')!=release['learning_protocol_hash']
            or canonical({k:v for k,v in final.items() if k not in ('trace_reuses','readout_rows')})!=canonical(cache)
            or type(final.get('trace_reuses')) is not int or final['trace_reuses']!=256
            or type(final.get('readout_rows')) is not int or final['readout_rows']!=5
            or any(type(cache.get(k)) is not int or cache[k]<0 for k in ('array_bytes','metadata_json_bytes'))
            or cache['array_bytes']+cache['metadata_json_bytes']>CACHE_BYTES
            or not isinstance(cache.get('traces'),list)
            or [r.get('subject') for r in cache['traces']]!=list(TRAIN)
            or semantic(cache['traces'])!=cache.get('cache_seal')):
        raise ValueError('Actual complete bounded teacher cache receipt required')
    for subject,row in zip(TRAIN,cache['traces']):
        teacher=read('teachers/'+subject+'/plan.json')
        replay=read('teachers/'+subject+'/native-replay.json')
        trace=read('teachers/'+subject+'/complete-trace.json')
        plan=teacher['plan'];geometry=replay['independent_geometry']
        if (row['plan_seal']!=teacher['plan_seal'] or semantic(plan)!=teacher['plan_seal']
                or row['context_hash']!=plan['context_hash'] or row['trace_seal']!=trace['trace_seal']
                or row['steps']!=len(plan['actions']) or row['stop_steps']!=plan['actions'].count('STOP')
                or row['observations']!=[d['observation_hash'] for d in trace['decisions']]
                or row['independent_replay_hash']!=semantic(geometry)
                or geometry.get('accepted') is not True
                or geometry.get('committed_history_hash')!=semantic(replay['metrics']['history'])
                or any(row[k]!=pins[subject][k] for k in ('context_hash','trace_seal','plan_seal','steps','stop_steps'))):
            raise ValueError('Cache entry differs from initial actual teacher/replay')
    if (sum(r['steps'] for r in cache['traces'])!=5 or sum(r['stop_steps'] for r in cache['traces'])!=4
            or sum(r['array_bytes'] for r in cache['traces'])!=cache['array_bytes']
            or sum(r['metadata_json_bytes'] for r in cache['traces'])!=cache['metadata_json_bytes']):
        raise ValueError('Cache totals changed')
    visits=costs['completed_patient_visits']
    if ([r['subject'] for r in visits]!=list(TRAIN)*2 or any(r['source_released'] is not True for r in visits)
            or [r['phase'] for r in visits]!=['offline.fixed_teacher_replay.'+s for s in TRAIN]
                +['deployment.reloaded_TRAIN_greedy_replay.'+s for s in TRAIN]):
        raise ValueError('Exact eight released source visits required')
    phases=costs['costs'];reuses=[r for name,r in phases.items() if '.cached_loss_backward.' in name]
    readouts=[r for name,r in phases.items() if name.startswith('offline.endpoint_cached_teacher_readout.')]
    logits=[r for name,r in phases.items() if name.startswith('offline.endpoint_teacher_logits.')]
    if (len(reuses)!=256 or sum(r.get('policy_forward_calls',0) for r in reuses)!=320
            or any(r.get(k,0)!=0 for r in reuses for k in ('native_preview_entries','native_transition_calls','inventory_request_calls','task_clone_calls'))
            or len(readouts)!=4 or len(logits)!=4 or sum(r.get('policy_forward_calls',0) for r in logits)!=5
            or any(r.get(k,0)!=0 for r in readouts+logits for k in ('native_preview_entries','native_transition_calls','inventory_request_calls','task_clone_calls'))
            or costs['total_policy_forward_calls']!=325+sum(r['steps'] for r in result['TRAIN_greedy'].values())):
        raise ValueError('Cache reuse performed extra geometry or wrong forward work')
    return {'cache_file_sha256':sha(output/'teacher-cache.json'),'cache_seal':cache['cache_seal'],
        'array_bytes':cache['array_bytes'],'metadata_json_bytes':cache['metadata_json_bytes'],
        'verified_entries':4,'initial_source_visits':4,'greedy_source_visits':4,
        'trace_reuses':256,'cached_loss_forwards':320,'cached_native_previews':0,
        'detached_teacher_readout_rows':5,'RL_reuse':False}
