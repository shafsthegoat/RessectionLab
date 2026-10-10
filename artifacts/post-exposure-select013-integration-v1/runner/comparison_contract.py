"""Metadata gates for one fixed public SELECT comparison; imports no model code."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = 'build/post-exposure-select013-comparison-v1/attempt-01'
TRAIN = ('ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045')
ENDPOINTS = ('IL64', 'RL8')
ARMS = ('observed_greedy', 'IL64', 'RL8', 'observed_beam')
OCCUPANCY = 'cerebrum_plus_supplied_tumor_with_preserved_partial_source_domain'
EXPOSURE = 'public-union-post-exposure-axis0-v1'
RL_DIRECTORY = 'build/post-exposure-learning-rl8-recovery-v1/RL8'
RL_RELEASE_SHA = '4da18b8bd21e6a54308d0b58b3d234c49c7351b56040e448ac7107a28a773e3e'
TEACHER_STEPS = {'ReMIND-002':1,'ReMIND-015':14,'ReMIND-018':1,'ReMIND-045':13}
UPSTREAM_RECOVERY_CONTRACT = 'build/post-exposure-learning-rl8-recovery-v1/pilot_contract.py'
UPSTREAM_RECOVERY_CONTRACT_SHA = '7f823d74b4da466056b195f943de184202d0766a2ad8822abd87a70232a2e65c'
COHORT = 'manifests/experiments/remind-component-cohort-v1.json'
COHORT_SHA = '326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05'
PUBLIC_INDEX = 'build/remind-select-public-preparation-v1/public-select-index-v2.json'
PUBLIC_SHA = 'adc7232f0fd05300f90078385bc2576150eccc882712e31bfc97534f68680d5f'
MODE = 'return_plus_opening_depth_volume_v1'
TARGET_CONTEXT = 'full-supplied-public-target-context-v1'
PROMOTED_SELECT_SOURCES = {
    'src/resectionlab/public_patient_factory.py':'fa5dbd25b707f345e3043bff488b62da48f12f96297365309165a908eb17258b',
    'src/resectionlab/patient_planning_admission.py':'96f43268f0b850e0b82ad6f3ad3e5d5a1bb20627887e7235b44904d7706726db',
    'src/resectionlab/patient_select_inference.py':'f7ca9fb8221e7c4db915ad5d109fdcdf2cb6912acf9be271fee725e6dcb7386a',
}
AXES = [[0,0],[-2,0],[2,0],[0,-2],[0,2],[-2,-2],[-2,2],[2,-2],[2,2],[-3,0],[3,0],[0,-3],[0,3]]
WORKER_SECONDS, PARENT_SECONDS = 900, 960
MEMORY_BYTES, OUTPUT_BYTES, SUPERVISION_BYTES = 3*1024**3, 128*1024**2, 16*1024**2
LIMITS = {'max_steps':24, 'max_optimizer_updates':0, 'max_native_previews':750000,
    'max_policy_forwards':48, 'worker_seconds':WORKER_SECONDS, 'memory_bytes':MEMORY_BYTES,
    'threads':1, 'search':{'max_calls':5640, 'beam_width':2, 'seconds':300}}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def semantic(value):
    return 'sha256:'+hashlib.sha256(canonical(value).encode()).hexdigest()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_pinned(ref, *, root=None):
    root = ROOT if root is None else Path(root)
    relative = Path(ref['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Repository-relative pinned metadata required')
    path = root/relative
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 2*1024**2:
        raise ValueError('Bounded regular metadata required: '+str(relative))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('Pinned metadata changed: '+str(relative))
    return json.loads(raw)

def select_public(index):
    rows = index['cases']
    if (len(rows)!=2 or [r['patient_id'] for r in rows]!=['ReMIND-013','ReMIND-037']
            or index['role']!='SELECT' or index['max_optimizer_updates']!=0
            or index['planned_case_denominator']!=2):
        raise ValueError('Original two-case SELECT denominator required')
    active, held = rows
    if (active['status']!='PUBLIC_PARTIAL_TARGET_CONDITION_QUALIFIED'
            or active['role']!='SELECT' or active['patient_group']!='ReMIND:013'
            or active['full_target_denominator_voxels']!=35260
            or active['supported_target_positive_voxels']!=1579
            or not active['path'] or not active['sha256']
            or held['status']!='HOLD_ENCODED_SOURCE_SUPPORT_ARTIFACT'
            or held['role']!='SELECT' or held['path'] is not None or held['sha256'] is not None
            or held['replacement'] is not None or held['kept_in_planned_denominator'] is not True):
        raise ValueError('Exact partial013 and held037 required; no replacement or mask repair')
    return active


def endpoint_refs(name, *, rl_result_sha256):
    # Required argument comes only from root's terminal review. Never infer a
    # final pin from a running result file or silently follow its current bytes.
    if name not in ENDPOINTS:raise ValueError('Fixed endpoint required')
    if (type(rl_result_sha256) is not str or len(rl_result_sha256)!=64
            or any(c not in '0123456789abcdef' for c in rl_result_sha256)):
        raise ValueError('Root-reviewed terminal RL result SHA is required')
    directory=Path('build/post-exposure-learning-v1/IL64' if name=='IL64' else RL_DIRECTORY)
    if name=='IL64':
        rows={k:ORIGINAL_IL[v] for k,v in {'release':'release','result':'result',
              'parent':'receipt','worker_final':'worker_final'}.items()}
        rows.update(status='original_failed_plus_accepted_replay',
            recovery_receipt=RECOVERY_INPUTS['receipt'])
    else:
        paths={'release':directory/'root-release.json','result':directory/'attempt-01/result.json',
            'parent':directory/'attempt-01.supervision/receipt.json',
            'worker_final':directory/'attempt-01.supervision/worker-final.json'}
        rows={k:{'path':str(p),'sha256':sha(ROOT/p)} for k,p in paths.items()}
        if rows['release']['sha256']!=RL_RELEASE_SHA or rows['result']['sha256']!=rl_result_sha256:
            raise ValueError('Exact terminal RL run differs from root-reviewed result')
        rows.update(status='terminal_bound')
    result=read_pinned(rows['result']);method='IL' if name=='IL64' else 'RL'
    rows.update(contexts={subject:{'path':str(directory/'attempt-01'/(subject+'-context.json')),
        'sha256':sha(ROOT/directory/'attempt-01'/(subject+'-context.json'))} for subject in TRAIN},
        checkpoint={'path':str(directory/'attempt-01'/(method+'-final.psckpt')),
            'sha256':result['checkpoints'][method]['sha256']})
    return rows


def comparison_world(protocol):
    execution = protocol['cohort_execution']
    return {key:execution[key] for key in ('max_steps','proposal_config','proposal_rule_hash',
        'search','retention_mode','task_condition','occupancy_condition','post_exposure_condition')} | {
        'public_target_context_variant':protocol['public_target_context_variant']}


def authenticate_endpoint(name, refs, *, root=None):
    """Metadata only; safe payload loader remains authority for weight bytes."""
    if name not in ENDPOINTS:raise ValueError('Fixed endpoint required')
    records={k:read_pinned(refs[k],root=root) for k in ('release','result','parent','worker_final')}
    release,result,parent,worker=(records[k] for k in ('release','result','parent','worker_final'))
    method='IL' if name=='IL64' else 'RL';updates=64 if name=='IL64' else 8
    if name=='IL64':
        if root is not None and Path(root)!=ROOT:
            raise ValueError('Historical IL recovery uses exact workspace pins')
        expected={k:ORIGINAL_IL[v] for k,v in {'release':'release','result':'result',
            'parent':'receipt','worker_final':'worker_final'}.items()}
        if refs.get('status')!='original_failed_plus_accepted_replay' or any(refs[k]!=v for k,v in expected.items()):
            raise ValueError('Original IL failure may not be rewritten as clean completion')
        evidence=recovered_il_evidence(refs['recovery_receipt'])
        recovery_parent=small(RECOVERY_INPUTS['receipt'])
        completion={'original_attempt_status':'failed_or_unresolved','original_result':refs['result'],
            'original_parent':refs['parent'],'separate_recovery':evidence['recovery_refs'],
            'fixed_endpoint_four_replays_accepted':True,
            'original_parent_seconds':parent['elapsed_seconds'],
            'original_parent_peak_rss_bytes':parent['sampled_peak_rss_bytes'],
            'separate_recovery_parent_seconds':recovery_parent['elapsed_seconds'],
            'separate_recovery_peak_rss_bytes':recovery_parent['sampled_peak_rss_bytes'],
            'additional_training_or_policy_forwards_in_recovery':0}
    else:
        if (refs.get('status')!='terminal_bound' or refs['release']['sha256']!=RL_RELEASE_SHA
                or refs['result']['path']!=RL_DIRECTORY+'/attempt-01/result.json'
                or parent['status']!='complete' or parent['exit_code']!=0 or parent['stop_reason'] is not None
                or parent['cleanup_errors'] or parent['final_owned_pids'] or not parent['worker_termination_confirmed']
                or parent['release_sha256']!=refs['release']['sha256'] or parent['result_sha256']!=refs['result']['sha256']
                or parent['worker_final_sha256']!=refs['worker_final']['sha256']
                or parent['source_index']!=release['source_index']
                or worker.get('result_sha256')!=refs['result']['sha256']
                or worker['canonical_result_sha256']!=refs['result']['sha256']
                or worker['status']!='complete_owned_matched_TRAIN_endpoint'
                or worker['method']!='RL' or worker['SELECT_EVAL_opened'] is not False
                or not parent['endpoint_control_sha256']
                or parent['endpoint_control_sha256']!=worker['endpoint_control_sha256']
                or not terminal_rl_complete(result)
                or release.get('matched_IL_recovery')!=RECOVERY_INPUTS['receipt']
                or release['output']!=RL_DIRECTORY+'/attempt-01'):
            raise ValueError('Own clean terminal scratch RL8 chain required')
        completion={'original_attempt_status':'complete','original_result':refs['result'],
            'original_parent':refs['parent'],'parent_seconds':parent['elapsed_seconds'],
            'parent_peak_rss_bytes':parent['sampled_peak_rss_bytes'],
            'fixed_endpoint_four_replays_accepted':True}
    protocol=release['learning_protocol'];execution=protocol['cohort_execution']
    if (release['status']!='released_one_attempt' or release['method']!=method
            or release['TRAIN']!=list(TRAIN) or release['new_four_training'] is not True
            or release['SELECT_EVAL_execution'] is not False
            or release['learning_protocol_hash']!=semantic(protocol)
            or protocol['version']!='fixed-four-TRAIN-post-exposure-learning-v1'
            or protocol['updates_per_method']!=updates
            or execution['patient_order']!=list(TRAIN) or execution['heldout_execution'] is not False
            or execution.get('post_exposure_condition')!=EXPOSURE
            or execution.get('occupancy_condition')!=OCCUPANCY
            or 'il_motion_supervision' in execution
            or execution['teacher_steps']!=list(TEACHER_STEPS.values())
            or execution['teacher_decisions']!=29
            or execution['teacher_source']!='fixed_saved_greedy_complete_histories_v1'
            or execution.get('il_teacher_weighting')!=('balanced_STOP_motion_CE_v1' if method=='IL' else None)
            or execution['teacher_observations']!=('cache_complete_replayed_TRAIN_teacher_traces_v1'
                if method=='IL' else 'recollect_complete_pinned_plan_each_IL_update')
            or execution.get('teacher_cache_payload_bytes')!=(256*1024**2 if method=='IL' else None)):
        raise ValueError('Exact unchanged post-exposure objective/endpoint required')
    if set(refs['contexts'])!=set(TRAIN):raise ValueError('Four exact new TRAIN contexts required')
    contexts={}
    for subject in TRAIN:
        row=read_pinned(refs['contexts'][subject],root=root)
        if (row['subject']!=subject or row['role']!='TRAIN' or row['patient_group']!='ReMIND:'+subject[-3:]
                or row['cohort_sha256']!=COHORT_SHA or row['learning_protocol_hash']!=semantic(protocol)
                or row['max_steps']!=24 or row['private_reference_in_task'] is not False
                or row['public_target_context_variant']!=TARGET_CONTEXT or row['occupancy_condition']!=OCCUPANCY
                or row['execution_kind']!='fixed_four_TRAIN_post_exposure_learning_v1'
                or row['post_exposure']['version']!=EXPOSURE):
            raise ValueError('Own complete TRAIN context differs: '+subject)
        contexts[row['patient_group']]=semantic(row)
    checkpoint=result['checkpoints'][method]
    if (checkpoint['path']!=method+'-final.psckpt' or not 0<checkpoint['bytes']<=8*1024**2
            or refs['checkpoint']!={'path':str(Path(release['output'])/checkpoint['path']),'sha256':checkpoint['sha256']}
            or (method=='RL' and checkpoint!=parent['checkpoints']['RL'])):
        raise ValueError('Terminal checkpoint descriptor differs')
    return {'protocol':protocol,'context_hashes':contexts,'method':method,'updates':updates,
        'checkpoint':refs['checkpoint'],'parameter_hash':checkpoint['parameter_hash'],
        'training_release_sha256':refs['release']['sha256'],'initial_parameter_hash':result['initial_parameter_hash'],
        'training_evidence':completion}


def authenticate_comparison(release, *, root=None):
    if set(release['endpoints'])!=set(ENDPOINTS): raise ValueError('Fixed matched IL64 and RL8 endpoints required')
    endpoints = {name:authenticate_endpoint(name, release['endpoints'][name], root=root) for name in ENDPOINTS}
    if len({row['initial_parameter_hash'] for row in endpoints.values()})!=1:
        raise ValueError('Matched endpoints must retain identical initial parameter identity')
    expected = comparison_world(endpoints[ENDPOINTS[0]]['protocol'])
    for endpoint in endpoints.values():
        if comparison_world(endpoint['protocol'])!=expected:
            raise ValueError('Comparison world/action geometry differs across TRAIN protocols')
    if (expected['max_steps']!=24 or expected['proposal_config']['max_candidates']!=120
            or expected['search']!=LIMITS['search'] or expected['retention_mode']!=MODE
            or expected['task_condition']!='PARTIAL_TARGET_PROGRESS'
            or expected['public_target_context_variant']!=TARGET_CONTEXT
            or expected['occupancy_condition']!=OCCUPANCY
            or expected['proposal_config'].get('obstruction_opening',False) is not False
            or expected['proposal_config']['offsets_source_voxels']!=AXES
            or expected['proposal_config']['intermediate_opening_mm']!=1.0
            or expected['post_exposure_condition']!=EXPOSURE
            or expected['proposal_config']['tool_footprint_opening'] is not True):
        raise ValueError('Fixed h24 cap120 depth-volume public context required')
    return endpoints

def complete_result(result):
    return (result.get('status')=='complete_post_exposure_SELECT013_comparison'
        and result.get('occupancy_condition')==OCCUPANCY and result.get('post_exposure_condition')==EXPOSURE
        and result.get('planned_SELECT_denominator')==2 and result.get('held_cases')==['ReMIND-037']
        and result.get('held_inputs_opened') is False and result.get('checkpoint_loads')==2
        and result.get('optimizer_attempts')==0 and result.get('gradient_attempts')==0
        and result.get('optimizer_updates_on_SELECT')==0 and result.get('private_reference_reads')==0
        and result.get('EVAL_opened') is False and set(result.get('arms',{}))==set(ARMS)
        and result.get('arm_order')==list(ARMS)
        and all(r.get('status')=='complete_replayed' for r in result['arms'].values())
        and result.get('total_policy_forward_calls')==sum(result['arms'][n]['steps'] for n in ENDPOINTS)
        and 0 <= result['total_policy_forward_calls'] <= LIMITS['max_policy_forwards'])

def source_guard(release, release_path, release_sha):
    if (release.get('status')!='released_one_attempt' or release.get('output')!=OUTPUT
            or release.get('arms')!=list(ARMS) or release.get('limits')!=LIMITS
            or release.get('public_index')!={'path':PUBLIC_INDEX,'sha256':PUBLIC_SHA}
            or release.get('automatic_retry') is not False or release.get('attempts')!=1
            or release.get('occupancy_condition')!=OCCUPANCY or release.get('post_exposure_condition')!=EXPOSURE
            or release.get('parent_seconds')!=PARENT_SECONDS or release.get('greedy_seconds')!=60
            or sha(release_path)!=release_sha):
        raise ValueError('Fixed complete comparison release required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    if head!=release['expected_head']: raise ValueError('Canonical HEAD changed')
    index=read_pinned(release['source_index'])
    if index['head']!=head: raise ValueError('Source index HEAD differs')
    for section in ('source_files','metadata_files'):
        for path, digest in index[section].items():
            relative=Path(path)
            if relative.is_absolute() or '..' in relative.parts or (ROOT/relative).is_symlink() or sha(ROOT/relative)!=digest:
                raise ValueError('Bound source/metadata changed: '+path)
    if sha(ROOT/COHORT)!=COHORT_SHA: raise ValueError('Original cohort changed')
    select_public(read_pinned(release['public_index']))
    authenticate_comparison(release)
    return index


# Exact previously reviewed historical evidence gates; no weights opened here.
small=read_pinned
ORIGINAL_IL = {'result': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/result.json', 'sha256': 'eedc4857c184d94913976cb5b59e01341e1b3a8bb0f001af43f8d079e3d87975'}, 'receipt': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01.supervision/receipt.json', 'sha256': 'c9702883bcbdb9fb524a99b88fde1cac464116d399fce8aa63d866b57b4a1c82'}, 'worker_final': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01.supervision/worker-final.json', 'sha256': '226d5a0e0efd2eafecd7301ff09bb7e302aa2a9a0bccbf818c00916cd98ff4fa'}, 'release': {'path': 'build/post-exposure-learning-v1/IL64/root-release.json', 'sha256': 'f0df97b2e0222b3656d755ce1ebdaee388b381ae8e2616e5a67e9c50dfdc888b'}, 'source_index': {'path': 'build/post-exposure-learning-v1/IL64/source-index.json', 'sha256': '126d67f10cd57255c8df47c42f1a92d2a4f1c49fd9476eaf199103b18c3a15f1'}, 'configuration': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/configuration.json', 'sha256': '55f11d31f34f5f44d697d25f010b1a6acd1a462533cf4ecf1010c36e8af1713b'}, 'reload': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/checkpoint-reload.json', 'sha256': 'd4441ca39c88921651639fd3bfb2896c1e04b7a3e5e693e3357cf5db6079865a'}, 'update64': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/IL/update-64/update.json', 'sha256': '1bda364b5b5410ee5bdb956ab097b1a366c1f8ec9091de7eab33533ddb3499f4'}, 'dynamics': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/training-dynamics.json', 'sha256': '6d3a58f1e44c433e287418cd165b891108320436d93f056a9913700be04a86cd'}, 'plan045': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/TRAIN-greedy/IL/ReMIND-045/plan.json', 'sha256': '7411153dabdb28c5d1a85c9d278db9671080231329eab434182dff677c417b84'}, 'plan_ReMIND-002': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/TRAIN-greedy/IL/ReMIND-002/plan.json', 'sha256': '1614f9d2206038c3bda4f85eb0906c31464715b305d11a03827d61b00396d09b'}, 'replay_ReMIND-002': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/TRAIN-greedy/IL/ReMIND-002/native-replay.json', 'sha256': '4611a32b328cd07d8c642e4246711a5255bd9b86435aaa290493c68316806d42'}, 'plan_ReMIND-015': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/TRAIN-greedy/IL/ReMIND-015/plan.json', 'sha256': '66dadfe3b434996fb03284757b6b76d32baf310d7154f122172e79008402bb58'}, 'replay_ReMIND-015': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/TRAIN-greedy/IL/ReMIND-015/native-replay.json', 'sha256': 'fc1f64be0a38aa2c3c192595d8b0cf5bf4a0464aaa3898f74a6f9b5421e92bf4'}, 'plan_ReMIND-018': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/TRAIN-greedy/IL/ReMIND-018/plan.json', 'sha256': '9c1b11d559158387dc0e92760973e825f437526274677b85ed83f6d24aca2000'}, 'replay_ReMIND-018': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/TRAIN-greedy/IL/ReMIND-018/native-replay.json', 'sha256': 'f4a61aa5ca95158d0deef861d11a7a892e728eb011d68cb9609f0a1f05bf0404'}}

ORIGINAL_IL_CHECKPOINT = {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/IL-final.psckpt', 'sha256': 'c174b92b09ed8eccd034a7bc8bc7cb9a4d64e09aa40d6f7a3d596ae88ea57c5c', 'bytes': 137636, 'parameter_hash': 'sha256:59506939917827f7555a61c930c33285dcb62feda52ec98576194a82e39587f3'}

RECOVERY_ROOT = 'build/post-exposure-IL045-evaluation-recovery-v1'

RECOVERY_INPUTS = {'receipt': {'path': 'build/post-exposure-IL045-evaluation-recovery-v1/attempt-01.supervision/receipt.json', 'sha256': 'c159b91ab2015dfd54dee677eb683b27e335a62657ba51838c398ed2d391a98e'}, 'result': {'path': 'build/post-exposure-IL045-evaluation-recovery-v1/attempt-01/result.json', 'sha256': '432535613f699604a4b0f3b0aaa0fedebbe932e5ca6b7006beedd66ff823a7ce'}, 'replay': {'path': 'build/post-exposure-IL045-evaluation-recovery-v1/attempt-01/native-replay.json', 'sha256': '529405a10f37c12f9ee4138fd0ef0d0824d7c3fba6f877802890971cc4bae862'}, 'release': {'path': 'build/post-exposure-IL045-evaluation-recovery-v1/root-release.json', 'sha256': 'e5dd29dc93460dc40c66a27dcf79986736797fd65ae9277e0e8026ea860d1797'}, 'source_index': {'path': 'build/post-exposure-IL045-evaluation-recovery-v1/source-index.json', 'sha256': '1fd10c334fc8a83bcf8141c010adf3948339632c1b3db6829c0ece8182011957'}}

def history_identity(history):
    return canonical([{k:v for k,v in row.items() if k!='outcome_scope'} for row in history])

def original_il_evidence():
    rows={key:small(ref) for key,ref in ORIGINAL_IL.items()}
    prior,receipt=rows['result'],rows['receipt'];pin=ORIGINAL_IL_CHECKPOINT
    if (prior['status']!='failed_or_unresolved' or prior['method']!='IL'
            or prior['TRAIN']!=list(TRAIN) or prior['optimizer_updates']!={'IL':64,'RL':0}
            or prior['failure']!={'type':'ValueError','message':'Independent native geometry rejected replay','committed':False}
            or prior['fresh_common_initialization_verified'] is not True
            or prior['teacher_decisions']!=29 or prior['teacher_steps']!=TEACHER_STEPS
            or prior['teacher_statuses']!={s:'complete_replayed' for s in TRAIN}
            or prior['loss_forward_calls']!=1856 or prior['teacher_logit_forwards']!=29
            or prior['checkpoint_loads']!=1 or prior['search_calls']!=0
            or prior['private_reference_reads']!=0 or prior['SELECT_EVAL_opened'] is not False
            or set(prior['TRAIN_greedy'])!=set(TRAIN[:-1])
            or receipt['status']!='failed_or_unresolved' or receipt['exit_code']!=1
            or receipt['cleanup_errors'] or receipt['final_owned_pids']
            or receipt['worker_termination_confirmed'] is not True
            or receipt['result_sha256']!=ORIGINAL_IL['result']['sha256']
            or receipt['release_sha256']!=ORIGINAL_IL['release']['sha256']
            or receipt['source_index']!=rows['release']['source_index']
            or rows['worker_final']['canonical_result_sha256']!=receipt['result_sha256']):
        raise ValueError('Exact original failed IL64 history and clean process termination required')
    checkpoint=prior['checkpoints']['IL']
    if (checkpoint!={'path':'IL-final.psckpt','sha256':pin['sha256'],'bytes':pin['bytes'],'parameter_hash':pin['parameter_hash']}
            or rows['reload']!={'completed_updates':64,'exact_parameter_match':True,'parameter_hash':pin['parameter_hash'],'sha256':pin['sha256']}
            or rows['update64']['completed_updates']!=64 or rows['update64']['after_parameter_hash']!=pin['parameter_hash']
            or len(rows['dynamics']['updates'])!=64
            or rows['dynamics']['updates'][-1]['after_parameter_hash']!=pin['parameter_hash']):
        raise ValueError('Exact saved/reloaded final64 checkpoint and update chain required')
    # Descriptor joined above; payload is authenticated only by load_select_checkpoint.
    for subject in TRAIN[:-1]:
        plan=rows['plan_'+subject];replay=rows['replay_'+subject];audit=replay['independent_geometry'];metrics=replay['metrics']
        if (prior['TRAIN_greedy'][subject]['complete'] is not True
                or prior['TRAIN_greedy'][subject]['parameter_hash']!=pin['parameter_hash']
                or plan['plan_seal']!=semantic(plan['plan'])
                or plan['plan_seal']!=prior['TRAIN_greedy'][subject]['plan_seal']
                or plan['plan']['parameter_hash']!=pin['parameter_hash'] or plan['plan']['learning_updates']!=64
                or audit['accepted'] is not True or audit['complete_episode'] is not True
                or audit['geometry']['feasible'] is not True or audit['geometry']['complete_tool_checked'] is not True
                or audit['committed_history_hash']!=semantic(metrics['history'])
                or history_identity(plan['plan']['history'])!=history_identity(metrics['history'])):
            raise ValueError('Original three accepted IL histories changed: '+subject)
    plan=rows['plan045']
    if (plan['plan_seal']!=semantic(plan['plan']) or plan['plan']['parameter_hash']!=pin['parameter_hash']
            or plan['plan']['learning_updates']!=64 or plan['plan']['terminal_reason']!='HORIZON'
            or len(plan['plan']['actions'])!=24):
        raise ValueError('Exact original frozen045 horizon plan required')
    return rows

def recovered_il_evidence(recovery_ref):
    # The pending placeholder is never executable, even if the old negative
    # diagnostic or an unbound similarly named file exists.
    if (not isinstance(recovery_ref,dict) or set(recovery_ref)!={'path','sha256'}
            or recovery_ref!=RECOVERY_INPUTS['receipt']
            or not isinstance(recovery_ref['sha256'],str) or len(recovery_ref['sha256'])!=64
            or any(c not in '0123456789abcdef' for c in recovery_ref['sha256'])):
        raise ValueError('Root-pinned accepted045 recovery receipt is still unavailable')
    original=original_il_evidence();receipt=small(recovery_ref)
    refs={'receipt':recovery_ref,'result':{'path':RECOVERY_ROOT+'/attempt-01/result.json','sha256':receipt['result_sha256']},
          'release':{'path':RECOVERY_ROOT+'/root-release.json','sha256':receipt['release_sha256']},
          'source_index':receipt['source_index']}
    if refs['source_index']['path']!=RECOVERY_ROOT+'/source-index.json':raise ValueError('Fixed recovery source index required')
    result=small(refs['result']);release=small(refs['release']);index=small(refs['source_index'])
    refs['replay']={'path':RECOVERY_ROOT+'/attempt-01/native-replay.json','sha256':result['native_replay_sha256']}
    if refs!=RECOVERY_INPUTS:raise ValueError('Exact accepted recovery inputs changed')
    replay=small(refs['replay']);audit=replay['independent_geometry'];metrics=replay['metrics'];plan=original['plan045']['plan']
    if (receipt['status']!='complete' or receipt['exit_code']!=0 or receipt['cleanup_errors']
            or receipt['remaining_owned_pids'] or receipt['worker_termination_confirmed'] is not True
            or result['status']!='complete_saved_plan_diagnostic' or result['subject']!='ReMIND-045'
            or result['independent_accepted'] is not True or result['full_history_equal'] is not True
            or result['committed_actions']!=24 or result['source_visits']!=1
            or any(result[k]!=0 for k in ('checkpoint_loads','optimizer_updates','policy_forwards','teacher_search_calls'))
            or result['training_admitted'] is not False or result['release_sha256']!=refs['release']['sha256']
            or result['checkpoint_sha256']!=ORIGINAL_IL_CHECKPOINT['sha256']
            or result['saved_plan_sha256']!=ORIGINAL_IL['plan045']['sha256']
            or result['original_IL_result_sha256']!=ORIGINAL_IL['result']['sha256']
            or result['original_IL_parent_receipt_sha256']!=ORIGINAL_IL['receipt']['sha256']
            or result['saved_plan_seal']!=original['plan045']['plan_seal']
            or result['parameter_hash']!=ORIGINAL_IL_CHECKPOINT['parameter_hash']
            or result['prior_learning_updates']!=64 or result['source_hash']!=plan['source_hash']
            or result['context_hash']!=plan['context_hash'] or result['decision_model_hash']!=plan['decision_model_hash']
            or result['saved_history_hash']!=semantic(plan['history'])
            or release['execution_released'] is not True or release['subject']!='ReMIND-045'
            or release['parameter_hash']!=ORIGINAL_IL_CHECKPOINT['parameter_hash']
            or release['source_index']!=refs['source_index'] or index['head']!=release['expected_head']
            or audit['accepted'] is not True or audit['complete_episode'] is not True
            or audit['geometry']['feasible'] is not True or audit['geometry']['complete_tool_checked'] is not True
            or audit['geometry']['action_count']!=24 or audit['geometry']['failures']
            or result['independent_geometry']!=audit['geometry']
            or audit['committed_history_hash']!=semantic(metrics['history'])
            or metrics['steps']!=24 or metrics['terminated'] is not True
            or audit['source_hash']!=plan['source_hash'] or metrics['source_hash']!=plan['source_hash']
            or audit['decision_model_hash']!=plan['decision_model_hash'] or metrics['decision_model_hash']!=plan['decision_model_hash']
            or [h['action_id'] for h in metrics['history']]!=plan['actions']
            or history_identity(metrics['history'])!=history_identity(plan['history'])):
        raise ValueError('Exact accepted045 reevaluation and clean owned completion required')
    joins={'failed_result':'result','failed_receipt':'receipt','original_release':'release',
           'original_source_index':'source_index','configuration':'configuration','plan':'plan045'}
    if any(release['inputs'][key]!=ORIGINAL_IL[prior] for key,prior in joins.items()):
        raise ValueError('Recovery must bind the untouched original failed IL evidence')
    if release['inputs']['checkpoint']!={k:ORIGINAL_IL_CHECKPOINT[k] for k in ('path','sha256')}:
        raise ValueError('Recovery checkpoint identity changed')
    # Bind the corrected checker used by recovery to this SELECT execution source,
    # without pretending the old failed endpoint ran under it.
    for path in ('src/resectionlab/evaluation.py','src/resectionlab/patient_planning_preflight.py'):
        if index['files'].get(path)!=sha(ROOT/path):raise ValueError('SELECT/recovery checker source differs: '+path)
    return {'original':original,'recovery':result,'recovery_refs':refs,
        'original_status_preserved':'failed_or_unresolved','reevaluation_accepted':True}

def terminal_rl_complete(result):
    method=result.get('method');cfg=({'updates':8,'source_visits':44,'teacher_trace_reuses':0,
        'native_preview_cap':385440,'policy_forward_cap':1661} if method=='RL' else {})
    if not cfg:return False
    return (result.get('status')=='complete_matched_TRAIN_endpoint'
        and result.get('TRAIN')==list(TRAIN) and result.get('post_exposure_condition')==EXPOSURE
        and result.get('fresh_common_initialization_verified') is True
        and result.get('optimizer_updates')=={m:cfg['updates'] if m==method else 0 for m in ('IL','RL')}
        and (result.get('loss_forward_calls')==1856 if method=='IL' else 32<=result.get('loss_forward_calls',0)<=768)
        and result.get('teacher_logit_forwards')==29 and result.get('checkpoint_loads')==1
        and result.get('completed_source_visits')==cfg['source_visits']
        and result.get('teacher_trace_reuses')==cfg['teacher_trace_reuses']
        and result.get('search_calls')==0 and result.get('private_reference_reads')==0 and result.get('SELECT_EVAL_opened') is False
        and result.get('teacher_decisions')==29
        and result.get('teacher_steps')==TEACHER_STEPS
        and 0<=result.get('native_preview_entries',-1)<=cfg['native_preview_cap']
        and result.get('total_policy_forward_calls')==result.get('loss_forward_calls',0)*(1 if method=='IL' else 2)+29+sum(r.get('steps',0) for r in result.get('TRAIN_greedy',{}).values())
        and result.get('total_policy_forward_calls',cfg['policy_forward_cap']+1)<=cfg['policy_forward_cap']
        and result.get('teacher_statuses')=={s:'complete_replayed' for s in TRAIN}
        and set(result.get('checkpoints',{}))=={method} and set(result.get('TRAIN_greedy',{}))==set(TRAIN)
        and all(result['TRAIN_greedy'][s].get('complete') is True for s in TRAIN)
        and result.get('selection_readiness',{}).get('ready') is True
        and result.get('selection_readiness',{}).get('execution_admitted') is False)
