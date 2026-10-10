"""One fixed64 balanced teacher IL follow-on; source preparation grants no execution."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUTPUT='build/balanced-teacher-il64-v1/attempt-01'
TRAIN=('ReMIND-008','ReMIND-010','ReMIND-020','ReMIND-025')
CLOSED={'SELECT':['ReMIND-013','ReMIND-037'],'EVAL':['ReMIND-067']}
MODE='return_plus_opening_depth_volume_v1'
WEIGHTING='balanced_STOP_motion_CE_v1'
PUBLIC_INDEX='build/remind-planning-qc-v1/public-cohort-v1/public-manifest-index.json'
PUBLIC_SHA='11b5c1ae8f9b2c03695e7eb0768d2cc340db68631f3ddb50ba365910fe0187f7'
COHORT='manifests/experiments/remind-component-cohort-v1.json'
COHORT_SHA='326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05'
WORKER_SECONDS=3540
PARENT_SECONDS=3600
MEMORY_BYTES=3*1024**3
OUTPUT_BYTES=128*1024**2
SUPERVISION_BYTES=16*1024**2
# The canonical two-method context grows with64; this narrower executed-work
# envelope caps the IL-only driver. Preview sizing is conservative, not cost.
EXECUTION={'optimizer_updates':64,'loss_forwards':320,'teacher_logit_forwards':5,
    'max_policy_forwards':421,'max_native_previews':2347680,'source_visits':268,
    'checkpoint_loads':1,'search_calls':0,'RL_updates':0,'TRAIN_greedy_replays':4}
BASELINE_FILES = {'release': {'path': 'build/cross-patient-planning-v1/fixed-four-train-eight-updates-v1/root-release.json', 'sha256': 'c1cde8eb92f9df5d85ccb91ed0e591adb1da70e011582f3be116d237fecf72db'}, 'result': {'path': 'build/cross-patient-planning-v1/fixed-four-train-eight-updates-v1/attempt-01/result.json', 'sha256': 'b75bd91d284cacba2c980283424586ca506799919f6850175b6f4eeea4315aeb'}, 'parent': {'path': 'build/cross-patient-planning-v1/fixed-four-train-eight-updates-v1/attempt-01.supervision/receipt.json', 'sha256': 'e1f18ca5dae011057f0c945690d71e9ba909041f7aa495e5ce4493e57f805fa2'}, 'source_index': {'path': 'build/cross-patient-planning-v1/fixed-four-train-eight-updates-v1/source-index.json', 'sha256': '3bf24bf5038c20b04e44281df76411b61588efd207c75c0fddc518d969c04ea6'}, 'teacher_ReMIND-008': {'path': 'build/cross-patient-planning-v1/fixed-four-train-eight-updates-v1/attempt-01/teachers/ReMIND-008/plan.json', 'sha256': '8b3cd2f7aad5308481534154cf55a400c8791dc4e18ba0275756a52a4bb2d502'}, 'state_ReMIND-008_0': {'path': 'build/five-state-policy-diagnostic-v1/attempt-01/ReMIND-008/state-00.json', 'sha256': '73d3f459f4a93b5d6851b60b36ad3268d4c5ecb6f9e279cc836e73a8a7b5e855'}, 'teacher_ReMIND-010': {'path': 'build/cross-patient-planning-v1/fixed-four-train-eight-updates-v1/attempt-01/teachers/ReMIND-010/plan.json', 'sha256': '1e862c792a8dafaf1102ffe2968b24b6272eb249426889efe1231eda53f49d68'}, 'state_ReMIND-010_0': {'path': 'build/five-state-policy-diagnostic-v1/attempt-01/ReMIND-010/state-00.json', 'sha256': '2c7cec5c1e6e7c3fb481b3c23a30fc4fa362b78a5ef5a96e9d4594176a322653'}, 'teacher_ReMIND-020': {'path': 'build/cross-patient-planning-v1/fixed-four-train-eight-updates-v1/attempt-01/teachers/ReMIND-020/plan.json', 'sha256': '7e5c5ee0a1068e88d171180622f705be68bd913032acdc78eaef070d20637e7e'}, 'state_ReMIND-020_0': {'path': 'build/five-state-policy-diagnostic-v1/attempt-01/ReMIND-020/state-00.json', 'sha256': 'f5d823d22855cb4ca2afbb631ad0efbc486a04f94e9d5311cbd2b2ff467c2e29'}, 'teacher_ReMIND-025': {'path': 'build/cross-patient-planning-v1/fixed-four-train-eight-updates-v1/attempt-01/teachers/ReMIND-025/plan.json', 'sha256': '843c09b795a684147dab7d479bcc00b45dcbee8fb4f7054da57a1f25f0ba90bb'}, 'state_ReMIND-025_0': {'path': 'build/five-state-policy-diagnostic-v1/attempt-01/ReMIND-025/state-00.json', 'sha256': '13fa5914bcf4cb88d0c593390acc2bae818fd000cf793e209fd268153118f5ae'}, 'state_ReMIND-025_1': {'path': 'build/five-state-policy-diagnostic-v1/attempt-01/ReMIND-025/state-01.json', 'sha256': '3f5aca840d6bb790a4bad950eac24bb03adba4ed59c73a01582c4108ae04e2a5'}, 'diagnostic_result': {'path': 'build/five-state-policy-diagnostic-v1/attempt-01/result.json', 'sha256': '9b3bfa60e67277091f6b04c35eea6697013e6cbbca211cb94a5229ec4ac73420'}}

BALANCED_FILES = {'release': {'path': 'build/balanced-teacher-il-v1/root-release.json', 'sha256': '63acce8556bda8a23552e8007b6851e11d80a022c4920ca3838104df6117e8f3'}, 'result': {'path': 'build/balanced-teacher-il-v1/attempt-01/result.json', 'sha256': 'e183018bc21400cc87b377b7cc7ee87eb0a8943ea09b88ff9b40b10cc10bebd3'}, 'parent': {'path': 'build/balanced-teacher-il-v1/attempt-01.supervision/receipt.json', 'sha256': 'ecf5b97efda838f7c1eafbefe727322fcdbd7b48a69ef14b8f78c1392455567d'}, 'source_index': {'path': 'build/balanced-teacher-il-v1/source-index.json', 'sha256': '6bfcaa037b3c9a350e181580f4c3d9dd09c8c9afc8e6675fde6d9a9187d18911'}, 'update_01': {'path': 'build/balanced-teacher-il-v1/attempt-01/IL/update-01/update.json', 'sha256': '87c7842af6288535f25a5edc03257ac6271909dd5938ec759c66defbf6882d85'}, 'update_02': {'path': 'build/balanced-teacher-il-v1/attempt-01/IL/update-02/update.json', 'sha256': 'f289f987b4cf1d68583fd5de897e66ec89b2c3f83704ff0b04014a1805af0e15'}, 'update_03': {'path': 'build/balanced-teacher-il-v1/attempt-01/IL/update-03/update.json', 'sha256': 'f802739dbae030cfd92dab5dd3c55ca20c3a4338daa43991074e03d97d3dc3c7'}, 'update_04': {'path': 'build/balanced-teacher-il-v1/attempt-01/IL/update-04/update.json', 'sha256': 'b579b9720c9b829774ba3d01f8cc361b56186efae120dc847f5db2dce73102c0'}, 'update_05': {'path': 'build/balanced-teacher-il-v1/attempt-01/IL/update-05/update.json', 'sha256': '306794a6a57bb5923547afe95e4e79fc5c5f87032b0c3c244652be5dc30a0407'}, 'update_06': {'path': 'build/balanced-teacher-il-v1/attempt-01/IL/update-06/update.json', 'sha256': 'e45b7afb4f66ee608eb76d796c122116871ebef0bbc3bbc798574e2c38d8fe3e'}, 'update_07': {'path': 'build/balanced-teacher-il-v1/attempt-01/IL/update-07/update.json', 'sha256': 'aecaf1e14c864cba7fec8c772378332223e180010e0d2d225ac85f21f2d2fc50'}, 'update_08': {'path': 'build/balanced-teacher-il-v1/attempt-01/IL/update-08/update.json', 'sha256': 'a7082f6838fc80894bcf40bb3bca2c04478e9933831c5d12b70d9e8ca24993e7'}, 'state_ReMIND-008_0': {'path': 'build/balanced-teacher-il-v1/attempt-01/teacher-readout/ReMIND-008/state-00.json', 'sha256': '49c3d73643f266bf184734e00138446ef92310005cb88d5fd01fdfaa055f2634'}, 'state_ReMIND-010_0': {'path': 'build/balanced-teacher-il-v1/attempt-01/teacher-readout/ReMIND-010/state-00.json', 'sha256': '500e5f880884f512e9cb012a30f38f75ebbc45bd38fbf40f6a2fc2a01892687a'}, 'state_ReMIND-020_0': {'path': 'build/balanced-teacher-il-v1/attempt-01/teacher-readout/ReMIND-020/state-00.json', 'sha256': 'c32b3cd8d6f93906f5ca5f9b7ad6a2b6aa582149ca764ae30f9cc7abb520d988'}, 'state_ReMIND-025_0': {'path': 'build/balanced-teacher-il-v1/attempt-01/teacher-readout/ReMIND-025/state-00.json', 'sha256': '0744e03c9007c0085c0d8ebea07ad2f4be2a8d7d3e6c881fc8d836205dfbbb26'}, 'state_ReMIND-025_1': {'path': 'build/balanced-teacher-il-v1/attempt-01/teacher-readout/ReMIND-025/state-01.json', 'sha256': 'b95ad2a884593fc1cc8579543addc93fb03fd572a7acce1d258b136784936ac2'}}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def semantic(value):return 'sha256:'+hashlib.sha256(canonical(value).encode()).hexdigest()

def read_pinned(ref):
    path=ROOT/ref['path']
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=2*1024**2 or sha(path)!=ref['sha256']:
        raise ValueError('Bounded pinned baseline metadata changed: '+ref['path'])
    return json.loads(path.read_text())

def require_baseline(protocol=None):
    records={k:read_pinned(ref) for k,ref in BASELINE_FILES.items()}
    result=records['result']; parent=records['parent']; old=records['release']
    if (parent['status']!='complete' or parent['exit_code']!=0 or parent['cleanup_errors']
            or parent['final_owned_pids'] or parent['worker_termination_confirmed'] is not True
            or parent['release_sha256']!=BASELINE_FILES['release']['sha256']
            or parent['result_sha256']!=BASELINE_FILES['result']['sha256']
            or parent['source_index']!=old['source_index']
            or old['source_index']!=BASELINE_FILES['source_index']
            or result['status']!='complete_TRAIN_only_not_heldout_performance'
            or result['optimizer_updates']!={'IL':8,'RL':8} or result['SELECT_EVAL_opened'] is not False
            or records['diagnostic_result']['status']!='complete_fixed_five_state_readout_and_native_greedy'):
        raise ValueError('Clean completed eight-update baseline and saved readout required')
    if protocol is not None:
        expected=json.loads(canonical(old['learning_protocol']))
        expected['cohort_execution']['il_teacher_weighting']=WEIGHTING
        expected['updates_per_method']=64
        if canonical(protocol)!=canonical(expected):
            raise ValueError('Only fixed updates64 and balanced teacher weighting may differ from original unweighted protocol')
    labels=stops=0
    for subject in TRAIN:
        saved=records['teacher_'+subject]; plan=saved['plan']
        if (semantic(plan)!=saved['plan_seal'] or plan['max_steps']!=24 or plan['learning_updates']!=0
                or plan['parameter_hash'] is not None or plan['architecture_hash'] is not None
                or plan['actions']!=[r['action_id'] for r in plan['history']]
                or plan['actions'][-1]!='STOP' or len(plan['actions'])!=(2 if subject=='ReMIND-025' else 1)):
            raise ValueError('Fixed complete teacher plan changed: '+subject)
        labels+=len(plan['actions']); stops+=plan['actions'].count('STOP')
        for step,action in enumerate(plan['actions']):
            row=records[f'state_{subject}_{step}']
            if row['subject']!=subject or row['step']!=step or row['teacher_action']!=action:
                raise ValueError('Saved five-state diagnostic differs from teacher')
    if (labels,stops)!=(5,4):raise ValueError('Exact five labels/four STOP/one motion required')
    records['balanced']=require_balanced_baseline(protocol)
    return records

def canonical_configuration():
    from resectionlab.core import thaw_json
    from resectionlab.native_proposals import NominalCavityProposalConfig
    from resectionlab.patient_planning_cohort_spec import sequential_learning_protocol,sequential_limits
    protocol=sequential_learning_protocol(updates=64,max_steps=24,
        search={'max_calls':5640,'beam_width':2,'seconds':300},
        proposal_config=NominalCavityProposalConfig(max_candidates=120,
            intermediate_opening_mm=1.,tool_footprint_opening=True),retention_mode=MODE,
        il_teacher_weighting=WEIGHTING)
    limits=sequential_limits(protocol,worker_seconds=WORKER_SECONDS,memory_bytes=MEMORY_BYTES,output_bytes=OUTPUT_BYTES)
    return thaw_json(protocol),thaw_json(limits)

def validate_release(release):
    if (release.get('status')!='released_one_attempt' or release.get('output')!=OUTPUT
            or release.get('TRAIN')!=list(TRAIN) or release.get('closed_roles')!=CLOSED
            or release.get('SELECT_EVAL_execution') is not False or release.get('automatic_retry') is not False
            or type(release.get('attempts')) is not int or release['attempts']!=1
            or release.get('execution_limits')!=EXECUTION
            or any(type(v) is not int for v in release.get('execution_limits',{}).values())
            or type(release.get('parent_seconds')) is not int or release['parent_seconds']!=PARENT_SECONDS
            or type(release.get('supervision_bytes')) is not int or release['supervision_bytes']!=SUPERVISION_BYTES
            or release.get('baseline')!=BASELINE_FILES or release.get('balanced_baseline')!=BALANCED_FILES
            or release.get('public_manifest_index')!={'path':PUBLIC_INDEX,'sha256':PUBLIC_SHA}):
        raise ValueError('Exact one-attempt balanced IL release required')
    expected={'max_steps':24,'max_optimizer_updates':128,'max_native_previews':7297920,
        'max_policy_forwards':18624,'worker_seconds':WORKER_SECONDS,'memory_bytes':MEMORY_BYTES,
        'threads':1,'search':{'max_calls':5640,'beam_width':2,'seconds':300},
        'output_bytes':OUTPUT_BYTES,'checkpoint_bytes':8*1024**2}
    limits=release['cohort_limits']
    if (canonical(limits)!=canonical(expected)
            or any(type(v) is not int for k,v in limits.items() if k!='search')
            or any(type(v) is not int for v in limits['search'].values())
            or canonical(release['limits'])!=canonical({k:v for k,v in limits.items() if k not in ('output_bytes','checkpoint_bytes')})
            or release['learning_protocol_hash']!=semantic(release['learning_protocol'])):
        raise ValueError('Exact canonical factory envelope required')
    require_baseline(release['learning_protocol'])

def source_guard(release,release_path,release_sha):
    validate_release(release)
    if sha(release_path)!=release_sha:raise ValueError('Release changed')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    if head!=release['expected_head']:raise ValueError('Canonical HEAD changed')
    ref=release['source_index']
    if sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Source index changed')
    index=json.loads((ROOT/ref['path']).read_text())
    if index['head']!=head:raise ValueError('Index HEAD differs')
    for section in ('source_files','metadata_files'):
        for relative,expected in index[section].items():
            path=Path(relative)
            if path.is_absolute() or '..' in path.parts or (ROOT/path).is_symlink() or sha(ROOT/path)!=expected:
                raise ValueError('Bound regular source/metadata changed: '+relative)
    return index

def complete_result(result):
    return (result.get('status')=='complete_balanced_IL64_TRAIN_only'
        and result.get('optimizer_updates')=={'IL':64,'RL':0}
        and result.get('loss_forward_calls')==320 and result.get('teacher_logit_forwards')==5
        and result.get('checkpoint_loads')==1 and result.get('search_calls')==0
        and result.get('completed_source_visits')==268
        and result.get('first_eight_update_control',{}).get('exact_match') is True
        and result.get('training_dynamics_updates')==64
        and result.get('teacher_statuses')=={s:'complete_replayed' for s in TRAIN}
        and result.get('selection_readiness',{}).get('ready') is True
        and result.get('selection_readiness',{}).get('execution_admitted') is False
        and result.get('SELECT_EVAL_opened') is False
        and set(result.get('checkpoints',{}))=={'IL'}
        and set(result.get('TRAIN_greedy',{}))==set(TRAIN)
        and all(result['TRAIN_greedy'][s].get('complete') is True for s in TRAIN))

def require_balanced_baseline(protocol=None):
    records={k:read_pinned(ref) for k,ref in BALANCED_FILES.items()}
    result=records['result'];parent=records['parent'];release=records['release']
    if (parent['status']!='complete' or parent['exit_code']!=0 or parent['cleanup_errors']
            or parent['final_owned_pids'] or parent['worker_termination_confirmed'] is not True
            or parent['release_sha256']!=BALANCED_FILES['release']['sha256']
            or parent['result_sha256']!=BALANCED_FILES['result']['sha256']
            or parent['source_index']!=BALANCED_FILES['source_index']
            or release['source_index']!=BALANCED_FILES['source_index']
            or result['status']!='complete_balanced_IL_TRAIN_only'
            or result['optimizer_updates']!={'IL':8,'RL':0} or result['loss_forward_calls']!=40
            or result['SELECT_EVAL_opened'] is not False):
        raise ValueError('Clean completed balanced8 baseline required')
    if protocol is not None:
        expected=json.loads(canonical(release['learning_protocol']));expected['updates_per_method']=64
        if canonical(protocol)!=canonical(expected):
            raise ValueError('Only fixed update count8 to64 may change from balanced8 protocol')
    previous=result['initial_parameter_hash']
    for i in range(1,9):
        row=records[f'update_{i:02d}']
        if (row['method']!='IL' or row['completed_updates']!=i or row['optimizer_updates']!=1
                or row['before_parameter_hash']!=previous or row['loss_forward_calls']!=5
                or row['IL_reduction']!=WEIGHTING):
            raise ValueError('Balanced8 numerical update chain changed')
        previous=row['after_parameter_hash']
    if previous!=result['checkpoints']['IL']['parameter_hash']:
        raise ValueError('Balanced8 final update does not bind checkpoint')
    return records

def first_eight_update_control(output,release):
    baseline=require_balanced_baseline(release['learning_protocol']);rows=[]
    for i in range(1,9):
        path=Path(output)/'IL'/f'update-{i:02d}'/'update.json'
        if path.is_symlink() or not path.is_file() or path.stat().st_size>2*1024**2:
            raise ValueError('Bounded first-eight update receipt required')
        actual=json.loads(path.read_text());prior=baseline[f'update_{i:02d}']
        excluded={'context_hashes','trace_seals'}
        if canonical({k:v for k,v in actual.items() if k not in excluded})!=canonical({k:v for k,v in prior.items() if k not in excluded}):
            raise ValueError('Exact first-eight numerical control differs at update '+str(i))
        rows.append({'update':i,'actual_sha256':sha(path),'baseline_sha256':BALANCED_FILES[f'update_{i:02d}']['sha256'],
            'loss':actual['loss'],'gradient_norm_before_clip':actual['gradient_norm_before_clip'],
            'before_parameter_hash':actual['before_parameter_hash'],'after_parameter_hash':actual['after_parameter_hash']})
    return {'exact_match':True,'updates':rows,'excluded_metadata_fields':['context_hashes','trace_seals'],
        'scope':'all remaining numerical/scalar update fields and canonical full tensor hashes identical'}

def endpoint_control(output,release):
    baseline=require_baseline(release['learning_protocol']);balanced=baseline['balanced']
    result=json.loads((Path(output)/'result.json').read_text())
    if not complete_result(result) or result['initial_parameter_hash']!=balanced['result']['initial_parameter_hash']:
        raise ValueError('Complete fixed64 endpoint with identical initial tensors required')
    matched=first_eight_update_control(output,release)
    if canonical(matched)!=canonical(result['first_eight_update_control']):
        raise ValueError('Result does not bind exact first-eight numerical control')
    dynamics=Path(output)/'training-dynamics.json'
    if (dynamics.is_symlink() or not dynamics.is_file() or dynamics.stat().st_size>2*1024**2
            or sha(dynamics)!=result['training_dynamics_sha256']
            or len(json.loads(dynamics.read_text())['updates'])!=64):
        raise ValueError('Complete64 loss/gradient dynamics changed')
    return {'status':'complete_matched_initialization_first_eight_and_fixed_teachers',
        'initial_parameter_hash':result['initial_parameter_hash'],'baseline':BALANCED_FILES,
        'only_protocol_change':'updates_per_method8_to64','teacher_labels':5,'loss_forwards':320,
        'comparison_scope':'same balanced objective/world/labels/initialization; fixed64 endpoint, no intermediate selection',
        'first_eight_update_control':matched,'training_dynamics_sha256':result['training_dynamics_sha256'],
        'balanced8_TRAIN_greedy':balanced['result']['TRAIN_greedy'],
        'balanced64_TRAIN_greedy':result['TRAIN_greedy'],
        'balanced8_parameter_hash':balanced['result']['checkpoints']['IL']['parameter_hash'],
        'balanced64_parameter_hash':result['checkpoints']['IL']['parameter_hash'],
        'heldout_or_clinical_claim':False}
