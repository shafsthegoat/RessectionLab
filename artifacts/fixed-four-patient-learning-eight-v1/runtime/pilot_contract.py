"""One fixed eight-update four-TRAIN follow-on; constants grant no execution."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
OUTPUT='build/cross-patient-planning-v1/fixed-four-train-eight-updates-v1/attempt-01'
MODE='return_plus_opening_depth_volume_v1'
TRAIN=('ReMIND-008','ReMIND-010','ReMIND-020','ReMIND-025')
CLOSED={'SELECT':['ReMIND-013','ReMIND-037'],'EVAL':['ReMIND-067']}
PUBLIC_INDEX='build/remind-planning-qc-v1/public-cohort-v1/public-manifest-index.json'
PUBLIC_SHA='11b5c1ae8f9b2c03695e7eb0768d2cc340db68631f3ddb50ba365910fe0187f7'
COHORT='manifests/experiments/remind-component-cohort-v1.json'
COHORT_SHA='326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05'
WORKER_SECONDS=3540
PARENT_SECONDS=3600
MEMORY_BYTES=3*1024**3
OUTPUT_BYTES=128*1024**2
SUPERVISION_BYTES=16*1024**2
BASELINE_DIRECTORY='build/cross-patient-planning-v1/fixed-four-train-pilot-v1'
BASELINE_FILES={
    'release':{'path':BASELINE_DIRECTORY+'/root-release.json','sha256':'01e428eed049432a051f8f4433b6d1da94fd5ee6d83c84efa2ebfd2e1e692aa7'},
    'source_index':{'path':BASELINE_DIRECTORY+'/source-index.json','sha256':'0b41669916b089f8fddbfda8eecaca2d9a66ea70e95c803b8a8563c7a160494c'},
    'result':{'path':BASELINE_DIRECTORY+'/attempt-01/result.json','sha256':'6cbc6899cea5ea9c86c913e0ef02f194e81a15b1c451c2d0b8c72aa4e8a5949c'},
    'parent':{'path':BASELINE_DIRECTORY+'/attempt-01.supervision/receipt.json','sha256':'36ec7b32ac7d2336065e3e81ae4c9f1b5ce291274b280786a5a8238aba40ab79'},
    'IL_first':{'path':BASELINE_DIRECTORY+'/attempt-01/IL/update-01/update.json','sha256':'98c8f43d2cae2aef82d1a467f2eeb1125776efee6459c3b31885614ff4ab2172'},
    'RL_first':{'path':BASELINE_DIRECTORY+'/attempt-01/RL/update-01/update.json','sha256':'abee51542f6a5f903332a173ae5849fba639b6e902f7cada8aec27f28b42221d'},
}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def semantic(value):return 'sha256:'+hashlib.sha256(canonical(value).encode()).hexdigest()

def require_matched_baseline(protocol,limits):
    """Small saved metadata only. No old checkpoint, observation or image load."""
    records={}
    for key,ref in BASELINE_FILES.items():
        path=ROOT/ref['path']
        if path.is_symlink() or not path.is_file() or path.stat().st_size>2*1024**2 or sha(path)!=ref['sha256']:
            raise ValueError('Completed one-update baseline metadata changed: '+key)
        records[key]=json.loads(path.read_text())
    previous=records['release'];parent=records['parent'];result=records['result']
    if (parent['status']!='complete' or parent['exit_code']!=0 or parent['cleanup_errors']
            or parent['final_owned_pids'] or parent['worker_termination_confirmed'] is not True
            or parent['release_sha256']!=BASELINE_FILES['release']['sha256']
            or parent['result_sha256']!=BASELINE_FILES['result']['sha256']
            or parent['source_index']!=BASELINE_FILES['source_index']
            or previous['source_index']!=BASELINE_FILES['source_index']
            or result['status']!='complete_TRAIN_only_not_heldout_performance'
            or result['optimizer_updates']!={'IL':1,'RL':1}
            or result['SELECT_EVAL_opened'] is not False):
        raise ValueError('Clean completed one-update baseline required')
    expected={**previous['learning_protocol'],'updates_per_method':8}
    if previous['learning_protocol']['updates_per_method']!=1 or canonical(protocol)!=canonical(expected):
        raise ValueError('Only shared update count may change in the learning protocol')
    expected_limits={**previous['cohort_limits'],'max_optimizer_updates':16,
        'max_native_previews':3373440,'max_policy_forwards':2496}
    if canonical(limits)!=canonical(expected_limits):
        raise ValueError('Only update-derived resource counts may change')
    for method in ('IL','RL'):
        row=records[method+'_first']
        if (row['method']!=method or row['completed_updates']!=1 or row['optimizer_updates']!=1
                or row['before_parameter_hash']!=result['initial_parameter_hash']
                or row['after_parameter_hash']!=result['checkpoints'][method]['parameter_hash']
                or parent['checkpoints'][method]!=result['checkpoints'][method]):
            raise ValueError('Baseline update does not bind initial and fixed checkpoint tensors')
    return records

def first_update_control(output,release):
    """Canonical parameter_hash binds every named tensor's dtype/shape/raw bytes.

Protocol/update-budget changes alter context and trace seals. Exclude only those
two metadata fields when comparing the otherwise identical first-step records.
No optimizer hook, weight load or additional policy forward is introduced.
"""
    baseline=require_matched_baseline(release['learning_protocol'],release['cohort_limits'])
    result=json.loads((Path(output)/'result.json').read_text())
    if result['initial_parameter_hash']!=baseline['result']['initial_parameter_hash']:
        raise ValueError('Same-seed initial tensors differ from one-update pilot')
    comparisons={}
    for method in ('IL','RL'):
        path=Path(output)/method/'update-01/update.json'
        if path.is_symlink() or not path.is_file() or path.stat().st_size>2*1024**2:
            raise ValueError('Bounded first-update metadata required')
        actual=json.loads(path.read_text());prior=baseline[method+'_first']
        exclude={'context_hashes','trace_seals'}
        if canonical({k:v for k,v in actual.items() if k not in exclude})!=canonical({k:v for k,v in prior.items() if k not in exclude}):
            raise ValueError('Exact first-update numerical control differs: '+method)
        comparisons[method]={'update_record_sha256':sha(path),
            'baseline_update_record_sha256':BASELINE_FILES[method+'_first']['sha256'],
            'before_parameter_hash':actual['before_parameter_hash'],
            'after_parameter_hash':actual['after_parameter_hash'],'exact_match':True}
    return {'status':'exact_first_update_control_passed','initial_parameter_hash':result['initial_parameter_hash'],
        'methods':comparisons,'excluded_metadata_fields':['context_hashes','trace_seals'],
        'scope':'exact canonical tensor hashes and numerical first-step receipts; no first-step checkpoint file created or reloaded'}

def canonical_configuration():
    """Called only in authorized freeze/owned worker; no source/model construction."""
    from resectionlab.core import thaw_json
    from resectionlab.native_proposals import NominalCavityProposalConfig
    from resectionlab.patient_planning_cohort_spec import sequential_learning_protocol,sequential_limits
    protocol=sequential_learning_protocol(updates=8,max_steps=24,
        search={'max_calls':5640,'beam_width':2,'seconds':300},
        proposal_config=NominalCavityProposalConfig(max_candidates=120,
            intermediate_opening_mm=1.,tool_footprint_opening=True),retention_mode=MODE)
    limits=sequential_limits(protocol,worker_seconds=WORKER_SECONDS,memory_bytes=MEMORY_BYTES,output_bytes=OUTPUT_BYTES)
    return thaw_json(protocol),thaw_json(limits)

def validate_release(release):
    if (release.get('status')!='released_one_attempt' or release.get('output')!=OUTPUT
            or release.get('TRAIN')!=list(TRAIN) or release.get('closed_roles')!=CLOSED
            or release.get('SELECT_EVAL_execution') is not False
            or release.get('parent_seconds')!=PARENT_SECONDS or type(release.get('parent_seconds')) is not int
            or release.get('supervision_bytes')!=SUPERVISION_BYTES or type(release.get('supervision_bytes')) is not int
            or release.get('attempts')!=1 or type(release.get('attempts')) is not int
            or release.get('automatic_retry') is not False):
        raise ValueError('Exact one-attempt four-TRAIN release required')
    limits=release['cohort_limits']
    expected={'max_steps':24,'max_optimizer_updates':16,'max_native_previews':3373440,
        'max_policy_forwards':2496,'worker_seconds':WORKER_SECONDS,'memory_bytes':MEMORY_BYTES,'threads':1,
        'search':{'max_calls':5640,'beam_width':2,'seconds':300},
        'output_bytes':OUTPUT_BYTES,'checkpoint_bytes':8*1024**2}
    if (canonical(limits)!=canonical(expected)
            or any(type(v) is not int for k,v in limits.items() if k!='search')
            or any(type(v) is not int for v in limits['search'].values())):
        raise ValueError('Exact integer canonical pilot limits required')
    admission={k:v for k,v in limits.items() if k not in ('output_bytes','checkpoint_bytes')}
    if canonical(release['limits'])!=canonical(admission):
        raise ValueError('Factory admission and runner limits differ')
    protocol=release['learning_protocol']
    if (release['learning_protocol_hash']!=semantic(protocol)
            or protocol['updates_per_method']!=8 or protocol['rl_episodes']!=4
            or protocol['cohort_execution']['retention_mode']!=MODE
            or protocol['public_target_context_variant']!='full-supplied-public-target-context-v1'):
        raise ValueError('Exact eight-update shared depth-volume/global-context protocol required')
    if release['public_manifest_index']!={'path':PUBLIC_INDEX,'sha256':PUBLIC_SHA}:
        raise ValueError('Original four-TRAIN public index required')
    if release.get('one_update_baseline')!=BASELINE_FILES:
        raise ValueError('Exact completed one-update control bindings required')

def source_guard(release,release_path,release_sha):
    validate_release(release)
    if sha(release_path)!=release_sha:raise ValueError('Release bytes changed')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    if head!=release['expected_head']:raise ValueError('Canonical HEAD changed')
    ref=release['source_index'];path=ROOT/ref['path']
    if sha(path)!=ref['sha256']:raise ValueError('Source index changed')
    index=json.loads(path.read_text())
    if index['head']!=head:raise ValueError('Source index HEAD differs')
    for relative,expected in index['source_files'].items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or (ROOT/path).is_symlink():
            raise ValueError('Local regular source path required')
        if sha(ROOT/path)!=expected:raise ValueError('Runtime source changed: '+relative)
    for relative,expected in index['metadata_files'].items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or (ROOT/path).is_symlink():
            raise ValueError('Local regular public metadata path required')
        if sha(ROOT/path)!=expected:raise ValueError('Public metadata changed: '+relative)
    if sha(ROOT/PUBLIC_INDEX)!=PUBLIC_SHA or sha(ROOT/COHORT)!=COHORT_SHA:
        raise ValueError('Frozen public index or cohort changed')
    require_matched_baseline(release['learning_protocol'],release['cohort_limits'])
    return index

def complete_result(result):
    return (result.get('status')=='complete_TRAIN_only_not_heldout_performance'
        and result.get('optimizer_updates')=={'IL':8,'RL':8}
        and result.get('teacher_statuses')=={s:'complete_replayed' for s in TRAIN}
        and result.get('selection_readiness',{}).get('ready') is True
        and result.get('selection_readiness',{}).get('execution_admitted') is False
        and result.get('SELECT_EVAL_opened') is False
        and set(result.get('checkpoints',{}))=={'IL','RL'}
        and set(result.get('TRAIN_greedy',{}))=={'IL','RL'}
        and all(set(result['TRAIN_greedy'][m])==set(TRAIN)
            and all(result['TRAIN_greedy'][m][s].get('complete') is True for s in TRAIN)
            for m in ('IL','RL')))
