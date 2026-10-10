"""One fixed four-TRAIN attempt. Constants do not grant execution authority."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
OUTPUT='build/cross-patient-planning-v1/fixed-four-train-pilot-v1/attempt-01'
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

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def semantic(value):return 'sha256:'+hashlib.sha256(canonical(value).encode()).hexdigest()

def canonical_configuration():
    """Called only in authorized freeze/owned worker; no source/model construction."""
    from resectionlab.core import thaw_json
    from resectionlab.native_proposals import NominalCavityProposalConfig
    from resectionlab.patient_planning_cohort_spec import sequential_learning_protocol,sequential_limits
    protocol=sequential_learning_protocol(updates=1,max_steps=24,
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
    expected={'max_steps':24,'max_optimizer_updates':2,'max_native_previews':2882880,
        'max_policy_forwards':480,'worker_seconds':WORKER_SECONDS,'memory_bytes':MEMORY_BYTES,'threads':1,
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
            or protocol['updates_per_method']!=1 or protocol['rl_episodes']!=4
            or protocol['cohort_execution']['retention_mode']!=MODE
            or protocol['public_target_context_variant']!='full-supplied-public-target-context-v1'):
        raise ValueError('Exact one-update shared depth-volume/global-context protocol required')
    if release['public_manifest_index']!={'path':PUBLIC_INDEX,'sha256':PUBLIC_SHA}:
        raise ValueError('Original four-TRAIN public index required')

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
    return index

def complete_result(result):
    return (result.get('status')=='complete_TRAIN_only_not_heldout_performance'
        and result.get('optimizer_updates')=={'IL':1,'RL':1}
        and result.get('teacher_statuses')=={s:'complete_replayed' for s in TRAIN}
        and result.get('selection_readiness',{}).get('ready') is True
        and result.get('selection_readiness',{}).get('execution_admitted') is False
        and result.get('SELECT_EVAL_opened') is False
        and set(result.get('checkpoints',{}))=={'IL','RL'}
        and set(result.get('TRAIN_greedy',{}))=={'IL','RL'}
        and all(set(result['TRAIN_greedy'][m])==set(TRAIN)
            and all(result['TRAIN_greedy'][m][s].get('complete') is True for s in TRAIN)
            for m in ('IL','RL')))
