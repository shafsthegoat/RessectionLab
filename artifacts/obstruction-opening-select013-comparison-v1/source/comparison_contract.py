"""Metadata gates for one fixed public SELECT comparison; imports no model code."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = 'build/obstruction-opening-select013-comparison-v1/attempt-01'
TRAIN = ('ReMIND-008', 'ReMIND-010', 'ReMIND-020', 'ReMIND-025')
ENDPOINTS = ('IL64', 'RL8')
ARMS = ('observed_greedy', 'IL64', 'RL8', 'observed_beam')
OCCUPANCY = 'cerebrum_plus_supplied_tumor_assumption'
COHORT = 'manifests/experiments/remind-component-cohort-v1.json'
COHORT_SHA = '326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05'
PUBLIC_INDEX = 'build/remind-select-public-preparation-v1/public-select-index-v2.json'
PUBLIC_SHA = 'adc7232f0fd05300f90078385bc2576150eccc882712e31bfc97534f68680d5f'
MODE = 'return_plus_opening_depth_volume_v1'
TARGET_CONTEXT = 'full-supplied-public-target-context-v1'
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

def endpoint_refs(name):
    if name not in ENDPOINTS:raise ValueError('Fixed endpoint required')
    method='IL' if name=='IL64' else 'RL'
    directory=Path('build/obstruction-opening-learning-v1')/name
    paths={'release':directory/'root-release.json','result':directory/'attempt-01/result.json',
        'parent':directory/'attempt-01.supervision/receipt.json',
        'worker_final':directory/'attempt-01.supervision/worker-final.json'}
    rows={key:{'path':str(path),'sha256':sha(ROOT/path)} for key,path in paths.items()}
    expected={'IL64':'b30dc1f8867b3c016ad9f88d2ded4265787e0c0e4ffc771c0fa28ef31a6bc518',
        'RL8':'60631a2fb4cebda2d6b29475156e487a8b51b4bd1c5cc1d6ca89dd2511e29897'}
    if rows['result']['sha256']!=expected[name]:raise ValueError('Exact completed matched TRAIN result required')
    result=read_pinned(rows['result'])
    rows.update(status='terminal_bound',contexts={subject:{
        'path':str(directory/'attempt-01'/(subject+'-context.json')),
        'sha256':sha(ROOT/directory/'attempt-01'/(subject+'-context.json'))} for subject in TRAIN},
        checkpoint={'path':str(directory/'attempt-01'/(method+'-final.psckpt')),
                    'sha256':result['checkpoints'][method]['sha256']})
    return rows


def comparison_world(protocol):
    execution = protocol['cohort_execution']
    return {key:execution[key] for key in ('max_steps','proposal_config','proposal_rule_hash',
        'search','retention_mode','task_condition','occupancy_condition')} | {
        'public_target_context_variant':protocol['public_target_context_variant']}

def authenticate_endpoint(name, refs, *, root=None):
    """Join own terminal TRAIN release, result, parent, worker and four contexts.

    Only JSON is opened here. The canonical safe weight loader additionally
    binds exact checkpoint bytes/architecture/parameters to these context pins.
    """
    if name not in ENDPOINTS or refs.get('status')!='terminal_bound':
        raise ValueError('Every predetermined endpoint must have terminal provenance')
    records = {k:read_pinned(refs[k], root=root) for k in
        ('release','result','parent','worker_final')}
    release, result, parent, worker = (records[k] for k in ('release','result','parent','worker_final'))
    method = 'RL' if name=='RL8' else 'IL'
    updates = 64 if name=='IL64' else 8
    balanced = name=='IL64'
    expected_status = 'complete_matched_TRAIN_endpoint'
    if (parent['status']!='complete' or parent['exit_code']!=0 or parent['stop_reason'] is not None
            or parent['cleanup_errors'] or parent['final_owned_pids']
            or parent['worker_termination_confirmed'] is not True
            or parent['release_sha256']!=refs['release']['sha256']
            or parent['result_sha256']!=refs['result']['sha256']
            or parent['worker_final_sha256']!=refs['worker_final']['sha256']
            or parent['source_index']!=release['source_index']
            or worker['result_sha256']!=refs['result']['sha256']
            or worker['canonical_result_sha256']!=refs['result']['sha256']
            or worker['status']!='complete_owned_matched_TRAIN_endpoint'
            or worker['method']!=method or worker['SELECT_EVAL_opened'] is not False
            or parent['endpoint_control_sha256']!=worker['endpoint_control_sha256']
            or result['status']!=expected_status
            or result['SELECT_EVAL_opened'] is not False or result['private_reference_reads']!=0
            or release['TRAIN']!=list(TRAIN) or release['SELECT_EVAL_execution'] is not False
            or release['status']!='released_one_attempt'
            or result['method']!=method or release['method']!=method
            or result['optimizer_updates']!={m:updates if m==method else 0 for m in ('IL','RL')}
            or release['new_four_training'] is not False
            or result['search_calls']!=0 or result['checkpoint_loads']!=1
            or result['teacher_logit_forwards']!=7
            or set(result['TRAIN_greedy'])!=set(TRAIN)
            or not all(row['complete'] is True for row in result['TRAIN_greedy'].values())
            or result['teacher_statuses']!={s:'complete_replayed' for s in TRAIN}
            or result['selection_readiness']['ready'] is not True):
        raise ValueError('Own clean terminal TRAIN chain required: '+name)
    protocol = release['learning_protocol']
    if (release['learning_protocol_hash']!=semantic(protocol) or protocol['updates_per_method']!=updates
            or (protocol['cohort_execution'].get('il_teacher_weighting')=='balanced_STOP_motion_CE_v1')!=balanced
            or protocol['cohort_execution']['patient_order']!=list(TRAIN)
            or protocol['cohort_execution']['heldout_execution'] is not False
            or protocol['cohort_execution'].get('occupancy_condition')!=OCCUPANCY
            or protocol['cohort_execution']['teacher_observations']!=('cache_complete_replayed_TRAIN_teacher_traces_v1'
                if balanced else 'recollect_complete_pinned_plan_each_IL_update')):
        raise ValueError('Declared fixed training objective/endpoint differs: '+name)
    if set(refs['contexts'])!=set(TRAIN): raise ValueError('Four original TRAIN contexts required')
    contexts = {}
    for subject in TRAIN:
        row = read_pinned(refs['contexts'][subject], root=root)
        if (row['subject']!=subject or row['role']!='TRAIN'
                or row['patient_group']!='ReMIND:'+subject[-3:] or row['cohort_sha256']!=COHORT_SHA
                or row['learning_protocol_hash']!=semantic(protocol) or row['max_steps']!=24
                or row['private_reference_in_task'] is not False
                or row['public_target_context_variant']!=TARGET_CONTEXT
                or row.get('occupancy_condition')!=OCCUPANCY
                or row.get('execution_kind')!='fixed_four_TRAIN_union_obstruction_learning_v1'):
            raise ValueError('Own original TRAIN context differs: '+subject)
        contexts[row['patient_group']] = semantic(row)
    checkpoint = result['checkpoints'][method]
    if (checkpoint!=parent['checkpoints'][method] or checkpoint['path']!=method+'-final.psckpt'
            or not 0 < checkpoint['bytes'] <= 8*1024**2
            or refs['checkpoint']!={'path':str(Path(release['output'])/checkpoint['path']),
                                   'sha256':checkpoint['sha256']}):
        raise ValueError('Terminal checkpoint descriptor differs')
    return {'protocol':protocol, 'context_hashes':contexts, 'method':method,
        'updates':updates, 'checkpoint':refs['checkpoint'], 'parameter_hash':checkpoint['parameter_hash'],
        'training_release_sha256':refs['release']['sha256'],
        'initial_parameter_hash':result['initial_parameter_hash']}

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
            or expected['proposal_config'].get('obstruction_opening') is not True
            or expected['proposal_config']['tool_footprint_opening'] is not True):
        raise ValueError('Fixed h24 cap120 depth-volume public context required')
    return endpoints

def complete_result(result):
    return (result.get('status')=='complete_union_obstruction_SELECT013_comparison'
        and result.get('occupancy_condition')==OCCUPANCY
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
            or release.get('occupancy_condition')!=OCCUPANCY
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
