"""One post-hoc TRAIN025 axis diagnostic, with explicit changed proposal identity."""
import hashlib
import json
from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUTPUT='build/union025-missing-axis-v2/attempt-02';SUBJECT='ReMIND-025';TRAIN=(SUBJECT,)
CONDITION='cerebrum_plus_supplied_tumor_assumption'
CLOSED={'SELECT':['ReMIND-013','ReMIND-037'],'EVAL':['ReMIND-067']}
WORKER_SECONDS=60;PARENT_SECONDS=90;MEMORY_BYTES=3*1024**3
OUTPUT_BYTES=32*1024**2;SUPERVISION_BYTES=16*1024**2
INPUT_INDEX='build/union025-missing-axis-v2/input-index.json';INPUT_SHA='7398be782bcf694958386c264a1ed9662d3d8247076bae38feec754817220653'
EXTRA_AXIS=(-1,4);CLEARANCE_VOXEL=(121,44,72);CLEARANCE_TOOL='generic_suction'
EXECUTION={'max_complete_histories':3,'task_max_steps':24,'candidate_cap':120,
    'max_native_previews':1000,'max_native_transition_calls':12,'max_rollout_calls':6,'max_replay_calls':6,
    'max_diagnostic_previews':2,'obstruction_cell_limit':4096,'policy_forwards':0,'optimizer_updates':0,
    'checkpoint_loads':0,'search_calls':0,'threads':1}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def semantic(value):return 'sha256:'+hashlib.sha256(canonical(value).encode()).hexdigest()
def small(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=4*1024**2:raise ValueError('Bounded regular JSON required')
    return json.loads(path.read_text())
def physical(row):return {k:row[k] for k in ('tool_id','entry_mm','tip_mm')}
def same_ray(row,ray):return canonical(physical(row))==canonical(physical(ray))
def without_ids(rows):return [{k:v for k,v in r.items() if k not in ('action_id','proposal_id')} for r in rows]

def inputs():
    if sha(ROOT/INPUT_INDEX)!=INPUT_SHA:raise ValueError('Input index changed')
    refs=small(ROOT/INPUT_INDEX)['files'];records={}
    for key,ref in refs.items():
        if sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Pinned metadata changed: '+key)
        records[key]=small(ROOT/ref['path'])
    wrapper=records['raw_plan'];assert semantic(wrapper['plan'])==wrapper['plan_seal']
    ray=physical(next(h for h in wrapper['plan']['history'] if h['action_id']!='STOP'))
    if (records['witness_result']['status']!='complete_fixed_ray_witness'
            or records['witness_receipt']['status']!='complete'
            or records['witness_receipt']['result_sha256']!=refs['witness_result']['sha256']
            or records['diagnosis']['blocker']['native_index']!=list(CLEARANCE_VOXEL)
            or records['public_manifest']['patient_id']!=SUBJECT or records['public_manifest']['role']!='TRAIN'):
        raise ValueError('Completed fixed TRAIN025 blocker evidence required')
    return records,refs,ray

def release_template(head,index_sha):
    records,refs,ray=inputs();prior=records['paired_release']['condition_admission'][CONDITION]
    protocol=semantic({'diagnostic':'posthoc-union025-one-extra-axis-v1','prior':prior['learning_protocol_hash'],
        'extra_axis':EXTRA_AXIS,'clearance_voxel':CLEARANCE_VOXEL,'execution':EXECUTION})
    return {'version':'union025-missing-axis-release-v2','status':'pending_root_release','expected_head':head,
        'output':OUTPUT,'TRAIN':list(TRAIN),'closed_roles':CLOSED,'SELECT_EVAL_execution':False,
        'attempts':1,'automatic_retry':False,'manual_corrected_attempt_ordinal':2,
        'correction':{'scope':'exclusive diagnostics write lifetime only','prior_result':refs['failed_result'],
            'prior_parent_receipt':refs['failed_receipt'],'prior_source_index':refs['failed_source_index'],
            'scientific_task_caps_status_literals_unchanged':True},'parent_seconds':PARENT_SECONDS,'worker_seconds':WORKER_SECONDS,
        'memory_bytes':MEMORY_BYTES,'output_bytes':OUTPUT_BYTES,'supervision_bytes':SUPERVISION_BYTES,
        'execution_limits':EXECUTION,'condition':CONDITION,
        'condition_admission':{**prior,'learning_protocol_hash':protocol},
        'proposal_change':{'append_offsets_source_voxels':[list(EXTRA_AXIS)],'original13_unchanged':True,
            'cap':120,'horizon':24,'all_other_config_fields_unchanged':True},
        'clearance_ray':{'tool_id':CLEARANCE_TOOL,'voxel':list(CLEARANCE_VOXEL),
            'formula':'tip=orthogonal_native_affine[:3,:3]@voxel+affine[:3,3]; entry=tip-dot(tip-access.center,access.normal)*access.normal',
            'frame_hash':records['union_world']['common_public_world']['native_frame_hash']},
        'target_ray':ray,'input_index':{'path':INPUT_INDEX,'sha256':INPUT_SHA},
        'source_index':{'path':'build/union025-missing-axis-v2/source-index.json','sha256':index_sha},
        'scope':'post-hoc TRAIN task/action-space diagnosis, not model selection or transfer evaluation',
        'completion':'STOP control plus prep+STOP if executable and prep+target+STOP if executable; every admitted history sealed and independently replayed',
        'selection':'maximum actual complete reward, STOP first on equality',
        'feasibility':'direct previews never authorize inventory injection; exact emitted legal geometry required',
        'unchanged':'public arrays/domain/access/tools/reward/material assumption; native collision and whole-cell removal',
        'private_reference_reads':0,'model_forwards':0,'optimizer_updates':0,'checkpoint_loads':0}

def validate_release(release):
    expected=release_template(release.get('expected_head'),release.get('source_index',{}).get('sha256'));expected['status']='released_one_attempt'
    if canonical(release)!=canonical(expected):raise ValueError('Exact single-axis release required')
    for value,n in ((release['expected_head'],40),(release['source_index']['sha256'],64)):
        if not isinstance(value,str) or len(value)!=n or any(c not in '0123456789abcdef' for c in value):raise ValueError('Invalid source binding')
def source_guard(release,release_path,release_sha):
    validate_release(release)
    if sha(release_path)!=release_sha:raise ValueError('Release changed')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip();ref=release['source_index']
    if head!=release['expected_head'] or sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('HEAD/source index changed')
    index=small(ROOT/ref['path'])
    if index['head']!=head or index['version']!='union025-missing-axis-runtime-v2':raise ValueError('Wrong runtime index')
    for section in ('source_files','metadata_files'):
        for relative,digest in index[section].items():
            p=Path(relative)
            if p.is_absolute() or '..' in p.parts or (ROOT/p).is_symlink() or sha(ROOT/p)!=digest:raise ValueError('Bound bytes changed: '+relative)
    return index
def complete_result(result):
    return (result.get('status')=='complete_missing_axis_diagnostic' and result.get('source_released') is True
        and result.get('SELECT_EVAL_opened') is False and 1<=len(result.get('histories',[]))<=3
        and all(r.get('status')=='complete' for r in result['histories'])
        and all(type(result.get(k)) is int and result[k]==0 for k in
            ('policy_forwards','optimizer_updates','checkpoint_loads','search_calls','private_reference_reads')))
def endpoint_control(output,release):
    root=Path(output);r=small(root/'result.json');cost=small(root/'costs.json');world=small(root/'world.json')
    records,refs,target=inputs()
    if not complete_result(r):raise ValueError('Incomplete missing-axis diagnostic')
    if any(world[k]==records['union_world'][k] for k in ('source_hash','decision_model_hash','initial_observation_hash')):
        raise ValueError('New axis must change source/model/observation identity')
    inv=small(root/'initial-inventory.json');old=records['union_inventory']
    oldrows=[row for row in inv['ledger'] if row['column_index']<13]
    if canonical(without_ids(oldrows))!=canonical(without_ids(old['ledger'])):raise ValueError('Original physical rows/dispositions changed')
    previews=small(root/'diagnostics.json');first=previews['clearance'];second=previews.get('target')
    expected_histories=1+int(first['action_status']=='executable')+int(second is not None and second['action_status']=='executable')
    if len(r['histories'])!=expected_histories:raise ValueError('Missing complete conditional alternative')
    pins=[]
    for i,row in enumerate(r['histories']):
        dest=root/f'history-{i:02d}';wrapper=small(dest/'plan.json');p=wrapper['plan'];re=small(dest/'replay.json');audit=re['independent_geometry']
        if (semantic(p)!=wrapper['plan_seal'] or row['plan_seal']!=wrapper['plan_seal']
                or p['actions']!=row['actions'] or p['actions'][-1]!='STOP' or len(p['actions'])!=i+1
                or canonical(p['history'])!=canonical(re['metrics']['history']) or re['metrics']['terminated'] is not True
                or audit['accepted'] is not True or audit['complete_episode'] is not True
                or audit['committed_history_hash']!=semantic(p['history']) or p['source_hash']!=world['source_hash']
                or p['decision_model_hash']!=world['decision_model_hash'] or p['context_hash']!=world['context_hash']
                or canonical(row['outcomes'])!=canonical(audit['outcomes'])):raise ValueError('History/replay/outcome join differs')
        if i>0 and not same_ray(p['history'][0],first):raise ValueError('Executed clearance ray differs')
        if i==2 and not same_ray(p['history'][1],target):raise ValueError('Executed target ray differs')
        pins.append({'ordinal':i,'plan_sha256':sha(dest/'plan.json'),'replay_sha256':sha(dest/'replay.json'),'seal':wrapper['plan_seal']})
    chosen=max(range(len(r['histories'])),key=lambda i:r['histories'][i]['outcomes']['total_reward'])
    n=sum(len(h['actions']) for h in r['histories'])
    if (r['selected_ordinal']!=chosen or r['selected_plan_seal']!=r['histories'][chosen]['plan_seal']
            or r['histories'][0]['outcomes']['total_reward']!=0 or cost['native_counts']!={'rollout':n,'replay':n}
            or n>6 or cost['diagnostic_previews']!=1+int(second is not None)
            or cost['native_budget']['native_preview_entries']>1000
            or cost['prohibited_calls']!={'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0,'search_calls':0}):
        raise ValueError('Selection/accounting differs')
    return {'status':'single_changed_axis_bound','histories':pins,'native_counts':cost['native_counts'],
        'diagnostic_previews':cost['diagnostic_previews'],'selected_ordinal':chosen,'zero_model_work':True}
