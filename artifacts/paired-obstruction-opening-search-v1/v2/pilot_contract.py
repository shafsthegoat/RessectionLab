"""Fixed four-TRAIN old/generalized opening comparison; metadata only."""
import hashlib
import json
from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUTPUT='build/paired-obstruction-opening-search-v2/attempt-02'
TRAIN=('ReMIND-008','ReMIND-010','ReMIND-020','ReMIND-025')
CONDITIONS=('original_13_axes','obstruction_opening')
OCCUPANCY='cerebrum_plus_supplied_tumor_assumption'
ARMS=tuple({'subject':s,'condition':c} for s in TRAIN for c in CONDITIONS)
CLOSED={'SELECT':['ReMIND-013','ReMIND-037'],'EVAL':['ReMIND-067']}
WORKER_SECONDS=600;PARENT_SECONDS=660;MEMORY_BYTES=3*1024**3
OUTPUT_BYTES=64*1024**2;SUPERVISION_BYTES=16*1024**2
INPUT_INDEX='build/paired-obstruction-opening-search-v2/input-index.json'
INPUT_SHA='aa2930fccf3e8a5812403e28725b6c2ec5e99a363de26796ef1cc3a909e17efc'
EXECUTION={'arms':8,'max_steps':24,'candidate_cap':120,'search_seconds_per_arm':45,
    'max_native_previews':100000,'max_search_commits':192,'max_execution_replay_commits':384,
    'max_native_transition_calls':576,'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0,
    'threads':1,'search_method':'observed_greedy_search'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def semantic(value):return 'sha256:'+hashlib.sha256(canonical(value).encode()).hexdigest()
def small(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=4*1024**2:raise ValueError('Bounded regular JSON required: '+str(path))
    return json.loads(path.read_text())
def without_ids(rows):return [{k:v for k,v in r.items() if k not in ('action_id','proposal_id')} for r in rows]
def inventory_summary(inv):
    families={}
    for r in inv['ledger']:
        f=families.setdefault(r['family'],{'slots':0,'emitted':0,'legal':0,'dispositions':{}})
        f['slots']+=1;f['emitted']+=int(r['proposal_reason']=='PROPOSED_UNCERTIFIED');f['legal']+=int(r['feasible'])
        reason=r.get('reason',r['proposal_reason']);f['dispositions'][reason]=f['dispositions'].get(reason,0)+1
    fields=('source_hash','decision_model_hash','cavity_state_hash','provider_version','declared_slots','emitted_count',
        'accepted_count','rejected_count','omitted_count','duplicate_count','unavailable_count','complete','ledger_complete',
        'candidate_cap_omitted_count','obstruction_selection_omitted_count','obstruction_accounting','discovery_complete')
    return {**{k:inv[k] for k in fields if k in inv},'families':families}
def inputs():
    if sha(ROOT/INPUT_INDEX)!=INPUT_SHA:raise ValueError('Input index changed')
    refs=small(ROOT/INPUT_INDEX)['files'];records={}
    for key,ref in refs.items():
        if sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Pinned metadata changed: '+key)
        records[key]=small(ROOT/ref['path'])
    rows=records['public_index']['cases'];prior=records['paired_result']
    if (len(rows)!=4 or {r['patient_id'] for r in rows}!=set(TRAIN) or any(r['role']!='TRAIN' for r in rows)
            or prior['status']!='complete_paired_TRAIN_search' or records['paired_receipt']['status']!='complete'
            or records['paired_receipt']['result_sha256']!=refs['paired_result']['sha256']
            or records['release']['learning_protocol']['cohort_execution']['proposal_config']['max_candidates']!=120
            or records['release']['limits']['max_steps']!=24):raise ValueError('Exact completed fixed TRAIN baseline required')
    for i,s in enumerate(TRAIN):
        world=records['world_'+s];plan=records['plan_'+s]
        if (world['subject']!=s or world['condition']!=OCCUPANCY or prior['arms'][2*i+1]['subject']!=s
                or semantic(plan['plan'])!=plan['plan_seal']
                or any(world[k]!=plan['plan'][k] for k in ('source_hash','decision_model_hash','initial_observation_hash'))):
            raise ValueError('Prior union world/plan join differs')
    return records,refs

def condition_admission(condition):
    records,_=inputs()
    if condition not in CONDITIONS:raise ValueError('Unknown proposal condition')
    prior=records['paired_release']['condition_admission'][OCCUPANCY]
    limits=json.loads(canonical(prior['limits']))
    if limits['max_policy_forwards']!=0 or limits['max_optimizer_updates']!=0:raise ValueError('Search-only union admission required')
    return {'limits':limits,'learning_protocol_hash':semantic({'experiment':'paired-public-obstruction-opening-v1',
        'prior_union_protocol_hash':prior['learning_protocol_hash'],'execution':EXECUTION,'conditions':CONDITIONS}),
        'initialization_performed':False,'actual_model_optimizer_checkpoint_calls':0}

def release_template(head,index_sha):
    records,refs=inputs()
    return {'version':'paired-TRAIN-obstruction-opening-search-release-v2','status':'pending_root_release',
        'expected_head':head,'output':OUTPUT,'TRAIN':list(TRAIN),'closed_roles':CLOSED,'SELECT_EVAL_execution':False,
        'attempts':1,'automatic_retry':False,'manual_corrected_attempt_ordinal':2,
        'correction':{'scope':'local selected_action rename only; no task/config/cap change',
            'prior_failed_result':refs['failed_result'],'prior_parent_receipt':refs['failed_receipt'],
            'prior_source_index':refs['failed_source_index']},'parent_seconds':PARENT_SECONDS,'worker_seconds':WORKER_SECONDS,
        'memory_bytes':MEMORY_BYTES,'output_bytes':OUTPUT_BYTES,'supervision_bytes':SUPERVISION_BYTES,
        'execution_limits':EXECUTION,'planned_arms':list(ARMS),'occupancy_condition':OCCUPANCY,
        'condition_admission':{c:condition_admission(c) for c in CONDITIONS},
        'input_index':{'path':INPUT_INDEX,'sha256':INPUT_SHA},
        'source_index':{'path':'build/paired-obstruction-opening-search-v2/source-index.json','sha256':index_sha},
        'recipe':{'baseline_config':records['release']['learning_protocol']['cohort_execution']['proposal_config'],
            'only_treatment_change':{'obstruction_opening':True},'domain':'unchanged default actor crop; no optional domain extension',
            'selector':'existing observed_greedy_search; STOP wins zero ties; no negative preparation/lookahead',
            'initial_prefix':'all original physical rows/dispositions unchanged except source-bound IDs',
            'tie_rule':'greedy preserves inventory insertion order; original prefix first, additions win only strictly higher reward; STOP wins zero ties',
            'state_inventory_capture':'existing search inventories once per native state; assert zero additional preview entries'},
        'scope':'post-hoc TRAIN action-space comparison; not selection, generalization or clinical validation',
        'completion':'eight fixed complete plans sealed/replayed/independently audited; any failed arm makes whole attempt unresolved',
        'discovery':'bounded first-failure evidence and selected cells; cap/truncation counts preserved; never all paths',
        'unchanged':'S union T, full supplied T/domain, source MRI/frame, fixed access, tools, reward, horizon, cap and normalization',
        'private_reference_reads':0,'model_forwards':0,'optimizer_updates':0,'checkpoint_loads':0}
def validate_release(release):
    expected=release_template(release.get('expected_head'),release.get('source_index',{}).get('sha256'));expected['status']='released_one_attempt'
    if canonical(release)!=canonical(expected):raise ValueError('Exact one-attempt paired release required')
    for value,n in ((release['expected_head'],40),(release['source_index']['sha256'],64)):
        if not isinstance(value,str) or len(value)!=n or any(c not in '0123456789abcdef' for c in value):raise ValueError('Invalid source binding')
def source_guard(release,release_path,release_sha):
    validate_release(release)
    if sha(release_path)!=release_sha:raise ValueError('Release changed')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip();ref=release['source_index']
    if head!=release['expected_head'] or sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('HEAD/source index changed')
    index=small(ROOT/ref['path'])
    if index['head']!=head or index['version']!='paired-TRAIN-obstruction-opening-search-runtime-v2':raise ValueError('Wrong runtime index')
    for section in ('source_files','metadata_files'):
        for relative,digest in index[section].items():
            p=Path(relative)
            if p.is_absolute() or '..' in p.parts or (ROOT/p).is_symlink() or sha(ROOT/p)!=digest:raise ValueError('Bound bytes changed: '+relative)
    return index

def complete_result(result):
    return (result.get('status')=='complete_paired_TRAIN_search'
        and [{k:r.get(k) for k in ('subject','condition')} for r in result.get('arms',[])]==list(ARMS)
        and all(r.get('status')=='complete' and r.get('source_released') is True for r in result['arms'])
        and all(type(result.get(k)) is int and result[k]==0 for k in ('policy_forwards','optimizer_updates','checkpoint_loads','private_reference_reads'))
        and result.get('SELECT_EVAL_opened') is False and result.get('global_resource_failure') is None)

def endpoint_control(output,release):
    root=Path(output);result=small(root/'result.json');costs=small(root/'costs.json');prior,_=inputs()
    if not complete_result(result):raise ValueError('Incomplete fixed eight-arm denominator')
    native=result['native_counts']
    if (native!=costs['native_counts'] or native['search']>192 or native['rollout']+native['replay']>384
            or sum(native.values())>576 or costs['native_budget']['native_preview_entries']>100000
            or costs['prohibited_calls']!={'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0}):raise ValueError('Execution accounting changed')
    rows=[];pairs={}
    for ordinal,row in enumerate(result['arms']):
        d=root/f'arm-{ordinal:02d}';plan=small(d/'plan.json');replay=small(d/'replay.json');search=small(d/'search.json')
        body=plan['plan'];audit=replay['independent_geometry'];metrics=replay['metrics'];world=small(d/'world.json');inv=small(d/'initial-inventory.json')
        for k in ('source_hash','decision_model_hash','initial_observation_hash','context_hash','common_public_world',
                  'normalization','derived_occupancy','supplied_goal_extent','proposal_config','proposal_rule_hash'):
            if canonical(row[k])!=canonical(world[k]):raise ValueError('Saved world binding changed: '+k)
        if row['condition']==CONDITIONS[0]:
            old=prior['world_'+row['subject']]
            if any(row[k]!=old[k] for k in ('source_hash','decision_model_hash','initial_observation_hash')):raise ValueError('Baseline union identity changed')
            if canonical(inv)!=canonical(prior['initial-inventory_'+row['subject']]):raise ValueError('Baseline full initial inventory changed')
            pairs[row['subject']]=(row,inv)
        else:
            base,baseinv=pairs[row['subject']]
            prefix=inv['ledger'][:len(baseinv['ledger'])]
            if (canonical(row['common_public_world'])!=canonical(base['common_public_world'])
                    or canonical(row['normalization'])!=canonical(base['normalization'])
                    or canonical(row['derived_occupancy'])!=canonical(base['derived_occupancy'])
                    or canonical(row['supplied_goal_extent'])!=canonical(base['supplied_goal_extent'])
                    or any(row[k]==base[k] for k in ('source_hash','decision_model_hash','initial_observation_hash'))
                    or canonical(without_ids(prefix))!=canonical(without_ids(baseinv['ledger']))
                    or row['pairing']['unchanged_public_world'] is not True):raise ValueError('Matched proposal-only pairing differs')
        if (semantic(body)!=plan['plan_seal'] or row['plan_seal']!=plan['plan_seal']
                or any(body[k]!=row[k] for k in ('source_hash','decision_model_hash','context_hash','initial_observation_hash'))
                or body['actions']!=search['actions'] or search['accounting']['complete'] is not True
                or canonical(body['history'])!=canonical(metrics['history']) or metrics['terminated'] is not True
                or audit['accepted'] is not True or audit['complete_episode'] is not True
                or audit['committed_history_hash']!=semantic(metrics['history'])
                or audit['source_hash']!=row['source_hash'] or audit['decision_model_hash']!=row['decision_model_hash']
                or canonical(row['outcomes'])!=canonical(audit['outcomes']) or len(body['actions'])>24
                or body['learning_updates']!=0 or body['parameter_hash'] is not None):raise ValueError('Completed seal/replay/outcome differs')
        snapshots=small(d/'search-inventories.json');by_state={}
        for saved in snapshots:
            p=d/saved['file'];value=small(p)
            if sha(p)!=saved['sha256'] or canonical(inventory_summary(value))!=canonical(saved['summary']):raise ValueError('Inventory evidence changed')
            by_state[value['cavity_state_hash']]=value
        for decision in search['accounting']['decisions']:
            v=by_state[decision['source_state_hash']]
            legal={r['action_id'] for r in v['ledger'] if r['feasible']}
            if (legal!={r['action_id'] for r in decision['scores'] if r['action_id']!='STOP'}
                    or decision['legal_nonstop_actions']!=len(legal) or decision['scored_nonstop_actions']!=len(legal)):
                raise ValueError('Scored actions do not match captured existing inventory')
        if row['native_counts']['search']!=search['accounting']['model_transition_calls']:raise ValueError('Search commit accounting differs')
        rows.append({'ordinal':ordinal,'plan_seal':plan['plan_seal'],'plan_sha256':sha(d/'plan.json'),
            'replay_sha256':sha(d/'replay.json'),'search_sha256':sha(d/'search.json'),
            'inventory_index_sha256':sha(d/'search-inventories.json')})
    return {'status':'eight_planned_TRAIN_rows_bound','arms':rows,'native_counts':native,
        'zero_model_work':True,'full_denominator_retained':True,'no_anatomical_validation':True}
