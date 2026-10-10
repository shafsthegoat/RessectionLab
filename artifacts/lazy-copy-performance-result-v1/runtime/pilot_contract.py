"""Two fresh implementations of one frozen TRAIN025 task; no model work."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PRIOR=ROOT/'build/paired-obstruction-opening-search-v2'
spec=importlib.util.spec_from_file_location('_paired_obstruction_inputs',PRIOR/'pilot_contract.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
sha=prior.sha;canonical=prior.canonical;semantic=prior.semantic;small=prior.small
without_ids=prior.without_ids;inventory_summary=prior.inventory_summary
PAIR_OUTPUT='build/lazy-copy-train025-pair-v2/attempt-02'
IMPLEMENTATIONS=('baseline','lazy_copy')
IMPLEMENTATION=os.environ.get('RESECTIONLAB_PERFORMANCE_ARM','baseline')
if IMPLEMENTATION not in IMPLEMENTATIONS:raise ValueError('Unknown fixed implementation')
OUTPUT=PAIR_OUTPUT+'/'+IMPLEMENTATION
TRAIN=('ReMIND-025',);CONDITIONS=prior.CONDITIONS;OCCUPANCY=prior.OCCUPANCY
ARMS=({'subject':'ReMIND-025','condition':'obstruction_opening'},)
CLOSED={'other_TRAIN':['ReMIND-008','ReMIND-010','ReMIND-020'],'SELECT':['ReMIND-013','ReMIND-037'],'EVAL':['ReMIND-067']}
WORKER_SECONDS=60;PARENT_SECONDS=90;MEMORY_BYTES=3*1024**3
OUTPUT_BYTES=32*1024**2;SUPERVISION_BYTES=16*1024**2
INPUT_INDEX=prior.INPUT_INDEX;INPUT_SHA=prior.INPUT_SHA
BASELINE='src/resectionlab/native_resection.py'
BASELINE_SHA='2aa191370f2dab441089dcd3d6b925c80d0da450d72fdc464d42fb93d13c8e1f'
CANDIDATE='build/immutable-preview-lazy-copy-v2/stage/resectionlab/native_resection.py'
CANDIDATE_SHA='79363aa6156917514d4aa57e328adcd16babff51518c0c4587e6dcaf94a5723d'
EXECUTION={'arms_per_child':1,'max_steps':24,'candidate_cap':120,'search_seconds_per_arm':45,
 'max_native_previews':10000,'max_search_commits':24,'max_execution_replay_commits':48,
 'max_native_transition_calls':72,'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0,
 'threads':1,'search_method':'observed_greedy_search','post_exposure_condition':None}
CORRECTION_INPUTS={'failed_pair_result': {'path': 'build/lazy-copy-train025-pair-v1/attempt-01/pair-result.json', 'sha256': '836be5c561393b7d1fb53b937245898012ff46b105ffaa72ceb8fbed6f715574'}, 'failed_release': {'path': 'build/lazy-copy-train025-pair-v1/root-release.json', 'sha256': '33231f5aa7f682facf01b9ad0fddcc00ab455d19bec30aa11034146079b25731'}, 'failed_source_index': {'path': 'build/lazy-copy-train025-pair-v1/source-index.json', 'sha256': 'b7e8a7d71fb74ce519d5965eb19b3d2fc6ae7ea6cd0ee4075f35360d08e512e5'}}
EXPECTED={
 'receipt':('build/paired-obstruction-opening-search-v2/attempt-02.supervision/receipt.json','54cef3f9163a1ebfb8d90752d930410e359747b8052051cdae320eb266320a99'),
 'result':('build/paired-obstruction-opening-search-v2/attempt-02/result.json','f9f690056ab5b263f7306f9ec6807fa3dc7629f0290b5077ae9c80d57b777d50'),
}
def inputs():
 records,refs=prior.inputs()
 for label,ref in CORRECTION_INPUTS.items():
  if sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Preserved failed attempt changed')
  refs[label]=ref
 for key,(path,digest) in EXPECTED.items():
  if sha(ROOT/path)!=digest:raise ValueError('Completed route evidence changed: '+key)
  records['expected_'+key]=small(ROOT/path);refs['expected_'+key]={'path':path,'sha256':digest}
 result=records['expected_result'];receipt=records['expected_receipt']
 if receipt['status']!='complete' or receipt['result_sha256']!=refs['expected_result']['sha256'] or receipt['final_owned_pids'] or receipt['cleanup_errors']:raise ValueError('Historical owned completion differs')
 if result['status']!='complete_paired_TRAIN_search' or result['arms'][7]['subject']!='ReMIND-025':raise ValueError('Exact completed route required')
 for name in ('world','initial-inventory','plan','replay','search'):
  path=PRIOR/'attempt-02/arm-07'/(name+'.json')
  records['expected_'+name]=small(path);refs['expected_'+name]={'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
 if semantic(records['expected_plan']['plan'])!=result['arms'][7]['plan_seal']:raise ValueError('Historical plan seal differs')
 for k in ('source_hash','decision_model_hash','initial_observation_hash'):
  if records['expected_world'][k]!=result['arms'][7][k] or records['expected_plan']['plan'][k]!=result['arms'][7][k]:raise ValueError('Historical world join differs')
 if canonical(records['expected_replay']['metrics']['history'])!=canonical(records['expected_plan']['plan']['history']):raise ValueError('Historical replay differs')
 return records,refs

def condition_admission():
 records,_=inputs();old=records['paired_release']['condition_admission'][OCCUPANCY]
 limits=json.loads(canonical(old['limits']))
 if limits['max_policy_forwards']!=0 or limits['max_optimizer_updates']!=0:raise ValueError('Search-only admission required')
 return {'limits':limits,'learning_protocol_hash':semantic({'experiment':'fixed025-lazy-copy-performance-v1',
  'prior_union_protocol_hash':old['learning_protocol_hash'],'execution':EXECUTION}),
  'initialization_performed':False,'actual_model_optimizer_checkpoint_calls':0}

def release_template(head,index_sha):
 return {'version':'fixed025-lazy-copy-pair-release-v2','status':'pending_root_release','expected_head':head,
  'output':PAIR_OUTPUT,'implementations':list(IMPLEMENTATIONS),'TRAIN':list(TRAIN),'closed_roles':CLOSED,
  'attempts':1,'automatic_retry':False,'manual_corrected_attempt_ordinal':2,
  'correction':{'scope':'parent declaration reads execution_limits_per_child; no scientific change','preserved_v1':CORRECTION_INPUTS},'parent_seconds_per_child':PARENT_SECONDS,'worker_seconds_per_child':WORKER_SECONDS,
  'maximum_pair_child_parent_seconds':2*PARENT_SECONDS,'memory_bytes':MEMORY_BYTES,'output_bytes_per_child':OUTPUT_BYTES,
  'supervision_bytes_per_child':SUPERVISION_BYTES,'execution_limits_per_child':EXECUTION,'planned_arms':list(ARMS),
  'condition_admission':{'obstruction_opening':condition_admission()},
  'source_index':{'path':'build/lazy-copy-train025-pair-v2/source-index.json','sha256':index_sha},
  'input_index':{'path':INPUT_INDEX,'sha256':INPUT_SHA},
  'engine_sources':{'baseline':{'path':BASELINE,'sha256':BASELINE_SHA},'lazy_copy':{'path':CANDIDATE,'sha256':CANDIDATE_SHA}},
  'recipe':{'task':'exact completed TRAIN025 S-union-T obstruction-opening h24/cap120',
   'selector':'unchanged observed_greedy_search; no new choices, thresholds or tuning',
   'only_change':'immutable preview working-mask copies deferred until first actual cut',
   'measurement':'same copy-count wrapper in both children; fresh processes baseline first; single pair not stable speedup benchmark',
   'parity':'exact source/model/observation/context, all inventories/scores/actions, native state/history/metrics and independent geometry; only timing excluded'},
  'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0,'private_reference_reads':0,'SELECT_EVAL_execution':False}
def validate_release(release):
 expected=release_template(release.get('expected_head'),release.get('source_index',{}).get('sha256'));expected['status']='released_one_attempt'
 if canonical(expected)!=canonical(release):raise ValueError('Exact one-pair release required')
def source_guard(release,release_path,release_sha):
 validate_release(release)
 if sha(release_path)!=release_sha:raise ValueError('Release changed')
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip();ref=release['source_index']
 if head!=release['expected_head'] or sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('HEAD/source index changed')
 index=small(ROOT/ref['path'])
 if index['head']!=head or index['version']!='fixed025-lazy-copy-pair-runtime-v2':raise ValueError('Wrong runtime index')
 for section in ('source_files','metadata_files'):
  for relative,digest in index[section].items():
   p=Path(relative)
   if p.is_absolute() or '..' in p.parts or (ROOT/p).is_symlink() or sha(ROOT/p)!=digest:raise ValueError('Bound bytes changed: '+relative)
 return index

def complete_result(result):
 return (result.get('status')=='complete_paired_TRAIN_search' and result.get('implementation')==IMPLEMENTATION
  and [{k:r.get(k) for k in ('subject','condition')} for r in result.get('arms',[])]==list(ARMS)
  and all(r.get('status')=='complete' and r.get('source_released') is True for r in result['arms'])
  and all(type(result.get(k)) is int and result[k]==0 for k in ('policy_forwards','optimizer_updates','checkpoint_loads','private_reference_reads'))
  and result.get('SELECT_EVAL_opened') is False and result.get('global_resource_failure') is None)

def invariant_search(value):
 value=json.loads(canonical(value));value['accounting'].pop('planning_seconds');return value

def invariant_replay(value):
 value=json.loads(canonical(value));value['independent_geometry'].pop('evaluation_seconds');return value

def endpoint_control(output,release):
 output=Path(output);result=small(output/'result.json');cost=small(output/'costs.json');records,_=inputs();d=output/'arm-00'
 if not complete_result(result):raise ValueError('Incomplete fixed TRAIN025 endpoint')
 row=result['arms'][0];w=small(d/'world.json');inv=small(d/'initial-inventory.json');plan=small(d/'plan.json');replay=small(d/'replay.json');search=small(d/'search.json')
 oldw=records['expected_world'];oldplan=records['expected_plan']['plan']
 for k in ('source_hash','decision_model_hash','initial_observation_hash','common_public_world','normalization','derived_occupancy','supplied_goal_extent','proposal_config','proposal_rule_hash'):
  if canonical(w[k])!=canonical(oldw[k]) or canonical(row[k])!=canonical(w[k]):raise ValueError('Historical world changed: '+k)
 if (canonical(inv)!=canonical(records['expected_initial-inventory']) or plan['plan']['actions']!=oldplan['actions']
   or canonical(plan['plan']['history'])!=canonical(oldplan['history']) or semantic(plan['plan'])!=plan['plan_seal']
   or row['plan_seal']!=plan['plan_seal'] or plan['plan']['actions']!=search['actions']
   or canonical(replay['metrics']['history'])!=canonical(plan['plan']['history'])
   or canonical(invariant_replay(replay))!=canonical(invariant_replay(records['expected_replay']))
   or canonical(invariant_search(search))!=canonical(invariant_search(records['expected_search']))):raise ValueError('Historical complete route/replay/selection differs')
 if (row['context_hash']!=plan['plan']['context_hash'] or replay['independent_geometry']['accepted'] is not True
   or result['native_counts']!=cost['native_counts'] or result['native_counts']['search']!=search['accounting']['model_transition_calls']
   or result['native_counts']['search']>24 or result['native_counts']['rollout']+result['native_counts']['replay']>48
   or cost['native_budget']['native_preview_entries']>10000
   or cost['temporary_mask_copies']['preview_calls']!=cost['native_budget']['native_preview_entries']):raise ValueError('Completion/accounting differs')
 index=small(d/'search-inventories.json');states={}
 for entry in index:
  p=d/entry['file'];v=small(p)
  if sha(p)!=entry['sha256'] or canonical(inventory_summary(v))!=canonical(entry['summary']):raise ValueError('Inventory record changed')
  states[v['cavity_state_hash']]=v
 for decision in search['accounting']['decisions']:
  legal={r['action_id'] for r in states[decision['source_state_hash']]['ledger'] if r['feasible']}
  if legal!={r['action_id'] for r in decision['scores'] if r['action_id']!='STOP'}:raise ValueError('Not all legal motions scored')
 return {'status':'fixed_TRAIN025_complete_exact_historical_route','implementation':IMPLEMENTATION,
  'plan_seal':plan['plan_seal'],'native_counts':result['native_counts'],'native_previews':cost['native_budget']['native_preview_entries'],
  'historical_world_and_route_equal':True,'zero_model_work':True}
