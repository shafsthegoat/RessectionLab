"""Single declared complete-route comparison; metadata-only release validation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OLD_RUN=ROOT/'build/select013-paged-route-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sha(OLD_RUN/'comparison_contract.py')!='b8b17c553c8b38d1d451647f7f374f548e22925992da0b62b73feae3c6a70ba8':raise ValueError('Historical h24 contract changed')
spec=importlib.util.spec_from_file_location('h48_prior_h24_contract',OLD_RUN/'comparison_contract.py')
old_contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_contract)
PRIOR=old_contract.PRIOR
prior_contract=old_contract.prior_contract;read_pinned=old_contract.read_pinned
OUTPUT='build/select013-paged-route-h48-v1/attempt-01'
ARMS=old_contract.ARMS
OLD_HORIZON=old_contract.LIMITS['max_steps']
WORKER_SECONDS,PARENT_SECONDS=1020,1080
MEMORY_BYTES,OUTPUT_BYTES,SUPERVISION_BYTES=3*1024**3,96*1024**2,16*1024**2
LIMITS={'max_steps':48,'beam_width':2,'search_seconds_per_arm':450,'selector_transition_ceiling_per_arm':14535,
 'max_expanded_states_per_arm':95,'base_candidates_per_state':120,'additional_pairs_per_state':32,
 'extra_pairs_per_page':8,'max_live_certificates_per_page':128,'native_previews_per_arm':30000,
 'native_previews_total':60000,'rollout_transitions_per_arm':48,'replay_transitions_per_arm':48,
 'native_transitions_total':29262,'model_forwards':0,'checkpoint_loads':0,'updates':0,'threads':1}
PINS={**old_contract.PINS,**{'h24_result': {'path': 'build/select013-paged-route-v1/attempt-01/result.json', 'sha256': '2c3b860f5006c0e37dbfd9af93786b6a87c7a1118a5b537f4ba6e5443fa51924'}, 'h24_receipt': {'path': 'build/select013-paged-route-v1/attempt-01.supervision/receipt.json', 'sha256': '639702db6c09cf73348ee4440ccf09240d5fdbc7dc297dde0d87614b551d2bd7'}, 'h24_release': {'path': 'build/select013-paged-route-v1/root-release.json', 'sha256': '6c79bcd485b9c53f361daee25fc0bebc21523fa55a93db78ac953af41f365b7c'}, 'h24_index': {'path': 'build/select013-paged-route-v1/source-index.json', 'sha256': '246cd50e603c0cdb87db5a2bddea37901a39076e141ab11dc091c2d353ff7175'}, 'original_lazy_beam_h24_plan': {'path': 'build/select013-paged-route-v1/attempt-01/original_lazy_beam/selected-action-plan.json', 'sha256': '8e007ac0a9feab4d9f8a066de5bfa850714f919a53a92168c4def0a444be6aec'}, 'paged_obstruction_lazy_beam_h24_plan': {'path': 'build/select013-paged-route-v1/attempt-01/paged_obstruction_lazy_beam/selected-action-plan.json', 'sha256': '76d2b9e639cee56508950fcc4c6499cae48d7fdc790ab167fa62bf4e09e2af9e'}}}

def historical():
 saved=old_contract.historical();r=read_pinned(PINS['h24_result']);p=read_pinned(PINS['h24_receipt'])
 if (not old_contract.complete_result(r) or p['status']!='complete' or p['exit_code']!=0
     or p['cleanup_errors'] or p['final_owned_pids'] or not p['worker_termination_confirmed']
     or p['result_sha256']!=PINS['h24_result']['sha256']
     or p['release_sha256']!=PINS['h24_release']['sha256']):raise ValueError('Exact completed h24 predecessor required')
 saved['h24_index']=read_pinned(PINS['h24_index']);saved['h24_result']=r
 saved['h24_plans']={arm:read_pinned(PINS[arm+'_h24_plan']) for arm in ARMS}
 return saved

def prefix_control(accounting,old_accounting):
 # Nominal action IDs bind source/cavity, not horizon. Both methods are actor-free.
 # The first23 complete layers precede the old forced terminal boundary. Existing
 # progress availability can be populated by next-layer expansion, as in h24.
 expected=old_accounting['layers'][:OLD_HORIZON-1]
 actual=accounting.get('layers',[])[:len(expected)]
 exact=json.dumps(actual,sort_keys=True)==json.dumps(expected,sort_keys=True)
 return {'exact':exact,'compared_complete_layers':len(expected),'old_horizon':OLD_HORIZON,
         'new_horizon':LIMITS['max_steps'],'excluded_old_terminal_layer':OLD_HORIZON,
         'scope':'complete layer paths,returns,progress,retention,counters; h24 terminal layer intentionally excluded'}

def complete_search(a):
 if a.get('call_cap_reached') is not False or a.get('time_cap_reached') is not False:return False
 layers=a.get('layers',[])
 if len(layers)!=a.get('completed_layers') or any(not x.get('completed') or x.get('depth')!=i+1 for i,x in enumerate(layers)):return False
 if len(layers)==LIMITS['max_steps']:return True
 if len(layers)>LIMITS['max_steps']:return False
 if not layers:return False
 final=layers[-1].get('retained_prefix_diagnostics',{})
 return final.get('selection_status')=='completed_layer' and final.get('retained_nonterminal_prefixes')==[]

def complete_result(r):
 return (r.get('status')=='complete_SELECT013_paged_route_h48_comparison' and r.get('all_action_plans_sealed_before_evaluation') is True
  and r.get('both_h24_prefix_controls_equal') is True and r.get('world_equal') is True
  and r.get('source_physics_equal') is True and r.get('held_inputs_opened') is False and r.get('EVAL_opened') is False
  and r.get('checkpoint_loads')==r.get('model_forwards')==r.get('optimizer_attempts')==r.get('gradient_attempts')==r.get('updates')==0
  and list(r.get('arms',{}))==list(ARMS) and all(x.get('status')=='complete_replayed' and x.get('independent_accepted') is True
      and x.get('full_history_equal') is True and complete_search(x.get('selector_accounting',{})) for x in r['arms'].values())
 )

def source_guard(release,path,digest):
 if (sha(path)!=digest or release.get('status')!='released_one_attempt' or release.get('output')!=OUTPUT
     or release.get('limits')!=LIMITS or release.get('arms')!=list(ARMS)
     or release.get('worker_seconds')!=WORKER_SECONDS or release.get('parent_seconds')!=PARENT_SECONDS
     or release.get('attempts')!=1 or release.get('automatic_retry') is not False or release.get('prior_pins')!=PINS):raise ValueError('Exact one-attempt release required')
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip();index=read_pinned(release['source_index'])
 if head!=release['expected_head'] or index['head']!=head:raise ValueError('HEAD changed')
 for section in ('source_files','metadata_files'):
  for name,d in index[section].items():
   p=Path(name)
   if p.is_absolute() or '..' in p.parts or (ROOT/p).is_symlink() or sha(ROOT/p)!=d:raise ValueError('Bound file changed: '+name)
 historical();return index
