"""Single declared complete-route comparison; metadata-only release validation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PRIOR=ROOT/'build/select013-obstruction-frontier-v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sha(PRIOR/'comparison_contract.py')!='15279484cd695d643f251566c075e0e0e541e6dece6940c26014bd28e393f5e4':raise ValueError('Historical helper changed')
spec=importlib.util.spec_from_file_location('paged_route_prior_contract',PRIOR/'comparison_contract.py')
old_contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_contract)
prior_contract=old_contract.prior_contract;read_pinned=old_contract.read_pinned
OUTPUT='build/select013-paged-route-v1/attempt-01'
ARMS=('original_lazy_beam','paged_obstruction_lazy_beam')
WORKER_SECONDS,PARENT_SECONDS=900,960
MEMORY_BYTES,OUTPUT_BYTES,SUPERVISION_BYTES=3*1024**3,96*1024**2,16*1024**2
LIMITS={'max_steps':24,'beam_width':2,'search_seconds_per_arm':300,'selector_transition_ceiling_per_arm':7191,
 'max_expanded_states_per_arm':47,'base_candidates_per_state':120,'additional_pairs_per_state':32,
 'extra_pairs_per_page':8,'max_live_certificates_per_page':128,'native_previews_per_arm':16000,
 'native_previews_total':32000,'rollout_transitions_per_arm':24,'replay_transitions_per_arm':24,
 'native_transitions_total':14478,'model_forwards':0,'checkpoint_loads':0,'updates':0,'threads':1}
PINS={**old_contract.PINS,
 'frontier_result':{'path':'build/select013-obstruction-frontier-v2/attempt-01/result.json','sha256':'6a59fbfcdf32abd8df46e5808f8cd175d5309f1b352ba07ab6d53f2baf050251'},
 'frontier_receipt':{'path':'build/select013-obstruction-frontier-v2/attempt-01.supervision/receipt.json','sha256':'b18a1ea56afccd0651ee87482daaf66405c3cd32adf08e55a872a4e5eafa6d1c'},
 'frontier_release':{'path':'build/select013-obstruction-frontier-v2/root-release.json','sha256':'65d7d22bc4c944d01991352abb1ebfe80b7fbdbfa475e863e5e71281310a8a99'},
 'frontier_index':{'path':'build/select013-obstruction-frontier-v2/source-index.json','sha256':'44e25177b0efd74241a23f4019b0994ff6a1815d15c1d5a50e1d96071703f7e9'}}

def historical():
 saved=old_contract.historical();r=read_pinned(PINS['frontier_result']);p=read_pinned(PINS['frontier_receipt'])
 if (not old_contract.complete_result(r) or p['status']!='complete' or p['exit_code']!=0
     or p['cleanup_errors'] or p['final_owned_pids'] or not p['worker_termination_confirmed']
     or p['result_sha256']!=PINS['frontier_result']['sha256']
     or p['release_sha256']!=PINS['frontier_release']['sha256']):raise ValueError('Exact completed predecessor required')
 saved['frontier_index']=read_pinned(PINS['frontier_index']);return saved

def complete_search(a):
 if a.get('call_cap_reached') is not False or a.get('time_cap_reached') is not False:return False
 layers=a.get('layers',[])
 if len(layers)!=a.get('completed_layers') or any(not x.get('completed') or x.get('depth')!=i+1 for i,x in enumerate(layers)):return False
 if len(layers)==24:return True
 if not layers:return False
 final=layers[-1].get('retained_prefix_diagnostics',{})
 return final.get('selection_status')=='completed_layer' and final.get('retained_nonterminal_prefixes')==[]

def complete_result(r):
 return (r.get('status')=='complete_SELECT013_paged_route_comparison' and r.get('all_action_plans_sealed_before_evaluation') is True
  and r.get('baseline_saved_accounting_equal') is True and r.get('world_equal') is True
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
