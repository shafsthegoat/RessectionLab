"""Single declared complete-route comparison; metadata-only release validation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OLD_RUN=ROOT/'build/select013-paged-route-h48-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sha(OLD_RUN/'comparison_contract.py')!='22ffd92bb134d1a7548f38a8e833427ed3b9aa0af7d9ff2fef90e922dc673c7a':raise ValueError('Historical h48 contract changed')
spec=importlib.util.spec_from_file_location('corridor_prior_h48_contract',OLD_RUN/'comparison_contract.py')
old_contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_contract)
PRIOR=old_contract.PRIOR
prior_contract=old_contract.prior_contract;read_pinned=old_contract.read_pinned
OUTPUT='build/select013-recursive-corridor-v1/attempt-01'
ARMS=('fresh_STOP','recursive_corridor')
WORKER_SECONDS,PARENT_SECONDS=600,660
MEMORY_BYTES,OUTPUT_BYTES,SUPERVISION_BYTES=3*1024**3,28*1024**2,8*1024**2
LIMITS={'max_steps':48,'search_seconds_per_arm':480,'selector_transition_ceiling_per_arm':48,
 'base_candidates_per_state':120,'additional_pairs_per_state':32,'extra_pairs_per_page':8,
 'diagnostic_goals_per_state':120,'dependency_previews_per_stalled_state':32,'max_live_certificates_per_page':128,'native_previews_per_arm':30000,
 'native_previews_total':30000,'rollout_transitions_per_arm':48,'replay_transitions_per_arm':48,
 'native_transitions_total':146,'model_forwards':0,'checkpoint_loads':0,'updates':0,'threads':1}
PINS={**old_contract.PINS,
 'h48_result':{'path':'build/select013-paged-route-h48-v1/attempt-01/result.json','sha256':'8dcbf51e12fc4a1175f52422bd5e4b745ce6d35300fd589b8b4b23404d7bdcf2'},
 'h48_receipt':{'path':'build/select013-paged-route-h48-v1/attempt-01.supervision/receipt.json','sha256':'f4924629b70c0c6a1a6b3bdf19b7fa7953076021a1788dfac7b5e2c26cde8b44'},
 'h48_release':{'path':'build/select013-paged-route-h48-v1/root-release.json','sha256':'a957077b04785355b3fd1c8354743526e4e70ae6087de249e325386badad8e3e'},
 'h48_index':{'path':'build/select013-paged-route-h48-v1/source-index.json','sha256':'23bf8d7bb299a05f7f24dfa5d372b6aec6ee4f0f47a88e1f007a8109d458526f'}}

COMPLETED=ROOT/'build/select013-complete-corridor-v1'
if sha(COMPLETED/'comparison_contract.py')!='41946a9e83c25550460fcf1e685f2d83f8cdd1266a0cdce01dba0677094709d8':raise ValueError('Completed corridor contract changed')
completed_spec=importlib.util.spec_from_file_location('recursive_completed_corridor_contract',COMPLETED/'comparison_contract.py')
completed_contract=importlib.util.module_from_spec(completed_spec);completed_spec.loader.exec_module(completed_contract)
PINS.update({
 'corridor_result':{'path':'build/select013-complete-corridor-v1/attempt-01/result.json','sha256':'6aecbed77aa0235abaa03147d2cd2769cdab712dba1b1bcbe69449b4a725cd84'},
 'corridor_receipt':{'path':'build/select013-complete-corridor-v1/attempt-01.supervision/receipt.json','sha256':'e69d62cb95c92f59140f71cfcec8b9df3969dda6cacb9213cc31ab73b3876caf'},
 'corridor_release':{'path':'build/select013-complete-corridor-v1/root-release.json','sha256':'447442457d29f1f74a19d1284a04b2f6cf6f1fba00999f9b5ac30f81489e7206'},
 'corridor_index':{'path':'build/select013-complete-corridor-v1/source-index.json','sha256':'82994d955b6d3c4c93132ecfb874d6b8034c072d08fe44c6bd6fda7489edaf73'}})

def historical():
 saved=old_contract.historical();r=read_pinned(PINS['h48_result']);p=read_pinned(PINS['h48_receipt'])
 if (not old_contract.complete_result(r) or p['status']!='complete' or p['exit_code']!=0
     or p['cleanup_errors'] or p['final_owned_pids'] or not p['worker_termination_confirmed']
     or p['result_sha256']!=PINS['h48_result']['sha256']
     or p['release_sha256']!=PINS['h48_release']['sha256']):raise ValueError('Completed historical h48 required')
 saved['h48_index']=read_pinned(PINS['h48_index']);saved['h48_result']=r
 cr=read_pinned(PINS['corridor_result']);cp=read_pinned(PINS['corridor_receipt'])
 if (not completed_contract.complete_result(cr) or cp['status']!='complete' or cp['exit_code']!=0
     or cp['cleanup_errors'] or cp['final_owned_pids'] or not cp['worker_termination_confirmed']
     or cp['result_sha256']!=PINS['corridor_result']['sha256']
     or cp['release_sha256']!=PINS['corridor_release']['sha256']):raise ValueError('Exact complete negative corridor predecessor required')
 if cr['comparison']['positive_complete_return'] is not False or cr['comparison']['target_progress'] is not False:
  raise ValueError('Preserve declared original negative, no retrospective favorable gate')
 saved['corridor_result']=cr;saved['corridor_index']=read_pinned(PINS['corridor_index'])
 return saved

def complete_result(r):
 return (r.get('status')=='complete_SELECT013_recursive_corridor_comparison'
  and r.get('all_action_plans_sealed_before_evaluation') is True and r.get('world_equal') is True
  and r.get('source_physics_equal') is True and r.get('exploration_geometry_feasible') is True and r.get('held_inputs_opened') is False and r.get('EVAL_opened') is False
  and r.get('checkpoint_loads')==r.get('model_forwards')==r.get('optimizer_attempts')==r.get('gradient_attempts')==r.get('updates')==0
  and list(r.get('arms',{}))==list(ARMS) and all(x.get('status')=='complete_replayed' and x.get('independent_accepted') is True
      and x.get('full_history_equal') is True and x.get('selector_accounting',{}).get('complete') is True for x in r['arms'].values()))

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
