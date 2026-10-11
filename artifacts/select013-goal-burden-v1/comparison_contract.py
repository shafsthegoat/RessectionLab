"""Initial-state-only same-tool burden diagnostic; retained public source authority."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PREVIOUS=ROOT/'build/select013-focused-corridor-v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
if sha(PREVIOUS/'comparison_contract.py')!='9f13ee1d277f6ec714c52017c43d5f47ece65e98f1864947e38c38522b3356f3':raise ValueError('Retained source contract changed')
spec=importlib.util.spec_from_file_location('burden_previous_source_contract',PREVIOUS/'comparison_contract.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
PRIOR=previous.PRIOR;prior_contract=previous.prior_contract;read_pinned=previous.read_pinned
OUTPUT='build/select013-goal-burden-v1/attempt-01';ARMS=('initial_state',)
WORKER_SECONDS,PARENT_SECONDS=120,150
MEMORY_BYTES,OUTPUT_BYTES,SUPERVISION_BYTES=2*1024**3,1024**2,8*1024**2
LIMITS={'max_steps_contract':48,'source_visits':1,'rank_helper_calls':1,'native_previews_total':1000,
 'native_previews_scope':'source_construction_only','native_transitions_total':0,'checkpoint_loads':0,
 'model_forwards':0,'updates':0,'threads':1,'helper_max_goals':120,
 'helper_max_bbox_cells_per_query':262144,'helper_max_total_bbox_cells':4000000,'helper_seconds':30}
PINS={**previous.PINS,
 'focused_index':{'path':'build/select013-focused-corridor-v1/source-index.json','sha256':'a7ef9e49f555f7b588296c0cee35874b56e52c48dbe840cffefac66c32d8ffd1'},
 'baseline_goal_witness':{'path':'build/select013-preparation-choice-v1/decision5-witness.json','sha256':'bd655fc084b1a2273110cf2ea89b01e48520efc35007e8d5c97f0f71de6b894a'}}
def historical():
 saved=previous.historical();saved['focused_index']=read_pinned(PINS['focused_index'])
 # Baseline goal metadata is compared only after ranking; never a helper argument.
 if sha(ROOT/PINS['baseline_goal_witness']['path'])!=PINS['baseline_goal_witness']['sha256']:raise ValueError('Baseline comparison witness changed')
 return saved

def complete_result(r):
 return (r.get('status')=='complete_SELECT013_initial_goal_burden'
  and r.get('world_equal') is True and r.get('source_physics_equal') is True
  and r.get('state_unchanged') is True and r.get('no_action_history') is True
  and r.get('held_inputs_opened') is False and r.get('EVAL_opened') is False
  and r.get('baseline_goal_control',{}).get('source_model_initial_state_match') is True
  and r.get('rank_helper_calls')==1 and r.get('native_previews_ranking')==0
  and r.get('checkpoint_loads')==r.get('model_forwards')==r.get('optimizer_attempts')==r.get('gradient_attempts')==r.get('updates')==r.get('native_transition_attempts')==r.get('native_commit_attempts')==0
  and isinstance(r.get('native_previews_total'),int) and 0<=r['native_previews_total']<=LIMITS['native_previews_total']
  and r.get('arms',{}).get('initial_state',{}).get('status')=='complete_initial_diagnostic'
  and set(r.get('arms',{}))==set(ARMS) and r.get('ranking_complete') is True)

def source_guard(release,path,digest):
 if (sha(path)!=digest or release.get('status')!='released_one_attempt' or release.get('output')!=OUTPUT
     or release.get('limits')!=LIMITS or release.get('arms')!=list(ARMS)
     or release.get('worker_seconds')!=WORKER_SECONDS or release.get('parent_seconds')!=PARENT_SECONDS
     or release.get('attempts')!=1 or release.get('automatic_retry') is not False or release.get('prior_pins')!=PINS):raise ValueError('Exact initial diagnostic release required')
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip();index=read_pinned(release['source_index'])
 if head!=release['expected_head'] or index['head']!=head:raise ValueError('HEAD changed')
 for section in ('source_files','metadata_files'):
  for name,d in index[section].items():
   p=Path(name)
   if p.is_absolute() or '..' in p.parts or (ROOT/p).is_symlink() or sha(ROOT/p)!=d:raise ValueError('Bound file changed: '+name)
 historical();return index
