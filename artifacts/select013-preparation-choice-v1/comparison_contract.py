"""One declared common-state preparation-choice intervention; metadata guards only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PREVIOUS=ROOT/'build/select013-focused-corridor-v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
if sha(PREVIOUS/'comparison_contract.py')!='9f13ee1d277f6ec714c52017c43d5f47ece65e98f1864947e38c38522b3356f3':raise ValueError('Focused predecessor contract changed')
spec=importlib.util.spec_from_file_location('choice_previous_focused_contract',PREVIOUS/'comparison_contract.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
PRIOR=previous.PRIOR;prior_contract=previous.prior_contract;read_pinned=previous.read_pinned
OUTPUT='build/select013-preparation-choice-v1/attempt-01'
ARMS=('narrow','wide')
WORKER_SECONDS,PARENT_SECONDS=600,660
MEMORY_BYTES,OUTPUT_BYTES,SUPERVISION_BYTES=3*1024**3,28*1024**2,8*1024**2
LIMITS={**previous.LIMITS,'native_transitions_total':288}
PINS={**previous.PINS,
 'focused_result':{'path':'build/select013-focused-corridor-v1/attempt-01/result.json','sha256':'046e9da6086020475e57b7eb79a3cc793d12283c77c1414ff01a98657cd7295d'},
 'focused_receipt':{'path':'build/select013-focused-corridor-v1/attempt-01.supervision/receipt.json','sha256':'db4f92d736fc57f7b5176988f51e5928b6eac72d2ed0ce22dfa51977db1cfed9'},
 'focused_release':{'path':'build/select013-focused-corridor-v1/root-release.json','sha256':'3bc3bdfb5d5d0c278289aee6a49712674d8cf2fcc2894eace3308606eec1e287'},
 'focused_index':{'path':'build/select013-focused-corridor-v1/source-index.json','sha256':'a7ef9e49f555f7b588296c0cee35874b56e52c48dbe840cffefac66c32d8ffd1'},
 'focused_selection':{'path':'build/select013-focused-corridor-v1/attempt-01/focused_corridor/selection.json','sha256':'a84461e39394a7d94e267e77981ce9a68565bb0d0c253061afd89bd089bce892'},
 'decision5_witness':{'path':'build/select013-preparation-choice-v1/decision5-witness.json','sha256':'bd655fc084b1a2273110cf2ea89b01e48520efc35007e8d5c97f0f71de6b894a'}}

def historical():
 saved=previous.historical();r=read_pinned(PINS['focused_result']);p=read_pinned(PINS['focused_receipt'])
 if (not previous.complete_result(r) or p['status']!='complete' or p['exit_code']!=0
     or p['cleanup_errors'] or p['final_owned_pids'] or not p['worker_termination_confirmed']
     or p['result_sha256']!=PINS['focused_result']['sha256']
     or p['release_sha256']!=PINS['focused_release']['sha256']):raise ValueError('Exact complete focused predecessor required')
 if r['comparison']['positive_complete_return'] is not False or r['comparison']['target_progress'] is not False:
  raise ValueError('Preserve original focused negative outcome')
 if r['focus_validity']['explicitly_cleared_count']!=0:raise ValueError('Original focused validity condition differs')
 saved['focused_index']=read_pinned(PINS['focused_index'])
 saved['focused_summary']={'outcomes':{a:row['outcomes'] for a,row in r['arms'].items()},
  'exploration_return':r['arms']['focused_corridor']['selector_accounting']['explored_total_return']}
 # Raw full-history fixity only. Decode/use for parity only after both new plans seal.
 if sha(ROOT/PINS['focused_selection']['path'])!=PINS['focused_selection']['sha256']:raise ValueError('Historical parity trace changed')
 witness=read_pinned(PINS['decision5_witness'])
 if set(witness['protocols'])!=set(ARMS) or witness['source_evidence']['focused_selection']['sha256']!=PINS['focused_selection']['sha256']:
  raise ValueError('Exact two-branch witness required')
 saved['intervention_protocols']=witness['protocols']
 return saved

def complete_result(r):
 return (r.get('status')=='complete_SELECT013_preparation_choice_comparison'
  and r.get('all_action_plans_sealed_before_evaluation') is True and r.get('world_equal') is True
  and r.get('focused_plan_context_replayed') is True and r.get('source_physics_equal') is True
  and r.get('exploration_geometry_feasible') is True and r.get('narrow_historical_control',{}).get('exact_history') is True
  and r.get('narrow_historical_control',{}).get('exact_action_records') is True
  and r.get('narrow_historical_control',{}).get('exact_selected_records') is True
  and r.get('held_inputs_opened') is False and r.get('EVAL_opened') is False
  and r.get('checkpoint_loads')==r.get('model_forwards')==r.get('optimizer_attempts')==r.get('gradient_attempts')==r.get('updates')==0
  and type(r.get('arms')) is dict and len(r['arms'])==len(ARMS) and set(r['arms'])==set(ARMS)
  and all(x.get('status')=='complete_replayed' and x.get('independent_accepted') is True
      and x.get('full_history_equal') is True and x.get('selector_accounting',{}).get('complete') is True
      and x.get('selector_accounting',{}).get('prefix_verified_count')==4
      and x.get('selector_accounting',{}).get('intervention_applied') is True
      and x.get('exploration_geometry_feasible') is True and x.get('focus_context_replayed') is True
      for x in r['arms'].values()))

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
