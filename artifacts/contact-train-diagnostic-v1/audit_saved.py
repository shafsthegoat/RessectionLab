"""Independent standard-library arithmetic/identity audit; no runtime imports."""
from pathlib import Path
import collections,hashlib,json,math,statistics,time
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
BASE=ROOT/'build/goal-conditioned-policy-v1';RUN=BASE/'train-diagnostic-run-v1';SUP=BASE/'train-diagnostic-run-v1.supervision'
files={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):files[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text())
def digest(value):return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def near(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
started=time.monotonic();result=read(RUN/'result.json');receipt=read(SUP/'receipt.json');decl=read(SUP/'declaration.json')
assert sha(RUN/'result.json')=='c6a36955965eec1902402b254e26b4cdb23dedf7c0fb2a9cd5016dfa55f1eec7'==receipt['result_sha256']
assert receipt['status']=='complete' and receipt['exit_code']==0 and not receipt['cleanup_errors'] and not receipt['final_owned_pids'] and receipt['worker_termination_confirmed']
assert receipt['elapsed_seconds']<30 and receipt['sampled_peak_rss_bytes']<1073741824
assert decl['source_index_sha256']=='46a262245eb04e7b9ec3457cf95d0a0f136392195246a18a17fd69c9e4d88411'
source_index=read(BASE/'train-diagnostic-source-index.json');assert sha(BASE/'train-diagnostic-source-index.json')==decl['source_index_sha256']
assert source_index['source_files']==decl['source_files']
for path,h in decl['source_files'].items():assert sha(ROOT/path)==h
inputs=read(BASE/'train-diagnostic-input-index.json');assert sha(BASE/'train-diagnostic-input-index.json')==decl['input_index_sha256']
prior=read(ROOT/'build/contact-learning-train-diagnosis-v1/train-audit.json')
for method in ('IL','RL'):
 assert inputs['checkpoints'][method]['parameter_hash']==prior['methods'][method]['final_parameter_hash']
assert result['status']=='complete_fixed_TRAIN_readout'
for key,n in {'checkpoint_loads':3,'forward_calls':120,'native_steps':40,'geometry_previews':760,'optimizer_updates':0,'search_calls':0,'sampled_actions':0,'new_teacher_calls':0,'SELECT_or_MEASUREMENT_reads':0,'patient_reads':0}.items():assert result[key]==n
assert sum(row.get('actor_forward_calls',0) for row in result['costs'].values())==120
assert sum(row.get('actor_forward_returned_calls',0) for row in result['costs'].values())==120
assert sum(row.get('native_transition_calls',0) for row in result['costs'].values())==40
assert sum(row.get('native_nonstop_commit_calls',0) for row in result['costs'].values())==32
assert sum(row.get('geometry_preview_calls',0) for row in result['costs'].values())==760
assert sum(row.get('task_clone_calls',0) for row in result['costs'].values())==24
assert not any(v for row in result['costs'].values() for k,v in row.items() if k.endswith('_failed_calls'))
obs={};states=[]
def input_json(name):
 row=inputs['data'][name];path=ROOT/row['path'];assert sha(path)==row['sha256'];return read(path)
for i in range(1,33):
 update=input_json(f'IL-update-{i:02d}.json');assert update['method']=='IL' and update['update']==i
 if i==1:assert update['update_receipt']['initial_parameter_hash']==inputs['checkpoints']['INITIAL']['parameter_hash']
 for s in update['samples']:
  key=(s['binding_hash'],s['action_id']);assert obs.setdefault(key,s['observation_hash'])==s['observation_hash']
for i in range(24):
 teacher=input_json(f'teacher-{i:02d}.json');assert teacher['role']=='TRAIN'
 binding=digest(teacher['binding']);package=teacher['strategy'];strategy=package['strategy']
 assert digest(strategy)==package['strategySeal']
 reconstructed=read(RUN/f'reconstruction-{i:02d}.json')
 assert reconstructed==result['completed_reconstructions'][i]
 assert reconstructed=={'teacher_index':i,'binding_hash':binding,'strategy_seal':package['strategySeal'],'history_hash':digest(strategy['history']),'geometry_audit_hash':digest(package['geometryAudit']),'states':len(strategy['actions'])}
 for step,action in enumerate(strategy['actions']):
  states.append({'teacher_index':i,'layout_id':teacher['layout_id'],'goal_id':teacher['goal_id'],'step':step,'binding_hash':binding,'teacher_action':action,'observation_hash':obs[(binding,action)]})
assert len(states)==40 and len(obs)==40
methods={};common={};arith_checks=0
for method in ('INITIAL','IL','RL'):
 rows=[];ck=inputs['checkpoints'][method]
 for i,state in enumerate(states):
  row=read(RUN/f'{method}-state-{i:02d}.json');assert all(row[k]==v for k,v in state.items())
  assert row['checkpoint']==method and row['checkpoint_file_sha256']==ck['sha256'] and row['parameter_hash']==ck['parameter_hash']
  identity={k:row[k] for k in ('action_ids','action_modes','legal_mask','teacher_action_index')}
  assert common.setdefault(i,identity)==identity
  ids=row['action_ids'];modes=row['action_modes'];mask=row['legal_mask'];z=row['logits'];legal=[j for j,m in enumerate(mask) if m]
  assert len(ids)==len(modes)==len(mask)==len(z) and ids[0]=='STOP' and modes[0]=='stop' and mask[0]
  assert len(set(ids))==len(ids) and all(type(m) is bool for m in mask)
  assert all((isinstance(v,(int,float)) and math.isfinite(v)) if mask[j] else v is None for j,v in enumerate(z))
  chosen=ids.index(state['teacher_action']);assert row['teacher_action_index']==chosen and mask[chosen]
  top=max(z[j] for j in legal);denom=math.fsum(math.exp(z[j]-top) for j in legal)
  probs=[math.exp(z[j]-top)/denom if mask[j] else 0. for j in range(len(z))]
  greedy=max(legal,key=lambda j:z[j]);movement=[j for j in legal if modes[j]!='stop']
  ce=math.log(denom)+top-z[chosen];entropy=-math.fsum(p*math.log(p) for p in probs if p)
  near(row['cross_entropy'],ce);near(row['entropy'],entropy);near(row['teacher_probability'],probs[chosen]);near(row['STOP_probability'],probs[0])
  margin=None if not movement else z[0]-max(z[j] for j in movement)
  if margin is None:assert row['STOP_minus_best_movement_margin'] is None
  else:near(row['STOP_minus_best_movement_margin'],margin)
  assert row['greedy_action']==ids[greedy] and row['greedy_mode']==modes[greedy] and row['correct']==(greedy==chosen)
  best_movement=max(movement,key=lambda j:z[j]) if movement else None
  rows.append({**row,'recomputed_cross_entropy':ce,'movement_only_teacher_correct':chosen==best_movement,
    'movement_score_range':max(z[j] for j in movement)-min(z[j] for j in movement) if movement else 0.,
    'teacher_mode':modes[chosen]})
  arith_checks+=1
 groups={'all40':rows,'root24':[r for r in rows if r['step']==0],'second16':[r for r in rows if r['step']==1]}
 for group,subset in groups.items():
  saved=result['readouts'][method][group]
  actual={'count':len(subset),'mean_cross_entropy':statistics.mean(r['cross_entropy'] for r in subset),'accuracy':statistics.mean(r['correct'] for r in subset),'mean_teacher_probability':statistics.mean(r['teacher_probability'] for r in subset),'mean_STOP_probability':statistics.mean(r['STOP_probability'] for r in subset),'greedy_STOP_count':sum(r['greedy_mode']=='stop' for r in subset)}
  for k,v in actual.items():near(saved[k],v)
 additional={}
 for group,subset in [('teacher_STOP8',[r for r in rows if r['teacher_mode']=='stop']),('teacher_movement32',[r for r in rows if r['teacher_mode']!='stop']),('root_aspiration16',[r for r in rows if r['teacher_mode']=='aspirate']),('second_probe16',[r for r in rows if r['teacher_mode']=='probe'])]:
  additional[group]={'count':len(subset),'mean_CE':statistics.mean(r['cross_entropy'] for r in subset),'mean_teacher_probability':statistics.mean(r['teacher_probability'] for r in subset),'mean_STOP_probability':statistics.mean(r['STOP_probability'] for r in subset),'movement_only_teacher_correct_count':sum(r['movement_only_teacher_correct'] for r in subset),'STOP_margin_min':min(r['STOP_minus_best_movement_margin'] for r in subset),'STOP_margin_max':max(r['STOP_minus_best_movement_margin'] for r in subset),'movement_score_range_mean':statistics.mean(r['movement_score_range'] for r in subset)}
 methods[method]={'verified_summary':result['readouts'][method],'additional_TRAIN_arithmetic':additional}
expected={'result.json',*(f'reconstruction-{i:02d}.json' for i in range(24)),*(f'{m}-state-{i:02d}.json' for m in ('INITIAL','IL','RL') for i in range(40))}
assert {p.name for p in RUN.iterdir()}==expected
inventory=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(RUN.iterdir())]
audit={'status':'PASS','scope':'independent_saved_TRAIN_only_standard_library_arithmetic_no_models_tasks_checkpoint_reads_search_or_heldout_access','verified_readouts':arith_checks,'verified_reconstructions':24,'native_steps':40,'native_nonstop_commits':32,'geometry_previews':760,'checkpoint_loads_reported':3,'forwards_reported_and_metered':120,'parent_elapsed_seconds':receipt['elapsed_seconds'],'parent_sampled_peak_rss_bytes':receipt['sampled_peak_rss_bytes'],'methods':methods,'source_index_sha256':decl['source_index_sha256'],'input_index_sha256':decl['input_index_sha256'],'result_sha256':receipt['result_sha256'],'receipt_sha256':sha(SUP/'receipt.json'),'source_files_verified':decl['source_files'],'read_files':files,'output_inventory':inventory,'output_files':len(inventory),'output_bytes':sum(v['bytes'] for v in inventory),'review_elapsed_seconds':time.monotonic()-started,'limits':'Checkpoint byte validity and native reconstruction are corroborated by reviewed worker receipts and recorded hashes; this saved-only review does not rerun either. Conditional movement rank is score analysis only, not a new executed selector.'}
(OUT/'audit.json').write_text(json.dumps(audit,sort_keys=True,indent=2)+'\n')
print(json.dumps({'status':'PASS','readouts':arith_checks,'outputs':len(inventory),'bytes':audit['output_bytes'],'elapsed_seconds':audit['review_elapsed_seconds'],'additional':{m:v['additional_TRAIN_arithmetic'] for m,v in methods.items()}},indent=2))
