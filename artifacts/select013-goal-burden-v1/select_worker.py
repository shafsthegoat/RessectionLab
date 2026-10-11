"""One current public initial-state burden measurement; no actions or learned work."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
from comparison_contract import (OUTPUT,OUTPUT_BYTES,WORKER_SECONDS,LIMITS,PINS,PRIOR,
 historical,prior_contract,read_pinned,complete_result,source_guard,sha)


def execute(release,release_sha,output,progress):
 from dataclasses import replace
 from contextlib import contextmanager
 import torch
 sys.path.append(str(PRIOR))
 from frontier_compare import source_physics,post_exposure_authority
 from resectionlab.core import semantic_digest,thaw_json
 from resectionlab.native_proposals import NominalCavityProposalConfig,PreparedNominalCavityProposer
 from resectionlab.native_resection import NativeResectionEngine
 from resectionlab.native_spatial_task import NativeSpatialTask
 from resectionlab.focused_paged_task import FocusedPagedObstructionTask
 from resectionlab.spatial_policy import SpatialPolicy
 from resectionlab.patient_planning_admission import make_patient_planning_task
 from resectionlab.patient_planning_cohort_io import OutputBudget
 from resectionlab.patient_planning_cohort_sequential import _bounded_writes
 from resectionlab.public_patient_factory import prepare_public_source
 from initial_goal_burden import rank_initial_goals,Limits,BurdenLimit
 output.mkdir(exist_ok=False);sink=OutputBudget(output,OUTPUT_BYTES)
 started=time.perf_counter();phase='setup';originals=[];phases={};profile={}
 result={'status':'started','subject':'ReMIND-013','role':'SELECT','planned_SELECT_denominator':1,
  'held_cases':['ReMIND-037'],'held_inputs_opened':False,'EVAL_opened':False,
  'checkpoint_loads':0,'model_forwards':0,'updates':0,'optimizer_attempts':0,'gradient_attempts':0,
  'native_transition_attempts':0,'native_commit_attempts':0,'native_previews_total':0,
  'native_previews_ranking':0,'rank_helper_calls':0,'private_reference_reads':0,
  'arms':{'initial_state':{'status':'not_started'}},
  'scope':'initial public goal-burden heuristic only; no accepted route, feasibility, action-count lower bound or clinical conclusion',
  'history_complete_attestation':None,'history_scope':'no actions are permitted; no complete-episode claim'}
 def guard():
  if time.perf_counter()-started>WORKER_SECONDS:raise TimeoutError('Initial diagnostic work bound')
  sink.check()
 def write(name,value):
  path=output/name;path.parent.mkdir(exist_ok=True,parents=True);sink.write(path,value);return sha(path)
 def install(owner,name,replacement):
  old=getattr(owner,name);originals.append((owner,name,old));setattr(owner,name,replacement);return old
 def forbid(owner,name,counter):
  def refused(*args,**kwargs):
   result[counter]+=1;raise RuntimeError('Initial diagnostic forbids '+name)
  install(owner,name,refused)
 @contextmanager
 def scope(name):
  nonlocal phase
  previous=phase;phase=name;tick=time.perf_counter();before=result['native_previews_total']
  try:yield
  finally:
   phases[name]={'wall_seconds':time.perf_counter()-tick,'native_previews':result['native_previews_total']-before};phase=previous
 def preview(self,*args,**kwargs):
  guard();result['native_previews_total']+=1
  if phase!='source_construction':
   result['native_previews_ranking']+=1;raise RuntimeError('Native preview outside source construction')
  if result['native_previews_total']>LIMITS['native_previews_total']:raise RuntimeError('Initial preview bound')
  tick=time.perf_counter();row=profile.setdefault(phase,{})
  try:
   value=native_preview(self,*args,**kwargs)
   label='feasible' if value.feasible else 'rejected:'+value.reason
   row[label]=row.get(label,0)+1;return value
  finally:row['native_preview_seconds']=row.get('native_preview_seconds',0.)+time.perf_counter()-tick
 def measure_provider(name):
  old=getattr(PreparedNominalCavityProposer,name)
  def measured(self,*args,**kwargs):
   guard();row=profile.setdefault(phase,{});key='provider'+name
   row[key+'_calls']=row.get(key+'_calls',0)+1;tick=time.perf_counter()
   try:return old(self,*args,**kwargs)
   finally:row[key+'_inclusive_seconds']=row.get(key+'_inclusive_seconds',0.)+time.perf_counter()-tick
  install(PreparedNominalCavityProposer,name,measured)
 try:
  forbid(torch,'load','checkpoint_loads');forbid(torch.optim.Optimizer,'__init__','optimizer_attempts')
  forbid(torch.autograd,'backward','gradient_attempts');forbid(torch.autograd,'grad','gradient_attempts')
  forbid(SpatialPolicy,'forward','model_forwards')
  forbid(NativeSpatialTask,'_transition','native_transition_attempts')
  forbid(NativeResectionEngine,'commit_preview','native_commit_attempts')
  native_preview=install(NativeResectionEngine,'preview_stroke',preview)
  measure_provider('propose');measure_provider('_tool_footprint_endpoint')
  saved=historical();training=saved['training'];cohort=(ROOT/prior_contract.COHORT).read_bytes()
  active=prior_contract.select_public(prior_contract.read_pinned(saved['release']['public_index']))
  with _bounded_writes(sink):
   with scope('source_construction'):
    progress('source_construction');folder=output/'source';folder.mkdir()
    case,binding,qc,protocol=prepare_public_source(folder,saved['release'],PINS['release']['sha256'],None,progress,
     public_manifest_path=ROOT/active['path'],public_manifest_sha256=active['sha256'],cohort_bytes=cohort,
     learning_protocol_hash=semantic_digest(training),proposal_config=NominalCavityProposalConfig(**training['cohort_execution']['proposal_config']),
     public_target_context_variant=training['public_target_context_variant'],expected_role='SELECT',
     checkpoint_lineage=saved['lineage'],occupancy_condition=prior_contract.OCCUPANCY,
     occupancy_inference_protocol=training,post_exposure_condition=prior_contract.EXPOSURE,partial_domain_inputs=True)
    base,context=make_patient_planning_task(case,cohort_bytes=cohort,source_binding=binding,qc_receipt=qc,protocol=protocol)
    obs=base.observation();world={'source_hash':case.source_hash,'decision_model_hash':base.decision_model_hash,
     'initial_observation_hash':obs.fingerprint,'max_steps':base.max_steps,'proposal_hash':case.proposal_config.fingerprint,
     'occupancy_condition':context.record()['occupancy_condition'],'post_exposure':None if case.post_exposure is None else thaw_json(case.post_exposure.record),
     'source_domains':thaw_json(case._domain_record),'action_ids':list(obs.action_ids),'action_mask':obs.action_mask.tolist()}
    exact=world==saved['world'] and context.record()==saved['context']
    write('original-world-control.json',{'exact':exact,'world':world,'context':context.record()})
    if not exact:raise ValueError('Original source/world/reward/initial observation changed')
    result['world_equal']=True
    changed=replace(case,proposal_config=replace(case.proposal_config,obstruction_opening=True,tool_footprint_opening=True,
     max_candidates=120,obstruction_page_size=8,obstruction_selection_rule='target_ray_coverage_v3'))
    if source_physics(changed)!=source_physics(case):raise ValueError('Diagnostic declaration changed source physics')
    authority=post_exposure_authority(case.post_exposure,changed.post_exposure)
    task=FocusedPagedObstructionTask(changed,max_steps=48,reward=base.reward_spec,cancelled=base._cancelled)
    write('task-binding.json',{'source_physics_equal':True,'post_exposure_authority':authority,
     'source_hash':task.case.source_hash,'decision_model_hash':task.decision_model_hash,
     'native_decision_model_hash':task._config.fingerprint,'initial_state_hash':task._engine.state_hash,
     'proposal_config':task.case.proposal_config.to_record(),'task_contract_horizon':task.max_steps,
     'scope':'same initial focused public world; no page certification or task observation before burden helper'})
    result['source_physics_equal']=True
    del case,binding,qc,protocol,obs,base,context
   initial_state=task._engine.state_hash
   with scope('initial_ranking'):
    progress('initial_ranking');result['rank_helper_calls']+=1
    try:
     report=thaw_json(rank_initial_goals(task,limits=Limits(max_goals=LIMITS['helper_max_goals'],
      max_bbox_cells_per_query=LIMITS['helper_max_bbox_cells_per_query'],max_total_bbox_cells=LIMITS['helper_max_total_bbox_cells'],
      seconds=LIMITS['helper_seconds']),cancelled=lambda:(guard() or False)))
    except BurdenLimit as error:
     write('initial-goal-burden-incomplete.json',thaw_json(error.accounting));raise
    result['ranking_report_sha256']=write('initial-goal-burden.json',report)
    if semantic_digest({k:v for k,v in report.items() if k!='seal'})!=report['seal'] or not report['complete']:
     raise ValueError('Incomplete or changed rank report')
    result['ranking_complete']=True
   # Historical comparison is outside the helper and follows its only invocation.
   with scope('saved_baseline_comparison'):
    witness=read_pinned(PINS['baseline_goal_witness'])['protocols']['narrow']
    baseline_goal=witness['decision']['active_goal']
    matched=(report['source_hash']==witness['source_hash'] and report['task_decision_model_hash']==witness['decision_model_hash']
     and report['native_decision_model_hash']==witness['native_decision_model_hash']
     and report['source_state_hash']==witness['prefix'][0]['source_state_hash'])
    rows=[x for x in report['rows'] if {k:x[k] for k in ('tool_id','entry_mm','tip_mm')}==baseline_goal]
    top=report['ranked_physical_goals'][0] if report['ranked_physical_goals'] else None
    comparison={'source_model_initial_state_match':matched,'historical_baseline_active_goal':baseline_goal,
     'historical_witness':PINS['baseline_goal_witness'],'helper_first_same_tool_goal':report['original_first_same_tool_goal'],
     'first_same_tool_equals_actual_baseline_goal':report['original_first_same_tool_goal']==baseline_goal,
     'baseline_present_count':len(rows),'baseline_row':rows[0] if len(rows)==1 else None,
     'proposed_ranked_goal':top,'proposed_goal_differs_from_actual_baseline':None if top is None else top!=baseline_goal,
     'scope':'comparison only after ranking; prior continuation is not helper input; changed heuristic ordering is not a demonstrated changed complete policy route'}
    write('baseline-goal-control.json',comparison);result['baseline_goal_control']={k:v for k,v in comparison.items() if k!='baseline_row'}
    if not matched or len(rows)!=1:raise ValueError('Exact historical baseline goal/state missing from current same-tool catalog')
    result['state_unchanged']=task._engine.state_hash==initial_state and task._steps==0 and not task.terminated and task._current_tool is None
    result['no_action_history']=not task._engine.history
    if not result['state_unchanged'] or not result['no_action_history']:raise ValueError('Initial measurement changed committed state')
    result['arms']['initial_state']={'status':'complete_initial_diagnostic','goal_count':report['unique_goal_count'],
     'rankable_goal_count':len(report['ranking_raw_orders']),'ranking_raw_orders':report['ranking_raw_orders'],
     'minimum_score_tie_count':report['minimum_score_tie_count'],'proposed_goal_differs_from_actual_baseline':comparison['proposed_goal_differs_from_actual_baseline']}
    result['helper_accounting']={k:report[k] for k in ('base_enumeration_calls','base_enumeration_wall_seconds','helper_capsule_cover_calls',
     'helper_containment_calls','bbox_cells_charged','native_preview_calls','native_transitions','wall_seconds')}
   guard();result['status']='complete_SELECT013_initial_goal_burden'
 except BaseException as error:
  result.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error),'phase':phase});raise
 finally:
  for owner,name,old in reversed(originals):setattr(owner,name,old)
  result['wall_seconds']=time.perf_counter()-started
  try:write('costs.json',{'phases':phases,'provider_and_native_profile':profile,
   'scope':'phase wall includes all source/setup/helper/comparison; provider/native timers are nested, not additive; helper geometry call counts cover only explicit helper calls'})
  finally:sink.write(output/'result.json',result,terminal=True)
 return result


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release',required=True,type=Path);parser.add_argument('--release-sha256',required=True);args=parser.parse_args()
 from resectionlab.legacy_transfer_worker import _require_parent_lease
 _require_parent_lease();release=json.loads(args.release.read_text());source_guard(release,args.release,args.release_sha256)
 output=ROOT/OUTPUT;supervision=output.with_name(output.name+'.supervision')
 if output.exists() or not supervision.is_dir():raise ValueError('Fresh owned output required')
 started=time.perf_counter();summary={'status':'started'}
 def progress(phase,**details):
  temp=supervision/'worker-progress.tmp';temp.write_text(json.dumps({'phase':phase,'seconds':time.perf_counter()-started,**details})+'\n');temp.replace(supervision/'worker-progress.json')
 def deadline(*unused):raise TimeoutError('Initial diagnostic worker deadline')
 old=signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,WORKER_SECONDS)
 try:
  import torch
  torch.set_num_threads(1);torch.set_num_interop_threads(1)
  result=execute(release,args.release_sha256,output,progress)
  if not complete_result(result):raise ValueError('Incomplete initial burden diagnostic')
  source_guard(release,args.release,args.release_sha256)
  summary.update(status='complete_owned_SELECT013_initial_goal_burden',result_sha256=sha(output/'result.json'))
 except BaseException as error:summary.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)});raise
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
  summary.update(wall_seconds=time.perf_counter()-started,canonical_result_sha256=sha(output/'result.json') if (output/'result.json').is_file() else None)
  with (supervision/'worker-final.json').open('x') as stream:json.dump(summary,stream,indent=2);stream.write('\n')
if __name__=='__main__':main()
