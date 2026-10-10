"""Tiny generated shared-engine diagnostic; never fits/loads a policy or patient."""
import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.path.insert(0,str(ROOT/'src'));START=time.monotonic()
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve()
        if ROOT/'data' in p.parents or p.name.endswith(('.pt','.ckpt','.safetensors','.nii','.nii.gz','.ressectionlab')):raise PermissionError('Generated/source only')
    if event in ('socket.connect','socket.bind'):raise PermissionError('No network')
sys.addaudithook(guard)
import numpy as np
from resectionlab.development_episode import make_development_task,TOOLS,_select
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.core import array_digest,semantic_digest
from resectionlab.observed_search import observed_beam_search
previews=0;original_preview=NativeResectionEngine.preview_stroke
def counted(*a,**kw):
    global previews
    previews+=1
    return original_preview(*a,**kw)
NativeResectionEngine.preview_stroke=counted
# Public, fixed before any transition; software goal, not a tissue diagnosis.
GOAL=(6,6,3)
base=make_development_task();root_source=base.case.source_hash
source_paths=['development_episode.py','native_spatial_task.py','native_resection.py','sequential_spatial_observation.py','shared_episode.py','spatial_policy.py','data_policy.py','observed_search.py','simulation.py']
hashes={p:hashlib.sha256((ROOT/'src/resectionlab'/p).read_bytes()).hexdigest() for p in source_paths}
def choose(task,mode,depth):return _select(task,TOOLS[0 if mode=='aspirate' else 1].tool_id,(6,6,depth))
def branch(seq):
    task=base.clone();rewards=[]
    for mode,depth in seq:
        step=task.step('STOP' if mode=='stop' else choose(task,mode,depth));rewards.append(step.reward)
    return task,rewards
def physical_inventory(task):
    return sorted((r['tool_id'],tuple(r['voxel']),r['feasible'],r['reason']) for r in task.candidate_inventory()['ledger'])
opened=base.clone();opened.step(choose(opened,'aspirate',2));before=opened.observation()
before_inv=physical_inventory(opened)
probed=opened.clone();step=probed.step(choose(probed,'probe',2));after=probed.observation()
assert step.reward<0 and not step.info['removed_indices_native']
assert np.array_equal(opened._engine.remaining_mask,probed._engine.remaining_mask)
assert np.array_equal(opened._engine.removed_mask,probed._engine.removed_mask)
assert np.array_equal(opened._engine.connected_free_mask,probed._engine.connected_free_mask)
assert np.array_equal(before.base.image_channels,after.base.image_channels)
assert before_inv==physical_inventory(probed)
assert not before.observed_probe_contact_grid.any() and after.observed_probe_contact_grid.any()
assert all(base.case.observed_support[tuple(c)] for c in step.info['probe_contact_indices_native'])
assert before.action_ids!=after.action_ids # State-bound IDs change, physical options do not.
full,full_rewards=branch([('aspirate',2),('probe',2),('aspirate',4),('probe',4),('aspirate',6),('stop',0)])
skipped,skipped_rewards=branch([('aspirate',2),('aspirate',4),('aspirate',6),('stop',0)])
assert np.array_equal(full._engine.removed_mask,skipped._engine.removed_mask)
assert sum(skipped_rewards)>sum(full_rewards)
for t in (full,skipped,probed):assert t.independent_geometry_check().feasible
# Independently score a proposed contact objective on existing committed histories.
# No production reward changed: final retained-goal contact minus declared effort.
def contact_score(task):
    history=task.metrics()['history'];active=[r for r in history if r['action_id']!='STOP']
    completion=bool(task._engine.probe_contact_mask[GOAL] and task._engine.remaining_mask[GOAL])
    removed=float(np.count_nonzero(task._engine.removed_mask)*task._config.voxel_volume_mm3)
    distance=sum(r['complete_tool_path_length_mm'] for r in active)
    switches=sum(a['tool_id']!=b['tool_id'] for a,b in zip(active,active[1:]))
    return {'retained_goal_contact':completion,'goal_retained':bool(task._engine.remaining_mask[GOAL]),'removed_mm3':removed,'nonstop_actions':len(active),'roundtrip_motion_mm':distance,'tool_changes':switches,'proposed_contact_return':float(completion)-.2*removed-.03*len(active)-.001*distance-.03*switches}
stop,_=branch([('stop',0)]);openstop,_=branch([('aspirate',2),('stop',0)]);good,_=branch([('aspirate',2),('probe',2),('stop',0)]);deep,_=branch([('aspirate',6),('stop',0)])
rows={name:contact_score(t) for name,t in [('stop',stop),('open_then_stop',openstop),('open_probe_then_stop',good),('deep_aspirate_then_stop',deep)]}
assert rows['open_probe_then_stop']['retained_goal_contact']
assert rows['open_probe_then_stop']['proposed_contact_return']>0
assert not rows['deep_aspirate_then_stop']['goal_retained']
# Existing search result on unchanged removal objective; not search on new objective.
search_start=previews
seq,accounting=observed_beam_search(base,max_calls=24,beam_width=2,seconds=6.,objective_source='unchanged_permitted_nominal_target',transition_mode='lazy_planning')
search=base.clone()
for aid in seq:search.step(aid)
assert search.terminated
result={'schema':'shared-probe-decision-audit-v1','scope':'generated software research diagnostic only; no learning or clinical/anatomical evidence','source_hash':root_source,'source_sha256':hashes,'probe_immediate_reward':step.reward,'probe_new_contact_cells':step.info['probe_contact_indices_native'],'probe_keeps_remaining_cavity_connected_free_and_base_images_equal':True,'probe_keeps_physical_inventory_equal':True,'probe_changes_action_ids_history_current_tool_and_remaining_budget':True,'unchanged_objective_comparison':{'scripted_rewards':full_rewards,'without_probes_rewards':skipped_rewards,'scripted_return':sum(full_rewards),'without_probes_return':sum(skipped_rewards),'same_final_removed_hash':array_digest(full._engine.removed_mask),'return_gain_removing_probes':sum(skipped_rewards)-sum(full_rewards)},'unchanged_objective_search':{'modes':[r.get('interaction_mode') for r in search.metrics()['history']],'return':search.metrics()['total_reward'],'accounting':accounting,'preview_calls_including_replay':previews-search_start},'proposed_public_geometric_contact_task':{'goal_native_index':GOAL,'goal_defined_before_transitions':True,'goal_semantics':'retained source cell must have a committed tangential probe contact; no biological inference','objective':'terminal completion1 minus .2 per removed mm3, .03 per nonstop action, .001 per roundtrip mm, .03 per tool change; contact potential may be gained and lost','comparison_from_current_committed_histories':rows,'objective_was_not_used_for_training_or_search':True},'total_preview_calls':previews,'runtime_seconds':time.monotonic()-START,'torch_imported':'torch' in sys.modules,'model_calls':0,'optimizer_updates':0,'patient_reads':0}
assert result['runtime_seconds']<20 and not result['torch_imported']
(OUT/'evidence.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:result[k] for k in ['probe_immediate_reward','unchanged_objective_comparison','proposed_public_geometric_contact_task','runtime_seconds','total_preview_calls']}))
