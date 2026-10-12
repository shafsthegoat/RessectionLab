"""Exact fixed-prefix/certificate joins; no planner, learned score or reward change."""
from dataclasses import asdict, replace
from pathlib import Path
import importlib.util
from resectionlab.core import semantic_digest, thaw_json
from resectionlab.focused_paged_task import FocusedPagedObstructionTask

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
PHYSICS_PATH='build/select013-obstruction-frontier-v2/frontier_compare.py'
spec=importlib.util.spec_from_file_location('_frontier_physics',ROOT/PHYSICS_PATH)
physics=importlib.util.module_from_spec(spec);spec.loader.exec_module(physics)


def exact(a,b,label):
    if semantic_digest(a)!=semantic_digest(b):raise ValueError('Exact join differs: '+label)


def compact_history(history):
    fields=('action_id','reward','target_removed_mm3','normal_removed_mm3','tool_id','entry_mm','tip_mm',
            'physical_start_mm','complete_tool_path_length_mm','interaction_mode','outcome_scope')
    return [{**{k:r[k] for k in fields if k in r},'history_row_hash':semantic_digest(r),
             'microsteps':len(r.get('microsteps',())),
             'removed_indices_hash':semantic_digest(r.get('removed_indices_native',[])),
             'contact_indices_hash':semantic_digest(r.get('contact_indices_native',[]))} for r in history]


def physical(row):
    # These five fields are the declared new catalog/model and nominal-vs-evaluator scope.
    return {k:v for k,v in row.items() if k not in physics.BINDING_KEYS|{'outcome_scope'}}


def ray_key(ray):return (ray.family,ray.tool_id,tuple(ray.entry_mm),tuple(ray.tip_mm),tuple(ray.voxel))


def enabled_task(base):
    before=base.case.proposal_config
    if before.obstruction_opening or before.max_candidates!=120 or base.max_steps!=24:
        raise ValueError('Original capped24 base catalog required')
    config=replace(before,obstruction_opening=True,tool_footprint_opening=True,
        obstruction_page_size=8,obstruction_selection_rule='target_ray_coverage_v3')
    case=replace(base.case,proposal_config=config)
    authority=physics.post_exposure_authority(base.case.post_exposure,case.post_exposure)
    exact(physics.source_physics(base.case),physics.source_physics(case),'all immutable physical source fields')
    result=FocusedPagedObstructionTask(case,max_steps=24,reward=base.reward_spec,cancelled=base._cancelled)
    # Native geometry is identical; only task/proposal/query bindings may differ.
    native_old=physics.plain(base._config);native_new=physics.plain(result._config)
    exact({k:v for k,v in native_old.items() if k!='source_hash'},
          {k:v for k,v in native_new.items() if k!='source_hash'},'all native physical configuration fields')
    if base._config.fingerprint==result._config.fingerprint or base.case.source_hash==case.source_hash:
        raise ValueError('Declared catalog/native geometry identities differ unexpectedly')
    return result,{'old_native_config':base._config.fingerprint,'new_native_config':result._config.fingerprint,
        'native_config_except_declared_source_hash_equal':True,'physical_source_sha256':semantic_digest(physics.source_physics(case)),
        'post_exposure_authority':authority,'old_source_hash':base.case.source_hash,'new_source_hash':case.source_hash,
        'old_task_model':base.decision_model_hash,'new_task_model':result.decision_model_hash,
        'old_proposals':before.to_record(),'new_proposals':config.to_record(),'reward':asdict(base.reward_spec),
        'changed_binding_scope':'proposal/task/source observation catalog changes; original native geometry, rewards and prepared authority retained'}


def frontier_record(task,records):
    task._assert_frozen()
    row={'schema':'authenticated-focused-fixed-prefix-v1','source_hash':task.case.source_hash,
         'decision_model_hash':task.decision_model_hash,'native_model_hash':task._config.fingerprint,
         'native_state_hash':task._engine.state_hash,'committed_masks':dict(task._engine.committed_mask_digests()),
         'steps':task._steps,'current_tool':task._current_tool,'total_reward':task._total_reward,
         'nominal_history_hash':semantic_digest(task.metrics()['history']),
         'action_packets_hash':semantic_digest(records)}
    return {**row,'seal':semantic_digest(row)}


def require_frontier(task,records,frontier):
    if (type(task) is not FocusedPagedObstructionTask or not task._planning or task.terminated
        or len(records)!=task._steps or not 0<task._steps<task.max_steps<=24
        or any(p['action_id']=='STOP' for p in records)):
        raise ValueError('Live genuinely executed focused prefix required')
    exact(frontier_record(task,records),frontier,'sealed committed frontier')
    history=task.metrics()['history']
    for i,(p,r) in enumerate(zip(records,history)):
        if (p['steps_taken']!=i or p['action_id']!=r['action_id'] or p['focus'] is not None
            or p['source_hash']!=task.case.source_hash or p['decision_model_hash']!=task.decision_model_hash
            or p['seal']!=semantic_digest({k:v for k,v in p.items() if k!='seal'})):
            raise ValueError('Fixed prefix packet/history binding differs')
        exact(p['physical_action'],{k:r[k] for k in ('tool_id','entry_mm','tip_mm')},'prefix packet physical action')
    return True


def map_prefix(original,focused,saved_actions,saved_history,*,checkpoint=lambda:None):
    """Execute every supplied old motion normally; re-certify the new base catalog."""
    if original._steps or focused._steps or original._planning or not focused._planning:
        raise ValueError('Fresh authoritative old and nominal new tasks required')
    if len(saved_actions)!=len(saved_history) or any(a=='STOP' for a in saved_actions):
        raise ValueError('Nonterminal fixed motion prefix required')
    records=[];joins=[]
    for i,(action,expected) in enumerate(zip(saved_actions,saved_history)):
        checkpoint();original.observation();focused._activate(0)
        old=original._proposal_batch.proposals;new=focused._base_batch.proposals
        exact([ray_key(r) for r in old],[ray_key(r) for r in new],'base physical catalog order at prefix'+str(i))
        ray=next((r for r in old if r.proposal_id==action),None)
        candidates=[] if ray is None else [r for r in new if ray_key(r)==ray_key(ray)]
        if len(candidates)!=1 or candidates[0].proposal_id not in focused._inventory:
            raise ValueError('Fixed physical action absent or uncertified in current base catalog')
        packet=focused.export_action(candidates[0].proposal_id)
        a=original.step(action);b=focused.advance_record(packet);records.append(packet)
        exact(compact_history([a.info])[0],expected,'retained authoritative prefix full-row projection')
        exact(physical(a.info),physical(b.info),'prefix native geometry and reward')
        exact(physics.mask_record(original),physics.mask_record(focused),'prefix committed native masks')
        if (a.terminated or b.terminated or original._current_tool!=focused._current_tool
            or original._total_reward!=focused._total_reward):
            raise ValueError('Prefix state/tool/reward differs')
        joins.append({'step':i+1,'old_action_id':action,'new_action_id':packet['action_id'],
            'physical_geometry':packet['physical_action'],'original_native_state_hash':original._engine.state_hash,'focused_native_state_hash':focused._engine.state_hash,
            'mask_hashes':physics.mask_record(original),'reward':a.reward,'cumulative_return':original._total_reward,
            'original_full_row_hash':semantic_digest(a.info),'focused_nominal_full_row_hash':semantic_digest(b.info),
            'physical_history_equal':True,'base_physical_catalog_equal':True})
    return records,frontier_record(focused,records),joins


def replay_focused_tree(fresh,selected,explored,expected_history,*,checkpoint=lambda:None):
    """One real fresh full exploration; clone its real selected prefix for STOP.

    Cloning shares only an already executed physical prefix. Both complete
    endpoints are independently audited by the caller; no invented task history.
    """
    if fresh._steps or fresh._planning or type(fresh) is not FocusedPagedObstructionTask:
        raise ValueError('Fresh authoritative focused replay required')
    if not selected or not explored or len(explored)>fresh.max_steps:raise ValueError('Complete packets required')
    selected_stop=selected[-1]['action_id']=='STOP'
    n=len(selected)-int(selected_stop)
    exact(selected[:n],explored[:n],'selected plan belongs to full exploration prefix')
    if not selected_stop and (len(selected)!=fresh.max_steps or len(selected)!=len(explored)):
        raise ValueError('NonSTOP selected plan must reach absolute horizon')
    chosen=None;extra=0
    for i,packet in enumerate(explored):
        checkpoint()
        if i==n and selected_stop:
            if semantic_digest(packet)==semantic_digest(selected[-1]) and i==len(explored)-1:
                pass  # The exploration's own terminal STOP is the selected endpoint.
            else:
                chosen=fresh.clone();chosen.step_record(selected[-1]);extra+=1
        transition=fresh.step_record(packet)
        exact(physical(transition.info),physical(expected_history[i]),'fresh explored full native geometry/reward')
        if transition.terminated and i!=len(explored)-1:raise ValueError('Exploration packets after termination')
    if chosen is None:chosen=fresh
    if not fresh.terminated or not chosen.terminated:raise ValueError('Both replay endpoints must be complete')
    exact([r['action_id'] for r in chosen.metrics()['history']],[p['action_id'] for p in selected],'selected replay IDs')
    return chosen,fresh,{'shared_fresh_prefix_transitions':n,'exploration_transitions':len(explored),
        'selected_additional_STOP_transitions':extra,'distinct_complete_endpoints':1 if chosen is fresh else 2}


