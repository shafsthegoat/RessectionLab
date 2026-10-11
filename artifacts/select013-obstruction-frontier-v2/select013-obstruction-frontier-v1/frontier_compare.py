"""Fixed physical-prefix comparison and bounded one-cut diagnostic branches."""
from dataclasses import asdict,fields,is_dataclass,replace
from collections.abc import Mapping
import numpy as np
from resectionlab.core import array_digest,semantic_digest,thaw_json
from resectionlab.native_proposals import OBSTRUCTION_OPENING_FAMILY
from resectionlab.native_spatial_task import NativeSpatialTask

BINDING_KEYS={'action_id','source_hash','source_state_hash','decision_model_hash'}

def plain(value):
    if isinstance(value,np.ndarray):return {'array_sha256':array_digest(value),'shape':list(value.shape),'dtype':str(value.dtype)}
    if is_dataclass(value):return {f.name:plain(getattr(value,f.name)) for f in fields(value) if f.init}
    if isinstance(value,Mapping):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError('Unaccounted source value '+type(value).__name__)


def source_physics(case):
    # All public constructor fields except the single declared proposal rule.
    return {f.name:plain(getattr(case,f.name)) for f in fields(case) if f.init and f.name!='proposal_config'}


def enabled_fresh(base):
    """Rebuild; never mutate a live task or manufacture an admission capability."""
    if base.case.proposal_config.obstruction_opening:raise ValueError('Original disabled rule required')
    config=replace(base.case.proposal_config,obstruction_opening=True)
    case=replace(base.case,proposal_config=config)
    before,after=source_physics(base.case),source_physics(case)
    if before!=after:raise ValueError('Changing the proposal rule changed physical source')
    other=NativeSpatialTask(case,max_steps=base.max_steps,reward=base.reward_spec,cancelled=base._cancelled).planning_clone()
    if (case.source_hash==base.case.source_hash or other.decision_model_hash==base.decision_model_hash
        or config.max_candidates!=base.case.proposal_config.max_candidates):raise ValueError('Declared model change/cap binding failed')
    return other,{'all_other_source_fields_equal':True,'public_source_physics':before,
        'baseline_source_hash':base.case.source_hash,'enabled_source_hash':case.source_hash,
        'baseline_decision_model_hash':base.decision_model_hash,'enabled_decision_model_hash':other.decision_model_hash,
        'baseline_proposal_config':base.case.proposal_config.to_record(),'enabled_proposal_config':config.to_record(),
        'max_steps':base.max_steps,'reward':asdict(base.reward_spec),
        'scope':'explicit proposal diagnostic; historical SELECT inference capability does not admit this changed model'}


def physical_history(row):return {k:v for k,v in row.items() if k not in BINDING_KEYS}

def physical_ledger(rows):return [{k:v for k,v in r.items() if k not in {'action_id','proposal_id'}} for r in rows]

def physical_key(row):return (row['family'],row['tool_id'],tuple(row['entry_mm']),tuple(row['tip_mm']))

def exact_emitted(inventory,*,action_id=None,key=None):
    rows=[r for r in inventory['emitted'] if (r['action_id']==action_id if action_id is not None else physical_key(r)==key)]
    if len(rows)!=1 or not rows[0]['feasible']:raise ValueError('Fixed physical action not uniquely emitted and feasible')
    return rows[0]


def mask_record(task):
    task._assert_frozen()
    return dict(task._engine.committed_mask_digests())


def compare_prefix(baseline,enabled,saved_actions,saved_history,*,write,guard=lambda:None,on_transition=lambda:None):
    if not baseline._planning or not enabled._planning:raise ValueError('Nominal planning feedback only')
    if len(saved_actions)!=len(saved_history):raise ValueError('Fixed history length differs')
    mapped=[];baseline_history=[];enabled_history=[]
    for i,action in enumerate(saved_actions):
        guard();old=baseline.candidate_inventory();new=enabled.candidate_inventory()
        prefix_equal=physical_ledger(old['ledger'])==physical_ledger(new['ledger'][:len(old['ledger'])])
        if not prefix_equal or mask_record(baseline)!=mask_record(enabled):raise ValueError('Original inventory/physical state differs')
        original=exact_emitted(old,action_id=action);added=exact_emitted(new,key=physical_key(original))
        a=baseline.step(action);on_transition();b=enabled.step(added['action_id']);on_transition()
        baseline_history.append(a.info);enabled_history.append(b.info);mapped.append(added['action_id'])
        full_old_equal=semantic_digest(a.info)==semantic_digest(saved_history[i])
        physical_equal=semantic_digest(physical_history(a.info))==semantic_digest(physical_history(b.info))
        masks_equal=mask_record(baseline)==mask_record(enabled)
        record={'step':i+1,'old_action_id':action,'new_action_id':added['action_id'],'physical_key':plain(physical_key(original)),
            'base_ledger_prefix_equal':prefix_equal,'baseline_saved_history_equal':full_old_equal,
            'physical_history_equal':physical_equal,'committed_masks_equal':masks_equal,
            'baseline_mask_hashes':mask_record(baseline),'enabled_mask_hashes':mask_record(enabled),
            'baseline_history':a.info,'enabled_history':b.info}
        write('prefix-%02d.json'%(i+1),record)
        if not full_old_equal or not physical_equal or not masks_equal:raise ValueError('Physical prefix mismatch at '+str(i+1))
    return mapped,baseline_history,enabled_history


def frontier_branches(task,baseline_inventory,excluded,*,write,inspect,guard=lambda:None,on_transition=lambda:None,on_preview=lambda:None,max_branches=3):
    """Try every feasible emitted addition in provider order; no reward filtering.

    Every trial starts at the exact same cloned frontier. Direct original-ray
    previews are evidence only and are never injected as task actions.
    """
    if not task._planning:raise ValueError('Nominal planning task required')
    current=task.candidate_inventory();base_len=len(baseline_inventory['ledger'])
    if physical_ledger(current['ledger'][:base_len])!=physical_ledger(baseline_inventory['ledger']):raise ValueError('Original full ledger changed')
    additions=[r for r in current['emitted'] if r['family']==OBSTRUCTION_OPENING_FAMILY]
    omissions=[r for r in current['ledger'] if r['family']==OBSTRUCTION_OPENING_FAMILY and r['proposal_reason'] in {'CANDIDATE_CAP','OBSTRUCTION_SELECTION_CAP'}]
    write('enabled-frontier-inventory.json',current)
    write('added-dispositions.json',{'emitted':additions,'omitted_cell_tool_pairs':omissions,
        'all_added_ledger_rows':current['ledger'][base_len:],'selection':'all emitted feasible additions in existing provider order',
        'discovery_complete':False,'obstruction_accounting':current['obstruction_accounting']})
    legal=[r for r in additions if r['feasible']]
    if len(additions)>max_branches or len(legal)>max_branches:raise ValueError('Predeclared added-slot cap changed')
    original_rays=tuple(task._proposal_batch.proposals[:baseline_inventory['emitted_count']])
    if len(original_rays)!=baseline_inventory['emitted_count']:raise ValueError('Original ray prefix missing')
    from types import SimpleNamespace
    # These are actual currently emitted original rays, with enabled-model IDs.
    if [physical_key({'family':r.family,'tool_id':r.tool_id,'entry_mm':r.entry_mm,'tip_mm':r.tip_mm}) for r in original_rays]!=[physical_key(r) for r in baseline_inventory['emitted']]:raise ValueError('Original ray order changed')
    origin=task._engine.state_hash;excluded={tuple(c) for c in excluded};branches=[]
    for i,row in enumerate(legal):
        guard();branch=task.clone();prior_total=branch._total_reward
        outcome=branch.step(row['action_id']);on_transition()
        removed={tuple(c) for c in outcome.info['removed_indices_native']};intersection=sorted(removed&excluded)
        previews=[];state=branch._engine.state_hash
        for j,ray in enumerate(original_rays):
            guard();record=inspect(branch,ray);on_preview();previews.append(record)
            write('branch-%02d/rejection-%03d.json'%(i,j),record)
        if branch._engine.state_hash!=state:raise ValueError('Re-preview changed committed state')
        success=[{'family':r['family'],'tool_id':r['tool_id'],'entry_mm':r['entry_mm'],'tip_mm':r['tip_mm']} for r in previews if r['feasible']]
        stop=branch.step('STOP');on_transition()
        if stop.reward!=0 or not branch.terminated:raise ValueError('Branch STOP failed')
        record={'branch':i,'preparation':row,'preparation_history':outcome.info,'excluded_cells_actually_removed':intersection,
            'any_excluded_cell_removed':bool(intersection),'restored_original_rays':success,'any_original_ray_feasible':bool(success),
            'preparation_target_mm3':outcome.info['target_removed_mm3'],'preparation_outside_target_mm3':outcome.info['normal_removed_mm3'],
            'prefix_return':prior_total,'preparation_return':outcome.reward,'complete_negative_prefix_plus_preparation_STOP_return':branch._total_reward,
            'root_STOP_return':0.,'frontier_STOP_increment':0.,'complete_target_route_claim':False,'accepted_plan_claim':False,
            'diagnostic_preview_count':len(previews),'history':branch.metrics()['history'],
            'remaining_mask_hashes':mask_record(branch),'closed_episode':True,
            'preview_scope':'first rejected interval only; direct feasible rays are not executed or injected into inventory'}
        write('branch-%02d/result.json'%i,record);branches.append(record)
        if task._engine.state_hash!=origin:raise ValueError('Preparation branch changed common frontier')
        del branch
    return {'branch_count':len(branches),'added_emitted':len(additions),'added_legal':len(legal),
        'omitted_cell_tool_pairs':len(omissions),'any_excluded_cell_removed':any(r['any_excluded_cell_removed'] for r in branches),
        'any_original_ray_feasible':any(r['any_original_ray_feasible'] for r in branches),
        'any_preparation_target_progress':any(r['preparation_target_mm3']>0 for r in branches),
        'best_closed_branch_return':max([0.,*(r['complete_negative_prefix_plus_preparation_STOP_return'] for r in branches)]),
        'complete_target_route_claim':False,'censored_by_omitted_pairs':bool(omissions),
        'branches':[{'branch':r['branch'],'return':r['complete_negative_prefix_plus_preparation_STOP_return']} for r in branches]}
