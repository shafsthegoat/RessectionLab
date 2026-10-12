"""One explicit TRAIN045 local access task; native geometry/reward remain unchanged.

Only the wrapper objective differs. A bonus requires a newly committed removal
of the declared public target cell. Arrival, action identity and mere preview
feasibility never earn a bonus. All current certified catalog actions are kept.
"""
from dataclasses import dataclass,asdict,replace
import math,time
import paths
import numpy as np
from resectionlab.core import freeze_json,thaw_json,semantic_digest
from resectionlab.focused_paged_task import FocusedPagedObstructionTask
from resectionlab.spatial_observations import SpatialAction,ObservedProcedureState,build_spatial_observation
import focused_policy as reused

VERSION='train045-public-local-goal-v1'
TOKEN=object()

@dataclass(frozen=True)
class GoalSpec:
    cell:tuple[int,int,int]
    tool_id:str
    entry_mm:tuple[float,float,float]
    tip_mm:tuple[float,float,float]
    bonus:float=10.
    action_cost:float=.03
    motion_per_mm:float=.001
    tool_change_cost:float=.03
    normal_per_mm3:float=.2
    def __post_init__(self):
        if (len(self.cell)!=3 or any(type(v) is not int or v<0 for v in self.cell)
            or not self.tool_id or len(self.entry_mm)!=3 or len(self.tip_mm)!=3
            or not all(math.isfinite(float(v)) for v in (*self.entry_mm,*self.tip_mm))
            or np.linalg.norm(np.subtract(self.tip_mm,self.entry_mm))<=1e-9
            or (self.bonus,self.action_cost,self.motion_per_mm,self.tool_change_cost,self.normal_per_mm3)!=(10.,.03,.001,.03,.2)):
            raise ValueError('Exact finite local-goal reward/physical declaration required')
    def record(self):
        return {**asdict(self),'version':VERSION,'success':'newly_committed_public_goal_cell_removal',
            'reward_scope':'separate_simulated_local_task;original_native_reward_and_history_unchanged',
            'focus_rule':'fixed_physical_goal_if_current_base_ray_else_none;goal_never_changes'}
    @property
    def key(self):return (self.tool_id,tuple(self.entry_mm),tuple(self.tip_mm))


@dataclass(frozen=True)
class LocalContext:
    original_context:object
    record:object
    seal:str
    goal:GoalSpec
    _config:object
    _prefix_hash:str
    _token:object
    def require_task(self,task):
        if (self._token is not TOKEN or semantic_digest(self.record)!=self.seal
            or semantic_digest(self.goal.record())!=self.record['goal_hash']
            or self.original_context.fingerprint!=self.record['original_context_hash']
            or type(task) is not FocusedPagedObstructionTask or not task._planning):
            raise ValueError('Exact declared local TRAIN task required')
        task._assert_frozen()
        if (task._config is not self._config or task.case._native_config is not self._config
            or task.case.reference_target is not task.case.nominal_target
            or task.case.source_hash!=self.record['focused_source_hash']
            or task.decision_model_hash!=self.record['focused_task_model']
            or task.max_steps!=24 or task.tool_modes is not None
            or not self.record['prefix_length']<=task._steps<=24
            or semantic_digest(task._history[:self.record['prefix_length']])!=self._prefix_hash
            or semantic_digest(asdict(task.reward_spec))!=self.record['native_reward_hash']):
            raise ValueError('Local source/native model/prefix/reward changed')
        return task


def admit_local_context(original,original_context,task,packets,frontier,goal,*,expected_prefix_actions=12):
    # Reuse the complete original/expanded source-and-prefix checks, but declare
    # a NEW task context. The old inference-only context is not a training grant.
    old=reused.admit_focused_context(original,original_context,task,packets,frontier,
        expected_prefix_actions=expected_prefix_actions)
    cell=goal.cell
    if (any(i>=n for i,n in zip(cell,task._engine.remaining_mask.shape))
        or not task._engine.remaining_mask[cell] or task._engine.removed_mask[cell]
        or float(task.case.nominal_target[cell])<=task.case.proposal_config.nominal_min_membership):
        raise ValueError('Initially remaining, public nominal-positive goal cell required')
    origin=task.case._crop_origin;shape=task.case._crop_shape
    if any(not a<=i<a+n for i,a,n in zip(cell,origin,shape)):
        raise ValueError('Goal cell must be represented in the public actor crop')
    base=task.case._nominal_proposer.propose(task._engine,cancelled=task._cancelled)
    if goal.key not in {reused.physical_key(r) for r in base.proposals if tuple(r.voxel)==cell}:
        raise ValueError('Declared goal must be an existing public base ray at qualification start')
    record=freeze_json({'version':VERSION,'scope':'explicit_simulated_local_goal_training_and_matched_search',
        'role':'TRAIN','subject':'ReMIND-045','original_context_hash':original_context.fingerprint,
        'focused_source_hash':task.case.source_hash,'focused_task_model':task.decision_model_hash,
        'focused_native_model':task._config.fingerprint,'prefix_length':expected_prefix_actions,
        'prefix_packets_hash':semantic_digest(packets),'prefix_state_hash':task._engine.state_hash,
        'physical_source_hash':old.record['physical_source_hash'],'source_prefix_validation':old.seal,
        'native_reward_hash':semantic_digest(asdict(task.reward_spec)),
        'goal':goal.record(),'goal_hash':semantic_digest(goal.record()),'absolute_horizon':24,
        'actions':'STOP_and_all_current_native_feasible_base_and_bounded_focus_pages',
        'immediate_positive_guard':False,'candidate_limit':153,'page_size':8,
        'observation':'raw_public_six_channels_and_coverage;separate_fixed_goal_geometry_and_cell',
        'reference_sequence_is_actor_input':False,'private_reference_is_actor_input':False,
        'evaluation_start_absolute_step':expected_prefix_actions})
    result=LocalContext(original_context,record,semantic_digest(record),goal,task._config,
        semantic_digest(task._history),TOKEN)
    result.require_task(task);return result


def collect_local_catalog(work,context,*,guard=lambda:None,costs=None):
    """Same complete pages/certificates, without automatic all-goal re-previews."""
    costs={} if costs is None else costs;started=time.perf_counter()
    context.require_task(work);guard()
    if work.terminated or work._active_page is not None or work._base_inventory is not None:
        raise ValueError('Fresh live uncertified state required')
    base=work.case._nominal_proposer.propose(work._engine,cancelled=work._cancelled)
    available={reused.physical_key(r) for r in base.proposals if
        float(work.case.nominal_target[r.voxel])>work.case.proposal_config.nominal_min_membership}
    focus=work.focus_record(context.goal.key) if context.goal.key in available else None
    work.set_focus(focus);work._prepare_base()
    pages=[];identifiers=['STOP'];owners=[0];packets=[];scores=[{'action_id':'STOP','reward':0.}]
    rows=[];base_digests={};page_records=[];tools={t.tool_id:t for t in work.case.tools}
    for page,ids in work.planning_action_pages():
        guard();context.require_task(page);page_index=len(pages);pages.append(page)
        for identifier in page._base_inventory:
            _,geometry=reused._certificate(page,identifier);digest=semantic_digest(geometry)
            if page_index==0:base_digests[identifier]=digest
            elif base_digests.get(identifier)!=digest:raise ValueError('Repeated base certificate differs')
        if page_index==0:packets.append(page.export_action('STOP'))
        for identifier in ids:
            if identifier in identifiers:raise ValueError('Duplicate union action')
            preview,geometry=reused._certificate(page,identifier)
            score=page._score_record({**geometry,'action_id':identifier},page._current_tool)
            if score['outcome_scope']!='permitted_nominal_model':raise ValueError('Private outcome at catalog boundary')
            identifiers.append(identifier);owners.append(page_index);scores.append(score)
            packets.append(page.export_action(identifier))
            rows.append(SpatialAction(identifier,preview.entry_mm,preview.tip_mm,tools[preview.tool_id]))
        page_records.append(page.planning_page_record())
    if not 1<=len(identifiers)<=153 or len(pages)>5:raise ValueError('Catalog bound exceeded')
    cavity=work._engine.connected_free_mask if work.case.post_exposure is not None else work._engine.removed_mask
    observation=build_spatial_observation(work.case.spatial_inputs(cavity),[SpatialAction('STOP'),*rows],
        ObservedProcedureState(work.case.access,work._steps,work.max_steps,work._current_tool))
    if work.case._public_target_source is not None:
        observation=replace(observation,public_target_context=work.case._public_target_source.observe(cavity))
    record={'schema':VERSION,'source_state_hash':work._engine.state_hash,'steps':work._steps,
        'goal_focus':work._goal_focus,'query_catalog_digest':work._query_catalog_digest,
        'fixed_goal':context.goal.record(),'goals':[],'pages':page_records,'complete_page_enumeration':True,
        'distinct_certified_actions':len(identifiers)-1,'observation_hash':observation.fingerprint,
        'automatic_goal_diagnostic_previews':0,'native_reward_not_actor_input':True}
    result=reused.Catalog(context,work,pages,identifiers,owners,packets,scores,observation,record)
    result.require_batched()
    for key,n in (('catalog_calls',1),('pages',len(pages)),('union_observations',1)):
        costs[key]=costs.get(key,0)+n
    costs['catalog_seconds']=costs.get('catalog_seconds',0.)+time.perf_counter()-started
    return result


def local_reward(goal,record,prior_tool,*,already_succeeded):
    """Pure accounting of an executed native row; no arrival/ID-only bonus."""
    if record['action_id']=='STOP':
        if record.get('removed_indices_native'):raise ValueError('STOP cannot remove goal')
        return 0.,False,{'goal_bonus':0.,'action_cost':0.,'travel_cost':0.,'tool_change_cost':0.,'outside_cost':0.}
    if (record.get('outcome_scope') not in ('permitted_nominal_model','separate_evaluator_reference')
        or not math.isfinite(float(record['normal_removed_mm3'])) or record['normal_removed_mm3']<0
        or not math.isfinite(float(record['complete_tool_path_length_mm'])) or record['complete_tool_path_length_mm']<0):
        raise ValueError('Executed finite native physical accounting required')
    cells={tuple(c) for c in record['removed_indices_native']}
    success=not already_succeeded and goal.cell in cells
    terms={'goal_bonus':goal.bonus if success else 0.,'action_cost':goal.action_cost,
        'travel_cost':goal.motion_per_mm*record['complete_tool_path_length_mm'],
        'tool_change_cost':goal.tool_change_cost*int(prior_tool is not None and prior_tool!=record['tool_id']),
        'outside_cost':goal.normal_per_mm3*record['normal_removed_mm3']}
    return terms['goal_bonus']-sum(v for k,v in terms.items() if k!='goal_bonus'),success,terms


def goal_geometry(context,task):
    """Public fixed RAS goal and source-cell indicator, separate from six channels."""
    goal=context.goal;shape=tuple(task.case._crop_shape);origin=tuple(task.case._crop_origin)
    cell=tuple(i-a for i,a in zip(goal.cell,origin));mask=np.zeros((1,*shape),np.float32);mask[(0,*cell)]=1.
    axis=np.subtract(goal.tip_mm,goal.entry_mm);axis=axis/np.linalg.norm(axis)
    tools={t.tool_id:t for t in task.case.tools};tool=tools[goal.tool_id]
    vector=np.asarray((*goal.entry_mm,*goal.tip_mm,*axis,tool.tip_radius_mm,tool.shaft_radius_mm),np.float32)
    return mask,vector


class LocalEpisode:
    """Native planning branch for Gym and same-objective bounded search."""
    def __init__(self,context,snapshot,*,guard=lambda:None):
        context.require_task(snapshot)
        if snapshot.terminated or snapshot._active_page is not None or snapshot._base_inventory is not None:
            raise ValueError('Clean genuine reset snapshot required')
        if snapshot._engine.removed_mask[context.goal.cell]:raise ValueError('Reset cannot start after success')
        self.context=context;self.work=snapshot.clone();self.guard=guard;self.catalog=None
        self.start_step=self.work._steps;self.packets=[];self.rows=[];self.total_reward=0.;self.success=False
        self.done=False;self.costs={};self.invalid_action_attempts=0;self.last_clean_snapshot=None
    def observe(self):
        self.guard();self.context.require_task(self.work)
        if self.done:raise ValueError('No new catalog after terminal closure')
        if self.catalog is None:self.catalog=collect_local_catalog(self.work,self.context,guard=self.guard,costs=self.costs)
        return self.catalog
    def clone(self):
        """Branch a clean committed state; no stale certificate is transplanted."""
        import copy
        if self.catalog is not None:raise ValueError('Clone before observation; use snapshot for a clean state')
        other=copy.copy(self);other.work=self.work.clone();other.packets=list(self.packets)
        other.rows=list(self.rows);other.costs=dict(self.costs);return other
    def step(self,index):
        catalog=self.observe()
        if type(index) is not int or not 0<=index<len(catalog.identifiers):
            self.invalid_action_attempts+=1;raise ValueError('Unmasked current union action index required')
        prior=self.work._current_tool;before=bool(self.work._engine.removed_mask[self.context.goal.cell])
        if before!=self.success:raise ValueError('Goal state/accounting mismatch')
        page,packet,transition=catalog.commit(index);self.catalog=None;self.work=page
        reward,new_success,terms=local_reward(self.context.goal,transition.info,prior,already_succeeded=self.success)
        after=bool(self.work._engine.removed_mask[self.context.goal.cell])
        if after!=(before or new_success):raise ValueError('Claimed goal removal differs from committed cavity')
        self.success=self.success or new_success;self.total_reward+=reward;self.packets.append(packet)
        self.rows.append({'absolute_step':page._steps,'action_id':packet['action_id'],'reward':reward,
            'native_reward':transition.reward,'terms':terms,'new_goal_removal':new_success,
            'source_state_hash':packet['source_state_hash'],'result_state_hash':page._engine.state_hash})
        self.costs['policy_native_transitions']=self.costs.get('policy_native_transitions',0)+1
        # Finite-horizon MDP: remaining absolute clock is observed. Resource
        # errors propagate as failures, never as successful STOP/truncation.
        self.done=transition.terminated or self.success
        self.last_clean_snapshot=None if self.done else self.work.clone()
        if self.success and not page.terminated:
            page._activate(0);stop=page.export_action('STOP');closed=page.advance_record(stop)
            if not closed.terminated:raise ValueError('Automatic success closure failed')
            self.packets.append(stop);self.costs['automatic_STOP_transitions']=self.costs.get('automatic_STOP_transitions',0)+1
        return reward,self.done,{'success':self.success,'task_return':self.total_reward,
            'native_reward':transition.reward,'terms':terms,'absolute_step':self.work._steps,
            'automatic_STOP_closure':self.success and packet['action_id']!='STOP' and self.packets[-1]['action_id']=='STOP'}
