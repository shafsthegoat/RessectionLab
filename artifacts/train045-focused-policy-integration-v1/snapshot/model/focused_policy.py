"""One explicit focused-catalog deployment capability, with unchanged actor inputs.

This is an action-family transfer experiment on a bound TRAIN world, not a new
training context. Old task/context/scorer guards remain untouched. Both methods
see the same fully certified catalog and use the same public focus controller.
"""
from dataclasses import dataclass, replace, asdict
import hashlib
import importlib.util
import math
from pathlib import Path
import time

import numpy as np
import torch

from resectionlab.core import freeze_json, thaw_json, semantic_digest
from resectionlab.patient_planning_admission import PatientPlanningContext
from resectionlab.native_spatial_task import NativeSpatialTask
from resectionlab.focused_paged_task import FocusedPagedObstructionTask
from resectionlab.focused_corridor_selector import physical_key, goal_record
from resectionlab.native_proposals import OBSTRUCTION_CELLS_PER_PREVIEW
from resectionlab.spatial_observations import (SpatialObservation, SpatialAction,
    ObservedProcedureState, build_spatial_observation)
from resectionlab.spatial_continuation_value import _world, _feature, _costs
from resectionlab.spatial_continuation_tail import FittedContinuationTail

VERSION = 'TRAIN045-focused-full-catalog-frozen-policy-v1'
CONTROLLER = 'initial_none_then_first_current_goal_overlap_active_first_v1'
_CAPABILITY = object()
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
_path = ROOT/'build/train045-frontier-preparation-v1/recovery-compact/frontier_protocol.py'
if hashlib.sha256(_path.read_bytes()).hexdigest() != '8c878476fa62e63ff721b98085fe78672162ffcbe75f76c8c615958ec2239df0':
    raise ValueError('Reviewed physical-prefix helper changed')
_physics = ROOT/'build/select013-obstruction-frontier-v2/frontier_compare.py'
if hashlib.sha256(_physics.read_bytes()).hexdigest() != 'e02d7b02ba8d4c8188f1a3643b44333fa08ab4db413355324704c58d72abd20b':
    raise ValueError('Reviewed source-physics helper changed')
_spec = importlib.util.spec_from_file_location('_focused_policy_frontier', _path)
protocol = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(protocol)


def _same(a, b, name):
    if semantic_digest(a) != semantic_digest(b):
        raise ValueError('Focused binding differs: ' + name)


@dataclass(frozen=True)
class FocusedContext:
    """Callback-local admission; holds no authority to admit different anatomy."""
    original_context: object
    record: object
    seal: str
    _config: object
    _prefix_hash: str
    _token: object

    def require_task(self, task):
        if (self._token is not _CAPABILITY or semantic_digest(self.record) != self.seal
                or self.original_context.fingerprint != self.record['original_context_hash']
                or type(task) is not FocusedPagedObstructionTask or not task._planning):
            raise ValueError('Exact admitted nominal focused task required')
        task._assert_frozen()
        if (task._config is not self._config or task.case._native_config is not self._config
                or task.case.source_hash != self.record['focused_source_hash']
                or task.decision_model_hash != self.record['focused_task_model']
                or task.case.reference_target is not task.case.nominal_target
                or task.max_steps != 24 or task.tool_modes is not None
                or not self.record['prefix_length'] <= task._steps <= 24
                or semantic_digest(task.metrics()['history'][:self.record['prefix_length']]) != self._prefix_hash):
            raise ValueError('Focused current source/world/prefix changed')
        return task


def admit_focused_context(original_task, original_context, focused_prefix_task,
        prefix_packets, frontier, *, expected_prefix_actions=12):
    """Check genuinely executed paired physical prefixes and declare only catalog delta.

    The runner fixes expected_prefix_actions=12. Generated interface controls
    explicitly declare a smaller genuinely executed prefix, never a forged state.
    """
    if type(original_context) is not PatientPlanningContext or type(original_task) is not NativeSpatialTask:
        raise TypeError('Exact original admitted task/context required')
    original_context.require_task(original_task)
    world = _world(original_context, training=False)
    if (original_context.role != 'TRAIN' or world['subject'] != 'ReMIND-045'
            or type(expected_prefix_actions) is not int or not 1 <= expected_prefix_actions < 24
            or original_task.max_steps != 24 or original_task._planning
            or original_task._steps != expected_prefix_actions or original_task.terminated):
        raise ValueError('Declared live TRAIN045 fixed prefix required')
    if (expected_prefix_actions != 12
            and original_context.record().get('evidence_domain') != 'generated_interface_control'):
        raise ValueError('Acquired comparison has exactly twelve fixed prefix actions')
    protocol.require_frontier(focused_prefix_task, prefix_packets, frontier)
    task = focused_prefix_task
    if task._steps != expected_prefix_actions or task.case.reference_target is not task.case.nominal_target:
        raise ValueError('Nominal focused prefix length/source differs')
    before = original_task.case.proposal_config
    expected = replace(before, obstruction_opening=True, tool_footprint_opening=True,
        obstruction_page_size=8, obstruction_selection_rule='target_ray_coverage_v3')
    if before.obstruction_opening or before.max_candidates != 120:
        raise ValueError('Original 120-pair catalog required')
    _same(task.case.proposal_config.to_record(), expected.to_record(), 'only declared provider delta')
    _same(protocol.physics.source_physics(original_task.case), protocol.physics.source_physics(task.case), 'all physical source fields')
    authority = protocol.physics.post_exposure_authority(original_task.case.post_exposure, task.case.post_exposure)
    a, b = protocol.physics.plain(original_task._config), protocol.physics.plain(task._config)
    _same({k:v for k,v in a.items() if k != 'source_hash'},
          {k:v for k,v in b.items() if k != 'source_hash'}, 'native physical configuration')
    _same(asdict(original_task.reward_spec), asdict(task.reward_spec), 'unchanged reward')
    _same([protocol.physical(r) for r in original_task.metrics()['history']],
          [protocol.physical(r) for r in task.metrics()['history']], 'all prefix physical transitions')
    _same(original_task._engine.committed_mask_digests(), task._engine.committed_mask_digests(), 'prefix masks')
    if (original_task._current_tool != task._current_tool or original_task._total_reward != task._total_reward
            or original_task.case.source_hash == task.case.source_hash):
        raise ValueError('Prefix state or explicit changed catalog binding differs')
    record = freeze_json({'version':VERSION, 'scope':'inference_only_new_focused_action_catalog',
        'original_context_hash':original_context.fingerprint, 'trained_world':world,
        'focused_source_hash':task.case.source_hash, 'focused_task_model':task.decision_model_hash,
        'focused_native_model':task._config.fingerprint, 'frontier':frontier,
        'prefix_length':expected_prefix_actions, 'prefix_packets_hash':semantic_digest(prefix_packets),
        'physical_source_hash':semantic_digest(protocol.physics.source_physics(task.case)),
        'physical_prefix_hash':semantic_digest([protocol.physical(r) for r in task.metrics()['history']]),
        'original_proposals':before.to_record(), 'focused_proposals':expected.to_record(),
        'authority':authority, 'reward':asdict(task.reward_spec), 'controller':CONTROLLER,
        'full_actor_catalog':'STOP_then_base_once_then_page_extras', 'candidate_limit':153,
        'feature_change':False, 'training_updates':0, 'private_reference_input':False})
    result = FocusedContext(original_context, record, semantic_digest(record), task._config,
        semantic_digest(task.metrics()['history']), _CAPABILITY)
    result.require_task(task)
    return result


def _certificate(page, identifier):
    page._assert_frozen()
    preview = page._inventory[identifier]
    if (not page._planning or not preview.feasible
            or page._engine._preview_records.get(id(preview)) is not preview
            or page._engine._preview_digests.get(id(preview)) != page._engine._result_digest(preview)):
        raise ValueError('Current owned nominal certificate required')
    geometry = preview.to_history_record()
    if (geometry['source_state_hash'] != page._engine.state_hash
            or geometry['decision_model_hash'] != page._config.fingerprint):
        raise ValueError('Foreign native certificate')
    return preview, geometry


class Catalog:
    """Owned union DTO and exact page packets; cannot authorize detached rows."""
    def __init__(self, context, work, pages, identifiers, owners, packets, scores, observation, record):
        self.context, self.work, self.pages = context, work, tuple(pages)
        self.identifiers, self.owners = tuple(identifiers), tuple(owners)
        self.packets, self.scores = freeze_json(packets), freeze_json(scores)
        self.observation, self.record = observation, freeze_json(record)
        self._token, self._consumed = _CAPABILITY, False
        self._seal = self._digest()

    def _digest(self):
        return semantic_digest({'context':self.context.seal, 'record':self.record,
            'ids':self.identifiers, 'owners':self.owners, 'packets':self.packets,
            'scores':self.scores, 'observation':self.observation.fingerprint})

    def require(self):
        if self._token is not _CAPABILITY or self._consumed or self._digest() != self._seal:
            raise ValueError('Consumed or altered focused catalog')
        self.context.require_task(self.work)
        self.observation.assert_intact()
        if (type(self.observation) is not SpatialObservation
                or self.observation.source_id != self.context.record['focused_source_hash']
                or tuple(self.observation.action_ids) != self.identifiers
                or not np.all(self.observation.action_mask)
                or float(self.observation.state_features[1]) != 24):
            raise ValueError('Union public observation/source/mask changed')
        for i, (identifier, owner) in enumerate(zip(self.identifiers, self.owners)):
            page = self.pages[owner]; self.context.require_task(page)
            _same(page.export_action(identifier), self.packets[i], 'current packet/page ownership')
            if identifier != 'STOP':
                _, geometry = _certificate(page, identifier)
                _same(page._score_record({**geometry,'action_id':identifier}, page._current_tool),
                      self.scores[i], 'current nominal score')
        return self

    def commit(self, index):
        self.require()
        if type(index) is not int or not 0 <= index < len(self.identifiers):
            raise ValueError('Current union row required')
        page = self.pages[self.owners[index]]
        packet = self.packets[index]
        self._consumed = True
        transition = page.advance_record(packet)
        return page, packet, transition


def collect_catalog(work, context, active_goal=None, *, guard=lambda:None, costs=None):
    """Exhaust all pages, then build exactly one ordinary public actor observation.

    Goal sidecars control focus only. They never enter policy tensors/features.
    Native work and the one observation are explicit; no successor observation.
    """
    costs = {} if costs is None else costs
    for key in ('goal_previews_started','goal_previews_completed','pages_completed','union_observations'):
        costs.setdefault(key,0)
    context.require_task(work); guard()
    if work.terminated or work._active_page is not None or work._base_inventory is not None:
        raise ValueError('Fresh live uncertified current state required')
    cleared = None
    if active_goal is not None:
        batch = work.case._nominal_proposer.propose(work._engine, cancelled=work._cancelled)
        current = {physical_key(r) for r in batch.proposals if
            float(work.case.nominal_target[r.voxel]) > work.case.proposal_config.nominal_min_membership}
        if active_goal not in current:
            cleared = goal_record(active_goal); active_goal = None
    work.set_focus(None if active_goal is None else work.focus_record(active_goal))
    work._prepare_base(); state = work._engine.state_hash
    keys = [] if active_goal is None else [active_goal]
    for ray in work._base_batch.proposals:
        if float(work.case.nominal_target[ray.voxel]) > work.case.proposal_config.nominal_min_membership:
            key = physical_key(ray)
            if key not in keys:keys.append(key)
    if len(keys)>120:raise ValueError('Goal query ceiling changed')
    goals=[]; diagnostic=work._engine.clone()
    for key in keys:
        guard(); costs['goal_previews_started']+=1
        outcome=diagnostic.preview_stroke(key[0],key[2],entry_mm=key[1],
            obstruction_diagnostics=True,obstruction_cell_limit=OBSTRUCTION_CELLS_PER_PREVIEW)
        costs['goal_previews_completed']+=1
        evidence=outcome.obstruction_diagnostic; cells=()
        if evidence is not None:
            if (semantic_digest({k:v for k,v in evidence.items() if k!='fingerprint'})!=evidence['fingerprint']
                    or evidence['source_state_hash']!=state or evidence['source_hash']!=work._config.source_hash
                    or evidence['decision_model_hash']!=work._config.fingerprint
                    or tuple(evidence['entry_mm'])!=key[1] or tuple(evidence['requested_tip_mm'])!=key[2]
                    or evidence['tool']['tool_id']!=key[0]):
                raise ValueError('Current public goal sidecar changed')
            cells=evidence['blocked_indices_native']
        elif outcome.reason=='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE':
            raise ValueError('Missing bounded current goal sidecar')
        goals.append({'goal':goal_record(key),'feasible':outcome.feasible,'reason':outcome.reason,
            'retained_blocker_cells':cells,'sidecar':evidence})
    del diagnostic
    pages=[]; owners=[0]; identifiers=['STOP']; packets=[]; scores=[{'action_id':'STOP','reward':0.}]
    rows=[]; base_digests={}; page_records=[]; tools={t.tool_id:t for t in work.case.tools}
    for page, ids in work.planning_action_pages():
        guard(); context.require_task(page); page_index=len(pages); pages.append(page)
        inventory=page.candidate_inventory()
        for identifier in page._base_inventory:
            _, geometry=_certificate(page,identifier)
            digest=semantic_digest(geometry)
            if page_index==0:base_digests[identifier]=digest
            elif base_digests.get(identifier)!=digest:raise ValueError('Repeated base certificate differs')
        if page_index==0:packets.append(page.export_action('STOP'))
        for identifier in ids:
            if identifier in identifiers:raise ValueError('Duplicate expansion row across pages')
            preview,geometry=_certificate(page,identifier)
            score=page._score_record({**geometry,'action_id':identifier},page._current_tool)
            if score.get('outcome_scope')!='permitted_nominal_model':raise ValueError('Evaluator reward at policy boundary')
            identifiers.append(identifier);owners.append(page_index);scores.append(score)
            packets.append(page.export_action(identifier))
            rows.append(SpatialAction(identifier,preview.entry_mm,preview.tip_mm,tools[preview.tool_id]))
        page_records.append(inventory);costs['pages_completed']+=1
    if not 1<=len(identifiers)<=153 or len(pages)>5:raise ValueError('Complete union bound changed')
    cavity=work._engine.removed_mask if work.case.post_exposure is None else work._engine.connected_free_mask
    costs['union_observations']+=1
    observation=build_spatial_observation(work.case.spatial_inputs(cavity),[SpatialAction('STOP'),*rows],
        ObservedProcedureState(work.case.access,work._steps,work.max_steps,work._current_tool))
    if work.case._public_target_source is not None:
        observation=replace(observation,public_target_context=work.case._public_target_source.observe(cavity))
    record={'schema':VERSION,'source_state_hash':state,'steps':work._steps,'goal_focus':work._goal_focus,
        'focus_explicitly_cleared':cleared,'active_goal':None if active_goal is None else goal_record(active_goal),
        'query_catalog_digest':work._query_catalog_digest,'goals':goals,'pages':page_records,
        'complete_page_enumeration':True,'base_repeats_checked':len(base_digests)*max(0,len(pages)-1),
        'distinct_certified_actions':len(identifiers)-1,'observation_hash':observation.fingerprint}
    return Catalog(context,work,pages,identifiers,owners,packets,scores,observation,record)


class FrozenFocusedTail:
    """Explicit deployment bridge, never mutates original trained-world admission."""
    def __init__(self, fitted_tail, context, *, max_scores=12, seconds=240., cancelled=None):
        if type(fitted_tail) is not FittedContinuationTail or type(context) is not FocusedContext:
            raise TypeError('Exact frozen tail and explicit focused capability required')
        fitted_tail.require();fitted_tail.bind_context(context.original_context)
        if type(max_scores) is not int or not 1<=max_scores<=24 or not 0<float(seconds)<=3600:
            raise ValueError('Bounded focused score budget required')
        self.fitted,self.context=fitted_tail,context
        self.max_scores,self.seconds,self.cancelled=max_scores,float(seconds),cancelled
        self.started=time.monotonic()
        self.record=freeze_json({'version':VERSION,'context_hash':context.seal,
            'original_tail_report_hash':semantic_digest(fitted_tail.report),
            'original_trained_worlds_hash':semantic_digest(fitted_tail._worlds),
            'head_hash':fitted_tail._head_hash,'feature_path':'unchanged_existing_actor_input',
            'semantics':'raw_tail_excludes_immediate_reward','max_scores':max_scores,'seconds':float(seconds)})
        self._seal=semantic_digest(self.record)
        self.costs={**_costs(),'score_calls_started':0,'score_calls_completed':0,
            'head_forwards_started':0,'head_forwards_completed':0,'candidate_rows_scored':0,'elapsed_seconds':0.}

    def require(self):
        self.fitted.require();self.fitted.bind_context(self.context.original_context)
        if (semantic_digest(self.record)!=self._seal or self.record['context_hash']!=self.context.seal
                or self.record['original_tail_report_hash']!=semantic_digest(self.fitted.report)
                or self.record['original_trained_worlds_hash']!=semantic_digest(self.fitted._worlds)
                or self.record['head_hash']!=self.fitted._head_hash):
            raise ValueError('Frozen focused inference declaration changed')

    def score_catalog(self,catalog):
        started=time.monotonic()
        try:
            self.require()
            if type(catalog) is not Catalog or catalog.context is not self.context:
                raise ValueError('Owned focused full catalog required')
            catalog.require()
            if (self.costs['score_calls_started']>=self.max_scores or time.monotonic()-self.started>self.seconds
                    or (self.cancelled and self.cancelled())):raise RuntimeError('Focused score cap')
            self.costs['score_calls_started']+=1
            features=_feature(self.fitted._guard.policy,catalog.observation,self.costs)
            self.costs['head_forwards_started']+=1
            with torch.no_grad():raw=self.fitted._head(torch.tensor(np.array(features,copy=True))).squeeze(-1)
            self.costs['head_forwards_completed']+=1
            if not torch.isfinite(raw).all():raise FloatingPointError('Nonfinite focused raw tail')
            result=(0.,*(float(v) for v in raw[1:]))
            self.require();catalog.require()
            self.costs['score_calls_completed']+=1;self.costs['candidate_rows_scored']+=len(result)
            return result
        except BaseException as error:
            intact=True;integrity_error=None
            try:self.require()
            except BaseException as check_error:
                intact=False;integrity_error=type(check_error).__name__+': '+str(check_error)[:200]
            error.focused_score_accounting={**self.costs,
                'elapsed_seconds':self.costs['elapsed_seconds']+time.monotonic()-started,
                'frozen_parameters_intact':intact,'integrity_error':integrity_error}
            raise
        finally:self.costs['elapsed_seconds']+=time.monotonic()-started


def choose(scores,raw_tails=None):
    rewards=tuple(float(r['reward']) for r in scores)
    if not rewards or rewards[0]!=0. or not all(math.isfinite(v) for v in rewards):raise ValueError('Finite STOP0 rewards required')
    if raw_tails is not None and (len(raw_tails)!=len(rewards) or raw_tails[0]!=0.
            or not all(math.isfinite(v) for v in raw_tails)):raise ValueError('Finite aligned STOP0 raw tails required')
    eligible=[i for i in range(1,len(rewards)) if rewards[i]>0.]
    values=tuple(r if raw_tails is None else r+max(0.,raw_tails[i]) for i,r in enumerate(rewards))
    if not all(math.isfinite(v) for v in values):raise ValueError('Combined value overflow')
    return (max(eligible,key=lambda i:values[i]) if eligible else 0),values


def _next_active(catalog,index):
    prior=catalog.record['active_goal']
    active=None if prior is None else (prior['tool_id'],tuple(prior['entry_mm']),tuple(prior['tip_mm']))
    if index==0:return active,None
    score=catalog.scores[index];removed={tuple(c) for c in score['removed_indices_native']}
    physical=(score['tool_id'],tuple(score['entry_mm']),tuple(score['tip_mm']))
    for row in catalog.record['goals']:
        goal=row['goal'];key=(goal['tool_id'],tuple(goal['entry_mm']),tuple(goal['tip_mm']))
        direct=physical==key and score['target_removed_mm3']>0
        overlap=sorted(removed & {tuple(c) for c in row['retained_blocker_cells']})
        if direct or overlap:
            return (None if direct else key),{'goal':goal,'direct_goal':direct,'current_blockers_removed':overlap}
    return active,None


def rollout(task,context,prefix_packets,frontier,*,method,scorer=None,seconds=120.,cancelled=None,progress=None):
    """Same complete catalog/positive guard, only GREEDY vs r+clipped frozen tail."""
    protocol.require_frontier(task,prefix_packets,frontier);context.require_task(task)
    if (method not in ('GREEDY','TAIL') or (method=='TAIL')!=(scorer is not None)
            or (scorer is not None and (type(scorer) is not FrozenFocusedTail or scorer.context is not context))
            or not math.isfinite(seconds) or not 0<seconds<=120):raise ValueError('Fixed matched method/cap required')
    _same(prefix_packets and semantic_digest(prefix_packets),context.record['prefix_packets_hash'],'admitted prefix packets')
    started=time.monotonic();work=task.clone();active=None;records=list(prefix_packets)
    account={'version':VERSION,'method':method,'complete':False,'context':context.record,'context_seal':context.seal,
        'controller':CONTROLLER,'initial_focus':None,'fixed_prefix_length':len(records),
        'fixed_prefix_return':float(frontier['total_reward']),'states':[],'model_transition_calls':0,
        'model_transition_completed':0,'costs':{},'positive_immediate_guard':True,'training_updates':0}
    def guard():
        if time.monotonic()-started>seconds:raise TimeoutError('Focused rollout time cap')
        if cancelled and cancelled():raise InterruptedError('Focused rollout cancelled')
    try:
        while not work.terminated:
            guard();catalog=collect_catalog(work,context,active,guard=guard,costs=account['costs'])
            row=thaw_json(catalog.record);account['states'].append(row)
            raw=None
            if scorer is not None and any(r['reward']>0 for r in catalog.scores[1:]):
                raw=scorer.score_catalog(catalog)
            index,values=choose(catalog.scores,raw);active,effect=_next_active(catalog,index)
            row['candidates']=[{'action_id':identifier,'owner_page':catalog.owners[i],
                'immediate_reward':catalog.scores[i]['reward'],'raw_tail':None if raw is None else raw[i],
                'clipped_tail':None if raw is None else max(0.,raw[i]),'ranking_value':values[i],
                'actual_positive':catalog.scores[i]['reward']>0} for i,identifier in enumerate(catalog.identifiers)]
            row['selected']={'action_id':catalog.identifiers[index],'packet':catalog.packets[index],
                'immediate_reward':catalog.scores[index]['reward'],'focus_effect':effect,
                'next_active_goal':None if active is None else goal_record(active)}
            guard();account['model_transition_calls']+=1
            work,packet,transition=catalog.commit(index)
            account['model_transition_completed']+=1;records.append(packet)
            if transition.reward!=catalog.scores[index]['reward']:raise ValueError('Certified reward changed on commit')
            row['selected'].update(result_state_hash=work._engine.state_hash,cumulative_return=work._total_reward)
            if progress:progress('focused_policy_commit',depth=work._steps,cumulative_return=work._total_reward,method=method)
        account.update(complete=True,selected_total_return=work._total_reward,
            selected_net_suffix_return=work._total_reward-float(frontier['total_reward']),
            selected_prefix_length=len(records),selected_equals_explored=True)
        return tuple(records),account
    except BaseException as error:
        error.focused_policy_accounting=account
        raise
    finally:
        account.update(elapsed_seconds=time.monotonic()-started,explored_action_records=records,
            explored_history=work.metrics()['history'],explored_actions=[p['action_id'] for p in records],
            scorer_costs=None if scorer is None else dict(scorer.costs))
