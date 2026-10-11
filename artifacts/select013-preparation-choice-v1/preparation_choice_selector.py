"""Persistent public-goal corridors, current native blockers, complete-prefix objective.

A first-failure sidecar is a bounded current obstruction witness, never a complete
corridor certificate. All preparation rewards count. Only current catalog actions
can commit; a diagnostic-only ray cannot become an action. No private evaluator.
"""
import math
import time
from resectionlab.core import semantic_digest, thaw_json, freeze_json
from resectionlab.observed_search import ObservedSearchLimit
from resectionlab.focused_paged_task import FocusedPagedObstructionTask
from resectionlab.native_proposals import OBSTRUCTION_CELLS_PER_PREVIEW

VERSION='sealed_preparation_choice_continuation_v1'
MAX_DEPENDENCY_PAIRS=32

def physical_key(ray):
    return (ray.tool_id, tuple(float(v) for v in ray.entry_mm), tuple(float(v) for v in ray.tip_mm))

def goal_record(key):
    return {'tool_id':key[0],'entry_mm':list(key[1]),'tip_mm':list(key[2])}

def complete_prefix(actions, best_length, horizon):
    chosen=tuple(actions[:best_length])
    return chosen if len(chosen)==horizon else (*chosen,'STOP')

def _current_prerequisites(work, blockers, state, row, report, guard):
    """One extra dependency edge from exact current emitted prep rays only.

    This neither constructs a new candidate nor commits a counterfactual cut.
    Failed first-shaft intervals are sampled dependency evidence, not complete
    corridor sets or a guarantee that removing one cell makes a ray feasible.
    """
    if work._engine.state_hash != state:
        raise ValueError('Dependency source state changed')
    candidates=[];seen=set()
    for ray in work._page_catalog.proposals:
        key=physical_key(ray)
        if tuple(ray.voxel) in blockers and key not in seen:
            candidates.append(ray);seen.add(key)
    record={'source_state_hash':state,'dependency_depth':2,
        'trigger':'established_active_goal_has_no_current_direct_legal_progress',
        'active_goal':goal_record(row['_active_key']),
        'active_retained_blockers':[list(c) for c in sorted(blockers)],
        'emitted_centered_pair_count':len(candidates),'pair_limit':MAX_DEPENDENCY_PAIRS,
        'pair_cap_reached':False,'probes':[],'selected_effect':None,
        'scope':'current emitted prep first-shaft-failure samples only; no full reachability or unlock guarantee'}
    row['dependency_diagnostics']=record
    if len(candidates)>MAX_DEPENDENCY_PAIRS:
        record['pair_cap_reached']=True;report['call_cap_reached']=True
        raise ValueError('Emitted blocker-centered pair count exceeds declared dependency bound')
    prerequisite_cells=set()
    # Independent certificate registry: these previews cannot evict page actions.
    diagnostic=work._engine.clone()
    try:
        for ray in candidates:
            guard();key=physical_key(ray)
            report['diagnostic_preview_calls']+=1;report['dependency_preview_calls']+=1
            outcome=diagnostic.preview_stroke(ray.tool_id,ray.tip_mm,entry_mm=ray.entry_mm,
                obstruction_diagnostics=True,obstruction_cell_limit=OBSTRUCTION_CELLS_PER_PREVIEW)
            evidence=None;cells=set()
            if outcome.obstruction_diagnostic is not None:
                evidence=thaw_json(outcome.obstruction_diagnostic)
                unsigned={k:v for k,v in evidence.items() if k!='fingerprint'}
                if (semantic_digest(unsigned)!=evidence['fingerprint'] or evidence['source_state_hash']!=state
                    or evidence['source_hash']!=work._config.source_hash
                    or evidence['decision_model_hash']!=work._config.fingerprint
                    or tuple(evidence['entry_mm'])!=key[1] or tuple(evidence['requested_tip_mm'])!=key[2]
                    or evidence['tool']['tool_id']!=key[0]
                    or evidence['reason']!='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE'
                    or outcome.reason!=evidence['reason'] or outcome.feasible):
                    raise ValueError('Current exact-preparation sidecar binding failed')
                cells={tuple(c) for c in evidence['blocked_indices_native']}
                if len(cells)>OBSTRUCTION_CELLS_PER_PREVIEW:
                    raise ValueError('Dependency sidecar exceeds retained-cell bound')
                prerequisite_cells.update(cells)
            elif outcome.reason=='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE':
                raise ValueError('Missing current preparation obstruction sidecar')
            record['probes'].append({**goal_record(key),'current_catalog_action_id':ray.proposal_id,
                'centered_active_blocker':list(ray.voxel),'feasible':bool(outcome.feasible),'reason':outcome.reason,
                'retained_prerequisite_cells':[list(c) for c in sorted(cells)],
                'sidecar':evidence})
    finally:
        del diagnostic
    if work._engine.state_hash!=state:
        raise ValueError('Dependency previews changed live state')
    record['retained_prerequisite_union']=[list(c) for c in sorted(prerequisite_cells)]
    return prerequisite_cells,record

PROTOCOL_SCHEMA='sealed-preparation-choice-v1'
BASELINE_SELECTOR_SHA256='e73b7bca632ed37b4afe3b513b494465910b3f90160cad22851b1a9f2bac1448'
INTERPRETATION='simulator_trace_causal_intervention_not_deployable_policy'

def _geometry_key(record):
    if not isinstance(record,dict) or set(record)!={'tool_id','entry_mm','tip_mm'}:
        raise ValueError('Explicit physical tool/entry/tip required')
    if not isinstance(record['tool_id'],str) or not record['tool_id']:
        raise ValueError('Explicit registered tool identity required')
    points=[]
    for field in ('entry_mm','tip_mm'):
        point=record[field]
        if (not isinstance(point,(tuple,list)) or len(point)!=3
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in point)):
            raise ValueError('Finite physical point required')
        points.append(tuple(float(v) for v in point))
    return record['tool_id'],*points

def seal_protocol(unsigned):
    """Seal a predeclared simulator-trace-derived intervention, not task authority."""
    record=thaw_json(unsigned)
    expected={'schema','baseline_selector_sha256','source_hash','decision_model_hash',
        'native_decision_model_hash','max_steps','prefix','decision','interpretation'}
    if not isinstance(record,dict) or set(record)!=expected:
        raise ValueError('Exact intervention protocol fields required')
    if (record['schema']!=PROTOCOL_SCHEMA or record['baseline_selector_sha256']!=BASELINE_SELECTOR_SHA256
            or record['interpretation']!=INTERPRETATION or type(record['max_steps']) is not int
            or not 5<=record['max_steps']<=48):
        raise ValueError('Fixed decision5 intervention and unchanged baseline policy required')
    for key in ('source_hash','decision_model_hash','native_decision_model_hash'):
        if not isinstance(record[key],str) or not record[key]:raise ValueError('Explicit source/model bindings required')
    prefix=record['prefix']
    if not isinstance(prefix,list) or len(prefix)!=4:raise ValueError('Exactly four ordinary prefix decisions required')
    for index,row in enumerate(prefix):
        if not isinstance(row,dict) or set(row)!={'physical_action','source_state_hash','result_state_hash'}:
            raise ValueError('Exact prefix physical action and pre/post states required')
        _geometry_key(row['physical_action'])
        for key in ('source_state_hash','result_state_hash'):
            if not isinstance(row[key],str) or not row[key]:raise ValueError('Explicit prefix states required')
        if index and row['source_state_hash']!=prefix[index-1]['result_state_hash']:
            raise ValueError('Prefix states must form one continuous history')
    decision=record['decision']
    if (not isinstance(decision,dict) or set(decision)!=
            {'index','source_state_hash','current_tool','active_goal','physical_action'}
            or type(decision['index']) is not int or decision['index']!=5
            or decision['source_state_hash']!=prefix[-1]['result_state_hash']
            or decision['current_tool']!=prefix[-1]['physical_action']['tool_id']):
        raise ValueError('Decision5 must join the exact prefix state and tool')
    _geometry_key(decision['active_goal']);_geometry_key(decision['physical_action'])
    return freeze_json({**record,'seal':semantic_digest(record)})

def _validate_protocol(task,protocol):
    try:
        record=thaw_json(protocol);unsigned={k:v for k,v in record.items() if k!='seal'}
        if thaw_json(seal_protocol(unsigned))!=record:raise ValueError('Intervention seal mismatch')
        if (record['source_hash']!=task.case.source_hash or record['decision_model_hash']!=task.decision_model_hash
                or record['native_decision_model_hash']!=task._config.fingerprint
                or record['max_steps']!=task.max_steps
                or record['prefix'][0]['source_state_hash']!=task._engine.state_hash):
            raise ValueError('Intervention protocol source/model/initial state differs')
        return record
    except (TypeError,KeyError,AttributeError) as error:
        raise ValueError('Malformed sealed intervention protocol') from error

def _current_choice(work,active,goals,legal,winner,protocol,row):
    """Select only an already freshly certified current catalog preparation."""
    decision=protocol['decision'];key=_geometry_key(decision['physical_action'])
    goal=_geometry_key(decision['active_goal'])
    if (work._steps!=4 or work._engine.state_hash!=decision['source_state_hash']
            or work._current_tool!=decision['current_tool'] or active!=goal
            or not goals or goals[0][0]!=active or work._goal_focus is None):
        raise ValueError('Decision5 common state/current tool/active focused goal differs')
    matches=[]
    for page,identifier,removed,reward,raw_order in legal:
        if page._engine.state_hash!=work._engine.state_hash:
            raise ValueError('Choice catalog contains stale page state')
        certificate=page._inventory[identifier];record=certificate.to_history_record()
        physical={k:record[k] for k in ('tool_id','entry_mm','tip_mm')}
        # Native records may contain NumPy scalars; physical comparison is exact.
        current=(record['tool_id'],tuple(float(v) for v in record['entry_mm']),tuple(float(v) for v in record['tip_mm']))
        if current==key:matches.append((page,identifier,removed,reward,physical))
    if len(matches)!=1:
        raise ValueError('Decision5 physical choice is unavailable or not currently native-certified')
    page,identifier,removed,reward,physical=matches[0]
    overlap=removed & goals[0][1]
    if not overlap or key==goal:
        raise ValueError('Decision5 preparation must actually remove a current active-goal first blocker')
    ordinary=None
    if winner is not None:
        prior=winner[0]._inventory[winner[1]].to_history_record()
        ordinary={k:prior[k] for k in ('tool_id','entry_mm','tip_mm')}
    row['choice_intervention']={'protocol_seal':protocol['seal'],'decision':5,
        'interpretation':INTERPRETATION,'source_state_hash':work._engine.state_hash,
        'current_tool':work._current_tool,'active_goal':decision['active_goal'],
        'ordinary_policy_choice':ordinary,'fixed_physical_choice':physical,
        'current_catalog_action_id':identifier,'query_catalog_digest':page._query_catalog_digest,
        'active_retained_first_blockers':[list(c) for c in sorted(goals[0][1])],
        'actual_removed_current_blockers':[list(c) for c in sorted(overlap)],
        'nominal_reward':reward,'fresh_current_certificate':True,
        'scope':'one explicit simulator-trace-derived action substitution; future outcomes unknown'}
    return page,identifier,active,len(overlap),False,reward

def select_corridor(task, *, seconds, intervention_protocol, cancelled=None, progress=None):
    if (type(task) is not FocusedPagedObstructionTask or task._steps!=0 or not 1<=task.max_steps<=48
        or task.case.proposal_config.obstruction_selection_rule!='target_ray_coverage_v3'
        or not math.isfinite(seconds) or seconds<=0):
        raise ValueError('Fresh bounded focused v3 paged task required')
    protocol=_validate_protocol(task,intervention_protocol)
    started=time.monotonic();work=task.planning_clone();actions=[];active=None
    best_length=0;best_return=0.;records=[];best_stop=None;report={'version':VERSION,'complete':False,'states':[],
        'model_transition_calls':0,'diagnostic_preview_calls':0,'goal_diagnostic_preview_calls':0,
        'dependency_preview_calls':0,'dependency_pair_limit_per_state':MAX_DEPENDENCY_PAIRS,'time_cap_reached':False,
        'call_cap_reached':False,'estimated_incremental_return':0.,'selected_prefix_length':0,
        'goal_scope':'public nominal-positive endpoint; current bounded first-shaft-failure cells only',
        'objective':'highest cumulative unchanged nominal reward complete prefix or STOP0',
        'all_preparation_costs_charged':True,'private_reference_used':False,
        'intervention_protocol':protocol,'prefix_verified_count':0,'intervention_applied':False}
    def guard():
        if time.monotonic()-started>seconds:
            report['time_cap_reached']=True
            raise TimeoutError('Persistent-corridor selector time cap')
        if cancelled is not None and cancelled():raise InterruptedError('Corridor cancelled')
    try:
        while not work.terminated:
            guard()
            # Focus is a state-local public query. A removed/disappeared goal is
            # explicitly cleared; no stale physical ray gains catalog authority.
            cleared=None
            if active is not None:
                proposer=work.case._nominal_proposer
                current_base=proposer.propose(work._engine,cancelled=cancelled)
                available={physical_key(ray) for ray in current_base.proposals
                    if float(work.case.nominal_target[ray.voxel])>work.case.proposal_config.nominal_min_membership}
                if active not in available:
                    cleared=goal_record(active);active=None
            work.set_focus(None if active is None else work.focus_record(active))
            work._prepare_base();state=work._engine.state_hash
            catalog={physical_key(r):r.proposal_id for r in work._page_catalog.proposals}
            keys=[]
            if active is not None:keys.append(active)
            for ray in work._base_batch.proposals:
                if float(work.case.nominal_target[ray.voxel])>work.case.proposal_config.nominal_min_membership:
                    key=physical_key(ray)
                    if key not in keys:keys.append(key)
            omitted=max(0,len(keys)-120);keys=keys[:120]
            row={'depth':work._steps+1,'source_state_hash':state,'goals_omitted_by_diagnostic_cap':omitted,
                'goals':[],'pages':[],'selected':None,'complete_page_enumeration':False,
                'goal_focus':thaw_json(work._goal_focus),'focus_explicitly_cleared':cleared}
            report['states'].append(row)
            # This isolated clone cannot evict the live page certificates.
            diagnostic=work._engine.clone();goals=[]
            for key in keys:
                guard();report['diagnostic_preview_calls']+=1;report['goal_diagnostic_preview_calls']+=1
                outcome=diagnostic.preview_stroke(key[0],key[2],entry_mm=key[1],
                    obstruction_diagnostics=True,obstruction_cell_limit=OBSTRUCTION_CELLS_PER_PREVIEW)
                cells=set();evidence=None
                if outcome.obstruction_diagnostic is not None:
                    evidence=thaw_json(outcome.obstruction_diagnostic)
                    unsigned={k:v for k,v in evidence.items() if k!='fingerprint'}
                    if (semantic_digest(unsigned)!=evidence['fingerprint'] or evidence['source_state_hash']!=state
                        or evidence['source_hash']!=work._config.source_hash
                        or evidence['decision_model_hash']!=work._config.fingerprint
                        or tuple(evidence['entry_mm'])!=key[1] or tuple(evidence['requested_tip_mm'])!=key[2]
                        or evidence['tool']['tool_id']!=key[0]):raise ValueError('Current exact-goal sidecar binding failed')
                    cells={tuple(c) for c in evidence['blocked_indices_native']}
                record={**goal_record(key),'current_catalog_action_id':catalog.get(key),'feasible':bool(outcome.feasible),
                    'reason':outcome.reason,'retained_blocker_cells':[list(c) for c in sorted(cells)],
                    'first_failure_set_complete':None if evidence is None else evidence['complete_first_failure_set'],
                    'sidecar_fingerprint':None if evidence is None else evidence['fingerprint'],
                    'eligible_legal_preparations':0,'direct_legal_goal_available':False}
                row['goals'].append(record);goals.append((key,cells,record))
            del diagnostic
            winner=None;winner_rank=None;order=0;stop_page=None;legal=[];dependency_effect=None
            for page,ids in work.planning_action_pages():
                guard()
                if stop_page is None:stop_page=page
                inventory=page.candidate_inventory()
                row['pages'].append({'page_index':page._active_page,'legal_actions':len(ids),
                    'base_emitted_count':inventory['base_emitted_count'],
                    'additional_emitted_count':inventory['additional_emitted_count']})
                if page._active_page==0:row['obstruction_accounting']=inventory['obstruction_accounting']
                for identifier in ids:
                    guard();certificate=page._inventory[identifier]
                    geometry=certificate.to_history_record()
                    score=page._score_record({**geometry,'action_id':identifier},page._current_tool)
                    removed={tuple(c) for c in geometry['removed_indices_native']}
                    legal.append((page,identifier,removed,score['reward'],order))
                    for priority,(key,cells,record) in enumerate(goals):
                        direct=(catalog.get(key)==identifier and score['target_removed_mm3']>0)
                        overlap=len(removed & cells)
                        if not direct and not overlap:continue
                        record['eligible_legal_preparations']+=int(bool(overlap))
                        record['direct_legal_goal_available']|=direct
                        rank=(priority,0 if direct else 1,-overlap,-score['reward'],order)
                        if winner_rank is None or rank<winner_rank:
                            winner_rank=rank;winner=(page,identifier,key,overlap,direct,score['reward'])
                    order+=1
            if len(records)==best_length:
                # Capture STOP from this already paid current catalog, before
                # commit; never precompute an unfocused successor inventory.
                best_stop=stop_page.export_action('STOP')
            row['complete_page_enumeration']=True
            for r in row['goals']:
                r['current_status']='progress_available' if (r['direct_legal_goal_available'] or r['eligible_legal_preparations']) else 'no_current_legal_blocker_removal_or_goal'
            # Existing direct progress for the active goal retains absolute priority.
            # Only an established active corridor can trigger this one new edge.
            if (active is not None and goals and goals[0][0]==active
                    and not goals[0][2]['direct_legal_goal_available']
                    and not goals[0][2]['eligible_legal_preparations']):
                row['_active_key']=active
                try:prerequisites,dependency=_current_prerequisites(work,goals[0][1],state,row,report,guard)
                finally:row.pop('_active_key',None)
                dependency_rank=None
                for candidate_page,identifier,removed,reward,raw_order in legal:
                    overlap=removed & prerequisites
                    if not overlap:continue
                    rank=(-len(overlap),-reward,raw_order)
                    if dependency_rank is None or rank<dependency_rank:
                        dependency_rank=rank
                        winner=(candidate_page,identifier,active,0,False,reward)
                        dependency_effect={'action_id':identifier,
                            'actual_removed_cells':[list(c) for c in sorted(removed)],
                            'sampled_prerequisite_cells_removed':[list(c) for c in sorted(overlap)],
                            'prep_ray_links':[{'prep_action_id':probe['current_catalog_action_id'],
                                'prep_geometry':{k:probe[k] for k in ('tool_id','entry_mm','tip_mm')},
                                'removed_prerequisites':[list(c) for c in sorted(removed & {tuple(v) for v in probe['retained_prerequisite_cells']})]}
                                for probe in dependency['probes']
                                if removed & {tuple(v) for v in probe['retained_prerequisite_cells']}],
                            'scope':'removed actual sampled prerequisites; prep feasibility must be refreshed after commit'}
                dependency['selected_effect']=dependency_effect
            guard()
            expected_prefix=None
            if work._steps<4:
                expected_prefix=protocol['prefix'][work._steps]
                if work._engine.state_hash!=expected_prefix['source_state_hash'] or winner is None:
                    raise ValueError('Ordinary prefix state or motion differs from declared common history')
                predicted=winner[0]._inventory[winner[1]].to_history_record()
                actual_key=(predicted['tool_id'],tuple(float(v) for v in predicted['entry_mm']),tuple(float(v) for v in predicted['tip_mm']))
                if actual_key!=_geometry_key(expected_prefix['physical_action']):
                    raise ValueError('Ordinary prefix physical action differs; no forced prefix bypass')
            elif work._steps==4:
                winner=_current_choice(work,active,goals,legal,winner,protocol,row)
                dependency_effect=None
                if 'dependency_diagnostics' in row:
                    row['dependency_diagnostics']['selected_effect']=None
            if winner is None:
                # Still a real planning STOP, with paid discovery and a complete trace.
                work=stop_page;before=work._steps;report['model_transition_calls']+=1
                packet=work.export_action('STOP')
                try:work.advance_record(packet)
                finally:
                    if work._steps>before:actions.append('STOP');records.append(packet)
                row['selected']={'action_id':'STOP','reason':'no_current_goal_progress'}
            else:
                page,identifier,key,overlap,direct,reward=winner
                # Mapping is current state, exact physical geometry, and current certificate.
                if page._engine.state_hash!=state or identifier not in page._inventory:raise ValueError('Stale selected certificate')
                selected_record=page._inventory[identifier].to_history_record()
                work=page;before=work._steps;report['model_transition_calls']+=1
                packet=work.export_action(identifier)
                try:work.advance_record(packet)
                finally:
                    if work._steps>before:actions.append(identifier);records.append(packet)
                if expected_prefix is not None:
                    if work._engine.state_hash!=expected_prefix['result_state_hash']:
                        raise ValueError('Ordinary prefix committed state differs')
                    report['prefix_verified_count']+=1
                if before==4:report['intervention_applied']=True
                active=None if direct else key
                row['selected']={'action_id':identifier,'goal':goal_record(key),'direct_goal':direct,
                    'current_blockers_removed':overlap,'nominal_reward':reward,
                    'action_geometry':{k:selected_record[k] for k in ('tool_id','entry_mm','tip_mm')},
                    'actual_removed_cells':selected_record['removed_indices_native'],
                    'result_state_hash':work._engine.state_hash,'cumulative_return':work._total_reward,
                    'progress_kind':'prerequisite' if dependency_effect is not None else ('goal' if direct else 'direct_blocker'),
                    'sampled_prerequisites_removed':0 if dependency_effect is None else len(dependency_effect['sampled_prerequisite_cells_removed'])}
                if work._total_reward>best_return:
                    best_return=work._total_reward;best_length=len(actions)
                if progress:progress('corridor_commit',depth=work._steps,direct_goal=direct,cumulative_return=work._total_reward)
            # Only immutable scalar records survive; live clones do not accumulate.
            del winner,stop_page,legal
        if report['prefix_verified_count']!=4 or not report['intervention_applied']:
            raise ValueError('Declared decision5 intervention did not complete')
        report.update(complete=True,estimated_incremental_return=best_return,selected_prefix_length=best_length,
            explored_actions=actions,explored_action_records=[thaw_json(p) for p in records],explored_total_return=work._total_reward,
            explored_history=work.metrics()['history'],terminal='STOP' if actions[-1]=='STOP' else 'HORIZON',
            wall_seconds=time.monotonic()-started)
        chosen=tuple(records[:best_length])
        if best_length<task.max_steps:
            if best_stop is None or best_stop['steps_taken']!=best_length:
                raise RuntimeError('Complete best-prefix STOP packet was not captured')
            chosen=(*chosen,best_stop)
        report['selected_action_records']=[thaw_json(p) for p in chosen]
        return chosen,report
    except BaseException as error:
        report.update(wall_seconds=time.monotonic()-started,explored_actions=list(actions),
            explored_action_records=[thaw_json(p) for p in records],
            estimated_incremental_return=best_return,selected_prefix_length=best_length,
            explored_total_return=work._total_reward,explored_history=work.metrics()['history'],
            failure={'type':type(error).__name__,'message':str(error)})
        raise ObservedSearchLimit('Corridor incomplete: '+str(error),accounting=report,
            best_sequence=()) from error

def check_exploration_geometry(task, history, *, cancelled=None):
    """Independent geometry only; never accept a planning trace as an episode."""
    from dataclasses import asdict
    from types import SimpleNamespace
    from resectionlab.evaluation import independent_check_native_history
    source=SimpleNamespace(mri=task.case.structural_intensity,affine=task.case._native_affine_ras_mm,
        frame='RAS+',semantic_hash=task.case.source_hash)
    audit=independent_check_native_history(source,task.case.tools,history,
        tissue_mask=task.case.observed_support,access=task.case.access,
        hard_exclusion=task._config.hard_exclusion,geometry_frame='RAS+',cancelled=cancelled,
        interaction_domain=task._config.interaction_domain,post_exposure=task._config.post_exposure,
        distance_backend='batch',distance_batch_size=256)
    return {'schema':'independent-planning-exploration-geometry-v1','geometry':asdict(audit),
        'feasible':bool(audit.feasible),'source_hash':task.case.source_hash,
        'decision_model_hash':task.decision_model_hash,'history_sha256':semantic_digest(history),
        'scope':'geometry of saved nominal exploration, not an accepted authoritative episode or policy outcome'}
