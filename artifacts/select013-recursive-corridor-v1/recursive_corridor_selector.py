"""Persistent public-goal corridors, current native blockers, complete-prefix objective.

A first-failure sidecar is a bounded current obstruction witness, never a complete
corridor certificate. All preparation rewards count. Only current catalog actions
can commit; a diagnostic-only ray cannot become an action. No private evaluator.
"""
import math
import time
from resectionlab.core import semantic_digest, thaw_json
from resectionlab.observed_search import ObservedSearchLimit
from resectionlab.paged_obstruction import PagedObstructionTask
from resectionlab.native_proposals import OBSTRUCTION_CELLS_PER_PREVIEW

VERSION='public_current_recursive_corridor_v1'
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

def select_corridor(task, *, seconds, cancelled=None, progress=None):
    if (type(task) is not PagedObstructionTask or task._steps!=0 or not 1<=task.max_steps<=48
        or task.case.proposal_config.obstruction_selection_rule!='target_ray_coverage_v3'
        or not math.isfinite(seconds) or seconds<=0):
        raise ValueError('Fresh bounded v3 paged task required')
    started=time.monotonic();work=task.planning_clone();actions=[];active=None
    best_length=0;best_return=0.;report={'version':VERSION,'complete':False,'states':[],
        'model_transition_calls':0,'diagnostic_preview_calls':0,'goal_diagnostic_preview_calls':0,
        'dependency_preview_calls':0,'dependency_pair_limit_per_state':MAX_DEPENDENCY_PAIRS,'time_cap_reached':False,
        'call_cap_reached':False,'estimated_incremental_return':0.,'selected_prefix_length':0,
        'goal_scope':'public nominal-positive endpoint; current bounded first-shaft-failure cells only',
        'objective':'highest cumulative unchanged nominal reward complete prefix or STOP0',
        'all_preparation_costs_charged':True,'private_reference_used':False}
    def guard():
        if time.monotonic()-started>seconds:
            report['time_cap_reached']=True
            raise TimeoutError('Persistent-corridor selector time cap')
        if cancelled is not None and cancelled():raise InterruptedError('Corridor cancelled')
    try:
        while not work.terminated:
            guard();work._prepare_base();state=work._engine.state_hash
            catalog={physical_key(r):r.proposal_id for r in work._page_catalog.proposals}
            keys=[]
            if active is not None:keys.append(active)
            for ray in work._base_batch.proposals:
                if float(work.case.nominal_target[ray.voxel])>work.case.proposal_config.nominal_min_membership:
                    key=physical_key(ray)
                    if key not in keys:keys.append(key)
            omitted=max(0,len(keys)-120);keys=keys[:120]
            row={'depth':work._steps+1,'source_state_hash':state,'goals_omitted_by_diagnostic_cap':omitted,
                'goals':[],'pages':[],'selected':None,'complete_page_enumeration':False}
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
            if winner is None:
                # Still a real planning STOP, with paid discovery and a complete trace.
                work=stop_page;before=work._steps;report['model_transition_calls']+=1
                try:work.advance_planning('STOP')
                finally:
                    if work._steps>before:actions.append('STOP')
                row['selected']={'action_id':'STOP','reason':'no_current_goal_progress'}
            else:
                page,identifier,key,overlap,direct,reward=winner
                # Mapping is current state, exact physical geometry, and current certificate.
                if page._engine.state_hash!=state or identifier not in page._inventory:raise ValueError('Stale selected certificate')
                selected_record=page._inventory[identifier].to_history_record()
                work=page;before=work._steps;report['model_transition_calls']+=1
                try:work.advance_planning(identifier)
                finally:
                    if work._steps>before:actions.append(identifier)
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
        report.update(complete=True,estimated_incremental_return=best_return,selected_prefix_length=best_length,
            explored_actions=actions,explored_total_return=work._total_reward,
            explored_history=work.metrics()['history'],terminal='STOP' if actions[-1]=='STOP' else 'HORIZON',
            wall_seconds=time.monotonic()-started)
        return complete_prefix(actions,best_length,task.max_steps),report
    except BaseException as error:
        report.update(wall_seconds=time.monotonic()-started,explored_actions=list(actions),
            estimated_incremental_return=best_return,selected_prefix_length=best_length,
            explored_total_return=work._total_reward,explored_history=work.metrics()['history'],
            failure={'type':type(error).__name__,'message':str(error)})
        raise ObservedSearchLimit('Corridor incomplete: '+str(error),accounting=report,
            best_sequence=complete_prefix(actions,best_length,task.max_steps)) from error

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
