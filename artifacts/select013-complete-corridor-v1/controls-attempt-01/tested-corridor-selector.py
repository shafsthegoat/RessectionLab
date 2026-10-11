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

VERSION='public_current_corridor_v1'

def physical_key(ray):
    return (ray.tool_id, tuple(float(v) for v in ray.entry_mm), tuple(float(v) for v in ray.tip_mm))

def goal_record(key):
    return {'tool_id':key[0],'entry_mm':list(key[1]),'tip_mm':list(key[2])}

def complete_prefix(actions, best_length, horizon):
    chosen=tuple(actions[:best_length])
    return chosen if len(chosen)==horizon else (*chosen,'STOP')

def select_corridor(task, *, seconds, cancelled=None, progress=None):
    if (type(task) is not PagedObstructionTask or task._steps!=0 or not 1<=task.max_steps<=48
        or task.case.proposal_config.obstruction_selection_rule!='target_ray_coverage_v3'
        or not math.isfinite(seconds) or seconds<=0):
        raise ValueError('Fresh bounded v3 paged task required')
    started=time.monotonic();work=task.planning_clone();actions=[];active=None
    best_length=0;best_return=0.;report={'version':VERSION,'complete':False,'states':[],
        'model_transition_calls':0,'diagnostic_preview_calls':0,'time_cap_reached':False,
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
                guard();report['diagnostic_preview_calls']+=1
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
            winner=None;winner_rank=None;order=0;stop_page=None
            for page,ids in work.planning_action_pages():
                guard()
                if stop_page is None:stop_page=page
                inventory=page.candidate_inventory()
                row['pages'].append({'page_index':page._active_page,'legal_actions':len(ids),
                    'base_emitted_count':inventory['base_emitted_count'],
                    'additional_emitted_count':inventory['additional_emitted_count'],
                    'obstruction_accounting':inventory['obstruction_accounting']})
                for identifier in ids:
                    guard();certificate=page._inventory[identifier]
                    geometry=certificate.to_history_record()
                    score=page._score_record({**geometry,'action_id':identifier},page._current_tool)
                    removed={tuple(c) for c in geometry['removed_indices_native']}
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
            guard()
            if winner is None:
                # Still a real planning STOP, with paid discovery and a complete trace.
                stop_page.advance_planning('STOP');work=stop_page;actions.append('STOP')
                report['model_transition_calls']+=1;row['selected']={'action_id':'STOP','reason':'no_current_goal_progress'}
            else:
                page,identifier,key,overlap,direct,reward=winner
                # Mapping is current state, exact physical geometry, and current certificate.
                if page._engine.state_hash!=state or identifier not in page._inventory:raise ValueError('Stale selected certificate')
                page.advance_planning(identifier);work=page;actions.append(identifier)
                report['model_transition_calls']+=1;active=None if direct else key
                row['selected']={'action_id':identifier,'goal':goal_record(key),'direct_goal':direct,
                    'current_blockers_removed':overlap,'nominal_reward':reward,
                    'result_state_hash':work._engine.state_hash,'cumulative_return':work._total_reward}
                if work._total_reward>best_return:
                    best_return=work._total_reward;best_length=len(actions)
                if progress:progress('corridor_commit',depth=work._steps,direct_goal=direct,cumulative_return=work._total_reward)
            # Only immutable scalar records survive; live clones do not accumulate.
            del winner,stop_page
        report.update(complete=True,estimated_incremental_return=best_return,selected_prefix_length=best_length,
            explored_actions=actions,explored_total_return=work._total_reward,
            explored_history=work.metrics()['history'],terminal='STOP' if actions[-1]=='STOP' else 'HORIZON',
            wall_seconds=time.monotonic()-started)
        return complete_prefix(actions,best_length,task.max_steps),report
    except BaseException as error:
        report.update(wall_seconds=time.monotonic()-started,explored_actions=list(actions),
            explored_total_return=work._total_reward,explored_history=work.metrics()['history'],
            failure={'type':type(error).__name__,'message':str(error)})
        raise ObservedSearchLimit('Corridor incomplete: '+str(error),accounting=report,
            best_sequence=complete_prefix(actions,best_length,task.max_steps)) from error
