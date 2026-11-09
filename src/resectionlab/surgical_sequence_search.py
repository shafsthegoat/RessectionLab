"""Finite native lookahead over the existing checked CUSA work/exit skills.

Geometry only proposes work. Every retained node has a separately checked exit,
and every speculative preview, commit and withdrawal shares the caller's meter.
Preparation occupies a reserved beam slot; immediate progress cannot eliminate
all cavity-changing prefixes before their continuation has been evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass

from .surgical_staged_skills import checked_cusa_motion
from .surgical_staged_exit import find_checked_exit


@dataclass
class SequenceNode:
    work: object
    exited: object
    path: tuple
    target_mm3: float
    non_target_mm3: float
    elapsed_seconds: float


def _key(node):
    return (-node.target_mm3, node.non_target_mm3, node.elapsed_seconds,
            tuple(row['tileId'] for row in node.path))


def _dominates(a, b):
    av = (-a.target_mm3, a.non_target_mm3, a.elapsed_seconds)
    bv = (-b.target_mm3, b.non_target_mm3, b.elapsed_seconds)
    return all(x <= y for x, y in zip(av, bv)) and any(x < y for x, y in zip(av, bv))


def _budget_reason(reason):
    return any(token in str(reason) for token in (
        'TransitionBudgetExceeded', 'TransitionReserveReached',
        'NATIVE_ACTION_BUDGET', 'TimeoutError'))


def checked_sequence_search(engine, route, *, propose, motion, hand, radius,
                            limits, meter, horizon=2, beam_width=3,
                            max_expansions=24, max_contact_replans=8, check=lambda: None,
                            can_expand=lambda: True, phase='staged.lookahead'):
    """Return a checked positive sequence, attempts, and explicit search evidence.

    ``horizon=1`` implements immediate-removal selection for matched fixtures.
    Depth is 1..3, beam width 2..4, at most six proposals/node and 24 expansions.
    Full native work states continue; exit probes are verification alternatives,
    not an action prefix committed between consecutive work skills.
    """
    if (type(horizon) is not int or not 1 <= horizon <= 3 or
            type(beam_width) is not int or not 2 <= beam_width <= 4 or
            type(max_expansions) is not int or not 1 <= max_expansions <= 24 or
            type(max_contact_replans) is not int or not 0 <= max_contact_replans <= 8):
        raise ValueError('Keep sequence horizon 1..3, beam 2..4 and expansions 1..24')
    baseline = engine.metrics()
    frontier = [SequenceNode(engine, None, (), 0., 0., 0.)]
    completed, attempts = [], []
    expansions, deepest, budget_stopped = 0, 0, False
    first_calls = meter.calls
    for depth in range(horizon):
        children = []
        for parent in frontier:
            check()
            proposals = propose(parent.work)[:6]
            for tile in proposals:
                check()
                if expansions >= max_expansions or not can_expand():
                    budget_stopped = True
                    break
                expansions += 1
                before = parent.work.metrics()
                work, receipt = checked_cusa_motion(parent.work, motion(parent.work, tile),
                    limits=limits, meter=meter, check=check,
                    max_contact_replans=max_contact_replans,
                    phase=f'{phase}.depth{depth}.{expansions}.work')
                row = {'tileId': tile['id'], 'radiusMm': radius, 'work': receipt,
                    'currentContacts': tile.get('currentContacts'),
                    'contactStateId': parent.work.state_id,
                    'workStartPosition': len(parent.work.actions),
                    'orientationMode': tile['orientationMode'],
                    'status': 'work_refused', 'lookaheadDepth': depth+1}
                attempts.append(row)
                if receipt['status'] != 'accepted':
                    if _budget_reason(receipt.get('reason')):
                        budget_stopped = True
                        break
                    continue
                delta = work.metrics()['targetRemovedMm3']-before['targetRemovedMm3']
                changed_cavity = set(work.state['removed']) != set(parent.work.state['removed'])
                if delta <= 0 and (horizon == 1 or not changed_cavity):
                    row['status'] = 'no_new_target' if horizon == 1 else 'no_cavity_or_target_progress'
                    continue
                exited, exit_receipt = find_checked_exit(work, route, hand=hand,
                    limits=limits, meter=meter, check=check)
                row['exit'] = exit_receipt
                if exit_receipt['status'] != 'accepted':
                    row['status'] = 'checked_exit_refused'
                    if _budget_reason(exit_receipt.get('reason')):
                        budget_stopped = True
                        break
                    continue
                values = exited.metrics()
                row.update(status='complete_work_and_exit_checked',
                    targetIncrementMm3=delta,
                    nonTargetIncludingExitMm3=values['nonTargetRemovedMm3']-before['nonTargetRemovedMm3'],
                    preparationOnly=delta <= 0,
                    workStateId=work.state_id, workEndPosition=len(work.actions),
                    checkedExitStateId=exited.state_id,
                    checkedExitActions=len(exited.actions)-len(work.actions))
                node = SequenceNode(work, exited, (*parent.path, row),
                    values['targetRemovedMm3']-baseline['targetRemovedMm3'],
                    values['nonTargetRemovedMm3']-baseline['nonTargetRemovedMm3'],
                    values['elapsedSeconds']-baseline['elapsedSeconds'])
                children.append(node)
                if node.target_mm3 > 0:
                    completed.append(node)
            if budget_stopped:
                break
        if children:
            deepest = depth+1
        if not children or budget_stopped:
            break
        # Keep distinct current cavities/poses even when their immediate outcome
        # is dominated. Their continuations need not share the same feasibility.
        ordered = sorted(children, key=_key)
        preparation = [n for n in ordered if n.path[-1]['preparationOnly']]
        frontier = ordered[:beam_width]
        if preparation and preparation[0] not in frontier:
            frontier = [*frontier[:beam_width-1], preparation[0]]
    pareto = [node for node in completed if not any(
        _dominates(other, node) for other in completed)]
    pareto.sort(key=_key)
    selected = pareto[0] if pareto else None
    evidence = {'horizon': horizon, 'beamWidth': beam_width,
        'maxExpansions': max_expansions, 'expansions': expansions,
        'maxProposalsPerNode': 6,
        'maxContactReplansPerSkill': max_contact_replans,
        'deepestCompletedDepth': deepest, 'budgetExhausted': budget_stopped,
        'physicalCalls': meter.calls-first_calls,
        'decision': 'CONTINUE' if selected is not None else 'STOP',
        'reason': 'CHECKED_SEQUENCE_PROGRESS' if selected is not None else
            'COMPUTATION_BUDGET' if budget_stopped else 'FINITE_GENERATOR_NO_CHECKED_PROGRESS',
        'preference': 'Hard native and 20/200 mm3 tissue limits first; retain target/non-target/time Pareto outcomes; recommend greatest target volume, then least non-target volume, then least modeled time. No scalar weights.',
        'pareto': [{'tileIds': [r['tileId'] for r in n.path],
            'targetMm3': n.target_mm3, 'nonTargetMm3': n.non_target_mm3,
            'elapsedSeconds': n.elapsed_seconds, 'finalStateId': n.exited.state_id}
            for n in pareto],
        'selectedTileIds': [r['tileId'] for r in selected.path] if selected else [],
        'scope': 'Finite checked native search; failure does not prove anatomical impossibility.'}
    return selected, attempts, evidence
