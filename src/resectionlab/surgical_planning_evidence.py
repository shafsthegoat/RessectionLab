"""Show executed consequences and recorded intent beside a checked native replay."""
from __future__ import annotations

import copy

from .core import semantic_digest
from .surgical_consequences import consequence_checkpoint_engine, consequence_vector
from .surgical_sequential_task import COMMON_CONFIG, TASK_PROPERTY, endpoint_assessment
from .surgical_treatment_intent import assess_treatment_execution, decide_treatment_intent

SCHEMA = 'resectionlab.recorded-planning-evidence.v1'
BRANCHES = ('conditional', 'harm-aware', 'search')
COUNTS = ('physicalCalls', 'planningPhysicalCalls', 'executionPhysicalCalls', 'canceledRequests')


def _checkpoint(scene, record, branch):
    shell = consequence_checkpoint_engine(scene, record['branches'][branch])
    shell.history = record['timeline'][branch]
    states = [shell.history[0]['stateBefore']] if shell.history else [shell.state_id]
    if len(shell.history) != len(shell.actions):
        raise ValueError('Planning evidence timeline differs from branch')
    for action, step in zip(shell.actions, shell.history):
        if step['stateBefore'] != states[-1] or step['action'] != action:
            raise ValueError('Planning evidence action and state bindings differ')
        states.append(step['stateAfter'])
    if states[-1] != shell.state_id:
        raise ValueError('Planning evidence timeline differs from branch')
    return shell, states


def _validate_decisions(engine, decisions, states, sequence):
    if type(decisions) is not list or len(decisions) > COMMON_CONFIG.max_skills:
        raise ValueError('Bounded treatment decision receipts required')
    if not decisions:
        return []
    from .surgical_procedure_policy import _validated, procedure_source_memberships
    policy = _validated(engine.scene)
    source_intent = policy.get('treatmentIntent')
    named = procedure_source_memberships(engine.scene)
    covered = named['intracranial'] <= named['observed']
    target_labels = {row['id'] for row in engine.scene.regions if row.get('target')}
    original = (source_intent['original'] if source_intent else
        {cell for cell, label in enumerate(engine.scene.labels.flat) if int(label) in target_labels})
    inventory = {item['id']: item for item in engine.scene.properties[TASK_PROPERTY]['inventory']}
    removed = set()
    position = 0
    positions = []
    for index, row in enumerate(decisions):
        if row['selectionIndex'] != index or row['sourceStateId'] != states[position] or len(row['alternatives']) > 512:
            raise ValueError('Treatment decision does not follow the recorded state')
        if row['coverageComplete'] != covered:
            raise ValueError('Treatment decision coverage differs from source')
        voxel = engine.scene.voxel_mm3
        if row['residual']['residualMm3'] != len(original - removed) * voxel:
            raise ValueError('Decision residual differs from recorded source occupancy')
        for metric, measure in row['incumbent']['harms'].items():
            subject = 'neural' if metric == 'neuralRemovedMm3' else metric.removesuffix(':removedMm3')
            if subject not in named or measure != len(removed & named[subject]) * voxel:
                raise ValueError('Decision prefix harm differs from source occupancy')
        positions.append(position)
        following = row.get('finalStateId', row['sourceStateId'])
        end = states.index(following, position)
        declared = row['decision']
        projection = row.get('projectionBinding')
        preference = policy['record'].get('preservationPreference')
        if preference is not None and (
                not isinstance(projection, dict)
                or projection.get('policyHash') != policy['record']['policyHash']
                or projection.get('sourceSceneId') != engine.scene.identity
                or projection.get('preservationPreference') != preference
                or declared['preferenceHarmMetrics'] != preference['metrics']
                or type(row.get('projectionComplete')) is not bool):
            raise ValueError('Recorded projection differs from the source preservation declaration')
        expected = decide_treatment_intent(
            row['alternatives'], incumbent=row['incumbent'], required_harms=tuple(row['incumbent']['harms']),
            preference_harms=declared['preferenceHarmMetrics'], residual=row['residual'],
            coverage_complete=row['coverageComplete'], projection_complete=row.get('projectionComplete', True))
        selected = declared['selectedId'].split('+')[0] if declared['selectedId'] else 'STOP'
        if expected != declared or row['selectedSkillId'] != selected or selected not in inventory:
            raise ValueError('Treatment intent differs from declared evidence and alternatives')
        actual = engine.actions[position:end]
        if row['executionStatus'] == 'unexecuted':
            if index != len(sequence) or actual:
                raise ValueError('Unexecuted intent has native actions')
        elif not index < len(sequence) or sequence[index] != selected:
            raise ValueError('Selected skill differs from executed inventory')
        if selected != 'STOP':
            planned = inventory[selected]['actions']
            if (len(actual) > len(planned) or
                    row['executionStatus'] == 'accepted' and len(actual) != len(planned) or
                    any(step != template and not (template['operation'] == 'place' and
                        template | {'operation': 'move'} == step) for step, template in zip(actual, planned))):
                raise ValueError('Executed actions differ from the selected source skill')
        elif any(step['operation'] != 'withdraw' for step in actual):
            raise ValueError('STOP preparation may only withdraw')
        for step in engine.history[position:end]:
            removed.update(step.get('deltas', {}).get('removedCells', []))
        position = end
    if position != len(engine.actions):
        raise ValueError('Treatment decisions do not cover the accepted action history')
    return positions


def planning_evidence(scene, record, rows, *, baseline_branch):
    baseline, _ = _checkpoint(scene, record, baseline_branch)
    alternatives = []
    for branch in BRANCHES:
        row = rows[branch]
        engine, _ = _checkpoint(scene, record, branch)
        verification = row['verification']
        counts = row['computation']
        alternatives.append({
            'branchId': branch, 'position': len(engine.actions), 'stateId': engine.state_id,
            'endpoint': copy.deepcopy(row['endpoint']), 'attempts': row['attempts'],
            'refusals': row['rejections'], 'sequence': list(row['selectedSkillIds']),
            'verification': {key: verification.get(key) for key in (
                'verified', 'adapterStopOutcome', 'adapterTerminationReason')},
            'computation': {key: counts[key] for key in COUNTS},
            'consequences': consequence_vector(engine, baseline, extended=True),
            'intentDecisions': copy.deepcopy(row.get('intentDecisions', [])),
        })
    packet = {
        'schema': SCHEMA, 'sourceSceneId': scene.identity, 'baselineBranchId': baseline_branch,
        'baselineStateId': baseline.state_id, 'alternatives': alternatives, 'clinicalValidation': False,
        'scope': 'Generated post-access control. Candidate contacts are nominal estimates. Recorded effects use native execution.',
    }
    packet['seal'] = semantic_digest(packet)
    validate_planning_evidence(scene, record, packet)
    return packet


def validate_planning_evidence(scene, record, packet):
    if (packet.get('schema') != SCHEMA or packet.get('sourceSceneId') != scene.identity or record.get('fullCase')
            or packet.get('clinicalValidation') is not False or TASK_PROPERTY not in scene.properties
            or [row['branchId'] for row in packet['alternatives']] != list(BRANCHES)
            or packet.get('seal') != semantic_digest({k: v for k, v in packet.items() if k != 'seal'})):
        raise ValueError('Recorded planning evidence is unbound')
    baseline, _ = _checkpoint(scene, record, packet['baselineBranchId'])
    if baseline.state_id != packet['baselineStateId']:
        raise ValueError('Recorded planning evidence prefix differs')
    positions = {}
    for row in packet['alternatives']:
        engine, states = _checkpoint(scene, record, row['branchId'])
        receipt = row['verification']
        stop = receipt.get('adapterStopOutcome')
        stopped = bool(stop and stop['status'] == 'accepted' and stop['reasonCode'] == 'PROCEDURE_STOP')
        if stopped and (stop['stateId'] != engine.state_id or stop['elapsedSeconds'] != 0.
                        or stop['completedFraction'] != 1. or stop['events'] or stop['deltas']
                        or row['sequence'][-1:] != ['STOP']):
            raise ValueError('STOP must bind the final checkpoint and have no physical effect')
        endpoint = endpoint_assessment(
            engine, COMMON_CONFIG, attempts=row['attempts'], invalid_actions=row['refusals'],
            stopped=stopped, externally_truncated=row['endpoint']['truncated'])
        current = consequence_vector(engine, baseline, extended=True,
            mechanism_revision=row['consequences'].get('mechanismEvidenceRevision'))
        saved = copy.deepcopy(row['consequences'])
        # The initial receipts predate explicit unknown eligible-removal credit.
        if current['treatmentIntent']['eligibleRemovedMm3'] is None:
            saved['treatmentIntent'].setdefault('eligibleRemovedMm3', None)
        if (row['position'] != len(engine.actions) or row['stateId'] != engine.state_id
                or current != saved
                or endpoint != row['endpoint'] or stopped and not receipt['verified']):
            raise ValueError('Recorded planning evidence differs from the native branch')
        counts = row['computation']
        if (any(type(counts[key]) is not int or counts[key] < 0 for key in COUNTS)
                or counts['physicalCalls'] != counts['planningPhysicalCalls'] + counts['executionPhysicalCalls']):
            raise ValueError('Recorded requests do not reconcile')
        if row['branchId'] != 'harm-aware' and row['intentDecisions']:
            raise ValueError('Only the evidence-aware controller emits intent decisions')
        positions[row['branchId']] = _validate_decisions(engine, row['intentDecisions'], states, row['sequence'])
    return positions


def planning_evidence_view(scene, record, packet):
    positions = validate_planning_evidence(scene, record, packet)
    view = copy.deepcopy(packet)
    for row in view['alternatives']:
        decisions = row['intentDecisions']
        if decisions:
            for decision, position in zip(decisions, positions[row['branchId']]):
                decision['sourcePosition'] = position
            verification = row['verification']
            row['intentExecution'] = assess_treatment_execution(
                decisions[-1]['decision'], stop_outcome=verification['adapterStopOutcome'],
                terminated=row['endpoint']['terminated'], truncated=row['endpoint']['truncated'],
                termination_reason=verification['adapterTerminationReason'])
    return view


def verify_planning_replay(scene, record, selected_engine, *, check):
    """Recheck all compared native histories and the zero-motion STOP on cold open."""
    from .surgical_engine import SurgicalEngine
    from .surgical_sequential_task import SequentialPrimitiveAdapter

    packet = record['planningEvidence']
    validate_planning_evidence(scene, record, packet)
    adapter = SequentialPrimitiveAdapter(scene)
    for branch in (packet['baselineBranchId'], *BRANCHES):
        snapshot = record['branches'][branch]
        engine = selected_engine if branch == record['branchId'] else SurgicalEngine.restore(
            scene, snapshot, check=check, force_verify=True)
        if engine.history != record['timeline'][branch]:
            raise ValueError('Recorded planning timeline differs from independently replayed actions')
        row = next((row for row in packet['alternatives'] if row['branchId'] == branch), None)
        stop = None if row is None else row['verification']['adapterStopOutcome']
        if stop is not None and stop['status'] == 'accepted':
            adapter.engine = engine
            adapter.procedure_evaluation = adapter._evaluate_procedure(engine)
            measured = adapter._stop_outcome({'slot': 'left', 'operation': 'stop', 'reason': stop['reason']})
            if measured != stop:
                raise ValueError('Recorded STOP differs from canonical final-state check')
    verify_generated_projections(scene, record, packet, check=check)


def verify_generated_projections(scene, record, packet, *, check):
    """Reproject each revised decision at an independently executed skill prefix."""
    from .surgical_generated_planning import SourceContactProjection
    from .surgical_procedure_policy import procedure_preservation_preference
    from .surgical_sequential_task import GeneratedTask, Skill, available_mask, execute_skill, observation

    if procedure_preservation_preference(scene) is None:
        return
    row = next(row for row in packet['alternatives'] if row['branchId'] == 'harm-aware')
    if not row['intentDecisions']:
        raise ValueError('Source-bound planning branch lacks recorded projections')
    skills = tuple(Skill(**record) for record in scene.properties[TASK_PROPERTY]['inventory'])
    task = GeneratedTask(scene, skills, {})
    projector = SourceContactProjection(task)
    engine = task.initial_engine()
    used = set()
    attempts = invalid = 0
    for selection, saved in enumerate(row['intentDecisions']):
        check()
        observed = observation(task, engine, attempts=attempts, invalid=invalid, selections=selection)
        index, expected = projector.project(observed, available_mask(task, engine, used))
        if (saved['sourceStateId'] != engine.state_id
                or any(saved.get(key) != value for key, value in expected.items())):
            raise ValueError('Recorded nominal projection differs from the source and checked prefix')
        if saved['executionStatus'] == 'unexecuted':
            break
        skill = skills[index]
        outcome = execute_skill(task, engine, skill, check=check)
        if (saved['finalStateId'] != engine.state_id
                or saved['checkedStopAttempted'] != outcome['checkedStopAttempted']
                or saved['executionStatus'] != outcome['status']):
            raise ValueError('Recorded projection selection differs from checked skill execution')
        used.add(skill.id)
        attempts += outcome['attemptsConsumed']
        invalid += outcome['invalidAttemptsConsumed']
    if (engine.snapshot() != record['branches']['harm-aware']
            or engine.history != record['timeline']['harm-aware']
            or row['attempts'] != attempts or row['refusals'] != invalid):
        raise ValueError('Recorded projection execution differs from the complete branch')
