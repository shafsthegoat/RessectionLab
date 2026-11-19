"""Evaluation-only composition of frozen route ranking and checked generated cranial skills."""
from __future__ import annotations

import copy
import time
from types import SimpleNamespace

import numpy as np

from generated_cranial_demo_case import build_generated_cranial_case
from resectionlab.core import array_digest
from resectionlab.functional_evidence import functional_consequence_evidence
from resectionlab.surgical_cranial_planning_roots import FIELD, validate_registry
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_joint_cranial_exit import find_checked_joint_exit, joint_exit_complete
from resectionlab.surgical_joint_cranial_search import canonical_endpoint_assessment
from resectionlab.surgical_joint_cranial_skills import (
    checked_joint_actions, checked_joint_cusa_motion, freeze_joint_cranial_limits, joint_cranial_ledger,
)
from resectionlab.surgical_planning_admission import require_planning_admission
from resectionlab.surgical_procedure_policy import PROPERTY, evaluate_procedure_sequence, procedure_source_memberships
from resectionlab.surgical_scene import original_target_memberships
from resectionlab.surgical_sequential_benchmark import feature_matrix, observed_greedy_search
from resectionlab.surgical_sequential_task import COMMON_CONFIG, SequentialPrimitiveAdapter, Skill, training_admission
from resectionlab.surgical_transition_meter import TransitionMeter


def deadline_check(deadline):
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('Frozen wall budget exhausted')
    return check


def shell_inputs(source, root, case):
    """Encode source memberships, declared treatment paths and the current working hand."""
    admission = require_planning_admission(source)
    if admission['sourceClass'] != 'generated-engineering' or source.actions or source.history:
        raise ValueError('Evaluation requires an unacted, generated, procedure-admitted root')
    policy = source.scene.properties[PROPERTY]
    if policy['startStage'] != 'post-scalp' or source._cranial_stages is None:
        raise ValueError('Source-bound closed-shell stages required')
    membership = dict(procedure_source_memberships(source.scene))
    membership['structure:corticospinal'] = membership['structure:corticospinal-tract']
    limits = {k: copy.deepcopy(v) for k, v in policy.items() if k.startswith('maximum')}
    limits['maximumStructureRemovalMm3']['corticospinal'] = limits['maximumStructureRemovalMm3']['corticospinal-tract']
    target = np.zeros(source.scene.shape, bool)
    for cells in original_target_memberships(source.scene).values():
        target.flat[cells] = True
    if array_digest(target) != case['targetMaskHash']:
        raise ValueError('Frozen original target differs')
    plan = root['planning']
    start = tuple(plan['preparationActions'][-1]['pose']['position'])
    routes = {r['id']: r['motion'] for r in plan['workRoutes']}
    if set(case['routeOrder']) != set(routes) | {'STOP'} or len(case['routeOrder']) != len(routes) + 1:
        raise ValueError('Source and frozen candidate inventories differ')
    skills = []
    for name in case['routeOrder']:
        if name == 'STOP':
            skills.append(Skill(name, 'stop', ()))
        else:
            motion = routes[name]
            if motion['slot'] != 'left' or motion['operation'] != 'apply':
                raise ValueError('Declared route must use the common working hand')
            skills.append(Skill(name, 'work', (motion,), (start, tuple(motion['pose']['position'])), radius=1.5))
    state = copy.deepcopy(source.state)
    state['slots']['right'] = copy.deepcopy(state['slots']['left'])
    state['slots']['right']['parameters'] = source.parameters(source.state['slots']['left'])
    observed = dict(source_labels=target, affine=source.scene.affine, memberships=membership,
                    native_state=state, accepted_history=[], declared_limits=limits,
                    exterior_withdrawal_plane_ras_mm=start[2],
                    remaining=dict(attempts=COMMON_CONFIG.max_attempts,
                                   seconds=COMMON_CONFIG.max_modeled_seconds, skills=1))
    return feature_matrix(observed, skills), np.ones(len(skills), dtype=bool), routes


def shell_replay(source, engine, root, endpoint, seconds):
    started = time.monotonic()
    check = deadline_check(started + seconds)
    result = dict(verified=False, physicalCalls=0, logicalStopAttempts=0)
    try:
        adapter = SequentialPrimitiveAdapter(source.scene, task_config=COMMON_CONFIG,
                                            cranial_stages=root['sourceSnapshot']['cranialStagesContract'])
        adapter.reset(seed=17)
        for event in engine.history:
            result['physicalCalls'] += 1
            _, _, _, _, info = adapter.step_action(event['action'], check=check)
            if info['outcome']['status'] != event['status'] or adapter.engine.state_id != event['stateAfter']:
                raise ValueError('Independent adapter and accepted native boundary disagree')
        result['logicalStopAttempts'] += 1
        _, _, _, _, info = adapter.stop('Frozen one-route inventory exhausted', check=check)
        result['checkedStop'] = info['outcome']
        result['finalStateId'] = adapter.engine.state_id
        result['adapterEndpoint'] = adapter.completion_assessment
        if (adapter.engine.state_id != engine.state_id or adapter.engine.state != engine.state or
                evaluate_procedure_sequence(adapter.engine) != evaluate_procedure_sequence(engine) or
                adapter.completion_assessment != endpoint):
            raise ValueError('Independent adapter and full-episode ledger or endpoint disagree')
        result['verified'] = True
    except Exception as error:
        result['failure'] = dict(type=type(error).__name__, message=str(error))
    result['wallSeconds'] = time.monotonic() - started
    return result


def closed_shell(case, method, model, profile):
    setup_start = time.monotonic()
    _, base, saved = build_generated_cranial_case(case['neuralLimitMm3'], target_source_cells=case['targetSourceCells'])
    root, = saved[FIELD]['roots']
    scene = validate_registry(base, saved[FIELD])[root['id']]
    source = SurgicalEngine.restore(scene, root['sourceSnapshot'], use_cache=False)
    try:
        training_admission(SimpleNamespace(scene=scene))
    except ValueError as error:
        refusal = str(error)
    else:
        raise ValueError('Closed shell unexpectedly admitted for finite-skill training')
    setup_seconds = time.monotonic() - setup_start
    start = time.monotonic()
    check = deadline_check(start + profile['maxOnlineSeconds'])
    meter = TransitionMeter(profile['maxPublicCalls'], check=check)
    features, mask, routes = shell_inputs(source, root, case)
    index = (model.select(features, mask, False)[0] if method == 'learned' else observed_greedy_search(features, mask))
    choice = case['routeOrder'][index]
    plan = root['planning']
    caps = {k: v for k, v in plan['limits'].items() if k != 'export_replay_reserve_calls'}
    caps.update(max_public_calls=profile['maxPublicCalls'], max_wall_seconds=profile['maxOnlineSeconds'],
                max_native_actions=profile['maxAcceptedActions'])
    limits = freeze_joint_cranial_limits(source, meter=meter, **caps)
    engine = meter.wrap(source)
    receipts = []
    if choice != 'STOP':
        prep = plan['preparationActions']
        suction, = [i for i, action in enumerate(prep) if action.get('instrument') == 'frazier_suction']
        cusa, = [i for i, action in enumerate(prep) if action.get('instrument') == 'cusa']
        programs = [('bone', plan['openingSkill']['actions']), ('membrane', prep[:suction]),
                    ('cleanup', prep[suction:cusa]), ('instrumentSetup', prep[cusa:])]
        for phase, actions in programs:
            before = engine.state['debrisMm3']
            engine, receipt = checked_joint_actions(engine, actions, limits=limits, check=check, phase=phase)
            receipts.append(dict(phase=phase, debrisBeforeMm3=before, debrisAfterMm3=engine.state['debrisMm3'],
                                 receipt=receipt))
            if receipt['status'] != 'accepted':
                break
        else:
            engine, receipt = checked_joint_cusa_motion(engine, routes[choice], limits=limits, check=check, phase='treatment')
            receipts.append(dict(phase='treatment', receipt=receipt))
    engine, exit_receipt = find_checked_joint_exit(engine, limits=limits, check=check, phase='exit')
    receipts.append(dict(phase='exit', receipt=exit_receipt))
    ledger = joint_cranial_ledger(engine, limits)
    native = engine.unwrap()
    endpoint = canonical_endpoint_assessment(native, within_limits=ledger['withinLimits'],
                                             refusal_reasons=ledger['refusalReasons'])
    computation = meter.report()
    computation['wallSeconds'] = time.monotonic() - start
    report = dict(sourceSceneId=scene.identity, initialStateId=source.state_id, finalStateId=engine.state_id,
                  voxelMm3=scene.voxel_mm3, trainingAdmissionRefusal=refusal, decisionInputs=[dict(
                      features=array_digest(features), mask=array_digest(mask))], featureMatrix=features.tolist(),
                  legalActions=case['routeOrder'], selected=choice, phases=receipts, ledger=ledger,
                  sequence=evaluate_procedure_sequence(native)['outcome'],
                  functional=functional_consequence_evidence(native, source),
                  endpoint=endpoint, progress=endpoint['progress'], setupSeconds=setup_seconds, computation=computation,
                  wholeToolExit=joint_exit_complete(native), acceptedActions=len(engine.actions),
                  priorScalpExposureCost=None, sourceStartingStage='post-scalp', snapshot=engine.snapshot())
    report['replay'] = shell_replay(source, native, root, endpoint, profile['maxReplaySeconds'])
    return report
