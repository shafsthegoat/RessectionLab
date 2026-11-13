"""Prospective, source-sized engineering tasks for complete native sequences.

Task freezing never changes anatomy or admits an execution. The existing native
guards, 200 mm3 whole-tissue cap and 20 mm3 treatment-continuation cap remain.
Original target masks, rather than categorical ownership, define the denominator.
"""
from __future__ import annotations

import math
import numpy as np

from .core import array_digest, semantic_digest, thaw_json
from .surgical_scene import original_target_memberships
from .surgical_sequence_outcomes import sequence_outcome
from .surgical_task_domain import reviewed_domain
from .surgical_tools import catalog
from .surgical_patient_role import patient_role_evidence

SCHEMA = 'resectionlab.patient-sequence-task.v1'
POLICY = {'minimumOriginalTargetFraction': .5, 'minimumContributingZones': 3,
          'minimumOriginalTargetFractionPerZone': .05,
          'maximumWholeNonTargetMm3': 200., 'maximumTreatmentNonTargetMm3': 20.,
          'maxNativeActions': 240, 'maxPhysicalCalls': 4000, 'maxWallSeconds': 240.}


def _target_cells(scene):
    memberships = original_target_memberships(scene)
    if memberships is None or not memberships:
        raise ValueError('ORIGINAL_TARGET_EVIDENCE_REQUIRED')
    cells = np.unique(np.concatenate(list(memberships.values())))
    if not len(cells):
        raise ValueError('NONEMPTY_ORIGINAL_TARGET_REQUIRED')
    return cells


def _zones(scene, cells):
    points = scene.points(cells)
    center = points.mean(axis=0)
    codes = ((points >= center)*np.array([4, 2, 1])).sum(axis=1)
    return {str(code): cells[codes == code] for code in range(8)}, center.tolist()


def _source_role(case):
    data = case.metadata.get('data_case', {})
    role, split = data.get('role'), data.get('split')
    if role in ('TRAIN', 'DEVELOPMENT') and split == role:
        return role
    if role is not None or split is not None:
        raise ValueError('DECLARED_UNRESERVED_SOURCE_ROLE_REQUIRED')
    if case.metadata.get('generated') is True and case.metadata.get('is_synthetic') is True:
        return 'GENERATED_CONTROL'
    raise ValueError('DECLARED_UNRESERVED_SOURCE_ROLE_REQUIRED')


def freeze_patient_task(case, engine, *, role_root=None):
    """Freeze before search; keep an ineligible prospectus inspectable.

    TRAIN remains TRAIN. No task construction can promote a reserved case,
    infer missing critical anatomy, erase a prefix cost or relax physical bounds.
    """
    engine.scene.require_cache_source()
    if engine.scene.source.get('caseHash') != case.semantic_hash:
        raise ValueError('PATIENT_TASK_SOURCE_MISMATCH')
    role = _source_role(case)
    role_evidence = patient_role_evidence(case, role_root)
    cells = _target_cells(engine.scene)
    zones, center = _zones(engine.scene, cells)
    volume = len(cells)*engine.scene.voxel_mm3
    baseline = sequence_outcome(engine)
    domain = reviewed_domain(engine.scene)
    blockers = []
    if role_evidence['status'] == 'unavailable':
        blockers.append(role_evidence['reason'])
    if engine.guard_report()['missingRequired']:
        blockers.append('MISSING_CRITICAL_ANNOTATIONS')
    # Match the engine's mandatory critical-presence floor. An editable empty
    # contract.required must not make an unreviewed brainstem task-eligible.
    elif any(annotation.review_status not in ('reviewed', 'engineering-reviewed')
             for annotation in engine.protection.annotations
             if annotation.structure_id in {'brainstem', *engine.protection.required}):
        blockers.append('CRITICAL_ANNOTATION_REVIEW_REQUIRED')
    if domain is None:
        blockers.append('REVIEWED_LOCAL_DOMAIN_REQUIRED')
    if not baseline['withdrawalComplete']:
        blockers.append('WITHDRAW_BOTH_TOOLS_BEFORE_PLANNING')
    if baseline['dimensions']['nonTargetRemovedMm3']['cumulative'] > POLICY['maximumWholeNonTargetMm3']+1e-9:
        blockers.append('PREFIX_EXCEEDS_WHOLE_NON_TARGET_CAP')
    if engine.state.get('toolMechanicsRevision') != 2 or engine.state.get('physicalClockRevision') != 1:
        blockers.append('STOCK_GEOMETRY_AND_SHARED_CLOCK_REQUIRED')
    stage = thaw_json(case.metadata.get('integrated_rehearsal', {}))
    if not stage.get('stage'):
        blockers.append('STARTING_ASSUMPTIONS_REQUIRED')
    task = {'schema': SCHEMA, 'caseHash': case.semantic_hash, 'sourceSceneId': engine.scene.identity,
        'sourceFamily': case.metadata.get('data_case', {}).get('family_id', case.case_id),
        'sourceRole': role, 'startingStateId': engine.state_id,
        'sourceRoleEvidence': role_evidence,
        'generatedControl': case.metadata.get('generated') is True,
        'prefixActions': len(engine.actions), 'prefixActionHash': semantic_digest(engine.actions),
        'originalTargetCellHash': array_digest(cells), 'originalTargetMm3': volume,
        'originalTargetOctantCenterRasMm': center,
        'originalTargetZones': {key: {'cellHash': array_digest(ids), 'sourceMm3': len(ids)*engine.scene.voxel_mm3}
                                for key, ids in zones.items()},
        'minimumWholeTargetMm3': volume*POLICY['minimumOriginalTargetFraction'],
        'minimumTargetPerZoneMm3': volume*POLICY['minimumOriginalTargetFractionPerZone'],
        'policy': dict(POLICY), 'startingAssumptions': stage,
        'modelBinding': semantic_digest({'catalog': catalog(), 'properties': engine.scene.properties,
            'protection': engine._protection_contract, 'domain': None if domain is None else vars(domain)}),
        'eligibilityBlockers': blockers, 'clinicalValidation': False,
        'scope': 'Prospectively frozen partial-target engineering task, including complete access/preparation/treatment/exit. This is not whole-tumor treatment or clinical safety.'}
    task['taskSeal'] = semantic_digest(task)
    return task


def assess_patient_task(case, engine, task, *, baseline, treatment_baseline=None,
                        exact_replay=False, physical_calls=None, wall_seconds=None,
                        role_root=None):
    """Evaluate an exact native continuation against its frozen prospectus.

    A treatment baseline must be the real accepted checkpoint after access and
    preparation. It changes only the additional-work bound; whole costs remain.
    Missing replay/computation evidence fails those gates, never defaults to zero.
    """
    canonical = freeze_patient_task(case, baseline, role_root=role_root)
    if task != canonical:
        raise ValueError('PATIENT_TASK_FROZEN_INPUTS_CHANGED')
    outcome = sequence_outcome(engine, baseline=baseline)
    treatment = (sequence_outcome(engine, baseline=treatment_baseline)
                 if treatment_baseline is not None else None)
    if treatment_baseline is not None:
        # The declared access checkpoint itself must extend the same source
        # prefix; a later unrelated or earlier cheaper baseline cannot refund it.
        access = sequence_outcome(treatment_baseline, baseline=baseline)
        if (access['dimensions']['targetRemovedMm3']['incremental'] != 0 or
                access['originalTarget']['incrementalRemovedMm3'] != 0):
            raise ValueError('TREATMENT_BASELINE_MUST_PRECEDE_TARGET_REMOVAL')
    cells = _target_cells(engine.scene)
    zones, _ = _zones(engine.scene, cells)
    removed = set(map(int, engine.state['removed']))
    owned = {region['id'] for region in engine.scene.regions if region.get('target')}
    credited = {cell for cell in removed if int(engine.scene.labels.flat[cell]) in owned}
    zone_progress = {key: sum(int(cell) in credited for cell in ids)*engine.scene.voxel_mm3
                     for key, ids in zones.items()}
    protected_effects = set(engine._protected_ids) & removed
    for field in ('cutEdges', 'transectedEdges'):
        protected_effects.update(set(engine._protected_ids) & {cell for edge in engine.state.get(field, []) for cell in edge})
    call_valid = type(physical_calls) is int and len(engine.actions) <= physical_calls <= POLICY['maxPhysicalCalls']
    wall_valid = type(wall_seconds) in (int, float) and math.isfinite(wall_seconds) and 0 <= wall_seconds <= POLICY['maxWallSeconds']
    gates = {'declaredEligibility': not task['eligibilityBlockers'],
        'originalTargetCoverage': outcome['originalTarget']['fractionRemoved'] >= POLICY['minimumOriginalTargetFraction'],
        'targetTreatmentCredit': outcome['dimensions']['targetRemovedMm3']['cumulative']+1e-9 >= task['minimumWholeTargetMm3'],
        'spatialProgress': sum(value+1e-9 >= task['minimumTargetPerZoneMm3'] for value in zone_progress.values()) >= POLICY['minimumContributingZones'],
        'wholeNonTarget': outcome['dimensions']['nonTargetRemovedMm3']['cumulative'] <= POLICY['maximumWholeNonTargetMm3']+1e-9,
        'treatmentNonTarget': treatment is not None and treatment['dimensions']['nonTargetRemovedMm3']['incremental'] <= POLICY['maximumTreatmentNonTargetMm3']+1e-9,
        'noProtectedRemovalOrCuts': not protected_effects,
        'finalWithdrawal': outcome['withdrawalComplete'],
        'nativeActions': len(engine.actions) <= POLICY['maxNativeActions'],
        'exactReplay': exact_replay is True, 'physicalCalls': call_valid, 'wallBudget': wall_valid}
    return {'schema': SCHEMA, 'taskSeal': task['taskSeal'], 'stateId': engine.state_id,
        'taskSatisfied': all(gates.values()), 'gates': gates,
        'failedGates': [name for name, passed in gates.items() if not passed],
        'zoneTreatmentMm3': zone_progress, 'protectedRemovedOrCutSourceCells': len(protected_effects),
        'outcome': outcome, 'treatmentBaselineStateId': treatment_baseline.state_id if treatment_baseline is not None else None,
        'physicalCalls': physical_calls, 'wallSeconds': wall_seconds,
        'clinicalValidation': False}
