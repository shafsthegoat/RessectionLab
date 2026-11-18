"""Source-bound treatment accounting and a named-harm research decision rule.

The EANS–EANO glioma guideline motivates separate residual compartments and
explicit diagnosis. Functional mapping motivates evidence-backed preservation
boundaries. Neither supplies a simulated injury threshold or exchange rate.
"""
from __future__ import annotations

import copy
import math


PRESERVATION_SELECTION_RULE = 'named_harm_pareto_preservation_v1'
PRESERVATION_PREFERENCE_SCHEMA = 'resectionlab.treatment-preservation-preference.v1'
SOURCE_INTENT_SCHEMA = 'resectionlab.treatment-source-intent.v1'
DECISION_SCHEMA = 'resectionlab.treatment-intent-decision.v1'
EXECUTION_SCHEMA = 'resectionlab.treatment-intent-execution.v1'
_TOLERANCE = 1e-9
_UNKNOWN_DIAGNOSIS = {'status': 'unknown', 'label': None}
_UNKNOWN_BASELINE = {'status': 'unknown', 'summary': None}


def _named(value, maximum=100):
    return type(value) is str and bool(value.strip()) and len(value) <= maximum


def _volume(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_preservation_preference(preference):
    if (type(preference) is not dict or set(preference) != {'schema', 'metrics'}
            or preference['schema'] != PRESERVATION_PREFERENCE_SCHEMA):
        raise ValueError('PROCEDURE_VERSIONED_PRESERVATION_PREFERENCE_REQUIRED')
    metrics = preference['metrics']
    if (type(metrics) is not list or not 1 <= len(metrics) <= 32
            or any(not _named(metric, 200) or metric == 'targetRemovedMm3' for metric in metrics)
            or len(metrics) != len(set(metrics))):
        raise ValueError('PROCEDURE_NAMED_PRESERVATION_HARMS_REQUIRED')


def make_treatment_intent(*, lesion_id, revision, original_cells, eligible_cells, evidence,
        boundaries=None, compartments=None, diagnosis=None, baseline_function=None):
    """Serialize immutable source membership and independent evidence declarations.

    A boundary names a declared structure and the original tumor cells whose
    preservation it supports. Compartments use enhancing, nonenhancing or
    indeterminate. Unknown diagnosis and baseline function are explicit.
    """
    from .surgical_procedure_policy import _packed

    def pack_rows(rows):
        return {name: {**copy.deepcopy(row), 'cells': _packed(row['cells'])}
                for name, row in (rows or {}).items()}

    return {'schema': SOURCE_INTENT_SCHEMA, 'lesionId': lesion_id, 'lesionRevision': revision,
        'evidence': copy.deepcopy(evidence), 'originalCells': _packed(original_cells),
        'eligibleCells': _packed(eligible_cells), 'boundaries': pack_rows(boundaries),
        'compartments': pack_rows(compartments),
        'diagnosis': copy.deepcopy(_UNKNOWN_DIAGNOSIS if diagnosis is None else diagnosis),
        'baselineFunction': copy.deepcopy(_UNKNOWN_BASELINE if baseline_function is None else baseline_function)}


def validate_source_intent(scene, record, policy, *, owned_targets, original_targets):
    from .surgical_procedure_policy import _cells, _evidence

    if (type(record) is not dict or record.get('schema') != SOURCE_INTENT_SCHEMA
            or not _named(record.get('lesionId')) or not _named(record.get('lesionRevision'))):
        raise ValueError('PROCEDURE_VERSIONED_SOURCE_TREATMENT_INTENT_REQUIRED')
    review_status = policy['record']['evidence']['reviewStatus']

    def evidence(row, name):
        _evidence(row, name)
        if row.get('reviewStatus') != review_status:
            raise ValueError('PROCEDURE_TREATMENT_SOURCE_REVIEW_REQUIRED: ' + name)

    evidence(record.get('evidence'), 'lesion and treatment eligibility')
    original = _cells(record.get('originalCells'), scene, 'intent_original_target', nonempty=True)
    eligible = _cells(record.get('eligibleCells'), scene, 'intent_eligible_target')
    if (not owned_targets <= original or not original <= policy['intracranial']
            or (original_targets is not None and original != original_targets)):
        raise ValueError('PROCEDURE_TREATMENT_ORIGINAL_BURDEN_MISMATCH')
    if not eligible <= original & owned_targets & policy['permitted']:
        raise ValueError('PROCEDURE_TREATMENT_ELIGIBILITY_OUTSIDE_PERMITTED_TARGET')
    allowed_labels = {row['id'] for row in scene.regions
        if policy['record']['tissuePermissions'][row['material']]['remove']}
    if any(int(scene.labels.flat[cell]) not in allowed_labels for cell in eligible):
        raise ValueError('PROCEDURE_TREATMENT_ELIGIBILITY_TISSUE_FORBIDDEN')

    def source_rows(field):
        rows = record.get(field)
        if type(rows) is not dict or len(rows) > 64 or any(not _named(name) for name in rows):
            raise ValueError('PROCEDURE_TREATMENT_SOURCE_ROWS_INVALID: ' + field)
        for name, row in sorted(rows.items()):
            evidence(row, name)
            cells = _cells(row.get('cells'), scene, field, nonempty=True)
            if not cells <= original:
                raise ValueError('PROCEDURE_TREATMENT_SOURCE_CELLS_OUTSIDE_LESION: ' + name)
            yield name, row, cells

    boundaries = {}
    for name, row, cells in source_rows('boundaries'):
        if (not _named(row.get('label'), 250) or not _named(row.get('structure'))
                or not policy['structureCells'].get(row['structure'])):
            raise ValueError('PROCEDURE_TREATMENT_NAMED_BOUNDARY_EVIDENCE_REQUIRED: ' + name)
        boundaries[name] = (row, cells)
    compartments = {}
    classified = set()
    for name, row, cells in source_rows('compartments'):
        if row.get('kind') not in ('enhancing', 'nonenhancing', 'indeterminate') or classified & cells:
            raise ValueError('PROCEDURE_TREATMENT_COMPARTMENTS_MUST_BE_TYPED_AND_DISJOINT: ' + name)
        compartments[name] = (row, cells)
        classified.update(cells)

    for field, text in (('diagnosis', 'label'), ('baselineFunction', 'summary')):
        row = record.get(field)
        if row == {'status': 'unknown', text: None}:
            continue
        evidence(row, field)
        if row.get('status') not in ('source-declared', 'generated-engineering') or not _named(row.get(text), 500):
            raise ValueError('PROCEDURE_TREATMENT_SOURCE_CONTEXT_INVALID: ' + field)
    return {'record': record, 'original': original, 'eligible': eligible,
        'boundaries': boundaries, 'compartments': compartments, 'unclassified': original - classified}


def procedure_treatment_intent(engine):
    """Measure original lesion, current residual and active eligibility from native state."""
    from .surgical_procedure_policy import _validated
    from .surgical_scene import original_target_memberships

    scene = engine.scene
    policy = _validated(scene)
    source = policy.get('treatmentIntent') if policy is not None else None
    raw = source['record'] if source is not None else {}
    original = source['original'] if source is not None else None
    if original is None:
        memberships = original_target_memberships(scene)
        if memberships is not None:
            original = frozenset(int(c) for cells in memberships.values() for c in cells)
    protected = frozenset() if engine.protection is None else engine._protected_ids
    removed = set(engine.state['removed'])
    voxel = scene.voxel_mm3
    remaining = None if original is None else original - removed
    eligible = None if source is None else source['eligible'] - protected
    eligible_remaining = None if eligible is None else eligible - removed

    def mm3(cells):
        return None if cells is None else float(len(cells) * voxel)

    boundaries = []
    covered = set()
    compartments = []
    if source is not None:
        for name, (row, cells) in source['boundaries'].items():
            affected = cells & remaining
            if affected:
                covered.update(affected)
                boundaries.append({'id': name, 'label': row['label'], 'structure': row['structure'],
                    'residualMm3': mm3(affected), 'eligibleResidualMm3': mm3(affected & eligible_remaining),
                    'provenance': row['provenance'], 'evidenceSha256': row['evidenceSha256']})
        for name, (row, cells) in source['compartments'].items():
            compartments.append({'id': name, 'kind': row['kind'], 'sourceVolumeMm3': mm3(cells),
                'removedMm3': mm3(cells & removed), 'residualMm3': mm3(cells & remaining),
                'provenance': row['provenance'], 'evidenceSha256': row['evidenceSha256']})
    residual = {'residualMm3': mm3(remaining), 'eligibleResidualMm3': mm3(eligible_remaining),
        'protectedResidualMm3': None if remaining is None or engine.protection is None else mm3(remaining & protected),
        'unexplainedResidualMm3': None if source is None else mm3(remaining - covered),
        'boundaries': boundaries}
    unclassified = original if source is None else source['unclassified']
    return {'schema': SOURCE_INTENT_SCHEMA, 'sourceSceneId': scene.identity, 'stateId': engine.state_id,
        'sourceBinding': None if policy is None else policy['record']['sourceBinding'],
        'policyHash': None if policy is None else policy['record']['policyHash'],
        'lesionId': raw.get('lesionId'), 'lesionRevision': raw.get('lesionRevision'),
        'sourceEvidence': copy.deepcopy(raw.get('evidence')),
        'coverageComplete': source is not None and engine.protection is not None and not engine.protection.missing,
        'originalBurdenMm3': mm3(original), 'originalRemovedMm3': None if original is None else mm3(original & removed),
        'eligibleTreatmentMm3': mm3(eligible),
        'eligibleRemovedMm3': None if eligible is None else mm3(eligible & removed),
        'declaredEligibleTreatmentMm3': None if source is None else mm3(source['eligible']), 'residual': residual,
        'tumorCompartments': compartments,
        'unclassifiedTumorMm3': mm3(unclassified),
        'unclassifiedResidualMm3': None if remaining is None else mm3(unclassified & remaining),
        'diagnosis': copy.deepcopy(raw.get('diagnosis', _UNKNOWN_DIAGNOSIS)),
        'baselineFunction': copy.deepcopy(raw.get('baselineFunction', _UNKNOWN_BASELINE)),
        'clinicalValidation': False}


def decide_treatment_intent(alternatives, *, incumbent=None, required_harms, preference_harms=(),
        residual=None, coverage_complete=False):
    """Select additional treatment by a paired-world partial order.

    Each row has an ID, incremental `treatmentMm3` (scalar or keyed by world),
    a `harms` mapping and `admissible` (true, false for known exclusion, or
    null for unknown execution/evidence). Hard bounds are supplied by the
    caller. The incumbent is the qualified zero-additional-treatment option.
    All named harms are retained in the full frontier. A declared preference
    can resolve it using a named subset while retaining the complete frontier.
    `coverage_complete` independently attests coverage of stopping evidence.
    """
    required = tuple(dict.fromkeys((*required_harms, *preference_harms)))
    preference = tuple(dict.fromkeys(preference_harms)) or required
    rows = list(alternatives)
    if any(type(row) is not dict or not _named(row.get('id')) for row in rows):
        raise ValueError('Treatment alternatives require known IDs')
    if len(rows) != len({row['id'] for row in rows}):
        raise ValueError('Treatment alternatives require unique IDs')
    rows.sort(key=lambda row: row['id'])
    result = {'schema': DECISION_SCHEMA, 'status': 'reassess', 'reason': 'required_evidence_unavailable',
        'selectedId': None, 'frontierIds': [], 'preferenceFrontierIds': [],
        'preferenceHarmMetrics': list(preference), 'controllingHarmMetrics': [], 'supportingBoundaries': [],
        'clinicalValidation': False}

    def treatment(row):
        data = row.get('treatmentMm3')
        return data if type(data) is dict else {'nominal': data}

    if incumbent is None or incumbent.get('admissible') is not True or not preference:
        return result
    zero = treatment(incumbent)
    if (not zero or any(not _volume(v) or v > _TOLERANCE for v in zero.values())
            or any(not _volume(incumbent.get('harms', {}).get(name)) for name in required)):
        return result
    evaluations = {}
    unknown = False
    scenario_dependent_benefit = False
    for row in rows:
        admissible = row.get('admissible')
        if admissible is False:
            continue
        benefit = treatment(row)
        harms = row.get('harms', {})
        if (admissible is not True or benefit.keys() != zero.keys()
                or not all(_volume(v) for v in benefit.values())
                or not all(_volume(harms.get(name)) for name in required)):
            unknown = True
            continue
        if all(benefit[world] > _TOLERANCE for world in zero):
            evaluations[row['id']] = (benefit, harms)
        elif any(benefit[world] > _TOLERANCE for world in zero):
            scenario_dependent_benefit = True

    def signed_values(row_id, harm_names):
        benefit, harms = evaluations[row_id]
        return (*(-benefit[world] for world in zero), *(harms[name] for name in harm_names))

    def dominates(a, b):
        return (all(x <= y + _TOLERANCE for x, y in zip(a, b))
            and any(x < y - _TOLERANCE for x, y in zip(a, b)))

    def frontier(harm_names):
        vectors = {row_id: signed_values(row_id, harm_names) for row_id in evaluations}
        return [row_id for row_id, vector in vectors.items() if not any(
            dominates(other, vector) for other in vectors.values())]

    full_front = frontier(required)
    preferred = frontier(preference)
    result.update(frontierIds=full_front, preferenceFrontierIds=preferred)
    if unknown:
        return result
    if scenario_dependent_benefit:
        return {**result, 'reason': 'additional_treatment_varies_by_world'}
    if preferred:
        chosen = min(preferred, key=lambda row_id: (row_id not in full_front, row_id))
        best = signed_values(chosen, preference)
        excess = [name for name in preference
            if evaluations[chosen][1][name] > incumbent['harms'][name] + _TOLERANCE]
        varying = [name for name in preference if any(abs(evaluations[f][1][name] - evaluations[chosen][1][name])
            > _TOLERANCE for f in preferred)]
        if excess or any(any(abs(x-y) > _TOLERANCE
                for x, y in zip(best, signed_values(row_id, preference))) for row_id in preferred):
            return {**result, 'reason': 'unresolved_treatment_harm_tradeoff',
                'controllingHarmMetrics': sorted(set(excess + varying))}
        matched_treatment = [row_id for row_id, (benefit, _) in evaluations.items()
            if row_id != chosen and all(abs(benefit[world] - evaluations[chosen][0][world]) <= _TOLERANCE
                                       for world in zero)]
        controlling = [name for name in preference if any(evaluations[row_id][1][name]
            > evaluations[chosen][1][name] + _TOLERANCE for row_id in matched_treatment)]
        return {**result, 'status': 'continue', 'reason': 'positive_treatment_within_preservation_preference',
            'selectedId': chosen, 'controllingHarmMetrics': controlling}

    if residual is None or not coverage_complete or not _volume(residual.get('residualMm3')):
        return result
    if residual['residualMm3'] <= _TOLERANCE:
        return {**result, 'status': 'stop_after_treatment', 'reason': 'original_treatment_burden_exhausted'}
    boundaries = {row['id']: row for row in residual.get('boundaries', ())
        if _volume(row.get('residualMm3'))
        and _TOLERANCE < row['residualMm3'] <= residual['residualMm3'] + _TOLERANCE}
    unexplained = residual.get('unexplainedResidualMm3')
    eligible = residual.get('eligibleResidualMm3')
    if (not boundaries or not _volume(unexplained) or unexplained > _TOLERANCE or not _volume(eligible)
            or eligible > residual['residualMm3'] + _TOLERANCE
            or sum(row['residualMm3'] for row in boundaries.values()) + _TOLERANCE < residual['residualMm3']):
        return {**result, 'reason': 'residual_preservation_boundary_unproven'}
    if eligible <= _TOLERANCE:
        supported = set(boundaries)
    else:
        eligible_by_boundary = {name: row.get('eligibleResidualMm3') for name, row in boundaries.items()}
        if (any(not _volume(value) or value > boundaries[name]['residualMm3'] + _TOLERANCE
                for name, value in eligible_by_boundary.items())
                or sum(eligible_by_boundary.values()) + _TOLERANCE < eligible):
            return {**result, 'reason': 'residual_preservation_boundary_unproven'}
        blocked = [set(row.get('blockingBoundaries', ())) & boundaries.keys()
            for row in rows if row.get('admissible') is False]
        if not blocked or not all(blocked):
            return {**result, 'reason': 'residual_preservation_boundary_unproven'}
        supported = set().union(*blocked)
        if not {name for name, value in eligible_by_boundary.items() if value > _TOLERANCE} <= supported:
            return {**result, 'reason': 'residual_preservation_boundary_unproven'}
        supported.update(name for name, value in eligible_by_boundary.items() if value <= _TOLERANCE)
    return {**result, 'status': 'stop_with_residual', 'reason': 'supported_preservation_boundary',
        'supportingBoundaries': [copy.deepcopy(boundaries[name]) for name in sorted(supported)]}


def assess_treatment_execution(decision, *, stop_outcome=None, terminated=False, truncated=False,
        termination_reason=None):
    """Attach canonical STOP, failure or truncation to an independent intent decision."""
    stop_code = None if stop_outcome is None else stop_outcome.get('reasonCode')
    stop_status = None if stop_outcome is None else stop_outcome.get('status')
    if truncated:
        assessment = 'truncated'
    elif terminated and isinstance(termination_reason, str) and termination_reason.startswith('damage_budget:'):
        assessment = 'failure'
    elif stop_outcome is None:
        assessment = 'terminated_without_stop' if terminated else 'not_executed'
    elif stop_status == 'rejected':
        assessment = 'stop_refused'
    elif (stop_status == 'accepted' and stop_code == 'PROCEDURE_STOP'
            and stop_outcome.get('elapsedSeconds') == 0. and stop_outcome.get('completedFraction') == 1.):
        assessment = {'continue': 'premature_stop', 'reassess': 'unresolved_stop',
            'stop_with_residual': 'justified_residual_stop',
            'stop_after_treatment': 'completed_treatment_stop'}[decision['status']]
    else:
        assessment = 'stop_unverified'
    return {'schema': EXECUTION_SCHEMA, 'intentStatus': decision['status'], 'executionStatus': assessment,
        'stopStatus': stop_status, 'stopReasonCode': stop_code, 'terminationReason': termination_reason,
        'terminated': bool(terminated), 'truncated': bool(truncated), 'clinicalValidation': False}
