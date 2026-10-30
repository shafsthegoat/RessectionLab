"""Versioned user research policies over one admitted, immutable comparison.

This module only compares saved vectors. It does not replay programs, generate
physical actions, plan continuations, fit parameters, or update an optimizer.
"""
from __future__ import annotations

import copy
import math

from .core import semantic_digest
from .surgical_consequences import compare_samples, validate_saved_comparison
from .surgical_engine import SCHEMA as ENGINE_SCHEMA

SCHEMA = 'resectionlab.surgical-consequence-user-analysis.v1'
VIEW_SCHEMA = 'resectionlab.surgical-consequence-analysis-view.v1'
MAX_REVISIONS = 32
REQUIRED_METRICS = ('supplyDeficitSeconds', 'motorEncounterMm3',
                    'languageEncounterMm3', 'patientShiftErrorMm')
OBJECTIVES = ['targetRemovedMm3', 'nonTargetRemovedMm3', 'elapsedSeconds']
BASIS = 'declared_research_task_not_clinical_threshold'
POLICY_KEYS = {'minimumAdditionalTargetMm3', 'maximumCumulativeNonTargetMm3',
               'requiredMetrics', 'maxBrainStrainSquaredMm3S',
               'maxBrainStressSquaredPa2Mm3S'}


def _same(left, right):
    """JSON identity, including numeric/boolean distinctions."""
    return semantic_digest(left) == semantic_digest(right)


def _number(value):
    if type(value) not in (int, float):
        raise ValueError('Research policy limits must be finite nonnegative numbers, never booleans')
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError('Research policy limit exceeds finite numeric range') from error
    if not math.isfinite(number) or number < 0:
        raise ValueError('Research policy limits must be finite nonnegative numbers, never booleans')
    return number


def normalize_policy_input(value):
    """Admit only the bounded editor controls, with fixed units and roles."""
    if not isinstance(value, dict) or set(value) != POLICY_KEYS:
        raise ValueError('Exact research policy editor fields required')
    required = value['requiredMetrics']
    if (not isinstance(required, list) or len(required) > len(REQUIRED_METRICS)
            or any(type(item) is not str or item not in REQUIRED_METRICS for item in required)
            or len(set(required)) != len(required)):
        raise ValueError('Required metrics must be unique supported evidence dimensions')
    result = {'minimumAdditionalTargetMm3': _number(value['minimumAdditionalTargetMm3']),
              'maximumCumulativeNonTargetMm3': _number(value['maximumCumulativeNonTargetMm3']),
              'requiredMetrics': list(required)}
    for name in ('maxBrainStrainSquaredMm3S', 'maxBrainStressSquaredPa2Mm3S'):
        result[name] = None if value[name] is None else _number(value[name])
    return result


def _policy(policy_input):
    constraints = [
        {'metric': 'targetRemovedMm3', 'scope': 'incremental', 'operator': 'at_least',
         'value': policy_input['minimumAdditionalTargetMm3']},
        {'metric': 'nonTargetRemovedMm3', 'scope': 'cumulative', 'operator': 'at_most',
         'value': policy_input['maximumCumulativeNonTargetMm3']},
    ]
    for input_name, metric in [('maxBrainStrainSquaredMm3S', 'brainStrainSquaredMm3S'),
                               ('maxBrainStressSquaredPa2Mm3S', 'brainStressSquaredPa2Mm3S')]:
        if policy_input[input_name] is not None:
            constraints.append({'metric': metric, 'scope': 'cumulative', 'operator': 'at_most',
                                'value': policy_input[input_name]})
    return {'id': 'user-consequence-analysis', 'basis': BASIS, 'objectives': list(OBJECTIVES),
            'constraints': constraints, 'requiredMetrics': list(policy_input['requiredMetrics'])}


def _decision(report, policy):
    return compare_samples(report['samples'], policy, stop_branch=report['stopBranchId'],
                           scenario_ids=[s['id'] for s in report['scenarios']],
                           branch_ids=report['alternativeBranchIds'])


def _source_binding(engine, record, report, case_hash):
    # The production validator retains its independent app declaration pin. There
    # is deliberately no artifact-provided declaration override on this API.
    validate_saved_comparison(engine.scene, record, report)
    if not isinstance(case_hash, str) or not 0 < len(case_hash) <= 200:
        raise ValueError('A bounded source case identity is required')
    if not _same(record.get('consequenceComparison'), report):
        raise ValueError('Analysis requires the original comparison saved in this case')
    branch_ids = report['alternativeBranchIds']
    baseline_id = report['baselineBranchId']
    if (len(branch_ids) != 3 or len(set(branch_ids)) != 3
            or report['stopBranchId'] not in branch_ids or baseline_id in branch_ids):
        raise ValueError('Analysis requires exactly two continuations, STOP and a separate baseline')
    baseline = record['branches'][baseline_id]
    programs = []
    for branch_id in [baseline_id, *branch_ids]:
        snapshot = record['branches'][branch_id]
        checkpoint = snapshot['checkpoint']
        if (snapshot.get('schema') != ENGINE_SCHEMA
                or snapshot.get('sceneId') != engine.scene.identity
                or type(checkpoint.get('position')) is not int
                or checkpoint['position'] != len(snapshot['actions'])
                or checkpoint.get('stateId') != snapshot.get('stateId')):
            raise ValueError('Analysis native program checkpoint metadata changed')
        programs.append({'branchId': branch_id, 'snapshotHash': semantic_digest(snapshot),
                         'checkpointHash': semantic_digest(checkpoint),
                         'actionHash': semantic_digest(snapshot['actions'])})
    baseline_context = {'branchId': baseline_id, 'position': len(baseline['actions']),
                        'stateId': baseline['stateId']}
    return {'baseline': {**baseline_context, **{key: value for key, value in programs[0].items()
                                               if key != 'branchId'}},
            'programs': programs[1:], 'scenariosHash': semantic_digest(report['scenarios']),
            'samplesHash': semantic_digest(report['samples']),
            'sourceGeometryHash': report['declaration']['sourceGeometryHash'],
            'sourceArchiveSha256': report['sourceArchiveSha256'],
            'declarationHash': report['declarationHash']}


def _analysis(report, case_hash, binding, revision, policy_input):
    policy_input = normalize_policy_input(policy_input)
    policy = _policy(policy_input)
    result = {'schema': SCHEMA, 'revision': revision, 'caseHash': case_hash,
              'sourceSceneId': report['sourceSceneId'], 'comparisonSeal': report['comparisonSeal'],
              'binding': copy.deepcopy(binding), 'policyInput': policy_input,
              'policy': policy, 'decision': _decision(report, policy),
              'optimizerUpdates': 0, 'newPhysicalActions': 0, 'clinicalValidation': False}
    result['analysisSeal'] = semantic_digest(result)
    return result


def _validated_history(record, report, case_hash, binding):
    history = record.get('consequenceAnalyses', [])
    if not isinstance(history, list) or len(history) > MAX_REVISIONS:
        raise ValueError('Bounded consequence analysis revision history required')
    for revision, item in enumerate(history, 1):
        if not isinstance(item, dict) or type(item.get('revision')) is not int or item['revision'] != revision:
            raise ValueError('Consequence analysis revisions must be consecutive and monotone')
        expected = _analysis(report, case_hash, binding, revision, item.get('policyInput'))
        if not _same(item, expected):
            raise ValueError('Consequence analysis seal, source binding, policy or decision changed')
    return history


def validate_saved_analyses(engine, record, report, case_hash):
    """Validate every revision against the admitted original report, without replay."""
    binding = _source_binding(engine, record, report, case_hash)
    _validated_history(record, report, case_hash, binding)


def _at_baseline(engine, record, report):
    baseline = record['branches'][report['baselineBranchId']]
    return (record.get('branchId') == report['baselineBranchId']
            and type(record.get('position')) is int
            and record['position'] == len(baseline['actions'])
            and _same(engine.snapshot(), baseline))


def analysis_view(engine, record, report, case_hash):
    """Return current editor availability and retain a historical saved decision."""
    binding = _source_binding(engine, record, report, case_hash)
    history = _validated_history(record, report, case_hash, binding)
    available = _at_baseline(engine, record, report)
    return {'schema': VIEW_SCHEMA, 'caseHash': case_hash, 'sourceSceneId': report['sourceSceneId'],
            'comparisonSeal': report['comparisonSeal'],
            'context': {'branchId': record.get('branchId'), 'position': record.get('position'),
                        'stateId': engine.state_id},
            'available': available,
            'reason': None if available else 'Return explicitly to the frozen common baseline branch endpoint to re-evaluate; saved analysis is historical.',
            'baseline': {key: binding['baseline'][key] for key in ('branchId', 'position', 'stateId')},
            'latest': copy.deepcopy(history[-1]) if history else None,
            'current': bool(history) and available, 'revisionCount': len(history)}


def reevaluate_analysis(engine, record, report, case_hash, policy_input, expected_comparison_seal):
    """Append a sealed comparison revision to a copied native record only."""
    binding = _source_binding(engine, record, report, case_hash)
    history = _validated_history(record, report, case_hash, binding)
    if not _same(expected_comparison_seal, report['comparisonSeal']):
        raise ValueError('Stale consequence comparison seal; reload the admitted comparison')
    if not _at_baseline(engine, record, report):
        raise ValueError('Re-evaluation requires the exact frozen common baseline branch endpoint and native state')
    if len(history) >= MAX_REVISIONS:
        raise ValueError('Consequence analysis revision limit reached; all saved revisions are preserved')
    revision = _analysis(report, case_hash, binding, len(history) + 1, policy_input)
    result = copy.deepcopy(record)
    result['consequenceAnalyses'] = [*copy.deepcopy(history), revision]
    return result
