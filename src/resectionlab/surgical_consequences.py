"""Source-bound consequence vectors and paired-scenario research comparisons.

There is no scalar reward or clinical acceptance here. Missing mechanisms stay
unknown; a STOP fallback does not pretend to meet the resection task. Scenario
dominance keeps each plan paired to the same declared parameter world.
"""
from __future__ import annotations

import copy
import math
import numpy as np

from .core import array_digest, semantic_digest
from .surgical_modeled_outcomes import summarize_engine_outcomes

SCHEMA = 'resectionlab.surgical-consequence-comparison.v1'
VECTOR_SCHEMA = 'resectionlab.surgical-consequence-vector.v1'
APP_DECLARATION_HASH = 'sha256:cb075f189f9455508a7ec454dda0e2bfe38809b2a0fed0b0a824b225aad3f3f9'
DECLARATION_KEYS = ('sourceArchiveSha256', 'sourceSceneId', 'sourceGeometryHash', 'prefixBranch', 'prefixActions',
                    'alternativeBranches', 'stopBranch', 'scenarios', 'policies')


def comparison_declaration(protocol):
    return {key: copy.deepcopy(protocol[key]) for key in DECLARATION_KEYS}
def objective_scope(name):
    return 'incremental' if name == 'targetRemovedMm3' else 'cumulative'
# Units and directions cannot be supplied or reversed by a score request.
METRICS = {
    'targetRemovedMm3': ('mm3', 'max'), 'nonTargetRemovedMm3': ('mm3', 'min'),
    'elapsedSeconds': ('s', 'min'), 'brainStrainSquaredMm3S': ('mm3*s', 'min'),
    'tumorStrainSquaredMm3S': ('mm3*s', 'min'),
    'brainStressSquaredPa2Mm3S': ('Pa2*mm3*s', 'min'),
    'thermalPeakC': ('degC', 'min'), 'thermalExposureMm3KS': ('mm3*K*s', 'min'),
    'bloodGeneratedMm3': ('mm3', 'min'), 'supplyDeficitSeconds': ('s', 'min'),
    'drainageDeficitSeconds': ('s', 'min'),
    'motorEncounterMm3': ('mm3', 'min'), 'languageEncounterMm3': ('mm3', 'min'),
    'patientShiftErrorMm': ('mm', 'min'),
}
UNKNOWN = {
    'motorEncounterMm3': 'No admitted source-bound motor encounter evaluator in this task; missing mapping is not negative mapping.',
    'languageEncounterMm3': 'No admitted source-bound language encounter evaluator in this task; cortical/subcortical probe type does not identify language.',
    'patientShiftErrorMm': 'No accepted withheld acquired shift observation for this model/task.',
}


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Consequence values must be finite numbers, never booleans')
    return float(value)


def _values(engine):
    metrics = engine.metrics()
    result = {name: None for name in METRICS}
    for name in ('targetRemovedMm3', 'nonTargetRemovedMm3', 'elapsedSeconds'):
        result[name] = metrics[name]
    continuum = engine.state.get('brainContinuumV1')
    if continuum is not None:
        rows = continuum['exposureBySourceLabel']
        for material, output in [('soft_tissue', 'brain'), ('tumor', 'tumor')]:
            labels = {str(r['id']) for r in engine.scene.regions if r['material'] == material}
            if labels:
                result[output + 'StrainSquaredMm3S'] = sum(rows.get(k, {}).get('strainSquaredVolumeTimeMm3Seconds', 0.) for k in labels)
                if output == 'brain':
                    result['brainStressSquaredPa2Mm3S'] = sum(rows.get(k, {}).get('stressSquaredVolumeTimePa2Mm3Seconds', 0.) for k in labels)
    if 'bipolarThermalV1' in engine.state:
        result['thermalPeakC'] = metrics['bipolarPeakTemperatureC']
        result['thermalExposureMm3KS'] = metrics['bipolarExcessTemperatureVolumeTimeMm3KSeconds']
    if 'vascularGraphV1' in engine.state:
        vascular = metrics['vesselPhysicsV2']
        result['bloodGeneratedMm3'] = vascular['generatedMm3']
        regions = vascular['regions']
        if regions:
            # Maximum declared interface deficit, not a sum of unknown/overlapping territories.
            result['supplyDeficitSeconds'] = max(r['supplyDeficitSeconds'] for r in regions.values())
            result['drainageDeficitSeconds'] = max(r['drainageDeficitSeconds'] for r in regions.values())
    elif any(r['material'] == 'vessel' and np.any(engine.scene.labels == r['id']) for r in engine.scene.regions):
        if 'vesselsV2' in engine.state:
            result['bloodGeneratedMm3'] = metrics['vesselPhysicsV2']['generatedMm3']
    return result


def consequence_vector(engine, baseline):
    # Reuse authoritative prefix, source-ownership and history validation.
    outcomes = summarize_engine_outcomes(engine, baseline=baseline)
    values, before = _values(engine), _values(baseline)
    dimensions = {}
    for name, (unit, direction) in METRICS.items():
        value, initial = values[name], before[name]
        unknown = value is None or initial is None
        dimensions[name] = {'unit': unit, 'direction': direction,
            'status': 'unavailable' if unknown else 'source_count' if name in ('targetRemovedMm3', 'nonTargetRemovedMm3', 'elapsedSeconds') else 'modeled_unvalidated',
            'cumulative': None if unknown else _finite(value),
            'incremental': None if unknown else _finite(value) - _finite(initial),
            'reason': UNKNOWN.get(name, 'Required explicit mechanism or mapped source coverage is absent.') if unknown else None}
    return {'schema': VECTOR_SCHEMA, 'sourceSceneId': engine.scene.identity,
        'sourceGeometryHash': semantic_digest({'labels': array_digest(engine.scene.labels), 'affine': engine.scene.affine.tolist()}),
        'stateId': engine.state_id, 'baselineStateId': baseline.state_id,
        'prefixActions': len(baseline.actions), 'nativeActions': len(engine.actions),
        'actionHash': semantic_digest(engine.actions),
        'cleanupComplete': not engine.state['holds'] and not any(s['active'] or s['inserted'] or s['grip'] for s in engine.state['slots'].values()),
        'residualTargetMm3': outcomes['target'].get('residualMm3'),
        'dimensions': dimensions, 'clinicalValidation': False,
        'scope': 'Source annotation counts and explicitly modeled consequences; not injury, histological margins or neurological outcome.'}


def validate_policy(policy):
    if (not isinstance(policy, dict) or set(policy) != {'id', 'basis', 'objectives', 'constraints', 'requiredMetrics'}
            or policy['basis'] != 'declared_research_task_not_clinical_threshold'
            or not isinstance(policy['id'], str) or not 0 < len(policy['id']) <= 100):
        raise ValueError('Explicit bounded research comparison policy required')
    for key in ('objectives', 'requiredMetrics'):
        values = policy[key]
        if (not isinstance(values, list) or len(values) > len(METRICS)
                or any(type(v) is not str or v not in METRICS for v in values)
                or len(set(values)) != len(values)):
            raise ValueError('Unknown or duplicate consequence dimension')
    if not policy['objectives']: raise ValueError('At least one objective required')
    if not isinstance(policy['constraints'], list) or len(policy['constraints']) > 30:
        raise ValueError('Bounded research constraints required')
    for c in policy['constraints']:
        if (not isinstance(c, dict) or set(c) != {'metric', 'scope', 'operator', 'value'}
                or c['metric'] not in METRICS or c['scope'] not in ('cumulative', 'incremental')
                or c['operator'] not in ('at_most', 'at_least')):
            raise ValueError('Unknown consequence constraint, scope or operator')
        _finite(c['value'])


def compare_samples(samples, policy, *, stop_branch, scenario_ids, branch_ids):
    """Pair every plan with every declared world; no averaging or probability.

    This pure comparison is downstream of canonical engine replays. The report
    builder and saved-artifact validator bind its data to source/native records.
    Missing/failed worlds may not be silently dropped from a plan.
    """
    validate_policy(policy)
    if not isinstance(samples, list) or not 1 <= len(samples) <= 64:
        raise ValueError('Bounded consequence samples required')
    if any(not isinstance(r, dict) or type(r.get('branchId')) is not str or not 0 < len(r['branchId']) <= 100
           or type(r.get('scenarioId')) is not str or not isinstance(r.get('vector'), dict) for r in samples):
        raise ValueError('Typed source-bound consequence samples required')
    if (not isinstance(scenario_ids, list) or not 1 <= len(scenario_ids) <= 8
            or any(type(s) is not str or not 0 < len(s) <= 100 for s in scenario_ids)
            or len(set(scenario_ids)) != len(scenario_ids)):
        raise ValueError('Explicit unique declared scenario IDs required')
    branches = sorted({r['branchId'] for r in samples}); worlds = sorted({r['scenarioId'] for r in samples})
    if (not isinstance(branch_ids, list) or len(set(branch_ids)) != len(branch_ids)
            or any(type(b) is not str or not 0 < len(b) <= 100 for b in branch_ids)
            or set(branches) != set(branch_ids)):
        raise ValueError('Every explicitly declared alternative must be retained')
    if set(worlds) != set(scenario_ids):
        raise ValueError('Declared scenario is missing or an undeclared world was added')
    if not 2 <= len(branches) <= 8 or not 1 <= len(worlds) <= 8 or stop_branch not in branches:
        raise ValueError('A bounded alternative set and explicit STOP branch are required')
    matrix = {(r['branchId'], r['scenarioId']): r for r in samples}
    if len(matrix) != len(samples) or len(matrix) != len(branches) * len(worlds):
        raise ValueError('Every alternative must retain every declared scenario exactly once')
    if len({r['vector']['sourceGeometryHash'] for r in samples}) != 1:
        raise ValueError('Compared scenarios must retain identical source geometry')
    for world in worlds:
        rows = [matrix[b, world]['vector'] for b in branches]
        if (len({r['sourceSceneId'] for r in rows}) != 1 or len({r['baselineStateId'] for r in rows}) != 1
                or len({r['prefixActions'] for r in rows}) != 1):
            raise ValueError('Paired scenarios must share the same scene and exact prefix')
    for row in samples:
        v = row['vector']
        if v.get('schema') != VECTOR_SCHEMA or set(v['dimensions']) != set(METRICS):
            raise ValueError('Complete typed consequence vector required')
        if type(v.get('cleanupComplete')) is not bool or v.get('clinicalValidation') is not False:
            raise ValueError('Exact cleanup boolean and nonclinical consequence scope required')
        if type(v.get('prefixActions')) is not int or not 0 <= v['prefixActions'] <= 10000:
            raise ValueError('Bounded integer prefix action count required')
        for name, (unit, direction) in METRICS.items():
            d = v['dimensions'][name]
            if d['unit'] != unit or d['direction'] != direction:
                raise ValueError('Consequence units or direction changed')
            if d['status'] == 'unavailable':
                if d['cumulative'] is not None or d['incremental'] is not None:
                    raise ValueError('Unknown consequence must retain null values')
            elif d['status'] in ('source_count', 'modeled_unvalidated'):
                if name in UNKNOWN:
                    raise ValueError('Unsupported evidence cannot become an available consequence')
                expected = 'source_count' if name in ('targetRemovedMm3', 'nonTargetRemovedMm3', 'elapsedSeconds') else 'modeled_unvalidated'
                if d['status'] != expected:
                    raise ValueError('Source count and unvalidated model status cannot be exchanged')
                if _finite(d['cumulative']) < 0 or _finite(d['incremental']) < 0:
                    raise ValueError('Source count, time and cumulative exposure cannot be negative')
            else: raise ValueError('Unknown consequence status')
    required = set(policy['objectives']) | set(policy['requiredMetrics']) | {c['metric'] for c in policy['constraints']}
    reasons = {}
    for branch in branches:
        violations = []
        for world in worlds:
            vector = matrix[branch, world]['vector']; dimensions = vector['dimensions']
            if not vector['cleanupComplete']: violations.append({'scenario': world, 'reason': 'cleanup_incomplete'})
            for name in sorted(required):
                if dimensions[name]['status'] == 'unavailable':
                    violations.append({'scenario': world, 'reason': 'unknown_required_consequence', 'metric': name})
            for c in policy['constraints']:
                value = dimensions[c['metric']][c['scope']]
                if value is not None and (value > c['value'] if c['operator'] == 'at_most' else value < c['value']):
                    violations.append({'scenario': world, 'reason': 'declared_research_constraint', **c, 'actual': value})
        reasons[branch] = violations
    # A valid STOP is an executed cleanup with zero additional source removal.
    stop_valid = all(matrix[stop_branch, w]['vector']['cleanupComplete'] and
        matrix[stop_branch, w]['vector']['dimensions']['targetRemovedMm3']['incremental'] == 0 and
        matrix[stop_branch, w]['vector']['dimensions']['nonTargetRemovedMm3']['incremental'] == 0 for w in worlds)
    eligible = [b for b in branches if b != stop_branch and not reasons[b]]
    def dominates(a, b):
        strict = False
        for world in worlds:
            for key in policy['objectives']:
                va = matrix[a, world]['vector']['dimensions'][key][objective_scope(key)]
                vb = matrix[b, world]['vector']['dimensions'][key][objective_scope(key)]
                sign = -1 if METRICS[key][1] == 'max' else 1
                if sign * va > sign * vb: return False
                strict |= sign * va < sign * vb
        return strict
    dominators = {b: [a for a in eligible if a != b and dominates(a, b)] for b in eligible}
    frontier = [b for b in eligible if not dominators[b]]
    selected = frontier[0] if len(frontier) == 1 else stop_branch if stop_valid else None
    return {'policy': copy.deepcopy(policy), 'scenarios': worlds, 'feasibleContinuations': eligible,
        'pairedScenarioParetoFront': frontier, 'dominatedBy': dominators, 'violations': reasons,
        'selectedBranchId': selected, 'stopFallback': selected == stop_branch,
        'stopExecutable': stop_valid, 'taskSatisfied': selected in eligible,
        'objectiveScopes': {key: objective_scope(key) for key in policy['objectives']},
        'reason': 'unique_paired_scenario_frontier' if len(frontier) == 1 else
            'no_executable_stop' if not stop_valid else 'unresolved_tradeoff_stop_with_residual' if frontier else 'no_qualified_continuation_stop_with_residual',
        'dominanceMeaning': 'No worse in every objective in every paired declared world; strictly better in at least one. No scalar weights, averages or clinical probabilities.'}


def seal_report(scene, samples, scenarios, policies, *, baseline_branch, stop_branch, source_archive_sha256, declaration):
    if (not isinstance(scenarios, list) or not 1 <= len(scenarios) <= 8
            or sum(s.get('id') == 'nominal' for s in scenarios) != 1
            or any(type(s.get('sourceSceneId')) is not str for s in scenarios)
            or not isinstance(policies, list) or not 1 <= len(policies) <= 8
            or len({p['id'] for p in policies}) != len(policies)):
        raise ValueError('A nominal world and nonempty unique research policies are required')
    declaration = comparison_declaration(declaration)
    report = {'schema': SCHEMA, 'sourceSceneId': scene.identity, 'sourceArchiveSha256': source_archive_sha256,
        'declarationHash': semantic_digest(declaration), 'declaration': declaration,
        'alternativeBranchIds': copy.deepcopy(declaration['alternativeBranches']),
        'sourceLabelsHash': array_digest(scene.labels), 'baselineBranchId': baseline_branch,
        'stopBranchId': stop_branch, 'sourceRole': 'DEVELOPMENT', 'scenarios': copy.deepcopy(scenarios),
        'samples': copy.deepcopy(samples), 'decisions': [compare_samples(samples, p, stop_branch=stop_branch,
            scenario_ids=[s['id'] for s in scenarios], branch_ids=declaration['alternativeBranches']) for p in policies],
        'optimizerUpdates': 0, 'physicalValidationPassed': False, 'physiologicalRL': 'hold',
        'limitations': ['Compared parameter worlds are declared sensitivity scenarios, not a patient distribution or confidence interval.',
            'Full cumulative source removal includes shared exposure. Incremental values describe the exact post-exposure decision prefix.',
            'STOP is checked cleanup with residual; it may leave the declared resection task unsatisfied.',
            'Model exposures are not injury or neurological outcome. Missing vascular, motor, language and acquired shift evidence remains unknown.']}
    report['comparisonSeal'] = semantic_digest(report)
    return report


def validate_saved_comparison(scene, record, report, *, declaration=None):
    """Check bindings and remeasure nominal source occupancy without new planning.

    The ordinary native path verifies causality when opening each branch. This
    integrity check is not an independent physical or counterfactual validation.
    """
    # App-loaded artifacts cannot choose a smaller experiment by editing their
    # self-computable seal. The app pins the expected declaration's hash in code,
    # so packaged builds need no repo-relative files or artifact-supplied paths.
    # Unit callers may pass a separately constructed trusted test declaration.
    if declaration is None:
        declaration = report.get('declaration')
        if not isinstance(declaration, dict) or semantic_digest(declaration) != APP_DECLARATION_HASH:
            raise ValueError('Consequence declaration is not the independently pinned app experiment')
    declaration = comparison_declaration(declaration)
    if (report.get('declarationHash') != semantic_digest(declaration)
            or report.get('declaration') != declaration
            or report.get('alternativeBranchIds') != declaration['alternativeBranches']
            or report.get('baselineBranchId') != declaration['prefixBranch']
            or report.get('stopBranchId') != declaration['stopBranch']
            or report.get('sourceArchiveSha256') != declaration['sourceArchiveSha256']):
        raise ValueError('Consequence report differs from independently expected experiment declaration')
    if (report.get('schema') != SCHEMA or report.get('sourceSceneId') != scene.identity
            or report.get('sourceLabelsHash') != array_digest(scene.labels)
            or report.get('optimizerUpdates') != 0 or report.get('physicalValidationPassed') is not False
            or report.get('sourceRole') != 'DEVELOPMENT'
            or report.get('physiologicalRL') != 'hold'
            or report.get('comparisonSeal') != semantic_digest({k: v for k, v in report.items() if k != 'comparisonSeal'})):
        raise ValueError('Consequence comparison source or seal changed')
    scenarios = report.get('scenarios'); decisions = report.get('decisions')
    if (not isinstance(scenarios, list) or not 1 <= len(scenarios) <= 8
            or any(not isinstance(s, dict) for s in scenarios)
            or sum(s.get('id') == 'nominal' for s in scenarios) != 1
            or not isinstance(decisions, list) or not 1 <= len(decisions) <= 8
            or any(not isinstance(d, dict) or not isinstance(d.get('policy'), dict) for d in decisions)
            or len({d['policy'].get('id') for d in decisions}) != len(decisions)):
        raise ValueError('Saved comparison requires nominal scenario and nonempty unique policies')
    baseline = record['branches'][report['baselineBranchId']]
    prefix = baseline['actions']; source = scene.labels.ravel()
    if (len(prefix) != declaration['prefixActions']
            or [{'id': s.get('id'), 'relaxationTimeScale': s.get('relaxationTimeScale')} for s in scenarios] != declaration['scenarios']
            or [d['policy'] for d in decisions] != declaration['policies']):
        raise ValueError('Consequence scenarios, policies or prefix differ from expected declaration')
    if any(branch not in record['branches'] for branch in declaration['alternativeBranches']):
        raise ValueError('Consequence native alternative is missing')
    # Validate the complete typed rectangle before any per-row metadata lookup.
    for decision in decisions:
        expected = compare_samples(report['samples'], decision['policy'], stop_branch=report['stopBranchId'],
            scenario_ids=[s['id'] for s in scenarios], branch_ids=declaration['alternativeBranches'])
        if decision != expected:
            raise ValueError('Consequence comparison decision differs from sealed vectors')
    targets = {r['id'] for r in scene.regions if r.get('target')}
    geometry = semantic_digest({'labels': array_digest(scene.labels), 'affine': scene.affine.tolist()})
    if declaration['sourceSceneId'] != scene.identity or declaration['sourceGeometryHash'] != geometry:
        raise ValueError('Consequence experiment belongs to different source anatomy or model')
    scenario_scenes = {s['id']: s.get('sourceSceneId') for s in scenarios}
    if scenario_scenes['nominal'] != scene.identity:
        raise ValueError('Nominal scenario scene differs from source')
    for s in scenarios:
        properties = copy.deepcopy(scene.properties)
        if 'brainContinuumV1' in properties:
            p = properties['brainContinuumV1']
            p['relaxationSeconds'] = [t * s['relaxationTimeScale'] for t in p['relaxationSeconds']]
        elif s['relaxationTimeScale'] != 1.:
            raise ValueError('Declared relaxation scenario requires explicit continuum')
        expected_scene = semantic_digest({'labels': array_digest(scene.labels), 'affine': scene.affine.tolist(),
            'regions': scene.regions, 'source': scene.source, 'properties': properties})
        if s.get('sourceSceneId') != expected_scene or s.get('profileHash') != semantic_digest(properties.get('brainContinuumV1')):
            raise ValueError('Consequence scenario profile or scene identity differs')
    for row in report['samples']:
        vector = row['vector']
        metadata = next(s for s in scenarios if s['id'] == row['scenarioId'])
        native = record['branches'][row['branchId']]
        if (vector.get('sourceGeometryHash') != geometry
                or vector.get('sourceSceneId') != scenario_scenes.get(row['scenarioId'])
                or vector.get('baselineStateId') != metadata.get('prefixStateId')
                or vector.get('prefixActions') != len(prefix)
                or vector.get('nativeActions') != len(native['actions'])
                or vector.get('actionHash') != semantic_digest(native['actions'])
                or vector.get('clinicalValidation') is not False):
            raise ValueError('Consequence vector geometry, scenario or nonclinical scope changed')
        if row['scenarioId'] != 'nominal': continue
        snapshot = record['branches'][row['branchId']]; vector = row['vector']
        if (snapshot['sceneId'] != scene.identity or snapshot['stateId'] != vector['stateId']
                or snapshot['actions'][:len(prefix)] != prefix or len(snapshot['actions']) != vector['nativeActions']
                or semantic_digest(snapshot['actions']) != vector['actionHash']
                or vector['baselineStateId'] != baseline['stateId'] or vector['prefixActions'] != len(prefix)):
            raise ValueError('Consequence comparison native branch or prefix changed')
        cells = snapshot['checkpoint']['state']['removed']; base_cells = baseline['checkpoint']['state']['removed']
        initial_target = int(np.count_nonzero(np.isin(source, list(targets)))) * scene.voxel_mm3
        residual = initial_target - sum(int(source[c]) in targets for c in cells) * scene.voxel_mm3
        if vector.get('residualTargetMm3') != residual:
            raise ValueError('Consequence residual target differs from source inventory')
        for entries, scope in [(cells, 'cumulative'), (set(cells) - set(base_cells), 'incremental')]:
            target = sum(int(source[c]) in targets for c in entries) * scene.voxel_mm3
            for name, value in [('targetRemovedMm3', target), ('nonTargetRemovedMm3', len(entries) * scene.voxel_mm3 - target)]:
                if vector['dimensions'][name][scope] != value:
                    raise ValueError('Consequence source occupancy differs')
        # Check the selected checkpoint's content and every nominal model ledger,
        # without treating that hash as proof of causality for an unselected branch.
        from .surgical_engine import SurgicalEngine
        def checkpoint_values(item):
            shell = SurgicalEngine(scene)
            shell.state = copy.deepcopy(item['checkpoint']['state'])
            shell._protection_contract = item.get('protectionContract')
            if shell.state_id != item['stateId']:
                raise ValueError('Consequence checkpoint content differs from state identity')
            return _values(shell), shell.state
        values, state = checkpoint_values(snapshot); initial, _ = checkpoint_values(baseline)
        for key in METRICS:
            d = vector['dimensions'][key]
            if d['cumulative'] != values[key] or d['incremental'] != (None if values[key] is None or initial[key] is None else values[key] - initial[key]):
                raise ValueError('Nominal consequence ledger differs from native checkpoint')
        cleanup = not state['holds'] and not any(s['active'] or s['inserted'] or s['grip'] for s in state['slots'].values())
        if vector['cleanupComplete'] != cleanup:
            raise ValueError('Consequence cleanup differs from native checkpoint')
