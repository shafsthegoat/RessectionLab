"""Source-bound consequence vectors and paired-scenario research comparisons.

There is no scalar reward or clinical acceptance here. Missing mechanisms stay
unknown; a STOP fallback does not pretend to meet the resection task. Scenario
dominance keeps each plan paired to the same declared parameter world.
"""
from __future__ import annotations

import copy
import math
import numpy as np

from .surgical_vascular_graph import maximum_interface_deficits
from .core import array_digest, semantic_digest
from .surgical_modeled_outcomes import summarize_engine_outcomes

SCHEMA = 'resectionlab.surgical-consequence-comparison.v1'
VECTOR_SCHEMA = 'resectionlab.surgical-consequence-vector.v1'
PROCEDURE_VECTOR_SCHEMA = 'resectionlab.surgical-consequence-vector.v2'
EVIDENCE_VECTOR_SCHEMA = 'resectionlab.surgical-consequence-vector.v3'
MECHANISM_EVIDENCE_REVISION = 2
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
PROCEDURE_METRICS = {name: ('mm3', 'min') for name in (
    'boneRemovedMm3', 'nonbonePreparationRemovedMm3', 'neuralRemovedMm3', 'protectedRemovedMm3', 'neuralCutSourceMm3')}
SOURCE_COUNTS = frozenset(('targetRemovedMm3', 'nonTargetRemovedMm3', 'elapsedSeconds'))
EXPOSURE_METRICS = ('brainStrainSquaredMm3S', 'brainStressSquaredPa2Mm3S',
    'thermalPeakC', 'thermalExposureMm3KS', 'bloodGeneratedMm3',
    'supplyDeficitSeconds', 'drainageDeficitSeconds')
_FLOW_DEFICITS = {'supplyDeficitSeconds': 'supply', 'drainageDeficitSeconds': 'drainage'}
_FUNCTIONAL_CONTACT = 'recordedContactCellSeconds'
_CONTINUUM_AGGREGATES = {'brainStrainSquaredMm3S': 'soft_tissue',
    'brainStressSquaredPa2Mm3S': 'soft_tissue', 'tumorStrainSquaredMm3S': 'tumor'}
PROCEDURE_PRIORITY = {
    'schema': 'resectionlab.procedure-selection-priority.v2',
    'order': ['declared-hard-bounds', 'maximum-worst-world-target-treatment',
        'componentwise-neural-and-structure-effects', 'componentwise-supported-exposures',
        'nonbone-preparation', 'bone', 'modeled-time'],
    'unresolved': 'Incomparable neural or exposure vectors require explicit unresolved STOP; bone/time cannot resolve them.',
    'thresholdMeaning': 'Declared engineering bounds only; no clinical weights or thresholds.'}


def procedure_priority_choice(branches, values, required, *, regional_exposures=False):
    """Prioritize checked outcomes without scalarizing neural/exposure effects."""
    remaining = list(branches)
    best = max(values(b, 'targetRemovedMm3') for b in remaining)
    remaining = [b for b in remaining if values(b, 'targetRemovedMm3') == best]
    neural = [k for k in ('neuralRemovedMm3', 'neuralCutSourceMm3', 'protectedRemovedMm3') if k in required]
    neural += sorted(k for k in required if k.startswith('structure:'))
    exposures = [k for k in EXPOSURE_METRICS if k in required]
    if regional_exposures:
        exposures += sorted(k for k in required if k.startswith(('functional:', 'mechanical:', 'vascular:')))
        if 'tumorStrainSquaredMm3S' in required:
            exposures.append('tumorStrainSquaredMm3S')
    for group, label in ((neural, 'neural'), (exposures, 'exposure')):
        def vector(branch):
            return tuple(values(branch, key) for key in group)
        remaining = [b for b in remaining if not any(
            all(x <= y for x, y in zip(vector(a), vector(b))) and
            any(x < y for x, y in zip(vector(a), vector(b))) for a in remaining)]
        if len({vector(b) for b in remaining}) > 1:
            return None, 'unresolved_' + label + '_tradeoff_stop_with_residual'
    for metric in ('nonbonePreparationRemovedMm3', 'boneRemovedMm3', 'elapsedSeconds'):
        if metric in required:
            best = min(values(b, metric) for b in remaining)
            remaining = [b for b in remaining if values(b, metric) == best]
    return sorted(remaining)[0], 'max_treatment_within_harm_bounds'


def _metric_spec(name):
    if name in METRICS:
        return METRICS[name]
    if name in PROCEDURE_METRICS:
        return PROCEDURE_METRICS[name]
    if (type(name) is str and name.startswith('structure:') and name.endswith((':removedMm3', ':cutSourceMm3')) and
            0 < len(name[len('structure:'):name.rfind(':')]) <= 100):
        return ('mm3', 'min')
    if type(name) is str:
        mechanism, _, region_metric = name.partition(':')
        region, _, metric = region_metric.rpartition(':')
        if 0 < len(region) <= 100:
            if mechanism == 'vascular' and metric in _FLOW_DEFICITS:
                return ('s', 'min')
            if mechanism == 'functional' and metric == _FUNCTIONAL_CONTACT:
                return ('cell*s', 'min')
            if mechanism == 'mechanical':
                from .surgical_mechanical_exposure import METRICS as mechanical_metrics
                if metric in mechanical_metrics:
                    return (mechanical_metrics[metric][0], 'min')
    raise ValueError('Unknown consequence dimension')


def _vector_metrics(vector):
    schema = vector.get('schema')
    revision = vector.get('mechanismEvidenceRevision')
    if revision is not None and (schema != EVIDENCE_VECTOR_SCHEMA or type(revision) is not int
                                 or revision != MECHANISM_EVIDENCE_REVISION):
        raise ValueError('Unsupported mechanism evidence revision')
    if schema == VECTOR_SCHEMA:
        return METRICS
    if schema not in (PROCEDURE_VECTOR_SCHEMA, EVIDENCE_VECTOR_SCHEMA):
        raise ValueError('Complete typed consequence vector required')
    ids = vector.get('structureIds', [])
    regional = {}
    if schema == EVIDENCE_VECTOR_SCHEMA:
        keys = vector.get('regionalMetricIds')
        prefixes = ('vascular:', 'mechanical:', 'functional:') if revision is not None else ('vascular:', 'mechanical:')
        if (type(keys) is not list or len(keys) > 512 or
                any(type(k) is not str or not k.startswith(prefixes) for k in keys)
                or len(set(keys)) != len(keys)):
            raise ValueError('Explicit regional mechanism dimensions required')
        regional = {key: _metric_spec(key) for key in keys}
        if 'procedurePolicyHash' not in vector:
            if ids:
                raise ValueError('Named source membership requires a bound procedure policy')
            return {**METRICS, **regional}
    if (type(ids) is not list or not ids or len(ids) != len(set(ids)) or
            any(type(name) is not str or not 0 < len(name) <= 100 for name in ids)):
        raise ValueError('Explicit structure identities required in procedure consequence vector')
    return {**METRICS, **PROCEDURE_METRICS, **regional,
        **{'structure:' + name + ':' + effect: ('mm3', 'min') for name in ids for effect in ('removedMm3', 'cutSourceMm3')}}


def _source_count(name):
    return name in SOURCE_COUNTS or name in PROCEDURE_METRICS or name.startswith('structure:')
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
        result.update(maximum_interface_deficits(vascular['regions']))
    elif any(r['material'] == 'vessel' and np.any(engine.scene.labels == r['id']) for r in engine.scene.regions):
        if 'vesselsV2' in engine.state:
            result['bloodGeneratedMm3'] = metrics['vesselPhysicsV2']['generatedMm3']
    return result


def _regional_dimension(value, before, unit, *, subject, mechanism, meaning):
    available = value is not None and before is not None
    return {'unit': unit, 'direction': 'min', 'subjectId': subject, 'mechanism': mechanism,
            'status': 'modeled_unvalidated' if available else 'unavailable',
            'cumulative': _finite(value) if available else None,
            'incremental': _finite(value) - _finite(before) if available else None,
            'incrementalMeaning': meaning,
            'reason': None if available else 'Measurement is unavailable in this model domain.'}


def _scope_model_dimension(dimension, before, *, source, aggregation, scope, meaning=None, reason=None):
    available = dimension['status'] != 'unavailable'
    dimension.update(evidenceClass='model_estimate', source=source, aggregation=aggregation,
                     baseline=_finite(before) if available else None, scope=scope)
    if meaning is not None:
        dimension['incrementalMeaning'] = meaning
    if reason is not None and not available:
        dimension['reason'] = reason


def _mechanical_unavailable_reason(row, metric):
    from .surgical_continuum import SOFT

    if row['material'] not in SOFT:
        return 'The declared source material is outside the soft mechanics model domain.'
    if metric in ('retractorForcePeakN', 'retractorForceTimeNs'):
        return 'Regional retractor force attribution is unavailable for a contact spanning source labels.'
    return 'The declared mechanics model does not measure this quantity.'


def _mechanism_evidence(engine, baseline, direct, *, revised):
    from .functional_evidence import functional_consequence_evidence
    from .surgical_vascular_graph import consequence_evidence as vascular_evidence
    from .surgical_mechanical_exposure import METRICS as mechanical_metrics, summarize

    vascular, vascular_before = vascular_evidence(engine), vascular_evidence(baseline)
    mechanical, mechanical_before = summarize(engine), summarize(baseline)
    functional = functional_consequence_evidence(engine, baseline, direct_dimensions=direct,
                                                all_declared_neural=revised)
    regional = {}
    if revised:
        for subject, row in functional['structures'].items():
            contact = row[_FUNCTIONAL_CONTACT]
            source = None if row['source'] is None else {**row['source'], 'sceneId': engine.scene.identity}
            dimension = _regional_dimension(contact['cumulative'], contact['baseline'], 'cell*s',
                subject=subject, mechanism=_FUNCTIONAL_CONTACT,
                meaning='Added source-indexed retained contact cell-time since the comparison prefix.')
            _scope_model_dimension(dimension, contact['baseline'], source=source,
                aggregation='time_integral', scope='Logged retained contact in the named source annotation.',
                reason=contact.get('reason'))
            regional[f'functional:{subject}:{_FUNCTIONAL_CONTACT}'] = dimension
    for subject, row in vascular['regions'].items():
        before = vascular_before['regions'].get(subject, {})
        for metric, interface in _FLOW_DEFICITS.items():
            dimension = _regional_dimension(
                row.get(metric), before.get(metric), 's', subject=subject, mechanism=metric,
                meaning='Added modeled flow-deficit exposure in the declared region since the comparison prefix.')
            if revised:
                source = {'sceneId': engine.scene.identity, 'model': vascular['model'],
                    'coverage': vascular['coverage'], 'provenance': row['source'], 'regionId': subject,
                    'edgeTerms': [{**term, 'kind': vascular['segments'][term['edge']]['kind']}
                                  for term in row[interface]],
                    'referenceFlowMm3PerSecond': row[f'reference{interface.capitalize()}Mm3PerSecond']}
                _scope_model_dimension(dimension, before.get(metric), source=source,
                    aggregation='time_integral', scope=f'Declared regional {interface} interface in the partial graph.')
            regional[f'vascular:{subject}:{metric}'] = dimension
    if mechanical is not None:
        regions = {str(region['id']): region for region in engine.scene.regions}
        for subject, row in mechanical['regions'].items():
            before = (mechanical_before or {}).get('regions', {}).get(subject, {})
            for metric, (unit, aggregation) in mechanical_metrics.items():
                dimension = _regional_dimension(
                    row.get(metric), before.get(metric), unit, subject=subject, mechanism=metric,
                    meaning=('Increase in the recorded cumulative peak since the comparison prefix.' if aggregation == 'peak'
                             else 'Added exposure since the comparison prefix.'))
                if revised:
                    region = regions[subject]
                    source = {'sceneId': engine.scene.identity, 'labelId': region['id'], 'name': region['name'],
                        'material': row['material'], 'model': mechanical['model'],
                        'retractorBoundary': mechanical['retractorBoundary'], 'coverage': 'source_region_label'}
                    _scope_model_dimension(dimension, before.get(metric), source=source,
                        aggregation='cumulative_peak' if aggregation == 'peak' else 'time_integral',
                        scope='Logged retained contact in the source region.' if metric == 'contactCellSeconds' else
                              'Accepted modeled steps in the source region.',
                        reason=_mechanical_unavailable_reason(row, metric) if dimension['status'] == 'unavailable' else None)
                regional[f'mechanical:{subject}:{metric}'] = dimension
    return {
        'functional': functional,
        'vascular': vascular,
        'mechanical': mechanical if mechanical is not None else {
            'status': 'unavailable', 'reason': 'Regional mechanical exposure was not enabled in this source.'},
        'clinicalOutcome': {'status': 'unavailable', 'reason': 'No admitted linked clinical outcome evidence.'},
    }, regional


def _scope_aggregate_mechanisms(engine, dimensions, before, vascular):
    graph_source = ({'sceneId': engine.scene.identity, 'model': vascular['model'], 'coverage': vascular['coverage']}
                    if 'model' in vascular else None)
    blood_source = graph_source
    if blood_source is None and dimensions['bloodGeneratedMm3']['status'] != 'unavailable':
        blood_source = {'sceneId': engine.scene.identity, 'model': 'vesselsV2', 'coverage': 'supplied_vessel_source_labels'}
    _scope_model_dimension(dimensions['bloodGeneratedMm3'], before['bloodGeneratedMm3'], source=blood_source,
        aggregation='cumulative_volume', scope='Blood generated into the modeled field by vessel wounds.',
        meaning='Additional modeled blood generated since the comparison prefix.',
        reason='No supported bleeding ledger bound to supplied vessel sources.')
    for metric, interface in _FLOW_DEFICITS.items():
        _scope_model_dimension(dimensions[metric], before[metric], source=graph_source,
            aggregation='maximum_regional_integral',
            scope=f'Maximum cumulative exposure across declared regional {interface} interfaces.',
            meaning='Change in the maximum cumulative regional deficit since the comparison prefix.',
            reason=f'No modeled regional {interface} interface with an explicit reference flow.')
    for metric, material in _CONTINUUM_AGGREGATES.items():
        labels = [row['id'] for row in engine.scene.regions if row['material'] == material]
        source = None if dimensions[metric]['status'] == 'unavailable' else {
            'sceneId': engine.scene.identity, 'model': 'brainContinuumV1', 'material': material,
            'labelIds': labels, 'coverage': 'source_region_labels'}
        _scope_model_dimension(dimensions[metric], before[metric], source=source, aggregation='time_integral',
            scope=f'Volume-time integral over the declared {material} label regions.',
            meaning='Added volume-time integral since the comparison prefix.',
            reason=f'No supported continuum ledger for the declared {material} material.')


def _scope_functional_counts(dimensions, functional):
    for subject, row in functional['structures'].items():
        source = row['source']
        for effect, scope in (('removedMm3', 'Unique removed source cells in the named annotation.'),
                              ('cutSourceMm3', 'Unique annotated source cells incident to a recorded cut or transection.')):
            if row[effect]['status'] == 'source_count':
                dimension = dimensions[f'structure:{subject}:{effect}']
                dimension.update(source=source, scope=scope,
                    evidenceClass='generated_evidence' if source['kind'] == 'generated_annotation' else 'supplied_annotation')


def consequence_vector(engine, baseline, *, extended=False, mechanism_revision=MECHANISM_EVIDENCE_REVISION):
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
    from .surgical_procedure_policy import PROPERTY
    procedure = PROPERTY in engine.scene.properties
    structure_ids = []
    complete = None
    if procedure:
        from .surgical_sequential_task import endpoint_physical_state
        complete = outcomes['sequence']['dimensions']
        for name in PROCEDURE_METRICS:
            d = complete[name]
            dimensions[name] = {**copy.deepcopy(d), 'direction': 'min'}
        for name, d in complete['structureRemovedMm3'].items():
            structure_ids.append(name)
            dimensions['structure:' + name + ':removedMm3'] = {**copy.deepcopy(d), 'direction': 'min'}
            dimensions['structure:' + name + ':cutSourceMm3'] = {**copy.deepcopy(complete['structureCutSourceMm3'][name]), 'direction': 'min'}
    vector = {'schema': PROCEDURE_VECTOR_SCHEMA if procedure else VECTOR_SCHEMA,
        **({'structureIds': structure_ids, 'procedurePolicyHash': engine.scene.properties[PROPERTY]['policyHash']} if procedure else {}),
        'sourceSceneId': engine.scene.identity,
        'sourceGeometryHash': semantic_digest({'labels': array_digest(engine.scene.labels), 'affine': engine.scene.affine.tolist()}),
        'stateId': engine.state_id, 'baselineStateId': baseline.state_id,
        'prefixActions': len(baseline.actions), 'nativeActions': len(engine.actions),
        'actionHash': semantic_digest(engine.actions),
        'cleanupComplete': (endpoint_physical_state(engine)['cleanupComplete'] if procedure else
            not engine.state['holds'] and not any(s['active'] or s['inserted'] or s['grip'] for s in engine.state['slots'].values())),
        'residualTargetMm3': outcomes['target'].get('residualMm3'),
        'dimensions': dimensions, 'clinicalValidation': False,
        'scope': 'Source annotation counts and explicitly modeled consequences; not injury, histological margins or neurological outcome.'}
    if extended:
        if mechanism_revision is not None and (type(mechanism_revision) is not int or
                                               mechanism_revision != MECHANISM_EVIDENCE_REVISION):
            raise ValueError('Unsupported mechanism evidence revision')
        revised = mechanism_revision is not None
        evidence, regional = _mechanism_evidence(engine, baseline, complete, revised=revised)
        vector.update(schema=EVIDENCE_VECTOR_SCHEMA, mechanismEvidence=evidence,
                      regionalMetricIds=sorted(regional))
        dimensions.update(regional)
        if revised:
            vector['mechanismEvidenceRevision'] = mechanism_revision
            _scope_aggregate_mechanisms(engine, dimensions, before, evidence['vascular'])
            _scope_functional_counts(dimensions, evidence['functional'])
        if procedure:
            from .surgical_treatment_intent import procedure_treatment_intent
            intent = procedure_treatment_intent(engine)
            initial_intent = procedure_treatment_intent(baseline)
            if intent['eligibleRemovedMm3'] is not None and initial_intent['eligibleRemovedMm3'] is not None:
                intent['incrementalEligibleRemovedMm3'] = (
                    intent['eligibleRemovedMm3'] - initial_intent['eligibleRemovedMm3'])
            vector['treatmentIntent'] = intent
            if intent['residual']['residualMm3'] is not None:
                vector['residualTargetMm3'] = intent['residual']['residualMm3']
    return vector


def validate_policy(policy):
    if (not isinstance(policy, dict) or set(policy) != {'id', 'basis', 'objectives', 'constraints', 'requiredMetrics'}
            or policy['basis'] != 'declared_research_task_not_clinical_threshold'
            or not isinstance(policy['id'], str) or not 0 < len(policy['id']) <= 100):
        raise ValueError('Explicit bounded research comparison policy required')
    for key in ('objectives', 'requiredMetrics'):
        values = policy[key]
        if (not isinstance(values, list) or len(values) > 200
                or any(type(v) is not str for v in values)
                or len(set(values)) != len(values)):
            raise ValueError('Unknown or duplicate consequence dimension')
        for name in values:
            _metric_spec(name)
    if not policy['objectives']: raise ValueError('At least one objective required')
    if not isinstance(policy['constraints'], list) or len(policy['constraints']) > 200:
        raise ValueError('Bounded research constraints required')
    for c in policy['constraints']:
        if (not isinstance(c, dict) or set(c) != {'metric', 'scope', 'operator', 'value'}
                or type(c['metric']) is not str or c['scope'] not in ('cumulative', 'incremental')
                or c['operator'] not in ('at_most', 'at_least')):
            raise ValueError('Unknown consequence constraint, scope or operator')
        _finite(c['value'])
        _metric_spec(c['metric'])


def _validate_mechanism_dimension(name, dimension, scene_id):
    mechanism, _, suffix = name.partition(':')
    subject, _, metric = suffix.rpartition(':')
    if name in _FLOW_DEFICITS:
        aggregation = 'maximum_regional_integral'
    elif name == 'bloodGeneratedMm3':
        aggregation = 'cumulative_volume'
    elif name in _CONTINUUM_AGGREGATES or mechanism in ('functional', 'vascular'):
        aggregation = 'time_integral'
    elif mechanism == 'mechanical':
        from .surgical_mechanical_exposure import METRICS as mechanical_metrics
        aggregation = 'cumulative_peak' if mechanical_metrics[metric][1] == 'peak' else 'time_integral'
    else:
        return

    source = dimension.get('source')
    if (dimension.get('aggregation') != aggregation or dimension.get('evidenceClass') != 'model_estimate'
            or type(dimension.get('scope')) is not str or not dimension['scope']
            or type(dimension.get('incrementalMeaning')) is not str or not dimension['incrementalMeaning']
            or (subject and (dimension.get('subjectId') != subject or dimension.get('mechanism') != metric))):
        raise ValueError('Modeled consequence scope or identity differs from its metric')
    if source is not None and (type(source) is not dict or source.get('sceneId') != scene_id):
        raise ValueError('Modeled consequence source differs from its scene')
    if dimension['status'] == 'unavailable':
        if dimension.get('baseline') is not None or type(dimension.get('reason')) is not str or not dimension['reason']:
            raise ValueError('Unavailable mechanism requires an explicit reason and null baseline')
    elif (source is None or _finite(dimension.get('baseline')) < 0.
            or dimension['cumulative'] - dimension['baseline'] != dimension['incremental']):
        raise ValueError('Modeled consequence source or prefix accounting differs')


def compare_samples(samples, policy, *, stop_branch, scenario_ids, branch_ids, selection_rule=None,
                    preservation_preference=None):
    """Pair every plan with every declared world; no averaging or probability.

    This pure comparison is downstream of canonical engine replays. The report
    builder and saved-artifact validator bind its data to source/native records.
    Missing/failed worlds may not be silently dropped from a plan.
    """
    validate_policy(policy)
    from .surgical_treatment_intent import PRESERVATION_SELECTION_RULE
    preservation = selection_rule == PRESERVATION_SELECTION_RULE
    if selection_rule not in (None, 'max_treatment_within_harm_bounds', PRESERVATION_SELECTION_RULE):
        raise ValueError('Unknown consequence selection rule')
    if preservation_preference is not None and not preservation:
        raise ValueError('Named preservation preference requires the versioned preservation rule')
    if preservation_preference is not None:
        from .surgical_treatment_intent import validate_preservation_preference
        validate_preservation_preference(preservation_preference)
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
    maximum_branches = 25 if selection_rule is not None else 8
    if not 2 <= len(branches) <= maximum_branches or not 1 <= len(worlds) <= 8 or stop_branch not in branches:
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
    revisions = set()
    mechanism_baselines = {}
    for row in samples:
        v = row['vector']
        metric_contract = _vector_metrics(v)
        revisions.add(v.get('mechanismEvidenceRevision'))
        if preservation:
            _validate_treatment_vector(v)
        if set(v['dimensions']) != set(metric_contract):
            raise ValueError('Complete typed consequence vector required')
        if type(v.get('cleanupComplete')) is not bool or v.get('clinicalValidation') is not False:
            raise ValueError('Exact cleanup boolean and nonclinical consequence scope required')
        if type(v.get('prefixActions')) is not int or not 0 <= v['prefixActions'] <= 10000:
            raise ValueError('Bounded integer prefix action count required')
        for name, (unit, direction) in metric_contract.items():
            d = v['dimensions'][name]
            if d['unit'] != unit or d['direction'] != direction:
                raise ValueError('Consequence units or direction changed')
            if d['status'] == 'unavailable':
                if d['cumulative'] is not None or d['incremental'] is not None:
                    raise ValueError('Unknown consequence must retain null values')
            elif d['status'] in ('source_count', 'modeled_unvalidated'):
                if name in UNKNOWN:
                    raise ValueError('Unsupported evidence cannot become an available consequence')
                expected = 'source_count' if _source_count(name) else 'modeled_unvalidated'
                if d['status'] != expected:
                    raise ValueError('Source count and unvalidated model status cannot be exchanged')
                if _finite(d['cumulative']) < 0 or _finite(d['incremental']) < 0:
                    raise ValueError('Source count, time and cumulative exposure cannot be negative')
            else: raise ValueError('Unknown consequence status')
            if v.get('mechanismEvidenceRevision') == MECHANISM_EVIDENCE_REVISION:
                _validate_mechanism_dimension(name, d, v['sourceSceneId'])
                if d['status'] == 'modeled_unvalidated' and 'baseline' in d:
                    previous = mechanism_baselines.setdefault((row['scenarioId'], name), d['baseline'])
                    if d['baseline'] != previous:
                        raise ValueError('Modeled consequences require the same baseline values in each scenario')
    if len(revisions) > 1:
        raise ValueError('Compared consequence vectors require the same mechanism evidence revision')
    required = set(policy['objectives']) | set(policy['requiredMetrics']) | {c['metric'] for c in policy['constraints']}
    if any(not required <= row['vector']['dimensions'].keys() for row in samples):
        raise ValueError('Required consequence dimension absent from versioned vector')
    if selection_rule is not None and (any(row['vector'].get('schema') not in (PROCEDURE_VECTOR_SCHEMA, EVIDENCE_VECTOR_SCHEMA) for row in samples) or
            'targetRemovedMm3' not in policy['objectives']):
        raise ValueError('Treatment-within-harm selection requires versioned procedure outcomes and a target objective')
    if preservation and any(row['vector']['schema'] != EVIDENCE_VECTOR_SCHEMA for row in samples):
        raise ValueError('Named preservation decisions require the evidence-aware consequence vector')
    if selection_rule is not None and (len({row['vector'].get('procedurePolicyHash') for row in samples}) != 1 or
            any(type(row['vector'].get('procedurePolicyHash')) is not str for row in samples)):
        raise ValueError('Compared procedure outcomes must retain the same declared policy binding')
    reasons = {}
    for branch in branches:
        violations = []
        for world in worlds:
            vector = matrix[branch, world]['vector']; dimensions = vector['dimensions']
            if not vector['cleanupComplete']: violations.append({'scenario': world, 'reason': 'cleanup_incomplete'})
            if (selection_rule is not None and branch != stop_branch and
                    dimensions['targetRemovedMm3']['incremental'] is not None and
                    dimensions['targetRemovedMm3']['incremental'] <= 0):
                violations.append({'scenario': world, 'reason': 'no_additional_target_treatment'})
            for name in sorted(required):
                if dimensions[name]['status'] == 'unavailable':
                    violations.append({'scenario': world, 'reason': 'unknown_required_consequence', 'metric': name})
            for c in policy['constraints']:
                value = dimensions[c['metric']][c['scope']]
                tolerance = 1e-9 if selection_rule is not None else 0.
                if value is not None and (value > c['value'] + tolerance if c['operator'] == 'at_most' else value < c['value'] - tolerance):
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
                sign = -1 if _metric_spec(key)[1] == 'max' else 1
                if sign * va > sign * vb: return False
                strict |= sign * va < sign * vb
        return strict
    dominators = {b: [a for a in eligible if a != b and dominates(a, b)] for b in eligible}
    frontier = [b for b in eligible if not dominators[b]]
    selected = frontier[0] if len(frontier) == 1 else stop_branch if stop_valid else None
    selection_reason, intent_decision = None, None
    if preservation:
        intent_decision = _preservation_choice(matrix, branches, worlds, reasons, required,
                                               stop_branch, stop_valid, preservation_preference)
        selected = intent_decision['selectedId']
        if stop_valid and intent_decision['status'] in ('stop_with_residual', 'stop_after_treatment'):
            selected = stop_branch
        selection_reason = intent_decision['reason']
    elif selection_rule is not None and frontier:
        def worst_value(branch, metric):
            values = [matrix[branch, world]['vector']['dimensions'][metric][objective_scope(metric)] for world in worlds]
            return min(values) if metric == 'targetRemovedMm3' else max(values)
        selected, selection_reason = procedure_priority_choice(frontier, worst_value, required,
            regional_exposures=revisions == {MECHANISM_EVIDENCE_REVISION})
        if selected is None and stop_valid:
            selected = stop_branch
    return {'policy': copy.deepcopy(policy), 'scenarios': worlds, 'feasibleContinuations': eligible,
        'pairedScenarioParetoFront': frontier, 'dominatedBy': dominators, 'violations': reasons,
        'selectedBranchId': selected, 'stopFallback': selected == stop_branch,
        'stopExecutable': stop_valid, 'taskSatisfied': selected in eligible,
        'objectiveScopes': {key: objective_scope(key) for key in policy['objectives']},
        **({'selectionRule': selection_rule} if selection_rule is not None else {}),
        **({'selectionPriority': copy.deepcopy(PROCEDURE_PRIORITY)} if selection_rule == 'max_treatment_within_harm_bounds' else {}),
        **({'treatmentDecision': intent_decision} if intent_decision is not None else {}),
        **({'frontierMeaning': 'The objective frontier uses the declared comparison objectives. The treatment decision retains its separate named-harm frontiers.'}
           if preservation else {}),
        'reason': selection_reason if selection_reason else
            'unique_paired_scenario_frontier' if len(frontier) == 1 else
            'no_executable_stop' if not stop_valid else 'unresolved_tradeoff_stop_with_residual' if frontier else 'no_qualified_continuation_stop_with_residual',
        'dominanceMeaning': 'No worse in every objective in every paired declared world; strictly better in at least one. No scalar weights, averages or clinical probabilities.'}


def _validate_treatment_vector(vector):
    from .surgical_treatment_intent import SOURCE_INTENT_SCHEMA
    intent = vector.get('treatmentIntent')
    if (not isinstance(intent, dict) or intent.get('schema') != SOURCE_INTENT_SCHEMA
            or intent.get('sourceSceneId') != vector['sourceSceneId'] or intent.get('stateId') != vector['stateId']
            or intent.get('policyHash') != vector.get('procedurePolicyHash')
            or type(intent.get('coverageComplete')) is not bool or not isinstance(intent.get('residual'), dict)
            or intent['residual'].get('residualMm3') != vector.get('residualTargetMm3')):
        raise ValueError('Treatment intent must bind source, state, policy and original residual')
    if intent['coverageComplete'] and (
            not intent.get('lesionId') or not intent.get('lesionRevision')
            or _finite(intent.get('incrementalEligibleRemovedMm3')) < 0):
        raise ValueError('Complete treatment intent requires a lesion revision and eligible treatment credit')


def _preservation_choice(matrix, branches, worlds, reasons, required, stop_branch, stop_valid, preference):
    from .surgical_treatment_intent import decide_treatment_intent
    harm_keys = sorted(required - {'targetRemovedMm3', 'elapsedSeconds'})
    preference_keys = (preference or {}).get('metrics', ())
    if any(key not in harm_keys for key in preference_keys):
        raise ValueError('Named preservation metrics must appear in the declared comparison policy')

    def world_harms(keys):
        return [f'{world}/{key}' for world in worlds for key in keys]

    def intent_row(branch):
        rows = {world: matrix[branch, world]['vector']['dimensions'] for world in worlds}
        intents = {world: matrix[branch, world]['vector']['treatmentIntent'] for world in worlds}
        missing = (any(r['reason'] == 'unknown_required_consequence' for r in reasons[branch])
            or any(not intent['coverageComplete'] for intent in intents.values()))
        return {'id': branch, 'admissible': None if missing else not reasons[branch],
                'treatmentMm3': {world: intent.get('incrementalEligibleRemovedMm3')
                                 for world, intent in intents.items()},
                'harms': {f'{world}/{key}': dims[key]['cumulative']
                          for world, dims in rows.items() for key in harm_keys},
                'blockingBoundaries': []}

    stop_intent = [matrix[stop_branch, w]['vector']['treatmentIntent'] for w in worlds]
    residual = stop_intent[0]['residual'] if all(row['residual'] == stop_intent[0]['residual'] for row in stop_intent) else None
    stop_harm_violation = any(
        row.get('metric') != 'targetRemovedMm3' or row.get('operator') != 'at_least'
        for row in reasons[stop_branch])
    complete = all(row['coverageComplete'] for row in stop_intent)
    incumbent = intent_row(stop_branch) if stop_valid and not stop_harm_violation and complete else None
    if incumbent is not None:
        # The treatment floor applies to continuations, while STOP carries
        # measured prefix harm and the unchanged original lesion denominator.
        incumbent['admissible'] = True
        incumbent['treatmentMm3'] = {world: 0. for world in worlds}
    return decide_treatment_intent(
        [intent_row(b) for b in branches if b != stop_branch], incumbent=incumbent,
        required_harms=world_harms(harm_keys), preference_harms=world_harms(preference_keys),
        residual=residual, coverage_complete=complete)


def procedure_comparison_options(scene):
    from .surgical_procedure_policy import PROPERTY, procedure_preservation_preference
    rule = scene.properties.get(PROPERTY, {}).get('selectionRule')
    if rule in (None, 'max_treatment_within_harm_bounds'):
        return {'selection_rule': rule}
    return {'selection_rule': rule, 'preservation_preference': procedure_preservation_preference(scene)}


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
            scenario_ids=[s['id'] for s in scenarios], branch_ids=declaration['alternativeBranches'],
            **procedure_comparison_options(scene)) for p in policies],
        'optimizerUpdates': 0, 'physicalValidationPassed': False, 'physiologicalRL': 'hold',
        'limitations': ['Compared parameter worlds are declared sensitivity scenarios, not a patient distribution or confidence interval.',
            'Full cumulative source removal includes shared exposure. Incremental values describe the exact post-exposure decision prefix.',
            'STOP is checked cleanup with residual; it may leave the declared resection task unsatisfied.',
            'Model exposures are not injury or neurological outcome. Missing vascular, motor, language and acquired shift evidence remains unknown.']}
    report['comparisonSeal'] = semantic_digest(report)
    return report


def consequence_checkpoint_engine(scene, snapshot):
    """Read a source-bound checkpoint without claiming causal replay."""
    from .surgical_engine import SurgicalEngine
    checkpoint = snapshot['checkpoint']
    if (snapshot['sceneId'] != scene.identity or checkpoint['stateId'] != snapshot['stateId'] or
            checkpoint['position'] != len(snapshot['actions'])):
        raise ValueError('Consequence checkpoint source or action count differs')
    options = {}
    if 'protectionContract' in snapshot:
        from .surgical_rl_protection import restore_protection_contract
        protection, guard = restore_protection_contract(scene, snapshot['protectionContract'])
        options.update(protection=protection, guard_config=guard)
    if 'cranialStagesContract' in snapshot:
        options['cranial_stages'] = snapshot['cranialStagesContract']
    shell = SurgicalEngine(scene, **options)
    shell.state = copy.deepcopy(checkpoint['state'])
    shell.actions = copy.deepcopy(snapshot['actions'])
    shell._protection_contract = snapshot.get('protectionContract')
    if shell.state_id != snapshot['stateId']:
        raise ValueError('Consequence checkpoint content differs from state identity')
    return shell


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
        from .surgical_procedure_policy import PROPERTY
        expected = compare_samples(report['samples'], decision['policy'], stop_branch=report['stopBranchId'],
            scenario_ids=[s['id'] for s in scenarios], branch_ids=declaration['alternativeBranches'],
            **procedure_comparison_options(scene))
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
        if vector['schema'] != EVIDENCE_VECTOR_SCHEMA and vector.get('residualTargetMm3') != residual:
            raise ValueError('Consequence residual target differs from source inventory')
        for entries, scope in [(cells, 'cumulative'), (set(cells) - set(base_cells), 'incremental')]:
            target = sum(int(source[c]) in targets for c in entries) * scene.voxel_mm3
            for name, value in [('targetRemovedMm3', target), ('nonTargetRemovedMm3', len(entries) * scene.voxel_mm3 - target)]:
                if vector['dimensions'][name][scope] != value:
                    raise ValueError('Consequence source occupancy differs')
        # Check the selected checkpoint's content and every nominal model ledger,
        # without treating that hash as proof of causality for an unselected branch.
        endpoint_shell = consequence_checkpoint_engine(scene, snapshot)
        initial_shell = consequence_checkpoint_engine(scene, baseline)
        values, state, initial = _values(endpoint_shell), endpoint_shell.state, _values(initial_shell)
        for key in METRICS:
            d = vector['dimensions'][key]
            if d['cumulative'] != values[key] or d['incremental'] != (None if values[key] is None or initial[key] is None else values[key] - initial[key]):
                raise ValueError('Nominal consequence ledger differs from native checkpoint')
        if 'procedurePolicyHash' in vector:
            from .surgical_procedure_policy import direct_removal_dimensions
            if vector.get('procedurePolicyHash') != scene.properties[PROPERTY]['policyHash']:
                raise ValueError('Consequence procedure policy binding differs')
            direct = direct_removal_dimensions(endpoint_shell, initial_shell)
            measured = {key: direct[key] for key in PROCEDURE_METRICS}
            for structure in vector['structureIds']:
                measured['structure:' + structure + ':removedMm3'] = direct['structureRemovedMm3'][structure]
                measured['structure:' + structure + ':cutSourceMm3'] = direct['structureCutSourceMm3'][structure]
            for key, measurement in measured.items():
                if any(vector['dimensions'][key].get(field) != measurement.get(field)
                        for field in ('status', 'cumulative', 'incremental')):
                    raise ValueError('Nominal direct structure consequence differs from source checkpoint')
        cleanup = not state['holds'] and not any(s['active'] or s['inserted'] or s['grip'] for s in state['slots'].values())
        if 'procedurePolicyHash' in vector:
            from .surgical_sequential_task import endpoint_physical_state
            cleanup = endpoint_physical_state(endpoint_shell)['cleanupComplete']
        if vector['cleanupComplete'] != cleanup:
            raise ValueError('Consequence cleanup differs from native checkpoint')
        if vector['schema'] == EVIDENCE_VECTOR_SCHEMA:
            from .surgical_engine import SurgicalEngine
            endpoint_engine = SurgicalEngine.restore(scene, snapshot, force_verify=True)
            initial_engine = SurgicalEngine.restore(scene, baseline, force_verify=True)
            if consequence_vector(endpoint_engine, initial_engine, extended=True,
                                   mechanism_revision=vector.get('mechanismEvidenceRevision')) != vector:
                raise ValueError('Nominal regional or source evidence differs from native replay')
