"""Opt-in stimulation-mapping estimate and structure-aware harm (``functionalMappingV1``).

A scene that declares ``functionalMappingV1`` names its tract regions (tract
bundles fitted from a published baseline tract, so ESTIMATES of where a tract
lies, never patient function). A probe then reports a labelled MODEL ESTIMATE
of whether a motor response would be evoked, from the distance between the tip
and the nearest retained cell of each declared tract and a declared
distance-current relation. Missing tract evidence, a tip outside tract coverage
or a family that needs an awake patient stays ``unknown``: no response is never
reported as safety. Nothing here is a deficit probability, and ischaemic
territory is not simulated (it is reported UNKNOWN, never zero).

Scenes without the property never reach this module and keep identical state.
"""
from __future__ import annotations

import copy
import math
import re

import numpy as np
from scipy.spatial import cKDTree

from .core import semantic_digest

PROPERTY = 'functionalMappingV1'
SCHEMA = 'resectionlab.functional-mapping.v1'
FAMILIES = ('motor', 'language', 'vision', 'other')
VESSEL_KINDS = ('artery', 'vein', 'unclassified')
_PROPERTY_KEYS = {'tracts', 'awakeScenario', 'maxSearchMm', 'ischaemicTerritory', 'note'}

# Function family by whole-word tract-name token (plural 's' dropped). Tract
# anatomy is not function: the family says which mapping task could test it.
_FAMILY_TOKENS = (
    ('motor', {'cst', 'corticospinal', 'pyramidal'}),
    ('language', {'arcuate', 'af', 'ifof', 'ifo', 'aslant', 'fat', 'uncinate', 'uf'}),
    ('vision', {'optic', 'or'}),
    ('other', {'cingulum', 'cg'}),
)

# Owner-supplied relation, source to be confirmed. Stimulation is assumed to
# follow it at the declared reference setting below (assumed-research).
RESPONSE_LAW = {
    'id': 'motor-distance-current-linear-v1',
    'relation': 'distance_mm = 1.42 + 0.81 * current_mA',
    'interceptMm': 1.42, 'slopeMmPerMa': 0.81,
    'appliesTo': 'motor (corticospinal) response to monopolar stimulation at the declared reference setting',
    'provenance': 'owner-supplied',
    'source': "Supplied by the project owner as one published series' fit; the source is to be confirmed. No citation is bound here.",
    'secondaryReference': {
        'relation': 'distance_mm = 1.0 * current_mA', 'interceptMm': 0., 'slopeMmPerMa': 1.,
        'provenance': 'assumed-research',
        'source': 'Commonly taught rule of roughly 1 mm per mA; no citation is bound here.'},
    'decision': 'The owner-supplied relation decides the result; the secondary reference only widens the stated uncertainty.',
}
# assumed-research: effective current through charge per phase (Weiss/Lapicque
# strength-duration form, threshold charge = rheobase * (pulse width + chronaxie))
# and a bounded temporal-summation factor for the train frequency. Monotone and
# bounded; neither constant is calibrated against patient stimulation.
CHARGE_SCALING = {
    'referencePulseWidthUs': 500., 'referenceFrequencyHz': 60., 'chronaxieUs': 200.,
    'pulseWidthFactorBounds': [.25, 1.4], 'frequencyExponent': .1, 'frequencyFactorBounds': [.8, 1.2],
    'pulseWidthFactor': '(1 + chronaxie/referencePulseWidth) / (1 + chronaxie/pulseWidth), clipped to its bounds',
    'frequencyFactor': '(frequency/referenceFrequency) ** frequencyExponent, clipped to its bounds',
    'effectiveCurrent': 'amplitude * pulseWidthFactor * frequencyFactor',
    'provenance': 'assumed-research',
}
DEFAULT_MAX_SEARCH_MM = 20.  # assumed-research: tract coverage around the tip.
ISCHAEMIC_TERRITORY = {'status': 'unknown', 'value': None,
                       'reason': 'Ischaemic territory of a sacrificed or injured vessel is not simulated; it is unknown, not zero.'}
PATIENT_OUTCOME = {'status': 'unknown', 'deficitProbability': None,
                   'reason': 'no_validated_clinical_outcome_model: tract distance and removal are not deficit probabilities.'}
LIMITS = ('Tractography is not function: a tract region is an estimate fitted from a published baseline tract.',
          'Every response is a model estimate from a declared distance-current relation, never an observed stimulation.',
          'Distances use undeformed source cell centres; brain shift and tract deformation are not modelled.',
          'No response does not establish absence of function. Language and vision need an awake patient performing tasks.',
          'No deficit probability is produced, and ischaemic territory is not simulated (unknown).')

_CACHE = {}
_CACHE_LIMIT = 8


def declaration(tracts=None, *, awake=False, max_search_mm=DEFAULT_MAX_SEARCH_MM):
    """The property value a caller adds to a scene to opt in."""
    result = {'ischaemicTerritory': 'unknown', 'awakeScenario': bool(awake), 'maxSearchMm': float(max_search_mm)}
    if tracts is not None:
        result['tracts'] = copy.deepcopy(list(tracts))
    return result


def family_for(name):
    tokens = {token[:-1] if len(token) > 3 and token.endswith('s') else token
              for token in re.split(r'[^a-z0-9]+', str(name).lower()) if token}
    for family, names in _FAMILY_TOKENS:
        if tokens & names:
            return family
    return 'unclassified'


def vessel_kind(region):
    kind = region.get('vesselKind')
    if kind in VESSEL_KINDS:
        return kind
    name = str(region.get('name', '')).lower()
    if 'arter' in name:
        return 'artery'
    if 'vein' in name or 'venous' in name or 'sinus' in name:
        return 'vein'
    return 'unclassified'


def _strip(name):
    return re.sub(r'\s*\(tract\)\s*$', '', str(name)).strip() or str(name)


def _validate(scene):
    value = scene.properties.get(PROPERTY)
    if not isinstance(value, dict) or set(value) - _PROPERTY_KEYS:
        raise ValueError('FUNCTIONAL_MAPPING_PROPERTY_INVALID: allowed keys are ' + ', '.join(sorted(_PROPERTY_KEYS)))
    if value.get('ischaemicTerritory') != 'unknown':
        raise ValueError("FUNCTIONAL_MAPPING_ISCHAEMIC_TERRITORY: declare ischaemicTerritory: 'unknown'; it is not simulated")
    if type(value.get('awakeScenario', False)) is not bool:
        raise ValueError('FUNCTIONAL_MAPPING_AWAKE_SCENARIO_MUST_BE_BOOLEAN')
    search = value.get('maxSearchMm', DEFAULT_MAX_SEARCH_MM)
    if type(search) not in (int, float) or not math.isfinite(search) or search <= 0:
        raise ValueError('FUNCTIONAL_MAPPING_MAX_SEARCH_MM_MUST_BE_POSITIVE')
    return value


def _tract_rows(scene, value):
    by_id = {region['id']: region for region in scene.regions}
    by_name = {region['name']: region for region in scene.regions}
    supplied = value.get('tracts')
    if supplied is None:
        supplied = [region['name'] for region in scene.regions if str(region['name']).rstrip().endswith('(tract)')]
    if not isinstance(supplied, list):
        raise ValueError('FUNCTIONAL_MAPPING_TRACTS_MUST_BE_A_LIST')
    rows, seen = [], set()
    for entry in supplied:
        spec = entry if isinstance(entry, dict) else {'region': entry}
        key = spec.get('region')
        region = by_id.get(key) if type(key) is int else by_name.get(key)
        if region is None:
            raise ValueError('FUNCTIONAL_MAPPING_TRACT_REGION_UNKNOWN: ' + str(key))
        if region['material'] == 'vessel':
            raise ValueError('FUNCTIONAL_MAPPING_TRACT_IS_A_VESSEL: ' + str(region['name']))
        identity = str(spec.get('id', _strip(region['name'])))
        family = spec.get('family', family_for(region['name']))
        if family not in FAMILIES + ('unclassified',):
            raise ValueError('FUNCTIONAL_MAPPING_FAMILY_UNKNOWN: ' + str(family))
        if identity in seen or not identity or ':' in identity:
            raise ValueError('FUNCTIONAL_MAPPING_TRACT_ID_INVALID: ' + identity)
        seen.add(identity)
        rows.append({'id': identity, 'regionId': region['id'], 'regionName': region['name'], 'family': family,
                     'provenance': spec.get('provenance', 'estimated'),
                     'evidence': 'ESTIMATE: tract region fitted from a published baseline tract; not patient function'})
    return rows


class _Model:
    """Static per-scene tract and vessel geometry, built once and kept off the scene."""

    def __init__(self, scene, value):
        labels = scene.labels.ravel()
        self.value = value
        self.max_search = float(value.get('maxSearchMm', DEFAULT_MAX_SEARCH_MM))
        self.awake = value.get('awakeScenario', False)
        from .surgical_rl_protection import cell_radius
        self.grid_mm = cell_radius(scene.affine)
        self.voxel = scene.voxel_mm3
        self.tracts = _tract_rows(scene, value)
        self.label_class = {}
        for row in self.tracts:
            cells = np.flatnonzero(labels == row['regionId']).astype(np.int64)
            row['cellCount'] = int(len(cells)); row['volumeMm3'] = len(cells) * self.voxel
            row['_cells'] = cells
            row['_tree'] = cKDTree(scene.points(cells)) if len(cells) else None
            self.label_class[row['regionId']] = ('tract', row['family'], row['id'])
        self.vessels = {}
        for region in scene.regions:
            if region['material'] == 'vessel':
                kind = vessel_kind(region)
                self.label_class[region['id']] = ('vessel', kind, kind)
                self.vessels.setdefault(kind, []).append(region['id'])
        for kind, ids in self.vessels.items():
            cells = np.flatnonzero(np.isin(labels, ids)).astype(np.int64)
            self.vessels[kind] = {'regionIds': sorted(ids), '_cells': cells,
                                  '_tree': cKDTree(scene.points(cells)) if len(cells) else None}

    def public(self):
        return {'schema': SCHEMA, 'property': PROPERTY, 'evidenceClass': 'model-estimate',
                'tracts': [{k: v for k, v in row.items() if not k.startswith('_')} for row in self.tracts],
                'familiesWithoutTracts': [f for f in FAMILIES if not any(t['family'] == f for t in self.tracts)],
                'vessels': {kind: {'regionIds': row['regionIds'], 'cellCount': int(len(row['_cells']))}
                            for kind, row in sorted(self.vessels.items())},
                'awakeScenario': self.awake, 'maxSearchMm': self.max_search, 'gridUncertaintyMm': self.grid_mm,
                'responseLaw': copy.deepcopy(RESPONSE_LAW), 'chargeScaling': copy.deepcopy(CHARGE_SCALING),
                'ischaemicTerritory': dict(ISCHAEMIC_TERRITORY), 'patientOutcome': dict(PATIENT_OUTCOME),
                'limits': list(LIMITS)}


def _model(scene):
    value = _validate(scene)
    key = (scene.identity, semantic_digest(value))
    model = _CACHE.get(key)
    if model is None:
        model = _Model(scene, value)
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = model
    return model


def declare(scene):
    """List the scene's declared tracts, families, response law and stated unknowns."""
    return _model(scene).public()


def _nearest_retained(tree, cells, point, removed, upper):
    """Distance to the nearest retained cell within ``upper`` mm, else None."""
    if tree is None:
        return None
    n, k = len(cells), 16
    while True:
        distance, index = tree.query(point, k=min(k, n), distance_upper_bound=upper)
        for d, i in zip(np.atleast_1d(distance), np.atleast_1d(index)):
            if not math.isfinite(d):
                return None
            if int(cells[i]) not in removed:
                return float(d)
        if k >= n:
            return None
        k *= 4


def effective_current(amplitude, pulse_width_us, frequency_hz):
    s = CHARGE_SCALING
    pw = (1 + s['chronaxieUs'] / s['referencePulseWidthUs']) / (1 + s['chronaxieUs'] / pulse_width_us)
    pw = min(max(pw, s['pulseWidthFactorBounds'][0]), s['pulseWidthFactorBounds'][1])
    f = (frequency_hz / s['referenceFrequencyHz']) ** s['frequencyExponent']
    f = min(max(f, s['frequencyFactorBounds'][0]), s['frequencyFactorBounds'][1])
    return amplitude * pw * f, pw, f


def reach_mm(current_ma):
    """Primary (owner-supplied) and secondary (assumed) distances implied by the current."""
    secondary = RESPONSE_LAW['secondaryReference']
    return (RESPONSE_LAW['interceptMm'] + RESPONSE_LAW['slopeMmPerMa'] * current_ma,
            secondary['interceptMm'] + secondary['slopeMmPerMa'] * current_ma)


def stimulation_observation(engine, probe, position, cells, parameters):
    """One labelled model-estimate mapping row for a probe application."""
    model = _model(engine.scene)
    point = np.asarray(position, float)
    removed = set(map(int, engine.state['removed']))
    tracts = []
    for row in model.tracts:
        distance = _nearest_retained(row['_tree'], row['_cells'], point, removed, model.max_search)
        tracts.append({'id': row['id'], 'regionName': row['regionName'], 'family': row['family'],
                       'provenance': row['provenance'],
                       'estimatedDistanceMm': distance,
                       'coverage': 'within declared search' if distance is not None else
                                   f'no retained cell within {model.max_search:g} mm (outside coverage)'})
    s = CHARGE_SCALING
    amplitude = parameters.get('amplitudeMa')
    pulse_width = parameters.get('pulseWidthUs', s['referencePulseWidthUs'])
    frequency = parameters.get('frequencyHz', s['referenceFrequencyHz'])
    assumed = [name for name in ('pulseWidthUs', 'frequencyHz') if name not in parameters]
    stimulus = None
    if amplitude is None:
        stimulus = 'No stimulation current was declared for this probe application.'
    elif amplitude <= 0 or pulse_width <= 0 or frequency <= 0:
        stimulus = 'A zero current, pulse width or frequency delivers no stimulation.'
    current = reach = secondary = None
    factors = {}
    if stimulus is None:
        current, pw_factor, f_factor = effective_current(float(amplitude), float(pulse_width), float(frequency))
        reach, secondary = reach_mm(current)
        factors = {'pulseWidthFactor': pw_factor, 'frequencyFactor': f_factor}
    disagreement = None if reach is None else abs(reach - secondary)
    uncertainty = model.grid_mm + (disagreement or 0.)
    families = {}
    for family in FAMILIES + ('unclassified',):
        members = [t for t in tracts if t['family'] == family]
        if not members:
            if family != 'unclassified':
                families[family] = {'result': 'unknown', 'reason': 'The scene declares no tract evidence for this function family.'}
            continue
        reason = None
        if stimulus is not None:
            reason = stimulus
        elif family == 'unclassified':
            reason = 'The tract has no declared function family.'
        elif family == 'other':
            reason = 'No stimulation response is modelled for this family.'
        elif family in ('language', 'vision') and not model.awake:
            reason = 'Mapping this family needs an awake patient performing tasks; the scene declares no awake scenario.'
        within = [t for t in members if t['estimatedDistanceMm'] is not None]
        if reason is None and not within:
            reason = f'The tip is outside declared tract coverage: no retained {family} tract within {model.max_search:g} mm.'
        if reason is not None:
            families[family] = {'result': 'unknown', 'reason': reason}
            continue
        nearest = min(within, key=lambda t: (t['estimatedDistanceMm'], t['id']))
        margin = reach - nearest['estimatedDistanceMm']
        response = margin >= -1e-9
        result = 'model_estimate_response' if response else 'model_estimate_no_response'
        families[family] = {
            'result': result, 'nearestTract': nearest['id'], 'estimatedDistanceMm': nearest['estimatedDistanceMm'],
            'marginMm': margin, 'decisionWithinUncertainty': abs(margin) <= uncertainty,
            **({'lawExtrapolated': 'The motor distance-current relation is applied to a non-motor tract.'}
               if family != 'motor' else {})}
        if probe == 'cortical' and not response:
            families[family] = {'result': 'unknown', 'estimatedDistanceMm': nearest['estimatedDistanceMm'],
                                'reason': 'Cortical function is not represented by tract evidence; '
                                          'a cortical probe without a tract-mediated response is not evidence.'}
    results = [row['result'] for row in families.values()]
    result = ('model_estimate_response' if 'model_estimate_response' in results else
              'model_estimate_no_response' if 'model_estimate_no_response' in results else 'unknown')
    if reach is None:
        band = None
    elif result == 'model_estimate_response':
        band = f'tract within ~{reach:.1f} mm (model estimate)'
    else:
        band = (f'no evaluated tract retained within ~{reach:.1f} mm (model estimate); '
                'absence of a response does not establish absence of function')
    unknown = sorted(f for f, row in families.items() if row['result'] == 'unknown')
    return {
        'kind': 'model-estimate', 'category': 'mapping', 'probe': probe,
        'position': point.tolist(), 'cells': sorted(map(int, cells)),
        'parameters': dict(parameters), 'parametersAssumedAtReference': assumed,
        'coverage': (f'declared tract estimates within {model.max_search:g} mm of the tip'
                     if any(t['estimatedDistanceMm'] is not None for t in tracts) else 'unknown'),
        'evidence': [], 'result': result,
        'uncertainty': (f'±{uncertainty:.2f} mm (grid ±{model.grid_mm:.2f} mm' +
                        (f'; reference relations differ by {disagreement:.2f} mm' if disagreement is not None else '') +
                        '; fit residual not supplied)'),
        'uncertaintyMm': uncertainty, 'gridUncertaintyMm': model.grid_mm, 'referenceDisagreementMm': disagreement,
        'effectiveCurrentMa': current, **factors,
        'estimatedReachMm': reach, 'secondaryReferenceReachMm': secondary, 'distanceBand': band,
        'tracts': tracts, 'families': families, 'unknownFamilies': unknown,
        'responseLaw': RESPONSE_LAW['id'], 'chargeScaling': 'assumed-research',
        'provenance': {'tracts': 'estimated', 'responseLaw': RESPONSE_LAW['provenance'],
                       'lawSource': RESPONSE_LAW['source'], 'chargeScaling': CHARGE_SCALING['provenance']},
        'ischaemicTerritory': 'unknown', 'deficitProbability': None}


def _band_stats(points_tree, cells, removed_mask, cavity_tree, voxel, cap):
    """Removed volume, centre distance to the cavity and retained cells in 0-2 and 2-5 mm bands."""
    if not len(cells):
        return {'cellCount': 0, 'removedMm3': 0., 'minDistanceMm': None, 'minDistanceStatus': 'no declared cells',
                'band0To2Mm': {'cells': 0, 'mm3': 0.}, 'band2To5Mm': {'cells': 0, 'mm3': 0.}}
    gone = removed_mask[cells]
    removed = int(gone.sum())
    band = [0, 0]
    distance, status = None, 'no resection cavity'
    if removed:
        distance, status = 0., 'cells removed'
    if cavity_tree is not None:
        retained = points_tree.data[~gone]
        if len(retained):
            d, _ = cavity_tree.query(retained, k=1, distance_upper_bound=max(5., cap))
            finite = d[np.isfinite(d)]
            band = [int(np.count_nonzero(finite <= 2. + 1e-9)), int(np.count_nonzero((finite > 2. + 1e-9) & (finite <= 5. + 1e-9)))]
            if not removed:
                if finite.size:
                    distance, status = float(finite.min()), 'measured'
                else:
                    distance, status = None, f'farther than {cap:g} mm'
    return {'cellCount': int(len(cells)), 'removedMm3': removed * voxel, 'minDistanceMm': distance,
            'minDistanceStatus': status,
            'band0To2Mm': {'cells': band[0], 'mm3': band[0] * voxel},
            'band2To5Mm': {'cells': band[1], 'mm3': band[1] * voxel}}


def _cavity(engine):
    """Removed non-bone, non-membrane cells: the resection cavity, not the craniotomy."""
    scene = engine.scene
    removed = np.asarray(sorted(map(int, engine.state['removed'])), dtype=np.int64)
    if not len(removed):
        return removed
    keep = np.zeros(65536, bool)
    for region in scene.regions:
        keep[region['id']] = region['material'] not in ('bone', 'membrane')
    return removed[keep[scene.labels.ravel()[removed]]]


def structure_harm(engine):
    """Per declared tract and vessel kind: removal, distance to the cavity and band volumes."""
    model = _model(engine.scene)
    scene = engine.scene
    removed_mask = np.zeros(scene.labels.size, bool)
    removed_mask[np.asarray(engine.state['removed'], dtype=np.int64)] = True
    cavity = _cavity(engine)
    cavity_tree = cKDTree(scene.points(cavity)) if len(cavity) else None
    cap = model.max_search
    tracts = {}
    for row in model.tracts:
        stats = _band_stats(row['_tree'], row['_cells'], removed_mask, cavity_tree, model.voxel, cap)
        tracts[row['id']] = {'family': row['family'], 'regionName': row['regionName'],
                             'provenance': row['provenance'], **stats}
    vessels = {kind: _band_stats(row['_tree'], row['_cells'], removed_mask, cavity_tree, model.voxel, cap)
               for kind, row in sorted(model.vessels.items())}
    families = {}
    for family in FAMILIES:
        members = [row for row in tracts.values() if row['family'] == family]
        if not members:
            families[family] = {'status': 'unknown', 'reason': 'No declared tract evidence for this family.'}
            continue
        distances = [row['minDistanceMm'] for row in members if row['minDistanceMm'] is not None]
        families[family] = {'status': 'model_estimate', 'removedMm3': sum(r['removedMm3'] for r in members),
                            'minDistanceMm': min(distances) if distances else None,
                            'band0To2Mm3': sum(r['band0To2Mm']['mm3'] for r in members),
                            'band2To5Mm3': sum(r['band2To5Mm']['mm3'] for r in members)}
    return {'schema': SCHEMA + '/structure-harm', 'sceneId': scene.identity, 'evidenceClass': 'model-estimate',
            'distanceMeaning': 'Centre-to-centre distance between a resection-cavity cell and a declared cell, '
                               f'searched to {cap:g} mm; None beyond that or without a cavity.',
            'cavity': {'cells': int(len(cavity)), 'mm3': len(cavity) * model.voxel,
                       'definition': 'Removed cells outside bone and membrane regions.'},
            'tracts': tracts, 'vessels': vessels, 'families': families,
            'ischaemicTerritory': dict(ISCHAEMIC_TERRITORY), 'patientOutcome': dict(PATIENT_OUTCOME)}


MAPPING_DIMENSION_METRICS = {'removedMm3': ('mm3', 'min'), 'minDistanceMm': ('mm', 'max')}


def consequence_dimensions(engine, baseline):
    """``functional:<tract>:removedMm3`` and ``:minDistanceMm`` plus separate model-estimate fields."""
    now, before = structure_harm(engine), structure_harm(baseline)
    cap = _model(engine.scene).max_search
    dimensions = {}
    for tract, row in now['tracts'].items():
        previous = before['tracts'][tract]
        for metric, (unit, direction) in MAPPING_DIMENSION_METRICS.items():
            if metric == 'removedMm3':
                value, start = float(row['removedMm3']), float(previous['removedMm3'])
                aggregation, meaning = 'cumulative_volume', 'Added removed volume of the declared tract estimate since the comparison prefix.'
            else:
                value = cap if row['minDistanceMm'] is None else min(float(row['minDistanceMm']), cap)
                start = cap if previous['minDistanceMm'] is None else min(float(previous['minDistanceMm']), cap)
                aggregation = 'minimum_distance'
                meaning = (f'Change in the cavity-to-tract distance since the comparison prefix; values are capped at '
                           f'{cap:g} mm, which means at least that far (or no cavity yet).')
            dimensions[f'functional:{tract}:{metric}'] = {
                'unit': unit, 'direction': direction, 'status': 'modeled_unvalidated',
                'cumulative': value, 'incremental': value - start, 'baseline': start, 'reason': None,
                'subjectId': tract, 'mechanism': metric, 'aggregation': aggregation, 'evidenceClass': 'model_estimate',
                'incrementalMeaning': meaning,
                'scope': 'Declared tract region (an estimate from a published baseline tract), not patient function.',
                'source': {'sceneId': engine.scene.identity, 'model': SCHEMA, 'family': row['family'],
                           'regionName': row['regionName'], 'provenance': row['provenance']}}
    families = {family: row for family, row in now['families'].items()}
    estimate = {
        'evidenceClass': 'model_estimate',
        'motorTractRemovedMm3ModelEstimate': families['motor'].get('removedMm3'),
        'languageTractRemovedMm3ModelEstimate': families['language'].get('removedMm3'),
        'motorTractMinDistanceMmModelEstimate': families['motor'].get('minDistanceMm'),
        'languageTractMinDistanceMmModelEstimate': families['language'].get('minDistanceMm'),
        'families': copy.deepcopy(families),
        'meaning': 'Removal of, and distance to, declared tract estimates. motorEncounterMm3 and '
                   'languageEncounterMm3 stay unknown as patient outcomes.',
        'ischaemicTerritory': dict(ISCHAEMIC_TERRITORY), 'patientOutcome': dict(PATIENT_OUTCOME)}
    return dimensions, estimate


STRUCTURE_METRICS = tuple(
    [f'structure_tract_{family}_{part}' for family in FAMILIES + ('unclassified',)
     for part in ('removed_mm3', 'band_0_2mm_mm3', 'band_2_5mm_mm3')] +
    [f'structure_vessel_{kind}_removed_mm3' for kind in VESSEL_KINDS] + ['structure_other_healthy_removed_mm3'])


def label_classes(scene):
    """``{label: ('tract', family, id) | ('vessel', kind, kind)}`` for declared structures."""
    return dict(_model(scene).label_class)


def band_volumes(engine):
    """Retained declared-tract volume within 0-2 mm and 2-5 mm of the cavity, by family."""
    harm = structure_harm(engine)
    result = {}
    for family in FAMILIES + ('unclassified',):
        members = [row for row in harm['tracts'].values() if row['family'] == family]
        result[family] = (sum(r['band0To2Mm']['mm3'] for r in members), sum(r['band2To5Mm']['mm3'] for r in members))
    return result


def missing_declaration_warning(scene):
    """Reported when a procedure-scale scene carries tract layers but no mapping declaration."""
    if PROPERTY in scene.properties:
        return None
    tracts = sorted(region['name'] for region in scene.regions if str(region['name']).rstrip().endswith('(tract)'))
    procedure = any(name in scene.properties for name in
                    ('operativeExposureV1', 'surgicalProcedurePolicyV1', 'hostRehearsalV1'))
    if not tracts or not procedure:
        return None
    return {'code': 'TRACT_LAYERS_WITHOUT_FUNCTIONAL_MAPPING',
            'tractRegions': tracts,
            'message': 'This scene carries tract layers but no functionalMappingV1 declaration; tract cells are '
                       'weighted as ordinary healthy tissue and probes return unknown.'}
