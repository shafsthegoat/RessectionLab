"""Bounded post-exposure geometry using current source ownership and engine capsules.

Generation changes no tissue and runs no continuum solve. Exact public access
and reverse withdrawal are required before a proposal can be installed.
"""
from __future__ import annotations

import copy
import time
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from .core import array_digest, semantic_digest
from .full_case_directional import directional_access_candidates
from .full_case_geometry import access_candidates
from .surgical_joint_geometry import post_exposure_proposal, post_exposure_candidate_pairs, _axial_cover, _params, _pose
from .surgical_task_admission import admission, bind_admission, validate_admission, check_program

SCHEMA = 'resectionlab.interactive-source-approaches.v1'
ADMITTED_SCHEMA = 'resectionlab.interactive-source-approaches.v2'
OBSERVED_SCHEMA = 'resectionlab.interactive-source-approaches.v3'
_SOURCE_LABELS_HASH = 'sha256:3c9911fed460598201fb904d36f752ef02129b779bc8dcf98b402ca35da3fa61'
_UNKNOWN = ('vessels', 'functional', 'shift', 'skull')
_SCOPE = ('Inactive approach inside an existing authored exposure; current source-cell geometry only. '
          'Zero additional tissue traversal/removal. Missing anatomy and clinical consequences remain unknown. '
          'Exact public access and withdrawal validation is required before installation.')
_ADMITTED_SCOPE = ('Inactive approach from a declared clear entry linked to removed source tissue at an exactly replayed, released and undeformed native baseline. '
                   'Source-annotation-assisted geometry only; zero additional tissue traversal/removal. '
                   'Missing anatomy and clinical consequences remain unknown; complete public access and withdrawal are required.')
_SAVED_VALIDATION_CACHE = OrderedDict()
_ROUTE_VALIDATION_CACHE = OrderedDict()
_VALIDATION_CACHE_ENTRIES = 32  # Holds all 16 saved revisions plus recent current variants.


def _number(value, name, low, high):
    if type(value) not in (int, float) or not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be a finite number in [{low}, {high}]')
    return float(value)


def _keys(value, names, name):
    if type(value) is not dict or set(value) - set(names):
        raise ValueError(name + ' contains unsupported fields or is not an object')


def _sphere(value, name):
    _keys(value, ('centerMm', 'radiusMm'), name)
    center = value.get('centerMm')
    if type(center) is not list or len(center) != 3:
        raise ValueError(name + '.centerMm requires three finite RAS millimeters')
    return {'centerMm': [_number(v, name + '.centerMm', -10000., 10000.) for v in center],
            'radiusMm': _number(value.get('radiusMm'), name + '.radiusMm', .5, 30.)}


def _source_supported(scene):
    adapter = scene.source.get('adapter')
    if type(adapter) is not dict:
        return False
    try:
        scene.require_cache_source()
    except ValueError:
        return False
    return (adapter.get('sourcePatient') == 'UCSF-PDGM-0007'
            and array_digest(scene.labels) == _SOURCE_LABELS_HASH
            and [(r['id'], r['material'], bool(r.get('target'))) for r in scene.regions] ==
                [(1, 'soft_tissue', False), (2, 'tumor', True)]
            and scene.shape == (240, 240, 155)
            and np.array_equal(scene.affine, [[-1., 0., 0., 0.], [0., -1., 0., 239.],
                                             [0., 0., 1., 0.], [0., 0., 0., 1.]]))


def _target_ids(engine):
    return {r['id'] for r in engine.scene.regions if r.get('target') and r['material'] == 'tumor'}


def _retained_targets(engine):
    cells = np.flatnonzero(np.isin(engine.scene.labels.ravel(), list(_target_ids(engine))))
    return np.setdiff1d(cells, engine.state['removed'], assume_unique=False)


def _exposure(engine):
    """Recorded authored removal, never a hole inferred from display surfaces."""
    admitted = admission(engine)
    if admitted is not None:
        # In the admitted workflow this is a declared proposal direction/entry,
        # not evidence that a requested cutting trajectory was completed.
        task = admitted['task']; entry = np.asarray(task['entry']['centerMm'], float)
        return entry, entry+np.asarray(task['axisRasUnit']), task['entry']['radiusMm']
    # Initial slots are outside anatomy; replay only declarative action poses.
    slots = {key: {'instrument': tool, 'parameters': {}, 'position': None}
             for key, tool in (('left', 'cranial_drill'), ('right', 'cusa'))}
    result = None
    for action in engine.actions:
        slot = slots[action['slot']]
        if action['operation'] == 'exchange':
            slot['instrument'] = action['instrument']; slot['parameters'] = {}
        slot['parameters'].update(action.get('parameters', {}))
        points = action.get('path') or ([action['pose']] if action.get('pose') else [])
        if action['operation'] == 'apply' and slot['instrument'] == 'cusa' and points and slot['position'] is not None:
            start, end = np.asarray(slot['position'], float), np.asarray(points[-1]['position'], float)
            if np.linalg.norm(end-start) > 1.:
                result = (start, end, float(slot['parameters'].get('radiusMm', 1.5)))
        if points:
            slot.update(points[-1])
    return result


def _evidence(engine):
    layers = (engine.scene.source.get('adapter') or {}).get('sourceLayers', {})
    guard = engine.guard_report()
    known = ['target', 'brain'] + [r['id'] for r in guard['structures'] if r.get('sourceCells')]
    target_removed = int(np.count_nonzero(np.isin(engine.scene.labels.ravel()[engine.state['removed']], list(_target_ids(engine)))))*engine.scene.voxel_mm3
    result = {'sourceCaseId': engine.scene.source.get('caseId'), 'sourcePatient': 'UCSF-PDGM-0007',
            'targetProvenance': copy.deepcopy(layers.get('tumor', {})),
            'brainProvenance': copy.deepcopy(layers.get('brain', {})),
            'protections': guard['structures'], 'guardContractId': guard['contractId'],
            'availableEvidence': known, 'unknownEvidence': [x for x in _UNKNOWN if x not in known],
            'historicalExposure': {'targetRemovedMm3': target_removed,
                                   'nonTargetRemovedMm3': len(engine.state['removed'])*engine.scene.voxel_mm3-target_removed}}
    admitted = admission(engine)
    if admitted is not None:
        source = admitted['sourceEvidence']
        result.update(sourcePatient=source['patientId'], availableEvidence=sorted(set(source['availableEvidence']) | set(known[2:])),
            unknownEvidence=source['unknownEvidence'], taskAdmissionSeal=admitted['admissionSeal'],
            sourceQualification=source, inputContract=admitted['task']['inputContract'])
    return result


def _defaults(engine, exposure):
    admitted = admission(engine)
    if admitted is not None:
        task = admitted['task']
        return {'entry': copy.deepcopy(task['entry']), 'target': copy.deepcopy(task['target']),
            'tool': {'instrument': 'cusa', 'radiusMm': task['toolRadiusMm']},
            'requiredEvidence': task['requiredEvidence'][:], 'nonTargetClearanceMm': 0.,
            'candidateCount': 3, 'maxCandidates': 32, 'minEntrySeparationMm': 2.,
            'minAngleDeg': 0., 'maxWallSeconds': 15.}
    start, end, radius = exposure
    axis = (end-start)/np.linalg.norm(end-start)
    projection = (engine.scene.points(engine.state['removed'])-start) @ axis
    entry = start + axis*(float(projection.min()) + min(2., radius/3.))
    points = engine.scene.points(_retained_targets(engine))
    relative = points-end
    depth = relative@axis
    radial = np.linalg.norm(relative-depth[:, None]*axis, axis=1)
    target = points[np.lexsort((np.linalg.norm(relative, axis=1), radial))[0]]
    return {'target': {'centerMm': target.tolist(), 'radiusMm': 12.},
            'entry': {'centerMm': entry.tolist(), 'radiusMm': max(.5, min(15., radius-2.))},
            'tool': {'instrument': 'cusa', 'radiusMm': 1.5}, 'requiredEvidence': [],
            'nonTargetClearanceMm': 0.,
            'candidateCount': 3, 'maxCandidates': 32, 'minEntrySeparationMm': 2.,
            'minAngleDeg': 0., 'maxWallSeconds': 15.}


def interactive_planning_availability(engine):
    reasons = []
    admitted = admission(engine)
    if admitted is not None:
        try:
            validate_admission(engine, admitted)
            from .surgical_planning_observation import validate_observation_scope
            validate_observation_scope(engine)
        except ValueError as error:
            return {'available': False, 'reason': str(error), 'defaults': None,
                    'evidence': _evidence(engine), 'scope': _ADMITTED_SCOPE, 'admission': admitted}
        return {'available': True, 'reason': None, 'defaults': _defaults(engine, None),
                'evidence': _evidence(engine), 'scope': _ADMITTED_SCOPE, 'admission': admitted}
    supported = _source_supported(engine.scene)
    if not supported:
        return {'available': False,
                'reason': ('A source-declared tumor target is required; this case has no supported removal target.' if not _target_ids(engine)
                           else 'A declared patient/task admission is required. Historical planning is limited to the documented UCSF-PDGM-0007 source grid.'),
                'defaults': None, 'evidence': None, 'scope': _SCOPE}
    exposure = _exposure(engine)
    if not engine.state['removed'] or exposure is None:
        reasons.append('Select the authored post-exposure checkpoint first.')
    if any(s['active'] or s['inserted'] or s['grip'] for s in engine.state['slots'].values()) or engine.state['holds']:
        reasons.append('Both tools must be inactive, released and withdrawn at the baseline.')
    if not len(_retained_targets(engine)):
        reasons.append('A retained source tumor target is required.')
    return {'available': not reasons, 'reason': ' '.join(reasons) if reasons else None,
            'defaults': _defaults(engine, exposure) if not reasons else None,
            'evidence': _evidence(engine) if supported else None, 'scope': _SCOPE}


def normalize_constraints(engine, raw, *, task=None):
    _keys(raw, ('target', 'entry', 'tool', 'requiredEvidence', 'candidateCount', 'maxCandidates',
                'minEntrySeparationMm', 'minAngleDeg', 'maxWallSeconds', 'nonTargetClearanceMm'), 'constraints')
    result = {'target': _sphere(raw.get('target'), 'target'), 'entry': _sphere(raw.get('entry'), 'entry')}
    tool = raw.get('tool'); _keys(tool, ('instrument', 'radiusMm'), 'tool')
    if tool.get('instrument') != 'cusa':
        raise ValueError('Only existing CUSA geometry is supported for this bounded source approach')
    result['tool'] = {'instrument': 'cusa', 'radiusMm': _number(tool.get('radiusMm'), 'tool.radiusMm', .1, 6.)}
    admitted = admission(engine)
    task = task if task is not None else (admitted['task'] if admitted is not None else None)
    if task is not None and result['tool']['radiusMm'] != task['toolRadiusMm']:
        raise ValueError('Task admission requires the fixed catalog CUSA radius')
    required = raw.get('requiredEvidence', [])
    if type(required) is not list or len(required) > 32 or any(type(x) is not str or not x or len(x) > 128 for x in required):
        raise ValueError('requiredEvidence must be a bounded list of evidence names')
    result['requiredEvidence'] = sorted(set(required))
    if task is not None and not set(task['requiredEvidence']).issubset(result['requiredEvidence']):
        raise ValueError('Task admission required evidence cannot be removed')
    if task is not None:
        for name in ('entry', 'target'):
            declared = task[name]; requested = result[name]
            if np.linalg.norm(np.subtract(requested['centerMm'], declared['centerMm'])) + requested['radiusMm'] > declared['radiusMm'] + 1e-9:
                raise ValueError('Requested '+name+' sphere exceeds the admitted task region')
    for name, default, upper in [('candidateCount', 3, 6), ('maxCandidates', 32, 96)]:
        value = raw.get(name, default)
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(name + f' must be an integer in [1, {upper}]')
        result[name] = value
    result['minEntrySeparationMm'] = _number(raw.get('minEntrySeparationMm', 2.), 'minEntrySeparationMm', .5, 15.)
    result['minAngleDeg'] = _number(raw.get('minAngleDeg', 0.), 'minAngleDeg', 0., 60.)
    result['maxWallSeconds'] = _number(raw.get('maxWallSeconds', 15.), 'maxWallSeconds', 1., 30.)
    result['nonTargetClearanceMm'] = _number(raw.get('nonTargetClearanceMm', 0.), 'nonTargetClearanceMm', 0., 2.)
    points = engine.scene.points(_retained_targets(engine))
    if not len(points) or not np.any(np.linalg.norm(points-result['target']['centerMm'], axis=1) <= result['target']['radiusMm']):
        raise ValueError('Target sphere must contain retained owned source tumor cells')
    exposure = _exposure(engine)
    if exposure is not None:
        baseline = _defaults(engine, exposure)['entry']
        if np.linalg.norm(np.subtract(result['entry']['centerMm'], baseline['centerMm'])) > baseline['radiusMm'] + 1e-9:
            raise ValueError('Entry center must remain inside the existing authored exposure region')
        if result['entry']['radiusMm'] > baseline['radiusMm'] + 1e-9:
            raise ValueError('Entry radius exceeds the existing authored exposure region')
    return result


def _binding(engine):
    admitted = admission(engine)
    return {**({'taskAdmissionSeal': admitted['admissionSeal']} if admitted is not None else {}),
            'sourceSceneId': engine.scene.identity, 'startStateId': engine.state_id,
            'prefixActions': len(engine.actions), 'baselineActionHash': semantic_digest(engine.actions),
            'baselineHistoryHash': semantic_digest(engine.history),
            'sourceLabelsHash': array_digest(engine.scene.labels), 'physicsHash': semantic_digest(engine.scene.properties),
            'guardContractId': engine.guard_report()['contractId'],
            'baselineSnapshotHash': semantic_digest(engine.snapshot())}



def generate_interactive_approaches(engine, constraints, *, check=lambda: None, progress=lambda *_: None):
    available = interactive_planning_availability(engine)
    if not available['available']:
        raise ValueError(available['reason'])
    admitted = admission(engine)
    if admitted is not None:
        from .surgical_planning_observation import project_observation
        observation = project_observation(engine, admitted['task'], available['evidence']['availableEvidence'])
        public = plan_observed_approaches(observation, constraints, check=check, progress=progress)
        # The complete public plan is sealed before the native audit envelope.
        report = {**public, 'schema': OBSERVED_SCHEMA, 'taskAdmission': admitted,
                  **_binding(engine), 'evidence': available['evidence']}
        report.pop('reportSeal', None)
        report['reportSeal'] = semantic_digest(report)
        return report
    normalized = normalize_constraints(engine, constraints)
    exposure = _exposure(engine)
    axis = exposure[1]-exposure[0]; axis /= np.linalg.norm(axis)
    report = {'schema': SCHEMA, **_binding(engine), 'constraints': normalized, 'evidence': available['evidence']}
    return _generate_geometry(engine, normalized, available['evidence'], available['scope'], axis, report,
                              stop_at_first_contact=False, check=check, progress=progress)


def plan_observed_approaches(observation, constraints, *, check=lambda: None, progress=lambda *_: None):
    """Candidate generation, previews, ordering and STOP see only public inputs."""
    from .core import thaw_json
    from .surgical_planning_observation import PlanningObservation
    if type(observation) is not PlanningObservation:
        raise ValueError('A typed public PlanningObservation is required')
    fields = thaw_json(observation.fields)
    engine = observation.planning_engine()
    normalized = normalize_constraints(engine, constraints, task=fields['task'])
    evidence = {'availableEvidence': fields['availableEvidence']}
    report = {'constraints': normalized, 'evidence': evidence}
    try:
        report = _generate_geometry(engine, normalized, evidence, _ADMITTED_SCOPE,
            np.asarray(fields['task']['axisRasUnit']), report, stop_at_first_contact=True,
            preview_public=True, check=check, progress=progress)
    except TimeoutError:
        # Proposal-source preparation is also inside the declared budget. A
        # timeout before the first ray is a sealed computational refusal.
        report['rejections'].append({'reason': 'GEOMETRY_WALL_BUDGET'})
        report['status'] = 'insufficient_distinct_routes' if report['routes'] else 'no_geometric_route'
    report.pop('reportSeal', None)
    report['publicObservation'] = observation.manifest()
    report['decision'] = {'operation': 'COMPARE' if report['routes'] else 'STOP',
        'candidateIds': [row['id'] for row in report['routes']], 'reason': report['status'],
        'rule': 'deterministic separated feasible candidates; no clinical preference score'}
    report['publicPlanSeal'] = semantic_digest(report)
    return report


def _generate_geometry(engine, normalized, evidence, scope, source_axis, report, *,
                       stop_at_first_contact, preview_public=False, check=lambda: None, progress=lambda *_: None):
    target_cells = _retained_targets(engine)
    target_points = engine.scene.points(target_cells)
    selected_mask = np.linalg.norm(target_points-normalized['target']['centerMm'], axis=1) <= normalized['target']['radiusMm']
    selected_cells = target_cells[selected_mask]
    report.update({'routes': [], 'rejections': [], 'limitingStructures': [], 'scope': scope,
              'targetSelection': {'retainedSourceCells': int(len(selected_cells)),
                                  'sourceCellHash': semantic_digest(selected_cells.tolist()),
                                  'volumeMm3': float(len(selected_cells)*engine.scene.voxel_mm3),
                                  'selectionRule': 'retained owned tumor cell centers inside requested RAS sphere'},
              'screening': {'method': 'shared_engine_current_occupancy_and_complete_capsules',
                            'continuumEvaluated': False, 'tissueAltered': False, 'publicValidationRequired': True}})
    missing = sorted(set(normalized['requiredEvidence'])-set(evidence['availableEvidence']))
    if missing:
        report.update(status='required_evidence_unavailable', missingRequiredEvidence=missing)
        report['rejections'] = [{'reason': 'REQUIRED_EVIDENCE_UNAVAILABLE', 'evidence': name} for name in missing]
        report['reportSeal'] = semantic_digest(report)
        return report
    started = time.monotonic()

    def bounded_check():
        check()
        if time.monotonic()-started > normalized['maxWallSeconds']:
            raise TimeoutError('Interactive geometry wall budget exhausted')

    entry = np.asarray(normalized['entry']['centerMm'])
    points = engine.scene.points(_retained_targets(engine))
    target_center = np.asarray(normalized['target']['centerMm'])
    selected = points[np.linalg.norm(points-target_center, axis=1) <= normalized['target']['radiusMm']]
    goal = selected[np.argmin(np.linalg.norm(selected-target_center, axis=1))]
    radius = normalized['entry']['radiusMm']
    # Existing source-grid generators supply additional proposals. A removed
    # overlying brain cavity may have no eligible original-label native window.
    native = []
    for axis in (2, 0, 1):
        bounded_check()
        try:
            proposed = (access_candidates(engine, limit=4, separation_mm=2.) if axis == 2 else
                        directional_access_candidates(engine, axis=axis, limit=4, separation_mm=2.))
        except ValueError:
            proposed = []
        for route in proposed:
            point = np.asarray(route['entryPose']['position'])
            if np.linalg.norm(point-target_center) <= normalized['target']['radiusMm']:
                native.append((entry, point))
    proposals = post_exposure_candidate_pairs(entry, goal, source_axis, radius)+native
    try:
        for index, (point, goal) in enumerate(proposals[:normalized['maxCandidates']]):
            bounded_check()
            row, rejection = post_exposure_proposal(engine, point, goal, normalized['tool']['radiusMm'], bounded_check,
                non_target_clearance_mm=normalized['nonTargetClearanceMm'], stop_at_first_contact=stop_at_first_contact)
            if rejection:
                report['rejections'].append({'proposal': index, 'entryPointMm': point.tolist(),
                    'sourceTargetPointMm': goal.tolist(), **rejection})
                if rejection.get('structureName') and rejection.get('positionMm') and len(report['limitingStructures']) < 12:
                    report['limitingStructures'].append({'name': rejection['structureName'], **rejection})
                continue
            if not set(row['requiredTargetRemoval']['sourceCells']).issubset(set(map(int, selected_cells))):
                report['rejections'].append({'proposal': index, 'reason': 'CONTACT_OUTSIDE_REQUESTED_TARGET_SUBREGION'})
                continue
            if preview_public:
                from .surgical_planning_observation import preview_inactive_program
                try:
                    row['publicExecution'] = preview_inactive_program(engine, row, check=bounded_check)
                except ValueError as error:
                    report['rejections'].append({'proposal': index, 'reason': 'PUBLIC_EXECUTION_REFUSED', 'detail': str(error)})
                    continue
            comparisons = []
            for existing in report['routes']:
                separation = float(np.linalg.norm(np.subtract(point, existing['entryPose']['position'])))
                angle = float(np.degrees(np.arccos(np.clip(np.dot(row['axisRasUnit'], existing['axisRasUnit']), -1., 1.))))
                comparisons.append({'routeId': existing['id'], 'entrySeparationMm': separation, 'angleDeg': angle})
            if any(x['entrySeparationMm'] < normalized['minEntrySeparationMm'] or
                   x['angleDeg']+1e-9 < normalized['minAngleDeg'] for x in comparisons):
                report['rejections'].append({'proposal': index, 'reason': 'INSUFFICIENT_ROUTE_SEPARATION'})
                continue
            row.update(id='approach-'+str(len(report['routes'])+1), label='Source approach '+str(len(report['routes'])+1),
                       separationComparisons=comparisons)
            report['routes'].append(row)
            progress(min(.95, (index+1)/len(proposals)), 'Screening complete tool access and withdrawal geometry')
            if len(report['routes']) >= normalized['candidateCount']:
                break
    except TimeoutError:
        report['rejections'].append({'reason': 'GEOMETRY_WALL_BUDGET'})
    report['status'] = ('geometric_candidates' if len(report['routes']) == normalized['candidateCount'] else
                        'insufficient_distinct_routes' if report['routes'] else 'no_geometric_route')
    report['reportSeal'] = semantic_digest(report)
    return report


def _validate_seal(report):
    if type(report) is not dict:
        raise ValueError('Interactive report must be an object')
    payload = copy.deepcopy(report); seal = payload.pop('reportSeal', None)
    if payload.get('schema') not in (SCHEMA, ADMITTED_SCHEMA, OBSERVED_SCHEMA) or semantic_digest(payload) != seal:
        raise ValueError('Interactive report seal differs')
    if (payload['schema'] in (ADMITTED_SCHEMA, OBSERVED_SCHEMA)) != ('taskAdmission' in payload):
        raise ValueError('Interactive report admission version differs')


def validate_saved_interactive_report(scene, snapshot, report):
    """Canonically admit saved content once, then reuse only complete-content keys."""
    _validate_seal(report)
    if report['schema'] == SCHEMA and not _source_supported(scene):
        raise ValueError('Saved interactive source is outside the documented source grid')
    if report['schema'] in (ADMITTED_SCHEMA, OBSERVED_SCHEMA):
        validate_admission(SimpleNamespace(scene=scene), report['taskAdmission'], require_baseline=False)
    cache_key = (scene.identity, semantic_digest(snapshot), semantic_digest(report))
    if cache_key in _SAVED_VALIDATION_CACHE:
        _SAVED_VALIDATION_CACHE.move_to_end(cache_key)
        return True
    state_id = semantic_digest({'scene': scene.identity, 'state': snapshot['checkpoint']['state'],
        **({'protectionContract': snapshot['protectionContract']['contentHash']} if snapshot.get('protectionContract') else {})})
    guard = (snapshot.get('protectionContract') or {}).get('contentHash')
    expected = {'sourceSceneId': scene.identity, 'sourceLabelsHash': array_digest(scene.labels),
                'physicsHash': semantic_digest(scene.properties), 'startStateId': snapshot['stateId'],
                'prefixActions': len(snapshot['actions']), 'baselineActionHash': semantic_digest(snapshot['actions']),
                'guardContractId': guard, 'baselineSnapshotHash': semantic_digest(snapshot)}
    if (snapshot.get('sceneId') != scene.identity or snapshot['checkpoint']['position'] != len(snapshot['actions']) or
            snapshot['checkpoint']['stateId'] != state_id or state_id != snapshot['stateId'] or
            any(report.get(key) != value for key, value in expected.items())):
        raise ValueError('Saved interactive report source or native baseline differs')
    # Cold admission uses the canonical public restore path. The cache key is
    # complete content, never caller object identity or mutable references.
    from .surgical_engine import SurgicalEngine
    baseline = SurgicalEngine.restore(scene, snapshot)
    if report['schema'] in (ADMITTED_SCHEMA, OBSERVED_SCHEMA):
        bind_admission(baseline, report['taskAdmission'])
        validate_admission(baseline, report['taskAdmission'])
        if report.get('taskAdmissionSeal') != report['taskAdmission']['admissionSeal']:
            raise ValueError('Saved interactive task admission differs')
    if report.get('baselineHistoryHash') != semantic_digest(baseline.history):
        raise ValueError('Saved interactive canonical native history differs')
    if normalize_constraints(baseline, report['constraints']) != report['constraints']:
        raise ValueError('Saved interactive constraints differ')
    if report['schema'] == OBSERVED_SCHEMA:
        _validate_observed_report(baseline, report)
    else:
        _validate_routes(baseline, report)
    _SAVED_VALIDATION_CACHE[cache_key] = True
    while len(_SAVED_VALIDATION_CACHE) > _VALIDATION_CACHE_ENTRIES:
        _SAVED_VALIDATION_CACHE.popitem(last=False)
    return True


def validate_interactive_report(engine, report, *, require_current=True):
    _validate_seal(report)
    engine.scene.require_cache_source()
    binding = _binding(engine)
    if report['schema'] in (ADMITTED_SCHEMA, OBSERVED_SCHEMA):
        admitted = admission(engine)
        if admitted is None or report['taskAdmission'] != admitted:
            raise ValueError('Interactive report task admission differs')
        if require_current: validate_admission(engine, admitted)
    fields = binding if require_current else ('sourceSceneId', 'sourceLabelsHash', 'physicsHash', 'guardContractId')
    if any(report.get(key) != binding[key] for key in fields):
        raise ValueError('Interactive report source or baseline differs; regenerate approaches')
    if require_current and normalize_constraints(engine, report['constraints']) != report['constraints']:
        raise ValueError('Interactive report constraints differ')
    if require_current:
        cache_key = (engine.scene.identity, binding['baselineSnapshotHash'], semantic_digest(report))
        if cache_key not in _ROUTE_VALIDATION_CACHE:
            if report['schema'] == OBSERVED_SCHEMA:
                _validate_observed_report(engine, report)
            else:
                _validate_routes(engine, report)
            _ROUTE_VALIDATION_CACHE[cache_key] = True
            while len(_ROUTE_VALIDATION_CACHE) > _VALIDATION_CACHE_ENTRIES:
                _ROUTE_VALIDATION_CACHE.popitem(last=False)
        else:
            _ROUTE_VALIDATION_CACHE.move_to_end(cache_key)
    return report.get('startStateId') == engine.state_id and report.get('baselineActionHash') == binding['baselineActionHash']


def _public_report_body(report):
    keys = ('constraints', 'routes', 'rejections', 'limitingStructures', 'scope', 'targetSelection',
            'screening', 'status', 'publicObservation', 'decision')
    body = {key: report[key] for key in keys}
    body['evidence'] = {'availableEvidence': report['evidence']['availableEvidence']}
    if 'missingRequiredEvidence' in report: body['missingRequiredEvidence'] = report['missingRequiredEvidence']
    return body


def _validate_observed_report(engine, report):
    from .surgical_planning_observation import project_observation
    admitted = admission(engine)
    if report['evidence'] != _evidence(engine):
        raise ValueError('Interactive report source evidence differs')
    observation = project_observation(engine, admitted['task'], report['evidence']['availableEvidence'])
    validate_public_approach_plan(observation, {**_public_report_body(report), 'publicPlanSeal': report['publicPlanSeal']})


def validate_public_approach_plan(observation, report):
    """Shared complete validator for a standalone public plan or native envelope."""
    from .core import thaw_json
    from .surgical_planning_observation import PlanningObservation, preview_inactive_program
    if type(observation) is not PlanningObservation or type(report) is not dict:
        raise ValueError('Typed public approach observation and plan required')
    fields = thaw_json(observation.fields)
    if normalize_constraints(observation.planning_engine(), report['constraints'], task=fields['task']) != report['constraints']:
        raise ValueError('Interactive public constraints differ')
    if report.get('evidence') != {'availableEvidence': fields['availableEvidence']}:
        raise ValueError('Interactive public evidence differs')
    if report.get('publicObservation') != observation.manifest():
        raise ValueError('Interactive public observation differs')
    body = _public_report_body(report)
    if set(report) != {*body, 'publicPlanSeal'}:
        raise ValueError('Interactive public plan schema differs')
    if report.get('publicPlanSeal') != semantic_digest(body):
        raise ValueError('Interactive public plan seal differs')
    expected = {'operation': 'COMPARE' if report['routes'] else 'STOP',
        'candidateIds': [row['id'] for row in report['routes']], 'reason': report['status'],
        'rule': 'deterministic separated feasible candidates; no clinical preference score'}
    if report.get('decision') != expected:
        raise ValueError('Interactive public STOP/comparison decision differs')
    missing = sorted(set(report['constraints']['requiredEvidence'])-set(report['evidence']['availableEvidence']))
    count = len(report['routes'])
    status = ('required_evidence_unavailable' if missing else 'geometric_candidates'
              if count == report['constraints']['candidateCount'] else 'insufficient_distinct_routes'
              if count else 'no_geometric_route')
    if (report['status'] != status or (report.get('missingRequiredEvidence') != missing if missing
                                      else 'missingRequiredEvidence' in report)):
        raise ValueError('Interactive public refusal status differs from evidence and route count')
    public = observation.planning_engine()
    _validate_routes(public, body, public_task=fields['task'])
    for row in report['routes']:
        if row.get('publicExecution') != preview_inactive_program(public, row):
            raise ValueError('Interactive public execution differs')


def _validate_routes(engine, report, *, public_task=None):
    constraints = report['constraints']
    admitted = admission(engine)
    if public_task is None and ((admitted is None and not _source_supported(engine.scene)) or report['evidence'] != _evidence(engine)):
        raise ValueError('Interactive report source evidence differs')
    if report.get('scope') != (_ADMITTED_SCOPE if admitted is not None or public_task is not None else _SCOPE) or report.get('screening') != {
            'method': 'shared_engine_current_occupancy_and_complete_capsules',
            'continuumEvaluated': False, 'tissueAltered': False, 'publicValidationRequired': True}:
        raise ValueError('Interactive report geometry scope differs')
    if report['routes'] and set(constraints['requiredEvidence'])-set(report['evidence']['availableEvidence']):
        raise ValueError('Interactive report requires unavailable evidence')
    if report['routes'] and report.get('status') not in ('geometric_candidates', 'insufficient_distinct_routes'):
        raise ValueError('Interactive refusal report cannot supply executable routes')
    if type(report['routes']) is not list or len(report['routes']) > constraints['candidateCount']:
        raise ValueError('Interactive route count differs')
    missing = sorted(set(constraints['requiredEvidence'])-set(report['evidence']['availableEvidence']))
    if missing and (report.get('status') != 'required_evidence_unavailable' or
                    report.get('missingRequiredEvidence') != missing):
        raise ValueError('Interactive required-evidence refusal differs')
    exposure = _exposure(engine)
    axis = np.asarray(public_task['axisRasUnit']) if public_task is not None else exposure[1]-exposure[0]
    axis /= np.linalg.norm(axis)
    _validate_diagnostics(engine, report, axis)
    seen, previous = set(), []
    retained = set(map(int, _retained_targets(engine)))
    ordered = np.asarray(sorted(retained), dtype=np.int64)
    selected = ordered[np.linalg.norm(engine.scene.points(ordered)-constraints['target']['centerMm'], axis=1) <= constraints['target']['radiusMm']]
    if report.get('targetSelection') != {'retainedSourceCells': int(len(selected)),
            'sourceCellHash': semantic_digest(selected.tolist()), 'volumeMm3': float(len(selected)*engine.scene.voxel_mm3),
            'selectionRule': 'retained owned tumor cell centers inside requested RAS sphere'}:
        raise ValueError('Interactive source target selection differs')
    for index, route in enumerate(report['routes'], 1):
        if route['id'] in seen:
            raise ValueError('Interactive route identity repeats')
        seen.add(route['id'])
        entry = np.asarray(route['entryPose']['position'], float)
        goal = np.asarray(route['sourceTargetPointMm'], float)
        delta = entry-constraints['entry']['centerMm']
        if (entry.shape != (3,) or goal.shape != (3,) or not np.isfinite([entry, goal]).all() or
                np.linalg.norm(delta) > constraints['entry']['radiusMm']+1e-7 or abs(delta@axis) > 1e-7 or
                np.linalg.norm(goal-constraints['target']['centerMm']) > constraints['target']['radiusMm']+1e-7):
            raise ValueError('Interactive route escapes requested source constraints')
        ijk = engine.scene.inverse[:3, :3]@goal+engine.scene.inverse[:3, 3]
        if (not np.allclose(ijk, np.rint(ijk), atol=1e-7) or np.any(ijk < 0) or
                np.any(ijk >= engine.scene.shape) or
                int(np.ravel_multi_index(np.rint(ijk).astype(int), engine.scene.shape)) not in retained):
            raise ValueError('Interactive route goal is not a retained source target')
        inward = (goal-entry)/np.linalg.norm(goal-entry)
        contact_points = engine.scene.points(route['requiredTargetRemoval']['sourceCells'])
        if np.any(np.linalg.norm(contact_points-constraints['target']['centerMm'], axis=1) > constraints['target']['radiusMm']+1e-7):
            raise ValueError('Interactive route contact escapes target subregion')
        for other_entry, other_axis in previous:
            angle = np.degrees(np.arccos(np.clip(inward@other_axis, -1., 1.)))
            if (np.linalg.norm(entry-other_entry)+1e-7 < constraints['minEntrySeparationMm'] or
                    angle+1e-7 < constraints['minAngleDeg']):
                raise ValueError('Interactive route separation differs')
        previous.append((entry, inward))
        expected, rejection = post_exposure_proposal(engine, entry, goal, constraints['tool']['radiusMm'], lambda: None,
            non_target_clearance_mm=constraints['nonTargetClearanceMm'], stop_at_first_contact=admitted is not None or public_task is not None)
        if public_task is not None and not rejection:
            keys = {*expected, 'id', 'label', 'separationComparisons', 'publicExecution'}
            if set(route) != keys:
                raise ValueError('Interactive public route schema differs')
            comparisons = [{'routeId': report['routes'][j]['id'],
                'entrySeparationMm': float(np.linalg.norm(entry-other_entry)),
                'angleDeg': float(np.degrees(np.arccos(np.clip(inward@other_axis, -1., 1.))))}
                for j, (other_entry, other_axis) in enumerate(previous[:-1])]
            if semantic_digest(route.get('separationComparisons')) != semantic_digest(comparisons):
                raise ValueError('Interactive public separation receipts differ')
        if (rejection or any(route.get(key) != value for key, value in expected.items()) or
                route['id'] != 'approach-'+str(index) or route.get('label') != 'Source approach '+str(index)):
            raise ValueError('Interactive approach program or geometry metrics differ')


def _validate_diagnostics(engine, report, source_axis):
    """A located marker must identify a retained source owner in its checked ray."""
    located = []
    constraints = report['constraints']
    owner = engine.occupancy()
    target_ids = _target_ids(engine)
    for rejection in report['rejections']:
        if 'positionMm' not in rejection:
            continue
        source_cell, occupancy_cell = rejection.get('limitingSourceCell'), rejection.get('occupancyCell')
        if (type(source_cell) is not int or type(occupancy_cell) is not int or
                not 0 <= source_cell < engine.scene.labels.size or not 0 <= occupancy_cell < engine.scene.labels.size or
                source_cell not in owner.material_ids([occupancy_cell])):
            raise ValueError('Interactive limiting source identity differs')
        region = engine.by_label[int(engine.scene.labels.ravel()[source_cell])]
        if (rejection.get('structureName') != region['name'] or
                rejection['positionMm'] != engine.scene.points([occupancy_cell])[0].tolist() or
                rejection.get('sourcePositionMm') != engine.scene.points([source_cell])[0].tolist() or
                rejection.get('part') not in ('shaft', 'working')):
            raise ValueError('Interactive limiting source location differs')
        entry, goal = np.asarray(rejection.get('entryPointMm'), float), np.asarray(rejection.get('sourceTargetPointMm'), float)
        if (entry.shape != (3,) or goal.shape != (3,) or not np.isfinite([entry, goal]).all() or
                np.linalg.norm(entry-constraints['entry']['centerMm']) > constraints['entry']['radiusMm']+1e-7 or
                abs((entry-constraints['entry']['centerMm'])@source_axis) > 1e-7 or
                np.linalg.norm(goal-constraints['target']['centerMm']) > constraints['target']['radiusMm']+1e-7 or
                np.linalg.norm(goal-entry) < 1.):
            raise ValueError('Interactive limiting ray escapes requested constraints')
        inward = (goal-entry)/np.linalg.norm(goal-entry)
        quaternion = Rotation.align_vectors([inward], [[0., 0., 1.]])[0].as_quat()
        radius = constraints['tool']['radiusMm']
        cover = _axial_cover(engine, _pose(entry-inward*max(30., radius+5.), quaternion),
                             _pose(goal, quaternion), _params(engine, radius))
        reason = rejection['reason'].removeprefix('WITHDRAWAL_')
        if reason == 'NON_TARGET_CLEARANCE':
            margin = constraints['nonTargetClearanceMm']
        elif reason in ('SHAFT_COLLISION', 'NON_TARGET_TRAVERSAL_REQUIRED', 'EXPOSURE_ENTRY_BLOCKED'):
            margin = 0.
        else:
            raise ValueError('Interactive limiting location has no source-contact reason')
        if ((reason in ('NON_TARGET_CLEARANCE', 'NON_TARGET_TRAVERSAL_REQUIRED') and region['id'] in target_ids) or
                (reason == 'SHAFT_COLLISION' and rejection['part'] != 'shaft') or
                not any(part == rejection['part'] and occupancy_cell in engine.cells(a, b, r+margin)
                        for a, b, r, part in cover)):
            raise ValueError('Interactive limiting source is outside the checked capsule cover')
        if len(located) < 12:
            located.append({'name': rejection['structureName'], **{key: value for key, value in rejection.items()
                if key not in ('proposal', 'entryPointMm', 'sourceTargetPointMm')}})
    if report.get('limitingStructures') != located:
        raise ValueError('Interactive limiting structure projection differs')


def materialize_interactive_approach(engine, report, candidate_id, *, check=lambda: None):
    """Return ``(approach_engine, receipt)`` after exact access AND withdrawal.

    The returned clone ends inactive and inserted at precontact. Withdrawal is
    validated on a separate clone. Original physics/guards are never replaced.
    """
    validate_interactive_report(engine, report)
    candidates = [r for r in report['routes'] if r.get('id') == candidate_id]
    if len(candidates) != 1:
        raise ValueError('Unknown interactive approach candidate')
    route = candidates[0]
    check_program(engine, route['actionProgram']+route['withdrawalProgram'])
    candidate = engine.clone(); before_removed = list(engine.state['removed']); receipts = []
    for action in route['actionProgram']:
        check(); result = candidate.step(copy.deepcopy(action), check=check)
        if result['status'] != 'accepted':
            raise ValueError('Public approach refused: '+str(result.get('reason')))
        receipts.append({'status': result['status'], 'stateId': candidate.state_id})
    cleanup = candidate.clone()
    for action in route['withdrawalProgram']:
        check(); result = cleanup.step(copy.deepcopy(action), check=check)
        if result['status'] != 'accepted':
            raise ValueError('Public withdrawal refused: '+str(result.get('reason')))
    if report['schema'] == OBSERVED_SCHEMA:
        from .surgical_planning_observation import physical_state_hash
        if (physical_state_hash(candidate) != route['publicExecution']['approachPhysicalStateHash'] or
                physical_state_hash(cleanup) != route['publicExecution']['withdrawalPhysicalStateHash']):
            raise ValueError('Native execution differs from sealed public prediction; no branch installed')
    if candidate.state['removed'] != before_removed or cleanup.state['removed'] != before_removed:
        raise ValueError('Inactive approach changed source tissue; candidate refused')
    return candidate, {'candidateId': candidate_id, 'reportSeal': report['reportSeal'],
        'startStateId': engine.state_id, 'approachStateId': candidate.state_id, 'withdrawalStateId': cleanup.state_id,
        **({'publicPlanSeal': report['publicPlanSeal'], 'publicInputHash': report['publicObservation']['inputHash'],
            'nativeMatchesPublicPrediction': True} if report['schema'] == OBSERVED_SCHEMA else {}),
        'actionCount': len(route['actionProgram']), 'withdrawalActionCount': len(route['withdrawalProgram']),
        'publicAccessAccepted': True, 'publicWithdrawalAccepted': True, 'exactPhysicsPreserved': True,
        'additionalTargetRemovedMm3': 0., 'additionalNonTargetRemovedMm3': 0., 'receipts': receipts,
        'scope': _ADMITTED_SCOPE if admission(engine) is not None else _SCOPE}


materialize = materialize_interactive_approach


def evaluate_sealed_contact_references(report, load_references):
    """Post-seal reference-contact diagnostic; never feeds proposal/STOP code.

    This is not clinical harm or independent complete-tool verification. The
    caller owns reference provenance and evaluation-split admission. No source
    mask is loaded until the public plan has been checked and frozen.
    """
    from .core import freeze_json, thaw_json
    body = _public_report_body(report)
    if semantic_digest(body) != report.get('publicPlanSeal'):
        raise ValueError('Public plan must be sealed before evaluation references are loaded')
    sealed = freeze_json(body)
    plan_seal = report['publicPlanSeal']
    references = load_references()
    body = thaw_json(sealed)  # A loader cannot mutate the plan being evaluated.
    if type(references) is not dict:
        raise ValueError('Named boolean reference masks are required')
    shape = tuple(body['publicObservation']['shape'])
    results = {}
    for name, mask in references.items():
        if type(name) is not str or type(mask) is not np.ndarray or mask.dtype != np.bool_ or mask.shape != shape:
            raise ValueError('Evaluation reference must be a named boolean mask on the declared grid')
        results[name] = {route['id']: int(np.count_nonzero(mask.ravel()[route['requiredTargetRemoval']['sourceCells']]))
                         for route in body['routes']}
    return {'schema': 'resectionlab.post-seal-reference-contacts.v1', 'publicPlanSeal': plan_seal,
            'publicInputHash': body['publicObservation']['inputHash'], 'referenceContactCells': results,
            'scope': 'Reference membership of proposed retained target contacts; no clinical outcome estimate'}
