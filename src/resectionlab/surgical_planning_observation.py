"""Public inputs for the bounded, annotation-assisted inactive-access planner.

Projection is an allowlist, never a rich engine clone with fields deleted.
Native source identity/provenance and evaluator references belong to the outer
qualification/installation envelope. This track deliberately exposes annotated
material ownership: it is not an MRI-only or partial-observation claim.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .core import array_digest, freeze_json, immutable_array, semantic_digest, thaw_json
from .surgical_engine import MAX_STEPS, SurgicalEngine
from .surgical_scene import SurgicalScene, MATERIALS
from .surgical_tools import TOOLS, catalog

SCHEMA = 'resectionlab.surgical-planning-observation.v1'
TRACK = 'annotation-assisted-inactive-access-v1'
_SOURCE = {'kind': 'public-planning-observation', 'track': TRACK}
_PROVENANCE = {'kind': 'assumed-research', 'description': 'Declared uncalibrated public planning parameters.'}
_EMPTY_LISTS = ('cuts', 'separated', 'vesselEncounters', 'sources', 'cutEdges', 'separatedAttachments',
                'transectedEdges', 'flapComponents', 'meshCutEdges', 'meshSeams')
_SCALARS = ('elapsedSeconds', 'travelMm', 'fluidAddedMm3', 'fluidRemovedMm3', 'fluidEscapedMm3', 'debrisMm3')
_MAPS = ('fluid', 'thermal', 'contact', 'debris')
_CUSA_MAPS = ('progressMm3', 'progressExact', 'contactSeconds')
_CUSA_SCALARS = ('inputCapacityMm3', 'aspiratedSolidMm3', 'unusedCapacityMm3', 'discardedCapacityMm3', 'effectiveActivationSeconds')
_DRILL_MAPS = ('workJ', 'temperatureC', 'thermalDoseCSeconds')
_DRILL_SCALARS = ('inputEnergyJ', 'cuttingWorkJ', 'heatGeneratedJ', 'coolingLossJ', 'unusedEnergyJ', 'removedVolumeMm3')
_VESSEL_SCALARS = ('nextWound', 'remainingMm3', 'refilledMm3', 'generatedMm3', 'initialMm3', 'elapsedSeconds', 'pressurePa')
_PARAMETERS = ('radiusMm', 'depthMm', 'flowMm3PerSecond', 'power', 'jawOpeningMm', 'amplitudeMa',
               'frequencyHz', 'pulseWidthUs', 'rpm', 'forceN', 'duty')


def _require(ok, reason):
    if not ok: raise ValueError('PLANNING_OBSERVATION: '+reason)


def _keys(value, keys, name):
    _require(type(value) is dict and set(value) == set(keys), name+' has an unsupported schema')


def _number(value, name):
    _require(type(value) in (int, float) and np.isfinite(value), name+' must be finite numeric state')
    return value


def _vector(value, size, name):
    _require(type(value) is list and len(value) == size, name+' must be a numeric vector')
    return [_number(x, name) for x in value]


def _cells(value, size, name):
    _require(type(value) is list and all(type(x) is int and 0 <= x < size for x in value), name+' must index the public grid')
    return value[:]


def _cell_map(value, size, name, *, exact=False):
    _require(type(value) is dict, name+' must be a numeric cell map')
    result = {}
    for key, row in value.items():
        _require(type(key) is str and key.isdecimal() and str(int(key)) == key and int(key) < size,
                 name+' has a non-cell key')
        if exact:
            _require(type(row) is list and len(row) == 2 and all(type(x) is int for x in row) and row[0] >= 0 and row[1] > 0,
                     name+' requires exact nonnegative capacity fractions')
            result[key] = row[:]
        else:
            result[key] = _number(row, name)
    return result


def _profile(scene):
    # The native validators check complete physical schemas; arbitrary nested
    # provenance is then replaced, rather than hashed into a decision input.
    _require('tissuePhysicsV2' in scene.properties and not set(scene.properties)-{'tissuePhysicsV2', 'cusaV3'},
             'this track supports only the V2 lattice and optional CUSA V3 profiles')
    scene._validate_properties()
    props = copy.deepcopy(scene.properties)
    props['tissuePhysicsV2']['provenance'] = dict(_PROVENANCE)
    from .surgical_deformation_v2 import profile
    # These two strings are descriptions, not material/solver selectors.
    for key in ('model', 'units'): props['tissuePhysicsV2']['brain'][key] = profile()[key]
    if 'cusaV3' in props: props['cusaV3']['provenance'] = dict(_PROVENANCE)
    return props


def _state(engine):
    s = engine.state; size = engine.scene.labels.size
    expected = {*_EMPTY_LISTS, *_SCALARS, *_MAPS, 'removed', 'holds', 'observations', 'slots',
                'materialExposure', 'deformation', 'deformationModel', 'drillingV2', 'vesselsV2',
                'cusaV3', 'toolMechanicsRevision', 'physicalClockRevision'}
    _keys(s, expected, 'physical state')
    _require(s['toolMechanicsRevision'] == 2 and s['physicalClockRevision'] == 1,
             'mechanics revision 2 and shared clock revision 1 are required')
    result = {'toolMechanicsRevision': 2, 'physicalClockRevision': 1}
    for name in _EMPTY_LISTS:
        _require(s[name] == [] and type(s[name]) is list, 'nonempty '+name+' is outside this task track')
        result[name] = []
    for name in ('holds', 'deformation'):
        _require(type(s[name]) is dict and not s[name], 'nonempty '+name+' is outside this task track')
        result[name] = {}
    # Stimulation readouts are historical outputs, never dynamics. No mapping
    # or information-acquisition action is supported by this track.
    _require(type(s['observations']) is list, 'observations must be a historical output list')
    result['observations'] = []
    result['removed'] = _cells(s['removed'], size, 'removed')
    for name in _SCALARS: result[name] = _number(s[name], name)
    for name in _MAPS: result[name] = _cell_map(s[name], size, name)
    _require(type(s['materialExposure']) is dict and not set(s['materialExposure'])-set(MATERIALS), 'unknown material exposure key')
    result['materialExposure'] = {key: _number(row, 'materialExposure') for key, row in s['materialExposure'].items()}
    _keys(s['slots'], ('left', 'right'), 'slots'); result['slots'] = {}
    for key, slot in s['slots'].items():
        _keys(slot, ('instrument', 'position', 'quaternion', 'active', 'inserted', 'grip', 'parameters'), 'tool slot')
        _require(slot['instrument'] in TOOLS and slot['active'] is False and slot['inserted'] is False and slot['grip'] == [],
                 'both instruments must be inactive, withdrawn and released')
        _require(type(slot['parameters']) is dict and not set(slot['parameters'])-set(_PARAMETERS), 'unknown tool parameter')
        result['slots'][key] = {'instrument': slot['instrument'], 'position': _vector(slot['position'], 3, 'position'),
            'quaternion': _vector(slot['quaternion'], 4, 'quaternion'), 'active': False, 'inserted': False, 'grip': [],
            'parameters': {n: _number(v, n) for n, v in slot['parameters'].items()}}
    d = s['deformationModel']; _keys(d, ('active', 'anchors', 'velocities', 'remainderSeconds'), 'deformation ledger')
    _require(d['active'] == [] and d['anchors'] == [] and d['velocities'] == {}, 'pending deformation is outside this task track')
    result['deformationModel'] = dict(active=[], anchors=[], velocities={}, remainderSeconds=_number(d['remainderSeconds'], 'deformation remainder'))
    d = s['drillingV2']; _keys(d, (*_DRILL_MAPS, *_DRILL_SCALARS), 'drilling ledger')
    result['drillingV2'] = {n: _cell_map(d[n], size, n) for n in _DRILL_MAPS}
    result['drillingV2'].update({n: _number(d[n], n) for n in _DRILL_SCALARS})
    d = s['cusaV3']; _keys(d, ('schema', *_CUSA_MAPS, *_CUSA_SCALARS), 'CUSA ledger')
    _require(d['schema'] == 'resectionlab.cusa-contact.v3', 'unsupported CUSA state')
    result['cusaV3'] = {'schema': d['schema'], **{n: _cell_map(d[n], size, n, exact=n == 'progressExact') for n in _CUSA_MAPS},
                        **{n: _number(d[n], n) for n in _CUSA_SCALARS}}
    d = s['vesselsV2']; _keys(d, ('faces', 'wounds', 'contactDose', *_VESSEL_SCALARS), 'vascular ledger')
    for n in ('faces', 'wounds', 'contactDose'):
        _require(type(d[n]) is dict and not d[n], 'pending vascular '+n+' is outside this task track')
    result['vesselsV2'] = {n: {} for n in ('faces', 'wounds', 'contactDose')}
    result['vesselsV2'].update({n: _number(d[n], n) for n in _VESSEL_SCALARS})
    return result


class _PublicPlanningEngine(SurgicalEngine):
    """Fresh public dynamics with the native remaining action budget."""
    def _validate(self, action):
        _require(len(self.actions) < self._remaining_actions, 'native remaining action budget exhausted')
        _require(action.get('operation') in ('exchange', 'place', 'move', 'withdraw'), 'operation exceeds inactive task')
        slot = self.state['slots'].get(action.get('slot'), {})
        tool = action.get('instrument', slot.get('instrument')) if action.get('operation') == 'exchange' else slot.get('instrument')
        _require(tool == 'cusa', 'instrument exceeds inactive task')
        return super()._validate(action)

    def fresh(self): raise ValueError('PLANNING_OBSERVATION: cannot reset observed anatomy or burdens')
    def reset(self, scene=None): return self.fresh()
    def replay(self, actions): return self.fresh()


def _validate_fields(labels, affine, fields):
    from types import SimpleNamespace
    _keys(fields, ('regions', 'properties', 'state', 'guard', 'task', 'availableEvidence',
                   'remainingActions', 'instrumentCatalog'), 'public observation')
    _require(type(fields['regions']) is list, 'public regions must be a list')
    for row in fields['regions']:
        _keys(row, ('id', 'name', 'material', 'target'), 'public region')
        _require(type(row['target']) is bool and row['name'] == row['material']+' '+str(row['id']),
                 'region display metadata must be canonical')
    scene = SurgicalScene(labels, affine, fields['regions'], _SOURCE, fields['properties'])
    _require(_profile(scene) == fields['properties'], 'profile metadata must be canonical')
    _require(_state(SimpleNamespace(scene=scene, state=fields['state'])) == fields['state'],
             'state contains unprojected output metadata')
    _require(type(fields['remainingActions']) is int and 0 <= fields['remainingActions'] <= MAX_STEPS,
             'remaining native actions must be bounded')
    _require(fields['instrumentCatalog'] == catalog(), 'instrument catalog differs')
    task = fields['task']
    _keys(task, ('entry', 'target', 'axisRasUnit', 'requiredEvidence', 'toolRadiusMm'), 'public task')
    from .surgical_task_admission import _sphere
    for name in ('entry', 'target'): _sphere(task[name])
    axis = _vector(task['axisRasUnit'], 3, 'proposal axis')
    _require(np.isclose(np.linalg.norm(axis), 1., rtol=0., atol=1e-12), 'proposal direction must be unit length')
    _require(task['toolRadiusMm'] == 1.5, 'fixed catalog CUSA radius required')
    for value in (task['requiredEvidence'], fields['availableEvidence']):
        _require(type(value) is list and all(type(x) is str and 0 < len(x) <= 128 for x in value)
                 and value == sorted(set(value)), 'evidence roles must be canonical')
    guard = fields['guard']
    if guard is not None:
        _keys(guard, ('clearanceMm', 'required', 'config', 'annotations'), 'public guard')
        _keys(guard['config'], ('require_critical_coverage', 'critical_max_displacement_mm'), 'public guard config')
        from .surgical_rl_protection import GuardConfig
        GuardConfig(**guard['config'])
        _require(_number(guard['clearanceMm'], 'clearance') >= 0., 'clearance must be nonnegative')
        _require(type(guard['annotations']) is list, 'public exclusion masks must be a list')
        _require(type(guard['required']) is list and all(type(x) is str and x for x in guard['required']), 'required mask roles must be named')
        for row in guard['annotations']:
            _keys(row, ('id', 'cells', 'reviewStatus'), 'public exclusion mask')
            _require(type(row['id']) is str and row['id'] and row['reviewStatus'] in ('reviewed', 'unreviewed'),
                     'public mask role/review status differs')
            _cells(row['cells'], labels.size, 'protection cells')
            _require(row['cells'] == sorted(set(row['cells'])), 'protection cells must be canonical')
        roles = [row['id'] for row in guard['annotations']]
        _require(len(set(roles)) == len(roles), 'public mask roles must be unique')
        present = {row['id'] for row in guard['annotations'] if row['cells']}
        _require(not guard['config']['require_critical_coverage'] or not ((set(guard['required']) | {'brainstem'})-present),
                 'required public critical masks are missing')


@dataclass(frozen=True)
class PlanningObservation:
    """No engine, source identity, callbacks, caches or evaluation references."""
    labels: np.ndarray
    affine: np.ndarray
    fields: object

    def __post_init__(self):
        labels = np.asarray(self.labels)
        _require(labels.ndim == 3 and labels.dtype.kind in 'iu' and labels.size > 0 and labels.min() >= 0 and labels.max() < 65536,
                 'public labels must be a nonnegative uint16-compatible volume')
        labels = immutable_array(labels, np.uint16)
        affine = immutable_array(self.affine, np.float64)
        fields = freeze_json(self.fields)
        _validate_fields(labels, affine, thaw_json(fields))
        object.__setattr__(self, 'labels', labels)
        object.__setattr__(self, 'affine', affine)
        object.__setattr__(self, 'fields', fields)

    def manifest(self):
        fields = thaw_json(self.fields)
        parts = {'labelsHash': array_digest(self.labels), 'affineHash': array_digest(self.affine),
                 **{key+'Hash': semantic_digest(value) for key, value in fields.items()}}
        return {'schema': SCHEMA, 'track': TRACK, 'components': parts,
                'inputHash': semantic_digest({'schema': SCHEMA, 'track': TRACK, **parts}), 'shape': list(self.labels.shape),
                'remainingActions': fields['remainingActions'], 'evaluationReferencesExposed': False}

    def planning_engine(self):
        fields = thaw_json(self.fields)
        _require(fields['instrumentCatalog'] == catalog(), 'instrument catalog changed after observation')
        scene = SurgicalScene(self.labels, self.affine, fields['regions'], _SOURCE, fields['properties'])
        protection = config = None
        guard = fields['guard']
        if guard is not None:
            from .surgical_rl_protection import ProtectionAnnotation, ProtectionSet, GuardConfig, grid_binding
            annotations = [ProtectionAnnotation(row['id'], tuple(row['cells']), grid_binding(scene),
                'Declared public exclusion mask', row['reviewStatus'], row['id']) for row in guard['annotations']]
            protection = ProtectionSet(scene, annotations, clearance_mm=guard['clearanceMm'], required=guard['required'])
            config = GuardConfig(**guard['config'])
        engine = _PublicPlanningEngine(scene, protection=protection, guard_config=config)
        engine.state = fields['state']
        engine._remaining_actions = fields['remainingActions']
        return engine


def project_observation(engine, task, available_evidence):
    """Qualification wrapper only: explicitly project declared public fields.

    ``task`` is the validated native inactive task. Public consumers receive
    only PlanningObservation, never this wrapper's rich engine arguments.
    """
    _require(task.get('id') == 'post-exposure-inactive-access-v1' and task.get('inputContract') == 'annotation-assisted-source-geometry-v1',
             'unsupported task/observation contract')
    _require(type(available_evidence) is list and all(type(x) is str for x in available_evidence), 'evidence roles must be declared')
    _require(0 <= len(engine.actions) <= MAX_STEPS, 'invalid native action count')
    state = _state(engine)
    guard = None
    if engine.protection is not None:
        from dataclasses import asdict
        guard = {'clearanceMm': engine.protection.clearance_mm, 'required': list(engine.protection.required),
            'config': asdict(engine.guard_config), 'annotations': [{'id': a.structure_id, 'cells': list(a.cells),
                'reviewStatus': 'reviewed' if a.review_status == 'reviewed' else 'unreviewed'} for a in engine.protection.annotations]}
    # Stable semantic roles replace display names, colors and source narratives.
    regions = [{'id': r['id'], 'name': r['material']+' '+str(r['id']), 'material': r['material'],
                'target': bool(r.get('target'))} for r in engine.scene.regions]
    public_task = {key: copy.deepcopy(task[key]) for key in ('entry', 'target', 'axisRasUnit', 'requiredEvidence', 'toolRadiusMm')}
    fields = {'regions': regions, 'properties': _profile(engine.scene), 'state': state, 'guard': guard,
              'task': public_task, 'availableEvidence': sorted(set(available_evidence)),
              'remainingActions': MAX_STEPS-len(engine.actions), 'instrumentCatalog': catalog()}
    return PlanningObservation(engine.scene.labels, engine.scene.affine, fields)


def physical_state_hash(engine):
    """Dynamics and accumulated burdens, excluding historical mapping outputs."""
    return semantic_digest({key: value for key, value in engine.state.items() if key != 'observations'})


def preview_inactive_program(engine, route, *, check=lambda: None):
    _require(type(engine) is _PublicPlanningEngine, 'preview requires the projected public model')
    candidate = engine.clone(); before = engine.state['removed'][:]
    for action in route['actionProgram']:
        check(); result = candidate.step(copy.deepcopy(action), check=check)
        _require(result['status'] == 'accepted', 'public access refused: '+str(result.get('reason')))
    approach_hash = physical_state_hash(candidate)
    for action in route['withdrawalProgram']:
        check(); result = candidate.step(copy.deepcopy(action), check=check)
        _require(result['status'] == 'accepted', 'public withdrawal refused: '+str(result.get('reason')))
    _require(candidate.state['removed'] == before, 'inactive access changed source tissue')
    return {'approachPhysicalStateHash': approach_hash, 'withdrawalPhysicalStateHash': physical_state_hash(candidate),
            'suffixActionHash': semantic_digest(route['actionProgram']+route['withdrawalProgram']),
            'accessAccepted': True, 'withdrawalAccepted': True, 'remainingNativeActions': candidate._remaining_actions-len(candidate.actions)}


def validate_observation_scope(engine):
    _profile(engine.scene)
    _state(engine)
    return True
