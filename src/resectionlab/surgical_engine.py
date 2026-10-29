"""Deterministic instrument transitions, independent of UI and wall clock.

Source cells retain their identity through removal and reversible, grid-snapped
patch translations. Collision and display both consume this SAME occupancy.
This deliberately coarse deformation model is not FEM or measured mechanics.
"""
from __future__ import annotations

import copy
import builtins as _removed_builtins
import _operator as _removed_operator
import hashlib
import json
import math
import itertools
from itertools import repeat as _strict_int_repeat
from operator import is_ as _strict_int_is
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .core import semantic_digest
from .geometry import GeometryScene, capsule_voxel_indices, segment_segment_distance
from .surgical_tools import TOOLS, catalog
from .surgical_cusa_v3 import profile as cusa_profile

SCHEMA = 'resectionlab.surgical-toolbox.v1'
MAX_STEPS = 8192

# Admission inspects bindings without invoking mutable caller type/all/map
# dispatch. The original inline predicates below still own every fallback.
_removed_stock_type = ().__class__.__class__
_removed_stock_list, _removed_stock_len = [].__class__, [].__class__.__len__
_removed_stock_dict = {}.__class__
_removed_c_function_type = [].append.__class__
_removed_stock_is, _removed_stock_repeat = _strict_int_is, _strict_int_repeat
_removed_stock_bindings = (('type', _removed_builtins.type), ('int', _removed_builtins.int),
                          ('all', _removed_builtins.all), ('map', _removed_builtins.map),
                          ('list', _removed_builtins.list), ('__import__', _removed_builtins.__import__))


def _removed_known_c_function(value, name, owner, module_name):
    return (_removed_stock_type(value) is _removed_c_function_type and
            value.__self__ is owner and value.__name__ == name and
            _removed_stock_type(value.__module__) is ''.__class__ and value.__module__ == module_name)


# Recognize captures before trusting later identity equality. In particular,
# Python replacements already installed at import must retain legacy dispatch.
# repeat is a genuine heap type: its own C __new__ binds it without rejecting
# heap types generally or invoking a captured constructor/descriptor.
_removed_repeat_new = (_removed_stock_repeat.__dict__.get('__new__')
                       if _removed_stock_type(_removed_stock_repeat) is _removed_stock_type else None)
_removed_captures_genuine = (
    _removed_stock_bindings[0][1] is _removed_stock_type and
    _removed_stock_bindings[1][1] is (0).__class__ and
    _removed_stock_bindings[4][1] is _removed_stock_list and
    _removed_known_c_function(_removed_stock_bindings[2][1], 'all', _removed_builtins, 'builtins') and
    _removed_known_c_function(_removed_stock_bindings[5][1], '__import__', _removed_builtins, 'builtins') and
    _removed_known_c_function(_removed_stock_is, 'is_', _removed_operator, '_operator') and
    _removed_stock_type(_removed_stock_bindings[3][1]) is _removed_stock_type and
    not (_removed_stock_bindings[3][1].__flags__ & 512) and
    _removed_stock_bindings[3][1].__name__ == 'map' and
    _removed_stock_type(_removed_stock_bindings[3][1].__module__) is ''.__class__ and
    _removed_stock_bindings[3][1].__module__ == 'builtins' and
    _removed_stock_type(_removed_stock_repeat) is _removed_stock_type and
    _removed_stock_repeat.__name__ == 'repeat' and
    _removed_stock_type(_removed_stock_repeat.__module__) is ''.__class__ and
    _removed_stock_repeat.__module__ == 'itertools' and
    _removed_stock_type(_removed_repeat_new) is _removed_c_function_type and
    _removed_repeat_new.__self__ is _removed_stock_repeat and _removed_repeat_new.__name__ == '__new__')


def _try_removed_native(value):
    if not _removed_captures_genuine:
        return None
    # Keep the fixed live-identity check off smaller ledgers; explicit adapter
    # calls still validate the selection for every input size.
    if _removed_stock_type(value) is not _removed_stock_list or _removed_stock_len(value) < 32768:
        return None
    if (_removed_stock_type(_removed_caller_globals) is not _removed_stock_dict or
            _removed_stock_type(_removed_caller_builtins) is not _removed_stock_dict):
        return None
    for name, original in _removed_stock_bindings:
        if (_removed_caller_globals.get(name, original) is not original or
                _removed_caller_builtins.get(name) is not original):
            return None
    if (_removed_caller_globals.get('_strict_int_is') is not _removed_stock_is or
            _removed_caller_globals.get('_strict_int_repeat') is not _removed_stock_repeat):
        return None
    from .surgical_idle_native import try_exact_int_list
    return try_exact_int_list(value)


# Function lookup uses the execution namespace's builtin mapping, which may be
# a copy rather than builtins.__dict__. Keep the actual caller mapping bound.
_removed_caller_globals = _try_removed_native.__globals__
_removed_caller_builtins = _try_removed_native.__builtins__


def _owned_digest(value):
    """Canonical hash of synchronously owned engine JSON, without detaching it.

    Transitions and cursor reads are serialized by the engine owner. Validate
    exact built-in containers/leaves before encoding so custom objects retain
    semantic_digest's validation/conversion path. No mutable copy escapes.
    """
    pending, seen = [value], set()
    atomic = (str, int, float, bool, type(None))
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        kind = type(current)
        if kind is dict:
            if any(type(key) is not str for key in current):
                return semantic_digest(value)
            children = current.values()
        elif kind is list or kind is tuple:
            children = current
        else:
            return semantic_digest(value)
        for item in children:
            child_kind = type(item)
            if child_kind is dict or child_kind is list or child_kind is tuple:
                pending.append(item)
            elif child_kind not in atomic:
                return semantic_digest(value)
    try:
        payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError):
        return semantic_digest(value)
    return 'sha256:' + hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _copy_engine_data(value, memo=None):
    """Detach engine JSON graphs without visiting every scalar via deepcopy.

    Long histories contain large flat cell/pose/event arrays. Built-in scalar
    leaves are immutable; copying their containing list/dict in C preserves
    their exact values and order. The shared memo retains aliases and cycles.
    Subclasses and other values keep ordinary deepcopy behavior.
    """
    memo = {} if memo is None else memo
    atomic = (str, int, float, bool, type(None))
    kind = type(value)
    if kind in atomic:
        return value
    if id(value) in memo:
        return memo[id(value)]
    if kind is list:
        if all(type(item) in atomic for item in value):
            result = value.copy()
            memo[id(value)] = result
            return result
        result = []
        memo[id(value)] = result
        result.extend(_copy_engine_data(item, memo) for item in value)
        return result
    if kind is dict and all(type(key) in atomic for key in value):
        if all(type(item) in atomic for item in value.values()):
            result = value.copy()
            memo[id(value)] = result
            return result
        result = {}
        memo[id(value)] = result
        for key, item in value.items():
            result[key] = _copy_engine_data(item, memo)
        return result
    return copy.deepcopy(value, memo)


class _OperationCheckFailed(Exception):
    """Keep cancellation callbacks out of surgical domain-error handlers."""
    def __init__(self, cause):
        self.cause = cause


def _checked_operation(check):
    try:
        check()
    except BaseException as cause:
        raise _OperationCheckFailed(cause) from None


def finite(value, size=None):
    a = np.asarray(value, float)
    if not np.isfinite(a).all() or size is not None and a.shape != (size,):
        raise ValueError('Finite coordinates with the declared dimension are required')
    return a


def rotation(q):
    q = finite(q, 4)
    if abs(np.linalg.norm(q) - 1) > 1e-5: raise ValueError('Orientation must be a unit xyzw quaternion')
    return Rotation.from_quat(q)


def _microstep_checkpoint(state):
    """Copy rollback state, preserving aliases and independently owned fields.

    Surgical state is a JSON data graph with built-in scalar leaves. Exact
    flat removed/debris containers and three scalar drilling maps can be copied
    in C and seeded into the deepcopy memo; all other data, including unusual
    nested/subclass values, still uses ordinary deepcopy. Fresh guards also
    handle public state edits.
    This preserves data-graph semantics, not adversarial custom __deepcopy__
    methods whose behavior depends on when another object enters the memo.

    Component summaries retain their existing exemption: microsteps never
    mutate them; finalize_action replaces them. Existing memo entries take
    precedence even if unusual public state aliases one of these fields.
    """
    components = state.get('flapComponents')
    memo = {id(components): components} if components is not None else {}
    for name in ('removed', 'debris'):
        value = state.get(name)
        if id(value) in memo:
            continue
        if name == 'removed':
            flat = type(value) is list and (_try_removed_native(value) is True or
                all(map(_strict_int_is, map(type, value), _strict_int_repeat(int))))
        else:
            flat = type(value) is dict and all(type(key) is str and type(amount) in (int, float)
                                               for key, amount in value.items())
        if flat:
            memo[id(value)] = value.copy()
    drilling = state.get('drillingV2')
    if type(drilling) is dict:
        for name in ('workJ', 'temperatureC', 'thermalDoseCSeconds'):
            value = drilling.get(name)
            if id(value) in memo:
                continue
            if type(value) is dict and all(type(key) is str and type(amount) in (int, float)
                                           for key, amount in value.items()):
                memo[id(value)] = value.copy()
    return copy.deepcopy(state, memo)


class SurgicalEngine:
    # Display adapters may reuse only fields with an authenticated resident
    # value. This advertises optional copy omission to older replay adapters;
    # ordinary/public views retain the complete detached payload.
    _view_omit_fields = frozenset({'flapComponents'})
    def __init__(self, scene, *, protection=None, guard_config=None):
        self.scene = scene
        from .surgical_rl_protection import GuardConfig, ProtectionSet, missing_critical, protection_contract
        self.protection = protection
        self.guard_config = guard_config
        self._protection_contract = None
        self._protected_ids = frozenset()
        if protection is not None or guard_config is not None:
            self.protection = protection if protection is not None else ProtectionSet(scene)
            self.guard_config = guard_config if guard_config is not None else GuardConfig()
            if not isinstance(self.guard_config, GuardConfig): raise ValueError('Shared GuardConfig required')
            if self.protection.scene.identity != scene.identity: raise ValueError('Protection scene mismatch')
            missing = missing_critical(self.protection)
            if self.guard_config.require_critical_coverage and missing:
                raise ValueError('MISSING_CRITICAL_ANNOTATIONS: ' + ', '.join(missing))
            self._protected_ids = frozenset(map(int, self.protection.cells))
            self._protection_contract = protection_contract(self.protection, self.guard_config)
        self.grid = GeometryScene(np.zeros(scene.shape, bool), scene.affine)
        from .surgical_geometry_cache import SurgicalCellCoverCache
        self._cell_cover_cache = SurgicalCellCoverCache(self.grid, scene.identity)
        self.by_label = {r['id']: r for r in scene.regions}
        corners = np.array(list(itertools.product(*[(0, n - 1) for n in scene.shape]))) @ scene.affine[:3, :3].T + scene.affine[:3, 3]
        center = corners.mean(axis=0)
        self.state = {'removed': [], 'cuts': [], 'separated': [], 'holds': {}, 'fluid': {}, 'thermal': {},
                      'contact': {}, 'vesselEncounters': [], 'materialExposure': {}, 'sources': copy.deepcopy(scene.properties.get('bleedingSources', [])),
                      'observations': [], 'elapsedSeconds': 0., 'travelMm': 0., 'fluidAddedMm3': 0.,
                      'fluidRemovedMm3': 0., 'fluidEscapedMm3': 0., 'debrisMm3': 0.,
                      'slots': {s: {'instrument': tool, 'position': [float(center[0] + dx), float(center[1]), float(corners[:, 2].min() - 5)],
                                    'quaternion': [0., 0., 0., 1.], 'active': False, 'inserted': False, 'grip': [], 'parameters': {}}
                                for s, tool, dx in [('left', 'cranial_drill', -6), ('right', 'cusa', 6)]}}
        self.actions = []
        self.history = []
        self._occupancy_cache = None
        from .surgical_effects import initialize_effect_state
        initialize_effect_state(self)

    @property
    def state_id(self):
        return _owned_digest({'scene': self.scene.identity, 'state': self.state,
            **({'protectionContract': self._protection_contract['contentHash']} if self._protection_contract else {})})

    def fresh(self):
        """Intact anatomy with the same evidence and guard policy."""
        return type(self)(self.scene, protection=self.protection, guard_config=self.guard_config)

    def guard_report(self):
        from .surgical_rl_protection import missing_critical
        if self.protection is None:
            return {'configured': False, 'mode': 'unconfigured-research', 'contractId': None,
                'status': 'unknown-coverage', 'missingRequired': ['brainstem'], 'clearanceMm': 0.,
                'criticalMaxDisplacementMm': None, 'anatomicalCompleteness': 'unknown',
                'clinicalValidation': 'none', 'structures': [], 'unknownStructures': ['critical anatomy not supplied']}
        missing = missing_critical(self.protection)
        return {**self.protection.report(), 'configured': True,
            'mode': 'strict-research' if self.guard_config.require_critical_coverage else 'relaxed-research',
            'contractId': self._protection_contract['contentHash'], 'missingRequired': list(missing),
            'status': 'unknown-coverage' if missing else 'declared-masks-only',
            'criticalMaxDisplacementMm': self.guard_config.critical_max_displacement_mm}

    def clone(self):
        other = copy.copy(self)
        other.state = _copy_engine_data(self.state)
        other.actions = _copy_engine_data(self.actions)
        other.history = _copy_engine_data(self.history)
        return other

    def _transition_clone(self):
        """Transactional state copy with append-only committed ancestry.

        Engine transitions never edit an earlier action or receipt. Sharing
        those entries while copying the two append targets avoids quadratic
        copying as a rehearsal grows. Public clone/view/snapshot still detach
        ancestry; callers should use those APIs when retaining editable copies.
        Invoke the subclass clone hook on an empty-ancestry shell so observers
        and reward ledgers preserve their transactional behavior too.
        """
        shell = copy.copy(self)
        shell.actions, shell.history = [], []
        other = shell.clone()
        other.actions, other.history = self.actions[:], self.history[:]
        return other

    def occupancy(self):
        changes = {'holds': self.state['holds'], **({'detachedFragments': self.state['detachedFragments']} if 'detachedFragments' in self.state else {}), **({'deformation': self.state['deformation']} if 'deformation' in self.state else {})}
        removed_ids = self.state['removed']
        # Built-in integer sequences have the same exact ordered identity in
        # tuples and canonical JSON. Avoid freezing/encoding thousands of IDs
        # at every occupancy lookup; snapshot the contents to detect in-place
        # edits. Other representations retain the complete legacy JSON key.
        if isinstance(removed_ids, list) and (
                (_try_removed_native(removed_ids) is True or
                 all(map(_strict_int_is, map(type, removed_ids), _strict_int_repeat(int))))
                if type(removed_ids) is list else all(type(cell) is int for cell in removed_ids)):
            key = (tuple(removed_ids), semantic_digest(changes))
        else:
            key = semantic_digest({'removed': removed_ids, **changes})
        if self._occupancy_cache is not None and self._occupancy_cache[0] == key:
            return self._occupancy_cache[1]
        from .surgical_occupancy import MaterialOccupancy
        removed = set(self.state['removed'])
        cleared = set(removed)
        moves = []
        held_ids = set()
        for hold in [*self.state['holds'].values(), *self.state.get('detachedFragments', [])]:
            if 'target' in hold: continue
            retained = np.array([i for i in hold['cells'] if i not in removed], dtype=np.int64)
            if not len(retained): continue
            if held_ids.intersection(retained): raise ValueError('MATERIAL_ALREADY_HELD')
            held_ids.update(retained)
            indices = np.column_stack(np.unravel_index(retained, self.scene.shape)) + hold['offset']
            if np.any(indices < 0) or np.any(indices >= self.scene.shape): raise ValueError('DISPLACEMENT_OUTSIDE_GRID')
            destination = np.ravel_multi_index(indices.T, self.scene.shape)
            cleared.update(retained)
            moves.append((retained, destination))
        destinations = {}
        base = self.scene.labels.ravel()
        for retained, destination in moves:
            for original, target in zip(retained, destination):
                if target in destinations or (base[target] and target not in cleared):
                    raise ValueError('DISPLACEMENT_COLLISION')
                destinations[int(target)] = int(original) + 1
        memberships = {}
        if 'deformation' in self.state:
            from .surgical_physics_v2 import raster
            pairs = raster(self)
            cleared.update(a for a, b in pairs)
            for original, target in pairs:
                if base[target] and target not in cleared:
                    material = self.by_label[int(base[target])]['material']
                    if material not in ('soft_tissue', 'tumor'): raise ValueError('DISPLACEMENT_COLLISION')
                    memberships.setdefault(target, []).append(target)
                memberships.setdefault(target, []).append(original)
            for target, originals in memberships.items():
                memberships[target] = sorted(set(originals))
                precedence = {r['id']: rank for rank, r in enumerate(self.scene.regions)}
                destinations[target] = max(originals, key=lambda c: (precedence[int(base[c])], c)) + 1
        owner = MaterialOccupancy(self.scene.labels, cleared, destinations, memberships)
        self._occupancy_cache = (key, owner)
        return owner

    def cells(self, a, b, radius):
        return self._cell_cover_cache.query(self.grid, a, b, radius, kernel=capsule_voxel_indices)

    def touched(self, a, b, radius):
        cells = self.cells(a, b, radius)
        return self.occupancy().material_ids(cells)

    def parameters(self, slot):
        tool = TOOLS[slot['instrument']]
        extra = {'rpm': 40000., 'forceN': 1., 'duty': 1.} if self.scene.properties.get('tissuePhysicsV2') else {}
        return {**extra, 'radiusMm': tool['radiusMm'], 'depthMm': 2., 'flowMm3PerSecond': 20., 'power': 1., 'jawOpeningMm': 2., **slot['parameters'], **({'contactGeometryRevision':2} if self.state.get('toolMechanicsRevision')==2 else {})}

    def preview(self, action, *, check=None, include_view=True, include_history=True):
        if check is not None: check()
        candidate = self._transition_clone()
        result = candidate.step(action) if check is None else candidate.step(action, check=check)
        view = candidate.view(_state_id=result['stateId'], include_history=include_history) if include_view else None
        if check is not None: check()
        return {**result, 'preview': True, **({'view': view} if include_view else {})}

    def reset(self, scene=None):
        if scene is not None and scene.identity != self.scene.identity:
            if self.guard_config is not None:
                raise ValueError('Construct a new engine with protection bound to the new scene')
            initial = type(self)(scene).__dict__
        else:
            initial = self.fresh().__dict__
        self.__dict__.clear()
        self.__dict__.update(initial)
        return self.observe()

    def replay(self, actions):
        result = self.fresh()
        for action in actions:
            if result.step(action)['status'] == 'rejected':
                raise ValueError('Replay contains an inadmissible action')
        return result

    def _validate(self, action):
        if not isinstance(action, dict) or set(action) - {'slot', 'operation', 'instrument', 'pose', 'path', 'parameters', 'duration', 'position', 'mechanicsRevision'}:
            raise ValueError('Unsupported action fields')
        if action.get('slot') not in self.state['slots']: raise ValueError('Choose left or right instrument slot')
        if action.get('operation') not in {'place', 'move', 'activate', 'deactivate', 'apply', 'hold', 'grasp', 'release', 'exchange', 'withdraw', 'extract', 'inspect', 'configure', 'position'}:
            raise ValueError('Unsupported instrument operation')
        duration = float(action.get('duration', 1.))
        if not math.isfinite(duration) or not 0 < duration <= 30: raise ValueError('Duration must be in (0, 30] seconds')
        if len(self.actions) >= MAX_STEPS: raise ValueError('Action limit reached; save and start a new episode')
        points = action.get('path', [action['pose']] if 'pose' in action else [])
        if 'mechanicsRevision' in action and (action['operation'] != 'configure' or type(action['mechanicsRevision']) is not int or action['mechanicsRevision'] not in (1,2)): raise ValueError('INVALID_INSTRUMENT_MECHANICS_REVISION')
        if 'position' in action and action['operation'] != 'position': raise ValueError('POSITION_PAYLOAD_REQUIRES_POSITION_OPERATION')
        if action['operation'] in ('configure', 'position'):
            if points or action.get('parameters') or action.get('instrument'): raise ValueError('SETUP_CANNOT_MOVE_INSTRUMENT')
            if action['operation'] == 'position':
                from .surgical_position import validate_position
                validate_position(action.get('position', {}))
        if not isinstance(points, list) or len(points) > 512: raise ValueError('Path requires at most 512 poses')
        for p in points:
            if set(p) != {'position', 'quaternion'}: raise ValueError('Each pose needs position and quaternion')
            finite(p['position'], 3); rotation(p['quaternion'])
        parameters = action.get('parameters', {})
        limits = {'radiusMm': (.1, 12.), 'depthMm': (.1, 20.), 'flowMm3PerSecond': (0., 1000.),
                  'power': (0., 10.), 'jawOpeningMm': (0., 12.),
                  'amplitudeMa': (0., 20.), 'frequencyHz': (0., 500.), 'pulseWidthUs': (0., 1000.)}
        if self.scene.properties.get('tissuePhysicsV2'):
            limits.update(rpm=(0., 80000.), forceN=(0., 20.), duty=(0., 1.))
        if set(parameters) - set(limits): raise ValueError('Unsupported instrument parameters')
        for name, value in parameters.items():
            if type(value) not in (int, float) or not math.isfinite(value) or not limits[name][0] <= value <= limits[name][1]:
                raise ValueError('Invalid parameter: ' + name)
        if self.scene.properties.get('tissuePhysicsV2'):
            slot = self.state['slots'][action['slot']]
            instrument = action.get('instrument', slot['instrument']) if action['operation'] == 'exchange' else slot['instrument']
            if instrument == 'cranial_drill':
                existing = {} if action['operation'] == 'exchange' else slot['parameters']
                radius = parameters.get('radiusMm', existing.get('radiusMm', TOOLS[instrument]['radiusMm']))
                if radius > self.scene.properties['tissuePhysicsV2']['bone']['maxRadiusMm']:
                    raise ValueError('DRILL_RADIUS_EXCEEDS_PROFILE: geometry and cutting radius must agree')
        return duration, points

    def step(self, action, *, check=None):
        if check is not None: check()
        before = self.state_id
        after = before
        before_time = self.state['elapsedSeconds']
        before_removed = set(self.state['removed'])
        candidate = self._transition_clone()
        try:
            duration, points = candidate._validate(action)
            result = candidate._execute(copy.deepcopy(action), duration, points, check=check)
        except _OperationCheckFailed as interrupted:
            raise interrupted.cause from None
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            if check is not None: check()
            return {'status': 'rejected', 'reason': str(error), 'reasonCode': str(error).split(':')[0] if str(error).split(':')[0].isupper() else 'INVALID_ACTION', 'completedFraction': 0., 'elapsedSeconds': 0., 'stateId': before, 'events': [], 'deltas': {}, 'metrics': self.metrics()}
        if check is not None: check()
        result['reasonCode'] = result['reason'].split(':')[0] if result.get('reason') else None
        result.setdefault('elapsedSeconds', candidate.state['elapsedSeconds'] - before_time if result['status'] != 'rejected' else 0.)
        if result['status'] != 'rejected':
            from .surgical_effects import finalize_action
            finalize_action(candidate)
            if check is not None: check()
            result['elapsedSeconds'] = candidate.state['elapsedSeconds'] - before_time
            # Finalization is the last state mutation. Reuse this exact digest
            # for result/history instead of serializing the large state three times.
            after = candidate.state_id
            result['deltas'] = {'removedCells': sorted(set(candidate.state['removed']) - before_removed),
                                'geometryRevision': after}
            candidate.actions.append(copy.deepcopy(action))
            candidate.history.append({'action': copy.deepcopy(action), 'stateBefore': before,
                                      'stateAfter': after, **_copy_engine_data(result)})
            if check is not None: check()
            self.__dict__.update(candidate.__dict__)
        return {**result, 'stateId': after, 'metrics': self.metrics()}

    def _execute(self, action, duration, points, *, check=None):
        key, op = action['slot'], action['operation']
        slot = self.state['slots'][key]
        previous_jaw = self.parameters(slot)['jawOpeningMm']
        events = []
        if op in ('configure', 'position'):
            if any(s['active'] or s['inserted'] for s in self.state['slots'].values()) or self.state['holds']:
                raise ValueError('WITHDRAW_BOTH_TOOLS_BEFORE_SETUP')
            if op == 'configure':
                self.state['toolMechanicsRevision'] = action.get('mechanicsRevision', 1)
                from .surgical_cusa_v3 import initialize
                initialize(self)
                events = [{'kind': 'instrument_mechanics_enabled', 'revision': self.state['toolMechanicsRevision']}]
            else:
                from .surgical_position import apply_position
                if not self.scene.properties.get('tissuePhysicsV2'): raise ValueError('POSITION_REQUIRES_V2_MECHANICS')
                apply_position(self.state, action['position'])
                events = [{'kind': 'patient_position', 'preset': self.state['patientPosition']['preset']}]
                for _ in range(max(1, math.ceil(duration/.1))):
                    if check is not None: _checked_operation(check)
                    events.extend(self._substep(key, 'inspect', np.asarray(slot['position']), np.asarray(slot['quaternion']), False, duration/max(1, math.ceil(duration/.1))))
            return {'status': 'accepted', 'reason': None, 'completedFraction': 1., 'events': events}
        if op == 'extract' and self.state.get('toolMechanicsRevision') in (1, 2):
            if slot['instrument'] != 'tumor_forceps' or points: raise ValueError('UNSUPPORTED_EXTRACTION')
            from .surgical_effects import extract_at_exit
            if key not in self.state['holds']: raise ValueError('GRASP_BEFORE_EXTRACTION')
            events = extract_at_exit(self, key)
            slot['active'] = False
            count=max(1,math.ceil(duration/.1))
            for _ in range(count):
                if check is not None: _checked_operation(check)
                events.extend(self._substep(key,'deactivate',np.asarray(slot['position']),np.asarray(slot['quaternion']),False,duration/count))
            return {'status': 'accepted', 'reason': None, 'completedFraction': 1., 'events': events}
        if op == 'exchange':
            if slot['inserted'] or slot['active'] or slot['grip'] or key in self.state['holds']:
                raise ValueError('WITHDRAW_BEFORE_EXCHANGE: deactivate, release, and withdraw first')
            if action.get('instrument') not in TOOLS: raise ValueError('UNKNOWN_INSTRUMENT')
            slot['instrument'] = action['instrument']; slot['parameters'] = {}
        slot['parameters'].update(action.get('parameters', {}))
        tool = TOOLS[slot['instrument']]
        params = self.parameters(slot)
        jaw_target = params['jawOpeningMm']
        articulate = self.state.get('toolMechanicsRevision') in (1, 2) and tool['geometry'] in ('forceps','scissors') and previous_jaw != jaw_target
        if articulate: slot['parameters']['jawOpeningMm'] = previous_jaw
        cusa_v3 = slot['instrument'] == 'cusa' and cusa_profile(self) is not None
        if cusa_v3:
            from .surgical_cusa_v3 import validate_parameters
            validate_parameters(params)
        progress_reason = ('BONE_CUTTING_IN_PROGRESS' if slot['instrument'] == 'cranial_drill' and self.scene.properties.get('tissuePhysicsV2')
            else 'TISSUE_ASPIRATION_IN_PROGRESS' if cusa_v3 else None)
        if op == 'hold' and tool['effect'] != 'retract' or op in ('grasp', 'extract') and tool['effect'] != 'grasp':
            raise ValueError('UNSUPPORTED_OPERATION_FOR_TOOL')
        if op in ('exchange', 'release', 'deactivate', 'activate', 'inspect', 'extract') and points:
            raise ValueError('This operation cannot also move the instrument')
        if op == 'deactivate': slot['active'] = False
        if op == 'activate': slot['active'] = True
        if op == 'release':
            retained = self.state['holds'].get(key)
            if retained and 'target' not in retained and self.state.get('toolMechanicsRevision') in (1, 2):
                self.state.setdefault('detachedFragments', []).append(copy.deepcopy(retained))
            self.state['holds'].pop(key, None); slot['grip'] = []; slot['active'] = False
            self.occupancy()
        if op == 'place':
            if slot['inserted'] or slot['active'] or slot['grip']: raise ValueError('WITHDRAW_BEFORE_PLACE')
            if len(points) != 1: raise ValueError('Placement requires one pose in free space')
            tip = points[0]['position']; axis = rotation(points[0]['quaternion']).apply([0, 0, 1])
            if self.touched(np.asarray(tip) - axis * 40, tip, max(params['radiusMm'], tool['shaftRadiusMm'])):
                raise ValueError('PLACE_OUTSIDE_MATERIAL: insert by a checked move or active stroke')
            slot.update(points[0])
            points = []
        if op == 'withdraw' and not points:
            axis = rotation(slot['quaternion']).apply([0, 0, 1])
            points = [{'position': (np.asarray(slot['position']) - 50 * axis).tolist(), 'quaternion': slot['quaternion']}]
        if op == 'withdraw':
            if slot['grip'] or key in self.state['holds']: raise ValueError('RELEASE_BEFORE_WITHDRAWAL')
            slot['active'] = False
        active = op in ('apply', 'hold', 'grasp', 'extract', 'activate') or slot['active'] and op == 'move' or op == 'inspect' and tool['effect'] == 'probe'
        poses = [{'position': slot['position'], 'quaternion': slot['quaternion']}, *points]
        if len(poses) == 1: poses.append(poses[0])
        segments = []
        for p, q in zip(poses, poses[1:]):
            distance = np.linalg.norm(np.subtract(q['position'], p['position']))
            angle = (rotation(q['quaternion']) * rotation(p['quaternion']).inv()).magnitude()
            n = max(1, math.ceil((distance + 40 * angle) / min(.4, float(self.scene.spacing.min()) / 3)),
                    math.ceil(duration / max(1, len(poses) - 1) / .1),
                    math.ceil(abs(jaw_target-previous_jaw)/min(.4,float(self.scene.spacing.min())/3)/max(1,len(poses)-1)) if articulate else 1)
            segments.append((p, q, n))
        total = sum(n for _, _, n in segments)
        if total > 20000: raise ValueError('PATH_TOO_LONG: split this stroke into shorter actions')
        if any(np.linalg.norm(np.subtract(q['position'], p['position'])) > 250 for p, q, _ in segments):
            raise ValueError('PATH_TOO_LONG')
        done = 0
        reason = None
        contact_position = None
        path_fraction = None
        completed_length = 0.
        for start, end, n in segments:
            interpolation = Slerp([0, 1], Rotation.from_quat([start['quaternion'], end['quaternion']]))
            for j in range(n):
                if check is not None: _checked_operation(check)
                t = (j + 1) / n
                p = ((1 - t) * finite(start['position'], 3) + t * finite(end['position'], 3))
                quat = interpolation(t).as_quat()
                old = _microstep_checkpoint(self.state)
                try:
                    if articulate: self.state['slots'][key]['parameters']['jawOpeningMm'] = jaw_target if done+1==total else previous_jaw+(jaw_target-previous_jaw)*(done+1)/total
                    delta_events = self._substep(key, op, p, quat, active, duration / total)
                except ValueError as error:
                    if check is not None: _checked_operation(check)
                    self.state = old
                    reason = str(error)
                    contact_position = p.tolist()
                    break
                events.extend(delta_events)
                done += 1
            if reason: break
            segment_length = float(np.linalg.norm(np.subtract(end['position'], start['position'])))
            # A stalled waypoint cannot be skipped in favor of a later leg:
            # that would change a drawn trajectory into an unintended shortcut.
            if len(segments) > 1 and active and progress_reason:
                remaining = float(np.linalg.norm(np.subtract(self.state['slots'][key]['position'], end['position'])))
                if remaining > 1e-5:
                    length = sum(float(np.linalg.norm(np.subtract(b['position'], a['position']))) for a, b, _ in segments)
                    path_fraction = (completed_length + max(0., segment_length - remaining)) / max(length, 1e-15)
                    reason = progress_reason
                    contact_position = self.state['slots'][key]['position'][:]
                    events.append({'kind': 'cusa_contact_dwell' if cusa_v3 else 'drill_contact_dwell', 'position': contact_position, 'remainingPathMm': length * (1. - path_fraction)})
                    break
            completed_length += segment_length
        if reason:
            events.append({'kind': 'contact_stop', 'reason': reason, 'position': contact_position})
        if done == 0: return {'status': 'rejected', 'reason': reason, 'completedFraction': 0., 'events': events}
        if op == 'withdraw' and not reason: slot = self.state['slots'][key]; slot['inserted'] = False
        if op in ('hold', 'grasp') and key in self.state['holds'] and 'target' not in self.state['holds'][key]:
            self.state['holds'][key]['origin'] = self.state['slots'][key]['position'][:]
        if active and not events:
            events.append({'kind': 'no_material_effect', 'reason': 'No compatible structure intersected by the working tip'})
        events.insert(0, {'kind': op, 'instrument': tool['name'], 'slot': key})
        fraction = done / total if path_fraction is None else path_fraction
        if not reason and points and active and progress_reason:
            remaining = float(np.linalg.norm(np.subtract(self.state['slots'][key]['position'], points[-1]['position'])))
            if remaining > 1e-5:
                length = sum(float(np.linalg.norm(np.subtract(b['position'], a['position']))) for a, b, _ in segments)
                fraction = max(0., 1. - remaining / max(length, remaining))
                reason = progress_reason
                contact_position = self.state['slots'][key]['position'][:]
                events.append({'kind': 'cusa_contact_dwell' if cusa_v3 else 'drill_contact_dwell', 'position': contact_position, 'remainingPathMm': remaining})
        return {'status': 'partial' if reason else 'accepted', 'reason': reason, 'completedFraction': fraction, 'contactPosition': contact_position, 'events': events}

    def _positions(self):
        """Current protected source membership in the shared occupancy raster."""
        if not self._protected_ids: return None
        mapping = {}
        for hold in [*self.state['holds'].values(), *self.state.get('detachedFragments', [])]:
            if 'target' in hold or not any(hold['offset']): continue
            for cell in self._protected_ids.intersection(hold['cells']):
                destination = np.asarray(np.unravel_index(cell, self.scene.shape)) + hold['offset']
                mapping[cell] = int(np.ravel_multi_index(destination, self.scene.shape))
        from .surgical_physics_v2 import raster
        mapping.update((cell, destination) for cell, destination in raster(self) if cell in self._protected_ids)
        if not any(cell != destination for cell, destination in mapping.items()): return None
        return self.scene.points([mapping.get(int(cell), int(cell)) for cell in self.protection.cells])

    def _guard_sweep(self, key, p, q, positions):
        for other_key, other in self.state['slots'].items():
            end = {'position': p, 'quaternion': q} if other_key == key else other
            if len(self.protection.sweep_hits(TOOLS[other['instrument']], self.parameters(other), other, end, positions=positions)):
                raise ValueError('PROTECTED_STRUCTURE: instrument or shaft enters the declared clearance')

    def _protected_exposure(self):
        fields = {'thermal': self.state['thermal'],
            'boneTemperature': self.state.get('drillingV2', {}).get('temperatureC', {}),
            'boneDose': self.state.get('drillingV2', {}).get('thermalDoseCSeconds', {})}
        return {(name, cell): value for name, values in fields.items() for cell, value in values.items() if int(cell) in self._protected_ids}

    def _substep(self, key, op, p, q, active, dt):
        if not self._protected_ids:
            return self._physical_substep(key, op, p, q, active, dt)
        self._guard_sweep(key, p, q, self._positions())
        previous_thermal = self._protected_exposure()
        events = self._physical_substep(key, op, p, q, active, dt)
        impacted = set(self.state['removed'])
        for field in ('cutEdges', 'transectedEdges'):
            impacted.update(cell for edge in self.state.get(field, []) for cell in edge)
        if impacted & self._protected_ids:
            raise ValueError('PROTECTED_STRUCTURE: indirect removal or cutting is blocked')
        limit = self.guard_config.critical_max_displacement_mm + 1e-9
        for cell, displacement in self.state.get('deformation', {}).items():
            if int(cell) in self._protected_ids and np.linalg.norm(displacement) > limit:
                raise ValueError('PROTECTED_STRUCTURE: indirect deformation exceeds the declared limit')
        for hold in self.state['holds'].values():
            if 'target' not in hold and self._protected_ids.intersection(hold['cells']) and np.linalg.norm(self.scene.affine[:3, :3] @ np.asarray(hold['offset'])) > limit:
                raise ValueError('PROTECTED_STRUCTURE: grasp displaces protected fragment membership')
        ambient = self.scene.properties.get('tissuePhysicsV2', {}).get('bone', {}).get('ambientTemperatureC', 37.)
        for (name, cell), value in self._protected_exposure().items():
            if value > previous_thermal.get((name, cell), ambient if name == 'boneTemperature' else 0.):
                raise ValueError('PROTECTED_STRUCTURE: indirect thermal exposure is blocked')
        for face in self.state.get('vesselsV2', {}).get('faces', {}).values():
            if face['damage'] > 0 and self._protected_ids.intersection(face['cells']):
                raise ValueError('PROTECTED_STRUCTURE: indirect vessel injury is blocked')
        # A coupled patch can enter a parked instrument after the tool sweep.
        positions = self._positions()
        if positions is not None: self._guard_sweep(key, p, q, positions)
        return events

    def _physical_substep(self, key, op, p, q, active, dt):
        from .surgical_motion import envelopes, swept_envelopes, pair_conflict
        slot = self.state['slots'][key]; tool = TOOLS[slot['instrument']]; params = self.parameters(slot)
        axis = rotation(q).apply([0, 0, 1]); previous = finite(slot['position'], 3)
        v2 = bool(self.scene.properties.get('tissuePhysicsV2'))
        if v2:
            from . import surgical_drilling_v2 as drill
            drill.begin_step(self)
            if active and slot['instrument'] == 'cranial_drill':
                p, _ = drill.contact_limited_pose(self, previous, p, params['radiusMm'])
        cusa_v3 = slot['instrument'] == 'cusa' and cusa_profile(self) is not None
        if cusa_v3 and active:
            from .surgical_cusa_v3 import contact_limited_pose
            p, _ = contact_limited_pose(self, previous, p, params['radiusMm'])
        tip_cells = self.touched(previous, p, params['radiusMm'])
        end_pose = {'position': p, 'quaternion': q}
        swept = swept_envelopes(tool, params, slot, end_pose)
        shaft_cells = set()
        revised_contact = self.state.get('toolMechanicsRevision') in (1, 2) and tool['effect'] not in ('remove', 'suction', 'irrigate', 'probe')
        working_cells = set() if revised_contact else set(tip_cells)
        for a, b, radius, part in swept:
            if part == 'shaft': shaft_cells.update(self.touched(a, b, radius))
            else: working_cells.update(self.touched(a, b, radius))
        if revised_contact: tip_cells = sorted(working_cells)
        held = set(self.state['holds'].get(key, {}).get('cells', []))
        if set(shaft_cells) - held: raise ValueError('SHAFT_COLLISION')
        if cusa_v3 and active and self.state.get('toolMechanicsRevision') != 2 and working_cells - held - set(self.touched(p, p, params['radiusMm'])):
            raise ValueError('RETAINED_TISSUE_COLLISION: CUSA body requires a cleared corridor')
        for other_key, other in self.state['slots'].items():
            if other_key == key: continue
            other_tool = TOOLS[other['instrument']]
            if pair_conflict(swept, envelopes(other_tool, self.parameters(other), other['position'], other['quaternion'])):
                raise ValueError('TOOL_COLLISION:' + other_key)
        if tool['effect'] not in ('suction', 'irrigate') and working_cells - held and op not in ('withdraw', 'release', 'deactivate', 'inspect', 'exchange'):
            if any(self.by_label[int(self.scene.labels.ravel()[i])]['material'] not in tool['materials'] for i in working_cells - held):
                raise ValueError('UNSUPPORTED_MATERIAL')
            if not active and key not in self.state['holds']: raise ValueError('RETAINED_TISSUE_COLLISION')
        effect_cells = sorted(working_cells) if self.state.get('toolMechanicsRevision') in (1, 2) and tool['effect'] not in ('remove','suction','irrigate','probe') else tip_cells
        eligible = [i for i in effect_cells if self.by_label[int(self.scene.labels.ravel()[i])]['material'] in tool['materials']]
        field_action = active and tool['effect'] in ('suction', 'irrigate')
        if field_action and np.linalg.norm(p - previous) > 1e-8:
            physical = set(self.touched(p, p, tool['shaftRadiusMm']))
            if physical - set(self.touched(previous, previous, tool['shaftRadiusMm'])):
                raise ValueError('RETAINED_TISSUE_COLLISION')
        if op == 'withdraw':
            previous_contacts = set(self.touched(previous, previous, params['radiusMm']))
            current_contacts = set(tip_cells)
            if revised_contact:
                previous_contacts = set()
                current_contacts = set()
                for a,b,radius,part in envelopes(tool,params,previous,slot['quaternion']):
                    if part != 'shaft': previous_contacts.update(self.touched(a,b,radius))
                for a,b,radius,part in envelopes(tool,params,p,q):
                    if part != 'shaft': current_contacts.update(self.touched(a,b,radius))
            if current_contacts - previous_contacts: raise ValueError('WITHDRAWAL_PATH_BLOCKED')
        if tip_cells and not field_action and (not active and key not in self.state['holds'] or any(self.by_label[int(self.scene.labels.ravel()[i])]['material'] not in tool['materials'] for i in tip_cells)):
            if op not in ('withdraw', 'release', 'deactivate', 'inspect', 'exchange'):
                raise ValueError('UNSUPPORTED_MATERIAL' if active else 'RETAINED_TISSUE_COLLISION')
        events = []
        if active:
            events = self._effect(key, op, eligible, p, q, dt)
        if key in self.state['holds']:
            self.occupancy()
        if key in self.state['holds'] and op == 'move':
            from .surgical_effects import validate_hold_motion
            validate_hold_motion(self, key)
            hold = self.state['holds'][key]
            if v2 and 'target' in hold:
                from .surgical_deformation_v2 import set_target
                set_target(self, key, p)
            else:
                if self.state.get('toolMechanicsRevision') in (1, 2) and (rotation(q)*rotation(slot['quaternion']).inv()).magnitude() > 1e-8:
                    raise ValueError('FRAGMENT_ROTATION_UNSUPPORTED: translate the source-bound fragment')
                delta = self.scene.inverse[:3, :3] @ (p - finite(hold['origin'], 3))
                hold['offset'] = (np.rint(delta).astype(int) + np.asarray(hold.get('baseOffset', [0,0,0]))).tolist()
            self.occupancy()
        # A stationary active instrument shares the same clock and exposure.
        for other_key, other in self.state['slots'].items():
            if other_key == key or not other['active']: continue
            other_tool = TOOLS[other['instrument']]
            other_cells = self.touched(other['position'], other['position'], self.parameters(other)['radiusMm'])
            if self.state.get('toolMechanicsRevision') in (1, 2) and other_tool['effect'] not in ('remove','suction','irrigate','probe'):
                contacts = set()
                for a,b,radius,part in envelopes(other_tool,self.parameters(other),other['position'],other['quaternion']):
                    if part != 'shaft': contacts.update(self.touched(a,b,radius))
                other_cells = sorted(contacts)
            other_eligible = [i for i in other_cells if self.by_label[int(self.scene.labels.ravel()[i])]['material'] in other_tool['materials']]
            if other_tool['materials'] and len(other_eligible) != len(other_cells):
                raise ValueError('OTHER_ACTIVE_TOOL_UNSUPPORTED_MATERIAL:' + other_key)
            events.extend(self._effect(other_key, 'apply', other_eligible, np.asarray(other['position']), np.asarray(other['quaternion']), dt))
            if v2:
                from .surgical_vessels_v2 import contact
                contact(self, other_cells, dt, self.parameters(other)['forceN'])
        vessels = [i for i in working_cells if self.by_label[int(self.scene.labels.ravel()[i])]['material'] == 'vessel']
        self.state['vesselEncounters'] = sorted(set(self.state['vesselEncounters']) | set(vessels))
        for i in working_cells:
            material = self.by_label[int(self.scene.labels.ravel()[i])]['material']
            self.state['materialExposure'][material] = self.state['materialExposure'].get(material, 0.) + dt
        contact_cells = set(working_cells) if tool['effect'] in ('elevate', 'dissect') else set()
        for hold in self.state['holds'].values(): contact_cells.update(hold['cells'])
        contact_cells.difference_update(self.state['removed'])
        for cell in contact_cells:
            k = str(cell); self.state['contact'][k] = self.state['contact'].get(k, 0.) + dt
        slot.update(position=p.tolist(), quaternion=q.tolist())
        if op not in ('place', 'exchange', 'release', 'inspect', 'deactivate'): slot['inserted'] = True
        self.state['elapsedSeconds'] += dt
        self.state['travelMm'] += float(np.linalg.norm(p - previous))
        from .surgical_effects import tick_fields
        if v2:
            from . import surgical_deformation_v2 as brain
            from . import surgical_vessels_v2 as vessel
            # Force is an explicit proxy, never inferred from a segmentation.
            vessel.contact(self, vessels, dt, params['forceN'] if active else 0.)
            brain.tick(self, dt)
            self.occupancy()
            # Coupled tissue moves after the ordinary swept-tool check.
            # Recheck every physical shaft against the updated occupancy so a
            # patch cannot enter a stationary instrument on the last substep.
            # Match the existing per-hand exemption for its own held cells.
            if self.state.get('deformation'):
                for shaft_key, shaft_slot in self.state['slots'].items():
                    own_held = set(self.state['holds'].get(shaft_key, {}).get('cells', ()))
                    shaft_tool = TOOLS[shaft_slot['instrument']]
                    for a, b, radius, part in envelopes(shaft_tool, self.parameters(shaft_slot),
                            shaft_slot['position'], shaft_slot['quaternion']):
                        if part == 'shaft' and set(self.touched(a, b, radius))-own_held:
                            raise ValueError('POST_DEFORMATION_SHAFT_COLLISION:'+shaft_key)
            drill.end_step(self, dt)
            vessel.tick(self, dt)
        tick_fields(self, dt)
        return events

    def _effect(self, key, op, cells, p, q, dt):
        slot = self.state['slots'][key]; tool = TOOLS[slot['instrument']]
        if tool['effect'] == 'remove':
            if slot['instrument'] == 'cusa' and cusa_profile(self) is not None:
                from .surgical_cusa_v3 import apply
                contacts = self.touched(p, p, self.parameters(slot)['radiusMm'])
                return apply(self, contacts, dt, self.parameters(slot), position=p)
            if slot['instrument'] == 'cranial_drill' and self.scene.properties.get('tissuePhysicsV2'):
                from .surgical_drilling_v2 import apply
                events, _, _ = apply(self, cells, dt, self.parameters(slot), cool=False)
                return events
            new = set(cells) - set(self.state['removed'])
            self.state['removed'] = sorted(set(self.state['removed']) | new)
            if slot['instrument'] == 'cranial_drill':
                for i in new: self.state['debris'][str(i)] = self.state['debris'].get(str(i), 0.) + self.scene.voxel_mm3
                self.state['debrisMm3'] = sum(self.state['debris'].values())
                # Preserve v1 replay interpretation for earlier scenes. The
                # explicit material profile enables updated bone connectivity.
                if new and self.scene.properties.get('tissuePhysics'):
                    self.state['boneTopologyDirty'] = True
            return [{'kind': 'removal', 'cells': len(new), 'volumeMm3': len(new) * self.scene.voxel_mm3}] if new else []
        if tool['effect'] == 'retract':
            if not cells and key not in self.state['holds']: return []
            if key not in self.state['holds']:
                if self.scene.properties.get('tissuePhysicsV2'):
                    from .surgical_deformation_v2 import begin_hold
                    begin_hold(self, key, cells, p)
                    return [{'kind': 'coupled_hold_created', 'cells': len(cells)}]
                self.state['holds'][key] = {'cells': cells, 'origin': p.tolist(), 'offset': [0, 0, 0]}
                return [{'kind': 'hold_created', 'cells': len(cells)}]
            return []
        from .surgical_effects import apply_effect
        return apply_effect(self, key, op, cells, p, q, dt)

    def metrics(self):
        from .surgical_physics import physics_metrics
        from .surgical_physics_v2 import metrics as v2_metrics
        by_label = {}
        for i in self.state['removed']:
            name = self.by_label[int(self.scene.labels.ravel()[i])]['name']
            by_label[name] = by_label.get(name, 0.) + self.scene.voxel_mm3
        target = sum(self.scene.voxel_mm3 for i in self.state['removed'] if self.by_label[int(self.scene.labels.ravel()[i])].get('target'))
        from .surgical_cusa_v3 import metrics as cusa_metrics
        return {**physics_metrics(self), **v2_metrics(self), **(cusa_metrics(self) if 'cusaV3' in self.state else {}), 'removedByLabelMm3': by_label, 'targetRemovedMm3': target,
                'nonTargetRemovedMm3': len(self.state['removed']) * self.scene.voxel_mm3 - target,
                'elapsedSeconds': self.state['elapsedSeconds'], 'travelMm': self.state['travelMm'],
                'contactCellSeconds': sum(self.state['contact'].values()), 'debrisMm3': self.state['debrisMm3'],
                'fluidAddedMm3': self.state['fluidAddedMm3'], 'fluidRemovedMm3': self.state['fluidRemovedMm3'],
                'debrisRemovedMm3': self.state.get('debrisRemovedMm3', 0.), 'debrisEscapedMm3': self.state.get('debrisEscapedMm3', 0.),
                'fluidPresentMm3': sum(self.state['fluid'].values()) + sum(s.get('reservoirMm3', 0.) for s in self.state['sources']),
                'fluidEscapedMm3': self.state['fluidEscapedMm3'],
                'thermalExposure': sum(self.state['thermal'].values()),
                'vesselEncounters': len(self.state['vesselEncounters']),
                'exposureByMaterialCellSeconds': copy.deepcopy(self.state['materialExposure']),
                'unknownStimulationSites': sum(o.get('result') == 'unknown' for o in self.state['observations']),
                'collisions': sum(h['status'] == 'partial' and h.get('reason') != 'TISSUE_ASPIRATION_IN_PROGRESS' and (not self.scene.properties.get('tissuePhysicsV2') or h.get('reason') != 'BONE_CUTTING_IN_PROGRESS') for h in self.history),
                'unknownMaterialCells': int(sum(np.count_nonzero(self.scene.labels == r['id']) for r in self.scene.regions if r['material'] == 'unknown'))}

    def observe(self, *, _state_id=None):
        from .surgical_position import describe_position
        slots = copy.deepcopy(self.state['slots'])
        if self.state.get('toolMechanicsRevision') in (1, 2):
            from .surgical_motion import envelopes
            for key, slot in slots.items():
                slot['contactGeometry'] = [{'a': a.tolist(), 'b': b.tolist(), 'radiusMm': radius, 'part': part}
                    for a, b, radius, part in envelopes(TOOLS[slot['instrument']], self.parameters(slot), [0.,0.,0.], [0.,0.,0.,1.])]
        from .surgical_physics import physics_descriptor
        if self.scene.properties.get('tissuePhysicsV2'):
            from .surgical_physics_v2 import descriptor as physics_descriptor
        physics = physics_descriptor(self)
        if 'cusaV3' in self.state:
            from .surgical_cusa_v3 import descriptor
            physics['cusaV3'] = descriptor(self)
        return {'schema': SCHEMA, 'stateId': self.state_id if _state_id is None else _state_id, 'slots': slots,
                **({'toolMechanicsRevision': self.state['toolMechanicsRevision']} if 'toolMechanicsRevision' in self.state else {}),
                **({'patientPosition': copy.deepcopy(self.state['patientPosition']), 'positionEffects': describe_position(self.state['patientPosition'])} if 'patientPosition' in self.state else {}),
                'protection': self.guard_report(),
                'physics': physics,
                'metrics': self.metrics(), 'observations': copy.deepcopy(self.state['observations']),
                'unknowns': ['unmapped anatomy', 'uncalibrated mechanics', 'clinical outcomes unknown']}

    def view(self, *, _state_id=None, include_history=True, _omit_fields=()):
        omitted = frozenset(_omit_fields)
        if omitted - self._view_omit_fields:
            raise ValueError('Unsupported display field omission')
        identity = self.state_id if _state_id is None else _state_id
        moved = []
        removed = set(self.state['removed'])
        for hold in [*self.state['holds'].values(), *self.state.get('detachedFragments', [])]:
            if not any(hold['offset']): continue
            for i in hold['cells']:
                if i in removed: continue
                target = np.asarray(np.unravel_index(i, self.scene.shape)) + hold['offset']
                moved.append([i, int(np.ravel_multi_index(target, self.scene.shape))])
        if 'deformation' in self.state:
            from .surgical_physics_v2 import raster
            moved.extend(raster(self))
        return {**self.observe(_state_id=identity), 'scene': self.scene.descriptor(), 'catalog': catalog(),
                'geometryRevision': identity, 'removed': self.state['removed'][:], 'moved': moved, 'cuts': copy.deepcopy(self.state['cuts']),
                'holds': copy.deepcopy(self.state['holds']),
                'contact': list(map(int, self.state['contact'])), 'thermal': copy.deepcopy(self.state['thermal']),
                'fluid': copy.deepcopy(self.state['fluid']), 'history': copy.deepcopy(self.history) if include_history else [],
                **{name: _copy_engine_data(self.state.get(name)) for name in ('cutEdges', 'separatedAttachments',
                    'transectedEdges', 'flapComponents', 'debris', 'sources', 'meshCutEdges', 'meshSeams',
                    'vesselConnections', 'vesselWounds', 'deformation', 'deformationModel', 'drillingV2', 'vesselsV2') if name not in omitted},
                **({'cusaV3': copy.deepcopy(self.state['cusaV3'])} if 'cusaV3' in self.state else {})}

    def snapshot(self):
        return {'schema': SCHEMA, 'sceneId': self.scene.identity, 'actions': copy.deepcopy(self.actions), 'stateId': self.state_id,
                **({'protectionContract': copy.deepcopy(self._protection_contract)} if self._protection_contract else {}),
                'checkpoint': {'position': len(self.actions), 'stateId': self.state_id, 'state': copy.deepcopy(self.state)}}

    @classmethod
    def restore(cls, scene, record, *, check=None, use_cache=False, force_verify=False, verified_cache=None):
        if check is not None: check()
        if record['schema'] != SCHEMA or record['sceneId'] != scene.identity: raise ValueError('Saved scene or engine version differs')
        cache = None
        if use_cache:
            from .surgical_verified_cache import VerifiedSurgicalCache
            scene.require_cache_source()
            cache = verified_cache if verified_cache is not None else VerifiedSurgicalCache()
            if not force_verify:
                cached = cache.restore(cls, scene, record, check=check)
                if cached is not None:
                    return cached
        options = {}
        if 'protectionContract' in record:
            from .surgical_rl_protection import restore_protection_contract
            protection, config = restore_protection_contract(scene, record['protectionContract'])
            options = {'protection': protection, 'guard_config': config}
        result = cls(scene, **options)
        writer = cache._begin_replay(result, record) if cache is not None else None
        try:
            if writer is not None:
                writer.capture(result, check)
            for action in record['actions']:
                if check is not None: check()
                outcome = result.step(action) if check is None else result.step(action, check=check)
                if outcome['status'] == 'rejected': raise ValueError('Saved action no longer replays')
                if check is not None: check()
                if writer is not None:
                    writer.capture(result, check)
            if result.state_id != record['stateId']: raise ValueError('Replay state identity differs')
            checkpoint = record.get('checkpoint')
            if checkpoint is not None and (checkpoint.get('position') != len(result.actions) or checkpoint.get('stateId') != result.state_id or semantic_digest(checkpoint.get('state')) != semantic_digest(result.state)):
                raise ValueError('Checkpoint differs from verified replay')
            if check is not None: check()
            if writer is not None:
                writer.publish(result, record, check)
        finally:
            if writer is not None:
                writer.abort()
        return result

    def branch(self, position, *, check=None):
        if type(position) is not int or not 0 <= position <= len(self.actions): raise ValueError('Invalid history position')
        result = self.fresh()
        for action in self.actions[:position]:
            if check is not None: check()
            if check is None: result.step(action)
            else: result.step(action, check=check)
        if check is not None: check()
        return result


# Definition-time originals for the drill's local pure-query fast path. Looking
# up both sides on SurgicalEngine later would incorrectly admit a class patch.
from . import geometry as _drill_contact_geometry
from . import surgical_geometry_cache as _drill_contact_cache
from .surgical_occupancy import MaterialOccupancy as _DrillContactOccupancy

_DRILL_CONTACT_QUERY_ORIGINALS = (
    SurgicalEngine, SurgicalEngine.__getattribute__,
    tuple((name, method, method.__code__, method.__defaults__, method.__kwdefaults__)
          for name in ('touched', 'cells', 'occupancy')
          for method in (getattr(SurgicalEngine, name),)),
    tuple((cls, cls.__getattribute__, tuple((name, vars(cls)[name]) for name in names)) for cls, names in (
        (_drill_contact_cache.SurgicalCellCoverCache, ('query', '_frame_identity', '_candidate_grid')),
        (GeometryScene, ('world_to_voxel', 'voxel_to_world', 'shape')),
        (_DrillContactOccupancy, ('material_ids', '__getitem__')))),
    tuple((namespace, name, function, function.__code__, function.__defaults__,
           None if function.__kwdefaults__ is None else tuple(function.__kwdefaults__.items()))
          for namespace, names in (
              (globals(), ('capsule_voxel_indices',)),
              (vars(_drill_contact_geometry), ('capsule_voxel_indices', '_vector', '_immutable',
                  'point_segment_distances', '_segment_cell_distances', '_capsule_candidate_grid')),
              (vars(_drill_contact_cache), ('immutable_array',)),
              (vars(_drill_contact_cache.SurgicalCellCoverCache), ('query', '_frame_identity', '_candidate_grid')),
              (vars(GeometryScene), ('world_to_voxel', 'voxel_to_world')),
              ({'shape': GeometryScene.shape.fget}, ('shape',)),
              (vars(_DrillContactOccupancy), ('material_ids', '__getitem__')))
          for name in names for function in (namespace[name],)))
