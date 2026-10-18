"""Deterministic instrument transitions, independent of UI and wall clock.

Source cells retain their identity through removal and reversible, grid-snapped
patch translations. Collision and display both consume this SAME occupancy.
This deliberately coarse deformation model is not FEM or measured mechanics.
"""
from __future__ import annotations

import copy
import math
import itertools
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .core import semantic_digest
from .geometry import GeometryScene, capsule_voxel_indices, segment_segment_distance
from .surgical_tools import TOOLS, catalog

SCHEMA = 'resectionlab.surgical-toolbox.v1'
MAX_STEPS = 2048


def finite(value, size=None):
    a = np.asarray(value, float)
    if not np.isfinite(a).all() or size is not None and a.shape != (size,):
        raise ValueError('Finite coordinates with the declared dimension are required')
    return a


def rotation(q):
    q = finite(q, 4)
    if abs(np.linalg.norm(q) - 1) > 1e-5: raise ValueError('Orientation must be a unit xyzw quaternion')
    return Rotation.from_quat(q)


class SurgicalEngine:
    def __init__(self, scene):
        self.scene = scene
        self.grid = GeometryScene(np.zeros(scene.shape, bool), scene.affine)
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
    def state_id(self): return semantic_digest({'scene': self.scene.identity, 'state': self.state})

    def clone(self):
        other = copy.copy(self)
        other.state = copy.deepcopy(self.state)
        other.actions = copy.deepcopy(self.actions)
        other.history = copy.deepcopy(self.history)
        return other

    def occupancy(self):
        key = semantic_digest({'removed': self.state['removed'], 'holds': self.state['holds'], **({'deformation': self.state['deformation']} if 'deformation' in self.state else {})})
        if self._occupancy_cache is not None and self._occupancy_cache[0] == key:
            return self._occupancy_cache[1]
        from .surgical_occupancy import MaterialOccupancy
        removed = set(self.state['removed'])
        cleared = set(removed)
        moves = []
        held_ids = set()
        for hold in self.state['holds'].values():
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
        indices = capsule_voxel_indices(self.grid, np.asarray(a), np.asarray(b), radius)
        if not len(indices): return np.empty(0, np.int64)
        return np.ravel_multi_index(indices.T, self.scene.shape)

    def touched(self, a, b, radius):
        cells = self.cells(a, b, radius)
        return self.occupancy().material_ids(cells)

    def parameters(self, slot):
        tool = TOOLS[slot['instrument']]
        extra = {'rpm': 40000., 'forceN': 1., 'duty': 1.} if self.scene.properties.get('tissuePhysicsV2') else {}
        return {**extra, 'radiusMm': tool['radiusMm'], 'depthMm': 2., 'flowMm3PerSecond': 20., 'power': 1., 'jawOpeningMm': 2., **slot['parameters']}

    def preview(self, action):
        candidate = self.clone()
        result = candidate.step(action)
        return {**result, 'preview': True, 'view': candidate.view()}

    def reset(self, scene=None):
        self.__init__(self.scene if scene is None else scene)
        return self.observe()

    def replay(self, actions):
        result = type(self)(self.scene)
        for action in actions:
            if result.step(action)['status'] == 'rejected':
                raise ValueError('Replay contains an inadmissible action')
        return result

    def _validate(self, action):
        if not isinstance(action, dict) or set(action) - {'slot', 'operation', 'instrument', 'pose', 'path', 'parameters', 'duration'}:
            raise ValueError('Unsupported action fields')
        if action.get('slot') not in self.state['slots']: raise ValueError('Choose left or right instrument slot')
        if action.get('operation') not in {'place', 'move', 'activate', 'deactivate', 'apply', 'hold', 'grasp', 'release', 'exchange', 'withdraw', 'extract', 'inspect'}:
            raise ValueError('Unsupported instrument operation')
        duration = float(action.get('duration', 1.))
        if not math.isfinite(duration) or not 0 < duration <= 30: raise ValueError('Duration must be in (0, 30] seconds')
        if len(self.actions) >= MAX_STEPS: raise ValueError('Action limit reached; save and start a new episode')
        points = action.get('path', [action['pose']] if 'pose' in action else [])
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

    def step(self, action):
        before = self.state_id
        before_time = self.state['elapsedSeconds']
        before_removed = set(self.state['removed'])
        candidate = self.clone()
        try:
            duration, points = candidate._validate(action)
            result = candidate._execute(copy.deepcopy(action), duration, points)
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            return {'status': 'rejected', 'reason': str(error), 'reasonCode': str(error).split(':')[0] if str(error).split(':')[0].isupper() else 'INVALID_ACTION', 'completedFraction': 0., 'elapsedSeconds': 0., 'stateId': before, 'events': [], 'deltas': {}, 'metrics': self.metrics()}
        result['reasonCode'] = result['reason'].split(':')[0] if result.get('reason') else None
        result.setdefault('elapsedSeconds', candidate.state['elapsedSeconds'] - before_time if result['status'] != 'rejected' else 0.)
        if result['status'] != 'rejected':
            from .surgical_effects import finalize_action
            finalize_action(candidate)
            result['elapsedSeconds'] = candidate.state['elapsedSeconds'] - before_time
            result['deltas'] = {'removedCells': sorted(set(candidate.state['removed']) - before_removed),
                                'geometryRevision': candidate.state_id}
            candidate.actions.append(copy.deepcopy(action))
            candidate.history.append({'action': copy.deepcopy(action), 'stateBefore': before,
                                      'stateAfter': candidate.state_id, **result})
            self.__dict__.update(candidate.__dict__)
        return {**result, 'stateId': self.state_id, 'metrics': self.metrics()}

    def _execute(self, action, duration, points):
        key, op = action['slot'], action['operation']
        slot = self.state['slots'][key]
        events = []
        if op == 'exchange':
            if slot['inserted'] or slot['active'] or slot['grip'] or key in self.state['holds']:
                raise ValueError('WITHDRAW_BEFORE_EXCHANGE: deactivate, release, and withdraw first')
            if action.get('instrument') not in TOOLS: raise ValueError('UNKNOWN_INSTRUMENT')
            slot['instrument'] = action['instrument']; slot['parameters'] = {}
        slot['parameters'].update(action.get('parameters', {}))
        tool = TOOLS[slot['instrument']]
        params = self.parameters(slot)
        if op == 'hold' and tool['effect'] != 'retract' or op in ('grasp', 'extract') and tool['effect'] != 'grasp':
            raise ValueError('UNSUPPORTED_OPERATION_FOR_TOOL')
        if op in ('exchange', 'release', 'deactivate', 'activate', 'inspect', 'extract') and points:
            raise ValueError('This operation cannot also move the instrument')
        if op == 'deactivate': slot['active'] = False
        if op == 'activate': slot['active'] = True
        if op == 'release':
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
                    math.ceil(duration / max(1, len(poses) - 1) / .1))
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
                t = (j + 1) / n
                p = ((1 - t) * finite(start['position'], 3) + t * finite(end['position'], 3))
                quat = interpolation(t).as_quat()
                # Bone component summaries are replaced only by finalize_action,
                # never mutated by a microstep. Share them within this action's
                # rollback checkpoints rather than copying thousands of source
                # components for every fraction of a millimeter.
                components = self.state.get('flapComponents')
                old = copy.deepcopy(self.state, {id(components): components} if components is not None else None)
                try:
                    delta_events = self._substep(key, op, p, quat, active, duration / total)
                except ValueError as error:
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
            if len(segments) > 1 and active and slot['instrument'] == 'cranial_drill' and self.scene.properties.get('tissuePhysicsV2'):
                remaining = float(np.linalg.norm(np.subtract(self.state['slots'][key]['position'], end['position'])))
                if remaining > 1e-5:
                    length = sum(float(np.linalg.norm(np.subtract(b['position'], a['position']))) for a, b, _ in segments)
                    path_fraction = (completed_length + max(0., segment_length - remaining)) / max(length, 1e-15)
                    reason = 'BONE_CUTTING_IN_PROGRESS'
                    contact_position = self.state['slots'][key]['position'][:]
                    events.append({'kind': 'drill_contact_dwell', 'position': contact_position, 'remainingPathMm': length * (1. - path_fraction)})
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
        if not reason and points and tool['name'] == 'Cranial drill' and self.scene.properties.get('tissuePhysicsV2'):
            remaining = float(np.linalg.norm(np.subtract(self.state['slots'][key]['position'], points[-1]['position'])))
            if remaining > 1e-5:
                length = sum(float(np.linalg.norm(np.subtract(b['position'], a['position']))) for a, b, _ in segments)
                fraction = max(0., 1. - remaining / max(length, remaining))
                reason = 'BONE_CUTTING_IN_PROGRESS'
                contact_position = self.state['slots'][key]['position'][:]
                events.append({'kind': 'drill_contact_dwell', 'position': contact_position, 'remainingPathMm': remaining})
        return {'status': 'partial' if reason else 'accepted', 'reason': reason, 'completedFraction': fraction, 'contactPosition': contact_position, 'events': events}

    def _substep(self, key, op, p, q, active, dt):
        from .surgical_motion import envelopes, swept_envelopes, pair_conflict
        slot = self.state['slots'][key]; tool = TOOLS[slot['instrument']]; params = self.parameters(slot)
        axis = rotation(q).apply([0, 0, 1]); previous = finite(slot['position'], 3)
        v2 = bool(self.scene.properties.get('tissuePhysicsV2'))
        if v2:
            from . import surgical_drilling_v2 as drill
            drill.begin_step(self)
            if active and slot['instrument'] == 'cranial_drill':
                p, _ = drill.contact_limited_pose(self, previous, p, params['radiusMm'])
        tip_cells = self.touched(previous, p, params['radiusMm'])
        end_pose = {'position': p, 'quaternion': q}
        swept = swept_envelopes(tool, params, slot, end_pose)
        shaft_cells = set()
        working_cells = set(tip_cells)
        for a, b, radius, part in swept:
            if part == 'shaft': shaft_cells.update(self.touched(a, b, radius))
            else: working_cells.update(self.touched(a, b, radius))
        held = set(self.state['holds'].get(key, {}).get('cells', []))
        if set(shaft_cells) - held: raise ValueError('SHAFT_COLLISION')
        for other_key, other in self.state['slots'].items():
            if other_key == key: continue
            other_tool = TOOLS[other['instrument']]
            if pair_conflict(swept, envelopes(other_tool, self.parameters(other), other['position'], other['quaternion'])):
                raise ValueError('TOOL_COLLISION:' + other_key)
        if tool['effect'] not in ('suction', 'irrigate') and working_cells - held and op not in ('withdraw', 'release', 'deactivate', 'inspect', 'exchange'):
            if any(self.by_label[int(self.scene.labels.ravel()[i])]['material'] not in tool['materials'] for i in working_cells - held):
                raise ValueError('UNSUPPORTED_MATERIAL')
            if not active and key not in self.state['holds']: raise ValueError('RETAINED_TISSUE_COLLISION')
        eligible = [i for i in tip_cells if self.by_label[int(self.scene.labels.ravel()[i])]['material'] in tool['materials']]
        field_action = active and tool['effect'] in ('suction', 'irrigate')
        if field_action and np.linalg.norm(p - previous) > 1e-8:
            physical = set(self.touched(p, p, tool['shaftRadiusMm']))
            if physical - set(self.touched(previous, previous, tool['shaftRadiusMm'])):
                raise ValueError('RETAINED_TISSUE_COLLISION')
        if op == 'withdraw':
            previous_contacts = set(self.touched(previous, previous, params['radiusMm']))
            if set(tip_cells) - previous_contacts: raise ValueError('WITHDRAWAL_PATH_BLOCKED')
        if tip_cells and not field_action and (not active and key not in self.state['holds'] or len(eligible) != len(tip_cells)):
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
                delta = self.scene.inverse[:3, :3] @ (p - finite(hold['origin'], 3))
                hold['offset'] = np.rint(delta).astype(int).tolist()
            self.occupancy()
        # A stationary active instrument shares the same clock and exposure.
        for other_key, other in self.state['slots'].items():
            if other_key == key or not other['active']: continue
            other_tool = TOOLS[other['instrument']]
            other_cells = self.touched(other['position'], other['position'], self.parameters(other)['radiusMm'])
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
            drill.end_step(self, dt)
            vessel.tick(self, dt)
        tick_fields(self, dt)
        return events

    def _effect(self, key, op, cells, p, q, dt):
        slot = self.state['slots'][key]; tool = TOOLS[slot['instrument']]
        if tool['effect'] == 'remove':
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
        return {**physics_metrics(self), **v2_metrics(self), 'removedByLabelMm3': by_label, 'targetRemovedMm3': target,
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
                'collisions': sum(h['status'] == 'partial' and (not self.scene.properties.get('tissuePhysicsV2') or h.get('reason') != 'BONE_CUTTING_IN_PROGRESS') for h in self.history),
                'unknownMaterialCells': int(sum(np.count_nonzero(self.scene.labels == r['id']) for r in self.scene.regions if r['material'] == 'unknown'))}

    def observe(self):
        from .surgical_physics import physics_descriptor
        if self.scene.properties.get('tissuePhysicsV2'):
            from .surgical_physics_v2 import descriptor as physics_descriptor
        return {'schema': SCHEMA, 'stateId': self.state_id, 'slots': copy.deepcopy(self.state['slots']),
                'physics': physics_descriptor(self),
                'metrics': self.metrics(), 'observations': copy.deepcopy(self.state['observations']),
                'unknowns': ['unmapped anatomy', 'uncalibrated mechanics', 'clinical outcomes unknown']}

    def view(self):
        moved = []
        removed = set(self.state['removed'])
        for hold in self.state['holds'].values():
            if not any(hold['offset']): continue
            for i in hold['cells']:
                if i in removed: continue
                target = np.asarray(np.unravel_index(i, self.scene.shape)) + hold['offset']
                moved.append([i, int(np.ravel_multi_index(target, self.scene.shape))])
        if 'deformation' in self.state:
            from .surgical_physics_v2 import raster
            moved.extend(raster(self))
        return {**self.observe(), 'scene': self.scene.descriptor(), 'catalog': catalog(),
                'geometryRevision': self.state_id, 'removed': self.state['removed'][:], 'moved': moved, 'cuts': copy.deepcopy(self.state['cuts']),
                'contact': list(map(int, self.state['contact'])), 'thermal': copy.deepcopy(self.state['thermal']),
                'fluid': copy.deepcopy(self.state['fluid']), 'history': copy.deepcopy(self.history),
                **{name: copy.deepcopy(self.state.get(name)) for name in ('cutEdges', 'separatedAttachments',
                    'transectedEdges', 'flapComponents', 'debris', 'sources', 'meshCutEdges', 'meshSeams',
                    'vesselConnections', 'vesselWounds', 'deformation', 'deformationModel', 'drillingV2', 'vesselsV2')}}

    def snapshot(self):
        return {'schema': SCHEMA, 'sceneId': self.scene.identity, 'actions': copy.deepcopy(self.actions), 'stateId': self.state_id,
                'checkpoint': {'position': len(self.actions), 'stateId': self.state_id, 'state': copy.deepcopy(self.state)}}

    @classmethod
    def restore(cls, scene, record):
        if record['schema'] != SCHEMA or record['sceneId'] != scene.identity: raise ValueError('Saved scene or engine version differs')
        result = cls(scene)
        for action in record['actions']:
            if result.step(action)['status'] == 'rejected': raise ValueError('Saved action no longer replays')
        if result.state_id != record['stateId']: raise ValueError('Replay state identity differs')
        checkpoint = record.get('checkpoint')
        if checkpoint is not None and (checkpoint.get('position') != len(result.actions) or checkpoint.get('stateId') != result.state_id or semantic_digest(checkpoint.get('state')) != semantic_digest(result.state)):
            raise ValueError('Checkpoint differs from verified replay')
        return result

    def branch(self, position):
        if type(position) is not int or not 0 <= position <= len(self.actions): raise ValueError('Invalid history position')
        result = type(self)(self.scene)
        for action in self.actions[:position]: result.step(action)
        return result
