"""Authoritative opt-in cranial entry policy bound to original source geometry.

The oriented rectangular prism is a declared engineering access envelope. It
never creates anatomy, removes obstacles or certifies an anatomical approach.
"""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

from .surgical_motion import envelopes, swept_envelopes
from .surgical_tools import TOOLS

BONE_PREPARATION_TOOLS = frozenset({'cranial_drill', 'craniotome',
    'periosteal_elevator', 'bone_flap_holder', 'dissector'})


def tool_is_exterior(engine, key):
    """Whole capsule extents must clear the declared original exit face."""
    contract = engine._cranial_access
    slot = engine.state['slots'][key]
    axis, side, plane = contract.exit.axis, contract.exit.side, contract.exit.plane_voxel_boundary
    inverse = engine.scene.inverse
    margin = np.linalg.norm(inverse[axis, :3])
    for a, b, radius, _ in envelopes(TOOLS[slot['instrument']], engine.parameters(slot), slot['position'], slot['quaternion']):
        coordinates = np.asarray([a, b])@inverse[axis, :3]+inverse[axis, 3]
        if side == -1 and coordinates.max()+radius*margin >= plane-1e-9:
            return False
        if side == 1 and coordinates.min()-radius*margin <= plane+1e-9:
            return False
    return True


def require_exterior_withdrawal(engine, key):
    if not tool_is_exterior(engine, key):
        raise ValueError('CRANIAL_ACCESS_WITHDRAWAL_NOT_AT_EXTERIOR_EXIT')


def _inside_entry(contract, segments):
    entry = contract.entry
    center, normal, u = map(np.asarray, (entry.center_ras_mm, entry.normal_ras_unit, entry.u_ras_unit))
    v = np.cross(normal, u)
    half = np.asarray(entry.half_extents_mm)
    enters = False
    for a, b, radius, _ in segments:
        a, b = np.asarray(a)-center, np.asarray(b)-center
        sa, sb = float(a@normal), float(b@normal)
        if max(sa, sb)+radius <= 1e-9:
            continue
        enters = True
        if max(sa, sb)+radius > entry.maximum_inward_mm+1e-9:
            raise ValueError('CRANIAL_ACCESS_DEPTH_OUTSIDE')
        # Restrict only the capsule portion touching the original inward
        # halfspace; a proximal extracranial shaft may be outside the rectangle.
        if sa < -radius and sb > sa:
            a = a+(-radius-sa)/(sb-sa)*(b-a)
        elif sb < -radius and sa > sb:
            b = b+(-radius-sb)/(sa-sb)*(a-b)
        projections = np.asarray([[a@u, a@v], [b@u, b@v]])
        if np.any(np.abs(projections)+radius > half+1e-9):
            raise ValueError('CRANIAL_ACCESS_WINDOW_OUTSIDE')
    return enters


def enforce(engine, key, op, position, quaternion, working_cells=()):
    """Check both complete instruments at every native substep, before effects."""
    if getattr(engine, '_cranial_stages', None) is not None:
        from .surgical_cranial_stages import enforce as enforce_stages
        if enforce_stages(engine, key, op, position, quaternion, working_cells):
            return
    contract = getattr(engine, '_cranial_access', None)
    if contract is None:
        return
    from .surgical_milling import is_milling, require_coverage
    from .surgical_bone_transport import aperture_evacuated
    from .surgical_task_domain import _outside, _posed
    workspace = SimpleNamespace(minimum=contract.exit.minimum_voxel_boundary,
                                maximum=contract.exit.maximum_voxel_boundary)
    milling = is_milling(contract)
    preparation_tools = frozenset({'cranial_drill', 'frazier_suction'}) if milling else BONE_PREPARATION_TOOLS
    if milling and (engine.state.get('toolMechanicsRevision') != 2 or engine.state.get('physicalClockRevision') != 1):
        raise ValueError('CRANIAL_ACCESS_MILLING_STOCK_AND_CLOCK_REQUIRED')
    # Milling remains a bone-only operation after clearance too. Keeping its
    # inner-boundary guard does not require rescanning replay history at every
    # physical microstep; completion is evaluated at accepted checkpoints.
    evacuated = False if milling else aperture_evacuated(engine, contract)
    for hand, slot in engine.state['slots'].items():
        if not _posed(engine, hand) and (hand != key or op in ('exchange', 'inspect', 'deactivate')):
            continue
        tool = TOOLS[slot['instrument']]
        params = engine.parameters(slot)
        end = {'position': position, 'quaternion': quaternion} if hand == key else slot
        segments = (swept_envelopes(tool, params, slot, end) if hand == key
                    else envelopes(tool, params, slot['position'], slot['quaternion']))
        if _outside(engine.scene, workspace, segments):
            raise ValueError('CRANIAL_ACCESS_OUTSIDE_REVIEWED_WORKSPACE')
        if not _inside_entry(contract, segments):
            continue
        require_coverage(contract)
        if milling and slot['instrument'] not in preparation_tools:
            # Bone-only clearance is not evidence of a dural opening or a
            # qualified intracranial treatment task.
            raise ValueError('CRANIAL_ACCESS_MILLING_TREATMENT_UNQUALIFIED')
        axis = Rotation.from_quat(end['quaternion']).apply([0., 0., 1.])
        if float(axis@np.asarray(contract.entry.normal_ras_unit)) < np.cos(np.deg2rad(contract.entry.maximum_angle_degrees))-1e-9:
            raise ValueError('CRANIAL_ACCESS_ATTITUDE_OUTSIDE')
        if not evacuated and slot['instrument'] not in preparation_tools:
            raise ValueError('CRANIAL_ACCESS_REQUIRES_EVACUATED_FLAP')
        if not evacuated and slot['instrument'] in preparation_tools:
            # Bone preparation has no permission to travel beyond the original
            # inner table. This also blocks a shortcut through a crop/model gap.
            bone = engine.scene.points(contract.window_cells)
            normal = np.asarray(contract.entry.normal_ras_unit)
            support = .5*float(np.abs(engine.scene.affine[:3, :3].T@normal).sum())
            inner = float(((bone-np.asarray(contract.entry.center_ras_mm))@
                           normal).max())+support
            for a, b, radius, _ in segments:
                depth = max(float((np.asarray(a)-contract.entry.center_ras_mm)@contract.entry.normal_ras_unit),
                            float((np.asarray(b)-contract.entry.center_ras_mm)@contract.entry.normal_ras_unit))+radius
                if depth > inner+1e-9:
                    raise ValueError('CRANIAL_ACCESS_BEYOND_INNER_TABLE_BEFORE_EVACUATION')
    # All source bone effects, including a stationary other active hand, remain
    # in the frozen kerf. A holder may contact the intended retained flap only.
    for hand, slot in engine.state['slots'].items():
        if hand != key and not slot['active']:
            continue
        name = slot['instrument']
        if milling and name in preparation_tools:
            params = engine.parameters(slot)
            if params['radiusMm'] != TOOLS[name]['radiusMm'] or params['depthMm'] != 2.:
                raise ValueError('CRANIAL_ACCESS_MILLING_STOCK_GEOMETRY_REQUIRED')
        if milling and name == 'frazier_suction':
            cells = set(working_cells) if hand == key else set()
            if hand != key:
                for a, b, radius, part in envelopes(TOOLS[name], engine.parameters(slot), slot['position'], slot['quaternion']):
                    if part != 'shaft':
                        cells.update(engine.touched(a, b, radius))
            if cells:
                raise ValueError('CRANIAL_ACCESS_MILLING_SUCTION_REQUIRES_CLEARED_VOID')
        if name not in ('cranial_drill', 'craniotome', 'bone_flap_holder'):
            continue
        if hand == key:
            cells = set(working_cells)
        else:
            cells = set()
            for a, b, radius, part in envelopes(TOOLS[name], engine.parameters(slot), slot['position'], slot['quaternion']):
                if part != 'shaft':
                    cells.update(engine.touched(a, b, radius))
        allowed = set(contract.flap_cells if name == 'bone_flap_holder' else contract.kerf_cells)
        if cells-allowed:
            raise ValueError('CRANIAL_ACCESS_BONE_WORK_OUTSIDE_DECLARED_SCOPE')
