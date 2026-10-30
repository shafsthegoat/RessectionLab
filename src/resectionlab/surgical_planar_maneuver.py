"""One opt-in canonical, fixed-topology planar-blade engineering maneuver.

The native action transaction owns all chronological JSON. The disposable body
cache calls the preceding contact kernel directly; it contains no episode state.
Historical scenes never enter this path. No removal, fitting or planning occurs.
"""
from __future__ import annotations

import copy
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation

from .core import array_digest, semantic_digest
from . import surgical_continuum as continuum
from . import surgical_planar_contact as contact

PROPERTY = 'planarBladeManeuverV1'
SCHEMA = 'resectionlab.planar-blade-maneuver.v1'
VIEW_SCHEMA = 'resectionlab.planar-blade-maneuver-view.v1'
PHASES = ('approach', 'indent', 'hold', 'withdraw')
STAGES = ('ready', 'approached', 'indented', 'held', 'withdrawn')
DURATIONS = (.1, .2, .2, .2)
OFFSETS = (0., .02, .02, -.1)
DT = .025
THICKNESS = .2
SHAFT_RADIUS = .3
TOPOLOGY_FIELDS = ('removed', 'cuts', 'separated', 'cutEdges', 'separatedAttachments',
                   'transectedEdges', 'meshCutEdges', 'meshSeams', 'detachedFragments')


def geometry_hash(scene):
    # CaseData's semantic hash includes this opt-in declaration. Hash the
    # actual physical arrays/metadata, excluding only that derived outer hash,
    # to avoid a self-referential caseHash -> declaration -> caseHash cycle.
    source = {k: v for k, v in scene.source.items() if k not in ('caseHash', 'environmentSchema')}
    return semantic_digest({'labels': array_digest(scene.labels), 'affine': scene.affine.tolist(),
                            'regions': scene.regions, 'source': source})


def declaration(scene, cells, contact_faces, support_faces, blade, binding=None, *,
                integration=contact.ADAPTIVE_INTEGRATION):
    """Declare geometry/assumptions explicitly, never infer patient exposure."""
    return {'schema': SCHEMA, 'sourceGeometryHash': geometry_hash(scene),
        'materialProfileHash': semantic_digest(scene.properties[continuum.PROPERTY]),
        'cells': list(cells), 'contactFaces': [list(f) for f in contact_faces],
        'supportFaces': [list(f) for f in support_faces],
        'blade': {'originMm': blade.origin_mm.tolist(), 'axisU': blade.axis_u.tolist(),
            'axisV': blade.axis_v.tolist(), 'inwardNormal': blade.inward_normal.tolist(),
            'sizeMm': blade.size_mm.tolist(), 'thicknessMm': THICKNESS, 'shaftRadiusMm': SHAFT_RADIUS},
        'schedule': {'phases': list(PHASES), 'durationsSeconds': list(DURATIONS),
            'endOffsetsMm': list(OFFSETS), 'initialOffsetMm': -.1, 'fixedStepSeconds': DT,
            **({'integration': integration} if integration != contact.LEGACY_INTEGRATION else {})},
        'binding': copy.deepcopy(binding or {'kind': 'generated-engineering-fixture', 'patientData': False}),
        'domain': 'One fixed rectangular reference face, fixed topology/support, normal-only motion; other hand parked. No tissue removal, friction, current edge migration, multi-tool coupling or RL.',
        'priorNumericalQualification': {'cycleWorkSensitivityPassed': False,
            'uniformRelativeChange': .10084383288364987, 'partialRelativeChange': .11413225425522984,
            'declaredLimit': .05, 'clinicalValidation': False, 'patientPredictionValidation': False}}


def validate(scene):
    p = scene.properties.get(PROPERTY)
    if p is None:
        return
    if not isinstance(p, dict) or set(p) != set(declaration(scene, [], [], [], contact.PlanarBlade.create(
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1], [4, 4]))):
        raise ValueError('PLANAR_MANEUVER_COMPLETE_DECLARATION_REQUIRED')
    if p['schema'] != SCHEMA or p['sourceGeometryHash'] != geometry_hash(scene):
        raise ValueError('PLANAR_MANEUVER_SOURCE_BINDING')
    if set(scene.properties) - {'tissuePhysicsV2', continuum.PROPERTY, PROPERTY, 'continuumLinearSolveV1'}:
        raise ValueError('PLANAR_MANEUVER_UNSUPPORTED_COUPLED_MODELS')
    b = p['blade']
    if not isinstance(b, dict) or set(b) != {'originMm', 'axisU', 'axisV', 'inwardNormal', 'sizeMm', 'thicknessMm', 'shaftRadiusMm'}:
        raise ValueError('PLANAR_MANEUVER_COMPLETE_BLADE_REQUIRED')
    blade = contact.PlanarBlade.create(b['originMm'], b['axisU'], b['axisV'], b['inwardNormal'], b['sizeMm'])
    integration = p.get('schedule', {}).get('integration', contact.LEGACY_INTEGRATION)
    if integration not in (contact.LEGACY_INTEGRATION, contact.ADAPTIVE_INTEGRATION):
        raise ValueError('PLANAR_MANEUVER_INTEGRATION_METHOD')
    expected = declaration(scene, p['cells'], p['contactFaces'], p['supportFaces'], blade, p['binding'], integration=integration)
    if semantic_digest(p) != semantic_digest(expected):
        raise ValueError('PLANAR_MANEUVER_FIXED_DOMAIN_REQUIRED')
    if not isinstance(p['binding'], dict) or not p['binding'].get('kind'):
        raise ValueError('PLANAR_MANEUVER_EXPLICIT_PROVENANCE_REQUIRED')


def engineering_scene():
    """Generated frozen uniform numerical coupon; contains no patient anatomy."""
    from .surgical_scene import SurgicalScene
    from .surgical_physics_v2 import default_profile
    labels = np.zeros((4, 4, 4), np.uint16); labels[1:3, 1:3, 1:3] = 1
    affine = np.diag([2., 2., 2., 1.]); affine[:3, 3] = -1
    props = {'tissuePhysicsV2': default_profile(), continuum.PROPERTY: continuum.research_profile({'1': {
        'shearPa': 1000., 'poissonRatio': 0., 'relaxationShearPa': [500.],
        'source': 'Generated engineering control, not patient measurements'}}, relaxation_seconds=[.5])}
    base = SurgicalScene(labels, affine, [{'id': 1, 'name': 'Generated Q1 specimen', 'material': 'soft_tissue'}],
        {'kind': 'generated-engineering-fixture', 'description': '4 mm numerical cube; no patient anatomy'}, props)
    cells = list(map(int, np.flatnonzero(labels)))
    top = [(c, 2, 1) for c in cells if np.unravel_index(c, labels.shape)[2] == 2]
    bottom = [(c, 2, -1) for c in cells if np.unravel_index(c, labels.shape)[2] == 1]
    blade = contact.PlanarBlade.create([2, 2, 4], [1, 0, 0], [0, 1, 0], [0, 0, -1], [4, 4])
    props[PROPERTY] = declaration(base, cells, top, bottom, blade)
    return SurgicalScene(base.labels, base.affine, base.regions, base.source, props)


def model(engine):
    engine.scene.require_cache_source(); validate(engine.scene)
    p = engine.scene.properties[PROPERTY]; identifier = semantic_digest(p)
    cached = getattr(engine, '_planar_maneuver_model', None)
    if cached is not None and cached[0] is engine.scene and cached[1] == identifier:
        return cached[2]
    b = p['blade']
    blade = contact.PlanarBlade.create(b['originMm'], b['axisU'], b['axisV'], b['inwardNormal'], b['sizeMm'])
    kernel = contact.build(engine.scene, p['cells'], contact_faces=p['contactFaces'], support_faces=p['supportFaces'],
        blade=blade, integration=p['schedule'].get('integration', contact.LEGACY_INTEGRATION))
    engine._planar_maneuver_model = (engine.scene, identifier, kernel)
    return kernel


def _initial(kernel):
    zero = kernel.initial_state()
    return contact.ContactState(zero.model_id, zero.displacement_mm, zero.memory_mm, -.1, 0)


def _kernel_state(kernel, raw):
    if raw is None:
        return _initial(kernel)
    if not isinstance(raw, dict) or raw.get('modelId') != kernel.identity:
        raise ValueError('PLANAR_MANEUVER_KERNEL_STATE_BINDING')
    return contact.ContactState(raw['modelId'], np.asarray(raw['displacementMm']), np.asarray(raw['memoryMm']).reshape(
        len(kernel.times), len(kernel.free)), raw['offsetMm'], raw['steps'])


def _topology(state):
    return semantic_digest({k: state.get(k) for k in TOPOLOGY_FIELDS})


def _admit(engine):
    if engine.protection is not None:
        raise ValueError('PLANAR_MANEUVER_PROTECTION_GEOMETRY_UNSUPPORTED')
    if any(engine.state.get(k) for k in TOPOLOGY_FIELDS) or engine.state.get('patientPosition') or engine.state.get('cusaV3'):
        raise ValueError('PLANAR_MANEUVER_FIXED_INTACT_TOPOLOGY_REQUIRED')
    other = engine.state['slots']['right']
    if other['inserted'] or other['active'] or other['grip'] or engine.state['holds']:
        raise ValueError('PLANAR_MANEUVER_OTHER_HAND_MUST_BE_PARKED')
    state = engine.state.get(PROPERTY)
    if state is None:
        if any(np.any(v) for v in engine.state.get('deformation', {}).values()) or engine.state.get(continuum.PROPERTY):
            raise ValueError('PLANAR_MANEUVER_EXISTING_HISTORY_UNSUPPORTED')
        if any(s['inserted'] or s['active'] or s['grip'] for s in engine.state['slots'].values()):
            raise ValueError('PLANAR_MANEUVER_PARK_BOTH_BEFORE_APPROACH')
    else:
        if state.get('declarationId') != semantic_digest(engine.scene.properties[PROPERTY]):
            raise ValueError('PLANAR_MANEUVER_DECLARATION_STATE_BINDING')
        if state['topologyBinding'] != _topology(engine.state) or state['parkedRight'] != other:
            raise ValueError('PLANAR_MANEUVER_TOPOLOGY_OR_PARKED_HAND_CHANGED')


def _pose(kernel, offset):
    b = kernel.blade
    frame = np.column_stack((b.axis_u, b.inward_normal, b.axis_v))
    if np.linalg.det(frame) < 1 - 1e-10:
        raise ValueError('PLANAR_MANEUVER_RIGHT_HANDED_TOOL_FRAME_REQUIRED')
    return b.origin_mm + offset * b.inward_normal + b.size_mm[1] / 2 * b.axis_v, Rotation.from_matrix(frame).as_quat()


def rigid_geometry(kernel, touching):
    width, depth = map(float, kernel.blade.size_mm)
    return {'sizeMm': [width, THICKNESS, depth], 'centerLocalMm': [0., -THICKNESS / 2, -depth / 2],
        'shaft': {'a': [0., -THICKNESS - SHAFT_RADIUS, -40.],
                  'b': [0., -THICKNESS - SHAFT_RADIUS, -depth], 'radiusMm': SHAFT_RADIUS}, 'inContact': touching}


def _collision(engine, kernel, u, offset, before_u=None, before_offset=None):
    """Conservative source/Q1 bounds for the exact rectangular pad and shaft.

    Q1 points lie in their corner convex hull. Positive box overlap that these
    bounds cannot resolve refuses; no center-raster exemption invents clearance.
    Axis-aligned source cells in this declared frame have exact interval bounds.
    """
    blade = kernel.blade; width, depth = blade.size_mm
    frame = np.column_stack((blade.axis_u, blade.inward_normal, blade.axis_v))
    active = {c: i for i, c in enumerate(kernel.mesh.cells)}
    local_u = u.reshape(-1, 3)
    origin = blade.origin_mm + offset * blade.inward_normal
    lo = np.array([-width / 2, -THICKNESS, -depth / 2]); hi = np.array([width / 2, 0., depth / 2])
    shaft_lo = np.array([0., -THICKNESS - SHAFT_RADIUS, depth / 2 - 40.])
    shaft_hi = np.array([0., -THICKNESS - SHAFT_RADIUS, -depth / 2])
    occupied = np.flatnonzero(engine.scene.labels)
    for start in range(0, len(occupied), 4096):
        check = getattr(engine, '_transition_check', None)
        if check is not None: check()
        ids = occupied[start:start + 4096]
        ijk = np.column_stack(np.unravel_index(ids, engine.scene.shape))
        points = (ijk[:, None, :] + continuum.CORNERS - .5) @ engine.scene.affine[:3, :3].T + engine.scene.affine[:3, 3]
        for i, cell in enumerate(ids):
            row = active.get(int(cell))
            if row is not None:
                points[i] = kernel.mesh.nodes[kernel.mesh.element_nodes[row]] + local_u[kernel.mesh.element_nodes[row]]
        local = (points - origin) @ frame
        low, high = local.min(axis=1), local.max(axis=1)
        if before_u is not None:
            # The fixed frame and linear interpolation of coupled corners/pose
            # give a multi-affine space-time map. Its endpoint corner hull
            # conservatively bounds the entire sweep, including between ticks.
            old_points = (ijk[:, None, :] + continuum.CORNERS - .5) @ engine.scene.affine[:3, :3].T + engine.scene.affine[:3, 3]
            old_field = np.asarray(before_u).reshape(-1, 3)
            for i, cell in enumerate(ids):
                row = active.get(int(cell))
                if row is not None:
                    old_points[i] = kernel.mesh.nodes[kernel.mesh.element_nodes[row]] + old_field[kernel.mesh.element_nodes[row]]
            old_local = (old_points - blade.origin_mm - before_offset * blade.inward_normal) @ frame
            low = np.minimum(low, old_local.min(axis=1)); high = np.maximum(high, old_local.max(axis=1))
        overlap = np.minimum(high, hi) - np.maximum(low, lo)
        if np.any(np.all(overlap > contact.GAP_MM, axis=1)):
            raise ValueError('PLANAR_MANEUVER_PAD_COLLISION_OR_UNRESOLVED_Q1_GEOMETRY')
        distance = np.maximum(np.maximum(low - shaft_hi, shaft_lo - high), 0.)
        if np.any(np.linalg.norm(distance, axis=1) < SHAFT_RADIUS - contact.GAP_MM):
            raise ValueError('PLANAR_MANEUVER_SHAFT_COLLISION')


def execute(engine, action, *, check=None):
    if (set(action) != {'slot', 'operation', 'maneuverPhase', 'duration'}
            or action['operation'] != 'bladeManeuver' or action['slot'] != 'left'
            or action['maneuverPhase'] not in PHASES):
        raise ValueError('PLANAR_MANEUVER_ONLY: use the declared four-phase maneuver')
    phase = action['maneuverPhase']; index = PHASES.index(phase)
    if type(action['duration']) not in (int, float) or action['duration'] != DURATIONS[index]:
        raise ValueError('PLANAR_MANEUVER_FIXED_DURATION_REQUIRED')
    _admit(engine); kernel = model(engine)
    state = engine.state.get(PROPERTY)
    if (state['phase'] if state else 'ready') != STAGES[index]:
        raise ValueError('PLANAR_MANEUVER_PHASE_ORDER: use the next declared phase')
    if state is None:
        initial = _initial(kernel)
        engine.state[PROPERTY] = state = {'schema': SCHEMA, 'declarationId': semantic_digest(engine.scene.properties[PROPERTY]),
            'phase': 'ready', 'activePhase': None, 'topologyBinding': _topology(engine.state),
            'parkedRight': copy.deepcopy(engine.state['slots']['right']), 'kernel': None, 'receipt': None,
            'inContact': False, 'initialEnergyNmm': 0., 'actuatorWorkNmm': 0., 'viscousDissipationNmm': 0.,
            'numericalDissipationNmm': 0., 'contactActivationDissipationNmm': 0., 'contactEvents': []}
        position, quat = _pose(kernel, initial.offset_mm)
        _collision(engine, kernel, initial.displacement_mm, initial.offset_mm)
        engine.state['slots']['left'].update(instrument='brain_retractor', position=position.tolist(), quaternion=quat.tolist(),
            parameters={'radiusMm': float(kernel.blade.size_mm[0] / 2), 'depthMm': float(kernel.blade.size_mm[1]),
                        'planarBladeContactRevision': 1}, active=False, inserted=True, grip=[],
            planarBladeGeometry=rigid_geometry(kernel, False))
    old = _kernel_state(kernel, state['kernel']); state['activePhase'] = phase
    events = []; count = round(DURATIONS[index] / DT)
    for tick in range(count):
        if check is not None: check()
        offset = old.offset_mm + (OFFSETS[index] - old.offset_mm) * (tick + 1) / count
        position, quaternion = _pose(kernel, offset)
        events.extend(engine._substep('left', 'bladeManeuver', position, quaternion, True, DT))
    state['phase'] = STAGES[index + 1]; state['activePhase'] = None
    if phase == 'withdraw':
        engine.state['slots']['left'].update(inserted=False, active=False)
    return {'status': 'accepted', 'reason': None, 'completedFraction': 1., 'events': events}


def advance(engine, key, position, quaternion, dt):
    _admit(engine); kernel = model(engine); state = engine.state[PROPERTY]
    if key != 'left' or dt != DT or state['activePhase'] not in PHASES:
        raise ValueError('PLANAR_MANEUVER_INVALID_INCREMENT')
    offset = float((position - kernel.blade.origin_mm - kernel.blade.size_mm[1] / 2 * kernel.blade.axis_v) @ kernel.blade.inward_normal)
    expected_p, expected_q = _pose(kernel, offset)
    if np.linalg.norm(position - expected_p) > 1e-10 or abs(float(np.asarray(quaternion) @ expected_q)) < 1 - 1e-10:
        raise ValueError('PLANAR_MANEUVER_UNSUPPORTED_POSE')
    check = getattr(engine, '_transition_check', None) or (lambda: None)
    previous = _kernel_state(kernel, state['kernel'])
    solved = kernel.step(previous, offset_mm=offset, dt_s=dt, check=check)
    result = solved['state']; receipt = {k: v for k, v in solved['receipt'].items() if k != 'wallSeconds'}
    if receipt['maxLoadedFootprintMigrationMm'] > contact.GAP_MM:
        raise ValueError('PLANAR_MANEUVER_CURRENT_FOOTPRINT_MIGRATION_UNSUPPORTED')
    _collision(engine, kernel, result.displacement_mm, offset, previous.displacement_mm, previous.offset_mm)
    _surface(kernel, {'kernel': {'modelId': result.model_id, 'displacementMm': result.displacement_mm,
        'memoryMm': result.memory_mm, 'offsetMm': result.offset_mm, 'steps': result.steps}})
    settings = engine.scene.properties['tissuePhysicsV2']['brain']
    displacement = result.displacement_mm.reshape(-1, 3)
    maximum = float(np.linalg.norm(displacement, axis=1).max(initial=0.))
    if maximum > settings['maxDisplacementMm'] or receipt['normalReactionN'] > settings['maxForceN']:
        raise ValueError('PLANAR_MANEUVER_DISPLACEMENT_OR_FORCE_LIMIT')
    touching = receipt['contactAreaMm2'] > 0.
    events = []
    if touching != state['inContact']:
        kind = 'blade_contact_acquired' if touching else 'blade_contact_released'
        event = {'kind': kind, 'step': result.steps, 'seconds': engine.state['elapsedSeconds'] + dt}
        state['contactEvents'].append(event); events.append(event)
    state['inContact'] = touching
    state['kernel'] = {'modelId': result.model_id, 'displacementMm': result.displacement_mm.tolist(),
        'memoryMm': result.memory_mm.tolist(), 'offsetMm': result.offset_mm, 'steps': result.steps}
    state['receipt'] = {**receipt, 'maxDisplacementMm': maximum}
    for field in ('actuatorWorkNmm', 'viscousDissipationNmm', 'numericalDissipationNmm', 'contactActivationDissipationNmm'):
        state[field] += receipt[field]
    balance = state['actuatorWorkNmm'] - receipt['storedEnergyNmm'] - sum(state[k] for k in (
        'viscousDissipationNmm', 'numericalDissipationNmm', 'contactActivationDissipationNmm'))
    if abs(balance) > contact.BALANCE_NMM:
        raise ValueError('PLANAR_MANEUVER_CUMULATIVE_ENERGY_BALANCE')
    state['cumulativeEnergyBalanceNmm'] = balance
    slot = engine.state['slots']['left']; engine.state['travelMm'] += float(np.linalg.norm(position - slot['position']))
    slot.update(position=position.tolist(), quaternion=quaternion.tolist(), active=touching,
                planarBladeGeometry=rigid_geometry(kernel, touching))
    engine.state['elapsedSeconds'] += dt
    engine.state[continuum.PROPERTY] = {'schema': continuum.SCHEMA,
        'cornerDisplacementMm': continuum._pack(kernel.mesh, result.displacement_mm),
        'memoryDisplacementMm': [continuum._pack(kernel.mesh, z) for z in result.memory_mm],
        'activeElements': len(kernel.mesh.cells), 'activeDofs': len(kernel.free),
        'dissipatedEnergyNmm': state['viscousDissipationNmm'], 'numericalDissipationNmm': state['numericalDissipationNmm'],
        'removedStoredEnergyNmm': 0., 'removedElementCount': 0, 'exposureBySourceLabel': {},
        'maxFreeResidualN': receipt['maxEquilibriumResidualN'], 'maxEnergyBalanceResidualNmm': abs(balance)}
    centers = displacement[kernel.mesh.element_nodes].mean(axis=1)
    engine.state['deformation'] = {str(c): centers[i].tolist() for i, c in enumerate(kernel.mesh.cells)}
    engine.state['deformationModel'].update(active=list(kernel.mesh.cells), anchors=[], velocities={},
        remainderSeconds=0., elasticEnergyNmm=receipt['storedEnergyNmm'], activeNodes=len(kernel.mesh.cells), model=continuum.SCHEMA)
    return events


def _surface(kernel, state):
    u = _kernel_state(kernel, state['kernel'] if state else None).displacement_mm.reshape(-1, 3)
    triangles = []; active = set(kernel.mesh.cells)
    for row, cell in enumerate(kernel.mesh.cells):
        ijk = np.array(np.unravel_index(cell, kernel.scene.shape))
        for axis, sign in product(range(3), (-1, 1)):
            neighbor = ijk.copy(); neighbor[axis] += sign
            if np.all((neighbor >= 0) & (neighbor < kernel.scene.shape)) and int(np.ravel_multi_index(neighbor, kernel.scene.shape)) in active:
                continue
            others = [a for a in range(3) if a != axis]
            face = []
            for j, k in ((0, 0), (1, 0), (1, 1), (0, 1)):
                corner = np.zeros(3, int); corner[axis] = (sign + 1) // 2; corner[others] = [j, k]
                local = int(np.flatnonzero(np.all(continuum.CORNERS == corner, axis=1))[0])
                face.append(int(kernel.mesh.element_nodes[row, local]))
            triangles.extend(((face[0], face[1], face[2]), (face[0], face[2], face[3])))
            corners = (kernel.mesh.nodes + u)[face]
            normal = np.cross(corners[1] - corners[0], corners[3] - corners[0])
            length = float(np.linalg.norm(normal))
            if length <= 1e-12 or abs(float((corners[2] - corners[0]) @ normal)) > contact.GAP_MM * length:
                raise ValueError('PLANAR_MANEUVER_NONPLANAR_RENDERED_Q1_FACE_UNSUPPORTED')
    return {'positionsMm': (kernel.mesh.nodes + u).tolist(), 'triangles': [list(t) for t in triangles],
            'sourceIds': list(kernel.mesh.cells)}


def view(engine):
    if PROPERTY not in engine.scene.properties:
        return None
    p = engine.scene.properties[PROPERTY]; state = engine.state.get(PROPERTY)
    phase = state['phase'] if state else 'ready'; receipt = state['receipt'] if state else None
    refusal = None; kernel = None
    try:
        _admit(engine); kernel = model(engine)
        ks = _kernel_state(kernel, state['kernel'] if state else None)
        _collision(engine, kernel, ks.displacement_mm, ks.offset_mm)
    except ValueError as error:
        refusal = str(error)
    next_actions = []
    if refusal is None and phase != 'withdrawn' and not (state and state['activePhase']):
        i = STAGES.index(phase)
        next_actions = [{'label': ('Approach blade', 'Indent blade', 'Hold blade', 'Withdraw blade')[i],
            'action': {'slot': 'left', 'operation': 'bladeManeuver', 'maneuverPhase': PHASES[i], 'duration': DURATIONS[i]}}]
    return {'schema': VIEW_SCHEMA, 'declarationId': semantic_digest(p), 'phase': phase,
        'activePhase': state['activePhase'] if state else None, 'available': refusal is None, 'refusal': refusal,
        'inContact': state['inContact'] if state else False,
        'normalReactionN': receipt['normalReactionN'] if receipt else (0. if kernel else None),
        'contactAreaMm2': receipt['contactAreaMm2'] if receipt else (0. if kernel else None),
        'maxDisplacementMm': receipt['maxDisplacementMm'] if receipt else (0. if kernel else None),
        'offsetMm': state['kernel']['offsetMm'] if state and state['kernel'] else (-.1 if kernel else None),
        'nextActions': next_actions, 'contactSurface': _surface(kernel, state) if kernel else None,
        **({'opticalFrame': {'targetMm': kernel.blade.origin_mm.tolist(),
            'outward': (-kernel.blade.inward_normal).tolist()}} if kernel else {}),
        'details': {'declaration': copy.deepcopy(p), 'materials': copy.deepcopy(engine.scene.properties[continuum.PROPERTY]),
            'lastReceipt': copy.deepcopy(receipt), 'cumulative': {k: state[k] for k in (
                'actuatorWorkNmm', 'viscousDissipationNmm', 'numericalDissipationNmm',
                'contactActivationDissipationNmm', 'cumulativeEnergyBalanceNmm') if state and k in state},
            'limitations': ['Fixed reference footprint; current loaded edge migration refuses.',
                'Prior full-cycle work sensitivity failed 5% gates (10.08% / 11.41%).',
                'Generated/research material and supports; no physiological or patient-prediction validation.']}}
