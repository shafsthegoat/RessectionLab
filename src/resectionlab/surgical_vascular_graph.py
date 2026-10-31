"""Opt-in source-bound RC network; numerical research, not patient perfusion.

P=V/C uses pressure-supporting storage with ZERO unstressed volume, relative to
an explicitly zero field pressure. Signed edge/boundary fluxes permit reversal.
An intact occluded edge has no wound; cutting disconnects it and opens two ports.
Geometry binds actions/outlets, never estimates lumen, conductance or territory.
"""
from __future__ import annotations

import copy
import math
import numpy as np
from scipy.integrate import solve_ivp

PROPERTY = 'vascularGraphV1'
SCHEMA = 'resectionlab.vascular-graph.v1'
MODEL_VERSION = 'vascular-graph-v1'
KINDS = ('artery', 'vein', 'sinus', 'uncertain', 'assumed-link')


def enabled(engine):
    return PROPERTY in engine.scene.properties


def _number(value, name, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or positive and value == 0:
        raise ValueError('INVALID_VASCULAR_GRAPH_NUMBER:' + name)
    return float(value)


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('VASCULAR_GRAPH_EXPLICIT_SOURCE_REQUIRED')


def _rows(rows, required, maximum):
    if not isinstance(rows, list) or len(rows) > maximum:
        raise ValueError('VASCULAR_GRAPH_SIZE_OR_ROWS')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError('COMPLETE_EXPLICIT_VASCULAR_GRAPH_ROW_REQUIRED')
        _text(row['id']); _text(row['source'])
        if row['id'] in result: raise ValueError('DUPLICATE_VASCULAR_GRAPH_ID')
        result[row['id']] = row
    return result


def validate(scene):
    p = scene.properties.get(PROPERTY)
    if p is None: return
    required = {'schema', 'provenance', 'coverage', 'fieldPressurePa', 'nodes', 'edges', 'boundaries', 'regions',
                'compressionForceN', 'coagulationDose', 'clotRatePerSecond'}
    if not isinstance(p, dict) or set(p) != required or p['schema'] != SCHEMA or 'tissuePhysicsV2' not in scene.properties:
        raise ValueError('COMPLETE_EXPLICIT_VASCULAR_GRAPH_V1_REQUIRED')
    if scene.properties.get('bleedingSources'):
        raise ValueError('VASCULAR_GRAPH_CANNOT_SHARE_DECLARED_BLEEDING_SOURCES')
    if p['fieldPressurePa'] != 0 or type(p['fieldPressurePa']) not in (int, float):
        raise ValueError('VASCULAR_GRAPH_REQUIRES_ZERO_GAUGE_FIELD_PRESSURE')
    if not isinstance(p['provenance'], dict) or set(p['provenance']) != {'kind', 'description'} or p['provenance']['kind'] != 'assumed-research':
        raise ValueError('VASCULAR_GRAPH_RESEARCH_PROVENANCE_REQUIRED')
    _text(p['provenance']['description'])
    coverage = p['coverage']
    if (not isinstance(coverage, dict) or set(coverage) != {'sourceCaseId', 'status', 'missing'}
            or coverage['status'] not in ('generated-control', 'partial-acquired')
            or not isinstance(coverage['missing'], list) or not coverage['missing']):
        raise ValueError('VASCULAR_GRAPH_PARTIAL_COVERAGE_REQUIRED')
    _text(coverage['sourceCaseId'])
    for item in coverage['missing']: _text(item)
    for key in ('compressionForceN', 'coagulationDose', 'clotRatePerSecond'):
        _number(p[key], key, positive=key != 'clotRatePerSecond')
    nodes = _rows(p['nodes'], {'id', 'complianceMm3PerPa', 'initialStorageMm3', 'source'}, 64)
    if not nodes: raise ValueError('VASCULAR_GRAPH_NODES_REQUIRED')
    rates = {name: 0. for name in nodes}
    for name, node in nodes.items():
        _number(node['complianceMm3PerPa'], 'compliance', True)
        _number(node['initialStorageMm3'], 'initialStorage')
        if not math.isfinite(node['initialStorageMm3']/node['complianceMm3PerPa']):
            raise ValueError('VASCULAR_GRAPH_PRESSURE_OUTSIDE_NUMERICAL_DOMAIN')
    if not math.isfinite(sum(n['initialStorageMm3'] for n in nodes.values())):
        raise ValueError('VASCULAR_GRAPH_STORAGE_OUTSIDE_NUMERICAL_DOMAIN')
    edges = _rows(p['edges'], {'id', 'from', 'to', 'kind', 'conductanceMm3PerPaS', 'sourceCells',
                              'sourceFace', 'woundConductanceMm3PerPaS', 'source'}, 128)
    claimed_faces = set(); labels = scene.labels.ravel()
    vessel_ids = {r['id'] for r in scene.regions if r['material'] == 'vessel'}
    for edge in edges.values():
        if edge['from'] not in nodes or edge['to'] not in nodes or edge['from'] == edge['to'] or edge['kind'] not in KINDS:
            raise ValueError('INVALID_VASCULAR_GRAPH_EDGE')
        g = _number(edge['conductanceMm3PerPaS'], 'conductance')
        cells = edge['sourceCells']; face = edge['sourceFace']; leaks = edge['woundConductanceMm3PerPaS']
        if (not isinstance(cells, list) or len(cells) > 256
                or any(type(c) is not int or not 0 <= c < labels.size or labels[c] not in vessel_ids for c in cells)
                or len(set(cells)) != len(cells) or not isinstance(face, list)
                or not isinstance(leaks, list) or len(leaks) != 2):
            raise ValueError('INVALID_VASCULAR_GRAPH_SOURCE_BINDING')
        for leak in leaks: _number(leak, 'woundConductance')
        if edge['kind'] == 'assumed-link':
            if cells or face or any(leaks): raise ValueError('ABSTRACT_VASCULAR_LINK_HAS_NO_ACQUIRED_GEOMETRY')
        else:
            if not isinstance(face, list) or len(face) != 2 or any(type(c) is not int or c not in cells for c in face):
                raise ValueError('VASCULAR_GRAPH_ORIENTED_SOURCE_FACE_REQUIRED')
            a, b = (np.array(np.unravel_index(c, scene.shape)) for c in face)
            key = tuple(sorted(face))
            if np.abs(a-b).sum() != 1 or key in claimed_faces:
                raise ValueError('VASCULAR_GRAPH_DISTINCT_NEIGHBOR_FACE_REQUIRED')
            claimed_faces.add(key)
        for name, leak in zip((edge['from'], edge['to']), leaks): rates[name] += g + leak
    boundaries = _rows(p['boundaries'], {'id', 'node', 'pressurePa', 'conductanceMm3PerPaS', 'source'}, 64)
    for b in boundaries.values():
        if b['node'] not in nodes: raise ValueError('INVALID_VASCULAR_GRAPH_BOUNDARY_NODE')
        _number(b['pressurePa'], 'nonnegativeGaugeBoundary')
        rates[b['node']] += _number(b['conductanceMm3PerPaS'], 'boundaryConductance')
        if not math.isfinite(b['pressurePa']*b['conductanceMm3PerPaS']):
            raise ValueError('VASCULAR_GRAPH_BOUNDARY_OUTSIDE_NUMERICAL_DOMAIN')
    if max(rates[n] / nodes[n]['complianceMm3PerPa'] for n in nodes) > 1e4 or p['clotRatePerSecond'] > 1e4:
        raise ValueError('VASCULAR_GRAPH_RATE_OUTSIDE_NUMERICAL_DOMAIN')
    regions = _rows(p['regions'], {'id', 'supply', 'drainage', 'storageNodes', 'referenceSupplyMm3PerSecond',
                                  'referenceDrainageMm3PerSecond', 'source'}, 32)
    for r in regions.values():
        for key in ('supply', 'drainage'):
            interface = r[key]
            if not isinstance(interface, list) or not interface or len(interface) > 128:
                raise ValueError('VASCULAR_GRAPH_EXPLICIT_REGION_INTERFACE_REQUIRED')
            seen = set()
            for item in interface:
                if (not isinstance(item, dict) or set(item) != {'edge', 'direction'} or item['edge'] not in edges
                        or type(item['direction']) is not int or item['direction'] not in (-1, 1) or item['edge'] in seen):
                    raise ValueError('INVALID_VASCULAR_GRAPH_REGION_INTERFACE')
                seen.add(item['edge'])
        storage = r['storageNodes']
        if (not isinstance(storage, list) or not storage or any(not isinstance(n,str) or n not in nodes for n in storage)
                or len(set(storage)) != len(storage)):
            raise ValueError('VASCULAR_GRAPH_EXPLICIT_REGION_STORAGE_REQUIRED')
        for key in ('referenceSupplyMm3PerSecond', 'referenceDrainageMm3PerSecond'): _number(r[key], key, True)


def initialize(engine):
    if not enabled(engine): return None
    engine.scene.require_cache_source()
    if PROPERTY in engine.state: return engine.state[PROPERTY]
    validate(engine.scene); p = engine.scene.properties[PROPERTY]
    state = {'storageMm3': {n['id']: float(n['initialStorageMm3']) for n in p['nodes']},
        'edges': {e['id']: {'cutFace': None, 'compressionFraction': 0., 'occlusionFraction': 0., 'coagulationDose': 0.} for e in p['edges']},
        'ports': {}, 'boundaryInMm3': 0., 'boundaryOutMm3': 0., 'generatedMm3': 0.,
        'roundoffCorrectionMm3': 0., 'elapsedSeconds': 0., 'interval': 0, 'pending': None,
        'unmappedContactCells': [], 'edgeFlowMm3PerSecond': {}, 'regions': {r['id']: {
            'supplyDeficitMm3': 0., 'drainageDeficitMm3': 0., 'supplyDeficitSeconds': 0., 'drainageDeficitSeconds': 0.} for r in p['regions']}}
    engine.state[PROPERTY] = state
    return state


def _idle(state):
    if state['pending'] is not None: raise ValueError('VASCULAR_GRAPH_MATCHING_FIELD_TICK_REQUIRED')


def contact(engine, cells, dt, force_n, temperature=None):
    """Encounter coverage only; no invented forcedamage or thermal injury law."""
    s = initialize(engine); _idle(s)
    p = engine.scene.properties[PROPERTY]
    bound = {c for e in p['edges'] for c in e['sourceCells']}
    labels = engine.scene.labels.ravel()
    unknown = {int(c) for c in cells if engine.by_label[int(labels[c])]['material'] == 'vessel' and c not in bound}
    s['unmappedContactCells'] = sorted(set(s['unmappedContactCells']) | unknown)


def damage(engine, faces, dt=0., kind='transection'):
    _number(dt, 'damageDuration')
    if kind != 'transection': raise ValueError('VASCULAR_GRAPH_UNQUALIFIED_DAMAGE_KIND')
    s = initialize(engine); _idle(s); p = engine.scene.properties[PROPERTY]
    bindings = {tuple(sorted(e['sourceFace'])): e for e in p['edges'] if e['sourceFace']}
    changes = []
    for face in sorted({tuple(sorted(map(int, pair))) for pair in faces}):
        if face not in bindings: raise ValueError('VASCULAR_GRAPH_UNMAPPED_CUT: refine source graph; missing connectivity is unknown')
        e = bindings[face]
        if s['edges'][e['id']]['cutFace'] is None: changes.append(e)
    # Validate the whole set before mutation. One oriented face per lumped edge.
    identifiers = {'vascular:' + e['id'] + ':' + str(side) for e in changes for side in (0, 1)}
    if identifiers.intersection(source['id'] for source in engine.state['sources']):
        raise ValueError('VASCULAR_GRAPH_SOURCE_ID_COLLISION')
    for e in changes:
        s['edges'][e['id']]['cutFace'] = list(e['sourceFace'])
        for side, (node, cell, leak) in enumerate(zip((e['from'], e['to']), e['sourceFace'], e['woundConductanceMm3PerPaS'])):
            identifier = 'vascular:' + e['id'] + ':' + str(side)
            s['ports'][identifier] = {'edge': e['id'], 'node': node, 'cell': cell, 'conductanceMm3PerPaS': float(leak),
                'sealFraction': 0., 'compressionFraction': 0., 'coagulationDose': 0.,
                'openedAtSeconds': s['elapsedSeconds'], 'generatedMm3': 0.}
            engine.state['sources'].append({'id': identifier, 'kind': 'modeled-vessel-wound', 'modelVersion': MODEL_VERSION,
                'cell': cell, 'rateMm3PerSecond': 0., 'emittedMm3': 0., 'reservoirMm3': 0.,
                'bleedingActive': True, 'patent': False, 'exposure': 0.})


def coagulate(engine, cells, power, dt):
    s = initialize(engine); _idle(s); p = engine.scene.properties[PROPERTY]
    dose = _number(power, 'power') * _number(dt, 'duration'); cells = set(cells)
    for edge in p['edges']:
        if cells.intersection(edge['sourceCells']):
            row = s['edges'][edge['id']]; row['coagulationDose'] += dose
            row['occlusionFraction'] = min(1., row['coagulationDose'] / p['coagulationDose'])
    # Only a contacted stump is sealed; distal backbleeding remains possible.
    for port in s['ports'].values():
        if port['cell'] in cells:
            port['coagulationDose'] += dose
            port['sealFraction'] = min(1., port['coagulationDose'] / p['coagulationDose'])


def refresh_compression(engine):
    """Closed bipolar contact uses the same capsules and electrode convention.

    forceN/fullCompressionForce is an explicit uncalibrated geometry proxy.
    Closed unpowered jaws compress; opening/releasing contact restores only this
    reversible factor. Coagulation and cut continuity remain independent.
    """
    s = initialize(engine); _idle(s); p = engine.scene.properties[PROPERTY]
    for row in s['edges'].values(): row['compressionFraction'] = 0.
    for row in s['ports'].values(): row['compressionFraction'] = 0.
    from .surgical_motion import envelopes
    from .surgical_tools import TOOLS
    from .surgical_effects import _current_points
    from scipy.spatial.transform import Rotation
    for slot in engine.state['slots'].values():
        if slot['instrument'] != 'bipolar_forceps' or not slot.get('inserted'): continue
        params = engine.parameters(slot)
        if params['jawOpeningMm'] > .5 or params['forceN'] == 0: continue
        cells = set()
        for a, b, radius, part in envelopes(TOOLS[slot['instrument']], params, slot['position'], slot['quaternion']):
            if part != 'shaft': cells.update(engine.touched(a, b, radius))
        cells = sorted(cells)
        local = Rotation.from_quat(slot['quaternion']).inv().apply(_current_points(engine, cells) - slot['position']) if cells else []
        contacts = {c for c, v in zip(cells, local) if abs(v[0]) <= .35 + params['jawOpeningMm']/2
                    and abs(v[1]) <= .35 and -params['depthMm']-.35 <= v[2] <= .35}
        fraction = min(1., params['forceN'] / p['compressionForceN'])
        for e in p['edges']:
            if contacts.intersection(e['sourceCells']):
                row = s['edges'][e['id']]; row['compressionFraction'] = max(row['compressionFraction'], fraction)
        for port in s['ports'].values():
            if port['cell'] in contacts: port['compressionFraction'] = max(port['compressionFraction'], fraction)


def tick(engine, dt):
    s = initialize(engine); _idle(s); p = engine.scene.properties[PROPERTY]
    _checked_sources(engine, s)
    dt = _number(dt, 'duration')
    if dt == 0: return
    nodes = p['nodes']; edges = p['edges']; bounds = p['boundaries']; regions = p['regions']
    ids = {n['id']: i for i, n in enumerate(nodes)}; n = len(nodes)
    compliance = np.array([r['complianceMm3PerPa'] for r in nodes])
    conductance = np.array([e['conductanceMm3PerPaS'] * (1-s['edges'][e['id']]['occlusionFraction'])
        * (1-s['edges'][e['id']]['compressionFraction']) if s['edges'][e['id']]['cutFace'] is None else 0. for e in edges])
    a = np.array([ids[e['from']] for e in edges], int); b = np.array([ids[e['to']] for e in edges], int)
    edge_ids = {e['id']: i for i, e in enumerate(edges)}
    ports = sorted(s['ports']); port_rows = [s['ports'][k] for k in ports]
    port_nodes = np.array([ids[r['node']] for r in port_rows], int)
    # State: storage, signed boundary in/out, each wound integral, each region's
    # supply/drainage deficit. All ledgers are integrated with the same solution.
    wound_start = n+2; region_start = wound_start+len(ports)
    initial = np.zeros(region_start+2*len(regions)); initial[:n] = [s['storageMm3'][r['id']] for r in nodes]

    def fluxes(t, y):
        pressure = y[:n] / compliance; q = conductance * (pressure[a]-pressure[b])
        wound = np.array([r['conductanceMm3PerPaS'] * (1-r['sealFraction']) * (1-r['compressionFraction'])
            * math.exp(-p['clotRatePerSecond'] * (s['elapsedSeconds']+t-r['openedAtSeconds']))
            * max(0., pressure[i]) for r, i in zip(port_rows, port_nodes)])
        return pressure, q, wound

    def derivative(t, y):
        out = np.zeros_like(y); pressure, q, wound = fluxes(t, y)
        np.add.at(out, a, -q); np.add.at(out, b, q)
        for boundary in bounds:
            i = ids[boundary['node']]; flow = boundary['conductanceMm3PerPaS'] * (boundary['pressurePa']-pressure[i])
            out[i] += flow; out[n] += max(flow, 0.); out[n+1] += max(-flow, 0.)
        np.add.at(out, port_nodes, -wound); out[wound_start:region_start] = wound
        for k, r in enumerate(regions):
            for j, key in enumerate(('supply', 'drainage')):
                delivery = sum(item['direction'] * q[edge_ids[item['edge']]] for item in r[key])
                target = r['referenceSupplyMm3PerSecond' if j == 0 else 'referenceDrainageMm3PerSecond']
                out[region_start+2*k+j] = max(0., target-max(0., delivery))
        return out

    try:
        result = solve_ivp(derivative, (0., dt), initial, method='LSODA', rtol=1e-10, atol=1e-12)
    except (RuntimeError, FloatingPointError) as exc:
        raise ValueError('VASCULAR_GRAPH_INTEGRATION_FAILED') from exc
    if not result.success or not np.isfinite(result.y).all(): raise ValueError('VASCULAR_GRAPH_INTEGRATION_FAILED')
    y = result.y[:, -1]
    tolerance = 1e-10 * max(1., float(initial[:n].sum()))
    if result.y[:n].min() < -tolerance or y[n:].min(initial=0.) < -tolerance:
        raise ValueError('VASCULAR_GRAPH_NEGATIVE_STORAGE_OR_LEDGER')
    correction = float(-np.minimum(y[:n], 0.).sum())
    residual = float(y[:n].sum()-initial[:n].sum()-y[n]+y[n+1]+y[wound_start:region_start].sum())
    if abs(residual) > tolerance: raise ValueError('VASCULAR_GRAPH_CONSERVATION_FAILURE')
    # Only sub-tolerance roundoff is clamped, exposed separately. No ledger is
    # repaired to force a conservation pass; node-level negatives above refuse.
    s['roundoffCorrectionMm3'] += correction
    s['storageMm3'] = {r['id']: max(0., float(y[i])) for i, r in enumerate(nodes)}
    s['boundaryInMm3'] += float(y[n]); s['boundaryOutMm3'] += float(y[n+1])
    amounts = {k: max(0., float(y[wound_start+i])) for i, k in enumerate(ports)}
    for key, amount in amounts.items(): s['ports'][key]['generatedMm3'] += amount
    s['generatedMm3'] += sum(amounts.values())
    s['elapsedSeconds'] += dt; s['interval'] += 1
    s['pending'] = {'interval': s['interval'], 'duration': dt, 'amountsMm3': amounts}
    _, q, _ = fluxes(0., y)
    s['edgeFlowMm3PerSecond'] = {e['id']: float(q[i]) for i, e in enumerate(edges)}
    for k, r in enumerate(regions):
        row = s['regions'][r['id']]
        for j, key in enumerate(('supply', 'drainage')):
            amount = max(0., float(y[region_start+2*k+j])); title = key + 'Deficit'
            row[title+'Mm3'] += amount
            row[title+'Seconds'] += amount / r['referenceSupplyMm3PerSecond' if j == 0 else 'referenceDrainageMm3PerSecond']
            row[key+'Mm3PerSecond'] = float(sum(item['direction']*q[edge_ids[item['edge']]] for item in r[key]))
        row['storageMm3'] = sum(s['storageMm3'][key] for key in r['storageNodes'])
        row['storageChangeMm3'] = row['storageMm3']-sum(node['initialStorageMm3'] for node in nodes if node['id'] in r['storageNodes'])


def _checked_sources(engine, s):
    rows = [r for r in engine.state['sources'] if r.get('modelVersion') == MODEL_VERSION]
    sources = {r['id']: r for r in rows}
    if len(rows) != len(engine.state['sources']) or len(rows) != len(sources) or set(sources) != set(s['ports']):
        raise ValueError('VASCULAR_GRAPH_SOURCE_IDENTITY_MISMATCH')
    pending = s['pending']['amountsMm3'] if s['pending'] else {}
    for key, source in sources.items():
        port = s['ports'][key]
        if (source.get('kind') != 'modeled-vessel-wound' or type(source.get('cell')) is not int
                or source['cell'] != port['cell']): raise ValueError('VASCULAR_GRAPH_SOURCE_IDENTITY_MISMATCH')
        for field in ('emittedMm3', 'reservoirMm3', 'rateMm3PerSecond'): _number(source.get(field), field)
        if (source['reservoirMm3'] > source['emittedMm3']+1e-10 or not math.isclose(
                port['generatedMm3'], source['emittedMm3']+pending.get(key, 0.), rel_tol=1e-10, abs_tol=1e-10)):
            raise ValueError('VASCULAR_GRAPH_SOURCE_LEDGER_MISMATCH')
    if not math.isclose(sum(r['generatedMm3'] for r in s['ports'].values()), s['generatedMm3'], rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError('VASCULAR_GRAPH_SOURCE_LEDGER_MISMATCH')
    return sources


def consume_field_interval(engine, dt):
    """Validate the COMPLETE interval before any fluid/source mutation."""
    s = initialize(engine); pending = s['pending']
    _number(dt, 'fieldDuration')
    if dt == 0 and pending is None:
        _checked_sources(engine, s)
        return False
    if pending is None or pending['interval'] != s['interval'] or dt != pending['duration']:
        raise ValueError('VASCULAR_GRAPH_MATCHING_FIELD_TICK_REQUIRED')
    sources = _checked_sources(engine, s)
    for key, amount in pending['amountsMm3'].items():
        source = sources[key]
        source['emittedMm3'] += amount; source['reservoirMm3'] += amount
        source['rateMm3PerSecond'] = amount / dt
        source['bleedingActive'] = amount > 0
        engine.state['fluidAddedMm3'] += amount
    s['pending'] = None
    return True


def metrics(engine):
    s = initialize(engine); p = engine.scene.properties[PROPERTY]
    sources = [r for r in engine.state['sources'] if r.get('modelVersion') == MODEL_VERSION]
    initial = sum(n['initialStorageMm3'] for n in p['nodes']); remaining = sum(s['storageMm3'].values())
    return {'model': SCHEMA, 'initialPressureSupportingStorageMm3': initial, 'storageMm3': remaining,
        'boundaryInMm3': s['boundaryInMm3'], 'boundaryOutMm3': s['boundaryOutMm3'],
        'generatedMm3': s['generatedMm3'], 'emittedMm3': sum(r['emittedMm3'] for r in sources),
        'pendingEmissionMm3': sum(s['pending']['amountsMm3'].values()) if s['pending'] else 0.,
        'blockedOutletReservoirMm3': sum(r['reservoirMm3'] for r in sources),
        'conservationResidualMm3': initial+s['boundaryInMm3']-s['boundaryOutMm3']-s['generatedMm3']-remaining,
        'roundoffCorrectionMm3': s['roundoffCorrectionMm3'], 'woundPortCount': len(s['ports']),
        'cutEdgeCount': sum(r['cutFace'] is not None for r in s['edges'].values()),
        'occludedEdgeCount': sum(r['occlusionFraction'] >= 1 for r in s['edges'].values()),
        'compressedEdgeCount': sum(r['compressionFraction'] > 0 for r in s['edges'].values()),
        'unmappedContactCells': len(s['unmappedContactCells']), 'regions': copy.deepcopy(s['regions'])}


def descriptor(engine):
    s = initialize(engine); p = engine.scene.properties[PROPERTY]
    bound = {c for e in p['edges'] for c in e['sourceCells']}
    vessel_ids = [r['id'] for r in engine.scene.regions if r['material'] == 'vessel']
    return {'model': SCHEMA, 'status': 'numerically tested research model; patient flow and injury unvalidated',
        'supersedesLegacyVesselProfile': True, 'profile': copy.deepcopy(p),
        'coverage': copy.deepcopy(p['coverage']), 'nodeCount': len(p['nodes']), 'edgeCount': len(p['edges']),
        'boundSourceCells': len(bound), 'vesselSourceCells': int(np.isin(engine.scene.labels, vessel_ids).sum()),
        'edgeKinds': {kind: sum(e['kind'] == kind for e in p['edges']) for kind in KINDS},
        'summary': metrics(engine), 'edges': copy.deepcopy(s['edges']),
        'limits': ['Pressure-supporting storage P=V/C, zero unstressed volume; no patient ICP or systemic blood-volume model.',
            'Declared conductance, compliance, fixed pressure boundaries and territory interfaces; no inferred microvascular tree.',
            'Linear resistors with signed flow, no autoregulation, collapse, pulsatility, oxygen transport or tamponade.',
            'Compression/coagulation/clotting parameters are engineering assumptions; no device/specimen calibration.',
            'Supply/drainage flow-deficit exposure and storage change are not infarction or neurological deficit predictions.',
            'Unmapped cuts refused; unobserved vessels and functional pathways remain unknown.']}
