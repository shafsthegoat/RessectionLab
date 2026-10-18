"""Deterministic cell topology and local fields, explicitly uncalibrated.

Edges refer to persistent source cells. Fluid/debris refer to spatial grid cells.
Cuts retain tissue; only a kerf or a detached forceps extraction removes cells.
"""
from __future__ import annotations
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .surgical_tools import TOOLS


def initialize_effect_state(engine):
    from .surgical_physics import initialize_physics
    initialize_physics(engine)
    for name in ('cutEdges', 'separatedAttachments', 'transectedEdges', 'flapComponents', 'meshCutEdges', 'meshSeams'):
        engine.state.setdefault(name, [])
    engine.state.setdefault('debris', {})
    for source in engine.state['sources']:
        source.setdefault('patent', None)
        source.setdefault('exposure', 0.)
        source.setdefault('bleedingActive', True)


def _neighbors(engine, cell):
    idx = np.array(np.unravel_index(cell, engine.scene.shape))
    for axis in range(3):
        for d in (-1, 1):
            nxt = idx.copy(); nxt[axis] += d
            if np.all(nxt >= 0) and np.all(nxt < engine.scene.shape):
                yield int(np.ravel_multi_index(nxt, engine.scene.shape))


def _edges(engine, cells, allowed=None):
    removed = set(engine.state['removed'])
    labels = engine.scene.labels.ravel()
    result = set()
    for i in cells:
        for j in _neighbors(engine, i):
            if j not in removed and labels[j] and (allowed is None or j in allowed):
                result.add(tuple(sorted((int(i), j))))
    return sorted(result)


def _record_edges(engine, field, edges):
    before = {tuple(e) for e in engine.state[field]}
    engine.state[field] = [list(e) for e in sorted(before | set(edges))]
    return len(set(edges) - before)


def _spatial(engine, p, radius):
    return list(map(int, engine.cells(p, p, radius)))


def _release(engine, cells, kinds):
    if 'attachments' not in engine.scene.properties:
        raise ValueError('MISSING_ATTACHMENT_PROPERTIES')
    done = set(engine.state['separatedAttachments']); changed = []
    for a in engine.scene.properties['attachments']:
        if a['kind'] in kinds and a['id'] not in done and set(a['cells']) & set(cells):
            done.add(a['id']); changed.append(a['id'])
    engine.state['separatedAttachments'] = sorted(done)
    engine.state['separated'] = sorted(done)
    return changed


def _bone_components(engine):
    """Compiled connected components; bounded summaries for patient grids."""
    from scipy.ndimage import label, find_objects
    bone_labels = [identifier for identifier, region in engine.by_label.items() if region['material'] == 'bone']
    mask = np.isin(engine.scene.labels, bone_labels)
    mask.ravel()[engine.state['removed']] = False
    components, count = label(mask)
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    anchors = engine.scene.properties.get('boneAnchors', [])
    anchored = set(map(int, components.ravel()[anchors])) if anchors else set()
    compact = bool(engine.scene.properties.get('tissuePhysics'))
    summaries = []
    for identifier, slices in enumerate(find_objects(components), 1):
        if slices is None: continue
        size = int(sizes[identifier]); cells = []
        if size <= 4096 or not compact:
            local = np.argwhere(components[slices] == identifier)
            local += np.array([part.start for part in slices])
            cells = sorted(map(int, np.ravel_multi_index(local.T, engine.scene.shape)))
        if not compact:
            # Preserve original v1 checkpoint contents for existing rehearsals.
            summaries.append({'cells': cells, 'separable': identifier not in anchored if anchors else None})
            continue
        summaries.append({'componentId': identifier, 'cellCount': size, 'cells': cells,
                          'cellsOmitted': size > 4096,
                          'boundsIjk': [[part.start for part in slices], [part.stop - 1 for part in slices]],
                          'separable': identifier not in anchored if anchors else None})
    engine.state['flapComponents'] = summaries


def apply_effect(engine, key, op, cells, p, q, dt):
    initialize_effect_state(engine)
    state = engine.state; slot = state['slots'][key]; name = slot['instrument']
    effect = TOOLS[name]['effect']; params = engine.parameters(slot)
    p = np.asarray(p, float); orientation = Rotation.from_quat(q)
    if effect in ('incise', 'sheet_cut', 'kerf', 'transect'):
        if cells:
            local = orientation.inv().apply(_current_points(engine, cells) - p)
            cells = [i for i, v in zip(cells, local) if abs(v[2]) <= params['depthMm']]
    if effect in ('incise', 'sheet_cut'):
        if effect == 'sheet_cut' and params['jawOpeningMm'] > 0.5: return []
        # A blade occupies the local x/z plane: roll changes its cutting normal.
        normal = orientation.apply([0., 1., 0.])
        edges = _edges(engine, cells, set(cells))
        cut = []
        for i, j in edges:
            xyz = _current_points(engine, [i, j]); side = (xyz - p) @ normal
            if side.min() <= 0 < side.max(): cut.append((i, j))
        count = _record_edges(engine, 'cutEdges', cut)
        mesh_count = _mesh_seam(engine, p, orientation, params) if effect == 'sheet_cut' else 0
        if count:
            state['cuts'].append({'tool': name, 'cells': sorted(cells), 'edges': [list(e) for e in cut], 'normal': normal.tolist()})
        return [{'kind': 'sheet_opening' if effect == 'sheet_cut' else 'incision', 'edges': count, 'meshEdges': mesh_count}] if count or mesh_count else []
    if effect in ('elevate', 'dissect'):
        released = _release(engine, cells, {'periosteal'} if effect == 'elevate' else {'soft'})
        return [{'kind': 'attachment_release', 'attachments': released}] if released else []
    if effect == 'kerf':
        local = orientation.inv().apply(_current_points(engine, cells) - p) if cells else []
        half_width = float(engine.scene.spacing.min()) * .5
        new = {i for i, v in zip(cells, local) if abs(v[1]) <= half_width} - set(state['removed'])
        state['removed'] = sorted(set(state['removed']) | new)
        for i in new: state['debris'][str(i)] = state['debris'].get(str(i), 0.) + engine.scene.voxel_mm3
        state['debrisMm3'] = sum(state['debris'].values())
        if new: state['boneTopologyDirty'] = True
        return [{'kind': 'bone_kerf', 'cells': len(new), 'components': len(state['flapComponents'])}] if new else []
    if effect == 'transect':
        if params['jawOpeningMm'] > 0.5: return []
        from .surgical_physics import vessel_profile, record_vessel_transections
        profile = vessel_profile(engine)
        if 'strands' not in engine.scene.properties and _source_geometry(engine).get('topology') != 'strand' and profile is None: raise ValueError('MISSING_STRAND_PROPERTIES')
        edges = []
        for strand in engine.scene.properties.get('strands', []):
            eligible = set(strand['cells'])
            edges.extend(_edges(engine, sorted(set(cells) & eligible), eligible))
        if profile is not None:
            labels = engine.scene.labels.ravel()
            local_vessels = [i for i in cells if engine.by_label[int(labels[i])]['material'] == 'vessel']
            for i in local_vessels:
                edges.extend((min(i, j), max(i, j)) for j in _neighbors(engine, i)
                             if labels[j] and j not in engine.state['removed'] and engine.by_label[int(labels[j])]['material'] == 'vessel')
        normal = orientation.apply([0., 1., 0.])
        edges = [edge for edge in edges
                 if (lambda side: side.min() <= 0 < side.max())((_current_points(engine, edge) - p) @ normal)]
        count = _record_edges(engine, 'transectedEdges', edges)
        record_vessel_transections(engine, edges)
        mesh_count = _mesh_seam(engine, p, orientation, params) if _source_geometry(engine).get('topology') == 'strand' else 0
        return [{'kind': 'strand_transection', 'edges': count, 'meshEdges': mesh_count, 'piecesRetained': True}] if count or mesh_count else []
    if effect == 'grasp':
        if op == 'extract':
            held = state['holds'].get(key)
            if not held: raise ValueError('GRASP_BEFORE_EXTRACTION')
            if 'attachments' not in engine.scene.properties: raise ValueError('MISSING_ATTACHMENT_PROPERTIES')
            attached = [a for a in engine.scene.properties['attachments'] if set(a['cells']) & set(held['cells']) and a['id'] not in state['separatedAttachments']]
            if any(not a.get('breakable', False) for a in attached): raise ValueError('ATTACHMENT_PREVENTS_EXTRACTION')
            state['separatedAttachments'] = sorted(set(state['separatedAttachments']) | {a['id'] for a in attached})
            removed = held['cells']; state['removed'] = sorted(set(state['removed']) | set(removed))
            del state['holds'][key]; slot['grip'] = []
            return [{'kind': 'detached_extraction', 'cells': len(removed)}]
        if cells and key not in state['holds']:
            if params['jawOpeningMm'] > .5: return []
            # Grasp the connected retained tumor fragment, not an arbitrary
            # sphere which could otherwise be extracted through intact tissue.
            labels = engine.scene.labels.ravel()
            blocked = {tuple(e) for e in state['cutEdges'] + state['transectedEdges']}
            removed = set(state['removed']); fragment = {int(cells[0])}; stack = list(fragment)
            while stack:
                i = stack.pop()
                for j in _neighbors(engine, i):
                    if j in fragment or j in removed or not labels[j]: continue
                    if engine.by_label[int(labels[j])]['material'] != 'tumor': continue
                    if tuple(sorted((i, j))) in blocked: continue
                    fragment.add(j); stack.append(j)
            state['holds'][key] = {'cells': sorted(fragment), 'origin': p.tolist(), 'offset': [0, 0, 0]}
            slot['grip'] = sorted(fragment)
            return [{'kind': 'grasp_created', 'cells': len(fragment)}]
        return []
    if effect in ('suction', 'irrigate'):
        spatial = _spatial(engine, p, params['radiusMm'])
        budget = params['flowMm3PerSecond'] * dt
        if effect == 'suction':
            amounts = {}
            for field in ('fluid', 'debris'):
                total = 0.
                for i in sorted(spatial):
                    k = str(i); take = min(budget, state[field].get(k, 0.))
                    if take:
                        state[field][k] -= take; total += take; budget -= take
                        if state[field][k] <= 1e-12: del state[field][k]
                amounts[field] = total
            state['fluidRemovedMm3'] += amounts['fluid']; state['debrisMm3'] = sum(state['debris'].values())
            if amounts['debris']: state['debrisRemovedMm3'] = state.get('debrisRemovedMm3', 0.) + amounts['debris']
            return [{'kind': 'aspiration', **amounts}]
        # Directed deposition at the outlet, with explicit out-of-grid escape.
        outlet = p + orientation.apply([0., 0., params['depthMm']])
        index = np.rint(engine.scene.inverse[:3, :3] @ outlet + engine.scene.inverse[:3, 3]).astype(int)
        if np.any(index < 0) or np.any(index >= engine.scene.shape): state['fluidEscapedMm3'] += budget
        else:
            cell = int(np.ravel_multi_index(index, engine.scene.shape))
            destination = _free_outlet(engine, cell)
            if destination is None: raise ValueError('FLUID_OUTLET_BLOCKED')
            k = str(destination)
            state['fluid'][k] = state['fluid'].get(k, 0.) + budget
        state['fluidAddedMm3'] += budget
        return [{'kind': 'irrigation', 'volumeMm3': budget, 'outlet': outlet.tolist()}]
    if effect == 'coagulate':
        exposure = params['power'] * dt
        for i in cells: state['thermal'][str(i)] = state['thermal'].get(str(i), 0.) + exposure
        changed = []
        for source in state['sources']:
            if source['cell'] in cells:
                if 'coagulationExposure' not in source: raise ValueError('MISSING_SOURCE_COAGULATION_PROPERTY')
                source['exposure'] += exposure
                if source['exposure'] >= source['coagulationExposure'] and source.get('bleedingActive', True):
                    source['bleedingActive'] = False; changed.append(source['id'])
        return [{'kind': 'local_energy', 'exposure': exposure, 'controlledBleedingSources': changed}]
    if effect == 'probe':
        probe = name.removesuffix('_probe')
        evidence = [e for e in engine.scene.properties.get('functionalEvidence', []) if e['probe'] == probe and e['cell'] in cells]
        observation = {'kind': 'mapping', 'probe': probe, 'position': p.tolist(), 'cells': sorted(cells),
                       'parameters': {k: params[k] for k in ('amplitudeMa', 'frequencyHz', 'pulseWidthUs') if k in params},
                       'coverage': 'supplied sites only' if evidence else 'unknown',
                       'evidence': copy.deepcopy(evidence), 'result': 'supplied_evidence' if evidence else 'unknown',
                       'uncertainty': 'source supplied' if evidence else 'no matching supplied functional evidence'}
        if not state['observations'] or state['observations'][-1] != observation: state['observations'].append(observation)
        return [observation]
    raise ValueError('UNSUPPORTED_MECHANISM')


def _free_outlet(engine, cell):
    """An outlet may wet its own free cell or an immediately adjacent free cell."""
    occupancy = engine.occupancy()
    if occupancy[cell] == 0: return cell
    return next((i for i in _neighbors(engine, cell) if occupancy[i] == 0), None)


def tick_fields(engine, dt):
    """Explicit bleeding emission, independent of vessel patency.

    A source with no adjacent free outlet accumulates a declared reservoir until
    opened. This avoids placing fluid inside solid tissue or losing mass.
    """
    for source in engine.state['sources']:
        if source.get('bleedingActive', True):
            if 'rateMm3PerSecond' not in source or 'cell' not in source: raise ValueError('MISSING_BLEEDING_SOURCE_PROPERTIES')
            amount = source['rateMm3PerSecond'] * dt
            if source.get('kind') == 'modeled-vessel-wound': source['emittedMm3'] += amount
            source['reservoirMm3'] = source.get('reservoirMm3', 0.) + amount
            engine.state['fluidAddedMm3'] += amount
        destination = _free_outlet(engine, source['cell'])
        if destination is not None:
            amount = source.get('reservoirMm3', 0.)
            if amount:
                k = str(destination); engine.state['fluid'][k] = engine.state['fluid'].get(k, 0.) + amount
                source['reservoirMm3'] = 0.
    # One bounded nearest-neighbor relaxation per microstep. A cell holds one
    # voxel-volume before overflow wets neighboring free cells. Boundary overflow
    # enters an explicit escaped reservoir; no unaccounted fluid disappears.
    occupancy = engine.occupancy(); capacity = engine.scene.voxel_mm3
    for key in list(engine.state['fluid']):
        if occupancy[int(key)]:
            destination = _free_outlet(engine, int(key))
            amount = engine.state['fluid'].pop(key)
            if destination is None:
                engine.state['fluidEscapedMm3'] += amount
            else:
                dk = str(destination); engine.state['fluid'][dk] = engine.state['fluid'].get(dk, 0.) + amount
    for key, amount in sorted(list(engine.state['fluid'].items()), key=lambda item: int(item[0])):
        cell = int(key); excess = max(0., engine.state['fluid'].get(key, 0.) - capacity)
        for neighbor in _neighbors(engine, cell):
            if not excess: break
            if occupancy[neighbor]: continue
            nk = str(neighbor)
            transfer = min(excess, max(0., capacity - engine.state['fluid'].get(nk, 0.)))
            if transfer:
                sediment = engine.state['debris'].get(key, 0.) * min(1., transfer / engine.state['fluid'][key])
                if sediment:
                    engine.state['debris'][key] -= sediment
                    engine.state['debris'][nk] = engine.state['debris'].get(nk, 0.) + sediment
                engine.state['fluid'][nk] = engine.state['fluid'].get(nk, 0.) + transfer
                engine.state['fluid'][key] -= transfer; excess -= transfer
        index = np.array(np.unravel_index(cell, engine.scene.shape))
        if excess and (np.any(index == 0) or np.any(index == np.array(engine.scene.shape) - 1)):
            sediment = engine.state['debris'].get(key, 0.) * min(1., excess / engine.state['fluid'][key])
            if sediment:
                engine.state['debris'][key] -= sediment
                engine.state['debrisEscapedMm3'] = engine.state.get('debrisEscapedMm3', 0.) + sediment
            engine.state['fluid'][key] -= excess; engine.state['fluidEscapedMm3'] += excess
    engine.state['debrisMm3'] = sum(engine.state['debris'].values())


def fixture_properties(labels):
    """Synthetic, declared properties for generated engineering fixtures only."""
    groups = {i: list(map(int, np.flatnonzero(labels.ravel() == i))) for i in range(1, 6)}
    return {'attachments': ([{'id': 'fixture-periosteal', 'kind': 'periosteal', 'cells': groups[1][:20]}] if groups[1] else []) +
            ([{'id': 'fixture-soft', 'kind': 'soft', 'cells': groups[2][:20]}] if groups[2] else []) +
            ([{'id': 'fixture-tumor', 'kind': 'tumor', 'cells': groups[3][:20], 'breakable': False}] if groups[3] else []),
            'boneAnchors': groups[1][:1], 'strands': [{'id': 'fixture-strand', 'cells': groups[5]}] if groups[5] else [],
            'bleedingSources': [{'id': 'fixture-leak', 'cell': groups[5][0], 'rateMm3PerSecond': 1., 'coagulationExposure': 2., 'patent': True}] if groups[5] else [],
            'functionalEvidence': [{'id': 'fixture-cortical-evidence', 'cell': groups[2][0], 'probe': 'cortical', 'result': 'synthetic positive', 'uncertainty': 'engineering fixture only'}] if groups[2] else []}


def validate_hold_motion(engine, key):
    """Prevent moving a forceps fragment through declared intact attachments."""
    if engine.state['slots'][key]['instrument'] != 'tumor_forceps': return
    if 'attachments' not in engine.scene.properties: raise ValueError('MISSING_ATTACHMENT_PROPERTIES')
    hold = engine.state['holds'][key]
    attached = [a for a in engine.scene.properties['attachments']
                if set(a['cells']) & set(hold['cells']) and a['id'] not in engine.state['separatedAttachments']]
    if any(not a.get('breakable', False) for a in attached): raise ValueError('ATTACHMENT_PREVENTS_DISPLACEMENT')
    engine.state['separatedAttachments'] = sorted(set(engine.state['separatedAttachments']) | {a['id'] for a in attached})


def finalize_action(engine):
    """Recompute bone connectivity once after an action that changed its kerf."""
    if engine.state.pop('boneTopologyDirty', False): _bone_components(engine)


def _mesh_seam(engine, p, orientation, params):
    """Cut a retained mesh edge graph and record world-coordinate face seams.

    Faces are retained source identities; this is a connectivity seam, not a
    remeshed finite-element surface or physical opening displacement.
    """
    source = _source_geometry(engine)
    if source.get('topology') not in ('sheet', 'strand'): return 0
    if any(any(hold['offset']) for hold in engine.state['holds'].values()):
        raise ValueError('MESH_SEAM_ON_DISPLACED_GEOMETRY_UNSUPPORTED')
    scale = {'mm': 1., 'meter': 1000., 'micron': .001}.get(source.get('units'))
    if scale is None or source.get('frame') not in ('RAS+', 'LPS+'):
        raise ValueError('MESH_SEAM_REQUIRES_DECLARED_FRAME_UNITS')
    vertices = np.asarray(source['vertices'], float) * scale
    if source['frame'] == 'LPS+': vertices[:, :2] *= -1
    faces = source.get('faces', []); lines = source.get('lines', [])
    edges = {tuple(sorted((int(a), int(b)))) for a, b in lines}
    for face in faces:
        edges.update(tuple(sorted((int(face[i]), int(face[(i + 1) % 3])))) for i in range(3))
    local = orientation.inv().apply(vertices - p); crossed = {}; old = {tuple(e) for e in engine.state['meshCutEdges']}
    for a, b in sorted(edges):
        u, v = local[a], local[b]
        if not min(u[1], v[1]) <= 0 < max(u[1], v[1]): continue
        t = -u[1] / (v[1] - u[1]); point = u + t * (v - u)
        if abs(point[2]) > params['depthMm'] or np.linalg.norm(point[[0, 2]]) > params['radiusMm']: continue
        crossed[(a, b)] = (vertices[a] + t * (vertices[b] - vertices[a])).tolist()
    count = _record_edges(engine, 'meshCutEdges', crossed)
    for face_id, face in enumerate(faces):
        hits = [crossed[e] for e in crossed if set(e) <= set(face)]
        if len(hits) >= 2:
            seam = {'faceId': face_id, 'points': hits[:2], 'kind': 'retained_sheet_seam'}
            if seam not in engine.state['meshSeams']: engine.state['meshSeams'].append(seam)
    for edge, point in crossed.items():
        if edge not in old and source['topology'] == 'strand':
            engine.state['meshSeams'].append({'edge': list(edge), 'points': [point], 'kind': 'retained_strand_break'})
    return count


def _source_geometry(engine):
    """Case adapters retain mesh geometry under source.adapter."""
    source = engine.scene.source
    adapter = source.get('adapter')
    return adapter if isinstance(adapter, dict) and adapter.get('type') in ('mesh', 'surface_buffers') else source


def _current_points(engine, cells):
    """Blade geometry follows persistent source cells through lattice holds."""
    ids = list(cells)
    points = engine.scene.points(ids)
    locations = {int(cell): index for index, cell in enumerate(ids)}
    for hold in engine.state['holds'].values():
        offset = engine.scene.affine[:3, :3] @ np.asarray(hold['offset'], float)
        for cell in hold['cells']:
            if cell in locations: points[locations[cell]] += offset
    return points
