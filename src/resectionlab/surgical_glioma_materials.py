"""Declared local material assignments for a generated glioma rehearsal.

Small-gradient, overdamped lattice with a parallel Maxwell branch. Source labels
remain immutable. Assumed spatial variation is independent of target flags.
"""

from __future__ import annotations

import math
import numpy as np

PROPERTY = 'gliomaMaterialsV1'
SCHEMA = 'resectionlab.generated-glioma-materials.v1'
RETAINED_VESSEL_SEGMENTS = 'retained-source-segments-v1'
SOFT = ('soft_tissue', 'tumor')


def research_profile():
    return {
        'schema': SCHEMA,
        'kind': 'assigned-generated-research',
        'equilibriumShearPa': {'soft_tissue': [300.0, 750.0], 'tumor': [250.0, 1200.0]},
        'referenceCapacityPerSolidVolume': {'soft_tissue': [1.0, 1.25], 'tumor': [1.0, 1.4]},
        'wavelengthMm': 8.0,
        'maxwellShearRatio': 2.8,
        'relaxationSeconds': 10.0,
        'vesselAttachments': [],
        'scalpSeamCell': None,
        'biopsyNeckFace': [],
        'axialStiffnessN': {'artery': 0.06, 'vein': 0.025},
        'effectiveLumenAreaMm2': {'artery': 0.8, 'vein': 1.2},
        'tractionFailureStrain': 0.18,
        'sources': [
            'https://pubmed.ncbi.nlm.nih.gov/27989920/',
            'https://pubmed.ncbi.nlm.nih.gov/28658600/',
            'https://pubmed.ncbi.nlm.nih.gov/36514891/',
            'https://doi.org/10.2147/IJN.S270147',
            'https://pubmed.ncbi.nlm.nih.gov/18855141/',
            'https://pmc.ncbi.nlm.nih.gov/articles/PMC9463293/',
        ],
        'assignment': 'Human ex vivo brain relaxation informs the 10 s branch. Ranges, spatial phase, vascular dimensions, '
        'injury threshold and ultrasonic resistance are assigned engineering scenarios. AFM tumor heterogeneity '
        'motivates variation; its indentation moduli are not transferred to tissue shear.',
        'boundary': 'Local patch clamped at retained exterior, weak unresolved tissue foundation, finite divergence '
        'penalty; no resolved interstitial flow or intrinsic gravity. A declared posture supplies body force.',
    }


def _bounded(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def validate(scene):
    from . import surgical_vessel_rods as rods
    from . import surgical_wall_sweep as wall_sweep

    p = scene.properties.get(PROPERTY)
    if p is None:
        return
    optional = {'scalpSeamCell', 'biopsyNeckFace', 'vesselMotionModel', rods.GEOMETRY, wall_sweep.PROPERTY}
    if (
        not isinstance(p, dict)
        or set(p) | optional != set(research_profile()) | optional
        or p['schema'] != SCHEMA
        or p['kind'] != 'assigned-generated-research'
        or 'tissuePhysicsV2' not in scene.properties
        or 'brainContinuumV1' in scene.properties
    ):
        raise ValueError('GLIOMA_MATERIALS_REQUIRE_VERSIONED_LOCAL_LATTICE')
    if 'vesselMotionModel' in p:
        if p['vesselMotionModel'] not in (RETAINED_VESSEL_SEGMENTS, rods.MODEL):
            raise ValueError('GLIOMA_VESSEL_MOTION_MODEL_REQUIRED')
        if 'vascularGraphV1' not in scene.properties:
            raise ValueError('GLIOMA_VESSEL_MOTION_REQUIRES_MAPPED_SOURCE_CHAINS')
    if wall_sweep.PROPERTY in p and (p[wall_sweep.PROPERTY] != wall_sweep.POLICY
                                     or p.get('vesselMotionModel') != rods.MODEL):
        raise ValueError('VESSEL_WALL_SWEEP_REQUIRES_NATIVE_DISTRIBUTED_RODS')
    if not all(isinstance(p[key], str) and p[key].strip() for key in ('assignment', 'boundary')) or not (
        isinstance(p['sources'], list)
        and p['sources']
        and all(isinstance(url, str) and url.startswith('https://') for url in p['sources'])
    ):
        raise ValueError('GLIOMA_MATERIAL_SOURCE_REQUIRED')
    for name, low, high in (
        ('equilibriumShearPa', 10.0, 20000.0),
        ('referenceCapacityPerSolidVolume', 1.0, 5.0),
    ):
        field = p[name]
        if not isinstance(field, dict) or set(field) != set(SOFT):
            raise ValueError('GLIOMA_MATERIAL_RANGES_REQUIRED')
        for pair in field.values():
            if not (
                isinstance(pair, list)
                and len(pair) == 2
                and all(_bounded(x, low, high) for x in pair)
                and pair[0] <= pair[1]
            ):
                raise ValueError('GLIOMA_MATERIAL_RANGE_OUTSIDE_DOMAIN')
    for name, low, high in (
        ('wavelengthMm', 1.0, 100.0),
        ('maxwellShearRatio', 0.0, 5.0),
        ('relaxationSeconds', 0.1, 100.0),
        ('tractionFailureStrain', 0.02, 1.0),
    ):
        if not _bounded(p[name], low, high):
            raise ValueError('GLIOMA_MATERIAL_PARAMETER_OUTSIDE_DOMAIN:' + name)
    kinds = {'artery', 'vein'}
    if 'vesselMotionModel' in p and isinstance(p['axialStiffnessN'], dict) and 'uncertain' in p['axialStiffnessN']:
        kinds.add('uncertain')
    for name in ('axialStiffnessN', 'effectiveLumenAreaMm2'):
        if not (
            isinstance(p[name], dict)
            and set(p[name]) == kinds
            and all(_bounded(x, 0.0001, 10.0) for x in p[name].values())
        ):
            raise ValueError('GLIOMA_VESSEL_IDENTITY_PROFILE_REQUIRED')
    attachments = p['vesselAttachments']
    distributed = p.get('vesselMotionModel') == rods.MODEL
    edges = {
        c
        for e in scene.properties.get('vascularGraphV1', {}).get('edges', [])
        if e['kind'] in kinds
        for c in e['sourceCells']
    }
    materials = {r['id']: r['material'] for r in scene.regions}
    if distributed:
        rods.validate_geometry(scene, p, edges)
    elif rods.GEOMETRY in p:
        raise ValueError('GLIOMA_ROD_GEOMETRY_REQUIRES_DISTRIBUTED_MOTION')
    if not isinstance(attachments, list) or len(attachments) > 512:
        raise ValueError('GLIOMA_VESSEL_ATTACHMENT_LIMIT')
    seen = set()
    for pair in attachments:
        if not (
            isinstance(pair, list)
            and len(pair) == 2
            and all(type(c) is int and 0 <= c < scene.labels.size for c in pair)
        ):
            raise ValueError('GLIOMA_ATTACHMENT_REQUIRES_SOURCE_PAIR')
        vessel, tissue = pair
        if (
            ((vessel, tissue) if distributed else vessel) in seen
            or vessel not in edges
            or materials.get(int(scene.labels.flat[tissue])) not in SOFT
        ):
            raise ValueError('GLIOMA_ATTACHMENT_REQUIRES_MAPPED_ADJACENT_TISSUE')
        if distributed and rods.GEOMETRY in p:
            from .surgical_vessel_surface import source_cell_radius
            cx, cy, cz, radius = p[rods.GEOMETRY]['sites'][str(vessel)]
            distance = np.linalg.norm(scene.points([tissue])[0] - (cx, cy, cz))
            valid = distance <= radius + 2 * source_cell_radius(scene) + 1e-7
        else:
            valid = sum(abs(a - b) for a, b in zip(np.unravel_index(vessel, scene.shape),
                                                  np.unravel_index(tissue, scene.shape))) == 1
        if not valid:
            raise ValueError('GLIOMA_ATTACHMENT_REQUIRES_MAPPED_ADJACENT_TISSUE')
        seen.add((vessel, tissue) if distributed else vessel)
    scalp_cell = p.get('scalpSeamCell')
    if scalp_cell is not None and (
        type(scalp_cell) is not int
        or not 0 <= scalp_cell < scene.labels.size
        or materials.get(int(scene.labels.flat[scalp_cell])) not in SOFT
        or not any(
            e['id'] == 'scalp' and e['kind'] == 'artery'
            for e in scene.properties.get('vascularGraphV1', {}).get('edges', [])
        )
    ):
        raise ValueError('GLIOMA_SCALP_BLEED_REQUIRES_SOFT_SEAM_AND_DECLARED_ARTERIAL_FACE')
    neck = p.get('biopsyNeckFace', [])
    bed = [
        e
        for e in scene.properties.get('vascularGraphV1', {}).get('edges', [])
        if e['kind'] == 'microvascular-bed'
    ]
    if (
        not isinstance(neck, list)
        or len(bed) != bool(neck)
        or neck
        and (
            len(neck) != 2
            or bed[0]['id'] != 'biopsy-bed'
            or bed[0]['sourceFace'] != neck
            or bed[0]['sourceCells'] != neck
            or any(
                type(c) is not int
                or not 0 <= c < scene.labels.size
                or materials.get(int(scene.labels.flat[c])) not in SOFT
                for c in neck
            )
        )
    ):
        raise ValueError('GLIOMA_BIOPSY_BED_REQUIRES_BOUND_TISSUE_FACE')


def retained_vessel_segments(engine):
    return engine.scene.properties.get(PROPERTY, {}).get('vesselMotionModel') == RETAINED_VESSEL_SEGMENTS


def vessel_displacements(engine):
    from . import surgical_vessel_rods as rods

    deformation = engine.state.get('deformation', {})
    if not deformation or not (retained_vessel_segments(engine) or rods.enabled(engine)):
        return {}
    removed = set(engine.state['removed'])
    cells = rods.definitions(engine).cells if rods.enabled(engine) else (
        vessel for vessel, _ in engine.scene.properties[PROPERTY]['vesselAttachments'])
    return {vessel: deformation[str(vessel)] for vessel in cells
            if vessel not in removed and str(vessel) in deformation and any(deformation[str(vessel)])}


def continuous_vessel_contacts(engine, a, b, radius, raster_contacts):
    """Test owned vessel voxels at their accepted subvoxel centers."""
    from . import surgical_vessel_rods as rods

    if rods.enabled(engine):
        return rods.contacts(engine, a, b, radius, raster_contacts)
    displacement = vessel_displacements(engine)
    if not displacement:
        return raster_contacts
    from .geometry import _EPS, _segment_cell_distances

    sites = list(displacement)
    indices = np.column_stack(np.unravel_index(sites, engine.scene.shape))
    moved = indices + np.asarray(list(displacement.values())) @ engine.scene.inverse[:3, :3].T
    distances = _segment_cell_distances(engine.grid, moved, np.asarray(a), np.asarray(b))
    accepted = {site for site, distance in zip(sites, distances) if distance <= radius + _EPS}
    return sorted((set(raster_contacts) - displacement.keys()) | accepted)


def validate_vessel_chains(scene):
    from . import surgical_vessel_rods as rods

    edges = scene.properties['vascularGraphV1']['edges']
    profile = scene.properties[PROPERTY]
    kinds = profile['axialStiffnessN']
    sites = profile.get(rods.GEOMETRY, {}).get('sites', {})
    chains = [edge for edge in edges if edge['kind'] in kinds]
    if not chains:
        raise ValueError('GLIOMA_VESSEL_MOTION_REQUIRES_MAPPED_SOURCE_CHAINS')
    claimed = {tuple(sorted(edge['sourceFace'])) for edge in edges if edge['kind'] not in kinds}
    for edge in chains:
        cells = edge['sourceCells']
        pairs = {tuple(sorted(pair)) for pair in zip(cells, cells[1:])}
        if (tuple(sorted(edge['sourceFace'])) not in pairs or pairs & claimed
                or any(sum(abs(i - j) for i, j in np.unravel_index(pair, scene.shape)) != 1
                       for pair in pairs)):
            raise ValueError('GLIOMA_VESSEL_MOTION_REQUIRES_UNIQUE_ORDERED_SOURCE_FACES')
        if sites and any(np.linalg.norm(np.subtract(sites[str(i)][:3], sites[str(j)][:3])) < .05 * min(scene.spacing)
                         for i, j in pairs):
            raise ValueError('GLIOMA_ROD_GEOMETRY_REQUIRES_DISTINCT_NEIGHBORS')
        claimed.update(pairs)


def oriented_source_cells(edge):
    cells = edge['sourceCells']
    start, end = edge['sourceFace']
    return cells if cells.index(start) < cells.index(end) else cells[::-1]


def material_values(engine, cells, field):
    """A source-bound physical wavelength, independent of action history or targets."""
    p = engine.scene.properties[PROPERTY]
    ids = np.asarray(cells, dtype=int)
    coords = engine.scene.points(ids)
    phase = (coords @ np.asarray([0.8, -0.48, 0.36])) * (2 * math.pi / p['wavelengthMm'])
    weights = 0.5 + 0.5 * np.sin(phase)
    ranges = np.asarray(
        [p[field][engine.by_label[int(engine.scene.labels.flat[i])]['material']] for i in ids]
    )
    return ranges[:, 0] + weights * (ranges[:, 1] - ranges[:, 0])


def generated_vascular_properties(
    shape,
    artery_ijk,
    vein_ijk,
    support_by_ijk,
    source_id,
    *,
    faces=None,
    scalp_ijk=None,
    scalp_seam_ijk=None,
    biopsy_neck_ijk=None,
    vessel_motion_model=None,
):
    """An explicit two-branch, three-node assumed RC circuit on generated centerlines."""
    line = lambda p: int(np.ravel_multi_index(tuple(p), shape))
    p = research_profile()
    if vessel_motion_model is not None:
        p['vesselMotionModel'] = vessel_motion_model
    if vessel_motion_model == RETAINED_VESSEL_SEGMENTS:
        p['assignment'] += (' Source-bound vessel sites follow their assigned soft support and retain their last '
                            'accepted position if that support disappears. Neighbor strain and wound behavior are '
                            'assigned engineering estimates. Free-vessel dynamics are unresolved.')
    p['vesselAttachments'] = sorted([line(v), line(t)] for v, t in support_by_ijk.items())
    source = 'Assigned small generated contact circuit; no subject pressure, flow or failure measurements.'
    nodes = [
        dict(id=key, complianceMm3PerPa=c, initialStorageMm3=c * pressure, source=source)
        for key, c, pressure in [('inlet', 0.002, 2400.0), ('bed', 0.004, 1400.0), ('outlet', 0.007, 400.0)]
    ]
    edges = []
    for name, path, start, end in [
        ('artery', artery_ijk, 'inlet', 'bed'),
        ('vein', vein_ijk, 'bed', 'outlet'),
    ]:
        cells = [line(x) for x in path]
        if len(cells) < 2:
            raise ValueError('MAPPED_VESSEL_REQUIRES_AT_LEAST_TWO_CELLS')
        mid = (len(cells) - 1) // 2
        face = [line(x) for x in faces[name]] if faces and name in faces else cells[mid : mid + 2]
        edges.append(
            {
                'id': name,
                'kind': name,
                'from': start,
                'to': end,
                'sourceCells': cells,
                'sourceFace': face,
                'conductanceMm3PerPaS': 0.0002,
                'woundConductanceMm3PerPaS': [0.00035, 0.00035],
                'source': source,
            }
        )
    graph = {
        'schema': 'resectionlab.vascular-graph.v1',
        'coverage': {
            'status': 'generated-control',
            'sourceCaseId': source_id,
            'missing': ['Systemic circulation, distal branches and patient perfusion'],
        },
        'provenance': {'kind': 'assumed-research', 'description': source},
        'fieldPressurePa': 0.0,
        'nodes': nodes,
        'edges': edges,
        'boundaries': [
            dict(id=id_, node=node, pressurePa=pressure, conductanceMm3PerPaS=0.001, source=source)
            for id_, node, pressure in [('supply', 'inlet', 2600.0), ('drain', 'outlet', 200.0)]
        ],
        'regions': [
            dict(
                id='local-bed',
                supply=[dict(edge='artery', direction=1)],
                drainage=[dict(edge='vein', direction=1)],
                storageNodes=['bed'],
                referenceSupplyMm3PerSecond=0.2,
                referenceDrainageMm3PerSecond=0.2,
                source=source,
            )
        ],
        'compressionForceN': 0.04,
        'coagulationDose': 2.0,
        'clotRatePerSecond': 0.02,
    }
    if scalp_ijk is not None or scalp_seam_ijk is not None:
        if scalp_ijk is None or len(scalp_ijk) < 2 or scalp_seam_ijk is None:
            raise ValueError('SCALP_GRAPH_REQUIRES_CHAIN_AND_INCISION_BINDING')
        cells = [line(x) for x in scalp_ijk]
        face = [line(x) for x in faces['scalp']] if faces and 'scalp' in faces else cells[:2]
        p['scalpSeamCell'] = line(scalp_seam_ijk)
        # Separate extracranial storage and boundary; the blind distal bed has
        # no imposed backbleeding and no connection to the cerebral network.
        nodes.extend(
            [
                dict(id='scalp-in', complianceMm3PerPa=0.002, initialStorageMm3=3.6, source=source),
                dict(id='scalp-bed', complianceMm3PerPa=0.002, initialStorageMm3=1.0, source=source),
            ]
        )
        graph['boundaries'].append(
            dict(
                id='scalp-supply',
                node='scalp-in',
                pressurePa=1800.0,
                conductanceMm3PerPaS=0.0004,
                source=source,
            )
        )
        edges.append(
            {
                'id': 'scalp',
                'kind': 'artery',
                'from': 'scalp-in',
                'to': 'scalp-bed',
                'sourceCells': cells,
                'sourceFace': face,
                'conductanceMm3PerPaS': 0.0001,
                'woundConductanceMm3PerPaS': [0.00022, 0.0],
                'source': source,
            }
        )
    if biopsy_neck_ijk is not None:
        distal, retained = (line(x) for x in biopsy_neck_ijk)
        face = [retained, distal]
        p['biopsyNeckFace'] = face
        bed_source = 'Generated finite microvascular storage carried by tissue at the specimen neck; no resolved vessel lumen.'
        nodes.extend(
            [
                dict(id='biopsy-bed-in', complianceMm3PerPa=0.008, initialStorageMm3=8.0, source=bed_source),
                dict(
                    id='biopsy-bed-tip', complianceMm3PerPa=0.0001, initialStorageMm3=0.1, source=bed_source
                ),
            ]
        )
        edges.append(
            {
                'id': 'biopsy-bed',
                'kind': 'microvascular-bed',
                'from': 'biopsy-bed-in',
                'to': 'biopsy-bed-tip',
                'sourceCells': face,
                'sourceFace': face,
                'conductanceMm3PerPaS': 0.0001,
                'woundConductanceMm3PerPaS': [0.0003, 0.0],
                'source': bed_source,
            }
        )
    return {PROPERTY: p, 'vascularGraphV1': graph}


def gradient_stencil(engine, ids, edges):
    """Centered differences average surviving bonds, using zero slope on free axes."""
    shape = engine.scene.shape
    grid = np.column_stack(np.unravel_index(ids, shape))
    pairs = np.asarray([(i, j) for i, j, _ in edges], dtype=int).reshape(-1, 2)
    i, j = pairs.T
    axes = np.argmax(grid[j] - grid[i], axis=1)
    counts = np.zeros((len(ids), 3))
    np.add.at(counts, (i, axes), 1)
    np.add.at(counts, (j, axes), 1)
    return i, j, axes, 1 / np.maximum(1, counts), engine.scene.inverse[:3, :3]


def _gradients(stencil, u):
    i, j, axes, weights, inverse = stencil
    sums = np.zeros((len(u), 3, 3))
    du = u[j] - u[i]
    np.add.at(sums, (i, axes), du)
    np.add.at(sums, (j, axes), du)
    return (sums * weights[:, :, None]).transpose(0, 2, 1) @ inverse


def limit_gradient(stencil, u):
    """Keep the reduced solution in its small-gradient, positive-volume domain."""
    grads = _gradients(stencil, u)
    peak = float(np.linalg.norm(grads, axis=(1, 2)).max(initial=0.0))
    scale = min(1.0, 0.2 / max(peak, 1e-15))
    ratios = np.linalg.det(np.eye(3) + grads * scale)
    if ratios.min() < 0.82 or ratios.max() > 1.18:
        # ||grad||_2 <= ||grad||_F: bound the product of the three singular
        # values of I+grad, including shear and anisotropic affine axes.
        scale = min(scale, (1.18 ** (1 / 3) - 1) / peak)
    return u if scale == 1.0 else u * scale


def record_lattice(engine, ids, stencil, u, memory, K, targets, held, settings):
    p = engine.scene.properties[PROPERTY]
    model = engine.state['deformationModel']
    model['maxwellMemoryMm'] = {str(c): memory[i].tolist() for i, c in enumerate(ids)}
    transient = (u - memory).ravel()
    model['elasticEnergyNmm'] += 0.5 * p['maxwellShearRatio'] * float(transient @ (K @ transient))
    grads = _gradients(stencil, u)
    ratios = np.linalg.det(np.eye(3) + grads)
    nodes = [i for i, c in enumerate(ids) if c in held]
    reactions = settings['holdStiffnessNPerMm'] * (targets[nodes] - u[nodes])
    model.update(
        model=SCHEMA,
        activeReferenceVolumeMm3=len(ids) * engine.scene.voxel_mm3,
        activeDeformedVolumeMm3=float(ratios.sum()) * engine.scene.voxel_mm3,
        minVolumeRatio=float(ratios.min()),
        maxVolumeRatio=float(ratios.max()),
        maxGradientNorm=float(np.linalg.norm(grads, axis=(1, 2)).max()),
        summedHandleLoadN=float(np.linalg.norm(reactions, axis=1).sum()),
    )


def move_vessels(engine):
    """A declared surface attachment transfers displacement to its source vessel cell."""
    from . import surgical_vessel_rods as rods

    if rods.enabled(engine):
        return
    p = engine.scene.properties[PROPERTY]
    deformation = engine.state['deformation']
    if not p['vesselAttachments'] or not deformation:
        return
    removed = set(engine.state['removed'])
    cut = {tuple(sorted(face)) for k in ('cutEdges', 'transectedEdges') for face in engine.state.get(k, [])}
    retain = retained_vessel_segments(engine)
    for vessel, tissue in p['vesselAttachments']:
        if vessel in removed:
            deformation.pop(str(vessel), None)
            continue
        if tissue in removed or tuple(sorted((tissue, vessel))) in cut:
            if not retain:
                deformation.pop(str(vessel), None)
            continue
        displacement = deformation.get(str(tissue))
        if displacement is None and retain:
            continue
        if displacement is not None and any(displacement):
            deformation[str(vessel)] = displacement.copy()
        else:
            deformation.pop(str(vessel), None)


def outlet_cell(engine, source_cell):
    displacement = engine.state.get('deformation', {}).get(str(source_cell))
    if displacement is None:
        return source_cell
    dst = np.rint(
        np.asarray(np.unravel_index(source_cell, engine.scene.shape))
        + engine.scene.inverse[:3, :3] @ displacement
    ).astype(int)
    return int(np.ravel_multi_index(dst, engine.scene.shape))


def scalp_wound(engine, graph_profile, graph_state):
    """An actual cut seam opens its bound arterial face; an applied Raney arrests local inflow."""
    seam_cell = engine.scene.properties[PROPERTY].get('scalpSeamCell')
    exposure = engine.state.get('operativeExposureV1')
    if seam_cell is None or not exposure:
        return
    profile = engine.scene.properties['operativeExposureV1']['bodies']['scalp']
    accepted = exposure['bodies']['scalp']['cut']
    edges = [profile['seamEdges'][i] for i in accepted if seam_cell in profile['seamEdges'][i]]
    if not edges:
        return
    from .surgical_vascular_graph import damage

    row = graph_state['edges']['scalp']
    if row['cutFace'] is None:
        face = next(e['sourceFace'] for e in graph_profile['edges'] if e['id'] == 'scalp')
        damage(engine, [face])
        row['injuryCause'] = 'incised-bound-scalp-margin'
    port = graph_state['ports']['vascular:scalp:0']
    center = engine.scene.points([port['cell']])[0]
    if any(
        obj['kind'] == 'raney-clip'
        and obj['status'] == 'placed'
        and obj.get('sourceCell') == port['cell']
        and np.linalg.norm(np.asarray(obj['positionMm']) - center) <= obj['radiusMm']
        for obj in exposure['objects']
    ):
        port['compressionFraction'] = 1.0


def biopsy_wound(engine, graph_state):
    face = engine.scene.properties[PROPERTY].get('biopsyNeckFace')
    if not face:
        return
    row = graph_state['edges']['biopsy-bed']
    if row['cutFace'] is None:
        if sorted(face) not in engine.state.get('cutEdges', []):
            return
        from .surgical_vascular_graph import damage

        damage(engine, [face])
        row['injuryCause'] = 'sectioned-specimen-neck'
    port = graph_state['ports']['vascular:biopsy-bed:0']
    for obj in engine.state.get('operativeExposureV1', {}).get('objects', []):
        if (
            obj['kind'] == 'hemostat'
            and obj['status'] == 'intended-retained'
            and obj.get('sourceCell') == port['cell']
        ):
            # This source-bound dressing follows its substrate. A finite
            # assumed pressure effect is independent of permanent bipolar seal.
            port['compressionFraction'] = max(port['compressionFraction'], 0.7)


def vessel_load(engine, edge, row):
    """Engineering tether traction at a mapped face; transection uses normal port accounting."""
    from . import surgical_vessel_rods as rods

    if retained_vessel_segments(engine) or rods.enabled(engine):
        return _source_segment_load(engine, edge, row)
    p = engine.scene.properties[PROPERTY]
    a, b = edge['sourceFace']
    deformation = engine.state.get('deformation', {})
    rest = engine.scene.points([a, b])
    span = rest[1] - rest[0]
    actual = span + np.subtract(
        deformation.get(str(b), [0.0, 0.0, 0.0]), deformation.get(str(a), [0.0, 0.0, 0.0])
    )
    strain = max(0.0, float(np.linalg.norm(actual) / np.linalg.norm(span) - 1))
    row['axialStrain'] = strain
    row['tractionN'] = p['axialStiffnessN'][edge['kind']] * strain if row['cutFace'] is None else 0.0
    row['peakAxialStrain'] = max(row.get('peakAxialStrain', 0.0), strain)
    if strain >= p['tractionFailureStrain'] and row['cutFace'] is None:
        from .surgical_vascular_graph import damage

        damage(engine, [edge['sourceFace']])
        row['tractionN'] = 0.0
        row['injuryCause'] = 'declared-tether-stretch'


def _source_segment_load(engine, edge, row):
    """Evaluate each surviving oriented bond of the declared source chain."""
    from . import surgical_vessel_rods as rods

    p = engine.scene.properties[PROPERTY]
    cells = oriented_source_cells(edge)
    removed = set(engine.state['removed'])
    cut_face = set(row['cutFace'] or ())
    blocked = {tuple(sorted(pair)) for name in ('cutEdges', 'transectedEdges')
               for pair in engine.state.get(name, ())} if rods.enabled(engine) else set()
    pairs = [(a, b) for a, b in zip(cells, cells[1:]) if a not in removed and b not in removed
             and {a, b} != cut_face and tuple(sorted((a, b))) not in blocked]
    strain = 0.0
    loaded_face = None
    resistance = 1.0
    if pairs:
        deformation = engine.state.get('deformation', {})
        sources = sorted({cell for pair in pairs for cell in pair})
        indices = {cell: i for i, cell in enumerate(sources)}
        a, b = np.asarray([[indices[cell] for cell in pair] for pair in pairs]).T
        rest = (np.array(list(rods.source_points(engine, sources, rest=True).values()))
                if rods.enabled(engine) else engine.scene.points(sources))
        current = rest + np.asarray([deformation.get(str(cell), [0., 0., 0.]) for cell in sources])
        spans = np.linalg.norm(rest[b] - rest[a], axis=1)
        extension = np.maximum(0., np.linalg.norm(current[b] - current[a], axis=1) / spans - 1)
        peak = int(extension.argmax())
        strain = float(extension[peak])
        loaded_face = list(pairs[peak]) if strain else None
        resistance = float(spans @ (1 + extension) ** 3 / spans.sum())
    row.update(axialStrain=strain, loadedFace=loaded_face, stretchResistanceRatio=resistance,
               tractionN=p['axialStiffnessN'][edge['kind']] * strain if row['cutFace'] is None else 0.0)
    if strain > row.get('peakAxialStrain', 0.) or 'peakAxialStrain' not in row:
        row['peakAxialStrain'] = strain
        row['peakLoadedFace'] = loaded_face
    if strain >= p['tractionFailureStrain'] and row['cutFace'] is None:
        from .surgical_vascular_graph import damage
        from .surgical_effects import _record_edges

        damage(engine, [loaded_face])
        _record_edges(engine, 'transectedEdges', [tuple(sorted(loaded_face))])
        row['tractionN'] = 0.0
        row['injuryCause'] = 'declared-tether-stretch'
