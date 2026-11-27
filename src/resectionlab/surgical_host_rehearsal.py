"""Operative rehearsal anatomy on an acquired host head with a generated tumour.

The host is a prepared, segmented study: an acquired T1, a model-estimated head
segmentation, estimated vessels, published tract bundles and an explicit
generated tumour. This module resamples that study once into a working grid
whose +k axis is the outward normal at the chosen entry (the head as positioned
with the operative site uppermost), adds one assumed dural layer, and declares
the scalp, bone and dural bodies of the opening.

Everything here is DEVELOPMENT rehearsal anatomy. Estimates stay estimates, the
dura is assumed, subarachnoid CSF is free space with a recorded membership, and
tract bundles are named memberships of white matter. No function, perfusion or
vascular completeness is implied. Coordinates are RAS millimetres throughout.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy import ndimage

from .core import array_digest, semantic_digest

PROPERTY = 'hostRehearsalV1'
SCHEMA = 'resectionlab.host-rehearsal.v1'
SOURCE_CLASS = 'synthetic-scenario-on-acquired-host'
SCOPE = ('Development rehearsal on an acquired healthy host with a generated tumour. Tissue boundaries are '
         'unreviewed model estimates, the dura is an assumed layer, and function, perfusion and vascular '
         'completeness are unknown. Not a patient plan.')

# Operative layers in ownership order; a later layer owns a shared cell.
# (name, material, display kind, colour, role, provenance)
LAYERS = (
    ('Scalp skin', 'soft_tissue', 'scalp', '#c99a8b', 'skin', 'estimated'),
    ('Subcutaneous tissue and galea', 'soft_tissue', 'galea', '#b9a89c', 'subcutaneous', 'estimated'),
    ('Other head tissue', 'soft_tissue', 'tissue', '#a88f84', 'extracranial', 'estimated'),
    ('Skull cortical bone', 'bone', 'skull', '#d8cbb0', 'cortical_bone', 'estimated'),
    ('Diploe', 'bone', 'diploe', '#a98f74', 'cancellous_bone', 'estimated'),
    ('Dura (assumed)', 'membrane', 'dura', '#cdbcc2', 'dura', 'assumed'),
    ('Cerebral cortex', 'soft_tissue', 'grayMatter', '#c59a94', 'cortex', 'estimated'),
    ('Cerebral white matter', 'soft_tissue', 'whiteMatter', '#d9d2bd', 'white', 'estimated'),
    ('Deep gray matter', 'soft_tissue', 'grayMatter', '#b98f8a', 'deep', 'estimated'),
    ('Cerebellum', 'soft_tissue', 'grayMatter', '#c0958f', 'cerebellum', 'estimated'),
    ('Brainstem', 'soft_tissue', 'whiteMatter', '#cfc6b2', 'brainstem', 'estimated'),
)
TUMOUR_LAYERS = (
    ('Tumour rim (generated)', 'tumor', 'tumor', '#b08a86', 'tumour_rim', 'simulated'),
    ('Tumour core (generated)', 'tumor', 'tumor', '#8f6f73', 'tumour_core', 'simulated'),
)
VESSEL_LAYERS = (
    ('Vein (estimated)', 'vessel', 'vein', '#6f5c86', 'vein', 'estimated'),
    ('Artery (estimated)', 'vessel', 'artery', '#a94a4a', 'artery', 'estimated'),
    ('Vessel (angiography estimate)', 'vessel', 'artery', '#b05555', 'angiography_vessel', 'estimated'),
)
_BRAIN_ROLES = ('cortex', 'white', 'deep', 'cerebellum', 'brainstem')


def host_role(name, layer):
    """Classify one host layer by its declared kind and label. Unknown layers return None."""
    kind, label = layer.get('kind'), str(layer.get('label', name))
    if kind == 'tumor':
        return 'tumour_core' if 'core' in label.lower() else 'tumour_rim'
    if kind == 'edema':
        return 'oedema'
    if kind == 'tract':
        return 'tract'
    if kind == 'skull':
        return 'cancellous_bone' if 'Cancellous' in label else 'cortical_bone'
    if kind == 'vessels':
        if 'MRA' in label or 'vesselFM' in label:
            return 'angiography_vessel'
        return 'vein' if 'Vein' in label else 'artery'
    if kind == 'brain':
        if name == 'brain':
            return None
        if 'Cerebral-Cortex' in label:
            return 'cortex'
        if 'Cerebral-White-Matter' in label or 'WM-hypointensities' in label:
            return 'white'
        if 'Cerebellum' in label:
            return 'cerebellum'
        if 'Brain-Stem' in label:
            return 'brainstem'
        return 'deep'
    if kind == 'context':
        if 'CSF' in label or 'Vent' in label:
            return 'csf'
        if 'Air' in label:
            return 'air'
        if 'Skin' in label:
            return 'skin'
        if 'Other-Tissues' in label:
            return 'subcutaneous'
        return 'extracranial'
    return None


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if vector.shape != (3,) or not np.isfinite(vector).all() or norm < 1e-9:
        raise ValueError('HOST_REHEARSAL_DIRECTION_REQUIRED')
    return vector / norm


def host_roles(case):
    """Boolean host-grid masks by operative role, using healthy memberships under the generated lesion."""
    layers = dict(case.metadata.get('anatomy_layers', {}))
    roles, tracts, unclassified = {}, {}, []
    for name, mask in case.compartments.items():
        role = host_role(name, dict(layers.get(name, {})))
        if role is None:
            if name != 'brain':
                unclassified.append(name)
            continue
        # The generated lesion replaced host tissue in the active masks. The
        # healthy membership says which tissue it grew in.
        source = case.source_compartments.get(name, mask) if role in _BRAIN_ROLES else mask
        source = np.asarray(source, dtype=bool)
        if role == 'tract':
            tracts[str(layers[name].get('label', name)).split('·')[0].strip()] = source
        else:
            roles[role] = roles[role] | source if role in roles else source.copy()
    if unclassified:
        raise ValueError('HOST_REHEARSAL_UNCLASSIFIED_LAYER: ' + ', '.join(sorted(unclassified)))
    for required in ('skin', 'cortical_bone', 'cortex', 'white', 'tumour_rim'):
        if required not in roles or not roles[required].any():
            raise ValueError('HOST_REHEARSAL_MISSING_ANATOMY: ' + required)
    return roles, tracts


def plan_entry(case, roles, *, outward=None, samples=4000, minimum_superior=0.15):
    """Choose the shortest convexity corridor from the tumour centroid, or validate a supplied outward direction.

    A convexity direction has a superior component of at least ``minimum_superior``,
    which keeps the skull base, face and neck out of the search. This is a
    geometric default for rehearsal, not an approach recommendation.
    """
    affine = np.asarray(case.affine, dtype=float)
    if case.frame == 'LPS+':
        affine = np.diag([-1., -1., 1., 1.]) @ affine
    inverse = np.linalg.inv(affine)
    tumour = roles['tumour_rim'] | roles.get('tumour_core', False)
    centroid = (np.argwhere(tumour).mean(axis=0) @ affine[:3, :3].T) + affine[:3, 3]
    head = np.zeros(tumour.shape, dtype=bool)
    for name, mask in roles.items():
        if name != 'air':
            head |= mask
    head = ndimage.binary_fill_holes(head)
    step = float(np.linalg.norm(affine[:3, :3], axis=0).min()) / 2.

    def exit_length(direction):
        distance = 0.
        while distance < 200.:
            index = np.rint(inverse[:3, :3] @ (centroid + direction * distance) + inverse[:3, 3]).astype(int)
            if (index < 0).any() or (index >= head.shape).any() or not head[tuple(index)]:
                return distance
            distance += step
        return None

    if outward is not None:
        direction = _unit(outward)
        length = exit_length(direction)
        if length is None:
            raise ValueError('HOST_REHEARSAL_ENTRY_NOT_FOUND')
    else:
        rng = np.random.default_rng(20261127)
        candidates = rng.normal(size=(samples, 3))
        candidates /= np.linalg.norm(candidates, axis=1)[:, None]
        candidates = candidates[candidates[:, 2] >= minimum_superior]
        lengths = [(exit_length(d), i) for i, d in enumerate(candidates)]
        lengths = [(length, i) for length, i in lengths if length is not None]
        if not lengths:
            raise ValueError('HOST_REHEARSAL_ENTRY_NOT_FOUND')
        length, index = min(lengths)
        direction = candidates[index]
    return {'targetRasMm': centroid.tolist(), 'outwardRasUnit': direction.tolist(),
            'entryRasMm': (centroid + direction * length).tolist(), 'depthMm': float(length)}


def working_frame(entry, *, spacing_mm=1., half_width_mm=75., depth_mm=110., above_mm=24.):
    """Table-aligned working grid: +k outward, +j toward the inferior flap hinge, +i completing a right hand."""
    outward = _unit(entry['outwardRasUnit'])
    superior = np.array([0., 0., 1.])
    tangent = superior - outward * (superior @ outward)
    if np.linalg.norm(tangent) < .2:
        # At the vertex the flap base faces posteriorly instead.
        anterior = np.array([0., 1., 0.])
        tangent = anterior - outward * (anterior @ outward)
    toward_hinge = -_unit(tangent)
    lateral = np.cross(toward_hinge, outward)
    rotation = np.column_stack((lateral, toward_hinge, outward))
    shape = (int(round(2 * half_width_mm / spacing_mm)) + 1, int(round(2 * half_width_mm / spacing_mm)) + 1,
             int(round((depth_mm + above_mm) / spacing_mm)) + 1)
    centre = np.array([(shape[0] - 1) / 2., (shape[1] - 1) / 2., round(depth_mm / spacing_mm)])
    affine = np.eye(4)
    affine[:3, :3] = rotation * spacing_mm
    affine[:3, 3] = np.asarray(entry['entryRasMm'], dtype=float) - affine[:3, :3] @ centre
    return shape, affine, centre


def _resample(volume, host_affine, shape, affine, order):
    grid = np.indices(shape, dtype=np.float64).reshape(3, -1)
    transform = np.linalg.inv(host_affine) @ affine
    coordinates = transform[:3, :3] @ grid + transform[:3, 3:4]
    return ndimage.map_coordinates(volume, coordinates, order=order, mode='constant', cval=0).reshape(shape)


def working_anatomy(case, roles, tracts, shape, affine):
    """Sample the host once into the working grid and derive the assumed dura."""
    host_affine = np.asarray(case.affine, dtype=float)
    if case.frame == 'LPS+':
        host_affine = np.diag([-1., -1., 1., 1.]) @ host_affine
    # One categorical host volume keeps a single nearest-neighbour sample per cell.
    order = ['air', 'csf', 'extracranial', 'subcutaneous', 'skin', 'cancellous_bone', 'cortical_bone',
             'deep', 'cerebellum', 'brainstem', 'white', 'cortex', 'tumour_rim', 'tumour_core',
             'vein', 'artery', 'angiography_vessel']
    code = {name: index + 1 for index, name in enumerate(order)}
    categorical = np.zeros(case.mri.shape, dtype=np.uint8)
    for name in order:
        if name in roles:
            categorical[roles[name]] = code[name]
    sampled = _resample(categorical, host_affine, shape, affine, 0)
    role = {name: sampled == value for name, value in code.items()}
    membership = {name: _resample(mask.astype(np.uint8), host_affine, shape, affine, 0).astype(bool)
                  for name, mask in tracts.items()}
    if 'oedema' in roles:
        membership['Peritumoural oedema (generated)'] = _resample(
            roles['oedema'].astype(np.uint8), host_affine, shape, affine, 0).astype(bool)
    image = _resample(np.asarray(case.mri, dtype=np.float32), host_affine, shape, affine, 1).astype(np.float32)
    bone = role['cortical_bone'] | role['cancellous_bone']
    intracranial = role['csf'] | np.any([role[name] for name in _BRAIN_ROLES], axis=0)
    # The estimated CSF shell contains the meninges. Its layer against the
    # inner table is assigned to an assumed dura one cell thick.
    # Edge neighbours are included so the sheet stays face-connected where the
    # inner table steps across the grid.
    against_bone = ndimage.binary_dilation(bone, structure=ndimage.generate_binary_structure(3, 2))
    role['dura'] = intracranial & against_bone
    for name in ('csf', *_BRAIN_ROLES):
        role[name] = role[name] & ~role['dura']
    return role, membership, image


def _disc(shape, centre, radius, *, offset=0., chord=None):
    i, j = np.ogrid[:shape[0], :shape[1]]
    inside = (i - centre[0]) ** 2 + (j - centre[1] - offset) ** 2 <= radius ** 2
    if chord is not None:
        inside &= (j - centre[1] - offset) <= chord
    return inside[:, :, None]


def opening_bodies(role, shape, centre, *, spacing_mm, bone_radius_mm=22., scalp_margin_mm=12.):
    """Scalp, bone and dural bodies of one circular craniotomy under a hinged scalp flap."""
    scale = 1. / spacing_mm
    roof = np.arange(shape[2])[None, None, :] >= centre[2] - 30. * scale
    bone = role['cortical_bone'] | role['cancellous_bone']
    scalp = role['skin'] | role['subcutaneous']
    scalp_radius = (bone_radius_mm + scalp_margin_mm) * scale
    members = {
        'scalp': scalp & roof & _disc(shape, centre, scalp_radius, offset=3. * scale, chord=.80 * scalp_radius),
        'bone': bone & roof & _disc(shape, centre, bone_radius_mm * scale),
        'dura': role['dura'] & roof & _disc(shape, centre, (bone_radius_mm - 3.) * scale,
                                              chord=.62 * (bone_radius_mm - 3.) * scale),
    }
    layers = {'scalp': scalp, 'bone': bone, 'dura': role['dura']}
    for name, mask in members.items():
        # A body is one face-connected piece of its layer under the opening.
        labelled, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
        if not count:
            raise ValueError('HOST_REHEARSAL_EMPTY_BODY: ' + name)
        sizes = ndimage.sum_labels(mask, labelled, index=np.arange(1, count + 1))
        members[name] = labelled == int(np.argmax(sizes)) + 1
    return ({name: {'cells': np.flatnonzero(mask).tolist(), 'layerCells': np.flatnonzero(layers[name]).tolist()}
             for name, mask in members.items()}, members)


def build_host_rehearsal(case, *, outward=None, spacing_mm=1., bone_radius_mm=22., half_width_mm=75.,
                         depth_mm=110.):
    """Return the derived working-grid case, its region mapping and the operative landmarks."""
    from dataclasses import replace
    from .core import CaseData, SourceRef, thaw_json
    from .surgical_cusa_v3 import default_profile as cusa_profile
    from .surgical_physics_v2 import default_profile as physics_profile
    from .surgical_exposure_geometry import exposure_declaration
    from .surgical_bone_rest import rest_declaration
    from .surgical_scene import from_case

    host = thaw_json(case.metadata)
    if host.get('is_synthetic') is not True or host.get('split', {}).get('role') != 'DEVELOPMENT':
        raise ValueError('HOST_REHEARSAL_REQUIRES_DEVELOPMENT_SYNTHETIC_SCENARIO')
    roles, tracts = host_roles(case)
    entry = plan_entry(case, roles, outward=outward)
    shape, affine, centre = working_frame(entry, spacing_mm=spacing_mm, half_width_mm=half_width_mm,
                                          depth_mm=depth_mm)
    role, membership, image = working_anatomy(case, roles, tracts, shape, affine)
    tract_layers = tuple((name + ' (published tract)', 'soft_tissue', 'tract', '#d6c79a', name, 'estimated')
                         for name in sorted(tracts))
    rows = [*LAYERS, *tract_layers, *TUMOUR_LAYERS, *VESSEL_LAYERS]
    brain = np.any([role[name] for name in ('white', 'deep', 'cortex')], axis=0)
    compartments, kept = {}, []
    for name, material, kind, colour, source, provenance in rows:
        mask = membership[source] & brain if source in membership else role.get(source)
        if mask is None or not mask.any():
            continue
        compartments[name] = mask
        kept.append((name, material, kind, colour, source, provenance))
    mapping = [dict(id=index, name=name, material=material, color=colour, target=material == 'tumor')
               for index, (name, material, _, colour, _, _) in enumerate(kept, 1)]
    labels = np.zeros(shape, dtype=np.uint16)
    for region in mapping:
        labels[compartments[region['name']]] = region['id']
    owner = {name: labels == region['id'] for region, (name, *_) in zip(mapping, kept)}
    # Each cell has one owner, so a source mask and its engine membership agree.
    compartments = dict(owner)
    final_role = {'skin': owner['Scalp skin'], 'subcutaneous': owner['Subcutaneous tissue and galea'],
                  'cortical_bone': owner['Skull cortical bone'],
                  'cancellous_bone': owner.get('Diploe', np.zeros(shape, dtype=bool)), 'dura': owner['Dura (assumed)']}
    bodies, members = opening_bodies(final_role, shape, centre, spacing_mm=spacing_mm, bone_radius_mm=bone_radius_mm)
    bone_layer = bodies['bone']['layerCells']
    rotation = affine[:3, :3] / spacing_mm
    lateral, toward_hinge, outward_unit = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    entry_point = np.asarray(entry['entryRasMm'])

    def surface_point(offset_i, offset_j, mask):
        """Outermost cell of ``mask`` in the working column at a lateral offset from the entry."""
        i, j = int(round(centre[0] + offset_i / spacing_mm)), int(round(centre[1] + offset_j / spacing_mm))
        column = np.flatnonzero(mask[i, j])
        if not len(column):
            raise ValueError('HOST_REHEARSAL_LANDMARK_WITHOUT_TISSUE')
        return (affine[:3, :3] @ np.array([i, j, column[-1]]) + affine[:3, 3]).tolist()

    brain_surface = np.any([owner[name] for name, material, kind, *_ in kept
                            if kind in ('grayMatter', 'whiteMatter', 'tract', 'tumor')], axis=0)
    park = (entry_point + lateral * (bone_radius_mm + 40.) + outward_unit * 12.).tolist()
    exposure = exposure_declaration(
        shape, affine, scalp=bodies['scalp'], bone=bodies['bone'], dura=bodies['dura'], bone_material=bone_layer,
        park_center=park, cottonoid_site=surface_point(-6., -4., brain_surface),
        hemostat_site=surface_point(5., -4., brain_surface), hinge_axis=lateral.tolist(),
        outward=outward_unit.tolist(), max_body_cells=60000)
    exposure['boneRest'] = rest_declaration(shape, affine, exposure['bodies']['bone'])
    exposure['instrumentGeometryRevision'] = 2
    exposure['exterior'] = {'normalRas': outward_unit.tolist(),
                            'pointRasMm': (entry_point + outward_unit * 4.).tolist()}
    declaration = {
        'schema': SCHEMA, 'sourceClass': SOURCE_CLASS, 'role': 'DEVELOPMENT', 'scope': SCOPE,
        'hostCaseId': case.case_id, 'hostCaseHash': case.semantic_hash, 'hostRevision': case.revision,
        'workingGrid': {'spacingMm': spacing_mm, 'shape': list(shape), 'sampling': 'nearest host cell',
                        'frame': 'RAS+ millimetres; +k is the outward normal at the entry'},
        'entry': entry,
        'layers': [{'name': name, 'role': source, 'provenance': provenance} for name, _, _, _, source, provenance in kept],
        'assumedLayers': ['Dura (assumed)'],
        'freeSpaceMemberships': {'Subarachnoid and ventricular CSF (estimated)': array_digest(role['csf'])},
        'overlayMemberships': {name: array_digest(mask) for name, mask in sorted(membership.items())
                               if name not in tracts},
        'unknown': ['patient function', 'perfusion and vascular completeness', 'dural and bridging-vein anatomy',
                    'tissue mechanics calibration', 'mass effect of the generated tumour'],
    }
    properties = {'tissuePhysicsV2': physics_profile(), 'cusaV3': cusa_profile(),
                  'operativeExposureV1': exposure, PROPERTY: declaration}
    layers_metadata = {name: {'kind': kind, 'color': colour, 'provenance': provenance,
                              'review_status': 'assumed' if provenance == 'assumed' else 'unreviewed'}
                       for name, _, kind, colour, _, provenance in kept}
    metadata = {
        'patient_group': host.get('patient_group'), 'patient_id': host.get('patient_id'),
        'split': {'role': 'DEVELOPMENT'}, 'training_admitted': False, 'is_synthetic': True,
        'selected_modality': host.get('selected_modality'),
        'display_label': 'Operative rehearsal · ' + str(host.get('display_label', case.case_id)),
        'anatomy_layers': layers_metadata,
        'surgical_adapter': {'type': 'host-rehearsal-derivative', 'ownershipPrecedence': [name for name, *_ in kept],
                             'materialByRegion': {region['name']: region['material'] for region in mapping}},
        'surgical_properties': properties,
        'data_case': {**dict(host.get('data_case', {})), 'classification': SOURCE_CLASS,
                      'derived_from_case_hash': case.semantic_hash},
    }
    references = (*case.source_refs, SourceRef('host-rehearsal-working-grid', 'derived://host-rehearsal/v1/'
                                                + case.semantic_hash.removeprefix('sha256:'), provenance='estimated'))
    derived = CaseData(case_id=case.case_id + '-OPERATIVE-REHEARSAL-v1', revision=1, affine=affine, mri=image,
                       compartments=compartments, source_refs=references,
                       unknowns=tuple(sorted({*case.unknowns, 'dura_assumed', 'working_grid_resampled'})),
                       metadata=metadata)
    scene = from_case(derived, mapping)
    tumour = owner['Tumour rim (generated)'] | owner.get('Tumour core (generated)', False)
    landmarks = {
        'caseHash': derived.semantic_hash, 'sceneId': scene.identity, 'sourceCaseId': derived.case_id,
        'entryCenterRasMm': entry['entryRasMm'], 'outwardRasUnit': outward_unit.tolist(),
        'lateralRasUnit': lateral.tolist(), 'towardHingeRasUnit': toward_hinge.tolist(),
        'targetRasMm': entry['targetRasMm'], 'tumourMm3': float(tumour.sum() * spacing_mm ** 3),
        'boneRadiusMm': bone_radius_mm, 'boneParkRasMm': park,
        'corticalContactRasMm': surface_point(0., 0., brain_surface),
        'shape': list(shape), 'affine': affine.tolist(),
        'bodyCells': {name: int(mask.sum()) for name, mask in members.items()},
    }
    return derived, scene, landmarks


def declaration(scene):
    """Validated host-rehearsal declaration of a scene, or None."""
    value = scene.properties.get(PROPERTY)
    if value is None:
        return None
    required = {'schema', 'sourceClass', 'role', 'scope', 'hostCaseId', 'hostCaseHash', 'hostRevision',
                'workingGrid', 'entry', 'layers', 'assumedLayers', 'freeSpaceMemberships',
                'overlayMemberships', 'unknown'}
    if (not isinstance(value, dict) or set(value) != required or value['schema'] != SCHEMA
            or value['sourceClass'] != SOURCE_CLASS or value['role'] != 'DEVELOPMENT'
            or not isinstance(value['hostCaseHash'], str) or not value['hostCaseHash'].startswith('sha256:')
            or not value['assumedLayers'] or not value['unknown']):
        raise ValueError('HOST_REHEARSAL_INVALID_DECLARATION')
    names = {region['name'] for region in scene.regions}
    declared = {row['name'] for row in value['layers']}
    if declared != names or not set(value['assumedLayers']) <= names:
        raise ValueError('HOST_REHEARSAL_LAYER_DECLARATION_MISMATCH')
    return copy.deepcopy(value)


def identity(scene):
    value = declaration(scene)
    return None if value is None else semantic_digest(value)


def development_practice_source(case):
    """True only for a case that declares this rehearsal on a DEVELOPMENT host."""
    metadata = case.metadata
    source = metadata.get('data_case') or {}
    declared = (metadata.get('surgical_properties') or {}).get(PROPERTY) or {}
    return (declared.get('schema') == SCHEMA and declared.get('role') == 'DEVELOPMENT'
            and declared.get('sourceClass') == SOURCE_CLASS and source.get('role') == 'DEVELOPMENT'
            and source.get('classification') == SOURCE_CLASS and not source.get('split')
            and not source.get('family_id') and (metadata.get('split') or {}).get('role') == 'DEVELOPMENT'
            and metadata.get('training_admitted') is False)
