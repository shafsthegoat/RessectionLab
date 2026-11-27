"""Operative rehearsal anatomy on any prepared, segmented head study with a tumour layer.

The host is a prepared study whose layers declare scalp or skin, skull, brain
and a tumour, from any source: acquired, estimated, supplied or generated. This
module resamples that study once into a working grid whose +k axis is the
outward normal at the chosen entry (the head as positioned with the operative
site uppermost), adds one assumed dural layer and an assumed epidural plane
under the craniotomy, and declares the scalp, bone and dural bodies of the
opening.

The ``hostRehearsalV1`` declaration records provenance: which layers are
estimated, supplied, generated or assumed, and what stays unknown. It grants no
permission. Missing evidence is reported by name, never invented; only the dura
and the epidural plane are assumed, and they are labelled so. Subarachnoid CSF
is free space with a recorded membership, and tract bundles are named
memberships of white matter. No function, perfusion or vascular completeness is
implied. Coordinates are RAS millimetres throughout.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from scipy import ndimage

from .core import array_digest, semantic_digest

PROPERTY = 'hostRehearsalV1'
SCHEMA = 'resectionlab.host-rehearsal.v1'
SCOPE = ('Rehearsal on a prepared head study. Tissue boundaries carry their source provenance, the dura and '
         'the epidural plane are assumed, and function, perfusion and vascular completeness are unknown. '
         'Not a patient plan.')
# assumed-research: the dura is stripped from the inner table under and around
# the planned craniotomy (burr-hole dissection and the craniotome footplate)
# before the guarded cutter passes. Two 1 mm cells are the smallest plane that
# keeps the footplate clear of both the inner table and the dura where the
# inner table steps by one cell along the kerf.
EPIDURAL_PLANE_MM = 2.
# assumed-research: the two-cell plane is an annulus from 2.5 mm inside to
# 6.5 mm outside the bone flap radius, covering the burr hole, the kerf and the
# footplate.
EPIDURAL_MARGIN_MM = 6.5
EPIDURAL_INNER_MM = 2.5
# assumed-research: inside the annulus the dura is stripped from the flap's
# inner table by one 1 mm cell, as a dissector frees it before the flap is
# lifted, so dural sutures stay clear of the replaced flap (a suture sits 0.5 mm
# outside the dura with a 0.45 mm radius). The cortex moves down by that cell.
EPIDURAL_INTERIOR_MM = 1.
# The scalp flap extends this far beyond the bone flap radius.
SCALP_MARGIN_MM = 6.
# assumed-research: adult scalp is about 5-8 mm thick; the flap takes at most
# the outer 10 mm of the estimated skin and subcutaneous classes, whose
# 'subcutaneous' estimate also includes deep temporal muscle and fat.
SCALP_FLAP_DEPTH_MM = 10.
# assumed-research: a thick scalp flap is rolled back over its base rather than
# creased. The roll length is chosen so that its outer layers stay in front of
# the hinge at full reflection (bend length > thickness x angle); the dura is
# thin and folds over a short length.
SCALP_BEND_MM = 25.
# The scalp hinge lies this far beyond the bone flap radius, so the rolled-back
# scalp clears the craniotome and the opening on the hinge side.
SCALP_HINGE_MARGIN_MM = 12.
DURAL_BEND_MM = 6.
# Flap forceps grip the outer face this far inside the free rim of the flap.
GRIP_INSET_MM = 4.
# The pilot burr hole is centred this far outside the bone flap radius, so the
# 2 mm burr radius plus a cell never reaches the flap; its port is the bone
# within PORT_RADIUS_MM of the burr axis (the burr radius and half a cell).
PILOT_OFFSET_MM = 2.6
PORT_RADIUS_MM = 2.5
# The guarded cutter may widen its trough within this distance of the kerf
# ring, so the trailing guard post stays inside cut bone on a digital circle
# and the trough reaches outer bone lying below a thin flap edge.
BLADE_CORRIDOR_MM = 4.
# assumed-research: host rehearsals declare a faster CUSA V3 capacity than the
# 2 mm3/s default (see ``host_cusa_profile``).
HOST_CUSA_MM3_PER_SECOND = 40.

# Operative layers in ownership order; a later layer owns a shared cell.
# (name, material, display kind, colour, role, provenance); None takes the
# provenance declared by the source layers of that role.
LAYERS = (
    ('Scalp skin', 'soft_tissue', 'scalp', '#c99a8b', 'skin', None),
    ('Subcutaneous tissue and galea', 'soft_tissue', 'galea', '#b9a89c', 'subcutaneous', None),
    ('Other head tissue', 'soft_tissue', 'tissue', '#a88f84', 'extracranial', None),
    ('Skull cortical bone', 'bone', 'skull', '#d8cbb0', 'cortical_bone', None),
    ('Diploe', 'bone', 'diploe', '#a98f74', 'cancellous_bone', None),
    ('Dura (assumed)', 'membrane', 'dura', '#cdbcc2', 'dura', 'assumed'),
    ('Cerebral cortex', 'soft_tissue', 'grayMatter', '#c59a94', 'cortex', None),
    ('Cerebral white matter', 'soft_tissue', 'whiteMatter', '#d9d2bd', 'white', None),
    ('Deep gray matter', 'soft_tissue', 'grayMatter', '#b98f8a', 'deep', None),
    ('Cerebellum', 'soft_tissue', 'grayMatter', '#c0958f', 'cerebellum', None),
    ('Brainstem', 'soft_tissue', 'whiteMatter', '#cfc6b2', 'brainstem', None),
)
TUMOUR_LAYERS = (
    ('Tumour rim', 'tumor', 'tumor', '#b08a86', 'tumour_rim', None),
    ('Tumour core', 'tumor', 'tumor', '#8f6f73', 'tumour_core', None),
)
VESSEL_LAYERS = (
    ('Vein', 'vessel', 'vein', '#6f5c86', 'vein', None),
    ('Artery', 'vessel', 'artery', '#a94a4a', 'artery', None),
    ('Vessel (angiography)', 'vessel', 'artery', '#b05555', 'angiography_vessel', None),
)
_BRAIN_ROLES = ('cortex', 'white', 'deep', 'cerebellum', 'brainstem')
# Each required role and the plain name reported when the study lacks it.
REQUIRED_ANATOMY = (('skin', 'scalp or skin'), ('cortical_bone', 'skull'), ('cortex', 'cerebral cortex'),
                    ('white', 'cerebral white matter'))


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


def _provenance(values):
    values = sorted({str(value) for value in values if value})
    if not values:
        return 'unknown'
    return values[0] if len(values) == 1 else 'mixed: ' + ', '.join(values)


def host_roles(case):
    """Boolean host-grid masks by operative role, their source provenance and the tract bundles.

    Under a lesion that replaced host tissue in the active masks, the healthy
    source membership says which tissue it grew in.
    """
    layers = dict(case.metadata.get('anatomy_layers', {}))
    roles, tracts, sources, unclassified = {}, {}, {}, []
    for name, mask in case.compartments.items():
        layer = dict(layers.get(name, {}))
        role = host_role(name, layer)
        if role is None:
            if name != 'brain':
                unclassified.append(name)
            continue
        source = (case.source_compartments or {}).get(name, mask) if role in _BRAIN_ROLES else mask
        source = np.asarray(source, dtype=bool)
        sources.setdefault(role, []).append(layer.get('provenance'))
        if role == 'tract':
            tracts[str(layer.get('label', name)).split('\u00b7')[0].strip()] = source
        else:
            roles[role] = roles[role] | source if role in roles else source.copy()
    if unclassified:
        raise ValueError('HOST_REHEARSAL_UNCLASSIFIED_LAYER: ' + ', '.join(sorted(unclassified)))
    missing = [label for role, label in REQUIRED_ANATOMY if role not in roles or not roles[role].any()]
    if not any(roles.get(role, np.zeros(1, bool)).any() for role in ('tumour_rim', 'tumour_core')):
        missing.append('tumour')
    if missing:
        raise ValueError('HOST_REHEARSAL_MISSING_ANATOMY: the study declares no ' + ', '.join(missing) + ' layer')
    if 'tumour_rim' not in roles:
        # A single tumour layer is one lesion; it is drawn and counted as the rim.
        roles['tumour_rim'] = roles.pop('tumour_core')
        sources['tumour_rim'] = sources.pop('tumour_core')
    return roles, tracts, {role: _provenance(values) for role, values in sources.items()}


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


def working_frame(entry, *, spacing_mm=1., half_width_mm=75., depth_mm=110., above_mm=56.):
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


def fill_skull_gaps(role, shape, centre, *, radius_mm, spacing_mm, roof_mm=30.):
    """Close gaps in a thin skull estimate under the planned craniotomy.

    A column within ``radius_mm`` of the entry whose scalp lies on CSF or empty
    space with no bone between is a dropout of the bone estimate. Bone of the
    median local thickness (within 4 mm) is assigned below the scalp, taking
    only CSF or empty cells. CSF or empty pockets up to 3 mm long between bone
    cells of the calvarium are filled as well. Returns the filled mask; ``role``
    is updated.
    """
    bone = role['cortical_bone'] | role['cancellous_bone']
    scalp = role['skin'] | role['subcutaneous'] | role['extracranial']
    occupied = np.any([mask for name, mask in role.items() if name not in ('csf', 'air')], axis=0)
    floor = int(round(centre[2] - roof_mm / spacing_mm))
    ii, jj = np.ogrid[:shape[0], :shape[1]]
    zone = np.hypot(ii - centre[0], jj - centre[1]) * spacing_mm <= radius_mm
    upper = bone[:, :, floor:]
    has_bone = upper.any(axis=2)
    top = np.where(has_bone, upper.shape[2] - 1 - np.argmax(upper[:, :, ::-1], axis=2), -1)
    thickness = np.zeros(shape[:2], dtype=int)
    for a, b in zip(*np.nonzero(has_bone & zone)):
        k = top[a, b]
        while k >= 0 and upper[a, b, k]:
            k -= 1
        thickness[a, b] = top[a, b] - k
    filled = np.zeros(shape, dtype=bool)
    reach = int(round(4. / spacing_mm))
    for a, b in zip(*np.nonzero(zone & ~has_bone)):
        column = scalp[a, b, floor:]
        if not column.any():
            continue
        k = column.shape[0] - 1 - np.argmax(column[::-1])
        while k > 0 and column[k - 1]:
            k -= 1
        near = thickness[max(0, a - reach):a + reach + 1, max(0, b - reach):b + reach + 1]
        depth = int(np.median(near[near > 0])) if (near > 0).any() else 3
        for d in range(1, depth + 1):
            cell = (a, b, floor + k - d)
            if cell[2] < 0 or occupied[cell]:
                break
            filled[cell] = True
    # Pockets of CSF or empty space up to 3 mm long inside the calvarium of a
    # column (bone above and below) are diploe the estimate missed.
    longest = int(round(3. / spacing_mm))
    for a, b in zip(*np.nonzero(zone & has_bone)):
        column = upper[a, b]
        k = top[a, b]
        while k > 0:
            if column[k]:
                k -= 1
                continue
            low = k
            while low - 1 >= 0 and not column[low - 1] and k - low + 1 <= longest:
                low -= 1
            if low - 1 >= 0 and column[low - 1] and k - low + 1 <= longest and not occupied[a, b, floor + low:floor + k + 1].any():
                filled[a, b, floor + low:floor + k + 1] = True
                k = low - 1
                continue
            break
    role['cortical_bone'] = role['cortical_bone'] | filled
    role['csf'] = role['csf'] & ~filled
    return filled


def working_anatomy(case, roles, tracts, shape, affine, *, fill_centre=None, fill_radius_mm=0.):
    """Sample the host once into the working grid and derive the assumed dura.

    With ``fill_centre``, gaps in the skull estimate within ``fill_radius_mm``
    are closed first (see fill_skull_gaps) and returned as the fourth value.
    """
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
        membership['Peritumoural oedema'] = _resample(
            roles['oedema'].astype(np.uint8), host_affine, shape, affine, 0).astype(bool)
    image = _resample(np.asarray(case.mri, dtype=np.float32), host_affine, shape, affine, 1).astype(np.float32)
    filled = np.zeros(shape, dtype=bool)
    if fill_centre is not None:
        filled = fill_skull_gaps(role, shape, fill_centre, radius_mm=fill_radius_mm,
                                 spacing_mm=float(np.linalg.norm(affine[:3, 0])))
    bone = role['cortical_bone'] | role['cancellous_bone']
    intracranial = role['csf'] | np.any([role[name] for name in _BRAIN_ROLES], axis=0)
    # The estimated CSF shell contains the meninges. Its layer against the
    # inner table is assigned to an assumed dura one cell thick.
    # Edge neighbours are included so the sheet stays face-connected where the
    # inner table steps across the grid.
    against_bone = ndimage.binary_dilation(bone, structure=ndimage.generate_binary_structure(3, 2))
    # CSF or brain estimated outside the calvarium (above the outermost bone
    # of a column near the operative site) is not intracranial.
    k = np.arange(shape[2])[None, None, :]
    upper = bone & (k >= shape[2] - 1 - int(round(80. / float(np.linalg.norm(affine[:3, 2])))))
    top = np.where(upper.any(axis=2), shape[2] - 1 - np.argmax(upper[:, :, ::-1], axis=2), -1)[:, :, None]
    role['dura'] = intracranial & against_bone & (k < top)
    for name in ('csf', *_BRAIN_ROLES):
        role[name] = role[name] & ~role['dura']
    if fill_centre is not None:
        return role, membership, image, filled
    return role, membership, image


def _disc(shape, centre, radius, *, offset=0., chord=None):
    i, j = np.ogrid[:shape[0], :shape[1]]
    inside = (i - centre[0]) ** 2 + (j - centre[1] - offset) ** 2 <= radius ** 2
    if chord is not None:
        inside &= (j - centre[1] - offset) <= chord
    return inside[:, :, None]


def _horseshoe(shape, centre, radius, *, offset, hinge):
    """A flap outline: a far semicircle and straight sides running to the hinge (+j).

    Straight sides along the grid keep each side cut parallel to the motion of
    a flap folding about a hinge across +j, so no cell crosses its own cut.
    """
    i, j = np.ogrid[:shape[0], :shape[1]]
    di, dj = i - centre[0], j - centre[1] - offset
    inside = ((di ** 2 + dj ** 2 <= radius ** 2) & (dj <= 0)) | ((np.abs(di) <= radius) & (dj > 0) & (dj <= hinge))
    return inside[:, :, None]


def _scalp_flap(role, shape, centre, *, spacing_mm, bone_radius_mm, scalp_margin_mm=SCALP_MARGIN_MM):
    """Scalp flap footprint: the horseshoe, above the roof, outside any other tissue of its column.

    Only the scalp run from the skin surface down to the first other tissue
    belongs to the flap; scalp estimated beneath bone or muscle stays.
    """
    scale = 1. / spacing_mm
    k = np.arange(shape[2])[None, None, :]
    # Deep enough that the flap never ends on a horizontal cut a vertical blade cannot make.
    roof = k >= centre[2] - 50. * scale
    other = np.zeros(shape, dtype=bool)
    for name, mask in role.items():
        if name not in ('skin', 'subcutaneous', 'csf', 'air'):
            other |= mask
    other &= roof
    top = np.where(other.any(axis=2), shape[2] - 1 - np.argmax(other[:, :, ::-1], axis=2), -1)
    # The estimated subcutaneous class also holds deep muscle and fat; only the
    # outer SCALP_FLAP_DEPTH_MM below the skin surface is raised with the flap.
    scalp = role['skin'] | role['subcutaneous']
    skin = np.where(scalp.any(axis=2), shape[2] - 1 - np.argmax(scalp[:, :, ::-1], axis=2), -1)
    floor = np.maximum(top, skin - int(round(SCALP_FLAP_DEPTH_MM * scale)))
    return roof & (k > floor[:, :, None]) & _horseshoe(
        shape, centre, (bone_radius_mm + scalp_margin_mm) * scale, offset=2. * scale,
        hinge=(bone_radius_mm + SCALP_HINGE_MARGIN_MM) * scale)


def subgaleal_plane(role, shape, centre, *, spacing_mm, bone_radius_mm):
    """Assumed dissection plane under the scalp flap.

    The scalp flap is elevated in the loose areolar plane before it is
    reflected: the deepest flap cell of each column becomes free space, since a
    rigid voxel flap cannot otherwise leave a 1 mm bump of the calvarial
    estimate. Over the craniotomy and its kerf (within the epidural margin of
    the bone flap) any soft tissue left between the flap and the calvarium,
    such as deep temporal muscle estimated as subcutaneous tissue, is taken as
    elevated too, so the burr and the cutter meet bone only. Returns the
    plane; ``role`` is updated in place.
    """
    flap = _scalp_flap(role, shape, centre, spacing_mm=spacing_mm, bone_radius_mm=bone_radius_mm)
    scalp = (role['skin'] | role['subcutaneous']) & flap
    present = scalp.any(axis=2)
    deepest = np.argmax(scalp, axis=2)
    plane = np.zeros(shape, dtype=bool)
    i, j = np.nonzero(present)
    plane[i, j, deepest[i, j]] = True
    soft = role['skin'] | role['subcutaneous'] | role['extracranial']
    bone = role['cortical_bone'] | role['cancellous_bone']
    ii, jj = np.ogrid[:shape[0], :shape[1]]
    over = (np.hypot(ii - centre[0], jj - centre[1]) * spacing_mm <= bone_radius_mm + EPIDURAL_MARGIN_MM) & present
    k = np.arange(shape[2])[None, None, :]
    floor = np.where(present, deepest, shape[2])[:, :, None]
    # Highest bone cell of each column near the surface: everything between it and the flap floor.
    near = bone & (k >= centre[2] - 30. / spacing_mm)
    top = np.where(near.any(axis=2), shape[2] - 1 - np.argmax(near[:, :, ::-1], axis=2), shape[2])[:, :, None]
    plane |= soft & over[:, :, None] & (k > top) & (k < floor)
    for name in ('skin', 'subcutaneous', 'extracranial'):
        role[name] = role[name] & ~plane
    return plane


def opening_bodies(role, shape, centre, *, spacing_mm, bone_radius_mm=22., scalp_margin_mm=SCALP_MARGIN_MM):
    """Scalp, bone and dural bodies of one circular craniotomy under a horseshoe scalp flap.

    The scalp flap and the dural flap are horseshoes whose straight bases are
    the hinges toward +j; the bone flap is a disc.
    """
    scale = 1. / spacing_mm
    roof = np.arange(shape[2])[None, None, :] >= centre[2] - 30. * scale
    bone = role['cortical_bone'] | role['cancellous_bone']
    scalp = role['skin'] | role['subcutaneous']
    # The dural flap lies inside the epidural annulus, on undisturbed cortex.
    dural_radius = (bone_radius_mm - 7.) * scale
    members = {
        'scalp': scalp & _scalp_flap(role, shape, centre, spacing_mm=spacing_mm, bone_radius_mm=bone_radius_mm,
                                     scalp_margin_mm=scalp_margin_mm),
        'bone': bone & roof & _disc(shape, centre, bone_radius_mm * scale),
        'dura': role['dura'] & roof & _horseshoe(shape, centre, dural_radius, offset=0., hinge=10. * scale),
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


def scalp_bone_islands(role, shape, centre, *, radius_mm, spacing_mm, depth_mm=10., longest_mm=2.):
    """Reassign flecks of the bone estimate inside the scalp to the galea layer.

    Near the entry the working columns run along the outward normal. A run of
    at most ``longest_mm`` of bone with scalp immediately above and below it,
    within ``depth_mm`` of the skin surface, is a fleck of the bone estimate
    inside the scalp, not calvarium. It is reassigned to subcutaneous tissue and
    the mask is returned for the provenance record; ``role`` is updated in place.
    """
    bone = role['cortical_bone'] | role['cancellous_bone']
    scalp = role['skin'] | role['subcutaneous']
    islands = np.zeros(shape, dtype=bool)
    reach, longest = int(round(depth_mm / spacing_mm)), int(round(longest_mm / spacing_mm))
    ii, jj = np.ogrid[:shape[0], :shape[1]]
    i, j = np.nonzero(np.hypot(ii - centre[0], jj - centre[1]) * spacing_mm <= radius_mm)
    for a, b in zip(i, j):
        occupied = np.flatnonzero(scalp[a, b] | bone[a, b])
        if not len(occupied) or not scalp[a, b, occupied[-1]]:
            continue
        surface = occupied[-1]
        k = surface - 1
        while k > surface - reach:
            if bone[a, b, k]:
                low = k
                while low - 1 > surface - reach - longest and bone[a, b, low - 1]:
                    low -= 1
                if k - low + 1 <= longest and scalp[a, b, k + 1] and low - 1 >= 0 and scalp[a, b, low - 1]:
                    islands[a, b, low:k + 1] = True
                k = low - 1
            else:
                k -= 1
    for name in ('cortical_bone', 'cancellous_bone'):
        role[name] = role[name] & ~islands
    role['subcutaneous'] = role['subcutaneous'] | islands
    return islands


def epidural_plane(role, shape, centre, *, inner_mm, radius_mm, depth_mm, spacing_mm, interior_mm=0.):
    """Assumed stripped epidural plane under the craniotomy.

    Between ``inner_mm`` and ``radius_mm`` of the entry column (the burr hole,
    the kerf and the footplate's path) the soft intracranial cells within
    ``depth_mm`` of the locally deepest bone become free space and the dura is
    assigned to the next shell. Inside ``inner_mm`` (under the bone flap) the
    plane is ``interior_mm`` deep: the dura stripped from the inner table, so
    the sutured dura stays clear of the replaced flap. One-column dural walls
    at the outer boundary, and at the step between the two depths, keep the
    dura continuous with the dura against the inner table outside. Vessel and
    tumour cells are left as they are. Returns the plane; ``role`` is updated.
    """
    cells = int(round(depth_mm / spacing_mm))
    interior = int(round(interior_mm / spacing_mm))
    if cells <= 0:
        return np.zeros(shape, dtype=bool)
    bone = role['cortical_bone'] | role['cancellous_bone']
    i, j = np.ogrid[:shape[0], :shape[1]]
    distance = np.hypot(i - centre[0], j - centre[1]) * spacing_mm
    zone = ((distance >= inner_mm) & (distance <= radius_mm))[:, :, None]
    outer = zone & (distance > radius_mm - spacing_mm)[:, :, None]
    step = zone & (distance < inner_mm + spacing_mm)[:, :, None]
    inside = (distance < inner_mm)[:, :, None]
    disc = zone | inside
    soft = role['dura'] | role['csf'] | np.any([role[name] for name in _BRAIN_ROLES], axis=0)
    # Face steps keep the plane at ``cells`` in every column without thickening
    # it on oblique bone; the dural shell uses edge neighbours, as the dura does.
    # Where the inner table steps, the plane follows the deepest bone within the
    # reach of the cutter and its footplate (3 mm across), so neither meets a dural step.
    across = np.zeros((7, 7, 1), dtype=bool)
    across[:, :, 0] = np.hypot(*np.mgrid[-3:4, -3:4]) <= 3.
    spread = ndimage.binary_dilation(bone, across)
    face = ndimage.generate_binary_structure(3, 1)
    near = ndimage.binary_dilation(spread, face, iterations=cells) & soft & disc
    shallow = (ndimage.binary_dilation(spread, face, iterations=interior) & soft & disc if interior > 0
               else np.zeros(shape, dtype=bool))
    plane = (near & zone & ~outer & ~step) | (shallow & (inside | step))
    walls = (near & outer) | (near & step & ~shallow)
    occupied = np.where(inside, shallow, near)
    deep = (ndimage.binary_dilation(plane | bone, ndimage.generate_binary_structure(3, 2))
            & soft & (disc if interior > 0 else zone) & ~occupied)
    keep = ~disc if interior > 0 else ~zone
    role['dura'] = (role['dura'] & keep) | walls | deep
    for name in ('csf', *_BRAIN_ROLES):
        role[name] = role[name] & ~plane & ~role['dura']
    return plane


def host_cusa_profile():
    """The CUSA V3 profile declared by host rehearsals, with its assumed-research rate."""
    from .surgical_cusa_v3 import default_profile
    profile = default_profile()
    profile['capacityMm3PerSecond'] = HOST_CUSA_MM3_PER_SECOND
    profile['provenance'] = {'kind': 'assumed-research', 'description': (
        'Host rehearsal rate of 40 mm3/s. Clinical ultrasonic aspirators debulk soft glioma on the order of '
        'cm3 per minute, so the 2 mm3/s default would take about two simulated hours for a 14 cm3 lesion. '
        'Uncalibrated contact-capacity surrogate; not a device or patient calibration.')}
    return profile


def _source_class(host):
    data = host.get('data_case') or {}
    if data.get('classification'):
        return str(data['classification'])
    if host.get('is_synthetic') is True or host.get('generated') is True:
        return 'synthetic'
    return 'acquired-or-unverified'


def _rim_grip(member, shape, centre, rim_mm, spacing_mm, inset_mm=GRIP_INSET_MM):
    """Highest outer cell of the flap within 2 mm of a point ``inset_mm`` inside its free rim.

    The extreme rim of a digital horseshoe can be a single column standing
    below its retained neighbours, where closed jaws would touch them. Closing
    jaws from above meet the highest nearby cell first, so the declared grip
    is that cell and the registered grip follows the declared trajectory.
    """
    ci = int(round(centre[0]))
    cj = int(round(centre[1] - (rim_mm - inset_mm) / spacing_mm))
    reach = int(round(2. / spacing_mm))
    best = None
    for i in range(ci - reach, ci + reach + 1):
        for j in range(cj - reach, cj + reach + 1):
            if math.hypot(i - ci, j - cj) * spacing_mm > 2.:
                continue
            column = np.flatnonzero(member[i, j])
            if len(column) and (best is None or column[-1] > best[2]):
                best = (i, j, int(column[-1]))
    if best is None:
        raise ValueError('HOST_REHEARSAL_GRIP_WITHOUT_TISSUE')
    return int(np.ravel_multi_index(best, shape))


def burr_hole(role, shape, affine, centre, flap_cells, *, spacing_mm, bone_radius_mm):
    """Pilot burr hole beside the bone flap, on its lateral (+i) side, through the full local skull thickness.

    The hole is centred ``PILOT_OFFSET_MM`` outside the flap radius so the
    4 mm burr never touches the flap, on the side where the scalp opening
    extends furthest beyond the bone flap, and its port is every non-flap bone cell
    the burr can reach between the outer surface and 1 mm above the inner
    table, so the contact-limited burr can always clear what it touches.
    """
    from .geometry import GeometryScene, capsule_voxel_indices
    bone = role['cortical_bone'] | role['cancellous_bone']
    i = int(round(centre[0] + (bone_radius_mm + PILOT_OFFSET_MM) / spacing_mm))
    j = int(round(centre[1]))
    column = np.flatnonzero(bone[i, j])
    if not len(column):
        raise ValueError('HOST_REHEARSAL_PILOT_WITHOUT_BONE')
    top = bottom = int(column[-1])
    while bottom - 1 >= 0 and bone[i, j, bottom - 1]:
        bottom -= 1

    def world(k):
        return affine[:3, :3] @ np.array([i, j, k], dtype=float) + affine[:3, 3]
    cover = capsule_voxel_indices(GeometryScene(np.zeros(shape, dtype=bool), affine),
                                  world(top + 6. / spacing_mm), world(bottom + 1. / spacing_mm), PORT_RADIUS_MM)
    flap = np.zeros(shape, dtype=bool)
    flap.flat[flap_cells] = True
    keep = bone[tuple(cover.T)] & ~flap[tuple(cover.T)]
    port = np.ravel_multi_index(cover[keep].T, shape)
    return {'pilotCenterMm': world((top + bottom) / 2.).tolist(), 'portCells': sorted(map(int, port))}


def build_host_rehearsal(case, *, outward=None, spacing_mm=1., bone_radius_mm=22., half_width_mm=75.,
                         depth_mm=110., above_mm=56., epidural_mm=EPIDURAL_PLANE_MM):
    """Return the derived working-grid case, its region mapping and the operative landmarks.

    The study needs layers declaring scalp or skin, skull, cerebral cortex,
    white matter and a tumour; a missing one is named in the error.
    """
    from .core import CaseData, SourceRef, thaw_json
    from .surgical_physics_v2 import default_profile as physics_profile
    from .surgical_exposure_geometry import FOLD_REFERENCE, exposure_declaration
    from .surgical_bone_rest import rest_declaration
    from .surgical_scene import from_case

    host = thaw_json(case.metadata)
    roles, tracts, provenance = host_roles(case)
    entry = plan_entry(case, roles, outward=outward)
    # The grid reaches far enough above the entry to hold the reflected scalp roll.
    shape, affine, centre = working_frame(entry, spacing_mm=spacing_mm, half_width_mm=half_width_mm,
                                          depth_mm=depth_mm, above_mm=above_mm)
    role, membership, image, filled = working_anatomy(
        case, roles, tracts, shape, affine, fill_centre=centre,
        fill_radius_mm=bone_radius_mm + EPIDURAL_MARGIN_MM + 2.)
    islands = scalp_bone_islands(role, shape, centre, radius_mm=bone_radius_mm + 24.,
                                 spacing_mm=spacing_mm)
    plane = epidural_plane(role, shape, centre, inner_mm=bone_radius_mm - EPIDURAL_INNER_MM,
                           radius_mm=bone_radius_mm + EPIDURAL_MARGIN_MM, depth_mm=epidural_mm,
                           interior_mm=EPIDURAL_INTERIOR_MM,
                           spacing_mm=spacing_mm)
    subgaleal = subgaleal_plane(role, shape, centre, spacing_mm=spacing_mm, bone_radius_mm=bone_radius_mm)
    tract_layers = tuple((name + ' (tract)', 'soft_tissue', 'tract', '#d6c79a', name, provenance.get('tract'))
                         for name in sorted(tracts))
    rows = [*LAYERS, *tract_layers, *TUMOUR_LAYERS, *VESSEL_LAYERS]
    brain = np.any([role[name] for name in ('white', 'deep', 'cortex')], axis=0)
    compartments, kept = {}, []
    for name, material, kind, colour, source, declared in rows:
        mask = membership[source] & brain if source in membership else role.get(source)
        if mask is None or not mask.any():
            continue
        compartments[name] = mask
        kept.append((name, material, kind, colour, source, declared or provenance.get(source, 'unknown')))
    mapping = [dict(id=index, name=name, material=material, color=colour, target=material == 'tumor')
               for index, (name, material, _, colour, _, _) in enumerate(kept, 1)]
    labels = np.zeros(shape, dtype=np.uint16)
    for region in mapping:
        labels[compartments[region['name']]] = region['id']
    owner = {name: labels == region['id'] for region, (name, *_) in zip(mapping, kept)}
    # Each cell has one owner, so a source mask and its engine membership agree.
    compartments = dict(owner)
    absent = np.zeros(shape, dtype=bool)
    final_role = {'skin': owner['Scalp skin'], 'subcutaneous': owner.get('Subcutaneous tissue and galea', absent),
                  'cortical_bone': owner['Skull cortical bone'],
                  'cancellous_bone': owner.get('Diploe', absent), 'dura': owner['Dura (assumed)']}
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
    # The parked flap stays inside the working grid, above the lateral scalp.
    park = (entry_point + lateral * (bone_radius_mm + 26.) + outward_unit * 14.).tolist()
    # Field-material sites on cortex inside the dural opening, clear of the planned corticotomy.
    cottonoid, hemostat = surface_point(9., -9., brain_surface), surface_point(-9., -9., brain_surface)
    exposure = exposure_declaration(
        shape, affine, scalp=bodies['scalp'], bone=bodies['bone'], dura=bodies['dura'], bone_material=bone_layer,
        park_center=park, cottonoid_site=cottonoid, hemostat_site=hemostat, hinge_axis=lateral.tolist(),
        outward=outward_unit.tolist(), max_body_cells=60000)
    # The thick scalp is gripped 4 mm inside its rim. The thin dura is gripped at
    # its rim, so after it is folded back the rim is what a closing jaw meets first.
    for name, bend, rim, inset in (('scalp', SCALP_BEND_MM, bone_radius_mm + SCALP_MARGIN_MM - 2., GRIP_INSET_MM),
                                   ('dura', DURAL_BEND_MM, bone_radius_mm - 7., 0.)):
        exposure['bodies'][name].update(foldReference=FOLD_REFERENCE, bendLengthMm=bend,
                                        gripCell=_rim_grip(members[name], shape, centre, rim, spacing_mm, inset))
    bone_body = exposure['bodies']['bone']
    # The guarded cutter may widen the trough beside the kerf ring, never into the flap.
    ring = np.zeros(shape, dtype=bool)
    ring.flat[bone_body['kerfCells']] = True
    near_ring = ndimage.distance_transform_edt(~ring, sampling=spacing_mm) <= BLADE_CORRIDOR_MM
    corridor = near_ring & (final_role['cortical_bone'] | final_role['cancellous_bone'])
    corridor.flat[bone_body['cells']] = False
    bone_body['bladeCorridorCells'] = np.flatnonzero(corridor).tolist()
    bone_body['bladeCorridorReachMm'] = BLADE_CORRIDOR_MM + .05
    bone_body.update(burr_hole(final_role, shape, affine, centre, bone_body['cells'], spacing_mm=spacing_mm,
                               bone_radius_mm=bone_radius_mm))
    exposure['boneRest'] = rest_declaration(shape, affine, bone_body)
    exposure['instrumentGeometryRevision'] = 3
    # Beyond the craniotomy the curved scalp lies below the dural plane; the
    # access-depth test applies within the bone opening only.
    exposure['accessDepthFootprintMm'] = bone_radius_mm + 2.
    exposure['exterior'] = {'normalRas': outward_unit.tolist(),
                            'pointRasMm': (entry_point + outward_unit * 4.).tolist()}
    assumed = ['Dura (assumed)']
    declaration = {
        'schema': SCHEMA, 'sourceClass': _source_class(host), 'role': (host.get('split') or {}).get('role'),
        'scope': SCOPE, 'hostCaseId': case.case_id, 'hostCaseHash': case.semantic_hash, 'hostRevision': case.revision,
        'workingGrid': {'spacingMm': spacing_mm, 'shape': list(shape), 'sampling': 'nearest host cell',
                        'frame': 'RAS+ millimetres; +k is the outward normal at the entry'},
        'entry': entry,
        'layers': [{'name': name, 'role': source, 'provenance': value} for name, _, _, _, source, value in kept],
        'assumedLayers': assumed,
        'freeSpaceMemberships': {'Subarachnoid and ventricular CSF (estimated)': array_digest(role['csf']),
                                 'Epidural plane under the craniotomy (assumed)': array_digest(plane),
                                 'Subgaleal plane under the scalp flap (assumed)': array_digest(subgaleal)},
        'overlayMemberships': {**{name: array_digest(mask) for name, mask in sorted(membership.items())
                                  if name not in tracts},
                               'Bone estimate inside the scalp, reassigned to galea': array_digest(islands),
                               'Gaps in the skull estimate under the craniotomy, filled (assumed)':
                                   array_digest(filled)},
        'unknown': ['patient function', 'perfusion and vascular completeness', 'dural and bridging-vein anatomy',
                    'dural adherence to the inner table', 'tissue mechanics calibration',
                    'mass effect of the tumour'],
    }
    properties = {'tissuePhysicsV2': physics_profile(), 'cusaV3': host_cusa_profile(),
                  'operativeExposureV1': exposure, PROPERTY: declaration}
    layers_metadata = {name: {'kind': kind, 'color': colour, 'provenance': value,
                              'review_status': 'assumed' if value == 'assumed' else 'unreviewed'}
                       for name, _, kind, colour, _, value in kept}
    metadata = {
        'patient_group': host.get('patient_group'), 'patient_id': host.get('patient_id'),
        'split': host.get('split'), 'training_admitted': False, 'is_synthetic': host.get('is_synthetic'),
        'selected_modality': host.get('selected_modality'),
        'display_label': 'Operative rehearsal · ' + str(host.get('display_label', case.case_id)),
        'anatomy_layers': layers_metadata,
        'surgical_adapter': {'type': 'host-rehearsal-derivative', 'ownershipPrecedence': [name for name, *_ in kept],
                             'materialByRegion': {region['name']: region['material'] for region in mapping}},
        'surgical_properties': properties,
        'data_case': {**dict(host.get('data_case') or {}), 'derived_from_case_hash': case.semantic_hash},
    }
    references = (*case.source_refs, SourceRef('host-rehearsal-working-grid', 'derived://host-rehearsal/v2/'
                                                + case.semantic_hash.removeprefix('sha256:'), provenance='estimated'))
    derived = CaseData(case_id=case.case_id + '-OPERATIVE-REHEARSAL-v2', revision=1, affine=affine, mri=image,
                       compartments=compartments, source_refs=references,
                       unknowns=tuple(sorted({*case.unknowns, 'dura_assumed', 'epidural_plane_assumed',
                                              'subgaleal_plane_assumed', 'skull_estimate_gaps_filled',
                                              'working_grid_resampled'})),
                       metadata=metadata)
    scene = from_case(derived, mapping)
    tumour = owner['Tumour rim'] | owner.get('Tumour core', absent)
    landmarks = {
        'caseHash': derived.semantic_hash, 'sceneId': scene.identity, 'sourceCaseId': derived.case_id,
        'entryCenterRasMm': entry['entryRasMm'], 'outwardRasUnit': outward_unit.tolist(),
        'lateralRasUnit': lateral.tolist(), 'towardHingeRasUnit': toward_hinge.tolist(),
        'targetRasMm': entry['targetRasMm'], 'tumourMm3': float(tumour.sum() * spacing_mm ** 3),
        'boneRadiusMm': bone_radius_mm, 'boneParkRasMm': park, 'pilotRasMm': bone_body['pilotCenterMm'],
        'cottonoidSiteRasMm': cottonoid, 'hemostatSiteRasMm': hemostat,
        'corticalContactRasMm': surface_point(0., 0., brain_surface),
        'shape': list(shape), 'affine': affine.tolist(),
        'bodyCells': {name: int(mask.sum()) for name, mask in members.items()},
        'epiduralPlaneCells': int(plane.sum()),
    }
    return derived, scene, landmarks


def declaration(scene):
    """Validated host provenance declaration of a scene, or None. It grants no permission."""
    value = scene.properties.get(PROPERTY)
    if value is None:
        return None
    required = {'schema', 'sourceClass', 'role', 'scope', 'hostCaseId', 'hostCaseHash', 'hostRevision',
                'workingGrid', 'entry', 'layers', 'assumedLayers', 'freeSpaceMemberships',
                'overlayMemberships', 'unknown'}
    if (not isinstance(value, dict) or set(value) != required or value['schema'] != SCHEMA
            or not isinstance(value['sourceClass'], str) or not value['sourceClass']
            or not (value['role'] is None or isinstance(value['role'], str))
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
