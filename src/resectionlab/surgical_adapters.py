"""Explicit, source-preserving import adapters for surgical simulation grids.

All output geometry is RAS+ millimeters. Source files are never modified.
Meshes retain their original topology alongside their declared grid approximation.
"""
from __future__ import annotations

import hashlib
import copy
from pathlib import Path
import struct
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
from .core import array_digest
from .surgical_scene import SurgicalScene, KINDS, MATERIALS


def _coordinates(frame, units):
    if frame not in ('RAS+', 'LPS+'):
        raise ValueError('Explicit RAS+ or LPS+ frame is required')
    if units not in ('mm', 'meter', 'micron'):
        raise ValueError('Explicit known units (mm, meter, micron) are required')
    scale = {'mm': 1., 'meter': 1000., 'micron': .001}[units]
    return np.diag([(-scale if frame == 'LPS+' else scale),
                    (-scale if frame == 'LPS+' else scale), scale, 1.])


def _file(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def _retain(scene, arrays):
    scene.source_arrays = {}
    for key, value in arrays.items():
        a = np.array(value, copy=True)
        a.flags.writeable = False
        scene.source_arrays[key] = a
    return scene


def _region(spec, index, name):
    r = dict(spec) if isinstance(spec, dict) else {'material': spec}
    r.setdefault('id', index)
    r.setdefault('name', str(name))
    r.setdefault('target', False)
    r.setdefault('color', '#92a7b8')
    if type(r['id']) is not int or not 0 < r['id'] < 65536:
        raise ValueError('Region IDs must be positive uint16 values')
    if r.get('material') not in MATERIALS:
        raise ValueError('Every region requires an explicit supported material')
    return r


def from_nifti(path, mapping, *, frame, units=None, properties=None):
    """Map arbitrary nonzero source labels to materials without changing the source.

    mapping is {source_label: material_or_region_dict}; region IDs default to 1..N.
    Unknown header units require an explicit units argument. Frame is mandatory.
    """
    im = nib.load(str(path))
    header_units = im.header.get_xyzt_units()[0]
    if units is not None and header_units != 'unknown' and units != header_units:
        raise ValueError('Declared units conflict with NIfTI header')
    units = units or header_units
    transform = _coordinates(frame, units)
    if not (int(im.header['sform_code']) or int(im.header['qform_code'])):
        raise ValueError('NIfTI requires an explicit coded spatial affine')
    stored = np.asanyarray(im.dataobj.get_unscaled())
    raw = np.asanyarray(im.dataobj)
    if raw.ndim != 3 or not raw.size or not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw != np.floor(raw)):
        raise ValueError('NIfTI must contain nonnegative integer categorical labels')
    present = set(np.unique(raw)) - {0}
    if set(mapping) != present:
        raise ValueError('Mapping must cover every nonzero source label exactly once')
    labels = np.zeros(raw.shape, np.uint16)
    regions = []
    for i, (value, spec) in enumerate(mapping.items(), 1):
        r = _region(spec, i, value)
        if type(r['id']) is not int or not 0 < r['id'] < 65536:
            raise ValueError('Region IDs must be positive uint16 values')
        regions.append(r)
        labels[raw == value] = r['id']
    source = {'type': 'nifti', **_file(path), 'frame': frame, 'units': units,
              'originalAffine': im.affine.tolist(), 'arrayHash': array_digest(raw),
              'storedArrayHash': array_digest(stored),
              'scaling': {'slope': float(im.dataobj.slope), 'intercept': float(im.dataobj.inter)},
              'labelMapping': {str(k): r['id'] for k, r in zip(mapping, regions)}}
    return _retain(SurgicalScene(labels, transform @ im.affine, regions, source, properties or {}), {'labels': raw, 'storedLabels': stored})


def _mesh(path):
    p = Path(path)
    vertices, faces, lines = [], [], []
    if p.suffix.lower() == '.obj':
        for line in p.read_text().splitlines():
            fields = line.partition('#')[0].split()
            if not fields:
                continue
            if fields[0] == 'v':
                vertices.append(list(map(float, fields[1:4])))
            elif fields[0] in ('f', 'l'):
                ids = [int(v.split('/')[0]) for v in fields[1:]]
                ids = [v - 1 if v > 0 else len(vertices) + v for v in ids]
                if fields[0] == 'f':
                    if len(ids) != 3:
                        raise ValueError('OBJ requires explicit triangular faces; triangulate upstream')
                    faces.append(ids)
                else:
                    lines.extend(zip(ids[:-1], ids[1:]))
    elif p.suffix.lower() == '.stl':
        data = p.read_bytes()
        triangles = []
        count = struct.unpack_from('<I', data, 80)[0] if len(data) >= 84 else 0
        if len(data) == 84 + 50 * count:
            for i in range(count):
                triangles.append(np.array(struct.unpack_from('<9f', data, 84 + i * 50 + 12)).reshape(3, 3))
        else:
            rows = [list(map(float, line.split()[1:])) for line in data.decode().splitlines() if line.strip().startswith('vertex ')]
            if len(rows) % 3:
                raise ValueError('Invalid STL triangle vertices')
            triangles = np.asarray(rows).reshape(-1, 3, 3)
        if len(triangles):
            vertices, inverse = np.unique(np.asarray(triangles).reshape(-1, 3), axis=0, return_inverse=True)
            faces = inverse.reshape(-1, 3)
    else:
        raise ValueError('Only OBJ and STL meshes are supported')
    v = np.asarray(vertices, float).reshape(-1, 3)
    f = np.asarray(faces, np.int64).reshape(-1, 3)
    e = np.asarray(lines, np.int64).reshape(-1, 2)
    if not len(v) or not np.isfinite(v).all() or any(a.size and (a.min() < 0 or a.max() >= len(v)) for a in (f, e)):
        raise ValueError('Invalid mesh vertices or indices')
    if len(f) and np.any(np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1) < 1e-12):
        raise ValueError('Degenerate mesh faces are not supported')
    return v, f, e


def _validate_surface(v, f, closed):
    if not len(f):
        raise ValueError('Surface requires triangular faces')
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2) or (closed and np.any(counts != 2)):
        raise ValueError('Solid requires a closed manifold mesh; open surfaces must declare sheet topology')
    directions = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    sums = np.bincount(inverse, weights=directions)
    if np.any(sums[counts == 2] != 0):
        raise ValueError('Mesh faces require consistent orientation')
    # A manifold vertex has one connected face fan, including at sheet boundaries.
    for vertex in np.unique(f):
        incident = np.flatnonzero(np.any(f == vertex, axis=1))
        remaining = set(map(int, incident))
        stack = [remaining.pop()]
        while stack:
            face = stack.pop()
            connected = [j for j in remaining if len(set(f[face]) & set(f[j])) >= 2]
            remaining.difference_update(connected)
            stack.extend(connected)
        if remaining:
            raise ValueError('Nonmanifold vertex has disconnected face fans')


def _segment_distance(p, a, b):
    d = b - a
    denominator = np.dot(d, d)
    if denominator <= 0:
        raise ValueError('Degenerate strand segment')
    t = np.clip((p - a) @ d / denominator, 0, 1)
    return np.linalg.norm(p - a - t[:, None] * d, axis=1)


def _triangle_distance(p, tri):
    a, b, c = tri
    u, w = b - a, c - a
    normal = np.cross(u, w)
    normal /= np.linalg.norm(normal)
    height = (p - a) @ normal
    q = p - height[:, None] * normal - a
    uu, uw, ww = u @ u, u @ w, w @ w
    denominator = uu * ww - uw * uw
    s = ((q @ u) * ww - (q @ w) * uw) / denominator
    t = ((q @ w) * uu - (q @ u) * uw) / denominator
    distance = np.minimum.reduce([_segment_distance(p, a, b), _segment_distance(p, b, c), _segment_distance(p, c, a)])
    inside = (s >= 0) & (t >= 0) & (s + t <= 1)
    distance[inside] = np.abs(height[inside])
    return distance


def from_mesh(path, material, *, resolution, frame, units, topology='solid', thickness_mm=None, name=None, properties=None):
    """Import closed triangular solids or explicitly declared sheets/OBJ strands.

    resolution is a scalar or XYZ spacing in millimeters. Sheets and strands
    require a declared positive thickness_mm, interpreted as full width.
    """
    raw, faces, lines = _mesh(path)
    return _from_surface(raw, faces, lines, material, resolution=resolution, frame=frame,
                         units=units, topology=topology, thickness_mm=thickness_mm,
                         name=name or Path(path).stem, properties=properties,
                         provenance={'type': 'mesh', **_file(path)})


def from_surface_buffers(vertices, faces, material, *, resolution, frame, units,
                         topology='solid', thickness_mm=None, lines=None, name=None, properties=None):
    """Adapt existing indexed triangular surface buffers without files or repairs.

    vertices is an Nx3 numeric array, faces an Mx3 integer index array. For a
    declared strand, use an empty (0,3) faces array and lines as Kx2 indices.
    material is a material string or a single region mapping (id, name,
    material, target, color). Multiple classified regions should be imported
    separately; there is no inference from labels or buffer names.

    frame (RAS+/LPS+), units (mm/meter/micron), and resolution (millimeters)
    must be explicit. Sheets/strands also require full-width thickness_mm.
    Source arrays, dtypes and index ordering are copied read-only and hashed;
    caller buffers remain unchanged. Faces must already be triangular and
    consistently oriented. Validation and voxel occupancy match from_mesh.
    """
    raw = np.asarray(vertices)
    triangles = np.asarray(faces)
    segments = np.empty((0, 2), np.int64) if lines is None else np.asarray(lines)
    if raw.ndim != 2 or raw.shape[1] != 3 or not len(raw) or raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all():
        raise ValueError('Vertices require a finite nonempty Nx3 numeric array')
    for array, width in ((triangles, 3), (segments, 2)):
        if array.ndim != 2 or array.shape[1] != width or array.dtype.kind not in 'iu':
            raise ValueError('Faces and lines require integer index arrays with declared dimensions')
        if array.size and (array.min() < 0 or array.max() >= len(raw)):
            raise ValueError('Surface buffer index outside vertices')
    v = raw.astype(float)
    if len(triangles) and np.any(np.linalg.norm(np.cross(v[triangles[:, 1]] - v[triangles[:, 0]], v[triangles[:, 2]] - v[triangles[:, 0]]), axis=1) < 1e-12):
        raise ValueError('Degenerate mesh faces are not supported')
    return _from_surface(raw, triangles, segments, material, resolution=resolution,
                         frame=frame, units=units, topology=topology,
                         thickness_mm=thickness_mm, name=name or 'Surface buffers',
                         properties=properties, provenance={'type': 'surface_buffers'})


def _from_surface(raw, faces, lines, material, *, resolution, frame, units, topology,
                  thickness_mm, name, properties, provenance):
    convert = _coordinates(frame, units)
    vertices = raw @ convert[:3, :3].T
    spacing = np.broadcast_to(np.asarray(resolution, float), (3,)).copy()
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Declare positive finite voxel resolution in millimeters')
    if topology not in ('solid', 'sheet', 'strand'):
        raise ValueError('Declare solid, sheet, or strand topology')
    radius = 0.
    if topology in ('sheet', 'strand'):
        if thickness_mm is None or not np.isfinite(thickness_mm) or thickness_mm <= 0:
            raise ValueError('Sheets and strands require declared positive thickness_mm')
        radius = thickness_mm / 2
    if topology == 'strand':
        if not len(lines) or len(faces):
            raise ValueError('Strand topology requires OBJ line segments without faces')
        if np.any(np.bincount(lines.ravel().astype(np.int64), minlength=len(vertices)) > 2):
            raise ValueError('Branched strands require separate explicitly classified components')
    else:
        if len(lines):
            raise ValueError('Surface meshes cannot contain unclassified line elements')
        _validate_surface(vertices, faces, topology == 'solid')
    lo, hi = vertices.min(axis=0) - radius, vertices.max(axis=0) + radius
    shape = np.maximum(1, np.ceil((hi - lo) / spacing).astype(int))
    if int(np.prod(shape)) > 8_000_000:
        raise ValueError('Mesh grid exceeds 8 million cells; choose coarser declared resolution')
    affine = np.diag([*spacing, 1.])
    affine[:3, 3] = lo + spacing / 2
    occupied = np.zeros(int(np.prod(shape)), bool)
    # Generalized winding number handles ray/edge ambiguities at voxel centers.
    for start in range(0, len(occupied), 16384):
        stop = min(start + 16384, len(occupied))
        p = np.column_stack(np.unravel_index(np.arange(start, stop), tuple(shape))) * spacing + affine[:3, 3]
        if topology == 'solid':
            winding = np.zeros(len(p))
            boundary = np.zeros(len(p), bool)
            for face in faces:
                tri = vertices[face]
                a, b, c = [vertex - p for vertex in tri]
                la, lb, lc = [np.linalg.norm(x, axis=1) for x in (a, b, c)]
                denominator = la * lb * lc + np.einsum('ij,ij->i', a, b) * lc + np.einsum('ij,ij->i', b, c) * la + np.einsum('ij,ij->i', c, a) * lb
                winding += 2 * np.arctan2(np.einsum('ij,ij->i', a, np.cross(b, c)), denominator)
                boundary |= _triangle_distance(p, tri) < 1e-9
            occupied[start:stop] = (np.abs(winding) > 2 * np.pi) | boundary
        else:
            distances = np.full(len(p), np.inf)
            for element in (faces if topology == 'sheet' else lines):
                d = _triangle_distance(p, vertices[element]) if topology == 'sheet' else _segment_distance(p, *vertices[element])
                distances = np.minimum(distances, d)
            occupied[start:stop] = distances <= radius + 1e-9
    region = _region(material, 1, name)
    source = {**provenance, 'frame': frame, 'units': units,
              'topology': topology, 'thicknessMm': thickness_mm, 'resolutionMm': spacing.tolist(),
              'vertices': raw.tolist(), 'faces': faces.tolist(), 'lines': lines.tolist(),
              'arrayHashes': {k: array_digest(v) for k, v in {'vertices': raw, 'faces': faces, 'lines': lines}.items() if v.size},
              'approximation': 'voxel-center occupancy; subvoxel features may be absent; original topology retained'}
    labels = occupied.reshape(tuple(shape)).astype(np.uint16) * region['id']
    return _retain(SurgicalScene(labels, affine, [region], source, properties or {}), {'vertices': raw, 'faces': faces, 'lines': lines})


def from_native_collection(collection, mapping=None, *, frame, units=None, properties=None):
    """Resample registered source masks onto the reference series grid, nearest-neighbor.

    mapping maps selected layer IDs to material strings or region dictionaries.
    Default maps all layers using declared kind; names never infer material.
    Original memberships and categorical arrays remain read-only source_arrays.
    """
    reference = collection.series(collection.document['reference_series'])
    declared_units = units or reference.get('units_status')
    convert = _coordinates(frame, declared_units)
    reference_affine = collection.patient_affine(reference)
    shape = collection.array(reference['data']).shape
    if len(shape) != 3:
        raise ValueError('Reference must be a 3D volume')
    rows = collection.document['layers']
    if mapping is not None:
        by_id = {r['id']: r for r in rows}
        if not set(mapping) <= set(by_id):
            raise ValueError('Mapping contains missing native layers')
        rows = [by_id[k] for k in mapping]
    if not rows:
        raise ValueError('No native segmentation layers selected')
    labels = np.zeros(shape, np.uint16)
    regions, provenance, arrays = [], [], {}
    for i, row in enumerate(rows, 1):
        series = collection.series(row['source_series'])
        row_units = series.get('units_status')
        if row_units not in ('unknown', None, declared_units) or (row_units in ('unknown', None) and units is None):
            raise ValueError('Every native series requires consistent declared units')
        native_affine = collection.patient_affine(row)  # rejects unreviewed/missing/source-mismatched registrations
        raw = collection.array(row['data'])
        mask = collection.mask(row)
        spec = mapping[row['id']] if mapping is not None else {'material': KINDS.get(row.get('kind'), 'unknown'), 'color': row.get('color', '#92a7b8')}
        region = _region(spec, i, row.get('label', row['id']))
        matrix = np.linalg.inv(native_affine) @ reference_affine
        sampled = affine_transform(mask.astype(np.uint8), matrix[:3, :3], matrix[:3, 3], output_shape=shape, order=0, mode='constant', cval=0, prefilter=False).astype(bool)
        labels[sampled] = region['id']
        regions.append(region)
        arrays[row['id']] = raw
        provenance.append({'layerId': row['id'], 'arrayHash': array_digest(raw), 'maskHash': array_digest(mask),
                           'originalAffine': row['affine'], 'registeredAffine': native_affine.tolist(),
                           'sourceSeries': row['source_series'], 'original': series.get('original'), 'labelValue': row['label_value'],
                           'reviewStatus': row.get('review_status', 'unreviewed'),
                           'revision': row.get('revision', 0), 'originalData': copy.deepcopy(row.get('original_data')),
                           'edits': copy.deepcopy(row.get('edits', [])),
                           'provenance': copy.deepcopy(row.get('provenance', {})),
                           'uncertainty': copy.deepcopy(row.get('uncertainty')),
                           'scoreMeaning': row.get('score_meaning'), 'probability': copy.deepcopy(row.get('probability'))})
    source = {'type': 'native_collection', 'collectionHash': collection.identity(), 'manifest': 'patient-anatomy.json',
              'frame': frame, 'units': declared_units, 'layers': provenance,
              'referenceSeries': reference['id'], 'originalAffine': reference_affine.tolist(),
              'overlapRule': 'last mapped region owns material; original arrays retain overlapping annotations',
              'approximation': 'nearest-neighbor reference-grid sampling; outside reference field is excluded'}
    return _retain(SurgicalScene(labels, convert @ reference_affine, regions, source, properties or {}), arrays)
