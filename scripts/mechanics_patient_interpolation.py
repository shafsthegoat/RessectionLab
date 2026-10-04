"""Independent, bounded point sampling of a previously frozen tet10 field.

No solver, registration, anatomical acceptance or V disclosure occurs here.
Queries locate ORIGINAL reference geometry; no snapping or extrapolation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

EDGES = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])
MAX_NODES, MAX_ELEMENTS, MAX_QUERIES = 10000, 20000, 4096
MAX_CANDIDATES = 4096
MAX_ARTIFACT_BYTES = 64*1024**2
BOUNDARY_TOLERANCE_M = 1e-12
MIDPOINT_TOLERANCE_M = 1e-12
CONTINUITY_TOLERANCE_M = 1e-12
MIN_SINGULAR_RATIO = 1e-10
MAX_RELATIVE_MIDPOINT_ERROR = 1e-8
ORDER = 'vertices0123_edges01_12_20_03_13_23'


def _immutable(value, dtype):
    a = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _points(value, limit):
    a = np.asarray(value)
    if (a.dtype.kind not in 'fiu' or a.ndim != 2 or a.shape[1:] != (3,)
            or not 1 <= len(a) <= limit or not np.isfinite(a).all()):
        raise ValueError('Finite bounded real point/vector array required')
    return np.array(a, dtype=np.float64, copy=True)


def _transform(value):
    a = np.asarray(value)
    if (a.dtype.kind not in 'fiu' or a.shape != (4, 4) or not np.isfinite(a).all()
            or not np.array_equal(a[3], [0, 0, 0, 1])):
        raise ValueError('Finite homogeneous native-meters to common-millimeters transform required')
    a = np.array(a, dtype=np.float64)
    rotation = a[:3, :3]/1000
    if (not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-10, rtol=0)
            or abs(np.linalg.det(rotation)-1) > 1e-10 or np.max(np.abs(a[:3, 3])) > 1e6):
        raise ValueError('Only proper rotation times1000 plus millimeter translation is supported')
    return a


def _geometry_status(x, cells):
    corners = x[cells[:, :4]]
    matrices = np.swapaxes(corners[:, 1:]-corners[:, :1], 1, 2)
    singular = np.linalg.svd(matrices, compute_uv=False)
    if (np.any(np.linalg.det(matrices) <= 0) or np.any(singular[:, 0] == 0)
            or np.any(singular[:, -1] <= MIN_SINGULAR_RATIO*singular[:, 0])):
        return 'unsupported_reference_geometry', 'degenerate_inverted_or_ill_conditioned_reference'
    midpoint_error = np.max(np.linalg.norm(x[cells[:, 4:]]-corners[:, EDGES].mean(axis=2), axis=2),axis=1)
    if np.any(midpoint_error > np.minimum(MIDPOINT_TOLERANCE_M,MAX_RELATIVE_MIDPOINT_ERROR*singular[:,-1])):
        return 'unsupported_reference_geometry', 'curved_reference_tet10_unsupported'
    if (len(np.unique(x, axis=0)) != len(x)
            or len(np.unique(np.sort(cells[:, :4], axis=1), axis=0)) != len(cells)):
        return 'nonconforming_reference_mesh', 'duplicate_nodes_or_elements'
    edge_ids = np.sort(cells[:, :4][:, EDGES], axis=2).reshape(-1, 2)
    _, inverse = np.unique(edge_ids, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind='stable'); mids = cells[:, 4:].ravel()[order]
    if np.any((np.diff(inverse[order]) == 0) & (np.diff(mids) != 0)):
        return 'nonconforming_reference_mesh', 'shared_edge_has_different_midpoint_ids'
    owners = {}
    for cell in cells:
        for opposite in range(4):
            face = tuple(sorted(np.delete(cell[:4], opposite).tolist()))
            owners.setdefault(face, []).append(int(cell[opposite]))
    for face, opposite in owners.items():
        if len(opposite) > 2:
            return 'nonconforming_reference_mesh', 'nonmanifold_shared_face'
        if len(opposite) == 2:
            a, b, c = x[list(face)]
            normal = np.cross(b-a, c-a)
            if np.prod((x[opposite]-a)@normal) >= 0:
                return 'nonconforming_reference_mesh', 'shared_face_cells_on_same_side'
    return 'supported', 'straight_reference_geometry'


@dataclass(frozen=True)
class FrozenTet10Field:
    nodes_m: np.ndarray
    tet10_indices: np.ndarray
    displacement_m: np.ndarray
    native_m_to_common_mm: np.ndarray

    def __post_init__(self):
        x = _points(self.nodes_m, MAX_NODES); u = _points(self.displacement_m, MAX_NODES)
        e = np.asarray(self.tet10_indices)
        if (x.shape != u.shape or np.max(np.abs(x)) > 10 or np.max(np.abs(u)) > 10
                or e.dtype.kind not in 'iu' or e.ndim != 2 or e.shape[1:] != (10,)
                or not 1 <= len(e) <= MAX_ELEMENTS or e.min() < 0 or e.max() >= len(x)
                or np.any(np.diff(np.sort(e, axis=1), axis=1) == 0) or len(np.unique(e)) != len(x)):
            raise ValueError('Complete bounded nodal field and valid zero-based tet10 connectivity required')
        transform = _transform(self.native_m_to_common_mm)
        for name, value, dtype in [('nodes_m',x,np.float64), ('displacement_m',u,np.float64),
                                   ('tet10_indices',e,np.int64), ('native_m_to_common_mm',transform,np.float64)]:
            object.__setattr__(self, name, _immutable(value,dtype))
        status, reason = _geometry_status(self.nodes_m, self.tet10_indices)
        object.__setattr__(self, 'geometry_status', status)
        object.__setattr__(self, 'geometry_reason', reason)
        midpoint_error = float(np.max(np.linalg.norm(self.nodes_m[self.tet10_indices[:,4:]]
            -self.nodes_m[self.tet10_indices[:,:4]][:,EDGES].mean(axis=2),axis=2)))
        object.__setattr__(self, 'maximum_midpoint_error_m', midpoint_error)
        object.__setattr__(self, 'sha256', _digest(*self._arrays()))

    def _arrays(self):
        return self.nodes_m, self.tet10_indices, self.displacement_m, self.native_m_to_common_mm

    def assert_intact(self):
        if _digest(*self._arrays()) != self.sha256:
            raise ValueError('Frozen field array metadata or content changed')


def tet10_values(barycentric):
    """Quadratic Lagrange basis; input coordinates are never clamped."""
    l = np.asarray(barycentric, dtype=np.float64)
    return np.concatenate((l*(2*l-1), 4*l[..., EDGES[:, 0]]*l[..., EDGES[:, 1]]), axis=-1)


def sample_field(field, source_ras_mm):
    """Detailed evaluator-only report; unsupported displacement is always None."""
    if type(field) is not FrozenTet10Field:
        raise TypeError('A complete immutable field is required')
    field.assert_intact()
    query = _points(source_ras_mm, MAX_QUERIES)
    if np.max(np.abs(query)) > 1e6:
        raise ValueError('Common-frame query range exceeded')
    transform = field.native_m_to_common_mm
    native = np.linalg.solve(transform[:3, :3], (query-transform[:3, 3]).T).T
    rows = []
    if field.geometry_status != 'supported':
        rows = [{'status':field.geometry_status, 'reason':field.geometry_reason,
                 'displacement_ras_mm':None, 'element_ids':[]} for _ in query]
    else:
        cells = field.tet10_indices; corners = field.nodes_m[cells[:, :4]]
        matrices = np.swapaxes(corners[:, 1:]-corners[:, :1], 1, 2)
        inverse = np.linalg.inv(matrices)
        gradients = np.concatenate((-inverse.sum(axis=1, keepdims=True), inverse), axis=1)
        bary_tolerance = np.linalg.norm(gradients, axis=2)*BOUNDARY_TOLERANCE_M
        centers = corners.mean(axis=1)
        radii = np.max(np.linalg.norm(corners-centers[:, None], axis=2), axis=1)
        tree = cKDTree(centers)
        low, high = corners.min(axis=1), corners.max(axis=1)
        for point in native:
            ids = np.asarray(tree.query_ball_point(point, float(radii.max()+BOUNDARY_TOLERANCE_M)), dtype=int)
            ids = np.sort(ids[np.all((point >= low[ids]-BOUNDARY_TOLERANCE_M)
                                   & (point <= high[ids]+BOUNDARY_TOLERANCE_M), axis=1)])
            row = {'status':'outside_reference_domain', 'reason':'no_containing_reference_cell',
                   'displacement_ras_mm':None, 'element_ids':[]}
            if len(ids) > MAX_CANDIDATES:
                row.update(status='unsupported_reference_geometry', reason='candidate_budget_exceeded')
            elif len(ids):
                local = np.einsum('eij,ej->ei', inverse[ids], point-corners[ids, 0])
                bary = np.column_stack((1-local.sum(axis=1),local))
                inside = np.all((bary >= 0) & (bary <= 1), axis=1)
                near = np.all((bary >= -bary_tolerance[ids]) & (bary <= 1+bary_tolerance[ids]), axis=1)
                if not inside.any() and near.any():
                    row.update(status='uncertain_reference_boundary', reason='roundoff_band_without_proven_nonnegative_barycentric')
                elif inside.any():
                    hit = ids[inside]; weights = bary[inside]; row['element_ids'] = hit.tolist()
                    common = set(cells[hit[0], :4].tolist())
                    for cell in cells[hit[1:], :4]:
                        common.intersection_update(cell.tolist())
                    conforming_contact = bool(common) and all(
                        all(weight == 0 for vertex,weight in zip(cells[index,:4], w) if int(vertex) not in common)
                        for index,w in zip(hit,weights))
                    if len(hit) > 1 and not conforming_contact:
                        band_only_contact = bool(common) and all(
                            all(abs(weight) <= tol for vertex,weight,tol in zip(cells[index,:4],w,bary_tolerance[index]) if int(vertex) not in common)
                            for index,w in zip(hit,weights))
                        if band_only_contact:
                            row.update(status='uncertain_reference_boundary', reason='shared_simplex_contact_only_within_roundoff_band')
                        else:
                            row.update(status='ambiguous_reference_location', reason='multiple_cells_without_shared_containing_simplex')
                    else:
                        values = np.einsum('en,eni->ei', tet10_values(weights), field.displacement_m[cells[hit]])
                        disagreement = float(pdist(values).max()) if len(values)>1 else 0.
                        row['maximum_containing_field_disagreement_m'] = disagreement
                        if disagreement > CONTINUITY_TOLERANCE_M:
                            row.update(status='nonconforming_reference_mesh', reason='containing_cell_field_discontinuity')
                        else:
                            # All containing cells checked; averaging their equal traces
                            # avoids arbitrary first-cell ownership on a conforming face.
                            motion = transform[:3,:3]@values.mean(axis=0)
                            row.update(status='supported', reason='inside_or_conforming_boundary', displacement_ras_mm=motion.tolist())
            rows.append(row)
    field.assert_intact()
    return {'schema':'tet10-point-samples-v1','field_sha256':field.sha256,'rows':rows,
            'counts':{status:sum(r['status']==status for r in rows) for status in sorted({r['status'] for r in rows})},
            'boundary_tolerance_m':BOUNDARY_TOLERANCE_M,'continuity_tolerance_m':CONTINUITY_TOLERANCE_M,
            'maximum_midpoint_error_m':field.maximum_midpoint_error_m,
            'straight_reference_geometry_approximation_bound_m':1.5*field.maximum_midpoint_error_m,
            'query_snapping':False,'extrapolation':False,'anatomical_alignment_accepted':False}


def _bound_payload(root, binding):
    if not isinstance(binding,dict) or set(binding) != {'path','sha256'}:
        raise ValueError('Exact artifact path/hash binding required')
    root = Path(root).resolve(); relative = binding['path']
    if not isinstance(relative,str) or Path(relative).is_absolute():
        raise ValueError('Relative field artifact path required')
    path = (root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Field artifact escapes root')
    with path.open('rb') as stream:
        payload = stream.read(MAX_ARTIFACT_BYTES+1)
    if len(payload) > MAX_ARTIFACT_BYTES or hashlib.sha256(payload).hexdigest() != binding['sha256']:
        raise ValueError('Oversized or changed frozen artifact')
    return payload


def _npz(payload, expected):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.infolist()) != len(expected) or set(archive.namelist()) != {k+'.npy' for k in expected}:
            raise ValueError('Exact complete field arrays required')
        if sum(i.file_size for i in archive.infolist()) > MAX_ARTIFACT_BYTES:
            raise ValueError('Expanded field arrays exceed bound')
        for key,(columns,maximum,dtype) in expected.items():
            with archive.open(key+'.npy') as stream:
                version = np.lib.format.read_magic(stream)
                if version not in ((1,0),(2,0)):
                    raise ValueError('Unsupported NPY header')
                reader = np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
                shape,fortran,actual_dtype = reader(stream)
                if (len(shape)!=2 or shape[1]!=columns or not 1<=shape[0]<=maximum
                        or actual_dtype != np.dtype(dtype) or fortran
                        or stream.tell()+int(np.prod(shape))*actual_dtype.itemsize != archive.getinfo(key+'.npy').file_size):
                    raise ValueError('Malformed or oversized array before allocation')
    with np.load(io.BytesIO(payload),allow_pickle=False) as values:
        return {key:values[key] for key in expected}


def load_frozen_field(root, manifest):
    """Validate every complete field artifact before accepting query coordinates."""
    from scripts import mechanics_patient_comparison as comparison
    comparison._hash(manifest.get('forward_hash'))
    comparison._check_external(root,manifest,forward_hash=manifest.get('forward_hash'),protocol_sha256=manifest.get('protocol_sha256'))
    if manifest.get('interpolator_entrypoint') != 'sample_frozen_field':
        raise ValueError('Frozen entrypoint differs')
    payloads = {key:_bound_payload(root,value) for key,value in manifest['artifacts'].items()}
    if payloads['interpolator_source'] != Path(__file__).read_bytes():
        raise ValueError('Executing interpolator source differs')
    mesh = _npz(payloads['reference_mesh'], {'nodes_m':(3,MAX_NODES,'float64'), 'tet10_indices':(10,MAX_ELEMENTS,'int64')})
    motion = _npz(payloads['nodal_displacements'], {'displacement_m':(3,MAX_NODES,'float64')})
    frame = json.loads(payloads['geometry_frame'])
    keys = {'schema','native_reference_frame','native_length_units','common_frame','common_length_units',
            'native_m_to_common_mm','baseline_alignment_sha256','alignment_status','source_image_sha256'}
    if (set(frame)!=keys or frame['schema']!='native-rest-to-common-ras-v1'
            or frame['native_reference_frame']!='MRI_RAS+' or frame['native_length_units']!='m'
            or frame['common_frame']!='RAS+' or frame['common_length_units']!='mm'
            or frame['alignment_status']!='provisional_baseline_diagnostic'):
        raise ValueError('Explicit native/common frame and provisional alignment required')
    for key in ('baseline_alignment_sha256','source_image_sha256'):
        if comparison._hash(frame[key]) != frame[key]:
            raise ValueError('Bare source/alignment SHA256 required')
    field = FrozenTet10Field(mesh['nodes_m'],mesh['tet10_indices'],motion['displacement_m'],frame['native_m_to_common_mm'])
    return field


def sample_frozen_field(root, manifest, queries):
    """Comparison adapter; receives only revealed V sources, never destinations."""
    from scripts import mechanics_patient_comparison as comparison
    if type(queries) is not comparison.FieldQueries:
        raise TypeError('Evaluator-only FieldQueries required')
    field = load_frozen_field(root,manifest)
    result = sample_field(field,queries.source_ras_mm)
    return comparison.FieldSamples(queries.sha256,
        tuple(None if row['displacement_ras_mm'] is None else tuple(row['displacement_ras_mm']) for row in result['rows']),
        tuple(row['status'] for row in result['rows']))
