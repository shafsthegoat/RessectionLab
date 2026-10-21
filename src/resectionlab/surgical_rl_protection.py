"""Explicit source-cell critical annotations, never inferred from display names.

Conservative circumspheres include entire affine cells (also oblique/sheared
cells). These research exclusion regions are not anatomical certification.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .core import array_digest, semantic_digest

GUARD_SCHEMA = 'resectionlab.surgical-protection-contract.v1'


@dataclass(frozen=True)
class GuardConfig:
    """Shared research exclusion policy; relaxed coverage is an explicit choice."""
    require_critical_coverage: bool = True
    critical_max_displacement_mm: float = 0.

    def __post_init__(self):
        if type(self.require_critical_coverage) is not bool:
            raise ValueError('Critical coverage choice must be boolean')
        if type(self.critical_max_displacement_mm) not in (int, float) or not np.isfinite(self.critical_max_displacement_mm) or self.critical_max_displacement_mm < 0:
            raise ValueError('Critical displacement allowance must be finite nonnegative millimeters')


def missing_critical(protection):
    present = {a.structure_id for a in protection.annotations if a.cells}
    return tuple(sorted(set(protection.missing) | ({'brainstem'} - present)))


def protection_contract(protection, config):
    """Self-contained, content-bound evidence for native Save/Open and replay."""
    record = {'schema': GUARD_SCHEMA, 'sceneId': protection.scene.identity,
        'sourceBinding': grid_binding(protection.scene), 'config': asdict(config),
        'clearanceMm': protection.clearance_mm, 'required': list(protection.required),
        'annotations': [{'structureId': a.structure_id, 'cells': list(a.cells),
            'provenance': a.provenance, 'reviewStatus': a.review_status, 'displayName': a.display_name}
            for a in protection.annotations]}
    record['contentHash'] = semantic_digest(record)
    return record


def restore_protection_contract(scene, record):
    record = dict(record)
    digest = record.pop('contentHash', None)
    if record.get('schema') != GUARD_SCHEMA or semantic_digest(record) != digest:
        raise ValueError('Protection contract integrity mismatch')
    if record.get('sceneId') != scene.identity or record.get('sourceBinding') != grid_binding(scene):
        raise ValueError('Protection contract source scene mismatch')
    annotations = [ProtectionAnnotation(row['structureId'], tuple(row['cells']), record['sourceBinding'],
        row['provenance'], row['reviewStatus'], row['displayName']) for row in record['annotations']]
    protection = ProtectionSet(scene, annotations, clearance_mm=record['clearanceMm'], required=record['required'])
    config = GuardConfig(**record['config'])
    if config.require_critical_coverage and missing_critical(protection):
        raise ValueError('MISSING_CRITICAL_ANNOTATIONS: ' + ', '.join(missing_critical(protection)))
    return protection, config


def grid_binding(scene):
    return semantic_digest({'labels': array_digest(scene.labels), 'affine': scene.affine.tolist(),
                            'source': scene.source})


def cell_radius(affine):
    corners = np.asarray(list(itertools.product((-.5, .5), repeat=3)))
    return float(np.linalg.norm(corners @ np.asarray(affine)[:3, :3].T, axis=1).max())


@dataclass(frozen=True)
class ProtectionAnnotation:
    structure_id: str
    cells: tuple[int, ...]
    source_binding: str
    provenance: str
    review_status: str = 'unreviewed'
    display_name: str = ''

    def __post_init__(self):
        if not self.structure_id or not self.source_binding or not self.provenance:
            raise ValueError('Explicit structure identity, source binding and provenance required')
        cells = tuple(self.cells)
        if any(not isinstance(c, (int, np.integer)) or c < 0 for c in cells):
            raise ValueError('Cells must be nonnegative integer source indices')
        object.__setattr__(self, 'cells', tuple(sorted(set(map(int, cells)))))

    @classmethod
    def from_mask(cls, scene, structure_id, mask, *, provenance, review_status='unreviewed', display_name=''):
        mask = np.asarray(mask)
        if mask.shape != scene.shape or mask.dtype != np.bool_:
            raise ValueError('A boolean mask on the exact scene grid is required')
        return cls(structure_id, tuple(np.flatnonzero(mask)), grid_binding(scene), provenance, review_status, display_name)


class ProtectionSet:
    def __init__(self, scene, annotations=(), *, clearance_mm=2., required=('brainstem',), strict=False):
        if not np.isfinite(clearance_mm) or clearance_mm < 0:
            raise ValueError('Clearance must be finite nonnegative millimeters')
        self.scene = scene
        self.annotations = tuple(annotations)
        self.clearance_mm = float(clearance_mm)
        self.required = tuple(required)
        binding = grid_binding(scene)
        if len({a.structure_id for a in self.annotations}) != len(self.annotations):
            raise ValueError('Duplicate structure identity')
        for annotation in self.annotations:
            if annotation.source_binding != binding or any(c >= scene.labels.size for c in annotation.cells):
                raise ValueError('Protection annotation does not match source grid')
        present = {a.structure_id for a in self.annotations if a.cells}
        self.missing = tuple(s for s in self.required if s not in present)
        if strict and self.missing:
            raise ValueError('Missing required critical annotations: ' + ', '.join(self.missing))
        self.cells = np.asarray(sorted({c for a in self.annotations for c in a.cells}), dtype=np.int64)
        self.cells.flags.writeable = False
        self._points = scene.points(self.cells)
        self._radius = cell_radius(scene.affine)
        self._tree = cKDTree(self._points)

    def report(self):
        return {'status': 'unknown-coverage' if self.missing else 'declared-masks-only',
                'missingRequired': list(self.missing), 'clearanceMm': self.clearance_mm,
                'anatomicalCompleteness': 'unknown', 'clinicalValidation': 'none',
                'unknownStructures': ['motor/language cortex', 'corticospinal tract', 'optic pathways', 'cranial nerves'],
                'structures': [{'id': a.structure_id, 'name': a.display_name,
                                'sourceCells': len(a.cells), 'provenance': a.provenance,
                                'reviewStatus': a.review_status} for a in self.annotations]}

    def intersects_cells(self, cells):
        return np.intersect1d(self.cells, np.asarray(list(cells), dtype=np.int64))

    def damage_mm3(self, cells):
        return float(len(self.intersects_cells(cells)) * self.scene.voxel_mm3)

    def capsule_hits(self, start, end, radius_mm=0., *, positions=None):
        """Return source IDs within full-cell conservative capsule clearance.

        Optional positions must be current RAS-mm centers in self.cells order.
        Source IDs remain fixed even when tissue moves or ownership overlaps.
        """
        a, b = np.asarray(start, float), np.asarray(end, float)
        p = self._points if positions is None else np.asarray(positions, float)
        if a.shape != (3,) or b.shape != (3,) or p.shape != self._points.shape or not np.isfinite([a, b]).all() or not np.isfinite(p).all() or not np.isfinite(radius_mm) or radius_mm < 0:
            raise ValueError('Finite RAS-mm capsule and source-cell positions required')
        candidate = np.arange(len(self.cells))
        if positions is None and len(self.cells):
            candidate = np.asarray(self._tree.query_ball_point((a + b) / 2,
                np.linalg.norm(b - a) / 2 + radius_mm + self.clearance_mm + self._radius + 1e-9), dtype=np.int64)
            p = p[candidate]
        delta = b - a
        length2 = float(delta @ delta)
        t = np.clip((p - a) @ delta / length2, 0, 1) if length2 else np.zeros(len(p))
        distance = np.linalg.norm(p - a - t[:, None] * delta, axis=1)
        return self.cells[candidate[distance <= radius_mm + self.clearance_mm + self._radius + 1e-9]]

    def sweep_hits(self, tool, parameters, start, end, *, positions=None):
        """Conservative continuous translation/rotation of working part AND shaft."""
        from .surgical_motion import swept_envelopes
        hits = [self.capsule_hits(a, b, radius, positions=positions)
                for a, b, radius, _ in swept_envelopes(tool, parameters, start, end)]
        return np.unique(np.concatenate(hits)) if hits else np.empty(0, np.int64)


def annotation_from_nifti(scene, path, structure_id, *, source_binding, registration_ras_mm,
                           provenance, label_value=1, review_status='unreviewed'):
    """Import an explicitly registered SAME scene-grid NIfTI only.

    Caller must supply exact source binding and RAS transform; no guessed atlas
    alignment, arbitrary resize, or implicit identity registration is accepted.
    """
    import nibabel as nib
    image = nib.load(str(path))
    transform = np.asarray(registration_ras_mm, float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]) or abs(np.linalg.det(transform[:3, :3])) < 1e-9:
        raise ValueError('Explicit nonsingular RAS-mm registration required')
    if source_binding != grid_binding(scene) or image.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('Source binding and explicit millimeter units required')
    if image.shape != scene.shape or not np.allclose(transform @ image.affine, scene.affine, rtol=0, atol=1e-5):
        raise ValueError('Registered mask must match scene grid exactly')
    return ProtectionAnnotation.from_mask(scene, structure_id, np.asarray(image.dataobj) == label_value,
        provenance=provenance, review_status=review_status)


def annotation_from_collection(scene, root, layer_id, structure_id, *, allow_unreviewed=False):
    """Conservatively register an existing native layer; never generate anatomy.

    Circumsphere overlap may enlarge boundaries, but cannot omit an intersecting
    source/target affine cell. Full source memberships survive material priority.
    """
    from .patient_anatomy import PatientImagingCollection
    collection = PatientImagingCollection(Path(root))
    adapter = scene.source.get('adapter') or {}
    if adapter.get('sourceCollectionHash') != collection.document.get('snapshot_id'):
        raise ValueError('Collection snapshot does not match scene source')
    row = collection.layer(layer_id)
    sid = row['source_series']
    if collection.series(sid).get('units_status') != 'mm':
        raise ValueError('Source units must be millimeters')
    current = sid
    while current:
        transform = collection.document['transforms'][current]
        if transform['status'] not in ('reference', 'reviewed', 'source-provided') and not allow_unreviewed:
            raise ValueError('Unreviewed registration requires explicit opt-in')
        current = transform.get('parent_series')
    if row.get('review_status') != 'reviewed' and not allow_unreviewed:
        raise ValueError('Unreviewed anatomy requires explicit research opt-in')
    matrix = collection.registration_matrix(sid)
    # Confirm the resolved native frame agrees with this derivative's saved
    # registration evidence, rather than trusting a snapshot label alone.
    reference_id = collection.document['reference_series']
    reference = collection.series(reference_id)
    expected_reference = collection.patient_affine(reference)
    brain_evidence = adapter.get('sourceLayers', {}).get('brain', {})
    if sid == reference_id and not np.allclose(brain_evidence.get('registeredAffine', np.zeros((4, 4))), expected_reference, rtol=0, atol=1e-6):
        raise ValueError('Collection registration disagrees with saved scene source')
    affine = matrix @ np.asarray(row['affine'])
    native = np.argwhere(collection.array(row['data']) == row['label_value'])
    mask = np.zeros(scene.shape, bool)
    if len(native):
        points = native @ affine[:3, :3].T + affine[:3, 3]
        tree = cKDTree(points)
        radius = cell_radius(affine) + cell_radius(scene.affine)
        indices = points @ scene.inverse[:3, :3].T + scene.inverse[:3, 3]
        padding = radius * np.linalg.norm(scene.inverse[:3, :3], axis=1)
        lo = np.maximum(0, np.floor(indices.min(axis=0) - padding).astype(int))
        hi = np.minimum(scene.shape, np.ceil(indices.max(axis=0) + padding).astype(int) + 1)
        for x in range(lo[0], hi[0]):
            ijk = np.stack(np.meshgrid([x], np.arange(lo[1], hi[1]), np.arange(lo[2], hi[2]), indexing='ij'), axis=-1).reshape(-1, 3)
            world = ijk @ scene.affine[:3, :3].T + scene.affine[:3, 3]
            keep = tree.query(world, distance_upper_bound=radius + 1e-9)[0] <= radius + 1e-9
            mask[tuple(ijk[keep].T)] = True
    provenance = semantic_digest({'collection': collection.document['snapshot_id'], 'layer': row,
                                  'registeredAffine': affine.tolist(), 'method': 'affine-cell circumsphere overlap'})
    return ProtectionAnnotation.from_mask(scene, structure_id, mask, provenance=(f"collection={collection.document['snapshot_id']}; layer={layer_id}; label={row['label_value']}; "
                    f"sourceArraySha256={row['data']['sha256']}; evidence={provenance}; sampling=affine-cell circumsphere overlap"),
        review_status=row.get('review_status', 'unknown'), display_name=row.get('label', layer_id))


def save_protection(protection, path):
    """Save explicit annotation cells with content hashes and source binding."""
    import json
    import hashlib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = path.with_suffix('.npz')
    arrays = {f'cells_{i}': np.asarray(a.cells, dtype=np.int64) for i, a in enumerate(protection.annotations)}
    np.savez_compressed(arrays_path, **arrays)
    rows = [{'structureId': a.structure_id, 'displayName': a.display_name,
             'provenance': a.provenance, 'reviewStatus': a.review_status,
             'array': f'cells_{i}', 'arrayHash': array_digest(arrays[f'cells_{i}'])}
            for i, a in enumerate(protection.annotations)]
    document = {'schema': 'surgical-rl-protection-v1', 'sourceBinding': grid_binding(protection.scene),
                'shape': list(protection.scene.shape), 'affineRasMm': protection.scene.affine.tolist(),
                'arrays': arrays_path.name, 'arraysSha256': hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
                'required': list(protection.required), 'clearanceMm': protection.clearance_mm,
                'annotations': rows, 'coverage': protection.report()}
    document['contentHash'] = semantic_digest(document)
    path.write_text(json.dumps(document, indent=2) + '\n')
    return document


def load_protection(scene, path, *, strict=False):
    import json
    import hashlib
    path = Path(path)
    document = json.loads(path.read_text())
    digest = document.pop('contentHash')
    if document.get('schema') != 'surgical-rl-protection-v1' or semantic_digest(document) != digest:
        raise ValueError('Protection manifest integrity mismatch')
    if document['sourceBinding'] != grid_binding(scene) or document['shape'] != list(scene.shape) or not np.array_equal(document['affineRasMm'], scene.affine):
        raise ValueError('Protection manifest source grid mismatch')
    arrays_path = path.parent / document['arrays']
    if arrays_path.parent.resolve() != path.parent.resolve():
        raise ValueError('Protection array must be beside manifest')
    if hashlib.sha256(arrays_path.read_bytes()).hexdigest() != document['arraysSha256']:
        raise ValueError('Protection array checksum mismatch')
    annotations = []
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for row in document['annotations']:
            cells = arrays[row['array']]
            if cells.ndim != 1 or cells.dtype.kind not in 'iu' or array_digest(cells) != row['arrayHash']:
                raise ValueError('Protection cells checksum or type mismatch')
            annotations.append(ProtectionAnnotation(row['structureId'], tuple(cells), document['sourceBinding'],
                row['provenance'], row['reviewStatus'], row['displayName']))
    return ProtectionSet(scene, annotations, clearance_mm=document['clearanceMm'], required=document['required'], strict=strict)
