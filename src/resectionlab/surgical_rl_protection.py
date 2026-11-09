"""Explicit source-cell critical annotations, never inferred from display names.

Conservative circumspheres include entire affine cells (also oblique/sheared
cells). These research exclusion regions are not anatomical certification.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
from pathlib import Path
import weakref
import numpy as np
from scipy.spatial import cKDTree
from .core import array_digest, immutable_array, semantic_digest

GUARD_SCHEMA = 'resectionlab.surgical-protection-contract.v1'
_INDEXES = weakref.WeakKeyDictionary()
_GEOMETRY = weakref.WeakKeyDictionary()


def _array_facts(array):
    """Bind immutable buffers as well as their mutable ndarray view metadata."""
    facts = []
    while isinstance(array, np.ndarray):
        facts.append((array, array.shape, array.dtype.str, array.strides,
            array.__array_interface__['data'][0], array.base))
        array = array.base
    if not isinstance(array, bytes):
        raise ValueError('Protection geometry requires an immutable byte owner')
    return tuple(facts)


def _array_current(array, facts):
    if array is not facts[0][0]:
        return False
    return all(not value.flags.writeable and value.shape == shape and
        value.dtype.str == dtype and value.strides == strides and
        value.__array_interface__['data'][0] == pointer and value.base is owner
        for value, shape, dtype, strides, pointer, owner in facts)


def _private_view(array):
    """Separate operative ndarray headers from exposed immutable inspection views."""
    owner = array
    while isinstance(owner, np.ndarray):
        owner = owner.base
    if not isinstance(owner, bytes):
        raise ValueError('Protection geometry requires an immutable byte owner')
    return np.frombuffer(owner, dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, eq=False)
class _ProtectionIndex:
    """Immutable inspection metadata; the operative KD tree never escapes.

    A read-only flag on cKDTree-owned indices can be reversed. Exposing those
    operative arrays would allow an index edit to evade an O(1) cache witness.
    These are immutable snapshots, used by existing owner fingerprints too.
    """
    data: np.ndarray
    indices: np.ndarray
    mins: np.ndarray
    maxes: np.ndarray
    n: int
    m: int
    leafsize: int

    def _validate_metadata(self):
        registered = _INDEXES.get(self)
        if registered is None:
            raise ValueError('Protection static index changed; construct a new ProtectionSet')
        _, arrays, scalars = registered
        current = (self.n, self.m, self.leafsize)
        if (any(type(a) is not type(b) or a != b for a, b in zip(current, scalars)) or any(
                not _array_current(getattr(self, name), facts) for name, facts in arrays)):
            raise ValueError('Protection static index changed; construct a new ProtectionSet')

    def query_ball_point(self, *args, **kwargs):
        _ProtectionIndex._validate_metadata(self)
        return _INDEXES[self][0].query_ball_point(*args, **kwargs)


def _make_index(points):
    tree = cKDTree(_private_view(points))
    index = _ProtectionIndex(points, immutable_array(tree.indices),
        immutable_array(tree.mins), immutable_array(tree.maxes), tree.n, tree.m, tree.leafsize)
    _INDEXES[index] = (tree, tuple((name, _array_facts(getattr(index, name)))
        for name in ('data', 'indices', 'mins', 'maxes')), (tree.n, tree.m, tree.leafsize))
    return index


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
        if type(self.structure_id) is not str or not self.structure_id or not self.source_binding or not self.provenance:
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
        self.cells = immutable_array(sorted({c for a in self.annotations for c in a.cells}), np.int64)
        self._points = immutable_array(scene.points(self.cells), np.float64)
        if self._points.shape != (len(self.cells), 3) or not np.isfinite(self._points).all():
            raise ValueError('Finite RAS-mm source-cell positions required')
        self._radius = cell_radius(scene.affine)
        self._tree = _make_index(self._points)
        # Direct aliases make the extra immutable index snapshots visible to
        # existing shared-protection array accounting, without counting data twice.
        self._index_indices = self._tree.indices
        self._index_mins = self._tree.mins
        self._index_maxes = self._tree.maxes
        _GEOMETRY[self] = (scene, scene.identity, self._radius, self._tree,
            tuple((name, _array_facts(getattr(self, name))) for name in
                ('cells', '_points', '_index_indices', '_index_mins', '_index_maxes')),
            _array_facts(scene.affine), _array_facts(scene.labels), _private_view(self.cells))

    def _require_geometry(self):
        original = _GEOMETRY.get(self)
        if original is None:
            raise ValueError('Protection static geometry changed; construct a new ProtectionSet')
        scene, identity, radius, tree, arrays, affine, labels, _ = original
        if (self.scene is not scene or scene.identity != identity or self._radius != radius or
                self._tree is not tree or not _array_current(scene.affine, affine) or
                not _array_current(scene.labels, labels) or any(
                    not _array_current(getattr(self, name), facts) for name, facts in arrays)):
            raise ValueError('Protection static geometry changed; construct a new ProtectionSet')

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
        # Instance attributes cannot replace these trusted validators or the
        # operative query. Authenticated private views also prevent a later
        # exposed-header edit from changing the geometry used by this call.
        ProtectionSet._require_geometry(self)
        _, _, source_radius, index, _, _, _, source_cells = _GEOMETRY[self]
        a, b = np.asarray(start, float), np.asarray(end, float)
        clearance = self.clearance_mm
        if (a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all() or
                not np.isfinite(radius_mm) or radius_mm < 0 or
                not np.isfinite(clearance) or clearance < 0):
            raise ValueError('Finite RAS-mm capsule and source-cell positions required')
        clearance, radius_mm = float(clearance), float(radius_mm)
        source_points = _INDEXES[index][0].data
        if positions is None:
            candidate = np.asarray(_ProtectionIndex.query_ball_point(index, (a + b) / 2,
                np.linalg.norm(b - a) / 2 + radius_mm + clearance + source_radius + 1e-9), dtype=np.int64)
            p = source_points[candidate]
        else:
            _ProtectionIndex._validate_metadata(index)
            p = np.asarray(positions, float)
            if p.shape != source_points.shape or not np.isfinite(p).all():
                raise ValueError('Finite RAS-mm capsule and source-cell positions required')
            candidate = np.arange(len(source_cells))
        delta = b - a
        length2 = float(delta @ delta)
        t = np.clip((p - a) @ delta / length2, 0, 1) if length2 else np.zeros(len(p))
        distance = np.linalg.norm(p - a - t[:, None] * delta, axis=1)
        return source_cells[candidate[distance <= radius_mm + clearance + source_radius + 1e-9]]

    def sweep_hits(self, tool, parameters, start, end, *, positions=None):
        """Conservative continuous translation/rotation of working part AND shaft."""
        from .surgical_motion import swept_envelopes
        hits = [self.capsule_hits(a, b, radius, positions=positions)
                for a, b, radius, _ in swept_envelopes(tool, parameters, start, end)]
        return np.unique(np.concatenate(hits)) if hits else np.empty(0, np.int64)


def declared_scene_protection(scene, protection=None):
    """Bind explicit exclusions, retaining memberships hidden by later owners.

    Supplemental contracts may enlarge these exclusions, never remove them.
    A material name alone supplies no anatomical protection evidence.
    """
    if protection is not None and protection.scene.identity != scene.identity:
        raise ValueError('Protection scene mismatch')
    rows = list(scene.source.get('protectedAnatomyV1', ()))
    declared = {row['structureId'] for row in rows}
    from .surgical_scene import explicitly_protected
    for region in scene.regions:
        if not explicitly_protected(region):
            continue
        identity = region.get('structureId', region['name'])
        if identity not in declared:
            rows.append({'structureId': identity, 'cells': np.flatnonzero(scene.labels == region['id']).tolist(),
                'provenance': 'Explicit protected material-region declaration',
                'reviewStatus': region.get('reviewStatus', 'unreviewed'), 'displayName': region['name']})
            declared.add(identity)
    if not rows:
        return protection
    binding = grid_binding(scene)
    annotations = {} if protection is None else {a.structure_id: a for a in protection.annotations}
    changed = False
    for row in rows:
        annotation = ProtectionAnnotation(row['structureId'], tuple(row['cells']), binding,
            row['provenance'], row.get('reviewStatus', 'unreviewed'), row.get('displayName', ''))
        old = annotations.get(annotation.structure_id)
        if old is not None:
            if set(annotation.cells) <= set(old.cells):
                continue
            annotation = ProtectionAnnotation(old.structure_id, tuple(sorted(set(old.cells) | set(annotation.cells))),
                binding, old.provenance + '; ' + annotation.provenance,
                old.review_status if old.review_status == annotation.review_status else 'unreviewed', old.display_name)
        annotations[annotation.structure_id] = annotation
        changed = True
    if protection is not None and not changed:
        return protection
    return ProtectionSet(scene, annotations.values(),
        clearance_mm=protection.clearance_mm if protection is not None else 2.,
        required=protection.required if protection is not None else ('brainstem',))


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


def annotation_from_collection(scene, root, layer_id, structure_id, *, allow_unreviewed=False,
        slab_transfer=None, max_native_protection_cells=2_000_000, check=lambda: None):
    """Conservatively register an existing native layer; never generate anatomy.

    The default circumsphere path is unchanged. An explicit slab_transfer uses
    complete affine-cell support for a verified full inferior container, with
    assumed continuous rigid stresses and outside-source-grid unknown exclusion.
    Neither path grants source/anatomy/registration review. Full source
    memberships survive material priority.
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
    if slab_transfer is not None:
        import json
        from .surgical_protection_transfer import verify_inferior_slab, transfer_inferior_slab
        if (type(slab_transfer) is not dict or set(slab_transfer) != {
                'assumed_translation_mm', 'assumed_rotation_degrees', 'rotation_pivot_target_ras_mm'}):
            raise ValueError('Explicit complete assumed slab-transfer stress declaration required')
        if type(max_native_protection_cells) is not int or max_native_protection_cells < 1:
            raise ValueError('A positive native protection allocation limit is required')
        slab = verify_inferior_slab(collection.array(row['data']), label_value=row['label_value'], check=check)
        result = transfer_inferior_slab(slab, row['affine'], scene.shape, scene.affine,
            matrix, **slab_transfer, check=check)
        if result['report']['combinedProtectedCells'] > max_native_protection_cells:
            raise ValueError('Native protection cell budget exceeded before point/tuple allocation')
        provenance = json.dumps({'collection': collection.document['snapshot_id'],
            'layerId': layer_id, 'sourceArraySha256': row['data']['sha256'],
            'sourceLayer': row, 'transfer': result['report'],
            'meaning': 'Registered conservative exclusion container, not an anatomical segmentation; assumed stress bounds are not measured registration error'},
            sort_keys=True)
        return ProtectionAnnotation.from_mask(scene, structure_id, result['protectedMask'],
            provenance=provenance, review_status=row.get('review_status', 'unknown'),
            display_name=row.get('label', layer_id))
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
