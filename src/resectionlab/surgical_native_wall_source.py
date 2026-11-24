"""Open an exact native-wall source study without replacing a live rehearsal."""

from contextlib import ExitStack
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import tempfile
from time import perf_counter
from urllib.parse import unquote, urlparse
import uuid

import numpy as np

from .anatomy_local_wall_review import native_wall_alignment_review
from .anatomy_package import open_package
from .anatomy_geometry import _indexed_contour, appearance_kind, condition_open_vascular_contour, local_source_contour, normals
from .anatomy_sampling import native_field_bounds, sample_layer
from .core import thaw_json
from .patient_anatomy import PatientImagingCollection, file_hash, world
from .surgical_patient import _check_layer_export, _resample, _source_registration, _wall_layer_issue
from .surgical_patient_wall import SCHEMA, native_wall_grid


@dataclass
class SourceInspection:
    key: str
    collection: PatientImagingCollection
    manifest_sha: str
    storage: tempfile.TemporaryDirectory
    scans: dict
    originals: dict
    transfer_paths: set[str] = field(default_factory=set)

    @property
    def workspace(self):
        return Path(self.storage.name)


def _manifest_hash(path):
    with path.open('rb') as stream:
        contents = stream.read(16 * 1024**2 + 1)
    if len(contents) > 16 * 1024**2:
        raise ValueError('The selected native study manifest exceeds the metadata limit')
    return hashlib.sha256(contents).hexdigest()


def _check_original(collection, original, check):
    path = collection.path(original['path'])
    before = _fingerprint(path)
    if file_hash(path, check=check) != original['sha256'] or _fingerprint(path) != before:
        raise ValueError('The acquired source file differs from the saved rehearsal')
    return before


def _fingerprint(path):
    status = path.stat()
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns


def check_source_current(inspection, series_id):
    collection = inspection.collection
    if (_manifest_hash(collection.root / 'patient-anatomy.json') != inspection.manifest_sha
            or _fingerprint(collection.path(collection.series(series_id)['original']['path'])) != inspection.originals[series_id]):
        raise ValueError('The native study changed; reopen the source inspection')


def _source_manifest_hint(ref):
    uri = urlparse(ref.uri)
    path = Path(unquote(uri.path))
    if (uri.scheme == 'file' and uri.netloc in ('', 'localhost') and not uri.query and not uri.fragment
            and path.is_absolute() and path.name == 'patient-anatomy.json'):
        return path
    return None


def _native_field(scan, displayed, field_grid, field_shape, vessel_start):
    native_shape = scan['data']['shape']
    if vessel_start is not None:
        start, extent, partial = np.asarray(vessel_start), np.asarray(field_shape), False
    else:
        start, stop, partial = native_field_bounds(displayed, native_shape, field_grid, field_shape)
        extent = stop - start
    if np.any(extent < 2) or np.prod(extent, dtype=np.int64) > 2_000_000:
        return None
    cropped = displayed.copy()
    cropped[:3, 3] = world(displayed, [start])[0]
    return {'startIJK': start.tolist(), 'shape': extent.tolist(), 'nativeShape': native_shape,
            'affine': cropped.tolist(), 'partialCoverage': partial}


def _verify_study(case, adapter, wall, refs, collection, check):
    if (collection.identity() != adapter['sourceCollectionHash']
            or collection.document != adapter['sourceManifest']
            or collection.document['patient_id'] != adapter['sourcePatient']):
        raise ValueError('Choose the exact study revision used to prepare this rehearsal')
    reference = collection.series(collection.document['reference_series'])
    rows = {role: collection.layer(saved['layerId']) for role, saved in adapter['sourceLayers'].items()}
    brain, vessels = rows['brain'], rows['vessels']
    if (wall['layerId'], wall['sourceSeries'], wall['sourceMaskSha256'], wall['sourceScoresSha256']) != (
            vessels['id'], vessels['source_series'], vessels['data']['sha256'], vessels.get('probability', {}).get('sha256')):
        raise ValueError('The saved wall differs from its source layer and score field')
    scoped = adapter.get('nativeWallAlignmentReview')
    scans = {}
    frames = {}
    originals = {}
    for sid in dict.fromkeys([reference['id'], *(row['source_series'] for row in rows.values())]):
        check()
        scan = collection.series(sid)
        original = scan['original']
        ref = refs.get(sid)
        if not ref or not ref.sha256 == original['sha256'] or ref.provenance != 'observed' or ref.native_frame != 'RAS+':
            raise ValueError('The acquired source binding differs from the saved rehearsal')
        originals[sid] = _check_original(collection, original, check)
        transform, unreviewed = _source_registration(collection, scan)
        if unreviewed and (sid != vessels['source_series'] or not scoped):
            raise ValueError('The saved rehearsal has no checked correspondence to this source')
        frames[sid] = transform
        center = np.asarray(wall['centerRasMm'])
        if unreviewed:
            center = world(np.linalg.inv(transform), [center])[0]
        scans[sid] = {'seriesId': sid, 'modality': scan['modality'], 'centerRasMm': center.tolist(),
                      'independent': bool(unreviewed), 'layers': []}
    grid = native_wall_grid(collection, vessels, frames[vessels['source_series']], wall['centerRasMm'])
    if (wall['sourceShape'] != list(grid.source_shape) or wall['sourceStartIJK'] != grid.start.tolist()
            or wall['sourceImageSha256'] != grid.source_sha256 or not np.array_equal(grid.affine, case.affine)
            or not np.array_equal(grid.scores, case.vascular_probabilities.get('vessels'))):
        raise ValueError('The angiography samples differ from the saved rehearsal field')
    if scoped and scoped != native_wall_alignment_review(collection, brain, vessels, grid):
        raise ValueError('The alignment review differs from the saved rehearsal field')
    if not np.array_equal(case.mri, _resample(collection.array(reference['data']),
                          collection.patient_affine(reference), grid.affine, grid.mask.shape)):
        raise ValueError('The acquired reference samples differ from the saved rehearsal field')
    for role, row in rows.items():
        check()
        _check_layer_export(collection, role, row)
        if issue := _wall_layer_issue(role, row):
            raise ValueError(issue)
        saved = adapter['sourceLayers'][role]
        if (saved['sourceArray'], saved['sourceAffine'], saved['revision']) != (row['data'], row['affine'], row.get('revision', 0)):
            raise ValueError('A selected anatomy layer differs from the saved rehearsal')
        source_sid = row['source_series']
        if role == 'vessels':
            mask = grid.mask
        else:
            source_frame = frames[source_sid]
            mask, _ = sample_layer(collection, row, grid.affine, grid.mask.shape, check, {}, frames=(
                source_frame @ np.asarray(row['affine']), source_frame @ np.asarray(collection.series(source_sid)['affine'])))
        if not np.array_equal(mask, case.compartments.get(role)):
            raise ValueError('Anatomy samples differ from the saved rehearsal field')
        scans[source_sid]['layers'].append({'id': row['id'], 'kind': row['kind'], 'label': row['label']})
    for sid, scan in scans.items():
        series = collection.series(sid)
        native = np.asarray(series['affine'])
        displayed = native if scan['independent'] else frames[sid] @ native
        start = grid.start if sid == vessels['source_series'] else None
        scan['localField'] = _native_field(series, displayed, grid.affine, grid.mask.shape, start)
    return scans, originals


def local_source_view(inspection, scan, snapshot_id, transfer, check, progress):
    collection = inspection.collection
    if snapshot_id != collection.identity():
        raise ValueError('The native study changed; reopen the source inspection')
    check_source_current(inspection, scan['seriesId'])
    field = scan['localField']
    if not field:
        raise ValueError('The saved field is unavailable in this native grid; open the full acquisition')
    series = collection.series(scan['seriesId'])
    shape = tuple(field['shape'])
    displayed = np.asarray(field['affine'])
    start = np.asarray(field['startIJK'])
    source_slices = tuple(slice(a, a + length) for a, length in zip(start, shape))
    independent = scan['independent']
    registration = np.eye(4) if independent else collection.registration_matrix(series['id'], require_geometry=True)
    full_affine = registration @ np.asarray(series['affine'])
    paths = set()

    def asset(data, dtype):
        descriptor = transfer(data, dtype)
        paths.add(descriptor['path'])
        return descriptor

    progress(.15, 'Reading the acquired native samples at the saved field')
    check()
    signal = asset(collection.array(series['data'], check=check)[source_slices], 'float32')
    layers = []
    for selected in scan['layers']:
        check()
        row = collection.layer(selected['id'])
        layer_affine = registration @ np.asarray(row['affine'])
        mask, coverage = sample_layer(collection, row, displayed, shape, check, {}, frames=(layer_affine, full_affine))
        started = perf_counter()
        faces = np.empty((0, 3), np.uint32)
        display_vertices = None
        diagnostics = {'policy': 'native-open-field-v1', 'kind': 'context' if row['kind'] == 'brain' else 'vessel',
                       'maxDisplacementMm': 0.}
        contour_field = 'limited-coverage'
        if coverage.all():
            if row['data']['shape'] == series['data']['shape'] and np.allclose(layer_affine, full_affine, rtol=0, atol=1e-5):
                points, faces, contour_field = local_source_contour(collection, row, source_slices)
                transform = np.linalg.solve(displayed, layer_affine)
                vertices = world(transform, points + start).astype(np.float32)
                if len(faces) and row['kind'] in {'vessel', 'vessels', 'artery', 'vein', 'sinus'} and contour_field == 'native-probability':
                    conditioned, diagnostics = condition_open_vascular_contour(points, faces, mask, layer_affine,
                                                                               cancelled=lambda: bool(check()))
                    if diagnostics['maxDisplacementMm'] > 0:
                        display_vertices = world(transform, conditioned + start).astype(np.float32)
                if np.linalg.det(transform[:3, :3]) < 0:
                    faces = faces[:, ::-1]
            else:
                # Independently sampled anatomy from a different grid uses its aligned label in this view.
                vertices, faces, _ = _indexed_contour(mask.astype(np.uint8), probability=False)
                contour_field = 'aligned-categorical'
        geometry = {'empty': True}
        if len(faces):
            vertex_asset = asset(vertices, 'float32')
            normal_asset = asset(normals(vertices, faces), 'float32')
            geometry = {'positions': asset(display_vertices, 'float32') if display_vertices is not None else vertex_asset,
                        'source': vertex_asset, 'indices': asset(faces, 'uint32'), 'sourceNormals': normal_asset,
                        'normals': asset(normals(display_vertices, faces), 'float32') if display_vertices is not None else normal_asset}
        layers.append({'name': row['id'], 'label': row['label'], 'kind': row['kind'], 'displayKind': appearance_kind(row),
                       'color': row['color'], 'sourceSeries': series['id'], 'mask': asset(mask, 'uint8'),
                       'alignment': 'independent' if independent else collection.document['transforms'][series['id']]['status'],
                       'surface': geometry, 'surfaceDiagnostics': {**diagnostics,
                           'surfaceField': contour_field, 'sourceSurfaceOnly': display_vertices is None,
                           'triangles': len(faces), 'preparationMs': (perf_counter() - started) * 1000}})
    check_source_current(inspection, scan['seriesId'])
    check()
    inspection.transfer_paths = paths
    return {'snapshotId': snapshot_id, 'caseId': collection.document['patient_id'],
            'caseHash': 'sha256:' + hashlib.sha256(f"{snapshot_id}:{series['id']}:local:{field['startIJK']}:{shape}".encode()).hexdigest(),
            'frame': 'RAS+', 'shape': field['shape'], 'affine': field['affine'], 'mri': signal, 'sourceField': field,
            'compartments': layers, 'modality': scan['modality'], 'unavailableLayers': [], 'alignmentPreviews': [],
            'displayPreview': None, 'patientFrameReady': not independent,
            'registrationStatus': 'independent source frame' if independent else 'patient frame'}


def open_source_inspection(session, case, selected, check):
    adapter = thaw_json(case.metadata.get('surgical_adapter') or {})
    wall = adapter.get('nativeWall') or {}
    refs = {ref.source_id: ref for ref in case.source_refs}
    manifest_ref = refs.get('source-manifest')
    if (adapter.get('type') != 'native-patient-research-derivative' or wall.get('schema') != SCHEMA
            or len(refs) != len(case.source_refs) or not manifest_ref or manifest_ref.provenance != 'observed'
            or not manifest_ref.sha256 or manifest_ref.sha256 != adapter.get('sourceManifestSha256')):
        raise ValueError('This rehearsal has no verified native-wall source binding')
    path = Path(selected) if selected else _source_manifest_hint(manifest_ref)
    if path is None or (not selected and not path.is_file()):
        return None
    if (not path.is_absolute() or not path.is_file()
            or not (path.name == 'patient-anatomy.json' or path.suffix == '.rsworkspace')):
        raise ValueError('Choose the saved patient-anatomy.json or portable study')
    try:
        with ExitStack() as cleanup:
            storage = tempfile.TemporaryDirectory(prefix='native-wall-source-', dir=session.transfers.root)
            cleanup.callback(storage.cleanup)
            workspace = Path(storage.name)
            manifest = open_package(path, workspace / 'study', check=check) if path.suffix == '.rsworkspace' else path
            if _manifest_hash(manifest) != manifest_ref.sha256:
                raise ValueError('Choose the exact study revision used to prepare this rehearsal')
            collection = PatientImagingCollection(manifest.parent)
            scans, originals = _verify_study(case, adapter, wall, refs, collection, check)
            if _manifest_hash(manifest) != manifest_ref.sha256:
                raise ValueError('The native study changed during source verification')
            check()
            current = getattr(session, 'native_wall_source', None)
            if current is not None:
                close_source_inspection(session, current.key)
            key = str(uuid.uuid4())
            session.native_wall_source = SourceInspection(key, collection, manifest_ref.sha256, storage, scans, originals)
            cleanup.pop_all()
    except (OSError, ValueError) as error:
        if selected:
            raise ValueError(str(error)) from error
        return None
    return {'collectionId': key, 'snapshotId': collection.identity(), 'caseHash': case.semantic_hash,
            'centerRasMm': wall['centerRasMm'], 'cropStartIJK': wall['sourceStartIJK'], 'cropShape': list(case.mri.shape),
            'locallyReviewed': bool(adapter.get('nativeWallAlignmentReview')), 'scans': list(scans.values())}


def close_source_inspection(session, key):
    inspection = getattr(session, 'native_wall_source', None)
    if inspection and inspection.key == key:
        session.native_wall_source = None
        cache = getattr(session, 'anatomy_view_cache', None)
        if cache and cache['key'][0][0] == key:
            session.anatomy_view_cache = None
            session.anatomy_transfer_paths.clear()
        inspection.storage.cleanup()
