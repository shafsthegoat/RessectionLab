"""Read-only native acquisition field display shared by Anatomy and rehearsal."""

import hashlib
from time import perf_counter

import numpy as np

from .anatomy_geometry import _indexed_contour, appearance_kind, condition_open_vascular_contour, local_source_contour, normals
from .anatomy_sampling import sample_layer
from .patient_anatomy import affine, projection_or_localizer, world


def source_field_view(collection, series, field, layer_ids, independent, transfer, check, progress):
    snapshot_id = collection.identity()
    shape = tuple(field['shape'])
    displayed = np.asarray(field['affine'])
    start = np.asarray(field['startIJK'])
    source_slices = tuple(slice(a, a + length) for a, length in zip(start, shape))
    registration = np.eye(4) if independent else collection.registration_matrix(series['id'], require_geometry=True)
    full_affine = registration @ np.asarray(series['affine'])
    paths = set()

    def asset(data, dtype):
        descriptor = transfer(data, dtype)
        paths.add(descriptor['path'])
        return descriptor

    progress(.15, 'Reading the acquired native samples in the local field')
    check()
    signal = asset(collection.array(series['data'], check=check)[source_slices], 'float32')
    layers = []
    for layer_id in layer_ids:
        check()
        row = collection.layer(layer_id)
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
                crop_affine = full_affine.copy()
                crop_affine[:3, 3] = world(full_affine, [start])[0]
                exact_grid = np.array_equal(layer_affine, full_affine) and np.array_equal(displayed, crop_affine)
                points, faces, contour_field = local_source_contour(
                    collection, row, source_slices, mask=mask if exact_grid else None, check=check)
                transform = np.linalg.solve(displayed, layer_affine)
                vertices = world(transform, points + start).astype(np.float32)
                if len(faces) and row['kind'] in {'vessel', 'vessels', 'artery', 'vein', 'sinus'} and contour_field in {'native-probability', 'filtered-native-probability'}:
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
    view = {'snapshotId': snapshot_id, 'caseId': collection.document['patient_id'],
            'caseHash': 'sha256:' + hashlib.sha256(f"{snapshot_id}:{series['id']}:local:{field['startIJK']}:{shape}".encode()).hexdigest(),
            'frame': 'RAS+', 'shape': field['shape'], 'affine': field['affine'], 'mri': signal, 'sourceField': field,
            'compartments': layers, 'modality': series['modality'], 'unavailableLayers': [], 'alignmentPreviews': [],
            'displayPreview': None, 'patientFrameReady': not independent,
            'registrationStatus': 'independent source frame' if independent else 'patient frame'}
    return view, paths


def anatomy_vessel_field(collection, series_id, selection, transfer, check, progress):
    """Open a bounded score layer from the current Anatomy scan and cursor."""
    from .anatomy_display import display_affine
    from .anatomy_geometry import vascular_cleanup_source
    from .anatomy_model_registry import ANGIOGRAPHY, source_volume

    if (not isinstance(selection, dict) or set(selection) != {'layerId', 'center', 'frame'}
            or not isinstance(selection['layerId'], str) or not isinstance(selection['frame'], str)):
        raise ValueError('Choose a scored vessel layer and its linked source cursor')
    center = selection['center']
    if not isinstance(center, list) or len(center) != 3 or any(type(v) not in (float, int) or not np.isfinite(v) for v in center):
        raise ValueError('Choose a finite source cursor')
    series = collection.series(series_id)
    row = collection.layer(selection['layerId'])
    if (row['kind'] not in {'vessel', 'vessels', 'artery', 'vein', 'sinus'} or row['source_series'] != series_id
            or series['modality'] not in ANGIOGRAPHY or not source_volume(series) or projection_or_localizer(series)
            or row.get('edits') or row.get('publisher_surface') or not (row.get('probability') or vascular_cleanup_source(collection, row))):
        raise ValueError('Select a native angiography vessel score layer on its active acquisition')
    collection.require_native_geometry(series)
    collection.require_native_geometry(row)
    native = affine(series['affine'])
    if row['data']['shape'] != series['data']['shape'] or not np.allclose(row['affine'], native, rtol=0, atol=1e-5):
        raise ValueError('The vessel layer does not share the native acquisition grid')
    frame = selection['frame']
    if frame == 'independent source frame':
        displayed, independent = native, True
    elif frame in {'alignment preview', 'patient frame'}:
        displayed, preview = display_affine(collection, series, preview=frame == 'alignment preview')
        if preview != (frame == 'alignment preview'):
            raise ValueError('The linked scan alignment changed; reopen the field')
        independent = preview
    else:
        raise ValueError('The source scan display is still loading')
    ijk = world(np.linalg.inv(displayed), [center])[0]
    native_shape = np.asarray(series['data']['shape'])
    if np.any(ijk < -.5) or np.any(ijk >= native_shape - .5):
        raise ValueError('The linked cursor is outside the acquired scan')
    spacing = np.linalg.norm(native[:3, :3], axis=0)
    half = np.ceil(12. / spacing).astype(int)
    anchor = np.clip(np.rint(ijk).astype(int), 0, native_shape - 1)
    start, stop = np.maximum(0, anchor - half), np.minimum(native_shape, anchor + half + 1)
    shape = stop - start
    if np.any(shape < 2) or np.prod(shape, dtype=np.int64) > 2_000_000:
        raise ValueError('This acquisition exceeds the local field budget')
    coordinates = native if independent else displayed
    cropped = coordinates.copy()
    cropped[:3, 3] = world(coordinates, [start])[0]
    field = {'startIJK': start.tolist(), 'shape': shape.tolist(), 'nativeShape': native_shape.tolist(),
             'affine': cropped.tolist(), 'partialCoverage': bool(np.any(shape < half * 2 + 1))}
    view, paths = source_field_view(collection, series, field, [row['id']], independent, transfer, check, progress)
    view['sourceFieldCursorRasMm'] = world(coordinates, [ijk])[0].tolist()
    return view, paths
