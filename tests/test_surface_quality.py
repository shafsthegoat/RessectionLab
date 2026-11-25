import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from skimage.measure import marching_cubes
from resectionlab.surface_fairing import fair_surface, intersection_vertices, POLICY
from resectionlab.anatomy_qa import mask_quality


@pytest.mark.parametrize('kind', ['tumor', 'vessel', 'context'])
def test_native_worker_policy_parity(kind):
    node = shutil.which('node')
    if not node: pytest.skip('Node required for cross-stack display parity')
    grid = np.indices((10, 10, 10))
    mask = ((grid-4.5)**2).sum(0) < 12
    points, faces, _, _ = marching_cubes(mask.astype(float), .5)
    affine = np.array([[-.7, .2, 0, 100], [0, 1.3, .2, -20], [0, 0, 2, 40], [0, 0, 0, 1.]])
    expected, diagnostics = fair_surface(points, faces, affine, kind)
    data = {'points': points.tolist(), 'faces': faces.ravel().tolist(), 'affine': affine.tolist(), 'kind': kind}
    code = "import {fairIndexedSurface} from './desktop/src/viewer/surfaceSmoothing.ts';let data='';for await(const c of process.stdin)data+=c;const d=JSON.parse(data);const s=fairIndexedSurface(Float32Array.from(d.points.flat()),Uint32Array.from(d.faces),d.affine,d.kind);process.stdout.write(JSON.stringify({points:[...s.positions],diagnostics:s.diagnostics}));"
    process = subprocess.run([node, '--experimental-strip-types', '--input-type=module', '-e', code], input=json.dumps(data), text=True, capture_output=True, cwd=Path(__file__).resolve().parents[1], check=True)
    actual = json.loads(process.stdout)
    np.testing.assert_allclose(np.array(actual['points']).reshape(-1, 3), expected, atol=2e-6, rtol=0)
    assert actual['diagnostics']['policy'] == POLICY['version']
    assert actual['diagnostics']['restoredComponents'] == diagnostics['restoredComponents']


def test_source_quality_retains_empty_fragmented_and_truncated_evidence():
    mask = np.zeros((8, 8, 8), bool); mask[0, 1, 1] = True; mask[5, 5, 5] = True
    before = mask.copy(); q = mask_quality(mask, 'bound-source', probability=np.zeros_like(mask, float), alignment_status='unreviewed')
    assert q['componentCount'] == 2 and q['boundaryTouchingVoxels'] == 1
    assert q['probabilityMismatchVoxels'] == 2 and q['accuracy'] == 'unmeasured'
    np.testing.assert_array_equal(before, mask)
    empty = mask_quality(np.zeros_like(mask), 'empty-source')
    assert empty['voxelCount'] == empty['componentCount'] == 0 and empty['issues']


def test_physical_intersection_gate_detects_only_new_contacts():
    points = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [.5, .5, -1], [.5, .5, 1], [1.5, .5, 0.]])
    faces = np.arange(6).reshape(2, 3)
    assert not len(intersection_vertices(points, points, faces))
    source = points.copy(); source[3:, 0] += 10
    assert len(intersection_vertices(source, points, faces)) == 6


def test_open_native_vessel_fairing_preserves_the_rim_collar_and_tiny_source_crossings():
    from scipy import ndimage as ndi
    from resectionlab.anatomy_geometry import _indexed_contour, condition_open_vascular_contour, vascular_mesh_source_cells
    from resectionlab.patient_anatomy import world

    x, y, z = np.indices((27, 25, 25), dtype=float)
    radius = np.hypot(y - 8. - .14 * x, z - 8.)
    scores = (.5 + np.clip((3.1 + .35 * np.sin(.9 * x) - radius) * .22, -.4, .4)).astype(np.float32)
    scores[13:15, 20, 19] = [.68, .59]
    mask = scores >= .5
    original_scores = scores.copy()
    source, faces, _ = _indexed_contour(scores, probability=True)
    affine = np.diag([-.55, .62, .8, 1.])
    display, diagnostic = condition_open_vascular_contour(source, faces, mask, affine)

    edges, uses = np.unique(np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0, return_counts=True)
    collar = np.isin(np.arange(len(source)), edges[uses == 1])
    assert collar.any()
    for _ in range(2):
        collar[edges[collar[edges].any(axis=1)].ravel()] = True
    np.testing.assert_array_equal(display[collar], source[collar])
    labels, _ = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    owners = vascular_mesh_source_cells(source, mask)
    tiny = labels.ravel()[owners] == labels[13, 20, 19]
    assert tiny.any()
    np.testing.assert_array_equal(display[tiny], source[tiny])
    np.testing.assert_array_equal(scores, original_scores)
    np.testing.assert_array_equal(mask, original_scores >= .5)
    assert diagnostic['policy'] == 'native-bounded-open-vessel-v3'
    assert 0 < diagnostic['maxDisplacementMm'] <= .4 * .7 * .55
    assert not len(intersection_vertices(world(affine, source), world(affine, display), faces))


def test_section_containment_requires_the_entire_void_ring():
    from resectionlab.vessel_section_stability import _section_roles

    points = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2],
                       [1, .4], [3, .4], [3, 1], [1, 1],
                       [-1.5, .4], [-.5, .4], [-.5, 1], [-1.5, 1]])
    rings = [list(range(i, i + 4)) for i in (0, 4, 8)]
    roles, hole_count = _section_roles(rings, points)
    assert roles.tolist() == [1] * 4 + [0] * 4 + [2] * 4
    assert hole_count == 1


def test_section_records_when_all_edges_of_an_enclosed_void_disappear():
    from resectionlab.vessel_section_stability import Section, _changed_section

    keys = np.arange(8)
    edges = np.column_stack((keys, keys + 8))
    source = Section(edges, keys, np.array([1] * 4 + [2] * 4), 1)
    closed = Section(edges[:4], keys[:4], np.ones(4, dtype=int), 0)

    assert _changed_section(source, closed)
    assert _changed_section(closed, source)


def test_open_section_guard_restores_a_merged_junction_without_reverting_other_moves(monkeypatch):
    from resectionlab import vessel_section_stability as section
    from resectionlab.anatomy_geometry import _indexed_contour

    x, y, z = np.indices((25, 25, 21), dtype=float)
    first, second = [2.4 - np.hypot(x - axis, y - 12) for axis in (7, 17)]
    bridge = np.minimum.reduce((1.2 - np.abs(y - 12), 5.1 - np.abs(x - 12), z - 10.3))
    scores = (.5 + np.clip(np.maximum.reduce((first, second, bridge)) * .2, -.4, .4)).astype(np.float32)
    source, faces, _ = _indexed_contour(scores, probability=True)
    candidate = source.copy()
    candidate[np.abs(source[:, 2] - 10.3) < .45, 2] -= .3
    elsewhere = (source[:, 2] > 14.) & (source[:, 2] < 17.)
    candidate[elsewhere, 1] += .08
    plane = np.array([7., 12., 10.15]), np.array([0., 0., 1.])
    before, _ = section._cut(source, faces, *plane)
    merged, _ = section._cut(candidate, faces, *plane)
    assert section._changed_section(before, merged)
    monkeypatch.setattr(section, '_source_stations', lambda *_: [plane])
    guarded, diagnostic = section.preserve_open_sections(source, candidate, faces, scores >= .5, np.eye(4), cancelled=lambda: False)
    after, _ = section._cut(guarded, faces, *plane)
    assert not section._changed_section(before, after)
    assert diagnostic['sectionCorrections'] == 1 and diagnostic['sectionRestoredVertices'] > 0
    np.testing.assert_array_equal(guarded[elsewhere], candidate[elsewhere])


def test_native_vascular_wall_reduces_caliber_error_and_keeps_source_owners():
    pytest.importorskip("vtkmodules")
    from resectionlab.anatomy_geometry import (
        conditioned_vascular_probability_mesh, vascular_mesh_source_cells, vascular_probability_mesh,
    )
    from resectionlab.patient_anatomy import world

    affine = np.diag([.55, .62, .8, 1.])
    indices = np.indices((37, 35, 29)).transpose(1, 2, 3, 0)
    grid = world(affine, indices)
    origin = np.array([0., 7.5, 7.])
    direction = np.array([1., .26, .19])
    direction /= np.linalg.norm(direction)

    def radial_geometry(positions):
        delta = positions - origin
        axial = delta @ direction
        radius = np.linalg.norm(delta - axial[..., None] * direction, axis=-1)
        return axial, radius

    axial, radius = radial_geometry(grid)
    # The acquired-grid score represents a known cylinder with periodic radius
    # error. A second strand is one sample wide and ends at the same cut plane.
    shaft = (1.4 - radius) / .9 + .13 * np.sin(2.4 * axial)
    thin = (.36 - np.hypot(grid[..., 1] - 3.1, grid[..., 2] - 3.2)) / .9
    scores = (.5 + np.clip(np.maximum(shaft, thin), -.47, .47)).astype(np.float32)
    mask = scores >= .5
    baseline_scores = scores.copy()
    wall = conditioned_vascular_probability_mesh(scores, mask, affine)
    raw, faces, _ = vascular_probability_mesh(scores, mask)
    np.testing.assert_array_equal(wall.source, raw)
    np.testing.assert_array_equal(wall.indices, faces)
    np.testing.assert_array_equal(wall.source_cells, vascular_mesh_source_cells(raw, mask))
    np.testing.assert_array_equal(scores, baseline_scores)
    np.testing.assert_array_equal(mask, scores >= .5)
    assert not wall.source.flags.writeable and not wall.positions.flags.writeable
    boundary = ((raw <= 0) | (raw >= np.asarray(mask.shape) - 1)).any(axis=1)
    np.testing.assert_array_equal(wall.positions[boundary], raw[boundary])
    thin_vertices = raw[:, 1] < 8
    shaft_interior = ~thin_vertices & (raw[:, 0] > 3) & (raw[:, 0] < 33)

    def radius_error(positions):
        _, distance = radial_geometry(world(affine, positions[shaft_interior]))
        return np.sqrt(np.mean((distance - 1.4) ** 2))

    assert radius_error(wall.positions) < .85 * radius_error(raw)
    assert 0 < wall.diagnostics["maxDisplacementMm"] <= .7 * .55
    assert wall.diagnostics["restoredComponents"] == 0
    assert not len(intersection_vertices(world(affine, raw), world(affine, wall.positions), faces))
    # Crossings owned by the thin strand survive on both sides of its axis.
    assert ((wall.positions[thin_vertices, 1] < 5) & (raw[thin_vertices, 0] > 3)).any()
    assert ((wall.positions[thin_vertices, 1] > 5) & (raw[thin_vertices, 0] > 3)).any()


def test_packaged_policy_copy_matches_canonical_resource():
    desktop = Path(__file__).resolve().parents[1]/'desktop/src/viewer/surface_policy.json'
    assert json.loads(desktop.read_text()) == POLICY


def test_native_wall_retains_only_complete_isolated_source_components():
    pytest.importorskip('vtkmodules')
    from resectionlab.anatomy_geometry import conditioned_vascular_probability_mesh

    shape = (25, 25, 23)
    _, y, z = np.ogrid[:25, :25, :23]
    scores = np.broadcast_to(np.clip(.5 + (2.45 - np.hypot(y - 12, z - 12)) / 4, 0, 1), shape).astype(np.float32)
    scores[6:10, 3:6, 5:8] = .42
    scores[7, 4, 6], scores[8, 4, 6] = .67, .59
    scores[7, 3, 6], scores[7, 5, 6] = .39, .47
    scores[:3, 19:22, 12:15] = .42
    scores[0, 20, 13], scores[1, 20, 13] = .67, .59
    mask = scores >= .5
    source_scores, source_mask = scores.copy(), mask.copy()
    affine = np.diag([.30, .30, .45, 1.])
    previous = conditioned_vascular_probability_mesh(scores, mask, affine)
    current = conditioned_vascular_probability_mesh(scores, mask, affine, retain_isolated_components=True)
    owners = np.ravel_multi_index(([7, 8], [4, 4], [6, 6]), shape)
    isolated = np.isin(previous.source_cells, owners)
    assert isolated.any() and np.any(previous.source[isolated] != previous.positions[isolated])
    assert current.diagnostics['sourceRetainedIsolatedComponents'] == 1
    np.testing.assert_array_equal(current.positions[isolated], previous.source[isolated])
    np.testing.assert_array_equal(current.positions[~isolated], previous.positions[~isolated])
    for name in ('source', 'indices', 'source_cells'):
        np.testing.assert_array_equal(getattr(current, name), getattr(previous, name))
    np.testing.assert_array_equal(scores, source_scores)
    np.testing.assert_array_equal(mask, source_mask)

    # A third original foreground sample was removed. The resulting pair
    # must keep the accepted edited-wall conditioning.
    scores[9, 4, 6] = .63
    edited_previous = conditioned_vascular_probability_mesh(scores, mask, affine)
    edited_current = conditioned_vascular_probability_mesh(
        scores, mask, affine, retain_isolated_components=True, native_mask=scores >= .5,
    )
    np.testing.assert_array_equal(edited_current.positions, edited_previous.positions)


def test_publisher_surface_retains_geometry_after_portable_reopen(tmp_path):
    import nibabel as nib
    from resectionlab.anatomy_geometry import mesh
    from resectionlab.anatomy_package import open_package, save_package
    from resectionlab.anatomy_publisher_surface import POLICY, publisher_surface
    from resectionlab.patient_anatomy import PatientImagingCollection, world

    matrix = np.diag([-0.5, 1.25, 2., 1.])
    matrix[:3, 3] = [100., -40., 20.]
    values = np.zeros((6, 6, 6), np.uint8)
    values[1:4, 1:4, 1:4] = 1
    source = tmp_path / 'source.nii.gz'
    image = nib.Nifti1Image(values, matrix)
    image.header.set_xyzt_units('mm')
    nib.save(image, source)
    raw_asset = tmp_path / 'mesh.vtp'
    raw_asset.write_bytes(b'synthetic publisher asset test')
    c = PatientImagingCollection.create(tmp_path / 'study', 'publisher-pilot')
    scan = c.add_series(source, series_id='cta', modality='CTA')
    row = c.add_layer(values, matrix, layer_id='arterial', source_series='cta', kind='artery',
                      role='context', color='#aabbcc')
    vertices = world(matrix, [[1., 1., 1.], [3., 1., 1.], [1., 3., 2.]]).astype(np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.uint32)
    row['publisher_surface'] = {
        'schema': POLICY, 'original': c.original(raw_asset),
        'vertices_ras_mm': c.put_array(vertices), 'triangles': c.put_array(triangles),
        'source_series': 'cta', 'source_sha256': scan['original']['sha256'], 'source_affine': scan['affine'],
        'mask_sha256': row['data']['sha256'], 'mask_label_value': 1, 'mask_revision': 0,
    }
    c.save()
    archive = tmp_path / 'roundtrip.rsworkspace'
    save_package(c, archive)
    restored = PatientImagingCollection(open_package(archive, tmp_path / 'reopen').parent)
    layer = restored.layer(row['id'])
    actual = mesh(restored, layer)
    np.testing.assert_array_equal(world(matrix, actual['positions']), vertices)
    np.testing.assert_array_equal(actual['source'], actual['positions'])
    np.testing.assert_array_equal(actual['indices'], triangles[:, ::-1])
    np.testing.assert_array_equal(restored.mask(layer), values.astype(bool))
    assert actual['policy_version'] == POLICY and actual['max_displacement_mm'] == 0.

    assert publisher_surface(restored, {**layer, 'revision': 1}, cancelled=lambda: False) is None
    assert publisher_surface(restored, {**layer, 'edits': [{'draft': True}]}, cancelled=lambda: False) is None
    changed = {**layer, 'publisher_surface': {**layer['publisher_surface'], 'source_sha256': 'a' * 64}}
    with pytest.raises(ValueError, match='binding differ'):
        mesh(restored, changed)
    restored.path(restored.series('cta')['original']['path']).write_bytes(b'changed acquired source')
    with pytest.raises(ValueError, match='acquired source checksum differs'):
        mesh(restored, layer)


def test_publisher_cta_fairing_uses_one_exact_full_mesh_for_named_partitions(tmp_path):
    import nibabel as nib
    from resectionlab.anatomy_geometry import _indexed_contour, mesh
    from resectionlab.anatomy_publisher_surface import POLICY
    from resectionlab.patient_anatomy import PatientImagingCollection, world
    from resectionlab.publisher_vessel_fairing import POLICY as GUIDED_POLICY
    from resectionlab.surface_fairing import PUBLISHER_CTA_POLICY

    frame = np.diag([-.5, .5, .625, 1.])
    frame[:3, 3] = [12., -18., 25.]
    shape = (24, 24, 24)
    x, y, z = np.indices(shape)
    mask = ((x - 10 + .17 * z)**2 + (y - 13 + .35 * z)**2 < 3.7**2) & (z > 2) & (z < 21)
    source = tmp_path / 'cta.nii'
    image = nib.Nifti1Image(np.where(mask, 450., 40.).astype(np.float32), frame)
    image.header.set_xyzt_units('mm')
    nib.save(image, source)
    asset = tmp_path / 'source.vtp'
    asset.write_bytes(b'source shape for the portable partition control')
    collection = PatientImagingCollection.create(tmp_path / 'study', 'publisher-fairing-control')
    scan = collection.add_series(source, series_id='cta', modality='CTA')
    raw, triangles, _ = _indexed_contour(mask.astype(np.uint8), False)
    exact = world(frame, raw).astype(np.float32)
    labels = np.where(mask, np.where(z > 11, 2, 1), 0).astype(np.uint8)

    def add(layer_id, values, label, points, indices):
        row = collection.add_layer(values, scan['affine'], layer_id=layer_id, source_series='cta',
                                   kind='artery', role='context', color='#ef7968', label_value=label)
        row['publisher_surface'] = {
            'schema': POLICY, 'original': collection.original(asset), 'vertices_ras_mm': collection.put_array(points),
            'triangles': collection.put_array(indices), 'source_series': 'cta', 'source_sha256': scan['original']['sha256'],
            'source_affine': scan['affine'], 'mask_sha256': row['data']['sha256'],
            'mask_label_value': label, 'mask_revision': 0,
        }
        return row

    whole = add('whole', mask.astype(np.uint8), 1, exact, triangles)
    assignments = raw[triangles, 2].mean(axis=1) > 11
    children = []
    for side in (False, True):
        faces = np.flatnonzero(assignments == side).astype(np.uint32)
        vertices, indices = np.unique(triangles[faces], return_inverse=True)
        member = add(f'branch-{int(side)}', labels, 1+int(side), exact[vertices], indices.reshape(-1, 3).astype(np.uint32))
        member['publisher_surface'].update(publisher_vertex_ids=collection.put_array(vertices),
                                           publisher_face_ids=collection.put_array(faces))
        children.append((member, vertices, faces))
    collection.save()
    assert np.intersect1d(children[0][1], children[1][1]).size > 0
    actual = mesh(collection, whole)
    source_only = mesh(collection, whole, conditioned=False)
    np.testing.assert_array_equal(actual['source'], source_only['positions'])
    np.testing.assert_array_equal(actual['indices'], triangles[:, ::-1])
    assert actual['max_displacement_mm'] > 0 and actual['policy_version'] == PUBLISHER_CTA_POLICY
    assert source_only['policy_version'] == POLICY
    for member, vertices, faces in children:
        displayed = mesh(collection, member)
        assert displayed['cache_hit'] and displayed['policy_version'] == PUBLISHER_CTA_POLICY
        np.testing.assert_array_equal(displayed['positions'], actual['positions'][vertices])
        np.testing.assert_array_equal(displayed['normals'], actual['normals'][vertices])
        np.testing.assert_array_equal(vertices[displayed['indices']], actual['indices'][faces])
        np.testing.assert_array_equal(source_only['positions'][vertices], mesh(collection, member, conditioned=False)['positions'])
    member, _, faces = children[0]
    bogus = {**member, 'publisher_surface': {**member['publisher_surface'],
                                           'publisher_face_ids': collection.put_array(np.roll(faces, 1))}}
    assert mesh(collection, bogus)['policy_version'] == POLICY

    neighboring = np.zeros(shape, np.uint8)
    neighboring[16, 13, 6:19] = 1
    vein = collection.add_layer(neighboring, scan['affine'], layer_id='nearby-vein', source_series='cta',
                                kind='vein', role='context', color='#609fdb')
    path = collection.path(vein['data']['path'])
    moved_path = path.with_suffix('.temporarily-unavailable')
    path.rename(moved_path)
    fallback = mesh(collection, whole)
    np.testing.assert_array_equal(fallback['positions'], actual['positions'])
    assert fallback['policy_version'] == PUBLISHER_CTA_POLICY
    moved_path.rename(path)

    refined = mesh(collection, whole)
    assert refined['policy_version'] == GUIDED_POLICY
    assert not np.array_equal(refined['positions'], actual['positions'])
    np.testing.assert_array_equal(refined['source'], source_only['positions'])
    np.testing.assert_array_equal(refined['indices'], actual['indices'])
    for member, vertices, faces in children:
        displayed = mesh(collection, member)
        assert displayed['cache_hit'] and displayed['policy_version'] == GUIDED_POLICY
        np.testing.assert_array_equal(displayed['positions'], refined['positions'][vertices])
        np.testing.assert_array_equal(displayed['normals'], refined['normals'][vertices])
        np.testing.assert_array_equal(vertices[displayed['indices']], refined['indices'][faces])


def test_empty_native_mesh_has_explicit_descriptor_without_zero_length_transfer(tmp_path):
    import nibabel as nib
    from resectionlab.patient_anatomy import PatientImagingCollection
    from resectionlab.desktop_bridge import BridgeSession, _Request
    source = tmp_path / 'source.nii.gz'
    empty = tmp_path / 'empty.nii.gz'
    for path, values in ((source, np.ones((6, 6, 6), np.float32)), (empty, np.zeros((6, 6, 6), np.uint8))):
        image = nib.Nifti1Image(values, np.eye(4))
        image.header.set_xyzt_units('mm')
        nib.save(image, path)
    c=PatientImagingCollection.create(tmp_path/'native','missing-target');c.add_series(source,series_id='t1',modality='T1');c.add_layer(np.zeros((6,6,6),np.uint8),np.eye(4),layer_id='empty',source_series='t1',kind='tumor',role='target',color='#aabbcc');c.save()
    bridge=BridgeSession(tmp_path/'transfers');request=_Request('empty');progress=lambda *_:None
    opened=bridge.execute('anatomyOpen',{'path':str(c.root/'patient-anatomy.json')},request,progress)
    view=bridge.execute('anatomyView',{'collectionId':opened['collectionId'],'snapshotId':c.identity(),'seriesId':'t1','layers':['empty']},request,progress)
    assert view['compartments'][0]['surface']=={'empty':True}
    assert view['compartments'][0]['qa']['voxelCount']==0


def test_native_anisotropic_context_moves_across_coarse_slices_with_bounded_topology(monkeypatch):
    from resectionlab.anatomy_geometry import _arrays, _poly
    from resectionlab.surface_fairing import NATIVE_POLICY

    grid = np.indices((33, 35, 15))
    mask = (((grid - np.array([16, 17, 7])[:, None, None, None])
             / np.array([11, 13, 5])[:, None, None, None]) ** 2).sum(0) < 1
    # The surface's opening and disconnected thin fragment exercise the
    # unchanged categorical connectivity and the independent contact audit.
    mask[15:17, 17, 9:13] = False
    mask[2, 4:8, 4] = True
    points, faces = _arrays(_poly(mask.astype(np.uint8), False))
    before = faces.copy()
    affine = np.diag([1.2, 1.2, 4., 1.])
    positions, diagnostics = fair_surface(points, faces, affine, 'brain', native=True)
    displacement = (positions - points) @ affine[:3, :3].T
    assert np.abs(displacement[:, 2]).max() > .65
    assert .65 < diagnostics['maxDisplacementMm'] <= 1.
    assert np.linalg.norm(positions-points, axis=1).max() <= .7
    assert not len(intersection_vertices(points @ affine[:3, :3].T, positions @ affine[:3, :3].T, faces))
    np.testing.assert_array_equal(faces, before)
    original = points[faces] @ affine[:3, :3].T
    display = positions[faces] @ affine[:3, :3].T
    original_normals, display_normals = (np.cross(f[:, 1]-f[:, 0], f[:, 2]-f[:, 0]) for f in (original, display))
    assert np.all(np.einsum('ij,ij->i', original_normals, display_normals) > 0)

    monkeypatch.setitem(NATIVE_POLICY, 'maxContextTriangles', len(faces)-1)
    unchanged, fallback = fair_surface(points, faces, affine, 'brain', native=True)
    np.testing.assert_array_equal(unchanged, points)
    assert fallback['maxDisplacementMm'] == 0
    assert 'audit budget' in fallback['fallbackReason']


def test_native_display_lesion_volume_is_bounded_per_closed_component():
    from resectionlab.anatomy_geometry import _arrays, _poly
    from resectionlab.surface_fairing import NATIVE_POLICY

    grid = np.indices((31, 31, 31))
    solid = ((grid - 15) ** 2).sum(0) < 81
    solid[2, 3:5, 3] = True
    points, faces = _arrays(_poly(solid.astype(np.uint8), False))
    before = faces.copy()
    positions, diagnostics = fair_surface(points, faces, np.eye(4), 'tumor', native=True)
    assert 0 < diagnostics['maxDisplacementMm'] <= 1
    assert diagnostics['maxComponentVolumeChangeFraction'] <= NATIVE_POLICY['lesionComponentVolumeTolerance']
    np.testing.assert_array_equal(faces, before)
    for mask in ((points[:, 0] < 5)[faces[:, 0]], (points[:, 0] > 5)[faces[:, 0]]):
        def six_times_volume(buffer):
            a, b, c = buffer[faces[mask]].transpose(1, 0, 2).astype(float)
            return np.einsum('ij,ij->', a, np.cross(b, c))
        change = abs(six_times_volume(positions) / six_times_volume(points) - 1)
        assert change <= NATIVE_POLICY['lesionComponentVolumeTolerance']


def test_whole_anatomy_exterior_reduces_staircase_without_relaxing_source_bounds():
    from resectionlab.anatomy_geometry import _arrays, _poly, display_surface_kind
    grid = np.indices((43, 43, 39), dtype=float)
    radius = np.array([17., 15., 14.])
    center = np.array([21.3, 20.2, 18.8])
    affine = np.diag([1., 1., 1.45, 1.])
    spacing = np.diag(affine)[:3]
    mask = np.sum((((grid-center[:, None, None, None])*spacing[:, None, None, None])
                   /radius[:, None, None, None])**2, axis=0) <= 1
    source, faces = _arrays(_poly(mask.astype(np.uint8), False))
    baseline, _ = fair_surface(source, faces, affine, 'context', native=True)
    updated, diagnostics = fair_surface(source, faces, affine, 'skull', native=True)

    def error(points):
        physical = points * spacing
        triangles = physical[faces]
        facet = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        normal = (triangles.mean(axis=1) - center * spacing)/radius**2
        cosine = np.abs((facet*normal).sum(1)/(np.linalg.norm(normal, axis=1)*np.linalg.norm(facet, axis=1)))
        radial = abs(np.linalg.norm((physical - center*spacing)/radius, axis=1)-1)*radius.mean()
        return np.degrees(np.arccos(cosine.clip(-1, 1))).mean(), np.percentile(radial, 90), radial.max()
    before, after = error(baseline), error(updated)
    assert after[0] < before[0] * .8
    assert after[1] < before[1] and after[2] < before[2]
    assert diagnostics['maxDisplacementMm'] <= 1
    assert np.linalg.norm(updated-source, axis=1).max() <= .7
    assert display_surface_kind({'kind':'context', 'provenance':{'model':'siam-v2','label_ontology':{'value':16}}}) == 'outer-head-tissue'
    assert display_surface_kind({'kind':'context','label_value':16}) == 'context'


def test_brain_display_refinement_preserves_raw_geometry_and_crop_faces(tmp_path):
    pytest.importorskip("vtkmodules")
    import nibabel as nib
    from resectionlab.anatomy_brain_surface import BRAIN_ENVELOPE_POLICY
    from resectionlab.anatomy_geometry import mesh
    from resectionlab.patient_anatomy import PatientImagingCollection

    spacing = np.array([.65, .8, 1.05])
    frame = np.diag([*spacing, 1.])
    center = np.array([24.3, 25.2, 22.8])
    radius = np.array([18., 18., 19.])
    native = np.moveaxis(np.indices((58, 57, 49)), 0, -1)
    distance = ((np.linalg.norm((native - center) * spacing / radius, axis=-1) - 1)
                * radius.mean())
    mask = distance <= 0
    signal = (28 + 65 * np.tanh(-distance / .82)).astype(np.float32)
    acquired = nib.Nifti1Image(signal, frame)
    acquired.header.set_xyzt_units("mm")
    path = tmp_path / "brain-fixture.nii"
    nib.save(acquired, path)
    collection = PatientImagingCollection.create(tmp_path / "native", "brain-display-fixture")
    scan = collection.add_series(path, series_id="t1", modality="T1")
    row = collection.add_layer(
        mask.astype(np.uint8), scan["affine"], layer_id="brain", kind="brain", source_series="t1",
        role="context", color="#91c4b6",
        provenance={"kind": "image_derived_estimate", "source": scan["original"]["sha256"]},
    )
    fallback = mesh(collection, {**row, "provenance": {"kind": "unrelated_fixture"}})
    refined = mesh(collection, row)
    assert refined["policy_version"] == BRAIN_ENVELOPE_POLICY
    np.testing.assert_array_equal(refined["source"], fallback["source"])
    np.testing.assert_array_equal(refined["indices"], fallback["indices"])
    np.testing.assert_array_equal(collection.mask(row), mask)
    np.testing.assert_array_equal(collection.array(scan["data"]), signal)
    distance_to_face = np.minimum(refined["source"], np.array(mask.shape) - 1 - refined["source"])
    pinned = (distance_to_face <= 2).any(axis=1)
    assert pinned.any() and not pinned.all()
    np.testing.assert_array_equal(refined["positions"][pinned], fallback["positions"][pinned])
    movement = np.linalg.norm((refined["positions"] - fallback["positions"]) * spacing, axis=1)
    assert .02 < movement.max() <= .4
    assert not len(intersection_vertices(
        fallback["positions"] * spacing, refined["positions"] * spacing, refined["indices"],
    ))
    reopened = mesh(collection, row)
    assert reopened["cache_hit"]
    np.testing.assert_array_equal(reopened["positions"], refined["positions"])


def test_conditioned_skull_surface_keeps_shape_binding_through_portable_reopen(tmp_path):
    import nibabel as nib
    from resectionlab.anatomy_geometry import CONDITIONED_SURFACE, mesh
    from resectionlab.anatomy_package import open_package, save_package
    from resectionlab.core import array_digest
    from resectionlab.patient_anatomy import PatientImagingCollection

    values = np.zeros((10, 10, 10), np.uint8)
    values[2:8, 2:8, 2:8] = 1
    matrix = np.diag([0.5, 0.5, 1., 1.])
    image = nib.Nifti1Image(values, matrix)
    image.header.set_xyzt_units("mm")
    source = tmp_path / "ct.nii.gz"
    nib.save(image, source)
    study = PatientImagingCollection.create(tmp_path / "study", "skull-conditioner")
    scan = study.add_series(source, series_id="ct", modality="CT")
    row = study.add_layer(values, matrix, layer_id="skull", source_series="ct",
                          kind="skull", role="context", color="#dddddd")
    stock = mesh(study, row)
    moved = stock["positions"].copy()
    eligible = stock["source"][:, 2] > 5
    moved[eligible, 2] += .01
    row["conditioned_surface"] = {
        "schema": CONDITIONED_SURFACE, "mask_revision": 0, "label_value": 1,
        "mask_sha256": row["data"]["sha256"], "source_array_sha256": scan["data"]["sha256"],
        "source_original_sha256": scan["original"]["sha256"], "affine": scan["affine"],
        "source_mesh_hash": array_digest(stock["source"]), "indices_hash": array_digest(stock["indices"]),
        "stock_conditioned_hash": array_digest(stock["positions"]), "protected_native_plane": 5,
        "positions": study.put_array(moved),
    }
    study.save()
    archive = tmp_path / "portable.rsworkspace"
    save_package(study, archive)
    reopened = PatientImagingCollection(open_package(archive, tmp_path / "fresh").parent)
    layer = reopened.layer("skull")
    restored = mesh(reopened, layer)
    assert restored["policy_version"] == CONDITIONED_SURFACE
    np.testing.assert_array_equal(restored["source"], stock["source"])
    np.testing.assert_array_equal(restored["indices"], stock["indices"])
    np.testing.assert_array_equal(restored["positions"], moved)
    np.testing.assert_array_equal(reopened.mask(layer), values.astype(bool))
    assert mesh(reopened, {**layer, "revision": 1})["policy_version"] != CONDITIONED_SURFACE
    invalid = {**layer["conditioned_surface"], "mask_sha256": "a" * 64}
    with pytest.raises(ValueError, match="bindings differ"):
        mesh(reopened, {**layer, "conditioned_surface": invalid})
    invalid = {**layer["conditioned_surface"], "protected_crop_margin_cells": 3}
    with pytest.raises(ValueError, match="source bounds"):
        mesh(reopened, {**layer, "conditioned_surface": invalid})
    moved[~eligible, 2] += .01
    invalid = {**layer["conditioned_surface"], "positions": reopened.put_array(moved)}
    with pytest.raises(ValueError, match="source bounds"):
        mesh(reopened, {**layer, "conditioned_surface": invalid})
