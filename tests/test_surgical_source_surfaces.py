"""Source-resolution display surfaces of an operative rehearsal, on a layered-sphere host.

The host is built here, so these checks run without the owner's drive: its
head model, T1, tumour mask, a vessel model on its own grid and a tract file
are written as NIfTI/TRK files beside a saved host study.
"""
from __future__ import annotations

import json
import zipfile

import nibabel as nib
import numpy as np
import pytest

from resectionlab import surgical_host_rehearsal as host
from resectionlab import surgical_source_surfaces as surfaces
from resectionlab.core import CaseData, SourceRef
from resectionlab.imaging import save_case

SPACING, SIZE = 1.5, 112
CODES = {'skin': 911, 'subcutaneous': 907, 'cortical': 915, 'cancellous': 916, 'csf': 24, 'cortex': 3, 'white': 2}
LAYERS = {
    'skin': {'kind': 'context', 'label': 'Skin · T1 MODEL ESTIMATE', 'provenance': 'estimated'},
    'subcutaneous': {'kind': 'context', 'label': 'Other-Tissues · T1 MODEL ESTIMATE', 'provenance': 'estimated'},
    'cortical': {'kind': 'skull', 'label': 'Bone-Cortical · T1 MODEL ESTIMATE', 'provenance': 'estimated'},
    'cancellous': {'kind': 'skull', 'label': 'Bone-Cancellous · T1 MODEL ESTIMATE', 'provenance': 'estimated'},
    'csf': {'kind': 'context', 'label': 'CSF · T1 MODEL ESTIMATE', 'provenance': 'estimated'},
    'cortex': {'kind': 'brain', 'label': 'Left-Cerebral-Cortex · T1 MODEL ESTIMATE', 'provenance': 'estimated'},
    'white': {'kind': 'brain', 'label': 'Left-Cerebral-White-Matter · T1 MODEL ESTIMATE', 'provenance': 'estimated'},
    'rim': {'kind': 'tumor', 'label': 'Synthetic Rim · GENERATED', 'provenance': 'simulated'},
    'core': {'kind': 'tumor', 'label': 'Synthetic Core · GENERATED', 'provenance': 'simulated'},
}
RADII = {'skin': 75., 'skullOuter': 68., 'skullInner': 60., 'pial': 58.5, 'white': 55.}


def _grid():
    affine = np.diag([SPACING, SPACING, SPACING, 1.])
    affine[:3, 3] = -SPACING * (SIZE - 1) / 2
    axis = np.arange(SIZE) * SPACING + affine[0, 3]
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    return affine, x, y, z


def sphere_host():
    """A layered head of 75 mm radius with a tumour under its vertex (as in the host rehearsal tests)."""
    affine, x, y, z = _grid()
    r = np.sqrt(x * x + y * y + z * z)
    tumour = np.sqrt(x * x + y * y + (z - 44.) ** 2)
    masks = {
        'skin': (r <= 75.) & (r > 72.), 'subcutaneous': (r <= 72.) & (r > 68.),
        'cortical': ((r <= 68.) & (r > 65.)) | ((r <= 62.) & (r > 60.)), 'cancellous': (r <= 65.) & (r > 62.),
        'csf': (r <= 60.) & (r > 58.5), 'cortex': (r <= 58.5) & (r > 55.), 'white': r <= 55.,
        'rim': (tumour <= 11.) & (tumour > 6.), 'core': tumour <= 6.,
    }
    healthy = {name: masks[name].copy() for name in ('cortex', 'white')}
    for name in ('cortex', 'white'):
        masks[name] &= tumour > 11.
    metadata = {'anatomy_layers': {name: dict(LAYERS[name]) for name in masks}, 'is_synthetic': True,
                'display_label': 'layered sphere host', 'split': {'role': 'DEVELOPMENT'},
                'data_case': {'classification': 'synthetic_scenario_on_test_host'}}
    mri = np.zeros((SIZE,) * 3, dtype=np.float32)
    for value, name in enumerate(masks, 1):
        mri[masks[name]] = value / 10.
    case = CaseData(case_id='sphere-host', revision=1, affine=affine, mri=mri, compartments=masks,
                    source_refs=(SourceRef('layered-sphere', 'synthetic://test/layered-sphere', provenance='simulated'),),
                    metadata=metadata, source_compartments={**masks, **healthy})
    return case, masks, healthy


def _nifti(path, data, affine):
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units('mm')
    image.set_sform(affine, 1)
    image.set_qform(affine, 1)
    nib.save(image, str(path))
    return str(path)


VESSEL_SHIFT = np.array([2., -1., 0.])


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    directory = tmp_path_factory.mktemp('source-surfaces')
    case, masks, healthy = sphere_host()
    host_path = directory / 'host.ressectionlab'
    save_case(case, host_path)
    operative, scene, landmarks = host.build_host_rehearsal(case)
    affine = case.affine
    labels = np.zeros(case.mri.shape, np.int16)
    for name in ('skin', 'subcutaneous', 'cortical', 'cancellous', 'csf'):
        labels[masks[name]] = CODES[name]
    labels[healthy['cortex']] = CODES['cortex']
    labels[healthy['white']] = CODES['white']
    tumour = np.zeros(case.mri.shape, np.uint8)
    tumour[masks['core']] = 1
    tumour[masks['rim']] = 2
    _, x, y, z = _grid()
    t1 = np.clip(75.5 - np.sqrt(x * x + y * y + z * z), 0, 1).astype(np.float32) * 800.
    # A vessel model on its own 0.5 mm grid: a straight tube 1 mm in radius across the cortex at
    # z = 57 (vessel frame), placed in the host by a translation.
    vessel_affine = np.diag([.5, .5, .5, 1.])
    vessel_affine[:3, 3] = [-20., -6., 51.]
    vi, vj, vk = np.meshgrid(*(np.arange(n) for n in (80, 24, 24)), indexing='ij')
    vy = vj * .5 - 6.
    vz = vk * .5 + 51.
    vessel = (np.hypot(vy, vz - 57.) <= 1.).astype(np.uint8)
    to_host = np.eye(4)
    to_host[:3, 3] = VESSEL_SHIFT
    tract_path = directory / 'bundle.trk'
    streamline = np.column_stack([np.linspace(-10, 10, 41), np.zeros(41), np.full(41, 40.)]).astype(np.float32)
    tractogram = nib.streamlines.Tractogram([streamline], affine_to_rasmm=np.eye(4))
    nib.streamlines.save(tractogram, str(tract_path))
    paths = {
        'hostStudy': str(host_path),
        'headLabels': _nifti(directory / 'labels.nii.gz', labels, affine),
        't1': _nifti(directory / 't1.nii.gz', t1, affine),
        'tumourLabels': _nifti(directory / 'tumour.nii.gz', tumour, affine),
        'tumourCodes': {'tumour_core': 1, 'tumour_rim': 2},
        'vesselMask': _nifti(directory / 'vessel.nii.gz', vessel, vessel_affine),
        'vesselToHost': to_host.tolist(),
        'tracts': [{'bundle': 'Bundle', 'path': str(tract_path), 'toHost': np.eye(4).tolist()}],
    }
    asset = surfaces.build_assets(operative, paths)
    return {'directory': directory, 'operative': operative, 'landmarks': landmarks, 'paths': paths, 'asset': asset,
            'context': surfaces.operative_context(operative)}


def _surface(asset, name):
    arrays = asset['arrays']
    key = 'surface.' + name
    return {field: arrays[key + '.' + field] for field in
            ('positions', 'normals', 'triangles', 'cells', 'ownerCells', 'body', 'bodyEdgeMm')
            if key + '.' + field in arrays}


def test_surfaces_are_smooth_and_follow_their_label_boundaries(built):
    header = built['asset']['header']
    assert header['schema'] == surfaces.SCHEMA and header['hostVerified'] is True
    rows = {row['name']: row for row in header['surfaces']}
    assert {'skin', 'skullOuter', 'skullInner', 'dura', 'pial', 'white', 'tumourRim', 'tumourCore'} <= set(rows)
    for name, radius in RADII.items():
        mesh = _surface(built['asset'], name)
        r = np.linalg.norm(mesh['positions'], axis=1)
        # Within half a source voxel of the layer boundary, on average and almost everywhere.
        assert abs(np.median(r) - radius) < .75, name
        assert rows[name]['labelDeviation']['meanAbsMm'] < .75, name
        # Smooth: neighbouring normals turn by a few degrees on a 60-75 mm sphere, not by label steps.
        assert rows[name]['quality']['normalStepMeanDeg'] < 4., name
        # Normals point outward (away from the sphere centre).
        assert np.mean(np.einsum('ij,ij->i', mesh['normals'], mesh['positions']) > 0) > .98, name
        assert mesh['positions'].dtype == np.float32 and mesh['triangles'].dtype == np.uint32
        assert .4 < rows[name]['quality']['meanEdgeMm'] < 1.1, name
    assert rows['skin']['provenance']['origin'] == 'acquired'
    assert rows['skullOuter']['provenance']['origin'] == 'estimated'
    assert 'not CT' in rows['skullOuter']['provenance']['estimate']
    assert rows['dura']['provenance']['origin'] == 'assumed'
    assert rows['tumourRim']['provenance']['origin'] == 'generated'
    for row in rows.values():
        assert row['provenance']['sha256'] and row['provenance']['resolutionMm']
    tumour = _surface(built['asset'], 'tumourRim')['positions']
    assert abs(np.median(np.linalg.norm(tumour - [0., 0., 44.], axis=1)) - 11.) < 1.


def test_vertices_map_back_to_their_cell_and_layer(built):
    context = built['context']
    affine, shape, labels = context['affine'], context['shape'], context['labels']
    names = context['names']
    rows = {row['name']: row for row in built['asset']['header']['surfaces']}
    inverse = np.linalg.inv(affine)
    for name in ('skin', 'skullOuter', 'pial', 'tumourRim'):
        mesh = _surface(built['asset'], name)
        cells, owners = mesh['cells'], mesh['ownerCells']
        ijk = np.rint(mesh['positions'].astype(float) @ inverse[:3, :3].T + inverse[:3, 3]).astype(int)
        inside = cells >= 0
        assert inside.mean() > .8  # the rest lies in the 10 mm margin around the grid
        assert np.array_equal(np.ravel_multi_index(ijk[inside].T, shape), cells[inside])
        owned = owners >= 0
        assert owned[inside].mean() > .95, name
        layer_ids = {names.index(layer) + 1 for layer in rows[name]['layers']}
        assert set(np.unique(labels.flat[owners[owned]]).tolist()) <= layer_ids, name
        # The owner lies near its vertex.
        owner_ijk = np.column_stack(np.unravel_index(owners[owned], shape))
        assert np.percentile(np.linalg.norm((owner_ijk - ijk[owned]), axis=1), 99) <= 3.
    assert rows['pial']['layers'] == ['Cerebral cortex']
    assert rows['skin']['layers'] == ['Scalp skin']


def test_flap_patches_bind_to_their_bodies_with_smooth_outlines(built):
    context, landmarks = built['context'], built['landmarks']
    affine = context['affine']
    entry = np.asarray(landmarks['entryCenterRasMm'])
    outward = np.asarray(landmarks['outwardRasUnit'])
    for name, code, radius in (('skin', 1, None), ('skullOuter', 2, 22.), ('skullInner', 2, 22.), ('dura', 3, None)):
        mesh = _surface(built['asset'], name)
        positions = mesh['positions'].astype(float)
        lateral = np.linalg.norm((positions - entry) - np.outer((positions - entry) @ outward, outward), axis=1)
        centre = lateral < 5.
        assert centre.any() and np.all(mesh['body'][centre] == code), name
        assert np.all(mesh['bodyEdgeMm'][centre] > 5.), name
        far = lateral > 60.
        assert not np.any(mesh['body'][far]), name
        if radius is not None:
            flagged = mesh['body'] == code
            # The bone flap is a 22 mm disc: flagged vertices stay inside it and fill it.
            assert lateral[flagged].max() < radius + 1.5 and np.mean(lateral[mesh['body'] == 0] > radius - 1.5) > .99
    outlines = built['asset']['header']['outlines']
    bone = built['asset']['arrays'][outlines['bone']['outer']['points']].astype(float)
    distance = np.linalg.norm((bone - entry) - np.outer((bone - entry) @ outward, outward), axis=1)
    assert abs(np.median(distance) - 22.) < 1.5 and distance.std() < 1.
    # Smooth: successive outline steps are 0.5 mm and turn gently.
    steps = np.diff(np.vstack([bone, bone[:1]]), axis=0)
    assert np.allclose(np.linalg.norm(steps, axis=1)[:-1], .5, atol=.05)
    turn = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', steps[:-1], steps[1:]) /
                                        (np.linalg.norm(steps[:-1], axis=1) * np.linalg.norm(steps[1:], axis=1)), -1, 1)))
    assert turn.max() < 10.
    # The outline lies on the outer table (radius 68 mm on the sphere).
    assert abs(np.median(np.linalg.norm(bone, axis=1)) - 68.) < 1.
    scalp = built['asset']['arrays'][outlines['scalp']['outer']['points']]
    hinge = built['asset']['arrays'][outlines['scalp']['outer']['hinge']]
    assert hinge.any() and not hinge.all() and abs(np.median(np.linalg.norm(scalp, axis=1)) - 75.) < 1.


def test_vessels_and_tracts_are_mapped_into_host_ras(built):
    header, arrays = built['asset']['header'], built['asset']['arrays']
    vessels = header['vessels']
    assert vessels['origin'] == 'estimated' and vessels['complete'] is False
    points = arrays['vessels.centrelinePoints']
    # The tube runs along x at (y, z) = (0, 57) in its own frame; the host sees it shifted.
    assert np.allclose(np.median(points[:, 1:], axis=0), [0. + VESSEL_SHIFT[1], 57. + VESSEL_SHIFT[2]], atol=.3)
    assert np.allclose(np.median(arrays['vessels.centrelineRadii']), 1., atol=.25)
    assert vessels['coverage']['centrelineMm'] > 25. and vessels['coverage']['graphComponents'] == 1
    assert set(np.unique(arrays['vessels.segmentKind']).tolist()) <= {0}
    assert arrays['vessels.wallTriangles'].dtype == np.uint32 and len(arrays['vessels.wallPositions'])
    tract = header['tracts'][0]
    assert tract['bundle'] == 'Bundle' and tract['exported'] == 1
    line = arrays[tract['arrays']['points']]
    assert np.allclose(line[[0, -1]], [[-10., 0., 40.], [10., 0., 40.]], atol=1e-4)


def test_asset_round_trip_and_refusals(built, tmp_path):
    asset, operative = built['asset'], built['operative']
    path = tmp_path / surfaces.ASSET_NAME
    header = surfaces.write_asset(asset, path)
    loaded = surfaces.read_asset(path, case_hash=operative.semantic_hash, host_case_hash=header['hostCaseHash'])
    assert loaded['header'] == header and set(loaded['arrays']) == set(asset['arrays'])
    for key, value in asset['arrays'].items():
        assert np.array_equal(loaded['arrays'][key], value) and loaded['arrays'][key].dtype == value.dtype
    with pytest.raises(ValueError, match='SOURCE_SURFACES_CASE_MISMATCH'):
        surfaces.read_asset(path, case_hash='sha256:' + '0' * 64)
    with pytest.raises(ValueError, match='SOURCE_SURFACES_HOST_MISMATCH'):
        surfaces.read_asset(path, host_case_hash='sha256:' + '1' * 64)
    # A changed array under the same header is refused.
    with np.load(path) as data:
        contents = {key: data[key] for key in data.files}
    contents['surface.skin.positions'] = contents['surface.skin.positions'] + np.float32(1.)
    tampered = tmp_path / 'tampered.npz'
    np.savez_compressed(tampered, **contents)
    with pytest.raises(ValueError, match='SOURCE_SURFACES_INVALID'):
        surfaces.read_asset(tampered)
    # A host study that is not the rehearsal host is refused at build time.
    other = tmp_path / 'other.ressectionlab'
    with zipfile.ZipFile(other, 'w') as archive:
        archive.writestr('manifest.json', json.dumps({'case_semantic_hash': 'sha256:' + '2' * 64}))
    with pytest.raises(ValueError, match='SOURCE_SURFACES_HOST_MISMATCH'):
        surfaces.build_assets(operative, {**built['paths'], 'hostStudy': str(other)})


def test_bridge_operation_reports_unavailable_then_returns_the_asset(built, tmp_path):
    from resectionlab.desktop_bridge import BridgeSession, _Request, BridgeError
    operative = built['operative']
    case_path = tmp_path / 'case' / 'host-rehearsal.ressectionlab'
    case_path.parent.mkdir()
    save_case(operative, case_path)
    session = BridgeSession(tmp_path / 'transfers')
    try:
        def call(operation, args):
            return session.execute(operation, args, _Request(operation), lambda *_: None)
        opened = call('loadCase', {'path': str(case_path)})
        case_hash = opened['caseHash']
        assert case_hash == operative.semantic_hash
        missing = call('operativeSourceSurfaces', {'caseHash': case_hash, 'casePath': str(case_path)})
        assert missing['status'] == 'unavailable' and missing['code'] == 'ASSET_NOT_FOUND' and missing['reason']
        assert call('operativeSourceSurfaces', {'caseHash': case_hash})['code'] == 'ASSET_PATH_REQUIRED'
        with pytest.raises(BridgeError):
            call('operativeSourceSurfaces', {'caseHash': case_hash, 'casePath': 'relative/case.ressectionlab'})
        foreign = dict(built['asset'], header={**built['asset']['header'], 'operativeCaseHash': 'sha256:' + '3' * 64})
        surfaces.write_asset(foreign, tmp_path / 'foreign.npz')
        refused = call('operativeSourceSurfaces', {'caseHash': case_hash, 'assetPath': str(tmp_path / 'foreign.npz')})
        assert refused['status'] == 'unavailable' and refused['code'] == 'SOURCE_SURFACES_CASE_MISMATCH'
        surfaces.write_asset(built['asset'], case_path.parent / surfaces.ASSET_NAME)
        result = call('operativeSourceSurfaces', {'caseHash': case_hash, 'casePath': str(case_path)})
        assert result['status'] == 'available' and result['header']['schema'] == surfaces.SCHEMA
        assert set(result['arrays']) == set(built['asset']['arrays'])
        descriptor = result['arrays']['surface.pial.positions']
        assert descriptor['dtype'] == 'float32' and descriptor['shape'][1] == 3
        raw = np.fromfile(descriptor['path'], dtype='<f4').reshape(descriptor['shape'])
        assert np.array_equal(raw, built['asset']['arrays']['surface.pial.positions'])
        assert result['arrays']['surface.pial.triangles']['dtype'] == 'uint32'
        assert result['arrays']['surface.skin.cells']['dtype'] == 'int32'
    finally:
        session.close()
