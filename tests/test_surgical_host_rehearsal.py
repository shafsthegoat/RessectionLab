"""Operative rehearsal on a host head: provenance, geometry, admission and instrument revision 3.

A small layered sphere stands in for a prepared host study, so these checks run
without the owner's drive.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest
from scipy import ndimage

from resectionlab.core import CaseData, SourceRef, semantic_digest, thaw_json
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab import surgical_host_rehearsal as host
from resectionlab.surgical_scene import from_case

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


def sphere_host(*, synthetic=True, layers=LAYERS):
    """A layered head of 75 mm radius with a tumour under its vertex, on a 1.5 mm RAS grid."""
    spacing, size = 1.5, 112
    affine = np.diag([spacing, spacing, spacing, 1.])
    affine[:3, 3] = -spacing * (size - 1) / 2
    axis = np.arange(size) * spacing + affine[0, 3]
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
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
    masks = {name: mask for name, mask in masks.items() if name in layers}
    metadata = {'anatomy_layers': {name: dict(layers[name]) for name in masks},
                'is_synthetic': synthetic, 'display_label': 'layered sphere host'}
    if synthetic:
        metadata.update(split={'role': 'DEVELOPMENT'}, data_case={'classification': 'synthetic_scenario_on_test_host'})
    mri = np.zeros((size,) * 3, dtype=np.float32)
    for value, name in enumerate(masks, 1):
        mri[masks[name]] = value / 10.
    return CaseData(case_id='sphere-host', revision=1, affine=affine, mri=mri, compartments=masks,
                    source_refs=(SourceRef('layered-sphere', 'synthetic://test/layered-sphere', provenance='simulated'),),
                    metadata=metadata, source_compartments={**masks, **healthy})


@pytest.fixture(scope='module')
def built():
    case = sphere_host()
    derived, scene, landmarks = host.build_host_rehearsal(case)
    return case, derived, scene, landmarks


def test_layer_roles_follow_declared_kind_and_label():
    assert host.host_role('x', {'kind': 'tumor', 'label': 'Core'}) == 'tumour_core'
    assert host.host_role('x', {'kind': 'tumor', 'label': 'Rim'}) == 'tumour_rim'
    assert host.host_role('x', {'kind': 'skull', 'label': 'Bone-Cancellous'}) == 'cancellous_bone'
    assert host.host_role('x', {'kind': 'skull', 'label': 'Bone-Cortical'}) == 'cortical_bone'
    assert host.host_role('x', {'kind': 'context', 'label': 'Skin'}) == 'skin'
    assert host.host_role('x', {'kind': 'context', 'label': 'CSF'}) == 'csf'
    assert host.host_role('x', {'kind': 'brain', 'label': 'Left-Cerebral-Cortex'}) == 'cortex'
    assert host.host_role('x', {'kind': 'vessels', 'label': 'MRA vessels · vesselFM'}) == 'angiography_vessel'
    assert host.host_role('x', {'kind': 'tract', 'label': 'CSTs'}) == 'tract'
    assert host.host_role('brain', {'kind': 'brain', 'label': 'overview'}) is None
    roles, tracts, provenance = host.host_roles(sphere_host())
    assert {'skin', 'subcutaneous', 'cortical_bone', 'cancellous_bone', 'csf', 'cortex', 'white',
            'tumour_rim', 'tumour_core'} <= set(roles) and not tracts
    # Tumour provenance follows its source layer, not a hard-coded 'generated'.
    assert provenance['tumour_rim'] == 'simulated' and provenance['cortex'] == 'estimated'


@pytest.mark.parametrize('missing,named', [('skin', 'scalp or skin'), ('cortical', 'skull')])
def test_missing_scalp_or_skull_is_named_never_invented(missing, named):
    layers = {name: row for name, row in LAYERS.items() if name not in (missing, 'cancellous')}
    with pytest.raises(ValueError, match='HOST_REHEARSAL_MISSING_ANATOMY: .*' + named):
        host.host_roles(sphere_host(layers=layers))


def test_entry_is_the_short_convexity_corridor_and_the_frame_is_outward():
    case = sphere_host()
    roles, _, _ = host.host_roles(case)
    entry = host.plan_entry(case, roles)
    assert np.dot(entry['outwardRasUnit'], [0., 0., 1.]) > .97
    assert entry['depthMm'] == pytest.approx(75. - 44., abs=3.)
    shape, affine, centre = host.working_frame(entry)
    rotation = affine[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert np.allclose(rotation[:, 2], entry['outwardRasUnit'])
    assert np.allclose(rotation[:, :2].T @ rotation[:, 2], 0.)
    assert np.allclose(affine[:3, :3] @ centre + affine[:3, 3], entry['entryRasMm'])


def test_built_scene_declares_provenance_and_face_connected_bodies(built):
    case, derived, scene, landmarks = built
    value = host.declaration(scene)
    assert value['sourceClass'] == 'synthetic_scenario_on_test_host' and value['role'] == 'DEVELOPMENT'
    layers = {row['name']: row['provenance'] for row in value['layers']}
    assert layers['Dura (assumed)'] == 'assumed' and layers['Tumour rim'] == 'simulated'
    assert layers['Skull cortical bone'] == 'estimated'
    assert {'Epidural plane under the craniotomy (assumed)', 'Subgaleal plane under the scalp flap (assumed)'} \
        <= set(value['freeSpaceMemberships'])
    exposure = scene.properties['operativeExposureV1']
    assert exposure['instrumentGeometryRevision'] == 3
    assert scene.properties['cusaV3']['capacityMm3PerSecond'] == host.HOST_CUSA_MM3_PER_SECOND
    assert scene.properties['cusaV3']['provenance']['kind'] == 'assumed-research'
    for name in ('scalp', 'bone', 'dura'):
        mask = np.zeros(scene.shape, dtype=bool)
        mask.flat[exposure['bodies'][name]['cells']] = True
        assert ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))[1] == 1, name
    # The dura lies under the bone flap, never outside the calvarium, stripped
    # from its inner table by the one-cell interior plane.
    dura = np.zeros(scene.shape, dtype=bool)
    dura.flat[exposure['bodies']['dura']['cells']] = True
    bone = np.zeros(scene.shape, dtype=bool)
    bone.flat[exposure['bodies']['bone']['cells']] = True
    gaps = [bone[i, j].nonzero()[0].min() - dura[i, j].nonzero()[0].max()
            for i, j in zip(*np.nonzero(dura.any(axis=2) & bone.any(axis=2)))]
    assert gaps and min(gaps) >= 1 + host.EPIDURAL_INTERIOR_MM


def test_declaration_is_validated(built):
    _, derived, scene, _ = built
    metadata = thaw_json(derived.metadata)
    metadata['surgical_properties']['hostRehearsalV1']['schema'] = 'other'
    tampered = from_case(replace(derived, metadata=metadata), scene.regions)
    with pytest.raises(ValueError, match='HOST_REHEARSAL_INVALID_DECLARATION'):
        host.declaration(tampered)
    metadata = thaw_json(derived.metadata)
    metadata['surgical_properties']['hostRehearsalV1']['layers'].pop()
    with pytest.raises(ValueError, match='HOST_REHEARSAL_LAYER_DECLARATION_MISMATCH'):
        host.declaration(from_case(replace(derived, metadata=metadata), scene.regions))


def _without_declaration(derived, scene):
    """The same operative scene as an acquired-class source with no provenance declaration."""
    metadata = thaw_json(derived.metadata)
    del metadata['surgical_properties']['hostRehearsalV1']
    metadata.update(is_synthetic=False, split=None, data_case={'source_subject': 'acquired-test-subject'})
    case = replace(derived, case_id='acquired-operative-test', metadata=metadata)
    return case, from_case(case, scene.regions)


def test_engine_admits_any_operative_reconstruction(built):
    _, derived, scene, _ = built
    engine = SurgicalEngine(scene)
    assert engine._operative_tool_revision == 3
    assert engine.state['operativeInstrumentCatalogHash'] == semantic_digest(
        __import__('resectionlab.surgical_tools', fromlist=['catalog']).catalog(operative=3))
    acquired_case, acquired = _without_declaration(derived, scene)
    assert host.declaration(acquired) is None
    engine = SurgicalEngine(acquired)
    outcome = engine.step({'slot': 'right', 'operation': 'inspect', 'duration': .1})
    assert outcome['status'] == 'accepted'


def test_practice_binds_the_scene_own_tumour_regions(built):
    from resectionlab.surgical_practice import source_binding
    _, derived, scene, _ = built
    for case, operative in [(derived, scene), _without_declaration(derived, scene)]:
        engine = SurgicalEngine(operative)
        snapshot = engine.snapshot()
        record = {'branchId': 'main', 'branches': {'main': snapshot}}
        binding = source_binding(engine, operative.source['caseHash'], snapshot, record)
        assert [row['name'] for row in binding['targetBinding']['regions']] == ['Tumour rim', 'Tumour core']


def test_table_frame_round_trip(built):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from host_rehearsal_frame import TableFrame
    from scipy.spatial.transform import Rotation
    _, _, scene, landmarks = built
    frame = TableFrame(scene)
    assert np.allclose(frame.point(landmarks['entryCenterRasMm']), 0., atol=1e-9)
    assert np.allclose(frame.vector(landmarks['outwardRasUnit']), [0., 0., 1.])
    point = [3., -7., 11.]
    assert np.allclose(frame.point(frame.ras_point(point)), point)
    q = Rotation.from_euler('xyz', [.3, -.2, 1.1]).as_quat()
    back = frame.quaternion(frame.ras_quaternion(q))
    assert Rotation.from_quat(back).approx_equal(Rotation.from_quat(q), atol=1e-9)


def test_instrument_revision_three_leaves_revisions_one_and_two_unchanged():
    from resectionlab.surgical_tools import catalog, operative_tool_parameters
    from resectionlab.surgical_tool_profiles import PROFILE_IDS, assembly_segments
    # Replay data recorded before revision 3 existed.
    assert semantic_digest(catalog(operative=True)) == \
        'sha256:2c9633f8e584adc5ec77245f230c14345e2a7dc85f5cbf9615b34c9681dfb5a8'
    assert semantic_digest(catalog(operative=2)) == \
        'sha256:7277795a0c6fd04fdabe496c50ddb4ebd4bb997392d2c1ca97a064a21c1536f6'
    assert PROFILE_IDS[:21][-1] == 'fixation_driver' and len(PROFILE_IDS) == 23
    two, three = catalog(operative=2), catalog(operative=3)
    changed = [b['id'] for a, b in zip(two, three) if a['physicalProfile']['geometryHash'] !=
               b['physicalProfile']['geometryHash']]
    assert changed == ['scalpel', 'craniotome']
    for name, reach in (('scalpel', 22.), ('craniotome', 14.)):
        params = operative_tool_parameters(name, {'jawOpeningMm': 0.}, 3)
        working = [part for part in assembly_segments(params['operativeToolProfileV1'], 0., 2) if part[3] == 'working']
        assert min(min(a[2], b[2]) for a, b, *_ in working) == pytest.approx(-reach)
        assert params['operativeToolAssemblyRevision'] == 2
    assert operative_tool_parameters('scalpel', {}, 2)['operativeToolProfileV1'] == 4
