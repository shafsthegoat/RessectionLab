import json
from copy import deepcopy
import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation
from resectionlab.anatomy_alignment import (
    Grid, estimate_pair, from_itk, to_itk, rigid, ensure_alignment, input_key,
    displacement, probe_points,
)
from resectionlab.anatomy_intracranial import enclosed_rings, intracranial_vessels
from resectionlab.patient_anatomy import world
from resectionlab.patient_anatomy import file_hash
from resectionlab.anatomy_import import review_registration
from test_patient_anatomy import patient


@pytest.mark.parametrize('spacing,rotation', [([1, 1, 1], 0), ([-.8, 1.1, 1.4], 12)])
def test_rigid_roundtrip_oblique_and_mirrored_source(spacing, rotation):
    a = np.eye(4); a[:3, :3] = Rotation.from_euler('z', rotation, degrees=True).as_matrix() @ np.diag(spacing)
    points = world(a, np.array([[0, 0, 0], [15, 22, 13], [61, 19, 44]]))
    t = np.eye(4); t[:3, :3] = Rotation.from_euler('xyz', [3, -5, 7], degrees=True).as_matrix(); t[:3, 3] = [4, -3, 8]
    assert np.allclose(world(from_itk(to_itk(t)), points), world(t, points), atol=1e-8)
    with pytest.raises(ValueError): rigid(np.diag([-1., 1., 1., 1.]))


@pytest.mark.parametrize('ct', [True, False])
@pytest.mark.parametrize('oblique', [False, True])
def test_known_translation_and_rotation_recovery(ct, oblique):
    xyz = np.indices((64, 64, 64)).astype(float)
    r = np.sqrt(((xyz-31.5)**2).sum(0)); head = r < 27
    data = np.zeros(r.shape)
    for center, width, intensity in [([21, 25, 31], 6, 120), ([40, 37, 28], 5, 190), ([30, 35, 44], 4, 80)]:
        data += intensity*np.exp(-sum((xyz[i]-center[i])**2 for i in range(3))/(2*width**2))
    data += head*30
    if ct: data[~head] = -1000
    a = np.eye(4)
    if oblique: a[:3, :3] = Rotation.from_euler('x', 12, degrees=True).as_matrix() @ np.diag([-.8, 1.1, 1.4])
    a[:3, 3] = -a[:3, :3] @ np.full(3, 31.5)
    t = np.eye(4); t[:3, :3] = Rotation.from_euler('z', 3, degrees=True).as_matrix(); t[:3, 3] = [3, -2, 1]
    moving = ndi.affine_transform(data, np.linalg.inv(a)@t@a, order=1, cval=-1000 if ct else 0)
    mh = ndi.affine_transform(head.astype(np.uint8), np.linalg.inv(a)@t@a, order=0) > 0
    f, m = Grid(data.astype(np.float32), a, head, ct=ct), Grid(moving.astype(np.float32), a, mh, ct=ct)
    fitted, report = estimate_pair(f, m)
    assert displacement(t, fitted, probe_points(m)) <= 1, report
    assert report['metrics']['coverage'] > .9


def test_uniform_images_are_not_certified_and_cancellation_is_observed():
    data = np.ones((24, 24, 24), np.float32); g = Grid(data, np.eye(4), data > 0)
    _, report = estimate_pair(g, g)
    assert report['status'] == 'warning'
    assert report['metrics']['similarity'] is None
    with pytest.raises(InterruptedError): estimate_pair(g, g, cancelled=lambda: True)


def test_graph_cache_history_and_atomic_cancellation(patient, tmp_path, monkeypatch):
    c = patient
    for sid in ['ct-bone', 'ct-brain']:
        c.add_series(tmp_path/'mri.nii.gz', series_id=sid, modality='CT')
    old = np.eye(4); old[:3, 3] = [10, 0, 0]
    c.set_registration('ct-bone', old, status='unreviewed', method='archived')
    original_file = (c.root/'patient-anatomy.json').read_bytes()
    calls = []
    def estimate(f, m, existing=None, cancelled=lambda: False):
        calls.append(existing)
        return np.eye(4), {'status': 'consistent', 'warnings': [], 'metrics': {}, 'candidates': []}
    monkeypatch.setattr('resectionlab.anatomy_alignment.estimate_pair', estimate)
    assert ensure_alignment(c)
    assert c.document['transforms']['ct-bone']['parent_series'] == 'ct-brain'
    assert c.document['transforms']['ct-bone']['status'] == 'unreviewed'
    assert c.document['registration_history'][0]['transform']['method'] == 'archived'
    assert np.allclose(c.document['transforms']['ct-bone']['matrix'], np.eye(4))
    assert not ensure_alignment(c)
    assert len(calls) == 2
    assert (c.root/'patient-anatomy.json').read_bytes() == original_file
    before = deepcopy(c.document)
    with pytest.raises(InterruptedError): ensure_alignment(c, force=True, cancelled=lambda: True)
    assert c.document == before
    c.document['transforms']['ct-brain']['matrix'][0][3] = 1
    with pytest.raises(ValueError, match='parent changed|no longer matches'): c.registration_matrix('ct-bone')
    assert input_key(c, c.series('ct-bone'), c.series('ct-brain')) != c.document['transforms']['ct-bone']['input_key']
    c.series('ct-brain')['original']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='source changed'): ensure_alignment(c)


def test_review_rejects_large_residual_and_preserves_automatic_provenance(patient, tmp_path):
    c = patient; c.add_series(tmp_path/'mri.nii.gz', series_id='ct', modality='CT')
    c.set_registration('ct', np.eye(4), status='unreviewed', method='test')
    c.document['transforms']['ct']['quality'] = {'status': 'warning', 'warnings': ['test']}
    pts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    with pytest.raises(ValueError, match='exceeds 2 mm'):
        review_registration(c, 'ct', pts, (np.array(pts)+10).tolist(), 'test')
    assert c.document['transforms']['ct']['status'] == 'unreviewed'
    review_registration(c, 'ct', pts, pts, 'test')
    assert c.document['transforms']['ct']['quality']['warnings'] == ['test']


def test_envelope_preserves_peripheral_space_and_rejects_open_or_unseeded_rings():
    x, y, z = np.indices((64, 64, 12)); r = np.sqrt((x-32)**2+(y-32)**2)
    ct = np.where(r > 26, -1000., np.where(r > 23, 1000., 30.)).astype(np.float32)
    brain = r < 19
    inside, outer, supported = enclosed_rings(ct, brain, 1)
    assert supported.all()
    assert inside[53, 32, 6] and not brain[53, 32, 6]  # outside cortex, inside skull
    assert not inside[60, 32, 6]
    ct[31:39, 50:, :] = 0  # opening wider than permitted 2 mm closing
    _, _, supported = enclosed_rings(ct, brain, 1)
    assert not supported.any()
    _, _, supported = enclosed_rings(np.zeros_like(ct), brain, 1)
    assert not supported.any()


def test_missing_ct_is_unknown_and_filter_does_not_rewrite_sources(patient):
    c = patient; original = deepcopy(c.document); full = c.mask(c.layer('vessels')).copy()
    filtered, report = intracranial_vessels(c, c.layer('vessels'))
    assert not c.mask(filtered).any()
    assert report['uncertainVoxels'] == full.sum()
    assert report['outsideVoxels'] == 0
    excluded, _ = intracranial_vessels(c, c.layer('vessels'), excluded=True)
    assert np.array_equal(c.mask(excluded), full)
    assert excluded['color'] == '#ffad54'
    assert c.document == original
    assert np.array_equal(c.mask(c.layer('vessels')), full)


@pytest.mark.parametrize('failure', ['patient', 'units', 'qform'])
def test_invalid_source_identity_or_geometry_never_reaches_optimizer(patient, tmp_path, monkeypatch, failure):
    import nibabel as nib
    c = patient
    row = c.add_series(tmp_path/'mri.nii.gz', series_id='ct', modality='CT')
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid source reached registration')
    monkeypatch.setattr('resectionlab.anatomy_alignment.estimate_pair', forbidden)
    if failure == 'patient':
        row['patient_id'] = 'different-patient'
    else:
        path = c.path(row['original']['path'])
        image = nib.load(path)
        replacement = nib.Nifti1Image(image.get_fdata(), image.affine, image.header.copy())
        if failure == 'units': replacement.header.set_xyzt_units('unknown')
        else:
            shifted = image.affine.copy(); shifted[0, 3] += 10
            replacement.set_qform(shifted, code=1)
            replacement.set_sform(image.affine, code=1)
        nib.save(replacement, path)
        # Updating the content hash cannot legitimize invalid geometry.
        for source in c.document['series']:
            if c.path(source['original']['path']) == path:
                source['original']['sha256'] = file_hash(path)
    with pytest.raises(ValueError, match='same patient|millimetre|qform/sform'):
        ensure_alignment(c)
    assert 'ct' not in c.document['transforms']
