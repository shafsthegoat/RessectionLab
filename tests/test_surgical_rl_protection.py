import numpy as np
import pytest
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_rl_protection import ProtectionAnnotation, ProtectionSet, grid_binding
from resectionlab.surgical_tools import TOOLS


def scene(affine=None, name='target'):
    labels = np.ones((9, 9, 9), np.uint16)
    return SurgicalScene(labels, np.eye(4) if affine is None else affine,
        [{'id': 1, 'name': name, 'material': 'tumor', 'target': True}], {'caseId': 'fixture'}, {})


def annotation(s):
    mask = np.zeros(s.shape, bool)
    mask[4, 4, 4] = True
    return ProtectionAnnotation.from_mask(s, 'brainstem', mask, provenance='synthetic test mask, not patient anatomy')


def test_overlap_and_display_rename_do_not_remove_membership():
    s = scene()
    a = annotation(s)
    renamed = scene(name='renamed display')
    assert grid_binding(s) == grid_binding(renamed)
    p = ProtectionSet(renamed, [a], clearance_mm=0)
    assert p.damage_mm3(a.cells) == 1
    assert s.labels.ravel()[a.cells[0]] == 1
    assert p.report()['status'] == 'declared-masks-only'
    with pytest.raises(ValueError):
        p.cells[0] = 0


def test_unknown_coverage_and_strict_requirement():
    p = ProtectionSet(scene())
    assert p.report()['missingRequired'] == ['brainstem']
    assert p.report()['status'] == 'unknown-coverage'
    with pytest.raises(ValueError, match='Missing required'):
        ProtectionSet(scene(), strict=True)


def test_oblique_anisotropic_physical_margin():
    from scipy.spatial.transform import Rotation
    affine = np.eye(4)
    affine[:3, :3] = Rotation.from_euler('z', 31, degrees=True).as_matrix() @ np.diag([1, 2, 4])
    s = scene(affine)
    p = ProtectionSet(s, [annotation(s)], clearance_mm=2)
    center = s.points(p.cells)[0]
    radius = np.sqrt(21) / 2
    assert len(p.capsule_hits(center + [radius + 1.99, 0, 0], center + [radius + 1.99, 0, 0])) == 1
    assert len(p.capsule_hits(center + [radius + 2.01, 0, 0], center + [radius + 2.01, 0, 0])) == 0


def test_long_sweep_and_shaft_intersection():
    s = scene()
    p = ProtectionSet(s, [annotation(s)], clearance_mm=0)
    pose = lambda v: {'position': v, 'quaternion': [0, 0, 0, 1]}
    assert len(p.sweep_hits(TOOLS['cusa'], {'radiusMm': 1.5}, pose([-30, 4, 4]), pose([40, 4, 4]))) == 1
    # Working tip is far beyond structure, but inserted shaft crosses it.
    assert len(p.sweep_hits(TOOLS['cusa'], {'radiusMm': 1.5}, pose([4, 4, 30]), pose([4, 4, 30]))) == 1


def test_source_binding_and_deformed_positions():
    s = scene()
    a = annotation(s)
    shifted = np.eye(4); shifted[0, 3] = 1
    with pytest.raises(ValueError, match='source grid'):
        ProtectionSet(scene(shifted), [a])
    p = ProtectionSet(s, [a], clearance_mm=0)
    assert len(p.capsule_hits([40, 4, 4], [40, 4, 4], positions=np.array([[40, 4, 4]]))) == 1
    assert len(p.capsule_hits([40, 4, 4], [40, 4, 4])) == 0


def test_roundtrip_and_tamper_rejection(tmp_path):
    from resectionlab.surgical_rl_protection import save_protection, load_protection
    s = scene()
    p = ProtectionSet(s, [annotation(s)])
    path = tmp_path / 'protection.json'
    save_protection(p, path)
    restored = load_protection(s, path, strict=True)
    assert restored.report() == p.report()
    assert np.array_equal(restored.cells, p.cells)
    path.with_suffix('.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        load_protection(s, path)


def test_nifti_requires_units_binding_and_registered_same_grid(tmp_path):
    nib = pytest.importorskip('nibabel')
    from resectionlab.surgical_rl_protection import annotation_from_nifti
    s = scene()
    image = nib.Nifti1Image(np.ones(s.shape, np.uint8), np.eye(4))
    path = tmp_path / 'mask.nii.gz'
    nib.save(image, path)
    kwargs = dict(source_binding=grid_binding(s), registration_ras_mm=np.eye(4), provenance='test fixture')
    with pytest.raises(ValueError, match='millimeter'):
        annotation_from_nifti(s, path, 'brainstem', **kwargs)
    image.header.set_xyzt_units('mm')
    nib.save(image, path)
    assert len(annotation_from_nifti(s, path, 'brainstem', **kwargs).cells) == s.labels.size
    kwargs['registration_ras_mm'] = np.diag([2., 1, 1, 1])
    with pytest.raises(ValueError, match='grid'):
        annotation_from_nifti(s, path, 'brainstem', **kwargs)
