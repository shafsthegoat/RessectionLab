import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

sitk=pytest.importorskip("SimpleITK")

spec=importlib.util.spec_from_file_location('data_integrated_healthy',Path(__file__).parents[1]/'scripts/data_integrated_healthy.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_nonzero_center_lps_pull_inverse_maps_ras_landmark_impulse_to_fixed_grid():
    fixed_affine=np.diag([2.,3.,4.,1.]);fixed_affine[:3,3]=[30.,-50.,100.]
    moving_affine=np.diag([-1.5,2.,2.5,1.]);moving_affine[:3,3]=[80.,-60.,70.]
    source_ijk=np.array([6,7,8]);target_ijk=np.array([4,5,3])
    source_ras=(moving_affine@np.r_[source_ijk,1])[:3]
    target_ras=(fixed_affine@np.r_[target_ijk,1])[:3]
    forward=np.eye(4);forward[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    forward[:3,3]=target_ras-forward[:3,:3]@source_ras
    pull_lps=module.FRAME_FLIP@np.linalg.inv(forward)@module.FRAME_FLIP
    transform=sitk.AffineTransform(3);center=np.array([10.,-20.,5.]);transform.SetCenter(tuple(center))
    transform.SetMatrix(tuple(pull_lps[:3,:3].ravel()))
    transform.SetTranslation(tuple(pull_lps[:3,3]-center+pull_lps[:3,:3]@center))
    assert np.allclose(module.forward_ras_from_pull(transform),forward,atol=1e-10)
    assert np.allclose(module.linear_transform_matrix(transform),pull_lps,atol=1e-10)
    moving=np.zeros((16,18,20),np.uint8);moving[tuple(source_ijk)]=1
    fixed=module.image_from_ras(np.zeros((12,14,10),np.float32),fixed_affine)
    sampled=module.warp_native(moving,moving_affine,fixed,transform,sitk.sitkNearestNeighbor)
    assert sampled.shape==(12,14,10)
    assert sampled[tuple(target_ijk)]==1 and sampled.sum()==1
    scalar=module.warp_native(moving.astype(np.float32)*123.5,moving_affine,fixed,transform,sitk.sitkLinear)
    coverage=module.warp_native(np.ones(moving.shape,np.uint8),moving_affine,fixed,transform,sitk.sitkNearestNeighbor)
    assert scalar.dtype==np.float32 and scalar[tuple(target_ijk)]==123.5
    assert coverage[tuple(target_ijk)]==1 and np.any(coverage==0)
    assert np.allclose(forward@np.r_[source_ras,1],np.r_[target_ras,1])
    # An accidental forward/pull swap must put the landmark elsewhere or outside.
    wrong=sitk.AffineTransform(transform.GetInverse())
    wrong_sampled=module.warp_native(moving,moving_affine,fixed,wrong,sitk.sitkNearestNeighbor)
    assert wrong_sampled[tuple(target_ijk)]==0


def test_oblique_image_conversion_preserves_physical_lps_locations():
    angle=.37;rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    affine=np.eye(4);affine[:3,:3]=rotation@np.diag([.7,1.3,2.1]);affine[:3,3]=[71.,-33.,104.]
    image=module.image_from_ras(np.zeros((8,9,10),np.float32),affine)
    for voxel in [(0,0,0),(3,4,5),(7,8,9)]:
        expected=module.FRAME_FLIP@affine@np.r_[voxel,1]
        assert np.allclose(image.TransformIndexToPhysicalPoint(voxel),expected[:3],atol=1e-10)


def test_review_axis_permutation_preserves_boolean_landmark_physical_location():
    affine=np.array([[0,0,1.2,-50],[-.9,0,0,80],[0,1.1,0,-20],[0,0,0,1.]])
    data=np.zeros((4,5,6),bool);data[1,2,3]=True
    result,new_affine=module.canonical_array(data,affine)
    coordinate=np.argwhere(result>0)
    assert result.sum()==1 and len(coordinate)==1
    assert np.allclose(new_affine@np.r_[coordinate[0],1],affine@np.array([1,2,3,1.]))


def test_capsule_fov_uses_true_oblique_face_normal_and_complete_radius():
    affine=np.array([[1.,.4,0,8],[0,2.,.3,-5],[0,0,3.,12],[0,0,0,1.]])
    inverse=np.linalg.inv(affine);normal=np.linalg.norm(inverse[0,:3])
    # The source x=0 face is at native coordinate -.5, not the RAS AABB.
    point=(affine@np.array([0.,5.,5.,1.]))[:3]
    expected=.5/normal-.2
    assert module.capsule_native_fov_margin((12,12,12),affine,point,point,.2)==pytest.approx(expected)
    assert module.capsule_native_fov_margin((12,12,12),affine,point,point,.5/normal+.01)<0
    boundary=(affine@np.array([-.5,5.,5.,1.]))[:3]
    assert module.capsule_native_fov_margin((12,12,12),affine,boundary,boundary,0)==pytest.approx(0)


def test_fixed_translation_route_preserves_endpoints_and_bounded_steps():
    a=np.array([1.,2.,3.]);b=np.array([2.01,2.3,2.8])
    points=module.fixed_translation_route(a,b)
    assert np.array_equal(points[0],a) and np.array_equal(points[-1],b)
    assert np.max(np.linalg.norm(np.diff(points,axis=0),axis=1))<=.25
    assert np.allclose(points,module.fixed_translation_route(b,a)[::-1])
    with pytest.raises(ValueError):module.fixed_translation_route(a,b,interval_mm=0)


def test_separate_exact_native_export_authority_required(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'checked_root', lambda *args, **kwargs: tmp_path)
    base = tmp_path / 'integrated-v1/healthy/IXI077'
    registration = base / 'rigid-v1/registration.json'
    proposal = base / 'source-reviewed-superficial-v1/proposal.json'
    registration.parent.mkdir(parents=True)
    proposal.parent.mkdir()
    registration.write_text('{}')
    registration_sha = module.sha256_file(registration)
    proposal.write_text(json.dumps({'registration_sha256': registration_sha,
                                    'target_sha256': 'a' * 64, 'support_sha256': 'b' * 64}))
    stage = 'conditional_exposed_brain_geometry'
    alternate = {'target_proposal_version': proposal.parent.name, 'anatomy_stage': stage}
    with pytest.raises(ValueError, match='require.*explicit conditional stage'):
        module.build_native(tmp_path, 'IXI077', target_proposal_version=proposal.parent.name)
    with pytest.raises(ValueError, match='separate native export decisions'):
        module.build_native(tmp_path, 'IXI077', **alternate)
    reviewed_registration = tmp_path / 'registration-decision.json'
    reviewed_target = tmp_path / 'target-decision.json'
    prospective_export = tmp_path / 'native-export-decision.json'
    reviewed_registration.write_text(json.dumps({
        'reviewer': 'fixture reviewer', 'decision': 'accepted_for_bounded_engineering_alignment',
        'registration_sha256': registration_sha}))
    reviewed_target.write_text(json.dumps({
        'reviewer': 'fixture reviewer', 'target_contract_accepted': True, 'native_export_approved': False,
        'decision': 'accepted_with_explicit_limits_for_exact_source_reviewed_synthetic_geometry_target',
        'proposal_sha256': module.sha256_file(proposal), 'target_sha256': 'a' * 64, 'support_sha256': 'b' * 64}))
    prospective_export.write_text(json.dumps({
        'schema': 'independent-IXI077-source-reviewed-superficial-native-export-authority-v1',
        'decision': 'accepted_for_one_exact_native_geometry_export_with_explicit_limits',
        'reviewer': 'fixture reviewer', 'native_export_approved': True, 'subject': 'IXI077', 'stage': stage,
        'proposal_sha256': module.sha256_file(proposal), 'registration_sha256': registration_sha,
        'target_decision_sha256': 'c' * 64, 'registration_decision_sha256': module.sha256_file(reviewed_registration)}))
    with pytest.raises(ValueError, match='missing exact source and stage bindings'):
        module.build_native(tmp_path, 'IXI077', registration_decision=reviewed_registration,
                            target_decision=reviewed_target, native_export_decision=prospective_export, **alternate)
