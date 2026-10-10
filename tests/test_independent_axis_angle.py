"""Small generated geometry only; no acquired masks, task or policy execution."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
from resectionlab.geometry import AccessWindow,ToolGeometry,ToolPose
from resectionlab.evaluation import independent_check_motion,_pose_values,_interpolate_axis

TOOL=ToolGeometry('angle_control',.25,.35,8.,tip_length_mm=1.)

def scene():
    affine=np.eye(4);affine[:3,3]=-8
    return SimpleNamespace(forbidden_mask=np.zeros((17,17,17),bool),affine=affine,
        sphere_obstacles=(),enforce_tip_in_bounds=True)

@pytest.mark.parametrize('axis',[(1.,2.,3.),(-.5174147942087322,.854647792855664,.04311706043873262),(0.,0.,1.)])
def test_exact_identical_axis_never_invents_rotation(axis):
    start=ToolPose(np.zeros(3),axis);end=ToolPose(2*start.axis_unit,axis)
    a0=_pose_values(start)[1];a1=_pose_values(end)[1]
    assert np.array_equal(a0,a1)
    result=independent_check_motion(TOOL,start,end,scene(),AccessWindow(np.zeros(3),a0,3.))
    assert result.feasible,result.failures

def test_roundoff_fixture_reproduces_original_false_rotation():
    axis=(-.5174147942087322,.854647792855664,.04311706043873262)
    a=_pose_values(ToolPose(np.zeros(3),axis))[1]
    assert math.acos(float(np.clip(a@a,-1.,1.)))>1e-10
    assert math.atan2(float(np.linalg.norm(np.cross(a,a))),float(a@a))==0.

@pytest.mark.parametrize('angle',[1e-8,1e-6,.02])
def test_genuine_rotation_stays_unsupported_with_access(angle):
    start=ToolPose(np.zeros(3),(0.,0.,1.))
    end=ToolPose((0.,0.,2.),(math.sin(angle),0.,math.cos(angle)))
    result=independent_check_motion(TOOL,start,end,scene(),AccessWindow(np.zeros(3),(0.,0.,1.),3.))
    assert result.failures==('continuous_rotating_access_window_check_unsupported',)

def test_antipodal_rotation_stays_rejected():
    result=independent_check_motion(TOOL,ToolPose(np.zeros(3),(0.,0.,1.)),
        ToolPose(np.zeros(3),(0.,0.,-1.)),scene())
    assert result.failures==('ambiguous_antipodal_rotation',)

def test_tiny_true_rotation_interpolation_does_not_collapse():
    angle=1e-8
    midpoint=_interpolate_axis(np.array([0.,0.,1.]),np.array([math.sin(angle),0.,math.cos(angle)]),.5)
    np.testing.assert_allclose(midpoint,[math.sin(angle/2),0.,math.cos(angle/2)],rtol=0,atol=1e-16)

def test_interpolation_antipodal_stays_refused():
    with pytest.raises(ValueError,match='Antipodal'):
        _interpolate_axis(np.array([0.,0.,1.]),np.array([0.,0.,-1.]),.5)

def test_identical_axis_still_checks_swept_collision():
    s=scene();s.forbidden_mask[8,8,9]=True
    result=independent_check_motion(TOOL,ToolPose((0.,0.,0.),(0.,0.,1.)),
        ToolPose((0.,0.,2.),(0.,0.,1.)),s,AccessWindow(np.zeros(3),(0.,0.,1.),3.))
    assert not result.feasible
    assert any('collision' in reason for reason in result.failures)

@pytest.mark.parametrize('clearance,expected',[(1e-6,True),(-1e-6,False)])
def test_aperture_boundary_is_not_relaxed(clearance,expected):
    x=3.-TOOL.envelope_radius_mm-clearance
    result=independent_check_motion(TOOL,ToolPose((x,0.,0.),(0.,0.,1.)),
        ToolPose((x,0.,2.),(0.,0.,1.)),scene(),AccessWindow(np.zeros(3),(0.,0.,1.),3.))
    assert result.feasible is expected
    if not expected:assert result.failures==('full_tool_does_not_fit_access_window',)

def test_tip_image_bounds_stay_rejected():
    result=independent_check_motion(TOOL,ToolPose((0.,0.,0.),(0.,0.,1.)),
        ToolPose((0.,0.,9.),(0.,0.,1.)),scene())
    assert result.failures==('tip_outside_image_coverage',)
