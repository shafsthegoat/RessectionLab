"""Small generated geometry controls; no patient or execution admission."""
import importlib.util
from pathlib import Path
import numpy as np
from resectionlab.geometry import segment_segment_distance

spec=importlib.util.spec_from_file_location('integrated_access',Path(__file__).parents[1]/'scripts/data_integrated_access.py')
access=importlib.util.module_from_spec(spec);spec.loader.exec_module(access)


def test_distance_uses_whole_source_voxel_boxes():
    centers=np.array([[2.,0.,0.]])
    assert access.distance_to_boxes(np.zeros(3),centers,np.array([.5,1.,1.]))==1.5
    assert access.distance_to_boxes(np.array([2.,0.,0.]),centers,np.array([.5,1.,1.]))==0.


def test_finite_path_can_steer_around_a_local_rim_without_smaller_tool():
    clear={(0,0,0):1.8,(1,0,0):1.1,(2,0,0):1.8,(0,1,0):1.8,(1,1,0):1.7,(2,1,0):1.6}
    path,bottleneck=access.widest_path(clear,(0,0,0),{(2,0,0)})
    assert bottleneck==1.6 and (1,0,0) not in path
    assert access.shortest_free_path(clear,(0,0,0),{(2,0,0)},1.5)==path


def test_complete_local_barrier_yields_no_fixed_tip_path():
    clear={(0,0,0):1.8,(1,0,0):1.2,(2,0,0):1.8}
    assert access.shortest_free_path(clear,(0,0,0),{(2,0,0)},1.5)==[]
    assert access.widest_path(clear,(0,0,0),{(2,0,0)})[1]==1.2


def test_inward_rounding_covers_closed_disk_for_lipschitz_certificate():
    step=1/16
    theta=np.linspace(0,2*np.pi,1001)
    points=np.column_stack([np.cos(theta),np.sin(theta)])
    rounded=np.trunc(points/step)*step
    assert np.all(np.linalg.norm(rounded,axis=1)<=1+1e-12)
    assert np.all(np.linalg.norm(points-rounded,axis=1)<=np.sqrt(2)*step+1e-12)


def test_stock_kerf_is_thin_crossbars_not_a_one_mm_sphere():
    cover=access.envelopes(access.TOOLS['craniotome'],{'radiusMm':1.,'depthMm':2.},[0.,0.,0.],[0.,0.,0.,1.])
    working=[part for part in cover if part[3]=='working']
    point=np.array([0.,.8,0.])
    assert np.linalg.norm(point)<1.
    assert len(working)==2
    assert all(segment_segment_distance(point,point,a,b)>radius for a,b,radius,_ in working)


def test_kerf_preview_requires_exact_bone_only_progress_before_commit():
    import copy
    before={'removedByLabelMm3':{'brain':.37,'estimated_skull':168.},'targetRemovedMm3':0,'vesselEncounters':0}
    policy={'authorized_kerf_source_cells':[12,13],'budgets':{'maximum_new_bone_mm3':.25,'maximum_new_brain_mm3':34}}
    preview={'status':'accepted','reason':None,'deltas':{'removedCells':[12,13]},
        'metrics':{**before,'removedByLabelMm3':{'brain':.37,'estimated_skull':168.25}}}
    assert access.preview_permitted('kerf',before,preview,policy,before)
    other=copy.deepcopy(preview);other['deltas']['removedCells'].append(99)
    assert not access.preview_permitted('kerf',before,other,policy,before)
    other=copy.deepcopy(preview);other['metrics']['removedByLabelMm3']['brain']+=.01
    assert not access.preview_permitted('kerf',before,other,policy,before)
    other=copy.deepcopy(preview);other['status']='partial'
    assert not access.preview_permitted('kerf',before,other,policy,before)


def test_spatial_schedule_is_not_determined_by_nominal_velocity_alone():
    pose=lambda x:{'position':[x,0.,0.],'quaternion':[0.,0.,0.,1.]}
    old=access.scheduled_segment(pose(0),pose(28),2.,[.5,.5,.75])
    new=access.scheduled_segment(pose(0),pose(11.912728811956812),1.,[.5,.5,.75])
    assert len(old)==168 and len(new)==72
    assert np.linalg.norm(np.subtract(new[-1]['position'],new[-2]['position']))<.5/3
    assert access.scheduled_segment(pose(0),pose(0),2/169,[.5,.5,.75])==[pose(0)]


def test_conservative_midpoint_cover_can_refuse_clear_sphere_translation():
    first={'position':[0.,0.,0.],'quaternion':[0.,0.,0.,1.]}
    last={'position':[0.,0.,.16],'quaternion':[0.,0.,0.,1.]}
    point=np.array([1.57,0.,.08])
    assert segment_segment_distance(point,point,np.array(first['position']),np.array(last['position']))>1.5
    cover=access.swept_envelopes(access.TOOLS['cusa'],{'radiusMm':1.5,'contactGeometryRevision':2},first,last)
    a,b,radius,_=next(c for c in cover if c[3]=='working')
    assert segment_segment_distance(point,point,a,b)<radius


def test_continuous_kerf_plane_filter_intersects_both_motion_directions():
    assert access.linear_filter_interval(0.,1.,1.,2.,.25)==(.375,.625)
    assert access.linear_filter_interval(0.,1.,-1.,-2.,.25)==(.375,.625)
    assert access.linear_filter_interval(0.,1.,.25,0.,.25)==(0.,1.)
    assert access.linear_filter_interval(0.,1.,.26,0.,.25)is None


def test_nonzero_tiny_filter_motion_is_not_rounded_to_stationary():
    assert access.linear_filter_interval(0.,1.,1e-16,2e-16,0.)==(.5,.5)
