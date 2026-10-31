"""Subvoxel blind spots, positive Q1 points and conservative separation."""
import numpy as np
import pytest
from itertools import product

from resectionlab.surgical_contact_witness import Q1ContactWitness,REFERENCE_POINTS,WEIGHTS
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.geometry import GeometryScene,capsule_voxel_indices


def cube(displacement=0.,affine=None):
    u=np.zeros((1,8,3));u[:,:,0]=displacement
    return Q1ContactWitness([13],[3,3,3],np.eye(4) if affine is None else affine,u)


def test_subvoxel_translation_changes_true_contact_without_raster_cell_change():
    start=end=np.array([2.04,1.,1.]);rest=cube();shifted=cube(.1)
    grid=GeometryScene(np.zeros((3,3,3),bool),np.eye(4));cover=capsule_voxel_indices(grid,start,end,.5)
    assert not np.any(np.all(cover==[1,1,1],axis=1))
    assert np.array_equal(np.rint([1.1,1.,1.]).astype(int),[1,1,1])
    assert rest.capsule(start,end,.5)['separatedRows'].tolist()==[0]
    result=shifted.capsule(start,end,.5)
    assert result['witnessedRows'].tolist()==[0]
    assert result['witnessDistancesMm'][0]==pytest.approx(.44)
    assert np.linalg.norm(np.asarray(result['witnessPointsMm'][0])-start)<.5


def test_tangent_is_not_promoted_to_positive_or_clear_and_far_geometry_separates():
    q=cube();tangent=q.capsule([2.,1.,1.],[2.,1.,1.],.5)
    assert tangent['witnessedRows'].size==0 and tangent['indeterminateRows'].tolist()==[0]
    assert q.capsule([10.,0.,0.],[10.,0.,3.],.1)['separatedRows'].tolist()==[0]


def test_nonaffine_q1_samples_are_inside_enclosure_and_positive_points_are_actual_material():
    affine=np.array([[-1.,.2,0.,4.],[0.,1.1,.1,-3.],[0.,0.,.9,2.],[0.,0.,0.,1.]])
    rng=np.random.default_rng(723);u=rng.normal(size=(1,8,3))*.02
    q=Q1ContactWitness([13],[3,3,3],affine,u);points=WEIGHTS@q.corners[0]
    assert np.all(points>=q.lower[0]-1e-14) and np.all(points<=q.upper[0]+1e-14)
    chosen=points[13];result=q.capsule(chosen,chosen,.001)
    assert result['witnessedRows'].tolist()==[0]
    ref=np.asarray(result['referencePoints'][0]);weights=np.prod(np.where(np.array(list(product((0.,1.),repeat=3)))>0,ref,1-ref),axis=1)
    np.testing.assert_allclose(weights@q.corners[0],result['witnessPointsMm'][0],atol=1e-13)


def test_blocking_point_and_removed_obstacle_are_separate_from_unknown_coverage():
    labels=np.zeros((3,3,3),np.uint16);labels[1,1,1]=1;labels[2,1,1]=2
    scene=SurgicalScene(labels,np.eye(4),[{'id':1,'material':'soft_tissue'},{'id':2,'material':'vessel'}],{'kind':'generated'}, {})
    # Rightmost source corners move strictly into the adjacent vessel cell.
    q=cube(.1);report=q.obstacle_points(scene,[])
    assert any(w['blockingSourceCell']==22 for w in report['blockingPointWitnesses'])
    assert not report['unknownAnatomyValidated']
    assert q.obstacle_points(scene,[22])['blockingPointWitnesses']==[]


def test_caller_mutation_and_cancellation_do_not_change_owned_geometry():
    u=np.zeros((1,8,3));q=Q1ContactWitness([13],[3,3,3],np.eye(4),u);u[:]=100
    assert np.max(q.corners)<2
    def cancel():raise InterruptedError('Unit geometry cancellation')
    with pytest.raises(InterruptedError):q.capsule([2.,1.,1.],[2.,1.,1.],.5,check=cancel)
    with pytest.raises(ValueError):q.corners[0]=0


@pytest.mark.parametrize('radius',[np.nan,-1.,True,1e308])
def test_invalid_capsule_radius_refuses(radius):
    with pytest.raises(ValueError):cube().capsule([0.,0.,0.],[1.,1.,1.],radius)


def test_obstacle_view_refuses_different_source_frame_or_removed_owners():
    labels=np.ones((3,3,3),np.uint16)
    scene=SurgicalScene(labels,np.eye(4),[{'id':1,'material':'soft_tissue'}],{'kind':'generated'}, {})
    with pytest.raises(ValueError,match='REMOVED_SOURCE_GEOMETRY'):cube().obstacle_points(scene,[13])
    moved=np.eye(4);moved[0,3]=1.
    with pytest.raises(ValueError,match='SCENE_FRAME'):cube(affine=moved).obstacle_points(scene,[])
