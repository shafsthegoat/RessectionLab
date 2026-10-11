"""Portable generated geometry/nominal-reward bound controls, no patient files."""
from dataclasses import asdict
import numpy as np
import pytest

from resectionlab.public_reward_bound import PublicTargetBoundIndex, RewardUpperBound
from resectionlab.geometry import GeometryScene
from resectionlab.native_resection import contained_capsule_cells
from resectionlab.simulation import RewardSpec

REWARD = asdict(RewardSpec(target_per_mm3=1., normal_per_mm3=.2,
                          action_cost=.03, motion_per_mm=.001, tool_change_cost=.03))


def prepared(t, *, affine=None, domain=None, cavity=None):
    return PublicTargetBoundIndex.prepare(affine=np.eye(4) if affine is None else affine,
        public_target=t, target_domain=np.ones(t.shape, bool) if domain is None else domain,
        observed_cavity=np.zeros(t.shape, bool) if cavity is None else cavity)


def query(index, **changes):
    args=dict(physical_start_mm=(5.,5.,4.), tip_mm=(5.,5.,8.), axis_unit=(0.,0.,1.),
              tip_length_mm=3., tip_radius_mm=1.1, reward=REWARD,
              reward_voxel_volume_mm3=1., tool_change=False)
    args.update(changes)
    return index.bound(**args)


def test_preentry_back_extent_and_endcap_are_not_entry_tip_samples():
    t=np.zeros((13,13,13)); t[5,5,1]=1.; t[5,5,9]=1.
    b=query(prepared(t))
    assert b.potential_target_cells==2 and b.reward_upper>=1.97
    assert not b.cannot_change_winner(0.)


@pytest.mark.parametrize("angle,reflected,spacing", [(0.,False,(1.,1.,1.)),(.41,True,(.4,.7,.6))])
def test_upper_bounds_union_of_actual_full_cell_microcapsules(angle,reflected,spacing):
    # Native containment is the test oracle only, never an input to the bound.
    c,s=np.cos(angle),np.sin(angle)
    rotation=np.array([[c,-s,0],[s,c,0],[0,0,-1 if reflected else 1]])
    a=np.eye(4);a[:3,:3]=rotation @ np.diag(spacing);a[:3,3]=[101.,-23.,8.]
    t=np.full((11,11,11),.37);t[::2,::2,::2]=1.
    cavity=np.zeros(t.shape,bool);cavity[5,5,4]=True
    index=prepared(t,affine=a,cavity=cavity)
    p=a[:3,:3]@np.array([5.,5.,3.])+a[:3,3]
    q=a[:3,:3]@np.array([5.,5.,8.])+a[:3,3]
    axis=(q-p)/np.linalg.norm(q-p); radius=1.4; length=2.
    scene=GeometryScene(np.zeros(t.shape,bool),a)
    candidates=np.argwhere(~cavity); removed=set();previous=p
    for fraction in np.linspace(0,1,17):
        current=p+(q-p)*fraction
        cells=contained_capsule_cells(scene,candidates,previous-length*axis,current,radius)
        removed.update(map(tuple,cells));previous=current
    assert removed
    volume=abs(float(np.linalg.det(a[:3,:3])))
    target=sum(float(t[cell]) for cell in removed)*volume
    actual=target-.2*(len(removed)*volume-target)-.03-.002*np.linalg.norm(q-p)-.03
    b=query(index,physical_start_mm=p,tip_mm=q,axis_unit=axis,tip_length_mm=length,
            tip_radius_mm=radius,reward_voxel_volume_mm3=volume,tool_change=True)
    assert b.potential_target_cells>=len(removed)
    assert b.reward_upper>=actual


def test_unknown_is_not_zero_even_if_cavity_true_and_outside_envelope_is_separate():
    t=np.zeros((13,13,13));d=np.ones(t.shape,bool);c=np.zeros(t.shape,bool)
    d[5,5,5]=False;c[5,5,5]=True
    assert query(prepared(t,domain=d,cavity=c)).reward_upper is None
    d[:]=True;d[0,0,0]=False
    assert query(prepared(t,domain=d)).cannot_change_winner(0.)


def test_snapshot_excludes_only_observed_cavity_and_is_immutable():
    t=np.zeros((13,13,13));t[5,5,5]=1.;c=np.zeros(t.shape,bool)
    before=prepared(t,cavity=c);c[5,5,5]=True;after=prepared(t,cavity=c)
    assert query(before).potential_target_cells==1
    assert query(after).potential_target_cells==0
    assert query(after).cannot_change_winner(0.)
    with pytest.raises(ValueError): before.target_lower[0,0]=0


@pytest.mark.parametrize("bad",[np.nan,-.01,1.01])
def test_invalid_public_membership_never_prunes(bad):
    t=np.zeros((13,13,13));t[0,0,0]=bad
    assert not query(prepared(t)).cannot_change_winner(0.)


@pytest.mark.parametrize("change",[{"reward":{**REWARD,"normal_per_mm3":-.1}},
    {"reward":{**REWARD,"motion_per_mm":float('nan')}}, {"interaction_mode":"probe"},
    {"axis_unit":(0,0,2)}, {"physical_start_mm":(float('inf'),0,0)}, {"tip_radius_mm":0.}])
def test_unsupported_queries_retain_exact_preview(change):
    assert query(prepared(np.zeros((13,13,13))),**change).reward_upper is None


def test_missing_coverage_and_sheared_grid_are_unbounded():
    t=np.zeros((3,3,3));a=np.eye(4);a[0,1]=.01
    assert prepared(t,affine=a).unavailable_reason=='unsupported_native_frame'
    missing=PublicTargetBoundIndex.prepare(affine=np.eye(4),public_target=t,
                                          target_domain=None,observed_cavity=t.astype(bool))
    assert query(missing).reward_upper is None


def test_raw_positive_ties_never_prune_but_STOP_zero_tie_can():
    assert not RewardUpperBound(1.,1,'test').cannot_change_winner(1.)
    assert RewardUpperBound(np.nextafter(1.,0.),1,'test').cannot_change_winner(1.)
    assert RewardUpperBound(0.,0,'test').cannot_change_winner(0.)
    assert not RewardUpperBound(None,None,'unknown').cannot_change_winner(0.)


def test_cancellation_is_not_converted_to_unbounded_success():
    def cancel():raise TimeoutError('generated cap')
    with pytest.raises(TimeoutError,match='generated cap'):
        PublicTargetBoundIndex.prepare(affine=np.eye(4),public_target=np.zeros((3,3,3)),
            target_domain=np.ones((3,3,3),bool),observed_cavity=np.zeros((3,3,3),bool),cancelled=cancel)
    with pytest.raises(TimeoutError,match='generated cap'):
        query(prepared(np.zeros((3,3,3))),cancelled=cancel)
