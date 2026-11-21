"""Distal profiles, passive guards and simultaneous jaw/rigid sweeps, in mm."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from resectionlab.geometry import point_segment_distances
from resectionlab.surgical_motion import envelopes, pair_conflict, swept_envelopes
from resectionlab.surgical_tools import catalog, operative_tool_parameters, tools_for_operative_profile


def model(name, jaw=2.):
    return tools_for_operative_profile(True)[name], operative_tool_parameters(name,{'jawOpeningMm':jaw})


def test_source_opt_in_and_stable_contact_geometry():
    assert len(catalog())==17 and all('physicalProfile' not in row for row in catalog())
    full=catalog(operative=True)
    assert len(full)==21 and len({row['id'] for row in full})==21
    for row in full:
        tool,params=model(row['id'])
        parts=envelopes(tool,params,[0,0,0],[0,0,0,1])
        assert len(row['physicalProfile']['parts'])==len(parts)
        assert {part for *_,part in parts}=={'shaft','working'}
        assert all(np.linalg.norm(a)<=42 and np.linalg.norm(b)<=42 and r>0 for a,b,r,_ in parts)
    assert 'needle_driver' not in tools_for_operative_profile(False)
    assert operative_tool_parameters('cusa',{'radiusMm':12.})['radiusMm']==1.5


def test_guard_cannot_cut_and_tip_clearance_does_not_clear_the_craniotome():
    tool,p=model('craniotome')
    cover=envelopes(tool,p,[0,0,0],[0,0,0,1])
    marker=np.array([1.55,0,.7])
    obstacle=[(marker,marker,.1,'shaft')]
    assert not pair_conflict([part for part in cover if part[-1]=='working'],obstacle)
    assert pair_conflict(cover,obstacle)
    # The same guard rotates with the roll of the entire instrument.
    rotated=envelopes(tool,p,[0,0,0],Rotation.from_euler('z',180,degrees=True).as_quat())
    assert not pair_conflict(rotated,obstacle)


@pytest.mark.parametrize('name',['bipolar_forceps','dural_scissors','bone_flap_holder','raney_clip_applier'])
def test_continuous_rotation_translation_and_aperture_are_inside_swept_cover(name):
    tool,p=model(name,jaw=8.)
    p['operativeJawSweepFromMm']=0.
    poses=[{'position':[0.,0.,0.],'quaternion':[0.,0.,0.,1.]},
           {'position':[.2,-.1,.1],'quaternion':Rotation.from_euler('xyz',[.03,-.015,.025]).as_quat()}]
    envelope=swept_envelopes(tool,p,*poses)
    rot=Slerp([0.,1.],Rotation.from_quat([v['quaternion'] for v in poses]))
    for t in np.linspace(0,1,17):
        _,live=model(name,jaw=8*t)
        at=envelopes(tool,live,t*np.asarray(poses[1]['position']),rot(t).as_quat())
        for (x,y,r,role),(a,b,inflated,check_role) in zip(at,envelope):
            assert role==check_role
            assert np.max(point_segment_distances([x,y],a,b))+r<=inflated+1.e-10


def test_stationary_opening_checks_obstacle_between_clear_jaw_endpoints():
    origin={'position':[0.,0.,0.],'quaternion':[0.,0.,0.,1.]}
    tool,closed=model('bipolar_forceps',0.); _,opened=model('bipolar_forceps',8.)
    center=np.array([2.,0.,0.]); obstruction=[(center,center,.05,'shaft')]
    assert not pair_conflict(envelopes(tool,closed,**origin),obstruction)
    assert not pair_conflict(envelopes(tool,opened,**origin),obstruction)
    sweep=swept_envelopes(tool,{**opened,'operativeJawSweepFromMm':0.},origin,origin)
    assert pair_conflict(sweep,obstruction)


@pytest.fixture(scope='module')
def generated_field():
    from scripts.generated_glioma_walkthrough_case import build_glioma_case
    return build_glioma_case()[:2]


def test_public_aperture_sweep_protection_rolls_back_candidate(generated_field):
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_rl_protection import GuardConfig, ProtectionAnnotation, ProtectionSet
    case,scene=generated_field
    # Independent exclusion in free space isolates a jaw from ordinary anatomy.
    world=np.array([35,32,74,1.])
    ijk=tuple(np.rint((scene.inverse@world)[:3]).astype(int))
    mask=np.zeros(scene.shape,bool);mask[ijk]=True
    annotation=ProtectionAnnotation.from_mask(scene,'jaw_test_obstacle',mask,provenance='synthetic sweep control')
    protection=ProtectionSet(scene,[annotation],clearance_mm=0.,required=())
    engine=SurgicalEngine(scene,protection=protection,guard_config=GuardConfig(require_critical_coverage=False))
    action=lambda op,**kw:dict(slot='left',operation=op,duration=.1,**kw)
    assert engine.step(action('exchange',instrument='bipolar_forceps',parameters={'jawOpeningMm':0.}))['status']=='accepted'
    pose={'position':[32.,32.,74.],'quaternion':[1.,0.,0.,0.]}
    assert engine.step(action('place',pose=pose))['status']=='accepted'
    source=engine.state_id
    command=action('move',parameters={'jawOpeningMm':10.})
    # The fully open tip and shaft clear the obstacle. Only the intervening
    # aperture places the positive tip at its declared location.
    tool,full=model('bipolar_forceps',10.)
    assert all(not len(protection.capsule_hits(a,b,r)) for a,b,r,_ in envelopes(tool,full,**pose))
    preview=engine.preview(command,include_view=False)
    assert engine.state_id==source and preview['status']=='rejected'
    assert preview['reason'].startswith('PROTECTED_STRUCTURE')
    committed=engine.step(command)
    assert committed['status']=='rejected' and committed['stateId']==source
    assert engine.parameters(engine.state['slots']['left'])['jawOpeningMm']==0.


def test_public_new_tools_require_source_opt_in_and_valid_exterior(generated_field):
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_scene import SurgicalScene
    from resectionlab.core import thaw_json
    from resectionlab.surgical_scene import from_case
    from dataclasses import replace
    case,scene=generated_field
    bare=SurgicalScene(np.zeros((9,9,9),np.uint16),np.eye(4),
                       [{'id':1,'name':'generic material','material':'soft_tissue'}],{}, {})
    engine=SurgicalEngine(bare)
    before=engine.state_id
    refused=engine.step(dict(operation='exchange',slot='left',instrument='needle_driver',duration=.1))
    assert refused['status']=='rejected' and 'UNKNOWN_INSTRUMENT' in refused['reason']
    assert engine.state_id==before
    metadata=thaw_json(case.metadata)
    metadata['surgical_properties']['operativeExposureV1']['exterior']['normalRas']=[0.,0.,0.]
    invalid=from_case(replace(case,metadata=metadata),scene.regions)
    with pytest.raises(ValueError,match='OPERATIVE_EXTERIOR_PLANE_REQUIRED'):
        SurgicalEngine(invalid)
