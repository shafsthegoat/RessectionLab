"""Assumed research laws: deterministic transitions, not physiology validation."""
import copy
import math
import numpy as np
import pytest
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics import SCHEMA, assumed_wound_flow, physics_metrics, physics_descriptor


def profile():
    return {'schema':SCHEMA,'provenance':{'kind':'assumed-research','description':'arbitrary engineering fixture assumptions'},
            'bone':{'model':'rigid-cell-removal-v1'},
            'vessel':{'model':'local-poiseuille-v1','radiusMm':.1,'pressurePa':10.,'viscosityPaS':.01,
                      'pathLengthMm':1.,'maxFlowMm3PerSecond':1.,'coagulationExposure':.1}}


def make_scene(properties=None,material='vessel'):
    labels=np.zeros((24,24,20),np.uint16);labels[8,7:10,8]=1
    return SurgicalScene(labels,np.eye(4),[{'id':1,'name':'Engineering strip','material':material}],{},
                         {'tissuePhysics':profile()} if properties is None else properties)


def setup(e,tool):
    assert e.step({'slot':'left','operation':'exchange','instrument':tool})['status']=='accepted'
    assert e.step({'slot':'left','operation':'place','pose':{'position':[8.,8.,4.],'quaternion':[0.,0.,0.,1.]},
                   'parameters':{'radiusMm':1.5,'jawOpeningMm':0.}})['status']=='accepted'


def cut_action():
    return {'slot':'left','operation':'apply','pose':{'position':[8.,8.,7.],'quaternion':[0.,0.,0.,1.]},'duration':1.}


def test_declared_flow_units_and_explicit_cap():
    p=profile()['vessel'];assert assumed_wound_flow(p)==pytest.approx(math.pi*.1**4*10/(8*.01*1))
    p['pressurePa']=1e300;p['radiusMm']=1e300
    assert assumed_wound_flow(p)==pytest.approx(1.)
    p['pressurePa']=0.;assert assumed_wound_flow(p)==0.


@pytest.mark.parametrize('mutation',[
    lambda p:p['vessel'].pop('radiusMm'), lambda p:p['vessel'].update(viscosityPaS=0.),
    lambda p:p['vessel'].update(pressurePa=float('nan')), lambda p:p['vessel'].update(pathLengthMm=-1.),
    lambda p:p.update(provenance={'kind':'patient-calibrated','description':'unsupported'}),
    lambda p:p.update(schema='unknown'),
])
def test_invalid_profile_rejected_before_any_action(mutation):
    p=profile();mutation(p)
    with pytest.raises(ValueError): SurgicalEngine(make_scene({'tissuePhysics':p}))


def test_damage_creates_local_wound_only_once_preview_and_replay():
    e=SurgicalEngine(make_scene());setup(e,'micro_scissors')
    assert not e.state['sources'] and e.metrics()['fluidAddedMm3']==0.
    before=e.state_id;preview=e.preview(cut_action())
    assert preview['status']=='accepted' and e.state_id==before and not e.state['sources']
    result=e.step(cut_action());assert result['status']=='accepted',result
    assert len(e.state['vesselConnections'])==1 and len(e.state['sources'])==1
    connection=next(iter(e.state['vesselConnections'].values()))
    assert connection['patent'] is False and connection['perfusion']=='unknown'
    assert not e.state['removed']
    assert e.step({'slot':'left','operation':'apply','duration':.5})['status']=='accepted'
    assert len(e.state['sources'])==1
    assert physics_metrics(e)['modeledVesselFluidGeneratedMm3']>0.
    assert SurgicalEngine.restore(e.scene,e.snapshot()).state_id==e.state_id
    m=e.metrics();assert m['fluidAddedMm3']==pytest.approx(m['fluidPresentMm3']+m['fluidRemovedMm3']+m['fluidEscapedMm3'])


def test_coagulation_seals_wound_without_repairing_connection():
    e=SurgicalEngine(make_scene());setup(e,'micro_scissors');assert e.step(cut_action())['status']=='accepted'
    assert e.step({'slot':'left','operation':'withdraw'})['status']=='accepted'
    setup(e,'bipolar_forceps');assert e.step(cut_action())['status']=='accepted'
    assert not e.state['sources'][0]['bleedingActive']
    assert e.state['sources'][0]['patent'] is False
    assert next(iter(e.state['vesselConnections'].values()))['continuity']=='severed'
    generated=physics_metrics(e)['modeledVesselFluidGeneratedMm3']
    assert e.step({'slot':'right','operation':'inspect'})['status']=='accepted'
    assert physics_metrics(e)['modeledVesselFluidGeneratedMm3']==generated
    assert physics_descriptor(e)['capabilities']['patientCalibratedPhysiology'] is False


def test_no_profile_preserves_legacy_missing_strand_error_and_unknown_material():
    e=SurgicalEngine(make_scene({}));setup(e,'micro_scissors')
    assert e.step(cut_action())['reason']=='MISSING_STRAND_PROPERTIES'
    assert 'vesselConnections' not in e.state
    unknown=SurgicalEngine(make_scene(material='unknown'));setup(unknown,'micro_scissors')
    result=unknown.step(cut_action());assert result['status'] in ('partial','rejected')
    assert not unknown.state['vesselConnections'] and not unknown.state['sources']


def test_wound_fluid_can_be_aspirated_without_tissue_or_patency_restoration():
    e=SurgicalEngine(make_scene());setup(e,'micro_scissors');assert e.step(cut_action())['status']=='accepted'
    assert e.step({'slot':'left','operation':'withdraw'})['status']=='accepted'
    setup(e,'frazier_suction')
    result=e.step({'slot':'left','operation':'apply','pose':{'position':[8.,8.,6.8],'quaternion':[0.,0.,0.,1.]},'duration':1.})
    assert result['status']=='accepted',result
    assert e.metrics()['fluidRemovedMm3']>0. and not e.state['removed']
    m=e.metrics();assert m['fluidAddedMm3']==pytest.approx(m['fluidPresentMm3']+m['fluidRemovedMm3']+m['fluidEscapedMm3'])
    assert next(iter(e.state['vesselConnections'].values()))['patent'] is False
    assert SurgicalEngine.restore(e.scene,e.snapshot()).state_id==e.state_id


def test_enclosed_modeled_source_preserves_conserved_reservoir():
    from resectionlab.surgical_physics import record_vessel_transections
    from resectionlab.surgical_effects import tick_fields
    labels=np.ones((5,5,5),np.uint16)
    e=SurgicalEngine(SurgicalScene(labels,np.eye(4),[{'id':1,'name':'Enclosed engineering volume','material':'vessel'}],{}, {'tissuePhysics':profile()}))
    a,b=[int(np.ravel_multi_index(p,labels.shape)) for p in [(2,2,2),(2,3,2)]]
    record_vessel_transections(e,[(a,b)]);tick_fields(e,2.)
    source=e.state['sources'][0]
    assert source['reservoirMm3']==pytest.approx(2*assumed_wound_flow(profile()['vessel']))
    assert not e.state['fluid']
    assert e.metrics()['fluidAddedMm3']==pytest.approx(e.metrics()['fluidPresentMm3'])


def test_large_bone_components_are_bounded_summaries_with_unknown_anchoring():
    from resectionlab.surgical_effects import finalize_action
    labels=np.ones((20,20,20),np.uint16)
    e=SurgicalEngine(SurgicalScene(labels,np.eye(4),[{'id':1,'name':'Bone block','material':'bone'}],{}, {'tissuePhysics':profile()}))
    e.state['boneTopologyDirty']=True;finalize_action(e)
    component=e.state['flapComponents'][0]
    assert component['cellCount']==8000 and component['cells']==[] and component['cellsOmitted']
    assert component['boundsIjk']==[[0,0,0],[19,19,19]] and component['separable'] is None


def test_legacy_bone_component_checkpoint_shape_is_preserved():
    from resectionlab.surgical_effects import finalize_action
    e=SurgicalEngine(make_scene({}, material='bone'))
    e.state['boneTopologyDirty']=True;finalize_action(e)
    assert e.state['flapComponents']==[{'cells':list(map(int,np.flatnonzero(e.scene.labels))), 'separable':None}]


def test_profiled_drill_refreshes_bone_components_and_preview_remains_isolated():
    source=make_scene(material='bone'); labels=source.labels.copy();labels[16,8,8]=1
    e=SurgicalEngine(SurgicalScene(labels,source.affine,source.regions,{},source.properties))
    setup(e,'cranial_drill');before=copy.deepcopy(e.state)
    preview=e.preview(cut_action())
    assert e.state==before and preview['view']['flapComponents']
    assert e.step(cut_action())['stateId']==preview['stateId']
    retained=sum(c['cellCount'] for c in e.state['flapComponents'])
    assert retained+len(e.state['removed'])==np.count_nonzero(e.scene.labels)
