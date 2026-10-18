"""Mechanism tests on synthetic cells; assertions concern model semantics only."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_effects import apply_effect, initialize_effect_state, tick_fields, validate_hold_motion, finalize_action
from resectionlab.surgical_tools import catalog


def engine(material='soft_tissue', properties=None):
    labels = np.zeros((7, 7, 7), np.uint16); labels[2:5, 2:5, 2:5] = 1
    scene = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'fixture', 'material': material}], {}, properties or {})
    e = SurgicalEngine(scene); initialize_effect_state(e)
    return e


def effect(e, tool, cells=None, p=(3, 3, 3), q=(0, 0, 0, 1), op='apply', dt=1):
    e.state['slots']['left']['instrument'] = tool
    return apply_effect(e, 'left', op, list(map(int, np.flatnonzero(e.scene.labels))) if cells is None else cells, np.array(p, float), q, dt)


def test_catalog_distinct_fifteen():
    tools = catalog()
    assert len(tools) == 15 and len({t['id'] for t in tools}) == 15
    assert len({t['effect'] for t in tools}) == 13


def test_blade_roll_changes_cut_plane_and_retains_tissue():
    e = engine(); other = e.clone()
    effect(e, 'scalpel')
    effect(other, 'scalpel', q=Rotation.from_euler('z', 90, degrees=True).as_quat())
    assert e.state['cutEdges'] and other.state['cutEdges']
    assert e.state['cutEdges'] != other.state['cutEdges']
    assert e.state['removed'] == other.state['removed'] == []
    membrane = engine('membrane'); membrane.state['slots']['left']['parameters']['jawOpeningMm'] = 0.; events = effect(membrane, 'dural_scissors')
    assert events[0]['kind'] == 'sheet_opening' and membrane.state['removed'] == []


def test_attachment_tools_require_properties_and_release_only_declared_kind():
    e = engine()
    with pytest.raises(ValueError, match='MISSING_ATTACHMENT'): effect(e, 'dissector')
    cell = int(np.flatnonzero(e.scene.labels)[0])
    e.scene.properties['attachments'] = [{'id':'p','kind':'periosteal','cells':[cell]}, {'id':'s','kind':'soft','cells':[cell]}]
    effect(e, 'periosteal_elevator')
    assert e.state['separatedAttachments'] == ['p']
    effect(e, 'dissector')
    assert e.state['separatedAttachments'] == ['p', 's'] and not e.state['removed']


def test_bone_kerf_retains_separated_component():
    e = engine('bone'); e.scene.properties['boneAnchors'] = [int(np.ravel_multi_index((2, 2, 2), e.scene.shape))]
    middle = [int(np.ravel_multi_index((x, 3, z), e.scene.shape)) for x in range(2,5) for z in range(2,5)]
    effect(e, 'craniotome', middle); finalize_action(e)
    assert len(e.state['flapComponents']) == 2
    assert sorted(c['separable'] for c in e.state['flapComponents']) == [False, True]
    assert len(e.state['removed']) == 9 and e.state['debrisMm3'] == 9


def test_fluid_mass_conservation_suction_local_and_no_tissue_removal():
    e = engine(); cell = int(np.ravel_multi_index((3,3,1),e.scene.shape))
    e.state['slots']['left']['parameters'] = {'depthMm':1., 'flowMm3PerSecond':4.}
    effect(e, 'irrigation', p=(3,3,0))
    assert e.state['fluid'] == {str(cell):4.}
    e.state['debris'][str(cell)] = 2.; e.state['debrisMm3'] = 2.
    effect(e, 'frazier_suction', p=(0,0,0))
    assert e.state['fluid'][str(cell)] == 4.
    effect(e, 'frazier_suction', p=(3,3,1))
    assert sum(e.state['fluid'].values()) == 0 and e.state['debrisMm3'] == 2.
    effect(e, 'frazier_suction', p=(3,3,1))
    assert e.state['debrisMm3'] == 0 and not e.state['removed']
    effect(e, 'irrigation', p=(3,3,7))
    assert e.state['fluidAddedMm3'] == e.state['fluidRemovedMm3'] + e.state['fluidEscapedMm3'] + sum(e.state['fluid'].values())


def test_bipolar_closes_only_local_declared_source_and_irrigation_does_not():
    e = engine(); ids = list(map(int,np.flatnonzero(e.scene.labels)))
    e.state['sources'] = [{'id':'a','cell':ids[0],'rateMm3PerSecond':1.,'coagulationExposure':2.,'patent':True,'exposure':0.}, {'id':'b','cell':ids[-1],'rateMm3PerSecond':1.,'coagulationExposure':2.,'patent':True,'exposure':0.}]
    effect(e, 'irrigation'); assert all(s['patent'] for s in e.state['sources'])
    effect(e, 'bipolar_forceps', cells=[ids[0]])
    assert e.state['sources'][0]['patent']
    effect(e, 'bipolar_forceps', cells=[ids[0]])
    assert all(s['patent'] for s in e.state['sources'])
    assert not e.state['sources'][0]['bleedingActive'] and e.state['sources'][1].get('bleedingActive', True)
    before = sum(e.state['fluid'].values()); tick_fields(e, 1.)
    assert sum(e.state['fluid'].values()) == before + 1.


def test_micro_scissors_require_strands_and_retain_pieces():
    e = engine('vessel'); e.state['slots']['left']['parameters']['jawOpeningMm'] = 0.
    with pytest.raises(ValueError, match='MISSING_STRAND'): effect(e, 'micro_scissors')
    ids = list(map(int,np.flatnonzero(e.scene.labels))); e.scene.properties['strands']=[{'id':'s','cells':ids[:3]}]
    effect(e, 'micro_scissors', cells=[ids[1]], q=Rotation.from_euler('x',90,degrees=True).as_quat())
    assert len(e.state['transectedEdges']) == 1 and not e.state['removed']


def test_forceps_fragment_identity_attachment_and_extraction():
    e = engine('tumor', {'attachments':[]}); e.state['slots']['left']['parameters']['jawOpeningMm'] = 0.; ids = list(map(int,np.flatnonzero(e.scene.labels)))
    effect(e, 'tumor_forceps', cells=[ids[0]], op='grasp')
    assert e.state['holds']['left']['cells'] == ids
    e.scene.properties['attachments'] = [{'id':'a','kind':'tumor','cells':[ids[-1]],'breakable':False}]
    with pytest.raises(ValueError, match='ATTACHMENT_PREVENTS'): effect(e, 'tumor_forceps', op='extract')
    with pytest.raises(ValueError, match='ATTACHMENT_PREVENTS'): validate_hold_motion(e, 'left')
    e.scene.properties['attachments'][0]['breakable'] = True
    validate_hold_motion(e, 'left'); effect(e, 'tumor_forceps', op='extract')
    assert e.state['removed'] == ids and not e.state['holds']


def test_probes_distinct_supplied_evidence_or_unknown():
    e = engine(); cell=int(np.flatnonzero(e.scene.labels)[0])
    evidence={'id':'e','cell':cell,'probe':'cortical','result':'test-only','uncertainty':'.5'}
    e.scene.properties['functionalEvidence']=[evidence]
    effect(e, 'cortical_probe'); effect(e, 'subcortical_probe')
    a,b=e.state['observations']
    assert a['evidence']==[evidence] and b['result']=='unknown'
    assert not e.state['removed'] and not e.state['thermal']


def test_open_jaws_do_not_transect_and_source_reservoir_conserves():
    e = engine('vessel'); ids = list(map(int,np.flatnonzero(e.scene.labels)))
    e.scene.properties['strands'] = [{'id':'s','cells':ids}]
    effect(e, 'micro_scissors')
    assert not e.state['transectedEdges']
    center = int(np.ravel_multi_index((3,3,3),e.scene.shape))
    e.state['sources'] = [{'id':'enclosed','cell':center,'rateMm3PerSecond':2.,'patent':True}]
    tick_fields(e, 1.)
    assert e.state['sources'][0]['reservoirMm3'] == 2. and not e.state['fluid']
    e.state['removed'] = [int(np.ravel_multi_index((3,3,2),e.scene.shape))]
    tick_fields(e, 1.)
    assert e.state['fluidAddedMm3'] == sum(e.state['fluid'].values()) + e.state['fluidEscapedMm3'] + e.state['sources'][0]['reservoirMm3']
    assert e.state['sources'][0]['patent']


def test_boundary_overflow_accounted_as_escape():
    e = engine(); e.state['fluid'] = {'0': 20.}; e.state['fluidAddedMm3'] = 20.
    tick_fields(e, .1)
    assert e.state['fluidEscapedMm3'] > 0
    assert sum(e.state['fluid'].values()) + e.state['fluidEscapedMm3'] == 20.


@pytest.mark.parametrize('tool,material,field', [
    ('cranial_drill','bone','removed'), ('cusa','tumor','removed'),
    ('brain_retractor','soft_tissue','holds'), ('scalpel','soft_tissue','cutEdges'),
    ('periosteal_elevator','bone','separatedAttachments'), ('craniotome','bone','removed'),
    ('dural_scissors','membrane','cutEdges'), ('dissector','soft_tissue','separatedAttachments'),
    ('frazier_suction','soft_tissue','fluidRemovedMm3'), ('bipolar_forceps','vessel','thermal'),
    ('micro_scissors','vessel','transectedEdges'), ('tumor_forceps','tumor','holds'),
    ('irrigation','soft_tissue','fluidAddedMm3'), ('cortical_probe','soft_tissue','observations'),
    ('subcortical_probe','soft_tissue','observations'),
])
def test_engine_step_instrument_admissibility(tool, material, field):
    labels = np.zeros((20,20,20), np.uint16); labels[7:10,7:10,8] = 1
    ids = list(map(int,np.flatnonzero(labels)))
    properties = {'attachments':[{'id':'p','kind':'periosteal','cells':ids}, {'id':'s','kind':'soft','cells':ids}], 'strands':[{'id':'strand','cells':ids}]}
    e = SurgicalEngine(SurgicalScene(labels,np.eye(4),[{'id':1,'name':'fixture','material':material}],{},properties))
    # Both slots start outside, and exchange/place/apply all traverse normal API.
    assert e.step({'slot':'left','operation':'exchange','instrument':tool})['status'] == 'accepted'
    pose = lambda z: {'position':[8.,8.,z], 'quaternion':[0.,0.,0.,1.]}
    assert e.step({'slot':'left','operation':'place','pose':pose(4.),'parameters':{'radiusMm':1.5,'jawOpeningMm':0.,'depthMm':2.}})['status'] == 'accepted'
    if tool == 'frazier_suction':
        # Free fluid immediately in front of the exposed surface.
        e.state['fluid'][str(int(np.ravel_multi_index((8,8,6),labels.shape)))] = 10.
        e.state['fluidAddedMm3'] = 10.
    result=e.step({'slot':'left','operation':'apply','pose':pose(6.8 if tool in ('frazier_suction','irrigation') else 7.),'duration':1.})
    assert result['status'] == 'accepted', result
    assert e.state[field], (tool,result)
    if tool not in ('cranial_drill','cusa','craniotome'): assert not e.state['removed']


def test_retained_mesh_sheet_seam_uses_native_lps_microns():
    e=engine('membrane'); e.state['slots']['left']['parameters']={'jawOpeningMm':0.,'radiusMm':3.,'depthMm':3.}
    e.scene.source={'topology':'sheet','units':'micron','frame':'LPS+',
                    'vertices':[[-2000,-2000,3000],[-4000,-2000,3000],[-3000,-4000,3000]],'faces':[[0,1,2]],'lines':[]}
    effect(e,'dural_scissors')
    assert len(e.state['meshCutEdges']) == 2 and len(e.state['meshSeams']) == 1
    seam=e.state['meshSeams'][0]
    assert all(abs(point[1]-3.) < 1e-8 for point in seam['points'])
    assert not e.state['removed'] and e.scene.source['faces'] == [[0,1,2]]


def test_kerf_roll_changes_thin_cut_and_defers_component_work():
    a=engine('bone'); b=a.clone()
    effect(a,'craniotome'); effect(b,'craniotome',q=Rotation.from_euler('z',90,degrees=True).as_quat())
    assert a.state['removed'] != b.state['removed']
    assert len(a.state['removed']) == len(b.state['removed']) == 9
    assert a.state['flapComponents'] == [] and a.state['boneTopologyDirty']
    finalize_action(a); assert len(a.state['flapComponents']) == 2
    assert 'boneTopologyDirty' not in a.state


def test_native_strand_graph_cut_retains_original_vertices():
    e=engine('vessel');e.state['slots']['left']['parameters']['jawOpeningMm']=0.
    e.scene.source={'topology':'strand','units':'mm','frame':'RAS+',
                    'vertices':[[3,2,3],[3,4,3]],'faces':[],'lines':[[0,1]]}
    effect(e,'micro_scissors')
    assert e.state['meshCutEdges'] == [[0,1]] and not e.state['removed']
    assert e.state['meshSeams'][0]['points'] == [[3.,3.,3.]]


def test_displaced_solid_cannot_retain_fluid_and_spill_is_conserved():
    e=engine(); center=int(np.ravel_multi_index((3,3,3),e.scene.shape))
    e.state['fluid']={str(center):2.}; e.state['fluidAddedMm3']=2.
    tick_fields(e,.1)
    assert str(center) not in e.state['fluid']
    assert e.state['fluidEscapedMm3'] + sum(e.state['fluid'].values()) == 2.


def test_nested_case_adapter_mesh_seam_uses_retained_source_geometry():
    e=engine('vessel');e.state['slots']['left']['parameters']['jawOpeningMm']=0.
    e.scene.source={'adapter':{'type':'mesh','topology':'strand','units':'mm','frame':'RAS+',
                              'vertices':[[3,2,3],[3,4,3]],'faces':[],'lines':[[0,1]]}}
    effect(e,'micro_scissors')
    assert e.state['meshCutEdges'] == [[0,1]]


def test_open_jaw_sweep_stops_at_unsupported_material_outside_tip_sphere():
    labels=np.zeros((24,24,20),np.uint16); labels[11,8,6]=1
    e=SurgicalEngine(SurgicalScene(labels,np.eye(4),[{'id':1,'name':'Obstacle','material':'bone'}],{},{}))
    assert e.step({'slot':'left','operation':'exchange','instrument':'micro_scissors'})['status']=='accepted'
    pose=lambda z:{'position':[8.,8.,z],'quaternion':[0.,0.,0.,1.]}
    assert e.step({'slot':'left','operation':'place','pose':pose(2.),'parameters':{'radiusMm':.5,'jawOpeningMm':6.}})['status']=='accepted'
    result=e.step({'slot':'left','operation':'apply','pose':pose(7.)})
    assert result['status'] in ('partial','rejected') and result['reason'] in ('UNSUPPORTED_MATERIAL','WORKING_PART_COLLISION'), result


def test_blade_cut_follows_displaced_source_cells():
    e=engine(); ids=list(map(int,np.flatnonzero(e.scene.labels)))
    e.state['holds']['right']={'cells':ids,'origin':[3.,3.,3.],'offset':[0,2,0]}
    effect(e,'scalpel',p=(3.,5.,3.))
    assert e.state['cutEdges']
    # Every edge straddles the current blade plane, though original y <= 4.
    for a,b in e.state['cutEdges']:
        y=e.scene.points([a,b])[:,1]+2
        assert y.min() <= 5 < y.max()


def _two_slot_coupon(material='tumor'):
    labels=np.zeros((30,24,20),np.uint16);labels[7:18,8,8]=1
    return SurgicalEngine(SurgicalScene(labels,np.eye(4),[{'id':1,'name':'Connected engineering strip','material':material}],{}, {'attachments':[]}))


def _position_slot(e, slot, tool, x):
    assert e.step({'slot':slot,'operation':'exchange','instrument':tool})['status']=='accepted'
    assert e.step({'slot':slot,'operation':'place','pose':{'position':[x,8.,4.],'quaternion':[0.,0.,0.,1.]},
                   'parameters':{'radiusMm':1.5,'jawOpeningMm':0.}})['status']=='accepted'


def test_stationary_active_bipolar_shares_other_slot_clock_and_deactivation():
    e=_two_slot_coupon('vessel');_position_slot(e,'left','bipolar_forceps',8.)
    result=e.step({'slot':'left','operation':'apply','pose':{'position':[8.,8.,7.],'quaternion':[0.,0.,0.,1.]}})
    assert result['status']=='accepted'
    assert e.step({'slot':'left','operation':'activate','duration':.5})['status']=='accepted'
    before=dict(e.state['thermal']); elapsed=e.state['elapsedSeconds']
    assert e.step({'slot':'right','operation':'inspect','duration':2.})['status']=='accepted'
    assert e.state['elapsedSeconds']==pytest.approx(elapsed+2.)
    for cell,value in before.items(): assert e.state['thermal'][cell]==pytest.approx(value+2.)
    assert e.step({'slot':'left','operation':'deactivate'})['status']=='accepted'
    after=dict(e.state['thermal'])
    assert e.step({'slot':'right','operation':'inspect','duration':2.})['status']=='accepted'
    assert e.state['thermal']==after
    assert SurgicalEngine.restore(e.scene,e.snapshot()).state_id==e.state_id


def test_two_forceps_cannot_hold_same_connected_fragment():
    e=_two_slot_coupon();_position_slot(e,'left','tumor_forceps',8.)
    assert e.step({'slot':'left','operation':'grasp','pose':{'position':[8.,8.,7.],'quaternion':[0.,0.,0.,1.]}})['status']=='accepted'
    _position_slot(e,'right','tumor_forceps',16.)
    result=e.step({'slot':'right','operation':'grasp','pose':{'position':[16.,8.,7.],'quaternion':[0.,0.,0.,1.]}})
    assert result['status'] in ('partial','rejected'),result
    assert 'right' not in e.state['holds']
    assert e.state['holds']['left']['cells']==list(map(int,np.flatnonzero(e.scene.labels)))


def test_releasing_hold_does_not_restore_cells_removed_by_other_instrument():
    e=_two_slot_coupon();_position_slot(e,'left','tumor_forceps',8.)
    assert e.step({'slot':'left','operation':'grasp','pose':{'position':[8.,8.,7.],'quaternion':[0.,0.,0.,1.]}})['status']=='accepted'
    displayed=e.view()['holds'];assert displayed['left']['cells']==e.state['holds']['left']['cells']
    displayed['left']['cells'].clear();assert e.state['holds']['left']['cells']
    _position_slot(e,'right','cusa',16.)
    assert e.step({'slot':'right','operation':'apply','pose':{'position':[16.,8.,7.],'quaternion':[0.,0.,0.,1.]}})['status']=='accepted'
    removed=e.state['removed'][:];assert removed
    assert e.step({'slot':'left','operation':'release'})['status']=='accepted'
    assert e.state['removed']==removed and 'left' not in e.state['holds']
    assert e.view()['holds']=={}
    assert all(e.occupancy()[cell]==0 for cell in removed)
    assert SurgicalEngine.restore(e.scene,e.snapshot()).state_id==e.state_id


def test_irrigation_transports_debris_with_conserved_bounded_fluid():
    from resectionlab.surgical_effects import tick_fields
    labels = np.zeros((5, 5, 5), np.uint16)
    e = SurgicalEngine(SurgicalScene(labels, np.eye(4), [], {}, {}))
    center = int(np.ravel_multi_index((2, 2, 2), labels.shape))
    e.state['fluid'][str(center)] = 4.
    e.state['fluidAddedMm3'] = 4.
    e.state['debris'][str(center)] = 2.
    e.state['debrisMm3'] = 2.
    tick_fields(e, .1)
    assert e.state['debris'][str(center)] < 2.
    assert len([v for v in e.state['debris'].values() if v > 0]) > 1
    assert sum(e.state['debris'].values()) + e.state.get('debrisEscapedMm3', 0.) == pytest.approx(2.)
    assert sum(e.state['fluid'].values()) + e.state['fluidEscapedMm3'] == pytest.approx(4.)
