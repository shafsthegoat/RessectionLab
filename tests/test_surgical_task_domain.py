"""Whole-instrument task restrictions through the actual desktop transaction API."""
import copy
from contextlib import contextmanager
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab.core import CaseData, SourceRef
from resectionlab.desktop_bridge import BridgeSession, _Request, BridgeError
from resectionlab.imaging import save_case
from resectionlab.surgical_motion import envelopes
from resectionlab.surgical_tools import TOOLS
from resectionlab.surgical_task_domain import reviewed_domain, _outside


def declaration(shape=(100,100,100)):
    return {'mode':'restrict-unknown','shape':list(shape),'minimumVoxelBoundary':[5.,5.,24.],
            'maximumVoxelBoundary':[94.,94.,95.],'reviewEvidenceSha256':'a'*64,'maskSha256':'b'*64}


@contextmanager
def desktop(tmp_path, domain=True):
    shape=(100,100,100);mask=np.zeros(shape,bool);mask[95,95,95]=True
    adapter={} if domain is False else {'reviewedTaskDomainV1':declaration() if domain is True else domain}
    case=CaseData(case_id='generated-task-box',mri=np.zeros(shape,np.float32),affine=np.eye(4),
        compartments={'target':mask},source_refs=(SourceRef('generated','synthetic://task-box',provenance='simulated'),),
        metadata={'is_synthetic':True,'surgical_adapter':adapter,'anatomy_layers':{'target':{'kind':'tumor'}}})
    path=save_case(case,tmp_path/'generated.ressectionlab')
    session=BridgeSession(tmp_path/'transfers')
    try:
        loaded=session.execute('loadCase',{'path':str(path)},_Request('load'),lambda *_:None)
        identity=loaded['caseHash']
        def call(command, previous=None, **extra):
            args={'caseHash':identity,'command':command,**extra}
            if previous is not None:args['expectedStateId']=previous['view']['stateId']
            return session.execute('surgicalRehearsal',args,_Request('tools'),lambda *_:None)
        yield session,identity,call
    finally:
        session.close()


def action(operation,position=None,quaternion=(0.,0.,0.,1.)):
    result={'slot':'right','operation':operation,'duration':.1}
    if position is not None:result['pose']={'position':list(position),'quaternion':list(quaternion)}
    return result


def test_desktop_empty_region_and_whole_shaft_refuse_atomically_but_observed_placement_passes(tmp_path):
    catalog=copy.deepcopy(TOOLS)
    with desktop(tmp_path) as (session,identity,call):
        created=call('create');before=created['view']['stateId']
        engine=session.surgical_sessions[identity][1];labels=engine.scene.labels.copy()
        for pose in ([2.,50.,70.],[50.,50.,50.]):
            # First is inside acquired FOV but outside reviewed box; second has
            # a supported tip but the 40 mm shaft crosses the artificial floor.
            result=call('step',created,action=action('place',pose))
            assert result['outcome']['reasonCode']=='REVIEWED_TASK_DOMAIN_OUTSIDE'
            assert result['outcome']['status']=='rejected' and not result['changed']
            assert result['view']['stateId']==before and result['position']==0
        admitted=call('step',created,action=action('place',[50.,50.,70.]))
        assert admitted['outcome']['status']=='accepted' and admitted['position']==1
        np.testing.assert_array_equal(session.surgical_sessions[identity][1].scene.labels,labels)
    assert TOOLS==catalog


def test_desktop_rotation_with_supported_endpoints_refuses_intermediate_shaft_in_preview_and_step(tmp_path):
    with desktop(tmp_path) as (session,identity,call):
        created=call('create')
        first=Rotation.from_euler('y',53.13010235415598,degrees=True).as_quat()
        second=Rotation.from_euler('y',-53.13010235415598,degrees=True).as_quat()
        placed=call('step',created,action=action('place',[50.,50.,50.],first))
        assert placed['outcome']['status']=='accepted'
        engine=session.surgical_sessions[identity][1];domain=reviewed_domain(engine.scene);slot=engine.state['slots']['right']
        # Both endpoint capsules fit, while the shortest rotational arc sends
        # the shaft through z=10, below reviewed z=24.
        for quaternion in (first,second):
            assert _outside(engine.scene,domain,envelopes(TOOLS['cusa'],engine.parameters(slot),[50,50,50],quaternion)) is None
        move=action('move',[50.,50.,50.],second);before=placed['view']['stateId']
        preview=call('preview',placed,action=move)
        assert preview['status']=='rejected' and preview['preview']
        assert preview['stateId']==before and preview['view']['stateId']==before
        result=call('step',placed,action=move)
        assert result['outcome']['reasonCode']=='REVIEWED_TASK_DOMAIN_OUTSIDE'
        assert result['view']['stateId']==before and result['position']==1 and not result['changed']


def test_implicit_withdrawal_refuses_but_explicit_observed_withdrawal_passes(tmp_path):
    with desktop(tmp_path) as (_,__,call):
        created=call('create');placed=call('step',created,action=action('place',[50.,50.,70.]))
        refused=call('step',placed,action=action('withdraw'))
        assert refused['outcome']['reasonCode']=='REVIEWED_TASK_DOMAIN_OUTSIDE'
        assert refused['view']['stateId']==placed['view']['stateId']
        admitted=call('step',placed,action=action('withdraw',[50.,50.,65.]))
        assert admitted['outcome']['status']=='accepted'


def test_malformed_shape_fails_closed_before_environment_is_persisted(tmp_path):
    malformed=declaration();malformed['shape']=[99,100,100]
    with desktop(tmp_path,malformed) as (session,identity,call):
        with pytest.raises(ValueError,match='REVIEWED_TASK_DOMAIN_INVALID'):
            call('create')
        assert 'surgical_toolbox' not in session.cases[identity].artifacts


def test_absent_guard_keeps_legacy_empty_outside_box_placement_and_blocks_no_commands(tmp_path,monkeypatch):
    import resectionlab.surgical_task_domain as task_domain
    monkeypatch.setattr(task_domain,'action_rejection',lambda *a,**k:pytest.fail('Absent declaration must not invoke action guard'))
    with desktop(tmp_path,False) as (_,__,call):
        created=call('create');result=call('step',created,action=action('place',[2.,50.,70.]))
        assert result['outcome']['status']=='accepted' and result['position']==1


def test_guarded_advanced_mutation_refuses_before_autonomous_recovery(tmp_path):
    with desktop(tmp_path) as (_,__,call):
        created=call('create')
        with pytest.raises(BridgeError) as error:
            call('recoverExit',created)
        assert error.value.code=='REVIEWED_TASK_DOMAIN_COMMAND_UNSUPPORTED'
        observed=call('inspect')
        assert observed['view']['stateId']==created['view']['stateId'] and observed['position']==0


def test_physical_capsule_radius_respects_oblique_anisotropic_voxel_faces():
    from resectionlab.surgical_scene import SurgicalScene
    shape=(10,11,12);labels=np.zeros(shape,np.uint16);labels[5,5,5]=1
    affine=np.eye(4);affine[:3,:3]=Rotation.from_euler('z',31,degrees=True).as_matrix()@np.diag([2.,1.,3.]);affine[:3,3]=[100,-50,20]
    raw=declaration(shape);raw['minimumVoxelBoundary']=[-.5]*3;raw['maximumVoxelBoundary']=(np.array(shape)-.5).tolist()
    scene=SurgicalScene(labels,affine,[{'id':1,'name':'source','material':'soft_tissue'}],{'adapter':{'reviewedTaskDomainV1':raw}}, {})
    point=(affine@np.array([0,5,6,1]))[:3];domain=reviewed_domain(scene)
    assert _outside(scene,domain,[(point,point,.8,'working')]) is None
    assert _outside(scene,domain,[(point,point,1.1,'working')]) is not None
    assert scene.labels.sum()==1 and len(scene.regions)==1


def test_active_other_hand_is_checked_even_when_selected_action_is_interior(tmp_path):
    from resectionlab.surgical_task_domain import action_rejection
    with desktop(tmp_path) as (session,identity,call):
        created=call('create');placed=call('step',created,action=action('place',[50.,50.,70.]))
        engine=session.surgical_sessions[identity][1].clone()
        engine.state['slots']['left'].update(position=[2.,50.,70.],active=True)
        before=engine.state_id
        refusal=action_rejection(engine,action('inspect'))
        assert refusal['reasonCode']=='REVIEWED_TASK_DOMAIN_OUTSIDE'
        assert engine.state_id==before


def test_reviewed_box_survives_native_save_reopen_and_public_refusal(tmp_path):
    saved=tmp_path/'saved-guarded.ressectionlab'
    with desktop(tmp_path/'first') as (session,identity,call):
        created=call('create');placed=call('step',created,action=action('place',[50.,50.,70.]))
        session.execute('saveCase',{'caseHash':identity,'path':str(saved),'workspace':{},'overwrite':False},_Request('save'),lambda *_:None)
        before=placed['view']['stateId']
    reopened=BridgeSession(tmp_path/'reopened-transfers')
    try:
        loaded=reopened.execute('loadCase',{'path':str(saved)},_Request('load'),lambda *_:None)
        identity=loaded['caseHash']
        observed=reopened.execute('surgicalRehearsal',{'caseHash':identity,'command':'inspect'},_Request('inspect'),lambda *_:None)
        assert observed['view']['stateId']==before and observed['position']==1
        result=reopened.execute('surgicalRehearsal',{'caseHash':identity,'command':'step','expectedStateId':before,
            'action':action('withdraw')},_Request('step'),lambda *_:None)
        assert result['outcome']['reasonCode']=='REVIEWED_TASK_DOMAIN_OUTSIDE'
        assert result['view']['stateId']==before and not result['changed']
    finally:
        reopened.close()


def test_absent_declaration_helper_never_reads_engine_state_or_validation():
    from types import SimpleNamespace
    from resectionlab.surgical_task_domain import action_rejection
    class NoEngineAccess:
        scene=SimpleNamespace(source={'adapter':{} })
        def __getattr__(self,name):
            pytest.fail('Absent declaration read engine '+name)
    assert action_rejection(NoEngineAccess(),{'operation':'move'}) is None
