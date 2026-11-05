import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('integrated_rebind',Path(__file__).parents[1]/'scripts/data_integrated_rebind.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_identity_projection_preserves_all_physical_fields_and_cell_ids():
    original={'sceneId':'old-scene','stateId':'old-state','actions':[{'operation':'apply'}],
              'checkpoint':{'stateId':'old-state','state':{'elapsedSeconds':17.,'thermal':{'42':99.},'removed':[7,42],'sourceCellId':42}}}
    other=copy.deepcopy(original);other['sceneId']='new-scene';other['stateId']='new-state';other['checkpoint']['stateId']='new-state'
    assert module.physical_snapshot(original)==module.physical_snapshot(other)
    for key,value in [('elapsedSeconds',0.),('thermal',{}),('removed',[]),('sourceCellId',7)]:
        changed=copy.deepcopy(other);changed['checkpoint']['state'][key]=value
        assert module.physical_snapshot(original)!=module.physical_snapshot(changed)
    outcome={'action':{'slot':'right','operation':'apply','duration':1.},'stateBefore':'a','stateAfter':'b','deltas':{'geometryRevision':'a','removedCells':[42]},'events':[{'cellId':42}]}
    changed=copy.deepcopy(outcome);changed['stateBefore']='x';changed['stateAfter']='y';changed['deltas']['geometryRevision']='x'
    assert module.physical_outcome(outcome)==module.physical_outcome(changed)
    changed['events'][0]['cellId']=7
    assert module.physical_outcome(outcome)!=module.physical_outcome(changed)
    changed=copy.deepcopy(outcome);changed['action']['duration']=2.
    assert module.physical_outcome(outcome)!=module.physical_outcome(changed)


def test_numeric_identity_refuses_a_single_changed_annotation_or_affine():
    first=SimpleNamespace(mri=np.arange(27,dtype=np.float32).reshape(3,3,3),affine=np.eye(4),
        compartments={'vessel':np.zeros((3,3,3),bool)},source_compartments={'vessel':np.zeros((3,3,3),bool)},
        brain_mask=None,metadata={})
    other=copy.deepcopy(first);module.assert_numeric_identity(first,other)
    other.compartments['vessel'][1,1,1]=True
    with pytest.raises(ValueError,match='annotation changed'):module.assert_numeric_identity(first,other)
    other=copy.deepcopy(first);other.affine[0,3]=.01
    with pytest.raises(ValueError,match='physical grid changed'):module.assert_numeric_identity(first,other)


def test_public_metadata_only_rebind_preserves_clock_prefix_and_cold_replay(tmp_path,monkeypatch):
    from resectionlab.core import CaseData,SourceRef
    from resectionlab.workspace_bundle import save_workspace,view_state,load_workspace
    from resectionlab.desktop_bridge import BridgeSession,_Request
    shape=(100,100,100);mask=np.zeros(shape,bool);mask[95,95,95]=True
    originals=[]
    metadata={'is_synthetic':True,'data_case':{'role':'DEVELOPMENT','family_id':'generated:1'},'anatomy_layers':{'target':{'kind':'tumor'}}}
    first=CaseData(case_id='old',mri=np.zeros(shape,np.float32),compartments={'target':mask},affine=np.eye(4),
        source_refs=(SourceRef('generated','synthetic://rebind-test',provenance='simulated'),),metadata=metadata)
    old_source=tmp_path/'old.ressectionlab';save_workspace(first,old_source,sources=[],imaging=view_state(first,[]),episode=None,selection=None,artifacts={},originals=originals)
    session=BridgeSession(tmp_path/'old-transfers')
    try:
        def call(op,args):return session.execute(op,args,_Request(op),lambda *_:None)
        identity=call('loadCase',{'path':str(old_source)})['caseHash'];current=call('surgicalRehearsal',{'caseHash':identity,'command':'create'})
        for action in [{'slot':'right','operation':'configure','mechanicsRevision':2,'clockRevision':1,'duration':.1},
                       {'slot':'right','operation':'place','pose':{'position':[50.,50.,70.],'quaternion':[0,0,0,1]},'duration':.1},
                       {'slot':'right','operation':'withdraw','path':[{'position':[50.,50.,65.],'quaternion':[0,0,0,1]}],'duration':.2}]:
            current=call('surgicalRehearsal',{'caseHash':identity,'command':'step','expectedStateId':current['view']['stateId'],'action':action})
            assert current['outcome']['status']=='accepted'
        history=tmp_path/'history.ressectionlab';call('saveCase',{'caseHash':identity,'path':str(history)})
    finally:session.close()
    new_metadata=copy.deepcopy(metadata);new_metadata['surgical_adapter']={'reviewedTaskDomainV1':{
        'mode':'restrict-unknown','shape':list(shape),'minimumVoxelBoundary':[5.,5.,24.],'maximumVoxelBoundary':[94.,94.,95.],
        'reviewEvidenceSha256':'a'*64,'maskSha256':'b'*64}}
    from dataclasses import replace
    new=replace(first,case_id='new',metadata=new_metadata);source=tmp_path/'guarded.ressectionlab'
    save_workspace(new,source,sources=[],imaging=view_state(new,[]),episode=None,selection=None,artifacts={},originals=originals)
    monkeypatch.setattr(module,'checked_root',lambda *a,**k:tmp_path)
    report=module.rebind(history,source,root=tmp_path,name='replay')
    assert report['full_prefix_action_count']==3 and report['new_actions_added']==0
    assert report['all_physical_checkpoint_fields_exact'] and report['cold_public_replay_exact']
    assert report['elapsed_physical_seconds']==pytest.approx(.3)
    _,before,_=load_workspace(history);_,after,_=load_workspace(tmp_path/report['saved'])
    old_snapshot=before['surgical_toolbox']['branches']['main'];new_snapshot=after['surgical_toolbox']['branches']['main']
    assert old_snapshot['stateId']!=new_snapshot['stateId']
    assert module.physical_snapshot(old_snapshot)==module.physical_snapshot(new_snapshot)
