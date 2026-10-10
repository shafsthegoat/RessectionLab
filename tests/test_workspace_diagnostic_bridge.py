"""Small generated BridgeRuntime -> existing auxiliary source -> diagnostic path."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import nibabel as nib
import pytest
from resectionlab import desktop_bridge as bridge
from resectionlab.desktop_bridge import BridgeRuntime
from resectionlab import workspace_diagnostic as module
from test_workspace_diagnostic import fixture


def assert_transfers_live(result):
    paths = set()
    for name in ('state', 'coverage'):
        descriptor = result[name]
        path = Path(descriptor['path'])
        raw = path.read_bytes()
        assert len(raw) == descriptor['byteLength']
        assert hashlib.sha256(raw).hexdigest() == descriptor['sha256']
        paths.add(path)
    return paths


def test_generated_bridge_binds_auxiliary_without_mutating_planning_case(tmp_path,monkeypatch):
    events=[];runtime=BridgeRuntime(tmp_path/'transfers',events.append)
    def call(op,args):
        identity=str(len(events));runtime.submit({'id':identity,'op':op,'args':args});assert runtime.wait_idle(5)
        return [e for e in events if e['id']==identity][-1]
    try:
        image=nib.Nifti1Image(np.arange(8,dtype=np.float32).reshape(2,2,2),np.eye(4));image.header.set_xyzt_units('mm')
        source_path=tmp_path/'selected.nii.gz';nib.save(image,source_path)
        primary_event=call('importNifti',{'structuralPath':str(source_path)})
        assert primary_event['event']=='result',primary_event
        primary=primary_event['result']
        selected=call('importDisplaySeries',{'caseHash':primary['caseHash'],'imagePath':str(source_path),'modality':'T1CE','annotationKind':'none'})['result']
        _,descriptor_path,d=fixture(tmp_path);d['source_sha256']['t1c']=hashlib.sha256(source_path.read_bytes()).hexdigest();descriptor_path.write_text(json.dumps(d))
        args={'caseHash':primary['caseHash'],'seriesId':selected['seriesId'],'descriptorPath':str(descriptor_path)}
        parent=runtime.session.cases[primary['caseHash']];before=parent.case
        monkeypatch.setattr(module,'CASE4_DESCRIPTOR_SHA256',None)
        assert call('importDiagnosticLayer',args)['error']['code']=='DIAGNOSTIC_NOT_PUBLISHED'
        monkeypatch.setattr(module,'CASE4_DESCRIPTOR_SHA256',hashlib.sha256(descriptor_path.read_bytes()).hexdigest())
        result=call('importDiagnosticLayer',args);assert result['event']=='result',result
        value=result['result'];assert value['state']['dtype']=='int8' and value['coverage']['dtype']=='uint8'
        assert value['caseHash']==primary['caseHash'] and value['seriesId']==selected['seriesId']
        assert parent.case is before and parent.case.semantic_hash==primary['caseHash'] and len(runtime.session.cases)==1
        assert selected['volume']['caseHash'] not in runtime.session.cases or selected['volume']['caseHash']==primary['caseHash']
        # call() waits for _work's finally/prune, not merely the result event.
        paths=assert_transfers_live(value)
        assert set(map(Path,parent.diagnostic_transfer_paths))==paths
        assert call('inspectEvidence',{'caseHash':primary['caseHash']})['event']=='result'
        assert assert_transfers_live(value)==paths
    finally:runtime.close()


@pytest.fixture
def diagnostic_runtime(tmp_path, monkeypatch):
    events=[]
    runtime=BridgeRuntime(tmp_path/'transfers',events.append,max_cases=1)
    sequence=0
    def call(op,args):
        nonlocal sequence
        sequence+=1
        identity=f'request-{sequence}'
        runtime.submit({'id':identity,'op':op,'args':args})
        assert runtime.wait_idle(5)
        return [event for event in events if event['id']==identity][-1]
    try:
        image=nib.Nifti1Image(np.arange(8,dtype=np.float32).reshape(2,2,2),np.eye(4))
        image.header.set_xyzt_units('mm')
        source_path=tmp_path/'selected.nii.gz'
        nib.save(image,source_path)
        primary=call('importNifti',{'structuralPath':str(source_path)})
        assert primary['event']=='result',primary
        primary=primary['result']
        selected=call('importDisplaySeries',{'caseHash':primary['caseHash'],
            'imagePath':str(source_path),'modality':'T1CE','annotationKind':'none'})
        assert selected['event']=='result',selected
        selected=selected['result']
        def prepare(state=None):
            _,descriptor_path,descriptor=fixture(tmp_path)
            descriptor['source_sha256']['t1c']=hashlib.sha256(source_path.read_bytes()).hexdigest()
            if state is not None:
                state=np.asarray(state,dtype=np.int8).reshape(2,2,2)
                coverage=(state!=-1).astype(np.uint8)
                for name,array in (('state',state),('coverage',coverage)):
                    image=nib.Nifti1Image(array,np.eye(4))
                    image.header.set_xyzt_units('mm');image.set_sform(np.eye(4),code=2);image.set_qform(None,code=0)
                    path=tmp_path/(name+'.nii.gz');nib.save(image,path)
                    descriptor['outputs'][name]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                descriptor['predicted_voxels']=int(coverage.sum())
                descriptor['unknown_voxels']=int((state==-1).sum())
            descriptor_path.write_text(json.dumps(descriptor))
            monkeypatch.setattr(module,'CASE4_DESCRIPTOR_SHA256',hashlib.sha256(descriptor_path.read_bytes()).hexdigest())
            return {'caseHash':primary['caseHash'],'seriesId':selected['seriesId'],
                    'descriptorPath':str(descriptor_path)}
        yield SimpleNamespace(runtime=runtime,call=call,prepare=prepare,primary=primary,
                              source_path=source_path,events=events)
    finally:
        runtime.close()


def test_latest_diagnostic_replacement_case_replacement_and_eviction_cleanup(diagnostic_runtime,tmp_path):
    h=diagnostic_runtime
    first=h.call('importDiagnosticLayer',h.prepare());assert first['event']=='result',first
    first_paths=assert_transfers_live(first['result'])
    args=h.prepare([-1,-1,-1,1,0,0,-1,-1])
    second=h.call('importDiagnosticLayer',args);assert second['event']=='result',second
    second_paths=assert_transfers_live(second['result'])
    assert first_paths.isdisjoint(second_paths)
    assert all(not path.exists() for path in first_paths)
    entry=h.runtime.session.cases[h.primary['caseHash']]
    assert set(map(Path,entry.diagnostic_transfer_paths))==second_paths
    # Live paths do not enter the workspace descriptor or primary CaseData.
    serialized=json.dumps(h.runtime.session._workspace_payload(entry))
    assert 'diagnostic_transfer_paths' not in serialized
    assert all(str(path) not in serialized for path in second_paths)
    replaced=h.call('importNifti',{'structuralPath':str(h.source_path)})
    assert replaced['event']=='result',replaced
    assert replaced['result']['caseHash']==h.primary['caseHash']
    assert h.runtime.session.cases[h.primary['caseHash']].diagnostic_transfer_paths==()
    assert all(not path.exists() for path in second_paths)
    restored=h.call('importDiagnosticLayer',args);assert restored['event']=='result',restored
    restored_paths=assert_transfers_live(restored['result'])
    other=nib.Nifti1Image((np.arange(8,dtype=np.float32)+20).reshape(2,2,2),np.eye(4))
    other.header.set_xyzt_units('mm');other_path=tmp_path/'other.nii.gz';nib.save(other,other_path)
    evicted=h.call('importNifti',{'structuralPath':str(other_path)})
    assert evicted['event']=='result',evicted
    assert h.primary['caseHash'] not in h.runtime.session.cases
    assert all(not path.exists() for path in restored_paths)


def test_cancel_before_diagnostic_commit_preserves_previous_files_and_prunes_new(diagnostic_runtime,monkeypatch):
    h=diagnostic_runtime
    first=h.call('importDiagnosticLayer',h.prepare());assert first['event']=='result',first
    old_paths=assert_transfers_live(first['result'])
    entry=h.runtime.session.cases[h.primary['caseHash']]
    old_retained=entry.diagnostic_transfer_paths
    args=h.prepare([-1,-1,-1,1,0,0,-1,-1])
    begin_commit=bridge._Request.begin_commit
    staged=[]
    def cancel_at_commit(request):
        staged.extend(set(h.runtime.session.transfers.root.glob('*.bin'))-
                      before_files)
        assert len(staged)==2  # Both validated snapshots exist before commit.
        h.runtime.submit({'id':'cancel-diagnostic','op':'cancel',
                          'args':{'requestId':request.request_id}})
        begin_commit(request)
    before_files=set(h.runtime.session.transfers.root.glob('*.bin'))
    monkeypatch.setattr(bridge._Request,'begin_commit',cancel_at_commit)
    cancelled=h.call('importDiagnosticLayer',args)
    assert cancelled['event']=='cancelled',cancelled
    assert [e for e in h.events if e['id']=='cancel-diagnostic'][-1]['result']['cancelled'] is True
    assert entry.diagnostic_transfer_paths==old_retained
    assert assert_transfers_live(first['result'])==old_paths
    assert all(not path.exists() for path in staged)
    assert set(h.runtime.session.transfers.root.glob('*.bin'))==before_files
