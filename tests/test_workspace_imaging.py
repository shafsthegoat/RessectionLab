"""Generated NIfTI -> actual proposed BridgeRuntime -> shared descriptor tests."""
from pathlib import Path
import importlib
import sys
import json
import os
import shutil
import subprocess
import numpy as np
import nibabel as nib
import pytest
ROOT=Path(__file__).resolve().parents[1]
from resectionlab.desktop_bridge import BridgeRuntime
from resectionlab.structural_evidence import structural_frame_hash

@pytest.fixture
def bridge(tmp_path):
    events=[];runtime=BridgeRuntime(tmp_path/'transfers',events.append)
    def call(op,args=None):
        identity=str(len(events));runtime.submit({'id':identity,'op':op,'args':args or {}});assert runtime.wait_idle(10)
        return [e for e in events if e['id']==identity][-1]
    yield runtime,call
    runtime.close()

def volume(path,*,affine=None,labels=False,shape=(4,5,6)):
    values=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)
    if labels:values=(values%2).astype(np.uint8)
    matrix=np.diag([1.,2.,3.,1.]) if affine is None else affine
    image=nib.Nifti1Image(values,matrix);image.header.set_xyzt_units('mm');image.set_qform(matrix,1);image.set_sform(matrix,1);nib.save(image,path);return path

def setup(bridge,tmp_path):
    runtime,call=bridge
    primary=call('importNifti',{'structuralPath':str(volume(tmp_path/'primary.nii.gz'))})['result']
    image=volume(tmp_path/'aux.nii.gz',affine=np.diag([2.,3.,4.,1.]))
    return runtime,call,primary,image

def test_real_import_attachment_does_not_change_actor_case_or_frames(bridge,tmp_path):
    runtime,call,primary,image=setup(bridge,tmp_path)
    before=runtime.session.cases[primary['caseHash']].case
    result=call('importDisplaySeries',{'caseHash':primary['caseHash'],'imagePath':str(image),'modality':'MRA','annotationKind':'none'})
    assert result['event']=='result',result
    attached=result['result'];parent=runtime.session.cases[primary['caseHash']]
    assert parent.case is before and parent.case.semantic_hash==primary['caseHash'] and parent.case.planning_hash==primary['planningHash']
    assert attached['referenceFrameHash']==structural_frame_hash(parent.case)
    assert attached['sourceFrameHash']!=attached['referenceFrameHash']
    assert not attached['registration']['overlayPermitted'] and not attached['planningEligible']
    assert not attached['association']['samePersonVerified']
    assert attached['volume']['caseHash'] not in runtime.session.cases
    assert Path(attached['volume']['mri']['path']).exists()
    denied=call('inspectEvidence',{'caseHash':attached['volume']['caseHash']})
    assert denied['error']['code']=='CASE_VERSION_UNAVAILABLE'
    assert len(runtime.session.cases)==1

def test_aligned_annotations_keep_estimate_identity_and_never_target_parent(bridge,tmp_path):
    runtime,call,primary,image=setup(bridge,tmp_path);mask=volume(tmp_path/'labels.nii.gz',affine=np.diag([2.,3.,4.,1.]),labels=True)
    result=call('importDisplaySeries',{'caseHash':primary['caseHash'],'imagePath':str(image),'annotationPath':str(mask),'modality':'TOF-MRA','annotationKind':'estimated'})
    assert result['event']=='result',result
    attached=result['result'];assert attached['annotationKind']=='estimated'
    assert attached['volume']['compartments'][0]['name']=='source_label_1'
    assert next(s for s in attached['volume']['sourceRefs'] if s['source_id']=='supplied_target_annotation')['provenance']=='estimated'
    assert runtime.session.cases[primary['caseHash']].case.compartments=={}
    assert attached['volume']['metadata']['model_identity'] is None

def test_mismatched_mask_refuses_without_attachment_or_actor_mutation(bridge,tmp_path):
    runtime,call,primary,image=setup(bridge,tmp_path);mask=volume(tmp_path/'labels.nii.gz',labels=True)
    result=call('importDisplaySeries',{'caseHash':primary['caseHash'],'imagePath':str(image),'annotationPath':str(mask),'modality':'MRA','annotationKind':'source-provided'})
    assert result['event']=='error';assert not runtime.session.cases[primary['caseHash']].display_series

def test_stale_case_bad_modality_and_missing_annotation_role_refuse(bridge,tmp_path):
    runtime,call,primary,image=setup(bridge,tmp_path)
    for updates in [{'caseHash':'sha256:'+'0'*64},{'modality':'DWI-4D'},{'annotationKind':'estimated'}]:
        args={'caseHash':primary['caseHash'],'imagePath':str(image),'modality':'T2','annotationKind':'none',**updates}
        assert call('importDisplaySeries',args)['event']=='error'
    assert not runtime.session.cases[primary['caseHash']].display_series

def test_four_dimensional_image_and_unknown_units_stay_unavailable(bridge,tmp_path):
    runtime,call,primary,image=setup(bridge,tmp_path)
    for n,shape,units in [('dwi',(2,2,2,3),'mm'),('units',(2,2,2),'unknown')]:
        p=tmp_path/(n+'.nii.gz');ni=nib.Nifti1Image(np.arange(np.prod(shape),dtype=np.float32).reshape(shape),np.eye(4));ni.header.set_xyzt_units(units);ni.set_qform(np.eye(4),1);ni.set_sform(np.eye(4),1);nib.save(ni,p)
        assert call('importDisplaySeries',{'caseHash':primary['caseHash'],'imagePath':str(p),'modality':'other-3D-scalar','annotationKind':'none'})['event']=='error'

def test_generated_bridge_host_asset_and_viewer_path(bridge,tmp_path):
    node=os.environ.get('NODE_EXECUTABLE') or shutil.which('node')
    if node is None:pytest.skip('Set NODE_EXECUTABLE to exercise the cross-language desktop boundary.')
    runtime,call,primary,_=setup(bridge,tmp_path)
    affine=np.diag([2.,3.,4.,1.]);affine[0,3]=100.
    image=volume(tmp_path/'shifted.nii.gz',affine=affine)
    mask=volume(tmp_path/'shifted-labels.nii.gz',affine=affine,labels=True)
    result=call('importDisplaySeries',{'caseHash':primary['caseHash'],'imagePath':str(image),
        'annotationPath':str(mask),'modality':'MRA','annotationKind':'source-provided'})
    assert result['event']=='result',result
    payload=tmp_path/'generated.json'
    payload.write_text(json.dumps({'transferRoot':str(tmp_path/'transfers'), 'primary':primary,
        'attached':result['result'],'expectedImage':np.arange(120,dtype=np.float32).tolist()}))
    completed=subprocess.run([node,'--experimental-strip-types',str(ROOT/'desktop/tests/workspace-imaging-crossstack.mjs'),str(payload)],
        capture_output=True,text=True,timeout=15)
    assert completed.returncode==0,completed.stderr
    assert 'native-frame selection passed' in completed.stdout
