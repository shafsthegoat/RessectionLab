"""Opt-in, independent audit of locally acquired POC artifacts; never inference.

Run from the repository root with the project Python and NODE_EXECUTABLE set.
The normal test suite does not discover this file. All saved copies are temporary.
"""
from pathlib import Path
from zipfile import ZipFile
from urllib.parse import urlparse, unquote
from unittest.mock import patch
import copy
import hashlib
import json
import os
import subprocess
import tempfile
import time

import nibabel as nib
import numpy as np
from resectionlab.desktop_bridge import BridgeRuntime
from resectionlab.workspace_bundle import load_workspace

ROOT=Path(__file__).resolve().parents[1]
MODEL_RECORDS=[
    'UCSF-PDGM-0004/synthstrip-main/fe693f2aed5cb0bb90aa',
    'UCSF-PDGM-0004/monai-brats-0.5.4/af24e4d8f13e49e37ff8',
    'SynthRAD2023-1BA336/totalsegmentator-skull/32e4c06b003e5b904387',
    'Lausanne-sub-000-ses-20110101/vesselfm-base/0cf5010a6c39c81d7f4b',
]

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

class Client:
    def __init__(self,root):
        self.root=root;self.events=[];self.serial=0
        self.runtime=BridgeRuntime(root,self.events.append)
    def call(self,op,args,error=False):
        self.serial+=1;identity=str(self.serial)
        self.runtime.submit({'id':identity,'op':op,'args':args})
        assert self.runtime.wait_idle(45),f'{op} timed out'
        response=[e for e in self.events if e['id']==identity][-1]
        assert response['event']==('error' if error else 'result'),response
        return response if error else response['result']
    def close(self):self.runtime.close()

def original_rows(path):
    result=[]
    with ZipFile(path) as archive:
        workspace=json.loads(archive.read('workspace.json'))
        for row in workspace['originalFiles']:
            data=archive.read(row['member'])
            assert len(data)==row['bytes']
            assert hashlib.sha256(data).hexdigest()==row['sha256']
            local=Path(unquote(urlparse(row['sourceUri']).path))
            assert local.is_file(),local
            assert local.read_bytes()==data,local
            result.append({k:row[k] for k in ('name','sha256','bytes','role','sourceUri')})
    return result

def model_checks():
    rows=[];watched={}
    for relative in MODEL_RECORDS:
        path=ROOT/'outputs/poc-models'/relative/'inference.json'
        record=json.loads(path.read_text());assert record['status']=='succeeded' and record['exit_code']==0
        inputs=record.get('inputs') or [record['input']]
        for item in inputs+record['weights']+record['outputs']:
            actual=Path(item['path']);assert digest(actual)==item['sha256'],actual
            watched[str(actual)]=(actual.stat().st_mtime_ns,digest(actual))
        watched[str(path)]=(path.stat().st_mtime_ns,digest(path))
        reference=nib.load(inputs[0]['path'])
        for item in inputs:
            image=nib.load(item['path'])
            assert image.shape==reference.shape and np.array_equal(image.affine,reference.affine)
            assert list(image.shape)==item['shape'] and np.array_equal(image.affine,item['affine_ras_mm'])
        outputs=[]
        for item in record['outputs']:
            image=nib.load(item['path']);array=np.asarray(image.dataobj)
            assert image.shape==reference.shape and np.array_equal(image.affine,reference.affine)
            assert int(np.count_nonzero(array))==item['foreground_voxels']>0
            assert set(np.unique(array))<=({0,1,2,4} if 'labels' in item['name'] else {0,1})
            outputs.append({'name':item['name'],'foreground_voxels':int(np.count_nonzero(array)),
                            'shape':list(image.shape),'exact_source_affine':True})
        rows.append({'model':record['model_id'],'case':record['case'],'all_hashes_match':True,
                     'weights_bytes':sum(w['bytes'] for w in record['weights']),'outputs':outputs})
    return rows,watched

def archived_voxel_checks(path):
    case,_,saved=load_workspace(path)
    sources=[case]+[source.case for source in saved['sources']]
    for source in sources:
        reference=next(ref for ref in source.source_refs if ref.source_id=='structural')
        image=nib.load(Path(unquote(urlparse(reference.uri).path)))
        assert np.array_equal(source.mri,np.asarray(image.dataobj,dtype=np.float32))
        assert np.array_equal(source.affine,image.affine)
    layers=[]
    for name,metadata in case.metadata['anatomy_layers'].items():
        source=Path(unquote(urlparse(metadata['source_uri']).path))
        assert digest(source)==metadata['source_sha256']
        image=nib.load(source);array=np.asarray(image.dataobj)
        mask=case.brain_mask if name=='brain' and case.brain_mask is not None else case.compartments[name]
        expected=(array==metadata['label_value']) if metadata['source_role']=='imported_source_annotation' else array.astype(bool)
        assert np.array_equal(mask,expected),name
        assert np.array_equal(image.affine,case.affine),name
        if metadata['source_role']=='cached_model_inference':
            record_path=source.parent/'inference.json';record=json.loads(record_path.read_text())
            assert digest(record_path)==metadata['inference_manifest_sha256']
            assert record['cache_key']==metadata['cache_key'] and record['case']==case.case_id
            assert metadata['provenance']=='estimated'
        else:
            assert metadata['model'] is None and metadata['provenance']=='source-provided'
        layers.append({'name':name,'provenance':metadata['provenance'],'stored_voxels_equal_source_mask':True})
    return {'all_source_volume_values_exact':True,'layers':layers}

def hydrate(client,payload,directory):
    labels=[x['name'] for x in payload['compartments']]
    if payload['brainMask'] is not None and 'brain' in payload['metadata'].get('anatomy_layers',{}) and 'brain' not in labels:labels.append('brain')
    fixture=directory/'hydration.json'
    fixture.write_text(json.dumps({'payload':payload,'transferRoot':str(client.root),'expectedLayers':labels}))
    completed=subprocess.run([os.environ['NODE_EXECUTABLE'],'--experimental-strip-types',
        str(ROOT/'desktop/tests/poc-acceptance-hydration.mjs'),str(fixture)],capture_output=True,text=True,timeout=40)
    assert completed.returncode==0,completed.stderr
    return json.loads(completed.stdout)

def case_checks(path,directory):
    before_originals=original_rows(path)
    voxel_checks=archived_voxel_checks(path)
    first=Client(directory/'first');second=None
    try:
        with patch('subprocess.Popen',side_effect=AssertionError('Loading/saving must not start inference')):
            primary=first.call('loadCase',{'path':str(path)})
        initial=hydrate(first,primary,directory)
        state=copy.deepcopy(primary['workspaceSession']['imagingState'])
        series=primary['workspaceSession']['displaySeries']
        for index,(key,value) in enumerate(state['states'].items()):
            cursor=value['cursor'];labels=list(value['visibleLayers'])
            value['visibleLayers']={name:i%2==1 for i,name in enumerate(labels)}
            value['viewer']={'layerOpacity':{name:.22+.06*i for i,name in enumerate(labels)},
                'layerColor':{name:'#178bca' for name in labels},
                'camera':{'position':[cursor[0]+200,cursor[1]-250,cursor[2]+180],'target':cursor,'up':[0,0,1]},
                'clipping':{'axis':'z','value':cursor[2],'enabled':True,'flipped':True},
                'layout':'3d-focus' if index%2==0 else 'mri-review','activePlane':'sagittal',
                'showPlane':False,'contrast':1.25,'fusion':{'seriesId':None,'opacity':.37}}
            value['cursor']=[cursor[0]+.25,cursor[1]-.25,cursor[2]+.25]
        if series:
            state['states']['primary']['viewer']['fusion']['seriesId']=series[-1]['seriesId']
            state['states'][series[0]['seriesId']]['viewer']['fusion']['seriesId']='primary'
            state['selectedSeriesId']=series[0]['seriesId']
        saved=directory/'saved.ressectionlab'
        with patch('subprocess.Popen',side_effect=AssertionError('Loading/saving must not start inference')):
            first.call('saveCase',{'caseHash':primary['caseHash'],'path':str(saved),'workspace':{'imaging':state}})
            first.close();first=None
            second=Client(directory/'second')
            reopened=second.call('loadCase',{'path':str(saved)})
        assert reopened['caseHash']==primary['caseHash']
        assert reopened['workspaceSession']['imagingState']==state
        assert reopened['mri']['sha256']==primary['mri']['sha256']
        assert reopened['workspaceSession']['originalFiles']==primary['workspaceSession']['originalFiles']
        assert original_rows(saved)==before_originals
        final=hydrate(second,reopened,directory)
        # Test refusal through the same sidecar save operation, not validator inspection alone.
        invalid=copy.deepcopy(state)
        invalid['states']['primary']['viewer']['fusion']['seriesId']='sha256:'+'f'*64
        rejected=second.call('saveCase',{'caseHash':reopened['caseHash'],'path':str(directory/'invalid.ressectionlab'),
            'workspace':{'imaging':invalid}},error=True)
        assert not (directory/'invalid.ressectionlab').exists()
        unreviewed_fusion=None
        if series:
            ct=ROOT/'data/qc/synthrad-task1-first-pair-v1/restricted/ct.nii.gz'
            unreviewed=second.call('importDisplaySeries',{'caseHash':reopened['caseHash'],
                'imagePath':str(ct),'modality':'CT','annotationKind':'none'})
            assert unreviewed['registration']['status']=='unreviewed'
            assert not unreviewed['registration']['overlayPermitted']
            assert not unreviewed['association']['samePersonVerified']
            invalid=copy.deepcopy(second.runtime.session.cases[reopened['caseHash']].imaging_state)
            invalid['states']['primary']['viewer']['fusion']['seriesId']=unreviewed['seriesId']
            unreviewed_fusion=second.call('saveCase',{'caseHash':reopened['caseHash'],
                'path':str(directory/'unregistered.ressectionlab'),'workspace':{'imaging':invalid}},error=True)
            assert not (directory/'unregistered.ressectionlab').exists()
        return {'case':primary['caseId'],'original_files':len(before_originals),
            'original_bytes':sum(x['bytes'] for x in before_originals),'original_bytes_exact':True,
            'case_identity_unchanged':True,'save_close_reopen_settings_exact':True,
            'process_spawn_during_save_reopen':False,'invalid_fusion_refused':rejected['error']['message'],
            'unregistered_real_CT_fusion_refused':None if unreviewed_fusion is None else unreviewed_fusion['error']['message'],
            'archived_voxels':voxel_checks,
            'initial_hydration':initial,'reopened_hydration':final}
    finally:
        if first:first.close()
        if second:second.close()

def import_regression(directory):
    path=directory/'generated-import-control.nii.gz'
    affine=np.diag([1.,2.,3.,1.]);image=nib.Nifti1Image(np.arange(120,dtype=np.float32).reshape(4,5,6),affine)
    image.header.set_xyzt_units('mm');image.set_qform(affine,1);image.set_sform(affine,1);nib.save(image,path)
    original=path.read_bytes();client=Client(directory/'import-first')
    try:
        result=client.call('importNifti',{'structuralPath':str(path),'caseId':'generated-import-control'})
        saved=directory/'imported.ressectionlab'
        client.call('saveCase',{'caseHash':result['caseHash'],'path':str(saved)})
    finally:client.close()
    path.unlink();client=Client(directory/'import-second')
    try:
        loaded=client.call('loadCase',{'path':str(saved)})
        assert loaded['caseHash']==result['caseHash']
        with ZipFile(saved) as archive:assert archive.read('workspace/original-0.bin')==original
    finally:client.close()
    return {'fresh_nifti_import':True,'saved_originals_survive_removed_source':True,'fixture':'generated, not patient evidence'}

def main():
    started=time.monotonic();report={'scope':'Backend + real host asset boundary + frontend hydration; no GUI or clinical accuracy claim'}
    report['models'],watched=model_checks();report['cases']=[]
    with tempfile.TemporaryDirectory(prefix='ressectionlab-acceptance-') as temporary:
        report['import_regression']=import_regression(Path(temporary))
        # Review-session copies can share this folder; audit the three canonical inputs.
        for name in ('Lausanne-sub-000-ses-20110101', 'SynthRAD2023-1BA336', 'UCSF-PDGM-0004'):
            path=ROOT/'outputs/poc-cases'/f'{name}.ressectionlab'
            folder=Path(temporary)/path.stem;folder.mkdir()
            report['cases'].append(case_checks(path,folder))
            print('PASS '+path.stem,flush=True)
    for path,(mtime,checksum) in watched.items():
        assert Path(path).stat().st_mtime_ns==mtime and digest(path)==checksum,path
    report['model_artifacts_unchanged_after_reopen']=True
    report['elapsed_seconds']=time.monotonic()-started
    destination=ROOT/'outputs/poc-acceptance/independent-backend-hydration.json'
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(report,indent=2)+'\n')
    print(destination)

if __name__=='__main__':main()
