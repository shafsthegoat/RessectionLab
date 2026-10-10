"""Generated source/episode fixtures only; exercise real bridge save-close-open."""
from pathlib import Path
from io import BytesIO
import copy
import hashlib
import json
from zipfile import ZipFile,ZIP_STORED
import numpy as np
import nibabel as nib
import pytest
from resectionlab.desktop_bridge import BridgeRuntime
from resectionlab.imaging import load_case,load_nifti_case,save_case
from resectionlab.core import semantic_digest
from resectionlab.workspace_bundle import canonical_episode,load_workspace,validate_episode,session_digest

class Client:
    def __init__(self,root):
        self.events=[];self.runtime=BridgeRuntime(root,self.events.append);self.serial=0
    def call(self,op,args=None,*,error=False):
        self.serial+=1;identity=str(self.serial)
        self.runtime.submit({'id':identity,'op':op,'args':args or {}})
        assert self.runtime.wait_idle(20)
        result=[e for e in self.events if e['id']==identity][-1]
        assert result['event']==('error' if error else 'result'),result
        return result if error else result['result']
    def close(self):self.runtime.close()

def image(path,labels=False):
    affine=np.diag([2.,3.,4.,1.]);affine[0,3]=100
    values=np.arange(120,dtype=np.float32).reshape(4,5,6)
    if labels:values=(values%2).astype(np.uint8)
    im=nib.Nifti1Image(values,affine);im.header.set_xyzt_units('mm');im.set_qform(affine,1);im.set_sform(affine,1);nib.save(im,path)
    return str(path)

@pytest.fixture
def saved(tmp_path):
    first=Client(tmp_path/'first');episode=first.call('executeDevelopmentEpisode',{'fixture':'generated-sequential-v1','selector':'scripted'})
    primary=episode['case'];source=image(tmp_path/'aux.nii.gz');mask=image(tmp_path/'label.nii.gz',True)
    aux=first.call('importDisplaySeries',{'caseHash':primary['caseHash'],'imagePath':source,'annotationPath':mask,'modality':'MRA','annotationKind':'estimated'})
    state={'selectedSeriesId':aux['seriesId'],'states':{'primary':{'cursor':[6.,6.,4.],'visibleLayers':{'generated_nominal_target':False}},
        aux['seriesId']:{'cursor':[103.,6.,10.],'visibleLayers':{'source_label_1':True}}}}
    selection={'episodeId':episode['episode']['episodeId'],'frameIndex':3,'visible':False}
    path=tmp_path/'workspace.ressectionlab'
    saved=first.call('saveCase',{'caseHash':primary['caseHash'],'path':str(path),'workspace':{'imaging':state,'episodeReplay':selection}})
    first.close();Path(source).unlink();Path(mask).unlink()
    return {'path':path,'primary':primary,'aux':aux,'state':state,'selection':selection,'episode':episode,'saved':saved}

def rewrite(path,change):
    with ZipFile(path) as z:members={i.filename:z.read(i.filename) for i in z.infolist()}
    doc=json.loads(members['workspace.json']);change(doc,members)
    doc['sessionHash']=session_digest(doc)
    members['workspace.json']=json.dumps(doc,sort_keys=True,separators=(',',':')).encode()
    manifest=json.loads(members['manifest.json']);manifest['workspace_sha256']=hashlib.sha256(members['workspace.json']).hexdigest()
    members['manifest.json']=json.dumps(manifest).encode()
    with ZipFile(path,'w',compression=ZIP_STORED) as z:
        for name,data in members.items():z.writestr(name,data)

def test_save_close_reopen_own_sources_and_episode_without_original_paths(saved,tmp_path):
    client=Client(tmp_path/'second')
    try:
        p=client.call('loadCase',{'path':str(saved['path'])});session=p['workspaceSession'];aux=session['displaySeries'][0]
        assert p['caseHash']==saved['primary']['caseHash'] and p['planningHash']==saved['primary']['planningHash']
        assert p['mri']['sha256']==saved['primary']['mri']['sha256'] and aux['volume']['mri']['sha256']==saved['aux']['volume']['mri']['sha256']
        assert aux['seriesId']==saved['aux']['seriesId'] and aux['sourceFrameHash']==saved['aux']['sourceFrameHash']
        assert session['imagingState']==saved['state'] and session['sessionHash']==saved['saved']['sessionHash']
        assert session['episodeReplay']['episodeCanonicalJson']==saved['episode']['episodeCanonicalJson']
        assert session['episodeReplay']['validation']=='authoritative_generated_replay'
        assert session['episodeReplay']['frameIndex']==3 and not session['episodeReplay']['visible']
        assert not aux['planningEligible'] and not aux['association']['samePersonVerified'] and not aux['registration']['overlayPermitted']
        assert aux['volume']['sourceRefs'][1]['provenance']=='estimated'
        assert aux['volume']['caseHash'] not in client.runtime.session.cases
        assert all(not r['planningInput'] for r in session['evidenceInventory'] if r['seriesId']!='primary')
        assert load_case(saved['path']).semantic_hash==p['caseHash']
        second=tmp_path/'second.ressectionlab';client.call('saveCase',{'caseHash':p['caseHash'],'path':str(second)})
        assert load_workspace(second)[2]['sessionHash']==session['sessionHash']
    finally:client.close()

def test_legacy_plain_bundle_remains_useful_and_does_not_restore_artifact_claims(tmp_path):
    source=image(tmp_path/'primary.nii.gz');case=load_nifti_case(source);plain=tmp_path/'plain.ressectionlab'
    save_case(case,plain,artifacts={'workspace':{'case_hash':case.semantic_hash,'selection_replay':{'independentlyAccepted':True}}})
    client=Client(tmp_path/'transfer')
    try:
        payload=client.call('loadCase',{'path':str(plain)})
        assert payload['caseHash']==case.semantic_hash and 'workspaceSession' not in payload
        assert payload['artifacts']['withheldTrainingReason']=='saved_training_requires_current_validation'
    finally:client.close()

@pytest.mark.parametrize('tamper',['frame','scope','selection','motion','audit','array_hash','unlisted'])
def test_resigned_tampering_refuses_atomically(saved,tmp_path,tamper):
    client=Client(tmp_path/'guard');old=client.call('loadCase',{'path':str(saved['path'])})
    before=client.runtime.session.cases[old['caseHash']];paths=set(client.runtime.session.transfers._entries)
    def change(doc,members):
        if tamper=='frame':doc['images'][0]['source']['sourceFrameHash']='sha256:'+'0'*64
        elif tamper=='scope':doc['images'][0]['source']['planningEligible']=True
        elif tamper=='selection':doc['imagingState']['selectedSeriesId']='sha256:'+'f'*64
        elif tamper in ('motion','audit'):
            envelope=doc['episodeReplay']['envelope'];ep=envelope['episode']
            if tamper=='motion':ep['replayFrames'][1]['tipRasMm'][0]+=.25
            else:ep['geometryAudit']['feasible']=False
            canonical=canonical_episode(ep);ep['episodeId']='sha256:'+hashlib.sha256(canonical.encode()).hexdigest()
            envelope['episodeCanonicalJson']=canonical;doc['episodeReplay']['selection']['episodeId']=ep['episodeId']
        elif tamper=='array_hash':doc['images'][0]['sha256']='0'*64
        else:members['../unlisted']=b'forbidden'
    rewrite(saved['path'],change)
    try:
        client.call('loadCase',{'path':str(saved['path'])},error=True)
        assert client.runtime.session.cases[old['caseHash']] is before
        assert set(client.runtime.session.transfers._entries)==paths
        assert all(Path(p['path']).exists() for p in client.runtime.session.transfers._entries.values())
    finally:client.close()

def test_renderer_cannot_supply_episode_or_stale_selection(saved,tmp_path):
    client=Client(tmp_path/'client')
    try:
        p=client.call('loadCase',{'path':str(saved['path'])})
        for workspace in [{'episode':saved['episode']['episode']},{'episodeReplay':{'episodeId':'wrong','frameIndex':0,'visible':True}},
                          {'imaging':{'selectedSeriesId':None,'states':{}}}]:
            destination=tmp_path/'not-written.ressectionlab'
            client.call('saveCase',{'caseHash':p['caseHash'],'path':str(destination),'workspace':workspace},error=True)
            assert not destination.exists()
    finally:client.close()

def test_search_revalidation_preserves_exact_body_and_checks_nontiming_accounting(tmp_path):
    client=Client(tmp_path/'search')
    try:
        result=client.call('executeDevelopmentEpisode',{'fixture':'generated-sequential-v1','selector':'SEARCH'})
        case=client.runtime.session.cases[result['case']['caseHash']].case
        envelope={k:result[k] for k in ('episode','episodeCanonicalJson')}
        assert validate_episode(case,envelope)==envelope
        changed=copy.deepcopy(envelope);changed['episode']['planning']['model_transition_calls']+=1
        canonical=canonical_episode(changed['episode']);changed['episodeCanonicalJson']=canonical
        changed['episode']['episodeId']='sha256:'+hashlib.sha256(canonical.encode()).hexdigest()
        with pytest.raises(ValueError,match='authoritative'):validate_episode(case,changed)
    finally:client.close()

def test_malicious_npy_shape_refuses_before_any_numeric_materialization(saved,monkeypatch):
    def change(doc,members):
        row=doc['images'][0]
        with ZipFile(BytesIO(members[row['member']])) as z:parts={n:z.read(n) for n in z.namelist()}
        with ZipFile(BytesIO(parts['arrays.npz'])) as z:arrays={n:z.read(n) for n in z.namelist()}
        header=BytesIO();np.lib.format.write_array_header_1_0(header,{'descr':'<f4','fortran_order':False,'shape':(10**12,1,1)})
        arrays['mri.npy']=header.getvalue()+b'\x00'*4
        payload=BytesIO()
        with ZipFile(payload,'w') as z:
            for name,data in arrays.items():z.writestr(name,data)
        parts['arrays.npz']=payload.getvalue();manifest=json.loads(parts['manifest.json'])
        manifest['array_sha256']=hashlib.sha256(parts['arrays.npz']).hexdigest();parts['manifest.json']=json.dumps(manifest).encode()
        auxiliary=BytesIO()
        with ZipFile(auxiliary,'w') as z:
            for name,data in parts.items():z.writestr(name,data)
        members[row['member']]=auxiliary.getvalue();row['bytes']=len(auxiliary.getvalue());row['sha256']=hashlib.sha256(auxiliary.getvalue()).hexdigest()
    rewrite(saved['path'],change)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Allocation occurred before malformed shape refusal'))
    with pytest.raises(ValueError,match='shape|expanded'):load_workspace(saved['path'])

def test_duplicate_archive_member_and_missing_bound_extension_refuse(saved):
    with ZipFile(saved['path']) as z:document=z.read('workspace.json')
    with pytest.warns(UserWarning):
        with ZipFile(saved['path'],'a') as z:z.writestr('workspace.json',document)
    with pytest.raises(ValueError,match='Duplicate'):load_workspace(saved['path'])

def test_oblique_view_cursor_aabb_preserved_and_foreign_cursor_refused(tmp_path):
    from resectionlab.workspace_bundle import view_state
    case=load_nifti_case(image(tmp_path/'source.nii.gz'))
    state={'selectedSeriesId':None,'states':{'primary':{'cursor':[99.,-1.5,-2.],'visibleLayers':{}}}}
    assert view_state(case,[],state)==state
    state['states']['primary']['cursor'][0]=-1e12
    with pytest.raises(ValueError,match='outside'):view_state(case,[],state)

def test_saved_cursors_survive_additional_import_and_replay_stays_hidden(saved,tmp_path):
    client=Client(tmp_path/'additional')
    try:
        p=client.call('loadCase',{'path':str(saved['path'])})
        aux=client.call('importDisplaySeries',{'caseHash':p['caseHash'],'imagePath':image(tmp_path/'other.nii.gz'),'modality':'T2','annotationKind':'none'})
        destination=tmp_path/'additional.ressectionlab'
        client.call('saveCase',{'caseHash':p['caseHash'],'path':str(destination)})
        restored=load_workspace(destination)[2]
        for key,value in saved['state']['states'].items():assert restored['imagingState']['states'][key]==value
        assert restored['imagingState']['selectedSeriesId']==aux['seriesId']
        assert restored['selection']['visible'] is False
        invalid={**saved['selection'],'visible':True}
        client.call('saveCase',{'caseHash':p['caseHash'],'path':str(tmp_path/'invalid.ressectionlab'),'workspace':{'episodeReplay':invalid}},error=True)
    finally:client.close()
