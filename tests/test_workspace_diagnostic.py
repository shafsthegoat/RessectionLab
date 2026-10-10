"""Generated-only saved-file importer controls; no Case4 payload access."""
import hashlib, json
from types import SimpleNamespace
import numpy as np
import nibabel as nib
import pytest
from resectionlab import workspace_diagnostic as m
def fixture(tmp_path):
    source=SimpleNamespace(mri=np.zeros((2,2,2)),affine=np.eye(4),frame='RAS+',source_refs=[SimpleNamespace(source_id='structural',provenance='observed',sha256='a'*64)])
    state=np.array([-1,0,1,-1,-1,-1,-1,-1],dtype=np.int8).reshape((2,2,2));coverage=(state!=-1).astype(np.uint8);outputs={}
    for name,array in [('state',state),('coverage',coverage)]:
        img=nib.Nifti1Image(array,np.eye(4));img.header.set_xyzt_units('mm');img.set_sform(np.eye(4),code=2);img.set_qform(None,code=0)
        path=tmp_path/(name+'.nii.gz');nib.save(img,path);outputs[name]={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'dtype':str(array.dtype)}
    d={'schema':'case4_unreviewed_diagnostic_display_layer_v1','scope':'display_only_atlas_native_grid','source_sha256':{'t1c':'a'*64},'model_output_verified':True,'anatomical_qc':'unreviewed_inferior_mask_omission','training_overlap_status':'unknown','same_grid_only':True,'planning_eligible':False,'evaluation_eligible':False,'clinical_evidence':False,'shape_xyz':[2,2,2],'affine_ras_mm':np.eye(4).tolist(),'outputs':outputs,'predicted_voxels':2,'unknown_voxels':6}
    path=tmp_path/'display-layer.json';path.write_text(json.dumps(d));return source,path,d

def test_signed_unknown_and_xyz_order(tmp_path):
    source,path,d=fixture(tmp_path);_,arrays,grids=m.load_saved_diagnostic(path,hashlib.sha256(path.read_bytes()).hexdigest(),source)
    assert arrays['state'].dtype==np.int8 and arrays['state'][0,1,0]==1 and arrays['state'][0,0,0]==-1
    assert grids['state']['sformCode']==2

@pytest.mark.parametrize('change',['source','pin','bytes','path','grid','counts','admission'])
def test_refuses_changed_bindings(tmp_path,change):
    source,path,d=fixture(tmp_path)
    if change=='source':source.source_refs[0].sha256='b'*64
    if change=='bytes':(tmp_path/'state.nii.gz').write_bytes(b'changed')
    if change=='path':d['outputs']['state']['path']='../state.nii.gz'
    if change=='grid':d['affine_ras_mm'][0][3]=1
    if change=='counts':d['predicted_voxels']=3
    if change=='admission':d['planning_eligible']=True
    path.write_text(json.dumps(d));pin=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):m.load_saved_diagnostic(path,'0'*64 if change=='pin' else pin,source)
