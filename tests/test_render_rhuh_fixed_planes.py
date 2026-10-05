"""Analytical asymmetric arrays and metadata fixtures; never patient images."""
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import types

import nibabel as nib
import numpy as np
import pytest

MODULE=Path(__file__).resolve().parents[1]/'scripts/render_rhuh_fixed_planes.py'
SPEC=importlib.util.spec_from_file_location('rhuh_render_owner',MODULE)
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


def array_fixture():
    i,j,k=np.indices((3,5,7))
    return (100*i+10*j+k).astype(np.float32)


def test_asymmetric_native_planes_preserve_each_voxel_and_axis_mapping():
    a=array_fixture();before=a.copy();views=m.plane_views(a,m.AFFINE,(1,2,3))
    for plane,(fixed,index,h,v) in zip(views,[(0,1,1,2),(1,2,0,2),(2,3,0,1)]):
        assert (plane['fixed_axis'],plane['index'],plane['horizontal_axis'],plane['vertical_axis'])==(fixed,index,h,v)
        for row in range(a.shape[v]):
            for col in range(a.shape[h]):
                xyz=[None]*3;xyz[fixed]=index;xyz[h]=col;xyz[v]=row
                assert plane['pixels'][row,col]==a[tuple(xyz)]
        assert not plane['pixels'].flags.writeable
    assert np.array_equal(a,before)
    assert [(p['left'],p['right'],p['bottom'],p['top']) for p in views]==[
        ('A','P','I','S'),('R','L','I','S'),('R','L','A','P')]


def test_full_fov_edges_include_all_first_and_last_voxels():
    a=array_fixture();views=m.plane_views(a,m.AFFINE,(0,4,6))
    assert views[0]['extent']==(-.5,4.5,-.5,6.5)
    assert views[1]['extent']==(-.5,2.5,-.5,6.5)
    assert views[2]['extent']==(-.5,2.5,-.5,4.5)
    assert views[0]['pixels'][0,0]==a[0,0,0]
    assert views[2]['pixels'][-1,-1]==a[-1,-1,-1]


def test_reflected_permuted_physical_axes_derive_labels_and_spacing():
    affine=np.array([[0.,-2.,0.,7.],[3.,0.,0.,-8.],[0.,0.,-4.,9.],[0.,0.,0.,1.]])
    views=m.plane_views(array_fixture(),affine,(1,2,3))
    assert (views[0]['left'],views[0]['right'],views[0]['bottom'],views[0]['top'])==('R','L','S','I')
    assert views[0]['horizontal_step_mm']==2 and views[0]['vertical_step_mm']==4
    assert (views[1]['left'],views[1]['right'])==('P','A')


@pytest.mark.parametrize('indices',[(True,2,3),(-1,2,3),(3,2,3),(1,2),(1,2,7)])
def test_invalid_plane_choice_refused(indices):
    with pytest.raises(m.Rejected,match='indices'):m.plane_views(array_fixture(),m.AFFINE,indices)


def test_oblique_or_singular_affines_not_given_misleading_cardinal_edges():
    for affine in [np.zeros((4,4)),np.array([[1.,.1,0,0],[0,1.,0,0],[0,0,1.,0],[0,0,0,1.]])]:
        with pytest.raises(m.Rejected):m.plane_views(array_fixture(),affine,(1,2,3))


def test_display_clipping_never_changes_pixels_or_selects_new_windows(tmp_path):
    a=array_fixture();before=a.copy();fig,views=m.build_figure(a,m.AFFINE,(1,2,3),analytical_fixture=True)
    try:
        images=[ax.images[0] for ax in fig.axes if ax.images]
        assert len(images)==6
        for idx,image in enumerate(images):
            assert np.array_equal(image.get_array(),views[idx%3]['pixels'])
            assert image.get_clim()==m.WINDOWS[idx//3]
            assert image.origin=='lower' and image.get_interpolation()=='nearest'
            assert image.axes.get_aspect()==1.
        assert np.array_equal(a,before)
        fig.savefig(tmp_path/'fixture.png',dpi=70)
        fig.savefig(tmp_path/'fixture.svg',metadata={'Date':None})
        assert (tmp_path/'fixture.png').read_bytes().startswith(b'\x89PNG')
        svg=(tmp_path/'fixture.svg').read_text()
        assert 'Constructed orientation fixture' in svg
        assert 'no anatomical acceptance' in svg
    finally:fig.clear()


def analytical_snapshot(monkeypatch):
    decoder=m._decoder(MODULE.parents[1])  # Reads reviewed source code, never a patient file.
    a=array_fixture();h=nib.Nifti1Header();h.set_data_shape(a.shape);h.set_data_dtype(np.float32)
    h.set_xyzt_units('mm');h.set_qform(m.AFFINE,code=1);h.set_sform(m.AFFINE,code=0);h['vox_offset']=352
    raw=h.binaryblock+b'\0'*4+a.tobytes(order='F');compressed=gzip.compress(raw,mtime=0)
    report=decoder.inspect_stream(io.BytesIO(compressed))
    monkeypatch.setattr(m,'SHAPE',a.shape);monkeypatch.setattr(m,'INDICES',(1,2,3))
    monkeypatch.setattr(m,'ORIGINAL_SHA',hashlib.sha256(compressed).hexdigest())
    monkeypatch.setattr(decoder,'PUBLISHED_TOKEN',hashlib.md5(compressed).hexdigest())
    return decoder,a,compressed,report


def test_reviewed_decoder_recovery_preserves_fortran_xyz_storage(monkeypatch):
    decoder,a,compressed,report=analytical_snapshot(monkeypatch)
    recovered=m._decode_snapshot(decoder,compressed,report)
    assert np.array_equal(recovered,a) and not recovered.flags.writeable


def test_changed_saved_inspection_refused_before_plane_decode(monkeypatch):
    decoder,_,compressed,report=analytical_snapshot(monkeypatch)
    report['intensity']['scaled_finite_mean']+=1
    with pytest.raises(m.Rejected,match='differs'):m._decode_snapshot(decoder,compressed,report)


def test_trailing_or_corrupt_stream_rejected_by_reused_decoder(monkeypatch):
    decoder,_,compressed,report=analytical_snapshot(monkeypatch)
    for payload in [compressed+b'junk',compressed[:-2]]:
        with pytest.raises(decoder.Rejected):m._decode_snapshot(decoder,payload,report)


def test_changed_frozen_decoder_refused_without_module_execution(tmp_path,monkeypatch):
    p=tmp_path/m.DECODER_PATH;p.parent.mkdir(parents=True);p.write_text('raise AssertionError("must not execute")')
    with pytest.raises(m.Rejected,match='decoder_hash'):m._decoder(tmp_path)


def bind(root,path,obj):
    p=root/path;p.parent.mkdir(parents=True,exist_ok=True)
    raw=obj if isinstance(obj,bytes) else json.dumps(obj,sort_keys=True).encode()
    p.write_bytes(raw);return {'path':path,'sha256':hashlib.sha256(raw).hexdigest()}


def execute_fixture(tmp_path,monkeypatch):
    decoder,a,compressed,report=analytical_snapshot(monkeypatch)
    original='outputs/rhuh-single-image-v2/quarantine/run-analytical/RHUH-0001_0_t1.nii.gz'
    request={'payload':{'path':original,'measured_compressed_bytes':len(compressed)}}
    monkeypatch.setattr(m,'metadata_preflight',lambda root:(decoder,report,request))
    monkeypatch.setattr(m,'_decoder',lambda root:decoder)
    release={'schema':'resectionlab.rhuh-fixed-plane-release.v1','released':True,
             'action':'fixed_three_planes_two_windows_once','source':m.SOURCE,
             'renderer_sha256':hashlib.sha256(MODULE.read_bytes()).hexdigest(),'decoder_sha256':m.DECODER_SHA,
             'inspection_sha256':m.REPORT['sha256'],'inspection_audit_sha256':m.AUDIT['sha256'],
             'original_compressed_sha256':m.ORIGINAL_SHA,'output_directory':'outputs/rhuh-fixed-plane-qc-v1/run-analytical'}
    return decoder,a,compressed,request,release


@pytest.mark.parametrize('key,value',[
    ('released',False),('inspection_sha256','0'*64),('decoder_sha256','0'*64),
    ('original_compressed_sha256','0'*64),('renderer_sha256','0'*64),
    ('action','render_alternative_views'),('output_directory','data/replace-original'),
])
def test_no_source_open_for_wrong_release(tmp_path,monkeypatch,key,value):
    _,_,_,request,release=execute_fixture(tmp_path,monkeypatch);release[key]=value
    binding=bind(tmp_path,'artifacts/analytical-release.json',release)
    assert not (tmp_path/request['payload']['path']).exists()
    with pytest.raises(m.Rejected):m.execute(tmp_path,binding)


def test_existing_output_refused_before_source_open(tmp_path,monkeypatch):
    _,_,_,request,release=execute_fixture(tmp_path,monkeypatch)
    out=tmp_path/release['output_directory'];out.mkdir(parents=True);(out/'keep').write_bytes(b'prior result')
    binding=bind(tmp_path,'artifacts/analytical-release.json',release)
    with pytest.raises(m.Rejected,match='fresh'):m.execute(tmp_path,binding)
    assert (out/'keep').read_bytes()==b'prior result'


def test_analytical_authorized_render_emits_bound_figures_and_receipt(tmp_path,monkeypatch):
    _,a,compressed,request,release=execute_fixture(tmp_path,monkeypatch)
    original=tmp_path/request['payload']['path'];original.parent.mkdir(parents=True);original.write_bytes(compressed)
    binding=bind(tmp_path,'artifacts/analytical-release.json',release)
    result=m.execute(tmp_path,binding)
    assert result['accepted'] and result['status']=='fixed_planes_rendered_review_required'
    assert not result['anatomical_validation'] and not result['scientific_use_released']
    out=tmp_path/release['output_directory']
    assert set(p.name for p in out.iterdir())=={'fixed-planes.png','fixed-planes.svg','receipt.json'}
    for name,info in result['artifacts'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==info['sha256']
    assert original.read_bytes()==compressed


def test_changed_original_keeps_failed_attempt_without_figures(tmp_path,monkeypatch):
    _,_,compressed,request,release=execute_fixture(tmp_path,monkeypatch)
    p=tmp_path/request['payload']['path'];p.parent.mkdir(parents=True);p.write_bytes(b'x'*len(compressed))
    result=m.execute(tmp_path,bind(tmp_path,'artifacts/analytical-release.json',release))
    assert not result['accepted'] and result['failure']=='original_compressed_digest'
    assert set(p.name for p in (tmp_path/release['output_directory']).iterdir())=={'receipt.json'}


def metadata_fixture(monkeypatch):
    _,_,_,report=analytical_snapshot(monkeypatch)
    monkeypatch.setattr(m,'WINDOWS',((report['intensity']['scaled_finite_min'],report['intensity']['scaled_finite_max']),(-3.,3.)))
    request_binding={'path':'artifacts/analytical-request.json','sha256':'1'*64}
    report['bindings']={'source':m.SOURCE,'request':request_binding}
    request={'source':m.SOURCE,'payload':{'sha256':m.ORIGINAL_SHA,'measured_compressed_bytes':report['compressed_bytes']}}
    docs={m.REPORT['path']:report,m.AUDIT['path']:{'schema':'resectionlab.rhuh-saved-inspection-independent-audit.v1','accepted_for_saved_result_consistency':True},
          m.PARENT['path']:{'accepted':True,'stdout_sha256':m.REPORT['sha256'],'scientific_use_released':False,'inspector_exit_code':0},
          m.TERMINAL['path']:{'accepted':True,'exit_code':0,'parent_summary_sha256':m.PARENT['sha256']},
          request_binding['path']:request}
    decoder=types.SimpleNamespace(_json=lambda root,b:docs[b['path']],preflight=lambda root,r:None)
    monkeypatch.setattr(m,'_decoder',lambda root:decoder)
    return docs,report


def test_saved_metadata_preflight_requires_no_image_path(tmp_path,monkeypatch):
    docs,report=metadata_fixture(monkeypatch)
    _,checked,request=m.metadata_preflight(tmp_path)
    assert checked is report and 'path' not in request['payload']


@pytest.mark.parametrize('field', ['shape','affine','audit','source','window','nonfinite','terminal'])
def test_changed_saved_metadata_refuses_figure_permission(tmp_path,monkeypatch,field):
    docs,r=metadata_fixture(monkeypatch)
    if field=='shape':r['shape'][0]+=1
    if field=='affine':r['geometry']['qform']['affine_in_source_units'][0][3]+=.1
    if field=='audit':docs[m.AUDIT['path']]['accepted_for_saved_result_consistency']=False
    if field=='source':r['bindings']['source']='/wrong-source'
    if field=='window':r['intensity']['scaled_finite_min']-=1
    if field=='nonfinite':r['intensity']['scaled_nonfinite_voxels']=1
    if field=='terminal':docs[m.TERMINAL['path']]['exit_code']=1
    with pytest.raises(m.Rejected):m.metadata_preflight(tmp_path)
