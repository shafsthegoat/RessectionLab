"""Independent Case3 registration preflight controls; no acquired arrays/tags.

Initial counterexamples use the retained source snapshot. Final controls require
explicit pins and exercise only generated arrays, headers and subprocesses.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time
import types
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/resect-case3-registration-preflight-independent-v1'
RUNNER=ROOT/'scripts/poc_resect_case3_image_registration.py'
SNAPSHOT=BASE/'initial-runner.py.txt'
SNAPSHOT_SHA='6de7f253efa7db3806fbe46fb63344d2c4ef9106462608e31c35ba4ad45a516c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
    module=types.ModuleType('independent_case3_registration')
    module.__file__=str(RUNNER)
    exec(compile(Path(path).read_bytes(),str(path),'exec'),module.__dict__)
    module.__file__=str(path)
    return module


def candidate(module,inverse=True):
    forward=np.eye(4);forward[:3,3]=[3.,-2.,1.]
    record={'schema':module.SCHEMA+'-candidate','case_id':module.CASE,
        'preflight_sha256':'a'*64,'fit_release_sha256':'b'*64,'runner_sha256':module.digest(module.__file__),
        'sources_rechecked_after_each_decode':True,
        'source_fixity_after_fit':{name:item['sha256'] for name,item in module.INPUTS.items()},
        'runtime':{'generated_fixture':True},
        'report':{'before_RAS_mm_to_during_RAS_mm':forward.tolist(),'landmarks_used':False,'configuration':module.CONFIG,
            'fixed':'before_resection_US','moving':'during_resection_US',
            'sampling_direction':'before-RAS output physical point -> during-RAS input physical point',
            'identity_baseline_before_RAS_to_during_RAS':np.eye(4).tolist()}}
    if inverse: record['report']['during_RAS_mm_to_before_RAS_mm']=np.linalg.inv(forward).tolist()
    return record


def supervision(module):
    return {'status':'completed','exit_code':0,'worker_reaped':True,'samples':2,
        'sampled_family_rss_stop_bytes':module.LIMITS['sampled_family_rss_stop_bytes'],
        'peak_family_rss_bytes_sampled':1024}


class CompletedProcessDouble:
    pid=99999999
    waited=False
    def poll(self):return 0
    def wait(self,timeout=None):self.waited=True;return 0


def initial_findings():
    assert sha(SNAPSHOT)==SNAPSHOT_SHA
    module=load(SNAPSHOT)
    findings=[]
    for kind in ('missing_inverse','inconsistent_inverse'):
        with tempfile.TemporaryDirectory(prefix='case3-independent-') as directory:
            output=Path(directory)
            record=candidate(module,inverse=kind!='missing_inverse')
            if kind=='inconsistent_inverse':record['report']['during_RAS_mm_to_before_RAS_mm']=np.eye(4).tolist()
            module.publish(output/'candidate.json',record)
            module.publish(output/'supervision.json',supervision(module))
            module.seal_attempt(output,'completed',0,'a'*64,'b'*64)
            assert (output/'frozen-transform.json').exists()
            findings.append({'name':kind,'actual':'success freeze published','expected':'reject before freeze'})
    with tempfile.TemporaryDirectory(prefix='case3-independent-') as directory:
        child=CompletedProcessDouble()
        with patch.object(module.subprocess,'Popen',return_value=child),patch.object(module.os,'killpg',side_effect=PermissionError('generated cleanup failure')):
            try:
                module.supervise_sampled(['generated'],Path(directory)/'log',deadline=time.monotonic()+5,
                    on_start=lambda _:None,observer=lambda *_:{'worker_rss_bytes':10,'family_rss_bytes':10,'available_bytes':100,'live_members':[]})
            except PermissionError:
                assert not child.waited
                findings.append({'name':'killpg_permission_error','actual':'exception escapes; process.wait not reached; no supervision record returned',
                    'expected':'retain cleanup failure, attempt reap, return durable failure evidence'})
            else:raise AssertionError('initial cleanup counterexample not reproduced')
    record={'schema':'independent-resect-case3-initial-findings-v1','status':'reproduced_prefit_defects',
        'source_snapshot_sha256':SNAPSHOT_SHA,'counterexamples':findings,
        'other_source_review_observations':[
            'Initial viewed worker omitted source rehash after decode/fit; owner had already added it before retained snapshot. No source-mutation reproduction claimed against snapshot.',
            'Snapshot run_fit does not route ordinary SIGTERM through termination_cleanup; source observation only at this stage.',
            'Snapshot runtime identity is version/path metadata without Python or SimpleITK binary hashes.'],
        'acquired_arrays_or_tags_opened':False,'optimizer_executed':False,'real_worker_spawned':False}
    target=BASE/'initial-findings.json'
    with target.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps({'path':str(target.relative_to(ROOT)),'sha256':sha(target),'counterexamples':len(findings)},indent=2))


def expect_rejection(call):
    try:call()
    except (ValueError,TypeError):return
    raise AssertionError('invalid generated input accepted')


def final_controls(module):
    checks=[]
    cohort={'members':[{'patient_group':module.CASE,'case_id':'Case3','role':'TRAIN','all_timepoints_and_derivatives_grouped':True}]}
    for kind in ('case','source_hash','source_path','role','duplicate','cohort_pin'):
        identity=module.CASE;inputs=deepcopy(module.INPUTS);roles=deepcopy(cohort);pin=module.COHORT_SHA
        if kind=='case':identity='RESECT:Case4'
        elif kind=='source_hash':inputs['before']['sha256']='0'*64
        elif kind=='source_path':inputs['during']['path']='data/other.nii.gz'
        elif kind=='role':roles['members'][0]['role']='SELECT'
        elif kind=='duplicate':roles['members']*=2
        else:pin='0'*64
        expect_rejection(lambda:module.validate_identity(identity,inputs,roles,pin));checks.append('identity_'+kind)
    import nibabel as nib
    h=nib.Nifti1Header();h.set_data_shape((9,10,11));h.set_data_dtype(np.float32);h.set_xyzt_units('mm')
    affine=np.array([[0.,-1.5,0.,7.],[2.,0.,0.,-5.],[0.,0.,2.5,3.],[0.,0.,0.,1.]])
    h.set_sform(affine,code=1);h.set_zooms((2.,1.5,2.5));h['qform_code']=0;h['pixdim'][0]=0
    h['vox_offset']=352;h['scl_slope']=.9;h['scl_inter']=0
    with patch.object(nib,'load',side_effect=AssertionError('header check opened array proxy')):
        observed=module.inspect_header(h.binaryblock)
        assert observed['selected_affine']==affine.tolist() and observed['inactive_qform_reconstructed'] is False
        assert observed['image_array_accessed'] is False and observed['qform_qfac_raw']==0
        for field,value in [('sform_code',0),('qform_code',1),('vox_offset',360),('scl_slope',0)]:
            changed=h.copy();changed[field]=value
            if field=='qform_code':changed['pixdim'][0]=1
            expect_rejection(lambda:module.inspect_header(changed.binaryblock));checks.append('header_'+field)
    checks.append('active_sform_inactive_invalid_qfac_preserved')
    # Point/voxel direction check without running any registration optimizer.
    import SimpleITK as sitk
    from resectionlab.preprocessing import to_sitk,sitk_transform_to_ras,RAS_TO_LPS
    a=np.zeros((9,10,11),np.float32);a[6,4,5]=1
    image=to_sitk(a,affine)
    for index in ((0,0,0),(6,4,5),(8,9,10)):
        expected=(RAS_TO_LPS@np.r_[nib.affines.apply_affine(affine,index),1.])[:3]
        np.testing.assert_allclose(image.TransformIndexToPhysicalPoint(index),expected,atol=1e-10)
    shift=affine[:3,0]*2
    transform=sitk.TranslationTransform(3,tuple(RAS_TO_LPS[:3,:3]@shift))
    matrix=sitk_transform_to_ras(transform)
    np.testing.assert_allclose(matrix[:3,3],shift,atol=1e-10)
    sampled=module.rigid.resample(a,affine,a,affine,transform,label=True)
    assert np.argwhere(sampled==1).tolist()==[[4,4,5]]
    before=nib.affines.apply_affine(affine,[4,4,5]);during=nib.affines.apply_affine(affine,[6,4,5])
    np.testing.assert_allclose(nib.affines.apply_affine(matrix,before),during)
    np.testing.assert_allclose(nib.affines.apply_affine(np.linalg.inv(matrix),during),before)
    checks.append('generated_RAS_LPS_and_fixed_to_moving_sampling_direction')
    # The worker read guard denies tags globally and every unlisted data path.
    with tempfile.TemporaryDirectory(prefix='case3-independent-') as directory:
        data=Path(directory)/'data';data.mkdir();allowed=data/'before.nii.gz';allowed.write_bytes(b'generated')
        guard=module.patient_read_guard(data,[allowed])
        guard('open',(str(allowed),'rb',os.O_RDONLY))
        for path,mode,flags in [(data/'other.nii.gz','rb',os.O_RDONLY),(Path(directory)/'points.tag','rb',os.O_RDONLY),
                                (allowed,'wb',os.O_WRONLY),(allowed,'rb',os.O_RDWR)]:
            try:guard('open',(str(path),mode,flags))
            except PermissionError:checks.append('patient_read_guard_'+path.name+'_'+str(flags))
            else:raise AssertionError('unallowed read or write admitted')
    for kind in ('missing_inverse','inconsistent_inverse','direction','identity','runtime','source_fixity','landmarks','config'):
        with tempfile.TemporaryDirectory(prefix='case3-independent-') as directory,patch.object(module,'runtime_record',return_value={'generated_fixture':True}):
            output=Path(directory);record=candidate(module)
            if kind=='missing_inverse':del record['report']['during_RAS_mm_to_before_RAS_mm']
            elif kind=='inconsistent_inverse':record['report']['during_RAS_mm_to_before_RAS_mm']=np.eye(4).tolist()
            elif kind=='direction':record['report']['sampling_direction']='during-to-before'
            elif kind=='identity':record['report']['identity_baseline_before_RAS_to_during_RAS'][0][3]=2
            elif kind=='runtime':record['runtime']={'changed':True}
            elif kind=='source_fixity':record['source_fixity_after_fit']['before']='0'*64
            elif kind=='landmarks':record['report']['landmarks_used']=True
            else:record['report']['configuration']={}
            module.publish(output/'candidate.json',record);module.publish(output/'supervision.json',supervision(module))
            expect_rejection(lambda:module.seal_attempt(output,'completed',0,'a'*64,'b'*64))
            assert not (output/'frozen-transform.json').exists();checks.append('seal_reject_'+kind)
    with tempfile.TemporaryDirectory(prefix='case3-independent-') as directory,patch.object(module,'runtime_record',return_value={'generated_fixture':True}):
        output=Path(directory);module.publish(output/'candidate.json',candidate(module));module.publish(output/'supervision.json',supervision(module))
        module.seal_attempt(output,'completed',0,'a'*64,'b'*64)
        expect_rejection(lambda:module.evaluation_gate(output,{}))
        before_hash=sha(output/'frozen-transform.json')
        try:module.publish(output/'frozen-transform.json',{'changed':True})
        except FileExistsError:pass
        else:raise AssertionError('freeze overwritten')
        assert sha(output/'frozen-transform.json')==before_hash
        checks.append('success_freeze_exclusive_and_evaluation_still_separately_gated')
    for state,code in [('timeout',-9),('memory_stop',-9),('cleanup_failed',0),('worker_failed',1),('supervisor_failed',None)]:
        with tempfile.TemporaryDirectory(prefix='case3-independent-') as directory:
            output=Path(directory);module.publish(output/'candidate.json',candidate(module))
            module.seal_attempt(output,state,code,'a'*64,'b'*64)
            assert not (output/'frozen-transform.json').exists();expect_rejection(lambda:module.evaluation_gate(output,{}))
            checks.append('failed_state_never_freezes_'+state)
    with tempfile.TemporaryDirectory(prefix='case3-independent-') as directory:
        child=CompletedProcessDouble()
        with patch.object(module.subprocess,'Popen',return_value=child),patch.object(module.os,'killpg',side_effect=PermissionError('generated cleanup failure')):
            result=module.supervise_sampled(['generated'],Path(directory)/'log',deadline=time.monotonic()+5,
                on_start=lambda _:None,observer=lambda *_:{'worker_rss_bytes':10,'family_rss_bytes':10,'available_bytes':100,'live_members':[]})
        assert result['status']=='cleanup_failed' and result['worker_reaped'] and child.waited
        checks.append('killpg_permission_failure_retained_and_reaped')
    return checks


def worker_continuity_controls(module):
    checks=[]
    for change in ('none','during_decode','during_fit','runtime','code'):
        with tempfile.TemporaryDirectory(prefix='case3-independent-') as directory,ExitStack() as stack:
            root=Path(directory);inputs={};arrays={};headers={}
            for key in ('before','during'):
                path=root/(key+'.bin');path.write_bytes(b'generated-'+key.encode())
                inputs[key]={'path':path.name,'bytes':path.stat().st_size,'sha256':sha(path)}
                headers[key]={'expected_expanded_bytes':352+4*4**3,'selected_affine':np.eye(4).tolist()}
            output=root/'output';output.mkdir();pre=root/'pre.json';rel=root/'release.json';pre.write_text('{}');rel.write_text('{}')
            runtime={'generated_fixture':True};baseline={'headers':headers,'runtime':runtime,'runner_sha256':module.digest(module.__file__)}
            calls=[]
            actual_verify=module.verify_source_file
            def verify(path,entry,*args,**kwargs):calls.append(Path(path).name);return actual_verify(path,entry,*args,**kwargs)
            def decode(path,digest):
                if change=='during_decode':Path(path).write_bytes(b'changed-source')
                return types.SimpleNamespace(affine=np.eye(4)),np.ones((4,4,4),np.float32),352+4*4**3
            fitted=False
            def fit(*args):
                nonlocal fitted
                fitted=True
                if change=='during_fit':(root/inputs['before']['path']).write_bytes(b'changed-source')
                return None,candidate(module)['report']
            def runtime_value():return {'changed':True} if change=='runtime' and fitted else runtime
            def code_check():
                if change=='code' and fitted:raise ValueError('generated code pin changed')
            for obj,name,value in [(module,'ROOT',root),(module,'INPUTS',inputs),(module,'require_fit_release',lambda *_:baseline),
                (module,'preflight',lambda:baseline),(module.sys,'addaudithook',lambda *_:None),(module.rigid,'checked_source',decode),
                (module,'verify_source_file',verify),(module,'fit_arrays',fit),(module,'runtime_record',runtime_value),(module,'verify_metadata',code_check)]:
                stack.enter_context(patch.object(obj,name,value))
            if change=='none':
                module.worker(pre,rel,output)
                assert (output/'candidate.json').exists() and calls==['before.bin','during.bin','before.bin','during.bin']
                assert json.loads((output/'candidate.json').read_text())['source_fixity_after_fit']=={k:v['sha256'] for k,v in inputs.items()}
            else:
                expect_rejection(lambda:module.worker(pre,rel,output))
                assert not (output/'candidate.json').exists()
                if change=='during_decode':assert not fitted
            checks.append('worker_source_runtime_code_continuity_'+change)
    return checks


def subprocess_controls(module):
    results={}
    with tempfile.TemporaryDirectory(prefix='case3-independent-') as directory:
        base=Path(directory)
        for name,command,cap,deadline in [
            ('timeout',[sys.executable,'-c','import time;time.sleep(10)'],None,.2),
            ('memory_stop',[sys.executable,'-c','import time;x=bytearray(32*1024**2);time.sleep(10)'],24*1024**2,3),
            ('success',[sys.executable,'-c','pass'],None,3)]:
            result=module.supervise_sampled(command,base/(name+'.log'),deadline=time.monotonic()+deadline,on_start=lambda _:None,rss_stop=cap)
            assert result['status']==('completed' if name=='success' else name)
            assert result['worker_reaped'] and result['process_group_cleanup_attempted']
            assert result['samples']>0 and result['minimum_available_memory_bytes_sampled']>0
            results[name]=result
        # Real SIGTERM reaches supervisor, whose handler must reap its own worker
        # and kill the worker's child. All three are generated sleeping processes.
        program=r'''
import sys,time,json,importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('qa',sys.argv[1]);qa=importlib.util.module_from_spec(spec);spec.loader.exec_module(qa)
runner=qa.load(qa.RUNNER);base=Path(sys.argv[2])
child_code="import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)']);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(20)"
result=runner.supervise_sampled([sys.executable,'-c',child_code,str(base/'grandchild.pid')],base/'sigterm.log',deadline=time.monotonic()+10,on_start=lambda pid:(base/'worker.pid').write_text(str(pid)))
(base/'sigterm-result.json').write_text(json.dumps(result))
'''
        supervisor=subprocess.Popen([sys.executable,'-c',program,__file__,directory],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            until=time.monotonic()+5
            while not (base/'grandchild.pid').exists() and time.monotonic()<until:time.sleep(.02)
            assert (base/'grandchild.pid').exists()
            supervisor.send_signal(signal.SIGTERM)
            stdout,stderr=supervisor.communicate(timeout=5);assert supervisor.returncode==0,(stdout,stderr)
            result=json.loads((base/'sigterm-result.json').read_text())
            assert result['status']=='supervisor_failed' and result['worker_reaped'] and 'SIGTERM' in result['error']
            import psutil
            for filename in ('worker.pid','grandchild.pid'):
                pid=int((base/filename).read_text())
                try:
                    child=psutil.Process(pid)
                    until=time.monotonic()+1
                    while child.is_running() and child.status()!=psutil.STATUS_ZOMBIE and time.monotonic()<until:time.sleep(.02)
                    assert not child.is_running() or child.status()==psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:pass
            results['SIGTERM_and_descendant_cleanup']=result
        finally:
            if supervisor.poll() is None:supervisor.kill();supervisor.wait()
            for filename in ('worker.pid','grandchild.pid'):
                if (base/filename).exists():
                    try:os.kill(int((base/filename).read_text()),signal.SIGKILL)
                    except ProcessLookupError:pass
    return results


def release_binding_controls(module):
    results=[]
    with tempfile.TemporaryDirectory(prefix='case3-release-fixture-') as directory,patch.object(module,'runtime_record',return_value={'generated_fixture':True}):
        base=Path(directory);pre=base/'pre.json';release=base/'release.json'
        output=ROOT/'outputs'/'independent_generated_release_fixture_not_created'
        before={'schema':module.SCHEMA+'-preflight','status':'headers_verified_fit_not_run','runner_sha256':sha(RUNNER),
            'runtime':{'generated_fixture':True},'pins':module.PINS,'configuration':module.CONFIG,'limits':module.LIMITS,
            'case_id':module.CASE,'inputs':module.INPUTS,'cohort_sha256':module.COHORT_SHA}
        grant={'schema':module.SCHEMA+'-root-release','decision':'release_one_fit','case_id':module.CASE,
            'runner_sha256':sha(RUNNER),'output_directory':str(output.resolve()),'landmark_access_permitted':False}
        def put(b,r):
            pre.write_text(json.dumps(b));r={**r,'preflight_sha256':sha(pre)};release.write_text(json.dumps(r))
        put(before,grant)
        assert module.require_fit_release(pre,release,output)==before
        results.append('valid_generated_release_binding_only_no_fit')
        for key,value in [('decision','NOT_RELEASED'),('case_id','RESECT:Case4'),('runner_sha256','0'*64),
                          ('landmark_access_permitted',True),('output_directory',str(base/'outside'))]:
            changed=deepcopy(grant);changed[key]=value;put(before,changed)
            expect_rejection(lambda:module.require_fit_release(pre,release,output));results.append('release_reject_'+key)
        for key,value in [('runtime',{}),('pins',{}),('configuration',{}),('limits',{}),('cohort_sha256','0'*64),('inputs',{})]:
            changed=deepcopy(before);changed[key]=value;put(changed,grant)
            expect_rejection(lambda:module.require_fit_release(pre,release,output));results.append('preflight_reject_'+key)
    return results


def header_only_final():
    preparation='artifacts/resect-case3-image-registration-preparation-v1/'
    pins={
        'scripts/poc_resect_case3_image_registration.py':'e17d755eeadb6a3d1fd365b7a3b4dc31f9e27c4f93afdb6045d8f6e27ef9612f',
        'tests/test_poc_resect_case3_image_registration.py':'bec991c70320d981e6d37cf65d4db988a50535f5744d6ccc27aab0771244f908',
        preparation+'header-preflight-rss-v2.json':'ef9e0cd6029e869a6364de6c946f0284bc0a7ca1f700695a8bb75b52e60fd7ba',
        preparation+'prospective-protocol.json':'b1b27118ae79b6d6ab576c97298749ba5f126d8fac985788c147e42961e14bd7',
        preparation+'preparation-receipt.json':'ece210465129719de3fa20d759f061109fa691252777278b6c7d784edf47f0d7',
        str((BASE/'initial-runner.py.txt').relative_to(ROOT)):SNAPSHOT_SHA,
        str((BASE/'initial-findings.json').relative_to(ROOT)):'0d451fd096b23fa356af2f1ed8d7b756e55602bee1a06a3a14c0063b0d6dd706',
        str((BASE/'generated-controls.json').relative_to(ROOT)):'094d0fd87b3a7591f8f2832e6dc684b9d281ff5df489b643407a1dc1279c9cd6',
        str((BASE/'generated-auditor.py.txt').relative_to(ROOT)):'c7b3a1c8e9e6e6fdf07114e0f1d5c8277e8f497d3360928c560a122f54e3035a',
    }
    for path,pin in pins.items():assert sha(ROOT/path)==pin,path
    module=load(RUNNER)
    receipt=json.loads((ROOT/preparation/'preparation-receipt.json').read_text())
    before=json.loads((ROOT/preparation/'header-preflight-rss-v2.json').read_text())
    protocol=json.loads((ROOT/preparation/'prospective-protocol.json').read_text())
    for path,item in receipt['files'].items():
        assert sha(ROOT/path)==item['sha256'] and (ROOT/path).stat().st_size==item['bytes']
        pins[path]=item['sha256']
    assert protocol['release_record_fields']['decision']=='NOT_RELEASED'
    assert protocol['evaluation']['status']=='not_released_not_run'
    assert protocol['prepared_evidence']['tag_coordinate_reads']==0
    assert before['configuration']==protocol['configuration']==module.CONFIG
    assert before['limits']==protocol['limits']==module.LIMITS
    assert before['inputs']==protocol['inputs']==module.INPUTS
    assert before['pins']==protocol['source_and_helper_pins']==module.PINS
    assert before['runtime']==module.runtime_record()
    for path,pin in module.PINS.items():assert sha(ROOT/path)==pin; pins[path]=pin
    cohort=json.loads((ROOT/module.COHORT).read_text())
    member=[m for m in cohort['members'] if m['patient_group']=='RESECT:Case3']
    assert len(member)==1 and member[0]['role']=='TRAIN' and member[0]['case_id']=='Case3'
    assert member[0]['all_timepoints_and_derivatives_grouped'] is True
    headers=[]
    for name,item in module.INPUTS.items():
        path=ROOT/item['path'];digest=hashlib.sha256();md5=hashlib.md5()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024**2),b''):digest.update(block);md5.update(block)
        assert digest.hexdigest()==item['sha256'] and md5.hexdigest()==item['expected_md5']
        assert path.stat().st_size==item['bytes']
        with gzip.open(path,'rb') as stream:raw=stream.read(348)
        assert len(raw)==348
        endian='<' if struct.unpack('<i',raw[:4])[0]==348 else '>'
        assert struct.unpack(endian+'i',raw[:4])[0]==348 and raw[344:348]==b'n+1\0'
        dim=struct.unpack_from(endian+'8h',raw,40);shape=list(dim[1:4]);assert dim[0]==3
        datatype,bitpix=struct.unpack_from(endian+'2h',raw,70);assert (datatype,bitpix)==(16,32)
        pixdim=struct.unpack_from(endian+'8f',raw,76)
        offset,slope,intercept=struct.unpack_from(endian+'3f',raw,108)
        assert offset==352 and np.isfinite([slope,intercept]).all() and slope>0
        assert raw[123]&7==2
        qform,sform=struct.unpack_from(endian+'2h',raw,252);assert qform==0 and sform==1
        affine=np.eye(4);affine[:3,:]=np.asarray(struct.unpack_from(endian+'12f',raw,280)).reshape(3,4)
        spacing=np.linalg.norm(affine[:3,:3],axis=0);basis=affine[:3,:3]/spacing
        np.testing.assert_allclose(basis.T@basis,np.eye(3),rtol=0,atol=1e-4)
        np.testing.assert_allclose(pixdim[1:4],spacing,rtol=1e-5,atol=1e-5)
        corners=np.asarray([[x,y,z] for x in (-.5,shape[0]-.5) for y in (-.5,shape[1]-.5) for z in (-.5,shape[2]-.5)])
        world=corners@affine[:3,:3].T+affine[:3,3]
        expected=before['headers'][name]
        assert hashlib.sha256(raw).hexdigest()==expected['header_sha256']
        assert shape==expected['shape'] and np.array_equal(affine,expected['selected_affine'])
        assert [float(slope),float(intercept)]==[expected['source_scaling_slope'],expected['source_scaling_intercept']]
        assert 352+int(np.prod(shape))*4==expected['expected_expanded_bytes']
        np.testing.assert_allclose(world.min(0),expected['full_cell_bounds_ras_mm']['minimum'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(world.max(0),expected['full_cell_bounds_ras_mm']['maximum'],rtol=0,atol=1e-12)
        assert sha(path)==item['sha256']
        headers.append({'phase':name,'compressed_sha256':item['sha256'],'compressed_md5':item['expected_md5'],
            'header_sha256':expected['header_sha256'],'shape':shape,'active_sform_code':sform,'inactive_qform_code':qform,
            'raw_qfac':pixdim[0],'selected_affine_matches':True,'all_eight_physical_corners_match':True,
            'header_bytes_returned':348,'voxel_arrays_decoded':False,'full_payload_length_or_finiteness_validated':False})
    release_checks=release_binding_controls(module)
    for path,pin in pins.items():assert sha(ROOT/path)==pin,path
    result={'schema':'independent-resect-case3-registration-preflight-v1','status':'pass_prospective_only',
        'source_and_evidence_pins_unchanged_before_after':pins,'auditor_sha256':sha(__file__),
        'exact_case_role':'RESECT:Case3 TRAIN','headers_and_compressed_fixity':headers,
        'runtime_hashes':{key:value['sha256'] for key,value in before['runtime']['binary_and_wrapper_bindings'].items()},
        'runtime_versions':{key:before['runtime'][key] for key in ('python_version','numpy','nibabel','SimpleITK','psutil')},
        'generated_guard_checks':36,'generated_subprocess_controls':4,'additional_release_binding_controls':release_checks,
        'resolved_initial_findings':['Missing/inconsistent inverse now rejected before freeze.','Cleanup PermissionError retained while reaping continues.',
            'SIGTERM independently verified to stop/reap worker and stop its child.','Sources rehashed after each decode and after fitting; runtime/code continuity checked before candidate and seal.'],
        'harness_correction':'The first active-qform negative fixture retained invalid inactive qfac0 and raised HeaderDataError; corrected qfac1 isolates rejection of active qform. No production change or patient decode.',
        'actual_fit_released_or_executed_by_QA':False,'registration_optimizer_executed':False,'tag_coordinates_read':False,
        'source_arrays_decoded':False,'anatomical_alignment_accepted':False,
        'limitations':['Source header convention and software controls do not establish anatomical correspondence or scanner/tracker calibration.',
            'Sampled RSS stop is not an OS hard cap and may miss transient peaks; family is the worker and observed descendants.',
            'Future full-tag evaluation requires a successful sealed transform and separate release; no registration accuracy result exists yet.']}
    target=BASE/'independent-qa.json'
    with target.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':result['status'],'sha256':sha(target),'release_controls':len(release_checks),'headers':len(headers)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial',action='store_true')
    parser.add_argument('--generated-checks',action='store_true')
    parser.add_argument('--final-header-only',action='store_true')
    args=parser.parse_args()
    if args.initial:initial_findings()
    elif args.generated_checks:
        module=load(RUNNER)
        before=sha(RUNNER)
        checks=final_controls(module)+worker_continuity_controls(module)
        processes=subprocess_controls(module)
        assert sha(RUNNER)==before,'runner changed during audit'
        result={'schema':'independent-resect-case3-generated-preflight-controls-v1','status':'pass',
            'runner_sha256':before,'auditor_sha256':sha(__file__),'checks':checks,'subprocess_controls':processes,
            'acquired_arrays_or_tags_opened':False,'registration_optimizer_executed':False,
            'initial_findings_sha256':sha(BASE/'initial-findings.json'),'initial_source_snapshot_sha256':sha(SNAPSHOT)}
        target=BASE/'generated-controls.json'
        with target.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
        print(json.dumps({'status':'pass','checks':len(checks),'process_controls':len(processes),'sha256':sha(target)},indent=2))
    elif args.final_header_only:header_only_final()
    else:parser.error('choose --initial, --generated-checks, or --final-header-only')
