"""Generated-only launcher regressions. No subprocess or patient source is used."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name('run_public_qc_owned.py')
spec=importlib.util.spec_from_file_location('train_public_launcher',SOURCE)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
class Process:
    pid=7654321
    def __init__(self,code=None,refuse=False):self.code=code;self.refuse=refuse;self.kills=0;self.waits=0
    def poll(self):return self.code
    def kill(self):
        self.kills+=1
        if self.refuse:raise PermissionError('generated kill refusal')
        self.code=-9
    def wait(self,timeout):
        self.waits+=1
        if self.code is None:raise subprocess.TimeoutExpired('generated',timeout)
        return self.code
class Sampler:
    def group(self,*args):raise PermissionError('generated sampler refusal')
    def signal_if_same_process(self,pid,identity,sig):return {'identity_checked':True,'pid':pid,'signal':int(sig)}
def raising_cleanup(*args):raise PermissionError('generated cleanup refusal')
def success_result(phase,subject,case_sha,reader_sha):
    result={'status':'saved_public_domain_geometry_checked_anatomy_unreviewed' if phase=='saved-domain-review' else 'public_crop_source_samples_and_label_grids_converted_anatomy_unreviewed','case_sha256':case_sha,'executing_script_sha256':reader_sha,'patient_id':subject,'role':'TRAIN','public_only':True,'private_reference_loaded':False,'training_admitted':False,'optimizer_updates_performed':0,'full_coverage_factory_compatible':False,'header_snapshot_sha256':'h'*64}
    if phase=='crop-mr-domains':
        result.update(crop_policy='public_source_domain_union_no_extrapolation',support_source_domain_required=True,annotations={k:{'placement':{'source_positive_centres_outside_target_grid':0}} for k in ('cerebrum','whole_tumor')})
    else:result.update(public_input_manifest=None,original_source_reopened=False)
    return result
class Controls(unittest.TestCase):
    def test_retained_child_killed_when_group_and_sampler_fail(self):
        p=Process();actions=[];errors=[]
        with patch.object(M.os,'killpg',side_effect=PermissionError('generated group refusal')):
            remaining=M.cleanup_phase(p,Sampler(),{'123':[1,2]},actions,errors,raising_cleanup)
        self.assertEqual(p.code,-9);self.assertEqual(p.kills,1);self.assertEqual(p.waits,1);self.assertEqual(remaining,[])
        self.assertTrue(any(x.startswith('owned_cleanup_error:') for x in errors))
        self.assertTrue(any(x.startswith('final_owned_sample_error:') for x in errors))
        self.assertTrue(any(a.get('identity_checked') for a in actions))
    def test_unreaped_child_remains_explicit_when_fallback_refuses(self):
        p=Process(refuse=True);errors=[]
        def interrupted(*args):raise InterruptedError('generated cancellation during cleanup')
        with patch.object(M.os,'killpg',side_effect=PermissionError('generated group refusal')):
            remaining=M.cleanup_phase(p,Sampler(),{},[],errors,interrupted)
        self.assertEqual(remaining,[p.pid]);self.assertIsNone(p.poll());self.assertTrue(any(x.startswith('direct_child_wait_error:') for x in errors))
    def test_fast_exit_log_overflow_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'worker.log'
            with p.open('wb') as f:f.truncate(M.CAPS['log_bytes_per_phase']+1)
            size,reason=M.final_log_size(p)
            self.assertEqual(size,M.CAPS['log_bytes_per_phase']+1);self.assertEqual(reason,'final_phase_log_cap')
    def test_phase_binding_and_boundary_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'result.json';subject='ReMIND-002';case='c'*64;reader='d'*64
            for phase in ['crop-mr-domains','saved-domain-review']:
                good=success_result(phase,subject,case,reader);p.write_text(json.dumps(good))
                self.assertIsNone(M.terminal_result(p,phase,subject,case,reader)[2])
                for key,value in [('status','unknown'),('case_sha256','wrong'),('executing_script_sha256','wrong'),('public_only',False),('training_admitted',True),('private_reference_loaded',True),('optimizer_updates_performed',1),('patient_id','ReMIND-067'),('full_coverage_factory_compatible',True)]:
                    bad={**good,key:value};p.write_text(json.dumps(bad))
                    self.assertIsNotNone(M.terminal_result(p,phase,subject,case,reader)[2],(phase,key))
            for raw in ['{','[]','null']:
                p.write_text(raw);self.assertIsNotNone(M.terminal_result(p,'crop-mr-domains',subject,case,reader)[2])
            p.unlink();self.assertIsNotNone(M.terminal_result(p,'crop-mr-domains',subject,case,reader)[2])
    def run_generated_main(self,*,cleanup= lambda *args: [],bad_final=False,large_log=False):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);here=base/'prepared';here.mkdir();reader=base/'src/resectionlab/remind_planning_qc.py';reader.parent.mkdir(parents=True);reader.write_text('# generated only\n')
            reader_sha=hashlib.sha256(reader.read_bytes()).hexdigest();reviewer=here/'audit_saved_domains.py';reviewer.write_text('# generated review only\n');review_sha=hashlib.sha256(reviewer.read_bytes()).hexdigest();bound=[(p,base/(p+'.json'),'c'*64,base/(p+'-headers.json'),'h'*64) for p in M.SUBJECTS]
            def receipt(path,value):path.write_text(json.dumps(value))
            helper=types.ModuleType('run_contact_owned');helper.FastDarwinSampler=Sampler;helper.cleanup_owned=cleanup;helper._receipt=receipt
            calls=[]
            def popen(argv,**kwargs):
                calls.append(argv);phase='saved-domain-review' if '--subject' in argv else 'crop-mr-domains';subject=argv[argv.index('--subject')+1] if '--subject' in argv else Path(argv[argv.index('--case')+1]).stem
                out=Path(argv[argv.index('--output')+1]);out.mkdir();result=success_result(phase,subject,'c'*64,review_sha if phase=='saved-domain-review' else reader_sha)
                if phase=='saved-domain-review':result['conversion_receipt_sha256']=argv[argv.index('--conversion-sha256')+1]
                if bad_final and len(calls)==8:result['status']='unknown'
                (out/('saved-domain-review.json' if phase=='saved-domain-review' else 'conversion-result.json')).write_text(json.dumps(result))
                if large_log:kwargs['stdout'].truncate(M.CAPS['log_bytes_per_phase']+1)
                return Process(code=0)
            argv=['generated','--release',str(base/'false.json'),'--release-sha256','f'*64]
            with patch.object(M,'ROOT',base),patch.object(M,'HERE',here),patch.object(M,'preflight',return_value=({'execution_released':True,'source_index':{},'public_index':{}},bound)),patch.dict(sys.modules,{'run_contact_owned':helper}),patch.object(sys,'argv',argv),patch.object(M.subprocess,'Popen',side_effect=popen),patch.object(M.os,'killpg',side_effect=AssertionError('no real process may be signaled')):
                code=M.main()
            output=here/'actual-domain-qc-v1';result=json.loads((output/'receipt.json').read_text())
            phase_receipts=[json.loads(p.read_text()) for p in output.glob('*-supervisor.json')]
            return code,result,phase_receipts,len(calls)
    def test_eight_bound_successes_complete_generated_batch(self):
        code,result,records,calls=self.run_generated_main()
        self.assertEqual(code,0);self.assertEqual(calls,8);self.assertTrue(all(r['phase_complete'] for r in records))
    def test_unknown_final_json_cannot_pass_eight_exit_zero(self):
        code,result,records,calls=self.run_generated_main(bad_final=True)
        self.assertEqual(code,1);self.assertEqual(calls,8);self.assertIn('phase_success_status',result['fatal']);self.assertFalse(all(r['phase_complete'] for r in records))
    def test_cleanup_exception_keeps_phase_and_failure_receipts(self):
        code,result,records,calls=self.run_generated_main(cleanup=raising_cleanup)
        self.assertEqual(code,1);self.assertEqual(calls,1);self.assertEqual(len(records),1);self.assertTrue(records[0]['reaped']);self.assertTrue(records[0]['cleanup_errors']);self.assertFalse(records[0]['phase_complete'])
    def test_fast_exit_log_failure_keeps_phase_receipt(self):
        code,result,records,calls=self.run_generated_main(large_log=True)
        self.assertEqual(code,1);self.assertEqual(calls,1);self.assertEqual(records[0]['stop_reason'],'final_phase_log_cap');self.assertGreater(records[0]['final_phase_log_bytes'],M.CAPS['log_bytes_per_phase'])
class DomainControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ast
        import numpy as np
        cls.np=np
        source=ast.parse(Path(__file__).with_name('audit_saved_domains.py').read_text())
        namespace={'np':np,'itertools':__import__('itertools')}
        exec(compile(ast.Module(body=[v for v in source.body if isinstance(v,ast.FunctionDef) and v.name in {'domain_plane_counts','source_centres_inside_crop'}],type_ignores=[]),'<generated-only saved reviewer helpers>','exec'),namespace)
        cls.counts=staticmethod(namespace['domain_plane_counts']);cls.coverage=staticmethod(namespace['source_centres_inside_crop'])
    def test_unknown_and_known_zero_remain_separate(self):
        np=self.np; r=self.counts(np.array([1,0,0,0],np.uint8),np.array([1,1,1,0],np.uint8),np.array([1,1,0,0],np.uint8),np.ones(4,np.uint8))
        self.assertEqual((r['target'],r['target_in_known_support_positive'],r['target_in_known_support_zero'],r['target_in_unknown_support_domain']),(3,1,1,1));self.assertEqual(r['target_outside_support'],2)
    def test_positive_labels_outside_source_domains_refused(self):
        np=self.np;a=np.ones(3,np.uint8);z=np.zeros(3,np.uint8)
        with self.assertRaises(AssertionError):self.counts(a,a,z,a)
        with self.assertRaises(AssertionError):self.counts(a,a,a,z)
    def test_nonbinary_or_wrong_dtype_refused(self):
        np=self.np;a=np.ones(3,np.uint8)
        with self.assertRaises(AssertionError):self.counts(a,a.astype(float),a,a)
        with self.assertRaises(AssertionError):self.counts(a,np.array([1,2,0],np.uint8),a,a)
    def test_affine_centre_coverage_has_half_open_boundary(self):
        np=self.np;affine=np.eye(4);g={'shape_xyz':[2,2,2],'affine_xyz_to_ras_mm':affine.tolist()}
        self.assertTrue(self.coverage(g,affine,[2,2,2])['all_source_grid_centres_inside_crop'])
        g['affine_xyz_to_ras_mm'][0][3]=.5
        with self.assertRaises(AssertionError):self.coverage(g,affine,[2,2,2])
if __name__=='__main__':unittest.main()

