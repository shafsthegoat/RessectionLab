"""Exercise the new shim around the actual old supervisor with fake child only."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import mechanics_hbe_v5_remaining_one_shot as old
from scripts import febio_runtime

SOURCE=Path(__file__).resolve().parents[1]/'hbe-v5-n12-nocache-comparison-v1/adoption_candidate/hbe_v5_nocache_v1.py'
spec=importlib.util.spec_from_file_location('independent_candidate',SOURCE)
candidate=importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)

class OwnedIntegration(unittest.TestCase):
    def test_real_old_supervisor_permission_failure_is_contained_and_policy_fails(self):
        class Child:
            pid=123456789
            running=True
            kill_calls=wait_calls=0
            def poll(self): return None if self.running else 0
            def kill(self): self.kill_calls+=1; self.running=False
            def wait(self,timeout):
                self.wait_calls+=1
                if self.running: raise AssertionError('uncontained child')
                return 0
        child=Child()
        captured=[]
        def source_chain(*args,**kwargs): return {}
        def source_validate(*args,**kwargs): return old.validate_prior_chain([],8)
        def refusal(*args,**kwargs): raise RuntimeError('generated old observer refusal')
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            def execute():
                old.validate_release({})
                return old.supervise_stage('native',['generated-only'],directory,{},
                    cwd=directory,environment={},wall_cap=5,rss_cap=1024**2,output_cap=1024**2,
                    rss_observer=refusal)
            original_supervise=old.supervise_stage
            with patch.object(old,'validate_prior_chain',source_chain), \
                 patch.object(old,'validate_release',source_validate), \
                 patch.dict(candidate.POLICY,expected_preflight_hint_opens=0,expected_preflight_hint_bytes=0), \
                 patch.object(candidate.subprocess,'Popen',return_value=child), \
                 patch.object(candidate.os,'killpg',side_effect=PermissionError('generated group refusal')), \
                 patch.object(candidate.os,'kill') as cached_pid_signal, \
                 patch.object(febio_runtime,'process_group_rss',return_value=(0,[])):
                with self.assertRaisesRegex(RuntimeError,'child cleanup incomplete'):
                    candidate.scoped_execute(old,None,[],execute,lambda *_:None,
                        native_stage_check=lambda:None,
                        stage_cleanup_callback=lambda stage,record:captured.append((stage,record)))
                cached_pid_signal.assert_not_called()
            self.assertIs(old.supervise_stage,original_supervise)
        self.assertFalse(child.running)
        self.assertEqual(child.kill_calls,1)
        self.assertEqual(child.wait_calls,1)
        self.assertEqual(captured[0][0],'native')
        record=captured[0][1]
        self.assertTrue(record['contained'] and record['direct_child_reaped'])
        self.assertTrue(record['fallback_used'])
        self.assertIn('killpg:PermissionError',record['errors'])

if __name__=='__main__': unittest.main()
