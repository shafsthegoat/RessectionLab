"""Tiny control exposing why the extension needs its own child handle."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import mechanics_hbe_v5_remaining_one_shot as remaining

class InheritedCleanupControl(unittest.TestCase):
    def test_permission_failure_skips_old_reap_and_needs_extension_fallback(self):
        class FakeProcess:
            pid=123456789
            running=True
            waits=0
            def poll(self): return None if self.running else 0
            def wait(self,timeout): self.waits+=1; return 0
        child=FakeProcess()
        def observer(*args,**kwargs): raise RuntimeError('generated observer refusal')
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            receipt={}
            with patch.object(remaining.os,'killpg',side_effect=PermissionError('generated group refusal')):
                result=remaining.supervise_stage('native',['generated-no-process'],directory,receipt,
                    cwd=directory,environment={},wall_cap=5,rss_cap=1024**2,output_cap=1024**2,
                    rss_observer=observer,popen=lambda *args,**kwargs:child)
            self.assertEqual(result['status'],'failed_or_incomplete')
            self.assertEqual(result['cleanup_error']['type'],'PermissionError')
            self.assertTrue(child.running)
            self.assertEqual(child.waits,0)

if __name__=='__main__': unittest.main()
