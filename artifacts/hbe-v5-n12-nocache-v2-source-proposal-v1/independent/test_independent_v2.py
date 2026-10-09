"""Independent generated namespace and policy-consumption checks; no actual execution."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from launchers import hbe_v5_nocache_v1 as v1
from scripts import mechanics_hbe_v5_remaining_one_shot as remaining

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'build/hbe-v5-n12-nocache-v2-candidate/hbe_v5_nocache_v2.py'
spec=importlib.util.spec_from_file_location('independent_nocache_v2',SOURCE)
v2=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

class IndependentV2(unittest.TestCase):
    def test_v2_rejects_every_v1_identity_alias_before_git_or_inner_read(self):
        base={'schema':'hbe-v5-nocache-launch-envelope-v2',
              'status':'root_released_one_native_call_with_declared_io_policy_v2',
              'source_commit':'a'*40,'extension_source_bindings':{name:'0'*64 for name in v2.SOURCES},
              'inner_release':{'path':'build/absent-inner.json','sha256':'1'*64},
              'sidecar_directory':v2.SIDECAR,'policy':copy.deepcopy(v2.POLICY)}
        aliases={
          'schema':'hbe-v5-nocache-launch-envelope-v1',
          'status':'root_released_one_native_call_with_declared_io_policy',
          'sidecar_directory':v1.SIDECAR,
          'policy':copy.deepcopy(v1.POLICY),
          'extension_source_bindings':{name:'0'*64 for name in v1.SOURCES},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()
            envelope_path=root/'envelope.json'
            for key,value in aliases.items():
                with self.subTest(key=key):
                    envelope=copy.deepcopy(base)
                    envelope[key]=value
                    envelope_path.write_text(json.dumps(envelope))
                    with patch.object(v2,'__file__',str(root/v2.SOURCES[0])), \
                         patch.object(v2,'_git',side_effect=AssertionError('Git must not be reached')):
                        with self.assertRaises(ValueError): v2._read_exact_release(root,envelope_path)
            self.assertFalse((root/'build/absent-inner.json').exists())

    def test_v1_rejects_v2_envelope_before_git_or_inner_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()
            envelope_path=root/'envelope.json'
            envelope={'schema':'hbe-v5-nocache-launch-envelope-v2',
              'status':'root_released_one_native_call_with_declared_io_policy_v2',
              'source_commit':'a'*40,'extension_source_bindings':{name:'0'*64 for name in v2.SOURCES},
              'inner_release':{'path':'build/absent-inner.json','sha256':'1'*64},
              'sidecar_directory':v2.SIDECAR,'policy':v2.POLICY}
            envelope_path.write_text(json.dumps(envelope))
            with patch.object(v1,'__file__',str(root/v1.SOURCES[0])), \
                 patch.object(v1,'_git',side_effect=AssertionError('Git must not be reached')):
                with self.assertRaises(ValueError): v1._read_exact_release(root,envelope_path)

    def _guard_prior_policy(self, *, symlink=False):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()
            old=root/v1.SIDECAR
            old.parent.mkdir(parents=True)
            if symlink: old.symlink_to(root/'nonexistent-policy-target',target_is_directory=True)
            else:
                old.mkdir()
                (old/'receipt.json').write_text('{"status":"failed_or_incomplete","native_calls_attempted":0}\n')
            bindings={name:'0'*64 for name in v2.SOURCES}
            decoded=({'source_commit':'a'*40,'extension_source_bindings':bindings},
                     b'outer',(1,2),{},root/'inner',b'inner',(3,4))
            host=SimpleNamespace(FastDarwinSampler=lambda:SimpleNamespace(host=lambda:{
                'available_percent':54,'kernel_pressure_mask':1}))
            with patch.object(v2,'_read_exact_release',return_value=decoded), \
                 patch.object(v2,'_load_sibling',side_effect=[SimpleNamespace(),host]), \
                 patch.object(remaining,'execute',side_effect=AssertionError('Old runner must not be reached')) as old_runner:
                with self.assertRaises(ValueError):
                    v2.execute_envelope(root/'outer',root=root)
                old_runner.assert_not_called()
            self.assertFalse((root/v2.SIDECAR).exists())
            self.assertFalse((root/v2.NATIVE_OUTPUT).exists())
            if not symlink:
                self.assertEqual((old/'receipt.json').read_text(),'{"status":"failed_or_incomplete","native_calls_attempted":0}\n')

    def test_prior_failed_v1_sidecar_refuses_without_v2_reservation(self):
        self._guard_prior_policy()

    def test_prior_dangling_v1_sidecar_refuses_without_v2_reservation(self):
        self._guard_prior_policy(symlink=True)

if __name__=='__main__': unittest.main()
