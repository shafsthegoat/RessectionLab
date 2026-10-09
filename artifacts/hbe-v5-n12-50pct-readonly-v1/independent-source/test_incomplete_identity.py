"""Independent generated controls through the real old supervisor; no process launches."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts import febio_runtime
from scripts import mechanics_hbe_v5_remaining_one_shot as remaining

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'build/hbe-v5-n12-nocache-50pct-probe-v1/probe.py'
spec = importlib.util.spec_from_file_location('independent_probe50', SOURCE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

class IndependentProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        cls.closure = probe.selected_sources(cls.commit)  # Old20 source-only checks.
        _, cls.original_diagnostic, _ = probe.load_dependencies()  # Load verified small source only.

    def fake_run(self, *, omit_deck=False, breach=False, kill_denied=False):
        states = iter([(50, 1), (50, 1), (44 if breach else 50, 1), (50, 1)])
        class Host:
            def host(self):
                value, mask = next(states)
                return {'available_percent': value, 'kernel_pressure_mask': mask, 'swap_used_bytes': 123}
        class Child:
            pid = 987654321
            running = breach
            kill_calls = 0
            wait_calls = 0
            def poll(self): return None if self.running else 0
            def kill(self):
                self.kill_calls += 1
                self.running = False
            def wait(self, timeout):
                self.wait_calls += 1
                if self.running: raise subprocess.TimeoutExpired('fake', timeout)
                return 0
        child = Child()
        diagnostic = SimpleNamespace(load_host_sampler=Host,
            OwnedWorker=self.original_diagnostic.OwnedWorker, THREAD_ENV={},
            WALL_CAP=150, WORKER_WALL_CAP=147, RSS_CAP=3*1024**3, OUTPUT_CAP=4*1024**2)
        compare = SimpleNamespace(selected_head=lambda value: value)
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'attempt-01'
            def fake_popen(*args, **kwargs):
                self.assertTrue(kwargs['start_new_session'])
                worker = {'status': 'read_only_validation_memory_diagnostic_complete',
                    'source_commit': self.commit, 'native_calls': 0,
                    'release_written': False, 'target_reserved': False}
                identity = {'index': 8, 'run_id': self.closure['run_id'],
                    'source_hashes': self.closure['old_source_hashes'],
                    'adapter_receipt': self.closure['adapter_receipt'],
                    'observed_head_preflight': self.commit}
                if not omit_deck:
                    worker['adapted_deck_sha256'] = self.closure['expected_adapted_deck_sha256']
                    worker['expected_adapted_deck_sha256'] = self.closure['expected_adapted_deck_sha256']
                    identity['deck_sha256'] = self.closure['expected_adapted_deck_sha256']
                hints = {'eligible_files': 14, 'hinted_file_opens': 14, 'eligible_bytes': 2801621755}
                for name, value in [('worker-result.json', worker),
                                    ('validation-identity.json', identity), ('hint-audit.json', hints)]:
                    (target / name).write_text(json.dumps(value))
                return child
            def group_signal(*args):
                if kill_denied: raise PermissionError('generated group denial')
                child.running = False
            def observer(*args, **kwargs):
                return (123, [{'pid': child.pid}]) if child.running else (0, [])
            with patch.object(probe, 'TARGET', target), \
                 patch.object(probe, 'load_dependencies', return_value=(compare, diagnostic, None)), \
                 patch.object(probe, 'selected_sources', return_value=self.closure), \
                 patch.object(self.original_diagnostic.subprocess, 'Popen', side_effect=fake_popen) as launch, \
                 patch.object(remaining.os, 'killpg', side_effect=group_signal), \
                 patch.object(febio_runtime, 'process_group_rss', side_effect=observer), \
                 redirect_stdout(io.StringIO()):
                status = probe.supervisor(self.commit)
                saved = json.loads((target / 'receipt.json').read_text())
                host_series = json.loads((target / 'host-time-series.json').read_text())
                preserved = (target / 'receipt.json').read_bytes()
                with self.assertRaisesRegex(RuntimeError, 'one-use'):
                    probe.supervisor(self.commit)
                self.assertEqual((target / 'receipt.json').read_bytes(), preserved)
                self.assertEqual(launch.call_count, 1)
            return status, saved, host_series, child

    def test_full_generated_success_through_real_supervisor(self):
        code, saved, series, child = self.fake_run()
        self.assertEqual(code, 0)
        self.assertEqual(saved['status'], 'completed_read_only_50pct_probe')
        self.assertEqual(saved['readout_calls_attempted'], 1)
        self.assertEqual(saved['readout_stage']['status'], 'completed_within_caps')
        self.assertEqual(len(series['samples']), 3)
        self.assertGreaterEqual(child.wait_calls, 2)

    def test_missing_actual_and_expected_deck_fail_with_other_checks_valid(self):
        code, saved, series, child = self.fake_run(omit_deck=True)
        self.assertEqual(code, 1)
        self.assertEqual(saved['status'], 'failed_or_incomplete')
        self.assertEqual(saved['readout_stage']['status'], 'completed_within_caps')
        self.assertTrue(saved['owned_worker_cleanup']['contained'])
        self.assertIsNone(series['breach'])

    def test_observed_host_breach_reaches_real_supervisor_and_reaps(self):
        code, saved, series, child = self.fake_run(breach=True)
        self.assertEqual(code, 1)
        self.assertEqual(saved['readout_stage']['kill_reason'], 'supervision_exception')
        self.assertEqual(series['breach']['available_percent'], 44)
        self.assertFalse(child.running)
        self.assertTrue(saved['owned_worker_cleanup']['direct_child_reaped'])

    def test_group_signal_denial_uses_owned_child_and_keeps_failure(self):
        code, saved, series, child = self.fake_run(breach=True, kill_denied=True)
        self.assertEqual(code, 1)
        self.assertEqual(child.kill_calls, 1)
        self.assertFalse(child.running)
        cleanup = saved['owned_worker_cleanup']
        self.assertTrue(cleanup['contained'])
        self.assertTrue(cleanup['direct_child_reaped'])
        self.assertTrue(cleanup['fallback_used'])
        self.assertIn('killpg:PermissionError', cleanup['errors'])
        self.assertEqual(saved['readout_stage']['cleanup_error']['type'], 'PermissionError')

if __name__ == '__main__': unittest.main()
