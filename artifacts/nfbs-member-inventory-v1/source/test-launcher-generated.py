"""Only generated tiny processes; an audit hook refuses every NFBS archive open."""
from pathlib import Path
import contextlib, hashlib, importlib.util, io, json, os, sys, tempfile, time, unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PREP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts import mechanics_hbe_v5_remaining_one_shot as supervisor
from scripts import febio_runtime

archive_open_attempts = []
def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        if str(args[0]).lower().endswith('nfbs_dataset.tar.gz'):
            archive_open_attempts.append(str(args[0]))
            raise AssertionError('NFBS archive open forbidden in generated process controls')
sys.addaudithook(guard)
spec = importlib.util.spec_from_file_location('launch', PREP / 'launch.py')
launch = importlib.util.module_from_spec(spec); spec.loader.exec_module(launch)
TEMPLATE = PREP / 'authorized-template.json'
TEMPLATE_SHA = hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
CONTROL_ROOT = Path(tempfile.mkdtemp(prefix='generated-process-controls-', dir=PREP))

class Controls(unittest.TestCase):
    def stage(self, name, code, *, wall=3, rss=512*1024**2, output=1024**2, observer=None):
        folder = CONTROL_ROOT / name; folder.mkdir()
        receipt = {'schema':'generated-process-control-v1', 'measured_data_used':False}
        environment = febio_runtime.private_environment({'caps': {'thread_environment': supervisor.io.THREAD_ENV}})
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        result = supervisor.supervise_stage('readout', [sys.executable, '-B', '-c', code], folder,
            receipt, cwd=folder, environment=environment, wall_cap=wall, rss_cap=rss,
            output_cap=output, rss_observer=observer or febio_runtime.process_group_rss)
        self.assertEqual(archive_open_attempts, [])
        if result.get('pid'):
            _, members = febio_runtime.process_group_rss(result['pid'], timeout_seconds=1)
            self.assertEqual(members, [])
        return result

    def test_completed(self):
        result = self.stage('completed', 'import time; print("generated-only"); time.sleep(.08)')
        self.assertEqual(result['status'], 'completed_within_caps')
        self.assertIsNone(result['kill_reason'])

    def test_wall(self):
        result = self.stage('wall', 'import time; time.sleep(3)', wall=.25)
        self.assertEqual(result['kill_reason'], 'wall_cap')

    def test_rss(self):
        result = self.stage('rss', 'import time; time.sleep(3)', rss=1)
        self.assertEqual(result['kill_reason'], 'process_group_rss_cap')

    def test_output(self):
        result = self.stage('output', 'from pathlib import Path; import time; Path("generated.bin").write_bytes(b"x"*16384); time.sleep(3)', output=8192)
        self.assertEqual(result['kill_reason'], 'active_output_cap')

    def test_observer_refusal(self):
        def refusing(*args, **kwargs):
            raise RuntimeError('generated observer failure')
        result = self.stage('observer', 'import time; time.sleep(3)', observer=refusing)
        self.assertEqual(result['kill_reason'], 'supervision_exception')

    def test_eperm_direct_child_cleanup(self):
        folder = CONTROL_ROOT / 'eperm'; folder.mkdir()
        receipt = {'schema':'generated-eperm-control-v1', 'measured_data_used':False}
        owner = launch.OwnedWorker()
        started = time.monotonic()
        with patch.object(supervisor.os, 'killpg', side_effect=PermissionError('generated EPERM')):
            result = supervisor.supervise_stage('readout',
                [sys.executable, '-B', '-c', 'import time; time.sleep(3)'], folder, receipt,
                cwd=folder, environment=dict(os.environ), wall_cap=1, rss_cap=1, output_cap=1024**2,
                rss_observer=febio_runtime.process_group_rss, popen=owner)
            cleanup = owner.cleanup(febio_runtime.process_group_rss, started+4)
        self.assertEqual(result['cleanup_error']['type'], 'PermissionError')
        self.assertTrue(cleanup['fallback_used'])
        self.assertTrue(cleanup['direct_child_reaped'])
        self.assertTrue(cleanup['contained'])
        self.assertEqual(cleanup['remaining_members'], [])
        self.assertIsNotNone(owner.process.poll())
        self.assertEqual(archive_open_attempts, [])
        receipt['launcher_cleanup'] = cleanup
        receipt['elapsed_seconds'] = time.monotonic()-started
        receipt['status'] = 'failed_stage_but_contained_and_reaped'
        supervisor.io.durable_json(folder/'receipt.json', receipt)

    def test_launcher_check_only(self):
        args = ['launch', '--release', str(TEMPLATE), '--release-sha256', TEMPLATE_SHA, '--check-only']
        out = io.StringIO()
        with patch.object(sys, 'argv', args), contextlib.redirect_stdout(out):
            launch.main()
        record = json.loads(out.getvalue())
        self.assertEqual(record['status'], 'launcher_bindings_valid')
        self.assertIs(record['execution_released'], False)
        self.assertEqual(archive_open_attempts, [])

    def test_launcher_false_release_refused(self):
        args = ['launch', '--release', str(TEMPLATE), '--release-sha256', TEMPLATE_SHA]
        with patch.object(sys, 'argv', args), self.assertRaisesRegex(RuntimeError, 'root_release_required'):
            launch.main()
        self.assertEqual(archive_open_attempts, [])

    def test_prepared_protocol_preserved(self):
        self.assertFalse(json.loads((ROOT / launch.PREPARED).read_bytes())['execution_released'])
        self.assertEqual(hashlib.sha256((ROOT / launch.PREPARED).read_bytes()).hexdigest(), launch.PREPARED_SHA)
        self.assertFalse((ROOT / launch.OUTPUT).exists())
        self.assertEqual(archive_open_attempts, [])

if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    (CONTROL_ROOT / 'summary.json').write_text(json.dumps({'tests':result.testsRun,
        'failures':len(result.failures), 'errors':len(result.errors), 'NFBS archive_open_attempts':archive_open_attempts,
        'scientific_file_execution':False}, sort_keys=True, indent=2)+'\n')
    print(CONTROL_ROOT)
    raise SystemExit(not result.wasSuccessful())
