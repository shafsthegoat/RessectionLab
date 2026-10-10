"""Unrun stdlib glue control; physical cache correctness has its existing suite.

Executes the exact nested run_arm source against generated boundary doubles.
No patient arrays, native geometry, numerical modules or model are used.
"""
import ast
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest
import weakref

HERE = Path(__file__).resolve().parent


class PairGlue(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        output = Path(self.temp.name)
        self.original = lambda *args: None
        native = SimpleNamespace(capsule_voxel_indices=self.original)
        self.calls = []; self.budgets = []; self.fail_cached = False
        result = {'arms': [], 'source_visits': 0, 'greedy_calls': 0, 'committed_actions': 0}
        metrics = {'steps': 2, 'terminated': True, 'history': ({'action_id': 'MOVE', 'reward': 1.}, {'action_id': 'STOP', 'reward': 0.})}
        plain = lambda v: json.loads(json.dumps(v))
        accounting = {'complete': True, 'planning_seconds': 1., 'scores': (('MOVE', 1.), ('STOP', 0.))}
        prior_plan = {'actions': ['MOVE', 'STOP'], 'accounting': plain(accounting)}
        case = self

        def need(condition, reason):
            if not condition: raise ValueError(reason)

        class Budget:
            def __init__(self, engine, *, max_native_previews, seconds):
                self.limit = max_native_previews; self.entries = 0; case.budgets.append(self)
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def check(self): need(self.entries <= self.limit, 'preview_cap')
            def snapshot(self): return {'native_preview_entries': self.entries}
            def complete(self, *, history_complete): need(history_complete, 'history_complete'); self.check()
            @contextmanager
            def phase(self, name): yield

        class Source:
            source_hash = 'source'

        class Task:
            def __init__(self, source): self._config = source; self.steps = 0; self.terminated = False
            decision_model_hash = 'model'
            def observation(self):
                return SimpleNamespace(fingerprint='observation', action_ids=('STOP', 'MOVE'),
                    action_mask=SimpleNamespace(tolist=lambda: [True, True]))
            def observed_greedy_search(self, *, seconds):
                cached = native.capsule_voxel_indices is not case.original
                case.calls.append(('greedy', cached)); case.budgets[-1].entries += 2
                if cached and case.fail_cached: raise RuntimeError('generated cached failure')
                return ('MOVE', 'STOP'), accounting
            def step(self, action):
                need(native.capsule_voxel_indices is case.original, 'cached_authoritative_replay')
                case.calls.append(('step', action)); self.steps += 1
                case.budgets[-1].entries += 1; self.terminated = action == 'STOP'
            def metrics(self): return metrics

        class Cache:
            def __init__(self, config, **kwargs): self.source = config
            def query(self, *args): pass
            def stats(self): return {'calls': 2, 'hits': 1, 'misses': 1}

        @contextmanager
        def inject(cache):
            need(native.capsule_voxel_indices is case.original, 'cache_not_restored')
            native.capsule_voxel_indices = cache.query
            try: yield
            finally: native.capsule_voxel_indices = case.original

        def prepare(*args, **kwargs):
            need(native.capsule_voxel_indices is case.original, 'cached_construction')
            case.calls.append(('construct',)); case.budgets[-1].entries += 2
            return Source(), {}, {}, {}

        def evaluate(task, **kwargs):
            need(native.capsule_voxel_indices is case.original, 'cached_independent_audit')
            need(task.terminated and task.steps == 2, 'incomplete_audit')
            case.calls.append(('audit',)); return {'accepted': True, 'outcomes': {'return': 1.}}

        def write(path, value, **kwargs):
            with path.open('x') as stream: json.dump(value, stream)

        self.env = dict(check=lambda: None, OUTPUT=output, result=result,
            CAPS={'native_previews': 12}, deadline=time.perf_counter()+30,
            resource=SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=0)),
            time=time, weakref=weakref, gc=gc, PlanningBudget=Budget, NativeResectionEngine=object,
            prepare_public_source=prepare, limits={}, INPUTS={'original_release': {'sha256': 'release'}},
            entry={'path': 'generated.json', 'sha256': 'generated'}, cohort_bytes=b'generated',
            semantic_digest=lambda value: 'digest', qualification={}, NominalCavityProposalConfig=lambda **kw: kw,
            original={'proposal_config': {}, 'occupancy_condition': 'generated', 'post_exposure_condition': 'generated'},
            TARGET_CONTEXT='generated', make_patient_planning_task=lambda source, **kw: (Task(source), SimpleNamespace(fingerprint='context')),
            prior={'source_hash': 'source', 'context_hash': 'context', 'initial_observation_hash': 'observation'},
            ExactCapsuleCoverCache=Cache, CACHE_LIMITS={}, inject_native_capsule_cache=inject,
            native_resection=native, original_cover=self.original, need=need, plain=plain,
            sink=SimpleNamespace(write=write), sha=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
            plan=prior_plan, prior_metrics=plain(metrics), prior_audit={'outcomes': {'return': 1.}},
            evaluate_native_spatial_episode=evaluate)
        tree = ast.parse((HERE/'initial_inventory_worker.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        arm = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == 'run_arm')
        exec(compile(ast.Module(body=[arm], type_ignores=[]), str(HERE/'initial_inventory_worker.py'), 'exec'), self.env)

    def test_two_fresh_arms_normalize_all_metrics_restore_before_replay_and_share_cap(self):
        for name in ('baseline', 'cached'):
            reference = self.env['run_arm'](name); gc.collect()
            self.assertIsNone(reference())
        arms = self.env['result']['arms']
        self.assertEqual([b.limit for b in self.budgets], [12, 6])
        self.assertEqual([r['native_previews'] for r in arms], [6, 6])
        self.assertEqual([r['status'] for r in arms], ['complete', 'complete'])
        self.assertEqual([c for c in self.calls if c[0] == 'greedy'], [('greedy', False), ('greedy', True)])
        self.assertEqual(self.env['result']['source_visits'], 2)
        self.assertEqual(self.env['result']['committed_actions'], 4)
        self.assertEqual(arms[0]['world'], arms[1]['world'])
        self.assertTrue(all(r['exact_original_metrics'] and r['cache_restored_before_replay'] for r in arms))
        self.assertEqual(arms[0]['files']['episode-metrics'], arms[1]['files']['episode-metrics'])

    def test_changed_full_metrics_refuse_and_preserve_failed_arm(self):
        self.env['prior_metrics']['history'][0]['reward'] = 2.
        with self.assertRaisesRegex(ValueError, 'exact_complete_serialized_metrics'):
            self.env['run_arm']('baseline')
        self.assertEqual(self.env['result']['arms'][0]['status'], 'failed_or_unresolved')
        self.assertTrue((self.env['OUTPUT']/'baseline/arm-result.json').is_file())

    def test_cached_failure_restores_before_propagation(self):
        self.fail_cached = True
        with self.assertRaisesRegex(RuntimeError, 'generated cached failure'):
            self.env['run_arm']('cached')
        self.assertIs(self.env['native_resection'].capsule_voxel_indices, self.original)
        self.assertFalse(any(c[0] in ('step', 'audit') for c in self.calls))
        self.assertEqual(self.env['result']['arms'][0]['status'], 'failed_or_unresolved')


if __name__ == '__main__':
    unittest.main(verbosity=2)
