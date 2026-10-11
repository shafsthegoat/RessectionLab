"""One fixed generated pair. All native previews, including replay, are charged."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(HERE))
from common import guard, sha, write_json, useful_gate, complete_search, TOL


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class Work:
    """Transparent instrumenting wrappers; inclusive timings must not be added."""
    def __init__(self, output, check):
        self.output, self.check = output, check
        self.phase = 'startup'; self.phases = {}; self.active = None; self.originals = []

    def row(self):
        return self.phases.setdefault(self.phase, {'previews_started': 0, 'previews_returned': 0,
            'previews_feasible': 0, 'previews_refused': 0, 'pre_union_refusals': 0, 'preview_exceptions': 0,
            'preview_seconds': 0., 'leaf_tests': 0, 'new_leaf_count': 0, 'witness_rows': 0,
            'peak_tracked_cells': 0, 'budget_refused': 0, 'returned_microsteps': 0,
            'transitions_started': 0, 'transitions_returned': 0, 'transition_seconds': 0.,
            'inventory_requests': 0, 'inventory_seconds': 0., 'cached_inventory_requests': 0,
            'snapshot_verifications': 0, 'snapshot_verification_seconds': 0.,
            'task_integrity_checks': 0, 'task_integrity_seconds': 0.})

    def progress(self):
        write_json(self.output/'progress.json', {'phase': self.phase, 'active_preview': self.active,
            'phases': self.phases, 'scope': 'All started preview work; an interrupted call has unknown additional work, not zero'}, 65536)

    def install(self, engine, task):
        def bind(cls, name, maker):
            old = getattr(cls, name); self.originals.append((cls, name, old)); setattr(cls, name, maker(old))
        def preview(old):
            def wrapped(instance, *args, **kwargs):
                self.check(); row = self.row(); row['previews_started'] += 1
                self.active = {'phase': self.phase, 'ordinal': row['previews_started']}; self.progress(); start = time.perf_counter()
                try:
                    value = old(instance, *args, **kwargs)
                    row['previews_returned'] += 1
                    row['previews_feasible' if value.feasible else 'previews_refused'] += 1
                    row['returned_microsteps'] += len(value.microsteps)
                    attempt = value.cut_union_attempt
                    if attempt is None: raise ValueError('Missing declared cut-union preview accounting')
                    # The pinned engine creates coverage before finish() for
                    # every returned preview. Only explicit zero sidecars and
                    # an empty traversal support zero-work early-refusal counts.
                    if (not value.feasible and not value.microsteps
                            and all(attempt[k] == 0 for k in ('leaf_tests', 'new_leaf_count', 'witness_rows'))):
                        row['pre_union_refusals'] += 1
                    for key in ('leaf_tests', 'new_leaf_count', 'witness_rows', 'budget_refused'):
                        row[key] += int(attempt[key])
                    row['peak_tracked_cells'] = max(row['peak_tracked_cells'], int(attempt['peak_tracked_cells']))
                    return value
                except BaseException:
                    row['preview_exceptions'] += 1; raise
                finally:
                    row['preview_seconds'] += time.perf_counter()-start; self.active = None; self.progress()
                    self.check()
            return wrapped
        def timed(old, count, seconds, returned=None, cache=False):
            def wrapped(instance, *args, **kwargs):
                self.check(); row = self.row(); row[count] += 1; start = time.perf_counter()
                if cache: row['cached_inventory_requests'] += int(instance._inventory is not None)
                try:
                    value = old(instance, *args, **kwargs)
                    if returned: row[returned] += 1
                    return value
                finally: row[seconds] += time.perf_counter()-start
            return wrapped
        bind(engine, 'preview_stroke', preview)
        bind(task, '_transition', lambda old: timed(old, 'transitions_started', 'transition_seconds', 'transitions_returned'))
        bind(task, '_prepare_inventory', lambda old: timed(old, 'inventory_requests', 'inventory_seconds', cache=True))
        bind(engine, 'committed_snapshot_identity', lambda old: timed(old, 'snapshot_verifications', 'snapshot_verification_seconds'))
        bind(task, '_assert_frozen', lambda old: timed(old, 'task_integrity_checks', 'task_integrity_seconds'))

    def restore(self):
        for cls, name, old in reversed(self.originals): setattr(cls, name, old)

    @contextmanager
    def at(self, name):
        previous = self.phase; self.phase = name; self.row(); self.progress()
        try: yield
        finally: self.progress(); self.phase = previous


def physical_from_audit(audit):
    if not audit.feasible or audit.certified_evacuated_geometry is None:
        raise ValueError('Independent declared-model geometry must pass')
    record = audit.to_dict()['evacuated_volume']
    lo, hi = record['evacuated_target_source_bounds_mm3']
    # In this fixture every parent target mass is exactly zero or one. Thus
    # the independent unlocated-mass interval collapses for any checked subset.
    if abs(hi-lo) > TOL: raise ValueError('Generated binary target interval did not collapse')
    return {'source_mm3': record['evacuated_source_volume_mm3'], 'target_mm3': lo,
            'outside_supplied_target_mm3': record['evacuated_source_volume_mm3']-lo,
            'geometry_digest': record['geometry_digest'], 'certified_leaf_count': record['certified_leaf_count'],
            'scope': 'Exact binary generated source support and public nominal target; outside supplied target is not clinical healthy-tissue harm'}


def preparation(history, final_outside):
    previous_outside = 0.
    for row in history:
        if row['target_removed_mm3'] > 0: return previous_outside
        volume = row.get('evacuated_volume')
        if volume:
            after = volume['after']; lo, hi = after['evacuated_target_source_bounds_mm3']
            if abs(hi-lo) > TOL: raise ValueError('Uncertain generated prefix target allocation')
            previous_outside = after['evacuated_source_volume_mm3']-lo
    return final_outside


def route_readout(task, actions, work, label, check):
    with work.at(label+'.replay'):
        for action in actions:
            check(); task.step(action)
        metrics = task.metrics()
    with work.at(label+'.independent_audit'):
        start = time.perf_counter(); check(); audit = task.independent_geometry_check(); check()
        audit_seconds = time.perf_counter()-start
    physical = physical_from_audit(audit)
    history = metrics.pop('history')
    return {'terminated': task.terminated, 'metrics': metrics, 'history': history,
            'audit': audit.to_dict(), 'audit_seconds': audit_seconds, 'physical': physical,
            'preparation_outside_mm3': preparation(history, physical['outside_supplied_target_mm3']),
            'chosen_nonstop_actions': sum(a != 'STOP' for a in actions), 'action_ids': list(actions)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--source-index-sha256', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    index_path = HERE/'source-index.json'
    if sha(index_path) != args.source_index_sha256: raise ValueError('Source index mismatch')
    index = json.loads(index_path.read_bytes()); guard(ROOT, index)
    config = json.loads((HERE/'config.json').read_bytes()); output = Path(args.output).resolve()
    if not index['execution_released'] or output != ROOT/config['output_directory'] or not output.is_dir():
        raise ValueError('Exact released output required')
    if not sys.dont_write_bytecode or not sys.pycache_prefix or Path(sys.pycache_prefix).exists():
        raise ValueError('Fresh absent bytecode prefix required')
    started = time.perf_counter(); deadline = started+config['caps']['work_seconds']; arm_deadline = [None]
    lease_fd = int(os.environ['RESECTIONLAB_PARENT_LEASE_FD'])
    def lease():
        os.read(lease_fd, 1); os._exit(72)
    threading.Thread(target=lease, daemon=True).start()
    def check():
        now = time.perf_counter()
        if now >= deadline or (arm_deadline[0] is not None and now >= arm_deadline[0]):
            raise TimeoutError('Declared work/arm elapsed limit')
    def boundary(event, values):
        if event.startswith(('socket.', 'subprocess.')) or event in ('os.fork', 'os.forkpty', 'os.posix_spawn', 'os.system'):
            raise PermissionError('No network, subprocess, model worker, or acquisition')
        if event == 'open' and isinstance(values[0], (str, bytes)):
            path = Path(os.fsdecode(values[0])).absolute()
            if path.is_relative_to(ROOT):
                if any(part in {'data', 'outputs', 'checkpoints', 'sources'} for part in path.relative_to(ROOT).parts):
                    raise PermissionError('No scientific source/archive/checkpoint read')
    sys.addaudithook(boundary)
    sys.path.insert(0, str(ROOT/'src'))
    import numpy as np
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.native_spatial_task import NativeSpatialTask, make_native_opening_task
    from resectionlab.native_cut_union import CutUnionSpec
    from resectionlab.observed_search import observed_beam_search
    work = Work(output, check); work.install(NativeResectionEngine, NativeSpatialTask)
    report = {'schema': config['schema'], 'status': 'running', 'config': config, 'arms': [], 'positive_control': {},
              'source_index_sha256': args.source_index_sha256, 'source_unchanged': False,
              'timing_scope': 'Inclusive timed scopes overlap; do not sum nested inventory/preview/integrity durations',
              'runtime': {'python': sys.version, 'numpy': np.__version__}}
    write_json(output/'report.json', report)
    try:
        for mode in config['arms']:
            arm = {'retention_mode': mode, 'status': 'running'}; report['arms'].append(arm)
            arm_start = time.perf_counter(); arm_deadline[0] = arm_start+config['search']['seconds']
            with work.at(mode+'.initial_inventory'):
                task = make_native_opening_task(max_steps=config['search']['max_steps'],
                    generated_cut_union=CutUnionSpec(**config['cut_union']), cancelled=lambda: time.perf_counter() >= arm_deadline[0])
                inventory = task.candidate_inventory(); initial = task.metrics()
                if not inventory['complete'] or inventory['omitted_count'] != 0: raise ValueError('Incomplete initial inventory')
                arm['identity'] = {'source_hash': initial['source_hash'], 'decision_model_hash': initial['decision_model_hash'],
                    'initial_inventory_sha256': digest(inventory), 'reward': asdict(task.reward_spec),
                    'model': task.case.generated_cut_union.to_record(),
                    'volume': task.case.generated_evacuated_volume.to_record()}
                arm['initial_inventory'] = inventory
            try:
                with work.at(mode+'.search'):
                    plan, search = observed_beam_search(task, max_calls=200, beam_width=2, seconds=120.,
                        transition_mode='lazy_planning', retention_mode=mode)
                    arm['search'] = search; arm['action_ids'] = list(plan)
                    if not complete_search(search): raise ValueError('Truncated/incomplete search is not a comparable outcome')
                with work.at(mode+'.replay_setup'): replay = task.fresh()
                arm.update(route_readout(replay, plan, work, mode, check))
                if not arm['terminated'] or abs(arm['metrics']['total_reward']-search['estimated_incremental_return']) > TOL:
                    raise ValueError('Replay return/termination disagrees with sealed chosen plan')
                phase = work.phases[mode+'.search']
                if phase['transitions_started'] != search['model_transition_calls']:
                    raise ValueError('Search transition accounting differs from actual invocation counter')
                arm['status'] = 'completed'
            except BaseException as error:
                if hasattr(error, 'accounting'): arm['search'] = error.accounting
                arm['failure'] = {'exception': type(error).__name__, 'message': str(error)}
                raise
            finally:
                arm['total_arm_seconds'] = time.perf_counter()-arm_start; arm_deadline[0] = None
                write_json(output/'report.json', report)
        # One known geometric positive route, not a third search or variant.
        with work.at('positive_control'):
            task = make_native_opening_task(max_steps=8, generated_cut_union=CutUnionSpec(**config['cut_union']),
                                           cancelled=lambda: time.perf_counter() >= deadline)
            selected = []
            for request in config['positive_control']:
                if request == 'STOP': action = 'STOP'
                else:
                    inventory = task.candidate_inventory()
                    matches = [row['action_id'] for row in inventory['ledger'] if row['feasible'] and
                               row['tool_id'] == request['tool_id'] and row['voxel'] == request['voxel']]
                    if len(matches) != 1: raise ValueError('Fixed positive control action unavailable or ambiguous')
                    action = matches[0]
                task.step(action); selected.append(action)
            start = time.perf_counter(); audit = task.independent_geometry_check(); check(); metrics = task.metrics()
            report['positive_control'] = {'action_ids': selected, 'audit': audit.to_dict(),
                'audit_seconds': time.perf_counter()-start, 'physical': physical_from_audit(audit),
                'target_removed_mm3': metrics['target_removed_mm3'], 'total_reward': metrics['total_reward'],
                'passed': audit.feasible and task.terminated and metrics['target_removed_mm3'] == 2. and metrics['total_reward'] > 0.}
        report['useful_criterion'] = useful_gate(report['arms'], report['positive_control'])
        if not report['useful_criterion']['comparison_admissible']: raise ValueError('Comparison admission failed')
        guard(ROOT, index); report['source_unchanged'] = True; report['status'] = 'completed'
    except BaseException as error:
        report['status'] = 'incomplete_or_failed'; report['failure'] = {'exception': type(error).__name__, 'message': str(error)}
        report['useful_criterion'] = useful_gate(report['arms'], report['positive_control'])
    finally:
        work.restore(); report['work_by_phase'] = work.phases
        report['all_preview_calls_started'] = sum(row['previews_started'] for row in work.phases.values())
        report['all_preview_calls_returned'] = sum(row['previews_returned'] for row in work.phases.values())
        report['all_task_transitions_started'] = sum(row['transitions_started'] for row in work.phases.values())
        report['worker_seconds'] = time.perf_counter()-started
        report['self_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['no_actor_or_optimizer_calls'] = True
        write_json(output/'report.json', report); work.progress()
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__': raise SystemExit(main())
