"""Prospective sequential four-TRAIN runner; an owned hard supervisor is required.

One live patient source and one complete trace at a time; one action autograd
graph at a time. Reconstruct/replay costs are measured, not hidden. No SELECT or
EVAL factory, private labels, patient adaptation or launch entry point exists.
"""
from contextlib import contextmanager
import gc
from pathlib import Path
import time
import weakref

import torch

from .core import freeze_json, semantic_digest, thaw_json
from .native_resection import NativeResectionEngine
from .native_proposals import NominalCavityProposalConfig
from .patient_planning_admission import PatientPlanningContext
from .patient_planning_accumulation import PatientGradientAccumulator
from .patient_planning_learning import PatientTrainSession, common_patient_policies
from .patient_planning_cohort_io import OutputBudget, _save_checkpoint
from .patient_planning_cohort_spec import (VERSION, TRAIN, CLOSED,
    validate_limits, validate_factories, preview_budget_sizing)
from . import patient_planning_preflight as preflight
from . import public_patient_factory as public_factory
from .planning_budget import PlanningBudget
from .spatial_policy import parameter_hash


@contextmanager
def _bounded_writes(sink):
    prior_replay, prior_source = preflight._write, public_factory.write
    preflight._write = public_factory.write = sink.write
    try: yield
    finally:
        preflight._write, public_factory.write = prior_replay, prior_source


def _require_source(base, context, subject, protocol, limits):
    if type(context) is not PatientPlanningContext:
        raise TypeError('Exact patient context required')
    context.require_training(); context.require_task(base); record = context.record()
    expected = NominalCavityProposalConfig(**thaw_json(protocol['cohort_execution']['proposal_config']))
    if (record['subject'] != subject or record['role'] != 'TRAIN'
            or record['learning_protocol_hash'] != semantic_digest(protocol)
            or record['max_optimizer_updates'] != limits['max_optimizer_updates']
            or record['max_steps'] != limits['max_steps']
            or record.get('public_target_context_variant') != protocol['public_target_context_variant']
            or any(record['budgets'][k] != limits[k] for k in record['budgets'])
            or base.case.proposal_mode != 'nominal_cavity_v1'
            or base.case.proposal_config.fingerprint != expected.fingerprint
            or base.terminated or base.observation().state_features[0] != 0
            or base.metrics()['planning_estimator_only'] is not False):
        raise ValueError('Reconstructed TRAIN source differs from the explicit run protocol')


class _SequentialVisits:
    """Keep source lifetime inside a callback; reject cached case retention.

Weak-reference closure is a bounded local ownership check, not a measured RSS
claim. Native search may hold several states of this same one source. Complete
trace observations are released when the callback returns. Errors stop the run.
"""
    def __init__(self, factories, protocol, limits, sink, costs, guard):
        self.factories, self.protocol, self.limits = factories, protocol, limits
        self.sink, self.costs, self.guard = sink, costs, guard
        self.contexts = {}; self.completed = []; self.active = False

    def run(self, subject, destination, phase, callback):
        if self.active: raise ValueError('Patient source visits cannot overlap')
        self.active = True
        base = None
        try:
            self.guard()
            source_dir = destination/'source'; source_dir.mkdir(exist_ok=False)
            with self.costs.scope(phase+'.public_reconstruction'):
                base, context = self.factories[subject](output=source_dir)
                _require_source(base, context, subject, self.protocol, self.limits)
                if subject in self.contexts:
                    if context.fingerprint != self.contexts[subject].fingerprint:
                        raise ValueError('Reconstructed context changed between patient visits')
                else:
                    self.contexts[subject] = context
                    self.sink.write(self.sink.root/(subject+'-context.json'), context.record())
                case_ref = weakref.ref(base.case)
                self.guard()
            with self.costs.scope(phase):
                # JSON-only return prevents accidentally retaining a trace,
                # graph or source in the cross-patient result map.
                result = thaw_json(freeze_json(callback(base, context)))
            del base
            base = None
            gc.collect()
            if case_ref() is not None:
                raise RuntimeError('Patient source retained after visit; refusing next patient')
            self.completed.append({'subject': subject, 'phase': phase, 'source_released': True})
            self.guard()
            return result
        finally:
            base = None
            self.active = False


def run_train_cohort_sequential(public_visit_factories, *, learning_protocol, limits, output):
    validate_factories(public_visit_factories); validate_limits(learning_protocol, limits)
    protocol = freeze_json(learning_protocol); limits = freeze_json(limits)
    if (torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1
            or torch.get_default_dtype() != torch.float32 or str(torch.get_default_device()) != 'cpu'):
        raise ValueError('Owned worker must establish one thread and CPU float32 defaults')
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    sink = OutputBudget(root, limits['output_bytes'])
    budget = PlanningBudget(NativeResectionEngine,
        max_native_previews=limits['max_native_previews'], seconds=limits['worker_seconds'])
    costs = preflight._CallCosts(budget, limits['max_policy_forwards'])
    started = time.perf_counter(); models = {}; sessions = {}; teachers = {}
    result = {'version': VERSION, 'status': 'started', 'TRAIN_subjects': list(TRAIN),
        'closed_roles': CLOSED, 'SELECT_EVAL_opened': False, 'private_reference_reads': 0,
        'optimizer_updates': {'IL': 0, 'RL': 0}, 'checkpoints': {}, 'TRAIN_greedy': {},
        'teachers': {s: {'status': 'not_started'} for s in TRAIN},
        'selection_readiness': {'ready': False, 'execution_admitted': False}, 'clinical_claim': False}
    def guard(): budget.check(); sink.check()
    def directory(*parts):
        target = root.joinpath(*parts); target.mkdir(parents=True, exist_ok=False); return target
    visits = _SequentialVisits(public_visit_factories, protocol, limits, sink, costs, guard)
    def collect_replay(base, context, dest, *, method, policy=None, updates=0, actions=None, generator=None):
        trace = preflight._collect(base, context, policy=policy, generator=generator,
            actions=actions, output=dest, guard=guard)
        sealed = preflight._seal_and_replay(base, context, trace,
            method=method, policy=policy, updates=updates, output=dest, guard=guard)
        return trace, sealed['plan_seal']
    sink.write(root/'configuration.json', {'learning_protocol': protocol, 'limits': limits,
        'preview_budget_sizing': preview_budget_sizing(protocol),
        'memory_plan': 'one_patient_source_one_complete_trace_one_action_graph',
        'runtime_feasibility_established': False, 'SELECT_EVAL_execution_admitted': False})
    try:
        with budget, costs, _bounded_writes(sink):
            for subject in TRAIN:
                dest = directory('teachers', subject)
                result['teachers'][subject] = {'status': 'started'}
                def teacher(base, context):
                    try:
                        actions, stats = preflight.observed_beam_search(base, **thaw_json(limits['search']),
                            objective_source='supplied_public_whole_tumor_and_frozen_geometric_costs',
                            transition_mode='lazy_planning',
                            retention_mode=protocol['cohort_execution']['retention_mode'],
                            retained_prefix_diagnostics=protocol['cohort_execution']['retained_prefix_diagnostics'])
                        sink.write(dest/'search.json', {'actions': actions, 'accounting': stats})
                        if stats['call_cap_reached'] or stats['time_cap_reached']:
                            raise InterruptedError('Teacher search unresolved at cap; no replacement label or patient')
                        trace, seal = collect_replay(base, context, dest, method='SEARCH', actions=actions)
                        total = sum(t.reward for t in trace.transitions)
                        return {'status': 'complete_replayed', 'actions': list(actions), 'accounting': stats,
                            'trace_seal': trace.seal_hash, 'plan_seal': seal, 'context_hash': context.fingerprint,
                            'labels': len(trace.transitions), 'public_return': total,
                            'positive_nonstop': total > 0 and any(t.action_id != 'STOP' for t in trace.transitions)}
                    except BaseException as error:
                        sink.write(dest/'failure.json', {'type': type(error).__name__, 'message': str(error),
                            'accounting': getattr(error, 'accounting', None),
                            'returned_actions': list(getattr(error, 'best_sequence', ()))})
                        raise
                try:
                    teachers[subject] = visits.run(subject, dest, 'offline.teacher.'+subject, teacher)
                    result['teachers'][subject] = teachers[subject]
                except BaseException as error:
                    result['teachers'][subject] = {'status': 'failed_or_unresolved',
                        'failure_type': type(error).__name__, 'message': str(error)}
                    raise
            sink.write(root/'teacher-readiness.json', teachers)
            if not any(t['positive_nonstop'] for t in teachers.values()):
                result['status'] = 'unresolved_no_positive_teacher_signal'
                raise InterruptedError('All four teachers lack positive non-STOP signal; no training')
            contexts = tuple(visits.contexts[s] for s in TRAIN)
            with costs.scope('offline.common_initialization'):
                il, rl = common_patient_policies(contexts, protocol)
                initial = parameter_hash(il); result['initial_parameter_hash'] = initial
                models = {'IL': il, 'RL': rl}
                sessions = {m: PatientTrainSession(contexts, m, p,
                    initial_parameter_hash=initial, protocol=protocol) for m,p in models.items()}
            pins = {teachers[s]['context_hash']: {'steps': teachers[s]['labels'],
                'trace_seal': teachers[s]['trace_seal']} for s in TRAIN}
            generator = torch.Generator(device='cpu').manual_seed(protocol['seed']+1)
            for method in ('IL', 'RL'):
                policy, session = models[method], sessions[method]
                for update in range(protocol['updates_per_method']):
                    dest = directory(method, f'update-{update+1:02d}')
                    with PatientGradientAccumulator(session, teacher_pins=pins if method == 'IL' else None) as accumulation:
                        for subject in TRAIN:
                            visit_dir = directory(method, f'update-{update+1:02d}', subject)
                            def contribute(base, context):
                                trace, seal = collect_replay(base, context, visit_dir,
                                    method='IL_TEACHER_RECOLLECTION' if method == 'IL' else 'RL_COLLECTION',
                                    policy=None if method == 'IL' else policy, updates=session.updates,
                                    actions=teachers[subject]['actions'] if method == 'IL' else None,
                                    generator=None if method == 'IL' else generator)
                                with costs.scope(f'offline.{method}.update-{update+1:02d}.action_backward.'+subject):
                                    details = accumulation.add_trace(trace, guard=guard)
                                return {'plan_seal': seal, **details}
                            details = visits.run(subject, visit_dir,
                                f'offline.{method}.update-{update+1:02d}.collection_replay.'+subject, contribute)
                            sink.write(visit_dir/'gradient-contribution.json', details)
                        with costs.scope(f'offline.{method}.update-{update+1:02d}.shared_optimizer_step'):
                            receipt = accumulation.finish(guard=guard)
                            result['optimizer_updates'][method] = session.updates
                            sink.write(dest/'update.json', receipt); guard()
                with costs.scope('offline.'+method+'.checkpoint_save'):
                    result['checkpoints'][method] = _save_checkpoint(policy, method=method, contexts=contexts,
                        protocol=protocol, updates=session.updates, initial_hash=initial, output=sink, limits=limits)
            sink.write(root/'checkpoint-freeze.json', {'selection': 'fixed_endpoint_no_heldout_selection',
                'learning_protocol_hash': semantic_digest(protocol), 'checkpoints': result['checkpoints']})
            for method, policy in models.items():
                result['TRAIN_greedy'][method] = {}; frozen = parameter_hash(policy)
                for subject in TRAIN:
                    dest = directory('TRAIN-greedy', method, subject)
                    def greedy(base, context):
                        trace, seal = collect_replay(base, context, dest, method=method,
                            policy=policy, updates=sessions[method].updates)
                        return {'plan_seal': seal, 'complete': True,
                            'public_return': sum(t.reward for t in trace.transitions),
                            'steps': len(trace.transitions), 'actions': [t.action_id for t in trace.transitions]}
                    result['TRAIN_greedy'][method][subject] = visits.run(subject, dest,
                        'deployment.TRAIN_greedy_replay.'+method+'.'+subject, greedy)
                if parameter_hash(policy) != frozen:
                    raise RuntimeError('Greedy deployment changed frozen weights')
            guard(); budget.complete(history_complete=True)
        result['status'] = 'complete_TRAIN_only_not_heldout_performance'
        result['selection_readiness'] = {'ready': True, 'execution_admitted': False,
            'meaning': 'both fixed endpoints and all8 complete TRAIN greedy replays retained',
            'SELECT_EVAL_opened': False, 'checkpoint_loader_available': True,
            'next_interface': 'separate admitted greedy SELECT pass from authenticated checkpoints; no EVAL'}
    except BaseException as error:
        result['selection_readiness'] = {'ready': False, 'execution_admitted': False}
        if result['status'] == 'started': result['status'] = 'failed_or_unresolved'
        result['failure'] = {'type': type(error).__name__, 'message': str(error),
            'committed': bool(getattr(error, 'committed', False))}
        raise
    finally:
        cost_error = None
        details = {'costs': costs.rows, 'native_budget': budget.snapshot(),
            'total_policy_forward_calls': costs.forwards, 'completed_patient_visits': visits.completed,
            'complete_wall_seconds': time.perf_counter()-started,
            'cost_scope': 'offline and deployment separate; nested instrumentation inclusive, do not sum',
            'memory_scope': 'source lifetime checked; actual peak tree RSS requires owned supervisor',
            'resource_authority': 'owned parent hard wall/tree RSS/output receipt required'}
        try: sink.write(root/'costs.json', details)
        except BaseException as error:
            cost_error = error; result['cost_record_omitted'] = True
            result['status'] = 'failed_or_unresolved'
            result['selection_readiness'] = {'ready': False, 'execution_admitted': False}
        result['complete_wall_seconds'] = details['complete_wall_seconds']
        result['output_bytes_before_terminal'] = sink.size()
        compact = {k:v for k,v in result.items() if k not in ('teachers', 'checkpoints')}
        compact['teacher_statuses'] = {k:v['status'] for k,v in result['teachers'].items()}
        compact['checkpoints'] = {k:{a:v[a] for a in ('path','sha256','bytes','parameter_hash')}
            for k,v in result['checkpoints'].items()}
        sink.write(root/'result.json', compact, terminal=True)
        if cost_error is not None: raise cost_error
    return result
