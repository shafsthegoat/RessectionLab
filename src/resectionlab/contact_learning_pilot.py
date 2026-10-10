"""Fixed, one-attempt shared-native IL/RL/SEARCH pilot; explicit execution only.

Default preflight creates no task, teacher, rollout, update, checkpoint load or
evaluation. Actual execution remains a separately released owned-worker action.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
import numpy as np
import torch

from .contact_checkpoint import (save_contact_checkpoint, load_contact_checkpoint,
    initial_lineage, final_lineage, partial_lineage)
from .contact_costs import ContactCostMeter, peak_rss_bytes
from .contact_family_episode import (freeze_final_checkpoints, make_frozen_evaluation_task,
    bind_family_episode, export_family_strategy)
from .contact_learning_contract import (PROTOCOL, ContactSample, ContactTransition,
    freeze_contact_experiment, bind_training_task, common_initial_policies)
from .contact_learning import (ContactLearningSession, contact_imitation_loss,
    contact_reinforce_loss, contact_gradient_step)
from .core import freeze_json, semantic_digest, thaw_json
from .goal_mode_episode_adapter import plan_goal_mode_strategy
from .observed_search import observed_beam_search, ObservedSearchLimit
from .public_contact_family import make_family_task
from .public_surface_contact import seal_complete_strategy, verify_complete_strategy
from .surface_contact_episode import episode_envelope
from .spatial_policy import parameter_hash


class PilotLimit(RuntimeError): pass


class Budget:
    """Cooperative cancellation; the owned parent must also supervise hard limits."""
    def __init__(self, seconds): self.started = time.perf_counter(); self.deadline = self.started+seconds
    def cancelled(self): return time.perf_counter() > self.deadline or peak_rss_bytes() > PROTOCOL['memory_bytes']
    def check(self):
        if self.cancelled(): raise PilotLimit('Fixed stage wall/RSS cap reached; retain partial result, no retry')


def _failure_details(error):
    return {'error_type': type(error).__name__, 'message': str(error),
        'durable_transition_committed': bool(getattr(error, 'committed', False)),
        'committed_transition': getattr(error, 'info', None),
        'committed_reward': getattr(error, 'reward', None),
        'partial_collection': getattr(error, 'contact_partial_record', None)}


def _write_json(path, data):
    # Each evidence item is append-only; no previous attempt is overwritten.
    with Path(path).open('x') as target:
        json.dump(thaw_json(freeze_json(data)), target, indent=2, allow_nan=False); target.write('\n')


def _optimizer(policy):
    return torch.optim.Adam(policy.parameters(), lr=PROTOCOL['learning_rate'], betas=tuple(PROTOCOL['betas']),
        **{key: PROTOCOL[key] for key in ('eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'fused')})


def _checked_trace(task, actions, binding, *, meter, phase):
    """Collect observations from actual shared execution, then validate full replay."""
    worker = task.planning_clone(); transitions = []
    with meter.scope(phase + '.trace'):
        for action in actions:
            obs = worker.observation(); binding.require_observation(obs)
            result = worker.step(action)
            transitions.append(ContactTransition(ContactSample(obs, action, binding), result.reward, result.terminated))
        if not worker.terminated: raise ValueError('Only complete strategies supply training trajectories')
    with meter.scope(phase + '.native_replay'):
        package = seal_complete_strategy(task, actions)
        if semantic_digest(worker.metrics()['history']) != semantic_digest(package['strategy']['history']):
            raise RuntimeError('Collected native rewards/actions differ from sealed replay')
        verify_complete_strategy(task, package)
    return tuple(transitions), package


def _teachers(experiment, output, meter):
    budget = Budget(PROTOCOL['teacher_seconds']); samples = []; records = []
    for index, (layout, goal) in enumerate(experiment.keys('TRAIN')):
        record = {'layout_id': layout, 'goal_id': goal, 'role': 'TRAIN', 'teacher_index': index}
        if budget.cancelled():
            record.update(status='unresolved_stage_cap', label_count=0); records.append(record)
            _write_json(output / f'teacher-{index:02d}.json', record); continue
        try:
            with meter.scope('teacher.source_and_inventory'):
                task = make_family_task(layout, goal, cancelled=budget.cancelled)
                binding = bind_training_task(experiment, task, layout_id=layout, goal_id=goal)
            with meter.scope('teacher.search'):
                actions, stats = observed_beam_search(task, **dict(PROTOCOL['search']),
                    objective_source='same_public_retained_surface_goal', transition_mode='lazy_planning')
            record['search'] = stats
            if stats['call_cap_reached'] or stats['time_cap_reached']:
                record.update(status='unresolved_search_cap', label_count=0, returned_actions=list(actions))
            else:
                trace, package = _checked_trace(task, actions, binding, meter=meter, phase='teacher')
                budget.check()
                samples.extend(t.sample for t in trace)
                record.update(status='complete_bounded_teacher', label_count=len(trace),
                    strategy=package, binding=binding.record(), global_optimality_claim=False)
        except (ObservedSearchLimit, InterruptedError, PilotLimit) as error:
            record.update(status='unresolved_limit', label_count=0, **_failure_details(error),
                search=getattr(error, 'accounting', {}),
                returned_actions=list(getattr(error, 'best_sequence', ())))
        records.append(record); _write_json(output / f'teacher-{index:02d}.json', record)
    return samples, records


def _collect_rl(experiment, session, keys, generator, budget, meter):
    behavior = parameter_hash(session.policy); episodes = []; packages = []
    for layout, goal in keys:
        budget.check()
        with meter.scope('RL.collection_source_and_inventory'):
            task = make_family_task(layout, goal, cancelled=budget.cancelled)
            binding = bind_training_task(experiment, task, layout_id=layout, goal_id=goal)
            worker = task.planning_clone()
        transitions, actions = [], []
        try:
            with meter.scope('RL.collection'):
                while not worker.terminated:
                    if parameter_hash(session.policy) != behavior: raise RuntimeError('Rollout policy changed within on-policy batch')
                    obs = worker.observation(); binding.require_observation(obs)
                    action = session.policy.act(obs, context=binding.context, stochastic=True, generator=generator)
                    result = worker.step(action); actions.append(action)
                    transitions.append(ContactTransition(ContactSample(obs, action, binding), result.reward, result.terminated))
            with meter.scope('RL.collection_native_replay'):
                package = seal_complete_strategy(task, actions)
                if semantic_digest(worker.metrics()['history']) != semantic_digest(package['strategy']['history']):
                    raise RuntimeError('RL collection differs from native strategy replay')
                verify_complete_strategy(task, package)
        except BaseException as error:
            error.contact_partial_record = {'layout_id': layout, 'goal_id': goal,
                'binding': binding.record(), 'native_metrics': worker.metrics(),
                'complete_episodes_in_batch': packages,
                'completed_transition_observations': [t.sample.observation.fingerprint for t in transitions]}
            raise
        episodes.append(tuple(transitions)); packages.append({'layout_id': layout, 'goal_id': goal,
            'binding': binding.record(), 'strategy': package})
    if parameter_hash(session.policy) != behavior: raise RuntimeError('RL batch weights changed')
    return episodes, packages, behavior


def _train(experiment, method, policy, samples, output, meter, *, session):
    optimizer = _optimizer(policy); budget = Budget(PROTOCOL['method_seconds'])
    rng = np.random.default_rng(PROTOCOL['seed']); generator = torch.Generator().manual_seed(PROTOCOL['seed']+1)
    keys = experiment.keys('TRAIN'); order = tuple(keys[int(i)] for i in rng.permutation(len(keys)))
    rollout_transitions = 0; update_records = []
    for update in range(PROTOCOL['updates']):
        budget.check(); packages = None
        if method == 'IL':
            if not samples: raise PilotLimit('No complete teacher labels; IL remains unresolved with no manufactured targets')
            batch = tuple(samples[int(i)] for i in rng.integers(0, len(samples), PROTOCOL['batch_size']))
            with meter.scope('IL.loss_forwards'):
                loss, loss_stats = contact_imitation_loss(session, batch)
            sample_ids = [{'binding_hash': s.binding.fingerprint, 'observation_hash': s.observation.fingerprint,
                           'action_id': s.action_id} for s in batch]
        else:
            batch_keys = tuple(order[(update*PROTOCOL['batch_size']+i) % len(order)] for i in range(PROTOCOL['batch_size']))
            episodes, packages, behavior = _collect_rl(experiment, session, batch_keys, generator, budget, meter)
            rollout_transitions += sum(map(len, episodes))
            if rollout_transitions > PROTOCOL['rl_rollout_transition_cap']: raise PilotLimit('Fixed RL transition cap exceeded')
            with meter.scope('RL.loss_forwards'):
                loss, loss_stats = contact_reinforce_loss(session, episodes, behavior_parameter_hash=behavior)
            sample_ids = [{'binding_hash': t.sample.binding.fingerprint, 'observation_hash': t.sample.observation.fingerprint,
                           'action_id': t.sample.action_id} for ep in episodes for t in ep]
        budget.check()
        with meter.scope(method + '.optimizer'):
            update_stats = contact_gradient_step(session, optimizer, loss)
        record = {'update': update+1, 'method': method, 'loss': loss_stats, 'update_receipt': update_stats,
            'samples': sample_ids, 'rollout_transitions_cumulative': rollout_transitions,
            'native_trajectories': packages, 'elapsed_seconds': time.perf_counter()-budget.started}
        update_records.append(record); _write_json(output / f'{method}-update-{update+1:02d}.json', record)
        budget.check()
    return session, update_records


def _online(experiment, role, artifacts, release, output, meter):
    rows = []; policies = {}; metadata = {}
    with meter.scope(role + '.checkpoint_load'):
        for method in ('IL', 'RL'):
            policies[method], metadata[method] = load_contact_checkpoint(artifacts[method]['path'],
                expected_sha256=artifacts[method]['sha256'], experiment=experiment, kind='final')
    for index, (layout, goal) in enumerate(experiment.keys(role)):
        order = ('STOP', 'SEARCH', 'IL', 'RL')
        order = order[index % 4:] + order[:index % 4]
        for method in order:
            prefix = f'{role}.{method}.{layout}.{goal}'; started = time.perf_counter()
            # Includes source construction, certified inventory, planning and replay.
            budget = Budget(PROTOCOL['online_seconds'])
            row = {'role': role, 'layout_id': layout, 'goal_id': goal, 'method': method}
            task = None
            try:
                with meter.scope(prefix + '.source_and_inventory'):
                    task = make_frozen_evaluation_task(experiment, layout, goal, release=release, cancelled=budget.cancelled)
                    context, _ = bind_family_episode(experiment, task, layout_id=layout, goal_id=goal, release=release)
                with meter.scope(prefix + '.planning'):
                    if method in ('IL', 'RL'):
                        plan, seal, accounting = plan_goal_mode_strategy(policies[method], task, context=context)
                    else:
                        if method == 'STOP':
                            actions, accounting = ('STOP',), {'selector': 'immediate_STOP', 'actor_forward_calls': 0,
                                                             'model_transition_calls': 0, 'optimizer_updates': 0}
                        else:
                            actions, accounting = observed_beam_search(task, **dict(PROTOCOL['search']),
                                objective_source='same_public_retained_surface_goal', transition_mode='lazy_planning')
                        package = seal_complete_strategy(task, actions)
                        plan, seal = package['strategy'], package['strategySeal']
                    budget.check()
                with meter.scope(prefix + '.execution_and_replay_export'):
                    display, episode = export_family_strategy(experiment, task, layout_id=layout, goal_id=goal,
                        selector=method, plan=plan, seal=seal, accounting=accounting, context=context,
                        release=release, checkpoint_metadata=metadata.get(method))
                    envelope = episode_envelope(episode)
                    _write_json(output / f'{role}-{index:02d}-{method}-episode.json', envelope)
                    budget.check()
                row.update(status='complete', metrics=episode['metrics'], planning=episode['planning'],
                    episode_id=episode['episodeId'], case_hash=display.semantic_hash,
                    total_wall_seconds=time.perf_counter()-started, global_optimality_claim=False)
            except (ObservedSearchLimit, InterruptedError, PilotLimit) as error:
                row.update(status='limit_or_unresolved', **_failure_details(error),
                    search=getattr(error, 'accounting', {}), total_wall_seconds=time.perf_counter()-started,
                    returned_actions=list(getattr(error, 'best_sequence', ())),
                    partial_native_metrics=None if task is None else task.metrics())
            with meter.scope(prefix + '.result_serialization'):
                _write_json(output / f'{role}-{index:02d}-{method}-result.json', row)
            row['complete_online_wall_seconds'] = time.perf_counter()-started
            row['timing_scope'] = 'includes_source_inventory_planning_execution_replay_and_method_evidence_writes; final_study_report_write_separate'
            rows.append(row)
    for method in ('IL', 'RL'):
        if parameter_hash(policies[method]) != artifacts[method]['parameter_hash']:
            raise RuntimeError('Inference changed frozen checkpoint weights')
    return rows


def execute_contact_learning_pilot(output, *, released_experiment_hash):
    """Explicit run action, never invoked by import or default preflight."""
    experiment = freeze_contact_experiment()
    if released_experiment_hash != experiment.fingerprint:
        raise ValueError('Exact prospective experiment digest must be released before any task/model work')
    if (torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1
            or torch.get_default_dtype() != torch.float32 or str(torch.get_default_device()) != 'cpu'):
        raise ValueError('Owned worker requires one Torch/inter-op thread, CPU and float32 defaults')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    _write_json(output / 'experiment-freeze.json', experiment.record())
    result = {'experiment_hash': experiment.fingerprint, 'status': 'running',
        'patient_reads': 0, 'seed': PROTOCOL['seed'], 'training': {}, 'teacher_slots': [],
        'measurement_status': 'closed_until_both_final_checkpoints_verified'}
    start = time.perf_counter()
    with ContactCostMeter() as meter:
        try:
            with meter.scope('initialization'):
                policies = common_initial_policies(experiment)
                result['initial_checkpoint'] = save_contact_checkpoint(output / 'common-initial.gmckpt', policies['IL'],
                    experiment, initial_lineage(experiment))
            samples, teachers = _teachers(experiment, output, meter); result['teacher_slots'] = teachers
            artifacts = {}
            for method in ('IL', 'RL'):
                session = ContactLearningSession(experiment, method, policies[method], parameter_hash(policies[method]))
                try:
                    session, updates = _train(experiment, method, policies[method], samples, output, meter, session=session)
                    with meter.scope(method + '.checkpoint_save'):
                        artifacts[method] = save_contact_checkpoint(output / f'{method}-final.gmckpt', policies[method],
                                                                    experiment, final_lineage(session))
                    result['training'][method] = {'status': 'completed_fixed_endpoint', 'updates': len(updates),
                        'checkpoint': artifacts[method]}
                except (InterruptedError, PilotLimit) as error:
                    result['training'][method] = {'status': 'incomplete_at_fixed_cap', **_failure_details(error), 'parameter_hash': parameter_hash(policies[method]),
                        'completed_updates': session.updates,
                        'partial_checkpoint': save_contact_checkpoint(output / f'{method}-partial.gmckpt', policies[method],
                            experiment, partial_lineage(session))}
            if set(artifacts) == {'IL', 'RL'}:
                with meter.scope('checkpoint_freeze_verification'):
                    release = freeze_final_checkpoints(experiment, artifacts)
                    _write_json(output / 'final-checkpoint-freeze.json', release.record())
                result['SELECT'] = _online(experiment, 'SELECT', artifacts, release, output, meter)
                result['MEASUREMENT_EVAL'] = _online(experiment, 'MEASUREMENT_EVAL', artifacts, release, output, meter)
                result['measurement_status'] = 'single_frozen_pass_finished_no_checkpoint_selection'
                result['status'] = 'complete_with_all_failures_preserved'
            else:
                result['status'] = 'incomplete_training_measurement_not_opened'
        except BaseException as error:
            result.update(status='failed', error_type=type(error).__name__, message=str(error))
            raise
        finally:
            result['costs'] = meter.rows
            result['complete_wall_seconds'] = time.perf_counter()-start
            result['process_peak_rss_bytes'] = peak_rss_bytes()
            result['cost_interpretation'] = 'separate offline/online scopes; inclusive timings not additive; no speed claim'
            _write_json(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--released-experiment-hash')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if not args.execute:
        experiment = freeze_contact_experiment()
        print(json.dumps({'status': 'source_only_preflight', 'experiment_hash': experiment.fingerprint,
            'protocol': thaw_json(PROTOCOL), 'task_constructions': 0, 'held_out_execution': False}, indent=2))
        return
    if args.output is None: parser.error('--output is required for a new single-attempt run')
    execute_contact_learning_pilot(args.output, released_experiment_hash=args.released_experiment_hash)


if __name__ == '__main__': main()
