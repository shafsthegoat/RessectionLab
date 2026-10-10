"""Canonical-only contact smoke/pilot child; requires the existing parent lease."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from resectionlab.legacy_transfer_worker import _require_parent_lease


def smoke(output):
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from resectionlab.contact_learning_contract import freeze_contact_experiment, common_initial_policies, bind_training_task
    from resectionlab.contact_learning import ContactLearningSession, contact_imitation_loss, contact_gradient_step
    from resectionlab.contact_checkpoint import save_contact_checkpoint, load_contact_checkpoint, partial_lineage
    from resectionlab.contact_costs import ContactCostMeter, peak_rss_bytes
    from resectionlab.contact_learning_pilot import _checked_trace, _optimizer, _write_json
    from resectionlab.contact_family_episode import bind_family_episode, export_family_strategy, SCHEMA
    from resectionlab.core import semantic_digest, thaw_json
    from resectionlab.public_contact_family import make_family_task
    from resectionlab.public_surface_contact import seal_complete_strategy, verify_complete_strategy
    from resectionlab.observed_search import observed_beam_search
    from resectionlab.goal_mode_episode_adapter import plan_goal_mode_strategy
    from resectionlab.spatial_policy import parameter_hash
    from resectionlab.surface_contact_episode import episode_envelope
    from resectionlab.development_episode import execute_development_episode

    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    declaration = {'version': 'public-contact-canonical-learning-smoke-v1',
        'scope': 'one_TRAIN_software_integration_only_not_pilot_performance_or_heldout',
        'layout': 'pcf-06', 'goal': 'surface', 'role': 'TRAIN',
        'teacher': {'max_calls': 256, 'beam_width': 96, 'seconds': 4.},
        'successful_updates': 1, 'learning_method': 'IL', 'pilot_endpoint_unchanged': 32,
        'checkpoint_kind': 'partial', 'learned_desktop_publication': False,
        'v3_exports': ['STOP', 'SEARCH'], 'legacy_default_compare': True,
        'patient_reads': 0, 'held_out_execution': False, 'automatic_retry': False}
    _write_json(output/'smoke-declaration.json', declaration)
    result = {'status': 'started', 'declaration': declaration}
    with ContactCostMeter() as meter:
        try:
            experiment = freeze_contact_experiment()
            result['experiment_hash'] = experiment.fingerprint
            with meter.scope('smoke.source_and_teacher'):
                task = make_family_task('pcf-06', 'surface')
                binding = bind_training_task(experiment, task, layout_id='pcf-06', goal_id='surface')
                actions, accounting = observed_beam_search(task, **declaration['teacher'],
                    objective_source='same_public_retained_surface_goal', transition_mode='lazy_planning')
                result['teacher_accounting'] = accounting
                if accounting['call_cap_reached'] or accounting['time_cap_reached']:
                    raise RuntimeError('Smoke teacher unresolved at declared cap; no replacement or update')
                trace, teacher = _checked_trace(task, actions, binding, meter=meter, phase='smoke.teacher')
                _write_json(output/'teacher-strategy.json', teacher)
            with meter.scope('smoke.loss_and_one_update'):
                policy = common_initial_policies(experiment)['IL']
                session = ContactLearningSession(experiment, 'IL', policy, parameter_hash(policy))
                batch = [trace[i % len(trace)].sample for i in range(4)]
                loss, loss_stats = contact_imitation_loss(session, batch)
                update = contact_gradient_step(session, _optimizer(policy), loss)
                if session.updates != 1 or not update['parameters_changed']:
                    raise AssertionError('Smoke requires one successful parameter-changing admitted update')
                result.update(loss=loss_stats, update=update)
            with meter.scope('smoke.partial_save_load'):
                receipt = save_contact_checkpoint(output/'smoke-partial.gmckpt', policy, experiment, partial_lineage(session))
                restored, metadata = load_contact_checkpoint(receipt['path'], expected_sha256=receipt['sha256'],
                    experiment=experiment, kind='partial')
                if parameter_hash(restored) != parameter_hash(policy) or metadata['lineage']['optimizer_updates'] != 1:
                    raise AssertionError('One-update partial checkpoint changed on verified reload')
                result['partial_checkpoint'] = receipt
            with meter.scope('smoke.loaded_policy_shared_plan_and_replay'):
                fresh = make_family_task('pcf-06', 'surface')
                context, _ = bind_family_episode(experiment, fresh, layout_id='pcf-06', goal_id='surface')
                plan, seal, learned_accounting = plan_goal_mode_strategy(restored, fresh, context=context)
                replay = seal_complete_strategy(fresh, plan['actions'])
                if semantic_digest(replay['strategy']) != semantic_digest(plan) or replay['strategySeal'] != seal:
                    raise AssertionError('Reloaded one-update policy plan differs from authoritative shared replay')
                verify_complete_strategy(fresh, replay)
                _write_json(output/'learned-partial-strategy-not-publishable.json',
                    {'strategy': plan, 'strategySeal': seal, 'accounting': learned_accounting,
                     'checkpoint': receipt, 'desktopPublication': False, 'scope': declaration['scope']})
                try:
                    export_family_strategy(experiment, fresh, layout_id='pcf-06', goal_id='surface',
                        selector='IL', plan=plan, seal=seal, accounting=learned_accounting,
                        context=context, checkpoint_metadata=metadata)
                except ValueError:
                    result['partial_learned_publication_refused'] = True
                else:
                    raise AssertionError('One-update smoke checkpoint must not publish as final32 learned v3')
            result['v3_exports'] = {}
            for selector in ('STOP', 'SEARCH'):
                with meter.scope('smoke.'+selector+'.v3_export_and_replay'):
                    fresh = make_family_task('pcf-06', 'surface')
                    context, _ = bind_family_episode(experiment, fresh, layout_id='pcf-06', goal_id='surface')
                    package = seal_complete_strategy(fresh, ['STOP']) if selector == 'STOP' else teacher
                    stats = {'actor_forward_calls': 0} if selector == 'STOP' else accounting
                    display, episode = export_family_strategy(experiment, fresh, layout_id='pcf-06', goal_id='surface',
                        selector=selector, plan=package['strategy'], seal=package['strategySeal'],
                        accounting=stats, context=context)
                    if (episode['schema'] != SCHEMA or episode['planning']['optimizer_updates'] != 0
                            or episode['learnedAuthorship'] is not None or episode['shape'] != [15,15,14]):
                        raise AssertionError('Versioned real family export has wrong authority/scope')
                    verify_complete_strategy(make_family_task('pcf-06', 'surface'), package)
                    _write_json(output/(selector+'-v3-episode.json'), episode_envelope(episode))
                    result['v3_exports'][selector] = {'episode_id': episode['episodeId'], 'case_hash': display.semantic_hash}
            with meter.scope('smoke.legacy_default_full_equality'):
                baseline = Path(__file__).with_name('legacy_development_baseline.py')
                if hashlib.sha256(baseline.read_bytes()).hexdigest() != 'de37e5f7b37f931c66b58296263dfd1d013bb324025a23112bfbef6131733162':
                    raise ValueError('Frozen pre-change legacy exporter baseline changed')
                spec = importlib.util.spec_from_file_location('resectionlab._contact_smoke_legacy_baseline', baseline)
                module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
                old_case, old_episode = module.execute_development_episode()
                new_case, new_episode = execute_development_episode()
                if old_case.semantic_hash != new_case.semantic_hash or semantic_digest(old_episode) != semantic_digest(new_episode):
                    raise AssertionError('Default legacy full episode changed under optional exporter extension')
                result['legacy_default'] = {'case_hash': new_case.semantic_hash, 'episode_id': new_episode['episodeId'],
                    'exact_full_episode_equal': True}
            result['status'] = 'complete'
        except BaseException as error:
            result.update(status='failed', error_type=type(error).__name__, message=str(error))
            raise
        finally:
            result.update(costs=meter.rows, total_wall_seconds=time.perf_counter()-started,
                process_peak_rss_bytes=peak_rss_bytes(), pilot_execution=False, held_out_execution=False,
                published_learned_result=False)
            _write_json(output/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke','pilot'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--released-experiment-hash')
    args = parser.parse_args(); _require_parent_lease()
    if args.mode == 'smoke':
        smoke(args.output)
    else:
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        from resectionlab.contact_learning_pilot import execute_contact_learning_pilot
        execute_contact_learning_pilot(args.output, released_experiment_hash=args.released_experiment_hash)


if __name__ == '__main__': main()
