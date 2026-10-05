#!/usr/bin/env python3
"""Source-derived V2 cost report. Reads raw JSON or verified lossless .json.gz.

This does not import a simulator, policy, torch, or an experiment runner.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

EPISODES = ('greedy', 'selection-1', 'selection-2', 'untrained-selection-1', 'untrained-selection-2')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def content_hash(value):
    return 'sha256:' + digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


class Inputs:
    def __init__(self, root):
        self.root, self.hashes = root, {}

    def read(self, relative):
        path = self.root / relative
        compressed = path.with_suffix(path.suffix + '.gz')
        if path.is_file():
            raw = path.read_bytes()
            if compressed.is_file() and gzip.decompress(compressed.read_bytes()) != raw:
                raise ValueError(f'Raw/compressed evidence disagreement: {relative}')
        else:
            raw = gzip.decompress(compressed.read_bytes())
        self.hashes[relative] = {'uncompressed_sha256': digest(raw), 'uncompressed_bytes': len(raw)}
        return json.loads(raw)


def extract(root, baseline_path):
    source = Inputs(root)
    launcher, status = source.read('launcher-status.json'), source.read('profile/status.json')
    assert launcher['status'] == status['status'] == 'completed'
    assert launcher['eligible_candidate_count'] == status['eligible_candidate_count'] == 2
    assert launcher['worker_returncode'] == 0 and not launcher['parent_timeout_requested']
    assert status['completed_selection_worlds'] == status['completed_untrained_policy_worlds'] == 2
    assert status['certified_complete_episodes'] == 5 and status['unique_native_audits'] == 2
    assert status['gradient_steps'] == 0 and not status['final_worlds_used'] and not status['stress_worlds_used']
    candidates = source.read('profile/candidate-records.json')
    assert status['candidate_records_sha256'] == source.hashes['profile/candidate-records.json']['uncompressed_sha256']
    initial = source.read('profile/initial/inventory.json')
    cold = source.read('profile/cold-setup.json')
    clone = source.read('profile/clone-probe.json')
    greedy_reset = source.read('profile/greedy-reset.json')
    preparation = source.read('preparation.json')
    worker = source.read('worker-resource.json')
    freeze = source.read('profile/untrained-policy-freeze.json')
    sequence = source.read('profile/sequence-freeze.json')
    model = source.read('profile/model.json')
    runtime = source.read('profile/source.json')
    declaration = source.read('declaration.json')
    baseline_raw = baseline_path.read_bytes()
    baseline = json.loads(baseline_raw)
    assert model['decision_model_hash'] == sequence['decision_model_hash'] == freeze['decision_model_hash']
    assert declaration['declaration_content_hash'] == baseline['declaration_content_hash']
    assert runtime['file_sha256']['scripts/preflight_native_axis.py'] == baseline['runner_sha256']
    assert model['max_steps'] == 3 and model['max_actions'] == 27
    assert all(value == 0 for key in ('rotation_scale_deg', 'translation_scale_mm')
               for value in preparation['selection']['generator'][key])
    panel_records = {name: source.read(f'profile/{name}-progress.json')
                     for name in ('selection', 'untrained-selection')}
    for panel in panel_records.values():
        assert panel['status'] == 'complete' and len(panel['worlds']) == 2
        assert panel['partition']['role'] == 'selection'
        assert panel['partition']['seeds'] == preparation['selection']['seeds']
    reset_seconds = {'greedy': greedy_reset['seconds']}
    for name, panel in panel_records.items():
        reset_seconds.update({f'{name}-{index + 1}': row['reset_seconds'] for index, row in enumerate(panel['worlds'])})
    rows, all_receipts = [], [initial['complete_inventory']]
    representative = {}
    for name in EPISODES:
        episode = source.read(f'profile/{name}/episode.json')
        timing = source.read(f'profile/{name}/timing.json')
        metrics = episode['metrics']
        receipts = metrics['inventory_receipts']
        assert episode['status'] == 'completed' and metrics['termination_reason'] == 'step_budget'
        assert len(episode['transitions']) == len(episode['actions']) == 3
        assert all(receipt['status'] == 'complete' for receipt in receipts)
        assert math.isclose(sum(t['reward'] for t in episode['transitions']), episode['total_reward'], abs_tol=1e-9)
        attempts = [attempt for receipt in receipts for attempt in receipt['attempts']]
        assert len(attempts) == metrics['proposal_accounting']['preview_calls']
        assert all(attempt['status'] == 'complete' and attempt['feasible'] for attempt in attempts)
        history_hash = content_hash(metrics['history'])
        family = 'greedy' if name in EPISODES[:3] else 'untrained'
        representative.setdefault(family, metrics)
        expected_hash = sequence['greedy_history_hash'] if family == 'greedy' else panel_records['untrained-selection']['worlds'][0]['history_hash']
        assert history_hash == expected_hash
        all_receipts.extend(receipts)
        rows.append({'episode': name, 'return': episode['total_reward'], 'reset_seconds': reset_seconds[name],
            'episode_seconds': timing['episode_seconds_including_export'],
            'episode_before_export_seconds': timing['episode_seconds_before_export'],
            'episode_export_seconds': timing['episode_seconds_including_export'] - timing['episode_seconds_before_export'],
            'reset_and_episode_seconds': reset_seconds[name] + timing['episode_seconds_including_export'],
            'decision_seconds': sum(t['decision_seconds'] for t in episode['transitions']),
            'transition_and_next_inventory_seconds': sum(t['transition_and_next_inventory_seconds'] for t in episode['transitions']),
            'each_transition_and_next_inventory_seconds': [t['transition_and_next_inventory_seconds'] for t in episode['transitions']],
            'preview_calls': len(attempts), 'preview_seconds': metrics['proposal_accounting']['preview_seconds'],
            'retained_integrity_calls': metrics['proposal_accounting']['integrity_calls'],
            'retained_integrity_seconds': metrics['proposal_accounting']['integrity_seconds'],
            'certified_nonstop_inventory_sizes': [len(r['certified_action_ids']) for r in receipts],
            'terminal_provider_proposals_uncertified': len(receipts[-1]['batch']['proposals']),
            'history_hash': history_hash, 'actions': episode['actions']})
    attempts_by_identity = defaultdict(list)
    phase_counts = Counter()
    for receipt in all_receipts:
        assert receipt['batch']['slot_count'] == 26
        for attempt in receipt['attempts']:
            identity = (model['decision_model_hash'], receipt['batch']['engine_model_hash'],
                receipt['batch']['cavity_state_hash'], attempt['proposal_id'], attempt['tool_id'],
                attempt['phase'], tuple(attempt['tip_mm']), tuple(attempt['entry_mm']))
            attempts_by_identity[identity].append(attempt['elapsed_seconds'])
            phase_counts[attempt['phase']] += 1
    geometry = {'actual_preview_calls': sum(map(len, attempts_by_identity.values())),
        'unique_full_model_cavity_proposal_phase_identities': len(attempts_by_identity),
        'preview_seconds': sum(sum(v) for v in attempts_by_identity.values()),
        'repeated_preview_calls': sum(len(v)-1 for v in attempts_by_identity.values()),
        'observed_repeated_preview_seconds_after_first': sum(sum(v[1:]) for v in attempts_by_identity.values()),
        'phases': dict(phase_counts), 'complete_inventory_receipts': len(all_receipts),
        'reconstructed_initial_inventories': sum(r['batch']['cavity_state_hash'] == all_receipts[0]['batch']['cavity_state_hash'] for r in all_receipts),
        'retained_integrity_calls': initial['proposal_accounting']['integrity_calls'] + sum(r['retained_integrity_calls'] for r in rows),
        'retained_integrity_seconds': initial['proposal_accounting']['integrity_seconds'] + sum(r['retained_integrity_seconds'] for r in rows),
        'scope': 'Previews count all retained complete inventories. Integrity counters are reset-local and omit checks immediately before reset plus other unretained checks; not the exact global integrity total. Repetition is measured work, not a cache speedup estimate.'}
    audits = [source.read('profile/independent-audit.json'), source.read('profile/untrained-independent-audit.json')]
    assert all(a['audit']['feasible'] and a['audit']['complete_tool_checked'] and a['audit']['frontier_checked']
               and a['audit']['unsupported_source_tissue_volume_mm3'] == 0 for a in audits)
    audit_seconds = sum(a['seconds'] for a in audits)
    assert math.isclose(audit_seconds, status['independent_audit_seconds'])
    panel_costs = {}
    for label, subset in [('frozen_greedy_sequence', rows[1:3]), ('untrained_raw_policy', rows[3:])]:
        panel_costs[label] = {'reset_seconds': sum(r['reset_seconds'] for r in subset),
            'episode_seconds_including_json_export': sum(r['episode_seconds'] for r in subset),
            'total_reset_plus_episode_seconds': sum(r['reset_and_episode_seconds'] for r in subset),
            'episode_json_export_seconds': sum(r['episode_export_seconds'] for r in subset),
            'scope': 'sum of two measured resets and episodes, including explicit preflight inventory re-reads and JSON export; excludes caller panel bookkeeping and per-world clone/factory setup; not an exact generic-learner cost or rigorous lower bound'}
    cancellation = source.read('profile/cancellation-probe.json')
    measured_profile_phases = (cold['factory_seconds'] + initial['inspection_seconds'] + clone['seconds']
        + rows[0]['reset_and_episode_seconds'] + sum(p['total_reset_plus_episode_seconds'] for p in panel_costs.values())
        + freeze['initialization_seconds'] + audit_seconds + cancellation['boundary_response_seconds'])
    outcomes = {}
    for family, metrics in representative.items():
        target = metrics['simulated_removed_target_volume_mm3']
        residual = metrics['modeled_residual_target_volume_mm3']
        outcomes[family] = {'return': metrics['total_reward'], 'target_removed_mm3': target,
            'normal_removed_mm3': metrics['simulated_removed_normal_volume_mm3'],
            'cumulative_partial_normal_contact_mm3': metrics['cumulative_partial_normal_contact_mm3'],
            'residual_target_mm3': residual, 'source_target_mm3': target + residual,
            'target_fraction_removed': target / (target + residual), 'clinical_deficit_probability': None,
            'history_hash': content_hash(metrics['history'])}
    result = {'status':'completed_preflight_cost_report','source_commit':baseline['source_commit'],
        'source_archive_sha256':baseline['source_archive_sha256'], 'runtime_content_hash':runtime['runtime_content_hash'],
        'decision_model_hash':model['decision_model_hash'], 'proposal_model_hash':model['proposal_model_hash'], 'native_config_hash':preparation['native_config_hash'],
        'declaration_hash':declaration['declaration_content_hash'], 'policy_hash':freeze['policy_hash'],
        'completed_episodes':5,'eligible_development_records':2,'gradient_steps':0,'final_worlds_used':False,'stress_worlds_used':False,
        'launcher_seconds':launcher['full_launcher_seconds'], 'worker_seconds_before_final_resource_export':worker['elapsed_seconds'],
        'profile_seconds_before_final_status_export':status['full_preflight_seconds'],
        'source_preparation_seconds':preparation['case_and_native_config_seconds'],
        'cold_factory_seconds':cold['factory_seconds'], 'initial_inventory_inspection_seconds':initial['inspection_seconds'],
        'initialization_components':cold['adapter_initialization'], 'initial_clone_seconds':clone['seconds'],
        'raw_initialization_and_checkpoint_seconds':freeze['initialization_seconds'],
        'panels':panel_costs, 'episodes':rows,'geometry_accounting':geometry,'outcomes':outcomes,
        'independent_audit_seconds':audit_seconds,'individual_audit_seconds':[a['seconds'] for a in audits],
        'cancellation':cancellation,'resource':worker,'background_scope':baseline['background_scope'],
        'load_average_before_launch':baseline['load_average_before_launch'],
        'timing_reconciliation':{'sum_named_profile_phase_seconds':measured_profile_phases,
            'profile_residual_seconds':status['full_preflight_seconds']-measured_profile_phases,
            'scope':'Residual includes unseparated source/integrity checks, receipt serialization and orchestration. Nested preview/integrity timings must not be added again.'},
        'storage':{'large_episode_raw_bytes':sum(source.hashes[f'profile/{n}/episode.json']['uncompressed_bytes'] for n in EPISODES)},
        'future_training_budget':None}
    provenance={'inputs':source.hashes,'baseline_sha256':digest(baseline_raw),
        'report_script_sha256':digest(Path(__file__).read_bytes()),
        'source_policy':'Original receipts unchanged. Hashes identify uncompressed bytes whether the consumer reads JSON or gzip.'}
    return result, provenance


def render(value):
    v=value; g=v['geometry_accounting']; p=v['panels']; out=v['outcomes']
    lines=['# Native axis public preflight V2', '',
        f"The completed preflight took **{v['launcher_seconds']:.3f} seconds** end to end and peaked at **{v['resource']['observed_peak_rss_bytes']/1024**3:.3f} GiB**. All five episodes and both native geometry audits completed; there were zero gradients, final-world uses or stress-world uses.", '',
        f"The unchanged seed-11 RAW policy needed **{p['untrained_raw_policy']['total_reset_plus_episode_seconds']:.3f} seconds** for its two-world panel using one reused simulator, including {p['untrained_raw_policy']['episode_json_export_seconds']:.3f} seconds of episode JSON export. The earlier 30-second online limit would not cover this observed initial panel. No future training budget is declared.", '',
        '| Episode | Reset (s) | Episode, including export (s) | Decisions (s) | Transitions + next inventory (s) | Previews, including reset |',
        '|---|---:|---:|---:|---:|---:|']
    for r in v['episodes']:
        lines.append(f"| {r['episode']} | {r['reset_seconds']:.3f} | {r['episode_seconds']:.3f} | {r['decision_seconds']:.3f} | {r['transition_and_next_inventory_seconds']:.3f} | {r['preview_calls']} |")
    lines += ['',f"The frozen greedy-sequence panel cost {p['frozen_greedy_sequence']['total_reset_plus_episode_seconds']:.3f} seconds. Both panel totals sum measured resets and episodes; they exclude caller bookkeeping and repeated clone/factory setup. One initial integrity-checked clone cost {v['initial_clone_seconds']:.3f} seconds. The preflight also re-reads the inventory and observation before each policy action, while the generic learner can reuse a returned observation. Cloning adds work and different read patterns may remove work; this panel is neither an exact generic-learner cost nor a rigorous lower bound.", '',
        f"Case/native preparation cost {v['source_preparation_seconds']:.3f} seconds; cold adapter construction, including the full initial inventory, cost {v['cold_factory_seconds']:.3f} seconds. Initial inventory inspection added {v['initial_inventory_inspection_seconds']:.3f} seconds. RAW policy import, seeded initialization and checkpoint export cost {v['raw_initialization_and_checkpoint_seconds']:.3f} seconds. Independent native checks cost {v['individual_audit_seconds'][0]:.3f} and {v['individual_audit_seconds'][1]:.3f} seconds ({v['independent_audit_seconds']:.3f} total), shared across exactly matching histories from the five completed episodes.", '',
        f"The initial inventory certified all 26 primary rays plus STOP. Across {g['complete_inventory_receipts']} complete inventories, there were {g['actual_preview_calls']} successful primary previews and no fallback attempts, consuming {g['preview_seconds']:.3f} seconds. The full model/cavity/proposal/phase keys identify {g['unique_full_model_cavity_proposal_phase_identities']} unique geometries and {g['repeated_preview_calls']} repeated previews ({g['observed_repeated_preview_seconds_after_first']:.3f} seconds of observed repeated work). The initial inventory was constructed {g['reconstructed_initial_inventories']} times. This is a candidate for a separately checked optimization, not a predicted caching speedup.", '',
        f"Retained reset-local counters record {g['retained_integrity_calls']} integrity checks and {g['retained_integrity_seconds']:.3f} seconds. They omit checks erased when counters reset and therefore are not a global total. Preview and integrity times overlap the phase totals above. The recorded profile duration is {v['profile_seconds_before_final_status_export']:.3f} seconds; the named, nonoverlapping phases leave {v['timing_reconciliation']['profile_residual_seconds']:.3f} seconds of unseparated checks, serialization and orchestration. The worker receipt covers {v['worker_seconds_before_final_resource_export']:.3f} seconds before its final export; launcher time additionally includes process startup, source-copy preparation and teardown.", '',
        '| Complete development path | Modeled return | Target removed (mm³) | Normal removed (mm³) | Cumulative partial normal contact (mm³) | Residual target (mm³) |',
        '|---|---:|---:|---:|---:|---:|',
        *[f"| {name} | {o['return']:.2f} | {o['target_removed_mm3']:.0f} | {o['normal_removed_mm3']:.0f} | {o['cumulative_partial_normal_contact_mm3']:.0f} | {o['residual_target_mm3']:.0f} |" for name,o in out.items()], '',
        f"The two seed replays are deterministic checks, not independent patients or uncertainty samples. These are a greedy path and an unchanged random initialization, not a trained-policy benchmark. Greedy removed {100*out['greedy']['target_fraction_removed']:.3f}% of the declared radiological target. All paths stopped at the three-cut horizon. Their terminal provider ledgers still list 14 greedy-path or 20 untrained-path primary proposals; these terminal proposals were not previewed or certified, so STOP-only terminal inventories do not show proposal exhaustion. Partial contact remains separate from removed tissue.", '',
        f"The cancellation callback ran {v['resource']['cancellation_callback_calls']} times for {v['resource']['cancellation_callback_seconds']:.6f} seconds. The deliberate request before cached inventory access returned in {1e6*v['cancellation']['boundary_response_seconds']:.2f} microseconds without changing committed state. This does not measure worst-case cancellation inside a preview. No wall/RSS limit was reached.", '',
        'A fair RAW comparison still needs a bounded measurement of a complete update batch and its accounting/factory overhead before caps are chosen. The current observations do not predict gradient throughput or the costs of other sampled paths. Any optimization must preserve source/cavity integrity, complete inventories, geometry certificates and the committed-transition cancellation contract.', '',
        'This is one previously studied structural mirror-derived patient with unreviewed support and hypothetical access; motor/language outcomes, clinical probabilities and tissue mechanics remain unassessed. Other agent computation was held; normal macOS/desktop activity and OS file caches were uncontrolled. The original V1 preparation failure remains preserved separately.', '',
        f"Source commit: `{v['source_commit']}`. Runtime: `{v['runtime_content_hash']}`. Decision model: `{v['decision_model_hash']}`. Exact values and input hashes are in `cost-summary.json` and `report-source.json`; `report.py` accepts raw JSON or lossless gzip without changing the original receipts.", '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--baseline',type=Path)
    args=parser.parse_args(); output=args.output or args.inputs
    baseline=args.baseline or args.inputs.parents[2]/'artifacts/validation/native-axis-public-v2/execution-baseline.json'
    value, provenance=extract(args.inputs,baseline)
    output.mkdir(parents=True,exist_ok=True)
    (output/'cost-summary.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    (output/'report-source.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
    (output/'RESULT.md').write_text(render(value))


if __name__=='__main__':
    main()
