"""Read completed JSON evidence only; no scientific imports, arrays or weights.

Run only after root supplies the terminal RL receipt hash. This is a compact
report projection, not a replacement admission or independent geometry audit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUBJECTS = ('ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045')
IL = Path('build/post-exposure-learning-v1/IL64/attempt-01')
RL = Path('build/post-exposure-learning-rl8-recovery-v1/RL8/attempt-01')
RECOVERY = Path('build/post-exposure-IL045-evaluation-recovery-v1/attempt-01')
DIAGNOSTIC = Path('build/post-exposure-IL045-evaluation-diagnostic-v1/attempt-01')
SEARCH = Path('build/remind-post-exposure-result-integration-v1/summary.json')
SEARCH_SHA = '56e9386b6203825e141f684079028d6466b2d5a5b5d50f88ec755752a946c30b'
PINS = {
    str(IL/'result.json'): 'eedc4857c184d94913976cb5b59e01341e1b3a8bb0f001af43f8d079e3d87975',
    str(IL.with_name('attempt-01.supervision')/'receipt.json'): 'c9702883bcbdb9fb524a99b88fde1cac464116d399fce8aa63d866b57b4a1c82',
    str(RECOVERY/'result.json'): '432535613f699604a4b0f3b0aaa0fedebbe932e5ca6b7006beedd66ff823a7ce',
    str(RECOVERY/'native-replay.json'): '529405a10f37c12f9ee4138fd0ef0d0824d7c3fba6f877802890971cc4bae862',
    str(RECOVERY.with_name('attempt-01.supervision')/'receipt.json'): 'c159b91ab2015dfd54dee677eb683b27e335a62657ba51838c398ed2d391a98e',
    str(DIAGNOSTIC/'result.json'): '2a6654ccc3855c180c2724cc9eb24d5720d3120a5a61bfd84e9ef416ba7304cc',
    str(DIAGNOSTIC.with_name('attempt-01.supervision')/'receipt.json'): '7602b33872a012b24e2c8cec10ef1f5a5bd6b53fd6c7e979678aaef6f7d499e7',
    str(SEARCH): SEARCH_SHA,
    'build/learned-deployment-cost-audit-v1/scalars.json': '10472987b4f018d5dab8594b2d7a5caeb495f2f3bb0a5fc7d354f7eb835c3d79',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rl-receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read_pins = {}

    def read(relative, expected=None, optional=False):
        relative = str(relative); path = ROOT/relative
        if optional and not path.exists():
            return None
        if path.is_symlink() or path.suffix != '.json' or not 0 < path.stat().st_size <= 8*1024**2:
            raise ValueError('Bounded regular JSON metadata required: '+relative)
        raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
        expected = expected or PINS.get(relative)
        if expected is not None and digest != expected:
            raise ValueError('Saved metadata changed: '+relative)
        read_pins[relative] = {'sha256': digest, 'bytes': len(raw)}
        return json.loads(raw)

    def receipt(base, expected=None):
        value = read(base.with_name(base.name+'.supervision')/'receipt.json', expected)
        if (value['status'] not in ('complete', 'failed_or_unresolved')
                or value['exit_code'] is None or value['worker_termination_confirmed'] is not True
                or value.get('final_owned_pids', value.get('remaining_owned_pids'))
                or value['cleanup_errors']):
            raise ValueError('Terminated and cleanly reaped parent required; no live/partial snapshot')
        return value

    # The terminal RL boundary is checked first. No active-run polling is supported.
    rl_parent = receipt(RL, args.rl_receipt_sha256)
    if rl_parent['release_sha256'] != '4da18b8bd21e6a54308d0b58b3d234c49c7351b56040e448ac7107a28a773e3e':
        raise ValueError('Different RL experiment release')
    rl_result = read(RL/'result.json', rl_parent.get('result_sha256'), optional=True)
    il_parent = receipt(IL); il_result = read(IL/'result.json', il_parent['result_sha256'])
    rec_parent = receipt(RECOVERY); recovered = read(RECOVERY/'result.json', rec_parent['result_sha256'])
    diag_parent = receipt(DIAGNOSTIC); diagnosed = read(DIAGNOSTIC/'result.json', diag_parent['result_sha256'])
    if (il_parent['status'] != 'failed_or_unresolved' or il_result['status'] != 'failed_or_unresolved'
            or il_result['optimizer_updates'] != {'IL':64, 'RL':0}
            or recovered['independent_accepted'] is not True or recovered['full_history_equal'] is not True
            or recovered['original_IL_result_sha256'] != PINS[str(IL/'result.json')]
            or recovered['original_IL_parent_receipt_sha256'] != PINS[str(IL.with_name('attempt-01.supervision')/'receipt.json')]
            or recovered['checkpoint_sha256'] != il_result['checkpoints']['IL']['sha256']
            or recovered['parameter_hash'] != il_result['checkpoints']['IL']['parameter_hash']):
        raise ValueError('Exact failed IL and separate accepted same-checkpoint recovery required')
    search = read(SEARCH)
    if [row['patient_id'] for row in search['cases']] != list(SUBJECTS):
        raise ValueError('Fixed four-case denominator changed')

    def learned_route(base, method, subject, recovery=False):
        folder = base/'TRAIN-greedy'/method/subject
        plan_record = read(folder/'plan.json', optional=True)
        path = RECOVERY/'native-replay.json' if recovery else folder/'native-replay.json'
        replay = read(path, optional=True)
        plan = None if plan_record is None else plan_record['plan']
        if replay is None:
            return {'evaluation_completed': False, 'geometry_accepted': None,
                'status': 'unresolved_or_not_reached', 'actions': None if plan is None else plan['actions'],
                'terminal_reason': None if plan is None else plan['terminal_reason'], 'certified_outcomes': None}
        audit = replay['independent_geometry']; metrics = replay['metrics']; history = metrics['history']
        identity = lambda rows:[{k:v for k,v in row.items() if k != 'outcome_scope'} for row in rows]
        if plan is None or identity(plan['history']) != identity(history) or plan['source_hash'] != metrics['source_hash']:
            raise ValueError('Saved learned plan/replay identity changed: '+subject)
        unknowns = set(metrics.get('unknowns', ()))
        for action in history:
            unknowns.update(action.get('unknowns', ()))
            unknowns.update(action.get('geometry_unknowns', ()))
        outcomes = audit.get('outcomes') if audit['accepted'] else None
        return {'evaluation_completed': bool(audit['complete_episode']), 'geometry_accepted': audit['accepted'],
            'status': 'accepted_separate_corrected_checker_recovery' if recovery else ('accepted' if audit['accepted'] else 'evaluated_rejected'),
            'decisions': len(plan['actions']), 'motions': sum(a != 'STOP' for a in plan['actions']),
            'actions': plan['actions'], 'terminal_reason': plan['terminal_reason'],
            'source_hash': plan['source_hash'], 'decision_model_hash': plan['decision_model_hash'],
            'plan_seal': plan_record['plan_seal'], 'parameter_hash': plan['parameter_hash'],
            'certified_outcomes': outcomes, 'full_tool_checked': audit['geometry']['complete_tool_checked'],
            'geometry_failures': audit['geometry']['failures'], 'unknowns': sorted(unknowns),
            'independent_replay_seconds_nested_in_execution': audit['evaluation_seconds'],
            'rejected_claims_not_certified': None if audit['accepted'] else {
                k:metrics.get(k) for k in ('target_removed_mm3','normal_removed_mm3','total_reward')},
            'contact_interpretation': 'Recorded partial active-tip contact upper bounds; full-tool certificate checks shaft/hard geometry. No force, injury or extracranial validation.'}

    rows = []
    for prior in search['cases']:
        subject = prior['patient_id']; il = learned_route(IL, 'IL', subject, subject == 'ReMIND-045')
        rl = learned_route(RL, 'RL', subject)
        search_metrics = read(Path('build/remind-post-exposure-feasibility-v1/attempt-01')/subject/'episode-metrics.json')
        search_unknowns = set(search_metrics.get('unknowns', ()))
        search_unknowns.update(prior['geometry_unknowns'])
        for action in search_metrics['history']:
            search_unknowns.update(action.get('unknowns', ()))
            search_unknowns.update(action.get('geometry_unknowns', ()))
        baseline = {'evaluation_completed': True, 'geometry_accepted': prior['independent_accepted'],
            'decisions': prior['action_count_including_STOP'], 'motions': prior['outcomes']['nonstop_actions'],
            'actions': prior['actions'], 'terminal_reason': 'STOP', 'certified_outcomes': prior['outcomes'],
            'source_hash': prior['source_hash'], 'decision_model_hash': search_metrics['decision_model_hash'],
            'unknowns': sorted(search_unknowns), 'full_tool_checked': prior['independent_accepted']}
        for method in (il, rl):
            if method['evaluation_completed']:
                method['same_source_and_decision_world_as_search'] = (
                    method['source_hash'] == baseline['source_hash']
                    and method['decision_model_hash'] == baseline['decision_model_hash'])
                if method['geometry_accepted']:
                    denominator = method['certified_outcomes']['total_reference_target_mm3']
                    if not math.isclose(denominator, prior['outcomes']['total_reference_target_mm3'], rel_tol=1e-12, abs_tol=1e-9):
                        raise ValueError('Full target denominator changed')
                    method['return_delta_from_search'] = method['certified_outcomes']['total_reward']-prior['outcomes']['total_reward']
                    method['exact_action_sequence_match'] = method['actions'] == prior['actions']
        rows.append({'patient_id':subject, 'role':'TRAIN', 'SEARCH':baseline, 'IL64':il, 'RL8':rl,
            'source_knownness_and_world_assumptions':prior['source_support_knownness_and_assumption']})

    def endpoint_costs(base, method, result, parent):
        saved = read(base/'costs.json', optional=True)
        if saved is None:
            return {'parent':parent, 'worker_costs_unavailable':True}
        phases = saved['costs']
        groups = {'fixed_teacher_reconstruction_and_replay':{}, 'common_initialization':{},
            'training_updates_including_RL_collection_and_replay':{}, 'checkpoint_save_reload':{},
            'endpoint_teacher_readout_including_RL_recollection_and_replay':{},
            'deployment_source_reconstruction':{}, 'deployment_collection_plus_authoritative_replay':{},
            'nested_components_not_additive':{}, 'other_recorded_scopes':{}}
        for key, value in phases.items():
            if key.startswith('offline.endpoint_teacher_logits.') or (method == 'RL' and '.action_backward.' in key):
                group = 'nested_components_not_additive'
            elif key.startswith('offline.fixed_teacher_replay.'):
                group = 'fixed_teacher_reconstruction_and_replay'
            elif key == 'offline.common_initialization': group = 'common_initialization'
            elif key.startswith('offline.'+method+'.update-'): group = 'training_updates_including_RL_collection_and_replay'
            elif key.endswith('.checkpoint_save_reload'): group = 'checkpoint_save_reload'
            elif key.startswith(('offline.endpoint_cached_teacher_readout.', 'offline.endpoint_teacher_recollection.')):
                group = 'endpoint_teacher_readout_including_RL_recollection_and_replay'
            elif key.startswith('deployment.reloaded_TRAIN_greedy_replay.'):
                group = 'deployment_source_reconstruction' if key.endswith('.public_reconstruction') else 'deployment_collection_plus_authoritative_replay'
            else: group = 'other_recorded_scopes'
            groups[group][key] = value
        group_walls = {k:sum(v.get('complete_wall_seconds',0.) for v in group.values())
            for k,group in groups.items() if k != 'nested_components_not_additive'}
        return {'parent_status':parent['status'], 'parent_wall_seconds':parent['elapsed_seconds'],
            'sampled_peak_RSS_bytes':parent['sampled_peak_rss_bytes'], 'worker_wall_seconds':saved['complete_wall_seconds'],
            'actual_counters':None if result is None else {k:result.get(k) for k in (
                'optimizer_updates','loss_forward_calls','teacher_logit_forwards','total_policy_forward_calls',
                'native_preview_entries','completed_source_visits','teacher_trace_reuses')},
            'outer_scope_wall_seconds':group_walls, 'recorded_phase_details':groups,
            'deployment_policy_selection_vs_authoritative_replay_seconds':None,
            'split_limit':'Original worker times policy collection and authoritative replay together. Independent geometry seconds are nested in that combined scope. No invented split or sum of nested inventory/transition/forward timers.'}

    summary = {'status':'post_terminal_saved_metadata_comparison', 'rows':rows,
        'condition':search['condition'], 'fixed_denominator':list(SUBJECTS),
        'IL64_original_attempt_status':il_parent['status'], 'IL64_original_failure':il_result['failure'],
        'IL64_interpretation':'64 updates/checkpoint preserved; three original accepted endpoint routes plus exact045 separately accepted under corrected independent angle checker. Original failed attempt is never relabeled complete.',
        'RL8_attempt_status':rl_parent['status'], 'RL8_result_status':None if rl_result is None else rl_result.get('status'),
        'RL8_failure':None if rl_result is None else rl_result.get('failure'),
        'fresh_initial_parameters_equal':None if not rl_result or not rl_result.get('initial_parameter_hash') else rl_result['initial_parameter_hash'] == il_result['initial_parameter_hash'],
        'endpoint_teacher_metrics':{'IL64':il_result['endpoint_teacher_metrics'], 'RL8':None if rl_result is None else rl_result.get('endpoint_teacher_metrics')},
        'costs':{'SEARCH':read('build/learned-deployment-cost-audit-v1/scalars.json')['new_four'],
            'IL64':endpoint_costs(IL,'IL',il_result,il_parent), 'RL8':endpoint_costs(RL,'RL',rl_result,rl_parent),
            'additional_IL045_negative_diagnostic':{'parent_wall_seconds':diag_parent['elapsed_seconds'],'worker_seconds':diagnosed['elapsed_seconds'],'native_previews':diagnosed['native_previews'],'forwards':0,'updates':0,'accepted':False},
            'additional_IL045_corrected_checker_recovery':{'parent_wall_seconds':rec_parent['elapsed_seconds'],'worker_seconds':recovered['elapsed_seconds'],'native_previews':recovered['native_previews'],'forwards':0,'updates':0,'accepted':True}},
        'limits':['TRAIN-only fitted comparison; no SELECT/EVAL or population generalization claim.',
            'Shared permitted world/reward; search sees full public target/grid while actor uses64-cube plus lossy summaries.',
            'Declared S OR T and post-exposure workspace, preserved Ds/fullT; not anatomical/material/clinical validation.',
            'Source-zero labels do not certify air. External E, outside-image extent and transfer between withdrawn poses remain unassessed.',
            'Outside supplied-target removal is not verified normal-tissue injury; contact upper bounds are not force or injury.',
            'Methods have unequal training exposure, trajectories and compute; no controlled latency advantage inferred.'],
        'metadata_inputs':read_pins, 'source_arrays_opened':0, 'checkpoint_payloads_opened':0,
        'native_calls':0, 'model_forwards':0, 'training_updates':0}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(summary,stream,indent=2,allow_nan=False);stream.write('\n')
    print(str(args.output))


if __name__ == '__main__':
    main()
