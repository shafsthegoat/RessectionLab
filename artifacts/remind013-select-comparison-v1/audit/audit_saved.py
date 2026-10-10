"""Independent saved-JSON audit only; no repository imports or binary reads."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'build/remind013-select-comparison-v1'
OUT = RUN / 'attempt-01'
ENDPOINTS = ('IL8_unweighted', 'RL8', 'IL8_balanced', 'IL64_balanced')
ARMS = ('observed_beam', 'observed_greedy') + ENDPOINTS
TRAIN = ('ReMIND-008', 'ReMIND-010', 'ReMIND-020', 'ReMIND-025')
PINS = {
    'comparison_contract.py': 'da24142acd1548d7d3e2109727c86d99f97ae3a13bbf3920e62b9d935e965805',
    'select_worker.py': 'd9be3261decf65a977dbf3ee07538b5be596979dc7c8cc4ea25c74837dcc9715',
    'run_owned.py': '3efa7070c9f8d1b1a7cc987b50f685e4d1efec814e935147b7f2fb7cfb967226',
    'freeze-runtime.py': '9a0784e64f18dfa199e4d1ae45b60c9c64891c73032e17e1051b31f7ed4b906c',
}
READS = {}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def semantic(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


def read(path, expected=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    relative = path.relative_to(ROOT)
    need('..' not in relative.parts and path.suffix == '.json', 'Only workspace JSON allowed')
    need(not path.is_symlink() and path.is_file(), 'Missing/nonregular saved JSON: ' + str(relative))
    need(0 < path.stat().st_size <= 8*1024**2, 'Saved JSON exceeds individual bound')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    need(expected is None or digest == expected, 'Saved JSON digest mismatch: ' + str(relative))
    READS[str(relative)] = {'sha256': digest, 'bytes': len(raw)}
    need(sum(r['bytes'] for r in READS.values()) <= 128*1024**2, 'Metadata read bound exceeded')
    return json.loads(raw)


def pinned(ref):
    return read(ref['path'], ref['sha256'])


def digest(path):
    return READS[str(Path(path).relative_to(ROOT))]['sha256']


def history_identity(rows):
    return semantic([{k: v for k, v in row.items() if k != 'outcome_scope'} for row in rows])


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


def audit(release_path, release_sha):
    release = read(release_path, release_sha)
    parent = read(OUT.with_name('attempt-01.supervision') / 'receipt.json')
    worker = read(OUT.with_name('attempt-01.supervision') / 'worker-final.json')
    result = read(OUT / 'result.json')
    index = pinned(release['source_index'])
    need(release['status'] == 'released_one_attempt' and release['arms'] == list(ARMS), 'Wrong released comparison')
    need(index['head'] == release['expected_head'], 'Source HEAD binding differs')
    for name, expected in PINS.items():
        need(index['source_files']['build/remind013-select-comparison-v1/' + name] == expected,
             'Reviewed source pin differs: ' + name)
    need(parent['release_sha256'] == release_sha and parent['source_index'] == release['source_index'], 'Parent release/index differs')
    need(parent['result_sha256'] == digest(OUT/'result.json') == worker['result_sha256'] == worker['canonical_result_sha256'], 'Result terminal digest differs')
    need(parent['worker_final_sha256'] == digest(OUT.with_name('attempt-01.supervision')/'worker-final.json'), 'Worker terminal digest differs')
    need(parent['status'] == 'complete' and parent['exit_code'] == 0 and parent['stop_reason'] is None
         and parent['worker_termination_confirmed'] is True and not parent['cleanup_errors']
         and not parent['final_owned_pids'], 'Parent incomplete/capped/unclean: ' + str(parent.get('stop_reason')))
    need(worker['status'] == 'complete_owned_SELECT013' and result['status'] == 'complete_public_SELECT013_comparison', 'Comparison incomplete')
    need(parent['elapsed_seconds'] < 960 and parent['sampled_peak_rss_bytes'] <= 3*1024**3
         and parent['output_bytes'] <= 128*1024**2 and parent['samples'] > 0, 'Owned resource cap exceeded')
    need(result['planned_SELECT_denominator'] == 2 and result['held_cases'] == ['ReMIND-037']
         and result['held_inputs_opened'] is False and result['EVAL_opened'] is False
         and result['private_reference_reads'] == result['optimizer_updates_on_SELECT']
         == result['optimizer_attempts'] == result['gradient_attempts'] == 0, 'Role/gradient boundary differs')
    need(result['checkpoint_loads'] == 4 and result['arm_order'] == list(ARMS) and set(result['arms']) == set(ARMS), 'Predetermined arm/load count differs')
    public = pinned(release['public_index'])
    need(public['planned_case_denominator'] == 2 and public['cases'][0]['status'] == 'PUBLIC_PARTIAL_TARGET_CONDITION_QUALIFIED'
         and public['cases'][1]['status'] == 'HOLD_ENCODED_SOURCE_SUPPORT_ARTIFACT'
         and public['cases'][1]['path'] is None and public['cases'][1]['replacement'] is None, 'QC admission/held status differs')
    public_manifest = pinned(public['cases'][0])
    need(public_manifest['patient_id'] == 'ReMIND-013' and public_manifest['role'] == 'SELECT'
         and set(public_manifest['input_files']) == {'image','supplied_support','supplied_whole_tumor','whole_tumor_domain'}
         and public_manifest['private_evaluation_files_included'] is False, 'Public manifest scope differs')
    lineages = {}
    for name in ENDPOINTS:
        refs = release['endpoints'][name]
        tr, rr, pr, wr = [pinned(refs[k]) for k in ('release','result','parent','worker_final')]
        lineage = read(OUT/(name+'-lineage.json'))
        method, updates = ('RL' if name == 'RL8' else 'IL'), (64 if name == 'IL64_balanced' else 8)
        need(pr['status'] == 'complete' and pr['exit_code'] == 0 and pr['worker_termination_confirmed'] is True
             and not pr['cleanup_errors'] and not pr['final_owned_pids'], 'Training terminal incomplete: '+name)
        need(pr['release_sha256'] == refs['release']['sha256'] and pr['result_sha256'] == refs['result']['sha256']
             == wr['result_sha256'] == wr['canonical_result_sha256']
             and pr['worker_final_sha256'] == refs['worker_final']['sha256'], 'Training terminal chain differs: '+name)
        need(rr['optimizer_updates'][method] == updates and rr['SELECT_EVAL_opened'] is False
             and rr['private_reference_reads'] == 0, 'Training endpoint scope differs')
        contexts = {('ReMIND:'+s[-3:]): semantic(pinned(refs['contexts'][s])) for s in TRAIN}
        checkpoint = rr['checkpoints'][method]
        need(checkpoint == pr['checkpoints'][method] and checkpoint['sha256'] == refs['checkpoint']['sha256']
             == lineage['checkpoint_sha256'] and checkpoint['parameter_hash'] == lineage['parameter_hash'], 'Checkpoint lineage differs: '+name)
        need(lineage['method'] == method and lineage['completed_updates'] == updates
             and lineage['optimizer_updates_on_SELECT'] == 0 and lineage['training_context_hashes'] == contexts
             and lineage['learning_protocol_hash'] == semantic(tr['learning_protocol'])
             and lineage['training_release_sha256'] == refs['release']['sha256'], 'Reloaded endpoint protocol/context differs')
        lineages[name] = lineage
    common_world = common_derivation = common_support = None
    arm_summary = {}
    for arm in ARMS:
        directory = OUT/arm
        summary, world, context, trace, replay, plan_record, receipt = [read(directory/name) for name in
            ('summary.json','comparison-world.json','context.json','complete-trace.json','native-replay.json','plan.json','select-replay.json')]
        derivation = read(directory/'source/public-task-derivation.json')
        plan, metrics, geometry = plan_record['plan'], replay['metrics'], replay['independent_geometry']
        selected = arm if arm in ENDPOINTS else ENDPOINTS[0]
        lineage = lineages[selected]
        need(summary == result['arms'][arm] and summary['status'] == 'complete_replayed', 'Arm not completely replayed: '+arm)
        need(common_world is None or world == common_world, 'Physical task/initial inventory differs: '+arm)
        need(common_derivation is None or derivation == common_derivation, 'Public access/mask derivation differs: '+arm)
        support = metrics['support_provenance']
        need(common_support is None or support == common_support, 'Public mask/source provenance differs: '+arm)
        common_world, common_derivation, common_support = world, derivation, support
        need(context['subject'] == 'ReMIND-013' and context['role'] == 'SELECT'
             and context['max_optimizer_updates'] == 0 and context['checkpoint_lineage'] == lineage
             and context['private_reference_in_task'] is False, 'Arm context differs')
        need(semantic(context) == plan['context_hash'] == trace['context_hash'] == receipt['context_hash'], 'Context seal differs')
        need(semantic(plan) == plan_record['plan_seal'] == summary['plan_seal'] == receipt['plan_seal'], 'Plan seal differs')
        need(plan['source_hash'] == world['source_hash'] == geometry['source_hash'] == metrics['source_hash']
             and plan['decision_model_hash'] == world['decision_model_hash'] == geometry['decision_model_hash']
             and plan['initial_observation_hash'] == world['initial_observation_hash'], 'World/replay identity differs')
        need(history_identity(plan['history']) == history_identity(metrics['history']) == history_identity(trace['metrics']['history']), 'Collected/replayed history differs')
        need(geometry['accepted'] is True and geometry['complete_episode'] is True
             and geometry['geometry']['feasible'] is True and not geometry['geometry']['failures']
             and geometry['geometry']['complete_tool_checked'] is True and geometry['geometry']['frontier_checked'] is True
             and geometry['committed_history_hash'] == semantic(metrics['history']), 'Independent complete native geometry failed')
        decisions = trace['decisions']
        need(1 <= len(decisions) <= 24 and len(decisions) == summary['steps'] == len(plan['actions'])
             and metrics['terminated'] is True and decisions[-1]['terminated'] is True
             and all(d['terminated'] is False for d in decisions[:-1]), 'Incomplete episode')
        need([d['action_id'] for d in decisions] == plan['actions'] == summary['actions'], 'Action sequence differs')
        need((plan['actions'][-1] == 'STOP' and 'STOP' not in plan['actions'][:-1])
             or ('STOP' not in plan['actions'] and len(decisions) == 24), 'Invalid STOP/horizon termination')
        for step, d in enumerate(decisions):
            need(d['step'] == step and d['action_mask'][d['action_ids'].index(d['action_id'])] is True, 'Illegal saved action')
        learned = arm in ENDPOINTS
        parameter = lineage['parameter_hash'] if learned else None
        need(plan['parameter_hash'] == parameter and all(d['behavior_parameter_hash'] == parameter for d in decisions)
             and receipt['behavior_parameter_hash'] == parameter and receipt['checkpoint_lineage'] == lineage
             and receipt['status'] == 'complete' and receipt['optimizer_updates_on_SELECT'] == 0, 'Frozen policy/replay lineage differs')
        expected_trace = {'context': semantic(context), 'observations': [d['observation_hash'] for d in decisions],
            'history': trace['metrics']['history'], 'checkpoint_lineage': lineage,
            'method': lineage['method'] if learned else 'SEARCH', 'behavior_parameter_hash': parameter}
        need(semantic(expected_trace) == trace['trace_seal'] == receipt['trace_seal'], 'Trace seal differs')
        if not learned:
            search = read(directory/'search-return.json')
            need(search['accounting']['call_cap_reached'] is False and search['accounting']['time_cap_reached'] is False
                 and search['actions'] == plan['actions'], 'Capped/changed planner accepted as complete')
        need(close(sum(d['reward'] for d in decisions), summary['public_return'])
             and close(sum(r.get('target_removed_mm3',0.) for r in metrics['history']), summary['target_removed_mm3'])
             and close(sum(r.get('normal_removed_mm3',0.) for r in metrics['history']), summary['outside_supplied_target_removed_mm3']), 'Saved totals differ')
        goal = metrics['supplied_goal_region']
        need(goal['full_region_positive_voxels'] == 35260 and goal['unsupported_region_positive_voxels'] == 33681
             and goal['target_modified'] is False and goal['occupancy_modified'] is False, 'Partial target denominator changed')
        arm_summary[arm] = {k: summary[k] for k in ('steps','actions','public_return','target_removed_mm3','outside_supplied_target_removed_mm3')}
        arm_summary[arm]['independent_geometry_accepted'] = True
    costs = read(OUT/'costs.json')
    forwards = sum(arm_summary[n]['steps'] for n in ENDPOINTS)
    budget = result['native_budget']
    need(result['total_policy_forward_calls'] == costs['policy_forwards'] == forwards <= 96, 'Forward accounting differs')
    need(all(budget[k] == costs['native_budget'][k] for k in
             ('native_preview_entries','counting_reliable','failure','blocked_preview_attempts','limits','status'))
         and budget['counting_reliable'] is True
         and budget['failure'] is None and budget['blocked_preview_attempts'] == 0
         and budget['native_preview_entries'] <= 750000, 'Native budget incomplete/capped')
    return {'decision':'PASS_SAVED_SELECT013_COMPARISON', 'arms':arm_summary,
        'world':common_world, 'planned_SELECT_denominator':2, 'held_cases':['ReMIND-037'],
        'policy_forwards':forwards, 'native_previews':budget['native_preview_entries'],
        'wall_seconds':parent['elapsed_seconds'], 'sampled_peak_rss_bytes':parent['sampled_peak_rss_bytes'],
        'checkpoint_loads':4, 'optimizer_updates_on_SELECT':0,
        'limits':'Saved receipts and reviewed source; no independent syscall trace, tensor reload, array reread, or geometric recomputation. Geometric annotation-conditioned result only; no clinical/physical-surgery claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.release, args.release_sha256)
    except Exception as error:
        result = {'decision':'HOLD_SAVED_RESULT_MISMATCH_OR_INCOMPLETE', 'reason':type(error).__name__+': '+str(error)}
    result.update(scope='saved_JSON_only', evidence_files=READS,
                  no_model_or_patient_array_imports=True, release_sha256=args.release_sha256)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'evidence_files'}, indent=2))
    return 0 if result['decision'].startswith('PASS_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
