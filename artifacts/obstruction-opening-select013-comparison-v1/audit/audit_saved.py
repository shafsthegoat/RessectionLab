"""Independent saved-JSON audit only; no repository imports or binary reads."""
import argparse
import hashlib
import json
import math
import signal
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'build/obstruction-opening-select013-comparison-v1'
OUT = RUN / 'attempt-01'
ENDPOINTS = ('IL64', 'RL8')
ARMS = ('observed_greedy', 'IL64', 'RL8', 'observed_beam')
OCCUPANCY = 'cerebrum_plus_supplied_tumor_assumption'
TRAIN = ('ReMIND-008', 'ReMIND-010', 'ReMIND-020', 'ReMIND-025')
# Exact reviewed runtime at root-frozen HEAD40137fe; audit only after terminal.
PINS = {'comparison_contract.py': '40aa835f8cbe6f47d917cf19e4c578ad9c7b1e1b157086a4936f0bacc25c4797', 'select_worker.py': 'e61b298a62379bac05886f476d7ebb8ee2b6ee2d544d1f690e1030f25662a98c', 'run_owned.py': '1129e9f2f92177e9985b3f782c622c01357cb5bfd65cd38334ef5ef6865c00b4', 'freeze-runtime.py': '6f6b0568b885619ea475f6b13823609cb07c8c992b4c30ddf31cef2c83d1403e'}
CHECKS = 0
READS = {}


def need(condition, reason):
    global CHECKS
    CHECKS += 1
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
    unresolved = result.get('status')=='comparison_retains_unresolved_search'
    need(release['status'] == 'released_one_attempt' and release['arms'] == list(ARMS), 'Wrong released comparison')
    need(index['head'] == release['expected_head'], 'Source HEAD binding differs')
    need(len(PINS)==4, 'Reviewed runtime pins are not frozen yet')
    for name, expected in PINS.items():
        need(index['source_files']['build/obstruction-opening-select013-comparison-v1/' + name] == expected,
             'Reviewed source pin differs: ' + name)
    for path, expected in index['source_files'].items():
        source=ROOT/path
        need(not Path(path).is_absolute() and '..' not in Path(path).parts and source.suffix=='.py'
             and source.is_file() and not source.is_symlink() and source.stat().st_size<=2*1024**2, 'Bounded source file required')
        raw=source.read_bytes();need(hashlib.sha256(raw).hexdigest()==expected,'Source bytes changed: '+path)
        READS[path]={'sha256':expected,'bytes':len(raw),'kind':'source_only'}
    for path, expected in index['metadata_files'].items():read(path,expected)
    need(parent['release_sha256'] == release_sha and parent['source_index'] == release['source_index'], 'Parent release/index differs')
    need(parent['result_sha256'] == digest(OUT/'result.json') == worker['canonical_result_sha256'], 'Result terminal digest differs')
    if not unresolved:need(worker['result_sha256']==digest(OUT/'result.json'),'Complete worker result differs')
    need(parent['worker_final_sha256'] == digest(OUT.with_name('attempt-01.supervision')/'worker-final.json'), 'Worker terminal digest differs')
    need(parent['worker_termination_confirmed'] is True and not parent['cleanup_errors']
         and not parent['final_owned_pids'], 'Owned process termination/cleanup incomplete')
    if unresolved:
        need(parent['status']=='failed_or_unresolved' and parent['exit_code']==1
             and parent['stop_reason']=='final_guard:ValueError:Incomplete public SELECT comparison'
             and worker['status']=='failed_or_unresolved'
             and worker['failure']=={'type':'ValueError','message':'Comparison has unresolved arms; no complete comparison claim'},
             'Unresolved comparison terminal failure is not exactly explained by local beam cap')
        need(result['arms']['observed_beam']['status']=='search_unresolved'
             and all(result['arms'][arm]['status']=='complete_replayed' for arm in ARMS if arm!='observed_beam'),
             'Only final beam may be unresolved in this branch')
    else:
        need(parent['status']=='complete' and parent['exit_code']==0 and parent['stop_reason'] is None
             and worker['status']=='complete_owned_union_obstruction_SELECT013'
             and result['status']=='complete_union_obstruction_SELECT013_comparison','Comparison terminal mismatch')
    need(parent['elapsed_seconds'] < 960 and parent['sampled_peak_rss_bytes'] <= 3*1024**3
         and parent['output_bytes'] <= 128*1024**2 and parent['samples'] > 0, 'Owned resource cap exceeded')
    need(result['planned_SELECT_denominator'] == 2 and result['held_cases'] == ['ReMIND-037']
         and result['held_inputs_opened'] is False and result['EVAL_opened'] is False
         and result['private_reference_reads'] == result['optimizer_updates_on_SELECT']
         == result['optimizer_attempts'] == result['gradient_attempts'] == 0, 'Role/gradient boundary differs')
    need(result['subject']=='ReMIND-013' and result['role']=='SELECT' and result['occupancy_condition']==OCCUPANCY
         and result['full_supplied_target_voxels']==35260 and result['raw_support_unsupported_target_voxels']==33681
         and result['derived_material_assumption'] is True and release['occupancy_condition']==OCCUPANCY, 'Exact declared union condition required')
    need(worker['runtime']['threads']==worker['runtime']['interop_threads']==1
         and worker['runtime']['device']=='cpu' and worker['runtime']['dtype']=='float32'
         and worker['runtime']['deterministic_algorithms'] is True,'Recorded inference runtime differs')
    need(result['checkpoint_loads'] == 2 and result['arm_order'] == list(ARMS) and set(result['arms']) == set(ARMS), 'Predetermined arm/load count differs')
    public = pinned(release['public_index'])
    need(public['planned_case_denominator'] == 2 and public['cases'][0]['status'] == 'PUBLIC_PARTIAL_TARGET_CONDITION_QUALIFIED'
         and public['cases'][1]['status'] == 'HOLD_ENCODED_SOURCE_SUPPORT_ARTIFACT'
         and public['cases'][1]['path'] is None and public['cases'][1]['replacement'] is None, 'QC admission/held status differs')
    public_manifest = pinned(public['cases'][0])
    need(public_manifest['patient_id'] == 'ReMIND-013' and public_manifest['role'] == 'SELECT'
         and set(public_manifest['input_files']) == {'image','supplied_support','supplied_whole_tumor','whole_tumor_domain'}
         and public_manifest['private_evaluation_files_included'] is False, 'Public manifest scope differs')
    lineages = {};training_worlds={};initials=[]
    for name in ENDPOINTS:
        refs = release['endpoints'][name]
        tr, rr, pr, wr = [pinned(refs[k]) for k in ('release','result','parent','worker_final')]
        lineage = read(OUT/(name+'-lineage.json'))
        method, updates = ('RL' if name == 'RL8' else 'IL'), (64 if name == 'IL64' else 8)
        need(pr['status'] == 'complete' and pr['exit_code'] == 0 and pr['worker_termination_confirmed'] is True
             and not pr['cleanup_errors'] and not pr['final_owned_pids'], 'Training terminal incomplete: '+name)
        need(pr['release_sha256'] == refs['release']['sha256'] and pr['result_sha256'] == refs['result']['sha256']
             == wr['result_sha256'] == wr['canonical_result_sha256']
             and pr['worker_final_sha256'] == refs['worker_final']['sha256'], 'Training terminal chain differs: '+name)
        need(rr['optimizer_updates'][method] == updates and rr['SELECT_EVAL_opened'] is False
             and rr['private_reference_reads'] == 0, 'Training endpoint scope differs')
        original_contexts={s:pinned(refs['contexts'][s]) for s in TRAIN}
        contexts={('ReMIND:'+s[-3:]):semantic(original_contexts[s]) for s in TRAIN}
        need(all(c['subject']==subject and c['role']=='TRAIN' and c['private_reference_in_task'] is False
             and c['occupancy_condition']==OCCUPANCY for subject,c in original_contexts.items()), 'Original public TRAIN contexts differ')
        execution=tr['learning_protocol']['cohort_execution']
        training_worlds[name]={k:execution[k] for k in ('max_steps','proposal_config','proposal_rule_hash','search','retention_mode','task_condition','occupancy_condition')}
        training_worlds[name]['public_target_context_variant']=tr['learning_protocol']['public_target_context_variant']
        initials.append(rr['initial_parameter_hash'])
        need(execution['occupancy_condition']==OCCUPANCY and execution['max_steps']==24
             and execution['proposal_config']['max_candidates']==120 and execution['proposal_config']['obstruction_opening'] is True
             and execution['proposal_config']['tool_footprint_opening'] is True,'Matched TRAIN world differs')
        checkpoint = rr['checkpoints'][method]
        need(checkpoint == pr['checkpoints'][method] and checkpoint['sha256'] == refs['checkpoint']['sha256']
             == lineage['checkpoint_sha256'] and checkpoint['parameter_hash'] == lineage['parameter_hash'], 'Checkpoint lineage differs: '+name)
        need(lineage['method'] == method and lineage['completed_updates'] == updates
             and lineage['optimizer_updates_on_SELECT'] == 0 and lineage['training_context_hashes'] == contexts
             and lineage['learning_protocol_hash'] == semantic(tr['learning_protocol'])
             and lineage['training_release_sha256'] == refs['release']['sha256'], 'Reloaded endpoint protocol/context differs')
        lineages[name] = lineage
    need(len(set(initials))==1 and training_worlds['IL64']==training_worlds['RL8'], 'Initial model/world not matched')
    common_world = common_derivation = common_support = None
    selection_costs={}
    arm_summary = {}
    for arm in ARMS:
        directory = OUT/arm
        summary,world,context=[read(directory/name) for name in ('summary.json','comparison-world.json','context.json')]
        derivation=read(directory/'source/public-task-derivation.json')
        if unresolved and arm=='observed_beam':
            partial=read(directory/'search-unresolved.json');accounting=partial['accounting']
            need(partial==summary==result['arms'][arm] and partial['status']=='search_unresolved'
                 and type(partial['partial_actions']) is list and 1<=len(partial['partial_actions'])<=24,
                 'Partial beam accounting must remain exact and separate')
            need((accounting['time_cap_reached'] is True or accounting['call_cap_reached'] is True)
                 and accounting['max_calls']==5640 and accounting['beam_width']==2
                 and 0<=accounting['model_transition_calls']<=5640 and 0<=accounting['completed_layers']<=24
                 and accounting['actor_forward_calls']==0 and accounting['guidance']=='none'
                 and accounting['objective_source']=='permitted_nominal_target_and_frozen_geometric_costs'
                 and math.isfinite(accounting['planning_seconds']) and accounting['planning_seconds']>=0,
                 'Local capped public beam accounting differs')
            if accounting['time_cap_reached']:need(accounting['planning_seconds']>300,'Declared300s beam cap not reached')
            need(world==common_world and derivation==common_derivation
                 and context['subject']=='ReMIND-013' and context['role']=='SELECT'
                 and context['max_optimizer_updates']==0 and context['checkpoint_lineage']==lineages['IL64']
                 and context['private_reference_in_task'] is False and context['occupancy_condition']==OCCUPANCY
                 and context['execution_kind']=='frozen_SELECT013_union_obstruction_inference_v1',
                 'Unresolved beam source/context must still match completed public arms')
            for name in ('complete-trace.json','native-replay.json','plan.json','select-replay.json','search-return.json'):
                need(not (directory/name).exists(), 'Capped prefix was promoted to accepted trace/plan: '+name)
            arm_summary[arm]={**partial,'accepted_plan':False,'executed_outcome':None,
                'interpretation':'Saved capped search proposal only; partial actions including any STOP are not an accepted/executed plan'}
            selection_costs[arm]={'planning_seconds':accounting['planning_seconds'],'scope':'capped selector only; no accepted collection/native replay'}
            continue
        trace,replay,plan_record,receipt=[read(directory/name) for name in ('complete-trace.json','native-replay.json','plan.json','select-replay.json')]
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
             and context['private_reference_in_task'] is False and context['occupancy_condition']==OCCUPANCY
             and context['max_steps']==24 and world['occupancy_condition']==OCCUPANCY
             and context['execution_kind']=='frozen_SELECT013_union_obstruction_inference_v1', 'Arm context differs')
        need(derivation['private_reference_used'] is False and derivation['route_search'] is False
             and derivation['public_target_positive_voxels']==35260
             and derivation['unsupported_target_positive_voxels']==33681, 'Private or changed public access derivation')
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
        need(decisions[0]['observation_hash']==world['initial_observation_hash'], 'Initial observation differs from collected input')
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
        need(plan['learning_updates']==(lineage['completed_updates'] if learned else 0)
             and plan['architecture_hash']==(lineage['architecture_hash'] if learned else None)
             and receipt['method']==(lineage['method'] if learned else 'SEARCH')
             and receipt['checkpoint_lineage_scope']==('executed_greedy_policy' if learned else 'comparison_reference_only_not_action_author')
             and receipt['private_reference_reads']==0 and receipt['EVAL_opened'] is False
             and receipt['patient_adaptation'] is False,'No SELECT learning or search policy authorship allowed')
        expected_trace = {'context': semantic(context), 'observations': [d['observation_hash'] for d in decisions],
            'history': trace['metrics']['history'], 'checkpoint_lineage': lineage,
            'method': lineage['method'] if learned else 'SEARCH', 'behavior_parameter_hash': parameter}
        need(semantic(expected_trace) == trace['trace_seal'] == receipt['trace_seal'], 'Trace seal differs')
        if not learned:
            search = read(directory/'search-return.json');selection=read(directory/'selection.json')
            accounting=search['accounting']
            need(search['actions']==plan['actions']==selection['actions'] and accounting==selection['accounting']==summary['selector_accounting'],
                 'Selector result differs from completely replayed action sequence')
            need(search['behavior_parameter_hash'] is None and search['checkpoint_lineage_scope']=='comparison_reference_only_not_action_author',
                 'Search falsely attributed to checkpoint policy')
            if arm=='observed_beam':
                need(accounting['call_cap_reached'] is False and accounting['time_cap_reached'] is False
                     and accounting['max_calls']==5640 and accounting['beam_width']==2
                     and accounting['objective_source']=='permitted_nominal_target_and_frozen_geometric_costs',
                     'Capped beam cannot become an accepted complete plan')
            else:
                rows=accounting['decisions'];scored=0;estimated=0.
                need(accounting['method']=='observed_greedy' and accounting['complete'] is True
                     and accounting['initial_steps']==0 and accounting['max_steps']==24
                     and accounting['model_transition_calls']==len(decisions)==len(rows)
                     and accounting['native_replay_required'] is True and accounting['global_optimality_proven'] is False
                     and accounting['objective_source']=='permitted_nominal_target_and_frozen_geometric_costs'
                     and 0<=accounting['planning_seconds']<=accounting['time_budget_seconds']==60,
                     'Native greedy completion/accounting differs')
                for step,(row,decision) in enumerate(zip(rows,decisions)):
                    scores=row['scores'];legal=[a for a,m in zip(decision['action_ids'],decision['action_mask']) if m]
                    need(row['step']==step and row['selected_action_id']==decision['action_id']
                         and row['all_current_legal_actions_scored'] is True
                         and row['legal_nonstop_actions']==row['scored_nonstop_actions']==len(scores)-1
                         and [r['action_id'] for r in scores]==legal and scores[0]['action_id']=='STOP'
                         and scores[0]['reward']==0.,'Greedy score inventory differs')
                    chosen=max(scores,key=lambda r:r['reward'])
                    need(chosen['action_id']==decision['action_id'] and chosen['reward']==decision['reward'],
                         'Greedy action/reward differs from stable public immediate winner')
                    scored+=row['scored_nonstop_actions'];estimated+=chosen['reward']
                need(accounting['evaluated_nonstop_actions']==scored and close(accounting['estimated_incremental_return'],estimated),
                     'Greedy scalar accounting differs')
            need(math.isfinite(selection['wall_seconds']) and selection['wall_seconds']>=0
                 and type(selection['native_previews']) is int and selection['native_previews']>=0,'Invalid selector costs')
            selection_costs[arm]={k:selection[k] for k in ('wall_seconds','native_previews','scope')}
        need(close(sum(d['reward'] for d in decisions), summary['public_return'])
             and close(sum(r.get('target_removed_mm3',0.) for r in metrics['history']), summary['target_removed_mm3'])
             and close(sum(r.get('normal_removed_mm3',0.) for r in metrics['history']), summary['outside_supplied_target_removed_mm3']), 'Saved totals differ')
        goal = metrics['supplied_goal_region']
        occupancy=goal['occupancy_derivation']
        need(goal['full_region_positive_voxels']==35260 and goal['unsupported_region_positive_voxels']==0
             and goal['target_modified'] is False and goal['occupancy_modified'] is True
             and goal['derived_occupancy'] is True and goal['occupancy_condition']==OCCUPANCY
             and goal['fraction_denominator']=='entire_unchanged_supplied_region'
             and occupancy['operation']=='S OR (T > 0)' and occupancy['condition']==OCCUPANCY
             and occupancy['added_region_positive_voxels']==33681 and occupancy['source_support_unchanged'] is True
             and occupancy['target_unchanged'] is True and occupancy['anatomical_or_material_validation'] is False,
             'Raw support deficit/full target/explicit derived occupancy changed')
        need(close(goal['fraction_of_full_region_removed'],summary['target_removed_mm3']/goal['full_region_membership_mm3']),
             'Whole supplied target denominator differs')
        arm_summary[arm] = {k: summary[k] for k in ('steps','actions','public_return','target_removed_mm3','outside_supplied_target_removed_mm3')}
        arm_summary[arm].update(independent_geometry_accepted=True,full_target_fraction=goal['fraction_of_full_region_removed'],
            terminal_reason=plan['terminal_reason'],checkpoint_author=summary['checkpoint_author'])
    costs = read(OUT/'costs.json')
    forwards = sum(arm_summary[n]['steps'] for n in ENDPOINTS)
    budget = result['native_budget']
    need(result['total_policy_forward_calls'] == costs['policy_forwards'] == forwards <= 48, 'Forward accounting differs')
    need(all(budget[k] == costs['native_budget'][k] for k in
             ('native_preview_entries','counting_reliable','failure','blocked_preview_attempts','limits','status'))
         and budget['counting_reliable'] is True
         and budget['failure'] is None and budget['blocked_preview_attempts'] == 0
         and budget['native_preview_entries'] <= 750000, 'Native budget incomplete/capped')
    need(sum(r.get('policy_forward_calls',0) for r in costs['phases'].values())==forwards, 'Phase forward count differs')
    for arm in ARMS:
        need(arm+'.public_construction' in costs['phases'] and arm+'.planning_and_replay' in costs['phases'], 'Missing charged method phase')
    return {'decision':'PASS_SAVED_UNRESOLVED_BEAM_ACCOUNTING' if unresolved else 'PASS_SAVED_UNION_OBSTRUCTION_SELECT013_COMPARISON',
        'complete_comparison':not unresolved,'unresolved_arms':['observed_beam'] if unresolved else [],
        'parent_status':parent['status'],'parent_exit_code':parent['exit_code'],'automatic_retry':False, 'arms':arm_summary,
        'world':common_world, 'planned_SELECT_denominator':2, 'held_cases':['ReMIND-037'],
        'policy_forwards':forwards, 'native_previews':budget['native_preview_entries'],
        'wall_seconds':parent['elapsed_seconds'], 'sampled_peak_rss_bytes':parent['sampled_peak_rss_bytes'],
        'checkpoint_loads':2, 'optimizer_updates_on_SELECT':0,'executed_SELECT_cases':1,'planned_SELECT_cases':2,
        'checks':CHECKS,'selector_costs':selection_costs,'cost_phases':costs['phases'],
        'occupancy_condition':OCCUPANCY,'raw_unsupported_target_voxels':33681,'full_supplied_target_voxels':35260,
        'information_scope':'same permitted arrays, physical task, initial inventory and public-target context; actor features and planner computation differ, so no identical-representation claim',
        'limits':'Saved receipts and reviewed source; no independent syscall trace, tensor reload, array reread or geometric recomputation. Explicit unvalidated S-union-T geometric world; no clinical/material validation, success threshold or equal-cost claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30s saved-only audit cap')))
    signal.alarm(30)
    try:
        result = audit(args.release, args.release_sha256)
    except Exception as error:
        result = {'decision':'HOLD_SAVED_RESULT_MISMATCH_OR_INCOMPLETE', 'reason':type(error).__name__+': '+str(error)}
    signal.alarm(0)
    result.update(audit_seconds=time.monotonic()-started,checks=CHECKS,scope='saved_JSON_and_source_only', evidence_files=READS,
                  no_model_or_patient_array_imports=True, release_sha256=args.release_sha256)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'evidence_files'}, indent=2))
    return 0 if result['decision'].startswith('PASS_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
