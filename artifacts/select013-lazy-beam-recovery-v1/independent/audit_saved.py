"""Independent saved-JSON audit only; no repository imports or binary reads."""
import argparse
import hashlib
import json
import math
import os
import resource
import signal
import stat
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'build/select013-lazy-beam-recovery-v1'
OUT = RUN / 'attempt-01'
ENDPOINTS = ('IL64',)
ARMS = ('observed_beam',)
OCCUPANCY = 'cerebrum_plus_supplied_tumor_with_preserved_partial_source_domain'
EXPOSURE = 'public-union-post-exposure-axis0-v1'
TRAIN = ('ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045')
# Reviewed sources; root supplies actual terminal release/receipt/result pins.
PINS = {'comparison_contract.py':'5f0e15aac211bce9e7817c11dcc2cf435de1d5c04d41e839a79a7e0b41cf2119',
        'select_worker.py':'703b1b53a21d00de197765a50a22454ea5d0bfc0c815ed9ff93e59237b205fc2',
        'run_owned.py':'d4c68ec70a11f83cd234657c4c61f1dd996e00534fe2d611274e03e76300a1bf',
        'freeze-runtime.py':'7522e783b641e410359ffba8bd92f41a0161102e303e5c0d863c5ab3751e1969'}
CHECKS = 0
READS = {}


def need(condition, reason):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise ValueError(reason)
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 512*1024**2:
        raise MemoryError('512 MiB saved audit cap')


def semantic(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


def bytes_read(path, expected=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    relative = path.relative_to(ROOT)
    need('..' not in relative.parts and path.suffix in ('.json','.py'), 'Only workspace JSON/source allowed; no weight/array payload')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(fd)
        need(stat.S_ISREG(info.st_mode) and 0<info.st_size<=8*1024**2, 'Bounded regular evidence')
        with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(8*1024**2+1)
        need(len(raw)==info.st_size and len(raw)<=8*1024**2,'Stable bounded read')
    finally:os.close(fd)
    digest = hashlib.sha256(raw).hexdigest()
    need(expected is None or digest == expected, 'Saved JSON digest mismatch: ' + str(relative))
    need(str(relative) not in READS or READS[str(relative)]['sha256']==digest,'Repeated evidence changed')
    READS[str(relative)] = {'sha256': digest, 'bytes': len(raw)}
    need(sum(r['bytes'] for r in READS.values()) <= 128*1024**2, 'Metadata read bound exceeded')
    return raw


def read(path, expected=None):
    need(Path(path).suffix=='.json','JSON only')
    return json.loads(bytes_read(path,expected),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def pinned(ref):
    return read(ref['path'], ref['sha256'])


def digest(path):
    return READS[str(Path(path).relative_to(ROOT))]['sha256']


def history_identity(rows):
    return semantic([{k: v for k, v in row.items() if k != 'outcome_scope'} for row in rows])


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


def audit(release_path, release_sha, receipt_sha, result_sha):
    for pin in (release_sha,receipt_sha,result_sha):need(len(pin)==64 and all(c in '0123456789abcdef' for c in pin),'Exact terminal pins required')
    parent = read(OUT.with_name('attempt-01.supervision') / 'receipt.json',receipt_sha)
    need(type(parent['exit_code']) is int and parent['worker_termination_confirmed'] is True
         and not parent['cleanup_errors'] and not parent['final_owned_pids'],'Root-bound terminal before other attempt reads')
    release = read(release_path, release_sha)
    worker = read(OUT.with_name('attempt-01.supervision') / 'worker-final.json')
    result = read(OUT / 'result.json',result_sha)
    index = pinned(release['source_index'])
    unresolved = result.get('status')=='failed_or_unresolved'
    need(release['status'] == 'released_one_attempt' and release['arms'] == list(ARMS), 'Wrong released comparison')
    need(index['head'] == release['expected_head']=='fe857eed61ad2fc3cace011bcef523f7b43a1923', 'Source HEAD binding differs')
    need(release_sha=='43e7cdf0e43a014154cc15236bbb4df1818ed1577b3cee5b8d75ac368e9aa3d9'
         and release['source_index']['sha256']=='1a9793219207a94ff3c71f654d50b02c7c196ab4cf3f0198260d4ea45aa258e1','exact root launched experiment')
    need(len(PINS)==4, 'Reviewed runtime pins are not frozen yet')
    for name, expected in PINS.items():
        need(index['source_files']['build/select013-lazy-beam-recovery-v1/' + name] == expected,
             'Reviewed source pin differs: ' + name)
    for path, expected in index['source_files'].items():
        source=ROOT/path
        need(not Path(path).is_absolute() and '..' not in Path(path).parts and source.suffix=='.py'
             and source.is_file() and not source.is_symlink() and source.stat().st_size<=2*1024**2, 'Bounded source file required')
        bytes_read(source,expected)
    excluded_weights={}
    for path, expected in index['metadata_files'].items():
        if Path(path).suffix=='.psckpt':excluded_weights[path]=expected
        else:read(path,expected)
    need(parent['release_sha256'] == release_sha and parent['source_index'] == release['source_index'], 'Parent release/index differs')
    need(parent['result_sha256'] == digest(OUT/'result.json') == worker['canonical_result_sha256'], 'Result terminal digest differs')
    if not unresolved:need(worker['result_sha256']==digest(OUT/'result.json'),'Complete worker result differs')
    need(parent['worker_final_sha256'] == digest(OUT.with_name('attempt-01.supervision')/'worker-final.json'), 'Worker terminal digest differs')
    need(parent['worker_termination_confirmed'] is True and not parent['cleanup_errors']
         and not parent['final_owned_pids'], 'Owned process termination/cleanup incomplete')
    if unresolved:
        need(parent['status']=='failed_or_unresolved' and parent['exit_code']==1
             and worker['status']=='failed_or_unresolved'
             and result.get('failure') is not None,
             'Incomplete attempt must retain a failed terminal and exact reason')
    else:
        need(parent['status']=='complete' and parent['exit_code']==0 and parent['stop_reason'] is None
             and worker['status']=='complete_owned_SELECT013_lazy_beam'
             and result['status']=='complete_SELECT013_lazy_beam_recovery','Comparison terminal mismatch')
    need(parent['elapsed_seconds'] < 960 and parent['sampled_peak_rss_bytes'] <= 3*1024**3
         and parent['output_bytes'] <= 128*1024**2 and parent['samples'] > 0, 'Owned resource cap exceeded')
    need(result['planned_SELECT_denominator'] == 2 and result['held_cases'] == ['ReMIND-037']
         and result['held_inputs_opened'] is False and result['EVAL_opened'] is False
         and result['private_reference_reads'] == result['optimizer_updates_on_SELECT']
         == result['optimizer_attempts'] == result['gradient_attempts'] == 0, 'Role/gradient boundary differs')
    need(result['subject']=='ReMIND-013' and result['role']=='SELECT' and result['occupancy_condition']==OCCUPANCY
         and result['full_supplied_target_voxels']==35260 and result['raw_support_unsupported_target_voxels']==33681
         and result['derived_material_assumption'] is True and release['occupancy_condition']==OCCUPANCY
         and result['post_exposure_condition']==release['post_exposure_condition']==EXPOSURE, 'Exact declared post-exposure union condition required')
    need(worker['runtime']['threads']==worker['runtime']['interop_threads']==1
         and worker['runtime']['device']=='cpu' and worker['runtime']['dtype']=='float32'
         and worker['runtime']['deterministic_algorithms'] is True,'Recorded inference runtime differs')
    need(result['arm_order'] == list(ARMS) and set(result['arms']) == set(ARMS), 'Predetermined arm denominator differs')
    need(result['partial_domain_inputs'] is True and release['partial_domain_inputs'] is True
         and result['public_preparation']==release['public_preparation'], 'Exact partial-domain preparation binding')
    public = pinned(release['public_index'])
    need(release['public_index']['sha256']=='a5a7ed8b504b3f768666cdec4c77908d680ff242eb157afb0f377b95105dc389'
         and public['planned_case_denominator']==2 and public['max_optimizer_updates']==0
         and public['cases'][0]['status']=='EXPANDED_SOURCE_PREPARED_PENDING_SEPARATE_INFERENCE_RELEASE',
         'Exact expanded source index')
    original_public=pinned(public['original_select_ledger'])
    need(public['cases'][1]==original_public['cases'][1]
         and public['cases'][1]['patient_id']=='ReMIND-037'
         and public['cases'][1]['status']=='HOLD_ENCODED_SOURCE_SUPPORT_ARTIFACT'
         and public['cases'][1]['path'] is None and public['cases'][1]['replacement'] is None,'Unchanged held037 denominator')
    public_manifest=pinned(public['cases'][0])
    need(public['cases'][0]['sha256']=='841261035a7d7be5c55984e717b8d65b712065a708e9260231ebcd49cc4d700e'
         and public_manifest['schema']=='remind-fixed-SELECT-partial-domain-public-inputs-v1'
         and public_manifest['patient_id']=='ReMIND-013' and public_manifest['role']=='SELECT'
         and set(public_manifest['input_files'])=={'image','supplied_support','supplied_whole_tumor','whole_tumor_domain','supplied_support_domain'}
         and public_manifest['shape_xyz']==[131,154,117]
         and public_manifest['public_support_domain_fully_covered'] is False
         and public_manifest['private_evaluation_files_included'] is False,'Exact expanded five public arrays; no payload opened')
    preparation=release['public_preparation'];prior=preparation['original_failed_attempt']
    original_result=pinned(prior['bindings']['result']);original_parent=pinned(prior['bindings']['parent'])
    need(original_result['status']=='failed_or_unresolved'
         and original_result['failure']['message']=='POST_EXPOSURE_EMPTY_FIXED_K'
         and original_result['total_policy_forward_calls']==0
         and original_result['checkpoint_loads']==2
         and all(r['status']=='not_started' for r in original_result['arms'].values())
         and original_parent['status']=='failed_or_unresolved'
         and prior['parent_seconds']==original_parent['elapsed_seconds'],'Original crop-edge failure/cost retained')
    need(preparation['manifest']=={'path':public['cases'][0]['path'],'sha256':public['cases'][0]['sha256']}
         and preparation['expanded_preparation_provenance']==public_manifest['expanded_preparation_provenance']
         and preparation['source_domain_counts']==public_manifest['public_source_domain_counts']
         and preparation['source_domain_counts']=={'support_domain':2184064,'target_in_known_support_positive':1579,'target_in_known_support_zero':33681,'target_in_unknown_support_domain':0},'Full target/domain qualification matches actual source')
    need(result['checkpoint_loads']==1 and set(release['endpoints'])=={'IL64'},'One IL reference load only')
    need(release['transition_mode']==result['transition_mode']=='lazy_planning'
         and release['actual_policy_forward_cap']==result['actual_policy_forward_cap']==0
         and result['reference_checkpoint_scope']=='admission_and_replay_only_not_action_author'
         and release['limits']=={'max_steps':24,'max_optimizer_updates':0,'max_native_previews':750000,
             'max_policy_forwards':48,'worker_seconds':900,'memory_bytes':3*1024**3,'threads':1,
             'search':{'max_calls':5640,'beam_width':2,'seconds':300}}
         and release['parent_seconds']==960 and release['attempts']==1
         and release['automatic_retry'] is False,'Exact historical admission and tighter no-forward runtime')
    prior={k:pinned(v) for k,v in release['prior_expanded_comparison'].items()}
    need(prior['result']['status']=='failed_or_unresolved' and prior['parent']['exit_code']==1
         and not prior['parent']['cleanup_errors'] and not prior['parent']['final_owned_pids']
         and prior['parent']['worker_termination_confirmed'] is True
         and prior['partial']==prior['result']['arms']['observed_beam']
         and prior['partial']['status']=='search_unresolved'
         and prior['partial']['accounting']['completed_layers']==4
         and prior['partial']['accounting']['model_transition_calls']==246
         and prior['partial']['accounting']['time_cap_reached'] is True,
         'Original incomplete four-arm result preserved')
    world_control=read(OUT/'observed_beam/prior-world-control.json')
    need(world_control==result['prior_world_control'] and world_control['exact'] is True
         and world_control['prior']==world_control['actual']==prior['world'],'Exact previous public world/inventory')
    lineages = {};training_worlds={};initials=[];training_evidence={}
    for name in ENDPOINTS:
        refs = release['endpoints'][name]
        tr, rr, pr, wr = [pinned(refs[k]) for k in ('release','result','parent','worker_final')]
        lineage = read(OUT/(name+'-lineage.json'))
        method, updates = ('RL' if name == 'RL8' else 'IL'), (64 if name == 'IL64' else 8)
        need(pr['status'] == ('failed_or_unresolved' if name=='IL64' else 'complete')
             and pr['exit_code'] == (1 if name=='IL64' else 0) and pr['worker_termination_confirmed'] is True
             and not pr['cleanup_errors'] and not pr['final_owned_pids'], 'Training terminal incomplete: '+name)
        need(pr['release_sha256'] == refs['release']['sha256'] and pr['result_sha256'] == refs['result']['sha256']
             == wr['canonical_result_sha256']
             and pr['worker_final_sha256'] == refs['worker_final']['sha256'], 'Training terminal chain differs: '+name)
        if name=='RL8':need(wr['result_sha256']==refs['result']['sha256'] and refs['result']['sha256']=='b3bf8ff60f49b728618efa91b253676a703e607fae99d09faf5375a48c4dc1b8','exact completed audited scratch RL')
        evidence=read(OUT/(name+'-training-evidence.json'));training_evidence[name]=evidence
        if name=='IL64':
            need(refs['status']=='original_failed_plus_accepted_replay' and rr['status']=='failed_or_unresolved'
                 and refs['result']['sha256']=='eedc4857c184d94913976cb5b59e01341e1b3a8bb0f001af43f8d079e3d87975'
                 and evidence['original_attempt_status']=='failed_or_unresolved','original IL failure is immutable')
            recovery=evidence['separate_recovery'];rp=pinned(recovery['receipt']);rrc=pinned(recovery['result']);cert=pinned(recovery['replay'])
            need(recovery['receipt']==refs['recovery_receipt'] and recovery['result']['sha256']=='432535613f699604a4b0f3b0aaa0fedebbe932e5ca6b7006beedd66ff823a7ce'
                 and rp['status']=='complete' and rp['exit_code']==0 and not rp['cleanup_errors']
                 and rrc['independent_accepted'] and rrc['full_history_equal'] and cert['independent_geometry']['accepted']
                 and rrc['original_IL_result_sha256']==refs['result']['sha256']
                 and rrc['checkpoint_sha256']==rr['checkpoints']['IL']['sha256']
                 and rrc['parameter_hash']==rr['checkpoints']['IL']['parameter_hash'],'separate exact045 accepted recovery')
        need(rr['optimizer_updates'][method] == updates and rr['SELECT_EVAL_opened'] is False
             and rr['private_reference_reads'] == 0, 'Training endpoint scope differs')
        original_contexts={s:pinned(refs['contexts'][s]) for s in TRAIN}
        contexts={('ReMIND:'+s[-3:]):semantic(original_contexts[s]) for s in TRAIN}
        need(all(c['subject']==subject and c['role']=='TRAIN' and c['private_reference_in_task'] is False
             and c['occupancy_condition']==OCCUPANCY for subject,c in original_contexts.items()), 'Original public TRAIN contexts differ')
        execution=tr['learning_protocol']['cohort_execution']
        need('il_motion_supervision' not in execution,'Original checkpoint objectives; ranking contrast excluded')
        training_worlds[name]={k:execution[k] for k in ('max_steps','proposal_config','proposal_rule_hash','search','retention_mode','task_condition','occupancy_condition','post_exposure_condition')}
        training_worlds[name]['public_target_context_variant']=tr['learning_protocol']['public_target_context_variant']
        initials.append(rr['initial_parameter_hash'])
        need(execution['occupancy_condition']==OCCUPANCY and execution['max_steps']==24
             and execution['post_exposure_condition']==EXPOSURE
             and execution['proposal_config']['max_candidates']==120 and execution['proposal_config'].get('obstruction_opening',False) is False
             and execution['proposal_config']['tool_footprint_opening'] is True,'Matched TRAIN world differs')
        checkpoint = rr['checkpoints'][method]
        if name=='RL8':need(checkpoint==pr['checkpoints'][method],'RL parent checkpoint descriptor')
        need(checkpoint['sha256'] == refs['checkpoint']['sha256']
             == lineage['checkpoint_sha256'] and checkpoint['parameter_hash'] == lineage['parameter_hash'], 'Checkpoint lineage differs: '+name)
        need(lineage['method'] == method and lineage['completed_updates'] == updates
             and lineage['optimizer_updates_on_SELECT'] == 0 and lineage['training_context_hashes'] == contexts
             and lineage['learning_protocol_hash'] == semantic(tr['learning_protocol'])
             and lineage['training_release_sha256'] == refs['release']['sha256'], 'Reloaded endpoint protocol/context differs')
        lineages[name] = lineage
    need(len(initials)==1,'Reference-only IL endpoint denominator')
    common_world = prior['world']; common_derivation = common_support = None
    selection_costs={}
    arm_summary = {}
    for arm in ARMS:
        directory = OUT/arm
        summary,world,context=[read(directory/name) for name in ('summary.json','comparison-world.json','context.json')]
        derivation=read(directory/'source/public-task-derivation.json')
        need(world==prior['world'] and context['subject']=='ReMIND-013' and context['role']=='SELECT'
             and context['max_optimizer_updates']==0 and context['checkpoint_lineage']==lineages['IL64']
             and context['private_reference_in_task'] is False and context['occupancy_condition']==OCCUPANCY
             and context['max_steps']==24 and context['execution_kind']=='frozen_SELECT013_post_exposure_inference_v1',
             'Same admitted public SELECT world/context')
        need(world['source_domains']['source_support_unknown_voxels']==176294
             and world['source_domains']['target_in_unknown_support_voxels']==0
             and world['source_domains']['assumed_material_beyond_source_domain_voxels']==0
             and world['source_domains']['source_domain_extended'] is False,'Known source domain preserved')
        if unresolved:
            partial=read(directory/'search-unresolved.json');accounting=partial['accounting']
            need(partial==summary==result['arms'][arm] and partial['status']=='search_unresolved'
                 and (accounting['time_cap_reached'] is True or accounting['call_cap_reached'] is True),
                 'Capped outcome remains unresolved')
            for name in ('complete-trace.json','native-replay.json','plan.json','select-replay.json','search-return.json'):
                need(not (directory/name).exists(),'Capped prefix was promoted to accepted trace/plan: '+name)
            arm_summary[arm]={'saved_status':'search_unresolved','partial_actions':partial['partial_actions'],
                'reason':partial['reason'],'accepted_plan':False,'executed_outcome':None}
            selection_costs[arm]={'planning_seconds':accounting['planning_seconds'],
                'scope':'capped selector only; no collection/native replay'}
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
             and context['execution_kind']=='frozen_SELECT013_post_exposure_inference_v1'
             and context['post_exposure']['version']==world['post_exposure']['version']==EXPOSURE, 'Arm context differs')
        domains=world['source_domains']
        need(domains['source_support_unknown_voxels']==176294 and domains['target_in_unknown_support_voxels']==0
             and domains['assumed_material_beyond_source_domain_voxels']==0 and domains['source_domain_extended'] is False,
             '013 known source domain stays distinct from explicit material assumption')
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
            search = read(directory/'search-return.json');selection=read(directory/'selection.json','8f62656d1081ea247499cb1ff7ed8b803bf2249d3e700703d3806506a151c8f4')
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
             and goal['derived_occupancy'] is True and goal['occupancy_condition']=='cerebrum_plus_supplied_tumor_assumption'
             and goal['fraction_denominator']=='entire_unchanged_supplied_region'
             and occupancy['operation']=='S OR (T > 0)' and occupancy['condition']=='cerebrum_plus_supplied_tumor_assumption'
             and occupancy['added_region_positive_voxels']==33681 and occupancy['source_support_unchanged'] is True
             and occupancy['target_unchanged'] is True and occupancy['anatomical_or_material_validation'] is False,
             'Raw support deficit/full target/explicit derived occupancy changed')
        need(close(goal['fraction_of_full_region_removed'],summary['target_removed_mm3']/goal['full_region_membership_mm3']),
             'Whole supplied target denominator differs')
        arm_summary[arm] = {k: summary[k] for k in ('steps','actions','public_return','target_removed_mm3','outside_supplied_target_removed_mm3')}
        arm_summary[arm].update(independent_geometry_accepted=True,full_target_fraction=goal['fraction_of_full_region_removed'],
            terminal_reason=plan['terminal_reason'],checkpoint_author=summary['checkpoint_author'],
            full_target_mm3=goal['full_region_membership_mm3'],certificate_outcomes=geometry['outcomes'],
            unknowns=sorted(set(metrics.get('unknowns',())) | {u for row in metrics['history'] for key in ('unknowns','geometry_unknowns') for u in row.get(key,())}))
    # Independent layer projection: only successor-availability timing is omitted.
    def projection(layer):
        d=layer['retained_prefix_diagnostics']
        return {'depth':layer['depth'],'calls':layer['model_transition_calls'],
            'negative':layer['negative_prefixes_evaluated'],'pruned':layer['negative_prefixes_pruned_by_beam'],
            'opening_depth_retention':layer['opening_depth_retention'],
            'prefixes':[{k:v for k,v in p.items() if k not in ('legal_nonstop_actions','successor_inventory_status')}
                for p in d['retained_nonterminal_prefixes']], 'terminals':d['best_terminal_prefixes']}
    old_layers=[r for r in prior['partial']['accounting']['layers'] if r['completed']]
    layers=accounting['layers']; completed=[r for r in layers if r['completed']]
    need(accounting['transition_mode']=='lazy_planning' and accounting['eager_transition_calls']==0
         and accounting['lazy_planning_transition_calls']==accounting['model_transition_calls']<=5640
         and accounting['actor_forward_calls']==0 and accounting['guidance']=='none'
         and accounting['max_calls']==5640 and accounting['beam_width']==2
         and accounting['objective_source']=='permitted_nominal_target_and_frozen_geometric_costs'
         and accounting['opening_depth_retention']['version']=='return_plus_opening_depth_volume_v1'
         and accounting['opening_depth_retention']['terminal_objective_changed'] is False
         and accounting['opening_depth_retention']['extra_observations_or_previews']==0,
         'Unchanged public beam semantics with deferred successor inventories')
    need([r['depth'] for r in layers]==list(range(1,len(layers)+1))
         and len(completed)==accounting['completed_layers']<=24
         and sum(r['model_transition_calls'] for r in layers)==accounting['model_transition_calls']
         and sum(r['negative_prefixes_evaluated'] for r in layers)==accounting['negative_prefixes_evaluated']
         and accounting['evaluated_transition_prefixes']==accounting['model_transition_calls']
         and accounting['implicit_stop_prefix_count']==accounting['evaluated_transition_prefixes']+1,
         'Exact layer, transition and implicit-STOP accounting')
    checked=min(4,len(completed))
    left=[projection(r) for r in old_layers[:checked]];right=[projection(r) for r in completed[:checked]]
    need(len(old_layers)==4 and left==right,'Historical completed layers changed')
    if not unresolved:
        control=read(OUT/'observed_beam/prior-layer-control.json')
        expected={'exact':True,'completed_layers_checked':4,
            'excluded_successor_fields':['legal_nonstop_actions','successor_inventory_status'],
            'prior_partial_fifth_layer_not_compared':True,'prior':left,'actual':right}
        need(checked==4 and control==result['prior_completed_layer_control']==expected,
             'Saved four-layer control must equal independent reconstruction')
        need(accounting['planning_seconds']<=300 and all(r['completed'] for r in layers),
             'Complete local search must finish uncapped')
        need(close(accounting['estimated_incremental_return'],arm_summary['observed_beam']['public_return']),
             'Selected public return equals complete replay')
    else:
        if accounting['time_cap_reached']:need(accounting['planning_seconds']>300,'Time-cap semantics')
        if not layers[-1]['completed']:
            need(layers[-1]['retained_prefix_diagnostics']['selection_status']=='partial_layer_not_advanced',
                 'Interrupted layer never promoted')
    progress=[]
    for layer in layers:
        d=layer['retained_prefix_diagnostics']
        progress.append({'depth':layer['depth'],'completed':layer['completed'],
            'calls':layer['model_transition_calls'],'negative_prefixes':layer['negative_prefixes_evaluated'],
            'opening_depth_retention':layer['opening_depth_retention'],
            'retained_nonterminal_prefixes':d['retained_nonterminal_prefixes'],
            'best_terminal_prefixes':d['best_terminal_prefixes']})
    progress_rows=[p for row in progress for key in ('retained_nonterminal_prefixes','best_terminal_prefixes') for p in row[key]]
    target_rows=[p for p in progress_rows if p.get('target_removed_mm3') is not None]
    costs=read(OUT/'costs.json','c7d0ea71d90c036cb66727f6cd96b6957ed03d5c299a7257ad06f7cb1ffcee17');budget=result['native_budget']
    need(result['total_policy_forward_calls']==costs['policy_forwards']==0
         and sum(row.get('policy_forward_calls',0) for row in costs['phases'].values())==0,'Zero actor forwards')
    need(all(budget[k]==costs['native_budget'][k] for k in
         ('native_preview_entries','counting_reliable','failure','blocked_preview_attempts','limits','status'))
         and budget['counting_reliable'] is True and budget['blocked_preview_attempts']==0
         and budget['native_preview_entries']<=750000 and budget['elapsed_seconds']<=900
         and budget['failure']==('arm_raised_ValueError' if unresolved else None)
         and budget['history_complete_caller_attestation'] is (None if unresolved else True),
         'Bounded preview history attestation and sticky failure')
    expected_phases={'checkpoint_reload.IL64','observed_beam.public_construction','observed_beam.planning'}
    if not unresolved:expected_phases.update({'observed_beam.collection','observed_beam.independent_replay'})
    need(set(costs['phases'])==expected_phases,'Exact separate load/construction/search/collection/replay phases')
    need(sum(p['native_preview_entries'] for p in costs['phases'].values())==budget['native_preview_entries'],
         'All native previews charged to distinct phases')
    for phase,row in costs['phases'].items():
        need(0<=row['complete_wall_seconds']<=parent['elapsed_seconds'] and row['native_preview_entries']>=0,
             'Invalid measured phase cost: '+phase)
    if not unresolved:
        need(selection_costs['observed_beam']['native_previews']==costs['phases']['observed_beam.planning']['native_preview_entries'],
             'Search-only native-preview count')
    return {'decision':'PASS_SAVED_LAZY_BEAM_RECOVERY' if not unresolved else 'PASS_RETAINED_UNRESOLVED_BEAM',
        'complete_recovered_beam':not unresolved,'original_four_arm_result_still_incomplete':True,
        'arms':arm_summary,'parent_status':parent['status'],'parent_exit_code':parent['exit_code'],
        'world':common_world,'prior_completed_layers_verified':checked,
        'layer_control_exclusions':['legal_nonstop_actions','successor_inventory_status'],
        'prior_partial_fifth_layer_compared':False,
        'search_accounting':{k:v for k,v in accounting.items() if k!='layers'},'layer_progress':progress,
        'max_saved_retained_or_terminal_target_mm3':max([p['target_removed_mm3'] for p in target_rows],default=None),
        'progress_scope':'Retained prefixes and terminal candidates only; this does not enumerate progress of every discarded successor.',
        'cost_phases':costs['phases'],'selector_costs':selection_costs,
        'cost_scope':'Load, construction, planning, collection and replay are separate measured phases; nested inventory/transition timers must not be summed.',
        'native_previews':budget['native_preview_entries'],'policy_forwards':0,'checkpoint_loads':1,
        'checkpoint_role':'Original IL64 reference for admission/replay only; it did not select actions.',
        'optimizer_updates_on_SELECT':0,'gradient_attempts':0,'private_reference_reads':0,
        'planned_SELECT_denominator':2,'executed_SELECT_cases':1,'held_cases':['ReMIND-037'],'EVAL_closed':True,
        'elapsed_seconds':parent['elapsed_seconds'],'sampled_peak_rss_bytes':parent['sampled_peak_rss_bytes'],
        'output_bytes':parent['output_bytes'],'cleanup_errors':parent['cleanup_errors'],
        'final_owned_pids':parent['final_owned_pids'],'worker_termination_confirmed':parent['worker_termination_confirmed'],
        'training_evidence':training_evidence,'excluded_weight_payload_pins_not_opened':excluded_weights,
        'limits':'Saved receipts/source only; no tensor/array reread, geometric recomputation or independent syscall tracing. More layers or STOP alone do not establish useful target progress, impossibility or clinical benefit. This is an explicit post-exposure S-union-T simulation with partial source domain.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--result-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30s saved-only audit cap')))
    signal.alarm(30)
    try:
        result = audit(args.release, args.release_sha256,args.receipt_sha256,args.result_sha256)
        for path,meta in list(READS.items()):bytes_read(ROOT/path,meta['sha256'])
    except Exception as error:
        result = {'decision':'HOLD_SAVED_RESULT_MISMATCH_OR_INCOMPLETE', 'reason':type(error).__name__+': '+str(error)}
    signal.alarm(0)
    result.update(audit_seconds=time.monotonic()-started,audit_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  checks=CHECKS,scope='saved_JSON_and_source_only', evidence_files=READS,
                  no_model_or_patient_array_imports=True, release_sha256=args.release_sha256)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:result.get(k) for k in ('decision','reason','audit_seconds','audit_peak_rss_bytes','checks','complete_recovered_beam','prior_completed_layers_verified','native_previews','policy_forwards','elapsed_seconds','max_saved_retained_or_terminal_target_mm3')}, indent=2))
    return 0 if result['decision'].startswith('PASS_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
