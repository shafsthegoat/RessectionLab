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
RUN = ROOT / 'build/post-exposure-select013-expanded-comparison-v1'
OUT = RUN / 'attempt-01'
ENDPOINTS = ('IL64', 'RL8')
ARMS = ('observed_greedy', 'IL64', 'RL8', 'observed_beam')
OCCUPANCY = 'cerebrum_plus_supplied_tumor_with_preserved_partial_source_domain'
EXPOSURE = 'public-union-post-exposure-axis0-v1'
TRAIN = ('ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045')
# Reviewed sources; root supplies actual terminal release/receipt/result pins.
PINS = {'comparison_contract.py':'2ff76c6f38c1b15a2457603f1357fb55d4836f258cb066def60f4bde7ff37c6f',
        'select_worker.py':'3a5ed23f6a41ef07e7c0ccd960dcfa5b2460c53fd7679444a3feca0849e49a1d',
        'run_owned.py':'8e41ef61c8ec624f32c2c0a408e8c838383246bc50521d76c1ea7ac8e719ea8f',
        'freeze-runtime.py':'e60b47b97c292c153d2e19937625ead851161f3eccdee622ea9545f071b84ef2'}
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
    need(index['head'] == release['expected_head']=='d081d145d6ae145eac1ae0394d0e979ee60558ca', 'Source HEAD binding differs')
    need(release_sha=='1cabbdf803598f03e04d878e99c0385616a79fb4982c1c0e060bd1e1ec7a98e4'
         and release['source_index']['sha256']=='345c6aa3a07ea322b9c2d41f955f11a69f8950db91c26c7478139d85f8bbd797','exact root launched experiment')
    need(len(PINS)==4, 'Reviewed runtime pins are not frozen yet')
    for name, expected in PINS.items():
        need(index['source_files']['build/post-exposure-select013-expanded-comparison-v1/' + name] == expected,
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
             and worker['status']=='complete_owned_post_exposure_SELECT013'
             and result['status']=='complete_post_exposure_SELECT013_comparison','Comparison terminal mismatch')
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
    if unresolved:
        partial={};loaded={}
        for name in ENDPOINTS:
            path=OUT/(name+'-lineage.json')
            if path.is_file():
                lineage=read(path);refs=release['endpoints'][name];upstream=pinned(refs['result'])
                method='IL' if name=='IL64' else 'RL';checkpoint=upstream['checkpoints'][method]
                need(lineage['checkpoint_sha256']==refs['checkpoint']['sha256']==checkpoint['sha256']
                     and lineage['parameter_hash']==checkpoint['parameter_hash']
                     and lineage['completed_updates']==(64 if name=='IL64' else 8)
                     and lineage['training_release_sha256']==refs['release']['sha256']
                     and lineage['optimizer_updates_on_SELECT']==0,'failed-attempt authenticated checkpoint lineage')
                loaded[name]=lineage
        need(len(loaded)==result['checkpoint_loads'],'actual saved load count')
        for arm in ARMS:
            directory=OUT/arm;row=result['arms'][arm];files={}
            for name in ('summary.json','comparison-world.json','context.json','search-unresolved.json','plan.json','native-replay.json','select-replay.json'):
                path=directory/name
                if path.is_file():files[name]=read(path)
            if 'summary.json' in files:need(files['summary.json']==row,'partial summary exact')
            if row['status']=='search_unresolved':
                need(files.get('search-unresolved.json')==row and 'select-replay.json' not in files,'capped prefix never completed replay')
                need(result['native_budget']['history_complete_caller_attestation'] is not True,'no false complete-history attestation')
            certificate=files.get('native-replay.json',{}).get('independent_geometry')
            if certificate is not None:
                need(certificate['committed_history_hash']==semantic(files['native-replay.json']['metrics']['history']),'partial saved certificate hash')
            partial[arm]={'saved_status':row['status'],'result_row':row,
                'saved_certificate_accepted':None if certificate is None else certificate['accepted'],
                'complete_accepted_route':row['status']=='complete_replayed' and certificate is not None and certificate['accepted'] is True,
                'certificate_outcomes':None if certificate is None or not certificate['accepted'] else certificate['outcomes']}
        costs=read(OUT/'costs.json')
        need(costs['policy_forwards']==result['total_policy_forward_calls'],'failed-attempt forwards retained')
        need(costs['native_budget']['native_preview_entries']==result['native_budget']['native_preview_entries'],'failed-attempt previews retained')
        return {'decision':'PASS_RETAINED_INCOMPLETE_ATTEMPT_EVIDENCE','complete_comparison':False,
            'parent_status':parent['status'],'failure':result['failure'],'arms':partial,
            'planned_SELECT_denominator':2,'held_cases':['ReMIND-037'],'EVAL_closed':True,
            'loaded_checkpoint_lineages':loaded,'checkpoint_loads':len(loaded),
            'policy_forwards':costs['policy_forwards'],'native_previews':result['native_budget']['native_preview_entries'],
            'native_budget':result['native_budget'],'costs':costs,
            'parent':{k:parent[k] for k in ('elapsed_seconds','sampled_peak_rss_bytes','output_bytes','cleanup_errors','final_owned_pids')},
            'expanded_preparation':preparation,
            'limits':'Partial evidence only; no full comparison or accepted capped-prefix claim. No geometry/model replay.'}
    need(result['checkpoint_loads']==2,'exact two checkpoint loads')
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
                 and context['execution_kind']=='frozen_SELECT013_post_exposure_inference_v1',
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
            terminal_reason=plan['terminal_reason'],checkpoint_author=summary['checkpoint_author'],
            full_target_mm3=goal['full_region_membership_mm3'],certificate_outcomes=geometry['outcomes'],
            unknowns=sorted(set(metrics.get('unknowns',())) | {u for row in metrics['history'] for key in ('unknowns','geometry_unknowns') for u in row.get(key,())}))
    costs = read(OUT/'costs.json')
    forwards = sum(arm_summary[n]['steps'] for n in ENDPOINTS)
    budget = result['native_budget']
    need(result['total_policy_forward_calls'] == costs['policy_forwards'] == forwards <= 48, 'Forward accounting differs')
    need(all(budget[k] == costs['native_budget'][k] for k in
             ('native_preview_entries','counting_reliable','failure','blocked_preview_attempts','limits','status'))
         and budget['counting_reliable'] is True
         and budget['failure'] is None and budget['blocked_preview_attempts'] == 0
         and budget['native_preview_entries'] <= 750000, 'Native budget incomplete/capped')
    need(budget['history_complete_caller_attestation'] is True,'completed four-arm history attested only after replay')
    need(sum(r.get('policy_forward_calls',0) for r in costs['phases'].values())==forwards, 'Phase forward count differs')
    for arm in ARMS:
        need(arm+'.public_construction' in costs['phases'] and arm+'.planning_and_replay' in costs['phases'], 'Missing charged method phase')
    return {'decision':'PASS_SAVED_POST_EXPOSURE_SELECT013_COMPARISON',
        'complete_comparison':not unresolved,'unresolved_arms':['observed_beam'] if unresolved else [],
        'parent_status':parent['status'],'parent_exit_code':parent['exit_code'],'automatic_retry':False, 'arms':arm_summary,
        'world':common_world, 'planned_SELECT_denominator':2, 'held_cases':['ReMIND-037'],
        'policy_forwards':forwards, 'native_previews':budget['native_preview_entries'],
        'wall_seconds':parent['elapsed_seconds'], 'sampled_peak_rss_bytes':parent['sampled_peak_rss_bytes'],
        'checkpoint_loads':2, 'optimizer_updates_on_SELECT':0,'executed_SELECT_cases':1,'planned_SELECT_cases':2,
        'checks':CHECKS,'selector_costs':selection_costs,'cost_phases':costs['phases'],
        'cost_scope':'Selector-only time excludes construction, collection and replay. Each planning_and_replay phase includes selection plus native/independent replay; inventory/transition/forward timers are nested and must not be added.',
        'training_evidence':training_evidence,'excluded_weight_payload_pins_not_opened':excluded_weights,
        'expanded_preparation':preparation,
        'occupancy_condition':OCCUPANCY,'raw_unsupported_target_voxels':33681,'full_supplied_target_voxels':35260,
        'information_scope':'same permitted arrays, physical task, initial inventory and public-target context; actor features and planner computation differ, so no identical-representation claim',
        'limits':'Saved receipts and reviewed source; no independent syscall trace, tensor reload, array reread or geometric recomputation. Explicit post-exposure S-union-T world, preserved full target and declared workspace; no clinical/material validation, success threshold or equal-cost claim. Source-zero is not proven air, outside-image extent and between-insertion transfers remain unassessed.'}


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
    print(json.dumps({k:v for k,v in result.items() if k != 'evidence_files'}, indent=2))
    return 0 if result['decision'].startswith('PASS_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
