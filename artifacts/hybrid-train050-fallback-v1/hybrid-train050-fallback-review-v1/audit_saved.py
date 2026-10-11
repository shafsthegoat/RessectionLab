"""Saved JSON/source audit. Run only after root's terminal notification.

No acquired arrays, model parameter JSON, image files, production imports,
native replay, or model evaluation are permitted in this reviewer.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'build/hybrid-train050-fallback-v1'
OUT=BASE/'attempt-01'
SUP=BASE/'attempt-01.supervision'
HEAD='d70d814b54929061c299a52507a29fd3910f379c'
RELEASE='6c22d2577e4175a530459bb0952a6e802938ae341324a88303d70339c9049280'
INDEX='d557c4b6c762950a846cb1f0e478a49425921447a732581a50af91ae72286d3e'
METHODS=['balanced_ridge_stop_fallback','exhaustive']
TRAIN=['ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045']
READS={}

def require(value,reason):
    if not value:raise ValueError(reason)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def semantic(value):return 'sha256:'+sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def raw(path,expected=None):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    require(path.suffix in ('.json','.py','.log'),'source_or_metadata_only')
    require(path.name not in ('balanced_ridge_top8.json','old_ridge_top8.json'),'no_model_parameter_read')
    st=path.lstat();require(stat.S_ISREG(st.st_mode) and st.st_size<=16*1024**2,'bounded_regular_saved_file')
    data=path.read_bytes();after=path.stat()
    require((st.st_size,st.st_mtime_ns)==(after.st_size,after.st_mtime_ns) and len(data)==st.st_size,'changing_file')
    digest=sha(data);require(expected is None or digest==expected,'file_hash:'+str(path))
    READS[str(path.relative_to(ROOT))]={'sha256':digest,'bytes':len(data)}
    return data
def read(path,expected=None):
    return json.loads(raw(path,expected),parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite_JSON')))
def close(a,b,tol=1e-7):require(math.isclose(a,b,rel_tol=1e-10,abs_tol=tol),'numeric_disagreement')
def extent(directory):
    total=0
    for path in directory.rglob('*'):
        st=path.lstat();require(not stat.S_ISLNK(st.st_mode),'output_symlink')
        if stat.S_ISREG(st.st_mode):total+=st.st_size
        else:require(stat.S_ISDIR(st.st_mode),'nonregular_output')
    return total
def determinant(a):
    return (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
        -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
        +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-sha256',required=True);p.add_argument('--result-sha256',required=True);args=p.parse_args()
    release=read(BASE/'root-release.json',RELEASE);index=read(BASE/'source-index.json',INDEX)
    require(release['execution_released'] is True and release['expected_head']==index['head']==HEAD,'exact_root_release')
    require(release['subjects']==['ReMIND-050'] and release['arms']==METHODS and release['K']==8 and release['horizon']==24,'fixed_denominator')
    require(release['zero_updates'] and release['zero_refits'] and release['SELECT_EVAL_closed'],'fixed_scope')
    source_delta=read(HERE/'runtime-source-delta.json','149405b2f66462ac4be7f4335f927a11f9986a7e4d327f10af216c9ed6b42d89')
    require(source_delta['head']==HEAD and source_delta['source_index_sha256']==INDEX and not source_delta['canonical_diff_from_executed_head'],'captured_runtime_source_delta')
    for name,pin in source_delta['changed_or_new_pins'].items():raw(name,pin)
    prior_index=read(ROOT/'build/hybrid-train050-transfer-v1/source-index.json','7926ad651b20e35d760cf3b0de3af0c547305204b3a3bc7503702d0986dfe2f6')
    require(all(prior_index['files'].get(k)==v for k,v in index['files'].items() if k not in source_delta['changed_or_new_pins']),'unchanged_previous_source_bindings')
    train=read(release['training_result']['path'],release['training_result']['sha256'])
    train_parent=read(release['training_parent']['path'],release['training_parent']['sha256'])
    train_release=read(release['training_release']['path'],release['training_release']['sha256'])
    require(train_parent['status']=='complete' and train_parent['exit_code']==0 and not train_parent['remaining_owned_pids'] and not train_parent['cleanup_errors'],'terminal_TRAIN_lineage')
    require(train_parent['result_sha256']==release['training_result']['sha256'] and train_parent['release_sha256']==release['training_release']['sha256'],'TRAIN_lineage_joins')
    require(train['status']=='complete_fixed_TRAIN_balanced_ridge_contrast' and train['fits']==2 and train['ridge_unchanged'],'exact_TRAIN_endpoint')
    require(train_release['subjects']==TRAIN and 'ReMIND-050' not in TRAIN,'unfitted_original_TRAIN_person')
    require(train['ridge_file_sha256']['balanced_ridge_top8']==release['ridge']['sha256'],'original_model_file_identity')
    require(train_release['K']==release['K'] and train_release['horizon']==release['horizon'] and train_release['proposal_config']==release['proposal_config'],'same_proposal_condition')
    manifest=read(release['inputs']['public_manifest']['path'],release['inputs']['public_manifest']['sha256'])
    read(release['inputs']['intended_use_review']['path'],release['inputs']['intended_use_review']['sha256'])
    require(manifest['role']=='TRAIN' and manifest['patient_id']=='ReMIND-050' and manifest['public_only'] and not manifest['private_evaluation_files_included'],'exact_public_manifest')
    require(manifest['public_target_support_consistency']['whole_tumor_positive_voxels']==19557 and manifest['public_support_source_domain_unknown_voxels']==183804,'unaltered_source_denominators')
    parent=read(SUP/'receipt.json',args.parent_sha256);result=read(OUT/'result.json',args.result_sha256)
    require(parent['result_sha256']==args.result_sha256 and parent['release_sha256']==result['release_sha256']==RELEASE,'runtime_receipt_joins')
    require(parent['status']=='complete' and parent['exit_code']==0 and parent['worker_termination_confirmed'] and not parent['remaining_owned_pids'] and not parent['cleanup_errors'] and parent['stop_reason'] is None,'terminal_clean_owned_worker')
    require(result['status']=='complete_TRAIN050_fallback_comparison' and result['role']=='TRAIN' and result['subject']=='ReMIND-050','complete_result')
    require(result['source_visits']==4 and result['ridge_loads']==1 and result['ridge_fingerprint']==release['ridge']['fingerprint'] and result['ridge_unchanged'],'exact_frozen_model_single_load')
    require(all(result[k]==0 for k in ('fits','model_calls','optimizer_calls','checkpoint_loads','blocked_external_calls')),'no_fitting_private_or_neural_work')
    require(result['SELECT_EVAL_opened'] is False and result['private_reference_loaded'] is False and result['cache_restored'] is True,'information_boundary')
    caps=release['caps'];require(parent['caps']==caps and parent['source_index']==release['source_index'],'exact_caps_and_runtime_sources')
    for key in ('native_previews','native_transitions','ridge_calls'):require(result[key]<=caps[key],'cap:'+key)
    require(result['elapsed_seconds']<=caps['worker_seconds'] and parent['elapsed_seconds']<=caps['parent_seconds'] and parent['sampled_peak_rss_bytes']<=caps['sampled_rss_bytes'],'owned_wall_and_sampled_memory_caps')
    require(parent['samples']>0 and parent['automatic_retry'] is False,'sampled_one_attempt')
    output_bytes=extent(OUT);supervision_bytes=extent(SUP)
    require(output_bytes==parent['output_bytes'] and output_bytes<=caps['output_bytes'] and supervision_bytes<=caps['supervision_bytes'],'output_caps')
    require((SUP/'worker.log').stat().st_size<=caps['log_bytes'],'final_log_cap')
    declaration=read(SUP/'declaration.json');require(declaration['release_sha256']==RELEASE and declaration['head']==HEAD and declaration['source_index']==release['source_index'],'supervisor_start_bindings')
    command=declaration['command'];require('-I' in command and '-B' in command and any(x.startswith('pycache_prefix=') for x in command),'isolated_fresh_prefix_recipe')
    budget=result['native_budget'];require(budget['counting_reliable'] and budget['failure'] is None and budget['blocked_preview_attempts']==0 and budget['history_complete_caller_attestation'] is True,'reliable_complete_preview_budget')
    require(budget['native_preview_entries']==result['native_previews'],'preview_total_join')
    arms=result['arms'];require([(x['method'],x['subject']) for x in arms]==[(m,'ReMIND-050') for m in METHODS],'both_arms_retained')
    require(all(x['complete'] and x['independent_accepted'] and x['status']=='complete' for x in arms),'both_full_accepted_episodes')
    visits=result['source_visits_detail'];require([x['phase'] for x in visits]==['planning.'+m for m in METHODS]+['replay.'+m for m in METHODS],'plans_before_replay_visit_order')
    world=result['public_world'];require(all(x['status']=='complete' and all(x[k]==v for k,v in world.items()) for x in visits),'same_world_all_visits')
    sealed=read(OUT/'sealed-plans.json');require(result['all_plans_sealed_before_evaluation'] and not sealed['private_reference_loaded'] and set(sealed['plans'])==set(METHODS),'both_plans_presealed')
    records=[];bindings=None;first_inventory=None;ridge_calls=0;transitions=0;previews=0
    for row in arms:
        mode=row['method'];folder=OUT/mode
        for path,pin in row['files'].items():raw(OUT/path,pin)
        plan=read(folder/'plan.json',row['files'][mode+'/plan.json'])
        unsigned={k:v for k,v in plan.items() if k!='seal'}
        require(semantic(unsigned)==plan['seal'] and plan==sealed['plans'][mode],'plan_seal')
        require(plan['method']==mode and plan['subject']=='ReMIND-050' and plan['source_hash']==world['source'] and plan['max_steps']==24,'plan_identity')
        require(plan['ridge_hash']==(release['ridge']['fingerprint'] if mode==METHODS[0] else None),'frozen_ranker_arm')
        actions=plan['actions'];accounting=plan['accounting'];require(1<=len(actions)<=24 and accounting['complete'] and not accounting['global_optimality_proven'],'complete_finite_horizon_plan')
        stops=[i for i,a in enumerate(actions) if a=='STOP'];require(stops==[len(actions)-1] if stops else len(actions)==24,'terminal_STOP_or_horizon')
        require(accounting['verification_scope']==('top8_then_conditional_full_emitted_catalog' if mode==METHODS[0] else 'full_emitted_catalog'),'honest_query_scope')
        decisions=accounting['decisions'];require([x['selected_action_id'] for x in decisions]==actions,'saved_decisions_match_plan')
        for dec in decisions:
            scores=dec['scores'];require(scores[0]['action_id']=='STOP' and scores[0]['reward']==0,'STOP_baseline')
            require(dec['all_queried_legal_actions_scored'] and dec['scored_nonstop_actions']==dec['legal_nonstop_actions']==len(scores)-1,'complete_queried_legal_scores')
            winner=max(scores,key=lambda s:s['reward']);require(winner['action_id']==dec['selected_action_id'],'unchanged_greedy_reward_and_ties')
        met=read(folder/'replay/episode-metrics.json');audit=read(folder/'replay/independent-episode.json')
        require(met['source_hash']==audit['source_hash']==plan['source_hash'] and met['decision_model_hash']==audit['decision_model_hash']==plan['decision_model_hash'],'replay_identity')
        history=met['history'];require([x['action_id'] for x in history]==actions and audit['committed_history_hash']==semantic(history),'exact_saved_history')
        require(met['terminated'] and met['steps']==len(history) and audit['accepted'] and audit['complete_episode'] and audit['geometry']['feasible'],'saved_full_geometry_acceptance')
        outcomes=audit['outcomes'];require(outcomes==row['outcomes'],'reported_outcomes_join')
        close(math.fsum(h['reward'] for h in history),outcomes['total_reward'])
        for key in ('target_removed_mm3','normal_removed_mm3','complete_tool_path_length_mm'):
            close(math.fsum(h[key] for h in history),outcomes[key])
        require(outcomes['nonstop_actions']==len(history)-len(stops),'nonstop_denominator')
        for key in ('target_removed_mm3','normal_removed_mm3','simulated_removed_volume_mm3','cumulative_contacted_tissue_upper_bound_mm3','currently_retained_contacted_tissue_upper_bound_mm3','total_reward'):close(met[key],outcomes[key])
        close(outcomes['reference_target_fraction_removed'],outcomes['target_removed_mm3']/outcomes['total_reference_target_mm3'])
        require(outcomes['clinical_deficit_probability'] is None and outcomes['motor_surrogate'] is None and outcomes['language_surrogate'] is None,'no_clinical_probability')
        for phase in ('source','replay/source'):
            public=read(folder/phase/'admitted-public-bindings.json');s,q,p=public['source_binding'],public['qc'],public['protocol']
            if bindings is None:bindings=public
            else:require(public==bindings,'identical_bound_world_all_four_visits')
            require(semantic(s)==world['binding'] and semantic(q)==world['qc'] and semantic(p)==world['protocol'],'world_metadata_hashes')
            require(s['subject']=='ReMIND-050' and s['patient_group']=='ReMIND:050' and p['role']=='TRAIN' and s['source_hash']==world['source'],'original_role_source')
            d=p['development_transfer'];require(d['training_subjects']==TRAIN and d['subject']=='ReMIND-050' and semantic(d)==p['learning_protocol_hash'],'unfitted_transfer_lineage')
            require(d['frozen_ridge_sha256']==release['ridge']['sha256'] and d['frozen_ridge_fingerprint']==release['ridge']['fingerprint'] and d['training_result_sha256']==release['training_result']['sha256'],'bound_ridge_identity')
            require(d['public_manifest_sha256']==release['inputs']['public_manifest']['sha256'] and d['intended_use_review_sha256']==release['inputs']['intended_use_review']['sha256'],'QC_manifest_lineage')
            require(p['max_optimizer_updates']==p['max_policy_forwards']==0 and not p['private_reference_used'] and not p['clinical_claim'] and not p['split_changes'],'zero_update_scope')
            require(public['supplied_goal_extent']['full_region_positive_voxels']==19557,'full_target_preserved')
            domains=s['source_and_simulated_domains'];require(domains['source_support_unknown_voxels']==183804 and domains['target_in_unknown_support_voxels']==0 and not domains['source_domain_extended'],'unknown_domain_preserved')
            require(not s['post_exposure']['physical_air_or_operative_opening_observed'] and not s['post_exposure']['source_material_deleted'] and not s['post_exposure']['source_target_deleted'],'declared_exposure_not_observed_surgery')
            a=public['native_grid_reconciliation']['derived_affine_ras_mm'];close(outcomes['total_reference_target_mm3'],19557*abs(determinant(a)))
        inventory_counts=[];gate_summaries=[]; online_inventories=[]
        for phase in ('inventories','replay/inventories'):
            paths=sorted((folder/phase).glob('state-*.json'));require(len(paths)==len(actions),'one_inventory_per_decision')
            local_previews=0;local_extra=0;local_triggers=0;local_checks=0
            for i,path in enumerate(paths):
                inv=read(path);short=inv['shortlist'];require(inv['step']==i,'inventory_order')
                ids=short['raw_ids'];selected=short['selected_ids'];n=short['raw_count'];k=short['queried_count']
                require(n==len(ids) and len(set(ids))==n and k==len(selected) and set(selected)<=set(ids) and short['not_queried_count']==n-k,'raw_query_denominators')
                require(len(set(selected))==k,'no_repeated_queries')
                require(short['priority_is_not_legality'] and short['exhaustive_over_emitted_catalog']==(n==k),'legality_scope')
                if mode==METHODS[0]:
                    scores=short['priority_scores'];require(len(scores)==n,'aligned_ranker_scores')
                    initial=short['initial_selected_ids'];extra=short['additional_selected_ids'];gate=short['initial_nominal_scores']
                    require(initial==[ids[j] for j in sorted(range(n),key=lambda j:(-scores[j],j))[:8]],'stable_original_top8_ranking')
                    require(short['fallback_completed'] and short['fallback_policy']['version']==result['policy_version'],'completed_exact_fallback')
                    triggered=not any(x['nominal_reward']>0 for x in gate)
                    require(short['fallback_triggered']==triggered,'exact_nonpositive_trigger')
                    require(extra==([x for x in ids if x not in set(initial)] if triggered else []),'remaining_ids_once_raw_order')
                    require(selected==initial+extra,'actual_query_sequence')
                    dec=decisions[i];by_id={x['action_id']:x for x in dec['scores'][1:]}
                    require(set(x['action_id'] for x in gate)==set(initial)&set(by_id),'all_initial_feasible_rewards')
                    for x in gate:close(x['nominal_reward'],by_id[x['action_id']]['reward'])
                    local_extra+=len(extra);local_triggers+=int(triggered);local_checks+=len(gate)
                    if phase=='inventories':gate_summaries.append({'step':i,'state':inv['source_state_hash'],'initial_legal':len(gate),'triggered':triggered,'additional_queries':len(extra),'final_legal':short['legal_count'],'selected_action_id':actions[i],'selected_reward':next(x['reward'] for x in dec['scores'] if x['action_id']==actions[i])})
                else:require(short['priority_scores'] is None and selected==ids,'lean_exhaustive')
                if i==0:
                    initial={'state':inv['source_state_hash'],'public_input':short['input_binding'],'raw_ids':ids}
                    if first_inventory is None:first_inventory=initial
                    else:require(initial==first_inventory,'same_initial_catalog_all_arms_replays')
                require(inv['source_state_hash']==decisions[i]['source_state_hash'],'decision_inventory_state')
                certified={x['proposal_id']:x for x in inv['ledger'] if x.get('query_disposition') in ('CERTIFIED_LEGAL','CERTIFIED_REJECTED')}
                require(set(certified)==set(selected),'every_query_has_certificate')
                legal=[x for x in ids if x in certified and certified[x]['feasible']]
                require(legal==[x['action_id'] for x in decisions[i]['scores'][1:]] and len(legal)==short['legal_count'],'scores_exact_certified_legal_raw_order')
                if phase=='inventories':online_inventories.append(inv)
                else:require(inv==online_inventories[i],'online_uncached_replay_inventory_parity')
                local_previews+=k
            costs=row['online_costs' if phase=='inventories' else 'replay_costs']['counts']
            require(costs.get('native_previews',0)==local_previews,'inventory_preview_sum')
            if mode==METHODS[0]:
                require(costs['fallback_gate_states']==len(actions) and costs['fallback_triggered_states']==local_triggers and costs['fallback_additional_previews']==local_extra and costs['fallback_nominal_reward_checks']==local_checks,'exact_fallback_costs')
            else:require(costs.get('ridge_calls',0)==costs.get('feature_rows',0)==0,'lean_exhaustive_no_ranker_cost')
            inventory_counts.append(local_previews);ridge_calls+=costs.get('ridge_calls',0)
        transitions+=accounting['model_transition_calls']+len(history)
        previews+=row['online_previews']+row['replay_previews']
        records.append({'method':mode,'endpoint':row['endpoint'],'actions':len(actions),'nonstop_actions':outcomes['nonstop_actions'],
            'target_removed_mm3':outcomes['target_removed_mm3'],'total_target_mm3':outcomes['total_reference_target_mm3'],
            'fraction_removed':outcomes['reference_target_fraction_removed'],'normal_removed_mm3':outcomes['normal_removed_mm3'],
            'reward':outcomes['total_reward'],'contact_upper_bound_mm3':outcomes['cumulative_contacted_tissue_upper_bound_mm3'],
            'online_previews':row['online_previews'],'replay_previews':row['replay_previews'],'inventory_previews':inventory_counts,
            'source_preparation_seconds':row['source_preparation_seconds'],'source_through_plan_seconds':row['source_through_plan_seconds'],
            'replay_source_preparation_seconds':row['replay_source_preparation_seconds'],'replay_and_audit_seconds':row['replay_and_audit_seconds'],
            'online_costs':row['online_costs'],'replay_costs':row['replay_costs'],
            'fallback_gates':gate_summaries,'STOP_scope':accounting['STOP_meaning'],'plan_sha256':row['files'][mode+'/plan.json']})
    require(ridge_calls==result['ridge_calls'] and transitions==result['native_transitions'] and previews==result['native_previews'],'aggregate_operation_accounting')
    require(first_inventory==result['initial_public_proposals'],'initial_catalog_summary')
    a,b=arms[0]['outcomes'],arms[1]['outcomes'];c=result['comparison'];tol=c['numeric_tolerance']
    quality=(a['target_removed_mm3']>=b['target_removed_mm3']-tol and a['total_reward']>=b['total_reward']-tol
        and a['normal_removed_mm3']<=b['normal_removed_mm3']+tol and a['cumulative_contacted_tissue_upper_bound_mm3']<=b['cumulative_contacted_tissue_upper_bound_mm3']+tol)
    require(c['no_quality_loss']==quality and c['favorable_outcome_required_for_completion'] is False,'unselected_quality_result')
    close(c['target_difference_mm3'],a['target_removed_mm3']-b['target_removed_mm3']);close(c['reward_difference'],a['total_reward']-b['total_reward'])
    close(c['preview_ratio'],arms[0]['online_previews']/max(1,arms[1]['online_previews']))
    close(c['source_through_plan_wall_ratio'],arms[0]['source_through_plan_seconds']/arms[1]['source_through_plan_seconds'])
    report={'status':'PASS_SAVED_TRAIN050_FALLBACK_COMPARISON_QUALITY_FALSE','release_sha256':RELEASE,'source_index_sha256':INDEX,'runtime_source_files':len(index['files']),
        'parent_sha256':args.parent_sha256,'result_sha256':args.result_sha256,'fixed_subject':'ReMIND-050','role':'TRAIN','planned_arms':2,'complete_arms':2,
        'model_identity':release['ridge'],'model_parameters_reopened':False,'fit_subjects':TRAIN,'arms':records,'comparison':c,
        'counts':{'native_previews':previews,'native_transitions':transitions,'ridge_calls':ridge_calls,'source_visits':4,'model_loads':1,'refits':0},
        'costs':{'worker_seconds':result['elapsed_seconds'],'parent_seconds':parent['elapsed_seconds'],'sampled_owned_peak_rss_bytes':parent['sampled_peak_rss_bytes'],'output_bytes':output_bytes,'supervision_bytes':supervision_bytes},
        'cleanup':{'exit_code':parent['exit_code'],'reaped':parent['worker_termination_confirmed'],'remaining':parent['remaining_owned_pids'],'errors':parent['cleanup_errors']},
        'source_scope':{'full_target_voxels':19557,'source_support_unknown_voxels':183804,'known_support_zero_target_voxels':9766,'target_in_unknown_support':0},
        'limitations':['One original TRAIN person outside this fixed ridge fit cohort; no independently held-out SELECT claim.',
            'Both methods greedily choose current verified reward; negative preparation and globally beneficial routes are not exhausted.',
            'Fallback STOP after expansion concerns the emitted finite catalog, not all possible tool paths. Positive initial shortlists are not expanded and can still omit higher-reward actions.',
            'Saved worker performed independent full-tool geometry checks; this reviewer verified saved metadata/history/accounting, not native geometry or patient arrays.',
            'One fixed-order pair; timings include cache/source components and are not a repeated causal speed estimate.',
            'RSS ceiling is sampled; dependency binaries/import origins are not independently reconstructed from these source pins.',
            'S union T rigid material and post-exposure external workspace are declared simulation assumptions, not physical/anatomical validation.'],
        'read_boundary':{'patient_arrays':False,'model_parameters':False,'original_DICOM':False,'images':False,'production_imports':False,'native_replay':False},'read_pins':READS}
    original=read(result['original_negative_result']['path'],result['original_negative_result']['sha256'])
    oldbase=ROOT/'build/hybrid-train050-transfer-v1/attempt-01'
    oldplan=read(oldbase/'balanced_ridge_top8/plan.json')
    oldex=read(oldbase/'exhaustive/plan.json')
    plans={m:sealed['plans'][m] for m in METHODS}
    fplan=plans[METHODS[0]];eplan=plans[METHODS[1]]
    require(fplan['actions'][:2]==oldplan['actions'][:2] and oldplan['actions'][2]=='STOP','unchanged_original_two_action_prefix')
    fdec=fplan['accounting']['decisions'];edec=eplan['accounting']['decisions'];olddec=oldplan['accounting']['decisions']
    require(fdec[:2]==olddec[:2] and fdec[2]['source_state_hash']==olddec[2]['source_state_hash'],'same_saved_original_STOP_state')
    require(fdec[2]['selected_action_id']!='STOP' and records[0]['fallback_gates'][2]['triggered'],'fallback_rescues_original_STOP')
    require(eplan['actions']==oldex['actions'] and arms[1]['outcomes']==original['arms'][1]['outcomes'] and all(v==original['public_world'][k] for k,v in result['public_world'].items() if k!='protocol'),'fresh_exhaustive_matches_historical_actions_outcomes_source')
    old_public=read(oldbase/'exhaustive/source/admitted-public-bindings.json')
    require({k:v for k,v in bindings['protocol'].items() if k!='runtime_release_sha256'}=={k:v for k,v in old_public['protocol'].items() if k!='runtime_release_sha256'},'historical_protocol_differs_only_exact_runtime_release')
    require(bindings['protocol']['runtime_release_sha256']==RELEASE and old_public['protocol']['runtime_release_sha256']==original['release_sha256'],'historical_runtime_release_bound')
    require(fdec[1]['source_state_hash']==edec[1]['source_state_hash'],'same_second_decision_state')
    missing=edec[1]['selected_action_id'];fi=read(OUT/METHODS[0]/'inventories/state-01.json')
    rank_ids=fi['shortlist']['raw_ids'];priority=fi['shortlist']['priority_scores'];rank=[rank_ids[j] for j in sorted(range(len(rank_ids)),key=lambda j:(-priority[j],j))].index(missing)+1
    require(missing not in fi['shortlist']['selected_ids'] and not fi['shortlist']['fallback_triggered'],'positive_shortlist_miss_persists')
    excess_actions=a['nonstop_actions']-b['nonstop_actions'];excess_switches=a['tool_changes']-b['tool_changes'];excess_path=a['complete_tool_path_length_mm']-b['complete_tool_path_length_mm']
    penalty=.03*excess_actions+.03*excess_switches+.001*excess_path
    close(-c['reward_difference'],penalty)
    histories={m:read(OUT/m/'replay/episode-metrics.json')['history'] for m in METHODS}
    removed=[]
    for m in METHODS:
        cells=[tuple(v) for h in histories[m] for v in h.get('removed_indices_native',[])]
        require(len(cells)==len(set(cells)),'no_repeat_removed_cells');removed.append(set(cells))
    require(len(removed[0])==len(removed[1])==37,'exact_saved_removed_cell_counts')
    report['cause']={'original_STOP_step':2,'rescued_action':fdec[2]['selected_action_id'],'rescued_state':fdec[2]['source_state_hash'],
        'positive_shortlist_miss':{'step':1,'missed_exhaustive_winner':missing,'ridge_rank':rank,'selected':fdec[1]['selected_action_id']},
        'same_saved_removed_cells':removed[0]==removed[1],'additional_actions':excess_actions,'additional_switches':excess_switches,'additional_complete_path_mm':excess_path,
        'reward_shortfall_components':{'action_cost':.03*excess_actions,'tool_switch_cost':.03*excess_switches,'motion_cost':.001*excess_path,'sum':penalty},
        'scope':'Saved same-state and history evidence; no counterfactual patient execution or global route claim.'}
    report['historical_exhaustive_control']=result['historical_exhaustive_control']
    output=HERE/'audit.json'
    with output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'path':str(output.relative_to(ROOT)),'sha256':sha(output.read_bytes()),'status':report['status'],'arms':[{k:v for k,v in r.items() if k in ('method','actions','nonstop_actions','target_removed_mm3','normal_removed_mm3','reward','online_previews','source_through_plan_seconds')} for r in records]},indent=2))

if __name__=='__main__':main()
