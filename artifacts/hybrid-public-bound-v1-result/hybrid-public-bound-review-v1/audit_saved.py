"""Independent saved-only audit of the first five-pair bound run. No runtime imports."""
import collections
import hashlib
import json
import math
from pathlib import Path
import stat
import subprocess

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
BASE=ROOT/'build/hybrid-public-bound-v1';OUT=BASE/'attempt-01';SUP=BASE/'attempt-01.supervision'
HEAD='f8c7e18120475e895453c9964bbd65fe40792292'
RELEASE='90dfb0a567f3e950c8fcf5d4d1b0917133ac976f9b7f40fc719672d463264a89'
INDEX='a0f1bcf643887a61d174d2f2459bd9736fc4d7d06ba3ce2bcdfb57c04ffae6da'
RESULT='2f07741f2605ca2033b3d78547447bb5cf345ea038db7a130d51caa356091707'
PARENT='1b1c603c7c9a2c06e59ee0733055b0f23e84a9f4a583dfbd59892e3ffd0d574b'
SUBJECTS=['ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045','ReMIND-050']
MODES=['switch_guard','public_bound'];READS={}
def need(ok,why):
    if not ok:raise ValueError(why)
def sha(data):return hashlib.sha256(data).hexdigest()
def semantic(value):return 'sha256:'+sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def raw(path,pin=None):
    path=ROOT/Path(path);need(path.suffix in ('.json','.py','.log'),'metadata_source_only')
    need(path.name not in ('balanced_ridge_top8.json','old_ridge_top8.json'),'no_model_parameters')
    a=path.lstat();need(stat.S_ISREG(a.st_mode) and a.st_size<16*1024**2,'bounded_regular_metadata')
    data=path.read_bytes();b=path.stat();need((a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns) and len(data)==a.st_size,'unchanged_file')
    h=sha(data);need(pin is None or h==pin,'sha:'+str(path));READS[str(path.relative_to(ROOT))]={'sha256':h,'bytes':len(data)};return data
def read(path,pin=None):return json.loads(raw(path,pin),parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite')))
def close(a,b):need(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-7),'numeric_join')
def quality(a,b):
    signs={'target_removed_mm3':1,'total_reward':1,'normal_removed_mm3':-1,'cumulative_contacted_tissue_upper_bound_mm3':-1,'currently_retained_contacted_tissue_upper_bound_mm3':-1,'complete_tool_path_length_mm':-1,'nonstop_actions':-1,'tool_changes':-1}
    q={k:v*(a[k]-b[k])>=-1e-7 for k,v in signs.items()}
    q.update(same_full_target_denominator=a['total_reference_target_mm3']==b['total_reference_target_mm3'],physical_probability_remains_null=a['clinical_deficit_probability'] is b['clinical_deficit_probability'] is None,unchanged_partial_contact_reward=a['partial_contact_reward_weight']==b['partial_contact_reward_weight']==0.)
    return q
def extent(folder):
    total=0
    for p in folder.rglob('*'):
        s=p.lstat();need(stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode),'regular_output_tree')
        if stat.S_ISREG(s.st_mode):total+=s.st_size
    return total
def main():
    rel=read(BASE/'root-release.json',RELEASE);idx=read(BASE/'source-index.json',INDEX)
    need(rel['expected_head']==idx['head']==HEAD and rel['execution_released'],'released_head')
    # Reuse reviewed mapping; hash only changed/new paths and compare canonical
    # sources with the executed commit, not a repeated full filesystem census.
    prior=read('build/hybrid-fixed-four-switch-regression-v1/source-index.json','bd9783ac5f0bc9dd94a889cb99c1bcfae9f6008a3190056d2bef424c0028cb32')
    delta={p:h for p,h in idx['files'].items() if prior['files'].get(p)!=h}
    for p,h in delta.items():raw(p,h)
    diff=subprocess.check_output(['git','diff','--name-only',HEAD,'--','src/resectionlab'],cwd=ROOT,text=True)
    need(not diff.strip(),'canonical_runtime_changed_before_capture')
    current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source_capture={'executed_head':HEAD,'captured_head':current,'canonical_diff':[],'changed_new_pins':delta,'inherited_unchanged_mappings':len(idx['files'])-len(delta)}
    (HERE/'runtime-source-capture.json').write_text(json.dumps(source_capture,indent=2)+'\n')
    parent=read(SUP/'receipt.json',PARENT);r=read(OUT/'result.json',RESULT);caps=rel['caps']
    need(parent['status']=='complete' and parent['exit_code']==0 and parent['worker_termination_confirmed'] and not parent['remaining_owned_pids'] and not parent['cleanup_errors'] and parent['stop_reason'] is None,'clean_complete')
    need(parent['result_sha256']==RESULT and parent['release_sha256']==r['release_sha256']==RELEASE and parent['source_index']==rel['source_index'] and parent['caps']==caps,'root_receipt_joins')
    need(rel['subjects']==r['subjects']==SUBJECTS and rel['arms']==MODES and rel['horizon']==24 and rel['K']==8,'fixed_five_pairs')
    need(r['status']=='complete_five_TRAIN_public_bound_comparison' and r['role']=='TRAIN' and r['source_visits']==20 and r['ridge_loads']==1,'complete_denominator')
    need(all(r[k]==0 for k in ('fits','optimizer_calls','model_calls','checkpoint_loads','blocked_external_calls')) and not r['SELECT_EVAL_opened'] and not r['private_reference_loaded'],'zero_updates_private_closed')
    need(r['ridge_unchanged'] and r['ridge_fingerprint']==rel['ridge']['fingerprint'] and r['cache_restored'],'frozen_model_cache_restored')
    for k in ('native_previews','native_transitions','ridge_calls'):need(r[k]<=caps[k],'cap:'+k)
    need(r['elapsed_seconds']<caps['worker_seconds'] and parent['elapsed_seconds']<caps['parent_seconds'] and parent['sampled_peak_rss_bytes']<caps['sampled_rss_bytes'],'resource_caps')
    outbytes=extent(OUT);supbytes=extent(SUP)
    need(outbytes==parent['output_bytes'] and outbytes<caps['output_bytes'] and supbytes<caps['supervision_bytes'],'output_cap')
    need((SUP/'worker.log').stat().st_size<caps['log_bytes'],'log_cap')
    launch=read(SUP/'declaration.json');need(launch['head']==HEAD and launch['release_sha256']==RELEASE and launch['source_index']==rel['source_index'],'launch_bindings')
    need('-I' in launch['command'] and '-B' in launch['command'] and any(x.startswith('pycache_prefix=') for x in launch['command']),'fresh_startup')
    train=read(rel['training_result']['path'],rel['training_result']['sha256']);tr=read(rel['training_release']['path'],rel['training_release']['sha256']);tp=read(rel['training_parent']['path'],rel['training_parent']['sha256'])
    need(tp['status']=='complete' and tp['result_sha256']==rel['training_result']['sha256'] and tp['release_sha256']==rel['training_release']['sha256'],'terminal_training')
    need(tr['subjects']==SUBJECTS[:4] and train['ridge_file_sha256']['balanced_ridge_top8']==rel['ridge']['sha256'],'exact_fitted_model')
    fifty=read(rel['prior_transfer_result']['path'],rel['prior_transfer_result']['sha256'])
    originals={(x['method'],x['subject']):x for x in train['arms']}
    public_index=read(rel['public_index']['path'],rel['public_index']['sha256']);entries={x['patient_id']:x for x in public_index['cases']};entries[SUBJECTS[-1]]=rel['transfer_inputs']['public_manifest']
    need(list(entries)==SUBJECTS,'exact_public_subjects')
    arms=r['arms'];keys=[m+'/'+s for s in SUBJECTS for m in MODES]
    need([x['method']+'/'+x['subject'] for x in arms]==keys and all(x['complete'] and x['independent_accepted'] for x in arms),'ten_complete_results')
    seals=read(OUT/'sealed-plans.json');need(set(seals['plans'])==set(keys) and r['all_plans_sealed_before_evaluation'] and not seals['private_reference_loaded'],'ten_plans_sealed')
    visits=r['source_visits_detail'];need([x['phase'] for x in visits]==['planning.'+s for s in SUBJECTS for _ in MODES]+['replay.'+s for s in SUBJECTS for _ in MODES],'all_planning_precedes_evaluation')
    totals=collections.Counter();stored={};rows=[];reason_totals=collections.Counter()
    for row in arms:
        s=row['subject'];m=row['method'];key=m+'/'+s;folder=OUT/key
        for p,h in row['files'].items():raw(OUT/p,h)
        plan=read(folder/'plan.json');mt=read(folder/'replay/episode-metrics.json');ev=read(folder/'replay/independent-episode.json')
        need(plan==seals['plans'][key] and semantic({k:v for k,v in plan.items() if k!='seal'})==plan['seal'],'immutable_plan')
        actions=plan['actions'];history=mt['history'];o=ev['outcomes'];ac=plan['accounting']
        need(plan['method']==m and plan['subject']==s and plan['max_steps']==24 and plan['ridge_hash']==rel['ridge']['fingerprint'] and len(actions)==row['action_count']<=24,'plan_contract')
        need(ac['complete'] and not ac['global_optimality_proven'] and [d['selected_action_id'] for d in ac['decisions']]==actions,'complete_decisions')
        for d in ac['decisions']:
            scores=d['scores'];need(scores[0]['action_id']=='STOP' and scores[0]['reward']==0 and max(scores,key=lambda x:x['reward'])['action_id']==d['selected_action_id'] and d['all_queried_legal_actions_scored'],'unchanged_greedy_reward')
        need([h['action_id'] for h in history]==actions and semantic(history)==ev['committed_history_hash'] and o==row['outcomes'],'replayed_history')
        need(ev['accepted'] and ev['complete_episode'] and ev['geometry']['feasible'] and mt['terminated'],'saved_geometry_acceptance')
        need(mt['source_hash']==ev['source_hash']==plan['source_hash'] and mt['decision_model_hash']==ev['decision_model_hash']==plan['decision_model_hash'],'replay_identity')
        for k,hk in [('total_reward','reward'),('target_removed_mm3','target_removed_mm3'),('normal_removed_mm3','normal_removed_mm3'),('complete_tool_path_length_mm','complete_tool_path_length_mm')]:close(math.fsum(h[hk] for h in history),o[k])
        old=originals['exhaustive',s] if s in SUBJECTS[:4] else next(x for x in fifty['arms'] if x['method']=='exhaustive')
        gates=quality(o,old['outcomes']);need(gates==row['historical_quality']['quality_gates_vs_saved_exhaustive'] and all(gates.values())==row['historical_quality']['no_quality_loss_vs_saved_exhaustive'],'historical_quality_recomputed')
        pb=read(folder/'source/admitted-public-bindings.json');rb=read(folder/'replay/source/admitted-public-bindings.json');need(pb==rb,'source_reload_exact')
        b,q,p=pb['source_binding'],pb['qc'],pb['protocol'];w=r['public_worlds'][s]
        need(b['subject']==p['subject']==s and p['role']=='TRAIN' and b['source_hash']==plan['source_hash']==w['source'],'fixed_role_world')
        need(semantic(b)==w['binding'] and semantic(q)==w['qc'] and semantic(p)==w['protocol'],'source_metadata_identity')
        need(all(v['status']=='complete' and all(v[k]==val for k,val in w.items()) for v in visits if v['subject']==s),'all_four_visits_same_world')
        need(p['max_optimizer_updates']==p['max_policy_forwards']==0 and not p['private_reference_used'] and not p['split_changes'],'no_updates_or_role_changes')
        if s in SUBJECTS[:4]:need('development_transfer' not in p and p['initialization']=='public_world_search_only' and w['source']==rel['source_world_hashes'][s],'original_four_admission')
        else:need(p['development_transfer']['subject']==s and p['runtime_release_sha256']==RELEASE and p['development_transfer']['training_subjects']==SUBJECTS[:4],'050_prior_transfer_admission')
        for ph,ck in [('online_costs','online_previews'),('replay_costs','replay_previews')]:
            c=row[ph]['counts'];need(c['native_previews']==row[ck] and c['ridge_calls']==c['proposal_calls']==c['fallback_gate_states']==len(actions),'phase_count_joins');totals['ridge_calls']+=c['ridge_calls'];totals['previews']+=row[ck]
        totals['transitions']+=ac['model_transition_calls']+len(history)
        bound_evidence=None
        if m=='public_bound':
            reasons=collections.Counter();setup_states=0
            for number in range(len(actions)):
                inv=read(folder/'inventories'/f'state-{number:02d}.json');ri=read(folder/'replay/inventories'/f'state-{number:02d}.json');need(inv==ri,'identical_bound_inventory_on_uncached_replay')
                sh=inv['shortlist'];need(inv['source_state_hash']==ac['decisions'][number]['source_state_hash'],'decision_bound_state')
                bounds=sh['public_reward_bounds'];reasons.update(x['reason'] for x in bounds)
                setup_states+=bool(sh['fallback_triggered'] or sh['current_tool_comparison_triggered'])
                need(all(x['reward_upper'] is None and not x['pruned'] and x['reason']=='public_index_workspace_limit' for x in bounds),'all_exact_cap_refusals')
                need(sh['bound_pruned_count']==0 and sh['additional_selected_ids']==[x['proposal_id'] for x in bounds],'every_unavailable_candidate_received_exact_preview')
            c=row['online_costs']['counts'];need(sum(reasons.values())==c['public_bound_queries']==c['public_bound_unavailable'] and setup_states==c['public_bound_states'],'bound_accounting')
            need(c['public_bound_pruned']==c['public_bound_index_target_cells']==c['public_bound_index_unknown_cells']==0,'zero_bound_activity')
            manifest=read(entries[s]['path'],entries[s]['sha256']);cells=math.prod(manifest['shape_xyz']);known=manifest['public_label_resampling']['whole_tumor']['saved_domain_voxels'];unknown=cells-known
            need(unknown>1000000 and manifest['input_files']['whole_tumor_domain']['sha256']==b['target_domain_source_sha256'],'target_domain_cap_proved_by_saved_metadata')
            bound_evidence={'reason_counts_online':dict(reasons),'reason_counts_replay':dict(reasons),'indexed_target_cells':0,'indexed_unknown_cells':0,'pruned':0,'setup_states_each_phase':setup_states,'grid_cells':cells,'source_target_domain_known_cells':known,'source_target_domain_unknown_cells':unknown,'global_index_cap':1000000,'source_support_domain_unknown_cells':b['source_and_simulated_domains']['source_support_unknown_voxels']}
            reason_totals.update(reasons);reason_totals.update(reasons)
        stored[key]=(plan,mt,ev,pb)
        rows.append({'subject':s,'method':m,'action_count':len(actions),'outcomes':o,'historical_quality_recomputed':gates,'online_previews':row['online_previews'],'replay_previews':row['replay_previews'],'source_through_plan_seconds':row['source_through_plan_seconds'],'replay_and_audit_seconds':row['replay_and_audit_seconds'],'source_preparation_seconds':row['source_preparation_seconds'],'replay_source_preparation_seconds':row['replay_source_preparation_seconds'],'online_costs':row['online_costs'],'replay_costs':row['replay_costs'],'bound_diagnosis':bound_evidence})
    pairs=[]
    for s in SUBJECTS:
        bp,bm,be,bs=stored['switch_guard/'+s];up,um,ue,us=stored['public_bound/'+s]
        result={'subject':s,'exact_actions':bp['actions']==up['actions'],'exact_metrics_except_outer_policy_identity':bm=={**um,'decision_model_hash':bm['decision_model_hash']},'exact_history':bm['history']==um['history'],'exact_outcomes':be['outcomes']==ue['outcomes'],'same_public_source_inputs':bs==us}
        declared=next(x for x in r['comparison']['pairs'] if x['subject']==s)
        need(all(v for k,v in result.items() if k!='subject') and all(declared[k]==result[k] for k in ('exact_actions','exact_metrics_except_outer_policy_identity','exact_history','exact_outcomes')),'exact_physical_pair')
        pairs.append(result)
    need(totals['previews']==r['native_previews'] and totals['transitions']==r['native_transitions'] and totals['ridge_calls']==r['ridge_calls'],'aggregate_costs')
    need(r['comparison']['exact_policy_equivalence'] and not r['comparison']['favorable_outcome_required_for_completion'],'retained_equivalence_negative_efficiency')
    budget=r['native_budget'];need(budget['counting_reliable'] and budget['failure'] is None and budget['native_preview_entries']==r['native_previews'] and budget['blocked_preview_attempts']==0,'native_accounting_guard')
    report={'status':'PASS_SAVED_EXACT_PAIRS_NEGATIVE_BOUND_ACTIVATION','result_sha256':RESULT,'parent_sha256':PARENT,'release_sha256':RELEASE,'source_index_sha256':INDEX,'source_capture':source_capture,'pairs':pairs,'rows':rows,'totals':dict(totals),'bound_refusals_both_phases':dict(reason_totals),'costs':{'worker_seconds':r['elapsed_seconds'],'parent_seconds':parent['elapsed_seconds'],'sampled_owned_peak_rss_bytes':parent['sampled_peak_rss_bytes'],'output_bytes':outbytes,'supervision_bytes':supbytes},'cleanup':{'reaped':True,'remaining':[],'errors':[]},'diagnosis':'Every conditional bound query fails open at the global one-million-cell index cap before per-envelope target-coverage testing. All five saved whole-tumor source domains leave more than one million unknown grid cells. Factory and wrapper use explicit source Dt correctly; no missing-field adapter bug. This does not prove every local tool envelope intersects unknown target coverage. Do not relabel unknown source coverage as known zero or inflate cap automatically.','limitations':['No acquired arrays, model parameters, runtime imports, tests or native replay in this independent audit. Geometry acceptance is checked from saved worker evaluation.','Exact observed policy equality is established here through full fallback: zero pruning means this run does not validate activated pruning on these acquired cases.','One baseline-first pair per already-exposed TRAIN case; no held-out or causal speed claim, sampled memory may miss peaks.','Full supplied target denominator and explicit S union T/post-exposure material and coverage assumptions remain unchanged.'],'read_pins':READS}
    with (HERE/'audit.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'audit_sha256':sha((HERE/'audit.json').read_bytes()),'pairs':len(pairs),'totals':dict(totals),'bound_refusals':dict(reason_totals)},indent=2))
if __name__=='__main__':main()
