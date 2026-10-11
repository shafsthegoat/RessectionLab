"""Saved JSON/source only: no project imports, arrays, geometry calls or models."""
import hashlib, json, math
from pathlib import Path
HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
ROOT=STAGE.parents[1]
checks=[]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def need(ok,label):
    if not ok: raise AssertionError(label)
    checks.append(label)
def pin(p,expected): need(sha(p)==expected,'hash '+str(p.relative_to(ROOT)))
def semantic(x): return 'sha256:'+hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def physical(row): return {k:row[k] for k in ('tool_id','entry_mm','tip_mm')}
def close(a,b): return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10)
result_path=STAGE/'attempt-01/result.json';receipt_path=STAGE/'attempt-01.supervision/receipt.json'
pin(result_path,'823687e42979c259d3167a07684ed69efc8a14ae4d602a7a3d9cc3d386f51a65')
pin(receipt_path,'44ff4d89b80395700a9e56137989d5a6b504a63f59847d3e298a3a8d08e71b2b')
r=read(result_path);receipt=read(receipt_path)
pin(STAGE/'root-release.json',receipt['release_sha256']);release=read(STAGE/'root-release.json')
need(release['source_index']==receipt['source_index'],'release receipt index identity')
pin(ROOT/release['source_index']['path'],release['source_index']['sha256']);index=read(ROOT/release['source_index']['path'])
need(index['head']==release['expected_head']=='e4c4105c36974538297f2109292c5d4993bf344d','frozen HEAD')
for name,digest in index['source_files'].items():
    need(Path(name).suffix=='.py','source-only suffix '+name);pin(ROOT/name,digest)
pin(STAGE/'attempt-01.supervision/worker-final.json',receipt['worker_final_sha256']);wf=read(STAGE/'attempt-01.supervision/worker-final.json')
need(wf['result_sha256']==wf['canonical_result_sha256']==receipt['result_sha256']==sha(result_path),'terminal result joins')
need(receipt['status']=='complete' and receipt['exit_code']==0 and receipt['worker_termination_confirmed'] and not receipt['cleanup_errors'] and not receipt['final_owned_pids'],'complete clean owned terminal')
need(r['status']==receipt['canonical_result_status']=='complete_SELECT013_initial_goal_burden','completion status')
need(r['planned_SELECT_denominator']==receipt['planned_SELECT_denominator']==1 and set(r['arms'])=={'initial_state'},'single diagnostic denominator')
need(r['arms']==receipt['arms'],'parent result arm identity')
need(receipt['elapsed_seconds']<=150 and receipt['sampled_peak_rss_bytes']<=2*1024**3 and receipt['output_bytes']<=1024**2,'owned caps')
actual_bytes=sum(p.stat().st_size for p in (STAGE/'attempt-01').rglob('*') if p.is_file())
need(actual_bytes==receipt['output_bytes'],'actual saved output byte count')
need(r['rank_helper_calls']==1 and r['native_previews_total']==117 and r['native_previews_ranking']==0,'one rank and construction-only preview count')
for key in ('checkpoint_loads','model_forwards','updates','optimizer_attempts','gradient_attempts','native_commit_attempts','native_transition_attempts','private_reference_reads'):
    need(r[key]==0,'zero '+key)
need(r['state_unchanged'] and r['no_action_history'] and r['history_complete_attestation'] is None,'unchanged initial state without episode claim')
need(not r['held_inputs_opened'] and not r['EVAL_opened'] and r['held_cases']==['ReMIND-037'],'held and EVAL remain closed')
qpath=STAGE/'attempt-01/initial-goal-burden.json';pin(qpath,r['ranking_report_sha256']);q=read(qpath)
need(semantic({k:v for k,v in q.items() if k!='seal'})==q['seal'],'independent ranking semantic seal')
need(q['complete'] and q['native_preview_calls']==q['native_transitions']==0 and not q['private_reference_used'],'helper completed without action/private read')
rows=q['rows'];need(len(rows)==q['unique_goal_count']==q['same_tool_goal_rows']==26,'all 26 unique existing same-tool goals')
need(len({json.dumps(physical(x),sort_keys=True) for x in rows})==26,'physical goal uniqueness')
need(q['base_emitted_count']==117 and q['base_omitted_pairs']==0,'declared base cap accounting')
need(q['helper_capsule_cover_calls']==2*len(rows) and q['helper_containment_calls']==len(rows) and q['base_enumeration_calls']==1,'helper geometry counts')
need(q['bbox_cells_charged']==31442 and q['bbox_cells_charged']<=q['limits']['max_total_bbox_cells'] and q['wall_seconds']<=q['limits']['seconds'],'helper work bounds')
binding=read(STAGE/'attempt-01/task-binding.json');public=read(STAGE/'attempt-01/source/admitted-public-bindings.json')
a=public['native_grid_reconciliation']['derived_affine_ras_mm']
vol=abs(a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))
compact=[]
for row in rows:
    label=str(row['raw_order'])
    need(row['estimated_nominal_preparation_cells']==row['occupied_shaft_cells']-row['optimistically_tip_contained_cells'],'preparation subtraction '+label)
    need(0<=row['optimistically_tip_contained_cells']<=row['occupied_shaft_cells']<=row['shaft_supercover_cells'],'nested scalar counts '+label)
    need(close(row['estimated_nominal_preparation_volume_mm3'],row['estimated_nominal_preparation_cells']*vol),'volume arithmetic '+label)
    cov=row['native_coverage']
    need(row['rankable'] and not row['unrankable_reasons'] and row['interaction_domain_declared'] and row['unknown_interaction_cells']==0 and row['hard_exclusion_supercover_cells']==0,'known rank eligibility '+label)
    need(not cov['inward_or_lateral_crop_uncertain'] and cov['orthogonal_native_grid'] and cov['source_axis_aligned'] and cov['entry_in_native_grid'],'reported coverage eligibility '+label)
    need(cov['outside_native_geometry_is_unassessed'] and len(cov['outside_native_faces'])==2 and all(x['proximal_only_unassessed'] for x in cov['outside_native_faces']) and row['external_workspace_unassessed_cells']==9,'explicit proximal uncertainty '+label)
    need(not row['static_aperture']['excluded'] and row['static_aperture']['clearance_mm']>1e-6 and row['full_tool_cells_outside_actor_crop']==0,'reported aperture and crop flags '+label)
    need(row['prior_cavity_microstep_and_exposure_constraints_ignored_optimistically'],'optimistic scope '+label)
    compact.append({k:row[k] for k in ('raw_order','estimated_nominal_preparation_cells','estimated_nominal_preparation_volume_mm3','occupied_shaft_cells','optimistically_tip_contained_cells','external_workspace_unassessed_cells')})
rank=sorted(rows,key=lambda x:(x['estimated_nominal_preparation_cells'],x['raw_order']))
need(q['ranking_raw_orders']==[x['raw_order'] for x in rank] and q['ranked_physical_goals']==[physical(x) for x in rank],'independent ranking and tie order')
need(q['minimum_score_tie_count']==sum(x['estimated_nominal_preparation_cells']==rank[0]['estimated_nominal_preparation_cells'] for x in rows)==1,'unique minimum')
control=read(STAGE/'attempt-01/baseline-goal-control.json');wp=control['historical_witness'];pin(ROOT/wp['path'],wp['sha256']);witness=read(ROOT/wp['path'])['protocols']['narrow']
need(wp['sha256']==index['metadata_files'][wp['path']],'baseline witness index join')
need(control['historical_baseline_active_goal']==witness['decision']['active_goal']==q['original_first_same_tool_goal'],'actual historical baseline physical goal equality')
base=[x for x in rows if physical(x)==witness['decision']['active_goal']]
need(len(base)==1 and base[0]==control['baseline_row'],'actual baseline row equality')
need(binding['source_hash']==q['source_hash']==witness['source_hash'] and binding['decision_model_hash']==q['task_decision_model_hash']==witness['decision_model_hash'] and binding['native_decision_model_hash']==q['native_decision_model_hash']==witness['native_decision_model_hash'] and binding['initial_state_hash']==q['source_state_hash']==witness['prefix'][0]['source_state_hash'],'exact source model initial state join')
need(control['proposed_ranked_goal']==physical(rank[0]) and physical(rank[0])!=physical(base[0]),'proposed top differs actual baseline')
need(r['baseline_goal_control']=={k:v for k,v in control.items() if k!='baseline_row'},'comparison result projection')
need(r['arms']['initial_state']['ranking_raw_orders']==q['ranking_raw_orders'] and r['helper_accounting']=={k:q[k] for k in r['helper_accounting']},'result accounting projections')
world=read(STAGE/'attempt-01/original-world-control.json');need(world['exact'] and r['world_equal'] and r['source_physics_equal'],'saved world equality')
extent=public['supplied_goal_extent'];need(extent['full_region_positive_voxels']==35260 and close(extent['full_region_membership_mm3'],35260*vol) and not extent['target_modified'],'full unchanged target denominator')
domains=extent['source_and_simulated_domains'];need(domains['source_support_unknown_voxels']==176294 and not domains['source_domain_extended'],'source unknown domain retained')
costs=read(STAGE/'attempt-01/costs.json');ph=costs['phases'];prof=costs['provider_and_native_profile']
need(sum(x['native_previews'] for x in ph.values())==117 and ph['source_construction']['native_previews']==117 and ph['initial_ranking']['native_previews']==0,'phase preview joins')
need(prof['source_construction']['feasible']+prof['source_construction']['rejected:SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE']==117,'native dispositions')
need(prof['initial_ranking']['providerpropose_calls']==1 and prof['initial_ranking']['provider_tool_footprint_endpoint_calls']==26,'separate provider count')
need(prof['initial_ranking']['provider_tool_footprint_endpoint_inclusive_seconds']<=prof['initial_ranking']['providerpropose_inclusive_seconds']<=q['base_enumeration_wall_seconds']<=q['wall_seconds']<=ph['initial_ranking']['wall_seconds'],'nested ranking timers')
need(sum(x['wall_seconds'] for x in ph.values())<=r['wall_seconds']<=wf['wall_seconds']<=receipt['elapsed_seconds'],'phase worker parent scope nesting')
summary={'status':'PASS','check_count':len(checks),'checks_sha256':semantic(checks),'result_sha256':sha(result_path),'receipt_sha256':sha(receipt_path),'source_files_verified':len(index['source_files']),
 'scope':'saved JSON and source hashes only; no project imports, original arrays, native geometry or model execution; stored geometric flags checked for consistency, not independently recomputed',
 'goal_count':26,'best_raw_order':rank[0]['raw_order'],'best_cells':rank[0]['estimated_nominal_preparation_cells'],'best_mm3':rank[0]['estimated_nominal_preparation_volume_mm3'],
 'baseline_raw_order':base[0]['raw_order'],'baseline_cells':base[0]['estimated_nominal_preparation_cells'],'baseline_mm3':base[0]['estimated_nominal_preparation_volume_mm3'],'baseline_rank_1based':rank.index(base[0])+1,
 'next_best_cells':rank[1]['estimated_nominal_preparation_cells'],'reduction_cells':base[0]['estimated_nominal_preparation_cells']-rank[0]['estimated_nominal_preparation_cells'],'reduction_fraction':1-rank[0]['estimated_nominal_preparation_cells']/base[0]['estimated_nominal_preparation_cells'],
 'rows':compact,'full_target_cells':35260,'source_unknown_domain_cells':176294,'all_rows_proximal_external_unassessed':True,'all_rows_E_cells':9,'reported_unresolved_interaction_cells_per_row':0,'aperture_clearance_range_mm':[min(x['static_aperture']['clearance_mm'] for x in rows),max(x['static_aperture']['clearance_mm'] for x in rows)],
 'helper_accounting':r['helper_accounting'],'costs':costs,'execution_seconds':r['wall_seconds'],'worker_scope_seconds':wf['wall_seconds'],'parent_seconds':receipt['elapsed_seconds'],'sampled_peak_rss_bytes':receipt['sampled_peak_rss_bytes'],'output_bytes':receipt['output_bytes'],
 'limitations':['Optimistic native-grid geometry proxy; not preparation action count, true minimal removal, route feasibility or patient improvement.','Exact tools lack completed resolution-consistency validation; full route test held pending that separate generated result.','All goals retain proximal outside-image and declared E uncertainty.','Timer scopes are nested; native/provider/helper durations must not be summed as disjoint costs.']}
(HERE/'audit.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:summary[k] for k in ('status','check_count','best_cells','baseline_cells','baseline_rank_1based','source_files_verified')}))
