"""Saved JSON/hash/scalar review only: no project import, arrays or native replay."""
from pathlib import Path
from collections import Counter
import hashlib,json,math,subprocess,traceback
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'build/native-cut-union-v1';OUT=BASE/'independent-result'
checks=0; inputs={}
def need(value,label):
 global checks
 checks+=1
 if not value:raise AssertionError(label)
def close(a,b,label):need(math.isclose(a,b,rel_tol=2e-10,abs_tol=2e-8),label)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def semantic(value):return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):
 p=ROOT/p if isinstance(p,str) else p
 inputs[str(p.relative_to(ROOT))]={'sha256':digest(p),'bytes':p.stat().st_size}
 return json.loads(p.read_text())
def det(a):return (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))
def physical(tool,d):
 ans=[];scale=d['preparation_lattice_mm'][tool]
 for i,(x,y) in enumerate(d['preparation_order_lattice_units'],1):
  ans.append({'index':i,'kind':'side_preparation','tool_id':tool,'entry_mm':[scale*x,scale*y,-2],'tip_mm':[scale*x,scale*y,d['preparation_tip_z_mm']]})
 ans.append({'index':10,'kind':'target_stroke','tool_id':tool,'entry_mm':d['entry_mm'],'tip_mm':[0.,0.,d['target_tip_z_mm']]})
 return ans

def main():
 r=read(BASE/'run-01/report.json'); launch=read(BASE/'run-01/launcher.json');d=read(BASE/'declaration.json')
 release=read(BASE/'root-release.json');pros=read(BASE/'run-01/prospective-source.json');progress=read(BASE/'run-01/progress.json')
 old=read(d['retained_baseline']['report_path']);od=read(d['retained_baseline']['declaration_path'])
 need(inputs[d['retained_baseline']['report_path']]['sha256']==d['retained_baseline']['report_sha256'],'retained baseline bytes')
 need(release['head']=='9fc4a50d66c562b5df065175686c3df3e6685ccb','root committed source')
 c2=read(BASE/'root-controls/attempt-02.json'); c3=read(BASE/'root-controls/attempt-03.json');parity=read(BASE/'root-controls/parity-diagnostic.json')
 need(digest(BASE/'root-controls/attempt-03.json')==release['controls_sha256'],'released corrected controls')
 need(c2['exit_code']==1 and c3['exit_code']==0 and c3['child_reaped'] and not c3['timeout'],'preserved failed/final control outcomes')
 need(pros['source']==release['source']==r['source'],'three exact source snapshots')
 need(pros['declaration']==d and r['declaration_hash']==semantic(d),'declaration joins')
 for p,h in r['source']['files'].items():need(digest(ROOT/p)==h,'source unchanged '+p)
 for p in ('src/resectionlab/native_resection.py','src/resectionlab/native_cut_union.py','src/resectionlab/independent_cut_union.py','src/resectionlab/evaluation.py'):
  raw=subprocess.check_output(['git','show',release['head']+':'+p],cwd=ROOT)
  need(hashlib.sha256(raw).hexdigest()==r['source']['files'][p],'committed scientific source '+p)
 for key in d['fixed_baseline_fields']:need(d[key]==od[key],'fixed physical/reward declaration '+key)
 need(r['status']==launch['status']=='completed' and launch['worker_exit_code']==0 and launch['parent_stop_reason'] is None,'terminal success')
 need(r['source_unchanged'] and launch['source_unchanged'] and launch['numerical_rows_complete'] and launch['independent_audits_complete'],'terminal checks')
 need(r['human_patients']==r['training_updates']==0 and r['numerical_threads']==1,'no patient or learning')
 expected=[g['name']+'-'+format(h,'g')+'mm' for h in d['spacings_mm'] for g in d['grids']]
 need(r['planned_rows']==launch['planned_rows']==expected and not r['missing_rows'] and not launch['missing_rows'],'18 planned rows')
 need(len(r['rows'])==r['completed_rows']==r['expected_rows']==18,'18 actual rows')
 need([x['row_id'] for x in r['rows']]==expected,'actual fixed order')
 need(r['planned_action_slots']==launch['planned_action_slots']==180,'180 denominator')
 prior={x['row_id']:x for x in old['rows']};matched={x['row_id']:x for x in r['matched_comparison']}
 need(len(prior)==len(matched)==18,'unique baseline/comparison')
 counts=Counter(); cost=Counter();rows=[];gained=[];regressed=[];positive_gain=[];audit_seconds=0.;row_seconds=0.;native_seconds=0.;max_leaf=max_witness=max_cells=0;coverage_only=0
 reward=d['reward'];union={**d['experimental_cut_union'],'mode':d['experimental_mode']}
 for row in r['rows']:
  rid=row['row_id'];baseline=prior[rid];native=row['native'];acts=row['actions'];cov=row['coverage'];tool=row['fixed_tool'];aff=row['affine'];volume=abs(det(aff))
  need(len(acts)==row['planned_actions']==10,rid+' slots')
  for k in ('spacing_mm','grid','shape','affine','source_hash','fixed_tool','physical_actions_sha256'):need(row[k]==baseline[k],rid+' matched '+k)
  need(row['cut_union_spec']==union,rid+' union spec')
  pa=physical(tool['tool_id'],d);need(semantic(pa)==row['physical_actions_sha256'],rid+' action digest')
  height=d['target_tip_z_mm']+tool['tip_radius_mm']-d['target_start_z_mm'];cap=math.pi*height**2*(tool['tip_radius_mm']-height/3)
  close(cap,row['exact_continuous_target_cap_mm3'],rid+' analytic target cap')
  need(d['preparation_tip_z_mm']+tool['tip_radius_mm']<d['target_start_z_mm'],rid+' preparations cannot reach target')
  accepted=[];attempts=[];first_refusal=None;prev_state=None;prev_coverage=None
  for a,declared in zip(acts,pa):
   for k,v in declared.items():need(a[k]==v,rid+' physical action '+k)
   counts[a['status']]+=1
   if a['status']=='unexecuted_prior_refusal':need(first_refusal is not None and 'cut_union_attempt' not in a,rid+' suffix no calls');continue
   need(first_refusal is None,rid+' no retry after refusal');attempts.append(a)
   need(a['native_seconds']>=0,rid+' action time');native_seconds+=a['native_seconds']
   if prev_state is not None:need(a['source_state_hash']==prev_state,rid+' state chain')
   work=a['cut_union_attempt'];max_leaf=max(max_leaf,work['leaf_tests']);max_witness=max(max_witness,work['witness_rows']);max_cells=max(max_cells,work['peak_tracked_cells'])
   for k in ('leaf_tests','new_leaf_count','witness_rows','peak_tracked_cells'):need(type(work[k]) is int and work[k]>=0,rid+' integer work '+k)
   need(work['leaf_tests']<=union['max_leaf_tests_per_preview'] and work['witness_rows']<=8192 and work['peak_tracked_cells']<=union['max_tracked_cells'],rid+' per-preview caps')
   need(work['budget_refused'] is False,rid+' no cap refusal')
   if a['status']=='rejected':
    first_refusal=a;need(a['source_state_hash']==a['result_state_hash'],rid+' rejected state unchanged')
    need(a['removed_source_cell_mm3']==a['removed_physical_mm3']==a['removed_physical_target_mm3']==0 and 'frozen_reward' not in a and 'cut_union_committed' not in a,rid+' rejected no credit')
   else:
    need(a['status']=='accepted',rid+' status');accepted.append(a);proof=a['cut_union_committed']
    need(a['source_state_hash']!=a['result_state_hash'],rid+' committed ancestry')
    if prev_coverage is not None:need(proof['before_digest']==prev_coverage,rid+' ledger continuity')
    prev_coverage=proof['after_digest'];prev_state=a['result_state_hash']
    for k in ('new_leaf_count','leaf_tests'):need(proof[k]==work[k],rid+' attempted committed '+k)
    no_removal=a['removed_source_cell_mm3']==0
    need(proof['coverage_only_progress']==(no_removal and proof['new_leaf_count']>0),rid+' coverage-only flag')
    if no_removal:
     coverage_only+=1;need(a['reason']=='NATIVE_CUT_COVERAGE_PROGRESS' and a['nominal_target_source_credit_mm3']==0,rid+' coverage-only not removal')
    target=a['nominal_target_source_credit_mm3'];normal=a['removed_source_cell_mm3']-target
    length=2*math.dist(a['entry_mm'],a['tip_mm'])
    calculated=reward['target_per_mm3']*target-reward['normal_per_mm3']*normal-reward['action_cost']-reward['motion_per_mm']*length
    close(calculated,a['frozen_reward'],rid+' reward')
    need(0<=a['removed_physical_target_mm3']<=a['removed_physical_mm3']+1e-8<=a['removed_source_cell_mm3']+2e-8,rid+' physical volume nesting')
    if a['kind']=='side_preparation':close(a['removed_physical_target_mm3'],0,rid+' prep zero target')
  need(row['native_calls']==len(attempts),rid+' call count');need(native['accepted_actions']==len(accepted),rid+' accepted count')
  complete=len(accepted)==10
  need(native['complete_sequence']==native['feasible']==complete,rid+' complete truth')
  need(native['target_stroke_status']==acts[-1]['status'],rid+' target step status')
  for k in ('removed_source_cell_mm3','removed_physical_mm3','removed_physical_target_mm3'):close(sum(a[k] for a in accepted),native[k],rid+' volume sum '+k)
  close(sum(a['frozen_reward'] for a in accepted),native['frozen_reward_total'],rid+' reward sum')
  close(native['removed_cells']*volume,native['removed_source_cell_mm3'],rid+' voxel volume')
  close(native['removed_source_cell_mm3']-native['removed_physical_mm3'],native['removed_outer_voxelization_excess_mm3'],rid+' outer excess')
  close(acts[-1].get('removed_physical_target_mm3',0),native['target_stroke_removed_physical_target_mm3'],rid+' final target sum')
  need(native['partial_cells_kept_occupied'] is True,rid+' partial occupancy')
  need(0<=native['removed_physical_target_mm3']<=cap+1e-8,rid+' target cap bound')
  need(native['removed_physical_mm3']<=row['static_not_executable']['contact_physical_upper_mm3']+1e-8,rid+' static upper')
  for akey,rkey in [('leaf_tests','attempted_leaf_tests'),('new_leaf_count','attempted_new_leaf_count'),('witness_rows','attempted_witness_rows')]:
   total=sum(a['cut_union_attempt'][akey] for a in attempts);need(cov[rkey]==total,rid+' work sum '+akey);cost[akey]+=total
  need(cov['peak_tracked_cells']==max((a['cut_union_attempt']['peak_tracked_cells'] for a in attempts),default=0),rid+' max cells')
  need(cov['committed_new_leaf_count']==sum(a['cut_union_committed']['new_leaf_count'] for a in accepted),rid+' committed leaf sum')
  need(cov['coverage_only_accepted_actions']==sum(a['cut_union_committed']['coverage_only_progress'] for a in accepted),rid+' coverage progress count')
  need(cov['budget_refusals']==0 and cov['partially_certified_occupied_cells']<=cov['peak_tracked_cells'],rid+' final cap accounting')
  if prev_coverage is not None:need(cov['final_coverage_digest']==prev_coverage,rid+' final ledger')
  audit=row['independent_audit']
  if accepted:
   need(audit['status']=='completed',rid+' completed certificate');cert=audit['certificate']
   need(cert['feasible'] and cert['complete_tool_checked'] and cert['frontier_checked'] and not cert['failures'],rid+' certificate feasible')
   need(cert['source_case_hash']==row['source_hash'] and cert['action_count']==len(accepted) and cert['checker_version']=='independent-native-cut-union-v1',rid+' certificate identity/scope')
   close(cert['claimed_source_tissue_volume_mm3'],native['removed_source_cell_mm3'],rid+' certificate claim');close(cert['contained_source_tissue_volume_mm3'],native['removed_source_cell_mm3'],rid+' certificate volume')
   need(cert['unsupported_source_tissue_volume_mm3']==0,rid+' no unsupported volume');audit_seconds+=audit['seconds'];counts['certificates']+=1
  else:need(audit['status']=='not_applicable_rejected_no_committed_history' and 'certificate' not in audit,rid+' empty no certificate');counts['empty_prefix']+=1
  oldcomplete=baseline['native']['complete_sequence'];delta=native['removed_physical_target_mm3']-baseline['native']['removed_physical_target_mm3']
  if complete and not oldcomplete:gained.append(rid)
  if oldcomplete and not complete:regressed.append(rid)
  if native['removed_physical_target_mm3']>0 and baseline['native']['removed_physical_target_mm3']==0:positive_gain.append(rid)
  m=matched[rid];need(m['baseline']==baseline['native'] and m['experimental']==native,rid+' matched native records')
  close(m['physical_target_difference_mm3'],delta,rid+' target delta')
  close(m['physical_removed_difference_mm3'],native['removed_physical_mm3']-baseline['native']['removed_physical_mm3'],rid+' volume delta')
  close(m['experimental_target_shortfall_mm3'],cap-native['removed_physical_target_mm3'],rid+' target shortfall')
  row_seconds+=row['timing_seconds']['numerical_row_total']
  rows.append({'row_id':rid,'complete_before':oldcomplete,'complete_after':complete,'accepted':len(accepted),'coverage_only':cov['coverage_only_accepted_actions'],'target_before_mm3':baseline['native']['removed_physical_target_mm3'],'target_after_mm3':native['removed_physical_target_mm3'],'target_delta_mm3':delta,'exact_cap_mm3':cap,'source_cell_volume_mm3':native['removed_source_cell_mm3'],'physical_removed_mm3':native['removed_physical_mm3'],'return':native['frozen_reward_total'],'first_refusal':None if first_refusal is None else {k:first_refusal[k] for k in ('index','reason','failure_tip_mm')},'audit_actions':len(accepted)})
 calls=counts['accepted']+counts['rejected'];need(calls==r['native_call_costs_including_incomplete_row']['native_strokes_started']==r['native_call_costs_including_incomplete_row']['native_strokes_returned']==146,'total calls')
 need(counts['accepted']+counts['rejected']+counts['unexecuted_prior_refusal']==180,'all slots')
 close(native_seconds,r['native_call_costs_including_incomplete_row']['native_seconds'],'native seconds sum')
 work=r['returned_preview_coverage_work_including_incomplete_row']
 for k,v in cost.items():need(work[k]==v,'global work '+k)
 need(work['returned_previews_accounted']==calls and work['peak_tracked_cells']==max_cells and work['budget_refusals']==0,'global coverage totals')
 need(launch['last_worker_progress']==progress and progress['native_call_in_progress'] is False,'final progress joined')
 need(progress['returned_preview_coverage_work_including_incomplete_row']==work and progress['native_call_costs_including_incomplete_row']==r['native_call_costs_including_incomplete_row'],'progress cost joins')
 complete=sum(x['complete_after'] for x in rows);need(complete==r['accepted_complete_sequences']==r['verified_complete_sequences']==13,'13 verified complete')
 useful=(len(rows)==18 and len(gained)>=d['predeclared_useful_criterion']['additional_complete_sequences_minimum'] and not regressed and counts['certificates']==15 and not r['coverage_budget_refused_rows'])
 need(useful and r['useful_criterion']['meets_predeclared_useful_criterion'],'independent useful predicate')
 need(sorted(gained)==r['useful_criterion']['additional_complete_rows'] and regressed==r['useful_criterion']['prior_complete_rows_now_incomplete_or_missing'],'criterion members')
 for k in ('all_18_rows_present','all_nonempty_prefix_audits_accepted','all_physical_target_volumes_within_exact_cap'):need(r['useful_criterion'][k] is True,'criterion '+k)
 need(native_seconds<=row_seconds<=r['worker_seconds'] and row_seconds+audit_seconds<=r['worker_seconds']+1e-6,'nested timing')
 need(r['worker_seconds']<d['caps']['cooperative_worker_seconds'] and launch['parent_wall_seconds']<d['caps']['parent_wall_seconds'],'time caps')
 need(r['peak_worker_rss_bytes']<d['caps']['worker_peak_rss_bytes'] and calls<=180 and counts['certificates']<=18,'resource ceilings')
 ratio=r['worker_seconds']/old['worker_seconds'];close(ratio,r['resource_comparison']['worker_runtime_ratio_to_retained'],'descriptive ratio')
 grouping={str(h):{'complete_before':sum(x['complete_before'] for x in rows if x['row_id'].endswith('-'+format(h,'g')+'mm')),'complete_after':sum(x['complete_after'] for x in rows if x['row_id'].endswith('-'+format(h,'g')+'mm')),'target_positive_after':sum(x['target_after_mm3']>0 for x in rows if x['row_id'].endswith('-'+format(h,'g')+'mm'))} for h in d['spacings_mm']}
 return {'status':'PASS','checks':checks,'scope':'Saved JSON/hash/scalar audit only; no project imports, patient arrays, native rerun or reconstructed leaf proof. Certificates authenticate reported nonempty prefixes; no certification of rejected actions.','inputs':inputs,'counts':dict(counts),'coverage_only_accepted_actions':coverage_only,'complete_before':sum(x['complete_before'] for x in rows),'complete_after':complete,'grouping':grouping,'additional_complete_rows':gained,'new_target_positive_rows':positive_gain,'regressions':regressed,'meets_predeclared_useful_criterion':useful,'coverage_work':dict(work),'maximum_per_preview':{'leaf_tests':max_leaf,'witness_rows':max_witness,'tracked_cells':max_cells},'costs':{'worker_seconds':r['worker_seconds'],'retained_worker_seconds':old['worker_seconds'],'nonpaired_worker_ratio':ratio,'parent_seconds':launch['parent_wall_seconds'],'native_seconds':native_seconds,'numerical_rows_seconds':row_seconds,'independent_audit_seconds':audit_seconds,'worker_peak_rss_bytes':r['peak_worker_rss_bytes'],'parent_polled_peak_worker_rss_bytes':launch['peak_polled_worker_rss_bytes'],'scope':'Native time is nested in numerical rows; rows plus audits nested in worker; parent encompasses worker. Nonpaired historical costs, no causal speed estimate.'},'rows':rows,'limitations':['Completion criterion concerns fixed generated complementary-cut sequences, not optimality or positive return.','Full source-cell removal and clipped physical volume are separate; initial occupied outer cells and partial-cell conservative occupancy remain.','Finite-grid target shortfalls and phase/rotation dependence persist; no convergence or patient-resolution claim.','No force/material/device validation, human patients, training updates, selected policy or RL improvement.','Saved report retains history hashes and full-tool certificate summaries, not raw per-leaf history; this metadata audit does not re-establish geometry independently.']}

if __name__=='__main__':
 try:result=main()
 except Exception as e:result={'status':'FAIL','checks':checks,'error':str(e),'traceback':traceback.format_exc(),'inputs':inputs}
 path=OUT/'audit.json'
 with path.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({k:result[k] for k in ('status','checks')}))
 if result['status']!='PASS':print(result['traceback']);raise SystemExit(1)
