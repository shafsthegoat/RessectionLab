"""Independent saved-data scalar/source audit. No native, NumPy or project imports."""
import ast, hashlib, json, math
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent;ROOT=STAGE.parents[1]
checks=[]
def need(ok,label):
 if not ok: raise AssertionError(label)
 checks.append(label)
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p,s): need(sha(p)==s,'hash '+str(p.relative_to(ROOT)))
def digest(x): return 'sha256:'+hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def close(x,y):return math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-8)
def det(a):return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])
def assigned(tree,name):return next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
rp=STAGE/'run-01/report.json';lp=STAGE/'run-01/launcher.json'
pin(rp,'8e40ef8f6c4504c8eedb020687250c59cb4e58f869fe02bf530a9a001761bc9f');pin(lp,'9df34b8d3ae5ddc4dda95d63bff83373427eb9e69c194c7ba7b7e235ee072628')
r=read(rp);launcher=read(lp);d=read(STAGE/'declaration.json');prospective=read(STAGE/'run-01/prospective-source.json')
release=read(STAGE/'root-release.json');dispatch=read(STAGE/'dispatch.json');control=read(STAGE/'controls-01.json');ready=read(STAGE/'source-ready.json')
pin(STAGE/'root-release.json',dispatch['root_release_sha256']);need(release['head']==dispatch['head']==control['head'],'dispatch release test HEAD join')
for f,h in release['files'].items():pin(STAGE/f,h)
for f,h in ready['source_files'].items():pin(STAGE/f,h)
pin(ROOT/'scripts/probe_native_discretization.py',ready['base_adapter_sha256'])
need(control['source_before']==control['source_after'] and control['source_unchanged'] and control['returncode']==0 and control['child_reaped'] and not control['timed_out'],'seven generated controls terminal unchanged')
pin(STAGE/'controls-01.log',control['log_sha256']);need('7 passed' in (STAGE/'controls-01.log').read_text(),'saved test denominator')
for f,h in control['source_before'].items():pin(ROOT/f,h)
need(prospective['declaration']==d and prospective['source']==r['source'] and r['declaration_hash']==digest(d),'prospective report declaration/source joins')
for f,h in r['source']['files'].items():
 need(Path(f).suffix in ('.py','.json'),'source-only receipt extension '+f);pin(ROOT/f,h)
# Independent AST read of actual canonical parameter definitions, without import.
tree=ast.parse((ROOT/'src/resectionlab/geometry.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ToolGeometry')
fields=[n.target.id for n in cls.body if isinstance(n,ast.AnnAssign)]
defaults={n.target.id:ast.literal_eval(n.value) for n in cls.body if isinstance(n,ast.AnnAssign) and n.value is not None}
tools=[]
for node in assigned(tree,'GENERIC_TOOLS').elts:
 t=defaults.copy();t.update(zip(fields,[ast.literal_eval(x) for x in node.args]));t.update({k.arg:ast.literal_eval(k.value) for k in node.keywords});tools.append(t)
need(tools==d['tools'],'canonical generic tool identity via AST')
reward_node=assigned(ast.parse((ROOT/'src/resectionlab/native_spatial_task.py').read_text()),'DEFAULT_NATIVE_SPATIAL_REWARD')
weights={k.arg:ast.literal_eval(k.value) for k in reward_node.keywords};need(weights==d['reward'],'canonical reward identity via AST')
need(r['status']==launcher['status']=='completed' and launcher['worker_exit_code']==0 and launcher['parent_stop_reason'] is None and r['source_unchanged'] and launcher['source_unchanged'],'owned terminal completion')
need(r['human_patients']==r['training_updates']==0 and r['numerical_threads']==1,'no patient or learning work')
need(launcher['numerical_rows_complete'] and launcher['independent_audits_complete'],'launcher completion accounting')
need(r['worker_seconds']<=d['caps']['cooperative_worker_seconds'] and launcher['parent_wall_seconds']<=d['caps']['parent_wall_seconds'] and max(r['peak_worker_rss_bytes'],launcher['peak_polled_worker_rss_bytes'])<=d['caps']['worker_peak_rss_bytes'],'declared resource caps')
expected=[f"{g['name']}-{h:g}mm" for h in d['spacings_mm'] for g in d['grids']]
need(r['planned_rows']==expected and [x['row_id'] for x in r['rows']]==expected and not r['missing_rows'],'exact 18 rows in declared order')
need(r['completed_rows']==r['expected_rows']==d['expected_rows']==len(expected)==18 and r['planned_action_slots']==d['expected_action_slots']==180,'full row and action-slot denominator')
need(d['max_tip_step_mm']==.0625 and d['entry_mm']==[0,0,-2] and d['access_radius_mm']==6 and d['anatomy']['nominal_target']=='half_space_z_greater_than_or_equal_to_2.8_mm; exact cell fraction','fixed scene/access/microstep')
summary=[];status_counts=Counter();failures=Counter();by_spacing={str(h):{'complete':0,'positive_target':0,'denominator':6} for h in d['spacings_mm']};timing_native=timing_num=timing_audit=0.;certs=empty=0
sequences={}
for row in r['rows']:
 label=row['row_id'];tool=next(t for t in tools if t['tool_id']==row['grid']['tool_id']);h=row['spacing_mm'];a=row['affine'];voxel=abs(det(a))
 need(row['grid'] in d['grids'] and row['fixed_tool']==tool and close(voxel,h**3),'declared tool/grid '+label)
 need(math.prod(row['shape'])<=d['caps']['max_grid_cells'],'grid-cell cap '+label)
 angle=math.radians(row['grid']['rotation_xyz_degrees'][1]);rotation=[[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]]
 need(all(close(a[i][j],h*rotation[i][j]) for i in range(3) for j in range(3)),'relative grid orientation '+label)
 # Affine origin in source-grid units has the prescribed phase modulo integers.
 origin_local=[sum(rotation[k][i]*a[k][3]/h for k in range(3)) for i in range(3)]
 need(all(close(x-phase,round(x-phase)) for x,phase in zip(origin_local,row['grid']['phase_source_voxels'])),'relative grid phase '+label)
 lattice=d['preparation_lattice_mm'][tool['tool_id']]
 actions=[{'index':i+1,'kind':'side_preparation','tool_id':tool['tool_id'],'entry_mm':[lattice*x,lattice*y,d['entry_mm'][2]],'tip_mm':[lattice*x,lattice*y,d['preparation_tip_z_mm']]} for i,(x,y) in enumerate(d['preparation_order_lattice_units'])]
 actions.append({'index':10,'kind':'target_stroke','tool_id':tool['tool_id'],'entry_mm':list(d['entry_mm']),'tip_mm':[0.,0.,d['target_tip_z_mm']]})
 need(row['physical_actions_sha256']==digest(actions) and len(row['actions'])==row['planned_actions']==10,'fixed physical sequence hash '+label)
 sequences.setdefault(tool['tool_id'],actions);need(sequences[tool['tool_id']]==actions,'within-tool all-grid action invariance '+label)
 radius=tool['tip_radius_mm'];shaft=tool['shaft_radius_mm'];length=tool['tip_length_mm']
 rationale={'preparation_shaft_highest_z_mm':1-length+shaft,'target_shaft_highest_z_mm':2.2-length+shaft,'disk_cover_margin_mm':radius-math.sqrt(2)*max(lattice/2,shaft-lattice),'aperture_margin_mm':6-(math.sqrt(2)*lattice+shaft),'preparation_to_target_plane_gap_mm':2.8-(1+radius),'target_stroke_cap_penetration_mm':2.2+radius-2.8}
 need(all(close(v,row['continuum_rationale'][k]) for k,v in rationale.items()),'continuum enclosure scalar arithmetic '+label)
 need(rationale['preparation_shaft_highest_z_mm']<0<rationale['target_shaft_highest_z_mm']<1 and all(v>0 for k,v in rationale.items() if k not in ('preparation_shaft_highest_z_mm','target_shaft_highest_z_mm')),'positive continuous margins '+label)
 accepted=[];rejected=[];seen_refusal=False;previous=None
 for expected_action,act in zip(actions,row['actions']):
  need(all(act[k]==v for k,v in expected_action.items()),'exact action geometry '+label+'/'+str(act['index']))
  status_counts[act['status']]+=1
  if seen_refusal:need(act['status']=='unexecuted_prior_refusal','preserved unexecuted suffix '+label);continue
  need(act['status'] in ('accepted','rejected'),'attempted prefix status '+label)
  if previous is not None:need(act['source_state_hash']==previous,'native state chain '+label+'/'+str(act['index']))
  previous=act['result_state_hash'];timing_native+=act['native_seconds']
  need(act['removed_source_cell_mm3']+1e-8>=act['removed_physical_mm3']>=act['removed_physical_target_mm3']-1e-8>=-1e-8,'physical/source volume separation '+label+'/'+str(act['index']))
  if act['status']=='rejected':
   seen_refusal=True;rejected.append(act);failures[act['reason']]+=1
   need(act['source_state_hash']==act['result_state_hash'] and act['removed_source_cell_mm3']==act['removed_physical_mm3']==act['removed_physical_target_mm3']==0,'refusal no state/removal '+label)
   continue
  accepted.append(act);need(act['failure_tip_mm'] is None and act['reason']=='NATIVE_CONNECTED_STROKE','accepted action status '+label)
  need('normal_tissue_exposure_unassessed' in act['geometry_unknowns'] and 'tool_geometry_outside_image_unassessed' in act['geometry_unknowns'],'uncertainty retained '+label)
  target=act['nominal_target_source_credit_mm3'];total=act['removed_source_cell_mm3'];distance=math.dist(act['entry_mm'],act['tip_mm'])
  reward=weights['target_per_mm3']*target-weights['normal_per_mm3']*(total-target)-weights['action_cost']-2*weights['motion_per_mm']*distance
  need(close(reward,act['frozen_reward']),'independent frozen reward arithmetic '+label+'/'+str(act['index']))
  need(abs(target-act['removed_physical_target_mm3'])<=max(1e-7,total*6e-8),'float32 target credit versus clipped target '+label)
  if act['index']<10:need(act['removed_physical_target_mm3']<=1e-10,'preparation cannot reach fixed target plane '+label)
 n=row['native'];static=row['static_not_executable'];certificate=row['independent_audit']
 need(row['native_calls']==len(accepted)+len(rejected) and n['accepted_actions']==len(accepted),'calls and accepted count '+label)
 need(n['complete_sequence']==n['feasible']==(len(accepted)==10) and n['target_stroke_status']==row['actions'][-1]['status'],'sequence and final target status '+label)
 for key,action_key in [('removed_source_cell_mm3','removed_source_cell_mm3'),('removed_physical_mm3','removed_physical_mm3'),('removed_physical_target_mm3','removed_physical_target_mm3'),('frozen_reward_total','frozen_reward')]:
  need(close(n[key],sum(x[action_key] for x in accepted)),'sum '+key+' '+label)
 need(close(n['removed_source_cell_mm3'],n['removed_cells']*voxel) and close(n['removed_outer_voxelization_excess_mm3'],n['removed_source_cell_mm3']-n['removed_physical_mm3']),'cell/excess volume arithmetic '+label)
 need(n['removed_physical_mm3']<=static['contained_physical_lower_mm3']+1e-8<=static['contact_physical_upper_mm3']+1e-8 and n['removed_physical_mm3']<=n['contacted_physical_upper_mm3']+1e-8<=static['contact_physical_upper_mm3']+1e-8,'non-executable static union enclosures '+label)
 need(n['partial_cells_kept_occupied'] and close(n['target_stroke_removed_physical_target_mm3'],row['actions'][-1].get('removed_physical_target_mm3',0.)),'target stroke and contact semantics '+label)
 if accepted:
  certs+=1;c=certificate['certificate'];timing_audit+=certificate['seconds']
  need(certificate['status']=='completed' and c['feasible'] and not c['failures'] and c['first_failed_action'] is None and c['complete_tool_checked'] and c['frontier_checked'],'independent accepted-prefix certificate '+label)
  need(c['source_case_hash']==row['source_hash'] and c['action_count']==len(accepted) and close(c['claimed_source_tissue_volume_mm3'],n['removed_source_cell_mm3']) and close(c['contained_source_tissue_volume_mm3'],n['removed_source_cell_mm3']) and c['unsupported_source_tissue_volume_mm3']==0 and close(c['source_voxel_volume_mm3'],voxel),'certificate source count volume joins '+label)
 else:
  empty+=1;need(certificate['status']=='not_applicable_rejected_no_committed_history' and 'certificate' not in certificate,'empty history has no certificate '+label)
 timing_num+=row['timing_seconds']['numerical_row_total'];by_spacing[str(h)]['complete']+=int(n['complete_sequence']);by_spacing[str(h)]['positive_target']+=int(n['removed_physical_target_mm3']>0)
 summary.append({'row_id':label,'complete':n['complete_sequence'],'accepted_actions':len(accepted),'native_calls':row['native_calls'],'first_refusal':None if not rejected else {k:rejected[0][k] for k in ('index','reason','failure_tip_mm')},'removed_cells':n['removed_cells'],'source_credit_mm3':n['removed_source_cell_mm3'],'physical_removal_mm3':n['removed_physical_mm3'],'physical_target_mm3':n['removed_physical_target_mm3'],'outer_voxel_excess_mm3':n['removed_outer_voxelization_excess_mm3'],'frozen_reward':n['frozen_reward_total'],'certificate_actions':len(accepted) if accepted else None})
need(status_counts==Counter(accepted=130,rejected=7,unexecuted_prior_refusal=43),'full 180-slot statuses')
need(sum(x['native_calls'] for x in r['rows'])==r['native_call_costs_including_incomplete_row']['native_strokes_started']==r['native_call_costs_including_incomplete_row']['native_strokes_returned']==137,'all native calls')
need(certs==14 and empty==4 and r['accepted_complete_sequences']==r['verified_complete_sequences']==11,'full and prefix certificate denominator')
need([by_spacing[str(h)]['complete'] for h in d['spacings_mm']]==[2,3,6],'acceptance by spacing')
need(timing_num+timing_audit<=r['worker_seconds']<=launcher['parent_wall_seconds'] and r['native_call_costs_including_incomplete_row']['native_seconds']<=timing_native<=timing_num,'nested cost scope')
finest={}
for tool in tools:
 selected=[x for x in r['rows'] if x['spacing_mm']==.25 and x['fixed_tool']==tool]
 finest[tool['tool_id']]={key:[min(x['native'][key] for x in selected),max(x['native'][key] for x in selected)] for key in ('removed_physical_mm3','removed_source_cell_mm3','removed_physical_target_mm3','frozen_reward_total')}
out={'status':'PASS','material_blockers':[],'scope':'Saved JSON, AST constants, source hashes and scalar arithmetic only; no native replay, original data or timed tests. Geometry and clipped-volume cell membership not independently reconstructed.','check_count':len(checks),'checks_hash':digest(checks),'report_sha256':sha(rp),'launcher_sha256':sha(lp),'source_files_verified':len(r['source']['files']),'rows':summary,'by_spacing':by_spacing,'slot_counts':dict(status_counts),'first_refusal_counts':dict(failures),'certificates_nonempty_prefixes':certs,'empty_histories_without_certificate':empty,'finest_spacing_ranges':finest,'costs':{'numerical_rows_seconds':timing_num,'independent_prefix_audits_seconds':timing_audit,'native_execution_included_seconds':r['native_call_costs_including_incomplete_row']['native_seconds'],'worker_seconds':r['worker_seconds'],'parent_seconds':launcher['parent_wall_seconds'],'worker_peak_rss_bytes':r['peak_worker_rss_bytes'],'parent_polled_rss_bytes':launcher['peak_polled_worker_rss_bytes']},'limitations':['11 complete sequences do not mean11 positive-target outcomes:9 are target-positive.','Native acceptance is not individually monotone across spacing.6/6 at0.25mm does not prove convergence.','Independent checker validates committed source-cell prefixes, not refusals, unexecuted suffixes, analytic physical-volume integration or real mechanics.','No patient anatomy, device calibration, forces, deformation or surgical efficacy evidence.','Source-ready/README unrun status is historical prospective metadata, superseded by exact7-pass receipt and completed numerical run.','Launcher retains/waits one child and samples worker RSS; receipt is not a process-tree survivor census.']} 
(HERE/'audit.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':out['status'],'checks':len(checks),'by_spacing':by_spacing,'slots':dict(status_counts),'certificates':certs}))
