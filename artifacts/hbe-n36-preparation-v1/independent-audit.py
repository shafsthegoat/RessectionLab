"""Saved-preparation audit only. Never meshes, solves or opens measured curves."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,tarfile,time,xml.etree.ElementTree as ET
import numpy as np
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent
P=R/'outputs/mechanics/hbe-01-03-halfheight-global-n36-v1/preparation'
A=R/'build/hbe-halfheight-global-n36-v1/source'
sys.path.insert(0,str(A))
from scripts import mechanics_hbe_halfheight_global_n36 as c
from scripts import mechanics_hbe_halfheight_global_n36_experiment as x
assert Path(c.__file__).resolve()==A/'scripts/mechanics_hbe_halfheight_global_n36.py'
def forbidden(*a,**kw):raise AssertionError('Native execution/meshing forbidden in preparation review')
x.subprocess.Popen=forbidden;x.runtime.supervise=forbidden;x.old.solve=forbidden;c.generate_full_mesh=forbidden
started=time.perf_counter();checked={}
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while chunk:=f.read(1024*1024):h.update(chunk)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_bytes())
def bind(b):
 p=(R/b['path']).resolve();h=digest(p);assert h==b['sha256'],str(p)
 checked[str(p)]={'sha256':h,'bytes':p.stat().st_size};return p
def binding(p):return {'path':str(Path(p).relative_to(R)),'sha256':digest(p)}
def tree(root):
 return {str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(root.rglob('*')) if p.is_file()}
before=tree(P)
assert digest(P/'result.json')=='a83b1278a21642a9961ca590d8207891c37fa6f8a84ec50744e8541e2b0b4930'
release_p=R/'build/hbe-halfheight-global-n36-v1/prepare-release.json'
assert digest(release_p)=='09f0f974f1e9a3d7e4765ac96c736eb72cc0c9ca250ba96336799e2c01689196'
release=load(release_p);result=load(P/'result.json');publication=load(P/'publication-check.json')
state=load(bind(result['state']));base=load(bind(result['baseline']));study=load(bind(release['study']))
assert release['authorized'] is True and release['phase']=='prepare' and release['source_commit']=='169efad4d2cb3b53a80091082c9845a2c45e8b99'
assert base['release_binding']==binding(release_p)
assert base['source_bindings']==release['source_bindings'] and base['study_binding']==release['study']
assert base['phase']=='prepare' and base['directory']==str(P) and base['output_root']==str(P.parent)
c.require_study(study);x.source_inventory(R,release,study)
for row in release['source_bindings'].values():bind(row)
bind(release['source_archive']);bind(release['interpreter']);bind(release['independent_implementation_review'])
for path,h in base['inputs'].items():bind({'path':path,'sha256':h})
assert result['schema']=='hbe-halfheight-global-n36-supervised-result-v1'
assert result['status']=='prepared_not_solved' and result['phase']=='prepare' and result['measured_data_accessed'] is False
x.require_publication(publication,result_binding=binding(P/'result.json'),phase='prepare',cap=60,output_cap=2*1024**3)
assert publication['result_status']==result['status'] and result['aggregate_cap_seconds']==60
sup=result['supervision'];assert load(P/'supervision/supervision.json')==sup
assert sup['status']=='completed' and type(sup['exit_code']) is int and sup['exit_code']==0
assert sup['kill_reason'] is sup['cleanup_error'] is sup['error'] is None and sup['no_retry'] is True
assert 0<=sup['elapsed_seconds']<sup['wall_cap_seconds']<=60
assert sup['rss_cap_bytes']==3*1024**3 and 0<sup['sampled_peak_process_group_rss_bytes']<=sup['rss_cap_bytes']
assert 0<=result['phase_elapsed_before_result_publication_seconds']<=publication['elapsed_through_result_publication_seconds']<60
assert sup['command'][1]==str(A/'scripts/mechanics_hbe_halfheight_global_n36_experiment.py')
assert sup['command'][5]==result['baseline']['path'] and sup['command'][7]==result['baseline']['sha256']
assert Path(sup['command'][0]).resolve()==Path(release['interpreter']['path']).resolve()
assert state['status']=='prepared_not_solved' and state['phase']=='prepare'
assert state['gmsh_generation_calls']==1 and state['solver_invocations']==0 and state['measured_data_accessed'] is False
assert state['runs']=={c.RUN_ID:{'status':'not_executed'}} and not state['reused_runs']
assert set(state['levels'])=={'N36'} and set(state['cases'])=={c.RUN_ID}
case=state['cases'][c.RUN_ID]
for row in case.values():bind(row)
receipt=load(bind(state['generation_receipt']))
assert receipt['status']=='prepared_not_solved' and receipt['gmsh_generation_calls']==1 and receipt['solver_calls']==0
assert receipt['runtime_unchanged'] is True and receipt['measured_data_accessed'] is False and 'finalize_error' not in receipt
assert receipt['source_sha256']==release['source_bindings']['halfheight_global_n36']['sha256']
assert receipt['generator_source_sha256']==study['inherited_source_sha256']['mesh_deck']
assert receipt['declaration_sha256']==release['study']['sha256']
assert receipt['full_mesh_sha256']==case['full_mesh']['sha256']
bind({'path':str((P/'N36/generated/specimen.msh').relative_to(R)),'sha256':receipt['native_mesh_sha256']})
# Re-read and independently reconstruct the already saved mesh/deck. This does
# not call a mesher or create/write a new mesh.
verified=c.verify_prepared_case(R,release['study'],case,case['reconstruction'])
_,wrapper,half,full=verified
assert wrapper['full_quality']==receipt['quality']
x.backend.verify_deck(bind(case['backend_source_deck']).read_bytes(),bind(case['deck']).read_bytes())
X=np.asarray(full['rest_nodes_m']); C=np.asarray(full['elements_hex8'])
Y=np.asarray(half['rest_nodes_m']); D=np.asarray(half['elements_hex8'])
assert X.shape==(75259,3) and C.shape==(69984,8) and Y.shape==(39610,3) and D.shape==(34992,8)
assert np.isfinite(X).all() and np.isfinite(Y).all()
M=wrapper['mapping'];nids=np.asarray(M['half_to_full_node_ids'])-1;eids=np.asarray(M['half_to_full_element_ids'])-1
assert np.array_equal(Y,X[nids]);assert np.array_equal(nids[D-1]+1,C[eids])
H=study['geometry']['height_m'];tol=study['geometry']['coordinate_tolerance_m']
assert len(np.unique(X[:,2]))==19 and len(np.unique(Y[:,2]))==10
assert np.count_nonzero(np.abs(X[:,2]-H/2)<=tol)==3961
node_coverage=set();cell_coverage=set();maxerr=0.
for row in M['reflections']:
 ni=np.asarray(row['node_full_ids'])-1;ei=np.asarray(row['element_full_ids'])-1;perm=np.asarray(row['element_half_to_full_local'])
 err=float(np.max(np.abs(X[ni]-(Y*np.asarray(row['signs'])+np.asarray(row['rest_offset_m'])))))
 assert err<=tol;maxerr=max(maxerr,err)
 assert np.array_equal(np.take_along_axis(C[ei],perm,axis=1),ni[D-1]+1)
 node_coverage.update(ni.tolist());cell_coverage.update(ei.tolist())
assert node_coverage==set(range(75259)) and cell_coverage==set(range(69984))
assert wrapper['verification']['straddling_cells']==0
# Independent XML count/material/constraints check in addition to exact full-deck regeneration.
xml=ET.parse(bind(case['deck'])).getroot()
assert len(xml.findall('./Mesh/Nodes/node'))==39610 and len(xml.findall('./Mesh/Elements/elem'))==34992
assert xml.find('./Mesh/Elements').attrib['type']=='hex8'
mat=xml.find('./Material/material');assert mat.attrib['type']=='Ogden'
assert float(mat.findtext('c1'))==2000 and float(mat.findtext('m1'))==2 and float(mat.findtext('k'))==149000/3
assert all(float(mat.findtext('c'+str(i)))==0 for i in range(2,7))
domain=xml.find('./MeshDomains/SolidDomain');assert domain.attrib['type']=='three-field-solid' and domain.findtext('laugon')=='0'
assert xml.findtext('./Control/time_steps')=='60' and float(xml.findtext('./Control/step_size'))==1/60
assert xml.findtext('./Control/time_stepper/max_retries')=='0'
assert xml.findtext('./Control/time_stepper/dtmin')==xml.findtext('./Control/time_stepper/dtmax')
bcs=xml.findall('./Boundary/bc');bottom=[b for b in bcs if b.attrib['node_set']=='bottom'];top=[b for b in bcs if b.attrib['node_set'].startswith('top_node_')]
assert len(bottom)==3 and {b.findtext('dof') for b in bottom}=={'x','y','z'}
assert len(top)==3961 and len(bcs)==3964
assert all(b.findtext('dof')=='z' and float(b.findtext('value'))==.5 for b in top)
loading=load(bind(case['loading']))
assert loading['prescribed_dofs']=={'bottom':'xyz','midplane':'z'} and loading['artificial_midplane_tangential_dofs']=='free'
assert loading['mu_Pa']==1000 and loading['full_energy_scale']==2 and loading['full_reaction_scale']==1
assert len(loading['times'])==len(loading['load_coordinate'])==61
watch=load(P/'output-watch.json')
assert watch['status']=='watching' and watch.get('reason') is None
assert watch['maximum_active_bytes']<=768*1024**2 and watch['maximum_total_bytes']<=2*1024**3
assert not list(P.rglob('nodes.log')) and not list(P.rglob('elements.log')) and not list(P.rglob('solver.log'))
assert not (P.parent/'experiment').exists()
whole=tree(P.parent)
assert sum(v['bytes'] for v in whole.values())==publication['retained_bytes_including_closeout']
assert publication['retained_bytes_before_closeout']+len((P/'publication-check.json').read_bytes())==publication['retained_bytes_including_closeout']
assert tree(P)==before
for p,row in checked.items():assert digest(p)==row['sha256']
(O/'input-hashes.json').write_text(json.dumps(checked,indent=2,sort_keys=True)+'\n')
report={'schema':'hbe-n36-independent-saved-preparation-review-v1','created_at':datetime.now(timezone.utc).isoformat(),
 'status':'actual_preparation_verified_for_separate_single_solve_release_review','blocking_findings':[],
 'result':binding(P/'result.json'),'publication':binding(P/'publication-check.json'),'preparation_release':binding(release_p),
 'source_commit':release['source_commit'],'source_archive':release['source_archive'],'source_modules':20,'declarations':6,
 'source_and_baseline_input_hashes_unchanged':True,'baseline_input_count':len(base['inputs']),
 'full_prepared_geometry_deck_loading_reconstruction_recheck':True,
 'independent_geometry':{'full_nodes':len(X),'full_hex8_cells':len(C),'half_nodes':len(Y),'half_hex8_cells':len(D),
 'full_axial_layers':18,'half_axial_layers':9,'midplane_nodes':3961,'straddling_cells':0,
 'half_coordinates_bit_exact_subset':True,'reflection_node_and_cell_coverage_exact':True,'maximum_reflection_coordinate_error_m':maxerr,
 'retained_quality':receipt['quality']},
 'native_call_counts':{'observed_preparation_gmsh_generation_calls':1,'observed_preparation_solver_calls':0,'review_gmsh_calls':0,'review_solver_calls':0},
 'resources':{'publication_elapsed_seconds':publication['elapsed_through_result_publication_seconds'],
 'inclusive_prepare_cap_seconds':60,'native_mesh_generation_seconds':receipt['elapsed_seconds'],
 'sampled_peak_worker_family_rss_B':sup['sampled_peak_process_group_rss_bytes'],'sampled_rss_cap_B':3*1024**3,
 'retained_bytes_including_closeout':publication['retained_bytes_including_closeout'],'output_watch':watch,
 'supervisor_completed_exit0_without_kill_cleanup_or_error':True},
 'limits':['Review verifies saved preparation only; no spatial, temporal, physical or calibration gate passed.',
 'No new mesh or solver was executed. Existing mesh arrays and deck were inspected and rechecked without writing derived geometry.',
 'Original input and native primitive files were hash-checked only; no raw response parsing or measured curve access.',
 'RSS/output maxima are sampled; cleanup evidence is the exact successful supervisor record, not a retrospective process census.',
 'Solve remains unreleased here and must bind both result and publication plus this review. One solver call only under unchanged prospective caps.',
 'The closing receipt explicitly excludes its own final fsync from the observed publication interval.'],
 'review_elapsed_seconds':time.perf_counter()-started,'review_runtime_is_not_native_performance':True,
 'input_hash_inventory':binding(O/'input-hashes.json'),'audit_source':binding(Path(__file__)),
 'tracked_edits':False,'release_created':False,'measured_curves_accessed':False,'prepared_files_unchanged':True}
(O/'verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':report['status'],'verification':binding(O/'verification.json'),'review_seconds':report['review_elapsed_seconds'],'checked_inputs':len(checked)},indent=2))
