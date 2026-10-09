"""Finish one independently replayed N36 S120 audit and frozen temporal comparison."""
import ast, hashlib, json, math, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit import ROOT,OUT,REL,REC,ATT,HEAD,REL_SHA,REC_SHA,read,sha,save,sources,bind
started=time.monotonic()
rel,rec,r=read(REL),read(REC),read(ATT/'readout.json')
source=read(OUT/'source-audit.json'); supervision=read(OUT/'receipt.json'); replay=read(OUT/'replay-comparison.json')
assert replay['entire_json_equal'] and replay['frame_count']==121
assert supervision['readout_stage']['exit_code']==0 and supervision['readout_stage']['status']=='completed_within_caps'
assert supervision['readout_stage']['kill_reason'] is None
assert sha(ATT/'readout.json')==sha(OUT/'replayed-readout.json')
assert r['numerical_passed'] is True and r['frame_count']==121 and r['steps']==120
assert r['representation']=='reconstructed_full'
assert r['provenance']['reconstruction_provenance']=='reflected_native_half_not_native_full'
assert r['load_coordinate_full_m']==rel['adapter_receipt']['full_load_coordinates_m']
assert r['minimum_sampled_J']>0 and r['minimum_logged_J']>0
assert all(math.isfinite(v) and 0<=v<=1 for v in r['criteria_max_ratio'].values())
mesh=read(ROOT/rel['native_mesh']['path'])
assert len(mesh['rest_nodes_m'])==39610 and len(mesh['elements_hex8'])==34992
streams={}
for name,count in [('nodes.log',39610),('elements.log',34992)]:
 frames=[];frame=None;min_j=math.inf
 with (ATT/name).open() as f:
  for line in f:
   if line.startswith('*Step'):
    if frame is not None:assert frame['rows']==count;frames.append(frame)
    frame={'step':int(line.split('=')[1]),'rows':0}
   elif line.startswith('*Time'):frame['time']=float(line.split('=')[1])
   elif line.startswith('*Data'):frame['data']=line.split('=')[1].strip()
   elif line.strip():
    first,rest=line.split(',',1);frame['rows']+=1;assert int(first)==frame['rows']
    if name=='elements.log':min_j=min(min_j,float(rest.rsplit(',',2)[1]))
 if frame is not None:assert frame['rows']==count;frames.append(frame)
 assert len(frames)==121 and [x['step'] for x in frames]==list(range(121))
 timing=max(abs(x['time']-i/120) for i,x in enumerate(frames));assert timing<=3.34e-10
 if name=='elements.log':assert min_j==r['minimum_logged_J']
 streams[name]={'frame_count':121,'rows_per_frame':count,'ids_ordered':True,'max_header_time_error':timing,'minimum_directly_scanned_J':min_j if name=='elements.log' else None}
q=r['load_coordinate_full_m'];forces=r['applied_force_N'];work=[0.]
for i in range(1,len(q)):work.append(work[-1]+.5*(forces[i]+forces[i-1])*(q[i]-q[i-1]))
assert work==r['full_energy_work']['work_J']
error=max(abs(w-(e-r['energy_J'][0])) for w,e in zip(work,r['energy_J']))
assert error==r['full_energy_work']['maximum_error_J']
solver_ratio=max(s['actual_N2']/s['limit_N2'] for s in r['solver']['states'])
assert solver_ratio==r['criteria_max_ratio']['solver_residual']
assert len(r['solver']['states'])==120 and all(s['passed'] for s in r['solver']['states'])

# Authenticate the comparator's complete static script closure at the same commit
# before importing it. No generated-row/native-admission function is called.
closure=set();todo=['scripts/mechanics_hbe_v5_comparison.py']
while todo:
 path=todo.pop()
 if path in closure:continue
 closure.add(path)
 raw=(ROOT/path).read_bytes()
 for node in ast.walk(ast.parse(raw)):
  deps=[]
  if isinstance(node,ast.ImportFrom) and node.module:
   if node.module=='scripts':deps=['scripts/'+a.name.replace('.','/')+'.py' for a in node.names]
   elif node.module.startswith('scripts.'):deps=[node.module.replace('.','/')+'.py']
  elif isinstance(node,ast.Import):deps=[a.name.replace('.','/')+'.py' for a in node.names if a.name.startswith('scripts.')]
  todo.extend(p for p in deps if p not in closure)
comparator_sources={}
for path in sorted(closure):
 digest=sha(ROOT/path)
 original=subprocess.check_output(['/usr/bin/git','cat-file','blob',HEAD+':'+path],cwd=ROOT,timeout=10)
 assert hashlib.sha256(original).hexdigest()==digest,path
 comparator_sources[path]=digest
from scripts import mechanics_hbe_v5_comparison as comparison
from scripts import mechanics_hbe_branch_calibration_v5 as v5
from scripts import mechanics_hbe_branch_calibration_v4 as v4
study,prior=v5.validate_preparation(ROOT)
coarse_att=ROOT/Path(rel['prior_receipts'][-1]['path']).parent
coarse_receipt=read(coarse_att/'receipt.json')
assert sha(coarse_att/'receipt.json')==rel['prior_receipts'][-1]['sha256']=='b7f59dada629cd1fb8e61eac48a6eda86724b755055bdb7aa5dbf812d83fb013'
bind(coarse_receipt['output_bindings']['readout.json'])
c=read(coarse_att/'readout.json')
coarse_audit=ROOT/'build/hbe-v5-n36-result-independent'
coarse_summary=read(coarse_audit/'summary.json')
coarse_replay_receipt=read(coarse_audit/'receipt.json')
assert coarse_replay_receipt['readout_stage']['status']=='completed_within_caps'
assert coarse_replay_receipt['readout_stage']['exit_code']==0 and coarse_replay_receipt['readout_stage']['kill_reason'] is None
assert coarse_summary['entire_readout_json_equal'] is True
assert coarse_summary['receipt_sha256']==rel['prior_receipts'][-1]['sha256']
assert sha(coarse_audit/'replayed-readout.json')==sha(coarse_att/'readout.json')==coarse_summary['byte_identical_readout_sha256']

def view(data):
 spec=v5.run_spec(study,prior,data['run_id'])
 assert data['load_coordinate_full_m']==v5.schedule(study,prior,data['run_id'])['full_coordinates_m']
 assert data['numerical_passed'] is True
 assert data['frame_count']==spec['steps']+1
 assert data['representation']=='reconstructed_full'
 assert len(data['applied_force_N'])==len(data['probe_displacements_m'])==spec['steps']+1
 assert all(math.isfinite(f) for f in data['applied_force_N'])
 assert all(len(ps)==75 and all(len(p)==3 and all(math.isfinite(v) for v in p) for p in ps) for ps in data['probe_displacements_m'])
 return {'run_id':data['run_id'],'branch':spec['branch'],'mesh_N':spec['N'],'steps':spec['steps'],'mu_Pa':spec['reference_mu_Pa'],'frame_count':spec['frame_count'],'load_coordinate_m':data['load_coordinate_full_m'],'applied_force_N':data['applied_force_N'],'probe_displacements_m':data['probe_displacements_m']}
coarse,fine=view(c),view(r)
v4_pair=v4.compare_pairwise_arrays(prior,'compression_temporal',coarse,fine)
v5_pair=comparison._signed_pair(coarse,fine)
for key in ('signed_force_difference_N','signed_probe_component_difference_m','maximum_force_change_N','force_limit_N','probe_norm_limit_m'):
 assert v4_pair[key]==v5_pair[key],key
assert math.isclose(v4_pair['maximum_probe_norm_change_m'],v5_pair['maximum_probe_norm_change_m'],rel_tol=1e-14,abs_tol=0.)
assert v4_pair['passed']==v5_pair['pairwise_limit_passed']
assert coarse['load_coordinate_m']==fine['load_coordinate_m'][::2]
force_delta=v5_pair['signed_force_difference_N']
max_force_state=max(range(61),key=lambda i:abs(force_delta[i]))
probe_candidates=[(math.hypot(*p),i,j,p) for i,ps in enumerate(v5_pair['signed_probe_component_difference_m']) for j,p in enumerate(ps)]
max_probe=max(probe_candidates,key=lambda x:x[0])
# Retain all state differences and locate decreases/increases and all exceedances.
per_state=[]
for i,(df,ps) in enumerate(zip(force_delta,v5_pair['signed_probe_component_difference_m'])):
 maximum=max(math.hypot(*p) for p in ps)
 per_state.append({'coarse_state':i,'fine_state':2*i,'full_coordinate_m':q[2*i],
   'signed_force_difference_N':df,'absolute_force_difference_N':abs(df),
   'maximum_probe_vector_difference_m':maximum,
   'force_within_limit':abs(df)<=v5_pair['force_limit_N'],
   'all_probes_within_limit':maximum<=v5_pair['probe_norm_limit_m']})
comparison_record={'status':'frozen_temporal_pairwise_diagnostic_only','native_admission_or_reference_release':False,
 'coarse_receipt_sha256':rel['prior_receipts'][-1]['sha256'],'fine_receipt_sha256':REC_SHA,
 'coarse_readout_sha256':sha(coarse_att/'readout.json'),'fine_readout_sha256':sha(ATT/'readout.json'),
 'coarse_independent_audit_summary_sha256':sha(coarse_audit/'summary.json'),
 'coarse_independent_replay_receipt_sha256':sha(coarse_audit/'receipt.json'),
 'existing_functions_only':['mechanics_hbe_branch_calibration_v4.compare_pairwise_arrays','mechanics_hbe_v5_comparison._signed_pair'],
 'comparator_sources_at_release_commit':comparator_sources,
 'v4_result':v4_pair,'v5_result':v5_pair,'per_common_state':per_state,
 'maximum_force_difference_state':max_force_state,'maximum_force_difference_fine_state':2*max_force_state,
 'maximum_probe_difference':{'norm_m':max_probe[0],'coarse_state':max_probe[1],'fine_state':2*max_probe[1],'probe_index':max_probe[2],'signed_components_m':max_probe[3]},
 'force_exceedance_states':[x['coarse_state'] for x in per_state if not x['force_within_limit']],
 'probe_exceedance_states':[x['coarse_state'] for x in per_state if not x['all_probes_within_limit']],
 'full_twelve_row_qualification':False,'physical_validation_pass':None}
save('temporal-comparison.json',comparison_record)
failed=read(ROOT/rel['prior_receipts'][1]['path'])
summary={'decision':'GO_numerical_specimen_fixture_only' if v5_pair['pairwise_limit_passed'] else 'numerical_row_pass_temporal_pair_failed',
 'source_commit':HEAD,'receipt_sha256':REC_SHA,'release_sha256':REL_SHA,'entire_readout_json_equal':True,
 'byte_identical_readout_sha256':sha(OUT/'replayed-readout.json'),'closed_output_bytes':source['closed_output_bytes'],
 'native_nodes':39610,'native_hex8':34992,'representation':r['representation'],
 'reconstruction_provenance':r['provenance']['reconstruction_provenance'],'streams':streams,
 'final_full_load_coordinate_m':q[-1],'final_full_simulated_force_N':forces[-1],'S60_final_full_simulated_force_N':c['applied_force_N'][-1],
 'endpoint_signed_force_change_N':force_delta[-1],
 'final_full_energy_J':r['energy_J'][-1],'independent_full_trapezoidal_work_J':work[-1],
 'full_work_energy_error_J':error,'full_work_energy_limit_J':r['full_energy_work']['limit_J'],
 'native_work_energy_error_J':r['native_energy_work']['maximum_error_J'],'native_work_energy_limit_J':r['native_energy_work']['limit_J'],
 'criteria_max_ratio':r['criteria_max_ratio'],'minimum_sampled_J':r['minimum_sampled_J'],'minimum_logged_J':r['minimum_logged_J'],
 'solver_state_count':120,'solver_residual_ratio':solver_ratio,
 'temporal_common_states':61,'temporal_probe_count':75,'temporal_pair_passed':v5_pair['pairwise_limit_passed'],
 'maximum_force_change_N':v5_pair['maximum_force_change_N'],'force_limit_N':v5_pair['force_limit_N'],
 'maximum_probe_norm_change_m':v5_pair['maximum_probe_norm_change_m'],'probe_norm_limit_m':v5_pair['probe_norm_limit_m'],
 'maximum_force_change_coarse_state':max_force_state,'maximum_probe_change_coarse_state':max_probe[1],
 'force_exceedance_states':comparison_record['force_exceedance_states'],'probe_exceedance_states':comparison_record['probe_exceedance_states'],
 'native_stage':rec['native_stage'],'original_readout_stage':rec['readout_stage'],
 'independent_audit_replay_stage':supervision['readout_stage'],'native_calls_in_this_audit':0,'saved_stream_replay_calls_in_this_audit':1,
 'n12_original_status':failed['status'],'n12_original_failure':failed.get('failure'),
 'n12_guard_gap':source['n12_admission']['historical_guard_gap'],'known_timing_gap':rec['known_timing_gap'],
 'measured_response_accessed':False,'patient_data_accessed':False,'physical_validation_pass':None,
 'full_twelve_row_qualification':False,'elapsed_seconds_direct_counts_comparison_summary':time.monotonic()-started}
sources(rel,rec)
for path,digest in comparator_sources.items():assert sha(ROOT/path)==digest
for b in rec['output_bindings'].values():bind(b)
assert sha(REC)==REC_SHA and sha(REL)==REL_SHA
summary['sources_and_outputs_unchanged_at_final_verification']=True
save('summary.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k not in ('native_stage','original_readout_stage','independent_audit_replay_stage','criteria_max_ratio')},indent=2))
