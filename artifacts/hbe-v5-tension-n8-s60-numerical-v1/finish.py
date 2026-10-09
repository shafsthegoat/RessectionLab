"""Finish the full-native tension N8 saved replay; no native execution or fit."""
import json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit import ROOT,OUT,REL,REC,ATT,HEAD,REL_SHA,REC_SHA,read,sha,save,sources,bind
from scripts import mechanics_hbe_v5_remaining_one_shot as runner
started=time.monotonic();rel,rec,r=read(REL),read(REC),read(ATT/'readout.json')
a=read(OUT/'source-audit.json');supervision=read(OUT/'receipt.json');replay=read(OUT/'replay-comparison.json')
assert replay['entire_json_equal'] and replay['frame_count']==61
assert supervision['readout_stage']['status']=='completed_within_caps' and supervision['readout_stage']['exit_code']==0 and supervision['readout_stage']['kill_reason'] is None
assert sha(ATT/'readout.json')==sha(OUT/'replayed-readout.json')
assert rec['caps']==rel['caps']==runner.caps(7) and rec['aggregate_caps']==rel['aggregate_caps']==runner.AGGREGATE
assert r['run_id']=='tension:N8:S60:reference' and r['numerical_passed'] and r['frame_count']==61 and r['steps']==60
assert r['representation']=='full_native_fixture' and r['provenance']['reconstruction_provenance']=='full_native'
assert r['native_energy_work'] is None and r['full_energy_work']['passed']
assert r['load_coordinate_full_m']==rel['adapter_receipt']['full_load_coordinates_m']
assert rel['adapter_receipt']['native_boundary_factor']==1.0
assert r['minimum_sampled_J']>0 and r['minimum_logged_J']>0
assert all(math.isfinite(x) and 0<=x<=1 for x in r['criteria_max_ratio'].values())
mesh=read(ROOT/rel['native_mesh']['path']);assert len(mesh['rest_nodes_m'])==1045 and len(mesh['elements_hex8'])==768
streams={}
for name,count in [('nodes.log',1045),('elements.log',768)]:
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
 assert len(frames)==61 and [x['step'] for x in frames]==list(range(61))
 timing=max(abs(x['time']-i/60) for i,x in enumerate(frames));assert timing<=3.34e-10
 if name=='elements.log':assert min_j==r['minimum_logged_J']
 streams[name]={'frame_count':61,'rows_per_frame':count,'ids_ordered':True,'max_header_time_error':timing,'minimum_directly_scanned_J':min_j if name=='elements.log' else None}
q=r['load_coordinate_full_m'];f=r['applied_force_N'];energy=r['energy_J'];probes=r['probe_displacements_m'];work=[0.]
assert len(q)==len(f)==len(energy)==len(probes)==61
assert q==[float('0.0007360099999999')*(i/60) for i in range(61)]
assert all(math.isfinite(x) for x in f+energy)
assert all(len(ps)==75 and all(len(p)==3 and all(math.isfinite(v) for v in p) for p in ps) for ps in probes)
for i in range(1,61):work.append(work[-1]+.5*(f[i]+f[i-1])*(q[i]-q[i-1]))
assert work==r['full_energy_work']['work_J']
error=max(abs(w-(e-energy[0])) for w,e in zip(work,energy));assert error==r['full_energy_work']['maximum_error_J']
solver_ratio=max(s['actual_N2']/s['limit_N2'] for s in r['solver']['states'])
assert solver_ratio==r['criteria_max_ratio']['solver_residual'] and len(r['solver']['states'])==60 and all(s['passed'] for s in r['solver']['states'])
path={'schema':'hbe-v5-tension-n8-s60-signed-path-diagnostic-v1','run_id':r['run_id'],'receipt_sha256':REC_SHA,
 'readout_sha256':sha(ATT/'readout.json'),'states':[
 {'state':i,'pseudotime':i/60,'full_and_native_coordinate_m':q[i],'signed_simulated_force_N':f[i],
  'full_stored_energy_J':energy[i],'trapezoidal_work_J':work[i],
  'maximum_probe_vector_norm_m':max(math.hypot(*p) for p in probes[i])} for i in range(61)],
 'loaded_force_positive':all(x>0 for x in f[1:]),'initial_force_N':f[0],
 'negative_loaded_force_states':[i for i in range(1,61) if f[i]<0],
 'force_decrease_states':[i for i in range(1,61) if f[i]<f[i-1]],
 'reference_release_or_mesh_convergence':False,'physical_validation_pass':None}
save('signed-path.json',path)
summary={'decision':'GO_numerical_specimen_fixture_only','source_commit':HEAD,'release_sha256':REL_SHA,'receipt_sha256':REC_SHA,
 'entire_readout_json_equal':True,'byte_identical_readout_sha256':sha(OUT/'replayed-readout.json'),
 'closed_output_bytes':a['closed_output_bytes'],'native_nodes':1045,'native_hex8':768,'representation':r['representation'],
 'reconstruction_provenance':r['provenance']['reconstruction_provenance'],'streams':streams,'probe_count':75,
 'final_full_and_native_load_coordinate_m':q[-1],'final_signed_simulated_force_N':f[-1],
 'minimum_loaded_signed_simulated_force_N':min(f[1:]),'loaded_force_positive':path['loaded_force_positive'],
 'force_decrease_states':path['force_decrease_states'],'negative_loaded_force_states':path['negative_loaded_force_states'],
 'final_full_energy_J':energy[-1],'independent_full_trapezoidal_work_J':work[-1],'full_work_energy_error_J':error,
 'full_work_energy_limit_J':r['full_energy_work']['limit_J'],'native_energy_work':None,
 'criteria_max_ratio':r['criteria_max_ratio'],'minimum_sampled_J':r['minimum_sampled_J'],'minimum_logged_J':r['minimum_logged_J'],
 'solver_state_count':60,'solver_residual_ratio':solver_ratio,'native_stage':rec['native_stage'],'original_readout_stage':rec['readout_stage'],
 'independent_audit_replay_stage':supervision['readout_stage'],'native_calls_in_this_audit':0,'saved_stream_replay_calls_in_this_audit':1,
 'n12_original_status':'failed_or_incomplete','n12_guard_gap':a['n12_admission']['historical_guard_gap'],'known_timing_gap':rec['known_timing_gap'],
 'cumulative_native_calls':rec['aggregate_native_calls'],'cumulative_native_wall_seconds':rec['aggregate_native_wall_seconds'],
 'cumulative_original_readout_wall_seconds':rec['aggregate_readout_wall_seconds'],'cumulative_preparation_wall_seconds':rec['aggregate_prep_wall_seconds'],
 'cumulative_known_combined_wall_seconds':rec['aggregate_combined_wall_seconds'],
 'cumulative_primary_closed_output_bytes':a['prior_ledger']['output_bytes']+a['closed_output_bytes'],
 'cumulative_combined_closed_output_bytes':a['prior_ledger']['combined_output_bytes']+a['closed_output_bytes'],
 'measured_response_accessed':False,'patient_data_accessed':False,'physical_validation_pass':None,'full_twelve_row_qualification':False,
 'elapsed_seconds_direct_counts_path_summary':time.monotonic()-started}
sources(rel,rec)
for b in rec['output_bindings'].values():bind(b)
assert sha(REC)==REC_SHA and sha(REL)==REL_SHA
runner.audit_imports(root=ROOT)
summary['sources_and_outputs_unchanged_at_final_verification']=True
save('summary.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k not in ('native_stage','original_readout_stage','independent_audit_replay_stage','criteria_max_ratio')},indent=2))
