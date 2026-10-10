"""One read-only 12-mask, 166-saved-motion measurement; no native engine."""
import argparse,gc,hashlib,json,os,signal,sys,time,threading
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from batch_contract import guard,read_bound,CAPS,OUTPUT,KEYS,sha,need

def write(path,data):
 with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True);a=p.parse_args()
 lease=os.environ.get('RESECTIONLAB_PARENT_LEASE_FD');need(lease is not None and lease.isdecimal() and int(lease)>=3,'owned_parent_lease_required');fd=int(lease);os.fstat(fd)
 def watch():
  try:
   while os.read(fd,1):pass
  finally:os._exit(143)
 threading.Thread(target=watch,name='saved-mask-parent-lease',daemon=True).start()
 release,index,_=guard(a.release.resolve(),a.release_sha256,execute=True)
 OUTPUT.mkdir(exist_ok=False);start=time.monotonic();active=set();result={'status':'started','release_sha256':a.release_sha256,'cases':[{'patient_id':x['patient_id'],'status':'not_attempted'} for x in index['cases']],
  'geometric_queries':0,'mask_loads':0,'native_previews':0,'transitions':0,'search_calls':0,'models':0,'source_array_writes':0,'blocked_calls':0,'state_or_source_modified':False,'training_admitted':False}
 def stop(*_):raise TimeoutError('saved-mask worker deadline/interruption')
 previous={s:signal.signal(s,stop) for s in (signal.SIGALRM,signal.SIGTERM)};signal.setitimer(signal.ITIMER_REAL,CAPS['worker_seconds'])
 try:
  def audit(event,args):
   if event=='import' and args and isinstance(args[0],str) and (args[0]=='torch' or args[0].startswith(('torch.','resectionlab.native_resection','resectionlab.native_spatial_task','resectionlab.observed_search','resectionlab.spatial_policy'))):
    result['blocked_calls']+=1;raise PermissionError('No model/native/search import')
   if event=='open' and args and isinstance(args[0],(str,bytes)):
    path=Path(os.fsdecode(args[0])).resolve();suffix=path.suffix.lower()
    writing=(isinstance(args[1],str) and any(c in args[1] for c in 'wax+')) or (isinstance(args[2],int) and args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
    if (suffix=='.npy' and (str(path) not in active or writing)) or suffix in ('.dcm','.nii','.gz','.pt','.pth','.ckpt') or path.is_relative_to(ROOT/'data'):
     result['blocked_calls']+=1;raise PermissionError('Only current pinned public S/T/Ds masks permitted')
   if event in ('subprocess.Popen','os.system','os.fork','socket.connect','socket.bind'):
    result['blocked_calls']+=1;raise PermissionError('No external worker activity')
  sys.addaudithook(audit)
  sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'build/goal-conditioned-policy-v1'))
  import numpy as np
  from diagnostic_core import diagnose_case
  from darwin_fast_sampler import FastDarwinSampler
  sampler=FastDarwinSampler()
  def checkpoint():
   if time.monotonic()-start>=CAPS['worker_seconds']:raise TimeoutError('worker time cap')
   if sampler.process_resident_bytes(os.getpid())>CAPS['sampled_rss_bytes']:raise MemoryError('worker sampled RSS cap')
  def on_query():
   checkpoint();need(result['geometric_queries']<CAPS['max_geometric_queries'],'geometric_query_cap');result['geometric_queries']+=1
  for number,row in enumerate(index['cases']):
   checkpoint();subject=row['patient_id'];result['cases'][number].update(status='measurement_started')
   manifest=read_bound(row['manifest']);inventory=read_bound(row['inventory']);derivation=read_bound(row['derivation']);bindings=read_bound(row['bindings'])
   active.clear();active.update(str(Path(v['path']).resolve()) for v in row['masks'].values());masks=[]
   for key in KEYS:
    record=row['masks'][key];path=Path(record['path']);need(path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],'mask_fixity:'+key)
    loaded=np.load(path,allow_pickle=False,mmap_mode='r');result['mask_loads']+=1
    need(list(loaded.shape)==manifest['shape_xyz'] and str(loaded.dtype)=='uint8' and np.isin(loaded,(0,1)).all(),'binary_mask_layout')
    value=np.asarray(loaded,dtype=bool);value.setflags(write=False);masks.append(value);del loaded
   source_affine=np.asarray(manifest['affine_ras_mm'],float);grid=bindings['native_grid_reconciliation'];affine=np.asarray(grid['derived_affine_ras_mm'],np.float64)
   h=hashlib.sha256();h.update(json.dumps({'shape':affine.shape,'dtype':affine.dtype.str},sort_keys=True).encode());h.update(memoryview(np.ascontiguousarray(affine)).cast('B'))
   need('sha256:'+h.hexdigest()==grid['derived_affine_hash'],'saved_native_affine_hash')
   axis=source_affine[:3,0]/np.linalg.norm(source_affine[:3,0]);sign=-1 if derivation['access_side']=='positive_source_axis0' else 1
   aperture={'centre_mm':(source_affine[:3,:3]@np.asarray(derivation['access_index'])+source_affine[:3,3]).tolist(),'normal_inward':(sign*axis).tolist(),'radius_mm':derivation['radius_mm']}
   for motion in inventory['emitted']:
    need(abs(float((np.asarray(motion['entry_mm'])-aperture['centre_mm'])@np.asarray(aperture['normal_inward'])))<1e-8,'saved_entry_not_in_fixed_plane')
   diagnostic=diagnose_case(*masks,affine,aperture,inventory,checkpoint,on_query)
   diagnostic.update(patient_id=subject,role='TRAIN',input_records=row,aperture=aperture,mask_original_hashes={k:v['sha256'] for k,v in row['masks'].items()},native_grid_reconciliation=grid,
     geometry_scope='Saved reconciled native affine used exactly; aperture retains original source-affine definition. Capsule localization only, not replay or feasibility parity.')
   path=OUTPUT/(subject+'-diagnostic.json');write(path,diagnostic)
   result['cases'][number]={'patient_id':subject,'status':'measured_only','saved_motions':len(inventory['emitted']),'geometric_queries':diagnostic['geometric_queries'],'result_file':path.name,'result_sha256':sha(path),'initial_exposure':diagnostic['initial_exposure'],'global_depth_counts':diagnostic['raw_S_full_T_U_global_depth_counts']}
   # Unchanged source hashes are checked again without loading a fourth array.
   for key in KEYS:need(sha(Path(row['masks'][key]['path']))==row['masks'][key]['sha256'],'postread_mask_changed')
   del diagnostic,masks,value;active.clear();gc.collect();checkpoint()
  need(result['mask_loads']==12 and result['geometric_queries']==664,'full_batch_accounting');result['status']='saved_mask_diagnostic_complete'
 except BaseException as error:result.update(status='failed_or_incomplete',error=type(error).__name__+':'+str(error))
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result['elapsed_seconds']=time.monotonic()-start;write(OUTPUT/'result.json',result)
  for s,v in previous.items():signal.signal(s,v)
 return 0 if result['status']=='saved_mask_diagnostic_complete' else 1
if __name__=='__main__':raise SystemExit(main())
