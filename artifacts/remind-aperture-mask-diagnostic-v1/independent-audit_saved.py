"""Saved JSON/source bytes only; no mask reads or geometry execution."""
import hashlib,json,math,os,resource,signal,stat,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'build/remind-aperture-mask-diagnostic-v1'
OUT=Path(__file__).resolve().parent
seen={};checks=0;started=time.monotonic()
signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('60s audit')))
signal.alarm(60)
def need(v,msg):
 global checks
 checks+=1
 if not v:raise AssertionError(msg)
def raw(path,expected=None):
 p=ROOT/path
 need(p.suffix in ('.json','.py','.log'),'saved text only')
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 try:
  need(stat.S_ISREG(os.fstat(fd).st_mode),'regular file')
  with os.fdopen(fd,'rb',closefd=False) as f:b=f.read(3*1024**2+1)
 finally:os.close(fd)
 need(len(b)<=3*1024**2,'bounded read')
 h=hashlib.sha256(b).hexdigest()
 if expected:need(h==expected,'hash:'+str(p))
 seen[str(p.relative_to(ROOT))]={'sha256':h,'bytes':len(b)}
 return b
def read(p,expected=None):return json.loads(raw(p,expected))
def bound(r):return read(r['path'],r['sha256'])
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-9)
BINS=('wholly_proximal','straddling_or_touching_aperture','wholly_inward')
parts=('initial_shaft','initial_tip','swept_shaft','swept_tip')
def summary(s,A,ap,shape):
 n=s['count'];need(type(n)is int and n>=0,'count')
 bins=s['by_full_cell_depth'];need(tuple(bins)==BINS and sum(x['count'] for x in bins.values())==n,'partition')
 need((s['depth_range_mm'] is None)==(n==0),'empty range')
 if n:need(all(math.isfinite(v) for v in s['depth_range_mm']) and s['depth_range_mm'][0]<=s['depth_range_mm'][1],'range')
 normal=ap['normal_inward'];entry=ap['centre_mm'];half=.5*sum(abs(sum(normal[r]*A[r][c] for r in range(3))) for c in range(3))
 for name,v in bins.items():
  need(type(v['count'])is int and v['count']>=0 and len(v['witness_indices'])==min(v['count'],4) and v['witness_limit']==4,'witness bound')
  for i in v['witness_indices']:
   need(len(i)==3 and all(type(x)is int and 0<=x<d for x,d in zip(i,shape)),'witness grid')
   world=[sum(A[r][c]*i[c] for c in range(3))+A[r][3] for r in range(3)]
   d=dot([world[r]-entry[r] for r in range(3)],normal);lo,hi=d-half,d+half
   kind=BINS[0] if hi < -1e-8 else BINS[2] if lo>1e-8 else BINS[1]
   need(kind==name,'saved witness depth arithmetic')
   need(s['depth_range_mm'][0]-1e-9<=lo<=hi<=s['depth_range_mm'][1]+1e-9,'witness range')
release=read(BASE/'root-release.json','e0444da5a30d9ada66f2663dc4b910407accd19777a99cf27260830cfbde42ac')
receipt=read(BASE/'attempt-01.supervision/receipt.json')
result=read(BASE/'attempt-01/result.json','c26cb2d0531133a19b7c64a3cfbb9cba10efaf07e7a760175bc2c609fce1dbf4')
index=bound(release['input_index']);source=bound(release['source_index'])
for p,h in source['files'].items():raw(p,h)
need(receipt['status']=='complete' and receipt['exit_code']==0 and receipt['stop_reason'] is None,'owned success')
need(receipt['worker_termination_confirmed'] and not receipt['remaining_owned_pids'] and not receipt['cleanup_errors'] and not receipt['cleanup_actions'],'clean reap')
need(receipt['elapsed_seconds']<60<receipt['caps']['parent_seconds'] and receipt['sampled_peak_rss_bytes']<receipt['caps']['sampled_rss_bytes'],'resource caps')
need(receipt['result_sha256']==seen[str((BASE/'attempt-01/result.json').relative_to(ROOT))]['sha256'],'result receipt')
need(receipt['release_sha256']==result['release_sha256']==seen[str((BASE/'root-release.json').relative_to(ROOT))]['sha256'],'release receipt')
need(receipt['source_index']==release['source_index'] and receipt['caps']==release['caps'],'source caps')
need(result['status']=='saved_mask_diagnostic_complete' and result['mask_loads']==12 and result['geometric_queries']==664,'completed denominator')
need(all(result[k]==0 for k in ('native_previews','transitions','search_calls','models','source_array_writes','blocked_calls')),'zero other work')
need(result['training_admitted'] is False and result['state_or_source_modified'] is False,'no mutation/admission')
subjects=['ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045'];counts=[52,52,10,52];rows=[]
need([x['patient_id'] for x in result['cases']]==subjects,'four case order')
for row,record,count in zip(result['cases'],index['cases'],counts):
 subject=row['patient_id'];d=read(BASE/'attempt-01'/row['result_file'],row['result_sha256']);m=bound(record['manifest']);inv=bound(record['inventory']);binding=bound(record['bindings'])
 need(row['status']=='measured_only' and row['saved_motions']==count and row['geometric_queries']==4*count==len(d['queries']),'per case denominator')
 need(d['input_records']==record and d['patient_id']==subject and d['role']=='TRAIN','exact case binding')
 need(d['native_grid_reconciliation']==binding['native_grid_reconciliation'],'saved exact native frame')
 need(d['mask_original_hashes']=={k:v['sha256'] for k,v in record['masks'].items()},'twelve mask identities without reading')
 need(d['classification_and_strict_disc_tolerance_mm']==1e-8 and not d['unchanged_world_legal_motion_claim'],'classification scope')
 A=d['native_grid_reconciliation']['derived_affine_ras_mm'];ap=d['aperture'];shape=m['shape_xyz'];N=math.prod(shape)
 g=d['raw_S_full_T_U_global_depth_counts'];need(g==row['global_depth_counts'],'copied global counts')
 for s in g.values():summary(s,A,ap,shape)
 domain=m['public_source_domain_counts'];T=domain['target_in_known_support_positive']+domain['target_in_known_support_zero']+domain['target_in_unknown_support_domain']
 need(g['full_T']['count']==T and g['derived_U']['count']==N-domain['support_domain']-domain['target_in_unknown_support_domain'],'full T/U partition')
 overlap=domain['target_in_known_support_positive'];O=g['raw_S']['count']+T-overlap
 need(O+g['derived_U']['count']<=N,'S T overlap inclusion exclusion')
 exposure=d['initial_exposure'];need(exposure==row['initial_exposure'],'copied exposure')
 need(exposure['not_an_admitted_cavity'] and not exposure['source_zero_is_physical_air'] and exposure['no_cells_cleared_seeded_or_marked_known'],'hypothesis only')
 K=exposure['source_known_zero_candidates'];summary(K,A,ap,shape)
 need(K['count']<=exposure['geometric_disc_face_candidates'] and K['by_full_cell_depth'][BINS[1]]['count']==K['count'],'K face touch')
 need(exposure['boundary_U_count']==exposure['boundary_cell_count'] and exposure['boundary_known_zero_seed_count']==0 and exposure['actual_initial_connected_free_count']==0,'unknown boundary and disconnected initial free')
 need(sum(x['source_known_zero_cells'] for x in exposure['candidate_connected_components'])<=N-O-g['derived_U']['count'],'known zero component bounds')
 stats={p:{k:{'motions_positive':0,'count_sum_across_queries':0,'bins_sum':dict.fromkeys(BINS,0),'range':None} for k in g} for p in parts}
 for j,motion in enumerate(inv['emitted']):
  queries=d['queries'][4*j:4*j+4];need([q['part'] for q in queries]==list(parts),'four query ordering')
  for q in queries:
   need(q['proposal_id']==motion['proposal_id'] and q['tool_id']==motion['tool_id'] and q['saved_reason']==motion['reason'],'saved motion identity')
   need(q['outside_image_geometry'].startswith('unassessed') and q['aperture_depth_is_anatomical_intracranial_classification'] is False,'scope')
   n=q['full_in_grid_supercover_cells'];c=q['intersections'];need(max(c['raw_S']['count'],c['full_T']['count'])+c['derived_U']['count']<=n,'disjoint U upper bound')
   for k,s in c.items():
    summary(s,A,ap,shape);need(s['count']<=min(n,g[k]['count']),'intersection upper bound')
    z=stats[q['part']][k];z['motions_positive']+=s['count']>0;z['count_sum_across_queries']+=s['count']
    for b in BINS:z['bins_sum'][b]+=s['by_full_cell_depth'][b]['count']
    if s['count']:
     r=s['depth_range_mm'];z['range']=r if z['range'] is None else [min(z['range'][0],r[0]),max(z['range'][1],r[1])]
 rows.append({'patient_id':subject,'motions':count,'raw_S':g['raw_S']['count'],'full_T':T,'S_intersection_T':overlap,'derived_U':g['derived_U']['count'],'initial_connected_free':0,'K':K['count'],'K_component_sizes':[x['source_known_zero_cells'] for x in exposure['candidate_connected_components']],'K_face_adjacent_S':exposure['candidate_face_adjacent_raw_S_cells'],'K_face_adjacent_T':exposure['candidate_face_adjacent_full_T_cells'],'query_stats_overlapping_not_union':stats})
for p,r in list(seen.items()):raw(p,r['sha256'])
value={'status':'PASS_saved_only','checks':checks,'seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'receipt':receipt,'cases':rows,'scope':'Saved text arithmetic only; no masks, scientific imports, geometry queries, native replay or model. Query counts overlap and are not removal. Witness depth classifications checked; complete cell memberships not independently regenerated.'}
need(value['peak_rss_bytes']<512*1024**2,'audit observed memory')
for name,obj in [('audit-result.json',value),('input-hashes.json',seen)]:
 with (OUT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':value['status'],'checks':checks,'seconds':value['seconds'],'peak_rss_bytes':value['peak_rss_bytes'],'cases':[{k:v for k,v in x.items() if k!='query_stats_overlapping_not_union'} for x in rows]}))
