"""Saved JSON/source and scalar affine audit, only after root terminal notice.

No DICOM, array, image, scientific import, geometry replay or model execution.
"""
import argparse,hashlib,itertools,json,math,resource,runpy,signal,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
BASE=ROOT/'build/balanced-teacher-il64-audit-v1/audit_saved.py'
BASE_SHA='b12157e2a3d92e959798092d22e17d6c29ebfa2070b726cbd8b5b932261fb1'
if hashlib.sha256(BASE.read_bytes()).hexdigest()!=BASE_SHA:raise ValueError('Bounded helper changed')
H=runpy.run_path(str(BASE));require=H['require'];read=H['read'];sha=H['sha'];near=H['near']
P=ROOT/'build/remind-train-domain-crop-preparation-v1';R=P/'actual-domain-qc-v1'
OLD=ROOT/'build/remind-train-public-qc-preparation-v1/actual-public-qc-v1'
SUBJECTS=['ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045']
def write(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def pinned(ref):
 p=ROOT/ref['path'];require(sha(p)==ref['sha256'] and H['seen'][ref['path']]['bytes']==ref['bytes'],'exact small metadata binding');return read(p)
def inverse(matrix):
 a=[list(map(float,row))+[float(i==j) for j in range(4)] for i,row in enumerate(matrix)]
 for c in range(4):
  p=max(range(c,4),key=lambda i:abs(a[i][c]));a[c],a[p]=a[p],a[c];v=a[c][c];require(abs(v)>1e-12,'nonsingular affine');a[c]=[x/v for x in a[c]]
  for i in range(4):
   if i!=c:v=a[i][c];a[i]=[x-v*y for x,y in zip(a[i],a[c])]
 return [row[4:] for row in a]
def multiply(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def bounds(g,affine,cells=False):
 t=multiply(inverse(affine),g['affine_xyz_to_ras_mm']);points=[]
 for p in itertools.product(*[(-.5,n-.5) if cells else (0,n-1) for n in g['shape_xyz']]):points.append([sum(x*y for x,y in zip(row,[*p,1])) for row in t[:3]])
 return {'minimum':[min(p[i] for p in points) for i in range(3)],'maximum':[max(p[i] for p in points) for i in range(3)]}
def det3(a):return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])
def main(receipt_sha,release_sha):
 start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60s metadata audit cap')));signal.alarm(60)
 require(sha(R/'receipt.json')==receipt_sha and sha(P/'root-release.json')==release_sha,'root terminal pins')
 receipt=read(R/'receipt.json');release=read(P/'root-release.json');prepared=read(P/'prepared-release.json');declaration=read(R/'declaration.json');index=pinned(release['source_index']);public=pinned(release['public_index'])
 require(prepared['execution_released'] is False and release=={**prepared,'execution_released':True},'only root execution release bit differs')
 require(declaration['release_sha256']==release_sha and declaration['source_index']==release['source_index'] and declaration['public_index']==release['public_index'],'source/release declaration joins')
 for ref in index['files']:
  require(sha(ROOT/ref['path'])==ref['sha256'] and H['seen'][ref['path']]['bytes']==ref['bytes'],'exact indexed source/metadata bytes')
 require(public['subjects']==release['subjects']==receipt['planned_subjects']==SUBJECTS and public['planned_denominator']==4,'all four fixed denominator')
 require(receipt['fatal'] is None and receipt['caps']==release['caps'] and receipt['elapsed_seconds']<receipt['caps']['total_seconds'],'whole batch complete within cap')
 require(receipt['public_only'] is True and receipt['roles_changed'] is False and receipt['training_admitted'] is False and receipt['failed_cases_replaced'] is False and receipt['automatic_source_artifact_clearance'] is False,'no scope/admission expansion')
 pending=read(R/'pending-qualification-index.json');require(pending['planned_denominator']==4 and [r['patient_id'] for r in pending['cases']]==SUBJECTS and all(r['public_input_manifest'] is None and r['training_admitted'] is False and r['role']=='TRAIN' for r in pending['cases']),'null manifests four denominator')
 phases=['crop-mr-domains','saved-domain-review'];require([(r['subject'],r['phase']) for r in receipt['outcomes']]==[(s,p) for s in SUBJECTS for p in phases],'exact eight phase order')
 rows=[];phase_records=[]
 for subject,binding in zip(SUBJECTS,public['cases']):
  case=pinned(binding['case']);headers=pinned(binding['headers']);require(binding['patient_id']==subject and case['patient_id']==headers['patient_id']==subject and case['role']==headers['role']=='TRAIN','frozen subject/role')
  reports={}
  for phase in phases:
   supervisor=read(R/f'{subject}-{phase}-supervisor.json');require(supervisor==next(r for r in receipt['outcomes'] if (r['subject'],r['phase'])==(subject,phase)),'phase receipt join')
   require(supervisor['phase_complete'] is True and supervisor['exit_code']==0 and supervisor['reaped'] is True and not supervisor['remaining_owned_pids'] and not supervisor['cleanup_errors'] and not supervisor['cleanup_actions'] and supervisor['stop_reason'] is None and supervisor['result_guard_error'] is None,'clean owned phase completion')
   require(supervisor['sampled_peak_rss_bytes']<=release['caps']['sampled_rss_bytes'] and supervisor['case_elapsed_seconds']<release['caps']['seconds_per_case'] and supervisor['case_output_bytes']<=release['caps']['output_bytes_per_case'] and supervisor['final_phase_log_bytes']<=release['caps']['log_bytes_per_phase'],'phase/case caps')
   name='conversion-result.json' if phase==phases[0] else 'saved-domain-review.json';path=R/f'{subject}-{phase}'/name;report=read(path)
   require(sha(path)==supervisor['result_sha256'] and report['status']==supervisor['result_status'],'terminal report bytes/status')
   source=ROOT/('src/resectionlab/remind_planning_qc.py' if phase==phases[0] else 'build/remind-train-domain-crop-preparation-v1/audit_saved_domains.py')
   require(report['executing_script_sha256']==sha(source) and report['case_sha256']==binding['case']['sha256'] and report['header_snapshot_sha256']==binding['headers']['sha256'],'case/header/executing source identity')
   require(report['patient_id']==subject and report['role']=='TRAIN' and report['public_only'] is True and report['private_reference_loaded'] is False and report['training_admitted'] is False and report['actor_inputs_admitted'] is False and report['optimizer_updates_performed']==0 and report['full_coverage_factory_compatible'] is False,'public-only metadata result without admission')
   reports[phase]=report;phase_records.append(supervisor)
  crop=reports[phases[0]];review=reports[phases[1]];old=read(OLD/f'{subject}-crop-mr/conversion-result.json');shape=crop['shape_xyz'];affine=crop['planning_derived_affine_ras_mm'];nvox=math.prod(shape);voxel=abs(det3(affine));counts=review['saved_counts']
  require(review['conversion_receipt_sha256']==sha(R/f'{subject}-crop-mr-domains/conversion-result.json') and review['original_source_reopened'] is False and review['public_input_manifest'] is None and review['all_saved_artifact_hashes_verified'] is True and review['all_source_domain_voxels_world_checked'] is True,'saved array-check receipt join/limits')
  require(review['visual_review']['expert_anatomy_review'] is False and 'PENDING_ROOT' in review['visual_review']['observation'],'visual not inferred by reader')
  require(crop['crop_policy']=='public_source_domain_union_no_extrapolation' and crop['support_source_domain_required'] is True and crop['registration_performed'] is False and crop['whole_tumor_is_prescribed_resection_target'] is False and crop['selected_MR_pixels_match_independent_raw_bytes'] is True,'unchanged acquired MRI/no registration/target claim')
  partition=crop['public_target_support_domain_relation'];require(counts['target']==partition['full_placed_target_voxels'] and counts['target']==sum(counts[k] for k in ('target_in_known_support_positive','target_in_known_support_zero','target_in_unknown_support_domain')),'three-way complete target partition')
  for k in ('target_in_known_support_positive','target_in_known_support_zero','target_in_unknown_support_domain'):require(counts[k]==partition[k],'conversion/independent partition join')
  require(counts['target_outside_support']==counts['target_in_known_support_zero']+counts['target_in_unknown_support_domain'] and partition['known_label_zero_is_physical_empty'] is False and partition['unknown_label_zero_is_observed_empty'] is False,'known label zero versus unknown retained')
  require(0<=counts['support']<=counts['support_domain']<=nvox<=32000000 and 0<counts['target']<=counts['target_domain']<=nvox,'count/domain bounds')
  require(nvox-counts['support_domain']==review['support_unknown_domain_voxels']==crop['public_support_source_domain_unknown_voxels'] and review['public_support_domain_fully_covered']==crop['public_support_source_domain_complete']==(nvox==counts['support_domain']),'explicit missing support-domain coverage')
  geometry={r['kind']:r['geometry'] for r in headers['series']};mr=geometry['structural_t1ce'];label_rows={};cell_bounds=[]
  for kind,total,domains in (('cerebrum',counts['support'],counts['support_domain']),('whole_tumor',counts['target'],counts['target_domain'])):
   annotation=crop['annotations'][kind];placement=annotation['placement'];saved=review['label_summaries'][kind];g=geometry[kind];inside=bounds(g,affine);within_mr=bounds(g,mr['affine_xyz_to_ras_mm']);cell=bounds(g,mr['affine_xyz_to_ras_mm'],True);cell_bounds.append(cell)
   require(all(a>=-.5 and b<n-.5 for a,b,n in zip(inside['minimum'],inside['maximum'],shape)) and all(a>=-.5 and b<n-.5 for a,b,n in zip(within_mr['minimum'],within_mr['maximum'],mr['shape_xyz'])),'independent full source-centre affine bounds')
   for edge in ('minimum','maximum'):require(all(near(a,b,1e-7) for a,b in zip(inside[edge],review['source_centre_coverage'][kind]['mapped_centre_'+edge])),'independent versus saved crop-centre bounds')
   require(placement['source_positive_centres_outside_target_grid']==saved['source_positive_centres_cropped_from_bound_execution']==0,'no source-positive crop loss')
   require(placement['source_positive_voxels']==annotation['source_positive_voxels']==old['annotations'][kind]['source_positive_voxels']==saved['source_positive_voxels_from_bound_execution'],'unchanged native source-positive denominator')
   require(total==annotation['placed_positive_voxels']==placement['resampled_positive_voxels']==saved['saved_positive_voxels'] and domains==annotation['placed_source_domain_voxels']==saved['saved_domain_voxels'],'saved NN/domain count joins')
   require(near(total*voxel,saved['saved_positive_volume_mm3'],1e-6) and near(total*voxel,placement['resampled_positive_volume_mm3'],1e-6) and near(placement['source_positive_volume_mm3'],placement['source_positive_voxels']*abs(det3(g['affine_xyz_to_ras_mm'])),1e-6),'independent determinant volume arithmetic')
   require(near(saved['volume_change_percent'],100*(total*voxel/placement['source_positive_volume_mm3']-1),1e-7) and saved['resampling_can_omit_subvoxel_labels'] is True,'sampling change remains explicit')
   label_rows[kind]={'native_positive_voxels':placement['source_positive_voxels'],'old_source_positive_centres_cropped':old['annotations'][kind]['placement']['source_positive_centres_outside_target_grid'],'new_source_positive_centres_cropped':0,'old_placed_positive_voxels':old['annotations'][kind]['placed_positive_voxels'],'new_placed_positive_voxels':total,'source_positive_mm3':placement['source_positive_volume_mm3'],'new_placed_positive_mm3':total*voxel,'NN_volume_change_percent':saved['volume_change_percent'],'source_centre_bounds_in_new_crop':inside}
  low=[math.floor(min(b['minimum'][i] for b in cell_bounds)+.5)-1 for i in range(3)];high=[math.ceil(max(b['maximum'][i] for b in cell_bounds)+.5)+1 for i in range(3)];start_index=[max(0,x) for x in low];stop=[min(n,x) for n,x in zip(mr['shape_xyz'],high)]
  require(crop['public_crop_start_MR']==start_index and shape==[b-a for a,b in zip(start_index,stop)],'independent native outer-box/one-cell halo formula')
  require(crop['source_cropping_map']['uniform_inset_MR_cells_per_face']==0 and crop['source_cropping_map']['roundoff_halo_MR_cells_per_face']==1 and crop['reindex_policy']['maximum_crop_world_difference_mm']<=.001,'declared affine roundoff and no inset')
  rows.append({'subject':subject,'shape_xyz':shape,'native_crop_start':start_index,'counts':counts,'support_unknown_domain_voxels':review['support_unknown_domain_voxels'],'labels':label_rows,'no_planning_admission':True,'root_visual_review_pending_in_saved_worker':True})
 for path,meta in list(H['seen'].items()):require(sha(ROOT/path)==meta['sha256'],'post-read evidence/source stable')
 summary={'status':'PASS_saved_domain_metadata_audit','planned_cases':4,'completed_phases':8,'training_admitted':0,'rows':rows,'receipt_sha256':receipt_sha,'release_sha256':release_sha,'source_index':release['source_index'],'checks':require.__globals__['checks'],'inputs':len(H['seen']),'batch_seconds':receipt['elapsed_seconds'],'peak_sampled_rss_bytes':max(r['sampled_peak_rss_bytes'] for r in phase_records),'phases':phase_records,'audit_seconds':time.monotonic()-start,'audit_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'Saved small JSON/source hashes and scalar affine/count arithmetic only. Existing bound worker owns all array/hash/finite/binary/domain checks; this independent audit opened no DICOM, arrays, images or original samples. Crop-centre coverage is distinct from NN label preservation and anatomy. Old full-support factory incompatible; no training admission.'}
 write('audit-result.json',summary);write('input-hashes.json',H['seen']);print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','phases')},sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--receipt-sha256',required=True);p.add_argument('--release-sha256',required=True);a=p.parse_args();main(a.receipt_sha256,a.release_sha256)
