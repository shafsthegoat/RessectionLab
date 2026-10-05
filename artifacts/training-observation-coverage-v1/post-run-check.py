"""Independent saved-only coverage audit. Never imports producer or loads images.

Summary absolute tolerance1e-12 was fixed before results; physical gates remain
1e-8mm and declared1e-6 mass limits. A passing audit cannot complete this study.
"""
from pathlib import Path
from itertools import product
import collections
import datetime
import hashlib
import json
import math
import subprocess
import tarfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/training-observation-coverage-v1/run-01'
REPORT = Path(__file__).with_suffix('.json')
ATOL, PHYSICAL_ATOL = 1e-12, 1e-8
SUBJECTS = ['sub-PAT05','sub-PAT16','sub-PAT20','sub-PAT22','sub-PAT25','sub-PAT28']
BLOCKED = {'sub-PAT16','sub-PAT20'}
NAMES = ['legacy_access_crop','target_local','whole_source']
UNION = 'target_local_plus_whole_source'
CHANNELS = ['structural_intensity','nominal_tissue','nominal_target','observed_cavity','nominal_motor','nominal_language']
COUNTS, MAXDIFF = collections.Counter(), collections.defaultdict(float)


def need(ok,message):
    if not ok: raise AssertionError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024**2),b''): h.update(b)
    return h.hexdigest()


def sem(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path): return json.loads(Path(path).read_text())


def close(a,b,where,*,physical=False):
    a,b=np.asarray(a,float),np.asarray(b,float)
    need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),where+': finite/shape')
    error=float(np.max(np.abs(a-b))) if a.size else 0.
    MAXDIFF['physical_mm' if physical else 'summary_arithmetic']=max(MAXDIFF['physical_mm' if physical else 'summary_arithmetic'],error)
    need(error <= (PHYSICAL_ATOL if physical else ATOL), f'{where}: absolute difference {error!r}')


def raw_inventory():
    return {str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}


def corners(shape,affine,cells=True):
    coords=list(product(*[(-.5,n-.5) if cells else (0.,n-1.) for n in shape]))
    return np.asarray(coords)@affine[:3,:3].T+affine[:3,3]


def clip_segment(a,b,lower,upper):
    # Intersect the six half-space inequalities with parameter t in[0,1].
    lows,highs=[0.],[1.]
    for x,d,l,h in zip(a,b-a,lower,upper):
        if d==0:
            if x<l or x>h: return None
        else:
            u,v=(l-x)/d,(h-x)/d
            lows.append(min(u,v));highs.append(max(u,v))
    lo,hi=max(lows),min(highs)
    return None if lo>hi else [float(lo),float(hi)]


def interval_length(interval): return 0. if interval is None else interval[1]-interval[0]


def union_length(a,b):
    if a is None:return interval_length(b)
    if b is None:return interval_length(a)
    overlap=max(0.,min(a[1],b[1])-max(a[0],b[0]))
    return interval_length(a)+interval_length(b)-overlap


def interval_equal(saved,expected,label):
    need((saved is None)==(expected is None),label+': null intersection')
    if expected is not None:close(saved,expected,label)


def segment(first,last,view,saved,label):
    affine=np.asarray(view['affine_ras_mm']);shape=np.asarray(view['shape'])
    points=first+np.arange(5)[:,None]/4*(last-first)
    close(saved['sample_points_ras_mm'],points,label+'/sample points',physical=True)
    inv=np.linalg.inv(affine)
    ijk=points@inv[:3,:3].T+inv[:3,3]
    normalized=2*ijk/(shape-1)-1
    # Frozen legacy sampler's declared64-epsilon boundary conditioning; this
    # preserves its precise center-domain convention, not physical clamping.
    edge=np.abs(np.abs(normalized)-1)<=64*np.finfo(float).eps
    normalized=np.where(edge,np.sign(normalized),normalized)
    inside=(np.abs(normalized)<=1).all(axis=1)
    cellinside=((ijk>=-.5)&(ijk<=shape-.5)).all(axis=1)
    need(saved['sample_inside_center_domain']==inside.tolist(),label+'/center samples')
    need(saved['sample_inside_fullcell_extent']==cellinside.tolist(),label+'/cell samples')
    ci=clip_segment(normalized[0],normalized[-1],[-1]*3,[1]*3)
    fi=clip_segment(ijk[0],ijk[-1],[-.5]*3,shape-.5)
    interval_equal(saved['center_domain_interval'],ci,label+'/center interval')
    interval_equal(saved['fullcell_extent_interval'],fi,label+'/cell interval')
    close(saved['continuous_center_domain_fraction'],interval_length(ci),label+'/center fraction')
    close(saved['continuous_fullcell_extent_fraction'],interval_length(fi),label+'/cell fraction')
    # All completed initial records declare constant full known coverage for
    # four allowed channels and no motor/language coverage. This permits a
    # saved-only coverage interpolation check without reopening their arrays.
    ranges=view['coverage_fraction_range']
    need(all(low==high for low,high in ranges),label+': constant coverage required for this saved-only audit')
    expected=inside[:,None]*np.array([low for low,high in ranges])[None,:]
    close(saved['sample_channel_coverage_fraction'],expected,label+'/known coverage')
    COUNTS['individual_segments']+=1;COUNTS['five_point_domain_samples']+=5
    return {'center':ci,'cell':fi,'inside':inside,'cellinside':cellinside,'coverage':expected}


def original(subject,declaration):
    if subject=='sub-PAT05':path=ROOT/declaration['common_task']['definition_path'];raw=path.read_bytes()
    else:
        folder=ROOT/'artifacts/remaining-training-frozen-spatial-float64-v1'
        path=folder/subject/'preparation.json'
        if path.exists():raw=path.read_bytes()
        else:
            with tarfile.open(folder/'completed-run.tar.gz') as archive:
                raw=archive.extractfile(subject+'/preparation.json').read()
    need(hashlib.sha256(raw).hexdigest()==declaration['members'][subject]['original_record_sha256'],'original anchor '+subject)
    return json.loads(raw)


def audit_case(subject,row,original_record,declaration):
    z=row['representation'];c=z['coverage'];prep=c['preparation_report'];views=c['views'];inv=z['initial_inventory']
    need(z['status']=='complete' and z['physical_catalog_unchanged'] is True,'completed representation '+subject)
    need(z['task_before']==z['task_after'],'unchanged saved state '+subject)
    for field in ['executed_transitions','policy_forwards','optimizer_updates']:need(z[field]==0,subject+'/'+field)
    need(inv==original_record['initial_inventory'],'original physical proposal inventory '+subject)
    before=z['task_before'];need(before['inventory_hash']==sem(inv),'inventory digest '+subject)
    physical=[{k:r[k] for k in ('tool_id','entry_mm','tip_mm','feasible','reason')} for r in inv['emitted']]
    need(before['physical_inventory_hash']==sem(physical),'physical inventory digest '+subject)
    need(before['observation_hash']==original_record['initial_observation_hash'],'original observation identity '+subject)
    need(inv['steps_taken']==0 and inv['remaining_steps']==inv['max_steps']==3 and inv['terminal'] is False,'initial horizon '+subject)
    emitted=inv['emitted']; accepted=sum(r['feasible'] for r in emitted)
    need(len(emitted)==inv['emitted_count']==inv['evaluated_slots']==row['initial_previews']==78,'exact preview count '+subject)
    need(accepted==inv['accepted_count']==c['accepted_actions'] and inv['rejected_count']==78-accepted,'accepted/rejected '+subject)
    need(c['emitted_actions']==len(c['actions'])==78 and len({r['action_id'] for r in emitted})==78,'one row per emitted action '+subject)
    need(inv['crop_clipping'] is False and inv['complete'] is True and inv['ledger_complete'] is True,'catalog semantics '+subject)
    aff=np.asarray(c['source_affine_ras_mm']);shape=np.asarray(c['source_shape']);cell=abs(float(np.linalg.det(aff[:3,:3])))
    grid=declaration['members'][subject]['member']['expected_native_grid_binding']
    close(aff,grid['derived_affine_ras_mm'],subject+'/declared derived frame',physical=True)
    close(c['original_source_affine_ras_mm'],grid['original_affine_ras_mm'],subject+'/original frame',physical=True)
    close(prep['source_affine_ras_mm'],aff,subject+'/source report frame',physical=True)
    close(prep['source_cell_volume_mm3'],cell,subject+'/cell volume')
    source_mass=c['source_nominal_mass_mm3']
    # Cross-check the original independently bound source-count/volume receipt;
    # we do not claim a fresh segmentation integral without reopening images.
    close(source_mass,original_record['coverage']['full_target_volume_mm3'],subject+'/original target volume')
    close(source_mass,original_record['coverage']['full_target_source_cells']*cell,subject+'/bound binary source-cell count')
    close(source_mass,prep['nominal_target_total_mass_mm3'],subject+'/reported nominal integral')
    close(c['source_fullcell_corners_ras_mm'],corners(shape,aff),subject+'/source extent',physical=True)
    lo,hi=np.asarray(prep['target_bbox_min_ijk']),np.asarray(prep['target_bbox_max_ijk'])
    local_shape=np.minimum(shape,declaration['settings']['local_shape'])
    start=np.clip((lo+hi-(local_shape-1))//2,0,shape-local_shape)
    need(start.tolist()==prep['local_start_ijk'],'bbox midpoint/clamp '+subject)
    local_aff=aff.copy();local_aff[:3,3]=aff[:3,:3]@start+aff[:3,3]
    coarse_shape=np.minimum(shape,declaration['settings']['coarse_shape']);scale=shape/coarse_shape
    coarse_aff=aff.copy();coarse_aff[:3,:3]=aff[:3,:3]*scale;coarse_aff[:3,3]=aff[:3,:3]@((scale-1)/2)+aff[:3,3]
    close(views['target_local']['affine_ras_mm'],local_aff,subject+'/local affine',physical=True)
    close(views['whole_source']['affine_ras_mm'],coarse_aff,subject+'/conservative affine',physical=True)
    need(views['target_local']['shape']==local_shape.tolist() and views['whole_source']['shape']==coarse_shape.tolist(),'view shapes '+subject)
    for name,view in views.items():
        a=np.asarray(view['affine_ras_mm']);s=np.asarray(view['shape'])
        close(view['fullcell_corners_ras_mm'],corners(s,a),subject+'/'+name+'/cell corners',physical=True)
        close(view['center_corners_ras_mm'],corners(s,a,False),subject+'/'+name+'/center corners',physical=True)
        invaff=np.linalg.inv(a);pts=corners(s,a)
        returned=(pts@invaff[:3,:3].T+invaff[:3,3])@a[:3,:3].T+a[:3,3]
        error=float(np.max(np.linalg.norm(returned-pts,axis=1)))
        need(error<=PHYSICAL_ATOL and view['roundtrip_max_error_mm']<=PHYSICAL_ATOL,'roundtrip gate '+subject+'/'+name)
        close(view['roundtrip_max_error_mm'],error,subject+'/'+name+'/roundtrip summary')
        close(view['nominal_mass_fraction'],view['nominal_mass_mm3']/source_mass,subject+'/'+name+'/mass ratio')
        close(view['nominal_mass_omitted_mm3'],max(0.,source_mass-view['nominal_mass_mm3']),subject+'/'+name+'/omitted mass')
        need(view['channel_available']==[True,True,True,True,False,False],'unknown functional channels '+subject+'/'+name)
    close(views['target_local']['nominal_mass_mm3'],prep['target_local_retained_mass_mm3'],subject+'/local retained integral')
    close(prep['target_local_clipped_fraction'],max(0.,min(1.,1-prep['target_local_retained_mass_mm3']/source_mass)),subject+'/clipped fraction')
    close(views['whole_source']['nominal_mass_mm3'],prep['channel_integrals']['nominal_target']['coarse_value_integral_mm3'],subject+'/coarse integral join')
    masserror=abs(views['whole_source']['nominal_mass_mm3']-source_mass)
    limit=1e-6 if source_mass<1 else 1e-6*source_mass
    need(masserror<=limit,'coarse mass gate '+subject)
    for key,expected in [('coarse_nominal_mass_absolute_error_mm3',masserror),('coarse_nominal_mass_relative_error',masserror/source_mass),('coarse_nominal_mass_allowed_absolute_error_mm3',limit)]:close(c[key],expected,subject+'/'+key)
    extent=float(np.max(np.linalg.norm(corners(shape,aff)-corners(coarse_shape,coarse_aff),axis=1)))
    need(extent<=PHYSICAL_ATOL,'whole source extent gate '+subject)
    close(c['global_extent_max_error_mm'],extent,subject+'/extent summary')
    close(prep['maximum_extent_corner_error_mm'],extent,subject+'/preparation extent summary')
    need(prep['target_local_fingerprint']==views['target_local']['fingerprint'] and prep['whole_source_fingerprint']==views['whole_source']['fingerprint'],'view/report fingerprints '+subject)
    tools={r['tool_id']:r for r in declaration['common_task']['tools']}
    for proposal,action in zip(emitted,c['actions'],strict=True):
        need(action['action_id']==proposal['action_id'] and action['tool_id']==proposal['tool_id'] and action['accepted']==proposal['feasible'] and action['reason']==proposal['reason'],'action join '+subject)
        tool=tools[proposal['tool_id']];entry=np.asarray(proposal['entry_mm']);tip=np.asarray(proposal['tip_mm'])
        axis=(tip-entry)/np.linalg.norm(tip-entry)
        segments={'entry_to_tip':(entry,tip),'approach_shaft_centerline':(entry-tool['working_length_mm']*axis,entry-tool['tip_length_mm']*axis),
                  'deepest_shaft_centerline':(tip-tool['working_length_mm']*axis,tip-tool['tip_length_mm']*axis)}
        for kind,(first,last) in segments.items():
            saved=action['visibility'][kind]
            recomputed={name:segment(first,last,views[name],saved[name],subject+'/'+proposal['action_id']+'/'+kind+'/'+name) for name in NAMES}
            a,b=recomputed['target_local'],recomputed['whole_source'];u=saved[UNION]
            need(u['sample_inside_center_domain']==(a['inside']|b['inside']).tolist(),'union center samples')
            need(u['sample_inside_fullcell_extent']==(a['cellinside']|b['cellinside']).tolist(),'union cell samples')
            close(u['continuous_center_domain_fraction'],union_length(a['center'],b['center']),'union center length')
            close(u['continuous_fullcell_extent_fraction'],union_length(a['cell'],b['cell']),'union cell length')
            close(u['sample_channel_coverage_fraction_max_across_views'],np.maximum(a['coverage'],b['coverage']),'union maximum coverage')
            COUNTS['union_segments']+=1
        COUNTS['emitted_actions']+=1
    for selection,predicate in [('emitted',lambda r:True),('accepted',lambda r:r['accepted'])]:
        rows=[r for r in c['actions'] if predicate(r)];summary=c['visibility_summary'][selection]
        need(summary['action_count']==len(rows),'summary denominator '+subject+'/'+selection)
        for name in [*NAMES,UNION]:
            for kind in ['entry_to_tip','approach_shaft_centerline','deepest_shaft_centerline']:
                actual=summary['views'][name][kind]
                for key,field in [('mean_center_domain_fraction','continuous_center_domain_fraction'),('mean_fullcell_extent_fraction','continuous_fullcell_extent_fraction'),('mean_center_sample_fraction','sample_inside_center_domain')]:
                    if not rows:need(actual[key] is None,'zero-accepted null '+subject)
                    else:
                        values=[sum(r['visibility'][kind][name][field])/5 if isinstance(r['visibility'][kind][name][field],list) else r['visibility'][kind][name][field] for r in rows]
                        close(actual[key],math.fsum(values)/len(values),'summary mean '+subject+'/'+selection+'/'+name+'/'+kind+'/'+key)
    return {'accepted':accepted,'emitted':78,'mass_fractions':{n:views[n]['nominal_mass_fraction'] for n in NAMES},
            'source_nominal_mass_mm3':source_mass,'whole_source_extent_error_mm':extent,'coarse_mass_error_mm3':masserror,
            'visibility_summary':c['visibility_summary'],'saved_task_invariants_equal':True}


def audit():
    before=raw_inventory();record=read(OUT/'receipt.json');declaration=read(OUT/'declaration-input.json');release=read(OUT/'release-input.json');parent=read(OUT/'supervisor.json')
    need(sha(OUT/'release-input.json')=='2dc9081a42205d6ba0dc20af5726164bf2c0e04a38e9bcb80b51cdcf2eaf566c','exact release snapshot')
    need(release['authorized'] is True and release['runtime_root']==str(ROOT),'release authority/root')
    need(sha(OUT/'declaration-input.json')==release['manifest_sha256']==parent['declaration_sha256'],'declaration byte joins')
    need(release['source_commit']=='7d585d87ca6584c77e31bfc3c0777f993bac03da','source commit')
    archive=Path(release['source_archive']);need(sha(archive)==release['source_archive_sha256'],'source archive bytes')
    hashes=declaration['source_sha256'];need(len(hashes)==release['source_file_count']==62,'source inventory count')
    expected={**hashes,'manifests/experiments/training-observation-coverage-v1.json':release['manifest_sha256']}
    with tarfile.open(archive) as tar:
        need(tar.pax_headers.get('comment')==release['source_commit'],'archive commit')
        seen=set()
        for member in tar:
            if member.isdir():continue
            need(member.isfile() and member.name in expected and member.name not in seen,'closed archived inventory')
            data=tar.extractfile(member).read();need(hashlib.sha256(data).hexdigest()==expected[member.name],'archive source '+member.name)
            git=subprocess.run(['git','show',release['source_commit']+':'+member.name],cwd=ROOT,capture_output=True,check=True)
            need(git.stdout==data,'committed archive source '+member.name);seen.add(member.name)
        need(seen==set(expected) and len(seen)==release['archive_file_count']==63,'exact63 source members')
    need(all(sha(ROOT/p)==h for p,h in hashes.items()),'current bound source preservation')
    need(sha(ROOT/release['review_receipt'])==release['review_sha256'],'preparation review bytes')
    need(declaration['subjects']==SUBJECTS and list(record['subjects'])==SUBJECTS,'ordered six patient denominator')
    need(record['settings']==declaration['settings'] and record['status']=='incomplete','incomplete declared study')
    need(parent['status']=='failed' and parent['returncode']==1 and parent['timed_out'] is False and parent['termination_reason'] is None and parent['automatic_retry'] is False,'terminal parent failure authority')
    need(read(OUT/'supervisor-failure.json')==parent,'durable failure receipt equality')
    need(record['cohort_denominator']==6 and record['historical_support_blocks']==2 and record['new_task_attempts']==4 and record['new_representation_completions']==3,'partial denominator')
    need(record['total_previews']==312,'preview total')
    for key in ['executed_transitions','policy_forwards','optimizer_updates']:need(record[key]==0,'zero '+key)
    need(record['clinical_deficit_probability'] is None,'unknown clinical harm')
    settings=declaration['settings']
    need(settings['extent_roundtrip_atol_mm']==1e-8 and settings['coarse_mass_rtol']==settings['coarse_mass_small_volume_atol_mm3']==1e-6,'unchanged scientific tolerances')
    need(record['elapsed_seconds']<170 and record['elapsed_seconds']<parent['seconds']<174 and settings['whole_worker_envelope_seconds']==180,'wall budgets')
    need(max(record['peak_rss_bytes'],parent['sampled_peak_rss_bytes'])<settings['max_rss_bytes']==6*1024**3,'memory budget')
    need(all(record['bundle_bytes_verified_before'].values()) and all(record['bundle_bytes_unchanged'].values()) and set(record['bundle_bytes_unchanged'])==set(SUBJECTS),'saved bundle preservation')
    need(record['sources_unchanged'] is True and record['original_records_unchanged'] is True,'saved source/metadata preservation')
    for stage in ['initial_import_paths','final_import_paths']:
        for name,path in record[stage].items():
            relative=str(Path(path).relative_to(ROOT));need(relative in hashes,'loaded module source closure '+name)
    completed={};bundle_hashes={};anchor_hashes={}
    for subject,row in record['subjects'].items():
        member=declaration['members'][subject]['member'];orig=original(subject,declaration)
        path=ROOT/member['case_bundle'];bundle_hashes[str(path.relative_to(ROOT))]=sha(path)
        need(bundle_hashes[str(path.relative_to(ROOT))]==member['case_bundle_sha256'],'opaque bundle hash '+subject)
        anchor_hashes[subject]=declaration['members'][subject]['original_record_sha256']
        if subject in BLOCKED:
            need(row['status']=='historical_support_block' and row['new_task_attempted'] is False and row['representation'] is None,'retained historical null '+subject)
            need(row['historical_preparation_sha256']==anchor_hashes[subject] and row['historical_coverage']==orig['coverage'],'historical evidence join '+subject)
        else:
            need(row['new_task_attempted'] is True and row['initial_previews']==78,'bounded actual attempt '+subject)
            if subject=='sub-PAT05':
                need(row['status']=='failed' and row['representation'] is None and row['failure']=={'type':'ValueError','message':'PAT05 physical task or native frame changed'},'preserved PAT05 failure')
            else:
                need(row['status']=='complete','completed status '+subject)
                completed[subject]=audit_case(subject,row,orig,declaration)
    need(sum(row.get('initial_previews',0) for row in record['subjects'].values())==312,'sum actual preview calls')
    need(before==raw_inventory(),'raw bytes unchanged during audit')
    need(all(sha(ROOT/p)==h for p,h in hashes.items()),'source after audit')
    return {'status':'passed_saved_partial_evidence','study_status':'incomplete','parent_status':'failed','parent_returncode':1,
        'fixed_tolerances':{'summary_absolute':ATOL,'summary_relative':0,'physical_mm':PHYSICAL_ATOL,'coarse_mass_rtol':1e-6,'coarse_mass_small_volume_atol_mm3':1e-6},
        'counts':dict(COUNTS),'maximum_absolute_differences':dict(MAXDIFF),'cohort':{'denominator':6,'attempted':4,'complete':3,'failed':1,'historical_blocks':2,'previews':312},
        'failure':record['subjects']['sub-PAT05']['failure'],'completed_cases':completed,'source_commit':release['source_commit'],
        'archive_sha256':release['source_archive_sha256'],'declaration_sha256':release['manifest_sha256'],'release_sha256':sha(OUT/'release-input.json'),
        'source_files_verified':62,'archive_files_verified':63,'source_anchors_verified':anchor_hashes,'opaque_bundle_hashes':bundle_hashes,'raw_before_after_sha256':before,
        'timing':{'worker_seconds':record['elapsed_seconds'],'parent_seconds':parent['seconds'],'worker_peak_rss_bytes':record['peak_rss_bytes'],'parent_sampled_peak_rss_bytes':parent['sampled_peak_rss_bytes'],'scope':'Nested intervals not additive; sampled RSS may miss transient peaks; failure was not a timeout.'},
        'patient_arrays_loaded':False,'native_calls':0,'policy_forwards':0,'refits':0,
        'limits':['Only3 completed coverage records verified; no representation result for PAT05 and no new result for historical blockedPAT16/20.',
            'Mass arithmetic joins independently bound original count/volume receipts and saved integrals; no new image or segmentation integral was computed.',
            'Five-point known coverage is reconstructed using the saved constant per-channel coverage ranges. Dense view arrays/fingerprints are not regenerated.',
            'Before/after state-array and actor hashes are saved equality evidence, not an independent reload of simulator arrays.',
            'Shaft centerline visibility and image extent are not finite-radius tool clearance, removal, safety or reachable resection.',
            'All3 completed local and legacy crops already contain the entire permitted target mass; this provides no target-coverage or RL-benefit improvement claim.',
            'PAT25 has0 accepted geometric actions despite78 emitted previews; preserve zero-denominator accepted summaries as null.']}


if __name__=='__main__':
    start=time.monotonic();prior=read(REPORT).get('attempts',[]) if REPORT.exists() else []
    try:result=audit()
    except Exception as error:result={'status':'failed_audit','study_status':'incomplete','error_type':type(error).__name__,'error':str(error),'counts':dict(COUNTS),'maximum_absolute_differences':dict(MAXDIFF)}
    result.update(checker_sha256=sha(__file__),recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),audit_seconds=time.monotonic()-start)
    result['attempts']=prior+[{k:result[k] for k in ['status','checker_sha256','recorded_at_utc','counts','error_type','error'] if k in result}]
    REPORT.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','counts','error','maximum_absolute_differences'] if k in result}))
    raise SystemExit(1 if result['status']=='failed_audit' else 0)
