#!/usr/bin/env python3
"""Bounded saved-JSON/opaque-byte audit; no image decode or geometry execution."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PROBE=ROOT/'build/lausanne-critical-geometry-probe-v1'
ATTEMPT=PROBE/'attempt-001'
BOUND={}


def require(condition, description):
    if not condition:raise AssertionError(description)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def bind(path, expected=None):
    value=sha(path)
    require(expected is None or value==expected,'Byte binding: '+str(path))
    BOUND[str(Path(path).relative_to(ROOT))]=value
    return value


def load(path, expected=None):
    bind(path,expected)
    return json.loads(Path(path).read_bytes(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def semantic(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def points(rows, shape):
    require(all(isinstance(r,list) and len(r)==3 and all(type(v) is int and 0<=v<n for v,n in zip(r,shape))
        for r in rows),'Bounded integer queried coordinates')
    result={tuple(row) for row in rows}
    require(len(result)==len(rows),'No duplicate queried coordinates')
    return result


def main():
    started=time.perf_counter()
    result={'schema':'lausanne-source-geometry-saved-review-v1','status':'checking',
        'reviewed_at':datetime.now(timezone.utc).isoformat(),
        'scope':'Source-frame static component; saved evidence only. No safe-background, full-planner or clinical claim.',
        'fixed_arithmetic_absolute_tolerance_mm':1e-12,
        'review_actions':{'image_decodes':0,'geometry_calls':0,'patient_builds':0,'network_requests':0,
            'routes':0,'model_execution':0,'training':0}}
    try:
        summary=load(PROBE/'RESULT.json')
        intent=load(ATTEMPT/'intent.json')
        outcome=load(ATTEMPT/'outcome.json')
        worker=load(ATTEMPT/'worker-result.json',outcome['worker_result_sha256'])
        bind(ATTEMPT/'intent.json',outcome['intent_sha256'])
        bind(ATTEMPT/'worker.log',outcome['worker_log_sha256'])
        bind(PROBE/'run_probe.py',intent['script_sha256'])
        bind(ATTEMPT/'run_probe.py',intent['script_sha256'])
        require(summary['records']['worker_result_sha256']==outcome['worker_result_sha256'],'Summary worker binding')
        require(all(r['status']=='passed' for r in (summary,outcome,worker)),'Completed passed producer records')
        require(outcome['worker_returncode']==0 and outcome['termination'] is None and outcome['worker_reaped'] is True,
            'Worker terminal authority')
        require(worker['pid']==outcome['worker_pid'],'Actual worker PID')
        require(intent['scientific_workers']==outcome['scientific_workers']==1,'One worker')
        require(intent['worker_seconds_limit']==60 and intent['worker_rss_limit_bytes']==1024**3,'Declared resource caps')
        require(0<worker['elapsed_seconds']<=outcome['elapsed_seconds']<60,'Worker and parent wall bounds')
        samples=outcome['rss_samples']
        require(len(samples)>0 and all(samples[i]['elapsed_seconds']<samples[i+1]['elapsed_seconds'] for i in range(len(samples)-1)),
            'Ordered RSS samples')
        require(all(type(s['rss_bytes']) is int and s['rss_bytes']>=0 for s in samples),'Actual RSS samples available')
        peak=max(s['rss_bytes'] for s in samples)
        require(peak==outcome['maximum_sampled_rss_bytes']<=1024**3,'Recorded sampled peak')
        require(worker['worker_peak_rss_bytes']==outcome['worker_peak_rss_bytes']<=1024**3,'Worker self peak')
        require(set(intent['native_thread_limits'].values())=={'1'},'Declared one-thread environment')
        bind(ATTEMPT/'worker-phases.jsonl')
        phases=[json.loads(line) for line in (ATTEMPT/'worker-phases.jsonl').read_text().splitlines()]
        require([p['phase'] for p in phases]==['importing_frozen_existing_source','building_actual_source_bound_case',
            'checking_actual_canonical_exclusion','checking_zero_exclusion_software_control','recording_result'],'Ordered scientific phases')
        require(all(p['elapsed_seconds']<=worker['elapsed_seconds'] and p['peak_rss_bytes_so_far']<=worker['worker_peak_rss_bytes']
            for p in phases),'Phase time/RSS within retained completion')
        inventory=intent['source_snapshot_sha256']
        root_source=ATTEMPT/'source/resectionlab'
        require({str(p.relative_to(root_source)) for p in root_source.rglob('*.py')}==set(inventory),'Closed executed source snapshot')
        differences=[]
        for relative,expected in inventory.items():
            bind(root_source/relative,expected)
            bind(ROOT/'src/resectionlab'/relative,expected)
            committed=subprocess.run(['git','show',intent['source_head']+':src/resectionlab/'+relative],
                cwd=ROOT,capture_output=True,check=False)
            if committed.returncode!=0 or hashlib.sha256(committed.stdout).hexdigest()!=expected:
                differences.append(relative)
        for relative,expected in intent['receipt_sha256'].items():bind(ROOT/relative,expected)
        component=load(ROOT/'artifacts/lausanne-sub476-annotation-v1/component.json',worker['component_receipt_sha256'])
        require(worker['committed_component_receipt_matched'] is True,'Canonical component assertion')
        require(component['case_hash']==worker['query']['case_hash']==summary['binding']['case_hash'],'Case identity')
        require(component['evidence']==worker['evidence'] and component['constraints']==worker['critical_receipt'],
            'Exact evidence and canonical constraint receipt')
        require(semantic(worker['critical_receipt'])==component['critical_binding_hash']==worker['query']['critical_binding_hash'],
            'Canonical critical fingerprint')
        require(component['positive_voxels']==worker['positive_annotation_cells']==193
            and component['negative_annotation_voxels']==worker['negative_annotation_cells']==0,'Positive-only source count binding')
        evidence=worker['evidence'];actual=worker['actual'];control=worker['software_control'];query=worker['query']
        require(actual['mask_hash']==actual['coverage_hash']==evidence['mask_hash']==evidence['annotation_coverage_hash'],
            'Canonical exclusion is the positive annotation and its domain')
        require(evidence['source_binding']['coverage_policy']=='positive_support_only'
            and evidence['source_binding']['background_meaning']=='unknown_for_vascular_anatomy','No inferred negative domain')
        require(evidence['source_binding']['participant']=='sub-476' and evidence['source_binding']['timepoint']=='ses-20140519',
            'Source participant/timepoint')
        manifest=load(ROOT/'manifests/lausanne-sub476-manual-annotation-v1.json')
        require(manifest['role']=='TRAIN' and manifest['subject']=='sub-476' and manifest['session']=='ses-20140519','Declared TRAIN case')
        require(worker['source_files_before']==worker['source_files_after'] and worker['original_sources_unchanged'] is True,
            'Source files unchanged by probe')
        for relative,record in worker['source_files_before'].items():
            bind(ROOT/relative,record['sha256'])
            require((ROOT/relative).stat().st_size==record['bytes'],'Opaque source file size')
        require(semantic(query)==worker['query_hash']==summary['query']['query_hash'],'Exact query hash')
        require(query['voxel_index']==[135,245,78] and query['access'] is None,'Declared source-positive query, no access')
        require(query['tool']['tip_radius_mm']==query['tool']['shaft_radius_mm']==.25
            and query['tool']['working_length_mm']==2. and query['tool']['tip_length_mm']==.5,'Declared mathematical probe')
        affine=evidence['affine_ras_mm'];index=query['voxel_index']
        expected_mm=[sum(affine[row][j]*index[j] for j in range(3))+affine[row][3] for row in range(3)]
        require(all(abs(a-b)<=1e-12 for a,b in zip(expected_mm,query['tip_source_reference_mm'])),'Scalar source voxel-to-point mapping')
        norm=math.sqrt(sum(affine[i][0]**2 for i in range(3)))
        require(all(abs(affine[i][0]/norm-query['axis_unit'][i])<=1e-12 for i in range(3)),'Normalized source-affine first axis')
        shape=evidence['shape'];query_cells=points(actual['geometry']['swept_voxel_indices'],shape)
        control_cells=points(control['geometry']['swept_voxel_indices'],shape)
        saved_positive=points(actual['positive_contact_cells'],shape)
        intersection=query_cells&saved_positive;unknown=query_cells-saved_positive
        require(query_cells==control_cells and len(query_cells)==27,'Identical 27-cell conservative envelopes')
        require(saved_positive==intersection and len(intersection)==actual['positive_contact_cell_count']==3,'Scalar positive intersection')
        require(tuple(index)==min(intersection) and tuple(index) in query_cells,'Selected point occurs in saved positive contacts')
        require(len(unknown)==24,'Remaining queried cells unknown')
        require(not actual['geometry']['feasible'] and [f['reason'] for f in actual['geometry']['failures']]==['FORBIDDEN_COLLISION'],
            'Actual exclusion collision')
        require(control['geometry']['feasible'] is True and control['geometry']['failures']==[]
            and control['canonical_case_or_annotations_modified'] is False
            and control['scope']=='zero_exclusion_software_ablation_only_not_real_safe_tissue','Software control scope')
        coverage=actual['annotation_domain_coverage']
        for name,row in coverage.items():
            require(row['in_image_swept_cells']==27 and row['outside_image']=='unassessed','Coverage scope')
            if name=='vessels':require(row['covered_cells']==3 and row['uncovered_cells']==24,'Vessel arithmetic')
            else:require(row['covered_cells'] is None and row['uncovered_cells'] is None,'Missing functional evidence remains unknown')
        require(math.prod(shape)-193==worker['background_unknown_cells']==18800447,'Whole-image unknown arithmetic')
        require(worker['critical_receipt']['objective_structures']==[],'No functional reward objectives')
        require(all(worker[k] is False for k in ('scanner_frame_admitted','spatial_planning_admitted','clinical_clearance')),
            'No frame/planner/clinical admission')
        require(all(worker[k]==0 for k in ('generated_routes','generated_targets','generated_access_support',
            'recorded_surgical_transitions','simulator_episodes','optimizer_updates')),'No invented planning stage')
        result.update(status='verified_saved_component_probe',source_commit=intent['source_head'],
            source_snapshot_files=len(inventory),snapshot_vs_commit_differences=differences,
            canonical_exclusion_count=193,selected_source_index=index,queried_cell_count=len(query_cells),
            intersecting_saved_positive_indices=sorted(intersection),unknown_query_indices=sorted(unknown),
            actual_failure='FORBIDDEN_COLLISION',software_control='zero modeled exclusions only; not safe tissue',
            worker_seconds=worker['elapsed_seconds'],supervised_seconds=outcome['elapsed_seconds'],
            worker_self_peak_rss_bytes=worker['worker_peak_rss_bytes'],sampled_peak_rss_bytes=peak,
            source_consumption_static_review='Frozen script explicitly passes canonical_hard_exclusion(case) as GeometryScene forbidden_mask with the exact case affine before check_pose; second call uses a separately labeled all-zero checker scene. Builder creates no target, brain support or context.',
            limitations=[
                'All 193 positive indices are not saved. Global first-positive ordering and mask count are authenticated through the frozen builder assertions, original opaque-file hashes and prior component receipt; only the saved three-cell intersection is independently reconstructed here.',
                'The 27 cells are a conservative full-tool envelope, not removed tissue or a surgical path.',
                'Snapshot/script copies match retained intent and current bytes; this is reproducibility evidence, not an OS-level import trace or continuously enforced source immutability.',
                'Parent launched the original probe script, whose bytes match its retained copy; a copied script in the attempt directory is not itself an independently usable command because its relative root changes.',
                'Worker RSS watchdog samples are not instantaneous kernel enforcement; no fresh peak or original geometry is measured by this review.',
                'Source-reference normalization is retrospective; scanner/interscan registration, unseen vessel anatomy, motor/language function and outside-image coverage remain unaccepted or unknown.',
                'No original image decoding, complete planner, route, tissue admission, safety, clinical efficacy or model performance is established.'])
    except BaseException as error:
        result.update(status='review_failed',error={'type':type(error).__name__,'message':str(error)})
    result['source_and_saved_evidence_sha256']=BOUND
    result['bound_files_unchanged']=all(sha(ROOT/path)==expected for path,expected in BOUND.items())
    result['elapsed_seconds']=time.perf_counter()-started
    result['review_script_sha256']=sha(__file__)
    target=OUT/'verification.json'
    require(not target.exists(),'Preserve previous independent review')
    target.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','elapsed_seconds','bound_files_unchanged')}
        | {'error':result.get('error'),'source_files':result.get('source_snapshot_files'),'commit_differences':result.get('snapshot_vs_commit_differences')}))
    raise SystemExit(0 if result['status']=='verified_saved_component_probe' and result['bound_files_unchanged'] else 1)


if __name__=='__main__':main()
