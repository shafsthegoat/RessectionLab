"""One released Case2 tag, independent Horn/IDW audit; no production math calls.

Quaternion reconstruction and scalar checks adapted from the committed Case4
evaluation-review/review.py. Outputs contain scalar errors, never coordinates.
"""
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import resource
import shlex
import signal
import statistics
import sys
import time
from datetime import datetime, timedelta

START = time.monotonic()
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('AUDIT_30_SECONDS')))
signal.alarm(30)
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'build/hybrid-real-observation-case2-v1'
RUN = BASE / 'attempt-01'
TAG = ROOT / 'data/anatomy/resect-v1/Case2/Landmarks/Case2-beforeUS-duringUS-full.tag'
TAG_SHA = 'a93b42da9a88eec1736fc287501d80b9aba89330ccaf970f6b0398c60b5a4a22'
INDEX_SHA = '0cf9ff70ff7b10b51e74ec29484687cbb81e1b8060cba917a8e0171e41fce461'
METHODS = ['no_shift', 'proper_rigid', 'inverse_distance_squared']
B = [1, 14, 13, 9, 2, 11]
V = [3, 4, 5, 6, 7, 8, 10, 12, 15, 16]
SEEN, CACHE, ACCESSED, DELTAS = {}, {}, [], []
CHECKS = 0

def check(value, label):
    global CHECKS
    CHECKS += 1
    if not value:
        raise AssertionError(label)

def guard(event, args):
    if event in ('subprocess.Popen', 'os.system', 'socket.connect'):
        raise PermissionError('NO_PROCESSES_OR_NETWORK')
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).absolute()
    resolved = path.resolve()
    if path.is_relative_to(ROOT / 'data') or resolved.is_relative_to(ROOT / 'data'):
        mode, flags = args[1:3]
        check(path == resolved == TAG, 'EXACT_RELEASED_TAG_ONLY')
        check(not (isinstance(mode, str) and any(c in mode for c in 'wax+')), 'NO_DATA_WRITE')
        check(not (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)), 'NO_DATA_WRITE')
        check(not ACCESSED, 'EXACTLY_ONE_TAG_OPEN')
        ACCESSED.append(str(path.relative_to(ROOT)))

sys.addaudithook(guard)

def digest(payload):
    return hashlib.sha256(payload).hexdigest()

def blob(path):
    path = Path(path)
    if path in CACHE:
        return CACHE[path]
    check(path.resolve() == path.absolute() and path.is_relative_to(ROOT), 'BOUND_NONALIAS_PATH')
    with path.open('rb') as handle:
        raw = handle.read(4 * 1024**2 + 1)
    check(len(raw) <= 4 * 1024**2, 'BOUNDED_METADATA')
    SEEN[str(path.relative_to(ROOT))] = digest(raw)
    CACHE[path] = raw
    return raw

def load(path):
    return json.loads(blob(path))

def bound(item):
    path = ROOT / item['path']
    raw = blob(path)
    check(digest(raw) == item['sha256'].removeprefix('sha256:'), 'EXACT_BOUND_BYTES')
    return json.loads(raw)

def semantic(value):
    return 'sha256:' + digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())

def array_hash(value):
    value = np.ascontiguousarray(value, dtype=np.float64)
    header = json.dumps({'shape': value.shape, 'dtype': value.dtype.str}, sort_keys=True).encode()
    return 'sha256:' + digest(header + value.tobytes())

def agreement(actual, expected, tolerance=1e-10):
    error = float(np.max(np.abs(np.asarray(actual, float) - np.asarray(expected, float))))
    check(math.isfinite(error) and error <= tolerance, 'NUMERICAL_AGREEMENT')
    DELTAS.append(error)
    return error

def summary(values):
    return {'count': len(values), 'rms_mm': math.sqrt(math.fsum(v*v for v in values)/len(values)),
            'median_mm': statistics.median(values), 'maximum_mm': max(values)}

def main():
    protocol = load(BASE / 'protocol.json')
    index = load(BASE / 'source-index.json')
    check(SEEN[str((BASE / 'source-index.json').relative_to(ROOT))] == INDEX_SHA, 'INDEX_PIN')
    check(protocol['patient_group'] == 'RESECT:Case2' and protocol['role'] == 'TRAIN', 'CASE_ROLE')
    check(protocol['inputs']['tag']['sha256'] == TAG_SHA and protocol['inputs']['tag']['bytes'] == 1097, 'ROOT_TAG_AUTHORITY')
    check(ROOT / protocol['inputs']['tag']['path'] == TAG, 'TAG_PATH')
    for name, pin in index['sources'].items():
        check(digest(blob(ROOT / name)) == pin, 'SOURCE_PIN')
    for record in index['generated_controls'].values():
        check(digest(blob(ROOT / record['path'])) == record['sha256'], 'CONTROL_PIN')
    prior_source = ROOT / 'artifacts/resect-case4-sparse-update-v1/evaluation-review/review.py'
    blob(prior_source)
    cohort = bound(protocol['role_binding'])
    roles = {r['patient_group']: r['role'] for r in cohort['members']}
    check(roles['RESECT:Case2'] == 'TRAIN' and roles['RESECT:Case4'] == 'DEVELOPMENT', 'UNCHANGED_ROLES')
    stages = {}
    for phase in ('qualify', 'fit', 'evaluate'):
        release_path = BASE / 'releases' / (phase + '.json')
        release = load(release_path)
        rh = digest(blob(release_path))
        check(release['phase'] == phase and release['authorized'] is True and release['authorizer'] == 'root', 'RELEASE')
        check(bound(release['source_index']) == index, 'SAME_PHASE_CLOSURE')
        for item in release['dependencies'].values():
            bound(item)
        records = {n: load(RUN / phase / (n + '.json')) for n in ('attempt', 'worker-started', 'result', 'supervision', 'process')}
        for n in ('attempt', 'worker-started', 'result', 'supervision'):
            check(records[n]['phase'] == phase and records[n]['release_sha256'] == rh, 'RELEASE_JOIN')
        result, sup = records['result'], records['supervision']
        check(result['status'] == sup['status'] == 'completed' and sup['exit_code'] == 0 and sup['scientific_result_accepted'] is True, 'ACCEPTED_PHASE')
        check(result['source_index_sha256'] == sup['source_index_sha256'] == INDEX_SHA, 'SOURCE_INDEX_JOIN')
        check(sup['result_sha256'] == digest(blob(RUN / phase / 'result.json')), 'RESULT_BINDING')
        check(result['original_bytes_unchanged'] is True and result['memory_is_observed_not_hard_limit'] is True, 'SOURCE_AND_MEMORY_SCOPE')
        check(0 <= result['elapsed_seconds'] <= sup['elapsed_seconds'] <= 30, 'PHASE_WALL')
        check(blob(RUN / phase / 'worker.log') == b'', 'EMPTY_CONSOLE')
        stages[phase] = {'release': release, 'result': result, 'supervision': sup}
    for earlier, later in (('qualify', 'fit'), ('fit', 'evaluate')):
        a, b = stages[earlier]['supervision'], stages[later]['supervision']
        check(datetime.fromisoformat(a['started_utc']) + timedelta(seconds=a['elapsed_seconds']) < datetime.fromisoformat(b['started_utc']), 'SEQUENTIAL_PHASE_CHRONOLOGY')
    q, f, e = (stages[p]['result'] for p in ('qualify', 'fit', 'evaluate'))
    partition, frame = bound(q['partition']), bound(q['frame'])
    check(partition['B_ids'] == B and partition['V_ids'] == V and partition['source_row_count'] == 16, 'ALL16_PARTITION')
    check(not set(B) & set(V) and sorted(B+V) == list(range(1,17)), 'DISJOINT_COMPLETE')
    check(q['destination_coordinates_parsed'] == partition['destination_coordinates_parsed'] == 0 and q['image_arrays_accessed'] is False, 'QUALIFY_SOURCE_ONLY')
    for phase in ('fit', 'evaluate'):
        dep = stages[phase]['release']['dependencies']
        for k in ('frame', 'partition'):
            check(dep['qualify_' + k] == q[k], 'QUALIFY_PREREQUISITE_JOIN')
    check(frame['convention'] == protocol['coordinate_convention'], 'CONVENTION')
    check(frame['source_world_to_ras_mm'] == frame['destination_world_to_ras_mm'] == np.eye(4).tolist(), 'IDENTITY_CONVENTION')
    check(frame['anatomical_alignment_accepted'] is False and frame['total_registration_uncertainty_mm'] is None, 'FRAME_LIMITS')
    for name in ('before', 'during'):
        h = frame['headers'][name]
        check(h['decompressed_bytes_returned'] == 348 and h['image_array_accessed'] is False, 'HEADER_ONLY')
        check(h['qform_code'] == 0 and h['sform_code'] == 1 and h['spatial_units'] == 'mm', 'NATIVE_HEADER_CONVENTION')
    prefix = RUN / 'fit/comparison'
    freeze = bound(f['freeze'])
    check(f['freeze']['sha256'] == '65af70b6997bad00f3d3c2cb4553f5a6d72843194573866390b7ee930c402453', 'ROOT_FREEZE_PIN')
    check(stages['evaluate']['release']['dependencies']['fit_freeze'] == f['freeze'], 'FROZEN_PREREQUISITE')
    model, predictions = bound(freeze['model']), bound(freeze['prediction_field'])
    check(bound(freeze['protocol']) == protocol and model['protocol'] == freeze['protocol'], 'FROZEN_PROTOCOL')
    check(predictions['external_field'] is None, 'NO_EXTERNAL_MODEL')
    field = predictions['baseline']
    check(digest((json.dumps(field, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()) == model['baseline_field_sha256'], 'FIELD_HASH')
    check(field['rules']['methods'] == METHODS and field['rules']['idw_power'] == 2 and field['rules']['idw_neighbors'] == 'all_six', 'FROZEN_METHODS')
    check(semantic(freeze['landmark_freeze']) == freeze['freeze_hash'], 'FREEZE_SEAL')
    for name, key in (('model', 'model_sha256'), ('prediction_field', 'prediction_sha256')):
        check(freeze['landmark_freeze'][key] == 'sha256:' + freeze[name]['sha256'], 'FREEZE_OBJECT_JOIN')
    marker = load(prefix / 'validation-attempt.json')
    report = bound(e['evaluation'])
    check(e['evaluation']['sha256'] == 'b44c61c43f6754a820e47985c8eb2272e219660f3e2bf91d7dd50662886e58b9', 'ROOT_EVALUATION_PIN')
    check(digest(blob(RUN / 'evaluate/result.json')) == 'bb491df8ce54e278448b55c55f23feb37e3ca0b4d3deb57f976a0992aa61b827', 'ROOT_RESULT_PIN')
    check(marker['freeze'] == report['freeze'] == f['freeze'] and report['landmark_freeze_hash'] == freeze['freeze_hash'], 'ONE_SHOT_FREEZE_JOIN')
    check(list(RUN.rglob('validation-attempt.json')) == [prefix / 'validation-attempt.json'], 'ONE_REVEAL_MARKER')
    check(f['B_ids'] == B and f['V_destinations_accessed'] == [] and e['V_ids'] == V, 'B_THEN_V_SCOPE')
    check(e['role'] == 'TRAIN' and e['future_untouched_validation_for_adapted_methods'] is False, 'V_NOW_CONSUMED')

    # Single explicitly authorized original read. Do not reopen during postguard.
    raw = blob(TAG)
    check(len(raw) == 1097 and digest(raw) == TAG_SHA, 'EXACT_ORIGINAL_TAG_FIXITY')
    check(hashlib.md5(raw).hexdigest() == protocol['inputs']['tag']['expected_md5'], 'PUBLISHED_TAG_MD5')
    text = raw.decode('ascii')
    check(text.startswith('MNI Tag Point File') and bool(re.search(r'Volumes\s*=\s*2\s*;', text)), 'TAG_FORMAT')
    body = re.split(r'Points\s*=', text, maxsplit=1)[1].strip()
    check(body.endswith(';'), 'TAG_TERMINATOR')
    rows = []
    for line in body[:-1].splitlines():
        tokens = shlex.split(line, comments=False)
        if tokens:
            check(len(tokens) in (6,7,9,10), 'TAG_ROW_FORM')
            rows.append([float(token) for token in tokens[:6]])
    rows = np.asarray(rows, dtype=np.float64)
    check(rows.shape == (16,6) and np.isfinite(rows).all(), 'TAG_ROW_DENOMINATOR')
    x, y = rows[:,:3].copy(), rows[:,3:].copy()
    check(len(np.unique(x,axis=0)) == 16, 'UNIQUE_SOURCES')
    chosen = [0]
    for _ in range(5):
        distances = [min(math.fsum((float(a)-float(b))**2 for a,b in zip(point,x[j])) for j in chosen) if i not in chosen else -1 for i,point in enumerate(x)]
        chosen.append(max(range(16),key=lambda i:(distances[i],-i)))
    check([i+1 for i in chosen] == B, 'INDEPENDENT_SOURCE_FPS')
    singular = np.linalg.svd(x[chosen]-x[chosen].mean(0),compute_uv=False)
    ratio = float(singular[-1]/singular[0])
    agreement(ratio,partition['rank_ratio'])
    check(ratio > 1e-6, 'SOURCE_RANK3')
    source_hash = semantic({'patient':'RESECT:Case2','role':'before_us_to_during_us','source_image':'sha256:'+protocol['inputs']['before']['sha256'],'source_world_mm':array_hash(x),'row_ids':list(range(1,17))})
    check(source_hash == partition['source_hash'], 'SOURCE_HASH')
    check(semantic({'source_hash':source_hash,'boundary_ids':B,'validation_ids':V,'rank_ratio':ratio,'rule':'source_fps_six_rank3_v1'}) == partition['partition_hash'] == freeze['landmark_freeze']['partition_hash'], 'PARTITION_SEAL')
    observations = field['observations']
    check(observations['boundary_ids'] == B and observations['source_ras_mm'] == x[chosen].tolist() and observations['observed_ras_mm'] == y[chosen].tolist(), 'SIX_B_EXACT_INPUT')
    check(semantic(observations) == field['forward_hash'] == model['forward_hash'] == freeze['landmark_freeze']['forward_hash'], 'FORWARD_SEAL')
    qc_hash = semantic({'source':'sha256:'+protocol['inputs']['before']['sha256'],'destination':'sha256:'+protocol['inputs']['during']['sha256'],'source_transform':array_hash(np.eye(4)),'destination_transform':array_hash(np.eye(4)),'report':'sha256:'+q['frame']['sha256'],'convention':frame['convention']})
    audit_hash = semantic({'patient_group':'RESECT:Case2','role':'before_us_to_during_us','tag_sha256':'sha256:'+TAG_SHA,'source_image_sha256':'sha256:'+protocol['inputs']['before']['sha256'],'destination_image_sha256':'sha256:'+protocol['inputs']['during']['sha256'],'frame_qc':qc_hash})
    check(audit_hash == freeze['landmark_freeze']['source_audit_hash'], 'SOURCE_FRAME_FREEZE_SEAL')

    # Case4 independent Horn quaternion algorithm; production uses Kabsch SVD.
    bx, by = x[chosen], y[chosen]
    cx, cy = bx.mean(0), by.mean(0)
    s = (bx-cx).T @ (by-cy)
    trace = float(np.trace(s))
    z = np.array([s[1,2]-s[2,1],s[2,0]-s[0,2],s[0,1]-s[1,0]])
    horn = np.empty((4,4)); horn[0,0] = trace; horn[0,1:] = horn[1:,0] = z
    horn[1:,1:] = s+s.T-trace*np.eye(3)
    values, vectors = np.linalg.eigh(horn)
    w,qx,qy,qz = vectors[:,-1]
    rotation = np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*w),2*(qx*qz+qy*w)], [2*(qx*qy+qz*w),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*w)], [2*(qx*qz-qy*w),2*(qy*qz+qx*w),1-2*(qx*qx+qy*qy)]])
    translation = cy-rotation@cx
    check(field['rigid']['status'] == 'available', 'RIGID_AVAILABLE')
    re = agreement(rotation,field['rigid']['rotation'])
    te = agreement(translation,field['rigid']['translation_mm'])
    check(abs(np.linalg.det(rotation)-1) < 1e-12 and np.max(np.abs(rotation.T@rotation-np.eye(3))) < 1e-12, 'PROPER_RIGID')
    def predict(points):
        idw = []
        for point in points:
            distance = [math.dist(point,anchor) for anchor in bx]
            if 0 in distance:
                motion = by[distance.index(0)]-bx[distance.index(0)]
            else:
                weight = [(min(distance)/d)**2 for d in distance]
                denominator = math.fsum(weight)
                motion = np.array([math.fsum(weight[i]*float(by[i,j]-bx[i,j]) for i in range(6))/denominator for j in range(3)])
            idw.append(point+motion)
        return {'no_shift':np.asarray(points),'proper_rigid':np.asarray(points)@rotation.T+translation,'inverse_distance_squared':np.asarray(idw)}
    vi = np.asarray(V)-1
    predicted, residuals = predict(x[vi]), predict(bx)
    metrics, bmetrics, errors = {}, {}, {}
    for method in METHODS:
        saved = report['methods'][method]
        check(saved['total_landmarks'] == 10 and saved['excluded_counts'] == {} and [r['row_id'] for r in saved['per_landmark']] == V, 'ALL_V_EVALUATED')
        check(all(r['status'] == 'supported' for r in saved['per_landmark']), 'SUPPORT_FLAGS')
        errors[method] = [math.dist(a,b) for a,b in zip(predicted[method],y[vi])]
        agreement(predicted[method],[r['predicted_ras_mm'] for r in saved['per_landmark']])
        agreement(errors[method],[r['error_mm'] for r in saved['per_landmark']])
        metrics[method] = summary(errors[method])
        # Independent scalar recomputation from the saved errors, separately.
        saved_summary = summary([r['error_mm'] for r in saved['per_landmark']])
        for key in ('rms_mm','median_mm','maximum_mm'):
            agreement(metrics[method][key], saved_summary[key])
            for support in ('all_supported','common_supported'):
                check(saved[support]['count'] == 10, 'METRIC_DENOMINATOR')
                agreement(metrics[method][key],saved[support][key])
        be = [math.dist(a,b) for a,b in zip(residuals[method],by)]
        bmetrics[method] = summary(be)
        bs = report['B_residuals'][method]
        check([r['row_id'] for r in bs['per_landmark']] == B and bs['summary']['count'] == 6, 'B_NOT_VALIDATION')
        agreement(be,[r['error_mm'] for r in bs['per_landmark']])
        for key in ('rms_mm','median_mm','maximum_mm'):
            agreement(bmetrics[method][key],bs['summary'][key])
    paired = {}
    for a,b in itertools.combinations(METHODS,2):
        delta = [u-v for u,v in zip(errors[a],errors[b])]
        key = a+'_minus_'+b; saved = report['paired_differences'][key]
        check(saved['count'] == 10 and [r['row_id'] for r in saved['per_landmark']] == V, 'PAIRED_DENOMINATOR')
        agreement(delta,[r['error_difference_mm'] for r in saved['per_landmark']])
        mean = math.fsum(delta)/10
        agreement(mean,saved['mean_error_difference_mm'])
        paired[key] = {'mean_error_difference_mm':mean,'first_lower_error_rows':sum(d<0 for d in delta),'second_lower_error_rows':sum(d>0 for d in delta),'exact_tie_rows':sum(d==0 for d in delta)}
    primary = metrics['proper_rigid']['rms_mm']-metrics['no_shift']['rms_mm']
    agreement(primary,e['primary_comparison']['rms_difference_mm'])
    check(e['primary_comparison']['required_all_V_count'] == 10 and report['common_supported_ids'] == V, 'PRIMARY_ALL_V')
    check(report['confidence_interval'] is None and report['clinical_injury_probability'] is None and report['physical_action_response_validated'] is False and report['external_mechanics_validated_by_this_helper'] is False, 'PHYSICAL_UNCERTAINTY_LIMITS')
    coverage = {}
    for name, points, image in (('source',x,'before'),('destination',y,'during')):
        h = frame['headers'][image]
        affine = np.asarray(h['selected_affine']); inverse = np.linalg.inv(affine)
        voxels = points@inverse[:3,:3].T+inverse[:3,3]
        inside = np.all((voxels>=-.5)&(voxels<=np.asarray(h['shape'])-.5),axis=1)
        saved = e['all_original_row_coverage'][name]
        check(saved['excluded_rows'] == [] and [r['row_id'] for r in saved['rows']] == list(range(1,17)), 'ALL_COVERAGE_ROWS')
        agreement(voxels,[r['native_voxel'] for r in saved['rows']])
        check(inside.tolist() == [r['inside_image_cell_box'] for r in saved['rows']], 'EXACT_CELL_BOX_FLAGS')
        coverage[name] = {'total':16,'inside_cell_box':int(inside.sum()),'outside_cell_box':int((~inside).sum()),'B_inside':int(inside[chosen].sum()),'V_inside':int(inside[vi].sum()),'excluded':0}
    check(e['all_original_row_coverage']['source'] == q['source_coverage'], 'SOURCE_COVERAGE_JOIN')
    dest = sorted(f['B_destination_coverage']['rows'] + e['V_destination_coverage']['rows'],key=lambda r:r['row_id'])
    check(dest == e['all_original_row_coverage']['destination']['rows'], 'B_V_COVERAGE_JOIN')
    paths = [prefix/'model.json',prefix/'prediction-field.json',prefix/'freeze.json',RUN/'fit/result.json',RUN/'fit/supervision.json',BASE/'releases/evaluate.json',RUN/'evaluate/attempt.json',prefix/'validation-attempt.json',prefix/'evaluation.json']
    times = [p.stat().st_mtime_ns for p in paths]
    check(times == sorted(times), 'CORROBORATING_FREEZE_REVEAL_MTIME_ORDER')
    total_bytes = sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())
    check(total_bytes <= 4*1024**2, 'AGGREGATE_OUTPUT_CAP')
    # Recheck all saved evidence/source; the already-opened original is excluded.
    for name, pin in SEEN.items():
        path = ROOT/name
        if path == TAG:
            continue
        with path.open('rb') as handle:
            after = handle.read(4*1024**2+1)
        check(digest(after) == pin, 'POSTREAD_SOURCE_EVIDENCE_UNCHANGED')
    check(ACCESSED == [str(TAG.relative_to(ROOT))], 'ONE_ORIGINAL_TAG_ONLY')
    return {'schema':'hybrid-case2-independent-numerical-audit-v1','status':'PASS','checks':CHECKS,
        'tag_bytes':len(raw),'tag_sha256':digest(raw),'original_tag_opens':len(ACCESSED),
        'case':'RESECT:Case2','role':'TRAIN','B_ids':B,'V_ids':V,'V_now_consumed':True,
        'B_source_rank_ratio':ratio,'V_metrics_mm':metrics,'B_conditioning_metrics_not_validation':bmetrics,
        'V_scalar_errors_mm':[{'row_id':row,**{m:errors[m][j] for m in METHODS}} for j,row in enumerate(V)],
        'paired_scalar_errors':paired,'primary_rigid_minus_static_RMS_mm':primary,
        'secondary_IDW_minus_rigid_RMS_mm':metrics['inverse_distance_squared']['rms_mm']-metrics['proper_rigid']['rms_mm'],
        'rigid_reconstruction':{'algorithm':'committed Case4 Horn quaternion, independent of production Kabsch','rotation_max_difference':re,'translation_max_difference_mm':te,'maximum_eigenvalue_gap_mm2':float(values[-1]-values[-2])},
        'maximum_numeric_disagreement':max(DELTAS),'agreement_tolerance':1e-10,'coverage':coverage,
        'phase_seconds':{p:s['supervision']['elapsed_seconds'] for p,s in stages.items()},
        'worker_peak_RSS_bytes':{p:s['result']['worker_peak_rss_bytes'] for p,s in stages.items()},
        'total_saved_output_bytes':total_bytes,'process_evidence':'Three exit0 accepted owned phases; source supervisor kills group/reaps in finally, no independent survivor sampling in these receipts.',
        'chronology_basis':'Authenticated phase prerequisites and reviewed freeze/reveal ordering; saved times corroborate, not hostile-host proof.',
        'scope':{'image_reads':0,'production_fit_evaluation_calls':0,'solver_model_training_calls':0,'other_original_reads':0,'patient_coordinates_written':0},
        'limitations':['One TRAIN person and10 correlated V, not patient-held-out generalization.','Six during-US B observations are supplied; no causal action-response or dense-field validation.','Unknown calibrated measurement uncertainty: IDW-rigid0.0043mm gap is not material/mechanistic proof.','Cell-box inclusion is not US signal, anatomy, cavity or clinical validation.','V consumed for future adapted methods; role unchanged.']}

try:
    result = main()
    result.update(audit_elapsed_seconds=time.monotonic()-START,
        audit_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024))
    check(result['audit_elapsed_seconds'] < 30, 'AUDIT_WALL')
except BaseException as exc:
    result = {'status':'FAILED_AUDIT','error_type':type(exc).__name__,'error_label':str(exc),'checks':CHECKS,'original_tag_opens':len(ACCESSED),'audit_elapsed_seconds':time.monotonic()-START}
    with (OUT/'failure.json').open('x') as handle:
        json.dump(result,handle,sort_keys=True,indent=2); handle.write('\n')
    raise
finally:
    signal.alarm(0)
with (OUT/'result.json').open('x') as handle:
    json.dump(result,handle,sort_keys=True,indent=2,allow_nan=False); handle.write('\n')
with (OUT/'input-index.json').open('x') as handle:
    json.dump({'sha256':SEEN,'source':{'path':str(Path(__file__).relative_to(ROOT)),'sha256':digest(Path(__file__).read_bytes())}},handle,sort_keys=True,indent=2);handle.write('\n')
print(json.dumps({'status':result['status'],'checks':result['checks'],'metrics':result['V_metrics_mm'],'primary_mm':result['primary_rigid_minus_static_RMS_mm'],'IDW_minus_rigid_mm':result['secondary_IDW_minus_rigid_RMS_mm'],'maximum_disagreement':result['maximum_numeric_disagreement'],'elapsed_seconds':result['audit_elapsed_seconds'],'peak_RSS_bytes':result['audit_peak_RSS_bytes']}))
