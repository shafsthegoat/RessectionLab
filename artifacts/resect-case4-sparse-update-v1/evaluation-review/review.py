"""Independent saved sparse-US audit; only original tag data may be opened.

No production fit/evaluation helper, image parser, solver, or model is invoked.
Quaternion reconstruction audits the already fixed B calculation; no tuning.
"""
from datetime import datetime, timedelta
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/resect-case4-sparse-update-v1'
COMMIT = '694022d13394ec9c2a28f0b56fcf1e00289321af'
B = [1, 14, 8, 17, 19, 7]
V = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 15, 16, 18]
METHODS = ['no_shift', 'proper_rigid', 'inverse_distance_squared']
START = time.monotonic()
SEEN = {}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def load(path):
    data = path.read_bytes()
    SEEN[str(path.relative_to(ROOT))] = digest(data)
    return json.loads(data)


def binding(item):
    path = ROOT / item['path']
    assert path.resolve() == path.absolute() and path.is_relative_to(ROOT)
    data = path.read_bytes()
    assert digest(data) == item['sha256'].removeprefix('sha256:')
    SEEN[item['path']] = digest(data)
    return json.loads(data)


def semantic(value):
    return 'sha256:' + digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def array_hash(value):
    value = np.ascontiguousarray(value, dtype=np.float64)
    header = json.dumps({'shape': value.shape, 'dtype': value.dtype.str}, sort_keys=True).encode()
    return 'sha256:' + digest(header + value.tobytes())


def summary(values):
    return {'count': len(values), 'rms_mm': math.sqrt(math.fsum(v*v for v in values)/len(values)),
            'median_mm': statistics.median(values), 'maximum_mm': max(values)}


def agreement(actual, expected, tolerance=1e-10):
    error = float(np.max(np.abs(np.asarray(actual, float)-np.asarray(expected, float))))
    assert error <= tolerance, (error, tolerance)
    return error


manifest_path = ROOT / 'manifests/experiments/resect-case4-sparse-update-v1.json'
manifest = load(manifest_path)
assert SEEN[str(manifest_path.relative_to(ROOT))] == '64229fe7523a65a90a878b02050273424ff8c9fb9f84d76c28110f097c2d7c5e'
assert manifest['B_ids'] == B and manifest['V_ids'] == V and manifest['role'] == 'DEVELOPMENT'
allowed_tag = ROOT / manifest['inputs']['tag']['path']
accessed = []


def guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).absolute()
    resolved = path.resolve()
    if path.is_relative_to(ROOT/'data') or resolved.is_relative_to(ROOT/'data'):
        assert path == resolved == allowed_tag, 'Only released original tag may be opened'
        mode, flags = args[1:3]
        assert not (isinstance(mode, str) and any(c in mode for c in 'wax+'))
        assert not (isinstance(flags, int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        accessed.append(str(path.relative_to(ROOT)))


sys.addaudithook(guard)
stages = {}
source_graph = None
for phase in ['header-qc', 'fit-freeze', 'evaluate']:
    release_path = BASE/'releases'/f'{phase}.json'
    release = load(release_path)
    release_sha = SEEN[str(release_path.relative_to(ROOT))]
    assert release['phase'] == phase and release['source_commit'] == COMMIT
    assert release['authorizer'] == 'root' and release['authorized'] is True
    assert release['declaration_sha256'] == SEEN[str(manifest_path.relative_to(ROOT))]
    if source_graph is None:
        source_graph = release['source_sha256']
    assert release['source_sha256'] == source_graph
    for item in release['dependencies'].values():
        binding(item)
    directory = BASE/phase
    records = {name: load(directory/f'{name}.json') for name in ['attempt', 'worker-started', 'result', 'supervision', 'process']}
    for name in ['attempt', 'worker-started', 'result', 'supervision']:
        assert records[name]['phase'] == phase and records[name]['release_sha256'] == release_sha
    result, sup = records['result'], records['supervision']
    assert result['status'] == sup['status'] == 'completed' and sup['exit_code'] == 0 and sup['scientific_result_accepted'] is True
    assert result['source_sha256'] == sup['source_sha256'] == source_graph
    assert result['runtime'] == manifest['runtime'] and result['original_bytes_unchanged'] is True
    assert 0 <= result['elapsed_seconds'] <= sup['elapsed_seconds'] <= 30
    assert result['peak_memory_is_observed_not_a_hard_limit'] is True
    assert (directory/'worker.log').stat().st_size == 0
    stages[phase] = {'release': release, 'result': result, 'supervision': sup}
for name, pin in {**source_graph, str(manifest_path.relative_to(ROOT)): release['declaration_sha256']}.items():
    assert digest((ROOT/name).read_bytes()) == pin
    saved = subprocess.run(['git', 'show', COMMIT+':'+name], cwd=ROOT, capture_output=True, check=True, timeout=5).stdout
    assert digest(saved) == pin
    SEEN[name] = pin
exe = Path(sys.executable).resolve()
runtime = {'python': sys.version, 'executable': str(exe), 'executable_sha256': digest(exe.read_bytes()),
           'numpy': importlib.metadata.version('numpy'), 'nibabel': importlib.metadata.version('nibabel')}
assert runtime == manifest['runtime']
for earlier, later in [('header-qc', 'fit-freeze'), ('fit-freeze', 'evaluate')]:
    first, second = stages[earlier]['supervision'], stages[later]['supervision']
    assert datetime.fromisoformat(first['started_at_utc'])+timedelta(seconds=first['elapsed_seconds']) < datetime.fromisoformat(second['started_at_utc'])

frame = load(BASE/'header-qc/frame-qc.json')
header_review_path = ROOT/'build/resect-case4-header-independent-review-v1/verification.json'
header_review = load(header_review_path)
assert SEEN[str(header_review_path.relative_to(ROOT))] == '0b6ed60315e54523ab993920f4fd7eacd95a2a22cc22e90de27d112091a46bd9'
for name, pin in header_review['stage_artifacts_sha256'].items():
    assert digest((ROOT/name).read_bytes()) == pin
for name in ['before', 'during']:
    assert header_review['original_compressed_fixity'][name]['sha256'] == manifest['inputs'][name]['sha256']
assert frame['source_world_to_ras_mm'] == frame['destination_world_to_ras_mm'] == np.eye(4).tolist()

prefix = BASE/'fit-freeze/comparison'
freeze = binding(stages['evaluate']['release']['dependencies']['freeze'])
assert stages['fit-freeze']['result']['freeze'] == stages['evaluate']['release']['dependencies']['freeze']
model = binding(freeze['model'])
predictions = binding(freeze['prediction_field'])
assert binding(freeze['protocol']) == manifest
assert predictions['external_field'] is None
field = predictions['baseline']
assert digest((json.dumps(field, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()) == model['baseline_field_sha256']
assert field['rules']['methods'] == METHODS and field['rules']['idw_power'] == 2 and field['rules']['idw_neighbors'] == 'all_six'
assert semantic(freeze['landmark_freeze']) == freeze['freeze_hash']
assert freeze['landmark_freeze']['model_sha256'] == 'sha256:'+freeze['model']['sha256']
assert freeze['landmark_freeze']['prediction_sha256'] == 'sha256:'+freeze['prediction_field']['sha256']
attempt = load(prefix/'validation-attempt.json')
assert attempt['freeze'] == stages['evaluate']['release']['dependencies']['freeze']
report = binding(stages['evaluate']['result']['evaluation'])
assert stages['evaluate']['result']['evaluation']['sha256'] == '928898feaed7b106e4fc3f4e7d280678402f1db24cdc4192cd58d10c3fe0480a'
assert report['freeze'] == attempt['freeze'] and report['landmark_freeze_hash'] == freeze['freeze_hash']
assert stages['fit-freeze']['result']['B_destinations_accessed'] == B and stages['fit-freeze']['result']['V_destinations_accessed'] == []
assert stages['evaluate']['result']['V_destinations_accessed'] == V and stages['evaluate']['result']['future_untouched_Case4_validation_available'] is False
chronology_paths = [prefix/'model.json', prefix/'prediction-field.json', prefix/'freeze.json', BASE/'fit-freeze/result.json', BASE/'fit-freeze/supervision.json', BASE/'releases/evaluate.json', BASE/'evaluate/attempt.json', prefix/'validation-attempt.json', prefix/'evaluation.json', BASE/'evaluate/result.json', BASE/'evaluate/supervision.json']
chronology = [{'path': str(p.relative_to(ROOT)), 'mtime_ns': p.stat().st_mtime_ns} for p in chronology_paths]
assert all(a['mtime_ns'] <= b['mtime_ns'] for a, b in zip(chronology, chronology[1:]))

# All 19 rows were explicitly released for this evaluation-only audit.
raw = allowed_tag.read_bytes()
tag = manifest['inputs']['tag']
assert len(raw) == tag['bytes'] and digest(raw) == tag['sha256'] and hashlib.md5(raw).hexdigest() == tag['expected_md5']
SEEN[tag['path']] = digest(raw)
text = raw.decode('ascii')
assert text.startswith('MNI Tag Point File') and re.search(r'Volumes\s*=\s*2\s*;', text)
body = re.split(r'Points\s*=', text, maxsplit=1)[1].strip()
assert body.endswith(';')
rows = []
for line in body[:-1].splitlines():
    tokens = shlex.split(line, comments=False)
    if tokens:
        assert len(tokens) in (6, 7, 9, 10)
        rows.append([float(token) for token in tokens[:6]])
rows = np.asarray(rows, dtype=np.float64)
assert rows.shape == (19, 6) and np.isfinite(rows).all()
x, y = rows[:, :3].copy(), rows[:, 3:].copy()
assert len(np.unique(x, axis=0)) == 19
chosen = [0]
for _ in range(5):
    distances = [min(math.fsum((float(a)-float(b))**2 for a, b in zip(point, x[j])) for j in chosen) if i not in chosen else -1 for i, point in enumerate(x)]
    chosen.append(max(range(19), key=lambda i: (distances[i], -i)))
assert [i+1 for i in chosen] == B
singular = np.linalg.svd(x[chosen]-x[chosen].mean(0), compute_uv=False)
ratio = float(singular[-1]/singular[0])
source_hash = semantic({'patient': 'RESECT:Case4', 'role': 'before_us_to_during_us', 'source_image': 'sha256:'+manifest['inputs']['before']['sha256'], 'source_world_mm': array_hash(x), 'row_ids': list(range(1,20))})
partition_hash = semantic({'source_hash': source_hash, 'boundary_ids': B, 'validation_ids': V, 'rank_ratio': ratio, 'rule': 'source_fps_six_rank3_v1'})
assert partition_hash == 'sha256:'+manifest['partition_sha256'] == freeze['landmark_freeze']['partition_hash']
observations = field['observations']
assert observations['boundary_ids'] == B and observations['source_ras_mm'] == x[chosen].tolist() and observations['observed_ras_mm'] == y[chosen].tolist()
assert semantic(observations) == field['forward_hash'] == model['forward_hash'] == freeze['landmark_freeze']['forward_hash']
qc_hash = semantic({'source':'sha256:'+manifest['inputs']['before']['sha256'], 'destination':'sha256:'+manifest['inputs']['during']['sha256'], 'source_transform':array_hash(np.eye(4)), 'destination_transform':array_hash(np.eye(4)), 'report':'sha256:'+digest((BASE/'header-qc/frame-qc.json').read_bytes()), 'convention':frame['convention']})
audit_hash = semantic({'patient_group':'RESECT:Case4','role':'before_us_to_during_us','tag_sha256':'sha256:'+tag['sha256'],'source_image_sha256':'sha256:'+manifest['inputs']['before']['sha256'],'destination_image_sha256':'sha256:'+manifest['inputs']['during']['sha256'],'frame_qc':qc_hash})
assert audit_hash == freeze['landmark_freeze']['source_audit_hash']

# Horn's symmetric 4x4 quaternion eigensystem, independent of production Kabsch SVD.
bx, by = x[chosen], y[chosen]
cx, cy = bx.mean(0), by.mean(0)
s = (bx-cx).T @ (by-cy)
trace = float(np.trace(s))
z = np.array([s[1,2]-s[2,1], s[2,0]-s[0,2], s[0,1]-s[1,0]])
horn = np.empty((4,4)); horn[0,0] = trace; horn[0,1:] = horn[1:,0] = z
horn[1:,1:] = s+s.T-trace*np.eye(3)
values, vectors = np.linalg.eigh(horn)
w, qx, qy, qz = vectors[:, -1]
rotation = np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*w), 2*(qx*qz+qy*w)], [2*(qx*qy+qz*w), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*w)], [2*(qx*qz-qy*w), 2*(qy*qz+qx*w), 1-2*(qx*qx+qy*qy)]])
translation = cy-rotation@cx
assert field['rigid']['status'] == 'available'
rotation_error = agreement(rotation, field['rigid']['rotation'])
translation_error = agreement(translation, field['rigid']['translation_mm'])
assert abs(np.linalg.det(rotation)-1) < 1e-12 and np.max(np.abs(rotation.T@rotation-np.eye(3))) < 1e-12


def predict(points):
    idw = []
    for point in points:
        distance = [math.dist(point, anchor) for anchor in bx]
        if 0 in distance:
            motion = by[distance.index(0)]-bx[distance.index(0)]
        else:
            weight = [(min(distance)/d)**2 for d in distance]
            denominator = math.fsum(weight)
            motion = np.array([math.fsum(weight[i]*float(by[i,j]-bx[i,j]) for i in range(6))/denominator for j in range(3)])
        idw.append(point+motion)
    return {'no_shift': np.asarray(points), 'proper_rigid': np.asarray(points)@rotation.T+translation, 'inverse_distance_squared': np.asarray(idw)}


vi = np.asarray(V)-1
predicted = predict(x[vi]); residuals = predict(bx)
metrics = {}; b_metrics = {}; differences = []; errors = {}
for method in METHODS:
    saved = report['methods'][method]
    assert saved['total_landmarks'] == 13 and saved['excluded_counts'] == {} and [r['row_id'] for r in saved['per_landmark']] == V
    assert all(r['status'] == 'supported' for r in saved['per_landmark'])
    e = [math.dist(a,b) for a,b in zip(predicted[method], y[vi])]; errors[method] = e
    differences.append(agreement(predicted[method], [r['predicted_ras_mm'] for r in saved['per_landmark']]))
    differences.append(agreement(e, [r['error_mm'] for r in saved['per_landmark']]))
    metrics[method] = summary(e)
    for support in ['all_supported', 'common_supported']:
        assert saved[support]['count'] == 13
        differences.append(agreement([metrics[method][k] for k in ['rms_mm','median_mm','maximum_mm']], [saved[support][k] for k in ['rms_mm','median_mm','maximum_mm']]))
    be = [math.dist(a,b) for a,b in zip(residuals[method],by)]; b_metrics[method] = summary(be)
    bs = report['B_residuals'][method]
    assert [r['row_id'] for r in bs['per_landmark']] == B and bs['summary']['count'] == 6
    differences.append(agreement(be, [r['error_mm'] for r in bs['per_landmark']]))
    differences.append(agreement([b_metrics[method][k] for k in ['rms_mm','median_mm','maximum_mm']], [bs['summary'][k] for k in ['rms_mm','median_mm','maximum_mm']]))
paired = {}
for a,b in itertools.combinations(METHODS,2):
    delta = [u-v for u,v in zip(errors[a],errors[b])]; key = a+'_minus_'+b; saved = report['paired_differences'][key]
    assert saved['count'] == 13 and [r['row_id'] for r in saved['per_landmark']] == V
    differences.append(agreement(delta,[r['error_difference_mm'] for r in saved['per_landmark']]))
    mean = math.fsum(delta)/13
    differences.append(agreement(mean,saved['mean_error_difference_mm']))
    paired[key] = {'mean_error_difference_mm':mean,'per_landmark':[{'row_id':i,'error_difference_mm':v} for i,v in zip(V,delta)]}
assert report['validation_landmarks'] == 13 and report['common_supported_ids'] == V
assert report['confidence_interval'] is None and report['clinical_injury_probability'] is None and report['physical_action_response_validated'] is False


def coverage(points, header):
    affine = np.asarray(header['selected_affine'])
    inverse = np.linalg.inv(affine)
    voxels = points@inverse[:3,:3].T+inverse[:3,3]
    shape = np.asarray(header['shape'])
    inside = np.all((voxels >= -.5)&(voxels <= shape-.5),axis=1)
    center = np.all((voxels >= 0)&(voxels <= shape-1),axis=1)
    roundtrip = voxels@affine[:3,:3].T+affine[:3,3]
    return {'count':len(points),'inside_full_cell_box':int(inside.sum()),'inside_voxel_center_box':int(center.sum()),'voxel_minimum':voxels.min(0).tolist(),'voxel_maximum':voxels.max(0).tolist(),'roundtrip_max_mm':float(np.max(np.abs(roundtrip-points)))}


coverage_report = {}
for name, points, image in [('source',x,'before'),('observed_destination',y,'during')]:
    coverage_report[name] = {'world_extent_min_mm':points.min(0).tolist(),'world_extent_max_mm':points.max(0).tolist(),'world_extent_size_mm':np.ptp(points,axis=0).tolist(),'all19':coverage(points,frame['headers'][image]),'B6':coverage(points[chosen],frame['headers'][image]),'V13':coverage(points[vi],frame['headers'][image])}
for name,pin in SEEN.items():
    assert digest((ROOT/name).read_bytes()) == pin
result = {'schema':'resect-case4-sparse-independent-evaluation-v1','status':'PASS_saved_fixed_sparse_comparison_verified','code':{'path':str(Path(__file__).relative_to(ROOT)),'sha256':digest(Path(__file__).read_bytes())},'source_commit':COMMIT,'all_bound_sha256':SEEN,'runtime':runtime,'case':'RESECT:Case4','role':'DEVELOPMENT','partition_hash':partition_hash,'B_ids':B,'V_ids':V,'B_source_rank_ratio':ratio,'stage_observed_seconds':{p:s['supervision']['elapsed_seconds'] for p,s in stages.items()},'stage_observed_worker_peak_rss_bytes':{p:s['result']['worker_peak_rss_bytes'] for p,s in stages.items()},'chronology_file_mtimes':chronology,'chronology_basis':'Release/freeze hash graph, authenticated unchanged code ordering and saved separate-stage times; file mtimes corroborate, not hostile-host proof.','rigid_reconstruction':{'method':'Horn quaternion maximum-eigenvalue proper rotation from fixed six B','rotation_max_difference':rotation_error,'translation_max_difference_mm':translation_error,'largest_eigenvalue_gap_mm2':float(values[-1]-values[-2])},'V_metrics':metrics,'V_per_landmark':[{'row_id':i,**{m:errors[m][j] for m in METHODS}} for j,i in enumerate(V)],'paired_errors':paired,'B_conditioning_metrics_not_validation':b_metrics,'primary_rigid_minus_no_shift_RMS_mm':metrics['proper_rigid']['rms_mm']-metrics['no_shift']['rms_mm'],'maximum_saved_numeric_disagreement_mm':max(differences),'audit_float_comparison_limit_mm':1e-10,'coverage':coverage_report,'originals':{'tag_bytes':len(raw),'tag_sha256':digest(raw),'tag_MD5':hashlib.md5(raw).hexdigest(),'US_fixity_source':'Previously authenticated header review; no original images reopened in this audit.'},'review_elapsed_seconds':time.monotonic()-START,'scope':{'original_patient_paths_opened':sorted(set(accessed)),'image_reads':0,'MRI_reads':0,'native_solver_calls':0,'production_fit_or_evaluate_calls':0,'training_calls':0,'tracked_writes':0},'limitations':['Conditional sparse observed correspondence from six intraoperative observations, not preoperative-only or dense tissue prediction.','All13V retained; same-case points are not independent patients and no confidence interval or clinical threshold is inferred.','Inside image extent is not observed tissue/cavity/clearance coverage or validated anatomy.','Case4 V is consumed for future tuned/selected methods; fixed IDW is reported without V-driven promotion.','Earlier failed cases remain failed; this does not complete full goal4A integration or validate forces/safety.']}
with (OUT/'verification.json').open('x') as handle:
    json.dump(result,handle,sort_keys=True,indent=2,allow_nan=False);handle.write('\n')
print(json.dumps({'receipt_sha256':digest((OUT/'verification.json').read_bytes()),'metrics':metrics,'primary_rigid_minus_no_shift_RMS_mm':result['primary_rigid_minus_no_shift_RMS_mm'],'max_numeric_difference_mm':max(differences),'coverage':coverage_report,'review_seconds':result['review_elapsed_seconds']}))
