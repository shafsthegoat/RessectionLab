"""Reconcile saved review JSON and affine metadata; do not open array bodies."""
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
ACTUAL = BASE / 'actual-public-qc-v1'
REVIEW = BASE / 'saved-review-preparation-v1'
OUT = BASE / 'saved-review-preservation-v1'
PINS = {}
SUBJECTS = ['ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045']

def read(path):
    raw = path.read_bytes()
    pin = {'path': str(path.relative_to(ROOT)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    PINS[pin['path']] = pin
    return json.loads(raw), pin

def inverse(matrix):
    a = [list(map(float, row)) + [float(i == j) for j in range(4)] for i, row in enumerate(matrix)]
    for c in range(4):
        p = max(range(c, 4), key=lambda i: abs(a[i][c])); a[c], a[p] = a[p], a[c]
        v = a[c][c]; assert abs(v) > 1e-12; a[c] = [x / v for x in a[c]]
        for i in range(4):
            if i != c:
                v = a[i][c]; a[i] = [x - v * y for x, y in zip(a[i], a[c])]
    return [row[4:] for row in a]

def multiply(a, b):
    return [[sum(x * y for x, y in zip(row, column)) for column in zip(*b)] for row in a]

def bounds(geometry, target_affine, cells=False):
    transform = multiply(inverse(target_affine), geometry['affine_xyz_to_ras_mm'])
    corners = itertools.product(*[(-.5, n-.5) if cells else (0, n-1) for n in geometry['shape_xyz']])
    points = [[sum(x * y for x, y in zip(row, [*p, 1])) for row in transform[:3]] for p in corners]
    return {'minimum': [min(p[i] for p in points) for i in range(3)], 'maximum': [max(p[i] for p in points) for i in range(3)]}

terminal, terminal_pin = read(REVIEW / 'actual-saved-review-v1.supervision/receipt.json')
result, result_pin = read(REVIEW / 'actual-saved-review-v1/result.json')
assert terminal['result_sha256'] == result_pin['sha256'] == 'e84b3d6331bd08cd510daa465dfd2f7fde5fab277a52d63da7bed89134349e2a'
assert terminal['status'] == 'complete' and terminal['exit_code'] == 0 and terminal['reaped']
assert not terminal['cleanup_actions'] and not terminal['cleanup_errors'] and not terminal['remaining_owned_pids']
assert result['patient_ids'] == SUBJECTS and result['public_input_manifests'] == [None] * 4
assert not result['original_source_reopened'] and not result['private_reference_loaded'] and not result['training_admitted']
index, index_pin = read(REVIEW / 'actual-saved-review-v1/array-audit/saved-review-index.json')
diagnostic, diagnostic_pin = read(REVIEW / 'actual-saved-review-v1/saved-support-pattern-v1.json')
assert index_pin['sha256'] == result['array_audit_index_sha256']
assert diagnostic_pin['sha256'] == result['support_diagnostic_sha256']
visual, visual_pin = read(BASE / 'root-overlay-review.json')
assert [c['patient_id'] for c in visual['cases']] == SUBJECTS and visual['all_four_retained']
rows = []
for subject, binding, view in zip(SUBJECTS, index['cases'], visual['cases']):
    assert binding['patient_id'] == subject and binding['public_input_manifest'] is None and not binding['training_admitted']
    audit, pin = read(ROOT / binding['saved_review_path'])
    assert pin['sha256'] == binding['saved_review_sha256'] == view['saved_review_sha256']
    crop, crop_pin = read(ACTUAL / (subject + '-crop-mr/conversion-result.json'))
    headers, header_pin = read(ACTUAL / (subject + '-headers/result.json'))
    assert audit['conversion_receipt_sha256'] == crop_pin['sha256'] and audit['header_snapshot_sha256'] == header_pin['sha256']
    assert audit['all_saved_artifact_hashes_verified'] and audit['all_source_domain_voxels_world_checked']
    assert audit['saved_counts']['target'] == crop['public_target_support_relation']['whole_tumor_positive_voxels']
    assert audit['saved_counts']['target_outside_support'] == crop['public_target_support_relation']['whole_tumor_positive_outside_public_support']
    assert len(view['views_inspected']) == 3 and {v['axis'] for v in view['views_inspected']} == {0, 1, 2}
    assert not view['expert_anatomical_accuracy_validated'] and not view['clinical_planning_admitted']
    for v in view['views_inspected']:
        assert v['sha256'] == crop['artifacts'][Path(v['path']).name]['sha256']
    geometry = {v['kind']: v['geometry'] for v in headers['series']}; mr = geometry['structural_t1ce']
    labels = {}
    for kind in ['whole_tumor', 'cerebrum']:
        centre = bounds(geometry[kind], mr['affine_xyz_to_ras_mm'])
        cells = bounds(geometry[kind], mr['affine_xyz_to_ras_mm'], True)
        in_mr = all(a >= -.5 and b < n-.5 for a, b, n in zip(centre['minimum'], centre['maximum'], mr['shape_xyz']))
        assert in_mr
        support = bounds(geometry[kind], geometry['cerebrum']['affine_xyz_to_ras_mm'])
        in_support = all(a >= -.5 and b < n-.5 for a, b, n in zip(support['minimum'], support['maximum'], geometry['cerebrum']['shape_xyz']))
        labels[kind] = {'voxel_centre_bounds_in_full_MRI': centre, 'voxel_cell_bounds_in_full_MRI': cells,
                       'entire_source_voxel_centre_box_inside_MRI': in_mr, 'voxel_centre_bounds_in_cerebrum_source_grid': support,
                       'entire_source_voxel_centre_box_inside_cerebrum_domain': in_support,
                       'native_positive_voxels': crop['annotations'][kind]['source_positive_voxels'],
                       'source_positive_centres_outside_current_crop': crop['annotations'][kind]['placement']['source_positive_centres_outside_target_grid']}
    low = [min(v['voxel_cell_bounds_in_full_MRI']['minimum'][i] for v in labels.values()) for i in range(3)]
    high = [max(v['voxel_cell_bounds_in_full_MRI']['maximum'][i] for v in labels.values()) for i in range(3)]
    start = [max(0, math.floor(v+.5)) for v in low]
    stop = [min(n, math.ceil(v+.5)) for v, n in zip(high, mr['shape_xyz'])]
    rows.append({'patient_id': subject, 'role': 'TRAIN', 'saved_review': pin, 'conversion_result': crop_pin,
        'headers': header_pin, 'saved_counts': audit['saved_counts'], 'source_geometry_refit': audit['source_geometry_refit_from_saved_headers'],
        'full_MRI_shape': mr['shape_xyz'], 'current_crop_start': crop['public_crop_start_MR'], 'current_crop_shape': crop['shape_xyz'],
        'current_uniform_inset_MRI_cells_per_face': crop['source_cropping_map']['uniform_inset_MR_cells_per_face'],
        'current_crop_clipped_to_MRI_coverage': crop['source_cropping_map']['clipped_to_MR_coverage'],
        'source_domains': labels, 'outer_annotation_domain_bbox_MRI_start': start, 'outer_annotation_domain_bbox_MRI_stop': stop,
        'outer_bbox_shape_without_roundoff_halo': [b-a for a,b in zip(start,stop)], 'root_visual_observation': view['observation'],
        'full_target_use_held': subject in ['ReMIND-002', 'ReMIND-045'], 'training_admitted': False,
        'interpretation': 'All positive source centres are geometrically inside acquired MRI; losses result from selected crop. Full source-domain centre boxes suffice for this proof without reading labels.'})
OUT.mkdir(exist_ok=False)
proof = {'schema': 'remind-four-TRAIN-saved-review-and-crop-coverage-proof-v1', 'scope': 'Saved JSON hashes and affine corner arithmetic only; no arrays, PNG bodies or original DICOM read.',
    'fixed_denominator': 4, 'terminal': terminal_pin, 'root_visual_note': visual_pin, 'elapsed_seconds': terminal['elapsed_seconds'],
    'sampled_peak_rss_bytes': terminal['sampled_peak_rss_bytes'], 'all_four_arrays_checked_by_bound_worker': True,
    'root_reviewed_12_overlays': True, 'expert_anatomy_validation': False, 'training_admitted': False,
    'rows': rows, 'proof_limit': 'Affine geometry/source-grid containment, not anatomical validity or annotation accuracy. A grid centre inside MRI coverage does not verify tissue identity.',
    'next_change': 'Preserve original crops. Separate conversion may retain a native-MRI outer box covering both public annotation domains, with source-domain masks retained and unknown outside. Do not use the current factory if it assumes full support coverage; no extrapolation, filling or automatic admission.'}
(OUT / 'summary.json').write_text(json.dumps(proof, indent=2, sort_keys=True)+'\n')
(OUT / 'preservation-index.json').write_text(json.dumps({'metadata_hashed': sorted(PINS.values(), key=lambda x:x['path']), 'array_or_image_body_reads': 0, 'original_DICOM_reads': 0}, indent=2)+'\n')
print(json.dumps({'status':'saved_metadata_reconciled', 'output':str(OUT.relative_to(ROOT)), 'all_source_positive_centres_within_acquired_MRI':True, 'full_target_holds':['ReMIND-002','ReMIND-045']}))
