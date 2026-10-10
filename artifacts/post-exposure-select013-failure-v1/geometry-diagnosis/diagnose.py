"""Inspect the failed fixed aperture using only its two permitted public masks.

No planning, policy, material modification, alternative access or removal.
Independent scalar reconstruction of the existing axis0 seed rule.
"""
from pathlib import Path
import hashlib
import json
import resource
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
t0 = time.monotonic()
manifest_path = ROOT/'build/remind-select-public-preparation-v1/public-select-inputs-v1/ReMIND-013-public-inputs-v1.json'
raw = manifest_path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '563963e7cf6ae7c1dff9f163db3208bf1af89b5611bc20e76a84c442a2b407a8'
manifest = json.loads(raw)
assert manifest['patient_id'] == 'ReMIND-013' and manifest['role'] == 'SELECT'
assert manifest['public_support_domain_fully_covered'] is True
pins = {}
arrays = []
for key in ('supplied_support', 'supplied_whole_tumor'):
    row = manifest['input_files'][key]
    path = Path(row['path'])
    assert path.is_relative_to(ROOT) and not path.is_symlink()
    assert path.stat().st_size == row['bytes'] < 3*1024**2
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    assert a.shape == tuple(manifest['shape_xyz']) and a.dtype == np.uint8
    assert np.isin(a, (0, 1)).all()
    arrays.append(a)
    pins[key] = row
S, T = arrays
O = (S != 0) | (T != 0)
center = np.argwhere(T).mean(axis=0)
extent = np.flatnonzero(np.any(S != 0, axis=(1, 2)))
positive_side = center[0] >= (extent[0]+extent[-1])/2
sign = -1 if positive_side else 1
transverse = np.rint(center[1:]).astype(int)
spacing = np.linalg.norm(np.asarray(manifest['affine_ras_mm'])[:3, :3], axis=0)
yy, zz = np.ogrid[:O.shape[1], :O.shape[2]]
dy = np.abs((yy-transverse[0])*spacing[1])
dz = np.abs((zz-transverse[1])*spacing[2])
near2 = np.maximum(dy-spacing[1]/2, 0)**2 + np.maximum(dz-spacing[2]/2, 0)**2
footprint = near2 <= (6.+1e-8)**2
occupied = np.flatnonzero(np.any(O & footprint[None, :, :], axis=(1, 2)))
face = float(occupied[0]-.5 if sign > 0 else occupied[-1]+.5)
depth = sign*(np.arange(O.shape[0])-face)*spacing[0]
lo, hi = depth-spacing[0]/2, depth+spacing[0]/2
proximal = (lo < -1e-8) & (np.abs(hi) <= 1e-8)
inside = (dy+spacing[1]/2)**2+(dz+spacing[2]/2)**2 < (6.-1e-8)**2
K = ~O & proximal[:, None, None] & inside[None, :, :]
assert not K.any()
row = dict(scope='diagnosis_of_failed_FIXED_SELECT013_start_not_a_new_route',
    manifest_sha256=hashlib.sha256(raw).hexdigest(), inputs=pins,
    shape=list(O.shape), spacing_mm=spacing.tolist(), support_extent_axis0=[int(extent[0]), int(extent[-1])],
    target_centroid_source_voxels=center.tolist(), transverse=transverse.tolist(), inward_axis0_sign=sign,
    disc_occupied_axis0_extent=[int(occupied[0]), int(occupied[-1])], outer_face_source_index=face,
    outer_face_at_image_boundary=face in (-.5, O.shape[0]-.5),
    proximal_slab_indices=np.flatnonzero(proximal).tolist(), proximal_slab_count=int(proximal.sum()),
    inside_disc_whole_cell_count=int(inside.sum()), seed_cells=int(K.sum()),
    source_support_cells_in_boundary_disc=int(np.count_nonzero((S[occupied[0] if sign > 0 else occupied[-1]] != 0) & footprint)),
    supplied_target_cells_in_boundary_disc=int(np.count_nonzero((T[occupied[0] if sign > 0 else occupied[-1]] != 0) & footprint)),
    min_absolute_distal_face_distance_mm=float(np.min(np.abs(hi))),
    existing_source_crop_positive_loss=manifest['public_label_resampling']['cerebrum']['saved_volume_summary']['source_positive_centres_cropped_from_bound_execution'],
    full_supplied_target_voxels=int(np.count_nonzero(T)), native_previews=0, policy_forwards=0,
    optimizer_updates=0, simulated_material_changes=0,
    conclusion='The fixed outer face has no preceding cell slab inside this cropped planning grid. Whole-cell aperture footprint exists. Do not fabricate known free space or change the original failed result.',
    elapsed_seconds=time.monotonic()-t0, process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
assert row['outer_face_at_image_boundary'] and row['proximal_slab_count']==0 and row['inside_disc_whole_cell_count']>0
for record in pins.values():
    assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()==record['sha256']
with (HERE/'result.json').open('x') as stream:
    json.dump(row, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({k:v for k,v in row.items() if k not in ('inputs',)},indent=2))
