"""Reproduce the diagnosed depth drift from public saved JSON; no array/model imports."""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
paths = {
    'public_source_manifest':'build/remind-planning-qc-v1/ReMIND-008-public-inputs-v2.json',
    'grid_reconciliation':'build/cross-patient-planning-v1/ReMIND-008-horizon24-search-v2/baseline-source/admitted-public-bindings.json',
    'access_derivation':'build/cross-patient-planning-v1/ReMIND-008-horizon24-search-v2/baseline-source/public-task-derivation.json',
    'recorded_cap120_depth':'build/cross-patient-planning-v1/ReMIND-008-footprint-cap120-search-v1/retained-progress.json',
    'recorded_cap96_depth':'build/cross-patient-planning-v1/ReMIND-008-opening-volume-retention-search-v1/retained-progress.json',
}
records = {key:json.loads((ROOT/path).read_text()) for key,path in paths.items()}
grid = records['grid_reconciliation']['native_grid_reconciliation']
original, native = grid['original_affine_ras_mm'], grid['derived_affine_ras_mm']
assert original == records['public_source_manifest']['affine_ras_mm']
access_index = records['access_derivation']['access_index']
dot = lambda a,b:sum(x*y for x,y in zip(a,b))
column = lambda affine,j:[affine[i][j] for i in range(3)]
spacing = math.sqrt(dot(column(original,0),column(original,0)))
normal = [-value/spacing for value in column(original,0)]
center = [dot(original[i][:3],access_index)+original[i][3] for i in range(3)]
coefficients = [dot(column(native,j),normal) for j in range(3)]
rows = []
for key, step, cell in [('recorded_cap120_depth',4,[240,208,118]),
                        ('recorded_cap96_depth',5,[240,206,118])]:
    layer = records[key]['new'][step-1]
    saved = next(row for row in layer['prefixes'] if row['retention_reason']=='opening')
    projected = dot([dot(native[i][:3],cell)+native[i][3]-center[i] for i in range(3)],normal)
    reported = saved['deepest_newly_removed_actor_covered_cell_center_mm']
    assert abs(projected-reported) < 1e-13
    rows.append({'native_cell':cell,'signed_axial_layer':-cell[0],
        'projected_depth_mm':projected,'saved_depth_mm':reported,
        'saved_source':key,'saved_step':step,'absolute_reproduction_error_mm':abs(projected-reported)})
evidence = {
    'inputs':{key:{'path':path,'sha256':hashlib.sha256((ROOT/path).read_bytes()).hexdigest()} for key,path in paths.items()},
    'original_affine_ras_mm':original,'derived_native_affine_ras_mm':native,
    'access_index':access_index,'access_center_ras_mm':center,'normal_from_original_source_axis':normal,
    'depth_coefficients_mm_per_native_index':coefficients,'cells':rows,
    'saved_difference_mm':rows[1]['saved_depth_mm']-rows[0]['saved_depth_mm'],
    'predicted_difference_mm':-2*coefficients[1],
    'existing_grid_reconciliation_max_corner_displacement_mm':grid['maximum_corner_displacement_mm'],
    'existing_grid_reconciliation_allowed_displacement_mm':grid['maximum_allowed_corner_displacement_mm'],
    'finding':'Same integer native axial layer. Reconciled native grid projected on original-grid aperture normal has tiny transverse coefficients; exact-float depth ordering suppresses same-layer volume ties.',
    'limits':'Metadata projection reproduces recorded maxima; it does not independently authenticate the unexported removed-cell history or establish a better counterfactual plan. No patient array/model/native/geometry execution.',
}
(HERE/'numerical-cause.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
print('Same-layer numerical cause reproduced from saved public JSON.')
