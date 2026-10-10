"""Aggregate the fixed saved JSON measurement; never open masks or run geometry."""
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
BASE = ROOT/'build/remind-aperture-mask-diagnostic-v1'
OUT = Path(__file__).resolve().parent
RESULT_SHA = 'c26cb2d0531133a19b7c64a3cfbb9cba10efaf07e7a760175bc2c609fce1dbf4'
BINS = ('wholly_proximal', 'straddling_or_touching_aperture', 'wholly_inward')
LABELS = ('raw_S', 'full_T', 'derived_U')
PARTS = ('initial_shaft', 'initial_tip', 'swept_shaft', 'swept_tip')
SUBJECTS = ('ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045')

def read_json(path, expected=None):
    assert path.suffix == '.json'
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        assert digest == expected, str(path)
    return json.loads(raw), {'path': str(path.relative_to(ROOT)), 'sha256': digest, 'bytes': len(raw)}

result, result_binding = read_json(BASE/'attempt-01/result.json', RESULT_SHA)
receipt, receipt_binding = read_json(BASE/'attempt-01.supervision/receipt.json')
assert receipt['result_sha256'] == RESULT_SHA
assert receipt['status'] == 'complete' and receipt['exit_code'] == 0
assert receipt['remaining_owned_pids'] == [] and receipt['worker_termination_confirmed']
release, release_binding = read_json(BASE/'root-release.json', result['release_sha256'])
source_index, source_binding = read_json(BASE/'source-index.json', receipt['source_index']['sha256'])
assert tuple(r['patient_id'] for r in result['cases']) == SUBJECTS
rows = []
for record in result['cases']:
    case, binding = read_json(BASE/'attempt-01'/record['result_file'], record['result_sha256'])
    assert case['patient_id'] == record['patient_id'] and case['role'] == 'TRAIN'
    assert case['initial_exposure'] == record['initial_exposure']
    assert case['raw_S_full_T_U_global_depth_counts'] == record['global_depth_counts']
    queries = case['queries']
    assert len(queries) == record['geometric_queries'] == record['saved_motions']*4
    by_part = {p: [q for q in queries if q['part'] == p] for p in PARTS}
    ids = {q['proposal_id'] for q in queries}
    assert len(ids) == record['saved_motions']
    assert all(len(qs) == len(ids) and {q['proposal_id'] for q in qs} == ids for qs in by_part.values())
    assert all(q['saved_reason'] == 'UNKNOWN_DOMAIN:FORBIDDEN_COLLISION' for q in queries)
    aggregate = {}
    for part, qs in by_part.items():
        aggregate[part] = {}
        for label in LABELS:
            for q in qs:
                hit = q['intersections'][label]
                assert hit['count'] == sum(hit['by_full_cell_depth'][b]['count'] for b in BINS)
            aggregate[part][label] = {
                'motions_with_any_hit': sum(q['intersections'][label]['count'] > 0 for q in qs),
                'motions_with_hit_by_full_cell_depth': {
                    b: sum(q['intersections'][label]['by_full_cell_depth'][b]['count'] > 0 for q in qs)
                    for b in BINS},
            }
    initial = by_part['initial_shaft']
    exposure = case['initial_exposure']
    global_counts = case['raw_S_full_T_U_global_depth_counts']
    rows.append({
        'patient_id': case['patient_id'], 'role': 'TRAIN', 'saved_motions': len(ids),
        'saved_result': binding, 'aperture': case['aperture'],
        'initial_shaft_motions_without_raw_S_or_full_T_contact': sum(
            not q['intersections']['raw_S']['count'] and not q['intersections']['full_T']['count'] for q in initial),
        'motions_with_intersections': aggregate,
        'global_source_positive_and_unknown_cells': {
            label: {'count': global_counts[label]['count'],
                    'by_full_cell_depth': {b: global_counts[label]['by_full_cell_depth'][b]['count'] for b in BINS}}
            for label in LABELS},
        'initial_exposure': {
            k: exposure[k] for k in ('boundary_cell_count', 'boundary_U_count', 'boundary_known_zero_seed_count',
               'actual_initial_connected_free_count', 'known_source_zero_component_count',
               'geometric_disc_face_candidates', 'candidate_connected_components',
               'candidate_face_adjacent_material_cells', 'candidate_face_adjacent_raw_S_cells',
               'candidate_face_adjacent_full_T_cells', 'not_an_admitted_cavity', 'source_zero_is_physical_air')},
        'hypothetical_K_count': exposure['source_known_zero_candidates']['count'],
        'hypothetical_K_indices_sha256': exposure['source_known_zero_candidates']['indices_sha256'],
        'native_affine_hash': case['native_grid_reconciliation']['derived_affine_hash'],
        'source_affine_hash': case['native_grid_reconciliation']['original_affine_hash'],
        'all_original_conditions_retained_as_failed': True,
    })
assert sum(r['saved_motions'] for r in rows) == 166
assert result['geometric_queries'] == 664 and result['mask_loads'] == 12
assert all(result[k] == 0 for k in ('native_previews', 'transitions', 'search_calls', 'models', 'source_array_writes'))
assert all(r['initial_exposure']['boundary_cell_count'] == r['initial_exposure']['boundary_U_count'] for r in rows)
assert all(r['initial_exposure']['actual_initial_connected_free_count'] == 0 for r in rows)
assert all(r['motions_with_intersections'][p]['derived_U']['motions_with_hit_by_full_cell_depth'][b] == 0
           for r in rows for p in PARTS for b in BINS[1:])
summary = {
    'schema': 'fixed-aperture-saved-mask-diagnostic-summary-v1',
    'analysis_scope': 'saved JSON only; no mask reload, geometry, preview, model or source modification',
    'source_result': result_binding, 'supervisor_receipt': receipt_binding,
    'root_release': release_binding, 'source_index': source_binding,
    'planned_cases': list(SUBJECTS), 'saved_motions': 166, 'measured_geometric_queries': 664,
    'measured_mask_loads': 12, 'analysis_mask_loads': 0, 'native_previews': 0, 'transitions': 0,
    'search_calls': 0, 'models': 0, 'training_admitted': False,
    'supervision': {k: receipt[k] for k in ('elapsed_seconds', 'sampled_peak_rss_bytes', 'exit_code',
        'remaining_owned_pids', 'worker_termination_confirmed', 'sampling_limit')},
    'count_semantics': 'Motion counts mean at least one intersected cell; raw S/T and tool-part/depth categories may overlap. Do not add them as a unique volume or removed-cell denominator.',
    'depth_semantics': 'Entire affine cell signed interval relative to fixed modeled aperture, not anatomical intracranial/extracranial status. Touching is grouped with straddling at1e-8mm; inherited capsule contact tolerance1e-9mm.',
    'domain_semantics': 'S supplied automatic cerebrum; T full supplied manual whole tumor; Ds observed support coverage; modeled O=S|T,D=Ds|T,U=~D. Neither Ds-known zero nor T-derived occupancy is validated physical anatomy.',
    'K_semantics': 'Hypothetical source-covered O-zero cells with distal face touching fixed aperture plane and whole projected footprint inside6mm disc. No seeding, clearance or access admission occurred.',
    'findings': {
        'all_initial_shaft_motions_hit_proximal_U': True,
        'any_U_hit_straddling_or_inward_across_all_parts': False,
        'current_exterior_seed_count_and_connected_free_count_zero_all_cases': True,
        'K_seeding_alone_cannot_resolve_shaft_U_or_known_material_collisions': True,
        'moving_aperture_to_union_boundary_alone_not_shown_sufficient': True,
        'outside_image_tool_geometry': 'unassessed',
        'current_fixed_condition_legal_motion_claim': False,
    },
    'cases': rows,
}
(OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
(OUT/'supervisor-receipt.json').write_bytes((BASE/'attempt-01.supervision/receipt.json').read_bytes())
print(json.dumps({'cases': len(rows), 'motions': 166, 'queries': 664, 'summary_sha256': hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest()}))
