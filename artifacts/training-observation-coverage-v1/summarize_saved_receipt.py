"""Summarize a terminal saved JSON receipt without loading patient arrays."""
from pathlib import Path
import argparse
import hashlib
import json

SUBJECTS = ('sub-PAT05', 'sub-PAT16', 'sub-PAT20', 'sub-PAT22', 'sub-PAT25', 'sub-PAT28')
VIEW_FIELDS = ('shape', 'value_dtype', 'channel_available', 'coverage_fraction_range',
               'nominal_mass_mm3', 'nominal_mass_fraction', 'nominal_mass_omitted_mm3',
               'nominal_positive_cells', 'roundtrip_max_error_mm', 'fingerprint')


def summarize(record):
    if record['version'] != 'fixed-training-observation-coverage-v1':
        raise ValueError('Unexpected saved study')
    if record['cohort_denominator'] != 6 or set(record['subjects']) != set(SUBJECTS):
        raise ValueError('All six fixed TRAIN rows must remain in the report')
    if record['status'] not in ('complete', 'incomplete'):
        raise ValueError('Wait for an explicitly terminal worker receipt')
    rows = []
    for subject in SUBJECTS:
        saved = record['subjects'][subject]
        row = {key: saved[key] for key in ('status', 'new_task_attempted', 'reason', 'failure',
               'historical_preparation_sha256', 'historical_coverage', 'initial_preparation_seconds',
               'initial_previews') if key in saved}
        row.update(subject=subject, representation=None)
        representation = saved.get('representation')
        if representation is not None:
            coverage = representation['coverage']
            row['representation'] = {
                'status': representation['status'],
                'source_nominal_mass_mm3': coverage['source_nominal_mass_mm3'],
                'views': {name: {key: view[key] for key in VIEW_FIELDS}
                          for name, view in coverage['views'].items()},
                'target_local_clipped_fraction': coverage['preparation_report']['target_local_clipped_fraction'],
                'visibility_summary': coverage['visibility_summary'],
                'emitted_actions': coverage['emitted_actions'],
                'accepted_actions': coverage['accepted_actions'],
                'global_extent_max_error_mm': coverage['global_extent_max_error_mm'],
                'coarse_nominal_mass_absolute_error_mm3': coverage['coarse_nominal_mass_absolute_error_mm3'],
                'coarse_nominal_mass_relative_error': coverage['coarse_nominal_mass_relative_error'],
                'physical_catalog_unchanged': representation['physical_catalog_unchanged'],
                'task_invariants_equal': representation['task_before'] == representation['task_after'],
                'timings_seconds': {key: representation[key] for key in ('source_preparation_seconds',
                                      'view_construction_seconds', 'coverage_diagnostic_seconds')},
                'peak_rss_bytes': representation['peak_rss_bytes']}
        rows.append(row)
    return {'schema': 'training-observation-coverage-summary-v1',
            'scope': 'Saved representation coverage only; no learning, cuts, search, or policy evaluation.',
            'interpretation': 'Shaft centerline visibility does not establish instrument-volume visibility or clearance. '
                              'Center interpolation domains and full-cell extents remain distinct. '
                              'Coarse nominal mass is a cell-average integral, not a count of native target cells. '
                              'The local-plus-global union is not a learned fusion or a clinical accuracy result.',
            'recorded_status': record['status'], 'settings': record['settings'],
            'channel_order': ['structural_intensity', 'nominal_tissue', 'nominal_target',
                              'observed_cavity', 'nominal_motor', 'nominal_language'],
            'cohort_denominator': 6, 'subjects': rows,
            'worker_totals': {key: record.get(key) for key in ('new_task_attempts', 'new_representation_completions',
                'historical_support_blocks', 'total_previews', 'executed_transitions', 'policy_forwards',
                'optimizer_updates', 'elapsed_seconds', 'peak_rss_bytes', 'failure')},
            'preservation': {key: record.get(key) for key in ('bundle_bytes_verified_before',
                'bundle_bytes_unchanged', 'sources_unchanged', 'original_records_unchanged')},
            'clinical_deficit_probability': None, 'independent_audit': 'pending separate saved-only audit'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    raw = args.receipt.read_bytes()
    summary = summarize(json.loads(raw))
    summary['receipt_sha256'] = hashlib.sha256(raw).hexdigest()
    summary['receipt_path'] = str(args.receipt)
    with args.output.open('x') as stream:
        stream.write(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')
