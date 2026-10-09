"""Exact, independently reviewed N12 saved-attempt numerical exception.

Only this one original failed receipt and supplement can be a numerical row.
This is neither a native retry nor generic failed-receipt admission.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import stat

from scripts import mechanics_hbe_v5_n8_one_shot as io
from scripts import mechanics_hbe_v5_n12_saved_replay as replay
from scripts import mechanics_hbe_v5_remaining_one_shot as runner


ROOT = Path(__file__).resolve().parents[1]
DECLARATION = 'manifests/experiments/hbe-v5-n12-exact-supplement-admission-v1.json'
DECLARATION_SHA = '84cd645a10d70c07613e12d9548076336326b8e97510cfe83e39e261ec0eaa5e'
SUPPLEMENT_SHA = '8985a0e23b5d3a3d0029c5d69b48caaa6aff54c0628e40dabd1ea33aa6f3bb24'
REVIEW = {'path': 'artifacts/hbe-v5-n12-saved-replay-v1/INDEPENDENT_REVIEW.md',
          'sha256': '39e832c7ccfb6f941e5e4db584f2849fa64d45cdd101ff7417cbc9db73fd0458'}
SUMMARY = {'path': 'artifacts/hbe-v5-n12-saved-replay-v1/summary.json',
           'sha256': '39d921b5a6ad0db2efa52a18aab6b8e030b302fdc5a5778556eb350e9c374f9e'}
HISTORICAL_REPLAY_SOURCE_PATHS = tuple(dict.fromkeys((
    'scripts/mechanics_hbe_v5_n12_saved_replay.py', *replay.ORIGINAL_SOURCE_PATHS)))
GAP = 'Original post-run source/runtime guards were not recorded after parent packaging failure.'


def _closed_supplement(directory: Path, receipt: dict, *, root: Path) -> dict:
    if {item.name for item in directory.iterdir()} != replay.OUTPUT_NAMES:
        raise ValueError('Supplement output directory changed')
    if runner.active_bytes(directory, replay.CAPS['active_output_bytes']) != 557392:
        raise ValueError('Supplement closed output bytes changed')
    records = {}
    for name in replay.OUTPUT_NAMES - {'receipt.json'}:
        path = io.local(replay.OUTPUT+'/'+name, root=root)
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError('Supplement output special file refused')
        records[name] = runner._regular_output_binding(path, root=root,
            limit=replay.CAPS['active_output_bytes'],
            allow_empty=name == 'readout-console.txt')
    if records != receipt.get('replay_output_bindings'):
        raise ValueError('Supplement output bindings differ from receipt')
    if (records['replay.json']['sha256'] !=
            '960c7a4fdfb37b0823b07f88c19fe280b72583169fc73825a90b0dfe8c86772f'
            or records['replay-work-order.json']['sha256'] !=
               'fd85ee65750f06869dbe83b394f38dc4571417151e2ae782b8ec030d10a6aaa0'
            or records['readout-console.txt']['sha256'] != io.sha(b'')
            or records['readout-console.txt']['bytes'] != 0):
        raise ValueError('Exact reviewed supplemental files differ')
    if io.file_hash(directory/'receipt.json', maximum=8*1024**2) != SUPPLEMENT_SHA:
        raise ValueError('Supplement receipt changed during inventory')
    return records


def verify_exact(*, root: Path = ROOT) -> dict:
    """Rehash both attempts, releases, immutable commits and independent GO."""
    declared = json.loads(io.bound({'path': DECLARATION, 'sha256': DECLARATION_SHA},
                                    DECLARATION, root=root))
    if (declared.get('schema') != 'hbe-v5-n12-exact-supplement-admission-v1'
            or declared.get('status') !=
               'specific_numerical_exception_after_independent_replay_review'
            or declared.get('run_id') != runner.ORDER[1]
            or declared.get('ordinal') != 1
            or declared.get('original_failed_receipt') != {
                'path': runner.receipt_path(1), 'sha256': replay.ORIGINAL_RECEIPT_SHA}
            or declared.get('original_release') != {
                'path': replay.ORIGINAL_RELEASE, 'sha256': replay.ORIGINAL_RELEASE_SHA}
            or declared.get('original_source_commit') != replay.ORIGINAL_SOURCE_COMMIT
            or declared.get('original_closed_output_bytes') != 50501397
            or declared.get('supplement_receipt') != {
                'path': replay.OUTPUT+'/receipt.json', 'sha256': SUPPLEMENT_SHA}
            or declared.get('supplement_source_commit') !=
               'f335975d4ed331eb33fc9806846cc19193037c1e'
            or declared.get('supplement_release') != {
                'path': 'build/hbe-v5-n12-saved-replay-release/release.json',
                'sha256': '250961266042ffde26fb779743bbf669551b65d8b976e57df74cf80ff3018326'}
            or declared.get('supplement_closed_output_bytes') != 557392
            or declared.get('independent_review') != REVIEW
            or declared.get('independent_summary') != SUMMARY
            or declared.get('historical_guard_gap') != GAP
            or declared.get('phase_gates') != {
                'exact_n12_numerical_predecessor': True,
                'exact_n12_native_comparator_input': True,
                'generic_failed_receipt_admission': False, 'native_retry': False,
                'fit': False, 'held_out_torque_access': False,
                'physical_validation': False,
                'patient_or_measured_response_access': False}):
        raise ValueError('Exact N12 exception declaration changed')
    review = io.bound(REVIEW, REVIEW['path'], root=root, maximum=128*1024)
    summary = json.loads(io.bound(SUMMARY, SUMMARY['path'], root=root,
                                  maximum=128*1024))
    if (b'Decision: GO for a separately labeled, saved-output numerical supplement' not in review
            or summary.get('status') != 'supplemental_saved_attempt_numerical_pass_only'
            or summary.get('receipt_sha256') != SUPPLEMENT_SHA
            or summary.get('original_status_preserved') != 'failed_or_incomplete'
            or summary.get('historical_guard_gap') != GAP
            or summary.get('replay_numerical_passed') is not True
            or summary.get('comparator_admitted') is not False
            or summary.get('predecessor_admitted') is not False
            or summary.get('closed_replay_bytes') != 557392):
        raise ValueError('Independent replay review/summary does not support exception')
    preparation = json.loads(io.bound({'path': replay.PREPARATION,
        'sha256': replay.PREPARATION_SHA}, replay.PREPARATION, root=root))
    replay.validate_preparation(preparation)
    original = replay.validate_original(preparation, root=root,
                                         audit_loaded_imports=False)
    directory = io.local(replay.OUTPUT, root=root)
    receipt = json.loads(io.bound({'path': replay.OUTPUT+'/receipt.json',
        'sha256': SUPPLEMENT_SHA}, replay.OUTPUT+'/receipt.json', root=root,
        maximum=8*1024**2))
    replay_release_raw = io.bound(declared['supplement_release'],
        declared['supplement_release']['path'], root=root)
    replay_release = json.loads(replay_release_raw)
    replay.committed_source(declared['supplement_source_commit'],
        replay_release.get('source_bindings'),
        paths=HISTORICAL_REPLAY_SOURCE_PATHS, root=root)
    files = _closed_supplement(directory, receipt, root=root)
    original_receipt = original['receipt']
    original_times = {
        'native_calls': 1,
        'native_wall_seconds': original_receipt['native_stage']['elapsed_seconds'],
        'readout_calls': 1,
        'readout_wall_seconds': original_receipt['readout_stage']['elapsed_seconds'],
        'preparation_wall_seconds': original_receipt['prep_elapsed_seconds'],
        'closed_output_bytes': preparation['closed_attempt_bytes']}
    supplemental_times = {
        'native_calls': 0, 'replay_calls': 1,
        'replay_wall_seconds': receipt.get('readout_stage', {}).get('elapsed_seconds'),
        'preparation_wall_seconds': receipt.get('preparation_elapsed_seconds'),
        'closed_output_bytes': 557392}
    stage = receipt.get('readout_stage', {})
    if (receipt.get('schema') != 'hbe-v5-n12-saved-attempt-supplement-v1'
            or receipt.get('status') != 'supplemental_saved_attempt_numerical_pass_only'
            or receipt.get('run_id') != runner.ORDER[1]
            or receipt.get('original_status_preserved') != 'failed_or_incomplete'
            or receipt.get('original_receipt_sha256') != replay.ORIGINAL_RECEIPT_SHA
            or receipt.get('original_release_sha256') != replay.ORIGINAL_RELEASE_SHA
            or receipt.get('original_source_commit') != replay.ORIGINAL_SOURCE_COMMIT
            or receipt.get('original_closed_attempt_bytes') != 50501397
            or receipt.get('original_file_bindings') != original['inventory']
            or receipt.get('historical_guard_gap') != GAP
            or receipt.get('current_source_commit') !=
               declared['supplement_source_commit']
            or receipt.get('release_sha256') != declared['supplement_release']['sha256']
            or replay_release.get('source_commit') !=
               declared['supplement_source_commit']
            or replay_release.get('output_directory') != replay.OUTPUT
            or replay_release.get('caps') != replay.CAPS
            or replay_release.get('status') != 'root_released_one_saved_output_replay'
            or receipt.get('current_source_hashes_before') != {
                name: binding['sha256'] for name, binding in
                replay_release['source_bindings'].items()}
            or receipt.get('current_source_hashes_after') !=
               receipt.get('current_source_hashes_before')
            or receipt.get('current_runtime_profile_rechecked') is not True
            or receipt.get('current_runtime_profile_rechecked_after') is not True
            or receipt.get('native_calls_attempted') != 0
            or receipt.get('replay_calls_attempted') != 1
            or receipt.get('readout_calls_attempted') != 1
            or receipt.get('no_retry') is not True
            or receipt.get('replay_numerical_passed') is not True
            or receipt.get('frame_count') != 61
            or receipt.get('predecessor_admitted') is not False
            or receipt.get('comparator_admitted') is not False
            or receipt.get('physical_validation_pass') is not None
            or receipt.get('measured_response_accessed') is not False
            or receipt.get('patient_data_accessed') is not False
            or stage.get('status') != 'completed_within_caps'
            or stage.get('exit_code') != 0 or stage.get('kill_reason') is not None
            or stage.get('wall_cap_seconds') != replay.CAPS['replay_wall_seconds']
            or stage.get('sampled_process_group_rss_cap_bytes') !=
               replay.CAPS['sampled_process_group_rss_bytes']
            or stage.get('active_output_cap_bytes') != replay.CAPS['active_output_bytes']
            or original_times != declared.get('original_accounting')
            or supplemental_times != declared.get('supplement_overhead')):
        raise ValueError('Specific supplemental receipt or stage/accounting differs')
    if (any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in (original_times['native_wall_seconds'],
                          original_times['readout_wall_seconds'],
                          original_times['preparation_wall_seconds'],
                          supplemental_times['replay_wall_seconds'],
                          supplemental_times['preparation_wall_seconds']))
            or not supplemental_times['replay_wall_seconds'] < replay.CAPS['replay_wall_seconds']
            or not supplemental_times['preparation_wall_seconds'] <
               replay.CAPS['preparation_wall_seconds']
            or summary.get('replay_wall_seconds') != supplemental_times['replay_wall_seconds']
            or summary.get('preparation_elapsed_seconds') !=
               supplemental_times['preparation_wall_seconds']):
        raise ValueError('Original or supplemental timing/cap differs')
    order = json.loads((directory/'replay-work-order.json').read_bytes())
    if (order.get('schema') != 'hbe-v5-n12-saved-replay-work-order-v1'
            or order.get('original_receipt_sha256') != replay.ORIGINAL_RECEIPT_SHA
            or order.get('source_commit') != declared['supplement_source_commit']
            or order.get('token_sha256') != receipt.get('replay_token_sha256')
            or order.get('bindings') != original['work_order']['bindings']
            or order.get('adapter_receipt') != original['work_order']['adapter_receipt']
            or files['replay.json']['sha256'] !=
               original['inventory']['readout.json']['sha256']):
        raise ValueError('Exact replay provenance or byte-identical saved JSON differs')
    return {'run_id': runner.ORDER[1], 'original_receipt_sha256':
            replay.ORIGINAL_RECEIPT_SHA, 'supplement_receipt_sha256': SUPPLEMENT_SHA,
            'original_backend_profile': original['release']['backend_profile'],
            'original_runtime_identity': original['release']['runtime_identity'],
            'readout': original['saved_json'],
            'original_accounting': original_times,
            'supplement_overhead': supplemental_times,
            'combined_wall_seconds': sum(original_times[key] for key in
                ('native_wall_seconds', 'readout_wall_seconds', 'preparation_wall_seconds'))
                + supplemental_times['replay_wall_seconds']
                + supplemental_times['preparation_wall_seconds'],
            'combined_output_bytes': original_times['closed_output_bytes'] +
                supplemental_times['closed_output_bytes'],
            'historical_guard_gap': GAP,
            'physical_validation_pass': None,
            'measured_response_accessed': False,
            'patient_data_accessed': False}
