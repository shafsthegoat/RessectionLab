#!/usr/bin/env python3
"""Check saved diagnostic records/source bytes; never load patient images."""
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent


def main():
    bindings = {}

    def payload(path):
        raw = path.read_bytes()
        bindings[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return raw

    def read(path):
        return json.loads(payload(path))

    declaration_path = FOLDER / 'run-01/declaration-input.json'
    declaration = read(declaration_path)
    declared_hash = bindings[str(declaration_path.relative_to(ROOT))]
    report = read(FOLDER / 'run-01/receipt.json')
    supervisor = read(FOLDER / 'run-01/supervisor.json')
    deviation = read(FOLDER / 'launch-deviation.json')
    preservation = read(FOLDER / 'post-run-source-receipt.json')
    original_ref = declaration['original_preparation']
    original_path = ROOT / original_ref['path']
    original = read(original_path)
    assert bindings[original_ref['path']] == original_ref['sha256'] == report['original_preparation_sha256']
    assert original['subject'] == declaration['subject'] == report['subject'] == 'sub-PAT25'
    assert original['role'] == declaration['role'] == 'TRAIN'
    assert original['status'] == 'prepared'
    assert original['binding_hash'] == original_ref['binding_hash']
    original_binding = json.dumps(original['binding'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    assert 'sha256:' + hashlib.sha256(original_binding).hexdigest() == original['binding_hash']
    assert declaration['support_acknowledgment'] == original['binding']['member']['research_support_acknowledgment']
    assert declaration['common_task'] == original['binding']['common_task']
    assert declaration['six_exits'] == original['binding']['member']['access_derivation']['six_axis_exit_distances_mm']
    assert not declaration['clinical_use_permitted']

    assert deviation['manifest_sha256'] == supervisor['declaration_sha256'] == preservation['manifest_sha256'] == declared_hash
    for field in ('pre_run_source_commit_completed', 'pre_run_source_archive_created', 'pre_run_release_file_created'):
        assert deviation[field] is False
    assert preservation['archived_before_run'] is False
    assert preservation['launch_deviation_sha256'] == bindings[str((FOLDER / 'launch-deviation.json').relative_to(ROOT))]
    sources = declaration['source_sha256']
    assert len(sources) == deviation['bound_source_count'] == preservation['source_files'] == 61
    assert sources['scripts/diagnose_pat25_access.py'] == deviation['reviewed_script_sha256']
    for name, expected in sources.items():
        assert hashlib.sha256(payload(ROOT / name)).hexdigest() == expected
    archive_path = ROOT / preservation['archive']
    assert hashlib.sha256(payload(archive_path)).hexdigest() == preservation['sha256']
    with tarfile.open(archive_path) as archive:
        files = [m for m in archive.getmembers() if m.isfile()]
        names = [m.name for m in files]
        assert len(names) == len(set(names)) == 62
        for name, expected in sources.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected
        manifest_name, = set(names) - set(sources)
        assert hashlib.sha256(archive.extractfile(manifest_name).read()).hexdigest() == declared_hash

    settings = declaration['settings']
    assert report['settings'] == settings
    assert report['status'] == supervisor['status'] == 'complete' and supervisor['returncode'] == 0
    assert not supervisor['timed_out'] and not supervisor['automatic_retry']
    assert report['elapsed_seconds'] < settings['cooperative_seconds']
    assert supervisor['seconds'] < settings['whole_worker_envelope_seconds']
    assert report['peak_rss_bytes'] < settings['max_rss_bytes']
    assert supervisor['sampled_peak_rss_bytes'] < settings['max_rss_bytes']
    assert report['commits'] == report['executed_transitions'] == report['optimizer_updates'] == 0
    assert not report['replacement_access_selected'] and report['clinical_deficit_probability'] is None
    assert report['all_six_initial_inventories_complete'] and len(report['exits']) == 6
    exit_summaries, selected_count, total_previews = [], 0, 0
    for index, (row, saved_exit) in enumerate(zip(report['exits'], declaration['six_exits'])):
        exit_record = row['exit']
        assert row['status'] == 'complete' and exit_record['exit_index'] == index
        assert (exit_record['axis'], exit_record['outward_sign']) == (index // 2, -1 if index % 2 == 0 else 1)
        for key, value in saved_exit.items():
            assert exit_record[key] == value
        assert row['commits'] == row['executed_transitions'] == 0
        assert row['reward'] == declaration['common_task']['objective']
        assert row['initial_state_hash'] == 'sha256:' + hashlib.sha256((row['native_config_hash'] + ':initial').encode()).hexdigest()
        inventory, trace = row['initial_inventory'], row['trace']
        assert inventory['complete'] and inventory['ledger_complete']
        assert inventory['declared_slots'] == settings['max_declared_slots_per_exit'] == 78
        assert inventory['omitted_count'] == 0
        assert row['preview_calls'] == len(trace) == inventory['emitted_count']
        assert inventory['accepted_count'] == sum(item['feasible'] for item in trace)
        assert inventory['rejected_count'] == sum(not item['feasible'] for item in trace)
        assert inventory['source_hash'] == row['source_hash']
        assert inventory['decision_model_hash'] == row['decision_model_hash']
        reasons = Counter(item['reason'] for item in trace)
        affine = row['proposal_coverage']['native_physical_affine_ras_mm']
        for emitted, item in zip(inventory['emitted'], trace):
            for key in ('action_id', 'tool_id', 'entry_mm', 'tip_mm', 'feasible', 'reason'):
                assert emitted[key] == item[key]
            assert item['source_state_hash'] == row['initial_state_hash']
            if item['reason'] == 'SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE':
                assert not item['feasible'] and item['preview_removed_cells'] == 0
                assert item['blocked_cell_count'] > 0
                lo, hi = item['blocked_cell_index_bounds']
                assert all(type(a) is int and type(b) is int and a <= b for a, b in zip(lo, hi))
                assert item['blocked_cell_count'] <= math.prod(b - a + 1 for a, b in zip(lo, hi))
                corners = itertools.product(*[(a - .5, b + .5) for a, b in zip(lo, hi)])
                physical = [[sum(affine[axis][j] * corner[j] for j in range(3)) + affine[axis][3]
                             for axis in range(3)] for corner in corners]
                bounds = [[operation(p[axis] for p in physical) for axis in range(3)] for operation in (min, max)]
                assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                           for aa, bb in zip(bounds, item['blocked_cell_ras_aabb_mm']) for a, b in zip(aa, bb))
            elif item['feasible']:
                assert item['reason'] == 'NATIVE_CONNECTED_STROKE' and item['preview_removed_cells'] > 0
                assert item['failure_tip_mm'] is None and item['blocked_cell_count'] is None
        if exit_record['selected_original']:
            selected_count += 1
            assert row['original_inventory_exactly_reproduced']
            assert inventory == original['initial_inventory']
            assert row['decision_model_hash'] == original['binding']['decision_model_hash']
            assert exit_record['access'] == original['binding']['member']['access']
            assert inventory['accepted_count'] == 0
        actor = row['actor_coverage']
        assert actor['nominal_target_positive_voxels_total'] == original['coverage']['full_target_source_cells']
        assert actor['crop_shape'] == [64, 64, 64]
        total_previews += len(trace)
        exit_summaries.append({'exit_index': index, 'axis': exit_record['axis'], 'outward_sign': exit_record['outward_sign'],
                              'original_selected': exit_record['selected_original'], 'accepted_previews': inventory['accepted_count'],
                              'emitted_previews': len(trace), 'reasons': dict(reasons),
                              'failure_step_counts': dict(Counter(str(item['failure_step_index']) for item in trace if not item['feasible'])),
                              'target_cells_visible_to_actor': actor['nominal_target_positive_voxels_in_crop'],
                              'target_cells_total': actor['nominal_target_positive_voxels_total'],
                              'recorded_outside_neighbor': row['exterior_neighbor']})
    assert selected_count == 1 and total_previews == report['total_previews'] <= settings['max_total_previews']
    for name, expected in bindings.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    result = {'status': 'passed', 'scope': 'Saved-record/source closure check; no patient decode or preview re-execution.',
              'input_sha256': bindings, 'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_files_verified': len(sources), 'complete_exits_checked': 6, 'preview_records_checked': total_previews,
              'exit_summaries': exit_summaries, 'original_selected_inventory_exact': True,
              'executed_transitions': 0, 'replacement_access_selected': False,
              'provenance': {'classification': 'prospective-source-hash-bound working-tree development diagnostic',
                             'pre_run_commit': False, 'pre_run_archive': False, 'pre_run_release_file': False,
                             'post_run_archive_verified': True, 'launch_head': deviation['git_head_at_launch'],
                             'deviation_preserved': deviation['cause']},
              'interpretation': 'The original local exit has no accepted initial actions; other fixed hypothetical exits admit previews. Crop coverage differs, including two exits with zero target visibility. No cuts, complete route evaluation, policy improvement, replacement-access approval or clinical accessibility is established.',
              'limits': ['Recorded exterior-free membership and blocked-cell sets were not recomputed from images.',
                         'Start/end/current source binding and a post-run archive do not establish a precommitted immutable launch.',
                         'Supervisor sampled RSS can miss transient peaks; wall time is not an isolated-machine benchmark.']}
    (FOLDER / 'post-run-check.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(f'61 source files, six complete exits and {total_previews} saved previews checked; launch deviation preserved')


if __name__ == '__main__':
    main()
