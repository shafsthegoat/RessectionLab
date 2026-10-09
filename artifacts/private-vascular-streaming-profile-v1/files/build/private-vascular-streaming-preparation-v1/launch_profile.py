"""Root-only one-shot adapter to the unchanged reviewed owned-child supervisor."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PREP = 'build/private-vascular-streaming-preparation-v1'
PROFILE = PREP+'/profile_generated.py'
OWNED = 'build/menichetti-structure-launch-preparation-v1/launch.py'
OUTPUT = 'build/private-vascular-streaming-profile-v1'
CACHE = PREP+'/launcher-unused-pycache'
HISTORICAL_COMMIT = '0f7eee3a17b6743fe15549cf8aba396fa4e98a2c'
HELPERS = {
    'scripts/mechanics_hbe_v5_remaining_one_shot.py': '90d855b1e9518793480688afbfb133b586a6ccbe5b974be30b6b40624336216b',
    'scripts/mechanics_hbe_v5_n8_one_shot.py': '82bb01dc2588f750489916c789e7afe85b7a9a6e1ef0fcd5820fe8e70be4f3a0',
    'scripts/febio_runtime.py': '679594d7f3759f5485b9fb868e7e5543ebb242bccd24d112f9ca862bb6d2a01e'}
SOURCES = {
    PROFILE: '1b149489ee2dfb0338c08518fba5835b689de3cace7f696796b08b42e27b4eda',
    PREP+'/streaming_contact.py': 'c85cca298b20190be865e23cb3d0bde3dc142e69f182e93e65c008895cec3d75',
    PREP+'/repository-source-pins.json': '2ced65d6f47c3771919e6a62f9fbbf54593db30f803c70343b798db83513fea8',
    'build/private-vascular-streaming-independent-v1/REPORT.md': '12d26ca020626ebc0f8ae3474161b2a8cf641610e0714afb63fbb721bf78d488',
    OWNED: '2f93c7fd776014417677ed7ea12112a8f237218e76d288ad04297b651816fea2',
    'build/menichetti-structure-launch-independent-v1/REPORT.md': '7b502043e15b8a402c1d02e2bc1338e50cbc201977af6297afd01c5af64731ff',
    **HELPERS}
CAPS = {'attempts': 1, 'worker_wall_seconds': 35, 'cleanup_reserve_seconds': 10,
        'lifecycle_wall_seconds': 45, 'sampled_process_group_rss_bytes': 536870912,
        'aggregate_output_bytes': 4194304, 'profile_json_bytes': 1048576, 'numerical_threads': 1}


def need(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def read_regular(path, cap=4*1024**2):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        need(stat.S_ISREG(info.st_mode) and info.st_size <= cap, 'regular_bounded_file_required')
        raw = stream.read(cap+1)
    need(len(raw) <= cap, 'read_budget')
    return raw


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bound(name, expected):
    raw = read_regular(ROOT/name)
    need(digest(raw) == expected, 'source_changed:'+name)
    return raw


def load_module(name, relative):
    bound(relative, SOURCES[relative])
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def child_release():
    value = {'scope': 'one_generated_256x256x192_streaming_contact_profile', 'root_release': True,
             'source_hashes': {name: SOURCES[PREP+'/'+name] for name in
                 ('streaming_contact.py', 'profile_generated.py', 'repository-source-pins.json')}}
    return (json.dumps(value, indent=2, sort_keys=True)+'\n').encode()


def validate_release(path, expected):
    raw = read_regular(path, 65536)
    need(digest(raw) == expected, 'root_release_hash')
    release = json.loads(raw)
    need(release == {'schema': 'generated-streaming-profile-supervised-release-v1',
                    'execution_released': release.get('execution_released'),
                    'launcher_sha256': digest(read_regular(Path(__file__))),
                    'source_bindings': SOURCES, 'caps': CAPS, 'output_directory': OUTPUT,
                    'child_release_sha256': digest(child_release()), 'patient_or_model_access_permitted': False}
         and type(release['execution_released']) is bool, 'root_release_contract')
    for name, expected_hash in SOURCES.items():
        bound(name, expected_hash)
    profile = load_module('root_streaming_profile_source', PROFILE)
    profile.source_pins()  # Stdlib source checks only; no scientific import.
    for name, expected_hash in HELPERS.items():
        historical = subprocess.run(['/usr/bin/git', 'show', HISTORICAL_COMMIT+':'+name], cwd=ROOT,
            capture_output=True, check=True, timeout=5).stdout
        need(digest(historical) == expected_hash, 'historical_helper_changed')
    return release, profile


def semantic_profile(raw):
    value = json.loads(raw)
    need(value['scope'] == 'one_generated_256x256x192_streaming_contact_profile'
         and value['source_hashes'] == json.loads(child_release())['source_hashes']
         and value['source_bytes_unchanged'] is True and value['prohibited_actions'] == [], 'profile_source_scope')
    inventory = json.loads(bound(PREP+'/repository-source-pins.json', SOURCES[PREP+'/repository-source-pins.json']))
    need(value['repository_sources_verified'] == len(inventory)
         and 'resectionlab.independent_geometry_batch' in value['loaded_repository_origins_before']
         and value['loaded_repository_origins_before'].items() <= value['loaded_repository_origins_after'].items(), 'profile_origin_scope')
    for mapping in (value['loaded_repository_origins_before'], value['loaded_repository_origins_after']):
        for module_name, source_path in mapping.items():
            stem = 'src/'+module_name.replace('.', '/')
            need(module_name == 'resectionlab' or module_name.startswith('resectionlab.'), 'origin_module_namespace')
            need(source_path in inventory and source_path in (stem+'.py', stem+'/__init__.py'), 'origin_manifest_path')
    result = value['result']
    need(result['status'] == 'complete_generated_contact_profile' and result['patient_admission'] is False
         and result['strategy_replay_or_admission_performed'] is False
         and result['grid_shape'] == [256, 256, 192] and result['grid_voxels'] == 12582912
         and result['capsule_count'] == 6 and result['action_count'] == 3
         and result['contact_tolerance_squared_mm2'] == 1e-10, 'profile_not_semantically_complete')
    need(set(result['per_action']) == {'first', 'repeated', 'crossing'}
         and result['per_action']['first'] == result['per_action']['repeated'], 'repeated_action_union')
    need(result['removed_overlap'] == {'status': 'not_evaluated_by_contact_kernel', 'outcomes': None,
         'existing_congruence_requirement_unchanged': True}, 'removal_scope')
    work, budget = result['work'], result['budget']
    expected_work = {'tile_capsule_pairs', 'cell_capsule_pairs', 'tiles_scanned', 'tiles_pruned', 'tiles_evaluated',
                     'reference_sample_calls', 'reference_sampled_cells', 'maximum_tile_cells',
                     'maximum_geometry_batch', 'maximum_coarse_geometry_batch', 'maximum_cell_geometry_batch', 'maximum_action_masks'}
    need(budget == {'tile_edge': 16, 'coarse_batch': 256, 'max_tile_capsule_pairs': 2000000,
                   'max_cell_capsule_pairs': 4000000, 'max_sampled_cells': 2000000, 'wall_seconds': 20.}, 'kernel_caps')
    need(set(work) == expected_work and all(type(n) is int and n >= 0 for n in work.values())
         and work['tiles_scanned'] == 3072 and work['tile_capsule_pairs'] == 18432
         and work['tiles_pruned'] + work['tiles_evaluated'] == work['tiles_scanned']
         and work['cell_capsule_pairs'] <= budget['max_cell_capsule_pairs']
         and work['reference_sampled_cells'] <= budget['max_sampled_cells']
         and work['maximum_tile_cells'] <= 4096 and work['maximum_geometry_batch'] <= 256
         and work['maximum_geometry_batch'] == max(work['maximum_coarse_geometry_batch'], work['maximum_cell_geometry_batch'])
         and work['maximum_action_masks'] <= 3
         and work['reference_sample_calls'] <= work['tiles_evaluated']
         and work['reference_sampled_cells'] <= work['cell_capsule_pairs'], 'profile_work_caps')
    need(result['whole_tool']['touched_reference_cells'] == work['reference_sampled_cells'], 'sampled_union_count')
    for name in ('profile_seconds',):
        need(type(value[name]) in (int, float) and math.isfinite(value[name]) and 0 <= value[name] <= 35, 'profile_time')
    need(type(result['elapsed_seconds']) in (int, float) and math.isfinite(result['elapsed_seconds'])
         and 0 <= result['elapsed_seconds'] <= 20, 'kernel_time')
    for name in ('tracemalloc_current_bytes', 'tracemalloc_peak_bytes', 'process_peak_rss_bytes'):
        need(type(value[name]) is int and 0 <= value[name] <= CAPS['sampled_process_group_rss_bytes'], 'profile_memory')
    need(value['tracemalloc_current_bytes'] <= value['tracemalloc_peak_bytes'], 'trace_memory_partition')
    for record in [result['shaft'], result['tip'], result['whole_tool'], *result['per_action'].values()]:
        need(all(type(record[k]) is int and 0 <= record[k] <= record['touched_reference_cells']
                 for k in ('touched_reference_cells', 'positive_reference_cells', 'unknown_reference_cells'))
             and record['touched_reference_cells'] <= result['grid_voxels']
             and record['positive_reference_cells'] + record['unknown_reference_cells'] <= record['touched_reference_cells']
             and record['biological_vessel_free'] is None and record['clinical_injury_probability'] is None, 'contact_count_or_claim_scope')
        positive, unknown, outside = record['positive_reference_cells'], record['unknown_reference_cells'], record['outside_reference_fov']
        need(type(outside) is bool and record['annotated_positive_encounter'] is (True if positive else None if unknown or outside else False)
             and record['annotation_coverage_complete_for_sweep'] is (not unknown and not outside), 'derived_coverage_flags')
        for name, count in (('positive_cell_volume_upper_bound_mm3', positive), ('unknown_in_grid_cell_volume_mm3', unknown)):
            need(type(record[name]) in (int, float) and math.isfinite(record[name])
                 and math.isclose(record[name], count*.7**3, rel_tol=1e-12, abs_tol=1e-9), 'discrete_cell_volume')
    for name in ('touched_reference_cells', 'positive_reference_cells', 'unknown_reference_cells'):
        need(max(result['shaft'][name], result['tip'][name]) <= result['whole_tool'][name]
             <= result['shaft'][name]+result['tip'][name]
             and all(row[name] <= result['whole_tool'][name] for row in result['per_action'].values()), 'union_bounds')
    return {'status': 'generated_streaming_profile_complete', 'work': work,
            'profile_seconds': value['profile_seconds'], 'tracemalloc_peak_bytes': value['tracemalloc_peak_bytes'],
            'process_peak_rss_bytes': value['process_peak_rss_bytes']}


def output_bindings(directory):
    worker = directory/'worker'
    need(not worker.is_symlink() and worker.is_dir()
         and {p.name for p in directory.iterdir()} == {'receipt.json', 'readout-console.txt', 'worker-release.json', 'worker'}
         and {p.name for p in worker.iterdir()} == {'attempt.json', 'profile.json'}, 'exact_output_inventory')
    result = {}
    for name in ('readout-console.txt', 'worker-release.json', 'worker/attempt.json', 'worker/profile.json'):
        raw = read_regular(directory/name, CAPS['aggregate_output_bytes'])
        result[name] = {'bytes': len(raw), 'sha256': digest(raw)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    launcher_sha = digest(read_regular(Path(__file__)))
    cache = ROOT/CACHE
    if not args.check_only:
        need(sys.dont_write_bytecode and sys.pycache_prefix == str(cache)
             and not cache.exists() and not cache.is_symlink(), 'fresh_parent_source_cache_required')
    release, profile = validate_release(args.release, args.release_sha256)
    if args.check_only:
        print(json.dumps({'status': 'bindings_valid', 'execution_released': release['execution_released'], 'caps': CAPS}))
        return 0
    need(release['execution_released'] is True, 'root_release_required')
    directory = ROOT/OUTPUT
    need(not directory.exists() and not directory.is_symlink(), 'one_attempt_output_exists')
    owned = load_module('reviewed_streaming_owned_worker', OWNED)
    sys.path.insert(0, str(ROOT))
    from scripts import mechanics_hbe_v5_remaining_one_shot as supervisor
    from scripts import febio_runtime
    for name, module in [(list(HELPERS)[0], supervisor), (list(HELPERS)[1], supervisor.io), (list(HELPERS)[2], febio_runtime)]:
        need(Path(module.__file__).resolve() == ROOT/name, 'supervisor_import_origin')
        bound(name, HELPERS[name])
    directory.mkdir(parents=False, exist_ok=False)
    worker_output = directory/'worker'
    worker_cache = worker_output/'unused-pycache'
    child = child_release(); child_path = directory/'worker-release.json'
    with child_path.open('xb') as stream:
        stream.write(child); stream.flush(); os.fsync(stream.fileno())
    environment = febio_runtime.private_environment({'caps': {'thread_environment': supervisor.io.THREAD_ENV}})
    environment.update({'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(ROOT/'src')})
    command = [str(ROOT/'.venv/bin/python'), '-B', '-X', 'pycache_prefix='+str(worker_cache), str(ROOT/PROFILE),
               '--release', str(child_path), '--output-directory', str(worker_output)]
    receipt = {'schema': 'generated-streaming-profile-supervision-v1', 'status': 'reserved_before_worker',
               'launcher_sha256': launcher_sha,
               'root_release_sha256': args.release_sha256, 'child_release_sha256': digest(child),
               'source_bindings': SOURCES, 'caps': CAPS, 'patient_or_model_access_permitted': False,
               'sampling_note': 'RSS/output are sampled; brief peaks can be missed. Whole child lifecycle is supervised.'}
    supervisor.io.durable_json(directory/'receipt.json', receipt)
    owner = owned.OwnedWorker(); started = time.monotonic(); stage = None
    try:
        stage = supervisor.supervise_stage('readout', command, directory, receipt, cwd=ROOT, environment=environment,
            wall_cap=CAPS['worker_wall_seconds'], rss_cap=CAPS['sampled_process_group_rss_bytes'],
            output_cap=CAPS['aggregate_output_bytes'], rss_observer=febio_runtime.process_group_rss, popen=owner)
    except BaseException as error:
        receipt['parent_error_type'] = type(error).__name__
    finally:
        try:
            receipt['cleanup'] = owner.cleanup(febio_runtime.process_group_rss, started+44)
        except BaseException as error:
            receipt['cleanup'] = {'contained': False, 'error_type': type(error).__name__}
        receipt['status'] = 'failed_or_incomplete'
        supervisor.io.durable_json(directory/'receipt.json', receipt)
    clean = receipt['cleanup']
    if stage and stage['status'] == 'completed_within_caps' and clean.get('contained') and clean.get('direct_child_reaped') and not clean.get('fallback_used') and not clean.get('errors'):
        try:
            receipt['output_bindings'] = output_bindings(directory)
            need(digest(read_regular(args.release, 65536)) == args.release_sha256
                 and digest(read_regular(child_path, 16384)) == digest(child), 'release_changed_after_launch')
            raw = read_regular(worker_output/'profile.json', CAPS['profile_json_bytes'])
            receipt['profile_sha256'] = digest(raw); receipt['profile_bytes'] = len(raw)
            receipt['semantics'] = semantic_profile(raw)
            for name, expected in SOURCES.items(): bound(name, expected)
            profile.source_pins()
            need(all(not path.exists() and not path.is_symlink() for path in (cache, worker_cache)), 'cache_created')
            receipt['status'] = receipt['semantics']['status']
        except Exception as error:
            receipt['status'] = 'failed_profile_or_source_validation'; receipt['validation_error_type'] = type(error).__name__
    if not clean.get('contained'): receipt['status'] = 'failed_cleanup_containment'
    receipt['lifecycle_seconds'] = time.monotonic()-started
    if receipt['lifecycle_seconds'] >= CAPS['lifecycle_wall_seconds']: receipt['status'] = 'failed_lifecycle_wall_cap'
    supervisor.io.durable_json(directory/'receipt.json', receipt)
    try:
        active = supervisor.active_bytes(directory, CAPS['aggregate_output_bytes'])
        need(active <= CAPS['aggregate_output_bytes'], 'final_output_cap')
    except Exception as error:
        active = None; receipt['status'] = 'failed_final_output_scan'; receipt['final_error_type'] = type(error).__name__
        supervisor.io.durable_json(directory/'receipt.json', receipt)
    if time.monotonic()-started >= CAPS['lifecycle_wall_seconds']:
        receipt['status'] = 'failed_lifecycle_wall_cap'; supervisor.io.durable_json(directory/'receipt.json', receipt)
    if receipt['status'] == 'generated_streaming_profile_complete':
        try:
            need(output_bindings(directory) == receipt['output_bindings'], 'final_output_changed')
            need(digest(read_regular(Path(__file__))) == launcher_sha
                 and digest(read_regular(args.release, 65536)) == args.release_sha256, 'final_launcher_or_release_changed')
            for name, expected in SOURCES.items(): bound(name, expected)
            profile.source_pins()
            expected_receipt = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
            actual_receipt = read_regular(directory/'receipt.json', CAPS['profile_json_bytes'])
            need(actual_receipt == expected_receipt and json.loads(actual_receipt) == receipt, 'final_receipt_changed')
            need(time.monotonic()-started < CAPS['lifecycle_wall_seconds'], 'final_wall_cap')
        except Exception as error:
            receipt['status'] = 'failed_final_evidence_validation'; receipt['final_error_type'] = type(error).__name__
            supervisor.io.durable_json(directory/'receipt.json', receipt)
    print(json.dumps({'status': receipt['status'], 'receipt': str(directory/'receipt.json'), 'final_output_bytes': active}))
    return 0 if receipt['status'] == 'generated_streaming_profile_complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
