"""One release-gated saved-log replay of the exact failed HBE v5 N12 attempt.

This module cannot run FEBio. Its output is a supplemental numerical software
classification, never a modification of the original failed receipt or an
automatic predecessor/comparator admission.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import subprocess
import sys
import time

from scripts import mechanics_hbe_v5_n8_one_shot as io
from scripts import mechanics_hbe_v5_remaining_one_shot as prior


ROOT = Path(__file__).resolve().parents[1]
PREPARATION = 'manifests/experiments/hbe-v5-n12-saved-attempt-replay-preparation-v1.json'
PREPARATION_SHA = '45c6728ee2ce183990a1110c4ca333e22ed175b597d9e5378a131d2d60577329'
ORIGINAL = prior.output_directory(1)
ORIGINAL_RECEIPT_SHA = '9020e6c7ea1f01360dc8d02e5efd16e4a29591c2f566d0194415b03d8b68bbc2'
ORIGINAL_RELEASE = 'build/hbe-v5-n12-one-shot-release/release.json'
ORIGINAL_RELEASE_SHA = 'b9e765529e98668e8fe675b61bc7084d48efddb1f431221174801d3bfe50acad'
ORIGINAL_SOURCE_COMMIT = '936cf2acb13cd01a80e88d8d5791afff9cf846f3'
OUTPUT = 'outputs/mechanics/hbe-v5-n12-saved-attempt-replay-v1/attempt-01'
CAPS = {'native_calls': 0, 'replay_calls': 1, 'attempts': 1,
        'replay_wall_seconds': 600, 'sampled_process_group_rss_bytes': 3*1024**3,
        'active_output_bytes': 64*1024**2, 'preparation_wall_seconds': 150,
        'numerical_threads': 1}
SOURCE_PATHS = tuple(dict.fromkeys((
    'scripts/mechanics_hbe_v5_n12_saved_replay.py', *prior.SOURCE_PATHS)))
ORIGINAL_SOURCE_PATHS = tuple(dict.fromkeys((
    'scripts/mechanics_hbe_v5_remaining_one_shot.py', *io.SOURCE_PATHS)))
OUTPUT_NAMES = {'receipt.json', 'replay-work-order.json',
                'readout-console.txt', 'replay.json'}


def audit_imports(*, root: Path = ROOT) -> None:
    allowed = set(SOURCE_PATHS)
    base = root.resolve()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith('scripts.'):
            continue
        origin = getattr(module, '__file__', None)
        if origin is None:
            continue
        path = Path(origin).resolve()
        if not path.is_relative_to(base / 'scripts') or str(path.relative_to(base)) not in allowed:
            raise ValueError('Unbound replay executing import: ' + name)
    fixture = sys.modules.get('scripts.mechanics_patient_constraints')
    if fixture is not None:
        patch = getattr(fixture, 'patch', None)
        if patch is None or Path(patch.__file__).resolve() != base / 'scripts/mechanics_febio_verification.py':
            raise ValueError('Unbound dynamically loaded checker')


def committed_source(commit: str, bindings: dict, *, paths: tuple[str, ...],
                     root: Path = ROOT, working: bool = False,
                     require_head: bool = False) -> str:
    """Authenticate every released blob; optionally check current working bytes."""
    if (not isinstance(commit, str) or not prior.COMMIT.fullmatch(commit)
            or not isinstance(bindings, dict) or set(bindings) != set(paths)):
        raise ValueError('Complete immutable source closure required')
    head = subprocess.run(['/usr/bin/git', 'rev-parse', 'HEAD'], cwd=root,
        env=io.git_env(), capture_output=True, check=True, timeout=10).stdout.decode().strip()
    if require_head and head != commit:
        raise ValueError('Replay release must name launch HEAD')
    for relative in paths:
        binding = bindings[relative]
        if (not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}
                or binding['path'] != relative
                or not isinstance(binding['sha256'], str)
                or not prior.HEX.fullmatch(binding['sha256'])):
            raise ValueError('Exact source path/hash binding required')
        name = f'{commit}:{relative}'
        size = subprocess.run(['/usr/bin/git', 'cat-file', '-s', name], cwd=root,
            env=io.git_env(), capture_output=True, check=True, timeout=10).stdout
        if int(size) > 1024**2:
            raise ValueError('Committed source exceeds bounded reader')
        original = subprocess.run(['/usr/bin/git', 'cat-file', 'blob', name], cwd=root,
            env=io.git_env(), capture_output=True, check=True, timeout=10).stdout
        if len(original) > 1024**2 or io.sha(original) != binding['sha256']:
            raise ValueError('Released source differs from immutable Git blob')
        if working:
            io.bound(binding, relative, root=root)
    return head


def validate_preparation(prep: dict) -> None:
    if (not isinstance(prep, dict)
            or prep.get('schema') != 'hbe-v5-n12-saved-attempt-replay-preparation-v1'
            or prep.get('status') != 'source_only_not_released_or_executed'
            or prep.get('release') is not None
            or prep.get('run_id') != prior.ORDER[1] or prep.get('ordinal') != 1
            or prep.get('original_attempt_directory') != ORIGINAL
            or prep.get('original_receipt_sha256') != ORIGINAL_RECEIPT_SHA
            or prep.get('original_release_path') != ORIGINAL_RELEASE
            or prep.get('original_release_sha256') != ORIGINAL_RELEASE_SHA
            or prep.get('original_source_commit') != ORIGINAL_SOURCE_COMMIT
            or prep.get('original_failure') != {
                'type': 'ValueError', 'message': 'Native output absent or above bound'}
            or prep.get('output_directory') != OUTPUT or prep.get('caps') != CAPS
            or prep.get('closed_attempt_bytes') != 50501397
            or set(prep.get('closed_attempt_files', {})) != prior.FINAL_FILES
            or prep['closed_attempt_files']['receipt.json'] != {
                'bytes': 7949, 'sha256': ORIGINAL_RECEIPT_SHA}
            or prep['closed_attempt_files']['readout-console.txt'] != {
                'bytes': 0, 'sha256': io.sha(b'')}
            or prep.get('phase_gates') != {
                'native_execution': False, 'generic_failed_predecessor_admission': False,
                'twelve_row_comparator_admission': False, 'fit': False,
                'held_out_torque_access': False, 'physical_validation': False,
                'patient_or_measured_response_access': False}):
        raise ValueError('Specific saved-attempt replay preparation changed')
    if (sum(row['bytes'] for row in prep['closed_attempt_files'].values()) !=
            prep['closed_attempt_bytes']
            or any(name != 'readout-console.txt' and row['bytes'] <= 0
                   for name, row in prep['closed_attempt_files'].items())
            or any(not isinstance(row['sha256'], str)
                   or not prior.HEX.fullmatch(row['sha256'])
                   or type(row['bytes']) is not int
                   for row in prep['closed_attempt_files'].values())):
        raise ValueError('Saved output whitelist malformed')


def original_inventory(prep: dict, *, root: Path = ROOT) -> dict:
    directory = io.local(ORIGINAL, root=root)
    if {entry.name for entry in directory.iterdir()} != prior.FINAL_FILES:
        raise ValueError('Original closed attempt inventory changed')
    if prior.active_bytes(directory, CAPS['active_output_bytes']) != prep['closed_attempt_bytes']:
        raise ValueError('Original closed attempt size changed')
    result = {}
    for name, expected in prep['closed_attempt_files'].items():
        relative = ORIGINAL + '/' + name
        path = io.local(relative, root=root)
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError('Original attempt contains nonregular file')
        observed = {'bytes': path.stat().st_size,
                    'sha256': io.file_hash(path, maximum=CAPS['active_output_bytes'])}
        if observed != expected:
            raise ValueError('Exact original file changed: ' + name)
        result[name] = {'path': relative, **observed}
    return result


def validate_original(prep: dict, *, root: Path = ROOT,
                      audit_loaded_imports: bool = True) -> dict:
    """Read-only original receipt/release/runtime/geometry and saved JSON check."""
    inventory = original_inventory(prep, root=root)
    receipt = json.loads(io.bound({'path': ORIGINAL+'/receipt.json',
        'sha256': ORIGINAL_RECEIPT_SHA}, ORIGINAL+'/receipt.json', root=root))
    release_raw = io.bound({'path': ORIGINAL_RELEASE, 'sha256': ORIGINAL_RELEASE_SHA},
                            ORIGINAL_RELEASE, root=root)
    released = json.loads(release_raw)
    original_source = released.get('source_bindings')
    committed_source(ORIGINAL_SOURCE_COMMIT, original_source,
                     paths=ORIGINAL_SOURCE_PATHS, root=root)
    if (released.get('schema') != 'hbe-v5-remaining-one-call-release-v1'
            or released.get('status') != 'root_released_one_native_call'
            or released.get('ordinal') != 1 or released.get('run_id') != prior.ORDER[1]
            or released.get('source_commit') != ORIGINAL_SOURCE_COMMIT
            or released.get('output_directory') != ORIGINAL
            or released.get('caps') != prior.caps(1)
            or released.get('aggregate_caps') != prior.AGGREGATE
            or released.get('prior_receipts') != [{'run_id': prior.ORDER[0],
                'path': prior.N8_PATH, 'sha256': prior.N8_SHA}]
            or receipt.get('schema') != 'hbe-v5-remaining-one-call-receipt-v1'
            or receipt.get('status') != 'failed_or_incomplete'
            or receipt.get('failure') != prep['original_failure']
            or receipt.get('ordinal') != 1 or receipt.get('run_id') != prior.ORDER[1]
            or receipt.get('release_sha256') != ORIGINAL_RELEASE_SHA
            or receipt.get('release_path') != str(io.local(ORIGINAL_RELEASE, root=root))
            or receipt.get('source_commit') != ORIGINAL_SOURCE_COMMIT
            or receipt.get('observed_head_preflight') != ORIGINAL_SOURCE_COMMIT
            or receipt.get('source_hashes_before') != {
                name: binding['sha256'] for name, binding in original_source.items()}
            or receipt.get('native_calls_attempted') != 1
            or receipt.get('readout_calls_attempted') != 1
            or receipt.get('no_retry') is not True
            or receipt.get('caps') != prior.caps(1)
            or receipt.get('aggregate_caps') != prior.AGGREGATE
            or receipt.get('prior_receipt_sha256') != [prior.N8_SHA]
            or receipt.get('source_hashes_after') is not None
            or receipt.get('runtime_profile_verified_after') is not None):
        raise ValueError('Whitelisted failed attempt or original release differs')
    for stage in ('native', 'readout'):
        record = receipt.get(stage+'_stage', {})
        if (record.get('status') != 'completed_within_caps'
                or record.get('exit_code') != 0
                or record.get('kill_reason') is not None
                or type(record.get('elapsed_seconds')) not in (int, float)
                or not 0 <= record['elapsed_seconds'] <
                   (prior.caps(1)['native_wall_seconds'] if stage == 'native'
                    else prior.READOUT_WALL)):
            raise ValueError('Original supervised stage did not complete within caps')
    from scripts import mechanics_hbe_branch_calibration_v5 as v5
    from scripts import mechanics_hbe_v5_source_bindings as sources
    from scripts import mechanics_hbe_backend as backend
    old_prep = json.loads(io.bound(released['preparation'], prior.PREPARATION, root=root))
    prior.validate_preparation(old_prep)
    study, ancestry = v5.validate_preparation(root)
    sources.validate_binding_manifest(root, inspect_sources=True)
    source_map = json.loads(io.bound(released['source_map'], prior.SOURCE_MAP, root=root))
    key = source_map['run_source_keys'][prior.ORDER[1]]
    source_row = source_map['source_decks'][key]
    mesh = source_map['meshes'][source_row['mesh']]
    old_binding = {name: source_row[name] for name in ('path', 'sha256')}
    if (released['old_source_deck'] != old_binding or released['native_mesh'] != mesh
            or released['v5_declaration'] != {'path': prior.V5,
                'sha256': v5.DECLARATION_SHA256}
            or released['source_map'] != {'path': prior.SOURCE_MAP,
                'sha256': sources.BINDING_SHA256}
            or released['backend_profile'] !=
               ancestry['preserved_v3_fields']['backend_profile']
            or released['runtime_identity'] !=
               ancestry['preserved_v3_fields']['runtime_identity']):
        raise ValueError('Original frozen deck/mesh/runtime ancestry changed')
    old = sources._read_bound(root, old_binding, maximum=sources.MAX_SOURCE_BYTES)
    native_mesh = sources._read_bound(root, mesh, maximum=sources.MAX_MESH_BYTES)
    sources.validate_topology_bc(json.loads(native_mesh), old,
        native_domain=v5.run_spec(study, ancestry, prior.ORDER[1])['native_domain'])
    _, adapted, adapter = v5.adapt_deck(study, ancestry, prior.ORDER[1], old)
    if (io.sha(adapted.encode()) != released['adapted_deck_sha256']
            or released['adapted_deck_sha256'] != inventory['specimen.feb']['sha256']
            or released['adapter_receipt'] != adapter
            or receipt.get('adapted_deck_sha256') != inventory['specimen.feb']['sha256']
            or receipt.get('runtime_identity_sha256') != released['runtime_identity']['sha256']
            or receipt.get('backend_profile_sha256') != released['backend_profile']['sha256']):
        raise ValueError('Original adapted deck, schedule or runtime receipt differs')
    profile = backend.verify_profile(root, released['backend_profile'])
    if (profile['runtime_identity'] != released['runtime_identity']
            or profile['profile_id'] != 'accelerate_csc_v1'):
        raise ValueError('Current repaired runtime/profile recheck failed')
    directory = io.local(ORIGINAL, root=root)
    prior._check_backend(directory)
    native_records = {name: {'path': inventory[name]['path'],
                             'sha256': inventory[name]['sha256'],
                             'bytes': inventory[name]['bytes']}
                      for name in prior.NATIVE_FILES}
    if receipt.get('native_output_bindings') != native_records:
        raise ValueError('Original native output receipt bindings differ')
    work_order = json.loads((directory / 'readout-work-order.json').read_bytes())
    if (work_order.get('readout_token_sha256') != receipt.get('readout_token_sha256')
            or work_order.get('source_commit') != ORIGINAL_SOURCE_COMMIT
            or work_order.get('release_sha256') != ORIGINAL_RELEASE_SHA):
        raise ValueError('Original readout work order differs')
    context = {'index': 1, 'run_id': prior.ORDER[1], 'deck': adapted.encode(),
               'native_domain': v5.run_spec(study, ancestry, prior.ORDER[1])['native_domain'],
               'v5_declaration_sha256': v5.DECLARATION_SHA256,
               'old_source': old_binding, 'mesh': mesh,
               'adapter_receipt': adapter}
    records, summary = prior.inspect_readout(directory, context, native_records,
                                               receipt, root=root)
    if (summary['readout_sha256'] != inventory['readout.json']['sha256']
            or any(records[name]['sha256'] != inventory[name]['sha256']
                   for name in records)):
        raise ValueError('Original saved readout identity differs')
    prior.validate_prior_chain(released['prior_receipts'], 1, root=root)
    if audit_loaded_imports:
        audit_imports(root=root)
    return {'receipt': receipt, 'release': released, 'release_raw': release_raw,
            'inventory': inventory, 'work_order': work_order,
            'saved_json': json.loads((directory / 'readout.json').read_bytes()),
            'current_runtime_profile': profile}


def validate_release(release: dict, *, root: Path = ROOT) -> dict:
    keys = {'schema', 'status', 'preparation', 'source_commit',
            'source_bindings', 'original_receipt', 'original_release',
            'output_directory', 'caps'}
    if (not isinstance(release, dict) or set(release) != keys
            or release['schema'] != 'hbe-v5-n12-saved-attempt-replay-release-v1'
            or release['status'] != 'root_released_one_saved_output_replay'
            or release['preparation'] != {'path': PREPARATION,
                'sha256': PREPARATION_SHA}
            or release['original_receipt'] != {'path': ORIGINAL+'/receipt.json',
                'sha256': ORIGINAL_RECEIPT_SHA}
            or release['original_release'] != {'path': ORIGINAL_RELEASE,
                'sha256': ORIGINAL_RELEASE_SHA}
            or release['output_directory'] != OUTPUT or release['caps'] != CAPS):
        raise ValueError('Separate exact-hash replay-only release required')
    prep = json.loads(io.bound(release['preparation'], PREPARATION, root=root))
    validate_preparation(prep)
    head = committed_source(release['source_commit'], release['source_bindings'],
                            paths=SOURCE_PATHS, root=root,
                            working=True, require_head=True)
    original = validate_original(prep, root=root)
    if io.local(OUTPUT, root=root).exists():
        raise ValueError('One-shot replay output directory already exists')
    if shutil.disk_usage(root).free < CAPS['active_output_bytes'] + 1024**3:
        raise ValueError('Replay output and local storage reserve unavailable')
    audit_imports(root=root)
    return {'preparation': prep, 'original': original,
            'observed_head_preflight': head,
            'source_hashes': {name: binding['sha256'] for name, binding in
                              release['source_bindings'].items()}}


def replay_worker(work_order_path: str, *, root: Path = ROOT) -> int:
    path = io.local(work_order_path, root=root)
    if (work_order_path != OUTPUT+'/replay-work-order.json'
            or path.stat().st_size > 1024**2):
        raise ValueError('Exact bounded replay work order required')
    order = json.loads(path.read_bytes())
    token = os.environ.pop('HBE_V5_N12_REPLAY_TOKEN', None)
    if (not isinstance(order, dict)
            or set(order) != {'schema', 'run_id', 'source_commit',
                              'original_receipt_sha256', 'token_sha256',
                              'bindings', 'adapter_receipt'}
            or order['schema'] != 'hbe-v5-n12-saved-replay-work-order-v1'
            or order['run_id'] != prior.ORDER[1]
            or order['original_receipt_sha256'] != ORIGINAL_RECEIPT_SHA
            or not isinstance(token, str) or len(token) != 64
            or io.sha(token.encode()) != order['token_sha256']):
        raise ValueError('Supervised saved replay token or identity differs')
    from scripts import mechanics_hbe_v5_stream as stream
    audit_imports(root=root)
    result = stream.read_bound_run(root, prior.ORDER[1], order['bindings'],
                                   order['adapter_receipt'])
    audit_imports(root=root)
    if result.get('numerical_passed') is not True or result.get('frame_count') != 61:
        raise ValueError('Frozen complete saved stream failed numerical gates')
    io.durable_json(path.parent / 'replay.json', result)
    return 0


def execute(release_path: Path, *, root: Path = ROOT) -> dict:
    """One supervised Python-only replay; original failed files stay read-only."""
    started = time.monotonic()
    raw, identity = io.read_release(release_path)
    release = json.loads(raw)
    context = validate_release(release, root=root)
    if time.monotonic()-started >= CAPS['preparation_wall_seconds']:
        raise ValueError('Replay preparation wall cap expired before reservation')
    output = io.local(OUTPUT, root=root)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'schema': 'hbe-v5-n12-saved-attempt-supplement-v1',
               'status': 'reserved', 'run_id': prior.ORDER[1],
               'original_status_preserved': 'failed_or_incomplete',
               'original_receipt_sha256': ORIGINAL_RECEIPT_SHA,
               'original_release_path': ORIGINAL_RELEASE,
               'original_release_sha256': ORIGINAL_RELEASE_SHA,
               'original_source_commit': ORIGINAL_SOURCE_COMMIT,
               'original_closed_attempt_bytes': context['preparation']['closed_attempt_bytes'],
               'original_file_bindings': context['original']['inventory'],
               'original_runtime_identity_sha256':
                   context['original']['release']['runtime_identity']['sha256'],
               'original_backend_profile_sha256':
                   context['original']['release']['backend_profile']['sha256'],
               'historical_guard_gap': 'Original post-run source/runtime guards were not recorded after parent packaging failure.',
               'current_runtime_profile_rechecked': True,
               'current_source_commit': release['source_commit'],
               'current_source_hashes_before': context['source_hashes'],
               'observed_head_preflight': context['observed_head_preflight'],
               'release_path': str(release_path.resolve()),
               'release_sha256': io.sha(raw),
               'preparation_sha256': PREPARATION_SHA, 'caps': CAPS,
               'native_calls_attempted': 0, 'readout_calls_attempted': 0,
               'replay_calls_attempted': 0,
               'no_retry': True,
               'physical_validation_pass': None,
               'measured_response_accessed': False,
               'patient_data_accessed': False,
               'comparator_admitted': False,
               'predecessor_admitted': False}
    io.durable_json(output / 'receipt.json', receipt)
    try:
        token = secrets.token_hex(32)
        original_work = context['original']['work_order']
        work = {'schema': 'hbe-v5-n12-saved-replay-work-order-v1',
                'run_id': prior.ORDER[1], 'source_commit': release['source_commit'],
                'original_receipt_sha256': ORIGINAL_RECEIPT_SHA,
                'token_sha256': io.sha(token.encode()),
                'bindings': original_work['bindings'],
                'adapter_receipt': original_work['adapter_receipt']}
        receipt['replay_token_sha256'] = work['token_sha256']
        io.durable_json(output / 'replay-work-order.json', work)
        environment = io.private_environment()
        environment['HBE_V5_N12_REPLAY_TOKEN'] = token
        from scripts.febio_runtime import process_group_rss
        audit_imports(root=root)
        stage = prior.supervise_stage('readout', [sys.executable, '-B', '-m',
            'scripts.mechanics_hbe_v5_n12_saved_replay', '--replay-worker',
            '--work-order', OUTPUT+'/replay-work-order.json'], output, receipt,
            cwd=root, environment=environment,
            wall_cap=CAPS['replay_wall_seconds'],
            rss_cap=CAPS['sampled_process_group_rss_bytes'],
            output_cap=CAPS['active_output_bytes'],
            rss_observer=process_group_rss)
        token = ''
        if stage['status'] != 'completed_within_caps':
            raise ValueError('Supervised saved-log replay failed or exceeded cap')
        if {entry.name for entry in output.iterdir()} != OUTPUT_NAMES:
            raise ValueError('Replay-only output inventory differs')
        bindings = {name: prior._regular_output_binding(output / name,
                    root=root, limit=CAPS['active_output_bytes'],
                    allow_empty=name == 'readout-console.txt')
                    for name in OUTPUT_NAMES - {'receipt.json'}}
        observed_work = json.loads((output / 'replay-work-order.json').read_bytes())
        if (observed_work != work
                or bindings['replay.json']['bytes'] > 16*1024**2
                or json.loads((output / 'replay.json').read_bytes()) !=
                   context['original']['saved_json']):
            raise ValueError('Saved replay differs from original complete numerical readout')
        receipt['replay_output_bindings'] = bindings
        post_original = validate_original(context['preparation'], root=root)
        if (post_original['inventory'] != context['original']['inventory']
                or post_original['release_raw'] != context['original']['release_raw']
                or io.read_release(release_path) != (raw, identity)):
            raise ValueError('Original assumptions, saved files or replay release changed')
        observed_head = committed_source(release['source_commit'],
            release['source_bindings'], paths=SOURCE_PATHS, root=root,
            working=True, require_head=False)
        from scripts import mechanics_hbe_backend as backend
        after = backend.verify_profile(root, context['original']['release']['backend_profile'])
        audit_imports(root=root)
        if after != context['original']['current_runtime_profile']:
            raise ValueError('Current repaired runtime/profile changed during replay')
        receipt['observed_head_after'] = observed_head
        receipt['current_source_hashes_after'] = context['source_hashes']
        receipt['current_runtime_profile_rechecked_after'] = True
        receipt['replay_numerical_passed'] = True
        receipt['frame_count'] = 61
        receipt['status'] = 'supplemental_saved_attempt_numerical_pass_only'
    except BaseException as error:
        receipt['status'] = 'failed_or_incomplete'
        receipt['failure'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        receipt['replay_calls_attempted'] = receipt.get('readout_calls_attempted', 0)
        receipt['preparation_elapsed_seconds'] = max(0., time.monotonic()-started-
            receipt.get('readout_stage', {}).get('elapsed_seconds', 0.))
        try:
            if (receipt['preparation_elapsed_seconds'] >= CAPS['preparation_wall_seconds']
                    or prior.active_bytes(output, CAPS['active_output_bytes']) >
                       CAPS['active_output_bytes']):
                raise ValueError('Preparation or output cap exceeded')
        except BaseException as error:
            receipt['status'] = 'failed_or_incomplete'
            receipt['final_resource_failure'] = {'type': type(error).__name__,
                                                 'message': str(error)}
        io.durable_json(output / 'receipt.json', receipt)
        # A mutable receipt cannot assert its own final hash/size. Independent
        # review rehashes the closed directory; only the cap is checked here.
        for _ in range(2):
            try:
                if prior.active_bytes(output, CAPS['active_output_bytes']) > \
                        CAPS['active_output_bytes']:
                    raise ValueError('Closed replay output cap exceeded')
                break
            except BaseException as error:
                receipt['status'] = 'failed_or_incomplete'
                receipt['final_resource_failure'] = {'type': type(error).__name__,
                                                     'message': str(error)}
                io.durable_json(output / 'receipt.json', receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--replay-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--work-order', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.replay_worker:
        if args.execute or args.release or not args.work_order:
            raise ValueError('Replay worker accepts only its bounded work order')
        return replay_worker(args.work_order)
    if not args.execute or args.release is None or args.work_order is not None:
        raise ValueError('Explicit replay-only release and --execute required')
    result = execute(args.release)
    print(json.dumps({'status': result['status'], 'receipt': OUTPUT+'/receipt.json'}))
    return 0 if result['status'] == 'supplemental_saved_attempt_numerical_pass_only' else 1


if __name__ == '__main__':
    raise SystemExit(main())
