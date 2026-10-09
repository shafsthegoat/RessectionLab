"""Root-released NFBS names/sizes launcher; existing command supervisor, no extraction or image parser."""
from pathlib import Path
import argparse, hashlib, json, os, signal, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[2]
PREPARED = 'build/nfbs-member-inventory-preparation-v1/prepared-protocol.json'
PREPARED_SHA = 'fcee25707c30ddae9ac38fb5cec0dfd10cd3712327b8df981b6cdb3e6ec5b724'
WORKER = 'build/nfbs-member-inventory-preparation-v1/inventory.py'
WORKER_SHA = '0488f36b8da0d176f4e53ca20c75c21e4ad0d62ff7f26a9d3894f6f3f2d3069f'
REVIEW = 'build/nfbs-member-inventory-independent-v1/REPORT.md'
REVIEW_SHA = '727fe71699da19c16eaa4a7d05e84d142cdb817e19fef31665c7998bc14aab8d'
OUTPUT = 'build/nfbs-member-inventory-execution-v1'
EXECUTING_SOURCE_COMMIT = '0f7eee3a17b6743fe15549cf8aba396fa4e98a2c'
CAPS = {'attempts': 1, 'wall_seconds': 130, 'sampled_process_group_rss_bytes': 268435456,
        'aggregate_output_bytes': 4194304, 'numerical_threads': 1,
        'worker_stage_wall_seconds': 120, 'cleanup_and_finalization_reserve_seconds': 10}
SUPERVISOR_FILES = {
    'scripts/mechanics_hbe_v5_remaining_one_shot.py': '90d855b1e9518793480688afbfb133b586a6ccbe5b974be30b6b40624336216b',
    'scripts/mechanics_hbe_v5_n8_one_shot.py': '82bb01dc2588f750489916c789e7afe85b7a9a6e1ef0fcd5820fe8e70be4f3a0',
    'scripts/febio_runtime.py': '679594d7f3759f5485b9fb868e7e5543ebb242bccd24d112f9ca862bb6d2a01e'}

def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def bound(relative, expected):
    path = ROOT / relative
    require(not path.is_symlink() and path.is_file() and path.stat().st_size < 16*1024**2, 'binding_type_or_size')
    raw = path.read_bytes()
    require(digest(raw) == expected, 'binding_changed:' + relative)
    return raw

def child_protocol():
    prepared = json.loads(bound(PREPARED, PREPARED_SHA))
    require(prepared['execution_released'] is False, 'prepared_protocol_not_false')
    released = dict(prepared, execution_released=True)
    return (json.dumps(released, sort_keys=True, indent=2) + '\n').encode()

class OwnedWorker:
    """Retain the exact child handle for bounded cleanup after supervisor EPERM."""
    def __init__(self, factory=subprocess.Popen):
        self.factory, self.process = factory, None

    def __call__(self, *args, **kwargs):
        require(self.process is None, 'second_worker_forbidden')
        self.process = self.factory(*args, **kwargs)
        return self.process

    def cleanup(self, observer, deadline):
        record = {'contained': self.process is None, 'direct_child_reaped': self.process is None,
                  'remaining_members': [], 'errors': [], 'fallback_used': False}
        if self.process is None:
            return record
        process = self.process
        def remaining(maximum):
            value = min(maximum, deadline-time.monotonic())
            require(value > 0, 'cleanup_deadline')
            return value
        members = []
        try:
            _, members = observer(process.pid, timeout_seconds=remaining(.5))
        except Exception as error:
            record['errors'].append('initial_observer:' + type(error).__name__)
        if process.poll() is None or members or record['errors']:
            record['fallback_used'] = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as error:
                record['errors'].append('killpg:' + type(error).__name__)
            # A failed group signal must not skip direct-child termination/reaping.
            try:
                if process.poll() is None:
                    process.kill()
            except Exception as error:
                record['errors'].append('direct_kill:' + type(error).__name__)
            for member in members:
                if member['pid'] != process.pid:
                    try:
                        os.kill(member['pid'], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except Exception as error:
                        record['errors'].append('member_kill:' + type(error).__name__)
        try:
            record['exit_code'] = process.wait(timeout=remaining(2.))
            record['direct_child_reaped'] = True
        except Exception as error:
            record['errors'].append('reap:' + type(error).__name__)
        try:
            _, record['remaining_members'] = observer(process.pid, timeout_seconds=remaining(.5))
            record['contained'] = record['direct_child_reaped'] and not record['remaining_members']
        except Exception as error:
            record['errors'].append('final_observer:' + type(error).__name__)
        return record

def validate(release_path, release_sha):
    require(not release_path.is_symlink() and release_path.is_file()
            and release_path.stat().st_size <= 65536, 'release_type_or_size')
    raw = release_path.read_bytes()
    require(len(raw) <= 65536 and digest(raw) == release_sha, 'release_changed')
    release = json.loads(raw)
    require(release['schema'] == 'nfbs-root-only-member-inventory-launch-v1', 'wrong_release_schema')
    require(release['launcher_sha256'] == digest(Path(__file__).read_bytes()), 'launcher_changed')
    require(release['worker'] == {'path': WORKER, 'sha256': WORKER_SHA}, 'worker_binding')
    require(release['prepared_protocol'] == {'path': PREPARED, 'sha256': PREPARED_SHA}, 'prepared_binding')
    require(release['independent_review'] == {'path': REVIEW, 'sha256': REVIEW_SHA}, 'review_binding')
    require(release['supervisor_files_sha256'] == SUPERVISOR_FILES, 'supervisor_binding')
    require(release['executing_source_commit'] == EXECUTING_SOURCE_COMMIT, 'historical_source_binding')
    require(release['caps'] == CAPS and release['output_directory'] == OUTPUT, 'cap_or_output_change')
    require(release['image_decode_allowed'] is False and release['role_assignments_changed'] is False, 'scope_broadened')
    for name, expected in SUPERVISOR_FILES.items():
        bound(name, expected)
        historical = subprocess.run(['/usr/bin/git', 'show', EXECUTING_SOURCE_COMMIT + ':' + name],
            cwd=ROOT, capture_output=True, check=True, timeout=5).stdout
        require(digest(historical) == expected, 'historical_source_changed:' + name)
    bound(WORKER, WORKER_SHA); bound(REVIEW, REVIEW_SHA)
    prepared = json.loads(bound(PREPARED, PREPARED_SHA))
    require(sys.implementation.name == 'cpython' and sys.version.split()[0] == prepared['python_version'], 'python_version_changed')
    import tarfile, gzip
    for name, module in [('tarfile', tarfile), ('gzip', gzip)]:
        require(digest(Path(module.__file__).read_bytes()) == prepared['stdlib_source_sha256'][name], 'stdlib_source_changed:' + name)
    bound(prepared['cohort_path'], prepared['cohort_sha256'])
    protocol = child_protocol()
    require(digest(protocol) == release['released_child_protocol_sha256'], 'child_protocol_changed')
    head = subprocess.run(['/usr/bin/git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    # Root may advance unrelated evidence commits. Executing bytes stay pinned.
    return release, protocol, head

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', type=Path, required=True)
    ap.add_argument('--release-sha256', required=True)
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    release, protocol, current_head = validate(args.release, args.release_sha256)
    if args.check_only:
        print(json.dumps({'status': 'launcher_bindings_valid', 'execution_released': release['execution_released'],
                          'original_archive_opens': 0, 'caps': CAPS, 'current_head': current_head,
                          'historical_executing_source_commit': EXECUTING_SOURCE_COMMIT}))
        return
    require(release['execution_released'] is True and release['status'] == 'root_released_structure_inventory_once', 'root_release_required')
    directory = ROOT / OUTPUT
    require(not directory.exists() and not directory.is_symlink(), 'one_attempt_directory_exists')
    # Import only the existing process supervisor, its I/O helper and observer.
    sys.path.insert(0, str(ROOT))
    from scripts import mechanics_hbe_v5_remaining_one_shot as supervisor
    from scripts import febio_runtime
    for name, module in [('scripts/mechanics_hbe_v5_remaining_one_shot.py', supervisor),
                         ('scripts/mechanics_hbe_v5_n8_one_shot.py', supervisor.io),
                         ('scripts/febio_runtime.py', febio_runtime)]:
        require(Path(module.__file__).resolve() == ROOT / name, 'wrong_import:' + name)
        bound(name, SUPERVISOR_FILES[name])
    directory.mkdir(parents=True, exist_ok=False)
    protocol_path = directory / 'released-protocol.json'
    with protocol_path.open('xb') as stream:
        stream.write(protocol)
    environment = febio_runtime.private_environment({'caps': {'thread_environment': supervisor.io.THREAD_ENV}})
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    command = [str(ROOT / '.venv/bin/python'), '-B', str(ROOT / WORKER),
               '--protocol', str(protocol_path), '--protocol-sha256', digest(protocol)]
    receipt = {'schema': 'nfbs-supervised-member-inventory-one-shot-v1',
               'release_sha256': args.release_sha256, 'child_protocol_sha256': digest(protocol),
               'worker_sha256': WORKER_SHA, 'independent_review_sha256': REVIEW_SHA,
               'historical_executing_source_commit': EXECUTING_SOURCE_COMMIT,
               'observed_head_at_root_release': release['observed_head'], 'observed_head_at_launch': current_head,
               'caps': CAPS, 'role_assignments_changed': False, 'image_decode_allowed': False,
               'status': 'running', 'sampling_note': 'RSS/output are sampled; brief between-sample peaks may be missed.'}
    owner = OwnedWorker()
    outer_started = time.monotonic()
    try:
        result = supervisor.supervise_stage('readout', command, directory, receipt,
            cwd=ROOT, environment=environment, wall_cap=CAPS['worker_stage_wall_seconds'],
            rss_cap=CAPS['sampled_process_group_rss_bytes'], output_cap=CAPS['aggregate_output_bytes'],
            rss_observer=febio_runtime.process_group_rss, popen=owner)
    finally:
        receipt['launcher_cleanup'] = owner.cleanup(febio_runtime.process_group_rss,
            outer_started+CAPS['wall_seconds']-1.)
        receipt['supervised_lifecycle_seconds'] = time.monotonic()-outer_started
        receipt['status'] = 'post_supervision_pending' if receipt['launcher_cleanup']['contained'] else 'failed_cleanup_containment'
        supervisor.io.durable_json(directory / 'receipt.json', receipt)
    # The supervisor watches the same directory the fixed worker writes into.
    # No mechanical solver/deck/readout function is called.
    inventory_path = directory / 'inventory.json'
    verified = (result['status'] == 'completed_within_caps' and inventory_path.is_file()
                and not inventory_path.is_symlink()
                and inventory_path.stat().st_size <= CAPS['aggregate_output_bytes']
                and receipt['launcher_cleanup']['contained']
                and not receipt['launcher_cleanup']['fallback_used']
                and not receipt['launcher_cleanup']['errors'])
    receipt['status'] = 'supervision_complete_inventory_uninterpreted' if verified else 'failed_or_incomplete'
    if not receipt['launcher_cleanup']['contained']:
        receipt['status'] = 'failed_cleanup_containment'
    if verified:
        receipt['inventory_bytes'] = inventory_path.stat().st_size
        receipt['inventory_sha256'] = digest(inventory_path.read_bytes())
    receipt['supervised_lifecycle_seconds'] = time.monotonic()-outer_started
    if receipt['supervised_lifecycle_seconds'] >= CAPS['wall_seconds']:
        receipt['status'] = 'failed_lifecycle_wall_cap'
    supervisor.io.durable_json(directory / 'receipt.json', receipt)
    try:
        final_bytes = supervisor.active_bytes(directory, CAPS['aggregate_output_bytes'])
        if final_bytes > CAPS['aggregate_output_bytes']:
            receipt['status'] = 'failed_final_aggregate_output_cap'
            supervisor.io.durable_json(directory / 'receipt.json', receipt)
    except Exception as error:
        final_bytes = None
        receipt['status'] = 'failed_final_aggregate_output_scan'
        receipt['final_scan_error_type'] = type(error).__name__
        supervisor.io.durable_json(directory / 'receipt.json', receipt)
    if time.monotonic()-outer_started >= CAPS['wall_seconds']:
        receipt['status'] = 'failed_lifecycle_wall_cap'
        supervisor.io.durable_json(directory / 'receipt.json', receipt)
    # Inventory semantic status is deliberately not inferred from exit zero.
    print(json.dumps({'status': receipt['status'], 'receipt': str(directory / 'receipt.json'),
                      'readout_status': result['status'], 'final_scanned_output_bytes': final_bytes}))
    return 0 if receipt['status'] == 'supervision_complete_inventory_uninterpreted' else 1

if __name__ == '__main__':
    raise SystemExit(main())
