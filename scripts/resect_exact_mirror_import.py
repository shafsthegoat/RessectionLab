#!/usr/bin/env python3
"""Offline installation of 24 authenticated RESECT TRAIN mirror bodies.

After archiving the immutable mirror metadata under artifacts, run:
  .venv/bin/python scripts/resect_exact_mirror_import.py check
  .venv/bin/python scripts/resect_exact_mirror_import.py import

--evidence-root can point to the original isolated acquisition directory before
the archive is installed. --staging-root relocates only the 24 exact body paths.
No source is downloaded, decoded, reviewed or admitted by this command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import uuid

import resect_train_intake as authority
from real_intake_io import IntakeDeadline, atomic_preserve, check_deadline, termination_cleanup

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/annotations/resect-seg-v1'
EVIDENCE = ROOT / 'artifacts/resect-exact-mirror-acquisition-v1'
STAGING = ROOT / 'build/resect-exact-mirror-acquisition-v1'
ORIGINAL_PREFIX = Path('build/resect-exact-mirror-acquisition-v1')
REVISION = 'e86fb37dd93f7a9c64e48952f71410af59b04b9b'
MAX_SECONDS = 120
PINS = {
    'declaration.json': 'dd99a2898c257a4249f48d30bca1a633a975461942552631c19cdfc582dbfb6f',
    'summary.json': '0f3fea574d6fd10a893a9e65318b611fd940fa338eb6ed17e372b24ecb034c76',
    'source-snapshot.py': 'e590509d9efa746b2b3c0cf89f20479ea07d338d323691b0aca7ba7549fe29d4',
}
TRACE_INDEX_SHA = '9ff805a8f6c718023e400576f22d090e5c852cc403ec6f377fc87d11e3d1421d'
HELPERS = {
    'scripts/resect_train_intake.py': 'f1db114086aae742b1913aa552095daa57eb321bfeafe6b78bd84693508bcdaa',
    'scripts/acquire_resect_cavity.py': '6ece649ba346c9e7a0e93af462a91a9bf863728bea3fba83f5eccd681e1d0d51',
    'scripts/real_intake_io.py': '496821360b1daee79459952f970d299856baacad75fcc57ee589e55115036fd8',
}
CLAIMS = {'header_qc': 'not_run', 'geometry_qc': 'not_run', 'anatomy_qc': 'not_run',
          'training_admitted': False, 'spatial_planning_admitted': False,
          'decoded_array_bytes': 0, 'optimizer_updates': 0, 'recorded_rl_transitions': 0}
MIRROR_CLAIMS = {k: CLAIMS[k] for k in ('header_qc', 'geometry_qc', 'anatomy_qc',
                                     'training_admitted', 'spatial_planning_admitted', 'decoded_array_bytes')}
GOOD = {'installed_mirror_bytes_verified', 'existing_canonical_bytes_verified'}
Refusal = authority.Refusal
safe, read_small, encode, digest = authority.safe_path, authority.read_small, authority.encode, authority.digest


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_preserve(path, encode(value))


def execution_binding(deadline):
    files = {}
    for name, expected in HELPERS.items():
        module = sys.modules[Path(name).stem]
        if Path(module.__file__).resolve() != ROOT / name:
            raise Refusal('imported_helper_outside_checkout')
        raw = read_small(ROOT / name, deadline=deadline)
        if digest(raw) != expected:
            raise Refusal('reviewed_helper_changed')
        files[name] = expected
    files['scripts/resect_exact_mirror_import.py'] = digest(read_small(Path(__file__), deadline=deadline))
    return {'files': files, 'python': sys.version, 'executable': sys.executable}


def check_claims(record):
    if any(type(record.get(k)) is not type(v) or record[k] != v for k, v in MIRROR_CLAIMS.items()):
        raise Refusal('mirror_claims_changed')


def validate_row(row, original, pair, release):
    name = Path(original['path']).name
    case = pair['patient_group'].split(':')[1]
    mirror_path = f'dataset/{case}/{name}'
    if (row['source_id'] != original['id'] or row['source_authority'] != original
            or row['role'] != 'TRAIN' or pair['role'] != 'TRAIN'
            or row['patient_group'] != pair['patient_group'] or original['kind'] != 'cavity_annotation'
            or original['file_revision'] != 2 or row['source_release'] != release
            or row['mirror_commit'] != REVISION or row['mirror_path'] != mirror_path
            or row['mirror_url'] != f'https://huggingface.co/datasets/MedOtter/RESECT-SEG/resolve/{REVISION}/{mirror_path}'
            or row['expected_bytes'] != original['bytes'] or row['sha256'] != original['sha256']
            or row['md5'] != original['expected_md5']):
        raise Refusal('mirror_row_authority_role_or_version')


def prepare(evidence, staging, *, deadline):
    """Authenticate portable metadata and exactly 24 original source identities."""
    evidence, staging = safe(evidence), safe(staging)
    retained = {name: read_small(evidence / name, deadline=deadline) for name in PINS}
    if any(digest(retained[name]) != pin for name, pin in PINS.items()):
        raise Refusal('frozen_mirror_evidence_changed')
    declaration, summary = json.loads(retained['declaration.json']), json.loads(retained['summary.json'])
    check_claims(declaration)
    check_claims(summary)
    if (summary['status'] != 'all_24_bodies_verified' or summary['files'] != 24
            or summary['bytes_read'] != 1561749 or summary['expected_bytes'] != 1561749
            or summary['declaration_sha256'] != PINS['declaration.json']
            or declaration['source_sha256'] != PINS['source-snapshot.py']
            or declaration['mirror_is_third_party_transport_only'] is not True):
        raise Refusal('mirror_completion_binding')
    references = {}
    for name, expected in declaration['bindings'].items():
        value = read_small(evidence / 'references' / name, deadline=deadline)
        if digest(value) != expected:
            raise Refusal('retained_reference_changed')
        references[name] = value
    manifest = authority.require_manifest(deadline=deadline)
    if declaration['rights'] != manifest['rights_and_semantics']:
        raise Refusal('mirror_rights_changed')
    rights = manifest['rights_prerequisite']
    authority.verify_file(safe(DATA / rights['path']), rights, deadline=deadline)
    comparison = json.loads(references['build/resect-osf-bottleneck-investigation-v1/mirror-metadata-comparison.json'])
    expected_ids = {r['source_id'] for r in comparison['rows'] if r['currently_missing']}
    originals = {s['id']: s for s in manifest['sources']}
    pairs = {p['id']: p for p in manifest['pairs']}
    declared = declaration['rows']
    if (len(declared) != 24 or len(expected_ids) != 24
            or {r['source_id'] for r in declared} != expected_ids or set(summary['outcomes']) != expected_ids
            or sum(r['expected_bytes'] for r in declared) != 1561749):
        raise Refusal('exact_24_mask_scope')
    entries, traces = [], {}
    for row in declared:
        source_id = row['source_id']
        original = originals[source_id]
        validate_row(row, original, pairs[original['pair_id']], manifest['release_by_kind']['cavity_annotation'])
        receipt_relative = Path('objects') / source_id / 'attempt-01/receipt.json'
        outcome = summary['outcomes'][source_id]
        if outcome['receipt'] != str(ORIGINAL_PREFIX / receipt_relative) or outcome['status'] != 'downloaded_verified':
            raise Refusal('mirror_receipt_path_or_status')
        raw = read_small(evidence / receipt_relative, deadline=deadline)
        if digest(raw) != outcome['receipt_sha256']:
            raise Refusal('mirror_file_receipt_changed')
        receipt = json.loads(raw)
        check_claims(receipt)
        body_relative = Path('objects') / source_id / Path(original['path']).name
        if (receipt['source'] != row or receipt['source_id'] != source_id
                or receipt['source_binding'] != digest(encode(row))
                or receipt['status'] != 'downloaded_verified'
                or receipt['declaration_sha256'] != PINS['declaration.json']
                or receipt['path'] != str(ORIGINAL_PREFIX / body_relative)
                or receipt['bytes'] != original['bytes'] or receipt['sha256'] != original['sha256']
                or receipt['md5'] != original['expected_md5'] or receipt['retained_partial_bytes'] != 0):
            raise Refusal('mirror_file_receipt_binding')
        for name in ('intent.json', 'request-01.json', 'response-01.json', 'request-02.json', 'response-02.json'):
            relative = receipt_relative.parent / name
            traces[str(relative)] = digest(read_small(evidence / relative, deadline=deadline))
        entries.append({'source_id': source_id, 'source_authority': original,
                        'mirror_source': row, 'mirror_receipt_relative': str(receipt_relative),
                        'mirror_receipt_sha256': outcome['receipt_sha256'],
                        'staged_path': str(safe(staging / body_relative).relative_to(ROOT)),
                        'target_path': str(safe(DATA / original['path']).relative_to(ROOT))})
    if digest(encode(traces)) != TRACE_INDEX_SHA:
        raise Refusal('frozen_transport_trace_changed')
    return entries, {'authoritative_manifest_sha256': authority.MANIFEST_SHA,
                     'authoritative_cohort': manifest['cohort'], 'rights': rights,
                     'mirror_evidence': PINS, 'transport_trace_index_sha256': TRACE_INDEX_SHA,
                     'evidence_root': str(evidence.relative_to(ROOT)),
                     'staging_root': str(staging.relative_to(ROOT))}


def preflight(entries, *, deadline):
    """Verify all bodies and existing targets before any canonical publication."""
    results = []
    for entry in entries:
        source = entry['source_authority']
        staged = safe(ROOT / entry['staged_path'])
        authority.verify_file(staged, source, deadline=deadline)
        target = safe(ROOT / entry['target_path'])
        if target.exists():
            authority.verify_file(target, source, deadline=deadline)
        results.append({'source_id': entry['source_id'], 'staged_sha256': source['sha256'],
                        'bytes': source['bytes'], 'existing_target_verified': target.exists()})
    check_deadline(deadline)
    return results


def copy_verified(staged, partial, source, *, deadline):
    """Copy into a private partial; never share a mutable inode with staging."""
    staged, partial = safe(staged), safe(partial)
    total, sha, md5 = 0, hashlib.sha256(), hashlib.md5()
    with os.fdopen(os.open(staged, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as incoming:
        before = os.fstat(incoming.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != source['bytes']:
            raise Refusal('staged_size_or_type')
        with partial.open('xb') as output:
            while total <= source['bytes']:
                check_deadline(deadline)
                block = incoming.read(min(65536, source['bytes'] + 1 - total))
                if not block:
                    break
                total += len(block)
                if total > source['bytes']:
                    raise Refusal('copy_byte_cap')
                output.write(block)
                sha.update(block)
                md5.update(block)
            output.flush()
            os.fsync(output.fileno())
        after = os.fstat(incoming.fileno())
    if (total != source['bytes'] or authority.stable_stat(before) != authority.stable_stat(after)
            or authority.stable_stat(staged.lstat()) != authority.stable_stat(after)
            or sha.hexdigest() != source['sha256'] or md5.hexdigest() != source['expected_md5']):
        raise Refusal('staged_body_changed_or_fixity')
    authority.verify_file(partial, source, deadline=deadline)


def install_one(entry, trial, bindings, *, deadline):
    source, target = entry['source_authority'], safe(ROOT / entry['target_path'])
    directory = safe(trial / 'files' / entry['source_id'])
    directory.mkdir(parents=True, exist_ok=False)
    partial = safe(directory / 'body.partial')
    receipt = {**entry, 'import_binding_sha256': digest(encode(bindings)),
               'status': 'failed', 'canonical_published': False, 'started_utc': utc(),
               'acquisition_transport': 'third_party_huggingface_mirror',
               'original_osf_attempts_modified': False, **CLAIMS}
    save(directory / 'intent.json', receipt)
    try:
        if target.exists():
            authority.verify_file(target, source, deadline=deadline)
            receipt.update(status='existing_canonical_bytes_verified',
                           existing_bytes_original_transport='not_inferred_from_byte_equality')
        else:
            copy_verified(ROOT / entry['staged_path'], partial, source, deadline=deadline)
            safe(target.parent).mkdir(parents=True, exist_ok=True)
            check_deadline(deadline)
            os.link(safe(partial), safe(target))
            receipt['canonical_published'] = True
            fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            authority.verify_file(target, source, deadline=deadline)
            partial.unlink()
            receipt['status'] = 'installed_mirror_bytes_verified'
        check_deadline(deadline)
        receipt.update(verified_bytes=source['bytes'], sha256=source['sha256'], md5=source['expected_md5'])
    except BaseException as error:
        receipt.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                       error_type=type(error).__name__,
                       error_code=str(error) if isinstance(error, Refusal) else 'offline_install_failed')
        raise
    finally:
        receipt.update(finished_utc=utc(), retained_partial_bytes=partial.stat().st_size if partial.exists() else 0)
        save(directory / 'receipt.json', receipt)
    return receipt


def collect_receipts(trial, entries):
    outcomes = []
    binding_path = trial / 'bindings.json'
    binding_sha = digest(read_small(binding_path)) if binding_path.exists() else None
    for entry in entries:
        path = trial / 'files' / entry['source_id'] / 'receipt.json'
        if path.exists():
            raw = read_small(path)
            receipt = json.loads(raw)
            valid = (all(receipt.get(k) == v for k, v in entry.items())
                     and receipt.get('import_binding_sha256') == binding_sha
                     and all(type(receipt.get(k)) is type(v) and receipt[k] == v for k, v in CLAIMS.items())
                     and receipt.get('original_osf_attempts_modified') is False
                     and type(receipt.get('canonical_published')) is bool
                     and receipt.get('status') in GOOD | {'failed', 'interrupted'})
            if receipt.get('status') in GOOD:
                source = entry['source_authority']
                valid = (valid and receipt.get('verified_bytes') == source['bytes']
                         and receipt.get('sha256') == source['sha256']
                         and receipt.get('md5') == source['expected_md5'])
            outcomes.append({'source_id': entry['source_id'],
                             'status': receipt['status'] if valid else 'receipt_binding_failed',
                             'canonical_published': receipt.get('canonical_published'),
                             'receipt': str(path.relative_to(ROOT)), 'receipt_sha256': digest(raw)})
    return outcomes


def import_all(evidence=EVIDENCE, staging=STAGING):
    started = time.monotonic()
    deadline = started + MAX_SECONDS
    safe(DATA).mkdir(parents=True, exist_ok=True)
    with safe(DATA / 'pilot.lock').open('a+') as lock, termination_cleanup():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        trial = safe(DATA / 'mirror-import-v1/attempts' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid.uuid4().hex[:8]))
        trial.mkdir(parents=True, exist_ok=False)
        result = {'status': 'failed', 'started_utc': utc(), 'expected_files': 24,
                  'expected_bytes': 1561749, 'network_requests': 0,
                  'attempt': str(trial.relative_to(ROOT)), 'files': [], **CLAIMS}
        entries = []
        try:
            execution = execution_binding(deadline)
            for name, sha in execution['files'].items():
                raw = read_small(ROOT / name, deadline=deadline)
                if digest(raw) != sha:
                    raise Refusal('execution_changed')
                atomic_preserve(safe(trial / 'source-snapshot' / name), raw)
            bindings = {'execution': execution, 'pins': PINS, 'seconds': MAX_SECONDS,
                        'evidence_root': str(safe(evidence).relative_to(ROOT)),
                        'staging_root': str(safe(staging).relative_to(ROOT))}
            save(trial / 'intent.json', bindings)
            entries, source_evidence = prepare(evidence, staging, deadline=deadline)
            bindings['source_evidence'] = source_evidence
            save(trial / 'bindings.json', bindings)
            save(trial / 'preflight.json', preflight(entries, deadline=deadline))
            for entry in entries:
                install_one(entry, trial, bindings, deadline=deadline)
            if execution_binding(deadline) != execution:
                raise Refusal('execution_changed_during_import')
            check_deadline(deadline)
            result['status'] = 'all_24_mirror_bodies_imported'
        except BaseException as error:
            result.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                          error_type=type(error).__name__,
                          error_code=str(error) if isinstance(error, Refusal) else 'offline_import_failed')
        finally:
            # Bind every closed per-file record even if publication or bookkeeping
            # failed, so a future idempotent run can account for partial completion.
            result['files'] = collect_receipts(trial, entries)
            result['verified_files'] = sum(r['status'] in GOOD for r in result['files'])
            result['unresolved_source_ids'] = [e['source_id'] for e in entries
                if not any(r['source_id'] == e['source_id'] and r['status'] in GOOD for r in result['files'])]
            elapsed = time.monotonic() - started
            if elapsed >= MAX_SECONDS:
                result.update(status='timeout', deadline_exceeded=True)
            elif result['status'] == 'all_24_mirror_bodies_imported' and result['verified_files'] != 24:
                result['status'] = 'incomplete'
            result.update(finished_utc=utc(), elapsed_seconds=elapsed)
            save(trial / 'summary.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'import'))
    parser.add_argument('--evidence-root', type=Path, default=EVIDENCE)
    parser.add_argument('--staging-root', type=Path, default=STAGING)
    args = parser.parse_args()
    try:
        if args.command == 'check':
            deadline = time.monotonic() + MAX_SECONDS
            execution_binding(deadline)
            entries, bindings = prepare(args.evidence_root, args.staging_root, deadline=deadline)
            results = preflight(entries, deadline=deadline)
            print(json.dumps({'status': 'all_24_staged_bodies_verified', 'files': len(results),
                              'bytes': sum(r['bytes'] for r in results), 'bindings': bindings, **CLAIMS}))
            return 0
        result = import_all(args.evidence_root, args.staging_root)
        print(json.dumps(result, allow_nan=False))
        return 0 if result['status'] == 'all_24_mirror_bodies_imported' else 2
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({'status': 'refused', 'error_type': type(error).__name__,
                          'error_code': str(error) if isinstance(error, Refusal) else 'offline_import_refused'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
