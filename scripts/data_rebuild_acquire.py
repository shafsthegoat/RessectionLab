#!/usr/bin/env python3
"""Acquire bounded healthy TRAIN hosts, preserving frozen person roles.

All downloaded and temporary payloads live on the verified external volume.
This produces byte/header-checked candidates, never accepted application cases.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path('/Volumes/shafrir/old coding projects/RessectionLabData')
UUID = '9CC3623C-E38B-3EEE-B6BB-B9A06C193700'
sys.path.insert(0, str(REPO / 'src'))
from resectionlab.data_root import checked_root


def mount_check(root: Path, incoming: int = 0):
    checked_root(root, required_bytes=incoming, expected_uuid=UUID)
    info = plistlib.loads(subprocess.check_output(['diskutil', 'info', '-plist', '/Volumes/shafrir']))
    if (info.get('VolumeUUID', '').upper() != UUID or not os.path.ismount('/Volumes/shafrir')
            or info.get('MountPoint') != '/Volumes/shafrir' or info.get('Internal', True)
            or info.get('FilesystemType') != 'exfat'):
        raise RuntimeError('Expected writable external ExFAT volume is unavailable')
    if not root.is_dir() or not root.resolve().is_relative_to(Path('/Volumes/shafrir')):
        raise RuntimeError('Canonical root must already exist on the external volume')
    usage = shutil.disk_usage(root)
    reserve = max(50 * 1024**3, usage.total // 10)
    if usage.free - incoming < reserve:
        raise RuntimeError('External reserve would be breached')
    return {'uuid': UUID, 'free_bytes': usage.free, 'reserve_bytes': reserve}


def digest(path, kind='sha256'):
    h = hashlib.new(kind)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(root, path, value):
    mount_check(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def request_range(url, first, last, total):
    request = urllib.request.Request(url, headers={'Range': f'bytes={first}-{last}',
                                                  'Accept-Encoding': 'identity'})
    response = urllib.request.urlopen(request, timeout=60)
    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {first}-{last}/{total}':
        response.close()
        raise RuntimeError('Server refused exact byte range; full archive response refused')
    return response


def acquire_file(root, entry, destination, reusable):
    size = entry['bytes']
    mount_check(root, size * 2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = entry.get('expected_md5')
    local_sha = entry.get('expected_local_sha256')
    if destination.exists():
        if (destination.stat().st_size != size or (expected and digest(destination, 'md5') != expected)
                or (local_sha and digest(destination) != local_sha)):
            raise RuntimeError(f'Existing source differs: {destination}')
        return 'existing_verified'
    # Reuse only a complete exact-size and known-checksum original. Unknown IXI
    # local files cannot replace the independently pinned HTTP tar header route.
    if expected:
        for candidate in reusable.get(destination.name, []):
            if candidate.stat().st_size == size and digest(candidate, 'md5') == expected:
                shutil.copyfile(candidate, destination)
                return 'copied_identical_local_original'
    part = root / 'staging' / 'healthy-acquisition' / (hashlib.sha256(str(destination).encode()).hexdigest() + '.part')
    part.parent.mkdir(parents=True, exist_ok=True)
    if part.exists() and part.stat().st_size > size:
        raise RuntimeError('Oversized retained partial; manual inspection required')
    offset = part.stat().st_size if part.exists() else 0
    if 'tar_member' in entry:
        source = entry['tar_member']
        with request_range(entry['source_url'], source['header_offset'], source['header_offset'] + 511,
                           entry['archive_bytes']) as response:
            header = response.read(513)
        if len(header) != 512 or hashlib.sha256(header).hexdigest() != source['header_sha256']:
            raise RuntimeError('Pinned source TAR header mismatch')
        member = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict')
        if member.name != source['name'] or member.size != size or not member.isfile():
            raise RuntimeError('Unsafe or changed source TAR member')
    while offset < size:
        mount_check(root, size - offset)
        length = min(8 * 1024**2, size - offset)
        if 'tar_member' in entry:
            first = entry['tar_member']['header_offset'] + 512 + offset
            response = request_range(entry['source_url'], first, first + length - 1, entry['archive_bytes'])
        else:
            request = urllib.request.Request(entry['source_url'], headers={
                'Accept-Encoding': 'identity', 'Range': f'bytes={offset}-{offset+length-1}'})
            response = urllib.request.urlopen(request, timeout=60)
            if response.status == 206:
                if response.headers.get('Content-Range') != f'bytes {offset}-{offset+length-1}/{size}':
                    response.close()
                    raise RuntimeError('Unexpected S3 range')
            elif (response.status == 200 and offset == 0
                  and size <= entry.get('maximum_full_response_bytes', 64 * 1024**2)
                  and int(response.headers.get('Content-Length', size)) == size):
                length = size
            else:
                response.close()
                raise RuntimeError('Unsafe full response or unsupported resume')
        with response, part.open('ab') as out:
            remaining = length
            while remaining:
                block = response.read(min(1024**2, remaining))
                if not block:
                    raise RuntimeError('Truncated response; partial retained')
                out.write(block)
                remaining -= len(block)
                if remaining % (64 * 1024**2) < len(block):
                    mount_check(root, remaining)
            if response.read(1):
                raise RuntimeError('Response exceeded declared range')
        offset = part.stat().st_size
    if (part.stat().st_size != size or (expected and digest(part, 'md5') != expected)
            or (local_sha and digest(part) != local_sha)):
        raise RuntimeError('Source size/checksum mismatch; partial retained')
    mount_check(root)
    os.replace(part, destination)
    return 'downloaded_verified'


def header_qc(path):
    image = nib.load(path)
    if len(image.shape) != 3 or np.prod(image.shape) > 200_000_000:
        raise RuntimeError('Unexpected image dimensionality or excessive decoding size')
    affine = image.affine
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise RuntimeError('Invalid physical affine')
    # Full decode also verifies gzip integrity; one case/volume at a time.
    array = image.get_fdata(dtype=np.float32)
    if not np.isfinite(array).all() or np.ptp(array) <= 0:
        raise RuntimeError('Nonfinite or constant source image')
    q, qc = image.get_qform(coded=True)
    s, sc = image.get_sform(coded=True)
    return {'shape': list(image.shape), 'spacing_mm': list(map(float, image.header.get_zooms())),
            'affine_ras': affine.tolist(), 'axis_codes': list(nib.aff2axcodes(affine)),
            'units': list(image.header.get_xyzt_units()), 'qform_code': int(qc), 'sform_code': int(sc),
            'qform_sform_max_coefficient_difference': float(np.max(np.abs(q-s))) if qc and sc else None,
            'minimum': float(array.min()), 'maximum': float(array.max()),
            'nonzero_voxels': int(np.count_nonzero(array)), 'finite': True,
            'registration_status': 'unverified; shared subject is not shared grid'}


def candidates(dataset):
    if dataset == 'topbrain':
        path = REPO / 'manifests/experiments/native-vascular-graph-v4.json'
        yield {'dataset': dataset, 'subject': 'archive-v3', 'person_id': 'TopBrain:archive-v3',
               'role': 'mixed_role_opaque_archive_only', 'diagnosis': 'clinical_vascular_cohort_not_healthy',
               'license': 'Open use; source attribution; commercial use requires permission (not sought)',
               'source_url': 'https://zenodo.org/records/21972006',
               'prior_exposure_evidence': str(path.relative_to(REPO)),
               'limitations': ['25 paired CTA/MRA source people overlap TopCoW; not new independent cohorts',
                               'No tumor or structural tumor MRI established', 'Do not extract subject images before family roles recorded'],
               'files': [{'path': 'TopBrain_Data_Release_Batches1n2nTA36_081726.zip',
                          **json.loads((DEFAULT_ROOT / 'catalog/source-research-topbrain-file.json').read_text())}]}
    elif dataset == 'ixi':
        path = REPO / 'manifests/experiments/ixi-component-person-cohort-v1.json'
        cohort = json.loads(path.read_text())
        archives = {x['filename']: x for x in json.loads((REPO / 'artifacts/ixi-t1-mra-vessel-byte-intake-v1/source-contract.json').read_text())['raw_archives']}
        for member in cohort['members']:
            if member['role'] != 'TRAIN' or not {'T1', 'MRA'}.issubset(member['raw_files']):
                continue
            files = []
            for modality in ('T1', 'MRA'):
                raw = member['raw_files'][modality]
                archive = archives[raw['archive']]
                files.append({'path': raw['name'], 'bytes': raw['size'], 'source_url': archive['url'],
                              'archive_bytes': archive['bytes'], 'tar_member': raw, 'modality': modality})
            yield {'dataset': dataset, 'subject': member['subject'], 'person_id': member['person_group'],
                   'role': 'TRAIN', 'diagnosis': 'healthy_control', 'files': files,
                   'license': 'CC BY-SA 3.0', 'source_url': 'https://brain-development.org/ixi-dataset/',
                   'cohort_manifest': str(path.relative_to(REPO)), 'cohort_sha256': digest(path)}
    else:
        path = REPO / 'manifests/lausanne-component-cohort-v1.json'
        cohort = json.loads(path.read_text())
        controls = {m['subject'] for m in cohort['members'] if m['role'] == 'TRAIN' and m['group'] == 'control'}
        index = json.loads((REPO / 'manifests/lausanne-train-originals-v1.json').read_text())
        seen = set()
        for row in index['sessions']:
            if row['subject'] not in controls or row['subject'] in seen:
                continue
            seen.add(row['subject'])
            yield {'dataset': dataset, 'subject': row['subject'], 'session': row['session'],
                   'person_id': 'Lausanne:' + row['subject'], 'role': 'TRAIN', 'diagnosis': 'healthy_control',
                   'files': row['files'], 'license': 'CC0',
                   'git_commit': index['git_commit'], 'source_url': 'https://openneuro.org/datasets/ds003949/versions/1.0.1',
                   'cohort_manifest': str(path.relative_to(REPO)), 'cohort_sha256': digest(path)}


def run(args):
    root = args.data_root
    mount_check(root)
    journal_path = root / 'catalog/source-acquisition.json'
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else {
        'schema': 'data-rebuild-healthy-acquisition-v1', 'candidates': [], 'attempts': [],
        'acceptance': 'No case automatically accepted; independent source/spatial/app review required'}
    reusable = {}
    for base, _, names in os.walk(REPO / 'testcases/downloaded-datasets'):
        for name in names:
            if name.endswith('.nii.gz'):
                reusable.setdefault(name, []).append(Path(base) / name)
    rows = list(candidates(args.dataset))
    if args.subject:
        rows = [r for r in rows if r['subject'] in args.subject]
        if len(rows) != len(set(args.subject)):
            raise RuntimeError('Requested subject unavailable or outside frozen healthy TRAIN roles')
    rows = rows[:args.count]
    for row in rows:
        result = {**row, 'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'in_progress', 'acquired_files': []}
        journal['attempts'].append(result)
        save(root, journal_path, journal)
        try:
            for original_entry in row['files']:
                entry = dict(original_entry)
                destination = root / 'sources' / args.dataset / entry['path']
                relative = str(destination.relative_to(root))
                known = {f['sha256'] for previous in [*journal['candidates'], *journal['attempts']]
                         for f in previous.get('acquired_files', []) if f['path'] == relative}
                if len(known) > 1:
                    raise RuntimeError('Prior source receipts disagree; manual provenance review required')
                if known:
                    entry['expected_local_sha256'] = next(iter(known))
                if 'metadata_base64' in entry:
                    payload = base64.b64decode(entry['metadata_base64'])
                    mount_check(root)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists() and destination.read_bytes() != payload:
                        raise RuntimeError('Existing sidecar differs')
                    if not destination.exists():
                        destination.write_bytes(payload)
                    status = 'pinned_embedded_source_metadata'
                else:
                    status = acquire_file(root, entry, destination, reusable)
                item = {'path': str(destination.relative_to(root)), 'bytes': destination.stat().st_size,
                        'sha256': digest(destination), 'status': status,
                        'source_url': entry['source_url'], 'modality': entry.get('modality', 'T1' if '_T1w' in entry['path'] else 'MRA')}
                if entry.get('sha256') and item['sha256'] != entry['sha256']:
                    raise RuntimeError('Pinned sidecar checksum differs')
                if destination.name.endswith('.nii.gz'):
                    item['header_qc'] = header_qc(destination)
                result['acquired_files'].append(item)
                save(root, journal_path, journal)
            result['status'] = ('opaque_archive_bytes_validated_not_subject_reviewed' if args.dataset == 'topbrain'
                                else 'bytes_and_header_validated_awaiting_independent_review')
            result['limitations'] = row.get('limitations') or ['No CT or measured bone', 'No source tumor; healthy host',
                                    'TOF/MRA does not establish venous or complete small-vessel coverage',
                                    'Registration and anatomical suitability unreviewed']
            journal['candidates'] = [r for r in journal['candidates'] if r['person_id'] != row['person_id']] + [result]
        except Exception as exc:
            result.update(status='failed_or_partial', error=f'{type(exc).__name__}: {exc}')
            save(root, journal_path, journal)
            raise
        save(root, journal_path, journal)
        print(json.dumps({'subject': row['subject'], 'status': result['status'], 'files': [x['path'] for x in result['acquired_files']]}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--dataset', choices=['ixi', 'lausanne', 'topbrain'], required=True)
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--subject', action='append')
    run(parser.parse_args())
