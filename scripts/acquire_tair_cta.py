#!/usr/bin/env python3
"""Acquire selected original CTA studies from TAIR's public S3 release."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from resectionlab.data_root import atomic_json, checked_root, safe_path

RELEASE = 'https://tair.kitware.com/clinical-data/'
BUCKET = 'https://multimodal-imaging-for-cerebrovascular-disease.s3.us-east-1.amazonaws.com'
PILOTS = {
    '00273': (55_307_282, '8640eafd5183c7ae943c2b5d37f9cf84-7', '6xY4r6eG2Uc='),
    '00244': (59_742_717, '326ee38ff765834120e1a35a42f87b93-8', '8t4gVTdwkjg='),
    '00410': (68_387_614, '788e35118c7ae6e95b5949e43fbc4214-9', 'IVlt2w0Sx48='),
}
FILENAME = 'CTA-H_defaced.nii.gz'
CHUNK = 1024**2


def checksums(path):
    # Reflected CRC-64/NVME polynomial, with all-one initialization and xorout.
    table = []
    for value in range(256):
        for _ in range(8):
            value = (value >> 1) ^ (0x9A6C9329AC4BC9B5 if value & 1 else 0)
        table.append(value)
    crc = initial = (1 << 64) - 1
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(CHUNK):
            sha.update(chunk)
            for byte in chunk:
                crc = table[(crc ^ byte) & 255] ^ (crc >> 8)
    return base64.b64encode((crc ^ initial).to_bytes(8, 'big')).decode(), sha.hexdigest()


def transfer(url, etag, size, partial):
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError(f'Oversized retained partial: {partial}')
    if offset == size:
        return 0
    quoted_etag = f'"{etag}"'
    request = Request(url, headers={
        'Accept-Encoding': 'identity', 'If-Match': quoted_etag,
        'Range': f'bytes={offset}-{size - 1}',
    })
    with urlopen(request, timeout=45) as response:
        if (response.geturl() != url or response.status != 206
                or response.headers.get('Content-Range') != f'bytes {offset}-{size - 1}/{size}'
                or response.headers.get('ETag') != quoted_etag
                or response.headers.get('Content-Encoding', 'identity') != 'identity'):
            raise ValueError('S3 response differs from the pinned CTA object range')
        with partial.open('ab') as dst:
            received = 0
            while chunk := response.read(CHUNK):
                received += len(chunk)
                if offset + received > size:
                    raise ValueError('S3 response exceeds the pinned object length')
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
    if offset + received != size:
        raise ValueError('S3 transfer truncated; partial retained for resumption')
    return received


def acquire(root, subject):
    size, etag, expected_crc = PILOTS[subject]
    url = f'{BUCKET}/{subject}/SCANS/{FILENAME}'
    source = Path('sources/tair-v1-cta-selective') / subject
    path = safe_path(root, source / FILENAME)
    partial = safe_path(root, source / f'{FILENAME}.partial')
    receipt = safe_path(root, source / 'acquisition.json')
    checked_root(root, required_bytes=size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / '.acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        retained = path if path.exists() else partial
        received = transfer(url, etag, size, partial) if retained == partial else 0
        crc, sha = checksums(retained)
        if retained.stat().st_size != size or crc != expected_crc:
            raise ValueError(f'CTA differs from the publisher length or CRC64NVME: {retained}')
        result = {
            'schema': 'ressectionlab-tair-cta-source-v1',
            'dataset': 'TAIR Multimodal Imaging Dataset for Cerebrovascular Disease v1.0',
            'publisher_release': RELEASE, 'license': 'CC-BY-4.0', 'role': 'DEVELOPMENT',
            'source_person_family': f'TAIR:{subject}', 'source_url': url,
            'path': str(path.relative_to(root)), 'bytes': size, 's3_etag': etag,
            'publisher_crc64nvme': crc, 'sha256': sha, 'defaced': True,
            'available_modalities': ['CTA'], 'supplied_segmentations': [],
        }
        if receipt.exists() and json.loads(receipt.read_text()) != result:
            raise ValueError(f'Existing acquisition receipt differs for TAIR:{subject}')
        checked_root(root)
        if retained == partial:
            partial.replace(path)
        if not receipt.exists():
            atomic_json(receipt, result)
    return {**result, 'new_transport_bytes': received}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, help='Verified external root. Defaults to RESSECTIONLAB_DATA_ROOT.')
    parser.add_argument('subjects', nargs='*', help='Pinned pilot IDs. Omit for all three.')
    args = parser.parse_args()
    subjects = args.subjects or PILOTS
    if set(subjects) - PILOTS.keys() or len(subjects) != len(set(subjects)):
        parser.error('Choose distinct pilot IDs from ' + ', '.join(PILOTS))
    root = checked_root(args.root)
    for subject in subjects:
        print(json.dumps(acquire(root, subject)), flush=True)


if __name__ == '__main__':
    main()
