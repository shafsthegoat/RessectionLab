#!/usr/bin/env python3
"""Retain only IXI265 from the official gzip-compressed diffusion tar stream.

No archive is persisted. Existing complete files are reused after hash checking.
The receipt records member hashes, not a claim of whole-archive verification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request

URL = 'https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-DTI.tar'
SUBJECT = 'IXI265-Guys-0845'


def selected(name):
    name = Path(name).name
    return name in {f'{SUBJECT}-DTI-{i:02d}.nii.gz' for i in range(16)}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Counted:
    def __init__(self, stream):
        self.stream, self.bytes, self.notice = stream, 0, 0

    def read(self, size):
        data = self.stream.read(size)
        self.bytes += len(data)
        if self.bytes - self.notice >= 64 * 1024**2:
            print(f'Archive prefix streamed: {self.bytes / 1024**2:.0f} MiB; archive not stored', flush=True)
            self.notice = self.bytes
        return data


def acquire(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'acquisition.json'
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if len(record['members']) == 16 and all(sha(output / r['name']) == r['sha256'] for r in record['members']):
            print('Reused all 16 verified cached diffusion members')
            return record
        raise ValueError('Cached diffusion receipt mismatch; preserve and inspect it')
    if shutil.disk_usage(output).free < 2 * 1024**3:
        raise ValueError('Keep at least 2 GiB free before acquisition')
    start = time.monotonic()
    members = []
    request = urllib.request.Request(URL, headers={'Range': 'bytes=0-', 'Accept-Encoding': 'identity'})
    with urllib.request.urlopen(request, timeout=90) as response:
        content_range = response.headers.get('Content-Range')
        if response.status == 206 and not (content_range or '').startswith('bytes 0-'):
            raise ValueError('Unexpected archive range')
        counted = Counted(response)
        with tarfile.open(fileobj=counted, mode='r|gz', bufsize=1024 * 1024) as archive:
            for member in archive:
                if not selected(member.name):
                    continue
                if not member.isfile() or not 0 < member.size < 4 * 1024**2:
                    raise ValueError('Unexpected selected member type or size')
                name = Path(member.name).name
                if name in {r['name'] for r in members}:
                    raise ValueError('Duplicate selected diffusion member')
                target = output / name
                partial = target.with_suffix(target.suffix + '.partial')
                with archive.extractfile(member) as source, partial.open('wb') as destination:
                    shutil.copyfileobj(source, destination, 1024 * 1024)
                if partial.stat().st_size != member.size:
                    raise ValueError('Truncated diffusion member')
                digest = sha(partial)
                if target.exists():
                    if sha(target) != digest:
                        raise ValueError('Distinct cached diffusion bytes; refusing overwrite')
                    partial.unlink()
                else:
                    partial.replace(target)
                members.append({'name': name, 'tar_member': member.name, 'bytes': member.size, 'sha256': digest})
                print(f'Retained {name} ({len(members)}/16)', flush=True)
                if len(members) == 16:
                    break
        record = {'subject': SUBJECT, 'source_url': URL, 'source_page': 'https://brain-development.org/ixi-dataset/',
                  'license': 'CC-BY-SA-3.0', 'etag': response.headers.get('ETag'),
                  'last_modified': response.headers.get('Last-Modified'), 'streamed_prefix_bytes': counted.bytes,
                  'archive_persisted': False, 'whole_archive_checksum_verified': False,
                  'verification': 'Tar header checksums, exact subject/member names, member sizes and local SHA256; gzip trailer not reached',
                  'members': sorted(members, key=lambda r: r['name']), 'elapsed_seconds': time.monotonic() - start}
    if len(members) != 16:
        raise ValueError('Subject does not have all 16 published diffusion volumes')
    for name in ('bvals.txt', 'bvecs.txt'):
        url = URL.rsplit('/', 1)[0] + '/' + name
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(65537)
        if len(data) > 65536:
            raise ValueError('Oversized gradient file')
        target = output / name
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Distinct existing gradient file')
        target.write_bytes(data)
        record[name] = {'source_url': url, 'sha256': sha(target), 'bytes': len(data)}
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('testcases/downloaded-datasets/IXI265-Guys-0845/diffusion'))
    print(json.dumps(acquire(parser.parse_args().output), indent=2))
