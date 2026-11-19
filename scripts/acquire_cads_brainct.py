#!/usr/bin/env python3
"""Retain one CADS BrainCT-1mm image and its published cranial segmentation parts."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import subprocess

from data_rebuild_acquire import UUID
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file

REPO = 'https://huggingface.co/datasets/huggingface/CADS-dataset'
REVISION = 'eee3c0a993636264fd04104f7e2a090c507c0c43'
LABEL_REVISION = '120d6789d5960a05c25e7fb58e41436364e25804'
SUBSET = '0042_new_brainct_1mm'
BASENAME = 'brain_ct_465_BEYIN_YENI_BEYIN_20190129011652_5'
FAMILY = 'CADS-BrainCT-1mm:465'
SOURCE = 'sources/cads-brainct-1mm/465'
SCHEMA = 'cads-brainct-1mm-selective-v1'
LICENSE = 'CC-BY-4.0'
ASSETS = {
    'ncct': (f'{REPO}/resolve/{REVISION}/{SUBSET}/images/{BASENAME}.nii.gz',
             92269250, '4b0bbc34af05b29486b905b5732d53e5aced83e86d0764d753ba20d0a93861fb'),
    'readme': (f'{REPO}/raw/{REVISION}/{SUBSET}/README_0042_new_brainct_1mm.md',
               1663, '1d0998eba25b6343ad3ca8ca56dba23e1c2ae85ee8750babfa1c281d580bb4ae'),
    'image_index': (f'{REPO}/raw/{REVISION}/{SUBSET}/{SUBSET}.csv',
                    44801, '3e38f4ed0f21b062eb747a5b3062667c36acecebd1152a147c3c5edb86676cca'),
    'label_map': (f'https://raw.githubusercontent.com/murong-xu/CADS/{LABEL_REVISION}/resources/info/labelmap.md',
                  8096, 'f533528fe2a3a9a2fb6738de06bc232ee25d11f0e007cad054bed0b6b03b38d0'),
}
PARTS = {
    553: (620602, '59a5b44273ba3cc51a0779d56831341903b8971eafb61e214b194093010df6cd'),
    557: (1486144, 'a23d4b74db35fef6f1d94263b1df7fcc7f95ed0b992b3dbbdaf47cabbba57e06'),
    558: (362634, '2206594ce5392a44bfe17f4cbc94a7d2b173da91aed10b7890157c871421144c'),
}
ASSETS.update({f'part_{number}': (
    f'{REPO}/resolve/{REVISION}/{SUBSET}/segmentations/{BASENAME}/{BASENAME}_part_{number}.nii.gz', size, digest,
) for number, (size, digest) in PARTS.items()})


def checked_member(path, size, digest):
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise ValueError(f'Publisher byte count or SHA-256 differs for {path}')


def acquire(root):
    root = checked_root(root, expected_uuid=UUID, required_bytes=2 * 1024**3)
    directory = safe_path(root, SOURCE)
    directory.mkdir(parents=True, exist_ok=True)
    members = {}
    with (directory / '.acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for key, (url, size, digest) in ASSETS.items():
            path = safe_path(root, Path(SOURCE) / url.rsplit('/', 1)[-1])
            if not path.exists():
                partial = path.with_name(path.name + '.part')
                if not partial.exists() or partial.stat().st_size < size:
                    subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
                                    '--connect-timeout', '15', '--max-time', '300',
                                    '--max-filesize', str(size), '--continue-at', '-',
                                    '--output', str(partial), url], check=True)
                checked_member(partial, size, digest)
                partial.rename(path)
            checked_member(path, size, digest)
            members[key] = {'path': str(path.relative_to(root)), 'url': url, 'bytes': size, 'sha256': digest}
            print(f'Verified {key} ({size:,} bytes)', flush=True)
        receipt = directory / 'acquisition.json'
        atomic_json(receipt, {'schema': SCHEMA, 'source_family': FAMILY, 'role': 'DEVELOPMENT',
                              'license': LICENSE, 'publisher_repository': REPO,
                              'revision': REVISION, 'label_map_revision': LABEL_REVISION,
                              'source_case_basename': BASENAME, 'retained_bytes': sum(r['bytes'] for r in members.values()),
                              'members': members})
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    print(acquire(parser.parse_args().root))
