"""Prepared NFBS archive names/sizes/types inventory. No extraction or image API.

Gzip is sequential: skipping tar contents still inflates those bytes. This reader
counts all decompressed bytes and stores only bounded member metadata. Actual
archive opening is disabled until a separately hash-bound protocol is released.
"""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import io
import json
import os
import signal
import stat
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = 'data/acquisition/nfbs-originals-v1/verified/NFBS_Dataset.tar.gz'
ARCHIVE_SHA = 'fd616b9ea21aad3d0d4cd05a25a71f4a3c251140df38188e58e499254440d2cc'
ARCHIVE_MD5 = '06c467ac9ab6cbdb8185e6643879721d'
ARCHIVE_SIZE = 1751464473


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(1024**2):
            h.update(b)
    return h.hexdigest()


def deadline_check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('inventory_wall_deadline')


class CountedInflate:
    def __init__(self, stream, cap, deadline):
        self.stream, self.cap, self.deadline = stream, cap, deadline
        self.bytes = 0

    def read(self, size):
        deadline_check(self.deadline)
        if size < 0 or size > 1024**2:
            raise ValueError('unbounded_decompression_read')
        # At most one byte beyond cap is requested to distinguish exact EOF.
        body = self.stream.read(min(size, self.cap - self.bytes + 1))
        self.bytes += len(body)
        if self.bytes > self.cap:
            raise ValueError('uncompressed_byte_cap')
        deadline_check(self.deadline)
        return body


class BoundedTarInfo(tarfile.TarInfo):
    @classmethod
    def fromtarfile(cls, archive):
        # TarFile.next normally tolerates malformed later headers. Convert all
        # non-end header errors to a fatal type it does not silently consume.
        header = archive.fileobj.read(tarfile.BLOCKSIZE)
        if header == bytes(tarfile.BLOCKSIZE):
            second = archive.fileobj.read(tarfile.BLOCKSIZE)
            if second != bytes(tarfile.BLOCKSIZE):
                raise ValueError('missing_second_zero_end_record')
            archive.strict_end_records_seen = True
            raise tarfile.EOFHeaderError('two_zero_end_records_verified')
        try:
            member = cls.frombuf(header, archive.encoding, archive.errors)
        except tarfile.HeaderError as error:
            raise ValueError('invalid_or_missing_tar_header:' + type(error).__name__) from error
        member.offset = archive.fileobj.tell() - tarfile.BLOCKSIZE
        return member._proc_member(archive)

    def _proc_member(self, archive):
        # Reject GNU/PAX/sparse extension records before tarfile may read an
        # attacker-sized extension body. Ordinary names fit the fixed header.
        if self.type not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE):
            raise ValueError('unsupported_link_special_or_sparse_member')
        return super()._proc_member(archive)


def inventory_archive(handle, *, expected_sha256, expected_md5, expected_bytes, limits):
    start = time.monotonic()
    deadline = start + limits['wall_seconds']
    before = os.fstat(handle.fileno())
    if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
        raise ValueError('original_size_or_type')
    # Two sequential bounded passes: original fixity, then names/sizes only.
    h, m, compressed = hashlib.sha256(), hashlib.md5(), 0
    while chunk := handle.read(1024**2):
        compressed += len(chunk)
        if compressed > expected_bytes:
            raise ValueError('compressed_byte_cap')
        h.update(chunk)
        m.update(chunk)
        deadline_check(deadline)
    if compressed != expected_bytes or h.hexdigest() != expected_sha256 or m.hexdigest() != expected_md5:
        raise ValueError('original_checksum')
    handle.seek(0)
    members, seen, declared_bytes = [], set(), 0
    with gzip.GzipFile(fileobj=handle, mode='rb') as gz:
        stream = CountedInflate(gz, limits['uncompressed_bytes'], deadline)
        with tarfile.open(fileobj=stream, mode='r|', bufsize=10240, tarinfo=BoundedTarInfo) as archive:
            for member in archive:
                deadline_check(deadline)
                name = member.name
                parts = PurePosixPath(name).parts
                if len(name.encode('utf-8')) > limits['member_name_bytes'] or not parts or name.startswith('/') or '..' in parts or '\\' in name:
                    raise ValueError('unsafe_or_overlong_member_name')
                if name in seen:
                    raise ValueError('duplicate_member_name')
                seen.add(name)
                if len(seen) > limits['members']:
                    raise ValueError('member_count_cap')
                if not (member.isfile() or member.isdir()) or member.issparse():
                    raise ValueError('unsupported_link_special_or_sparse_member')
                if member.pax_headers:
                    raise ValueError('pax_extension_requires_separate_review')
                if member.size < 0 or member.size > limits['single_member_bytes']:
                    raise ValueError('single_member_size_cap')
                declared_bytes += member.size
                if declared_bytes > limits['uncompressed_bytes']:
                    raise ValueError('declared_member_bytes_cap')
                members.append({'name': name, 'bytes': member.size, 'type': 'file' if member.isfile() else 'directory'})
                # Do not call extractfile/extract/load/read image data. Streaming
                # tar next() skips bodies through bounded CountedInflate.read().
            if not getattr(archive, 'strict_end_records_seen', False):
                raise ValueError('missing_tar_end_records')
            # Drain through tarfile's _Stream before close: it may hold bytes
            # read ahead beyond the end records. Bypassing it loses that tail.
            while extra := archive.fileobj.read(10240):
                if any(extra):
                    raise ValueError('nonzero_unparsed_tar_trailer')
    after = os.fstat(handle.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('original_changed_during_inventory')
    return {'status': 'member_metadata_only', 'archive_sha256': h.hexdigest(), 'archive_md5': m.hexdigest(), 'archive_bytes': compressed,
            'members': members, 'member_count': len(members), 'declared_member_bytes': declared_bytes, 'decompressed_bytes_traversed': stream.bytes,
            'wall_seconds': time.monotonic() - start, 'extracted_files': 0, 'image_headers_decoded': 0, 'image_arrays_decoded': 0,
            'interpretation': 'Only tar member names, byte sizes and file/directory type. All intervening gzip bytes are inflated opaquely. No anatomy, label validity, subject independence or admission follows.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--protocol-sha256', required=True)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    assert sha(a.protocol) == a.protocol_sha256
    d = json.loads(a.protocol.read_text())
    assert sha(Path(__file__)) == d['reader_sha256']
    assert d['archive'] == ARCHIVE and d['archive_sha256'] == ARCHIVE_SHA and d['archive_md5'] == ARCHIVE_MD5 and d['archive_bytes'] == ARCHIVE_SIZE
    assert sha(ROOT / d['cohort_path']) == d['cohort_sha256']
    assert sys.implementation.name == 'cpython' and sys.version.split()[0] == d['python_version']
    for name, module in [('tarfile', tarfile), ('gzip', gzip)]:
        assert sha(Path(module.__file__)) == d['stdlib_source_sha256'][name]
    assert not d['extraction_allowed'] and not d['image_decode_allowed'] and not d['training_admitted']
    if a.check_only:
        print(json.dumps({'status': 'prepared_protocol_valid', 'archive_opens': 0, 'execution_released': d['execution_released']}))
        return
    assert d['execution_released'] is True
    limits = d['limits']
    assert limits == {'members': 2048, 'member_name_bytes': 1024, 'single_member_bytes': 1024**3, 'uncompressed_bytes': 64*1024**3, 'wall_seconds': 120}
    target = ROOT / d['output']
    assert target.is_relative_to(ROOT / 'build') and not target.exists()
    def alarm(signum, frame):
        raise TimeoutError('inventory_process_deadline')
    prior = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, limits['wall_seconds'])
    try:
        path = ROOT / ARCHIVE
        assert not path.is_symlink()
        with path.open('rb') as f:
            result = inventory_archive(f, expected_sha256=ARCHIVE_SHA, expected_md5=ARCHIVE_MD5, expected_bytes=ARCHIVE_SIZE, limits=limits)
        result.update(protocol_sha256=a.protocol_sha256, reader_sha256=sha(Path(__file__)), finished_utc=datetime.now(timezone.utc).isoformat())
        payload=(json.dumps(result,indent=2)+'\n').encode()
        assert len(payload)<=4*1024**2
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:
            f.write(payload);f.flush();os.fsync(f.fileno())
        print(json.dumps({'status':result['status'],'member_count':result['member_count'],'receipt':str(target)}))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,prior)


if __name__=='__main__':
    main()
