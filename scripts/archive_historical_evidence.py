#!/usr/bin/env python3
"""Exact local PAX archives and overwrite-refusing restoration; never removes originals."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tarfile

SCHEMA = 'resectionlab.historical-evidence.v1'
CHUNK = 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative_name(value):
    path = PurePosixPath(value)
    require(value and not path.is_absolute() and '..' not in path.parts
            and str(path) == value and '\\' not in value, 'Unsafe archive path: ' + value)
    return value


def metadata(path, name):
    st = path.lstat()
    row = {'path': relative_name(name), 'mode': stat.S_IMODE(st.st_mode), 'mtimeNs': st.st_mtime_ns}
    if stat.S_ISREG(st.st_mode):
        row.update(type='file', bytes=st.st_size, sha256=digest(path))
    elif stat.S_ISLNK(st.st_mode):
        row.update(type='symlink', target=os.readlink(path))
    elif stat.S_ISDIR(st.st_mode):
        row.update(type='directory')
    else:
        raise ValueError('Unsupported evidence type: ' + name)
    now = path.lstat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_mode, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    require(identity(now) == identity(st), 'Evidence changed while inventoried: ' + name)
    return row


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def inventory(root, paths):
    names = sorted(relative_name(name) for name in paths)
    require(len(names) == len(set(names)), 'Duplicate source path')
    rows = {}
    for name in names:
        for parent in PurePosixPath(name).parents:
            if str(parent) == '.':
                continue
            key = str(parent)
            if key not in rows:
                row = metadata(root / key, key)
                require(row['type'] == 'directory', 'Parent is not a real directory: ' + key)
                rows[key] = row
        row = metadata(root / name, name)
        require(row['type'] != 'directory', 'List explicit files/links, not recursive directories: ' + name)
        rows[name] = row
    return sorted(rows.values(), key=lambda row: (len(PurePosixPath(row['path']).parts), row['path']))


def tar_metadata(row):
    info = tarfile.TarInfo(row['path'])
    info.mode = row['mode']
    seconds, nanos = divmod(row['mtimeNs'], 10**9)
    info.mtime = seconds
    info.pax_headers = {'mtime': f'{seconds}.{nanos:09d}'}
    info.uid = info.gid = 0
    if row['type'] == 'directory':
        info.type = tarfile.DIRTYPE
    elif row['type'] == 'symlink':
        info.type, info.linkname = tarfile.SYMTYPE, row['target']
    else:
        info.size = row['bytes']
    return info


def manifest_rows(manifest):
    require(manifest.get('schema') == SCHEMA, 'Unknown archive manifest schema')
    rows = {relative_name(row['path']): row for row in manifest['members']}
    require(len(rows) == len(manifest['members']), 'Duplicate manifest member')
    for name, row in rows.items():
        require(row['type'] in ('file', 'directory', 'symlink'), 'Unsupported manifest type')
        for parent in PurePosixPath(name).parents:
            if str(parent) != '.':
                require(rows.get(str(parent), {}).get('type') == 'directory', 'Unsafe or absent parent: ' + name)
    return rows


def consume(archive, manifest, destination=None):
    rows, seen = manifest_rows(manifest), set()
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            name = relative_name(member.name.rstrip('/') if member.isdir() else member.name)
            require(name in rows and name not in seen, 'Extra or duplicate archive member: ' + name)
            seen.add(name)
            row, expected = rows[name], tar_metadata(rows[name])
            require(member.type == expected.type and member.mode == row['mode'], 'Archive type/mode differs: ' + name)
            require(member.pax_headers.get('mtime') == expected.pax_headers['mtime'], 'Archive nanosecond mtime differs: ' + name)
            target = destination / name if destination is not None else None
            if row['type'] == 'directory':
                if target is not None:
                    target.mkdir()
            elif row['type'] == 'symlink':
                require(member.linkname == row['target'], 'Archive link differs: ' + name)
                if target is not None:
                    target.symlink_to(row['target'])
            else:
                require(member.size == row['bytes'], 'Archive size differs: ' + name)
                hashed, size = hashlib.sha256(), 0
                source = stream.extractfile(member)
                output = target.open('xb') if target is not None else None
                try:
                    while chunk := source.read(CHUNK):
                        hashed.update(chunk)
                        size += len(chunk)
                        if output is not None:
                            if not chunk.strip(b'\0'):
                                output.seek(len(chunk), os.SEEK_CUR)
                            else:
                                output.write(chunk)
                    if output is not None:
                        output.truncate(size)
                finally:
                    if output is not None:
                        output.close()
                    source.close()
                require(size == row['bytes'] and hashed.hexdigest() == row['sha256'], 'Archive hash differs: ' + name)
    require(seen == set(rows), 'Archive is missing members: ' + str(sorted(set(rows) - seen)[:3]))
    if destination is not None:
        for row in reversed(manifest['members']):
            path = destination / row['path']
            if row['type'] != 'symlink':
                path.chmod(row['mode'])
            os.utime(path, ns=(row['mtimeNs'], row['mtimeNs']), follow_symlinks=False)
        for row in manifest['members']:
            require(metadata(destination / row['path'], row['path']) == row, 'Restored inventory differs: ' + row['path'])
    return {'members': len(rows), 'files': sum(row['type'] == 'file' for row in rows.values()),
            'contentBytes': sum(row.get('bytes', 0) for row in rows.values())}


def build(root, paths_file, output, data_root, revision, preservation_ref):
    root, output, data_root = Path(root).resolve(), Path(output).absolute(), Path(data_root).resolve(strict=True)
    identity = json.loads((data_root / '.ressectionlab-data-root.json').read_text())
    mount = Path(identity['mount_point']).resolve(strict=True)
    require(os.path.ismount(mount) and data_root.stat().st_dev == mount.stat().st_dev
            and data_root.stat().st_dev != root.stat().st_dev, 'Expected independent external volume is not mounted')
    require(output.parent.resolve(strict=True).is_relative_to(data_root / 'archive'), 'Output must use existing external archive area')
    require(not output.exists() and not output.is_symlink(), 'Archive output already exists')
    rows = inventory(root, Path(paths_file).read_text().splitlines())
    capacity = os.statvfs(data_root)
    require(capacity.f_bavail * capacity.f_frsize > sum(row.get('bytes', 0) for row in rows) + CHUNK * 32,
            'Insufficient external capacity')
    actual_revision = subprocess.check_output(['git', 'rev-parse', revision + '^{commit}'], cwd=root, text=True).strip()
    actual_ref = subprocess.check_output(['git', 'rev-parse', preservation_ref + '^{commit}'], cwd=root, text=True).strip()
    require(actual_revision == actual_ref, 'Preservation reference differs from source revision')
    manifest = {'schema': SCHEMA, 'sourceRevision': actual_revision, 'preservationRef': preservation_ref,
                'sourceRoot': str(root), 'externalRootIdentity': identity, 'members': rows,
                'scope': 'Exact listed bytes/links, mode and nanosecond mtime. Ownership restores as current user. Originals are never removed.'}
    output.mkdir(mode=0o700)
    manifest_path, archive = output / 'manifest.json', output / 'archive.tar.gz'
    write_json(manifest_path, manifest)
    with archive.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as stream:
            for row in rows:
                path = root / row['path']
                if row['type'] != 'directory':
                    require(metadata(path, row['path']) == row, 'Source changed before archive write: ' + row['path'])
                else:
                    require(path.is_dir() and not path.is_symlink(), 'Source parent changed type: ' + row['path'])
                if row['type'] == 'file':
                    with path.open('rb') as source:
                        stream.addfile(tar_metadata(row), source)
                else:
                    stream.addfile(tar_metadata(row))
                if row['type'] != 'directory':
                    require(metadata(path, row['path']) == row, 'Source changed during archive write: ' + row['path'])
    with archive.open('rb') as stream:
        os.fsync(stream.fileno())
    checked = consume(archive, manifest)
    receipt = {'schema': SCHEMA, 'archiveSha256': digest(archive), 'archiveBytes': archive.stat().st_size,
               'manifestSha256': digest(manifest_path), 'sourceRevision': actual_revision,
               'preservationRef': preservation_ref, 'verified': checked}
    write_json(output / 'receipt.json', receipt)
    return receipt


def verify(archive, manifest_path, receipt_path=None, destination=None):
    archive, manifest_path = Path(archive), Path(manifest_path)
    receipt = json.loads(Path(receipt_path or manifest_path.with_name('receipt.json')).read_text())
    require(digest(archive) == receipt['archiveSha256'] and digest(manifest_path) == receipt['manifestSha256'],
            'Archive/manifest differs from receipt; check versioned archive index')
    manifest = json.loads(manifest_path.read_text())
    if destination is None:
        return consume(archive, manifest)
    destination = Path(destination).absolute()
    require(not destination.exists() and not destination.is_symlink(), 'Restore destination already exists; refusing overwrite')
    consume(archive, manifest)
    destination.mkdir(mode=0o700, parents=True)
    return consume(archive, manifest, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('build')
    for name in ('root', 'paths-file', 'output', 'data-root', 'revision', 'preservation-ref'):
        create.add_argument('--' + name, required=True)
    for name in ('verify', 'restore'):
        command = commands.add_parser(name)
        command.add_argument('--archive', required=True, type=Path)
        command.add_argument('--manifest', required=True, type=Path)
        command.add_argument('--receipt', type=Path)
        if name == 'restore':
            command.add_argument('--destination', required=True, type=Path)
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'build':
        result = build(**args)
    else:
        args['manifest_path'] = args.pop('manifest')
        args['receipt_path'] = args.pop('receipt')
        result = verify(**args)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
