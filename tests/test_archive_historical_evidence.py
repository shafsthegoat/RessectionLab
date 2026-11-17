"""Archive completeness, exact metadata, sparse restore and refusal boundaries."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('historical_archive', ROOT / 'scripts/archive_historical_evidence.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def fixture(tmp_path):
    source = tmp_path / 'source'
    (source / 'case').mkdir(parents=True)
    (source / 'case/data').write_bytes(b'original scientific bytes\n')
    (source / 'case/data').chmod(0o640)
    os.utime(source / 'case/data', ns=(1790000000123456789, 1790000000123456789))
    with (source / 'case/sparse').open('wb') as stream:
        stream.seek(4 * 1024**2)
        stream.write(b'end')
    (source / 'case/link').symlink_to('data')
    paths = ['case/data', 'case/sparse', 'case/link']
    rows = archive.inventory(source, paths)
    manifest = {'schema': archive.SCHEMA, 'members': rows}
    return source, rows, manifest


def write_tar(path, source, rows):
    with gzip.open(path, 'wb') as zipped:
        with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as stream:
            for row in rows:
                if row['type'] == 'file':
                    with (source / row['path']).open('rb') as content:
                        stream.addfile(archive.tar_metadata(row), content)
                else:
                    stream.addfile(archive.tar_metadata(row))


def test_complete_restore_preserves_hashes_modes_nanoseconds_links_and_sparse_data(tmp_path):
    source, rows, manifest = fixture(tmp_path)
    packed = tmp_path / 'archive.tar.gz'
    write_tar(packed, source, rows)
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    receipt = {'archiveSha256': archive.digest(packed), 'manifestSha256': archive.digest(manifest_path)}
    (tmp_path / 'receipt.json').write_text(json.dumps(receipt))
    restored = tmp_path / 'new-parent' / 'restored'
    result = archive.verify(packed, manifest_path, destination=restored)
    assert result['files'] == 2
    assert archive.inventory(restored, ['case/data', 'case/sparse', 'case/link']) == rows
    # Filesystems may allocate short seek-created holes eagerly. Preserve the
    # original allocation bound when this filesystem actually made it sparse.
    original_stat = (source / 'case/sparse').stat()
    if original_stat.st_blocks * 512 < original_stat.st_size // 2:
        assert (restored / 'case/sparse').stat().st_blocks <= original_stat.st_blocks + 2048
    with pytest.raises(ValueError, match='already exists'):
        archive.verify(packed, manifest_path, destination=restored)
    alias = tmp_path / 'destination-link'
    alias.symlink_to(restored)
    with pytest.raises(ValueError, match='already exists'):
        archive.verify(packed, manifest_path, destination=alias)
    packed.write_bytes(packed.read_bytes() + b'changed transport')
    with pytest.raises(ValueError, match='differs from receipt'):
        archive.verify(packed, manifest_path, destination=tmp_path / 'not-created')
    assert not (tmp_path / 'not-created').exists()


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'extra', 'mtime', 'hash'])
def test_archive_member_failures_refuse_before_source_removal(tmp_path, change):
    source, rows, manifest = fixture(tmp_path)
    members = [dict(row) for row in rows]
    if change == 'missing':
        members.pop()
    elif change == 'duplicate':
        members.append(members[-1])
    elif change == 'extra':
        members.append({'path': 'extra', 'mode': 0o755, 'mtimeNs': 123, 'type': 'directory'})
    elif change == 'mtime':
        members[-1]['mtimeNs'] += 1
    else:
        manifest['members'] = [dict(row) for row in rows]
        next(row for row in manifest['members'] if row['type'] == 'file')['sha256'] = '0' * 64
    packed = tmp_path / 'broken.tar.gz'
    write_tar(packed, source, members)
    with pytest.raises(ValueError):
        archive.consume(packed, manifest)
    assert (source / 'case/data').read_bytes() == b'original scientific bytes\n'


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'case/../escape', './case', 'case//data', 'case\\data', ''])
def test_unsafe_archive_names_rejected(name):
    with pytest.raises(ValueError, match='Unsafe'):
        archive.relative_name(name)


def test_manifest_symlink_parent_and_duplicate_paths_rejected(tmp_path):
    _, rows, manifest = fixture(tmp_path)
    manifest['members'] = rows + [dict(rows[0])]
    with pytest.raises(ValueError, match='Duplicate'):
        archive.manifest_rows(manifest)
    manifest['members'] = [{'path': 'parent', 'type': 'symlink', 'target': '/outside'},
                           {'path': 'parent/escape', 'type': 'file'}]
    with pytest.raises(ValueError, match='parent'):
        archive.manifest_rows(manifest)
