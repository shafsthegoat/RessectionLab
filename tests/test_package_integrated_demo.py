"""Small fixity and archive-safety contracts for the local demo packager."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

PACKAGE_PATH = Path(__file__).resolve().parents[1]/'scripts/package_integrated_demo.py'
_SPEC = importlib.util.spec_from_file_location('package_integrated_demo', PACKAGE_PATH)
package = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(package)


def _manifest(path, rows):
    (path/'manifest.json').write_text(json.dumps({
        'schema': 'resectionlab.integrated-local-package.v1',
        'sourceCommit': package.COMMIT, 'files': rows,
        'fileCount': len(rows), 'contentBytes': sum(r['bytes'] for r in rows)}))


def test_frozen_selection_contains_entire_tracked_src_and_producer_inputs():
    selected = {p for p, _, _ in package.committed_files(package.COMMIT)}
    expected = set(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only',
        package.COMMIT, 'src'], cwd=package.ROOT, text=True).splitlines())
    assert {p for p in selected if p.startswith('src/')} == expected
    assert {'scripts/run_patient_cusa_learning.py', 'scripts/compare_anatomy_access.py',
            'desktop/pnpm-lock.yaml', 'desktop/electron/verify-integrated-research.cjs',
            'src/resectionlab/surface_policy.json',
            'src/resectionlab/surgical_prefix_json_native.c'} <= selected


def test_manifest_refuses_tampered_and_symlinked_members(tmp_path):
    bundle = tmp_path/'bundle'
    bundle.mkdir()
    member = bundle/'fixture.bin'
    member.write_bytes(b'original')
    _manifest(bundle, [{'path': 'fixture.bin', 'bytes': 8,
                        'sha256': hashlib.sha256(b'original').hexdigest()}])
    member.write_bytes(b'changed!')
    with pytest.raises(ValueError, match='Package member changed'):
        package.verify(bundle)
    member.unlink()
    target = tmp_path/'original.bin'
    target.write_bytes(b'original')
    member.symlink_to(target)
    with pytest.raises(ValueError, match='Missing/unsafe package file'):
        package.verify(bundle)


def test_extraction_rejects_parent_traversal(tmp_path):
    archive = tmp_path/'unsafe.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        data = b'x'
        info = tarfile.TarInfo('../escape')
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    output = tmp_path/'extracted'
    with pytest.raises(ValueError, match='Unsafe archive member'):
        package.extract(archive, output)
    assert not (output.parent/'escape').exists()
