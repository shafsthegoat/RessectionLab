"""Safety and source-selection contracts for the current local demo bundle."""
import io
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import package_current_integrated_demo as package  # noqa: E402


def test_current_snapshot_includes_full_native_and_desktop_source():
    commit = '12af2fb8185af4ed54f21c57d18802ba59c8cbe2'
    selected = {path for path, _, _ in package.historical.committed_files(commit)}
    tracked_src = set(subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', commit, 'src'],
        cwd=ROOT, text=True).splitlines())
    assert {path for path in selected if path.startswith('src/')} == tracked_src
    assert {'desktop/electron/frozen-cusa.cjs',
            'desktop/src/FrozenCusaExecutionPanel.tsx',
            'scripts/run_patient_cusa_learning.py',
            'src/resectionlab/surgical_prefix_json_native.c',
            'src/resectionlab/surface_policy.json'} <= selected


def test_current_extraction_rejects_parent_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(package, 'ROOT', tmp_path)
    archive = tmp_path/'unsafe.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        info = tarfile.TarInfo('../escape')
        info.size = 1
        tar.addfile(info, io.BytesIO(b'x'))
    output = tmp_path/'testcases/extracted'
    with pytest.raises(ValueError, match='Unsafe current package archive member'):
        package.extract(archive, output)
    assert not (output.parent/'escape').exists()
