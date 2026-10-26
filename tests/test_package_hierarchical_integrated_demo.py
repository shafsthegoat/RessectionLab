"""Safety and source-snapshot contracts for the local hierarchy bundle."""
from __future__ import annotations

import io
from pathlib import Path
import sys
import tarfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import package_hierarchical_integrated_demo as package


def test_extraction_rejects_parent_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(package, 'ROOT', tmp_path)
    base = tmp_path/'testcases'
    base.mkdir()
    archive = base/'malicious.tar.gz'
    payload = b'overwrite'
    with tarfile.open(archive, 'w:gz') as tar:
        entry = tarfile.TarInfo('../escape.txt')
        entry.size = len(payload)
        tar.addfile(entry, io.BytesIO(payload))
    with pytest.raises(ValueError, match='Unsafe hierarchical archive member'):
        package.extract(archive, base/'extracted')
    assert not (base/'escape.txt').exists()


def test_extraction_rejects_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(package, 'ROOT', tmp_path)
    base = tmp_path/'testcases'
    base.mkdir()
    archive = base/'malicious-link.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        entry = tarfile.TarInfo('model.pt')
        entry.type = tarfile.SYMTYPE
        entry.linkname = '/outside/model.pt'
        tar.addfile(entry)
    with pytest.raises(ValueError, match='Unsafe hierarchical archive member'):
        package.extract(archive, base/'extracted')


def test_build_requires_two_real_desktop_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(package, 'ROOT', tmp_path)
    (tmp_path/'testcases').mkdir()
    with pytest.raises(ValueError, match='both development cases'):
        package.build(tmp_path/'testcases/new', 'HEAD', {})


def test_instructions_preserve_failed_transfer_and_runtime_limit():
    text = package._instructions('a'*40)
    assert '0005' in text and 'incomplete' in text
    assert 'zero optimizer updates' in text
    assert 'same-host Python environment' in text
    assert 'six' in text and 'twelve' in text
