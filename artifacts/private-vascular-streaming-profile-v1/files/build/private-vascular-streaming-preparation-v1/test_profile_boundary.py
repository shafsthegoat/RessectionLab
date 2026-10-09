"""Source/release boundary only; the realistic generated profile is never run."""
from pathlib import Path
import builtins
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

import pytest

loader = importlib.util.spec_from_file_location('generated_profile_boundary', Path(__file__).with_name('profile_generated.py'))
p = importlib.util.module_from_spec(loader)
sys.modules[loader.name] = p
loader.loader.exec_module(p)


def test_bounded_release_reader_refuses_fifo_and_oversize(tmp_path):
    fifo = tmp_path/'release.fifo'
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match='Regular bounded'):
        p.read_regular(fifo, 32)
    oversized = tmp_path/'large.json'
    oversized.write_bytes(b'x'*33)
    with pytest.raises(ValueError, match='Regular bounded'):
        p.read_regular(oversized, 32)


def test_release_reader_refuses_symlink(tmp_path):
    target = tmp_path/'target.json'; target.write_text('{}')
    link = tmp_path/'link.json'; link.symlink_to(target)
    with pytest.raises(OSError):
        p.read_regular(link)


def test_false_release_refuses_before_scientific_import_or_output(tmp_path, monkeypatch):
    pins, _ = p.source_pins()
    release = tmp_path/'disabled.json'
    release.write_text(json.dumps({'scope': p.SCOPE, 'source_hashes': pins, 'root_release': False}))
    output = tmp_path/'must-stay-absent'
    monkeypatch.setattr(sys, 'argv', ['profile', '--release', str(release), '--output-directory', str(output)])
    real_import = builtins.__import__
    def checked_import(name, *args, **kwargs):
        assert name != 'numpy', 'Scientific import before release refusal'
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', checked_import)
    with pytest.raises(SystemExit, match='Exact root release refused'):
        p.main()
    assert not output.exists()


def test_repository_inventory_checks_expected_helper_bytes(tmp_path, monkeypatch):
    inventory = json.loads((p.PREP/'repository-source-pins.json').read_text())
    assert 'src/resectionlab/independent_geometry_batch.py' in inventory
    assert 'src/resectionlab/evaluation.py' in inventory
    real_sha = p.sha
    monkeypatch.setattr(p, 'sha', lambda path: '0'*64 if str(path).endswith('independent_geometry_batch.py') else real_sha(path))
    with pytest.raises(ValueError, match='Repository source pin changed'):
        p.source_pins()


def test_check_only_does_not_import_scientific_stack_or_evaluate(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['profile', '--check-only'])
    real_import = builtins.__import__
    def checked_import(name, *args, **kwargs):
        assert name != 'numpy'
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', checked_import)
    p.main()
    result = json.loads(capsys.readouterr().out)
    assert result['release'] is False and result['grid_voxels'] == 12582912


def test_shadow_repository_module_refuses(tmp_path, monkeypatch):
    path = tmp_path/'shadow.py'; path.write_text('# Generated wrong-origin module\n')
    module = SimpleNamespace(__file__=str(path), __spec__=SimpleNamespace(origin=str(path)))
    monkeypatch.setitem(sys.modules, 'resectionlab.evaluation', module)
    with pytest.raises(ValueError, match='Loaded repository origin mismatch'):
        p.verify_loaded_repository_origins()
