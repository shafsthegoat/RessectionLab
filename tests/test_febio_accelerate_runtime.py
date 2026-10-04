"""Source-text and mocked build controls; no compiler, fixture or solver launch."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from scripts import febio_accelerate_runtime as d


def v1():
    return json.loads((d.ROOT/'manifests/experiments/febio-accelerate-csc-runtime-v1.json').read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_declaration_loads_saved_positive_only_evidence_without_execution(monkeypatch):
    monkeypatch.setattr(d.subprocess, 'run', lambda *a, **kw: pytest.fail('No subprocess expected'))
    spec, original = d.load_declaration(d.rt.sha(d.DECLARATION))
    assert spec['basis']['sha256'] == d.V1_SHA
    assert original['patch']['patched_sha256'] == d.PATCHED_SHA
    assert original['caps']['configure_seconds'] == 120
    assert original['caps']['build_and_install_seconds'] == 900
    assert original['build']['parallel_jobs'] == 2


def test_exact_reviewed_patch_is_text_only_and_rejects_second_or_fuzzy_application():
    original = v1()
    source = (d.OLD_PREFIX/'source'/original['patch']['relative_source']).read_bytes()
    patch = (d.ROOT/original['patch']['file']['path']).read_bytes()
    repaired = d.exact_single_hunk(source, patch, original['patch']['relative_source'])
    assert hashlib.sha256(source).hexdigest() == original['patch']['original_sha256']
    assert hashlib.sha256(repaired).hexdigest() == d.PATCHED_SHA
    for changed in (repaired, b'\n'+source, source.replace(b'imp->m_nrhs = 1;', b'imp->m_nrhs = 2;')):
        with pytest.raises(ValueError, match='context|position'):
            d.exact_single_hunk(changed, patch, original['patch']['relative_source'])


def test_commands_preserve_every_original_cache_option_and_two_job_build():
    original = v1()
    configure, = d.commands('configure', original)
    options = dict(item[2:].split('=', 1) for item in configure if item.startswith('-D'))
    assert options == original['configure']['cache_options']
    build, install = d.commands('build', original)
    assert build[-4:] == ['--target', 'febio4', '--parallel', '2']
    assert install[-2:] == ['--prefix', str(d.PREFIX/'install')]
    with pytest.raises(ValueError):
        d.commands('bounds', original)


def test_child_environment_is_private_and_worker_startup_overrides_removed(monkeypatch):
    for key in ('PYTHONINSPECT', 'PYTHONSTARTUP', 'PYTHONPATH', 'DYLD_INSERT_LIBRARIES',
                'LD_PRELOAD', 'LD_LIBRARY_PATH', '__PYVENV_LAUNCHER__', 'MAKEFLAGS'):
        monkeypatch.setenv(key, 'untrusted-setting')
    monkeypatch.setenv('HOME', '/preserved-home')
    monkeypatch.setenv('CODEX_HOME', '/preserved-codex-home')
    environment = d.private_environment(v1())
    assert not any(key.startswith(('PYTHON', 'DYLD_')) for key in environment)
    assert all(key not in environment for key in ('LD_PRELOAD', 'LD_LIBRARY_PATH', '__PYVENV_LAUNCHER__', 'MAKEFLAGS'))
    assert environment['HOME'] == '/preserved-home'
    assert environment['CODEX_HOME'] == '/preserved-codex-home'
    assert environment['OMP_NUM_THREADS'] == '1'


def test_source_inventory_accepts_only_declared_single_difference(tmp_path, monkeypatch):
    prefix = tmp_path/'prefix'
    source = prefix/'source'
    source.mkdir(parents=True)
    (source/'a.cpp').write_bytes(b'original')
    (source/'unchanged.h').write_bytes(b'fixed')
    expected = {'source/'+key: value for key, value in d.rt.inventory(source).items()}
    inventory = tmp_path/'inventory.json'
    write(inventory, expected)
    spec = {'upstream': {'original_input_inventory': {'path': str(inventory),
        'sha256': d.rt.sha(inventory), 'bytes': inventory.stat().st_size}},
        'patch': {'relative_source': 'a.cpp', 'patched_sha256': hashlib.sha256(b'patched').hexdigest()}}
    monkeypatch.setattr(d, 'PREFIX', prefix)
    d.verify_source(spec, patched=False)
    (source/'a.cpp').write_bytes(b'patched')
    d.verify_source(spec)
    (source/'unchanged.h').write_bytes(b'changed')
    with pytest.raises(ValueError, match='source difference'):
        d.verify_source(spec)


@pytest.mark.parametrize('change', ['dependency', 'rpath'])
def test_unknown_dependency_or_rpath_is_rejected(change):
    dependency = '@rpath/libfecore.dylib'
    rpath = '@loader_path'
    text = f'binary:\n\t{dependency} (compatibility version 0.0.0)\n'
    commands = f' cmd LC_RPATH\n cmdsize 32\n path {rpath} (offset 12)\n'
    assert d.linkage_details(text, commands, [dependency], [rpath])['dependencies'] == [dependency]
    if change == 'dependency':
        text += '\t/unknown/libforeign.dylib (compatibility version 0.0.0)\n'
    else:
        commands += ' cmd LC_RPATH\n cmdsize 32\n path /unknown (offset 12)\n'
    with pytest.raises(ValueError, match='Undeclared'):
        d.linkage_details(text, commands, [dependency], [rpath])


def test_declared_old_openmp_is_reused_and_rpath_shadowing_refused(tmp_path, monkeypatch):
    prefix = tmp_path/'new'
    (prefix/'install/bin').mkdir(parents=True)
    (prefix/'install/lib').mkdir()
    omp = tmp_path/'old/openmp/lib/libomp.dylib'
    omp.parent.mkdir(parents=True)
    omp.write_bytes(b'pinned old OpenMP')
    monkeypatch.setattr(d, 'PREFIX', prefix)
    spec = {'tools': {'openmp_library': {'path': str(omp)}}}
    details = {'dependencies': ['@rpath/libomp.dylib'],
               'rpaths': ['@loader_path', '@executable_path/../lib', str(omp.parent)]}
    resolved = d.resolve_linkage(prefix/'install/bin/febio4', details, spec)
    assert resolved['@rpath/libomp.dylib']['path'] == str(omp)
    (prefix/'install/lib/libomp.dylib').write_bytes(omp.read_bytes())
    with pytest.raises(ValueError, match='outside declared'):
        d.resolve_linkage(prefix/'install/bin/febio4', details, spec)


def test_macho_inventory_finds_unexpected_executable_without_dylib_suffix(tmp_path, monkeypatch):
    prefix = tmp_path/'new'
    (prefix/'install/share').mkdir(parents=True)
    (prefix/'install/share/hidden-payload').write_bytes(bytes.fromhex('cffaedfe')+b'fixture only')
    (prefix/'install/share/notice.txt').write_text('ordinary notice')
    monkeypatch.setattr(d, 'PREFIX', prefix)
    assert d.installed_macho_names(d.rt.inventory(prefix/'install')) == {'install/share/hidden-payload'}


def configure_fixture(tmp_path, monkeypatch):
    output = tmp_path/'receipts'
    prefix = tmp_path/'prefix'
    directory = output/'configure-01'
    directory.mkdir(parents=True)
    (prefix/'build').mkdir(parents=True)
    (prefix/'build/CMakeCache.txt').write_text('fixture')
    monkeypatch.setattr(d, 'PREFIX', prefix)
    monkeypatch.setattr(d, 'OUTPUT', output)
    monkeypatch.setattr(d, 'verify_source', lambda *a, **k: {})
    monkeypatch.setattr(d, 'verify_cache', lambda *a, **k: {})
    result = {'status': 'completed', 'stage': 'configure', 'declaration_sha256': 'a'*64,
              'driver_sha256': d.rt.sha(Path(d.__file__)), 'solver_executed': False,
              'cache_sha256': d.rt.sha(prefix/'build/CMakeCache.txt')}
    supervision = {'status': 'completed', 'exit_code': 0, 'kill_reason': None, 'cleanup_error': None,
                   'wall_cap_seconds': 120, 'rss_cap_bytes': 3*1024**3}
    for name in d.STAGE_FILES['configure']:
        write(directory/name, result if name == 'result.json' else supervision if name == 'supervision.json' else {})
    acceptance = {'status': 'completed', 'declaration_sha256': 'a'*64,
        'driver_sha256': d.rt.sha(Path(d.__file__)),
        'artifacts': {name: d.rt.sha(directory/name) for name in d.STAGE_FILES['configure']}}
    write(directory/'acceptance.json', acceptance)
    return directory, acceptance


@pytest.mark.parametrize('target,update', [
    ('result.json', {'status': 'failed_or_incomplete'}),
    ('result.json', {'stage': 'different'}),
    ('supervision.json', {'status': 'failed_or_incomplete'}),
    ('supervision.json', {'wall_cap_seconds': 999}),
])
def test_resealed_acceptance_cannot_hide_failed_or_wrong_configure_stage(tmp_path, monkeypatch, target, update):
    directory, acceptance = configure_fixture(tmp_path, monkeypatch)
    d.accepted_config(v1(), 'a'*64)
    data = json.loads((directory/target).read_bytes())
    data.update(update)
    write(directory/target, data)
    acceptance['artifacts'][target] = d.rt.sha(directory/target)
    write(directory/'acceptance.json', acceptance)
    with pytest.raises(ValueError, match='Configure worker/supervision'):
        d.accepted_config(v1(), 'a'*64)


def test_failed_stage_keeps_attempt_and_rechecks_original_without_runtime_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'OUTPUT', tmp_path/'receipts')
    monkeypatch.setattr(d, 'load_declaration', lambda *a: ({}, v1()))
    checked = []
    monkeypatch.setattr(d, 'verify_original', lambda *a: checked.append('after') or {'unchanged': True})
    calls = []
    def failed_supervision(command, output, **kwargs):
        calls.append(kwargs)
        assert command[1:4] == ['-I', '-S', '-B']
        output.mkdir()
        receipt = {'status': 'failed_or_incomplete'}
        write(output/'supervision.json', receipt)
        return receipt
    monkeypatch.setattr(d.rt, 'supervise', failed_supervision)
    monkeypatch.setattr(sys, 'argv', ['driver', 'build', '--declaration-sha256', 'a'*64])
    assert d.main() == 1
    assert checked == ['after']
    assert calls[0]['seconds'] == 900 and calls[0]['rss_bytes'] == 3*1024**3
    assert (d.OUTPUT/'build-attempt.json').exists()
    assert not (d.OUTPUT/'runtime-identity.json').exists()
    with pytest.raises(FileExistsError):
        d.main()
    assert len(calls) == 1


def test_wrong_declaration_sha_refuses_without_stage_or_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'OUTPUT', tmp_path/'never-created')
    monkeypatch.setattr(d.rt, 'supervise', lambda *a, **kw: pytest.fail('No supervision allowed'))
    monkeypatch.setattr(sys, 'argv', ['driver', 'configure', '--declaration-sha256', 'f'*64])
    with pytest.raises(ValueError, match='declaration'):
        d.main()
    assert not d.OUTPUT.exists()
