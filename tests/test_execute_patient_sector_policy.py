"""Cold exports retain training exposure without running fit or native physics."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location('cold_sector',
    Path(__file__).resolve().parents[1] / 'scripts/execute_patient_sector_policy.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def payload(stage='bc-initialized-ppo', paired=True):
    train = {'sceneId': 'source-7', 'sourceArchiveSha256': 'original-bytes'}
    proof = {}
    if paired:
        proof = {'ppoSourceSchedule': 'paired-development',
                 'optimizationSourceBindings': {'train': train,
                    'transfer': {'sceneId': 'source-5'}}}
    return {'stage': stage, 'sourceBinding': train, 'trainingReceipt': proof}


@pytest.mark.parametrize('scene,role', [('source-7', 'train'), ('source-5', 'transfer')])
def test_paired_exposure_survives_desktop_archive_serialization(scene, role):
    source_key, schedule, bindings = module.execution_export_context(payload(), scene)
    assert source_key == role
    assert schedule == 'paired-development'
    assert set(bindings) == {'train', 'transfer'}


def test_paired_export_rejects_unlisted_patient_exposure():
    with pytest.raises(ValueError, match='outside this paired'):
        module.execution_export_context(payload(), 'other-patient')


def test_bc_second_development_case_remains_evaluation_only():
    source_key, schedule, bindings = module.execution_export_context(
        payload(stage='bc', paired=False), 'source-5')
    assert (source_key, schedule) == ('transfer', 'train-only')
    assert set(bindings) == {'train'}


def test_cold_export_rejects_conflicting_primary_binding():
    value = payload()
    value['trainingReceipt']['optimizationSourceBindings']['train'] = {'sceneId': 'wrong'}
    with pytest.raises(ValueError, match='bindings differ'):
        module.execution_export_context(value, 'source-7')


def test_runtime_fallback_requires_explicit_opt_in_before_source_or_output(tmp_path, monkeypatch):
    checkout = tmp_path / 'checkout'
    (checkout / 'testcases').mkdir(parents=True)
    output = checkout / 'testcases/new-execution'
    source = checkout / 'testcases/source.ressectionlab'
    checkpoint = checkout / 'testcases/model.pt'
    checkpoint.write_bytes(b'fixture checkpoint, no native or pixel data')
    monkeypatch.setattr(module, 'ROOT', checkout)
    monkeypatch.setattr(module.torch, 'load', lambda *_, **__: {'producerIdentity': {'old': True}})
    monkeypatch.setattr(module, 'producer_identity', lambda: {'new': True})
    base = ['cold-sector', '--source', str(source), '--checkpoint', str(checkpoint),
            '--output', str(output), '--source-role', 'DEVELOPMENT']
    monkeypatch.setattr(sys, 'argv', base)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert not output.exists()
    def selected(*args):
        assert args[:3] == (source.resolve(), checkpoint.resolve(), output.resolve())
        raise RuntimeError('matching runtime admitted before source use')
    monkeypatch.setattr(module, '_execute_matching_runtime', selected)
    monkeypatch.setattr(sys, 'argv', [*base, '--allow-matching-runtime'])
    with pytest.raises(RuntimeError, match='admitted before source use'):
        module.main()
    assert not output.exists()


def test_matching_current_producer_keeps_normal_cold_execution_path(tmp_path, monkeypatch):
    checkout = tmp_path / 'checkout'
    (checkout / 'testcases').mkdir(parents=True)
    output = checkout / 'testcases/normal-execution'
    checkpoint = checkout / 'testcases/model.pt'
    checkpoint.write_bytes(b'fixture checkpoint, no native or pixel data')
    monkeypatch.setattr(module, 'ROOT', checkout)
    monkeypatch.setattr(module.torch, 'load', lambda *_, **__: {'producerIdentity': {'same': True}})
    monkeypatch.setattr(module, 'producer_identity', lambda: {'same': True})
    monkeypatch.setattr(module, '_execute_matching_runtime', lambda *_: pytest.fail('fallback was selected'))
    monkeypatch.setattr(module, 'run', lambda *_, **__: {'status': 'evaluated',
        'execution': {'status': 'abstained'}})
    monkeypatch.setattr(sys, 'argv', ['cold-sector', '--source', str(checkout / 'testcases/source'),
        '--checkpoint', str(checkpoint), '--output', str(output),
        '--source-role', 'DEVELOPMENT', '--allow-matching-runtime'])
    module.main()
    assert output.is_dir()


def test_pinned_runtime_manifest_rejects_changed_or_added_source(tmp_path, monkeypatch):
    runtime = tmp_path / 'runtime'
    source = runtime / 'source'
    code = source / 'src/resectionlab'
    code.mkdir(parents=True)
    scripts = source / 'scripts'
    scripts.mkdir()
    files = {}
    for index in range(234):
        path = code / f'module_{index:03d}.py'
        path.write_text(f'VALUE = {index}\n')
        files[str(path.relative_to(source))] = module.sha256(path)
    for name in ('execute_patient_sector_policy.py', 'run_patient_sector_learning.py'):
        path = scripts / name
        path.write_text('VALUE = 1\n')
        files[str(path.relative_to(source))] = module.sha256(path)
    checkout = tmp_path / 'checkout'
    (checkout / 'testcases').mkdir(parents=True)
    (source / 'testcases').symlink_to(checkout / 'testcases', target_is_directory=True)
    producer = {'old': True}
    manifest = {'schema': 'resectionlab.frozen-matching-runtime-files.v1',
        'sourceArchiveSha256': module.MATCHING_RUNTIME_ARCHIVE_SHA256,
        'gitCommit': module.MATCHING_RUNTIME_COMMIT,
        'producerIdentity': producer, 'files': files}
    path = runtime / 'files-manifest.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(module, 'MATCHING_RUNTIME', runtime)
    monkeypatch.setattr(module, 'ROOT', checkout)
    monkeypatch.setattr(module, 'MATCHING_RUNTIME_MANIFEST_SHA256', module.sha256(path))
    assert module._matching_runtime_files(producer)[0] == source
    (code / 'module_001.py').write_text('CHANGED = True\n')
    with pytest.raises(ValueError, match='source file changed'):
        module._matching_runtime_files(producer)
    (code / 'module_001.py').write_text('VALUE = 1\n')
    (code / 'extra.py').write_text('UNDECLARED = True\n')
    with pytest.raises(ValueError, match='inventory changed'):
        module._matching_runtime_files(producer)
