"""Access/configuration controls only; never import SynthStrip or open a patient array."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'scripts/run_resect_brain_envelope.py'
spec = importlib.util.spec_from_file_location('case4_brain_adapter', PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
DECLARATION = json.loads((ROOT / runner.MANIFEST).read_text())


def test_frozen_declaration_and_existing_pipeline_pins():
    assert runner.sha(ROOT / runner.MANIFEST) == runner.MANIFEST_SHA
    runner.check_spec(DECLARATION)
    for name in ('wrapper', 'supervisor'):
        runner.verify_file(ROOT, DECLARATION['pipeline'][name])
    assert DECLARATION['caps']['whole_worker_seconds'] == 420
    assert DECLARATION['qc']['native_midplane_indices'] == [128, 128, 96]
    assert DECLARATION['interpretation']['working_brain_mask'] is None


def test_one_exact_main_native_input_and_no_optional_model_inputs(tmp_path):
    model = Mock(return_value={'not_an_inference': True})
    result = runner.invoke(SimpleNamespace(run_synthstrip=model), tmp_path, DECLARATION)
    model.assert_called_once_with(
        tmp_path / runner.INPUT, tmp_path / 'data/models/synthstrip-v1', tmp_path / runner.OUTPUT,
        model='main', device='mps', allow_download=False, timeout_seconds=300,
        maximum_rss_bytes=6*1024**3,
        inference_python=tmp_path / '.tools/synthstrip-runtime/bin/python')
    assert result == {'not_an_inference': True}


@pytest.mark.parametrize('field,value', [('model_variant', 'nocsf'), ('device', 'cpu'),
    ('allow_download', True), ('border_mm', 2), ('cpu_threads', 4)])
def test_altered_pipeline_fails_before_model_call(tmp_path, field, value):
    declaration = deepcopy(DECLARATION)
    declaration['pipeline'][field] = value
    model = Mock()
    with pytest.raises(ValueError, match='configuration'):
        runner.invoke(SimpleNamespace(run_synthstrip=model), tmp_path, declaration)
    model.assert_not_called()


@pytest.mark.parametrize('path', ['Landmarks/Case4-beforeUS-duringUS-full.tag',
    'US/Case4-US-during.nii.gz', 'MRI/Case4-FLAIR.nii.gz',
    'data/diffusion_source/ds001226-v5.0.1/sub-PAT29/anat.nii.gz'])
def test_other_inputs_cannot_reach_model(tmp_path, path):
    declaration = deepcopy(DECLARATION)
    declaration['input']['path'] = path
    model = Mock()
    with pytest.raises(ValueError, match='Only the declared'):
        runner.invoke(SimpleNamespace(run_synthstrip=model), tmp_path, declaration)
    model.assert_not_called()


def test_existing_output_never_reused_or_overwritten(tmp_path):
    output = tmp_path / runner.OUTPUT
    output.mkdir(parents=True)
    sentinel = output / 'original'
    sentinel.write_bytes(b'preserve this exact fixture')
    model = Mock()
    with pytest.raises(FileExistsError):
        runner.invoke(SimpleNamespace(run_synthstrip=model), tmp_path, DECLARATION)
    model.assert_not_called()
    assert sentinel.read_bytes() == b'preserve this exact fixture'


def test_byte_binding_rejects_same_size_mutation(tmp_path):
    path = tmp_path / 'source'
    path.write_bytes(b'abc')
    binding = {'path': 'source', 'bytes': 3, 'sha256': runner.sha(path)}
    runner.verify_file(tmp_path, binding)
    path.write_bytes(b'abd')
    with pytest.raises(ValueError, match='Pinned file changed'):
        runner.verify_file(tmp_path, binding)


def test_release_hash_mismatch_precedes_import_or_patient_access(tmp_path):
    release = tmp_path / 'release.json'
    release.write_text('{}')
    with pytest.raises(ValueError, match='Root release changed'):
        runner.contract(tmp_path, release, '0'*64)


def test_mutable_checkout_execution_rejected(tmp_path):
    snapshot = tmp_path / 'build/snapshot'
    snapshot.mkdir(parents=True)
    release = tmp_path / 'release.json'
    release.write_text(json.dumps({'scope': runner.SCOPE, 'manifest_sha256': runner.MANIFEST_SHA,
                                   'snapshot_root': str(snapshot)}))
    with pytest.raises(ValueError, match='mutable checkout'):
        runner.contract(tmp_path, release, runner.sha(release))


def test_an_estimate_cannot_be_declared_working_support(tmp_path):
    declaration = deepcopy(DECLARATION)
    declaration['interpretation']['working_brain_mask'] = 'promoted'
    model = Mock()
    with pytest.raises(ValueError, match='working support'):
        runner.invoke(SimpleNamespace(run_synthstrip=model), tmp_path, declaration)
    model.assert_not_called()
