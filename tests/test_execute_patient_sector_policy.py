"""Cold exports retain training exposure without running fit or native physics."""
import importlib.util
from pathlib import Path

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
