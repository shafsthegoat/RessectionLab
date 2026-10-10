"""Metadata and process admission only; no task/checkpoint/model execution."""
import sys
import pytest
from resectionlab import contact_family_desktop_bridge as bridge
from resectionlab import contact_family_supervisor as supervisor


def test_refit_select_refuses_before_publication_or_controller(tmp_path, monkeypatch):
    select = next(row['recipe']['layout_id'] for row in bridge.family_record()['layouts'] if row['role'] == 'SELECT')
    def forbidden(*args, **kwargs):
        pytest.fail('TRAIN-only refusal must precede publication/controller')
    monkeypatch.setattr(bridge, 'read_published_train_refit', forbidden)
    monkeypatch.setattr(bridge, '_read_exact_supervisor', forbidden)
    with pytest.raises(bridge.ContactReleaseUnavailable, match='TRAIN_REFIT_ONLY'):
        bridge.execute_public_contact_family_episode(attempts_root=tmp_path, layout_id=select,
                                                     goal_id='surface', selector='IL_TRAIN_REFIT')
    assert not list(tmp_path.iterdir())


def test_refit_catalog_unavailable_keeps_original_slots(monkeypatch):
    monkeypatch.setattr(bridge, '_controller_ready', lambda: True)
    monkeypatch.setattr(bridge, 'RELEASE_MANIFEST_SHA256', None)
    def unavailable(*args, **kwargs):
        raise bridge.ContactReleaseUnavailable('missing refit')
    monkeypatch.setattr(bridge, 'read_published_train_refit', unavailable)
    value = bridge.public_contact_family_availability()
    assert value['version'] == 'generated-public-contact-learning-availability-v2'
    assert value['methods']['STOP'] == value['methods']['SEARCH'] == {'available': True, 'reason': None}
    assert not value['methods']['IL']['available'] and not value['methods']['RL']['available']
    refit = value['methods']['IL_TRAIN_REFIT']
    assert refit['allowedRoles'] == ['TRAIN'] and refit['available'] is False
    assert all(refit[key] is None for key in ('experimentHash', 'releaseHash', 'checkpointFileSha256',
                                             'parameterHash', 'evidence', 'knownTRAINOutcome', 'trainingBudget'))


def test_direct_worker_refuses_before_output_or_torch(tmp_path, monkeypatch):
    from resectionlab import contact_family_worker as worker
    before = 'torch' in sys.modules
    monkeypatch.setattr(worker, '_PARENT_LEASE_ACTIVE', False)
    with pytest.raises(RuntimeError, match='active parent lease'):
        worker.execute(tmp_path / 'result.json', layout_id='pcf-06', goal_id='surface', selector='IL_TRAIN_REFIT')
    assert ('torch' in sys.modules) == before and not list(tmp_path.iterdir())


def test_refit_controller_accepts_selector_but_cancel_precedes_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, '_check_sources', lambda: pytest.fail('source/child after cancellation'))
    with pytest.raises(InterruptedError):
        supervisor.run_attempt(tmp_path / 'cancel', layout_id='pcf-06', goal_id='surface',
                               selector='IL_TRAIN_REFIT', cancelled=lambda: True)


def test_refit_source_closure_keeps_relation_and_descriptor():
    assert set(supervisor.SOURCE_SHA256) == supervisor.SOURCE_FILES
    assert {'local:contact_train_refit_release.py', 'src/resectionlab/goal_relation_spatial_policy.py',
            'artifacts/public-contact-train-refit-v1/desktop-release.json'} <= supervisor.SOURCE_FILES
    assert supervisor.MAX_RESULT_BYTES == 2 * 1024 * 1024
    assert supervisor.MAX_RSS_BYTES == 1024 * 1024 * 1024
