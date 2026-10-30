"""Generated numerical controls, not acquired anatomy or clinical validation."""
import copy

import numpy as np
import pytest

from resectionlab.full_case_readiness import (
    ProtectionBindingError, full_case_readiness,
)
from resectionlab.full_case_rehearsal import bind_case_protection
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_rl_protection import (
    GuardConfig, ProtectionAnnotation, ProtectionSet, grid_binding,
    protection_contract, save_protection,
)
from resectionlab.surgical_scene import SurgicalScene


def _scene(*, source='generated-control', target=True, affine=None, shift=False):
    labels = np.zeros((6, 6, 6), np.uint16)
    labels[3 if shift else 2, 2, 2] = 1
    return SurgicalScene(labels, np.eye(4) if affine is None else affine,
        [{'id': 1, 'name': 'generated target', 'material': 'tumor', 'target': target}],
        {'kind': source}, {})


def _protection(scene, *, cells=(0,), extra=(), required=('brainstem',), clearance=2.):
    rows = [('brainstem', cells), *extra]
    annotations = [ProtectionAnnotation(name, tuple(indices), grid_binding(scene),
        'generated numerical control; not anatomical evidence') for name, indices in rows]
    return ProtectionSet(scene, annotations, required=required, clearance_mm=clearance)


def _engine(*, config=None, **kwargs):
    scene = _scene()
    return SurgicalEngine(scene, protection=_protection(scene, **kwargs),
                          guard_config=config or GuardConfig())


def _inspect(engine):
    assert engine.step({'slot': 'right', 'operation': 'inspect', 'duration': .2})['status'] == 'accepted'


@pytest.mark.parametrize('config', [GuardConfig(), GuardConfig(require_critical_coverage=False),
                                   GuardConfig(critical_max_displacement_mm=.1)])
def test_initial_saved_source_bound_evidence_is_ready_and_binds_exact_default_policy(config):
    engine = _engine(config=config)
    before = copy.deepcopy(engine.snapshot())
    result = full_case_readiness(engine)
    assert result == {'schema': 'resectionlab.full-case-readiness.v1',
        'sceneId': engine.scene.identity, 'stateId': engine.state_id, 'status': 'ready',
        'availableActions': ['inspect', 'replay', 'primary', 'vascular'], 'blockers': [],
        'recovery': None,
        'protectionContractId': protection_contract(engine.protection, GuardConfig())['contentHash']}
    bound = bind_case_protection(engine)
    assert bound.guard_config == GuardConfig()
    assert bound._protected_ids == engine._protected_ids
    assert (bound is engine) == (config == GuardConfig())
    assert engine.snapshot() == before


def test_missing_evidence_is_actionable_without_source_or_filename_guessing(tmp_path, monkeypatch):
    engine = SurgicalEngine(_scene())
    # A tempting local manifest is irrelevant unless explicitly selected.
    manifest = tmp_path / 'rl-protection.json'
    save_protection(_protection(engine.scene), manifest)
    monkeypatch.chdir(tmp_path)
    before = engine.snapshot()
    result = full_case_readiness(engine)
    assert result['status'] == 'blocked'
    assert result['blockers'][0]['code'] == 'missing-protection'
    assert result['availableActions'] == ['inspect', 'replay', 'attachProtection']
    assert result['recovery'] == {'action': 'attachProtection', 'label': 'Attach matching protection'}
    with pytest.raises(ProtectionBindingError, match='Attach source-bound'):
        bind_case_protection(engine)
    assert engine.snapshot() == before
    bound = bind_case_protection(engine, manifest)
    assert full_case_readiness(bound)['status'] == 'ready'


@pytest.mark.parametrize('contents', [None, '3', '{"schema": "unsupported"}'])
def test_missing_or_malformed_explicit_manifest_has_useful_blocker(tmp_path, contents):
    engine = SurgicalEngine(_scene())
    manifest = tmp_path / 'generated-invalid.json'
    if contents is not None:
        manifest.write_text(contents)
    with pytest.raises(ProtectionBindingError) as error:
        bind_case_protection(engine, manifest)
    assert error.value.code == 'invalid-protection-manifest'
    assert error.value.details


@pytest.mark.parametrize('change', ['source', 'labels', 'affine'])
def test_explicit_manifest_rejects_each_source_grid_mismatch(tmp_path, change):
    source = _scene()
    manifest = tmp_path / 'generated-control.json'
    save_protection(_protection(source), manifest)
    affine = np.eye(4)
    affine[0, 3] = .25
    scene = _scene(**({'source': 'different-generated-control'} if change == 'source' else
                      {'shift': True} if change == 'labels' else {'affine': affine}))
    engine = SurgicalEngine(scene)
    before = engine.snapshot()
    with pytest.raises(ProtectionBindingError) as error:
        bind_case_protection(engine, manifest)
    assert error.value.code == 'invalid-protection-manifest'
    assert 'source grid' in error.value.details[0]
    assert engine.snapshot() == before


@pytest.mark.parametrize('change', ['source', 'labels', 'affine'])
def test_saved_contract_revalidates_current_scene_source_labels_and_affine(change):
    engine = _engine()
    affine = np.eye(4)
    affine[0, 3] = .25
    engine.scene = _scene(**({'source': 'different-generated-control'} if change == 'source' else
                             {'shift': True} if change == 'labels' else {'affine': affine}))
    before = engine.snapshot()
    result = full_case_readiness(engine)
    assert result['status'] == 'blocked'
    assert result['blockers'][0]['code'] == 'invalid-protection-contract'
    assert result['availableActions'] == ['inspect', 'replay']
    assert result['recovery'] is None
    assert engine.snapshot() == before


def test_saved_contract_integrity_and_live_protection_are_rechecked():
    engine = _engine()
    engine._protection_contract['clearanceMm'] = .1
    result = full_case_readiness(engine)
    assert result['blockers'][0]['code'] == 'invalid-protection-contract'
    assert result['recovery'] is None
    engine = _engine()
    engine.protection.clearance_mm = .1
    assert full_case_readiness(engine)['blockers'][0]['code'] == 'invalid-protection-contract'
    engine = _engine()
    engine._protected_ids = frozenset()
    assert full_case_readiness(engine)['blockers'][0]['code'] == 'invalid-protection-contract'


def test_mutated_scene_metadata_blocks_readiness():
    engine = _engine()
    engine.scene.source['kind'] = 'changed'
    result = full_case_readiness(engine)
    assert result['status'] == 'blocked'
    assert result['blockers'][0]['code'] == 'source-mismatch'
    assert result['recovery'] is None


@pytest.mark.parametrize('missing', ['brainstem', 'declared'])
def test_brainstem_and_all_declared_coverage_required_for_saved_and_explicit_evidence(tmp_path, missing):
    scene = _scene()
    protection = _protection(scene, cells=() if missing == 'brainstem' else (0,),
                             required=() if missing == 'brainstem' else ('brainstem', 'optic'))
    engine = SurgicalEngine(scene, protection=protection,
                            guard_config=GuardConfig(require_critical_coverage=False))
    result = full_case_readiness(engine)
    assert result['status'] == 'blocked'
    assert result['blockers'][0]['code'] == 'missing-critical-annotations'
    assert ('brainstem' if missing == 'brainstem' else 'optic') in result['blockers'][0]['details']
    manifest = tmp_path / 'generated-incomplete.json'
    save_protection(protection, manifest)
    with pytest.raises(ProtectionBindingError):
        bind_case_protection(SurgicalEngine(scene), manifest)


@pytest.mark.parametrize('config', [GuardConfig(), GuardConfig(require_critical_coverage=False),
                                   GuardConfig(critical_max_displacement_mm=.1)])
def test_action_history_accepts_only_identical_strict_contract(config):
    engine = _engine(config=config)
    _inspect(engine)
    before = engine.snapshot()
    result = full_case_readiness(engine)
    if config == GuardConfig():
        assert result['status'] == 'ready'
        assert bind_case_protection(engine) is engine
    else:
        assert result['status'] == 'blocked'
        assert result['blockers'][0]['code'] == 'initial-checkpoint-required'
        assert result['availableActions'] == ['inspect', 'replay', 'seekInitial']
        assert result['recovery'] == {'action': 'seekInitial', 'label': 'Seek initial checkpoint', 'position': 0}
        with pytest.raises(ProtectionBindingError, match='initial checkpoint'):
            bind_case_protection(engine)
    assert engine.snapshot() == before


def test_identical_explicit_contract_after_actions_is_accepted_but_added_cells_are_not(tmp_path):
    engine = _engine()
    _inspect(engine)
    manifest = tmp_path / 'generated-identical.json'
    save_protection(engine.protection, manifest)
    assert bind_case_protection(engine, manifest) is engine
    save_protection(_protection(engine.scene, cells=(0, 1)), manifest)
    with pytest.raises(ProtectionBindingError) as error:
        bind_case_protection(engine, manifest)
    assert error.value.code == 'initial-checkpoint-required'


@pytest.mark.parametrize('reduction', ['clearance', 'cells', 'requirements', 'structure-cells'])
def test_attach_cannot_relax_existing_evidence(tmp_path, reduction):
    engine = _engine(cells=(0, 1), extra=(('optic', (2,)),), required=('brainstem', 'optic'))
    changes = {'cells': (0, 1), 'extra': (('optic', (2,)),), 'required': ('brainstem', 'optic')}
    if reduction == 'clearance': changes['clearance'] = 1.
    elif reduction == 'cells': changes['cells'] = (0,)
    elif reduction == 'requirements': changes['required'] = ('brainstem',)
    else: changes.update(extra=(('optic', (3,)), ('other', (2,))))
    manifest = tmp_path / 'generated-reduced.json'
    save_protection(_protection(engine.scene, **changes), manifest)
    before = engine.snapshot()
    with pytest.raises(ProtectionBindingError) as error:
        bind_case_protection(engine, manifest)
    assert error.value.code == 'protection-relaxation'
    assert engine.snapshot() == before


def test_attach_can_add_protection_at_initial_checkpoint(tmp_path):
    engine = _engine()
    manifest = tmp_path / 'generated-increased.json'
    save_protection(_protection(engine.scene, cells=(0, 1), extra=(('optic', (2,)),),
                                required=('brainstem', 'optic'), clearance=3.), manifest)
    bound = bind_case_protection(engine, manifest)
    assert bound is not engine
    assert bound._protected_ids == frozenset({0, 1, 2})
    assert full_case_readiness(bound)['status'] == 'ready'


def test_missing_owned_tumor_does_not_admit_scenarios():
    scene = _scene(target=False)
    engine = SurgicalEngine(scene, protection=_protection(scene))
    result = full_case_readiness(engine)
    assert result['status'] == 'blocked'
    assert result['availableActions'] == ['inspect', 'replay']
    assert result['blockers'][0]['code'] == 'missing-owned-tumor-target'
    assert result['recovery'] is None


def test_action_history_without_evidence_offers_seek_initial():
    engine = SurgicalEngine(_scene())
    _inspect(engine)
    result = full_case_readiness(engine)
    assert result['availableActions'] == ['inspect', 'replay', 'seekInitial']
    assert result['recovery']['position'] == 0


@pytest.mark.parametrize('grid', ['oblique', 'superior-decreasing'])
def test_compatible_protection_does_not_admit_unsupported_director_affine(grid):
    affine = np.eye(4)
    if grid == 'oblique': affine[0, 1] = .2
    else: affine[2, 2] = -1.
    scene = _scene(affine=affine)
    engine = SurgicalEngine(scene, protection=_protection(scene))
    before = engine.snapshot()
    result = full_case_readiness(engine)
    assert result['status'] == 'blocked'
    assert result['availableActions'] == ['inspect', 'replay']
    assert result['blockers'][0]['code'] == 'unsupported-director-grid'
    assert result['recovery'] is None
    assert engine.snapshot() == before


@pytest.mark.parametrize('intrinsic', ['missing-owned-tumor-target', 'unsupported-director-grid'])
@pytest.mark.parametrize('has_actions', [False, True])
def test_intrinsic_blocker_precedes_missing_evidence_and_has_no_misleading_recovery(intrinsic, has_actions):
    affine = np.eye(4)
    if intrinsic == 'unsupported-director-grid':
        affine[0, 1] = .2
    engine = SurgicalEngine(_scene(target=intrinsic != 'missing-owned-tumor-target', affine=affine))
    if has_actions:
        _inspect(engine)
    before = engine.snapshot()
    result = full_case_readiness(engine)
    assert result['status'] == 'blocked'
    assert [b['code'] for b in result['blockers']] == [intrinsic, 'missing-protection']
    assert result['availableActions'] == ['inspect', 'replay', 'seekInitial' if has_actions else 'attachProtection']
    assert result['recovery'] is None
    assert engine.snapshot() == before
