"""Generated milling declaration/query controls, never patient qualification."""
import copy
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from resectionlab.core import array_digest, immutable_array
from resectionlab.surgical_craniotomy import PROPERTY as FLAP_PROPERTY
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_scene import SurgicalScene, _bind_original_target_memberships
from resectionlab.surgical_milling import (
    BASE_CASE_HASH, PROPERTY, SCHEMA, MillingContract, binding, contract_hash,
    default_evidence, is_milling, milling_readiness, require_coverage,
    resolve_contract, validate_contract, validate_source_scope)
from test_surgical_bone_transport import shell


def fixture(*, reviewed=True, source=None, extra=None):
    generated, _, _, _ = shell()
    old = generated.scene
    props = {key: copy.deepcopy(value) for key, value in old.properties.items() if key != FLAP_PROPERTY}
    props['startingStage'] = 'declared generated post-scalp exposure; dura unknown'
    props['tissuePhysicsV2'] = default_profile()
    props.update(extra or {})
    if props.get('tissuePhysicsV2') is None:
        props.pop('tissuePhysicsV2', None)
    base = SurgicalScene(old.labels, old.affine, old.regions, old.source if source is None else source, props)
    aperture = sorted(int(np.ravel_multi_index((x, y, 60), base.shape))
                      for x in range(28, 37) for y in range(28, 37))
    bone = set(map(int, np.flatnonzero(base.labels == 1)))
    evidence = default_evidence('idealized-engineering')
    if reviewed:
        for key, proof in evidence.items():
            if key != 'scope':
                proof.update(status='reviewed', provenance='Declared generated bone-only control; not patient evidence', evidenceSha256='1'*64)
    raw = {'schema': SCHEMA, 'method': 'milling', **binding(base),
        'apertureCells': aperture, 'retainedSkullCells': sorted(bone-set(aperture)),
        'boneAnchorCells': props['boneAnchors'][:],
        'entry': {'centerRasMm': [32., 32., 59.5], 'normalRasUnit': [0., 0., 1.],
            'uRasUnit': [1., 0., 0.], 'halfExtentsMm': [6., 6.],
            'maximumInwardMm': 30., 'maximumAngleDegrees': 30.},
        'exit': {'axis': 2, 'side': -1, 'planeVoxelBoundary': 59.,
            'minimumVoxelBoundary': [-.5]*3, 'maximumVoxelBoundary': [63.5, 63.5, 95.5]},
        'evidence': evidence}
    raw['contentHash'] = contract_hash(raw)
    scene = SurgicalScene(base.labels, base.affine, base.regions, base.source, {**props, PROPERTY: raw})
    return SurgicalEngine(scene), raw


def rebuild(engine, raw, *, source=None, labels=None, affine=None, properties=None):
    old = engine.scene
    props = {key: copy.deepcopy(value) for key, value in old.properties.items() if key != PROPERTY}
    props.update(properties or {})
    base = SurgicalScene(old.labels if labels is None else labels,
        old.affine if affine is None else affine, old.regions,
        old.source if source is None else source, props)
    raw = copy.deepcopy(raw)
    raw.update(binding(base)); raw['contentHash'] = contract_hash(raw)
    scene = SurgicalScene(base.labels, base.affine, base.regions, base.source, {**props, PROPERTY: raw})
    return SurgicalEngine(scene), raw


def test_exact_source_bound_milling_is_immutable_with_explicit_nonflap_aliases():
    engine, raw = fixture()
    before = engine.snapshot()
    contract = resolve_contract(engine.scene)
    assert isinstance(contract, MillingContract) and is_milling(contract)
    assert contract.as_dict() == raw
    assert contract.method == 'milling'
    assert contract.aperture_cells == contract.window_cells == contract.kerf_cells
    assert contract.flap_cells == () and len(contract.aperture_cells) == 81
    assert contract.bone_anchor_cells
    assert validate_contract(engine.scene, contract) == contract
    assert require_coverage(contract) == contract
    assert validate_source_scope(engine) == contract
    assert engine.scene.properties['startingStage'].endswith('dura unknown')
    with pytest.raises(FrozenInstanceError):
        contract.content_hash = ''
    detached = contract.as_dict(); detached['apertureCells'].clear()
    assert contract.aperture_cells
    assert engine.snapshot() == before


def test_v1_resolver_and_evidence_are_exact_unchanged_fallback():
    from resectionlab.surgical_craniotomy import require_evidence, validate_contract as old_validate
    old, _, _, _ = shell()
    before = old.snapshot()
    contract = resolve_contract(old.scene)
    assert contract == old_validate(old.scene) and not is_milling(contract)
    assert require_coverage(contract) == require_evidence(contract)
    assert validate_source_scope(old) == contract
    assert milling_readiness(old) is None
    assert old.snapshot() == before


def test_default_unknown_evidence_is_inspectable_but_blocked():
    engine, raw = fixture(reviewed=False)
    contract = resolve_contract(engine.scene)
    assert contract.evidence_scope == 'idealized-engineering'
    assert all(proof.status == 'unknown' and proof.evidence_sha256 is None for proof in contract.proofs)
    with pytest.raises(ValueError, match='WINDOW_EVIDENCE_REQUIRED'):
        require_coverage(contract)
    result = milling_readiness(engine)
    assert result['apertureCleared'] is False and result['openingComplete'] is False
    assert 'WINDOW_EVIDENCE_REQUIRED' in result['reason']


@pytest.mark.parametrize('name', ['window', 'boneScope', 'innerBoundary', 'exit'])
def test_every_independent_proof_is_required(name):
    engine, raw = fixture()
    raw['evidence'][name].update(status='unreviewed', evidenceSha256=None)
    engine, raw = rebuild(engine, raw)
    with pytest.raises(ValueError, match='EVIDENCE_REQUIRED'):
        validate_source_scope(engine)


@pytest.mark.parametrize('change', ['source', 'labels', 'affine', 'target', 'stage', 'attachments'])
def test_source_bindings_cannot_be_reused_after_any_independent_input_changes(change):
    engine, raw = fixture()
    old = engine.scene
    source, regions = copy.deepcopy(old.source), copy.deepcopy(old.regions)
    props = {key: copy.deepcopy(value) for key, value in old.properties.items() if key != PROPERTY}
    labels, affine = old.labels.copy(), old.affine.copy()
    if change == 'source': source['kind'] = 'different source'
    elif change == 'labels': labels[32, 32, 68] = 0
    elif change == 'affine': affine[0, 3] += .1
    elif change == 'target': source['annotationHashes'] = {'Generated target': 'different target'}
    elif change == 'stage': props['startingStage'] = 'undeclared dural opening'
    else: props['attachments'].append({'id': 'new', 'kind': 'periosteal', 'cells': [raw['apertureCells'][0]]})
    # The new scene validator may reject before a native engine is constructed.
    with pytest.raises(ValueError, match='binding'):
        scene = SurgicalScene(labels, affine, regions, source, {**props, PROPERTY: raw})
        validate_contract(scene)


@pytest.mark.parametrize('change', ['overlap', 'omit', 'anchor_in_A', 'no_anchors', 'empty_A', 'extra_key'])
def test_exact_partition_anchor_and_schema_rules(change):
    engine, raw = fixture()
    if change == 'overlap': raw['retainedSkullCells'] = sorted([*raw['retainedSkullCells'], raw['apertureCells'][0]])
    elif change == 'omit': raw['retainedSkullCells'].pop()
    elif change == 'anchor_in_A': raw['boneAnchorCells'] = [raw['apertureCells'][0]]
    elif change == 'no_anchors': raw['boneAnchorCells'] = []
    elif change == 'empty_A': raw['apertureCells'] = []
    else: raw['flapCells'] = []  # No v1-empty-flap shortcut in the new schema.
    raw['contentHash'] = contract_hash(raw)
    with pytest.raises(ValueError):
        rebuild(engine, raw)


def test_explicit_empty_unknown_support_is_not_silently_filled_from_other_state():
    engine, raw = fixture()
    raw['boneAnchorCells'] = []
    engine, raw = rebuild(engine, raw, properties={'boneAnchors': []})
    assert resolve_contract(engine.scene).bone_anchor_cells == ()
    with pytest.raises(ValueError, match='SUPPORT_EVIDENCE_REQUIRED'):
        validate_source_scope(engine)
    assert not milling_readiness(engine)['openingComplete']


@pytest.mark.parametrize('membership', ['protected', 'original_target', 'unknown_original_target'])
def test_full_A_preserves_protected_and_original_target_memberships(membership):
    engine, raw = fixture(); cell = raw['apertureCells'][0]
    if membership == 'protected':
        engine._protected_ids = frozenset([cell])
        reason = 'PROTECTED_STRUCTURE'
    else:
        mask = engine.scene.labels == 2; mask.ravel()[cell] = True
        engine, raw = rebuild(engine, raw, source={**engine.scene.source,
            'annotationHashes': {'Generated target': array_digest(mask)}})
        if membership == 'original_target':
            _bind_original_target_memberships(engine.scene,
                {'Generated target': immutable_array(np.flatnonzero(mask), np.int64)})
        reason = 'TARGET_OVERLAP' if membership == 'original_target' else 'ORIGINAL_TARGET_EVIDENCE_REQUIRED'
    with pytest.raises(ValueError, match=reason):
        validate_source_scope(engine)


def test_raw_partial_or_complete_removal_and_clearance_flags_are_not_execution_proof():
    for full in (False, True):
        engine, raw = fixture()
        engine.state['removed'] = raw['apertureCells'][:] if full else raw['apertureCells'][:1]
        engine.state['debrisRemovedMm3'] = float(len(engine.state['removed']))
        engine.state['millingComplete'] = True
        result = milling_readiness(engine)
        assert not result['historyVerified'] and not result['apertureCleared']
        assert not result['openingComplete'] and result['reason'] == 'MILLING_NATIVE_HISTORY_REQUIRED'
        assert result['generatedBoneDebrisMm3'] is None


@pytest.mark.parametrize('change', ['remove_R', 'hold_R', 'move_R', 'detached_R'])
def test_retained_skull_cannot_be_removed_held_or_transformed(change):
    engine, raw = fixture(); cell = raw['retainedSkullCells'][0]
    if change == 'remove_R': engine.state['removed'] = [cell]
    elif change == 'move_R': engine.state['deformation'] = {str(cell): [1, 0, 0]}
    elif change == 'hold_R': engine.state['holds']['left'] = {'cells': [cell], 'offset': [0, 0, 0]}
    else: engine.state['detachedFragments'] = [{'cells': [cell], 'offset': [0, 0, 0]}]
    with pytest.raises(ValueError, match='RETAINED_SKULL_CHANGED'):
        validate_source_scope(engine)


def test_induced_case_hash_projection_excludes_only_the_new_computed_field():
    engine, raw = fixture(source={'kind': 'generated', 'caseHash': 'sha256:'+'1'*64,
                                  BASE_CASE_HASH: 'sha256:'+'2'*64})
    before = binding(engine.scene)
    source = {**engine.scene.source, 'caseHash': 'sha256:'+'3'*64}
    engine, _ = rebuild(engine, raw, source=source)
    assert binding(engine.scene) == before
    source = {**source, 'unrelated': 'changed'}
    other, _ = rebuild(engine, raw, source=source)
    assert binding(other.scene) != before


def test_typed_forgery_is_fully_revalidated_and_contracts_are_exclusive():
    engine, raw = fixture()
    forged = replace(resolve_contract(engine.scene), aperture_cells=())
    with pytest.raises(ValueError):
        validate_contract(engine.scene, forged)
    old, _, _, _ = shell()
    with pytest.raises(ValueError, match='exclusive'):
        SurgicalScene(engine.scene.labels, engine.scene.affine, engine.scene.regions,
            engine.scene.source, {**engine.scene.properties, FLAP_PROPERTY: old.scene.properties[FLAP_PROPERTY]})


def test_empty_initial_native_state_is_not_milled_or_treatment_ready():
    engine, _ = fixture()
    result = milling_readiness(engine)
    assert result['historyVerified'] and result['debrisConserved']
    assert result['generatedBoneDebrisMm3'] == result['debrisPresentMm3'] == 0.
    assert not result['apertureRemoved'] and not result['apertureCleared']
    assert not result['openingComplete'] and not result['treatmentEntrySupported']


def test_every_face_connected_retained_component_needs_explicit_support():
    engine, raw = fixture()
    labels = engine.scene.labels.copy(); labels[10, 10, 70] = 1
    island = int(np.ravel_multi_index((10, 10, 70), labels.shape))
    raw['retainedSkullCells'] = sorted([*raw['retainedSkullCells'], island])
    engine, raw = rebuild(engine, raw, labels=labels)
    with pytest.raises(ValueError, match='RETAINED_SUPPORT_REQUIRED'):
        validate_source_scope(engine)
    assert not milling_readiness(engine)['retainedSupportVerified']
    raw['boneAnchorCells'] = sorted([*raw['boneAnchorCells'], island])
    engine, _ = rebuild(engine, raw)
    assert validate_source_scope(engine).bone_anchor_cells == tuple(raw['boneAnchorCells'])


def test_interior_exit_plane_and_whole_affine_window_corners_are_invalid():
    engine, raw = fixture()
    raw['exit']['planeVoxelBoundary'] = 60.
    with pytest.raises(ValueError, match='original source exterior'):
        rebuild(engine, raw)
    engine, raw = fixture()
    raw['entry']['halfExtentsMm'] = [4., 4.]  # A extends to4.5 at full cell faces.
    with pytest.raises(ValueError, match='full affine cells'):
        rebuild(engine, raw)


@pytest.fixture(scope='module')
def native_progress():
    """Actual finite-work public native prefixes; no state injection positive."""
    from resectionlab.surgical_transition_meter import TransitionMeter
    from test_surgical_milling_native import milling_shell, checked
    from test_surgical_bone_transport import pose
    source, aperture = milling_shell()
    meter = TransitionMeter(40)
    engine = meter.wrap(source).clone()
    checked(engine, 'configure', mechanicsRevision=2, clockRevision=1)
    checked(engine, 'place', pose=pose(29.5, 29.5, 55.))
    checked(engine, 'apply', pose=pose(29.5, 29.5, 58.3), duration=30., allow_capacity_partial=True)
    result = milling_readiness(engine)
    assert result['historyVerified'] and result['debrisConserved'], result
    assert 0 < result['removedBoneMm3'] < len(aperture)
    return engine.unwrap().clone(), source.snapshot(), meter.report()


def reseal_last_history_state_for_negative(engine):
    """Deliberate state/history forgery; never a native-positive fixture."""
    engine.history[-1]['stateAfter'] = engine.state_id
    if 'stateId' in engine.history[-1]:
        engine.history[-1]['stateId'] = engine.state_id


def test_real_capacity_partial_has_complete_native_bone_debris_provenance(native_progress):
    engine, source_snapshot, meter = native_progress
    result = milling_readiness(engine)
    assert result['historyVerified'] and result['debrisConserved']
    assert result['generatedBoneDebrisMm3'] == result['removedBoneMm3'] == result['debrisPresentMm3']
    assert result['debrisRemovedMm3'] == result['debrisEscapedMm3'] == 0.
    assert not result['apertureCleared'] and not result['openingComplete']
    assert source_snapshot['actions'] == [] and source_snapshot['checkpoint']['state']['removed'] == []
    assert meter['physicalCalls'] == 6


@pytest.mark.parametrize('counter', ['debrisRemovedMm3', 'debrisEscapedMm3', 'debrisMm3'])
def test_unknown_debris_counter_cannot_be_inferred_from_aperture_or_display(native_progress, counter):
    source, _, _ = native_progress
    engine = source.clone(); engine.state[counter] = None
    reseal_last_history_state_for_negative(engine)
    result = milling_readiness(engine)
    assert result['historyVerified']
    assert result['reason'] == 'MILLING_DEBRIS_EVIDENCE_REQUIRED'
    assert not result['debrisConserved'] and not result['apertureCleared']


def test_escaped_bone_debris_can_conserve_mass_but_never_count_as_clearance(native_progress):
    source, _, _ = native_progress
    engine = source.clone()
    key = next(iter(engine.state['debris']))
    escaped = engine.state['debris'][key]
    del engine.state['debris'][key]
    engine.state['debrisMm3'] -= escaped
    engine.state['debrisEscapedMm3'] = escaped
    reseal_last_history_state_for_negative(engine)
    result = milling_readiness(engine)
    assert result['historyVerified'] and result['debrisConserved']
    assert result['debrisEscapedMm3'] > 0.
    assert not result['apertureCleared'] and not result['openingComplete']


@pytest.mark.parametrize('change', ['invent_suction', 'erase_present', 'extra_present', 'fake_escape'])
def test_global_conservation_and_actual_aspiration_receipts_are_required(native_progress, change):
    source, _, _ = native_progress
    engine = source.clone()
    if change == 'invent_suction':
        engine.state['debrisRemovedMm3'] = engine.state['debrisMm3']
        engine.state['debris'].clear(); engine.state['debrisMm3'] = 0.
    elif change == 'erase_present':
        engine.state['debris'].clear(); engine.state['debrisMm3'] = 0.
    elif change == 'extra_present':
        key = next(iter(engine.state['debris']))
        engine.state['debris'][key] += 1.; engine.state['debrisMm3'] += 1.
    else:
        engine.state['debrisEscapedMm3'] = 1.
    reseal_last_history_state_for_negative(engine)
    result = milling_readiness(engine)
    assert result['historyVerified']
    assert not result['debrisConserved'] and not result['apertureCleared']


@pytest.mark.parametrize('change', ['orphan', 'collision_partial', 'capacity_no_time', 'broken_chain', 'configure_removal'])
def test_native_history_requires_exact_accepted_ancestry_and_supported_capacity_progress(native_progress, change):
    source, _, _ = native_progress
    engine = source.clone()
    event = engine.history[-1]
    if change == 'orphan': engine.actions.clear()
    elif change == 'collision_partial': event.update(status='partial', reason='SHAFT_COLLISION')
    elif change == 'capacity_no_time': event.update(status='partial', reason='BONE_CUTTING_IN_PROGRESS', completedFraction=0., elapsedSeconds=0.)
    elif change == 'broken_chain': event['stateBefore'] = 'unrelated'
    else:
        engine.actions[-1]['operation'] = 'configure'
        event['action']['operation'] = 'configure'
    result = milling_readiness(engine)
    assert not result['historyVerified'] and result['reason'] == 'MILLING_NATIVE_HISTORY_REQUIRED'
    assert not result['apertureCleared']


def test_unposed_canonical_hand_does_not_impose_a_phantom_nonZ_exit():
    from resectionlab.surgical_milling import _withdrawn
    engine, raw = fixture()
    raw['exit'].update(axis=0, planeVoxelBoundary=19.)
    engine, _ = rebuild(engine, raw)
    contract = resolve_contract(engine.scene)
    # This is only an exterior-status query. A is intact; no opening is ready.
    assert _withdrawn(engine, contract)
    assert not milling_readiness(engine)['openingComplete']
    engine.state['slots']['right']['active'] = True
    assert not _withdrawn(engine, contract)
    engine.state['slots']['right']['active'] = False
    engine.actions.append({'slot': 'right', 'operation': 'place'})
    assert not _withdrawn(engine, contract)


@pytest.mark.parametrize('change', ['boolean_fraction', 'wrong_volume', 'duplicate_source_id'])
def test_progress_receipts_require_strict_native_numbers_and_unique_source_cells(native_progress, change):
    source, _, _ = native_progress
    engine = source.clone(); event = engine.history[-1]
    if change == 'boolean_fraction':
        event.update(status='accepted', completedFraction=True)
    else:
        effect = next(e for e in event['events'] if e.get('kind') == 'drill_work' and e.get('removedCells'))
        if change == 'wrong_volume': effect['removedVolumeMm3'] += 1.
        else: effect['removedCells'].append(effect['removedCells'][0])
    result = milling_readiness(engine)
    assert not result['historyVerified'] and not result['apertureCleared']


def test_legacy_instantaneous_bone_model_is_inspectable_but_not_milling_execution():
    engine, _ = fixture(extra={'tissuePhysicsV2': None})
    assert resolve_contract(engine.scene).method == 'milling'
    with pytest.raises(ValueError, match='V2_BONE_PROFILE_REQUIRED'):
        validate_source_scope(engine)
    assert not milling_readiness(engine)['openingComplete']


def test_zero_spatial_fraction_can_retain_real_native_time_and_pending_drill_work():
    from resectionlab.surgical_transition_meter import TransitionMeter
    from test_surgical_milling_native import milling_shell, checked
    from test_surgical_bone_transport import pose
    source, _ = milling_shell(); meter = TransitionMeter(20)
    engine = meter.wrap(source).clone()
    checked(engine, 'configure', mechanicsRevision=2, clockRevision=1)
    checked(engine, 'place', pose=pose(29.5, 29.5, 55.))
    for _ in range(2):
        result = checked(engine, 'apply', pose=pose(29.5, 29.5, 58.3), duration=.1, allow_capacity_partial=True)
    assert result['status'] == 'partial' and result['reason'] == 'BONE_CUTTING_IN_PROGRESS'
    assert result['completedFraction'] == 0. and result['elapsedSeconds'] > 0.
    assert engine.state['drillingV2']['cuttingWorkJ'] > 0.
    status = milling_readiness(engine)
    assert status['historyVerified'] and status['debrisConserved'], status
    assert not status['openingComplete'] and not status['apertureCleared']
    assert meter.calls == 8 and source.actions == []


@pytest.mark.parametrize('kind', ['bone_kerf', 'removal'])
def test_instant_cell_generation_never_substitutes_for_finite_milling(native_progress, kind):
    source, _, _ = native_progress
    engine = source.clone()
    effect = next(e for e in engine.history[-1]['events'] if e.get('kind') == 'drill_work' and e.get('removedCells'))
    effect['kind'] = kind
    assert not milling_readiness(engine)['historyVerified']


@pytest.mark.parametrize('change', ['missing_source', 'wrong_slot', 'wrong_tool', 'wrong_operation'])
def test_per_effect_native_source_provenance_is_required(native_progress, change):
    source, _, _ = native_progress
    engine = source.clone()
    effect = next(e for e in engine.history[-1]['events'] if e.get('kind') == 'drill_work')
    if change == 'missing_source': effect.pop('sourceSlot')
    elif change == 'wrong_slot': effect['sourceSlot'] = 'right'
    elif change == 'wrong_tool': effect['instrumentId'] = 'frazier_suction'
    else: effect['effectOperation'] = 'configure'
    result = milling_readiness(engine)
    assert not result['historyVerified'] and not result['apertureCleared']
