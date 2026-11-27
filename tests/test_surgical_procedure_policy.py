"""Generated source declarations test permission; no acquired anatomy claims.

Since the 2026-11-27 owner decision an acquired-class source with the same
structural declaration is admitted and labelled; it is never refused for class.
"""
import copy

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_sequence_outcomes import sequence_outcome
from resectionlab.surgical_procedure_policy import (
    PROPERTY, make_procedure_policy, procedure_admission, procedure_protection,
    procedure_source_memberships, evaluate_procedure_sequence,
)
from resectionlab.surgical_rl_protection import ProtectionSet


def source(*, neural_limit=0., overlap=False, critical_shortcut=False, unknown=False, cutting=False, bone=False):
    labels = np.zeros((16, 16, 40), np.uint16)
    labels[8, 8, 20] = labels[11, 8, 20] = 1
    labels[8, 8, 18] = labels[2, 2, 20] = labels[2, 3, 20] = 2
    if cutting:
        labels[8, 9, 18] = 2
    if bone:
        labels[11, 8, 18] = 3
    regions = [dict(id=1, name='Target owner', material='tumor', target=True),
        dict(id=2, name='Mechanically soft', material='soft_tissue')]
    if bone:
        regions.append(dict(id=3, name='Bone owner', material='bone'))
    base = SurgicalScene(labels, np.diag([5., 5., 5., 1.]), regions,
        {'kind': 'generated-control', 'provenance': 'Explicit independent generated geometry'}, {})
    index = lambda xyz: int(np.ravel_multi_index(xyz, labels.shape))
    brainstem = index((2, 2, 20))
    cerebellum = index((8, 8, 18)) if critical_shortcut else index((2, 3, 20))
    cortical = index((2, 3, 20)) if critical_shortcut else index((8, 8, 18))
    intracranial = list(map(int, np.flatnonzero(labels)))
    allowed = sorted(set(intracranial) - {brainstem, cerebellum})
    evidence = dict(origin='independent-source-declaration', reviewStatus='generated-engineering',
        evidenceSha256='a'*64, provenance='Declared generated source masks, not a proposed route')
    structures = {name: dict(evidence, kind='neural', cells=[cell], coverageCells=intracranial)
        for name, cell in [('brainstem', brainstem), ('cerebellum', cerebellum), ('cortex', cortical)]}
    if cutting:
        structures['cortex']['cells'].append(index((8, 9, 18)))
        structures['cortex']['cells'].sort()
    if overlap:
        structures['cortex']['cells'].append(index((11, 8, 20)))
        structures['cortex']['cells'].sort()
    observed = sorted(set(range(labels.size)) - ({index((11, 8, 18))} if unknown else set()))
    raw = make_procedure_policy(base, policy_id='generated-procedure-v1', evidence=evidence,
        structures=structures, intracranial_cells=intracranial, observed_cells=observed,
        entry_regions={'supplied-entry': dict(evidence, cells=allowed)},
        permitted_compartments={'supplied-work': dict(evidence, cells=allowed)},
        work_domain=dict(evidence, minimumVoxelBoundary=[-.5]*3,
            maximumVoxelBoundary=(np.asarray(labels.shape)-.5).tolist()),
        tissue_permissions={row['material']: dict(remove=True, cut=True) for row in regions},
        neural_max_mm3=neural_limit,
        bone_max_mm3=0., preparation_nonbone_max_mm3=0., modeled_time_max_seconds=240.,
        structure_max_mm3={'brainstem': 0., 'cerebellum': 0., 'cortex': neural_limit})
    return SurgicalScene(labels, base.affine, regions, base.source, {PROPERTY: raw})


def revised(scene, change):
    raw = copy.deepcopy(scene.properties[PROPERTY]); change(raw)
    raw['policyHash'] = semantic_digest({k: v for k, v in raw.items() if k != 'policyHash'})
    return SurgicalScene(scene.labels, scene.affine, scene.regions, scene.source, {PROPERTY: raw})


def action(operation, position=None, **extra):
    value = dict(slot='right', operation=operation, duration=.1)
    value.update(extra)
    if position is not None:
        value['pose'] = dict(position=position, quaternion=[0., 0., 0., 1.])
    return value


def prepared(scene, x=55.):
    engine = SurgicalEngine(scene)
    result = engine.step(action('place', [x, 40., 65.]))
    assert result['status'] == 'accepted', result
    return engine


@pytest.mark.parametrize('missing', ['brainstem', 'cerebellum'])
def test_missing_critical_coverage_is_specific_and_cannot_be_research_acknowledged(missing):
    scene = revised(source(), lambda raw: raw['structures'].pop(missing))
    result = procedure_admission(scene)
    assert not result['available']
    assert result['blockers'][0]['code'] == 'PROCEDURE_MISSING_' + missing.upper() + '_COVERAGE'
    with pytest.raises(ValueError, match=result['blockers'][0]['code']):
        SurgicalEngine(scene)


def test_incomplete_intracranial_and_structure_negative_coverage_stays_unknown():
    scene = revised(source(), lambda raw: raw['structures']['cerebellum'].update(coverageCells={'runs': []}))
    assert not procedure_admission(scene)['available']
    scene = revised(source(), lambda raw: raw.update(intracranialCells={'runs': []}))
    assert not procedure_admission(scene)['available']


@pytest.mark.parametrize('change,reason', [
    (lambda raw: raw['workDomain'].update(origin='planner-proposed-tunnel'), 'PROCEDURE_INDEPENDENT_EVIDENCE_REQUIRED'),
    (lambda raw: raw['evidence'].update(reviewStatus=''), 'PROCEDURE_REVIEW_STATUS_LABEL_REQUIRED'),
    (lambda raw: raw['structures']['cortex'].pop('reviewStatus'), 'PROCEDURE_REVIEW_STATUS_LABEL_REQUIRED'),
    (lambda raw: raw.update(maximumStructureRemovalMm3={'brainstem': 0., 'cerebellum': 1.}), 'PROCEDURE_EXPLICIT_NEURAL_BUDGET_REQUIRED'),
])
def test_no_self_admitted_domain_or_relaxed_critical_budget(change, reason):
    result = procedure_admission(revised(source(), change))
    assert not result['available'] and result['blockers'][0]['code'] == reason


def test_stronger_existing_critical_requirement_is_preserved_and_not_unknown_zero():
    scene = source()
    protection = ProtectionSet(scene, required=('optic-tract',), clearance_mm=3.)
    assert not procedure_admission(scene, protection)['available']
    merged = procedure_protection(scene, protection)
    assert set(merged.required) == {'brainstem', 'cerebellum', 'optic-tract'}
    assert merged.clearance_mm == 3.


@pytest.mark.parametrize('critical_shortcut', [False, True])
def test_neural_budget_or_cerebellar_shortcut_refuses_without_any_committed_change(critical_shortcut):
    engine = prepared(source(critical_shortcut=critical_shortcut), x=40.)
    before = copy.deepcopy(engine.snapshot())
    proposal = action('apply', [40., 40., 100.], duration=1.)
    preview = engine.preview(proposal, include_view=False)
    result = engine.step(proposal)
    assert result['status'] == preview['status'] == 'rejected'
    assert result['reasonCode'] == ('PROTECTED_STRUCTURE' if critical_shortcut else 'PROCEDURE_HARM_BUDGET_EXCEEDED')
    assert engine.snapshot() == before
    assert sequence_outcome(engine)['dimensions']['targetRemovedMm3']['cumulative'] == 0.


def test_unknown_interior_and_cropped_shaft_are_forbidden_without_committing():
    engine = prepared(source(unknown=True))
    before = engine.snapshot()
    result = engine.step(action('apply', [55., 40., 100.], duration=1.))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'PROTECTED_STRUCTURE'
    assert engine.snapshot() == before
    result = engine.step(action('move', [55., 40., 5.], duration=1.))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'PROCEDURE_WORK_DOMAIN_VIOLATION'
    assert engine.snapshot() == before


def test_native_cut_edges_cannot_bypass_a_zero_neural_cut_budget_with_zero_removal():
    engine = SurgicalEngine(source(cutting=True))
    assert engine.step(action('exchange', instrument='scalpel'))['status'] == 'accepted'
    assert engine.step(action('place', [40., 42.5, 65.]))['status'] == 'accepted'
    before = engine.snapshot()
    result = engine.step(action('apply', [40., 42.5, 90.], duration=1.))
    assert result['status'] == 'rejected', result
    assert result['reasonCode'] == 'PROCEDURE_HARM_BUDGET_EXCEEDED'
    assert result['constraintViolations'][0]['metric'] == 'neuralCutSourceMm3'
    assert engine.snapshot() == before
    assert sequence_outcome(engine)['dimensions']['neuralCutSourceMm3']['cumulative'] == 0.


def test_bone_and_modeled_time_caps_are_precommit_and_independent_of_neural_budget():
    engine = SurgicalEngine(source(bone=True))
    assert engine.step(action('exchange', instrument='cranial_drill'))['status'] == 'accepted'
    assert engine.step(action('place', [55., 40., 65.]))['status'] == 'accepted'
    before = engine.snapshot()
    result = engine.step(action('apply', [55., 40., 90.], duration=1.))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'PROCEDURE_HARM_BUDGET_EXCEEDED'
    assert result['constraintViolations'][0]['metric'] == 'boneRemovedMm3'
    assert engine.snapshot() == before
    scene = revised(source(), lambda raw: raw.update(maximumModeledElapsedSeconds=.15))
    engine = prepared(scene); before = engine.snapshot()
    result = engine.step(action('inspect'))
    assert result['status'] == 'rejected' and result['constraintViolations'][0]['metric'] == 'elapsedSeconds'
    assert engine.snapshot() == before


def test_overlap_removal_is_neural_even_when_categorical_owner_is_target_and_survives_replay():
    engine = prepared(source(neural_limit=125., overlap=True))
    result = engine.step(action('apply', [55., 40., 100.], duration=1.))
    assert result['status'] == 'accepted', result
    outcome = sequence_outcome(engine)
    dims = outcome['dimensions']
    assert dims['targetRemovedMm3']['cumulative'] == dims['neuralRemovedMm3']['cumulative'] == pytest.approx(125.)
    assert dims['structureRemovedMm3']['cortex']['cumulative'] == pytest.approx(125.)
    assert dims['boneRemovedMm3']['cumulative'] == 0.
    assert dims['motorEncounterMm3']['status'] == 'unavailable'
    assert evaluate_procedure_sequence(engine)['admissible']
    assert engine.step(action('withdraw', [55., 40., 65.], duration=1.))['status'] == 'accepted'
    assert sequence_outcome(engine)['withdrawalComplete']
    restored = SurgicalEngine.restore(engine.scene, engine.snapshot())
    assert restored.snapshot() == engine.snapshot()
    assert sequence_outcome(restored) == sequence_outcome(engine)
    stale = copy.deepcopy(engine.snapshot()); stale.pop('procedurePolicy')
    with pytest.raises(ValueError, match='PROCEDURE_SNAPSHOT_INCOMPATIBLE'):
        SurgicalEngine.restore(engine.scene, stale)


def test_critical_membership_hidden_by_target_ownership_still_blocks_the_complete_tool():
    scene = source()
    memberships = procedure_source_memberships(scene)
    target = int(np.ravel_multi_index((11, 8, 20), scene.shape))
    critical = sorted(memberships['structure:cerebellum'] | {target})
    scene = revised(scene, lambda raw: raw['structures']['cerebellum'].update(cells=critical))
    engine = prepared(scene)
    before = engine.snapshot()
    result = engine.step(action('apply', [55., 40., 100.], duration=1.))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'PROTECTED_STRUCTURE'
    assert engine.snapshot() == before
    assert sequence_outcome(engine)['dimensions']['targetRemovedMm3']['cumulative'] == 0.


def relabelled(raw, status):
    """Replace every declared review label; structure and cells are unchanged."""
    raw = copy.deepcopy(raw)
    for row in (raw['evidence'], raw['workDomain'], *raw['structures'].values(),
                *raw['entryRegions'].values(), *raw['permittedCompartments'].values()):
        row['reviewStatus'] = status
    return raw


def acquired_scene(original, marker, status='unreviewed'):
    from resectionlab.surgical_procedure_policy import procedure_source_binding
    provenance = {**original.source, **marker}
    unbound = SurgicalScene(original.labels, original.affine, original.regions, provenance, {})
    raw = relabelled(original.properties[PROPERTY], status)
    raw['sourceBinding'] = procedure_source_binding(unbound)
    raw['policyHash'] = semantic_digest({k: v for k, v in raw.items() if k != 'policyHash'})
    return SurgicalScene(original.labels, original.affine, original.regions, provenance, {PROPERTY: raw})


def test_unreviewed_label_on_generated_source_is_admitted_and_carried_into_protection():
    scene = revised(source(), lambda raw: raw.update(relabelled(raw, 'unreviewed')))
    assert procedure_admission(scene)['available']
    engine = SurgicalEngine(scene)
    rows = {a.structure_id: a.review_status for a in engine.protection.annotations}
    assert rows['brainstem'] == rows['procedure-forbidden-or-unknown'] == 'unreviewed'


@pytest.mark.parametrize('review', ['generated-engineering', 'reviewed', 'unreviewed', 'estimated'])
def test_acquired_adapter_is_admitted_with_its_label_but_cannot_claim_generated(review):
    adapter = {'adapter': {'type': 'native-patient-research-derivative', 'sourcePatient': 'supplied'}}
    acquired = acquired_scene(source(), adapter, review)
    result = procedure_admission(acquired)
    if review == 'generated-engineering':
        # A generated label would misstate this source's provenance.
        assert result['blockers'][0]['code'] == 'PROCEDURE_GENERATED_LABEL_ON_NONGENERATED_SOURCE'
        return
    assert result['available'] and not result['blockers']
    engine = SurgicalEngine(acquired)
    assert {a.review_status for a in engine.protection.annotations} == {review}


@pytest.mark.parametrize('marker', [
    {'patientId': 'supplied'}, {'patient_id': 'supplied'}, {'source_patient_id': 'supplied'}, {'source_patient': 'supplied'},
    {'source_manifest': {}}, {'source_collection_hash': 'b'*64},
    {'adapter': {'patientId': 'supplied'}}, {'adapter': {'patient_id': 'supplied'}}, {'adapter': {'source_patient': 'supplied'}},
    *[{'adapter': {'type': kind}} for kind in ('nifti', 'nifti-import', 'mesh', 'mesh-derived', 'native-patient')],
])
def test_freshly_bound_policy_on_patient_marked_source_is_admitted_and_keeps_its_class(marker):
    # Owner decision 2026-11-27: the class is reported, never a refusal; the
    # structural procedure declaration is what admission still checks.
    from resectionlab.surgical_planning_admission import planning_admission
    contradictory = acquired_scene(source(), marker)
    admitted = planning_admission(contradictory)
    assert admitted['sourceClass'] == 'patient-or-unverified' and admitted['available']
    assert admitted['scope'] == 'procedure-aware' and not admitted['blockers']
    assert admitted['sourceRole']['role'] == 'UNKNOWN' and admitted['sourceRole']['verified'] is False
    assert procedure_admission(contradictory)['available']
    engine = SurgicalEngine(contradictory)
    assert engine.snapshot()['procedurePolicy']['policyHash'] == contradictory.properties[PROPERTY]['policyHash']
    from resectionlab.surgical_rl_env import SurgicalRLEnv
    env = SurgicalRLEnv(contradictory)
    _, info = env.reset(seed=3)
    assert info['source_class'] == 'patient-or-unverified' and info['procedure_admission']['available']
    assert env.target_basis == 'owned_target_labels'
    # A generated review label on this class remains a provenance contradiction.
    mislabelled = acquired_scene(source(), marker, 'generated-engineering')
    assert procedure_admission(mislabelled)['blockers'][0]['code'] == 'PROCEDURE_GENERATED_LABEL_ON_NONGENERATED_SOURCE'


def test_observable_memberships_are_explicit_frozen_source_sets_and_protection_is_idempotent():
    scene = source()
    memberships = procedure_source_memberships(scene)
    assert isinstance(memberships['observed'], frozenset)
    assert memberships['structure:cerebellum'] <= memberships['neural']
    engine = SurgicalEngine(scene)
    assert procedure_protection(scene, engine.protection) is engine.protection
    assert engine.fresh().snapshot() == engine.snapshot()


@pytest.mark.parametrize('status', ['generated-engineering', 'unreviewed'])
def test_observed_source_reference_cannot_be_promoted_by_generated_case_metadata(status):
    from dataclasses import replace
    from resectionlab.core import SourceRef, thaw_json
    from resectionlab.surgical_planning_admission import planning_admission
    from resectionlab.surgical_scene import from_case
    from resectionlab.surgical_procedure_policy import procedure_source_binding
    from test_structure_aware_cranial_planning import procedure_case
    case, _, record = procedure_case()
    metadata = thaw_json(case.metadata)
    raw = relabelled(metadata['surgical_properties'].pop(PROPERTY), status)
    observed = SourceRef('observed-acquired-image', 'file:///preserved/acquired-image.nii.gz',
        '1'*64, provenance='observed')
    spoof = replace(case, source_refs=(observed,), metadata=metadata)
    pre_policy = from_case(spoof, record['mapping'])
    raw['sourceBinding'] = procedure_source_binding(pre_policy)
    raw['policyHash'] = semantic_digest({k: v for k, v in raw.items() if k != 'policyHash'})
    metadata['surgical_properties'][PROPERTY] = raw
    spoof = replace(spoof, metadata=metadata)
    scene = from_case(spoof, record['mapping'])
    # Generated case metadata cannot promote the observed reference: it stays acquired.
    assert scene.source['procedurePolicySourceClass'] == 'acquired-or-unknown'
    assert planning_admission(scene)['sourceClass'] == 'patient-or-unverified'
    if status == 'generated-engineering':
        reason = 'PROCEDURE_GENERATED_LABEL_ON_NONGENERATED_SOURCE'
        assert procedure_admission(scene)['blockers'][0]['code'] == reason
        with pytest.raises(ValueError, match=reason):
            SurgicalEngine(scene)
        return
    assert procedure_admission(scene)['available']
    assert SurgicalEngine(scene).snapshot()['procedurePolicy']['policyHash'] == raw['policyHash']
    pytest.importorskip('gymnasium')
    from resectionlab.surgical_rl_env import SurgicalRLEnv
    env = SurgicalRLEnv(scene)
    assert env.source_class == 'patient-or-unverified' and env.target_basis == 'original_annotation'


@pytest.mark.parametrize('case', ['neural', 'cut', 'bone', 'time', 'tissue'])
def test_owned_transition_assessment_matches_complete_outcome_on_physical_candidates(monkeypatch, case):
    from resectionlab import surgical_procedure_policy as policy
    original = policy.procedure_transition_assessment
    compared = []
    def checked(candidate, *, source):
        fast = original(candidate, source=source)
        complete = evaluate_procedure_sequence(candidate)
        for key in ('admissible', 'violations', 'policyId', 'policyHash'):
            assert fast[key] == complete[key]
        compared.append(fast)
        return fast
    monkeypatch.setattr(policy, 'procedure_transition_assessment', checked)
    if case == 'neural':
        engine = prepared(source(), x=40.)
        proposal = action('apply', [40., 40., 100.], duration=1.)
    elif case == 'cut':
        engine = SurgicalEngine(source(cutting=True))
        assert engine.step(action('exchange', instrument='scalpel'))['status'] == 'accepted'
        assert engine.step(action('place', [40., 42.5, 65.]))['status'] == 'accepted'
        proposal = action('apply', [40., 42.5, 90.], duration=1.)
    elif case == 'bone':
        engine = SurgicalEngine(source(bone=True))
        assert engine.step(action('exchange', instrument='cranial_drill'))['status'] == 'accepted'
        assert engine.step(action('place', [55., 40., 65.]))['status'] == 'accepted'
        proposal = action('apply', [55., 40., 90.], duration=1.)
    elif case == 'time':
        engine = prepared(revised(source(), lambda raw: raw.update(maximumModeledElapsedSeconds=.15)))
        proposal = action('inspect')
    else:
        engine = prepared(revised(source(), lambda raw: raw['tissuePermissions']['tumor'].update(remove=False)))
        proposal = action('apply', [55., 40., 100.], duration=1.)
    before = engine.snapshot()
    assert engine.step(proposal)['status'] == 'rejected'
    assert engine.snapshot() == before
    assert any(row['admissible'] for row in compared) and any(not row['admissible'] for row in compared)


def test_transition_gate_preserves_protected_accounting_and_refuses_stale_source_history():
    from resectionlab.surgical_procedure_policy import procedure_transition_assessment
    engine = prepared(source())
    candidate, result = engine._prepare_step(action('inspect'))
    assert result['status'] == 'accepted'
    # A function-level corrupted receipt tests the shared protected-count gate;
    # the physical native protection guard independently prevents this effect.
    protected = next(iter(procedure_source_memberships(engine.scene)['structure:brainstem']))
    candidate.state['removed'].append(protected)
    candidate.history[-1]['stateAfter'] = candidate.state_id
    fast = procedure_transition_assessment(candidate, source=engine)
    complete = evaluate_procedure_sequence(candidate)
    assert fast['violations'] == complete['violations']
    assert any(row.get('metric') == 'protectedRemovedMm3' for row in fast['violations'])
    engine.state['elapsedSeconds'] += .01
    assert procedure_transition_assessment(candidate, source=engine)['violations'] == [
        {'code': 'PROCEDURE_TRANSITION_PREFIX_MISMATCH'}]


@pytest.mark.parametrize('standardized',[False,True])
def test_critical_shortcut_refusal_is_atomic_with_both_declared_physics_profiles(standardized):
    scene=source(critical_shortcut=True)
    if standardized:
        from resectionlab.surgical_environment import standardize
        scene=standardize(scene)
        from resectionlab.surgical_procedure_policy import procedure_source_binding
        raw=copy.deepcopy(scene.properties[PROPERTY])
        raw['sourceBinding']=procedure_source_binding(scene)
        raw['policyHash']=semantic_digest({k:v for k,v in raw.items() if k!='policyHash'})
        scene=SurgicalScene(scene.labels,scene.affine,scene.regions,scene.source,{**scene.properties,PROPERTY:raw})
    engine=prepared(scene,x=40.);before=engine.snapshot()
    command=action('apply',[40.,40.,100.],duration=1.)
    preview=engine.preview(command,include_view=False)
    actual=engine.step(command)
    assert preview['status']==actual['status']=='rejected'
    assert actual['reasonCode']=='PROTECTED_STRUCTURE'
    assert engine.snapshot()==before


def test_free_space_placement_and_withdrawal_require_continuous_exterior_access():
    # Completely remove the sole target on this axis. Its physical cavity is
    # free space, but the original intracranial domain still crosses the shaft.
    engine = prepared(source(neural_limit=125., overlap=True))
    assert engine.step(action('apply', [55., 40., 100.], duration=1.))['status'] == 'accepted'
    before = engine.snapshot()
    # The tip is distal to the cavity; it is the shaft that remains intracranial.
    inside = action('withdraw', [55., 40., 115.], duration=1.)
    for result in (engine.preview(inside, include_view=False), engine.step(inside)):
        assert result['reasonCode'] == 'PROCEDURE_EXTERIOR_WITHDRAWAL_REQUIRED'
    assert engine.snapshot() == before
    assert engine.step(action('withdraw', [55., 40., 65.], duration=1.))['status'] == 'accepted'
    outside = engine.snapshot()
    result = engine.step(action('place', [55., 40., 115.]))
    assert result['reasonCode'] == 'PROCEDURE_EXTERIOR_PLACEMENT_REQUIRED'
    assert engine.snapshot() == outside
