"""Every source class reaches planning under the same structural requirements.

Owner decision 2026-11-27: planning, search and the RL environment operate on
any reconstruction, including acquired patient anatomy. Source class is a
reported label, never a refusal; provenance tampering is still refused.
Tiny generated arrays test provenance routing, not acquired anatomy or outcomes.
"""
import copy
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene, from_case


SOURCE_KINDS = ('anonymous', 'generated-kind-spoof', 'native-adapter', 'direct-source-spoof', 'patient-key', 'manifest-key', 'ambiguous-case',
                'observed-case', 'generated-metadata-spoof', 'mixed-source-case')


def source(kind, *, as_case=False):
    labels = np.zeros((8, 8, 16), np.uint16)
    labels[4, 4, 8] = 1
    regions = [dict(id=1, name='Declared target', material='tumor', target=True)]
    direct = {
        'anonymous': {},
        'generated-kind-spoof': {'kind': 'generated clinical MRI'},
        'native-adapter': {'adapter': {'type': 'native-patient-research-derivative'}},
        'direct-source-spoof': {'kind': 'generated-control', 'evidenceClass': 'generated-engineering',
            'adapter': {'type': 'native-patient-research-derivative'}, 'patient': 'fabricated-routing-test'},
        'patient-key': {'patient': 'fabricated-routing-test'},
        'manifest-key': {'sourceManifest': 'fabricated-routing-test'},
        'ambiguous-case': {'caseHash': 'sha256:' + '1' * 64},
        'direct-generated': {'kind': 'generated-control', 'provenance': 'Explicit generated software geometry'},
    }
    if kind in direct:
        return SurgicalScene(labels, np.eye(4), regions, direct[kind], {})
    generated = kind in ('generated-metadata-spoof', 'mixed-source-case', 'generated-case')
    references = [SourceRef('observed-control', 'file:///fabricated-routing-test.nii', provenance='observed')]
    if kind == 'generated-case':
        references = [SourceRef('generated-control', 'synthetic://routing-test', provenance='simulated')]
    if kind == 'mixed-source-case':
        references.append(SourceRef('generated-control', 'synthetic://routing-test', provenance='simulated'))
    case = CaseData(case_id='generated-routing-test' if generated else 'observed-routing-test',
        mri=labels.astype(np.float32), compartments={'Declared target': labels == 1}, affine=np.eye(4),
        source_refs=tuple(references), metadata={
            'anatomy_layers': {'Declared target': {'kind': 'tumor'}},
            **({'generated': True, 'is_synthetic': True,
                'surgical_adapter': {'type': 'generated-engineering-control'}} if generated else {})})
    return case if as_case else from_case(case)


def generators(engine):
    from resectionlab.full_case_rehearsal import new_director, run_slice
    from resectionlab.surgical_access_planning import compare_access_approaches
    from resectionlab.surgical_cranial_planning import compare_cranial_strategies
    from resectionlab.surgical_interactive_planning import generate_interactive_approaches
    from resectionlab.surgical_joint_planning import compare_joint_strategies
    from resectionlab.surgical_planning import compare_continuations
    from resectionlab.surgical_preparation_planning import generate as prepare, suggest
    from resectionlab.surgical_precision_reentry import continue_precision_reentry
    from resectionlab.surgical_sector_planning import compare_sector_strategies, execute_strategy_option
    from resectionlab.surgical_staged_planning import compare_staged_strategies
    from resectionlab.surgical_staged_exit import recover_checked_cusa
    from resectionlab.surgical_strategy_planning import compare_rescue
    from resectionlab.surgical_two_tool import generate_comparison
    return {
        'access': lambda: compare_access_approaches(engine, acknowledge_unknowns=True),
        'cranial': lambda: compare_cranial_strategies(engine.scene, {}),
        'interactive': lambda: generate_interactive_approaches(engine, {}),
        'joint': lambda: compare_joint_strategies(engine, acknowledge_unknowns=True),
        'continuation': lambda: compare_continuations(engine, acknowledge_unknowns=True),
        'preparation': lambda: prepare(engine, {}, None),
        'preparation-suggest': lambda: suggest(engine, None),
        'sectors': lambda: compare_sector_strategies(engine, {}, {}, acknowledge_unknowns=True),
        'sector-macro': lambda: execute_strategy_option(engine, {}, 'unqualified'),
        'staged': lambda: compare_staged_strategies(engine, {}, acknowledge_unknowns=True),
        'rescue': lambda: compare_rescue(engine, {}, acknowledge_unknowns=True),
        'two-tool': lambda: generate_comparison(engine, {}),
        'full-director': lambda: new_director(engine),
        'full-slice': lambda: run_slice(engine, {}),
        'precision-reentry': lambda: continue_precision_reentry(engine, {}, acknowledge_unknowns=True),
        'recover-cusa': lambda: recover_checked_cusa(engine),
    }


class _ReachedNative(Exception):
    pass


def _outcome(call):
    """Comparable result of one public call with native work intercepted.

    A refusal compares by its code (text before the first colon), so a task
    contract that names a different missing structural field still matches.
    """
    try:
        result = call()
    except _ReachedNative:
        return ('native-work',)
    except Exception as error:
        return ('raised', type(error).__name__, str(error).split(':')[0])
    return ('returned', type(result).__name__)


def _forbid_native(monkeypatch):
    def reached(*args, **kwargs):
        raise _ReachedNative()
    monkeypatch.setattr(SurgicalEngine, 'preview', reached)
    monkeypatch.setattr(SurgicalEngine, 'step', reached)
    return reached


@pytest.mark.parametrize('kind', SOURCE_KINDS)
def test_missing_procedure_falls_back_to_labelled_legacy_rl_for_any_source(kind):
    from resectionlab.surgical_rl_env import RLConfig, SurgicalRLEnv
    env = SurgicalRLEnv(source(kind), config=RLConfig(require_critical_coverage=False))
    _, info = env.reset(seed=5)
    assert not env.procedure_enabled and not env.evaluation_only
    assert info['source_class'] == env.source_class == 'patient-or-unverified'
    assert info['source_role']['verified'] is False and info['target_basis'] == 'owned_target_labels'


@pytest.mark.parametrize('kind', SOURCE_KINDS)
@pytest.mark.parametrize('name', ('access', 'cranial', 'interactive', 'joint', 'continuation',
                                  'preparation', 'preparation-suggest', 'sectors', 'sector-macro',
                                  'staged', 'rescue', 'two-tool', 'full-director', 'full-slice',
                                  'precision-reentry', 'recover-cusa'))
def test_all_public_generators_treat_every_source_class_like_generated(monkeypatch, kind, name):
    from resectionlab.surgical_planning_admission import REQUIRED, planning_admission
    engine = SurgicalEngine(source(kind))
    generated = SurgicalEngine(source('direct-generated'))
    admission = planning_admission(engine.scene, engine.protection)
    assert admission['available'] and admission['sourceClass'] == 'patient-or-unverified'
    assert planning_admission(generated.scene)['sourceClass'] == 'generated-engineering'
    before = copy.deepcopy(engine.snapshot())
    _forbid_native(monkeypatch)
    actual, expected = _outcome(generators(engine)[name]), _outcome(generators(generated)[name])
    # Only the structural inputs decide; the class never changes the result.
    assert actual == expected and REQUIRED not in str(actual)
    assert engine.snapshot() == before


@pytest.mark.parametrize('kind', SOURCE_KINDS)
def test_inspection_and_historical_engine_replay_remain_available(kind):
    scene = source(kind)
    engine = SurgicalEngine(scene)
    assert engine.step({'slot': 'left', 'operation': 'inspect', 'duration': .1})['status'] == 'accepted'
    snapshot = engine.snapshot()
    assert SurgicalEngine.restore(scene, snapshot, use_cache=False).snapshot() == snapshot


@pytest.mark.parametrize('kind', ('direct-generated', 'generated-case'))
def test_explicit_generated_engineering_retains_legacy_rl(kind):
    from resectionlab.surgical_rl_env import RLConfig, SurgicalRLEnv
    env = SurgicalRLEnv(source(kind), config=RLConfig(require_critical_coverage=False))
    env.reset(seed=5)
    assert not env.procedure_enabled and not env.evaluation_only
    assert env.engine.scene.identity == env.scene.identity
    assert env.source_class == 'generated-engineering' and env.source_role['role'] == 'GENERATED_CONTROL'


@pytest.mark.parametrize('kind', ('observed-case', 'generated-metadata-spoof', 'mixed-source-case'))
def test_native_case_adapter_admits_acquired_case_with_acquired_label(monkeypatch, kind):
    from resectionlab.native_spatial_task import (ACQUIRED_LEGACY_ADMISSION_SCHEMA,
        generated_legacy_planning_evidence, native_spatial_task_from_case)
    from resectionlab import structural_evidence
    case = source(kind, as_case=True)
    evidence = generated_legacy_planning_evidence(case)
    # Generated metadata cannot relabel an observed reference as generated.
    assert evidence['schema'] == ACQUIRED_LEGACY_ADMISSION_SCHEMA
    assert evidence['sourceClass'] == 'patient-or-unverified' and evidence['sourceCaseHash'] == case.semantic_hash
    reached = []
    def support(*args, **kwargs):
        reached.append(True)
        return None, {}
    monkeypatch.setattr(structural_evidence, 'planning_brain_support', support)
    with pytest.raises(ValueError, match='ESSENTIAL_EVIDENCE_MISSING'):
        native_spatial_task_from_case(case, access=None, tools=())
    assert reached == [True]


@pytest.mark.parametrize('kind', SOURCE_KINDS)
def test_public_projection_treats_patient_provenance_like_generated(monkeypatch, kind):
    from resectionlab.surgical_planning_observation import project_observation
    engine = SurgicalEngine(source(kind)); before = engine.snapshot()
    generated = SurgicalEngine(source('direct-generated'))
    _forbid_native(monkeypatch)
    actual = _outcome(lambda: project_observation(engine, {}, []))
    assert actual == _outcome(lambda: project_observation(generated, {}, []))
    assert actual == ('raised', 'ValueError', 'PLANNING_OBSERVATION')
    assert engine.snapshot() == before


@pytest.mark.parametrize('track', ('annotation_assisted', 'inference_only'))
def test_direct_native_constructor_admits_an_unrecorded_source_as_unverified(track):
    # Owner decision 2026-11-27: a direct source without a CaseData record is
    # admitted; it cannot prove a generated class, so it is labelled unverified.
    from resectionlab import native_spatial_task
    from test_patient_planning_admission import fixture_contract
    generated, _ = fixture_contract()
    source = replace(generated, track=track)
    assert 'generatedLegacyPlanningEvidence' not in source.support_provenance
    task = native_spatial_task.NativeSpatialTask(source, max_steps=2)
    assert task.source_class == 'patient-or-unverified'
    assert task.observation().action_ids


def test_generated_case_native_adapter_retains_exact_generated_legacy_evidence():
    from resectionlab.native_spatial_task import (
        OPENING_TOOLS, generated_legacy_planning_evidence, make_native_opening_task,
        native_spatial_task_from_case,
    )
    from test_native_spatial_task import annotated_source
    original = annotated_source()
    generated = replace(original, source_refs=(SourceRef('generated', 'synthetic://native-boundary',
        provenance='simulated'),), metadata={**dict(original.metadata), 'generated': True, 'is_synthetic': True})
    task = native_spatial_task_from_case(generated, access=make_native_opening_task().case.access,
        tools=OPENING_TOOLS)
    evidence = task.case.support_provenance['generatedLegacyPlanningEvidence']
    assert dict(evidence) == generated_legacy_planning_evidence(generated)
    assert evidence['sourceCaseHash'] == generated.semantic_hash
    assert evidence['schema'] == 'resectionlab.generated-native-legacy-admission.v1'
    assert evidence['provenance'] == 'Explicit generated-source legacy geometry task; no patient qualification.'
    assert task.case.track == 'annotation_assisted' and task.observation().action_ids
    assert not task.supported_procedure_policy


def test_acquired_case_native_adapter_builds_the_same_task_with_an_acquired_label():
    from resectionlab.native_spatial_task import (
        OPENING_TOOLS, make_native_opening_task, native_spatial_task_from_case,
    )
    from test_native_spatial_task import annotated_source
    acquired = annotated_source()
    generated = replace(acquired, source_refs=(SourceRef('generated', 'synthetic://native-boundary',
        provenance='simulated'),), metadata={**dict(acquired.metadata), 'generated': True, 'is_synthetic': True})
    access = make_native_opening_task().case.access
    task = native_spatial_task_from_case(acquired, access=access, tools=OPENING_TOOLS)
    reference = native_spatial_task_from_case(generated, access=access, tools=OPENING_TOOLS)
    evidence = task.case.support_provenance['generatedLegacyPlanningEvidence']
    assert evidence['sourceClass'] == 'patient-or-unverified' and evidence['sourceCaseHash'] == acquired.semantic_hash
    # Action ids embed the source hash; the inventory itself is the same size.
    assert len(task.observation().action_ids) == len(reference.observation().action_ids) > 0


@pytest.mark.parametrize('track', ('annotation_assisted', 'inference_only', 'synthetic_scan'))
@pytest.mark.parametrize('marker', ('public-manifest', 'native-adapter', 'patient-key', 'acquired-class'))
def test_native_acquired_markers_win_over_generated_label_but_admit_acquired_evidence(track, marker):
    from resectionlab.core import array_digest
    from resectionlab.native_spatial_task import (ACQUIRED_LEGACY_ADMISSION_SCHEMA, NativeSpatialTask,
        generated_legacy_planning_evidence)
    from test_patient_planning_admission import fixture_contract
    native, _ = fixture_contract()
    evidence = generated_legacy_planning_evidence(source('generated-case', as_case=True))
    provenance = {
        'public-manifest': {'public_manifest_sha256': 'a' * 64},
        'native-adapter': {'adapter': {'type': 'native-patient-research-derivative', 'sourcePatient': 'routing-probe'}},
        'patient-key': {'patient': 'routing-probe'},
        'acquired-class': {'procedurePolicySourceClass': 'acquired-or-unknown'},
    }[marker]
    mislabelled = replace(native, track=track, support_provenance={
        **provenance, 'generatedLegacyPlanningEvidence': evidence})
    with pytest.raises(ValueError, match='NATIVE_LEGACY_SOURCE_EVIDENCE_INVALID: acquired source markers'):
        NativeSpatialTask(mislabelled, max_steps=2)
    acquired = {**evidence, 'schema': ACQUIRED_LEGACY_ADMISSION_SCHEMA, 'sourceClass': 'patient-or-unverified',
        'sourceImageHash': array_digest(native.structural_intensity),
        'sourceAffineHash': array_digest(native.affine_ras_mm),
        'provenance': 'Acquired source bound to this exact image and frame'}
    admitted = replace(native, track=track, support_provenance={**provenance, 'generatedLegacyPlanningEvidence': acquired})
    task = NativeSpatialTask(admitted, max_steps=2)
    assert task.source_class == 'patient-or-unverified'
    assert task.case.support_provenance['generatedLegacyPlanningEvidence']['sourceClass'] == 'patient-or-unverified'


def test_native_cannot_borrow_unrelated_generated_source_declaration():
    from resectionlab.native_spatial_task import NativeSpatialTask, generated_legacy_planning_evidence
    from test_patient_planning_admission import fixture_contract
    native, _ = fixture_contract()
    unrelated = generated_legacy_planning_evidence(source('generated-case', as_case=True))
    native = replace(native, track='annotation_assisted', support_provenance={
        'generatedLegacyPlanningEvidence': unrelated})
    with pytest.raises(ValueError, match='NATIVE_LEGACY_SOURCE_EVIDENCE_INVALID'):
        NativeSpatialTask(native, max_steps=2)


@pytest.mark.parametrize('kind', SOURCE_KINDS)
@pytest.mark.parametrize('search', ('sequence', 'joint'))
def test_direct_checked_searches_treat_every_source_class_like_generated(monkeypatch, kind, search):
    from resectionlab.surgical_sequence_search import checked_sequence_search
    from resectionlab.surgical_joint_cranial_search import checked_joint_sequence_search
    engine = SurgicalEngine(source(kind)); before = engine.snapshot()
    generated = SurgicalEngine(source('direct-generated'))
    reached = _forbid_native(monkeypatch)
    def run(target):
        if search == 'sequence':
            return lambda: checked_sequence_search(target, {}, propose=reached, motion=reached,
                hand='left', radius=1., limits=None, meter=None)
        return lambda: checked_joint_sequence_search(target, propose=reached, limits=None, goal=None)
    assert _outcome(run(engine)) == _outcome(run(generated))
    assert engine.snapshot() == before


@pytest.mark.parametrize('kind', SOURCE_KINDS)
@pytest.mark.parametrize('patient_position', (0, 1))
def test_forest_admits_all_sources_and_reaches_the_same_structural_check(monkeypatch, kind, patient_position):
    from resectionlab.surgical_joint_cranial_forest import OpeningComparisonBudget, search_opening_forest
    def forest(patient_kind):
        generated = SurgicalEngine(source('direct-generated'))
        patient = SurgicalEngine(source(patient_kind))
        engines = [generated]; engines.insert(patient_position, patient)
        before = [e.snapshot() for e in engines]
        budget = OpeningComparisonBudget(10, 10., export_reserve=0)
        result = _outcome(lambda: search_opening_forest([{'engine': e} for e in engines], budget=budget))
        assert [e.snapshot() for e in engines] == before and budget.calls == 0
        return result
    _forbid_native(monkeypatch)
    assert forest(kind) == forest('direct-generated')


@pytest.mark.parametrize('mutation', ('erase', 'replace'))
def test_case_provenance_proof_cannot_be_erased_or_replaced_to_relabel_patient(mutation):
    from resectionlab.surgical_planning_admission import planning_admission
    scene = source('generated-metadata-spoof')
    if mutation == 'erase':
        del scene._planning_source_evidence
        del scene._planning_source_proof
        # Admitted, and erasing the proof cannot promote the label.
        result = planning_admission(scene)
        assert result['available'] and result['sourceClass'] == 'patient-or-unverified'
    else:
        old = scene._planning_source_evidence
        scene._planning_source_evidence = ('generated-engineering', *old[1:])
        with pytest.raises(ValueError, match='PROCEDURE_SOURCE_PROVENANCE_CHANGED'):
            planning_admission(scene)
