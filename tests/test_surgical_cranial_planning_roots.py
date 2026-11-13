"""Opening-root persistence, without patient qualification or treatment claims."""
import copy
import json

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef, semantic_digest
from resectionlab.surgical_cranial_planning_roots import (
    FIELD, bind_branch_root, candidate_scene, create_registry, same_source_anatomy, scene_for_branch, source_binding,
    validate_registry)
from resectionlab.surgical_environment import (
    SCHEMA as ENVIRONMENT_SCHEMA, load_environment, resolve_branch_scene, standardize)
from resectionlab.surgical_scene import from_case, original_target_memberships


def reseal(document, name):
    document[name] = semantic_digest({k: v for k, v in document.items() if k != name})


@pytest.fixture(scope='module')
def roots():
    # Reuse the existing generated bone/membrane geometry. No opening is run;
    # native zero-action roots verify their complete stage/guard declarations.
    from test_surgical_membrane_transport import joint_source
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_craniotomy import binding, contract_hash
    from resectionlab.surgical_cranial_stages import freeze_cranial_stage_contract
    template, stage, _, _, _ = joint_source()
    props = copy.deepcopy(template.scene.properties)
    opening = props.pop('cranialAccessV1')
    masks = {r['name']: template.scene.labels == r['id'] for r in template.scene.regions}
    case = CaseData(case_id='generated-immutable-opening-roots',
        mri=np.zeros(template.scene.shape, np.float32), compartments=masks,
        affine=template.scene.affine,
        source_refs=(SourceRef('generated', 'synthetic://immutable-opening-roots', provenance='simulated'),),
        metadata={'generated': True, 'is_synthetic': True,
            'surgical_properties': props,
            'anatomy_layers': {r['name']: {'kind': r['material']} for r in template.scene.regions}})
    base = standardize(from_case(case, template.scene.regions))
    values = []
    for index in range(2):
        raw = copy.deepcopy(opening)
        if index:
            # A different immutable A/R declaration, with all source bone and
            # the original target still present. It is not a feasibility claim.
            cell = raw['kerfCells'].pop(0)
            raw['windowCells'].remove(cell)
            raw['retainedSkullCells'] = sorted([*raw['retainedSkullCells'], cell])
        raw.update(binding(base))
        raw['contentHash'] = contract_hash(raw)
        scene = candidate_scene(base, 'cranialAccessV1', raw)
        source = SurgicalEngine(scene)
        bound = freeze_cranial_stage_contract(source,
            membrane_cells=stage.membrane_cells,
            retained_membrane_cells=stage.retained_membrane_cells,
            membrane_anchor_cells=stage.membrane_anchor_cells,
            work_cells=stage.work_cells, evidence=stage.as_dict()['evidence'])
        source = SurgicalEngine(scene, cranial_stages=bound)
        values.append({'id': f'opening-{index}', 'label': f'Generated opening {index}',
            'openingProperty': 'cranialAccessV1', 'openingContract': raw,
            'sourceSnapshot': source.snapshot()})
    registry = create_registry(base, values, {'first': 'opening-0', 'second': 'opening-1'})
    record = {'environmentSchema': ENVIRONMENT_SCHEMA,
        'mapping': base.regions, 'physicsProfile': base.properties['tissuePhysicsV2'],
        'branches': {'first': values[0]['sourceSnapshot'], 'second': values[1]['sourceSnapshot']},
        'branchId': 'second', 'position': 0, FIELD: registry}
    return case, base, values, record


def test_distinct_opening_roots_preserve_full_anatomy_and_original_target(roots):
    _, base, values, record = roots
    before = source_binding(base)
    scenes = validate_registry(base, record[FIELD])
    assert len({scene.identity for scene in scenes.values()}) == 2
    target = original_target_memberships(base)
    for scene in scenes.values():
        assert np.array_equal(scene.labels, base.labels)
        assert np.array_equal(scene.affine, base.affine)
        assert scene.regions == base.regions and scene.source == base.source
        assert all(np.array_equal(ids, original_target_memberships(scene)[name])
                   for name, ids in target.items())
        assert same_source_anatomy(base, scene)
        assert {k: v for k, v in scene.properties.items() if k != 'cranialAccessV1'} == base.properties
    assert source_binding(base) == before
    assert record[FIELD]['admittedPatientTask'] is False
    assert values[0]['openingContract']['windowCells'] != values[1]['openingContract']['windowCells']


def test_json_roundtrip_resolves_each_exact_native_source_and_restores(roots):
    from resectionlab.surgical_engine import SurgicalEngine
    _, base, _, record = roots
    loaded = json.loads(json.dumps(record))
    for branch, snapshot in loaded['branches'].items():
        scene = resolve_branch_scene(base, loaded, branch)
        assert scene.identity == snapshot['sceneId']
        assert SurgicalEngine.restore(scene, snapshot, use_cache=False).snapshot() == snapshot
    assert resolve_branch_scene(base, loaded).identity == loaded['branches']['second']['sceneId']


def test_private_frozen_workspace_resolves_without_thawing_action_histories(roots):
    from resectionlab.core import freeze_json
    _, base, _, record = roots
    frozen = freeze_json(record)
    for branch, snapshot in frozen['branches'].items():
        assert resolve_branch_scene(base, frozen, branch).identity == snapshot['sceneId']
    assert frozen['branches']['first']['actions'] == ()


def test_saved_case_cold_load_selects_candidate_scene_without_changing_source(roots, tmp_path):
    from resectionlab.imaging import save_case, load_case
    from resectionlab.surgical_engine import SurgicalEngine
    case, _, _, record = roots
    path = tmp_path/'generated-two-openings.ressectionlab'
    save_case(case, path, artifacts={'surgical_toolbox': record})
    assert load_case(path).semantic_hash == case.semantic_hash
    for branch, snapshot in record['branches'].items():
        scene = load_environment(path, branch_id=branch)
        assert scene.identity == snapshot['sceneId']
        assert SurgicalEngine.restore(scene, snapshot, use_cache=False).snapshot() == snapshot


def test_resealed_source_or_branch_substitution_refuses(roots):
    _, base, _, original = roots
    bad = copy.deepcopy(original)
    bad[FIELD]['sourceBinding']['originalTargetHash'] = 'substituted target'
    reseal(bad[FIELD], 'registrySeal')
    with pytest.raises(ValueError, match='SOURCE_IDENTITY_CHANGED'):
        scene_for_branch(base, bad, 'first')
    bad = copy.deepcopy(original)
    bad[FIELD]['branchRoots']['first'] = 'opening-1'
    reseal(bad[FIELD], 'registrySeal')
    with pytest.raises(ValueError, match='BRANCH_SCENE_CHANGED'):
        scene_for_branch(base, bad, 'first')


def test_branch_stage_cannot_switch_to_another_root_or_weaken_protection(roots):
    _, base, _, original = roots
    bad = copy.deepcopy(original)
    bad['branches']['first']['cranialStagesContract'] = bad['branches']['second']['cranialStagesContract']
    with pytest.raises(ValueError, match='BRANCH_STAGE_OR_PROTECTION_CHANGED'):
        scene_for_branch(base, bad, 'first')
    bad = copy.deepcopy(original)
    bad['branches']['first']['protectionContract'] = {'config': {'require_critical_coverage': False}}
    with pytest.raises(ValueError, match='BRANCH_STAGE_OR_PROTECTION_CHANGED'):
        scene_for_branch(base, bad, 'first')


def test_empty_actions_and_resealed_metadata_do_not_authenticate_unlogged_state(roots):
    _, base, values, _ = roots
    bad = copy.deepcopy(values)
    bad[0]['sourceSnapshot']['checkpoint']['state']['elapsedSeconds'] = 1.
    with pytest.raises(ValueError, match='[Cc]heckpoint|CANONICAL_ROOT'):
        create_registry(base, bad)


def test_new_child_binding_is_atomic_and_cannot_change_an_accepted_root(roots):
    _, base, _, original = roots
    record = copy.deepcopy(original)
    record['branches']['child'] = copy.deepcopy(record['branches']['first'])
    registry = bind_branch_root(base, record, 'child', 'opening-0')
    assert registry['branchRoots']['child'] == 'opening-0'
    assert scene_for_branch(base, record, 'child').identity == record['branches']['first']['sceneId']
    before = copy.deepcopy(record[FIELD])
    with pytest.raises(ValueError, match='ACCEPTED_BRANCH_ROOT_CHANGED'):
        bind_branch_root(base, record, 'child', 'opening-1')
    assert record[FIELD] == before
    record['branches']['invalid'] = copy.deepcopy(record['branches']['second'])
    with pytest.raises(ValueError, match='BRANCH_SCENE_CHANGED'):
        bind_branch_root(base, record, 'invalid', 'opening-0')
    assert record[FIELD] == before


def test_registered_planning_metadata_is_bounded_immutable_and_not_a_default_policy(roots):
    _, base, values, _ = roots
    declarations = copy.deepcopy(values)
    planning = {'limits': {'maxBoneMm3': 285., 'maxPreparationNonboneMm3': 49.},
        'openingSkill': {'kind': 'opening', 'actions': []}, 'preparationActions': [],
        'workRoutes': [], 'outsidePoses': {},
        'qualification': {'status': 'unknown', 'blockers': ['ENGINEERING_CONTROL_ONLY']}}
    declarations[0]['planning'] = planning
    registry = create_registry(base, declarations)
    assert registry['roots'][0]['planning'] == planning
    planning['limits']['maxBoneMm3'] = 1.
    assert registry['roots'][0]['planning']['limits']['maxBoneMm3'] == 285.
    assert 'planning' not in registry['roots'][1]
    oversized = copy.deepcopy(declarations)
    oversized[0]['planning']['qualification']['detail'] = 'x'*(256*1024)
    with pytest.raises(ValueError, match='PLANNING_METADATA_TOO_LARGE'):
        create_registry(base, oversized)


def test_legacy_single_scene_remains_exact_and_incompatible_snapshot_is_refused(roots):
    from resectionlab.surgical_engine import SurgicalEngine
    _, base, values, _ = roots
    original = SurgicalEngine(base).snapshot()
    legacy = {'branches': {'main': original}, 'branchId': 'main'}
    assert resolve_branch_scene(base, legacy) is base
    legacy['branches']['other'] = values[0]['sourceSnapshot']
    with pytest.raises(ValueError, match='UNREGISTERED_SCENE'):
        resolve_branch_scene(base, legacy)


def test_missing_original_target_duplicate_and_unknown_roots_refuse(roots):
    from resectionlab.surgical_scene import SurgicalScene
    _, base, values, original = roots
    bare = SurgicalScene(base.labels, base.affine, base.regions, base.source, base.properties)
    with pytest.raises(ValueError, match='ORIGINAL_TARGET_EVIDENCE_REQUIRED'):
        create_registry(bare, values)
    with pytest.raises(ValueError, match='DUPLICATE_ROOT_ID'):
        create_registry(base, [values[0], values[0]])
    bad = copy.deepcopy(original)
    bad[FIELD]['branchRoots']['first'] = 'missing'
    reseal(bad[FIELD], 'registrySeal')
    with pytest.raises(ValueError, match='INVALID_BRANCH_ROOT_BINDING'):
        resolve_branch_scene(base, bad)
