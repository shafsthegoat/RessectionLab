"""Portable fixture and frozen comparison boundaries, without patient assets."""
import json
from pathlib import Path
import runpy

import numpy as np

from resectionlab.core import array_digest
from resectionlab.generated_cpp_fixture import cranial_generate_fixture
from resectionlab.surgical_cranial_planning import cranial_planning_availability


ROOT = Path(__file__).resolve().parents[1]


def test_generated_source_is_pristine_displayable_and_bound_to_frozen_programs():
    manifest = json.loads((ROOT/'manifests/experiments/cpp-acceleration-v1.json').read_text())
    expected = manifest['workloads']['generate']
    case, base, roots, record = cranial_generate_fixture()
    assert np.isfinite(case.mri).all() and np.ptp(case.mri) > 0
    assert case.metadata['generated'] and case.metadata['is_synthetic']
    assert base.identity == expected['sourceSceneId']
    assert array_digest(base.labels) == expected['labelsHash']
    assert array_digest(case.mri) == expected['displayHash']
    assert case.semantic_hash == expected['caseSemanticHash']
    assert 'tissuePhysicsV2' not in base.properties
    assert 'environmentSchema' not in record and 'physicsProfile' not in record
    assert record['position'] == 0 and record['branchId'] == 'source'
    assert record['branches']['source']['actions'] == []
    assert roots[0]['sourceSnapshot']['actions'] == []
    assert cranial_planning_availability(base, record)['available']
    for key in ('openingSkill', 'preparationActions', 'workRoutes', 'limits', 'outsidePoses'):
        assert roots[0]['planning'][key] == expected[key]


def test_standard_physics_is_a_separate_explicit_negative_source():
    _, legacy, _, _ = cranial_generate_fixture()
    _, standard, roots, record = cranial_generate_fixture(physics='standardized')
    assert 'tissuePhysicsV2' in standard.properties
    assert record['environmentSchema'] and record['physicsProfile'] == standard.properties['tissuePhysicsV2']
    assert legacy.identity != standard.identity
    assert np.array_equal(legacy.labels, standard.labels)
    assert cranial_planning_availability(standard, record)['available']
    applies = [a for a in roots[0]['planning']['openingSkill']['actions'] if a['operation'] == 'apply']
    assert applies and all(a['duration'] == 15. for a in applies)


def test_saved_generated_legacy_source_cold_load_preserves_mapping_physics_and_root(tmp_path):
    from resectionlab.imaging import load_case, save_case
    from resectionlab.surgical_environment import load_environment
    case, base, roots, record = cranial_generate_fixture()
    path = tmp_path/'generated-pristine.ressectionlab'
    save_case(case, path, artifacts={'surgical_toolbox': record})
    assert load_case(path).semantic_hash == case.semantic_hash
    cold = load_environment(path)
    assert cold.identity == base.identity and cold.regions == base.regions
    assert 'tissuePhysicsV2' not in cold.properties
    record['branchId'] = 'root'
    record['branches']['root'] = roots[0]['sourceSnapshot']
    from resectionlab.surgical_cranial_planning_roots import bind_branch_root
    bind_branch_root(base, record, 'root', roots[0]['id'])
    save_case(case, path, artifacts={'surgical_toolbox': record})
    assert load_environment(path).identity == roots[0]['sourceSnapshot']['sceneId']


def test_only_enumerated_wall_and_derived_report_paths_are_excluded():
    runner = runpy.run_path(str(ROOT/'scripts/benchmark_cpp_acceleration.py'))
    project = runner['canonical_result']
    value = {'report': {'comparisonSeal': 'derived', 'cost': {'wallSeconds': 1., 'physicalCalls': 3},
        'alternatives': [{'ledger': {'wallSecondsSinceMeterCreation': 2., 'modeledElapsedSeconds': 4.},
                          'goalAssessment': {'goalHash': 'process-bound', 'completedFrozenTask': True},
                          'metadata': {'wallSeconds': 9.}}]},
        'snapshots': [{'comparisonSeal': 'physical', 'wallSeconds': 5.}]}
    expected = {'report': {'cost': {'physicalCalls': 3}, 'alternatives': [
        {'ledger': {'modeledElapsedSeconds': 4.},
         'goalAssessment': {'completedFrozenTask': True}, 'metadata': {'wallSeconds': 9.}}]},
        'snapshots': [{'comparisonSeal': 'physical', 'wallSeconds': 5.}]}
    assert project(value) == expected
    assert value['report']['comparisonSeal'] == 'derived'
