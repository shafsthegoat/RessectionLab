"""Source-only controls. Task construction, preview, and actions are forbidden."""
from collections import Counter
import json
import sys

import numpy as np
import pytest

from resectionlab import public_contact_family as family
from resectionlab.core import semantic_digest
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialTask


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Structural controls must not execute or preview a task')
    for owner, names in ((NativeSpatialTask, ('__init__', 'step', 'planning_step')),
            (NativeResectionEngine, ('__init__', 'preview_stroke', 'execute_stroke'))):
        for name in names:
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, forbidden)
    yield
    assert 'torch' not in sys.modules
    assert 'nibabel' not in sys.modules


@pytest.fixture(scope='module')
def bindings():
    # The autouse function fixture does not cover module setup; install guards
    # explicitly here too, so manifest construction cannot quietly do previews.
    with pytest.MonkeyPatch.context() as patch:
        def forbidden(*args, **kwargs):
            raise AssertionError('Manifest may construct sources, never engines')
        patch.setattr(NativeSpatialTask, '__init__', forbidden)
        patch.setattr(NativeResectionEngine, '__init__', forbidden)
        patch.setattr(NativeResectionEngine, 'preview_stroke', forbidden)
        return family.family_manifest()


def test_split_is_layout_bound_and_exact(bindings):
    rows = bindings['source_bindings']
    assert len(rows) == 24
    assert Counter(r['role'] for r in rows) == {'TRAIN': 12, 'SELECT': 4, 'MEASUREMENT_EVAL': 8}
    assert len({r['layout_id'] for r in rows}) == 24
    assert bindings['family_hash'] == semantic_digest(family.family_record())
    assert bindings['split_unit'] == 'whole_layout_including_both_goals_and_all_state_descendants'
    assert not bindings['training_performed'] and bindings['held_out_task_executions'] == 0
    for role in family.ROLES:
        depths = {r['recipe']['front_depths'][2] for r in bindings['layouts'] if r['role'] == role}
        assert depths == {2, 3, 4}


def test_geometry_and_source_are_not_transformed_copies(bindings):
    rows = bindings['source_bindings']
    for key in ('recipe_hash', 'source_hash', 'support_hash', 'topology_hash'):
        assert len({r[key] for r in rows}) == 24, key
    # Every column is one of the authoritative fixed lattice's three rays.
    assert len({tuple(r['recipe']['front_depths'][1:4]) for r in bindings['layouts']}) == 24
    goals = [g['objective_hash'] for row in rows for g in row['goals'].values()]
    assert len(set(goals)) == 48
    assert all(row['decision_model_hash'] is None for row in rows)
    assert all(row['actions_executed'] == row['geometry_previews'] == 0 for row in rows)


@pytest.mark.parametrize('layout_id', [f'pcf-{i:02d}' for i in range(24)])
def test_generated_source_and_goal_contract(layout_id):
    source = family.build_family_source(layout_id)
    metadata = family.layout_metadata(layout_id)
    assert source.track == 'synthetic_scan'
    assert np.array_equal(source.structural_intensity > 0, source.observed_support)
    assert not np.any(source.reference_target) and not np.any(source.nominal_target)
    assert not np.any(source._native_config.target_labels)
    assert source._candidate_scope == 'fixed_access_grid_3columns_8depths_within_actor_crop'
    assert len(source._candidate_voxels) == 15
    assert len(source.tools) == 2
    assert source._crop_origin == (0, 0, 0) and source._crop_shape == family.SHAPE
    for goal in metadata['goals'].values():
        assert source.observed_support[goal['native_index']]
    # Actor source fields disclose geometry, not split/recipe labels or hidden labels.
    assert source.access.window_id == 'generated-family-access'
    actor_metadata = [source.support_derivation, source.target_derivation, source.support_provenance]
    assert layout_id not in str(actor_metadata)
    assert not any(role in str(actor_metadata) for role in family.ROLES)
    with pytest.raises(ValueError):
        source.observed_support.setflags(write=True)
    assert source.source_hash == family.build_family_source(layout_id).source_hash


def test_topology_check_removes_translation_rotation_and_reflection():
    source = family.build_family_source('pcf-00').observed_support
    rotated = np.flip(source.transpose(2, 0, 1), axis=1)
    shifted = np.pad(rotated, ((2, 3), (1, 4), (3, 2)))
    assert family.topology_digest(source) == family.topology_digest(shifted)


def test_metadata_mutation_does_not_relabel_canonical_roles():
    row = family.layout_metadata('pcf-00')
    expected = row['role']
    digest = family.family_digest()
    row['role'] = 'UNRESTRICTED'
    row['goals']['surface']['native_index'] = (0, 0, 0)
    record = family.family_record()
    record['layouts'][0]['role'] = 'UNRESTRICTED'
    assert family.layout_metadata('pcf-00')['role'] == expected
    assert family.family_digest() == digest


def test_all_held_out_tasks_refuse_before_source_build(monkeypatch, bindings):
    def source_forbidden(*args, **kwargs):
        raise AssertionError('Held-out refusal must precede even source construction')
    monkeypatch.setattr(family, 'build_family_source', source_forbidden)
    for row in bindings['source_bindings']:
        if row['role'] == 'MEASUREMENT_EVAL':
            for goal in family.GOAL_IDS:
                with pytest.raises(ValueError, match='HELD_OUT_EXECUTION_CLOSED'):
                    family.make_family_task(row['layout_id'], goal)
                with pytest.raises(ValueError, match='HELD_OUT_EXECUTION_CLOSED'):
                    family.bind_family_context(object(), layout_id=row['layout_id'],
                        goal_id=goal, experiment_hash='sha256:'+'a'*64)
            with pytest.raises(TypeError):
                family.make_family_task(row['layout_id'], 'surface', role='TRAIN')


@pytest.mark.parametrize('layout_id', ('pcf-24', 'TRAIN', '', 0, True))
def test_unknown_identity_refused(layout_id):
    with pytest.raises(ValueError, match='Unknown fixed'):
        family.build_family_source(layout_id)


def test_invalid_goal_refused_without_task_construction(bindings):
    selected = next(r['layout_id'] for r in bindings['source_bindings'] if r['role'] == 'SELECT')
    with pytest.raises(ValueError, match='prospectively fixed'):
        family.make_family_task(selected, 'best_after_search')
    assert not family.layout_metadata(selected)['training_admission']


def test_manifest_round_trip_is_stable_and_no_executable_claim(bindings):
    encoded = json.loads(json.dumps(bindings))
    assert semantic_digest(encoded) == semantic_digest(bindings)
    assert not encoded['held_out_execution_admitted']
    assert not encoded['training_admission']
    assert all(r['execution_binding_status'] == 'not_materialized_source_only'
        for r in encoded['source_bindings'])
