"""Independent generated adapter controls; root executes, no acquired inputs.

The author fixture performs a real native prefix. These controls add checks of
the union actor inputs and a counterfactual evaluator-reference permutation,
without minting a prefix/context or changing production admission.
"""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / 'model-recovery'
sys.path.insert(0, str(MODEL))
spec = importlib.util.spec_from_file_location('_independent_focused_fixture', MODEL/'test_focused_policy.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
adapter = fixture.adapter
spec = importlib.util.spec_from_file_location('_independent_focused_scalar', HERE.parent/'independent/scalar_oracles.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)

from resectionlab.core import semantic_digest
from resectionlab.focused_paged_task import FocusedPagedObstructionTask
from resectionlab.spatial_continuation_value import _feature, _costs


def test_real_page_union_preserves_every_actor_row_and_exact_owner():
    f = fixture.generated_fixture()
    catalog = adapter.collect_catalog(f['focused'].clone(), f['admission'])
    assert len(catalog.pages) > 1 and any(i > 0 for i in catalog.owners)
    public_pages = []
    for page in catalog.pages:
        public_pages.append([{'action_id': key, **{name: record[name] for name in
            ('tool_id', 'entry_mm', 'tip_mm')}} for key, preview in page._inventory.items()
            for record in (preview.to_history_record(),)])
    rows, owners = oracle.union_rows(public_pages)
    assert catalog.identifiers == ('STOP', *(row['action_id'] for row in rows))
    assert catalog.owners == (0, *(owners[key] for key in catalog.identifiers[1:]))
    paid = _costs()
    union = _feature(f['policy'], catalog.observation, paid)
    for page_index, page in enumerate(catalog.pages):
        observation = page.observation()
        separate = _feature(f['policy'], observation, paid)
        # Same public image, coverage, state and physical ray must give the
        # exact actor-input row; other candidate rows may only affect critic.
        for j, identifier in enumerate(observation.action_ids):
            i = catalog.identifiers.index(identifier)
            np.testing.assert_array_equal(union[i], separate[j])
            if catalog.owners[i] == page_index:
                assert semantic_digest(page.export_action(identifier)) == semantic_digest(catalog.packets[i])
    assert paid['policy_forwards_completed'] == 1 + len(catalog.pages)
    assert catalog.require() is catalog
    assert f['focused']._steps == 1 and len(f['focused'].metrics()['history']) == 1


def test_hidden_evaluator_reference_is_removed_before_new_context_and_scoring():
    f = fixture.generated_fixture()
    # Build a genuinely different evaluator case normally, then use canonical
    # planning_clone to remove its private field before executing the prefix.
    private_case = replace(f['template'].case,
        reference_target=np.zeros_like(f['template'].case.nominal_target))
    private = FocusedPagedObstructionTask(private_case, max_steps=24,
        reward=f['template'].reward_spec)
    assert not np.array_equal(private.case.reference_target, private.case.nominal_target)
    nominal = private.planning_clone()
    for packet in f['packets']:
        nominal.advance_record(packet)
    frontier = adapter.protocol.frontier_record(nominal, f['packets'])
    admitted = adapter.admit_focused_context(f['original'], f['context'], nominal,
        f['packets'], frontier, expected_prefix_actions=1)
    left = adapter.collect_catalog(f['focused'].clone(), f['admission'])
    right = adapter.collect_catalog(nominal, admitted)
    assert left.observation.fingerprint == right.observation.fingerprint
    assert left.identifiers == right.identifiers and left.owners == right.owners
    assert semantic_digest(left.packets) == semantic_digest(right.packets)
    assert semantic_digest(left.scores) == semantic_digest(right.scores)
    assert semantic_digest(left.record) == semantic_digest(right.record)
    tail = fixture.generated_tail(f, constant=2.)
    original_worlds = semantic_digest(tail._worlds)
    # The legacy API must remain closed for the new source; no ID/world rewrite.
    with pytest.raises(ValueError):
        tail.score_tails(left.observation, f['context'])
    assert tail.costs['policy_forwards_started'] == 0
    a = adapter.FrozenFocusedTail(tail, f['admission']).score_catalog(left)
    b = adapter.FrozenFocusedTail(tail, admitted).score_catalog(right)
    assert a == b and a[0] == 0.
    assert semantic_digest(tail._worlds) == original_worlds


def test_actual_adapter_choice_agrees_with_independent_scalar_oracle():
    identifiers = ('STOP', 'z-first', 'a-second')
    cases = (((0., 2., 5.), None), ((0., 2., 5.), (0., 4., -3.)),
        ((0., 2., 5.), (0., 3., 0.)), ((0., 2., 5.), (0., -9., -3.)),
        ((0., -1., 0.), (0., 100., 100.)))
    for rewards, tails in cases:
        rows = [{'action_id': key, 'reward': reward} for key, reward in zip(identifiers, rewards)]
        index, values = adapter.choose(rows, tails)
        assert identifiers[index] == oracle.choose_positive(identifiers, rewards, tails)
        assert values == tuple(r if tails is None else r + max(0., t)
            for r, t in zip(rewards, tails or (0.,) * len(rewards)))
