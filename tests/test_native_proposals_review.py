"""Independent proposal-binding regressions; no native geometry rollouts."""
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.geometry import AccessWindow, _immutable
from resectionlab.native_proposals import PreparedAxisColumnProposer
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig, NativeResectionEngine


def fixture():
    tissue = np.ones((9, 9, 10), bool)
    labels = np.zeros(tissue.shape, np.int16)
    labels[:, :, 3:9] = 1
    config = NativeResectionConfig(
        tissue, labels, np.eye(4), AccessWindow((4, 4, -.5), (0, 0, 1), 6),
        tuple(replace(tool) for tool in NATIVE_GENERIC_TOOLS), "same-declared-source", "synthetic review support",
    )
    return NativeResectionEngine(config), tissue, labels


def test_external_input_arrays_cannot_change_prepared_source():
    engine, original_tissue, original_labels = fixture()
    provider = PreparedAxisColumnProposer(engine.config)
    before = provider.propose(engine)
    original_tissue[:] = False
    original_labels[:] = 0
    assert provider.propose(engine) == before
    with pytest.raises(ValueError):
        engine.config.target_labels.setflags(write=True)


@pytest.mark.parametrize("field", ["target_labels", "tissue_mask", "hard_exclusion", "affine"])
def test_source_content_changed_before_preparation_cannot_reuse_cached_identity(field):
    engine, _, _ = fixture()
    changed = getattr(engine.config, field).copy()
    if field == "target_labels":
        changed[:, :, 8] = 0
    elif field == "tissue_mask":
        changed[0, 0, 0] = False
    elif field == "hard_exclusion":
        changed[0, 0, 0] = True
    else:
        changed[0, 3] = 1.
    object.__setattr__(engine.config, field, _immutable(changed))
    with pytest.raises(ValueError):
        PreparedAxisColumnProposer(engine.config)


@pytest.mark.parametrize("field", ["target_labels", "tissue_mask"])
def test_legitimate_same_shape_source_variants_get_distinct_model_identity(field):
    engine, _, _ = fixture()
    original = PreparedAxisColumnProposer(engine.config)
    changed = getattr(engine.config, field).copy()
    if field == "target_labels":
        changed[:, :, 8] = 0
    else:
        changed[0, 0, 0] = False
    replacement = replace(engine.config, **{field: changed})
    different_engine = NativeResectionEngine(replacement)
    alternative = PreparedAxisColumnProposer(replacement)
    assert alternative.model_hash != original.model_hash
    assert alternative.propose(different_engine).source_hash == original.propose(engine).source_hash
    with pytest.raises(ValueError):
        original.propose(different_engine)


@pytest.mark.parametrize("field", ["remaining_mask", "removed_mask", "contact_mask", "connected_free_mask"])
def test_external_cavity_array_mutation_rejected_under_cached_ancestry(field):
    engine, _, _ = fixture()
    provider = PreparedAxisColumnProposer(engine.config)
    before = provider.propose(engine)
    array = getattr(engine, field)
    array[4, 4, 8] = ~array[4, 4, 8]
    assert engine.state_hash == before.cavity_state_hash  # Cached ancestry alone cannot detect this edit.
    with pytest.raises(RuntimeError):
        provider.propose(engine)


def test_source_array_descriptor_is_immutable_after_preparation():
    engine, _, _ = fixture()
    provider = PreparedAxisColumnProposer(engine.config)
    before = provider.propose(engine)
    with pytest.raises(ValueError):
        engine.config.target_labels.shape = (3, 27, 10)
    assert provider.propose(engine) == before


def test_consistent_external_mask_edits_cannot_fabricate_committed_cavity():
    engine, _, _ = fixture()
    provider = PreparedAxisColumnProposer(engine.config)
    key = (4, 4, 0)
    engine.remaining_mask[key] = False
    engine.removed_mask[key] = True
    engine.contact_mask[key] = True
    engine.connected_free_mask[key] = True
    assert engine.revision == 0 and engine.history == []
    with pytest.raises(RuntimeError):
        provider.propose(engine)


def test_batch_validation_accepts_current_clone_and_rejects_modified_ray_or_ledger():
    engine, _, _ = fixture()
    provider = PreparedAxisColumnProposer(engine.config)
    batch = provider.propose(engine)
    assert provider.validate_batch(batch, engine.clone()) is None
    changed_ray = replace(batch.proposals[0], primary_target_mm=(4., 4., 7.))
    altered = replace(batch, proposals=(changed_ray,) + batch.proposals[1:])
    with pytest.raises(ValueError):
        provider.validate_batch(altered, engine)
    changed_slot = replace(batch.ledger[0], reason="CERTIFIED")
    altered = replace(batch, ledger=(changed_slot,) + batch.ledger[1:])
    with pytest.raises(ValueError):
        provider.validate_batch(altered, engine)


def test_batch_is_stale_after_one_short_paid_analytic_stroke():
    engine, _, _ = fixture()
    provider = PreparedAxisColumnProposer(engine.config)
    before = provider.propose(engine)
    cut = engine.execute_stroke("native-fine-aspiration", (4., 4., 3.), entry_mm=(4., 4., -.5))
    assert cut.feasible and len(cut.removed_indices_native) > 0
    after = provider.propose(engine)
    assert after.cavity_state_hash != before.cavity_state_hash
    with pytest.raises(ValueError):
        provider.validate_batch(before, engine)
    assert provider.validate_batch(after, engine) is None
