"""Tiny generated tensor/forward controls; no native transition or model load."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from resectionlab.core import array_digest, semantic_digest
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.goal_mode_spatial_policy import (CHECKPOINT_VERSION, POLICY_VERSION,
    GoalModeSpatialPolicy, _mode_candidate_critic_context)
from resectionlab.public_surface_contact import (PublicSurfaceGoal, SurfaceContactObservation,
    SurfaceContactDevelopmentContext)
from resectionlab.sequential_spatial_observation import SequentialSpatialObservation
from resectionlab.spatial_observations import (ObservedChannel, ObservedProcedureState,
    SpatialAction, SpatialInputs, build_spatial_observation)
from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig, parameter_hash


def observation(*, goal=(2, 2, 3), exhausted=False, modes=("stop", "aspirate", "probe"),
                channel_change=None, contact=False):
    shape = (5, 5, 5)
    x, y, z = np.indices(shape)
    scan = (x + 2*y + 3*z).astype(np.float32) / 30.
    support = np.ones(shape, np.float32)
    cavity = np.zeros(shape, np.float32)
    if channel_change == "scan": scan[1:4, 1:4, 1:4] += 1.
    if channel_change == "support": support[1:4, 1:4, 1:4] = 0.
    if channel_change == "cavity": cavity[1:4, 1:4, 1:3] = 1.
    coverage = np.ones(shape, bool); coverage[0, 0, 0] = False
    source = semantic_digest({"fixture": "tiny_goal_mode_forward_only", "scan": array_digest(scan),
                              "support": array_digest(support)})
    inputs = SpatialInputs({
        "structural_intensity": ObservedChannel(scan, coverage, "synthetic_scan", "analytic ramp"),
        "nominal_tissue": ObservedChannel(support, source_kind="derived_from_scan",
            derivation="generated support feature for tensor controls", derived_from=("structural_intensity",)),
        "observed_cavity": ObservedChannel(cavity, source_kind="observed_procedure_state")},
        np.eye(4), "synthetic_scan", source)
    tools = (ToolGeometry("tool-a", .4, .2, 10., tip_length_mm=.5),
             ToolGeometry("tool-b", .4, .2, 10., tip_length_mm=.5))
    actions = (SpatialAction("STOP"), SpatialAction("candidate-a", (2., 2., 0.), (1., 2., 3.), tools[0]),
               SpatialAction("candidate-b", (2., 2., 0.), (3., 2., 3.), tools[1]))
    state = ObservedProcedureState(AccessWindow((2., 2., 0.), (0., 0., 1.), 3.), 2 if exhausted else 0, 2)
    base = build_spatial_observation(inputs, actions, state)
    contacts = np.zeros(shape, bool)
    if contact: contacts[2, 2, 3] = True
    sequential = SequentialSpatialObservation(base, modes, contacts)
    grid = np.zeros(shape, bool); grid[goal] = True
    return SurfaceContactObservation(sequential, grid, PublicSurfaceGoal(source, goal), (0, 0, 0),
                                     semantic_digest({"fixed_public_task": 1, "source": source}))


def context(obs):
    return SurfaceContactDevelopmentContext(semantic_digest({"unit_control": True}), obs.source_id,
        obs.decision_model_hash, obs.objective_hash, array_digest(obs.public_goal_grid),
        obs.crop_origin_native, array_digest(obs.base.base.affine_ras_mm))


@pytest.fixture
def model():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        return GoalModeSpatialPolicy(SpatialPolicyConfig(encoder_channels=(2, 3), hidden_features=8,
                                    ray_samples=3, critic_candidate_context=True))


def test_new_architecture_and_checkpoint_identity_without_loading(model):
    legacy = SpatialPolicy(model.config)
    assert model.architecture_hash != legacy.architecture_hash
    assert model.architecture_record()["version"] == POLICY_VERSION
    identity = model.checkpoint_identity()
    assert identity["version"] == CHECKPOINT_VERSION
    assert identity["parameter_hash"] == parameter_hash(model)
    assert identity["training_lineage"] == "not_attested_by_this_forward_only_adapter"
    assert model.encoder[0].weight.shape[1] == 24
    assert legacy.encoder[0].weight.shape[1] == 18
    assert model.actor[0].weight.shape[1] == legacy.actor[0].weight.shape[1] + 3


def test_exact_dto_and_context_required_before_encoder(model):
    obs = observation(); calls = []
    handle = model.encoder.register_forward_pre_hook(lambda *args: calls.append(True))
    try:
        with pytest.raises(TypeError): model(obs)
        with pytest.raises(TypeError): model(obs, context=None)
        with pytest.raises(TypeError): model(obs.base, context=context(obs))
        with pytest.raises(TypeError): model(obs.base.base, context=context(obs))
        with pytest.raises(TypeError): SpatialPolicy()(obs)
        assert calls == []
    finally: handle.remove()


@pytest.mark.parametrize("field", ["source_hash", "decision_model_hash", "objective_hash",
                                   "goal_grid_hash", "crop_affine_hash", "crop_origin_native"])
def test_wrong_expected_binding_refused_before_encoder(model, field):
    obs = observation(); bound = context(obs)
    wrong = (1, 0, 0) if field == "crop_origin_native" else "sha256:" + "1"*64
    calls = []; handle = model.encoder.register_forward_pre_hook(lambda *args: calls.append(True))
    try:
        with pytest.raises(ValueError): model(obs, context=replace(bound, **{field: wrong}))
        assert calls == []
    finally: handle.remove()


def test_goal_grid_and_shifted_crop_cannot_be_reinterpreted(model):
    obs = observation(); grid = obs.public_goal_grid.copy(); grid[:] = False; grid[1, 2, 3] = True
    with pytest.raises(ValueError): replace(obs, public_goal_grid=grid)
    internally_consistent = replace(obs, public_goal_grid=grid, crop_origin_native=(1, 0, 0))
    with pytest.raises(ValueError): model(internally_consistent, context=context(obs))
    shifted = obs.base.base.affine_ras_mm.copy(); shifted[0, 3] += 1.
    changed_frame = replace(obs, base=replace(obs.base, base=replace(obs.base.base, affine_ras_mm=shifted)))
    with pytest.raises(ValueError): model(changed_frame, context=context(obs))


def test_source_and_objective_cannot_be_consistently_relabelled(model):
    obs = observation(); other_source = "sha256:" + "a"*64
    base = replace(obs.base, base=replace(obs.base.base, source_id=other_source))
    claimed = replace(obs, base=base, objective=PublicSurfaceGoal(other_source, obs.objective.native_index))
    with pytest.raises(ValueError): model(claimed, context=context(obs))
    other_goal = observation(goal=(3, 2, 3))
    with pytest.raises(ValueError): model(other_goal, context=context(obs))


def test_mutated_context_and_goal_rejected(model):
    obs = observation(); bound = context(obs)
    object.__setattr__(bound, "goal_grid_hash", "sha256:" + "b"*64)
    with pytest.raises(ValueError): model(obs, context=bound)
    obs = observation(); bound = context(obs)
    object.__setattr__(obs.objective, "native_index", (3, 2, 3))
    with pytest.raises(ValueError): model(obs, context=bound)


def test_eight_channel_values_coverage_availability_and_modes(model):
    obs = observation(contact=True)
    volume, _, _, geometry, _, _, _ = model._inputs(obs, context=context(obs))
    assert tuple(volume.shape) == (24, 5, 5, 5)
    assert volume[6, 2, 2, 3] == volume[7, 2, 2, 3] == 1
    assert not volume[:16, 0, 0, 0][[0, 6, 7, 8, 14, 15]].any()
    assert volume[22:].all()
    torch.testing.assert_close(geometry[:, 16:], torch.eye(3))


@pytest.mark.parametrize("kind", ["scan", "support", "cavity", "goal", "contact", "mode", "tool"])
def test_public_features_can_change_untrained_outputs(model, kind):
    # Deterministic positive test weights keep each feature path active. These
    # are tensor controls, not optimizer updates or a learned checkpoint.
    with torch.no_grad():
        for parameter in model.parameters():
            ramp = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(parameter.shape)
            parameter.copy_(.01 + .01 * ramp / max(1, parameter.numel()))
    obs = observation()
    if kind == "goal": other = observation(goal=(3, 2, 3))
    elif kind == "contact": other = observation(contact=True)
    elif kind == "mode": other = observation(modes=("stop", "probe", "aspirate"))
    elif kind == "tool":
        geometry = obs.base.base.action_geometry.copy(); geometry[1, 10] *= 1.5
        other = replace(obs, base=replace(obs.base, base=replace(obs.base.base, action_geometry=geometry)))
    else: other = observation(channel_change=kind)
    with torch.no_grad():
        logits, value = model(obs, context=context(obs))
        changed, other_value = model(other, context=context(other))
    assert not torch.equal(logits, changed)
    assert not torch.equal(value, other_value)


def test_action_renaming_and_row_permutation_cannot_create_id_shortcuts(model):
    obs = observation(); base = obs.base.base
    renamed = replace(obs, base=replace(obs.base, base=replace(base,
        action_ids=("STOP", "new-alpha", "new-beta"), action_tool_ids=(None, "renamed-a", "renamed-b"))))
    order = [0, 2, 1]
    permuted = replace(obs, base=replace(obs.base, base=replace(base,
        action_ids=tuple(base.action_ids[i] for i in order),
        action_tool_ids=tuple(base.action_tool_ids[i] for i in order),
        action_geometry=base.action_geometry[order], action_mask=base.action_mask[order]),
        action_modes=tuple(obs.action_modes[i] for i in order)))
    with torch.no_grad():
        original, value = model(obs, context=context(obs))
        names, names_value = model(renamed, context=context(renamed))
        rows, rows_value = model(permuted, context=context(permuted))
    torch.testing.assert_close(names, original, atol=0, rtol=0)
    torch.testing.assert_close(names_value, value, atol=0, rtol=0)
    torch.testing.assert_close(rows, original[order], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(rows_value, value, atol=1e-7, rtol=1e-6)


def test_exhausted_budget_is_stop_only_and_forward_preserves_parameters(model):
    obs = observation(exhausted=True); before = parameter_hash(model)
    with torch.no_grad(): logits, value = model(obs, context=context(obs))
    assert torch.isfinite(logits[0]) and torch.isneginf(logits[1:]).all() and torch.isfinite(value)
    assert model.act(obs, context=context(obs)) == "STOP"
    assert model.act(obs, context=context(obs), stochastic=True,
                     generator=torch.Generator().manual_seed(5)) == "STOP"
    assert parameter_hash(model) == before
    assert all(p.grad is None for p in model.parameters())


def test_mode_grouped_critic_keeps_empty_mode_zero_and_public_geometry_binding(model):
    obs = observation(modes=("stop", "aspirate", "aspirate"))
    _, _, _, geometry, _, _, mask = model._inputs(obs, context=context(obs))
    rays = geometry.new_ones((len(geometry), 4))
    summary = _mode_candidate_critic_context(geometry, rays, mask)
    half = len(summary) // 2
    assert summary[:half].abs().sum() > 0
    assert summary[half:].abs().sum() == 0
    assert summary[half-1] == pytest.approx(2 / 1023)
    exhausted = _mode_candidate_critic_context(geometry, rays, torch.tensor([True, False, False]))
    assert exhausted.abs().sum() == 0
