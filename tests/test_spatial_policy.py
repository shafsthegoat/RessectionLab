"""Small algebra and single-update checks, not a generalization experiment."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from resectionlab.core import array_digest, semantic_digest
from resectionlab.data_policy import DataPolicyError, GeneratedDevelopmentContext
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.spatial_observations import (ObservedChannel, ObservedProcedureState,
    SpatialAction, SpatialInputs, build_spatial_observation)
from resectionlab.spatial_policy import (SpatialPolicy, SpatialPolicyConfig, SpatialTransition,
    gradient_step, imitation_loss, parameter_hash, reinforce_loss, sample_ray_features,
    world_to_sample_grid)


def observation(*, transform=None, order=(0, 1, 2), exhausted=False, concealed=0.):
    shape = (5, 6, 7)
    x, y, z = np.indices(shape)
    scan = (x + 2*y + 3*z).astype(np.float32)/40.
    coverage = np.ones(shape, bool)
    coverage[0, 0, 0] = False
    scan[0, 0, 0] = concealed
    affine = np.diag([1., 1.5, 2., 1.])
    affine[:3, 3] = [11., -8., 3.]
    frame = np.eye(4) if transform is None else transform
    point = lambda p: tuple((frame @ affine @ np.r_[p, 1.])[:3])
    normal = frame[:3, :3] @ np.array([0., 0., 1.])
    access = AccessWindow(point((2., 2., 0.)), tuple(normal), 4.)
    tool = ToolGeometry("unit_tool", .5, .3, 20., tip_length_mm=1.)
    actions = (SpatialAction("STOP"),
        SpatialAction("left", point((2., 2., 0.)), point((1., 2., 3.)), tool),
        SpatialAction("right", point((2., 2., 0.)), point((3., 2., 4.)), tool))
    inputs = SpatialInputs({
        "structural_intensity": ObservedChannel(scan, coverage, "synthetic_scan", "analytic ramp"),
        "observed_cavity": ObservedChannel(np.zeros(shape), source_kind="observed_procedure_state")},
        frame @ affine, "synthetic_scan", semantic_digest({
            "fixture": "spatial-policy-analytic-ramp-v1", "scan": array_digest(scan),
            "coverage": array_digest(coverage), "affine": array_digest(frame @ affine)}))
    state = ObservedProcedureState(access, 2 if exhausted else 0, 2)
    return build_spatial_observation(inputs, tuple(actions[i] for i in order), state)


def generated_context(obs):
    # Software-only declaration for the explicit generated two-step contract;
    # no patient, checkpoint, native reward fidelity, or generalization claim.
    return GeneratedDevelopmentContext(
        semantic_digest({"fixture": "spatial-policy-unit-controls-v1", "patient_count": 0}),
        (obs.source_id,),
        semantic_digest({"reward_source": "test-specified analytical constants", "max_steps": 2}))


@pytest.mark.parametrize("function", [imitation_loss, reinforce_loss, gradient_step])
def test_missing_generated_context_rejected_before_inputs(function):
    with pytest.raises(DataPolicyError, match="RECORDED_EXPERIENCE_REQUIRED"):
        function()


def policy():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        return SpatialPolicy()


def test_actual_spatial_encoder_receives_scan_gradient_and_updates():
    model = policy(); obs = observation()
    initial = parameter_hash(model)
    loss, stats = reinforce_loss(model, [[SpatialTransition(obs, "left", 2., True)]],
                                 learning_context=generated_context(obs))
    receipt = gradient_step(model, torch.optim.Adam(model.parameters(), lr=.001), loss,
                            learning_context=generated_context(obs))
    assert stats["loss_forward_calls"] == 1 and stats["completed_episodes"] == 1
    assert receipt["encoder_gradient_norm_before_clip"] > 0
    assert receipt["actor_gradient_norm_before_clip"] > 0
    assert receipt["critic_gradient_norm_before_clip"] > 0
    assert receipt["parameters_changed"] and parameter_hash(model) != initial
    assert receipt["gradient_norm_after_clip"] <= 5.00001


def test_behavior_cloning_has_real_encoder_gradients_without_claiming_rl():
    model = policy(); obs = observation()
    loss, stats = imitation_loss(model, [(obs, "right")], learning_context=generated_context(obs))
    loss.backward()
    assert stats["kind"] == "search_action_behavior_cloning"
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.critic.parameters())


def test_value_and_action_predictions_change_when_only_spatial_scan_changes():
    model = policy(); first = observation()
    image = first.image_channels.copy()
    image[0, 1:3, 1:4, 1:5] += 2.
    second = replace(first, image_channels=image)
    logits1, value1 = model(first); logits2, value2 = model(second)
    assert not torch.equal(logits1, logits2)
    assert not torch.equal(value1, value2)
    assert np.array_equal(first.state_features, second.state_features)
    assert np.array_equal(first.action_geometry, second.action_geometry)


def test_row_permutation_and_physical_frame_reexpression():
    model = policy(); original = observation()
    permuted = observation(order=(0, 2, 1))
    transform = np.array([[0., -1., 0., 60.], [1., 0., 0., -30.],
                          [0., 0., 1., 11.], [0., 0., 0., 1.]])
    moved = observation(transform=transform)
    logits, value = model(original)
    other, v_other = model(permuted)
    physical, v_physical = model(moved)
    torch.testing.assert_close(other, logits[[0, 2, 1]], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(v_other, value, atol=0, rtol=0)
    torch.testing.assert_close(physical, logits, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(v_physical, value, atol=1e-6, rtol=1e-5)


def test_masked_budget_and_concealed_samples_cannot_influence_policy():
    model = policy(); obs = observation(exhausted=True)
    logits, value = model(obs)
    assert torch.isfinite(logits[0]) and torch.isneginf(logits[1:]).all()
    assert model.act(obs, stochastic=True, generator=torch.Generator().manual_seed(4)) == "STOP"
    other, v_other = model(observation(exhausted=True, concealed=900.))
    torch.testing.assert_close(other, logits, atol=0, rtol=0)
    torch.testing.assert_close(v_other, value, atol=0, rtol=0)


def test_grid_coordinate_order_and_strict_outside_unknown():
    x, y, z = np.indices((4, 5, 6))
    features = torch.tensor((x+10*y+100*z)[None], dtype=torch.float32)
    points = np.array([[[1., 2., 3.], [1.5, 2.25, 3.5], [3.25, 2., 3.]]])
    grid = torch.tensor(world_to_sample_grid(points, np.eye(4), (4, 5, 6)), dtype=torch.float32)
    sampled = sample_ray_features(features, grid)
    torch.testing.assert_close(sampled[0, :, 0], torch.tensor([321., 374., 0.]), atol=1e-4, rtol=0)


@pytest.mark.parametrize("bad", ["stale_action", "partial_episode", "masked_action", "nonfinite_reward"])
def test_invalid_training_evidence_rejected_before_backward(bad):
    model = policy(); obs = observation(exhausted=bad == "masked_action")
    action = "obsolete" if bad == "stale_action" else "left"
    reward = float("nan") if bad == "nonfinite_reward" else 1.
    episode = [SpatialTransition(obs, action, reward, bad != "partial_episode")]
    expected = {"stale_action": "absent", "partial_episode": "complete episodes",
                "masked_action": "masked", "nonfinite_reward": "Nonfinite transition reward"}
    with pytest.raises(ValueError, match=expected[bad]):
        reinforce_loss(model, [episode], learning_context=generated_context(obs))
    assert all(p.grad is None for p in model.parameters())


def test_frozen_inference_changes_no_weights_or_gradients():
    model = policy().eval().requires_grad_(False)
    before = parameter_hash(model)
    result = model.act(observation())
    assert result in observation().action_ids and parameter_hash(model) == before
    assert all(p.grad is None for p in model.parameters())


def test_configuration_and_nonfinite_geometry_fail_closed():
    with pytest.raises(ValueError):
        SpatialPolicyConfig(ray_samples=1)
    with pytest.raises(ValueError):
        world_to_sample_grid([[complex(1, 2), 0, 0]], np.eye(4), (4, 5, 6))
    with pytest.raises(ValueError):
        world_to_sample_grid([[0., 0., 0.]], np.eye(4), (1, 5, 6))


class AnalyticPolicy(torch.nn.Module):
    architecture_hash = semantic_digest({"fixture": "analytic-two-logit-policy-v1",
                                         "logits": ["zero", "theta"], "value": "zero"})

    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, observation):
        return torch.stack((self.theta * 0., self.theta)), self.theta * 0.


def test_analytic_variable_length_expected_return_gradient():
    model = AnalyticPolicy()
    obs = observation(order=(0, 1))
    episodes = [[SpatialTransition(obs, "left", 1., True)],
                [SpatialTransition(obs, "left", 0., False), SpatialTransition(obs, "left", 2., True)]]
    loss, _ = reinforce_loss(model, episodes, entropy_weight=0., value_weight=0.,
                             learning_context=generated_context(obs))
    loss.backward()
    # At theta0, dlogpi(left)/dtheta=.5. Mean trajectory sums:
    # -(1*.5 + (2*.5+2*.5))/2 = -1.25, not mean-per-transition -.75.
    assert model.theta.grad.item() == pytest.approx(-1.25)


def test_analytic_discounted_start_state_return_gradient():
    model = AnalyticPolicy()
    obs = observation(order=(0, 1))
    episode = [SpatialTransition(obs, "left", 0., False), SpatialTransition(obs, "left", 2., True)]
    loss, _ = reinforce_loss(model, [episode], gamma=.5, entropy_weight=0., value_weight=0.,
                             learning_context=generated_context(obs))
    loss.backward()
    # Returns[1,2] at steps[0,1]: -(.5*1 + .5*.5*2) = -1.
    assert model.theta.grad.item() == pytest.approx(-1.)
