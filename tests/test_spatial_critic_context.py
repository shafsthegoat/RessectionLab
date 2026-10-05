"""Candidate-value algebra and compatibility; no training or anatomy benchmark."""
import numpy as np
import pytest
import torch

from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.spatial_observations import (
    MAX_ACTIONS, ObservedChannel, ObservedProcedureState, SpatialAction,
    SpatialInputs, build_spatial_observation,
)
from resectionlab.spatial_policy import (
    CANDIDATE_CRITIC_POLICY_VERSION, SpatialPolicy, SpatialPolicyConfig,
    _candidate_critic_context,
)


def observation(*, tip_radius=.5, reverse=False, exhausted=False, concealed=0.):
    shape = (5, 6, 7)
    x, y, z = np.indices(shape)
    scan = (x + 2 * y + 3 * z + 1).astype(np.float32) / 40
    coverage = np.ones(shape, bool)
    coverage[0, 0, 0] = False
    scan[0, 0, 0] = concealed
    tool = ToolGeometry("algebra_tool", tip_radius, .3, 20., tip_length_mm=1.)
    access = AccessWindow((2., 2., 0.), (0., 0., 1.), 4.)
    actions = [SpatialAction("left", access.center_mm, (1., 2., 3.), tool),
               SpatialAction("right", access.center_mm, (3., 2., 4.), tool)]
    if reverse:
        actions.reverse()
    inputs = SpatialInputs({
        "structural_intensity": ObservedChannel(scan, coverage, "synthetic_scan", "unit algebra only"),
        "observed_cavity": ObservedChannel(np.zeros(shape), source_kind="observed_procedure_state"),
    }, np.eye(4), "synthetic_scan", "critic-context-unit-algebra")
    state = ObservedProcedureState(access, 3 if exhausted else 0, 3)
    return build_spatial_observation(inputs, (SpatialAction("STOP"), *actions), state)


def model(*, enabled=True, **kwargs):
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        return SpatialPolicy(SpatialPolicyConfig(critic_candidate_context=enabled, **kwargs))


@pytest.mark.parametrize("options,old_hash,count", [
    ({}, "sha256:10ad4b9b95a2e257d83625516b70aa27908789d644d239cb6a67f660ce49bbde", 24331),
    ({"encoder_channels": (4, 7), "hidden_features": 19, "ray_samples": 3,
      "physical_reference_mm": 7.},
     "sha256:debf3cf2ef95df2e84e4d39fd9c36808ef55b8eac0277e89ded49f61263816a3", 7521),
])
def test_default_architecture_is_exact_committed_v1(options, old_hash, count):
    # Independently imported from git show 00dbe7e:src/resectionlab/spatial_policy.py;
    # source SHA a346b2e187d943238f2676ce2cb90ef609f4655dc7e798e21b247fae41778cb5.
    legacy = model(enabled=False, **options)
    assert legacy.architecture_hash == old_hash
    assert "critic_candidate_context" not in legacy.architecture_record()["config"]
    assert sum(p.numel() for p in legacy.parameters()) == count


def test_opt_in_version_and_actor_weights_remain_unchanged():
    legacy, updated = model(enabled=False), model()
    record = updated.architecture_record()
    assert record["version"] == CANDIDATE_CRITIC_POLICY_VERSION
    assert record["critic_candidate_context"]["count_divisor"] == MAX_ACTIONS - 1
    assert updated.architecture_hash != legacy.architecture_hash
    assert sum(p.numel() for p in updated.parameters()) == 30827
    for name, parameter in legacy.state_dict().items():
        if not name.startswith("critic."):
            torch.testing.assert_close(parameter, updated.state_dict()[name], atol=0, rtol=0)
    torch.testing.assert_close(legacy(observation())[0], updated(observation())[0], atol=0, rtol=0)
    # A v1 checkpoint cannot silently enter the changed critic architecture.
    with pytest.raises(RuntimeError, match="size mismatch"):
        updated.load_state_dict(legacy.state_dict())


def test_legacy_weights_match_an_independent_layer_construction():
    # Same committed layer order/shapes; no dependence on a particular torch RNG version.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        encoder = torch.nn.Sequential(torch.nn.Conv3d(18, 8, 3, padding=1), torch.nn.ReLU(),
                                      torch.nn.Conv3d(8, 16, 3, padding=1), torch.nn.ReLU())
        head = lambda inputs: torch.nn.Sequential(torch.nn.Linear(inputs, 32), torch.nn.ReLU(),
                                                  torch.nn.Linear(32, 1))
        actor, stop, critic = head(242), head(141), head(141)
    legacy = model(enabled=False)
    for module, reference in ((legacy.encoder, encoder), (legacy.actor, actor),
                              (legacy.stop, stop), (legacy.critic, critic)):
        for actual, expected in zip(module.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_tool_context_resolves_controlled_input_alias_without_return_claim():
    small, large = observation(tip_radius=.5), observation(tip_radius=1.5)
    legacy = model(enabled=False)
    torch.testing.assert_close(legacy(small)[1], legacy(large)[1], atol=0, rtol=0)
    updated = model()
    base_context = 16 * 8 + 10 + 3
    # Select the mean normalized tip radius through a single positive ReLU.
    with torch.no_grad():
        for parameter in updated.critic.parameters():
            parameter.zero_()
        updated.critic[0].weight[0, base_context + 10] = 1.
        updated.critic[2].weight[0, 0] = 1.
    torch.testing.assert_close(updated(small)[1], torch.tensor(.05))
    torch.testing.assert_close(updated(large)[1], torch.tensor(.15))


def test_permutation_preserves_value_and_reorders_actor_only():
    updated = model()
    logits, value = updated(observation())
    other, other_value = updated(observation(reverse=True))
    torch.testing.assert_close(other, logits[[0, 2, 1]], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(other_value, value, atol=1e-7, rtol=1e-6)


def test_pooling_excludes_stop_and_masked_rows_and_has_exact_count():
    geometry = torch.tensor([[999., 999.], [1., 3.], [500., 500.], [3., 1.]])
    rays = torch.tensor([[999.], [2.], [500.], [4.]])
    mask = torch.tensor([True, True, False, True])
    expected = torch.tensor([2., 2., 3., 3., 3., 4., 2 / (MAX_ACTIONS - 1)])
    torch.testing.assert_close(_candidate_critic_context(geometry, rays, mask), expected)


def test_pool_gradient_ignores_stop_and_masked_candidate_features():
    geometry = torch.arange(8, dtype=torch.float32).reshape(4, 2).requires_grad_()
    rays = torch.arange(4, dtype=torch.float32).reshape(4, 1).requires_grad_()
    mask = torch.tensor([True, True, False, True])
    _candidate_critic_context(geometry, rays, mask).sum().backward()
    for gradient in (geometry.grad, rays.grad):
        assert torch.count_nonzero(gradient[[0, 2]]) == 0
        assert (gradient[[1, 3]] > 0).all()


@pytest.mark.parametrize("rows", [1, 3])
def test_stop_only_pooling_is_finite_zero(rows):
    geometry, rays = torch.ones(rows, 16), torch.ones(rows, 85)
    mask = torch.zeros(rows, dtype=torch.bool)
    mask[0] = True
    actual = _candidate_critic_context(geometry, rays, mask)
    torch.testing.assert_close(actual, torch.zeros(203), atol=0, rtol=0)
    logits, value = model()(observation(exhausted=True))
    assert torch.isfinite(value) and torch.isfinite(logits[0])
    assert torch.isneginf(logits[1:]).all()


def test_coverage_erases_concealed_intensity_before_critic():
    updated = model()
    logits, value = updated(observation(concealed=0.))
    other_logits, other_value = updated(observation(concealed=999.))
    torch.testing.assert_close(other_logits, logits, atol=0, rtol=0)
    torch.testing.assert_close(other_value, value, atol=0, rtol=0)


def test_candidate_ray_value_gradient_reaches_encoder_without_actor_head():
    updated = model()
    base_context = 16 * 8 + 10 + 3
    with torch.no_grad():
        for layer in (updated.encoder[0], updated.encoder[2]):
            layer.weight.fill_(.01)
            layer.bias.fill_(.1)
        for parameter in updated.critic.parameters():
            parameter.zero_()
        updated.critic[0].weight[0, base_context + 16] = 1.
        updated.critic[2].weight[0, 0] = 1.
    before = [parameter.detach().clone() for parameter in updated.parameters()]
    updated(observation())[1].backward()
    assert updated.encoder[0].weight.grad.abs().sum() > 0
    assert updated.encoder[2].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in updated.actor.parameters())
    # Derivative check only: no optimizer, parameter update, rollout or reward.
    for original, actual in zip(before, updated.parameters(), strict=True):
        torch.testing.assert_close(original, actual.detach(), atol=0, rtol=0)


@pytest.mark.parametrize("bad", [0, 1, "yes", None])
def test_opt_in_requires_a_boolean(bad):
    with pytest.raises(ValueError, match="boolean"):
        SpatialPolicyConfig(critic_candidate_context=bad)
