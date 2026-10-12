"""Small generated tensor tests; no acquired files, weights or native trials."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy import LocalGoalPolicy, module_hash, MAX_ACTIONS
from paths import ROOT
sys.path.insert(0, str(ROOT/'tests'))
from test_public_target_context import observation
from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig
from resectionlab.public_target_context import VERSION, public_target_features
import numpy as np
import torch
import pytest
from gymnasium import spaces


def fixture(frozen=False):
    torch.manual_seed(17)
    original = SpatialPolicy(SpatialPolicyConfig(critic_candidate_context=True),
                             public_target_context_variant=VERSION).eval()
    dto, _, _ = observation(goal=(9, 3, 3))
    volume, grid, inside, geometry, state, spacing, mask = original._inputs(dto)
    target, relation = public_target_features(dto, reference_mm=10.)
    def pad(value, fill=0):
        out = torch.full((MAX_ACTIONS, *value.shape[1:]), fill, dtype=torch.float32)
        out[:len(value)] = torch.as_tensor(value); return out
    arrays = {'images': torch.tensor(np.array(dto.image_channels)),
        'coverage': torch.tensor(np.array(dto.coverage), dtype=torch.uint8),
        'available': torch.tensor(np.array(dto.channel_available), dtype=torch.uint8),
        'geometry': pad(geometry), 'grid': pad(grid, 2), 'inside': pad(inside),
        'state': state, 'spacing': spacing, 'public_target': torch.tensor(target).float(),
        'public_relation': pad(relation), 'goal_global': torch.tensor([1., .8, .3, .2]),
        'goal_relation': pad(np.tile([.3, .1, .2, .374], (len(mask), 1))),
        'action_mask': pad(mask).to(torch.uint8)}
    space = spaces.Dict({k: spaces.Box(0, 1, tuple(v.shape), np.uint8) if v.dtype == torch.uint8
                        else spaces.Box(-np.inf, np.inf, tuple(v.shape), np.float32) for k, v in arrays.items()})
    policy = LocalGoalPolicy(space, spaces.Discrete(MAX_ACTIONS), lambda _: 3e-4, freeze_encoder=frozen)
    policy.adopt_il64(original)
    batch = {k: v[None] for k, v in arrays.items()}
    return original, dto, policy, batch


def test_initial_IL64_public_target_exact_path_and_batch_permutation():
    original, dto, policy, batch = fixture()
    with torch.no_grad():
        expected, value = original(dto); actual, actual_value = policy.logits_values(batch)
        torch.testing.assert_close(actual[0, :len(expected)], expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual_value[0, 0], value, rtol=2e-5, atol=2e-6)
        doubled = {k: v.repeat(2, *([1]*(v.ndim-1))) for k, v in batch.items()}
        logits, values = policy.logits_values(doubled)
        torch.testing.assert_close(logits[0], actual[0], rtol=2e-5, atol=2e-6)
        # Move the real ray to a distant padded position. No slot embedding exists.
        order = torch.arange(MAX_ACTIONS); order[1], order[91] = order[91].clone(), order[1].clone()
        changed = {k: v[:, order] if k in ('geometry','grid','inside','public_relation','goal_relation','action_mask') else v
                   for k, v in batch.items()}
        moved, moved_value = policy.logits_values(changed)
        torch.testing.assert_close(moved[:, order], actual, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(moved_value, actual_value, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('frozen', [False, True])
def test_raw_pixel_encoder_and_goal_column_gradients_and_counted_update(frozen):
    _, _, policy, batch = fixture(frozen)
    before = module_hash(policy.spatial.encoder)
    batch['images'].requires_grad_(True)
    logits, values = policy.logits_values(batch)
    loss = -logits[:, :2].log_softmax(-1)[:, 1].mean() + (values - 3.).square().mean()
    policy.optimizer.zero_grad(); loss.backward()
    enc_grads = [p.grad for p in policy.spatial.encoder.parameters()]
    assert (all(g is None for g in enc_grads)) == frozen
    if not frozen:
        assert sum(float(g.abs().sum()) for g in enc_grads) > 0
        assert batch['images'].grad is not None and batch['images'].grad.abs().sum() > 0
    assert policy.spatial.actor[0].weight.grad[:, -8:].abs().sum() > 0
    torch.nn.utils.clip_grad_norm_(policy.parameters(), .5); policy.optimizer.step()
    assert (module_hash(policy.spatial.encoder) == before) == frozen
    assert policy.costs['optimizer_attempted'] == policy.costs['optimizer_completed'] == 1
    assert (policy.update_rows[-1]['post_clip_encoder_gradient_norm'] == 0) == frozen
    changed = {**batch, 'goal_global': batch['goal_global'] + 2}
    with torch.no_grad():
        one, _ = policy.logits_values(batch); two, _ = policy.logits_values(changed)
    assert not torch.equal(one[:, :2], two[:, :2])


def test_mask_unknown_payload_STOP_and_no_extra_observation_api():
    _, _, policy, batch = fixture()
    mask = batch['action_mask'].bool(); wrong = mask.clone(); wrong[:, 2] = True
    with pytest.raises(ValueError, match='mask mismatch'): policy.forward(batch, action_masks=wrong)
    clean = {k: v.clone() for k, v in batch.items()}
    clean['coverage'][:, :, 0] = 0
    altered = {k: v.clone() for k, v in clean.items()}; altered['images'][:, :, 0] = 999
    with torch.no_grad():
        a, av = policy.logits_values(clean); b, bv = policy.logits_values(altered)
    torch.testing.assert_close(a, b, rtol=0, atol=0); torch.testing.assert_close(av, bv, rtol=0, atol=0)
    clean['action_mask'][:, 1:] = 0
    action, value, logprob = policy.forward(clean, action_masks=clean['action_mask'])
    assert action.tolist() == [0] and torch.isfinite(value).all() and logprob.item() == 0
    bad = {**batch, 'images': batch['images'].clone()}; bad['images'].flatten()[0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'): policy.forward(bad)


def test_real_SB3_rollout_update_checkpoint_restores_optimizer(tmp_path):
    import gymnasium as gym
    from sb3_contrib import MaskablePPO
    _, _, source, batch = fixture()
    class Tiny(gym.Env):
        observation_space = source.observation_space
        action_space = source.action_space
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed); return {k: v[0].detach().numpy().copy() for k,v in batch.items()}, {}
        def step(self, action):
            return self.reset()[0], float(action == 1), True, False, {}
        def action_masks(self): return batch['action_mask'][0].bool().numpy()
    model = MaskablePPO(LocalGoalPolicy, Tiny(), n_steps=4, batch_size=2, n_epochs=1, seed=3, device='cpu')
    model.policy.load_state_dict(source.state_dict(), strict=True)
    model.learn(4)
    assert model.num_timesteps == 4 and model.policy.costs['optimizer_completed'] == 2
    path = tmp_path/'generated-ppo.zip'; model.save(path)
    restored = MaskablePPO.load(path, env=Tiny(), device='cpu')
    assert module_hash(restored.policy.spatial) == module_hash(model.policy.spatial)
    assert len(restored.policy.optimizer.state) == len(model.policy.optimizer.state) > 0
    assert restored.predict(Tiny().reset()[0], deterministic=True, action_masks=Tiny().action_masks())[0] in (0,1)
