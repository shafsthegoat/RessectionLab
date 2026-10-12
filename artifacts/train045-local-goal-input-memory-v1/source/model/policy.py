"""Batched MaskablePPO policy with the exact IL64 public spatial feature path.

No native task, candidate reward, cached actor feature, or reference anatomy is
available here. The only extension is eight/four zero-initialized goal columns.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths
import hashlib
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig
from resectionlab.public_target_context import VERSION as TARGET_VERSION

VERSION = 'il64-batched-public-local-goal-ppo-v1'
MAX_ACTIONS = 153


class TensorBoundary(BaseFeaturesExtractor):
    """SB3 constructor metadata only; never flatten spatial observations."""
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=1)
    def forward(self, observations):
        raise RuntimeError('Use the explicit batched spatial tensor path')


def module_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def _extend(head, extra):
    old = head[0]
    new = nn.Linear(old.in_features + extra, old.out_features,
                    device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_(); new.weight[:, :old.in_features].copy_(old.weight)
        new.bias.copy_(old.bias)
    head[0] = new


class LocalGoalPolicy(MaskableActorCriticPolicy):
    """Shared row scorer, separate STOP, mask-aware critic; actual raw CNN grads."""
    def __init__(self, observation_space, action_space, lr_schedule, *,
                 freeze_encoder=False, **kwargs):
        self.freeze_encoder = bool(freeze_encoder)
        for key in ('net_arch', 'features_extractor_class', 'ortho_init', 'normalize_images'):
            kwargs.pop(key, None)
        super().__init__(observation_space, action_space, lr_schedule, net_arch=[],
            features_extractor_class=TensorBoundary, ortho_init=False,
            normalize_images=False, **kwargs)

    def _build(self, lr_schedule):
        if self.action_space.n != MAX_ACTIONS:
            raise ValueError('Fixed 153 padded current candidate rows required')
        self.spatial = SpatialPolicy(SpatialPolicyConfig(critic_candidate_context=True),
                                     public_target_context_variant=TARGET_VERSION)
        _extend(self.spatial.actor, 8); _extend(self.spatial.stop, 4); _extend(self.spatial.critic, 4)
        self.spatial.encoder.requires_grad_(not self.freeze_encoder)
        self.optimizer = self.optimizer_class((p for p in self.parameters() if p.requires_grad),
            lr=lr_schedule(1), **self.optimizer_kwargs)
        self.costs = {'forward_calls': 0, 'forward_rows': 0, 'forward_seconds': 0.,
                      'optimizer_attempted': 0, 'optimizer_completed': 0}
        self.update_rows = []
        self.optimizer.register_step_pre_hook(self._before_update)
        self.optimizer.register_step_post_hook(self._after_update)

    def _before_update(self, optimizer, args, kwargs):
        self.costs['optimizer_attempted'] += 1
        self._update_started = time.perf_counter()
        all_sq = enc_sq = 0.
        encoder = {id(p) for p in self.spatial.encoder.parameters()}
        for p in self.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    raise FloatingPointError('Nonfinite PPO gradient')
                square = float(p.grad.detach().double().square().sum())
                all_sq += square
                if id(p) in encoder: enc_sq += square
        self._pending_update = {'attempt': self.costs['optimizer_attempted'],
            'post_clip_gradient_norm': all_sq ** .5, 'post_clip_encoder_gradient_norm': enc_sq ** .5}
        if self.freeze_encoder and (enc_sq != 0. or any(p.grad is not None for p in self.spatial.encoder.parameters())):
            raise RuntimeError('Frozen encoder received a gradient')

    def _after_update(self, optimizer, args, kwargs):
        self.costs['optimizer_completed'] += 1
        self.update_rows.append({**self._pending_update,
            'seconds': time.perf_counter() - self._update_started,
            'completed': self.costs['optimizer_completed']})

    def adopt_il64(self, original):
        """Call once before PPO learns; copy exact old tensors, zero only new cols."""
        if (type(original) is not SpatialPolicy or original.config != self.spatial.config
                or original.public_target_context_variant != TARGET_VERSION
                or self.costs['optimizer_attempted'] or self.optimizer.state):
            raise ValueError('Original compatible IL64 and untouched optimizer required')
        old = original.state_dict(); new = self.spatial.state_dict()
        with torch.no_grad():
            for name, value in old.items():
                target = new[name]
                if name in ('actor.0.weight', 'stop.0.weight', 'critic.0.weight'):
                    target.zero_(); target[:, :value.shape[1]].copy_(value)
                else:
                    if target.shape != value.shape: raise ValueError('IL64 tensor dimensions changed')
                    target.copy_(value)
        self.spatial.load_state_dict(new, strict=True)
        self.initial_encoder_hash = module_hash(self.spatial.encoder)
        return {'version': VERSION, 'original_architecture': original.architecture_hash,
            'encoder_hash': self.initial_encoder_hash, 'new_columns_zero': True,
            'head_input_widths': [270, 161, 364], 'freeze_encoder': self.freeze_encoder}

    def _get_constructor_parameters(self):
        result = super()._get_constructor_parameters()
        result['freeze_encoder'] = self.freeze_encoder
        return result

    def logits_values(self, obs, action_masks=None):
        started = time.perf_counter(); self.costs['forward_calls'] += 1
        try:
            image = obs['images'].float(); coverage = obs['coverage'].bool()
            available = obs['available'].bool()
            if image.ndim != 5 or image.shape[1] != 6 or coverage.shape != image.shape:
                raise ValueError('Batched raw six-channel spatial values/coverage required')
            batch = image.shape[0]; self.costs['forward_rows'] += batch
            expected = {'available': (batch, 6), 'geometry': (batch, MAX_ACTIONS, 16),
                'grid': (batch, MAX_ACTIONS, 5, 3), 'inside': (batch, MAX_ACTIONS, 5),
                'state': (batch, 10), 'spacing': (batch, 3), 'public_target': (batch, 16),
                'public_relation': (batch, MAX_ACTIONS, 4), 'goal_global': (batch, 4),
                'goal_relation': (batch, MAX_ACTIONS, 4), 'action_mask': (batch, MAX_ACTIONS)}
            for name, shape in expected.items():
                if tuple(obs[name].shape) != shape or not torch.isfinite(obs[name]).all():
                    raise ValueError('Malformed public tensor: ' + name)
            if not torch.isfinite(image).all(): raise ValueError('Nonfinite public image')
            mask = obs['action_mask'].bool()
            if not mask[:, 0].all(): raise ValueError('STOP must be available')
            if action_masks is not None:
                external = torch.as_tensor(action_masks, device=mask.device, dtype=torch.bool).reshape_as(mask)
                if not torch.equal(external, mask): raise ValueError('Stored observation/mask mismatch')
            visible = coverage & available[:, :, None, None, None]
            # Preserve the exact 18-channel input without three dense float
            # operands coexisting with a concatenated copy.
            volume = image.new_empty((batch, 18, *image.shape[2:]))
            volume[:, :6].copy_(image)
            volume[:, :6].masked_fill_(~visible, 0.)
            volume[:, 6:12].copy_(visible)
            volume[:, 12:].copy_(available[:, :, None, None, None])
            if self.freeze_encoder:
                with torch.no_grad(): encoded = self.spatial.encoder(volume)
            else: encoded = self.spatial.encoder(volume)
            del volume
            context = torch.cat((F.adaptive_avg_pool3d(encoded, (2, 2, 2)).flatten(1),
                                 obs['state'].float(), obs['spacing'].float()), dim=1)
            grid = obs['grid'].float(); inside = obs['inside'].float()
            sampled = F.grid_sample(encoded, grid[:, :, :, None, :], mode='bilinear',
                                     padding_mode='zeros', align_corners=True)
            rays = sampled[..., 0].permute(0, 2, 3, 1)
            rays = rays * (grid.abs() <= 1).all(-1, keepdim=True) * inside[..., None]
            rays = torch.cat((rays, inside[..., None]), dim=-1).flatten(2)
            del encoded
            geometry = obs['geometry'].float(); target = obs['public_target'].float()
            goal = obs['goal_global'].float()
            action_input = torch.cat((context[:, None, :].expand(-1, MAX_ACTIONS, -1), geometry, rays,
                target[:, None, :].expand(-1, MAX_ACTIONS, -1), obs['public_relation'].float(),
                goal[:, None, :].expand(-1, MAX_ACTIONS, -1), obs['goal_relation'].float()), dim=-1)
            action_logits = self.spatial.actor(action_input).squeeze(-1)
            stop = self.spatial.stop(torch.cat((context, target, goal), dim=1))
            logits = torch.cat((stop, action_logits[:, 1:]), dim=1).masked_fill(~mask, -torch.inf)
            candidates = torch.cat((geometry, rays), dim=-1)[:, 1:]
            legal = mask[:, 1:]; count = legal.sum(1, keepdim=True)
            mean = (candidates * legal[..., None]).sum(1) / count.clamp_min(1)
            maximum = candidates.masked_fill(~legal[..., None], -torch.inf).amax(1)
            maximum = torch.where(count > 0, maximum, 0.)
            # Preserve the original DTO bound's divisor, not the padded count.
            from resectionlab.spatial_observations import MAX_ACTIONS as DTO_MAX_ACTIONS
            critic = torch.cat((context, mean, maximum, count / (DTO_MAX_ACTIONS - 1), target, goal), dim=1)
            values = self.spatial.critic(critic)
            if not torch.isfinite(logits[mask]).all() or not torch.isfinite(values).all():
                raise FloatingPointError('Nonfinite batched spatial policy output')
            return logits, values
        finally: self.costs['forward_seconds'] += time.perf_counter() - started

    def _distribution(self, obs, action_masks=None):
        logits, values = self.logits_values(obs, action_masks)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        distribution.apply_masking(obs['action_mask'].bool())
        return distribution, values

    def forward(self, obs, deterministic=False, action_masks=None):
        distribution, values = self._distribution(obs, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions.reshape(-1), values, distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions, action_masks=None):
        distribution, values = self._distribution(obs, action_masks)
        return values, distribution.log_prob(actions.long().flatten()), distribution.entropy()

    def get_distribution(self, obs, action_masks=None):
        return self._distribution(obs, action_masks)[0]

    def predict_values(self, obs):
        return self.logits_values(obs)[1]
