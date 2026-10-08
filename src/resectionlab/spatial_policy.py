"""Small scan-conditioned research policy; independent of legacy feature policies.

The policy consumes only the explicit observation DTO. It never receives a
simulator, reference segmentation, reward estimate, or per-action tissue sum.
Axes follow the repository's C,X,Y,Z arrays; torch grid coordinates are Z,Y,X.
No clinical or real-image transfer claim is made by this prototype.
"""
from __future__ import annotations

from .data_policy import GeneratedDevelopmentContext, generated_development_only

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

POLICY_VERSION = "spatial-scan-ray-conv-policy-v1"
CANDIDATE_CRITIC_POLICY_VERSION = "spatial-scan-ray-conv-candidate-critic-v2"
GRID_ROUNDOFF_BAND = 64 * np.finfo(np.float64).eps


@dataclass(frozen=True)
class SpatialPolicyConfig:
    encoder_channels: tuple[int, int] = (8, 16)
    hidden_features: int = 32
    ray_samples: int = 5
    physical_reference_mm: float = 10.0
    critic_candidate_context: bool = False

    def __post_init__(self):
        if (len(self.encoder_channels) != 2 or any(type(v) is not int or not 1 <= v <= 64
                for v in self.encoder_channels) or type(self.hidden_features) is not int
                or not 1 <= self.hidden_features <= 128 or type(self.ray_samples) is not int
                or not 2 <= self.ray_samples <= 17):
            raise ValueError("Invalid bounded spatial architecture")
        if not np.isfinite(self.physical_reference_mm) or self.physical_reference_mm <= 0:
            raise ValueError("Physical reference unit must be positive and finite")
        if type(self.critic_candidate_context) is not bool:
            raise ValueError("Candidate critic context requires an explicit boolean")
        object.__setattr__(self, "encoder_channels", tuple(self.encoder_channels))


def _real_finite(value, name: str) -> np.ndarray:
    value = np.asarray(value)
    if np.iscomplexobj(value) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(name + " must be real and finite")
    return np.asarray(value, dtype=np.float64)


def world_to_sample_grid(points_mm, affine_ras_mm, shape_xyz) -> np.ndarray:
    """Map physical points to align_corners=True ZYX coordinates, unclipped.

Out-of-domain samples retain their coordinates and are separately flagged;
they are never clamped onto observed tissue. Singleton axes are unsupported.
"""
    points = _real_finite(points_mm, "points")
    affine = _real_finite(affine_ras_mm, "affine")
    shape = np.asarray(shape_xyz)
    if (points.shape[-1:] != (3,) or affine.shape != (4, 4)
            or not np.array_equal(affine[3], [0., 0., 0., 1.])
            or shape.shape != (3,) or shape.dtype.kind not in "iu" or np.any(shape < 2)):
        raise ValueError("Invalid affine/points/spatial shape")
    try:
        inv = np.linalg.inv(affine)
    except np.linalg.LinAlgError as error:
        raise ValueError("Singular spatial affine") from error
    with np.errstate(over="ignore", invalid="ignore"):
        voxels = points @ inv[:3, :3].T + inv[:3, 3]
        normalized = 2. * voxels / (shape - 1) - 1.
    if not np.isfinite(normalized).all():
        raise ValueError("Nonfinite physical-to-grid conversion")
    # Exact affine-transformed corners can round a few ulps outside. Snap only
    # within this declared normalized-coordinate machine-roundoff band, never
    # a material outside point. This is numerical conditioning, not clearance.
    normalized = np.where(np.abs(np.abs(normalized) - 1.) <= GRID_ROUNDOFF_BAND,
                          np.sign(normalized), normalized)
    return normalized[..., ::-1].copy()


def sample_ray_features(encoded: Tensor, grid_zyx: Tensor) -> Tensor:
    """CXYZ feature grid and [actions,samples,3] grid -> [actions,samples,C]."""
    if (encoded.ndim != 4 or grid_zyx.ndim != 3 or grid_zyx.shape[-1] != 3
            or not torch.isfinite(encoded).all() or not torch.isfinite(grid_zyx).all()):
        raise ValueError("Invalid spatial feature sampling tensors")
    # torch's D,H,W axes here are X,Y,Z, so the final grid coordinate is Z,Y,X.
    sampled = F.grid_sample(encoded.unsqueeze(0), grid_zyx[None, :, :, None, :],
                            mode="bilinear", padding_mode="zeros", align_corners=True)
    sampled = sampled[0, :, :, :, 0].permute(1, 2, 0)
    # grid_sample otherwise interpolates a partial boundary value just outside
    # [-1,1]. The explicit contract treats the entire outside sample as unknown.
    return sampled * (grid_zyx.abs() <= 1.).all(dim=-1, keepdim=True)


def _candidate_critic_context(geometry: Tensor, rays: Tensor, mask: Tensor) -> Tensor:
    """Mean/max permitted features over legal non-STOP rows and bounded count.

    STOP placeholders and masked actions cannot contribute. This describes the
    current inventory, not tools absent from it or future attainable return.
    """
    from .spatial_observations import MAX_ACTIONS

    candidates = torch.cat((geometry, rays), dim=-1)[1:][mask[1:]]
    if not len(candidates):
        return geometry.new_zeros(2 * (geometry.shape[-1] + rays.shape[-1]) + 1)
    count = candidates.new_tensor([len(candidates) / (MAX_ACTIONS - 1)])
    return torch.cat((candidates.mean(dim=0), candidates.amax(dim=0), count))


class SpatialPolicy(nn.Module):
    """Shared geometry-conditioned action scorer and spatial value baseline.

Row permutation changes only the logit order. Global 2x2x2 pooling deliberately
retains coarse spatial information; neither this CNN nor that pooling is
claimed equivariant to voxel reindexing or complete for partially observed tasks.
"""
    def __init__(self, config: SpatialPolicyConfig | None = None):
        super().__init__()
        self.config = config or SpatialPolicyConfig()
        a, b = self.config.encoder_channels
        self.encoder = nn.Sequential(nn.Conv3d(18, a, 3, padding=1), nn.ReLU(),
                                     nn.Conv3d(a, b, 3, padding=1), nn.ReLU())
        # Original values, coverage, and explicit availability have separate
        # channels. Missing values are zeroed before any convolution.
        global_count = b * 8
        context_count = global_count + 10 + 3  # observed procedure and spacing
        action_count = context_count + 16 + self.config.ray_samples * (b + 1)
        width = self.config.hidden_features
        self.actor = nn.Sequential(nn.Linear(action_count, width), nn.ReLU(), nn.Linear(width, 1))
        self.stop = nn.Sequential(nn.Linear(context_count, width), nn.ReLU(), nn.Linear(width, 1))
        candidate_count = 2 * (16 + self.config.ray_samples * (b + 1)) + 1
        critic_count = context_count + (candidate_count if self.config.critic_candidate_context else 0)
        self.critic = nn.Sequential(nn.Linear(critic_count, width), nn.ReLU(), nn.Linear(width, 1))

    def architecture_record(self) -> dict:
        config_record = asdict(self.config)
        if not self.config.critic_candidate_context:
            # Preserve the exact v1 serialized architecture/hash, not just its
            # layer shapes, when the new research variant is not requested.
            del config_record["critic_candidate_context"]
        record = {"version": CANDIDATE_CRITIC_POLICY_VERSION if self.config.critic_candidate_context else POLICY_VERSION,
                "config": config_record,
                "image_layout": "CXYZ", "channels": 6, "grid_sample_order": "ZYX",
                "align_corners": True, "padding": "zeros_with_inbounds_flags",
                "grid_roundoff_band": GRID_ROUNDOFF_BAND,
                "image_input": "masked_values+coverage+availability",
                "action_geometry": "stop,entry3,tip3,axis3,tip_radius,shaft_radius,working_length,tip_length,max_access_angle_deg,tool_change",
                "procedure_state": "steps_taken,max_steps,current_tool_present,access_center3,access_normal3,access_radius",
                "spatial_pool": [2, 2, 2], "return_transform": "none",
                "inference_ties": "argmax_first_in_inventory; STOP is index0",
                "normalization": "source-grid entry/tip/access; normalized source-axis direction; physical dimensions/reference_mm; angle/90; budget fraction; spacing/reference_mm"}
        if self.config.critic_candidate_context:
            from .spatial_observations import MAX_ACTIONS
            record["critic_candidate_context"] = {
                "rows": "legal_non_stop_only", "features": "normalized_geometry+masked_sampled_ray_features+inbounds",
                "pooling": "mean_then_max", "count_divisor": MAX_ACTIONS - 1,
                "empty_inventory": "zero_summary_and_count",
                "tool_coverage": "current_legal_inventory_only_not_full_tool_catalog"}
        return record

    @property
    def architecture_hash(self) -> str:
        return "sha256:" + hashlib.sha256(json.dumps(self.architecture_record(),
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def _inputs(self, observation):
        from .spatial_observations import MAX_ACTIONS, SpatialObservation
        if type(observation) is not SpatialObservation:
            raise TypeError("SpatialPolicy requires the explicit SpatialObservation DTO")
        observation.assert_intact()
        images = np.asarray(observation.image_channels)
        coverage = np.asarray(observation.coverage)
        available = np.asarray(observation.channel_available)
        geometry = _real_finite(observation.action_geometry, "action geometry")
        state = _real_finite(observation.state_features, "procedure state").copy()
        affine = _real_finite(observation.affine_ras_mm, "affine")
        mask = np.asarray(observation.action_mask)
        if (images.ndim != 4 or images.shape[0] != 6 or any(v < 2 or v > 64 for v in images.shape[1:])
                or coverage.shape != images.shape or coverage.dtype != np.bool_
                or available.shape != (6,) or available.dtype != np.bool_
                or geometry.shape != (len(observation.action_ids), 16)
                or not 1 <= len(geometry) <= MAX_ACTIONS or state.shape != (10,)
                or mask.shape != (len(geometry),) or mask.dtype != np.bool_
                or not mask[0] or observation.action_ids[0] != "STOP"
                or len(set(observation.action_ids)) != len(observation.action_ids)
                or not np.isfinite(images).all() or np.iscomplexobj(images)):
            raise ValueError("Malformed bounded spatial observation")
        shape = images.shape[1:]
        visible = coverage & available[:, None, None, None]
        safe_images = np.where(visible, images, 0.).astype(np.float32)
        availability = np.broadcast_to(available[:, None, None, None], images.shape)
        volume = np.concatenate((safe_images, visible, availability), axis=0).astype(np.float32)
        # Coordinate conversion uses float64 before casting to model float32.
        fractions = np.linspace(0., 1., self.config.ray_samples)
        points = geometry[:, None, 1:4] + fractions[None, :, None] * (
            geometry[:, None, 4:7] - geometry[:, None, 1:4])
        grid = world_to_sample_grid(points, affine, shape)
        # STOP has no tool ray. It uses a separate head and does not consume
        # features sampled at its placeholder world origin.
        grid[0] = 2.
        inbounds = (np.abs(grid) <= 1.).all(axis=-1).astype(np.float32)
        normalized = geometry.copy()
        normalized[:, 1:4] = world_to_sample_grid(geometry[:, 1:4], affine, shape)[..., ::-1]
        normalized[:, 4:7] = world_to_sample_grid(geometry[:, 4:7], affine, shape)[..., ::-1]
        inverse_linear = np.linalg.inv(affine[:3, :3])
        directions = geometry[:, 7:10] @ inverse_linear.T
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        normalized[:, 7:10] = directions / np.where(norms > 0, norms, 1.)
        normalized[:, 10:14] /= self.config.physical_reference_mm
        normalized[:, 14] /= 90.
        normalized[0] = 0.; normalized[0, 0] = 1.
        if state[1] <= 0 or not 0 <= state[0] <= state[1]:
            raise ValueError("Invalid observed action budget")
        state[0] /= state[1]
        # Explicit maximum horizon remains an input (reference10 actions).
        state[1] /= 10.
        state[3:6] = world_to_sample_grid(state[None, 3:6], affine, shape)[0, ::-1]
        normal = inverse_linear @ state[6:9]
        if np.linalg.norm(normal) == 0:
            raise ValueError("Access normal is zero")
        state[6:9] = normal / np.linalg.norm(normal)
        state[9] /= self.config.physical_reference_mm
        spacing = _real_finite(observation.spacing_mm, "spacing") / self.config.physical_reference_mm
        if spacing.shape != (3,) or np.any(spacing <= 0):
            raise ValueError("Invalid physical spacing")
        if not all(np.isfinite(v).all() for v in (volume, grid, normalized, state, spacing)):
            raise ValueError("Nonfinite normalized observation")
        device = next(self.parameters()).device
        tensor = lambda value: torch.tensor(np.array(value, copy=True), dtype=torch.float32, device=device)
        return (tensor(volume), tensor(grid), tensor(inbounds), tensor(normalized),
                tensor(state), tensor(spacing), torch.tensor(mask.copy(), device=device))

    def forward(self, observation) -> tuple[Tensor, Tensor]:
        volume, grid, inside, geometry, state, spacing, mask = self._inputs(observation)
        features = self.encoder(volume[None])[0]
        spatial = F.adaptive_avg_pool3d(features[None], (2, 2, 2)).flatten()
        context = torch.cat((spatial, state, spacing))
        rays = sample_ray_features(features, grid)
        # Float32 grid casting can round a barely-outside coordinate back onto
        # the boundary. Preserve the authoritative pre-cast coverage decision.
        rays = rays * inside[:, :, None]
        rays = torch.cat((rays, inside[:, :, None]), dim=-1).flatten(1)
        action_inputs = torch.cat((context.expand(len(geometry), -1), geometry, rays), dim=-1)
        non_stop = self.actor(action_inputs).squeeze(-1)
        logits = torch.cat((self.stop(context).reshape(1), non_stop[1:]))
        logits = logits.masked_fill(~mask, -torch.inf)
        critic_context = (torch.cat((context, _candidate_critic_context(geometry, rays, mask)))
                          if self.config.critic_candidate_context else context)
        value = self.critic(critic_context).squeeze(-1)
        if not torch.isfinite(logits[mask]).all() or not torch.isfinite(value):
            raise FloatingPointError("Nonfinite spatial policy output")
        return logits, value

    @torch.no_grad()
    def act(self, observation, *, stochastic=False, generator=None) -> str:
        logits, _ = self(observation)
        index = (torch.multinomial(logits.softmax(-1), 1, generator=generator).item()
                 if stochastic else logits.argmax().item())
        return observation.action_ids[index]


@dataclass(frozen=True)
class SpatialTransition:
    observation: object
    action_id: str
    reward: float
    terminated: bool


def _chosen(observation, action_id: str) -> int:
    if action_id not in observation.action_ids:
        raise ValueError("Action ID is absent from this observation")
    index = observation.action_ids.index(action_id)
    if not observation.action_mask[index]:
        raise ValueError("Chosen action is masked")
    return index


def _generated_loss_binding(policy, learning_context):
    return (learning_context.fingerprint, parameter_hash(policy), id(policy),
            policy.architecture_hash, tuple((name, id(p)) for name, p in policy.named_parameters()))


@generated_development_only
def imitation_loss(policy: SpatialPolicy, samples: Sequence[tuple[object, str]], *,
                   learning_context: GeneratedDevelopmentContext) -> tuple[Tensor, dict]:
    """BC on simulated teacher actions, with teacher cost owned by the runner."""
    if not samples:
        raise ValueError("Imitation batch is empty")
    learning_context.require_observations(observation for observation, _ in samples)
    terms = []
    for observation, action_id in samples:
        index = _chosen(observation, action_id)
        logits, _ = policy(observation)
        terms.append(-logits.log_softmax(-1)[index])
    loss = torch.stack(terms).mean()
    loss._generated_learning_binding = _generated_loss_binding(policy, learning_context)
    return loss, {"kind": "search_action_behavior_cloning", "loss": float(loss.detach()),
                  "loss_forward_calls": len(samples), "supervised_actions": len(samples)}


@generated_development_only
def reinforce_loss(policy: SpatialPolicy, episodes: Sequence[Sequence[SpatialTransition]], *,
                   gamma: float = 1., entropy_weight: float = .01, value_weight: float = .5,
                   learning_context: GeneratedDevelopmentContext) -> tuple[Tensor, dict]:
    """Masked on-policy Monte Carlo policy gradient with a spatial value baseline.

Collect each batch under unchanged current weights. Loss construction repeats
the forward pass for autograd, so those calls must be counted as well as action
collection. This is REINFORCE, not PPO, AWAC, or an off-policy replay algorithm.
"""
    if (not episodes or not 0 <= gamma <= 1 or not np.isfinite([gamma, entropy_weight, value_weight]).all()
            or entropy_weight < 0 or value_weight < 0):
        raise ValueError("Invalid policy-gradient batch/settings")
    learning_context.require_observations(t.observation for ep in episodes for t in ep)
    policy_terms, value_terms, entropies, returns = [], [], [], []
    for episode in episodes:
        if not episode or not episode[-1].terminated or any(t.terminated for t in episode[:-1]):
            raise ValueError("Only complete episodes are eligible for this Monte Carlo update")
        discounted, targets = 0., []
        for transition in reversed(episode):
            if not np.isfinite(transition.reward):
                raise ValueError("Nonfinite transition reward")
            discounted = float(transition.reward) + gamma * discounted
            targets.append(discounted)
        targets.reverse(); returns.append(targets[0])
        ep_actor, ep_value, ep_entropy = [], [], []
        for step, (transition, target) in enumerate(zip(episode, targets)):
            index = _chosen(transition.observation, transition.action_id)
            logits, value = policy(transition.observation)
            distribution = torch.distributions.Categorical(logits=logits)
            log_prob = distribution.log_prob(torch.tensor(index, device=logits.device))
            ep_actor.append(-(gamma ** step) * log_prob * (value.new_tensor(target) - value.detach()))
            ep_value.append((value - target).square())
            ep_entropy.append(distribution.entropy())
        # Episodic start-state return has a SUM of score-function terms per
        # trajectory. Averaging by its realized length biases toward STOP/short
        # trajectories. gamma**step is required for discounted start returns.
        policy_terms.append(torch.stack(ep_actor).sum())
        value_terms.append(torch.stack(ep_value).mean())
        entropies.append(torch.stack(ep_entropy).mean())
    actor = torch.stack(policy_terms).mean()
    value = torch.stack(value_terms).mean()
    entropy = torch.stack(entropies).mean()
    loss = actor + value_weight * value - entropy_weight * entropy
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite spatial learning loss")
    loss._generated_learning_binding = _generated_loss_binding(policy, learning_context)
    return loss, {"kind": "masked_reinforce_spatial_value_v1", "loss": float(loss.detach()),
        "policy_loss": float(actor.detach()), "value_loss": float(value.detach()),
        "entropy": float(entropy.detach()), "mean_return": float(np.mean(returns)),
        "actor_reduction": "mean_episodes_sum_discounted_score_terms",
        "regularizer_reduction": "mean_episodes_mean_actions",
        "completed_episodes": len(episodes), "loss_forward_calls": sum(map(len, episodes))}


@generated_development_only
def gradient_step(policy: SpatialPolicy, optimizer, loss: Tensor, *, max_norm: float = 5.,
                  learning_context: GeneratedDevelopmentContext) -> dict:
    """One actual optimizer step; report submodule gradients before global clip."""
    if not np.isfinite(max_norm) or max_norm <= 0 or not torch.isfinite(loss):
        raise ValueError("Invalid loss or clipping bound")
    before = parameter_hash(policy)
    if getattr(loss, "_generated_learning_binding", None) != _generated_loss_binding(policy, learning_context):
        raise ValueError("Gradient requires a matching admitted loss and unchanged policy parameters")
    members = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(members) != len(set(members)) or set(members) != {id(p) for p in policy.parameters()}:
        raise ValueError("Optimizer must own exactly the admitted policy parameters")
    loss._generated_learning_binding = None  # One backward/update attempt per admitted loss.
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norms = {}
    for name in ("encoder", "actor", "stop", "critic"):
        gradients = [p.grad.detach().square().sum() for p in getattr(policy, name).parameters() if p.grad is not None]
        norms[name + "_gradient_norm_before_clip"] = float(torch.stack(gradients).sum().sqrt()) if gradients else 0.
    total = float(nn.utils.clip_grad_norm_(policy.parameters(), max_norm, error_if_nonfinite=True))
    clipped = float(torch.stack([p.grad.detach().square().sum() for p in policy.parameters()
                                 if p.grad is not None]).sum().sqrt())
    optimizer.step()
    after = parameter_hash(policy)
    return {"optimizer_steps": 1, "gradient_norm_before_clip": total,
            "gradient_norm_after_clip": clipped, **norms,
            "initial_parameter_hash": before, "updated_parameter_hash": after,
            "parameters_changed": before != after}


def parameter_hash(policy: SpatialPolicy) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(policy.state_dict().items()):
        array = parameter.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return "sha256:" + digest.hexdigest()
