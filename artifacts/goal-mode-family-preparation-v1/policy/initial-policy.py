"""Forward-only public-goal/mode actor for the shared native contact episode.

The actor receives a detached public observation and its independently captured
context, never a task, reference target, action reward or simulator property.
This module adds no training admission, checkpoint loader or alternate simulator.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .core import semantic_digest
from .public_surface_contact import (CONTEXT_VERSION, OBJECTIVE_VERSION,
    OBSERVATION_VERSION, SurfaceContactDevelopmentContext, SurfaceContactObservation)
from .spatial_policy import (GRID_ROUNDOFF_BAND, SpatialPolicy, SpatialPolicyConfig,
    _candidate_critic_context, parameter_hash, sample_ray_features)

POLICY_VERSION = "public-goal-mode-spatial-ray-conv-policy-v1"
CHECKPOINT_VERSION = "public-goal-mode-spatial-checkpoint-v1"
MODE_ORDER = ("stop", "aspirate", "probe")


class GoalModeSpatialPolicy(nn.Module):
    """New weights/identity, with the existing physical spatial-input adapter.

    Action names and source IDs bind records but never become neural features.
    The public objective currently has fixed cost/completion coefficients; a
    changed objective schema needs a new model contract, not a silent relabel.
    """
    def __init__(self, config: SpatialPolicyConfig | None = None):
        super().__init__()
        self.config = config or SpatialPolicyConfig(critic_candidate_context=True)
        if type(self.config) is not SpatialPolicyConfig:
            raise TypeError("Use the exact bounded shared SpatialPolicyConfig")
        a, b = self.config.encoder_channels
        self.encoder = nn.Sequential(nn.Conv3d(24, a, 3, padding=1), nn.ReLU(),
                                     nn.Conv3d(a, b, 3, padding=1), nn.ReLU())
        context_count = b * 8 + 10 + 3
        row_count = 19 + self.config.ray_samples * (b + 1)
        width = self.config.hidden_features
        self.actor = nn.Sequential(nn.Linear(context_count + row_count, width), nn.ReLU(),
                                   nn.Linear(width, 1))
        self.stop = nn.Sequential(nn.Linear(context_count, width), nn.ReLU(), nn.Linear(width, 1))
        critic_count = context_count + (2 * row_count + 1 if self.config.critic_candidate_context else 0)
        self.critic = nn.Sequential(nn.Linear(critic_count, width), nn.ReLU(), nn.Linear(width, 1))

    def architecture_record(self) -> dict:
        return {"version": POLICY_VERSION, "config": asdict(self.config),
            "checkpoint_version": CHECKPOINT_VERSION,
            "observation_version": OBSERVATION_VERSION, "context_version": CONTEXT_VERSION,
            "objective_version": OBJECTIVE_VERSION, "objective_coefficients": "fixed_by_public_objective_v1",
            "image_layout": "CXYZ", "channels": ["structural_intensity", "nominal_tissue",
                "nominal_target", "observed_cavity", "nominal_motor", "nominal_language",
                "public_goal", "observed_probe_contact"],
            "image_input": "8_masked_values+8_coverage+8_availability",
            "added_channel_coverage": "structural_coverage_and_availability",
            "action_geometry": "shared_normalized_16_columns+mode_one_hot_3",
            "mode_order": list(MODE_ORDER), "action_id_or_tool_id_embedding": False,
            "input_adapter": "SpatialPolicy._inputs_exact_base_DTO_after_goal_context_validation",
            "grid_sample_order": "ZYX", "align_corners": True,
            "padding": "zeros_with_inbounds_flags", "grid_roundoff_band": GRID_ROUNDOFF_BAND,
            "spatial_pool": [2, 2, 2], "return_transform": "none",
            "critic_candidate_context": "legal_non_stop_geometry_modes_rays_mean_max_count"
                if self.config.critic_candidate_context else "none",
            "inference_ties": "argmax_first_in_inventory_STOP_index0",
            "legacy_aspiration_checkpoint_compatible": False,
            "deployment_scope": "bound_generated_public_surface_contact_only"}

    @property
    def architecture_hash(self) -> str:
        return semantic_digest(self.architecture_record())

    def checkpoint_identity(self) -> dict:
        """Identity declaration only: no checkpoint serialization or load occurs."""
        return {"version": CHECKPOINT_VERSION, "architecture_hash": self.architecture_hash,
            "parameter_hash": parameter_hash(self), "policy_version": POLICY_VERSION,
            "observation_version": OBSERVATION_VERSION, "context_version": CONTEXT_VERSION,
            "objective_version": OBJECTIVE_VERSION,
            "training_lineage": "not_attested_by_this_forward_only_adapter"}

    def _inputs(self, observation, *, context):
        # The task captures context BEFORE detached consumer invocation. A DTO
        # that is internally consistent but moved to another crop/frame fails.
        if type(context) is not SurfaceContactDevelopmentContext:
            raise TypeError("Goal/mode forward requires the exact public goal/grid context")
        if type(observation) is not SurfaceContactObservation:
            raise TypeError("Goal/mode forward requires the exact public goal/mode DTO")
        context.require_observation(observation)
        # Reuse normalization only, without constructing/loading a legacy model.
        volume, grid, inside, geometry, state, spacing, mask = SpatialPolicy._inputs(
            self, observation.base.base)
        base = observation.base.base
        coverage = np.asarray(base.coverage[0]) & bool(base.channel_available[0])
        extra = np.stack((observation.public_goal_grid,
                          observation.base.observed_probe_contact_grid))
        covered = np.broadcast_to(coverage, extra.shape)
        available = np.ones(extra.shape, dtype=np.float32)
        tensor = lambda value: torch.tensor(np.array(value, copy=True), dtype=torch.float32,
                                            device=volume.device)
        volume = torch.cat((volume[:6], tensor(extra & covered), volume[6:12],
                            tensor(covered), volume[12:18], tensor(available)))
        modes = np.zeros((len(observation.action_ids), len(MODE_ORDER)), dtype=np.float32)
        modes[np.arange(len(modes)), [MODE_ORDER.index(m) for m in observation.action_modes]] = 1.
        geometry = torch.cat((geometry, tensor(modes)), dim=-1)
        return volume, grid, inside, geometry, state, spacing, mask

    def forward(self, observation, *, context: SurfaceContactDevelopmentContext) -> tuple[Tensor, Tensor]:
        volume, grid, inside, geometry, state, spacing, mask = self._inputs(observation, context=context)
        features = self.encoder(volume[None])[0]
        spatial = F.adaptive_avg_pool3d(features[None], (2, 2, 2)).flatten()
        global_context = torch.cat((spatial, state, spacing))
        rays = sample_ray_features(features, grid) * inside[:, :, None]
        rays = torch.cat((rays, inside[:, :, None]), dim=-1).flatten(1)
        rows = torch.cat((global_context.expand(len(geometry), -1), geometry, rays), dim=-1)
        scores = self.actor(rows).squeeze(-1)
        logits = torch.cat((self.stop(global_context).reshape(1), scores[1:])).masked_fill(~mask, -torch.inf)
        critic_context = (torch.cat((global_context, _candidate_critic_context(geometry, rays, mask)))
                          if self.config.critic_candidate_context else global_context)
        value = self.critic(critic_context).squeeze(-1)
        if not torch.isfinite(logits[mask]).all() or not torch.isfinite(value):
            raise FloatingPointError("Nonfinite public goal/mode policy output")
        return logits, value

    @torch.no_grad()
    def act(self, observation, *, context: SurfaceContactDevelopmentContext,
            stochastic: bool = False, generator=None) -> str:
        logits, _ = self(observation, context=context)
        index = (torch.multinomial(logits.softmax(-1), 1, generator=generator).item()
                 if stochastic else logits.argmax().item())
        return observation.action_ids[index]
