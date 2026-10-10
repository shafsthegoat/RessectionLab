"""Fixed, named policy input units; physical simulator observations stay raw.

The 648 mm³ divisor is the declared procedural support volume, a reference
unit rather than a bound on patient actions or spatial surrogate integrals.
There are no learned statistics, clipping, centering or return transforms.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


NATIVE_ACTION_NAMES = (
    "stop", "target_benefit", "normal_volume", "motor_evidence", "language_evidence",
    "insertion_distance", "tip_radius", "shaft_radius", "tool_change", "depth",
    "remaining_target_fraction", "adjacent_target_fraction", "partial_normal_contact_volume",
    "partial_motor_contact_surrogate", "partial_language_contact_surrogate",
)
NATIVE_STATE_NAMES = (
    "remaining_target_fraction", "action_budget_fraction_used", "removed_tissue_fraction",
    "nominal_severed_edge_fraction", "motor_evidence_available", "language_evidence_available",
)
FEATURE_UNITS_DIVISORS = (1., 648., 648., 648., 648., 120., 2.25, 1.10, 1., 1., 1., 1., 648., 648., 648.)


@dataclass(frozen=True)
class PolicyInputProfile:
    profile_id: str
    version: str
    action_feature_names: tuple[str, ...]
    action_divisors: tuple[float, ...]
    state_feature_names: tuple[str, ...]
    state_divisors: tuple[float, ...]

    def to_dict(self) -> dict:
        return {"profile_id": self.profile_id, "version": self.version,
            "action_feature_names": list(self.action_feature_names), "action_divisors": list(self.action_divisors),
            "state_feature_names": list(self.state_feature_names), "state_divisors": list(self.state_divisors),
            "centering": False, "clipping": False, "running_statistics": False,
            "return_divisor": 1.0, "critic_output_divisor": 1.0}

    @property
    def fingerprint(self) -> str:
        return "sha256:" + hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def policy_input_profile(profile_id: str = "RAW", *, action_features: int = 15,
                         state_features: int = 6) -> PolicyInputProfile:
    """Resolve a registry entry; callers cannot supply outcome-fitted divisors."""
    if profile_id not in ("RAW", "FEATURE_UNITS"):
        raise ValueError("Unknown policy input profile")
    if min(action_features, state_features) < 1:
        raise ValueError("Policy feature dimensions must be positive")
    if (action_features, state_features) == (15, 6):
        return PolicyInputProfile(profile_id, "native-fixed-action-units-v1", NATIVE_ACTION_NAMES,
            FEATURE_UNITS_DIVISORS if profile_id == "FEATURE_UNITS" else (1.,) * 15,
            NATIVE_STATE_NAMES, (1.,) * 6)
    if profile_id != "RAW":
        raise ValueError("FEATURE_UNITS requires the exact native 15+6 feature schema")
    # Preserve identity policies for older analytic/tiny fixtures. These names
    # make no claim to native feature semantics and cannot load a native profile.
    return PolicyInputProfile("RAW", "generic-identity-v1",
        tuple(f"action_{index}" for index in range(action_features)), (1.,) * action_features,
        tuple(f"state_{index}" for index in range(state_features)), (1.,) * state_features)
