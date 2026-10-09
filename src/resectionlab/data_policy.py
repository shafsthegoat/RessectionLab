"""Explicit source-use admission, including October 8 generated development.

This module does not certify data as eligible. It closes known incompatible
execution paths while their code and saved results remain available for reading.
The narrow generated-native development contract does not admit historical
models, patients or clinical use, or convert simulator experience into recordings.
"""
from __future__ import annotations

from functools import wraps
from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Callable, NoReturn, ParamSpec, TypeVar

POLICY_VERSION = "explicit-source-use-2026-10-08"
GENERATED_DEVELOPMENT_SCOPE = "native-opening-generated-development-v1"

# Published SynthStrip v1 assets already present in the historical evidence.
SYNTHETIC_MODEL_SHA256 = frozenset({
    "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33",
    "62bf01137c45b5f0cc04d59dbaed5b9ac138b3f25b766c062a7c1a0d696ecb28",
})

LEGACY_OPERATION_EXCLUSIONS = {
    "createSyntheticCase": "SYNTHETIC_CASE_DISABLED",
    "nativeTraining": "RECORDED_EXPERIENCE_REQUIRED",
    "trainPatient": "RECORDED_EXPERIENCE_REQUIRED",
    "importStructuralEvidence": "SYNTHETIC_MODEL_INELIGIBLE",
}

REASONS = {
    "RECORDED_EXPERIENCE_REQUIRED": (
        "This historical implicit learning pathway remains closed. The spatial "
        "learner can use an explicit October 8 generated-development contract; "
        "other recorded or generated workflows require separate source admission."
    ),
    "SYNTHETIC_MODEL_INELIGIBLE": (
        "The pinned SynthStrip weights were trained on synthetic images. Those "
        "weights and their derived masks are excluded from the current pipeline."
    ),
    "WEIGHT_LINEAGE_UNVERIFIED": (
        "This model dependency has no verified training ancestry contract "
        "for this use. File integrity alone does not establish eligibility."
    ),
    "GENERATED_POLICY_INELIGIBLE": (
        "This frozen policy was trained on simulated experience or generated "
        "teacher labels. Its saved results remain historical; its weights cannot "
        "enter a new compliant learning or evaluation run."
    ),
    "SYNTHETIC_CASE_DISABLED": (
        "Generated patient cases are excluded from training, evaluation and "
        "demonstrations. Open an eligible acquired case instead."
    ),
}


@dataclass(frozen=True)
class GeneratedDevelopmentContext:
    """Bounded research declaration; source authenticity is verified by the runner.

    This deliberately admits only the analytical native opening development
    study, not arbitrary patients or legacy model ancestry. Its source IDs join
    the complete immutable observation DTO; rewards stay outside actor inputs.
    """
    declaration_sha256: str
    source_ids: tuple[str, ...]
    decision_model_hash: str
    scope: str = GENERATED_DEVELOPMENT_SCOPE
    transition_lineage: str = "simulator_generated"
    real_patient_count: int = 0
    max_steps: int = 2

    def __post_init__(self):
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        self.assert_intact()

    def assert_intact(self):
        def valid_hash(value):
            return (isinstance(value, str) and len(value.removeprefix("sha256:")) == 64
                    and all(c in "0123456789abcdef" for c in value.removeprefix("sha256:")))
        if (self.scope != GENERATED_DEVELOPMENT_SCOPE or self.transition_lineage != "simulator_generated"
                or type(self.real_patient_count) is not int or self.real_patient_count != 0
                or type(self.max_steps) is not int or self.max_steps != 2
                or type(self.source_ids) is not tuple or len(self.source_ids) != 1
                or not all(valid_hash(v) for v in (*self.source_ids, self.decision_model_hash,
                                                  self.declaration_sha256))):
            raise ValueError("Only the explicitly bound native opening generated-development contract is admitted")

    @property
    def fingerprint(self):
        self.assert_intact()
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    def require_observations(self, observations):
        from .spatial_observations import SpatialObservation
        self.assert_intact()
        for observation in observations:
            if type(observation) is not SpatialObservation:
                raise TypeError("Generated learning requires the validated spatial observation DTO")
            observation.assert_intact()
            if (observation.track != "synthetic_scan" or observation.source_id not in self.source_ids
                    or observation.state_features[1] != self.max_steps):
                raise ValueError("Generated-development source/track differs from the declared fixture")

    def require_task(self, task):
        """Runner boundary: observations cannot independently prove a reward model."""
        from .native_spatial_task import NativeSpatialTask
        self.assert_intact()
        if type(task) is not NativeSpatialTask:
            raise TypeError("The generated opening contract requires the exact native task")
        task._assert_frozen()
        if (task.case.source_hash not in self.source_ids or task.case.track != "synthetic_scan"
                or task.decision_model_hash != self.decision_model_hash or task.max_steps != self.max_steps):
            raise ValueError("Generated task source/reward/horizon differs from the declared model")


def generated_development_only(function):
    """Opt in only the three spatial loss/update APIs; default refusal is retained."""
    @wraps(function)
    def admitted(*args, **kwargs):
        context = kwargs.get("learning_context")
        if type(context) is not GeneratedDevelopmentContext:
            raise DataPolicyError("RECORDED_EXPERIENCE_REQUIRED", function.__qualname__)
        context.assert_intact()
        return function(*args, **kwargs)
    return admitted


class DataPolicyError(ValueError):
    """A prohibited operation, rejected before its body or failure logger runs."""

    def __init__(self, code: str, operation: str):
        self.code = code
        self.operation = operation
        self.policy_version = POLICY_VERSION
        super().__init__(f"{code}: {operation}: {REASONS[code]}")


def require_admitted_model(model_sha256: str | None, operation: str) -> None:
    """Reject a model dependency at consumption, including cached mask use.

    No model ancestry has yet been admitted. A missing model hash is merely
    outside this exclusion check; it does not certify acquisition/annotation
    lineage. Bundle loading and historical metadata inspection remain available.
    """
    if model_sha256 is None:
        return
    digest = model_sha256.removeprefix("sha256:")
    reason = ("SYNTHETIC_MODEL_INELIGIBLE" if digest in SYNTHETIC_MODEL_SHA256
              else "WEIGHT_LINEAGE_UNVERIFIED")
    raise DataPolicyError(reason, operation)


P = ParamSpec("P")
R = TypeVar("R")


def historical_only(code: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Preserve an incompatible implementation but refuse new execution.

    Put this outermost when an older decorator creates failure/run artifacts.
    Reading historical JSON or performing independent algebra is unaffected.
    """
    if code not in REASONS:
        raise ValueError(f"Unknown real-observation policy exclusion: {code}")

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def refuse(*args: P.args, **kwargs: P.kwargs) -> NoReturn:
            raise DataPolicyError(code, function.__qualname__)

        return refuse

    return decorate
