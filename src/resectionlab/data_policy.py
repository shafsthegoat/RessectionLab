"""Execution exclusions required by the October 6 real-observation policy.

This module does not certify data as eligible. It closes known incompatible
execution paths while their code and saved results remain available for reading.
Future component/offline learners need independently admitted source records;
there is deliberately no flag that converts simulator experience into recordings.
"""
from __future__ import annotations

from functools import wraps
from typing import Callable, NoReturn, ParamSpec, TypeVar

POLICY_VERSION = "real-observations-only-2026-10-06"

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
        "This historical learner uses simulated transitions or generated teacher "
        "labels. New learning requires eligible recorded observations, actions, "
        "next observations, timing, censoring and observed endpoint support."
    ),
    "SYNTHETIC_MODEL_INELIGIBLE": (
        "The pinned SynthStrip weights were trained on synthetic images. Those "
        "weights and their derived masks are excluded from the current pipeline."
    ),
    "WEIGHT_LINEAGE_UNVERIFIED": (
        "This legacy checkpoint loader has no verified real-only training "
        "ancestry contract. File integrity alone does not establish eligibility."
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
