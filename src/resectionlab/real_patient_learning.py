"""Metadata-only roles and batch bookkeeping for the fixed BTC development set.

These helpers neither open images nor run an optimizer. SELECT is never a
population-gradient source. Patient identity stays in bookkeeping rather than
the actor observation. A new cohort requires a separately reviewed protocol;
unopened transfer patients cannot be admitted by editing a caller's role label.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

COHORT_SHA256 = "962d964e1d71427f3625cdbebc0f7e4759e5d2345d8f95cb211ed45810ed2985"
TRAIN_SUBJECTS = ("sub-PAT05", "sub-PAT16", "sub-PAT20", "sub-PAT22", "sub-PAT25", "sub-PAT28")
SELECT_SUBJECTS = ("sub-PAT26", "sub-PAT27")
TRAIN_GROUPS = tuple("BTC:" + subject for subject in TRAIN_SUBJECTS)
SELECT_GROUPS = tuple("BTC:" + subject for subject in SELECT_SUBJECTS)
_SOURCE_ROLES = {
    **{subject: "previously_consulted_training_and_method_development" for subject in TRAIN_SUBJECTS
       if subject not in {"sub-PAT22", "sub-PAT25"}},
    "sub-PAT22": "population_training", "sub-PAT25": "population_training",
    **{subject: "checkpoint_selection_development" for subject in SELECT_SUBJECTS},
}


@dataclass(frozen=True)
class PatientRole:
    subject: str
    patient_group: str
    role: str


def require_development_role(cohort: Mapping, subject: str, *, role: str) -> PatientRole:
    """Validate canonical identity and locked role before a caller opens a case.

Call read_development_cohort to establish the pinned byte identity first. This
pure validator also checks every permitted member, so omissions and duplicate
identities cannot silently turn into outcome-selected training subsets.
"""
    allowed = TRAIN_SUBJECTS if role == "TRAIN" else SELECT_SUBJECTS if role == "SELECT" else ()
    if subject not in allowed:
        raise ValueError("Only the six TRAIN or two SELECT patients have permitted development roles")
    if (cohort.get("kind") != "prospective_BTC_spatial_development_cohort"
            or cohort.get("source", {}).get("accession") != "ds001226"
            or cohort.get("source", {}).get("release") != "5.0.1"):
        raise ValueError("Expected the fixed preoperative BTC development cohort")
    records = [*cohort.get("existing_development_records", ()), *cohort.get("candidates", ())]
    groups = {}
    canonical_groups = set()
    for row in records:
        identity = row.get("subject")
        group = row.get("patient_group")
        if identity in groups or group in canonical_groups:
            raise ValueError("Duplicate patient identity in development cohort")
        groups[identity] = row
        canonical_groups.add(group)
    for identity, expected in _SOURCE_ROLES.items():
        row = groups.get(identity, {})
        if (row.get("patient_group") != "BTC:" + identity or row.get("visit") != "ses-preop"
                or row.get("development_role") != expected or row.get("outer_role") != "development"
                or row.get("eligible_for_external_final") is not False):
            raise ValueError("Missing or changed locked BTC patient role")
    return PatientRole(subject, "BTC:" + subject, role)


def read_development_cohort(path: str | Path) -> dict:
    """Read only the original metadata manifest, checking bytes before parsing."""
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != COHORT_SHA256:
        raise ValueError("Original frozen BTC development cohort bytes changed")
    cohort = json.loads(payload)
    require_development_role(cohort, TRAIN_SUBJECTS[0], role="TRAIN")
    return cohort


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")


def patient_uniform_bc_indices(sample_counts: Mapping[str, int], *, batch_size: int,
                               generator: np.random.Generator) -> tuple[tuple[str, int], ...]:
    """Draw TRAIN patient uniformly, then one state from that patient's teacher.

Empty/missing patients are failures to resolve, not permission to omit them.
STOP-only teachers remain represented. Returned indices can feed the existing
imitation_loss without changing its loss reduction or loading patient data.
"""
    _positive_integer(batch_size, "batch_size")
    if set(sample_counts) != set(TRAIN_GROUPS):
        raise ValueError("BC requires exactly the six declared TRAIN patient groups")
    for count in sample_counts.values():
        _positive_integer(count, "Per-patient teacher-state count")
    selected = []
    for _ in range(batch_size):
        group = TRAIN_GROUPS[int(generator.integers(len(TRAIN_GROUPS)))]
        selected.append((group, int(generator.integers(sample_counts[group]))))
    return tuple(selected)


def training_patient_schedule(*, rounds: int, generator: np.random.Generator) -> tuple[str, ...]:
    """A shuffled full-six schedule; each patient contributes once per round."""
    _positive_integer(rounds, "rounds")
    return tuple(TRAIN_GROUPS[int(index)] for _ in range(rounds)
                 for index in generator.permutation(len(TRAIN_GROUPS)))


@dataclass(frozen=True)
class PatientEpisode:
    patient_group: str
    behavior_parameter_hash: str
    transitions: tuple


def validate_on_policy_batch(episodes: Sequence[PatientEpisode], *, parameter_hash: str) -> dict:
    """Require complete TRAIN trajectories collected under one unchanged policy.

Validation is independent of reward sign and episode length. The unchanged
reinforce_loss must receive full transition sequences: its actor reduction is
mean over episodes of the sum of score-function terms, never mean over steps.
This helper checks provenance consistency, not whether rollout code actually
measured its parameter hash; the caller must measure before and after rollout.
"""
    if (not isinstance(parameter_hash, str) or not parameter_hash.startswith("sha256:")
            or len(parameter_hash) != 71 or any(c not in "0123456789abcdef" for c in parameter_hash[7:])):
        raise ValueError("A measured spatial-policy parameter hash is required")
    if not episodes:
        raise ValueError("An on-policy batch requires complete episodes")
    counts = Counter()
    transitions = 0
    for episode in episodes:
        if not isinstance(episode, PatientEpisode) or episode.patient_group not in TRAIN_GROUPS:
            raise ValueError("Population gradients may use only declared TRAIN patients")
        if episode.behavior_parameter_hash != parameter_hash:
            raise ValueError("On-policy batch mixes changed or stale behavior weights")
        trajectory = episode.transitions
        if (not trajectory or not trajectory[-1].terminated or any(row.terminated for row in trajectory[:-1])
                or any(not math.isfinite(row.reward) for row in trajectory)):
            raise ValueError("Only complete finite-reward trajectories may enter REINFORCE")
        counts[episode.patient_group] += 1
        transitions += len(trajectory)
    return {"behavior_parameter_hash": parameter_hash, "completed_episodes": len(episodes),
            "transitions": transitions, "episodes_by_patient": {group: counts[group] for group in TRAIN_GROUPS},
            "actor_reduction": "mean_episodes_sum_discounted_score_terms"}
