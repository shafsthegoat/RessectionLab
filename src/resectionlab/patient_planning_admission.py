"""Source-bound ReMIND planning context over the existing public native world.

No file, image, model or evaluator loader lives here. The owning runner verifies
the QC receipt's provenance before supplying it: a digest and a `pass` assertion
do not perform or authenticate anatomical QC. Generated controls remain generated.
Private ventricular annotations are intentionally absent from every argument.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .core import array_digest, freeze_json, semantic_digest, thaw_json
from .native_spatial_task import MAX_NATIVE_SPATIAL_STEPS, NativeSpatialCase, NativeSpatialTask
from .spatial_observations import SpatialObservation

VERSION = "remind-public-annotation-assisted-native-v1"
COHORT_SHA256 = "326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05"
QC_SCOPE = "public_T1_cerebrum_tumor_native_geometry_coverage_and_intended_use"
SOURCE_FIELDS = frozenset({"version", "evidence_domain", "subject", "patient_group",
    "cohort_sha256", "t1_source_sha256", "support_source_sha256", "target_source_sha256",
    "image_array_hash", "support_array_hash", "target_array_hash", "affine_array_hash",
    "source_hash", "availability_basis", "acquired_at", "annotation_available_at",
    "support_semantics", "target_semantics"})
QC_FIELDS = frozenset({"version", "evidence_domain", "subject", "public_source_binding_hash",
    "status", "scope", "source_linkage_checked", "frame_geometry_checked",
    "coverage_checked", "annotation_meaning_checked", "public_support_assumption",
    "native_domain_fully_covered", "hypothetical_access_assumption", "evidence_record_sha256"})
PROTOCOL_FIELDS = frozenset({"version", "scope", "subject", "role", "evidence_domain",
    "public_source_binding_hash", "qc_receipt_hash", "max_steps", "max_optimizer_updates",
    "search", "max_native_previews", "max_policy_forwards", "worker_seconds",
    "memory_bytes", "threads", "initialization", "private_reference_used",
    "clinical_claim", "split_changes", "runtime_release_sha256", "learning_protocol_hash"})
SELECT_SUBJECTS = ("ReMIND-013", "ReMIND-037")
SELECT_INITIALIZATION = "frozen_TRAIN_checkpoint_reload"
CHECKPOINT_LINEAGE_FIELDS = frozenset({"version", "checkpoint_sha256", "method",
    "training_release_sha256", "learning_protocol_hash", "initial_parameter_hash",
    "parameter_hash", "architecture_hash", "completed_updates", "training_context_hashes",
    "public_target_context_variant", "optimizer_updates_on_SELECT"})


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _digest(value):
    _need(isinstance(value, str), "digest_required")
    token = value.removeprefix("sha256:")
    _need(len(token) == 64 and all(c in "0123456789abcdef" for c in token), "digest_required")
    return token


def _fields(value, expected, reason):
    _need(isinstance(value, Mapping) and set(value) == expected, reason)
    return freeze_json(value)


def _same_hash(a, b):
    return _digest(a) == _digest(b)


def validate_select_checkpoint_lineage(value, *, learning_protocol_hash=None):
    """Validate declared lineage only; the inference loader authenticates weights.

The owning runner binds the original TRAIN release/terminal receipts before
supplying external pins. This metadata validator does not prove their provenance.
The exact64 endpoint also requires the balanced sequential protocol at the
inference loader; a digest-only lineage cannot authenticate that objective.
"""
    lineage = _fields(value, CHECKPOINT_LINEAGE_FIELDS, "exact_SELECT_checkpoint_lineage_required")
    _need(lineage["version"] == "frozen-TRAIN-checkpoint-lineage-v1"
          and lineage["method"] in {"IL", "RL"}
          and type(lineage["completed_updates"]) is int
          and (1 <= lineage["completed_updates"] <= 32 or lineage["completed_updates"] == 64)
          and type(lineage["optimizer_updates_on_SELECT"]) is int
          and lineage["optimizer_updates_on_SELECT"] == 0,
          "frozen_TRAIN_endpoint_zero_SELECT_updates_required")
    for key in ("checkpoint_sha256", "training_release_sha256"):
        _need(type(lineage[key]) is str and len(lineage[key]) == 64, "raw_checkpoint_or_release_sha_required")
        _digest(lineage[key])
    for key in ("learning_protocol_hash", "initial_parameter_hash", "parameter_hash", "architecture_hash"):
        _need(type(lineage[key]) is str and lineage[key].startswith("sha256:"), "semantic_checkpoint_hash_required")
        _digest(lineage[key])
    groups = {"ReMIND:"+suffix for suffix in ("008", "010", "020", "025")}
    _need(isinstance(lineage["training_context_hashes"], Mapping)
          and set(lineage["training_context_hashes"]) == groups, "original_four_TRAIN_context_pins_required")
    for digest in lineage["training_context_hashes"].values():
        _need(type(digest) is str and digest.startswith("sha256:"), "semantic_TRAIN_context_hash_required")
        _digest(digest)
    from .public_target_context import VERSION as TARGET_CONTEXT
    _need(lineage["public_target_context_variant"] in (None, TARGET_CONTEXT), "known_checkpoint_observation_variant_required")
    if learning_protocol_hash is not None:
        _need(_same_hash(lineage["learning_protocol_hash"], learning_protocol_hash), "checkpoint_learning_protocol_mismatch")
    return lineage


def _public_observation_binding(observation):
    """Static evidence and access; committed cavity/actions stay trace-bound."""
    indices = [0, 1, 2, 4, 5]
    return semantic_digest({"images": array_digest(observation.image_channels[indices]),
        "coverage": array_digest(observation.coverage),
        "available": array_digest(observation.channel_available),
        "affine": array_digest(observation.affine_ras_mm),
        "spacing": array_digest(observation.spacing_mm),
        "access": array_digest(observation.state_features[3:]),
        "provenance": observation.channel_provenance,
        **({} if observation.public_target_context is None else {
            'public_target_context':observation.public_target_context.static_fingerprint})})


@dataclass(frozen=True)
class PatientPlanningContext:
    """Exact task construction receipt; not a hostile detached-object sandbox."""
    _record: Mapping
    _seal: str

    def record(self):
        _need(semantic_digest(self._record) == self._seal, "patient_context_changed")
        return thaw_json(self._record)

    @property
    def fingerprint(self):
        self.record()
        return self._seal

    @property
    def patient_group(self):
        return self.record()["patient_group"]

    @property
    def role(self):
        return self.record()["role"]

    @property
    def max_steps(self):
        return self.record()["max_steps"]

    @property
    def protocol_sha256(self):
        return self.record()["protocol_sha256"]

    def require_training(self):
        record = self.record()
        _need(record["role"] == "TRAIN" and record["cohort_sha256"] == COHORT_SHA256
              and record["scope"] == "patient_native_planning_experiment"
              and record["max_optimizer_updates"] > 0,
              "only_declared_TRAIN_gradient_budget_admitted")

    def require_task(self, task):
        record = self.record()
        _need(type(task) is NativeSpatialTask, "exact_native_task_required")
        task._assert_frozen()
        _need(task.case.source_hash == record["source_hash"]
              and task.decision_model_hash == record["decision_model_hash"]
              and task.max_steps == record["max_steps"]
              and task.case.track == record["observation_track"]
              and task.tool_modes is None and task.case.nominal_target is not None
              and np.array_equal(task.case.reference_target, task.case.nominal_target)
              and not np.any(task._config.hard_exclusion),
              "public_patient_task_or_private_reference_changed")

    def require_observations(self, observations):
        record = self.record()
        for observation in observations:
            _need(type(observation) is SpatialObservation, "exact_spatial_observation_required")
            observation.assert_intact()
            _need(observation.source_id == record["source_hash"]
                  and observation.track == record["observation_track"]
                  and float(observation.state_features[1]) == record["max_steps"]
                  and _public_observation_binding(observation) == record["public_observation_binding"],
                  "patient_observation_source_or_horizon_changed")


def make_patient_planning_task(case: NativeSpatialCase, *, cohort_bytes: bytes,
        source_binding: Mapping, qc_receipt: Mapping, protocol: Mapping):
    """Construct the declared public task, returning (task, exact context).

The source annotation-assisted target and support are disclosed inputs. The
reference_target slot must already equal the nominal target; it never carries
ventricular evaluation labels. Full source coverage is required for this first
native adapter, which has no partial-support coverage argument. Missing coverage
must be refused by upstream QC, not filled with background. Runtime budgets are
bound here and enforced by the separately supervised caller.
"""
    _need(type(case) is NativeSpatialCase, "exact_public_native_source_required")
    _need(type(cohort_bytes) is bytes and hashlib.sha256(cohort_bytes).hexdigest() == COHORT_SHA256,
          "original_ReMIND_cohort_bytes_required")
    cohort = json.loads(cohort_bytes)
    source = _fields(source_binding, SOURCE_FIELDS, "exact_public_source_fields_required")
    qc = _fields(qc_receipt, QC_FIELDS, "exact_public_QC_fields_required")
    checkpoint_reload = isinstance(protocol, Mapping) and protocol.get("initialization") == SELECT_INITIALIZATION
    plan = _fields(protocol, PROTOCOL_FIELDS | ({"checkpoint_lineage"} if checkpoint_reload else set()),
        "exact_preflight_protocol_fields_required")
    domain = source["evidence_domain"]
    _need(domain in {"acquired_patient", "generated_interface_control"}, "explicit_evidence_domain_required")
    _need(source["version"] == qc["version"] == plan["version"] == VERSION
          and qc["evidence_domain"] == plan["evidence_domain"] == domain,
          "evidence_domain_or_version_mismatch")
    matches = [row for row in cohort["members"] if row["subject"] == source["subject"]]
    _need(len(matches) == 1, "unknown_ReMIND_person")
    member = matches[0]
    _need(source["patient_group"] == member["patient_group"]
          and source["subject"] == qc["subject"] == plan["subject"]
          and plan["role"] == member["role"] and member["role"] in {"TRAIN", "SELECT"}
          and member["diagnosis_eligibility_stratum"] == "strict_three_label_glioma"
          and _same_hash(source["cohort_sha256"], COHORT_SHA256),
          "frozen_TRAIN_or_SELECT_role_required")
    _need(source["availability_basis"] == "retrospective_annotation_assisted_source_preop"
          and source["acquired_at"] is None and source["annotation_available_at"] is None,
          "unknown_dates_preserved_retrospective_annotation_assistance_required")
    semantics = (("source_automatic_Brainlab_cerebrum_annotation", "source_manual_whole_tumor_annotation")
                 if domain == "acquired_patient" else
                 ("generated_support_interface_control", "generated_target_interface_control"))
    _need((source["support_semantics"], source["target_semantics"]) == semantics,
          "exact_disclosed_annotation_semantics_required")
    for field in ("t1_source_sha256", "support_source_sha256", "target_source_sha256"):
        _digest(source[field])
    case.assert_intact()
    _need(case.track == ("annotation_assisted" if domain == "acquired_patient" else "synthetic_scan")
          and case.support_source_kind == case.target_source_kind ==
              ("supplied_annotation" if domain == "acquired_patient" else "derived_from_scan")
          and case.nominal_target is not None and np.any(case.nominal_target)
          and np.array_equal(case.reference_target, case.nominal_target)
          and case.proposal_mode in {"fixed_lattice", "nominal_cavity_v1"}
          and not np.any(case._native_config.hard_exclusion),
          "public_source_target_support_or_track_required")
    for name, array in (("image_array_hash", case.structural_intensity),
                        ("support_array_hash", case.observed_support),
                        ("target_array_hash", case.nominal_target),
                        ("affine_array_hash", case.affine_ras_mm)):
        _need(_same_hash(source[name], array_digest(array)), "public_array_binding_mismatch")
    _need(source["source_hash"] == case.source_hash, "public_source_hash_mismatch")
    source_hash = semantic_digest(source)
    _need(qc["public_source_binding_hash"] == plan["public_source_binding_hash"] == source_hash
          and plan["qc_receipt_hash"] == semantic_digest(qc), "source_QC_protocol_join_required")
    _need(qc["status"] == "pass" and qc["scope"] == QC_SCOPE
          and all(qc[key] is True for key in ("source_linkage_checked", "frame_geometry_checked",
              "coverage_checked", "annotation_meaning_checked", "public_support_assumption",
              "native_domain_fully_covered", "hypothetical_access_assumption")),
          "intended_use_QC_not_header_only_required")
    _digest(qc["evidence_record_sha256"])
    lineage = None
    if checkpoint_reload:
        _need(source["subject"] in SELECT_SUBJECTS and member["role"] == "SELECT"
              and type(plan["max_optimizer_updates"]) is int and plan["max_optimizer_updates"] == 0,
              "checkpoint_reload_requires_fixed_SELECT_zero_gradients")
        lineage = validate_select_checkpoint_lineage(plan["checkpoint_lineage"],
            learning_protocol_hash=plan["learning_protocol_hash"])
        _need(case.public_target_context_variant == lineage["public_target_context_variant"],
              "checkpoint_observation_variant_mismatch")
    _need(plan["scope"] == "patient_native_planning_experiment"
          and type(plan["max_steps"]) is int and 1 <= plan["max_steps"] <= MAX_NATIVE_SPATIAL_STEPS
          and type(plan["max_optimizer_updates"]) is int and plan["max_optimizer_updates"] >= 0
          and (member["role"] == "TRAIN" or plan["max_optimizer_updates"] == 0)
          and type(plan["threads"]) is int and plan["threads"] == 1
          and (checkpoint_reload or plan["initialization"] == "fresh_seeded_shared_initialization")
          and plan["private_reference_used"] is False and plan["clinical_claim"] is False
          and plan["split_changes"] is False, "exact_nonclinical_role_and_budget_contract_required")
    for key in ("max_native_previews", "max_policy_forwards", "worker_seconds", "memory_bytes"):
        _need(type(plan[key]) is int and plan[key] > 0, "positive_integer_runtime_budgets_required")
    search = _fields(plan["search"], {"max_calls", "beam_width", "seconds"}, "exact_search_budget_required")
    _need(all(type(search[k]) is int and search[k] > 0 for k in search), "positive_integer_search_budgets_required")
    _digest(plan["runtime_release_sha256"])
    _digest(plan["learning_protocol_hash"])
    task = NativeSpatialTask(case, max_steps=plan["max_steps"])
    record = freeze_json({"version": VERSION, "scope": plan["scope"], "subject": source["subject"],
        **({} if lineage is None else {"initialization": SELECT_INITIALIZATION, "checkpoint_lineage": lineage}),
        **({} if case.public_target_context_variant is None else {
            'public_target_context_variant':case.public_target_context_variant}),
        "patient_group": source["patient_group"], "role": member["role"], "evidence_domain": domain,
        "real_patient_count": 1 if domain == "acquired_patient" else 0,
        "cohort_sha256": COHORT_SHA256, "source_hash": case.source_hash,
        "decision_model_hash": task.decision_model_hash, "max_steps": task.max_steps,
        "public_observation_binding": _public_observation_binding(task.observation()),
        "observation_track": case.track, "protocol_sha256": semantic_digest(plan),
        "learning_protocol_hash": plan["learning_protocol_hash"],
        "public_source_binding_hash": source_hash, "qc_receipt_hash": semantic_digest(qc),
        "max_optimizer_updates": plan["max_optimizer_updates"], "budgets": {key:plan[key] for key in
            ("search", "max_native_previews", "max_policy_forwards", "worker_seconds", "memory_bytes", "threads")},
        "private_reference_in_task": False, "ventricular_evidence": "unavailable_to_actor_and_search",
        "actions": "simulated_not_recorded", "clinical_claim": False})
    context = PatientPlanningContext(record, semantic_digest(record))
    context.require_task(task)
    return task, context
