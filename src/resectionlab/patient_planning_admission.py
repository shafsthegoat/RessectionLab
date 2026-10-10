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
from .native_spatial_task import (MAX_NATIVE_SPATIAL_STEPS, NativeSpatialCase, NativeSpatialTask,
    SUPPLIED_TUMOR_UNION_OCCUPANCY, DERIVED_OCCUPANCY_SOURCE_KIND)
from .spatial_observations import SpatialObservation

VERSION = "remind-public-annotation-assisted-native-v1"
COHORT_SHA256 = "326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05"
QC_SCOPE = "public_T1_cerebrum_tumor_native_geometry_coverage_and_intended_use"
PARTIAL_DOMAIN_TRAIN_SUBJECTS = ("ReMIND-002", "ReMIND-015", "ReMIND-018", "ReMIND-045")
PARTIAL_DOMAIN_UNION_OCCUPANCY = "cerebrum_plus_supplied_tumor_with_preserved_partial_source_domain"
PARTIAL_DOMAIN_QC_SCOPE = "public_T1_cerebrum_tumor_preserved_source_domains_search_only"
PARTIAL_DOMAIN_SOURCE_FIELDS = frozenset({"support_domain_source_sha256", "support_domain_binary_hash", "source_and_simulated_domains"})
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
        if task.case.support_domain is not None:
            _need(record.get("occupancy_condition") == PARTIAL_DOMAIN_UNION_OCCUPANCY
                  and record.get("source_and_simulated_domains") == thaw_json(task.case._domain_record),
                  "partial_source_domain_context_changed")

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
ventricular evaluation labels. The default requires complete support coverage.
The explicit fixed-four TRAIN search-only condition preserves observed Ds and
the declared D=Ds union T interaction domain; unknown support is not background.
Runtime budgets are bound here and enforced by the separately supervised caller.
"""
    _need(type(case) is NativeSpatialCase, "exact_public_native_source_required")
    _need(type(cohort_bytes) is bytes and hashlib.sha256(cohort_bytes).hexdigest() == COHORT_SHA256,
          "original_ReMIND_cohort_bytes_required")
    cohort = json.loads(cohort_bytes)
    derived_occupancy = case.occupancy_source_support is not None
    partial_domain = case.support_domain is not None
    _need(not partial_domain or derived_occupancy, "partial_domain_requires_explicit_S_union_T_assumption")
    source = _fields(source_binding, SOURCE_FIELDS | ({"occupancy_derivation",
        "target_domain_source_sha256", "target_domain_binary_hash"} if derived_occupancy else set())
        | (PARTIAL_DOMAIN_SOURCE_FIELDS if partial_domain else set()),
        "exact_public_source_fields_required")
    qc = _fields(qc_receipt, QC_FIELDS | ({"derived_occupancy_anatomically_validated"}
        if derived_occupancy else set()) | ({"partial_source_domain_preserved"} if partial_domain else set()),
        "exact_public_QC_fields_required")
    checkpoint_reload = isinstance(protocol, Mapping) and protocol.get("initialization") == SELECT_INITIALIZATION
    occupancy_learning = isinstance(protocol, Mapping) and protocol.get("occupancy_learning_protocol") is not None
    _need(not occupancy_learning or (derived_occupancy and not partial_domain),
          "union_learning_requires_original_full_coverage_condition")
    plan = _fields(protocol, PROTOCOL_FIELDS | ({"checkpoint_lineage"} if checkpoint_reload else set())
        | ({"occupancy_learning_protocol"} if occupancy_learning else set())
        | ({"occupancy_condition"} if derived_occupancy else set()),
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
    if derived_occupancy:
        semantics = ("derived_simulated_S_union_T_occupancy_assumption", semantics[1])
        expected_condition = PARTIAL_DOMAIN_UNION_OCCUPANCY if partial_domain else SUPPLIED_TUMOR_UNION_OCCUPANCY
        expected_subjects = PARTIAL_DOMAIN_TRAIN_SUBJECTS if partial_domain else ("ReMIND-008", "ReMIND-010", "ReMIND-020", "ReMIND-025")
        _need(domain == "acquired_patient" and member["role"] == "TRAIN"
              and source["subject"] in expected_subjects
              and not checkpoint_reload and plan["occupancy_condition"] == expected_condition
              and (occupancy_learning or (
                  plan["initialization"] == "public_world_search_only"
                  and type(plan["max_optimizer_updates"]) is int and plan["max_optimizer_updates"] == 0
                  and type(plan["max_policy_forwards"]) is int and plan["max_policy_forwards"] == 0)),
              "derived_occupancy_requires_fixed_TRAIN_search_only_or_explicit_learning")
        if occupancy_learning:
            from .patient_planning_cohort_spec import validate_union_obstruction_learning
            learning = validate_union_obstruction_learning(plan["occupancy_learning_protocol"],
                learning_protocol_hash=plan["learning_protocol_hash"], proposal_config=case.proposal_config,
                max_steps=plan["max_steps"])
            _need(plan["initialization"] == "fresh_seeded_shared_initialization"
                  and case.proposal_mode == "nominal_cavity_v1"
                  and case.public_target_context_variant == learning["public_target_context_variant"]
                  and type(plan["max_optimizer_updates"]) is int
                  and plan["max_optimizer_updates"] >= 2*learning["updates_per_method"]
                  and type(plan["max_policy_forwards"]) is int and plan["max_policy_forwards"] > 0,
                  "union_learning_requires_fresh_declared_public_context_and_positive_budgets")
        _need(qc["derived_occupancy_anatomically_validated"] is False
              and semantic_digest(source["occupancy_derivation"]) == semantic_digest(case._occupancy_derivation)
              and _same_hash(case._occupancy_derivation["source_support_hash"],
                  case.support_provenance.get("raw_support_array_hash"))
              and case.public_target_domain is not None
              and _same_hash(source["target_domain_binary_hash"], array_digest(case.public_target_domain))
              and case.support_provenance.get("occupancy_condition") == expected_condition
              and case.support_provenance.get("occupancy_unchanged") is
                  (case._occupancy_derivation["added_region_positive_voxels"] == 0)
              and case.support_provenance.get("target_unchanged") is True,
              "derived_occupancy_provenance_or_QC_scope_mismatch")
        for key, provenance_key in (("support_source_sha256", "supplied_support_file_sha256"),
                ("target_source_sha256", "supplied_target_file_sha256"),
                ("target_domain_source_sha256", "target_domain_file_sha256"),
                ("target_domain_binary_hash", "target_domain_binary_hash")):
            _need(_same_hash(source[key], case.support_provenance.get(provenance_key)),
                  "unchanged_public_annotation_provenance_required")
        if partial_domain:
            _need(qc["partial_source_domain_preserved"] is True
                  and qc["native_domain_fully_covered"] is False
                  and case.public_target_context_variant is not None
                  and semantic_digest(source["source_and_simulated_domains"]) == semantic_digest(case._domain_record)
                  and _same_hash(source["support_domain_binary_hash"], array_digest(case.support_domain))
                  and _same_hash(source["support_domain_source_sha256"], case.support_provenance.get("support_domain_file_sha256"))
                  and _same_hash(source["support_domain_binary_hash"], case.support_provenance.get("support_domain_binary_hash"))
                  and np.array_equal(case._native_config.interaction_domain, case.support_domain | (case.nominal_target > 0))
                  and not np.any(case.occupancy_source_support & ~case.support_domain),
                  "unchanged_source_Ds_full_T_and_explicit_simulated_D_required")
    _need((source["support_semantics"], source["target_semantics"]) == semantics,
          "exact_disclosed_annotation_semantics_required")
    for field in ("t1_source_sha256", "support_source_sha256", "target_source_sha256"):
        _digest(source[field])
    case.assert_intact()
    _need(case.track == ("annotation_assisted" if domain == "acquired_patient" else "synthetic_scan")
          and case.support_source_kind == (DERIVED_OCCUPANCY_SOURCE_KIND if derived_occupancy else
              "supplied_annotation" if domain == "acquired_patient" else "derived_from_scan")
          and case.target_source_kind == ("supplied_annotation" if domain == "acquired_patient" else "derived_from_scan")
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
    _need(qc["status"] == "pass" and qc["scope"] == (PARTIAL_DOMAIN_QC_SCOPE if partial_domain else QC_SCOPE)
          and all(qc[key] is True for key in ("source_linkage_checked", "frame_geometry_checked",
              "coverage_checked", "annotation_meaning_checked", "public_support_assumption",
              "hypothetical_access_assumption"))
          and qc["native_domain_fully_covered"] is (not partial_domain),
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
          and (checkpoint_reload or derived_occupancy or plan["initialization"] == "fresh_seeded_shared_initialization")
          and plan["private_reference_used"] is False and plan["clinical_claim"] is False
          and plan["split_changes"] is False, "exact_nonclinical_role_and_budget_contract_required")
    for key in ("max_native_previews", "max_policy_forwards", "worker_seconds", "memory_bytes"):
        _need(type(plan[key]) is int and (plan[key] == 0 if derived_occupancy and not occupancy_learning and key == "max_policy_forwards"
              else plan[key] > 0), "positive_integer_runtime_budgets_required")
    search = _fields(plan["search"], {"max_calls", "beam_width", "seconds"}, "exact_search_budget_required")
    _need(all(type(search[k]) is int and search[k] > 0 for k in search), "positive_integer_search_budgets_required")
    _digest(plan["runtime_release_sha256"])
    _digest(plan["learning_protocol_hash"])
    task = NativeSpatialTask(case, max_steps=plan["max_steps"])
    record = freeze_json({"version": VERSION, "scope": plan["scope"], "subject": source["subject"],
        **({} if lineage is None else {"initialization": SELECT_INITIALIZATION, "checkpoint_lineage": lineage}),
        **({"occupancy_condition": plan["occupancy_condition"],
            "occupancy_derivation": case._occupancy_derivation,
            "execution_kind": "fixed_four_TRAIN_union_obstruction_learning_v1" if occupancy_learning else "search_only_no_policy",
            "derived_occupancy_anatomically_validated": False,
            "policy_comparison_permitted": occupancy_learning,
            **({"comparison_scope": "same_declared_union_world_fixed_four_TRAIN_only"} if occupancy_learning else {})}
           if derived_occupancy else {}),
        **({"source_and_simulated_domains": case._domain_record,
            "source_domain_fully_covered": False} if partial_domain else {}),
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
