"""Exact-input, observation-only PAT05 checkpoint compatibility diagnostic.

This is an additive exception for two frozen forwards, not anatomy admission.
No task/engine is constructed, saved proposals are not recertified, and neither
an action nor a loss is executed. Existing model/support/learning gates remain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

VERSION = "pat05-forward-diagnostic-v1"
READINESS = "artifacts/btc-train-transfer-readiness-v1/contract.json"
COHORT = "manifests/experiments/btc-spatial-development-cohort-v1.json"
HISTORICAL = "artifacts/pat05-real-geometric-learning-v1/"
RL = "artifacts/native-opening-rl-capacity-v1/"
ARCHITECTURE = "sha256:a3b6740ab188f01016610c566e2009c57ba422c8f9be24adca014f695ae9c917"
OBSERVATION = "sha256:645770594d7988980781df8324fb76ed347f47ff8d7452ca7d032d31efc93f90"
SOURCE = "sha256:27b51658c4445dc0eb12c0ecef2a2709e1b5873bb5bc8a3f9034e8195f4b9267"
METADATA_SHA256 = {
    READINESS: "d7130daa2a989ada77298bd754403195f6caebae4b2483db16e759ad05c0f1e5",
    COHORT: "962d964e1d71427f3625cdbebc0f7e4759e5d2345d8f95cb211ed45810ed2985",
    HISTORICAL + "declaration-input.json": "d69c9b6cf1f0f329d698c3857bca13efaa4a4a9a7b424208711affd4ac8480f4",
    HISTORICAL + "receipt.json": "fdc575e6695a7f949f65de93e7165f7833b514fcd193094a24ac3e0f94e7ee77",
    HISTORICAL + "initial_policy.json": "9ed4ba0c05f1823efbedd1ea0f61200d8d9ca30d98d06c29acf30702668ebe19",
    HISTORICAL + "output-sha256.json": "2df29d18c484dd5120129cc6915eb5fad5544654b9e824a801bbbc25e1025219",
    RL + "summary.json": "f71c9d5e2c7e28692fefed904e57240c23f0fb00b7725a1fb83086bf1f4c78e8",
    RL + "independent-verification.json": "2489aff1b8a783f1fd6ad97a1085abb4b54aff30b44c337deb42a654d5afc95f",
    RL + "declaration-input.json": "14df1a1fbb9bb8187d97bc7dcf99cacf923a9e5e9f4182693de0e1d01a5e1285",
    RL + "output-sha256.json": "d378524c205838c28f09b2bc02e2a1e08dadc9a25acd63a0170043afc3aeea3d",
}
CHECKPOINTS = {
    "initial": {"path": RL + "RL-000.pt", "file_sha256": "433c95336d11a5b3d2e5b98c1d53c2b989b63ae0e277e3b43f835c4cbc2a8257",
        "parameter_hash": "sha256:2fa97c0e730db09371786d5ba903ffdf467b3cc7149a111c182d6a7499cdd9b8", "updates": 0},
    "RL256": {"path": RL + "RL-256.pt", "file_sha256": "1d391665f66cdd0c1fa1db150261b853820a9d30d62fc20c99cf021cf0229fbd",
        "parameter_hash": "sha256:f52e14097ea6a35436eaf30f0ece5658e24987e289ca30ff6a529816545b5721", "updates": 256},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def checked_bytes(root, relative, expected):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Input is outside the declared repository")
    payload = path.read_bytes()
    if digest_bytes(payload) != expected:
        raise ValueError("Exact input bytes changed: " + relative)
    return payload


def read_authorities(root):
    """Metadata only: authenticate immutable historical authorities before decode."""
    records = {path: json.loads(checked_bytes(root, path, expected)) for path, expected in METADATA_SHA256.items()}
    from .real_patient_learning import require_development_role
    require_development_role(records[COHORT], "sub-PAT05", role="TRAIN")
    old = records[HISTORICAL + "declaration-input.json"]
    member = old["member"]
    readiness = next(r for r in records[READINESS]["train_records"] if r["subject"] == "sub-PAT05")
    binding = readiness["bindings"]
    if member["subject"] != "sub-PAT05" or member["role"] != "TRAIN" or readiness["role"] != "TRAIN":
        raise ValueError("Only historical PAT05 TRAIN is admitted for this diagnostic")
    for key in ("case_bundle", "case_bundle_sha256", "case_semantic_hash"):
        if member[key] != binding[key]:
            raise ValueError("Readiness and historical patient identity disagree")
    initial = records[HISTORICAL + "initial_policy.json"]["decisions"][0]
    receipt = records[HISTORICAL + "receipt.json"]
    if (initial["observation_hash"] != OBSERVATION or receipt["initial_task_metrics"]["source_hash"] != SOURCE
            or len(initial["action_ids"]) != 71 or old["settings"]["max_steps"] != 3):
        raise ValueError("Historical initial observation identity differs")
    for name, checkpoint in CHECKPOINTS.items():
        index = records[RL + "output-sha256.json"]
        if index[Path(checkpoint["path"]).name] != checkpoint["file_sha256"]:
            raise ValueError("Checkpoint differs from the original execution index")
    summary = records[RL + "summary.json"]
    if (summary["status"] != "complete_and_independently_verified"
            or summary["fixed_final_argmax"]["parameter_hash"] != CHECKPOINTS["RL256"]["parameter_hash"]):
        raise ValueError("Committed fixed-final RL evidence differs")
    return records


def _rebuild_saved_observation(source, inventory, first_decision, *, max_steps):
    """Pure DTO reconstruction from already recorded geometry; no recertification.

    The caller authenticates all records. This small helper also permits tiny
    analytical software tests; only Pat05ForwardContext permits model execution.
    """
    from .spatial_observations import SpatialAction, ObservedProcedureState, build_spatial_observation
    if inventory.get("complete") is not True or first_decision["step"] != 0:
        raise ValueError("An exact completed historical initial inventory is required")
    accepted = [row for row in inventory["emitted"] if row["feasible"] is True]
    ids = ["STOP", *(row["action_id"] for row in accepted)]
    if ids != first_decision["action_ids"] or inventory["accepted_count"] != len(accepted):
        raise ValueError("Recorded accepted proposal ordering/count differs")
    tools = {tool.tool_id: tool for tool in source.tools}
    actions = [SpatialAction("STOP")]
    for row in accepted:
        actions.append(SpatialAction(row["action_id"], row["entry_mm"], row["tip_mm"], tools[row["tool_id"]]))
    observation = build_spatial_observation(source.spatial_inputs(np.zeros(source.observed_support.shape, bool)),
        actions, ObservedProcedureState(source.access, 0, max_steps, None))
    if (list(observation.action_mask) != first_decision["action_mask"]
            or observation.fingerprint != first_decision["observation_hash"]):
        raise ValueError("Reconstructed full historical DTO fingerprint/mask differs; no fallback")
    return observation


def reconstruct_pat05(root, authorities):
    """Decode one bound real source, admitting its estimate solely as DTO content."""
    from .imaging import load_case
    from .geometry import AccessWindow, ToolGeometry
    from .native_spatial_task import NativeSpatialCase
    from .structural_evidence import validate_support_assumption
    from .core import thaw_json
    # Reauthenticate mutable caller records against the pinned bytes before any
    # array load. This is not a caller-editable patient/support allowlist.
    authentic = read_authorities(root)
    if canonical(authorities) != canonical(authentic):
        raise ValueError("Historical authority records changed")
    old = authentic[HISTORICAL + "declaration-input.json"]
    member = old["member"]
    binding = authentic[READINESS]["train_records"][0]["bindings"]
    checked_bytes(root, member["case_bundle"], member["case_bundle_sha256"])
    case = load_case(Path(root) / member["case_bundle"])
    checked_bytes(root, member["case_bundle"], member["case_bundle_sha256"])
    if case.semantic_hash != binding["case_semantic_hash"] or case.planning_hash != binding["planning_hash"]:
        raise ValueError("Decoded PAT05 source or planning metadata differs")
    item = case.structural_evidence[binding["evidence_id"]]
    item.assert_matches(case)
    if item.provenance != "estimated" or item.review_status != "review_required":
        raise ValueError("Historical support review/estimate state changed")
    for key in ("evidence_hash", "source_image_hash", "source_frame_hash", "mask_hash", "model_sha256", "run_sha256"):
        if getattr(item, key) != binding[key]:
            raise ValueError("Exact estimated support identity changed: " + key)
    ack = member["research_support_acknowledgment"]
    validate_support_assumption(case, item.mask, ack)
    metrics = authentic[HISTORICAL + "receipt.json"]["initial_task_metrics"]
    if canonical(metrics["support_provenance"]["acknowledgment"]) != canonical(ack):
        raise ValueError("Historical support acknowledgment differs")
    affine = np.asarray(case.affine)
    if case.frame == "LPS+":
        affine = np.diag([-1., -1., 1., 1.]) @ affine
    elif case.frame != "RAS+":
        raise ValueError("Unknown source physical coordinate convention")
    # Source compartments are supplied annotations, not a model's inference.
    supplied = np.logical_or.reduce(tuple(case.compartments.values()))
    source = NativeSpatialCase(case.mri, item.mask, supplied, affine,
        AccessWindow(**member["access"]), tuple(ToolGeometry(**r) for r in old["tools"]),
        track="annotation_assisted", support_source_kind="derived_from_scan",
        support_derivation="explicitly_acknowledged_unreviewed_model_support; cortical access unverified",
        nominal_target=supplied, target_source_kind="supplied_annotation",
        target_derivation="supplied preoperative source compartments; annotation-assisted target model",
        support_provenance=metrics["support_provenance"], **old["adapter_options"])
    if (source.source_hash != SOURCE
            or canonical(thaw_json(source._grid_record)) != canonical(metrics["native_grid_reconciliation"])
            or canonical(thaw_json(source._normalization_record)) != canonical(metrics["intensity_normalization"])
            or list(source._crop_origin) != metrics["crop"]["origin_voxels"]
            or list(source._crop_shape) != metrics["crop"]["shape"]):
        raise ValueError("Historical source/full grid/normalization/crop differs")
    observation = _rebuild_saved_observation(source, authentic[HISTORICAL + "receipt.json"]["initial_inventory"],
        authentic[HISTORICAL + "initial_policy.json"]["decisions"][0], max_steps=3)
    return observation, {"source_hash": source.source_hash, "observation_hash": observation.fingerprint,
        "native_grid_reconciliation": thaw_json(source._grid_record),
        "normalization": thaw_json(source._normalization_record), "crop": metrics["crop"],
        "support_provenance": metrics["support_provenance"], "source_shape": list(case.mri.shape),
        "supplied_annotation_cells_full_source": int(supplied.sum()),
        "supplied_annotation_cells_in_crop": int(np.count_nonzero(observation.image_channels[2])),
        "annotation": {"source": "supplied BTC preoperative source compartment",
            "availability_timestamp": None, "expert_review_status": "unknown", "learned_segmentation": False},
        "unknown_evidence": ["motor", "language", "vessels", "deformation", "cutting_forces"],
        "cavity_lineage": "hypothetical empty initial simulator state; not an observed operative cavity",
        "proposal_lineage": "70 historical simulator-feasibility-filtered proposals; simulator-derived information, not demonstrated scan-only deployment input; no new certification or action execution",
        "anatomy_admission": False, "clinical_admission": False}


@dataclass(frozen=True)
class Pat05ForwardContext:
    """One-use-per-checkpoint exact observation admission; no training authority."""
    declaration_sha256: str
    _attempted: set = field(default_factory=set, init=False, repr=False)
    _forward_attempted: set = field(default_factory=set, init=False, repr=False)
    _completed: set = field(default_factory=set, init=False, repr=False)

    def __post_init__(self):
        if (not isinstance(self.declaration_sha256, str) or len(self.declaration_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.declaration_sha256)):
            raise ValueError("Exact raw declaration SHA256 is required")

    def require_observation(self, observation):
        from .spatial_observations import SpatialObservation
        if type(observation) is not SpatialObservation:
            raise TypeError("Only the immutable spatial DTO is permitted")
        observation.assert_intact()
        if (observation.fingerprint != OBSERVATION or observation.source_id != SOURCE
                or observation.track != "annotation_assisted" or tuple(observation.image_channels.shape) != (6,64,64,64)
                or len(observation.action_ids) != 71 or observation.state_features[1] != 3
                or observation.state_features[0] != 0 or np.any(observation.image_channels[3])):
            raise ValueError("Only the exact historical initial PAT05 observation is admitted")

    def snapshot(self):
        return {"declaration_sha256": self.declaration_sha256,
            "checkpoint_attempts": len(self._attempted),
            "forward_attempts": len(self._forward_attempted), "completed_forwards": len(self._completed),
            "attempted_checkpoints": sorted(self._attempted), "completed_checkpoints": sorted(self._completed),
            "native_previews": 0, "executed_actions": 0, "optimizer_updates": 0,
            "scope": "annotation-assisted fixed-input compatibility; no patient-performance assessment"}

    def forward_checkpoint(self, root, name, observation):
        """Authenticate bytes and metadata, then exactly one frozen forward."""
        import torch
        from .spatial_policy import SpatialPolicy, SpatialPolicyConfig, parameter_hash
        self.require_observation(observation)
        if name not in CHECKPOINTS or name in self._attempted:
            raise ValueError("Only one attempt per declared checkpoint; no retries")
        self._attempted.add(name)
        spec = CHECKPOINTS[name]
        payload = checked_bytes(root, spec["path"], spec["file_sha256"])
        # This narrow allowlist handles known NumPy scalar metadata in the two
        # exact local artifacts. It does not relax any general checkpoint loader.
        unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(io.BytesIO(payload)))
        if unsafe - {"numpy._core.multiarray.scalar", "numpy.dtype"}:
            raise ValueError("Unexpected checkpoint serialization globals")
        with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype, type(np.dtype("f8"))]):
            checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if (checkpoint["updates"] != spec["updates"] or checkpoint["parameter_hash"] != spec["parameter_hash"]
                or checkpoint["initial_parameter_hash"] != CHECKPOINTS["initial"]["parameter_hash"]
                or checkpoint["context"]["declaration_sha256"] != METADATA_SHA256[RL + "declaration-input.json"]):
            raise ValueError("Frozen generated-training checkpoint ancestry differs")
        with torch.random.fork_rng(devices=[]):
            model = SpatialPolicy(SpatialPolicyConfig(**checkpoint["architecture"]["config"]))
        model.load_state_dict(checkpoint["policy"], strict=True)
        model.eval().requires_grad_(False)
        if model.architecture_hash != ARCHITECTURE or parameter_hash(model) != spec["parameter_hash"]:
            raise ValueError("Checkpoint tensor/architecture identity differs")
        if canonical(model.architecture_record()) != canonical(checkpoint["architecture"]):
            raise ValueError("Checkpoint architecture metadata differs")
        self.require_observation(observation)
        began = time.perf_counter()
        with torch.inference_mode():
            self._forward_attempted.add(name)
            logits, value = model(observation)
            probabilities = logits.softmax(-1)
        seconds = time.perf_counter() - began
        self.require_observation(observation)
        if (parameter_hash(model) != spec["parameter_hash"] or model.architecture_hash != ARCHITECTURE
                or any(p.grad is not None or p.requires_grad for p in model.parameters())):
            raise RuntimeError("Frozen forward changed weights/architecture or acquired gradients")
        checked_bytes(root, spec["path"], spec["file_sha256"])
        self._completed.add(name)
        return {"status": "complete", **spec, "architecture_hash": model.architecture_hash,
            "parameter_count": sum(p.numel() for p in model.parameters()), "forward_seconds": seconds,
            "observation_hash": observation.fingerprint, "action_ids": list(observation.action_ids),
            "action_mask": observation.action_mask.tolist(), "logits": logits.tolist(),
            "probabilities": probabilities.tolist(), "value_uncalibrated": value.item(),
            "entropy": float(-(probabilities * probabilities.clamp_min(1e-30).log()).sum()),
            "highest_ranked_id_not_executed": observation.action_ids[int(logits.argmax())],
            "tie_rule": "argmax_first_in_saved_inventory; STOP at index0", "weights_unchanged": True,
            "training_context_preserved": checkpoint["context"]}


def describe_observation(observation):
    """Read-only ranges/coverage; no policy forward or inferred missing evidence."""
    from .spatial_policy import world_to_sample_grid
    from .spatial_observations import CHANNEL_NAMES, ACTION_GEOMETRY_NAMES, STATE_FEATURE_NAMES
    observation.assert_intact()
    geometry = observation.action_geometry
    points = geometry[1:, None, 1:4] + np.linspace(0,1,5)[None,:,None] * (geometry[1:,None,4:7]-geometry[1:,None,1:4])
    grid = world_to_sample_grid(points, observation.affine_ras_mm, observation.image_channels.shape[1:])
    channels = []
    for i, name in enumerate(CHANNEL_NAMES):
        covered = observation.coverage[i] & observation.channel_available[i]
        values = observation.image_channels[i][covered]
        channels.append({"name": name, "available": bool(observation.channel_available[i]),
            "covered_voxels": int(covered.sum()), "coverage_fraction": float(covered.mean()),
            "min": float(values.min()) if values.size else None, "max": float(values.max()) if values.size else None})
    return {"channels": channels, "geometry_ranges": {name: [float(geometry[1:,i].min()),float(geometry[1:,i].max())]
            for i,name in enumerate(ACTION_GEOMETRY_NAMES)},
        "state": dict(zip(STATE_FEATURE_NAMES, observation.state_features.tolist())),
        "ray_samples_inside_crop": int((np.abs(grid)<=1).all(axis=-1).sum()), "ray_samples_total": int(grid.size//3),
        "shift": {"generated_shape": [9,9,7], "real_crop_shape": [64,64,64],
            "generated_initial_action_count": 5, "real_initial_action_count": 71,
            "generated_horizon": 2, "real_historical_horizon": 3,
            "generated_access_radius_mm": 2.4, "real_access_radius_mm": 6.,
            "generated_working_lengths_mm": [2.2,12.], "real_working_lengths_mm": [120.,120.],
            "generated_tip_radii_mm": [.9,2.25], "real_tip_radii_mm": [1.25,2.25],
            "generated_tip_lengths_mm": [.75,3.], "real_tip_lengths_mm": [2.,3.],
            "intensity": "generated discrete0/.2/.8 versus support-percentile-normalized real T1",
            "generalization_or_performance_established": False}}
