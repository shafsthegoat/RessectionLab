"""Local JSONL sidecar for Electron. No GUI imports, sockets, or arbitrary code.

Electron's main process owns file dialogs and grants only chosen paths. Binary
descriptors are internal to that process; the renderer receives opaque asset
IDs. Every geometry operation names a cached immutable case version.
"""

from __future__ import annotations

from .data_policy import DataPolicyError, LEGACY_OPERATION_EXCLUSIONS, historical_only

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import hashlib
import hmac
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Callable

import numpy as np

from .core import CaseData, freeze_json, thaw_json
from .geometry import AccessWindow, GENERIC_TOOLS
from .native_resection import NATIVE_GENERIC_TOOLS
from .imaging import create_synthetic_case, inspect_nifti, load_case, load_nifti_case, read_case_artifacts, save_case
from .planning import SearchConfig, generate_candidate_routes

PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 1024 * 1024
MAX_ARRAY_BYTES = 512 * 1024 * 1024
MAX_CASE_BYTES = 1024 * 1024 * 1024
MAX_WORKSPACE_BYTES = 512 * 1024
MAX_PENDING = 8
MAX_PRIOR_PROPOSALS = 16
MAX_AXIS_INSPECTION_VOXELS = 16_000_000
MAX_AXIS_INSPECTION_METADATA_BYTES = 256 * 1024
MAX_AXIS_INSPECTION_RESULT_BYTES = 2 * 1024 * 1024
OPERATIONS = frozenset({"ping", "executeDevelopmentEpisode", "loadCase", "importNifti", "importDisplaySeries", "importStructuralEvidence", "importPriorProposals", "saveCase", "generateRoutes", "generateNativeRoutes", "inspectRefinement", "inspectAxisPlanning", "inspectObservedLandmarkUpdate", "cancel", "inspectEvidence", "createSyntheticCase", "nativeTraining", "trainPatient", "listRuns", "replayTraining", "evaluateCandidate", "exportCandidate", "shutdown"})
MAX_RUN_JSON_BYTES = 32 * 1024 * 1024
RESEARCH_TOOLS = GENERIC_TOOLS + NATIVE_GENERIC_TOOLS
RUN_INTEGRITY_FILES = {"checkpointSha256": "checkpoint.pt", "contractSha256": "contract.json",
                       "nativeRequestSha256": "native-request.json", "reportSha256": "native-refinement.json",
                       "functionalFreezeSha256": "native-functional-freeze.json",
                       "functionalReportSha256": "native-functional-events.json"}


class BridgeError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _string(value: Any, name: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or "\x00" in value:
        raise BridgeError("INVALID_ARGUMENT", f"{name} must be a nonempty bounded string")
    return value


def _keys(args: dict, allowed: set[str]) -> None:
    extra = set(args) - allowed
    if extra:
        raise BridgeError("INVALID_ARGUMENT", "Unsupported argument(s): " + ", ".join(sorted(extra)))


def _require_json_budget(value: Any, limit: int, code: str, message: str) -> None:
    """Bound a new inspection payload without assembling a second full JSON string."""
    size = 0
    for chunk in json.JSONEncoder(separators=(",", ":"), allow_nan=False).iterencode(value):
        size += len(chunk.encode("utf-8"))
        if size > limit:
            raise BridgeError(code, message)


def _require_axis_output_binding(binding: dict, *, case: CaseData, access: AccessWindow,
                                 preset: dict, expected_binding: str | None) -> None:
    """Join a returned report to server inputs, not merely to its own hashes."""
    from .critical_evidence import resolve_critical_evidence
    expected_critical = resolve_critical_evidence(case).planning_binding
    if binding.get("critical_evidence") != expected_critical:
        raise BridgeError("AXIS_SOURCE_BINDING_MISMATCH", "Returned critical evidence differs from this case")
    requested = {"center_mm": access.center_mm.tolist(), "normal_inward": access.normal_inward.tolist(),
                 "radius_mm": access.radius_mm, "window_id": access.window_id}
    canonical_affine = (np.diag([-1., -1., 1., 1.]) @ case.affine
                        if case.frame == "LPS+" else case.affine)
    if (binding.get("case_id") != case.case_id or binding.get("source_frame") != case.frame
            or binding.get("geometry_frame") != "RAS+" or binding.get("source_shape") != list(case.mri.shape)
            or binding.get("native_affine_ras_mm") != canonical_affine.tolist()
            or binding.get("requested_access_ras") != requested
            or any(binding.get(key) != value for key, value in preset.items())):
        raise BridgeError("AXIS_SOURCE_BINDING_MISMATCH", "Returned inspection does not match the source, window or declared preset")
    native_access = binding.get("access")
    if (not isinstance(native_access, dict) or set(native_access) != set(requested)
            or any(native_access[key] != requested[key] for key in ("center_mm", "radius_mm", "window_id"))):
        raise BridgeError("AXIS_SOURCE_BINDING_MISMATCH", "Returned native access differs from the requested window")
    try:
        normal = np.asarray(native_access["normal_inward"], dtype=float)
    except (TypeError, ValueError) as error:
        raise BridgeError("AXIS_SOURCE_BINDING_MISMATCH", "Returned native access has an invalid direction") from error
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise BridgeError("AXIS_SOURCE_BINDING_MISMATCH", "Returned native access has an invalid direction")
    difference = float(np.max(np.abs(normal - access.normal_inward)))
    tolerance = float(4 * np.finfo(np.float64).eps) if case.frame == "LPS+" else 0.
    expected_conversion = {
        "source_frame": case.frame, "normal_absolute_tolerance": tolerance,
        "normal_maximum_absolute_difference": difference,
        "policy": "LPS unit-direction renormalization roundoff only; center/radius/id exact",
    }
    if difference > tolerance or binding.get("access_conversion") != expected_conversion:
        raise BridgeError("AXIS_SOURCE_BINDING_MISMATCH", "Returned native direction exceeds the declared normalization contract")
    if expected_binding is not None and binding.get("binding_hash") != expected_binding:
        raise BridgeError("AXIS_BINDING_CHANGED", "Returned inspection differs from the requested prior binding")


def _path(value: Any, *, kind: str, output: bool = False) -> Path:
    path = Path(_string(value, "path")).expanduser()
    if not path.is_absolute():
        raise BridgeError("INVALID_PATH", "File-dialog paths must be absolute")
    if path.is_symlink():
        raise BridgeError("INVALID_PATH", "Select the original file rather than a symbolic link")
    path = path.resolve()
    endings = {"case": (".ressectionlab", ".rslab"), "json": (".json",), "NIfTI": (".nii", ".nii.gz")}[kind]
    if not any(path.name.lower().endswith(suffix) for suffix in endings):
        raise BridgeError("UNSUPPORTED_FILE_TYPE", f"Expected a {kind} file")
    if not output and (not path.is_file() or path.stat().st_size > MAX_CASE_BYTES):
        raise BridgeError("INVALID_PATH", "Selected file is missing, nonregular, or too large")
    if output and (not path.parent.is_dir() or (path.exists() and not path.is_file())):
        raise BridgeError("INVALID_PATH", "Save location must have an existing directory")
    return path


def _directory(value: Any) -> Path:
    path = Path(_string(value, "directory")).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise BridgeError("INVALID_PATH", "Select an existing original directory with the native file dialog")
    return path.resolve()


class BinaryTransfers:
    """Only generated numeric snapshots live here; source paths are never exposed."""

    def __init__(self, parent: Path):
        parent = Path(parent)
        if parent.is_symlink():
            raise BridgeError("INVALID_TRANSFER_DIR", "Transfer directory cannot be a symbolic link")
        parent.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="bridge-", dir=parent.resolve()))
        self._entries: dict[tuple, dict] = {}

    def array(self, value: np.ndarray, dtype: str) -> dict:
        converted = np.ascontiguousarray(value, dtype=np.dtype(dtype).newbyteorder("<"))
        if converted.nbytes > MAX_ARRAY_BYTES:
            raise BridgeError("ARRAY_SIZE_LIMIT", "Array exceeds the local desktop transfer limit")
        content = memoryview(converted).cast("B")
        digest = hashlib.sha256(content).hexdigest()
        key = (digest, converted.dtype.name, converted.shape)
        if key not in self._entries:
            destination = self.root / (digest + ".bin")
            if not destination.exists():
                with destination.open("xb") as output:
                    output.write(content)
                destination.chmod(0o400)
            self._entries[key] = {
                "path": str(destination), "dtype": converted.dtype.name,
                "shape": list(converted.shape), "byteOrder": "little", "order": "C",
                "sha256": digest, "byteLength": converted.nbytes,
            }
        return dict(self._entries[key])

    def prune(self, retained: set[str]) -> None:
        for key, descriptor in list(self._entries.items()):
            if descriptor["path"] not in retained:
                Path(descriptor["path"]).unlink(missing_ok=True)
                self._entries.pop(key)

    def close(self) -> None:
        shutil.rmtree(self.root)


@dataclass
class _CaseEntry:
    case: CaseData
    descriptor: dict
    artifacts: Any
    routes: Any = None
    display_series: dict[str, dict] = field(default_factory=dict)


@dataclass
class _Request:
    request_id: str
    cancelled: threading.Event = field(default_factory=threading.Event)
    terminal: bool = False
    timer: threading.Timer | None = None
    committing: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def check(self) -> None:
        if self.cancelled.is_set():
            raise BridgeError("CANCELLED", "Operation was cancelled")

    def begin_commit(self) -> None:
        with self.lock:
            self.check()
            self.committing = True

    def cancel(self) -> bool:
        with self.lock:
            if self.committing:
                return False
            self.cancelled.set()
            return True


class BridgeSession:
    def __init__(self, transfer_dir: Path, *, max_cases: int = 2, run_dir: Path | None = None,
                 observed_source_root: Path | None = None):
        if not isinstance(max_cases, int) or not 1 <= max_cases <= 4:
            raise ValueError("max_cases must be between one and four")
        self.transfers = BinaryTransfers(transfer_dir)
        self.max_cases = max_cases
        self.cases: OrderedDict[str, _CaseEntry] = OrderedDict()
        self.replay_transfers: OrderedDict[tuple, dict] = OrderedDict()
        self.run_reports: OrderedDict[str, Any] = OrderedDict()
        self.observed_source_root = observed_source_root
        self.observed_replay = None
        self.run_dir = None
        self._run_key = None
        if run_dir is not None:
            run_dir = Path(run_dir)
            if run_dir.is_symlink():
                raise BridgeError("INVALID_RUN_DIR", "Run directory cannot be a symbolic link")
            run_dir.mkdir(parents=True, exist_ok=True)
            self.run_dir = run_dir.resolve()
            key_path = self.run_dir / ".bridge-integrity-key"
            if key_path.is_symlink():
                raise BridgeError("INVALID_RUN_DIR", "Run integrity key cannot be a symbolic link")
            if not key_path.exists():
                descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as output:
                    output.write(os.urandom(32))
            if key_path.stat().st_size != 32:
                raise BridgeError("INVALID_RUN_DIR", "Run integrity key is invalid")
            self._run_key = key_path.read_bytes()
            if len(self._run_key) != 32:
                raise BridgeError("INVALID_RUN_DIR", "Run integrity key is invalid")

    def _get_case(self, value: Any) -> _CaseEntry:
        case_hash = _string(value, "caseHash", maximum=128)
        if case_hash not in self.cases:
            raise BridgeError("CASE_VERSION_UNAVAILABLE", "Case version is stale or no longer cached; reopen the case")
        self.cases.move_to_end(case_hash)
        entry = self.cases[case_hash]
        if entry.case.semantic_hash != case_hash:
            raise BridgeError("CASE_VERSION_MISMATCH", "Cached case has changed")
        return entry

    @staticmethod
    def _case_array_bytes(case: CaseData) -> int:
        arrays = [case.mri, case.affine, *case.compartments.values(), *case.source_compartments.values()]
        arrays.extend(item.mask for item in case.structural_evidence.values())
        for item in case.critical_evidence.values():
            arrays.extend((item.mask, item.annotation_coverage, item.affine_ras_mm))
        if case.brain_mask is not None:
            arrays.append(case.brain_mask)
        for item in case.prior_proposals.values():
            arrays.extend((item.data, item.sampling_coverage, item.affine_ras_mm,
                           item.source_prior_affine_ras_mm, item.mni_ras_to_patient_ras_mm))
        if case.functional_evidence is not None:
            evidence = case.functional_evidence
            arrays.extend(value for value in (evidence.motor, evidence.language,
                evidence.motor_coverage, evidence.language_coverage, evidence.affine_ras_mm)
                if value is not None)
        return sum(array.nbytes for array in arrays)

    def _prior_descriptor(self, item: Any) -> dict:
        manifest = item.to_manifest()
        return {"proposalId": item.proposal_id, "mapId": item.map_id,
                "component": item.component, "mapKind": item.map_kind,
                "title": item.metadata.get("title", item.map_id.replace("_", " ")),
                "provenance": "prior", "reviewStatus": "alignment_review_required",
                "viewOnly": True, "planningEligible": False, "patientSpecificFunction": False,
                "clinicalDeficitProbability": None, "clinicalRiskReason": "no_validated_clinical_outcome_model",
                "frame": "RAS+", "spatialUnits": "mm", "valueUnits": "unitless",
                "affine": item.affine_ras_mm.tolist(), "shape": list(item.data.shape),
                "data": self.transfers.array(item.data, "float32"),
                "samplingCoverage": self.transfers.array(item.sampling_coverage, "uint8"),
                "samplingCoverageFraction": manifest["sampling_coverage_fraction"],
                "coverageMeaning": manifest["coverage_meaning"], "interpolation": item.interpolation,
                "evidenceHash": manifest["evidence_hash"], "registrationHash": item.registration_hash,
                "sourceImageHash": item.source_image_hash, "sourceFrameHash": item.source_frame_hash,
                "sourcePlanningHash": item.source_case_planning_hash,
                "registrationCaseHash": item.registration_case_hash,
                "source": item.source.to_dict(), "metadata": thaw_json(item.metadata),
                "provenanceRecord": manifest}

    def _install_case(self, case: CaseData, artifacts: dict, request: _Request) -> dict:
        request.check()
        if case.mri.size * 4 > MAX_ARRAY_BYTES:
            raise BridgeError("ARRAY_SIZE_LIMIT", "Selected MRI is too large for this desktop view")
        if (len(case.compartments) > 32 or len(case.structural_evidence) > 8 or len(case.critical_evidence) > 3
                or len(case.prior_proposals) > MAX_PRIOR_PROPOSALS or self._case_array_bytes(case) > MAX_CASE_BYTES):
            raise BridgeError("CASE_SIZE_LIMIT", "Expanded case arrays exceed the desktop cache limit")
        if case.semantic_hash in self.cases:
            entry = self.cases[case.semantic_hash]
            request.begin_commit()
            entry.artifacts = freeze_json(artifacts)
            self.cases.move_to_end(case.semantic_hash)
            return {**entry.descriptor, "artifacts": self._display_artifacts(entry)}
        # Native voxels stay unchanged. The affine explicitly declares RAS or
        # LPS physical coordinates; renderers must respect that declaration.
        from .structural_evidence import planning_brain_support
        from .critical_evidence import resolve_critical_evidence
        critical = resolve_critical_evidence(case)
        brain_support = {"usableForResearchSimulation": False, "reviewStatus": "unassessed",
                         "corticalAccessPermitted": False}
        if case.brain_mask is not None:
            try:
                _, support = planning_brain_support(case)
                brain_support.update(usableForResearchSimulation=True, reviewStatus=support["review_status"], provenance=support)
            except ValueError as error:
                brain_support.update(reviewStatus="supplied_unverified", reason=str(error))
        descriptor = {
            "caseId": case.case_id, "caseHash": case.semantic_hash, "planningHash": case.planning_hash,
            "revision": case.revision, "frame": case.frame, "physicalUnits": "mm",
            "affine": case.affine.tolist(), "shape": list(case.mri.shape), "spacingMm": list(case.spacing_mm),
            "mri": self.transfers.array(case.mri, "float32"),
            "intensityRange": [float(case.mri.min()), float(case.mri.max())],
            "compartments": [{"name": name, "volumeMm3": float(np.count_nonzero(mask) * case.voxel_volume_mm3),
                              "array": self.transfers.array(mask, "uint8"),
                              "sourceArray": self.transfers.array(case.source_compartments.get(name, mask), "uint8")}
                             for name, mask in case.compartments.items()],
            "brainMask": None if case.brain_mask is None else self.transfers.array(case.brain_mask, "uint8"),
            "brainSupport": brain_support,
            "structuralEvidence": [{"evidenceId": item.evidence_id, "kind": "whole_brain_envelope",
                "provenance": item.provenance, "reviewStatus": item.review_status,
                "reviewRequired": item.review is None, "corticalAccessPermitted": False,
                "sourceHash": item.source_image_hash, "sourceFrameHash": item.source_frame_hash,
                "sourceFileHash": item.source_file_sha256, "maskHash": item.mask_hash,
                "modelHash": item.model_sha256, "runHash": item.run_sha256,
                "evidenceHash": item.evidence_hash, "method": item.method,
                "array": self.transfers.array(item.mask, "uint8"), "metadata": thaw_json(item.metadata),
                "review": None if item.review is None else item.review.to_manifest()}
                for item in case.structural_evidence.values()],
            "priorProposals": [self._prior_descriptor(item) for _, item in sorted(case.prior_proposals.items())],
            "functionalEvidence": (None if case.functional_evidence is None
                                   else case.functional_evidence.to_manifest()),
            "criticalEvidence": thaw_json(critical.receipt),
            "unknowns": list(case.unknowns), "metadata": thaw_json(case.metadata),
            "context": None if case.context is None else case.context.planning_view(),
            "planningAsOf": None if case.context is None else case.context.planning_as_of.isoformat(),
            "sourceRefs": [source.to_dict() for source in case.source_refs],
            "clinicalDeficitProbability": None, "clinicalRiskReason": "no_validated_clinical_outcome_model",
            "clinicalUseStatus": "research_only",
        }
        request.begin_commit()
        entry = _CaseEntry(case, descriptor, freeze_json(artifacts))
        self.cases[case.semantic_hash] = entry
        while len(self.cases) > self.max_cases:
            self.cases.popitem(last=False)
        self.prune_transfers()
        return {**descriptor, "artifacts": self._display_artifacts(entry)}

    def prune_transfers(self) -> None:
        """Discard generated arrays belonging to failed or cancelled imports."""
        retained = set()
        for cached in self.cases.values():
            from .workspace_imaging import retained_paths
            retained.update(retained_paths(cached.display_series))
            retained.add(cached.descriptor["mri"]["path"])
            if cached.descriptor["brainMask"]:
                retained.add(cached.descriptor["brainMask"]["path"])
            for compartment in cached.descriptor["compartments"]:
                retained.update((compartment["array"]["path"], compartment["sourceArray"]["path"]))
            for proposal in cached.descriptor["structuralEvidence"]:
                retained.add(proposal["array"]["path"])
            for proposal in cached.descriptor["priorProposals"]:
                retained.update((proposal["data"]["path"], proposal["samplingCoverage"]["path"]))
        for key, descriptor in list(self.replay_transfers.items()):
            if key[0] in self.cases:
                retained.add(descriptor["path"])
            else:
                self.replay_transfers.pop(key)
        self.transfers.prune(retained)

    @staticmethod
    def _display_artifacts(entry: _CaseEntry) -> dict:
        result = thaw_json(entry.artifacts)
        workspace = result.get("workspace", {})
        if workspace and workspace.get("case_hash") != entry.case.semantic_hash:
            result.pop("workspace", None)
            result["withheldWorkspaceReason"] = "saved_workspace_case_version_mismatch"
        elif workspace:
            routes = workspace.get("routes", [])
            if not isinstance(routes, list) or any(not isinstance(item, dict) or item.get("case_hash") != entry.case.semantic_hash for item in routes):
                workspace.pop("routes", None)
                result["withheldRoutesReason"] = "saved_route_case_version_mismatch"
            elif routes and entry.routes is not None and routes == thaw_json(entry.routes)["candidates"]:
                result["savedRouteValidation"] = "matches_current_session_evaluation"
                result["searchModels"] = thaw_json(entry.routes).get("search_models", [])
            elif routes:
                # Bundle artifacts are not covered by the imaging identity.
                # Preserve them privately for round-trip, but do not let a
                # matching case hash certify arbitrary imported metrics.
                workspace.pop("routes", None)
                result["savedRouteCount"] = len(routes)
                result["withheldRoutesReason"] = "saved_routes_require_current_evaluation"
            for key in ("training_report", "selection_replay"):
                if workspace.pop(key, None) is not None:
                    result["withheldTrainingReason"] = "saved_training_requires_current_validation"
        result.pop("route_search", None)
        return result

    @staticmethod
    def _json_bytes(value: Any) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    def _run_path(self, run_id: Any) -> Path:
        if self.run_dir is None:
            raise BridgeError("RUN_STORAGE_REQUIRED", "Configure a persistent local run directory before training")
        run_id = _string(run_id, "runId", maximum=36)
        try:
            if str(uuid.UUID(run_id)) != run_id:
                raise ValueError("Noncanonical run identifier")
        except ValueError as error:
            raise BridgeError("INVALID_RUN_ID", "Run ID must be issued by this desktop installation") from error
        directory = self.run_dir / run_id
        if directory.is_symlink() or directory.resolve().parent != self.run_dir:
            raise BridgeError("INVALID_RUN_ID", "Run directory has changed")
        return directory

    def _write_run(self, directory: Path, manifest: dict) -> None:
        # Local keyed integrity binds a run to the installation that created it.
        # It is not a clinical signature and never authenticates imported files.
        record = dict(manifest)
        record.pop("integrity", None)
        record["integrity"] = hmac.new(self._run_key, self._json_bytes(record), hashlib.sha256).hexdigest()
        temporary = directory / ".bridge-run.tmp"
        if temporary.is_symlink() or (directory / "bridge-run.json").is_symlink():
            raise BridgeError("INVALID_RUN_ID", "Run metadata path has changed")
        temporary.write_bytes(self._json_bytes(record))
        os.replace(temporary, directory / "bridge-run.json")

    def _read_run(self, case: CaseData, run_id: Any) -> tuple[Path, dict]:
        directory = self._run_path(run_id)
        path = directory / "bridge-run.json"
        if not directory.is_dir() or path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_WORKSPACE_BYTES:
            raise BridgeError("RUN_UNAVAILABLE", "Run is missing or its metadata is invalid")
        # Numerical writers use several fixed filenames. Never resume into a
        # directory containing redirected files, even if its manifest is valid.
        if any(child.is_symlink() for child in directory.rglob("*")):
            raise BridgeError("INVALID_RUN_ID", "Run files cannot be symbolic links")
        record = json.loads(path.read_bytes())
        integrity = record.pop("integrity", None)
        expected = hmac.new(self._run_key, self._json_bytes(record), hashlib.sha256).hexdigest()
        if not isinstance(integrity, str) or not hmac.compare_digest(integrity, expected):
            raise BridgeError("RUN_INTEGRITY_FAILED", "Saved run metadata changed outside this desktop")
        if record.get("runId") != run_id or record.get("schema") != 1:
            raise BridgeError("RUN_INTEGRITY_FAILED", "Saved run identity is invalid")
        if record.get("caseHash") != case.semantic_hash or record.get("planningHash") != case.planning_hash:
            raise BridgeError("CASE_VERSION_MISMATCH", "Training run belongs to another case version")
        if not self._persisted_run_matches(directory, record):
            raise BridgeError("RUN_INTEGRITY_FAILED", "Saved run artifacts changed outside this desktop")
        return directory, record

    @staticmethod
    def _persisted_run_matches(directory: Path, manifest: dict) -> bool:
        for key, name in RUN_INTEGRITY_FILES.items():
            if manifest.get(key) is not None:
                path = directory / name
                if (path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RUN_JSON_BYTES
                        or hashlib.sha256(path.read_bytes()).hexdigest() != manifest[key]):
                    return False
        return True

    @staticmethod
    def _run_options(config: dict, case: CaseData) -> dict:
        from .critical_evidence import resolve_critical_evidence
        critical = resolve_critical_evidence(case)
        if (critical.planning_binding is not None or "criticalEvidenceHash" in config) and config.get("criticalEvidenceHash") != critical.fingerprint:
            raise BridgeError("EVIDENCE_VERSION_MISMATCH", "Critical evidence differs from the frozen route")
        access = None if config.get("access") is None else AccessWindow(**config["access"])
        ids = config.get("toolIds")
        tools = None if ids is None else tuple(tool for tool in RESEARCH_TOOLS if tool.tool_id in ids)
        if ids is not None and len(tools) != len(ids):
            raise BridgeError("TOOL_UNAVAILABLE", "Frozen run references an unknown research tool")
        result = {"access": access, "tools": tools, "selected_entry_mm": config.get("selectedEntryMm"),
                  "selected_target_mm": config.get("selectedTargetMm")}
        if critical.hard_exclusion is not None:
            result.update(hard_exclusion=critical.hard_exclusion, hard_exclusion_provenance=critical.fingerprint)
        evidence = case.functional_evidence
        if evidence is not None:
            from .worlds import content_hash
            evidence.assert_matches(case)
            if (config.get("functionalEvidenceHash") != evidence.fingerprint
                    or config.get("worldGeneratorHash") != evidence.uncertainty.fingerprint
                    or content_hash(config.get("worldGenerator")) != evidence.uncertainty.fingerprint):
                raise BridgeError("EVIDENCE_VERSION_MISMATCH", "Run evidence or uncertainty differs from the current case")
            result["world_generator"] = evidence.uncertainty
        elif any(key in config for key in ("functionalEvidenceHash", "worldGeneratorHash", "worldGenerator")):
            raise BridgeError("EVIDENCE_VERSION_MISMATCH", "Run requires functional evidence absent from this case")
        return result

    @staticmethod
    def _route_config(entry: _CaseEntry, route_id: Any) -> dict:
        from .critical_evidence import resolve_critical_evidence
        critical = resolve_critical_evidence(entry.case)
        config = {"routeId": route_id, "access": None, "toolIds": None,
                  "selectedEntryMm": None, "selectedTargetMm": None, "coordinateFrame": "RAS+",
                  "scope": "default_native_candidate_search", "optimizationChoiceScope": "finite_native_candidate_actions"}
        if critical.planning_binding is not None:
            config["criticalEvidenceHash"] = critical.fingerprint
        if entry.case.functional_evidence is not None:
            evidence = entry.case.functional_evidence
            evidence.assert_matches(entry.case)
            config.update(functionalEvidenceHash=evidence.fingerprint,
                worldGenerator=evidence.uncertainty.to_dict(), worldGeneratorHash=evidence.uncertainty.fingerprint)
        if route_id is None:
            return config
        route_id = _string(route_id, "routeId", maximum=128)
        candidates = [] if entry.routes is None else thaw_json(entry.routes)["candidates"]
        route = next((item for item in candidates if item["route_id"] == route_id), None)
        if route is None or route["geometry"].get("feasible") is not True:
            raise BridgeError("ROUTE_UNAVAILABLE", "Choose a feasible route evaluated for this case in the current session")
        tool = next((item for item in RESEARCH_TOOLS if item.tool_id == route["tool"]["tool_id"]), None)
        if tool is None or route["tool"] != asdict(tool):
            raise BridgeError("TOOL_UNAVAILABLE", "Selected route has an unsupported or changed tool definition")
        conversion = np.array([-1., -1., 1.]) if entry.case.frame == "LPS+" else np.ones(3)
        access = dict(route["window"])
        for key in ("center_mm", "normal_inward"):
            access[key] = (np.asarray(access[key]) * conversion).tolist()
        config.update(access=access, toolIds=[tool.tool_id],
            selectedEntryMm=(np.asarray(route["entry_mm"]) * conversion).tolist(),
            selectedTargetMm=(np.asarray(route["target_mm"]) * conversion).tolist(),
            routePlanningModelHash=route["planning_model_hash"], scope="exact_selected_research_route",
            optimizationChoiceScope="STOP_or_declared_native_stroke")
        return config

    @staticmethod
    def _merge_search_records(prior: dict | None, current: dict) -> dict:
        """Keep separate frozen models and Pareto cohorts in one comparison list."""
        def receipt(record):
            return {**{key: value for key, value in record.items() if key != "candidates"},
                    "candidate_ids": [item["route_id"] for item in record["candidates"]]}
        records = [] if prior is None else prior.get("search_models", [receipt(prior)])
        records = [item for item in records if item["planning_model_hash"] != current["planning_model_hash"]]
        records.append(receipt(current))
        candidates = {} if prior is None else {item["route_id"]: item for item in prior["candidates"]}
        candidates.update((item["route_id"], item) for item in current["candidates"])
        return {**current, "candidates": list(candidates.values()), "search_models": records,
                "combined_models": len(records) > 1,
                "planning_model_hash": records[0]["planning_model_hash"] if len(records) == 1 else None,
                "planning_model_hashes": [item["planning_model_hash"] for item in records],
                "elapsed_seconds": sum(item["elapsed_seconds"] for item in records),
                "requested_candidates": sum(item["requested_candidates"] for item in records),
                "access_support": current["access_support"] if len(records) == 1 else {},
                "optimizer_version": current["optimizer_version"] if len(records) == 1 else "multiple_explicit_search_models",
                "assumptions": list(dict.fromkeys(statement for item in records for statement in item["assumptions"]))}

    def _inspect_axis_planning(self, args: dict, request: _Request, progress: Callable) -> dict:
        """Read-only expansion of one cached window; never install new route IDs."""
        _keys(args, {"caseHash", "planningHash", "routeId", "routePlanningModelHash", "toolIds",
                     "acknowledgeNeighboringColumns", "acknowledgeEstimatedSupport", "expectedBindingHash"})
        entry = self._get_case(args.get("caseHash"))
        planning_hash = _string(args.get("planningHash"), "planningHash", maximum=128)
        if planning_hash != entry.case.planning_hash:
            raise BridgeError("CASE_VERSION_MISMATCH", "Planning identity changed; inspect the current case")
        route_id = _string(args.get("routeId"), "routeId", maximum=128)
        route_model = _string(args.get("routePlanningModelHash"), "routePlanningModelHash", maximum=128)
        config = self._route_config(entry, route_id)
        route = next(item for item in thaw_json(entry.routes)["candidates"] if item["route_id"] == route_id)
        if route.get("case_hash") != args["caseHash"] or config.get("routePlanningModelHash") != route_model:
            raise BridgeError("ROUTE_VERSION_MISMATCH", "Selected route belongs to another case or planning model")
        selected = args.get("toolIds")
        known = {tool.tool_id for tool in NATIVE_GENERIC_TOOLS}
        if (not isinstance(selected, list) or not 1 <= len(selected) <= len(known)
                or any(not isinstance(item, str) or item not in known for item in selected)
                or len(selected) != len(set(selected))):
            raise BridgeError("INVALID_ARGUMENT", "Choose one or two distinct declared native research instruments")
        if args.get("acknowledgeNeighboringColumns") is not True:
            raise BridgeError("AXIS_ACKNOWLEDGEMENT_REQUIRED", "Explicitly acknowledge neighboring entries and endpoints")
        if type(args.get("acknowledgeEstimatedSupport")) is not bool:
            raise BridgeError("INVALID_ARGUMENT", "acknowledgeEstimatedSupport must be an explicit boolean")
        expected_binding = None
        if "expectedBindingHash" in args:
            expected_binding = _string(args["expectedBindingHash"], "expectedBindingHash", maximum=71)
            if (len(expected_binding) != 71 or not expected_binding.startswith("sha256:")
                    or any(character not in "0123456789abcdef" for character in expected_binding[7:])):
                raise BridgeError("INVALID_ARGUMENT", "expectedBindingHash must be a canonical SHA-256 identity")
        if (entry.case.mri.size > MAX_AXIS_INSPECTION_VOXELS
                or self._case_array_bytes(entry.case) > MAX_CASE_BYTES):
            raise BridgeError("AXIS_INPUT_SIZE_LIMIT", "Source grid exceeds the read-only inspection input limit")
        _require_json_budget({"metadata": thaw_json(entry.case.metadata), "unknowns": entry.case.unknowns,
            "structuralEvidence": [item.to_manifest() for item in entry.case.structural_evidence.values()]},
            MAX_AXIS_INSPECTION_METADATA_BYTES, "AXIS_METADATA_SIZE_LIMIT",
            "Source support/provenance exceeds the inspection metadata limit")
        request.check()
        tools = tuple(tool for tool in NATIVE_GENERIC_TOOLS if tool.tool_id in selected)
        anchor = config["access"]  # Exact current-session route window, converted into RAS+.
        access = AccessWindow(**anchor)
        normalization_delta = float(np.max(np.abs(access.normal_inward - anchor["normal_inward"])))
        normalization_atol = float(4 * np.finfo(np.float64).eps)
        if (normalization_delta > normalization_atol
                or not np.array_equal(access.center_mm, anchor["center_mm"])
                or access.radius_mm != anchor["radius_mm"] or access.window_id != anchor["window_id"]):
            raise BridgeError("ROUTE_VERSION_MISMATCH", "Cached window changed beyond unit-normal roundoff")
        from .native_axis_refinement import inspect_axis_planning
        from .native_proposals import AxisColumnProposalConfig
        from .simulation import RewardSpec
        from .worlds import WorldGeneratorConfig, content_hash
        reward, world_generator, proposal = RewardSpec(), WorldGeneratorConfig(), AxisColumnProposalConfig()
        evidence = entry.case.functional_evidence
        if evidence is not None:
            evidence.assert_matches(entry.case)
            world_generator = evidence.uncertainty
        preset = json.loads(json.dumps({"input_profile": "RAW", "max_steps": 3,
            "neighboring_columns_acknowledged": True,
            "world_role": None, "world_partitions_created": False,
            "population_priors_used": evidence is not None,
            "max_tip_step_mm": .25, "partial_contact_weight": .05,
            "max_actions": proposal.max_primary_rays + 1,
            "proposal_rule": asdict(proposal), "proposal_rule_hash": proposal.fingerprint,
            "reward": asdict(reward), "world_generator": world_generator.to_dict(),
            "world_generator_hash": world_generator.fingerprint}, allow_nan=False))
        if evidence is not None:
            preset["functional_evidence"] = json.loads(json.dumps(evidence.to_manifest(), allow_nan=False))
        progress(0., "Inspecting neighboring paths within the selected window; no tissue is removed")
        try:
            snapshot = inspect_axis_planning(entry.case, access_ras=access, tools=tools,
                acknowledge_neighboring_columns=True,
                acknowledge_estimated_support=args["acknowledgeEstimatedSupport"],
                expected_case_hash=args["caseHash"], expected_planning_hash=planning_hash,
                reward=reward, world_generator=world_generator, proposal_config=proposal, max_steps=3,
                partial_contact_weight=.05, max_tip_step_mm=.25,
                expected_binding_hash=expected_binding, cancelled=request.cancelled.is_set)
        except InterruptedError as error:
            raise BridgeError("CANCELLED", "Inspection cancelled; no inventory published") from error
        except ValueError as error:
            message = str(error)
            code = ("AXIS_ACCESS_UNSUPPORTED" if message.startswith("UNSUPPORTED_") else
                    "AXIS_SUPPORT_ACKNOWLEDGEMENT_REQUIRED" if "estimated-support acknowledgement" in message else
                    "AXIS_SUPPORT_UNAVAILABLE" if "BRAIN_MASK_" in message or "Reviewed brain support" in message else
                    "AXIS_BINDING_CHANGED" if "Stale axis inspection binding" in message else
                    "CASE_VERSION_MISMATCH" if "Stale source" in message else "AXIS_INSPECTION_REJECTED")
            raise BridgeError(code, message) from error
        except RuntimeError as error:
            raise BridgeError("AXIS_INSPECTION_INTEGRITY_FAILED", str(error)) from error
        report = snapshot.to_dict()
        binding, inventory = report.get("binding", {}), report.get("inventory", {})
        _require_axis_output_binding(binding, case=entry.case, access=access,
                                     preset=preset, expected_binding=expected_binding)
        actions = report.get("actions", [])
        batch = inventory.get("batch", {})
        accounting = report.get("accounting", {})
        ids = [action.get("action_id") for action in actions]
        if (report.get("version") != "native-axis-inspection-v1" or report.get("role") != "inspection"
                or report.get("inventory_complete") is not True or inventory.get("status") != "complete"
                or batch.get("unsupported_reason") is not None
                or report.get("status") not in {"ready", "no_actionable_moves"}
                or ids != ["STOP", *inventory.get("certified_action_ids", [])]
                or len(set(ids)) != len(ids) or report.get("legal_non_stop_actions") != len(ids) - 1
                or (report["status"] == "ready") != (len(ids) > 1)
                or batch.get("slot_count") != 13 * len(tools)
                or len(batch.get("ledger", [])) != batch.get("slot_count")
                or len(inventory.get("attempts", [])) > 52
                or any(row.get("status") != "complete" for row in inventory.get("attempts", []))
                or binding.get("case_hash") != args["caseHash"] or binding.get("planning_hash") != planning_hash
                or binding.get("tools") != [asdict(tool) for tool in tools]
                or report.get("candidate_eligible") is not False or report.get("removal_authorized") is not False
                or report.get("clinical_deficit_probability", "missing") is not None
                or any(type(accounting.get(key)) is not int or accounting[key] != 0 for key in
                       ("gradient_steps", "executed_transitions", "native_commits"))
                or type(accounting.get("simulated_removed_volume_mm3")) not in {int, float}
                or accounting["simulated_removed_volume_mm3"] != 0.
                or report.get("inspection_hash") != content_hash({k: v for k, v in report.items() if k != "inspection_hash"})
                or binding.get("binding_hash") != content_hash({k: v for k, v in binding.items() if k != "binding_hash"})):
            raise BridgeError("AXIS_INCOMPLETE_INSPECTION", "Inspection is incomplete or violates its read-only contract")
        request.check()
        if self._get_case(args["caseHash"]) is not entry or self._route_config(entry, route_id) != config:
            raise BridgeError("ROUTE_VERSION_MISMATCH", "Selected source window changed during inspection")
        result = {"schemaVersion": 1, "caseHash": args["caseHash"], "planningHash": planning_hash,
            "routeId": route_id, "routePlanningModelHash": route_model,
            "accessSource": "selected_route_window_only", "anchorWindowRas": anchor,
            "anchorWindowNormalization": {"normalAbsoluteTolerance": normalization_atol,
                                           "normalMaximumDifference": normalization_delta},
            "requestedToolIds": [tool.tool_id for tool in tools], "inspection": report}
        _require_json_budget(result, MAX_AXIS_INSPECTION_RESULT_BYTES, "AXIS_RESULT_SIZE_LIMIT",
                             "Complete inspection exceeds the response limit; no partial inventory published")
        request.check()
        progress(1., "Complete initial path inventory inspected; no actions executed")
        request.check()
        return result

    @staticmethod
    def _report_summary(manifest: dict, report: dict | None = None) -> dict:
        summary = {key: manifest[key] for key in ("runId", "caseHash", "planningHash", "createdAt", "status", "config")}
        summary["clinicalDeficitProbability"] = None
        summary["clinicalRiskReason"] = "no_validated_clinical_outcome_model"
        summary["evaluationSealed"] = manifest.get("evaluationSealed", False)
        summary["functionalEvaluationStatus"] = manifest.get("functionalEvaluationStatus", "not_evaluated")
        if manifest.get("lastResumeAttempt") is not None:
            summary["lastResumeAttempt"] = manifest["lastResumeAttempt"]
        if report is not None:
            training = {key: value for key, value in report.items() if key not in {"replay", "output_dir", "native_certificate"}}
            replay = report.get("replay")
            if replay is not None:
                training["replay"] = {key: replay[key] for key in ("role", "case_hash", "decision_model_hash", "checkpoint_hash", "artifact_hash", "shape", "affine", "final_evaluation", "scope")}
                training["replay"].update(stepCount=len(replay["metrics"]["history"]),
                    metrics={key: value for key, value in replay["metrics"].items() if key != "history"},
                    native_certificate=replay["native_certificate"])
                if replay.get("functional_assessment") is not None:
                    training["replay"]["functional_assessment"] = replay["functional_assessment"]
            else:
                training["replay"] = None
            summary["training"] = training
        return summary

    def _saved_report(self, case: CaseData, run_id: str, request: _Request) -> tuple[Path, dict, dict]:
        directory, manifest = self._read_run(case, run_id)
        path = directory / "native-refinement.json"
        if not path.is_file() or path.stat().st_size > MAX_RUN_JSON_BYTES:
            raise BridgeError("REPLAY_UNAVAILABLE", "This run has no completed selection report")
        contents = path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != manifest.get("reportSha256"):
            raise BridgeError("RUN_INTEGRITY_FAILED", "Saved training report changed after completion")
        if run_id in self.run_reports:
            self.run_reports.move_to_end(run_id)
            report = thaw_json(self.run_reports[run_id])
            self._require_selected_replay(directory, report)
            return directory, manifest, report
        report = json.loads(contents)
        if report.get("case_hash") != case.semantic_hash or report.get("final_evaluation") is not False:
            raise BridgeError("CASE_VERSION_MISMATCH", "Saved report has an invalid case or partition")
        replay = report.get("replay")
        if replay is not None:
            self._require_selected_replay(directory, report)
            from .native_refinement import recheck_native_replay
            from .learning import load_policy, policy_hash
            request.check()
            if policy_hash(load_policy(directory / "checkpoint.pt")) != replay.get("checkpoint_hash"):
                raise BridgeError("RUN_INTEGRITY_FAILED", "Selected replay does not match the saved policy")
            report["replay"] = recheck_native_replay(case, replay, cancelled=request.cancelled.is_set,
                                                    **self._run_options(manifest["config"], case))
        self.run_reports[run_id] = freeze_json(json.loads(self._json_bytes(report)))
        while len(self.run_reports) > 4:
            self.run_reports.popitem(last=False)
        return directory, manifest, report

    @staticmethod
    def _require_selected_replay(directory: Path, report: dict) -> None:
        if report.get("replay") is None:
            return
        from .native_refinement import require_completed_selection_replay
        try:
            contract = json.loads((directory / "contract.json").read_text())
            require_completed_selection_replay(report, contract)
        except (ValueError, OSError, TypeError) as error:
            raise BridgeError("REPLAY_UNAVAILABLE", "This checkpoint has no completed matching selection panel") from error

    @historical_only("RECORDED_EXPERIENCE_REQUIRED")
    def _train_patient(self, args: dict, request: _Request, progress: Callable) -> dict:
        _keys(args, {"caseHash", "budgetSeconds", "seed", "routeId", "resumeRunId"})
        entry = self._get_case(args.get("caseHash"))
        if self.run_dir is None:
            raise BridgeError("RUN_STORAGE_REQUIRED", "Configure a persistent local run directory before training")
        budget, seed = args.get("budgetSeconds", 30), args.get("seed", 0)
        if type(budget) not in (int, float) or not math.isfinite(budget) or not 5 <= budget <= 120:
            raise BridgeError("INVALID_ARGUMENT", "Training budget must be between 5 and 120 seconds")
        if type(seed) is not int or not 0 <= seed < 2 ** 31:
            raise BridgeError("INVALID_ARGUMENT", "seed must be a nonnegative 31-bit integer")
        resume = args.get("resumeRunId") is not None
        if resume:
            directory, manifest = self._read_run(entry.case, args["resumeRunId"])
            if manifest.get("evaluationSealed") or (directory / "native-functional-freeze.json").exists():
                raise BridgeError("RUN_EVALUATION_SEALED", "Independent evaluation has sealed this candidate. Resume its evaluation or inspect the frozen result; optimizer training cannot resume after evaluation worlds are revealed.")
            if manifest.get("checkpointSha256") is None or manifest.get("contractSha256") is None:
                raise BridgeError("RUN_NOT_RESUMABLE", "Run ended before a checkpoint could be integrity-checked; start a new run")
            for name in ("budgetSeconds", "seed", "routeId"):
                if name in args and args[name] != manifest["config"].get(name):
                    raise BridgeError("RESUME_CONTRACT_CHANGED", "Resume retains the original budget, seed, and access geometry")
        else:
            if sum(child.is_dir() for child in self.run_dir.iterdir()) >= 256:
                raise BridgeError("RUN_STORAGE_LIMIT", "Local run limit reached; archive old runs before starting another")
            config = {"budgetSeconds": float(budget), "seed": seed, **self._route_config(entry, args.get("routeId"))}
            run_id = str(uuid.uuid4())
            directory = self._run_path(run_id)
            directory.mkdir(mode=0o700)
            manifest = {"schema": 1, "runId": run_id, "caseHash": entry.case.semantic_hash,
                        "planningHash": entry.case.planning_hash, "createdAt": time.time(), "status": "preparing", "config": config}
        original_manifest = dict(manifest)
        config = manifest["config"]
        manifest["status"] = "running"
        manifest.pop("reportSha256", None)
        manifest.pop("replayStatus", None)
        self.run_reports.pop(manifest["runId"], None)
        self._write_run(directory, manifest)
        def training_progress(snapshot: dict) -> None:
            # The learner emits progress after atomically saving a checkpoint.
            # Sign that exact checkpoint so an interrupted process can resume
            # only from counters, optimizer state and weights we actually saw.
            manifest["checkpointSha256"] = hashlib.sha256((directory / "checkpoint.pt").read_bytes()).hexdigest()
            manifest["contractSha256"] = hashlib.sha256((directory / "contract.json").read_bytes()).hexdigest()
            manifest["nativeRequestSha256"] = hashlib.sha256((directory / "native-request.json").read_bytes()).hexdigest()
            self._write_run(directory, manifest)
            if not request.cancelled.is_set():
                fraction = min(.9, float(snapshot.get("elapsed_seconds", 0)) / config["budgetSeconds"])
                progress(fraction, "Updating and selecting patient-specific policy", {"runId": manifest["runId"], "phase": "training", "metrics": snapshot})
        try:
            progress(0., "Preparing native patient simulation", {"runId": manifest["runId"], "phase": "preparing"})
            from .native_refinement import run_native_refinement
            report = run_native_refinement(entry.case, directory, budget_seconds=config["budgetSeconds"], seed=config["seed"],
                resume=resume, cancelled=request.cancelled.is_set, progress=training_progress,
                **self._run_options(config, entry.case))
            manifest["status"] = "cancelled" if request.cancelled.is_set() else report["status"]
            report_path = directory / "native-refinement.json"
            report_bytes = report_path.read_bytes()
            report = json.loads(report_bytes)
            self._require_selected_replay(directory, report)
            manifest["reportSha256"] = hashlib.sha256(report_bytes).hexdigest()
            if report["status"] != "no_actionable_moves":
                manifest["checkpointSha256"] = hashlib.sha256((directory / "checkpoint.pt").read_bytes()).hexdigest()
                manifest["contractSha256"] = hashlib.sha256((directory / "contract.json").read_bytes()).hexdigest()
            manifest["nativeRequestSha256"] = hashlib.sha256((directory / "native-request.json").read_bytes()).hexdigest()
            manifest["replayStatus"] = report.get("replay_status")
            self._write_run(directory, manifest)
            request.begin_commit()
            self.run_reports[manifest["runId"]] = freeze_json(report)
            while len(self.run_reports) > 4:
                self.run_reports.popitem(last=False)
            return self._report_summary(manifest, report)
        except Exception as error:
            if resume and self._persisted_run_matches(directory, original_manifest):
                original_manifest["lastResumeAttempt"] = {"status": "cancelled" if request.cancelled.is_set() else "failed",
                    "reason": str(error)[:1000], "at": time.time()}
                self._write_run(directory, original_manifest)
                raise
            manifest["status"] = "cancelled" if request.cancelled.is_set() else "failed"
            manifest["failure"] = str(error)[:1000]
            checkpoint = directory / "checkpoint.pt"
            if checkpoint.is_file() and not checkpoint.is_symlink() and checkpoint.stat().st_size <= MAX_RUN_JSON_BYTES:
                manifest["checkpointSha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            contract = directory / "contract.json"
            if contract.is_file() and not contract.is_symlink() and contract.stat().st_size <= MAX_RUN_JSON_BYTES:
                manifest["contractSha256"] = hashlib.sha256(contract.read_bytes()).hexdigest()
            native_request = directory / "native-request.json"
            if native_request.is_file() and not native_request.is_symlink() and native_request.stat().st_size <= MAX_RUN_JSON_BYTES:
                manifest["nativeRequestSha256"] = hashlib.sha256(native_request.read_bytes()).hexdigest()
            self._write_run(directory, manifest)
            raise

    def execute(self, operation: str, args: dict, request: _Request, progress: Callable[[float, str], None]) -> Any:
        request.check()
        if operation in LEGACY_OPERATION_EXCLUSIONS:
            raise DataPolicyError(LEGACY_OPERATION_EXCLUSIONS[operation], operation)
        if operation == "inspectObservedLandmarkUpdate":
            from .observed_landmark_replay import ObservedLandmarkReplay, STUDY_ID
            action = args.get("action")
            fields = {"open": {"action", "studyId"},
                      "advance": {"action", "studyId", "expectedStateHash", "phase", "landmarkId"},
                      "reopen": {"action", "studyId", "expectedStateHash", "snapshot"}}
            if not isinstance(action, str) or action not in fields or set(args) != fields[action] or args.get("studyId") != STUDY_ID:
                raise BridgeError("INVALID_ARGUMENT", "Use the fixed observation study and exact open, advance or reopen fields")
            _require_json_budget(args, 16 * 1024, "REQUEST_SIZE_LIMIT", "Observation request exceeds 16 KiB")
            current = self.observed_replay
            if action != "open":
                expected = None if current is None else current.frame()["stateHash"]
                if args["expectedStateHash"] != expected:
                    raise BridgeError("OBSERVED_STATE_STALE", "Observation state changed; retrieve the current frame before updating")
                if action == "advance" and current is None:
                    raise BridgeError("OBSERVED_STATE_UNAVAILABLE", "Open the observation study before advancing")
            # Every operation rechecks the one pinned field, even when cached.
            loaded = ObservedLandmarkReplay.load(self.observed_source_root)
            candidate = loaded if current is None else loaded.reopen(current.snapshot())
            if action == "advance":
                candidate = candidate.advance(args["phase"], args["landmarkId"])
            elif action == "reopen":
                candidate = loaded.reopen(args["snapshot"])
            result = candidate.frame()
            _require_json_budget(result, 32 * 1024, "RESULT_SIZE_LIMIT", "Observation result exceeds 32 KiB")
            request.begin_commit()
            self.observed_replay = candidate
            return result
        if operation == "executeDevelopmentEpisode":
            # Explicitly authorized generated development path. The legacy
            # createSyntheticCase policy exclusion remains unchanged above.
            _keys(args, {"fixture", "selector"})
            if (set(args) != {"fixture", "selector"} or args.get("fixture") != "generated-sequential-v1"
                    or args.get("selector") not in ("scripted", "SEARCH")):
                raise BridgeError("INVALID_ARGUMENT", "Choose the fixed generated episode and selector")
            from .development_episode import execute_development_episode
            progress(0.1, "Executing the generated multistep software fixture")
            case, episode = execute_development_episode(selector=args["selector"], cancelled=request.cancelled.is_set)
            request.check()
            if (episode.get("caseHash") != case.semantic_hash or episode.get("patientAdmission") is not False
                    or episode.get("clinicalValidation") is not False or episode.get("evidenceKind") != "generated_software_fixture"):
                raise BridgeError("EPISODE_BINDING_MISMATCH", "Generated episode differs from its display case")
            _require_json_budget(episode, 2 * 1024 * 1024, "EPISODE_SIZE_LIMIT", "Generated episode exceeds 2 MiB")
            progress(0.9, "Publishing checked native history and source-grid transfers")
            canonical = json.dumps({key: value for key, value in episode.items() if key != "episodeId"},
                                   sort_keys=True, separators=(",", ":"), allow_nan=False)
            if "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest() != episode.get("episodeId"):
                raise BridgeError("EPISODE_DIGEST_MISMATCH", "Generated episode identity changed before publication")
            return {"case": self._install_case(case, {}, request), "episode": episode, "episodeCanonicalJson": canonical}
        if operation == "createSyntheticCase":
            _keys(args, {"shape"})
            shape = args.get("shape", [64, 64, 64])
            if not isinstance(shape, list) or len(shape) != 3 or any(type(item) is not int or not 24 <= item <= 128 for item in shape):
                raise BridgeError("INVALID_ARGUMENT", "Synthetic shape needs three integers between 24 and 128")
            progress(0.1, "Creating the synthetic geometry fixture")
            return self._install_case(create_synthetic_case(tuple(shape)), {}, request)
        if operation == "loadCase":
            _keys(args, {"path"})
            path = _path(args.get("path"), kind="case")
            progress(0.1, "Checking saved imaging and source versions")
            case = load_case(path)
            request.check()
            artifacts = read_case_artifacts(path)
            return self._install_case(case, artifacts, request)
        if operation == "importDisplaySeries":
            from .workspace_imaging import import_display_series
            return import_display_series(self, args, request, progress)
        if operation == "importNifti":
            _keys(args, {"structuralPath", "tumorMaskPath", "brainMaskPath", "caseId", "labelMap"})
            structural = _path(args.get("structuralPath"), kind="NIfTI")
            tumor = None if args.get("tumorMaskPath") is None else _path(args["tumorMaskPath"], kind="NIfTI")
            brain = None if args.get("brainMaskPath") is None else _path(args["brainMaskPath"], kind="NIfTI")
            case_id = None if args.get("caseId") is None else _string(args["caseId"], "caseId", maximum=256)
            label_map = args.get("labelMap")
            if label_map is not None:
                if not isinstance(label_map, dict) or len(label_map) > 32:
                    raise BridgeError("INVALID_ARGUMENT", "labelMap must have at most 32 labels")
                label_map = {int(_string(key, "label number", maximum=8)): _string(name, "label name", maximum=100) for key, name in label_map.items()}
            # Inspect declared dimensions before any scaled voxel allocation.
            for selected_path in (structural, tumor, brain):
                if selected_path is not None:
                    qc = inspect_nifti(selected_path)
                    count = math.prod(qc["shape"])
                    if count * 4 > MAX_ARRAY_BYTES:
                        raise BridgeError("ARRAY_SIZE_LIMIT", "NIfTI dimensions exceed the desktop import limit")
            if tumor is not None:
                import nibabel as nib
                annotation = nib.load(tumor)
                unique = set()
                for z in range(0, annotation.shape[2], 8):
                    request.check()
                    block = np.asarray(annotation.dataobj[:, :, z:z + 8])
                    if not np.all(np.isfinite(block)) or np.any(block < 0) or np.any(block != np.floor(block)):
                        raise BridgeError("INVALID_ANNOTATION", "Target annotation must contain finite nonnegative integer labels")
                    unique.update(np.unique(block).tolist())
                    if sum(value > 0 for value in unique) > 32:
                        raise BridgeError("LABEL_COUNT_LIMIT", "Annotation has more than 32 labels or fractional values; use a reviewed annotation importer")
                count = math.prod(annotation.shape)
                if count * (4 + 2 * sum(value > 0 for value in unique) + (2 if brain else 0)) > MAX_CASE_BYTES:
                    raise BridgeError("CASE_SIZE_LIMIT", "Expanded target masks exceed the desktop cache limit")
            progress(0.1, "Validating NIfTI coordinates and supplied annotations")
            case = load_nifti_case(structural, tumor, case_id=case_id, brain_mask_path=brain, label_map=label_map)
            return self._install_case(case, {}, request)
        if operation == "inspectEvidence":
            _keys(args, {"caseHash"})
            case = self._get_case(args.get("caseHash")).case
            from .critical_evidence import resolve_critical_evidence
            return {"caseHash": case.semantic_hash, "sourceRefs": [source.to_dict() for source in case.source_refs],
                    "metadata": thaw_json(case.metadata), "unknowns": list(case.unknowns),
                    "patientContext": None if case.context is None else case.context.planning_view(),
                    "structuralEvidence": [item.to_manifest() for item in case.structural_evidence.values()],
                    "criticalEvidence": thaw_json(resolve_critical_evidence(case).receipt),
                    "priorProposals": [item.to_manifest() for _, item in sorted(case.prior_proposals.items())],
                    "clinicalDeficitProbability": None, "clinicalRiskReason": "no_validated_clinical_outcome_model"}
        if operation == "importStructuralEvidence":
            _keys(args, {"caseHash", "sourceImagePath", "maskPath", "reportPath", "variant"})
            entry = self._get_case(args.get("caseHash"))
            if len(entry.case.structural_evidence) >= 8:
                raise BridgeError("CASE_SIZE_LIMIT", "Structural proposal limit reached")
            source = _path(args.get("sourceImagePath"), kind="NIfTI")
            mask = _path(args.get("maskPath"), kind="NIfTI")
            report = _path(args.get("reportPath"), kind="json")
            if report.stat().st_size > MAX_RUN_JSON_BYTES:
                raise BridgeError("REPORT_SIZE_LIMIT", "Structural extraction report is too large")
            if args.get("variant") not in {"nocsf", "main"}:
                raise BridgeError("INVALID_ARGUMENT", "Select the recorded main or nocsf model variant")
            for selected in (source, mask):
                if math.prod(inspect_nifti(selected)["shape"]) * 4 > MAX_ARRAY_BYTES:
                    raise BridgeError("ARRAY_SIZE_LIMIT", "Structural extraction grid exceeds the desktop limit")
            from .imaging import import_brain_extraction_evidence
            progress(.1, "Checking extraction source, model, output hashes and native frame")
            case = import_brain_extraction_evidence(entry.case, source_image_path=source, mask_path=mask,
                                                    report_path=report, variant=args["variant"])
            return self._install_case(case, thaw_json(entry.artifacts), request)
        if operation == "importPriorProposals":
            _keys(args, {"caseHash", "registrationDirectory", "sourceCacheDirectory", "priorManifestPath",
                         "sourceImagePath", "registrationImagePath", "lesionPath"})
            entry = self._get_case(args.get("caseHash"))
            directory = _directory(args.get("registrationDirectory"))
            cache = _directory(args.get("sourceCacheDirectory"))
            manifest = _path(args.get("priorManifestPath"), kind="json")
            source = _path(args.get("sourceImagePath"), kind="NIfTI")
            registration_image = _path(args.get("registrationImagePath"), kind="NIfTI")
            lesion = _path(args.get("lesionPath"), kind="NIfTI")
            inventory = _path(str(directory / "functional_overlay_inventory.json"), kind="json")
            report = _path(str(directory / "registration_comparison.json"), kind="json")
            if any(path.stat().st_size > 16 * 1024**2 for path in (inventory, report, manifest)):
                raise BridgeError("REPORT_SIZE_LIMIT", "Prior proposal metadata exceeds the local import limit")
            layers = json.loads(inventory.read_text())
            if not isinstance(layers, list) or not 0 < len(layers) <= MAX_PRIOR_PROPOSALS:
                raise BridgeError("CASE_SIZE_LIMIT", "Prior proposal inventory exceeds the desktop layer limit")
            # Reserve the whole requested collection before any resampling. This
            # conservative bound includes already cached alternative proposals.
            if (len(entry.case.prior_proposals) + len(layers) > MAX_PRIOR_PROPOSALS
                    or self._case_array_bytes(entry.case) + len(layers) * (entry.case.mri.size * 5 + 384) > MAX_CASE_BYTES):
                raise BridgeError("CASE_SIZE_LIMIT", "Registered prior arrays exceed the desktop cache limit")
            for selected in (source, registration_image, lesion):
                if math.prod(inspect_nifti(selected)["shape"]) * 4 > MAX_ARRAY_BYTES:
                    raise BridgeError("ARRAY_SIZE_LIMIT", "Prior registration source grid exceeds the desktop limit")
            request.check()
            from .prior_proposals import import_registered_prior_proposals
            progress(.1, "Verifying saved population maps, source anatomy, transforms and atlas coverage")
            case = import_registered_prior_proposals(entry.case, registration_directory=directory,
                source_cache_directory=cache, prior_manifest_path=manifest, source_image_path=source,
                registration_image_path=registration_image, lesion_path=lesion,
                cancelled=request.cancelled.is_set)
            return self._install_case(case, thaw_json(entry.artifacts), request)
        if operation == "generateRoutes":
            _keys(args, {"caseHash", "toolIds", "allowEstimatedSupport", "config"})
            entry = self._get_case(args.get("caseHash"))
            selected = args.get("toolIds", [tool.tool_id for tool in GENERIC_TOOLS])
            if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)) or any(item not in {tool.tool_id for tool in GENERIC_TOOLS} for item in selected):
                raise BridgeError("INVALID_ARGUMENT", "toolIds must select distinct known generic instruments")
            config_data = args.get("config", {})
            if not isinstance(config_data, dict):
                raise BridgeError("INVALID_ARGUMENT", "config must be an object")
            _keys(config_data, {"targetsPerCompartment", "maxWindows"})
            targets, windows = config_data.get("targetsPerCompartment", 3), config_data.get("maxWindows", 3)
            if type(targets) is not int or not 1 <= targets <= 5 or type(windows) is not int or not 1 <= windows <= 3:
                raise BridgeError("INVALID_ARGUMENT", "Route search bounds exceeded")
            if len(entry.case.compartments) * targets * windows * len(selected) > 180:
                raise BridgeError("BUDGET_EXCEEDED", "Route request exceeds the desktop candidate budget")
            support = None
            provenance = None
            if entry.case.brain_mask is None:
                if args.get("allowEstimatedSupport") is not True:
                    raise BridgeError("ACCESS_SUPPORT_REQUIRED", "Review estimated intracranial support or supply a brain mask before generating routes")
                if entry.case.metadata.get("allow_nonzero_mri_access_support") is False or entry.case.metadata.get("structural_coverage") == "full_head":
                    raise BridgeError("FULL_HEAD_SUPPORT_NOT_CORTEX", "Full-head MRI cannot define cortical access; supply a reviewed cerebral mask")
                support = np.isfinite(entry.case.mri) & (entry.case.mri != 0)
                provenance = {"source": entry.case.semantic_hash, "method": "finite_nonzero_structural_MRI_support_v1", "evidence_type": "estimated"}
            result = generate_candidate_routes(entry.case, tools=tuple(tool for tool in GENERIC_TOOLS if tool.tool_id in selected),
                       config=SearchConfig(targets_per_compartment=targets, max_windows=windows),
                       support_mask=support, support_provenance=provenance, cancel=request.cancelled.is_set,
                       progress=lambda done, total: progress(done / max(total, 1), f"Checking complete instruments: {done}/{total}"))
            request.check()
            if result.cancelled:
                raise BridgeError("CANCELLED", "Route generation was cancelled")
            record = result.to_dict()
            request.begin_commit()
            entry.routes = freeze_json(record)
            return record
        if operation == "generateNativeRoutes":
            _keys(args, {"caseHash"})
            entry = self._get_case(args.get("caseHash"))
            from .native_routes import generate_native_axis_routes
            result = generate_native_axis_routes(entry.case, cancel=request.cancelled.is_set,
                progress=lambda done, total: progress(done / max(total, 1), f"Checking separate native research approaches: {done}/{total}"))
            request.check()
            if result.cancelled:
                raise BridgeError("CANCELLED", "Native research route generation was cancelled")
            record = self._merge_search_records(None if entry.routes is None else thaw_json(entry.routes), result.to_dict())
            request.begin_commit()
            entry.routes = freeze_json(record)
            return record
        if operation == "inspectRefinement":
            _keys(args, {"caseHash", "routeId"})
            entry = self._get_case(args.get("caseHash"))
            route_id = _string(args.get("routeId"), "routeId", maximum=128)
            config = self._route_config(entry, route_id)
            from .native_refinement import inspect_native_refinement
            progress(0., "Checking initial cutting actions for the selected entry, target and instrument")
            readiness = inspect_native_refinement(entry.case, cancelled=request.cancelled.is_set,
                                                 **self._run_options(config, entry.case))
            request.check()
            return {**readiness, "caseHash": entry.case.semantic_hash, "routeId": route_id,
                    "optimizationChoiceScope": config["optimizationChoiceScope"],
                    "frozenRoute": config, "clinicalDeficitProbability": None,
                    "clinicalRiskReason": "no_validated_clinical_outcome_model"}
        if operation == "inspectAxisPlanning":
            return self._inspect_axis_planning(args, request, progress)
        if operation == "saveCase":
            _keys(args, {"caseHash", "path", "workspace", "overwrite"})
            entry = self._get_case(args.get("caseHash"))
            destination = _path(args.get("path"), kind="case", output=True)
            if destination.exists() and args.get("overwrite") is not True:
                raise BridgeError("DESTINATION_EXISTS", "Save destination already exists; the native dialog must confirm replacement")
            workspace = args.get("workspace", {})
            if not isinstance(workspace, dict) or len(json.dumps(workspace, allow_nan=False).encode()) > MAX_WORKSPACE_BYTES:
                raise BridgeError("INVALID_ARGUMENT", "Workspace must be a bounded JSON object")
            if "routes" in workspace or "training_report" in workspace or "selection_replay" in workspace:
                raise BridgeError("INVALID_ARGUMENT", "Evaluator results cannot be supplied by the renderer")
            artifacts = thaw_json(entry.artifacts)
            prior_workspace = artifacts.get("workspace", {})
            if prior_workspace.get("case_hash") != entry.case.semantic_hash:
                prior_workspace = {}
            prior_workspace.update(workspace)
            prior_workspace["case_hash"] = entry.case.semantic_hash
            if entry.routes is not None:
                prior_workspace["routes"] = thaw_json(entry.routes)["candidates"]
                artifacts["route_search"] = thaw_json(entry.routes)
            artifacts["workspace"] = prior_workspace
            request.check()
            progress(0.2, "Saving imaging, source versions, and evaluated routes")
            with tempfile.TemporaryDirectory(prefix=".resection-save-", dir=destination.parent) as temporary:
                staged = save_case(entry.case, Path(temporary) / "case.ressectionlab", artifacts=artifacts)
                request.begin_commit()
                os.replace(staged, destination)
            entry.artifacts = freeze_json(artifacts)
            return {"caseHash": entry.case.semantic_hash, "path": str(destination), "saved": True}
        if operation in {"nativeTraining", "trainPatient"}:
            return self._train_patient(args, request, progress)
        if operation == "listRuns":
            _keys(args, {"caseHash"})
            case = self._get_case(args.get("caseHash")).case
            runs = []
            if self.run_dir is not None:
                for directory in self.run_dir.iterdir():
                    request.check()
                    if not directory.is_dir() or directory.name.startswith("."):
                        continue
                    try:
                        _, manifest = self._read_run(case, directory.name)
                    except (BridgeError, ValueError, OSError):
                        continue
                    summary = self._report_summary(manifest)
                    summary["hasCheckpoint"] = (directory / "checkpoint.pt").is_file()
                    summary["hasAcceptedReplay"] = manifest.get("replayStatus") == "accepted_independent_geometry"
                    if summary["hasAcceptedReplay"]:
                        try:
                            retained = json.loads((directory / "native-refinement.json").read_text())
                            self._require_selected_replay(directory, retained)
                            summary["hasAcceptedReplay"] = retained.get("replay") is not None
                        except (BridgeError, ValueError, OSError):
                            summary["hasAcceptedReplay"] = False
                    runs.append(summary)
            return {"caseHash": case.semantic_hash, "runs": sorted(runs, key=lambda value: value["createdAt"], reverse=True)}
        if operation in {"replayTraining", "evaluateCandidate", "exportCandidate"}:
            allowed = ({"caseHash", "runId", "step"} if operation == "replayTraining" else
                       {"caseHash", "runId"} if operation == "evaluateCandidate" else
                       {"caseHash", "runId", "path", "overwrite"})
            _keys(args, allowed)
            case = self._get_case(args.get("caseHash")).case
            directory, manifest, report = self._saved_report(case, args.get("runId"), request)
            replay = report.get("replay")
            if replay is None or report.get("replay_status") != "accepted_independent_geometry":
                raise BridgeError("REPLAY_UNAVAILABLE", "Run has no independently accepted native selection replay")
            from .native_refinement import native_replay_mask, validate_native_replay
            validate_native_replay(case, replay)
            if operation == "evaluateCandidate":
                if case.functional_evidence is None:
                    raise BridgeError("FUNCTIONAL_EVIDENCE_UNAVAILABLE", "Select source-bound functional evidence before evaluating sensitivity; structural-only planning remains unassessed.")
                from .native_refinement import evaluate_native_refinement_candidate
                def sealed(_seal):
                    manifest["evaluationSealed"] = True
                    manifest["functionalEvaluationStatus"] = "sealed_pending_evaluation"
                    manifest["functionalFreezeSha256"] = hashlib.sha256(
                        (directory / "native-functional-freeze.json").read_bytes()).hexdigest()
                    self._write_run(directory, manifest)
                progress(0., "Sealing the selected sequence before independent functional sensitivity evaluation")
                try:
                    report = evaluate_native_refinement_candidate(case, directory, replay,
                        cancelled=request.cancelled.is_set, on_seal=sealed,
                        **self._run_options(manifest["config"], case))
                except Exception as error:
                    self.run_reports.pop(manifest["runId"], None)
                    if manifest.get("evaluationSealed"):
                        manifest["functionalEvaluationStatus"] = (
                            "sealed_interrupted" if isinstance(error, InterruptedError) or request.cancelled.is_set()
                            else "sealed_failed")
                        self._write_run(directory, manifest)
                    if isinstance(error, InterruptedError):
                        raise BridgeError("CANCELLED", str(error)) from error
                    raise
                manifest["reportSha256"] = hashlib.sha256((directory / "native-refinement.json").read_bytes()).hexdigest()
                manifest["functionalReportSha256"] = hashlib.sha256((directory / "native-functional-events.json").read_bytes()).hexdigest()
                manifest["functionalEvaluationStatus"] = "complete"
                self._write_run(directory, manifest)
                self.run_reports[manifest["runId"]] = freeze_json(json.loads(self._json_bytes(report)))
                return {**self._report_summary(manifest, report),
                        "functionalAssessment": report["replay"]["functional_assessment"]}
            if operation == "exportCandidate":
                destination = _path(args.get("path"), kind="json", output=True)
                if destination.exists() and args.get("overwrite") is not True:
                    raise BridgeError("DESTINATION_EXISTS", "The native dialog must confirm replacement")
                payload = {"schema": "ressectionlab.native-selection-candidate.v1", "run": self._report_summary(manifest, report),
                           "candidate": replay, "clinical_deficit_probability": None,
                           "clinical_risk_reason": "no_validated_clinical_outcome_model", "final_evaluation": False}
                with tempfile.TemporaryDirectory(prefix=".resection-export-", dir=destination.parent) as temporary:
                    staged = Path(temporary) / "candidate.json"
                    staged.write_bytes(self._json_bytes(payload))
                    request.begin_commit()
                    os.replace(staged, destination)
                return {"runId": manifest["runId"], "caseHash": case.semantic_hash, "path": str(destination), "exported": True}
            history = replay["metrics"]["history"]
            step = args.get("step", len(history))
            if type(step) is not int or not 0 <= step <= len(history):
                raise BridgeError("INVALID_ARGUMENT", "Replay step is outside the certified history")
            removed = native_replay_mask(replay, step)
            request.check()
            descriptor = self.transfers.array(removed, "uint8")
            request.begin_commit()
            key = (case.semantic_hash, manifest["runId"], step)
            self.replay_transfers[key] = descriptor
            self.replay_transfers.move_to_end(key)
            while len(self.replay_transfers) > 4:
                self.replay_transfers.popitem(last=False)
            target = np.zeros(case.mri.shape, bool)
            for mask in case.compartments.values():
                target |= mask
            volume = case.voxel_volume_mm3
            return {"runId": manifest["runId"], "caseHash": case.semantic_hash, "step": step,
                    "stepCount": len(history), "role": "selection", "finalEvaluation": False,
                    "frame": "RAS+", "affine": replay["affine"], "removedMask": descriptor,
                    "simulatedRemovedTargetVolumeMm3": float(np.count_nonzero(removed & target) * volume),
                    "simulatedRemovedNormalVolumeMm3": float(np.count_nonzero(removed & ~target) * volume),
                    "modeledResidualTargetVolumeMm3": float(np.count_nonzero(target & ~removed) * volume),
                    "clinicalDeficitProbability": None, "clinicalRiskReason": "no_validated_clinical_outcome_model",
                    "functionalAssessment": replay.get("functional_assessment"),
                    "functionalAssessmentScope": "entire_frozen_sequence_not_current_replay_prefix" if replay.get("functional_assessment") else None,
                    "training": self._report_summary(manifest, report)["training"]}
        raise BridgeError("UNSUPPORTED_OPERATION", "Unknown sidecar operation")

    def close(self) -> None:
        self.cases.clear()
        self.observed_replay = None
        self.transfers.close()


class BridgeRuntime:
    """One numerical worker; cancellation, timeouts and ping never wait on it."""

    def __init__(self, transfer_dir: Path, emit: Callable[[dict], None], *, max_cases: int = 2, run_dir: Path | None = None,
                 observed_source_root: Path | None = None):
        self.session = BridgeSession(transfer_dir, max_cases=max_cases, run_dir=run_dir,
                                     observed_source_root=observed_source_root)
        self.run_dir = self.session.run_dir
        self.emit = emit
        self._lock = threading.RLock()
        self._requests: dict[str, _Request] = {}
        self._seen: set[str] = set()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="resection-numeric")
        self._closed = False

    def _terminal(self, request: _Request, event: str, **payload: Any) -> None:
        with self._lock:
            if request.terminal:
                return
            request.terminal = True
            if request.timer:
                request.timer.cancel()
            self.emit({"id": request.request_id, "event": event, **payload})

    def submit(self, message: Any) -> None:
        request_id = message.get("id") if isinstance(message, dict) else None
        try:
            if not isinstance(message, dict):
                raise BridgeError("INVALID_REQUEST", "Request must be an object")
            _keys(message, {"id", "op", "args", "timeoutMs"})
            request_id = _string(request_id, "id", maximum=128)
            operation = _string(message.get("op"), "op", maximum=64)
            if operation not in OPERATIONS:
                raise BridgeError("UNSUPPORTED_OPERATION", "Unknown sidecar operation")
            args = message.get("args", {})
            if not isinstance(args, dict):
                raise BridgeError("INVALID_ARGUMENT", "args must be an object")
            timeout = message.get("timeoutMs", 120000)
            if type(timeout) is not int or not 100 <= timeout <= 300000:
                raise BridgeError("INVALID_ARGUMENT", "timeoutMs must be an integer between 100 and 300000")
            if len(json.dumps(message, allow_nan=False).encode()) > MAX_REQUEST_BYTES:
                raise BridgeError("REQUEST_SIZE_LIMIT", "Request exceeds 1 MiB")
            copied_args = thaw_json(freeze_json(args))
            with self._lock:
                if self._closed:
                    raise BridgeError("SHUTDOWN", "Bridge is shutting down")
                if request_id in self._seen:
                    self.emit({"id": None, "rejectedRequestId": request_id, "event": "error", "error": {"code": "DUPLICATE_REQUEST_ID", "message": "Request IDs cannot be reused"}})
                    return
                if len(self._seen) >= 10000:
                    raise BridgeError("SESSION_LIMIT", "Restart the local bridge after 10000 requests")
                self._seen.add(request_id)
                if operation == "ping":
                    _keys(args, set())
                    self.emit({"id": request_id, "event": "result", "result": {"protocolVersion": PROTOCOL_VERSION,
                               "operations": sorted(OPERATIONS), "tools": [asdict(tool) for tool in GENERIC_TOOLS],
                               "nativeResearchTools": [asdict(tool) for tool in NATIVE_GENERIC_TOOLS]}})
                    return
                if operation == "cancel":
                    _keys(args, {"requestId"})
                    target_id = _string(args.get("requestId"), "requestId", maximum=128)
                    target = self._requests.get(target_id)
                    active = target is not None and not target.terminal
                    if active:
                        active = target.cancel()
                        if active:
                            self._terminal(target, "cancelled")
                    self.emit({"id": request_id, "event": "result", "result": {"requestId": target_id, "cancelled": active}})
                    return
                if operation == "shutdown":
                    _keys(args, set())
                    for target in self._requests.values():
                        if not target.terminal:
                            if target.cancel():
                                self._terminal(target, "cancelled")
                    self._closed = True
                    self.emit({"id": request_id, "event": "result", "result": {"shutdown": True}})
                    return
                if len(self._requests) >= MAX_PENDING:
                    raise BridgeError("BUSY", "Too many queued numerical jobs")
                request = _Request(request_id)
                self._requests[request_id] = request
                request.timer = threading.Timer(timeout / 1000, self._timeout, args=(request,))
                request.timer.daemon = True
                request.timer.start()
                self.emit({"id": request_id, "event": "started"})
                self._executor.submit(self._work, operation, copied_args, request)
        except Exception as exc:
            self.emit({"id": request_id if isinstance(request_id, str) else None, "event": "error",
                       "error": {"code": getattr(exc, "code", "INVALID_REQUEST"), "message": str(exc)}})

    def _timeout(self, request: _Request) -> None:
        if request.cancel():
            self._terminal(request, "error", error={"code": "TIMEOUT", "message": "Operation exceeded its declared time budget"})

    def _work(self, operation: str, args: dict, request: _Request) -> None:
        def progress(fraction: float, message: str, details: dict | None = None) -> None:
            with self._lock:
                if not request.terminal and not request.cancelled.is_set():
                    self.emit({"id": request.request_id, "event": "progress", "progress": {"fraction": float(fraction), "message": message, **(details or {})}})
        try:
            result = self.session.execute(operation, args, request, progress)
            request.check()
            self._terminal(request, "result", result=result)
        except Exception as exc:
            if request.cancelled.is_set() or getattr(exc, "code", None) == "CANCELLED":
                self._terminal(request, "cancelled")
            else:
                self._terminal(request, "error", error={"code": getattr(exc, "code", "OPERATION_FAILED"), "message": str(exc)})
        finally:
            self.session.prune_transfers()
            with self._lock:
                self._requests.pop(request.request_id, None)

    def wait_idle(self, timeout: float = 30) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if not self._requests:
                    return True
            time.sleep(0.005)
        return False

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for request in self._requests.values():
                if request.cancel():
                    self._terminal(request, "cancelled")
        self._executor.shutdown(wait=True, cancel_futures=False)
        self.session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-dir", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path, help="Persistent local checkpoint directory supplied by Electron main")
    parser.add_argument("--observed-source-root", type=Path, help="Trusted source root supplied by Electron main; never a renderer argument")
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    protocol_output = sys.stdout
    # Imported numerical libraries may print diagnostics; reserve stdout for
    # the wire protocol and route all incidental prints to stderr.
    sys.stdout = sys.stderr
    output_lock = threading.Lock()
    def emit(event: dict) -> None:
        encoded = json.dumps(event, separators=(",", ":"), allow_nan=False)
        with output_lock:
            protocol_output.write(encoded + "\n")
            protocol_output.flush()
    runtime = BridgeRuntime(args.transfer_dir, emit, run_dir=args.run_dir, observed_source_root=args.observed_source_root)
    try:
        while not runtime._closed:
            line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_REQUEST_BYTES:
                emit({"id": None, "event": "error", "error": {"code": "REQUEST_SIZE_LIMIT", "message": "Request exceeds 1 MiB"}})
                # Resynchronise at the next line; never parse a truncated tail.
                while line and not line.endswith(b"\n"):
                    line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
                continue
            try:
                request = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON is forbidden")))
            except (ValueError, UnicodeDecodeError, RecursionError) as exc:
                emit({"id": None, "event": "error", "error": {"code": "INVALID_JSON", "message": str(exc)}})
                continue
            runtime.submit(request)
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
