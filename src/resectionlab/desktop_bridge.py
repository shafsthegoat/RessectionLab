"""Local JSONL sidecar for Electron. No GUI imports, sockets, or arbitrary code.

Electron's main process owns file dialogs and grants only chosen paths. Binary
descriptors are internal to that process; the renderer receives opaque asset
IDs. Every geometry operation names a cached immutable case version.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import hashlib
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
from typing import Any, Callable

import numpy as np

from .core import CaseData, freeze_json, thaw_json
from .geometry import GENERIC_TOOLS
from .imaging import create_synthetic_case, inspect_nifti, load_case, load_nifti_case, read_case_artifacts, save_case
from .planning import SearchConfig, generate_candidate_routes

PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 1024 * 1024
MAX_ARRAY_BYTES = 512 * 1024 * 1024
MAX_CASE_BYTES = 1024 * 1024 * 1024
MAX_WORKSPACE_BYTES = 512 * 1024
MAX_PENDING = 8
OPERATIONS = frozenset({"ping", "loadCase", "importNifti", "saveCase", "generateRoutes", "cancel", "inspectEvidence", "createSyntheticCase", "nativeTraining", "shutdown"})


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


def _path(value: Any, *, kind: str, output: bool = False) -> Path:
    path = Path(_string(value, "path")).expanduser()
    if not path.is_absolute():
        raise BridgeError("INVALID_PATH", "File-dialog paths must be absolute")
    if path.is_symlink():
        raise BridgeError("INVALID_PATH", "Select the original file rather than a symbolic link")
    path = path.resolve()
    endings = (".ressectionlab", ".rslab") if kind == "case" else (".nii", ".nii.gz")
    if not any(path.name.lower().endswith(suffix) for suffix in endings):
        raise BridgeError("UNSUPPORTED_FILE_TYPE", f"Expected a {kind} file")
    if not output and (not path.is_file() or path.stat().st_size > MAX_CASE_BYTES):
        raise BridgeError("INVALID_PATH", "Selected file is missing, nonregular, or too large")
    if output and (not path.parent.is_dir() or (path.exists() and not path.is_file())):
        raise BridgeError("INVALID_PATH", "Save location must have an existing directory")
    return path


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
    def __init__(self, transfer_dir: Path, *, max_cases: int = 2):
        if not isinstance(max_cases, int) or not 1 <= max_cases <= 4:
            raise ValueError("max_cases must be between one and four")
        self.transfers = BinaryTransfers(transfer_dir)
        self.max_cases = max_cases
        self.cases: OrderedDict[str, _CaseEntry] = OrderedDict()

    def _get_case(self, value: Any) -> _CaseEntry:
        case_hash = _string(value, "caseHash", maximum=128)
        if case_hash not in self.cases:
            raise BridgeError("CASE_VERSION_UNAVAILABLE", "Case version is stale or no longer cached; reopen the case")
        self.cases.move_to_end(case_hash)
        entry = self.cases[case_hash]
        if entry.case.semantic_hash != case_hash:
            raise BridgeError("CASE_VERSION_MISMATCH", "Cached case has changed")
        return entry

    def _install_case(self, case: CaseData, artifacts: dict, request: _Request) -> dict:
        request.check()
        if case.mri.size * 4 > MAX_ARRAY_BYTES:
            raise BridgeError("ARRAY_SIZE_LIMIT", "Selected MRI is too large for this desktop view")
        if case.semantic_hash in self.cases:
            entry = self.cases[case.semantic_hash]
            request.begin_commit()
            entry.artifacts = freeze_json(artifacts)
            self.cases.move_to_end(case.semantic_hash)
            return {**entry.descriptor, "artifacts": self._display_artifacts(entry)}
        # Native voxels stay unchanged. The affine explicitly declares RAS or
        # LPS physical coordinates; renderers must respect that declaration.
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
            retained.add(cached.descriptor["mri"]["path"])
            if cached.descriptor["brainMask"]:
                retained.add(cached.descriptor["brainMask"]["path"])
            for compartment in cached.descriptor["compartments"]:
                retained.update((compartment["array"]["path"], compartment["sourceArray"]["path"]))
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

    def execute(self, operation: str, args: dict, request: _Request, progress: Callable[[float, str], None]) -> Any:
        request.check()
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
                    unique.update(np.unique(np.asarray(annotation.dataobj[:, :, z:z + 8])).tolist())
                    if len(unique) > 33:
                        raise BridgeError("LABEL_COUNT_LIMIT", "Annotation has more than 32 labels or fractional values; use a reviewed annotation importer")
                count = math.prod(annotation.shape)
                if count * (4 + 2 * max(0, len(unique) - 1) + (2 if brain else 0)) > MAX_CASE_BYTES:
                    raise BridgeError("CASE_SIZE_LIMIT", "Expanded target masks exceed the desktop cache limit")
            progress(0.1, "Validating NIfTI coordinates and supplied annotations")
            case = load_nifti_case(structural, tumor, case_id=case_id, brain_mask_path=brain, label_map=label_map)
            return self._install_case(case, {}, request)
        if operation == "inspectEvidence":
            _keys(args, {"caseHash"})
            case = self._get_case(args.get("caseHash")).case
            return {"caseHash": case.semantic_hash, "sourceRefs": [source.to_dict() for source in case.source_refs],
                    "metadata": thaw_json(case.metadata), "unknowns": list(case.unknowns),
                    "patientContext": None if case.context is None else case.context.planning_view(),
                    "clinicalDeficitProbability": None, "clinicalRiskReason": "no_validated_clinical_outcome_model"}
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
        if operation == "nativeTraining":
            raise BridgeError("NOT_AVAILABLE", "Native training is not exposed through this bridge version")
        raise BridgeError("UNSUPPORTED_OPERATION", "Unknown sidecar operation")

    def close(self) -> None:
        self.cases.clear()
        self.transfers.close()


class BridgeRuntime:
    """One numerical worker; cancellation, timeouts and ping never wait on it."""

    def __init__(self, transfer_dir: Path, emit: Callable[[dict], None], *, max_cases: int = 2):
        self.session = BridgeSession(transfer_dir, max_cases=max_cases)
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
                               "operations": sorted(OPERATIONS), "tools": [asdict(tool) for tool in GENERIC_TOOLS]}})
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
        def progress(fraction: float, message: str) -> None:
            request.check()
            with self._lock:
                if not request.terminal:
                    self.emit({"id": request.request_id, "event": "progress", "progress": {"fraction": float(fraction), "message": message}})
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
    runtime = BridgeRuntime(args.transfer_dir, emit)
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
