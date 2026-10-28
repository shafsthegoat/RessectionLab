#!/usr/bin/env python3
"""Declare, seal, then explicitly acquire one UCSF MRI-only assessment case.

Declare and seal inspect metadata/checkpoints only. ``acquire --pixel-go`` is a
separate, deliberately gated command for use after the development app flows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import copy
import traceback
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CASE = "UCSF-PDGM-0008"
REVISION = "e9372219cf1cd2fdd52260cd45f7514b4aa7638e"
INVENTORY = ROOT / "testcases/outputs/integrated-goal-20261026/ucsf-mirror-root.json"
INVENTORY_SHA256 = "578078ff5a5c658d619b0665004948bebaf23169a75b2b03dfa657014ac14ea8"
DEVELOPMENT = ROOT / "testcases/downloaded-datasets/UCSF-fresh-development-20261026/development-declaration.json"
ROSTER = tuple(f"{stage}-seed{seed}.pt" for seed in (17, 29, 43) for stage in ("bc", "ppo"))
CHOSEN = "ppo-seed17.pt"
ACCESS_METHOD = "axis0-access"
SOURCE_SUBDIR = "downloaded-datasets/UCSF-frozen-assessment-20261027"
OUTPUT_SUBDIR = "outputs/integrated-goal-20261026/frozen-assessment"


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path: Path):
    return json.loads(Path(path).read_text())


def _write_once(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def source_root(package_root: Path) -> Path:
    return Path(package_root) / SOURCE_SUBDIR


def output_root(package_root: Path) -> Path:
    return Path(package_root) / OUTPUT_SUBDIR


def protocol_path(package_root: Path) -> Path:
    return source_root(package_root) / "protocol.json"


def freeze_path(package_root: Path) -> Path:
    return source_root(package_root) / "checkpoint-freeze.json"


def _inventory_candidate(inventory: Path, development: Path) -> dict:
    if digest(inventory) != INVENTORY_SHA256:
        raise ValueError("Pinned mirror directory inventory changed")
    declaration = _json(development)
    if (declaration.get("role") != "DEVELOPMENT" or
            declaration.get("mirror_revision") != REVISION or
            declaration.get("cases") != ["UCSF-PDGM-0005", "UCSF-PDGM-0007"]):
        raise ValueError("Existing development roles or mirror revision changed")
    previously_used = {"UCSF-PDGM-0004", *declaration["cases"]}
    directories = sorted((row for row in _json(inventory)
                          if row.get("type") == "directory" and
                          row.get("path", "").startswith("UCSF-PDGM-")),
                         key=lambda row: row["path"])
    first = next((row for row in directories if row["path"] not in previously_used), None)
    if first is None or first["path"] != CASE or not first.get("oid"):
        raise ValueError("Prospective first unused mirror folder is not the declared case")
    return first


def declare(package_root: Path, *, inventory: Path = INVENTORY,
            development: Path = DEVELOPMENT) -> Path:
    """Create the immutable patient/role/analysis rule before any patient bytes."""
    candidate = _inventory_candidate(inventory, development)
    path = protocol_path(package_root)
    if path.exists():
        validate_protocol(package_root, inventory=inventory, development=development)
        return path
    if (source_root(package_root) / CASE).exists():
        raise ValueError("Assessment source directory already exists before protocol")
    protocol = {
        "schema": "resectionlab.ucsf-frozen-assessment-protocol.v1",
        "declaredAt": datetime.now(timezone.utc).isoformat(),
        "declarationState": "pending_no_pixels",
        "patientId": CASE,
        "role": "FROZEN_ASSESSMENT",
        "trainingProhibited": True,
        "checkpointSelectionProhibitedOnAssessment": True,
        "routeTuningProhibitedOnAssessment": True,
        "replacementBasedOnOutcomesProhibited": True,
        "mirrorRepository": "MedOtter/UCSF-PDGM",
        "mirrorRevision": REVISION,
        "inventoryPath": str(Path(inventory).resolve()),
        "inventorySha256": INVENTORY_SHA256,
        "directoryOid": candidate["oid"],
        "developmentDeclarationPath": str(Path(development).resolve()),
        "developmentDeclarationSha256": digest(development),
        "selectionRule": "First ascending pinned UCSF mirror directory ID excluding previously used 0004 and predeclared DEVELOPMENT 0005/0007; fixed before image access",
        "checkpointRosterRule": "Exactly final BC and BC-initialized PPO checkpoints for paired DEVELOPMENT fit seeds 17/29/43",
        "checkpointRoster": list(ROSTER),
        "primaryCheckpointRule": "Final paired-development PPO seed 17, chosen before assessment pixels or outcomes",
        "primaryCheckpoint": CHOSEN,
        "accessMethodRule": "Use saved axis0-access route; no assessment-outcome route ranking",
        "accessMethod": ACCESS_METHOD,
        "nativeBudget": {"preparationMaxPhysicalCalls": 1500,
                         "preparationMaxWallSeconds": 120,
                         "perExecutionMaxPhysicalCalls": 512,
                         "perExecutionMaxWallSeconds": 600,
                         "assessmentAggregateMaxPhysicalCalls": 10000,
                         "assessmentAggregateMaxWallSeconds": 900},
        "sourceScope": "skull-stripped, co-registered T1/T1c/T2/FLAIR plus supplied tumor segmentation",
        "missingAnatomy": ["skull", "vessels", "angiography", "functional mapping", "tractography"],
        "pretrainingOverlap": "unknown",
        "clinicalIndependence": "unknown",
        "officialTciaByteEquivalence": "unknown",
        "claimLimit": "MRI structural partial-scope transfer only; no full-anatomy or clinical validation",
        "imageFetchCapBytes": 60_000_000,
    }
    return _write_once(path, protocol)


def validate_protocol(package_root: Path, *, inventory: Path = INVENTORY,
                      development: Path = DEVELOPMENT) -> dict:
    candidate = _inventory_candidate(inventory, development)
    protocol = _json(protocol_path(package_root))
    expected = {"schema": "resectionlab.ucsf-frozen-assessment-protocol.v1",
                "patientId": CASE, "role": "FROZEN_ASSESSMENT",
                "declarationState": "pending_no_pixels",
                "mirrorRepository": "MedOtter/UCSF-PDGM",
                "mirrorRevision": REVISION, "inventorySha256": INVENTORY_SHA256,
                "directoryOid": candidate["oid"],
                "developmentDeclarationSha256": digest(development),
                "checkpointRoster": list(ROSTER), "primaryCheckpoint": CHOSEN,
                "accessMethod": ACCESS_METHOD,
                "imageFetchCapBytes": 60_000_000,
                "nativeBudget": {"preparationMaxPhysicalCalls": 1500,
                                 "preparationMaxWallSeconds": 120,
                                 "perExecutionMaxPhysicalCalls": 512,
                                 "perExecutionMaxWallSeconds": 600,
                                 "assessmentAggregateMaxPhysicalCalls": 10000,
                                 "assessmentAggregateMaxWallSeconds": 900}}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen assessment protocol fields changed")
    if any(protocol.get(key) is not True for key in (
            "trainingProhibited", "checkpointSelectionProhibitedOnAssessment",
            "routeTuningProhibitedOnAssessment", "replacementBasedOnOutcomesProhibited")):
        raise ValueError("Assessment role restrictions changed")
    if (Path(protocol.get("inventoryPath", "")).resolve() != Path(inventory).resolve() or
            Path(protocol.get("developmentDeclarationPath", "")).resolve() != Path(development).resolve()):
        raise ValueError("Assessment protocol source paths changed")
    return protocol


def seal(package_root: Path, fit_dir: Path) -> Path:
    """Bind all six final paired-DEV weights and current producer before pixels."""
    protocol = validate_protocol(package_root)
    path = freeze_path(package_root)
    if path.exists():
        raise FileExistsError("Assessment checkpoint freeze is immutable")
    if (source_root(package_root) / CASE).exists():
        raise ValueError("Assessment pixels or source directory existed before checkpoint seal")
    from run_patient_sector_learning import producer_identity
    from resectionlab.core import semantic_digest
    from resectionlab.patient_sector_learning import frozen_actor, parameter_hash
    import torch

    fit_dir = Path(fit_dir).resolve()
    fit_receipt_path = fit_dir / "receipt.json"
    fit = _json(fit_receipt_path)
    producer = producer_identity()
    if (fit.get("status") != "saved" or fit.get("mode") != "fit" or
            fit.get("sourceRole") != "DEVELOPMENT" or
            fit.get("ppoSourceSchedule") != "paired-development" or
            fit.get("producerIdentity") != producer or
            fit.get("seeds") != [17, 29, 43]):
        raise ValueError("Single paired DEVELOPMENT fit or current producer mismatch")
    sources = fit.get("sources", {})
    if any(sources.get(key, {}).get("roleDeclaration", {}).get("patientId") != patient or
           sources.get(key, {}).get("roleDeclaration", {}).get("role") != "DEVELOPMENT" or
           sources.get(key, {}).get("roleDeclaration", {}).get("mirrorRevision") != REVISION
           for key, patient in (("train", "UCSF-PDGM-0007"),
                                ("transfer", "UCSF-PDGM-0005"))):
        raise ValueError("Fit source roles differ from declared development cases")
    source_bindings = {key: _source_binding_from_fit(fit, key)
                       for key in ("train", "transfer")}
    episode_plan = fit.get("ppoEpisodePlan")
    if episode_plan != [
            {"update": update, "episode": episode,
             "sourceKey": "train" if episode == 0 else "transfer"}
            for update in range(2) for episode in range(2)]:
        raise ValueError("Paired DEVELOPMENT episode assignment plan changed")
    runs = fit.get("seedRuns", [])
    if len(runs) != 3 or [run.get("seed") for run in runs] != [17, 29, 43]:
        raise ValueError("Fit must retain the prespecified three seed runs")
    checkpoints = {}
    for run in runs:
        for stage in ("bc", "ppo"):
            name = f"{stage}-seed{run['seed']}.pt"
            model = run.get(stage, {}).get("model", {})
            file = fit_dir / name
            if (not file.is_file() or Path(model.get("path", "")).resolve() != file or
                    model.get("fileSha256") != digest(file)):
                raise ValueError(f"Fit checkpoint path or hash differs: {name}")
            expected_stage = "bc" if stage == "bc" else "bc-initialized-ppo"
            payload = torch.load(file, map_location="cpu", weights_only=True)
            proof = payload.get("trainingReceipt", {})
            if (model.get("stage") != expected_stage or
                    payload.get("stage") != expected_stage or
                    payload.get("sourceBinding") != model.get("sourceBinding") or
                    payload.get("sourceBinding") != source_bindings["train"] or
                    proof.get("sourceBinding") != source_bindings["train"] or
                    semantic_digest(proof) !=
                        model.get("trainingReceiptSha256") or
                    (stage == "ppo" and not _ppo_proof_matches(
                        proof, run, source_bindings, episode_plan))):
                raise ValueError(f"Fit checkpoint stage or paired exposure differs: {name}")
            actor = frozen_actor(file, expected_producer=producer)
            if parameter_hash(actor) != model.get("actorSha256"):
                raise ValueError(f"Fit actor hash differs: {name}")
            checkpoints[name] = {"path": str(file), "fileSha256": digest(file),
                "actorSha256": model["actorSha256"], "stage": model["stage"],
                "trainingReceiptSha256": model["trainingReceiptSha256"]}
    if set(checkpoints) != set(protocol["checkpointRoster"]):
        raise ValueError("Six-checkpoint roster differs")
    freeze = {"schema": "resectionlab.ucsf-assessment-checkpoint-freeze.v1",
        "sealedAt": datetime.now(timezone.utc).isoformat(),
        "protocolPath": str(protocol_path(package_root).resolve()),
        "protocolSha256": digest(protocol_path(package_root)),
        "fitReceiptPath": str(fit_receipt_path), "fitReceiptSha256": digest(fit_receipt_path),
        "producerIdentity": producer, "checkpoints": checkpoints,
        "primaryCheckpoint": CHOSEN, "accessMethod": ACCESS_METHOD,
        "modelSelectionUsedAssessmentPixels": False,
        "optimizerUpdatesAfterFreeze": 0}
    return _write_once(path, freeze)


def _source_binding_from_fit(fit: dict, key: str) -> dict:
    """Read only an existing DEV archive manifest, never image array members."""
    source = fit["sources"][key]
    path = Path(source.get("path", ""))
    if not path.is_file() or path.suffix != ".ressectionlab":
        raise ValueError(f"Paired DEVELOPMENT {key} archive missing")
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    record = manifest.get("artifacts", {}).get("surgical_toolbox", {})
    planning = record.get("planning", {})
    alternatives = [row for row in planning.get("alternatives", [])
                    if row.get("method") == source.get("accessMethod")]
    comparison_seal = planning.get("comparisonSeal")
    if (manifest.get("case_id") != source.get("caseId") or
            digest(path) != source.get("fileSha256") or
            planning.get("sourceSceneId") != source.get("sceneId") or
            len(alternatives) != 1 or
            alternatives[0].get("planSeal") != source.get("accessPlanSeal") or
            not comparison_seal or
            record.get("sectorPlanning", {}).get("sourceAccessComparisonSeal") != comparison_seal or
            set(source.get("sectorCellSha256", {})) != {"near", "far"} or
            not all(isinstance(value, str) and len(value) == 64
                    for value in source["sectorCellSha256"].values()) or
            not isinstance(source.get("fileSha256"), str) or
            len(source["fileSha256"]) != 64):
        raise ValueError(f"Paired DEVELOPMENT {key} archive/scene/access/sector binding differs")
    return {"sourceArchiveSha256": source["fileSha256"],
        "sceneId": source["sceneId"],
        "accessComparisonSeal": comparison_seal,
        "accessPlanSeal": source["accessPlanSeal"],
        "accessMethod": source["accessMethod"],
        "sectorCellSha256": source["sectorCellSha256"],
        "taskGate": fit["taskGate"]}


def _ppo_proof_matches(proof: dict, run: dict, bindings: dict,
                       episode_plan: list) -> bool:
    assignments = proof.get("sampledEpisodeAssignments")
    if (proof.get("ppoSourceSchedule") != "paired-development" or
            proof.get("optimizationSourceBindings") != bindings or
            proof.get("ppoEpisodePlan") != episode_plan or
            assignments != run.get("ppoSampledEpisodeAssignments") or
            not isinstance(assignments, list) or len(assignments) != 4):
        return False
    return all(
        {field: actual.get(field) for field in ("update", "episode", "sourceKey")}
            == planned and
        actual.get("sourceArchiveSha256") ==
            bindings[planned["sourceKey"]]["sourceArchiveSha256"] and
        actual.get("optimizationRole") == "DEV_TRAINED_FOR_PPO"
        for actual, planned in zip(assignments, episode_plan))


def validate_freeze(package_root: Path) -> tuple[dict, dict]:
    protocol = validate_protocol(package_root)
    freeze = _json(freeze_path(package_root))
    if (freeze.get("schema") != "resectionlab.ucsf-assessment-checkpoint-freeze.v1" or
            freeze.get("protocolSha256") != digest(protocol_path(package_root)) or
            freeze.get("primaryCheckpoint") != CHOSEN or
            freeze.get("accessMethod") != ACCESS_METHOD or
            freeze.get("modelSelectionUsedAssessmentPixels") is not False or
            freeze.get("optimizerUpdatesAfterFreeze") != 0 or
            Path(freeze.get("protocolPath", "")).resolve() != protocol_path(package_root).resolve()):
        raise ValueError("Assessment checkpoint freeze changed")
    fit_path = Path(freeze.get("fitReceiptPath", ""))
    if not fit_path.is_file() or digest(fit_path) != freeze.get("fitReceiptSha256"):
        raise ValueError("Sealed paired DEVELOPMENT fit receipt changed")
    from run_patient_sector_learning import producer_identity
    from resectionlab.patient_sector_learning import frozen_actor, parameter_hash
    producer = producer_identity()
    if producer != freeze.get("producerIdentity"):
        raise ValueError("Current producer differs from frozen checkpoint producer")
    checkpoints = freeze.get("checkpoints", {})
    if set(checkpoints) != set(ROSTER):
        raise ValueError("Sealed six-checkpoint roster changed")
    for name in ROSTER:
        row = checkpoints[name]
        path = Path(row.get("path", ""))
        expected_stage = "bc" if name.startswith("bc-") else "bc-initialized-ppo"
        if (path.name != name or path.resolve() != fit_path.parent / name or
                not path.is_file() or digest(path) != row.get("fileSha256") or
                row.get("stage") != expected_stage):
            raise ValueError(f"Sealed checkpoint changed: {name}")
        if parameter_hash(frozen_actor(path, expected_producer=producer)) != row.get("actorSha256"):
            raise ValueError(f"Sealed actor changed: {name}")
    return protocol, freeze


def acquire_and_prepare(package_root: Path, *, pixel_go: bool, infer_brain: bool) -> dict:
    if pixel_go is not True:
        raise ValueError("Explicit PIXEL GO is required after both DEVELOPMENT app flows")
    if infer_brain is not True:
        raise ValueError("Frozen assessment preparation requires the existing brain inference path")
    protocol, _ = validate_freeze(package_root)
    import prepare_fresh_ucsf_demo as fresh
    fresh.SOURCE_ROOT = source_root(package_root)
    fresh.OUTPUT_ROOT = output_root(package_root)
    declaration = {"role": "FROZEN_ASSESSMENT", "mirror_revision": REVISION,
                   "image_fetch_cap_bytes": protocol["imageFetchCapBytes"],
                   "role_declaration_sha256": digest(protocol_path(package_root)),
                   "checkpoint_freeze_sha256": digest(freeze_path(package_root))}
    receipt_path = source_root(package_root) / CASE / "source-receipt.json"
    report_path = output_root(package_root) / CASE / "preparation-receipt.json"
    if report_path.exists():
        return validate_acquired_role(package_root)[1]
    if receipt_path.exists():
        receipt = _json(receipt_path)
        if (receipt.get("role") != "FROZEN_ASSESSMENT" or
                receipt.get("patient_id") != CASE or
                receipt.get("mirror_revision") != REVISION or
                receipt.get("role_declaration_sha256") != digest(protocol_path(package_root)) or
                receipt.get("checkpoint_freeze_sha256") != digest(freeze_path(package_root)) or
                len(receipt.get("files", [])) != 5 or
                any(digest(ROOT / row["path"]) != row["sha256"]
                    for row in receipt["files"])):
            raise ValueError("Existing frozen source receipt or bytes changed")
    else:
        receipt = fresh.acquire(CASE, declaration)
    if (receipt["role"] != "FROZEN_ASSESSMENT" or receipt["patient_id"] != CASE or
            receipt.get("role_declaration_sha256") != digest(protocol_path(package_root)) or
            receipt.get("checkpoint_freeze_sha256") != digest(freeze_path(package_root))):
        raise ValueError("Frozen source receipt lost its predeclared role")
    _reject_duplicate_development_bytes(receipt)
    report = fresh.prepare(CASE, receipt, infer_brain=infer_brain)
    if report["source_role"] != "FROZEN_ASSESSMENT":
        raise ValueError("Frozen preparation receipt lost its predeclared role")
    return report


class NativeCallCounter:
    """Count original public native calls, including raw restore/reopen replay.

    One preview costs one call; its internal step is observed but not charged
    twice. This does not patch an engine method or its producer fingerprint.
    """
    def __init__(self, max_calls: int, max_seconds: float):
        from resectionlab.surgical_engine import SurgicalEngine
        self.preview_code = SurgicalEngine.preview.__code__
        self.step_code = SurgicalEngine.step.__code__
        self.max_calls = max_calls
        self.max_seconds = max_seconds
        self.started = time.monotonic()
        self.preview_calls = 0
        self.direct_steps = 0
        self.internal_steps = 0
        self.phase = "unlabeled"
        self.phases = {}
        self.execution_started = None
        self.execution_calls_started = None
        self.execution_max_calls = None
        self.execution_max_seconds = None
        self._prior = None

    @property
    def calls(self):
        return self.preview_calls + self.direct_steps

    def check(self):
        self.check_time()
        if self.calls >= self.max_calls:
            raise TimeoutError("Frozen assessment public native-call budget exhausted")
        if (self.execution_started is not None and
                self.calls - self.execution_calls_started >= self.execution_max_calls):
            raise TimeoutError("Frozen assessment execution native-call budget exhausted")

    def check_time(self):
        """Check wall ceilings after non-native export/save/reopen work too."""
        if time.monotonic() - self.started > self.max_seconds:
            raise TimeoutError("Frozen assessment wall budget exhausted")
        if self.execution_started is not None:
            if time.monotonic() - self.execution_started > self.execution_max_seconds:
                raise TimeoutError("Frozen assessment execution wall budget exhausted")

    def begin_execution(self, *, max_calls=512, max_seconds=600):
        if self.execution_started is not None:
            raise RuntimeError("Prior execution is still metered")
        self.execution_started = time.monotonic()
        self.execution_calls_started = self.calls
        self.execution_max_calls = max_calls
        self.execution_max_seconds = max_seconds

    def end_execution(self):
        if self.execution_started is None:
            raise RuntimeError("No execution meter is active")
        result = {"physicalCalls": self.calls-self.execution_calls_started,
                  "wallSeconds": time.monotonic()-self.execution_started,
                  "maxPhysicalCalls": self.execution_max_calls,
                  "maxWallSeconds": self.execution_max_seconds}
        self.execution_started = self.execution_calls_started = None
        self.execution_max_calls = self.execution_max_seconds = None
        return result

    def _profile(self, frame, event, arg):
        if event != "call":
            return
        if frame.f_code is self.preview_code:
            self.check()
            self.preview_calls += 1
            self.phases.setdefault(self.phase, {"previewCalls": 0, "directStepCalls": 0})["previewCalls"] += 1
        elif frame.f_code is self.step_code:
            if frame.f_back is not None and frame.f_back.f_code is self.preview_code:
                self.internal_steps += 1
            else:
                self.check()
                self.direct_steps += 1
                self.phases.setdefault(self.phase, {"previewCalls": 0, "directStepCalls": 0})["directStepCalls"] += 1

    def __enter__(self):
        self._prior = sys.getprofile()
        if self._prior is not None:
            raise RuntimeError("Another native profiler is already installed")
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *_):
        sys.setprofile(self._prior)

    def report(self):
        return {"previewCalls": self.preview_calls,
                "directStepCalls": self.direct_steps,
                "previewInternalStepCallsExcluded": self.internal_steps,
                "physicalCalls": self.calls, "phases": self.phases,
                "wallSeconds": time.monotonic() - self.started,
                "maxPhysicalCalls": self.max_calls,
                "maxWallSeconds": self.max_seconds}


def validate_acquired_role(package_root: Path) -> tuple[dict, dict]:
    """Reject a DEVELOPMENT receipt or changed source bytes before scene use."""
    protocol, _ = validate_freeze(package_root)
    receipt_path = source_root(package_root) / CASE / "source-receipt.json"
    preparation_path = output_root(package_root) / CASE / "preparation-receipt.json"
    receipt, preparation = _json(receipt_path), _json(preparation_path)
    if (receipt.get("role") != "FROZEN_ASSESSMENT" or
            receipt.get("patient_id") != CASE or
            receipt.get("mirror_revision") != REVISION or
            receipt.get("role_declaration_sha256") != digest(protocol_path(package_root)) or
            receipt.get("checkpoint_freeze_sha256") != digest(freeze_path(package_root)) or
            preparation.get("source_role") != "FROZEN_ASSESSMENT" or
            preparation.get("patient_id") != CASE or
            preparation.get("checkpoint_freeze_sha256") != digest(freeze_path(package_root)) or
            preparation.get("scene_reopened") is not True or
            set(preparation.get("missing_anatomy", [])) != set(protocol["missingAnatomy"])):
        raise ValueError("Acquired source or preparation lost frozen assessment identity")
    files = receipt.get("files", [])
    if len(files) != 5:
        raise ValueError("Frozen source must retain four MRI and one label source")
    _reject_duplicate_development_bytes(receipt)
    for row in files:
        path = ROOT / row.get("path", "")
        if (not path.absolute().is_relative_to((source_root(package_root) / CASE).absolute()) or
                not path.is_file() or digest(path) != row.get("sha256")):
            raise ValueError("Frozen source image bytes changed")
    return receipt, preparation


def _reject_duplicate_development_bytes(receipt: dict) -> None:
    frozen_hashes = sorted(row["sha256"] for row in receipt.get("files", []))
    for patient in ("UCSF-PDGM-0005", "UCSF-PDGM-0007"):
        path = DEVELOPMENT.parent / patient / "source-receipt.json"
        if not path.is_file():
            raise ValueError("DEVELOPMENT source receipt missing for distinct-anatomy audit")
        if frozen_hashes == sorted(row["sha256"] for row in _json(path).get("files", [])):
            raise ValueError("Frozen patient source bytes duplicate a DEVELOPMENT case")


def prepare_access_comparison(package_root: Path, output: Path, *, pixel_go: bool) -> dict:
    """Source-only joint shortlist, then at most three native access routes.

    Sector options and model decisions are intentionally not evaluated here.
    The saved access comparison is the standard prepare_case-compatible shape.
    """
    if pixel_go is not True:
        raise ValueError("Explicit PIXEL GO is required for patient scene access")
    started = time.monotonic()
    protocol, freeze = validate_freeze(package_root)
    validate_acquired_role(package_root)
    from resectionlab.core import array_digest, semantic_digest
    from resectionlab.surgical_access_planning import AccessNotExecutable, _execute, access_availability
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_environment import load_environment
    from resectionlab.surgical_joint_geometry import select_joint_corridors
    from resectionlab.surgical_planning import validate_saved_comparison
    from resectionlab.surgical_transition_meter import TransitionBudgetExceeded, TransitionMeter
    from resectionlab.workspace_bundle import load_workspace, save_workspace

    output = Path(output).resolve()
    source = output_root(package_root) / CASE / "scene/patient.ressectionlab"
    if output.exists() or not output.is_relative_to((ROOT / "testcases").resolve()) or not source.is_file():
        raise ValueError("Choose a new ignored output and an acquired frozen assessment scene")
    output.mkdir(parents=True)
    receipt = {"schema": "resectionlab.frozen-assessment-access-preparation.v1",
        "status": "running", "patientId": CASE, "sourceRole": "FROZEN_ASSESSMENT",
        "source": str(source.resolve()), "sourceSha256": digest(source),
        "protocolSha256": digest(protocol_path(package_root)),
        "freezeSha256": digest(freeze_path(package_root)),
        "optimizerUpdates": 0}
    counter = NativeCallCounter(1500, 120.)
    counter.started = started
    meter = None
    geometry = None
    try:
        with counter:
            counter.phase = "source-open"
            case, artifacts, workspace = load_workspace(source)
            scene = load_environment(source)
            record = artifacts.get("surgical_toolbox")
            if (record is None or record.get("branchId") not in record.get("branches", {})
                    or record.get("planning") is not None or
                    record["branches"][record["branchId"]].get("actions") != []):
                raise ValueError("Assessment source must be an intact unplanned scene")
            adapter = case.metadata.get("surgical_adapter", {})
            source_manifest = adapter.get("sourceManifest", {})
            if (adapter.get("sourcePatient") != CASE or
                    source_manifest.get("patient_id") != CASE or
                    {r.get("source_mirror_revision") for r in source_manifest.get("series", [])} != {REVISION}):
                raise ValueError("Assessment scene patient or mirror revision differs")
            intact = SurgicalEngine.restore(scene, record["branches"][record["branchId"]],
                                            check=counter.check)
            if intact.actions or intact.state_id != intact.fresh().state_id:
                raise ValueError("Assessment access starts at intact anatomy")
            readiness = access_availability(intact)
            if not readiness["accessAvailable"] or readiness["skullAvailable"]:
                raise ValueError("Frozen MRI-only source does not support mapped brain access")
            counter.phase = "source-only-shortlist"
            routes, geometry = select_joint_corridors(scene, check=counter.check)
            selection_file = output / "source-only-selection.json"
            selection_file.write_text(
                json.dumps({"geometry": geometry, "routes": routes}, indent=2, allow_nan=False)+"\n")
            receipt["sourceOnlySelectionFileSha256"] = digest(selection_file)
            methods = {2: "superior-access", 0: "axis0-access", 1: "axis1-access"}
            labels = {2: "Superior access", 0: "Native axis 0 access", 1: "Native axis 1 access"}
            # At most six retained success/failure branches can require 288
            # replay steps after save. Keep those inside the 1,500-call cap.
            meter = TransitionMeter(1200, check=counter.check)
            base = meter.wrap(intact)
            branches, failures, failed_branches = [], [], []
            for axis in (2, 0, 1):
                route = next((r for r in routes if r["nativeAxis"] == axis), None)
                method = methods[axis]
                if route is None:
                    failures.append({"nativeAxis": axis, "method": method,
                                     "reason": "SOURCE_ONLY_SHORTLIST_NO_DISTINCT_ROUTE"})
                    continue
                try:
                    counter.phase = "native-access-" + method
                    with meter.phase(counter.phase, reserve_calls=150):
                        candidate, plan = _execute(base, {}, method, labels[axis], route,
                            max_actions=48, skull_available=False, healthy_cap=200.,
                            check=counter.check)
                    plan["branchId"] = "frozen-access-" + method
                    branches.append((candidate.unwrap(), plan))
                except (AccessNotExecutable, TransitionBudgetExceeded, TimeoutError) as error:
                    failure = {"nativeAxis": axis, "method": method,
                               "reason": type(error).__name__ + ": " + str(error),
                               "status": "unpassed_access"}
                    preserved = meter.last_committed_engine(phase=counter.phase)
                    if preserved is not None and preserved.actions:
                        failure.update(branchId="frozen-failed-access-" + method,
                                       finalStateId=preserved.state_id,
                                       nativeActions=len(preserved.actions),
                                       cleanupComplete=not any(
                                           slot["inserted"] or slot["active"]
                                           for slot in preserved.state["slots"].values()))
                        failed_branches.append((preserved, failure))
                    failures.append(failure)
                    if isinstance(error, (TransitionBudgetExceeded, TimeoutError)):
                        break
            for axis in (2, 0, 1):
                if axis not in {plan["route"]["nativeAxis"] for _, plan in branches} and axis not in {
                        item["nativeAxis"] for item in failures}:
                    failures.append({"nativeAxis": axis, "method": methods[axis],
                                     "reason": "AGGREGATE_BUDGET_STOPPED"})
            access = {"schema": "resectionlab.surgical-planning.v1", "task": "anatomy-access",
                "status": "three_source_tissue_routes_skull_unsupported" if len(branches) == 3
                          else "insufficient_distinct_routes",
                "requiredAlternatives": 3,
                "directionFailures": failures,
                "sourceSceneId": scene.identity, "startStateId": intact.state_id,
                "sourceLabelsHash": array_digest(scene.labels),
                "guardContractId": intact.guard_report()["contractId"],
                "sourceRole": "FROZEN_ASSESSMENT source-only joint shortlist",
                "allowModeledBrainCorridor": True, "availability": readiness,
                "optimizerUpdates": 0,
                "limits": {"maxActionsPerApproach": 48,
                           "maxAdditionalHealthyRemovedMm3": 200.},
                "limitations": ["Skull, vessels and function absent; no clinical access claim.",
                                "Source-only ranking precedes all native outcomes."],
                "alternatives": [plan for _, plan in branches]}
            access["comparisonSeal"] = semantic_digest(access)
            saved = copy.deepcopy(record)
            for engine, plan in branches:
                saved["branches"][plan["branchId"]] = engine.snapshot()
                saved.setdefault("parents", {})[plan["branchId"]] = {
                    "branchId": record["branchId"], "position": 0}
            for engine, failure in failed_branches:
                saved["branches"][failure["branchId"]] = engine.snapshot()
                saved.setdefault("parents", {})[failure["branchId"]] = {
                    "branchId": record["branchId"], "position": 0}
            saved["planning"] = access
            saved["frozenFailedAccessExecutions"] = [failure for _, failure in failed_branches]
            validate_saved_comparison(scene, saved, access)
            arts = copy.deepcopy(artifacts)
            arts["surgical_toolbox"] = saved
            destination = output / "frozen-access-comparison.ressectionlab"
            counter.phase = "save-reopen-validation"
            if workspace is None:
                from resectionlab.imaging import save_case
                save_case(case, destination, artifacts=arts)
            else:
                save_workspace(case, destination, sources=workspace["sources"],
                    imaging=workspace["imagingState"], episode=workspace["episode"],
                    selection=workspace["selection"], originals=workspace["originals"],
                    artifacts=arts)
            reopened_case, reopened_artifacts, _ = load_workspace(destination)
            reopened_scene = load_environment(destination)
            validate_saved_comparison(reopened_scene,
                reopened_artifacts["surgical_toolbox"],
                reopened_artifacts["surgical_toolbox"]["planning"])
            if reopened_case.semantic_hash != case.semantic_hash:
                raise ValueError("Assessment case changed after save/reopen")
            for engine, plan in [*branches, *failed_branches]:
                branch_id = plan["branchId"]
                replay = SurgicalEngine.restore(reopened_scene,
                    reopened_artifacts["surgical_toolbox"]["branches"][branch_id],
                    check=counter.check)
                if replay.state_id != engine.state_id or replay.actions != engine.actions:
                    raise ValueError("Assessment access branch failed exact replay")
            if time.monotonic() - counter.started > counter.max_seconds:
                raise TimeoutError("Frozen assessment access preparation wall budget exhausted")
            receipt.update(status="saved", sourceSceneId=scene.identity,
                selectedRouteIds=[r["id"] for r in routes],
                executableMethods=[plan["method"] for _, plan in branches],
                failedExecutions=[failure for _, failure in failed_branches],
                directionFailures=failures,
                assessmentRouteSupported=any(plan["method"] == ACCESS_METHOD for _, plan in branches),
                accessComparisonSeal=access["comparisonSeal"],
                output=str(destination), outputSha256=digest(destination),
                sourceOnlySelectionSeal=geometry["selectionSeal"],
                meter=meter.report())
    except Exception:
        receipt.update(status="failed", failure=traceback.format_exc())
    finally:
        receipt["nativeCalls"] = counter.report()
        if meter is not None:
            receipt["meter"] = meter.report()
        if geometry is not None:
            receipt["sourceOnlySelectionSeal"] = geometry["selectionSeal"]
        receipt["sourceUnchanged"] = digest(source) == receipt["sourceSha256"]
        receipt["freezeUnchanged"] = digest(freeze_path(package_root)) == receipt["freezeSha256"]
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "seal", "acquire", "prepare-access"))
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--package-root", type=Path, default=ROOT / "testcases")
    parser.add_argument("--pixel-go", action="store_true")
    parser.add_argument("--infer-brain", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    if not package_root.is_relative_to((ROOT / "testcases").resolve()):
        parser.error("Package root must remain under ignored testcases/")
    if args.command == "declare":
        if args.fit_dir or args.pixel_go or args.infer_brain or args.output:
            parser.error("Declaration is metadata only")
        result = declare(package_root)
    elif args.command == "seal":
        if args.fit_dir is None or args.pixel_go or args.infer_brain or args.output:
            parser.error("Seal requires only --fit-dir and reads no patient pixels")
        result = seal(package_root, args.fit_dir)
    elif args.command == "acquire":
        if args.fit_dir or not args.pixel_go or not args.infer_brain or args.output:
            parser.error("Acquisition requires --pixel-go --infer-brain and no fit directory")
        result = acquire_and_prepare(package_root, pixel_go=args.pixel_go,
                                     infer_brain=args.infer_brain)
    else:
        if args.fit_dir or not args.pixel_go or args.infer_brain or args.output is None:
            parser.error("Access preparation requires --pixel-go and a new --output")
        result = prepare_access_comparison(package_root, args.output,
                                           pixel_go=args.pixel_go)
    summary = {"command": args.command}
    if isinstance(result, Path):
        summary.update(status="saved", path=str(result))
    elif args.command == "prepare-access":
        summary.update(status=result["status"],
            receipt=str(args.output.resolve() / "receipt.json"),
            assessmentRouteSupported=result.get("assessmentRouteSupported"))
    else:
        summary.update(status="prepared", patientId=result.get("patient_id"),
            receipt=str(output_root(package_root) / CASE / "preparation-receipt.json"))
    print(json.dumps(summary, allow_nan=False))
    if isinstance(result, dict) and result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
