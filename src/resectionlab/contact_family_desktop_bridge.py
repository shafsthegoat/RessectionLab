"""Fixed generated family catalog and one owned v3 episode for the desktop.

This module never accepts checkpoint paths, roles, models, or task arrays from
the renderer. The old near/costly v2 public-contact fixture is separate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import ModuleType
import uuid

from .contact_family_desktop_release import (
    ContactReleaseUnavailable, RELEASE_MANIFEST_SHA256, read_published_contact_release)
from .core import CaseData, SourceRef, semantic_digest
from .public_contact_family import (FAMILY_VERSION, GOAL_IDS, build_family_source,
                                    family_digest, family_manifest, family_record,
                                    layout_metadata)


CATALOG_VERSION = "generated-public-contact-learning-availability-v1"
EXPECTED_FAMILY_VERSION = "generated-public-contact-family-v2"
EXPECTED_PROPOSAL_VERSION = "fixed_lattice_access_centerline_v1"
SELECTORS = ("STOP", "SEARCH", "IL", "RL")
MAX_CONTROLLER_BYTES = 128 * 1024
# This binds the reviewed controller bytes. Its own closure checks the worker,
# published-release verifier, and canonical task/model source before release.
SUPERVISOR_SHA256 = "59f152ba79ac952e1c9270ea1c4776bb48b0c29ca790ef30551fd2d3d50c1495"


def _root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Fixed generated contact-family source checkout unavailable")


def _read_exact_supervisor():
    if SUPERVISOR_SHA256 is None:
        raise RuntimeError("No reviewed contact-family controller has been promoted")
    path = Path(__file__).with_name("contact_family_supervisor.py")
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CONTROLLER_BYTES:
        raise RuntimeError("Bound contact-family controller path changed")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SUPERVISOR_SHA256:
        raise RuntimeError("Bound contact-family controller source changed")
    module = ModuleType("fixed_reviewed_contact_family_supervisor")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def _controller_ready() -> bool:
    try:
        _read_exact_supervisor()._check_sources()
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _display_case(layout_id: str) -> tuple[CaseData, str]:
    # Exact source-grid display construction used by the v3 exporter. It has
    # no task transition and no private reference. The child-returned case
    # hash must equal this independently materialized DTO.
    source = build_family_source(layout_id)
    display = CaseData(f"{FAMILY_VERSION}:{layout_id}", source.structural_intensity,
        {"generated_nominal_target": source.nominal_target > 0}, source.affine_ras_mm,
        (SourceRef(layout_id, "generated://" + FAMILY_VERSION + "/" + layout_id,
                   native_frame="RAS+", provenance="simulated"),), frame="RAS+",
        brain_mask=source.observed_support,
        metadata={"is_synthetic": True, "evidence_kind": "generated_software_fixture",
            "patient_admission": False, "source_task_hash": source.source_hash,
            "family_hash": family_digest(), "layout_id": layout_id,
            "scope": "public_generated_contact_learning_no_patient_transfer"})
    return display, source.source_hash


def public_contact_family_availability() -> dict:
    family = family_record()
    if (FAMILY_VERSION != EXPECTED_FAMILY_VERSION or
            family.get("version") != EXPECTED_FAMILY_VERSION or
            family.get("proposal_mode") != EXPECTED_PROPOSAL_VERSION or
            family.get("source_candidate_version") != EXPECTED_PROPOSAL_VERSION):
        raise RuntimeError("Reviewed public contact-family v2 source is not installed")
    layouts = [{"layoutId": row["recipe"]["layout_id"], "role": row["role"],
                "goals": list(GOAL_IDS), "interactive": row["role"] in ("TRAIN", "SELECT")}
               for row in family["layouts"]]
    release = None
    experiment_hash = None
    controller_ready = _controller_ready()
    unavailable = "no_reviewed_final_32_update_pair_and_completed_pilot"
    if RELEASE_MANIFEST_SHA256 is not None:
        try:
            from .contact_learning_contract import freeze_contact_experiment
            experiment = freeze_contact_experiment()
            publication = read_published_contact_release(_root(), family_manifest=family_manifest(),
                                                           experiment_hash=experiment.fingerprint)
            release = "sha256:" + publication["manifestSha256"]
            experiment_hash = experiment.fingerprint
        except (ContactReleaseUnavailable, RuntimeError, ValueError, OSError):
            unavailable = "backend_contact_release_failed_verification"
    return {"version": CATALOG_VERSION, "fixture": FAMILY_VERSION,
        "familyHash": family_digest(), "experimentHash": experiment_hash,
        "releaseHash": release, "layouts": layouts,
        "methods": {method: {"available": controller_ready,
                             "reason": None if controller_ready else
                                       "backend_controller_not_promoted"}
            if method in ("STOP", "SEARCH")
            else {"available": release is not None and controller_ready,
                  "reason": (None if release is not None and controller_ready else
                             unavailable if release is None else
                             "backend_controller_not_promoted")} for method in SELECTORS}}


def execute_public_contact_family_episode(*, attempts_root: Path, layout_id: str,
                                          goal_id: str, selector: str, cancelled=None) -> dict:
    if (type(layout_id) is not str or type(goal_id) is not str or
            type(selector) is not str or goal_id not in GOAL_IDS or selector not in SELECTORS):
        raise ValueError("Choose an exact generated family layout, goal and selector")
    row = layout_metadata(layout_id)
    if row["role"] not in ("TRAIN", "SELECT"):
        raise ValueError("HELD_OUT_EXECUTION_CLOSED")
    if selector in ("IL", "RL"):
        if RELEASE_MANIFEST_SHA256 is None:
            raise ContactReleaseUnavailable("No reviewed final learned contact-family endpoint is available")
        if public_contact_family_availability()["releaseHash"] is None:
            raise ContactReleaseUnavailable("No reviewed final learned contact-family endpoint is available")
    supervisor = _read_exact_supervisor()
    result = supervisor.run_attempt(Path(attempts_root) / ("contact-family-" + uuid.uuid4().hex),
                                    layout_id=layout_id, goal_id=goal_id,
                                    selector=selector, cancelled=cancelled)
    episode = result.get("episode")
    case, source_hash = _display_case(layout_id)
    if (type(episode) is not dict or result.get("caseHash") != case.semantic_hash or
            result.get("layoutId") != layout_id or result.get("goalId") != goal_id or
            result.get("selector") != selector or result.get("splitRole") != row["role"] or
            episode.get("schema") != "resectionlab.shared-native-contact-learning-episode.v3" or
            episode.get("taskKind") != "generated_family_public_retained_surface_contact" or
            episode.get("fixture") != FAMILY_VERSION or episode.get("layoutId") != layout_id or
            episode.get("splitRole") != row["role"] or episode.get("selector") != selector or
            episode.get("publicGoal", {}).get("goalId") != goal_id or
            episode.get("caseHash") != case.semantic_hash or
            episode.get("sourceHash") != source_hash or
            episode.get("patientAdmission") is not False or episode.get("clinicalValidation") is not False):
        raise RuntimeError("Owned contact-family result differs from the fixed v3 native case")
    contract = episode.get("taskContract")
    declaration = contract.get("declaration") if type(contract) is dict else None
    if (type(declaration) is not dict or
            contract.get("proposalMode") != EXPECTED_PROPOSAL_VERSION or
            contract.get("sourceCandidateVersion") != EXPECTED_PROPOSAL_VERSION or
            declaration.get("proposal_mode") != EXPECTED_PROPOSAL_VERSION or
            declaration.get("source_candidate_version") != EXPECTED_PROPOSAL_VERSION):
        raise RuntimeError("Owned contact-family candidate rule differs from the reviewed v2 source")
    canonical = json.dumps({key: value for key, value in episode.items() if key != "episodeId"},
                           sort_keys=True, separators=(",", ":"), allow_nan=False)
    if result.get("episodeCanonicalJson") != canonical or semantic_digest(
            {key: value for key, value in episode.items() if key != "episodeId"}) != episode.get("episodeId"):
        raise RuntimeError("Contact-family episode canonical identity changed after child completion")
    response = {"case": case, "episode": episode, "episodeCanonicalJson": canonical}
    if selector in ("IL", "RL"):
        authorship, release = episode.get("learnedAuthorship"), result.get("releaseEvidence")
        if (type(authorship) is not dict or type(release) is not dict or
                authorship.get("method") != selector or
                type(authorship.get("completedUpdates")) is not int or
                authorship["completedUpdates"] != 32 or
                type(authorship.get("inferenceOptimizerUpdates")) is not int or
                authorship["inferenceOptimizerUpdates"] != 0 or
                authorship.get("checkpointFileSha256") != release.get("checkpointFileSha256") or
                authorship.get("experimentHash") != result.get("experimentHash") or
                authorship.get("familyHash") != result.get("familyHash")):
            raise RuntimeError("Owned contact-family learned provenance differs from final checkpoint")
        response["executionProvenance"] = {
            "version": "generated-contact-family-execution-v1", "selector": selector,
            "layoutId": layout_id, "goalId": goal_id, "splitRole": row["role"],
            "experimentHash": authorship["experimentHash"],
            "familyHash": authorship["familyHash"],
            "releaseManifestSha256": release["releaseManifestSha256"],
            "pilotResultSha256": release["pilotResultSha256"],
            "finalFreezeSha256": release["finalFreezeSha256"],
            "checkpointFileSha256": release["checkpointFileSha256"],
            "architectureHash": authorship["architectureHash"],
            "parameterHash": authorship["parameterHash"],
            "trainingLineageHash": authorship["trainingLineageHash"],
            "completedUpdates": 32, "inferenceOptimizerUpdates": 0,
            "ownedResultSha256": result["ownedResultSha256"],
            "ownedSupervisionSha256": result["ownedSupervisionSha256"]}
    elif episode.get("learnedAuthorship") is not None or result.get("releaseEvidence") is not None:
        raise RuntimeError("SEARCH/STOP contact-family result falsely claims checkpoint authorship")
    return response
