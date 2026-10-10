"""One owned generated contact-family episode; never trains or reads patients.

The original pair enables IL/RL; a separate audited TRAIN-only refit enables IL_TRAIN_REFIT.
SEARCH and STOP use the same exact native task and v3 replay exporter.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading


def _activate_source():
    here = Path(__file__).resolve()
    root = next((parent for parent in here.parents if (parent / "pyproject.toml").is_file()), None)
    if root is None:
        raise RuntimeError("The fixed contact-family source checkout is unavailable")
    sys.path.insert(0, str(root / "src"))
    import resectionlab
    resectionlab.__path__.insert(0, str(here.parent))
    return root


ROOT = _activate_source()
_PARENT_LEASE_ACTIVE = False


def _require_parent_lease():
    global _PARENT_LEASE_ACTIVE
    encoded = os.environ.get("RESECTIONLAB_PARENT_LEASE_FD")
    if encoded is None or not encoded.isdecimal() or int(encoded) < 3:
        raise RuntimeError("Contact-family worker requires an owned parent lease")
    fd = int(encoded)
    os.fstat(fd)
    def watch():
        try:
            while os.read(fd, 1):
                pass
        finally:
            os._exit(143)
    threading.Thread(target=watch, name="contact-family-parent-lease", daemon=True).start()
    _PARENT_LEASE_ACTIVE = True


def _write_reserved(handle, record):
    handle.seek(0)
    handle.truncate()
    json.dump(record, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def execute(output: Path, *, layout_id: str, goal_id: str, selector: str):
    if not _PARENT_LEASE_ACTIVE:
        raise RuntimeError("Contact-family worker requires its active parent lease before model import")
    # One fresh output is reserved before any task, checkpoint, or native call.
    with output.open("x") as handle:
        _write_reserved(handle, {"status": "started", "scope": "generated_contact_family_v3",
                                 "automaticRetry": False})
        stage = "source_and_role"
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            if torch.get_default_dtype() != torch.float32 or str(torch.get_default_device()) != "cpu":
                raise RuntimeError("Contact-family worker requires CPU float32 defaults")
            from resectionlab.contact_learning_contract import freeze_contact_experiment, PROTOCOL
            from resectionlab.public_contact_family import (FAMILY_VERSION, SOURCE_CANDIDATE_VERSION)
            from resectionlab.contact_family_episode import (
                freeze_final_checkpoints, make_frozen_evaluation_task,
                bind_family_episode, export_family_strategy, SCHEMA)
            from resectionlab.contact_checkpoint import load_contact_checkpoint
            from resectionlab.contact_family_desktop_release import read_published_contact_release
            from resectionlab.goal_mode_episode_adapter import plan_goal_mode_strategy
            from resectionlab.observed_search import observed_beam_search
            from resectionlab.public_surface_contact import seal_complete_strategy
            from resectionlab.surface_contact_episode import episode_envelope
            from resectionlab.core import semantic_digest

            if selector not in ("STOP", "SEARCH", "IL", "RL", "IL_TRAIN_REFIT"):
                raise ValueError("Unknown fixed contact-family method")
            if (FAMILY_VERSION != "generated-public-contact-family-v2" or
                    SOURCE_CANDIDATE_VERSION != "fixed_lattice_access_centerline_v1"):
                raise RuntimeError("Reviewed public contact-family v2 source is not installed")
            experiment = freeze_contact_experiment()
            row = experiment.row(layout_id)
            if row["role"] not in ("TRAIN", "SELECT") or goal_id not in row["goals"]:
                raise ValueError("HELD_OUT_EXECUTION_CLOSED: choose a TRAIN/SELECT public family row")

            refit = selector == "IL_TRAIN_REFIT"
            algorithm = "IL" if refit else selector
            if refit and row["role"] != "TRAIN":
                raise ValueError("TRAIN_REFIT_ONLY: refuse role before artifact/model decode")
            refit_publication = None
            release = None
            checkpoint_metadata = None
            release_evidence = None
            policy = None
            if selector in ("IL", "RL"):
                stage = "both_final_checkpoint_admission"
                release_evidence = read_published_contact_release(ROOT,
                    family_manifest=experiment.manifest, experiment_hash=experiment.fingerprint)
                records = {method: {"path": release_evidence["checkpointPaths"][method],
                    "sha256": release_evidence["manifest"]["checkpoints"][method]["file"]["sha256"],
                    "parameter_hash": release_evidence["manifest"]["checkpoints"][method]["parameterHash"]}
                    for method in ("IL", "RL")}
                release = freeze_final_checkpoints(experiment, records)
                if semantic_digest(release.record()) != semantic_digest(
                        release_evidence["finalFreezeRecord"]):
                    raise RuntimeError("Decoded checkpoint pair differs from the declared final freeze")
                policy, checkpoint_metadata = load_contact_checkpoint(records[selector]["path"],
                    expected_sha256=records[selector]["sha256"], experiment=experiment, kind="final")
                if checkpoint_metadata.file_sha256 != records[selector]["sha256"]:
                    raise RuntimeError("Selected verified checkpoint file differs from the publication")

            if refit:
                stage = "TRAIN_refit_checkpoint_admission"
                from resectionlab.contact_train_refit_release import load_train_refit_for_owned_inference
                experiment, policy, checkpoint_metadata, refit_publication = load_train_refit_for_owned_inference(
                    ROOT, layout_id=layout_id, goal_id=goal_id)
                row = experiment.row(layout_id)

            stage = "public_task_inventory_and_planning"
            task = make_frozen_evaluation_task(experiment, layout_id, goal_id, release=release)
            context, declaration = bind_family_episode(experiment, task, layout_id=layout_id,
                                                       goal_id=goal_id, release=release)
            if declaration["role"] != row["role"] or context.source_hash != row["source_hash"]:
                raise RuntimeError("Contact-family task differs from the fixed source/role")
            if (declaration.get("proposal_mode") != SOURCE_CANDIDATE_VERSION or
                    declaration.get("source_candidate_version") != SOURCE_CANDIDATE_VERSION):
                raise RuntimeError("Contact-family declaration changed candidate rule")
            if algorithm in ("IL", "RL"):
                plan, seal, accounting = plan_goal_mode_strategy(policy, task, context=context)
            else:
                if selector == "STOP":
                    actions = ("STOP",)
                    accounting = {"selector": "immediate_STOP", "actor_forward_calls": 0,
                                  "model_transition_calls": 0, "optimizer_updates": 0}
                else:
                    actions, accounting = observed_beam_search(task, **dict(PROTOCOL["search"]),
                        objective_source="same_public_retained_surface_goal",
                        transition_mode="lazy_planning")
                package = seal_complete_strategy(task, actions)
                plan, seal = package["strategy"], package["strategySeal"]

            stage = "sealed_native_execution_and_replay"
            display, episode = export_family_strategy(experiment, task, layout_id=layout_id,
                goal_id=goal_id, selector=algorithm, plan=plan, seal=seal, accounting=accounting,
                context=context, release=release, checkpoint_metadata=checkpoint_metadata)
            if (episode.get("schema") != SCHEMA or episode.get("layoutId") != layout_id or
                    episode.get("splitRole") != row["role"] or episode.get("selector") != algorithm or
                    episode.get("publicGoal", {}).get("goalId") != goal_id or
                    episode.get("caseHash") != display.semantic_hash or
                    episode.get("patientAdmission") is not False or
                    episode.get("clinicalValidation") is not False):
                raise RuntimeError("Contact-family v3 episode differs from fixed task and native display")
            contract = episode.get("taskContract")
            if (type(contract) is not dict or
                    contract.get("proposalMode") != SOURCE_CANDIDATE_VERSION or
                    contract.get("sourceCandidateVersion") != SOURCE_CANDIDATE_VERSION):
                raise RuntimeError("Contact-family episode changed candidate rule")
            if algorithm in ("IL", "RL"):
                authorship = episode.get("learnedAuthorship")
                if (authorship is None or authorship.get("checkpointFileSha256") !=
                        checkpoint_metadata.file_sha256 or authorship.get("method") != algorithm or
                        type(authorship.get("inferenceOptimizerUpdates")) is not int or
                        authorship["inferenceOptimizerUpdates"] != 0):
                    raise RuntimeError("Learned v3 episode omitted verified endpoint identity")
            elif episode.get("learnedAuthorship") is not None:
                raise RuntimeError("SEARCH/STOP v3 episode falsely claims learned authorship")
            envelope = episode_envelope(episode)
            stage = "result_serialization"
            result = {"status": "complete", "scope": "generated_contact_family_v3",
                "clinicalValidation": False, "newOptimizerUpdates": 0,
                "layoutId": layout_id, "goalId": goal_id, "selector": selector,
                "splitRole": row["role"], "experimentHash": experiment.fingerprint,
                "familyHash": experiment.manifest["family_hash"],
                "caseHash": display.semantic_hash, "episode": episode,
                "episodeCanonicalJson": envelope["episodeCanonicalJson"],
                "releaseEvidence": None if release_evidence is None else {
                    "releaseManifestSha256": release_evidence["manifestSha256"],
                    "pilotResultSha256": release_evidence["pilotResultSha256"],
                    "finalFreezeSha256": release_evidence["finalFreezeSha256"],
                    "checkpointFileSha256": checkpoint_metadata.file_sha256}}
            if refit:
                files = refit_publication["manifest"]["files"]
                result["policyVariant"] = "IL_TRAIN_REFIT"
                result["releaseEvidence"] = {
                    "releaseManifestSha256": refit_publication["manifestSha256"],
                    "fitResultSha256": files["fitResult"]["sha256"],
                    "rolloutResultSha256": files["rolloutResult"]["sha256"],
                    "independentAuditSha256": files["independentAudit"]["sha256"],
                    "checkpointFileSha256": checkpoint_metadata.file_sha256}
            _write_reserved(handle, result)
            return result
        except BaseException as error:
            _write_reserved(handle, {"status": "failed", "stage": stage,
                "errorType": type(error).__name__, "error": str(error),
                "automaticRetry": False, "scope": "generated_contact_family_v3"})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout-id", required=True)
    parser.add_argument("--goal-id", required=True)
    parser.add_argument("--selector", required=True)
    args = parser.parse_args(argv)
    _require_parent_lease()
    execute(args.output, layout_id=args.layout_id, goal_id=args.goal_id, selector=args.selector)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
