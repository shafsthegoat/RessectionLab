#!/usr/bin/env python3
"""Add explicit evidence labels to an already verified learned native export."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from patient_rescue_env import sha
from resectionlab.workspace_bundle import load_workspace, save_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("bc", "ppo"), required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to(ROOT / "testcases") or destination.exists():
        parser.error("Use a new destination under ignored testcases")
    report = json.loads((args.run / "result.json").read_text())
    case, artifacts, workspace = load_workspace(args.case)
    original = copy.deepcopy(artifacts["surgical_toolbox"]["branches"])
    prefix_actions = len(original["catalog-rescue-task-start"]["actions"])
    if args.kind == "bc":
        if report["status"] != "completed_bc_only" or not report["exactReplay"]["passed"]:
            raise ValueError("Verified successful BC source required")
        key = "patient_rescue_bc"
        evidence = artifacts[key]
        if evidence["checkpointSha256"] != report["checkpoint"]["sha256"]:
            raise ValueError("BC export checkpoint differs")
        evidence.update(method="frozen_behavior_cloning", branchId=report["workspace"]["branch"],
            prefixActions=prefix_actions,
            finalStateId=original[report["workspace"]["branch"]]["stateId"],
            actualSupervisedOptimizerSteps=report["fit"]["optimizerSteps"],
            actualRlOptionTransitions=0, auditPassed=True, samePreparedInjuryOnly=True,
            noGeneralizationClaim=True, promoted=False,
            actualActorChoices=report["live"]["options"])
    else:
        if args.audit is None:
            parser.error("PPO export needs its independent audit receipt")
        audit = json.loads(args.audit.read_text())
        audit_case_sha = audit["workspace"]["sha256"]
        current_case_sha = sha(args.case)
        if current_case_sha != audit_case_sha:
            previous = json.loads(args.case.with_suffix(".metadata-receipt.json").read_text())
            if (previous["sourceSha256"] != audit_case_sha
                    or previous["outputSha256"] != current_case_sha
                    or not previous["sourceAndAllNativeBranchesExact"]):
                raise ValueError("Metadata-only case chain differs from the independent audit")
        if (report["status"] != "completed_trained_success" or audit["status"] != "passed"
                or audit_case_sha is None):
            raise ValueError("Verified successful PPO source required")
        key = "patient_rescue_ppo_bc"
        evidence = artifacts[key]
        if evidence["checkpointSha256"] != report["checkpoint"]["sha256"]:
            raise ValueError("PPO export checkpoint differs")
        evidence.update(method="ppo_initialized_from_behavior_cloning",
            seed=report["seed"], branchId=audit["workspace"]["branch"],
            prefixActions=prefix_actions,
            finalStateId=original[audit["workspace"]["branch"]]["stateId"],
            initialBCCheckpointSha256=report["bcCheckpointSha256"],
            observationSchema="resectionlab.prepared-rescue-observation.v2",
            actualPpoOptionTransitions=report["training"]["optionDecisions"],
            actualPpoUpdateEpochs=report["training"].get("ppoUpdateEpochs"),
            actualOptimizerSteps=report["training"].get("optimizerSteps"),
            auditPassed=True, samePreparedInjuryOnly=True,
            noGeneralizationClaim=True, noIncrementalRLBenefitClaim=True,
            promoted=False, actualActorChoices=report["trained"]["options"])
    save_workspace(case, destination, sources=workspace["sources"],
        imaging=workspace["imagingState"], episode=workspace["episode"],
        selection=workspace["selection"], originals=workspace["originals"], artifacts=artifacts)
    reopened, saved, _ = load_workspace(destination)
    if (reopened.semantic_hash != case.semantic_hash
            or saved["surgical_toolbox"]["branches"] != original
            or saved[key] != evidence):
        raise ValueError("Evidence metadata save/reopen differs")
    receipt = dict(schema="resectionlab.patient-rescue-learning-metadata-upgrade.v1",
        kind=args.kind, sourceCase=str(args.case.resolve()), sourceSha256=sha(args.case),
        outputCase=str(destination), outputSha256=sha(destination),
        artifactKey=key, artifactSchema=evidence["schema"],
        sourceAndAllNativeBranchesExact=True, evidence=evidence)
    path = destination.with_suffix(".metadata-receipt.json")
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"outputCase": str(destination), "sha256": receipt["outputSha256"],
                      "artifactKey": key}))


if __name__ == "__main__":
    main()
