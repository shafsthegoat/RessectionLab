"""Saved-only audit of three root-owned generated desktop episodes.

No backend import, checkpoint read, model call, native replay, or UI action.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = Path("/Users/sayemkamal/Library/Application Support/ressectionlab-desktop/"
            "research-runs/generated-contact-family-attempts")
HERE = Path(__file__).resolve().parent
ATTEMPTS = {
    "SEARCH": "06d5dfa679d347e0b719ec34f9e0ef08",
    "IL": "88183e132b694b11baccef591fa7d7e8",
    "RL": "3474935409744af38e889ac3eb47d131",
}
RELEASE_SHA = "4bc4a93f453fb4064c3291b55c60481de554adfe706ab7db24a3a023f5e20dae"
RESULT_SHA = "d57a6590c2ecc058429576d9dd0e0714278dbac5fec5029ee22ef7844ce213c1"
FREEZE_SHA = "49fadaae1a891b0c29c9d4076e13505811ae24d0c62beb4cc62cfa763a794625"
EXPERIMENT = "sha256:a01e6ba30fbf1e6efabf70f9637d42c0813931624cb50d2d393858029d4fd894"
FAMILY = "sha256:ad21aff05f8a6df7b75cd86b12a5730d9554864fe9792a649489812a0fb053ed"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(path: Path, limit: int) -> tuple[dict, dict]:
    assert path.is_file() and not path.is_symlink(), str(path)
    data = path.read_bytes()
    assert len(data) <= limit, str(path)
    def unique(pairs):
        value = {}
        for key, item in pairs:
            assert key not in value, str(path)
            value[key] = item
        return value
    value = json.loads(data, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
    assert type(value) is dict, str(path)
    return value, {"sha256": sha(data), "bytes": len(data)}


def canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def audit_one(selector: str, suffix: str, manifest: dict, freeze: dict) -> dict:
    path = RUNS / ("contact-family-" + suffix)
    assert path.is_dir() and not path.is_symlink()
    files = {}
    records = {}
    for name, limit in (("attempt.json", 4096), ("completion.json", 4096),
                        ("progress.json", 4096), ("supervision.json", 4096),
                        ("result.json", 2 * 1024 * 1024)):
        records[name], files[name] = read(path / name, limit)
    log = path / "worker.log"
    assert log.is_file() and not log.is_symlink() and log.stat().st_size <= 1024 * 1024
    files["worker.log"] = {"sha256": sha(log.read_bytes()), "bytes": log.stat().st_size}
    attempt, completion, progress, supervision, result = (records[name] for name in
        ("attempt.json", "completion.json", "progress.json", "supervision.json", "result.json"))
    assert set(path.iterdir()) == {path / name for name in files}, suffix
    assert attempt["status"] == "started" and attempt["selector"] == selector
    assert attempt["layoutId"] == "pcf-10" and attempt["goalId"] == "surface"
    assert attempt["automaticRetry"] is False and attempt["maxWallSeconds"] == 20.0
    assert attempt["maxSampledWorkerRssBytes"] == 1024 ** 3
    assert completion["status"] == "complete"
    assert completion["resultSha256"] == files["result.json"]["sha256"]
    assert completion["supervisionSha256"] == files["supervision.json"]["sha256"]
    assert supervision["status"] == "complete" and supervision["workerReturnCode"] == 0
    assert supervision["stopReason"] is None and supervision["cleanupErrors"] == []
    assert supervision["workerTerminationConfirmed"] is True
    assert supervision["unresolvedWorkerPid"] is None and supervision["automaticRetry"] is False
    assert type(supervision["samples"]) is int and supervision["samples"] > 0
    assert 0 < supervision["sampledPeakRssBytes"] < 1024 ** 3
    assert 0 < supervision["elapsedSeconds"] < 25.0
    assert progress["samples"] == supervision["samples"]
    assert progress["sampledPeakRssBytes"] == supervision["sampledPeakRssBytes"]
    assert progress["stopReason"] is None
    assert result["status"] == "complete" and result["selector"] == selector
    assert result["layoutId"] == "pcf-10" and result["goalId"] == "surface"
    assert result["splitRole"] == "TRAIN" and result["newOptimizerUpdates"] == 0
    assert result["clinicalValidation"] is False
    assert result["experimentHash"] == EXPERIMENT and result["familyHash"] == FAMILY
    episode = result["episode"]
    body = {key: value for key, value in episode.items() if key != "episodeId"}
    encoded = canonical(body)
    assert result["episodeCanonicalJson"] == encoded
    assert episode["episodeId"] == "sha256:" + sha(encoded.encode())
    assert result["caseHash"] == episode["caseHash"]
    assert episode["schema"] == "resectionlab.shared-native-contact-learning-episode.v3"
    assert episode["fixture"] == "generated-public-contact-family-v2"
    assert episode["selector"] == selector and episode["splitRole"] == "TRAIN"
    assert episode["patientAdmission"] is False and episode["clinicalValidation"] is False
    assert episode["familyHash"] == FAMILY
    planning = episode["planning"]
    assert planning["experimentHash"] == EXPERIMENT
    strategy = planning["strategy"]
    assert planning["strategySeal"] == "sha256:" + sha(canonical(strategy).encode())
    actions = [item["action_id"] for item in episode["history"]]
    assert actions == strategy["actions"]
    assert strategy["history"] == episode["history"]
    assert planning["optimizer_updates"] == 0
    metrics = episode["metrics"]
    assert metrics["clinical_validation"] is False
    assert metrics["planning_estimator_only"] is False
    assert metrics["source_hash"] == episode["sourceHash"]
    if selector == "SEARCH":
        assert [item["interaction_mode"] for item in episode["history"]] == ["aspirate", "probe"]
        assert metrics["goal_contacted_and_retained"] is True
        assert metrics["goal_retained"] is True
        assert abs(metrics["total_reward"] - 0.704) < 1e-12
        assert metrics["removed_volume_mm3"] == 1.0
        assert planning["learnedPolicyExecuted"] is False
        assert planning["actor_forward_calls"] == 0 and planning["model_transition_calls"] == 20
        assert planning["time_cap_reached"] is False and planning["call_cap_reached"] is False
        assert episode["learnedAuthorship"] is None and result["releaseEvidence"] is None
    else:
        assert actions == ["STOP"] and episode["history"][0]["interaction_mode"] == "stop"
        assert metrics["goal_contacted_and_retained"] is False
        assert metrics["goal_retained"] is True
        assert metrics["total_reward"] == 0.0 and metrics["removed_volume_mm3"] == 0.0
        assert planning["learnedPolicyExecuted"] is True
        assert planning["actor_forward_calls"] == 1 and planning["model_transition_calls"] == 1
        assert planning["training_or_checkpoint_lineage_verified"] is True
        authorship, release = episode["learnedAuthorship"], result["releaseEvidence"]
        checkpoint = manifest["checkpoints"][selector]
        frozen = freeze["checkpoints"][selector]
        assert authorship["method"] == selector and authorship["checkpointKind"] == "final"
        assert authorship["completedUpdates"] == 32 and authorship["inferenceOptimizerUpdates"] == 0
        assert authorship["experimentHash"] == EXPERIMENT and authorship["familyHash"] == FAMILY
        assert authorship["checkpointFileSha256"] == checkpoint["file"]["sha256"] == frozen["file_sha256"]
        assert authorship["parameterHash"] == checkpoint["parameterHash"] == frozen["parameter_hash"]
        assert authorship["trainingLineageHash"] == frozen["lineage_hash"]
        assert release == {"releaseManifestSha256": RELEASE_SHA,
                           "pilotResultSha256": RESULT_SHA,
                           "finalFreezeSha256": FREEZE_SHA,
                           "checkpointFileSha256": checkpoint["file"]["sha256"]}
    return {"selector": selector, "attemptId": suffix, "fileRecords": files,
            "episodeId": episode["episodeId"], "caseHash": episode["caseHash"],
            "sourceHash": episode["sourceHash"], "initialStateId": episode["initialStateId"],
            "initialObservationHash": episode["initialObservationBinding"]["observationHash"],
            "strategySeal": planning["strategySeal"], "actions": actions,
            "actionModes": [item["interaction_mode"] for item in episode["history"]],
            "goalContactedAndRetained": metrics["goal_contacted_and_retained"],
            "totalReward": metrics["total_reward"],
            "removedVolumeMm3": metrics["removed_volume_mm3"],
            "actorForwardCalls": planning["actor_forward_calls"],
            "modelTransitionCalls": planning["model_transition_calls"],
            "newOptimizerUpdates": result["newOptimizerUpdates"],
            "elapsedSeconds": supervision["elapsedSeconds"],
            "sampledPeakRssBytes": supervision["sampledPeakRssBytes"],
            "samples": supervision["samples"],
            "workerTerminationConfirmed": supervision["workerTerminationConfirmed"],
            "cleanupErrors": supervision["cleanupErrors"],
            "checkpointFileSha256": (episode["learnedAuthorship"] or {}).get("checkpointFileSha256"),
            "parameterHash": (episode["learnedAuthorship"] or {}).get("parameterHash"),
            "trainingLineageHash": (episode["learnedAuthorship"] or {}).get("trainingLineageHash")}


def main() -> None:
    manifest_path = ROOT / "artifacts/public-contact-learning-v1/desktop-release.json"
    manifest, manifest_record = read(manifest_path, 32 * 1024)
    assert manifest_record["sha256"] == RELEASE_SHA
    freeze_path = ROOT / manifest["finalFreeze"]["relativePath"]
    freeze, freeze_record = read(freeze_path, 128 * 1024)
    assert freeze_record["sha256"] == FREEZE_SHA
    assert manifest["pilotResult"]["sha256"] == RESULT_SHA
    assert manifest["experimentHash"] == EXPERIMENT and manifest["familyHash"] == FAMILY
    audits = [audit_one(selector, suffix, manifest, freeze) for selector, suffix in ATTEMPTS.items()]
    common = ("caseHash", "sourceHash", "initialStateId", "initialObservationHash")
    assert all(all(row[key] == audits[0][key] for key in common) for row in audits)
    document = {"schema": "generated-contact-family-desktop-live-saved-audit-v1",
                "scope": "three completed desktop attempts, saved metadata and result only",
                "releaseManifestSha256": RELEASE_SHA,
                "pilotResultSha256": RESULT_SHA,
                "finalFreezeSha256": FREEZE_SHA,
                "samePublicInitialStateAndObservation": True,
                "checkpointBodiesRead": False, "modelForwardsRunByAudit": 0,
                "nativeTransitionsRunByAudit": 0, "uiActionsRunByAudit": 0,
                "attempts": audits}
    output = HERE / "audit.json"
    payload = json.dumps(document, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    if output.exists():
        assert output.read_bytes() == payload, "Saved audit bytes changed"
    else:
        output.write_bytes(payload)
    print("saved_audit_sha256", sha(payload))
    print("saved_audit_attempts", len(audits))


if __name__ == "__main__":
    main()
