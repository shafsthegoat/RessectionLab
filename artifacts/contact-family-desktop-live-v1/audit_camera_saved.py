"""Saved-only receipt for the separate camera-fit SEARCH UI regression."""
from __future__ import annotations

import json
from pathlib import Path

from audit_saved import (ATTEMPTS, FREEZE_SHA, HERE, RELEASE_SHA, ROOT, audit_one,
                         canonical, read, sha)


CAMERA_ID = "71711c7a0fcf4d27bfa63cd79785a509"


def main() -> None:
    manifest, record = read(ROOT / "artifacts/public-contact-learning-v1/desktop-release.json", 32 * 1024)
    assert record["sha256"] == RELEASE_SHA
    freeze, record = read(ROOT / manifest["finalFreeze"]["relativePath"], 128 * 1024)
    assert record["sha256"] == FREEZE_SHA
    original = json.loads((HERE / "audit.json").read_text())["attempts"][0]
    assert original["attemptId"] == ATTEMPTS["SEARCH"]
    camera = audit_one("SEARCH", CAMERA_ID, manifest, freeze)
    for key in ("caseHash", "sourceHash", "initialStateId", "initialObservationHash",
                "strategySeal", "actions", "actionModes", "goalContactedAndRetained",
                "totalReward", "removedVolumeMm3"):
        assert camera[key] == original[key], key
    document = {"schema": "generated-contact-family-camera-fit-saved-audit-v1",
                "purpose": "separate UI camera regression, not another planned method comparison or timing replicate",
                "originalSearchAttemptId": original["attemptId"],
                "samePublicInitialStateStrategyActionsAndOutcome": True,
                "screenshotEvidencePath": "build/episode-camera-fit-v1/live-probe-final.png",
                "screenshotPixelsInspectedByThisAudit": False,
                "checkpointBodiesRead": False, "modelForwardsRunByAudit": 0,
                "nativeTransitionsRunByAudit": 0, "uiActionsRunByAudit": 0,
                "cameraAttempt": camera}
    payload = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    output = HERE / "camera-audit.json"
    if output.exists():
        assert output.read_bytes() == payload
    else:
        output.write_bytes(payload)
    print("camera_saved_audit_sha256", sha(payload))


if __name__ == "__main__":
    main()
