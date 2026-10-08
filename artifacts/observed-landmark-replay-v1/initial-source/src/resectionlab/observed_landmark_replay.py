"""Two acquired RESECT phases, using only the already released six-B field.

This is an inspection-point replay, not an anatomical case or surgical plan.
The complete file digest is an audit binding; the active state digest contains
only the observations available at the selected acquisition phase. No raw data,
validation results, images, clocks, interpolated motion or refitting are used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Mapping

from .core import freeze_json, semantic_digest, thaw_json

STUDY_ID = "resect-case4-sparse-update-v1"
FIELD_PATH = Path("artifacts") / STUDY_ID / "fit-freeze/comparison/prediction-field.json"
FIELD_SHA256 = "5e3944a40a0240af99e8c82360fb02241ea4ab3a7148c60950e2829e64e79858"
B_IDS = (1, 14, 8, 17, 19, 7)
SOURCE_IMAGE_SHA256 = "sha256:62f6b6217774653267702acd4a55b16ad5c8cc4980237afaa39e92b102240379"
DESTINATION_IMAGE_SHA256 = "sha256:90fd3e82fbf99f7d45ed25108829f8e61a632ecf6d9a8706b5e4ac6866760a8a"
PARTITION_HASH = "sha256:cc9d7776cbb4bb6ba745519e24f008c53e556bfe60d3c0509917377c9db02982"
PROTOCOL_SHA256 = "64229fe7523a65a90a878b02050273424ff8c9fb9f84d76c28110f097c2d7c5e"
MAX_FIELD_BYTES = 64 * 1024


class ObservedLandmarkError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _selection(phase: Any, landmark_id: Any) -> None:
    if phase not in ("before", "during") or type(landmark_id) is not int or landmark_id not in B_IDS:
        raise ObservedLandmarkError("INVALID_OBSERVED_SELECTION", "Select before/during and one of the six original B row IDs")


@dataclass(frozen=True)
class ObservedLandmarkReplay:
    _field: Mapping
    phase: str = "before"
    landmark_id: int = 1

    @classmethod
    def load(cls, source_root: Path | None) -> "ObservedLandmarkReplay":
        """Read a single fixed, bounded artifact from an engine-owned root."""
        if source_root is None:
            raise ObservedLandmarkError("OBSERVED_SOURCE_UNAVAILABLE", "No trusted observation source root is configured")
        root = Path(source_root)
        if not root.is_absolute():
            raise ObservedLandmarkError("OBSERVED_SOURCE_UNAVAILABLE", "Observation source root must be absolute")
        path = root / FIELD_PATH
        try:
            # Reject redirects at every component, including ancestors of root.
            if any(item.is_symlink() for item in (path, *path.parents)):
                raise ObservedLandmarkError("OBSERVED_SOURCE_INVALID", "Observation source cannot contain symbolic links")
            metadata = path.stat()
            if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= MAX_FIELD_BYTES:
                raise ObservedLandmarkError("OBSERVED_SOURCE_INVALID", "Observation field must be a bounded regular file")
            with path.open("rb") as source:
                payload = source.read(MAX_FIELD_BYTES + 1)
        except OSError as error:
            raise ObservedLandmarkError("OBSERVED_SOURCE_UNAVAILABLE", "The pinned six-B observation artifact is unavailable") from error
        if hashlib.sha256(payload).hexdigest() != FIELD_SHA256:
            raise ObservedLandmarkError("OBSERVED_SOURCE_CHANGED", "Observation artifact differs from the released six-B field")
        field = json.loads(payload)
        baseline = field["baseline"]
        observations = baseline["observations"]
        if (field["external_field"] is not None or baseline["schema"] != "conditional-baseline-field-v1"
                or baseline["protocol_sha256"] != PROTOCOL_SHA256
                or observations["boundary_ids"] != list(B_IDS)
                or observations["partition_hash"] != PARTITION_HASH
                or observations["source_image_sha256"] != SOURCE_IMAGE_SHA256
                or observations["patient_group"] != "RESECT:Case4"
                or observations["frame"] != "RAS+" or observations["units"] != "mm"
                or baseline["rigid"]["status"] != "available"):
            raise ObservedLandmarkError("OBSERVED_SOURCE_INVALID", "Observation field has an incompatible source contract")
        return cls(freeze_json(field))

    @property
    def binding(self) -> dict:
        return {"studyId": STUDY_ID, "patientGroup": "RESECT:Case4", "dataRole": "DEVELOPMENT",
                "fieldSha256": FIELD_SHA256, "partitionHash": PARTITION_HASH,
                "protocolSha256": PROTOCOL_SHA256,
                "sourceImageSha256": SOURCE_IMAGE_SHA256,
                "destinationImageSha256": DESTINATION_IMAGE_SHA256,
                "sourcePair": "original RESECT v1 Case4_US_before-during.tag",
                "sourceDoi": "10.11582/2017.00004", "sourceLicense": "CC-BY-4.0"}

    def advance(self, phase: str, landmark_id: int) -> "ObservedLandmarkReplay":
        _selection(phase, landmark_id)
        if self.phase == "during" and phase == "before":
            raise ObservedLandmarkError("OBSERVED_REPLAY_BACKWARDS", "Reopen a bound snapshot to revisit the before phase")
        return replace(self, phase=phase, landmark_id=landmark_id)

    def frame(self) -> dict:
        _selection(self.phase, self.landmark_id)
        observations = self._field["baseline"]["observations"]
        released = []
        for index, row_id in enumerate(B_IDS):
            source = list(observations["source_ras_mm"][index])
            point = {"landmarkId": row_id, "sourceRasMm": source}
            if self.phase == "during":
                observed = list(observations["observed_ras_mm"][index])
                point.update(observedRasMm=observed,
                             measuredDisplacementMm=[after - before for before, after in zip(source, observed)])
            released.append(point)
        selected = released[B_IDS.index(self.landmark_id)]
        state = {
            "schemaVersion": 1, "studyId": STUDY_ID, "patientGroup": "RESECT:Case4", "dataRole": "DEVELOPMENT",
            "mode": "observed_correspondence_replay", "phase": self.phase,
            "phaseOrdinal": 0 if self.phase == "before" else 1,
            "phaseOrderEvidence": "original paired US acquisition labels before and during resection",
            "acquisitionTimestamp": None, "elapsedSeconds": None, "measurementAvailabilityTimestamp": None,
            "timeSemantics": "recorded acquisition order only; acquisition time and measurement availability unknown",
            "frame": "RAS+", "units": "mm", "partitionHash": PARTITION_HASH,
            "observationRole": observations["role"], "boundaryIds": list(B_IDS),
            "sourceImageSha256": SOURCE_IMAGE_SHA256,
            "sourceWorldToRasMm": thaw_json(observations["source_world_to_ras_mm"]),
            "observations": released,
            "selectedInspectionPoint": {"landmarkId": self.landmark_id,
                "kind": "source_linked_correspondence_not_anatomical_target",
                "positionRasMm": selected.get("observedRasMm", selected["sourceRasMm"])},
            "anatomicalTarget": None, "toolPose": None, "brainSupport": None, "cavitySupport": None,
            "physicalClearanceMm": None, "totalRegistrationUncertaintyMm": None,
            "selectedAction": "ABSTAIN_UNSUPPORTED_CLEARANCE",
            "decisionReason": "Sparse correspondence does not establish retained tissue, cavity, tool clearance or a surgical action.",
        }
        prediction = None
        if self.phase == "during":
            state["destinationImageSha256"] = DESTINATION_IMAGE_SHA256
            state["destinationWorldToRasMm"] = thaw_json(observations["destination_world_to_ras_mm"])
            rigid = self._field["baseline"]["rigid"]
            source = selected["sourceRasMm"]
            # Apply the saved proper-rigid field. There is no fit or method selection here.
            position = [sum(rotation[j] * source[j] for j in range(3)) + offset
                        for rotation, offset in zip(rigid["rotation"], rigid["translation_mm"])]
            prediction = {"method": "proper_rigid", "fieldSha256": FIELD_SHA256,
                          "positionRasMm": position,
                          "scope": "frozen estimate fitted from these same six B observations; not independent validation"}
        result = {"schemaVersion": 1, "binding": self.binding, "state": state,
                  "stateHash": semantic_digest(state), "prediction": prediction}
        result["resultHash"] = semantic_digest(result)
        result["snapshot"] = {"schemaVersion": 1, "kind": "observed_landmark_snapshot",
                              "binding": self.binding, "phase": self.phase, "landmarkId": self.landmark_id,
                              "stateHash": result["stateHash"], "resultHash": result["resultHash"]}
        return result

    def snapshot(self) -> dict:
        return self.frame()["snapshot"]

    def reopen(self, snapshot: dict) -> "ObservedLandmarkReplay":
        """Reproduce a complete snapshot against this freshly checked source."""
        if not isinstance(snapshot, dict) or set(snapshot) != {
                "schemaVersion", "kind", "binding", "phase", "landmarkId", "stateHash", "resultHash"}:
            raise ObservedLandmarkError("OBSERVED_SNAPSHOT_MISMATCH", "Invalid observation snapshot fields")
        _selection(snapshot["phase"], snapshot["landmarkId"])
        replay = replace(self, phase=snapshot["phase"], landmark_id=snapshot["landmarkId"])
        # Canonical comparison also distinguishes JSON true from the integer 1.
        if semantic_digest(snapshot) != semantic_digest(replay.snapshot()):
            raise ObservedLandmarkError("OBSERVED_SNAPSHOT_MISMATCH", "Snapshot source, phase, point or result binding changed")
        return replay

    def assert_result_current(self, result: dict) -> None:
        if semantic_digest(result) != semantic_digest(self.frame()):
            raise ObservedLandmarkError("OBSERVED_RESULT_STALE", "Observation result does not match the current source, phase, point and frame")
