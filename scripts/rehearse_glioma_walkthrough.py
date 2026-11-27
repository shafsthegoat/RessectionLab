"""Run or audit the source-bound generated walkthrough using the desktop's public actions.

The input is a regular editable .ressectionlab workspace. Each stage writes its
actual accepted action boundaries for the desktop recorder. A refused action
leaves a reopenable checkpoint and the original refusal for inspection.

Create a fresh run under testcases/outputs using
  run SOURCE --landmarks LANDMARKS.json --output DIRECTORY [--through scalp]
and audit the physical snapshot in a fresh Python process with
  audit DIRECTORY/complete.ressectionlab
Resume a named earlier stage on its verified episode with
  run CHECKPOINT --resume --landmarks LANDMARKS.json --output NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
from contextlib import closing
import gzip
from hashlib import file_digest
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from resectionlab.core import semantic_digest
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.imaging import read_case_artifacts
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment


def dump(path, data):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def file_hash(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


class DesktopEpisode:
    """A single case, native practice branch and verified current state."""

    def __init__(self, bridge, source, output, landmarks, resume):
        self.bridge = bridge
        self.output = output
        self.rows = []
        self.attempts = []
        self.phase = "inspection"
        payload = self.dispatch("loadCase", path=str(source.resolve(strict=True)))
        self.case_hash = payload["caseHash"]
        self.scene = load_environment(source)
        if landmarks["caseHash"] != self.case_hash:
            raise ValueError("The landmarks are bound to a different case")
        center = list(landmarks["entryCenterRasMm"])
        target = [center[0], center[1], (center[2] + landmarks["targetRasMm"][2]) / 2.]
        viewer = {"operativeView": "surgeon", "smoothSurfaces": True, "microscope": {
            "target": target, "outward": landmarks["outwardRasUnit"], "angleDeg": 26., "azimuthDeg": 0.,
            "workingDistanceMm": 160., "fieldMm": 38., "focusOffsetMm": 0., "light": 1.6,
            "inspectionCutaway": False, "inspectionTransparency": False, "tissueStyle": "operative"}}
        visible = {r["name"]: True for r in payload["compartments"]}
        self.workspace = {"cursor": center, "visibleLayers": visible, "imaging": {"selectedSeriesId": None,
            "states": {"primary": {"cursor": center, "visibleLayers": visible, "viewer": viewer}}}}
        before = self.command("inspect")
        self.parent = {"caseHash": self.case_hash, "sourceSha256": file_hash(source),
                       "position": before["position"], "branchId": before["branchId"],
                       "stateId": before["view"]["stateId"]}
        if landmarks["sceneId"] != before["view"]["scene"]["sceneId"]:
            raise ValueError("The source landmarks do not match this physical scene")
        self.resume = None
        if resume:
            witness = json.loads(source.with_suffix(".json").read_text())
            self.episode = before["practiceEpisode"]["episodeId"]
            assert witness["sha256"] == file_hash(source)
            assert witness["position"] == before["position"] and witness["finalId"] == before["view"]["stateId"]
            assert witness["branchId"] == before["branchId"] == self.episode
            self.parent = witness["parent"]
            self.rows = witness["actions"]
            self.attempts = list(self.rows)
            self.resume = None if not self.rows and source.stem == "editable-start" else source.stem
            self.phase = witness["stage"]
            self.last = before
            return
        admission = before["practiceAdmission"]
        if not admission["available"]:
            raise ValueError(f"Practice unavailable for this checkpoint: {admission['reason']}")
        self.last = self.command("practice", expectedStateId=self.parent["stateId"],
                                 position=before["position"], branchId=before["branchId"],
                                 checkpointSeal=admission["checkpointSeal"])
        self.episode = self.last["practiceEpisode"]["episodeId"]
        assert self.episode == self.last["branchId"]
        self.save("editable-start")

    def dispatch(self, operation, **arguments):
        return self.bridge.execute(operation, arguments, _Request("generated-walkthrough"), lambda *_: None)

    def command(self, name, **arguments):
        return self.dispatch("surgicalRehearsal", command=name, caseHash=self.case_hash, **arguments)

    @property
    def view(self):
        return self.last["view"]

    def step(self, label, action, *, allowed_partial=()):
        previous = self.last
        start = perf_counter()
        reply = self.command("step", expectedStateId=self.view["stateId"],
                             expectedEpisodeId=self.episode, action=action)
        latency = (perf_counter() - start) * 1000
        view, outcome = reply["view"], reply["outcome"]
        row = {"index": reply["position"], "phase": self.phase, "label": label, "action": action,
               "status": outcome["status"], "reason": outcome.get("reason"),
               "stateBefore": previous["view"]["stateId"], "stateAfter": view["stateId"],
               "instrument": view["slots"][action["slot"]]["instrument"],
               "acceptedPose": {k: view["slots"][action["slot"]][k] for k in ("position", "quaternion")},
               "operativeBodies": {k: {field: b[field] for field in ("status", "angleRad", "cutCount", "heldBy")}
                    for k, b in view["operativeExposureV1"]["bodies"].items()},
               "completedFraction": outcome["completedFraction"], "wallMs": latency,
               "elapsedSeconds": view["metrics"]["elapsedSeconds"], "events": outcome["events"],
               "metrics": {k: view["metrics"].get(k) for k in (
                   "targetRemovedMm3", "nonTargetRemovedMm3", "debrisMm3", "debrisRemovedMm3",
                   "fluidAddedMm3", "fluidPresentMm3", "fluidEscapedMm3", "fluidRemovedMm3")}}
        self.attempts.append(row)
        self.last = reply
        if outcome["status"] == "rejected":
            assert reply["position"] == previous["position"] and row["stateAfter"] == row["stateBefore"]
            raise ValueError(f"{label}: {outcome['reason']}")
        if previous["position"] == reply["position"]:
            assert action["operation"] == "exchange" and row["stateBefore"] == row["stateAfter"]
            return outcome
        assert reply["position"] == previous["position"] + 1
        self.rows.append(row)
        print(f'{row["index"]:4d}  {self.phase}  {label}  {row["status"]}', flush=True)
        if outcome["status"] == "partial" and (outcome.get("reasonCode") or outcome["reason"]) not in allowed_partial:
            raise ValueError(f"{label}: unexpected partial: {outcome['reason']}")
        return outcome

    def save(self, name):
        path = self.output / (name + ".ressectionlab")
        self.dispatch("saveCase", caseHash=self.case_hash, path=str(path.absolute()), workspace=self.workspace)
        result = {"schema": "resectionlab.glioma-walkthrough-history.v1", "caseHash": self.case_hash,
                  "branchId": self.episode, "parent": self.parent, "stage": self.phase, "file": path.name,
                  "sha256": file_hash(path), "position": self.last["position"],
                  "finalId": self.view["stateId"], "actions": self.rows, "attempts": len(self.attempts)}
        bounds = {}
        for item in self.rows:
            phase = bounds.setdefault(item["phase"], {"first": item["index"], "last": item["index"]})
            phase["last"] = item["index"]
        result["phaseBoundaries"] = bounds
        if self.attempts and self.attempts[-1]["status"] == "rejected":
            result["lastRefusal"] = self.attempts[-1]
        dump(self.output / (name + ".json"), result)
        print(f"{name}   {result['position']}   {result['finalId']}   {path}", flush=True)


def perform_recorded(live, path):
    """Submit the archived operator inputs to a fresh editable desktop branch."""
    with gzip.open(path, "rt") as stream:
        guide = json.load(stream)
    if (guide["schema"] != "resectionlab.convexity-operating-inputs.v1"
            or guide["caseHash"] != live.case_hash or guide["intactState"] != live.view["stateId"]
            or live.last["position"] != 0):
        raise ValueError("Recorded operator inputs require their original intact source")
    for index, step in enumerate(guide["steps"], 1):
        live.phase = step["phase"]
        response = live.step(step["label"], step["action"], allowed_partial=(step["reason"],))
        if (response["status"] != step["status"] or live.view["stateId"] != step["stateAfter"]
                or live.last["position"] != index):
            raise ValueError(f"The operator input at boundary {index} reached a different physical outcome")
    if live.view["stateId"] != guide["finalState"] or not live.view["operativeExposureV1"]["complete"]:
        raise ValueError("The recorded inputs did not reproduce the complete episode")


def audit(package, guided_start=None, *, require_complete=False):
    """Independently replay the exact saved source and check each submitted request."""
    receipt = json.loads(package.with_suffix(".json").read_text())
    assert file_hash(package) == receipt["sha256"]
    toolbox = read_case_artifacts(package)["surgical_toolbox"]
    branch = toolbox["branchId"]
    assert branch == receipt["branchId"]
    snapshot = toolbox["branches"][branch]
    state_file = package.with_suffix(".states.jsonl.gz")
    binding = {"scene": snapshot["sceneId"], **{
        k: snapshot[k]["contentHash"] for k in ("protectionContract", "cranialStagesContract") if k in snapshot}}
    with TemporaryDirectory(dir=package.parent) as scratch:
        pending = Path(scratch) / state_file.name
        with pending.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=4, mtime=0) as states:
            def write_boundary(engine):
                position = len(engine.actions)
                row = {"position": position, "stateId": engine.state_id, "state": engine.state}
                if position:
                    row["action"] = engine.actions[-1]
                states.write((json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode())

            native = SurgicalEngine.restore(load_environment(package, branch_id=branch), snapshot,
                                            use_cache=False, force_verify=True, on_boundary=write_boundary)
        with gzip.open(pending, "rt") as records:
            expected = [receipt["parent"]["stateId"], *(r["stateAfter"] for r in receipt["actions"])]
            for i, (line, state_id) in enumerate(zip(records, expected, strict=True)):
                row = json.loads(line)
                if (row["position"] != i or row["stateId"] != state_id or
                        semantic_digest({**binding, "state": row["state"]}) != state_id or
                        (i and row["action"] != snapshot["actions"][i - 1])):
                    raise ValueError(f"Serialized native history differs at boundary {i}")
        pending.replace(state_file)
    assert native.snapshot() == snapshot
    rows = native.history[receipt["parent"]["position"]:]
    for expected, actual in zip(receipt["actions"], rows, strict=True):
        for key in ("stateBefore", "stateAfter", "status", "action", "events", "completedFraction"):
            assert expected[key] == actual[key], (expected["index"], key)
    assert receipt["finalId"] == native.state_id and len(native.history) == receipt["position"]
    from resectionlab.surgical_operative_exposure import summary
    endpoint = summary(native)
    if require_complete and not endpoint["complete"]:
        raise ValueError("The replayed branch did not reach the closed, counted and withdrawn endpoint")
    dump(package.with_suffix(".replay.json"), {
        "sourceFileSha256": receipt["sha256"], "branchId": branch, "actionCount": len(native.history),
        "expectedEndpoint": native.state_id, "normalizedHistorySha256": semantic_digest(native.history),
        "completeSnapshotIdentical": True, "allActionBoundariesIdentical": True,
        "fullStateHistory": {"file": state_file.name, "sha256": file_hash(state_file),
                             "boundaries": len(native.actions) + 1, "identityBinding": binding},
        "clinicalScope": "Synthetic assigned mechanistic scenario without patient-specific function or perfusion.",
        "physicalEndpoint": {"complete": endpoint["complete"], "phase": receipt["stage"],
                             "bodies": {k: {n: b[n] for n in ("status", "cutCount", "edgeCount",
                                                             "closedAnchors", "heldBy")}
                                        for k, b in endpoint["bodies"].items()},
                             "objects": endpoint["objects"], "metrics": native.metrics(),
                             "woundPorts": native.state.get("vascularGraphV1", {}).get("ports", {}),
                             "observations": native.state["observations"]},
        "events": native.history,
    })
    if guided_start is not None:
        from resectionlab.surgical_walkthrough_guide import attach
        template, destination = guided_start
        attach(template, package, package.with_suffix(".json"), destination, verified_engine=native)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("source", type=Path)
    run.add_argument("--resume", action="store_true", help="Resume a named verified checkpoint on the same practice branch")
    run.add_argument("--landmarks", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--recorded-inputs", type=Path, help="Submit a sealed complete operator input sequence on the intact source")
    run.add_argument("--through", choices=("approach", "scalp", "bone", "dura", "treatment", "closure"), default="closure")
    verify = commands.add_parser("audit")
    verify.add_argument("package", type=Path)
    verify.add_argument("--require-complete", action="store_true", help="Require closed layers, complete material counts and withdrawn tools")
    verify.add_argument("--guided-start", nargs=2, type=Path, metavar=("TEMPLATE", "DESTINATION"))
    args = parser.parse_args()
    if args.command == "audit":
        audit(args.package, args.guided_start, require_complete=args.require_complete)
        return
    if args.recorded_inputs and (args.resume or args.through != "closure"):
        parser.error("Recorded operator input replay starts from intact skin and ends at closure")
    output = args.output.absolute()
    if ".." in output.parts or not output.is_relative_to(ROOT / "testcases" / "outputs"):
        parser.error("Choose a new directory under testcases/outputs")
    output.mkdir(parents=True, exist_ok=False)
    from glioma_walkthrough_actions import perform
    landmarks = json.loads(args.landmarks.read_text())
    with TemporaryDirectory() as scratch, closing(BridgeSession(Path(scratch))) as bridge:
        live = DesktopEpisode(bridge, args.source, output, landmarks, args.resume)
        if live.resume is not None:
            phases = ("approach", "scalp", "bone", "dura", "treatment", "closure")
            if phases.index(args.through) < phases.index(live.phase):
                parser.error("The selected checkpoint is already beyond the requested phase")
        try:
            if args.recorded_inputs:
                perform_recorded(live, args.recorded_inputs)
            else:
                perform(live, landmarks, args.through)
        except Exception:
            live.save("last-accepted")
            raise
        live.save("complete" if args.through == "closure" else "early-" + args.through)


if __name__ == "__main__":
    main()
