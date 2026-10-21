"""Opt-in independent first-release artifact audit; no inference or patient edits.

Run with this checkout's src on PYTHONPATH. --artifacts-root points to the checkout
holding retained outputs. This file is deliberately outside pytest discovery.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback
from dataclasses import replace
from unittest.mock import patch
from urllib.parse import unquote, urlparse
from zipfile import ZipFile

import nibabel as nib
import numpy as np
from resectionlab.desktop_bridge import BridgeRuntime
from resectionlab.workspace_bundle import load_workspace

ROOT = Path(__file__).resolve().parents[1]
CASES = ("Lausanne-sub-000-ses-20110101", "SynthRAD2023-1BA336", "UCSF-PDGM-0004")
RECORDS = (
    "UCSF-PDGM-0004/synthstrip-main/fe693f2aed5cb0bb90aa",
    "UCSF-PDGM-0004/monai-brats-0.5.4/af24e4d8f13e49e37ff8",
    "SynthRAD2023-1BA336/totalsegmentator-skull/32e4c06b003e5b904387",
    "Lausanne-sub-000-ses-20110101/vesselfm-base/0cf5010a6c39c81d7f4b",
)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def local(uri):
    return Path(unquote(urlparse(uri).path))


def geometry(image):
    return {"shape": list(image.shape), "spacing_mm": list(map(float, nib.affines.voxel_sizes(image.affine))),
            "affine_ras_mm": image.affine.tolist(), "axis_codes": list(nib.aff2axcodes(image.affine)),
            "units": image.header.get_xyzt_units()[0], "qform_code": int(image.header["qform_code"]),
            "sform_code": int(image.header["sform_code"])}


class Client:
    def __init__(self, path):
        self.events, self.serial = [], 0
        self.runtime = BridgeRuntime(path, self.events.append)

    def call(self, op, args, error=False):
        self.serial += 1
        key = str(self.serial)
        self.runtime.submit({"id": key, "op": op, "args": args})
        assert self.runtime.wait_idle(60), f"{op} did not complete"
        result = [row for row in self.events if row["id"] == key][-1]
        assert result["event"] == ("error" if error else "result"), result
        return result if error else result["result"]

    def close(self):
        self.runtime.close()


def model_audit(base, watch):
    rows = []
    for relative in RECORDS:
        path = base / "outputs/poc-models" / relative / "inference.json"
        record = json.loads(path.read_text())
        assert record["status"] == "succeeded" and record["exit_code"] == 0
        inputs = record.get("inputs") or [record["input"]]
        all_items = inputs + record["weights"] + record["outputs"]
        for item in all_items:
            p = Path(item["path"])
            assert sha(p) == item["sha256"], p
            if "bytes" in item:
                assert p.stat().st_size == item["bytes"], p
            watch[str(p)] = (p.stat().st_mtime_ns, sha(p))
        watch[str(path)] = (path.stat().st_mtime_ns, sha(path))
        source = nib.load(inputs[0]["path"])
        assert np.isfinite(source.affine).all() and abs(np.linalg.det(source.affine[:3, :3])) > 0
        source_geometry = geometry(source)
        for item in inputs:
            image = nib.load(item["path"])
            assert image.shape == source.shape and np.array_equal(image.affine, source.affine)
            assert list(image.shape) == item["shape"] and np.array_equal(image.affine, item["affine_ras_mm"])
        outputs = []
        for item in record["outputs"]:
            image = nib.load(item["path"])
            array = np.asanyarray(image.dataobj)
            assert image.shape == source.shape and np.array_equal(image.affine, source.affine)
            assert int(np.count_nonzero(array)) == item["foreground_voxels"] > 0
            assert np.isfinite(array).all()
            outputs.append({"name": item["name"], "sha256": sha(item["path"]),
                            "foreground_voxels": int(np.count_nonzero(array)),
                            "foreground_ml": float(np.count_nonzero(array) * abs(np.linalg.det(source.affine[:3, :3])) / 1000),
                            "values": np.unique(array).tolist(), "exact_source_geometry": True})
        rows.append({"case": record["case"], "model": record["model_id"], "record_path": str(path),
                     "record_sha256": sha(path), "source_geometry": source_geometry,
                     "input_hashes": [{"path": i["path"], "sha256": i["sha256"], "modality": i.get("modality")} for i in inputs],
                     "weights": record["weights"], "model_identity": record["model"], "outputs": outputs,
                     "recorded_runtime_seconds": record.get("elapsed_seconds"),
                     "recorded_sampled_peak_rss_bytes": record.get("peak_process_rss_bytes_sampled", record.get("peak_process_family_rss_bytes_sampled")),
                     "measurement_caveat": record.get("memory_caveat"), "inference_rerun_by_qa": False})
    return rows


def original_audit(path):
    with ZipFile(path) as z:
        state = json.loads(z.read("workspace.json"))
        records = []
        for row in state["originalFiles"]:
            raw = z.read(row["member"])
            assert len(raw) == row["bytes"]
            assert hashlib.sha256(raw).hexdigest() == row["sha256"] == sha(local(row["sourceUri"]))
            records.append({k: row[k] for k in ("name", "sha256", "bytes", "role", "sourceUri")})
        return records


def case_audit(base, name, directory):
    path = base / "outputs/poc-cases" / f"{name}.ressectionlab"
    archive_sha = sha(path)
    originals = original_audit(path)
    case, _, saved = load_workspace(path)
    assert case.case_id == name
    for source in [case] + [x.case for x in saved["sources"]]:
        ref = next(r for r in source.source_refs if r.source_id == "structural")
        image = nib.load(local(ref.uri))
        assert np.array_equal(image.affine, source.affine)
        assert np.array_equal(np.asarray(image.dataobj, dtype=np.float32), source.mri)
    layers = []
    for name_, metadata in case.metadata["anatomy_layers"].items():
        source = local(metadata["source_uri"])
        assert sha(source) == metadata["source_sha256"]
        image = nib.load(source)
        array = np.asarray(image.dataobj)
        expected = array == metadata["label_value"] if metadata["source_role"] == "imported_source_annotation" else array.astype(bool)
        actual = case.brain_mask if name_ == "brain" and case.brain_mask is not None else case.compartments[name_]
        assert np.array_equal(actual, expected) and np.array_equal(image.affine, case.affine)
        if metadata["source_role"] == "cached_model_inference":
            record_path = source.parent / "inference.json"
            record = json.loads(record_path.read_text())
            assert sha(record_path) == metadata["inference_manifest_sha256"]
            assert record["case"] == case.case_id and record["cache_key"] == metadata["cache_key"]
            assert metadata["provenance"] == "estimated"
        else:
            assert metadata["model"] is None and metadata["provenance"] == "source-provided"
        layers.append({"name": name_, "provenance": metadata["provenance"], "source_sha256": sha(source),
                       "foreground_voxels": int(np.count_nonzero(actual)), "stored_mask_exact": True})
    first, second = Client(directory / "a"), None
    try:
        with patch("subprocess.Popen", side_effect=AssertionError("No inference may launch during open/save")):
            payload = first.call("loadCase", {"path": str(path)})
        state = copy.deepcopy(payload["workspaceSession"]["imagingState"])
        for item in state["states"].values():
            cursor = item["cursor"]
            item["viewer"] = {"camera": {"position": [cursor[0]+180, cursor[1]-240, cursor[2]+130], "target": cursor, "up": [0, 0, 1]},
                              "layerOpacity": {k: .37 for k in item["visibleLayers"]},
                              "clipping": {"axis": "z", "value": cursor[2], "enabled": True, "flipped": False},
                              "layout": "mri-review", "activePlane": "coronal", "showPlane": True, "contrast": 1.2}
        dest = directory / "reopened.ressectionlab"
        with patch("subprocess.Popen", side_effect=AssertionError("No inference may launch during open/save")):
            first.call("saveCase", {"caseHash": payload["caseHash"], "path": str(dest), "workspace": {"imaging": state}})
            first.close(); first = None
            second = Client(directory / "b")
            reopened = second.call("loadCase", {"path": str(dest)})
        assert reopened["caseHash"] == payload["caseHash"]
        assert reopened["workspaceSession"]["imagingState"] == state
        assert original_audit(dest) == originals
        labels = [x["name"] for x in reopened["compartments"]]
        if reopened["brainMask"] is not None and "brain" not in labels:
            labels.append("brain")
        fixture = directory / "hydration.json"
        fixture.write_text(json.dumps({"payload": reopened, "transferRoot": str(directory / "b"), "expectedLayers": labels}))
        child = subprocess.run([os.environ["NODE_EXECUTABLE"], "--experimental-strip-types", str(ROOT / "desktop/tests/poc-acceptance-hydration.mjs"), str(fixture)], capture_output=True, text=True, timeout=60)
        assert child.returncode == 0, child.stderr
        hydration = json.loads(child.stdout)
        assert sha(path) == archive_sha
        return {"case": name, "bundle_sha256": archive_sha, "original_files": len(originals), "originals": originals,
                "case_hash": payload["caseHash"], "layers": layers, "source_volumes_exact": True,
                "save_reopen_state_exact": True, "source_archive_unchanged": True,
                "current_frontend_hydration": hydration, "gui_covered": False}
    finally:
        if first: first.close()
        if second: second.close()


def reference_audit(base):
    source = base / "data/public_mirrors/MedOtter-UCSF-PDGM/e9372219cf1cd2fdd52260cd45f7514b4aa7638e/UCSF-PDGM-0004/UCSF-PDGM-0004_tumor_segmentation.nii.gz"
    reference = nib.load(source)
    labels = np.asarray(reference.dataobj)
    folder = base / "outputs/poc-models/UCSF-PDGM-0004/monai-brats-0.5.4/af24e4d8f13e49e37ff8"
    rows = {}
    for key, values in (("whole_tumor", [1, 2, 4]), ("tumor_core", [1, 4]), ("enhancing_tumor", [4])):
        image = nib.load(folder / f"monai_{key}_mask.nii.gz")
        assert image.shape == reference.shape and np.array_equal(image.affine, reference.affine)
        pred, truth = np.asarray(image.dataobj).astype(bool), np.isin(labels, values)
        tp = int(np.count_nonzero(pred & truth)); fp = int(np.count_nonzero(pred & ~truth)); fn = int(np.count_nonzero(~pred & truth))
        rows[key] = {"TP_voxels": tp, "FP_voxels": fp, "FN_voxels": fn, "dice": 2*tp/(2*tp+fp+fn),
                     "precision": tp/(tp+fp), "recall": tp/(tp+fn), "inferred_voxels": int(pred.sum()), "reference_voxels": int(truth.sum())}
    return {"scope": "Within-case source-annotation overlap; NOT held-out accuracy or clinical validation. Known BraTS2021 training overlap; this checkpoint's patient exclusion is not established.",
            "source_annotation_sha256": sha(source), "source_labels": np.unique(labels).tolist(), "comparisons": rows,
            "skull_reference": "No reference skull annotation identified for this case", "vessel_reference": "No reference vessel annotation identified for sub-000", "brain_extraction_reference": "No independent brain reference annotation identified; skull-stripped input is not independent ground truth"}


def box_capsule_cells(affine, shape, a, b, radius):
    """Independent exact AABB/capsule oracle for source-axis-aligned segments.

    Does not call the runtime's geometric enumeration or collision checker.
    Applicable here because both real branches use only source-axis translation.
    """
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    axes = affine[:3, :3] / spacing
    start = axes.T @ (np.asarray(a)-affine[:3, 3])
    end = axes.T @ (np.asarray(b)-affine[:3, 3])
    changing = np.flatnonzero(np.abs(end-start) > 1e-8)
    assert len(changing) <= 1, "Oracle restricted to source-axis-aligned segments"
    lo = np.maximum(0, np.floor((np.minimum(start, end)-radius)/spacing-.5).astype(int))
    hi = np.minimum(np.asarray(shape)-1, np.ceil((np.maximum(start, end)+radius)/spacing+.5).astype(int))
    if np.any(lo > hi): return set()
    indices = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i]+1) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    centers = indices * spacing
    # For an axis-aligned segment its Cartesian extent is the segment itself.
    # Distance to a cell is the norm of the three disjoint interval gaps.
    gap = np.maximum(0, np.maximum(np.minimum(start, end)-(centers+.5*spacing), (centers-.5*spacing)-np.maximum(start, end)))
    hit = indices[np.sum(gap*gap, axis=1) <= radius*radius+1e-9]
    return set(map(int, np.ravel_multi_index(hit.T, shape))) if len(hit) else set()


def native_ids(rows, shape):
    return set(map(int, np.ravel_multi_index(np.asarray(rows, dtype=int).T, shape))) if rows else set()


def independent_episode_frames(case, exported):
    """Recompute cell removal, contact, field contact and accounting per frame."""
    support = np.asarray(case.brain_mask).ravel()
    target = (case.compartments["monai_tumor_core"] & case.brain_mask).ravel()
    shape, affine = case.mri.shape, case.affine
    exposure = native_ids(exported["assumedExposureIndicesNative"], shape)
    initial = exported["replayFrames"][0]
    first_tool = next(t for t in initial["tools"] if t["toolId"] == "aspirator")
    point, axis = np.array(first_tool["tipRasMm"]), np.array(first_tool["axis"])
    expected_exposure = box_capsule_cells(affine, shape, point-100*axis, point, 9.)
    expected_exposure = {i for i in expected_exposure if support[i] and
        ((affine[:3, :3] @ np.array(np.unravel_index(i, shape)) + affine[:3, 3])-point) @ axis <= first_tool["tipRadiusMm"]+.5*max(nib.affines.voxel_sizes(affine))+1e-8}
    assert exposure == expected_exposure
    removed, contacts, field_managed, burden = set(), {}, {}, 0.
    previous = initial
    voxel = float(abs(np.linalg.det(affine[:3, :3])))
    for frame in exported["replayFrames"][1:]:
        assert frame["stateBefore"] == previous["stateAfter"]
        dt = frame["elapsedSeconds"]-previous["elapsedSeconds"]
        assert dt >= 0
        old = {t["toolId"]: t for t in previous["tools"]}
        new = {t["toolId"]: t for t in frame["tools"]}
        delta, field_contact = set(), set()
        if frame["kind"] == "aspirate":
            t = new["aspirator"]
            begin = np.array(old["aspirator"]["tipRasMm"])-t["tipLengthMm"]*np.array(t["axis"])
            hit = box_capsule_cells(affine, shape, begin, t["tipRasMm"], t["tipRadiusMm"])
            delta = {i for i in hit if support[i]}-exposure-removed
        elif frame["kind"] == "clear_field":
            t = new["field_manager"]
            begin = np.array(t["tipRasMm"])-t["tipLengthMm"]*np.array(t["axis"])
            hit = box_capsule_cells(affine, shape, begin, t["tipRasMm"], 2.5)
            field_contact = {i for i in hit if support[i]}-exposure-removed
        assert delta == native_ids(frame["removedIndicesNative"], shape)
        assert field_contact == native_ids(frame["fieldContactIndicesNative"], shape)
        assert delta | field_contact == native_ids(frame["contactIndicesNative"], shape)
        assert not removed & delta
        removed |= delta
        for who,t in new.items():
            p0=np.array(old[who]["tipRasMm"]);p1=np.array(t["tipRasMm"]);direction=np.array(t["axis"])
            # Both actual branches move axially forward; the swept union of each
            # shaft/tip capsule is another axis-aligned capsule.
            shaft=box_capsule_cells(affine,shape,p0-t["workingLengthMm"]*direction,p1-t["tipLengthMm"]*direction,t["shaftRadiusMm"])
            tip_radius=t["shaftRadiusMm"] if who=="aspirator" and frame["kind"]=="aspirate" else t["tipRadiusMm"]
            tip_cells=box_capsule_cells(affine,shape,p0-t["tipLengthMm"]*direction,p1,tip_radius)
            blocked={i for i in shaft|tip_cells if support[i]}-exposure-removed
            assert not blocked,(frame["frameIndex"],who,len(blocked))
        for i in delta | field_contact: contacts[i] = contacts.get(i, 0)+dt
        for i in field_contact: field_managed[i] = field_managed.get(i, 0)+dt
        burden = min(1., burden+len(delta)*.002)
        if field_contact: burden = max(0., burden-.2*dt)
        m = frame["metrics"]
        target_removed = sum(bool(target[i]) for i in removed)
        assumed_target = sum(bool(target[i]) for i in exposure)
        expected = {"removedMm3":len(removed)*voxel, "targetRemovedMm3":target_removed*voxel,
                    "nonTargetRemovedMm3":(len(removed)-target_removed)*voxel,
                    "residualTargetMm3":(int(target.sum())-assumed_target-target_removed)*voxel,
                    "remainingTissueMm3":(int(support.sum())-len(exposure)-len(removed))*voxel,
                    "contactCellSeconds":sum(contacts.values()), "fieldManagementCellSeconds":sum(field_managed.values()), "fieldBurden":burden}
        for key, value in expected.items(): assert np.isclose(m[key], value, rtol=0, atol=1e-8), (key, m[key], value)
        assert np.isclose(m["remainingTissueMm3"]+m["removedMm3"]+m["assumedExposureTissueMm3"], m["sourceTissueMm3"])
        assert np.isclose(m["residualTargetMm3"]+m["targetRemovedMm3"]+m["assumedExposureTargetMm3"], m["sourceTargetMm3"])
        assert m["vascularEvents"] is m["motorRisk"] is m["languageRisk"] is None
        # Both real programs keep parallel instruments separated by 6 mm.
        # Independently certify their full axis-aligned capsule envelopes using
        # the lateral centerline separation, a lower bound on segment distance.
        ta, tb = new["aspirator"], new["field_manager"]
        a = np.array(ta["axis"]); assert np.allclose(a, tb["axis"])
        separation = np.array(tb["tipRasMm"])-np.array(ta["tipRasMm"])
        lateral = np.linalg.norm(separation-(separation@a)*a)
        assert lateral > max(ta["tipRadiusMm"], ta["shaftRadiusMm"])+max(tb["tipRadiusMm"], tb["shaftRadiusMm"])
        previous = frame
    assert removed == native_ids(exported["finalRemovedIndicesNative"], shape)
    return {"frames": len(exported["replayFrames"]), "independent_capsule_cell_removal_exact":True,
            "persistent_unique_removal":True,"source_tissue_and_target_accounting_exact":True,
            "contact_field_management_and_burden_accounting_exact":True,"full_tool_parallel_clearance":True,
            "independent_full_shaft_and_tip_retained_tissue_clearance":True,
            "metrics":exported["metrics"]}


def episode_audit(base):
    from resectionlab.rehearsal_episode import create_episode, restore_episode, execute_alternatives
    from resectionlab.core import semantic_digest, thaw_json
    case, _, _ = load_workspace(base / "outputs/poc-cases/UCSF-PDGM-0004.ressectionlab")
    episode = create_episode(case)
    before = episode.serialize()
    report = {"scope":"Independent real-case replay/cell accounting and generated adversarial software controls, not physiological validation", "source_case_hash":case.semantic_hash,"checks":{},"failures":[]}
    def check(name, operation):
        try:
            report["checks"][name] = operation()
        except Exception:
            report["failures"].append({"check":name,"traceback":traceback.format_exc()})
    def branches():
        result = execute_alternatives(episode)
        assert episode.serialize() == before and result["snapshot"] == before
        rows=[]
        for row in result["branches"]:
            assert row["branchPointStateId"] == episode.state_id
            assert not row["attemptDiagnostics"]
            exported=row["episode"]
            assert restore_episode(case,json.loads(json.dumps(exported["savedEpisode"]))).export() == exported
            rows.append({"id":row["id"],**independent_episode_frames(case,exported)})
        assert rows[0]["metrics"]["targetRemovedMm3"] != rows[1]["metrics"]["targetRemovedMm3"]
        return rows
    check("two_alternatives_exact_replay_and_independent_accounting",branches)
    def persistence():
        e=create_episode(case);tip=e.poses["aspirator"]
        assert e.step({"kind":"aspirate","toolId":"aspirator","tipRasMm":(tip.tip_mm+2*tip.axis_unit).tolist(),"durationSeconds":2.})["accepted"]
        serial=e.serialize(); clone=e.fork()
        assert clone.state_id==e.state_id and clone.pending_events==e.pending_events and clone.contact==e.contact
        assert clone.step({"kind":"inspect","durationSeconds":1.})["accepted"]
        assert e.serialize()==serial and e.removed==clone.removed
        assert restore_episode(case,json.loads(json.dumps(serial))).export()==e.export()
        assert clone.step({"kind":"aspirate","toolId":"aspirator","durationSeconds":1.})["accepted"]
        assert clone.removed==e.removed
        return {"saved_nonempty_cavity_contact_pending_events_replayed":True,"fork_mutations_isolated":True,"repeated_aspiration_no_double_removal":True}
    check("nonempty_snapshot_persistence",persistence)
    def observation_isolation():
        e=create_episode(case);snapshot=e.serialize();obs=e.planning_observation()
        obs["targetNames"].append("not_authorized")
        assert e.serialize()==snapshot,"Planning observation aliases authoritative config"
        f=e.fork()
        try:f.config["target_names"].append("not_authorized")
        except (TypeError, AttributeError):pass
        assert e.serialize()==snapshot,"Fork config aliases parent"
        return True
    check("planning_observation_and_fork_detached",observation_isolation)
    def hidden_independence():
        hidden=np.zeros(case.mri.shape,bool);hidden[20:23,20:23,20:23]=True
        changed=replace(case,source_compartments={"privileged_evaluator_target":hidden},metadata={**thaw_json(case.metadata),"private_future_outcome":"QA-private-label"})
        altered=create_episode(changed)
        assert altered.planning_observation()==episode.planning_observation()
        assert execute_alternatives(altered)["branches"][0]["strategy"]==execute_alternatives(episode)["branches"][0]["strategy"]
        assert "QA-private-label" not in json.dumps(altered.planning_observation())
        assert changed.semantic_hash != case.semantic_hash
        return {"private_source_compartments_and_future_metadata_do_not_change_deployed_observation_or_strategy":True}
    check("privileged_information_intervention",hidden_independence)
    def collision_controls():
        e=create_episode(case);a,b=e.poses["aspirator"],e.poses["field_manager"];snapshot=e.serialize()
        # Endpoint tips are 8 mm apart, but 100 mm shafts lie on the same line.
        action={"kind":"move","toolId":"field_manager","tipRasMm":(a.tip_mm-8*a.axis_unit).tolist(),"durationSeconds":3.}
        result=e.step(action);assert not result["accepted"] and "TOOL_COLLISION" in result["reason"]
        assert e.serialize()==snapshot
        # Swap 6 mm-separated parallel tools: clear endpoints, intersecting midpoint.
        action={"kind":"move","motions":[{"toolId":"aspirator","tipRasMm":b.tip_mm.tolist()},{"toolId":"field_manager","tipRasMm":a.tip_mm.tolist()}],"durationSeconds":2.}
        result=e.step(action);assert not result["accepted"] and "TOOL_COLLISION" in result["reason"]
        assert e.serialize()==snapshot
        # Explicitly generated obstacle, located on shaft 20 mm behind tip.
        obstacle=np.zeros(case.mri.shape,bool)
        voxel=np.rint(np.linalg.inv(case.affine)@np.r_[a.tip_mm-20*a.axis_unit,1]).astype(int)[:3]
        assert all(0<=voxel[i]<case.mri.shape[i] for i in range(3))
        obstacle[tuple(voxel)]=True
        metadata=thaw_json(case.metadata);metadata["anatomy_layers"]["qa_vessel_control"]={"kind":"vessels","source_sha256":"a"*64,"source_role":"explicit_generated_QA_obstacle"}
        controlled=replace(case,compartments={**case.compartments,"qa_vessel_control":obstacle},metadata=metadata)
        try:create_episode(controlled)
        except ValueError as error: assert "conflicts" in str(error)
        else:raise AssertionError("Full shaft anatomical collision admitted")
        return {"separated_tips_colliding_shafts_rejected":True,"synchronized_midpoint_collision_rejected":True,"generated_shaft_anatomy_collision_rejected":True,"rejections_atomic":True}
    check("full_tool_swept_and_anatomical_collision_controls",collision_controls)
    def identity_controls():
        modified=copy.deepcopy(before);modified["actions"].append({"kind":"inspect","durationSeconds":1.})
        try:restore_episode(case,modified)
        except ValueError:pass
        else:raise AssertionError("Tampered serialized actions accepted")
        for alt in (replace(case,case_id=case.case_id+"-different-person"),replace(case,affine=case.affine+np.array([[0,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]))):
            try:restore_episode(alt,before)
            except ValueError:pass
            else:raise AssertionError("Changed source/frame accepted")
        rehashed=copy.deepcopy(before);rehashed["finalStateId"]="sha256:"+"1"*64
        rehashed["recordHash"]=semantic_digest({k:v for k,v in rehashed.items() if k!="recordHash"})
        try:restore_episode(case,rehashed)
        except ValueError:pass
        else:raise AssertionError("Self-consistent hash with wrong final state accepted")
        return {"tampering_wrong_patient_wrong_frame_wrong_terminal_state_refused":True}
    check("serialization_source_frame_and_action_integrity",identity_controls)
    def stop_budget():
        e=create_episode(case)
        for _ in range(128): assert e.step({"kind":"inspect","durationSeconds":.001})["accepted"]
        assert e.step({"kind":"STOP","durationSeconds":0.})["accepted"]
        assert restore_episode(case,e.serialize()).state_id==e.state_id
        return True
    check("budget_terminal_stop_save_reopen",stop_budget)
    report["passed"]=not report["failures"]
    return report


def episode_bridge_audit(base):
    path = base / "outputs/poc-cases/UCSF-PDGM-0004.ressectionlab"
    source_digest=sha(path)
    case,_,_=load_workspace(path)
    report={"scope":"Actual Python desktop operation boundary with temporary saved case, fresh client restore and independent replay accounting; not Electron GUI","failures":[]}
    with tempfile.TemporaryDirectory(prefix="first-release-bridge-independent-") as temp:
        directory=Path(temp);a=Client(directory/"a");b=None
        try:
            p=a.call("loadCase",{"path":str(path)});key=p["caseHash"]
            created=a.call("createRehearsal",{"caseHash":key})
            stale_id=created["stateId"]
            tool=next(t for t in created["episode"]["replayFrames"][0]["tools"] if t["toolId"]=="aspirator")
            tip=(np.array(tool["tipRasMm"])+2*np.array(tool["axis"])).tolist()
            advanced=a.call("stepRehearsal",{"caseHash":key,"expectedStateId":stale_id,"action":{"kind":"aspirate","toolId":"aspirator","durationSeconds":2.,"tipRasMm":tip}})
            assert advanced["episode"]["metrics"]["removedMm3"]>0
            stale=a.call("stepRehearsal",{"caseHash":key,"expectedStateId":stale_id,"action":{"kind":"inspect","durationSeconds":1.}},error=True)
            assert stale["error"]["code"]=="REHEARSAL_STATE_STALE"
            compared=a.call("compareRehearsal",{"caseHash":key,"expectedStateId":advanced["stateId"]})
            assert compared["stateId"]==advanced["stateId"]
            assert compared["comparison"]["snapshotId"]==advanced["stateId"]
            prefix=advanced["episode"]["replayFrames"]
            for branch in compared["comparison"]["alternatives"]:
                assert branch["episode"]["replayFrames"][:len(prefix)]==prefix
            view={"branchId":compared["comparison"]["alternatives"][0]["id"],"frameIndex":len(prefix),"selectedTool":"field_manager","visible":True,"episodeInitialStateId":created["episode"]["initialStateId"]}
            dest=directory/"rehearsal-saved.ressectionlab"
            a.call("saveCase",{"caseHash":key,"path":str(dest),"workspace":{"rehearsalView":view}})
            assert original_audit(dest)==original_audit(path)
            rejected=directory/"forged.ressectionlab"
            a.call("saveCase",{"caseHash":key,"path":str(rejected),"workspace":{"patient_rehearsal":{}}},error=True)
            assert not rejected.exists()
            a.close();a=None
            b=Client(directory/"b")
            reopened=b.call("loadCase",{"path":str(dest)})
            restored=b.call("inspectRehearsal",{"caseHash":reopened["caseHash"]})
            assert restored["savedView"]==view
            assert restored["episode"]==compared["episode"]
            assert restored["comparison"]==compared["comparison"]
            report["branches"]=[{"id":branch["id"],**independent_episode_frames(case,branch["episode"])} for branch in restored["comparison"]["alternatives"]]
            report.update({"case_hash":key,"saved_bundle_sha256":sha(dest),"original_bytes_preserved":True,
                           "live_state_exact_after_fresh_client_reopen":True,"two_branches_exact_after_fresh_client_reopen":True,
                           "branch_frame_prefixes_equal_saved_nonempty_snapshot":True,"view_restored_exact":True,
                           "stale_actions_refused":True,"renderer_cannot_replace_episode_artifact":True,
                           "source_archive_unchanged":sha(path)==source_digest})
        except Exception:report["failures"].append(traceback.format_exc())
        finally:
            if a:a.close()
            if b:b.close()
    report["passed"]=not report["failures"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/poc-first-qa/gate1-independent.json")
    parser.add_argument("--episode-only", action="store_true", help="Audit the first source-bound two-tool episode")
    parser.add_argument("--bridge-only", action="store_true", help="Audit source-bound episode desktop operations and persistence")
    args = parser.parse_args()
    if args.episode_only or args.bridge_only:
        watched_sources={name:sha(ROOT/name) for name in ["tests/first_release_qa.py","src/resectionlab/rehearsal_episode.py","src/resectionlab/rehearsal_bridge.py","src/resectionlab/desktop_bridge.py","src/resectionlab/geometry.py","src/resectionlab/evaluation.py"]}
        report=episode_bridge_audit(args.artifacts_root) if args.bridge_only else episode_audit(args.artifacts_root)
        report["tested_source_sha256"]=watched_sources
        report["tested_sources_unchanged_during_check"]=all(sha(ROOT/name)==value for name,value in watched_sources.items())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"passed":report["passed"],"failures":report["failures"],"output":str(args.output)}))
        raise SystemExit(0 if report["passed"] else 1)
    started = time.monotonic()
    report = {"scope": "Independent local bytes, geometry, archive, bridge save/reopen and current frontend hydration; no inference, GUI, physical or clinical validation", "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "failures": [], "cases": []}
    watch = {}
    for label, fn in (("models", lambda: model_audit(args.artifacts_root, watch)), ("references", lambda: reference_audit(args.artifacts_root))):
        try: report[label] = fn()
        except Exception: report["failures"].append({"section": label, "traceback": traceback.format_exc()})
    with tempfile.TemporaryDirectory(prefix="first-release-independent-") as temporary:
        for name in CASES:
            directory = Path(temporary) / name; directory.mkdir()
            try:
                report["cases"].append(case_audit(args.artifacts_root, name, directory))
                print("PASS", name, flush=True)
            except Exception:
                report["failures"].append({"section": name, "traceback": traceback.format_exc()})
                print("FAIL", name, flush=True)
    report["model_cache_unchanged"] = all((Path(p).stat().st_mtime_ns, sha(p)) == before for p, before in watch.items())
    report["elapsed_seconds"] = time.monotonic()-started
    report["passed"] = not report["failures"] and report["model_cache_unchanged"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"passed": report["passed"], "failures": report["failures"], "output": str(args.output)}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
