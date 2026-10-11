"""Generated-only phase glue; never reads a Case2 source or patient array."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
for path in (BASE, ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))
import case2_adapter as adapter
import run_phase as runner


def generated(monkeypatch):
    source = [(0,0,0),(10,0,0),(0,10,0),(0,0,10),(10,10,0),(10,0,10),
              (0,10,10),(10,10,10),(3,4,5),(7,2,4),(2,8,6),(8,7,3),(4,6,9),(6,3,8)]
    observed = [(x+1+.001*y*y, y+.5, z+.2) for x,y,z in source]
    lines = [" ".join(map(str, (*s, *d))) for s,d in zip(source,observed)]
    payload = ("MNI Tag Point File\nVolumes = 2;\nPoints =\n"+"\n".join(lines)+"\n;\n").encode()
    monkeypatch.setattr(adapter, "TAG_SHA", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(adapter, "TAG_BYTES", len(payload))
    header = {"shape": [100,100,100], "selected_affine": np.eye(4).tolist()}
    frame = {"status": "coordinate_convention_verified", "source_image_sha256": adapter.BEFORE_SHA,
        "destination_image_sha256": adapter.DURING_SHA, "convention": adapter.CONVENTION,
        "source_world_to_ras_mm": np.eye(4).tolist(), "destination_world_to_ras_mm": np.eye(4).tolist(),
        "anatomical_alignment_accepted": False, "physical_clearance_mm": None,
        "cavity_support": None, "total_registration_uncertainty_mm": None,
        "headers": {"before": header, "during": header}}
    return payload, frame


def test_rotated_coverage_refuses_world_aabb_false_positive():
    angle = np.pi/4
    affine = np.eye(4)
    affine[:2,:2] = [[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    point = [[1.3,0,0]]  # Inside world AABB; outside native y=-.5 face.
    result = adapter.native_coverage(point,[1],{"shape":[2,2,2],"selected_affine":affine.tolist()})
    assert result["rows"][0]["inside_image_cell_box"] is False
    assert result["excluded_rows"] == []


def test_closed_full_cell_boundary_and_no_exclusion():
    points = [[-.5,0,0],[1.5,0,0],[np.nextafter(1.5,np.inf),0,0]]
    result = adapter.native_coverage(points,[1,2,3],{"shape":[2,2,2],"selected_affine":np.eye(4).tolist()})
    assert [row["inside_image_cell_box"] for row in result["rows"]] == [True,True,False]
    assert len(result["rows"]) == 3 and not result["excluded_rows"]


def test_primary_is_difference_of_rms_not_mean_error_difference():
    report = {"validation_landmarks":2,"methods":{
        "proper_rigid":{"all_supported":{"count":2,"rms_mm":np.sqrt(2)}},
        "no_shift":{"all_supported":{"count":2,"rms_mm":2}}}}
    result = adapter.primary_rms_difference(report)
    assert result["rms_difference_mm"] == pytest.approx(np.sqrt(2)-2)
    assert result["rms_difference_mm"] != pytest.approx(-1)  # mean([0,2])-mean([2,2])
    report["methods"]["proper_rigid"]["all_supported"]["count"] = 1
    assert adapter.primary_rms_difference(report)["rms_difference_mm"] is None


def test_exact_convention_rejected_before_B(monkeypatch):
    payload, frame = generated(monkeypatch)
    partition = adapter.source_partition(payload)
    frame["convention"] += " changed"
    monkeypatch.setattr(adapter.lm,"build_forward_landmarks",lambda *a: pytest.fail("B must stay closed"))
    with pytest.raises(ValueError,match="QUALIFIED_FRAME"):
        adapter._prepared(payload,partition,frame,"f"*64)


def test_frame_bytes_hash_bound(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    path = tmp_path / "frame.json"
    payload = b'{"convention":"original"}\n'
    path.write_bytes(payload)
    binding = {"path":"frame.json","sha256":hashlib.sha256(payload).hexdigest()}
    assert runner.checked(binding,path)["convention"] == "original"
    path.write_text('{"convention":"changed"}')
    with pytest.raises(ValueError,match="BYTES_CHANGED"):
        runner.checked(binding,path)


def test_generated_partition_fit_freeze_then_all_V_once(tmp_path,monkeypatch):
    payload, frame = generated(monkeypatch)
    partition = adapter.source_partition(payload)
    assert partition["destination_coordinates_parsed"] == 0
    assert len(partition["B_ids"]) == 6 and len(partition["V_ids"]) == 8
    assert not set(partition["B_ids"]) & set(partition["V_ids"])
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"generated_control":true}\n')
    fit = adapter.fit_and_freeze(tmp_path,payload=payload,partition_record=partition,
        frame=frame,frame_sha256="f"*64,
        protocol_binding={"path":"protocol.json","sha256":hashlib.sha256(protocol.read_bytes()).hexdigest()},
        output_directory="comparison")
    assert fit["V_destinations_accessed"] == []
    assert not (tmp_path / "comparison/validation-attempt.json").exists()
    result = adapter.evaluate_once(tmp_path,payload=payload,partition_record=partition,
        frame=frame,frame_sha256="f"*64,freeze_binding=fit["freeze"])
    assert result["validation_rows"] == 8
    assert len(result["V_destination_coverage"]["rows"]) == 8
    assert result["primary_comparison"]["status"] == "available"
    assert (tmp_path / "comparison/validation-attempt.json").exists()
    with pytest.raises((ValueError,FileExistsError)):
        adapter.evaluate_once(tmp_path,payload=payload,partition_record=partition,
            frame=frame,frame_sha256="f"*64,freeze_binding=fit["freeze"])


def test_exact_parent_dependency_keys():
    assert not runner.dependency_paths("qualify")
    assert set(runner.dependency_paths("fit")) == {
        "qualify_frame","qualify_partition","qualify_result","qualify_supervision"}
    assert set(runner.dependency_paths("evaluate")) == {
        "qualify_frame","qualify_partition","qualify_result","qualify_supervision",
        "fit_result","fit_supervision","fit_freeze"}


def test_prior_phase_success_and_exact_output_joins():
    release = {"source_index":{"sha256":"a"*64},"dependencies":{
        "qualify_result":{"sha256":"b"*64},"qualify_frame":{"sha256":"c"*64},
        "qualify_partition":{"sha256":"d"*64}}}
    result = {"status":"completed","phase":"qualify","source_index_sha256":"a"*64,
        "release_sha256":"e"*64,"frame":release["dependencies"]["qualify_frame"],
        "partition":release["dependencies"]["qualify_partition"]}
    receipt = {"status":"completed","exit_code":0,"scientific_result_accepted":True,
        "source_index_sha256":"a"*64,"release_sha256":"e"*64,"result_sha256":"b"*64}
    deps = {"qualify_result":result,"qualify_supervision":receipt}
    runner.validate_predecessors("fit",deps,release)
    receipt["scientific_result_accepted"] = False
    with pytest.raises(ValueError,match="PRIOR_PHASE"):
        runner.validate_predecessors("fit",deps,release)
    receipt["scientific_result_accepted"] = True
    result["frame"] = {"sha256":"f"*64}
    with pytest.raises(ValueError,match="OUTPUT_JOIN"):
        runner.validate_predecessors("fit",deps,release)
    with pytest.raises(KeyError):
        runner.validate_predecessors("evaluate",deps,release)


def test_unreleased_phase_refuses_before_source_or_patient_reads(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    monkeypatch.setattr(runner,"BASE",tmp_path / "stage")
    path = tmp_path / "stage/releases/qualify.json"
    path.parent.mkdir(parents=True)
    release = {"schema":runner.SCHEMA+"-release","phase":"qualify","authorizer":"root",
        "authorized":False,"source_index":{},"dependencies":{}}
    payload = runner.encode(release)
    path.write_bytes(payload)
    monkeypatch.setattr(runner,"source_hashes",lambda: pytest.fail("unreleased read"))
    with pytest.raises(ValueError,match="ROOT_PHASE_RELEASE"):
        runner.authenticate("qualify",path,runner.sha(payload))


def test_exclusive_output_and_aggregate_cap(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    monkeypatch.setattr(runner,"OUT",tmp_path / "out")
    path = tmp_path / "out/result.json"
    runner.save(path,{"generated":True})
    with pytest.raises(ValueError,match="EXCLUSIVE"):
        runner.save(path,{"generated":True})
    monkeypatch.setattr(runner,"CAP",1)
    with pytest.raises(ValueError,match="OUTPUT_CAP"):
        runner.save(tmp_path / "out/extra.json",{})
