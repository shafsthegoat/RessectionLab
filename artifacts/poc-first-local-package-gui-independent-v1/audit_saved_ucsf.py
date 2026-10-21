"""Opt-in independent saved-workspace audit; standard library only.

Streams NPY bytes from the two authorized archives; never imports the engine,
models, NumPy, or image readers. Only the 16 affine scalars are decoded.
"""
import argparse
import ast
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import resource
import struct
from datetime import datetime, timezone
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
BASE = "outputs/poc-cases/UCSF-PDGM-0004.ressectionlab"
SAVED = "outputs/poc-first-local-package-gui-v1/saved/UCSF-packaged-rehearsal-reviewed.ressectionlab"
PINS = {BASE: "73cd0792c025e846395bafffe5ec793ef9e1823ceec8e4b0aeb8317783041571",
        SAVED: "0294cd855debabcf6183a437b2b2f6222b873534206c6ba374da3ec7f46fc950"}
CHUNK = 1024 * 1024


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def exact(left, right):
    assert encoded(left) == encoded(right), "JSON values/types/ordered arrays differ"


def semantic(value):
    return "sha256:" + sha_bytes(encoded(value))


def stream_sha(handle):
    digest = hashlib.sha256()
    size = 0
    while data := handle.read(CHUNK):
        size += len(data)
        digest.update(data)
    return {"sha256": digest.hexdigest(), "bytes": size}


def logical_members(index, names=()):
    result = {"mri": "mri.npy", "affine": "affine.npy"}
    if "brain_mask.npy" in names:
        result["brain_mask"] = "brain_mask.npy"
    assert set(index) == {"compartments", "source_compartments"}
    for category, rows in index.items():
        for name, member in rows.items():
            assert re.fullmatch(category + r"_\d+", member)
            result[category + "/" + name] = member + ".npy"
    assert len(set(result.values())) == len(result), "Array-index alias"
    return result


def npy_record(handle):
    prefix = handle.read(8)
    assert prefix[:6] == b"\x93NUMPY" and prefix[6:] in (b"\x01\x00", b"\x02\x00")
    width = 2 if prefix[6] == 1 else 4
    length_raw = handle.read(width)
    length = int.from_bytes(length_raw, "little")
    assert length <= 65536
    header_raw = handle.read(length)
    header = ast.literal_eval(header_raw.decode("latin1").strip())
    assert set(header) == {"descr", "fortran_order", "shape"} and header["fortran_order"] is False
    assert header["descr"] in ("<f4", "<f8", "|b1")
    unit = {"<f4": 4, "<f8": 8, "|b1": 1}[header["descr"]]
    expected = math.prod(header["shape"]) * unit
    whole = hashlib.sha256(prefix + length_raw + header_raw)
    array_digest = hashlib.sha256(json.dumps({"shape": header["shape"], "dtype": header["descr"]}, sort_keys=True).encode())
    size = 0
    foreground = 0
    tiny = bytearray() if header["shape"] == (4, 4) and header["descr"] == "<f8" else None
    while data := handle.read(CHUNK):
        size += len(data)
        whole.update(data)
        array_digest.update(data)
        if header["descr"] == "|b1":
            assert data.count(0) + data.count(1) == len(data)
            foreground += data.count(1)
        if tiny is not None:
            tiny.extend(data)
    assert size == expected
    result = {"npy_sha256": whole.hexdigest(), "array_digest": "sha256:" + array_digest.hexdigest(),
              "dtype": header["descr"], "shape": list(header["shape"]), "payload_bytes": size}
    if header["descr"] == "|b1":
        result["foreground_cells"] = foreground
    affine = None if tiny is None else [list(struct.unpack("<4d", tiny[i:i + 32])) for i in range(0, 128, 32)]
    return result, affine


def compare_case(left, right, primary=False):
    lm, rm = [json.loads(z.read("manifest.json")) for z in (left, right)]
    allowed = {"array_index", "array_sha256"} | ({"artifacts", "workspace_sha256"} if primary else set())
    assert set(lm) == set(rm)
    exact({k: v for k, v in lm.items() if k not in allowed}, {k: v for k, v in rm.items() if k not in allowed})
    blobs = [z.read("arrays.npz") for z in (left, right)]
    for blob, manifest in zip(blobs, (lm, rm)):
        assert sha_bytes(blob) == manifest["array_sha256"]
    arrays = [ZipFile(io.BytesIO(blob)) for blob in blobs]
    indices = [logical_members(m["array_index"], a.namelist()) for m, a in zip((lm, rm), arrays)]
    assert set(indices[0]) == set(indices[1])
    for array, index in zip(arrays, indices):
        assert len(array.namelist()) == len(set(array.namelist()))
        assert set(array.namelist()) == set(index.values())
    records, native_affine = {}, None
    for logical in indices[0]:
        pair = []
        affines = []
        for array, index in zip(arrays, indices):
            with array.open(index[logical]) as handle:
                record, affine = npy_record(handle)
                pair.append(record)
                affines.append(affine)
        exact(pair[0], pair[1])
        exact(affines[0], affines[1])
        records[logical] = pair[0]
        if logical == "affine":
            native_affine = affines[0]
    for array in arrays:
        array.close()
    return {"case_id": lm["case_id"], "case_semantic_hash": lm["case_semantic_hash"],
            "frame": lm["frame"], "array_members": records,
            "nonstorage_manifest_sha256": sha_bytes(encoded({k: v for k, v in lm.items() if k not in allowed})),
            "array_storage_index_renumbered": lm["array_index"] != rm["array_index"],
            "logical_npy_payloads_exact": True}, (lm, rm, native_affine)


def record_check(record, binding):
    assert set(record) == {"actions", "config", "finalStateId", "initialStateId", "recordHash", "schema", "sourceBinding"}
    assert record["schema"] == "resectionlab.source-bound-rehearsal.v2"
    exact(record["sourceBinding"], binding)
    body = {k: v for k, v in record.items() if k != "recordHash"}
    assert semantic(body) == record["recordHash"]
    return {"actions": len(record["actions"]), "action_kinds": [a["kind"] for a in record["actions"]],
            "elapsed_seconds": sum(a["durationSeconds"] for a in record["actions"]),
            "record_hash": record["recordHash"], "initial_state_id": record["initialStateId"],
            "final_state_id": record["finalStateId"]}


def controls():
    exact({"a": 1, "b": [False]}, {"b": [False], "a": 1})
    for a, b in [({"a": 1}, {"a": True}), ([1, 2], [2, 1]), ({"a": 1}, {"a": 2}), ({"a": 1}, {"a": "1"})]:
        try:
            exact(a, b)
        except AssertionError:
            pass
        else:
            raise AssertionError("Mutation accepted")
    try:
        logical_members({"compartments": {"a": "compartments_0", "b": "compartments_0"}, "source_compartments": {}})
    except AssertionError:
        pass
    else:
        raise AssertionError("Alias accepted")
    empty = {"compartments": {}, "source_compartments": {}}
    assert set(logical_members(empty)) == {"mri", "affine"}
    assert set(logical_members(empty, ["brain_mask.npy"])) == {"mri", "affine", "brain_mask"}
    return 8


def main():
    started = datetime.now(timezone.utc).isoformat()
    generated = controls()
    before = {name: file_sha(ROOT / name) for name in PINS}
    exact(before, PINS)
    with ZipFile(ROOT / BASE) as left, ZipFile(ROOT / SAVED) as right:
        assert len(left.namelist()) == len(set(left.namelist()))
        assert set(left.namelist()) == set(right.namelist()) and len(right.namelist()) == len(set(right.namelist()))
        primary, (lm, rm, affine) = compare_case(left, right, True)
        assert lm["artifacts"] == {} and set(rm["artifacts"]) == {"patient_rehearsal", "workspace"}
        lw, rw = [json.loads(z.read("workspace.json")) for z in (left, right)]
        for archive, manifest in ((left, lm), (right, rm)):
            assert sha_bytes(archive.read("workspace.json")) == manifest["workspace_sha256"]
        assert len(lw["originalFiles"]) == len(rw["originalFiles"]) == 18
        exact(lw["originalFiles"], rw["originalFiles"])
        originals = []
        for item in lw["originalFiles"]:
            hashes = []
            for archive in (left, right):
                with archive.open(item["member"]) as handle:
                    hashes.append(stream_sha(handle))
            exact(hashes[0], hashes[1])
            assert hashes[0] == {"bytes": item["bytes"], "sha256": item["sha256"]}
            originals.append({"name": item["name"], "role": item["role"], **hashes[0]})
        assert len(lw["images"]) == len(rw["images"]) == 3
        displays = []
        for a, b in zip(lw["images"], rw["images"]):
            exact({k: v for k, v in a.items() if k not in {"bytes", "sha256"}}, {k: v for k, v in b.items() if k not in {"bytes", "sha256"}})
            nested = [archive.read(row["member"]) for archive, row in ((left, a), (right, b))]
            for blob, row in zip(nested, (a, b)):
                assert len(blob) == row["bytes"] and sha_bytes(blob) == row["sha256"]
            with ZipFile(io.BytesIO(nested[0])) as az, ZipFile(io.BytesIO(nested[1])) as bz:
                assert set(az.namelist()) == set(bz.namelist()) == {"manifest.json", "arrays.npz"}
                result, _ = compare_case(az, bz)
            displays.append(result)
        for workspace in (lw, rw):
            for image in workspace["images"]:
                image.pop("bytes"); image.pop("sha256")
        exact(lw, rw)
        artifact = rm["artifacts"]["patient_rehearsal"]
        assert set(artifact) == {"schema", "live", "comparison"}
        assert artifact["schema"] == "patient-rehearsal-workspace-v1"
        binding = artifact["live"]["sourceBinding"]
        assert binding["caseHash"] == rm["case_semantic_hash"] and binding["caseId"] == rm["case_id"]
        exact(binding["sourceRefs"], rm["source_refs"])
        exact(binding["affine"], affine); exact(binding["affineRasMm"], affine)
        assert binding["frame"] == rm["frame"] == "RAS+"
        arrays = primary["array_members"]
        exact(binding["shape"], arrays["mri"]["shape"])
        assert binding["imageHash"] == arrays["mri"]["array_digest"]
        assert binding["supportHash"] == arrays["brain_mask"]["array_digest"]
        assert binding["targetNames"] == ["monai_tumor_core"]
        assert binding["targetHash"] == arrays["compartments/monai_tumor_core"]["array_digest"]
        assert binding["frameHash"] == semantic({"shape": binding["shape"], "affine": affine, "frame": "RAS+", "physical_units": "mm"})
        permitted = {k: binding[k] for k in ("imageHash", "frameHash", "supportHash", "targetHash", "targetNames", "forbiddenHash", "authority")}
        assert semantic(permitted) == binding["planningInputHash"]
        assert binding["privateReferencePublished"] is False and binding["constraints"] == {}
        live = record_check(artifact["live"], binding)
        assert live["action_kinds"] == ["aspirate", "clear_field"]
        assert [a["toolId"] for a in artifact["live"]["actions"]] == ["aspirator", "field_manager"]
        comparison = artifact["comparison"]
        assert set(comparison) == {"snapshotId", "snapshot", "alternatives"}
        exact(comparison["snapshot"], artifact["live"])
        assert comparison["snapshotId"] == live["final_state_id"]
        assert [b["id"] for b in comparison["alternatives"]] == ["short_pass_field_management", "deeper_pass_reassess"]
        branches = []
        for branch in comparison["alternatives"]:
            row = record_check(branch["record"], binding)
            exact(branch["record"]["actions"][:2], artifact["live"]["actions"])
            exact(branch["record"]["config"], artifact["live"]["config"])
            assert row["initial_state_id"] == live["initial_state_id"] and row["action_kinds"][-1] == "STOP"
            branches.append({"id": branch["id"], **row})
        assert len({b["final_state_id"] for b in branches}) == 2
        workspace = rm["artifacts"]["workspace"]
        assert set(workspace) == {"case_hash", "category", "cursor", "overlayOpacity", "rehearsalView", "routeA", "routeB", "visibleLayers"}
        assert workspace["case_hash"] == rm["case_semantic_hash"] and workspace["category"] == "pareto"
        assert workspace["routeA"] == workspace["routeB"] == "" and workspace["overlayOpacity"] == .32
        assert len(workspace["cursor"]) == 3 and all(type(v) in (int, float) and math.isfinite(v) for v in workspace["cursor"])
        exact(workspace["visibleLayers"], {"FLAIR_abnormality": True, "brain": True, "enhancing_target": True,
             "monai_enhancing_tumor": False, "monai_tumor_core": False, "monai_whole_tumor": False, "nonenhancing_or_necrotic_core": True})
        exact(workspace["rehearsalView"], {"branchId": "deeper_pass_reassess", "episodeInitialStateId": live["initial_state_id"],
              "frameIndex": 21, "selectedTool": "field_manager", "visible": True})
    evidence = {}
    for stem in ("06-native-save", "07-fresh-rehearsal-reopen"):
        for ext in ("png", "ax.txt"):
            relative = f"outputs/poc-first-local-package-gui-v1/{stem}.{ext}"
            evidence[relative] = file_sha(ROOT / relative)
    ax = (ROOT / "outputs/poc-first-local-package-gui-v1/07-fresh-rehearsal-reopen.ax.txt").read_text()
    for expected in ("button Play rehearsal", "21  /  21", "STOP  ·  8  simulated s", "Field manager Held · inactive, Value: 1", "Deeper pass + reassess Viewing replay, Value: 1"):
        assert expected in ax
    exact({name: file_sha(ROOT / name) for name in PINS}, before)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert rss < 1024 ** 3
    report = {"schema": "packaged-native-ucsf-save-independent-v1", "status": "passed", "started_utc": started,
              "finished_utc": datetime.now(timezone.utc).isoformat(), "auditor_sha256": file_sha(__file__),
              "inputs": {name: {"sha256": digest, "bytes": (ROOT / name).stat().st_size} for name, digest in before.items()},
              "generated_controls_passed": generated, "peak_process_rss_bytes_macos": rss,
              "primary": primary, "display_sources": displays, "original_files": originals,
              "workspace_except_recomputed_nested_container_hashes_exact": True,
              "primary_metadata_role_source_refs_context_and_semantic_identity_exact": True,
              "source_binding": {k: binding[k] for k in ("imageHash", "frameHash", "supportHash", "targetHash", "planningInputHash", "privateReferencePublished")},
              "live": live, "branches": branches, "saved_view": workspace["rehearsalView"],
              "native_capture_hashes": evidence, "fresh_capture_has_selected_branch_frame_tool_and_paused_play_button": True,
              "archive_inputs_unchanged": True,
              "scope": "Independent logical NPY byte, native affine, original-byte, source metadata, saved-action seal/prefix and saved-view audit. No engine/model execution or replay by this auditor.",
              "limits": ["Archive compression and numbered compartment storage keys differ on resave; all logically named NPY bytes match exactly.",
                         "Saved action records contain no replay frame arrays; their integrity and common prefix were verified, not numerical trajectory recomputation.",
                         "Native reopen evidence is ROOT-captured accessibility text and screenshots; QA performed no GUI action.",
                         "No anatomy, physics, surgical or clinical acceptance is inferred."]}
    destination = Path(__file__).with_name("saved-ucsf-independent-qa.json")
    with destination.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False); handle.write("\n")
    destination.chmod(0o444); Path(__file__).chmod(0o444)
    print(json.dumps({"status": "passed", "report_sha256": file_sha(destination), "auditor_sha256": file_sha(__file__),
                      "logical_arrays": sum(len(v["array_members"]) for v in [primary] + displays), "original_files": len(originals), "peak_rss_bytes": rss}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-released-audit", action="store_true")
    args = parser.parse_args()
    if args.execute_released_audit:
        main()
    else:
        print(json.dumps({"generated_controls_passed": controls(), "actual_payloads_read": False}))
