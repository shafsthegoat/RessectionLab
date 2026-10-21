"""Opt-in byte/metadata audit of the released native MR-selected saved copy."""
import importlib.util
import io
import json
import math
from pathlib import Path
import resource
from datetime import datetime, timezone
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
HELPER = Path(__file__).with_name("audit_saved_ucsf.py")
HELPER_SHA = "c58f95dc03c2526d0090b0f6ae123ff577e695745ed5616c00e9d2001cad4929"
import hashlib
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA
spec = importlib.util.spec_from_file_location("frozen_native_save_byte_oracle", HELPER)
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)
BASE = "outputs/poc-synthrad-bc-source-v1/1BC077/SynthRAD2023-1BC077.ressectionlab"
SAVED = "outputs/poc-first-local-package-gui-v1/saved/SynthRAD-packaged-MR-selected.ressectionlab"
PINS = {BASE: "171101168d1244d509fc64249cd9039a74aa9a35f35535dddc753307e80f905d",
        SAVED: "9a992b202cdcea35efd266e000ba7a2921d6f4416848ce6f683aed4ba3080557"}
SERIES = "sha256:2508a110e0226daff3463491a3ae44e5d79ac8b2c833d703f0eee1c1bc5c6cdd"


def main():
    before = {name: qa.file_sha(ROOT / name) for name in PINS}
    qa.exact(before, PINS)
    with ZipFile(ROOT / BASE) as left, ZipFile(ROOT / SAVED) as right:
        assert set(left.namelist()) == set(right.namelist())
        for archive in (left, right):
            assert len(archive.namelist()) == len(set(archive.namelist()))
        primary, (lm, rm, _) = qa.compare_case(left, right, True)
        assert lm["artifacts"] == {} and set(rm["artifacts"]) == {"workspace"}
        lw, rw = [json.loads(z.read("workspace.json")) for z in (left, right)]
        sessions = []
        for document in (lw, rw):
            logical = {k: v for k, v in document.items() if k not in ("sessionHash", "images")}
            logical["images"] = [row["source"] for row in document["images"]]
            assert qa.semantic(logical) == document["sessionHash"]
            sessions.append(document["sessionHash"])
        for archive, manifest in ((left, lm), (right, rm)):
            assert qa.sha_bytes(archive.read("workspace.json")) == manifest["workspace_sha256"]
        assert len(lw["originalFiles"]) == len(rw["originalFiles"]) == 5
        qa.exact(lw["originalFiles"], rw["originalFiles"])
        originals = []
        for item in lw["originalFiles"]:
            pair = []
            for archive in (left, right):
                with archive.open(item["member"]) as handle:
                    pair.append(qa.stream_sha(handle))
            qa.exact(pair[0], pair[1])
            assert pair[0] == {"bytes": item["bytes"], "sha256": item["sha256"]}
            originals.append({"name": item["name"], "role": item["role"], **pair[0]})
        assert len(lw["images"]) == len(rw["images"]) == 1
        a, b = lw["images"][0], rw["images"][0]
        qa.exact({k: v for k, v in a.items() if k not in {"bytes", "sha256"}}, {k: v for k, v in b.items() if k not in {"bytes", "sha256"}})
        nested = [z.read(row["member"]) for z, row in ((left, a), (right, b))]
        for blob, row in zip(nested, (a, b)):
            assert qa.sha_bytes(blob) == row["sha256"] and len(blob) == row["bytes"]
        with ZipFile(io.BytesIO(nested[0])) as az, ZipFile(io.BytesIO(nested[1])) as bz:
            assert set(az.namelist()) == set(bz.namelist()) == {"arrays.npz", "manifest.json"}
            display, (_, mr, mr_affine) = qa.compare_case(az, bz)
        assert set(display["array_members"]) == {"mri", "affine"}
        source = b["source"]
        assert source["seriesId"] == SERIES and source["modality"] == "T1"
        assert source["annotationKind"] == "none" and source["scope"] == "display-only-native-grid"
        assert source["sourceCaseHash"] == mr["case_semantic_hash"]
        frame_hash = qa.semantic({"shape": display["array_members"]["mri"]["shape"], "affine": mr_affine, "frame": mr["frame"], "physical_units": "mm"})
        assert source["sourceFrameHash"] == frame_hash
        for manifest in (rm, mr):
            assert manifest["metadata"]["workspace_registration"]["overlayPermitted"] is False
            assert manifest["metadata"]["workspace_registration"]["kind"] == "source-preprocessing-unreviewed"
            assert manifest["metadata"]["split"]["role"] == "TRAIN"
            assert manifest["metadata"]["split"]["training_admitted"] is False
            assert manifest["metadata"]["training_admitted"] is False
            assert manifest["metadata"]["is_synthetic"] is False
        assert lw["imagingState"]["selectedSeriesId"] is None
        assert rw["imagingState"]["selectedSeriesId"] == SERIES
        assert rw["imagingState"]["states"][SERIES]["visibleLayers"] == {}
        assert rw["imagingState"]["states"][SERIES].get("viewer", {}) == {}
        number_spelling_paths = []
        for key in ("primary", SERIES):
            original_cursor = lw["imagingState"]["states"][key]["cursor"]
            saved_cursor = rw["imagingState"]["states"][key]["cursor"]
            assert len(original_cursor) == len(saved_cursor) == 3
            assert type(original_cursor[0]) is float and type(saved_cursor[0]) is int
            assert original_cursor[0] == saved_cursor[0]
            saved_cursor[0] = original_cursor[0]
            qa.exact(original_cursor, saved_cursor)
            number_spelling_paths.append(f"imagingState.states.{key}.cursor[0]")
        rw["imagingState"]["selectedSeriesId"] = None
        for workspace in (lw, rw):
            workspace["images"][0].pop("bytes"); workspace["images"][0].pop("sha256")
            workspace.pop("sessionHash")
        qa.exact(lw, rw)
        workspace = rm["artifacts"]["workspace"]
        assert set(workspace) == {"case_hash", "category", "cursor", "overlayOpacity", "routeA", "routeB", "visibleLayers"}
        assert workspace["case_hash"] == rm["case_semantic_hash"] and workspace["category"] == "pareto"
        assert workspace["routeA"] == workspace["routeB"] == "" and workspace["overlayOpacity"] == .32
        assert len(workspace["cursor"]) == 3 and all(type(v) in (int, float) and math.isfinite(v) for v in workspace["cursor"])
        qa.exact(workspace["visibleLayers"], {"brain": True, "skull": True})
    captures = {}
    for stem in ("12-mri-view-saved", "13-fresh-mri-reopen"):
        for ext in ("png", "ax.txt"):
            relative = f"outputs/poc-first-local-package-gui-v1/{stem}.{ext}"
            captures[relative] = qa.file_sha(ROOT / relative)
        ax = (ROOT / f"outputs/poc-first-local-package-gui-v1/{stem}.ax.txt").read_text()
        assert "(selected) T1 · image · separate frame" in ax and "text Anatomy 0 / 0" in ax
    assert "Workspace reopened · native image views restored" in ax
    qa.exact({name: qa.file_sha(ROOT / name) for name in PINS}, before)
    assert qa.file_sha(HELPER) == HELPER_SHA
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert peak < 1024 ** 3
    result = {"schema": "packaged-native-mr-selected-save-independent-v1", "status": "passed",
              "created_utc": datetime.now(timezone.utc).isoformat(), "auditor_sha256": qa.file_sha(__file__),
              "byte_oracle_source_sha256": HELPER_SHA, "byte_oracle_generated_controls": qa.controls(),
              "inputs": {name: {"sha256": digest, "bytes": (ROOT / name).stat().st_size} for name, digest in before.items()},
              "primary": primary, "native_mri": display, "original_files": originals,
              "selected_series_id": SERIES, "native_mri_source": source,
              "native_mri_layers": 0, "overlay_permitted": False, "training_admitted": False,
              "workspace_changes": ["selectedSeriesId becomes existing native T1 source", "recomputed nested source container bytes/hash", "native workspace view record added", "session identity independently recomputed", "two cursor JSON number spellings normalize without changing numeric values"],
              "source_and_saved_session_hashes": sessions, "number_spelling_only_paths": number_spelling_paths,
              "all_other_source_case_role_model_registration_and_workspace_fields_exact": True,
              "native_capture_hashes": captures, "fresh_native_mri_selection_and_zero_layers_in_captures": True,
              "inputs_and_frozen_oracle_unchanged": True, "peak_process_rss_bytes_macos": peak,
              "scope": "Independent exact logical NPY and original-byte preservation, native frame and selected-view metadata audit; no image array decoding or engine/model execution.",
              "limits": ["QA inspected ROOT-captured accessibility text and bound screenshots without performing GUI actions.",
                         "No cross-frame fusion, training admission or anatomical/clinical acceptance is introduced."]}
    destination = Path(__file__).with_name("saved-synthrad-independent-qa.json")
    with destination.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False); handle.write("\n")
    destination.chmod(0o444); Path(__file__).chmod(0o444)
    print(json.dumps({"status": "passed", "report_sha256": qa.file_sha(destination), "auditor_sha256": qa.file_sha(__file__),
                      "logical_arrays": len(primary["array_members"]) + len(display["array_members"]), "original_files": 5, "peak_rss_bytes": peak}))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--execute-released-audit"]:
        main()
    else:
        raise SystemExit("Explicit --execute-released-audit required")
