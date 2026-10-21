#!/usr/bin/env python3
"""Two fixed TRAIN SynthRAD CT/MR source pairs; source preparation only.

No download or registration is supported. Model execution requires a separate
exact source/code/budget release; the initial source admission does not grant it.
"""
from __future__ import annotations
import argparse
from contextlib import closing
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import time
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"outputs/poc-synthrad-bc-source-v1"
MANIFEST = ROOT/"manifests/poc-synthrad-bc-source-admission-v1.json"
SUBJECTS = ("1BB033", "1BC077")
SOURCE_RESULT_SHA = "4d2e85f0e2ba808f922f4d1f6f364e52e5f51d589e83746eb81793106b19e2ff"
ADMISSION_SHA = "75dc377966ab15da03ebb454a79059414876676cf39cd790019798924f828dd8"
SOURCE_HASHES = {
    "1BB033": {"ct": "4999e29411f5c76d986eb188f59a8148f8dbad7ef8bd30abef2e155a1681cafd", "mr": "68fe4d165a8b3063bfce58f7b75bf7b22ce52243a7dd3c0704b2a29984b7a228"},
    "1BC077": {"ct": "5a33487b11e9762c84f6b1ce8fb3e9f2c008e1ede78c57cd2a053c3cb14dead7", "mr": "f41eb00d101bab4b27665a6870794d6e4eb169d70bea0f9cfc28c02625dd72c0"},
}
ARCHIVE = ROOT/"data/acquisition/synthrad-task1-v1/verified/Task1.zip"
ARCHIVE_SHA = "43526037fcd23bdf4318bc193edd4f810e9b8806b6b8f57d41b8ef4e39c1f4cf"
WEIGHT = ROOT/"data/models/poc-totalsegmentator/weights/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
WEIGHT_SHA = "1e38e40356adc2706a662e365405a97f862d62a7da65f6bf81d025aee1b979ac"
MODEL_SUPPORT = {"dataset.json": "e804e08ad8912e3a6f541741c8c78799d75acbd3f4f3f18ff5f211bbd3da961a",
    "plans.json": "d1cb3c15f53dc36fdb618e0f9082573d1f50b74a9b5fa73094ac8680845839f1",
    "dataset_fingerprint.json": "f1336cd0df44e0eeecfa69143226c05b246c550e690b7b12929f65ca9843260f"}
MODEL_BUDGET = {"workers": 1, "subjects_sequential": list(SUBJECTS), "threads": 2,
    "wall_seconds_per_inference": 180, "process_family_rss_bytes": 5*1024**3,
    "minimum_system_available_bytes": 1024**3, "campaign_wall_seconds": 600,
    "automatic_retries": 0, "device": "mps", "model_parameters_changed": False,
    "allocator_cap": "Existing worker0.55 recommended-memory fraction, unchanged",
    "new_outputs_bytes": 512*1024**2, "free_disk_reserve_bytes": 50*1024**3}
PINS = {
    "manifests/synthrad-component-cohort-v1.json": "3bb4dd48e2e674bd921653d938ec7c7695c77afb8776c517800bbefb6a32ff30",
    "manifests/synthrad-task1-original-archive-v1.json": "f8b854a961ef0c799c96643445501c145630329fe61714e1dc89a263a3544c5b",
    "artifacts/synthrad-first-pair-qc-preparation-v1/central-directory.json": "d22c9e6a7dd28c2425b8e7c76b3bd323d6033d57a128e23ca6d3dfe7da4af3a7",
    "artifacts/synthrad-first-pair-qc-preparation-v1/directory-review.json": "96d29da46f2ba4ee856fe04f08cf751671f1b9cf80f4274c0aa7af101081680e",
    "artifacts/synthrad-first-pair-qc-preparation-v1/brain-pair-map.json": "2d3a10f7bdbeec1c6d6499d487e42d789eb8d46f92185389ef6ea76647feee20",
    "artifacts/synthrad-acquisition-preparation-v1/source-metadata/zenodo-record-7260705.json": "1761a3dc69342aa82bf77cdbaf961bda1698cb4bf06196aad04f7be136afe14b",
    "data/acquisition/synthrad-task1-v1/objects/synthrad-2023-task1-archive/completion.json": "5dbe792ca0f65ecedcb5036b9d54ebd6033f240b5f457dea376a5727e6645172",
    "scripts/synthrad_first_pair_qc_v1.py": "be2a90d009989d25cababc46945b6b40e1d5268b318b2c3894e1df9ff1cff4f3",
    "src/resectionlab/poc_models.py": "7446332d8e948624c055f3c19ad5d64c12018095d8e4b330535d62c47908c28b",
}
BOUNDS = {"new_disk_bytes": 512*1024**2, "free_disk_reserve_bytes": 50*1024**3,
    "max_native_payload_bytes_per_image": 512*1024**2, "max_voxels_per_image": 128*1024**2,
    "max_nifti_data_offset": 1024**2, "max_scalar_workspace_bytes": 8*1024**2,
    "scalar_chunk_voxels": 65536, "worker_rss_stop_bytes": 768*1024**2, "wall_seconds": 600,
    "max_extracted_member_bytes": 64*1024**2, "threads": 1, "automatic_retries": 0}


def require(condition, message):
    if not condition: raise ValueError(message)


def digest(path, check=lambda: None):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1024**2):
            check(); result.update(block)
    return result.hexdigest()


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def helper():
    path = ROOT/"scripts/synthrad_first_pair_qc_v1.py"
    require(digest(path) == PINS[str(path.relative_to(ROOT))], "Source decoder changed")
    spec = importlib.util.spec_from_file_location("synthrad_source_helpers", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def select_pairs(cohort, pairs):
    """Reproduce identity-based order, then bind the exact two released people."""
    namespace = cohort["split"]["namespace"]
    for center in "ABC":
        rows = sorted((r for r in cohort["members"] if r["anonymous_center"] == center), key=lambda r:(r["identity_sha256"], r["subject_id"]))
        require(len(rows) == 60, "Center denominator changed")
        for rank, row in enumerate(rows, 1):
            require(row["identity_sha256"] == hashlib.sha256((namespace+row["subject_id"]).encode()).hexdigest(), "Identity hash changed")
            expected_role = "TRAIN" if rank <= 42 else "SELECT" if rank <= 51 else "MEASUREMENT_EVAL"
            require(row["rank_within_center"] == rank and row["role"] == expected_role, "Frozen role/rank changed")
    train = sorted((r for r in cohort["members"] if r["role"] == "TRAIN"), key=lambda r:(r["rank_within_center"], r["anonymous_center"], r["subject_id"]))
    require([r["subject_id"] for r in train[:3]] == ["1BA336", *SUBJECTS], "Frozen TRAIN order changed")
    result = []
    for row in train[1:3]:
        matches = [p for p in pairs if p["subject_id"] == row["subject_id"]]
        require(len(matches) == 1, "Missing or duplicate source pair")
        pair = matches[0]
        for key in ("subject_id", "patient_group", "anonymous_center", "rank_within_center", "role", "identity_sha256"):
            require(pair[key] == row[key], "Source pair role/identity mismatch")
        names = [f"Task1/brain/{row['subject_id']}/{m}.nii.gz" for m in ("ct", "mr")]
        require(row["candidate_acquired_members"] == names and [m["name"] for m in pair["candidate_acquired_members"]] == names, "Acquired CT/MR member names changed")
        result.append(pair)
    return result


def admission():
    values = {}
    for name, expected in PINS.items():
        path = ROOT/name
        require(digest(path) == expected, "Pinned metadata or helper changed: "+name)
        if path.suffix == ".json": values[name] = json.loads(path.read_text())
    get = lambda suffix: next(v for k,v in values.items() if k.endswith(suffix))
    cohort, directory = get("synthrad-component-cohort-v1.json"), get("directory-review.json")
    pairs = select_pairs(cohort, get("brain-pair-map.json")["pairs"])
    source = get("synthrad-task1-original-archive-v1.json")
    require(source["rights"]["license_spdx"] == "CC-BY-NC-4.0" and source["dataset"]["release_record"] == 7260705, "Source rights/release changed")
    require(directory["archive_stat_after"] == directory["archive_stat_before"], "Prior archive continuity failed")
    require(get("completion.json")["sha256"] == ARCHIVE_SHA, "Archive completion binding changed")
    require(digest(WEIGHT) == WEIGHT_SHA, "Existing model checkpoint changed")
    h = helper()
    require(h.stat_record(ARCHIVE.resolve().stat()) == directory["archive_stat_after"], "Archive stat continuity changed")
    return {"schema": "poc-synthrad-two-train-source-admission-v1", "authorization_scope": "Exactly four named acquired CT/MR members; source verification/header/finiteness only",
        "subjects": list(SUBJECTS), "pairs": pairs, "preserve_existing_person": "1BA336", "cohort_sha256": PINS["manifests/synthrad-component-cohort-v1.json"],
        "selection": "Ranks1 B/C in frozen full_train_order; no performance or pixel selection", "bounds": BOUNDS,
        "pins": PINS, "archive": {"path": str(ARCHIVE.resolve()), "sha256": ARCHIVE_SHA, "stat": directory["archive_stat_after"], "central_start": directory["central_directory_start"]},
        "rights": source["rights"], "lineage": source["lineage"], "source_url": "https://zenodo.org/records/7260705",
        "training_admitted": False, "model_execution_authorized": False, "registration_execution_authorized": False,
        "model_assets": {"name": "TotalSegmentator", "version": "2.12.0 / Dataset297 v2.0.0-weights", "license": "Apache-2.0",
            "weights_path": str(WEIGHT.resolve()), "weights_sha256": WEIGHT_SHA, "weights_bytes": WEIGHT.stat().st_size,
            "input": "CT only; clinical HU interpretation remains unresolved until source qualification", "parameters": {"task": "total", "fast": True, "tta": False, "skull_label": 91, "brain_label": 90}},
        "denominator": [{"subject_id": p["subject_id"], "patient_group": p["patient_group"], "role": "TRAIN", "status": "not_attempted", "training_admitted": False} for p in pairs]}


class Budget:
    def __init__(self, output):
        self.output = output; self.started = time.monotonic(); self.peak_rss = 0
    def check(self):
        import psutil
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
        require(self.peak_rss <= BOUNDS["worker_rss_stop_bytes"], "Preparation RSS cap exceeded")
        require(time.monotonic()-self.started <= BOUNDS["wall_seconds"], "Preparation wall cap exceeded")
        require(shutil.disk_usage(self.output).free >= BOUNDS["free_disk_reserve_bytes"], "Free-disk reserve breached")
        require(sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file()) <= BOUNDS["new_disk_bytes"], "Preparation output cap exceeded")


def member_bytes(reader, row, offset, allowed, check):
    """Strict raw-deflate stream, with explicit per-call acquired member allowlist."""
    require(row["name"] in allowed, "Member body is outside exact TRAIN CT/MR allowlist")
    require(0 < row["zip_entry_bytes"] <= BOUNDS["max_extracted_member_bytes"], "Compressed-image bound exceeded")
    reader.seek(offset); decoder = zlib.decompressobj(-15)
    pending = b""; remaining = row["compressed_bytes"]; count = crc = 0
    while not decoder.eof:
        check()
        if not pending:
            require(remaining > 0, "Truncated ZIP deflate")
            expected = min(65536, remaining); pending = reader.read(expected)
            require(len(pending) == expected, "Truncated member bytes")
            remaining -= len(pending)
        block = decoder.decompress(pending, min(1024**2, row["zip_entry_bytes"]-count+1))
        pending = decoder.unconsumed_tail; count += len(block)
        require(count <= row["zip_entry_bytes"], "ZIP expanded length exceeded")
        crc = zlib.crc32(block, crc)
        if block: yield block
    require(not pending and not decoder.unused_data and remaining == 0, "Trailing ZIP compressed bytes")
    require(count == row["zip_entry_bytes"] and crc == int(row["crc32"],16), "ZIP CRC/length mismatch")


def qualify(path, proof, h, budget):
    import numpy as np
    with h.StrictGzip(path, proof) as stream: info = h.parse_header(stream, BOUNDS)
    grid = h.geometry(info["header"], info["shape"])
    info["proof"] = proof; count = zeros = 0; low = math.inf; high = -math.inf
    for values in h.scalar_chunks(path, info, budget):
        count += values.size; zeros += int(np.count_nonzero(values == 0))
        low = min(low, float(values.min())); high = max(high, float(values.max()))
    require(count == info["voxels"] and low < high, "Empty or constant source image")
    return {"path": str(path), "sha256": proof["sha256"], "bytes": proof["bytes"], "header": info["summary"], "grid": grid,
        "scalar": {"finite_voxels": count, "minimum": low, "maximum": high, "zero_fraction": zeros/count,
            "units": "source effective values; HU not independently calibrated"}, "gzip_crc_and_exact_payload_length": "passed",
        "privacy_review": "pending", "anatomical_coverage_review": "pending", "clinical_or_training_admission": False}


def execute_sources():
    frozen = json.loads(MANIFEST.read_text()); current = admission()
    require(frozen == current, "Source admission differs from current pinned inputs")
    OUTPUT.mkdir(parents=True, exist_ok=False); os.chmod(OUTPUT, 0o700)
    h = helper(); budget = Budget(OUTPUT); rows = json.loads((ROOT/"artifacts/synthrad-first-pair-qc-preparation-v1/central-directory.json").read_text())["entries"]
    result = {"schema": "poc-synthrad-bc-source-result-v1", "admission_sha256": digest(MANIFEST), "denominator": frozen["denominator"], "pairs": [], "training_admitted": False, "model_runs": 0, "registration_runs": 0}
    try:
        budget.check(); archive = Path(frozen["archive"]["path"])
        require(digest(archive, budget.check) == ARCHIVE_SHA, "Full archive SHA256 mismatch")
        require(h.stat_record(archive.stat()) == frozen["archive"]["stat"], "Archive changed during full verification")
        write_new(OUTPUT/"archive-verified.json", {"sha256": ARCHIVE_SHA, "bytes": archive.stat().st_size, "fresh_full_hash": True, "source_admission_sha256": digest(MANIFEST)})
        with closing(h.ArchiveReader(archive, frozen["archive"]["stat"], frozen["archive"]["central_start"], budget.check)) as reader:
            with zipfile.ZipFile(reader) as zipped:
                require([h.zip_row(i) for i in zipped.infolist()] == rows and not zipped.comment, "Central directory changed")
                ends = sorted([r["header_offset"] for r in rows]+[zipped.start_dir])
            for pair, denominator in zip(frozen["pairs"], result["denominator"], strict=True):
                denominator["status"] = "started"
                folder = OUTPUT/pair["subject_id"]; folder.mkdir(mode=0o700)
                report = {"subject_id": pair["subject_id"], "patient_group": pair["patient_group"], "role": "TRAIN", "training_admitted": False, "images": {}}
                result["pairs"].append(report)
                try:
                    allowed = {r["name"] for r in pair["candidate_acquired_members"]}
                    for row in pair["candidate_acquired_members"]:
                        require(row in rows, "Member is absent from pinned source directory")
                        local = h.local_header(reader, row, next(v for v in ends if v > row["header_offset"]))
                        name = Path(row["name"]).name; target = folder/name; partial = folder/(name+".partial")
                        hasher = hashlib.sha256(); count = 0
                        with partial.open("xb") as output:
                            os.chmod(partial, 0o600)
                            for block in member_bytes(reader, row, local["data_offset"], allowed, budget.check):
                                require(sum(p.stat().st_size for p in OUTPUT.rglob("*") if p.is_file())+len(block) <= BOUNDS["new_disk_bytes"], "Output reserve exceeded")
                                output.write(block); hasher.update(block); count += len(block)
                        partial.rename(target)
                        proof = {"bytes": count, "sha256": hasher.hexdigest(), "staged_stat": h.stat_record(target.stat()), "member": row["name"]}
                        write_new(folder/(name+".extraction.json"), proof)
                        report["images"][name[:2]] = qualify(target, proof, h, budget)
                    import numpy as np
                    ct, mr = (report["images"][m]["grid"] for m in ("ct", "mr"))
                    report["coded_grids_equal"] = ct["shape"] == mr["shape"] and bool(np.allclose(ct["affine_ras_mm"], mr["affine_ras_mm"], rtol=0, atol=1e-5))
                    report["registration"] = "Publisher declares rigid preprocessing; independent anatomical correspondence pending; no transform chain available"
                    report["coverage"] = "Defaced and cropped source acquisition; calvarium/skull-base/vertex coverage unreviewed"
                    denominator["status"] = "source_checks_passed_anatomy_unreviewed"
                except Exception as error:
                    denominator["status"] = "failed"; report["failure"] = type(error).__name__+": "+str(error)
                write_new(folder/"source.json", report)
            result["archive_read_bytes_after_hash"] = reader.read_bytes
    except Exception as error:
        result["failure"] = type(error).__name__+": "+str(error)
    finally:
        result["runtime"] = {"elapsed_seconds": time.monotonic()-budget.started, "peak_rss_bytes_sampled": budget.peak_rss,
                             "output_bytes": sum(p.stat().st_size for p in OUTPUT.rglob("*") if p.is_file())}
        write_new(OUTPUT/"source-result.json", result)
    return result


def bound_source(source_report):
    """Bind caller metadata independently before opening any CT/MR payload."""
    sid = source_report.get("subject_id")
    require(sid in SUBJECTS and source_report.get("role") == "TRAIN"
            and source_report.get("training_admitted") is False, "Source outside fixed TRAIN denominator")
    path = OUTPUT/"source-result.json"
    require(digest(path) == SOURCE_RESULT_SHA, "Frozen source-result identity changed")
    result = json.loads(path.read_text())
    expected = next(p for p in result["pairs"] if p["subject_id"] == sid)
    require(source_report == expected, "Source report differs from frozen pair receipt")
    require(expected["patient_group"] == "SynthRAD2023:"+sid and set(expected["images"]) == {"ct", "mr"}, "Source pair identity changed")
    for modality in ("ct", "mr"):
        image = expected["images"][modality]
        require(image["path"] == str(OUTPUT/sid/(modality+".nii.gz"))
                and image["sha256"] == SOURCE_HASHES[sid][modality]
                and image["gzip_crc_and_exact_payload_length"] == "passed", "Source path/hash is not the independently pinned input")
    # All metadata/path/role checks precede either payload hash read.
    for modality in ("ct", "mr"):
        image = expected["images"][modality]
        require(digest(image["path"]) == SOURCE_HASHES[sid][modality], "Acquired source bytes changed")
    return expected


def model_request(source_report, output):
    """Prepare one CT-only request; no model imports, subprocess or execution."""
    source_report = bound_source(source_report)
    ct = source_report["images"]["ct"]
    require(ct["gzip_crc_and_exact_payload_length"] == "passed" and Path(ct["path"]).name == "ct.nii.gz", "CT source qualification missing")
    require(digest(ct["path"]) == ct["sha256"] and digest(WEIGHT) == WEIGHT_SHA, "Model source/weights changed")
    require(digest(ROOT/"src/resectionlab/poc_models.py") == PINS["src/resectionlab/poc_models.py"], "Model worker changed")
    for name, expected in MODEL_SUPPORT.items():
        require(digest(WEIGHT.parent.parent/name) == expected, "Model plan/dataset metadata changed")
    return {"case": "SynthRAD2023-"+source_report["subject_id"], "model_id": "totalsegmentator-skull", "modality": "CT", "image": ct["path"],
        "device": "mps", "output": str(output), "weights": [{"path": str(WEIGHT.resolve()), "sha256": WEIGHT_SHA, "bytes": WEIGHT.stat().st_size}],
        "parameters": {"task": "total", "fast": True, "tta": False, "skull_label": 91, "brain_label": 90}}


def package_pair(source_report, inference_path, output):
    """Later-stage adapter: canonical model persistence plus a separate MR source."""
    from resectionlab.imaging import load_nifti_case
    from resectionlab.patient_workspace import prepare_model_workspace
    from resectionlab.workspace_bundle import OriginalFile, load_workspace, save_workspace, view_state
    from resectionlab.workspace_imaging import DisplaySource, source_registration
    from resectionlab.poc_models import MODEL_CONFIG
    source_report = bound_source(source_report)
    report = json.loads(Path(inference_path).read_text()); sid = source_report["subject_id"]
    require(report["case"] == "SynthRAD2023-"+sid and report["model_id"] == "totalsegmentator-skull"
            and report["input"]["sha256"] == SOURCE_HASHES[sid]["ct"]
            and report["input"]["path"] == source_report["images"]["ct"]["path"], "Model input differs from acquired CT")
    expected_request = model_request(source_report, Path(inference_path).parent)
    expected_identity = {"image_sha256": SOURCE_HASHES[sid]["ct"], "weights": [WEIGHT_SHA],
        "model_id": "totalsegmentator-skull", "parameters": expected_request["parameters"], "device": "mps",
        "runner_sha256": PINS["src/resectionlab/poc_models.py"]}
    expected_key = hashlib.sha256(json.dumps(expected_identity, sort_keys=True).encode()).hexdigest()[:20]
    require(report["status"] == "succeeded" and report["weights"] == expected_request["weights"]
            and report["identity"] == expected_identity and report["cache_key"] == expected_key
            and report["device"] == "mps" and report["model"] == MODEL_CONFIG["totalsegmentator-skull"], "Model receipt differs from the existing pinned worker/configuration")
    require({row["name"] for row in report["outputs"]} == {"brain_mask.nii.gz", "skull_mask.nii.gz"}
            and len(report["outputs"]) == 2, "Model receipt has unexpected anatomy outputs")
    for row in report["outputs"]:
        require(Path(row["path"]).resolve() == (Path(inference_path).parent/row["name"]).resolve(), "Mask path is outside the model attempt")
        require(digest(row["path"]) == row["sha256"], "Saved model output bytes changed")
    metadata = {"case_id": report["case"], "patient_group": source_report["patient_group"],
        "split": {"role": "TRAIN", "patient_group": source_report["patient_group"], "training_admitted": False},
        "cohort_role": "TRAIN", "training_admitted": False, "source_name": "SynthRAD2023 acquired CT/MR pair "+sid,
        "source_url": "https://zenodo.org/records/7260705", "source_license": "CC-BY-NC-4.0", "source_image_sha256": report["input"]["sha256"],
        "cohort_sha256": PINS["manifests/synthrad-component-cohort-v1.json"], "acquisition_datetime": None,
        "workspace_registration": {"kind": "source-preprocessing-unreviewed", "overlayPermitted": False},
        "missing_anatomy": ["tumor", "vessels", "ventricles"], "source_checks": source_report,
        "limitation": "Defaced/cropped source; estimated skull/brain, anatomy and HU calibration unreviewed; no vascular/tumor or synthetic-CT claim"}
    output = Path(output); require(not output.exists(), "Do not overwrite an existing pair bundle")
    base = output.with_name(output.stem+"-ct-only.ressectionlab")
    require(not base.exists(), "Do not overwrite an existing CT estimate bundle")
    prepare_model_workspace(inference_path, base, source_metadata=metadata)
    primary, artifacts, saved = load_workspace(base)
    mr = source_report["images"]["mr"]; require(digest(mr["path"]) == mr["sha256"], "Acquired MR changed")
    auxiliary = load_nifti_case(mr["path"], case_id=report["case"], source_url=metadata["source_url"], license=metadata["source_license"],
        metadata={**metadata, "selected_modality": "T1", "source_image_sha256": mr["sha256"], "annotation_use": "none"})
    sources = [DisplaySource(auxiliary, "T1", "none")]
    originals = [*saved["originals"], OriginalFile.from_path(mr["path"], role="imaging", expected_sha256=mr["sha256"])]
    state = view_state(primary, sources); state["states"]["primary"] = saved["imagingState"]["states"]["primary"]
    identity = save_workspace(primary, output, sources=sources, imaging=state, episode=None, selection=None, artifacts=artifacts, originals=originals)
    reopened, _, restored = load_workspace(output)
    require(reopened.semantic_hash == primary.semantic_hash and restored["sessionHash"] == identity
        and restored["originals"] == originals and restored["sources"][0].case.semantic_hash == auxiliary.semantic_hash, "Paired bundle changed on reopen")
    require(not source_registration(primary, auxiliary)[1]["overlayPermitted"], "Unreviewed pair gained fusion permission")
    return {"bundle": str(output), "sha256": digest(output), "source_reopened_identical": True, "fusion_permitted": False}


def release_template():
    require(digest(MANIFEST) == ADMISSION_SHA, "Source-only admission changed")
    require(digest(OUTPUT/"source-result.json") == SOURCE_RESULT_SHA, "Source result changed")
    sources = json.loads((OUTPUT/"source-result.json").read_text())["pairs"]
    requests = [model_request(source, ROOT/"outputs/poc-models"/("SynthRAD2023-"+source["subject_id"])) for source in sources]
    return {"schema": "poc-synthrad-two-train-model-release-v1", "execution_authorized": False,
        "source_result_sha256": SOURCE_RESULT_SHA, "source_admission_sha256": ADMISSION_SHA,
        "source_hashes": SOURCE_HASHES, "script_sha256": digest(__file__),
        "worker_sha256": PINS["src/resectionlab/poc_models.py"], "model_support": MODEL_SUPPORT,
        "requests": requests, "budget": MODEL_BUDGET, "training_admitted": False,
        "registration_review": "pending; matching source grids do not enable fusion"}


def model_resource_failure(*, rss, available, elapsed, free_disk, output_bytes):
    checks = [(rss > MODEL_BUDGET["process_family_rss_bytes"], "process_family_RSS_limit"),
        (available < MODEL_BUDGET["minimum_system_available_bytes"], "global_available_memory_floor"),
        (elapsed > MODEL_BUDGET["campaign_wall_seconds"], "campaign_wall_limit"),
        (free_disk < MODEL_BUDGET["free_disk_reserve_bytes"], "free_disk_reserve"),
        (output_bytes > MODEL_BUDGET["new_outputs_bytes"], "output_bytes_limit")]
    return next((reason for exceeded, reason in checks if exceeded), None)


def run_models(release_path):
    """Sequential existing canonical worker with extra process/global guards."""
    expected = release_template(); expected["execution_authorized"] = True
    require(json.loads(Path(release_path).read_text()) == expected, "Exact model/source/code/budget release missing")
    import psutil
    from resectionlab.poc_models import run as canonical_run
    sources = json.loads((OUTPUT/"source-result.json").read_text())["pairs"]
    result_path = OUTPUT/"model-campaign.json"
    require(not result_path.exists(), "Model campaign already has a retained result")
    campaign_started = time.monotonic()
    campaign = {"release_sha256": digest(release_path), "source_result_sha256": SOURCE_RESULT_SHA,
                "subjects": [], "training_admitted": False, "registration_runs": 0}
    write_new(OUTPUT/"model-denominator-initial.json", {"release_sha256": digest(release_path),
        "source_result_sha256": SOURCE_RESULT_SHA, "subjects": [{"subject_id": sid, "role": "TRAIN", "training_admitted": False,
        "status": "not_attempted"} for sid in SUBJECTS]})
    try:
        for source in sources:
            request = model_request(source, OUTPUT/source["subject_id"])
            row = {"subject_id": source["subject_id"], "status": "started", "peak_process_family_rss_bytes_sampled": 0,
                   "minimum_system_available_bytes_sampled": psutil.virtual_memory().available, "guard_failure": None}
            campaign["subjects"].append(row)
            def cancelled():
                process = psutil.Process()
                family = [process, *process.children(recursive=True)]
                measured = 0
                for item in family:
                    try: measured += item.memory_info().rss
                    except psutil.NoSuchProcess: pass
                row["peak_process_family_rss_bytes_sampled"] = max(row["peak_process_family_rss_bytes_sampled"], measured)
                row["minimum_system_available_bytes_sampled"] = min(row["minimum_system_available_bytes_sampled"], psutil.virtual_memory().available)
                roots = [OUTPUT, *[ROOT/"outputs/poc-models"/("SynthRAD2023-"+sid) for sid in SUBJECTS]]
                written = sum(p.stat().st_size for root in roots for p in root.rglob("*") if p.is_file())
                row["guard_failure"] = model_resource_failure(rss=measured,
                    available=row["minimum_system_available_bytes_sampled"], elapsed=time.monotonic()-campaign_started,
                    free_disk=shutil.disk_usage(OUTPUT).free, output_bytes=written)
                return row["guard_failure"] is not None
            try:
                require(not cancelled(), "Resource guard failed before model launch")
                inference_record = canonical_run(Path(request["image"]), request["case"], request["model_id"], "CT", "mps",
                                                MODEL_BUDGET["wall_seconds_per_inference"], cancelled=cancelled)
                model_folder = Path(inference_record["outputs"][0]["path"]).parent
                row["inference_json"] = str(model_folder/"inference.json")
                row["inference_sha256"] = digest(model_folder/"inference.json")
                row["bundle"] = package_pair(source, model_folder/"inference.json", OUTPUT/source["subject_id"]/(request["case"]+".ressectionlab"))
                require(not cancelled(), "Resource guard failed after bundle packaging")
                row["status"] = "model_saved_bundle_unreviewed"
            except Exception as error:
                row["status"] = "failed"; row["failure"] = type(error).__name__+": "+str(error)
                if row["guard_failure"]: break
            finally:
                write_new(OUTPUT/source["subject_id"]/"model-stage-result.json", row)
    finally:
        attempted = {row["subject_id"] for row in campaign["subjects"]}
        campaign["subjects"].extend({"subject_id": sid, "status": "not_attempted"} for sid in SUBJECTS if sid not in attempted)
        campaign["elapsed_seconds"] = time.monotonic()-campaign_started
        campaign["memory_caveat"] = "Sampled process RSS and Metal driver/tensor allocations overlap; do not sum; transient peaks may be missed"
        write_new(result_path, campaign)
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "sources", "model-template", "models"])
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        value = admission(); write_new(MANIFEST, value); print(json.dumps({"manifest": str(MANIFEST), "sha256": digest(MANIFEST), "subjects": value["subjects"]}))
    elif args.action == "sources":
        result = execute_sources(); print(json.dumps(result, indent=2))
        if result.get("failure") or any(r["status"] != "source_checks_passed_anatomy_unreviewed" for r in result["denominator"]): raise SystemExit(1)
    elif args.action == "model-template":
        path = OUTPUT/"model-release-template.json"; write_new(path, release_template()); print(path)
    else:
        require(args.release is not None, "Explicit frozen model release file required")
        print(json.dumps(run_models(args.release), indent=2))


if __name__ == "__main__": main()
