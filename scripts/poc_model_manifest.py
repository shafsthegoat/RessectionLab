#!/usr/bin/env python3
"""Publish small reproducibility manifests; never include patient/weight payloads."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("outputs/poc-models/SynthRAD2023-1BA336/totalsegmentator-skull/32e4c06b003e5b904387/inference.json", {
        "source_url": "https://zenodo.org/records/7260705", "license": "CC BY-NC 4.0",
        "archive": "data/acquisition/synthrad-task1-v1/verified/Task1.zip", "archive_member": "Task1/brain/1BA336/ct.nii.gz",
        "extraction_record": "data/qc/synthrad-task1-first-pair-v1/ct-extraction.json",
        "acquired_not_synthetic": True, "limitations": ["Provider rigidly registered, defaced and cropped clinical CT", "3 mm fast model gives coarse bone margins", "Skull/brain masks are estimates without reviewed voxel labels"]}),
    ("outputs/poc-models/Lausanne-sub-000-ses-20110101/vesselfm-base/0cf5010a6c39c81d7f4b/inference.json", {
        "source_url": "https://openneuro.org/datasets/ds003949/versions/1.0.1", "license": "CC0",
        "identity": "sub-000/ses-20110101", "acquired_not_synthetic": True,
        "limitations": ["TOF regional slab: not complete head vascular coverage", "Cross-scan T1/TOF registration unresolved; only overlay on original TOF", "Generic vessel detector includes extracranial vessels and possible false positives; no artery names or clinical validation"]}),
    ("outputs/poc-models/UCSF-PDGM-0004/synthstrip-main/fe693f2aed5cb0bb90aa/inference.json", {
        "source_url": "https://huggingface.co/datasets/MedOtter/UCSF-PDGM/tree/e9372219cf1cd2fdd52260cd45f7514b4aa7638e",
        "identity": "UCSF-PDGM-0004", "license": "CC BY 4.0 collection; pinned public mirror provenance retained separately",
        "acquired_not_synthetic": True,
        "limitations": ["Source image is already skull stripped; model estimates whole-brain envelope from that image", "Brain extraction is not skull-bone segmentation or anatomical parcellation", "Source mirror identity and registration claims do not establish original scanner frame"]}),
]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def portable(value):
    if isinstance(value, str) and value.startswith(str(ROOT)+"/"):
        return str(Path(value).relative_to(ROOT))
    if isinstance(value, dict):
        return {k:portable(v) for k,v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    return value


def main():
    models = []
    for path, source in RUNS:
        record = json.loads((ROOT/path).read_text())
        if record["status"] != "succeeded":
            raise ValueError("Cannot publish failed inference as success")
        for artifact in record["weights"] + record["outputs"] + [record["input"]]:
            if digest(artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"Changed artifact: {artifact['path']}")
        entry = {"case": record["case"], "source": source, "inference_record": path,
                 "inference_record_sha256": digest(ROOT/path), "actual_inference": True,
                 "model_id": record["model_id"], "model": record["model"], "weights": record["weights"],
                 "input": record["input"], "outputs": record["outputs"], "cache_key": record["cache_key"],
                 "elapsed_seconds": record["elapsed_seconds"], "device": record["device"],
                 "peak_process_family_rss_bytes_sampled": record["peak_process_family_rss_bytes_sampled"],
                 "worker": record.get("worker"), "memory_caveat": record["memory_caveat"]}
        if "synthstrip_mps_allocator_samples" in record:
            entry["synthstrip_mps_allocator_samples"] = record["synthstrip_mps_allocator_samples"]
        models.append(portable(entry))
    manifest = {"schema": "resectionlab-poc-models-v1", "date": "2026-10-14", "host": "Apple Silicon Mac, 16 GB unified memory",
                "purpose": "Local noncommercial research anatomy workbench", "clinical_validation": "none",
                "models": models,
                "cache_verification": "outputs/poc-models/cache-reopen-check.json",
                "same_patient_gap": "No public case in this demonstration has glioma MRI, head CT and TOF/CTA together. These three named people are separate cases and must never be anatomically fused.",
                "not_completed": ["SynthSeg anatomical parcellation/ventricles not run", "TopBrain named arterial classes not run", "No patient-specific functional or surgical safety inference"],
                "alternatives_audited": [{"model": "TopBrain", "url": "https://zenodo.org/records/21959166", "decision": "Not downloaded: 4.8–11.9 GB containers, NVIDIA Docker-oriented runner; selected small vesselFM with actual native MPS inference instead"}],
                "source_files": [{"path": path, "sha256": digest(ROOT/path)} for path in ["src/resectionlab/poc_models.py", "scripts/poc_model_assets.py", "scripts/poc_model_setup.py", "scripts/poc_model_qc.py", "manifests/poc-model-runtime-requirements.txt"]]}
    destination = ROOT / "manifests/poc-models-v1.json"
    destination.write_text(json.dumps(manifest, indent=2)+"\n")
    print(destination)


if __name__ == "__main__":
    main()
