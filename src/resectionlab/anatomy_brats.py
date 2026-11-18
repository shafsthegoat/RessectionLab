"""Four-channel MONAI BraTS estimates on an acquired, preprocessed shared grid."""
from importlib.metadata import distributions
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from .data_root import atomic_json, checked_root
from .patient_anatomy import digest, file_hash, reconstruction_lock

ROOT = Path(__file__).resolve().parents[2]
MODEL = "monai-brats-0.5.4"
CHANNELS = ("T1CE", "T1", "T2", "FLAIR")
MANIFEST = ROOT / "docs/poc-brats-model-manifest.json"
SOURCE = ROOT / "manifests/data_acquisition_ucsf.json"
WORKER = ROOT / "scripts/poc_brats_inference.py"
PARTS = (
    ("whole_tumor", "tumor-whole", "Whole abnormality", "#af79cf"),
    ("tumor_core", "tumor-core", "Tumor core", "#cf8c69"),
    ("enhancing_tumor", "enhancing-tumor", "Enhancing tumor", "#cf659b"),
)


def assets():
    from .anatomy_model_registry import _identities
    if sys.platform != "darwin":
        raise ValueError("The pinned MONAI BraTS runtime requires MPS")
    bundle = ROOT / "data/models/poc-monai-brats-0.5.4"
    if os.environ.get("RESSECTIONLAB_MODEL_ROOT"):
        bundle = Path(os.environ["RESSECTIONLAB_MODEL_ROOT"]).resolve() / bundle.name
        if not bundle.is_relative_to(checked_root()):
            raise ValueError("Configured model cache must be inside the external data root")
    elif not bundle.is_dir() and os.environ.get("RESSECTIONLAB_DATA_ROOT"):
        bundle = checked_root() / "cache/models" / bundle.name
    if bundle.resolve().is_relative_to("/Volumes"):
        checked_root(bundle.resolve().parents[2])
    manifest = json.loads(MANIFEST.read_text())
    files = {"brats_adapter": Path(__file__), "brats_worker": WORKER, "model_manifest": MANIFEST, "source_manifest": SOURCE}
    for name in ("models/model.pt", "configs/inference.json"):
        entry = next(row for row in manifest["acquisition"]["files"] if row["path"].endswith("/" + name))
        path = bundle / name
        if _identities([path])[str(path)] != entry["sha256"]:
            raise ValueError(f"Pinned MONAI BraTS file changed: {name}")
        files[name] = path
    python = ROOT / ".tools/poc-model-runtime/venv/bin/python"
    expected = manifest["runtime"]["package_versions"]
    package_dirs = list(python.parent.parent.glob("lib/python*/site-packages"))
    packages = {dist.metadata["Name"].lower(): dist for dist in distributions(path=package_dirs)}
    for name, version in expected.items():
        dist = packages.get(name)
        if not dist or dist.version != version:
            raise ValueError(f"MONAI BraTS runtime requires {name} {version}")
        record = next(item for item in dist.files if item.name == "RECORD" and item.parent.name.endswith(".dist-info"))
        files["runtime/" + name] = Path(dist.locate_file(record))
    return {**files, "python": python}, manifest["model"]


def inputs(c):
    """Only a verified source release or reviewed same-frame, brain-extracted volumes qualify."""
    from .anatomy_alignment import VERSION, input_key, matches_alignment_key
    from .anatomy_model_registry import _identities, source_volume
    reference = c.series(c.document["reference_series"])
    rows = []
    for modality in CHANNELS:
        matches = [row for row in c.document["series"] if row["modality"] == modality and row["timepoint"] == reference["timepoint"]]
        if len(matches) != 1 or not source_volume(matches[0]):
            raise ValueError(f"Exactly one acquired {modality} volume from this timepoint is required")
        rows.append(matches[0])
    if len({row["original"]["sha256"] for row in rows}) != len(CHANNELS):
        raise ValueError("Four distinct acquired sequences are required; a channel was duplicated")
    manifest = json.loads(SOURCE.read_text())
    model_source = json.loads(MANIFEST.read_text())["case_provenance"]
    if file_hash(SOURCE) != model_source["source_manifest_sha256"]:
        raise ValueError("Pinned preprocessed source declaration changed")
    expected = {r["modality"].replace("T1c", "T1CE"): r["sha256"] for r in manifest["files"] if r["role"] == "structural_mri"}
    pinned = all(row["original"]["sha256"] == expected[row["modality"]] for row in rows)
    if pinned and reference["id"] not in {row["id"] for row in rows}:
        raise ValueError("The pinned release uses a structural MRI reference from the declared four-channel set")
    records = []
    for row in rows:
        c.require_native_geometry(row)
        for item in (row["original"], row["data"]):
            path = c.path(item["path"])
            if _identities([path])[str(path)] != item["sha256"]:
                raise ValueError(f"Source content changed for {row['id']}")
        coverage = {**c.document.get("coverage_notes", {}), **row.get("coverage_notes", {}), **row.get("coverage", {})}
        limitations = coverage.get("limitations", [])
        stripping_only = bool(limitations) and all(item.get("kind") == "skull-stripped" for item in limitations)
        limited_field = coverage.get("status") == "limited" and not stripping_only
        if limited_field or any(coverage.get(k) or row.get(k) for k in ("cropped", "privacy_redacted", "defaced")):
            raise ValueError(f"{row['id']} has limited or altered coverage")
        stripped = coverage.get("skull_stripped") is True or any(
            item.get("kind") == "skull-stripped" and item.get("basis") == "source metadata" for item in limitations)
        if not pinned and not (stripped and coverage.get("whole_brain") is True):
            raise ValueError("Documented skull stripping and whole-brain coverage are required for every channel")
        shape = row["data"]["shape"]
        if shape != rows[0]["data"]["shape"] or not np.allclose(row["affine"], rows[0]["affine"], atol=1e-5, rtol=0):
            raise ValueError("Four registered volumes on one native grid are required; resampling is unavailable")
        if nib.aff2axcodes(np.asarray(row["affine"])) != ("L", "P", "S"):
            raise ValueError("The verified adapter requires LPS voxel order; reorientation is unavailable")
        if not np.allclose(nib.affines.voxel_sizes(np.asarray(row["affine"])), [1, 1, 1], atol=1e-4, rtol=0):
            raise ValueError("The four inputs require 1 mm isotropic spacing")
        registration = c.document["transforms"].get(row["id"])
        # The publisher's checksum-bound native frame is independent of the app's
        # automatic display overlay. No display transform enters the network or
        # gains a reviewed status. An explicit conflicting transform still blocks.
        if registration or not pinned:
            display_only = bool(pinned and registration and registration["status"] == "unreviewed"
                                and registration.get("algorithm_version") == VERSION
                                and registration.get("parent_series") == reference["id"]
                                and matches_alignment_key(c, row, reference, input_key(c, row, reference)))
            if registration and registration.get("quality", {}).get("status") == "warning":
                raise ValueError("Registration quality warning requires source review")
            matrix = c.registration_matrix(row["id"], require_review=not display_only, require_geometry=True)
            if not display_only and not np.allclose(matrix, np.eye(4), atol=1e-5, rtol=0):
                raise ValueError("Registration differs from the preprocessed image frame")
        records.append({"series_id": row["id"], "modality": row["modality"], "patient_id": row["patient_id"],
                        "timepoint": row["timepoint"], "original_sha256": row["original"]["sha256"],
                        "array_sha256": row["data"]["sha256"], "shape": shape, "affine": row["affine"],
                        "coverage": coverage, "registration": "source release common grid" if pinned else registration})
    return {"channels": records, "preprocessing": ("checksum-matched UCSF-PDGM preprocessed structural subset"
            if pinned else "documented whole-brain extraction; reviewed or source-provided common grid"),
            "display_registration_used_for_inference": False,
            "source_manifest_sha256": model_source["source_manifest_sha256"] if pinned else None,
            "registration_evidence_sha256": model_source["registration_evidence_sha256"] if pinned else None,
            "development_overlap": model_source["training_overlap"] if pinned else "unassessed"}


def run(c, stage, cancelled):
    from .anatomy_model_registry import model_environment
    from .anatomy_models import _model_layer_fields, run_bounded
    evidence = inputs(c)
    if evidence != stage["options"]["brats_inputs"]:
        raise ValueError("Multimodal inputs changed before inference")
    anchor = c.series(evidence["channels"][0]["series_id"])
    files, model = assets()
    environment = model_environment(MODEL)
    identity = digest({"inputs": evidence, "environment": environment})
    directory = c.root / "reconstructions" / ("brats-" + identity)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    support = None
    for channel in evidence["channels"]:
        row = c.series(channel["series_id"])
        path = c.path(row["original"]["path"])
        if file_hash(path) != channel["original_sha256"]:
            raise ValueError("Multimodal source checksum changed")
        data = nib.load(path).get_fdata(dtype=np.float32)
        if not np.array_equal(data, c.array(row["data"])) or not np.isfinite(data).all():
            raise ValueError("Native image and imported intensities differ or contain nonfinite values")
        mask = data != 0
        if not mask.any() or mask.all():
            raise ValueError("Brain-extracted image requires foreground and zero background")
        if support is None:
            support = mask
        elif np.count_nonzero(mask & support) < .98 * np.count_nonzero(mask | support):
            raise ValueError("Nonzero brain supports disagree across channels; check extraction and coverage")
        paths.append({"path": str(path), "sha256": channel["original_sha256"]})
    receipt = directory / "inference.json"
    with reconstruction_lock():
        if not receipt.exists():
            request = {"inputs": paths, "device": "mps", "weights": {"path": str(files["models/model.pt"])}, "output": str(directory)}
            atomic_json(directory / "request.json", request)
            stats = run_bounded([str(files["python"]), str(WORKER), "--worker", str(directory / "request.json")],
                                directory, cancelled=cancelled, memory_bytes=7 * 1024**3, timeout=1800)
            if any(file_hash(row["path"]) != row["sha256"] for row in paths):
                raise ValueError("Source changed during multimodal inference")
            records = []
            for filename, kind, label, color in PARTS:
                file = directory / f"monai_{filename}_mask.nii.gz"
                records.append({"filename": file.name, "kind": kind, "label": label, "color": color, "sha256": file_hash(file)})
            atomic_json(receipt, {"identity": identity, "model": MODEL, "inputs": evidence, "model_release": model,
                                  "environment": environment, "execution": stats, "outputs": records})
        result = json.loads(receipt.read_text())
    if result["identity"] != identity or result["inputs"] != evidence or result["environment"] != environment:
        raise ValueError("Cached multimodal result identity changed")
    outputs = []
    foreground = 0
    for part, (filename, kind, label, color) in zip(result["outputs"], PARTS, strict=True):
        path = directory / f"monai_{filename}_mask.nii.gz"
        if file_hash(path) != part["sha256"] or part["kind"] != kind:
            raise ValueError("Cached BraTS output changed")
        image = nib.load(path)
        values = np.asarray(image.dataobj)
        if (values.shape != tuple(anchor["data"]["shape"]) or not np.isin(values, [0, 1]).all()
                or not np.allclose(image.affine, anchor["affine"], atol=1e-5, rtol=0)):
            raise ValueError("BraTS output labels or physical grid changed")
        count = int(values.sum())
        key = f"brats-{identity[:16]}-{kind}"
        record = {"model": MODEL, "kind": kind, "source": anchor["original"]["sha256"], "inputs": evidence,
                  "model_release": model, "configuration": environment, "output_sha256": part["sha256"],
                  "evidence_use": "context_only", "clinical_validation": False, "foreground_voxels": count}
        if not any(layer["id"] == key for layer in c.document["layers"]):
            layer = c.add_layer(values, image.affine, layer_id=key, source_series=anchor["id"], kind=kind,
                                role="context", color=color, label=f"{label} · MONAI BraTS estimate")
            layer.update(_model_layer_fields(c, anchor["id"], image, record), preparation_input=stage["inputKey"],
                         availability="available" if count else "empty_output")
            layer["transform_dependencies"] = [row["series_id"] for row in evidence["channels"]]
        layer = c.layer(key)
        outputs.append({"id": key, "sha256": layer.get("original_data", layer["data"])["sha256"]})
        foreground += count
    return ("cached" if foreground else "empty_output", "Four-channel compartment estimates require review", outputs)
