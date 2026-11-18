"""Source-bound ingest and admission of the local SIAM head tissue estimates."""
from importlib.metadata import distributions
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from . import siam_worker as siam
from .data_root import atomic_json, checked_root
from .patient_anatomy import digest, file_hash, reconstruction_lock

ROOT = Path(__file__).resolve().parents[2]


def assets():
    home = Path(os.environ.get("RESECTIONLAB_SIAM_HOME", ROOT / "data/models/siam-v2/DS715_NODA")).resolve()
    if home.is_relative_to("/Volumes") and not home.is_relative_to(checked_root()):
        raise ValueError("External SIAM assets must be in the verified data root")
    python = Path(os.environ.get("RESECTIONLAB_SIAM_PYTHON", ROOT / ".tools/poc-model-runtime/venv/bin/python"))
    return python, home


def runtime_files(python):
    """Read installed package versions without starting the GPU runtime during planning."""
    package_dirs = list(python.parent.parent.glob("lib/python*/site-packages"))
    records = {}
    for dist in distributions(path=package_dirs):
        name = dist.metadata.get("Name", "").lower().replace("_", "-")
        if name not in siam.RUNTIME:
            continue
        if dist.version != siam.RUNTIME[name]:
            raise ValueError(f"SIAM needs {name} {siam.RUNTIME[name]}; installed {dist.version}")
        for entry in dist.files or ():
            if entry.name == "RECORD" and entry.parent.name.endswith(".dist-info"):
                records["runtime/" + name] = Path(dist.locate_file(entry))
                break
    missing = set(siam.RUNTIME) - {key.split("/", 1)[1] for key in records}
    if missing:
        raise FileNotFoundError(f"SIAM's pinned runtime is missing packages: {', '.join(sorted(missing))}")
    return records


def categorical(image, row):
    if (image.shape != tuple(row["data"]["shape"]) or image.header.get_xyzt_units()[0] != "mm"
            or not np.allclose(image.affine, row["affine"], rtol=0, atol=1e-4)):
        raise ValueError("SIAM output must match its native source grid and millimeter units")
    labels = np.asarray(image.dataobj)
    if labels.dtype != np.uint8 or labels.max() >= len(siam.LABELS):
        raise ValueError("SIAM output contains values outside its v2 ontology")
    return labels


def reconstruct(c, row, source, *, confirmed, cancelled):
    from .anatomy_model_registry import head_coverage, model_environment
    from .anatomy_models import run_bounded

    c.require_native_geometry(row)
    if row.get("synthetic"):
        raise ValueError("SIAM requires acquired source voxels")
    coverage = head_coverage(c, row, confirmed=confirmed)
    if coverage["status"] != "eligible":
        raise ValueError(coverage["reason"])
    source_image = nib.load(source)
    if not np.array_equal(source_image.get_fdata(dtype=np.float32), c.array(row["data"])):
        raise ValueError("SIAM original intensities differ from the imported scan")
    del source_image
    environment = model_environment("siam-v2")
    python, home = assets()
    identity = {
        "model": "siam-v2", "version": "DS715_NODA single fold 1", "source_series": row["id"],
        "source": row["original"]["sha256"], "source_array_sha256": row["data"]["sha256"],
        "source_modality": row["modality"], "native_affine": row["affine"], "configuration": environment,
    }
    directory = c.root / "reconstructions" / ("siam-" + digest(identity))
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "labels.nii.gz"
    receipt = directory / "receipt.json"
    with reconstruction_lock():
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if (any(record.get(key) != value for key, value in identity.items())
                    or file_hash(output) != record["output_sha256"]):
                raise ValueError("Cached SIAM source, model or output differs")
            categorical(nib.load(output), row)
            return {**record, "path": str(output)}
        stats = run_bounded([str(python), str(Path(siam.__file__).resolve()), str(source), str(output), str(home)],
                            directory, cancelled=cancelled, memory_bytes=8 * 1024**3,
                            minimum_available_bytes=768 * 1024**2, timeout=3600)
        if cancelled():
            raise InterruptedError("SIAM cancelled before admission")
        if file_hash(source) != identity["source"]:
            raise ValueError("Original source changed during SIAM inference")
        labels = categorical(nib.load(output), row)
        counts = np.bincount(labels.ravel(), minlength=len(siam.LABELS))
        record = {
            **identity, **stats, "path": str(output), "output_sha256": file_hash(output),
            "model_source": siam.RECORD, "publisher_code": siam.PUBLISHER, "license": "weights CC BY 4.0",
            "weight_author": "Romain Valabregue",
            "method": "image_derived_estimate", "evidence_use": "context_only", "clinical_validation": False,
            "training_data": "Synthetic contrasts from six high-quality anatomical templates",
            "training_admission": False, "benchmark_admission": False,
            "parameters": siam.PARAMETERS, "coverage_screen": coverage,
            "coverage_limitation": "Acquired MRI field. Thin dura, skull boundaries, vascular extent and tissue classes need review.",
            "foreground_voxels": int(counts[1:].sum()), "label_counts": counts.tolist(),
            "availability": "available" if counts[1:].any() else "empty_output",
        }
        atomic_json(receipt, record)
        return record


def import_tissues(c, series_id, record):
    from .anatomy_models import _model_layer_fields

    row = c.series(series_id)
    if row["data"]["sha256"] != record["source_array_sha256"]:
        raise ValueError("SIAM native source data differs")
    image = nib.load(record["path"])
    labels = categorical(image, row)
    codes = [int(code) for code in np.unique(labels) if code]
    ref = c.put_array(labels) if codes else None
    existing = {row["id"] for row in c.document["layers"]}
    base = "siam-" + digest({"series": series_id, "output": record["output_sha256"]})[:12]
    for code in codes:
        key = f"{base}-{code}"
        if key in existing:
            continue
        name, label, kind, color = siam.LABELS[code]
        fields = _model_layer_fields(c, series_id, image, record)
        fields["provenance"]["label_ontology"] = {
            "value": code, "name": name, "meaning": label, "dataset_sha256": siam.ASSETS["dataset.json"],
        }
        c.document["layers"].append({
            "id": key, "patient_id": c.document["patient_id"], "source_series": series_id,
            "label": f"{label} · SIAM MRI estimate", "kind": kind, "role": "context", "color": color,
            "affine": row["affine"], "data": ref, "label_value": code, "revision": 0,
            "availability": "available", **fields,
        })
    return ref
