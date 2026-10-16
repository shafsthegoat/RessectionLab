"""Local, hash-pinned anatomical proposal adapters with serialized bounded jobs.

FastSurfer is restricted to full-head noncontrast T1; SynthSeg accepts structural
MRI and retains its model's 1 mm output grid. Neither implies reviewed anatomy.
"""

from pathlib import Path
import json
import os
import signal
import subprocess
import time
import sys
import shutil
import uuid
import nibabel as nib
import numpy as np
import psutil
from .patient_anatomy import reconstruction_lock, digest, file_hash, affine

ROOT = Path(__file__).resolve().parents[2]


def run_bounded(
    command,
    directory,
    *,
    cancelled=lambda: False,
    memory_bytes=8 * 1024**3,
    timeout=1800,
    environment_overrides=None,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    peak = 0
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "TF_NUM_INTRAOP_THREADS": "2",
        "TF_NUM_INTEROP_THREADS": "2",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        **(environment_overrides or {}),
    }
    log_path = directory / "run.log"
    if log_path.exists():
        log_path.rename(directory / ("attempt-" + uuid.uuid4().hex + ".log"))
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                if cancelled():
                    raise InterruptedError("Reconstruction cancelled")
                if time.monotonic() - started > timeout:
                    raise TimeoutError("Reconstruction time limit exceeded")
                try:
                    parent = psutil.Process(process.pid)
                    rss = sum(
                        p.memory_info().rss
                        for p in [parent, *parent.children(recursive=True)]
                    )
                    peak = max(peak, rss)
                except psutil.NoSuchProcess:
                    continue
                if peak > memory_bytes:
                    raise MemoryError("Reconstruction exceeded bounded RSS budget")
                time.sleep(0.25)
            if process.returncode:
                raise RuntimeError(
                    f"Reconstruction failed ({process.returncode}); inspect {directory / 'run.log'}"
                )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    return {
        "elapsed_seconds": time.monotonic() - started,
        "peak_sampled_rss_bytes": peak,
        "memory_limit_bytes": memory_bytes,
        "command": command,
    }


def reconstruct(
    collection, series_id, model, *, cancelled=lambda: False, synthseg_version=2
):
    c = collection
    row = c.series(series_id)
    source = c.path(row["original"]["path"])
    if file_hash(source) != row["original"]["sha256"]:
        raise ValueError("Original source changed before reconstruction")
    if model in {"vesselfm", "vesselfm-t1ce-experimental", "synthstrip"}:
        from .poc_models import run

        modality = row["modality"]
        if model == "vesselfm-t1ce-experimental" and modality != "T1CE":
            raise ValueError("Experimental T1CE adapter requires T1CE")
        if model == "synthstrip" and modality == "T1CE":
            modality = "T1c"
        if modality == "TOF":
            modality = "MRA"
        record = run(
            source,
            "anatomy-" + digest(c.document["patient_id"])[:16],
            "synthstrip-main" if model == "synthstrip" else "vesselfm-base",
            modality,
            experimental_t1ce=model == "vesselfm-t1ce-experimental",
            blend="gaussian",
            cancelled=cancelled,
        )
        return {
            "model": model,
            "kind": "brain" if model == "synthstrip" else "vessels",
            "inference": record,
            "path": record["outputs"][0]["path"],
            "output_sha256": record["outputs"][0]["sha256"],
            "probability": record.get("probability"),
        }
    if model in {"meningioma", "glioma"}:
        return reconstruct_tumor(c, row, model, cancelled=cancelled)
    if model == "fastsurfer" and row["modality"] != "T1":
        raise ValueError(
            "FastSurfer requires an appropriate noncontrast full-head T1; T1CE is not silently substituted"
        )
    if model == "synthseg" and row["modality"] not in {
        "T1",
        "T1CE",
        "T2",
        "FLAIR",
        "MRI",
        "CT",
    }:
        raise ValueError("Unsupported SynthSeg input")
    if model == "synthseg":
        home = Path(
            os.environ.get("RESECTIONLAB_SYNTHSEG_HOME", ROOT / ".tools/SynthSeg")
        ).resolve()
        python = Path(
            os.environ.get(
                "RESECTIONLAB_SYNTHSEG_PYTHON",
                ROOT / ".tools/synthseg-runtime/bin/python",
            )
        ).absolute()
        names = (
            ["synthseg_1.0.h5"]
            if synthseg_version == 1
            else [
                "synthseg_robust_2.0.h5",
                "synthseg_parc_2.0.h5",
                "synthseg_qc_2.0.h5",
            ]
        )
        weights = [home / "models" / name for name in names]
        script = home / "scripts/commands/SynthSeg_predict.py"
    elif model == "fastsurfer":
        home = Path(
            os.environ.get("RESECTIONLAB_FASTSURFER_HOME", ROOT / ".tools/FastSurfer")
        ).resolve()
        python = Path(
            os.environ.get(
                "RESECTIONLAB_FASTSURFER_PYTHON",
                ROOT / ".tools/poc-model-runtime/venv/bin/python",
            )
        ).absolute()
        weights = sorted((home / "checkpoints").glob("*.pkl"))
        script = home / "FastSurferCNN/run_prediction.py"
    else:
        raise ValueError("Unknown reconstruction adapter")
    if (
        not python.exists()
        or not script.exists()
        or not weights
        or any(not p.exists() for p in weights)
    ):
        raise FileNotFoundError(
            f"{model} runtime or pinned weights missing. Install the official model locally; no substitute output is used."
        )
    identity = {
        "model": model,
        "version": synthseg_version if model == "synthseg" else None,
        "source": row["original"]["sha256"],
        "runner_sha256": file_hash(script),
        "adapter_sha256": file_hash(__file__),
        "weights": {p.name: file_hash(p) for p in weights},
    }
    directory = c.root / "reconstructions" / digest(identity)
    receipt = directory / "receipt.json"
    output = directory / "labels.nii.gz"
    with reconstruction_lock():
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if file_hash(output) != record["output_sha256"]:
                raise ValueError("Cached segmentation changed")
            return record
        directory.mkdir(parents=True, exist_ok=True)
        command = [str(python), str(script)]
        if model == "synthseg":
            command += [
                "--i",
                str(source),
                "--o",
                str(output),
                "--cpu",
                "--threads",
                "2",
                "--resample",
                str(directory / "inference-image.nii.gz"),
            ]
            if synthseg_version == 1:
                command += ["--v1"]
            else:
                command += ["--robust", "--parc", "--qc", str(directory / "qc.csv")]
            if row["modality"] == "CT":
                command += ["--ct"]
        else:
            command += [
                "--t1",
                str(source),
                "--sid",
                "patient",
                "--sd",
                str(directory),
                "--segfile",
                str(output),
                "--device",
                "mps",
                "--viewagg_device",
                "cpu",
                "--threads",
                "2",
            ]
        stats = run_bounded(command, directory, cancelled=cancelled)
        if cancelled():
            raise InterruptedError("Reconstruction cancelled before admission")
        im = nib.load(output)
        labels = np.asarray(im.dataobj)
        if (
            labels.ndim != 3
            or not np.isfinite(labels).all()
            or not np.array_equal(labels, np.rint(labels))
            or labels.max() > 65535
        ):
            raise ValueError("Invalid categorical reconstruction")
        record = {
            **identity,
            **stats,
            "path": str(output),
            "output_sha256": file_hash(output),
            "affine": affine(im.affine).tolist(),
            "review": "manual review required, especially tumor-distorted regions",
            "clinical_validation": None,
            "qc": (directory / "qc.csv").read_text()
            if (directory / "qc.csv").exists()
            else "automated QC model unavailable for this version; visual review required",
        }
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        return record


def import_parcellation(collection, series_id, record):
    """One shared categorical array, one descriptor per region; masks loaded lazily."""
    c = collection
    im = nib.load(record["path"])
    labels = np.asarray(im.dataobj).astype(np.uint16)
    ref = c.put_array(labels)
    names = {}
    if record["model"] == "synthseg":
        home = (
            Path(os.environ.get("RESECTIONLAB_SYNTHSEG_HOME", ROOT / ".tools/SynthSeg"))
            / "data/labels_classes_priors"
        )
        suffix = "" if record["version"] == 1 else "_2.0"
        for ids, n in [
            (
                "synthseg_segmentation_labels" + suffix,
                "synthseg_segmentation_names" + suffix,
            ),
            ("synthseg_parcellation_labels", "synthseg_parcellation_names"),
        ]:
            if (home / (ids + ".npy")).exists() and (home / (n + ".npy")).exists():
                names.update(
                    zip(
                        np.load(home / (ids + ".npy")).tolist(),
                        np.load(home / (n + ".npy")).tolist(),
                    )
                )
    existing = {x["id"] for x in c.document["layers"]}
    for value in np.unique(labels):
        if not value:
            continue
        key = f"{record['model']}-{digest({'source': series_id, 'output': record['output_sha256']})[:10]}-{int(value)}"
        if key in existing:
            continue
        c.document["layers"].append(
            {
                "id": key,
                "patient_id": c.document["patient_id"],
                "source_series": series_id,
                "label": str(
                    names.get(int(value), f"{record['model']} region {value}")
                ),
                "kind": "brain-region",
                "role": "context",
                "color": "#9dbccf",
                "affine": affine(im.affine).tolist(),
                "data": ref,
                "label_value": int(value),
                "review_status": "unreviewed; tumor-distorted regions require review",
                "provenance": record,
                "coverage": "model inference field; anatomical accuracy unassessed",
                "revision": 0,
            }
        )
    return ref


def import_proposal(c, series_id, record):
    if record["model"] in {"synthseg", "fastsurfer"}:
        return import_parcellation(c, series_id, record)
    if file_hash(record["path"]) != record["output_sha256"]:
        raise ValueError("Model output changed before admission")
    im = nib.load(record["path"])
    values = im.get_fdata(dtype=np.float32)
    probability = None
    if record.get("probability"):
        item = record["probability"]
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("Model probability changed")
        p = nib.load(item["path"])
        if p.shape != im.shape or not np.allclose(p.affine, im.affine):
            raise ValueError("Probability and mask geometry differ")
        probability = p.get_fdata(dtype=np.float32)
    key = (
        record["model"]
        + "-"
        + digest({"series": series_id, "output": record["output_sha256"]})[:12]
    )
    if any(r["id"] == key for r in c.document["layers"]):
        return key
    kind = record["kind"]
    c.add_layer(
        (values > 0).astype(np.uint8),
        im.affine,
        layer_id=key,
        source_series=series_id,
        kind=kind,
        role={"brain": "context", "vessels": "critical", "tumor": "target"}[kind],
        color={"brain": "#91c4b6", "vessels": "#ef5679", "tumor": "#c56df1"}[kind],
        label=kind.title() + " · " + record["model"],
        probability=probability,
        provenance=record,
    )
    return key


def reconstruct_tumor(c, row, model, *, cancelled=lambda: False):
    """Publisher-configured single-sequence Raidionics adapter; no missing-input substitution."""
    home = Path(
        os.environ.get(
            "RESECTIONLAB_RAIDIONICS_MODELS", ROOT / "data/models/raidionics"
        )
    )
    folder = home / {"meningioma": "MRI_Meningioma", "glioma": "MRI_Glioma"}[model]
    config = folder / "pipeline.json"
    if not config.exists():
        raise FileNotFoundError(f"Official {model} model assets missing: {folder}")
    pipeline = json.loads(config.read_text())
    steps = list(pipeline.values())
    inputs = steps[0].get("inputs", {}) if len(steps) == 1 else {}
    supported = {"T1-CE": "T1CE", "T1": "T1", "T2": "T2", "FLAIR": "FLAIR"}
    if (
        len(inputs) != 1
        or supported.get(next(iter(inputs.values())).get("sequence")) != row["modality"]
    ):
        raise ValueError(
            "This tumor model requires its publisher-declared input sequence(s); no missing or unregistered input is substituted"
        )
    files = [
        folder / n
        for n in [
            "model.onnx",
            "pipeline.json",
            "pre_processing.ini",
            "model_config.txt",
        ]
    ]
    if any(not p.exists() for p in files):
        raise FileNotFoundError("Incomplete tumor model configuration")
    packages = Path(
        os.environ.get(
            "RESECTIONLAB_RAIDIONICS_PACKAGES", ROOT / ".tools/raidionics-packages"
        )
    )
    runner = packages / "raidionicsseg/fit.py"
    if not runner.exists():
        raise FileNotFoundError("Local Raidionics runtime missing")
    identity = {
        "model": model,
        "source": row["original"]["sha256"],
        "files": {p.name: file_hash(p) for p in files},
        "runner_sha256": file_hash(runner),
        "adapter_sha256": file_hash(__file__),
    }
    directory = c.root / "reconstructions" / digest(identity)
    with reconstruction_lock():
        receipt = directory / "receipt.json"
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if (
                file_hash(record["path"]) != record["output_sha256"]
                or file_hash(record["probability"]["path"])
                != record["probability"]["sha256"]
            ):
                raise ValueError("Cached tumor reconstruction changed")
            return record
        input_dir = directory / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        source = c.path(row["original"]["path"])
        shutil.copyfile(source, input_dir / "input0.nii.gz")
        output = directory / "output"
        output.mkdir(exist_ok=True)
        ini = directory / "raidionics.ini"
        ini.write_text(
            f"[System]\ngpu_id=-1\ninputs_folder={input_dir}\noutput_folder={output}\nmodel_folder={folder}\n[Runtime]\nbatch_size=1\nfolds_ensembling=False\nreconstruction_method=probabilities\nreconstruction_order=resample_first\n"
        )
        script = directory / "run.py"
        script.write_text(
            "import sys, json, importlib.metadata\nimport onnxruntime as ort\noriginal=ort.InferenceSession\nclass BoundedSession(original):\n def __init__(self,*args,**kwargs):\n  options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1\n  kwargs['sess_options']=options\n  super().__init__(*args,**kwargs)\nort.InferenceSession=BoundedSession\nfrom raidionicsseg.fit import run_model\nrun_model(sys.argv[1])\nprint(json.dumps({p:importlib.metadata.version(p) for p in ['raidionicsseg','onnxruntime','numpy']}))\n"
        )
        stats = run_bounded(
            [sys.executable, str(script), str(ini)],
            directory,
            cancelled=cancelled,
            environment_overrides={"PYTHONPATH": str(packages)},
        )
        if cancelled():
            raise InterruptedError("Tumor reconstruction cancelled before admission")
        probability = output / "pred_Tumor.nii.gz"
        im = nib.load(probability)
        p = im.get_fdata(dtype=np.float32)
        if (
            p.shape != tuple(row["data"]["shape"])
            or not np.allclose(im.affine, row["affine"], atol=1e-4)
            or not np.isfinite(p).all()
            or p.min() < 0
            or p.max() > 1
        ):
            raise ValueError("Invalid tumor probability or native geometry")
        mask_path = output / "tumor_mask.nii.gz"
        nib.save(nib.Nifti1Image((p >= 0.5).astype(np.uint8), im.affine), mask_path)
        record = {
            **identity,
            **stats,
            "kind": "tumor",
            "path": str(mask_path),
            "output_sha256": file_hash(mask_path),
            "probability": {"path": str(probability), "sha256": file_hash(probability)},
            "foreground_voxels": int((p >= 0.5).sum()),
            "maximum_score": float(p.max()),
            "status": "proposal"
            if (p >= 0.5).any()
            else "empty tumor prediction; localization failed",
            "review": "unreviewed; probability is not calibrated confidence",
            "inference_geometry_config": (folder / "pre_processing.ini").read_text(),
        }
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        return record
