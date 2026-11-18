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
import configparser
import hashlib
import nibabel as nib
import numpy as np
import psutil
from .patient_anatomy import reconstruction_lock, digest, file_hash, affine
from .data_root import atomic_json
from .anatomy_model_registry import (
    ANGIOGRAPHY, POC_ADAPTERS, STRUCTURAL_MRI, TUMOR_ADAPTERS, ct_intensities,
    head_coverage, matching_layer, source_volume, supersynth_home, synthseg_assets,
    SYNTHSEG_MAX_PADDED_CELLS, synthseg_requires_roi, synthseg_working_shape, tumor_assets,
)

ROOT = Path(__file__).resolve().parents[2]


def run_bounded(
    command,
    directory,
    *,
    cancelled=lambda: False,
    memory_bytes=8 * 1024**3,
    timeout=1800,
    environment_overrides=None,
    minimum_available_bytes=0,
):
    if cancelled():
        raise InterruptedError("Reconstruction cancelled")
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
                if minimum_available_bytes and psutil.virtual_memory().available < minimum_available_bytes:
                    raise MemoryError("Reconstruction stopped to preserve available system memory")
                if log_path.stat().st_size > 8 * 1024**2:
                    raise RuntimeError(f"Reconstruction exceeded log size limit; inspect {log_path}")
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
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
    return {
        "elapsed_seconds": time.monotonic() - started,
        "peak_sampled_rss_bytes": peak,
        "memory_limit_bytes": memory_bytes,
        "command": command,
    }


def reconstruct(
    collection, series_id, model, *, cancelled=lambda: False, synthseg_version=2,
    synthseg_crop=None, brain_layer_id=None, full_head_confirmed=False,
    include_mri_skull=True, cranial_layer_id=None, synthseg_resource_bound=False,
):
    c = collection
    row = c.series(series_id)
    source = c.path(row["original"]["path"])
    if file_hash(source) != row["original"]["sha256"]:
        raise ValueError("Original source changed before reconstruction")
    if cancelled():
        raise InterruptedError("Reconstruction cancelled before inference")
    if model == "supersynth":
        return reconstruct_head(c, row, source, brain_layer_id=brain_layer_id,
                                confirmed=full_head_confirmed, include_skull=include_mri_skull,
                                cancelled=cancelled)
    if model == "dural-envelope":
        from .anatomy_dural import reconstruct_dura
        return reconstruct_dura(c, row, cranial_layer_id=cranial_layer_id,
                                brain_layer_id=brain_layer_id, cancelled=cancelled)
    if model in POC_ADAPTERS:
        return reconstruct_poc(c, row, source, model, cancelled)
    if model in TUMOR_ADAPTERS:
        return reconstruct_tumor(c, row, model, cancelled=cancelled, brain_layer_id=brain_layer_id)
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
        if synthseg_crop is not None and (type(synthseg_crop) is not int or not 128 <= synthseg_crop <= 256 or synthseg_crop % 32):
            raise ValueError('SynthSeg crop must be an explicit 128–256 multiple of 32')
        python, script, weights = synthseg_assets(synthseg_version)
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
        "source_series": row["id"],
        "runner_sha256": file_hash(script),
        "adapter_sha256": file_hash(__file__),
        "weights": {p.name: file_hash(p) for p in weights},
    }
    if model == "synthseg" and synthseg_resource_bound:
        identity["resource_bound"] = {"padded_1mm_cell_limit": SYNTHSEG_MAX_PADDED_CELLS,
                                      "rss_limit_bytes": int(6.5 * 1024**3)}
        if synthseg_requires_roi(row):
            brain = c.layer(brain_layer_id) if brain_layer_id else matching_layer(c, row, {"brain"}, native=True)
            identity["inference_source_roi"] = _synthseg_brain_bounds(c, row, brain)
            identity["coverage_limitation"] = (
                "Model field covers the source-grid brain estimate plus available physical margin. "
                "Outside this field remains uncovered. Brain-envelope accuracy needs review."
            )
    if model == 'synthseg' and synthseg_crop is not None:
        identity['inference_crop_voxels'] = synthseg_crop
        identity['coverage_limitation'] = 'Center-cropped 1 mm inference field; structures outside this field are unknown, not negative.'
    from .anatomy_model_registry import model_environment
    if model == "synthseg":
        identity["configuration"] = model_environment(model, synthseg_version=synthseg_version)
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
            model_source = source
            if "inference_source_roi" in identity:
                model_source = _write_synthseg_roi(c, row, source, directory, identity["inference_source_roi"])
            command += [
                "--i",
                str(model_source),
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
            if synthseg_crop is not None:
                command += ['--crop', str(synthseg_crop)]
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
        limits = {"memory_bytes": int(6.5 * 1024**3),
                  "minimum_available_bytes": int(.875 * 1024**3)} if synthseg_resource_bound else {}
        stats = run_bounded(command, directory, cancelled=cancelled, **limits)
        if cancelled():
            raise InterruptedError("Reconstruction cancelled before admission")
        im = nib.load(output)
        labels = np.asarray(im.dataobj)
        if (
            labels.ndim != 3
            or not np.isfinite(labels).all()
            or not np.array_equal(labels, np.rint(labels))
            or labels.min() < 0
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
            "clinical_validation": False,
            "availability": "available" if labels.any() else "empty_output",
            "method": "image_derived_estimate",
            "label_ontology": "SynthSeg / FreeSurfer codes" if model == "synthseg" else "FastSurfer / FreeSurfer codes",
            "qc": (directory / "qc.csv").read_text()
            if (directory / "qc.csv").exists()
            else "automated QC model unavailable for this version; visual review required",
        }
        atomic_json(receipt, record)
        return record


def _synthseg_brain_bounds(c, row, brain):
    if (brain is None or brain["kind"] != "brain" or brain["source_series"] != row["id"]
            or brain["data"]["shape"] != row["data"]["shape"]
            or not np.allclose(brain["affine"], row["affine"], atol=1e-5)):
        raise ValueError("Regional estimate requires a same-source native brain envelope for bounded inference")
    mask = c.mask(brain)
    if not mask.any():
        raise ValueError("An empty brain envelope cannot define a regional inference field")
    spacing = nib.affines.voxel_sizes(affine(row["affine"]))
    margin_mm = 8
    margin = np.ceil(margin_mm / spacing).astype(int)
    bounds = []
    for axis in range(3):
        coordinates = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
        bounds.append([max(0, int(coordinates[0] - margin[axis])),
                       min(mask.shape[axis], int(coordinates[-1] + 1 + margin[axis]))])
    padded = synthseg_working_shape([hi - lo for lo, hi in bounds], spacing)
    if int(np.prod(padded)) > SYNTHSEG_MAX_PADDED_CELLS:
        raise ValueError("The complete brain-envelope field exceeds the bounded SynthSeg configuration")
    return {"bounds_native_exclusive": bounds, "margin_mm": margin_mm,
            "padded_working_shape": padded.tolist(), "brain_layer": brain["id"],
            "brain_mask_sha256": brain["data"]["sha256"], "brain_revision": brain.get("revision", 0),
            "estimated_brain_voxels_in_field": int(np.count_nonzero(mask))}


def _write_synthseg_roi(c, row, source, directory, roi):
    bounds = roi["bounds_native_exclusive"]
    slices = tuple(slice(lo, hi) for lo, hi in bounds)
    translation = np.eye(4)
    translation[:3, 3] = [lo for lo, _ in bounds]
    a = affine(row["affine"]) @ translation
    header = nib.load(source).header.copy()
    im = nib.Nifti1Image(np.asarray(c.array(row["data"])[slices]), a, header)
    im.set_data_dtype(np.float32)
    im.set_sform(a, code=1)
    model_source = directory / "source-brain-field.nii.gz"
    nib.save(im, model_source)
    return model_source


def import_parcellation(collection, series_id, record):
    """One shared categorical array, one descriptor per region; masks loaded lazily."""
    c = collection
    im = nib.load(record["path"])
    labels = np.asarray(im.dataobj)
    if labels.ndim != 3 or not np.isfinite(labels).all() or not np.array_equal(labels, np.rint(labels)) or labels.min() < 0 or labels.max() > 65535:
        raise ValueError("Invalid categorical reconstruction")
    labels = labels.astype(np.uint16)
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
                    names.get(int(value), _STRUCTURE_NAMES.get(int(value), f"{record['model']} region {value}"))
                ),
                "kind": _brain_kind(int(value)),
                "role": "context",
                "color": "#9dbccf",
                "affine": affine(im.affine).tolist(),
                "data": ref,
                "label_value": int(value),
                "revision": 0,
                **_model_layer_fields(c, series_id, im, record),
            }
        )
    return ref


def _model_layer_fields(c, series_id, im, record):
    source = c.series(series_id)
    transform = c.document["transforms"].get(series_id, {})
    provenance = {**record, "kind": record.get("method", "image_derived_estimate")}
    provenance.pop("extra_outputs", None)
    crop = record.get("inference_crop_voxels")
    bounds = None
    if crop is not None:
        # The publisher center-crops in RAS voxel orientation and restores the
        # output to pre-crop axis order. Flip the odd-voxel margin accordingly.
        shape = np.asarray(im.shape)
        low = np.maximum((shape - crop) // 2, 0)
        high = np.minimum(low + crop, shape)
        flipped = nib.orientations.io_orientation(im.affine)[:, 1] < 0
        bounds = [np.where(flipped, shape - high, low).tolist(),
                  (np.where(flipped, shape - low, high) - 1).tolist()]
    return {
        "units_status": im.header.get_xyzt_units()[0],
        "provenance": provenance,
        "coverage": record.get("coverage_limitation", "Source image and model field; anatomical detection sensitivity unassessed"),
        "coverage_fields": {
            "source": {"shape": source["data"]["shape"], "affine": source["affine"]},
            "output": {"shape": list(im.shape), "affine": affine(im.affine).tolist()},
            "inference_crop_voxels": crop, "inference_bounds_voxel_inclusive": bounds,
            "supported_axial_indices": record.get("supported_axial_indices"),
            "outside": "uncovered", "zero_meaning": "unlabelled; anatomical absence unassessed",
        },
        "acquisition_grid_spacing_mm": nib.affines.voxel_sizes(affine(source["affine"])).tolist(),
        "acquisition_resolution": source.get("acquisition_resolution", "unreported"),
        "output_spacing_mm": nib.affines.voxel_sizes(im.affine).tolist(),
        "transform_dependencies": [series_id],
        "source_transform_at_admission": transform.get("status"),
        "review_status": ("unreviewed; tumor-distorted regions require review"
                          if record["model"] in {"synthseg", "fastsurfer", "supersynth"} else "unreviewed"),
    }


def import_proposal(c, series_id, record):
    if record.get("source_series", series_id) != series_id:
        raise ValueError("Model output belongs to a different source series")
    if record.get("source", c.series(series_id)["original"]["sha256"]) != c.series(series_id)["original"]["sha256"]:
        raise ValueError("Model output source identity differs")
    if file_hash(record["path"]) != record["output_sha256"]:
        raise ValueError("Model output changed before admission")
    if record["model"] in {"synthseg", "fastsurfer"}:
        return import_parcellation(c, series_id, record)
    if record["model"] == "supersynth":
        return import_head(c, series_id, record)
    im = nib.load(record["path"])
    values = im.get_fdata(dtype=np.float32)
    if values.ndim != 3 or not np.isin(values, [0, 1]).all():
        raise ValueError("Model output requires a finite binary mask")
    probability = None
    if record.get("probability"):
        item = record["probability"]
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("Model score changed")
        p = nib.load(item["path"])
        if p.shape != im.shape or not np.allclose(p.affine, im.affine):
            raise ValueError("Model score and mask geometry differ")
        probability = p.get_fdata(dtype=np.float32)
        if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
            raise ValueError("Model score must be finite and in [0,1]")
    key = record["model"] + "-" + digest({"series": series_id, "output": record["output_sha256"]})[:12]
    kind = record["kind"]
    if kind == "brain" and record["model"] == "totalsegmentator-skull":
        key += "-brain"
    if any(r["id"] == key for r in c.document["layers"]):
        for extra in record.get("extra_outputs", []):
            import_proposal(c, series_id, extra)
        return key
    available = bool(values.any())
    labels = {
        "brain": ("Brain envelope", "#91c4b6"),
        "vessels": ("Unclassified vessel candidates", "#ef5679"),
        "tumor": ("Lesion candidate", "#c56df1"),
        "skull": ("Skull estimate from acquired CT", "#d8e7f8"),
        "dura": ("Estimated dural envelope — geometric context", "#eccca7"),
    }
    label, color = labels[kind]
    if kind == "vessels" and record["model"] == "vesselfm-t1ce-experimental":
        label = "Inferred vessel candidates · experimental T1 postcontrast"
    if kind != "dura":
        label += " · " + record["model"]
    if not available:
        label += " · empty output"
    role = "context"
    if kind == "vessels" and available:
        from .anatomy_evidence import vascular_evidence_supported
        layer = {"id": key, "kind": kind, "source_series": series_id, "provenance": record}
        if vascular_evidence_supported(c, layer):
            role = "critical"
    row = c.add_layer(values.astype(np.uint8), im.affine, layer_id=key, source_series=series_id,
                      kind=kind, role=role, color=color, label=label, probability=probability)
    row.update(_model_layer_fields(c, series_id, im, record),
               availability="available" if available else "empty_output")
    for extra in record.get("extra_outputs", []):
        import_proposal(c, series_id, extra)
    return key


def reconstruct_tumor(c, row, model, *, cancelled=lambda: False, brain_layer_id=None):
    """Publisher-configured single-sequence Raidionics adapter; no missing-input substitution."""
    folder, packages = tumor_assets(model)
    config = folder / "pipeline.json"
    if not config.exists():
        raise FileNotFoundError(f"Official {model} model assets missing: {folder}. Install pinned assets with python -m resectionlab.reconstruction_assets; use the explicit GBM T1CE or LGGlioma FLAIR adapter.")
    pipeline = json.loads(config.read_text())
    steps = list(pipeline.values())
    # Published glioma archives describe a brain stage followed by the tumor
    # stage. The local runner executes the selected tumor model only, with an
    # explicitly selected, matching brain mask for its declared preprocessing.
    tumor_steps = [step for step in steps if step.get("model") == folder.name and step.get("target") == ["Tumor"]]
    selected = tumor_steps[0] if len(tumor_steps) == 1 else steps[0] if len(steps) == 1 else {}
    inputs = selected.get("inputs", {})
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
    preprocessing = configparser.ConfigParser()
    preprocessing.read(folder / "pre_processing.ini")
    brain = None
    if preprocessing.get("PreProcessing", "background_cropping", fallback="") in {"brain_mask", "brain_clip"}:
        if not brain_layer_id:
            raise ValueError("This glioma model requires a brain mask: reconstruct or import brain on the selected scan, then choose its layer explicitly")
        brain_row = c.layer(brain_layer_id)
        if brain_row["kind"] != "brain" or brain_row["source_series"] != row["id"]:
            raise ValueError("Select a brain layer belonging to the same source scan; cross-scan registration is not substituted")
        brain = c.mask(brain_row)
        if brain.shape != tuple(row["data"]["shape"]) or not np.allclose(brain_row["affine"], row["affine"], atol=1e-5) or not brain.any():
            raise ValueError("Glioma brain mask must be nonempty and match the selected native scan geometry")
    runner = packages / "raidionicsseg/fit.py"
    if not runner.exists():
        raise FileNotFoundError("Local Raidionics runtime missing")
    identity = {
        "model": model,
        "source": row["original"]["sha256"],
        "files": {p.name: file_hash(p) for p in files},
        "runner_sha256": file_hash(runner),
        "runtime_source_sha256": digest({str(p.relative_to(packages)): file_hash(p)
            for p in sorted((packages / 'raidionicsseg').rglob('*.py'))}),
        "adapter_sha256": file_hash(__file__),
    }
    if brain is not None:
        identity["brain_mask"] = {"layer_id": brain_layer_id, "data": brain_row["data"],
                                 "affine": brain_row["affine"], "revision": brain_row.get("revision", 0),
                                 "selected_mask_sha256": hashlib.sha256(np.ascontiguousarray(brain, dtype=np.uint8)).hexdigest(),
                                 "provenance": brain_row.get("provenance")}
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
        # raidionicsseg 1.5.x discovers weights one directory below model_folder;
        # official 1.2 archives store them at the root. Preserve publisher assets
        # and stage a single, hash-checked fold for the installed runner.
        staged = directory / 'model'
        (staged / 'fold-0').mkdir(parents=True, exist_ok=True)
        for name in ('pre_processing.ini', 'model_config.txt', 'pipeline.json'):
            shutil.copyfile(folder / name, staged / name)
        staged_weights = staged / 'fold-0/model.onnx'
        if not staged_weights.exists():
            staged_weights.symlink_to((folder / 'model.onnx').resolve())
        if file_hash(staged_weights) != identity['files']['model.onnx']:
            raise ValueError('Staged model differs from installed weight identity')
        output = directory / "output"
        output.mkdir(exist_ok=True)
        ini = directory / "raidionics.ini"
        brain_config = ""
        if brain is not None:
            brain_path = directory / "brain-mask.nii.gz"
            brain_image = nib.Nifti1Image(brain.astype(np.uint8), row["affine"])
            brain_image.header.set_xyzt_units('mm')
            nib.save(brain_image, brain_path)
            brain_config = f"[Neuro]\nbrain_segmentation_filename={brain_path}\n"
        ini.write_text(
            f"[System]\ngpu_id=-1\ninputs_folder={input_dir}\noutput_folder={output}\nmodel_folder={staged}\n[Runtime]\nbatch_size=1\nfolds_ensembling=False\nreconstruction_method=probabilities\nreconstruction_order=resample_first\n"
            + brain_config
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
        mask_image = nib.Nifti1Image((p >= 0.5).astype(np.uint8), im.affine)
        mask_image.header.set_xyzt_units('mm')
        nib.save(mask_image, mask_path)
        record = {
            **identity,
            **stats,
            "kind": "tumor",
            "source_series": row["id"],
            "version": "Raidionics publisher archive 1.2.0",
            "method": "image_derived_estimate",
            "label_ontology": {"value": 1, "meaning": {
                "meningioma": "Meningioma model lesion candidate on T1 postcontrast",
                "glioma-gbm": "Glioblastoma model lesion candidate on T1 postcontrast",
                "glioma-lgg": "Lower-grade glioma model lesion candidate on FLAIR",
            }.get(model, "Publisher-defined glioma model lesion candidate")},
            "availability": "available" if (p >= .5).any() else "empty_output",
            "parameters": {"threshold": .5, "foreground_definition": "publisher tumor channel"},
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
            "license": "BSD-2-Clause; retain publisher notice",
            "model_source": "https://github.com/raidionics/Raidionics-models/releases/tag/1.2.0",
            "scope": "Selected publisher tumor model with explicit native brain mask where required; unreviewed research estimate, no clinical diagnosis or validation",
        }
        atomic_json(receipt, record)
        return record


def reconstruct_poc(c, row, source, model, cancelled):
    from .poc_models import run

    modality = row["modality"]
    if model == "totalsegmentator-skull" and not ct_intensities(c, row):
        raise ValueError("TotalSegmentator skull requires an acquired CT or CTA source in Hounsfield units")
    if model == "vesselfm-t1ce-experimental" and modality != "T1CE":
        raise ValueError("Experimental T1CE adapter requires T1CE")
    if model == "vesselfm" and (modality not in ANGIOGRAPHY or not source_volume(row)):
        raise ValueError("Vascular proposals require volumetric CTA, TOF or MRA source data")
    if model == "synthstrip" and modality not in STRUCTURAL_MRI:
        raise ValueError("The installed SynthStrip adapter requires one structural MRI")
    runtime_modality = {"TOF": "MRA", "T1CE": "T1c"}.get(modality, modality) if model != "vesselfm-t1ce-experimental" else modality
    inference = run(
        source, "anatomy-" + digest(c.document["patient_id"])[:16],
        POC_ADAPTERS[model], runtime_modality, experimental_t1ce=model == "vesselfm-t1ce-experimental",
        blend="gaussian", cancelled=cancelled,
    )
    if cancelled():
        raise InterruptedError("Reconstruction cancelled before admission")
    outputs = {item["name"]: item for item in inference["outputs"]}
    if model == "totalsegmentator-skull":
        kind, name = "skull", "skull_mask.nii.gz"
        ontology = {"value": 1, "source_label": 91, "meaning": "Estimated skull from acquired CT-family source"}
    elif model == "synthstrip":
        kind, name = "brain", "brain_mask.nii.gz"
        ontology = {"value": 1, "meaning": "Extracted brain mask"}
    else:
        kind, name = "vessels", "vessels_mask.nii.gz"
        ontology = {"value": 1, "meaning": "Unclassified vessel candidate"}
    output = outputs.get(name)
    if output is None:
        raise ValueError(f"{model} did not produce its {kind} mask")
    if model == "totalsegmentator-skull":
        output = restrict_ct_skull(c, row, output)
    model_info = inference.get("model", {})
    record = {
        "model": model, "kind": kind, "source_series": row["id"], "source": row["original"]["sha256"],
        "source_modality": modality, "inference": inference, "version": model_info.get("version"),
        "parameters": model_info.get("parameters", {}), "path": output["path"], "output_sha256": output["sha256"],
        "probability": inference.get("probability"), "clinical_validation": False, "label_ontology": ontology,
        "method": "image_derived_estimate",
        "availability": "empty_output" if output.get("foreground_voxels") == 0 else "available",
        "evidence_use": "positive_labels_only" if model == "vesselfm" else "context_only",
    }
    if model == "totalsegmentator-skull":
        record["inference_spacing_mm"] = [3, 3, 3]
        record["parameters"] = {**record["parameters"], "positive_ct_bone_gate_hu": 250}
        record["coverage_limitation"] = "Acquired CT field with positive dense-bone gating. Incomplete detection, cortical thickness and drilling support unassessed."
        if "brain_mask.nii.gz" in outputs:
            brain = outputs["brain_mask.nii.gz"]
            record["extra_outputs"] = [{
                **record, "kind": "brain", "path": brain["path"], "output_sha256": brain["sha256"],
                "availability": "empty_output" if brain.get("foreground_voxels") == 0 else "available",
                "label_ontology": {"value": 1, "source_label": 90, "meaning": "Estimated brain envelope"},
            }]
    elif model == "vesselfm":
        record["coverage_limitation"] = "Acquired angiography field is partial; vascular detection and unlabelled space remain unassessed"
    elif model == "vesselfm-t1ce-experimental":
        record["coverage_limitation"] = "Experimental T1 postcontrast inference. Angiographic vascular evidence is unavailable."
    return record


def reconstruct_head(c, row, source, *, brain_layer_id, confirmed, include_skull, cancelled):
    from .head_segmentation import ASSETS, MODEL_VERSION, PARAMETERS, segment_head

    brain = c.layer(brain_layer_id) if brain_layer_id else matching_layer(c, row, {"brain"}, native=True)
    coverage = head_coverage(c, row, confirmed=confirmed, brain=brain)
    if coverage["status"] != "eligible":
        raise ValueError(coverage["reason"])
    if brain is None:
        raise ValueError("Prepare a same-source brain envelope to assess exterior coverage")
    model_root = supersynth_home()
    identity = {"source": row["original"]["sha256"], "pinned_assets": ASSETS, "parameters": PARAMETERS}
    folder = c.root / "reconstructions" / ("supersynth-" + digest(identity))
    record = segment_head(source, folder, model_root, retry=True, cancelled=cancelled)
    return {
        "model": "supersynth", "kind": "head", "version": MODEL_VERSION, "source_series": row["id"],
        "source": row["original"]["sha256"], "clinical_validation": False,
        "parameters": PARAMETERS, "method": "image_derived_estimate",
        "path": str(Path(record["output"]) / "labels_1mm.nii.gz"), "output_sha256": record["output_sha256"],
        "inference": record, "availability": "available" if any(record["labels_present"]) else "empty_output",
        "coverage_screen": coverage, "include_mri_skull": include_skull, "evidence_use": "context_only",
        "label_ontology": "Pinned FreeSurfer v8.2.0 color LUT / August 2025 SuperSynth segmentation class order",
        "coverage_limitation": "Only acquired MRI field is supported. Segmentation, tiled adaptation and extracranial accuracy require review.",
        "brain_coverage_dependency": {"id": brain["id"], "data": brain["data"], "revision": brain.get("revision", 0)},
    }


_STRUCTURE_NAMES = {
    7: "Left cerebellar white matter", 8: "Left cerebellar cortex", 16: "Brainstem",
    46: "Right cerebellar white matter", 47: "Right cerebellar cortex",
}
_HEAD_KIND = {902: "artery", 914: "vein", 911: "scalp", 915: "skull", 916: "skull"}


def _brain_kind(code):
    return {
        7: "cerebellar-white-matter", 8: "cerebellar-cortex", 16: "brainstem",
        46: "cerebellar-white-matter", 47: "cerebellar-cortex",
    }.get(code, "brain-region")


def import_head(c, series_id, record):
    from .head_segmentation import ASSETS

    im = nib.load(record["path"])
    labels = np.asanyarray(im.dataobj)
    if labels.ndim != 3 or labels.dtype != np.uint16:
        raise ValueError("Pinned SuperSynth output requires its 3D uint16 class map")
    lut = supersynth_home() / "FreeSurferColorLUT.txt"
    if file_hash(lut) != ASSETS[lut.name]:
        raise ValueError("Pinned FreeSurfer labels changed")
    ontology = {}
    for line in lut.read_text().splitlines():
        fields = line.split()
        if len(fields) > 5 and fields[0].isdigit():
            ontology[int(fields[0])] = (fields[1].replace("-", " "), "#" + "".join(f"{int(v):02x}" for v in fields[2:5]))
    codes = sorted(int(code) for code in np.unique(labels) if code)
    if set(codes) - ontology.keys():
        raise ValueError("SuperSynth output includes a code outside its pinned ontology")
    ref = c.put_array(labels) if codes else None
    existing = {row["id"] for row in c.document["layers"]}
    base = "supersynth-" + digest({"series": series_id, "output": record["output_sha256"]})[:12]
    for code in codes:
        layer_id = f"{base}-{code}"
        if layer_id in existing or (code in {915, 916} and not record["include_mri_skull"]):
            continue
        kind = _HEAD_KIND.get(code, _brain_kind(code) if code < 900 else "context")
        label, color = ontology[code]
        if kind in {"artery", "vein"}:
            label += " · inferred MRI context"
        elif kind == "skull":
            label += " · approximate MRI skull"
        else:
            label += " · SuperSynth estimate"
        fields = _model_layer_fields(c, series_id, im, record)
        fields["provenance"]["label_ontology"] = {
            "name": label.split(" · ")[0], "value": code, "lut_sha256": ASSETS[lut.name],
        }
        c.document["layers"].append({
            "id": layer_id, "patient_id": c.document["patient_id"], "source_series": series_id,
            "label": label, "kind": kind, "role": "context", "color": color,
            "affine": affine(im.affine).tolist(), "data": ref, "label_value": code, "revision": 0,
            "availability": "available", **fields,
        })
    return ref


def restrict_ct_skull(c, row, output):
    """Keep only estimated skull supported by dense signal in the native CT."""
    source = c.array(row["data"])
    im = nib.load(output["path"])
    if im.shape != source.shape or not np.allclose(im.affine, row["affine"], atol=1e-5):
        raise ValueError("CT skull estimate differs from its native source grid")
    mask = np.asanyarray(im.dataobj)
    if not np.isin(mask, [0, 1]).all() or file_hash(output["path"]) != output["sha256"]:
        raise ValueError("CT skull estimate is invalid or changed")
    kept = (mask > 0) & (source >= 250)
    count = int(np.count_nonzero(kept))
    if count == int(np.count_nonzero(mask)):
        return {**output, "foreground_voxels": count}
    identity = {"adapter": file_hash(__file__), "v": 1, "ct": row["original"]["sha256"],
                "skull": output["sha256"], "threshold_hu": 250}
    folder = c.root / "reconstructions" / digest(identity)
    path = folder / "ct-supported-skull.nii.gz"
    receipt = folder / "receipt.json"
    with reconstruction_lock():
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record["sha256"] != file_hash(path):
                raise ValueError("CT skull derivative changed")
            return record
        folder.mkdir(parents=True, exist_ok=True)
        saved = nib.Nifti1Image(kept.astype(np.uint8), im.affine, im.header.copy())
        saved.set_data_dtype(np.uint8)
        nib.save(saved, path)
        record = {**output, "path": str(path), "sha256": file_hash(path), "foreground_voxels": count}
        atomic_json(receipt, record)
        return record
