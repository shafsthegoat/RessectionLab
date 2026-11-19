"""Preparation capabilities and source eligibility for installed adapters."""

from functools import lru_cache
import os
from pathlib import Path
import re
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from .patient_anatomy import file_hash, projection_or_localizer, source_metadata_phrases

ROOT = Path(__file__).resolve().parents[2]
SYNTHSEG_MAX_PADDED_CELLS = 192 * 192 * 160
STRUCTURAL_MRI = ("T1", "T1CE", "T2", "FLAIR", "MRI")
ANGIOGRAPHY = {"CTA", "TOF", "MRA"}
TUMOR_ADAPTERS = {
    "meningioma": ("MRI_Meningioma", "T1CE"),
    "glioma": ("MRI_Glioma", None),
    "glioma-gbm": ("MRI_GBM", "T1CE"),
    "glioma-lgg": ("MRI_LGGlioma", "FLAIR"),
}
POC_ADAPTERS = {
    "synthstrip": "synthstrip-main",
    "vesselfm": "vesselfm-base",
    "vesselfm-t1ce-experimental": "vesselfm-base",
    "totalsegmentator-skull": "totalsegmentator-skull",
}


def synthseg_assets(version):
    if version not in {1, 2}:
        raise ValueError("Select SynthSeg version 1 or 2")
    home = Path(os.environ.get("RESECTIONLAB_SYNTHSEG_HOME", ROOT / ".tools/SynthSeg"))
    python = Path(os.environ.get("RESECTIONLAB_SYNTHSEG_PYTHON", ROOT / ".tools/synthseg-runtime/bin/python"))
    names = ("synthseg_1.0.h5",) if version == 1 else (
        "synthseg_robust_2.0.h5", "synthseg_parc_2.0.h5", "synthseg_qc_2.0.h5",
    )
    return python, home / "scripts/commands/SynthSeg_predict.py", [home / "models" / name for name in names]


def synthseg_working_shape(shape, spacing):
    """Bound the 1 mm volume padded by the pinned model to multiples of 32."""
    resized = np.asarray(shape)
    if np.any(np.abs(spacing - 1) > .05):
        resized = np.ceil(resized * spacing).astype(int)
    return np.maximum(128, ((resized + 31) // 32) * 32)


def synthseg_requires_roi(row):
    spacing = nib.affines.voxel_sizes(np.asarray(row["affine"]))
    return int(np.prod(synthseg_working_shape(row["data"]["shape"], spacing))) > SYNTHSEG_MAX_PADDED_CELLS


def tumor_assets(model):
    home = Path(os.environ.get("RESECTIONLAB_RAIDIONICS_MODELS", ROOT / "data/models/raidionics"))
    folder = home / TUMOR_ADAPTERS[model][0]
    packages = Path(os.environ.get("RESECTIONLAB_RAIDIONICS_PACKAGES", ROOT / ".tools/raidionics-packages"))
    return folder, packages


def supersynth_home():
    explicit = os.environ.get("RESECTIONLAB_SUPERSYNTH_HOME")
    if explicit:
        home = Path(explicit).resolve()
        if home.is_relative_to(Path("/Volumes")):
            from .data_root import checked_root
            if not home.is_relative_to(checked_root()):
                raise ValueError("SuperSynth assets must be inside the verified data root")
        return home
    if os.environ.get("RESSECTIONLAB_DATA_ROOT"):
        from .data_root import checked_root
        return checked_root() / "cache/models/supersynth-source"
    return ROOT / "data/models/supersynth-source"


@lru_cache(maxsize=512)
def _file_identity(path, stat_key):
    return file_hash(path)


def _identities(paths):
    result = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing runtime or model asset: {path.name}")
        stat = path.stat()
        result[str(path)] = _file_identity(
            path, (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns),
        )
    return result

def model_environment(model, *, include_paths=False, **options):
    """Read-only preflight with content hashes keyed by each asset's logical role."""
    from . import anatomy_models, head_segmentation, poc_models

    assets = {}
    configuration = {}
    if model == "monai-brats-0.5.4":
        from . import anatomy_brats
        assets, configuration["model_release"] = anatomy_brats.assets()
    elif model in POC_ADAPTERS:
        config = poc_models.MODEL_CONFIG[POC_ADAPTERS[model]]
        home = os.environ.get("RESSECTIONLAB_MODEL_ROOT")
        model_root = Path(home).resolve() if home else ROOT / "data/models"
        if home:
            from .data_root import checked_root
            if not model_root.is_relative_to(checked_root()):
                raise ValueError("Configured model cache must be inside the external data root")
        weights = sorted(model_root.glob(config["weights_glob"].removeprefix("data/models/")))
        if not weights:
            raise FileNotFoundError(f"Local {config['name']} weights are missing")
        assets = {"python": poc_models.RUNTIME, "poc_worker": Path(poc_models.__file__)}
        assets.update({f"weights/{path.relative_to(model_root).as_posix()}": path for path in weights})
        if model == "synthstrip":
            assets.update({
                "model/synthstrip-v1/mri_synthstrip": model_root / "synthstrip-v1/mri_synthstrip",
                "brain_extraction": ROOT / "src/resectionlab/brain_extraction.py",
            })
        if sys.platform != "darwin":
            raise ValueError(f"The configured {config['name']} workflow requires an MPS runtime")
    elif model == "synthseg":
        python, script, weights = synthseg_assets(options.get("synthseg_version", 2))
        home = script.parents[2]
        paths = [script, *weights, *sorted((home / "data/labels_classes_priors").glob("*.npy"))]
        for folder in ("SynthSeg", "ext"):
            paths += sorted((home / folder).rglob("*.py"))
        assets = {"python": python, **{f"bundle/{path.relative_to(home).as_posix()}": path for path in paths}}
    elif model in TUMOR_ADAPTERS:
        folder, packages = tumor_assets(model)
        assets = {"python": Path(sys.executable), **{f"model/{folder.name}/{name}": folder / name for name in (
            "model.onnx", "pipeline.json", "pre_processing.ini", "model_config.txt",
        )}}
        paths = [packages / "raidionicsseg/fit.py", *sorted((packages / "raidionicsseg").rglob("*.py"))]
        assets.update({f"bundle/{path.relative_to(packages).as_posix()}": path for path in paths})
    elif model == "siam-v2":
        from . import anatomy_siam, siam_worker
        if sys.platform != "darwin":
            raise ValueError("The bounded SIAM v2 adapter requires the Metal runtime")
        python, home = anatomy_siam.assets()
        weights = {name: home / name for name in siam_worker.ASSETS}
        if not all(path.is_file() for path in weights.values()):
            raise FileNotFoundError("Local SIAM v2 weights or plans are missing. Run scripts/install_siam_v2.py explicitly.")
        verified = _identities(weights.values())
        if any(verified[str(path)] != siam_worker.ASSETS[name] for name, path in weights.items()):
            raise ValueError("SIAM weights or plans differ from the pinned publisher release")
        assets = {"python": python, "siam_worker": Path(siam_worker.__file__),
                  "siam_adapter": Path(anatomy_siam.__file__), **weights, **anatomy_siam.runtime_files(python)}
        configuration.update(parameters=siam_worker.PARAMETERS, runtime=siam_worker.RUNTIME)
    elif model == "supersynth":
        if sys.platform != "darwin":
            raise ValueError("The pinned tiled SuperSynth worker requires MPS")
        home = supersynth_home()
        assets = {"python": Path(sys.executable), "head_worker": Path(head_segmentation.__file__)}
        for name in head_segmentation.ASSETS:
            if not (home / name).is_file():
                raise FileNotFoundError(f"Pinned SuperSynth asset missing: {name}. Configure RESECTIONLAB_SUPERSYNTH_HOME or mount the model cache.")
        # The worker verifies these exact pinned hashes before inference.
        configuration.update(pinned_assets=head_segmentation.ASSETS, parameters=head_segmentation.PARAMETERS)
    elif model == "dural-envelope":
        from . import anatomy_dural
        assets = {"dural_adapter": Path(anatomy_dural.__file__),
                  "intracranial_geometry": ROOT / "src/resectionlab/anatomy_intracranial.py"}
    else:
        raise ValueError(f"Unknown preparation adapter: {model}")
    assets.update(adapter=Path(anatomy_models.__file__), registry=Path(__file__))
    hashes = _identities(assets.values())
    configuration["files"] = {role: hashes[str(path)] for role, path in assets.items()}
    if include_paths:
        configuration["file_paths"] = {role: str(path) for role, path in assets.items()}
    return configuration


def source_volume(row):
    """Recognize acquired volumes and exclude projections and localizers."""
    return (not row.get("synthetic") and len(row["data"]["shape"]) == 3 and min(row["data"]["shape"]) >= 8
            and not projection_or_localizer(row))


def ct_intensities(c, row):
    """Check documented calibration or a conservative air-to-bone signature."""
    if row["modality"] not in {"CT", "CTA"} or row.get("synthetic") or not source_volume(row):
        return False
    metadata = row.get("metadata", {})
    if str(row.get("intensity_units") or metadata.get("Units", "")).upper() in {"HU", "HOUNSFIELD", "HOUNSFIELD UNITS"}:
        return True
    if row.get("dicom_modality") == "CT" or row.get("dicom", {}).get("modality") == "CT":
        return True
    data = c.array(row["data"])
    step = np.maximum(1, np.ceil(np.asarray(data.shape) / 96).astype(int))
    sample = data[tuple(slice(None, None, s) for s in step)]
    low, middle, high = np.percentile(sample, (2, 50, 98))
    if not (-1100 <= low <= -850 and high >= 250):
        return False
    # A large air-padded field can put the median at the scanner floor.
    # Retain the air/dense-bone checks and require a soft-tissue population.
    return -1000 <= middle <= 300 or (middle < -1000 and np.mean((sample >= -150) & (sample <= 150)) >= .05)

def head_coverage(c, row, *, confirmed=False, brain=None):
    """Withhold whole-head estimates on ambiguous, truncated or altered MRI."""
    coverage = {**c.document.get("coverage_notes", {}), **row.get("coverage_notes", {}), **row.get("coverage", {})}

    def unsupported(reason):
        return {"status": "unsupported_input", "reason": reason}

    if coverage.get("status") == "limited" or any(coverage.get(key) or row.get(key)
            for key in ("privacy_redacted", "defaced", "skull_stripped", "cropped")):
        return unsupported("Source metadata indicates redaction, skull stripping or cropping")
    if re.search(r"mri_brain|skull.?stripp|brain.?extract|brain.?only|defac|redact|crop|(?:^|[_. -])strip(?:[_. -]|$)",
                 source_metadata_phrases(row)):
        return unsupported("Source designation indicates redaction, skull stripping or cropping")
    if row["modality"] not in STRUCTURAL_MRI or not source_volume(row):
        return unsupported("A full-head structural source volume is required")
    if row.get("units_status") != "mm":
        return unsupported("Full-head inference requires explicit millimeter source geometry")

    native = c.array(row["data"])
    stride = np.maximum(1, np.ceil(np.asarray(native.shape) / 96).astype(int))
    sliced = tuple(slice(None, None, step) for step in stride)
    a = np.asarray(row["affine"]) @ np.diag([*stride, 1])
    orientation = nib.orientations.io_orientation(a)
    sample = nib.orientations.apply_orientation(native[sliced], orientation)
    spacing = nib.affines.voxel_sizes(a)[np.argsort(orientation[:, 0])]
    if np.any(np.asarray(sample.shape) * spacing < [140, 150, 150]):
        return unsupported("Acquisition field too small to support complete outer head")
    low, high = np.percentile(sample, [1, 99])
    foreground = np.asarray(sample > low + .10 * (high - low))
    components, count = ndi.label(ndi.binary_opening(foreground))
    if high <= low or not count:
        return unsupported("Insufficient measured head signal")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    support = components == sizes.argmax()
    bounds = np.array([(item.start, item.stop) for item in ndi.find_objects(support, max_label=1)[0]])
    widths = (bounds[:, 1] - bounds[:, 0]) * spacing
    if np.any(widths < [122, 140, 135]):
        return unsupported("Head signal extent consistent with cropping or skull stripping")

    # Inferior extension is compatible with full-head MRI. Screen the other
    # faces for truncation and require anterior lower facial signal.
    for axis, index in ((0, 0), (0, -1), (1, 0), (1, -1), (2, -1)):
        if ndi.binary_opening(np.take(foreground, index, axis=axis)).mean() > .035:
            return unsupported("Measured head signal reaches a lateral, anterior, posterior or superior scan boundary")
    (lx, hx), (ly, hy), (lz, hz) = bounds
    face = foreground[lx:hx, int(ly + .72 * (hy - ly)):hy, int(lz + .15 * (hz - lz)):int(lz + .48 * (hz - lz))]
    if not face.size or float(face.mean()) < .10:
        return unsupported("Anterior lower head signal is incomplete or could be privacy-redacted")

    if brain is not None:
        if brain["source_series"] != row["id"] or tuple(brain["data"]["shape"]) != native.shape or not np.allclose(brain["affine"], row["affine"]):
            return unsupported("Head coverage needs a same-grid brain extraction")
        brain_data = c.array(brain["data"])[sliced] == brain["label_value"]
        brain_sample = nib.orientations.apply_orientation(brain_data, orientation)
        if not brain_sample.any():
            return unsupported("Brain extraction is empty; exterior coverage cannot be established")
        exterior = ndi.distance_transform_edt(~brain_sample, sampling=spacing) > 8
        exterior_fraction = np.count_nonzero(support & exterior) / np.count_nonzero(support)
        if exterior_fraction < .12:
            return unsupported("Too little measured tissue outside the brain envelope; likely skull-stripped")
    if not (confirmed or coverage.get("full_head_acquired") is True):
        return {"status": "unprocessed", "reason": "Confirm this scan includes the full head before running the outer-head model"}
    return {"status": "eligible", "reason": "Measured coverage screen passed; exterior completeness and model accuracy need review",
            "screen": "source-signal-coverage-v1", "support_extents_mm": widths.tolist(),
            "basis": "documented acquisition" if coverage.get("full_head_acquired") else "user-confirmed coverage"}


def matching_layer(c, row, kinds, *, native=False, supplied_only=False):
    if row is None:
        return None
    for layer in c.document["layers"]:
        provenance = layer.get("provenance", {})
        if layer["kind"] not in kinds or layer["source_series"] != row["id"]:
            continue
        if supplied_only and (provenance.get("model") or provenance.get("kind") in {
            "model_estimate", "image_derived_estimate", "geometric_estimate", "synthetic", "simulated",
        }):
            continue
        if native and (layer["data"]["shape"] != row["data"]["shape"] or not np.allclose(layer["affine"], row["affine"])):
            continue
        if np.any(c.array(layer["data"]) == layer["label_value"]):
            return layer
    return None

def preparation_model_jobs(c, *, indication=None, full_head_confirmed=False, include_dural_proxy=False):
    """Ordered in-study native proposals; registration is managed by preparation."""
    from .anatomy_model_cache import reusable_model_layer

    jobs = []
    reference_id = c.document.get("reference_series")
    reference = c.series(reference_id) if reference_id else None
    series = sorted((row for row in c.document["series"] if not reference
                     or row["timepoint"] == reference["timepoint"]), key=lambda r: r["id"] != reference_id)

    def sources(modalities):
        return (r for mod in modalities for r in series if r["modality"] == mod)

    def pick(modalities):
        return next((row for row in sources(modalities) if source_volume(row)), None)

    def job(job_id, capability, row=None, model=None, *, status="ready", reason=None,
            options=None, depends_on=(), existing=None):
        options = options or {}
        environment = None
        if existing is not None:
            provenance = existing.get("provenance", {})
            estimated = bool(provenance.get("model") or provenance.get("inference"))
            status = "available"
            reason = "Verified existing estimate; review status retained" if estimated else "A supported supplied layer is already present"
        elif status == "ready":
            try:
                environment = model_environment(model, include_paths=True, **options)
            except (OSError, ValueError) as error:
                status, reason = "unavailable", str(error)
        stage = {"id": job_id, "capability": capability, "series_id": row["id"] if row else None,
                 "model": model, "options": options, "status": status, "reason": reason,
                 "depends_on": list(depends_on), "environment": environment}
        if existing is not None:
            stage["layer_ids"] = [existing["id"]]
        jobs.append(stage)
        return stage

    mri = reference if reference and reference["modality"] in STRUCTURAL_MRI and source_volume(reference) else pick(STRUCTURAL_MRI)
    ct_sources = list(sources(("CT", "CTA")))
    calibrated = [row for row in ct_sources if ct_intensities(c, row)]
    ct = next(iter(calibrated), None)
    ct_status = "unsupported_input" if ct_sources else "missing_input"
    calibrated_ids = {row["id"] for row in calibrated}
    unsupported_ct = [row for row in ct_sources if row["id"] not in calibrated_ids]
    for index, row in enumerate([*calibrated, *unsupported_ct]):
        stage_id = "ct-skull" if index == 0 else "ct-skull:" + row["id"]
        label = f"Skull and brain · {row.get('label') or row['id']} ({row['modality']})"
        if row["id"] not in calibrated_ids:
            job(stage_id, "skull", row, status="unsupported_input",
                reason="An acquired volumetric CT with supported HU intensities is required")["label"] = label
            continue
        # The CT adapter returns both masks. A supplied skull alone leaves its
        # same-acquisition brain estimate useful for viewing and dural context.
        skull = matching_layer(c, row, {"skull"}, supplied_only=True)
        brain = matching_layer(c, row, {"brain"}, supplied_only=True) if skull else None
        stage = job(stage_id, "skull", row, "totalsegmentator-skull", existing=skull if brain else None)
        if skull and brain:
            stage["layer_ids"].append(brain["id"])
        stage["label"] = label
    brain_jobs = {}

    region_kinds = {"brain-region", "brainstem", "cerebellum", "cerebellar-white-matter", "cerebellar-cortex"}

    def brain_job(row, *, native=True):
        key = row["id"]
        prior = brain_jobs.get(key)
        if prior and native and prior.get("layer_ids"):
            candidate = c.layer(prior["layer_ids"][0])
            if candidate["data"]["shape"] != row["data"]["shape"] or not np.allclose(candidate["affine"], row["affine"]):
                key += ":native"
        if key not in brain_jobs:
            existing = matching_layer(c, row, {"brain"}, native=native, supplied_only=True)
            if existing is None:
                existing = reusable_model_layer(c, row, {"brain"}, model="synthstrip", native=native)
            if existing is None and not native:
                existing = reusable_model_layer(c, row, {"brain"}, model="supersynth")
            brain_jobs[key] = job("brain:" + key, "brain", row, "synthstrip", existing=existing)
        return brain_jobs[key]["id"]

    if mri:
        brain_job(mri, native=False)
        regions = (reusable_model_layer(c, mri, region_kinds, model="supersynth")
                   or reusable_model_layer(c, mri, region_kinds, model="synthseg"))
        region_options = {"synthseg_version": 2, "synthseg_resource_bound": True}
        dependencies = []
        if regions is None:
            try:
                model_environment("synthseg", **region_options)
            except (OSError, ValueError):
                region_options["synthseg_version"] = 1
            if synthseg_requires_roi(mri):
                brain = matching_layer(c, mri, {"brain"}, native=True)
                if brain:
                    region_options["brain_layer_id"] = brain["id"]
                else:
                    dependencies.append(brain_job(mri, native=True))
        job("brain-regions", "brain_regions", mri, "synthseg", options=region_options,
            depends_on=dependencies, existing=regions)
    elif not ct:
        job("brain", "brain", status="missing_input", reason="Structural MRI or HU-calibrated CT required")
        job("brain-regions", "brain_regions", status="missing_input", reason="Structural MRI required for the configured regional workflow")
    else:
        job("brain-regions", "brain_regions", status="unsupported_input",
            reason="Regional estimates are configured for structural MRI")

    head_profiles = [(row, head_coverage(c, row, confirmed=full_head_confirmed))
                     for row in series if row["modality"] in STRUCTURAL_MRI]
    head = next((row for row, profile in head_profiles if profile["status"] == "eligible"), None)
    reused_head = next((existing for row, profile in head_profiles if profile["status"] in {"eligible", "unprocessed"}
                       if (existing := reusable_model_layer(c, row, {"context", "scalp"}, model="supersynth"))), None)
    if reused_head is not None:
        head = c.series(reused_head["source_series"])
        job("mri-head", "outer_head", head, "supersynth", existing=reused_head)
    elif head:
        brain = matching_layer(c, head, {"brain"}, native=True)
        # An acquired CT has its own field and may exclude part of the skull.
        options = {"include_mri_skull": True, "full_head_confirmed": full_head_confirmed}
        if brain:
            options["brain_layer_id"] = brain["id"]
        job("mri-head", "outer_head", head, "supersynth", options=options,
            depends_on=[] if brain else [brain_job(head)])
    else:
        # If the reference is cropped, identify the other scan which passed
        # the measured screen and is waiting for coverage confirmation.
        row, profile = next(((row, profile) for row, profile in head_profiles if profile["status"] == "unprocessed"),
                            next(iter(head_profiles), (None, None)))
        job("mri-head", "outer_head", row, status=profile["status"] if profile else "missing_input",
            reason=profile["reason"] if profile else "Full-head structural MRI required")
    # A reduced-field reference cannot hide an intact acquisition. Each MRI
    # retains its own native labels, coverage result and checkpoint.
    priority = {"eligible": 0, "unprocessed": 1}
    if not head_profiles:
        stage = job("siam-tissues", "cranial_tissues", model="siam-v2", status="missing_input",
                    reason="Acquired full-head structural MRI required")
        stage["label"] = "SIAM v2 · dura, skull, vessels and brain tissues"
    for index, (row, profile) in enumerate(sorted(head_profiles, key=lambda item: priority.get(item[1]["status"], 2))):
        eligible = profile["status"] == "eligible"
        stage_id = "siam-tissues" if index == 0 else "siam-tissues:" + row["id"]
        stage = job(stage_id, "cranial_tissues", row, "siam-v2",
                    status="ready" if eligible else profile["status"],
                    reason=None if eligible else profile["reason"],
                    options={"full_head_confirmed": full_head_confirmed})
        stage["label"] = f"SIAM v2 dura, skull, vessels and brain tissues · {row.get('label') or row['id']} ({row['modality']})"

    if not ct and not head:
        job("skull", "skull", status=ct_status, reason="Acquired HU-calibrated CT or covered full-head MRI required")

    # Supplied annotations label positive anatomy only. Neither a named
    # artery nor an unclassified mask accounts for the remaining vessels.
    angiograms = sorted(sources(("CTA", "TOF", "MRA")), key=lambda row: not source_volume(row))
    angio_reason = "Vascular proposals require acquired volumetric CTA, TOF or MRA source images"
    for index, row in enumerate(angiograms):
        eligible = source_volume(row)
        existing = reusable_model_layer(c, row, {"vessels"}, model="vesselfm-base") if eligible else None
        stage = job("vascular" if index == 0 else "vascular:" + row["id"], "vessels", row, "vesselfm",
                    status="ready" if eligible else "unsupported_input", reason=None if eligible else angio_reason,
                    existing=existing)
        stage["label"] = f"Vessel candidates · {row.get('label') or row['id']} ({row['modality']})"
    if not angiograms:
        projection = any(row["modality"] == "MIP" and (not reference or row.get("timepoint") == reference["timepoint"])
                         for row in c.document.get("source_assets", []))
        job("vascular", "vessels", status="unsupported_input" if projection else "missing_input", reason=angio_reason)
    venous = next(sources(("MRV", "SWI", "SWI_MAGNITUDE", "SWI_PHASE")), None)
    for stage_id, capability, kind in (("venous", "veins", "vein"), ("sinuses", "dural_venous_sinuses", "sinus")):
        supplied = next((layer for row in series if (layer := matching_layer(c, row, {kind}, supplied_only=True))), None)
        job(stage_id, capability, c.series(supplied["source_series"]) if supplied else venous,
            status="unsupported_input" if venous else "missing_input", existing=supplied,
            reason="No installed venous adapter supports this source contrast" if venous else "Dedicated venous source images unavailable")

    tumor_kinds = {"tumor", "enhancing-tumor", "nonenhancing-tumor", "lesion", "tumor-core", "tumor-whole"}
    supplied = next((layer for row in series
                     if (layer := matching_layer(c, row, tumor_kinds, supplied_only=True))), None)
    if indication == "monai-brats-0.5.4":
        from . import anatomy_brats
        if supplied:
            job("supplied-tumor", "supplied_tumor", c.series(supplied["source_series"]), existing=supplied)
        try:
            if not reference:
                raise ValueError("A patient reference series is required")
            evidence = anatomy_brats.inputs(c)
        except (ValueError, OSError) as error:
            job("tumor", "four_channel_tumor_compartments", status="unsupported_input", reason=str(error))
        else:
            job("tumor", "four_channel_tumor_compartments", c.series(evidence["channels"][0]["series_id"]),
                indication, options={"brats_inputs": evidence}, reason="Unreviewed estimates on the common preprocessed grid")
    elif supplied:
        job("tumor", "tumor", c.series(supplied["source_series"]), existing=supplied)
    elif indication in TUMOR_ADAPTERS and indication != "glioma":
        _, modality = TUMOR_ADAPTERS[indication]
        tumor_mri = pick((modality,))
        if tumor_mri:
            existing = reusable_model_layer(c, tumor_mri, tumor_kinds, model=indication, native=True)
            options, dependencies = {}, []
            if existing is None and indication in {"glioma-gbm", "glioma-lgg"}:
                brain = matching_layer(c, tumor_mri, {"brain"}, native=True)
                if brain:
                    options["brain_layer_id"] = brain["id"]
                else:
                    dependencies.append(brain_job(tumor_mri))
            job("tumor", "tumor", tumor_mri, indication, options=options, depends_on=dependencies, existing=existing)
        else:
            job("tumor", "tumor", status="missing_input", reason=f"The requested tumor adapter requires {modality}")
    else:
        job("tumor", "tumor", status="unsupported_input" if mri else "missing_input",
            reason=("Select a supported tumor model indication or import a supplied lesion label" if mri
                    else "A supported MRI sequence and tumor indication or supplied lesion label is required"))

    if include_dural_proxy:
        potential = ct or head
        if potential:
            options = {}
            brain = matching_layer(c, potential, {"brain"})
            skull = matching_layer(c, potential, {"skull"}) if not ct else None
            if brain:
                options["brain_layer_id"] = brain["id"]
            if skull:
                options["cranial_layer_id"] = skull["id"]
            job("dural-envelope", "dural_envelope", potential, "dural-envelope", options=options,
                depends_on=["ct-skull" if ct else "mri-head"])
        else:
            job("dural-envelope", "dural_envelope", status="missing_input",
                reason="No source-supported cranial envelope available for a geometric proxy")
    tracts = bool(c.document.get("tractography")) or any(row["kind"] in {"tract", "tractography"} for row in c.document["layers"])
    raw_dwi = any(row.get("storage") == "source-asset" and row.get("modality") == "DWI"
                  and (not reference or row.get("timepoint") == reference["timepoint"])
                  for row in c.document.get("source_assets", []))
    if tracts:
        job("tractography", "tractography", status="available", reason="Supplied tracts retained with their limitations")
    elif raw_dwi:
        job("tractography", "tractography", status="unsupported_input",
            reason="Raw diffusion retained. Named tracts require reviewed preprocessing, gradient orientation and bundle evidence.")
    else:
        job("tractography", "tractography", status="missing_input", reason="No supplied patient-specific tractography")
    functional = any(row["kind"] in {"motor", "language", "functional"} for row in c.document["layers"])
    job("functional", "functional", status="available" if functional else "missing_input",
        reason="Supplied functional maps retained with their limitations" if functional else "No supplied patient-specific functional evidence")
    return jobs
