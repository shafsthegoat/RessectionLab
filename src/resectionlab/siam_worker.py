"""Bounded single-fold SIAM v2 inference on an acquired structural head MRI."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

RECORD = "https://doi.org/10.5281/zenodo.15780983"
PUBLISHER = "https://github.com/romainVala/SIAM/tree/52baf64d51275a949d46ee36967b76f8ceaa99d8"
ARCHIVE_MD5 = "a05a10af1f4b02ef1dcf0dac36cc4a2b"
ASSETS = {
    "fold_1/checkpoint_final.pth": "33cc846f7fb3cf89aa69ac4723ab30aeecc7d84eebfc39e31e225db8bc1aa896",
    "dataset.json": "e79a9d921827ef67feda0da22da917278c4dd97fe0cd2c3593307d08d49238a2",
    "plans.json": "659259bb07fbcce82b346421b1a808add3fd9e2552d9cbd977b1263187263797",
}
RUNTIME = {"torch": "2.14.1", "nnunetv2": "2.5.2", "dynamic-network-architectures": "0.3.1",
           "numpy": "2.0.1", "scipy": "1.15.3", "nibabel": "5.4.2"}
PARAMETERS = {"fold": 1, "configuration": "3d_fullres", "device": "mps", "network_precision": "float32",
              "logit_accumulator": "nnU-Net float16 CPU",
              "tile_zyx": [128, 128, 128], "tile_step_fraction": 0.75, "gaussian_blend": True,
              "mirroring": False, "max_working_voxels": 45_000_000}
# Publisher labels for model 2. Head designates remaining head tissue.
LABELS = (
    ("background", "Background", "context", "#000000"),
    ("GM", "Gray matter", "brain-region", "#ba8e9a"),
    ("WM", "White matter", "brain-region", "#e9d9ca"),
    ("CSF", "Cerebrospinal fluid", "csf", "#629cd4"),
    ("CSFv", "Ventricular CSF", "ventricles", "#3c79b6"),
    ("cerGM", "Cerebellar gray matter", "cerebellar-cortex", "#c194a3"),
    ("Thal", "Thalamus", "brain-region", "#b19bcc"),
    ("Pal", "Pallidum", "brain-region", "#8aadc3"),
    ("Put", "Putamen", "brain-region", "#d69b7b"),
    ("Caud", "Caudate", "brain-region", "#b4cd83"),
    ("Accu", "Nucleus accumbens", "brain-region", "#d7c67b"),
    ("Amyg", "Amygdala", "brain-region", "#d398ae"),
    ("Hippo", "Hippocampus", "brain-region", "#bd84ab"),
    ("Dura", "Dura mater", "dura", "#d4ad84"),
    ("vascular", "Vascular tissue", "vessels", "#bc5465"),
    ("Skull", "Skull", "skull", "#d9cdb6"),
    ("Head", "Other head tissue", "context", "#b19e90"),
)


def validate_assets(folder):
    for name, expected in ASSETS.items():
        with (folder / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Pinned SIAM asset differs: {name}")
    labels = json.loads((folder / "dataset.json").read_text())["labels"]
    if labels != {row[0]: i for i, row in enumerate(LABELS)}:
        raise ValueError("SIAM v2 publisher dataset ontology differs")


def restore_logits(logits, props, plans, config):
    """Argmax after per-class publisher resampling, with a single full-size score buffer."""
    shape = props["shape_after_cropping_and_before_resampling"]
    spacing = [props["spacing"][axis] for axis in plans.transpose_forward]
    maximum = np.full(shape, -np.inf, dtype=np.float32)
    labels = np.zeros(shape, np.uint8)
    for code in range(len(LABELS)):
        score = np.asarray(config.resampling_fn_probabilities(
            logits[code:code + 1].float().numpy(), shape, config.spacing, spacing,
        )[0])
        if not np.isfinite(score).all():
            raise ValueError("SIAM returned nonfinite class scores")
        wins = score > maximum
        labels[wins] = code
        np.maximum(maximum, score, out=maximum)
    uncropped = np.zeros(props["shape_before_cropping"], np.uint8)
    uncropped[tuple(slice(lo, hi) for lo, hi in props["bbox_used_for_cropping"])] = labels
    return uncropped.transpose(plans.transpose_backward)


def restore_orientation(canonical_zyx, original):
    ras = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(ras, nib.orientations.io_orientation(original.affine))
    labels = nib.orientations.apply_orientation(canonical_zyx.T, transform)
    if labels.shape != original.shape:
        raise ValueError("SIAM restoration failed to recover the native shape")
    return labels


def predict(source, output, folder):
    validate_assets(folder)
    versions = {name: metadata.version(name) for name in RUNTIME}
    if versions != RUNTIME:
        raise ValueError(f"SIAM requires the pinned POC runtime; found {versions}")
    os.environ.update(nnUNet_compile="false", nnUNet_def_n_proc="1", PYTORCH_ENABLE_MPS_FALLBACK="1")
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    if not torch.backends.mps.is_available():
        raise ValueError("The SIAM single-fold adapter requires the Metal runtime")
    torch.set_num_threads(2)
    torch.mps.set_per_process_memory_fraction(0.55)
    original = nib.load(source)
    canonical = nib.as_closest_canonical(original)
    spacing = nib.affines.voxel_sizes(canonical.affine)[::-1]
    predictor = nnUNetPredictor(tile_step_size=PARAMETERS["tile_step_fraction"], use_gaussian=True,
                                use_mirroring=False, perform_everything_on_device=False,
                                device=torch.device("mps"), allow_tqdm=True)
    predictor.initialize_from_trained_model_folder(str(folder), [PARAMETERS["fold"]])
    plans, config = predictor.plans_manager, predictor.configuration_manager
    if plans.image_reader_writer_class.__name__ not in {"SimpleITKIO", "NibabelIO"}:
        raise ValueError("Unexpected nnU-Net image reader axes")
    shape = np.asarray(canonical.shape[::-1])[plans.transpose_forward]
    field = np.ceil(shape * spacing[plans.transpose_forward] / config.spacing).astype(int)
    if np.prod(field) > PARAMETERS["max_working_voxels"]:
        raise ValueError(f"SIAM head field {field.tolist()} exceeds the bounded working volume")
    # nnU-Net readers use ZYX on a canonical RAS input. Only permute and flip
    # voxel axes here so oblique acquisitions retain their source geometry.
    image = np.ascontiguousarray(canonical.get_fdata(dtype=np.float32).T[None])
    props = {"spacing": spacing.tolist()}
    data, mask = config.preprocessor_class(verbose=False).run_case_npy(
        image, None, props, plans, config, predictor.dataset_json)
    del image, mask, canonical
    config.configuration["patch_size"] = PARAMETERS["tile_zyx"]
    predictor.list_of_parameters = None  # already loaded by initialization
    with torch.inference_mode():
        logits = predictor.predict_sliding_window_return_logits(torch.from_numpy(data))
    del predictor, data
    torch.mps.empty_cache()
    labels = restore_orientation(restore_logits(logits, props, plans, config), original)
    saved = nib.Nifti1Image(labels, original.affine, original.header.copy())
    saved.set_data_dtype(np.uint8)
    nib.save(saved, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "folder"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    predict(args.source, args.output, args.folder)
