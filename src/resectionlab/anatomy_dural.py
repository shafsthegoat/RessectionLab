"""Optional display-only proxy from defensibly enclosed axial cranial regions."""


import nibabel as nib
from nibabel.spaces import vox2out_vox
import numpy as np
from scipy import ndimage as ndi

from .anatomy_intracranial import enclosed_rings
from .anatomy_model_registry import ct_intensities, matching_layer
from .data_root import atomic_json
from .patient_anatomy import digest, file_hash, reconstruction_lock, resample

SPACING_MM = 1.5


def reconstruct_dura(c, row, *, cranial_layer_id, brain_layer_id, cancelled):
    brain = c.layer(brain_layer_id) if brain_layer_id else matching_layer(c, row, {"brain"})
    if brain is None or brain["source_series"] != row["id"] or brain["kind"] != "brain":
        raise ValueError("Dural context needs a same-source brain estimate to locate the cranial interior")
    hu = ct_intensities(c, row)
    skull = None
    if not hu:
        skull = c.layer(cranial_layer_id) if cranial_layer_id else matching_layer(c, row, {"skull"})
        if (skull is None or skull["source_series"] != row["id"] or skull["kind"] != "skull"
                or skull.get("provenance", {}).get("model") != "supersynth"
                or skull.get("provenance", {}).get("coverage_screen", {}).get("status") != "eligible"):
            raise ValueError("Dural context requires acquired CT in HU or a covered pinned SuperSynth cranial estimate")
    dependencies = [{"id": r["id"], "data": r["data"], "affine": r["affine"],
                     "value": r["label_value"], "revision": r.get("revision", 0)}
                    for r in (brain, skull) if r is not None]
    parameters = {
        "grid_spacing_mm": SPACING_MM,
        "wall": "voxels above 300 HU" if hu else "pinned categorical MRI cortical and cancellous bone estimates",
        "shell": "one in-plane grid cell at inner boundary of supported axial rings, within 3 mm of source wall",
        "ring_algorithm": "brain-seeded enclosing axial rings with a binary closing radius of ceil(2 mm / spacing) cells",
        "interpolation": "linear for HU, nearest for labels",
    }
    identity = {"model": "dural-envelope", "version": 1, "source": row["original"]["sha256"],
                "dependencies": dependencies, "parameters": parameters,
                "adapter_sha256": file_hash(__file__)}
    folder = c.root / "reconstructions" / digest(identity)
    path = folder / "dural-proxy.nii.gz"
    receipt = folder / "receipt.json"
    with reconstruction_lock():
        if receipt.exists():
            import json
            record = json.loads(receipt.read_text())
            if record["output_sha256"] != file_hash(path):
                raise ValueError("Cached dural proxy changed")
            return record
        shape, a = vox2out_vox((tuple(row["data"]["shape"]), np.asarray(row["affine"])), voxel_sizes=(SPACING_MM,) * 3)
        if np.prod(shape) > 20_000_000:
            raise ValueError("Dural proxy exceeds bounded grid size")
        if hu:
            wall = resample(c.array(row["data"]), row["affine"], shape, a, order=1)
        else:
            # Both published bone classes use one categorical model field.
            # At the edge of the acquisition, withheld space stays unknown.
            labels = c.array(skull["data"])
            b = np.isin(labels, [915, 916]) if not skull.get("edits") else c.mask(skull)
            wall = resample(b.astype(np.uint16) * 301, skull["affine"], shape, a, order=0)
        seed = resample(c.mask(brain).astype(np.uint8), brain["affine"], shape, a, order=0) > 0
        inside, _, supported = enclosed_rings(wall, seed, SPACING_MM, cancelled)
        shell = np.zeros(shape, np.uint8)
        for z in np.flatnonzero(supported):
            interior = inside[:, :, z]
            edge = interior & ~ndi.binary_erosion(interior)
            distance_from_wall = ndi.distance_transform_edt(wall[:, :, z] <= 300, sampling=SPACING_MM)
            shell[:, :, z] = edge & (distance_from_wall <= 3)
        if cancelled():
            raise InterruptedError("Dural proxy cancelled before admission")
        folder.mkdir(parents=True, exist_ok=True)
        im = nib.Nifti1Image(shell, a)
        im.header.set_xyzt_units("mm")
        nib.save(im, path)
        slices = np.flatnonzero(shell.any(axis=(0, 1))).tolist()
        record = {
            **identity, "kind": "dura", "source_series": row["id"], "method": "geometric_estimate",
            "path": str(path), "output_sha256": file_hash(path),
            "availability": "available" if slices else "empty_output",
            "label_ontology": {"value": 1, "meaning": "Geometric inner-cranial envelope proxy"},
            "evidence_use": "context_only", "clinical_validation": False,
            "supported_axial_indices": slices,
            "coverage_limitation": "Only supported axial rings near source wall. Actual dura, sinuses, thickness, measured craniotomy and drilling are unassessed.",
        }
        atomic_json(receipt, record)
        return record
