"""Four acquired channels have independent identity, coverage and alignment gates."""
import json

import nibabel as nib
import numpy as np
import pytest

from resectionlab import anatomy_brats as brats
from resectionlab.anatomy_import import inspect_nifti_series
from resectionlab.patient_anatomy import PatientImagingCollection


def study(tmp_path):
    c = PatientImagingCollection.create(tmp_path / "patient", "test")
    image_affine = np.diag([-1., -1., 1., 1.])
    for index, modality in enumerate(brats.CHANNELS):
        path = tmp_path / f"{modality}.nii.gz"
        data = np.zeros((12, 14, 16), np.float32)
        data[1:-1, 1:-1, 1:-1] = index + 1
        image = nib.Nifti1Image(data, image_affine)
        image.header.set_xyzt_units("mm")
        nib.save(image, path)
        sidecar = tmp_path / f"{modality}.json"
        sidecar.write_text(json.dumps({"BrainExtracted": True, "WholeBrainAcquired": True}))
        row = c.add_series(path, series_id=modality, modality=modality)
        row["coverage"] = inspect_nifti_series(path, sidecar=sidecar)["coverage"]
        if index:
            c.set_registration(modality, np.eye(4), status="source-provided", method="fixture")
    return c


def test_all_channels_bind_native_content_and_changes(tmp_path):
    c = study(tmp_path)
    evidence = brats.inputs(c)
    assert [r["modality"] for r in evidence["channels"]] == list(brats.CHANNELS)
    assert evidence["development_overlap"] == "unassessed"
    c.path(c.series("FLAIR")["data"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="content changed"):
        brats.inputs(c)


@pytest.mark.parametrize("problem", ["missing", "coverage", "skull", "alignment", "translation", "duplicate"])
def test_multimodal_abstentions(tmp_path, problem):
    c = study(tmp_path)
    row = c.series("T2")
    if problem == "missing":
        c.document["series"].remove(row)
    elif problem == "coverage":
        row["coverage"]["limitations"].append({"kind": "cropped"})
    elif problem == "skull":
        row.pop("coverage")
    elif problem == "alignment":
        c.document["transforms"][row["id"]]["status"] = "unreviewed"
    elif problem == "translation":
        matrix = np.eye(4)
        matrix[0, 3] = 7
        c.set_registration("T2", matrix, status="source-provided", method="negative control")
    else:
        row["original"] = c.series("T1")["original"]
    with pytest.raises(ValueError):
        brats.inputs(c)
