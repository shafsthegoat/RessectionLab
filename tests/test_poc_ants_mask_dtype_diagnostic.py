"""Generated controls only; no ANTs import, patient decoding, or fitting."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("generated_mask_probe", ROOT/"scripts/poc_ants_mask_dtype_diagnostic.py")
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)


def native(mask, affine, arm):
    return SimpleNamespace(pixeltype={"uint8":"unsigned char", "float32":"float"}[arm],
        shape=mask.shape, numpy=lambda: mask.astype(arm), **probe.reference().ras_to_lps_geometry(affine))


def test_generated_phantom_deterministic_positive_three_classes_independent_geometry():
    values, mask, labels, affine = probe.phantom()
    assert values.shape == (48,48,48) and values.dtype == np.float32
    assert np.array_equal(affine, np.eye(4))
    # Independent point checks across concentric regions and outside support.
    for index, expected in [((23,23,23),3), ((35,23,23),2), ((41,23,23),1), ((46,23,23),0)]:
        assert labels[index] == expected
    coords = np.meshgrid(np.arange(48)-23.5, np.arange(48)-23.5, np.arange(48)-23.5, indexing="ij")
    assert np.array_equal(mask, coords[0]**2+coords[1]**2+coords[2]**2 <= 400)
    assert np.all(values[mask] > 0) and np.all(values[~mask] == 0)
    assert np.array_equal(probe.phantom()[0], values)


@pytest.mark.parametrize("arm", probe.ARMS)
def test_only_mask_representation_changes_and_native_frame_is_exact(arm):
    _, mask, _, affine = probe.phantom()
    original = mask.copy()
    receipt = probe.boundary_checks(mask, native(mask, affine, arm), affine, arm, probe.reference())
    assert receipt["support_exact"] and receipt["values_exact"]
    assert receipt["frame"]["eight_cell_corner_errors_mm"] == [0]*8
    assert np.array_equal(mask, original)


@pytest.mark.parametrize("fault", ["type", "dtype", "support", "origin", "shape"])
def test_float_boundary_rejects_wrong_type_changed_values_or_frame(fault):
    _, mask, _, affine = probe.phantom()
    candidate = native(mask, affine, "float32")
    if fault == "type": candidate.pixeltype = "unsigned char"
    if fault == "dtype": candidate.numpy = lambda: mask.astype(np.float64)
    if fault == "support":
        changed = mask.astype(np.float32); changed[23,23,23] = 0
        candidate.numpy = lambda: changed
    if fault == "origin": candidate.origin = (1,0,0)
    if fault == "shape": candidate.shape = (47,48,48)
    with pytest.raises(ValueError): probe.boundary_checks(mask, candidate, affine, "float32", probe.reference())


def test_unreleased_probe_does_not_access_files_runtime_or_inputs(monkeypatch):
    monkeypatch.setattr(probe, "reference", lambda: pytest.fail("Unreleased access"))
    with pytest.raises(ValueError, match="release"): probe.run()


def test_each_resource_control_is_effective():
    safe = [100, 2*1024**3, 1, 1, 100*1024**3]
    assert probe.violation(*safe) is None
    for index, bad in enumerate([1024**3+1, 1024**3-1, 120.01, 16*1024**2+1, 50*1024**3-1]):
        args = safe.copy(); args[index] = bad
        assert probe.violation(*args) is not None


def test_reference_and_fitting_parameters_remain_frozen():
    base = probe.reference()
    assert probe.sha(probe.REFERENCE_PATH) == probe.REFERENCE_SHA
    assert base.N4_PARAMETERS["spline_param"] == 150
    assert base.N4_PARAMETERS["convergence"] == {"iters":[30,20,10], "tol":1e-7}
    assert base.ATROPOS_PARAMETERS == {"i":"KMeans[3]", "m":"[0.1,1x1x1]", "c":"[10,0]", "priorweight":0., "r":0, "v":1}


def test_runtime_alias_requires_same_physical_target_and_identical_binary_pins(tmp_path):
    real = tmp_path/"runtime"; real.mkdir()
    alias = tmp_path/"alias"; alias.symlink_to(real, target_is_directory=True)
    a = {"prefix":str(real), "binary_sha":"frozen"}
    b = {"prefix":str(alias), "binary_sha":"frozen"}
    assert probe.same_runtime(a,b)
    b["binary_sha"] = "foreign"
    assert not probe.same_runtime(a,b)
    b = {"prefix":str(tmp_path/"other"), "binary_sha":"frozen"}
    assert not probe.same_runtime(a,b)
