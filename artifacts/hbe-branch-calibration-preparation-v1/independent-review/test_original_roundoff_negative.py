"""Preserved original readout bytes, narrow analytical rounding regression."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scripts import mechanics_hbe_halfheight_readout as legacy


def test_original_reference_half_frame_exactness():
    path=Path(__file__).parent/'initial-source/mechanics_hbe_branch_calibration_readout.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='bdda5bf052ce29058f7c5ec99de8e56a0f63240bb5e2739af6ba23f547aa0887'
    spec=importlib.util.spec_from_file_location('preserved_branch_readout',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    m=SimpleNamespace(Xh=np.array([[0.,0.,0.],[.004,.004,.005]]),bottom=np.array([0]),midplane=np.array([1]),radius=.004,
                      half=SimpleNamespace(deformation=lambda x,mu:{'energy_J':mu}))
    raw=np.array([[0.,0.,0.],[1e-12,0.,0.]])
    original=legacy.HalfHeightReconstruction.half_frame(m,m.Xh,raw,full_displacement_m=0.)
    actual=module.half_frame(m,m.Xh,raw,full_displacement_m=0.,mu=1000.)
    assert actual==original
