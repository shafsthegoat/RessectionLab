"""PyInstaller entrypoint; also exposes an offline bundle dependency check."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import traceback


def package_check(report_path: Path) -> int:
    """Check imports inside the frozen interpreter, outside the source checkout."""
    report = {
        "schema_version": 1,
        "frozen": bool(getattr(sys, "frozen", False)),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "checks": {},
        "clinical_use_status": "research_only",
    }
    checks = report["checks"]
    try:
        if report["frozen"]:
            package_root = Path(sys._MEIPASS)
            provenance_path = package_root / "build_info" / "BUILD_INPUT_MANIFEST.json"
            provenance = json.loads(provenance_path.read_text())
            source_hashes = provenance["package_source_sha256"]
            checks["bundled_source_matches_snapshot"] = bool(source_hashes) and all(
                hashlib.sha256((package_root / relative).read_bytes()).hexdigest() == expected
                for relative, expected in source_hashes.items()
            )
            report["source_snapshot_sha256"] = provenance["source_snapshot_sha256"]
            report["original_source_revision"] = provenance["original_source_revision"]
            report["original_source_was_dirty"] = provenance["original_source_was_dirty"]
            report["bundled_source_files_verified"] = len(source_hashes)
        import numpy as np
        import nibabel as nib
        import scipy.ndimage
        import torch
        from PySide6 import QtCore, QtWidgets
        from vtkmodules.vtkCommonCore import vtkVersion
        from vtkmodules.vtkRenderingCore import vtkRenderer
        import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
        from resectionlab.app.desktop import main

        checks["numpy"] = bool(np.isfinite(np.ones(3)).all())
        checks["nibabel"] = nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)).shape == (2, 2, 2)
        checks["scipy"] = bool(scipy.ndimage.binary_dilation(np.array([False, True, False])).all())
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        parameter.square().sum().backward()
        optimizer.step()
        checks["torch_gradient_update"] = 0 < float(parameter.item()) < 1.0
        checks["qt"] = bool(QtCore.qVersion()) and callable(QtWidgets.QApplication)
        checks["vtk"] = bool(vtkRenderer()) and bool(vtkVersion.GetVTKVersion())
        checks["desktop_entrypoint"] = callable(main)

        # Exercise diagnostic APIs with an analytic signal, including their lazy
        # imports and packaged DIPY sphere data. This remains a synthetic check.
        from resectionlab.diffusion import DiffusionData, fit_tensor, probabilistic_crop_audit

        rng = np.random.default_rng(1729)
        directions = rng.normal(size=(32, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        vectors = np.vstack((np.zeros((1, 3)), directions, directions))
        bvals = np.r_[0.0, np.full(32, 1000.0), np.full(32, 2800.0)]
        eigenvalues = np.array([0.0015, 0.0004, 0.0004])
        attenuation = np.einsum("ni,i,ni->n", vectors, eigenvalues, vectors)
        signal = np.broadcast_to(1000.0 * np.exp(-bvals * attenuation), (8, 8, 8, len(bvals))).copy()
        diffusion = DiffusionData(signal, np.diag([2.0, 2.0, 2.0, 1.0]), bvals, vectors, "image_axes")
        tensor = fit_tensor(diffusion, np.ones((8, 8, 8), dtype=bool), diagnostic_only=True)
        expected_fa = np.sqrt(1.5 * np.sum((eigenvalues - eigenvalues.mean()) ** 2) / np.sum(eigenvalues ** 2))
        checks["dipy_analytic_tensor_fit"] = bool(
            np.allclose(tensor["fa"], expected_fa, atol=1e-4)
            and np.allclose(tensor["md"], eigenvalues.mean(), atol=1e-8)
            and np.all(np.abs(tensor["principal_ras"][..., 0]) > 0.999)
            and not tensor["report"]["usable_for_tract_aware_planning"]
        )
        tracking = probabilistic_crop_audit(
            diffusion, tensor, crop_start=(0, 0, 0), crop_shape=(8, 8, 8), seed_count=4, realizations=2,
        )
        checks["dipy_probabilistic_diagnostic"] = bool(
            tracking["report"]["streamline_count"] > 0
            and np.isfinite(tracking["points_ras_mm"]).all()
            and not tracking["report"]["usable_for_tract_aware_planning"]
            and tracking["report"]["clinical_deficit_probability"] is None
        )
        report["diffusion_check_scope"] = "analytic synthetic signal; diagnostic numerical execution only"
        report["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "nibabel", "torch", "PySide6", "vtk", "dipy")
        }
        report["passed"] = all(checks.values())
    except Exception:
        report["passed"] = False
        report["error"] = traceback.format_exc()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


def run() -> int:
    multiprocessing.freeze_support()
    if "--package-check" in sys.argv:
        parser = argparse.ArgumentParser(description="Verify the packaged runtime offline")
        parser.add_argument("--package-check", type=Path, required=True, metavar="REPORT_JSON")
        return package_check(parser.parse_args().package_check)
    from resectionlab.app import main

    return main() or 0


if __name__ == "__main__":
    raise SystemExit(run())
