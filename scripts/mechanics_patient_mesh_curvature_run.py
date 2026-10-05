#!/usr/bin/env python3
"""Explicit configuration of the shared reviewed one-attempt mesh launcher."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED_SHA = 'cfee357faf585f78afccf812d06c00de9022494e8ed4d82ef3fa68a97eaa38cb'
HELPER_SHA = '6b5aff2ee96be527f9824109e19522aa31b004c4a5a049f077cc1179e822af79'


def load(relative, expected, name):
    path = ROOT/relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Reviewed launcher/helper source changed')
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


SHARED = load('scripts/mechanics_patient_mesh_candidate_run.py', SHARED_SHA, 'curvature_launcher_shared')
HELPER = load('scripts/mechanics_patient_mesh_curvature.py', HELPER_SHA, 'curvature_helper')
SPEC = SHARED.ExecutionSpec(
    version='resect-case4-patient-mesh-curvature-v3-release',
    candidate_manifest=HELPER.MANIFEST,
    closure=SHARED.CLOSURE | frozenset([
        'scripts/mechanics_patient_mesh_curvature.py',
        'scripts/mechanics_patient_mesh_curvature_run.py', HELPER.MANIFEST]),
    context_roles=SHARED.CONTEXT_ROLES,
    helper_path='scripts/mechanics_patient_mesh_curvature.py',
    entrypoint='scripts/mechanics_patient_mesh_curvature_run.py',
    output_bytes=32*1024**2, file_bytes=8*1024**2, output_files=32,
    validate_config=HELPER.validate_config)


if __name__ == '__main__':
    raise SystemExit(SHARED.main(spec=SPEC))
