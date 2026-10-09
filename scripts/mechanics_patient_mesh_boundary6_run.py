#!/usr/bin/env python3
"""Source-bound, separately released Case4 boundary-6 geometry attempt."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED_SHA = 'cfee357faf585f78afccf812d06c00de9022494e8ed4d82ef3fa68a97eaa38cb'
HELPER_SHA = 'd4ee218653edb0a44e865c39699e3fc6e182a81947c43cc99d2bf2ab030ddf2e'


def load(relative, expected, name):
    path = ROOT / relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Reviewed launcher/helper source changed')
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


SHARED = load('scripts/mechanics_patient_mesh_candidate_run.py', SHARED_SHA,
              'boundary6_launcher_shared')
HELPER = load('scripts/mechanics_patient_mesh_boundary6.py', HELPER_SHA,
              'boundary6_helper')
SPEC = SHARED.ExecutionSpec(
    version='resect-case4-patient-mesh-boundary6-v4-release',
    candidate_manifest=HELPER.MANIFEST,
    closure=SHARED.CLOSURE | frozenset([
        'scripts/mechanics_patient_mesh_boundary6.py',
        'scripts/mechanics_patient_mesh_boundary6_run.py', HELPER.MANIFEST]),
    context_roles=SHARED.CONTEXT_ROLES,
    helper_path='scripts/mechanics_patient_mesh_boundary6.py',
    entrypoint='scripts/mechanics_patient_mesh_boundary6_run.py',
    output_bytes=32*1024**2, file_bytes=8*1024**2, output_files=32,
    validate_config=HELPER.validate_config)


if __name__ == '__main__':
    raise SystemExit(SHARED.main(spec=SPEC))
