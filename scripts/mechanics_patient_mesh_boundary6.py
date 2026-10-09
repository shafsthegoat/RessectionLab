#!/usr/bin/env python3
"""One source-only 6/24 mm Case4 mesh candidate; no patient reader or native call.

This thin version binds the existing reviewed geometry implementation and keeps
all physical gates unchanged. A separate source-bound release is needed before
the retained surface may be decoded or Gmsh may run.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'manifests/experiments/resect-case4-patient-mesh-boundary6-v4.json'
PREVIOUS = 'manifests/experiments/resect-case4-patient-mesh-graded-v2.json'
PREVIOUS_SHA = 'ff27037c4141c5da99266da0c92efd5f6a673f5eb0bba6ac3032e916ba5db54e'
CANDIDATE_SHA = 'e1d45b6371bc64baaf6cacbc6f98926728dcf7699c0dffb00bea67df167edc4c'
EVIDENCE_BINDINGS = [
    {'path': 'artifacts/mechanics/resect-case4-patient-mesh-graded-v2/summary.json',
     'sha256': 'a4262bbd81b327c75aabd8c1a125d37f127ed6135ada4d204486ed943be92a99'},
    {'path': 'artifacts/mechanics/resect-case4-patient-mesh-curvature-v3/summary.json',
     'sha256': 'f27114619be8012c837ed3ad6bf77353231fdb5a97eac97527383f74211162a9'},
    {'path': 'artifacts/mechanics/resect-case4-saved-mesh-diagnostic-v1/summary.json',
     'sha256': '875c25269be2f1a44c91d9744822fb7e448af9f662fdea6b03cc45edd90bab10'},
]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_candidate():
    path = ROOT / 'scripts/mechanics_patient_mesh_candidate.py'
    if sha(path) != CANDIDATE_SHA:
        raise ValueError('Reviewed shared candidate changed')
    spec = importlib.util.spec_from_file_location('boundary6_candidate_shared', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


SHARED = load_candidate()
BASE = SHARED.BASE


def expected_declaration():
    """Permit only the one frozen v2-to-v4 size/count/cap delta."""
    if sha(ROOT / PREVIOUS) != PREVIOUS_SHA:
        raise ValueError('Previous candidate declaration changed')
    value = json.loads((ROOT / PREVIOUS).read_text())
    value.update(
        version='resect-case4-patient-mesh-boundary6-v4',
        candidate_id='case4_boundary6_interior24_curvature_off_single',
        geometry_operations='Unchanged retained source and v2 discrete-surface Distance+Threshold workflow; only requested boundary size and compatible global floor change.',
        output_limits={'maximum_total_bytes': 32*1024**2,
                       'maximum_file_bytes': 8*1024**2, 'maximum_files': 32},
        solver_admission={'authorized': False, 'backend_validated_for_patient': False,
                          'limitation': 'Geometry-only candidate; no patient-size sparse solve, displacement validation, tool force, or clinical admission.'},
        evidence_bindings=EVIDENCE_BINDINGS,
    )
    value['size_profile'].update(
        boundary_size_m=.006,
        meaning='Prospective 6 mm boundary / 24 mm interior request, 2–24 mm transition; no patient accuracy or runtime prediction. Sampled Distance is not the independent surface-error oracle.')
    value['eligibility'] = {'maximum_nodes': 64000, 'maximum_elements': 80000,
        'meaning': 'Geometry-research count ceiling only; no patient-size solver admission or clinical acceptance.'}
    value['diagnostic_limits'].update(maximum_nodes=80000,
                                     maximum_elements=100000, maximum_total_bytes=16*1024**2)
    value['caps'].update(aggregate_seconds=180, process_group_rss_bytes=3*1024**3,
                         numerical_threads=1, maximum_generations=1, retries=0)
    value['execution_release']['required'] = ('Separate independent source review and root release of exact committed ten-file closure, '
        'immutable source/runtime/context bindings, and one fresh source-only geometry attempt; no retry.')
    value['prohibited'] = ['new patient arrays before root release', 'B/V landmarks or during-US',
        'source smoothing/dilation/enlargement/repair',
        'extra candidate generation, automatic retry, or mesh sweep',
        'reuse or acceptance of failed v1/v2/v3 meshes', 'solver calls', 'training or policy use']
    value['references'] += ['https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes']
    return value


def validate_config(config):
    if config != expected_declaration():
        raise ValueError('Only the exact declared boundary-6 candidate is permitted')


def declaration():
    value = json.loads((ROOT / MANIFEST).read_text())
    validate_config(value)
    return value


def generate_one(gmsh, vertices, faces, config, charge):
    validate_config(config)
    return SHARED.generate_one(gmsh, vertices, faces, config, charge)


def assess_candidate(gmsh, vertices, faces, output, *, distance_factory, config=None):
    config = declaration() if config is None else config
    validate_config(config)
    return SHARED.assess_candidate(gmsh, vertices, faces, output,
        distance_factory=distance_factory, config=config, generator=generate_one)
