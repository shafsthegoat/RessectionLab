#!/usr/bin/env python3
"""One declared curvature-aware candidate; no image reader or native initialization."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'manifests/experiments/resect-case4-patient-mesh-curvature-v3.json'
PREVIOUS = 'manifests/experiments/resect-case4-patient-mesh-graded-v2.json'
PREVIOUS_SHA = 'ff27037c4141c5da99266da0c92efd5f6a673f5eb0bba6ac3032e916ba5db54e'
CANDIDATE_SHA = 'e1d45b6371bc64baaf6cacbc6f98926728dcf7699c0dffb00bea67df167edc4c'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_candidate():
    path = ROOT/'scripts/mechanics_patient_mesh_candidate.py'
    if sha(path) != CANDIDATE_SHA:
        raise ValueError('Reviewed shared candidate changed')
    spec = importlib.util.spec_from_file_location('curvature_candidate_shared', path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


SHARED = load_candidate()
BASE = SHARED.BASE


def expected_declaration():
    """Exact prospective delta; frozen previous declaration is never rewritten."""
    if sha(ROOT/PREVIOUS) != PREVIOUS_SHA:
        raise ValueError('Previous candidate declaration changed')
    value = json.loads((ROOT/PREVIOUS).read_text())
    value.update(version='resect-case4-patient-mesh-curvature-v3',
        candidate_id='case4_curvature24_min3_boundary12_interior24_single',
        geometry_operations='Unchanged source and discrete parametrization; curvature sizing and compatible global floor only.',
        global_size_profile={'curvature_elements_per_two_pi':24, 'minimum_m':.003, 'maximum_m':.024,
            'meaning':'Joint profile tested on one analytical ellipsoid; no patient count, fidelity or runtime prediction.'},
        output_limits={'maximum_total_bytes':32*1024**2, 'maximum_file_bytes':8*1024**2, 'maximum_files':32},
        solver_admission={'authorized':False, 'backend_validated_for_patient':False,
            'limitation':'Higher counts are bounded geometry-research eligibility only; no patient solver admission or memory prediction.'})
    value['gmsh_options']['Mesh.MeshSizeFromCurvature'] = 24
    value['eligibility'].update(maximum_nodes=64000, maximum_elements=80000)
    value['diagnostic_limits'].update(maximum_nodes=80000, maximum_elements=100000,
        maximum_total_bytes=16*1024**2)
    value['execution_release']['required'] = 'Separate root release of the exact committed ten-file source closure and one fresh attempt; no retry.'
    value['references'] += ['https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes',
                            'https://gmsh.info/doc/texinfo/gmsh.html#t13']
    value['evidence_bindings'] = EVIDENCE_BINDINGS
    return value


# These compact, frozen receipts contain no new patient data and are rehashed
# by the source-bound launcher before any retained surface is decoded.
EVIDENCE_BINDINGS = [{'path': 'artifacts/mechanics/gmsh-discrete-curvature-capability-v1/declaration.json', 'sha256': '05c356691e809c53f846d23fee5ef78cbc99a90971087a7791117741cc9a4b8e'}, {'path': 'artifacts/mechanics/gmsh-discrete-curvature-capability-v1/summary.json', 'sha256': 'e10e1c5c3ece08ff46de3eecf253fbcd4ac4e909042828ba383ca4fbd12900f0'}, {'path': 'artifacts/mechanics/gmsh-discrete-curvature-capability-independent-review-v1/verification.json', 'sha256': '4706e9dbd581303a393f22ed68846b48e1cd1fde4bf46a0a0d902f9489d640e9'}, {'path': 'artifacts/mechanics/resect-case4-saved-mesh-diagnostic-v1/summary.json', 'sha256': '875c25269be2f1a44c91d9744822fb7e448af9f662fdea6b03cc45edd90bab10'}, {'path': 'artifacts/mechanics/resect-case4-patient-mesh-graded-independent-audit-v2/verification.json', 'sha256': 'faaa14716d937535d29107f1aea7db5be576200efa4da7e0cd820171d29f94b3'}]


def validate_config(config):
    if config != expected_declaration():
        raise ValueError('Only the exact declared curvature candidate is permitted')


def declaration():
    value = json.loads((ROOT/MANIFEST).read_text()); validate_config(value)
    return value


def configure_profile(gmsh, tags, profile):
    # The old field sets a 12 mm global floor. Apply the tested 3 mm floor LAST,
    # leaving the Distance/Threshold field itself and the 24 mm maximum intact.
    SHARED.configure_profile(gmsh, tags, profile)
    gmsh.option.setNumber('Mesh.MeshSizeMin', .003)


def generate_one(gmsh, vertices, faces, config, charge):
    validate_config(config)
    return SHARED.generate_one(gmsh, vertices, faces, config, charge,
                               profile_callback=configure_profile)


def assess_candidate(gmsh, vertices, faces, output, *, distance_factory, config=None):
    config = declaration() if config is None else config
    validate_config(config)
    return SHARED.assess_candidate(gmsh, vertices, faces, output,
        distance_factory=distance_factory, config=config, generator=generate_one)
