#!/usr/bin/env python3
"""Export paired generated approach controls through the native procedure policy."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy.spatial.transform import Rotation

from resectionlab.core import CaseData, SourceRef, array_digest, semantic_digest, thaw_json
from resectionlab.imaging import load_case, read_case_artifacts, save_case
from resectionlab.native_backend import backend_identity
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_procedure_policy import PROPERTY, evaluate_procedure_sequence, make_procedure_policy
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_sequential_task import SequentialPrimitiveAdapter


ROOT = Path(__file__).resolve().parents[1]
SUPERIOR_BRANCH = 'superior-treatment-and-exit'
INFERIOR_BRANCH = 'inferior-posterior-attempt'


def generated_source():
    labels = np.zeros((24, 24, 56), np.uint16)
    affine = np.diag([3., 3., 3., 1.])
    labels[13, 15, 28:30] = labels[9, 15, 28:30] = 1
    labels[12, 13, 26] = 2
    labels[13, 13, 26] = 3
    regions = [dict(id=1, name='Generated supratentorial target', material='tumor', target=True, color='#da6ea7'),
               dict(id=2, name='Generated brainstem', material='soft_tissue', color='#d2ab56'),
               dict(id=3, name='Generated cerebellum', material='soft_tissue', color='#ca7157')]
    case = CaseData(case_id='Generated approach controls', mri=labels.astype(np.float32), affine=affine,
                    compartments={r['name']: labels == r['id'] for r in regions},
                    source_refs=(SourceRef('generated-approach-controls', 'synthetic://approach-procedure/v1',
                                           sha256=array_digest(labels), provenance='simulated'),),
                    metadata={'generated': True, 'is_synthetic': True,
                              'surgical_adapter': {'type': 'generated-engineering-control'},
                              'surgical_properties': {},
                              'anatomy_layers': {r['name']: {'kind': r['material'], 'color': r['color']}
                                                 for r in regions}})
    base = from_case(case, regions)

    def flat(ijk):
        return int(np.ravel_multi_index(ijk, labels.shape))

    intracranial = [flat((x, y, z)) for x in range(7, 18) for y in range(12, 19) for z in range(26, 34)]
    work = [flat((x, y, z)) for x in range(11, 16) for y in range(14, 17) for z in range(28, 34)]
    entry = [flat((x, y, 33)) for x in range(11, 16) for y in range(14, 17)]
    critical = {'brainstem': [flat((12, 13, 26))], 'cerebellum': [flat((13, 13, 26))]}
    evidence = dict(origin='independent-source-declaration', reviewStatus='generated-engineering',
                    provenance='Declared stylized source compartments, boundary and critical memberships.',
                    evidenceSha256=semantic_digest([array_digest(labels), intracranial, work, entry, critical])[7:])
    structures = {name: dict(evidence, kind='neural', cells=cells, coverageCells=intracranial)
                  for name, cells in critical.items()}
    policy = make_procedure_policy(
        base, policy_id='generated-approach-comparison-v1', evidence=evidence, structures=structures,
        intracranial_cells=intracranial, observed_cells=range(labels.size),
        entry_regions={'superior-post-access-entry': dict(evidence, cells=entry)},
        permitted_compartments={'declared-superior-work': dict(evidence, cells=work)},
        work_domain=dict(evidence, minimumVoxelBoundary=[-.5] * 3,
                         maximumVoxelBoundary=(np.asarray(labels.shape) - .5).tolist()),
        tissue_permissions={'tumor': {'remove': True, 'cut': False},
                            'soft_tissue': {'remove': False, 'cut': False}},
        neural_max_mm3=0., bone_max_mm3=0., preparation_nonbone_max_mm3=0., modeled_time_max_seconds=30.,
        structure_max_mm3={'brainstem': 0., 'cerebellum': 0.})
    metadata = thaw_json(case.metadata)
    metadata['surgical_properties'] = {PROPERTY: policy}
    case = replace(case, metadata=metadata)
    return case, from_case(case, regions)


def _pose(position, axis):
    rotation, _ = Rotation.align_vectors([axis], [[0., 0., 1.]])
    return dict(position=list(position), quaternion=rotation.as_quat().tolist())


def _programs():
    target = [39., 45., 84.]
    outer = _pose([39., 45., 109.], [0., 0., -1.])
    wrong_start = [39., 30., 69.]
    direction = np.asarray(target) - wrong_start
    wrong_axis = direction / np.linalg.norm(direction)
    wrong_outer = _pose(wrong_start, wrong_axis)

    def request(operation, *, slot='right', duration=.1, **fields):
        return dict(operation=operation, slot=slot, duration=duration, **fields)

    support = _pose([21., 45., 109.], [0., 0., -1.])
    prepare = [request('exchange', instrument='cusa', parameters={'radiusMm': .4}),
               request('place', slot='left', pose=support)]
    superior = [*prepare, request('place', pose=outer),
                request('apply', pose=_pose(target, [0., 0., -1.]), duration=1.),
                request('withdraw', pose=outer, duration=1.), request('withdraw', slot='left', pose=support)]
    inferior = [*prepare, request('place', pose=wrong_outer),
                request('apply', pose=_pose(target, wrong_axis), duration=1.)]
    return superior, inferior


def _execute(scene, commands, *, expect_veto=False):
    adapter = SequentialPrimitiveAdapter(scene)
    adapter.reset(seed=0)
    engine = adapter.engine
    attempts = []
    for index, command in enumerate(commands):
        before = engine.snapshot()
        _, _, _, truncated, step = adapter.step_action(command)
        assert not truncated
        outcome = step['outcome']
        if expect_veto and index == len(commands) - 1:
            assert outcome['status'] == 'rejected' and outcome['reasonCode'] == 'PROTECTED_STRUCTURE', outcome
            assert engine.snapshot() == before, 'Refused ingress changed the accepted prefix'
        else:
            assert outcome['status'] == 'accepted', outcome
        attempts.append(dict(action=command, outcome=outcome, stateId=engine.state_id))
    _, _, terminated, truncated, step = adapter.stop('Generated approach demonstration ended')
    assert terminated and not truncated and step['outcome']['status'] == 'accepted', step
    endpoint = adapter.completion_assessment
    assert endpoint['validWithdrawal'] and endpoint['cleanupComplete'], endpoint
    evaluation = evaluate_procedure_sequence(engine)
    assert evaluation['admissible'] and not evaluation['violations']
    progress = endpoint['progress']
    assert (progress['originalTargetTreatmentMm3'] == 0.) == expect_veto
    assert 0 < progress['residualOriginalTargetCells'] <= progress['originalTargetCells'] == 4
    return engine, dict(endpoint=endpoint, wholeToolExit=endpoint['validWithdrawal'], checkedStop=step['outcome'],
                        physicalRequests=len(attempts), attempts=attempts, sequence=evaluation['outcome'])


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def demonstrate(out):
    started = time.monotonic()
    case, scene = generated_source()
    superior, inferior = _programs()
    pos, accepted = _execute(scene, superior)
    neg, refused = _execute(scene, inferior, expect_veto=True)
    branches = {SUPERIOR_BRANCH: pos, INFERIOR_BRANCH: neg}
    record = dict(mapping=scene.regions, branches={key: engine.snapshot() for key, engine in branches.items()},
                  branchId=SUPERIOR_BRANCH, position=len(pos.actions), parents={key: None for key in branches},
                  timeline={key: engine.history for key, engine in branches.items()})
    path = out / 'generated-approach-controls.ressectionlab'
    save_case(case, path, artifacts={'surgical_toolbox': record})
    cold_case = load_case(path)
    cold_record = read_case_artifacts(path)['surgical_toolbox']
    cold_scene = from_case(cold_case, cold_record['mapping'])
    cold_checks = []
    for key, program, prior in ((SUPERIOR_BRANCH, superior, accepted), (INFERIOR_BRANCH, inferior, refused)):
        native = SurgicalEngine.restore(cold_scene, cold_record['branches'][key], use_cache=False, force_verify=True)
        replay, proof = _execute(cold_scene, program, expect_veto=key == INFERIOR_BRANCH)
        assert native.snapshot() == replay.snapshot() == record['branches'][key]
        assert native.history == replay.history and proof == prior
        cold_checks.append(dict(branch=key, wholeToolExit=proof['wholeToolExit'],
                                nativeHistoryReplayRequests=len(native.actions), checkedAttemptReplayRequests=len(program),
                                checkedStopRequests=1, stateId=native.state_id))
    with path.open('rb') as stream:
        package_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(schema='resectionlab.approach-procedure-controls.v1',
                  generatedOnly=True, startStage='post-access', caseHash=cold_case.semantic_hash,
                  sourceSceneId=cold_scene.identity, policyHash=pos.snapshot()['procedurePolicy']['policyHash'],
                  sourceImplementationSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  package=str(path), packageSha256=package_hash, accepted=accepted, refused=refused,
                  coldChecks=cold_checks, positiveBranch=SUPERIOR_BRANCH, negativeBranch=INFERIOR_BRANCH,
                  earlierAccessCosts=None, targetBasis='Full generated original target including the excluded residual compartment.',
                  negativeScope='Constructed inferior/posterior trajectory against declared entry and compartment exclusions.',
                  clinicalValidation=False, wallSeconds=time.monotonic() - started,
                  nativeNumerics=backend_identity()['selected'], peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    _write_json(out / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / 'testcases'):
        parser.error('Keep evidence in ignored testcases/')
    out.mkdir(parents=True, exist_ok=False)
    try:
        report = demonstrate(out)
    except Exception as error:
        _write_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    print(json.dumps({k: report[k] for k in ('package', 'wallSeconds', 'peakRssBytes')}))


if __name__ == '__main__':
    main()
