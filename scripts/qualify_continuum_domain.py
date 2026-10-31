#!/usr/bin/env python3
"""Non-affine patch-boundary audit on existing acquired-source action prefixes.

Probe endpoints still hold an instrument. They are numerical measurements of a
recorded history, never complete surgical plans or independent observations.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import copy
import gzip
import json
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.core import array_digest, semantic_digest
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_scene import SurgicalScene, from_case
from resectionlab.surgical_environment import standardize
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab import surgical_continuum as continuum
from resectionlab.simulator_qualification import file_sha256, removed_accounting

PROTOCOL = ROOT / 'manifests/experiments/continuum-domain-sensitivity-v1.json'


def relative_difference(a, b):
    if any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 for v in (a, b)):
        raise ValueError('Finite nonnegative magnitudes required')
    return abs(a-b) / max(a, b) if max(a, b) else 0.


def compare_domains(rows, protocol):
    expected = {(b, r) for b in protocol['branches'] for r in protocol['patchRadiiMm']}
    if ({(r['branchId'], r['patchRadiusMm']) for r in rows} != expected or len(rows) != len(expected)):
        raise ValueError('Every declared branch/domain must be retained exactly once')
    outcomes = []
    low, high = protocol['stability']['lastTwoRadiiMm']; limits = protocol['stability']
    for branch in protocol['branches']:
        branch_rows = [r for r in rows if r['branchId'] == branch]
        if any(r['status'] not in ('completed', 'refused') for r in branch_rows):
            raise ValueError('Every declared domain needs a terminal result')
        if any(r['status'] == 'completed' and
               (type(r.get('acceptedActions')) is not int or r['acceptedActions'] != protocol['probeActions']
                or r.get('exactCanonicalReplay') is not True) for r in branch_rows):
            raise ValueError('Completed domain must contain the exact canonically replayed probe')
        a, b = [next(r for r in rows if r['branchId'] == branch and r['patchRadiusMm'] == radius) for radius in (low, high)]
        if any(r['status'] != 'completed' for r in branch_rows):
            outcomes.append({'branchId': branch, 'status': 'not_estimable', 'radiusStable': False,
                'reason': 'A required numerical domain refused the exact recorded probe; no missing world becomes zero.'})
            continue
        if (len({r['fixedProbeCellsHash'] for r in branch_rows}) != 1 or
                len({r['removal']['sourceIdsSha256'] for r in branch_rows}) != 1 or
                len({r['probeActionHash'] for r in branch_rows}) != 1):
            raise ValueError('Domain comparison changed fixed source support, removal or action history')
        u, v = np.asarray(a['probeCenterDisplacementMm']), np.asarray(b['probeCenterDisplacementMm'])
        if u.ndim != 2 or u.shape != v.shape or u.shape[1] != 3 or not len(u) or not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError('Identical finite source-center vector fields required')
        differences = {
            'centerDisplacementRmsDifferenceMm': float(np.sqrt(np.mean(np.sum((u-v)**2, axis=1)))),
            'reactionRelativeDifference': relative_difference(a['reactionNormN'], b['reactionNormN']),
            'fixedProbeStrainSquaredRelativeDifference': relative_difference(a['fixedProbeStrainSquaredMm3'], b['fixedProbeStrainSquaredMm3']),
            'fixedProbeStressSquaredRelativeDifference': relative_difference(a['fixedProbeStressSquaredPa2Mm3'], b['fixedProbeStressSquaredPa2Mm3'])}
        checks = {key: value <= limits['maximum'+key[0].upper()+key[1:]] for key, value in differences.items()}
        outcomes.append({'branchId': branch, 'status': 'scored', 'radiusStable': all(checks.values()),
                         'differences': differences, 'checks': checks})
    return outcomes


def probe(engine, common, path):
    system = engine._continuum_system_v1[1]; mesh = system['mesh']
    indexes = {c: i for i, c in enumerate(mesh.cells)}
    if not set(common) <= set(indexes): raise ValueError('Fixed source probe became unsupported')
    chosen = [indexes[c] for c in common]
    subset = replace(mesh, cells=tuple(common), element_nodes=mesh.element_nodes[chosen])
    state = engine.state[continuum.PROPERTY]; p = engine.scene.properties[continuum.PROPERTY]
    u = continuum._unpack(mesh, state['cornerDisplacementMm'])
    memories = [continuum._unpack(mesh, field) for field in state['memoryDisplacementMm']]
    instantaneous = continuum.exposure_increment(engine.scene, subset, system['gradients'], u, memories, p['materials'], 1.)
    centers = u.reshape((-1, 3))[mesh.element_nodes[chosen]].mean(axis=1)
    reactions = state['handleReactionsN']; reaction = np.sum(list(reactions.values()), axis=0)
    contacts = system['contacts']; targets = []
    for key, c, _, _ in contacts:
        hold = engine.state['holds'][key]
        targets.append(np.subtract(hold['target'], hold['origin']) + np.asarray(hold.get('baseDisplacements', {}).get(str(c), [0., 0., 0.])))
    labels = np.array([int(engine.scene.labels.flat[c]) for c in mesh.cells])
    np.savez_compressed(path, cells=np.asarray(mesh.cells), affine=engine.scene.affine, nodes=mesh.nodes,
        elementNodes=mesh.element_nodes, labels=labels, fixedDofs=~system['free'], displacement=u.reshape((-1, 3)),
        memories=np.asarray(memories).reshape((len(memories), -1, 3)), commonCells=common,
        muPa=np.asarray([p['materials'][str(label)]['shearPa'] for label in labels]),
        nu=np.asarray([p['materials'][str(label)]['poissonRatio'] for label in labels]),
        relaxationMuPa=np.asarray([p['materials'][str(label)]['relaxationShearPa'] for label in labels]),
        contactNodes=np.asarray([nodes for _, _, nodes, _ in contacts]),
        contactK=np.asarray([k for _, _, _, k in contacts]), contactTargets=np.asarray(targets))
    return {'activeSourceCells': len(mesh.cells), 'cornerDofs': len(u), 'artificialFixedDofs': int((~system['free']).sum()),
        'fixedProbeCellsHash': semantic_digest(common), 'fixedProbeCells': len(common),
        'probeCenterDisplacementMm': centers.tolist(), 'reactionN': reaction.tolist(),
        'reactionNormN': float(np.linalg.norm(reaction)),
        'fixedProbeStrainSquaredMm3': sum(v['strainSquaredVolumeTimeMm3Seconds'] for v in instantaneous.values()),
        'fixedProbeStressSquaredPa2Mm3': sum(v['stressSquaredVolumeTimePa2Mm3Seconds'] for v in instantaneous.values()),
        'wholePatchCumulativeExposureBySourceLabel': state['exposureBySourceLabel'],
        'maxGradientNorm': state['maxGradientNorm'], 'maxFreeResidualN': state['maxFreeResidualN'],
        'maxEnergyBalanceResidualNmm': state['maxEnergyBalanceResidualNmm'],
        'rawFields': path.name, 'rawFieldsSha256': file_sha256(path)}


def prepare(output):
    output = output.resolve()
    if not output.is_relative_to(ROOT/'testcases') or output.exists(): raise ValueError('Use a NEW output under testcases/')
    protocol = json.loads(PROTOCOL.read_text()); source = ROOT/protocol['sourceArchive']
    if file_sha256(source) != protocol['sourceArchiveSha256']: raise ValueError('Frozen source differs')
    output.mkdir(parents=True); start = time.perf_counter(); rows = []
    report = {'schema': 'resectionlab.continuum-domain-sensitivity-result.v1', 'protocolSha256': file_sha256(PROTOCOL),
        'sourceArchiveSha256': file_sha256(source), 'sourceRole': protocol['sourceRole'],
        'sourceHashes': {str(p.relative_to(ROOT)): file_sha256(p) for p in [Path(__file__), ROOT/'src/resectionlab/surgical_continuum.py', ROOT/'src/resectionlab/surgical_deformation_v2.py']},
        'completeSurgicalPlans': False, 'physicalValidationPassed': False, 'physiologicalRL': 'hold', 'trainingUpdates': 0}
    def check():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        if time.perf_counter()-start > protocol['caps']['wallSeconds']: raise TimeoutError('Domain audit wall cap exceeded')
        if rss > protocol['caps']['maximumRssBytes']: raise MemoryError('Domain audit RSS cap exceeded')
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > protocol['caps']['outputBytes']:
            raise ValueError('Domain audit storage cap exceeded; evidence retained')
    try:
        case, artifacts, _ = load_workspace(source); record = artifacts['surgical_toolbox']
        original = standardize(from_case(case, record['mapping']), physics_profile=record['physicsProfile'])
        if original.identity != protocol['sourceSceneId']: raise ValueError('Frozen scene differs')
        before_labels = array_digest(original.labels)
        prefix = record['branches']['retraction-start']; common = sorted(prefix['checkpoint']['state']['deformationModel']['active'])
        for radius in protocol['patchRadiiMm']:
            properties = copy.deepcopy(original.properties); properties['tissuePhysicsV2']['brain']['radiusMm'] = radius
            scene = original if radius == 4. else SurgicalScene(original.labels, original.affine, original.regions, original.source, properties)
            for branch in protocol['branches']:
                check(); engine = SurgicalEngine(scene)
                actions = record['branches'][branch]['actions'][:protocol['probeActions']]
                if len(actions) != protocol['probeActions']: raise ValueError('Declared probe action count is unavailable')
                if actions[:protocol['prefixActions']] != prefix['actions']: raise ValueError('Exact common action prefix differs')
                row = {'branchId': branch, 'patchRadiusMm': radius, 'sourceSceneId': scene.identity,
                    'probeActionHash': semantic_digest(actions), 'status': 'started'}
                rows.append(row)
                for index, action in enumerate(actions):
                    before = engine.snapshot(); result = engine.step(action, check=check)
                    if result['status'] == 'rejected':
                        if engine.snapshot() != before: raise ValueError('Refused domain mutated accepted history')
                        row.update(status='refused', refusedAction=index+1, reason=result['reason'], exactRollback=True,
                                   acceptedActions=len(engine.actions), stateId=engine.state_id)
                        break
                    if radius == 4. and index+1 == protocol['prefixActions'] and engine.snapshot() != prefix:
                        raise ValueError('Nominal original common prefix differs')
                else:
                    snapshot = engine.snapshot(); replay = SurgicalEngine.restore(scene, snapshot, check=check)
                    if replay.state != engine.state or replay.actions != engine.actions or replay.history != engine.history:
                        raise ValueError('Independent canonical probe replay differs')
                    stem = f'{branch}-radius-{radius:g}'
                    raw = output/(stem+'.npz'); row.update(probe(engine, common, raw))
                    saved = output/(stem+'.snapshot.json.gz')
                    saved.write_bytes(gzip.compress(json.dumps(snapshot, sort_keys=True, allow_nan=False).encode(), mtime=0))
                    row.update(status='completed', exactCanonicalReplay=True, acceptedActions=len(engine.actions),
                        stateId=engine.state_id, removal=removed_accounting(scene, engine.state),
                        snapshot=saved.name, snapshotSha256=file_sha256(saved))
                (output/'samples.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
                print(json.dumps({'branch': branch, 'radiusMm': radius, 'status': row['status'], 'elapsedWallSeconds': time.perf_counter()-start}), flush=True)
        if array_digest(original.labels) != before_labels or file_sha256(source) != protocol['sourceArchiveSha256']:
            raise ValueError('Source data changed')
        report.update(status='completed', samples=rows, outcomes=compare_domains(rows, protocol), sourceArraysPreserved=True)
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        check()
    except BaseException as error:
        report.update(status='invalid_or_incomplete', samples=rows, error=type(error).__name__+': '+str(error))
        raise
    finally:
        report.update(wallSeconds=time.perf_counter()-start,
            peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'outcomes':report['outcomes'],'wallSeconds':report['wallSeconds']}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('output',type=Path)
    prepare(parser.parse_args().output)
