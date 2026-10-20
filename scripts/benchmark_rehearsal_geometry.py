#!/usr/bin/env python3
"""Supplemental exact backend action receipts; Electron owns display timing.

Run separately against a frozen src tree and the current src tree. Inputs are
generated engineering controls and/or an existing source bundle, never patients
invented for a benchmark. No source bundle is modified and no policy is trained.
"""
from pathlib import Path
import argparse
import cProfile
import copy
import hashlib
import json
import platform
import pstats
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path('src'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--case', type=Path, action='append', default=[])
    parser.add_argument('--profile', action='store_true', help='Separate attribution run; never included in primary samples')
    parser.add_argument('--cold-covers', action='store_true', help='New empty grid-cover cache before each preview; simulation/setup remain warm')
    parser.add_argument('--disable-covers', action='store_true', help='Disable retained covers to isolate exact deformation work')
    parser.add_argument('--distinct-poses', action='store_true', help='Vary each requested endpoint by 0.015 mm along physical x; records every action/signature')
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20: parser.error('repeats must be 1..20')
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    import scipy
    from scipy.spatial.transform import Rotation
    from resectionlab.core import array_digest, semantic_digest
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_scene import SurgicalScene, from_case
    from resectionlab.surgical_physics_v2 import default_profile
    from resectionlab.imaging import load_case

    def command(operation, position=None, **extra):
        return {'slot':'left', 'operation':operation, 'duration':.2,
            **({'pose': dict(position=list(position), quaternion=[0., 0., 0., 1.])} if position is not None else {}), **extra}

    def advance(engine, action):
        result = engine.step(action)
        if result['status'] == 'rejected': raise AssertionError(result)
        return result

    def generated(material, dense=False):
        labels = np.zeros((32, 32, 24), np.uint16)
        labels[4:21, 4:21, 8:17] = 1 if dense else 0
        if not dense: labels[6:11, 6:11, 8] = 1
        s = SurgicalScene(labels, np.eye(4), [dict(id=1, name='Generated engineering tissue', material=material, target=material == 'tumor')],
            dict(provenance='generated software control; not acquired anatomy'), dict(tissuePhysicsV2=default_profile()))
        e = SurgicalEngine(s); x = 12. if dense else 8.
        setup = [command('exchange', instrument='brain_retractor' if material == 'soft_tissue' else 'cusa'),
            command('place', [x, x, 4.], parameters=dict(radiusMm=1.5, jawOpeningMm=0.))]
        if material == 'soft_tissue': setup += [command('hold', [x, x, 7.])]
        for action in setup: advance(e, action)
        measured = command('move', [x + .4, x, 7.], duration=.4) if material == 'soft_tissue' else command('apply', [x, x, 7.])
        return e, setup, measured

    def patient(path):
        case = load_case(path); s = from_case(case); labels = s.labels
        # Choose the first source cell along a declared affine axis, rather than
        # borrowing action coordinates from another source/version.
        present = np.any(labels != 0, axis=2)
        first = np.argmax(labels != 0, axis=2)
        xs, ys = np.nonzero(present)
        material_by_id = {r['id']: r['material'] for r in s.regions}
        candidates = [(int(x), int(y), int(first[x, y])) for x, y in zip(xs, ys)
            if material_by_id[int(labels[x, y, first[x, y]])] == 'bone']
        if not candidates: raise ValueError('Source has no exposed bone along affine axis2; no invented target')
        index = min(candidates, key=lambda c:(c[2], abs(c[0] - s.shape[0]/2) + abs(c[1] - s.shape[1]/2)))
        center = s.points([np.ravel_multi_index(index, s.shape)])[0]
        axis = s.affine[:3, 2] / np.linalg.norm(s.affine[:3, 2])
        quat = Rotation.align_vectors([axis], [[0., 0., 1.]])[0].as_quat().tolist()
        setup = [command('place', center - 3 * axis, parameters=dict(radiusMm=1., rpm=1000., forceN=.1))]
        setup[0]['pose']['quaternion'] = quat
        measured = command('apply', center - .5 * axis, duration=.2)
        measured['pose']['quaternion'] = quat
        e = SurgicalEngine(s)
        for action in setup: advance(e, action)
        return e, setup, measured

    def measure(name, engine, setup, action, source):
        samples = dict(previewMs=[], commitAfterPreviewMs=[], previewAndCommitMs=[])
        signatures = []
        measured_actions = []
        for index in range(args.repeats):
            measured_action = copy.deepcopy(action)
            if args.distinct_poses:
                measured_action['pose']['position'][0] += index * .015
            measured_actions.append(measured_action)
            cache = getattr(engine, '_cell_cover_cache', None)
            if cache is not None and (args.cold_covers or args.disable_covers):
                engine._cell_cover_cache = type(cache)(engine.grid, engine.scene.identity,
                    **(dict(max_bytes=0, max_entries=0) if args.disable_covers else {}))
            started = time.perf_counter(); preview = engine.preview(measured_action); middle = time.perf_counter()
            candidate = engine.clone(); outcome = advance(candidate, measured_action); ended = time.perf_counter()
            if preview['stateId'] != outcome['stateId']: raise AssertionError('Preview differs from commit')
            signatures.append(semantic_digest(dict(state=candidate.state, history=candidate.history, outcome=outcome)))
            samples['previewMs'].append((middle - started) * 1000)
            samples['commitAfterPreviewMs'].append((ended - middle) * 1000)
            samples['previewAndCommitMs'].append((ended - started) * 1000)
        if not args.distinct_poses and len(set(signatures)) != 1: raise AssertionError('Deterministic complete state/history differs')
        row = dict(name=name, source=source, sceneId=engine.scene.identity, labelHash=array_digest(engine.scene.labels),
            shape=engine.scene.shape, setupActions=setup, measuredAction=action, initialStateId=engine.state_id,
            activeDeformationNodes=len(engine.state.get('deformationModel', {}).get('active', [])),
            resultSignature=semantic_digest(signatures) if args.distinct_poses else signatures[0], resultSignatures=signatures,
            measuredActions=measured_actions, status=outcome['status'], reason=outcome.get('reason'),
            outputStateId=outcome['stateId'], outcome=outcome, samples=samples,
            timings={k:dict(firstMs=v[0], warmMedianMs=float(np.median(v[1:] or v)),
                medianMs=float(np.median(v)), p95Ms=float(np.percentile(v, 95))) for k, v in samples.items()})
        cache = getattr(engine, '_cell_cover_cache', None)
        if cache is not None: row['coverCache'] = cache.stats()
        if args.profile:
            p = cProfile.Profile(); p.runcall(lambda: engine.preview(action))
            stats = pstats.Stats(p)
            row['separateProfile'] = [dict(file=Path(key[0]).name, line=key[1], function=key[2], calls=value[1],
                selfSeconds=value[2], cumulativeSeconds=value[3]) for key, value in
                sorted(stats.stats.items(), key=lambda item:item[1][3], reverse=True)[:35]]
            stages = {}
            for key, value in stats.stats.items():
                filename, _, function = key
                if Path(filename).name in ('geometry.py', 'surgical_geometry_cache.py', 'surgical_motion.py'):
                    stage = 'geometryAndCover'
                elif Path(filename).name == 'surgical_deformation_v2.py' or 'SuperLU' in function:
                    stage = 'deformationIncludingSolver'
                elif Path(filename).name == 'copy.py': stage = 'stateCopy'
                elif Path(filename).name in ('core.py', 'encoder.py', 'decoder.py') or 'sha256' in function:
                    stage = 'hashingJsonAndValueValidation'
                elif Path(filename).name in ('surgical_vessels_v2.py', 'surgical_drilling_v2.py', 'surgical_effects.py'):
                    stage = 'otherPersistentPhysics'
                else: stage = 'engineAndOtherLibraries'
                stages[stage] = stages.get(stage, 0.) + value[2] * 1000
            row['separateProfileExclusiveStageMs'] = stages
        return row

    report = dict(schema='resectionlab.rehearsal-geometry-backend-benchmark.v1', sourceRoot=str(args.source_root.resolve()),
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
        definitions=dict(scope='Public SurgicalEngine preview + cloned commit; excludes Electron, bridge, transport and display',
            first='First measured operation after untimed source installation and setup, not process startup',
            warm='Subsequent identical operations from same authoritative state; geometry may be resident',
            parity='Full state/history/outcome semantic digest, separately compared against frozen source implementation'),
        conditions=dict(coldCovers=args.cold_covers, coversDisabled=args.disable_covers, distinctPoses=args.distinct_poses),
        acceptance=dict(minPreviewCommitMedianSpeedup=1.2, minHeldActionMedianSpeedup=1.15,
            maxUncachedActionRegressionRatio=1.1, exactResultSignatures=True), cases=[])
    for name, material, dense in [('generated-tumor-contact', 'tumor', False), ('generated-local-deformation', 'soft_tissue', True)]:
        e, setup, action = generated(material, dense)
        report['cases'].append(measure(name, e, setup, action, dict(kind='generated software control; not acquired anatomy')))
    for path in args.case:
        e, setup, action = patient(path)
        report['cases'].append(measure(path.stem, e, setup, action,
            dict(kind='existing acquired source with declared anatomy estimates', path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(args.output)


if __name__ == '__main__': main()
