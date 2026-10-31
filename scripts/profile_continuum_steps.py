#!/usr/bin/env python3
"""Isolate old/new source-sized step kernels; no complete-program admission."""
from pathlib import Path
import argparse
import cProfile
import hashlib
import io
import json
import os
import pstats
import resource
import subprocess
import sys
import time
import types

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from resectionlab.surgical_continuum_exterior import ExteriorOperator, ExteriorHistorySolve, source_stiffness
from profile_continuum_exterior import load_geometry, free_partition, bound, sha, write
from qualify_continuum_exterior import contact_matrix
from febio_runtime import supervise

PROTOCOL = ROOT/'manifests/experiments/continuum-step-profile-v1.json'
SOURCES = [Path(__file__), *[ROOT/'scripts'/name for name in
    ('profile_continuum_exterior.py', 'qualify_continuum_exterior.py', 'febio_runtime.py')],
    *[ROOT/'src/resectionlab'/name for name in
       ('surgical_continuum_exterior.py', 'surgical_continuum_partition.py', 'surgical_continuum.py')]]


def legacy_source(protocol):
    entry = protocol['legacy']
    data = subprocess.check_output(['git', 'show', entry['commit']+':'+entry['path']], cwd=ROOT)
    if hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError('Frozen legacy Git blob changed')
    witness = json.loads(bound(protocol['priorWitnessReport']).read_text())
    if witness['sourceHashes'][entry['path']] != entry['sha256'] or witness['status'] != 'completed':
        raise ValueError('Legacy kernel is not the completed source witness')
    module = types.ModuleType('resectionlab._profile_legacy_exterior')
    module.__package__ = 'resectionlab'
    exec(compile(data, entry['commit']+':'+entry['path'], 'exec'), module.__dict__)
    return module.ExteriorHistorySolve


def fingerprint(result):
    digest = hashlib.sha256()
    for key, value in sorted(result.items()):
        array = np.asarray(value)
        digest.update(key.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def compare(left, right, limits):
    if left.keys() != right.keys():
        raise ValueError('Kernel return fields differ')
    errors = {}
    for key in left:
        error = float(np.max(np.abs(np.asarray(left[key])-right[key]), initial=0.))
        category = ('displacementMm' if key == 'displacement' else 'memoryMm' if key == 'memories'
                    else 'forceN' if key in ('contactReactionN', 'maxFreeResidualN') else 'energyNmm')
        if not np.isfinite(error) or error > limits[category]:
            raise ValueError('Old/new step disagreement: '+key)
        errors[key] = error
    return errors


def worker(output, launch):
    output.mkdir(parents=True, exist_ok=False)
    p = json.loads(PROTOCOL.read_text()); identity = json.loads(launch.read_text())
    started = time.perf_counter()
    report = {'schema': 'resectionlab.continuum-step-profile-result.v1', 'status': 'started',
        'phase': 'admission', 'sourceCommit': identity['commit'], 'sourceHashes': identity['sourceHashes'],
        'protocolSha256': sha(PROTOCOL), 'sourceRole': p['sourceRole'],
        'legacy': p['legacy'], 'physicalValidationPassed': False, 'physiologicalRL': 'hold',
        'largeScaleRL': 'hold', 'completeProgramAdmitted': False, 'newAcquiredDatasetBytes': 0,
        'scope': p['scope'], 'trainingUpdates': 0}
    def check():
        if time.perf_counter()-started > p['caps']['wallSeconds']-10:
            raise TimeoutError('Step profiling cooperative wall cap')
    def check_sources():
        if identity['protocolSha256'] != sha(PROTOCOL): raise ValueError('Launch protocol changed')
        for path, digest in identity['sourceHashes'].items():
            if sha(ROOT/path) != digest: raise ValueError('Launch source changed: '+path)
        for entry in [*p['geometry'].values(), p['partition'], p['sample'], p['priorWitnessReport']]: bound(entry)
    try:
        check_sources(); legacy_class = legacy_source(p)
        geometry, summary, partition = load_geometry(p); free, core = free_partition(geometry, partition)
        with np.load(bound(p['geometry']['reference']), allow_pickle=False) as raw:
            alpha, times, spring, cells = (raw[key] for key in ('alpha', 'times', 'spring', 'contactCells'))
        c = contact_matrix(geometry, cells)[:, free]
        with np.load(bound(p['sample']), allow_pickle=False) as raw:
            sample = {key: raw[key] for key in raw.files}
        if not np.array_equal(sample['freeIndices'], free): raise ValueError('Frozen sample free-DOF map differs')
        before = sample['beforeDisplacement']; memories = sample['beforeMemories']; body = np.zeros(len(free))
        target = sample['target']; previous = sample['previousTarget']; h = float(sample['h'])
        report.update(partition=summary, phase='assembly'); write(output/'report.json', report)
        begin = time.perf_counter(); full = source_stiffness(geometry, check=check)
        stiffness = full[free][:, free].tocsr(); del full
        report['assemblySeconds'] = time.perf_counter()-begin
        report['phase'] = 'shared-factor-construction'; write(output/'report.json', report)
        operator = ExteriorOperator(stiffness, core, check=check); del stiffness
        current = ExteriorHistorySolve(operator, c, spring, alpha, times, check=check)
        report.update(operator=dict(operator.stats), phase='fixed-step-kernels'); write(output/'report.json', report)
        # Both kernels borrow precisely the same admitted tissue/Schur/contact
        # operators. This isolates the step change; it does not compare setup.
        legacy = object.__new__(legacy_class)
        for name in ('operator', 'alpha', 'times', 'c', 'spring', 'ci', 'schur', 'core_factor', 'basis', 'compliance'):
            setattr(legacy, name, getattr(current, name))
        def call(solver):
            return solver.step(before, memories, body, target, h, previous_target=previous)
        input_hash = fingerprint(dict(before=before, memories=memories, body=body, target=target, previous=previous))
        baseline = call(legacy); updated = call(current); check()
        errors = compare(baseline, updated, p['tolerances'])
        saved_errors = {key: float(np.max(np.abs(updated[key]-sample[key]), initial=0.))
                        for key in ('displacement', 'memories')}
        if (saved_errors['displacement'] > p['tolerances']['displacementMm']
                or saved_errors['memories'] > p['tolerances']['memoryMm']):
            raise ValueError('Updated kernel disagrees with previously qualified source state')
        solvers = {'legacy': legacy, 'current': current}; first = {'legacy': baseline, 'current': updated}
        timing = {name: [] for name in solvers}; hashes = {name: fingerprint(value) for name,value in first.items()}
        # Repeated calls start from the same immutable frozen state. Alternate
        # order to expose timing bias, and exclude comparison/fingerprinting.
        for index in range(p['repetitions']):
            for name in (('legacy', 'current') if index % 2 == 0 else ('current', 'legacy')):
                check(); begin = time.perf_counter(); value = call(solvers[name])
                timing[name].append(time.perf_counter()-begin)
                if fingerprint(value) != hashes[name]: raise ValueError('Repeated fixed-state step is not byte deterministic')
                compare(first[name], value, p['tolerances'])
        for name, solver in solvers.items():
            check(); profiler = cProfile.Profile(); profiler.enable(); value = call(solver); profiler.disable()
            compare(first[name], value, p['tolerances']); profiler.dump_stats(output/(name+'.prof'))
            stream = io.StringIO(); pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(30)
            (output/(name+'-profile.txt')).write_text(stream.getvalue())
        begin = time.perf_counter(); released = current.rebind(sparse.csr_matrix((0,len(free))), [], check=check)
        release_seconds = time.perf_counter()-begin
        begin = time.perf_counter(); rebound = released.rebind(c, spring, check=check)
        rebind_seconds = time.perf_counter()-begin
        if released.prepared is not current.prepared or rebound.prepared is not current.prepared:
            raise ValueError('Contact change rebuilt tissue preparation')
        compare(updated, call(rebound), p['tolerances'])
        if fingerprint(dict(before=before, memories=memories, body=body, target=target, previous=previous)) != input_hash:
            raise ValueError('Profiling mutated episode input')
        check_sources(); legacy_source(p); check()
        medians = {name: float(np.median(values)) for name,values in timing.items()}
        report.update(status='completed', phase='completed', timingSeconds=timing, medianStepSeconds=medians,
            medianSpeedup=medians['legacy']/medians['current'], maximumOldNewDisagreements=errors,
            frozenSampleDisagreements=saved_errors, repeatedStatesByteExact=True, resultFingerprints=hashes,
            sharedPreparedFactorAcrossRebind=True, releaseBindingSeconds=release_seconds,
            heldBindingSeconds=rebind_seconds, inputFieldsUnchanged=True, sourceInputsUnchanged=True,
            suffixFixedSteps=p['suffixFixedSteps'],
            kernelOnlyExtrapolatedSeconds={name: value*p['suffixFixedSteps'] for name,value in medians.items()},
            extrapolationExcludes=['source topology/Schur reconstruction', 'source ownership/history transfer',
                'source-cell gradient/probe metrics', 'detailed controls', 'independent validation', 'public action/collision admission'])
    except BaseException as error:
        report.update(status='invalid_or_incomplete', error=type(error).__name__+': '+str(error)); raise
    finally:
        report.update(wallSeconds=time.perf_counter()-started,
            peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        write(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('output', type=Path)
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--launch', type=Path)
    args = parser.parse_args(); output = args.output.resolve()
    if not output.is_relative_to(ROOT/'testcases'): raise ValueError('Evidence stays under testcases/')
    if args.worker:
        if args.launch is None: raise ValueError('Supervised launch required')
        worker(output, args.launch); return
    if output.exists(): raise ValueError('Use a fresh output; no automatic retry')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Commit prospective source/profile declaration first')
    launch = output.with_name(output.name+'-launch.json')
    if launch.exists(): raise ValueError('Launch already exists')
    write(launch, {'protocolSha256':sha(PROTOCOL), 'sourceHashes':{str(f.relative_to(ROOT)):sha(f) for f in SOURCES},
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    p = json.loads(PROTOCOL.read_text()); env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    receipt = supervise([sys.executable,str(Path(__file__)),'--worker','--launch',str(launch),str(output/'result')],
        output,cwd=ROOT,environment=env,seconds=p['caps']['wallSeconds'],rss_bytes=p['caps']['rssBytes'])
    valid = receipt['status']=='completed'
    if valid:
        result = json.loads((output/'result/report.json').read_text()); valid = result['status']=='completed'
        if sum(f.stat().st_size for f in output.rglob('*') if f.is_file()) > p['caps']['outputBytes']: valid=False
    write(output/'acceptance.json', {'stepProfileComplete':valid,'completeProgramAdmitted':False,
        'physicalValidationPassed':False,'physiologicalRL':'hold','supervisionStatus':receipt['status']})
    if not valid: raise RuntimeError('Source step profile failed/incomplete; evidence retained')
    print(json.dumps({key:result[key] for key in ('wallSeconds','medianStepSeconds','medianSpeedup','heldBindingSeconds')}))


if __name__ == '__main__': main()
