#!/usr/bin/env python3
"""Supervised setup pilot: assemble/factor and time one exterior basis batch."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from resectionlab.surgical_continuum_partition import GEOMETRY_KEYS, source_partition
from resectionlab.surgical_continuum_exterior import ExteriorOperator, source_stiffness
from febio_runtime import supervise

PROTOCOL = ROOT/'manifests/experiments/continuum-exterior-pilot-v1.json'
SOURCES = [Path(__file__), *[ROOT/'src/resectionlab'/name for name in
    ('surgical_continuum_exterior.py', 'surgical_continuum_partition.py', 'surgical_continuum.py')]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def bound(entry):
    path = ROOT/entry['path']
    if sha(path) != entry['sha256']:
        raise ValueError('Frozen input changed: '+entry['path'])
    return path


def load_geometry(protocol):
    data = {}
    for name, entry in protocol['geometry'].items():
        keys = GEOMETRY_KEYS+(('muPa', 'nu') if name == 'reference' else ())
        with np.load(bound(entry), allow_pickle=False) as archive:
            data[name] = {key: archive[key] for key in keys}
    summary, arrays = source_partition(data['core'], data['reference'])
    with np.load(bound(protocol['partition']), allow_pickle=False) as prior:
        if set(prior.files) != set(arrays) or any(not np.array_equal(prior[k], v) for k,v in arrays.items()):
            raise ValueError('Frozen source partition changed')
    return data['reference'], summary, arrays


def free_partition(geometry, partition):
    free = np.flatnonzero(geometry['freeDofs'])
    full_core = (3*partition['coreNodesInReference'][:,None]+np.arange(3)).ravel()
    selected = full_core[geometry['freeDofs'][full_core]]
    core = np.searchsorted(free, selected)
    if not np.array_equal(free[core], selected):
        raise ValueError('Induced core free DOFs differ from reference supports')
    return free, core


def worker(output, launch):
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text()); identity = json.loads(launch.read_text())
    started = time.perf_counter(); report = {
        'schema': 'resectionlab.continuum-exterior-pilot-result.v1', 'status': 'started',
        'phase': 'admission', 'sourceCommit': identity['commit'], 'sourceHashes': identity['sourceHashes'],
        'protocolSha256': sha(PROTOCOL), 'sourceRole': protocol['sourceRole'],
        'numpyVersion': np.__version__, 'scipyVersion': scipy.__version__,
        'physicalValidationPassed': False, 'physiologicalRL': 'hold', 'trainingUpdates': 0,
        'mechanicalLoadHistoryComputed': False, 'coreSupportCertified': False,
        'largeDomainPhysicallyAccepted': False, 'newAcquiredDatasetBytes': 0}
    def check():
        if time.perf_counter()-started > protocol['caps']['wallSeconds']-5:
            raise TimeoutError('Exterior setup pilot cooperative wall cap')
    try:
        if identity['protocolSha256'] != sha(PROTOCOL): raise ValueError('Launch protocol changed')
        for path, digest in identity['sourceHashes'].items():
            if sha(ROOT/path) != digest: raise ValueError('Launch source changed')
        geometry, summary, partition = load_geometry(protocol)
        report.update(partition=summary, phase='assembly'); write(output/'report.json', report)
        begin = time.perf_counter(); full = source_stiffness(geometry, check=check)
        free, core = free_partition(geometry, partition)
        stiffness = full[free][:,free].tocsr(); del full
        report.update(assemblyAndPartitionSeconds=time.perf_counter()-begin,
            stiffnessNnz=stiffness.nnz, phase='exterior-factor'); write(output/'report.json', report)
        operator = ExteriorOperator(stiffness, core, check=check); del stiffness
        report.update(operator=operator.stats, phase='single-batch'); write(output/'report.json', report)
        columns = operator.interface[:protocol['caps']['batchColumns']]
        if not len(columns): raise ValueError('Expected source interface absent')
        begin = time.perf_counter(); values, residual = operator.columns(columns); check()
        seconds = time.perf_counter()-begin
        batches = (len(operator.interface)+len(columns)-1)//len(columns)
        dense_bytes = len(core)**2*8
        report.update(status='completed', phase='completed', exteriorSupportCertified=True,
            sampledColumns=len(columns), sampledBatchSeconds=seconds, sampledBatchResidualN=residual,
            sampledSchurColumnBytes=values.nbytes, predictedBatches=batches,
            extrapolatedAllBatchSeconds=seconds*batches,
            denseCoreSchurBytes=dense_bytes, denseCoreSchurAndCholeskyBytes=2*dense_bytes,
            unmeasuredCosts=['full Schur build/positive-definite admission', 'core factorization/contact basis',
                             'history advancement, field/energy checks and independent verification'],
            arrayEstimateExcludes='Sparse factors/certificate intermediates and numerical workspaces')
        for path, digest in identity['sourceHashes'].items():
            if sha(ROOT/path) != digest: raise ValueError('Source changed during pilot')
    except BaseException as error:
        report.update(status='invalid_or_incomplete', error=type(error).__name__+': '+str(error)); raise
    finally:
        report.update(wallSeconds=time.perf_counter()-started,
            peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024))
        write(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path); parser.add_argument('--worker', action='store_true')
    parser.add_argument('--launch', type=Path); args = parser.parse_args(); output = args.output.resolve()
    if not output.is_relative_to(ROOT/'testcases'): raise ValueError('Evidence stays under testcases/')
    if args.worker:
        if args.launch is None: raise ValueError('Supervised launch required')
        worker(output, args.launch); return
    if output.exists(): raise ValueError('Use a new output path; no automatic retry')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Commit prospective source and pilot declaration first')
    launch = output.with_name(output.name+'-launch.json')
    if launch.exists(): raise ValueError('Launch already exists')
    write(launch, {'protocolSha256': sha(PROTOCOL), 'sourceHashes': {str(p.relative_to(ROOT)): sha(p) for p in SOURCES},
                  'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    protocol = json.loads(PROTOCOL.read_text()); env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    receipt = supervise([sys.executable, str(Path(__file__)), '--worker', '--launch', str(launch), str(output/'result')],
        output, cwd=ROOT, environment=env, seconds=protocol['caps']['wallSeconds'], rss_bytes=protocol['caps']['rssBytes'])
    valid = receipt['status'] == 'completed'
    if valid:
        result = json.loads((output/'result/report.json').read_text()); valid = result['status'] == 'completed'
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > protocol['caps']['outputBytes']: valid = False
    write(output/'acceptance.json', {'setupPilotComplete': valid, 'physicalValidationPassed': False,
        'physiologicalRL': 'hold', 'supervisionStatus': receipt['status']})
    if not valid: raise RuntimeError('Exterior pilot failed/incomplete; evidence retained')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
