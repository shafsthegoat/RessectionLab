#!/usr/bin/env python3
"""Prepare a fixed-source exterior partition from already exposed geometry only."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.surgical_continuum_partition import GEOMETRY_KEYS, source_partition


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output, manifest):
    started = time.perf_counter()
    if not output.resolve().is_relative_to(ROOT / 'testcases'):
        raise ValueError('Evidence stays under testcases/')
    if output.exists():
        raise ValueError('Use a new output directory')
    protocol = json.loads(manifest.read_text())
    data = {}
    for name, entry in protocol['geometry'].items():
        path = ROOT / entry['path']
        if sha(path) != entry['sha256']:
            raise ValueError('Frozen geometry changed: ' + name)
        with np.load(path, allow_pickle=False) as archive:
            data[name] = {key: archive[key] for key in GEOMETRY_KEYS}
    summary, arrays = source_partition(data['core'], data['reference'])
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'source-partition.npz', **arrays)
    summary.update(schema=protocol['schema'], sourceRole=protocol['sourceRole'],
        protocolSha256=sha(manifest), sourceArchiveSha256=protocol['sourceArchiveSha256'],
        geometry=protocol['geometry'], partitionSha256=sha(output / 'source-partition.npz'),
        codeSha256={str(path.relative_to(ROOT)): sha(path) for path in
                    (Path(__file__), ROOT / 'src/resectionlab/surgical_continuum_partition.py')},
        wallSeconds=time.perf_counter() - started)
    (output / 'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'manifests/experiments/continuum-exterior-preparation-v1.json')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output.resolve(), args.manifest.resolve()), indent=2))
