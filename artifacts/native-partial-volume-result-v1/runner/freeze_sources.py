"""Metadata-only source/reference freeze after root promotes focused-tested APIs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',action='store_true');a=p.parse_args()
    paths=list(HERE.glob('*.py'))+[HERE/'config.json']
    paths+=list((ROOT/'src/resectionlab').glob('*.py'))
    paths += [ROOT/r for r in ('build/native-cut-union-v1/probe_union.py','build/native-cut-union-v1/declaration.json',
        'build/native-cut-union-v1/run-01/report.json','artifacts/native-generic-discretization-v1/probe.py',
        'artifacts/native-generic-discretization-v1/declaration.json','artifacts/native-generic-discretization-v1/run-01/report.json',
        'scripts/probe_native_discretization.py','build/native-partial-volume-v1/independent-review/plane_volume_evaluator.py',
        'build/goal-conditioned-policy-v1/run_contact_owned.py','build/goal-conditioned-policy-v1/darwin_fast_sampler.py')]
    files={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(paths))}
    result={'schema':'partial_volume_fixed18_source_v1','execution_released':a.release,'files':files,'python':sys.version,
            'scope':'Generated matrix only; no acquired images/checkpoints, no training'}
    raw=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode();out=HERE/'source-index.json'
    with out.open('xb') as stream:stream.write(raw)
    print(hashlib.sha256(raw).hexdigest())


if __name__=='__main__':main()
