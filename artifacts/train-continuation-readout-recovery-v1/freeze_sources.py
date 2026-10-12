"""Root-only source/metadata freeze; no weights, arrays or tasks are loaded."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ROOT, HELPER, HELPER_SHA, SAMPLER, SAMPLER_SHA, CHECKPOINTS, read_json, sha, canonical, runtime_roots, CACHE_SOURCE_INDEX,CACHE_SOURCE_SHA
from inputs import metadata_pins


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--release', action='store_true'); args=parser.parse_args()
    paths = list(HERE.glob('*.py'))+[HERE/'config.json']+list((ROOT/'src/resectionlab').glob('*.py'))
    paths += [ROOT/HELPER, ROOT/SAMPLER]
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    if files[HELPER]!=HELPER_SHA or files[SAMPLER]!=SAMPLER_SHA: raise ValueError('Owned helper pin changed')
    original=read_json(ROOT/CACHE_SOURCE_INDEX,CACHE_SOURCE_SHA)
    for path,digest in original['files'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Original cache-producing source changed')
        if path in files and files[path]!=digest:raise ValueError('Conflicting original source binding')
        files[path]=digest
    metadata, arrays = metadata_pins()
    value={'schema':'fixed-four-TRAIN-cache-recovery-source-v1','execution_released':args.release,'files':files,
        'input_metadata':metadata,'public_arrays':arrays,'checkpoints':{'IL64':CHECKPOINTS['IL64']},'runtime_roots':runtime_roots(),
        'scope':'same four TRAIN, no new training trajectories or held-out roles; source/world admission and new head objective separate'}
    raw=canonical(value)+b'\n'
    with (HERE/'source-index.json').open('xb') as f:f.write(raw)
    print(hashlib.sha256(raw).hexdigest())


if __name__=='__main__':main()
