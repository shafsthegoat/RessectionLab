"""Root-only source/metadata freeze; no weights, arrays or tasks are loaded."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ROOT, HELPER, HELPER_SHA, SAMPLER, SAMPLER_SHA, CHECKPOINTS, read_json, sha, canonical, runtime_roots, CACHE_SOURCE_INDEX,CACHE_SOURCE_SHA, PLANNER_SOURCE_SHA
from recovery_inputs import metadata_pins,OLD_INDEX,OLD_INDEX_SHA


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--release', action='store_true'); args=parser.parse_args()
    paths = list(HERE.glob('*.py'))+[HERE/'config.json']+list((ROOT/'src/resectionlab').glob('*.py'))
    paths += [ROOT/HELPER, ROOT/SAMPLER]
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    if files[HELPER]!=HELPER_SHA or files[SAMPLER]!=SAMPLER_SHA: raise ValueError('Owned helper pin changed')
    old=read_json(ROOT/OLD_INDEX,OLD_INDEX_SHA)
    for path in files:
        if path.startswith('src/resectionlab/') and files[path]!=old['files'].get(path):
            raise ValueError('Original replay source changed: '+path)
    metadata, arrays = metadata_pins()
    value={'schema':'sealed-endgame-replay-source-v1','execution_released':args.release,'files':files,
        'input_metadata':metadata,'public_arrays':arrays,'checkpoints':{},'runtime_roots':runtime_roots(),
        'scope':'same045 TRAIN source and both sealed plans only; no policy/cache/checkpoint payload, fitting, scoring or replanning'}
    raw=canonical(value)+b'\n'
    with (HERE/'source-index.json').open('xb') as f:f.write(raw)
    print(hashlib.sha256(raw).hexdigest())


if __name__=='__main__':main()
