"""Root-run metadata freeze; does not construct a task or release by default."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0, str(HERE))
from common import HELPER, HELPER_SHA, SAMPLER, SAMPLER_SHA


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--release', action='store_true'); args = parser.parse_args()
    paths = list(HERE.glob('*.py'))+[HERE/'config.json']
    # Bind the small project Python source tree, including lazy local imports;
    # no data files, historical run bodies, or prior numerical matrices.
    paths += list((ROOT/'src/resectionlab').glob('*.py'))
    paths += [ROOT/HELPER, ROOT/SAMPLER]
    files = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
    if files[HELPER] != HELPER_SHA or files[SAMPLER] != SAMPLER_SHA:
        raise ValueError('Reviewed lifecycle/sampler source changed')
    value = {'schema': 'opening-pair-source-v1', 'execution_released': args.release, 'files': files,
             'scope': 'Generated fixture only; no actor/training/patient/source archive', 'python': sys.version}
    raw = (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()
    with (HERE/'source-index.json').open('xb') as stream: stream.write(raw)
    print(hashlib.sha256(raw).hexdigest())


if __name__ == '__main__': main()
