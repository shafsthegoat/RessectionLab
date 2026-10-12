"""Root-only source/metadata freeze for a single existing-source qualification."""
import argparse,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import paths
from paths import HERE,ROOT,OLD_RUNNER,OLD_MODEL,DEPS
from common import canonical,sha,read_json,runtime_roots,HELPER,HELPER_SHA,SAMPLER,SAMPLER_SHA
from source_inputs import metadata_pins,PRIOR_INDEX,PRIOR_INDEX_SHA

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',action='store_true');args=parser.parse_args()
    prior=read_json(ROOT/PRIOR_INDEX,PRIOR_INDEX_SHA)
    # Preserve pinned reusable helpers and current canonical sources; no new
    # broad data admission, checkpoints, arrays or feature-cache reads.
    files={name:digest for name,digest in prior['files'].items() if not name.startswith('src/')}
    for name,digest in files.items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen reusable source changed: '+name)
    for p in [*HERE.glob('*.py'),HERE/'config.json',*(ROOT/'src/resectionlab').glob('*.py')]:
        files[str(p.relative_to(ROOT))]=sha(p)
    if files[HELPER]!=HELPER_SHA or files[SAMPLER]!=SAMPLER_SHA:raise ValueError('Owned helper pins changed')
    metadata,arrays=metadata_pins()
    for name,digest in metadata.items():
        if sha(ROOT/name)!=digest:raise ValueError('Qualification metadata changed')
    value={'schema':'train045-local-goal-source-v1','execution_released':args.release,'files':files,
        'input_metadata':metadata,'public_arrays':arrays,'checkpoints':{},
        'runtime_roots':runtime_roots()+[str(DEPS.resolve())],
        'scope':read_json(HERE/'config.json')['scope']}
    raw=canonical(value)+b'\n'
    with (HERE/'source-index.json').open('xb') as stream:stream.write(raw)
    print(hashlib.sha256(raw).hexdigest())
if __name__=='__main__':main()
