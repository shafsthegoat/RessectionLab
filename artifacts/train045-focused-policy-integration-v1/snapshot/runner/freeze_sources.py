"""Root-only metadata/source freeze; never opens arrays or model artifacts."""
import argparse,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import ROOT,HELPER,HELPER_SHA,SAMPLER,SAMPLER_SHA,read_json,sha,canonical,runtime_roots
from inputs import (metadata_pins,REPORT,REPORT_SHA,PLANS,PLANS_SHA,retained_prefix,
    PRIOR_REPORT,PRIOR_REPORT_SHA,PRIOR_PARENT,PRIOR_PARENT_SHA,require_reference,
    TAIL_REPORT,TAIL_REPORT_SHA,require_tail_metadata,CHECKPOINTS)
PHYSICS='build/select013-obstruction-frontier-v2/frontier_compare.py'
PREFIX='build/train045-frontier-preparation-v1/recovery-compact/frontier_protocol.py'
PREFIX_SHA='8c878476fa62e63ff721b98085fe78672162ffcbe75f76c8c615958ec2239df0'
PHYSICS_SHA='e02d7b02ba8d4c8188f1a3643b44333fa08ab4db413355324704c58d72abd20b'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',action='store_true');args=parser.parse_args()
    config=read_json(HERE/'config.json')
    if sha(ROOT/PREFIX)!=PREFIX_SHA:raise ValueError('Reviewed focused deployment prefix helper changed')
    if sha(ROOT/PHYSICS)!=PHYSICS_SHA:raise ValueError('Reviewed physical-field helper changed')
    retained_prefix(read_json(ROOT/REPORT,REPORT_SHA),read_json(ROOT/PLANS,PLANS_SHA))
    require_reference(read_json(ROOT/PRIOR_REPORT,PRIOR_REPORT_SHA),read_json(ROOT/PRIOR_PARENT,PRIOR_PARENT_SHA))
    require_tail_metadata(read_json(ROOT/TAIL_REPORT,TAIL_REPORT_SHA))
    paths=list(HERE.glob('*.py'))+[HERE/'config.json']+list((ROOT/'src/resectionlab').glob('*.py'))
    paths+=list((HERE.parent/'model').glob('*.py'))+list((HERE.parent/'independent').glob('*.py'))
    paths += [ROOT/HELPER,ROOT/SAMPLER,ROOT/PHYSICS,ROOT/PREFIX]
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    if files[HELPER]!=HELPER_SHA or files[SAMPLER]!=SAMPLER_SHA:raise ValueError('Owned lifecycle pin changed')
    for name in ('focused_policy.py','saved_tail.py'):
        if not (HERE.parent/'model'/name).is_file():raise ValueError('Final model source required')
    metadata,arrays=metadata_pins()
    value={'schema':'fixed045-focused-policy-source-v1','execution_released':args.release,'files':files,
        'input_metadata':metadata,'public_arrays':arrays,'checkpoints':CHECKPOINTS,'runtime_roots':runtime_roots(),
        'scope':config['scope'],'root_declared_head':config['baseline_canonical_head']}
    raw=canonical(value)+b'\n'
    with (HERE/'source-index.json').open('xb') as stream:stream.write(raw)
    print(hashlib.sha256(raw).hexdigest())
if __name__=='__main__':main()
