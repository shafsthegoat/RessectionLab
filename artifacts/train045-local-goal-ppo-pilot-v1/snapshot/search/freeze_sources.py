"""Root-only additive freeze on the exact independently qualified local task."""
import argparse,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent
sys.path.insert(0,str(STAGE))
import paths
from paths import ROOT
from common import canonical,sha,read_json

QUALIFICATION_SHA='3ea0c726e744acee73a59380da36746cf3eefd8ca948ed28cdc2693d8fb29a05'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',action='store_true')
    parser.add_argument('--qualification-launcher-sha256',required=True);args=parser.parse_args()
    config=read_json(HERE/'config.json')
    q=read_json(ROOT/config['qualification_report'],QUALIFICATION_SHA)
    parent=read_json(ROOT/config['qualification_launcher'],args.qualification_launcher_sha256)
    if (q['status']!='completed' or not q['qualified'] or not q['source_unchanged']
        or not q['independent_audit']['accepted'] or not q['local_reward_audit']['accepted']
        or parent['status']!='completed' or parent['report_sha256']!=QUALIFICATION_SHA
        or q['config']['goal']!=config['goal']):raise ValueError('Exact qualified local task required')
    prior=read_json(STAGE/'source-index.json',q['source_index_sha256'])
    files=dict(prior['files'])
    for name,digest in files.items():
        if sha(ROOT/name)!=digest:raise ValueError('Qualified frozen source changed: '+name)
    for p in [*HERE.glob('*.py'),HERE/'config.json']:
        files[str(p.relative_to(ROOT))]=sha(p)
    metadata=dict(prior['input_metadata'])
    metadata.update({config['qualification_report']:QUALIFICATION_SHA,
        config['qualification_launcher']:args.qualification_launcher_sha256})
    for name,digest in metadata.items():
        if sha(ROOT/name)!=digest:raise ValueError('Pinned metadata changed: '+name)
    value={**prior,'schema':config['schema'],'execution_released':args.release,'files':files,
        'input_metadata':metadata,'checkpoints':{},'scope':config['scope'],
        'qualification_source_index_sha256':q['source_index_sha256'],
        'search_rule':config['search_rule'],'search_limits':config['search_limits']}
    raw=canonical(value)+b'\n'
    with (HERE/'source-index.json').open('xb') as stream:stream.write(raw)
    print(hashlib.sha256(raw).hexdigest())
if __name__=='__main__':main()
