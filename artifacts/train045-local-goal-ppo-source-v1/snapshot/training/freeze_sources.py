"""Root-only metadata freeze; no task, checkpoint payload or model imports."""
import argparse,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent
sys.path.insert(0,str(STAGE))
import paths
from paths import ROOT,DEPS
from common import canonical,sha,read_json,runtime_roots,IL,TRAIN
from inputs import CHECKPOINTS,TAIL_INDEX,TAIL_INDEX_SHA

QUALIFICATION_SHA='3ea0c726e744acee73a59380da36746cf3eefd8ca948ed28cdc2693d8fb29a05'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',action='store_true')
    parser.add_argument('--qualification-launcher-sha256',required=True);args=parser.parse_args()
    config=read_json(HERE/'config.json')
    if config['arms']!=['frozen_encoder','trainable_encoder'] or config['seeds']!=[0] or config['milestones']!=[64]:
        raise ValueError('This freeze releases only fixed64-per-arm seed0 pilot')
    q=read_json(ROOT/config['qualification_report'],QUALIFICATION_SHA)
    parent=read_json(ROOT/config['qualification_launcher'],args.qualification_launcher_sha256)
    if (q['status']!='completed' or not q['qualified'] or not q['source_unchanged']
        or parent['status']!='completed' or parent['report_sha256']!=QUALIFICATION_SHA
        or q['config']['goal']!=config['goal']):raise ValueError('Fixed qualified existing task required')
    prior=read_json(STAGE/'source-index.json',q['source_index_sha256'])
    files=dict(prior['files'])
    for name,digest in files.items():
        if sha(ROOT/name)!=digest:raise ValueError('Qualified or reusable source changed: '+name)
    for p in [*(STAGE/'model').glob('*.py'),*HERE.glob('*.py'),HERE/'config.json']:
        files[str(p.relative_to(ROOT))]=sha(p)
    metadata=dict(prior['input_metadata'])
    metadata.update({config['qualification_report']:QUALIFICATION_SHA,
        config['qualification_launcher']:args.qualification_launcher_sha256})
    tail=read_json(ROOT/TAIL_INDEX,TAIL_INDEX_SHA)
    metadata[TAIL_INDEX]=TAIL_INDEX_SHA
    for subject in TRAIN:
        name=IL+'/'+subject+'-context.json';metadata[name]=tail['input_metadata'][name]
    for name,digest in metadata.items():
        if sha(ROOT/name)!=digest:raise ValueError('Pinned metadata changed: '+name)
    # Root-approved raw hash remains mandatory; decoding happens only in worker.
    if sha(ROOT/CHECKPOINTS['IL64']['path'])!=CHECKPOINTS['IL64']['sha256']:
        raise ValueError('Original IL64 checkpoint changed')
    value={'schema':config['schema'],'execution_released':args.release,'files':files,
        'input_metadata':metadata,'public_arrays':prior['public_arrays'],
        'checkpoints':{'IL64':CHECKPOINTS['IL64']},'runtime_roots':runtime_roots()+[str(DEPS.resolve())],
        'qualification_source_index_sha256':q['source_index_sha256'],
        'scope':config['scope'],'training_spec':{'algorithm':'sb3_contrib.MaskablePPO2.9.0',
            'raw_spatial_encoder':True,'goal_columns_zero_initialized':True,'oracle_reward_features':False}}
    raw=canonical(value)+b'\n'
    with (HERE/'source-index.json').open('xb') as stream:stream.write(raw)
    print(hashlib.sha256(raw).hexdigest())
if __name__=='__main__':main()
