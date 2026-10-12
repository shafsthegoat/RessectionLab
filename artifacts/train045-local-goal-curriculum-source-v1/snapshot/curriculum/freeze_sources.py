"""Root-only conditional metadata freeze; incomplete/successful256 closes release."""
import argparse,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent
sys.path.insert(0,str(STAGE));sys.path.insert(0,str(HERE))
import paths
from paths import ROOT
from common import canonical,sha,read_json
from curriculum_contract import require_campaign


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',action='store_true')
    for name in ('campaign-report','campaign-launcher','campaign-source-index'):
        p.add_argument('--'+name+'-sha256',required=True)
    args=p.parse_args();config=read_json(HERE/'config.json')
    report=read_json(ROOT/config['campaign_report'],args.campaign_report_sha256)
    parent=read_json(ROOT/config['campaign_launcher'],args.campaign_launcher_sha256)
    admission=require_campaign(report,parent,report_sha=args.campaign_report_sha256,
        index_sha=args.campaign_source_index_sha256,configuration=config)
    prior=read_json(ROOT/config['campaign_source_index'],args.campaign_source_index_sha256)
    if not prior['execution_released']:raise ValueError('Original campaign was not released')
    files=dict(prior['files'])
    for name,digest in files.items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen dependency changed: '+name)
    for file in [*HERE.glob('*.py'),HERE/'config.json']:files[str(file.relative_to(ROOT))]=sha(file)
    metadata={**prior['input_metadata'],config['campaign_report']:args.campaign_report_sha256,
        config['campaign_launcher']:args.campaign_launcher_sha256,
        config['campaign_source_index']:args.campaign_source_index_sha256}
    output=ROOT/report['config']['output_directory']
    row=report['arms']['frozen_encoder-seed0']['checkpoints'][0];pin=row['checkpoint'];file=output/pin['path']
    if file.resolve().parent!=output/'frozen_encoder-seed0' or file.name!='ppo-0256.zip':
        raise ValueError('Only exact frozen256 checkpoint is eligible')
    if file.stat().st_size!=pin['bytes'] or sha(file)!=pin['sha256']:raise ValueError('Checkpoint bytes changed')
    checkpoints={**prior['checkpoints'],'FROZEN256':{**pin,'path':str(file.relative_to(ROOT))}}
    for arm in report['arms'].values():
        for key in ('sealed_plan','history'):
            artifact=arm['checkpoints'][0]['evaluation'][key];file=output/artifact['path']
            if not file.resolve().is_relative_to(output.resolve()) or sha(file)!=artifact['sha256']:
                raise ValueError('Inherited256 audit artifact changed')
            metadata[str(file.relative_to(ROOT))]=artifact['sha256']
    for name,digest in metadata.items():
        if sha(ROOT/name)!=digest:raise ValueError('Pinned metadata changed: '+name)
    value={**prior,'schema':config['schema'],'execution_released':args.release,'files':files,
        'input_metadata':metadata,'checkpoints':checkpoints,'scope':config['scope'],
        'campaign_source_index_sha256':args.campaign_source_index_sha256,'campaign_admission':admission,
        'ancestor_continuation':prior['continuation'],
        'continuation':{'prior_decisions_per_method':256,'additional_decisions_per_method':768,'milestones':[512,1024]},
        'comparison':{'only_intervention':'reset scheduling','shared_frozen256_checkpoint':pin['sha256'],
            'additional_decisions_per_method':768,'milestones':[512,1024],'fresh_RNG_and_episode':True}}
    raw=canonical(value)+b'\n'
    with (HERE/'source-index.json').open('xb') as f:f.write(raw)
    print(hashlib.sha256(raw).hexdigest())
if __name__=='__main__':main()
