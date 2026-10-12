"""Root-only scalar/byte freeze of completed64 evidence and fixed256 continuation."""
import argparse,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent
sys.path.insert(0,str(STAGE));sys.path.insert(0,str(HERE))
import paths
from paths import ROOT
from common import canonical,sha,read_json
from contract import ARMS,require_pilot


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',action='store_true')
    for name in ('pilot-report','pilot-launcher','pilot-source-index'):
        parser.add_argument('--'+name+'-sha256',required=True)
    args=parser.parse_args();config=read_json(HERE/'config.json')
    report=read_json(ROOT/config['pilot_report'],args.pilot_report_sha256)
    parent=read_json(ROOT/config['pilot_launcher'],args.pilot_launcher_sha256)
    admission=require_pilot(report,parent,report_sha=args.pilot_report_sha256,
        index_sha=args.pilot_source_index_sha256,configuration=config)
    prior=read_json(ROOT/config['pilot_source_index'],args.pilot_source_index_sha256)
    if not prior['execution_released']:raise ValueError('Original pilot was not released')
    files=dict(prior['files'])
    for name,digest in files.items():
        if sha(ROOT/name)!=digest:raise ValueError('Immutable pilot dependency changed: '+name)
    for p in [*HERE.glob('*.py'),HERE/'config.json']:
        files[str(p.relative_to(ROOT))]=sha(p)
    metadata=dict(prior['input_metadata'])
    metadata.update({config['pilot_report']:args.pilot_report_sha256,
        config['pilot_launcher']:args.pilot_launcher_sha256,
        config['pilot_source_index']:args.pilot_source_index_sha256})
    checkpoints=dict(prior['checkpoints']);pilot_output=ROOT/report['config']['output_directory']
    for name in ARMS:
        arm=report['arms'][name];row=arm['checkpoints'][0];binding=row['checkpoint']
        file=pilot_output/binding['path']
        if file.resolve().parent!=pilot_output/name or file.name!='ppo-0064.zip':
            raise ValueError('Original exact arm64 checkpoint path required')
        if file.stat().st_size!=binding['bytes'] or sha(file)!=binding['sha256']:
            raise ValueError('Original complete checkpoint bytes changed')
        checkpoints[name]={**binding,'path':str(file.relative_to(ROOT))}
        for evaluation in (arm['initial_evaluation'],row['evaluation']):
            for key in ('sealed_plan','history'):
                artifact=evaluation[key];path=pilot_output/artifact['path']
                if not path.resolve().is_relative_to(pilot_output.resolve()):raise ValueError('Evidence path escaped')
                # Whole compressed bytes remain original; no decoding/copies here.
                digest=artifact.get('compressed_sha256',artifact.get('sha256'))
                if not isinstance(digest,str) or sha(path)!=digest:raise ValueError('Inherited audit evidence changed')
                metadata[str(path.relative_to(ROOT))]=digest
    for name,digest in metadata.items():
        if sha(ROOT/name)!=digest:raise ValueError('Pinned metadata changed: '+name)
    value={**prior,'schema':config['schema'],'execution_released':args.release,'files':files,
        'input_metadata':metadata,'checkpoints':checkpoints,'scope':config['scope'],
        'pilot_source_index_sha256':args.pilot_source_index_sha256,'pilot_admission':admission,
        'continuation':{'prior_decisions_per_arm':64,'additional_decisions_per_arm':192,
            'endpoint_decisions_per_arm':256,'optimizer_preserved':True,
            'physical_episode_continuation':False,'initial_and64_audits_repeated':False}}
    raw=canonical(value)+b'\n'
    with (HERE/'source-index.json').open('xb') as stream:stream.write(raw)
    print(hashlib.sha256(raw).hexdigest())
if __name__=='__main__':main()
