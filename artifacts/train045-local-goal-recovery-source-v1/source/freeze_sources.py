"""Root-only recovery freeze; never upgrades the failed ancestor to completed."""
import argparse,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent
sys.path.insert(0,str(STAGE));sys.path.insert(0,str(HERE))
import paths
from paths import ROOT
from common import canonical,sha,read_json
from recovery_contract import require_failure


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',action='store_true')
    for name in ('failed-index','archive-inventory','postmortem','memory-policy','memory-receipt'):
        p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--memory-receipt',required=True)
    args=p.parse_args();cfg=read_json(HERE/'config.json');archive=ROOT/cfg['failed_archive']
    old=read_json(ROOT/cfg['failed_source_index'],args.failed_index_sha256)
    inventory=read_json(archive/'INVENTORY.json',args.archive_inventory_sha256)
    postmortem=read_json(archive/'source-postmortem.json',args.postmortem_sha256)
    # Root externally pins the preservation inventory, which authenticates every
    # old report byte even though the killed worker never wrote a final report.
    entries=inventory['files']
    def archived(name):
        path=archive/name
        row=next((r for r in entries if r.get('archived_path')==name),None) if isinstance(entries,list) else entries.get(name)
        if row is None:raise ValueError('Archive inventory lacks '+name)
        if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('Archived bytes differ: '+name)
        return read_json(path,row['sha256'])
    report=archived('stopped-report.json');parent=archived('launcher.json');curve=archived('reverse512-curve.json')
    pins={'report':sha(archive/'stopped-report.json'),'launcher':sha(archive/'launcher.json'),
          'index':args.failed_index_sha256,'postmortem':args.postmortem_sha256}
    admission=require_failure(report,parent,curve,postmortem,old,pins=pins,configuration=cfg)
    files=dict(old['files']);policy=str((STAGE/'model/policy.py').relative_to(ROOT));original_sha=files[policy]
    if sha(archive/'original-policy.py')!=original_sha:raise ValueError('Preserved original policy differs')
    if sha(ROOT/policy)!=args.memory_policy_sha256:raise ValueError('Reviewed memory policy differs')
    files[policy]=args.memory_policy_sha256
    for name,digest in files.items():
        if sha(ROOT/name)!=digest:raise ValueError('Unexpected source delta: '+name)
    files[str((archive/'original-policy.py').relative_to(ROOT))]=original_sha
    for path in [*HERE.glob('*.py'),HERE/'config.json']:files[str(path.relative_to(ROOT))]=sha(path)
    receipt=(ROOT/args.memory_receipt).resolve()
    if not receipt.is_relative_to(STAGE.resolve()) or sha(receipt)!=args.memory_receipt_sha256:
        raise ValueError('Root-pinned memory parity receipt required')
    evidence=read_json(receipt,args.memory_receipt_sha256)
    parity=evidence['runs'][0]['result']
    if (evidence['status']!='pass' or not evidence['all_children_reaped']
        or any(r['exit_code']!=0 or r['stop_reason'] is not None for r in evidence['runs'])
        or parity['status']!='pass' or parity['failures'] or parity['errors']
        or parity['source_sha256'][policy]!=args.memory_policy_sha256
        or not evidence['assembly_comparison']['exact_volume']):
        raise ValueError('Completed separately pinned memory parity controls required')
    metadata={**old['input_metadata'],cfg['failed_source_index']:args.failed_index_sha256,
        str(receipt.relative_to(ROOT)):args.memory_receipt_sha256}
    for name in ('stopped-report.json','launcher.json','reverse512-curve.json','source-postmortem.json',
                 'INVENTORY.json','last-progress.json'):
        metadata[str((archive/name).relative_to(ROOT))]=sha(archive/name)
    q=curve['checkpoints'][0];pin=q['checkpoint'];path=ROOT/cfg['failed_output_directory']/pin['path']
    if path.name!='ppo-0512.zip' or path.parent.name!='reverse-seed0' or sha(path)!=pin['sha256'] or path.stat().st_size!=pin['bytes']:
        raise ValueError('Exact original reverse512 optimizer archive required')
    checkpoints={**old['checkpoints'],'REVERSE512':{**pin,'path':str(path.relative_to(ROOT))}}
    for endpoint in (q,report['arms']['uniform-seed0']['checkpoints'][-1]):
        for key in ('sealed_plan','history'):
            item=endpoint['evaluation'][key];path=ROOT/cfg['failed_output_directory']/item['path']
            if not path.resolve().is_relative_to((ROOT/cfg['failed_output_directory']).resolve()) or sha(path)!=item['sha256']:
                raise ValueError('Retained independent endpoint artifact changed')
            metadata[str(path.relative_to(ROOT))]=item['sha256']
    for name,digest in metadata.items():
        if sha(ROOT/name)!=digest:raise ValueError('Pinned metadata changed: '+name)
    value={**old,'schema':cfg['schema'],'execution_released':args.release,'files':files,'input_metadata':metadata,
        'checkpoints':checkpoints,'scope':cfg['scope'],'recovery_admission':admission,
        'memory_change':{'original_policy_sha256':original_sha,'new_policy_sha256':args.memory_policy_sha256,
            'receipt_path':str(receipt.relative_to(ROOT)),'receipt_sha256':args.memory_receipt_sha256,
            'root_reviewed_before_release':args.release,'scope':'Arithmetic-preserving input assembly only; policy parameters and PPO settings unchanged'},
        'continuation':{'restored_decisions':512,'additional_decisions':512,'milestones':[1024],
            'original_reverse_reset_offset':256,'fresh_RNG_and_episode':True}}
    raw=canonical(value)+b'\n'
    with (HERE/'source-index.json').open('xb') as f:f.write(raw)
    print(hashlib.sha256(raw).hexdigest())

if __name__=='__main__':main()
