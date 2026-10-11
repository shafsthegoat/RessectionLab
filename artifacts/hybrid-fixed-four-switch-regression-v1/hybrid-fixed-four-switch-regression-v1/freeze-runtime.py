"""Root-only metadata freeze; opens no model or source-array payload."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
if __name__=='__main__' and (not sys.pycache_prefix or Path(sys.pycache_prefix).exists()):
    raise ValueError('Fresh nonexistent -X pycache_prefix required before local imports')
sys.path.insert(0,str(HERE));sys.dont_write_bytecode=True
from batch_contract import need,sha,input_refs,metadata,template,read_bytes

def main():
    p=argparse.ArgumentParser();p.add_argument('--head',required=True);a=p.parse_args()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();need(head==a.head,'root_HEAD')
    for name in ('attempt-01','attempt-01.supervision','source-index.json','release-template.json'):need(not (HERE/name).exists(),'one_fresh_freeze')
    metadata();refs=input_refs()
    for ref in refs:read_bytes(ref['path'],ref['sha256'])
    paths=list((ROOT/'src/resectionlab').glob('*.py'))
    paths += [HERE/name for name in ('batch_contract.py','patient_regression.py','fallback_policy.py','hybrid_proposal.py','initial_inventory_worker.py','run_owned.py','freeze-runtime.py')]
    paths += [ROOT/'build/goal-conditioned-policy-v1'/name for name in ('run_contact_owned.py','darwin_fast_sampler.py')]
    files={}
    for path in paths:
        relative=path.relative_to(ROOT).as_posix();files[relative]=sha(path)
        if relative.startswith('src/'):
            need(subprocess.check_output(['git','show',head+':'+relative],cwd=ROOT)==read_bytes(path),'uncommitted_runtime_source:'+relative)
    files.update({r['path']:r['sha256'] for r in refs})
    index={'version':'fixed-four-frozen-switch-regression-runtime-v1','head':head,'files':files,'payloads_opened':False}
    encode=lambda v:(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    raw=encode(index);ref={'path':(HERE/'source-index.json').relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(raw).hexdigest()}
    release=template(head,ref)
    need(not any(x in sys.modules for x in ('numpy','torch','resectionlab')),'metadata_only_freeze')
    (HERE/'source-index.json').write_bytes(raw);(HERE/'release-template.json').write_bytes(encode(release))
    print(json.dumps({'source_index':ref,'false_release_sha256':sha(HERE/'release-template.json'),'execution_released':False}))
if __name__=='__main__':main()
