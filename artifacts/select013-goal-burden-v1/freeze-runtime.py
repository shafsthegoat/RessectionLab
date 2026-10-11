"""Source/metadata-only freeze. No acquired arrays, checkpoints or model imports."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
from comparison_contract import OUTPUT,ARMS,LIMITS,PINS,WORKER_SECONDS,PARENT_SECONDS,sha,historical,read_pinned

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--head',required=True);args=parser.parse_args()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    if head!=args.head:raise ValueError('Exact root-supplied HEAD required')
    for p in (ROOT/OUTPUT,(ROOT/OUTPUT).with_name('attempt-01.supervision'),HERE/'source-index.json',HERE/'release-template.json'):
        if p.exists() or p.is_symlink():raise FileExistsError('Preserve earlier freeze/attempt: '+str(p))
    saved=historical();old=saved['focused_index'];bindings={};metadata=dict(old['metadata_files']);successors={}
    # Reuse the conservative historical closure; hold every ignored dependency.
    # Canonical definitions are bound to the exact committed current revision.
    for name,digest in old['source_files'].items():
        p=ROOT/name
        if name.startswith('src/'):
            current=sha(p)
            if subprocess.check_output(['git','show',head+':'+name],cwd=ROOT,timeout=5)!=p.read_bytes():raise ValueError('Uncommitted canonical source: '+name)
            if current!=digest:successors[name]={'before':digest,'after':current}
            bindings[name]=current
        else:
            if sha(p)!=digest:raise ValueError('Historical ignored source changed: '+name)
            bindings[name]=digest
    # Include newly imported current canonical modules as conservative source-only closure.
    for p in sorted((ROOT/'src/resectionlab').glob('*.py')):
        name=p.relative_to(ROOT).as_posix()
        if subprocess.check_output(['git','show',head+':'+name],cwd=ROOT,timeout=5)!=p.read_bytes():raise ValueError('Uncommitted source: '+name)
        bindings[name]=sha(p)
    for name in ('comparison_contract.py','select_worker.py','run_owned.py','freeze-runtime.py','initial_goal_burden.py'):
        p=HERE/name;bindings[p.relative_to(ROOT).as_posix()]=sha(p)
    for ref in list(PINS.values())+[saved['release']['source_index']]:
        
        if sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Pinned metadata changed: '+ref['path'])
        metadata[ref['path']]=ref['sha256']
    for name,digest in metadata.items():
        if sha(ROOT/name)!=digest:raise ValueError('Metadata changed: '+name)
    index={'version':'SELECT013-initial-goal-burden-runtime-v1','head':head,'source_files':bindings,
        'metadata_files':metadata,'canonical_successors':successors,
        'scope':'metadata and source only; current committed closure, historical inputs unchanged'}
    release={'status':'pending_root_release','expected_head':head,'output':OUTPUT,'arms':list(ARMS),
        'limits':LIMITS,'worker_seconds':WORKER_SECONDS,'parent_seconds':PARENT_SECONDS,
        'public_index':saved['release']['public_index'],'prior_pins':PINS,'attempts':1,'automatic_retry':False,
        'source_index':{'path':str((HERE/'source-index.json').relative_to(ROOT)),'sha256':None},
        'selection':'initial-state-only existing suction goals ordered by heuristic nominal preparation estimate; same public world; no action or route and no selection of a clinical policy',
        'information_boundary':'only current public task enters rank helper; frozen historical active goal is compared afterward; no prior continuation, postoperative or private reference policy input',
        'models_or_weights_loaded':False,'training_or_private_reference_execution':False,
        'coverage':'existing BASE same-tool public goals only; rankability flags retain unknown/coverage/aperture constraints; no feasibility or lower-bound guarantee'}
    for name,value in [('source-index.json',index),('release-template.json',release)]:
        if name=='release-template.json':release['source_index']['sha256']=sha(HERE/'source-index.json')
        with (HERE/name).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({'source_index':sha(HERE/'source-index.json'),'release_template':sha(HERE/'release-template.json'),'head':head}))
if __name__=='__main__':main()
