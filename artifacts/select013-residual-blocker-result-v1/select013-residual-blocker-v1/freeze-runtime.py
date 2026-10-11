"""Metadata-only root freeze of one prospective residual-catalog diagnostic."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
from comparison_contract import (OUTPUT,ARMS,LIMITS,PINS,WORKER_SECONDS,PARENT_SECONDS,
    sha,historical,read_pinned)

SOURCE_SUCCESSORS={
 'src/resectionlab/native_proposals.py':{
  'before':'ec35077fc63e19dff422d2072c7fbe705f798ae46696ae5978f040db4b0e73a8',
  'after':'8d92d6f4c68f394f17d6c2357d8748e7abab8bfac0b555dbf3bb5d81df325a31'},
}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--head',required=True);args=parser.parse_args()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    if head!=args.head:raise ValueError('Exact root-supplied HEAD required')
    for p in (ROOT/OUTPUT,(ROOT/OUTPUT).with_name('attempt-01.supervision'),HERE/'runtime-source-index.json',HERE/'release-template.json'):
        if p.exists() or p.is_symlink():raise FileExistsError('Preserve previous freeze/attempt: '+str(p))
    prior=historical();old=prior['h48_index'];bindings=dict(old['source_files']);metadata=dict(old['metadata_files'])
    for path,digest in list(bindings.items()):
        if path in SOURCE_SUCCESSORS:
            successor=SOURCE_SUCCESSORS[path]
            if digest!=successor['before']:raise ValueError('Unexpected predecessor source: '+path)
            digest=successor['after'];bindings[path]=digest
        if sha(ROOT/path)!=digest:raise ValueError('Prior runtime source changed: '+path)
    for name in ('comparison_contract.py','residual_frontier.py','select_worker.py','run_owned.py','freeze-runtime.py'):
        p=HERE/name;bindings[p.relative_to(ROOT).as_posix()]=sha(p)
    for ref in list(PINS.values())+[prior['release']['source_index']]:
        read_pinned(ref);metadata[ref['path']]=ref['sha256']
    for path,digest in metadata.items():
        if sha(ROOT/path)!=digest:raise ValueError('Prior metadata changed: '+path)
    for path in bindings:
        if path.startswith('src/') and subprocess.check_output(['git','show',head+':'+path],cwd=ROOT,timeout=5)!=(ROOT/path).read_bytes():
            raise ValueError('Uncommitted source in closure: '+path)
    index={'version':'SELECT013-prospective-residual-frontier-runtime-v1','head':head,
        'source_files':bindings,'metadata_files':metadata,'exact_source_successors':SOURCE_SUCCESSORS,
        'scope':'historical H48 closure plus prospective helper; no arrays or weights opened by freeze'}
    release={'status':'pending_root_release','expected_head':head,'output':OUTPUT,'arms':list(ARMS),
        'limits':LIMITS,'worker_seconds':WORKER_SECONDS,'parent_seconds':PARENT_SECONDS,
        'public_index':prior['release']['public_index'],'prior_pins':PINS,'attempts':1,'automatic_retry':False,
        'source_index':{'path':str((HERE/'runtime-source-index.json').relative_to(ROOT)),'sha256':None},
        'selection':'exact saved old47-action prefix, separate residual catalog, all<=32 extra pairs verified; <=3 most-actual-first-blocker-removal branches with emitted-order ties',
        'models_or_weights_loaded':False,'training_or_private_reference_execution':False,
        'geometry':'canonical preview and commit on separate clones; deterministic native branch replay plus independent full-history source-cell geometry certificates',
        'coverage':'bounded sampled first failure sets, static aperture necessary-only filter and explicit16cell/32pair omissions; no new-rule prefix or accepted complete route claim'}
    for name,value in [('runtime-source-index.json',index),('release-template.json',release)]:
        if name=='release-template.json':release['source_index']['sha256']=sha(HERE/'runtime-source-index.json')
        with (HERE/name).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({'index_sha256':sha(HERE/'runtime-source-index.json'),'release_template_sha256':sha(HERE/'release-template.json'),
        'head':head,'source_files':len(bindings),'metadata_files':len(metadata)}))
if __name__=='__main__':main()
