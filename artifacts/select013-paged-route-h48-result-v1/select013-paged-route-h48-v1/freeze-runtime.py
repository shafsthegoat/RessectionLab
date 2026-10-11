"""Metadata-only root freeze of the two complete-route page expansion arms."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
from comparison_contract import (OUTPUT,ARMS,LIMITS,PINS,WORKER_SECONDS,PARENT_SECONDS,
    sha,historical,read_pinned)

# Exact reviewed core successors; default-off identities are retained.
SOURCE_SUCCESSORS={
 'src/resectionlab/native_spatial_task.py':{'before':'f30a536aa2960f854765eca54504dbd3d43a9698ccfd9f9c8b12712d32b5afbf','after':'df6ee55e36b273ccc0fa726716ede724a3a892d6db24839e9e31d159070c7f03'},
}
NEW_SOURCE={}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--head',required=True);args=parser.parse_args()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    if head!=args.head:raise ValueError('Exact root-supplied HEAD required')
    for p in (ROOT/OUTPUT,(ROOT/OUTPUT).with_name('attempt-01.supervision'),HERE/'source-index.json',HERE/'release-template.json'):
        if p.exists() or p.is_symlink():raise FileExistsError('Preserve previous freeze/attempt: '+str(p))
    prior=historical();old=prior['h24_index'];bindings=dict(old['source_files']);metadata=dict(old['metadata_files'])
    # The old conservative closure already contains every canonical dependency
    # of this smaller worker. Hold all its bytes, not merely the apparent imports.
    for path,digest in list(bindings.items()):
        if path in SOURCE_SUCCESSORS:
            successor=SOURCE_SUCCESSORS[path]
            if digest!=successor['before']:raise ValueError('Unexpected predecessor source: '+path)
            digest=successor['after'];bindings[path]=digest
        if sha(ROOT/path)!=digest:raise ValueError('Prior runtime source changed: '+path)
    for path,digest in NEW_SOURCE.items():
        if sha(ROOT/path)!=digest:raise ValueError('New paged dependency changed: '+path)
        bindings[path]=digest
    for name in ('comparison_contract.py','select_worker.py','run_owned.py','freeze-runtime.py'):
        p=HERE/name;bindings[p.relative_to(ROOT).as_posix()]=sha(p)
    for ref in list(PINS.values())+[prior['release']['source_index']]:
        read_pinned(ref);metadata[ref['path']]=ref['sha256']
    for path,digest in metadata.items():
        if sha(ROOT/path)!=digest:raise ValueError('Prior metadata changed: '+path)
    for path in bindings:
        if path.startswith('src/') and subprocess.check_output(['git','show',head+':'+path],cwd=ROOT,timeout=5)!=(ROOT/path).read_bytes():raise ValueError('Uncommitted source in closure: '+path)
    index={'version':'SELECT013-complete-paged-route-h48-runtime-v1','head':head,'source_files':bindings,'metadata_files':metadata,
        'exact_source_successors':SOURCE_SUCCESSORS,'new_source':NEW_SOURCE,
        'scope':'retained conservative prior closure plus fixed diagnostic helper; metadata only, no arrays/weights opened'}
    release={'status':'pending_root_release','expected_head':head,'output':OUTPUT,'arms':list(ARMS),
        'limits':LIMITS,'worker_seconds':WORKER_SECONDS,'parent_seconds':PARENT_SECONDS,
        'public_index':prior['release']['public_index'],'prior_pins':PINS,'attempts':1,'automatic_retry':False,
        'source_index':{'path':str((HERE/'source-index.json').relative_to(ROOT)),'sha256':None},
        'selection':'two full lazy beams; identical beam2/h48/450s/14535-call/depth-volume retention; all selected32 additional pairs considered before layer pruning; no favorable page retention',
        'models_or_weights_loaded':False,'training_or_private_reference_execution':False,
        'coverage':'base120 plus bounded32 added pairs in pages8; discovery remains evidence/16-cell-selection censored; accepted complete replay required, STOP0 competes; no global reachability or clinical safety claim'}
    for name,value in [('source-index.json',index),('release-template.json',release)]:
        if name=='release-template.json':release['source_index']['sha256']=sha(HERE/'source-index.json')
        with (HERE/name).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({'index_sha256':sha(HERE/'source-index.json'),'release_template_sha256':sha(HERE/'release-template.json'),'head':head,'source_files':len(bindings),'metadata_files':len(metadata)}))
if __name__=='__main__':main()
