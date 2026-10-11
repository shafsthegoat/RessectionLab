"""Metadata-only root freeze of the same-world obstruction proposal contrast."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
from comparison_contract import (OUTPUT,ARMS,LIMITS,PINS,WORKER_SECONDS,PARENT_SECONDS,
    sha,historical,read_pinned)

# Reviewed opt-in TRAIN050 development admission was committed after the
# witness. This worker omits that new option, preserving the original013 branch.
# Bind only these exact successors; every other historical dependency stays exact.
SOURCE_SUCCESSORS={
    'src/resectionlab/patient_planning_admission.py': {
        'before':'ab063b5b5a67b825009dbc9052befb6270ff39d8d5ce87ed555a1ab79394aed9',
        'after':'ff599ea5ba3d026f8342b54ea18a395c24719c57e6d860b41dfb177721247f95'},
    'src/resectionlab/public_patient_factory.py': {
        'before':'aa4c0a07ecb4988934bb3911927d7ce724037fec63aecad7c50f8a8f2d7a7dc5',
        'after':'da5ec355048effd160b3f7211544cf7ace4912aac4cd21e768ce424e22f9a2aa'},
}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--head',required=True);args=parser.parse_args()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    if head!=args.head:raise ValueError('Exact root-supplied HEAD required')
    for p in (ROOT/OUTPUT,(ROOT/OUTPUT).with_name('attempt-01.supervision'),HERE/'source-index.json',HERE/'release-template.json'):
        if p.exists() or p.is_symlink():raise FileExistsError('Preserve previous freeze/attempt: '+str(p))
    prior=historical();old=prior['witness_index'];bindings=dict(old['source_files']);metadata=dict(old['metadata_files'])
    # The old conservative closure already contains every canonical dependency
    # of this smaller worker. Hold all its bytes, not merely the apparent imports.
    for path,digest in list(bindings.items()):
        if path in SOURCE_SUCCESSORS:
            successor=SOURCE_SUCCESSORS[path]
            if digest!=successor['before']:raise ValueError('Unexpected predecessor source: '+path)
            digest=successor['after'];bindings[path]=digest
        if sha(ROOT/path)!=digest:raise ValueError('Prior runtime source changed: '+path)
    for name in ('comparison_contract.py','frontier_compare.py','select_worker.py','run_owned.py','freeze-runtime.py'):
        p=HERE/name;bindings[p.relative_to(ROOT).as_posix()]=sha(p)
    for ref in list(PINS.values())+[prior['release']['source_index']]:
        read_pinned(ref);metadata[ref['path']]=ref['sha256']
    for path,digest in metadata.items():
        if sha(ROOT/path)!=digest:raise ValueError('Prior metadata changed: '+path)
    for path in bindings:
        if path.startswith('src/') and subprocess.check_output(['git','show',head+':'+path],cwd=ROOT,timeout=5)!=(ROOT/path).read_bytes():raise ValueError('Uncommitted source in closure: '+path)
    index={'version':'fixed-obstruction-frontier-runtime-v1','head':head,'source_files':bindings,'metadata_files':metadata,
        'exact_source_successors':SOURCE_SUCCESSORS,
        'scope':'retained conservative prior closure plus fixed diagnostic helper; metadata only, no arrays/weights opened'}
    release={'status':'pending_root_release','expected_head':head,'output':OUTPUT,'arms':list(ARMS),
        'limits':LIMITS,'worker_seconds':WORKER_SECONDS,'parent_seconds':PARENT_SECONDS,
        'public_index':prior['release']['public_index'],'prior_pins':PINS,'attempts':1,'automatic_retry':False,
        'source_index':{'path':str((HERE/'source-index.json').relative_to(ROOT)),'sha256':None},
        'selection':'exact physical saved11 prefix in both proposal conditions; provider-order feasible added single-preparation branches; original-ray previews; STOP costs retained',
        'models_or_weights_loaded':False,'training_or_private_reference_execution':False,
        'coverage':'bounded existing public obstruction-opening rule at120; omission-censored discovery; direct original-ray clearance not a complete route'}
    for name,value in [('source-index.json',index),('release-template.json',release)]:
        if name=='release-template.json':release['source_index']['sha256']=sha(HERE/'source-index.json')
        with (HERE/name).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({'index_sha256':sha(HERE/'source-index.json'),'release_template_sha256':sha(HERE/'release-template.json'),'head':head,'source_files':len(bindings),'metadata_files':len(metadata)}))
if __name__=='__main__':main()
