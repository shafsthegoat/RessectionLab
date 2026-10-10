"""Owned eight-update four-TRAIN follow-on; no retry or held-out factory."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(HERE))
from pilot_contract import (OUTPUT,WORKER_SECONDS,TRAIN,PUBLIC_INDEX,PUBLIC_SHA,COHORT,
    canonical_configuration,canonical,source_guard,sha,complete_result,first_update_control)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args()
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    release=json.loads(args.release.read_text())
    source_guard(release,args.release,args.release_sha256)
    output=ROOT/OUTPUT
    if output.exists():raise FileExistsError('One-attempt output already exists')
    supervision=output.with_name(output.name+'.supervision')
    if not supervision.is_dir():raise ValueError('Owned parent reservation required')
    started=time.perf_counter();summary={'status':'started','TRAIN':list(TRAIN),
        'SELECT_EVAL_opened':False,'automatic_retry':False}
    def write(path,value):
        with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    def progress(phase,**details):
        row={'phase':phase,'worker_seconds':time.perf_counter()-started,**details}
        temporary=supervision/'worker-progress.tmp'
        temporary.write_text(json.dumps(row,allow_nan=False)+'\n')
        temporary.replace(supervision/'worker-progress.json')
    def deadline(*unused):raise TimeoutError('Complete four-TRAIN worker deadline')
    previous=signal.signal(signal.SIGALRM,deadline)
    signal.setitimer(signal.ITIMER_REAL,WORKER_SECONDS)
    try:
        import torch
        import numpy as np
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        summary['runtime']={'python':sys.version,'executable':sys.executable,
            'torch':torch.__version__,'numpy':np.__version__,'threads':torch.get_num_threads(),
            'interop_threads':torch.get_num_interop_threads(),'device':'cpu','dtype':'float32'}
        protocol,limits=canonical_configuration()
        if (canonical(protocol)!=canonical(release['learning_protocol'])
                or canonical(limits)!=canonical(release['cohort_limits'])):
            raise ValueError('Canonical protocol/limits differ from released configuration')
        from resectionlab.patient_planning_cohort_visits import make_train_visit_factories
        from resectionlab.patient_planning_cohort_sequential import run_train_cohort_sequential
        factories=make_train_visit_factories(manifest_index_path=ROOT/PUBLIC_INDEX,
            manifest_index_sha256=PUBLIC_SHA,cohort_bytes=(ROOT/COHORT).read_bytes(),
            learning_protocol=protocol,limits=limits,released_record=release,
            released_sha256=args.release_sha256,progress=progress)
        # The canonical runner completes all four teachers before testing its
        # all-no-positive gate. A negative008 teacher never excludes020/025.
        run_train_cohort_sequential(factories,learning_protocol=protocol,limits=limits,output=output)
        result=json.loads((output/'result.json').read_text())
        if not complete_result(result):raise ValueError('Incomplete four-TRAIN updates/checkpoints/greedy replays')
        control=first_update_control(output,release)
        write(supervision/'first-update-control.json',control)
        for method,row in result['checkpoints'].items():
            if row['path']!=method+'-final.psckpt' or sha(output/row['path'])!=row['sha256']:
                raise ValueError('Saved fixed endpoint changed: '+method)
        source_guard(release,args.release,args.release_sha256)
        summary.update(status='complete_owned_fixed_four_TRAIN',result_sha256=sha(output/'result.json'),
            optimizer_updates=result['optimizer_updates'],complete_TRAIN_greedy_replays=8,
            first_update_control_sha256=sha(supervision/'first-update-control.json'))
    except BaseException as error:
        summary.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
        summary['wall_seconds']=time.perf_counter()-started
        summary['canonical_result_sha256']=sha(output/'result.json') if (output/'result.json').is_file() else None
        write(supervision/'worker-final.json',summary)

if __name__=='__main__':main()
