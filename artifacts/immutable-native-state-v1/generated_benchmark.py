"""One generated20.77m-voxel arm; root must serialize the owned parent."""
import argparse
from dataclasses import replace
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import types

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
SHAPE=(251,331,250)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('baseline','stage'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    if args.output.exists():raise FileExistsError('one generated arm only')
    started=time.perf_counter();counts={'native_previews':0,'commits':0};result={'variant':args.variant,'shape':SHAPE,'status':'started','counts':counts}
    def stop(*args):raise TimeoutError('generated benchmark25s arm cap')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,25)
    try:
        def audit(event,values):
            if event=='open' and values and isinstance(values[0],(str,bytes)):
                path=Path(os.fsdecode(values[0])).resolve()
                if path.is_relative_to(ROOT/'data') or path.suffix.lower() in ('.npy','.npz','.dcm','.pt','.pth'):
                    raise PermissionError('only generated arrays; no patient or model files')
            if event in ('subprocess.Popen','os.system','os.fork','socket.connect','socket.bind'):
                raise PermissionError('no external work in generated benchmark')
        sys.addaudithook(audit)
        import numpy as np
        package=types.ModuleType('_native_hash_benchmark')
        package.__path__=[str(HERE/args.variant/'resectionlab'),str(ROOT/'src/resectionlab')]
        sys.modules[package.__name__]=package
        taskmod=importlib.import_module(package.__name__+'.native_spatial_task')
        enginemod=importlib.import_module(package.__name__+'.native_resection')
        core=importlib.import_module(package.__name__+'.core')
        def check():
            if time.perf_counter()-started>=25:stop()
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2*1024**3:raise MemoryError('generated benchmark2GiB cap')
        original_preview=enginemod.NativeResectionEngine.preview_stroke
        def preview(self,*a,**k):
            check();counts['native_previews']+=1
            if counts['native_previews']>64:raise RuntimeError('generated benchmark preview cap')
            return original_preview(self,*a,**k)
        enginemod.NativeResectionEngine.preview_stroke=preview
        # The original six-cell software fixture is embedded without changing
        # its1mm geometry. Only array cardinality matches008; no anatomy claim.
        small=taskmod.make_native_opening_task().case
        image=np.zeros(SHAPE,np.float32);support=np.zeros(SHAPE,bool);target=np.zeros(SHAPE,np.float32)
        region=tuple(slice(0,n) for n in small.observed_support.shape)
        image[region]=small.structural_intensity;support[region]=small.observed_support;target[region]=small.nominal_target
        case=replace(small,structural_intensity=image,observed_support=support,reference_target=target,nominal_target=target,crop_shape=(16,16,16))
        del image,support,target,small;gc.collect();check()
        task=taskmod.NativeSpatialTask(case,max_steps=2).planning_clone()
        identifier=next(r['action_id'] for r in task.candidate_inventory()['ledger'] if r.get('voxel')==[4,4,1] and r['tool_id']==case.tools[0].tool_id and r['feasible'])
        result['setup_seconds']=time.perf_counter()-started
        result['source_hash']=case.source_hash;result['model_hash']=task.decision_model_hash
        digest_counts={'calls':0,'bytes':0};original_digest=core.array_digest
        def counted(array):
            digest_counts['calls']+=1;digest_counts['bytes']+=array.nbytes
            return original_digest(array)
        for name,module in list(sys.modules.items()):
            if name.startswith(package.__name__+'.') and getattr(module,'array_digest',None) is original_digest:
                module.array_digest=counted
        mark=time.perf_counter()
        for _ in range(16):check();task._assert_frozen()
        result['unchanged_guards']={'seconds':time.perf_counter()-mark,**digest_counts}
        digest_counts.update(calls=0,bytes=0);mark=time.perf_counter();signatures=[]
        for _ in range(16):
            check();branch=task.clone();step=branch.advance_planning(identifier);counts['commits']+=1
            branch._assert_frozen()
            signatures.append({'state_seal':branch._state_seal,'engine_state':branch._engine.state_hash,
                'step_hash':core.semantic_digest(step.info),'state_record_hash':core.semantic_digest(branch._state_record())})
            del step,branch;gc.collect()
        result['branch_commits']={'seconds':time.perf_counter()-mark,**digest_counts,'signatures':signatures}
        result['status']='complete_generated_cardinality_control'
        result['native_geometry']='same six-cell1mm fixture; source array251x331x250; no patient-derived samples'
    except BaseException as error:result.update(status='terminal_cap_or_failure',error=type(error).__name__+':'+str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        result.update(elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),patient_reads=0,model_calls=0)
        with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return 0 if result['status']=='complete_generated_cardinality_control' else 1


if __name__=='__main__':raise SystemExit(main())
