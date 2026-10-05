"""Bounded development profiling; never opens final/selection worlds."""
import argparse
import cProfile
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform
import pstats
import resource
import time
import numpy as np
from resectionlab.imaging import load_case
from resectionlab.native_simulation import make_native_patient_simulator


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)).encode()).hexdigest()


def observe(sim):
    obs=sim.observation()
    return {'action_ids':obs.action_ids,'action_features':obs.action_features.tolist(),'action_mask':obs.action_mask.tolist(),'state_features':obs.state_features.tolist()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    phases={}
    def timed(name,fn):
        profile=cProfile.Profile()
        started=time.perf_counter()
        value=profile.runcall(fn)
        elapsed=time.perf_counter()-started
        stream=io.StringIO()
        pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(35)
        (args.output/(name+'.profile.txt')).write_text(stream.getvalue())
        phases[name]={'elapsed_seconds':elapsed,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        return value
    case=timed('load_case',lambda:load_case(args.case))
    sim=timed('prepare',lambda:make_native_patient_simulator(case,candidate_count=4,max_steps=3,max_actions=7))
    records={'source_hash':case.semantic_hash,'decision_model_hash':sim.decision_model_hash,'initial':observe(sim),'resets':[],'rollouts':[]}
    for i in range(3):
        timed('reset_'+str(i),lambda:sim.reset(i))
        records['resets'].append(observe(sim))
    for i in range(2):
        timed('rollout_reset_'+str(i),lambda:sim.reset(0))
        steps=[]
        for point in (0,3):
            action='NATIVE:native-wide-aspiration:'+str(point)
            if action not in sim.observation().action_ids:
                break
            outcome=timed('rollout_'+str(i)+'_point_'+str(point),lambda:sim.step(action))
            steps.append({'reward':outcome.reward,'info':outcome.info,'observation':observe(sim)})
        sim.step('STOP')
        records['rollouts'].append({'steps':steps,'metrics':sim.metrics()})
    timed('clone_after_rollout',sim.clone)
    del sim
    import gc
    gc.collect()
    # A separate factory remains cold; cache setup cannot be hidden across arms.
    fresh=timed('second_prepare',lambda:make_native_patient_simulator(case,candidate_count=4,max_steps=3,max_actions=7))
    records['second_initial']=observe(fresh)
    records['identity_digest']=digest(records)
    (args.output/'result_identity.json.gz').write_bytes(gzip.compress(json.dumps(records,indent=2,sort_keys=True).encode(),mtime=0))
    summary={'phase':'development_performance_only','final_worlds_opened':False,'hardware':platform.platform(),'physical_memory_budget_gib':16,'profile_overhead_included':True,'phases':phases,'identity_digest':records['identity_digest'],'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
