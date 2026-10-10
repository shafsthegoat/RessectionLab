"""One stdlib call profile; invoke only inside the existing owned worker."""
import cProfile
import json
from pathlib import Path
import pstats
import sys
import time

def profiled_call(call, destination, *args, **kwargs):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink() or sys.getprofile() is not None:
        raise ValueError('Fresh profile path and no competing CPython profiler required')
    profiler = cProfile.Profile()
    started = time.perf_counter()
    returned = False
    try:
        profiler.enable(subcalls=True, builtins=True)
        value = call(*args, **kwargs)
        returned = True
        return value
    finally:
        profiler.disable()
        elapsed = time.perf_counter()-started
        stats = pstats.Stats(profiler)
        def key(k): return {'file':k[0], 'line':k[1], 'function':k[2]}
        ordered_self = sorted(stats.stats, key=lambda k:stats.stats[k][2], reverse=True)[:100]
        ordered_cumulative = sorted(stats.stats, key=lambda k:stats.stats[k][3], reverse=True)[:100]
        selected = list(dict.fromkeys(ordered_self+ordered_cumulative))
        rows = []
        for k in selected:
            cc, nc, tt, ct, callers = stats.stats[k]
            edges = sorted(callers.items(), key=lambda item:item[1][3], reverse=True)[:20]
            rows.append({**key(k),'primitive_calls':cc,'total_calls':nc,'self_seconds':tt,
                'cumulative_seconds':ct,'callers':[{'caller':key(c),'counts_and_times':list(v)} for c,v in edges],
                'caller_count':len(callers),'callers_truncated':len(callers)>20})
        record = {'window':'one native greedy call after construction; no replay or independent audit',
            'call_returned':returned,'profiled_wall_seconds':elapsed,'total_calls':stats.total_calls,
            'primitive_calls':stats.prim_calls,'sum_self_seconds':stats.total_tt,
            'function_count':len(stats.stats),'top_self':[key(k) for k in ordered_self],
            'top_cumulative':[key(k) for k in ordered_cumulative], 'functions':rows,
            'limitation':'Instrumentation changes timing. Inclusive times overlap; do not sum them. '
                         'C-extension internals, allocation traffic, cache misses and the live RL path are not resolved.'}
        payload=(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        if len(payload)>2*1024**2: raise ValueError('Profile JSON exceeded 2 MiB')
        with destination.open('xb') as stream: stream.write(payload)
