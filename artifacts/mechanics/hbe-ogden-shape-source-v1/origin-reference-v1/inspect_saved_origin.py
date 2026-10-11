"""Root-authorized arithmetic on the one already exposed DEVELOPMENT fit only."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time

START=time.monotonic()
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
INPUT=ROOT/'artifacts/hbe-v5-axial-calibration-result-v1/execution/fit.json'
EXPECTED='b6b23a297f81b9481a6a0c7b584c8d1d84907baf86dd4ecb62b78474caf9666a'
LIMIT=65536


def need(ok,message):
    if not ok:raise ValueError(message)


def checked(path,expected=None):
    need(not any(p.is_symlink() for p in (path,*path.parents)),'symlink input')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and 0<before.st_size<=LIMIT,'bounded regular input')
        raw=os.read(fd,LIMIT+1)
        after=os.fstat(fd)
        key=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        need(key(before)==key(after)==key(path.lstat()) and len(raw)==before.st_size,'input changed')
    finally:os.close(fd)
    sha=hashlib.sha256(raw).hexdigest()
    need(expected is None or sha==expected,'exact input SHA required')
    return raw,sha


def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def publish(path,raw):
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


def counts(values):return {'positive':sum(x>0 for x in values),'zero':sum(x==0 for x in values),'negative':sum(x<0 for x in values)}


def inspect_branch(saved):
    vectors=[saved[name] for name in ('coordinate','observed','reference_prediction')]
    need(all(type(row) is list and len(row)==30 and all(type(v) in (int,float) and math.isfinite(v) for v in row) for row in vectors),'exact finite30 rows')
    x,y,g=vectors
    need(len(set(x))==30,'distinct coordinates required')
    zero=[i for i in range(30) if x[i]==0]
    nearest=sorted(range(30),key=lambda i:abs(x[i]))[:3]
    sampled=[{'saved_row_index':i,'coordinate_m':x[i],'observed_force_N':y[i],
        'reference_force_N_at_mu1000Pa':g[i],
        'observed_force_over_coordinate_N_per_m':y[i]/x[i] if x[i] else None,
        'reference_force_over_coordinate_N_per_m':g[i]/x[i] if x[i] else None} for i in nearest]
    slopes=[{'near_origin_pair_saved_indices':[i,j],
        'delta_coordinate_m':x[j]-x[i],
        'observed_secant_slope_N_per_m':(y[j]-y[i])/(x[j]-x[i]),
        'reference_secant_slope_N_per_m':(g[j]-g[i])/(x[j]-x[i])} for i,j in zip(nearest,nearest[1:])]
    return {'sample_count':30,'measured_source_sha256':saved['source_sha256'],
        'coordinate_units':'m','force_units':'N','secant_units':'N/m',
        'coordinate_range_m':[min(x),max(x)],'observed_force_range_N':[min(y),max(y)],
        'minimum_absolute_sample_coordinate_m':min(map(abs,x)),
        'exact_zero_coordinate_observed':bool(zero),
        'observed_forces_at_exact_zero_coordinate_N':[y[i] for i in zero],
        'reference_forces_at_exact_zero_coordinate_N':[g[i] for i in zero],
        'coordinate_sign_counts':counts(x),'observed_force_sign_counts':counts(y),
        'force_coordinate_product_sign_counts':counts([a*b for a,b in zip(x,y)]),
        'nearest_three_samples':sampled,'successive_near_origin_secants':slopes,
        'force_offset_estimated':False,'force_offset_subtracted':False,
        'zero_force_offset_identifiable_from_saved_origin':bool(zero),
        'local_tangent_identified':False}


def expired(*_):raise TimeoutError('Saved arithmetic9 second deadline')


def main():
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,9.)
    source,source_sha=checked(Path(__file__).resolve())
    record={'schema':'hbe-saved-origin-reference-receipt-v1','status':'started',
        'source_sha256':source_sha,'input_sha256':EXPECTED,'input_path':str(INPUT.relative_to(ROOT)),
        'new_fits':0,'native_calls':0,'archive_reads':0,'torsion_reads':0,'other_donor_reads':0,
        'wall_cap_seconds':10,'combined_output_cap_bytes':LIMIT}
    thread_keys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
    record['thread_environment']={k:os.environ.get(k) for k in thread_keys}
    need(all(v=='1' for v in record['thread_environment'].values()),'one thread environment required')
    outputs={HERE/'metrics.json',HERE/'receipt.json'}
    need(all(not p.exists() for p in outputs),'attempt exists')
    allowed={INPUT,Path(__file__).resolve(),*outputs}
    def audit(event,args):
        if event in ('subprocess.Popen','os.fork','os.posix_spawn','os.system','socket.__new__'):
            raise PermissionError('No process or network permitted')
        if event=='open' and not isinstance(args[0],int):
            path=Path(os.fsdecode(args[0])).resolve()
            need(path in allowed,'outside exact saved input/source/output')
            mode,flags=args[1:3]
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            need(not writing or path in outputs,'input mutation refused')
    sys.addaudithook(audit)
    try:
        raw,record['input_before_sha256']=checked(INPUT,EXPECTED)
        fit=json.loads(raw)
        need(fit['schema']=='hbe-positive-scale-fit-v1' and set(fit['modes'])=={'compression','tension'}
            and fit['held_out_values_used'] is False and fit['offset_or_sign_fitted'] is False,'saved fit scope differs')
        branches={name:inspect_branch(fit['modes'][name]) for name in ('compression','tension')}
        comp,tens=branches['compression'],branches['tension']
        first=lambda branch:next(row for row in branch['nearest_three_samples'] if row['coordinate_m']!=0)
        c,t=first(comp),first(tens)
        cs=comp['successive_near_origin_secants'][0]['observed_secant_slope_N_per_m']
        ts=tens['successive_near_origin_secants'][0]['observed_secant_slope_N_per_m']
        result={'schema':'hbe-saved-origin-reference-v1','input_sha256':EXPECTED,'specimen':'HBE_01_03',
            'donor_role':'DEVELOPMENT','independent_validation':False,'rows':60,'branches':branches,
            'nearest_nonzero_coordinate_magnitudes_equal_exactly':abs(c['coordinate_m'])==abs(t['coordinate_m']),
            'nearest_nonzero_origin_secant_ratio_compression_over_tension':c['observed_force_over_coordinate_N_per_m']/t['observed_force_over_coordinate_N_per_m'] if t['observed_force_over_coordinate_N_per_m'] else None,
            'first_successive_secant_ratio_compression_over_tension':cs/ts if ts else None,
            'formula':'Origin secant = F/u for each nonzero sample; successive secant = (F_j-F_i)/(u_j-u_i) on nearest three |u| samples. No regression, intercept, subtraction or extrapolated origin.',
            'interpretation_limit':'Finite sampled secants are not an infinitesimal tangent. Force at missing u=0 is unknown; nonzero force at nonzero displacement cannot establish a preload. Reference identification and processing uncertainty are not resolved by these saved rows.',
            'offset_fitted_or_subtracted':False,'alpha_fitted':False,'constitutive_validation_pass':None}
        _,record['input_after_sha256']=checked(INPUT,EXPECTED)
        checked(Path(__file__).resolve(),source_sha)
        data=encoded(result);need(len(source)+len(data)+8192<LIMIT,'bounded result')
        publish(HERE/'metrics.json',data)
        record.update(status='completed_saved_origin_arithmetic',metrics_sha256=hashlib.sha256(data).hexdigest(),metrics_bytes=len(data))
    except Exception as exc:
        record.update(status='failed_saved_origin_arithmetic',error={'type':type(exc).__name__,'message':str(exc)[:512]})
    finally:
        record['elapsed_seconds']=time.monotonic()-START
        record['self_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        if record['elapsed_seconds']>=10:record['status']='failed_total_time_cap'
        receipt=encoded(record);need(len(source)+record.get('metrics_bytes',0)+len(receipt)<LIMIT,'total bound')
        publish(HERE/'receipt.json',receipt);signal.setitimer(signal.ITIMER_REAL,0.)
    print(json.dumps({k:record.get(k) for k in ('status','elapsed_seconds','self_peak_rss_bytes','metrics_sha256','error')}))
    return 0 if record['status']=='completed_saved_origin_arithmetic' else 1


if __name__=='__main__':raise SystemExit(main())
