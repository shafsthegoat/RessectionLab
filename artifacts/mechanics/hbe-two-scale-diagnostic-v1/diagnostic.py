"""One root-authorized saved DEVELOPMENT model-adequacy diagnostic; stdlib only."""
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
EXPECTED_BASELINE={
    'compression':{'RMSE_N':0.008129197501465504,'MAE_N':0.0050151552480008615,
        'observed_RMS_N':0.02184284507740553,'normalized_RMSE':0.3721675208819034,
        'signed_endpoint_bias_N':0.02179112581061647},
    'tension':{'RMSE_N':0.008641906582590213,'MAE_N':0.007890160444294584,
        'observed_RMS_N':0.003040697833426589,'normalized_RMSE':2.842080027679559,
        'signed_endpoint_bias_N':0.012315256832152465}}


def need(ok,message):
    if not ok:raise ValueError(message)


def raw_checked(path,expected=None):
    need(not any(p.is_symlink() for p in (path,*path.parents)),'symlink input')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and 0<before.st_size<=LIMIT,'bounded regular input')
        with os.fdopen(fd,'rb',buffering=0,closefd=False) as stream:raw=stream.read(LIMIT+1)
        after=os.fstat(fd)
        identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        need(identity(before)==identity(after)==identity(path.lstat()) and len(raw)==before.st_size,'input mutation')
    finally:os.close(fd)
    sha=hashlib.sha256(raw).hexdigest()
    need(expected is None or sha==expected,'input SHA256 mismatch')
    return raw,sha


def encode(obj):return (json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def new_file(path,raw):
    with path.open('xb') as output:output.write(raw);output.flush();os.fsync(output.fileno())


def near(actual,expected):return math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-15)


def finite_vector(values):
    need(type(values) is list and len(values)==30,'exact 30 saved rows per branch')
    need(all(type(v) in (int,float) and math.isfinite(v) for v in values),'finite saved vector')
    return values


def branch_metrics(x,y,g,w,scale):
    total=math.fsum(w);prediction=[scale*v for v in g]
    residual=[p-v for p,v in zip(prediction,y,strict=True)]
    rms=math.sqrt(math.fsum(q*r*r for q,r in zip(w,residual,strict=True))/total)
    observed=math.sqrt(math.fsum(q*v*v for q,v in zip(w,y,strict=True))/total)
    endpoint=max(range(len(x)),key=lambda i:abs(x[i]))
    return {'prediction_N':prediction,'residual_N':residual,'RMSE_N':rms,
        'MAE_N':math.fsum(q*abs(r) for q,r in zip(w,residual,strict=True))/total,
        'observed_RMS_N':observed,'normalized_RMSE':rms/observed,
        'signed_endpoint_bias_N':residual[endpoint],
        'quadrature_weighted_SSE_N2':math.fsum(q*r*r for q,r in zip(w,residual,strict=True))}


def timeout(*_):raise TimeoutError('9 second work deadline; 10 second total cap')


def main():
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,9.)
    source_raw,source_sha=raw_checked(Path(__file__))
    receipt={'schema':'hbe-development-two-scale-diagnostic-receipt-v1','status':'started',
        'source_sha256':source_sha,'input':{'path':str(INPUT.relative_to(ROOT)),'sha256':EXPECTED},
        'wall_cap_seconds':10,'source_metrics_receipt_cap_bytes':LIMIT,
        'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
        'solver_calls':0,'archive_reads':0,'torsion_reads':0,'other_donor_reads':0,
        'checkpoint_reads':0,'parameter_sweeps':0,'fit_before_sha256':None,'fit_after_sha256':None}
    need(all(v=='1' for v in receipt['thread_environment'].values()),'single thread environment required')
    need(not (HERE/'metrics.json').exists() and not (HERE/'receipt.json').exists(),'attempt already exists')
    allowed={INPUT,Path(__file__).resolve(),HERE/'metrics.json',HERE/'receipt.json'}
    def audit(event,args):
        if event in ('subprocess.Popen','os.fork','os.posix_spawn','os.system','socket.__new__'):
            raise PermissionError('diagnostic cannot spawn or network')
        if event=='open' and not isinstance(args[0],int):
            path=Path(os.fsdecode(args[0])).resolve()
            need(path in allowed,'open outside exact source/saved fit/outputs')
            mode,flags=args[1],args[2]
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
            need(not writing or path in {HERE/'metrics.json',HERE/'receipt.json'},'input write forbidden')
    sys.addaudithook(audit)
    try:
        raw,receipt['fit_before_sha256']=raw_checked(INPUT,EXPECTED)
        fit=json.loads(raw,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
        need(fit['schema']=='hbe-positive-scale-fit-v1' and set(fit['modes'])=={'compression','tension'},'fit schema')
        need(fit['fitted_parameters']==['positive_mu_scale_only'] and fit['held_out_values_used'] is False
             and fit['offset_or_sign_fitted'] is False,'original fit scope')
        need(near(fit['scale'],.715361571139082) and near(fit['mu_Pa'],715.361571139082)
             and fit['reference_mu_Pa']==1000.,'fixed baseline identity')
        branches={};numerators=[];denominators=[]
        for name in ('compression','tension'):
            saved=fit['modes'][name]
            x,y,g,w=[finite_vector(saved[key]) for key in ('coordinate','observed','reference_prediction','weights')]
            need(all(v>0 for v in w) and near(math.fsum(w),.5),'original equal-mode quadrature weights')
            num=math.fsum(q*a*b for q,a,b in zip(w,g,y,strict=True))
            den=math.fsum(q*a*a for q,a in zip(w,g,strict=True))
            need(den>fit['denominator_minimum'],'identified branch denominator')
            scale=num/den;need(math.isfinite(scale) and scale>0,'positive branch scale required')
            baseline=branch_metrics(x,y,g,w,fit['scale']);separate=branch_metrics(x,y,g,w,scale)
            need(all(near(baseline[key],value) for key,value in EXPECTED_BASELINE[name].items()),'saved baseline metrics reproduction')
            need(separate['quadrature_weighted_SSE_N2']<=baseline['quadrature_weighted_SSE_N2']+1e-15,'LS diagnostic objective')
            numerators.append(num);denominators.append(den)
            branches[name]={'row_count':30,'source_sha256':saved['source_sha256'],
                'coordinate_m':x,'observed_force_N':y,'reference_force_N':g,'weights':w,
                'coefficient_numerator_N2':num,'coefficient_denominator_N2':den,
                'branch_scale':scale,'branch_effective_mu_Pa':1000.*scale,
                'baseline':baseline,'separate_scale_diagnostic':separate}
        common=math.fsum(numerators)/math.fsum(denominators)
        need(near(math.fsum(numerators),fit['numerator']) and near(math.fsum(denominators),fit['denominator'])
             and near(common,fit['scale']),'frozen common-scale reproduction')
        ratio=branches['compression']['branch_scale']/branches['tension']['branch_scale']
        result={'schema':'hbe-development-two-scale-model-adequacy-v1','specimen':'HBE_01_03',
            'donor_role':'DEVELOPMENT','training_or_independent_validation':False,
            'scope':'post-hoc diagnostic of common-amplitude adequacy on the same two calibration branches',
            'input_sha256':EXPECTED,'source_sha256':source_sha,'rows':60,
            'formula':'a_b=sum_i(w_i*g_i*y_i)/sum_i(w_i*g_i*g_i); original w sums to 0.5 per branch',
            'baseline_scale':fit['scale'],'baseline_mu_Pa':fit['mu_Pa'],
            'reproduced_common_scale':common,'branch_scale_ratio_compression_over_tension':ratio,
            'branches':branches,'new_branch_coefficients':2,'new_native_solves':0,
            'baseline_equal_branch_RMSE_N':math.sqrt(math.fsum(b['baseline']['RMSE_N']**2 for b in branches.values())/2),
            'diagnostic_equal_branch_RMSE_N':math.sqrt(math.fsum(b['separate_scale_diagnostic']['RMSE_N']**2 for b in branches.values())/2),
            'baseline_reproductions_passed':True,'physical_validation_pass':None,
            'held_out_improvement_claim':False,'constitutive_law_admitted':False,
            'uncertainty_or_row_IID_test':None,
            'interpretation':'Separate coefficients diagnose amplitude incompatibility; lower calibration error with an extra coefficient is not independent validation. Geometry/bonding/preprocessing remain alternative causes. No living-brain retraction/cutting claim.'}
        _,receipt['fit_after_sha256']=raw_checked(INPUT,EXPECTED)
        raw_checked(Path(__file__),source_sha)
        data=encode(result)
        need(len(source_raw)+len(data)+8192<=LIMIT,'combined output cap with receipt reserve')
        new_file(HERE/'metrics.json',data)
        receipt.update(status='completed_saved_development_diagnostic',metrics_sha256=hashlib.sha256(data).hexdigest(),
            metrics_bytes=len(data),source_bytes=len(source_raw),baseline_reproductions_passed=True)
    except Exception as error:
        receipt.update(status='failed_saved_diagnostic',error={'type':type(error).__name__,'message':str(error)[:512]})
    finally:
        receipt['elapsed_seconds']=time.monotonic()-START
        receipt['self_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        if receipt['elapsed_seconds']>=10:receipt['status']='failed_total_time_cap'
        encoded=encode(receipt)
        need(len(source_raw)+receipt.get('metrics_bytes',0)+len(encoded)<=LIMIT,'total file cap')
        new_file(HERE/'receipt.json',encoded)
        signal.setitimer(signal.ITIMER_REAL,0.)
    print(json.dumps({k:receipt.get(k) for k in ('status','elapsed_seconds','self_peak_rss_bytes','metrics_sha256','error')}))
    return 0 if receipt['status']=='completed_saved_development_diagnostic' else 1


if __name__=='__main__':raise SystemExit(main())
