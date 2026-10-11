"""Exact saved60-row arithmetic only; no fitting, source-curve access or solver."""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FIT=ROOT/'artifacts/hbe-v5-axial-calibration-result-v1/execution/fit.json'
EXPECTED='b6b23a297f81b9481a6a0c7b584c8d1d84907baf86dd4ecb62b78474caf9666a'


def read():
    assert FIT.is_file() and not FIT.is_symlink() and FIT.stat().st_size<=65536
    raw=FIT.read_bytes()
    assert len(raw)<=65536 and hashlib.sha256(raw).hexdigest()==EXPECTED
    return raw


if __name__=='__main__':
    def expired(*_):raise TimeoutError('saved arithmetic10s')
    signal.signal(signal.SIGALRM,expired);signal.alarm(10)
    started=time.monotonic();raw=read();fit=json.loads(raw)
    assert set(fit['modes'])=={'compression','tension'}
    assert all(len(row['coordinate'])==len(row['observed'])==30 for row in fit['modes'].values())
    tension=fit['modes']['tension'];peak=max(tension['observed'])
    negatives=[{'saved_row_index':i,'coordinate_m':x,'force_N':y,
        'absolute_force_percent_of_tensile_peak':100*abs(y)/peak}
        for i,(x,y) in enumerate(zip(tension['coordinate'],tension['observed'])) if y<0]
    assert len(negatives)==4 and peak>0
    weights=tension['weights']
    baseline_rmse=math.sqrt(math.fsum(w*(fit['scale']*g-y)**2 for w,g,y in zip(
        weights,tension['reference_prediction'],tension['observed']))/math.fsum(weights))
    assert raw==read()
    result={'schema':'hbe-saved-negative-force-fraction-v1','input_sha256_before_and_after':EXPECTED,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'specimen':'HBE_01_03','role':'DEVELOPMENT','tension_sample_count':30,'negative_count':4,
        'tensile_peak_force_N':peak,'negative_samples':negatives,
        'maximum_negative_force_percent_of_tensile_peak':max(row['absolute_force_percent_of_tensile_peak'] for row in negatives),
        'baseline_weighted_RMSE_N_reproduction':baseline_rmse,
        'baseline_weighted_RMSE_percent_of_tensile_peak':100*baseline_rmse/peak,
        'formula':'100*abs(F_negative)/max(F_tension); original RMSE=sqrt(sum(w*(fixed_scale*g-y)^2)/sum(w))',
        'noise_bound_inferred':False,'rows_removed':0,'offset_or_sign_fitted':False,
        'new_fits':0,'native_calls':0,'raw_curve_or_archive_reads':0,
        'elapsed_seconds':time.monotonic()-started}
    encoded=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    assert len(encoded)<=16384
    with (HERE/'negative-force-fraction.json').open('xb') as output:
        output.write(encoded);output.flush();os.fsync(output.fileno())
    signal.alarm(0)
    print(encoded.decode())
