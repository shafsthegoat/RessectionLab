"""Analytical fixtures only: no released RHUH row is read or fitted here."""
import csv
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from scripts import run_rhuh_preoperative_baseline as runner
from resectionlab import rhuh_outcomes as adapter

ROOT=Path(__file__).resolve().parents[1]


def study():return json.loads((ROOT/runner.DECLARATION_PATH).read_text())


def artificial_csv():
    # IDs identify the allowed fixture schema only; every value below is invented
    # for software testing, independently of the downloaded clinical table.
    categories=['No']*26+['Transient']*6+['Minor Persistent']*6+['Major Persistent']*2
    out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=adapter.RHUH_HEADERS);writer.writeheader()
    for i,category in enumerate(categories):
        row={key:'' for key in adapter.RHUH_HEADERS}
        row.update({'Patient ID':f'RHUH-{i+1:04d}','Preoperative KPS':str(60+10*(i%4)),
            'Postoperative KPS':str(50+10*(i%5)),adapter.PREOPERATIVE_CE_VOLUME_FIELD:str(1+i%7),
            'Age':str(30+i),'Postoperative Neurological Deficit':category})
        writer.writerow(row)
    return out.getvalue().encode()


def test_shadow_preserves_targets_features_and_hashes_but_not_mixed_source():
    raw=artificial_csv();shadow=runner.prohibited_shadow(raw)
    original=runner.projection(adapter,raw,runner.digest(raw),study())
    changed=runner.projection(adapter,shadow,runner.digest(shadow),study())
    assert runner.digest(raw)!=runner.digest(shadow)
    runner.validate_invariance(original,changed)
    assert original['projection_records']==changed['projection_records']
    changed['X'][0,0]+=1
    with pytest.raises(ValueError,match='features'):runner.validate_invariance(original,changed)


def test_projection_never_maps_missing_target_to_negative():
    raw=artificial_csv().replace(b',No,',b',,',1)
    with pytest.raises(ValueError,match='cohort/categories'):
        runner.projection(adapter,raw,runner.digest(raw),study())


def analytical_arrays():
    return np.array([[1.,2.],[2.,3.],[3.,2.],[4.,1.],[5.,4.],[80.,50.]]),np.array([0,1,0,1,0,1])


def test_real_sklearn_fold_scaler_is_train_only_and_shadow_identical():
    pytest.importorskip('sklearn', reason='Requires the isolated RHUH research runtime')
    X,y=analytical_arrays();rows=[];counts={'attempted':0,'completed':0}
    output=runner.loo(X,y,[f'synthetic-{i}' for i in range(6)],study(),deadline=runner.time.monotonic()+15,
        counts=counts,on_fold=rows.append,job='synthetic',shadow_X=X.copy())
    assert len(rows)==6 and counts=={'attempted':12,'completed':12}
    assert np.array_equal(rows[-1]['models']['KPS_CE_VOLUME']['scaler_mean'],X[:-1].mean(axis=0))
    assert rows[-1]['models']['KPS_CE_VOLUME']['scaler_mean']!=X.mean(axis=0).tolist()
    assert rows[-1]['training_patient_ids']==[f'synthetic-{i}' for i in range(5)]
    assert all(model['shadow_prediction_identical'] for row in rows for name,model in row['models'].items() if name!='FOLD_PREVALENCE')
    assert output['FOLD_PREVALENCE']==[(3-int(label))/5 for label in y]


def test_metrics_ties_null_precision_and_prevalence_auc_rule():
    result=runner.metrics([0,0,1,1],{'FOLD_PREVALENCE':[2/3,2/3,1/3,1/3],
        'KPS':[.5,.5,.5,.5],'KPS_CE_VOLUME':[.2,.2,.2,.2]})
    assert result['models']['FOLD_PREVALENCE']['roc_auc'] is None
    assert result['models']['KPS']['roc_auc']==.5
    assert result['models']['KPS']['confusion']=={'TN':0,'FP':2,'FN':0,'TP':2}
    assert result['models']['KPS_CE_VOLUME']['precision']=={'value':None,'null_reason':'zero_denominator'}
    with pytest.raises(ValueError,match='Complete'):
        runner.metrics([0,0,1,1],{'FOLD_PREVALENCE':[.5]*3})


@pytest.mark.parametrize('failure',['warning','limit','nonfinite','deadline'])
def test_failed_fit_is_recorded_without_continuing(failure):
    pytest.importorskip('sklearn', reason='Requires the isolated RHUH research runtime')
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
    X,y=analytical_arrays();rows=[];counts={'attempted':0,'completed':0}
    class Model:
        def fit(self,X,y):
            self.coef_=np.array([[np.nan if failure=='nonfinite' else 0.]])
            self.intercept_=np.array([0.]);self.classes_=np.array([0,1]);self.n_iter_=np.array([1000 if failure=='limit' else 1])
            if failure=='warning':warnings.warn('synthetic convergence failure',ConvergenceWarning)
        def predict_proba(self,X):return np.array([[.5,.5]])
    with pytest.raises((ValueError,RuntimeError,TimeoutError)):
        runner.loo(X,y,[str(i) for i in range(6)],study(),deadline=runner.time.monotonic()+(-1 if failure=='deadline' else 10),
            counts=counts,on_fold=rows.append,job='analytical-failure',factory=lambda:(StandardScaler(),Model()))
    assert counts['completed']==0
    if failure=='deadline':assert counts['attempted']==0 and rows==[]
    else:assert counts['attempted']==1 and len(rows)==1 and rows[0]['status']=='failed'


def test_complete_fixed_diagnostic_sequence_uses_shared_permuted_labels(tmp_path,monkeypatch):
    raw=artificial_csv();data=runner.projection(adapter,raw,runner.digest(raw),study());calls=[]
    def fake_loo(X,y,ids,config,*,counts,job,on_fold,**kwargs):
        calls.append((job,y.copy(),ids.copy()))
        counts['attempted']+=2*len(y);counts['completed']+=2*len(y)
        on_fold({'job':job,'status':'completed','synthetic_only':True})
        return {name:[.35]*len(y) for name in runner.MODELS}
    monkeypatch.setattr(runner,'loo',fake_loo)
    result=runner.evaluate(data,data,study(),tmp_path,runner.time.monotonic()+15)
    assert result['status']=='completed_observational_baseline' and result['fits']['completed']==9092
    assert len(calls)==114 and sum(job.startswith('permutation:') for job,_,_ in calls)==99
    rng=np.random.Generator(np.random.PCG64(20261005))
    for index in range(99):assert np.array_equal(calls[1+index][1],rng.permutation(data['y']))
    deletions=calls[100:]
    assert [job for job,_,_ in deletions]==['delete:'+data['ids'][i] for i in np.flatnonzero(data['y'])]
    assert all(len(labels)==39 and labels.sum()==13 for _,labels,_ in deletions)
    assert result['permutation']['p_value']==1 and result['influence']['summary']['zero']==14


def test_incomplete_permutation_keeps_fixed_denominator_and_null_p(tmp_path,monkeypatch):
    raw=artificial_csv();data=runner.projection(adapter,raw,runner.digest(raw),study());jobs=[]
    def fake_loo(X,y,ids,config,*,counts,job,on_fold,**kwargs):
        jobs.append(job)
        if job=='permutation:1':raise RuntimeError('synthetic terminal failure')
        counts['attempted']+=2*len(y);counts['completed']+=2*len(y)
        return {name:[.35]*len(y) for name in runner.MODELS}
    monkeypatch.setattr(runner,'loo',fake_loo)
    with pytest.raises(RuntimeError,match='terminal'):
        runner.evaluate(data,data,study(),tmp_path,runner.time.monotonic()+15)
    result=json.loads((tmp_path/'state.json').read_text())
    assert result['permutation']['planned']==99 and result['permutation']['completed']==1
    assert result['permutation']['p_value'] is None and result['influence']['completed']==0
    assert jobs==['observed','permutation:0','permutation:1']


def test_unauthorized_release_stops_before_adapter_runtime_or_csv(tmp_path,monkeypatch):
    (tmp_path/'manifest.json').write_bytes((ROOT/runner.DECLARATION_PATH).read_bytes())
    release=tmp_path/'release.json';release.write_text('{"authorized":false}')
    monkeypatch.setattr(runner.importlib,'import_module',lambda *a:pytest.fail('No import before release'))
    with pytest.raises(ValueError,match='Explicit separately'):
        runner.preflight(tmp_path,{'path':'manifest.json','sha256':runner.DECLARATION_SHA},
                         {'path':'release.json','sha256':runner.supervisor.sha(release)})


def test_source_drift_and_no_overwrite_guard(tmp_path,monkeypatch):
    target=tmp_path/'bound';target.write_text('before');inputs={str(target):runner.supervisor.sha(target)}
    target.write_text('after')
    with pytest.raises(ValueError,match='changed'):runner.recheck(inputs)
    output=tmp_path/'outputs'/'attempt';output.mkdir(parents=True)
    monkeypatch.setattr(runner,'preflight',lambda *a:{'directory':str(output)})
    monkeypatch.setattr(runner.supervisor,'supervise',lambda *a,**k:pytest.fail('No overwrite'))
    with pytest.raises(FileExistsError):runner.launch(tmp_path,{}, {})
