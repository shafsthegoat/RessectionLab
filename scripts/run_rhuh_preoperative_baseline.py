#!/usr/bin/env python3
"""One declared RHUH observational baseline; no tuning, route reward or resume."""
from __future__ import annotations
import time
PROCESS_STARTED = time.monotonic()
import argparse
from collections import Counter
import csv
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import warnings

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from scripts import febio_runtime as supervisor

DECLARATION_PATH = 'manifests/experiments/rhuh-preoperative-baseline-v1.json'
DECLARATION_SHA = 'c73547faa86074a1293787c56f623a8d665ee61455085dd7eccb50be8857e87b'
MODELS = ('FOLD_PREVALENCE', 'KPS', 'KPS_CE_VOLUME')
FEATURES = ('preoperative_kps', 'preoperative_ce_volume_cm3')
PROTECTED_COLUMNS = ('Patient ID', 'Postoperative Neurological Deficit', 'Preoperative KPS',
                     'Preoperative  contrast enhancing tumor volume (cm3)')
THREADS = {key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def bound(root, record, *, json_value=False):
    path = Path(record['path'])
    path = path if path.is_absolute() else root/path
    path = path.resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or supervisor.sha(path) != record['sha256']:
        raise ValueError('Bound local input differs: '+str(path))
    return json.loads(path.read_text()) if json_value else path


def recheck(inputs):
    for path, expected in inputs.items():
        if supervisor.sha(path) != expected:
            raise ValueError('Source or input changed: '+path)


def emit(path, value):
    """One durable logical completed fold/job; no overwrite or hidden retries."""
    with path.open('a') as stream:
        stream.write(canonical(value).decode()+'\n'); stream.flush(); os.fsync(stream.fileno())


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Declared worker allowance exhausted')


def projection(adapter, raw, expected_sha, study):
    import numpy as np
    dataset = adapter.parse_rhuh_csv(raw, expected_sha256=expected_sha)
    records = tuple(dataset.records)
    ids = [record.patient_id for record in records]
    categories = [record.deficit_category for record in records]
    labels = [record.any_recorded_postoperative_deficit for record in records]
    if (ids != study['cohort']['ordered_patient_ids']
            or dict(Counter(categories)) != study['endpoint']['expected_raw_counts']
            or any(type(value) is not bool for value in labels)):
        raise ValueError('Exact declared cohort/categories and nonmissing binary endpoints required')
    projected = [record.preoperative_projection() for record in records]
    X = np.array([[getattr(value, name) for name in FEATURES] for value in projected], dtype=float)
    y = np.array(labels, dtype=int)
    if not np.isfinite(X).all() or X.shape != (40,2):
        raise ValueError('Exactly forty complete finite two-feature projections required')
    if dict(Counter(map(str,y.tolist()))) != study['endpoint']['expected_binary_counts']:
        raise ValueError('Binary endpoint counts differ')
    return {'ids':ids,'categories':categories,'X':X,'y':y,
            'projection_hashes':[value.semantic_hash for value in projected],
            'projection_records':[value.to_dict() for value in projected]}


def prohibited_shadow(raw):
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    rows = list(reader); fields = reader.fieldnames
    if len(rows) != 40 or any(name not in fields for name in PROTECTED_COLUMNS):
        raise ValueError('Shadow control requires the exact four protected columns and40rows')
    changed = [{name:row[name] if name in PROTECTED_COLUMNS else rows[(i+1)%40][name]
                for name in fields} for i,row in enumerate(rows)]
    stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader(); writer.writerows(changed)
    return stream.getvalue().encode('utf-8')


def validate_invariance(original, shadow):
    import numpy as np
    for key in ('ids','categories','projection_hashes'):
        if original[key] != shadow[key]:
            raise ValueError('Prohibited-field control changed '+key)
    if not np.array_equal(original['X'], shadow['X']) or not np.array_equal(original['y'], shadow['y']):
        raise ValueError('Prohibited-field control changed permitted features or fixed labels')


def sklearn_factory(study):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cfg = study['logistic']
    def create():
        scaler = StandardScaler(with_mean=True, with_std=True)
        model = LogisticRegression(penalty='l2', C=cfg['C'], solver='lbfgs', fit_intercept=True,
            tol=cfg['tol'], max_iter=cfg['max_iter'], class_weight=None, random_state=0, warm_start=False)
        return scaler, model
    return create


def finite(values, name):
    import numpy as np
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise ValueError('Nonfinite '+name)


def loo(X, y, ids, study, *, deadline, counts, on_fold, job, factory=None, shadow_X=None):
    """Every heldout record is predicted by newly fitted training-fold transforms."""
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    factory = factory or sklearn_factory(study)
    n = len(ids); probabilities = {name:[] for name in MODELS}
    for holdout in range(n):
        check_deadline(deadline)
        train = np.arange(n) != holdout
        if len(set(y[train].tolist())) != 2:
            raise ValueError('Training fold must retain both endpoint classes')
        row = {'job':job,'holdout_index':holdout,'holdout_patient_id':ids[holdout],
               'training_patient_ids':[ids[i] for i in range(n) if train[i]],
               'target':int(y[holdout]),'models':{}}
        try:
            probability = float(np.mean(y[train])); probabilities[MODELS[0]].append(probability)
            row['models'][MODELS[0]] = {'probability':probability,'training_positive_count':int(y[train].sum()),
                'training_count':int(train.sum()),'fit':False,'elapsed_seconds':0.,
                'shadow_prediction_identical':shadow_X is not None}
            for model_id, width in ((MODELS[1],1),(MODELS[2],2)):
                check_deadline(deadline)
                if counts['attempted'] >= study['resources']['max_logistic_fits']:
                    raise RuntimeError('Declared logistic fit allocation exhausted')
                row['active_model'] = model_id; counts['attempted'] += 1
                started = time.monotonic(); scaler, model = factory()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    trained = scaler.fit_transform(X[train,:width])
                    model.fit(trained, y[train])
                check_deadline(deadline)
                messages = [{'category':warning.category.__name__,'message':str(warning.message)} for warning in caught]
                row['models'][model_id] = {'features':list(FEATURES[:width]),'warnings':messages,
                                          'elapsed_seconds':time.monotonic()-started}
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise RuntimeError('Logistic convergence warning')
                for value, name in ((model.coef_,'coefficients'),(model.intercept_,'intercept'),
                                    (scaler.mean_,'scaler mean'),(scaler.scale_,'scaler scale'),(scaler.var_,'scaler variance')):
                    finite(value,name)
                iterations = np.asarray(model.n_iter_,dtype=int)
                if (iterations < 0).any() or (iterations >= study['logistic']['max_iter']).any():
                    raise RuntimeError('Logistic iteration limit reached')
                if model.classes_.tolist() != [0,1]:
                    raise ValueError('Binary model class order differs')
                probability = float(model.predict_proba(scaler.transform(X[holdout:holdout+1,:width]))[0,1])
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError('Invalid heldout probability')
                if shadow_X is not None:
                    shadow_p = float(model.predict_proba(scaler.transform(shadow_X[holdout:holdout+1,:width]))[0,1])
                    if probability != shadow_p:
                        raise ValueError('Prohibited-field heldout prediction changed')
                counts['completed'] += 1
                probabilities[model_id].append(probability)
                row['models'][model_id].update(probability=probability,scaler_mean=scaler.mean_.tolist(),
                    scaler_scale=scaler.scale_.tolist(),scaler_variance=scaler.var_.tolist(),
                    coefficients=model.coef_.tolist(),intercept=model.intercept_.tolist(),iterations=iterations.tolist(),
                    converged=True,shadow_prediction_identical=shadow_X is not None,elapsed_seconds=time.monotonic()-started)
            for model_record in row['models'].values():
                p = model_record['probability']
                model_record.update(predicted=int(p >= .5),squared_error=(p-int(y[holdout]))**2)
            row['status'] = 'completed'; row.pop('active_model',None)
        except BaseException as error:
            row.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
            on_fold(row)
            raise
        on_fold(row)
    return probabilities


def metrics(y, predictions):
    import numpy as np
    y = np.asarray(y,dtype=int)
    if len(set(y.tolist())) != 2:
        raise ValueError('Pooled metrics need both declared classes')
    result = {}
    for model_id in MODELS:
        p = np.asarray(predictions[model_id],dtype=float)
        if p.shape != y.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('Complete finite heldout predictions required')
        pred = p >= .5
        tn,fp,fn,tp = (int(np.sum(mask)) for mask in ((y==0)&~pred,(y==0)&pred,(y==1)&~pred,(y==1)&pred))
        def ratio(num,den):
            return {'value':num/den if den else None,'null_reason':None if den else 'zero_denominator'}
        auc = None if model_id == MODELS[0] else float(np.mean((p[y==1,None]>p[None,y==0])+.5*(p[y==1,None]==p[None,y==0])))
        result[model_id] = {'brier':float(np.mean((p-y)**2)),'roc_auc':auc,
            'auc_null_reason':'LOO_prevalence_inverse_label_ordering_artifact' if model_id==MODELS[0] else None,
            'confusion':dict(TN=tn,FP=fp,FN=fn,TP=tp),'sensitivity':ratio(tp,tp+fn),'specificity':ratio(tn,tn+fp),
            'precision':ratio(tp,tp+fp),'negative_predictive_value':ratio(tn,tn+fn),
            'balanced_accuracy':.5*(tp/(tp+fn)+tn/(tn+fp)),'accuracy':(tp+tn)/len(y),'n':len(y)}
    return {'models':result,'paired_brier':{'KPS_CE_VOLUME_minus_KPS':result[MODELS[2]]['brier']-result[MODELS[1]]['brier'],
        'KPS_minus_FOLD_PREVALENCE':result[MODELS[1]]['brier']-result[MODELS[0]]['brier'],
        'KPS_CE_VOLUME_minus_FOLD_PREVALENCE':result[MODELS[2]]['brier']-result[MODELS[0]]['brier']},
        'confidence_intervals':None,'clinical_calibration_validated':False}


def evaluate(data, shadow, study, output, deadline, *, factory=None):
    """Fixed observed→99null→14event-deletion sequence; partial state stays honest."""
    import numpy as np
    counts = {'attempted':0,'completed':0}
    state = {'status':'running','fits':counts,'observed':None,'permutation':{'planned':99,'completed':0,'p_value':None},
             'influence':{'planned':14,'completed':0,'summary':None},'failure':None}
    def save():supervisor.write_json(output/'state.json',state)
    save(); started=time.monotonic()
    try:
        validate_invariance(data,shadow)
        X,y,ids = data['X'],data['y'],data['ids']
        observed = loo(X,y,ids,study,deadline=deadline,counts=counts,job='observed',factory=factory,
            shadow_X=shadow['X'],on_fold=lambda row:emit(output/'observed-folds.jsonl',row))
        state['observed'] = metrics(y,observed); state['observed_seconds']=time.monotonic()-started;save()
        rng=np.random.Generator(np.random.PCG64(20261005)); nulls=[]; stage=time.monotonic()
        for index in range(99):
            labels=rng.permutation(y); folds=[]
            # Compact per-job receipts retain model coefficients/predictions while
            # training IDs are reversible from the shared ordered IDs and holdout.
            def compact(row):
                row.pop('training_patient_ids',None);row.pop('holdout_patient_id',None)
                emit(output/'permutation-folds.jsonl',row);folds.append(row)
            try:
                predictions=loo(X,labels,ids,study,deadline=deadline,counts=counts,on_fold=compact,
                                job=f'permutation:{index}',factory=factory)
                report=metrics(labels,predictions); statistic=report['paired_brier']['KPS_CE_VOLUME_minus_FOLD_PREVALENCE']
                nulls.append(statistic);state['permutation']['completed']+=1
                emit(output/'permutations.jsonl',{'index':index,'label_vector':labels.tolist(),'status':'completed',
                    'statistic':statistic,'metrics':report,'fold_record_count':len(folds),
                    'folds_ledger':'permutation-folds.jsonl'});save()
            except BaseException as error:
                emit(output/'permutations.jsonl',{'index':index,'label_vector':labels.tolist(),'status':'failed',
                    'error':{'type':type(error).__name__,'message':str(error)},'fold_record_count':len(folds),
                    'folds_ledger':'permutation-folds.jsonl'});raise
        statistic=state['observed']['paired_brier']['KPS_CE_VOLUME_minus_FOLD_PREVALENCE']
        state['permutation'].update(p_value=(1+sum(v<=statistic for v in nulls))/100,
                                    statistic=statistic,elapsed_seconds=time.monotonic()-stage);save()
        diffs=[];stage=time.monotonic()
        for deleted in np.flatnonzero(y==1):
            keep=np.arange(len(y))!=deleted; kept_ids=[pid for i,pid in enumerate(ids) if keep[i]]; folds=[]
            try:
                prediction=loo(X[keep],y[keep],kept_ids,study,deadline=deadline,counts=counts,
                    on_fold=lambda row:(emit(output/'influence-folds.jsonl',row),folds.append(row)),
                    job='delete:'+ids[deleted],factory=factory)
                report=metrics(y[keep],prediction);difference=report['paired_brier']['KPS_CE_VOLUME_minus_KPS'];diffs.append(difference)
                state['influence']['completed']+=1
                emit(output/'influence.jsonl',{'deleted_patient_id':ids[deleted],'status':'completed','metrics':report,
                    'paired_difference':difference,'change_from_full_cohort':difference-state['observed']['paired_brier']['KPS_CE_VOLUME_minus_KPS'],
                    'fold_record_count':len(folds),'folds_ledger':'influence-folds.jsonl'});save()
            except BaseException as error:
                emit(output/'influence.jsonl',{'deleted_patient_id':ids[deleted],'status':'failed',
                    'error':{'type':type(error).__name__,'message':str(error)},'fold_record_count':len(folds),
                    'folds_ledger':'influence-folds.jsonl'});raise
        if len(diffs)!=14 or counts['completed'] != study['resources']['max_logistic_fits']:
            raise ValueError('Exact declared diagnostic/fit denominator incomplete')
        state['influence'].update(summary={'minimum':min(diffs),'maximum':max(diffs),
            'negative':sum(v<0 for v in diffs),'zero':sum(v==0 for v in diffs),'positive':sum(v>0 for v in diffs)},
            elapsed_seconds=time.monotonic()-stage)
        check_deadline(deadline);state['status']='completed_observational_baseline'
    except BaseException as error:
        state.update(status='failed_or_incomplete',failure={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        state['evaluation_elapsed_seconds']=time.monotonic()-started;save()
    return state


def preflight(root, study_binding, release_binding):
    root=root.resolve(); inputs={}
    def check(record, *, json_value=False):
        path=bound(root,record); inputs[str(path)]=record['sha256']
        return json.loads(path.read_text()) if json_value else path
    if study_binding['sha256'] != DECLARATION_SHA:
        raise ValueError('Only the frozen RHUH protocol is supported')
    study=check(study_binding,json_value=True);release=check(release_binding,json_value=True)
    if (release.get('schema')!='rhuh-baseline-release-v1' or release.get('authorized') is not True
            or release.get('study')!=study_binding):
        raise ValueError('Explicit separately committed root release required')
    adapter=importlib.import_module('resectionlab.rhuh_outcomes')
    package=importlib.import_module('resectionlab')
    expected={'runner':Path(__file__).resolve(),'adapter':Path(adapter.__file__).resolve(),
              'package_init':Path(package.__file__).resolve(),'supervisor':Path(supervisor.__file__).resolve()}
    sources=release['source_bindings']
    if set(sources)!=set(expected):raise ValueError('Exact four-source execution closure required')
    for key,path in expected.items():
        if check(sources[key])!=path:raise ValueError('Imported source origin differs: '+key)
    wanted={'scripts/run_rhuh_preoperative_baseline.py':sources['runner']['sha256'],
            'scripts/febio_runtime.py':sources['supervisor']['sha256'],
            'src/resectionlab/rhuh_outcomes.py':sources['adapter']['sha256'],
            'src/resectionlab/__init__.py':sources['package_init']['sha256'],DECLARATION_PATH:DECLARATION_SHA}
    seen=set()
    with tarfile.open(check(release['source_archive']),'r:*') as archive:
        if archive.pax_headers.get('comment')!=release['source_commit']:raise ValueError('Source commit differs')
        for member in archive:
            if member.isdir():continue
            if member.name not in wanted or member.name in seen or not member.isfile() or member.size>1024**2:
                raise ValueError('Unexpected archive source member')
            stream=archive.extractfile(member)
            if stream is None or digest(stream.read())!=wanted[member.name]:raise ValueError('Archived source differs')
            seen.add(member.name)
    if seen!=set(wanted):raise ValueError('Incomplete exact source archive')
    python=Path(release['interpreter']['path']).resolve()
    if python!=Path(sys.executable).resolve() or supervisor.sha(python)!=release['interpreter']['sha256']:
        raise ValueError('Bound interpreter differs')
    inputs[str(python)]=release['interpreter']['sha256']
    dependency=check(release['dependency_manifest'],json_value=True)
    directory=Path(dependency['directory']).resolve()
    if not directory.is_relative_to(root) or not directory.is_dir():raise ValueError('Private dependency directory required')
    inventories={'isolated':{'directory':str(directory),'files':dependency['files']}}
    if set(dependency.get('shared_packages',{}))!={'numpy','scipy'}:
        raise ValueError('Exact shared numerical runtime inventory required')
    inventories.update(dependency['shared_packages'])
    for name,inventory in inventories.items():
        folder=Path(inventory['directory']).resolve()
        if name!='isolated' and inventory['version']!=dependency['packages'][name]:
            raise ValueError('Shared numerical runtime version differs')
        actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        if actual!=set(inventory['files']):raise ValueError('Numerical dependency inventory changed: '+name)
        for filename,expected_sha in inventory['files'].items():
            path=(folder/filename).resolve()
            if not path.is_relative_to(folder) or not path.is_file() or supervisor.sha(path)!=expected_sha:
                raise ValueError('Numerical runtime file changed: '+str(path))
            inputs[str(path)]=expected_sha
    check({'path':study['source']['local_path'],'sha256':study['source']['sha256']})
    output=root/release['output_directory']
    if not output.resolve().is_relative_to(root/'outputs') or output.resolve()==root/'outputs':
        raise ValueError('Fresh outputs subdirectory required')
    return {'study':study,'study_binding':study_binding,'release_binding':release_binding,'inputs':inputs,
            'dependency':dependency,'directory':str(output.resolve()),'source_bindings':sources}


def runtime_record(plan):
    import contextlib
    import importlib.metadata
    import platform
    import numpy as np
    import scipy
    import sklearn
    import threadpoolctl
    directory=Path(plan['dependency']['directory']).resolve()
    if not Path(sklearn.__file__).resolve().is_relative_to(directory):raise ValueError('Wrong sklearn import origin')
    for name,module in (('numpy',np),('scipy',scipy)):
        expected=Path(plan['dependency']['shared_packages'][name]['directory']).resolve()/'__init__.py'
        if Path(module.__file__).resolve()!=expected:
            raise ValueError('Wrong shared numerical import origin: '+name)
    for name in ('joblib','threadpoolctl','narwhals'):
        module=importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(directory):
            raise ValueError('Wrong isolated dependency import origin: '+name)
    versions={name:importlib.metadata.version(name) for name in plan['dependency']['packages']}
    if versions!=plan['dependency']['packages']:raise ValueError('Numerical package versions differ')
    pools=threadpoolctl.threadpool_info()
    if any(row['num_threads']!=1 for row in pools) or any(os.environ.get(k)!='1' for k in THREADS):
        raise ValueError('Single numerical thread required')
    configuration=io.StringIO()
    with contextlib.redirect_stdout(configuration):np.show_config()
    return {'python':platform.python_version(),'platform':platform.platform(),'packages':versions,
            'numerical_configuration':configuration.getvalue(),'threadpools':pools,
            'thread_environment':{key:os.environ.get(key) for key in THREADS},
            'loaded_module_files':{m.__name__:{'path':m.__file__,'sha256':supervisor.sha(m.__file__)} for m in (np,scipy,sklearn)}}


def worker(root, plan):
    output=Path(plan['directory']); deadline=PROCESS_STARTED+plan['study']['resources']['worker_wall_seconds']
    check_deadline(deadline)
    sys.path.insert(0,plan['dependency']['directory'])
    record=runtime_record(plan);supervisor.write_json(output/'runtime.json',record)
    adapter=importlib.import_module('resectionlab.rhuh_outcomes')
    raw=(root/plan['study']['source']['local_path']).read_bytes()
    data=projection(adapter,raw,plan['study']['source']['sha256'],plan['study'])
    shadow_raw=prohibited_shadow(raw);shadow=projection(adapter,shadow_raw,digest(shadow_raw),plan['study'])
    validate_invariance(data,shadow)
    supervisor.write_json(output/'cohort.json',{'patient_ids':data['ids'],'raw_categories':data['categories'],
        'labels':data['y'].tolist(),'feature_order':list(FEATURES),'features':data['X'].tolist(),
        'projection_hashes':data['projection_hashes'],'projection_records':data['projection_records'],
        'source_sha256_provenance_only':digest(raw),'shadow_sha256':digest(shadow_raw),
        'shadow_scope':'derived excluded-column rotation; target fixed; no extra patient or fit',
        'invariance_features_hashes_labels':True,'claim':plan['study']['scope']})
    try:
        result=evaluate(data,shadow,plan['study'],output,deadline)
        recheck(plan['inputs']);check_deadline(deadline)
        result['worker_elapsed_seconds']=time.monotonic()-PROCESS_STARTED
        supervisor.write_json(output/'worker-result.json',result)
    except BaseException as error:
        supervisor.write_json(output/'worker-result.json',{'status':'failed_or_incomplete',
            'error':{'type':type(error).__name__,'message':str(error)},
            'worker_elapsed_seconds':time.monotonic()-PROCESS_STARTED})
        raise


def launch(root,study_binding,release_binding):
    plan=preflight(root,study_binding,release_binding)
    output=Path(plan['directory']);output.mkdir(parents=True,exist_ok=False)
    supervisor.write_json(output/'baseline.json',plan)
    baseline={'path':str((output/'baseline.json').relative_to(root)),'sha256':supervisor.sha(output/'baseline.json')}
    command=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),
             '--worker-baseline',baseline['path'],'--worker-baseline-sha256',baseline['sha256']]
    environment=supervisor.private_environment({'caps':{'thread_environment':THREADS}})
    environment.update(PYTHONPATH=plan['dependency']['directory']+os.pathsep+str(ROOT/'src'),
                       PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    resources=plan['study']['resources']
    supervision=supervisor.supervise(command,output/'supervision',cwd=root,environment=environment,
        seconds=resources['worker_wall_seconds']+resources['termination_grace_seconds'],rss_bytes=resources['rss_limit_bytes'])
    result={'status':'failed_or_incomplete','baseline':baseline,'supervision':supervision,'worker':None,
            'claim':plan['study']['scope'],'no_retry':True}
    try:
        recheck(plan['inputs'])
        worker_path=output/'worker-result.json'
        if worker_path.is_file():
            result['worker']={'path':str(worker_path.relative_to(root)),'sha256':supervisor.sha(worker_path)}
            record=json.loads(worker_path.read_text())
            if (supervision['status']=='completed' and record['status']=='completed_observational_baseline'
                    and record['worker_elapsed_seconds']<resources['worker_wall_seconds']):
                result['status']='completed_observational_baseline'
    except BaseException as error:result['error']={'type':type(error).__name__,'message':str(error)}
    supervisor.write_json(output/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--manifest',default=DECLARATION_PATH);parser.add_argument('--manifest-sha256',default=DECLARATION_SHA)
    parser.add_argument('--release');parser.add_argument('--release-sha256');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker-baseline',help=argparse.SUPPRESS);parser.add_argument('--worker-baseline-sha256',help=argparse.SUPPRESS)
    args=parser.parse_args();root=args.root.resolve()
    if args.worker_baseline:
        if os.getpid()!=os.getpgrp():raise ValueError('Worker must lead its supervised process group')
        plan=bound(root,{'path':args.worker_baseline,'sha256':args.worker_baseline_sha256},json_value=True)
        current=preflight(root,plan['study_binding'],plan['release_binding'])
        if canonical(plan)!=canonical(current):raise ValueError('Worker plan changed')
        worker(root,plan);return
    study={'path':args.manifest,'sha256':args.manifest_sha256};release={'path':args.release,'sha256':args.release_sha256}
    if args.execute:
        if launch(root,study,release)['status']!='completed_observational_baseline':raise SystemExit(1)
    else:
        preflight(root,study,release);print('Metadata checked; no model has been fitted.')


if __name__=='__main__':main()
