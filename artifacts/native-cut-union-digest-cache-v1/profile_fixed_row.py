"""UNRUN fixed generated-row attribution/parity worker, under root's owned parent.

Run each variant in a fresh process, one thread. No acquired inputs, models,
full matrix, or checkpoint access. The only implementation difference is the
preview ledger digest cache. Inclusive timers are explicitly nested; do not sum
cover/digest/JSON timers as independent costs. JSON digest and leaf distances are
disjoint function scopes. A timer-instrumented pair is not a production speedup.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
HERE=Path(__file__).resolve().parent
BASELINE=HERE.parent/'stage/resectionlab/native_cut_union.py'
CANDIDATE=HERE/'stage/resectionlab/native_cut_union.py'
ROW_ID='aligned-aspirator-0.25mm'
BASE_SHA='bd74877ae7448a7acc42d83f837e1933bb73564d45efc356929178f9dc5ba241'
CANDIDATE_SHA='10bc2eb2b4b7034f8d6c51791cd4c6bd6d2a5e235cf18a11f8a1c55b67e92443'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('baseline','candidate'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh output file required')
    prior_path=HERE.parent/'run-01/report.json'; prior=json.loads(prior_path.read_text())
    if prior['status']!='completed' or prior['expected_rows']!=18:raise ValueError('Complete retained report required')
    for rel in ('src/resectionlab/native_resection.py','src/resectionlab/evaluation.py','src/resectionlab/independent_cut_union.py','src/resectionlab/geometry.py','src/resectionlab/core.py','build/native-cut-union-v1/probe_union.py'):
        if sha(ROOT/rel)!=prior['source']['files'][rel]:raise ValueError('Fixed source changed: '+rel)
    if sha(BASELINE)!=BASE_SHA or sha(CANDIDATE)!=CANDIDATE_SHA:raise ValueError('Fixed producer source changed')
    sys.path.insert(0,str(ROOT/'src'))
    # Install exactly one class identity before native_resection imports it.
    import resectionlab
    selected=BASELINE if args.variant=='baseline' else CANDIDATE
    module=load('resectionlab.native_cut_union',selected)
    counters={}
    def timed(label,fn):
        def wrapped(*a,**kw):
            stamp=time.perf_counter()
            try:return fn(*a,**kw)
            finally:
                row=counters.setdefault(label,{'calls':0,'seconds_inclusive':0.})
                row['calls']+=1;row['seconds_inclusive']+=time.perf_counter()-stamp
        return wrapped
    cls=module.PreviewCutCoverage
    cls.digest=property(timed('preview_digest_property',cls.digest.fget))
    for name in ('cover','finish_step','freeze'):
        setattr(cls,name,timed('preview_'+name,getattr(cls,name)))
    module.semantic_digest=timed('ledger_canonical_json_freeze_thaw_hash',module.semantic_digest)
    module.point_segment_distances=timed('leaf_corner_distance',module.point_segment_distances)
    probe=load('_fixed_union_probe',HERE.parent/'probe_union.py')
    from resectionlab.native_resection import NativeResectionEngine
    NativeResectionEngine._result_digest=staticmethod(timed('result_history_json_digest',NativeResectionEngine._result_digest))
    from resectionlab.evaluation import independent_check_native_history
    spec=probe.declaration()
    grid=next(g for g in spec['grids'] if g['name']=='aligned-aspirator')
    started=time.perf_counter()
    row,case,config,history=probe.numerical_row(.25,grid,spec)
    numerical_seconds=time.perf_counter()-started
    if row['row_id']!=ROW_ID:raise AssertionError('Unexpected generated row')
    audit_start=time.perf_counter()
    certificate=independent_check_native_history(case,config.tools,history,tissue_mask=config.tissue_mask,
        access=config.access,cut_union_spec=config.cut_union.to_record(),distance_backend='batch',distance_batch_size=256)
    audit_seconds=time.perf_counter()-audit_start
    if not certificate.feasible:raise AssertionError(certificate.failures)
    # Timings differ by definition; every remaining row/certificate/history
    # field must match between arms (and scientific history matches saved v1).
    projection={k:v for k,v in row.items() if k not in ('timing_seconds','independent_audit')}
    projection['actions']=[{k:v for k,v in action.items() if k!='native_seconds'} for action in row['actions']]
    old=next(r for r in prior['rows'] if r['row_id']==ROW_ID)
    if row['native']!=old['native'] or row['coverage']!=old['coverage']:raise AssertionError('Retained native/certificate counters changed')
    record={'schema':'one_fixed_generated_union_digest_profile_v1','variant':args.variant,'row_id':ROW_ID,
        'source_sha256':sha(selected),'retained_report_sha256':sha(prior_path),'harness_sha256':sha(Path(__file__)),
        'native_seconds':sum(a.get('native_seconds',0.) for a in row['actions']),
        'numerical_row_seconds':numerical_seconds,'independent_audit_seconds':audit_seconds,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        'timers':counters,'timer_scope':'Inclusive/nested; no additive attribution. Instrumentation overhead is included.',
        'projection':projection,'projection_sha256':digest(projection),
        'exact_full_history_sha256':digest(history),'independent_certificate':certificate.to_dict(),
        'retained_native_and_work_exact':True,'acquired_inputs':False,'model_calls':0}
    with args.output.open('x') as out:json.dump(record,out,indent=2,sort_keys=True);out.write('\n')
    print(json.dumps({k:record[k] for k in ('variant','row_id','native_seconds','numerical_row_seconds','independent_audit_seconds','exact_full_history_sha256')}))

if __name__=='__main__':main()
