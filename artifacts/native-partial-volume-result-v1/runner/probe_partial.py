"""Fixed18 partial-evacuation readout; wraps the unchanged prescribed union arm."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
UNION_PATH=ROOT/'build/native-cut-union-v1/probe_union.py'
UNION_SHA='0a7168a3c2f071536ec9dbc5b45515cb5d829cbc47c2c316d640d336e2ba1530'
EVALUATOR_PATH=ROOT/'build/native-partial-volume-v1/independent-review/plane_volume_evaluator.py'
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'


def source_module(name,path,expected):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Pinned adapter differs: '+str(path))
    spec=importlib.util.spec_from_loader(name,loader=None);module=importlib.util.module_from_spec(spec)
    module.__file__=str(path);sys.modules[name]=module
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def bound_json(binding):
    path=ROOT/binding['path'];raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=binding['sha256']:raise ValueError('Bound saved reference differs')
    return json.loads(raw)


def config():return json.loads((HERE/'config.json').read_bytes())


def declaration():
    current=config();spec=bound_json(current['retained_union_declaration'])
    spec.update(version=current['version'],caps=current['caps'])
    return spec


def rounding_bounds(fractions,tissue):
    """Public float32 mass enclosure; no analytic plane or subcell labels."""
    import numpy as np
    f=np.asarray(fractions)
    if f.dtype!=np.float32 or not np.isfinite(f).all() or np.any((f<0)|(f>1)) or f.shape!=tissue.shape:
        raise ValueError('Original public float32 nominal target field required')
    center=f.astype(np.float64)
    previous=np.nextafter(f,np.float32(-np.inf)).astype(np.float64)
    following=np.nextafter(f,np.float32(np.inf)).astype(np.float64)
    lower=np.maximum(0.,np.nextafter((previous+center)/2,-np.inf))
    upper=np.minimum(1.,np.nextafter((center+following)/2,np.inf))
    lower[~tissue]=0.;upper[~tissue]=0.
    return lower,upper


def grid_case(spacing,grid,spec):
    global LAST_CONFIG
    from resectionlab.native_cut_union import EvacuatedVolumeSpec
    source,base=UNION.GENERIC.grid_case(spacing,grid,spec)
    lo,hi=rounding_bounds(source.reference_target,base.tissue_mask)
    volume=EvacuatedVolumeSpec(lo,hi,config()['parent_mass_semantics'])
    LAST_CONFIG=replace(base,cut_union=UNION.cut_spec(spec),evacuated_volume=volume)
    return source,LAST_CONFIG


def source_receipt():
    index=json.loads((HERE/'source-index.json').read_bytes())
    files={}
    for name,expected in index['files'].items():
        actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError('Executing source/reference changed: '+name)
        files[name]=actual
    return {'files':files,'python':sys.version,'platform':UNION.BASE.platform.platform()}


def projection(row):
    """Old physical/action/reward semantics; v2 bound state identities differ."""
    native={k:v for k,v in row['native'].items() if k!='committed_history_hash'}
    actions=[]
    ignored={'source_state_hash','result_state_hash','native_seconds','partial_evacuation'}
    for action in row['actions']:actions.append({k:v for k,v in action.items() if k not in ignored})
    return {'native':native,'actions':actions}


def equivalent(left,right,tolerance=1e-10):
    if isinstance(left,(list,tuple)) and isinstance(right,(list,tuple)):
        return len(left)==len(right) and all(equivalent(a,b,tolerance) for a,b in zip(left,right))
    if isinstance(left,dict) and isinstance(right,dict):
        return set(left)==set(right) and all(equivalent(left[k],right[k],tolerance) for k in left)
    if type(left) in (int,float) and type(right) in (int,float):return abs(left-right)<=tolerance
    return type(left)==type(right) and left==right


PENDING={}
OUTPUT=None


def numerical_row(spacing,grid,spec):
    row,case,cfg,history=ORIGINAL_ROW(spacing,grid,spec)
    row['partial_evacuation']={'status':'pending_history_check' if history else 'empty_no_committed_history'}
    for action,record in zip((a for a in row['actions'] if a['status']=='accepted'),history):
        if 'evacuated_volume' not in record:raise ValueError('Explicit partial-volume macro absent')
        action['partial_evacuation']=deepcopy(record['evacuated_volume'])
    row['public_parent_mass_spec']=cfg.evacuated_volume.to_record()
    prior=UNION_BY_ID[row['row_id']]
    for key in ('spacing_mm','grid','shape','affine','source_hash','fixed_tool','physical_actions_sha256'):
        if row[key]!=prior[key]:raise ValueError('Fixed scene/action identity changed: '+key)
    row['retained_union_projection_sha256']=UNION.BASE.digest(projection(prior))
    row['current_legacy_projection_sha256']=UNION.BASE.digest(projection(row))
    row['legacy_projection_equal_within_tolerance']=equivalent(projection(prior),projection(row))
    if history:
        PENDING[(case.semantic_hash,UNION.BASE.digest(history))]=row
    else:
        row['partial_evacuation'].update(source_evacuated_mm3=0.,physical_tissue_evacuated_lower_mm3=0.,
            physical_target_evacuated_mm3=0.,evacuated_target_source_bounds_mm3=[0.,0.],
            whole_cell_physical_lower_mm3=0.,lower_bound_gain_mm3=0.,
            gap_before_mm3=row['static_not_executable']['contact_physical_upper_mm3'],
            gap_after_mm3=row['static_not_executable']['contact_physical_upper_mm3'],
            physical_target_inside_source_interval=True,within_analytic_target_cap=True)
    return row,case,cfg,history


def checked_history(checker,case,tools,history,**kwargs):
    from resectionlab.core import thaw_json
    key=(case.semantic_hash,UNION.BASE.digest(history));row=PENDING[key]
    cfg=LAST_CONFIG
    if cfg.source_hash!=case.semantic_hash or cfg.tissue_mask is not kwargs['tissue_mask']:
        raise ValueError('Checker config must be the just-reconstructed native case')
    v=cfg.evacuated_volume
    audit=checker(case,tools,history,**kwargs,distance_backend='batch',distance_batch_size=256,
        cut_union_spec=cfg.cut_union.to_record(),evacuated_volume_spec=v.to_record(),
        evacuated_target_fraction_lower=v.target_fraction_lower,evacuated_target_fraction_upper=v.target_fraction_upper)
    if not audit.feasible:return audit
    geometry=thaw_json(audit.certified_evacuated_geometry)
    removed=geometry['whole_removed_indices'];partial={tuple(c['index']):int(c['covered_bits_hex'],16) for c in geometry['partial_cells']}
    keys=sorted({tuple(c) for c in removed}|set(partial))
    bounds={cell:(float(v.target_fraction_lower[cell]),float(v.target_fraction_upper[cell])) for cell in keys}
    metrics=EVALUATOR.evaluate_certified_union(cfg.affine.tolist(),cfg.cut_union.max_depth,removed,partial,bounds,
        tissue_plane_z_mm=0.,target_plane_z_mm=declaration()['target_start_z_mm'],cancelled=kwargs.get('cancelled'))
    before=UNION_BY_ID[row['row_id']]['native']['removed_physical_mm3']
    upper=row['static_not_executable']['contact_physical_upper_mm3'];new=metrics['physical_tissue_evacuated_lower_mm3']
    target=metrics['physical_target_evacuated_mm3'];interval=metrics['cumulative_target_interval_source_mm3'];tol=config()['physical_volume_tolerance_mm3']
    after=history[-1]['evacuated_volume']['after']
    if (abs(metrics['source_evacuated_mm3']-after['evacuated_source_volume_mm3'])>tol
            or not equivalent(interval,after['evacuated_target_source_bounds_mm3'],tol)
            or abs(sum(h['evacuated_volume']['delta_evacuated_source_volume_mm3'] for h in history)-metrics['source_evacuated_mm3'])>tol):
        raise ValueError('Independent final union and executed cumulative/delta volumes differ')
    metrics.update(status='independently_checked_and_clipped',lower_bound_gain_mm3=new-before,
        union_whole_cell_lower_mm3=before,planned_contact_upper_mm3=upper,gap_before_mm3=upper-before,gap_after_mm3=upper-new,
        physical_target_inside_source_interval=interval[0]-tol<=target<=interval[1]+tol,
        within_analytic_target_cap=target<=row['exact_continuous_target_cap_mm3']+tol,
        target_shortfall_mm3=row['exact_continuous_target_cap_mm3']-target)
    # One final sparse union, never the bulk execution/proof history. Parent-mass
    # rows let the saved evaluator reconstruct allocation bounds independently.
    sparse={'schema':'checked_partial_evacuation_row_geometry_v1','row_id':row['row_id'],
        'source_hash':case.semantic_hash,'history_hash':row['native']['committed_history_hash'],
        'geometry':geometry,'parent_mass_bounds':[[list(cell),*bounds[cell]] for cell in keys]}
    raw=json.dumps(sparse,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
    if len(raw)>2*1024**2:raise ValueError('Single sparse-row output cap')
    destination=OUTPUT/'geometry'/f"{row['row_id']}.json"
    with destination.open('xb') as stream:stream.write(raw)
    metrics['certified_geometry']={'path':str(destination.relative_to(OUTPUT)),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    metrics['checker_geometry_digest']=geometry['geometry_digest']
    row['partial_evacuation']=metrics
    return audit


def criterion(prior_default,prior_union,report):
    rows={r['row_id']:r for r in report['rows']};default_complete={r['row_id'] for r in prior_default['rows'] if r['native']['complete_sequence']}
    union_complete={r['row_id'] for r in prior_union['rows'] if r['native']['complete_sequence']}
    completed={r['row_id'] for r in rows.values() if r['native']['complete_sequence'] and not r['coverage']['budget_refusals']}
    tol=config()['physical_volume_tolerance_mm3']
    present=(len(report['rows'])==18 and set(rows)=={r['row_id'] for r in prior_union['rows']}
             and all(len(r['actions'])==10 for r in rows.values()))
    audited=all(r['independent_audit'].get('status')=='completed' and r['independent_audit'].get('certificate',{}).get('feasible') is True
                for r in rows.values() if r['native']['accepted_actions'])
    physical=all(r['partial_evacuation'].get('status') in ('independently_checked_and_clipped','empty_no_committed_history')
        and r['partial_evacuation']['lower_bound_gain_mm3']>=-tol
        and r['partial_evacuation']['gap_after_mm3']>=-tol
        and r['partial_evacuation']['physical_target_inside_source_interval']
        and r['partial_evacuation'].get('parent_target_mass_enclosure_pass',True)
        and r['partial_evacuation']['within_analytic_target_cap'] for r in rows.values())
    fine=[r for r in rows.values() if r['spacing_mm']==.25]
    tightened=len(fine)==6 and all(r['partial_evacuation'].get('lower_bound_gain_mm3',0)>tol for r in fine)
    parity=all(r['legacy_projection_equal_within_tolerance'] for r in rows.values())
    gained=sorted(completed-default_complete);lost_default=sorted(default_complete-completed);lost_union=sorted(union_complete-completed)
    accepted=(report['status']=='completed' and report['source_unchanged'] and present and audited and physical and tightened
              and parity and len(gained)>=1 and not lost_default and not lost_union)
    return {'meets_predeclared_useful_criterion':accepted,'all_18_rows_present':present,'all_nonempty_audits_accepted':audited,
        'all_physical_bounds_respected':physical,'all_six_fine_row_bounds_strictly_tightened':tightened,
        'legacy_action_whole_cell_reward_projection_unchanged':parity,'additional_default_completions':gained,
        'lost_default_completions':lost_default,'lost_union_completions':lost_union,
        'primary_outcome':'conservative_physical_volume_bound_tightening','target_or_reward_improvement_is_not_actor_success':True}


def worker(output):
    global OUTPUT,UNION_BY_ID
    sys.path.insert(0,str(ROOT/'src'))
    from resectionlab import evaluation
    output=output.resolve();OUTPUT=output;(output/'geometry').mkdir()
    spec=declaration();prior=bound_json(config()['retained_union_report']);UNION_BY_ID={r['row_id']:r for r in prior['rows']}
    default=UNION.retained_baseline(spec);PENDING.clear();original=evaluation.independent_check_native_history
    evaluation.independent_check_native_history=lambda *a,**kw:checked_history(original,*a,**kw)
    UNION.RUN_COUNTS.update(native_strokes_started=0,native_strokes_returned=0,native_seconds=0.)
    UNION.COVERAGE_WORK.update({name:0 for name in UNION.COVERAGE_WORK})
    UNION.DEADLINE=time.perf_counter()+spec['caps']['cooperative_worker_seconds'];UNION.LIVE_OUTPUT=output
    try:code=UNION._original_worker(output)
    finally:evaluation.independent_check_native_history=original;UNION.DEADLINE=None;UNION.LIVE_OUTPUT=None
    report=json.loads((output/'report.json').read_bytes())
    report['planned_rows']=[r['row_id'] for r in prior['rows']];seen={r['row_id'] for r in report['rows']}
    report['missing_rows']=[r for r in report['planned_rows'] if r not in seen];report['planned_action_slots']=180
    report['native_call_costs_including_incomplete_row']=dict(UNION.RUN_COUNTS)
    report['returned_preview_coverage_work_including_incomplete_row']=dict(UNION.COVERAGE_WORK)
    report['retained_union_report']=config()['retained_union_report'];report['retained_default_report']=spec['retained_baseline']
    report['useful_criterion']=criterion(default,prior,report)
    report['cost_scope']='retained reference timing descriptive, not a paired speed experiment'
    UNION.BASE.write_json(output/'report.json',report)
    return code


def install(index):
    global UNION,ORIGINAL_ROW,EVALUATOR
    UNION=source_module('partial_fixed_union_adapter',UNION_PATH,UNION_SHA)
    EVALUATOR=source_module('partial_independent_plane_evaluator',EVALUATOR_PATH,index['files'][str(EVALUATOR_PATH.relative_to(ROOT))])
    ORIGINAL_ROW=UNION.numerical_row
    UNION.grid_case=grid_case
    UNION.BASE.__file__=str(Path(__file__).resolve())
    UNION.BASE.declaration=declaration;UNION.BASE.source_receipt=source_receipt
    UNION.BASE.grid_case=grid_case;UNION.BASE.numerical_row=numerical_row


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-index-sha256',required=True);a=p.parse_args()
    raw=(HERE/'source-index.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=a.source_index_sha256:raise ValueError('Exact source index')
    if not a.worker or 'RESECTIONLAB_PARENT_LEASE_FD' not in os.environ:raise ValueError('Owned worker only')
    import select,threading
    fd=int(os.environ['RESECTIONLAB_PARENT_LEASE_FD'])
    def lease():
        while True:
            if select.select([fd],[],[],.2)[0] and not os.read(fd,1):os._exit(125)
    threading.Thread(target=lease,daemon=True).start()
    install(json.loads(raw));source_receipt()
    def audit(event,args):
        if event in ('subprocess.Popen','os.fork','os.forkpty','os.posix_spawn','socket.connect','socket.getaddrinfo'):
            raise RuntimeError('Generated worker cannot spawn or network')
        if event=='open' and isinstance(args[0],(str,bytes)):
            path=Path(os.fsdecode(args[0])).resolve()
            if any(path.is_relative_to(ROOT/name) for name in ('data','sources')):raise RuntimeError('No acquired inputs')
    sys.addaudithook(audit)
    return worker(a.output)


if __name__=='__main__':raise SystemExit(main())
