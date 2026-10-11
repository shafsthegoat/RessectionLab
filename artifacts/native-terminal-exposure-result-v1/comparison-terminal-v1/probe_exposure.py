"""Fixed18 terminal-microstep exposure against the same saved partial-volume arm."""
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
    exposure=replace(UNION.cut_spec(spec),exposure_mode=config()['exposure_mode'],
                     closure_schedule=config()['closure_schedule'])
    LAST_CONFIG=replace(base,cut_union=exposure,evacuated_volume=volume)
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


def normalized_projection(row):
    # Match the finite JSON representation used in reports/digests. Refused
    # failure-tip tuples can contain numpy.float64 before serialization.
    return json.loads(json.dumps(projection(row),allow_nan=False))


def accepted_target_cap(row):
    # The first nine fixed preparation capsules do not reach z>=2.8 for either
    # retained tool. Only committed action10 contributes the fixed target cap;
    # a refused preview or unexecuted suffix contributes no removal.
    action=row['actions'][-1]
    return row['exact_continuous_target_cap_mm3'] if action['index']==10 and action['status']=='accepted' else 0.


def comparison(row):
    prior=BASELINE_BY_ID[row['row_id']]
    before=prior['partial_evacuation'];after=row['partial_evacuation']
    return {'baseline_complete':prior['native']['complete_sequence'],
        'current_complete':row['native']['complete_sequence'],
        'physical_target_difference_mm3':after['physical_target_evacuated_mm3']-before['physical_target_evacuated_mm3'],
        'physical_tissue_difference_mm3':after['physical_tissue_evacuated_lower_mm3']-before['physical_tissue_evacuated_lower_mm3'],
        'source_evacuated_difference_mm3':after['source_evacuated_mm3']-before['source_evacuated_mm3'],
        'whole_cell_physical_target_difference_mm3':row['native']['removed_physical_target_mm3']-prior['native']['removed_physical_target_mm3'],
        'whole_cell_reward_difference':row['native']['frozen_reward_total']-prior['native']['frozen_reward_total'],
        'baseline_physical_target_mm3':before['physical_target_evacuated_mm3'],
        'current_physical_target_mm3':after['physical_target_evacuated_mm3'],
        'action_result_projection_equal':row['action_result_projection_equal'],
        'scope':'Fixed inputs and unchanged reward rule; removals, rewards, history and completion may change with exposure model'}


def secondary_comparison(row):
    prior=SECONDARY_BY_ID[row['row_id']]
    for key in ('spacing_mm','grid','shape','affine','source_hash','fixed_tool','physical_actions_sha256'):
        if row[key]!=prior[key]:raise ValueError('Secondary reference changed fixed input: '+key)
    current=row['partial_evacuation'];old=prior['partial_evacuation']
    return {'reference':'comparison-run-02 all-capsule closure, descriptive only',
        'prior_complete':prior['native']['complete_sequence'],'current_complete':row['native']['complete_sequence'],
        'prior_preview_budget_refusals':prior['coverage']['budget_refusals'],
        'physical_target_difference_mm3':current['physical_target_evacuated_mm3']-old['physical_target_evacuated_mm3'],
        'physical_tissue_difference_mm3':current['physical_tissue_evacuated_lower_mm3']-old['physical_tissue_evacuated_lower_mm3'],
        'whole_cell_reward_difference':row['native']['frozen_reward_total']-prior['native']['frozen_reward_total'],
        'used_in_primary_criterion':False}


def prior_attempt_costs():
    rows=[]
    for binding in config()['prior_exposure_attempt_progress']:
        saved=bound_json(binding)
        rows.append({'binding':binding,'native_calls':saved['native_call_costs_including_incomplete_row'],
                     'coverage_work':saved['returned_preview_coverage_work_including_incomplete_row'],
                     'native_call_in_progress':saved['native_call_in_progress']})
    return {'scope':'Earlier exposure attempts only; retained partial-volume baseline cost is a separate reference',
        'attempts':rows,'native_strokes_started':sum(r['native_calls']['native_strokes_started'] for r in rows),
        'native_strokes_returned':sum(r['native_calls']['native_strokes_returned'] for r in rows),
        'native_seconds':sum(r['native_calls']['native_seconds'] for r in rows),
        'preview_budget_refusals':sum(r['coverage_work']['budget_refusals'] for r in rows)}


def equivalent(left,right,tolerance=1e-10):
    if isinstance(left,(list,tuple)) and isinstance(right,(list,tuple)):
        return len(left)==len(right) and all(equivalent(a,b,tolerance) for a,b in zip(left,right))
    if isinstance(left,dict) and isinstance(right,dict):
        return set(left)==set(right) and all(equivalent(left[k],right[k],tolerance) for k in left)
    if type(left) in (int,float) and type(right) in (int,float):return abs(left-right)<=tolerance
    return type(left)==type(right) and left==right


PENDING={}
OUTPUT=None
AUDIT_WORK={'history_audits_started':0,'history_audits_returned':0,'history_audit_seconds':0.,
            'plane_evaluations_started':0,'plane_evaluations_returned':0,'plane_evaluation_seconds':0.}


def numerical_row(spacing,grid,spec):
    row,case,cfg,history=ORIGINAL_ROW(spacing,grid,spec)
    row['partial_evacuation']={'status':'pending_history_check' if history else 'empty_no_committed_history'}
    for action,record in zip((a for a in row['actions'] if a['status']=='accepted'),history):
        if 'evacuated_volume' not in record:raise ValueError('Explicit partial-volume macro absent')
        action['partial_evacuation']=deepcopy(record['evacuated_volume'])
    row['public_parent_mass_spec']=cfg.evacuated_volume.to_record()
    prior=BASELINE_BY_ID[row['row_id']]
    for key in ('spacing_mm','grid','shape','affine','source_hash','fixed_tool','physical_actions_sha256'):
        if row[key]!=prior[key]:raise ValueError('Fixed scene/action identity changed: '+key)
    old_projection=normalized_projection(prior);new_projection=normalized_projection(row)
    row['retained_baseline_projection']=old_projection
    row['retained_baseline_projection_sha256']=UNION.BASE.digest(old_projection)
    row['current_action_result_projection_sha256']=UNION.BASE.digest(new_projection)
    row['action_result_projection_equal']=equivalent(old_projection,new_projection)
    row['exposure_model']=cfg.cut_union.to_record()
    if history:
        PENDING[(case.semantic_hash,UNION.BASE.digest(history))]=row
    else:
        row['partial_evacuation'].update(source_evacuated_mm3=0.,physical_tissue_evacuated_lower_mm3=0.,
            physical_target_evacuated_mm3=0.,evacuated_target_source_bounds_mm3=[0.,0.],
            whole_cell_physical_lower_mm3=0.,lower_bound_gain_mm3=0.,
            gap_before_mm3=row['static_not_executable']['contact_physical_upper_mm3'],
            gap_after_mm3=row['static_not_executable']['contact_physical_upper_mm3'],
            physical_target_inside_source_interval=True,physical_target_within_source_interval=True,
            parent_target_mass_enclosure_pass=True,within_analytic_target_cap=True,
            accepted_target_cap_mm3=0.,target_shortfall_mm3=0.)
        row['baseline_comparison']=comparison(row)
        row['secondary_all_capsule_comparison']=secondary_comparison(row)
    return row,case,cfg,history


def checked_history(checker,case,tools,history,**kwargs):
    from resectionlab.core import thaw_json
    key=(case.semantic_hash,UNION.BASE.digest(history));row=PENDING[key]
    cfg=LAST_CONFIG
    if cfg.source_hash!=case.semantic_hash or cfg.tissue_mask is not kwargs['tissue_mask']:
        raise ValueError('Checker config must be the just-reconstructed native case')
    v=cfg.evacuated_volume
    if AUDIT_WORK['history_audits_started']>=config()['caps']['max_independent_audits']:
        raise TimeoutError('Declared independent audit ceiling')
    AUDIT_WORK['history_audits_started']+=1;stamp=time.perf_counter()
    try:
        audit=checker(case,tools,history,**kwargs,distance_backend='batch',distance_batch_size=256,
            cut_union_spec=cfg.cut_union.to_record(),evacuated_volume_spec=v.to_record(),
            evacuated_target_fraction_lower=v.target_fraction_lower,evacuated_target_fraction_upper=v.target_fraction_upper)
        AUDIT_WORK['history_audits_returned']+=1
    finally:AUDIT_WORK['history_audit_seconds']+=time.perf_counter()-stamp
    if not audit.feasible:return audit
    geometry=thaw_json(audit.certified_evacuated_geometry)
    removed=geometry['whole_removed_indices'];partial={tuple(c['index']):int(c['covered_bits_hex'],16) for c in geometry['partial_cells']}
    keys=sorted({tuple(c) for c in removed}|set(partial))
    bounds={cell:(float(v.target_fraction_lower[cell]),float(v.target_fraction_upper[cell])) for cell in keys}
    AUDIT_WORK['plane_evaluations_started']+=1;stamp=time.perf_counter()
    try:
        metrics=EVALUATOR.evaluate_certified_union(cfg.affine.tolist(),cfg.cut_union.max_depth,removed,partial,bounds,
            tissue_plane_z_mm=0.,target_plane_z_mm=declaration()['target_start_z_mm'],cancelled=kwargs.get('cancelled'))
        AUDIT_WORK['plane_evaluations_returned']+=1
    finally:AUDIT_WORK['plane_evaluation_seconds']+=time.perf_counter()-stamp
    before=row['native']['removed_physical_mm3']
    upper=row['static_not_executable']['contact_physical_upper_mm3'];new=metrics['physical_tissue_evacuated_lower_mm3']
    target=metrics['physical_target_evacuated_mm3'];interval=metrics['cumulative_target_interval_source_mm3'];tol=config()['physical_volume_tolerance_mm3']
    after=history[-1]['evacuated_volume']['after']
    if (abs(metrics['source_evacuated_mm3']-after['evacuated_source_volume_mm3'])>tol
            or not equivalent(interval,after['evacuated_target_source_bounds_mm3'],tol)
            or abs(sum(h['evacuated_volume']['delta_evacuated_source_volume_mm3'] for h in history)-metrics['source_evacuated_mm3'])>tol):
        raise ValueError('Independent final union and executed cumulative/delta volumes differ')
    cap=accepted_target_cap(row)
    metrics.update(status='independently_checked_and_clipped',lower_bound_gain_mm3=new-before,
        current_whole_cell_lower_mm3=before,planned_contact_upper_mm3=upper,gap_before_mm3=upper-before,gap_after_mm3=upper-new,
        physical_target_inside_source_interval=interval[0]-tol<=target<=interval[1]+tol,
        within_analytic_target_cap=target<=cap+tol,accepted_target_cap_mm3=cap,
        target_shortfall_mm3=cap-target)
    # One final sparse union, never the bulk execution/proof history. Parent-mass
    # rows let the saved evaluator reconstruct allocation bounds independently.
    sparse={'schema':'checked_same_capsule_exposure_row_geometry_v1','row_id':row['row_id'],
        'source_hash':case.semantic_hash,'history_hash':row['native']['committed_history_hash'],
        'geometry':geometry,'parent_mass_bounds':[[list(cell),*bounds[cell]] for cell in keys]}
    raw=json.dumps(sparse,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
    if len(raw)>2*1024**2:raise ValueError('Single sparse-row output cap')
    destination=OUTPUT/'geometry'/f"{row['row_id']}.json"
    with destination.open('xb') as stream:stream.write(raw)
    metrics['certified_geometry']={'path':str(destination.relative_to(OUTPUT)),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    metrics['checker_geometry_digest']=geometry['geometry_digest']
    row['partial_evacuation']=metrics
    row['baseline_comparison']=comparison(row)
    row['secondary_all_capsule_comparison']=secondary_comparison(row)
    return audit


def criterion(prior,report):
    rows={r['row_id']:r for r in report['rows']}
    baseline_complete={r['row_id'] for r in prior['rows'] if r['native']['complete_sequence']}
    completed={r['row_id'] for r in rows.values() if r['native']['complete_sequence'] and not r['coverage']['budget_refusals']}
    tol=config()['physical_volume_tolerance_mm3']
    present=(len(report['rows'])==18 and set(rows)=={r['row_id'] for r in prior['rows']}
             and all(len(r['actions'])==10 for r in rows.values()))
    audited=all(r['independent_audit'].get('status')=='completed' and r['independent_audit'].get('certificate',{}).get('feasible') is True
                for r in rows.values() if r['native']['accepted_actions'])
    physical=all(r['partial_evacuation'].get('status') in ('independently_checked_and_clipped','empty_no_committed_history')
        and r['partial_evacuation']['lower_bound_gain_mm3']>=-tol
        and r['partial_evacuation']['gap_after_mm3']>=-tol
        and r['partial_evacuation']['physical_target_inside_source_interval']
        and r['partial_evacuation']['physical_target_within_source_interval']
        and r['partial_evacuation']['parent_target_mass_enclosure_pass']
        and r['partial_evacuation']['within_analytic_target_cap'] for r in rows.values())
    fine=[r for r in rows.values() if r['spacing_mm']==.25]
    no_target_regression=len(fine)==6 and all(r.get('baseline_comparison',{}).get('physical_target_difference_mm3',-float('inf'))>=-tol for r in fine)
    strict_gains=sorted(r['row_id'] for r in fine if r.get('baseline_comparison',{}).get('physical_target_difference_mm3',0)>tol)
    gained=sorted(completed-baseline_complete);lost=sorted(baseline_complete-completed)
    no_refusal=all(not r['coverage']['budget_refusals'] for r in rows.values())
    accepted=(report['status']=='completed' and report['source_unchanged'] and present and audited and physical
              and len(baseline_complete)==13 and not lost and no_target_regression and bool(strict_gains) and no_refusal)
    return {'meets_predeclared_useful_criterion':accepted,'all_18_rows_present':present,'all_nonempty_audits_accepted':audited,
        'all_physical_bounds_respected':physical,'baseline_completion_count':len(baseline_complete),
        'all_six_fine_target_volumes_nonregressing':no_target_regression,'strict_fine_target_gain_rows':strict_gains,
        'additional_baseline_completions':gained,'lost_baseline_completions':lost,'no_preview_budget_refusal':no_refusal,
        'action_result_equality_required':False,'primary_outcome':'certified_physical_target_gain_under_same_capsule_exposure_closure',
        'target_or_reward_improvement_is_not_actor_success':True}


def worker(output):
    global OUTPUT,BASELINE_BY_ID,SECONDARY_BY_ID
    sys.path.insert(0,str(ROOT/'src'))
    from resectionlab import evaluation
    output=output.resolve();OUTPUT=output;(output/'geometry').mkdir()
    spec=declaration();prior=bound_json(config()['retained_baseline_report']);BASELINE_BY_ID={r['row_id']:r for r in prior['rows']}
    secondary=bound_json(config()['secondary_all_capsule_report']);SECONDARY_BY_ID={r['row_id']:r for r in secondary['rows']}
    if len(secondary['rows'])!=18 or set(SECONDARY_BY_ID)!=set(BASELINE_BY_ID):raise ValueError('Exact secondary18 rows required')
    prior_costs=prior_attempt_costs()
    correction=bound_json(config()['retained_baseline_parity_correction'])
    if not correction['all_18_saved_projections_exactly_equal'] or not correction['corrected_criterion']['meets_predeclared_useful_criterion']:
        raise ValueError('Exact accepted saved baseline correction required')
    PENDING.clear();original=evaluation.independent_check_native_history
    evaluation.independent_check_native_history=lambda *a,**kw:checked_history(original,*a,**kw)
    UNION.RUN_COUNTS.update(native_strokes_started=0,native_strokes_returned=0,native_seconds=0.)
    UNION.COVERAGE_WORK.update({name:0 for name in UNION.COVERAGE_WORK})
    AUDIT_WORK.update({name:0. if name.endswith('_seconds') else 0 for name in AUDIT_WORK})
    UNION.DEADLINE=time.perf_counter()+spec['caps']['cooperative_worker_seconds'];UNION.LIVE_OUTPUT=output
    try:code=UNION._original_worker(output)
    finally:evaluation.independent_check_native_history=original;UNION.DEADLINE=None;UNION.LIVE_OUTPUT=None
    report=json.loads((output/'report.json').read_bytes())
    report['planned_rows']=[r['row_id'] for r in prior['rows']];seen={r['row_id'] for r in report['rows']}
    report['missing_rows']=[r for r in report['planned_rows'] if r not in seen];report['planned_action_slots']=180
    report['native_call_costs_including_incomplete_row']=dict(UNION.RUN_COUNTS)
    report['returned_preview_coverage_work_including_incomplete_row']=dict(UNION.COVERAGE_WORK)
    report['independent_work_including_incomplete_row']=dict(AUDIT_WORK)
    report['prior_exposure_attempt_costs']=prior_costs
    report['all_exposure_attempt_native_calls']={
        name:prior_costs[name]+UNION.RUN_COUNTS[name] for name in ('native_strokes_started','native_strokes_returned','native_seconds')}
    report['retained_baseline_report']=config()['retained_baseline_report']
    report['retained_baseline_parity_correction']=config()['retained_baseline_parity_correction']
    report['secondary_all_capsule_report']=config()['secondary_all_capsule_report']
    report['model']=config()['model'];report['useful_criterion']=criterion(prior,report)
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
