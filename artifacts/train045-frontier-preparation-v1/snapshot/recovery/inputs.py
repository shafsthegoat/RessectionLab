"""Fixed completed045 metadata and existing public-source allowlist only."""
from pathlib import Path
from common import ROOT,IL,PUBLIC,PUBLIC_SHA,COHORT,COHORT_SHA,read_json,semantic,PAIR_SUBJECT
OLD_INDEX='build/continuation-endgame-pair-v1/runner/source-index.json'
OLD_INDEX_SHA='c2a0b2f70fbc8692db425c85132082b26216aa62f0470c641077bb94a5458a16'
REPORT='build/continuation-endgame-pair-v1/recovery-01/report.json'
REPORT_SHA='e0ab3670571e64ddf5818c190a7d599f55f96dd5e59be24ca29b1038e7b29751'
PLANS='build/continuation-endgame-pair-v1/run-01/plans.json'
PLANS_SHA='6798288cbb05cce8692f61f64438b68fcad8aef15ebc5ec008b2f3e3a151521b'
WORLD_KEYS=('subject','source_hash','decision_model_hash','public_observation_binding','observation_track','max_steps')


def metadata_pins():
    old=read_json(ROOT/OLD_INDEX,OLD_INDEX_SHA)
    pins={OLD_INDEX:OLD_INDEX_SHA,REPORT:REPORT_SHA,PLANS:PLANS_SHA,PUBLIC:PUBLIC_SHA,COHORT:COHORT_SHA,
          IL+'/configuration.json':old['input_metadata'][IL+'/configuration.json']}
    public=read_json(ROOT/PUBLIC,PUBLIC_SHA);arrays={}
    for row in public['cases']:
        path=Path(row['path']);pins[str(path.relative_to(ROOT))]=row['sha256'];record=read_json(path,row['sha256'])
        if record['patient_id']!=row['patient_id'] or record['private_evaluation_files_included'] is not False:
            raise ValueError('Original public-only source manifest required')
        if row['patient_id']==PAIR_SUBJECT:
            arrays={str(Path(v['path']).relative_to(ROOT)):{k:v[k] for k in ('sha256','bytes','dtype')}
                for v in record['input_files'].values()}
    if arrays!=old['public_arrays'] or len(arrays)!=5:raise ValueError('Exact original045 allowlist required')
    pins.update({FAILED_REPORT:FAILED_REPORT_SHA,FAILED_PARENT:FAILED_PARENT_SHA,FAILED_INDEX:FAILED_INDEX_SHA})
    return pins,arrays


def retained_prefix(report,plans):
    arm=report['branches']['RETAINED_TAIL'];plan=plans['RETAINED_TAIL']
    if (report['status']!='completed' or not report['source_unchanged'] or not report['baseline_reproduced']
        or not arm['complete'] or arm['audit']['accepted'] is not True or not plan['complete']
        or arm['actions']!=plan['actions'] or len(arm['actions'])!=13 or arm['actions'][-1]!='STOP'
        or arm['audit']['outcomes']['nonstop_actions']!=12 or arm['world']['subject']!=PAIR_SUBJECT
        or arm['world']['max_steps']!=24 or plan['plan_seal']!=semantic({k:v for k,v in plan.items() if k!='plan_seal'})
        or arm['plan_seal']!=plan['plan_seal']):
        raise ValueError('Exact completed fixed12 retained045 plan required')
    return arm,plan


FAILED_REPORT='build/train045-frontier-preparation-v1/run-01/report.json'
FAILED_REPORT_SHA='0a2027b9a7cf8ca48cd3dcd8692dfee84b883e4bbe66e2a4d0d557ec3e215870'
FAILED_PARENT='build/train045-frontier-preparation-v1/run-01/launcher.json'
FAILED_PARENT_SHA='938ba71ecc04d60bf426dd91f7daf31ac02442323691021adc68b32ec08a71c4'
FAILED_INDEX='build/train045-frontier-preparation-v1/source-index.json'
FAILED_INDEX_SHA='07e36b83b13c9d30567e32913b47c6767efa78b5556472c356b37b91f929001f'


def require_failed_attempt(report,parent):
    budget=report['native_budget'];cleanup=parent['owned_lifecycle']
    if (report['status']!='failed_or_incomplete' or not report['source_unchanged']
        or report['failure']['phase']!='saved_metadata_admission'
        or report['failure']['type']!='PlanningBudgetViolation'
        or report['failure']['message']!='preview_wrapper_replaced'
        or budget['native_preview_entries'] is not None or budget['counting_reliable'] is not False
        or budget['observed_admissions_before_accounting_failure']!=0
        or budget['native_preview_profile']['phases'] or report['native_transition_attempts']!=0
        or report['total_frozen_policy_forwards']!=0 or report['branches']
        or parent['status']!='incomplete_or_failed' or parent['worker_exit_code']!=1
        or not cleanup['verified'] or not cleanup['child_reaped'] or cleanup['exit_code']!=1
        or cleanup['errors'] or cleanup['remaining_owned_pids'] or cleanup['remaining_state_unknown']
        or parent['report_sha256']!=FAILED_REPORT_SHA
        or report['source_index_sha256']!=FAILED_INDEX_SHA or parent['source_index_sha256']!=FAILED_INDEX_SHA):
        raise ValueError('Exact preserved pre-source instrumentation failure required')
    return {'report_path':FAILED_REPORT,'report_sha256':FAILED_REPORT_SHA,'launcher_sha256':FAILED_PARENT_SHA,
        'source_index_sha256':FAILED_INDEX_SHA,'failure':report['failure'],
        'native_preview_entries':None,'counting_reliable':False,'observed_admissions_before_accounting_failure':0,
        'native_transition_attempts':report['native_transition_attempts'],
        'worker_seconds':report['worker_seconds'],'parent_seconds':parent['elapsed_seconds'],
        'sampled_peak_group_rss_bytes':parent['sampled_peak_group_rss_bytes'],
        'cost_scope':'Paid failed attempt preserved; unreliable preview counter is null, never a certified zero.'}
