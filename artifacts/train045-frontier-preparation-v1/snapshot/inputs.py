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
