"""Small pinned provenance and unchanged045 source allowlist; no payload loads."""
from pathlib import Path
from common import ROOT,IL,TRAIN,PUBLIC,PUBLIC_SHA,COHORT,COHORT_SHA,read_json,semantic,PAIR_SUBJECT

PRIOR_INDEX='build/train045-frontier-preparation-v1/recovery-compact/source-index.json'
PRIOR_INDEX_SHA='db913711734a2a0f820a8b433e144ef1a488ced3c935488cb6e22746d704c182'
PRIOR_REPORT='build/train045-frontier-preparation-v1/recovery-compact-01/report.json'
PRIOR_REPORT_SHA='3d9898b3222edca89ec681ed1a6cba7dd62c54d779638e98285341ef3258d35d'
PRIOR_PARENT='build/train045-frontier-preparation-v1/recovery-compact-01/launcher.json'
PRIOR_PARENT_SHA='946e2e08048798d8473b6113252a3ae676a279e1025c4c91557721d14130b145'
REPORT='build/continuation-endgame-pair-v1/recovery-01/report.json'
REPORT_SHA='e0ab3670571e64ddf5818c190a7d599f55f96dd5e59be24ca29b1038e7b29751'
PLANS='build/continuation-endgame-pair-v1/run-01/plans.json'
PLANS_SHA='6798288cbb05cce8692f61f64438b68fcad8aef15ebc5ec008b2f3e3a151521b'
TAIL_INDEX='build/continuation-tail-v1/runner/source-index.json'
TAIL_INDEX_SHA='b5f70a83311adb2915e81dac2747a0315e704e2141d214016dba9245c7fccb04'
TAIL_REPORT='build/continuation-tail-v1/run-01/report.json'
TAIL_REPORT_SHA='1bee2b1d4c40a55b666ce3269a39c4a5052e1911297b22c6c8702fb21000af66'
CHECKPOINTS={
 'IL64':{'path':IL+'/IL-final.psckpt','sha256':'c174b92b09ed8eccd034a7bc8bc7cb9a4d64e09aa40d6f7a3d596ae88ea57c5c',
         'bytes':137636,'parameters':'sha256:59506939917827f7555a61c930c33285dcb62feda52ec98576194a82e39587f3'},
 'TAIL':{'path':'build/continuation-tail-v1/run-01/TAIL-head.qckpt',
         'sha256':'d93ecc4e30bec0b11da9318f64327df0f35456094794d71fb561e8657030258e',
         'bytes':36038,'parameters':'sha256:171cae6bc2c23740f0e09cef094408491af937995b0af6d1d8416eedef3e42fb'}}
WORLD_KEYS=('subject','source_hash','decision_model_hash','public_observation_binding','observation_track','max_steps')


def metadata_pins():
    prior=read_json(ROOT/PRIOR_INDEX,PRIOR_INDEX_SHA);tail=read_json(ROOT/TAIL_INDEX,TAIL_INDEX_SHA)
    pins={PRIOR_INDEX:PRIOR_INDEX_SHA,PRIOR_REPORT:PRIOR_REPORT_SHA,PRIOR_PARENT:PRIOR_PARENT_SHA,
          REPORT:REPORT_SHA,PLANS:PLANS_SHA,TAIL_INDEX:TAIL_INDEX_SHA,TAIL_REPORT:TAIL_REPORT_SHA,
          PUBLIC:PUBLIC_SHA,COHORT:COHORT_SHA,IL+'/configuration.json':tail['input_metadata'][IL+'/configuration.json']}
    for subject in TRAIN:
        name=IL+'/'+subject+'-context.json';pins[name]=tail['input_metadata'][name]
    public=read_json(ROOT/PUBLIC,PUBLIC_SHA);arrays={}
    if [r['patient_id'] for r in public['cases']]!=list(TRAIN):raise ValueError('Unchanged four TRAIN manifests required')
    for row in public['cases']:
        path=Path(row['path']);pins[str(path.relative_to(ROOT))]=row['sha256'];manifest=read_json(path,row['sha256'])
        if manifest['patient_id']!=row['patient_id'] or manifest['private_evaluation_files_included'] is not False:
            raise ValueError('Original public-only metadata required')
        if row['patient_id']==PAIR_SUBJECT:
            arrays={str(Path(v['path']).relative_to(ROOT)):{k:v[k] for k in ('sha256','bytes','dtype')}
                    for v in manifest['input_files'].values()}
    if arrays!=prior['public_arrays'] or len(arrays)!=5:raise ValueError('Exact inherited045 array allowlist required')
    if tail['checkpoints']['IL64']!=CHECKPOINTS['IL64']:raise ValueError('Original frozen IL64 binding changed')
    failed=read_json(ROOT/FAILED_REPORT,FAILED_REPORT_SHA);parent=read_json(ROOT/FAILED_PARENT,FAILED_PARENT_SHA)
    require_failed_attempt(failed,parent)
    pins.update({FAILED_REPORT:FAILED_REPORT_SHA,FAILED_PARENT:FAILED_PARENT_SHA,FAILED_TRACE:FAILED_TRACE_SHA})
    return pins,arrays


def retained_prefix(report,plans):
    arm=report['branches']['RETAINED_TAIL'];plan=plans['RETAINED_TAIL']
    if (report['status']!='completed' or not report['source_unchanged'] or not report['baseline_reproduced']
        or not arm['complete'] or arm['audit']['accepted'] is not True or not plan['complete']
        or arm['actions']!=plan['actions'] or len(arm['actions'])!=13 or arm['actions'][-1]!='STOP'
        or arm['audit']['outcomes']['nonstop_actions']!=12 or arm['world']['subject']!=PAIR_SUBJECT
        or arm['world']['max_steps']!=24 or plan['plan_seal']!=semantic({k:v for k,v in plan.items() if k!='plan_seal'})
        or arm['plan_seal']!=plan['plan_seal']):
        raise ValueError('Exact completed original12 retained045 plan required')
    return arm,plan


def require_reference(report,parent):
    cleanup=parent['owned_lifecycle'];branches=report['branches']
    if (report['status']!='completed' or not report['source_unchanged'] or not report['baseline_reproduced']
        or report['diagnosis']['useful_gain'] is not True or parent['status']!='completed'
        or parent['report_sha256']!=PRIOR_REPORT_SHA or parent['source_index_sha256']!=PRIOR_INDEX_SHA
        or not cleanup['verified'] or not cleanup['child_reaped'] or cleanup['exit_code']!=0
        or cleanup['errors'] or cleanup['remaining_owned_pids'] or cleanup['remaining_state_unknown']
        or set(branches)!={'ORIGINAL_STOP','FOCUSED_PREPARATION'}
        or any(not a['complete'] or not a['audit']['accepted'] for a in branches.values())
        or branches['ORIGINAL_STOP']['audit']['outcomes']['positive_target_source_cells_removed']!=51
        or branches['FOCUSED_PREPARATION']['audit']['outcomes']['positive_target_source_cells_removed']!=80):
        raise ValueError('Exact accepted old capability reference required')
    return {'report_sha256':PRIOR_REPORT_SHA,'launcher_sha256':PRIOR_PARENT_SHA,
        'source_index_sha256':PRIOR_INDEX_SHA,'outcomes':{k:v['audit']['outcomes'] for k,v in branches.items()},
        'worker_seconds':report['worker_seconds'],'parent_seconds':parent['elapsed_seconds'],
        'native_previews':report['native_budget']['native_preview_entries'],
        'native_transition_counts':report['native_transition_counts'],
        'preserved_failed_attempt':report['preserved_failed_attempt'],
        'preserved_output_failed_attempt':report['preserved_output_failed_attempt'],
        'comparison_role':'retained capability reference only; not the matched learned baseline'}


def require_tail_metadata(report):
    fit=report['tail_fit'];head=report['head_checkpoints']['TAIL']
    if (report['status']!='completed' or not report['source_unchanged'] or not report['baseline_reproduced']
        or report['source_index_sha256']!=TAIL_INDEX_SHA or report['tail_calls_attempted']!=1
        or report['tail_calls_completed']!=1 or head['sha256']!=CHECKPOINTS['TAIL']['sha256']
        or head['bytes']!=CHECKPOINTS['TAIL']['bytes'] or head['file']!='TAIL-head.qckpt'
        or fit['base_parameter_hash']!=CHECKPOINTS['IL64']['parameters']
        or fit['fitted_head_hash']!=CHECKPOINTS['TAIL']['parameters']
        or fit['corpus_seal']!=report['corpus_seal'] or fit['manifest_hash']!=report['trace_manifest_hash']):
        raise ValueError('Exact completed frozen TAIL provenance required')
    worlds=report['corpus']['worlds']
    if set(worlds)!=set(TRAIN):raise ValueError('Original four training world bindings required')
    return fit,worlds

FAILED_REPORT='build/train045-focused-policy-v1/run-01/report.json'
FAILED_REPORT_SHA='c31b88e88c77b8f50d262193ce2424d4abf203eb23fbab14eed63de032229038'
FAILED_PARENT='build/train045-focused-policy-v1/run-01/launcher.json'
FAILED_PARENT_SHA='5dcad2ddc1f1ba893a1982d579598cf7dc18006cbb3703cd8d69e29f9bf10784'
FAILED_INDEX_SHA='46e709594e58de75233d1cb993efd4a3ec24f510a4c903938aac6069396dfb6a'
FAILED_TRACE='build/train045-focused-policy-v1/run-01/GREEDY-partial-selection.json.gz'
FAILED_TRACE_SHA='2ff156084aeae3d8cbe84f1307bb39fb8577f44bb19c7b2000220bcba7b61c50'


def require_failed_attempt(report,parent):
    cleanup=parent['owned_lifecycle'];binding=report['incomplete_selection']['GREEDY']
    if (report['status']!='failed_or_incomplete' or report['source_unchanged'] is not True
        or report['failure']!={'committed':False,'message':'Focused rollout time cap',
            'phase':'GREEDY_suffix_planning','type':'TimeoutError'}
        or report['plans_sealed_before_replay'] is not False or report['branches']
        or parent['status']!='incomplete_or_failed' or parent['report_sha256']!=FAILED_REPORT_SHA
        or parent['source_index_sha256']!=FAILED_INDEX_SHA or report['source_index_sha256']!=FAILED_INDEX_SHA
        or cleanup['verified'] is not True or not cleanup['child_reaped'] or cleanup['exit_code']!=1
        or cleanup['errors'] or cleanup['remaining_owned_pids'] or cleanup['remaining_state_unknown']
        or report['native_budget']['counting_reliable'] is not True
        or report['native_budget']['native_preview_entries']!=4733
        or report['native_transition_counts']!={'attempted':35,'completed':35,'raised':0}
        or report['total_frozen_policy_forwards']!=0 or report['checkpoint_loads']!=2
        or binding['path']!='GREEDY-partial-selection.json.gz' or binding['sha256']!=FAILED_TRACE_SHA
        or binding['bytes']!=1306073 or binding['decompressed_bytes']!=11053782
        or binding['decompressed_sha256']!='c4865e55bed6af29000632bcf1bfff86431b8805c6bbc12ec88179f46b1dfe52'):
        raise ValueError('Exact cleanly reaped timed-out first attempt required')
    return {'report_sha256':FAILED_REPORT_SHA,'launcher_sha256':FAILED_PARENT_SHA,
        'source_index_sha256':FAILED_INDEX_SHA,'status':'failed_or_incomplete','failure':report['failure'],
        'worker_seconds':report['worker_seconds'],'parent_seconds':parent['elapsed_seconds'],
        'native_previews':report['native_budget']['native_preview_entries'],
        'native_transition_counts':report['native_transition_counts'],
        'checkpoint_loads':report['checkpoint_loads'],'policy_forwards':report['total_frozen_policy_forwards'],
        'phase_costs':report['costs'],'native_budget':report['native_budget'],
        'retained_partial_trace':{**binding,'path':FAILED_TRACE},
        'role':'Failed attempt retained, not a complete arm or quality comparison; exact11-decision parity after both recovery plans seal.'}
