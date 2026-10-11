"""Fixed fitted-cohort inference regression. Metadata only; no payload imports."""
import hashlib,json,os,stat,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUTPUT=HERE/'attempt-01'
FIT_SUBJECTS=['ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045']
SUBJECT='ReMIND-050';SUBJECTS=FIT_SUBJECTS+[SUBJECT]
ARMS=('switch_guard','public_bound')
VERSION='five-TRAIN-public-reward-window-comparison-v2'
TRANSFER_VERSION='frozen-ridge-TRAIN-development-transfer-v1'
POLICY_VERSION='nominal-top8-current-tool-public-window-v2'
CAPS={'worker_seconds':600,'parent_seconds':660,'sampled_rss_bytes':3*1024**3,
      'output_bytes':108*1024**2,'supervision_bytes':16*1024**2,'log_bytes':2*1024**2,
      'native_previews':40000,'native_transitions':480,'source_visits':20,'threads':1,
      'fits':0,'feature_count':681,'ridge_calls':500}
CACHE_LIMITS={'max_payload_bytes':32*1024**2,'max_entries':16384}
COHORT='manifests/experiments/remind-component-cohort-v1.json'
COHORT_SHA='326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05'
TRAIN_RESULT={'path':'artifacts/hybrid-proposal-balanced-v2/result.json','sha256':'360253e27ce3992facec4938e018bc02707248c1efad1b6cd318ed7ecd5ed4ca'}
TRAIN_PARENT={'path':'artifacts/hybrid-proposal-balanced-v2/parent-receipt.json','sha256':'4b275debe63c4dd75bcec7fb2485265a1b9637c808f4a1e741885bfe00e246bf'}
TRAIN_RELEASE={'path':'artifacts/hybrid-proposal-balanced-v2/root-release.json','sha256':'e81ab753baecc359fe6c067278308836d8cbd18804c8fbe2c6a955a6d2e0cb8c'}
TRAIN_OUTPUT=Path('build/hybrid-proposal-balanced-v2/attempt-01')
SOURCE_RELEASE={'path':'build/remind-post-exposure-feasibility-v1/root-release.json','sha256':'8315099d95b16a9dacac54a65c4d84d6385e458534b6dab26b73a37555d4af1a'}
RIDGE={'path':'build/hybrid-proposal-balanced-v2/attempt-01/balanced_ridge_top8.json',
       'sha256':'6a07215506ac68fa42afce40ad12a9efe9b042d83ee5d5bc1383b249a5ffe824',
       'fingerprint':'sha256:4578737059c005c411668ea70b148f1214b483f2d89c677605bbe5e3c67aa57a',
       'data_hash':'sha256:75adaa10137e33a2bd0ef193916f952b59f1d25cc6075cd16f58afbd5fc539a9'}
PROPOSALS={'intermediate_opening_mm':1.,'max_candidates':120,'nominal_min_membership':0.,
    'offsets_source_voxels':[[0,0],[-2,0],[2,0],[0,-2],[0,2],[-2,-2],[-2,2],[2,-2],[2,2],[-3,0],[3,0],[0,-3],[0,3]],
    'tool_footprint_opening':True}
OCCUPANCY='cerebrum_plus_supplied_tumor_with_preserved_partial_source_domain'
EXPOSURE='public-union-post-exposure-axis0-v1'
HELPER_SHA='457945c4c35bdf61ab8388e30ec4be81e5e1aba3003ff98e4573ecb995c021b8'
POLICY_SHA='01d9da472c6f53fe318f8dde9cbd6594aba2fbe7948b022dfde514a9788d427c'
FIFTY_RESULT={'path':'build/hybrid-train050-switch-guard-v1/attempt-01/result.json','sha256':'83011a701cc36da2295c074ea5c2daf506191d281e5f8cd09dad8a4a9ea3efed'}
FIFTY_PARENT={'path':'build/hybrid-train050-switch-guard-v1/attempt-01.supervision/receipt.json','sha256':'cc7a64c632c2027da861924ef6eb162e607ffe7714e8ae27f038486e8163f748'}
FIFTY_RELEASE={'path':'build/hybrid-train050-switch-guard-v1/root-release.json','sha256':'27f6a652c4f3f7fe660b5cf18b503457aaf66889116aa35a3a98b8f62420e52e'}
FIFTY_OUTPUT=Path('build/hybrid-train050-switch-guard-v1/attempt-01')

def need(ok,why):
    if not ok:raise ValueError(why)
def digest(value):
    need(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'explicit_sha256_required')
    return value
def read_bytes(path,expected=None,limit=8*1024**2):
    path=ROOT/Path(path);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(fd);need(stat.S_ISREG(info.st_mode) and 0<info.st_size<=limit,'bounded_regular_file')
        with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(limit+1)
        need(len(raw)==info.st_size and len(raw)<=limit,'file_size_changed')
    finally:os.close(fd)
    if expected is not None:need(hashlib.sha256(raw).hexdigest()==digest(expected),'exact_file_pin:'+str(path))
    return raw
def sha(path):return hashlib.sha256(read_bytes(path)).hexdigest()
def small(path,expected=None):
    return json.loads(read_bytes(path,expected),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite_json')))

def metadata():
    result=small(TRAIN_RESULT['path'],TRAIN_RESULT['sha256'])
    parent=small(TRAIN_PARENT['path'],TRAIN_PARENT['sha256']);release=small(TRAIN_RELEASE['path'],TRAIN_RELEASE['sha256'])
    source=small(SOURCE_RELEASE['path'],SOURCE_RELEASE['sha256'])
    need(parent['status']=='complete' and parent['exit_code']==0 and not parent['cleanup_errors'] and not parent['remaining_owned_pids'],'clean_frozen_training')
    need(parent['result_sha256']==TRAIN_RESULT['sha256'] and parent['release_sha256']==TRAIN_RELEASE['sha256'],'training_receipt_joins')
    need(result['status']=='complete_fixed_TRAIN_balanced_ridge_contrast' and result['ridge_unchanged']
         and result['fits']==2 and result['teacher_decisions']==29 and result['source_visits']==28
         and result['ridge_file_sha256']['balanced_ridge_top8']==RIDGE['sha256'],'exact_balanced_endpoint')
    need(release['subjects']==source['subjects']==FIT_SUBJECTS and release['K']==8 and release['horizon']==source['max_steps']==24
         and release['proposal_config']==source['proposal_config']==PROPOSALS
         and release['occupancy_condition']==source['occupancy_condition']==OCCUPANCY
         and release['post_exposure_condition']==source['post_exposure_condition']==EXPOSURE
         and release['public_index']==source['public_index'] and release['cohort']==source['cohort']=={'path':COHORT,'sha256':COHORT_SHA},'unchanged_fitted_cohort_world')
    need([(r['method'],r['subject']) for r in result['arms']]==[(m,s) for m in ('old_ridge_top8','balanced_ridge_top8','exhaustive') for s in FIT_SUBJECTS]
         and all(r['complete'] and r['independent_accepted'] for r in result['arms']),'accepted_training_denominator')
    return result,release,source

def transfer_metadata():
    result=small(FIFTY_RESULT['path'],FIFTY_RESULT['sha256'])
    receipt=small(FIFTY_PARENT['path'],FIFTY_PARENT['sha256'])
    release=small(FIFTY_RELEASE['path'],FIFTY_RELEASE['sha256'])
    need(receipt['status']=='complete' and receipt['exit_code']==0 and not receipt['cleanup_errors']
         and not receipt['remaining_owned_pids'] and receipt['result_sha256']==FIFTY_RESULT['sha256']
         and receipt['release_sha256']==FIFTY_RELEASE['sha256'],'clean_exposed_050_history')
    need(result['status']=='complete_TRAIN050_switch_guard_comparison' and result['ridge_unchanged']
         and result['fits']==result['optimizer_calls']==result['model_calls']==0
         and all(r['complete'] and r['independent_accepted'] for r in result['arms'])
         and release['subjects']==[SUBJECT] and release['ridge']==RIDGE and release['horizon']==24
         and release['proposal_config']==PROPOSALS and release['occupancy_condition']==OCCUPANCY
         and release['post_exposure_condition']==EXPOSURE,'same_exposed_050_world_and_frozen_model')
    return result,release

def transfer_record(inputs):
    return {'version':TRANSFER_VERSION,'subject':SUBJECT,'role':'TRAIN','scope':'development_transfer_not_heldout_SELECT',
        'methods':['balanced_ridge_top8','exhaustive'],'K':8,'max_steps':24,'refits':0,'optimizer_updates':0,'neural_forwards':0,
        'training_subjects':FIT_SUBJECTS,'public_manifest_sha256':digest(inputs['public_manifest']['sha256']),
        'intended_use_review_sha256':digest(inputs['intended_use_review']['sha256']),
        'frozen_ridge_sha256':RIDGE['sha256'],'frozen_ridge_fingerprint':RIDGE['fingerprint'],'training_result_sha256':TRAIN_RESULT['sha256']}

def historical_rows(result):
    return {(r['method'],r['subject']):r for r in result['arms'] if r['method'] in ('balanced_ridge_top8','exhaustive')}

def input_refs():
    result,release,source=metadata();refs=[TRAIN_RESULT,TRAIN_PARENT,TRAIN_RELEASE,SOURCE_RELEASE,release['public_index'],release['cohort']]
    index=small(**{'path':release['public_index']['path'],'expected':release['public_index']['sha256']})
    need([r['patient_id'] for r in index['cases']]==FIT_SUBJECTS,'exact_public_four_order')
    refs += [{'path':r['path'],'sha256':r['sha256']} for r in index['cases']]
    for row in historical_rows(result).values():
        for p,h in row['files'].items():refs.append({'path':str(TRAIN_OUTPUT/p),'sha256':h})
    fifty,old=transfer_metadata();transfer_record(old['inputs'])
    refs += [FIFTY_RESULT,FIFTY_PARENT,FIFTY_RELEASE,*old['inputs'].values()]
    for row in fifty['arms']:
        for p,h in row['files'].items():refs.append({'path':str(FIFTY_OUTPUT/p),'sha256':h})
    # The frozen parameter file is explicitly excluded from metadata preflight.
    return refs

def template(head,index):
    _,old,_=metadata();_,fifty=transfer_metadata()
    return {'version':VERSION,'policy_version':POLICY_VERSION,'execution_released':False,'expected_head':head,'source_index':index,
        'caps':CAPS,'subjects':SUBJECTS,'arms':list(ARMS),'public_index':old['public_index'],'cohort':old['cohort'],
        'occupancy_condition':OCCUPANCY,'post_exposure_condition':EXPOSURE,'proposal_config':PROPOSALS,'K':8,'horizon':24,
        'search_seconds':60,'cache_limits':CACHE_LIMITS,'ridge':RIDGE,'training_result':TRAIN_RESULT,'training_parent':TRAIN_PARENT,
        'training_release':TRAIN_RELEASE,'source_qualification_release':SOURCE_RELEASE,'source_world_hashes':old['source_world_hashes'],
        'transfer_inputs':fifty['inputs'],'prior_transfer_result':FIFTY_RESULT,
        'scope':'four fitted TRAIN plus already exposed TRAIN050 development comparison; no SELECT or new training','fits':0,'optimizer_updates':0,
        'neural_forwards':0,'SELECT_EVAL_closed':True,'all_plans_before_evaluation':True,
        'comparison':{'numeric_tolerance':1e-7,'fresh_reference':'switch_guard',
            'exact_actions_history_and_outcomes_required_for_quality':True,
            'historical_timing_is_not_a_matched_control':True,'favorable_outcome_required_for_completion':False}}

def guard(path,expected,*,execute):
    release=small(path,expected);need(type(release.get('execution_released')) is bool,'typed_root_release')
    if execute:need(release['execution_released'],'root_dispatch_required')
    need({**release,'execution_released':False}==template(release['expected_head'],release['source_index']),'exact_regression_contract')
    for ref in input_refs():read_bytes(ref['path'],ref['sha256'])
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();need(head==release['expected_head'],'HEAD_changed')
    index=small(release['source_index']['path'],release['source_index']['sha256']);need(index['head']==head,'index_HEAD')
    required={str((HERE/n).relative_to(ROOT)) for n in RUNTIME_FILES}
    required|={str(p.relative_to(ROOT)) for p in (ROOT/'src/resectionlab').glob('*.py')}
    required|={'build/goal-conditioned-policy-v1/run_contact_owned.py','build/goal-conditioned-policy-v1/darwin_fast_sampler.py'}
    required|={r['path'] for r in input_refs()}
    need(required<=set(index['files']),'declared_runtime_sources_and_metadata')
    for name,h in index['files'].items():need(sha(name)==h,'source_changed:'+name)
    need(sha(HERE/'hybrid_proposal.py')==HELPER_SHA and sha(HERE/'fallback_policy.py')==POLICY_SHA,'unchanged_model_and_switch_guard')
    return release,index,True

def complete_result(result,release,release_sha):
    need(result['status']=='complete_five_TRAIN_public_bound_comparison' and result['release_sha256']==release_sha,'complete_comparison')
    need(result['policy_version']==POLICY_VERSION and result['training_result']==TRAIN_RESULT,'frozen_endpoint_identity')
    need(result['source_visits']==20 and result['ridge_loads']==1 and result['fits']==result['optimizer_calls']==result['model_calls']==result['checkpoint_loads']==0,'no_training_or_neural_load')
    need(result['native_transitions']<=CAPS['native_transitions'] and result['native_previews']<=CAPS['native_previews'] and result['ridge_calls']<=CAPS['ridge_calls'],'work_caps')
    need(result['blocked_external_calls']==0 and result['all_plans_sealed_before_evaluation'] and result['cache_restored'] and result['ridge_unchanged'],'information_boundary')
    need([(r['method'],r['subject']) for r in result['arms']]==[(m,s) for s in SUBJECTS for m in ARMS]
         and all(r['complete'] and r['independent_accepted'] for r in result['arms']),'all_ten_complete_geometry_results')
    need(result['native_transitions']==2*sum(r['action_count'] for r in result['arms'])
         and result['native_previews']==sum(r['online_previews']+r['replay_previews'] for r in result['arms']),'exact_paid_counter_joins')
    for row in result['arms']:
        for path,h in row['files'].items():need(sha(OUTPUT/path)==h,'saved_output_changed:'+path)
    # A complete negative regression remains a complete measured result.

RUNTIME_FILES=('batch_contract.py','hybrid_proposal.py','patient_regression.py','patient_transfer.py',
    'policy_tasks.py','fallback_policy.py','bound_policy.py','stage/resectionlab/public_reward_window.py',
    'initial_inventory_worker.py','run_owned.py','freeze-runtime.py')
