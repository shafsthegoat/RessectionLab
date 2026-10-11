"""Metadata-only, fixed TRAIN050 frozen inference contract. No payload imports."""
import hashlib,json,os,stat,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUTPUT=HERE/'attempt-01'
SUBJECTS=['ReMIND-050'];SUBJECT=SUBJECTS[0];ARMS=('balanced_ridge_switch_guard','exhaustive')
VERSION='frozen-ridge-TRAIN-development-transfer-v1'
EXPERIMENT_VERSION='TRAIN050-current-tool-comparison-v1'
POLICY_VERSION='nominal-top8-current-tool-comparison-v1'
BASELINE={'path':'build/hybrid-train050-transfer-v1/attempt-01/result.json',
          'sha256':'01ace7a4e5efc8b59fb1fe6d8e2996a595923c179e9d9c0af593210c2549b629'}
PREVIOUS_FALLBACK={'path':'build/hybrid-train050-fallback-v1/attempt-01/result.json','sha256':'dbabca44c7c31db7c1d610b6a69184e3b3826091ad61fa9f644cd3a2c8efe3c7'}
PREVIOUS_PARENT={'path':'build/hybrid-train050-fallback-v1/attempt-01.supervision/receipt.json','sha256':'de944be8efa438901d5f81603fa7a5271b0d2f1bb961600399ee8e3fc1aabb7b'}
PREVIOUS_RELEASE={'path':'build/hybrid-train050-fallback-v1/root-release.json','sha256':'6c22d2577e4175a530459bb0952a6e802938ae341324a88303d70339c9049280'}
CAPS={'worker_seconds':300,'parent_seconds':330,'sampled_rss_bytes':3*1024**3,
      'output_bytes':64*1024**2,'supervision_bytes':8*1024**2,'log_bytes':2*1024**2,
      'native_previews':40000,'native_transitions':96,'source_visits':4,'threads':1,
      'fits':0,'feature_count':681,'ridge_calls':48}
CACHE_LIMITS={'max_payload_bytes':32*1024**2,'max_entries':16384}
COHORT='manifests/experiments/remind-component-cohort-v1.json'
COHORT_SHA='326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05'
TRAIN_RESULT={'path':'artifacts/hybrid-proposal-balanced-v2/result.json','sha256':'360253e27ce3992facec4938e018bc02707248c1efad1b6cd318ed7ecd5ed4ca'}
TRAIN_PARENT={'path':'artifacts/hybrid-proposal-balanced-v2/parent-receipt.json','sha256':'4b275debe63c4dd75bcec7fb2485265a1b9637c808f4a1e741885bfe00e246bf'}
TRAIN_RELEASE={'path':'artifacts/hybrid-proposal-balanced-v2/root-release.json','sha256':'e81ab753baecc359fe6c067278308836d8cbd18804c8fbe2c6a955a6d2e0cb8c'}
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
    result=small(**{'path':TRAIN_RESULT['path'],'expected':TRAIN_RESULT['sha256']})
    parent=small(TRAIN_PARENT['path'],TRAIN_PARENT['sha256'])
    release=small(TRAIN_RELEASE['path'],TRAIN_RELEASE['sha256'])
    need(parent['status']=='complete' and parent['exit_code']==0 and not parent['cleanup_errors'] and not parent['remaining_owned_pids'],'clean_frozen_training')
    need(parent['result_sha256']==TRAIN_RESULT['sha256'] and parent['release_sha256']==TRAIN_RELEASE['sha256'],'training_receipt_joins')
    need(result['status']=='complete_fixed_TRAIN_balanced_ridge_contrast' and result['ridge_unchanged']
         and result['fits']==2 and result['teacher_decisions']==29 and result['source_visits']==28
         and result['ridge_file_sha256']['balanced_ridge_top8']==RIDGE['sha256'],'exact_balanced_endpoint')
    need(release['subjects']==['ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045'] and release['K']==8
         and release['horizon']==24 and release['proposal_config']==PROPOSALS
         and release['occupancy_condition']==OCCUPANCY and release['post_exposure_condition']==EXPOSURE,'unchanged_fit_cohort_world_and_K')
    need(len(result['arms'])==12 and all(r['complete'] and r['independent_accepted'] for r in result['arms']),'accepted_training_routes')
    return result

def previous_evidence():
    prior=small(PREVIOUS_FALLBACK['path'],PREVIOUS_FALLBACK['sha256'])
    parent=small(PREVIOUS_PARENT['path'],PREVIOUS_PARENT['sha256'])
    previous_release=small(PREVIOUS_RELEASE['path'],PREVIOUS_RELEASE['sha256'])
    need(parent['status']=='complete' and parent['exit_code']==0 and parent['worker_termination_confirmed']
         and not parent['remaining_owned_pids'] and not parent['cleanup_errors']
         and parent['result_sha256']==PREVIOUS_FALLBACK['sha256'] and parent['release_sha256']==PREVIOUS_RELEASE['sha256'], 'clean_prior_fallback')
    need(prior['status']=='complete_TRAIN050_fallback_comparison' and prior['comparison']['no_quality_loss'] is False
         and prior['original_negative_result']==BASELINE and prior['ridge_fingerprint']==RIDGE['fingerprint']
         and prior['fits']==prior['optimizer_calls']==prior['model_calls']==0
         and prior['all_plans_sealed_before_evaluation'] and all(row['complete'] and row['independent_accepted'] for row in prior['arms']), 'prior_fallback_quality_negative_retained')
    need(previous_release['original_negative_result']==BASELINE and previous_release['ridge']==RIDGE,'prior_frozen_model_lineage')
    return prior

def transfer_record(inputs):
    manifest,review=inputs['public_manifest'],inputs['intended_use_review']
    need(isinstance(manifest,dict) and isinstance(review,dict),'pending_TRAIN050_QC_and_manifest')
    digest(manifest['sha256']);digest(review['sha256'])
    return {'version':VERSION,'subject':SUBJECT,'role':'TRAIN','scope':'development_transfer_not_heldout_SELECT',
        'methods':['balanced_ridge_top8','exhaustive'],'K':8,'max_steps':24,'refits':0,'optimizer_updates':0,'neural_forwards':0,
        'training_subjects':['ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045'],
        'public_manifest_sha256':manifest['sha256'],'intended_use_review_sha256':review['sha256'],
        'frozen_ridge_sha256':RIDGE['sha256'],'frozen_ridge_fingerprint':RIDGE['fingerprint'],
        'training_result_sha256':TRAIN_RESULT['sha256']}

def template(head,index,inputs):
    return {'version':EXPERIMENT_VERSION,'policy_version':POLICY_VERSION,'original_negative_result':BASELINE,'previous_fallback_result':PREVIOUS_FALLBACK,'previous_fallback_parent':PREVIOUS_PARENT,'previous_fallback_release':PREVIOUS_RELEASE,'execution_released':False,'expected_head':head,'source_index':index,'inputs':inputs,
        'caps':CAPS,'subjects':SUBJECTS,'arms':list(ARMS),'public_index':inputs['public_manifest'],
        'cohort':{'path':COHORT,'sha256':COHORT_SHA},'occupancy_condition':OCCUPANCY,'post_exposure_condition':EXPOSURE,
        'proposal_config':PROPOSALS,'K':8,'horizon':24,'search_seconds':60,'cache_limits':CACHE_LIMITS,
        'ridge':RIDGE,'training_result':TRAIN_RESULT,'training_parent':TRAIN_PARENT,'training_release':TRAIN_RELEASE,
        'scope':'development transfer on one untrained TRAIN anatomy; no held-out SELECT claim',
        'zero_updates':True,'zero_refits':True,'SELECT_EVAL_closed':True}

def guard(path,expected,*,execute):
    release=small(path,expected);need(type(release.get('execution_released')) is bool,'typed_root_release')
    if execute:need(release['execution_released'],'root_dispatch_required')
    need({**release,'execution_released':False}==template(release['expected_head'],release['source_index'],release['inputs']),'exact_transfer_contract')
    record=transfer_record(release['inputs']) # Null dependencies fail before any payload.
    review=release['inputs']['intended_use_review'];read_bytes(review['path'],review['sha256'])
    m=release['inputs']['public_manifest'];manifest=small(m['path'],m['sha256'])
    need(manifest['patient_id']==SUBJECT and manifest['role']=='TRAIN' and manifest['public_only'] is True
         and manifest['training_admitted'] is False and manifest['private_evaluation_files_included'] is False
         and manifest['source_bindings']['intended_use_review_sha256']==record['intended_use_review_sha256']
         and set(manifest['input_files'])=={'image','supplied_support','supplied_whole_tumor','whole_tumor_domain','supplied_support_domain'},'qualified_public_050_manifest')
    cohort=small(COHORT,COHORT_SHA);member=[m for m in cohort['members'] if m['subject']==SUBJECT]
    need(len(member)==1 and member[0]['role']=='TRAIN' and member[0]['patient_group']==manifest['patient_group'],'original_050_TRAIN_role')
    metadata();previous_evidence()
    baseline=small(BASELINE['path'],BASELINE['sha256'])
    need(baseline['status']=='complete_TRAIN050_development_transfer' and baseline['comparison']['no_quality_loss'] is False
         and baseline['ridge_fingerprint']==RIDGE['fingerprint'] and baseline['all_plans_sealed_before_evaluation'], 'original_negative_preserved')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();need(head==release['expected_head'],'HEAD_changed')
    index=small(release['source_index']['path'],release['source_index']['sha256']);need(index['head']==head,'index_HEAD')
    required={str((HERE/n).relative_to(ROOT)) for n in ('batch_contract.py','hybrid_proposal.py','patient_transfer.py','fallback_policy.py','initial_inventory_worker.py','run_owned.py')}
    required|={str(p.relative_to(ROOT)) for p in (ROOT/'src/resectionlab').glob('*.py')}
    required|={'build/goal-conditioned-policy-v1/run_contact_owned.py','build/goal-conditioned-policy-v1/darwin_fast_sampler.py'}
    need(required<=set(index['files']),'declared_runtime_sources')
    for name,h in index['files'].items():need(sha(name)==h,'source_changed:'+name)
    need(sha(HERE/'hybrid_proposal.py')==HELPER_SHA,'unchanged_sparse_model_implementation')
    return release,index,True

def complete_result(result,release,release_sha):
    need(result['status']=='complete_TRAIN050_switch_guard_comparison' and result['release_sha256']==release_sha,'complete_transfer')
    need(result['policy_version']==POLICY_VERSION and result['original_negative_result']==BASELINE and result['previous_fallback_result']==PREVIOUS_FALLBACK,'explicit_policy_and_negative')
    need(result['source_visits']==4 and result['ridge_loads']==1 and result['fits']==result['optimizer_calls']==result['model_calls']==result['checkpoint_loads']==0,'no_training_or_neural_load')
    need(result['native_transitions']<=96 and result['native_previews']<=40000 and result['ridge_calls']<=48,'work_caps')
    need(result['blocked_external_calls']==0 and result['all_plans_sealed_before_evaluation'] and result['cache_restored'] and result['ridge_unchanged'],'information_boundary')
    need([(r['method'],r['subject']) for r in result['arms']]==[(m,SUBJECT) for m in ARMS]
         and all(r['complete'] and r['independent_accepted'] for r in result['arms']),'both_complete_geometry_results')
    for row in result['arms']:
        for path,h in row['files'].items():need(sha(OUTPUT/path)==h,'saved_output_changed:'+path)
    # Outcome quality is deliberately not an execution-completion criterion.
