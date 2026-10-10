"""Exact metadata/source preflight for one initial-inventory batch; no array reads."""
import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SUBJECTS=('ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045')
CONDITION='cerebrum_plus_supplied_tumor_with_preserved_partial_source_domain'
OUTPUT=HERE/'attempt-01'
CAPS={'worker_seconds':300,'parent_seconds':330,'sampled_rss_bytes':3*1024**3,
      'output_bytes':64*1024**2,'supervision_bytes':8*1024**2,'log_bytes':4*1024**2,
      'threads':1,'max_native_previews':512,'attempts':1,'automatic_retry':False}
ZERO_COUNTERS=('model_calls','optimizer_calls','checkpoint_loads','search_calls','transition_calls','blocked_file_or_external_calls')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(ok,reason):
    if not ok:raise ValueError(reason)
def read_bound(record):
    path=ROOT/record['path'];need(path.is_file() and not path.is_symlink() and path.stat().st_size<=2*1024**2,'bounded_regular_metadata')
    raw=path.read_bytes();need(hashlib.sha256(raw).hexdigest()==record['sha256'],'bound_metadata_changed:'+str(path));return json.loads(raw)
def guard(path,expected,*,execute=False):
    release=read_bound({'path':str(path),'sha256':expected});template=json.loads((HERE/'prepared-release.json').read_bytes())
    prepared=dict(release);prepared['execution_released']=False;prepared['expected_head']=None
    need(prepared==template and type(release['execution_released']) is bool,'only_release_boolean_and_head_may_change')
    need(release['caps']==CAPS and release['subjects']==list(SUBJECTS) and release['occupancy_condition']==CONDITION,'fixed_scope_caps')
    source=read_bound(release['source_index'])
    for name,digest in source['files'].items():need(sha(ROOT/name)==digest,'source_or_metadata_changed:'+name)
    canonical_ready=all(sha(ROOT/r['canonical_path'])==r['sha256'] for r in source['promotions'])
    index=read_bound(release['public_index']);need(tuple(r['patient_id'] for r in index['cases'])==SUBJECTS,'fixed_four_order')
    cohort=read_bound(release['cohort']);members={m['subject']:m for m in cohort['members']}
    for row in index['cases']:
        m=read_bound(row);p=row['patient_id']
        need(m['patient_id']==p and m['role']==row['role']==members[p]['role']=='TRAIN','frozen_TRAIN_roles')
        need(m['source_domain_condition']==CONDITION and m['training_admitted'] is False and m['private_evaluation_files_included'] is False,'public_search_only')
        need(set(m['input_files'])=={'image','supplied_support','supplied_whole_tumor','whole_tumor_domain','supplied_support_domain'},'exact_five_public_arrays')
    recipe=read_bound(release['historical_learning_release'])['learning_protocol']['cohort_execution']
    need(release['proposal_config']==recipe['proposal_config'] and release['max_steps']==recipe['max_steps']==24,'historical13axis120_horizon')
    need(release['retention_reference']==recipe['retention_mode'] and release['retention_applied'] is False,'no_search_retention_applied')
    if execute:
        need(release['execution_released'] is True and canonical_ready,'root_dispatch_and_canonical_promotion_required')
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();need(release['expected_head']==head,'frozen_HEAD_changed')
    return release,index,canonical_ready

def complete_result(result,release,expected):
    need(result.get('status') in {'initial_inventory_batch_complete','initial_inventory_batch_complete_with_case_failures'},'terminal_status')
    need(result.get('release_sha256')==expected and result.get('occupancy_condition')==CONDITION,'terminal_release_condition')
    need(tuple(r['patient_id'] for r in result['cases'])==SUBJECTS and result['all_four_cases_retained'] is True and result['failed_cases_replaced'] is False,'terminal_full_denominator')
    need(all(result.get(k)==0 for k in ZERO_COUNTERS) and result.get('source_arrays_written')==0 and result.get('training_admitted') is False,'terminal_zero_work_boundaries')
    need(all(r['status'] in {'constructed_initial_inventory_only','case_failed_retained'} for r in result['cases']),'terminal_all_dispositions')
    need(result['proposal_config']==release['proposal_config'] and result['max_steps']==24,'terminal_frozen_proposals')
    need(result['native_budget']['status']=='complete_history_awaiting_independent_audit','terminal_budget_completed')
    for row in result['cases']:
        if row['status']=='constructed_initial_inventory_only':
            need(row['role']=='TRAIN' and row['steps_taken']==0 and row['model_or_optimizer_calls']==0,'initial_only')
            need(row['accepted_count']<=row['emitted_count']<=120,'emission_cap')
            need(row['source_domain_extended'] is False and row['full_target_preserved'] is True,'source_truth_preserved')
    return True
