"""One saved-plan replay diagnostic; negative geometry is a completed measurement."""
import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SUBJECTS=['ReMIND-045'];OUTPUT=HERE/'attempt-01'
CAPS={'worker_seconds':120,'parent_seconds':150,'sampled_rss_bytes':3*1024**3,
      'output_bytes':32*1024**2,'supervision_bytes':4*1024**2,'log_bytes':2*1024**2,
      'native_previews':6000,'threads':1}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(condition,reason):
    if not condition:raise ValueError(reason)
def small(path,digest=None):
    p=ROOT/path
    need(not p.is_symlink() and p.is_file() and 0<p.stat().st_size<=4*1024**2,'bounded_metadata')
    if digest is not None:need(sha(p)==digest,'metadata_hash:'+str(path))
    return json.loads(p.read_bytes())
def guard(path,digest,*,execute):
    release=small(path,digest)
    need(set(release)=={'version','execution_released','expected_head','subject','caps','source_index',
        'inputs','public_index','occupancy_condition','parameter_hash','permitted_source_delta'},'exact_release_fields')
    need(release['caps']==CAPS and release['subject']=='ReMIND-045','fixed_scope')
    need(type(release['execution_released']) is bool,'typed_release')
    if execute:need(release['execution_released'],'root_release_required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    need(head==release['expected_head'],'source_HEAD_changed')
    index=small(release['source_index']['path'],release['source_index']['sha256'])
    for p,d in index['files'].items():need(sha(ROOT/p)==d,'source_or_input_changed:'+p)
    return release,index,True
def complete_result(result,release,release_sha):
    need(result['status']=='complete_saved_plan_diagnostic' and result['subject']=='ReMIND-045','complete_measurement')
    need(result['release_sha256']==release_sha and result['saved_plan_sha256']==release['inputs']['plan']['sha256'],'receipt_binding')
    need(result['source_visits']==1 and result['committed_actions']==24 and result['full_history_equal'] is True,'exact_replay')
    need(all(result[k]==0 for k in ('policy_forwards','optimizer_updates','checkpoint_loads','teacher_search_calls')),'evaluation_only')
    need(type(result['independent_accepted']) is bool,'saved_geometry_disposition')
    need(result['native_previews']<=CAPS['native_previews'],'preview_cap')
    need(sha(OUTPUT/'native-replay.json')==result['native_replay_sha256'],'certificate_binding')
    certificate=small(OUTPUT/'native-replay.json',result['native_replay_sha256'])
    need(certificate['independent_geometry']['accepted']==result['independent_accepted']
        and certificate['independent_geometry']['complete_episode'] is True
        and certificate['independent_geometry']['geometry']==result['independent_geometry'],'exact_audit_disposition')
