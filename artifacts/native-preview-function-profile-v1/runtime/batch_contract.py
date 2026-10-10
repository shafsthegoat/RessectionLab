"""Staged one-case profile; root must bind then-current sources before release."""
import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SUBJECTS=['ReMIND-045'];OUTPUT=HERE/'attempt-01'
CAPS={'worker_seconds':120,'parent_seconds':150,'sampled_rss_bytes':3*1024**3,
      'output_bytes':32*1024**2,'supervision_bytes':4*1024**2,'log_bytes':2*1024**2,
      'native_previews':6000,'threads':1}
FOLDER='build/remind-post-exposure-feasibility-v1/'
INPUTS={
 'original_release':{'path':FOLDER+'root-release.json','sha256':'8315099d95b16a9dacac54a65c4d84d6385e458534b6dab26b73a37555d4af1a'},
 'case':{'path':FOLDER+'attempt-01/ReMIND-045/construction-result.json','sha256':'0a3b97bc77ea10452b56aa45e0105b289030c74f3b4f7ce38c6e8242655b40ce'},
 'plan':{'path':FOLDER+'attempt-01/ReMIND-045/greedy-plan.json','sha256':'da7b49d6fde2c97c9f6911f42475afb7a4cc946e8c894e8267cc4a82679c58c1'},
 'metrics':{'path':FOLDER+'attempt-01/ReMIND-045/episode-metrics.json','sha256':'fa67b7ee6862222d480d60a570161e7990082e854e0327814191a629a48f6e21'},
 'audit':{'path':FOLDER+'attempt-01/ReMIND-045/independent-episode.json','sha256':'a18c03193e4968bf2f2caa44257117c51107091ee0bdfe18083b7f5edd1b0db9'}}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(ok,why):
    if not ok:raise ValueError(why)
def small(path,digest=None):
    path=ROOT/path
    need(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=4*1024**2,'bounded_metadata')
    if digest is not None:need(sha(path)==digest,'metadata_hash:'+str(path))
    return json.loads(path.read_bytes())
def release_template(head,source_index):
    original=small(INPUTS['original_release']['path'],INPUTS['original_release']['sha256'])
    return {'version':'native-function-profile-v1','execution_released':False,'expected_head':head,
        'subject':'ReMIND-045','caps':CAPS,'source_index':source_index,'inputs':INPUTS,
        'public_index':original['public_index'],'occupancy_condition':original['occupancy_condition']}
def guard(path,digest,*,execute):
    release=small(path,digest)
    need(set(release)=={'version','execution_released','expected_head','subject','caps','source_index','inputs','public_index','occupancy_condition'},'exact_release_fields')
    need(release['version']=='native-function-profile-v1' and release['caps']==CAPS
         and release['subject']=='ReMIND-045' and release['inputs']==INPUTS,'fixed_measurement')
    need(type(release['execution_released']) is bool,'typed_release')
    if execute:need(release['execution_released'],'root_release_required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    need(head==release['expected_head'],'then_current_HEAD_changed')
    index=small(release['source_index']['path'],release['source_index']['sha256'])
    need(index['head']==head,'source_index_HEAD')
    required={'src/resectionlab/native_resection.py','src/resectionlab/geometry.py',
              'src/resectionlab/native_spatial_evaluation.py','src/resectionlab/evaluation.py'}
    required.update(str((HERE/n).relative_to(ROOT)) for n in ('run_owned.py','initial_inventory_worker.py','batch_contract.py','profile_window.py'))
    need(required<=set(index['files']),'profile_and_current_evaluator_bound')
    for p,d in index['files'].items():need(sha(ROOT/p)==d,'source_or_input_changed:'+p)
    original=small(INPUTS['original_release']['path'],INPUTS['original_release']['sha256'])
    need(release['public_index']==original['public_index'] and release['occupancy_condition']==original['occupancy_condition'],'public_inputs_unchanged')
    return release,index,True
def complete_result(result,release,release_sha):
    need(result['status']=='complete_profiled_native_route' and result['release_sha256']==release_sha,'complete_profile')
    need(result['subject']=='ReMIND-045' and result['source_visits']==1 and result['greedy_calls']==1
         and result['committed_actions']==13 and result['exact_history_and_actions']
         and result['independent_accepted'],'same_accepted_route')
    need(all(result[k]==0 for k in ('model_calls','optimizer_calls','checkpoint_loads','blocked_external_calls')),'zero_model_or_external_work')
    need(result['native_previews']<=CAPS['native_previews'],'preview_cap')
    need(sha(OUTPUT/'native-functions.json')==result['profile_sha256'],'profile_binding')
    need(small(OUTPUT/'native-functions.json')['call_returned'],'profile_window_complete')
