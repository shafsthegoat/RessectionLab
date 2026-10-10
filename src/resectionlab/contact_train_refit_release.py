"""Separate fixed TRAIN-only refit publication; original IL/RL release unchanged.

Parent metadata admission creates no task or model. Only an owned worker may call
load_train_refit_for_owned_inference; it reuses the existing bounded decoder.
"""
from pathlib import Path
from .contact_family_desktop_release import ContactReleaseUnavailable, _file, _json_file

VARIANT = 'IL_TRAIN_REFIT'
RELEASE_VERSION = 'public-contact-train-refit-desktop-release-v1'
RELEASE_RELATIVE_PATH = 'artifacts/public-contact-train-refit-v1/desktop-release.json'
RELEASE_MANIFEST_SHA256 = "68091a27acf63715f23e5a649de1f06dee56b2391e6bb6919fc8bbb36a2d8887"
EXPERIMENT_HASH = 'sha256:fd211e00dbe6dd1c9ed50a58d6c3b9f9f8896a0acd3a7815b2c52e4a8dc1014f'
PARAMETER_HASH = 'sha256:0bdd6713358937ac2c665ff700210b24eb6a88433f6f410fc87bb62f23f7fe20'
CHECKPOINT_SHA256 = '5d7151397141c92ff82d8684814b9a0caed111f1809268bd448b8c1ea26d6bf7'
FIT = 'build/goal-conditioned-policy-v1/full-teacher-refit-run-v1/'
ROLLOUT = 'build/goal-conditioned-policy-v1/full-teacher-rollout-run-v1/'
SLOTS = {
    'experiment': (FIT+'experiment-freeze.json', 256*1024),
    'fitResult': (FIT+'result.json', 2*1024*1024),
    'fitSupervision': (FIT[:-1]+'.supervision/receipt.json', 32*1024),
    'rolloutResult': (ROLLOUT+'result.json', 2*1024*1024),
    'rolloutSupervision': (ROLLOUT[:-1]+'.supervision/receipt.json', 32*1024),
    'independentAudit': ('build/contact-full-teacher-rollout-result-v1/index.json', 64*1024),
    'checkpoint': (FIT+'refit-IL-final.gmckpt', 4*1024*1024),
}


def _check_row(record):
    if type(record) is not dict or set(record)!={'relativePath','sha256','bytes'}:
        raise ContactReleaseUnavailable('TRAIN refit file descriptor changed')


def read_published_train_refit(root, *, family_manifest):
    """Pin completed fit plus all24 native outcomes, without decoding a model."""
    if RELEASE_MANIFEST_SHA256 is None:
        raise ContactReleaseUnavailable('No reviewed TRAIN-only refit has been published')
    root=Path(root);path=root/RELEASE_RELATIVE_PATH
    # Fixed path, bounded file and expected SHA; reuse the existing publication reader.
    path=_file(root,RELEASE_RELATIVE_PATH,RELEASE_RELATIVE_PATH,32*1024,
               RELEASE_MANIFEST_SHA256,path.stat().st_size if path.exists() else 1)
    manifest=_json_file(path,32*1024,RELEASE_MANIFEST_SHA256)
    if (set(manifest)!={'version','variant','scope','experimentHash','familyHash','parameterHash','files','trainingBudget','knownTRAINOutcome'}
            or manifest['version']!=RELEASE_VERSION or manifest['variant']!=VARIANT
            or manifest['scope']!='TRAIN_only_generated_extra_compute_refit'
            or manifest['experimentHash']!=EXPERIMENT_HASH or manifest['parameterHash']!=PARAMETER_HASH
            or manifest['familyHash']!=family_manifest['family_hash']
            or type(manifest['files']) is not dict or set(manifest['files'])!=set(SLOTS)
            or manifest['trainingBudget']!={'updates':32,'statesPerUpdate':40,'lossForwards':1280,'fixedReadoutForwards':80}):
        raise ContactReleaseUnavailable('TRAIN refit publication identity/scope differs')
    paths={};records={}
    for key,(relative,limit) in SLOTS.items():
        record=manifest['files'][key];_check_row(record)
        paths[key]=_file(root,record['relativePath'],relative,limit,record['sha256'],record['bytes'])
        if key!='checkpoint':records[key]=_json_file(paths[key],limit,record['sha256'])
    from .core import semantic_digest
    experiment,fit,rollout=records['experiment'],records['fitResult'],records['rolloutResult']
    if (semantic_digest(experiment)!=EXPERIMENT_HASH
            or experiment['version']!='generated-public-contact-full-teacher-refit-v1'
            or semantic_digest(experiment['family_manifest'])!=semantic_digest(family_manifest)
            or fit.get('status')!='complete_fixed_full40_TRAIN_refit'
            or fit.get('experiment_hash')!=EXPERIMENT_HASH
            or tuple(fit.get(k) for k in ('optimizer_updates','loss_forward_calls','readout_forward_calls'))!=(32,1280,80)
            or fit.get('SELECT_or_MEASUREMENT_reads')!=0 or fit.get('patient_reads')!=0
            or fit.get('final_checkpoint',{}).get('sha256')!=CHECKPOINT_SHA256
            or fit['final_checkpoint'].get('parameter_hash')!=PARAMETER_HASH
            or manifest['files']['checkpoint']['sha256']!=CHECKPOINT_SHA256):
        raise ContactReleaseUnavailable('TRAIN refit fit/experiment/final endpoint differs')
    expected=[(row['layout_id'],goal) for row in family_manifest['source_bindings'] if row['role']=='TRAIN'
              for goal in ('surface','deep')]
    rows=rollout.get('rows')
    if (rollout.get('status')!='complete_fixed24_TRAIN_greedy_pass'
            or rollout.get('experiment_hash')!=EXPERIMENT_HASH
            or rollout.get('checkpoint',{}).get('sha256')!=CHECKPOINT_SHA256
            or rollout['checkpoint'].get('parameter_hash')!=PARAMETER_HASH
            or rollout.get('parameter_hash_after')!=PARAMETER_HASH
            or any(rollout.get(k)!=0 for k in ('optimizer_updates','new_search_calls','new_teacher_calls','patient_reads','SELECT_or_MEASUREMENT_reads'))
            or type(rows) is not list or len(rows)!=24
            or [(r.get('layout_id'),r.get('goal_id')) for r in rows]!=expected
            or any(r.get('status')!='complete' or r.get('role')!='TRAIN' for r in rows)):
        raise ContactReleaseUnavailable('TRAIN refit native rollout is incomplete or substituted')
    for label,target in [('fitSupervision','fitResult'),('rolloutSupervision','rolloutResult')]:
        receipt=records[label]
        if (receipt.get('status')!='complete' or receipt.get('exit_code')!=0 or receipt.get('cleanup_errors')!=[]
                or receipt.get('final_owned_pids')!=[] or receipt.get('worker_termination_confirmed') is not True
                or receipt.get('result_sha256')!=manifest['files'][target]['sha256']):
            raise ContactReleaseUnavailable('TRAIN refit owned completion differs')
    known={'tasks':24,'goalContacts':sum(bool(r['metrics']['goal_contacted_and_retained']) for r in rows),
        'savedSEARCHContacts':sum(bool(r['saved_search_metrics']['goal_contacted_and_retained']) for r in rows),
        'STOPOnly':sum(r['actions']==['STOP'] for r in rows),'meanReturn':sum(r['metrics']['total_reward'] for r in rows)/24,
        'scope':'generated_TRAIN_native_results_no_heldout_claim'}
    if manifest['knownTRAINOutcome']!=known or records['independentAudit'].get('status')!='PASS_SAVED_ONLY_ROLLOUT_PACKAGE':
        raise ContactReleaseUnavailable('TRAIN refit publication hid or changed its audited native outcome')
    return {'manifestSha256':RELEASE_MANIFEST_SHA256,'manifest':manifest,'paths':paths,
        'experimentRecord':experiment,'allowedTRAINKeys':tuple(expected),'knownTRAINOutcome':known}


def require_train_refit_request(publication, *, layout_id, goal_id):
    if type(layout_id) is not str or type(goal_id) is not str or (layout_id,goal_id) not in publication['allowedTRAINKeys']:
        raise ContactReleaseUnavailable('TRAIN_REFIT_ONLY: SELECT and MEASUREMENT layouts are unavailable')


def load_train_refit_for_owned_inference(root, *, layout_id, goal_id):
    """Owned-worker entry only. No user-supplied artifact path or fallback."""
    from .public_contact_family import family_manifest,layout_metadata
    if layout_metadata(layout_id)['role']!='TRAIN' or goal_id not in ('surface','deep'):
        raise ContactReleaseUnavailable('TRAIN_REFIT_ONLY: refuse role before artifact/model read')
    publication=read_published_train_refit(root,family_manifest=family_manifest())
    require_train_refit_request(publication,layout_id=layout_id,goal_id=goal_id)
    from .contact_learning_contract import freeze_full_teacher_refit
    from .contact_checkpoint import load_contact_checkpoint
    experiment=freeze_full_teacher_refit(publication['experimentRecord']['teacher_states'])
    if experiment.fingerprint!=EXPERIMENT_HASH:raise ContactReleaseUnavailable('Refit experiment changed')
    policy,metadata=load_contact_checkpoint(publication['paths']['checkpoint'],expected_sha256=CHECKPOINT_SHA256,
                                            experiment=experiment,kind='final')
    if metadata['parameter_hash']!=PARAMETER_HASH or metadata['lineage']['method']!='IL':
        raise ContactReleaseUnavailable('Verified refit checkpoint identity differs')
    return experiment,policy,metadata,publication
