"""Exactly192 additional PPO decisions per arm; old objective/model stays frozen."""
from pathlib import Path
import sys,io,hashlib,json,time,gc
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent
sys.path.insert(0,str(STAGE));sys.path.insert(0,str(STAGE/'training'))
sys.path.insert(0,str(STAGE/'model'));sys.path.insert(0,str(HERE))
import paths
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.save_util import load_from_zip_file
from learning import UniformCurriculum,Curves,PPO_CONFIG
from policy import LocalGoalPolicy,module_hash
from contract import EXPECTED_PPO,merge_experience


def exact_state(actual,expected,path='state'):
    if torch.is_tensor(expected):
        if (not torch.is_tensor(actual) or actual.shape!=expected.shape or actual.dtype!=expected.dtype
            or not torch.isfinite(actual).all() or not torch.isfinite(expected).all()
            or not torch.equal(actual.cpu(),expected.cpu())):raise ValueError('Checkpoint tensor mismatch: '+path)
    elif isinstance(expected,dict):
        if not isinstance(actual,dict) or actual.keys()!=expected.keys():raise ValueError('Checkpoint mapping mismatch: '+path)
        for key in expected:exact_state(actual[key],expected[key],path+'.'+str(key))
    elif isinstance(expected,(list,tuple)):
        if type(actual) is not type(expected) or len(actual)!=len(expected):raise ValueError('Checkpoint sequence mismatch: '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):exact_state(a,b,path+'.'+str(i))
    elif type(actual) is not type(expected) or actual!=expected:raise ValueError('Checkpoint scalar mismatch: '+path)


def state_hash(value):
    digest=hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous().numpy()
            digest.update(b'tensor'+str(a.shape).encode()+str(a.dtype).encode()+a.tobytes())
        elif isinstance(x,dict):
            digest.update(b'dict')
            for k in sorted(x,key=lambda k:(type(k).__name__,str(k))):add(k);add(x[k])
        elif isinstance(x,(list,tuple)):
            digest.update(type(x).__name__.encode())
            for v in x:add(v)
        else:digest.update((type(x).__name__+':'+json.dumps(x,sort_keys=True,allow_nan=False)).encode())
    add(value);return digest.hexdigest()


def load_verified(raw,env,*,expected_sha256,prior,frozen,seed):
    """Only externally pinned own SB3 checkpoint bytes may enter its pickle loader."""
    started=time.perf_counter()
    if len(raw)>2*1024**2 or hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('Exact bounded root-approved checkpoint bytes required')
    # Inspect the same saved tensors the mature loader restores; do not fit or
    # synthesize moments. Every optimizer scalar and tensor is compared exactly.
    _,saved,_=load_from_zip_file(io.BytesIO(raw),device='cpu',
                               custom_objects={'policy_class':LocalGoalPolicy})
    model=MaskablePPO.load(io.BytesIO(raw),env=env,device='cpu',force_reset=True,
                          custom_objects={'policy_class':LocalGoalPolicy})
    if (type(model.policy) is not LocalGoalPolicy or model.policy.freeze_encoder!=frozen
        or model.seed!=seed or model.num_timesteps!=prior['decisions']
        or model._n_updates!=prior['sb3_epoch_updates']):raise ValueError('Resumed mode/seed/update identity differs')
    for key,value in EXPECTED_PPO.items():
        if key=='device':actual=str(model.device)
        elif key=='clip_range':actual=model.clip_range(1.)
        else:actual=getattr(model,key)
        if actual!=value:raise ValueError('PPO configuration changed: '+key)
    if (model.clip_range_vf is not None or not model.normalize_advantage or model.use_sde
        or any(model.lr_schedule(x)!=.0003 or model.clip_range(x)!=.2 for x in (0.,.5,1.))):
        raise ValueError('Original PPO schedule/defaults changed')
    actual=model.get_parameters();exact_state(actual,saved)
    if module_hash(model.policy.spatial)!=prior['model_hash'] or module_hash(model.policy.spatial.encoder)!=prior['encoder_hash']:
        raise ValueError('Reported checkpoint parameter digest differs')
    for value in actual['policy.optimizer']['state'].values():
        if int(value['step'])!=prior['optimizer_completed']:raise ValueError('Adam moment step does not match actual completed updates')
    result={'policy_hash':state_hash(actual['policy']),'optimizer_hash':state_hash(actual['policy.optimizer']),
        'archive_policy_hash':state_hash(saved['policy']),'archive_optimizer_hash':state_hash(saved['policy.optimizer']),
        'all_tensors_and_optimizer_scalars_exact':True,'prior_decisions':model.num_timesteps,
        'prior_optimizer_completed':prior['optimizer_completed'],'prior_sb3_epochs':model._n_updates,
        'force_reset':True,'RNG_continuity_claim':False,
        'archive_decode_passes':2,'restore_and_compare_seconds':time.perf_counter()-started}
    model.policy.costs.update(prior['policy_costs'])
    return model,result


def run_resumed_arm(original,context,snapshots,*,frozen,seed,milestones,guard,progress,
        evaluate,save_checkpoint,episode_sink,curve_sink,resume):
    if milestones!=[256] or seed!=0 or resume['arm']['milestones']!=[64]:
        raise ValueError('Only declared64→256 seed0 continuation is released')
    prior_arm=resume['arm'];prior=prior_arm['checkpoints'][0]
    env=UniformCurriculum(context,snapshots,seed=seed,guard=guard,episode_sink=episode_sink)
    model=None;started=time.perf_counter();callback=None;failure=None
    result={'arm':prior_arm['arm'],'seed':seed,'ppo_config':PPO_CONFIG,'milestones':[256],
        'checkpoints':[],'status':'running','initialization':prior_arm['initialization'],
        'initial_evaluation':prior_arm['initial_evaluation'],
        'inherited64_endpoint':prior,'inherited_report':resume['report_binding'],
        'inherited_paths_relative_to':resume['prior_output_directory'],
        'inherited_evaluations_reexecuted':False,'additional_decisions_declared':192,
        'resume_reset':'New uniform12..16 episode; inherited unfinished episode retired without relabeling',
        'previous_arm_elapsed_seconds':prior_arm['elapsed_seconds']}
    try:
        guard();model,join=load_verified(resume['raw'],env,expected_sha256=resume['checkpoint']['sha256'],
                                        prior=prior,frozen=frozen,seed=seed)
        result['optimizer_resume_join']=join
        callback=Curves(guard,progress)
        model.learn(total_timesteps=192,reset_num_timesteps=False,callback=callback)
        if model.num_timesteps!=256 or env.counts['decisions']!=192:raise RuntimeError('Resume decision count differs')
        encoder=module_hash(model.policy.spatial.encoder)
        if frozen and encoder!=prior_arm['initialization']['encoder_hash']:raise RuntimeError('Frozen original encoder changed')
        updates=model.policy.costs['optimizer_completed'];attempted=model.policy.costs['optimizer_attempted']
        if not 0<updates-prior['optimizer_completed']<=192 or attempted!=updates:
            raise RuntimeError('Actual continuation optimizer count differs')
        checkpoint=save_checkpoint(model,256);endpoint=evaluate(model,'256')
        experience=merge_experience(prior_arm['experience'],env.summary())
        row={'decisions':256,'additional_decisions':192,'checkpoint':checkpoint,'evaluation':endpoint,
            'optimizer_attempted':attempted,'optimizer_completed':updates,
            'additional_optimizer_completed':updates-prior['optimizer_completed'],
            'sb3_epoch_updates':model._n_updates,'encoder_hash':encoder,
            'encoder_changed':encoder!=prior_arm['initialization']['encoder_hash'],
            'model_hash':module_hash(model.policy.spatial),'experience':experience,
            'segment_experience':env.summary(),'policy_costs':dict(model.policy.costs),
            'rollouts':[*prior['rollouts'],*callback.rows],
            'gradient_updates':[*prior['gradient_updates'],*model.policy.update_rows],
            'sb3_train_metrics':{k:float(v) if np.isfinite(v) else None for k,v in model.logger.name_to_value.items()
                if k.startswith('train/') and np.isscalar(v)},
            'segment_seconds':time.perf_counter()-started}
        result['checkpoints'].append(row);result['status']='completed'
    except BaseException as error:
        failure=error
        result['status']='failed_or_incomplete';result['failure']={'type':type(error).__name__,'message':str(error)}
        if model is not None:
            result['partial_policy_costs']=dict(model.policy.costs)
            result['partial_update_rows']=list(model.policy.update_rows);result['decisions']=model.num_timesteps
        if env.episode is not None:result['partial_episode']=env.episode_record(complete=env.episode.done)
        error.ppo_partial_result=result
        raise
    finally:
        result['segment_experience']=env.summary()
        result['experience']=merge_experience(prior_arm['experience'],env.summary())
        result['elapsed_seconds']=time.perf_counter()-started
        result['cumulative_arm_seconds']=prior_arm['elapsed_seconds']+result['elapsed_seconds']
        try:
            curve_sink(result)
        except BaseException as cleanup:
            if failure is None:raise
            result['final_curve_write_failure']={'type':type(cleanup).__name__,'message':str(cleanup)}
        finally:
            try:
                env.close()
            except BaseException as cleanup:
                if failure is None:raise
                result['environment_close_failure']={'type':type(cleanup).__name__,'message':str(cleanup)}
            finally:
                # Checkpoints already contain the optimizer. Detach only finished
                # worker objects so the next arm does not retain a64-volume buffer.
                if callback is not None:
                    callback.locals={};callback.globals={};callback.model=None
                if model is not None:
                    model.env=None;model.rollout_buffer=None;model._last_obs=None
                model=None;callback=None;gc.collect()
    return result
