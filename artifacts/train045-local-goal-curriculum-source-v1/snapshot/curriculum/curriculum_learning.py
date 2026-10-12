"""One reset-only comparison; exact existing policy, PPO and optimizer loader."""
from pathlib import Path
import sys,time,gc,hashlib,random
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent
for path in (STAGE,STAGE/'training',STAGE/'model',STAGE/'campaign',HERE):sys.path.insert(0,str(path))
import paths
import numpy as np
import torch
from learning import UniformCurriculum,Curves,PPO_CONFIG,STARTS
from gym_adapter import LocalGoalEnv
from continuation import load_verified,state_hash
from policy import module_hash
from curriculum_contract import METHODS,WINDOWS,reverse_start,merge_experience,semantic


class ScheduledCurriculum(UniformCurriculum):
    """Schedule evaluated only by reset; step never truncates or changes a live start."""
    def __init__(self,context,snapshots,*,method,seed,guard,episode_sink):
        if method not in METHODS:raise ValueError('Declared reset method required')
        super().__init__(context,snapshots,seed=seed,guard=guard,episode_sink=episode_sink)
        self.method=method;self.decisions_by_start={str(s):0 for s in STARTS}
        self.successes_by_start={str(s):0 for s in STARTS};self.bonus_events=0;self.reset_records=[]

    def reset(self,*,seed=None,options=None):
        if options:raise ValueError('Prospective schedule cannot accept caller-selected starts')
        if self.episode is not None and not self.episode.done:
            raise ValueError('Schedule changes only at a natural terminal reset')
        if seed is not None:self.curriculum_rng=np.random.default_rng(seed)
        local=self.counts['decisions']
        step=reverse_start(local) if self.method=='reverse' else int(self.curriculum_rng.choice(STARTS))
        self._start=step
        obs,info=LocalGoalEnv.reset(self,seed=seed,options={'start_step':step})
        self.counts['resets']+=1;self.starts[str(step)]+=1
        self.reset_records.append({'local_decisions':local,'cumulative_decisions':256+local,'start_step':step})
        return obs,info

    def step(self,action):
        start=str(self._start);result=super().step(action)
        self.decisions_by_start[start]+=1
        if result[2]:self.successes_by_start[start]+=int(result[4]['success'])
        self.bonus_events+=int(result[4]['terms']['goal_bonus']>0)
        return result

    def summary(self):
        return {**super().summary(),'reset_method':self.method,'decisions_by_start':dict(self.decisions_by_start),
            'successful_episodes_by_start':dict(self.successes_by_start),'goal_bonus_events':self.bonus_events,
            'reset_records':list(self.reset_records),'windows':list(WINDOWS) if self.method=='reverse' else None,
            'scope':'Exposure counts in this new768-decision segment; inherited summaries remain separate'}


class ExposureCurves(Curves):
    def __init__(self,guard,progress,env):
        super().__init__(guard,progress);self.exposure_env=env
    def _on_rollout_start(self):
        self.before_bonus=self.exposure_env.bonus_events
        self.before_decisions=dict(self.exposure_env.decisions_by_start)
    def _on_rollout_end(self):
        super()._on_rollout_end()
        bonus=self.exposure_env.bonus_events-self.before_bonus
        self.rows[-1].update(goal_bonus_events=bonus,bonus_bearing_rollout=bool(bonus),
            decisions_by_start={k:v-self.before_decisions[k] for k,v in self.exposure_env.decisions_by_start.items()})


def _observation_hash(obs):
    h=hashlib.sha256()
    if obs is None:return None
    for key in sorted(obs):
        a=np.asarray(obs[key]);h.update(key.encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()


def isolated_evaluation(model,env,evaluate,label):
    """Evaluation is paid but cannot advance/alter the live training trajectory or RNG."""
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    training=model.policy.training
    def stamp():
        episode=getattr(env,'episode',None)
        return {'policy':state_hash(model.policy.state_dict()),'optimizer':state_hash(model.policy.optimizer.state_dict()),
            'experience':semantic(env.summary()),'last_observation':_observation_hash(model._last_obs),
            'last_episode_starts':None if model._last_episode_starts is None else model._last_episode_starts.tolist(),
            'num_timesteps':model.num_timesteps,'epochs':model._n_updates,
            'curriculum_rng':getattr(getattr(env,'curriculum_rng',None),'bit_generator',None).state if hasattr(env,'curriculum_rng') else None,
            'native_state':None if episode is None else episode.work._engine.state_hash}
    before=stamp();started=time.perf_counter()
    try:result=evaluate(model,label)
    finally:
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
        model.policy.set_training_mode(training)
    after=stamp()
    if before!=after:raise ValueError('Evaluation changed live training state/optimizer/observation')
    return result,{'unchanged_training_state':True,'rng_restored':True,'policy_mode_restored':True,
        'state_seal':semantic(before),'paid_seconds':time.perf_counter()-started}


def run_curriculum_arm(original,context,snapshots,*,method,seed,milestones,guard,progress,
        evaluate,save_checkpoint,episode_sink,curve_sink,resume):
    if method not in METHODS or seed!=0 or milestones!=[512,1024] or resume['arm']['milestones']!=[256]:
        raise ValueError('Only fixed frozen256→512→1024 reset comparison is declared')
    prior_arm=resume['arm'];prior=prior_arm['checkpoints'][0]
    env=ScheduledCurriculum(context,snapshots,method=method,seed=seed,guard=guard,episode_sink=episode_sink)
    model=None;callback=None;failure=None;started=time.perf_counter()
    result={'arm':method,'seed':seed,'ppo_config':PPO_CONFIG,'milestones':milestones,'status':'running',
        'checkpoints':[],'initialization':prior_arm['initialization'],'initial_evaluation':prior_arm['initial_evaluation'],
        'inherited256_endpoint':prior,'inherited_report':resume['report_binding'],
        'inherited_paths_relative_to':resume['prior_output_directory'],'inherited_evaluations_reexecuted':False,
        'additional_decisions_declared':768,'shared_starting_checkpoint':resume['checkpoint'],
        'fresh_episode_and_RNG_at_resume':True,'RNG_continuity_claim':False,
        'windows':list(WINDOWS) if method=='reverse' else None,
        'shared_prior_arm_seconds':prior_arm['cumulative_arm_seconds']}
    try:
        guard();model,join=load_verified(resume['raw'],env,expected_sha256=resume['checkpoint']['sha256'],
            prior=prior,frozen=True,seed=seed)
        result['optimizer_resume_join']=join;callback=ExposureCurves(guard,progress,env)
        for endpoint in milestones:
            model.learn(total_timesteps=endpoint-model.num_timesteps,reset_num_timesteps=False,callback=callback)
            if model.num_timesteps!=endpoint or env.counts['decisions']!=endpoint-256:
                raise RuntimeError('Actual added decision count differs')
            encoder=module_hash(model.policy.spatial.encoder)
            if encoder!=prior['encoder_hash']:raise RuntimeError('Frozen shared encoder changed')
            updates=model.policy.costs['optimizer_completed'];attempted=model.policy.costs['optimizer_attempted']
            if not 0<updates-prior['optimizer_completed']<=endpoint-256 or attempted!=updates:
                raise RuntimeError('Actual optimizer counts differ')
            checkpoint=save_checkpoint(model,endpoint)
            evaluated,isolation=isolated_evaluation(model,env,evaluate,str(endpoint))
            row={'decisions':endpoint,'additional_decisions':endpoint-256,'checkpoint':checkpoint,'evaluation':evaluated,
                'evaluation_isolation':isolation,'optimizer_attempted':attempted,'optimizer_completed':updates,
                'additional_optimizer_completed':updates-prior['optimizer_completed'],'sb3_epoch_updates':model._n_updates,
                'encoder_hash':encoder,'encoder_changed':False,'model_hash':module_hash(model.policy.spatial),
                'experience':merge_experience(prior_arm['experience'],env.summary()),'segment_experience':env.summary(),
                'policy_costs':dict(model.policy.costs),'rollouts':[*prior['rollouts'],*callback.rows],
                'gradient_updates':[*prior['gradient_updates'],*model.policy.update_rows],
                'sb3_train_metrics':{k:float(v) if np.isfinite(v) else None for k,v in model.logger.name_to_value.items()
                    if k.startswith('train/') and np.isscalar(v)},'segment_seconds':time.perf_counter()-started}
            result['checkpoints'].append(row);curve_sink(result)
        result['status']='completed'
    except BaseException as error:
        failure=error;result['status']='failed_or_incomplete'
        result['failure']={'type':type(error).__name__,'message':str(error)}
        if model is not None:
            result['partial_policy_costs']=dict(model.policy.costs);result['partial_update_rows']=list(model.policy.update_rows)
            result['decisions']=model.num_timesteps
        if env.episode is not None:result['partial_episode']=env.episode_record(complete=env.episode.done)
        error.ppo_partial_result=result;raise
    finally:
        result['segment_experience']=env.summary();result['experience']=merge_experience(prior_arm['experience'],env.summary())
        result['elapsed_seconds']=time.perf_counter()-started
        result['cumulative_branch_seconds']=prior_arm['cumulative_arm_seconds']+result['elapsed_seconds']
        try:curve_sink(result)
        except BaseException as cleanup:
            if failure is None:raise
            result['final_curve_write_failure']={'type':type(cleanup).__name__,'message':str(cleanup)}
        finally:
            try:env.close()
            finally:
                if callback is not None:
                    callback.locals={};callback.globals={};callback.model=None;callback.exposure_env=None
                if model is not None:model.env=None;model.rollout_buffer=None;model._last_obs=None
                model=None;callback=None;gc.collect()
    return result
