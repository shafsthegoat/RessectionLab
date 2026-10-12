"""One reverse512 checkpoint continuation; schedule clock retains original offset."""
from pathlib import Path
import sys,time,gc
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent
for p in (STAGE,STAGE/'training',STAGE/'model',STAGE/'campaign',STAGE/'curriculum',HERE):sys.path.insert(0,str(p))
import paths
import numpy as np
from curriculum_learning import ScheduledCurriculum,ExposureCurves,isolated_evaluation
from curriculum_contract import merge_experience
from continuation import load_verified
from gym_adapter import LocalGoalEnv
from policy import module_hash
from recovery_contract import recovery_start


class RecoveryCurriculum(ScheduledCurriculum):
    """New episode/RNG; old curriculum local256 corresponds to checkpoint512."""
    def reset(self,*,seed=None,options=None):
        if options:raise ValueError('Recovery reset schedule cannot be overridden')
        if self.episode is not None and not self.episode.done:
            raise ValueError('Only natural terminal reset may advance curriculum')
        if seed is not None:self.curriculum_rng=np.random.default_rng(seed)
        local=self.counts['decisions'];step=recovery_start(local);self._start=step
        obs,info=LocalGoalEnv.reset(self,seed=seed,options={'start_step':step})
        self.counts['resets']+=1;self.starts[str(step)]+=1
        self.reset_records.append({'recovery_decisions':local,'original_curriculum_decisions':256+local,
            'cumulative_decisions':512+local,'start_step':step})
        return obs,info
    def summary(self):
        return {**super().summary(),'original_curriculum_offset':256,
            'scope':'New512-decision recovery exposure only; prior512 and discarded failed work are separate'}


def run_recovery_arm(original,context,snapshots,*,method,seed,milestones,guard,progress,
        evaluate,save_checkpoint,episode_sink,curve_sink,resume):
    if method!='reverse' or seed!=0 or milestones!=[1024]:raise ValueError('Fixed reverse recovery only')
    prior_arm=resume['arm'];prior=prior_arm['checkpoints'][0]
    if prior['decisions']!=512 or prior['optimizer_completed']!=353:raise ValueError('Exact reverse512 required')
    env=RecoveryCurriculum(context,snapshots,method='reverse',seed=0,guard=guard,episode_sink=episode_sink)
    model=None;callback=None;failure=None;started=time.perf_counter()
    result={'arm':'reverse','seed':0,'ppo_config':prior_arm['ppo_config'],'milestones':[1024],
        'status':'running','checkpoints':[],'initialization':prior_arm['initialization'],
        'initial_evaluation':prior_arm['initial_evaluation'],'inherited512_endpoint':prior,
        'inherited_report':resume['report_binding'],'inherited_evaluations_reexecuted':False,
        'inherited_paths_relative_to':resume['prior_output_directory'],
        'additional_decisions_declared':512,'restored_checkpoint':resume['checkpoint'],
        'fresh_episode_and_RNG_at_resume':True,'RNG_continuity_claim':False,
        'original_curriculum_offset':256,'discarded_postcheckpoint_work_enters_training':False,
        'scope':'Restarted reverse trajectory, same objective/PPO; arithmetic-preserving input assembly'}
    try:
        guard();model,join=load_verified(resume['raw'],env,expected_sha256=resume['checkpoint']['sha256'],
            prior=prior,frozen=True,seed=0)
        result['optimizer_resume_join']=join;callback=ExposureCurves(guard,progress,env)
        model.learn(total_timesteps=512,reset_num_timesteps=False,callback=callback)
        if model.num_timesteps!=1024 or env.counts['decisions']!=512:raise RuntimeError('Recovery decision count differs')
        encoder=module_hash(model.policy.spatial.encoder)
        if encoder!=prior['encoder_hash']:raise RuntimeError('Frozen encoder changed')
        updates=model.policy.costs['optimizer_completed'];attempted=model.policy.costs['optimizer_attempted']
        if not 0<updates-353<=512 or attempted!=updates:raise RuntimeError('Actual Adam count differs')
        checkpoint=save_checkpoint(model,1024)
        evaluated,isolation=isolated_evaluation(model,env,evaluate,'1024')
        result['checkpoints'].append({'decisions':1024,'additional_decisions':512,'checkpoint':checkpoint,
            'evaluation':evaluated,'evaluation_isolation':isolation,'optimizer_attempted':attempted,
            'optimizer_completed':updates,'additional_optimizer_completed':updates-353,
            'sb3_epoch_updates':model._n_updates,'encoder_hash':encoder,'encoder_changed':False,
            'model_hash':module_hash(model.policy.spatial),'experience':merge_experience(prior['experience'],env.summary()),
            'segment_experience':env.summary(),'policy_costs':dict(model.policy.costs),
            'rollouts':[*prior['rollouts'],*callback.rows],
            'gradient_updates':[*prior['gradient_updates'],*model.policy.update_rows],
            'sb3_train_metrics':{k:float(v) if np.isfinite(v) else None for k,v in model.logger.name_to_value.items()
                if k.startswith('train/') and np.isscalar(v)},'segment_seconds':time.perf_counter()-started})
        result['status']='completed'
    except BaseException as error:
        failure=error;result['status']='failed_or_incomplete'
        result['failure']={'type':type(error).__name__,'message':str(error)}
        if model is not None:
            result['partial_policy_costs']=dict(model.policy.costs)
            result['partial_update_rows']=list(model.policy.update_rows);result['decisions']=model.num_timesteps
        if env.episode is not None:result['partial_episode']=env.episode_record(complete=env.episode.done)
        error.ppo_partial_result=result;raise
    finally:
        result['segment_experience']=env.summary();result['experience']=merge_experience(prior['experience'],env.summary())
        result['elapsed_seconds']=time.perf_counter()-started
        result['cumulative_lineage_arm_seconds']=(prior_arm['shared_prior_arm_seconds']+
            prior['segment_seconds']+result['elapsed_seconds'])
        result['failed_branch_tail_cost_excluded_from_lineage_but_charged_in_worker_report']=True
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
