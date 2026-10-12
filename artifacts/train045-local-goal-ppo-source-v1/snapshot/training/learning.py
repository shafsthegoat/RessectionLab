"""One fixed MaskablePPO configuration; raw observations, genuine resets, paid work."""
from pathlib import Path
import sys
STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE)); sys.path.insert(0, str(STAGE/'model'))
import paths
import time
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from gym_adapter import LocalGoalEnv
from policy import LocalGoalPolicy, module_hash
from resectionlab.core import semantic_digest, freeze_json, thaw_json

PPO_CONFIG = dict(n_steps=64, batch_size=4, n_epochs=4, learning_rate=3e-4,
    gamma=1., gae_lambda=1., clip_range=.2, ent_coef=.01, vf_coef=.5,
    max_grad_norm=.5, target_kl=.03, device='cpu', verbose=0)
STARTS = (12,13,14,15,16)


class UniformCurriculum(LocalGoalEnv):
    """Only reset choice changes: uniform over independently qualified real states."""
    def __init__(self, context, snapshots, *, seed, guard, episode_sink):
        if set(snapshots) != set(STARTS): raise ValueError('All five genuine starts required')
        super().__init__(context, snapshots, guard=guard)
        self.curriculum_rng = np.random.default_rng(seed)
        self.episode_sink = episode_sink
        self.counts = {'patients':1, 'resets':0, 'decisions':0, 'completed_episodes':0,
            'successful_episodes':0, 'automatic_STOP_transitions':0}
        self.starts = {str(s):0 for s in STARTS}; self.states = set(); self.state_actions = set()
        self.episode_returns = []; self.catalog_costs = {}; self._start = None

    def reset(self, *, seed=None, options=None):
        if options: raise ValueError('Training curriculum is prospectively uniform, not caller chosen')
        if seed is not None: self.curriculum_rng = np.random.default_rng(seed)
        step = int(self.curriculum_rng.choice(STARTS)); self._start = step
        obs, info = super().reset(seed=seed, options={'start_step':step})
        self.counts['resets'] += 1; self.starts[str(step)] += 1
        return obs, info

    def step(self, action):
        cat = self.episode.observe(); state = cat.work._engine.state_hash
        self.states.add(state)
        if 0 <= int(action) < len(cat.packets):
            self.state_actions.add(semantic_digest({'state':state,'action':cat.packets[int(action)]['physical_action']}))
        obs, reward, done, truncated, info = super().step(action)
        self.counts['decisions'] += 1
        if done:
            self.counts['completed_episodes'] += 1
            self.counts['successful_episodes'] += int(self.episode.success)
            self.counts['automatic_STOP_transitions'] += self.episode.costs.get('automatic_STOP_transitions',0)
            self.episode_returns.append(float(self.episode.total_reward))
            for key, value in self.episode.costs.items(): self.catalog_costs[key] = self.catalog_costs.get(key,0) + value
            self.episode_sink(self.episode_record(complete=True))
        return obs, reward, done, truncated, info

    def episode_record(self, *, complete):
        e = self.episode
        return {'episode':self.counts['completed_episodes'], 'complete':complete,
            'start_absolute_step':self._start, 'end_absolute_step':e.work._steps,
            'return':e.total_reward, 'success':e.success, 'action_records':e.packets,
            'local_rows':e.rows, 'native_history_hash':semantic_digest(e.work._history),
            'final_native_state_hash':e.work._engine.state_hash, 'costs':e.costs,
            'scope':'Executed simulated training trajectory; no independent replay claim'}

    def summary(self):
        return {**self.counts, 'start_counts':dict(self.starts), 'distinct_source_states':len(self.states),
            'distinct_state_physical_actions':len(self.state_actions),
            'source_state_hashes':sorted(self.states), 'state_physical_action_hashes':sorted(self.state_actions),
            'unfinished_episode_decisions':0 if self.episode is None or self.episode.done else len(self.episode.rows),
            'unfinished_executed_episodes':int(self.episode is not None and not self.episode.done and bool(self.episode.rows)),
            'episode_returns':list(self.episode_returns), 'completed_catalog_costs':dict(self.catalog_costs),
            'live_episode_costs':{} if self.episode is None or self.episode.done else dict(self.episode.costs)}


def ranges(values):
    a = np.asarray(values, np.float64)
    if not a.size or not np.isfinite(a).all(): raise ValueError('Finite nonempty diagnostic array required')
    return {'min':float(a.min()), 'max':float(a.max()), 'mean':float(a.mean()), 'std':float(a.std())}


class Curves(BaseCallback):
    def __init__(self, guard, progress):
        super().__init__(); self.guard=guard; self.progress=progress; self.rows=[]
    def _on_step(self):
        self.guard()
        self.progress('training_decision', decisions=self.num_timesteps,
            optimizer_attempted=self.model.policy.costs['optimizer_attempted'],
            optimizer_completed=self.model.policy.costs['optimizer_completed'])
        return True
    def _on_rollout_end(self):
        b=self.model.rollout_buffer
        self.rows.append({'decisions':self.num_timesteps, 'rollout_values':ranges(b.values),
            'advantages':ranges(b.advantages), 'returns':ranges(b.returns), 'rewards':ranges(b.rewards),
            'optimizer_completed_before_update':self.model.policy.costs['optimizer_completed']})


def make_model(original, env, *, seed, frozen):
    model=MaskablePPO(LocalGoalPolicy, env, seed=seed,
        policy_kwargs={'freeze_encoder':frozen}, **PPO_CONFIG)
    initialization=model.policy.adopt_il64(original)
    return model,initialization


def run_arm(original,context,snapshots,*,frozen,seed,milestones,guard,progress,
            evaluate,save_checkpoint,episode_sink,curve_sink,resume=None):
    """Continue in one worker across predeclared64/256/1024; every endpoint evaluated.

    A resumed SB3 checkpoint restores optimizer moments. The new process resets
    to the same uniform genuine-start distribution, not a claimed exact mid-
    episode RNG/physical-state continuation. Resume paths require caller pins.
    """
    if not milestones or tuple(sorted(set(milestones)))!=tuple(milestones) or any(x not in (64,256,1024) for x in milestones):
        raise ValueError('Fixed progressive milestones64/256/1024 required')
    env=UniformCurriculum(context,snapshots,seed=seed,guard=guard,episode_sink=episode_sink)
    model=None; started=time.perf_counter()
    result={'arm':'frozen_encoder' if frozen else 'trainable_encoder','seed':seed,
        'ppo_config':PPO_CONFIG,'milestones':list(milestones),'checkpoints':[],
        'status':'running','progression_policy':'fixed endpoints; no tuning or checkpoint selection'}
    try:
        if resume is None:
            model,initialization=make_model(original,env,seed=seed,frozen=frozen)
            result['initialization']=initialization
            result['initial_evaluation']=evaluate(model,'init')
            original_encoder=initialization['encoder_hash']
        else:
            model=MaskablePPO.load(resume['path'],env=env,device='cpu')
            if type(model.policy) is not LocalGoalPolicy or model.policy.freeze_encoder!=frozen:
                raise ValueError('Resume policy or encoder mode changed')
            if model.num_timesteps!=resume['decisions'] or model.num_timesteps>=milestones[0]:
                raise ValueError('Resume must precede the declared milestones')
            original_encoder=resume['initial_encoder_hash']
            model.policy.costs.update(resume['policy_costs'])
            result['resume']={k:v for k,v in resume.items() if k!='path'}
            result['initial_evaluation']=evaluate(model,'resume')
        callback=Curves(guard,progress)
        for milestone in milestones:
            guard(); before=model.num_timesteps
            model.learn(total_timesteps=milestone-before, reset_num_timesteps=False, callback=callback)
            if model.num_timesteps!=milestone:raise RuntimeError('PPO decision denominator differs')
            encoder=module_hash(model.policy.spatial.encoder)
            if frozen and encoder!=original_encoder:raise RuntimeError('Frozen encoder changed')
            steps=model.policy.costs['optimizer_completed']
            if steps<=0 or steps>milestone//64*64:raise RuntimeError('Actual optimizer step count outside fixed bound')
            checkpoint=save_checkpoint(model,milestone)
            endpoint=evaluate(model,str(milestone))
            row={'decisions':milestone,'checkpoint':checkpoint,'evaluation':endpoint,
                'optimizer_attempted':model.policy.costs['optimizer_attempted'], 'optimizer_completed':steps,
                'sb3_epoch_updates':model._n_updates,'encoder_hash':encoder,
                'encoder_changed':encoder!=original_encoder,'model_hash':module_hash(model.policy.spatial),
                'experience':env.summary(),'policy_costs':dict(model.policy.costs),
                'rollouts':list(callback.rows),'gradient_updates':list(model.policy.update_rows),
                'sb3_train_metrics':{k:float(v) if np.isfinite(v) else None for k,v in model.logger.name_to_value.items()
                                     if k.startswith('train/') and np.isscalar(v)},
                'undefined_metrics':[k for k,v in model.logger.name_to_value.items()
                                     if k.startswith('train/') and np.isscalar(v) and not np.isfinite(v)],
                'elapsed_seconds':time.perf_counter()-started}
            result['checkpoints'].append(row);curve_sink(result)
        result['status']='completed'
    except BaseException as error:
        result['failure']={'type':type(error).__name__,'message':str(error)}
        result['status']='failed_or_incomplete'
        if model is not None:
            result['partial_policy_costs']=dict(model.policy.costs)
            result['partial_update_rows']=list(model.policy.update_rows)
            result['decisions']=model.num_timesteps
        if env.episode is not None:
            result['partial_episode']=env.episode_record(complete=env.episode.done)
        curve_sink(result)
        error.ppo_partial_result=result
        raise
    finally:
        result['experience']=env.summary();result['elapsed_seconds']=time.perf_counter()-started
        env.close()
        if model is not None:model.env=None
    return result
