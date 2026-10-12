"""Generated scalar scheduling/real SB3 checkpoint and curve transport controls."""
from pathlib import Path
import sys,io,json,zipfile
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'model'))
from learning import PPO_CONFIG,STARTS,ranges,UniformCurriculum
from checkpoints import checkpoint_bytes
from test_policy import fixture
from policy import LocalGoalPolicy,module_hash
import numpy as np
import pytest
import gymnasium as gym
from sb3_contrib import MaskablePPO


def test_declared_configuration_and_finite_metrics():
    assert (PPO_CONFIG['n_steps'],PPO_CONFIG['batch_size'],PPO_CONFIG['n_epochs'])==(64,4,4)
    assert (PPO_CONFIG['gamma'],PPO_CONFIG['gae_lambda'],PPO_CONFIG['target_kl'])==(1.,1.,.03)
    assert STARTS==(12,13,14,15,16)
    assert ranges([-2.,0.,10.])['min']==-2.
    with pytest.raises(ValueError):ranges([float('nan')])


def test_recompressed_SB3_optimizer_checkpoint_no_last_images():
    _,_,source,batch=fixture()
    class Tiny(gym.Env):
        observation_space=source.observation_space;action_space=source.action_space
        def reset(self,*,seed=None,options=None):
            super().reset(seed=seed);return {k:v[0].detach().numpy().copy() for k,v in batch.items()},{}
        def step(self,action):return self.reset()[0],float(action==1),True,False,{}
        def action_masks(self):return batch['action_mask'][0].bool().numpy()
    model=MaskablePPO(LocalGoalPolicy,Tiny(),n_steps=4,batch_size=2,n_epochs=1,seed=3,device='cpu')
    model.policy.load_state_dict(source.state_dict(),strict=True);model.learn(4)
    raw,record=checkpoint_bytes(model)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        data=json.loads(archive.read('data'))
        assert '_last_obs' not in data and '_last_original_obs' not in data
        assert all(v.compress_type==zipfile.ZIP_DEFLATED for v in archive.infolist())
    restored=MaskablePPO.load(io.BytesIO(raw),env=Tiny(),device='cpu')
    assert module_hash(restored.policy.spatial)==module_hash(model.policy.spatial)
    assert len(restored.policy.optimizer.state)==len(model.policy.optimizer.state)>0
    assert all(int(v['step'])==2 for v in restored.policy.optimizer.state.values())
    assert record['last_image_observations_excluded'] and record['includes_optimizer_state']


def test_curriculum_uniform_real_start_dispatch_without_oracle(monkeypatch):
    # The real native qualification checks source snapshots. This unit isolates
    # only the added scheduler and proves it passes actual declared start keys.
    from gym_adapter import LocalGoalEnv
    calls=[]
    def parent_reset(self,*,seed=None,options=None):
        calls.append(options['start_step']);return {},{'start_step':options['start_step']}
    monkeypatch.setattr(LocalGoalEnv,'reset',parent_reset)
    env=object.__new__(UniformCurriculum);env.curriculum_rng=np.random.default_rng(0)
    env.counts={'resets':0};env.starts={str(s):0 for s in STARTS}
    for i in range(200):env.reset(seed=0 if i==0 else None)
    assert set(calls)==set(STARTS) and sum(env.starts.values())==200
    with pytest.raises(ValueError):env.reset(options={'start_step':16})
    first=list(calls);calls.clear();env.reset(seed=0)
    assert calls[0]==first[0]
