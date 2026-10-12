"""Small Gymnasium boundary over certified native local episodes; no reward oracle."""
import paths
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from resectionlab.core import semantic_digest
from resectionlab.spatial_policy import world_to_sample_grid
from resectionlab.public_target_context import public_target_features
from local_goal import LocalEpisode,goal_geometry

MAX_ACTIONS=153
RAY_SAMPLES=5
REFERENCE_MM=10.


def public_arrays(catalog,context):
    """SpatialPolicy-compatible normalization, before any trainable encoder."""
    obs=catalog.observation;obs.assert_intact();task=catalog.work
    g=np.asarray(obs.action_geometry,np.float64);a=np.asarray(obs.affine_ras_mm,np.float64)
    shape=obs.image_channels.shape[1:];n=len(obs.action_ids)
    state=np.asarray(obs.state_features,np.float64).copy();inverse=np.linalg.inv(a[:3,:3])
    visible=obs.coverage & obs.channel_available[:,None,None,None]
    images=np.where(visible,obs.image_channels,0.).astype(np.float32)
    fractions=np.linspace(0.,1.,RAY_SAMPLES)
    points=g[:,None,1:4]+fractions[None,:,None]*(g[:,None,4:7]-g[:,None,1:4])
    grid=world_to_sample_grid(points,a,shape);grid[0]=2.
    inside=(np.abs(grid)<=1.).all(axis=-1)
    geometry=g.copy()
    geometry[:,1:4]=world_to_sample_grid(g[:,1:4],a,shape)[...,::-1]
    geometry[:,4:7]=world_to_sample_grid(g[:,4:7],a,shape)[...,::-1]
    directions=g[:,7:10]@inverse.T;norms=np.linalg.norm(directions,axis=-1,keepdims=True)
    geometry[:,7:10]=directions/np.where(norms>0,norms,1.)
    geometry[:,10:14]/=REFERENCE_MM;geometry[:,14]/=90.;geometry[0]=0.;geometry[0,0]=1.
    state[0]/=state[1];state[1]/=10.
    state[3:6]=world_to_sample_grid(state[None,3:6],a,shape)[0,::-1]
    normal=inverse@state[6:9];state[6:9]=normal/np.linalg.norm(normal);state[9]/=REFERENCE_MM
    target,relation=public_target_features(obs,reference_mm=REFERENCE_MM)
    axes=a[:3,:3]/np.linalg.norm(a[:3,:3],axis=0)
    goal=np.asarray(context.goal.tip_mm);access=np.asarray(obs.state_features[3:6])
    relative=(goal-g[:,4:7])@axes/REFERENCE_MM
    goal_relation=np.c_[relative,np.linalg.norm(relative,axis=1)];goal_relation[0]=0.
    goal_global=np.r_[1.,(goal-access)@axes/REFERENCE_MM]
    goal_mask,goal_vector=goal_geometry(context,task)
    def pad(value,fill=0):
        out=np.full((MAX_ACTIONS,*value.shape[1:]),fill,np.float32);out[:n]=value;return out
    mask=np.zeros(MAX_ACTIONS,dtype=np.uint8);mask[:n]=obs.action_mask
    result={'images':images,'coverage':visible.astype(np.uint8),'available':obs.channel_available.astype(np.uint8),
        'geometry':pad(geometry),'grid':pad(grid,2.),'inside':pad(inside),
        'state':state.astype(np.float32),'spacing':np.asarray(obs.spacing_mm/REFERENCE_MM,np.float32),
        'public_target':np.asarray(target,np.float32),'public_relation':pad(relation),
        'goal_global':np.asarray(goal_global,np.float32),'goal_relation':pad(goal_relation),
        'goal_mask':goal_mask,'goal_geometry':goal_vector,
        'focus_active':np.asarray([catalog.record['goal_focus'] is not None],np.uint8),'action_mask':mask}
    if any(not np.isfinite(v).all() for v in result.values()):raise ValueError('Finite public Gym tensors required')
    return result


class LocalGoalEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,context,snapshots,*,eval_mode=False,guard=lambda:None):
        super().__init__();self.context=context;self.guard=guard;self.eval_mode=bool(eval_mode)
        self._snapshots={};self._snapshot_seals={}
        for step,task in snapshots.items():
            context.require_task(task)
            if (type(step) is not int or step!=task._steps or task.terminated
                or task._active_page is not None or task._base_inventory is not None
                or task._engine.removed_mask[context.goal.cell]):raise ValueError('Genuine clean pre-success snapshot required')
            self._snapshots[step]=task.clone();self._snapshot_seals[step]=self._stamp(task)
        start=context.record['prefix_length']
        if start not in self._snapshots:raise ValueError('Qualification start must remain available')
        self.default_start=start;self.episode=None;self.last_observation=None;self.reset_count=0
        shape=tuple(snapshots[start].case._crop_shape)
        f=lambda s:spaces.Box(-np.inf,np.inf,s,dtype=np.float32)
        b=lambda s:spaces.Box(0,1,s,dtype=np.uint8)
        self.action_space=spaces.Discrete(MAX_ACTIONS)
        self.observation_space=spaces.Dict({'images':f((6,*shape)),'coverage':b((6,*shape)),'available':b((6,)),
            'geometry':f((MAX_ACTIONS,16)),'grid':f((MAX_ACTIONS,RAY_SAMPLES,3)),'inside':f((MAX_ACTIONS,RAY_SAMPLES)),
            'state':f((10,)),'spacing':f((3,)),'public_target':f((16,)),'public_relation':f((MAX_ACTIONS,4)),
            'goal_global':f((4,)),'goal_relation':f((MAX_ACTIONS,4)),'goal_mask':f((1,*shape)),
            'goal_geometry':f((11,)),'focus_active':b((1,)),'action_mask':b((MAX_ACTIONS,))})
    @staticmethod
    def _stamp(task):
        task._assert_frozen();return semantic_digest({'state':task._engine.state_hash,'history':task._history,
            'task_seal':task._state_seal,'steps':task._steps,'masks':task._engine.committed_mask_digests()})
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);options={} if options is None else options
        if set(options)-{'start_step'}:raise ValueError('Only a qualified curriculum start may change')
        step=options.get('start_step',self.default_start)
        if self.eval_mode and step!=self.default_start:raise ValueError('Evaluation always starts at fixed prefix')
        if step not in self._snapshots or self._stamp(self._snapshots[step])!=self._snapshot_seals[step]:
            raise ValueError('Unknown or mutated qualified snapshot')
        self.episode=LocalEpisode(self.context,self._snapshots[step],guard=self.guard);self.reset_count+=1
        self.last_observation=public_arrays(self.episode.observe(),self.context)
        return self.last_observation,{'start_step':step,'goal_hash':self.context.record['goal_hash'],
            'snapshot_seal':self._snapshot_seals[step],'role':'TRAIN','source_condition_count':1}
    def action_masks(self):
        if self.episode is None:raise ValueError('Reset required')
        return np.asarray(self.last_observation['action_mask'],bool).copy()
    def step(self,action):
        if self.episode is None or self.episode.done:raise ValueError('Live reset episode required')
        if isinstance(action,(bool,np.bool_)) or not isinstance(action,(int,np.integer)):
            raise ValueError('Integer public candidate index required')
        reward,done,info=self.episode.step(int(action))
        if done:
            # Terminal placeholder carries no stale candidates/certificates.
            out={k:np.array(v,copy=True) for k,v in self.last_observation.items()}
            for k in ('geometry','grid','inside','public_relation','goal_relation','action_mask'):out[k][...]=0
            out['geometry'][0,0]=1.;out['action_mask'][0]=1
            out['state'][0]=self.episode.work._steps/self.episode.work.max_steps
            self.last_observation=out
        else:self.last_observation=public_arrays(self.episode.observe(),self.context)
        return self.last_observation,float(reward),bool(done),False,info
    def close(self):
        self.episode=None;self.last_observation=None;self._snapshots.clear();self._snapshot_seals.clear()
