"""Gymnasium policy interface over authoritative surgical transitions.

A partial-observation research environment, not clinical guidance. Protection
masks and reward weights are explicit episode inputs. No learning loop is run.

The v2 observation adds normalized remaining step/time/invalid-action budgets,
physical cleanup ledgers in CLEANUP_METRICS order, and the local sampling grid's
millimeter spacing as a one-element ``local_spacing_mm`` array. Candidate
geometry must use that spacing. Global target and injury ledgers remain declared
modeled task observations; info/world/snapshot diagnostics are privileged.
Target-fraction attainment only succeeds after tools are inactive/withdrawn,
grips/holds released, bleeding inactive, and declared fluid/debris limits met.
The default zero-fluid/debris limits are engineering task choices, not clinical
thresholds. v1 snapshots must be evaluated using their original implementation;
their episode and observation semantics cannot be silently migrated to v2.
V2 also binds working radii to the instrument catalog. A smaller tip variant
must be selected while withdrawn/inactive; inserted tools cannot resize.
Scenes explicitly opting into cusaV3 use an assumed contact/time-capacity law;
its rate, capture and tissue response remain uncalibrated. Without that profile,
the shared legacy engine retains instant compatible-cell deletion. Neither
path establishes patient-calibrated mechanics or full RL readiness.
"""
from __future__ import annotations
import copy
from dataclasses import asdict, dataclass, fields
import json
import math
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError('Install the RL extra: pip install -e ".[rl]"') from exc
from .core import semantic_digest
from .surgical_engine import SurgicalEngine, MAX_STEPS
from .surgical_tools import TOOLS
from .surgical_scene import MATERIALS
from .surgical_rl_protection import ProtectionSet, GuardConfig
from .surgical_rl_rewards import RewardConfig, RewardTracker, compute_reward

# v2 deliberately invalidates v1 episode snapshots: success now includes
# cleanup, and the observation declares the remaining episode budgets.
SCHEMA = 'resectionlab.surgical-rl.v2'
ACTION_CONTRACT = 'catalog-radius-fixed-while-inserted.v1'
OPERATIONS = ('place','move','activate','deactivate','apply','hold','grasp','release','exchange','withdraw','extract','inspect')
INSTRUMENTS = tuple(TOOLS)
PARAMETERS = ('radiusMm','depthMm','flowMm3PerSecond','power','jawOpeningMm','amplitudeMa','frequencyHz','pulseWidthUs','rpm','forceN','duty')
PARAMETER_LOW = np.array([.1,.1,0,0,0,0,0,0,0,0,0],dtype=np.float64)
PARAMETER_HIGH = np.array([max(tool['radiusMm'] for tool in TOOLS.values()),20,1000,10,12,20,500,1000,80000,20,1],dtype=np.float64)
OBS_METRICS = ('tumor_removed_mm3','healthy_removed_mm3','bone_removed_mm3','healthy_cut_mm2','deformation_injury_mm3','healthy_thermal_exposure','vessel_damage_mm2','bleeding_generated_mm3','protected_damage_mm3','near_contact_exposure_mm3_seconds','elapsed_seconds','attempts','invalid_actions','remaining_target_fraction')
LOCAL_CHANNELS = ('material','target','protected','fluid','bone_temperature','displacement','cut','known_grid')
CLEANUP_METRICS = ('fluid_present_mm3', 'debris_present_mm3', 'active_bleeding_sources')
BUDGET_METRICS = ('remaining_step_fraction', 'remaining_time_fraction', 'remaining_invalid_fraction')

@dataclass(frozen=True)
class RLConfig:
    max_steps: int = 128
    max_seconds: float = 60.
    max_action_seconds: float = 5.
    success_fraction: float = .95
    max_success_fluid_mm3: float = 0.
    max_success_debris_mm3: float = 0.
    max_healthy_removed_mm3: float = 5.
    max_vessel_damage_mm2: float = .5
    max_bleeding_mm3: float = 10.
    max_invalid_actions: int = 16
    require_critical_coverage: bool = True
    critical_max_displacement_mm: float = 0.
    near_margin_mm: float = 2.
    local_spacing_mm: float = 2.

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self,f.name)
            if f.name == 'require_critical_coverage':
                if type(value) is not bool: raise ValueError(f.name + ' must be boolean')
            elif f.name in ('max_steps','max_invalid_actions'):
                if type(value) is not int or not 1 <= value <= MAX_STEPS: raise ValueError('Invalid ' + f.name)
            elif type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid ' + f.name)
        if not 0 < self.success_fraction <= 1 or not .01 <= self.max_action_seconds <= 30 or self.max_seconds <= 0 or not .1 <= self.local_spacing_mm <= 10:
            raise ValueError('Invalid time, success or observation resolution')

class _ProtectedEngine(SurgicalEngine):
    """Reward-only ledgers; the base engine owns all protection decisions."""
    def __init__(self, scene, protection, config):
        self.rl_config = config
        self.rl_strain_peaks = {}
        self.rl_near_exposure = 0.
        super().__init__(scene, protection=protection, guard_config=GuardConfig(
            require_critical_coverage=config.require_critical_coverage,
            critical_max_displacement_mm=config.critical_max_displacement_mm))

    def fresh(self):
        return type(self)(self.scene, self.protection, self.rl_config)

    def clone(self):
        result = super().clone()
        result.rl_strain_peaks = self.rl_strain_peaks.copy()
        return result

    def _substep(self, key, op, p, q, active, dt):
        from .surgical_motion import swept_envelopes
        nearby = set()
        if self.rl_config.near_margin_mm and self._protected_ids:
            positions = self._positions()
            for other_key, other in self.state['slots'].items():
                end = {'position': p, 'quaternion': q} if other_key == key else other
                for a, b, radius, _ in swept_envelopes(TOOLS[other['instrument']], self.parameters(other), other, end):
                    nearby.update(map(int, self.protection.capsule_hits(a, b,
                        radius + self.rl_config.near_margin_mm, positions=positions)))
        events = super()._substep(key, op, p, q, active, dt)
        # Shared guard checks finish before reward ledgers can change.
        from .surgical_rl_rewards import deformation_scores
        for cell, value in deformation_scores(self).items():
            self.rl_strain_peaks[cell] = max(self.rl_strain_peaks.get(cell, 0.), value)
        self.rl_near_exposure += len(nearby) * self.scene.voxel_mm3 * dt
        return events

class SurgicalRLEnv(gym.Env):
    metadata = {'render_modes':['ansi']}
    def __init__(self,scene,*,protection=None,config=None,reward_config=None,render_mode=None):
        self.scene = scene
        self.config = config or RLConfig()
        self.reward_config = reward_config or RewardConfig()
        self.protection = protection or ProtectionSet(scene)
        if self.protection.scene.identity != scene.identity: raise ValueError('Protection scene mismatch')
        present = {a.structure_id for a in self.protection.annotations if a.cells}
        self.missing_critical = tuple(sorted(set(self.protection.missing) | ({'brainstem'} - present)))
        if self.config.require_critical_coverage and self.missing_critical:
            raise ValueError('MISSING_CRITICAL_ANNOTATIONS: '+', '.join(self.missing_critical))
        if render_mode not in (None,'ansi'): raise ValueError('Supported render_mode: ansi')
        self.render_mode = render_mode
        self._local_offsets = np.stack(np.meshgrid(*([np.arange(-4,5)*self.config.local_spacing_mm]*3),indexing='ij'),axis=-1).reshape(-1,3)
        self._protected = frozenset(map(int,self.protection.cells))
        self._target_labels = [r['id'] for r in scene.regions if r.get('target') is True and r['material']=='tumor' and not r.get('protected') and not r.get('critical')]
        target = np.flatnonzero(np.isin(scene.labels,self._target_labels))
        target = np.asarray([i for i in target if int(i) not in self._protected],np.int64)
        self.initial_target_mm3 = len(target) * scene.voxel_mm3
        if not len(target): raise ValueError('No unprotected explicitly designated tumor is available')
        self.target_center = scene.points(target).mean(axis=0)
        self.protection_id = semantic_digest({'report':self.protection.report(),'cells':self.protection.cells.tolist()})
        self._radius_caps = {name:float(tool['radiusMm']) for name,tool in TOOLS.items()}
        self.action_contract = {'id':ACTION_CONTRACT,'radiusCapsMm':self._radius_caps.copy(),
            'radiusChanges':'withdrawn, inactive and released only'}
        self.environment_id = semantic_digest({'schema':SCHEMA,'scene':scene.identity,'protection':self.protection_id,'config':asdict(self.config),'reward':asdict(self.reward_config),'actionContract':self.action_contract})
        self.action_space = spaces.Dict({'slot':spaces.Discrete(2),'operation':spaces.Discrete(len(OPERATIONS)),
            'instrument':spaces.Discrete(len(INSTRUMENTS)), 'pose':spaces.Box(-1e6,1e6,(7,),np.float64),
            'use_pose':spaces.Discrete(2), 'parameters':spaces.Box(PARAMETER_LOW,PARAMETER_HIGH,dtype=np.float64),
            'parameter_mask':spaces.MultiBinary(len(PARAMETERS)),
            'duration':spaces.Box(.01,self.config.max_action_seconds,(1,),np.float64)})
        self.observation_space = spaces.Dict({'local':spaces.Box(0,1,(2,8,9,9,9),np.float32),
            'slots':spaces.Box(-1e6,1e6,(2,12),np.float32), 'metrics':spaces.Box(0,np.inf,(len(OBS_METRICS),),np.float64),
            'tool_parameters':spaces.Box(np.tile(PARAMETER_LOW,(2,1)),np.tile(PARAMETER_HIGH,(2,1)),dtype=np.float64),
            'goal':spaces.Box(-1e6,1e6,(4,),np.float64), 'coverage':spaces.MultiBinary(3),
            'local_spacing_mm':spaces.Box(.1,10,(1,),np.float64),
            'cleanup':spaces.Box(0,np.inf,(len(CLEANUP_METRICS),),np.float64),
            'budgets':spaces.Box(0,1,(len(BUDGET_METRICS),),np.float64)})
        self.engine = None
        self._done = False

    @classmethod
    def from_patient(cls,path,*,protection_path=None,**kwargs):
        from .surgical_environment import load_environment
        from .surgical_rl_protection import load_protection, restore_protection_contract
        from .imaging import read_case_artifacts
        scene = load_environment(path)
        record = read_case_artifacts(path).get('surgical_toolbox') or {}
        snapshot = record.get('branches', {}).get(record.get('branchId'), {})
        embedded = snapshot.get('protectionContract')
        protection = load_protection(scene,protection_path) if protection_path else None
        if embedded is not None:
            saved_protection, saved_config = restore_protection_contract(scene, embedded)
            if protection is not None:
                from .surgical_rl_protection import protection_contract
                if protection_contract(protection, saved_config) != embedded:
                    raise ValueError('Explicit protection differs from saved environment contract')
            else: protection = saved_protection
            config = kwargs.get('config')
            if config is None:
                kwargs['config'] = RLConfig(require_critical_coverage=saved_config.require_critical_coverage,
                    critical_max_displacement_mm=saved_config.critical_max_displacement_mm)
            elif config.require_critical_coverage != saved_config.require_critical_coverage or config.critical_max_displacement_mm != saved_config.critical_max_displacement_mm:
                raise ValueError('RL guard configuration differs from saved environment contract')
        return cls(scene,protection=protection,**kwargs)

    def reset(self,*,seed=None,options=None):
        if options: raise ValueError('Reset options unsupported; construct a new source-bound environment')
        super().reset(seed=seed)
        if seed is not None: self.action_space.seed(seed)
        self.engine = _ProtectedEngine(self.scene,self.protection,self.config)
        self.tracker = RewardTracker(self.engine,protected_source_ids=self.protection.cells)
        self.attempts = []; self.invalid_actions = 0; self.total_reward = 0.; self._done=False
        self.terminated=False; self.truncated=False; self.termination_reason=None
        self.measurements = self._measure()
        return self._observe(), self._info()

    def _measure(self):
        # The episode's protection source IDs are fixed. Intersect the sparse
        # removal ledger directly; np.intersect1d would sort the full patient
        # protection array even for a no-op with no removed material.
        protected_damage=len(self._protected.intersection(self.engine.state['removed']))*self.scene.voxel_mm3
        return self.tracker.measure(self.engine,{'protected_damage_mm3':protected_damage,
            'near_contact_exposure_mm3_seconds':self.engine.rl_near_exposure,
            'deformation_cell_peaks':self.engine.rl_strain_peaks})

    def encode_action(self,action):
        if self.engine is None: raise RuntimeError('Call reset before encoding actions')
        slot = ('left','right').index(action['slot'])
        operation = OPERATIONS.index(action['operation'])
        points = action.get('path',[action['pose']] if 'pose' in action else [])
        if len(points)>1: raise ValueError('Fixed-space actions contain one endpoint; use step_action for checked paths')
        pose = points[0] if points else self.engine.state['slots'][action['slot']]
        parameters=PARAMETER_LOW.copy(); mask=np.zeros(len(PARAMETERS),np.int8)
        for k,v in action.get('parameters',{}).items():
            i=PARAMETERS.index(k);parameters[i]=v;mask[i]=1
        return dict(slot=slot,operation=operation,instrument=INSTRUMENTS.index(action.get('instrument',self.engine.state['slots'][action['slot']]['instrument'])),
            pose=np.array([*pose['position'],*pose['quaternion']],np.float64),use_pose=int(bool(points)),parameters=parameters,
            parameter_mask=mask,duration=np.array([action.get('duration',1.)],np.float64))

    def decode_action(self,action):
        if not self.action_space.contains(action): raise ValueError('Action is outside its declared space')
        op=OPERATIONS[int(action['operation'])]
        result={'slot':('left','right')[int(action['slot'])],'operation':op,'duration':float(action['duration'][0]),
            'parameters':{k:float(v) for k,v,used in zip(PARAMETERS,action['parameters'],action['parameter_mask']) if used}}
        if op=='exchange': result['instrument']=INSTRUMENTS[int(action['instrument'])]
        if action['use_pose']:
            q=np.asarray(action['pose'][3:],float);norm=np.linalg.norm(q)
            if norm<1e-12: raise ValueError('Quaternion cannot be zero')
            result['pose']={'position':np.asarray(action['pose'][:3],float).tolist(),'quaternion':(q/norm).tolist()}
        return result

    def step(self,action):
        try: action=self.decode_action(action)
        except (ValueError,TypeError,KeyError,IndexError) as error:
            return self._transition(None,str(error))
        return self.step_action(action)

    def step_action(self,action,*,check=None):
        if check is not None: check()
        try:
            # JSON roundtrip detaches caller state and rejects nonserializable actions.
            action=json.loads(json.dumps(action,allow_nan=False))
        except (ValueError,TypeError) as error: return self._transition(None,str(error),check=check)
        return self._transition(action,check=check)

    def _validate_action_radius(self,action):
        slot=self.engine.state['slots'][action['slot']]
        exchange=action['operation']=='exchange'
        instrument=action.get('instrument',slot['instrument']) if exchange else slot['instrument']
        if instrument not in self._radius_caps:raise ValueError('UNKNOWN_INSTRUMENT')
        current=self.engine.parameters(slot)['radiusMm']
        radius=action.get('parameters',{}).get('radiusMm',self._radius_caps[instrument] if exchange else current)
        if radius>self._radius_caps[instrument]:
            raise ValueError('INSTRUMENT_RADIUS_EXCEEDS_CATALOG: select a declared compatible tip while withdrawn')
        if radius!=current and (slot['inserted'] or slot['active'] or slot['grip'] or action['slot'] in self.engine.state['holds']):
            raise ValueError('WITHDRAW_BEFORE_RADIUS_CHANGE: an inserted or active instrument cannot resize')

    def _transition(self,action,error=None,*,check=None):
        if check is not None: check()
        check_failed = []
        def operation_check():
            try: check()
            except BaseException:
                check_failed.append(True)
                raise
        if self.engine is None: raise RuntimeError('Call reset before step')
        if self._done: raise RuntimeError('Episode finished; call reset')
        before=self.measurements.copy(); outcome=None
        try:
            if error: raise ValueError(error)
            if not isinstance(action,dict) or not isinstance(action.get('parameters',{}),dict):
                raise ValueError('INVALID_ACTION: action and parameters must be objects')
            duration,points=self.engine._validate(action)
            self._validate_action_radius(action)
            if duration>self.config.max_action_seconds: raise ValueError('Action exceeds RL duration limit')
            if action['operation']=='place' and len(points)==1:
                p=np.asarray(points[0]['position']);ijk=self.scene.inverse[:3,:3]@p+self.scene.inverse[:3,3]
                radius=action.get('parameters',{}).get('radiusMm',self.engine.parameters(self.engine.state['slots'][action['slot']])['radiusMm'])
                pad=radius*np.linalg.norm(self.scene.inverse[:3,:3],axis=1)
                if np.all((ijk>=-.5-pad)&(ijk<=np.array(self.scene.shape)-.5+pad)):
                    raise ValueError('EXTERIOR_PLACEMENT_REQUIRED: use checked movement to enter anatomy')
            remaining=self.config.max_seconds-self.engine.state['elapsedSeconds']
            if duration>remaining+1e-9: raise ValueError('EPISODE_TIME_REMAINING: action duration exceeds remaining simulated time')
            outcome=self.engine.step(action) if check is None else self.engine.step(action,check=operation_check)
        except (ValueError,TypeError,KeyError,OverflowError) as exc:
            if check_failed: raise
            if check is not None: check()
            outcome={'status':'rejected','reason':str(exc),'reasonCode':str(exc).split(':')[0],'completedFraction':0.,'elapsedSeconds':0.,'stateId':self.engine.state_id,'events':[]}
        if outcome['status']=='rejected':self.invalid_actions+=1
        self.measurements=self._measure()
        reward,terms=compute_reward(before,self.measurements,outcome,self.reward_config)
        self.total_reward+=reward
        self.attempts.append({'action':copy.deepcopy(action),'decodeError':error,'stateId':self.engine.state_id,'reward':reward,'terms':terms,'status':outcome['status'],'reason':outcome.get('reason')})
        m=self.measurements
        failures=[]
        for key,limit in [('healthy_removed_mm3',self.config.max_healthy_removed_mm3),('vessel_damage_mm2',self.config.max_vessel_damage_mm2),('bleeding_generated_mm3',self.config.max_bleeding_mm3)]:
            if m[key]>limit:failures.append(key)
        if m['protected_damage_mm3']>0:failures.append('protected_damage_mm3')
        success=self._target_reached() and not self._cleanup_state()['pending']
        self.terminated=bool(failures or success)
        self.truncated=not self.terminated and (len(self.attempts)>=self.config.max_steps or self.invalid_actions>=self.config.max_invalid_actions or m['elapsed_seconds']>=self.config.max_seconds-1e-9)
        self.termination_reason=('damage_budget:'+','.join(failures)) if failures else 'target_fraction_reached' if success else 'attempt_limit' if len(self.attempts)>=self.config.max_steps else 'invalid_action_limit' if self.invalid_actions>=self.config.max_invalid_actions else 'time_limit' if self.truncated else None
        self._done=self.terminated or self.truncated
        return self._observe(),float(reward),self.terminated,self.truncated,self._info(outcome,terms)

    def _target_reached(self):
        return self.measurements['tumor_removed_mm3']/self.initial_target_mm3>=self.config.success_fraction

    def _cleanup_state(self):
        """Explicit engineering endpoint gates, not clinical clearance criteria."""
        state=self.engine.state
        fluid=sum(state['fluid'].values())+sum(source.get('reservoirMm3',0.) for source in state['sources'])
        debris=float(state['debrisMm3'])
        bleeding=sum(bool(source.get('bleedingActive',True)) for source in state['sources'])
        pending=[]
        for key,slot in state['slots'].items():
            if slot['active']:pending.append(key+':active')
            if slot['inserted']:pending.append(key+':inserted')
            if slot['grip']:pending.append(key+':grip')
        if state['holds']:pending.append('holds')
        if bleeding:pending.append('active_bleeding')
        if fluid>self.config.max_success_fluid_mm3+1e-9:pending.append('fluid')
        if debris>self.config.max_success_debris_mm3+1e-9:pending.append('debris')
        return {'pending':pending,'fluid_present_mm3':fluid,'debris_present_mm3':debris,
            'active_bleeding_sources':bleeding}

    def _observe(self):
        e=self.engine;local=np.zeros((2,8,9,9,9),np.float32);slots=np.zeros((2,12),np.float32)
        tool_parameters=np.tile(PARAMETER_LOW,(2,1))
        offsets=self._local_offsets
        occupancy=e.occupancy(); labels=self.scene.labels.ravel();cut={c for field in ('cutEdges','transectedEdges') for edge in e.state.get(field,[]) for c in edge}
        deformation=e.state.get('deformation',{});temperature=e.state.get('drillingV2',{}).get('temperatureC',{})
        for hand,key in enumerate(('left','right')):
            slot=e.state['slots'][key]
            settings=e.parameters(slot)
            tool_parameters[hand]=[settings.get(name,PARAMETER_LOW[i]) for i,name in enumerate(PARAMETERS)]
            slots[hand]=[*slot['position'],*slot['quaternion'],INSTRUMENTS.index(slot['instrument']),int(slot['active']),int(slot['inserted']),int(key in e.state['holds']),len(slot['grip'])]
            xyz=offsets+np.asarray(slot['position']);ijk=np.rint(xyz@self.scene.inverse[:3,:3].T+self.scene.inverse[:3,3]).astype(int)
            inside=np.all((ijk>=0)&(ijk<self.scene.shape),axis=1);flat=local[hand].reshape(8,-1)
            for j in np.flatnonzero(inside):
                spatial=int(np.ravel_multi_index(ijk[j],self.scene.shape));identity=int(occupancy[spatial])-1
                members=occupancy.memberships.get(spatial,[identity] if identity>=0 else [])
                flat[7,j]=1
                if identity>=0:
                    r=e.by_label[int(labels[identity])];flat[0,j]=(MATERIALS.index(r['material'])+1)/len(MATERIALS)
                flat[1,j]=any(labels[c] in self._target_labels and c not in self._protected for c in members)
                flat[2,j]=spatial in self._protected or any(c in self._protected for c in members)
                flat[3,j]=min(1.,e.state['fluid'].get(str(spatial),0.)/1000.)
                # Only the bone model supplies Celsius. Generic bipolar
                # exposure is a proxy and must not masquerade as temperature.
                flat[4,j]=min(1.,max(0.,temperature.get(str(spatial),37.)/200.))
                flat[5,j]=min(1.,max((np.linalg.norm(deformation.get(str(c),[0,0,0])) for c in members),default=0.)/3.)
                flat[6,j]=any(c in cut for c in members)
        remaining=max(0.,1.-self.measurements['tumor_removed_mm3']/self.initial_target_mm3)
        values={**self.measurements,'attempts':len(self.attempts),'invalid_actions':self.invalid_actions,'remaining_target_fraction':remaining}
        reviewed=bool(self.protection.annotations) and all(a.review_status=='reviewed' for a in self.protection.annotations)
        cleanup=self._cleanup_state()
        return {'local':local,'slots':slots,'tool_parameters':tool_parameters,'metrics':np.array([values[k] for k in OBS_METRICS],np.float64),
            'goal':np.array([*self.target_center,remaining],np.float64),'coverage':np.array([not self.missing_critical,reviewed,False],np.int8),
            'local_spacing_mm':np.array([self.config.local_spacing_mm],np.float64),
            'cleanup':np.array([cleanup[key] for key in CLEANUP_METRICS],np.float64),
            'budgets':np.maximum(0.,np.array([1.-len(self.attempts)/self.config.max_steps,
                1.-self.measurements['elapsed_seconds']/self.config.max_seconds,
                1.-self.invalid_actions/self.config.max_invalid_actions],np.float64))}

    def _info(self,outcome=None,terms=None):
        return {'schema':SCHEMA,'environmentId':self.environment_id,'stateId':self.engine.state_id,
            'action_contract':copy.deepcopy(self.action_contract),
            'config':asdict(self.config),'reward_config':asdict(self.reward_config),
            'metrics':self.measurements.copy(),'reward_terms':dict(terms or {}),'total_reward':self.total_reward,
            'outcome':copy.deepcopy(outcome),'termination_reason':self.termination_reason,'success':self.termination_reason=='target_fraction_reached',
            'target_fraction_reached':self._target_reached(),'cleanup':self._cleanup_state(),
            'protection':self.engine.guard_report(),'observation_semantics':'partial modeled segmentation observations; not a full Markov state',
            'policy_input_contract':'observation only; info, inspect_world, engine, snapshots and proposal diagnostics are privileged audit data',
            'open_mechanics':(['CUSA v3 contact rate, full-fragment capture and tissue response remain uncalibrated engineering assumptions']
                if 'cusaV3' in self.scene.properties else
                ['bounded-duration CUSA removal; shared legacy engine uses instant compatible-cell deletion']),
            'unassessed':['functional motor/language cortex','corticospinal and other white-matter tracts','optic pathways unless separately supplied'],
            'approximations':['uncalibrated dynamics and rewards','conservative registered-mask exclusion zones','local 2 mm sampled observations by default']}

    def inspect_world(self):
        if self.engine is None:raise RuntimeError('Call reset first')
        return self.engine.view()

    def snapshot(self):
        if self.engine is None:raise RuntimeError('Call reset first')
        return {'schema':SCHEMA,'environmentId':self.environment_id,'attempts':copy.deepcopy(self.attempts),
            'actionContract':copy.deepcopy(self.action_contract),
            'config':asdict(self.config),'reward_config':asdict(self.reward_config),
            'stateId':self.engine.state_id,'measurements':self.measurements.copy(),'totalReward':self.total_reward,
            'terminated':self.terminated,'truncated':self.truncated,'rng':copy.deepcopy(self.np_random.bit_generator.state),
            'actionSpaceRng':copy.deepcopy(self.action_space.np_random.bit_generator.state),
            'actionRng':{k:copy.deepcopy(space.np_random.bit_generator.state) for k,space in self.action_space.spaces.items()}}

    def restore(self,record):
        if record.get('schema')!=SCHEMA or record.get('environmentId')!=self.environment_id:raise ValueError('RL snapshot environment identity mismatch')
        if record.get('actionContract')!=self.action_contract:raise ValueError('RL snapshot action contract differs')
        candidate=type(self)(self.scene,protection=self.protection,config=self.config,reward_config=self.reward_config,render_mode=self.render_mode)
        candidate.reset(seed=0)
        for item in record['attempts']:
            candidate._transition(item['action'],item.get('decodeError'))
            if candidate.attempts[-1]!=item:raise ValueError('RL attempted transition replay differs')
        if candidate.engine.state_id!=record['stateId'] or candidate.measurements!=record['measurements'] or candidate.total_reward!=record['totalReward'] or candidate.terminated!=record['terminated'] or candidate.truncated!=record['truncated']:
            raise ValueError('RL replay identity or reward ledger differs')
        candidate.np_random.bit_generator.state=copy.deepcopy(record['rng'])
        candidate.action_space.np_random.bit_generator.state=copy.deepcopy(record['actionSpaceRng'])
        for k,state in record['actionRng'].items():candidate.action_space[k].np_random.bit_generator.state=copy.deepcopy(state)
        self.__dict__.update(candidate.__dict__)
        return self._observe(),self._info()

    def branch(self,position):
        """Replay a physical prefix and fork the caller's current random streams."""
        if type(position) is not int or not 0<=position<=len(self.attempts):raise ValueError('Invalid attempt position')
        result=type(self)(self.scene,protection=self.protection,config=self.config,reward_config=self.reward_config,render_mode=self.render_mode)
        result.reset(seed=0)
        for item in self.attempts[:position]:result._transition(item['action'],item.get('decodeError'))
        result.np_random.bit_generator.state=copy.deepcopy(self.np_random.bit_generator.state)
        result.action_space.np_random.bit_generator.state=copy.deepcopy(self.action_space.np_random.bit_generator.state)
        for key,space in self.action_space.spaces.items():
            result.action_space[key].np_random.bit_generator.state=copy.deepcopy(space.np_random.bit_generator.state)
        return result

    def render(self):
        return json.dumps(self._info(),sort_keys=True) if self.render_mode=='ansi' and self.engine is not None else None

    def close(self):
        self.engine=None
