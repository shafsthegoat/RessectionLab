"""Small observation-only skill interface over the public surgical RL API.

These are parameterized engineering primitives, not learned skills or clinical
procedures. Poses come explicitly from the caller; proposals never search the
source anatomy, preview transitions, or promise that a corridor is safe. Exact
material/protection checks remain authoritative at execution. Receipts stay in
memory; callers choose when to save or independently audit an episode.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from .core import array_digest, semantic_digest
from .surgical_rl_env import INSTRUMENTS, OBS_METRICS, RLConfig
from .surgical_tools import TOOLS

SCHEMA = 'resectionlab.surgical-skills.v1'
TASK_HORIZONS = {
    'access': (128, 300.),
    'resection': (4096, 7200.),
    'field_management': (256, 300.),
}
# No automatic search for a route, tissue, wound or fluid destination.
EFFECT_TOOLS = {'drill': 'cranial_drill', 'resect': 'cusa',
                'aspirate': 'frazier_suction', 'irrigate': 'irrigation',
                'coagulate': 'bipolar_forceps'}
SKILLS = ('inspect', 'approach', *EFFECT_TOOLS, 'release', 'withdraw', 'stop')
_OBSERVATION_KEYS = frozenset(('local','slots','tool_parameters','metrics','goal',
    'coverage','local_spacing_mm','cleanup','budgets'))


@dataclass(frozen=True)
class SkillTask:
    name: str
    mode: str = 'prevention'

    def __post_init__(self):
        if self.name not in TASK_HORIZONS:
            raise ValueError('Unknown skill task')
        if self.mode not in ('prevention', 'preinjured_rescue'):
            raise ValueError('Declare prevention or preinjured_rescue')

    def config(self, **overrides):
        """Explicit defaults; a full case must still declare its own budgets."""
        steps, seconds = TASK_HORIZONS[self.name]
        return RLConfig(**{'max_steps': steps, 'max_seconds': seconds, **overrides})


def _json_detach(value):
    import json
    return json.loads(json.dumps(value, allow_nan=False))


def propose_skill(observation, name, *, slot='left', pose=None, parameters=None,
                  duration=.1, instrument=None):
    """Return a nominal primitive sequence from published observations only.

    Tool changes require an observed withdrawn, inactive, released slot. The
    caller supplies each endpoint; approach uses checked movement, never place.
    Missing local support is reported as unknown rather than a feasibility mask.
    STOP releases and withdraws controls; it cannot clean fluid or seal wounds.
    """
    if name not in SKILLS or slot not in ('left', 'right'):
        raise ValueError('Unknown skill or slot')
    if not isinstance(observation,dict) or set(observation)!=_OBSERVATION_KEYS:
        raise ValueError('Expected only the published surgical RL v2 observation fields')
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not .01 <= duration <= 30.:
        raise ValueError('Skill duration must be in [0.01, 30] seconds')
    parameters = _json_detach({} if parameters is None else parameters)
    if not isinstance(parameters, dict):raise ValueError('Skill parameters must be an object')
    pose = _json_detach(pose)
    if pose is not None:
        if not isinstance(pose,dict) or set(pose) != {'position','quaternion'}:
            raise ValueError('An explicit endpoint needs position and quaternion')
        position=np.asarray(pose['position'],float);quaternion=np.asarray(pose['quaternion'],float)
        if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion)<1e-12:
            raise ValueError('Invalid explicit endpoint')
    slots=np.asarray(observation['slots'],float)
    if slots.shape != (2,12) or not np.isfinite(slots).all():raise ValueError('Published slot observations required')
    actions=[]
    def add(operation, hand=slot, **values):
        actions.append({'slot':hand,'operation':operation,'duration':float(duration),**values})
    row=slots[('left','right').index(slot)]
    required=EFFECT_TOOLS.get(name,instrument)
    selected=required or INSTRUMENTS[int(row[7])]
    current_instrument=INSTRUMENTS[int(row[7])]
    current_radius=float(observation['tool_parameters'][('left','right').index(slot),0])
    radius=parameters.get('radiusMm',TOOLS[selected]['radiusMm'] if selected!=current_instrument else current_radius) if selected in TOOLS else None
    if radius is not None:
        if isinstance(radius,bool) or not isinstance(radius,(int,float)) or not math.isfinite(radius) or radius<.1 or radius>TOOLS[selected]['radiusMm']:
            raise ValueError('Skill radius must fit the declared instrument catalog')
        if radius!=current_radius and np.any(row[8:]):
            raise ValueError('Withdraw, deactivate and release before changing physical radius')
    if instrument is not None and name in EFFECT_TOOLS and instrument != required:
        raise ValueError('Skill instrument differs from its declared mechanism')
    if required is not None:
        if required not in INSTRUMENTS:raise ValueError('Unknown skill instrument')
        if int(row[7]) != INSTRUMENTS.index(required):
            if np.any(row[8:]):raise ValueError('Withdraw, deactivate and release before changing skill instrument')
            add('exchange',instrument=required)
    if name=='stop':
        if pose is not None or parameters or instrument is not None:
            raise ValueError('STOP accepts no route, parameters or instrument')
        for hand,state in zip(('left','right'),slots):
            if state[8]:add('deactivate',hand)
            if state[10] or state[11]:add('release',hand)
            if state[9]:add('withdraw',hand)
    elif name=='approach':
        if pose is None:raise ValueError('Approach requires an explicit checked endpoint')
        if row[8]:add('deactivate')
        if row[10] or row[11]:raise ValueError('Release held material before an access approach')
        add('move',pose=pose,parameters=parameters)
    elif name in EFFECT_TOOLS:
        add('apply',parameters=parameters,**({'pose':pose} if pose is not None else {}))
    else:
        if pose is not None:raise ValueError('This control skill does not accept a route')
        if name=='withdraw' and (row[10] or row[11]):
            raise ValueError('Release held material before withdrawal')
        add(name,parameters=parameters)
    local=np.asarray(observation['local'])
    if local.shape != (2,8,9,9,9):raise ValueError('Published local observations required')
    support='sampled_local_grid' if np.any(local[('left','right').index(slot),7]) else 'unknown_local_support'
    return {'schema':SCHEMA,'skill':name,'slot':slot,'nominal':True,
        'support':support,'feasibility':'unknown_until_public_execution',
        'observationId':semantic_digest({key:array_digest(value) for key,value in observation.items()}),
        'actions':actions}


class SkillExecutor:
    """Execute bounded primitives with cancellation, retaining committed prefixes.

    A skill is not atomic: cancellation rolls back its current engine primitive,
    while earlier committed primitives remain recorded. STOP ends this executor
    and reports the actual environment success/cleanup state without setting it.
    """
    def __init__(self, env, task):
        if not isinstance(task, SkillTask):raise TypeError('Explicit SkillTask required')
        if env.engine is None:raise RuntimeError('Reset the raw environment first')
        observation=env._observe()
        injuries=('healthy_removed_mm3','healthy_cut_mm2','deformation_injury_mm3',
            'healthy_thermal_exposure','vessel_damage_mm2','bleeding_generated_mm3','protected_damage_mm3')
        preinjured=observation['cleanup'][2]>0 or any(
            observation['metrics'][OBS_METRICS.index(key)]>0 for key in injuries)
        if task.mode=='prevention' and preinjured:
            raise ValueError('Observed entry injury/bleeding requires explicit preinjured_rescue task')
        self.env=env;self.task=task;self.stopped=False

    def execute(self, name, *, check=None, max_wall_seconds=30., **request):
        if self.stopped:raise RuntimeError('Skill executor stopped')
        if isinstance(max_wall_seconds,bool) or not isinstance(max_wall_seconds,(int,float)) or not math.isfinite(max_wall_seconds) or max_wall_seconds<=0:
            raise ValueError('Finite positive skill deadline required')
        deadline=time.perf_counter()+max_wall_seconds
        def checked():
            if check is not None:check()
            if time.perf_counter()>=deadline:raise TimeoutError('Skill wall-time budget exhausted')
        checked()
        plan=propose_skill(self.env._observe(),name,**request)
        if any(action['duration']>self.env.config.max_action_seconds for action in plan['actions']):
            raise ValueError('Skill primitive exceeds configured action duration')
        rows=[];status='completed';total=0.
        for action in plan['actions']:
            checked()
            if self.env._done:status='episode_finished';break
            _,reward,terminated,truncated,info=self.env.step_action(action,check=checked)
            rows.append({'action':copy.deepcopy(action),'reward':reward,
                'stateId':info['stateId'],'status':info['outcome']['status'],
                'reason':info['outcome'].get('reason'),
                'reasonCode':info['outcome'].get('reasonCode')})
            total+=reward
            if info['outcome']['status']!='accepted':status=info['outcome']['status'];break
            if terminated or truncated:status='episode_finished';break
        checked()
        if name=='stop':self.stopped=True
        info=self.env._info()
        return {'schema':SCHEMA,'task':asdict(self.task),'plan':plan,'status':status,
            'horizon':{'max_steps':self.env.config.max_steps,'max_seconds':self.env.config.max_seconds,
                'max_action_seconds':self.env.config.max_action_seconds},
            'stopped':self.stopped,'transitions':rows,'reward':total,
            'terminated':self.env.terminated,'truncated':self.env.truncated,
            'success':info['success'],'cleanup':info['cleanup'],'stateId':info['stateId']}
