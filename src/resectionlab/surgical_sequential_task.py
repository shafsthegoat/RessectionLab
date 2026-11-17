"""Generated-only finite skills over checked transitions, with a shared endpoint.

This is a discrete post-access engineering task. It does not admit the primitive
evaluation adapter to general training, acquired anatomy, or dynamic physics.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import time
import numpy as np

from .core import semantic_digest
from .surgical_engine import SurgicalEngine
from .surgical_procedure_policy import (PROPERTY, make_procedure_policy,
    procedure_source_memberships, evaluate_procedure_sequence)
from .surgical_joint_cranial_search import assess_original_target_progress
from .surgical_rl_env import SurgicalRLEnv, RLConfig
from .surgical_scene import SurgicalScene

TASK_PROPERTY = 'sequentialSkillTaskV1'
SCHEMA = 'resectionlab.generated-sequential-skills.v2'


class SkillExecutionInterrupted(RuntimeError):
    """A compute limit interrupts a skill without erasing accepted work."""


EXCLUDED = ('vascularGraphV1', 'bipolarThermalV1', 'brainContinuumV1',
    'tissuePhysics', 'tissuePhysicsV2', 'cusaV3', 'cranialAccessV1', 'cranialMillingV1')


@dataclass(frozen=True)
class SequentialTaskConfig:
    schema: str = 'resectionlab.sequential-task-config.v1'
    max_attempts: int = 512
    max_modeled_seconds: float = 240.
    max_primitive_seconds: float = 30.
    max_invalid_actions: int = 16
    max_skills: int = 8
    minimum_target_fraction: float = .5
    minimum_zones: int = 3
    minimum_zone_fraction_of_original: float = .05
    maximum_endpoint_fluid_mm3: float = 0.
    maximum_endpoint_debris_mm3: float = 0.
    discount_per_second: float = .997
    require_explicit_stop: bool = True

    def __post_init__(self):
        numeric = (self.max_modeled_seconds, self.max_primitive_seconds, self.minimum_target_fraction,
            self.minimum_zone_fraction_of_original, self.discount_per_second)
        if any(type(v) not in (float, int) or not np.isfinite(v) for v in numeric):
            raise ValueError('Finite engineering configuration required')
        if (self.schema != 'resectionlab.sequential-task-config.v1' or
                type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 10000 or
                type(self.max_skills) is not int or not 1 <= self.max_skills <= 32 or
                type(self.max_invalid_actions) is not int or not 1 <= self.max_invalid_actions <= 10000 or
                not 0 < self.max_modeled_seconds or not .01 <= self.max_primitive_seconds <= 30 or
                not .5 <= self.minimum_target_fraction <= 1 or not 3 <= self.minimum_zones <= 8 or
                not .05 <= self.minimum_zone_fraction_of_original <= 1 or
                self.maximum_endpoint_fluid_mm3 != 0 or self.maximum_endpoint_debris_mm3 != 0 or
                not 0 < self.discount_per_second <= 1 or self.require_explicit_stop is not True):
            raise ValueError('Unsupported finite sequential task configuration')

    @property
    def fingerprint(self):
        return semantic_digest(asdict(self))

    def rl_config(self):
        return RLConfig(max_steps=self.max_attempts, max_seconds=self.max_modeled_seconds,
            max_action_seconds=self.max_primitive_seconds, max_invalid_actions=self.max_invalid_actions,
            max_success_fluid_mm3=self.maximum_endpoint_fluid_mm3,
            max_success_debris_mm3=self.maximum_endpoint_debris_mm3, near_margin_mm=0.)


COMMON_CONFIG = SequentialTaskConfig()


def endpoint_physical_state(engine, config=COMMON_CONFIG):
    """Shared physical cleanup and whole-body withdrawal; no state changes."""
    from .surgical_task_domain import _posed
    from .surgical_motion import envelopes
    from .surgical_tools import TOOLS
    from .surgical_rl_protection import ProtectionSet, ProtectionAnnotation, grid_binding
    procedure = PROPERTY in engine.scene.properties
    state = engine.state
    pending = []
    intracranial = procedure_source_memberships(engine.scene)['intracranial'] if procedure else None
    index = getattr(engine.scene, '_sequential_exterior_index', None)
    if procedure and index is None:
        annotation = ProtectionAnnotation('sequential-intracranial', tuple(sorted(intracranial)),
            grid_binding(engine.scene), 'Generated declared intracranial coverage', 'unreviewed', 'intracranial')
        index = ProtectionSet(engine.scene, [annotation], required=(), clearance_mm=0.)
        engine.scene._sequential_exterior_index = index
    for key, slot in state['slots'].items():
        if slot['active'] or slot['inserted'] or slot['grip'] or key in state['holds']:
            pending.append(key + ':withdrawal')
        if procedure and _posed(engine, key) and any(len(index.capsule_hits(a, b, radius)) for a, b, radius, _ in
                envelopes(TOOLS[slot['instrument']], engine.parameters(slot), slot['position'], slot['quaternion'])):
            pending.append(key + ':not-exterior')
        elif not procedure and _posed(engine, key):
            from .surgical_cranial_guards import tool_is_exterior
            if getattr(engine, '_cranial_access', None) is None or not tool_is_exterior(engine, key):
                pending.append(key + ':not-exterior')
    if state.get('detachedFragments'): pending.append('detached-fragments')
    fluid = sum(state['fluid'].values()) + sum(s.get('reservoirMm3', 0.) for s in state['sources'])
    if fluid > config.maximum_endpoint_fluid_mm3 + 1e-9: pending.append('fluid')
    if state['debrisMm3'] > config.maximum_endpoint_debris_mm3 + 1e-9: pending.append('debris')
    if any(s.get('bleedingActive', True) for s in state['sources']): pending.append('bleeding')
    return dict(pending=pending, cleanupComplete=not pending, fluidMm3=fluid, debrisMm3=float(state['debrisMm3']))


def endpoint_assessment(engine, config=COMMON_CONFIG, *, attempts=None, invalid_actions=0,
        stopped=False, externally_truncated=False, additional_violations=(), completion_blockers=()):
    """Shared native/adapter classification, including complete-tool exterior."""
    progress = assess_original_target_progress(engine,
        minimum_target_fraction=config.minimum_target_fraction, minimum_zones=config.minimum_zones,
        minimum_zone_fraction_of_original=config.minimum_zone_fraction_of_original)
    if (not isinstance(additional_violations, (list, tuple)) or
            not isinstance(completion_blockers, (list, tuple)) or
            any(not isinstance(v, str) or not v for v in (*additional_violations, *completion_blockers))):
        raise ValueError('Endpoint blockers must be explicit additional reasons')
    procedure = PROPERTY in engine.scene.properties
    if not procedure:
        from .surgical_planning_admission import require_planning_admission
        require_planning_admission(engine)
    evaluation = evaluate_procedure_sequence(engine) if procedure else {'admissible': True}
    physical = endpoint_physical_state(engine, config)
    state = engine.state;pending = physical['pending'];fluid = physical['fluidMm3']
    attempts = len(engine.actions) if attempts is None else attempts
    truncated = bool(externally_truncated or attempts >= config.max_attempts or
        invalid_actions >= config.max_invalid_actions or state['elapsedSeconds'] >= config.max_modeled_seconds - 1e-9)
    admissible = evaluation['admissible'] and not additional_violations
    valid = not pending and admissible
    completed = valid and progress['completedProgressRequirement'] and stopped and not completion_blockers
    classification = ('constraint-failure' if not admissible else
        'truncated' if truncated else 'successful-treatment' if completed else
        'valid-stop-with-residual' if stopped and valid else 'in-progress')
    return dict(schema=config.schema, configHash=config.fingerprint, progress=progress,
        completedTask=completed and not truncated, completeOriginalTargetRemoval=progress['residualOriginalTargetCells'] == 0,
        validWithdrawal=valid, cleanupComplete=not pending, pending=pending, fluidMm3=fluid,
        debrisMm3=float(state['debrisMm3']), classification=classification,
        terminated=classification in ('constraint-failure', 'successful-treatment', 'valid-stop-with-residual'),
        truncated=truncated, clinicalValidation=False)


class SequentialPrimitiveAdapter(SurgicalRLEnv):
    """Matched endpoint adapter; still evaluation-only, never the learner itself."""
    def __init__(self, scene, *, task_config=COMMON_CONFIG, **kwargs):
        self.task_config = task_config
        supplied = kwargs.pop('config', task_config.rl_config())
        if supplied != task_config.rl_config():
            raise ValueError('SEQUENTIAL_CONFIG_PARITY_REQUIRED')
        super().__init__(scene, config=supplied, **kwargs)
        self.schema = 'resectionlab.surgical-rl.matched-sequential-evaluation.v1'
        self.environment_id = semantic_digest(dict(parent=self.environment_id,
            schema=self.schema, taskConfig=asdict(task_config)))

    def reset(self, **kwargs):
        self._sequential_stop_attempt = False
        self.external_truncation_reason = None
        return super().reset(**kwargs)

    def _reconstruction_kwargs(self):
        return {'task_config': self.task_config}

    def truncate_external(self, reason):
        if not isinstance(reason,str) or not reason:
            raise ValueError('Explicit external truncation reason required')
        self.external_truncation_reason=reason
        self.completion_assessment=endpoint_assessment(self.engine,self.task_config,
            attempts=len(self.attempts),invalid_actions=self.invalid_actions,externally_truncated=True)
        self.terminated=False;self.truncated=True;self._done=True;self.termination_reason=reason
        return self.completion_assessment

    def snapshot(self):
        return {**super().snapshot(),'externalTruncationReason':self.external_truncation_reason}

    def _restore_episode_boundary(self, record):
        if record.get('externalTruncationReason'):
            self.truncate_external(record['externalTruncationReason'])

    def _transition(self, action, error=None, *, check=None):
        self._sequential_stop_attempt = isinstance(action, dict) and action.get('operation')=='stop'
        result = super()._transition(action, error, check=check)
        stopped = (self._sequential_stop_attempt and result[4]['outcome']['status']=='accepted')
        assessment = endpoint_assessment(self.engine, self.task_config,
            attempts=len(self.attempts), invalid_actions=self.invalid_actions, stopped=stopped)
        self.terminated, self.truncated = assessment['terminated'], assessment['truncated']
        self._done = self.terminated or self.truncated
        self.termination_reason = ('target_fraction_reached' if assessment['classification']=='successful-treatment' else
            'stopped_with_residual' if assessment['classification']=='valid-stop-with-residual' else
            'attempt_limit' if self.truncated and len(self.attempts)>=self.config.max_steps else
            'invalid_action_limit' if self.truncated and self.invalid_actions>=self.config.max_invalid_actions else
            'time_limit' if self.truncated else None)
        self.completion_assessment = assessment
        return self._observe(), result[1], self.terminated, self.truncated, self._info(result[4]['outcome'], result[4]['reward_terms'])

    def _assess_completion(self):
        if self.engine is None: return None
        # A supplied post-access source is explicitly a continuation. It makes
        # no opening claim. Opening demonstrations retain native stage checks.
        result = endpoint_assessment(self.engine, self.task_config,
            attempts=len(getattr(self, 'attempts', [])), invalid_actions=getattr(self, 'invalid_actions', 0),
            stopped=getattr(self, '_sequential_stop_attempt', False))
        if self._cranial_stage_record is not None:
            from .surgical_cranial_stages import stage_readiness
            ready = stage_readiness(self.engine) or {}
            if not ready.get('workEntrySupported') or not ready.get('wholeExitComplete'):
                result['completedTask'] = False
        return result


def action(operation, position=None, **extra):
    result = dict(slot='right', operation=operation, duration=.1)
    result.update(extra)
    if position is not None:
        result['pose'] = dict(position=list(map(float, position)), quaternion=[0., 0., 0., 1.])
    return result


@dataclass(frozen=True)
class Skill:
    id: str
    kind: str
    actions: tuple
    route: tuple = ()
    radius: float | None = None


@dataclass
class GeneratedTask:
    scene: SurgicalScene
    skills: tuple
    audit: dict
    config: SequentialTaskConfig = COMMON_CONFIG

    def initial_engine(self):
        return SurgicalEngine(self.scene)


def generated_task(parent: int, *, config=COMMON_CONFIG, neural_budget=None, motor_shift=0):
    """Prospective geometry, masks and routes; no expected winner in observations.

    Split assignment and pattern are audit-only. Geometry shifts, lengths,
    entrance offsets, target masks and motor memberships vary by parent seed.
    """
    rng = np.random.default_rng(parent + 1729)
    shape = (22, 22, 48)
    labels = np.zeros(shape, np.uint16)
    affine = np.diag([3., 3., 3., 1.])
    shift = rng.integers(-1, 2, 2)
    columns = [(8 + int(shift[0]), 8 + int(shift[1])), (12 + int(shift[0]), 8 + int(shift[1])),
        (8 + int(shift[0]), 12 + int(shift[1])), (12 + int(shift[0]), 12 + int(shift[1]))]
    rng.shuffle(columns)
    pattern = parent % 4
    lengths = [1, int(rng.integers(2, 4)), int(rng.integers(2, 4)), int(rng.integers(1, 3))]
    zbase = 29 + int(rng.integers(-1, 2))
    flat = lambda xyz: int(np.ravel_multi_index(xyz, shape))
    motor, cst, critical = set(), set(), {flat((2, 2, zbase)), flat((2, 3, zbase))}
    for i, (x, y) in enumerate(columns):
        labels[x, y, zbase:zbase + lengths[i]] = 1
        if i > 0 and pattern in (0, 1):
            gate = (x + 1 + motor_shift, y, zbase - 2)
            labels[gate] = 2; motor.add(flat(gate)); cst.add(flat(gate))
        if pattern == 2:
            # Direct target/motor overlap preserves both memberships.
            for z in range(zbase, zbase + min(lengths[i], 2 if i == 1 else 1)):
                motor.add(flat((x, y, z)))
                if i % 2: cst.add(flat((x, y, z)))
        if pattern == 3 and i == 3 or pattern == 1 and i == 2:
            # A declared critical intersection; treatment cannot relabel it.
            critical.update(flat((x, y, z)) for z in range(zbase, zbase + lengths[i]))
    labels[2, 2, zbase] = labels[2, 3, zbase] = 2
    regions = [dict(id=1, name='Generated target', material='tumor', target=True),
        dict(id=2, name='Generated neural owner', material='soft_tissue')]
    source = dict(kind='generated-control', provenance='Prospective generated post-access finite-skill geometry',
        evidenceClass='generated-engineering')
    base = SurgicalScene(labels, affine, regions, source, {})
    # Domain coverage is independently supplied, including explicit negative
    # generated annotations; empty categorical labels do not establish it.
    intracranial = [flat((x, y, z)) for x in range(2, 19) for y in range(2, 19) for z in range(zbase-3, zbase+5)]
    evidence = dict(origin='independent-source-declaration', reviewStatus='generated-engineering',
        provenance='Stylized generated motor cortex/CST anatomy, declared before any decisions; no patient function',
        evidenceSha256=semantic_digest(dict(labels=labels.tolist(), motor=sorted(motor), cst=sorted(cst), critical=sorted(critical))).split(':')[-1])
    brainstem = {flat((2, 2, zbase))}
    cerebellum = critical - brainstem
    structures = {name: dict(evidence, kind='neural', cells=sorted(cells), coverageCells=intracranial)
        for name, cells in [('brainstem', brainstem), ('cerebellum', cerebellum), ('motor-cortex', motor), ('corticospinal', cst)]}
    budget = float((2 if pattern == 2 else 0) * 27 if neural_budget is None else neural_budget)
    raw = make_procedure_policy(base, policy_id='generated-post-access-sequential-v1', evidence=evidence,
        structures=structures, intracranial_cells=intracranial, observed_cells=range(labels.size),
        entry_regions={'supplied-post-access-domain': dict(evidence, cells=sorted(set(intracranial)-critical))},
        permitted_compartments={'supplied-work': dict(evidence, cells=sorted(set(intracranial)-critical))},
        work_domain=dict(evidence, minimumVoxelBoundary=[-.5]*3, maximumVoxelBoundary=(np.array(shape)-.5).tolist()),
        tissue_permissions={'tumor': dict(remove=True, cut=False), 'soft_tissue': dict(remove=True, cut=False)},
        neural_max_mm3=budget, bone_max_mm3=0., preparation_nonbone_max_mm3=0.,
        modeled_time_max_seconds=config.max_modeled_seconds,
        structure_max_mm3={'brainstem':0., 'cerebellum':0., 'motor-cortex':budget, 'corticospinal':budget})
    skills = []
    for radius in (.4, 1.5):
        actions = (action('exchange', instrument='cusa', parameters={'radiusMm':radius}),)
        skills.append(Skill(semantic_digest(actions)[7:19], 'prepare', actions, radius=radius))
    for (x, y), length in zip(columns, lengths):
        # Fractional approach is declared geometry, independent of outcomes.
        xy = [x*3.+.4, y*3.]
        exterior = [*xy, 60.]
        end = [*xy, (zbase+length-1)*3.]
        commands = (action('place', exterior), action('apply', end, duration=1.), action('withdraw', exterior, duration=1.))
        skills.append(Skill(semantic_digest(commands)[7:19], 'work', commands,
            route=(tuple(exterior), tuple(end))))
    stop = Skill('STOP', 'stop', ())
    skills.append(stop)
    # Candidate order is independently shuffled, never a quality ranking.
    rng.shuffle(skills)
    declaration = dict(schema=SCHEMA, config=asdict(config), configHash=config.fingerprint,
        inventory=[asdict(s) for s in skills], start='supplied-post-access', earlierAccessCosts='unknown',
        exteriorWithdrawalPlaneRasMm=60.,
        stopExpansion='Required checked withdrawal at current RAS x/y to the supplied exterior plane, then checked STOP; every primitive, attempt and time increment is charged.',
        physics='discrete-compatible-cell-removal', excluded=list(EXCLUDED),
        motorMeaning='Generated source memberships/direct removal only. Functional motor/language and encounter unknown.')
    scene = SurgicalScene(labels, affine, regions, source, {PROPERTY:raw, TASK_PROPERTY:declaration})
    return GeneratedTask(scene, tuple(skills), dict(parent=parent, pattern=pattern,
        geometryHash=semantic_digest(dict(labels=labels.tolist(), affine=affine.tolist())),
        sourceSceneId=scene.identity, configHash=config.fingerprint), config=config)


def training_admission(task):
    """Narrow generated-only admission, distinct from general primitive RL."""
    from .surgical_planning_admission import source_classification, require_planning_admission
    require_planning_admission(task.scene)
    declaration = task.scene.properties.get(TASK_PROPERTY)
    if (source_classification(task.scene) != 'generated-engineering' or declaration is None or
            declaration.get('schema') != SCHEMA or declaration.get('configHash') != task.config.fingerprint or
            declaration.get('exteriorWithdrawalPlaneRasMm')!=60. or
            declaration.get('inventory') != [asdict(s) for s in task.skills] or
            set(task.scene.properties)!={PROPERTY,TASK_PROPERTY} or
            any(name in task.scene.properties for name in EXCLUDED) or
            task.scene.properties[PROPERTY]['startStage'] != 'post-access' or
            any(s.kind not in ('prepare', 'work', 'stop') for s in task.skills)):
        raise ValueError('GENERATED_FINITE_SKILL_TRAINING_ADMISSION_REFUSED')
    for skill in task.skills:
        operations=tuple(a.get('operation') for a in skill.actions)
        if (skill.kind=='stop' and (skill.id!='STOP' or skill.actions or skill.route) or
                skill.kind=='prepare' and (operations!=('exchange',) or skill.radius not in (.4,1.5) or skill.route or
                    skill.actions[0].get('parameters')!={'radiusMm':skill.radius}) or
                skill.kind=='work' and (operations!=('place','apply','withdraw') or len(skill.route)!=2 or
                    tuple(skill.actions[0].get('pose',{}).get('position',()))!=skill.route[0] or
                    tuple(skill.actions[1].get('pose',{}).get('position',()))!=skill.route[1] or
                    skill.actions[2].get('pose')!=skill.actions[0].get('pose')) or
                skill.kind!='stop' and skill.id!=semantic_digest(skill.actions)[7:19]):
            raise ValueError('GENERATED_FINITE_SKILL_TRAINING_INVENTORY_REFUSED')
        for a in skill.actions:
            if (a.get('slot') != 'right' or a.get('operation') not in ('place', 'apply', 'withdraw', 'exchange') or
                    a.get('instrument', 'cusa') != 'cusa' or
                    a.get('duration', 1.) > task.config.max_primitive_seconds):
                raise ValueError('GENERATED_FINITE_SKILL_TRAINING_INVENTORY_REFUSED')
    return dict(schema='resectionlab.generated-finite-skill-training-admission.v1', available=True,
        scope=SCHEMA, sourceSceneId=task.scene.identity, configHash=task.config.fingerprint,
        generalPrimitiveTraining=False, patientTraining=False, dynamicPhysicsTraining=False)


def observation(task, engine, *, attempts=0, invalid=0, selections=0):
    """Complete supplied anatomy/current modeled state; no IDs or future results."""
    return dict(source_labels=task.scene.labels.copy(), affine=task.scene.affine.copy(),
        memberships=procedure_source_memberships(task.scene), native_state=copy.deepcopy(engine.state),
        accepted_history=[copy.deepcopy(a) for a in engine.actions],
        declared_limits={k: copy.deepcopy(v) for k, v in task.scene.properties[PROPERTY].items()
            if k.startswith('maximum')},
        exterior_withdrawal_plane_ras_mm=task.scene.properties[TASK_PROPERTY]['exteriorWithdrawalPlaneRasMm'],
        remaining=dict(attempts=task.config.max_attempts-attempts,
            invalid=task.config.max_invalid_actions-invalid, skills=task.config.max_skills-selections,
            seconds=task.config.max_modeled_seconds-engine.state['elapsedSeconds']))


def available_mask(task, engine, used):
    slot=engine.state['slots']['right']
    radius = engine.parameters(slot)['radiusMm']
    can_exchange=not (slot['inserted'] or slot['active'] or slot['grip'] or 'right' in engine.state['holds'])
    return np.array([s.kind == 'stop' or s.id not in used and
        (s.kind != 'prepare' or can_exchange and abs(radius-s.radius)>1e-8) for s in task.skills], bool)


def execute_skill(task, engine, skill, *, charge=None, check=None):
    """Sequential checked transitions. Every accepted prefix survives failure."""
    admission = training_admission(task)
    if engine.scene is not task.scene or not any(skill is candidate for candidate in task.skills):
        raise ValueError('SEQUENTIAL_BOUND_INVENTORY_AND_ENGINE_REQUIRED')
    accounting = copy.deepcopy(getattr(engine, '_sequential_accounting', None))
    if accounting is None:
        if engine.actions or engine.history or engine.state_id!=engine.fresh().state_id:
            raise ValueError('SEQUENTIAL_UNCHARGED_PREFIX_REFUSED')
        accounting = dict(sourceSceneId=admission['sourceSceneId'], configHash=task.config.fingerprint,
            attempts=0, invalid=0, skills=0, acceptedActions=0, stateId=engine.state_id, stopped=False)
    if (accounting['sourceSceneId']!=task.scene.identity or accounting['configHash']!=task.config.fingerprint or
            accounting['acceptedActions']!=len(engine.actions) or accounting['stateId']!=engine.state_id):
        raise ValueError('SEQUENTIAL_EPISODE_BINDING_CHANGED')
    if (accounting['stopped'] or accounting['attempts']>=task.config.max_attempts or
            accounting['invalid']>=task.config.max_invalid_actions or accounting['skills']>=task.config.max_skills):
        raise RuntimeError('SEQUENTIAL_EPISODE_FINISHED')
    engine._sequential_accounting = accounting
    accounting['skills'] += 1
    before = evaluate_procedure_sequence(engine)['outcome']['dimensions']
    invalid_before=accounting['invalid']
    started = time.monotonic(); receipts = []; interruption=None
    commands=list(skill.actions)
    if skill.kind=='stop' and not endpoint_assessment(engine,task.config)['validWithdrawal']:
        slot=engine.state['slots']['right']
        commands=[action('withdraw',[slot['position'][0],slot['position'][1],
            task.scene.properties[TASK_PROPERTY]['exteriorWithdrawalPlaneRasMm']],duration=1.)]
    for template in commands:
        command=copy.deepcopy(template)
        from .surgical_task_domain import _posed
        if command['operation']=='place' and _posed(engine,command['slot']):
            # Only an unposed tool uses initial exterior placement. Later
            # relocations use checked swept motion, including empty-space travel.
            command['operation']='move'
        if accounting['attempts']>=task.config.max_attempts or accounting['invalid']>=task.config.max_invalid_actions:
            break
        request_started=False
        try:
            if charge: charge()
            request_started=True
            if command.get('duration', 1.) > task.config.max_modeled_seconds-engine.state['elapsedSeconds']+1e-9:
                outcome = dict(status='rejected', reasonCode='EPISODE_TIME_REMAINING',
                    reason='EPISODE_TIME_REMAINING: action duration exceeds remaining simulated time',
                    completedFraction=0., elapsedSeconds=0., stateId=engine.state_id, events=[], deltas={})
            else:
                outcome = engine.step(copy.deepcopy(command),check=check)
        except (SkillExecutionInterrupted, TimeoutError, InterruptedError) as error:
            interruption=dict(reason=str(error),command=command,acceptedPrefixActions=len(receipts),
                canceledRequest=request_started)
            break
        accounting['attempts'] += 1
        accounting['invalid'] += int(outcome['status']=='rejected')
        accounting['acceptedActions'] = len(engine.actions); accounting['stateId'] = engine.state_id
        receipts.append(dict(action=copy.deepcopy(command), outcome=outcome))
        if outcome['status'] != 'accepted': break
    after = evaluate_procedure_sequence(engine)['outcome']['dimensions']
    delta = lambda key: after[key]['cumulative'] - before[key]['cumulative']
    # Diagnostic reward is computed only from accepted primitive increments.
    # Rejected attempts incur the explicit attempt charge, no treatment credit.
    target, neural, bone, seconds = (delta(k) for k in
        ('targetRemovedMm3', 'neuralRemovedMm3', 'boneRemovedMm3', 'elapsedSeconds'))
    stop_attempted=(skill.kind=='stop' and interruption is None and
        all(r['outcome']['status']=='accepted' for r in receipts) and len(receipts)==len(commands) and
        accounting['attempts']<task.config.max_attempts and accounting['invalid']<task.config.max_invalid_actions)
    attempts_consumed = len(receipts) + int(stop_attempted)
    if stop_attempted: accounting['attempts']+=1
    if stop_attempted and not endpoint_assessment(engine,task.config)['validWithdrawal']:
        accounting['invalid']+=1
    endpoint = endpoint_assessment(engine, task.config, attempts=accounting['attempts'],
        invalid_actions=accounting['invalid'], stopped=stop_attempted,
        externally_truncated=interruption is not None or accounting['skills']>=task.config.max_skills and not stop_attempted)
    accounting['stopped'] = endpoint['terminated'] or endpoint['truncated']
    reward = target/27. - neural/27. - .05*bone/27. - .01*seconds - .01*attempts_consumed
    return dict(skillId=skill.id, kind=skill.kind, receipts=receipts,
        status='interrupted' if interruption else 'truncated' if endpoint['truncated'] else
            receipts[-1]['outcome']['status'] if receipts and receipts[-1]['outcome']['status']!='accepted' else
            'rejected' if skill.kind=='stop' and not endpoint['validWithdrawal'] else 'accepted',
        attemptsConsumed=attempts_consumed,
        invalidAttemptsConsumed=accounting['invalid']-invalid_before,checkedStopAttempted=stop_attempted,
        interruption=interruption,
        reward=reward, elapsedSeconds=seconds, discount=task.config.discount_per_second**seconds,
        targetIncrementMm3=target, neuralIncrementMm3=neural, boneIncrementMm3=bone,
        finalStateId=engine.state_id, wallSeconds=time.monotonic()-started, endpoint=endpoint)
