"""Measured controllers over the exact generated finite-skill inventory."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import time
from types import SimpleNamespace
import numpy as np

from .core import array_digest
from .surgical_sequential_task import (COMMON_CONFIG, execute_skill, endpoint_assessment,
    observation, available_mask, training_admission, SequentialPrimitiveAdapter, SkillExecutionInterrupted)
from .surgical_procedure_policy import evaluate_procedure_sequence
from .surgical_joint_cranial_search import _retain_beam, _key, _dominates
from .surgical_consequences import procedure_priority_choice


class OnlineLimit(SkillExecutionInterrupted):
    pass


@dataclass(frozen=True)
class ComputeTier:
    name: str
    max_physical_calls: int
    max_wall_seconds: float
    horizon: int
    beam_width: int
    max_expansions: int

    def __post_init__(self):
        if (not isinstance(self.name,str) or not self.name or
                type(self.max_physical_calls) is not int or self.max_physical_calls<1 or
                type(self.max_wall_seconds) not in (float,int) or not np.isfinite(self.max_wall_seconds) or self.max_wall_seconds<=0 or
                type(self.horizon) is not int or not 1<=self.horizon<=5 or
                type(self.beam_width) is not int or not 2<=self.beam_width<=4 or
                type(self.max_expansions) is not int or not 1<=self.max_expansions<=24):
            raise ValueError('Bounded finite-skill compute tier required')


class OnlineLedger:
    def __init__(self, tier, deadline=None):
        self.tier = tier
        self.started = time.monotonic()
        self.deadline = deadline
        self.calls = self.planning_calls = self.execution_calls = 0
        self.candidate_generations = 0
        self.candidate_seconds = self.policy_seconds = self.planning_seconds = 0.
        self.failures = []
        self.previews = []
        self.canceled_requests = 0

    def check(self, reserve=0):
        if self.calls + reserve >= self.tier.max_physical_calls:
            raise OnlineLimit('physical-call-limit')
        self.wall_check()

    def wall_check(self):
        if time.monotonic()-self.started >= self.tier.max_wall_seconds:
            raise OnlineLimit('online-wall-limit')
        if self.deadline is not None and time.monotonic()>=self.deadline:
            raise OnlineLimit('campaign-wall-limit')

    def charge(self, planning=False):
        self.check(24 if planning else 0)
        if planning:self.planning_wall_check()
        self.calls += 1
        if planning: self.planning_calls += 1
        else: self.execution_calls += 1

    def planning_wall_check(self):
        self.wall_check()
        if time.monotonic()-self.started >= self.tier.max_wall_seconds*.8:
            raise OnlineLimit('planning-wall-reserve-for-execution')

    def report(self):
        return dict(tier=asdict(self.tier), physicalCalls=self.calls,
            planningPhysicalCalls=self.planning_calls, executionPhysicalCalls=self.execution_calls,
            candidateGenerations=self.candidate_generations, candidateSeconds=self.candidate_seconds,
            policySeconds=self.policy_seconds, planningSeconds=self.planning_seconds,
            wallSeconds=time.monotonic()-self.started, failures=copy.deepcopy(self.failures),
            previews=copy.deepcopy(self.previews),canceledRequests=self.canceled_requests)


def _inventory(task, engine, used, ledger):
    started = time.monotonic()
    ledger.check()
    mask = available_mask(task, engine, used)
    ledger.candidate_generations += 1
    ledger.candidate_seconds += time.monotonic()-started
    return mask


def _node(task, engine, path):
    outcome = evaluate_procedure_sequence(engine)['outcome']
    dims = outcome['dimensions']
    progress = endpoint_assessment(engine, task.config)['progress']
    return SimpleNamespace(work=engine, path=path, goal_assessment=progress,
        sequence_outcome=outcome, ledger=dict(boneRemovedMm3=dims['boneRemovedMm3']['cumulative'],
            preparationNonboneRemovedMm3=dims['nonbonePreparationRemovedMm3']['cumulative'],
            workNonTargetRemovedMm3=dims['neuralRemovedMm3']['cumulative'],
            wholeNonTargetRemovedMm3=dims['nonTargetRemovedMm3']['cumulative'],
            modeledElapsedSeconds=engine.state['elapsedSeconds']))


def _choose(nodes):
    if not nodes: return None
    names = ['targetRemovedMm3','neuralRemovedMm3','neuralCutSourceMm3',
        'protectedRemovedMm3','nonbonePreparationRemovedMm3','boneRemovedMm3','elapsedSeconds']
    structures = sorted(nodes[0].sequence_outcome['dimensions']['structureRemovedMm3'])
    names += ['structure:'+name+':'+effect for name in structures for effect in ('removedMm3','cutSourceMm3')]
    def value(branch, metric):
        node=nodes[int(branch)];dims=node.sequence_outcome['dimensions']
        if metric=='targetRemovedMm3':return node.goal_assessment['originalTargetTreatmentMm3']
        if metric.startswith('structure:'):
            _,name,effect=metric.split(':')
            return dims['structureRemovedMm3' if effect=='removedMm3' else 'structureCutSourceMm3'][name]['cumulative']
        return dims[metric]['cumulative']
    selected,reason=procedure_priority_choice([str(i) for i in range(len(nodes))],value,names)
    return nodes[int(selected)] if selected is not None else None


def constrained_skill_search(task, engine, used, ledger):
    """Adapt the existing constrained beam retention to finite post-access skills.

    Same authoritative transitions, source harm gate, Pareto dominance and beam
    retention as joint cranial search. No bone/membrane stage claims are added
    to supplied post-access scenes. Every discarded branch is charged.
    """
    started = time.monotonic(); frontier = [(engine, (), set(used))]
    stop_engine=engine.clone();stop_skill=next(s for s in task.skills if s.kind=='stop')
    stop_receipt=execute_skill(task,stop_engine,stop_skill,charge=lambda:ledger.charge(True),check=ledger.planning_wall_check)
    ledger.canceled_requests+=int(bool(stop_receipt['interruption'] and stop_receipt['interruption']['canceledRequest']))
    ledger.previews.append(dict(sourceStateId=engine.state_id,skillId=stop_skill.id,receipt=stop_receipt,
        prefixSnapshot=stop_engine.snapshot() if stop_receipt['status']!='accepted' else None))
    nodes, attempts = ([_node(task,stop_engine,())] if stop_receipt['status']=='accepted' else []), []
    expansions = 0
    try:
        for depth in range(ledger.tier.horizon):
            children = []
            for parent, path, parent_used in frontier:
                mask = _inventory(task, parent, parent_used, ledger)
                for index, skill in enumerate(task.skills):
                    if not mask[index] or skill.kind == 'stop': continue
                    if expansions >= ledger.tier.max_expansions: break
                    ledger.check(24)
                    branch = parent.clone()
                    expansions += 1
                    receipt = execute_skill(task, branch, skill, charge=lambda: ledger.charge(True),check=ledger.planning_wall_check)
                    ledger.canceled_requests+=int(bool(receipt['interruption'] and receipt['interruption']['canceledRequest']))
                    attempts.append(dict(skillId=skill.id, depth=depth+1, status=receipt['status'],
                        sourceStateId=parent.state_id,
                        acceptedPrefixActions=len(branch.actions)-len(parent.actions),
                        targetIncrementMm3=receipt['targetIncrementMm3'],
                        neuralIncrementMm3=receipt['neuralIncrementMm3'], finalStateId=branch.state_id,
                        reason=next((r['outcome'].get('reasonCode') for r in receipt['receipts']
                            if r['outcome']['status']!='accepted'), None)))
                    ledger.previews.append(dict(sourceStateId=parent.state_id,skillId=skill.id,
                        receipt=receipt,prefixSnapshot=branch.snapshot() if receipt['status']!='accepted' else None))
                    if receipt['status'] != 'accepted': continue
                    step = dict(proposalId=skill.id, preparationOnly=receipt['targetIncrementMm3']<=0,
                        skillIndex=index)
                    node = _node(task, branch, (*path, step))
                    node.used = parent_used | {skill.id}
                    children.append(node); nodes.append(node)
                if expansions >= ledger.tier.max_expansions: break
            if not children or expansions >= ledger.tier.max_expansions: break
            retained = _retain_beam(children, ledger.tier.beam_width)
            frontier = [(n.work, n.path, n.used) for n in retained]
    except OnlineLimit as error:
        ledger.failures.append(dict(stage='search', reason=str(error), expansions=expansions))
    ledger.planning_seconds += time.monotonic()-started
    # All supported neural/structure quantities are source-bound and were
    # enforced precommit. Retain incomparable effects; in this discrete task
    # equal-treatment counts are compared before time, as the shared rule does.
    valid = [n for n in nodes if endpoint_assessment(n.work, task.config)['validWithdrawal']]
    pareto = [n for n in valid if not any(_dominates(o, n) for o in valid)]
    selected = _choose(pareto)
    return ([p['skillIndex'] for p in selected.path] if selected is not None else []), attempts


def greedy_skill(task, engine, used, ledger):
    started=time.monotonic()
    mask = _inventory(task, engine, used, ledger)
    candidates = []
    for index, skill in enumerate(task.skills):
        if not mask[index] or skill.kind=='stop': continue
        branch = engine.clone()
        try:
            receipt = execute_skill(task, branch, skill, charge=lambda: ledger.charge(True),check=ledger.wall_check)
            ledger.canceled_requests+=int(bool(receipt['interruption'] and receipt['interruption']['canceledRequest']))
        except OnlineLimit as error:
            ledger.failures.append(dict(stage='greedy-preview', reason=str(error))); break
        ledger.previews.append(dict(sourceStateId=engine.state_id,skillId=skill.id,
            receipt=receipt,prefixSnapshot=branch.snapshot() if receipt['status']!='accepted' else None))
        if receipt['status']=='accepted' and receipt['targetIncrementMm3'] > 0:
            node=_node(task, branch, (dict(proposalId=skill.id, preparationOnly=False),))
            node.skill_index=index;candidates.append(node)
    ledger.planning_seconds += time.monotonic()-started
    if not candidates:
        return next(i for i, s in enumerate(task.skills) if s.kind=='stop')
    selected=_choose(candidates)
    return selected.skill_index if selected is not None else next(i for i,s in enumerate(task.skills) if s.kind=='stop')


# Every feature is a deterministic reduction of supplied anatomy/current state
# and the current inventory. No preview, learned label, case ID, split, pattern,
# branch outcome, future reward, source hash or held-out ranking is an input.
FEATURE_NAMES = ('kind_prepare','kind_work','kind_stop','current_radius','requested_radius',
    'route_target_cells','route_motor_cells','route_cst_cells','route_critical_cells',
    'route_remaining_target_cells','route_min_neural_distance','route_length',
    'global_target_remaining','global_motor_remaining','global_cst_remaining',
    'remaining_neural_budget','remaining_motor_budget','remaining_cst_budget',
    'remaining_attempts','remaining_seconds','remaining_skills',
    'maximum_route_remaining_target','minimum_route_neural_distance','mean_route_neural_distance',
    'right_inserted','right_active','right_placed','withdrawal_height_distance')


def feature_matrix(observed, skills):
    labels = observed['source_labels']; state = observed['native_state']
    affine = observed['affine']; memberships = observed['memberships']
    removed = set(state['removed']); targets = set(map(int, np.flatnonzero(labels==1)))
    motor = set(memberships['structure:motor-cortex']); cst = set(memberships['structure:corticospinal'])
    critical = set(memberships['structure:brainstem']) | set(memberships['structure:cerebellum'])
    neural = set(memberships['neural'])
    cells = sorted(targets | neural)
    points = np.array(np.unravel_index(cells, labels.shape)).T @ affine[:3,:3].T + affine[:3,3]
    lookup = dict(zip(cells, points))
    current_radius = state['slots']['right']['parameters'].get('radiusMm', 1.5)
    def route_stats(skill):
        if not skill.route: return [0.]*7
        a,b = np.asarray(skill.route); direction=b-a
        distances={c:float(np.linalg.norm(p-(a+np.clip(np.dot(p-a,direction)/max(np.dot(direction,direction),1e-12),0,1)*direction)))
            for c,p in lookup.items()}
        # Fixed source-neighborhood descriptor, not a simulated effect/mask.
        near = {c for c,d in distances.items() if d < 3.}
        return [len(near & targets)/8., len(near & motor)/4., len(near & cst)/4.,
            len(near & critical)/4.,len((near & targets)-removed)/8.,
            min((d for c,d in distances.items() if c in neural-removed),default=30.)/30.,
            float(np.linalg.norm(direction))/60.]
    stats=[route_stats(s) for s in skills]
    workstats=[r for s,r in zip(skills,stats) if s.kind=='work']
    limits=observed['declared_limits']; remaining=observed['remaining']
    global_values=[len(targets-removed)/16.,len(motor-removed)/8.,len(cst-removed)/8.,
        (limits['maximumNeuralRemovalMm3']-len(neural & removed)*27.)/108.,
        (limits['maximumStructureRemovalMm3']['motor-cortex']-len(motor & removed)*27.)/108.,
        (limits['maximumStructureRemovalMm3']['corticospinal']-len(cst & removed)*27.)/108.,
        remaining['attempts']/512.,remaining['seconds']/240.,remaining['skills']/8.,
        max(r[4] for r in workstats),min(r[5] for r in workstats),float(np.mean([r[5] for r in workstats])),
        float(state['slots']['right']['inserted']),float(state['slots']['right']['active']),
        float(any(a['slot']=='right' and a['operation'] in ('place','move','apply','withdraw') for a in observed['accepted_history'])),
        abs(state['slots']['right']['position'][2]-observed['exterior_withdrawal_plane_ras_mm'])/60.]
    rows=[]
    for skill, spatial in zip(skills,stats):
        rows.append([float(skill.kind==k) for k in ('prepare','work','stop')] +
            [current_radius/1.5,(skill.radius or current_radius)/1.5] + spatial + global_values)
    result=np.array(rows,np.float32)
    assert result.shape==(len(skills),len(FEATURE_NAMES)) and np.isfinite(result).all()
    return result


def observed_greedy_search(features, mask):
    """Enumerate available working routes. Rank observed target minus annotated encounters.

    Counts use the same corridor neighborhood as the frozen actor's input.
    Radius preparation has no observed immediate treatment. STOP wins ties at zero.
    """
    def counts(name, scale):
        return features[:, FEATURE_NAMES.index(name)] * scale
    predicted_gain = (counts('route_remaining_target_cells', 8) - counts('route_motor_cells', 4)
                      - counts('route_cst_cells', 4) - counts('route_critical_cells', 4))
    stop, = np.flatnonzero(mask & features[:, FEATURE_NAMES.index('kind_stop')].astype(bool))
    work = np.flatnonzero(mask & features[:, FEATURE_NAMES.index('kind_work')].astype(bool))
    if len(work):
        best = work[np.argmax(predicted_gain[work])]
        if predicted_gain[best] > 0:
            return int(best)
    return int(stop)


def run_episode(task, method, tier, *, selector=None, stochastic=False, seed=0,
        retain_engine=False, gradient_records=False, deadline=None, initial_prepare_radius=None,
        record_decision_inputs=False):
    training_admission(task)
    initial_prepare = None
    if initial_prepare_radius is not None:
        matches = [i for i, skill in enumerate(task.skills)
                   if skill.kind == 'prepare' and skill.radius == initial_prepare_radius]
        if len(matches) != 1:
            raise ValueError('Exactly one declared preparation for the requested initial radius is required')
        initial_prepare = matches[0]
    reset_started=time.monotonic(); engine=task.initial_engine(); reset_seconds=time.monotonic()-reset_started
    ledger=OnlineLedger(tier,deadline); used=set(); history=[]; attempts=invalid=0; stopped=False
    decision_inputs = []
    selections=0; plan=None; search_attempts=[]; gradient=[]; timed_out=False; search_exhausted=False
    try:
        if method=='search' and initial_prepare is None:
            plan,search_attempts=constrained_skill_search(task,engine,used,ledger)
            search_exhausted=not plan
        for selection in range(task.config.max_skills):
            ledger.check(); selections+=1
            if selection == 0 and initial_prepare is not None:
                index = initial_prepare
            elif method=='greedy': index=greedy_skill(task,engine,used,ledger)
            elif method=='search':
                # Lookahead bounds computation, not the physical episode.
                # Replan over the remaining identical inventory when a checked
                # prefix ends, within the original shared online ledger.
                if not plan and not search_exhausted:
                    plan,extra=constrained_skill_search(task,engine,used,ledger)
                    search_attempts.extend(extra);search_exhausted=not plan
                index=plan.pop(0) if plan else next(i for i,s in enumerate(task.skills) if s.kind=='stop')
            elif method in ('learned', 'conditional', 'observed_greedy_search'):
                mask=_inventory(task,engine,used,ledger)
                observed=observation(task,engine,attempts=attempts,invalid=invalid,selections=selection)
                started=time.monotonic()
                if method == 'conditional':
                    index, record = selector(observed, task.skills, mask), {}
                else:
                    features = feature_matrix(observed, task.skills)
                    if record_decision_inputs:
                        decision_inputs.append(dict(features=array_digest(features), mask=array_digest(mask)))
                    if method == 'observed_greedy_search':
                        index, record = observed_greedy_search(features, mask), {}
                    else:
                        index, record = selector(features, mask, stochastic)
                ledger.policy_seconds+=time.monotonic()-started
                if gradient_records: gradient.append(record)
            else: raise ValueError('Unknown finite skill method')
            skill=task.skills[index]
            receipt=execute_skill(task,engine,skill,charge=ledger.charge,check=ledger.wall_check)
            ledger.canceled_requests+=int(bool(receipt['interruption'] and receipt['interruption']['canceledRequest']))
            attempts+=receipt['attemptsConsumed']; invalid+=receipt['invalidAttemptsConsumed']
            # A failed skill retains its accepted prefix and remains an
            # attempted inventory member; no reset/refund/teleport occurs.
            used.add(skill.id)
            history.append(receipt)
            if receipt['endpoint']['truncated']:break
            if skill.kind=='stop' and receipt['status']=='accepted': stopped=True;break
    except OnlineLimit as error:
        ledger.failures.append(dict(stage='episode',reason=str(error)));timed_out=True
    externally_truncated=timed_out or not stopped
    endpoint=endpoint_assessment(engine,task.config,attempts=attempts,invalid_actions=invalid,
        stopped=stopped,externally_truncated=externally_truncated)
    outcome=evaluate_procedure_sequence(engine)['outcome']; dims=outcome['dimensions']
    row=dict(parent=task.audit['parent'],pattern=task.audit['pattern'],method=method,seed=seed,
        tier=tier.name,sourceSceneId=task.scene.identity,configHash=task.config.fingerprint,
        selectedSkillIds=[r['skillId'] for r in history],skillHistory=history,
        searchAttempts=search_attempts,progress=endpoint['progress'],endpoint=endpoint,
        neuralRemovedMm3=dims['neuralRemovedMm3']['cumulative'],neuralCutSourceMm3=dims['neuralCutSourceMm3']['cumulative'],
        structureRemovedMm3={k:v['cumulative'] for k,v in dims['structureRemovedMm3'].items()},
        committedHardViolations=len(evaluate_procedure_sequence(engine)['violations']),
        boneRemovedMm3=dims['boneRemovedMm3']['cumulative'],modeledSeconds=engine.state['elapsedSeconds'],
        attempts=attempts,rejections=invalid,reward=sum(r['reward'] for r in history),
        computation=ledger.report(),resetSeconds=reset_seconds,
        startScope='supplied-post-access-continuation',earlierAccessCosts=None,
        wholeProcedureSuccess=False,finalStateId=engine.state_id,
        snapshot=engine.snapshot(),missingEvidence=outcome['missingEvidence'])
    if initial_prepare is not None:
        row['initialPreparationRadiusMm'] = initial_prepare_radius
    if record_decision_inputs: row['decisionInputs'] = decision_inputs
    if retain_engine: row['_engine']=engine
    if gradient_records: row['_gradient']=gradient
    return row


class ReplayVerificationError(ValueError):
    def __init__(self,message,receipt):
        super().__init__(message);self.receipt=receipt


def verify_replay(task,row, *, deadline=None):
    started=time.monotonic()
    receipt=dict(verified=False,physicalCalls=0,completedPhysicalCalls=0,adapterStopRequests=0)
    try:
        receipt.update(_verify_replay_core(task,row,deadline=deadline,counters=receipt))
    except Exception as error:
        receipt.update(wallSeconds=time.monotonic()-started,errorType=type(error).__name__,error=str(error),
            canceledRequests=receipt['physicalCalls']-receipt['completedPhysicalCalls'])
        raise ReplayVerificationError(str(error),receipt) from error
    receipt.update(wallSeconds=time.monotonic()-started,
        canceledRequests=receipt['physicalCalls']-receipt['completedPhysicalCalls'])
    return receipt


def _verify_replay_core(task,row, *, deadline=None,counters):
    """Actual native and matched primitive replay, including attempted refusals."""
    started=time.monotonic();native=task.initial_engine()
    def replay_check():
        if deadline is not None and time.monotonic()>=deadline:
            raise TimeoutError('Replay wall deadline reached')
    adapter=SequentialPrimitiveAdapter(task.scene,task_config=task.config);adapter.reset(seed=0)
    compared=0;adapter_stopped=False
    for skill in row['skillHistory']:
        for receipt in skill['receipts']:
            command=receipt['action'];counters['physicalCalls']+=1
            actual=native.step(command,check=replay_check);counters['completedPhysicalCalls']+=1
            if actual['status']!=receipt['outcome']['status']:raise ValueError('Native status replay differs')
            counters['physicalCalls']+=1
            result=adapter.step_action(command,check=replay_check);counters['completedPhysicalCalls']+=1;compared+=1
            if result[4]['outcome']['status']!=actual['status'] or adapter.engine.state_id!=native.state_id:
                raise ValueError('Matched primitive replay differs')
            if result[2] or result[3]:adapter_stopped=True
        if skill['kind']=='stop' and skill['checkedStopAttempted'] and not adapter_stopped:
            counters['adapterStopRequests']+=1
            result=adapter.stop('Finite generated inventory exhausted');adapter_stopped=result[2] or result[3]
            counters['adapterStopOutcome'] = result[4]['outcome']
    if native.snapshot()!=row['snapshot'] or native.state_id!=adapter.engine.state_id:
        raise ValueError('Final accepted state replay differs')
    stopped=bool(row['skillHistory'] and row['skillHistory'][-1]['kind']=='stop' and row['skillHistory'][-1]['status']=='accepted')
    if len(adapter.attempts)!=row['attempts'] or adapter.invalid_actions!=row['rejections']:
        raise ValueError('Adapter attempt/refusal accounting differs')
    if not stopped:adapter.truncate_external('Finite skill or online computation boundary')
    expected=endpoint_assessment(native,task.config,attempts=len(adapter.attempts),invalid_actions=adapter.invalid_actions,
        stopped=stopped,externally_truncated=not stopped)
    if expected!=row['endpoint']:raise ValueError('Endpoint replay differs')
    if adapter.completion_assessment!=expected:
        raise ValueError('Adapter terminal classification differs')
    if evaluate_procedure_sequence(native)!=evaluate_procedure_sequence(adapter.engine):
        raise ValueError('Complete accounting replay differs')
    return dict(verified=True,primitiveActions=compared,
        wallSeconds=time.monotonic()-started,finalStateId=native.state_id,classification=expected['classification'],
        adapterTerminationReason=adapter.termination_reason)
