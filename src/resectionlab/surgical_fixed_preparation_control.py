"""One costed narrow preparation, then the unmodified finite-skill greedy rule."""
from __future__ import annotations
import time
from resectionlab.surgical_sequential_benchmark import OnlineLedger, OnlineLimit, greedy_skill
from resectionlab.surgical_sequential_task import execute_skill, endpoint_assessment, training_admission
from resectionlab.surgical_procedure_policy import evaluate_procedure_sequence

METHOD = 'fixed-narrow-greedy'

def run_fixed_preparation(task, tier, *, deadline=None):
    """Share the original ledger, eight selections, engine and endpoint rules."""
    training_admission(task)
    narrow = [i for i,s in enumerate(task.skills) if s.kind=='prepare' and s.radius==.4]
    if len(narrow)!=1: raise ValueError('Exactly one declared0.4mm preparation required')
    started=time.monotonic();engine=task.initial_engine();reset_seconds=time.monotonic()-started
    ledger=OnlineLedger(tier,deadline);used=set();history=[];attempts=invalid=0;stopped=timed_out=False
    try:
        for selection in range(task.config.max_skills):
            ledger.check()
            # The sole intervention; later decisions call the historical rule.
            index=narrow[0] if selection==0 else greedy_skill(task,engine,used,ledger)
            skill=task.skills[index]
            receipt=execute_skill(task,engine,skill,charge=ledger.charge,check=ledger.wall_check)
            ledger.canceled_requests+=int(bool(receipt['interruption'] and receipt['interruption']['canceledRequest']))
            attempts+=receipt['attemptsConsumed'];invalid+=receipt['invalidAttemptsConsumed']
            used.add(skill.id);history.append(receipt)
            if receipt['endpoint']['truncated']:break
            if skill.kind=='stop' and receipt['status']=='accepted':stopped=True;break
    except OnlineLimit as error:
        ledger.failures.append(dict(stage='episode',reason=str(error)));timed_out=True
    endpoint=endpoint_assessment(engine,task.config,attempts=attempts,invalid_actions=invalid,
        stopped=stopped,externally_truncated=timed_out or not stopped)
    sequence=evaluate_procedure_sequence(engine);dims=sequence['outcome']['dimensions']
    return dict(parent=task.audit['parent'],pattern=task.audit['pattern'],method=METHOD,seed=0,
        tier=tier.name,sourceSceneId=task.scene.identity,configHash=task.config.fingerprint,
        selectedSkillIds=[r['skillId'] for r in history],skillHistory=history,searchAttempts=[],
        progress=endpoint['progress'],endpoint=endpoint,
        neuralRemovedMm3=dims['neuralRemovedMm3']['cumulative'],neuralCutSourceMm3=dims['neuralCutSourceMm3']['cumulative'],
        structureRemovedMm3={k:v['cumulative'] for k,v in dims['structureRemovedMm3'].items()},
        committedHardViolations=len(sequence['violations']),boneRemovedMm3=dims['boneRemovedMm3']['cumulative'],
        modeledSeconds=engine.state['elapsedSeconds'],attempts=attempts,rejections=invalid,
        reward=sum(r['reward'] for r in history),computation=ledger.report(),resetSeconds=reset_seconds,
        startScope='supplied-post-access-continuation',earlierAccessCosts=None,wholeProcedureSuccess=False,
        finalStateId=engine.state_id,snapshot=engine.snapshot(),missingEvidence=sequence['outcome']['missingEvidence'],
        diagnosticIntervention=dict(radiusMm=.4,preparations=1,
            selectionBudget='Preparation consumes one of the same eight selections',
            laterPolicy='Unchanged original greedy_skill; no extra previews or privileged outcomes'))
