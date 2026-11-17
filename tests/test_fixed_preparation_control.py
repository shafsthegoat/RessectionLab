"""The fixed intervention consumes real shared budget and keeps partial state."""
from resectionlab.surgical_fixed_preparation_control import run_fixed_preparation
from resectionlab.surgical_sequential_benchmark import ComputeTier,verify_replay
from resectionlab.surgical_sequential_task import generated_task

def test_rerun_refuses_to_replace_existing_attempts_before_loading_environment(tmp_path):
 import importlib.util
 from pathlib import Path
 import pytest
 path=Path(__file__).resolve().parents[1]/'scripts/run_fixed_preparation_diagnostic.py'
 spec=importlib.util.spec_from_file_location('fixed_runner_preservation',path)
 runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
 (tmp_path/'results').mkdir()
 with pytest.raises(ValueError,match='Preserve existing'):
  runner.run(tmp_path,[200])

def test_fixed_preparation_spends_shared_request_and_preserves_cap_prefix():
 task=generated_task(1000)
 row=run_fixed_preparation(task,ComputeTier('one-request',1,120.,2,2,8))
 assert row['computation']['physicalCalls']==1
 assert row['computation']['executionPhysicalCalls']==1
 assert row['computation']['planningPhysicalCalls']==0
 assert row['attempts']==1 and row['rejections']==0
 assert len(row['skillHistory'])==1 and row['skillHistory'][0]['kind']=='prepare'
 assert row['snapshot']['checkpoint']['state']['slots']['right']['parameters']['radiusMm']==.4
 assert row['endpoint']['truncated'] and not row['endpoint']['terminated']
 assert row['progress']['originalTargetTreatmentMm3']==0
 assert row['reward']<0
 assert verify_replay(task,row)['verified']

def test_fixed_preparation_then_original_greedy_real_replay_and_costs():
 task=generated_task(1000)
 row=run_fixed_preparation(task,ComputeTier('development-test',96,120.,2,2,8))
 history=row['skillHistory'];receipts=[r for s in history for r in s['receipts']]
 assert history[0]['kind']=='prepare' and history[0]['receipts'][0]['action']['parameters']=={'radiusMm':.4}
 assert sum(s['kind']=='prepare' for s in history)==1
 assert len(history)<=task.config.max_skills
 assert history[-1]['kind']=='stop' and history[-1]['checkedStopAttempted']
 assert row['endpoint']['validWithdrawal'] and not row['endpoint']['truncated']
 assert row['endpoint']['debrisMm3']==row['endpoint']['fluidMm3']==0
 assert row['neuralRemovedMm3']==row['committedHardViolations']==0
 assert row['computation']['executionPhysicalCalls']==len(receipts)
 assert row['computation']['physicalCalls']==len(receipts)+row['computation']['planningPhysicalCalls']
 assert row['computation']['planningPhysicalCalls']>0
 assert row['attempts']==len(receipts)+1
 assert row['earlierAccessCosts'] is None and not row['wholeProcedureSuccess']
 assert verify_replay(task,row)['verified']
