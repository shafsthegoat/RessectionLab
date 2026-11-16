import numpy as np
import pytest
from resectionlab.surgical_preparation_variation import variation_task
from resectionlab.surgical_sequential_task import generated_task,training_admission
from resectionlab.surgical_procedure_policy import procedure_source_memberships

def test_variation_changes_only_one_source_route_and_binds_inventory():
 task=variation_task(4000);base=generated_task(16000)
 assert np.array_equal(task.scene.labels,base.scene.labels)
 assert np.array_equal(task.scene.affine,base.scene.affine)
 assert procedure_source_memberships(task.scene)==procedure_source_memberships(base.scene)
 assert task.scene.properties['surgicalProcedurePolicyV1']==base.scene.properties['surgicalProcedurePolicyV1']
 assert task.config==base.config
 assert task.scene.identity!=base.scene.identity
 original={s.id:s for s in base.skills}
 changed=[s for s in task.skills if s.id not in original]
 assert len(changed)==1 and changed[0].kind=='work'
 assert task.audit['developmentOnly'] and task.audit['parent']==4000
 assert training_admission(task)['available']

def test_future_groups_are_proposal_only_not_executable():
 with pytest.raises(ValueError,match='Development-only'):
  variation_task(8200)

def test_state_rule_changes_order_for_observed_tip_and_used_inventory(monkeypatch):
 """Observation fixture only; the separate two episodes supply physical proof."""
 import copy,importlib.util
 from pathlib import Path
 from resectionlab.surgical_sequential_task import observation,available_mask
 from resectionlab.surgical_sequential_benchmark import feature_matrix
 root=Path(__file__).resolve().parents[1];monkeypatch.syspath_prepend(str(root/'scripts'))
 spec=importlib.util.spec_from_file_location('prefix_rule_test',root/'scripts/run_preparation_state_intervention.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 task=variation_task(4000);engine=task.initial_engine()
 original=observation(task,engine);features=feature_matrix(original,task.skills);mask=available_mask(task,engine,set())
 adaptive=module.PrefixRule(True);fixed=module.PrefixRule(False)
 for rule in (adaptive,fixed):
  first,_=rule(features,mask);assert task.skills[first].kind=='prepare' and task.skills[first].radius==.4
 # Exact relevant post-prefix observation fields and used mask; no engine step.
 observed=copy.deepcopy(original);observed['native_state']['slots']['right']['parameters']['radiusMm']=.4
 observed['native_state']['elapsedSeconds']=.1
 after=feature_matrix(observed,task.skills);after_mask=mask.copy()
 for i,skill in enumerate(task.skills):
  if skill.kind=='prepare':after_mask[i]=skill.radius==1.5
 selected,_=adaptive(after,after_mask)
 assert task.skills[selected].kind=='work' and after[selected,10]*30<3
 selected,_=fixed(after,after_mask)
 assert task.skills[selected].kind=='prepare' and task.skills[selected].radius==1.5
