"""Read-only regression of the existing private-target boundary; no training.

This probes current behavior, not a new process-isolated boundary. It does not
assert hidden-support or patient-reference isolation that is not implemented.
"""
from dataclasses import replace
from pathlib import Path
import json
import time
import numpy as np
import torch
from resectionlab.core import semantic_digest
from resectionlab.native_spatial_task import NativeSpatialTask, make_native_opening_task
from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
from resectionlab.observed_search import observed_beam_search
from resectionlab.spatial_policy import SpatialPolicy

started = time.perf_counter()
torch.set_num_threads(1)
base = make_native_opening_task()
changed = NativeSpatialTask(replace(base.case, reference_target=np.zeros_like(base.case.reference_target)), max_steps=2)
assert base.case.source_hash == changed.case.source_hash
assert base.case.reference_hash != changed.case.reference_hash
assert base.observation().fingerprint == changed.observation().fingerprint
assert base.candidate_inventory() == changed.candidate_inventory()
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(31)
    policy = SpatialPolicy().eval()
with torch.no_grad():
    logits_a, value_a = policy(base.observation())
    logits_b, value_b = policy(changed.observation())
torch.testing.assert_close(logits_a, logits_b, rtol=0, atol=0)
torch.testing.assert_close(value_a, value_b, rtol=0, atol=0)
arguments = dict(max_calls=128, beam_width=16, seconds=5.0, transition_mode='lazy_planning')
sequence_a, cost_a = observed_beam_search(base, **arguments)
sequence_b, cost_b = observed_beam_search(changed, **arguments)
assert sequence_a == sequence_b
assert cost_a['model_transition_calls'] == cost_b['model_transition_calls']
# This seal is committed before reading either evaluator output.
seal = {'source_hash':base.case.source_hash, 'decision_model_hash':base.decision_model_hash,
        'initial_observation_hash':base.observation().fingerprint, 'sequence':list(sequence_a)}
seal_hash = semantic_digest(seal)
for action in sequence_a:
    left, right = base.step(action), changed.step(action)
    assert left.observation.fingerprint == right.observation.fingerprint
    assert base.candidate_inventory() == changed.candidate_inventory()
assert base.terminated and changed.terminated
left = evaluate_native_spatial_episode(base)
right = evaluate_native_spatial_episode(changed)
assert left['geometry']['feasible'] and right['geometry']['feasible']
assert left['outcomes']['target_removed_mm3'] > right['outcomes']['target_removed_mm3']
assert left['outcomes']['total_reward'] != right['outcomes']['total_reward']
assert semantic_digest(seal) == seal_hash
result = {'status':'passed', 'scope':'existing generated private-target invariance only',
          'training_updates':0,'real_patient_count':0,'hidden_support_isolation_tested':False,
          'file_access_isolation_tested':False,'sequence_seal':seal,'seal_hash':seal_hash,
          'target_removed_mm3':[left['outcomes']['target_removed_mm3'],right['outcomes']['target_removed_mm3']],
          'reward':[left['outcomes']['total_reward'],right['outcomes']['total_reward']],
          'model_transition_calls':[cost_a['model_transition_calls'],cost_b['model_transition_calls']],
          'seconds':time.perf_counter()-started}
Path(__file__).with_name('existing-target-boundary-probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
