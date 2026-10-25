#!/usr/bin/env python3
"""Export and canonically reopen a generated readiness demo; no patient or learning."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import signal
import time

import numpy as np

from check_surgical_rl_readiness import (
    ROOT, Budget, execute_requests, file_hash, make_surrogate, require,
    script_requests, status_counts, write_json,
)
from resectionlab.core import CaseData, SourceRef, array_digest
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.imaging import load_case, read_case_artifacts, save_case
from resectionlab.surgical_rl_env import SurgicalRLEnv
from resectionlab.surgical_replay_cache import SurgicalReplayCache
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_verified_cache import VerifiedSurgicalCache

SCHEMA = 'resectionlab.generated-readiness-desktop-demo.v1'
BRANCH = 'generated-cusa-v3-readiness-demo'


def generated_case(scene):
    """Bind the unchanged coupon/profile into a labeled, portable CaseData."""
    return CaseData(
        case_id='GENERATED-RL-READINESS-CUSA-V3-NOT-A-PATIENT',
        mri=scene.labels.astype(np.float32),
        compartments={'Synthetic target': scene.labels == 1}, affine=scene.affine,
        source_refs=(SourceRef(
            'generated-readiness-coupon-v1',
            'synthetic://resectionlab/generated-readiness-cusa-v3-coupon',
            sha256=array_digest(scene.labels), native_frame='RAS+', provenance='simulated'),),
        unknowns=('Generated engineering coupon; not patient anatomy',
                  'No supplied brainstem, functional anatomy or vascular tree',
                  'Uncalibrated CUSA contact-capacity and tissue physics'),
        metadata={'is_synthetic': True, 'generated': True,
                  'anatomy_layers': {'Synthetic target': {
                      'kind': 'tumor', 'color': '#9863db',
                      'provenance': 'Generated 27 mm3 engineering target; not anatomy'}},
                  'surgical_adapter': {'ownershipPrecedence': ['Synthetic target']},
                  'surgical_properties': copy.deepcopy(scene.properties)},
    )


def export_demo(output, budget):
    reference, task = make_surrogate()
    reference.reset(seed=17)
    reference_episode, _, _ = execute_requests(reference, task, script_requests(), budget)
    require(reference._info()['success'], 'Reference scripted coupon did not succeed')
    case = generated_case(reference.scene)
    # Construct the case-bound scene BEFORE execution. Replacing scene IDs in a
    # snapshot after running the standalone coupon would break native binding.
    scene = from_case(case)
    require(scene.identity != reference.scene.identity, 'Case-bound source identity must be distinct')
    require(np.array_equal(scene.labels, reference.scene.labels)
            and np.array_equal(scene.affine, reference.scene.affine)
            and scene.properties == reference.scene.properties, 'Case adapter changed geometry/profile')
    env = SurgicalRLEnv(scene, config=reference.config, reward_config=reference.reward_config)
    env.reset(seed=17)
    episode, prefix, prefix_native = execute_requests(env, task, script_requests(), budget, capture_prefix=True)
    require(env._info()['success'] and env._cleanup_state()['pending'] == [], 'Case-bound script failed cleanup')
    require(env.engine.state == reference.engine.state, 'Case-bound physical state differs from original coupon')
    reference_outcomes = [(row['action'], row['status'], row['reason']) for row in reference.attempts]
    require([(row['action'], row['status'], row['reason']) for row in env.attempts] == reference_outcomes,
            'Case-bound scripted actions/statuses differ')
    require(env.measurements == reference.measurements and env.total_reward == reference.total_reward,
            'Case-bound reward/physical ledger differs')
    require(env.engine.state_id != reference.engine.state_id, 'Different sources must not share state identity')
    native = json.loads(json.dumps(env.engine.snapshot(), allow_nan=False))
    artifacts = {'surgical_toolbox': {
        # No environmentSchema migration is requested: the bridge will use the
        # same from_case(mapping) adapter and explicit saved source properties.
        'mapping': copy.deepcopy(scene.regions), 'branches': {BRANCH: native},
        'branchId': BRANCH, 'position': len(native['actions']), 'parents': {BRANCH: None},
        'timeline': {BRANCH: copy.deepcopy(env.engine.history)},
        'readinessDemo': {'schema': SCHEMA, 'generated': True, 'patient': False,
                          'no_training': True, 'scope': 'this export command only'},
    }}
    workspace = output / 'GENERATED-readiness-cusa-v3-demo.ressectionlab'
    _, save_seconds = budget.timed(lambda: save_case(case, workspace, artifacts=artifacts))
    source_hash = file_hash(workspace, budget)
    reopened = load_case(workspace)
    saved = read_case_artifacts(workspace)['surgical_toolbox']
    require(reopened.semantic_hash == case.semantic_hash, 'Native CaseData source binding changed')
    require(from_case(reopened, saved['mapping']).identity == scene.identity, 'Native adapter scene binding differs')
    require(saved['branches'][BRANCH] == native, 'Native actions/guard/checkpoint changed during save')
    session = BridgeSession(output / 'bridge-transfers')
    session.surgical_replay_cache = SurgicalReplayCache(
        verified_cache=VerifiedSurgicalCache(output / 'verified-cache'))
    calls = []
    def call(operation, values):
        result, seconds = budget.timed(lambda: session.execute(
            operation, values, _Request(operation), lambda *_: None))
        calls.append({'operation': operation, 'command': values.get('command'), 'seconds': seconds})
        return result
    try:
        loaded = call('loadCase', {'path': str(workspace)})
        key = loaded['caseHash']
        require(key == case.semantic_hash, 'Fresh BridgeSession loaded a different source')
        final = call('surgicalRehearsal', {'caseHash': key, 'command': 'inspect', 'branchId': BRANCH})
        require(final['view']['stateId'] == native['stateId'], 'Fresh public canonical inspect differs from endpoint')
        require(final['view']['cusaV3'] == env.engine.state['cusaV3'], 'Fresh endpoint CUSA ledger differs')
        require([(row['status'], row.get('reason')) for row in final['timeline']]
                == [(row['status'], row['reason']) for row in env.attempts], 'Fresh canonical statuses differ')
        boundary = len(prefix['attempts'])
        partial = call('surgicalRehearsal', {
            'caseHash': key, 'command': 'seek', 'position': boundary,
            'expectedStateId': final['view']['stateId']})
        require(partial['view']['stateId'] == prefix_native['stateId'], 'Public pending boundary state differs')
        require(partial['view']['cusaV3'] == prefix_native['checkpoint']['state']['cusaV3'],
                'Public boundary exact pending CUSA state differs')
        pending = sum(partial['view']['cusaV3']['progressMm3'].values())
        require(pending > 0 and partial['view']['cusaV3']['progressExact'], 'Public partial boundary lost pending work')
        returned = call('surgicalRehearsal', {
            'caseHash': key, 'command': 'seek', 'position': len(native['actions']),
            'expectedStateId': partial['view']['stateId']})
        require(returned['view']['stateId'] == native['stateId']
                and returned['view']['cusaV3'] == env.engine.state['cusaV3'], 'Public final return differs')
    finally:
        session.close()
    require(file_hash(workspace, budget) == source_hash, 'Verification modified the saved companion')
    return {'schema': SCHEMA, 'scope': 'this generated exporter only',
            'no_training': True, 'optimizer_updates': 0, 'policy_executions': 0, 'patient_actions': 0,
            'workspace': str(workspace), 'workspace_sha256': source_hash,
            'case_hash': case.semantic_hash, 'standalone_fixture_scene_id': reference.scene.identity,
            'case_bound_scene_id': scene.identity, 'standalone_fixture_state_id': reference.engine.state_id,
            'case_bound_endpoint_state_id': native['stateId'], 'physical_state_equal': True,
            'case_bound_identity_deliberately_distinct': True, 'profile': scene.properties['cusaV3'],
            'scripted_skill_decisions': episode['skill_decisions'],
            'primitive_status_counts': status_counts(row['status'] for row in env.attempts),
            'reference_script_seconds': reference_episode['wall_seconds'],
            'case_bound_script_seconds': episode['wall_seconds'], 'save_seconds': save_seconds,
            'bridge_calls': calls, 'fresh_canonical_endpoint_equal': True,
            'public_partial_boundary': boundary, 'public_pending_capacity_mm3': pending,
            'public_partial_state_id': prefix_native['stateId'],
            'public_final_return_equal': True, 'saved_workspace_bytes_unchanged': True,
            'isolated_verified_cache': str(output / 'verified-cache'),
            'metrics': env.measurements, 'cleanup': env._cleanup_state(),
            'gui_checked': False, 'clinical_validation': 'open', 'patient_qualification': 'open'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New ignored directory below testcases/outputs/')
    parser.add_argument('--max-wall-seconds', type=float, default=30., help='Evaluation deadline in (0,30] seconds')
    args = parser.parse_args()
    if not math.isfinite(args.max_wall_seconds) or not 0 < args.max_wall_seconds <= 30:
        parser.error('--max-wall-seconds must be in (0,30]')
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / 'testcases/outputs').resolve()) or output.exists():
        parser.error('--output must be a new directory below this checkout testcases/outputs/')
    if not hasattr(signal, 'setitimer'):
        parser.error('This bounded exporter requires POSIX setitimer')
    output.mkdir(parents=True)
    budget = Budget(args.max_wall_seconds)
    report = {'schema': SCHEMA, 'scope': 'this generated exporter only',
              'no_training': True, 'optimizer_updates': 0, 'policy_executions': 0, 'patient_actions': 0}
    def timeout(*_):
        raise TimeoutError('Generated export wall-time budget exhausted')
    previous = signal.signal(signal.SIGALRM, timeout)
    code = 0
    try:
        signal.setitimer(signal.ITIMER_REAL, args.max_wall_seconds)
        report.update(export_demo(output, budget))
        report['status'] = 'passed_generated_native_binding_checks'
    except Exception as error:
        report.update(status='incomplete' if isinstance(error, TimeoutError) else 'failed',
                      error={'type': type(error).__name__, 'message': str(error)})
        code = 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        report['wall_seconds'] = time.perf_counter() - budget.started
        write_json(output / 'report.json', report)
    print(json.dumps({'status': report['status'], 'report': str(output / 'report.json'),
                      'wall_seconds': report['wall_seconds']}, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
