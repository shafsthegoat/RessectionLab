#!/usr/bin/env python3
"""Bounded fixed-input surgical latency/profile receipts; never trains a policy."""
from pathlib import Path
from dataclasses import asdict
import argparse
import cProfile
import hashlib
import json
import platform
import pstats
import time
import numpy as np
from resectionlab.core import array_digest, semantic_digest
from resectionlab.surgical_scene import SurgicalScene, from_case
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_rl_protection import ProtectionSet, ProtectionAnnotation, load_protection
from resectionlab.surgical_rl_env import SurgicalRLEnv, RLConfig
from resectionlab.imaging import load_case
from resectionlab.surgical_tools import TOOLS


def small_case():
    labels = np.zeros((24, 24, 24), np.uint16)
    labels[8:12, 8:12, 8:12] = 1
    scene = SurgicalScene(labels, np.eye(4), [dict(id=1, name='Engineering target', material='tumor', target=True)],
        {'provenance': 'fixed synthetic benchmark; not patient anatomy'}, {'tissuePhysicsV2': default_profile()})
    mask = np.zeros(scene.shape, bool); mask[20, 20, 20] = True
    protection = ProtectionSet(scene, [ProtectionAnnotation.from_mask(scene, 'brainstem', mask,
        provenance='Synthetic exclusion zone; not anatomy')], clearance_mm=2.)
    setup = [dict(slot='left', operation='exchange', instrument='cusa', duration=.1),
        dict(slot='left', operation='place', duration=.1, parameters={'radiusMm': .5},
             pose={'position': [9., 9., -5.], 'quaternion': [0., 0., 0., 1.]})]
    action = dict(slot='left', operation='apply', duration=.2,
        pose={'position': [9., 9., 9.], 'quaternion': [0., 0., 0., 1.]})
    return scene, protection, setup, action, {}


def acquired_case(root):
    case_path = root / 'patient.ressectionlab'; manifest = root / 'rl-protection.json'
    scene = from_case(load_case(case_path)); protection = load_protection(scene, manifest)
    actions = [row['action'] for row in json.loads((root / 'drill-verification.json').read_text())['actions']]
    return scene, protection, actions[:1], actions[1], {
        'casePath': str(case_path.resolve()), 'caseSha256': hashlib.sha256(case_path.read_bytes()).hexdigest(),
        'protectionManifestSha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}


def measure(fn, repeats):
    values = []
    for _ in range(repeats):
        started = time.perf_counter(); fn(); values.append(time.perf_counter() - started)
    return {'medianMs': float(np.median(values) * 1000), 'p95Ms': float(np.percentile(values, 95) * 1000),
        'samplesMs': [v * 1000 for v in values]}


def benchmark(case, repeats):
    scene, protection, setup, action, source = case
    config = RLConfig(max_seconds=600., max_healthy_removed_mm3=1e9, max_vessel_damage_mm2=1e9, max_bleeding_mm3=1e9)
    env = SurgicalRLEnv(scene, protection=protection, config=config)
    reset = measure(lambda: env.reset(seed=17), repeats)
    for command in setup:
        outcome = env.step_action(command)[4]['outcome']
        if outcome['status'] != 'accepted': raise AssertionError(outcome)
    base = env.engine
    results = []
    def step():
        engine = base.clone(); result = engine.step(action)
        results.append((result['status'], result['reasonCode'], engine.state_id))
    timings = {'reset': reset, 'clone': measure(base.clone, repeats), 'step': measure(step, repeats),
        'preview': measure(lambda: base.preview(action), repeats)}
    slot = base.state['slots'][action['slot']]
    timings['guardSweep'] = measure(lambda: protection.sweep_hits(TOOLS[slot['instrument']], base.parameters(slot), slot, action['pose']), repeats)
    profile = cProfile.Profile(); profile.runcall(step)
    stats = pstats.Stats(profile)
    top = sorted(stats.stats.items(), key=lambda row: row[1][3], reverse=True)[:25]
    if len(set(results)) != 1: raise AssertionError('Identical input replay changed')
    physical_reference = base.clone(); physical_reference.step(action)
    physical_id = semantic_digest({'scene': scene.identity, 'state': physical_reference.state})
    return {'source': source, 'sceneId': scene.identity, 'labelHash': array_digest(scene.labels), 'shape': list(scene.shape),
        'rlConfig': asdict(config), 'rewardConfig': asdict(env.reward_config),
        'protectedCells': len(protection.cells), 'protectionId': semantic_digest({'report': protection.report(), 'cells': protection.cells.tolist()}),
        'seed': 17, 'setupActions': setup, 'measuredAction': action, 'preStateId': base.state_id,
        'outcome': {**dict(zip(('status', 'reasonCode', 'stateId'), results[0])), 'physicalStateId': physical_id}, 'timings': timings,
        'profile': [{'file': Path(key[0]).name, 'line': key[1], 'function': key[2], 'calls': row[1],
                     'selfSeconds': row[2], 'cumulativeSeconds': row[3]} for key, row in top]}


def benchmark_rl_step(case, repeats):
    """Full wrapper transition, preparing each identical prefix outside timing."""
    scene, protection, setup, action, source = case
    config = RLConfig(max_seconds=600., max_healthy_removed_mm3=1e9, max_vessel_damage_mm2=1e9, max_bleeding_mm3=1e9)
    samples, signatures = [], []
    for _ in range(repeats):
        env = SurgicalRLEnv(scene, protection=protection, config=config); env.reset(seed=17)
        for command in setup:
            if env.step_action(command)[4]['outcome']['status'] != 'accepted':
                raise AssertionError('Fixed setup rejected')
        pre_state_id = env.engine.state_id
        started = time.perf_counter()
        observation, reward, terminated, truncated, info = env.step_action(action)
        samples.append((time.perf_counter() - started) * 1000)
        signatures.append(semantic_digest({'stateId': env.engine.state_id, 'reward': reward,
            'terms': info['reward_terms'], 'terminated': terminated, 'truncated': truncated,
            'observation': {key: value.tolist() for key, value in observation.items()}}))
    if len(set(signatures)) != 1: raise AssertionError('Full RL transition differs for identical inputs')
    return {'source': source, 'sceneId': scene.identity, 'labelHash': array_digest(scene.labels),
        'rlConfig': asdict(config), 'rewardConfig': asdict(env.reward_config),
        'shape': list(scene.shape), 'protectedCells': len(protection.cells),
        'protectionId': semantic_digest({'report': protection.report(), 'cells': protection.cells.tolist()}),
        'seed': 17, 'setupActions': setup, 'measuredAction': action, 'preStateId': pre_state_id,
        'outcome': {'status': info['outcome']['status'], 'reasonCode': info['outcome']['reasonCode'],
            'stateId': env.engine.state_id, 'reward': reward, 'terminated': terminated, 'truncated': truncated,
            'transitionSignature': signatures[0], 'identicalAcrossRepeats': True},
        'timings': {'rlStepAction': {'medianMs': float(np.median(samples)),
            'p95Ms': float(np.percentile(samples, 95)), 'samplesMs': samples}}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--patient-root', type=Path, default=Path('outputs/surgical-patient-v2'))
    parser.add_argument('--small-only', action='store_true')
    parser.add_argument('--rl-step-only', action='store_true', help='Measure full RL step_action including reward and observation')
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100: parser.error('repeats must be 1..100')
    report = {'schema': 'resectionlab.surgical-action-benchmark.v1', 'repeats': args.repeats,
        'python': platform.python_version(), 'platform': platform.platform(),
        'definitions': {'reset': 'Gym reset including observations', 'step': 'clone fixed base then engine step',
            'preview': 'engine preview including view', 'guardSweep': 'full measured action swept geometry only; excludes indirect postchecks',
            'clone': 'clone fixed base with setup history',
            'rlStepAction': 'Full RL step_action, including validation, engine, reward, observation, info; identical reset/setup excluded'}, 'cases': {}}
    for name, case in [('small', small_case()), *([] if args.small_only else [('sub-06', acquired_case(args.patient_root))])]:
        print('Benchmarking ' + name, flush=True)
        report['cases'][name] = (benchmark_rl_step if args.rl_step_only else benchmark)(case, args.repeats)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(str(output), flush=True)
