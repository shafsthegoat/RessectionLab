#!/usr/bin/env python3
"""Verify RL transitions on intact V2 patient anatomy; no training or state injection."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import time
import resource
import sys
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation
from resectionlab.core import array_digest
from resectionlab.imaging import load_case, save_case
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_rl_env import SurgicalRLEnv, RLConfig, RewardConfig
from resectionlab.surgical_rl_protection import GuardConfig, load_protection


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def recipe(path):
    return [copy.deepcopy(row['action']) for row in json.loads(Path(path).read_text())['actions']]


def verify_default_vessel_boundary(scene, protection, actions):
    """A deliberately injurious public recipe must fail the default safety task."""
    env = SurgicalRLEnv(scene, protection=protection)
    env.reset(seed=17)
    rows = []
    for action in actions:
        _, reward, terminated, truncated, info = env.step_action(action)
        rows.append({'action': action, 'reward': reward, 'terminated': terminated,
                     'truncated': truncated, 'info': info})
        if terminated or truncated:
            if not terminated or truncated or not str(info['termination_reason']).startswith('damage_budget:'):
                raise AssertionError('Default vessel injury did not terminate on a damage budget')
            if reward >= 0: raise AssertionError('Default vessel injury received a nonnegative reward')
            return {'status': 'passed', 'seed': 17, 'actions': rows,
                    'terminationReason': info['termination_reason'], 'reward': reward,
                    'preRemovedCells': 0, 'defaultConfig': True}
    raise AssertionError('Deliberate vessel injury did not reach the default damage boundary')


def verify(case_path='outputs/surgical-patient-v2/patient.ressectionlab',
           output='outputs/surgical-rl-patient',
           protection_path='outputs/surgical-patient-v2/rl-protection.json',
           reference_directory='outputs/surgical-patient-v2',
           original_reference_directory='outputs/surgical-patient-sub06'):
    started = time.perf_counter()
    case_path = Path(case_path).resolve(); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    source_hash = file_hash(case_path); case = load_case(case_path); scene = from_case(case)
    labels_hash = array_digest(scene.labels); intensity_hash = array_digest(case.mri)
    if 'tissuePhysicsV2' not in scene.properties: raise AssertionError('Explicit V2 patient required')
    protection = load_protection(scene, protection_path, strict=False)
    references = Path(reference_directory); original = Path(original_reference_directory)
    # Deliberately damaging fixtures need broad budgets to complete their checks.
    config = RLConfig(max_steps=128, max_seconds=600., max_action_seconds=5.,
        max_healthy_removed_mm3=1e9, max_vessel_damage_mm2=1e9, max_bleeding_mm3=1e9)
    sequences = {'drill': recipe(references / 'drill-verification.json'),
        'vessel': recipe(references / 'vessel-verification.json'),
        'brain': recipe(references / 'brain-verification.json')}
    # This unchanged patient requires eleven five-second work-limited strokes
    # along the saved drill path before CUSA can reach the target.
    # Retraction's old free-space placement was inside the image box. Start
    # outside it so the approach is checked rather than teleported.
    place = sequences['brain'][1]
    axis = Rotation.from_quat(place['pose']['quaternion']).apply([0., 0., 1.])
    point = np.asarray(place['pose']['position'])
    for _ in range(500):
        index = scene.inverse[:3, :3] @ point + scene.inverse[:3, 3]
        if np.any(index < -5) or np.any(index > np.asarray(scene.shape) + 4): break
        point = point - axis
    place['pose']['position'] = point.tolist()
    sequences['cusa'] = [sequences['drill'][0]] + [copy.deepcopy(sequences['drill'][1]) for _ in range(11)] + recipe(original / 'verification.json')[2:]
    episodes = {}
    for name, actions in sequences.items():
        print('Verifying RL patient ' + name, flush=True)
        env = SurgicalRLEnv(scene, protection=protection, config=config)
        reset_started = time.perf_counter()
        obs, initial_info = env.reset(seed=17)
        reset_seconds = time.perf_counter() - reset_started
        reference = SurgicalEngine(scene, protection=protection, guard_config=GuardConfig(
            require_critical_coverage=config.require_critical_coverage,
            critical_max_displacement_mm=config.critical_max_displacement_mm))
        if env.engine.state['removed']: raise AssertionError('Episode did not begin intact')
        if not env.observation_space.contains(obs): raise AssertionError('Reset observation outside declared space')
        rows = []
        for action in actions:
            result = reference.step(action)
            action_started = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step_action(action)
            action_seconds = time.perf_counter() - action_started
            blocked = name == 'brain' and str(info['outcome'].get('reason', '')).startswith('PROTECTED_STRUCTURE')
            if env.engine.state_id != reference.state_id:
                raise AssertionError(name + ' wrapper/engine transition differs: ' + str(info))
            if result['status'] == 'rejected': raise AssertionError('Known patient action rejected: ' + str(result))
            if not env.observation_space.contains(obs) or not np.isfinite(reward):
                raise AssertionError('Invalid observation or reward')
            rows.append({'action': action, 'reward': reward, 'terminated': terminated, 'truncated': truncated,
                         'wallSeconds': action_seconds, 'engineStatus': result['status'], 'stateId': env.engine.state_id, 'info': info})
            if blocked:
                if set(env.engine.state['removed']) & set(map(int, protection.cells)):
                    raise AssertionError('Protected source tissue removed')
                break
            if terminated or truncated: raise AssertionError('Verification episode ended before checks completed')
        metrics = env.engine.metrics()
        if name == 'brain' and not blocked: raise AssertionError('Protected brain approach was not blocked')
        if name == 'drill' and metrics.get('drillRemovedVolumeMm3', 0) <= 0:
            raise AssertionError('No work-limited drilling demonstrated')
        if name == 'cusa' and metrics['targetRemovedMm3'] <= 0:
            raise AssertionError('No public CUSA target removal demonstrated')
        if name == 'cusa' and not any(row['info']['reward_terms'].get('tumor_removal', 0) > 0 for row in rows):
            raise AssertionError('Target removal did not earn explicit tumor credit')
        if name == 'vessel' and metrics.get('vesselPhysicsV2', {}).get('generatedMm3', 0) <= 0:
            raise AssertionError('No injury-triggered bleeding demonstrated')
        snapshot = json.loads(json.dumps(env.snapshot(), allow_nan=False))
        restored = SurgicalRLEnv(scene, protection=protection, config=config); restored.restore(snapshot)
        if restored.engine.state_id != env.engine.state_id: raise AssertionError('Snapshot replay differs')
        # Comparing the next transition checks wrapper reward/budget history too.
        probe = {'slot': 'right', 'operation': 'inspect', 'duration': .1}
        expected = env.step_action(probe); actual = restored.step_action(probe)
        if expected[1:4] != actual[1:4] or restored.engine.state_id != env.engine.state_id:
            raise AssertionError('Post-restore reward or transition differs')
        snapshot_path = output / (name + '-rl-snapshot.json')
        snapshot_path.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + '\n')
        fresh_verified = False
        if name == 'brain':
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--reopen', str(snapshot_path),
                '--case', str(case_path), '--protection', str(Path(protection_path).resolve())],
                check=True, capture_output=True, text=True)
            fresh = json.loads(child.stdout)
            if fresh['stateId'] != snapshot['stateId'] or fresh['totalReward'] != snapshot['totalReward']:
                raise AssertionError('Fresh-process guarded RL replay differs')
            fresh_verified = True
        # Native snapshots carry the complete guard, including partial stops.
        native_engine = SurgicalEngine.restore(scene, restored.engine.snapshot())
        if native_engine.guard_report() != restored.engine.guard_report():
            raise AssertionError('Native engine protection contract differs')
        native = save_case(case, output / ('patient-' + name + '-rl.ressectionlab'), artifacts={'surgical_toolbox': {
            'mapping': scene.regions, 'branches': {'main': restored.engine.snapshot()},
            'branchId': 'main', 'position': len(restored.engine.actions), 'parents': {'main': None}}})
        episodes[name] = {'actions': rows, 'metrics': metrics, 'preRemovedCells': 0,
            'resetWallSeconds': reset_seconds, 'initialInfo': initial_info, 'engineTransitionsIdentical': True, 'protectedApproachBlocked': name == 'brain', 'jsonReplayVerified': True,
            'postRestoreTransitionVerified': True, 'freshProcessReplayVerified': fresh_verified, 'nativeCase': str(native) if native else None}
    if file_hash(case_path) != source_hash or array_digest(scene.labels) != labels_hash or array_digest(case.mri) != intensity_hash:
        raise AssertionError('Source patient changed')
    report = {'schema': 'resectionlab.surgical-rl-patient-verification.v1', 'status': 'passed',
        'sourceCase': str(case_path), 'sourceFileSha256': source_hash, 'sceneId': scene.identity,
        'labelArrayHash': labels_hash, 'intensityArrayHash': intensity_hash, 'sourceUnchanged': True,
        'arraysUnchanged': True, 'preRemovedCells': 0, 'seed': 17,
        'protectionManifest': str(Path(protection_path).resolve()), 'episodes': episodes,
        'defaultVesselDamageBoundary': verify_default_vessel_boundary(scene, protection, sequences['vessel']),
        'elapsedWallSeconds': time.perf_counter() - started,
        'processPeakRssBytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
        'limitations': ['Software contract verification, not clinical validation',
            'Damaging test episodes use deliberately relaxed budgets',
            'Cached segmentation estimates are unreviewed unless their source states otherwise',
            'Policy observations are partial; no policy was trained']}
    (output / 'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reopen', help='Internal fresh-process RL snapshot replay check')
    parser.add_argument('--case', default='outputs/surgical-patient-v2/patient.ressectionlab')
    parser.add_argument('--output', default='outputs/surgical-rl-patient')
    parser.add_argument('--protection', default='outputs/surgical-patient-v2/rl-protection.json')
    parser.add_argument('--reference-directory', default='outputs/surgical-patient-v2')
    parser.add_argument('--original-reference-directory', default='outputs/surgical-patient-sub06')
    args = parser.parse_args()
    if args.reopen:
        record = json.loads(Path(args.reopen).read_text())
        env = SurgicalRLEnv.from_patient(args.case, protection_path=args.protection,
            config=RLConfig(**record['config']), reward_config=RewardConfig(**record['reward_config']))
        env.restore(record)
        print(json.dumps({'stateId': env.engine.state_id, 'totalReward': env.total_reward}))
        raise SystemExit(0)
    report = verify(args.case, args.output, args.protection, args.reference_directory, args.original_reference_directory)
    print(json.dumps({key: report[key] for key in ('status', 'sourceUnchanged', 'elapsedWallSeconds')}, indent=2))
