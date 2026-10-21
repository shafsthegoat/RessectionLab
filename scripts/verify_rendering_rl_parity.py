#!/usr/bin/env python3
"""Fingerprint fixed RL episodes before/after display work without saving cases.
Run --output baseline.json, then --compare baseline.json --output current.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from resectionlab.surgical_rl_env import SurgicalRLEnv, RLConfig
from resectionlab.surgical_environment import load_environment


def fingerprint(value):
    if isinstance(value, np.ndarray):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, dict): return {str(k): fingerprint(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [fingerprint(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    return value


def verify(paths):
    result = {}
    for path in paths:
        scene = load_environment(path)
        # This renderer-invariance check does not grant training admission.
        try:
            env = SurgicalRLEnv(scene, config=RLConfig(require_critical_coverage=False, max_steps=16))
        except ValueError as exc:
            result[str(Path(path).resolve())] = {'labels': fingerprint(scene.labels), 'scene': scene.identity, 'admissionRejection': str(exc)}
            print('Retained RL admission rejection '+str(path), flush=True)
            continue
        obs, info = env.reset(seed=17)
        rows = [{'observation': fingerprint(obs), 'state': env.engine.state_id}]
        actions = [{'slot': 'left', 'operation': 'inspect', 'duration': .1},
                   {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .01},
                   {'slot': 'left', 'operation': 'place', 'pose': {'position': [0., 0., 0.], 'quaternion': [0., 0., 0., 0.]}, 'duration': .1},
                   {'slot': 'right', 'operation': 'exchange', 'instrument': 'bipolar', 'duration': .01}]
        for action in actions:
            obs, reward, terminated, truncated, info = env.step_action(action)
            rows.append({'action': action, 'observation': fingerprint(obs), 'reward': reward,
                         'terminated': terminated, 'truncated': truncated, 'info': fingerprint(info),
                         'state': env.engine.state_id, 'removed': sorted(env.engine.state['removed'])})
            if terminated or truncated: break
        result[str(Path(path).resolve())] = {'labels': fingerprint(scene.labels), 'scene': scene.identity, 'steps': rows}
        print('Fingerprinted '+str(path), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('cases', nargs='+'); parser.add_argument('--output', required=True); parser.add_argument('--compare')
    args = parser.parse_args(); result = verify(args.cases)
    if args.compare:
        baseline = json.loads(Path(args.compare).read_text())
        if baseline != result: raise AssertionError('Display work changed a fixed RL episode')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()
