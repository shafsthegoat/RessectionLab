#!/usr/bin/env python3
"""Read-only reconstruction of Q updates, task outcomes and source identities."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def verify(folder):
    def read(name): return json.loads((folder / name).read_text())
    protocol, result = read('protocol.json'), read('result.json')
    root = Path(__file__).resolve().parents[1]
    for path, expected in protocol['code_hashes'].items():
        if path == 'scripts/train_coordinated_burring.py' and (folder / 'finalization-repair.json').exists():
            repair = read('finalization-repair.json')
            assert repair['original_runner_sha256'] == expected and repair['extra_optimizer_updates'] == 0
            expected = repair['repaired_runner_sha256']
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected, path
    assert hashlib.sha256(Path(protocol['source']).read_bytes()).hexdigest() == protocol['source_sha256']
    config, settings = protocol['config'], protocol['settings']
    curves = read('learning-curve.json')
    counts = []
    for seed in settings['seeds']:
        table, updates = {}, 0
        for episode in [row for row in curves if row['seed'] == seed]:
            observation = (0., 0., config['horizon'])
            assert abs(sum(d['reward'] for d in episode['trajectory']) - episode['return_']) < 1e-12
            for decision in episode['trajectory']:
                assert abs(sum(decision['terms'].values()) - decision['reward']) < 1e-12
                after = decision['observation']
                following = (round(after['bone_mm3'], 6), round(after['debris_mm3'], 6), after['remaining'])
                action = protocol['actions'].index(decision['action'])
                values = table.setdefault(observation, np.zeros(3))
                bootstrap = 0. if decision['reason'] is not None else settings['gamma'] * max(table.get(following, np.zeros(3)))
                values[action] += settings['alpha'] * (decision['reward'] + bootstrap - values[action])
                observation = following
                updates += 1
            assert updates == episode['updates']
        checkpoint = read(f'q-seed-{seed}.json')
        assert updates == checkpoint['updates'] and any(np.any(v) for v in table.values())
        assert len(checkpoint['table']) == len(table)
        for row in checkpoint['table']:
            np.testing.assert_array_equal(table[tuple(row['state'])], row['values'])
        assert hashlib.sha256((folder / f'q-seed-{seed}.json').read_bytes()).hexdigest() == result['checkpoints'][str(seed)]
        counts.append(updates)
    assert result['baselines']['scripted']['success']
    for name in ('drill_only', 'suction_only', 'reversed_once', 'stop'):
        assert not result['baselines'][name]['success']
    learned = [row for row in result['evaluation'] if row['method'] == 'Q_learning']
    for row in learned:
        assert row['success'] and row['actions'] == ['BURR', 'CLEAR']
        assert config['minimum_bone_mm3'] <= row['measurements']['bone_removed_mm3'] <= config['maximum_bone_mm3']
        assert row['debris_mm3'] <= config['maximum_debris_mm3']
        assert row['aspirated_mm3'] == row['measurements']['bone_removed_mm3']
        for key in ('healthy_removed_mm3', 'tumor_removed_mm3', 'vessel_damage_mm2', 'bleeding_generated_mm3'):
            assert row['measurements'][key] == 0
        assert row['guard']['configured'] and row['guard']['mode'] == 'strict-research'
        replay = read(f"replay-seed-{row['seed']}.json")
        assert replay['stateId'] == row['state_id'] and replay['protectionContract']
    assert result['source_arrays_unchanged']
    return dict(status='verified', independently_reconstructed_q_updates=counts,
                learned_successes=len(learned), code_and_source_hashes_match=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.folder.resolve()), indent=2))
