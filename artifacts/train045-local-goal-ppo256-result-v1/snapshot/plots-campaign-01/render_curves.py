"""Render measured dev-only PPO endpoints and executed training episodes."""
from pathlib import Path
import gzip
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
os.environ['MPLCONFIGDIR'] = str(STAGE / 'plots-pilot-01/mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), {'path': str(path.relative_to(STAGE)),
                             'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    pilot, pilot_ref = read(STAGE / 'pilot-01/report.json')
    campaign, campaign_ref = read(STAGE / 'campaign-256-01/report.json')
    search, search_ref = read(STAGE / 'search-01/report.json')
    if campaign['status'] != 'completed':
        raise ValueError('Both terminal training arms required')
    records = {'scope': 'One exposed TRAIN045 local simulated goal; no held-out or clinical claim',
               'source_reports': [pilot_ref, campaign_ref, search_ref], 'arms': {}}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {'frozen_encoder-seed0': '#187a8d', 'trainable_encoder-seed0': '#b45a35'}
    for name, color in colors.items():
        old = pilot['arms'][name]
        arm = campaign['arms'][name]
        previous = old['checkpoints'][0]
        end = arm['checkpoints'][-1]
        evaluations = [old['initial_evaluation'], previous['evaluation'], end['evaluation']]
        xs = [0, 64, 256]
        ys = [e['return'] for e in evaluations]
        label = name.replace('_encoder-seed0', '').capitalize() + ' encoder'
        for axis in axes[0]:
            axis.plot(xs, ys, 'o-', label=label, color=color)
        episode_rows = []
        for run, offset in [('pilot-01', 0), ('campaign-256-01', 64)]:
            episodes = [json.loads(gzip.decompress(f.read_bytes()))
                        for f in (STAGE / run / name).glob('episode*.json.gz')]
            consumed = offset
            for episode in sorted(episodes, key=lambda e: e['episode']):
                # Automatic success STOP is a native transition, not a policy decision.
                count = episode['costs']['policy_native_transitions']
                consumed += count
                episode_rows.append({'run': run, 'episode': episode['episode'],
                    'end_training_decision': consumed, 'decisions': count,
                    'start_absolute_step': episode['start_absolute_step'],
                    'return': episode['return'], 'success': episode['success']})
        axes[1, 0].plot([e['end_training_decision'] for e in episode_rows],
                        [e['return'] for e in episode_rows], 'o-', markersize=3,
                        alpha=.8, label=label, color=color)
        rollouts = end['rollouts']
        steps = [r['optimizer_completed_before_update'] for r in rollouts[1:]]
        steps.append(end['optimizer_completed'])
        axes[1, 1].plot([0] + [r['decisions'] for r in rollouts], [0] + steps,
                        'o-', color=color, label=label)
        records['arms'][name] = {'decisions': xs, 'evaluation_return': ys,
            'evaluation_success': [e['success'] for e in evaluations],
            'optimizer_steps_after_rollout': steps,
            'completed_episodes': episode_rows, 'checkpoint': end['checkpoint']}
    search_return = search['evaluation']['return']
    records['search_return'] = search_return
    axes[0, 0].axhline(search_return, color='#39434b', linestyle='--', label='Successful search')
    solved = any(a['evaluation_success'][-1] for a in records['arms'].values())
    axes[0, 0].set(title='Full-task endpoint compared with successful search', ylabel='Frozen task return')
    axes[0, 1].set(title='PPO endpoint detail' + ('' if solved else ' (penalty reduction only)'),
                   ylabel='Full-task return')
    axes[1, 0].set(title='Training episodes (mixed starts 12–16)', ylabel='Episode return')
    axes[1, 1].set(title='Actual Adam steps (not epochs)', ylabel='Cumulative optimizer steps')
    for axis in axes.flat:
        axis.set_xlabel('Training decisions per arm')
        axis.grid(alpha=.18)
        axis.legend(fontsize=8)
    fig.suptitle('TRAIN045 simulated local goal • fixed task • seed 0', fontsize=15)
    fig.savefig(HERE / 'learning-curves.png', dpi=140)
    fig.savefig(HERE / 'learning-curves.svg')
    svg = HERE / 'learning-curves.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    (HERE / 'learning-curves.json').write_text(json.dumps(records, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
