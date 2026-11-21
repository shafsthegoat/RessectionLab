"""Build a portable guided rehearsal and, by default, its hidden desktop walkthrough."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build(output, episode_only):
    # Keep the checkout's testcases/outputs path when it points at an
    # external evidence volume. The public authoring tool uses that path.
    output = output.absolute()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory for this rehearsal.')
    output.mkdir(parents=True, exist_ok=True)
    source = output/'source'
    episode = output/'episode'
    complete = episode/'complete.ressectionlab'
    history = complete.with_suffix('.json')

    def run(*command):
        subprocess.run([str(arg) for arg in command], cwd=ROOT, check=True)

    run(sys.executable, 'scripts/generated_glioma_walkthrough_case.py', source)
    run(sys.executable, 'scripts/rehearse_glioma_walkthrough.py', 'run', source/'generated-glioma.ressectionlab',
        '--landmarks', source/'landmarks.json', '--output', episode)
    run(sys.executable, 'scripts/rehearse_glioma_walkthrough.py', 'audit', complete, '--require-complete', '--guided-start',
        episode/'editable-start.ressectionlab', output/'guided-start.ressectionlab')
    if episode_only:
        return
    shot_plan = output/'shot-plan.json'
    record = output/'recording'
    run('pnpm', '--dir', 'desktop', 'build')
    run(sys.executable, 'scripts/plan_glioma_walkthrough.py', history, shot_plan, '--seconds', '96')
    run(ROOT/'desktop/node_modules/.bin/electron', 'desktop/electron/record-surgical-walkthrough.cjs', complete, record, shot_plan)
    run(sys.executable, 'scripts/encode_surgical_walkthrough.py', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--episode-only', action='store_true', help='Build and verify the episode and guided practice without recording')
    args = parser.parse_args()
    build(args.output, args.episode_only)
