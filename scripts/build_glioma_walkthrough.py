"""Build a portable guided rehearsal and, by default, its hidden desktop walkthrough."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RECORDED = ROOT/'artifacts/generated-convexity-rehearsal-v1'


def build(output, episode_only, recorded_episode=None, shot_plan=None):
    # Keep the checkout's testcases/outputs path when it points at an
    # external evidence volume. The public authoring tool uses that path.
    output = output.absolute()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory for this rehearsal.')
    output.mkdir(parents=True, exist_ok=True)
    source = output/'source'
    episode = output/'episode'

    def run(*command):
        subprocess.run([str(arg) for arg in command], cwd=ROOT, check=True)

    if recorded_episode is None:
        complete = episode/'complete.ressectionlab'
        run(sys.executable, 'scripts/generated_glioma_walkthrough_case.py', source)
        run(sys.executable, 'scripts/rehearse_glioma_walkthrough.py', 'run', source/'generated-glioma.ressectionlab',
            '--landmarks', source/'landmarks.json', '--output', episode, '--recorded-inputs', RECORDED/'inputs.json.gz')
        run(sys.executable, 'scripts/rehearse_glioma_walkthrough.py', 'audit', complete, '--require-complete', '--guided-start',
            episode/'editable-start.ressectionlab', output/'guided-start.ressectionlab')
        if episode_only:
            return
    else:
        complete = recorded_episode.resolve(strict=True)
    plan = output/'shot-plan.json'
    if shot_plan is not None:
        shutil.copyfile(shot_plan, plan)
    elif recorded_episode is not None:
        run(sys.executable, 'scripts/plan_glioma_walkthrough.py', complete.with_suffix('.json'), plan, '--seconds', '96')
    else:
        # New independently created practice branches have their own UUID and
        # portable file hash. Rebind optics only after matching every state.
        optics = json.loads((RECORDED/'shot-plan.json').read_text())
        history = json.loads(complete.with_suffix('.json').read_text())
        if (optics['caseHash'] != history['caseHash'] or
                [a['stateAfter'] for a in history['actions']] != [a['stateAfter'] for a in optics['actions']]):
            raise ValueError('The saved optical plan requires the exact recorded physical sequence')
        optics.update(branchId=history['branchId'], sourceFileSha256=history['sha256'])
        plan.write_text(json.dumps(optics, indent=2) + '\n')
    record = output/'recording'
    run('pnpm', '--dir', 'desktop', 'build')
    run(ROOT/'desktop/node_modules/.bin/electron', 'desktop/electron/record-surgical-walkthrough.cjs', complete, record, plan)
    run(sys.executable, 'scripts/encode_surgical_walkthrough.py', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--episode-only', action='store_true', help='Build and verify the episode and guided practice without recording')
    mode.add_argument('--record-episode', type=Path, help='Re-export an existing portable episode with its recorded actions')
    parser.add_argument('--shot-plan', type=Path, help='Use the saved camera plan for that exact episode')
    args = parser.parse_args()
    if args.shot_plan and not args.record_episode:
        parser.error('--shot-plan requires --record-episode')
    build(args.output, args.episode_only, args.record_episode, args.shot_plan)
