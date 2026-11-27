#!/usr/bin/env python3
"""Run a complete operative rehearsal on a host head through the desktop's public actions.

The source is a case written by build_host_glioma_rehearsal.py. Every motion is
a checked engine action submitted through the same bridge the desktop uses. The
operator works in the operating-table frame (see host_rehearsal_frame.py); the
engine receives and records patient RAS millimetres. Each named checkpoint is
saved as an editable workspace with its accepted action receipt.

  rehearse_host_glioma.py run SOURCE --landmarks LANDMARKS.json --output DIRECTORY [--through STAGE]
  rehearse_host_glioma.py run CHECKPOINT --resume --landmarks LANDMARKS.json --output NEW_DIRECTORY [--through STAGE]
  rehearse_host_glioma.py audit DIRECTORY/complete.ressectionlab [--require-complete]
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

STAGES = ('approach', 'scalp', 'bone', 'dura', 'treatment', 'closure')


def perform(live, landmarks, through):
    """Advance from the intact source, or from a named checkpoint, through the requested stage."""
    from host_rehearsal_actions import CHECKPOINTS, hands
    if live.scene.identity != landmarks['sceneId']:
        raise ValueError('Source geometry and landmarks differ')
    left, right = hands(live)
    names = [name for name, *_ in CHECKPOINTS]
    first = 0 if live.resume is None else names.index(live.resume) + 1
    for name, phase, work in CHECKPOINTS[first:]:
        if STAGES.index(phase) > STAGES.index(through):
            return
        live.phase = phase
        work(live, landmarks, left, right)
        live.save(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('source', type=Path)
    run.add_argument('--landmarks', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--through', choices=STAGES, default='closure')
    run.add_argument('--resume', action='store_true', help='Continue a named checkpoint on its own practice branch')
    verify = commands.add_parser('audit')
    verify.add_argument('package', type=Path)
    verify.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    from rehearse_glioma_walkthrough import DesktopEpisode, audit
    if args.command == 'audit':
        audit(args.package, require_complete=args.require_complete)
        return
    output = args.output.absolute()
    if '..' in output.parts or not output.is_relative_to(ROOT / 'testcases' / 'outputs'):
        parser.error('Choose a new directory under testcases/outputs')
    output.mkdir(parents=True, exist_ok=False)
    from resectionlab.desktop_bridge import BridgeSession
    from host_rehearsal_frame import TableEpisode
    landmarks = json.loads(args.landmarks.read_text())
    with TemporaryDirectory() as scratch, closing(BridgeSession(Path(scratch))) as bridge:
        live = TableEpisode(DesktopEpisode(bridge, args.source, output, landmarks, args.resume))
        try:
            perform(live, landmarks, args.through)
        except Exception:
            live.save('last-accepted')
            raise
        live.save('complete' if args.through == 'closure' else 'early-' + args.through)


if __name__ == '__main__':
    main()
