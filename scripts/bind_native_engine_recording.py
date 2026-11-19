#!/usr/bin/env python3
"""Bind an existing engine recording to a measured native approach and save a new replay-only archive."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from resectionlab.data_root import sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection
from resectionlab.surgical_recording_contract import bind_recording
from resectionlab.surgical_study_replay import verify_recording
from resectionlab.workspace_bundle import load_workspace, save_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True, help='Directory or patient-anatomy.json')
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--approach', required=True, help='Saved assessed route ID')
    parser.add_argument('--target', required=True, help='Selected native target layer ID')
    parser.add_argument('--kind', required=True, choices=('scripted', 'search', 'manual', 'rl'))
    parser.add_argument('--name', required=True, help='Display name of the action producer')
    parser.add_argument('--generator-source', type=Path, required=True, help='Executed script, search implementation, or manual input log')
    parser.add_argument('--checkpoint', type=Path, help='Weights used for an RL recording')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.recording.resolve():
        parser.error('Select an unused path for the bound recording')
    if (args.kind == 'rl') != bool(args.checkpoint):
        parser.error('Supply a checkpoint for an RL producer')
    study = PatientImagingCollection(args.study.parent if args.study.is_file() else args.study)
    case, artifacts, workspace = load_workspace(args.recording)
    record = artifacts['surgical_toolbox']
    if 'exactGridReplay' in record or 'nativeStudyReplay' in artifacts:
        parser.error('Select the original engine recording before its native study binding')
    generator = {'kind': args.kind, 'name': args.name, 'sourceSha256': sha256_file(args.generator_source)}
    if args.checkpoint:
        generator['checkpointSha256'] = sha256_file(args.checkpoint)
    record['exactGridReplay'] = bind_recording(study, case, record, approach_id=args.approach,
                                                target_id=args.target, generator=generator)
    receipt, _ = verify_recording(study, args.target, case, record, approach_id=args.approach)
    artifacts['nativeStudyReplay'] = receipt
    workspace = workspace or {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as scratch:
        verified = Path(scratch)/'verified.ressectionlab'
        save_workspace(case, verified, sources=workspace.get('sources', ()), originals=workspace.get('originals', ()),
            imaging=workspace.get('imagingState'), episode=workspace.get('episode'), selection=workspace.get('selection'),
            artifacts=artifacts)
        # Reserve the final name exclusively before the same-filesystem rename.
        # This works on exFAT source/evidence drives as well as APFS.
        with args.output.open('xb'):
            pass
        try:
            verified.replace(args.output)
        except OSError:
            args.output.unlink()
            raise
    print(receipt['contractHash'])


if __name__ == '__main__':
    main()
