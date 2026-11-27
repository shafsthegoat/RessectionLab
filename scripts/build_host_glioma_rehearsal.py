#!/usr/bin/env python3
"""Build an operative rehearsal scene on a prepared, segmented head study with a tumour layer.

The host is any prepared study whose layers declare scalp or skin, skull,
cerebral cortex, white matter and a tumour (acquired, estimated, supplied or
generated; a missing layer is named in the error). The output is a new case on
a working grid with the operative site uppermost, plus the landmarks the
rehearsal driver uses. Source arrays are never modified.

  build_host_glioma_rehearsal.py HOST.ressectionlab DIRECTORY [--outward R A S] [--bone-radius-mm 22]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from resectionlab.imaging import load_case, read_case_artifacts, save_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_host_rehearsal import build_host_rehearsal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', type=Path, help='Prepared segmented host study')
    parser.add_argument('directory', type=Path, help='Directory for host-rehearsal.ressectionlab and landmarks.json')
    parser.add_argument('--outward', type=float, nargs=3, metavar=('R', 'A', 'S'),
                        help='Outward RAS direction of the approach; default is the shortest convexity corridor')
    parser.add_argument('--bone-radius-mm', type=float, default=22.)
    args = parser.parse_args()
    case, scene, landmarks = build_host_rehearsal(load_case(args.host), outward=args.outward,
                                                  bone_radius_mm=args.bone_radius_mm)
    args.directory.mkdir(parents=True, exist_ok=True)
    destination = args.directory / 'host-rehearsal.ressectionlab'
    engine = SurgicalEngine(scene)
    record = {'mapping': scene.regions, 'branches': {'main': engine.snapshot()}, 'branchId': 'main',
              'position': 0, 'parents': {'main': None}, 'timeline': {'main': []}}
    if destination.exists():
        if (load_case(destination).semantic_hash != case.semantic_hash
                or read_case_artifacts(destination).get('surgical_toolbox') != record):
            raise ValueError('Existing case differs. Select a new output directory.')
    else:
        save_case(case, destination, artifacts={'surgical_toolbox': record})
    receipt = args.directory / 'landmarks.json'
    if receipt.exists():
        if json.loads(receipt.read_text()) != landmarks:
            raise ValueError('Existing landmark receipt differs. Select a new output directory.')
    else:
        with receipt.open('x') as out:
            json.dump(landmarks, out, separators=(',', ':'), allow_nan=False)
            out.write('\n')
    print(json.dumps({'case': str(destination), 'caseHash': case.semantic_hash, 'sceneId': scene.identity,
                      'entryRasMm': landmarks['entryCenterRasMm'], 'tumourMm3': landmarks['tumourMm3']}))


if __name__ == '__main__':
    main()
