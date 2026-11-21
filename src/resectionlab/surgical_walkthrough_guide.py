"""Editable, state-bound practice cues extracted from a saved physical episode."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import json

from .core import semantic_digest, thaw_json
from .imaging import load_case, read_case_artifacts, save_case

KEY = 'operativeWalkthroughGuideV1'
SCHEMA = 'resectionlab.operative-walkthrough-guide.v1'


def attach(start_path, completed_path, labels_path, destination, *, verified_engine=None):
    """Package the original editable start with the exact recorded input actions."""
    if Path(destination).exists():
        raise ValueError('Select a new guided-start file')
    case = load_case(start_path)
    artifacts = read_case_artifacts(start_path)
    start = artifacts['surgical_toolbox']
    final = read_case_artifacts(completed_path)['surgical_toolbox']
    branch = final['branchId']
    endpoint = final['branches'][branch]
    initial = start['branches'][start['branchId']]
    labels = json.loads(Path(labels_path).read_text())
    # Portable practice saves can omit the display timeline. Reconstruct its
    # receipts from the native input sequence and verify the saved checkpoint.
    from .surgical_engine import SurgicalEngine
    from .surgical_environment import load_environment
    if verified_engine is None:
        verified_engine = SurgicalEngine.restore(load_environment(completed_path, branch_id=branch),
                                                endpoint, use_cache=False, force_verify=True)
    elif verified_engine.snapshot() != endpoint:
        raise ValueError('The verified rehearsal must match the complete saved snapshot')
    timeline = verified_engine.history
    if (load_case(completed_path).semantic_hash != case.semantic_hash or branch != start['branchId']
            or initial['actions'] or initial['sceneId'] != endpoint['sceneId']
            or labels['caseHash'] != case.semantic_hash or labels['finalId'] != endpoint['stateId']
            or labels['branchId'] != branch or not timeline or not start.get('practiceEpisodes')):
        raise ValueError('A matching intact practice child and its completed episode are required')
    state = initial['stateId']
    steps = []
    for action, receipt, label in zip(endpoint['actions'], timeline, labels['actions'], strict=True):
        if (receipt['stateBefore'] != state or label['stateBefore'] != state or label['action'] != action
                or label['stateAfter'] != receipt['stateAfter'] or label['status'] != receipt['status']
                or receipt['status'] not in ('accepted', 'partial')):
            raise ValueError('Recorded inputs, labels and physical checkpoints differ')
        steps.append({'before':state, 'after':receipt['stateAfter'], 'action':action,
                      'label':label['label'], 'phase':label['phase']})
        state = receipt['stateAfter']
    if state != endpoint['stateId']:
        raise ValueError('The last cue must reach the recorded endpoint')
    guide = {'schema':SCHEMA, 'caseHash':case.semantic_hash, 'sceneId':initial['sceneId'],
             'branchId':branch, 'endpoint':state, 'steps':steps}
    artifacts[KEY] = {**guide, 'guideHash':semantic_digest(guide)}
    save_case(case, destination, artifacts=artifacts)


def next_cue(guide, engine, branch, case_hash):
    """Return a proposal only at its exact physical boundary. A cue grants no permission."""
    if not (engine._operative and isinstance(guide, Mapping) and guide.get('schema') == SCHEMA
            and guide.get('caseHash') == case_hash and guide.get('sceneId') == engine.scene.identity
            and guide.get('branchId') == branch):
        return None
    steps = guide.get('steps', ())
    position = len(engine.actions)
    if not isinstance(steps, (list, tuple)) or not 0 < len(steps) <= 8192:
        return None
    status = {'position':position, 'count':len(steps)}
    state = engine.state_id
    if position == len(steps) and guide.get('endpoint') == state:
        return {**status, 'status':'complete'}
    if position >= len(steps) or not isinstance(steps[position], Mapping) or steps[position].get('before') != state:
        return {**status, 'status':'modified'}
    row = steps[position]
    if (not isinstance(row.get('label'), str) or not isinstance(row.get('phase'), str)
            or not isinstance(row.get('action'), Mapping)):
        return None
    return {**status, 'status':'ready', 'label':row['label'], 'phase':row['phase'],
            'action':thaw_json(row['action'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=attach.__doc__)
    for name in ('start', 'completed', 'labels', 'destination'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    attach(args.start, args.completed, args.labels, args.destination)
