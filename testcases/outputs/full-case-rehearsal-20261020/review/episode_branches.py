"""Select a genuine saved common prefix, or combine independently run branches.

This utility never constructs or changes patient engine state. Every included
snapshot retains its original public-action log and must pass native admission.
"""
import argparse
import copy
from pathlib import Path

from resectionlab.workspace_bundle import load_workspace, save_workspace
from resectionlab.surgical_environment import load_environment


def save(case, artifacts, workspace, output, source):
    if output.resolve() == source.resolve():
        raise ValueError('Use a distinct derivative output')
    output.parent.mkdir(parents=True, exist_ok=True)
    save_workspace(case, output, sources=workspace['sources'], imaging=workspace['imagingState'],
                   episode=workspace['episode'], selection=workspace['selection'],
                   originals=workspace['originals'], artifacts=artifacts)


def vascular_input(primary, output):
    case, artifacts, workspace = load_workspace(primary)
    record = artifacts['surgical_toolbox']
    prefix = record['branches']['exposure-prefix']
    active = record['branches'][record['branchId']]
    count = len(prefix['actions'])
    assert active['actions'][:count] == prefix['actions']
    assert active['sceneId'] == prefix['sceneId']
    assert active.get('protectionContract') == prefix.get('protectionContract')
    # A selected saved prefix is a native checkpoint, never synthesized tissue.
    record['branches'] = {'exposure-prefix': copy.deepcopy(prefix)}
    record['parents'] = {'exposure-prefix': None}
    record['fullCase'] = {}
    record.pop('timeline', None)
    record.update(branchId='exposure-prefix', position=count)
    artifacts['full_case_branch_provenance'] = {
        'sourceEpisode': str(primary.resolve()), 'savedPrefixBranch': 'exposure-prefix',
        'savedPrefixStateId': prefix['stateId'], 'savedPrefixPosition': count,
    }
    save(case, artifacts, workspace, output, primary)
    print(output, 'prefix actions', count, 'state', prefix['stateId'])


def assemble(primary, vascular, output):
    case, artifacts, workspace = load_workspace(primary)
    other_case, other_artifacts, _ = load_workspace(vascular)
    assert case.semantic_hash == other_case.semantic_hash
    assert load_environment(primary).identity == load_environment(vascular).identity
    record = artifacts['surgical_toolbox']
    other = other_artifacts['surgical_toolbox']
    branch_id = other['branchId']
    snapshot = other['branches'][branch_id]
    parent = other['parents'][branch_id]
    prefix = record['branches'][parent['branchId']]
    assert parent['position'] == len(prefix['actions'])
    assert other['branches'][parent['branchId']] == prefix
    assert snapshot['actions'][:parent['position']] == prefix['actions']
    assert snapshot['sceneId'] == prefix['sceneId']
    assert snapshot.get('protectionContract') == prefix.get('protectionContract')
    if branch_id in record['branches']:
        assert record['branches'][branch_id] == snapshot
    record['branches'][branch_id] = copy.deepcopy(snapshot)
    record.setdefault('parents', {})[branch_id] = copy.deepcopy(parent)
    record.setdefault('fullCase', {})[branch_id] = copy.deepcopy(other['fullCase'][branch_id])
    record.pop('timeline', None)
    artifacts['full_case_branch_provenance'] = {
        'primaryEpisode': str(primary.resolve()), 'vascularEpisode': str(vascular.resolve()),
        'commonPrefixBranch': parent['branchId'], 'commonPrefixPosition': parent['position'],
        'commonPrefixStateId': prefix['stateId'],
    }
    save(case, artifacts, workspace, output, primary)
    _, restored, _ = load_workspace(output)
    assert restored['surgical_toolbox'] == record
    print(output, 'branches', {name: len(value['actions']) for name,value in record['branches'].items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['vascular-input', 'assemble'])
    parser.add_argument('primary', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--vascular', type=Path)
    args = parser.parse_args()
    if args.mode == 'vascular-input':
        vascular_input(args.primary, args.output)
    else:
        if args.vascular is None: parser.error('assemble requires --vascular')
        assemble(args.primary, args.vascular, args.output)
