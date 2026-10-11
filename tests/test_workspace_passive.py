"""Generated paired passive workspace controls; no patient or model inputs."""
from __future__ import annotations

from hashlib import sha256
import json
from zipfile import ZipFile, ZIP_STORED

import pytest

from resectionlab.desktop_bridge import BridgeRuntime
from resectionlab.workspace_bundle import load_workspace, session_digest


class Client:
    def __init__(self, root):
        self.events = []
        self.runtime = BridgeRuntime(root, self.events.append)
        self.serial = 0

    def call(self, operation, args=None, *, error=False):
        self.serial += 1
        identity = str(self.serial)
        self.runtime.submit({'id': identity, 'op': operation, 'args': args or {}})
        assert self.runtime.wait_idle(20)
        result = [event for event in self.events if event['id'] == identity][-1]
        assert result['event'] == ('error' if error else 'result'), result
        return result if error else result['result']

    def close(self):
        self.runtime.close()


def _rewrite(path, change):
    with ZipFile(path) as archive:
        members = {item.filename: archive.read(item.filename) for item in archive.infolist()}
    document = json.loads(members['workspace.json'])
    change(document)
    document['sessionHash'] = session_digest(document)
    members['workspace.json'] = json.dumps(document, sort_keys=True, separators=(',', ':')).encode()
    manifest = json.loads(members['manifest.json'])
    manifest['workspace_sha256'] = sha256(members['workspace.json']).hexdigest()
    members['manifest.json'] = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    with ZipFile(path, 'w', compression=ZIP_STORED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)


@pytest.fixture
def saved(tmp_path):
    first = Client(tmp_path / 'first')
    try:
        result = first.call('executePersistentPassiveEpisode',
                            {'fixture': 'generated-persistent-passive-v1', 'paired': True})
        path = tmp_path / 'passive.ressectionlab'
        first.call('saveCase', {'caseHash': result['case']['caseHash'], 'path': str(path),
                               'workspace': {'episodeReplay': {
                                   'episodeId': result['episode']['episodeId'],
                                   'frameIndex': 4, 'visible': True}}})
        return {'path': path, 'result': result}
    finally:
        first.close()


def test_paired_save_close_reopen_retains_both_access_poses_and_generated_clock(saved, tmp_path):
    second = Client(tmp_path / 'second')
    try:
        loaded = second.call('loadCase', {'path': str(saved['path'])})
        replay = loaded['workspaceSession']['episodeReplay']
        assert replay['validation'] == 'authoritative_generated_passive_replay'
        assert replay['accountingQualification'] == 'declared_generated_schedule_not_measured'
        assert replay['sourceBinding'] == saved['result']['sourceBinding']
        assert replay['episode'] == saved['result']['episode']
        assert replay['frameIndex'] == 4 and replay['visible'] is True
        assert len(replay['episode']['frames']) == 6
        assert all(not tool['physical_absence_claim'] for frame in replay['episode']['frames']
                   for tool in (frame['operative']['moving_tool'], frame['operative']['other_tool']))
        assert replay['episode']['frames'][4]['operative']['other_tool']['retained_at_depth'] is False
        assert replay['episode']['frames'][4]['operative']['other_tool']['pose_status'] == 'access_pose_recorded'
        assert loaded['metadata']['patient_admission'] is False
        assert load_workspace(saved['path'])[2]['episode'] == {
            key: saved['result'][key] for key in ('episode', 'episodeCanonicalJson', 'sourceBinding')}
    finally:
        second.close()


@pytest.mark.parametrize('mutate', [
    lambda ep, binding: ep['frames'][4]['operative']['other_tool'].update(
        {'physical_absence_claim': True}),
    lambda ep, binding: ep['frames'][3]['operative'].update(
        {'elapsed_generated_seconds': 123.}),
    lambda ep, binding: binding['tools'][0].update({'shaft_radius_mm': 9.}),
    lambda ep, binding: binding.update({'sourceHash': 'sha256:' + '0' * 64}),
])
def test_resigned_passive_motion_clock_tool_and_source_tamper_refuse(saved, tmp_path, mutate):
    _rewrite(saved['path'], lambda document: mutate(
        document['episodeReplay']['envelope']['episode'],
        document['episodeReplay']['envelope']['sourceBinding']))
    client = Client(tmp_path / 'guard')
    try:
        result = client.call('loadCase', {'path': str(saved['path'])}, error=True)
        assert result['error']['code']
        assert not client.runtime.session.cases
    finally:
        client.close()
