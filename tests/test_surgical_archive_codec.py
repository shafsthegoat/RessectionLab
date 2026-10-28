"""Lossless full-case artifact storage keeps native replay/source contracts."""
import base64
import copy
import gzip
from hashlib import sha256
import json
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef
from resectionlab.imaging import ImagingError, MAX_MANIFEST_BYTES, load_case, read_case_artifacts, save_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import from_case
from resectionlab import surgical_archive_codec as codec
from resectionlab.workspace_bundle import load_workspace, save_workspace


def fixture():
    mask = np.zeros((8, 8, 8), bool); mask[3:5, 3:5, 3:5] = True
    case = CaseData(case_id='explicit-codec-engineering-fixture', mri=np.arange(mask.size, dtype=np.float32).reshape(mask.shape),
                    affine=np.eye(4), compartments={'target': mask},
                    source_refs=(SourceRef('fixture', 'synthetic://surgical-archive-codec', provenance='simulated'),),
                    metadata={'anatomy_layers': {'target': {'kind': 'tumor'}}})
    engine = SurgicalEngine(from_case(case))
    for _ in range(4):
        assert engine.step({'slot': 'left', 'operation': 'inspect', 'duration': .1})['status'] == 'accepted'
    toolbox = {'mapping': engine.scene.regions, 'branches': {'main': engine.snapshot(),
                'exposure-prefix': engine.branch(2).snapshot()}, 'branchId': 'main', 'position': 4,
               'parents': {'main': {'branchId': 'exposure-prefix', 'position': 2}, 'exposure-prefix': None},
               'fullCase': {'main': {'schema': 'resectionlab.full-case-rehearsal.v1', 'sourceSceneId': engine.scene.identity,
                                     'lastStateId': engine.state_id, 'stage': 'inspect', 'status': 'running'}}}
    return case, engine, {'surgical_toolbox': toolbox, 'unrelated': {'keep': [1, 'unchanged', True]}}


def manifest(path):
    with ZipFile(path) as archive:
        return json.loads(archive.read('manifest.json'))


def test_actual_workspace_roundtrip_retains_all_branches_arrays_and_source_bytes(tmp_path):
    case, engine, artifacts = fixture()
    original = tmp_path / 'original.ressectionlab'
    save_workspace(case, original, sources=[], imaging=None, episode=None, selection=None, artifacts={})
    source_bytes = original.read_bytes()
    destination = tmp_path / 'full-case.ressectionlab'
    before = copy.deepcopy(artifacts)
    save_workspace(case, destination, sources=[], imaging=None, episode=None, selection=None, artifacts=artifacts)
    stored = manifest(destination)
    envelope = stored['artifacts']['surgical_toolbox']
    assert envelope['schema'] == codec.SCHEMA
    assert stored['artifacts']['unrelated'] == artifacts['unrelated']
    reopened, decoded, _ = load_workspace(destination)
    assert decoded == before and artifacts == before
    assert reopened.semantic_hash == case.semantic_hash
    assert reopened.source_refs == case.source_refs
    np.testing.assert_array_equal(reopened.mri, case.mri)
    np.testing.assert_array_equal(reopened.affine, case.affine)
    for name in case.compartments:
        np.testing.assert_array_equal(reopened.compartments[name], case.compartments[name])
    restored = SurgicalEngine.restore(from_case(reopened, decoded['surgical_toolbox']['mapping']),
                                      decoded['surgical_toolbox']['branches']['main'], use_cache=False)
    assert restored.snapshot() == engine.snapshot()
    assert original.read_bytes() == source_bytes
    with ZipFile(original) as old, ZipFile(destination) as new:
        assert old.read('arrays.npz') == new.read('arrays.npz')


def test_full_case_larger_than_outer_manifest_limit_saves_without_shortening_history(tmp_path):
    case, engine, artifacts = fixture()
    # Large repetitive controller bookkeeping exercises the codec without
    # inventing actions, anatomy, or physics state in the verified fixture.
    artifacts['surgical_toolbox']['fullCase']['main']['codecFixturePadding'] = 'recorded-bookkeeping;' * 900_000
    raw = json.dumps(artifacts, separators=(',', ':'), sort_keys=True).encode()
    assert len(raw) > MAX_MANIFEST_BYTES
    destination = tmp_path / 'large.ressectionlab'
    save_workspace(case, destination, sources=[], imaging=None, episode=None, selection=None, artifacts=artifacts)
    with ZipFile(destination) as archive:
        assert archive.getinfo('manifest.json').file_size < MAX_MANIFEST_BYTES
    assert read_case_artifacts(destination) == artifacts
    assert len(read_case_artifacts(destination)['surgical_toolbox']['branches']['main']['actions']) == len(engine.actions)


def test_old_raw_toolbox_archives_remain_readable(tmp_path):
    case, _, artifacts = fixture()
    del artifacts['surgical_toolbox']['fullCase']
    path = save_case(case, tmp_path / 'old.ressectionlab', artifacts=artifacts)
    assert manifest(path)['artifacts'] == artifacts
    assert read_case_artifacts(path) == artifacts


def test_large_staged_evidence_roundtrips_without_hiding_native_branches(tmp_path):
    case, engine, artifacts = fixture()
    del artifacts['surgical_toolbox']['fullCase']
    report = {'schema': codec.STAGED_REPORT_SCHEMA,
              'explicitGeneratedCodecPadding': 'native-probe-ledger;'*950_000}
    artifacts['surgical_toolbox']['stagedPlanning'] = report
    artifacts['surgical_toolbox']['stagedSummary'] = {'generatedCodecFixture': True}
    original = copy.deepcopy(artifacts)
    assert len(json.dumps(artifacts)) > MAX_MANIFEST_BYTES
    path = save_case(case, tmp_path/'staged.ressectionlab', artifacts=artifacts)
    stored = manifest(path)['artifacts']['surgical_toolbox']
    assert stored['branches'] == original['surgical_toolbox']['branches']
    assert stored['stagedPlanning']['schema'] == codec.STAGED_SCHEMA
    assert 'payload' not in stored, 'historical readers retain ordinary native branch access'
    assert read_case_artifacts(path) == original
    assert artifacts == original
    restored = SurgicalEngine.restore(engine.scene, stored['branches']['main'])
    assert restored.state_id == engine.state_id
    assert load_case(path).semantic_hash == case.semantic_hash
    corrupt = copy.deepcopy({'surgical_toolbox': stored})
    corrupt['surgical_toolbox']['stagedPlanning']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='checksum differs'):
        codec.decode_artifacts(corrupt)


def test_nested_staged_evidence_shares_the_expansion_budget(monkeypatch):
    staged = codec._encode({'schema': codec.STAGED_REPORT_SCHEMA, 'generatedPadding': 'x'*2048}, codec.STAGED_SCHEMA)
    outer = codec._encode({'fullCase': {'main': {}}, 'stagedPlanning': staged}, codec.SCHEMA)
    monkeypatch.setattr(codec, 'MAX_UNCOMPRESSED_BYTES',
                        outer['uncompressedBytes']+staged['uncompressedBytes']-1)
    with pytest.raises(ValueError, match='bounded JSON limit'):
        codec.decode_artifacts({'surgical_toolbox': outer})


@pytest.mark.parametrize('change', [
    {'schema': 'resectionlab.surgical-toolbox-archive.v99'}, {'schema': 'wrong'}, {'encoding': 'unknown'},
    {'sha256': '0' * 64}, {'uncompressedBytes': True}, {'uncompressedBytes': -1},
    {'uncompressedBytes': codec.MAX_UNCOMPRESSED_BYTES + 1}, {'payload': '!invalid-base64!'},
    {'payload': base64.b64encode(b'not gzip').decode()}, {'unexpected': 1},
])
def test_native_envelope_tampering_is_rejected(tmp_path, change):
    case, _, artifacts = fixture()
    path = save_case(case, tmp_path / 'tampered.ressectionlab', artifacts=artifacts)
    with ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    data = json.loads(members['manifest.json'])
    data['artifacts']['surgical_toolbox'].update(change)
    members['manifest.json'] = json.dumps(data).encode()
    with ZipFile(path, 'w', compression=ZIP_DEFLATED) as archive:
        for name, payload in members.items(): archive.writestr(name, payload)
    assert load_case(path).semantic_hash == case.semantic_hash, 'artifact corruption cannot alter source images'
    with pytest.raises(ImagingError, match='INVALID_SURGICAL_ARCHIVE'):
        read_case_artifacts(path)


def test_inflate_length_and_checksum_are_checked_before_decoded_json_parser(monkeypatch):
    raw = json.dumps({'fullCase': {'main': {}}, 'padding': 'x' * 2048}).encode()
    envelope = {'schema': codec.SCHEMA, 'encoding': codec.ENCODING,
                'sha256': sha256(raw).hexdigest(), 'uncompressedBytes': 512,
                'payload': base64.b64encode(gzip.compress(raw)).decode()}
    monkeypatch.setattr(codec, 'MAX_UNCOMPRESSED_BYTES', 1024)
    monkeypatch.setattr(codec.json, 'loads', lambda *_a, **_k: pytest.fail('Unverified inflated JSON reached the parser'))
    with pytest.raises(ValueError, match='length differs'):
        codec.decode_artifacts({'surgical_toolbox': envelope})
    envelope['uncompressedBytes'] = 2048
    with pytest.raises(ValueError, match='bounded JSON limit'):
        codec.decode_artifacts({'surgical_toolbox': envelope})
    envelope['uncompressedBytes'] = len(raw)
    monkeypatch.setattr(codec, 'MAX_UNCOMPRESSED_BYTES', len(raw))
    envelope['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='checksum differs'):
        codec.decode_artifacts({'surgical_toolbox': envelope})


@pytest.mark.parametrize('workspace', [False, True])
def test_failed_codec_save_never_overwrites_existing_destination(tmp_path, monkeypatch, workspace):
    case, _, artifacts = fixture()
    path = tmp_path / 'preserved.ressectionlab'
    save_case(case, path, artifacts={'existing': True})
    before = path.read_bytes()
    monkeypatch.setattr(codec, 'MAX_UNCOMPRESSED_BYTES', 20)
    with pytest.raises(ImagingError, match='128 MiB expanded JSON limit'):
        if workspace:
            save_workspace(case, path, sources=[], imaging=None, episode=None, selection=None, artifacts=artifacts)
        else:
            save_case(case, path, artifacts=artifacts)
    assert path.read_bytes() == before
