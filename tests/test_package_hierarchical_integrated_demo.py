"""Safety and source-snapshot contracts for the local hierarchy bundle."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import package_hierarchical_integrated_demo as package


def test_extraction_rejects_parent_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(package, 'ROOT', tmp_path)
    base = tmp_path/'testcases'
    base.mkdir()
    archive = base/'malicious.tar.gz'
    payload = b'overwrite'
    with tarfile.open(archive, 'w:gz') as tar:
        entry = tarfile.TarInfo('../escape.txt')
        entry.size = len(payload)
        tar.addfile(entry, io.BytesIO(payload))
    with pytest.raises(ValueError, match='Unsafe hierarchical archive member'):
        package.extract(archive, base/'extracted')
    assert not (base/'escape.txt').exists()


def test_extraction_rejects_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(package, 'ROOT', tmp_path)
    base = tmp_path/'testcases'
    base.mkdir()
    archive = base/'malicious-link.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        entry = tarfile.TarInfo('model.pt')
        entry.type = tarfile.SYMTYPE
        entry.linkname = '/outside/model.pt'
        tar.addfile(entry)
    with pytest.raises(ValueError, match='Unsafe hierarchical archive member'):
        package.extract(archive, base/'extracted')


def test_build_requires_two_real_desktop_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(package, 'ROOT', tmp_path)
    (tmp_path/'testcases').mkdir()
    with pytest.raises(ValueError, match='both development cases'):
        package.build(tmp_path/'testcases/new', 'HEAD', {})


def test_instructions_preserve_failed_transfer_and_runtime_limit():
    text = package._instructions('a'*40)
    assert '0005' in text and 'incomplete' in text
    assert 'zero optimizer updates' in text
    assert 'same-host Python environment' in text
    assert 'six' in text and 'twelve' in text


def test_exit_profile_pins_v4_inputs_and_cleanup_semantics():
    profile = package._exit_profile('a'*64, {'0007': 'b'*64, '0005': 'c'*64})
    assert profile['id'] == 'exit-preflight-v1'
    assert profile['sources']['0007'] == {
        'path': 'cases/0007-sector-v4.ressectionlab',
        'sha256': '2441fc3f9c39ceaa0be495806fe9b28882d973527970685b373b645f8e084143',
        'receipt': 'evidence/sector-comparison-0007-v4/receipt.json',
    }
    assert profile['sources']['0005']['sha256'] == (
        '4047e1dcd8197b2319add61ba24c0c0b36782b15725c8cb156ecd864d288a8c7')
    assert profile['fitting']['receipt'] == 'evidence/patient-sector-fit-v5/receipt.json'
    assert profile['taskGates'] == {'0007': True, '0005': False}
    assert profile['cleanupComplete'] == {'0007': True, '0005': True}


def test_exit_verifier_rejects_relabelled_v3_source_in_complete_inventory(tmp_path):
    bundle = tmp_path/'bundle'
    required = {
        'src/resectionlab/surgical_engine.py',
        'src/resectionlab/surgical_prefix_json_native.c',
        'src/resectionlab/surgical_sector_planning.py',
        'src/resectionlab/patient_sector_learning.py',
        'scripts/run_patient_sector_learning.py',
        'scripts/execute_patient_sector_policy.py',
        'scripts/compare_anatomy_sectors.py',
        'scripts/package_hierarchical_integrated_demo.py',
        'desktop/pnpm-lock.yaml',
        'evidence/patient-sector-fit-v5/receipt.json',
        'evidence/desktop-0007/report.json',
        'evidence/desktop-0005/report.json',
    }
    files = []
    for name in sorted(required):
        path = bundle/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
        files.append({'path': name, 'bytes': path.stat().st_size,
                      'sha256': package.historical.sha(path)})
    profile = package._exit_profile(package.historical.sha(
        bundle/'evidence/patient-sector-fit-v5/receipt.json'),
        {patient: package.historical.sha(bundle/f'evidence/desktop-{patient}/report.json')
         for patient in ('0007', '0005')})
    profile['sources']['0007']['path'] = 'cases/0007-sector-v3.ressectionlab'
    manifest = {'schema': package.EXIT_SCHEMA, 'sourceCommit': 'a'*40,
                'profile': profile, 'files': files, 'fileCount': len(files),
                'contentBytes': sum(row['bytes'] for row in files)}
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='profile paths or hashes differ'):
        package.verify(bundle)


def test_legacy_manifest_routes_to_legacy_verifier(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({
        'schema': package.SCHEMA, 'selectedDemoSeed': 29,
        'sourceCommit': 'a'*40,
    }))
    with pytest.raises(ValueError, match='Hierarchical package schema or source commit differs'):
        package.verify(tmp_path)


def test_native_snapshot_rejects_forged_case_and_branch_identity():
    source = {
        'sourceCaseHash': 'sha256:source', 'sceneId': 'sha256:scene',
        'accessComparisonSeal': 'sha256:access',
        'sectorComparisonSeal': 'sha256:sector',
        'originalBranchInventory': [
            {'branchId': 'access-a', 'stateId': 'sha256:state',
             'actionsHash': 'sha256:actions'}],
        'learnedBranches': 0,
    }
    package._assert_same_native_source(source, json.loads(json.dumps(source)))
    forged_case = json.loads(json.dumps(source))
    forged_case['sourceCaseHash'] = 'sha256:other'
    with pytest.raises(ValueError, match='source case/comparison/branch state'):
        package._assert_same_native_source(source, forged_case)
    spliced_branch = json.loads(json.dumps(source))
    spliced_branch['originalBranchInventory'][0]['actionsHash'] = 'sha256:other-actions'
    with pytest.raises(ValueError, match='source case/comparison/branch state'):
        package._assert_same_native_source(source, spliced_branch)
