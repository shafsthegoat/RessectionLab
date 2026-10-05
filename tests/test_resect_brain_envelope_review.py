"""Independent metadata/mocked failure controls; no images or inference loaded."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('review_resect_envelope', ROOT/'scripts/run_resect_brain_envelope.py')
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)
DECLARATION = json.loads((ROOT/r.MANIFEST).read_text())


def snapshot_fixture(tmp_path, monkeypatch):
    snapshot = tmp_path/'build'/'frozen'
    required = [r.MANIFEST, 'scripts/run_resect_brain_envelope.py',
                'scripts/febio_runtime.py', 'src/resectionlab/brain_extraction.py']
    hashes = {}
    for name in required:
        path = snapshot/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes() if name == r.MANIFEST else b'# arithmetic source fixture\n')
        hashes[name] = r.sha(path)
    release = {'scope':r.SCOPE, 'manifest_sha256':r.MANIFEST_SHA,
               'snapshot_root':str(snapshot), 'snapshot_files_sha256':hashes,
               'execution_commit':'unit fixture; not an actual released commit'}
    path = tmp_path/'release.json'
    path.write_text(json.dumps(release))
    monkeypatch.setattr(r, '__file__', str(snapshot/'scripts/run_resect_brain_envelope.py'))
    return snapshot, path, release


@pytest.mark.parametrize('fault', ['changed_bound_source', 'unbound_extra_source', 'omitted_required_source'])
def test_archive_contract_stops_before_input_access_for_incomplete_source_closure(tmp_path,monkeypatch,fault):
    snapshot,path,release=snapshot_fixture(tmp_path,monkeypatch)
    r.contract(tmp_path,path,r.sha(path))
    if fault=='changed_bound_source':
        (snapshot/'src/resectionlab/brain_extraction.py').write_text('# altered source\n')
    elif fault=='unbound_extra_source':
        (snapshot/'src/resectionlab/unbound.py').write_text('# unexpected dependency\n')
    else:
        release['snapshot_files_sha256'].pop('scripts/febio_runtime.py')
        path.write_text(json.dumps(release))
    with pytest.raises(ValueError,match='snapshot|binding'):
        r.contract(tmp_path,path,r.sha(path))


def worker_fixture(tmp_path,monkeypatch,mode):
    spec=deepcopy(DECLARATION)
    snapshot=tmp_path/'build'/'frozen';snapshot.mkdir(parents=True)
    wrapper=snapshot/spec['pipeline']['wrapper']['path']
    wrapper.parent.mkdir(parents=True);wrapper.write_text('# mock wrapper; never executable\n')
    marker=snapshot/'marker.py';marker.write_text('# immutable dependency\n')
    release={'execution_commit':'arithmetic fixture', 'snapshot_files_sha256':{'marker.py':r.sha(marker)}}
    evidence=tmp_path/'evidence';evidence.mkdir()
    files=[];origins={}
    for package in ('numpy','torch','surfa','scipy','nibabel'):
        path=tmp_path/'runtime'/package/'__init__.py'
        path.parent.mkdir(parents=True);path.write_text('# mock package identity\n')
        files.append({'path':str(path.relative_to(tmp_path)),'sha256':r.sha(path),'bytes':path.stat().st_size})
        origins[package]=str(path)
    origins['xxhash']='not imported; wrapper records selected-byte attestation limitation'
    spec['runtime']['selected_file_bindings']=files
    probe={'versions':spec['runtime']['expected_versions'],'origins':origins,'mps_available':True,'sys_path':['mocked']}
    calls=[];verification_calls=[]
    monkeypatch.setattr(r,'contract',lambda *a:(release,snapshot,spec))
    monkeypatch.setattr(r,'verify_inputs',lambda *a:verification_calls.append('verified_metadata_fixture'))
    monkeypatch.setattr(sys,'path',sys.path.copy())
    def preflight(*args,**kwargs):
        calls.append(('runtime_probe',args,kwargs))
        if mode=='preflight_timeout':
            raise subprocess.TimeoutExpired(args[0],30)
        result=deepcopy(probe)
        if mode=='wrong_runtime_origin':result['origins']['torch']=str(tmp_path/'outside.py')
        return SimpleNamespace(stdout=json.dumps(result))
    monkeypatch.setattr(r.subprocess,'run',preflight)
    extraction=ModuleType('resectionlab.brain_extraction');extraction.__file__=str(wrapper)
    def infer(image,cache,output,**kwargs):
        calls.append(('model_invocation',image,cache,output,kwargs))
        output.mkdir(parents=True)
        (output/'partial.log').write_text('retained raw mock evidence\n')
        if mode=='inference_failure':raise RuntimeError('mocked native inference failure')
        if mode=='source_drift':marker.write_text('# changed during mocked child\n')
        return {'executed_runner_sha256':spec['pipeline']['expected_executed_runner_sha256'],
                'runtime_versions':{name:spec['runtime']['expected_versions'][name] for name in ('python','numpy','torch','surfa')},
                'configuration':{'device':'mps'},'brain_reviewed':False,'provenance':'estimated'}
    extraction.run_synthstrip=infer
    package=ModuleType('resectionlab');package.__file__=str(snapshot/'src/resectionlab/__init__.py')
    package.brain_extraction=extraction
    # Existing pytest imports are irrelevant to the isolated -I/-S real worker;
    # replace them only for this local fixture, restoring them automatically.
    for name in list(sys.modules):
        if name=='resectionlab' or name.startswith('resectionlab.'):
            monkeypatch.delitem(sys.modules,name)
    monkeypatch.setitem(sys.modules,'resectionlab',package)
    monkeypatch.setitem(sys.modules,'resectionlab.brain_extraction',extraction)
    return evidence,calls,verification_calls


@pytest.mark.parametrize('mode', ['preflight_timeout','wrong_runtime_origin','inference_failure','source_drift'])
def test_failure_is_durable_unapproved_and_does_not_retry_or_remove_partial_files(tmp_path,monkeypatch,mode):
    evidence,calls,verifications=worker_fixture(tmp_path,monkeypatch,mode)
    status=r.worker(tmp_path,tmp_path/'unused-release.json','bound-fixture-hash',evidence)
    record=json.loads((evidence/'worker-record.json').read_text())
    assert status==1 and record['status']=='failed_or_incomplete'
    assert 'failure' in record
    assert record['brain_reviewed'] is False
    assert record['working_brain_mask'] is None
    assert record['world_transform_accepted'] is False
    assert record['motion_or_annotation_access'] is False
    invocations=[call for call in calls if call[0]=='model_invocation']
    expected=0 if mode in {'preflight_timeout','wrong_runtime_origin'} else 1
    assert len(invocations)==record['model_inferences_requested']==expected
    if expected:
        assert (tmp_path/r.OUTPUT/'partial.log').read_text()=='retained raw mock evidence\n'
        assert invocations[0][1]==tmp_path/r.INPUT
        assert invocations[0][-1]['model']=='main'
        assert invocations[0][-1]['allow_download'] is False
        assert invocations[0][-1]['device']=='mps'
    else:
        assert not (tmp_path/r.OUTPUT).exists()


def test_mocked_success_is_still_unreviewed_and_never_runs_qc_or_other_inputs(tmp_path,monkeypatch):
    evidence,calls,verifications=worker_fixture(tmp_path,monkeypatch,'success')
    assert r.worker(tmp_path,tmp_path/'unused-release.json','bound-fixture-hash',evidence)==0
    record=json.loads((evidence/'worker-record.json').read_text())
    assert record['status']=='completed_unreviewed_estimate_pending_independent_QC'
    assert record['model_inferences_requested']==1 and len(verifications)==2
    assert [call[0] for call in calls]==['runtime_probe','model_invocation']
    assert record['brain_reviewed'] is False and record['working_brain_mask'] is None
    assert record['world_transform_accepted'] is False and record['motion_or_annotation_access'] is False
    assert record['source_and_runtime_pins_unchanged'] is True
    assert 'partial.log' in record['output_files']
    assert all(Path(value).is_relative_to(tmp_path/'build/frozen/src') for value in record['project_module_origins'].values())
