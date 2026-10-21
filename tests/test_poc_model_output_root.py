"""Output-boundary controls; no inference runtime or patient input is opened."""
from pathlib import Path

import pytest
from resectionlab import poc_models


@pytest.mark.parametrize('escape', ['outside', 'traversal', 'symlink'])
def test_explicit_app_testing_cache_cannot_escape_testcases(tmp_path, monkeypatch, escape):
    repository = tmp_path/'repository'
    testing = repository/'testcases'; testing.mkdir(parents=True)
    weights = repository/'data/models/synthstrip-v1/synthstrip.1.pt'
    weights.parent.mkdir(parents=True); weights.write_bytes(b'generated test bytes; never loaded')
    source = tmp_path/'generated-source'; source.write_bytes(b'not interpreted as an image')
    outside = tmp_path/'outside'
    if escape == 'symlink':
        outside.mkdir(); (testing/'linked').symlink_to(outside,target_is_directory=True)
        output = testing/'linked'
    elif escape == 'traversal': output = testing/'..'/'escaped'
    else: output = outside
    monkeypatch.setattr(poc_models,'ROOT',repository)
    monkeypatch.setattr(poc_models.subprocess,'Popen',lambda *a,**k:pytest.fail('Inference started before output admission'))
    with pytest.raises(ValueError,match='testcases'):
        poc_models._run(source,'generated','synthstrip-main','T1',output_root=output)
    assert not (output/'generated').exists()


def test_explicit_cache_is_used_without_touching_historical_outputs(tmp_path, monkeypatch):
    repository = tmp_path/'repository'; testing = repository/'testcases'; testing.mkdir(parents=True)
    weights = repository/'data/models/synthstrip-v1/synthstrip.1.pt'
    weights.parent.mkdir(parents=True); weights.write_bytes(b'generated test bytes; never loaded')
    source = tmp_path/'generated-source'; source.write_bytes(b'not interpreted as an image')
    monkeypatch.setattr(poc_models,'ROOT',repository)
    class StopBeforeInference(Exception): pass
    def stop(*args,**kwargs): raise StopBeforeInference
    monkeypatch.setattr(poc_models.subprocess,'Popen',stop)
    with pytest.raises(StopBeforeInference):
        poc_models._run(source,'generated','synthstrip-main','T1',output_root=testing/'inference')
    records = list((testing/'inference').rglob('request.json'))
    assert len(records)==1
    assert not (repository/'outputs').exists()
