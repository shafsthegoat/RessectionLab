"""Five metadata/refusal controls; no patient arrays or checkpoint forwards."""
import copy
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from resectionlab import pat05_forward_diagnostic as bridge
import run_pat05_forward_diagnostic as runner

def forbidden(*args,**kwargs):raise AssertionError('No arrays, checkpoint load, forward or learning permitted')

def test_changed_case_authority_refused_before_array_load(monkeypatch):
    import resectionlab.imaging as imaging
    records=bridge.read_authorities(ROOT)
    records[bridge.HISTORICAL+'declaration-input.json']['member']['case_bundle']='unpermitted-case.npz'
    monkeypatch.setattr(imaging,'load_case',forbidden)
    with pytest.raises(ValueError,match='authority records changed'):
        bridge.reconstruct_pat05(ROOT,records)

def test_non_dto_refused_before_checkpoint_byte_access(monkeypatch):
    import torch
    monkeypatch.setattr(bridge,'checked_bytes',forbidden)
    monkeypatch.setattr(torch,'load',forbidden)
    context=bridge.Pat05ForwardContext('0'*64)
    with pytest.raises(TypeError,match='immutable spatial DTO'):
        context.forward_checkpoint(ROOT,'initial',object())
    assert context.snapshot()['checkpoint_attempts']==0 and context.snapshot()['forward_attempts']==0

def test_new_context_has_no_training_or_support_admission():
    from resectionlab.data_policy import DataPolicyError,require_admitted_model
    from resectionlab.spatial_policy import imitation_loss,reinforce_loss,gradient_step
    context=bridge.Pat05ForwardContext('0'*64)
    with pytest.raises(DataPolicyError):imitation_loss(None,[],learning_context=context)
    with pytest.raises(DataPolicyError):reinforce_loss(None,[],learning_context=context)
    with pytest.raises(DataPolicyError):gradient_step(None,None,None,learning_context=context)
    with pytest.raises(DataPolicyError):require_admitted_model('sha256:37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33','independent_pinned_support_refusal')

def test_failed_checkpoint_bytes_consume_attempt_without_forward(monkeypatch):
    import torch
    context=bridge.Pat05ForwardContext('0'*64)
    # The observation verifier is stubbed only to reach the byte-failure branch;
    # no observation arrays or checkpoint payload are fabricated or read.
    monkeypatch.setattr(bridge.Pat05ForwardContext,'require_observation',lambda *a:None)
    def fail_bytes(*a):raise ValueError('Exact input bytes changed')
    monkeypatch.setattr(bridge,'checked_bytes',fail_bytes)
    monkeypatch.setattr(torch,'load',forbidden)
    with pytest.raises(ValueError,match='bytes changed'):context.forward_checkpoint(ROOT,'initial',object())
    assert context.snapshot()['checkpoint_attempts']==1 and context.snapshot()['forward_attempts']==0
    with pytest.raises(ValueError,match='no retries'):context.forward_checkpoint(ROOT,'initial',object())

def test_terminal_checkpoint_failure_closes_running_method(monkeypatch,tmp_path):
    import torch
    monkeypatch.setattr(torch,'load',forbidden)
    monkeypatch.setattr(runner,'validate',lambda _: {bridge.HISTORICAL+'receipt.json':{'preparation_seconds':0.}})
    monkeypatch.setattr(bridge,'reconstruct_pat05',lambda *a:(object(),{}))
    monkeypatch.setattr(bridge,'describe_observation',lambda *a:{})
    monkeypatch.setattr(bridge.Pat05ForwardContext,'require_observation',lambda *a:None)
    def refuse(*a):raise ValueError('CHECKPOINT_BYTES_CHANGED')
    monkeypatch.setattr(bridge.Pat05ForwardContext,'forward_checkpoint',refuse)
    with pytest.raises(ValueError,match='CHECKPOINT_BYTES_CHANGED'):
        runner.worker({},tmp_path,'0'*64)
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['status']=='failed'
    assert result['methods']['initial']['status']=='failed'
    assert result['methods']['initial']['outputs'] is None
    assert result['methods']['RL256']=={'status':'not_started','outputs':None}
