"""Focused generated-DTO controls; never load the real experiment checkpoints."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_native_opening_length_sensitivity as runner


def forbidden(*args,**kwargs):
    raise AssertionError("Prohibited engine/proposer/patient/gradient call")


@pytest.fixture(autouse=True)
def thread_and_prohibitions(monkeypatch):
    from resectionlab.native_spatial_task import NativeSpatialTask
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.native_proposals import PreparedNominalCavityProposer
    from resectionlab.imaging import load_case
    import resectionlab.imaging as imaging
    old=torch.get_num_threads();torch.set_num_threads(1)
    monkeypatch.setattr(NativeSpatialTask,"__init__",forbidden)
    monkeypatch.setattr(NativeResectionEngine,"__init__",forbidden)
    monkeypatch.setattr(PreparedNominalCavityProposer,"propose",forbidden)
    monkeypatch.setattr(imaging,"load_case",forbidden)
    monkeypatch.setattr(torch.optim.Adam,"__init__",forbidden)
    monkeypatch.setattr(torch.Tensor,"backward",forbidden)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def prepared():
    records,baselines=runner.read_inputs()
    original=runner.reconstruct_root(records,baselines)
    return original,runner.intervene(original),baselines


def test_exact_historical_root_reconstruction_without_engine_or_forward(monkeypatch,prepared):
    from resectionlab.spatial_policy import SpatialPolicy
    monkeypatch.setattr(SpatialPolicy,"forward",forbidden)
    original,changed,_=prepared
    assert original.fingerprint==runner.ROOT_OBSERVATION_HASH
    assert original.source_id==runner.SOURCE_HASH
    assert original.action_geometry[1:,12].tolist()==[2.2,12.,2.2,12.]
    assert changed.action_geometry[1:,12].tolist()==[120.]*4
    assert original.image_channels.shape==(6,9,9,7)
    assert changed.fingerprint!=original.fingerprint
    runner.validate_intervention(original,changed)
    differences=np.argwhere(original.action_geometry!=changed.action_geometry)
    assert differences.tolist()==[[1,12],[2,12],[3,12],[4,12]]


@pytest.mark.parametrize("field",["rays","mask","state","image","ids","source","stop","length"])
def test_undeclared_changes_refused_before_loader(monkeypatch,prepared,field):
    original,changed,baselines=prepared
    monkeypatch.setattr(runner,"load_frozen_checkpoint",forbidden)
    if field in {"rays","stop","length"}:
        a=np.array(changed.action_geometry)
        if field=="rays":
            a[1,1]+=0.01;a[1,4]+=0.01
        elif field=="stop": a[0,12]=1.
        else: a[[1,3],12]=119.
        # STOP malformed rows fail at DTO construction rather than downstream.
        if field=="stop":
            with pytest.raises(ValueError): replace(changed,action_geometry=a)
            return
        bad=replace(changed,action_geometry=a)
    elif field=="mask":
        a=np.array(changed.action_mask);a[1]=False
        with pytest.raises(ValueError): replace(changed,action_mask=a)
        return
    elif field=="state":
        a=np.array(changed.state_features);a[1]=3;bad=replace(changed,state_features=a)
    elif field=="image":
        a=np.array(changed.image_channels);a[0,0,0,0]=.1;bad=replace(changed,image_channels=a)
    elif field=="ids": bad=replace(changed,action_ids=("STOP","different",*changed.action_ids[2:]))
    else: bad=replace(changed,source_id="generated-other")
    ledger={"checkpoint_attempts":[],"forward_attempts":[],"completed_forwards":[]}
    with pytest.raises(ValueError): runner.forward_once("initial",original,bad,baselines["initial"],ledger)
    assert ledger["checkpoint_attempts"]==[]


def test_wrong_root_cannot_be_relabelled_as_intervention(prepared):
    original,changed,_=prepared
    with pytest.raises(ValueError): runner.intervene(changed)
    with pytest.raises(ValueError): runner.require_original(replace(original,source_id="different"))


def test_saved_baseline_order_and_finite_probabilities(prepared):
    original,_,baselines=prepared
    b=baselines["initial"]
    c=runner.contrast(b["logits"],b["probabilities"],b["critic_value"],b,original.action_ids)
    assert c["margin_change"]==0 and c["stop_probability_change"]==0
    assert c["stop_logit_change"]==0
    wrong=copy.deepcopy(b);wrong["action_ids"][1:]=reversed(wrong["action_ids"][1:])
    with pytest.raises(ValueError): runner.contrast(b["logits"],b["probabilities"],0.,wrong,original.action_ids)
    with pytest.raises(ValueError): runner.contrast([float('nan')]*5,b["probabilities"],0.,b,original.action_ids)


def test_two_call_ledger_with_small_spy_no_real_checkpoint_load(monkeypatch,prepared):
    """Exercise the controller, not real checkpoint inference, before release."""
    original,changed,baselines=prepared
    import resectionlab.spatial_policy as policies
    calls=[]
    class Spy:
        architecture_hash=runner.shared.ARCHITECTURE
        def parameters(self): return ()
        def __call__(self,observation):
            assert torch.is_inference_mode_enabled()
            assert observation is changed
            calls.append(observation.fingerprint)
            return torch.tensor([0.,1.,2.,3.,4.]),torch.tensor(.5)
    spy=Spy();active=[]
    def load(name): active[:]=[name];return spy
    monkeypatch.setattr(runner,"load_frozen_checkpoint",load)
    monkeypatch.setattr(policies,"parameter_hash",lambda model:runner.shared.CHECKPOINTS[active[0]]["parameter_hash"])
    monkeypatch.setattr(runner.shared,"checked_bytes",lambda *a:b"spy-only")
    ledger={"checkpoint_attempts":[],"forward_attempts":[],"completed_forwards":[]}
    for name in ("initial","RL256"):
        result=runner.forward_once(name,original,changed,baselines[name],ledger)
        assert result["status"]=="complete" and result["weights_unchanged"]
    assert len(calls)==2 and ledger["completed_forwards"]==["initial","RL256"]
    with pytest.raises(ValueError): runner.forward_once("initial",original,changed,baselines['initial'],ledger)
    assert len(calls)==2


def test_model_exception_retains_attempt_not_completion(monkeypatch,prepared):
    original,changed,baselines=prepared
    class Bad:
        def __call__(self,observation):raise RuntimeError("injected forward failure")
    monkeypatch.setattr(runner,"load_frozen_checkpoint",lambda name:Bad())
    ledger={"checkpoint_attempts":[],"forward_attempts":[],"completed_forwards":[]}
    with pytest.raises(RuntimeError,match="injected"):
        runner.forward_once("initial",original,changed,baselines['initial'],ledger)
    assert ledger=={"checkpoint_attempts":["initial"],"forward_attempts":["initial"],"completed_forwards":[]}
    with pytest.raises(ValueError):runner.forward_once("initial",original,changed,baselines['initial'],ledger)


def test_checkpoint_byte_failure_before_deserialization(monkeypatch):
    def wrong(*args): raise ValueError("exact byte failure")
    monkeypatch.setattr(runner.shared,"checked_bytes",wrong)
    monkeypatch.setattr(torch,"load",forbidden)
    with pytest.raises(ValueError,match="exact byte"):runner.load_frozen_checkpoint("initial")


def test_software_checkpoint_loader_ancestry_without_forward(monkeypatch,tmp_path):
    from resectionlab.spatial_policy import SpatialPolicy,SpatialPolicyConfig,parameter_hash
    monkeypatch.setattr(SpatialPolicy,"forward",forbidden)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11);model=SpatialPolicy(SpatialPolicyConfig(critic_candidate_context=True))
    initial=runner.shared.CHECKPOINTS["initial"]["parameter_hash"]
    assert parameter_hash(model)==initial
    checkpoint={"updates":0,"parameter_hash":initial,"initial_parameter_hash":initial,
        "architecture":model.architecture_record(),"policy":model.state_dict(),
        "context":{"declaration_sha256":runner.INPUTS[runner.RL+"declaration-input.json"],
            "source_ids":(runner.SOURCE_HASH,),"decision_model_hash":runner.MODEL_HASH,"max_steps":2}}
    path=tmp_path/"software-only.pt";torch.save(checkpoint,path)
    specs=copy.deepcopy(runner.shared.CHECKPOINTS)
    specs["initial"].update(path=path.name,file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(runner,"ROOT",tmp_path);monkeypatch.setattr(runner.shared,"CHECKPOINTS",specs)
    loaded=runner.load_frozen_checkpoint("initial")
    assert parameter_hash(loaded)==initial and not any(p.requires_grad for p in loaded.parameters())
    checkpoint["context"]["decision_model_hash"]="sha256:"+"0"*64
    torch.save(checkpoint,path);specs["initial"]["file_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match="ancestry"):runner.load_frozen_checkpoint("initial")


def test_forward_order_refused_before_load(monkeypatch,prepared):
    original,changed,baselines=prepared
    monkeypatch.setattr(runner,"load_frozen_checkpoint",forbidden)
    ledger={"checkpoint_attempts":[],"forward_attempts":[],"completed_forwards":[]}
    with pytest.raises(ValueError,match="declared order"):
        runner.forward_once("RL256",original,changed,baselines["RL256"],ledger)
    assert ledger["checkpoint_attempts"]==[]


def test_worker_failure_preserves_both_declared_slots(monkeypatch,tmp_path,prepared):
    original,_,baselines=prepared
    monkeypatch.setattr(runner,"validate",lambda record:({},baselines))
    monkeypatch.setattr(runner,"reconstruct_root",lambda *a:original)
    def fail(*args): raise ValueError("injected load rejection")
    monkeypatch.setattr(runner,"load_frozen_checkpoint",fail)
    with pytest.raises(ValueError):runner.worker({},tmp_path,"0"*64)
    result=json.loads((tmp_path/"result.json").read_text())
    assert result["status"]=="failed"
    assert set(result["methods"])=={"initial","RL256"}
    assert all(row["outputs"] is None for row in result["methods"].values())
    assert result["ledger"]["checkpoint_attempts"]==["initial"]
    assert result["ledger"]["forward_attempts"]==[]


def test_closed_declaration_no_patient_authority_reads(monkeypatch):
    record=runner.declaration()
    monkeypatch.setattr(runner.shared,"read_authorities",forbidden)
    records,_=runner.validate(record)
    assert all("pat05" not in path.lower() for path in records)
    assert "src/resectionlab/data_policy.py" in record["source_sha256"]
    for mutation in ("source","input","count","length","checkpoint"):
        bad=copy.deepcopy(record)
        if mutation=="source":bad["source_sha256"].pop("src/resectionlab/data_policy.py")
        elif mutation=="input":bad["input_sha256"].pop(RL_KEY)
        elif mutation=="count":bad["settings"]["new_forwards_per_checkpoint"]=2
        elif mutation=="length":bad["settings"]["working_length_descriptor_mm"]=12.
        else:bad["checkpoints"]["RL256"]["parameter_hash"]="sha256:"+"0"*64
        with pytest.raises(ValueError):runner.validate(bad)


RL_KEY=runner.RL+"result.json"


def test_manifest_after_freeze():
    path=ROOT/"manifests/experiments/native-opening-length-sensitivity-v1.json"
    if not path.exists():pytest.skip("written after final source freeze")
    runner.validate(json.loads(path.read_text()))


@pytest.mark.parametrize("failed_name", ["initial", "RL256"])
def test_terminal_failure_closes_active_slot_preserves_prior(monkeypatch, tmp_path, prepared, failed_name):
    original, changed, baselines = prepared
    monkeypatch.setattr(runner, "validate", lambda record: ({}, baselines))
    monkeypatch.setattr(runner, "reconstruct_root", lambda *args: original)
    def fake_forward(name, original, changed, baseline, ledger):
        ledger["checkpoint_attempts"].append(name)
        if name == failed_name:
            raise ValueError("injected checkpoint load refusal")
        # Pure controller completion, no checkpoint/model was loaded or called.
        ledger["forward_attempts"].append(name)
        ledger["completed_forwards"].append(name)
        return {"margin_change": 0., "stop_probability": .2, "stop_logit_change": 0.,
                "highest_ranked_id_not_executed": "STOP", "scope": "test-controller-only"}
    monkeypatch.setattr(runner, "forward_once", fake_forward)
    with pytest.raises(ValueError, match="injected checkpoint"):
        runner.worker({}, tmp_path, "0" * 64)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "failed"
    expected_attempts = ["initial"] if failed_name == "initial" else ["initial", "RL256"]
    expected_completed = [] if failed_name == "initial" else ["initial"]
    assert result["ledger"]["checkpoint_attempts"] == expected_attempts
    assert result["ledger"]["completed_forwards"] == expected_completed
    assert result["methods"][failed_name] == {"status": "failed", "outputs": None}
    if failed_name == "initial":
        assert result["methods"]["RL256"] == {"status": "not_started", "outputs": None}
        assert not (tmp_path / "initial.json").exists()
    else:
        assert result["methods"]["initial"]["status"] == "complete"
        assert result["methods"]["initial"]["outputs"] == "initial.json"
        assert json.loads((tmp_path / "initial.json").read_text())["scope"] == "test-controller-only"
    assert not (tmp_path / "RL256.json").exists()

