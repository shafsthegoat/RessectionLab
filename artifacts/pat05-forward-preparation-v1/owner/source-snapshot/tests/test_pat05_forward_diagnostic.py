"""Analytical software controls only; no PAT05 decode or checkpoint forward."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from resectionlab import pat05_forward_diagnostic as bridge
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.native_spatial_task import NativeSpatialCase, NativeSpatialTask
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_proposals import PreparedNominalCavityProposer
from resectionlab.spatial_observations import SpatialAction, ObservedProcedureState, build_spatial_observation
from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig, parameter_hash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_pat05_forward_diagnostic as runner


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def forbidden(*args, **kwargs):
    raise AssertionError("No task, native preview/step, loss or optimizer may run")


def tiny(monkeypatch):
    monkeypatch.setattr(NativeSpatialTask, "__init__", forbidden)
    monkeypatch.setattr(NativeResectionEngine, "__init__", forbidden)
    monkeypatch.setattr(PreparedNominalCavityProposer, "propose", forbidden)
    shape = (5,5,5)
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    support = np.ones(shape, bool); target = np.zeros(shape, bool); target[2,2,3] = True
    tool = ToolGeometry("tiny-test", .9, .45, 12., tip_length_mm=.75, max_access_angle_deg=35.)
    source = NativeSpatialCase(image,support,target,np.eye(4),AccessWindow((2.,2.,0.),(0.,0.,1.),2.,"tiny"),
        (tool,),nominal_target=target,track="annotation_assisted",target_derivation="analytical software annotation",
        crop_shape=shape,intensity_normalization="support_percentile_1_99",proposal_mode="nominal_cavity_v1")
    actions = [SpatialAction("STOP"),SpatialAction("recorded-only",(2.,2.,0.),(2.,2.,3.),tool)]
    obs = build_spatial_observation(source.spatial_inputs(np.zeros(shape,bool)),actions,
        ObservedProcedureState(source.access,0,3,None))
    inventory = {"complete":True,"accepted_count":1,"emitted":[{"feasible":True,"action_id":"recorded-only",
        "entry_mm":[2.,2.,0.],"tip_mm":[2.,2.,3.],"tool_id":tool.tool_id}]}
    decision = {"step":0,"observation_hash":obs.fingerprint,"action_ids":list(obs.action_ids),"action_mask":obs.action_mask.tolist()}
    return source,obs,inventory,decision


def test_metadata_authorities_without_case_or_checkpoint_decode(monkeypatch):
    import resectionlab.imaging as imaging
    monkeypatch.setattr(imaging,"load_case",forbidden)
    monkeypatch.setattr(torch,"load",forbidden)
    records=bridge.read_authorities(ROOT)
    assert records[bridge.READINESS]["roles"]["SELECT"] == ["sub-PAT26","sub-PAT27"]
    assert records[bridge.HISTORICAL+"initial_policy.json"]["decisions"][0]["observation_hash"] == bridge.OBSERVATION


def test_actual_native_case_reconstruction_never_instantiates_engine(monkeypatch):
    source,obs,inventory,decision=tiny(monkeypatch)
    rebuilt=bridge._rebuild_saved_observation(source,inventory,decision,max_steps=3)
    assert rebuilt.fingerprint == obs.fingerprint
    np.testing.assert_array_equal(rebuilt.image_channels,obs.image_channels)
    assert rebuilt.channel_available.tolist() == [True,True,True,True,False,False]
    assert not rebuilt.coverage[4:].any()


@pytest.mark.parametrize("change",["geometry","ordering","mask","fingerprint","horizon","count","partial_inventory"])
def test_reconstruction_rejects_whole_dto_changes(monkeypatch,change):
    source,obs,inventory,decision=tiny(monkeypatch)
    horizon=3
    if change=="geometry": inventory["emitted"][0]["tip_mm"][2]=2.5
    elif change=="ordering": decision["action_ids"].reverse()
    elif change=="mask": decision["action_mask"][1]=False
    elif change=="fingerprint": decision["observation_hash"]="sha256:"+"0"*64
    elif change=="horizon": horizon=2
    elif change=="count": inventory["accepted_count"]=2
    elif change=="partial_inventory": inventory["complete"]=False
    with pytest.raises(ValueError): bridge._rebuild_saved_observation(source,inventory,decision,max_steps=horizon)


def test_context_refuses_non_pat05_before_checkpoint_load(monkeypatch):
    _,obs,_,_=tiny(monkeypatch)
    monkeypatch.setattr(torch,"load",forbidden)
    context=bridge.Pat05ForwardContext("0"*64)
    with pytest.raises(ValueError,match="exact historical"):
        context.forward_checkpoint(ROOT,"initial",obs)
    assert context.snapshot()["forward_attempts"] == 0


def software_checkpoint_context(monkeypatch,tmp_path):
    """Explicitly replaced test authority, never a patient admission artifact."""
    _,obs,_,_=tiny(monkeypatch)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11);model=SpatialPolicy(SpatialPolicyConfig(critic_candidate_context=True))
    specs={}
    for name,updates in [("initial",0),("RL256",256)]:
        path=tmp_path/(name+".pt")
        torch.save({"updates":updates,"parameter_hash":parameter_hash(model),"initial_parameter_hash":parameter_hash(model),
            "architecture":model.architecture_record(),"policy":model.state_dict(),
            "context":{"declaration_sha256":bridge.METADATA_SHA256[bridge.RL+"declaration-input.json"]}},path)
        specs[name]={"path":path.name,"file_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "parameter_hash":parameter_hash(model),"updates":updates}
    monkeypatch.setattr(bridge,"CHECKPOINTS",specs)
    context=bridge.Pat05ForwardContext("0"*64)
    def require_exact(value):
        value.assert_intact()
        if value is not obs: raise ValueError("Wrong software fixture observation")
    monkeypatch.setattr(bridge.Pat05ForwardContext,"require_observation",lambda self,value: require_exact(value))
    return context,obs


def test_exact_two_software_forwards_frozen_no_optimizer_or_actions(monkeypatch,tmp_path):
    context,obs=software_checkpoint_context(monkeypatch,tmp_path)
    monkeypatch.setattr(torch.optim.Adam,"__init__",forbidden)
    monkeypatch.setattr(torch.Tensor,"backward",forbidden)
    rows=[context.forward_checkpoint(tmp_path,name,obs) for name in ("initial","RL256")]
    assert rows[0]["logits"] == rows[1]["logits"]
    assert all(r["weights_unchanged"] for r in rows)
    assert context.snapshot()["completed_forwards"] == 2
    with pytest.raises(ValueError,match="no retries"): context.forward_checkpoint(tmp_path,"initial",obs)
    with pytest.raises(ValueError): context.forward_checkpoint(tmp_path,"other",obs)


def test_bad_checkpoint_bytes_fail_before_model_forward(monkeypatch,tmp_path):
    context,obs=software_checkpoint_context(monkeypatch,tmp_path)
    (tmp_path/"initial.pt").write_bytes(b"corrupt")
    monkeypatch.setattr(SpatialPolicy,"forward",forbidden)
    with pytest.raises(ValueError,match="bytes changed"): context.forward_checkpoint(tmp_path,"initial",obs)
    assert context.snapshot()["completed_forwards"] == 0
    with pytest.raises(ValueError,match="no retries"): context.forward_checkpoint(tmp_path,"initial",obs)


def test_stale_training_ancestry_fails_before_forward(monkeypatch,tmp_path):
    context,obs=software_checkpoint_context(monkeypatch,tmp_path)
    path=tmp_path/"initial.pt"
    checkpoint=torch.load(path,weights_only=False)
    checkpoint["context"]["declaration_sha256"]="1"*64
    torch.save(checkpoint,path)
    bridge.CHECKPOINTS["initial"]["file_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(SpatialPolicy,"forward",forbidden)
    with pytest.raises(ValueError,match="ancestry"): context.forward_checkpoint(tmp_path,"initial",obs)


def test_frozen_weight_mutation_cannot_publish(monkeypatch,tmp_path):
    context,obs=software_checkpoint_context(monkeypatch,tmp_path)
    original=SpatialPolicy.forward
    def bad(self,observation):
        result=original(self,observation)
        next(self.parameters()).add_(1)
        return result
    monkeypatch.setattr(SpatialPolicy,"forward",bad)
    with pytest.raises(RuntimeError,match="changed weights"): context.forward_checkpoint(tmp_path,"initial",obs)
    assert context.snapshot()["completed_forwards"] == 0


def test_existing_model_and_learning_gates_unchanged():
    from resectionlab.data_policy import require_admitted_model,DataPolicyError
    from resectionlab.spatial_policy import imitation_loss
    with pytest.raises(DataPolicyError):
        require_admitted_model("sha256:37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33","test")
    with pytest.raises(DataPolicyError): imitation_loss(None,[])


def test_closed_source_inventory_and_no_other_case_paths():
    paths=runner.dependency_paths()
    assert "src/resectionlab/pat05_forward_diagnostic.py" in paths
    assert "src/resectionlab/native_spatial_task.py" in paths
    assert "src/resectionlab/data_policy.py" in paths
    assert "src/resectionlab/desktop_bridge.py" not in paths
    record=runner.declaration()
    runner.validate(record)
    for change in ("source","subject","resources","checkpoint","forward_count"):
        bad=copy.deepcopy(record)
        if change=="source": bad["source_sha256"].pop("src/resectionlab/data_policy.py")
        elif change=="subject": bad["subject"]="sub-PAT26"
        elif change=="resources": bad["settings"]["max_wall_seconds"]=120.
        elif change=="checkpoint": bad["checkpoints"]["RL256"]["parameter_hash"]="sha256:"+"0"*64
        else: bad["settings"]["forwards_per_checkpoint"]=2
        with pytest.raises(ValueError): runner.validate(bad)


def test_metadata_mutation_precedes_decode(monkeypatch):
    import resectionlab.imaging as imaging
    records=bridge.read_authorities(ROOT)
    records[bridge.HISTORICAL+"declaration-input.json"]["member"]["role"]="SELECT"
    monkeypatch.setattr(imaging,"load_case",forbidden)
    with pytest.raises(ValueError,match="authority records changed"): bridge.reconstruct_pat05(ROOT,records)


def test_semantic_report_keeps_unknowns_and_domain_shift(monkeypatch):
    _,obs,_,_=tiny(monkeypatch)
    report=bridge.describe_observation(obs)
    assert report["channels"][4]["min"] is None
    assert report["channels"][4]["coverage_fraction"] == 0
    assert report["shift"]["generated_horizon"] == 2
    assert report["shift"]["real_historical_horizon"] == 3
    assert report["shift"]["generalization_or_performance_established"] is False


def test_manifest_matches_frozen_four_file_preparation():
    path=ROOT/"manifests/experiments/pat05-forward-diagnostic-v1.json"
    if not path.exists(): pytest.skip("manifest written after final source snapshot")
    import json
    runner.validate(json.loads(path.read_text()))
