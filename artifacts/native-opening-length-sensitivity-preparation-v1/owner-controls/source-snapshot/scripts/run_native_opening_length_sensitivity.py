#!/usr/bin/env python3
"""Two frozen forwards after a single uncertified working-length descriptor change.

The original generated source/DTO must reconstruct exactly. Modified descriptors
do not define new physical tools or a newly complete/certified action inventory.
Saved baseline scores require no new baseline calls. No patient input is read.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from resectionlab import pat05_forward_diagnostic as shared
import run_pat05_forward_diagnostic as infrastructure
from preflight_real_spatial_policy import supervise_worker, write_json, sha256

VERSION = "native-opening-length-sensitivity-v1"
SCRIPT = "scripts/run_native_opening_length_sensitivity.py"
PROPOSAL_SHA256 = "6daf836532a40387e8ded69bebbc9b5a455b32778eeca487d749ccec88f5c2c5"
ORIGINAL = "artifacts/native-opening-learning-v1/"
RL = "artifacts/native-opening-rl-capacity-v1/"
SOURCE_HASH = "sha256:9b39c31e1e012b0822d538451a0452fa9c564acc863d22dabb311ec4cf8c5d4e"
ROOT_OBSERVATION_HASH = "sha256:c9458a99068bfaae90ba69062073a674c55ee31ad66d11a5c86461ef2c834eba"
MODEL_HASH = "sha256:f607b51b66d00164c33ff9c8fa7725b0e6e15bab1b566d161efbe3fb4f4e730b"
INPUTS = {
    ORIGINAL + "declaration-input.json": "963238d34dce36f4a9128c782f0d3ca667640123a58a2eb2498abb01d68c5212",
    ORIGINAL + "preparation.json": "63181c0754443730925e758019007ff472d58b9db1b72d9e3aacf9a28d8a3884",
    RL + "result.json": "301e690fb87adff99b715c1371cb0fc692b0adc16fc3576ff7b0725336c56f68",
    RL + "declaration-input.json": "14df1a1fbb9bb8187d97bc7dcf99cacf923a9e5e9f4182693de0e1d01a5e1285",
    RL + "output-sha256.json": "d378524c205838c28f09b2bc02e2a1e08dadc9a25acd63a0170043afc3aeea3d",
    RL + "summary.json": "f71c9d5e2c7e28692fefed904e57240c23f0fb00b7725a1fb83086bf1f4c78e8",
    RL + "independent-verification.json": "2489aff1b8a783f1fd6ad97a1085abb4b54aff30b44c337deb42a654d5afc95f",
}
SETTINGS = {"max_wall_seconds": 20., "max_rss_bytes": 1024**3, "cpu_threads": 1,
    "forward_order": ["initial", "RL256"], "new_forwards_per_checkpoint": 1,
    "new_baseline_forwards": 0, "working_length_descriptor_mm": 120.,
    "native_previews": 0, "proposer_calls": 0, "executed_actions": 0,
    "optimizer_updates": 0, "patient_inputs": 0, "automatic_retry": False}


def sources():
    # Conservative existing local-import closure covers the reused DTO,
    # checkpoint idiom and supervisor. No patient-authority function is called.
    paths = set(infrastructure.dependency_paths()) | {SCRIPT}
    return {p: sha256(ROOT / p) for p in sorted(paths)}


def specification():
    return {"version": VERSION, "approved_proposal_sha256": PROPOSAL_SHA256,
        "settings": SETTINGS, "input_sha256": INPUTS, "checkpoints": shared.CHECKPOINTS,
        "architecture_hash": shared.ARCHITECTURE, "original_source_hash": SOURCE_HASH,
        "original_observation_hash": ROOT_OBSERVATION_HASH, "original_task_model_hash": MODEL_HASH,
        "intervention": "replace nonSTOP action_geometry[:,12] working_length_mm with120 only; STOP row unchanged",
        "unchanged": ["images", "coverage", "availability", "affine", "spacing", "state", "action_ids", "action_order",
            "action_mask", "action_tool_ids", "provenance", "entry_tip_rays", "all_other_geometry", "weights", "architecture"],
        "input_identity": "new whole DTO fingerprint; inherited source_id records original ancestry only, not certification of modified descriptors",
        "primary": "change in STOP logit minus maximum nonSTOP logit against saved original root, per checkpoint",
        "secondary": ["STOP probability", "per-action logit/probability changes", "entropy", "highest-ranked ID not executed"],
        "baseline": "original audited RL capacity saved root readouts at updates0/256; no baseline forwards",
        "scope": "generated fixed-input model sensitivity; uncertified descriptor intervention, not changed physical legal actions",
        "prohibited": ["engine", "proposer", "task", "step", "search", "optimizer", "loss", "backward", "patient_reads"],
        "clinical_correctness_measured": False, "planning_efficacy_measured": False,
        "failure": "retain two slots, attempted/completed call counts and null partial outputs; no retry or cap extension"}


def declaration():
    return {**specification(), "source_sha256": sources(), "runtime": infrastructure.declaration()["runtime"]}


def read_inputs():
    """Generated-experiment JSON only; never the imported helper's PAT05 inputs."""
    records = {p: json.loads(shared.checked_bytes(ROOT, p, h)) for p,h in INPUTS.items()}
    prep, result = records[ORIGINAL+"preparation.json"], records[RL+"result.json"]
    if (prep["source_hash"] != SOURCE_HASH or prep["initial_observation_hash"] != ROOT_OBSERVATION_HASH
            or prep["decision_model_hash"] != MODEL_HASH or result["status"] != "complete"
            or result["updates"] != 256 or result["architecture_hash"] != shared.ARCHITECTURE):
        raise ValueError("Original generated source/checkpoint experiment differs")
    baselines = {}
    index, review = records[RL+"output-sha256.json"], records[RL+"independent-verification.json"]
    for name, spec in shared.CHECKPOINTS.items():
        update = str(spec["updates"])
        states = [r for r in result["readouts"][update]["states"] if r["observation_hash"] == ROOT_OBSERVATION_HASH]
        if (len(states) != 1 or index[Path(spec["path"]).name] != spec["file_sha256"]
                or review["checkpoints"][update]["tensor_sha256"] != spec["parameter_hash"]
                or result["readouts"][update]["parameter_hash"] != spec["parameter_hash"]):
            raise ValueError("Original fixed baseline/independent checkpoint binding differs")
        baselines[name] = states[0]
    if baselines["initial"]["action_ids"] != baselines["RL256"]["action_ids"]:
        raise ValueError("Saved original root action order differs")
    return records, baselines


def validate(record):
    if shared.canonical(record) != shared.canonical(declaration()):
        raise ValueError("Fixed source, input, runtime, settings or checkpoint declaration changed")
    infrastructure.assert_imports(record)
    if Path(__file__).resolve() != (ROOT / SCRIPT).resolve():
        raise ValueError("Executing runner outside frozen source root")
    return read_inputs()


def reconstruct_root(records, baselines):
    """Recreate the historical generated arrays and recorded geometry, no engine."""
    from resectionlab.geometry import AccessWindow
    from resectionlab.native_spatial_task import NativeSpatialCase, OPENING_TOOLS, SYNTHETIC_TARGET_THRESHOLD
    support = np.zeros((9,9,7), bool)
    support[4,4,1:6], support[5,5,1] = True, True
    target = np.zeros(support.shape, bool); target[4,4,4:6] = True
    scan = np.where(target,.8,np.where(support,.2,0.)).astype(np.float32)
    source = NativeSpatialCase(scan,support,target,np.eye(4),
        AccessWindow((3.9,4.,.5),(0.,0.,1.),2.4,"analytic-offset-opening"),OPENING_TOOLS,
        track="synthetic_scan",support_source_kind="derived_from_scan",
        support_derivation="unit-fixture forward model: nonzero structural signal identifies support",
        nominal_target=scan >= SYNTHETIC_TARGET_THRESHOLD,target_source_kind="derived_from_scan",
        target_derivation="unit-fixture .2/.8 signal midpoint .5; never an MRI estimator")
    prep=records[ORIGINAL+"preparation.json"]
    if source.source_hash != SOURCE_HASH or source.reference_hash != prep["reference_hash"]:
        raise ValueError("Generated original source changed before any forward")
    first={"step":0,"action_ids":baselines["initial"]["action_ids"],
        "action_mask":[True]*5,"observation_hash":ROOT_OBSERVATION_HASH}
    observation=shared._rebuild_saved_observation(source,prep["inventory"],first,max_steps=2)
    require_original(observation)
    return observation


def require_original(observation):
    from resectionlab.spatial_observations import SpatialObservation
    if type(observation) is not SpatialObservation:
        raise TypeError("Exact immutable generated observation required")
    observation.assert_intact()
    if (observation.source_id != SOURCE_HASH or observation.fingerprint != ROOT_OBSERVATION_HASH
            or observation.track != "synthetic_scan" or observation.image_channels.shape != (6,9,9,7)
            or observation.state_features[0] != 0 or observation.state_features[1] != 2
            or len(observation.action_ids) != 5):
        raise ValueError("Only the exact original generated root is admitted")


def intervene(original):
    require_original(original)
    geometry=np.array(original.action_geometry,copy=True)
    geometry[1:,12]=SETTINGS["working_length_descriptor_mm"]
    changed=replace(original,action_geometry=geometry)
    validate_intervention(original,changed)
    return changed


def validate_intervention(original,changed):
    require_original(original)
    if type(changed) is not type(original):
        raise TypeError("Intervention must retain the exact validated DTO type")
    changed.assert_intact()
    for field in ("image_channels","coverage","channel_available","affine_ras_mm","spacing_mm","state_features","action_mask"):
        a,b=getattr(original,field),getattr(changed,field)
        if a.dtype != b.dtype or not np.array_equal(a,b):
            raise ValueError("An undeclared observation field changed: "+field)
    for field in ("source_id","track","action_ids","action_tool_ids","channel_provenance"):
        if getattr(original,field) != getattr(changed,field):
            raise ValueError("An undeclared observation identity changed: "+field)
    expected=np.array(original.action_geometry,copy=True); expected[1:,12]=120.
    if changed.action_geometry.dtype != original.action_geometry.dtype or not np.array_equal(changed.action_geometry,expected):
        raise ValueError("Only nonSTOP working-length descriptors may change to120")
    if changed.fingerprint == original.fingerprint:
        raise ValueError("Descriptor intervention requires its own distinct DTO identity")


def load_frozen_checkpoint(name):
    """Reuse the narrow exact-byte/weights-only idiom; no optimizer is created."""
    import torch
    from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig, parameter_hash
    spec=shared.CHECKPOINTS[name]
    payload=shared.checked_bytes(ROOT,spec["path"],spec["file_sha256"])
    unsafe=set(torch.serialization.get_unsafe_globals_in_checkpoint(io.BytesIO(payload)))
    if unsafe-{"numpy._core.multiarray.scalar","numpy.dtype"}:
        raise ValueError("Unexpected globals in exact local checkpoint")
    with torch.serialization.safe_globals([np._core.multiarray.scalar,np.dtype,type(np.dtype("f8"))]):
        checkpoint=torch.load(io.BytesIO(payload),map_location="cpu",weights_only=True)
    if (checkpoint["updates"] != spec["updates"] or checkpoint["parameter_hash"] != spec["parameter_hash"]
            or checkpoint["initial_parameter_hash"] != shared.CHECKPOINTS["initial"]["parameter_hash"]
            or checkpoint["context"]["declaration_sha256"] != INPUTS[RL+"declaration-input.json"]
            or checkpoint["context"]["source_ids"] != (SOURCE_HASH,)
            or checkpoint["context"]["decision_model_hash"] != MODEL_HASH
            or checkpoint["context"]["max_steps"] != 2):
        raise ValueError("Frozen generated checkpoint ancestry differs")
    with torch.random.fork_rng(devices=[]):
        model=SpatialPolicy(SpatialPolicyConfig(**checkpoint["architecture"]["config"]))
    model.load_state_dict(checkpoint["policy"],strict=True); model.eval().requires_grad_(False)
    if (parameter_hash(model) != spec["parameter_hash"] or model.architecture_hash != shared.ARCHITECTURE
            or shared.canonical(model.architecture_record()) != shared.canonical(checkpoint["architecture"])):
        raise ValueError("Frozen tensor/architecture identity differs")
    return model


def contrast(logits,probabilities,value,baseline,action_ids):
    if (baseline["action_ids"] != list(action_ids) or len(logits) != 5 or len(probabilities) != 5
            or not all(math.isfinite(x) for x in [*logits,*probabilities,value])
            or any(p<0 or p>1 for p in probabilities) or not math.isclose(math.fsum(probabilities),1.,abs_tol=2e-6)):
        raise ValueError("Output/baseline action/probability binding differs")
    margin=logits[0]-max(logits[1:]); old=baseline["logits"][0]-max(baseline["logits"][1:])
    return {"logits":logits,"probabilities":probabilities,"value_uncalibrated":value,
        "stop_minus_best_nonstop_logit":margin,"saved_original_margin":old,"margin_change":margin-old,
        "stop_probability":probabilities[0],"saved_original_stop_probability":baseline["probabilities"][0],
        "stop_probability_change":probabilities[0]-baseline["probabilities"][0],
        "per_action_logit_change":[x-y for x,y in zip(logits,baseline["logits"])],
        "per_action_probability_change":[x-y for x,y in zip(probabilities,baseline["probabilities"])],
        "stop_logit_change":logits[0]-baseline["logits"][0],
        "stop_head_invariant_within_1e_minus6":abs(logits[0]-baseline["logits"][0])<=1e-6,
        "entropy":-math.fsum(p*math.log(p) for p in probabilities if p),
        "highest_ranked_id_not_executed":action_ids[max(range(5),key=logits.__getitem__)],
        "action_ids":list(action_ids),"comparison_scope":"descriptor sensitivity only; no recertified physical actions or outcomes"}


def forward_once(name,original,changed,baseline,ledger):
    import torch
    from resectionlab.spatial_policy import parameter_hash
    validate_intervention(original,changed)
    attempts=ledger["checkpoint_attempts"]
    if len(attempts)>=2 or name!=SETTINGS["forward_order"][len(attempts)]:
        raise ValueError("Only one attempt per fixed checkpoint in declared order; no retry")
    ledger["checkpoint_attempts"].append(name)
    model=load_frozen_checkpoint(name); spec=shared.CHECKPOINTS[name]
    validate_intervention(original,changed)
    ledger["forward_attempts"].append(name)
    began=time.perf_counter()
    with torch.inference_mode():
        logits,value=model(changed); probabilities=logits.softmax(-1)
    seconds=time.perf_counter()-began
    validate_intervention(original,changed)
    if (parameter_hash(model) != spec["parameter_hash"] or model.architecture_hash != shared.ARCHITECTURE
            or any(p.requires_grad or p.grad is not None for p in model.parameters())):
        raise RuntimeError("Frozen forward changed policy or gradient state")
    shared.checked_bytes(ROOT,spec["path"],spec["file_sha256"])
    row=contrast(logits.tolist(),probabilities.tolist(),value.item(),baseline,changed.action_ids)
    row.update(status="complete",forward_seconds=seconds,checkpoint=spec,
        architecture_hash=model.architecture_hash,original_observation_hash=original.fingerprint,
        intervention_observation_hash=changed.fingerprint,weights_unchanged=True)
    ledger["completed_forwards"].append(name)
    return row


def worker(record,output,declaration_sha256):
    import torch
    started=time.perf_counter(); torch.set_num_threads(1)
    ledger={"checkpoint_attempts":[],"forward_attempts":[],"completed_forwards":[]}
    result={"version":VERSION,"status":"running","declaration_sha256":declaration_sha256,
        "methods":{n:{"status":"not_started","outputs":None} for n in SETTINGS["forward_order"]},
        "new_baseline_forwards":0,"native_previews":0,"executed_actions":0,"optimizer_updates":0,"patient_inputs":0,
        "scope":specification()["scope"]}
    def guard():
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)
        if time.perf_counter()-started>=SETTINGS["max_wall_seconds"] or peak>SETTINGS["max_rss_bytes"]:
            raise InterruptedError("Fixed20s/1GiB diagnostic resource envelope exhausted")
        return peak
    def save():
        result.update(elapsed_seconds=time.perf_counter()-started,ledger=ledger)
        write_json(output/"result.json",result)
    save()
    try:
        records,baselines=validate(record); guard(); began=time.perf_counter()
        original=reconstruct_root(records,baselines); changed=intervene(original)
        result["reconstruction_seconds"]=time.perf_counter()-began
        write_json(output/"inputs.json",{"original_source_hash":SOURCE_HASH,"original_task_model_hash":MODEL_HASH,
            "original_observation_hash":original.fingerprint,"intervention_observation_hash":changed.fingerprint,
            "action_ids":list(original.action_ids),"action_mask":original.action_mask.tolist(),
            "original_action_geometry":original.action_geometry.tolist(),"intervention_action_geometry":changed.action_geometry.tolist(),
            "state_features":original.state_features.tolist(),"only_changed_column":12,
            "source_id_interpretation":"unchanged original-source ancestry; not modified physical-tool certification",
            "saved_baselines":baselines})
        for name in SETTINGS["forward_order"]:
            guard(); validate(record)
            result["methods"][name]["status"]="running"; save(); began=time.perf_counter()
            row=forward_once(name,original,changed,baselines[name],ledger)
            guard(); validate(record)
            row["load_and_forward_seconds"]=time.perf_counter()-began
            write_json(output/(name+".json"),row)
            result["methods"][name]={"status":"complete","outputs":name+".json",
                "margin_change":row["margin_change"],"stop_probability":row["stop_probability"],
                "stop_logit_change":row["stop_logit_change"],"highest_ranked_id_not_executed":row["highest_ranked_id_not_executed"]}
            save()
        if ledger["completed_forwards"] != SETTINGS["forward_order"]:
            raise ValueError("Both fixed calls must complete")
        result["peak_worker_rss_bytes"]=guard();result["status"]="complete"
    except BaseException as error:
        result["status"]="failed";result["error"]={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}
        raise
    finally:
        save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration",type=Path,required=True);parser.add_argument("--output",type=Path)
    parser.add_argument("--execute",action="store_true");parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--expected-declaration-sha256",help=argparse.SUPPRESS)
    args=parser.parse_args();raw=args.declaration.read_bytes();identity=shared.digest_bytes(raw);record=json.loads(raw)
    if args.expected_declaration_sha256 and identity!=args.expected_declaration_sha256:
        raise ValueError("Supervisor-bound declaration bytes changed")
    validate(record)
    if args.worker:
        if not args.expected_declaration_sha256 or args.output is None: raise ValueError("Worker requires frozen identity/output")
        worker(record,args.output,identity);return
    if not args.execute:
        print(json.dumps({"status":"metadata_preflight_passed","declaration_sha256":identity,"new_forwards":0,"patient_inputs":0}));return
    if args.output is None or args.output.exists(): raise ValueError("Use a fresh output directory; preserve earlier attempts")
    args.output.mkdir(parents=True);(args.output/"declaration-input.json").write_bytes(raw)
    for folder,inventory in (("source-snapshot",record["source_sha256"]),("input-metadata",INPUTS)):
        for relative,expected in inventory.items():
            target=args.output/folder/relative;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(shared.checked_bytes(ROOT,relative,expected))
    command=[sys.executable,str(ROOT/SCRIPT),"--declaration",str(args.output.resolve()/"declaration-input.json"),
        "--output",str(args.output.resolve()),"--worker","--expected-declaration-sha256",identity]
    status=supervise_worker(command,args.output,SETTINGS,identity)
    validate(record);infrastructure.write_output_index(args.output)
    if status["status"]!="complete": raise SystemExit(1)


if __name__=="__main__":
    main()
