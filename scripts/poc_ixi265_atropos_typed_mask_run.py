"""Separately released v3 patient supervisor; default is source-only planning.

The frozen typed-mask kernel repairs only the N4 API representation. This
runner retains frozen v2 positive support, eight UNKNOWN cells, native frame,
parameters, resource caps and unaccepted posterior/class semantics. Preparing
this source does not authorize a patient decode, marker creation, or fit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT/"scripts/poc_ixi265_atropos_typed_mask.py"
KERNEL_SHA = "211d715a05b2dd269358b7bc5585488a053859ba65c04b9f31f60b69e41ec954"
PREPARATION = ROOT/"artifacts/poc-ixi265-atropos-typed-mask-runner-guard-followup-v3"
OUTPUT = ROOT/"outputs/poc-ixi265-atropos-typed-mask-v3"
RUNTIME = ROOT/".tools/poc-antspyx-0.6.3-py312-v1/runtime"
MARKER = "ixi265-n4-atropos-typed-mask-v3-attempt.json"
LINEAGE = {
    "artifacts/poc-ants-typed-mask-native-preflight-v3/outcome.json": "b9b8841d81d1ac9f10303aecdbfa838bba59fe4764065cd7898f44026b129c78",
    "artifacts/ants-mask-dtype-independent-v1/independent-qa.json": "9c4b3bc648b9920ba7de4c5fd0c921b08382796e2ead3467caffc4213ee0c128",
    "artifacts/ixi265-typed-mask-independent-v3/kernel-preflight-qa.json": "08fc33228534fd2adb392a9093b57790695560ab9a881ee96cebddf0a23636b5",
    "artifacts/poc-ants-mask-dtype-generated-v1/outcome.json": "f7a0844d9569ddd8add755c167d316a0c8a7fd840aef76a9ce35ed8674a53102",
    "artifacts/poc-ixi265-atropos-positive-domain-run-v2/outcome.json": "dce67f0acde21aa11fa99bdaa270108c8b53b5a091514182870e69e01fa78e21"}


def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()


def kernel():
    if sha(KERNEL_PATH)!=KERNEL_SHA: raise ValueError("Frozen typed-mask kernel changed")
    spec=importlib.util.spec_from_file_location("frozen_typed_mask_v3",KERNEL_PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def method():
    return {**kernel().method(), "status":"patient_runner_prepared_fit_unreleased",
        "native_kernel_preflight_passed":True,"native_kernel_preflight_outcome_sha256":next(iter(LINEAGE.values())),
        "patient_fitting_released":False,"kernel_sha256":KERNEL_SHA,
        "schema":"resectionlab-ixi265-atropos-typed-mask-patient-runner-v3"}


def canonical_paths(paths):
    expected={"T1":ROOT/"outputs/poc-ixi265/originals/IXI265-T1.nii.gz",
        "brain_mask":ROOT/"outputs/poc-ixi265/synthstrip/brain_mask.nii.gz",
        "host_bundle":ROOT/"outputs/poc-ixi265/IXI265-Guys-0845.ressectionlab"}
    if set(paths)!=set(expected): raise ValueError("Exactly three frozen source paths required")
    # Check every path before any payload is hashed, including failure postchecks.
    for name,path in paths.items():
        supplied=Path(path)
        if supplied!=expected[name] or supplied.resolve()!=expected[name]:
            raise ValueError("Only fixed nonredirected IXI265 paths are admitted")
    return expected


def lineage():
    for relative,pin in LINEAGE.items():
        path=ROOT/relative
        if path.resolve()!=path or sha(path)!=pin: raise ValueError("Changed or redirected prerequisite lineage")
    proof=json.loads((ROOT/next(iter(LINEAGE))).read_text())
    if (proof["status"]!="exact_v3_kernel_native_preflight_passed" or proof["exit_code"]!=0
            or proof["runtime_unchanged"] is not True or proof["patient_inputs_read"] is not False):
        raise ValueError("Successful generated preflight required")
    declaration=ROOT/"artifacts/poc-ants-typed-mask-native-preflight-v3/declaration.json"
    if sha(declaration)!=proof["declaration_sha256"]: raise ValueError("Changed native preflight declaration")
    if json.loads(declaration.read_text())["pins"]["scripts/poc_ixi265_atropos_typed_mask.py"]!=KERNEL_SHA:
        raise ValueError("Native preflight tested another kernel")
    kernel().previous().diagnostic_lineage()
    return dict(LINEAGE)


def preparation_intact():
    document=json.loads((PREPARATION/"preparation.json").read_text())
    rows={r["path"]:r["sha256"] for r in document["frozen_files"]}
    if rows.get("scripts/poc_ixi265_atropos_typed_mask_run.py")!=sha(__file__):
        raise ValueError("Runner is not the frozen prepared source")
    for relative,pin in rows.items():
        path=ROOT/relative
        if not path.is_relative_to(ROOT) or path.resolve()!=path or sha(path)!=pin:
            raise ValueError("Prepared source/control/lineage changed")
    lineage()
    return document


def verify_runtime(base):
    actual=base.runtime_audit(RUNTIME)
    frozen=json.loads((base.PREPARATION/"installed-runtime-audit.json").read_text())
    a,b=dict(actual),dict(frozen)
    a["prefix"]=str(Path(a["prefix"]).resolve());b["prefix"]=str(Path(b["prefix"]).resolve())
    if a!=b: raise ValueError("Frozen standalone runtime changed")
    return actual


def source_continuity(paths, metadata, base):
    """Postcheck never conceals the original worker failure or reads redirected sources."""
    try:
        base.require_admission(metadata);canonical_paths(paths)
    except Exception as error:
        return {"all_unchanged":False,"guard_failure":str(error),"files":{},"payloads_hashed":False}
    expected={"T1":base.T1_SHA,"brain_mask":base.MASK_SHA,"host_bundle":base.HOST_SHA}
    rows={}
    for name,path in paths.items():
        try:
            actual=sha(path);rows[name]={"sha256":actual,"expected_sha256":expected[name],"unchanged":actual==expected[name]}
        except Exception as error: rows[name]={"unchanged":False,"read_failure":str(error)}
    return {"all_unchanged":all(r["unchanged"] for r in rows.values()),"files":rows,"payloads_hashed":True}


def attach_continuity(record,continuity):
    record["source_continuity_after_attempt"]=continuity
    if not continuity["all_unchanged"]:
        if record.get("status")!="failed": record.update(status="failed",failure="Source continuity postcheck failed")
        else: record["additional_source_continuity_failure"]=True


def cleanup_worker(process):
    """Bounded best-effort cleanup; never prevent independent final postchecks."""
    receipt={"normal_exit_races":[],"errors":[]}
    if process is None:
        receipt["worker_started"]=False
        return receipt
    receipt["worker_started"]=True
    def attempt(name,operation):
        try:return operation()
        except ProcessLookupError:
            receipt["normal_exit_races"].append(name)
        except BaseException as error:
            receipt["errors"].append({"operation":name,"type":type(error).__name__,"message":str(error)})
        return None
    if attempt("poll",process.poll) is not None:return receipt
    attempt("SIGTERM",lambda:os.killpg(process.pid,signal.SIGTERM))
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        attempt("SIGKILL",lambda:os.killpg(process.pid,signal.SIGKILL))
        attempt("wait_after_SIGKILL",lambda:process.wait(timeout=5))
    except ProcessLookupError:receipt["normal_exit_races"].append("wait_after_SIGTERM")
    except BaseException as error:
        receipt["errors"].append({"operation":"wait_after_SIGTERM","type":type(error).__name__,"message":str(error)})
    receipt["returncode_after_cleanup"]=getattr(process,"returncode",None)
    return receipt


def worker(spec_path):
    spec=json.loads(Path(spec_path).read_text())
    if spec.get("released_single_attempt") is not True: raise ValueError("Separate v3 patient release required")
    typed=kernel();old=typed.previous();base=old.reference()
    base.require_admission(spec["metadata"]);canonical_paths(spec["paths"])
    if Path(spec["output"])!=OUTPUT or Path(spec["runtime"]).resolve()!=RUNTIME.resolve():
        raise ValueError("Exact v3 output and runtime required")
    preparation_intact()
    if spec["preparation_sha256"]!=sha(PREPARATION/"preparation.json"):
        raise ValueError("Changed prepared patient specification")
    declaration=json.loads((OUTPUT/"declaration.json").read_text())
    if (declaration["runner_sha256"]!=sha(__file__) or declaration["kernel_sha256"]!=KERNEL_SHA
            or declaration["sources"]!={k:str(v) for k,v in spec["paths"].items()}):
        raise ValueError("Changed prospective patient declaration")
    base.bound_inputs(spec["paths"],spec["metadata"]);runtime=verify_runtime(base)
    import nibabel as nib
    import ants
    image,brain_image=(nib.load(spec["paths"][name]) for name in ("T1","brain_mask"))
    if image.shape!=old.EXPECTED_SHAPE: raise ValueError("Changed frozen source shape")
    for item in (image,brain_image):
        q,qc=item.get_qform(coded=True);s,sc=item.get_sform(coded=True)
        if not(qc or sc) or qc and sc and not np.allclose(q,s,rtol=0,atol=base.FRAME_TOLERANCE_MM):
            raise ValueError("Missing/conflicting source NIfTI frame")
    mask_frame=base.require_native_frame(image.shape,image.affine,brain_image.shape,**base.ras_to_lps_geometry(brain_image.affine))
    values=np.asarray(image.dataobj,dtype=np.float32);binary=np.asarray(brain_image.dataobj)
    if not np.isin(binary,[0,1]).all():raise ValueError("Binary original brain mask required")
    brain=binary.astype(bool)
    if int(brain.sum())!=old.EXPECTED_BRAIN_VOXELS:raise ValueError("Changed original brain support")
    domain,unknown,support=old.positive_domain(values,brain)
    stages=[]
    def stage(name,details):
        stages.append({"name":name,"details":details})
        base.write_json(OUTPUT/"stages.json",{"worker_pid":os.getpid(),"support":support,"stages":stages})
    corrected,labels,posteriors,audit=typed.fit_arrays(values,domain,image.affine,ants,stage_callback=stage)
    old.require_unknown_partition(brain,domain,unknown,labels,posteriors)
    rows=[("n4_corrected_estimated",corrected,"bias_corrected_intensity_not_acquired_MRI"),
        ("class_labels_raw",labels,"unmodified_fitted_class_ids_on_positive_intensity_domain"),
        ("fit_domain_native",domain.astype(np.uint8),"declared_fitting_support_not_replacement_brain"),
        ("excluded_unknown_native",unknown.astype(np.uint8),"UNKNOWN_ABSTAINED_missing_intensity_not_CSF")]
    rows.extend((f"class_{i}_posterior",p,"unmodified_membership_probability_not_PVE") for i,p in enumerate(posteriors,1))
    rows.extend((name,(labels==code).astype(np.uint8),"hard_class_under_declared_T1_intensity_assumption") for code,name in base.ASSUMPTIONS.items())
    outputs=[]
    for name,data,semantics in rows:
        header=image.header.copy();header.set_data_dtype(data.dtype)
        result=nib.Nifti1Image(data,image.affine,header)
        result.set_qform(image.get_qform(),int(image.header["qform_code"]))
        result.set_sform(image.get_sform(),int(image.header["sform_code"]))
        path=OUTPUT/(name+".nii.gz");nib.save(result,path);reopened=nib.load(path)
        base.require_native_frame(image.shape,image.affine,reopened.shape,**base.ras_to_lps_geometry(reopened.affine))
        if not np.array_equal(np.asarray(reopened.dataobj),data): raise ValueError("Saved candidate array changed")
        outputs.append({"name":name,"sha256":sha(path),"bytes":path.stat().st_size,"semantics":semantics,"native_shape":list(data.shape)})
    base.bound_inputs(spec["paths"],spec["metadata"])
    base.write_json(OUTPUT/"candidate.json",{**method(),"status":"fitted_candidate_unaccepted","patient_fitting_released":True,"runner_sha256":sha(__file__),
        "runtime":runtime,"outputs":outputs,"support":support,"numerical_audit":audit,"source_mask_frame_check":mask_frame,
        "source_bytes_unchanged":True,"lineage":LINEAGE,"anatomy_accepted":False,"host_accepted":False,"growth_use_accepted":False})


def run(paths,metadata,runtime,output,*,released_single_attempt=False):
    if not released_single_attempt:raise ValueError("V3 patient fitting is not released")
    typed=kernel();base=typed.previous().reference()
    base.require_admission(metadata);canonical_paths(paths)
    if Path(runtime).resolve()!=RUNTIME.resolve() or Path(output)!=OUTPUT or OUTPUT.resolve()!=OUTPUT:
        raise ValueError("Exact runtime and new nonredirected v3 output required")
    if OUTPUT.exists():raise ValueError("New v3 output only; no retry/overwrite")
    preparation=preparation_intact();runtime_receipt=verify_runtime(base)
    import psutil
    peak=base.sampled_rss(psutil);minimum=psutil.virtual_memory().available;disk=psutil.disk_usage(OUTPUT.parent).free
    reason=base.resource_violation(peak,minimum,0,0,disk)
    if reason:raise ValueError(reason)
    base.bound_inputs(paths,metadata)
    retained_markers={name:sha(RUNTIME/name) for name in ("ixi265-n4-atropos-single-attempt.json","ixi265-n4-atropos-positive-domain-v2-attempt.json")}
    declaration={**method(),"status":"declared_before_source_decode","released_single_attempt":True,"patient_fitting_released":True,
        "created_utc":datetime.now(timezone.utc).isoformat(),"runner_sha256":sha(__file__),"kernel_sha256":KERNEL_SHA,
        "preparation_sha256":sha(PREPARATION/"preparation.json"),"frozen_preparation_files":preparation["frozen_files"],
        "sources":{k:str(v) for k,v in paths.items()},"metadata":metadata,"runtime":runtime_receipt,
        "lineage":LINEAGE,"retained_historical_markers":retained_markers,"attempts":1,"automatic_retry_or_tuning":False}
    with (RUNTIME/MARKER).open("x") as stream:json.dump(declaration,stream,indent=2)
    OUTPUT.mkdir();temporary=OUTPUT/"temporary";temporary.mkdir();base.write_json(OUTPUT/"declaration.json",declaration)
    spec={"released_single_attempt":True,"paths":{k:str(v) for k,v in paths.items()},"metadata":metadata,
        "runtime":str(RUNTIME),"output":str(OUTPUT),"preparation_sha256":sha(PREPARATION/"preparation.json")}
    base.write_json(OUTPUT/"worker-input.json",spec)
    record={"schema":"resectionlab-typed-mask-patient-attempt-v3","status":"running","started_utc":datetime.now(timezone.utc).isoformat(),
        "supervisor_pid":os.getpid(),"runner_sha256":sha(__file__),"kernel_sha256":KERNEL_SHA,"resources":base.plan()["resources"],
        "memory_scope":"supervising_Python_plus_worker_and_recursive_descendants"}
    base.write_json(OUTPUT/"attempt.json",record)
    env={**os.environ,"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS":"2","OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2",
        "VECLIB_MAXIMUM_THREADS":"2","MKL_NUM_THREADS":"2","NUMEXPR_NUM_THREADS":"2","PYTHONHASHSEED":"0",
        "PYTHONNOUSERSITE":"1","TMPDIR":str(temporary),"MPLCONFIGDIR":str(temporary/"matplotlib")}
    env.pop("PYTHONPATH",None)
    process=None;started=time.monotonic()
    try:
        with (OUTPUT/"worker.log").open("x") as log:
            process=subprocess.Popen([str(RUNTIME/"bin/python"),"-I",str(Path(__file__).resolve()),"--worker",str(OUTPUT/"worker-input.json")],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record["worker_pid"]=process.pid
            while process.poll() is None:
                try:peak=max(peak,base.sampled_rss(psutil,process.pid))
                except psutil.NoSuchProcess:pass
                minimum=min(minimum,psutil.virtual_memory().available);disk=min(disk,psutil.disk_usage(OUTPUT).free)
                reason=base.resource_violation(peak,minimum,time.monotonic()-started,base.directory_bytes(OUTPUT),disk)
                if reason:raise RuntimeError(reason)
                time.sleep(.2)
        reason=base.resource_violation(peak,minimum,time.monotonic()-started,base.directory_bytes(OUTPUT),disk)
        if reason or process.returncode:raise RuntimeError(reason or "worker_failed_see_retained_log")
        if not(OUTPUT/"candidate.json").is_file():raise RuntimeError("Missing candidate receipt")
        record["status"]="fitted_candidate_unaccepted"
    except BaseException as error:
        record.update(status="failed",failure=str(error))
        try:record["cleanup"]=cleanup_worker(process)
        except BaseException as cleanup_error:
            record["cleanup"]={"errors":[{"operation":"cleanup_worker","type":type(cleanup_error).__name__,"message":str(cleanup_error)}]}
    # Runs after success or failure; does not replace the original failure.
    attach_continuity(record,source_continuity(paths,metadata,base))
    try:
        record["historical_markers_unchanged"]={name:sha(RUNTIME/name)==pin for name,pin in retained_markers.items()}
        record["runtime_unchanged"]=verify_runtime(base)==runtime_receipt
        record["lineage_unchanged"]=lineage()==LINEAGE
        if not all(record["historical_markers_unchanged"].values()):raise ValueError("Historical marker changed")
        if record["runtime_unchanged"] is not True:raise ValueError("Runtime continuity comparison failed")
        if record["lineage_unchanged"] is not True:raise ValueError("Lineage continuity comparison failed")
    except Exception as error:
        record["continuity_guard_error"]=str(error)
        if record["status"]!="failed":record.update(status="failed",failure="Runtime/history continuity guard failed")
    record.update(exit_code=None if process is None else process.returncode,elapsed_seconds=time.monotonic()-started,
        peak_supervisor_and_process_family_rss_bytes_sampled=peak,minimum_available_memory_bytes_sampled=minimum,
        minimum_free_disk_bytes_sampled=disk,output_and_temporary_bytes=base.directory_bytes(OUTPUT),
        finished_utc=datetime.now(timezone.utc).isoformat(),declaration_sha256=sha(OUTPUT/"declaration.json"),
        sampling_limitation="Between-sample peaks are not a hard memory reservation",
        anatomy_accepted=False,host_accepted=False,growth_use_accepted=False,
        manifest=[{"path":str(p.relative_to(OUTPUT)),"sha256":sha(p),"bytes":p.stat().st_size}
            for p in sorted(OUTPUT.rglob("*")) if p.is_file() and p.name!="attempt.json"])
    base.write_json(OUTPUT/"attempt.json",record)
    return record


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",type=Path)
    parser.add_argument("--released-single-attempt-spec",type=Path)
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.released_single_attempt_spec:
        spec=json.loads(args.released_single_attempt_spec.read_text())
        if spec.pop("released_single_attempt",None) is not True:raise ValueError("Separate v3 release required")
        result=run(**spec,released_single_attempt=True);print(json.dumps(result,indent=2))
        raise SystemExit(0 if result["status"]=="fitted_candidate_unaccepted" else 1)
    else:print(json.dumps(method(),indent=2))
