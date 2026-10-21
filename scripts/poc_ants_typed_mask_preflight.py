"""One released native preflight of the frozen v3 kernel on generated arrays."""
from datetime import datetime, timezone
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "scripts/poc_ixi265_atropos_typed_mask.py": "211d715a05b2dd269358b7bc5585488a053859ba65c04b9f31f60b69e41ec954",
    "scripts/poc_ants_mask_dtype_diagnostic.py": "1feb74ffca22a14d226bc9b52b0b5049a607225e95bb8dee6d11df5151aa003c"}
OUTPUT = ROOT/"outputs/poc-ants-typed-mask-native-preflight-v3"
ARTIFACT = ROOT/"artifacts/poc-ants-typed-mask-native-preflight-v3"


def sha(path):
    import hashlib
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()


def load(relative):
    if sha(ROOT/relative) != PINS[relative]: raise ValueError("Frozen preflight dependency changed")
    spec=importlib.util.spec_from_file_location(Path(relative).stem,ROOT/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def dependencies():
    kernel=load("scripts/poc_ixi265_atropos_typed_mask.py")
    probe=load("scripts/poc_ants_mask_dtype_diagnostic.py")
    return kernel,probe,probe.reference()


def worker():
    kernel,probe,base=dependencies()
    declaration=json.loads((OUTPUT/"declaration.json").read_text())
    if (declaration["released_native_generated_preflight"] is not True
            or declaration["runner_sha256"]!=sha(__file__) or declaration["pins"]!=PINS
            or declaration["generated_input"]!=probe.generated_record()
            or declaration["resources"]!=probe.RESOURCES):
        raise ValueError("Changed preflight declaration")
    actual=base.runtime_audit(probe.RUNTIME)
    if not probe.same_runtime(actual,declaration["runtime"]): raise ValueError("Changed runtime")
    import numpy as np
    import ants
    values,mask,truth,affine=probe.phantom()
    originals=[probe.array_record(a) for a in (values,mask,truth,affine)]
    stages=[]
    def stage(name,details):
        stages.append({"name":name,"details":details})
        base.write_json(OUTPUT/"stages.json",{"worker_pid":os.getpid(),"stages":stages})
    corrected,labels,posteriors,audit=kernel.fit_arrays(values,mask,affine,ants,stage_callback=stage)
    if originals != [probe.array_record(a) for a in (values,mask,truth,affine)]:
        raise ValueError("Generated input mutation")
    if not np.array_equal(corrected[~mask],values[~mask]): raise ValueError("Outside-mask change")
    if not np.array_equal(labels,truth): raise ValueError("Generated class truth mismatch")
    rows=[("corrected_raw",corrected),("classes_raw",labels),*[(f"posterior_{i}_raw",p) for i,p in enumerate(posteriors,1)]]
    for name,array in rows: np.save(OUTPUT/(name+".npy"),array,allow_pickle=False)
    base.write_json(OUTPUT/"result.json",{"status":"exact_v3_kernel_native_preflight_passed",
        "kernel_sha256":PINS["scripts/poc_ixi265_atropos_typed_mask.py"],"generated_input":probe.generated_record(),
        "generated_truth_agreement":1.,"original_generated_arrays_unchanged":True,"audit":audit,
        "raw_outputs":[{"name":n,"array":probe.array_record(a)} for n,a in rows],
        "patient_sources_read":False,"clinical_validity_inferred":False,"runtime":actual})


def run(*,released=False):
    if not released: raise ValueError("Explicit native generated preflight release required")
    if OUTPUT.exists() or ARTIFACT.exists(): raise ValueError("New single-attempt preflight only; no retry")
    kernel,probe,base=dependencies()
    runtime=base.runtime_audit(probe.RUNTIME)
    if not probe.same_runtime(runtime,json.loads((base.PREPARATION/"installed-runtime-audit.json").read_text())):
        raise ValueError("Frozen runtime changed")
    import psutil
    peak=base.sampled_rss(psutil);minimum=psutil.virtual_memory().available;disk=psutil.disk_usage(OUTPUT.parent).free
    reason=probe.violation(peak,minimum,0,0,disk)
    if reason: raise ValueError(reason)
    OUTPUT.mkdir();ARTIFACT.mkdir();temporary=OUTPUT/"temporary";temporary.mkdir()
    declaration={"schema":"resectionlab-exact-typed-mask-kernel-native-preflight-v3",
        "released_native_generated_preflight":True,"created_utc":datetime.now(timezone.utc).isoformat(),
        "runner_sha256":sha(__file__),"pins":PINS,"runtime":runtime,
        "generated_input":probe.generated_record(),"resources":probe.RESOURCES,
        "parameters":kernel.method()["parameters"],"attempts":1,"patient_fitting_released":False}
    base.write_json(OUTPUT/"declaration.json",declaration);base.write_json(ARTIFACT/"declaration.json",declaration)
    env={**os.environ,"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS":"2","OMP_NUM_THREADS":"2",
        "OPENBLAS_NUM_THREADS":"2","VECLIB_MAXIMUM_THREADS":"2","MKL_NUM_THREADS":"2","NUMEXPR_NUM_THREADS":"2",
        "PYTHONHASHSEED":"0","PYTHONNOUSERSITE":"1","TMPDIR":str(temporary),"MPLCONFIGDIR":str(temporary/"matplotlib")}
    env.pop("PYTHONPATH",None)
    started=time.monotonic();process=None
    record={"started_utc":datetime.now(timezone.utc).isoformat(),"supervisor_pid":os.getpid(),"resources":probe.RESOURCES}
    try:
        with (OUTPUT/"worker.log").open("x") as log:
            process=subprocess.Popen([str(probe.RUNTIME/"bin/python"),"-I",str(Path(__file__).resolve()),"--worker"],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record["worker_pid"]=process.pid
            while process.poll() is None:
                try: peak=max(peak,base.sampled_rss(psutil,process.pid))
                except psutil.NoSuchProcess: pass
                minimum=min(minimum,psutil.virtual_memory().available);disk=min(disk,psutil.disk_usage(OUTPUT).free)
                reason=probe.violation(peak,minimum,time.monotonic()-started,base.directory_bytes(OUTPUT),disk)
                if reason: raise RuntimeError(reason)
                time.sleep(.1)
        reason=probe.violation(peak,minimum,time.monotonic()-started,base.directory_bytes(OUTPUT),disk)
        if reason or process.returncode: raise RuntimeError(reason or "worker_failed_see_retained_log")
        if not (OUTPUT/"result.json").is_file(): raise RuntimeError("Missing preflight result")
        record["status"]="exact_v3_kernel_native_preflight_passed"
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        record.update(status="failed",failure=str(error))
    record.update(exit_code=None if process is None else process.returncode,elapsed_seconds=time.monotonic()-started,
        peak_supervisor_and_process_family_rss_bytes_sampled=peak,minimum_available_memory_bytes_sampled=minimum,
        minimum_free_disk_bytes_sampled=disk,output_and_temporary_bytes=base.directory_bytes(OUTPUT),
        finished_utc=datetime.now(timezone.utc).isoformat(),declaration_sha256=sha(OUTPUT/"declaration.json"),
        runtime_unchanged=probe.same_runtime(base.runtime_audit(probe.RUNTIME),runtime),patient_inputs_read=False,
        sampling_limitation="Between-sample peaks are not a hard memory reservation",
        manifest=[{"path":str(p.relative_to(OUTPUT)),"sha256":sha(p),"bytes":p.stat().st_size}
            for p in sorted(OUTPUT.rglob("*")) if p.is_file()])
    base.write_json(OUTPUT/"outcome.json",record);base.write_json(ARTIFACT/"outcome.json",record)
    for name in ["result.json","stages.json","worker.log"]:
        if (OUTPUT/name).is_file(): (ARTIFACT/name).write_bytes((OUTPUT/name).read_bytes())
    return record


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--released-native-preflight",action="store_true")
    args=parser.parse_args()
    if args.worker:worker()
    elif args.released_native_preflight:print(json.dumps(run(released=True),indent=2))
    else:print(json.dumps({"released":False,"pins":PINS}))
