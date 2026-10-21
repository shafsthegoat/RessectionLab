"""Prepare a separate, display-only IXI265 tissue-estimate workspace.

No fitting, resampling, threshold tuning, source replacement, catalog update,
anatomical acceptance, or planning admission occurs. Default execution only
prints the prospective plan. Actual packaging requires a separate release.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PREPARATION=ROOT/"artifacts/ixi265-tissue-display-preparation-v1"
HOST=ROOT/"outputs/poc-ixi265/IXI265-Guys-0845.ressectionlab"
HOST_SHA="c735daaf6d60f5f0d1567760a2accc18d4bd4838a096d8e0b5f490ee88abaff1"
T1_SHA="317faf612b8e8a4e319fe3ec67404561b928ea9090fc23d1da20ce0900257950"
RESULT=ROOT/"outputs/poc-ixi265-atropos-typed-mask-v3"
SEALED=ROOT/"artifacts/poc-ixi265-atropos-typed-mask-run-v3"
RESULT_QA=ROOT/"artifacts/ixi265-typed-mask-independent-v3/result-qa.json"
QA_SHA="0b2918d3d6d1c17e772cad4649689f67ff9546083eced8bfe7d5796db0ed6509"
OUTCOME_SHA="a9c241064f0ebc67e460c2f500f6b389e23bdb09fbda3667ab41d02b544c1bcf"
OUTPUT=ROOT/"outputs/poc-ixi265-tissue-display-v1"
BUNDLE_NAME="IXI265-Guys-0845-tissue-estimates.ressectionlab"
MARKER=ROOT/"outputs/poc-ixi265-tissue-display-v1-attempt.json"
RESOURCES={"threads":2,"rss_limit_bytes":2*1024**3,"seconds":300.,"available_memory_floor_bytes":1024**3,
    "output_temporary_limit_bytes":512*1024**2,"disk_floor_bytes":50*1024**3,"sample_interval_seconds":.2,"attempts":1,
    "memory_scope":"supervising_Python_plus_worker_and_recursive_descendants"}
LAYER_NAMES=("atropos_csf_like_estimate","atropos_gray_matter_like_estimate","atropos_white_matter_like_estimate","unknown_missing_intensity")
LABELS=("CSF-like estimate","Gray matter-like estimate","White matter-like estimate","Unknown · 8 voxels")
COLORS=("#65bfff","#e8b56a","#d5a0ff","#ff4fa3")


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def helpers():
    path=ROOT/"scripts/poc_ixi265_atropos_typed_mask_run.py"
    if sha(path)!="b8d65e0522130df1a4e012811527ab17116a46436805c0152c0a3956a921e9ec":raise ValueError("Frozen supervisor helper changed")
    spec=importlib.util.spec_from_file_location("guarded_tissue_fitting_helpers",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,module.kernel().previous(),module.kernel().previous().reference()


def plan():
    return {"schema":"ixi265-display-only-tissue-package-v1","patient_group":"IXI:265","role":"TRAIN",
        "display_label":"T1 · estimated tissues","layer_names":list(LAYER_NAMES),"layer_labels":list(LABELS),
        "source_host_sha256":HOST_SHA,"acquired_primary_T1_sha256":T1_SHA,"fit_outcome_sha256":OUTCOME_SHA,
        "numerical_result_QA_sha256":QA_SHA,"resources":RESOURCES,"output":str(OUTPUT/BUNDLE_NAME),
        "preserved_original_count":16,"preserved_reference_display_count":2,"new_display_count":1,
        "estimated_total_attachment_expanded_bytes":334233600,"source_primary_unchanged":True,
        "raw_posteriors":"archived_unmodified_class_membership_probabilities_not_physical_PVE",
        "unknown":"eight_missing_intensity_cells_remain_unclassified_not_CSF",
        "frame_policy":"native_indices_preserved; original_T1_affine_retained; any_tolerated_header_roundoff_explicitly_recorded",
        "cross_source_fusion_permitted":False,"anatomy_accepted":False,"host_accepted":False,"growth_use_accepted":False,
        "training_admitted":False,"fitting_or_registration_run":False,"catalog_modified":False,"packaging_released":False}


def preparation():
    record=json.loads((PREPARATION/"preparation.json").read_text())
    if record["script_sha256"]!=sha(__file__):raise ValueError("Packaging script differs from preparation")
    for row in record["frozen_files"]:
        path=ROOT/row["path"]
        if path.resolve()!=path or sha(path)!=row["sha256"]:raise ValueError("Packaging prerequisite changed")
    return record


def verify_inputs():
    """All fixed locations before any source byte reads; no caller source path API."""
    inventory=json.loads((PREPARATION/"expected-inputs.json").read_text())
    for row in inventory["files"]:
        path=ROOT/row["path"]
        if not path.is_relative_to(ROOT) or path.resolve()!=path:raise ValueError("Redirected packaging source")
    for row in inventory["files"]:
        if sha(ROOT/row["path"])!=row["sha256"]:raise ValueError("Changed packaging source: "+row["path"])
    if sha(HOST)!=HOST_SHA or sha(RESULT_QA)!=QA_SHA or sha(SEALED/"outcome.json")!=OUTCOME_SHA:
        raise ValueError("Foreign host or result provenance")
    qa=json.loads(RESULT_QA.read_text())
    if qa["schema"]!="ixi265-typed-mask-independent-result-audit-v3" or qa["status"]!="pass_saved_numerical_candidate_remains_unaccepted":
        raise ValueError("Independent saved-array verification required")
    return inventory


def require_host(primary,saved,*,original_count=16,reference_count=2):
    refs={r.source_id:r for r in primary.source_refs}
    if (saved is None or len(saved["originals"])!=original_count or len(saved["sources"])!=reference_count
            or primary.compartments or primary.brain_mask is None or primary.prior_proposals or primary.critical_evidence
            or primary.metadata["patient_group"]!="IXI:265" or primary.metadata["split"]["role"]!="TRAIN"
            or primary.metadata["training_admitted"] is not False or primary.metadata["split"]["training_admitted"] is not False
            or refs["structural"].sha256!=T1_SHA or primary.metadata["anatomy_workbench_only"] is not True):
        raise ValueError("Canonical unpromoted source-host contract changed")


def validate_partition(primary,labels,domain,unknown,hard_masks,*,expected_unknown):
    shape=primary.mri.shape
    if (any(a.shape!=shape for a in [labels,domain,unknown,*hard_masks])
            or not np.isfinite(labels).all() or not np.isin(labels,[0,1,2,3]).all()
            or domain.dtype!=np.bool_ or unknown.dtype!=np.bool_):raise ValueError("Invalid native class/domain arrays")
    expected=np.zeros(shape,bool)
    for index in expected_unknown:expected[index]=True
    if len(expected_unknown)!=8 or int(expected.sum())!=8 or not np.array_equal(unknown,expected):
        raise ValueError("Exactly the frozen eight UNKNOWN cells required")
    if (not np.array_equal(domain,primary.brain_mask & (primary.mri>0))
            or not np.array_equal(domain|unknown,primary.brain_mask) or np.any(domain&unknown)
            or not np.array_equal(unknown,primary.brain_mask & (primary.mri==0))
            or not np.array_equal(labels!=0,domain)):
        raise ValueError("Fitting support or UNKNOWN partition changed")
    if len(hard_masks)!=3 or any(a.dtype!=np.bool_ or not np.array_equal(a,labels==i) for i,a in enumerate(hard_masks,1)):
        raise ValueError("Hard estimated class masks differ from raw class IDs")
    return {"fit_domain_voxels":int(domain.sum()),"unknown_voxels":int(unknown.sum()),
        "class_voxels":[int(a.sum()) for a in hard_masks],"source_arrays_changed":False,"tissue_identity_confirmed":False}


def tissue_source(primary,masks,unknown,provenance,frames):
    from resectionlab.core import SourceRef,thaw_json
    from resectionlab.workspace_imaging import DisplaySource
    model={"name":"N4 + Atropos within-image fitting","version":"ANTsPy 0.6.3","license":"Apache-2.0",
        "device":"CPU · 2 threads","requirements":"Acquired T1 and positive estimated brain support; no tissue priors",
        "weights_path":"None · no pretrained weights","weights_sha256":"not applicable"}
    layers={}
    for i,name in enumerate(LAYER_NAMES):
        layers[name]={"name":name,"label":LABELS[i],"kind":"unknown" if i==3 else "tissue_estimate",
            "color":COLORS[i],"provenance":"observed" if i==3 else "estimated",
            "source_role":"missing_intensity_abstention" if i==3 else "display_only_fitted_intensity_class",
            "policy_use":"excluded","source_sha256":provenance["unknown_sha256"] if i==3 else provenance["mask_sha256"][i],
            "model":None if i==3 else model,"runtime_seconds":provenance["runtime_seconds"],
            "review_status":"Unclassified missing intensity" if i==3 else "Anatomical identity unreviewed",
            "coverage":"Eight exact zero-intensity source voxels are UNKNOWN; not CSF or corrected anatomy." if i==3 else
                "Estimated ordered T1 intensity class; tissue name is an assumption, not confirmed anatomy. Raw posteriors are probabilities, not physical partial-volume fractions.",
            "source_class_id":None if i==3 else i+1,"fit_outcome_sha256":OUTCOME_SHA,"independent_numerical_QA_sha256":QA_SHA}
    labels={**dict(zip(LAYER_NAMES[:3],masks)),LAYER_NAMES[3]:unknown}
    metadata={**thaw_json(primary.metadata),"display_label":"T1 · estimated tissues","anatomy_layers":layers,
        "annotation_use":"display_only","annotation_policy_use":"excluded","anatomical_review":"pending",
        "workspace_registration":{"kind":"estimated-unreviewed","overlayPermitted":False},
        "missing_anatomy":["skull","ventricles","confirmed_tissue_identity","fresh_vessel_inference"],
        "tissue_estimate_provenance":{**provenance,"native_output_frames":frames,"posterior_semantics":"membership_probabilities_not_PVE",
            "native_indices_unchanged":True,"display_affine_is_unchanged_acquired_T1":True,"resampling_or_thresholding":False},
        "training_admitted":False,"anatomy_accepted":False,"host_accepted":False,"growth_use_accepted":False,
        "model_identity":model,"segmentation_cache_policy":"saved masks only; never refit on open"}
    observed=next(r for r in primary.source_refs if r.source_id=="structural")
    display=replace(primary,compartments=labels,source_compartments=labels,brain_mask=None,context=None,
        structural_evidence={},critical_evidence={},prior_proposals={},functional_evidence=None,metadata=metadata,
        source_refs=(observed,SourceRef("supplied_target_annotation",(RESULT/"class_labels_raw.nii.gz").as_uri(),
            sha256=provenance["labels_sha256"],license="Apache-2.0 software; source acquisition CC-BY-SA-3.0",native_frame="RAS+",provenance="estimated")))
    return DisplaySource(display,"T1","estimated")


def save_and_check(primary,artifacts,saved,estimate,originals,destination):
    from resectionlab.workspace_bundle import load_workspace,save_workspace,view_state
    from resectionlab.workspace_imaging import source_registration
    sources=[*saved["sources"],estimate]
    state=view_state(primary,sources);state["states"].update(deepcopy(saved["imagingState"]["states"]))
    state["selectedSeriesId"]=estimate.identity(primary)
    state["states"][estimate.identity(primary)]["viewer"]={"layerOpacity":{n:.48 if i<3 else 1. for i,n in enumerate(LAYER_NAMES)},
        "layerColor":dict(zip(LAYER_NAMES,COLORS))}
    session=save_workspace(primary,destination,sources=sources,imaging=state,episode=saved["episode"],selection=saved["selection"],artifacts=artifacts,originals=originals)
    reopened,restored_artifacts,restored=load_workspace(destination)
    if (reopened.semantic_hash!=primary.semantic_hash or reopened.planning_hash!=primary.planning_hash
            or restored_artifacts!=artifacts or restored["sessionHash"]!=session or restored["imagingState"]!=state
            or originals!=restored["originals"]):raise ValueError("Saved primary identity, state, artifacts, or original bytes changed")
    for a,b in [(primary,reopened),*[(s.case,r.case) for s,r in zip(sources,restored["sources"],strict=True)]]:
        if a.semantic_hash!=b.semantic_hash:raise ValueError("Display identity changed on reopen")
        for field in ["mri","affine","brain_mask"]:
            if not np.array_equal(getattr(a,field),getattr(b,field)):raise ValueError("Source image/frame/brain changed")
        for group in ["compartments","source_compartments"]:
            if set(getattr(a,group))!=set(getattr(b,group)) or any(not np.array_equal(v,getattr(b,group)[k]) for k,v in getattr(a,group).items()):
                raise ValueError("Source or estimate layer changed on reopen")
    if any(source_registration(reopened,s.case)[1]["overlayPermitted"] for s in restored["sources"]):
        raise ValueError("Separate reference or estimate admitted for fusion")
    return {"session_hash":session,"primary_case_hash":reopened.semantic_hash,"primary_planning_hash":reopened.planning_hash,
        "estimated_series_id":estimate.identity(primary),"source_manifest":estimate.manifest(primary),
        "originals_count":len(originals),"preserved_originals_count":len(saved["originals"]),"all_original_bytes_unchanged":True,
        "primary_and_existing_references_unchanged":True,"all_native_arrays_verified_after_reopen":True,"fusion_permitted":False}


def worker():
    if not json.loads((OUTPUT/"declaration.json").read_text()).get("released_single_package"):
        raise ValueError("Separate packaging release required")
    preparation();inventory=verify_inputs();_,old,base=helpers()
    sys.path.insert(0,str(ROOT/"src"))
    from resectionlab.workspace_bundle import load_workspace,OriginalFile
    import nibabel as nib
    primary,artifacts,saved=load_workspace(HOST);require_host(primary,saved)
    expected=json.loads((PREPARATION/"expected-host.json").read_text())
    if primary.semantic_hash!=expected["primary_case_hash"] or [x.manifest(i) for i,x in enumerate(saved["originals"])]!=expected["originals"]:
        raise ValueError("Canonical host inventory changed")
    if [s.manifest(primary) for s in saved["sources"]]!=expected["reference_sources"]:raise ValueError("Canonical display sources changed")
    frames={}
    def array(name,binary=False):
        image=nib.load(RESULT/(name+".nii.gz"))
        frames[name]=base.require_native_frame(primary.mri.shape,primary.affine,image.shape,**base.ras_to_lps_geometry(image.affine))
        values=np.asarray(image.dataobj)
        if not np.isfinite(values).all():raise ValueError("Nonfinite estimate")
        if binary:
            if not np.isin(values,[0,1]).all():raise ValueError("Nonbinary saved estimate")
            return values.astype(bool)
        return values
    labels=array("class_labels_raw");domain=array("fit_domain_native",True);unknown=array("excluded_unknown_native",True)
    names=["csf_like_intensity_estimated","gray_matter_like_intensity_estimated","white_matter_like_intensity_estimated"]
    masks=[array(n,True) for n in names]
    partition=validate_partition(primary,labels,domain,unknown,masks,expected_unknown=old.EXPECTED_UNKNOWN)
    total=np.zeros(primary.mri.shape,np.float64)
    for i in (1,2,3):
        probability=array(f"class_{i}_posterior")
        if np.any(probability<0) or np.any(probability>1) or np.any(probability[~domain]!=0):raise ValueError("Invalid raw posterior")
        total+=probability
    error=float(np.max(np.abs(total[domain]-1)))
    if error>2e-5:raise ValueError("Raw posterior normalization failure; no renormalization")
    del probability,total
    corrected=array("n4_corrected_estimated")
    if np.any(corrected[domain]<=0):raise ValueError("Invalid archived corrected intensity")
    del corrected
    candidate=json.loads((RESULT/"candidate.json").read_text());outcome=json.loads((SEALED/"outcome.json").read_text())
    raw_frame_checks=json.loads(RESULT_QA.read_text())["result"]["raw_native_temporary_posterior_comparisons"]
    provenance={"labels_sha256":sha(RESULT/"class_labels_raw.nii.gz"),"unknown_sha256":sha(RESULT/"excluded_unknown_native.nii.gz"),
        "mask_sha256":[sha(RESULT/(n+".nii.gz")) for n in names],"runtime_seconds":outcome["resources"]["elapsed_seconds"],
        "candidate_sha256":sha(RESULT/"candidate.json"),"parameters":candidate["parameters"],"source_binding":candidate["source_hashes"],
        "partition":partition,"posterior_sum_max_abs_error":error,"raw_native_posterior_header_roundoff_record":
        {"independent_result_QA_sha256":QA_SHA,"limit_mm":1e-4,"per_class_native_checks":raw_frame_checks,
            "raw_probability_values_unchanged":True,"bitwise_affine_equality_claimed":False,"no_resampling":True}}
    estimate=tissue_source(primary,masks,unknown,provenance,frames)
    originals=[*saved["originals"],*[OriginalFile.from_path(ROOT/row["path"],role=row["role"],expected_sha256=row["sha256"])
        for row in inventory["append_originals"]]]
    receipt=save_and_check(primary,artifacts,saved,estimate,originals,OUTPUT/BUNDLE_NAME)
    verify_inputs()
    base.write_json(OUTPUT/"package.json",{**plan(),**receipt,"status":"display_only_candidate_packaged_unaccepted",
        "packaging_released":True,"bundle_sha256":sha(OUTPUT/BUNDLE_NAME),"bundle_bytes":(OUTPUT/BUNDLE_NAME).stat().st_size,
        "script_sha256":sha(__file__),"preparation_sha256":sha(PREPARATION/"preparation.json"),"provenance":provenance,
        "native_output_frames":frames,"source_host_bytes_unchanged":True})


def violation(rss,available,elapsed,output_bytes,disk):
    if rss>RESOURCES["rss_limit_bytes"]:return "supervisor_and_worker_RSS_limit"
    if available<RESOURCES["available_memory_floor_bytes"]:return "available_memory_floor"
    if elapsed>RESOURCES["seconds"]:return "time_limit"
    if output_bytes>RESOURCES["output_temporary_limit_bytes"]:return "output_temporary_limit"
    if disk<RESOURCES["disk_floor_bytes"]:return "disk_floor"
    return None


def run(*,released=False,expected_preparation_sha256=None):
    if not released:raise ValueError("Packaging is not released")
    if OUTPUT.exists() or MARKER.exists():raise ValueError("One new packaging attempt only; no overwrite/retry")
    if OUTPUT.resolve()!=OUTPUT or MARKER.resolve()!=MARKER:raise ValueError("Redirected output refused")
    if sha(PREPARATION/"preparation.json")!=expected_preparation_sha256:raise ValueError("Explicit frozen preparation binding required")
    preparation();inventory=verify_inputs();runner,_,base=helpers()
    import psutil
    peak=base.sampled_rss(psutil);available=psutil.virtual_memory().available;disk=psutil.disk_usage(OUTPUT.parent).free
    reason=violation(peak,available,0,0,disk)
    if reason:raise ValueError(reason)
    declaration={**plan(),"released_single_package":True,"script_sha256":sha(__file__),"preparation_sha256":expected_preparation_sha256,
        "created_utc":datetime.now(timezone.utc).isoformat(),"expected_inputs":inventory}
    with MARKER.open("x") as f:json.dump(declaration,f,indent=2)
    OUTPUT.mkdir();base.write_json(OUTPUT/"declaration.json",declaration)
    env={**os.environ,"OPENBLAS_NUM_THREADS":"2","OMP_NUM_THREADS":"2","VECLIB_MAXIMUM_THREADS":"2","MKL_NUM_THREADS":"2","NUMEXPR_NUM_THREADS":"2"}
    env.pop("PYTHONPATH",None);started=time.monotonic();process=None
    record={"status":"running","supervisor_pid":os.getpid(),"resources":RESOURCES}
    try:
        with (OUTPUT/"worker.log").open("x") as log:
            process=subprocess.Popen([sys.executable,"-I",str(Path(__file__).resolve()),"--worker"],env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True);record["worker_pid"]=process.pid
            while process.poll() is None:
                try:peak=max(peak,base.sampled_rss(psutil,process.pid))
                except psutil.NoSuchProcess:pass
                available=min(available,psutil.virtual_memory().available);disk=min(disk,psutil.disk_usage(OUTPUT).free)
                reason=violation(peak,available,time.monotonic()-started,base.directory_bytes(OUTPUT),disk)
                if reason:raise RuntimeError(reason)
                time.sleep(.2)
        reason=violation(peak,available,time.monotonic()-started,base.directory_bytes(OUTPUT),disk)
        if reason or process.returncode:raise RuntimeError(reason or "worker_failed_see_retained_log")
        if not (OUTPUT/"package.json").is_file():raise RuntimeError("Missing package verification receipt")
        record["status"]="display_only_candidate_packaged_unaccepted"
    except BaseException as error:
        record.update(status="failed",failure=str(error))
        try:record["cleanup"]=runner.cleanup_worker(process)
        except BaseException as cleanup_error:record["cleanup_error"]=str(cleanup_error)
    try:verify_inputs();record["source_and_result_inputs_unchanged"]=True
    except Exception as error:
        record["source_and_result_inputs_unchanged"]=False;record["continuity_error"]=str(error)
        if record["status"]!="failed":record.update(status="failed",failure="Post-package source continuity failed")
    record.update(elapsed_seconds=time.monotonic()-started,exit_code=None if process is None else process.returncode,
        peak_supervisor_and_worker_rss_bytes_sampled=peak,minimum_available_memory_bytes_sampled=available,
        minimum_free_disk_bytes_sampled=disk,output_bytes=base.directory_bytes(OUTPUT),
        sampling_limitation="Between-sample peaks are not a hard reservation",anatomy_accepted=False,host_accepted=False,growth_use_accepted=False,
        finished_utc=datetime.now(timezone.utc).isoformat())
    base.write_json(OUTPUT/"attempt.json",record);return record


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--worker",action="store_true");p.add_argument("--released-package-spec",type=Path);a=p.parse_args()
    if a.worker:worker()
    elif a.released_package_spec:
        spec=json.loads(a.released_package_spec.read_text())
        if set(spec)!={"released_single_package","preparation_sha256"} or spec["released_single_package"] is not True:raise ValueError("Separate explicit release required")
        result=run(released=True,expected_preparation_sha256=spec["preparation_sha256"]);print(json.dumps(result,indent=2))
        raise SystemExit(0 if result["status"]=="display_only_candidate_packaged_unaccepted" else 1)
    else:print(json.dumps(plan(),indent=2))
