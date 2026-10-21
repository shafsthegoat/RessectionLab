#!/usr/bin/env python3
"""Fixed descriptive native-MRA set agreement, never independent accuracy.

The incomplete derived reference did not enter the frozen inference. No model,
T1 registration, brain clipping, threshold search or postprocessing is used.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"outputs/poc-ixi265-vessel-agreement-v1"
ARTIFACT = ROOT/"artifacts/poc-ixi265-vessel-agreement-v1"
PROTOCOL = ARTIFACT/"protocol.json"
METADATA = {
 "outputs/poc-ixi265/source-receipt.json": "2f4f6507f56e6592f851ddc9c4520d217f9e85fd649c45025f4d0e079559eac6",
 "outputs/poc-ixi265/vesselfm-base-v1/inference.json": "a1b7a4198b33b2f429f058b658ecc4a70c33c31f7e25d7ad17e4d2b002eedfdf",
 "outputs/poc-ixi265/vesselfm-base-v1/registered.json": "de0430fd5600a22c26437b1928a56237f2e76423cbd9d21bc3f8afb18595680e",
 "artifacts/ixi-t1-mra-vessel-byte-intake-v1/source-contract.json": "159a4bba5fd9b8b5db0a4195b089f42850da4f3dae86bf101efc1a72de4ab387",
 "manifests/experiments/ixi-component-person-cohort-v1.json": "b27f28c21089e61f6c4899f172f608cd0f81391588777443939196b1700cb127",
}
INPUTS = {
 "MRA": ("outputs/poc-ixi265/originals/IXI265-MRA.nii.gz", "a87b381b882242b7d61a1930a8332d44200fdb3c56c68fc09c24a057eea86dc4"),
 "reference": ("outputs/poc-ixi265/originals/IXI265-derived_vessel_label.nii.gz", "9da13c332156c71a5200dbf88368bb87478914fe73034a42ca4eabf1dd5cdbfe"),
 "prediction": ("outputs/poc-ixi265/vesselfm-base-v1/vessels_mask.nii.gz", "d3eedb5a89e598cddd5eb017054f7fb0ce91c678278411c285cc00c2b8d277bc"),
 "score": ("outputs/poc-ixi265/vesselfm-base-v1/vessels_probability.nii.gz", "1745f34949bec22c4c2418e0f7e115c8e3acf0cd974e6ba4b18f7e81e80ecc75"),
 "coverage": ("outputs/poc-ixi265/vesselfm-base-v1/native_sampling_coverage.nii.gz", "a92bf82175e1208ebef037bf3117fc40f779c4cf5669f70422be7b0015bc863b"),
}
BOUNDS = {"wall_seconds": 120, "family_rss_bytes": 1024**3, "minimum_available_bytes": 1024**3,
          "output_bytes": 16*1024**2, "free_disk_reserve_bytes": 50*1024**3, "threads": 2}


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as source: return hashlib.file_digest(source,"sha256").hexdigest()


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as destination:
        json.dump(value, destination, indent=2, allow_nan=False); destination.write("\n")


def corner_error(shape, first, second):
    import numpy as np
    corners = np.array([(*point,1.) for point in itertools.product(*[(0,n-1) for n in shape])])
    return float(np.max(np.linalg.norm((corners@(np.asarray(first)-np.asarray(second)).T)[:,:3],axis=1)))


def prepare():
    """Only pinned JSON metadata; no image stat, open, hash, header or array read."""
    values = {}
    for path, digest in METADATA.items():
        require(sha(ROOT/path) == digest, "Pinned metadata changed")
        values[path] = json.loads((ROOT/path).read_text())
    source = values["outputs/poc-ixi265/source-receipt.json"]
    infer = values["outputs/poc-ixi265/vesselfm-base-v1/inference.json"]
    registered = values["outputs/poc-ixi265/vesselfm-base-v1/registered.json"]
    require(source["frozen_role"] == "TRAIN" and source["training_admitted"] is False
        and infer["split"] == {"role":"TRAIN","patient_group":"IXI:265","training_admitted":False}, "Fixed source role changed")
    require(infer["status"] == "succeeded" and infer["model_id"] == "vesselfm-base"
        and infer["parameters"]["threshold"] == .5 and infer["parameters"]["postprocessing"] is False
        and infer["source_reference_read_or_used"] is False, "Prediction is not the frozen independent-of-reference execution")
    for key in ("MRA", "reference"):
        require(source["sources"][key]["sha256"] == INPUTS[key][1], "Source input binding changed")
    require(infer["input"]["sha256"] == INPUTS["MRA"][1], "Inference source binding changed")
    for key, name in [("prediction","vessels_mask.nii.gz"),("score","vessels_probability.nii.gz")]:
        require(next(r for r in infer["outputs"] if r["name"] == name)["sha256"] == INPUTS[key][1], "Saved prediction binding changed")
    require(any(Path(r["path"]).name == "native_sampling_coverage.nii.gz" and r["sha256"] == INPUTS["coverage"][1] for r in registered["outputs"]), "Native domain binding changed")
    mr, ref = (source["sources"][k] for k in ("MRA","reference"))
    error = corner_error(mr["shape"],mr["affine_ras_mm"],ref["affine_ras_mm"])
    require(mr["shape"] == ref["shape"] == infer["input"]["shape"] == [512,512,100] and error < 1e-4, "Native grids are not comparable")
    contract = values["artifacts/ixi-t1-mra-vessel-byte-intake-v1/source-contract.json"]
    return {"schema":"poc-ixi265-fixed-native-vessel-agreement-v1", "case":"IXI265-Guys-0845", "patient_group":"IXI:265",
        "role":"TRAIN", "training_admitted":False, "metadata_pins":METADATA,
        "inputs":{k:{"path":str(ROOT/p),"sha256":d} for k,(p,d) in INPUTS.items()},
        "input_fixity": "Expected hashes from frozen receipts; real payload hashes verified only during released execution",
        "script_sha256":sha(__file__), "tests_sha256":sha(ROOT/"tests/test_poc_ixi265_vessel_agreement.py"),
        "shape":mr["shape"], "affine_ras_mm":mr["affine_ras_mm"], "metadata_reference_corner_error_mm":error,
        "comparison_admission":"Qualified descriptive comparison of two binary vessel-positive sets; target completeness/subtype equivalence not established",
        "annotation_lineage":contract["annotation_lineage"], "annotation_scope":contract["annotation_coverage"],
        "annotation_review_ambiguity":contract["annotation_review_ambiguity"], "training_lineage":infer["training_lineage"],
        "model":infer["model"], "source_reference_used_in_inference":False,
        "source_rights":{"MRA":"CC-BY-SA-3.0", "derived_reference":"CC-BY-4.0", "reference_url":"https://zenodo.org/records/17393202"},
        "fixed_parameters":{"threshold":.5,"postprocessing":False,"connected_components":26,"boundary_connectivity":6,
            "domain":"Frozen native MRA sampling coverage: one border voxel excluded; no T1/brain clipping"},
        "metrics":{"dice":"2*intersection/(prediction+reference) within fixed domain", "jaccard":"intersection/union within fixed domain",
            "positive_reference_coverage":"intersection/reference positive voxels within domain; not all-vessel sensitivity",
            "prediction_outside_reference":"UNLABELLED; never counted as false positive", "empty_denominator":"null, never fabricated perfect agreement"},
        "independent_accuracy_claim":False,"clinical_accuracy_claim":False,"held_out_claim":False,
        "anatomical_registration_used":False,"model_runs":0,"threshold_candidates":1,"bounds":BOUNDS}


def set_counts(prediction, reference, coverage, voxel_mm3):
    import numpy as np
    require(prediction.shape == reference.shape == coverage.shape, "Set shapes differ")
    p = prediction & coverage; r = reference & coverage
    npred, nref = int(p.sum()), int(r.sum()); intersection = int(np.count_nonzero(p&r))
    union = npred+nref-intersection
    ratio = lambda a,b: a/b if b else None
    return {"sampling_domain_voxels":int(coverage.sum()), "native_grid_voxels":int(coverage.size),
        "native_prediction_voxels":int(prediction.sum()),"native_reference_positive_voxels":int(reference.sum()),
        "prediction_in_domain_voxels":npred,"reference_in_domain_voxels":nref,"intersection_voxels":intersection,"union_voxels":union,
        "positive_mask_dice":ratio(2*intersection,npred+nref),"positive_mask_jaccard":ratio(intersection,union),
        "provided_positive_reference_coverage_fraction":ratio(intersection,nref),
        "reference_positive_voxels_not_covered_by_prediction":nref-intersection,
        "prediction_in_UNLABELLED_reference_voxels":npred-intersection,
        "reference_positive_fraction_retained_by_sampling_domain":ratio(nref,int(reference.sum())),
        "prediction_voxels_excluded_by_sampling_domain":int(prediction.sum())-npred,
        "reference_voxels_excluded_by_sampling_domain":int(reference.sum())-nref,
        "physical_volumes_mm3":{"prediction_in_domain":npred*voxel_mm3,"reference_in_domain":nref*voxel_mm3,"intersection":intersection*voxel_mm3},
        "interpretation":"Descriptive foreground set agreement only; unlabelled reference background is unknown"}


def component_boundary_summary(mask, coverage):
    import numpy as np
    from scipy import ndimage
    # One int32 component volume at a time; never retain a label volume.
    masked = mask & coverage
    labels, count = ndimage.label(masked, structure=ndimage.generate_binary_structure(3,3))
    sizes = np.bincount(labels.ravel())[1:]; del labels
    structure = ndimage.generate_binary_structure(3,1)
    surface = masked & ~ndimage.binary_erosion(masked, structure=structure, border_value=0)
    edge = coverage & ~ndimage.binary_erosion(coverage, structure=structure, border_value=0)
    return {"connectivity":26,"connected_components":int(count),"largest_component_voxels":int(sizes.max()) if sizes.size else 0,
        "component_count_by_size":{"1":int((sizes==1).sum()),"2_to_9":int(((sizes>=2)&(sizes<=9)).sum()),
            "10_to_99":int(((sizes>=10)&(sizes<=99)).sum()),"100_or_more":int((sizes>=100).sum())},
        "boundary_connectivity":6,"boundary_voxels":int(surface.sum()),"foreground_on_sampling_domain_boundary_voxels":int(np.count_nonzero(masked&edge)),
        "interpretation":"Voxel connectivity and boundary counts after fixed sampling restriction; no pruning, anatomical topology or surface-distance accuracy claim"}


def verify_threshold(prediction, score):
    import numpy as np
    require(prediction.shape == score.shape, "Score grid differs")
    require(np.isfinite(score).all() and score.min()>=0 and score.max()<=1, "Invalid saved model score")
    require(np.array_equal(prediction,score>.5), "Saved prediction differs from fixed score>0.5")


def execute():
    import nibabel as nib
    import numpy as np
    protocol = json.loads(PROTOCOL.read_text()); require(protocol == prepare(), "Frozen protocol changed")
    # All identities checked before any image decode.
    for row in protocol["inputs"].values(): require(sha(row["path"]) == row["sha256"], "Frozen image bytes changed")
    images = {k:nib.load(row["path"]) for k,row in protocol["inputs"].items()}
    grid_checks = {}
    for key,image in images.items():
        error = corner_error(protocol["shape"],image.affine,protocol["affine_ras_mm"])
        require(list(image.shape) == protocol["shape"] and image.header.get_xyzt_units()[0] == "mm" and error < 1e-4, "Native output grid changed")
        grid_checks[key] = {"shape":list(image.shape),"maximum_same_index_corner_error_mm":error}
    def binary(key):
        values = np.asanyarray(images[key].dataobj)
        require(np.all((values==0)|(values==1)), "Nonbinary saved mask")
        return values.astype(bool)
    p,r,c = (binary(key) for key in ("prediction","reference","coverage"))
    expected = np.zeros(c.shape,bool); expected[1:-1,1:-1,1:-1] = True
    require(np.array_equal(c,expected), "Native coverage is not the frozen geometric interior")
    del expected
    score = np.asanyarray(images["score"].dataobj)
    verify_threshold(p,score)
    del score
    voxel_mm3 = float(abs(np.linalg.det(images["MRA"].affine[:3,:3])))
    result = {"schema":"poc-ixi265-descriptive-native-set-agreement-result-v1", "protocol_sha256":sha(PROTOCOL),
        "protocol":protocol,"actual_grid_checks":grid_checks,"saved_score_threshold_equivalence_verified":True,
        "counts_and_agreement":set_counts(p,r,c,voxel_mm3),
        "prediction_components":component_boundary_summary(p,c),"reference_components":component_boundary_summary(r,c),
        "reference_used_for_diagnostic_only":True,"new_model_runs":0,"new_thresholds_or_postprocessing":False,
        "no_independent_clinical_or_held_out_accuracy_claim":True}
    write_new(OUTPUT/"agreement.json",result)


def run(release):
    import psutil
    expected = {"protocol_sha256":sha(PROTOCOL),"script_sha256":sha(__file__),"execute_fixed_comparison":True}
    require(json.loads(Path(release).read_text()) == expected, "Fixed diagnostic release mismatch")
    require(json.loads(PROTOCOL.read_text()) == prepare(), "Protocol or source metadata changed before launch")
    require(psutil.virtual_memory().available >= BOUNDS["minimum_available_bytes"], "Prelaunch available memory below floor")
    require(shutil.disk_usage(ROOT).free >= BOUNDS["free_disk_reserve_bytes"], "Prelaunch disk below reserve")
    OUTPUT.mkdir(parents=True,exist_ok=False)
    start = time.monotonic(); peak = 0; minimum = psutil.virtual_memory().available; failure = None
    env = dict(os.environ,PYTHONPATH=str(ROOT/"src"),OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
    with (OUTPUT/"diagnostic.log").open("x") as log:
        process = subprocess.Popen([sys.executable,__file__,"worker"],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while process.poll() is None:
            members = [psutil.Process()]
            try:
                worker = psutil.Process(process.pid)
                members.extend([worker,*worker.children(recursive=True)])
            except psutil.NoSuchProcess: pass
            current = 0
            for member in members:
                try: current += member.memory_info().rss
                except psutil.NoSuchProcess: pass
            peak = max(peak,current); minimum = min(minimum,psutil.virtual_memory().available)
            written = sum(p.stat().st_size for p in OUTPUT.rglob("*") if p.is_file())
            failures = [(peak>BOUNDS["family_rss_bytes"],"RSS_limit"),(minimum<BOUNDS["minimum_available_bytes"],"available_memory_floor"),
                (time.monotonic()-start>BOUNDS["wall_seconds"],"wall_limit"),(written>BOUNDS["output_bytes"],"output_limit"),
                (shutil.disk_usage(OUTPUT).free<BOUNDS["free_disk_reserve_bytes"],"disk_reserve")]
            failure = next((why for exceeded,why in failures if exceeded),None)
            if failure:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=3)
                except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL)
                break
            time.sleep(.1)
        process.wait()
    result = {"status":"succeeded" if not failure and process.returncode==0 else "failed", "exit_code":process.returncode,
        "guard_failure":failure,"elapsed_seconds":time.monotonic()-start,"peak_family_plus_supervisor_rss_bytes_sampled":peak,
        "minimum_available_bytes_sampled":minimum,"sampling_seconds":.1,"release_sha256":sha(release),"bounds":BOUNDS}
    if (OUTPUT/"agreement.json").exists(): result["agreement_sha256"] = sha(OUTPUT/"agreement.json")
    write_new(OUTPUT/"runtime.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("action",choices=["prepare","run","worker"])
    parser.add_argument("--release",type=Path); args=parser.parse_args()
    if args.action=="prepare": write_new(PROTOCOL,prepare())
    elif args.action=="worker": execute()
    else:
        require(args.release is not None,"Released fixed diagnostic declaration required")
        result=run(args.release); print(json.dumps(result)); raise SystemExit(0 if result["status"]=="succeeded" else 1)


if __name__=="__main__": main()
