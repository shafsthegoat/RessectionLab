"""Small generated workspace controls; no patient image decoding or fitting."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from resectionlab.core import CaseData,SourceRef,thaw_json
from resectionlab.workspace_bundle import OriginalFile,load_workspace,view_state
from resectionlab.workspace_imaging import DisplaySource,source_registration

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("tissue_display",ROOT/"scripts/prepare_ixi265_tissue_display.py")
package=importlib.util.module_from_spec(spec);spec.loader.exec_module(package)
UNKNOWN=((1,1,1),(1,3,2),(2,4,3),(3,5,4),(4,2,5),(5,6,2),(6,4,1),(7,5,3))


def fixture():
    shape=(9,8,7);brain=np.ones(shape,bool);brain[:,0,:]=False
    labels=np.broadcast_to(np.repeat([1,2,3],3)[:,None,None],shape).copy();labels[~brain]=0
    values=np.array([0,20,50,90],np.float32)[labels]
    unknown=np.zeros(shape,bool)
    for index in UNKNOWN:unknown[index]=True;values[index]=0;labels[index]=0
    angle=.43;affine=np.eye(4);affine[:3,:3]=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])@np.diag([-1.7,2.1,2.8]);affine[:3,3]=[31,-28,9]
    source=SourceRef("structural","synthetic://package-control",sha256=hashlib.sha256(values.tobytes()).hexdigest(),native_frame="RAS+",provenance="observed")
    primary=CaseData("generated-package-control",values,{},affine,(source,),brain_mask=brain,
        metadata={"is_synthetic":True,"patient_group":"generated-control","split":{"role":"TRAIN","training_admitted":False},
            "training_admitted":False,"anatomy_workbench_only":True,"workspace_registration":{"kind":"estimated-unreviewed","overlayPermitted":False},
            "missing_anatomy":["skull","tissue_classes","ventricles"],"anatomy_layers":{"brain":{"kind":"brain","provenance":"estimated"}}})
    domain=brain&~unknown;masks=[labels==i for i in (1,2,3)]
    provenance={"unknown_sha256":"e"*64,"labels_sha256":"a"*64,"mask_sha256":[str(i)*64 for i in (1,2,3)],"runtime_seconds":1.,"generated_control":True}
    return primary,labels,domain,unknown,masks,provenance


def test_generated_native_partition_and_no_source_or_primary_metadata_mutation():
    primary,labels,domain,unknown,masks,provenance=fixture()
    before=primary.semantic_hash;before_metadata=thaw_json(primary.metadata)
    check=package.validate_partition(primary,labels,domain,unknown,masks,expected_unknown=UNKNOWN)
    estimate=package.tissue_source(primary,masks,unknown,provenance,{})
    assert check["unknown_voxels"]==8 and sum(check["class_voxels"])+8==int(primary.brain_mask.sum())
    assert primary.semantic_hash==before and thaw_json(primary.metadata)==before_metadata
    assert np.array_equal(estimate.case.mri,primary.mri) and np.array_equal(estimate.case.affine,primary.affine)
    assert estimate.case.brain_mask is None and not source_registration(primary,estimate.case)[1]["overlayPermitted"]
    assert estimate.case.metadata["display_label"]=="T1 · estimated tissues"
    assert estimate.case.metadata["training_admitted"] is False
    assert "tissue_classes" in primary.metadata["missing_anatomy"] and "tissue_classes" not in estimate.case.metadata["missing_anatomy"]
    assert "confirmed_tissue_identity" in estimate.case.metadata["missing_anatomy"]
    for name in package.LAYER_NAMES:
        layer=estimate.case.metadata["anatomy_layers"][name]
        assert layer["policy_use"]=="excluded"
    assert estimate.case.metadata["anatomy_layers"][package.LAYER_NAMES[-1]]["label"]=="Unknown · 8 voxels"
    assert estimate.case.metadata["tissue_estimate_provenance"]["posterior_semantics"]=="membership_probabilities_not_PVE"


@pytest.mark.parametrize("fault",["unknown_to_CSF","ninth_unknown","changed_domain","changed_hard_mask","fractional_class","outside_class"])
def test_estimated_labels_cannot_change_support_hide_unknowns_or_relabel(fault):
    p,l,d,u,m,_=fixture()
    if fault=="unknown_to_CSF":l[UNKNOWN[0]]=1;m[0][UNKNOWN[0]]=True
    if fault=="ninth_unknown":u[0,1,0]=True
    if fault=="changed_domain":d[0,1,0]=False
    if fault=="changed_hard_mask":m[1][0,1,0]=True
    if fault=="fractional_class":l=l.astype(float);l[0,1,0]=1.5
    if fault=="outside_class":l[0,0,0]=1
    with pytest.raises(ValueError):package.validate_partition(p,l,d,u,m,expected_unknown=UNKNOWN)


def test_generated_bundle_roundtrip_preserves_primary_two_references_16_originals_and_raw_memberships(tmp_path):
    p,l,d,u,m,provenance=fixture();estimate=package.tissue_source(p,m,u,provenance,{})
    refs=[]
    for index,modality in enumerate(["MRA","T1"]):
        native=replace(p,mri=p.mri+index+1,brain_mask=None,metadata={**thaw_json(p.metadata),"reference_control":index})
        refs.append(DisplaySource(native,modality,"none"))
    originals=[]
    for i in range(16):
        data=f"exact-generated-original-{i}".encode();originals.append(OriginalFile(f"source{i}.bin",f"synthetic://original/{i}",hashlib.sha256(data).hexdigest(),data,"metadata"))
    raw=np.array([0,.01,.4,.59,1],np.float32).tobytes()
    extra=OriginalFile("generated-posterior.bin","synthetic://posterior",hashlib.sha256(raw).hexdigest(),raw,"model-result")
    saved={"sources":refs,"originals":originals,"imagingState":view_state(p,refs),"episode":None,"selection":None}
    report=package.save_and_check(p,{"generated":"artifact"},saved,estimate,[*originals,extra],tmp_path/"display.ressectionlab")
    reopened,artifacts,after=load_workspace(tmp_path/"display.ressectionlab")
    assert report["primary_and_existing_references_unchanged"] and report["preserved_originals_count"]==16
    assert reopened.semantic_hash==p.semantic_hash and reopened.planning_hash==p.planning_hash
    assert after["originals"][:16]==originals and after["originals"][-1].data==raw
    assert after["imagingState"]["selectedSeriesId"]==estimate.identity(p)
    assert len(after["sources"])==3
    for before,after_source in zip([*refs,estimate],after["sources"],strict=True):
        assert before.manifest(p)==after_source.manifest(reopened)
    assert set(after["sources"][-1].case.compartments)==set(package.LAYER_NAMES)


def test_unreleased_package_never_reads_source_or_creates_output(monkeypatch):
    monkeypatch.setattr(package,"preparation",lambda:pytest.fail("Unreleased preparation access"))
    monkeypatch.setattr(package,"verify_inputs",lambda:pytest.fail("Unreleased source access"))
    with pytest.raises(ValueError,match="not released"):package.run()


def test_redirected_input_is_rejected_before_any_source_hash(tmp_path,monkeypatch):
    prep=tmp_path/"prep";prep.mkdir();real=tmp_path/"real";real.write_text("protected sentinel")
    redirect=tmp_path/"foreign";redirect.symlink_to(real)
    (prep/"expected-inputs.json").write_text(json.dumps({"files":[{"path":"foreign","sha256":"a"*64}]}))
    monkeypatch.setattr(package,"ROOT",tmp_path);monkeypatch.setattr(package,"PREPARATION",prep)
    monkeypatch.setattr(package,"sha",lambda p:pytest.fail("Redirected source bytes read"))
    with pytest.raises(ValueError,match="Redirected"):package.verify_inputs()


def test_packaging_resources_and_fixed_no_promotion_contract():
    safe=[100,2*1024**3,1,1,100*1024**3]
    assert package.violation(*safe) is None
    for i,value in enumerate([2*1024**3+1,1024**3-1,300.1,512*1024**2+1,50*1024**3-1]):
        args=safe.copy();args[i]=value;assert package.violation(*args)
    plan=package.plan()
    assert plan["preserved_original_count"]==16 and plan["preserved_reference_display_count"]==2
    assert not any(plan[k] for k in ["anatomy_accepted","host_accepted","growth_use_accepted","training_admitted","fitting_or_registration_run","packaging_released"])


@pytest.mark.parametrize("existing",["output","marker"])
def test_existing_attempt_or_output_refused_before_input_access(tmp_path,monkeypatch,existing):
    output=tmp_path/"output";marker=tmp_path/"attempt.json"
    if existing=="output":output.mkdir()
    else:marker.write_text("preserved previous attempt")
    monkeypatch.setattr(package,"OUTPUT",output);monkeypatch.setattr(package,"MARKER",marker)
    monkeypatch.setattr(package,"sha",lambda p:pytest.fail("Prior attempt must block all input reads"))
    with pytest.raises(ValueError,match="no overwrite/retry"):package.run(released=True)


def test_input_continuity_rejects_changed_result_without_decoding_any_images(tmp_path,monkeypatch):
    prep=tmp_path/"prep";prep.mkdir();sealed=tmp_path/"sealed";sealed.mkdir()
    host=tmp_path/"host.bin";host.write_bytes(b"generated original host bytes")
    result=tmp_path/"posterior.bin";result.write_bytes(b"generated raw membership bytes")
    qa=tmp_path/"qa.json";qa.write_text(json.dumps({"schema":"ixi265-typed-mask-independent-result-audit-v3","status":"pass_saved_numerical_candidate_remains_unaccepted"}))
    outcome=sealed/"outcome.json";outcome.write_text('{"generated":true}')
    entries=[{"path":str(p.relative_to(tmp_path)),"sha256":package.sha(p)} for p in [host,result,qa,outcome]]
    (prep/"expected-inputs.json").write_text(json.dumps({"files":entries}))
    monkeypatch.setattr(package,"ROOT",tmp_path);monkeypatch.setattr(package,"PREPARATION",prep)
    monkeypatch.setattr(package,"HOST",host);monkeypatch.setattr(package,"RESULT_QA",qa);monkeypatch.setattr(package,"SEALED",sealed)
    monkeypatch.setattr(package,"HOST_SHA",package.sha(host));monkeypatch.setattr(package,"QA_SHA",package.sha(qa));monkeypatch.setattr(package,"OUTCOME_SHA",package.sha(outcome))
    assert package.verify_inputs()["files"]==entries
    result.write_bytes(b"changed membership bytes")
    with pytest.raises(ValueError,match="Changed packaging source"):package.verify_inputs()
