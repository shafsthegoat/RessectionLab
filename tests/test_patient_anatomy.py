from pathlib import Path
import json
import numpy as np
import nibabel as nib
import pytest
from resectionlab.patient_anatomy import (
    PatientImagingCollection as Collection,
    PatientAnatomySnapshot as Snapshot,
    resample,
    world,
    reconstruction_lock,
)
from resectionlab.anatomy_geometry import mesh, vessel_graph
from resectionlab.anatomy_editor import AnatomyDraft
from resectionlab.anatomy_import import review_registration
from resectionlab.anatomy_planning import (
    planning_case,
    route_clearance,
    removal_preview,
)
from resectionlab.critical_evidence import resolve_critical_evidence


@pytest.fixture
def patient(tmp_path):
    a = np.array([[-0.5, 0, 0, 8], [0, 0, -0.7, 12], [0, 0.6, 0, -4], [0, 0, 0, 1.0]])
    image = nib.Nifti1Image(np.ones((24, 25, 26), np.float32), a)
    image.header.set_xyzt_units("mm")
    nib.save(image, tmp_path / "mri.nii.gz")
    c = Collection.create(tmp_path / "patient", "patient-a")
    s = c.add_series(
        tmp_path / "mri.nii.gz", series_id="mri", modality="T1", patient_id="patient-a"
    )
    a = np.array(s["affine"])
    xyz = np.indices(image.shape)
    sphere = ((xyz[0] - 12) ** 2 + (xyz[1] - 12) ** 2 + (xyz[2] - 12) ** 2) < 36
    vessel = np.zeros(image.shape, np.uint8)
    vessel[6:19, 10, 10] = 1
    vessel[6:19, 13, 10] = 1
    c.add_layer(
        sphere.astype(np.uint8),
        a,
        layer_id="tumor",
        source_series="mri",
        kind="tumor",
        role="target",
        color="#aa00bb",
        probability=sphere * 0.8 + 0.1,
    )
    c.add_layer(
        vessel,
        a,
        layer_id="vessels",
        source_series="mri",
        kind="vessels",
        role="critical",
        color="#ff0000",
    )
    c.add_layer(
        np.ones(image.shape, np.uint8),
        a,
        layer_id="brain",
        source_series="mri",
        kind="brain",
        role="context",
        color="#888888",
    )
    c.document["missing"] = ["angiography"]
    c.save()
    return c


def test_native_roundtrip_and_roles(patient):
    c = Collection(patient.root)
    snap = Snapshot.capture(c)
    a = c.patient_affine(c.layer("tumor"))
    assert snap.sample(world(a, [12, 12, 12]))["layers"]["tumor"]["occupied"]
    assert not snap.sample(world(a, [-1, 12, 12]))["layers"]["tumor"]["known"]
    grid = snap.simulation_grid()
    assert set(grid["targets"]) == {"tumor"} and set(grid["critical"]) == {"vessels"}
    case = planning_case(snap)
    assert set(case.compartments) == {"tumor"}
    assert resolve_critical_evidence(case).hard_exclusion.sum() == 26
    assert case.metadata["patient_anatomy_snapshot"] == snap.snapshot_id
    assert snap.distance("vessels",world(a,[12,10,10]))["distance_lower_bound_mm"] < 0
    assert not snap.distance("vessels",world(a,[-2,10,10]))["available"]


def test_edit_undo_commit_and_invalidation(patient):
    c = patient
    before = Snapshot.capture(c)
    row = c.layer("tumor")
    original = row["data"].copy()
    a = c.patient_affine(row)
    draft = AnatomyDraft(c, "tumor")
    draft.apply(c, "erase", points=[world(a, [12, 12, 12]).tolist()], radius_mm=1)
    edited = draft.mask.copy()
    assert not np.array_equal(edited, draft.original)
    draft.step("undo")
    assert np.array_equal(draft.mask, draft.original)
    draft.step("redo")
    assert np.array_equal(draft.mask, edited)
    c.commit_edit(
        "tumor",
        edited,
        author="test reviewer",
        expected=before.snapshot_id,
        operations=draft.operations,
    )
    assert (
        c.identity() != before.snapshot_id
        and c.layer("tumor")["original_data"] == original
    )
    assert before.sample(world(a, [12, 12, 12]))["layers"]["tumor"]["occupied"]
    assert not Snapshot.capture(c).sample(world(a, [12, 12, 12]))["layers"]["tumor"][
        "occupied"
    ]
    with pytest.raises(ValueError, match="stale"):
        draft.apply(c, "erase", points=[], radius_mm=1)
    assert Collection(c.root).identity() == c.identity()


def test_registration_not_guessed_from_grid(patient, tmp_path):
    c = patient
    c.add_series(
        c.path(c.series("mri")["original"]["path"]), series_id="other", modality="CT"
    )
    with pytest.raises(ValueError, match="review"):
        c.patient_affine(c.series("other"))
    c.set_registration("other", np.eye(4), status="unreviewed", method="test")
    with pytest.raises(ValueError, match="review"):
        c.patient_affine(c.series("other"))
    with pytest.raises(ValueError, match="collinear"):
        review_registration(
            c,
            "other",
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            "r",
        )
    pts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    review_registration(c, "other", pts, pts, "r")
    assert c.patient_affine(c.series("other")).shape == (4, 4)
    with pytest.raises(ValueError, match="different patients"):
        c.add_series(
            c.path(c.series("mri")["original"]["path"]),
            series_id="bad",
            modality="CT",
            patient_id="patient-b",
        )


def test_surfaces_do_not_merge_thin_branches_or_modify_masks(patient):
    c = patient
    row = c.layer("vessels")
    before = c.mask(row).copy()
    g = mesh(c, row)
    assert len(g["positions"]) > 0
    assert np.array_equal(before, c.mask(row))
    assert g["indices"].max() < len(g["positions"])
    assert np.isfinite(g["normals"]).all()
    assert g["max_displacement_mm"] <= 0.151
    assert np.array_equal(g["indices"], mesh(c, row)["indices"])
    graph = vessel_graph(before, row["affine"])
    assert graph["components"] == 2
    f = g["positions"][g["indices"]]
    assert np.all(
        np.linalg.norm(np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0]), axis=1) > 0
    )
    tumor = mesh(c, c.layer("tumor"))
    assert float(tumor["max_displacement_mm"]) <= 0.251


def test_conservative_route_and_reversible_removal(patient):
    snap = Snapshot.capture(patient)
    a = patient.patient_affine(patient.series("mri"))
    crossing = route_clearance(snap, world(a, [3, 10, 10]), world(a, [20, 10, 10]), 0.3)
    clear = route_clearance(snap, world(a, [3, 21, 20]), world(a, [20, 21, 20]), 0.3)
    assert crossing["constraints"]["vessels"]["intersects_or_unresolved"]
    assert not clear["constraints"]["vessels"]["intersects_or_unresolved"]
    removal = removal_preview(snap, world(a, [12, 12, 12]), 1)
    baseline = snap.simulation_grid()
    preview = snap.simulation_grid(removed=removal)
    assert preview["targets"]["tumor"].sum() < baseline["targets"]["tumor"].sum()
    assert np.array_equal(
        preview["critical"]["vessels"], baseline["critical"]["vessels"]
    )
    assert (
        snap.simulation_grid()["targets"]["tumor"].sum()
        == baseline["targets"]["tumor"].sum()
    )
    assert (
        planning_case(snap, removed=removal).planning_hash
        != planning_case(snap).planning_hash
    )


def test_mirrored_oblique_resampling_and_content_integrity(patient):
    data = np.zeros((8, 9, 10), np.uint8)
    data[2, 3, 4] = 1
    a = np.eye(4)
    theta = 0.6
    a[:2, :2] = [[-np.cos(theta), -np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    assert np.array_equal(resample(data, a, data.shape, a), data)
    shifted = a.copy()
    shifted[:3, 3] = a[:3, 0]
    out = resample(data, a, data.shape, shifted)
    assert out[1, 3, 4] == 1
    with pytest.raises(ValueError, match="escapes"):
        patient.path("../escape.npy")
    doc = json.loads((patient.root / "patient-anatomy.json").read_text())
    doc["patient_id"] = "other"
    (patient.root / "patient-anatomy.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="fingerprint"):
        Collection(patient.root)


def test_job_serialization(tmp_path):
    with reconstruction_lock(tmp_path):
        with pytest.raises(ValueError, match="Another"):
            with reconstruction_lock(tmp_path):
                pass


def test_oblique_sections_and_registration_checkerboard(patient):
    from resectionlab.anatomy_sections import section

    c = patient
    centre = world(c.patient_affine(c.series("mri")), [12, 12, 12])
    result = section(
        c,
        series_id="mri",
        center=centre,
        normal=[1, 1, 1],
        size=32,
        spacing_mm=0.5,
        layers=["tumor"],
    )
    assert result["pixels"].shape == (32, 32, 4) and result["frame"] == "patient"
    assert abs(np.dot(result["u"], result["v"])) < 1e-6
    assert np.all(result["pixels"][..., 3] == 255)
    with pytest.raises(ValueError):
        section(c, series_id="mri", center=centre, normal=[0, 0, 0])


def test_annotation_metrics_do_not_claim_accuracy_without_reference(patient):
    from resectionlab.anatomy_evaluation import evaluate

    row = patient.layer("vessels")
    p = patient.mask(row)
    before = p.copy()
    assert evaluate(p, None, row["affine"])["accuracy"] is None
    r = evaluate(p, p, row["affine"], annotation_coverage=p)
    assert (
        r["centerline_recall"] == 1
        and r["centerline_precision"] == 1
        and r["boundary_hd95_mm"] == 0
    )
    assert np.array_equal(p, before)
    missing = p.copy()
    missing[:, 13, :] = False
    assert evaluate(missing, p, row["affine"])["centerline_recall"] < 1


def test_bridge_reopens_rejects_stale_and_persists_corrections(patient, tmp_path):
    from resectionlab.desktop_bridge import BridgeSession, _Request

    b = BridgeSession(tmp_path / "transfers")
    request = _Request("test")
    progress = lambda *x: None
    opened = b.execute(
        "anatomyOpen",
        {"path": str(patient.root / "patient-anatomy.json")},
        request,
        progress,
    )
    args = {
        "collectionId": opened["collectionId"],
        "snapshotId": opened["document"]["snapshot_id"],
    }
    view = b.execute(
        "anatomyView",
        {**args, "seriesId": "mri", "layers": ["vessels", "tumor"]},
        request,
        progress,
    )
    assert (
        view["shape"] == [24, 25, 26]
        and view["compartments"][0]["surface"]["indices"]["dtype"] == "uint32"
    )
    with pytest.raises(ValueError, match="changed"):
        b.execute(
            "anatomyView",
            {**args, "snapshotId": "stale", "layers": []},
            request,
            progress,
        )
    a = patient.patient_affine(patient.layer("tumor"))
    point = world(a, [12, 12, 12]).tolist()
    draft = b.execute(
        "anatomyEdit",
        {
            **args,
            "action": "erase",
            "layerId": "tumor",
            "points": [point],
            "radiusMm": 1,
        },
        request,
        progress,
    )
    assert draft["draft"]["changedVoxels"] > 0
    assert Collection(patient.root).identity() == args["snapshotId"]
    b.execute("anatomyEdit", {**args, "action": "reject"}, request, progress)
    assert Collection(patient.root).identity() == args["snapshotId"]
    request.cancelled.set()
    with pytest.raises(Exception, match="cancelled"):
        b.execute("anatomyView", {**args, "layers": []}, request, progress)


def test_shared_parcellation_storage_and_fast_surfer_input_gate(patient, tmp_path):
    from resectionlab.anatomy_models import import_parcellation, reconstruct

    im = nib.Nifti1Image(np.indices((6, 6, 6))[0].astype(np.uint16), np.eye(4))
    nib.save(im, tmp_path / "regions.nii.gz")
    import_parcellation(
        patient,
        "mri",
        {
            "model": "fastsurfer",
            "path": str(tmp_path / "regions.nii.gz"),
            "output_sha256": "1" * 64,
        },
    )
    regions = [x for x in patient.document["layers"] if x["kind"] == "brain-region"]
    assert len(regions) == 5 and len({x["data"]["path"] for x in regions}) == 1
    patient.series("mri")["modality"] = "T1CE"
    with pytest.raises(ValueError, match="noncontrast"):
        reconstruct(patient, "mri", "fastsurfer")


def test_model_job_cancellation_and_memory_limit(tmp_path):
    import sys
    from resectionlab.anatomy_models import run_bounded

    with pytest.raises(InterruptedError):
        run_bounded(
            [sys.executable, "-c", "import time;time.sleep(10)"],
            tmp_path / "cancel",
            cancelled=lambda: True,
        )
    with pytest.raises(MemoryError):
        run_bounded(
            [sys.executable, "-c", "import time;time.sleep(10)"],
            tmp_path / "memory",
            memory_bytes=1,
        )


def test_existing_route_generator_consumes_snapshot_vessels(patient):
    from resectionlab.planning import generate_candidate_routes, SearchConfig
    from resectionlab.geometry import ToolGeometry, AccessWindow

    a = patient.patient_affine(patient.series("mri"))
    start, target = world(a, [3, 10, 10]), world(a, [12, 10, 10])
    direction = (target - start) / np.linalg.norm(target - start)

    def evaluate(c):
        case = planning_case(Snapshot.capture(c))
        result = generate_candidate_routes(
            case,
            windows=[AccessWindow(start, direction, 2)],
            tools=[ToolGeometry("test", 0.2, 0.2, 20, tip_length_mm=1)],
            explicit_targets=[("tumor", target)],
            config=SearchConfig(targets_per_compartment=1, max_windows=1),
        )
        return case, result.candidates[0]

    case, crossing = evaluate(patient)
    assert not crossing.feasible
    assert any(f.reason == "FORBIDDEN_COLLISION" for f in crossing.geometry.failures)
    assert crossing.structure_contact_volume_mm3["vessels"] > 0
    mesh(patient, patient.layer("vessels"))
    assert planning_case(Snapshot.capture(patient)).planning_hash == case.planning_hash
    # Controlled software counterfactual, never a patient correction: same route
    # with a separate saved fixture revision missing vascular constraints.
    other = Collection(patient.root, patient.document)
    other.document["layers"] = [
        r for r in other.document["layers"] if r["id"] != "vessels"
    ]
    other.save("test-only vessel omission", expected=patient.identity())
    _, without = evaluate(other)
    assert without.feasible and without.structure_contact_volume_mm3["vessels"] is None


def test_tract_functional_import_and_graph_cache(patient, tmp_path):
    from resectionlab.anatomy_import import import_anatomy_derivative
    from resectionlab.anatomy_geometry import inspect_vessel_geometry
    from resectionlab.anatomy_sections import section

    a = patient.patient_affine(patient.series("mri"))
    points = world(a, np.array([[8, 8, 8], [12, 12, 12], [15, 15, 15]])).astype(
        np.float32
    )
    tract = nib.streamlines.Tractogram([points], affine_to_rasmm=np.eye(4))
    nib.streamlines.save(tract, str(tmp_path / "tracts.trk"))
    t = import_anatomy_derivative(
        patient,
        tmp_path / "tracts.trk",
        source_series="mri",
        kind="tractography",
        label="tract",
    )
    assert np.allclose(patient.array(t["points_ras_mm"]), points)
    scores = np.zeros((24, 25, 26), np.float32)
    scores[12, 12, 12] = 3
    nib.save(nib.Nifti1Image(scores, a), tmp_path / "functional.nii.gz")
    f = import_anatomy_derivative(
        patient,
        tmp_path / "functional.nii.gz",
        source_series="mri",
        kind="functional",
        label="task map",
    )
    patient.save("test source-associated derivatives")
    sample = Snapshot.capture(patient).sample(world(a, [12, 12, 12]))["layers"][f["id"]]
    assert (
        sample["score"] == 3 and sample["role"] == "context" and not sample["occupied"]
    )
    rendered = section(
        patient,
        series_id="mri",
        center=world(a, [12, 12, 12]),
        normal=[1, 0, 0],
        size=32,
        layers=[f["id"]],
    )
    assert rendered["pixels"].shape == (32, 32, 4)
    g = inspect_vessel_geometry(patient, "vessels", world(a, [12, 10, 10]))
    assert g["components"] == 2 and g["nearest"]["radius_mm"] > 0
    assert inspect_vessel_geometry(patient, "vessels")["graph_id"] == g["graph_id"]


def test_model_adapters_reject_missing_inputs_and_preserve_empty_output(
    patient, tmp_path, monkeypatch
):
    from resectionlab.anatomy_models import reconstruct, import_proposal
    from resectionlab.patient_anatomy import file_hash

    with pytest.raises(ValueError, match="T1CE"):
        reconstruct(patient, "mri", "vesselfm-t1ce-experimental")
    home = tmp_path / "models"
    folder = home / "MRI_Glioma"
    folder.mkdir(parents=True)
    (folder / "pipeline.json").write_text(
        json.dumps(
            {"1": {"inputs": {"0": {"sequence": "T1-CE"}, "1": {"sequence": "FLAIR"}}}}
        )
    )
    monkeypatch.setenv("RESECTIONLAB_RAIDIONICS_MODELS", str(home))
    with pytest.raises(ValueError, match="publisher-declared"):
        reconstruct(patient, "mri", "glioma")
    mask = tmp_path / "empty.nii.gz"
    nib.save(
        nib.Nifti1Image(
            np.zeros((24, 25, 26), np.uint8), np.array(patient.series("mri")["affine"])
        ),
        mask,
    )
    key = import_proposal(
        patient,
        "mri",
        {
            "model": "glioma",
            "kind": "tumor",
            "path": str(mask),
            "output_sha256": file_hash(mask),
            "status": "empty tumor prediction; localization failed",
        },
    )
    assert not patient.mask(patient.layer(key)).any()


def test_fragment_review_preview_reject_commit_and_stale_identity(patient, tmp_path):
    from resectionlab.desktop_bridge import BridgeSession, _Request
    b = BridgeSession(tmp_path / 'transfers')
    def call(op, **args):
        return b.execute(op, args, _Request('test'), lambda *a: None)
    opened = call('anatomyOpen', path=str(patient.root / 'patient-anatomy.json'))
    identity = patient.identity()
    args = dict(collectionId=opened['collectionId'], snapshotId=identity)
    review = call('anatomyComponents', **args, layerId='vessels')
    assert len(review['components']) == 2
    assert [r['volumeMm3'] for r in review['components']] == sorted(r['volumeMm3'] for r in review['components'])
    with pytest.raises(ValueError, match='review changed'):
        call('anatomyEdit', **args, action='remove-components', layerId='vessels', reviewId='stale', componentIds=[1])
    assert not b.anatomy_drafts
    preview = call('anatomyEdit', **args, action='remove-components', layerId='vessels', reviewId=review['reviewId'], componentIds=[1])
    assert preview['draft']['changedVoxels'] == 13
    view = call('anatomyView', **args, seriesId='mri', layers=['vessels'], highlightLayer='vessels', highlightComponents=[1], reviewId=review['reviewId'])
    assert view['compartments'][-1]['name'] == '__fragment_review'
    assert Collection(patient.root).identity() == identity
    assert call('anatomyEdit', **args, action='undo', layerId='vessels')['draft']['changedVoxels'] == 0
    assert call('anatomyEdit', **args, action='redo', layerId='vessels')['draft']['changedVoxels'] == 13
    call('anatomyEdit', **args, action='reject')
    assert Collection(patient.root).identity() == identity
    original = patient.layer('vessels')['data'].copy()
    call('anatomyEdit', **args, action='remove-components', layerId='vessels', reviewId=review['reviewId'], componentIds=[1])
    committed = call('anatomyEdit', **args, action='commit', author='test-reviewer')
    updated = Collection(patient.root)
    assert updated.identity() != identity
    assert updated.mask(updated.layer('vessels')).sum() == 13
    assert updated.array(original).sum() == 26
    with pytest.raises(ValueError, match='changed'):
        call('anatomyView', **args, seriesId='mri', layers=['vessels'])
    assert committed['draft'] is None


def test_native_view_state_roundtrip_is_separate_from_anatomy(patient, tmp_path):
    from resectionlab.anatomy_workspace import save_view, read_view
    identity = patient.identity()
    state = dict(schema='patient-anatomy-view-v1', patientId='patient-a', snapshotId=identity, seriesId='mri', states={'mri': dict(cursor=[0,0,0],visible={'brain':True},settings={'layout':'3d-focus','volumeRendering':{'enabled':True}})})
    save_view(patient,state)
    assert read_view(patient) == state
    assert Collection(patient.root).identity() == identity
    with pytest.raises(ValueError):save_view(patient,{**state,'snapshotId':'stale'})
    state['states']['mri']['visible']['unknown']=True
    with pytest.raises(ValueError):save_view(patient,state)


def test_many_categorical_regions_share_one_view_transfer(patient, tmp_path):
    from resectionlab.desktop_bridge import BridgeSession, _Request
    data=np.zeros((24,25,26),np.uint16)
    for i in range(1,33):data[(i//6)+2,(i%6)+2,3:5]=i
    for i in range(1,33):
        patient.add_layer(data,patient.layer('brain')['affine'],layer_id=f'region-{i}',source_series='mri',kind='brain-region',role='context',color='#8899aa',label_value=i)
    patient.save()
    b=BridgeSession(tmp_path/'transfers')
    call=lambda op,args:b.execute(op,args,_Request('test'),lambda *a:None)
    opened=call('anatomyOpen',{'path':str(patient.root/'patient-anatomy.json')})
    view=call('anatomyView',{'collectionId':opened['collectionId'],'snapshotId':patient.identity(),'seriesId':'mri','layers':[f'region-{i}' for i in range(1,33)]})
    assert len(view['compartments'])==32
    assert len(view['labelSets'])==1
    assert all('mask' not in l and l['labelValue']>0 for l in view['compartments'])


def test_empty_segmentation_view_and_invalid_view_preferences(patient, tmp_path):
    from resectionlab.desktop_bridge import BridgeSession, _Request
    from resectionlab.anatomy_workspace import save_view
    data=np.zeros((24,25,26),np.uint8)
    patient.add_layer(data,patient.layer('brain')['affine'],layer_id='empty',source_series='mri',kind='tumor',role='target',color='#aabbcc')
    patient.save()
    b=BridgeSession(tmp_path/'transfers')
    call=lambda op,args:b.execute(op,args,_Request('test'),lambda *a:None)
    opened=call('anatomyOpen',{'path':str(patient.root/'patient-anatomy.json')})
    args={'collectionId':opened['collectionId'],'snapshotId':patient.identity()}
    view=call('anatomyView',{**args,'seriesId':'mri','layers':['empty']})
    assert view['compartments'][0]['surface']['positions']['byteLength']==0
    state=dict(schema='patient-anatomy-view-v1',patientId='patient-a',snapshotId=patient.identity(),seriesId='mri',states={'mri':dict(cursor=None,visible={},settings={'volumeRendering':{'density':999}})})
    with pytest.raises(ValueError,match='volume'):save_view(patient,state)
