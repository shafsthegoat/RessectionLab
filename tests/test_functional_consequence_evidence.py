"""Generated checkpoint controls for replay-bound functional evidence."""

from copy import deepcopy

import numpy as np

from resectionlab.core import semantic_digest
from resectionlab.functional_evidence import functional_consequence_evidence
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_procedure_policy import PROPERTY, direct_removal_dimensions, procedure_source_memberships
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_sequential_task import generated_task


def generated_motor_scene(*, language=False):
    original = generated_task(2).scene
    if not language:
        return original
    properties = deepcopy(original.properties)
    policy = properties[PROPERTY]
    policy["structures"]["language-cortex"] = {
        **policy["structures"]["corticospinal"],
        "provenance": "Generated overlapping language-cortex annotation",
    }
    policy["policyHash"] = semantic_digest({k: v for k, v in policy.items() if k != "policyHash"})
    return SurgicalScene(original.labels, original.affine, original.regions, original.source, properties)


def scopes(quantity):
    return [quantity[scope] for scope in ("baseline", "incremental", "cumulative")]


def test_generated_membership_preserves_overlaps_and_prefix_across_native_effects():
    scene = generated_motor_scene(language=True)
    memberships = procedure_source_memberships(scene)
    first, second, *_ = sorted(memberships["structure:motor-cortex"] & memberships["structure:corticospinal"])
    other = min(memberships["structure:brainstem"])
    assert scene.labels.flat[first] == scene.labels.flat[second] == 1  # Both also belong to the target.

    prefix = SurgicalEngine(scene)
    prefix.state.update(removed=[first], cutEdges=[[first, other]], contact={str(first): .125})
    end = prefix.clone()
    end.state.update(transectedEdges=[[first, other], [second, other]],
                     contact={str(first): .25, str(second): .5, str(other): 4.})
    report = functional_consequence_evidence(end, prefix)
    assert report == functional_consequence_evidence(end, prefix,
        direct_dimensions=direct_removal_dimensions(end, prefix))
    assert report["sourceSceneId"] == scene.identity and report["clinicalValidation"] is False
    for name in ("motor-cortex", "corticospinal", "language-cortex"):
        row = report["structures"][name]
        assert row["source"]["kind"] == "generated_annotation"
        assert row["source"]["sourceBinding"] == scene.properties[PROPERTY]["sourceBinding"]
        assert row["source"]["coverageHash"] == semantic_digest(scene.properties[PROPERTY]["structures"][name]["coverageCells"])
        assert row["removedMm3"]["unit"] == row["cutSourceMm3"]["unit"] == "mm3"
        assert scopes(row["removedMm3"]) == [scene.voxel_mm3, 0., scene.voxel_mm3]
        assert scopes(row["cutSourceMm3"]) == [scene.voxel_mm3, scene.voxel_mm3, 2 * scene.voxel_mm3]
        assert row["recordedContactCellSeconds"]["unit"] == "cell*s"
        assert scopes(row["recordedContactCellSeconds"]) == [.125, .625, .75]


def test_absent_structure_coverage_and_missing_native_ledgers_stay_unknown():
    base = SurgicalEngine(generated_motor_scene())
    end = base.clone()
    del end.state["contact"]
    del end.state["transectedEdges"]
    report = functional_consequence_evidence(end, base)
    language = report["structures"]["language-cortex"]
    assert language["source"] is None
    assert all(language[key]["status"] == "unavailable" and scopes(language[key]) == [None] * 3
               for key in ("removedMm3", "cutSourceMm3", "recordedContactCellSeconds"))
    motor = report["structures"]["motor-cortex"]
    assert scopes(motor["removedMm3"]) == [0.] * 3
    assert scopes(motor["cutSourceMm3"]) == scopes(motor["recordedContactCellSeconds"]) == [None] * 3


def test_native_probe_ledger_keeps_site_matches_and_unknowns_independent_of_patient_function():
    labels = np.ones((2, 2, 2), np.uint16)
    regions = [{"id": 1, "name": "Generated tissue", "material": "soft_tissue"}]
    evidence = [{"id": "probe-site", "probe": "cortical", "cell": 0, "result": "generated response"}]
    scene = SurgicalScene(labels, np.eye(4), regions, {"kind": "generated-control"}, {"functionalEvidence": evidence})
    base = SurgicalEngine(scene)
    base.state["slots"]["right"]["instrument"] = "cortical_probe"
    quaternion = [0., 0., 0., 1.]
    base._effect("right", "apply", [0], scene.points([0])[0], quaternion, .1)
    end = base.clone()
    end.state["slots"]["right"]["instrument"] = "subcortical_probe"
    end._effect("right", "apply", [0], scene.points([0])[0], quaternion, .1)

    report = functional_consequence_evidence(end, base)
    sources = report["evidence"]
    probes = sources["nativeProbeObservations"]["probes"]
    assert probes["cortical"]["suppliedSiteIds"] == ["probe-site"]
    assert scopes(probes["cortical"]["suppliedEvidenceRecords"]) == [1., 0., 1.]
    assert scopes(probes["subcortical"]["unknownRecords"]) == [0., 1., 1.]
    for modality in ("populationPrior", "patientTractography", "intraoperativeMapping",
                     "intraoperativeMonitoring", "observedClinicalOutcome", "fullToolContact"):
        assert sources[modality]["status"] == "unavailable"
    assert report["structures"]["motor-cortex"]["removedMm3"]["cumulative"] is None
