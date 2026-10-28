"""Exact source-cell subdivision, source preservation and save/reopen contracts."""
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef, array_digest, thaw_json
from resectionlab.imaging import load_case, save_case
from resectionlab import surgical_environment as environment
from resectionlab.surgical_environment import refine_working_grid, default_target_names
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_scene import from_case


def fixture_case(affine=None, *, case_id='generated-grid-fixture', frame='RAS+'):
    shape = (3, 4, 5)
    brain = np.ones(shape, bool)
    tumor = np.zeros(shape, bool)
    tumor[1:, 1:3, 1:4] = True
    flair = np.zeros(shape, bool)
    flair[:, 1:3, :] = True
    source_tumor = tumor.copy()
    source_tumor[2, 2, 3] = False
    return CaseData(case_id=case_id, mri=np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        compartments={'brain': brain, 'enhancing_target': tumor, 'FLAIR_abnormality': flair},
        source_compartments={'brain': brain, 'enhancing_target': source_tumor, 'FLAIR_abnormality': flair},
        brain_mask=brain, affine=np.eye(4) if affine is None else affine, frame=frame,
        source_refs=(SourceRef('generated', 'synthetic://working-grid-test', provenance='simulated'),),
        unknowns=('complete vascular anatomy unavailable',),
        metadata={'surgical_adapter': {'sourcePatient': 'generated-research-source',
            'sourceManifest': {'patient_id': 'generated-research-source',
                'series': [{'patient_id': 'generated-research-source', 'source_mirror_revision': 'fixture'}]},
            'ownershipPrecedence': ['brain', 'FLAIR_abnormality', 'enhancing_target'],
            'targetAffine': (np.eye(4) if affine is None else affine).tolist(), 'resolutionMm': 1.},
            'surgical_properties': {'tissuePhysicsV2': default_profile()},
            'anatomy_layers': {'brain': {'kind': 'brain', 'rehearsal_target': False},
                'enhancing_target': {'kind': 'tumor', 'rehearsal_target': True},
                'FLAIR_abnormality': {'kind': 'tumor', 'rehearsal_target': False}}})


@pytest.mark.parametrize('affine,frame', [
    (np.eye(4), 'RAS+'),
    (np.array([[0., -2., .2, 31.], [.7, 0., 0., -9.], [0., .3, 3., 44.], [0., 0., 0., 1.]]), 'RAS+'),
    (np.array([[-.8, .1, 0., -22.], [0., 1.7, .2, 11.], [0., 0., 2.3, 7.], [0., 0., 0., 1.]]), 'LPS+'),
])
def test_refinement_preserves_every_parent_cell_bounds_and_material_volume(affine, frame):
    parent = fixture_case(affine, frame=frame)
    parent_manifest = parent.to_manifest()
    child = refine_working_grid(parent)
    assert child.mri.shape == tuple(2*s for s in parent.mri.shape)
    assert child.case_id == parent.case_id+'-working-grid-2x'
    assert child.semantic_hash != parent.semantic_hash and child.planning_hash != parent.planning_hash
    assert parent.to_manifest() == parent_manifest
    assert child.source_refs == parent.source_refs and child.frame == parent.frame
    assert child.unknowns == parent.unknowns and child.context == parent.context
    assert child.revision == parent.revision+1
    np.testing.assert_allclose(child.affine[:3, :3], parent.affine[:3, :3]/2, atol=1e-14)
    np.testing.assert_allclose(child.affine[:3, 3], parent.affine[:3, 3]+parent.affine[:3, :3]@np.full(3, -.25), atol=1e-14)
    assert child.voxel_volume_mm3*8 == pytest.approx(parent.voxel_volume_mm3)
    for index in np.ndindex(parent.mri.shape):
        block = tuple(slice(2*i, 2*i+2) for i in index)
        assert np.all(child.mri[block] == parent.mri[index])
        for group in ('compartments', 'source_compartments'):
            for name, mask in getattr(parent, group).items():
                assert np.all(getattr(child, group)[name][block] == mask[index])
        for side in product((-.5, .5), repeat=3):
            parent_corner = np.asarray(index)+side
            child_corner = 2*np.asarray(index)+np.asarray(side)*2+.5
            np.testing.assert_allclose(child.voxel_to_world(child_corner), parent.voxel_to_world(parent_corner), atol=1e-12)
    for group in ('compartments', 'source_compartments'):
        for name, mask in getattr(parent, group).items():
            assert getattr(child, group)[name].sum()*child.voxel_volume_mm3 == pytest.approx(mask.sum()*parent.voxel_volume_mm3)
    assert child.brain_mask.sum()*child.voxel_volume_mm3 == pytest.approx(parent.brain_mask.sum()*parent.voxel_volume_mm3)
    parent_scene, child_scene = from_case(parent), from_case(child)
    expected_labels = parent_scene.labels.repeat(2, 0).repeat(2, 1).repeat(2, 2)
    np.testing.assert_array_equal(child_scene.labels, expected_labels)
    assert child_scene.identity != parent_scene.identity and child_scene.regions == parent_scene.regions
    for region in parent_scene.regions:
        assert np.count_nonzero(child_scene.labels == region['id'])*child_scene.voxel_mm3 == pytest.approx(np.count_nonzero(parent_scene.labels == region['id'])*parent_scene.voxel_mm3)
    assert child_scene.properties == parent_scene.properties
    assert default_target_names(child) == default_target_names(parent) == ['enhancing_target']
    for key, value in parent.metadata.items():
        assert thaw_json(child.metadata[key]) == thaw_json(value)
    refinement = thaw_json(child.metadata['working_grid_refinement'])
    assert refinement['parentCaseHash'] == parent.semantic_hash
    assert refinement['parentAffine'] == parent.affine.tolist()
    assert refinement['parentShape'] == list(parent.mri.shape) and refinement['factor'] == 2
    assert refinement['parentFrame'] == refinement['childFrame'] == parent.frame
    assert refinement['parentGrid']['frame'] == refinement['childGrid']['frame'] == parent.frame
    assert child_scene.source['workingGridRefinement'] == refinement
    assert refinement['sourceImageIsInterpolated'] is True and refinement['noNewAnatomy'] is True
    assert refinement['parentGrid']['affine'] == parent.affine.tolist()
    assert refinement['childGrid']['affine'] == child.affine.tolist()
    assert refinement['mapping']['samePhysicalCellUnion'] is True
    assert 'not a new observation' in refinement['mriSampling']
    assert refinement['clinicalValidation'] is False and refinement['nativeStateMigrated'] is False
    for array in (child.mri, child.affine, child.brain_mask, *child.compartments.values(), *child.source_compartments.values()):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_refined_case_persistence_retains_exact_new_identity_and_parent_provenance(tmp_path):
    parent = fixture_case()
    child = refine_working_grid(parent)
    parent_file = save_case(parent, tmp_path/'parent.ressectionlab')
    before = parent_file.read_bytes()
    path = save_case(child, tmp_path/'refined.ressectionlab')
    reopened = load_case(path)
    assert reopened.to_manifest() == child.to_manifest()
    assert reopened.semantic_hash == child.semantic_hash
    assert from_case(reopened).identity == from_case(child).identity
    assert parent_file.read_bytes() == before and load_case(parent_file).semantic_hash == parent.semantic_hash


@pytest.mark.parametrize('factor', [1, 3, 2.0, True, None])
def test_refinement_refuses_unsupported_factors(factor):
    with pytest.raises(ValueError, match='factor 2 only'):
        refine_working_grid(fixture_case(), factor=factor)


def test_cell_cap_is_checked_before_allocating_child_arrays(monkeypatch):
    parent = fixture_case()
    assert environment.MAX_WORKING_GRID_CELLS == 128_000_000
    monkeypatch.setattr(environment, 'MAX_WORKING_GRID_CELLS', parent.mri.size*8-1)
    with pytest.raises(ValueError, match='128 million'):
        refine_working_grid(parent)


def test_refinement_refuses_stale_grid_evidence_nonvoxel_properties_and_embedded_history():
    from resectionlab.structural_evidence import StructuralEvidence, structural_frame_hash
    parent = fixture_case()
    evidence = StructuralEvidence('generated-envelope', parent.brain_mask,
        array_digest(parent.mri), structural_frame_hash(parent), None, None,
        'a'*64, 'generated test mask', provenance='simulated')
    with pytest.raises(ValueError, match='Grid-bound evidence'):
        refine_working_grid(replace(parent, structural_evidence={evidence.evidence_id: evidence}))
    for extra in ({'surgical_properties': {'attachments': [{'cells': [1]}]}},
                  {'surgical_properties': {'cusaV3': {'schema': 'not-remapped'}}},
                  {'surgical_toolbox': {'branches': {'old': {'removed': [1]}}}}):
        metadata = thaw_json(parent.metadata)
        metadata.update(extra)
        with pytest.raises(ValueError, match='Nonvoxel|Embedded surgical'):
            refine_working_grid(replace(parent, metadata=metadata))
    with pytest.raises(ValueError, match='already has a refined'):
        refine_working_grid(refine_working_grid(parent))
    with pytest.raises(TypeError, match='CaseData'):
        refine_working_grid(object())
