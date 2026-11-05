"""Physical synthetic target controls; no acquired subject qualification."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('integrated_cases', Path(__file__).parents[1] / 'scripts/data_integrated_cases.py')
cases = importlib.util.module_from_spec(spec); spec.loader.exec_module(cases)


def test_target_uses_physical_radii_and_source_handedness():
    affine = np.diag([-2., 1., .5, 1.]); affine[:3, 3] = [40., 5., -20.]
    center = (affine @ [10., 10., 10., 1.])[:3]
    mask = cases.ellipsoid((24, 24, 24), affine, center, [4., 3., 2.])
    assert mask[8, 10, 10] and mask[12, 10, 10]
    assert mask[10, 7, 10] and mask[10, 10, 6]
    assert not mask[7, 10, 10] and not mask[10, 6, 10] and not mask[10, 10, 5]


def test_target_cannot_be_clipped_to_source_crop():
    with pytest.raises(ValueError, match='beyond source coverage'):
        cases.ellipsoid((16, 16, 16), np.eye(4), [1., 8., 8.], [3., 3., 3.])


@pytest.mark.parametrize('radii', [[0, 1, 1], [-1, 1, 1], [1, float('nan'), 1]])
def test_invalid_physical_target_is_refused(radii):
    with pytest.raises(ValueError, match='positive'):
        cases.ellipsoid((16, 16, 16), np.eye(4), [8., 8., 8.], radii)


def test_native_builder_preserves_baseline_and_supplied_obstacles(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    import json
    import nibabel as nib
    from resectionlab.core import CaseData, SourceRef
    from resectionlab.workspace_bundle import save_workspace, load_workspace, view_state, OriginalFile
    shape = (20, 20, 20); affine = np.diag([-1., 1., 1., 1.]); affine[0, 3] = 19.
    original = tmp_path / 'baseline.nii.gz'
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    nib.save(nib.Nifti1Image(image, affine), original)
    source_hash = cases.sha256_file(original)
    vessel = np.zeros(shape, bool); vessel[3, 3, 3:15] = True
    host = CaseData(case_id='generated-host', mri=image, affine=affine,
        compartments={'arterial_source': vessel}, source_refs=(SourceRef('generated-baseline', original.as_uri(), source_hash),),
        metadata={'data_case': {'family_id': 'GENERATED:control', 'role': 'TRAIN'},
                  'anatomy_layers': {'arterial_source': {'kind': 'vessels'}}})
    source = tmp_path / 'source.ressectionlab'
    save_workspace(host, source, sources=[], imaging=view_state(host, []), episode=None, selection=None,
                   artifacts={}, originals=[OriginalFile.from_path(original, role='imaging')])
    source_before = source.read_bytes(); proposal = tmp_path / 'proposal'; proposal.mkdir()
    brain = np.zeros(shape, np.uint8); brain[5:15, 5:15, 5:15] = 1
    skull = np.zeros(shape, np.uint8); skull[1:3, 1:18, 1:18] = 1
    outputs = {}
    for name, array in [('brain_estimate', brain), ('observed_skull', skull), ('tumor_placement_support', brain)]:
        path = proposal / (name + '.nii.gz'); nib.save(nib.Nifti1Image(array, affine), path)
        outputs[name] = {'path': str(path.relative_to(tmp_path)), 'sha256': cases.sha256_file(path)}
    (proposal / 'provenance.json').write_text(json.dumps({'family_id': 'GENERATED:control',
        'source_image_sha256': source_hash, 'outputs': outputs, 'limitations': ['generated control only']}))
    monkeypatch.setattr(cases, 'checked_root', lambda *args, **kwargs: tmp_path)
    report = cases.build(source, proposal, [9., 10., 10.], [2., 2., 2.], 'test-v1', root=tmp_path)
    result, _, workspace = load_workspace(tmp_path / report['path'])
    assert source.read_bytes() == source_before
    assert np.array_equal(result.mri, image) and np.array_equal(result.affine, affine)
    assert np.array_equal(result.compartments['arterial_source'], vessel)
    assert workspace['originals'][0].sha256 == source_hash
    assert result.metadata['data_case']['observations']['primary_image'] == 'baseline_acquired_background_geometry_only'
    assert result.metadata['surgical_adapter']['ownershipPrecedence'] == ('brain', 'synthetic_target', 'estimated_skull', 'arterial_source')
    domain_path = tmp_path / 'generated-reviewed-domain.json'
    domain_path.write_text(json.dumps({'mode': 'restrict-unknown', 'shape': list(shape),
        'minimumVoxelBoundary': [1.5, 1.5, 1.5], 'maximumVoxelBoundary': [18.5, 18.5, 18.5],
        'reviewEvidenceSha256': 'a'*64, 'maskSha256': 'b'*64}))
    conditional_report = cases.build(source, proposal, [9., 10., 10.], [2., 2., 2.], 'conditional-test-v1',
        root=tmp_path, stage='conditional-post-scalp-exposure', reviewed_task_domain=domain_path)
    conditional, _, _ = load_workspace(tmp_path / conditional_report['path'])
    assert np.array_equal(conditional.mri, result.mri)
    assert all(np.array_equal(mask, conditional.compartments[name]) for name, mask in result.compartments.items())
    stage = conditional.metadata['integrated_rehearsal']
    assert stage['intact_head_access_supported'] is False
    assert stage['sourceImageRole'] == 'acquired_baseline_before_assumed_exposure'
    assert any('Dura mechanics/opening are not modeled' in assumption for assumption in stage['assumptions'])
    assert conditional.metadata['surgical_adapter']['reviewedTaskDomainV1']['shape'] == shape
    copy_report = cases.build(source, proposal, [9., 10., 10.], [2., 2., 2.], 'query-copy-v1',
        root=tmp_path, stage='conditional-post-scalp-exposure', reviewed_task_domain=domain_path,
        task_metadata_copy_of=tmp_path/conditional_report['path'])
    copied, _, _ = load_workspace(tmp_path/copy_report['path'])
    assert copied.metadata['data_case']['synthetic_variant_count'] == 0
    assert copy_report['new_synthetic_variant_count'] == 0
    with pytest.raises(ValueError, match='preserve every source and target array'):
        cases.build(source, proposal, [9., 10., 10.], [3., 3., 3.], 'false-copy-v1', root=tmp_path,
            stage='conditional-post-scalp-exposure', task_metadata_copy_of=tmp_path/conditional_report['path'])
