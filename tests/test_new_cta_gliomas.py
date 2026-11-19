"""Small offline checks for CTA source binding, generation, and portable studies."""

from pathlib import Path
import sys
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import pytest

from resectionlab.anatomy_package import open_package
from resectionlab.anatomy_planning import native_target_options
from resectionlab.data_root import atomic_json, sha256_file
from resectionlab.organic_tumors import signed_domain
from resectionlab.patient_anatomy import PatientImagingCollection, world

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_new_cta_gliomas as gliomas
import prepare_tair_cta
import tair00273_glioma_placement as tair


AFFINE = np.array([[-1.5, 0., 0., 82.], [0., -1.5, 0., 74.], [0., 0., 1.5, -54.], [0., 0., 0., 1.]])
SHAPE = (80, 76, 60)


def nifti(path, data, affine=AFFINE):
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units('mm')
    nib.save(image, path)
    return path


def add_context(study, data, key, kind, series, source=None):
    return study.add_layer(data, AFFINE, layer_id=key, source_series=series, kind=kind, role='context',
                           color='#dd8074', provenance={'kind': 'image_derived_estimate',
                           'source': source, 'evidence_use': 'context_only'})


@pytest.fixture
def isles_study(tmp_path):
    study = PatientImagingCollection.create(tmp_path / gliomas.SOURCE, gliomas.FAMILY)
    ncct = np.full(SHAPE, 34, dtype=np.int16)
    cta = np.full(SHAPE, 52, dtype=np.int16)
    ncct[16:22, 20:28, 28:36] = 5
    cta[57, 28, 35] = 170
    native = np.full((44, 42, 32), 67, dtype=np.int16)
    native_affine = np.diag([-3., -3., 3., 1.])
    native_affine[:3, 3] = (91., 81., -76.)
    for key, data, affine, modality in zip(gliomas.SCANS, (ncct, cta, native),
                                           (AFFINE, AFFINE, native_affine), ('CT', 'CTA', 'CTA')):
        original = nifti(tmp_path / f'{key}.nii.gz', data, affine)
        study.add_series(original, series_id=key, modality=modality)['source_license'] = gliomas.LICENSE
    study.set_registration('cta-ncct-space', np.eye(4), status='source-provided', method='Fixture paired grid')

    brain = np.zeros(SHAPE, dtype=np.uint8)
    brain[8:72, 8:68, 7:54] = 1
    skull = np.zeros_like(brain)
    skull[7, 8:68, 7:54] = 1
    skull[72, 8:68, 7:54] = 1
    cow = np.zeros_like(brain)
    cow[31:39, 30:40, 9:25] = 3
    cow[52:60, 29:43, 10:24] = 7
    cow_source = study.original(nifti(tmp_path / 'publisher-cow.nii.gz', cow))
    add_context(study, brain, gliomas.BRAIN, 'brain', 'ncct')
    add_context(study, skull, gliomas.SKULL, 'skull', 'ncct')
    add_context(study, (cow > 0).astype(np.uint8), gliomas.COW, 'artery', 'cta-ncct-space', cow_source)
    return study


def freeze_fixture(root, study, monkeypatch):
    snapshot = study.save('Seal source fixture')
    monkeypatch.setattr(gliomas, 'SNAPSHOT', snapshot)
    review = root / gliomas.REVIEW
    review.parent.mkdir(parents=True)
    atomic_json(review, {'family': gliomas.FAMILY, 'license': gliomas.LICENSE, 'role': 'DEVELOPMENT',
                         'native_manifest_sha256': sha256_file(study.root / 'patient-anatomy.json')})


@pytest.mark.parametrize('wrong_transform', (None, 'paired-offset', 'independent-mapped'))
def test_source_admission_preserves_independence_of_acquired_cta(tmp_path, monkeypatch, isles_study, wrong_transform):
    if wrong_transform == 'paired-offset':
        shift = np.eye(4)
        shift[0, 3] = 2.
        isles_study.set_registration('cta-ncct-space', shift, status='source-provided', method='Offset fixture')
    elif wrong_transform == 'independent-mapped':
        isles_study.set_registration('native-cta', np.eye(4), status='reviewed', method='False fusion fixture')
    freeze_fixture(tmp_path, isles_study, monkeypatch)

    if wrong_transform is not None:
        with pytest.raises(ValueError, match='paired reference grids and independent native CTA'):
            gliomas.source_study(tmp_path)
        return

    reopened, evidence = gliomas.source_study(tmp_path)
    np.testing.assert_array_equal(
        reopened.registration_matrix('cta-ncct-space', require_review=True, require_geometry=True), np.eye(4))
    with pytest.raises(ValueError, match='Missing source registration'):
        reopened.registration_matrix('native-cta')
    assert evidence['scans']['native-cta']['affine_ras_mm'] != evidence['scans']['ncct']['affine_ras_mm']
    assert evidence['scans']['native-cta']['source']['sha256'] != evidence['scans']['cta-ncct-space']['source']['sha256']
    assert evidence['layers'][gliomas.COW]['source_series'] == 'cta-ncct-space'


def test_tair_display_sampling_keeps_intensity_and_source_coordinates(monkeypatch):
    shape = (8, 7, 9)
    values = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    affine = np.array([[0., -.5, 0., 82.], [-.375, 0., 0., 74.], [0., 0., .3125, -54.], [0., 0., 0., 1.]])
    source = nib.Nifti1Image(values, affine)
    source.set_qform(affine, code=1)
    source.set_sform(affine, code=2)
    source.header.set_xyzt_units('mm')
    monkeypatch.setattr(prepare_tair_cta, 'MAX_ARRAY', shape[0] * shape[1] * 4 * 5)

    display, stride = prepare_tair_cta.display_image(source)

    assert stride == 2
    assert display.header.get_xyzt_units()[0] == 'mm'
    np.testing.assert_array_equal(display.dataobj, values[..., ::2])
    for name in ('qform', 'sform'):
        source_form, source_code = getattr(source, 'get_' + name)(coded=True)
        display_form, display_code = getattr(display, 'get_' + name)(coded=True)
        assert display_code == source_code
        np.testing.assert_allclose(world(display_form, (5, 2, 4)), world(source_form, (5, 2, 8)), atol=1e-6)
    np.testing.assert_array_equal(world(display.affine, (5, 2, 4)), world(source.affine, (5, 2, 8)))


@pytest.fixture
def isles_geometry(isles_study):
    domain, affine, offset, bounds, midline, clearance = gliomas.placement(isles_study)
    distance = signed_domain(domain, affine)
    geometry = gliomas.generate_variant(domain, affine, distance, bounds, midline, 1, [])
    return domain, affine, offset, bounds, midline, clearance, distance, geometry


def test_isles_generation_is_deterministic_and_uses_only_supported_tissue(isles_study, isles_geometry):
    domain, affine, offset, bounds, midline, clearances, distance, geometry = isles_geometry
    masks, recipe, region = geometry
    generated_again, recipe_again, region_again = gliomas.generate_variant(
        domain, affine, distance, bounds, midline, 1, [])
    assert recipe_again == recipe
    assert region_again == region
    for name, mask in masks.items():
        np.testing.assert_array_equal(generated_again[name], mask)
        assert mask.any()

    union = np.logical_or.reduce(list(masks.values()))
    assert union.sum() == sum(mask.sum() for mask in masks.values())
    assert domain[region][union].all()
    for clearance, threshold in zip(clearances, (4., 5.2, 1.4)):
        assert clearance[region][union].min() >= threshold
    full_region = tuple(slice(first, first + size) for first, size in zip(offset, domain.shape))
    for key in (gliomas.SKULL, gliomas.COW):
        assert not np.any(domain & isles_study.mask(isles_study.layer(key))[full_region])
    assert not domain[tuple(np.array((57, 28, 35)) - offset)]
    assert not domain[tuple(np.array((19, 24, 33)) - offset)]
    np.testing.assert_array_equal(affine[:3, 3], world(AFFINE, offset))
    assert recipe['center_ras_mm'][0] < midline[0] and recipe['center_ras_mm'][1] > midline[1]
    assert .60 <= recipe['normalized_superior_height'] <= .80


def test_tair_placement_excludes_source_boundaries_and_partial_vessel(tmp_path):
    study = PatientImagingCollection.create(tmp_path / 'tair', tair.FAMILY)
    signal = np.full(SHAPE, 35, dtype=np.int16)
    signal[61, 24, 36] = 210
    signal[16:22, 20:27, 30:36] = 4
    scan = study.add_series(nifti(tmp_path / 'sampled-cta.nii.gz', signal), series_id=tair.SERIES, modality='CTA')
    scan['intensity_units'] = 'HU'
    study.document['data_case'] = {'source_family_id': tair.FAMILY}

    brain = np.zeros(SHAPE, dtype=np.uint8)
    brain[0:76, 4:68, 7:54] = 1
    skull = np.zeros_like(brain)
    skull[76, 4:68, 7:54] = 1
    vessel = np.zeros_like(brain)
    vessel[60:64, 32, 30] = 1
    add_context(study, brain, tair.BRAIN, 'brain', tair.SERIES)
    add_context(study, skull, tair.SKULL, 'skull', tair.SERIES)
    add_context(study, vessel, tair.VESSEL, 'vessels', tair.SERIES)

    domain, affine, offset, _, _, clearance = tair.placement(study)
    assert domain.any()
    for grid, threshold in zip(clearance, (tair.SKULL_MARGIN_MM, tair.VESSEL_MARGIN_MM, tair.BRIGHT_MARGIN_MM)):
        assert grid[domain].min() >= threshold
        assert grid.max() <= tair.DISTANCE_CAP_MM
    restored = np.zeros(SHAPE, dtype=bool)
    restored[tuple(slice(first, first + size) for first, size in zip(offset, domain.shape))] = domain
    assert not restored[:3].any()
    assert not restored[61, 24, 36]
    assert not restored[19, 24, 33]
    assert not np.any(restored & (~brain.astype(bool) | skull.astype(bool) | vessel.astype(bool)))
    np.testing.assert_array_equal(affine[:3, 3], world(AFFINE, offset))


def test_portable_variant_cold_reopen_preserves_source_and_keeps_truth_outside(tmp_path, isles_study, isles_geometry):
    domain, affine, offset, _, _, clearances, _, geometry = isles_geometry
    isles_study.save('Seal source fixture')
    layer_ids = (gliomas.BRAIN, gliomas.SKULL, gliomas.COW)
    showcase = gliomas.copy_study(isles_study, tmp_path / 'showcase', gliomas.SHOWCASE, layer_ids, gliomas.LIMITS)
    source_manifest = (showcase.root / 'patient-anatomy.json').read_bytes()
    case_id = 'fixture-cta-organic-01'
    row = gliomas.archive_scenario(showcase, tmp_path, lambda path: gliomas.binding(tmp_path, path),
                                   showcase.array(showcase.series('ncct')['data']), affine, offset,
                                   domain, clearances, geometry, case_id, gliomas.FAMILY)

    archive = tmp_path / row['native_case']['path']
    with ZipFile(archive) as zipped:
        assert not any('research-truth' in name for name in zipped.namelist())
    cold = PatientImagingCollection(open_package(archive, tmp_path / 'fresh-cache').parent)
    assert cold.document['patient_id'] == case_id
    assert cold.identity() == row['snapshot_id']
    assert cold.document['transforms'] == showcase.document['transforms']
    assert set(layer['id'] for layer in cold.document['layers']) == {*layer_ids, gliomas.OBSERVATION}
    assert (showcase.root / 'patient-anatomy.json').read_bytes() == source_manifest
    for key in gliomas.SCANS:
        source = showcase.series(key)
        restored = cold.series(key)
        assert restored['affine'] == source['affine']
        assert restored['original'] == source['original']
        assert sha256_file(cold.path(restored['original']['path'])) == source['original']['sha256']
        np.testing.assert_array_equal(cold.array(restored['data']), showcase.array(source['data']))
    for key in layer_ids:
        np.testing.assert_array_equal(cold.mask(cold.layer(key)), showcase.mask(showcase.layer(key)))

    truth = nib.load(tmp_path / row['truth']['path'])
    labels = np.asarray(truth.dataobj)
    np.testing.assert_array_equal(truth.affine, AFFINE)
    assert set(np.unique(labels)) == {0, *gliomas.TRUTH.values()}
    assert row['truth']['planning_eligible'] is False
    target = cold.layer(gliomas.OBSERVATION)
    observation = cold.mask(target)
    assert target['role'] == 'target' and target['provenance']['kind'] == 'simulated'
    assert int(observation.sum()) == row['target']['positive_voxels']
    assert not np.array_equal(observation, np.isin(labels, (gliomas.TRUTH['synthetic_core'], gliomas.TRUTH['synthetic_rim'])))
    assert [option['layerId'] for option in native_target_options(cold)] == [gliomas.OBSERVATION]
