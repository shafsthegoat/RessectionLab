#!/usr/bin/env python3
"""Generate native synthetic glioma studies on the retained ISLES admission pair."""
from __future__ import annotations

import argparse
from copy import deepcopy
import errno
from hashlib import sha256
import os
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label

from audit_cta_organic_cohort import OUTPUT as PREVIOUS, measure
from build_cta_organic_gliomas import CALIBRATION, OBSERVATION, TRUTH, observed_contour, read_json
from data_integrated_hosts import UUID, binding, checked_file
from resectionlab.anatomy_package import open_package, referenced_files, save_package
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.organic_tumors import organic_geometry, signed_domain
from resectionlab.patient_anatomy import PatientImagingCollection, reconstruction_lock, world

OUTPUT = 'prepared/isles0146-cta-gliomas-v1'
SCHEMA = 'ressectionlab-new-cta-gliomas-v1'
FAMILY = 'ISLES24-v7:sub-stroke0146'
SOURCE = 'prepared/isles24-v7-native/sub-stroke0146'
REVIEW = 'qc/isles24-v7-native/sub-stroke0146/verification.json'
SNAPSHOT = '4e08cc0d00a066774edf3300504ca5e875e10db9fed0db270d5dbf4aeb2b9b2b'
LICENSE = 'CC-BY-NC-SA-4.0'
BRAIN = 'totalsegmentator-skull-13d01e3cd6a4-brain'
SKULL = 'totalsegmentator-skull-a64fb148aae8'
COW = 'publisher-estimated-cow'
SHOWCASE = 'ISLES24-0146-CTA-ANATOMY-v1'
SCANS = ('ncct', 'cta-ncct-space', 'native-cta')
TARGET = 'Simulated glioma contour'
LIMITS = [
    'The brain and skull are retained estimates. The scan is defaced and skull-base coverage is partial.',
    'The automated CoW covers selected arteries. Distal vessels, veins, and functional anatomy are unknown.',
    'The acquired CTA has no independently reviewed transform to the NCCT frame.',
    'The target is generated. Source signals, anatomy and acquisition timepoints are retained.',
    'No generated MRI, simulated enhancement or mass effect. Research preview; surgical outcomes unvalidated.',
]
# Laterality, anterior/posterior sector, relative superior height, existing profile,
# radius-volume guess and allowable measured core-and-rim mL.
RECIPES = (
    ('left', 'anterior', (.60, .80), 'circumscribed', 11., (7., 14.)),
    ('right', 'posterior', (.63, .82), 'lobulated', 18., (13., 22.)),
    ('right', 'anterior', (.42, .62), 'broad-edema', 25., (19., 31.)),
    ('left', 'posterior', (.37, .58), 'eden-v2', 32., (25., 40.)),
)


def source_study(root):
    directory = safe_path(root, SOURCE)
    study = PatientImagingCollection(directory)
    review_path = safe_path(root, REVIEW)
    review = read_json(review_path)
    manifest = binding(root, directory / 'patient-anatomy.json')
    if (study.identity() != SNAPSHOT or study.document['patient_id'] != FAMILY or review['family'] != FAMILY
            or manifest['sha256'] != review['native_manifest_sha256'] or review['license'] != LICENSE
            or review['role'] != 'DEVELOPMENT' or study.document['reference_series'] != 'ncct'):
        raise ValueError('Expected the reviewed, frozen ISLES admission pair')
    evidence = {'manifest': manifest, 'review': binding(root, review_path), 'scans': {}, 'layers': {}}
    for key in SCANS:
        row = study.series(key)
        path = study.path(row['original']['path'])
        image = nib.load(path)
        original = binding(root, path)
        if (original['sha256'] != row['original']['sha256'] or row['source_license'] != LICENSE
                or list(image.shape) != row['data']['shape'] or not np.array_equal(row['affine'], image.affine)):
            raise ValueError(f'Acquired original changed: {key}')
        evidence['scans'][key] = {'source': original, 'array': row['data'], 'affine_ras_mm': row['affine']}
    ncct, cta = (study.series(key) for key in SCANS[:2])
    transforms = study.document['transforms']
    if (not np.array_equal(ncct['affine'], cta['affine']) or ncct['data']['shape'] != cta['data']['shape']
            or transforms[cta['id']]['status'] != 'source-provided'
            or not np.array_equal(transforms[cta['id']]['matrix'], np.eye(4)) or SCANS[2] in transforms):
        raise ValueError('Expected exact paired reference grids and independent native CTA')
    for key, source in ((BRAIN, 'ncct'), (SKULL, 'ncct'), (COW, 'cta-ncct-space')):
        row = study.layer(key)
        if (row['role'] != 'context' or row['provenance'].get('evidence_use') != 'context_only'
                or row['source_series'] != source or row['data']['shape'] != ncct['data']['shape']
                or not np.array_equal(row['affine'], ncct['affine'])):
            raise ValueError(f'Layer does not match the reference lattice and context scope: {key}')
        evidence['layers'][key] = {'array': row['data'], 'source_series': source,
                                  'origin': row['provenance']['kind'], 'original': row['provenance']['source']}
    cow = study.layer(COW)
    raw = np.asanyarray(nib.load(study.path(cow['provenance']['source']['path'])).dataobj)
    mask = study.mask(cow) > 0
    if not np.array_equal(raw > 0, mask) or int(mask.sum()) != 2848 or label(mask)[1] != 2:
        raise ValueError('The exact supplied CoW union or connectivity changed')
    return study, evidence


def placement(study):
    scan = study.series('ncct')
    affine = np.asarray(scan['affine'])
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if not np.array_equal(affine[:3, :3], np.diag(np.diag(affine[:3, :3]))) or affine[2, 2] <= 0:
        raise ValueError('Expected the pinned orthogonal NCCT coordinate directions')
    brain = study.mask(study.layer(BRAIN)) > 0
    points = np.argwhere(brain)
    low, high = points.min(0), points.max(0) + 1
    bounds = world(affine, np.stack((low, high - 1)))
    mid = bounds.mean(0)
    pad = np.ceil(10 / spacing).astype(int)
    start = np.maximum(low - pad, 0)
    stop = np.minimum(high + pad, brain.shape)
    crop = tuple(slice(a, b) for a, b in zip(start, stop))
    local = affine.copy()
    local[:3, 3] = world(affine, start)
    ncct = study.array(scan['data'])[crop]
    cta = study.array(study.series('cta-ncct-space')['data'])[crop]
    brain = brain[crop].copy()
    skull = study.mask(study.layer(SKULL))[crop] > 0
    cow = study.mask(study.layer(COW))[crop] > 0
    smooth = gaussian_filter(ncct, .8 / spacing)
    skull_distance = distance_transform_edt(~skull, sampling=spacing).astype(np.float32)
    cow_distance = distance_transform_edt(~cow, sampling=spacing).astype(np.float32)
    bright_distance = distance_transform_edt(~((cta >= 130.) & brain), sampling=spacing).astype(np.float32)
    x = local[0, 0] * np.arange(brain.shape[0]) + local[0, 3]
    z = local[2, 2] * np.arange(brain.shape[2]) + local[2, 3]
    # Intensity and anatomical exclusions are tissue-support proxies.
    domain = (brain & (ncct >= 18.) & (ncct <= 85.) & (smooth >= 24.) & (smooth <= 65.)
              & (skull_distance >= 4.) & (cow_distance >= 5.2) & (bright_distance >= 1.4))
    domain &= np.abs(x[:, None, None] - mid[0]) >= 16.
    domain &= z[None, None, :] >= bounds[0, 2] + .20 * np.ptp(bounds[:, 2])
    for axis in range(3):
        domain[(slice(None),) * axis + (0,)] = False
        domain[(slice(None),) * axis + (-1,)] = False
    return domain, local, start, bounds, mid, (skull_distance, cow_distance, bright_distance)


def generate_variant(domain, affine, distance, bounds, mid, number, prior, *, family=FAMILY):
    side, ap, heights, profile, requested, sizes = RECIPES[number - 1]
    seed = int.from_bytes(sha256(f'{SCHEMA}:{family}:{number}'.encode()).digest()[:4], 'big')
    rng = np.random.default_rng(seed)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    step = np.maximum(np.rint(2.8 / spacing).astype(int), 1)
    sampled = distance[tuple(slice(None, None, n) for n in step)]
    candidates = np.argwhere(sampled >= 5.) * step
    x, y, z = world(affine, candidates).T
    frac_z = (z - bounds[0, 2]) / np.ptp(bounds[:, 2])
    matching = ((x < mid[0]) if side == 'left' else (x > mid[0]))
    matching &= (y > mid[1]) if ap == 'anterior' else (y < mid[1])
    matching &= (frac_z >= heights[0]) & (frac_z <= heights[1])
    candidates = candidates[matching]
    score = distance[tuple(candidates.T)] + rng.uniform(0, 4, len(candidates))
    sites = []
    for rank in np.argsort(-score):
        center = candidates[rank]
        point = world(affine, center)
        if any(np.linalg.norm(point - old) < 22. for old in prior):
            continue
        if all(np.linalg.norm(point - old) >= 8. for _, old in sites):
            sites.append((center, point))
        if len(sites) == 12:
            break
    aspects = ((.88, .96, 1.18), (.76, 1.03, 1.27), (.77, .99, 1.24), (.86, .94, 1.22))
    radius = (requested * 1000 * 3 / (4 * np.pi)) ** (1 / 3)
    half = np.ceil(42. / spacing).astype(int)
    refusals = []
    for index, (center, point) in enumerate(sites):
        low = np.maximum(center - half, 0)
        high = np.minimum(center + half + 1, domain.shape)
        region = tuple(slice(a, b) for a, b in zip(low, high))
        local = affine.copy()
        local[:3, 3] = world(affine, low)
        radii = radius * np.asarray(aspects[number - 1]) * (1. + .07 * (index % 3 - 1))
        try:
            masks, generated = organic_geometry(domain[region], local, center_voxel=center - low,
                                                 seed=seed + index, radii_mm=radii,
                                                 signed=distance[region], profile=profile)
        except ValueError as error:
            refusals.append(str(error))
            continue
        target = masks['synthetic_core'] | masks['synthetic_rim']
        volume = generated['metrics']['volume_mm3'] / 1000
        edema_fraction = int(masks['synthetic_edema'].sum()) / int((target | masks['synthetic_edema']).sum())
        if not sizes[0] <= volume <= sizes[1] or not .28 <= edema_fraction <= .78:
            refusals.append(f'Volume {volume:.2f} mL, edema fraction {edema_fraction:.3f}')
            continue
        generated.update(base_seed=seed, attempt=index, profile=profile, laterality=side, ap_sector=ap,
                         normalized_superior_height=float((point[2] - bounds[0, 2]) / np.ptp(bounds[:, 2])),
                         requested_radius_volume_ml=requested, achieved_volume_bin_ml=list(sizes), mass_effect=False,
                         edema_fraction=edema_fraction)
        return masks, generated, region
    raise ValueError(f'Variant {number} failed all {len(sites)} bounded physical placements: {refusals}')


def renamed_study(source, case_id):
    snapshot = deepcopy(source.document)
    snapshot.pop('snapshot_id', None)
    snapshot.update(patient_id=case_id, history=[], revision=0)
    for row in snapshot['series'] + snapshot['layers']:
        row['patient_id'] = case_id
    return snapshot


def copy_study(source, directory, title, layers, limits):
    document = renamed_study(source, title)
    document['display'] = {'initial_layers': list(layers), 'initial_preset': 'Structures'}
    document['missing'].extend(item for item in limits if item not in document['missing'])
    directory.mkdir()
    study = PatientImagingCollection(directory, document)
    for name, digest in set(referenced_files(document)):
        target = study.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source.path(name), target)
        if sha256_file(target) != digest:
            raise ValueError(f'Copied study asset differs from the sealed source: {name}')
    study.save('Prepare intact source anatomy and limited vascular context for inspection')
    return study


def export_variant(showcase, destination, case_id, family, observation, recipe):
    snapshot = renamed_study(showcase, case_id)
    snapshot['display']['initial_layers'].append(OBSERVATION)
    snapshot['display']['cursor_ras_mm'] = recipe['center_ras_mm']
    snapshot['development_scenario'] = {'source_family_id': family, 'source_family_role': 'DEVELOPMENT',
                                        'role': 'DEVELOPMENT', 'simulated_image_signal': False, 'mass_effect': False}
    archive = destination / (case_id + '.rsworkspace')
    with TemporaryDirectory(dir=destination, prefix='.native-variant-') as scratch:
        stage = Path(scratch) / 'stage'
        stage.mkdir()
        case = PatientImagingCollection(stage, snapshot)
        for name, _ in set(referenced_files(snapshot)):
            source = showcase.path(name)
            output = case.path(name)
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, output)
            except OSError as error:
                if error.errno not in {errno.EXDEV, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM}:
                    raise
                copyfile(source, output)
        reference = snapshot['reference_series']
        target = case.add_layer(observation.astype(np.uint8), case.series(reference)['affine'], layer_id=OBSERVATION,
                                label=TARGET, source_series=reference, kind='synthetic-target', role='target', color='#f18fbd',
                                provenance={'kind': 'simulated', 'source_family_id': family, 'generation': SCHEMA,
                                            'method': 'Approximate simulated partial-volume contour'})
        target['coverage'] = 'Approximate generated contour on the physical reference grid'
        case.save('Publish the simulated contour and exact source members for portable export')
        save_package(case, archive)
        payload = open_package(archive, Path(scratch) / 'verify')
        recovered = PatientImagingCollection(payload.parent)
        if (recovered.identity() != case.identity()
                or not np.array_equal(recovered.mask(recovered.layer(OBSERVATION)), observation)):
            raise ValueError('Portable reopen changed the native study or simulated contour')
        with ZipFile(archive) as portable:
            if any('research-truth' in name for name in portable.namelist()):
                raise ValueError('Privileged research truth reached the native app package')
    return archive, case.identity(), target


def archive_scenario(showcase, stage, record, scan_data, affine, offset, domain, clearances, geometry, case_id, family):
    from types import SimpleNamespace
    from build_cta_organic_gliomas import write_qa

    masks, recipe, local_region = geometry
    target = masks['synthetic_core'] | masks['synthetic_rim']
    whole = target | masks['synthetic_edema']
    support = domain[local_region]
    if (any(np.any(mask & ~support) for mask in masks.values())
            or sum(int(mask.sum()) for mask in masks.values()) != int(whole.sum())):
        raise ValueError('Exact compartments are not exclusive on supported tissue')
    scan = showcase.series(showcase.document['reference_series'])
    native_affine = np.asarray(scan['affine'])
    spacing = np.linalg.norm(native_affine[:3, :3], axis=0)
    contour, dice = observed_contour(target, support, spacing)
    region = tuple(slice(part.start + off, part.stop + off) for part, off in zip(local_region, offset))
    observation = np.zeros(scan['data']['shape'], dtype=bool)
    observation[region] = contour
    folder = stage / case_id
    folder.mkdir()
    archive, snapshot, layer = export_variant(showcase, folder, case_id, family, observation, recipe)
    truth = np.zeros(observation.shape, dtype=np.uint8)
    for key, mask in masks.items():
        truth[region][mask] = TRUTH[key]
    truth_path = folder / 'research-truth.nii.gz'
    image = nib.Nifti1Image(truth, native_affine)
    image.header.set_xyzt_units('mm')
    nib.save(image, truth_path)
    restored = nib.load(truth_path)
    if not np.array_equal(restored.affine, native_affine) or not np.array_equal(restored.dataobj, truth):
        raise ValueError('Isolated truth did not reopen on the reference grid')
    qa = folder / 'source-overlays.png'
    recipe['placement'] = [recipe['laterality'], recipe['ap_sector'], f'z={recipe["normalized_superior_height"]:.2f}']
    write_qa(qa, SimpleNamespace(mri=scan_data, spacing_mm=spacing, case_id=case_id), region, masks, contour, recipe)
    distances = dict(zip(('skull', 'retained_vascular_context', 'bright_cta_exclusion'),
                         (float(grid[local_region][whole | contour].min()) for grid in clearances)))
    row = {'case_id': case_id, 'person_id': family, 'role': 'DEVELOPMENT', 'snapshot_id': snapshot,
           'native_case': record(archive), 'truth': {**record(truth_path), 'labels': TRUTH, 'planning_eligible': False},
           'overlays': record(qa), 'target': {'layer_id': OBSERVATION, 'label': TARGET,
               'array_sha256': layer['data']['sha256'], 'positive_voxels': int(observation.sum()),
               'dice_to_exact_core_and_rim': dice},
           'geometry': {'core_plus_enhancing': measure(target, affine), 'whole_including_edema': measure(whole, affine)},
           'minimum_exclusion_center_distance_mm': distances, 'recipe': recipe, 'production_reopen': True}
    atomic_json(folder / 'receipt.json', row)
    print(f'{case_id} {recipe["laterality"]} {recipe["ap_sector"]} '
          f'{row["geometry"]["core_plus_enhancing"]["volume_ml"]:.2f} mL', flush=True)
    return row


def scenario_distribution(rows, previous):
    result = {}
    for region in ('core_plus_enhancing', 'whole_including_edema'):
        result[region] = {}
        for metric in ('volume_ml', 'sphericity', 'minor_to_major_pca_ratio'):
            values = [row['geometry'][region][metric] for row in rows]
            result[region][metric] = {
                'new': {'min': min(values), 'median': float(np.median(values)), 'max': max(values)},
                'previous': previous['scenario_summary'][region][metric],
                'train_reference': previous['reference_summary'][region][metric],
            }
    return result


ISLES_POLICY = (
    'Retained brain with raw and smoothed NCCT intensity bounds, at least 16 mm lateral, above the bottom fifth '
    'of brain height, at least 4 mm from skull centres, 5.2 mm from CoW centres and 1.4 mm from bright CTA centres. '
    'These exclusions leave distal vessels and CSF borders unknown.'
)


def prepare(root, stage, final, showcase, evidence, place, family, prefix, source_license, policy, limits, showcase_record=None):
    guidance_path = safe_path(root, CALIBRATION)
    guidance = read_json(guidance_path)
    previous_path = safe_path(root, f'qc/{PREVIOUS}/report.json')
    previous = read_json(previous_path)
    bound_guidance = binding(root, guidance_path)
    if (guidance['n'] != 46 or guidance['role'] != 'TRAIN' or not previous['passed']
            or previous['shape_guidance'] != bound_guidance):
        raise ValueError('Expected the frozen 46-person TRAIN-only calibration')
    domain, affine, offset, bounds, mid, clearances = place(showcase)
    distance = signed_domain(domain, affine)
    scan = showcase.series(showcase.document['reference_series'])
    scan_data = np.load(showcase.path(scan['data']['path']), allow_pickle=False, mmap_mode='r')
    rows, centres = [], []

    def record(path):
        return {'path': str((final / path.relative_to(stage)).relative_to(root)), 'sha256': sha256_file(path),
                'bytes': path.stat().st_size}

    for number in range(1, len(RECIPES) + 1):
        geometry = generate_variant(domain, affine, distance, bounds, mid, number, centres, family=family)
        centres.append(np.asarray(geometry[1]['center_ras_mm']))
        case_id = f'{prefix}-CTA-GLIOMA-{number:02d}-v1'
        rows.append(archive_scenario(showcase, stage, record, scan_data, affine, offset, domain, clearances, geometry,
                                     case_id, family))
    report = {'schema': SCHEMA, 'promoted_source_people': 1, 'new_downloads': 0, 'new_inference_runs': 0,
              'new_synthetic_variants': len(rows), 'person_id': family, 'source': evidence, 'license': source_license,
              'showcase': showcase_record or record(showcase.root / 'patient-anatomy.json'), 'cases': rows,
              'limits': limits, 'guidance': bound_guidance, 'previous_audit': binding(root, previous_path),
              'distributions': scenario_distribution(rows, previous), 'placement_rule': policy}
    atomic_json(stage / 'collection.json', report)
    return report


def register(root, report, libraries):
    if not libraries:
        return
    files = {row['case_id'] + '.rsworkspace': checked_file(root, row['native_case']) for row in report['cases']}
    manifest = checked_file(root, report['showcase'])
    files[read_json(manifest)['patient_id'] + '.anatomy.json'] = manifest
    for library in set(path.resolve(strict=True) for path in libraries):
        for name, source in files.items():
            link = library / name
            if not link.is_symlink() and not link.exists():
                link.symlink_to(source)
            if link.resolve(strict=True) != source:
                raise ValueError(f'Cases registration conflicts with an existing entry: {link}')


def probe_guarded_rehearsal(root):
    """Check that standard planning admits the mapped ISLES contour with its source label.

    Owner decision 2026-11-27: acquired anatomy is admitted and labelled. The
    retained ``qc/isles0146-cta-guarded-refusal-v1.json`` records the earlier refusal.
    """
    from resectionlab.full_case_readiness import full_case_readiness
    from resectionlab.imaging import read_case_artifacts
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_environment import load_environment
    from resectionlab.surgical_patient import prepare_surgical_environment, surgical_environment_options
    from resectionlab.surgical_planning_admission import ACQUIRED, planning_admission, require_planning_admission

    cohort = safe_path(root, f'{OUTPUT}/collection.json')
    row = read_json(cohort)['cases'][1]
    archive = checked_file(root, row['native_case'])
    output = safe_path(root, 'qc/isles0146-cta-labelled-admission-v1.json')
    if output.exists():
        report = read_json(output)
        if report['input'] != row['native_case'] or report['planning_admission'].get('sourceClass') != ACQUIRED:
            raise ValueError('The retained admission proof belongs to a different case')
        return output
    with TemporaryDirectory(dir=output.parent, prefix='.isles-guarded-rehearsal-') as scratch:
        native = open_package(archive, Path(scratch) / 'native')
        study = PatientImagingCollection(native.parent)
        target = study.layer(OBSERVATION)
        source_snapshot = study.identity()
        if (source_snapshot != row['snapshot_id'] or target['data']['sha256'] != row['target']['array_sha256']
                or target['kind'] != 'synthetic-target' or target['provenance']['kind'] != 'simulated'):
            raise ValueError('Expected the sealed simulated contour on the ISLES reference')
        # The standard rehearsal importer uses the tumor field semantic. Only
        # the ephemeral input descriptor changes; the mask and provenance are exact.
        target['kind'] = 'tumor'
        study.save('Map the exact simulated contour to the standard rehearsal target field', expected=source_snapshot)
        selected = {'brain': BRAIN, 'bone': SKULL, 'tumor': OBSERVATION}
        options = surgical_environment_options(study)
        if options['referenceIssue'] or any(not any(
                (opt['role'], opt['layerId'], opt['status']) == (role, key, 'ready') for opt in options['layers'])
                for role, key in selected.items()):
            raise ValueError('Standard rehearsal setup did not admit the three source-bound input layers')
        prepared = prepare_surgical_environment(native, Path(scratch) / 'environment', layer_ids=selected)
        scene = load_environment(prepared['casePath'])
        toolbox = read_case_artifacts(prepared['casePath'])['surgical_toolbox']
        engine = SurgicalEngine.restore(scene, toolbox['branches']['main'])
        before = engine.snapshot()
        readiness = full_case_readiness(engine)
        admission = planning_admission(engine.scene, engine.protection)
        target_volume = prepared['environment']['target']['initialVolumeMm3']
        if (before['actions'] or not target_volume or toolbox['timeline']['main'] or not admission['available']
                or admission['sourceClass'] != ACQUIRED):
            raise ValueError('Expected a populated target and an unacted scene admitted with its acquired-source label')
        if (require_planning_admission(engine) != admission or engine.snapshot() != before
                or checked_file(root, row['native_case']) != archive):
            raise ValueError('The admission changed the unacted scene or differs from the source policy')
        report = {'schema': 'ressectionlab-new-cta-labelled-admission-v1', 'input': row['native_case'],
                  'source_snapshot': source_snapshot, 'temporary_mapped_snapshot': study.identity(),
                  'target_array_sha256': target['data']['sha256'], 'target_scene_volume_mm3': target_volume,
                  'scene_id': scene.identity, 'initial_state_id': engine.state_id, 'actions_before_and_after': [0, 0],
                  'planning_admission': admission, 'full_case_readiness': readiness,
                  'selected_layer_ids': selected}
        checked_root(root, expected_uuid=UUID)
        atomic_json(output, report)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--register', type=Path, action='append', default=[])
    parser.add_argument('--host', choices=('isles', 'tair'), default='isles')
    parser.add_argument('--guarded-rehearsal', action='store_true', help='Check labelled standard planning admission on ISLES variant 02')
    args = parser.parse_args()
    root = checked_root(args.root, expected_uuid=UUID, required_bytes=5 * 1024**3)
    with reconstruction_lock():
        if args.guarded_rehearsal:
            if args.host != 'isles':
                parser.error('--guarded-rehearsal applies to ISLES variant 02')
            print(probe_guarded_rehearsal(root))
            return
        if args.host == 'tair':
            import prepare_tair00273_showcase as tair
            import tair00273_glioma_placement as ta
            showcase_dir = safe_path(root, tair.DESTINATION)
            receipt_path = showcase_dir / 'receipt.json'
            receipt = read_json(receipt_path)
            showcase_record = binding(root, showcase_dir / 'patient-anatomy.json')
            study = PatientImagingCollection(showcase_dir)
            if (study.identity() != receipt['snapshot_id'] or receipt['source_person_id'] != ta.FAMILY
                    or receipt['manifest'] != showcase_record):
                raise ValueError('Expected the source-validated TAIR display showcase')
            evidence = {'manifest': showcase_record, 'review': binding(root, receipt_path)}
            parameters = (ta.placement, ta.FAMILY, 'TAIR-00273', 'CC-BY-4.0', ta.POLICY, tair.LIMITS, showcase_record)
            output = 'prepared/tair00273-cta-gliomas-v1'
        else:
            study, evidence = source_study(root)
            parameters = (placement, FAMILY, 'ISLES24-0146', LICENSE, ISLES_POLICY, LIMITS)
            output = OUTPUT
        destination = safe_path(root, output)
        if not destination.exists():
            with TemporaryDirectory(dir=destination.parent, prefix='.cta-gliomas-') as scratch:
                stage = Path(scratch) / 'cohort'
                stage.mkdir()
                if args.host == 'isles':
                    study = copy_study(study, stage / SHOWCASE, SHOWCASE, (BRAIN, SKULL, COW), LIMITS)
                report = prepare(root, stage, destination, study, evidence, *parameters)
                checked_root(root, expected_uuid=UUID)
                stage.rename(destination)
        else:
            report = read_json(destination / 'collection.json')
            if report['schema'] != SCHEMA or report['source'] != evidence:
                raise ValueError('Existing native cohort differs from its frozen source evidence')
            for row in report['cases']:
                for key in ('native_case', 'truth', 'overlays'):
                    checked_file(root, row[key])
        register(root, report, args.register)
    print(destination / 'collection.json')


if __name__ == '__main__':
    main()
