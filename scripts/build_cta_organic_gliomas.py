#!/usr/bin/env python3
"""Prepare organic glioma scenarios on reviewed native CTA hosts."""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from data_integrated_hosts import ORDER, UUID, binding, checked_file
from resectionlab.core import SourceRef, array_digest, thaw_json
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.organic_tumors import organic_geometry, signed_domain
from resectionlab.patient_anatomy import reconstruction_lock
from resectionlab.workspace_bundle import extension_layout, load_workspace, save_workspace, view_state

COLLECTION = 'prepared/cta-organic-gliomas-v1'
RELEASE = 'catalog/releases/2026-11-05-acquired-cta-anatomy-hosts-v1.json'
CALIBRATION = 'prepared/cta-organic-glioma-guidance-v1/ucsf-train-shape-summary.json'
SUBJECTS = ('006', '012', '014', '018')
OBSERVATION = 'synthetic_target_observation'
VERSION = 'cta-organic-gliomas-v1'
TRUTH = {'synthetic_core': 1, 'synthetic_rim': 2, 'synthetic_edema': 3}
LIMITS = (
    'Acquired CTA unchanged. The approximate generated contour is a simulated observation.',
    'Placement uses superior intracranial support with low-attenuation and medial exclusions. Exact CSF and ventricular borders are unavailable.',
    'Brain and bone are same-person estimates. Supplied vessels have partial coverage.',
    'No generated MRI, tractography, image enhancement, spatial deformation or modeled mass effect.',
    'The generated contour supports inspection and development planning. No reviewed instrument task or clinical validation.',
)
# Physical position, EDEN geometry profile and initial axis ratios.
SCENARIOS = (
    ('left', 'anterior', 'superior', 'circumscribed', (.76, 1.02, 1.28)),
    ('right', 'posterior', 'inferior', 'lobulated', (.78, .87, 1.47)),
    ('right', 'anterior', 'superior', 'broad-edema', (.67, 1.11, 1.35)),
    ('left', 'posterior', 'inferior', 'eden-v2', (.73, 1.10, 1.25)),
)


def read_json(path):
    return json.loads(path.read_text())


def checked_mask(root, row, host):
    image = nib.load(checked_file(root, row))
    data = np.asanyarray(image.dataobj)
    if (image.header.get_xyzt_units()[0] != 'mm' or image.shape != host.mri.shape
            or not np.array_equal(image.affine, host.affine) or not np.isin(data, (0, 1)).all()):
        raise ValueError('Expected a binary mask on the exact native millimeter CTA grid')
    return data.astype(bool)


def load_host(root, subject, release):
    host_id = f'TopCoW-{subject}-CTA-ANATOMY-v1'
    row = next(row for row in release['cases'] if row['case_id'] == host_id)
    if (row['decision'] != 'accepted_with_explicit_limitations' or row['role'] != 'TRAIN'
            or row['person_id'] != f'TopCoW:{subject}'):
        raise ValueError('Expected independently reviewed anatomy on a frozen TRAIN source family')
    review = {'path': row['review_path'], 'sha256': row['review_sha256']}
    checked_file(root, review)
    host, artifacts, workspace = load_workspace(checked_file(root, row))
    data = host.metadata['data_case']
    if (host.case_id != host_id or data['family_id'] != row['person_id'] or data['role'] != 'TRAIN'
            or host.frame != 'RAS+' or set(host.compartments) != set(ORDER) or artifacts
            or workspace is None or workspace['episode'] is not None or workspace['sources']
            or any(not np.array_equal(host.compartments[n], host.source_compartments[n]) for n in ORDER)):
        raise ValueError('Exact target-free native CTA host required')
    provenance = thaw_json(data['host_provenance'])
    source, supplied = provenance['source_CTA_and_v1_labels']
    ct = nib.load(checked_file(root, source))
    labels = nib.load(checked_file(root, supplied))
    if (ct.header.get_xyzt_units()[0] != 'mm' or labels.header.get_xyzt_units()[0] not in {'mm', 'unknown'}
            or any(image.shape != host.mri.shape or not np.array_equal(image.affine, host.affine)
                   for image in (ct, labels))):
        raise ValueError('Acquired source and supplied labels differ from the host native grid')
    values = np.asanyarray(labels.dataobj)
    if (not np.array_equal(ct.get_fdata(dtype=np.float32), host.mri)
            or not np.isin(values, np.arange(41)).all()
            or not np.array_equal((values >= 1) & (values <= 34), host.compartments[ORDER[2]])
            or not np.array_equal(values >= 35, host.compartments[ORDER[3]])):
        raise ValueError('Source CTA values or native V1 vascular memberships differ')
    originals = workspace['originals']
    if not {source['sha256'], supplied['sha256']} <= {item.sha256 for item in originals}:
        raise ValueError('Acquired original CTA and supplied vascular label bytes required')
    masks = provenance['anatomy_masks']
    for name, key in zip(ORDER, ('brain_estimate', 'observed_skull')):
        if not np.array_equal(checked_mask(root, masks[key], host), host.compartments[name]):
            raise ValueError(f'Existing {name} differs from the reviewed host')
    evidence = {**provenance, 'host': {'path': row['path'], 'sha256': row['sha256']}, 'host_review': review,
                'publisher_labels_header_units': labels.header.get_xyzt_units()[0],
                'publisher_labels_physical_basis': 'exact paired CTA grid in millimeters'}
    return host, originals, masks, evidence


def placement_domain(root, host, mask_rows):
    """Constrain all compartments to superior lateral, source-supported tissue."""
    a = host.affine
    if not np.array_equal(a[:3, :3], np.diag(np.diag(a[:3, :3]))) or a[2, 2] <= 0:
        raise ValueError('CTA placement requires a nonoblique RAS-aligned grid with superior third axis')
    enclosure = checked_mask(root, mask_rows['enclosed_coverage'], host)
    lower, upper = np.flatnonzero(enclosure.any(axis=(0, 1)))[[0, -1]]
    start = max(0, int(lower) - 1)
    stop = min(host.mri.shape[2], int(upper) + 2)
    region = np.s_[:, :, start:stop]
    brain = host.compartments['brain'][region]
    occupied_x = np.flatnonzero(brain.any(axis=(1, 2)))
    x = a[0, 0] * np.arange(brain.shape[0]) + a[0, 3]
    midline = float((x[occupied_x[0]] + x[occupied_x[-1]]) / 2)
    spacing = np.asarray(host.spacing_mm)
    attenuation = gaussian_filter(host.mri[region], .8 / spacing)
    support = enclosure[region] & brain & (attenuation >= 24.) & (attenuation <= 85.)
    support &= np.abs(x[:, None, None] - midline) >= 20.
    margin = int(np.ceil(3. / spacing[2]))
    origin = max(0, start - margin)
    buffered_region = np.s_[:, :, origin:min(host.mri.shape[2], stop + margin)]
    vessels = host.compartments[ORDER[2]][buffered_region] | host.compartments[ORDER[3]][buffered_region]
    support &= ~host.compartments[ORDER[1]][region]
    if vessels.any():
        clearance = distance_transform_edt(~vessels, sampling=spacing)
        support &= clearance[:, :, start-origin:stop-origin] > 3.
    support[:, :, [0, -1]] = False
    affine = a.copy()
    affine[:3, 3] = host.voxel_to_world((0, 0, start))
    return support, affine, start, midline


def generate(domain, affine, distance, midline, family, variant, previous, target_ml):
    side, ap, si, profile, aspect = SCENARIOS[variant - 1]
    seed = int.from_bytes(sha256(f'{VERSION}:{family}:{variant}'.encode()).digest()[:4], 'big')
    rng = np.random.default_rng(seed)
    spacing = nib.affines.voxel_sizes(affine)
    step = tuple(max(1, int(round(2.5 / size))) for size in spacing)
    sample = distance[tuple(slice(None, None, stride) for stride in step)]
    grid = np.argwhere(sample >= 4.5) * step
    x, y, z = nib.affines.apply_affine(affine, grid).T
    hemi = x < midline if side == 'left' else x > midline
    if not hemi.any():
        raise ValueError('No supported hemisphere placements')
    y_cut, z_cut = np.median(y[hemi]), np.median(z[hemi])
    group = grid[hemi & ((y > y_cut) if ap == 'anterior' else (y < y_cut))
                 & ((z > z_cut) if si == 'superior' else (z < z_cut))]
    radii = (target_ml * 1000 * 3 / (4 * np.pi)) ** (1 / 3) * np.asarray(aspect)
    half = np.ceil((max(radii) * 2.8 + 3.) / spacing).astype(int)
    for attempt, index in enumerate(rng.permutation(len(group))[:24]):
        center = group[index]
        point = nib.affines.apply_affine(affine, center)
        if any(np.linalg.norm(point - old) < 18. for old in previous):
            continue
        low = np.maximum(center - half, 0)
        high = np.minimum(center + half + 1, domain.shape)
        region = tuple(slice(lo, hi) for lo, hi in zip(low, high))
        crop_affine = affine.copy()
        crop_affine[:3, 3] = nib.affines.apply_affine(affine, low)
        try:
            masks, recipe = organic_geometry(domain[region], crop_affine, center_voxel=center - low,
                                             seed=seed + attempt, radii_mm=radii, signed=distance[region],
                                             profile=profile)
        except ValueError:
            continue
        volume = recipe['metrics']['volume_mm3']
        if not 420 * target_ml <= volume <= 1950 * target_ml:
            continue
        recipe.update(version=VERSION, profile=profile, placement=[side, ap, si],
                      requested_volume_ml=target_ml, base_seed=seed, attempt=attempt,
                      placement_midline_ras_x_mm=midline, minimum_center_separation_mm=18., no_mass_effect=True,
                      equivalent_spherical_diameter_mm=(6 * volume / np.pi) ** (1 / 3))
        return masks, recipe, region
    raise ValueError(f'No valid organic target survived 24 draws for {family} variant {variant}')


def observed_contour(target, support, spacing, *, sigma_mm=1.3):
    # Filter the target envelope, keeping distinct generator compartments out
    # of the planning workspace. The acquired CTA intensities remain exact.
    contour = (gaussian_filter(target.astype(np.float32), sigma_mm / np.asarray(spacing)) >= .54) & support
    dice = 2 * np.count_nonzero(contour & target) / (np.count_nonzero(contour) + np.count_nonzero(target))
    if not .75 <= dice < .995:
        raise ValueError('Approximate observation is too degraded or effectively exact')
    return contour, float(dice)


def scenario_case(host, case_id, observation, evidence):
    metadata = thaw_json(host.metadata)
    data = metadata['data_case']
    metadata.update(is_synthetic=True, selected_modality='CTA (ACQUIRED BASELINE)',
                    display_label='Acquired CTA with simulated tumor contour',
                    observation_role='baseline_reference_before_synthetic_contour',
                    annotation_role='Unchanged source anatomy and an approximate generated preoperative contour')
    metadata['anatomy_layers'][OBSERVATION] = {
        'kind': 'tumor', 'label': 'Simulated tumor contour', 'color': '#f18fbd', 'provenance': 'simulated',
        'coverage': LIMITS[0], 'review_status': 'synthetic measurement proxy; engineering QC only',
    }
    data.update(case_id=case_id, host_case_id=host.case_id, case_type='synthetic',
                classification='synthetic_scenario_on_acquired_host', role='DEVELOPMENT', source_family_role=data['role'],
                status='development_synthetic_observation', accepted=False, new_real_patient_count=0,
                synthetic_variant_count=1, clinical_validation=False, limitations=list(LIMITS),
                observations={'primary_image': 'baseline_acquired_background_geometry_only',
                              'target': 'simulated_partial_volume_contour'},
                capabilities=['native_source_inspection', 'synthetic_target_inspection', 'development_planning_observation'],
                simulation_admission='Development anatomy and simulated-contour inspection; instrument task unreviewed',
                generator_version=VERSION, host_package=evidence['host'])
    if 'split' in data:
        data['source_family_split'] = data.pop('split')
    data['components']['tumor'] = {'provenance': 'simulated', 'coverage': LIMITS[0]}
    metadata['simulation_admission'] = data['simulation_admission']
    metadata['surgical_adapter'].update(ownershipPrecedence=[ORDER[0], OBSERVATION, *ORDER[1:]],
                                        coverageLimitations=list(LIMITS))
    ref = SourceRef(OBSERVATION, f'synthetic://{case_id}/preoperative-contour',
                    array_digest(observation).removeprefix('sha256:'), native_frame=host.frame, provenance='simulated')
    compartments = {**host.compartments, OBSERVATION: observation}
    return replace(host, case_id=case_id, compartments=compartments, source_compartments=compartments,
                   metadata=metadata, source_refs=(*host.source_refs, ref), unknowns=(*host.unknowns, *LIMITS))


def write_truth(path, masks, region, host):
    labels = np.zeros(host.mri.shape, np.uint8)
    for name, value in TRUTH.items():
        labels[region][masks[name]] = value
    image = nib.Nifti1Image(labels, host.affine)
    image.header.set_xyzt_units('mm')
    nib.save(image, path)
    check = nib.load(path)
    if not np.array_equal(check.affine, host.affine) or not np.array_equal(np.asanyarray(check.dataobj), labels):
        raise ValueError('Generated truth failed native-grid restore')


def write_qa(path, host, region, masks, contour, recipe):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from skimage.measure import marching_cubes

    center = recipe['center_voxel']
    target = masks['synthetic_core'] | masks['synthetic_rim']
    full_center = np.asarray(center) + [part.start for part in region]
    figure = plt.figure(figsize=(12, 3.4), layout='constrained')
    layers = [(contour, '#fff1bd', 'Simulated contour'),
              (masks['synthetic_core'], '#f65493', 'Generated core'),
              (masks['synthetic_edema'], '#72c7fc', 'Generated edema')]
    for axis, plane in enumerate(('Sagittal', 'Coronal', 'Axial')):
        plot = figure.add_subplot(1, 4, axis + 1)
        image = np.take(host.mri[region], center[axis], axis=axis).T
        spacing = np.delete(host.spacing_mm, axis)
        plot.imshow(image, cmap='gray', origin='lower', vmin=7, vmax=105, aspect=spacing[1] / spacing[0])
        for part, color, _ in layers:
            section = np.take(part, center[axis], axis=axis).T
            if section.any() and not section.all():
                plot.contour(section, levels=[.5], linewidths=.7, colors=[color])
        plot.set_title(f'{plane} {full_center[axis]}', fontsize=9)
        plot.axis('off')
    plot = figure.add_subplot(1, 4, 4, projection='3d')
    low, high = np.asarray(recipe['metrics']['bounds_native_exclusive'])
    compact = np.pad(target[tuple(slice(x, y) for x, y in zip(low, high))], 2)
    vertices, faces, _, _ = marching_cubes(gaussian_filter(compact.astype(np.float32), .45), .5,
                                           spacing=host.spacing_mm)
    plot.plot_trisurf(*vertices.T, triangles=faces, color='#f191b9', linewidth=0, shade=True)
    plot.set_box_aspect(np.maximum(np.ptp(vertices, axis=0), 1))
    plot.set_title('Generated target surface', fontsize=9)
    plot.set_axis_off()
    placement = ' '.join(recipe['placement'])
    volume_ml = recipe['metrics']['volume_mm3'] / 1000
    figure.suptitle(f'{host.case_id} | {placement} | {volume_ml:.1f} mL', fontsize=11)
    figure.legend(handles=[Line2D([], [], color=color, label=label) for _, color, label in layers],
                  loc='outside lower center', ncol=3, fontsize=8, frameon=False)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def load_existing(root, directory, case_id, evidence, target_ml):
    path = directory / case_id / 'receipt.json'
    if not path.is_file():
        return None
    receipt = read_json(path)
    for key in ('native_case', 'truth', 'qa'):
        checked_file(root, receipt[key])
    if (receipt['case_id'] != case_id or receipt['recipe']['version'] != VERSION
            or receipt['source']['host'] != evidence['host']
            or receipt['source']['statistical_guidance'] != evidence['statistical_guidance']
            or receipt['recipe']['requested_volume_ml'] != target_ml):
        raise ValueError(f'Existing immutable scenario differs: {case_id}')
    return receipt


def save_variant(root, final, case_id, host, originals, evidence, geometry, domain, z_start, *, observation_sigma_mm=1.3):
    masks, recipe, local_region = geometry
    region = (*local_region[:2], slice(local_region[2].start + z_start, local_region[2].stop + z_start))
    support = domain[local_region]
    target = masks['synthetic_core'] | masks['synthetic_rim']
    if (np.any(masks['synthetic_core'] & masks['synthetic_rim']) or np.any(target & masks['synthetic_edema'])
            or any(np.any(mask & ~support) or np.any(mask & ~host.compartments['brain'][region])
                   or any(np.any(mask & host.compartments[key][region]) for key in ORDER[1:])
                   for mask in masks.values())):
        raise ValueError('Generated compartments fail the exclusive source-tissue gate')
    contour, dice = observed_contour(target, support, host.spacing_mm, sigma_mm=observation_sigma_mm)
    observation = np.zeros(host.mri.shape, bool)
    observation[region] = contour
    case = scenario_case(host, case_id, observation, evidence)
    state = view_state(case, [])
    state['states']['primary']['cursor'] = recipe['center_ras_mm']
    state['states']['primary'].setdefault('viewer', {})['volumeRendering'] = {'enabled': False}
    with TemporaryDirectory(prefix=f'.{case_id}-', dir=final.parent) as tmp:
        staging = Path(tmp) / case_id
        staging.mkdir()
        native = staging / (case_id + '.ressectionlab')
        truth = staging / 'research-truth.nii.gz'
        qa = staging / 'source-shape-qa.png'
        session = save_workspace(case, native, sources=[], imaging=state, episode=None, selection=None,
                                 artifacts={}, originals=originals)
        reopened, artifacts, restored = load_workspace(native)
        with ZipFile(native) as archive:
            raw = extension_layout(archive)
        if (reopened.semantic_hash != case.semantic_hash or raw['sessionHash'] != session or artifacts
                or restored is None or restored['episode'] is not None or restored['sources']
                or restored['originals'] != originals or not np.array_equal(reopened.mri, host.mri)
                or set(reopened.compartments) != {*ORDER, OBSERVATION}
                or set(reopened.source_compartments) != {*ORDER, OBSERVATION}):
            raise ValueError('Production reopen did not preserve host or observation isolation')
        for name in ORDER:
            if not (np.array_equal(reopened.compartments[name], host.compartments[name]) and
                    np.array_equal(reopened.source_compartments[name], host.source_compartments[name])):
                raise ValueError(f'Native project changed intact source tissue {name}')
        del reopened, restored
        write_truth(truth, masks, region, host)
        write_qa(qa, host, region, masks, contour, recipe)

        def destination_record(path):
            return {'path': str((final / path.name).relative_to(root)),
                    'sha256': sha256_file(path), 'bytes': path.stat().st_size}

        offset = [int(part.start) for part in region]
        coordinates = np.argwhere(target) * np.asarray(host.spacing_mm)
        covariance = np.linalg.eigvalsh(np.cov(coordinates, rowvar=False))
        receipt = {'schema': VERSION, 'case_id': case_id, 'person_id': host.metadata['data_case']['family_id'],
                   'role': 'DEVELOPMENT', 'source_family_role': 'TRAIN', 'source': evidence,
                   'native_case': destination_record(native), 'session_hash': session, 'case_hash': case.semantic_hash,
                   'truth': {**destination_record(truth), 'label_codes': TRUTH, 'planning_eligible': False},
                   'qa': destination_record(qa), 'clinical_validation': False, 'new_real_patient_count': 0,
                   'source_volume_hash': array_digest(host.mri), 'source_affine_ras_mm': host.affine.tolist(),
                   'native_grid_shape': list(host.mri.shape), 'native_grid_spacing_mm': list(host.spacing_mm),
                   'source_membership_hashes': {name: array_digest(host.compartments[name]) for name in ORDER},
                   'recipe': {**recipe, 'local_crop_origin_native_voxel': offset,
                              'center_full_native_voxel': (np.asarray(recipe['center_voxel']) + offset).tolist(),
                              'target_minor_major_pca_ratio': float(np.sqrt(covariance[0] / covariance[-1])),
                              'placement_support_rule': 'brain estimate intersected with superior enclosure, 0.8 mm-smoothed CTA values 24–85, 20 mm lateral to estimated midline; skull excluded and 3 mm from supplied vessels',
                              'outside_supported_tissue_voxels': 0,
                              'overlap_with_source_bone_or_supplied_vessels_voxels': 0,
                              'compartment_volume_ml': {name: float(mask.sum() * host.voxel_volume_mm3 / 1000)
                                                        for name, mask in masks.items()}},
                   'observation': {'name': OBSERVATION, 'volume_mm3': float(observation.sum() * host.voxel_volume_mm3),
                                   'array_hash': array_digest(observation), 'truth_overlap_dice': dice,
                                   'generated_image_signal': False, 'filter_sigma_mm': observation_sigma_mm, 'threshold': .54},
                   'production_workspace_reopen': True, 'limitations': list(LIMITS)}
        atomic_json(staging / 'receipt.json', receipt)
        staging.rename(final)
    return receipt


def register(root, rows, libraries):
    for library in libraries:
        library.mkdir(parents=True, exist_ok=True)
        for row in rows:
            source = checked_file(root, row['native_case'])
            link = library / source.name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(source)
            if not link.is_symlink() or link.resolve() != source:
                raise ValueError(f'Conflicting Cases entry: {link}')


def write_summary(root, directory):
    from PIL import Image, ImageOps

    rows = [read_json(path) for path in sorted(directory.glob('TopCoW-*/receipt.json'))]
    if not rows:
        raise ValueError('No verified synthetic scenarios to index')
    if len({row['truth']['sha256'] for row in rows}) != len(rows):
        raise ValueError('Duplicate generated truth in the native collection')
    width, height = 800, 236
    sheet = Image.new('RGB', (2 * width, ((len(rows) + 1) // 2) * height), '#e9edf2')
    for index, row in enumerate(rows):
        with Image.open(checked_file(root, row['qa'])) as image:
            panel = ImageOps.contain(image.convert('RGB'), (width - 6, height - 6), Image.Resampling.LANCZOS)
            sheet.paste(panel, ((index % 2) * width + 3, (index // 2) * height + 3))
    with TemporaryDirectory(prefix='.contact-sheet-', dir=directory) as temporary:
        staged = Path(temporary) / 'source-shape-contact-sheet.png'
        sheet.save(staged, optimize=True)
        contact_sheet = directory / staged.name
        staged.replace(contact_sheet)
    volumes = [row['recipe']['metrics']['volume_mm3'] / 1000 for row in rows]
    summary = {'schema': VERSION, 'role': 'DEVELOPMENT', 'clinical_validation': False,
               'statistical_guidance': rows[0]['source']['statistical_guidance'], 'new_real_patient_count': 0,
               'source_people': len({row['person_id'] for row in rows}), 'generated_scenarios': len(rows),
               'target_volume_ml_range': [min(volumes), max(volumes)], 'new_segmentation_inferences': 0,
               'bytes': sum(row[key]['bytes'] for row in rows for key in ('native_case', 'truth', 'qa')),
               'source_shape_contact_sheet': binding(root, contact_sheet),
               'cases': [{key: row[key] for key in ('case_id', 'person_id', 'native_case', 'truth', 'qa')} for row in rows]}
    atomic_json(directory / 'collection.json', summary)
    return summary


def build(root, subjects, libraries):
    root = checked_root(root, expected_uuid=UUID, required_bytes=3 * 1024**3)
    directory = safe_path(root, COLLECTION)
    release = safe_path(root, RELEASE)
    guidance = safe_path(root, CALIBRATION)
    release_binding = binding(root, release)
    guidance_binding = binding(root, guidance)
    guidance_data = read_json(guidance)
    volumes = guidance_data['suggested_target_volume_ml']
    if (guidance_data['role'] != 'TRAIN' or guidance_data['independent_evaluation_claim'] is not False
            or len(volumes) != len(SCENARIOS) or not all(2. <= volume <= 35. for volume in volumes)):
        raise ValueError('Expected four bounded sizes in the verified UCSF TRAIN shape guidance')
    cohort = read_json(release)
    with reconstruction_lock():
        directory.mkdir(parents=True, exist_ok=True)
        for subject in subjects:
            host, originals, source_masks, evidence = load_host(root, subject, cohort)
            evidence.update(release=release_binding, statistical_guidance=guidance_binding)
            domain, affine, start, midline = placement_domain(root, host, source_masks)
            previous, distance, signatures = [], None, set()
            for variant, volume in enumerate(volumes, 1):
                case_id = f'TopCoW-{subject}-CTA-ORGANIC-{variant:02d}-v1'
                row = load_existing(root, directory, case_id, evidence, volume)
                if row is None:
                    if distance is None:
                        distance = signed_domain(domain, affine)
                    geometry = generate(domain, affine, distance, midline, f'TopCoW:{subject}',
                                        variant, previous, volume)
                    row = save_variant(root, directory / case_id, case_id, host, originals,
                                       evidence, geometry, domain, start)
                    del geometry
                previous.append(np.asarray(row['recipe']['center_ras_mm']))
                if row['truth']['sha256'] in signatures:
                    raise ValueError('Duplicate generated truth in one source family')
                signatures.add(row['truth']['sha256'])
                ml = row['recipe']['metrics']['volume_mm3'] / 1000
                print(f'{case_id} · {ml:.2f} mL · {row["native_case"]["path"]}', flush=True)
                register(root, [row], libraries)
            del host, originals, domain, distance
        return write_summary(root, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--subject', action='append', choices=SUBJECTS)
    parser.add_argument('--register', type=Path, action='append', default=[])
    args = parser.parse_args()
    subjects = list(dict.fromkeys(args.subject or SUBJECTS[:3]))
    summary = build(args.root, subjects, args.register)
    print(json.dumps({key: value for key, value in summary.items() if key != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
