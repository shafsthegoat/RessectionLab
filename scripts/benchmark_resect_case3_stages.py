#!/usr/bin/env python3
"""Compare Case3 observed during/after ultrasound and transport both cavities."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

import nibabel as nib
import numpy as np

from acquire_resect_cavity import require_manifest
from benchmark_resect_surgical_change import CASE3_CAVITY, CASE3_WORKSPACE, CASE3_WORKSPACE_SHA, metrics
from data_integrated_hosts import UUID, binding, checked_file
from evaluate_resect_case3_image_registration import parse_full_tag
from mechanics_patient_comparison import _proper_rigid
from resectionlab.data_root import atomic_json, checked_root, safe_path
from resectionlab.imaging import _maximum_corner_displacement, inspect_nifti
from resectionlab.patient_anatomy import reconstruction_lock, resample, world
from resectionlab.workspace_bundle import OriginalFile, load_workspace, save_workspace, view_state
from resectionlab.workspace_imaging import DisplaySource

PREFIX = 'https://s3.nird.sigma2.no/archive-ro/5686d8fa-2003-4837-8e66-8e887fabe21e/RESECT/NIFTI/Case3/'
MIRROR = ('https://huggingface.co/datasets/MedOtter/RESECT-SEG/resolve/'
          'e86fb37dd93f7a9c64e48952f71410af59b04b9b/dataset/Case3/')
IMAGE_DIR = Path('sources/RESECT/v1/Case3')
AFTER_MASK = CASE3_CAVITY.with_name('Case3-US-after-resection.nii.gz')
STAGES = ('before', 'during', 'after')
CASE = 'RESECT-Case3-observed-cavity-stages-v1'
QC = Path('qc/surgical-change')
SOURCE = {
    str(AFTER_MASK): (44564, '2a8d15c1808df9a5028e4a3049a6af6164b1d300d6f08e5bdc632f7038687693',
                      MIRROR + AFTER_MASK.name),
    **{str(IMAGE_DIR / 'Landmarks' / f'Case3-beforeUS-{stage}US.tag'):
       (792, digest, PREFIX + f'Landmarks/Case3-beforeUS-{stage}US.tag')
       for stage, digest in (
           ('during', '73771147280c4e62fe1b9e95e45f202ad86164686bb9960de8132bbbc2518c9e'),
           ('after', '76310fc932976706411297074a9c2441f0979558998abc15ded25e7d35514a7f'))},
}
US_HASH = {
    'before': '9fbb4d63cb26d324c7c8281a2b2b932a55a5f3abf4a8fd571a0e0518ac996192',
    'during': '431ae1e23bd2984801c0b0d2720c9f4c240a404bab5fe37f5d8afdac521ddb20',
    'after': '83ce8eb0fa0182b1dde08f28f74e7308b932e95a4625f029e276fb4abdaee764',
}
RIGHTS = 'CC-BY-NC-SA-4.0'


def recover_sources(root):
    """Fetch just the absent exact releases, preserving valid local bytes."""
    records = {}
    for name, (size, digest, url) in SOURCE.items():
        path = safe_path(root, name)
        from_existing = path.is_file()
        if not from_existing:
            query = Request(url, headers={'User-Agent': 'RessectionLab source reproducibility'})
            with urlopen(query, timeout=25) as response:
                data = response.read(size + 1)
            if len(data) != size or sha256(data).hexdigest() != digest:
                raise ValueError(f'Publisher-bound bytes did not match: {path.name}')
            checked_root(root, expected_uuid=UUID)
            path.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=path.parent, prefix='.resect-source-') as scratch:
                staged = Path(scratch) / path.name
                staged.write_bytes(data)
                if not path.exists():
                    staged.rename(path)
        checked_file(root, {'path': name, 'sha256': digest, 'bytes': size})
        records[name] = {'bytes': size, 'sha256': digest, 'reused_local_bytes': from_existing,
                         'retrieval_url': url,
                         'publisher_url': 'https://osf.io/download/64cd21fcc7ab292b21d4e00a/?revision=2'
                         if path == root / AFTER_MASK else url}
    return records


def matched_points(root, scans):
    """Join by exact common before-US point triples and verify every stage's FOV."""
    lists = {}
    unmatched = {}
    for stage in ('during', 'after'):
        tag = safe_path(root, IMAGE_DIR / 'Landmarks' / f'Case3-beforeUS-{stage}US.tag')
        rows = parse_full_tag(tag.read_bytes())
        lists[stage] = {tuple(row[:3]): row[3:] for row in rows}
        if len(rows) != 11 or len(lists[stage]) != len(rows):
            raise ValueError('The shared Set 1 requires 11 unique before-US keys')
    common = lists['during'].keys() & lists['after'].keys()
    before = sorted(common)
    for stage, rows in lists.items():
        unmatched[stage] = len(rows.keys() - common)
    if len(before) != 10 or set(unmatched.values()) != {1}:
        raise ValueError('Expected ten exact common before-US keys and one unmatched per file')
    points = {'before': np.asarray(before), **{name: np.asarray([rows[key] for key in before])
                                               for name, rows in lists.items()}}
    for name, xyz in points.items():
        case = scans[name].case
        ijk = world(np.linalg.inv(case.affine), xyz)
        if (len(np.unique(xyz, axis=0)) != len(before) or np.any(ijk < -.5 - 1e-7)
                or np.any(ijk > np.asarray(case.mri.shape) - .5 + 1e-7)):
            raise ValueError(f'One or more source landmarks fall outside the native {name} lattice')
    a, = lists['during'].keys() - common
    b, = lists['after'].keys() - common
    return points, {'source_rows_per_file': 11, 'exact_common_keys': len(before),
                    'unmatched_by_source_file': unmatched,
                    'unmatched_before_coordinate_distance_mm': float(np.linalg.norm(np.subtract(a, b))),
                    'unmatched_after_destinations_used': False}


def prediction(stage, points):
    # Sort by the common before-US coordinates, so partition selection never
    # uses after positions, cavity visibility or point prediction errors.
    fit, held = np.arange(0, len(points['before']), 2), np.arange(1, len(points['before']), 2)
    rigid = _proper_rigid(points['during'][fit], points['after'][fit])
    if rigid['status'] != 'available':
        raise ValueError('Fixed conditioning points did not determine a proper rigid transform')
    matrix = np.eye(4)
    matrix[:3, :3] = rigid['rotation']
    matrix[:3, 3] = rigid['translation_mm']
    guesses = {'unchanged_before_position': points['before'][held],
               'carry_during_position_forward': points['during'][held],
               'conditioned_rigid_during_to_after': world(matrix, points['during'][held])}
    frozen = {'direction': 'during native RAS mm to after native RAS mm', 'matrix': matrix.tolist(),
              'sorted_source_indices_one_based': {'fit': (fit + 1).tolist(), 'held_out': (held + 1).tolist()},
              'partition': 'Lexicographic common before-US coordinates. Odd ranks fit, even ranks evaluate.',
              'held_out_prediction_ras_mm': {k: xyz.tolist() for k, xyz in guesses.items()}}
    atomic_json(stage / 'frozen-predictions.json', frozen)
    observed = points['after'][held]
    error = {k: metrics(np.linalg.norm(xyz - observed, axis=1).tolist()) for k, xyz in guesses.items()}
    observed_motion = np.linalg.norm(points['after'] - points['during'], axis=1)
    return matrix, {'held_out_error': error, 'matched_observed_during_to_after_motion_norms': metrics(observed_motion.tolist()),
                    'historical_case_role': 'TRAIN', 'split_scope': 'Withheld from this fit on previously exposed Case3'}


def cavity_layer(root, stage, source, declaration):
    path, expected = ((CASE3_CAVITY, declaration['files'][2]['sha256']) if stage == 'during'
                      else (AFTER_MASK, SOURCE[str(AFTER_MASK)][1]))
    path = checked_file(root, {'path': str(path), 'sha256': expected})
    header = inspect_nifti(path)
    original = source.case
    corners = _maximum_corner_displacement(original.affine, np.asarray(header['affine_ras_mm']), original.mri.shape)
    values = np.asarray(nib.load(path).dataobj)
    if (header['shape'] != list(original.mri.shape) or corners > declaration['maximum_grid_corner_difference_mm']
            or not np.array_equal(np.unique(values), [0, 1])):
        raise ValueError(f'{stage} mask does not match its native source')
    mask = values == 1
    layer = f'Observed visible cavity · {stage} resection'
    ref = replace(original.source_refs[0], source_id='supplied_target_annotation',
                  uri=declaration['files'][2]['source_url'] if stage == 'during'
                  else 'https://osf.io/download/64cd21fcc7ab292b21d4e00a/?revision=2',
                  sha256=expected, license=RIGHTS)
    derived = replace(original, source_refs=(*original.source_refs, ref), compartments={layer: mask},
                      source_compartments={layer: mask}, metadata={**original.metadata,
                          'annotationTimeRole': 'intraoperative_observed_evaluator_only',
                          'annotationPublisherDOI': declaration['annotation_release_doi'],
                          'annotationRightsSHA256': declaration['files'][0]['sha256'],
                          'annotationNativeGrid': header,
                          'display_label': f'US {stage} resection · supplied visible cavity'})
    return DisplaySource(derived, source.modality, 'source-provided'), path, mask, {
        'positive_voxels': int(mask.sum()), 'native_volume_mm3': float(mask.sum() * header['voxel_volume_mm3']),
        'source_mask_to_us_max_corner_mm': corners, 'annotation_sha256': expected,
        'shape': list(mask.shape), 'voxel_spacing_mm': header['spacing_mm']}


def stage_overlap(during, after, masks, transform):
    # Regrid an explicit hypothesis on the after lattice. Restrict the overlap
    # score to acquired nonzero scalar support in both scans, a visibility proxy.
    mapped_affine = transform @ during.affine
    moved = resample(masks['during'], mapped_affine, after.mri.shape, after.affine, order=0)
    supported = resample(during.mri > 0, mapped_affine, after.mri.shape, after.affine, order=0)
    shared = supported & (after.mri > 0)
    before_mask = moved & shared
    after_mask = masks['after'] & shared
    a, b = int(before_mask.sum()), int(after_mask.sum())
    overlap = int(np.count_nonzero(before_mask & after_mask))
    return {'common_nonzero_signal_voxels': int(shared.sum()),
            'mapped_during_positive_voxels': int(moved.sum()), 'mapped_during_positive_in_shared': a,
            'after_positive_in_shared': b, 'intersection_voxels': overlap,
            'dice_in_shared_signal_support': 2 * overlap / (a + b) if a + b else None,
            'uses_categorical_nearest_neighbor': True}


def plot(stage, scans, masks):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.3), layout='constrained')
    for row, key in enumerate(('during', 'after')):
        data, label = scans[key].case.mri, masks[key]
        points = np.argwhere(label)
        low = np.maximum(0, points.min(0) - 12)
        high = np.minimum(data.shape, points.max(0) + 13)
        span = tuple(slice(a, b) for a, b in zip(low, high))
        crop, contour = data[span], label[span]
        bounds = np.percentile(crop[crop > 0], (2, 98))
        for axis, panel in enumerate(axes[row]):
            index = int(np.argmax(np.bincount(points[:, axis]))) - low[axis]
            spacing = np.delete(scans[key].case.spacing_mm, axis)
            panel.imshow(np.take(crop, index, axis=axis).T, origin='lower', cmap='gray',
                         vmin=bounds[0], vmax=bounds[1], aspect=spacing[1] / spacing[0])
            panel.contour(np.take(contour, index, axis=axis).T, [.5], colors=['#67dfce'], linewidths=[1])
            panel.set_title(f'{key.capitalize()} · native axis {axis}')
            panel.set_axis_off()
    fig.suptitle('RESECT Case3 · visible cavity at two acquired ultrasound stages')
    fig.savefig(stage / 'native-cavity-stages.png', dpi=150)
    plt.close(fig)


def build(root, stage, restored_sources):
    declaration = require_manifest()
    source_package = checked_file(root, {'path': str(CASE3_WORKSPACE), 'sha256': CASE3_WORKSPACE_SHA})
    primary, artifacts, workspace = load_workspace(source_package)
    if (primary.metadata['data_case']['family_id'] != 'RESECT:Case3'
            or primary.metadata['data_case']['role'] != 'TRAIN' or primary.compartments
            or artifacts or not workspace or workspace['episode'] is not None):
        raise ValueError('Expected the sealed Case3 source-only MRI and ultrasound study')
    scans, image_bindings = {}, {}
    for name in STAGES:
        source, = (s for s in workspace['sources'] if s.case.metadata.get('acquisition_stage') == f'intraoperative-{name}')
        path = checked_file(root, {'path': str(IMAGE_DIR / f'Case3-US-{name}.nii.gz'), 'sha256': US_HASH[name]})
        image = nib.load(path)
        if (source.case.source_refs[0].sha256 != US_HASH[name] or source.case.compartments
                or not np.array_equal(image.affine, source.case.affine)
                or not np.array_equal(image.get_fdata(dtype=np.float32), source.case.mri)):
            raise ValueError(f'Saved {name} ultrasound differs from its acquired source')
        scans[name] = source
        image_bindings[name] = binding(root, path)
    points, join = matched_points(root, scans)
    matrix, correspondence = prediction(stage, points)
    correspondence['join'] = join
    masks, observations, originals = {}, {}, []
    rights = Path(__file__).resolve().parents[1] / 'artifacts/resect-component-admission-v1/rights-review-02/README.txt'
    if sha256(rights.read_bytes()).hexdigest() != declaration['files'][0]['sha256']:
        raise ValueError('Expected exact publisher annotation terms and version')
    for name in ('during', 'after'):
        scans[name], path, masks[name], observations[name] = cavity_layer(root, name, scans[name], declaration)
        originals.append(OriginalFile.from_path(path, role='annotation'))
    overlap = {name: stage_overlap(scans['during'].case, scans['after'].case, masks, affine)
               for name, affine in (('nominal_acquired_RAS_identity', np.eye(4)), ('conditioned_rigid', matrix))}
    # The full source workspace remains available for FLAIR and before-US.
    # Both annotated volumes at native resolution fit the display budget as
    # two display-only companions to the original MRI reference.
    study_sources = [scans[name] for name in ('during', 'after')]
    state = view_state(primary, study_sources)
    for name in ('during', 'after'):
        identity = scans[name].identity(primary)
        state['states'][identity]['viewer'] = {'volumeRendering': {'enabled': False, 'preset': 'mri'}}
        state['states'][identity]['cursor'] = world(scans[name].case.affine, np.argwhere(masks[name]).mean(0)).tolist()
    archive = stage / (CASE + '.ressectionlab')
    save_workspace(primary, archive, sources=study_sources, imaging=state, episode=None, selection=None, artifacts={},
                   originals=[*workspace['originals'], *originals, OriginalFile.from_path(rights, role='annotation')])
    reopened, artifacts, restored = load_workspace(archive)
    if (reopened.semantic_hash != primary.semantic_hash or artifacts or restored['episode'] is not None
            or [s.case.semantic_hash for s in restored['sources']] != [s.case.semantic_hash for s in study_sources]):
        raise ValueError('Native reopen changed the source, observations or current state')
    for name in ('during', 'after'):
        source = next(s for s in restored['sources'] if s.case.semantic_hash == scans[name].case.semantic_hash)
        mask, = source.case.compartments.values()
        if source.manifest(reopened)['scope'] != 'display-only-native-grid' or not np.array_equal(mask, masks[name]):
            raise ValueError(f'Observed {name} mask changed or became a planning input')
        observations[name]['exact_native_mask_reopen'] = True
    plot(stage, scans, masks)
    return {'schema': 'resectionlab-case3-observed-stage-benchmark-v1', 'patient': 'RESECT:Case3',
            'role': 'TRAIN', 'sources': image_bindings, 'restored_sources': restored_sources,
            'source_workspace': binding(root, source_package), 'landmarks': correspondence,
            'stage_visibility': observations, 'cross_stage_mask_agreement': overlap,
            'image_license': 'CC-BY-4.0', 'annotation_license': RIGHTS,
            'annotation_release': declaration['annotation_release_doi'],
            'interpretation': 'Visible cavity at each acquisition. Overlap reflects stage and visibility differences, '
                              'including probe position and contrast. Neither registration accuracy nor removed-tissue volume '
                              'is inferred from cross-stage Dice.',
            'prediction_scope': 'Landmark-conditioned alignment on a previously exposed TRAIN case. '
                                'No simulated actuator, cavity prediction, or biomechanical displacement field.',
            'workspace_display': 'Original T1 reference and two independent annotated ultrasound stages; '
                                 'before-US and FLAIR remain in the original source-stage workspace.',
            'postoperative_observation_excluded_from_preoperative_case': True,
            'members': {path.name: sha256(path.read_bytes()).hexdigest()
                        for path in (archive, stage / 'native-cavity-stages.png', stage / 'frozen-predictions.json')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output-name', default='RESECT-Case3-observed-stages-20261118-v1')
    args = parser.parse_args()
    if not args.output_name or Path(args.output_name).name != args.output_name or args.output_name.startswith('.'):
        parser.error('Use one fresh output directory name under qc/surgical-change')
    root = checked_root(args.root, expected_uuid=UUID, required_bytes=512 * 1024**2)
    destination = safe_path(root, QC / args.output_name)
    with reconstruction_lock():
        if destination.exists():
            raise FileExistsError(destination)
        sources = recover_sources(root)
        with TemporaryDirectory(dir=destination.parent, prefix='.resect-stages-') as scratch:
            stage = Path(scratch) / 'study'
            stage.mkdir()
            report = build(root, stage, sources)
            atomic_json(stage / 'report.json', report)
            checked_root(root, expected_uuid=UUID)
            stage.rename(destination)
    print(destination / 'report.json')


if __name__ == '__main__':
    main()
