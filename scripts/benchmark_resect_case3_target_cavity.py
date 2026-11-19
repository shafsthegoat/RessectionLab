#!/usr/bin/env python3
"""Measure Case3 tumor and observed cavity overlap using independently held-out landmarks."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass

from benchmark_resect_case3_stages import (
    IMAGE_DIR, MIRROR, PREFIX, QC, RIGHTS, SOURCE, US_HASH, cavity_layer, recover_sources,
)
from benchmark_resect_surgical_change import CASE3_CAVITY, CASE3_WORKSPACE, CASE3_WORKSPACE_SHA, metrics, require_manifest
from data_integrated_hosts import UUID, binding, checked_file
from evaluate_resect_case3_image_registration import parse_full_tag
from mechanics_patient_comparison import _proper_rigid
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.imaging import _maximum_corner_displacement, inspect_nifti
from resectionlab.patient_anatomy import reconstruction_lock, resample, world
from resectionlab.workspace_bundle import OriginalFile, load_workspace, save_workspace, view_state
from resectionlab.workspace_imaging import DisplaySource, source_registration

CASE = 'RESECT-Case3-observed-target-cavity-v1'
FLAIR_CASE = 'RESECT-Case3-observed-flair-target-v1'
SCHEMA = 'resectionlab-case3-observed-target-cavity-v1'
OUTPUT = 'RESECT-Case3-observed-target-cavity-20261119-v1'
REPO = Path(__file__).resolve().parents[1]
TARGETS = {
    'flair': (CASE3_CAVITY.with_name('Case3-FLAIR-tumor.nii.gz'), 110906,
              '5a7bfd52adde03416f5b7dec9b23550c82d180cce40c6a9c73e62db857ce8835',
              'https://osf.io/download/64cd21fca2be6b3add31480b/?revision=1'),
    'before': (CASE3_CAVITY.with_name('Case3-US-before-tumor.nii.gz'), 32184,
               '0ed377835dedff666bbfc1ef9e37a29f135ce7921e7244bf4d82e02895182b92',
               'https://osf.io/download/64cd21fca2be6b3ada3146f6/?revision=2'),
}
TAG = {
    'flair_after': ('Case3-MRI-afterUS.tag', 1058, '8a1fcec7a7df9b2ad5f47d3cb08635444798e2060ec08209a4b0ea6237e6f45b'),
    'flair_before': ('Case3-MRI-beforeUS.tag', 1057, 'e1c9f9c648079a467f69920881440f88a2e517701478986fa025b455b1a32104'),
    'before_after': ('Case3-beforeUS-afterUS.tag', *SOURCE[str(IMAGE_DIR / 'Landmarks' / 'Case3-beforeUS-afterUS.tag')][:2]),
}
RELATIONS = (('flair_after', 'flair', 'after'), ('before_after', 'before', 'after'),
             ('flair_before', 'flair', 'before'))
PREVIEW_KEYS = ('flair', 'before', 'after')
PREVIEWS = {CASE: ('before', 'after'), FLAIR_CASE: ('flair',)}
SAVED_LOCALIZATION = {
    'receipt.json': 'c4918c44745d4159e64f8cc731c8ffdf8f0d01913daa5d1e5cd91eea51d5886b',
    'brain-mask.nii.gz': '2aa81407edaefa2b27976f9c6c2236cae3e214e299cc24cac01cd4ad7c053258',
    'output/tumor_mask.nii.gz': '157af98154cebf2851cb0ca757d377936dd1be2a67bda7250d25c3df8abd51ab',
}


def acquire(root):
    sources = {str(path): (size, dg, MIRROR + path.name) for path, size, dg, _ in TARGETS.values()}
    urls = {str(path): url for path, _, _, url in TARGETS.values()}
    sources.update({str(IMAGE_DIR / 'Landmarks' / name): (size, dg, PREFIX + 'Landmarks/' + name)
                    for name, size, dg in TAG.values()})
    return recover_sources(root, sources, urls)


SIGNAL = {
    'flair': ('Case3-FLAIR.nii.gz', 'cf4e3bb122722693cf626e0dcb6d43892c4e1092cf8fbc5c419741fa9ec4aafa'),
    **{name: (f'Case3-US-{name}.nii.gz', US_HASH[name]) for name in ('before', 'after')},
}
LABEL = {'flair': 'Supplied preoperative FLAIR tumor', 'before': 'Supplied pre-resection US tumor',
         'after': 'Observed visible cavity · after resection'}
STAGE = {'flair': 'preoperative', 'before': 'intraoperative-before', 'after': 'intraoperative-after'}
ANNOTATION_ROLE = {'flair': 'preoperative_observed_benchmark_only', 'before': 'intraoperative_observed_evaluator_only',
                   'after': 'intraoperative_observed_evaluator_only'}


def source_study(root):
    source_file = checked_file(root, {'path': str(CASE3_WORKSPACE), 'sha256': CASE3_WORKSPACE_SHA})
    primary, artifacts, workspace = load_workspace(source_file)
    if (primary.case_id != 'RESECT-Case3' or primary.metadata['data_case']['family_id'] != 'RESECT:Case3'
            or primary.metadata['data_case']['role'] != 'TRAIN' or primary.compartments
            or artifacts or not workspace or workspace['episode'] is not None):
        raise ValueError('Expected the sealed source-only Case3 MRI and ultrasound study')
    sources, image_records = {}, {}
    for key, (filename, digest) in SIGNAL.items():
        path = checked_file(root, {'path': str(IMAGE_DIR / filename), 'sha256': digest})
        source, = (row for row in workspace['sources'] if row.case.source_refs[0].sha256 == digest)
        image = nib.load(path)
        if (source.case.compartments or source.case.frame != 'RAS+' or image.header.get_xyzt_units()[0] != 'mm'
                or not np.array_equal(source.case.affine, image.affine)
                or not np.array_equal(source.case.mri, image.get_fdata(dtype=np.float32))):
            raise ValueError(f'{key} differs from its exact acquired physical grid or source array')
        sources[key], image_records[key] = source, binding(root, path)
    return primary, workspace, sources, image_records, binding(root, source_file)


def within_landmark_lattice(case, points):
    native = world(np.linalg.inv(case.affine), points)
    return ((native >= -.5 - 1.e-7) & (native <= np.asarray(case.mri.shape) - .5 + 1.e-7)).all(axis=1)


def fit_landmarks(root, sources):
    results = {}
    for name, start, end in RELATIONS:
        filename, _, digest = TAG[name]
        tag_path = checked_file(root, {'path': str(IMAGE_DIR / 'Landmarks' / filename), 'sha256': digest})
        pairs = np.asarray(parse_full_tag(tag_path.read_bytes()), dtype=float)
        src, dst = pairs[:, :3], pairs[:, 3:]
        order = np.lexsort(src.T[::-1])
        fit, held = order[::2], order[1::2]
        if (len(held) < 5 or any(len(np.unique(points, axis=0)) != len(pairs) for points in (src, dst))
                or not within_landmark_lattice(sources[start].case, src).all()
                or not within_landmark_lattice(sources[end].case, dst).all()):
            raise ValueError(f'Incomplete, duplicate or out-of-grid acquired landmark set {name}')
        rigid = _proper_rigid(src[fit], dst[fit])
        if rigid['status'] != 'available':
            raise ValueError(f'Degenerate conditioning landmarks for {name}')
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rigid['rotation'], rigid['translation_mm']
        observed = dst[held]
        predictions = {'identity': src[held], 'rigid': world(transform, src[held])}
        rows = []
        for i, old in enumerate(held):
            rows.append({'publisher_row': int(old + 1), 'source_ras_mm': src[old].tolist(),
                         'observed_destination_ras_mm': observed[i].tolist(),
                         'predicted_destination_ras_mm': {k: value[i].tolist() for k, value in predictions.items()},
                         'error_mm': {k: float(np.linalg.norm(value[i] - observed[i])) for k, value in predictions.items()}})
        results[name] = {
            'source': start, 'destination': end, 'source_tag': binding(root, tag_path),
            'source_to_destination_ras_mm': transform.tolist(),
            'fit_publisher_rows': (fit + 1).tolist(), 'held': rows,
            'metrics': {key: metrics([row['error_mm'][key] for row in rows]) for key in predictions},
            'held_predictions_in_destination_lattice': {
                key: int(within_landmark_lattice(sources[end].case, xyz).sum()) for key, xyz in predictions.items()},
        }
    return {'scope': 'Previously exposed TRAIN patient; held points excluded from each individual fit.',
            'partition': 'Lexicographic native source RAS coordinates; odd ranks fit and even ranks score.',
            'fit_uses_only_landmarks': True, 'relations': results}


def target_layer(root, stage, source, declaration):
    relative, _, digest, publisher = TARGETS[stage]
    path = checked_file(root, {'path': str(relative), 'sha256': digest})
    header = inspect_nifti(path)
    original = source.case
    image = nib.load(path)
    corners = _maximum_corner_displacement(original.affine, np.asarray(header['affine_ras_mm']), original.mri.shape)
    values = np.asarray(image.dataobj)
    if (header['shape'] != list(original.mri.shape)
            or corners > declaration['maximum_grid_corner_difference_mm']
            or image.header.get_xyzt_units()[0] != 'mm' or not np.array_equal(np.unique(values), [0, 1])):
        raise ValueError(f'Published {stage} tumor mask does not match its exact native lattice')
    mask = values == 1
    ref = replace(original.source_refs[0], source_id='supplied_target_annotation', uri=publisher,
                  sha256=digest, license=RIGHTS)
    layer = replace(original, source_refs=(*original.source_refs, ref), compartments={LABEL[stage]: mask},
                    source_compartments={LABEL[stage]: mask}, metadata={**original.metadata,
                        'acquisition_stage': STAGE[stage], 'annotationTimeRole': ANNOTATION_ROLE[stage],
                        'annotationPublisherDOI': declaration['annotation_release_doi'],
                        'annotationRightsSHA256': declaration['files'][0]['sha256'],
                        'annotationNativeGrid': header,
                        'display_label': 'Preoperative FLAIR · supplied tumor' if stage == 'flair'
                        else 'US before resection · supplied tumor'})
    return DisplaySource(layer, source.modality, 'source-provided'), path, mask, {
        'positive_voxels': int(mask.sum()), 'native_volume_mm3': float(mask.sum() * header['voxel_volume_mm3']),
        'source_mask_to_image_max_corner_mm': corners, 'annotation_sha256': digest, 'shape': list(mask.shape),
        'voxel_spacing_mm': header['spacing_mm']}


def fraction(part, whole):
    return part / whole if whole else None


def overlap_metrics(mapped, observed, visibility, voxel_mm3):
    source_count = int(np.count_nonzero(mapped & visibility))
    observed_count = int(np.count_nonzero(observed & visibility))
    common = int(np.count_nonzero(mapped & observed & visibility))
    return {
        'shared_support_voxels': int(visibility.sum()),
        'mapped_target_voxels_in_support': source_count,
        'observed_annotation_voxels_in_support': observed_count,
        'intersection_voxels': common,
        'intersection_mm3': common * voxel_mm3,
        'dice': fraction(2 * common, source_count + observed_count),
        'fraction_of_visible_mapped_target_in_observation': fraction(common, source_count),
        'fraction_of_visible_observation_in_mapped_target': fraction(common, observed_count),
    }


def sparse_intersection(source, destination, transform, source_mask, observed, source_support):
    """Independently verify dense nearest-neighbor results at observed positive cells."""
    ijk = np.argwhere(observed & (destination.mri > 0))
    pull = np.linalg.inv(source.affine) @ np.linalg.inv(transform) @ destination.affine
    continuous = world(pull, ijk)
    within = ((continuous >= 0) & (continuous <= np.asarray(source.mri.shape) - 1)).all(axis=1)
    indices = np.floor(continuous[within] + .5).astype(int)
    sampled = tuple(indices.T)
    return int(np.count_nonzero(source_support[sampled] & source_mask[sampled]))


def comparison(start, source, destination, source_mask, observed, landmark_result):
    hypotheses = {'identity': np.eye(4), 'rigid': np.asarray(landmark_result['source_to_destination_ras_mm'])}
    source_support = np.ones(source_mask.shape, dtype=bool) if start == 'flair' else source.mri > 0
    destination_signal = destination.mri > 0
    results, raw = {}, {}
    for key, matrix in hypotheses.items():
        transformed = matrix @ source.affine
        mapped = resample(source_mask, transformed, destination.mri.shape, destination.affine, order=0)
        support = resample(source_support, transformed, destination.mri.shape, destination.affine, order=0)
        visible = support & destination_signal
        values = overlap_metrics(mapped, observed, visible, destination.voxel_volume_mm3)
        values.update(mapped_target_voxels_on_destination_grid=int(mapped.sum()),
                      observed_annotation_voxels_total=int(observed.sum()))
        if key == 'rigid':
            witness = sparse_intersection(source, destination, matrix, source_mask, observed, source_support)
            if witness != values['intersection_voxels']:
                raise ValueError('Independent positive-voxel pullback disagrees with the dense intersection')
            values['independent_sparse_intersection_voxels'] = witness
        results[key], raw[key] = values, (mapped, visible)
    fixed = raw['identity'][1] & raw['rigid'][1]
    shared_score = {key: overlap_metrics(mapped, observed, fixed, destination.voxel_volume_mm3)
                    for key, (mapped, _) in raw.items()}
    centroid_ras = world(source.affine, center_of_mass(source_mask))
    nearest = float(min(np.linalg.norm(centroid_ras - np.asarray(row['source_ras_mm']))
                        for row in landmark_result['held']))
    results.update(same_support_under_both_hypotheses=shared_score,
                   source_target_centroid_to_nearest_held_landmark_mm=nearest,
                   source_support='Physical FLAIR lattice' if start == 'flair' else 'Positive before-US signal proxy',
                   destination_support='Positive destination US signal proxy',
                   destination_voxel_volume_mm3=float(destination.voxel_volume_mm3),
                   source_target_native_volume_mm3=float(source_mask.sum() * source.voxel_volume_mm3),
                   observed_annotation_native_volume_mm3=float(observed.sum() * destination.voxel_volume_mm3))
    return results, raw['rigid'][0] & raw['rigid'][1]


def plot_overlap(stage, scans, masks, mapped):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2), layout='constrained')
    after = scans['after'].case
    cavity = masks['after']
    center = np.rint(center_of_mass(cavity)).astype(int)
    margin = np.ceil(30 / np.asarray(after.spacing_mm)).astype(int)
    lower = np.maximum(center - margin, 0)
    upper = np.minimum(center + margin + 1, cavity.shape)
    crop = tuple(slice(lo, hi) for lo, hi in zip(lower, upper))
    gray = after.mri[crop]
    limits = np.percentile(gray[gray > 0], [2, 98])
    for row, key in enumerate(('flair_after', 'before_after')):
        signal = mapped[key][crop]
        for axis, view in enumerate(axes[row]):
            level = center[axis] - lower[axis]
            plane = np.take(gray, level, axis=axis).T
            spacing = np.delete(after.spacing_mm, axis)
            view.imshow(plane, origin='lower', cmap='gray', vmin=limits[0], vmax=limits[1],
                        aspect=spacing[1] / spacing[0])
            for mask, color in ((cavity[crop], '#72e7d7'), (signal, '#ffc17a')):
                pixels = np.take(mask, level, axis=axis).T
                if pixels.any() and not pixels.all():
                    view.contour(pixels, [.5], colors=[color], linewidths=[1.1])
            label = 'Preop FLAIR' if key == 'flair_after' else 'Before US'
            view.set_title(f'{label} · native after-US axis {axis}', fontsize=9)
            view.set_axis_off()
    fig.suptitle('RESECT Case3 · rigid target transport and observed cavity')
    handles = [Line2D([0], [0], color=color, label=name)
               for color, name in (('#ffc17a', 'Mapped pre-resection target within signal support'),
                                   ('#72e7d7', 'Observed after-resection cavity'))]
    fig.legend(handles=handles, loc='outside lower center', ncols=2, frameon=False, fontsize=9)
    fig.savefig(stage / 'target-cavity-overlays.png', dpi=150)
    plt.close(fig)


def freeze_predictions(path, landmarks):
    predictions = {}
    for key, relation in landmarks['relations'].items():
        rows = [{field: row[field] for field in ('publisher_row', 'source_ras_mm', 'predicted_destination_ras_mm')}
                for row in relation['held']]
        predictions[key] = {field: relation[field] for field in (
            'source', 'destination', 'source_tag', 'fit_publisher_rows', 'source_to_destination_ras_mm')}
        predictions[key]['held'] = rows
    atomic_json(path, {'partition': landmarks['partition'], 'relations': predictions})


def package_preview(stage, filename, keys, primary, scans, masks, originals):
    study = [scans[name] for name in keys]
    state = view_state(primary, study)
    for name in keys:
        case = scans[name].case
        viewer = state['states'][scans[name].identity(primary)]
        viewer['viewer'] = {'volumeRendering': {'enabled': False, 'preset': 'mri'}}
        viewer['cursor'] = world(case.affine, center_of_mass(masks[name])).tolist()
    archive = stage / filename
    save_workspace(primary, archive, sources=study, imaging=state, episode=None, selection=None, artifacts={},
                   originals=originals)
    reopened, artifacts, restored = load_workspace(archive)
    if (reopened.semantic_hash != primary.semantic_hash or reopened.compartments or artifacts
            or restored['episode'] is not None or len(restored['sources']) != len(study)):
        raise ValueError('Native reopen changed the source reference or benchmark scope')
    for name, source in zip(keys, restored['sources']):
        manifest = source.manifest(reopened)
        _, registration = source_registration(reopened, source.case)
        if (source.case.semantic_hash != scans[name].case.semantic_hash
                or manifest['scope'] != 'display-only-native-grid'
                or registration['sourceToReferenceRasMm'] is not None
                or not np.array_equal(source.case.compartments[LABEL[name]], masks[name])):
            raise ValueError(f'Observed {name} mask changed or became a reference-space input')
    return archive


def build(root, stage, restored_sources):
    declaration = require_manifest()
    primary, workspace, scans, images, source_workspace = source_study(root)
    landmarks = fit_landmarks(root, scans)
    frozen = stage / 'frozen-predictions.json'
    freeze_predictions(frozen, landmarks)
    rights = REPO / 'artifacts/resect-component-admission-v1/rights-review-02/README.txt'
    if sha256_file(rights) != declaration['files'][0]['sha256']:
        raise ValueError('Expected exact publisher annotation terms and version')
    terms = OriginalFile.from_path(rights, role='annotation')
    masks, observations, originals = {}, {}, {}
    for name in PREVIEW_KEYS:
        read_layer = cavity_layer if name == 'after' else target_layer
        source, path, mask, details = read_layer(root, name, scans[name], declaration)
        if (source.case.metadata['acquisition_stage'] != STAGE[name]
                or source.case.metadata['annotationTimeRole'] != ANNOTATION_ROLE[name]):
            raise ValueError(f'Unexpected acquisition stage or annotation role for {name}')
        scans[name], masks[name] = source, mask
        if nib.load(path).header.get_xyzt_units()[0] != 'mm':
            raise ValueError(f'Unexpected annotation length unit for {name}')
        originals[name] = OriginalFile.from_path(path, role='annotation')
        observations[name] = {**details, 'layer_label': LABEL[name], 'acquisition_stage': STAGE[name],
                              'annotation_time_role': ANNOTATION_ROLE[name], 'annotation': binding(root, path),
                              'display_label': source.case.metadata['display_label'],
                              'positive_outside_nonzero_signal': int(np.count_nonzero(mask & (source.case.mri <= 0)))}
    comparisons, mapped = {}, {}
    for name, start, end in RELATIONS:
        comparisons[name], transported = comparison(start, scans[start].case, scans[end].case,
                                                    masks[start], masks[end], landmarks['relations'][name])
        if end == 'after':
            mapped[name] = transported
    plot_overlap(stage, scans, masks, mapped)
    previews = {name + '.ressectionlab': list(keys) for name, keys in PREVIEWS.items()}
    for filename, keys in previews.items():
        package_preview(stage, filename, keys, primary, scans, masks,
                        [*workspace['originals'], terms, *(originals[key] for key in keys)])
    for value in observations.values():
        value['exact_native_mask_reopen'] = True
    return {
        'schema': SCHEMA, 'patient': 'RESECT:Case3', 'role': 'TRAIN',
        'sources': images, 'restored_sources': restored_sources, 'source_workspace': source_workspace,
        'landmarks': landmarks, 'preview_studies': previews, 'stage_visibility': observations,
        'comparison': comparisons, 'image_license': 'CC-BY-4.0', 'annotation_license': RIGHTS,
        'annotation_release': declaration['annotation_release_doi'],
        'interpretation': 'Registration-conditional agreement between publisher labels within acquired signal '
                          'support. Positive ultrasound intensity is a visibility proxy. FLAIR and ultrasound '
                          'tumor annotations have modality-dependent extents. The visible cavity cannot establish '
                          'residual tumor, removed tissue volume or extent of resection.',
        'prediction_scope': 'Sparse landmark-conditioned rigid transport. The held points are not used in their '
                            'individual fit; Case3 was previously exposed. No simulator output or dense motion '
                            'estimate is scored.',
        'postoperative_observation_excluded_from_preoperative_case': True,
        'members': {name: sha256_file(stage / name) for name in (*previews, 'target-cavity-overlays.png', frozen.name)},
    }


def score_saved_localization(root, benchmark, run):
    parent_report = benchmark / 'report.json'
    report = json.loads(parent_report.read_text())
    if report['schema'] != SCHEMA or report['patient'] != 'RESECT:Case3':
        raise ValueError('Expected the existing Case3 target and cavity benchmark')
    inputs = {name: checked_file(run, {'path': name, 'sha256': digest})
              for name, digest in SAVED_LOCALIZATION.items()}
    receipt = json.loads(inputs['receipt.json'].read_text())
    if (receipt['source'] != report['sources']['flair']['sha256'] or receipt['source'] != SIGNAL['flair'][1]
            or receipt['output_sha256'] != SAVED_LOCALIZATION['output/tumor_mask.nii.gz']):
        raise ValueError('Saved prediction is not linked to the benchmark FLAIR acquisition')
    image = nib.load(checked_file(root, report['sources']['flair']))
    annotation = report['stage_visibility']['flair']['annotation']
    if annotation['sha256'] != TARGETS['flair'][2]:
        raise ValueError('Expected the author-supplied Case3 FLAIR tumor label')
    mask_paths = (checked_file(root, annotation), inputs['output/tumor_mask.nii.gz'], inputs['brain-mask.nii.gz'])
    masks = []
    for path in mask_paths:
        loaded = nib.load(path)
        if (loaded.shape != image.shape
                or _maximum_corner_displacement(image.affine, loaded.affine, image.shape) > 1.e-3):
            raise ValueError(f'Saved comparison input does not match the FLAIR lattice: {path.name}')
        values = np.asarray(loaded.dataobj)
        if not np.isin(np.unique(values), (0, 1)).all():
            raise ValueError(f'Expected binary values for {path.name}')
        masks.append(values == 1)
    observed, predicted, brain = masks
    support = np.ones(observed.shape, dtype=bool)
    if int(predicted.sum()) != receipt['foreground_voxels'] or not observed.any():
        raise ValueError('Saved output foreground or publisher annotation differs from its receipt')
    result = {
        'schema': 'resectionlab-case3-saved-preoperative-localization-v1', 'patient': 'RESECT:Case3', 'role': 'TRAIN',
        'benchmark': binding(root, parent_report), 'flair': report['sources']['flair'], 'annotation': annotation,
        'inputs': SAVED_LOCALIZATION, 'model': receipt['model'], 'version': receipt['version'],
        'status': receipt['status'], 'native_lattice': overlap_metrics(predicted, observed, support,
                                                                      float(abs(np.linalg.det(image.affine[:3, :3])))),
        'supplied_target_voxels_inside_model_brain_mask': int(np.count_nonzero(observed & brain)),
        'scope': 'A saved preoperative localization run compared to its acquired FLAIR annotation. The brain-mask '
                 'intersection audits coarse input support; it does not establish patch-level inference coverage. '
                 'This diagnostic provides no simulator or residual-tumor score.',
    }
    destination = benchmark / 'historical-localization'
    if destination.exists():
        raise FileExistsError(destination)
    with TemporaryDirectory(dir=benchmark, prefix='.localization-') as scratch:
        stage = Path(scratch) / 'run'
        for name, path in inputs.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            checked_file(stage, {'path': name, 'sha256': SAVED_LOCALIZATION[name]})
        atomic_json(stage / 'report.json', result)
        checked_root(root, expected_uuid=UUID)
        stage.rename(destination)
    return destination / 'report.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output-name', default=OUTPUT)
    parser.add_argument('--saved-localization-run', type=Path,
                        help='Score and preserve the exact historical run beside an existing benchmark')
    args = parser.parse_args()
    if not args.output_name or Path(args.output_name).name != args.output_name or args.output_name.startswith('.'):
        parser.error('Use one fresh output directory name under qc/surgical-change')
    root = checked_root(args.root, expected_uuid=UUID, required_bytes=512 * 1024**2)
    destination = safe_path(root, QC / args.output_name)
    with reconstruction_lock():
        if args.saved_localization_run is not None:
            print(score_saved_localization(root, destination, args.saved_localization_run.resolve(strict=True)))
            return
        if destination.exists():
            raise FileExistsError(destination)
        sources = acquire(root)
        with TemporaryDirectory(dir=destination.parent, prefix='.resect-target-') as scratch:
            stage = Path(scratch) / 'study'
            stage.mkdir()
            atomic_json(stage / 'report.json', build(root, stage, sources))
            checked_root(root, expected_uuid=UUID)
            stage.rename(destination)
    print(destination / 'report.json')


if __name__ == '__main__':
    main()
