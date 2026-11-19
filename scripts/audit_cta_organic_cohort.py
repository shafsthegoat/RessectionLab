#!/usr/bin/env python3
"""Check the native CTA scenarios against isolated truth and UCSF TRAIN shape summaries."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import nibabel as nib
import numpy as np
from scipy.ndimage import label
from skimage.measure import marching_cubes, mesh_surface_area

from build_cta_organic_gliomas import CALIBRATION, COLLECTION, OBSERVATION, TRUTH, read_json
from data_integrated_hosts import ORDER, UUID, binding, checked_file
from resectionlab.core import array_digest
from resectionlab.data_root import atomic_json, checked_root, safe_path
from resectionlab.patient_anatomy import reconstruction_lock
from resectionlab.workspace_bundle import extension_layout, load_workspace

OUTPUT = 'cta-organic-cohort-shape-audit-v1'
FAMILIES = ('TopCoW:006', 'TopCoW:012', 'TopCoW:014')
REGIONS = ('core_plus_enhancing', 'whole_including_edema')
METRICS = ('volume_ml', 'sphericity', 'minor_to_major_pca_ratio', 'components_6_connected')
QUANTILES = ('min', 'p10', 'p25', 'median', 'p75', 'p90', 'max')


def measure(mask, affine):
    points = np.argwhere(mask)
    lower, upper = points.min(axis=0), points.max(axis=0) + 1
    crop = tuple(slice(start, stop) for start, stop in zip(lower, upper))
    compact = np.pad(mask[crop], 1)
    linear = affine[:3, :3]
    volume = len(points) * abs(np.linalg.det(linear))
    vertices, faces, _, _ = marching_cubes(compact, level=.5)
    area = mesh_surface_area(vertices @ linear.T, faces)
    eigenvalues = np.linalg.eigvalsh(np.cov(points @ linear.T, rowvar=False))
    return {'volume_ml': float(volume / 1000),
            'sphericity': float(np.cbrt(36 * np.pi * volume**2) / area),
            'minor_to_major_pca_ratio': float(np.sqrt(eigenvalues[0] / eigenvalues[-1])),
            'components_6_connected': label(compact)[1]}


def check_case(root, record, guidance):
    native = checked_file(root, record['native_case'])
    truth_path = checked_file(root, record['truth'])
    checked_file(root, record['qa'])
    receipt_path = native.with_name('receipt.json')
    receipt = read_json(receipt_path)
    if (any(receipt[key] != value for key, value in record.items())
            or receipt['truth']['planning_eligible'] is not False
            or receipt['truth']['label_codes'] != TRUTH
            or receipt['role'] != 'DEVELOPMENT' or receipt['source_family_role'] != 'TRAIN'
            or receipt['source']['statistical_guidance'] != guidance):
        raise ValueError(f'{native.name} differs from the sealed cohort or its frozen roles')

    case, artifacts, workspace = load_workspace(native)
    data_case = case.metadata['data_case']
    expected = {*ORDER, OBSERVATION}
    if (case.case_id != receipt['case_id'] or case.semantic_hash != receipt['case_hash']
            or data_case['family_id'] != receipt['person_id'] or data_case['role'] != 'DEVELOPMENT'
            or data_case['source_family_role'] != 'TRAIN' or case.frame != 'RAS+'
            or list(case.mri.shape) != receipt['native_grid_shape']
            or not np.array_equal(case.affine, receipt['source_affine_ras_mm'])
            or array_digest(case.mri) != receipt['source_volume_hash']
            or artifacts or workspace is None or workspace['episode'] is not None or workspace['sources']
            or set(case.compartments) != expected or set(case.source_compartments) != expected):
        raise ValueError(f'{case.case_id} changed its source grid, acquired CTA or planning layout')

    hashes = {**receipt['source_membership_hashes'], OBSERVATION: receipt['observation']['array_hash']}
    for key, expected_hash in hashes.items():
        if (array_digest(case.source_compartments[key]) != expected_hash
                or not np.array_equal(case.compartments[key], case.source_compartments[key])):
            raise ValueError(f'{case.case_id} changed baseline membership {key}')
    with ZipFile(native) as archive:
        layout = extension_layout(archive)
    with ZipFile(checked_file(root, receipt['source']['host'])) as archive:
        host_originals = extension_layout(archive)['originalFiles']
    originals = {row['sha256'] for row in layout['originalFiles']}
    bound_sources = {row['sha256'] for row in receipt['source']['source_CTA_and_v1_labels']}
    if (layout['sessionHash'] != receipt['session_hash'] or layout['originalFiles'] != host_originals
            or not bound_sources <= originals or receipt['truth']['sha256'] in originals):
        raise ValueError(f'{case.case_id} changed attachments or packaged private truth')

    truth_image = nib.load(truth_path)
    truth = np.asanyarray(truth_image.dataobj)
    if (truth.shape != case.mri.shape or not np.array_equal(truth_image.affine, case.affine)
            or truth_image.header.get_xyzt_units()[0] != 'mm' or set(np.unique(truth)) != {0, *TRUTH.values()}):
        raise ValueError(f'{case.case_id} has invalid private truth or a mismatched frame')
    core_rim = (truth == TRUTH['synthetic_core']) | (truth == TRUTH['synthetic_rim'])
    whole = truth != 0
    observation = case.compartments[OBSERVATION]
    dice = float(2 * np.count_nonzero(observation & core_rim) / (observation.sum() + core_rim.sum()))
    if (np.any(whole & ~case.compartments[ORDER[0]])
            or any(np.any(whole & case.compartments[key]) for key in ORDER[1:])
            or not .75 <= dice < .995 or not np.isclose(dice, receipt['observation']['truth_overlap_dice'])):
        raise ValueError(f'{case.case_id} fails tissue containment or proxy separation')

    return {'case_id': case.case_id, 'person_id': receipt['person_id'], 'case_hash': case.semantic_hash,
            'receipt': binding(root, receipt_path), 'seed': receipt['recipe']['seed'],
            'profile': receipt['recipe']['profile'], 'placement': receipt['recipe']['placement'],
            'center_ras_mm': receipt['recipe']['center_ras_mm'], 'observation_dice': dice,
            'truth_sha256': record['truth']['sha256'], 'truth_array_hash': array_digest(truth),
            'observation_hash': hashes[OBSERVATION],
            'core_plus_enhancing': measure(core_rim, case.affine),
            'whole_including_edema': measure(whole, case.affine),
            'edema_fraction_of_whole': float(np.count_nonzero(truth == TRUTH['synthetic_edema']) / whole.sum())}


def summarize(values):
    return dict(zip(QUANTILES, np.percentile(values, [0, 10, 25, 50, 75, 90, 100]).tolist()))


def plot(path, rows, reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    columns = ('Volume (mL, log scale)', 'Sphericity', 'Minor-to-major axis ratio')
    colors = dict(zip(FAMILIES, ('#007682', '#af4d2d', '#5854a3')))
    figure, axes = plt.subplots(2, 3, figsize=(11, 5.7), sharey=True, layout='constrained')
    positions = list(range(len(rows)))
    labels = [' / '.join(row['case_id'].split('-')[index] for index in (1, 4)) for row in rows]
    for row_index, region in enumerate(REGIONS):
        for col_index, (metric, title) in enumerate(zip(METRICS, columns)):
            axis = axes[row_index, col_index]
            stats = reference[region][metric]
            axis.axvspan(stats['min'], stats['max'], color='#e7edf1')
            axis.axvspan(stats['p10'], stats['p90'], color='#c0cfda')
            axis.axvline(stats['median'], color='#456176', lw=1, ls='--')
            axis.scatter([row[region][metric] for row in rows], positions,
                         c=[colors[row['person_id']] for row in rows], s=26, zorder=3)
            if metric == 'volume_ml':
                axis.set_xscale('log')
            axis.set_yticks(positions, labels)
            axis.tick_params(axis='y', length=0, labelsize=8)
            if col_index == 0:
                axis.set_ylabel(('Core and rim', 'Core, rim and edema')[row_index])
            if row_index == 0:
                axis.set_title(title, fontsize=10)
    axes[0, 0].invert_yaxis()
    figure.suptitle('Synthetic CTA targets and UCSF TRAIN mask measurements', x=.01, ha='left', fontsize=14)
    handles = [Line2D([], [], marker='o', color=color, ls='none', label=family.replace(':', ' '))
               for family, color in colors.items()]
    handles.extend((Patch(color='#e7edf1', label='UCSF observed range'),
                    Patch(color='#c0cfda', label='UCSF 10th–90th percentile'),
                    Line2D([], [], color='#456176', ls='--', label='UCSF median')))
    figure.legend(handles=handles, ncols=6, loc='outside lower center', frameon=False, fontsize=9)
    figure.savefig(path, dpi=155, facecolor='white')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output-name', default=OUTPUT)
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or args.output_name in ('', '.', '..'):
        parser.error('--output-name must be a new directory name under qc')
    root = checked_root(args.root, expected_uuid=UUID)
    collection_path = safe_path(root, f'{COLLECTION}/collection.json')
    collection = read_json(collection_path)
    guidance_path = checked_file(root, collection['statistical_guidance'])
    guidance = read_json(guidance_path)
    if (guidance_path != safe_path(root, CALIBRATION) or guidance['role'] != 'TRAIN' or guidance['n'] != 46
            or guidance['independent_evaluation_claim'] is not False or collection['role'] != 'DEVELOPMENT'
            or collection['generated_scenarios'] != 12
            or Counter(row['person_id'] for row in collection['cases']) != Counter({family: 4 for family in FAMILIES})):
        raise ValueError('Expected the sealed 12-scenario CTA cohort and 46 retained UCSF TRAIN masks')
    output = safe_path(root, f'qc/{args.output_name}')
    with reconstruction_lock():
        if output.exists():
            raise FileExistsError(output)
        rows = []
        for record in collection['cases']:
            row = check_case(root, record, collection['statistical_guidance'])
            rows.append(row)
            print(f"Verified {row['case_id']}", flush=True)
        for identity in ('case_id', 'case_hash', 'seed', 'truth_sha256', 'truth_array_hash', 'observation_hash'):
            if len({row[identity] for row in rows}) != len(rows):
                raise ValueError(f'Duplicate generated case identity in {identity}')
        report = {'schema': OUTPUT, 'passed': True, 'collection': binding(root, collection_path),
                  'shape_guidance': collection['statistical_guidance'], 'reference_n': guidance['n'],
                  'measurement': guidance['measurement'], 'reviewed_cases': len(rows),
                  'checked_tissues': list(ORDER), 'private_truth_in_planning_workspace': False,
                  'limits': ['UCSF and CTA labels only conventionally match by region.',
                             'This engineering cohort does not estimate a clinical phenotype distribution.',
                             'Containment uses the available modeled brain and source structures; exact CSF coverage is unavailable.'],
                  'reference_summary': guidance['summary'], 'cases': rows}
        report['scenario_summary'] = {region: {metric: summarize([row[region][metric] for row in rows])
                                               for metric in METRICS} for region in REGIONS}
        for scalar in ('edema_fraction_of_whole', 'observation_dice'):
            report['scenario_summary'][scalar] = summarize([row[scalar] for row in rows])
        with TemporaryDirectory(prefix='.cta-shape-audit-', dir=output.parent) as temporary:
            stage = Path(temporary) / args.output_name
            stage.mkdir()
            figure = stage / 'cohort-shape-comparison.png'
            plot(figure, rows, guidance['summary'])
            report['shape_comparison'] = {**binding(root, figure),
                                          'path': str((output / figure.name).relative_to(root))}
            atomic_json(stage / 'report.json', report)
            stage.rename(output)
    print(output / 'report.json')


if __name__ == '__main__':
    main()
