#!/usr/bin/env python3
"""Build source-bound CTA/MRA companions with original publisher CoW geometry."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from build_cta_native_showcases import ARTERIES, OUTPUT as SHOWCASES, SERIES, TIMEPOINT, register
from data_integrated_hosts import UUID, binding, checked_file
from resectionlab.anatomy_package import referenced_files
from resectionlab.anatomy_publisher_surface import POLICY, publisher_surface
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection, nifti_geometry, reconstruction_lock, resample

FIRST_REVIEW = 'qc/multimodal-anatomy-20261118-cow-vtp-v1'
LAYER = 'publisher_cow_arterial_mesh'
QUALIFICATIONS = {
    '006': '0af09afa1649c578b46cb787d548711bef09236e2ba9022bc9d8fe3783c1acee',
    '012': '6d87a8a0ce6a1a0e02605dd56d8422c24798cd4783e11164fd9d36ee03348ccc',
    '014': 'd3cc3c79f086a39160c318af6ebb4beb3139569431767b86fa2039db2c0b3451',
}


def publisher_inputs(root, subject):
    review = FIRST_REVIEW if subject == '006' else f'sources/topcow-cow-vtp-selective-v1/{subject}'
    qualified = checked_file(root, {'path': f'{review}/verification.json', 'sha256': QUALIFICATIONS[subject]})
    evidence = json.loads(qualified.read_text())
    receipt = checked_file(root, {'path': f'{review}/source-receipt.json', 'sha256': evidence['source_receipt_sha256']})
    d = json.loads(receipt.read_text())
    family = f'TopCoW:{subject}'
    if (d['family'] != family or evidence['family'] != family or d['role'] != 'TRAIN'
            or not evidence['original_and_topbrain_v1_mask_grid_exact'] or not evidence['original_cta_member_reused']
            or (subject != '006' and (evidence.get('display_qualified') is not True
                                     or evidence.get('frame_transform_applied') is not False))):
        raise ValueError('Unexpected publisher geometry qualification')
    members = {}
    for source in d['releases'].values():
        for m in source['extracted']:
            member = f'{review}/publisher-members/{m["archive_path"]}'
            members[m['archive_path']] = checked_file(root, {'path': member, 'sha256': m['sha256']})
    if subject != '006':
        for name, member in publisher_inputs(root, '006')[-1].items():
            if Path(name).name in {'License.txt', 'README.txt'}:
                members.setdefault(name, member)
    return receipt, qualified, d, evidence, members


def read_mesh(path, expected):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

    reader = vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if (poly is None or poly.GetNumberOfPoints() != expected['vertices']
            or poly.GetNumberOfCells() != poly.GetNumberOfPolys() or poly.GetNumberOfPolys() != expected['triangles']):
        raise ValueError('Publisher VTP differs from the qualified mesh inventory')
    cells = poly.GetPolys()
    if not np.all(np.diff(vtk_to_numpy(cells.GetOffsetsArray())) == 3):
        raise ValueError('Publisher surface must retain the supplied triangle topology')
    points = vtk_to_numpy(poly.GetPoints().GetData()).copy()
    faces = vtk_to_numpy(cells.GetConnectivityArray()).reshape(-1, 3)
    labels = vtk_to_numpy(poly.GetCellData().GetArray('labels')).copy()
    if (points.dtype != np.float32 or not np.isfinite(points).all() or faces.min() < 0
            or faces.max() >= len(points) or len(labels) != len(faces)
            or not np.array_equal(np.unique(labels), expected['nonempty_class_ids'])):
        raise ValueError('Publisher surface coordinates, indices or class labels failed source QC')
    return points, faces.astype(np.uint32), labels


def copy_showcase(root, staging, base, case_id):
    document = deepcopy(base.document)
    document.pop('snapshot_id')
    document.update(patient_id=case_id, revision=0, history=[])
    for row in document['series'] + document['layers']:
        row['patient_id'] = case_id
    c = PatientImagingCollection(staging, document)
    for row in c.document['layers']:
        provenance = row['provenance']
        source = provenance['source']
        if 'path' not in source:
            continue
        if row['kind'] == 'tumor':
            # Its exact visible contour is already embedded as a numeric array.
            provenance['source'] = {'root_relative_locator': source['path'], 'sha256': source['sha256']}
        elif not base.path(source['path']).is_file():
            provenance['source'] = c.original(checked_file(root, source))
    for name, checksum in set(referenced_files(c.document)):
        instudy = c.path(name)
        if instudy.is_file() and sha256_file(instudy) == checksum:
            continue
        original = base.path(name)
        if sha256_file(original) != checksum:
            raise ValueError('Native showcase member changed')
        instudy.parent.mkdir(parents=True, exist_ok=True)
        copyfile(original, instudy)
    return c


def add_paired_mra(root, c, subject):
    path = safe_path(root, f'catalog/topbrain/{subject}-selective-extraction.json')
    evidence = json.loads(path.read_text())
    family = evidence['family']
    if family['family_id'] != f'TopCoW:{subject}' or family['role'] != 'TRAIN':
        raise ValueError('Paired MRA family or locked role differs')
    image_row = next(row for row in evidence['files'] if row['source_member'] == family['images']['mr'])
    mask_row = next(row for row in evidence['files'] if '/labelsTr_topbrain_v1_mr/' in row['source_member'])
    image_path, mask_path = (checked_file(root, row) for row in (image_row, mask_row))
    scan = c.add_series(image_path, series_id='mra', modality='MRA', timepoint=TIMEPOINT)
    scan.update(label='TopBrain MRA · independent acquired grid', publisher_dataset='https://zenodo.org/records/21972006',
                source_license=c.series(SERIES)['source_license'])
    c.require_native_geometry(scan)
    image = nib.load(mask_path)
    geometry = nifti_geometry(image)
    if (image.shape != tuple(scan['data']['shape']) or not np.array_equal(image.affine, scan['affine'])
            or any(not np.allclose(geometry[key], scan['affine'], rtol=0, atol=1e-5)
                   for key in ('qform', 'sform') if geometry[key + '_code'])):
        raise ValueError('Publisher MRA label differs from its native source lattice')
    grouping_path = safe_path(root, f'prepared/topbrain/{subject}/mr/annotation-grouping.json')
    grouping = json.loads(grouping_path.read_text())
    if grouping['source_mask_sha256'] != mask_row['sha256']:
        raise ValueError('Retained MRA grouping binds a different source label')
    grouped = nib.load(checked_file(root, {'path': f'prepared/topbrain/{subject}/mr/supplied-v1-annotation-groups.nii.gz',
                                          'sha256': grouping['grouped_mask_sha256']}))
    union = np.asarray(grouped.dataobj)
    if (not np.array_equal(grouped.affine, image.affine) or not np.isin(union, (0, 1)).all()
            or not np.array_equal(union.astype(bool), np.asarray(image.dataobj) > 0)):
        raise ValueError('Retained MRA arterial union differs from its publisher label')
    original = c.original(mask_path)
    row = c.add_layer(union, image.affine, layer_id='mra-publisher-artery', source_series='mra', kind='artery',
                      role='context', color='#f06455', label='TopBrain supplied MRA arterial annotation',
                      provenance={'kind': 'supplied_annotation', 'source': original, 'grouping': c.original(grouping_path)})
    row.update(units_status=geometry['spatial_units'], coverage='Publisher arterial annotation on independent acquired MRA grid')
    c.document['source_associations'] = {'family_id': family['family_id'], 'frozen_source_role': 'TRAIN',
                                         'paired_acquisitions': ['ct', 'mra'], 'pair_receipt': c.original(path),
                                         'physical_relationship': 'CTA and MRA retained on independent native grids'}


def attach_publisher(root, c, subject):
    receipt, qualification, evidence, measured, members = publisher_inputs(root, subject)
    cta = c.series(SERIES)
    if (cta['original']['sha256'] != evidence['local_cta']['sha256']
            or cta['data']['shape'] != measured['native_cta_shape']
            or not np.array_equal(cta['affine'], measured['native_cta_ras_affine_mm'])):
        raise ValueError('Display parent differs from the qualified publisher CTA frame')
    mask_path = members[f'TopCoW2024_Data_Release/cow_seg_labelsTr/topcow_ct_{subject}.nii.gz']
    mesh_path = members[f'CoW_Centerline_Data/cow_meshes/topcow_ct_{subject}.vtp']
    label = nib.load(mask_path)
    values = np.asarray(label.dataobj)
    if (values.shape != tuple(cta['data']['shape']) or not np.array_equal(label.affine, cta['affine'])
            or values.dtype != np.uint8 or np.count_nonzero(values) != measured['original_CoW_vs_topbrain_v1']['original_positive']):
        raise ValueError('Original CoW label differs from the qualified publisher CTA lattice')
    points, triangles, classes = read_mesh(mesh_path, measured['mesh'])
    display = c.add_layer((values > 0).astype(np.uint8), label.affine, layer_id=LAYER,
                          source_series=SERIES, kind='artery', role='context', color='#fa795d',
                          label='Circle of Willis · publisher mesh', provenance={
                              'kind': 'supplied_annotation', 'source': c.original(mask_path),
                              'evidence_use': 'publisher mesh and separate native label for display',
                              'publisher_dataset': 'https://zenodo.org/records/17358162'})
    display.update(units_status=label.header.get_xyzt_units()[0],
                   coverage='Publisher Circle of Willis arterial subset; wider TopBrain extent is a separate layer',
                   publisher_surface={'schema': POLICY, 'original': c.original(mesh_path),
                                      'vertices_ras_mm': c.put_array(points), 'triangles': c.put_array(triangles),
                                      'cell_labels': c.put_array(classes), 'source_series': SERIES,
                                      'source_sha256': cta['original']['sha256'], 'source_affine': cta['affine'],
                                      'mask_sha256': display['data']['sha256'], 'mask_label_value': display['label_value'],
                                      'mask_revision': display['revision']})
    c.document['publisher_cow_evidence'] = {'assets': {name: c.original(path) for name, path in sorted(members.items())},
                                           'source_receipt': c.original(receipt), 'qualification': c.original(qualification),
                                           'mesh_license': 'CC BY-NC 4.0', 'source_family_id': evidence['family']}
    c.document['display']['initial_layers'] = [LAYER if name == ARTERIES else name
                                               for name in c.document['display']['initial_layers']]
    c.document['display']['cursor_ras_mm'] = points.mean(axis=0, dtype=np.float64).tolist()
    c.document['missing'] = [name for name in c.document['missing'] if name != 'MRI and tractography']
    c.document['missing'] += ['Structural brain MRI and tractography', 'Qualified CTA-to-MRA registration']
    return points, triangles


def transfer_expectation(c, vertices, triangles):
    row = c.layer(LAYER)
    supplied = publisher_surface(c, row, cancelled=lambda: False)
    affine = np.array(row['affine'])
    relation = np.linalg.inv(affine)
    source = (vertices.astype(np.float64) @ relation[:3, :3].T + relation[:3, 3]).astype(np.float32)
    if not np.array_equal(supplied['positions'].astype(np.float32), source) or not np.array_equal(supplied['indices'], triangles):
        raise ValueError('Display geometry changed publisher points or triangle order')
    mask = resample(c.mask(row).astype(np.uint8), affine, tuple(row['data']['shape']), affine, order=0)
    def sha(value):
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    return {'layerId': LAYER, 'policy': POLICY, 'sourceVtp': row['publisher_surface']['original'],
            'points': len(vertices), 'triangles': len(triangles), 'transfer': {
                'sourceMeshSha256': sha(source), 'displayMeshSha256': sha(source),
                'indicesSha256': sha(triangles), 'maskSha256': sha(mask)}}


def build(root, subject='006'):
    case_id = f'TopCoW-{subject}-CTA-MRA-PUBLISHED-COW-v1'
    family = f'TopCoW:{subject}'
    output = f'prepared/cta-published-cow-v1/{case_id}'
    destination = safe_path(root, output)
    manifest = destination / 'patient-anatomy.json'
    base_name = f'TopCoW-{subject}-CTA-ORGANIC-01-v1-NATIVE-SHOWCASE'
    base_receipt = safe_path(root, f'{SHOWCASES}/{base_name}/receipt.json')
    base_info = json.loads(base_receipt.read_text())
    base = PatientImagingCollection(checked_file(root, base_info['manifest']).parent)
    if base.document['data_case']['source_family_id'] != family or base.identity() != base_info['snapshot_id']:
        raise ValueError('Existing CTA study changed')
    if destination.exists():
        report = json.loads((destination / 'receipt.json').read_text())
        parent_hashes = {sha256_file(base_receipt)}
        if original_parent := base_info.get('stock_receipt'):
            parent_hashes.add(sha256_file(checked_file(root, original_parent)))
        if (checked_file(root, report['manifest']) != manifest
                or report['base_receipt']['path'] != str(base_receipt.relative_to(root))
                or report['base_receipt']['sha256'] not in parent_hashes
                or checked_file(root, report['transfer_expectation']).parent != destination):
            raise ValueError('Existing publisher companion has a different source or transfer receipt')
        c = PatientImagingCollection(destination)
        for member, checksum in set(referenced_files(c.document)):
            if sha256_file(c.path(member)) != checksum:
                raise ValueError(f'Existing portable source member changed: {member}')
        return report
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.cow-build-', dir=destination.parent) as temp:
        stage = Path(temp) / case_id
        stage.mkdir()
        c = copy_showcase(root, stage, base, case_id)
        points, triangles = attach_publisher(root, c, subject)
        add_paired_mra(root, c, subject)
        c.save('Retain publisher CTA geometry and matched MRA on its independent grid')
        reopened = PatientImagingCollection(stage)
        expectation = stage / 'transfer-expectation.json'
        atomic_json(expectation, transfer_expectation(reopened, points, triangles))
        report = {'schema': 'ressectionlab-published-cow-native-study-v1', 'case_id': case_id, 'role': 'DEVELOPMENT',
                  'source_family_id': family, 'new_people': 0, 'new_scenarios': 0, 'reconstruction_reruns': 0,
                  'base_receipt': binding(root, base_receipt), 'snapshot_id': reopened.identity(),
                  'manifest': {'path': str(manifest.relative_to(root)), 'sha256': sha256_file(stage / manifest.name)},
                  'transfer_expectation': {'path': f'{output}/{expectation.name}', 'sha256': sha256_file(expectation)}}
        atomic_json(stage / 'receipt.json', report)
        checked_root(root, expected_uuid=UUID)
        stage.rename(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--register', type=Path)
    parser.add_argument('--subject', choices=tuple(QUALIFICATIONS), action='append')
    args = parser.parse_args()
    root = checked_root(args.root, required_bytes=2 * 1024**3, expected_uuid=UUID)
    with reconstruction_lock():
        reports = [build(root, subject) for subject in dict.fromkeys(args.subject or ['006'])]
        if args.register:
            for report in reports:
                register(root, args.register, report)
    print(json.dumps(reports[0] if len(reports) == 1 else reports, indent=2))


if __name__ == '__main__':
    main()
