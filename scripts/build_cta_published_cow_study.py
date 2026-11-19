#!/usr/bin/env python3
"""Build the source-bound TopCoW 006 CTA/MRA companion with a publisher mesh."""
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

FAMILY = 'TopCoW:006'
BASE = 'TopCoW-006-CTA-ORGANIC-01-v1-NATIVE-SHOWCASE'
CASE = 'TopCoW-006-CTA-MRA-PUBLISHED-COW-v1'
OUTPUT = f'prepared/cta-published-cow-v1/{CASE}'
REVIEW = 'qc/multimodal-anatomy-20261118-cow-vtp-v1'
LAYER = 'publisher_cow_arterial_mesh'
QUALIFICATION_SHA = '0af09afa1649c578b46cb787d548711bef09236e2ba9022bc9d8fe3783c1acee'


def publisher_inputs(root):
    qualified = checked_file(root, {'path': f'{REVIEW}/verification.json', 'sha256': QUALIFICATION_SHA})
    evidence = json.loads(qualified.read_text())
    receipt = checked_file(root, {'path': f'{REVIEW}/source-receipt.json', 'sha256': evidence['source_receipt_sha256']})
    d = json.loads(receipt.read_text())
    if (d['family'] != FAMILY or evidence['family'] != FAMILY or d['role'] != 'TRAIN'
            or not evidence['original_and_topbrain_v1_mask_grid_exact'] or not evidence['original_cta_member_reused']):
        raise ValueError('Unexpected publisher geometry qualification')
    members = {}
    for source in d['releases'].values():
        for m in source['extracted']:
            member = f'{REVIEW}/publisher-members/{m["archive_path"]}'
            members[m['archive_path']] = checked_file(root, {'path': member, 'sha256': m['sha256']})
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


def copy_showcase(root, staging, base):
    document = deepcopy(base.document)
    document.pop('snapshot_id')
    document.update(patient_id=CASE, revision=0, history=[])
    for row in document['series'] + document['layers']:
        row['patient_id'] = CASE
    c = PatientImagingCollection(staging, document)
    for row in c.document['layers']:
        provenance = row['provenance']
        source = provenance['source']
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


def add_paired_mra(root, c):
    path = safe_path(root, 'catalog/topbrain/006-selective-extraction.json')
    evidence = json.loads(path.read_text())
    family = evidence['family']
    if family['family_id'] != FAMILY or family['role'] != 'TRAIN':
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
    grouping_path = safe_path(root, 'prepared/topbrain/006/mr/annotation-grouping.json')
    grouping = json.loads(grouping_path.read_text())
    if grouping['source_mask_sha256'] != mask_row['sha256']:
        raise ValueError('Retained MRA grouping binds a different source label')
    grouped = nib.load(checked_file(root, {'path': 'prepared/topbrain/006/mr/supplied-v1-annotation-groups.nii.gz',
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
    c.document['source_associations'] = {'family_id': FAMILY, 'frozen_source_role': 'TRAIN',
                                         'paired_acquisitions': ['ct', 'mra'], 'pair_receipt': c.original(path),
                                         'physical_relationship': 'CTA and MRA retained on independent native grids'}


def attach_publisher(root, c):
    receipt, qualification, evidence, measured, members = publisher_inputs(root)
    cta = c.series(SERIES)
    if (cta['original']['sha256'] != evidence['local_cta']['sha256']
            or cta['data']['shape'] != measured['native_cta_shape']
            or not np.array_equal(cta['affine'], measured['native_cta_ras_affine_mm'])):
        raise ValueError('Display parent differs from the qualified publisher CTA frame')
    mask_path = members['TopCoW2024_Data_Release/cow_seg_labelsTr/topcow_ct_006.nii.gz']
    mesh_path = members['CoW_Centerline_Data/cow_meshes/topcow_ct_006.vtp']
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
                                           'mesh_license': 'CC BY-NC 4.0', 'source_family_id': FAMILY}
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


def build(root):
    destination = safe_path(root, OUTPUT)
    manifest = destination / 'patient-anatomy.json'
    base_receipt = safe_path(root, f'{SHOWCASES}/{BASE}/receipt.json')
    base_info = json.loads(base_receipt.read_text())
    base = PatientImagingCollection(checked_file(root, base_info['manifest']).parent)
    if base.document['data_case']['source_family_id'] != FAMILY or base.identity() != base_info['snapshot_id']:
        raise ValueError('Existing CTA study changed')
    if destination.exists():
        report = json.loads((destination / 'receipt.json').read_text())
        if (checked_file(root, report['manifest']) != manifest or report['base_receipt'] != binding(root, base_receipt)
                or checked_file(root, report['transfer_expectation']).parent != destination):
            raise ValueError('Existing publisher companion has a different source or transfer receipt')
        c = PatientImagingCollection(destination)
        for member, checksum in set(referenced_files(c.document)):
            if sha256_file(c.path(member)) != checksum:
                raise ValueError(f'Existing portable source member changed: {member}')
        return report
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.cow-build-', dir=destination.parent) as temp:
        stage = Path(temp) / CASE
        stage.mkdir()
        c = copy_showcase(root, stage, base)
        points, triangles = attach_publisher(root, c)
        add_paired_mra(root, c)
        c.save('Retain publisher CTA geometry and matched MRA on its independent grid')
        reopened = PatientImagingCollection(stage)
        expectation = stage / 'transfer-expectation.json'
        atomic_json(expectation, transfer_expectation(reopened, points, triangles))
        report = {'schema': 'ressectionlab-published-cow-native-study-v1', 'case_id': CASE, 'role': 'DEVELOPMENT',
                  'source_family_id': FAMILY, 'new_people': 0, 'new_scenarios': 0, 'reconstruction_reruns': 0,
                  'base_receipt': binding(root, base_receipt), 'snapshot_id': reopened.identity(),
                  'manifest': {'path': str(manifest.relative_to(root)), 'sha256': sha256_file(stage / manifest.name)},
                  'transfer_expectation': {'path': f'{OUTPUT}/{expectation.name}', 'sha256': sha256_file(expectation)}}
        atomic_json(stage / 'receipt.json', report)
        checked_root(root, expected_uuid=UUID)
        stage.rename(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--register', type=Path)
    args = parser.parse_args()
    root = checked_root(args.root, required_bytes=2 * 1024**3, expected_uuid=UUID)
    with reconstruction_lock():
        report = build(root)
        if args.register:
            register(root, args.register, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
