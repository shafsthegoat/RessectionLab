#!/usr/bin/env python3
"""Expose exact publisher CoW classes beside the existing TopCoW 014 anatomy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from scipy.ndimage import label as label_components
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from build_cta_native_showcases import SERIES, register
from build_cta_published_cow_study import LAYER, copy_showcase, publisher_inputs, read_mesh
from data_integrated_hosts import UUID, binding, checked_file
from resectionlab.anatomy_package import referenced_files
from resectionlab.anatomy_publisher_surface import POLICY, publisher_surface
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection, reconstruction_lock, world

SUBJECT = '014'
FAMILY = 'TopCoW:014'
CASE = 'TopCoW-014-CTA-PUBLISHER-BRANCHES-v1'
PARENT = 'prepared/cta-published-cow-v1/TopCoW-014-CTA-MRA-PUBLISHED-COW-v1'
OUTPUT = f'prepared/topcow014-publisher-branches-v1/{CASE}'
SCHEMA = 'ressectionlab-publisher-cow-native-branches-v1'
MASK = 'TopCoW2024_Data_Release/cow_seg_labelsTr/topcow_ct_014.nii.gz'
MESH = 'CoW_Centerline_Data/cow_meshes/topcow_ct_014.vtp'
GRAPH = 'CoW_Centerline_Data/cow_graphs/topcow_ct_014.vtp'
README = 'CoW_Centerline_Data/README.txt'
# Source IDs and short labels use the retained CoW centerline README.
CLASSES = {
    1: ('BA', 'Basilar artery', '#f6c35e'),
    2: ('R-PCA', 'Right posterior cerebral artery', '#f28c46'),
    3: ('L-PCA', 'Left posterior cerebral artery', '#ae9df2'),
    4: ('R-ICA', 'Right internal carotid artery', '#e7696e'),
    5: ('R-MCA', 'Right middle cerebral artery', '#ff9caa'),
    6: ('L-ICA', 'Left internal carotid artery', '#278c69'),
    7: ('L-MCA', 'Left middle cerebral artery', '#72d49b'),
    8: ('R-Pcom', 'Right posterior communicating artery', '#dc9067'),
    9: ('L-Pcom', 'Left posterior communicating artery', '#b99ae4'),
    10: ('Acom', 'Anterior communicating artery', '#f3edac'),
    11: ('R-ACA', 'Right anterior cerebral artery', '#cd6fb0'),
    12: ('L-ACA', 'Left anterior cerebral artery', '#52bfaf'),
    15: ('3rd-A2', 'Third A2 segment', '#e5a0d8'),
}


def read_document(path):
    return json.loads(path.read_text())


def check_packaged_members(collection):
    for name, checksum in set(referenced_files(collection.document)):
        if sha256_file(collection.path(name)) != checksum:
            raise ValueError(f'Native study member changed: {name}')


def read_inputs(root, parent):
    source, qualified, _, evidence, members = publisher_inputs(root, SUBJECT)
    source_row = parent.series(SERIES)
    full = parent.layer(LAYER)
    attachment = full['publisher_surface']
    if (parent.document['reference_series'] != SERIES or parent.document['data_case']['source_family_id'] != FAMILY
            or parent.document['publisher_cow_evidence']['source_family_id'] != FAMILY
            or source_row['modality'] != 'CTA'
            or source_row['data']['shape'] != evidence['native_cta_shape']
            or source_row['affine'] != evidence['native_cta_ras_affine_mm']):
        raise ValueError('Expected the native CTA014 publisher companion')
    originals = {
        MASK: full['provenance']['source'], MESH: attachment['original'],
        GRAPH: parent.document['publisher_cow_evidence']['assets'][GRAPH],
        README: parent.document['publisher_cow_evidence']['assets'][README],
    }
    for name, original in originals.items():
        archive_hash = sha256_file(members[name])
        if original['sha256'] != archive_hash or sha256_file(parent.path(original['path'])) != archive_hash:
            raise ValueError(f'Parent original differs from the qualified publisher member: {name}')
    class_line = next(line for line in members[README].read_text().splitlines() if '1: BA, 2:' in line)
    ontology = {int(value): code for value, code in re.findall(r'(\d+):\s*([\w-]+)', class_line)}
    if ontology != {value: spec[0] for value, spec in CLASSES.items()}:
        raise ValueError('Publisher class ontology differs from the retained release')
    image = nib.load(members[MASK])
    values = np.asarray(image.dataobj)
    present = set(np.unique(values).tolist()) - {0}
    points, faces, labels = read_mesh(members[MESH], evidence['mesh'])
    if (values.dtype != np.uint8 or list(values.shape) != source_row['data']['shape']
            or not np.array_equal(image.affine, source_row['affine'])
            or int(np.count_nonzero(values)) != evidence['original_CoW_vs_topbrain_v1']['original_positive']
            or present != set(labels.tolist()) or not present <= set(ontology)):
        raise ValueError('Original mask and mesh have different CTA geometry or class coverage')
    if (not np.array_equal(parent.mask(full), values > 0)
            or not np.array_equal(parent.array(attachment['vertices_ras_mm']), points)
            or not np.array_equal(parent.array(attachment['triangles']), faces)
            or not np.array_equal(parent.array(attachment['cell_labels']), labels)):
        raise ValueError('Parent display must retain exactly the original CoW mask and mesh')
    records = {'source_receipt': binding(root, source), 'qualification': binding(root, qualified)}
    geometry = (values, points, faces, labels)
    topology = graph_topology(members[GRAPH], evidence['graph'], present)
    return records, geometry, topology


def graph_topology(path, expected, present):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

    reader = vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    graph = reader.GetOutput()
    point_count = graph.GetNumberOfPoints()
    cells = graph.GetLines()
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    if (point_count != expected['points'] or graph.GetNumberOfCells() != graph.GetNumberOfLines()
            or graph.GetNumberOfLines() != expected['line_cells']
            or not np.all(np.diff(offsets) == 2)):
        raise ValueError('Publisher centerline connectivity differs from qualification')
    edges = vtk_to_numpy(cells.GetConnectivityArray()).reshape(-1, 2)
    source_degrees = vtk_to_numpy(graph.GetPointData().GetArray('degree'))
    labels = vtk_to_numpy(graph.GetCellData().GetArray('labels'))
    if (edges.min() < 0 or edges.max() >= point_count or set(labels.tolist()) != present
            or not np.array_equal(source_degrees, np.bincount(edges.ravel(), minlength=point_count))):
        raise ValueError('Publisher centerline degrees, classes or indices differ')
    adjacency = coo_matrix((np.ones(len(edges)), edges.T), shape=(point_count, point_count)).tocsr()
    components, membership = connected_components(adjacency, directed=False)
    if (components != expected['connected_components']
            or int(np.bincount(membership).max()) != expected['largest_component_points']):
        raise ValueError('Publisher centerline component inventory differs')
    return {'points': point_count, 'edges': len(edges), 'components': components,
            'component_sizes': sorted(np.bincount(membership).tolist(), reverse=True),
            'branch_nodes': int(np.count_nonzero(source_degrees > 2)),
            'endpoints': int(np.count_nonzero(source_degrees == 1)),
            'cycle_rank': len(edges) - point_count + components,
            'edges_per_class': {str(value): int(np.count_nonzero(labels == value))
                                for value in sorted(present)}}


def layer_id(value):
    return f'cow-{value:02d}-{CLASSES[value][0].lower()}'


def add_branches(collection, geometry):
    values, points, faces, labels = geometry
    full = collection.layer(LAYER)
    present = np.unique(labels).astype(int).tolist()
    for value in present:
        code, name, color = CLASSES[value]
        # Compact each partition for independent picking and framing. The stored
        # maps restore every selected triangle's original vertex and face IDs.
        face_ids = np.flatnonzero(labels == value).astype(np.uint32)
        vertex_ids, indices = np.unique(faces[face_ids], return_inverse=True)
        row = collection.add_layer(
            values, full['affine'], layer_id=layer_id(value), source_series=SERIES,
            kind='artery', role='context', color=color, label=name, label_value=value,
            provenance={**full['provenance'], 'publisher_class_id': value,
                        'publisher_class_label': code, 'evidence_use': 'Publisher class and mesh partition for display'},
        )
        row.update(
            units_status=full['units_status'],
            coverage='Publisher anatomical class within the sampled Circle of Willis',
            publisher_surface={
                **full['publisher_surface'], 'mask_sha256': row['data']['sha256'],
                'mask_label_value': value, 'mask_revision': row['revision'],
                'vertices_ras_mm': collection.put_array(points[vertex_ids]),
                'triangles': collection.put_array(indices.reshape(-1, 3).astype(np.uint32)),
                'cell_labels': collection.put_array(labels[face_ids]),
                'publisher_vertex_ids': collection.put_array(vertex_ids),
                'publisher_face_ids': collection.put_array(face_ids),
            },
        )
    initial = collection.document['display']['initial_layers']
    color_layers = [layer_id(value) for value in present]
    collection.document['display']['initial_layers'] = [name for name in initial if name != LAYER] + color_layers


def verify_branches(collection, parent, geometry):
    values, points, faces, labels = geometry
    old_layers = {row['id']: row for row in parent.document['layers']}
    new_layers = {name: collection.layer(name) for name in old_layers}
    if (any({**old, 'patient_id': CASE} != new_layers[name] for name, old in old_layers.items())
            or any({**row, 'patient_id': CASE} != collection.series(row['id']) for row in parent.document['series'])
            or collection.document['transforms'] != parent.document['transforms']
            or collection.document['publisher_cow_evidence'] != parent.document['publisher_cow_evidence']):
        raise ValueError('Branch companion changed a retained parent source, annotation or geometry')
    present = np.unique(labels).astype(int).tolist()
    branch_ids = [layer_id(value) for value in present]
    if set(row['id'] for row in collection.document['layers']) != set(old_layers) | set(branch_ids):
        raise ValueError('Branch companion has an unexpected layer inventory')
    inverse = np.linalg.inv(np.asarray(collection.series(SERIES)['affine']))
    original_seen = np.zeros(len(faces), dtype=np.uint8)
    ids, counts = np.unique(values, return_counts=True)
    counts_by_class = dict(zip(ids.tolist(), counts.tolist()))
    results = []
    for value in present:
        row = collection.layer(layer_id(value))
        surface = row['publisher_surface']
        vertex_ids = collection.array(surface['publisher_vertex_ids'])
        face_ids = collection.array(surface['publisher_face_ids'])
        indices = collection.array(surface['triangles'])
        branch_positions = collection.array(surface['vertices_ras_mm'])
        rendered = publisher_surface(collection, row, cancelled=lambda: False)
        expected_faces = np.flatnonzero(labels == value)
        if (row['kind'] != 'artery' or row['role'] != 'context' or row['label_value'] != value
                or not np.array_equal(collection.array(row['data']), values)
                or not np.array_equal(collection.mask(row), values == value)
                or not np.array_equal(face_ids, expected_faces)
                or not np.array_equal(vertex_ids[indices], faces[face_ids])
                or not np.array_equal(branch_positions, points[vertex_ids])
                or not np.array_equal(collection.array(surface['cell_labels']), labels[face_ids])
                or rendered['policy_version'] != POLICY or rendered['max_displacement_mm'] != 0
                or not np.array_equal(rendered['positions'], world(inverse, branch_positions))
                or not np.array_equal(rendered['indices'], indices)):
            raise ValueError(f'Exact publisher class or triangle partition differs: {row["id"]}')
        original_seen[face_ids] += 1
        where = np.nonzero(values == value)
        bounds = tuple(slice(index.min(), index.max() + 1) for index in where)
        _, count = label_components(values[bounds] == value, np.ones((3, 3, 3)))
        results.append({'layer': row['id'], 'name': row['label'],
                        'publisher_class_id': value, 'publisher_class': CLASSES[value][0],
                        'voxels': counts_by_class[value], 'voxel_components_26': count,
                        'points': len(vertex_ids), 'triangles': len(face_ids)})
    if not np.all(original_seen == 1) or sum(record['voxels'] for record in results) != np.count_nonzero(values):
        raise ValueError('Branch partitions must preserve the complete CoW mesh and voxel coverage')
    return results


def build(root):
    destination = safe_path(root, OUTPUT)
    path = safe_path(root, f'{PARENT}/receipt.json')
    parent_receipt = read_document(path)
    parent = PatientImagingCollection(checked_file(root, parent_receipt['manifest']).parent)
    if parent.identity() != parent_receipt['snapshot_id']:
        raise ValueError('Publisher companion differs from its receipt')
    sources, geometry, topology = read_inputs(root, parent)
    parent_binding = binding(root, path)
    if destination.exists():
        report = read_document(destination / 'receipt.json')
        collection = PatientImagingCollection(checked_file(root, report['manifest']).parent)
        if (collection.root != destination or collection.identity() != report['snapshot_id']
                or report['parent'] != parent_binding or report['qualified_sources'] != sources
                or report['centerline_topology'] != topology
                or verify_branches(collection, parent, geometry) != report['branches']):
            raise ValueError('Existing branch companion differs from its verified receipt')
        check_packaged_members(collection)
        return report
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.cow-branches-', dir=destination.parent) as temporary:
        stage = Path(temporary) / CASE
        stage.mkdir()
        collection = copy_showcase(root, stage, parent, CASE)
        add_branches(collection, geometry)
        collection.document['publisher_cow_branch_display'] = {
            'parent_manifest_sha256': parent_receipt['manifest']['sha256'], 'role': 'context',
            'method': 'Exact categorical source and triangle subsets of the original publisher mesh',
            'coverage': 'Selected CoW arteries within the original CTA',
        }
        collection.save('Expose publisher Circle of Willis classes in their native CTA frame')
        reopened = PatientImagingCollection(stage)
        branches = verify_branches(reopened, parent, geometry)
        check_packaged_members(reopened)
        report = {'schema': SCHEMA, 'case_id': CASE, 'source_family_id': FAMILY, 'role': 'DEVELOPMENT',
                  'scope': 'Native display and source inspection', 'new_people': 0, 'new_inference_runs': 0,
                  'parent': parent_binding, 'qualified_sources': sources, 'snapshot_id': reopened.identity(),
                  'manifest': {'path': f'{OUTPUT}/patient-anatomy.json',
                               'sha256': sha256_file(stage / 'patient-anatomy.json')},
                  'centerline_topology': topology, 'branches': branches}
        atomic_json(stage / 'receipt.json', report)
        checked_root(root, expected_uuid=UUID)
        stage.rename(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--register', type=Path, help='Optional local Cases directory for the native study link')
    args = parser.parse_args()
    root = checked_root(args.root, expected_uuid=UUID, required_bytes=2 * 1024**3)
    with reconstruction_lock():
        result = build(root)
        if args.register is not None:
            register(root, args.register, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
