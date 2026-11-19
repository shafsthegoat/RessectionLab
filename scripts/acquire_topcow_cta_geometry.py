#!/usr/bin/env python3
"""Select and qualify the publisher CTA Circle of Willis geometry for 012 and 014."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile
import zlib

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from data_rebuild_acquire import DEFAULT_ROOT, UUID, request_range
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file


DESTINATION = Path('sources/topcow-cow-vtp-selective-v1')
SUBJECTS = ('012', '014')
COW_IDS = {*range(1, 13), 15}
MIB = 1024**2
TRANSFER_CAP = 8 * MIB  # Per publisher archive, including the index and both subjects.
MEMBER_CAP = 4 * MIB


@dataclass(frozen=True)
class Release:
    record: int
    name: str
    size: int
    md5: str
    entries: int
    central: tuple[int, int, str]
    tail_sha256: str

    @property
    def url(self):
        return f'https://zenodo.org/api/records/{self.record}/files/{self.name}/content'


# The index and footer digests are from the independently checked 006 receipt.
RELEASES = {
    'geometry': Release(17358162, 'CoW_Centerline_Data.zip', 373_088_743,
                        '8e414a6cd50e85ef4a8f68e37fb25fc3', 1258,
                        (372_928_858, 159_863, '185d18c83237a2356dc4a4b41c2c85aa879aad68b9e316b008f1a4a44b2c1e80'),
                        '99d2ad0350618a22af73a72223f947a6ad91c25367b2e46436a4d0318ffaeaec'),
    'original': Release(15692630, 'TopCoW2024_Data_Release.zip', 10_512_833_650,
                        'd248a932166316c6c906644b22296d6c', 1018,
                        (10_512_693_270, 140_282, 'fe69b05f6516cee9106117893b397760a40b5d45cfb0cdc95c9e9d3ece5c35d9'),
                        '5c085c41f4152d70ab70bb77c59ecb82ee0b089889b53e4d1e30774d2ebf5974'),
}


class RangeArchive(io.RawIOBase):
    """Pin the official ZIP index and reject full or transformed responses."""

    def __init__(self, release):
        self.release = release
        self.position = 0
        self.ranges = []
        self.transferred = 0
        self.index_start, size, digest = release.central
        self.index = self.fetch(self.index_start, release.size - self.index_start)
        if (hashlib.sha256(self.index[:size]).hexdigest() != digest
                or hashlib.sha256(self.index[-65_557:]).hexdigest() != release.tail_sha256):
            raise ValueError('Publisher ZIP indexes differ from the verified 006 release')

    def fetch(self, first, size):
        if (size <= 0 or size > MEMBER_CAP or first < 0 or first + size > self.release.size
                or size + self.transferred > TRANSFER_CAP):
            raise ValueError('Range exceeds the pinned publisher ZIP or transfer cap')
        self.transferred += size
        with request_range(self.release.url, first, first + size - 1, self.release.size) as response:
            if (response.headers.get('Content-Encoding', 'identity') != 'identity'
                    or int(response.headers.get('Content-Length', size)) != size):
                raise ValueError('Publisher returned a transformed or wrong-length archive range')
            data = response.read(size)
        if len(data) != size:
            raise ValueError('Publisher ZIP range was truncated')
        self.ranges.append({'start': first, 'length': size, 'sha256': hashlib.sha256(data).hexdigest()})
        return data

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        start = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.release.size}[whence]
        self.position = start + offset
        if self.position < 0:
            raise ValueError('Negative ZIP seek')
        return self.position

    def read(self, count=-1):
        available = self.release.size - self.position
        count = min(count, available) if count >= 0 else available
        if count <= 0:
            return b''
        cache_start = self.position - self.index_start
        data = (self.index[cache_start:cache_start + count] if cache_start >= 0
                else self.fetch(self.position, count))
        self.position += len(data)
        return data


def archive_record(info):
    return {'archive_path': info.filename, 'bytes': info.file_size, 'compressed_bytes': info.compress_size,
            'crc32': f'{info.CRC:08x}', 'header_offset': info.header_offset}


def locked(root, entry):
    path = safe_path(root, entry.get('retained_path') or entry['path'])
    if path.stat().st_size != entry['bytes'] or sha256_file(path) != entry['sha256']:
        raise ValueError(f'Source identity mismatch: {path}')
    return path


def selected_member(root, directory, archive, name):
    info = archive.getinfo(name)
    if (not 0 < info.file_size <= MEMBER_CAP or not 0 < info.compress_size <= MEMBER_CAP
            or info.is_dir() or info.flag_bits & 1 or info.compress_type not in (0, 8)):
        raise ValueError(f'Unsupported selected ZIP member: {name}')
    path = safe_path(root, directory / 'publisher-members' / name)
    payload = path.read_bytes() if path.exists() else archive.read(info)
    if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
        raise ValueError(f'Archive CRC or member size differs for {name}')
    if not path.exists():
        checked_root(root, required_bytes=len(payload), expected_uuid=UUID)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = safe_path(root, path.relative_to(root).with_name(path.name + '.partial'))
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return {**archive_record(info), 'retained_path': str(path.relative_to(root)),
            'sha256': hashlib.sha256(payload).hexdigest()}


def acquire(root, subject, open_archive):
    directory = DESTINATION / subject
    receipt = safe_path(root, directory / 'source-receipt.json')
    topbrain = safe_path(root, f'catalog/topbrain/{subject}-selective-extraction.json')
    local = json.loads(topbrain.read_text())
    family = f'TopCoW:{subject}'
    if local['family']['family_id'] != family or local['family']['role'] != 'TRAIN':
        raise ValueError('Source family or frozen TRAIN role differs')
    old_cta = next(row for row in local['files'] if row['path'].endswith(f'/imagesTr_topbrain/topcow_ct_{subject}_0000.nii.gz'))
    old_v1 = next(row for row in local['files'] if row['path'].endswith(f'/labelsTr_topbrain_v1_ct/topcow_ct_{subject}.nii.gz'))
    topbrain_sha256 = sha256_file(topbrain)
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if (result['family'] != family or result['role'] != 'TRAIN' or result['local_cta'] != old_cta
                or result['local_topbrain_v1'] != old_v1 or result['topbrain_acquisition_sha256'] != topbrain_sha256):
            raise ValueError('Existing source receipt differs from the frozen association')
        return receipt, result
    for row in (old_cta, old_v1):
        locked(root, row)
    releases = {}
    for key in ('original', 'geometry'):
        reader, archive = open_archive(key)
        start = len(reader.ranges)
        reused = []
        if key == 'original':
            prefix = 'TopCoW2024_Data_Release/'
            names = [f'{prefix}cow_seg_labelsTr/topcow_ct_{subject}.nii.gz']
            original = archive_record(archive.getinfo(f'{prefix}imagesTr/topcow_ct_{subject}_0000.nii.gz'))
            if original['bytes'] != old_cta['bytes'] or original['crc32'] != old_cta['zip_crc32']:
                raise ValueError('Source TopCoW CTA differs from the retained original TopBrain CTA member')
            reused.append({**original, 'local_reused_path': old_cta['path'], 'local_reused_sha256': old_cta['sha256'],
                           'basis': 'same subject, exact publisher member size and CRC, retained SHA-256'})
        else:
            prefix = 'CoW_Centerline_Data/'
            names = [f'{prefix}cow_{group}/topcow_ct_{subject}.{suffix}'
                     for group, suffix in [('graphs', 'vtp'), ('meshes', 'vtp'), ('nodes', 'json')]]
        extracted = [selected_member(root, directory, archive, name) for name in names]
        selected_ranges = reader.ranges[start:]
        releases[key] = {
            'url': reader.release.url, 'archive_bytes': reader.release.size,
            'publisher_archive_md5_unverified': reader.release.md5, 'zip_entries': reader.release.entries,
            'archive_index_range': reader.ranges[0], 'extracted': extracted, 'reused': reused,
            'fetched_ranges': selected_ranges, 'network_bytes': sum(row['length'] for row in selected_ranges),
        }
    result = {'schema': 'ressectionlab-publisher-cow-cta-geometry-pilot-source-v1', 'family': family,
              'role': 'TRAIN', 'local_cta': old_cta, 'local_topbrain_v1': old_v1, 'releases': releases,
              'topbrain_acquisition_sha256': topbrain_sha256, 'new_people': 0,
              'geometry_license': 'CC BY-NC 4.0',
              'original_license': 'OpenData Swiss custom: attribution; commercial use requires prior owner permission'}
    checked_root(root, expected_uuid=UUID)
    atomic_json(receipt, result)
    return receipt, result


def polydata(path, kind, width):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

    reader = vtkXMLPolyDataReader()
    if not reader.CanReadFile(str(path)):
        raise ValueError(f'Invalid publisher VTP: {path.name}')
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetPoints() is None or poly.GetCellData().GetArray('labels') is None:
        raise ValueError(f'Publisher VTP lacks points or class labels: {path.name}')
    points = vtk_to_numpy(poly.GetPoints().GetData()).copy()
    cells = getattr(poly, kind)()
    indices = vtk_to_numpy(cells.GetConnectivityArray()).copy()
    labels = vtk_to_numpy(poly.GetCellData().GetArray('labels')).copy()
    if (points.dtype != np.float32 or not np.isfinite(points).all() or not len(indices)
            or indices.min() < 0 or indices.max() >= len(points)
            or not np.all(np.diff(vtk_to_numpy(cells.GetOffsetsArray())) == width)
            or len(labels) != len(indices) // width or len(labels) != poly.GetNumberOfCells()):
        raise ValueError(f'VTP topology or cell labels differ: {path.name}')
    index = poly.GetPointData().GetArray('degree')
    return points, indices.reshape(-1, width), labels, vtk_to_numpy(index).copy() if index is not None else None


def point_support(points, reference, inverse):
    indices = np.rint(nib.affines.apply_affine(inverse, points)).astype(np.int64)
    valid = ((indices >= 0) & (indices < reference.shape)).all(axis=1)
    indexes = np.zeros(len(indices), dtype=reference.dtype)
    indexes[valid] = reference[tuple(indices[valid].T)]
    return indexes, {'points_in_source_fov': int(valid.sum()), 'points_in_source_positive_mask': int(np.count_nonzero(indexes))}


def connectivity(cells, size):
    edges = np.concatenate([cells[:, [0, i]] for i in range(1, cells.shape[1])])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(size, size))
    count, components = connected_components(graph.tocsr(), directed=False)
    return {'connected_components': count, 'largest_component_points': int(np.bincount(components).max())}


def distance_summary(distances):
    return dict(zip(('p50', 'p95', 'p99', 'max'), map(float, np.percentile(distances, [50, 95, 99, 100]))))


def qualify(root, receipt, source):
    from vtkmodules.vtkCommonCore import vtkVersion

    geometry = {Path(row['archive_path']).parent.name: locked(root, row)
                for row in source['releases']['geometry']['extracted']}
    mask_path = locked(root, source['releases']['original']['extracted'][0])
    cta, cow, topbrain = map(nib.load, (locked(root, source['local_cta']), mask_path, locked(root, source['local_topbrain_v1'])))
    if (cta.header.get_xyzt_units()[0] != 'mm' or cta.header.get_sform(coded=True)[1] <= 0
            or any(image.shape != cta.shape or not np.array_equal(image.affine, cta.affine)
                   or image.header.get_sform(coded=True)[1] <= 0 for image in (cow, topbrain))):
        raise ValueError('Publisher original CoW and TopBrain labels differ from the native CTA grid')
    reference, wide = np.asanyarray(cow.dataobj), np.asanyarray(topbrain.dataobj)
    classes = set(np.unique(reference)) - {0}
    if reference.dtype != np.uint8 or not classes or not classes <= COW_IDS:
        raise ValueError('Unexpected publisher CoW categorical labels')
    positive = reference > 0
    overlap = {'original_positive': int(positive.sum()),
               'original_positives_equal_id_in_topbrain_v1': int(np.count_nonzero(wide[positive] == reference[positive])),
               'topbrain_v1_positives_with_an_inherited_cow_id': int(np.count_nonzero(np.isin(wide, sorted(COW_IDS))))}
    del wide

    graph, lines, line_classes, degree = polydata(geometry['cow_graphs'], 'GetLines', 2)
    mesh, faces, face_classes, _ = polydata(geometry['cow_meshes'], 'GetPolys', 3)
    if (not set(line_classes) <= classes or not set(face_classes) <= classes
            or not np.array_equal(np.bincount(lines.ravel(), minlength=len(graph)), degree)
            or np.any(lines[:, 0] == lines[:, 1]) or len(np.unique(np.sort(lines, axis=1), axis=0)) != len(lines)):
        raise ValueError('Publisher geometry levels, graph degree or edge topology disagree')
    nodes = [node for segment in json.loads(geometry['cow_nodes'].read_text()).values()
             for entries in segment.values() for node in entries]
    node_error = max(float(np.linalg.norm(graph[node['id']] - node['coords'])) for node in nodes)
    if node_error >= 1e-4:
        raise ValueError('Publisher named nodes differ from graph positions')

    inverse = np.linalg.inv(cta.affine)
    _, graph_support = point_support(graph, reference, inverse)
    _, flipped_graph = point_support(graph * [-1, -1, 1], reference, inverse)
    midpoint_values, midpoint_support = point_support(graph[lines].mean(axis=1), reference, inverse)
    midpoint_support['correct_class_voxels'] = int(np.count_nonzero(midpoint_values == line_classes))
    _, vertex_support = point_support(mesh, reference, inverse)
    _, flipped_mesh = point_support(mesh * [-1, -1, 1], reference, inverse)
    mask_boundary = np.column_stack(np.nonzero(positive & ~binary_erosion(positive)))
    boundary_points = nib.affines.apply_affine(cta.affine, mask_boundary)
    mesh_distances = distance_summary(cKDTree(boundary_points).query(mesh)[0])
    incidence = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), return_counts=True, axis=0)[1]
    graph_ok = (graph_support['points_in_source_positive_mask'] > .95 * len(graph)
                and midpoint_support['points_in_source_positive_mask'] > .95 * len(lines)
                and flipped_graph['points_in_source_positive_mask'] < .05 * len(graph))
    mesh_ok = (vertex_support['points_in_source_fov'] == len(mesh)
               and mesh_distances['p95'] <= np.linalg.norm(nib.affines.voxel_sizes(cta.affine))
               and flipped_mesh['points_in_source_positive_mask'] < .05 * len(mesh))
    measurements = {
        'schema': 'ressectionlab-publisher-cow-cta-geometry-pilot-verification-v1',
        'source_receipt_sha256': sha256_file(receipt), 'vtk_version': vtkVersion.GetVTKVersion(),
        'family': source['family'], 'role': source['role'], 'new_people': 0,
        'native_cta_shape': list(cta.shape), 'native_cta_ras_affine_mm': cta.affine.tolist(),
        'original_and_topbrain_v1_mask_grid_exact': True, 'original_cta_member_reused': True,
        'native_graph_coordinates': 'native CTA RAS+ mm' if graph_ok else 'unverified stored frame',
        'native_mesh_coordinates': 'native CTA RAS+ mm' if mesh_ok else 'unverified stored frame',
        'frame_transform_applied': False, 'display_qualified': bool(graph_ok and mesh_ok),
        'original_CoW_vs_topbrain_v1': overlap,
        'graph': {'points': len(graph), 'line_cells': len(lines), **connectivity(lines, len(graph)),
                  'nonempty_class_ids': sorted(set(line_classes.tolist())), 'point_support': graph_support,
                  'lps_negative_control': flipped_graph, 'edge_midpoint_support': midpoint_support,
                  'named_node_count': len(nodes), 'named_node_max_error_mm': node_error,
                  'frame_qualified': bool(graph_ok)},
        'mesh': {'vertices': len(mesh), 'triangles': len(faces), **connectivity(faces, len(mesh)),
                 'unique_edges': len(incidence), 'boundary_edges': int(np.count_nonzero(incidence == 1)),
                 'nonmanifold_edges': int(np.count_nonzero(incidence > 2)),
                 'nonempty_class_ids': sorted(set(face_classes.tolist())), 'vertex_support': vertex_support,
                 'lps_negative_control': flipped_mesh, 'frame_qualified': bool(mesh_ok),
                 'vertex_to_nearest_original_mask_boundary_center_mm': mesh_distances},
        'used': 'Publisher CoW arterial-region display geometry; preserves broader TopBrain masks and frozen source role',
    }
    # Record each part separately if one original is spatially inconsistent.
    if not (graph_ok and mesh_ok):
        measurements['unapplied_bounds_diagnostic_ras_mm'] = {
            name: [points.min(axis=0).tolist(), points.max(axis=0).tolist()]
            for name, points in [('original_mask_boundary', boundary_points), ('mesh', mesh), ('graph', graph)]
        }
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(os.environ.get('RESSECTIONLAB_DATA_ROOT') or DEFAULT_ROOT))
    parser.add_argument('subjects', nargs='*', help='012 and/or 014, default both')
    args = parser.parse_args()
    subjects = args.subjects or SUBJECTS
    if set(subjects) - set(SUBJECTS) or len(subjects) != len(set(subjects)):
        parser.error('Choose distinct subjects 012 and/or 014')
    root = checked_root(args.data_root, expected_uuid=UUID, required_bytes=16 * MIB)
    output = safe_path(root, DESTINATION)
    output.mkdir(parents=True, exist_ok=True)
    with safe_path(root, DESTINATION / '.acquisition.lock').open('a') as lock, ExitStack() as stack:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        archives = {}

        def open_archive(key):
            if key not in archives:
                reader = stack.enter_context(RangeArchive(RELEASES[key]))
                archive = stack.enter_context(zipfile.ZipFile(reader))
                if len(archive.infolist()) != reader.release.entries:
                    raise ValueError('Publisher archive member inventory differs')
                archives[key] = (reader, archive)
            return archives[key]

        for subject in subjects:
            receipt, source = acquire(root, subject, open_archive)
            result = qualify(root, receipt, source)
            verification = safe_path(root, DESTINATION / subject / 'verification.json')
            checked_root(root, expected_uuid=UUID)
            atomic_json(verification, result)
            print(json.dumps({'family': source['family'], 'verification': str(verification),
                              'display_qualified': result['display_qualified'],
                              'graph_points': result['graph']['points'], 'mesh_triangles': result['mesh']['triangles'],
                              'network_bytes_this_run': sum(reader.transferred for reader, _ in archives.values())}), flush=True)


if __name__ == '__main__':
    main()
