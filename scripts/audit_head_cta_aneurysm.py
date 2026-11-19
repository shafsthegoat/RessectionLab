#!/usr/bin/env python3
"""Compare axis conventions for the retained Head CTA 013 aneurysm STL."""

import argparse
import csv
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
import vtk
from scipy.ndimage import binary_dilation, label
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from acquire_head_cta import RELEASE, UUID
from data_integrated_hosts import binding, checked_file
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection, reconstruction_lock, world


QC = Path('qc/head-cta-15697196-native/013')
FAMILY = 'HeadCTA15697196:013'


def voxelize(poly, vertices, hu):
    low = np.floor(vertices.min(axis=0)).astype(int) - 4
    high = np.ceil(vertices.max(axis=0)).astype(int) + 4
    if np.any(low < 0) or np.any(high >= hu.shape):
        return {'inside_grid_with_margin': False}, None
    region = tuple(slice(a, b + 1) for a, b in zip(low, high))
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=True))
    native = vtk.vtkPolyData()
    native.ShallowCopy(poly)
    native.SetPoints(points)
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(native)
    stencil.SetOutputOrigin(0, 0, 0)
    stencil.SetOutputSpacing(1, 1, 1)
    stencil.SetOutputWholeExtent(tuple(np.column_stack((low, high)).ravel()))
    image = vtk.vtkImageStencilToImage()
    image.SetInputConnection(stencil.GetOutputPort())
    image.SetInsideValue(1)
    image.SetOutsideValue(0)
    image.Update()
    mask = vtk_to_numpy(image.GetOutput().GetPointData().GetScalars()).reshape(high - low + 1, order='F') > 0
    crop = hu[region]
    inside = crop[mask]
    shell = crop[binary_dilation(mask, iterations=2) & ~mask]
    return {'inside_grid_with_margin': True, 'voxel_centers': int(inside.size),
            'components_6_connected': int(label(mask)[1]),
            'inside_hu_p10_p50_p90': np.percentile(inside, (10, 50, 90)).tolist(),
            'two_step_shell_hu_p10_p50_p90': np.percentile(shell, (10, 50, 90)).tolist(),
            'inside_at_least_150_hu_fraction': float(np.mean(inside >= 150))}, (region, mask)


def plot_review(output, hu, placements, axial, midline):
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 6), layout='constrained')
    grid = fig.add_gridspec(2, 4)
    full = fig.add_subplot(grid[:, :2])
    full.imshow(hu[:, :, axial].T, cmap='gray', vmin=0, vmax=300, origin='lower')
    full.axvline(midline, color='#68b9ff', linestyle='--', linewidth=.8)
    full.text(.03, .95, 'R', transform=full.transAxes, color='white')
    full.text(.94, .95, 'L', transform=full.transAxes, color='white')
    full.set_title('Native axial CTA')
    full.set_axis_off()
    for index, (name, (region, mask)) in enumerate(placements.items()):
        a, b, c = region
        cut = axial - c.start
        color = '#ff9f5a' if name == 'LPS' else '#f0e762'
        full.contour(np.arange(a.start, a.stop), np.arange(b.start, b.stop), mask[:, :, cut].T,
                     [.5], colors=[color], linewidths=[1.2])
        full.text(a.start - 3, b.stop + 2, name, color=color, fontsize=9)
        ax = fig.add_subplot(grid[index // 2, index % 2 + 2])
        ax.imshow(hu[region][:, :, cut].T, cmap='gray', vmin=0, vmax=300, origin='lower')
        ax.contour(mask[:, :, cut].T, [.5], colors=[color], linewidths=[1.1])
        ax.set_title(f'{name} hypothesis')
        ax.set_axis_off()
    fig.suptitle('Head CTA 013 · diagnostic STL frame review')
    fig.savefig(output, dpi=150)
    plt.close(fig)


def audit(root, stage):
    review = safe_path(root, QC / 'verification.json')
    verified = json.loads(review.read_text())
    manifest = checked_file(root, {'path': verified['native_manifest'], 'sha256': verified['native_manifest_sha256']})
    receipt = checked_file(root, {'path': verified['source_acquisition_receipt'],
                                 'sha256': verified['source_acquisition_receipt_sha256']})
    acquisition = json.loads(receipt.read_text())
    study = PatientImagingCollection(manifest.parent)
    scan, = study.document['series']
    if (acquisition['source_person_family'] != FAMILY or study.document['patient_id'] != FAMILY
            or acquisition['publisher_release'] != RELEASE or scan['intensity_units'] != 'HU'
            or scan['data']['shape'] != [512, 512, 289] or scan['affine'] != verified['native_affine_ras_mm']
            or scan['data']['sha256'] != verified['native_scalar_sha256']
            or scan['original']['sha256'] != verified['native_nifti_sha256']):
        raise ValueError('Expected the reviewed Head CTA 013 scan and DICOM frame')
    scalar = study.path(scan['data']['path'])
    if sha256_file(scalar) != verified['native_scalar_sha256']:
        raise ValueError('Native CTA scalar checksum differs')
    hu = np.load(scalar, mmap_mode='r', allow_pickle=False)
    if hu.shape != (512, 512, 289) or hu.dtype != np.float32:
        raise ValueError('Native scalar array does not match the verified lattice')
    files = {item['source_member']: item for item in acquisition['files']}
    mesh = checked_file(root, files['Annotations/an_013.stl'])
    metadata = checked_file(root, files['Metadata.csv'])
    with metadata.open(encoding='cp1251', newline='') as stream:
        row, = (line for line in csv.DictReader(stream, delimiter=';') if line['ID'] == '013')
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(mesh))
    reader.Update()
    poly = reader.GetOutput()
    if not poly.GetNumberOfPolys():
        raise ValueError('Publisher STL contains no polygons')
    vertices = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(poly)
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    if not np.isfinite(vertices).all() or edges.GetOutput().GetNumberOfCells():
        raise ValueError('A finite closed manifold is required for voxelization')
    mass = vtk.vtkMassProperties()
    mass.SetInputData(poly)
    affine = np.asarray(scan['affine'])
    inverse = np.linalg.inv(affine)
    candidates, placements = {}, {}
    for axes in product(('R', 'L'), ('A', 'P'), ('S', 'I')):
        name = ''.join(axes)
        signs = [1 if axis in 'RAS' else -1 for axis in axes]
        native = world(inverse, vertices * signs)
        stats, placement = voxelize(poly, native, hu)
        candidates[name] = {**stats, 'axis_signs_to_ras': signs, 'centroid_ijk': native.mean(0).tolist()}
        if placement is not None:
            placements[name] = placement
    if list(placements) != ['RAS', 'RPS', 'LAS', 'LPS']:
        raise ValueError('Native-grid hypotheses differ from the reviewed case')
    x, _, z = candidates['LPS']['centroid_ijk']
    axial = int(np.rint(z))
    dense_x = np.where(hu[:, :, axial] > 500)[0]
    midline = float((dense_x.max() + dense_x.min()) / 2)
    overlay = stage / 'native-frame-review.png'
    plot_review(overlay, hu, placements, axial, midline)
    return {'schema': 'head-cta-013-stl-frame-diagnostic-v1', 'family_id': FAMILY,
            'publisher_release': RELEASE, 'publisher_methods': 'https://www.mdpi.com/2306-5729/11/4/74',
            'source_review': binding(root, review), 'source_acquisition': binding(root, receipt),
            'native_manifest': binding(root, manifest), 'native_scalar_sha256': verified['native_scalar_sha256'],
            'original_stl': binding(root, mesh), 'original_metadata': binding(root, metadata),
            'publisher_parent_artery': row['Artery'], 'publisher_volume_mm3': float(row['Vobject, mm3'].replace(',', '.')),
            'native_affine_ras_mm': scan['affine'], 'mesh_points': len(vertices), 'mesh_triangles': poly.GetNumberOfPolys(),
            'mesh_volume_mm3': float(mass.GetVolume()), 'vtk_version': vtk.vtkVersion.GetVTKVersion(), 'hypotheses': candidates,
            'midline_basis': 'Midpoint of the above-500 HU x extents at the LPS centroid axial slice',
            'LPS_offset_ras_x_from_approximate_axial_skull_center_mm': float((x - midline) * affine[0, 0]),
            'decision': 'diagnostic_only', 'reason': 'The LPS hypothesis is supported by enhanced CTA tissue on the '
            'higher-R side of the approximate calvarial midline. The source metadata names the left MCA. '
            'The publisher does not declare the STL axis convention. No fitted transform was attempted.',
            'case_layer_added': False, 'overlay_sha256': sha256_file(overlay)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output-name', default='stl-frame-v1')
    args = parser.parse_args()
    if not args.output_name or Path(args.output_name).name != args.output_name or args.output_name.startswith('.'):
        parser.error('Choose one new output directory name')
    root = checked_root(args.root, expected_uuid=UUID)
    destination = safe_path(root, QC / args.output_name)
    with reconstruction_lock():
        if destination.exists():
            raise FileExistsError(destination)
        with TemporaryDirectory(prefix='.stl-audit-', dir=destination.parent) as temp:
            stage = Path(temp) / 'review'
            stage.mkdir()
            atomic_json(stage / 'receipt.json', audit(root, stage))
            checked_root(root, expected_uuid=UUID)
            stage.rename(destination)
    print(destination / 'receipt.json')


if __name__ == '__main__':
    main()
