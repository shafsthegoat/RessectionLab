#!/usr/bin/env python3
"""Native CTA engineering anatomy proposals; never alters acquired/native cases.

Enclosure is geometric evidence for an intracranial envelope, not clinical brain
segmentation. Open/cropped slices stay unknown. All outputs require review.
"""
from __future__ import annotations
import argparse
import itertools
import json
import re
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import numpy as np
from scipy import ndimage as ndi
from resectionlab.data_root import checked_root, safe_path, sha256_file, atomic_json
from data_rebuild_acquire import UUID


def physical_bounds(shape, affine):
    corners = np.array(list(itertools.product(*[(-.5, n-.5) for n in shape])))
    world = corners @ affine[:3, :3].T + affine[:3, 3]
    return {'minimum_ras_mm': world.min(0).tolist(), 'maximum_ras_mm': world.max(0).tolist()}


def largest(mask):
    labels, number = ndi.label(mask)
    if not number:
        return np.zeros(mask.shape, bool)
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    return labels == sizes.argmax()


def disk_mm(radius, spacing):
    x, y = np.ogrid[tuple(slice(-int(np.ceil(radius/s)), int(np.ceil(radius/s))+1) for s in spacing)]
    return (x*spacing[0])**2 + (y*spacing[1])**2 <= radius**2


def estimate(ct, vessels, spacing, *, bone_cutoff=500., closing_mm=2.):
    """Conservative enclosed soft-tissue proposal and observed dense skull.

    Physical axial slices must be the third native axis; callers reject oblique
    grids. Do not pad a cropped skull closed or infer unseen anatomy. The largest
    3D enclosed cavity rejects small sinus/orbit cavities. Dense skull is measured
    source signal near that cavity, with supplied vascular labels excluded.
    """
    if ct.shape != vessels.shape or ct.ndim != 3 or not np.isfinite(ct).all():
        raise ValueError('Invalid paired source arrays')
    if len(spacing) != 3 or min(spacing) <= 0:
        raise ValueError('Invalid physical spacing')
    dense = ct >= bone_cutoff
    cavity = np.zeros(ct.shape, bool)
    kernel = disk_mm(closing_mm, spacing[:2])
    for z in range(ct.shape[2]):
        wall = ndi.binary_closing(dense[:, :, z], structure=kernel)
        enclosed = ndi.binary_fill_holes(wall) & ~wall
        cavity[:, :, z] = largest(enclosed)
    cavity = largest(cavity)
    if not cavity.any():
        raise ValueError('No enclosed intracranial candidate; do not extrapolate cropped FOV')
    # Preserve fluid spaces and vessel locations inside the geometric envelope;
    # exclude air, dense mineralization and any extracranial enclosure islands.
    brain = cavity & (ct >= -10) & (ct <= 150)
    brain = largest(brain)
    distance = ndi.distance_transform_edt(~cavity, sampling=spacing)
    vessel_distance = ndi.distance_transform_edt(~(vessels > 0), sampling=spacing)
    skull = dense & (distance <= 12.) & (vessel_distance > 1.5)
    # Preserve source-observed disconnected calvarial fragments; don't invent a
    # bridge or keep distant calcification. Each fragment remains reviewable.
    labels, count = ndi.label(skull)
    sizes = np.bincount(labels.ravel()); keep = sizes * np.prod(spacing) >= 20.
    keep[0] = False
    skull = keep[labels]
    support = (brain & (ct >= 10) & (ct <= 90) & (vessel_distance > 3.)
               & (ndi.distance_transform_edt(brain, sampling=spacing) >= 4.))
    return {'brain_estimate': brain, 'observed_skull': skull, 'enclosed_coverage': cavity,
            'tumor_placement_support': support}


def metrics(mask, affine, spacing):
    labels, count = ndi.label(mask)
    counts = np.bincount(labels.ravel())[1:]
    coordinates = np.array(np.where(mask))
    result = {'voxels': int(mask.sum()), 'volume_ml': float(mask.sum()*np.prod(spacing)/1000),
              'components_6_connected': int(count), 'largest_component_fraction': float(counts.max()/mask.sum()) if mask.any() else 0.,
              'boundary_voxels_per_face': [int(np.take(mask, i, axis=a).sum()) for a in range(3) for i in [0, -1]]}
    if mask.any():
        low, high = coordinates.min(1), coordinates.max(1)
        corners = np.array(list(itertools.product(*zip(low-.5, high+.5))))
        world = corners @ affine[:3,:3].T + affine[:3,3]
        result['minimum_ras_mm'] = world.min(0).tolist(); result['maximum_ras_mm'] = world.max(0).tolist()
    return result


def prepare(root, subject, *, version='enclosure-v1'):
    if not isinstance(version,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',version):
        raise ValueError('Use a plain safe immutable version token')
    root = checked_root(root, expected_uuid=UUID, required_bytes=1024**3)
    destination = safe_path(root, f'integrated-v1/anatomy/TopCoW-{subject}/{version}')
    if destination.exists(): raise FileExistsError(f'Immutable output exists: {destination}')
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    started = time.monotonic()
    receipt_path = safe_path(root, f'catalog/topbrain/{subject}-selective-extraction.json')
    receipt = json.loads(receipt_path.read_text())
    image_entry = next(e for e in receipt['files'] if '/imagesTr_topbrain/topcow_ct_' in e['source_member'])
    label_entry = next(e for e in receipt['files'] if '/labelsTr_topbrain_v1_ct/' in e['source_member'])
    sources = []
    for entry in [image_entry, label_entry]:
        p = safe_path(root, entry['path'])
        if sha256_file(p) != entry['sha256']: raise ValueError('Acquired source hash mismatch')
        sources.append(nib.load(p))
    image, label = sources
    if image.shape != label.shape or not np.allclose(image.affine, label.affine, atol=1e-5, rtol=0):
        raise ValueError('Source annotation grid mismatch')
    if image.header.get_xyzt_units()[0] != 'mm': raise ValueError('Source image not millimetres')
    affine = image.affine
    if not np.allclose(affine[:3,:3], np.diag(np.diag(affine[:3,:3])), atol=1e-5) or affine[2,2] <= 0:
        raise ValueError('Method requires axis-aligned axial native CTA; no resampling allowed')
    spacing = np.linalg.norm(affine[:3,:3], axis=0)
    ct = np.asarray(image.dataobj, dtype=np.float32); vessels = np.asarray(label.dataobj)
    proposals = estimate(ct, vessels, spacing)
    checked_root(root, expected_uuid=UUID, required_bytes=ct.nbytes)
    destination.mkdir(parents=True)
    outputs = {}
    for name, data in proposals.items():
        path = destination / (name+'.nii.gz')
        header = image.header.copy(); header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(data.astype(np.uint8), affine, header), path)
        saved = nib.load(path)
        if saved.shape != image.shape or not np.array_equal(saved.affine, affine): raise ValueError('Output grid changed')
        outputs[name] = {'path': str(path.relative_to(root)), 'sha256': sha256_file(path), **metrics(data, affine, spacing)}
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    z_values = np.linspace(0, ct.shape[2]-1, 7, dtype=int)[1:-1]
    for column, z in enumerate(z_values):
        for row, window in enumerate([(-10,100),(-200,1500),(-10,100)]):
            ax = axes[row,column]; ax.imshow(ct[:,:,z].T, origin='lower', cmap='gray', vmin=window[0], vmax=window[1], aspect=spacing[1]/spacing[0])
            if row == 1: ax.contour(proposals['observed_skull'][:,:,z].T, levels=[.5], colors=['cyan'], linewidths=.5)
            if row == 2:
                ax.contour(proposals['brain_estimate'][:,:,z].T, levels=[.5], colors=['lime'], linewidths=.5)
                ax.contour(vessels[:,:,z].T > 0, levels=[.5], colors=['red'], linewidths=.4)
            ax.set_title(f'Native axial z={z}'); ax.axis('off')
    fig.suptitle(f'TopCoW {subject}: native CTA / measured dense skull / brain estimate + supplied vessels\nEngineering estimates: pending independent review; open/cropped anatomy unknown')
    fig.tight_layout(); fig.savefig(destination/'axial-review.png', dpi=120); plt.close(fig)
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    for axis, orientation in enumerate(['Sagittal', 'Coronal', 'Axial']):
        slices = np.linspace(0, ct.shape[axis]-1, 7, dtype=int)[1:-1]
        remaining = [i for i in range(3) if i != axis]
        for column, index in enumerate(slices):
            ax = axes[axis,column]
            ax.imshow(np.take(ct,index,axis=axis).T, origin='lower', cmap='gray', vmin=-10, vmax=100,
                      aspect=spacing[remaining[1]]/spacing[remaining[0]])
            for name, color in [('brain_estimate','lime'), ('observed_skull','cyan')]:
                section = np.take(proposals[name],index,axis=axis).T
                if section.any(): ax.contour(section,levels=[.5],colors=[color],linewidths=.5)
            section = np.take(vessels,index,axis=axis).T > 0
            if section.any(): ax.contour(section,levels=[.5],colors=['red'],linewidths=.4)
            ax.set_title(f'{orientation} native index={index}');ax.axis('off')
    fig.suptitle(f'TopCoW {subject}: native CTA context | green brain estimate | cyan measured skull | red supplied vessels\nAll estimates pending review; no inferred extension beyond acquired FOV')
    fig.tight_layout();fig.savefig(destination/'three-plane-review.png',dpi=120);plt.close(fig)
    result = {'schema':'ressectionlab-integrated-anatomy-estimate-v1', 'preparation_version':version, 'family_id':receipt['family']['family_id'],
              'sources':[image_entry,label_entry], 'receipt_sha256':sha256_file(receipt_path),
              'script_sha256':sha256_file(__file__), 'method':'native axial bone enclosure; 500 scaled-intensity dense wall; 2mm slice closing; largest 3D cavity; -10..150 intracranial tissue; dense skull within12mm cavity, >1.5mm supplied vessels; >=20mm3 fragments',
              'intensity_status':'HU-like source-scaled CT; original DICOM calibration unavailable',
              'physical_bounds':physical_bounds(image.shape,affine), 'shape':list(image.shape), 'affine_ras':affine.tolist(),
              'spacing_mm':spacing.tolist(), 'outputs':outputs, 'review_image':str((destination/'three-plane-review.png').relative_to(root)),
              'elapsed_seconds':time.monotonic()-started, 'accepted':False,
              'brain_semantics':'Estimated intracranial envelope within closed-cavity support; may include CSF; never a measured clinical brain segmentation',
              'artificial_boundaries':{'brain_inferior_plane':'Artificial enclosure-support cutoff; never a pial or approach surface', 'skull_context_cutoff':'12mm cavity-distance cutoff; never a measured skull edge', 'FOV_faces':'Missing anatomy; entry and tool sweep through crop faces unsupported'},
              'entry_scope':'Full entry and tool sweep require independent observed-calvarium and acquired-coverage review; unsupported boundary bypass forbidden',
              'limitations':['Engineering estimated intracranial envelope may include CSF; not measured brain segmentation', 'Tumor placement support is a conservative tissue/intensity/clearance proxy, not tissue diagnosis', 'Bone closing supports enclosure only; skull output uses original measured intensities', 'Open/cropped slices remain unknown; whole-head completeness not claimed', 'Dense skull may contain calcification/artifact; supplied vascular background is unknown', 'Inferior cavity, cortex, crop boundaries and skull topology require independent review']}
    checked_root(root, expected_uuid=UUID)
    atomic_json(destination/'provenance.json',result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--subject',required=True);parser.add_argument('--version',default='enclosure-v1')
    args=parser.parse_args()
    if not args.subject.isdigit() or len(args.subject)!=3: parser.error('Three digit source subject required')
    print(json.dumps(prepare(Path(args.root),args.subject,version=args.version),indent=2))
