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
import shutil
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


def additive_brain_proposal(ct, vessels, spacing, old_brain, old_skull, model_brain):
    """Add coarse brain protection; never delete or reinterpret old dense bone."""
    arrays = [np.asarray(a) for a in (ct, vessels, old_brain, old_skull, model_brain)]
    if (arrays[0].ndim != 3 or any(a.shape != arrays[0].shape for a in arrays)
            or any(not np.isfinite(a).all() for a in arrays)):
        raise ValueError('Finite, identical native source shapes required')
    for mask in arrays[2:]:
        if not np.isin(mask, [0, 1]).all():
            raise ValueError('Binary old/model estimates required')
    spacing = np.asarray(spacing, float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Finite positive native spacing required')
    brain = arrays[2].astype(bool) | arrays[4].astype(bool)
    vessel_distance = ndi.distance_transform_edt(~(arrays[1] > 0), sampling=spacing)
    support = (brain & (arrays[0] >= 10) & (arrays[0] <= 90)
               & (vessel_distance > 3.)
               & (ndi.distance_transform_edt(brain, sampling=spacing) >= 4.))
    return {'brain_estimate': brain, 'observed_skull': arrays[3].astype(bool).copy(),
            'tumor_placement_support': support}


def prepare_additive(root, subject, *, version, model_pilot, prior_version='enclosure-v1'):
    """Unaccepted native-grid union proposal, with byte-identical prior skull.

    The learned skull is provenance/display context only. Original enclosure
    coverage remains a distinct, unchanged superior support observation; adding
    a brain estimate does not enlarge an entry/task domain or authorize actions.
    """
    for token in (version, prior_version):
        if not isinstance(token, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', token):
            raise ValueError('Use a plain safe immutable version token')
    if subject not in {'004', '006', '012', '014', '018'}:
        raise ValueError('Only the five pinned TRAIN source people are permitted')
    root = checked_root(root, expected_uuid=UUID, required_bytes=1024**3)
    destination = safe_path(root, f'integrated-v1/anatomy/TopCoW-{subject}/{version}')
    if destination.exists():
        raise FileExistsError(f'Immutable output exists: {destination}')
    import nibabel as nib
    started = time.monotonic()
    prior = safe_path(root, f'integrated-v1/anatomy/TopCoW-{subject}/{prior_version}')
    pilot = safe_path(root, model_pilot)
    receipt_path = safe_path(root, f'catalog/topbrain/{subject}-selective-extraction.json')
    receipt = json.loads(receipt_path.read_text())
    if receipt['family']['family_id'] != f'TopCoW:{subject}' or receipt['family']['role'] != 'TRAIN':
        raise ValueError('Pinned TRAIN source identity required')
    entries = [next(e for e in receipt['files'] if text in e['source_member'])
               for text in ('/imagesTr_topbrain/topcow_ct_', '/labelsTr_topbrain_v1_ct/')]
    source_images = []
    for entry in entries:
        path = safe_path(root, entry['path'])
        if sha256_file(path) != entry['sha256']:
            raise ValueError('Acquired source hash mismatch')
        source_images.append(nib.load(path))
    image, annotation = source_images
    affine = image.affine
    if (image.header.get_xyzt_units()[0] != 'mm' or annotation.shape != image.shape
            or not np.array_equal(annotation.affine, affine)
            or not np.allclose(affine[:3, :3], np.diag(np.diag(affine[:3, :3])), atol=1e-5)):
        raise ValueError('Exact orthogonal native image/annotation grid required; no resampling')
    prior_report_path = prior/'provenance.json'
    prior_report = json.loads(prior_report_path.read_text())
    if (prior_report['family_id'] != f'TopCoW:{subject}'
            or [e['sha256'] for e in prior_report['sources']] != [e['sha256'] for e in entries]):
        raise ValueError('Prior proposal must bind the same acquired sources')
    method_path, result_path = pilot/'prospective-method.json', pilot/'result.json'
    method, result = json.loads(method_path.read_text()), json.loads(result_path.read_text())
    if (method['subject'] != subject or method['role'] != 'TRAIN'
            or method['source']['sha256'] != entries[0]['sha256']
            or method['source']['path'] != entries[0]['path']
            or result['status'] != 'succeeded_pending_independent_review'
            or not result.get('versions') or not method.get('weights')):
        raise ValueError('Successful cached model must bind this exact TRAIN source')
    for weight in method['weights']:
        if sha256_file(safe_path(root, weight['path'])) != weight['sha256']:
            raise ValueError('Cached model weight hash mismatch')
    inputs = {}

    def read_mask(name, path, digest):
        if sha256_file(path) != digest:
            raise ValueError('Estimate input hash mismatch: '+name)
        obj = nib.load(path)
        if obj.shape != image.shape or not np.array_equal(obj.affine, affine):
            raise ValueError('Exact estimate native grid required: '+name)
        data = np.asanyarray(obj.dataobj)
        if not np.isfinite(data).all() or not np.isin(data, [0, 1]).all():
            raise ValueError('Finite binary estimate required: '+name)
        inputs[name] = {'path':str(path.relative_to(root)), 'sha256':digest}
        return data.astype(bool)

    old = {}
    for name in ('brain_estimate', 'observed_skull', 'enclosed_coverage', 'tumor_placement_support'):
        row = prior_report['outputs'][name]
        path = safe_path(root, row['path'])
        if path != prior/(name+'.nii.gz'):
            raise ValueError('Prior output path does not belong to the bound proposal')
        old[name] = read_mask('prior_'+name, path, row['sha256'])
    models = {}
    for name in ('brain', 'skull'):
        path = pilot/(name+'_mask.nii.gz')
        rows = [row for row in result['outputs'] if row['path'] == str(path.relative_to(root))]
        if len(rows) != 1:
            raise ValueError('Model result must bind its exact exported masks')
        models[name] = read_mask('model_'+name, path, rows[0]['sha256'])
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    ct, vessels = np.asarray(image.dataobj, np.float32), np.asanyarray(annotation.dataobj)
    proposals = additive_brain_proposal(ct, vessels, spacing, old['brain_estimate'],
                                        old['observed_skull'], models['brain'])
    proposals['enclosed_coverage'] = old['enclosed_coverage']
    checked_root(root, expected_uuid=UUID, required_bytes=ct.nbytes)
    destination.mkdir(parents=True)
    outputs = {}
    for name, data in proposals.items():
        path = destination/(name+'.nii.gz')
        if name in ('observed_skull', 'enclosed_coverage'):
            shutil.copyfile(prior/(name+'.nii.gz'), path)
            if sha256_file(path) != prior_report['outputs'][name]['sha256']:
                raise ValueError('Unchanged prior mask bytes were not preserved')
        else:
            header = image.header.copy(); header.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(data.astype(np.uint8), affine, header), path)
        saved = nib.load(path)
        if (saved.shape != image.shape or not np.array_equal(saved.affine, affine)
                or not np.array_equal(np.asanyarray(saved.dataobj), data)):
            raise ValueError('Output native grid/array changed')
        outputs[name] = {'path':str(path.relative_to(root)), 'sha256':sha256_file(path),
                         **metrics(data, affine, spacing)}
    report = {'schema':'ressectionlab-additive-model-brain-proposal-v1',
              'family_id':f'TopCoW:{subject}', 'role':'TRAIN', 'preparation_version':version,
              'sources':entries, 'receipt_sha256':sha256_file(receipt_path),
              'prior_proposal':{'path':str(prior_report_path.relative_to(root)), 'sha256':sha256_file(prior_report_path)},
              'model_method':{'path':str(method_path.relative_to(root)), 'sha256':sha256_file(method_path)},
              'model_result':{'path':str(result_path.relative_to(root)), 'sha256':sha256_file(result_path)},
              'model_versions':result['versions'], 'model_weights':method['weights'], 'inputs':inputs,
              'method':'brain=old_brain OR model_brain; skull and enclosed_coverage copied byteexact from prior; no modelskull mechanics',
              'support_method':'union_brain AND originalCT10..90 AND supplied_vessel_distance>3mm AND union_brain_EDT>=4mm, exactly prior support thresholds',
              'outputs':outputs, 'shape':list(image.shape), 'affine_ras':affine.tolist(),
              'physical_bounds':physical_bounds(image.shape, affine), 'spacing_mm':spacing.tolist(),
              'old_brain_retained_all':bool(np.all(proposals['brain_estimate'][old['brain_estimate']])),
              'model_skull_mechanical':False, 'accepted':False, 'review_status':'pending_independent_proposal_review',
              'task_domain_changed':False, 'native_exported':False, 'actions_executed':0,
              'brain_semantics':'Coarse estimated observed brain/intracranial envelope including possible CSF; not clinical parenchyma or pial boundary',
              'coverage_semantics':'enclosed_coverage remains the original superior-only observation; it is not broadened to match the union brain and does not authorize new entry',
              'limitations':['Model exclusions remain unknown, never newly free space; old brain protection retained',
                             'Prior dense skull retained byteexact including all old obstacles; model skull is uncertainty/display only',
                             'Crop faces and original reviewed task box unchanged; no approach or action approval',
                             'HU-like CT, original DICOM calibration unverified; fast3mm model restoration is not native-resolution evidence',
                             'Supplied vessels partial, no vessel-free truth; target unchanged only in existing native case'],
              'script_sha256':sha256_file(__file__), 'elapsed_seconds':time.monotonic()-started}
    (destination/'data_integrated_anatomy.py.snapshot').write_bytes(Path(__file__).read_bytes())
    checked_root(root, expected_uuid=UUID)
    atomic_json(destination/'provenance.json', report)
    return report


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
    parser.add_argument('--mode',choices=['enclosure','additive-model-brain'],default='enclosure')
    parser.add_argument('--model-pilot',help='Existing successful model pilot directory relative to DATA')
    parser.add_argument('--prior-version',default='enclosure-v1')
    args=parser.parse_args()
    if not args.subject.isdigit() or len(args.subject)!=3: parser.error('Three digit source subject required')
    if args.mode == 'additive-model-brain':
        if not args.model_pilot: parser.error('--model-pilot required for additive-model-brain')
        result=prepare_additive(Path(args.root),args.subject,version=args.version,
                                model_pilot=args.model_pilot,prior_version=args.prior_version)
    else:
        if args.model_pilot: parser.error('--model-pilot only belongs to additive-model-brain')
        result=prepare(Path(args.root),args.subject,version=args.version)
    print(json.dumps(result,indent=2))
