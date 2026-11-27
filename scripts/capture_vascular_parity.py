#!/usr/bin/env python3
"""Capture and measure the four vascular hero cases at fixed cameras.

Every phase of the vascular rendering work reruns this harness so that
captures, frame timing, geometry and reference agreement stay comparable.
Evidence is written to the verified external data root, never to the checkout.
Source scans and labels are only read. Like the desktop app, the wall reader
keeps its derived contour cache in each study's meshes folder.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess

import nibabel as nib
import numpy as np
from PIL import Image

from resectionlab.anatomy_bridge import open_anatomy_manifest
from resectionlab.anatomy_geometry import mesh, vascular_cleanup_source
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import affine, digest, world
from resectionlab.vascular_cleanup import rooted_vessel_layer
from resectionlab.vascular_parity import (
    agreement, betti, floating_components, image_sharpness, source_deviation, surface_quality,
)
from resectionlab.vascular_rooting import MAJOR_VESSELS, SCHEMES, image_modality

HDD_ROOT = Path('/Volumes/shafrir/old coding projects/RessectionLabData')
HDD_UUID = '9CC3623C-E38B-3EEE-B6BB-B9A06C193700'
EVIDENCE = Path('qc/medivis-parity-20261125')
TOPBRAIN = 'sources/topbrain/selective/{case}/TopBrain_Data_Release_Batches1n2nTA36_081726/labelsTr_topbrain_v1_ct/topcow_ct_{case}.nii.gz'
# TopBrain v1 CT values: 1-34 arteries, 35-40 veins and sinuses. Roots are the
# basilar (1), internal carotids (4, 6), vertebrals (23, 24), the vein of Galen
# (35) and the straight and superior sagittal sinuses (36, 40).
ROOT_LABELS = SCHEMES['topbrain-v1-ct'].roots
# Phase 2 ensembles: <PHASE2>/<phase2 case>/<ensemble folder>/ (the folder is a parameter).
PHASE2 = EVIDENCE / 'phase2-segmentation' / 'cases'

HERO = {
    'CTA-A': {
        'title': 'TopCoW-014 CTA with skull', 'held_out': False,
        'manifest': 'prepared/cta-native-showcases-v1/TopCoW-014-CTA-ORGANIC-01-v1-NATIVE-SHOWCASE/patient-anatomy.json',
        'series': 'ct', 'vessels': ['supplied_arterial_annotation_union', 'supplied_venous_sinus_annotation_union'],
        'anchor': [4.07, 210.81, -67.40], 'reference': TOPBRAIN.format(case='014'), 'phase2': 'TopCoW-014-ct',
    },
    'CTA-B': {
        'title': 'TopCoW-018 CTA, held out', 'held_out': True,
        'manifest': 'prepared/annotated-cta-vascular-comparison-v1/TopCoW-018-CTA-VESSEL-COMPARISON-v1/patient-anatomy.json',
        'series': 'cta', 'vessels': ['publisher-arteries', 'publisher-veins'],
        'anchor': [-5.07, 208.28, 78.50], 'reference': TOPBRAIN.format(case='018'),
        'model_probability': 'cache/poc-model-results/TopCoW-018-annotated-cta-benchmark-v1/vesselfm-base',
        'phase2': 'TopCoW-018-ct',
    },
    'TOF-A': {
        'title': 'EDEN PTNT_4509 TOF', 'held_out': False,
        'manifest': 'qc/eden2020-glioma-5714812/vascular-reconstruction-20261123-v1/studies/PTNT_4509/patient-anatomy.json',
        'series': 'tof', 'vessels': ['vesselfm-cleanup'], 'reference': None, 'cohort_view': True,
        'anchor': [3.2779, 5.3626, -13.3764], 'phase2': 'PTNT_4509-mr',
    },
    'TOF-B': {
        'title': 'EDEN PTNT_9937 TOF, held out', 'held_out': True,
        'manifest': 'prepared/Eden 2020 ResectionLab Projects (restricted)/PTNT_9937/vascular-review.anatomy.json',
        'series': 'tof', 'vessels': ['vesselfm-cleanup'], 'reference': None, 'cohort_view': True,
        'anchor': [1.3713, 6.0276, -22.1647], 'phase2': 'PTNT_9937-mr',
    },
    # Two further EDEN cases, chosen on 2026-11-26 from the cohort's earlier captures as the cleanest trees.
    # Anchors: the internal carotid centroid of the named labels, shifted by (-0.2, -3.4, -3.6) mm.
    'TOF-C': {
        'title': 'EDEN PTNT_4503 TOF, held out', 'held_out': True,
        'manifest': 'prepared/Eden 2020 ResectionLab Projects (restricted)/PTNT_4503/vascular-review.anatomy.json',
        'series': 'tof', 'vessels': ['vesselfm-cleanup'], 'reference': None, 'cohort_view': True,
        'anchor': [3.14, 23.77, -12.94], 'phase2': 'PTNT_4503-mr',
    },
    'TOF-D': {
        'title': 'EDEN PTNT_7086 TOF, held out', 'held_out': True,
        'manifest': 'prepared/Eden 2020 ResectionLab Projects (restricted)/PTNT_7086/vascular-review.anatomy.json',
        'series': 'tof', 'vessels': ['vesselfm-cleanup'], 'reference': None, 'cohort_view': True,
        'anchor': [-1.52, 21.22, -22.08], 'phase2': 'PTNT_7086-mr',
    },
}
# View direction is from the camera toward the anchor, in patient RAS.
PRESETS = {
    'anterior': {'from': [0, 1, 0.12], 'up': [0, 0, 1], 'distance': 1.75},
    'left-lateral': {'from': [-1, 0.05, 0.1], 'up': [0, 0, 1], 'distance': 1.75},
    'superior': {'from': [0, -0.02, 1], 'up': [0, 1, 0], 'distance': 1.75},
    'oblique': {'from': [-0.62, 0.62, 0.48], 'up': [0, 0, 1], 'distance': 1.75},
    'skull-base': {'from': [0, -0.25, 1], 'up': [0, 1, 0], 'distance': 1.05, 'clip': ('z', 22.0, True)},
    'axial-clip': {'from': [-0.35, 0.75, 0.56], 'up': [0, 0, 1], 'distance': 1.55, 'clip': ('z', 6.0, True)},
    # Variants framed like the reference stills; the six above stay the fixed baseline.
    'skull-base-wide': {'from': [0, -0.25, 1], 'up': [0, 1, 0], 'distance': 1.5, 'clip': ('z', 22.0, True)},
    'axial-clip-wide': {'from': [0, 0.85, 0.53], 'up': [0, 0, 1], 'distance': 1.6, 'clip': ('z', 6.0, True)},
    'anterior-coronal-cut': {'from': [0, 1, 0.12], 'up': [0, 0, 1], 'distance': 1.75, 'clip': ('y', 30.0, True)},
    # Bone framings of the reference stills: an opaque skull, and a sagittal cut. They need
    # acquired bone and the path-traced mode, whose bone is solid under an opaque skull or a cut.
    'left-lateral-solid': {'from': [-1, 0.05, 0.1], 'up': [0, 0, 1], 'distance': 2.1, 'skull': 'exterior', 'bone': True},
    'oblique-solid': {'from': [-0.62, 0.62, 0.48], 'up': [0, 0, 1], 'distance': 2.1, 'skull': 'exterior', 'bone': True},
    'left-lateral-sagittal-cut': {'from': [-1, 0.05, 0.1], 'up': [0, 0, 1], 'distance': 1.9, 'clip': ('x', -8.0, False), 'bone': True},
}
REFERENCE_FRAME = {'skull-base-wide': 'skull-base', 'axial-clip-wide': 'axial-clip', 'anterior-coronal-cut': 'anterior',
                   'left-lateral-solid': 'left-lateral', 'oblique-solid': 'oblique', 'left-lateral-sagittal-cut': 'left-lateral'}
INTERACTION_PRESET = 'anterior'


def evidence(root, label):
    if not re.fullmatch(r'phase\d-[a-z0-9-]{1,48}', label):
        raise ValueError('Evidence labels look like phase0-baseline')
    return safe_path(root, EVIDENCE / label)


def scan_geometry(collection, case):
    scan = collection.series(case['series'])
    frame = affine(scan['affine'])
    shape = np.asarray(scan['data']['shape'])
    corners = world(frame, np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)]))
    return scan, frame, float(np.ptp(corners, axis=0).max())


def vessel_mask(collection, case):
    mask = None
    for layer_id in case['vessels']:
        layer = collection.layer(layer_id)
        current = collection.mask(layer)
        mask = current if mask is None else mask | current
    return mask


# The last two are display-only candidates (topbrain_native.py display-additions): every vesselFM voxel at 0.5
# outside the tree, and on CT the thin branches followed outward in the acquired image.
ENSEMBLE_FILES = ('named_labels', 'vessel_probability', 'vesselfm_connected_probability',
                  'vesselfm_candidate_probability', 'fine_branch_score')


def _ensemble_image(path, series):
    image = nib.load(path)
    if image.shape != tuple(series['data']['shape']) or not np.allclose(image.affine, series['affine'], atol=1e-3):
        raise ValueError(f'{path.name} is not on the {series["id"]} scan grid')
    return np.asarray(image.dataobj)


WIRED_MODELS = {'topbrain-2025-ensemble', 'vesselfm-connected-addition', 'vesselfm-candidate-addition', 'signal-grown-fine-branches'}
VESSEL_KINDS = {'vessel', 'vessels', 'artery', 'vein', 'sinus'}


def _wired(row):
    return (row.get('provenance') or {}).get('model') in WIRED_MODELS and 'phase2_binding' in row.get('provenance', {})


def _array_paths(value, found=None):
    """Every stored array path a document (or part of one) refers to."""
    found = set() if found is None else found
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and value['path'].startswith('arrays/'):
            found.add(value['path'])
        for item in value.values():
            _array_paths(item, found)
    elif isinstance(value, list):
        for item in value:
            _array_paths(item, found)
    return found


def _data_root_relative(folder):
    """The folder relative to the data root that holds it (marked by .ressectionlab-data-root.json)."""
    folder = Path(folder).resolve()
    for parent in folder.parents:
        if (parent / '.ressectionlab-data-root.json').is_file():
            return str(folder.relative_to(parent))
    raise ValueError(f'{folder} is not inside a marked data root; pass source= relative to it')


def _linked(collection):
    return (collection.document.get('vascular_review') or {}).get('linked_source') is True


def _publish(collection, manifest, event):
    """Save a native study, or write a source-linked review document beside its pinned source."""
    if not _linked(collection):
        return collection.save(event)
    from datetime import datetime, timezone
    from resectionlab.patient_anatomy import PatientImagingCollection
    document = {key: value for key, value in collection.document.items() if key != 'snapshot_id'}
    document['revision'] = document.get('revision', 0) + 1
    document['history'] = [*document.get('history', []), {'event': event, 'author': 'local researcher',
                                                           'at': datetime.now(timezone.utc).isoformat()}]
    document['snapshot_id'] = PatientImagingCollection(collection.root, dict(document)).identity()
    PatientImagingCollection(collection.root, document)  # Validate before replacing the file.
    atomic_json(manifest, document)
    collection.document.clear()
    collection.document.update(document)
    return document['snapshot_id']


def _guard_save_side_effects(collection):
    """Refuse when saving would change anything beyond the wiring's declared edits."""
    from copy import deepcopy
    from resectionlab.anatomy_display import retire_stale_cta_coverage
    from resectionlab.patient_anatomy import PatientImagingCollection
    if _linked(collection):
        return
    document = {key: value for key, value in deepcopy(collection.document).items() if key != 'snapshot_id'}
    shadow = PatientImagingCollection(collection.root, deepcopy(document))
    retire_stale_cta_coverage(shadow)
    if {key: value for key, value in shadow.document.items() if key != 'snapshot_id'} != document:
        raise ValueError('Saving this study would also retire stale CTA coverage; resolve that before wiring')


def wire_ensemble(collection, series_id, folder, *, manifest=None, source=None):
    """Make a Phase 2 ensemble the study's displayed vessels, as model-estimate layers.

    Adds the named tree (``named_labels``, scheme, soft probability) and the
    connected vesselFM score as a separate child layer (``rooted_parent``) that
    is shown only where it reaches the tree's shown display. Both become the
    study's initial vessel layers in place of the previous initial vessel
    layers; those (for example supplied annotations) stay in the study,
    unchanged and selectable, and non-vessel initial layers are untouched. A
    re-wire retires the previous generation of these wired layers (their entries
    and arrays, reproducible from the ensemble folder) so two generations are
    never drawn together. Nothing else changes: the save is all or nothing, and
    it refuses if saving would alter anything else. Returns a receipt.
    """
    from copy import deepcopy
    folder = Path(folder)
    manifest = Path(manifest) if manifest else collection.root / 'patient-anatomy.json'
    meta = json.loads((folder / 'ensemble.json').read_text())
    modality = meta.get('modality')
    series = collection.series(series_id)
    if modality not in ('ct', 'mr') or image_modality(series['modality']) != modality:
        raise ValueError(f'A {modality} ensemble cannot be wired to a {series["modality"]} scan')
    scheme = f'topbrain-v1-{modality}'
    files = {name: folder / f'{name}.nii.gz' for name in ENSEMBLE_FILES if (folder / f'{name}.nii.gz').is_file()}
    if not {'named_labels', 'vessel_probability'} <= set(files):
        raise ValueError(f'{folder} lacks named labels or the vessel probability')
    hashes = {name: sha256_file(path) for name, path in files.items()}
    binding = digest({'ensemble': meta, 'files': hashes})[:12]
    ids = [f'ensemble-{binding}', f'ensemble-{binding}-vesselfm', f'ensemble-{binding}-fine']
    source = source or _data_root_relative(folder)
    before_sha, before_snapshot = sha256_file(manifest), collection.identity()
    display = collection.document.setdefault('display', {})
    initial_before = list(display.get('initial_layers', []))
    rows = {row['id']: row for row in collection.document['layers']}
    current = [row['id'] for row in collection.document['layers'] if _wired(row) and row['provenance']['phase2_binding'] == binding]
    retired = [row for row in collection.document['layers'] if _wired(row) and row['provenance']['phase2_binding'] != binding]
    wiring = display.get('ensemble_wiring') or {}
    replaced = wiring.get('replacedInitialLayers',
                          [layer for layer in initial_before if rows.get(layer, {}).get('kind') in VESSEL_KINDS and not _wired(rows[layer])])
    arrays, revisions = collection.root / 'arrays', collection.root / 'revisions'
    existing = {path for folder_ in (arrays, revisions) if folder_.is_dir() for path in folder_.iterdir()}
    previous_document = deepcopy(collection.document)
    manifest_bytes = manifest.read_bytes()
    try:
        _guard_save_side_effects(collection)
        added = []
        if not current:
            named = _ensemble_image(files['named_labels'], series)
            unknown = set(np.unique(named).tolist()) - {0} - set(SCHEMES[scheme].names)
            if not np.issubdtype(named.dtype, np.integer) or unknown:
                raise ValueError(f'Named labels outside {scheme}: {sorted(unknown)}')
            if not named.any():
                raise ValueError(f'{folder} has an empty named vessel map')
            named = named.astype(np.uint8)
            probability = np.clip(_ensemble_image(files['vessel_probability'], series).astype(np.float32), 0, 1)
            provenance = {'kind': 'image_derived_estimate', 'model': 'topbrain-2025-ensemble', 'estimate': 'model',
                          'evidence_use': 'display_only', 'teams': meta.get('teams', []), 'scheme': scheme,
                          'ensemble': folder.name, 'source': source, 'files_sha256': hashes,
                          'ensemble_sha256': sha256_file(folder / 'ensemble.json'), 'phase2_binding': binding}
            teams = '+'.join(meta.get('teams', [])) or 'model'
            row = collection.add_layer((named > 0).astype(np.uint8), series['affine'], layer_id=ids[0], source_series=series_id,
                                       kind='vessels', role='context', color='#aa352b',
                                       label=f'Model vessels · TopBrain {teams} ensemble', probability=probability,
                                       provenance=provenance)
            row['named_labels'] = {'data': collection.put_array(named), 'scheme': scheme}
            added.append(ids[0])
            # The candidates (connected or not) replace the scored connected addition as the drawn layer when present.
            candidate = 'vesselfm_candidate_probability' in files
            extra_file = files.get('vesselfm_candidate_probability') or files.get('vesselfm_connected_probability')
            parent = ids[0]
            if extra_file:
                score = np.clip(_ensemble_image(extra_file, series).astype(np.float32), 0, 1)
                score[named > 0] = 0
                if score.any():
                    extra = collection.add_layer(
                        (score > 0).astype(np.uint8), series['affine'], layer_id=ids[1], source_series=series_id,
                        kind='vessels', role='context', color='#aa352b',
                        label=('Model vessels · vesselFM candidates (unnamed)' if candidate
                               else 'Model vessels · connected vesselFM addition (unnamed)'), probability=score,
                        provenance={**provenance, 'model': 'vesselfm-candidate-addition' if candidate else 'vesselfm-connected-addition',
                                    'addition': {'connected': False, 'rule': 'topbrain_native.DISPLAY_CANDIDATES'} if candidate
                                    else meta.get('addition')})
                    extra['rooted_parent'] = parent
                    added.append(ids[1])
                    parent = ids[1]
            if 'fine_branch_score' in files:
                fine = np.clip(_ensemble_image(files['fine_branch_score'], series).astype(np.float32), 0, 1)
                taken = named > 0
                if parent == ids[1]:
                    taken = taken | (score > 0)
                fine[taken] = 0
                if fine.any():
                    branch = collection.add_layer(
                        (fine > 0).astype(np.uint8), series['affine'], layer_id=ids[2], source_series=series_id,
                        kind='vessels', role='context', color='#aa352b',
                        label='Image-derived fine branches (unnamed)', probability=fine,
                        provenance={**provenance, 'model': 'signal-grown-fine-branches', 'estimate': 'image-derived tube response',
                                    'score_meaning': 'uncalibrated tube response, not a probability'})
                    branch['rooted_parent'] = parent
                    added.append(ids[2])
            current = added
        if retired:
            collection.document['layers'] = [row for row in collection.document['layers'] if row not in retired]
        others = [layer for layer in initial_before if layer not in replaced and not (layer in rows and _wired(rows[layer]))]
        initial_after = others + current
        changed = bool(added or retired or initial_after != initial_before or not wiring)
        if changed:
            display['initial_layers'] = initial_after
            display['ensemble_wiring'] = {'binding': binding, 'ensemble': folder.name, 'replacedInitialLayers': replaced}
            _publish(collection, manifest, f'wire Phase 2 ensemble {folder.name} as the displayed model vessels')
            from resectionlab.anatomy_bridge import open_anatomy_manifest
            reopened = open_anatomy_manifest(manifest)
            if reopened.identity() != collection.identity():
                raise ValueError('The saved study does not reopen as written')
    except BaseException:
        # All or nothing: restore the manifest and remove the arrays this call wrote.
        if manifest.read_bytes() != manifest_bytes:
            manifest.write_bytes(manifest_bytes)
        collection.document.clear()
        collection.document.update(previous_document)
        for path in {path for folder_ in (arrays, revisions) if folder_.is_dir() for path in folder_.iterdir()} - existing:
            path.unlink(missing_ok=True)
        raise
    removed = []
    if retired:
        keep = _array_paths(collection.document)
        for path in sorted(_array_paths(retired) - keep):
            collection.path(path).unlink(missing_ok=True)
            removed.append(path)
    return {'case_manifest': manifest.name, 'ensemble': folder.name, 'source': source, 'binding': binding,
            'layers': current, 'added': added, 'retired': [row['id'] for row in retired], 'removed_arrays': removed,
            'changed': changed, 'manifest_sha256': {'before': before_sha, 'after': sha256_file(manifest)},
            'snapshot': {'before': before_snapshot, 'after': collection.identity()},
            'initial_layers': {'before': initial_before, 'after': display.get('initial_layers', initial_before)},
            'replaced_initial_layers': replaced,
            'undo': f'PYTHONPATH=src .venv/bin/python scripts/capture_vascular_parity.py unwire --case <case>'}


def unwire(collection, *, manifest=None):
    """Undo the wiring: remove every wired generation and restore the replaced initial vessel layers."""
    manifest = Path(manifest) if manifest else collection.root / 'patient-anatomy.json'
    display = collection.document.setdefault('display', {})
    wiring = display.get('ensemble_wiring')
    wired = [row for row in collection.document['layers'] if _wired(row)]
    if not wiring and not wired:
        return {'changed': False}
    _guard_save_side_effects(collection)
    ids = {row['id'] for row in wired}
    collection.document['layers'] = [row for row in collection.document['layers'] if row['id'] not in ids]
    replaced = (wiring or {}).get('replacedInitialLayers', [])
    display['initial_layers'] = [layer for layer in display.get('initial_layers', []) if layer not in ids] + \
        [layer for layer in replaced if layer not in display.get('initial_layers', [])]
    display.pop('ensemble_wiring', None)
    _publish(collection, manifest, 'unwire Phase 2 ensemble layers')
    keep = _array_paths(collection.document)
    removed = sorted(_array_paths(wired) - keep)
    for path in removed:
        collection.path(path).unlink(missing_ok=True)
    return {'changed': True, 'removed_layers': sorted(ids), 'removed_arrays': removed, 'initial_layers': display['initial_layers']}


def ensemble_layers(collection, folder):
    """Layer ids wired from an ensemble folder, refusing when the folder no longer matches them."""
    folder = Path(folder)
    rows = [row for row in collection.document['layers'] if _wired(row) and row['provenance'].get('ensemble') == folder.name]
    command = f'capture_vascular_parity.py wire --ensemble {folder.name}'
    if not rows:
        raise LookupError(f'No layers wired from {folder.name}; run {command}')
    for row in rows:
        recorded = row['provenance'].get('files_sha256', {})
        present = {name: sha256_file(folder / f'{name}.nii.gz') for name in recorded if (folder / f'{name}.nii.gz').is_file()}
        if present != recorded or sha256_file(folder / 'ensemble.json') != row['provenance'].get('ensemble_sha256'):
            raise LookupError(f'{folder.name} changed since it was wired into this study; run {command}')
    return [row['id'] for row in rows]


def displayed_vessels(collection, case, ensemble=None, root=None):
    if not ensemble:
        return list(case['vessels'])
    folder = safe_path(root, PHASE2 / case['phase2'] / ensemble) if root else Path(ensemble)
    ids = ensemble_layers(collection, folder)
    if not set(ids) <= set(collection.document.get('display', {}).get('initial_layers', [])):
        raise LookupError(f'The wired layers are not the study default; re-run wire --ensemble {Path(ensemble).name}')
    return ids


def rooted_display(collection, layer_ids, *, check=lambda: None):
    """The default view's vessels: the rooted union of the layers, and each layer's receipt."""
    union, receipts = None, {}
    for layer_id in layer_ids:
        display, receipt, _, split = rooted_vessel_layer(collection, collection.layer(layer_id), check=check)
        shown = collection.mask(display)
        if split is not None:
            shown |= collection.mask(split)
        union = shown if union is None else union | shown
        receipts[layer_id] = receipt
    return union, receipts


def _inside(mask, points):
    """Vertices inside a voxel mask: every voxel whose cell the vertex touches is set.

    A vertex off a cell boundary rounds to its nearest voxel; one on a boundary
    between voxels (a half-integer coordinate) needs all of them, so a surface
    that only abuts the mask does not count as inside it."""
    lo, hi = np.floor(points + 0.499).astype(int), np.floor(points + 0.501).astype(int)
    inside = np.ones(len(points), bool)
    for corner in range(8):
        cells = np.where([(corner >> axis) & 1 for axis in range(3)], hi, lo)
        valid = ((cells >= 0) & (cells < mask.shape)).all(1)
        inside &= valid
        inside[valid] &= mask[tuple(cells[valid].T)]
    return inside


def drawn_solid(collection, row):
    """What a part's surface encloses: its displayed voxels, or with thin-branch tubes the field it was contoured from."""
    from resectionlab.vessel_display_tubes import DISPLAY_TUBES, surface_field, unpack
    mask = collection.mask(row)
    if row.get(DISPLAY_TUBES) is None:
        return mask
    field, replaced = unpack(collection.array(row[DISPLAY_TUBES]))
    return np.maximum((mask & ~replaced).astype(np.float32), surface_field(field))


def mesh_floating(surfaces, roots, shape, masks=None):
    """Floating pieces of the drawn vessel surface, counted on the meshes the desktop receives.

    ``surfaces`` are (name, mesh) pairs. Pieces of different parts that share a
    vertex position (touching surfaces) are one piece; so are pieces of
    different parts where a vertex of one lies inside the other (``masks``,
    one per part, or callables loading it), joined to that part's nearest
    vertex. A boolean mask means inside the displayed voxels; a float array is
    the field the part's surface was contoured from, and inside means at or
    above one half there (a part whose thin branches are drawn as tubes).
    A piece floats when no vertex lies within one voxel of a root-label voxel.
    Independent of the rooting code, so it can come out non-zero.
    """
    from scipy import ndimage as ndi
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    points, faces, owner, offset = [], [], [], 0
    for part, (_, surface) in enumerate(surfaces):
        raw = np.asarray(surface['source'], float).reshape(-1, 3)
        index = np.asarray(surface['indices'], np.int64).reshape(-1, 3)
        points.append(raw)
        faces.append(index + offset)
        owner.append(np.full(len(raw), part))
        offset += len(raw)
    if not offset:
        return {'pieces': 0, 'floating': 0, 'parts': len(surfaces)}
    points, faces, owner = np.concatenate(points), np.concatenate(faces), np.concatenate(owner)
    # Within one part only shared triangles join; separate parts join where their surfaces touch.
    keys, shared = np.unique(np.round(points, 3), axis=0, return_inverse=True)
    shared = shared.ravel()
    parts_at = np.zeros(len(keys), int)
    for part in range(len(surfaces)):
        parts_at[np.unique(shared[owner == part])] += 1
    touching = parts_at[shared] > 1
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    nodes = len(points) + len(keys)
    links = [edges, np.column_stack([np.flatnonzero(touching), len(points) + shared[touching]])]
    for part, mask in enumerate(masks or ()):
        own = np.flatnonzero(owner == part)
        others = np.flatnonzero(owner != part)
        if mask is None or not len(own) or not len(others):
            continue
        solid = np.asarray(mask() if callable(mask) else mask)
        if solid.dtype.kind == 'f':
            inside = others[ndi.map_coordinates(solid, points[others].T, order=1, mode='constant') >= .5]
        else:
            inside = others[_inside(solid.astype(bool), points[others])]
        del solid
        if len(inside):
            links.append(np.column_stack([inside, own[cKDTree(points[own]).query(points[inside])[1]]]))
    links = np.concatenate(links)
    graph = coo_matrix((np.ones(len(links)), (links[:, 0], links[:, 1])), shape=(nodes, nodes))
    _, component = connected_components(graph, directed=False)
    used = np.unique(faces)
    pieces = np.unique(component[used])
    # Inner walls of hollow parts are closed pieces wound the other way round;
    # they are reported, not counted as floating vessels.
    a, b, c = (points[faces[:, corner]] for corner in range(3))
    signed = np.bincount(component[faces[:, 0]], weights=np.einsum('ij,ij->i', a, np.cross(b, c)), minlength=nodes)
    outward = np.sign(signed[pieces[np.argmax(np.abs(signed[pieces]))]])
    cavities = {int(piece) for piece in pieces if np.sign(signed[piece]) == -outward}
    near = ndi.binary_dilation(roots, np.ones((3, 3, 3), bool)) if roots is not None else None
    anchored = set()
    if near is not None:
        cells = np.clip(np.rint(points[used]).astype(int), 0, np.array(shape) - 1)
        hit = near[tuple(cells.T)]
        anchored = set(np.unique(component[used][hit]).tolist())
    outer = [int(piece) for piece in pieces if int(piece) not in cavities]
    floating = [piece for piece in outer if piece not in anchored] if near is not None else outer[1:]
    return {'pieces': int(len(pieces)), 'outerPieces': len(outer), 'innerWalls': len(cavities),
            'floating': len(floating), 'parts': len(surfaces),
            'roots': 'root-label voxels' if near is not None else 'none: every outer piece but one counts as floating'}


REFERENCE_COVERAGE = 0.9  # Declared before measuring: below this a major vessel counts as lost.


def named_review(receipts, *, display=None, reference=None, spacing=None):
    """Named vessels not fully shown by default, and the "no named major vessel lost" gate.

    With a reference annotation, each major vessel in it is measured on the
    reference: the share of its centerline within one voxel of the default
    display. Below REFERENCE_COVERAGE (or absent from the display) it is lost;
    from there to 100 % it is listed as partly in review. Without a reference the
    gate uses the prediction-side status only and says so. Without named labels
    nothing is measured. TopBrain merges A1 with A2 and P1 with P2, so those
    segments are one class each.
    """
    from scipy import ndimage as ndi
    from resectionlab.vascular_parity import centerline

    rows, major, schemes, unavailable = [], [], set(), []
    for layer_id, receipt in receipts.items():
        rooting = receipt.get('rooting') or {}
        scheme = (rooting.get('roots') or {}).get('scheme')
        if receipt.get('namedLabelsUnavailable'):
            unavailable.append(f"{layer_id}: {receipt['namedLabelsUnavailable']}")
        if scheme and not receipt.get('namedLabelsUnavailable'):
            schemes.add(scheme)
        for row in rooting.get('namedLabels', []):
            if row['status'] in {'shown', 'bridged'}:
                continue
            item = {'layer': layer_id, 'label': row['label'], 'name': row['name'], 'status': row['status'],
                    'voxels': row['voxels'], 'review': row['review'], 'lodHidden': row['lodHidden']}
            rows.append(item)
            if scheme and row['label'] in MAJOR_VESSELS[scheme]:
                major.append(item)
    result = {'inReview': rows, 'majorInReview': major, 'mergedClasses': 'A1-A2 and P1-P2 are one TopBrain class each',
              'majorList': {name: [SCHEMES[name].names[v] for v in labels] for name, labels in MAJOR_VESSELS.items()}}
    if not schemes and reference is None:
        reason = '; '.join(unavailable) or 'No named labels are bound to the displayed layers'
        return {**result, 'gate': 'not-measured', 'reason': reason, 'referenceCheck': 'not available'}
    scheme = next(iter(schemes)) if len(schemes) == 1 else 'topbrain-v1-ct'
    if reference is None or display is None:
        lost = [item for item in major if item['status'] in {'review', 'lod'}]
        return {**result, 'gate': 'not met' if lost else 'met', 'lost': lost,
                'partlyInReview': [item for item in major if item['status'] == 'partly-review'],
                'basis': 'prediction-side status only', 'referenceCheck': 'not available (no reference annotation)'}
    near = ndi.binary_dilation(display, np.ones((3, 3, 3), bool))
    step = float(np.mean(spacing)) if spacing is not None else 1.0
    measured, lost, partly = [], [], []
    for value in MAJOR_VESSELS[scheme]:
        label = reference == value
        if not label.any():
            continue
        box = ndi.find_objects(label.astype(np.uint8))[0]
        line = centerline(label[box])
        covered = line & near[box]
        fraction = float(covered.sum() / max(int(line.sum()), 1))
        item = {'label': int(value), 'name': SCHEMES[scheme].names[value], 'referenceVoxels': int(label.sum()),
                'centerlineVoxels': int(line.sum()), 'covered': round(fraction, 4),
                'uncoveredCenterlineMm': round(float((line & ~covered).sum()) * step, 2),
                'shownVoxels': int((label & display).sum())}
        measured.append(item)
        if not (label & near).any() or fraction < REFERENCE_COVERAGE:
            lost.append(item)
        elif fraction < 1:
            partly.append(item)
    return {**result, 'gate': 'not met' if lost else 'met', 'basis': f'reference centerline coverage >= {REFERENCE_COVERAGE:g}',
            'referenceCheck': measured, 'lost': lost, 'partlyInReview': partly}


def anchor(case):
    """Frozen circle-of-Willis level. Cameras must not move when a reconstruction changes."""
    return np.asarray(case['anchor'], float)


def displayed_surface(collection, layer_id, scope='all'):
    """The walls the desktop opens with: (name, layer row, mesh) for the layer and, in the
    rooted view, its other-class part. Restricted TOF companions show the unconditioned score contour."""
    layer = collection.layer(layer_id)
    companion = (collection.document.get('vascular_review') or {}).get('schema') == 'eden2020-restricted-vascular-companion-v1'
    source = vascular_cleanup_source(collection, layer)
    if scope != 'rooted':
        return [(layer_id, layer, mesh(collection, layer, display_source=source, conditioned=not companion))]
    display, _, _, split = rooted_vessel_layer(collection, layer)
    parts = [(layer_id, display)] + ([(f'{layer_id}:{split["kind"]}', split)] if split else [])
    # As in the desktop view: the filtered mask with the unfiltered score field where it still fits.
    return [(name, row, mesh(collection, row, display_source=source or layer, conditioned=not companion)
             if row is display else mesh(collection, row)) for name, row in parts]


def preset_applies(preset, volume_mode, modality):
    """Bone framings need a CT scan and the path-traced mode; every other preset applies everywhere."""
    return not preset.get('bone') or (volume_mode == 'cinematic' and image_modality(modality) == 'ct')


def requested_presets(text):
    """Preset names from a comma-separated option; an unknown name is an error, not a skip."""
    names = [name.strip() for name in text.split(',') if name.strip()]
    unknown = [name for name in names if name not in PRESETS]
    if unknown or not names:
        raise ValueError(f"Unknown presets {unknown}; choose from {', '.join(PRESETS)}")
    return names


def recipes(collection, case, default_view, default_layers, volume_mode=None, scope='rooted', only=None):
    """View recipes for every preset that applies, then the timed drag and the orbit.
    ``only`` names presets to capture alone, without the timed rows."""
    scan, _, extent = scan_geometry(collection, case)
    target = anchor(case)
    settings = {key: value for key, value in default_view['settings'].items() if key not in {'camera', 'clipping'}}
    if volume_mode:
        # The path-traced mode draws vessels from the acquired volume on every case.
        settings['volumeRendering'] = {**settings.get('volumeRendering', {}), 'enabled': True, 'mode': volume_mode}
    rows = []
    for name, preset in PRESETS.items():
        if only is not None and name not in only:
            continue
        if not preset_applies(preset, volume_mode, scan['modality']):
            continue
        direction = np.asarray(preset['from'], float)
        direction /= np.linalg.norm(direction)
        view = {**settings, 'camera': {
            'position': (target + direction * preset['distance'] * extent).round(4).tolist(),
            'target': target.round(4).tolist(), 'up': preset['up'],
        }}
        if 'skull' in preset:
            view['skullMode'] = preset['skull']
        row = {'name': name, 'seriesId': scan['id'], 'layers': default_layers, 'cursor': target.round(4).tolist(), 'settings': view,
               'scope': scope}
        if 'clip' in preset:
            axis, above, flipped = preset['clip']
            clipping = {'axis': axis, 'value': round(float(target['xyz'.index(axis)] + above), 4), 'enabled': True, 'flipped': flipped}
            view['clipping'] = clipping
            row['patientCutaway'] = {'clipping': clipping}
        rows.append(row)
    if only is not None:
        return rows
    timed = next(row for row in rows if row['name'] == INTERACTION_PRESET)
    rows.append({**timed, 'name': 'interaction', 'interaction': {'frames': 180}})
    rows.append({**timed, 'name': 'orbit', 'rotationFrames': 48})
    return rows


def run_capture(desktop, manifest, views, output, log, *, width, height):
    runtime = desktop / 'node_modules' / 'electron'
    electron = runtime / 'dist' / (runtime / 'path.txt').read_text().strip()
    command = [str(electron), 'electron/capture-vascular-surfaces.cjs', '--anatomy', str(manifest), '--views', str(views),
               '--output', str(output), '--width', str(width), '--height', str(height),
               # Every phase is measured on the built-in Retina panel, whichever display is primary.
               '--display-scale', '2']
    with log.open('w') as stream, subprocess.Popen(
            command, cwd=desktop, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True) as process:
        try:
            code = process.wait(timeout=1500)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    report = json.loads((output / 'report.json').read_text()) if (output / 'report.json').is_file() else {}
    if code or not report.get('passed') or report.get('windowEverVisible'):
        raise RuntimeError(f'Hidden capture failed. See {log}')
    return report


def pane_image(view):
    pane = view['pane']
    box = (pane['x'], pane['y'], pane['x'] + pane['width'], pane['y'] + pane['height'])
    image = Image.open(view['filename']).convert('RGB').crop(box)
    # macOS sometimes returns a Retina backing bitmap for a hidden window.
    # Measure every frame at the logical display scale.
    scale = view['captureScale']
    if scale != 1:
        image = image.resize((round(image.width / scale), round(image.height / scale)), Image.LANCZOS)
    return image


def preset_metrics(view, output):
    image = pane_image(view)
    image.save(output / f"{view['name']}.png")
    quality = view['frame'].get('scanQuality') or {}
    settled = [row['gpu']['milliseconds'] for row in view['settledSamples'] if row.get('gpu', {}).get('completed')]
    return {
        'image': f"{view['name']}.png", 'pane': [image.width, image.height], 'camera': view['frame']['cameraState'],
        'volume': view['frame'].get('volume'), 'mode': quality.get('mode', 'standard'), 'settled_samples': quality.get('samples'),
        'settled_resolution': [quality.get('width'), quality.get('height')],
        'settled_sample_gpu_ms': settled,
        # The app labels an active cut along the bottom of the pane.
        'sharpness': image_sharpness(np.asarray(image)[:-56]),
    }


def add_presets(root, name, case, label, presets):
    """Capture further presets into an existing capture of the same study, settings and window.

    Works on captures that record their scope and vessel layers (Phase 3 onward). Returns the
    presets added and those skipped with the reason."""
    desktop = Path(__file__).resolve().parents[1] / 'desktop'
    manifest = safe_path(root, case['manifest'])
    collection = open_anatomy_manifest(manifest)
    output = evidence(root, label) / 'captures' / name
    metrics = json.loads((output / 'metrics.json').read_text())
    existing = json.loads((output / 'views.json').read_text())
    template = next(row for row in existing if row['name'] == INTERACTION_PRESET)
    if 'vessels' not in metrics or 'scope' not in template:
        raise ValueError(f'{label} predates recorded scope and vessel layers; capture it again instead of adding to it')
    if metrics['manifest_sha256'] != sha256_file(manifest):
        raise ValueError(f'{name}: the study changed since {label} was captured')
    volume_mode = None if metrics['volume_mode'] in (None, 'default') else metrics['volume_mode']
    modality = collection.series(case['series'])['modality']
    skipped = {preset: 'already captured' for preset in presets if preset in metrics['presets']}
    skipped.update({preset: 'needs a CT scan and the path-traced mode' for preset in presets
                    if preset not in skipped and not preset_applies(PRESETS[preset], volume_mode, modality)})
    wanted = [preset for preset in presets if preset not in skipped]
    if not wanted:
        return [], skipped
    rows = recipes(collection, case, {'settings': template['settings']}, template['layers'], volume_mode, template['scope'], only=wanted)
    for layer_id in metrics['vessels']:
        displayed_surface(collection, layer_id, template['scope'])
    atomic_json(output / 'views-added.json', rows)
    report = run_capture(desktop, manifest, output / 'views-added.json', output / 'presets-added', output / 'presets-added.log',
                         width=metrics['window']['width'], height=metrics['window']['height'])
    if (report['snapshotId'], report['windowSize'], report['deviceScale']) != (
            metrics['snapshot_id'], metrics['window'], metrics['device_scale']):
        raise ValueError(f'{name}: the study snapshot or the capture window differs from the one {label} was captured with')
    for view in report['views']:
        metrics['presets'][view['name']] = preset_metrics(view, output)
    atomic_json(output / 'views.json', [*existing, *rows])
    atomic_json(output / 'metrics.json', metrics)
    return wanted, skipped


def capture(root, name, case, label, *, width, height, volume_mode=None, ensemble=None, scope='rooted'):
    """Capture today's default appearance at every fixed preset and time one continuous orbit.

    ``ensemble`` shows the layers wired from that Phase 2 ensemble folder in place of
    the case's vessel layers; ``scope`` is the vessel display ('rooted' is the default view)."""
    desktop = Path(__file__).resolve().parents[1] / 'desktop'
    manifest = safe_path(root, case['manifest'])
    collection = open_anatomy_manifest(manifest)
    scan = collection.series(case['series'])
    output = evidence(root, label) / 'captures' / name
    if output.exists():
        raise FileExistsError(f'{output} already holds a capture. Choose a new label or remove it deliberately.')
    output.mkdir(parents=True)
    vessels = displayed_vessels(collection, case, ensemble, root)
    # With an ensemble the wired layers are the study's own default, so the
    # default-state recipe, these recipes and the geometry gate show the same thing.
    initial = list(collection.document['display']['initial_layers']) if ensemble else \
        [layer for layer in collection.document['display']['initial_layers'] if layer not in case['vessels']] + vessels
    # A slow first contour read would outlast the bounded interactive request.
    for layer_id in vessels:
        displayed_surface(collection, layer_id, scope)
    if case.get('cohort_view'):
        # Restricted TOF companions open through the cohort review view, as in the earlier EDEN captures.
        default_view, initial = {'settings': {}}, vessels
    else:
        default_recipe = [{'name': 'default', 'seriesId': scan['id'], 'layers': initial, 'useDefaultState': True}]
        atomic_json(output / 'default-view.json', default_recipe)
        default = run_capture(desktop, manifest, output / 'default-view.json', output / 'default', output / 'default.log',
                              width=width, height=height)
        default_view = default['views'][0]['view']
    rows = recipes(collection, case, default_view, initial, volume_mode, scope)
    atomic_json(output / 'views.json', rows)
    report = run_capture(desktop, manifest, output / 'views.json', output / 'presets', output / 'presets.log',
                         width=width, height=height)
    metrics = {'case': name, 'title': case['title'], 'held_out': case['held_out'], 'label': label, 'volume_mode': volume_mode or 'default',
               'scope': scope, 'vessels': vessels, 'ensemble': ensemble,
               'manifest_sha256': sha256_file(manifest), 'snapshot_id': report['snapshotId'],
               'window': report['windowSize'], 'device_scale': report['deviceScale'], 'display': report.get('display'), 'presets': {}}
    for view in report['views']:
        if view['name'] in PRESETS:
            metrics['presets'][view['name']] = preset_metrics(view, output)
        elif view['name'] == 'interaction':
            metrics['interaction'] = {key: value for key, value in view['interaction'].items() if key != 'samples'}
        elif view['name'] == 'orbit':
            response = [row['toFencePollMs'] for row in view['rotationResponse']['samples']]
            metrics['orbit'] = {'frames': view['rotationFrames'], 'input_to_fence_poll_ms_p50_p95': np.percentile(response, [50, 95]).tolist()}
            frames = output / 'presets' / 'orbit-rotation'
            movie = output / 'orbit.mp4'
            pane, scale = view['pane'], view['captureScale']
            crop = (f"crop={pane['width'] // 2 * 2}:{pane['height'] // 2 * 2}:{pane['x']}:{pane['y']},"
                    f"scale=trunc(iw/{scale}/2)*2:-2")
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', '12', '-i', str(frames / '%04d.png'), '-vf', crop,
                            '-pix_fmt', 'yuv420p', '-crf', '18', str(movie)], check=True)
            # The frames exist only to encode the film. exFAT sidecar files
            # disappear with their owners, so tolerate entries that vanish.
            shutil.rmtree(frames, ignore_errors=True)
            if frames.exists():
                shutil.rmtree(frames)
            metrics['orbit']['movie'] = movie.name
    atomic_json(output / 'metrics.json', metrics)
    return metrics


def reference_labels(root, case, shape, frame):
    if not case.get('reference'):
        return None
    image = nib.load(safe_path(root, case['reference']))
    if image.shape != tuple(shape) or not np.allclose(image.affine, frame, atol=1e-3):
        raise ValueError('Reference annotation and study grids differ')
    return np.asarray(image.dataobj)


def display_roots(collection, layer_ids, shape, receipts=None, display=None):
    """Root voxels from the named labels bound to the displayed layers; without
    any, the seed voxels of the unnamed fallback roots recorded in the receipts; else None."""
    from resectionlab.vascular_cleanup import rooted_named_labels
    roots = None
    for layer_id in layer_ids:
        layer = collection.layer(layer_id)
        named, scheme, _, _ = rooted_named_labels(collection, layer, collection.series(layer['source_series']))
        if named is not None and named.shape == tuple(shape):
            found = np.isin(named, SCHEMES[scheme].roots)
            roots = found if roots is None else roots | found
    if roots is None and receipts:
        seeds = [row['seedIjk'] for receipt in receipts.values()
                 for row in ((receipt.get('rooting') or {}).get('roots') or {}).get('acceptedRoots', []) if 'seedIjk' in row]
        if seeds:
            # A seed is the deepest voxel of its root component; its root geometry is the
            # ball reaching from it to the wall of the displayed vessel.
            from scipy import ndimage as ndi
            roots = np.zeros(shape, bool)
            for seed in seeds:
                box = tuple(slice(max(0, v - 12), min(n, v + 13)) for v, n in zip(seed, shape))
                local = display[box] if display is not None else np.ones([b.stop - b.start for b in box], bool)
                depth = ndi.distance_transform_edt(local)
                centre = tuple(v - b.start for v, b in zip(seed, box))
                grid = np.ogrid[tuple(slice(0, b.stop - b.start) for b in box)]
                ball = sum((g - c) ** 2 for g, c in zip(grid, centre)) <= (depth[centre] + 1.5) ** 2
                roots[box] |= ball & local
    return roots


def geometry(root, name, case, label, *, ensemble=None, scope='rooted'):
    """Measure the displayed vessel geometry, and reference agreement where an annotation exists.

    In the rooted scope the measured mask is what the default view shows: the rooted
    union of the displayed layers, with the named vessels left in review listed."""
    collection = open_anatomy_manifest(safe_path(root, case['manifest']))
    scan, frame, _ = scan_geometry(collection, case)
    spacing = np.linalg.norm(frame[:3, :3], axis=0)
    signal = collection.array(scan['data'])
    vessels = displayed_vessels(collection, case, ensemble, root)
    receipts = {}
    if scope == 'rooted':
        mask, receipts = rooted_display(collection, vessels)
    else:
        mask = vessel_mask(collection, {'vessels': vessels})
    labels = reference_labels(root, case, mask.shape, frame)
    roots = display_roots(collection, vessels, mask.shape, receipts, mask)
    root_source = 'named roots of the displayed layers, else the recorded fallback root seeds'
    if roots is None and labels is not None:
        roots, root_source = np.isin(labels, ROOT_LABELS), 'named roots of the reference annotation'
    if roots is not None and not (roots & mask).any():
        roots = None
    result = {'case': name, 'label': label, 'scope': scope, 'ensemble': ensemble, 'spacing_mm': spacing.tolist(),
              'displayed_layers': vessels, 'displayed_voxels': int(mask.sum()),
              'floating': {**floating_components(mask, roots, spacing),
                           'roots': root_source if roots is not None else 'largest component'},
              'betti': betti(mask), 'surfaces': {}}
    result['named_review'] = named_review(receipts, display=mask, reference=labels, spacing=spacing)
    if receipts:
        result['rooting'] = {layer_id: {key: receipt['rooting'][key] for key in ('roots', 'attached', 'bridged', 'review', 'lod', 'display')}
                             for layer_id, receipt in receipts.items() if 'rooting' in receipt}
    parts = [part for vessel in vessels for part in displayed_surface(collection, vessel, scope)]
    # The gate counts pieces of the drawn surfaces, independently of the rooting code.
    result['floating_mesh'] = mesh_floating([(name, surface) for name, _, surface in parts], roots, mask.shape,
                                            [lambda row=row: drawn_solid(collection, row) for _, row, _ in parts])
    from scipy import ndimage as ndi
    faces6, count6 = ndi.label(mask)
    rooted6 = set(np.unique(faces6[roots & mask]).tolist()) - {0} if roots is not None else set()
    result['floating_face_voxels'] = {'components': int(count6), 'floating': int(count6 - len(rooted6)) if roots is not None else None}
    for layer_id, layer, surface in parts:
        points, faces = np.asarray(surface['positions'], float), np.asarray(surface['indices']).reshape(-1, 3)
        physical = world(affine(layer['affine']), points)
        a, b, c = physical[faces].transpose(1, 0, 2)
        face_normals = np.cross(b - a, c - a)
        # A left-handed scan affine reverses the winding. Enclosed volume fixes which side is outside.
        if np.einsum('ij,ij->', a, np.cross(b, c)) < 0:
            face_normals = -face_normals
        vertex_normals = np.zeros_like(physical)
        for corner in range(3):
            np.add.at(vertex_normals, faces[:, corner], face_normals)
        vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1), 1e-20)[:, None]
        result['surfaces'][layer_id] = {
            'surface_field': str(surface.get('surface_field', 'binary-mask')), 'vertices': int(len(points)),
            **surface_quality(physical, faces),
            'source_deviation': source_deviation(physical, vertex_normals, signal, frame),
        }
    if labels is not None:
        supplied = bool((mask == (labels > 0)).all())
        result['reference'] = {'file': case['reference'], 'displayed': {
            **agreement(mask, labels > 0, spacing),
            'independent': not supplied,
            **({'note': 'The displayed layers are the annotation itself, so this row is self-agreement'} if supplied else {}),
        }}
        unrooted = floating_components(labels > 0, roots, spacing)
        result['reference']['annotation_floating'] = unrooted
        if case.get('model_probability'):
            runs = sorted(safe_path(root, case['model_probability']).glob('*/vessels_probability.nii.gz'))
            for run in runs:
                image = nib.load(run)
                if not np.allclose(image.affine, frame, atol=1e-3):
                    raise ValueError('Model output and study grids differ')
                probability = np.asarray(image.dataobj, np.float32)
                result['reference'][f'vesselfm-{run.parent.name[:8]}'] = {
                    'file': str(run.relative_to(root)), 'threshold': 0.5, **agreement(probability >= 0.5, labels > 0, spacing),
                    'floating': floating_components(probability >= 0.5, roots, spacing),
                }
    else:
        result['reference'] = {'status': 'unavailable', 'reason': 'No independent vessel annotation exists for this study'}
    output = evidence(root, label) / 'geometry'
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / f'{name}.json', result)
    return result


def sheet(root, label, references, compare=None):
    """One row per preset: reference frame, each compared label, then this label."""
    labels = [*(compare or []), label]
    output = evidence(root, label) / 'sheets'
    output.mkdir(parents=True, exist_ok=True)
    cell = (560, 420)
    for name in HERO:
        rows, used = [], []
        for preset in PRESETS:
            folders = [evidence(root, item) / 'captures' / name for item in labels]
            # A capture without its metrics stopped part-way and is not evidence.
            if not all((folder / 'metrics.json').is_file() for folder in folders):
                break
            if not any((folder / f'{preset}.png').is_file() for folder in folders):
                continue  # A preset that does not apply to this case, such as a bone framing on TOF.
            images = [references.get(preset)] + [folder / f'{preset}.png' for folder in folders]
            used.append(preset)
            rows.append([Image.open(path).convert('RGB') if path and Path(path).is_file() else None for path in images])
        if not rows:
            continue
        canvas = Image.new('RGB', (cell[0] * len(rows[0]), cell[1] * len(rows)), (8, 8, 8))
        for y, row in enumerate(rows):
            for x, image in enumerate(row):
                if image is None:
                    continue
                image.thumbnail(cell, Image.LANCZOS)
                canvas.paste(image, (x * cell[0] + (cell[0] - image.width) // 2, y * cell[1] + (cell[1] - image.height) // 2))
        canvas.save(output / f'{name}.jpg', quality=90)
        atomic_json(output / f'{name}.json', {'columns': ['reference', *labels], 'rows': used})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['wire', 'unwire', 'capture', 'geometry', 'sheet'])
    parser.add_argument('--label', help='Evidence folder under the parity root, such as phase0-baseline')
    parser.add_argument('--case', action='append', choices=list(HERO))
    parser.add_argument('--compare', action='append', help='Earlier labels to place before this one on a sheet')
    parser.add_argument('--references', type=Path, help='Folder of reference frames named after each preset')
    parser.add_argument('--volume-mode', choices=['cinematic'], help='Capture with this volume render mode instead of the default view')
    parser.add_argument('--ensemble', help='Phase 2 ensemble folder name (for example ensemble-UZH-ARG-KDH): wire it, '
                        'or show its wired layers instead of the case vessel layers')
    parser.add_argument('--add', help='With capture: comma-separated presets to add to an existing capture of this label '
                        '(Phase 3 onward), with its saved settings, layers, scope and window; --volume-mode, --ensemble, '
                        '--scope, --width and --height are ignored, and presets that do not apply to a case are reported as skipped')
    parser.add_argument('--scope', choices=['rooted', 'all'], default='rooted',
                        help='Vessel display to capture and measure; rooted is the default view')
    parser.add_argument('--width', type=int, default=1440)
    parser.add_argument('--height', type=int, default=960)
    args = parser.parse_args()
    root = checked_root(HDD_ROOT, expected_uuid=HDD_UUID)
    cases = args.case or list(HERO)
    if args.command in {'wire', 'unwire'}:
        if args.command == 'wire' and not args.ensemble:
            parser.error('wire needs --ensemble')
        failures = 0
        for name in cases:
            manifest = safe_path(root, HERO[name]['manifest'])
            try:
                collection = open_anatomy_manifest(manifest)
                if args.command == 'unwire':
                    receipt = unwire(collection, manifest=manifest)
                else:
                    relative = PHASE2 / HERO[name]['phase2'] / args.ensemble
                    folder = safe_path(root, relative)
                    if not (folder / 'ensemble.json').is_file():
                        print(name, 'no ensemble at', relative, flush=True)
                        continue
                    receipt = wire_ensemble(collection, HERO[name]['series'], folder, manifest=manifest, source=str(relative))
                    receipt['undo'] = receipt['undo'].replace('<case>', name)
                    if receipt['changed']:
                        record = evidence(root, 'phase3-cleanup') / 'wiring' / f"{name}-{receipt['binding']}-{receipt['snapshot']['after'][:12]}.json"
                        record.parent.mkdir(parents=True, exist_ok=True)
                        atomic_json(record, receipt)
                print(name, args.command, json.dumps({key: receipt.get(key) for key in ('changed', 'layers', 'added', 'retired', 'removed_layers')}), flush=True)
            except Exception as error:  # One case must not stop the others.
                failures += 1
                print(name, args.command, 'failed:', type(error).__name__, error, flush=True)
        return 1 if failures else 0
    if not args.label:
        parser.error(f'{args.command} needs --label')
    if args.command == 'sheet':
        references = {preset: args.references / f'{REFERENCE_FRAME.get(preset, preset)}.png' for preset in PRESETS} if args.references else {}
        sheet(root, args.label, references, args.compare)
        return 0
    mesh_failures = []
    for name in cases:
        if args.command == 'capture' and args.add:
            added, skipped = add_presets(root, name, HERO[name], args.label, requested_presets(args.add))
            print(name, 'added', ', '.join(added) or 'none', '; skipped',
                  ', '.join(f'{preset} ({why})' for preset, why in skipped.items()) or 'none', flush=True)
            continue
        if args.command == 'capture':
            metrics = capture(root, name, HERO[name], args.label, width=args.width, height=args.height,
                              volume_mode=args.volume_mode, ensemble=args.ensemble, scope=args.scope)
            timing = metrics.get('interaction', {})
            print(name, 'frame interval p95', timing.get('frameIntervalMs', {}).get('p95'), 'ms; settle',
                  (timing.get('settle') or {}).get('milliseconds'), 'ms', flush=True)
        else:
            result = geometry(root, name, HERO[name], args.label, ensemble=args.ensemble, scope=args.scope)
            drawn = result['floating_mesh']
            print(name, 'mesh floating', drawn['floating'], 'of', drawn['pieces'], 'pieces; voxel floating',
                  result['floating']['floating'], 'of', result['floating'].get('components'), 'components; betti', result['betti'],
                  'major named in review', [row['name'] for row in result.get('named_review', {}).get('majorInReview', [])],
                  flush=True)
            if args.scope == 'rooted' and drawn['floating']:
                mesh_failures.append(name)
    if mesh_failures:
        print('mesh floating pieces in the default view:', ', '.join(mesh_failures), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
