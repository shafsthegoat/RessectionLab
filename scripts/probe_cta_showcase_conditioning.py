#!/usr/bin/env python3
"""Audit fixed-topology skull and brain display refinement on the three native CTA showcases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_cta_native_showcases import BRAIN, OUTPUT as SHOWCASES, SERIES, SKULL, SUBJECTS
from data_integrated_hosts import UUID, checked_file
from probe_cads_vault_conditioning import assess, retain_result
from resectionlab.anatomy_geometry import mesh
from resectionlab.data_root import checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection, reconstruction_lock

SCHEMA = 'cta-showcase-surface-conditioning-v1'
CROP_MARGIN = 2
CROP_TAPER = 6
SUPERIOR_DEPTH_MM = 70
SUPERIOR_TAPER_MM = 15


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def open_stock(root, subject):
    case = f'TopCoW-{subject}-CTA-ORGANIC-01-v1-NATIVE-SHOWCASE'
    directory = safe_path(root, f'{SHOWCASES}/{case}')
    receipt = json.loads((directory / 'receipt.json').read_text())
    manifest = receipt.get('stock_manifest')
    if manifest is None:
        # The saved stock revision remains valid if a live manifest was
        # replaced before its outer receipt could be updated.
        manifest = {**receipt['manifest'], 'path':
                    f"{SHOWCASES}/{case}/revisions/{receipt['snapshot_id']}.json"}
    collection = PatientImagingCollection(directory, json.loads(checked_file(root, manifest).read_text()))
    scan = collection.series(SERIES)
    if (collection.document['patient_id'] != case or collection.document['data_case']['role'] != 'DEVELOPMENT'
            or collection.document['data_case']['source_family_id'] != f'TopCoW:{subject}'
            or scan['modality'] != 'CTA' or scan['original']['sha256'] != receipt['source']['cta']['sha256']
            or sha256_file(collection.path(scan['original']['path'])) != scan['original']['sha256']):
        raise ValueError('Expected the exact reviewed development CTA showcase')
    return collection, manifest, scan


def probe(root, subject, layer_id, output):
    collection, manifest, scan = open_stock(root, subject)
    layer = collection.layer(layer_id)
    matrix = np.asarray(layer['affine'])
    linear = matrix[:3, :3]
    if (layer['role'] != 'context' or layer.get('edits') or layer.get('conditioned_surface')
            or layer['revision'] != 0 or not np.array_equal(matrix, scan['affine'])
            or not np.array_equal(linear, np.diag(np.diag(linear))) or linear[2, 2] <= 0):
        raise ValueError('Expected the unedited scanner-aligned source layers')
    ct = collection.array(scan['data'])
    collection.array(layer['data'])
    roof = int(np.flatnonzero(collection.mask(collection.layer(BRAIN)).any(axis=(0, 1)))[-1])
    lower = int(np.ceil(roof - SUPERIOR_DEPTH_MM / linear[2, 2]))
    upper = lower + int(np.ceil(SUPERIOR_TAPER_MM / linear[2, 2]))

    stock = mesh(collection, layer)
    raw = stock['source']
    crop_distance = np.minimum(raw, np.asarray(ct.shape) - 1 - raw).min(axis=1)
    protected = (raw[:, 2] <= lower) | (crop_distance <= CROP_MARGIN)
    measured = (raw[:, 2] >= upper) & (crop_distance >= CROP_MARGIN + CROP_TAPER)
    weight = smoothstep((raw[:, 2] - lower) / (upper - lower)) * smoothstep((crop_distance - CROP_MARGIN) / CROP_TAPER)
    if not protected.any() or np.count_nonzero(measured) < 3000:
        raise ValueError('Insufficient retained source surface above the protected levels')
    print(f'{collection.document["patient_id"]} · {layer_id} · levels {lower}–{upper}', flush=True)
    report, candidate = assess(stock, ct, linear, protected, measured, weight)
    report.update(schema=SCHEMA, case_id=collection.document['patient_id'], source_person=f'TopCoW:{subject}',
                  source_manifest=manifest, source_ct=scan['original'], source_ct_array=scan['data'],
                  mask=layer['data'], snapshot=collection.identity(), layer=layer_id,
                  affine=matrix.tolist(), protected_native_plane_inclusive=lower,
                  full_effect_native_plane=upper, protected_crop_margin_cells=CROP_MARGIN,
                  crop_taper_cells=CROP_TAPER, estimated_brain_superior_index=roof,
                  measured_vertices=int(np.count_nonzero(measured)))
    return retain_result(root, output, report, candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--layer', choices=[SKULL, BRAIN], action='append')
    parser.add_argument('--output-name', default='cta-showcase-conditioned-surfaces-v1')
    parser.add_argument('subjects', choices=sorted(SUBJECTS), nargs='+')
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or args.output_name in ('', '.', '..'):
        parser.error('--output-name must name a directory under qc')
    root = checked_root(args.root, expected_uuid=UUID, required_bytes=1024**3)
    folder = safe_path(root, f'qc/{args.output_name}')
    with reconstruction_lock():
        folder.mkdir(exist_ok=True)
        for subject in dict.fromkeys(args.subjects):
            for layer in dict.fromkeys(args.layer or (SKULL, BRAIN)):
                path = folder / f'{subject}-{layer}'
                if path.exists():
                    raise FileExistsError(path)
                result = probe(root, subject, layer, path)
                print(f"{subject} {layer} numeric gate {result['passed_numeric_gate']}: {path / 'report.json'}", flush=True)


if __name__ == '__main__':
    main()
