#!/usr/bin/env python3
"""Attach audited CADS vault positions without replacing the native label or topology."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_integrated_hosts import UUID, binding, checked_file
from prepare_cads_brainct import CASE, DESTINATION, SERIES
from probe_cads_vault_conditioning import OUTPUT, LOWER_INDEX
from resectionlab.anatomy_geometry import CONDITIONED_SURFACE, mesh
from resectionlab.anatomy_package import referenced_files
from resectionlab.core import array_digest
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection, reconstruction_lock


def attach_positions(root, collection, row, scan, proof, numeric_path):
    positions = np.load(checked_file(root, proof['positions']), allow_pickle=False)
    if array_digest(positions) != proof['candidate_hash']:
        raise ValueError('Candidate geometry bytes differ')
    row['conditioned_surface'] = {
        'schema': CONDITIONED_SURFACE, 'mask_revision': row['revision'],
        'label_value': row['label_value'], 'mask_sha256': row['data']['sha256'],
        'source_array_sha256': scan['data']['sha256'], 'source_original_sha256': scan['original']['sha256'],
        'affine': row['affine'], 'source_mesh_hash': proof['source_mesh_hash'],
        'indices_hash': proof['indices_hash'], 'stock_conditioned_hash': proof['stock_conditioned_hash'],
        'stock_manifest_sha256': proof['source_manifest']['sha256'],
        'protected_native_plane': proof['protected_native_plane_inclusive'],
        'positions': collection.put_array(positions), 'proof': collection.original(numeric_path),
    }
    if 'protected_crop_margin_cells' in proof:
        row['conditioned_surface']['protected_crop_margin_cells'] = proof['protected_crop_margin_cells']
    geometry = mesh(collection, row)
    if array_digest(geometry['positions']) != proof['candidate_hash']:
        raise ValueError('Native renderer geometry does not use the measured candidate')


def install(root, study, numeric_path, desktop_path):
    proof = json.loads(numeric_path.read_text())
    if proof['schema'] != OUTPUT or proof['passed_numeric_gate'] is not True or proof['new_contacts'] != 0:
        raise ValueError('Expected successful source-bound geometric verification')
    canonical = safe_path(root, DESTINATION)
    receipt_path = canonical / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if study == canonical:
        if desktop_path is None:
            raise ValueError('Provide the successful hidden A/B report before publishing')
        desktop = json.loads(desktop_path.read_text())
        comparison = desktop.get('refinedComparison', {})
        if (desktop.get('passed') is not True or desktop.get('baselineMatched') is not True
                or desktop.get('refinedColdReopened') is not True
                or desktop.get('windowEverVisible') is not False or comparison.get('layer') != 'head-bone'
                or comparison.get('stockManifest') != proof['source_manifest']['sha256']
                or comparison.get('probeSha256') != sha256_file(numeric_path)):
            raise ValueError('Desktop evidence does not verify this exact candidate against stock')
    elif not study.is_relative_to(safe_path(root, 'qc')):
        raise ValueError('Stage a trial under the verified QC directory')

    collection = PatientImagingCollection(study)
    row = collection.layer('head-bone')
    scan = collection.series(SERIES)
    if (collection.document['patient_id'] != CASE or proof['layer'] != row['id']
            or row.get('edits') or row['revision'] != 0
            or row['data'] != proof['mask'] or scan['data'] != proof['source_ct_array']
            or scan['publisher_source'] != proof['source_ct'] or scan['affine'] != proof['affine']
            or proof['protected_native_plane_inclusive'] != LOWER_INDEX):
        raise ValueError('Refinement belongs to different anatomy or a different source study')

    manifest = study / 'patient-anatomy.json'
    stock = collection.path(f"revisions/{proof['snapshot']}.json")
    stock_sha = proof['source_manifest']['sha256']
    if sha256_file(stock) != stock_sha:
        raise ValueError('The immutable stock revision does not match the numerical proof')
    if row.get('conditioned_surface') is None:
        if sha256_file(manifest) != stock_sha:
            raise ValueError('Start from the audited stock study snapshot')
        attach_positions(root, collection, row, scan, proof, numeric_path)
        collection.document['data_case']['source_receipt'] = collection.original(checked_file(root, receipt['acquisition']))
        if any(sha256_file(collection.path(name)) != sha for name, sha in set(referenced_files(collection.document))):
            raise ValueError('A portable archive member is missing or changed')
        collection.save('Attach source-bound superior vault display positions', expected=proof['snapshot'])
    elif row['conditioned_surface']['proof']['sha256'] != sha256_file(numeric_path):
        raise ValueError('The existing study uses different conditioning evidence')

    if study == canonical:
        receipt.update(stock_manifest=binding(root, stock), manifest=binding(root, manifest),
                       snapshot_id=collection.identity(), conditioned_vault={
                           'numeric': binding(root, numeric_path), 'desktop': binding(root, desktop_path),
                           'source_displacement_mm': proof['source_displacement_mm'],
                           'mean_squared_turn_reduction': proof['mean_squared_turn_reduction'],
                       })
        atomic_json(receipt_path, receipt)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--study', type=Path, help='A staged copy below qc. Default is the live CADS study.')
    parser.add_argument('--probe', type=Path, default=Path(f'qc/{OUTPUT}/report.json'))
    parser.add_argument('--desktop-proof', type=Path)
    args = parser.parse_args()
    root = checked_root(args.root, expected_uuid=UUID, required_bytes=1024**3)
    study = safe_path(root, args.study or DESTINATION).resolve()
    numeric = safe_path(root, args.probe)
    desktop = safe_path(root, args.desktop_proof) if args.desktop_proof else None
    with reconstruction_lock():
        print(install(root, study, numeric, desktop))


if __name__ == '__main__':
    main()
