#!/usr/bin/env python3
"""Stage or install reviewed CTA display refinement without changing source anatomy."""
from __future__ import annotations

import argparse
from pathlib import Path
from shutil import copytree, ignore_patterns

import numpy as np

from build_cta_native_showcases import BRAIN, SERIES, SKULL, SUBJECTS, read_json
from data_integrated_hosts import UUID, binding, checked_file
from install_cads_vault_conditioning import attach_positions
from probe_cta_showcase_conditioning import CROP_MARGIN, SCHEMA, open_stock
from resectionlab.anatomy_package import referenced_files
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection, reconstruction_lock

LAYERS = (SKULL, BRAIN)


def portable_provenance(root, collection):
    for row in collection.document['layers']:
        source = row['provenance']['source']
        if 'path' not in source or collection.path(source['path']).is_file():
            continue
        upstream = checked_file(root, source)
        row['provenance']['source'] = (
            {'root_relative_locator': source['path'], 'sha256': source['sha256']}
            if row['kind'] == 'tumor' else collection.original(upstream)
        )


def install(root, subject, proofs, trial, desktop):
    stock, manifest, _ = open_stock(root, subject)
    receipt_path = stock.root / 'receipt.json'
    receipt = read_json(receipt_path)
    if trial is None:
        if desktop is None:
            raise ValueError('The live study requires reviewed GPU evidence for both layers')
        evidence = read_json(desktop)
        compared = evidence.get('refinedComparisons', [])
        comparisons = {item['layer']: item for item in compared}
        if (evidence.get('passed') is not True or evidence.get('baselineMatched') is not True
                or evidence.get('windowEverVisible') is not False
                or len(compared) != len(LAYERS) or set(comparisons) != set(LAYERS)
                or any(comparisons[k]['stockManifest'] != manifest['sha256']
                       or comparisons[k]['probeSha256'] != sha256_file(proofs[k]) for k in LAYERS)):
            raise ValueError('The hidden A/B proof does not bind both exact candidates')
        directory = stock.root
    else:
        if not trial.is_relative_to(safe_path(root, 'qc')):
            raise ValueError('Stage the trial under the verified QC directory')
        if not trial.exists():
            copytree(stock.root, trial, ignore=ignore_patterns('._*'))
            # The copy can be staged from a live study already using refinement.
            (trial / 'patient-anatomy.json').write_bytes(checked_file(root, manifest).read_bytes())
        directory = trial
    collection = PatientImagingCollection(directory)
    scan = collection.series(SERIES)
    numeric = {}
    expected_snapshot = stock.identity()
    stock_copy = collection.path(f'revisions/{expected_snapshot}.json')
    if sha256_file(stock_copy) != manifest['sha256']:
        raise ValueError('The immutable stock snapshot differs')
    if collection.identity() != expected_snapshot and any(
            collection.layer(k).get('conditioned_surface', {}).get('proof', {}).get('sha256') != sha256_file(proofs[k])
            for k in LAYERS):
        raise ValueError('Refinement requires the frozen stock snapshot or exact attached candidate')
    if collection.identity() == expected_snapshot:
        for key in LAYERS:
            row = collection.layer(key)
            proof = read_json(proofs[key])
            if (proof['schema'] != SCHEMA or not proof['passed_numeric_gate'] or proof['new_contacts'] != 0
                    or proof['snapshot'] != expected_snapshot or proof['case_id'] != collection.document['patient_id']
                    or proof['source_manifest']['sha256'] != manifest['sha256'] or proof['layer'] != key
                    or proof['mask'] != row['data'] or proof['source_ct_array'] != scan['data']
                    or proof['source_ct'] != scan['original'] or proof['affine'] != scan['affine']
                    or proof['protected_crop_margin_cells'] != CROP_MARGIN
                    or not np.array_equal(row['affine'], scan['affine'])):
                raise ValueError('The candidate belongs to different source anatomy')
            attach_positions(root, collection, row, scan, proof, proofs[key])
        portable_provenance(root, collection)
        for name, digest in set(referenced_files(collection.document)):
            if sha256_file(collection.path(name)) != digest:
                raise ValueError(f'The portable source member differs: {name}')
        collection.save('Attach independently verified source-bound brain and skull display surfaces',
                        expected=expected_snapshot)

    path = directory / 'patient-anatomy.json'
    if trial is None:
        if 'stock_receipt' not in receipt:
            historical = collection.path(f'revisions/{expected_snapshot}-receipt.json')
            historical.write_bytes(receipt_path.read_bytes())
            receipt['stock_receipt'] = binding(root, historical)
        for key in LAYERS:
            numeric[key] = binding(root, proofs[key])
        receipt.update(stock_manifest=binding(root, stock_copy), manifest=binding(root, path),
                       snapshot_id=collection.identity(),
                       conditioned_surfaces={'numeric': numeric, 'desktop': binding(root, desktop)})
        atomic_json(receipt_path, receipt)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--proofs', type=Path, default=Path('qc/cta-showcase-conditioned-surfaces-v1'))
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument('--trial', type=Path)
    destination.add_argument('--desktop-proof', type=Path)
    parser.add_argument('subject', choices=sorted(SUBJECTS))
    args = parser.parse_args()
    root = checked_root(args.root, expected_uuid=UUID, required_bytes=1024**3)
    proofs = {name: safe_path(root, args.proofs / f'{args.subject}-{name}/report.json') for name in LAYERS}
    trial = safe_path(root, args.trial) if args.trial else None
    desktop = safe_path(root, args.desktop_proof) if args.desktop_proof else None
    with reconstruction_lock():
        print(install(root, args.subject, proofs, trial, desktop))


if __name__ == '__main__':
    main()
