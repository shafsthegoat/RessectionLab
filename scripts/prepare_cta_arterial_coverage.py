#!/usr/bin/env python3
"""Create a native CTA workspace with combined and focused supplied arteries."""
from argparse import ArgumentParser
from copy import deepcopy
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np

from resectionlab.anatomy_display import add_cta_arterial_coverage
from resectionlab.anatomy_package import referenced_files
from resectionlab.data_root import atomic_json, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection


def prepare(manifest, destination, series_id, focus, wider):
    if isinstance(wider, str):
        wider = [wider]
    manifest = manifest.resolve(strict=True)
    if manifest.name != 'patient-anatomy.json':
        raise ValueError('Select the native patient-anatomy.json')
    source_sha = sha256_file(manifest)
    source = PatientImagingCollection(manifest.parent)
    destination = destination.resolve()
    if destination.exists() or source.root in destination.parents:
        raise ValueError('Use a new destination outside the source workspace')
    destination.parent.mkdir(parents=True, exist_ok=True)
    inventory = set(referenced_files(source.document))
    with TemporaryDirectory(prefix='.cta-arterial-coverage-', dir=destination.parent) as tmp:
        stage = Path(tmp) / destination.name
        stage.mkdir()
        document = deepcopy(source.document)
        document.pop('snapshot_id')
        document.update(patient_id=destination.name, revision=0, history=[])
        for row in document['series'] + document['layers']:
            row['patient_id'] = destination.name
        collection = PatientImagingCollection(stage, document)
        for name, digest in sorted(inventory):
            new_file = collection.path(name)
            new_file.parent.mkdir(parents=True, exist_ok=True)
            copyfile(source.path(name), new_file)
            if sha256_file(new_file) != digest:
                raise ValueError(f'Original study member changed: {name}')
        combined, counts = add_cta_arterial_coverage(collection, series_id, focus, wider)
        collection.save('Combine supplied arterial positives on the native CTA grid')
        cold = PatientImagingCollection(stage)
        expected = source.mask(source.layer(wider[0]))
        for name in [*wider[1:], *focus]:
            expected |= source.mask(source.layer(name))
        if sha256_file(manifest) != source_sha or not np.array_equal(cold.mask(cold.layer(combined['id'])), expected):
            raise ValueError('Prepared study or immutable source no longer matches its inputs')
        report = {'input_snapshot': source.identity(), 'input_manifest_sha256': source_sha,
                  'manifest_sha256': sha256_file(stage / manifest.name), 'snapshot_id': cold.identity(),
                  'series_id': series_id, 'source_scan_sha256': source.series(series_id)['original']['sha256'],
                  'combined_layer': combined['id'], 'exact_input_union': True, **counts}
        atomic_json(stage / 'arterial-coverage-receipt.json', report)
        stage.rename(destination)
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--series', required=True)
    parser.add_argument('--focus', required=True, nargs='+')
    parser.add_argument('--wider', required=True, nargs='+')
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.destination, args.series, args.focus, args.wider), indent=2))


if __name__ == '__main__':
    main()
