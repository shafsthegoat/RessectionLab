#!/usr/bin/env python3
"""Install checked source graphs and capture the staged native TOF cohort, one subject at a time."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile

import numpy as np

from build_eden2020_vascular_companion import CANDIDATE, HDD_ROOT, HDD_UUID, RUN, SOURCE
from resectionlab.anatomy_geometry import mesh, read_source_branch_graph, source_branch_cache_path, vascular_cleanup_source
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection, world


def install_graph(collection, candidate, prepared):
    if not prepared.is_file():
        return None
    source = vascular_cleanup_source(collection, candidate)
    if source is None:
        raise ValueError('No matching native baseline score and membership binding')
    cache = source_branch_cache_path(collection, source)
    checksum = sha256_file(prepared)
    if cache.is_file() and sha256_file(cache) == checksum:
        return {'file': str(cache), 'sha256': checksum, 'reused': True}
    read_source_branch_graph(collection, source, prepared)
    cache.parent.mkdir(exist_ok=True)
    fd, name = tempfile.mkstemp(dir=cache.parent, prefix='source-branch-')
    os.close(fd)
    provisional = Path(name)
    try:
        shutil.copyfile(prepared, provisional)
        if sha256_file(provisional) != checksum:
            raise ValueError('Source branch report changed during installation')
        os.replace(provisional, cache)
    finally:
        provisional.unlink(missing_ok=True)
    return {'file': str(cache), 'sha256': checksum, 'reused': False}


def capture_case(electron, desktop, manifest, collection, selected, output, subject):
    scan = collection.series(selected['source_series'])
    frame = np.asarray(scan['affine'])
    shape = np.asarray(scan['data']['shape'])
    center = world(frame, (shape-1)/2)
    positive = np.argwhere(collection.mask(selected))
    if not len(positive):
        raise ValueError('Selected vessel mask is empty')
    # Prime assets outside the bounded interactive request. A slow native score
    # read can finish once here and all subsequent views reuse its contour.
    surface = mesh(collection, selected, display_source=vascular_cleanup_source(collection, selected), conditioned=False)
    if not len(surface['indices']):
        raise ValueError('Selected vessel mask did not yield a native contour')
    print(f'{subject} source contour prepared ({len(surface["indices"]):,} triangles)', flush=True)
    points = world(frame, positive)
    cursor = np.median(points, axis=0)
    cursor[2] = np.percentile(points[:, 2], 30)
    recipe = [{
        'name': 'native-tof-superior', 'seriesId': scan['id'], 'layers': [selected['id']],
        'cursor': cursor.tolist(), 'sourceSection': True,
        'settings': {'activePlane': 'axial', 'camera': {
            'position': (center+[0., -.47, 470.]).tolist(), 'target': center.tolist(), 'up': [0., 1., 0.],
        }},
    }]
    attempt = 1
    while (output/f'{subject}-review-{attempt:02d}').exists():
        attempt += 1
    run = output/f'{subject}-review-{attempt:02d}'
    config = output/f'{run.name}-view.json'
    log = output/f'{run.name}.log'
    atomic_json(config, recipe)
    with log.open('w') as stream:
        command = [str(electron), 'electron/capture-vascular-surfaces.cjs', '--anatomy', str(manifest),
                   '--views', str(config), '--output', str(run)]
        with subprocess.Popen(command, cwd=desktop, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True) as process:
            try:
                exit_code = process.wait(timeout=480)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
    report = run/'report.json'
    receipt = json.loads(report.read_text()) if report.is_file() else {}
    if exit_code or not receipt.get('passed') or not receipt['views'][0].get('section'):
        raise ValueError(f'Hidden application capture failed. See {log}')
    return {'report': str(report), 'snapshot_id': receipt['snapshotId'], 'surface_field': str(surface['surface_field']),
            'layer': selected['id'], 'images': [receipt['views'][0]['filename'], receipt['views'][0]['section']['filename']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', action='append', help='Repeat to restrict to specific staged subjects')
    parser.add_argument('--graphs-only', action='store_true')
    args = parser.parse_args()
    data_root = checked_root(HDD_ROOT, expected_uuid=HDD_UUID)
    root = data_root/RUN
    studies = root/'studies'
    source_studies = data_root/SOURCE
    output = root/'app'/'cohort'
    output.mkdir(exist_ok=True)
    status_path = output/('graph-status.json' if args.graphs_only else 'status.json')
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    packaging_path = root/'cohort-packaging.json'
    packaged = json.loads(packaging_path.read_text()).get('cases', {}) if packaging_path.is_file() else {}
    staged = {path.name for path in studies.iterdir() if re.fullmatch(r'PTNT_\d{4}', path.name)}
    linked = {path.parent.name for path in source_studies.glob('PTNT_*/vascular-review.anatomy.json')}
    subjects = args.subject or sorted(staged|linked|set(packaged))
    desktop = Path(__file__).resolve().parents[1]/'desktop'
    runtime = desktop/'node_modules'/'electron'
    electron = runtime/'dist'/(runtime/'path.txt').read_text().strip()
    for subject in subjects:
        if not re.fullmatch(r'PTNT_\d{4}', subject):
            parser.error(f'Invalid EDEN subject {subject!r}')
        record = status.setdefault(subject, {})
        try:
            packaging = packaged.get(subject, {})
            published = packaging.get('manifest')
            manifest = safe_path(data_root, published['path']) if published else studies/subject/'patient-anatomy.json'
            if not manifest.is_file() and not published:
                manifest = source_studies/subject/'vascular-review.anatomy.json'
            if not manifest.is_file():
                record.update(state='awaiting native study', packaging_status=packaging.get('status'))
                continue
            if manifest.parent.name != subject or not re.fullmatch(r'patient-anatomy\.json|vascular-review(?:\.[a-f0-9]{12})?\.anatomy\.json', manifest.name):
                raise ValueError('Packaging receipt names an unexpected study path')
            if published and sha256_file(manifest) != published['sha256']:
                raise ValueError('Published vascular study manifest checksum differs')
            collection = PatientImagingCollection(manifest.parent, json.loads(manifest.read_text()))
            candidate = collection.layer(CANDIDATE)
            record['partition_status'] = collection.document['vascular_review']['status']
            try:
                record['graph'] = install_graph(collection, candidate, root/'cases'/subject/'source-branches.json')
                record.pop('graph_error', None)
            except (OSError, ValueError) as error:
                record['graph_error'] = str(error)
                if args.graphs_only:
                    raise
            previous = record.get('capture', {})
            evidence = [previous.get('report'), *previous.get('images', [])]
            complete = previous.get('snapshot_id') == collection.identity() and all(p and Path(p).is_file() for p in evidence)
            if not args.graphs_only and not complete:
                selected = collection.layer(collection.document['display']['initial_layers'][0])
                record['capture'] = capture_case(electron, desktop, manifest, collection, selected, output, subject)
            record['state'] = 'captured' if record.get('capture') else 'graph prepared' if record['graph'] else 'graph unavailable'
            record.pop('error', None)
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            record.update(state='failed', error=str(error))
        atomic_json(status_path, status)
        print(f"{subject} {record['state']}", flush=True)
    atomic_json(status_path, status)
    return int(any(status[subject]['state']=='failed' for subject in subjects))


if __name__ == '__main__':
    raise SystemExit(main())
