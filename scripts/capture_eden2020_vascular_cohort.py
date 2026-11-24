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
import time

import numpy as np

from build_eden2020_vascular_companion import CANDIDATE, HDD_ROOT, HDD_UUID, RUN, SOURCE
from resectionlab.anatomy_bridge import VASCULAR_REVIEW_REVISION, open_anatomy_manifest
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


def prepare_contour(collection, layer, subject):
    # Prime assets outside the bounded interactive request. A slow native score
    # read can finish once here and all subsequent views reuse its contour.
    surface = mesh(collection, layer, display_source=vascular_cleanup_source(collection, layer), conditioned=False)
    triangles, field = len(surface['indices']), str(surface.get('surface_field', ''))
    if not triangles or field not in {'native-probability', 'filtered-native-probability'}:
        raise ValueError(f'{layer["id"]} did not yield a validated native score contour')
    print(f'{subject} {layer["id"]}: {triangles:,} native triangles', flush=True)
    return field


def reviewed_manifest(data_root, published_path, packaging, override):
    """Select a checked revision beside the published companion for that subject."""
    if (packaging.get('status') != 'published' or not isinstance(override, dict)
            or set(override) != {'path', 'sha256'} or not isinstance(override['path'], str)
            or not isinstance(override['sha256'], str) or not re.fullmatch(r'[a-f0-9]{64}', override['sha256'])):
        raise ValueError('A reviewed override requires an existing publication and a path and checksum')
    path = safe_path(data_root, override['path'])
    if path.parent != published_path.parent or not VASCULAR_REVIEW_REVISION.fullmatch(path.name):
        raise ValueError('Reviewed override must be a linked revision beside the published companion')
    if sha256_file(path) != override['sha256']:
        raise ValueError('Reviewed vascular study manifest checksum differs')
    published = open_anatomy_manifest(published_path)
    original = published.layer(packaging['baseline_layer'])
    collection = open_anatomy_manifest(path)
    baseline = vascular_cleanup_source(collection, collection.layer(CANDIDATE))
    if (collection.document['patient_id'] != published.document['patient_id'] or baseline is None
            or baseline['id'] != original['id'] or baseline['affine'] != original['affine']
            or baseline['data'] != original['data'] or baseline['probability'] != original['probability']
            or collection.series(baseline['source_series'])['original'] != published.series(original['source_series'])['original']):
        raise ValueError('Reviewed override does not reuse the published subject, acquisition and score layer')
    return path, collection


def capture_case(electron, desktop, manifest, collection, selected, output, subject,
                 *, baseline=None, display_settings=None):
    scan = collection.series(selected['source_series'])
    frame = np.asarray(scan['affine'])
    shape = np.asarray(scan['data']['shape'])
    center = world(frame, (shape-1)/2)
    positive = np.argwhere(collection.mask(selected))
    if not len(positive):
        raise ValueError('Selected vessel mask is empty')
    surface_field = prepare_contour(collection, selected, subject)
    if baseline is not None:
        if baseline['source_series'] != selected['source_series'] or baseline['affine'] != selected['affine']:
            raise ValueError('Matched views require the same native acquisition and lattice')
        fields = {baseline['id']: prepare_contour(collection, baseline, subject), selected['id']: surface_field}
        layer_ids = [baseline['id'], selected['id']]
    else:
        fields = {selected['id']: surface_field}
        layer_ids = [selected['id']]
    points = world(frame, positive)
    cursor = np.median(points, axis=0)
    cursor[2] = np.percentile(points[:, 2], 30)
    recipe = [{
        'name': 'native-tof-superior', 'seriesId': scan['id'], 'layers': layer_ids,
        'cursor': cursor.tolist(), 'sourceSection': True,
        'settings': {**(display_settings or {}), 'activePlane': 'axial', 'camera': {
            'position': (center+[0., -.47, 470.]).tolist(), 'target': center.tolist(), 'up': [0., 1., 0.],
        }},
        **({'compareLayerIds': layer_ids} if baseline is not None else {}),
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
    view = receipt['views'][0]
    submitted = {item['name']: item['surfaceDiagnostics'] for item in view['surfaces']}
    for layer_id, field in fields.items():
        diagnostics = submitted.get(layer_id, {})
        if not diagnostics.get('sourceSurfaceOnly') or diagnostics.get('surfaceField') != field:
            raise ValueError('Hidden application submitted a different vessel representation')
    if baseline is not None:
        if (view.get('comparison', {}).get('layer') != selected['id'] or not view.get('baselineSection')
                or view['view']['visible'].get(selected['id']) is not False):
            raise ValueError('Hidden application did not record a distinct baseline and selected layer')
        baseline_images = [view['filename'], view['baselineSection']['filename']]
        selected_images = [view['comparison']['filename'], view['section']['filename']]
    else:
        baseline_images = []
        selected_images = [view['filename'], view['section']['filename']]
    if not all(Path(path).is_file() for path in [*baseline_images, *selected_images]):
        raise ValueError('Hidden application capture evidence is incomplete')
    return {'report': str(report), 'snapshot_id': receipt['snapshotId'], 'surface_field': surface_field,
            'layer': selected['id'], 'images': selected_images,
            'baseline_layer': baseline['id'] if baseline is not None else None,
            'baseline_images': baseline_images, 'display_settings': display_settings or {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', action='append', help='Repeat to restrict to specific staged subjects')
    parser.add_argument('--graphs-only', action='store_true')
    parser.add_argument('--follow-minutes', type=int, default=0, help='Wait for publication of pending studies within this window')
    parser.add_argument('--match-baseline', action='store_true', help='Capture the original and selected vessel layers at the same camera')
    parser.add_argument('--output', type=Path, help='Capture directory relative to the verified HDD root')
    parser.add_argument('--display-settings', type=Path, help='Shared viewer settings JSON for the matched comparison')
    parser.add_argument('--reviewed-manifests', type=Path, help='Per-subject reviewed manifest references, relative to the verified HDD root')
    args = parser.parse_args()
    if not 0<=args.follow_minutes<=720:
        parser.error('Follow minutes must be between 0 and 720')
    data_root = checked_root(HDD_ROOT, expected_uuid=HDD_UUID)
    root = data_root/RUN
    studies = root/'studies'
    source_studies = data_root/SOURCE
    output = safe_path(data_root, args.output) if args.output else root/'app'/('cohort-matched' if args.match_baseline else 'cohort')
    output.mkdir(parents=True, exist_ok=True)
    display_settings = json.loads(args.display_settings.read_text()) if args.display_settings else {}
    if not isinstance(display_settings, dict):
        parser.error('Display settings must be a JSON object')
    reviewed = json.loads(safe_path(data_root, args.reviewed_manifests).read_text()) if args.reviewed_manifests else {}
    if not isinstance(reviewed, dict) or any(not re.fullmatch(r'PTNT_\d{4}', subject) for subject in reviewed):
        parser.error('Reviewed manifests must be a JSON object keyed by restricted subject')
    status_path = output/('graph-status.json' if args.graphs_only else 'status.json')
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    packaging_path = root/'cohort-packaging.json'
    desktop = Path(__file__).resolve().parents[1]/'desktop'
    runtime = desktop/'node_modules'/'electron'
    electron = runtime/'dist'/(runtime/'path.txt').read_text().strip()
    deadline = time.monotonic()+60*args.follow_minutes
    attempted = set()
    while True:
        packaged = json.loads(packaging_path.read_text()).get('cases', {}) if packaging_path.is_file() else {}
        staged = {path.name for path in studies.iterdir() if re.fullmatch(r'PTNT_\d{4}', path.name)}
        linked = {path.parent.name for path in source_studies.glob('PTNT_*/vascular-review.anatomy.json')}
        subjects = args.subject or sorted(staged|linked|set(packaged))
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
                    publication = packaging.get('status')
                    state = publication if publication in {'missing_input','failed','review_required'} else 'awaiting native study'
                    record.update(state=state, packaging_status=publication, reason=packaging.get('reason'))
                    continue
                direct_study = manifest.name == 'patient-anatomy.json' and manifest.parent.parent == studies
                standalone = direct_study and (
                    manifest.parent.name == subject
                    or re.fullmatch(rf'{subject}-reviewed-[a-f0-9]{{12}}', manifest.parent.name)
                )
                linked_study = (
                    manifest.parent == source_studies/subject
                    and re.fullmatch(r'vascular-review(?:\.[a-f0-9]{12})?\.anatomy\.json', manifest.name)
                )
                if not (standalone or linked_study):
                    raise ValueError('Packaging receipt names an unexpected study path')
                attempt = (manifest, published['sha256'] if published else manifest.stat().st_mtime_ns)
                if attempt in attempted:
                    continue
                attempted.add(attempt)
                if published and sha256_file(manifest) != published['sha256']:
                    raise ValueError('Published vascular study manifest checksum differs')
                if subject in reviewed:
                    if not published:
                        raise ValueError('Reviewed override requires a published vascular companion')
                    manifest, collection = reviewed_manifest(data_root, manifest, packaging, reviewed[subject])
                else:
                    collection = open_anatomy_manifest(manifest)
                candidate = collection.layer(CANDIDATE)
                record['partition_status'] = collection.document['vascular_review']['status']
                try:
                    record['graph'] = install_graph(collection, candidate, root/'cases'/subject/'source-branches.json')
                    record.pop('graph_error', None)
                except (OSError, ValueError) as error:
                    record['graph_error'] = str(error)
                    if args.graphs_only:
                        raise
                if not args.graphs_only:
                    selected = collection.layer(collection.document['display']['initial_layers'][0])
                    baseline = vascular_cleanup_source(collection, candidate) if args.match_baseline else None
                    if args.match_baseline and baseline is None:
                        raise ValueError('Original baseline membership cannot be resolved for a matched capture')
                    if baseline is not None and baseline['id'] == selected['id']:
                        selected = candidate
                    previous = record.get('capture', {})
                    evidence = [previous.get('report'), *previous.get('images', []), *previous.get('baseline_images', [])]
                    complete = (previous.get('snapshot_id') == collection.identity()
                                and previous.get('layer') == selected['id']
                                and previous.get('baseline_layer') == (baseline['id'] if baseline is not None else None)
                                and previous.get('display_settings', {}) == display_settings
                                and all(path and Path(path).is_file() for path in evidence))
                    if not complete:
                        record['capture'] = capture_case(electron, desktop, manifest, collection, selected, output, subject,
                                                         baseline=baseline, display_settings=display_settings)
                record['state'] = 'captured' if record.get('capture') else 'graph prepared' if record['graph'] else 'graph unavailable'
                record.pop('error', None)
            except (OSError, ValueError, KeyError, StopIteration, subprocess.TimeoutExpired) as error:
                record.update(state='failed', error=str(error))
            atomic_json(status_path, status)
            print(f"{subject} {record['state']}", flush=True)
        atomic_json(status_path, status)
        waiting = not subjects or any(status[subject]['state']=='awaiting native study' for subject in subjects)
        remaining = deadline-time.monotonic()
        if not waiting or remaining<=0:
            break
        time.sleep(min(30, remaining))
    return int(any(status[subject]['state']=='failed' for subject in subjects))


if __name__ == '__main__':
    raise SystemExit(main())
