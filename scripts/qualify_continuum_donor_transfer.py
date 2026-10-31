#!/usr/bin/env python3
"""Prospectively frozen donor transfer; never run fitting after holdout reveal.

prepare reads only the historical exposed development receipt. acquire verifies
the source archive as opaque bytes. score opens exactly two declared members
after a committed numerical/parameter freeze. All datasets remain in testcases.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.request
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab import surgical_continuum as continuum
from resectionlab import surgical_deformation_v2 as legacy
from qualify_continuum_mechanics import prescribed_solve, source, material

PROTOCOL = ROOT / 'manifests/experiments/continuum-donor-transfer-v1.json'
FREEZE = ROOT / 'manifests/experiments/continuum-donor-transfer-v1-freeze.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def checked_protocol():
    p = json.loads(PROTOCOL.read_text())
    for binding in [p['source']['metadata'], p['development']['source'],
                    p['continuum']['source'], p['legacy']['source']]:
        if sha(ROOT / binding['path']) != binding['sha256']:
            raise ValueError('Frozen source changed: ' + binding['path'])
    return p


def local_output(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT / 'testcases') or path.exists():
        raise ValueError('Choose a new output under testcases/')
    return path


def fixture(radius, height, xy_count, z_count, output):
    spacing = 2 * radius / xy_count
    axis = (np.arange(xy_count) + .5) * spacing - radius
    labels = np.broadcast_to((axis[:, None]**2 + axis[None, :]**2 <= radius**2)[:, :, None],
                             (xy_count, xy_count, z_count)).astype(np.uint16).copy()
    affine = np.diag([spacing, spacing, height / z_count, 1.])
    affine[:3, 3] = [axis[0], axis[0], height / z_count / 2.]
    case = source(labels, affine)
    case.properties[continuum.PROPERTY]['materials'] = {'1': material(1.)}
    cells = list(map(int, np.flatnonzero(labels)))
    mesh = continuum.voxel_mesh(case, cells)
    matrices, gradients = continuum.assemble(case, mesh, case.properties[continuum.PROPERTY]['materials'])
    matrix = matrices[0]
    bottom = np.isclose(mesh.nodes[:, 2], 0.)
    top = np.isclose(mesh.nodes[:, 2], height)
    imposed = {3 * int(i) + a: (1. if top[i] and a == 2 else 0.)
               for i in np.flatnonzero(top | bottom) for a in range(3)}
    u, reaction = prescribed_solve(matrix, imposed)
    slope = float(reaction.reshape((-1, 3))[top, 2].sum())
    free = np.ones(len(u), bool); free[list(imposed)] = False
    max_gradient = float(np.linalg.norm(np.einsum('eai,gaj->egij',
                         u.reshape((-1, 3))[mesh.element_nodes], gradients), axis=(2, 3)).max())
    continuum_residual = float(np.max(np.abs(reaction[free]), initial=0.))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, affine=affine, labels=labels, nodes=mesh.nodes,
                        elements=mesh.element_nodes, displacement_per_mm=u.reshape((-1, 3)),
                        reaction_per_mm_per_pa=reaction.reshape((-1, 3)))
    result = {'xyCount': xy_count, 'zCount': z_count, 'spacingXyMm': spacing,
              'spacingZMm': height / z_count, 'sourceElements': len(cells), 'dofs': len(u),
              'nominalVolumeMm3': float(np.pi * radius**2 * height),
              'voxelVolumeMm3': len(cells) * case.voxel_mm3,
              'continuumSlopeNPerMmPerPa': slope,
              'continuumFreeResidualNPerMmPerPa': continuum_residual,
              'gradientNormPerMm': max_gradient,
              'basis': {'path': str(output.relative_to(ROOT)), 'sha256': sha(output)}}
    del matrices, matrix, mesh, u, reaction
    gc.collect()
    old = SimpleNamespace(scene=case, state={'holds': {}}, by_label={1: {'material': 'soft_tissue'}})
    owned = set(cells)
    bonds = sorted((c, j) for c in cells for j, _ in legacy._neighbors(old, c) if c < j and j in owned)
    matrix = legacy._system(old, legacy.profile(), cells, [], bonds)[-1]
    centers = case.points(cells)
    bottom = np.isclose(centers[:, 2], centers[:, 2].min())
    top = np.isclose(centers[:, 2], centers[:, 2].max())
    imposed = {3 * int(i) + a: (1. if top[i] and a == 2 else 0.)
               for i in np.flatnonzero(top | bottom) for a in range(3)}
    u, reaction = prescribed_solve(matrix, imposed)
    legacy_slope = float(reaction.reshape((-1, 3))[top, 2].sum())
    span = float(np.ptp(centers[:, 2]))
    result.update(legacySlopeNPerMm=legacy_slope, legacyCenterSpanMm=span,
                  legacyMatchedStrainSlopeNPerMm=legacy_slope * span / height)
    return result


def prepare(output):
    p = checked_protocol(); output = local_output(output); output.mkdir(parents=True)
    started = time.monotonic()
    measured = json.loads((ROOT / p['development']['source']['path']).read_text())
    if measured['held_out_values_used'] is not False:
        raise ValueError('Development receipt role changed')
    d = p['development']; s = p['selected']['geometry_m']
    fixtures = {}
    for name, radius, height in [('development', d['radiusMm'], d['heightMm']),
                                 ('evaluation', s['radius'] * 1000., s['height'] * 1000.)]:
        fixtures[name] = []
        for m in p['meshes']:
            row = fixture(radius, height, m['xyCount'], m['zCount'],
                          output / f'{name}-{m["xyCount"]}.npz')
            fixtures[name].append(row)
            print(json.dumps({'specimen': name, **{k: row[k] for k in
                  ['xyCount', 'sourceElements', 'dofs', 'continuumSlopeNPerMmPerPa']}}), flush=True)
    numerator = denominator = 0.; development_points = {}
    slope = fixtures['development'][-1]['continuumSlopeNPerMmPerPa']
    for name, values in measured['modes'].items():
        x = np.asarray(values['coordinate']) * 1000.; y = np.asarray(values['observed'])
        mask = np.abs(x / d['heightMm']) <= p['strainWindow']
        x, y = x[mask], y[mask]
        if len(x) < 3: raise ValueError('Insufficient development points in frozen strain window')
        basis = x * slope
        numerator += float(basis @ y); denominator += float(basis @ basis)
        development_points[name] = len(x)
    mu = numerator / denominator
    gates = {}; limits = p['admissionBeforeHeldOutRead']
    prior = None
    if 'priorFailedAttempt' in p:
        binding = p['priorFailedAttempt']
        prior_path = ROOT / binding['preparation']
        if sha(prior_path) != binding['preparationSha256']:
            raise ValueError('Prior numerical failure evidence changed')
        prior = json.loads(prior_path.read_text())
    for name, rows in fixtures.items():
        height = d['heightMm'] if name == 'development' else s['height'] * 1000.
        change = abs(rows[-1]['continuumSlopeNPerMmPerPa'] / rows[-2]['continuumSlopeNPerMmPerPa'] - 1.)
        for row in rows:
            row['maximumGradientAtWindow'] = row['gradientNormPerMm'] * height * p['strainWindow']
            row['freeResidualAtWindowN'] = row['continuumFreeResidualNPerMmPerPa'] * mu * height * p['strainWindow']
            row['continuumSlopeNPerMm'] = row['continuumSlopeNPerMmPerPa'] * mu
        gates[name] = {'finestReactionRelativeChange': change,
                      'finestVolumeRelativeError': abs(rows[-1]['voxelVolumeMm3'] / rows[-1]['nominalVolumeMm3'] - 1.),
                      'passed': change <= limits['finestReactionSlopeRelativeChange']
                      and abs(rows[-1]['voxelVolumeMm3'] / rows[-1]['nominalVolumeMm3'] - 1.) <= limits['finestVolumeRelativeError']
                      and all(r['maximumGradientAtWindow'] <= limits['maximumElementGradient']
                              and r['freeResidualAtWindowN'] <= limits['freeResidualN'] for r in rows)}
        if prior is not None:
            slopes = [r['continuumSlopeNPerMmPerPa'] for r in rows[-2:]]
            slopes.append(prior['fixtures'][name][-1]['continuumSlopeNPerMmPerPa'])
            envelope = (max(slopes) - min(slopes)) / rows[-1]['continuumSlopeNPerMmPerPa']
            gates[name]['lastTwoPlusPriorFineSlopeEnvelope'] = envelope
            gates[name]['passed'] &= envelope <= limits['lastTwoPlusPriorFineSlopeEnvelope']
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin': rss *= 1024
    result = {'schema': 'resectionlab.continuum-donor-transfer-preparation.v1',
              'protocolSha256': sha(PROTOCOL), 'scriptSha256': sha(Path(__file__)),
              'helperSha256': sha(ROOT / 'scripts/qualify_continuum_mechanics.py'),
              'fit': {'muPa': mu, 'poissonRatio': .45, 'numerator': numerator, 'denominator': denominator,
                      'developmentPoints': development_points, 'parametersFitted': ['positive_shear_scale_only']},
              'fixtures': fixtures, 'numericalGates': gates, 'heldOutResponseMembersOpened': [],
              'wallSeconds': time.monotonic() - started, 'peakRssBytes': rss,
              'engineRevision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    result['admittedForScoring'] = bool(np.isfinite(mu) and mu > 0 and all(g['passed'] for g in gates.values())
        and result['wallSeconds'] <= p['caps']['prepareWallSeconds'] and rss <= p['caps']['peakRssBytes'])
    write(output / 'preparation.json', result)
    print(json.dumps({k: result[k] for k in ['fit', 'numericalGates', 'admittedForScoring', 'wallSeconds', 'peakRssBytes']}))
    return result['admittedForScoring']


def acquire(output):
    p = checked_protocol(); output = local_output(output)
    # Check by exact size first; compressed source names may differ or be symlinks.
    candidates = []
    for here, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {'.git', '.venv', 'node_modules', '__pycache__'}]
        for name in files:
            path = Path(here) / name
            if path.is_file() and path.stat().st_size == p['source']['archiveBytes']:
                candidates.append(path)
    matches = [q for q in candidates if sha(q) == p['source']['archiveSha256']]
    if matches:
        print(json.dumps({'reusedExisting': str(matches[0]), 'downloadedBytes': 0})); return
    request = urllib.request.Request(p['source']['archiveUrl'])
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(p['source']['archiveBytes'] + 1)
    if (len(data) != p['source']['archiveBytes'] or hashlib.md5(data).hexdigest() != p['source']['archiveMd5']
            or hashlib.sha256(data).hexdigest() != p['source']['archiveSha256']):
        raise ValueError('Provider archive byte identity mismatch')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream: stream.write(data)
    write(output.with_suffix('.acquisition.json'), {'url': p['source']['archiveUrl'],
          'bytes': len(data), 'sha256': sha(output), 'duplicateCandidates': [str(q.relative_to(ROOT)) for q in candidates],
          'responseMembersOpened': [], 'license': p['source']['license'], 'citation': p['source']['doi']})
    print(json.dumps({'path': str(output), 'downloadedBytes': len(data), 'responseMembersOpened': []}))


def score(archive, preparation, output):
    p = checked_protocol(); output = local_output(output)
    freeze = json.loads(FREEZE.read_text())
    # This prospective freeze must exist in the current commit, byte for byte.
    if subprocess.check_output(['git', 'show', 'HEAD:' + str(FREEZE.relative_to(ROOT))], cwd=ROOT) != FREEZE.read_bytes():
        raise ValueError('Commit the parameter/prediction freeze before opening held-out members')
    report = json.loads(preparation.read_text())
    if (freeze['preparationSha256'] != sha(preparation) or freeze['protocolSha256'] != sha(PROTOCOL)
            or report['scriptSha256'] != sha(Path(__file__))
            or report['helperSha256'] != sha(ROOT / 'scripts/qualify_continuum_mechanics.py')
            or report['admittedForScoring'] is not True or sha(archive) != p['source']['archiveSha256']):
        raise ValueError('Numerical, source, script or freeze binding mismatch')
    if freeze['muPa'] != report['fit']['muPa']:
        raise ValueError('Fitted material changed')
    rows = report['fixtures']['evaluation']; height = p['selected']['geometry_m']['height'] * 1000.
    results = {}; accesses = []
    ledger = output.with_suffix('.access.jsonl')
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open('x') as stream:
        stream.write(json.dumps({'event': 'intent', 'freezeSha256': sha(FREEZE),
                    'authorizedMembers': [m['name'] for m in p['authorizedNewCurveMembers']]}) + '\n')
    # Claim remains descriptive; no acceptance threshold is fabricated from these values.
    with zipfile.ZipFile(archive) as bundle:
        for member in p['authorizedNewCurveMembers']:
            name = member['name']
            infos = [i for i in bundle.infolist() if i.filename == name]
            if len(infos) != 1 or infos[0].file_size != member['bytes'] or f'{infos[0].CRC:08x}' != member['crc32']:
                raise ValueError('Declared unique response member identity mismatch')
            with ledger.open('a') as stream:
                stream.write(json.dumps({'event': 'read-start', 'member': name}) + '\n')
            data = bundle.read(name)
            accesses.append({'member': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            with ledger.open('a') as stream:
                stream.write(json.dumps({'event': 'read-complete', **accesses[-1]}) + '\n')
            records = list(csv.reader(io.StringIO(data.decode('utf-8')), delimiter=p['csvSchema']['delimiter']))
            if not records or records.pop(0) != p['csvSchema']['header'] or any(len(r) != 2 for r in records):
                raise ValueError('Unexpected held-out CSV schema; do not substitute a specimen')
            values = np.asarray(records, dtype=float)
            if not np.isfinite(values).all(): raise ValueError('Nonfinite measured values')
            x = values[:, 0] * 1000.; y = values[:, 1]
            chosen = np.abs(x / height) <= p['strainWindow']; x, y = x[chosen], y[chosen]
            if len(x) < 3: raise ValueError('Insufficient frozen-window reference points')
            endpoint = int(np.argmax(np.abs(x))); predictions = []
            for row in rows:
                models = {'continuum': x * row['continuumSlopeNPerMm'],
                          'legacyDefault': x * row['legacySlopeNPerMm'],
                          'legacyMatchedStrain': x * row['legacyMatchedStrainSlopeNPerMm'], 'zeroForce': np.zeros_like(x)}
                metrics = {m: {'forceN': f.tolist(), 'forceRmseN': float(np.sqrt(np.mean((f-y)**2))),
                              'maximumAbsoluteErrorN': float(np.max(np.abs(f-y))),
                              'signedEndpointErrorN': float(f[endpoint]-y[endpoint])} for m, f in models.items()}
                predictions.append({'xyCount': row['xyCount'], 'models': metrics})
            results[Path(name).stem] = {'admittedPoints': len(x), 'totalSourceRows': len(values),
                'displacementMm': x.tolist(), 'observedForceN': y.tolist(), 'predictions': predictions}
    write(output, {'schema': 'resectionlab.continuum-donor-transfer-result.v1',
          'protocolSha256': sha(PROTOCOL), 'freezeSha256': sha(FREEZE), 'preparationSha256': sha(preparation),
          'archiveSha256': sha(archive), 'newResponseAccesses': accesses, 'results': results,
          'independentUnits': 1, 'fitAfterReveal': False, 'physicalValidationPassed': False,
          'patientMechanicsValidated': False, 'largeScaleRL': 'hold'})
    print(json.dumps({'output': str(output), 'newResponseMembersOpened': len(accesses),
          'finestRmseN': {n: {m: v['forceRmseN'] for m, v in z['predictions'][-1]['models'].items()} for n, z in results.items()}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'acquire', 'score'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--preparation', type=Path)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--freeze', type=Path, default=FREEZE)
    args = parser.parse_args()
    PROTOCOL = args.protocol.resolve(); FREEZE = args.freeze.resolve()
    if args.phase == 'prepare':
        if not prepare(args.output): raise SystemExit(2)
    elif args.phase == 'acquire': acquire(args.output)
    elif args.archive and args.preparation: score(args.archive, args.preparation, args.output)
    else: parser.error('score requires --archive and --preparation')
