#!/usr/bin/env python3
"""Freeze, run and resume a matched generated post-access baseline comparison."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

from resectionlab.core import semantic_digest
from resectionlab.geometry import GeometryScene, capsule_voxel_indices
from resectionlab.native_backend import backend_identity
from resectionlab.surgical_preparation_variation import variation_task
from resectionlab.surgical_procedure_policy import evaluate_procedure_sequence
from resectionlab.surgical_sequential_benchmark import (
    ComputeTier, ReplayVerificationError, feature_matrix, run_episode, verify_replay,
)
from resectionlab.surgical_sequential_task import COMMON_CONFIG, generated_task, training_admission


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_SECONDS = 3600
TIERS = (
    ComputeTier('low', 64, 45., 2, 2, 12),
    ComputeTier('high', 128, 90., 3, 3, 24),
)
CASES = (
    ('gated-work', 200, False, None),
    ('critical-residual', 201, False, None),
    ('permitted-overlap', 202, False, None),
    ('off-axis-wide-start', 4000, True, None),
    ('off-axis-narrow-prefix', 4000, True, .4),
)
METHODS = ('fixed-narrow-greedy', 'conditional', 'greedy', 'search')


def conditional_controller(observed, skills, mask):
    """Use supplied geometry and remaining work to choose a tip or checked STOP."""
    features = feature_matrix(observed, skills)
    work = [i for i, row in enumerate(features) if mask[i] and row[1] and row[9] > 0 and row[8] == 0]
    wide = features[0, 3] > .4 / 1.5 + 1e-6
    remaining = observed['source_labels'] == 1
    remaining.ravel()[observed['native_state']['removed']] = False
    target = GeometryScene(remaining, observed['affine'])

    def contacts(index, radius):
        cells = capsule_voxel_indices(target, *skills[index].route, radius)
        return int(np.count_nonzero(remaining[tuple(cells.T)]))

    def reachable(radius):
        return {i: contacts(i, radius)
                for i in work if radius == .4 or features[i, 10] * 30 >= 3 - 1e-6}

    scores = reachable(1.5 if wide else .4)
    if any(scores.values()):
        return max(scores, key=scores.get)
    radius = .4 if wide else 1.5
    if any(reachable(radius).values()):
        prep = next((i for i, row in enumerate(features)
                     if mask[i] and row[0] and abs(row[4] * 1.5 - radius) < 1e-6), None)
        if prep is not None:
            return prep
    return next(i for i, row in enumerate(features) if mask[i] and row[2])


def _read(path):
    return json.loads(path.read_text())


def _save(path, value):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    pending.replace(path)


def _file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def freeze(profile):
    backend = backend_identity()
    identities = {key: backend[key] for key in ('selected', 'binarySha256', 'nativeThreads', 'fallbackReason')}
    package = ROOT / 'src/resectionlab'
    sources = sorted(package.glob('surgical_*.py'))
    sources += [package / f'{name}.py' for name in ('core', 'geometry', 'native_backend')]
    sources += [Path(__file__).resolve()]
    bindings = {str(path.relative_to(ROOT)): _file_sha(path) for path in sources}
    audits = []
    for label, parent, variant, initial in CASES:
        task = variation_task(parent) if variant else generated_task(parent)
        audits.append(dict(id=label, group=f'generated-parent-{parent}', role='DEVELOPMENT',
                           initialPrepareRadiusMm=initial, source=task.audit,
                           admission=training_admission(task)))
    declaration = dict(schema='resectionlab.approach-procedure-baselines.v1',
                       campaignSeconds=CAMPAIGN_SECONDS, tiers=[asdict(tier) for tier in TIERS],
                       methods=list(METHODS), cases=audits, sourceHashes=bindings,
                       config=asdict(COMMON_CONFIG), seed=0,
                       profileSha256=_file_sha(profile), python=platform.python_version(),
                       platform=platform.platform(), nativeBackend=identities,
                       observation='Same supplied generated anatomy, finite inventory, current state and source guards. '
                                   'Greedy and search additionally pay for engine previews from that state. '
                                   'No source case ID, split, other runs, or evaluator-only anatomy is a selector input.',
                       evaluator='Unchanged procedure policy, full original target and cumulative harm. '
                                 'Physical requests include previews, execution and exit motion. '
                                 'Logical STOP and verification are accounted separately.',
                       learningQuestion='Fewer engine evaluations at comparable separated outcomes.',
                       learningFrozen=False)
    return {**declaration, 'identity': semantic_digest(declaration)}


def _execute(case, method, tier, campaign_deadline):
    label, parent, variant, initial_radius = case
    task = variation_task(parent) if variant else generated_task(parent)
    deadline = min(campaign_deadline, time.monotonic() + tier.max_wall_seconds)
    policy = 'greedy' if method == 'fixed-narrow-greedy' else method
    if method == 'fixed-narrow-greedy':
        initial_radius = .4
    row = run_episode(task, policy, tier, retain_engine=True, deadline=deadline,
                      selector=conditional_controller if method == 'conditional' else None,
                      initial_prepare_radius=initial_radius)
    engine = row.pop('_engine')
    row.update(method=method, case=label, voxelMm3=engine.scene.voxel_mm3,
               sequence=evaluate_procedure_sequence(engine)['outcome'])
    try:
        row['verification'] = verify_replay(task, row, deadline=campaign_deadline)
    except ReplayVerificationError as error:
        row['verification'] = error.receipt
    return row


def _summary(row):
    endpoint = row['endpoint']
    computation = row['computation']
    dimensions = row['sequence']['dimensions']
    harm = {name: dimensions[name]['cumulative'] for name in (
        'boneRemovedMm3', 'nonbonePreparationRemovedMm3', 'neuralRemovedMm3',
        'neuralCutSourceMm3', 'protectedRemovedMm3',
    )}
    for group in ('structureRemovedMm3', 'structureCutSourceMm3'):
        harm.update({f'{group}:{name}': value['cumulative'] for name, value in dimensions[group].items()})
    previews = (event for preview in computation['previews'] for event in preview['receipt']['receipts'])
    return dict(case=row['case'], tier=row['tier'], method=row['method'],
                originalMm3=row['progress']['originalTargetCells'] * row['voxelMm3'],
                treatmentMm3=row['progress']['originalTargetTreatmentMm3'],
                residualMm3=row['progress']['residualOriginalTargetCells'] * row['voxelMm3'], harm=harm,
                checkedStop=any(skill['checkedStopAttempted'] for skill in row['skillHistory']),
                previewStops=sum(item['receipt']['checkedStopAttempted'] for item in computation['previews']),
                wholeToolExit=endpoint['validWithdrawal'], endpoint=endpoint['classification'],
                previewRefusals=sum(event['outcome']['status'] == 'rejected' for event in previews),
                actualRefusals=row['rejections'], physicalRequests=computation['physicalCalls'],
                previewRequests=computation['planningPhysicalCalls'],
                executionRequests=computation['executionPhysicalCalls'], canceledRequests=computation['canceledRequests'],
                seconds=computation['wallSeconds'] + row['resetSeconds'], modeledSeconds=row['modeledSeconds'],
                replayRequests=row['verification']['physicalCalls'], replaySeconds=row['verification']['wallSeconds'],
                replayVerified=row['verification']['verified'],
                exchanges=[event['action']['parameters']['radiusMm'] for skill in row['skillHistory']
                           for event in skill['receipts'] if event['action']['operation'] == 'exchange'
                           and event['outcome']['status'] == 'accepted'])


def _comparable(candidate, reference):
    return (candidate['case'] == reference['case'] and candidate['wholeToolExit'] and candidate['checkedStop']
            and candidate['replayVerified'] and candidate['endpoint'] in ('valid-stop-with-residual', 'successful-treatment')
            and candidate['treatmentMm3'] >= reference['treatmentMm3']
            and (reference['endpoint'] != 'successful-treatment' or candidate['endpoint'] == 'successful-treatment')
            and all(candidate['harm'][key] <= cost for key, cost in reference['harm'].items()))


def run(out, profile, row_limit):
    protocol = freeze(profile)
    protocol_path = out / 'protocol.json'
    if protocol_path.exists():
        saved = _read(protocol_path)
        saved.pop('sourceCommitAtFreeze')
        if saved != protocol:
            raise ValueError('Source, profile, evaluator or frozen campaign settings changed. Preserve this campaign.')
    else:
        _save(protocol_path, {**protocol, 'sourceCommitAtFreeze': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    state_file = out / 'progress.json'
    state = _read(state_file) if state_file.exists() else dict(spentSeconds=0., completed={}, interruptions={})
    active = state.pop('active', None)
    if active:
        key = active['key']
        # The previous writer was interrupted. Its reservation is spent; a
        # result written atomically before the interruption is still reusable.
        state['spentSeconds'] += active['secondsReserved']
        if (out / f'{key}.json').exists():
            state['completed'][key] = _file_sha(out / f'{key}.json')
        else:
            state['interruptions'][key] = dict(reason='process interrupted before row committed', **active)
        _save(state_file, state)
    ran = 0
    for tier in TIERS:
        for case in CASES:
            for method in METHODS:
                key = f'{tier.name}-{case[0]}-{method}'
                if key in state['completed'] or key in state['interruptions']:
                    continue
                remaining = CAMPAIGN_SECONDS - state['spentSeconds']
                if remaining <= 0 or ran == row_limit:
                    break
                started = time.monotonic()
                reservation = min(remaining, tier.max_wall_seconds + 60.)
                state['active'] = dict(key=key, secondsReserved=reservation,
                                       startedUtc=datetime.now(timezone.utc).isoformat())
                _save(state_file, state)
                path = out / f'{key}.json'
                row = _execute(case, method, tier, started + reservation)
                _save(path, row)
                state['completed'][key] = _file_sha(path)
                state['spentSeconds'] += time.monotonic() - started
                state.pop('active')
                _save(state_file, state)
                print(json.dumps(_summary(row), sort_keys=True), flush=True)
                ran += 1
    rows = []
    for key, digest in state['completed'].items():
        path = out / f'{key}.json'
        if _file_sha(path) != digest:
            raise ValueError(f'Completed row changed: {key}')
        rows.append(_summary(_read(path)))
    comparability = []
    for reference in rows:
        if reference['tier'] == 'high' and reference['method'] == 'search' and _comparable(reference, reference):
            comparability.append(dict(case=reference['case'], reference=reference,
                                      matched=[row for row in rows if _comparable(row, reference)]))
    report = dict(protocolIdentity=protocol['identity'], rows=rows, qualityAtMatchedBudgets=True,
                  costsForComparableQuality=comparability, interruptions=state['interruptions'],
                  missingRows=len(TIERS) * len(CASES) * len(METHODS) - len(rows),
                  campaignSeconds=state['spentSeconds'], optimizerUpdates=0,
                  peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    _save(out / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--rows', type=int, help='Stop after this many new rows and preserve campaign budget')
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / 'testcases'):
        parser.error('Keep evidence in ignored testcases/')
    if args.rows is not None and args.rows < 1:
        parser.error('--rows must be positive')
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.lock').open('w') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run(out, args.profile, args.rows)
    return int((result['missingRows'] > 0 and args.rows is None)
               or any(not row['replayVerified'] for row in result['rows']))


if __name__ == '__main__':
    sys.exit(main())
