#!/usr/bin/env python3
"""Compare existing acquired-source programs; no search, fitting or training."""
from pathlib import Path
import argparse
import copy
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.core import array_digest, semantic_digest
from resectionlab.imaging import save_case
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_scene import SurgicalScene, from_case
from resectionlab.surgical_environment import standardize
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.simulator_qualification import file_sha256
from resectionlab.surgical_consequences import consequence_vector, seal_report, validate_saved_comparison

PROTOCOL = ROOT / 'manifests/experiments/consequence-comparison-v1.json'


def prepare(output):
    output = output.resolve()
    if not output.is_relative_to(ROOT / 'testcases') or output.exists():
        raise ValueError('Use a new output under testcases/')
    protocol = json.loads(PROTOCOL.read_text()); source = ROOT / protocol['sourceArchive']
    if file_sha256(source) != protocol['sourceArchiveSha256']:
        raise ValueError('Source archive differs from frozen development input')
    output.mkdir(parents=True); started = time.perf_counter()
    case, artifacts, _ = load_workspace(source); record = copy.deepcopy(artifacts['surgical_toolbox'])
    original = standardize(from_case(case, record['mapping']), physics_profile=record['physicsProfile'])
    original.require_cache_source()
    source_hashes = (array_digest(case.mri), array_digest(original.labels))
    base_snapshot = record['branches'][protocol['prefixBranch']]; prefix = base_snapshot['actions']
    assert len(prefix) == protocol['prefixActions']
    for branch in protocol['alternativeBranches']:
        if record['branches'][branch]['actions'][:len(prefix)] != prefix:
            raise ValueError('Alternative does not share exact declared prefix')
    samples = []; scenario_records = []; receipts = []

    def check():
        if time.perf_counter() - started > protocol['caps']['wallSeconds']:
            raise TimeoutError('Consequence qualification wall budget exceeded')

    for scenario in protocol['scenarios']:
        properties = copy.deepcopy(original.properties)
        continuum = properties['brainContinuumV1']
        continuum['relaxationSeconds'] = [t * scenario['relaxationTimeScale'] for t in continuum['relaxationSeconds']]
        scene = original if scenario['id'] == 'nominal' else SurgicalScene(original.labels, original.affine,
            original.regions, original.source, properties)
        base = SurgicalEngine(scene)
        for action in prefix:
            result = base.step(action, check=check)
            if result['status'] == 'rejected': raise ValueError('Counterfactual prefix refused: ' + result['reason'])
        if scenario['id'] == 'nominal' and base.snapshot() != base_snapshot:
            raise ValueError('Nominal source prefix differs from retained recording')
        scenario_records.append({**scenario, 'sourceSceneId': scene.identity,
            'prefixStateId': base.state_id, 'profileHash': semantic_digest(scene.properties['brainContinuumV1']),
            'interpretation': protocol['uncertaintyMeaning']})
        for branch in protocol['alternativeBranches']:
            snapshot = record['branches'][branch]
            if len(snapshot['actions']) > protocol['caps']['maximumNativeActionsPerPlan']:
                raise ValueError('Native action budget exceeded')
            candidate = base.clone(); replay = base.clone(); trace = []
            for action in snapshot['actions'][len(prefix):]:
                check(); result = candidate.step(action, check=check)
                repeated = replay.step(action, check=check)
                if result != repeated or candidate.state != replay.state:
                    raise ValueError('Separate native suffix replay differs')
                if result['status'] == 'rejected':
                    # A failed declared world aborts the comparison; it cannot be dropped.
                    raise ValueError(f'Declared scenario {scenario["id"]}/{branch} refused: {result["reason"]}')
                trace.append(semantic_digest({'state': candidate.state, 'receipt': result}))
            if scenario['id'] == 'nominal' and candidate.snapshot() != snapshot:
                raise ValueError('Nominal complete recording differs')
            vector = consequence_vector(candidate, base)
            if vector != consequence_vector(replay, base):
                raise ValueError('Repeated consequence vector differs')
            samples.append({'scenarioId': scenario['id'], 'branchId': branch, 'vector': vector})
            receipts.append({'scenarioId': scenario['id'], 'branchId': branch,
                'sameSourceLabels': array_digest(scene.labels) == source_hashes[1],
                'exactEverySuffixStateAndReceipt': True, 'traceHash': semantic_digest(trace),
                'stateId': candidate.state_id, 'suffixActions': len(trace)})
            (output / 'samples.json').write_text(json.dumps(samples, indent=2, allow_nan=False) + '\n')
            (output / 'replay-receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
            print(json.dumps({'scenario': scenario['id'], 'branch': branch,
                'elapsedWallSeconds': time.perf_counter() - started, 'stateId': candidate.state_id}), flush=True)
    comparison = seal_report(original, samples, scenario_records, protocol['policies'],
        baseline_branch=protocol['prefixBranch'], stop_branch=protocol['stopBranch'],
        source_archive_sha256=protocol['sourceArchiveSha256'], declaration=protocol)
    validate_saved_comparison(original, record, comparison)
    record['consequenceComparison'] = comparison
    record['branchId'] = comparison['decisions'][0]['selectedBranchId']
    record['position'] = len(record['branches'][record['branchId']]['actions'])
    destination = output / 'consequence-comparison.ressectionlab'
    artifacts['surgical_toolbox'] = record
    save_case(case, destination, artifacts=artifacts)
    if source_hashes != (array_digest(case.mri), array_digest(original.labels)) or file_sha256(source) != protocol['sourceArchiveSha256']:
        raise ValueError('Source arrays/archive changed')
    result = {'schema': 'resectionlab.consequence-qualification.v1', 'case': str(destination),
        'sourceCase': str(source), 'sourceArchivePreserved': True, 'sourceArraysPreserved': True,
        'sourceSceneId': original.identity, 'sourceLabelsHash': source_hashes[1],
        'comparison': comparison, 'wallSeconds': time.perf_counter() - started,
        'trainingUpdates': 0, 'newAcquiredDatasetBytes': 0, 'physiologicalRL': 'hold'}
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > protocol['caps']['outputBytes']:
        raise ValueError('Consequence output exceeded the declared storage budget; evidence retained')
    print(json.dumps({'case': str(destination), 'decisions': comparison['decisions'], 'wallSeconds': result['wallSeconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('output', type=Path)
    prepare(parser.parse_args().output)
