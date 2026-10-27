#!/usr/bin/env python3
"""Run deterministic staged skills on declared DEVELOPMENT, preserving source."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_staged_planning import (
    compare_staged_strategies, install_staged_comparison, staged_summary,
    validate_staged_report, MAX_CALLS, MAX_SECONDS)
from resectionlab.workspace_bundle import load_workspace, save_workspace
from run_patient_sector_learning import development_declaration


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(source, output, *, explore_fresh_corridors=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT/'testcases'):
        raise ValueError('Choose a new ignored testcases output directory')
    output.mkdir(parents=True)
    started = time.monotonic()
    receipt = {'schema': 'resectionlab.staged-export.v1', 'status': 'running',
        'source': str(source), 'sourceSha256': digest(source),
        'maxPhysicalCalls': MAX_CALLS, 'maxWallSeconds': MAX_SECONDS}
    result = None
    try:
        declaration = development_declaration(source)
        receipt['sourceRoleDeclaration'] = declaration
        case, artifacts, workspace = load_workspace(source)
        scene = load_environment(source)
        record = copy.deepcopy(artifacts['surgical_toolbox'])
        if record.get('position') != 0 or record['branches'][record['branchId']]['actions']:
            raise ValueError('Staged comparison starts at the common intact source')
        original_hash = semantic_digest(record['branches'])
        engine = SurgicalEngine.restore(scene, record['branches'][record['branchId']])
        def check():
            if time.monotonic()-started > MAX_SECONDS:
                raise TimeoutError('Staged complete source/export wall budget exhausted')
        result = compare_staged_strategies(engine, record,
            acknowledge_unknowns=True, check=check,
            explore_fresh_corridors=explore_fresh_corridors,
            max_wall_seconds=max(1., MAX_SECONDS-(time.monotonic()-started)))
        if result['report'].get('interruption'):
            receipt['report'] = result['report']
            raise TimeoutError(result['report']['interruption'])
        install_staged_comparison(record, result, scene)
        arts = copy.deepcopy(artifacts)
        arts['surgical_toolbox'] = record
        destination = output/'staged-comparison.ressectionlab'
        def save():
            if workspace is None:
                from resectionlab.imaging import save_case
                save_case(case, destination, artifacts=arts)
            else:
                save_workspace(case, destination, sources=workspace['sources'],
                    imaging=workspace['imagingState'], episode=workspace['episode'],
                    selection=workspace['selection'], originals=workspace['originals'], artifacts=arts)
        save()
        reopened_case, reopened_artifacts, _ = load_workspace(destination)
        saved = reopened_artifacts['surgical_toolbox']
        if reopened_case.semantic_hash != case.semantic_hash:
            raise ValueError('Staged source image identity changed during export')
        with result['meter'].phase('staged.export_reopen_replay'):
            for candidate, row in result['branches']:
                replay = result['meter'].wrap(engine.fresh())
                for action in saved['branches'][row['branchId']]['actions']:
                    check()
                    replay.step(action, check=check)
                if replay.state_id != candidate.state_id or replay.state != candidate.state:
                    raise ValueError('Staged native export replay differs')
        # Re-save only accounting metadata after replay; the verified native
        # trajectories and source arrays are byte-for-byte unchanged.
        report = result['report']
        report['cost'] = result['meter'].report()
        report['wallSeconds'] = time.monotonic()-started
        report['comparisonSeal'] = semantic_digest({
            key: value for key, value in report.items() if key != 'comparisonSeal'})
        record['stagedPlanning'] = copy.deepcopy(report)
        record['stagedSummary'] = staged_summary(report)
        validate_staged_report(scene, record, report)
        save()
        final_case, final_arts, _ = load_workspace(destination)
        if (final_case.semantic_hash != case.semantic_hash or
                final_arts['surgical_toolbox']['branches'] != saved['branches'] or
                semantic_digest(artifacts['surgical_toolbox']['branches']) != original_hash):
            raise ValueError('Final accounting save changed native source/trajectories')
        check()
        receipt.update(status='saved', output=str(destination),
            outputSha256=digest(destination), report=report,
            exactExportReplay=True, originalBranchesPreserved=True)
    except BaseException:
        receipt.update(status='failed', failure=traceback.format_exc())
    finally:
        if result is not None:
            receipt['nativeCalls'] = result['meter'].report()
        receipt['sourceUnchanged'] = digest(source) == receipt['sourceSha256']
        receipt['wallSeconds'] = time.monotonic()-started
        (output/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--explore-staged-source', action='store_true',
        help='Explicit experimental source-only corridor selection; default retains measured access baselines')
    args = parser.parse_args()
    receipt = run(args.case, args.output, explore_fresh_corridors=args.explore_staged_source)
    print(json.dumps({'status': receipt['status'], 'wallSeconds': receipt['wallSeconds'],
        'nativeCalls': receipt.get('nativeCalls', {}).get('physicalCalls'),
        'receipt': str(args.output/'receipt.json')}), flush=True)
    if receipt['status'] != 'saved':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
