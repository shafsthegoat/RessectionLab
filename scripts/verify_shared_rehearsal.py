#!/usr/bin/env python3
"""Portable handoff checks; public physics replay is always an explicit option."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'testcases/outputs/full-case-rehearsal-20261020'
MANIFEST = ROOT / 'artifacts/full-case-rehearsal-20261020/manifest.json'
sys.path.insert(0, str(ROOT / 'src'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def check_files():
    manifest = json.loads(MANIFEST.read_text())
    total = 0
    for row in manifest['files']:
        relative = Path(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid manifest path')
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), 'Missing regular file: ' + str(relative))
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'File differs: ' + str(relative))
        total += row['bytes']
    return manifest, total


def quick_records():
    # Match the historical source-only audit. No engine constructor/restore/step.
    from resectionlab.core import semantic_digest
    from resectionlab.surgical_environment import load_environment
    from resectionlab.workspace_bundle import load_workspace
    helper = TASK / 'review/accept_native.py'
    spec = importlib.util.spec_from_file_location('shared_native_audit', helper)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)

    def no_engine(frame, event, arg):
        if (event == 'call' and frame.f_globals.get('__name__') == 'resectionlab.surgical_engine'
                and frame.f_code.co_name not in ('<module>', 'SurgicalEngine', '_OperationCheckFailed')):
            raise RuntimeError('Source-only check attempted engine execution: ' + frame.f_code.co_name)

    require(sys.getprofile() is None, 'Run the quick check without an existing profiler')
    sys.setprofile(no_engine)
    try:
        native = TASK / 'final/rehearsal.ressectionlab'
        source, _, source_workspace = load_workspace(audit.SOURCE)
        case, artifacts, workspace = load_workspace(native)
        require(audit.array_hashes(source) == audit.array_hashes(case), 'Source arrays differ')
        require(source.semantic_hash == case.semantic_hash and source.source_refs == case.source_refs,
                'Source identity/reference mismatch')
        require(audit.originals(source_workspace) == audit.originals(workspace), 'Embedded originals differ')
        scene = load_environment(native)
        require(scene.identity == audit.SCENE_ID, 'Scene differs')
        toolbox = artifacts['surgical_toolbox']
        branches = toolbox['branches']
        counts = {key: len(value['actions']) for key, value in branches.items()}
        require(counts == {'full-primary': 2148, 'full-vascular': 73, 'exposure-prefix': 42,
                           'tapered-frontier-prefix': 19}, 'Branch counts differ')
        records = {}
        for name, snapshot in branches.items():
            state = snapshot['checkpoint']['state']
            state_id = semantic_digest({'scene': scene.identity, 'state': state,
                                       'protectionContract': snapshot['protectionContract']['contentHash']})
            require(state_id == snapshot['stateId'] == snapshot['checkpoint']['stateId'], 'State hash differs')
            require(snapshot['checkpoint']['position'] == len(snapshot['actions']), 'Position differs')
            audit.protection_audit(scene, snapshot)
            if name in ('full-primary', 'full-vascular'):
                accounting = audit.source_accounting(scene, state)
                records[name] = {'actions': len(snapshot['actions']), 'stateId': state_id,
                                 'target': accounting['target'], 'fluid': audit.fluid_audit(state),
                                 'wounds': audit.wound_audit(state, scene)}
                require(not state['holds'] and all(not s['inserted'] and not s['active'] and not s['grip']
                        for s in state['slots'].values()), 'Endpoint tool/hold mismatch')
                require(all(not s['bleedingActive'] for s in state['sources']), 'Endpoint still bleeding')
        for name in ('full-primary', 'full-vascular'):
            require(branches[name]['actions'][:42] == branches['exposure-prefix']['actions'], 'Prefix differs')
        require(branches['full-primary']['actions'][:19] == branches['tapered-frontier-prefix']['actions'],
                'Tapered prefix differs')
        require(toolbox['parents']['full-vascular'] == {'branchId': 'exposure-prefix', 'position': 42},
                'Vascular parent differs')
        target = records['full-primary']['target']
        require((target['removedOwnedCells'], target['retainedOwnedCells']) == (35444, 8129),
                'Target accounting differs')
        require(toolbox['fullCase']['full-primary']['status'] == 'stopped', 'Endpoint status differs')
        return records
    finally:
        sys.setprofile(None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--files-only', action='store_true', help='Stdlib-only published byte/hash check')
    mode.add_argument('--canonical', choices=('primary', 'vascular'), help='Explicit slow forced public replay')
    mode.add_argument('--prepare-frames', choices=('primary', 'vascular'), help='Explicit .2s replay; requires local canonical cache')
    mode.add_argument('--audit-frames', choices=('primary', 'vascular'), help='Cache-only every-frame audit; no physics fallback')
    parser.add_argument('--canonical-report', type=Path, help='Local canonical report required with --audit-frames')
    args = parser.parse_args()
    require(__debug__, 'Do not run verification with Python -O')
    require(not args.canonical_report or args.audit_frames, '--canonical-report requires --audit-frames')
    manifest, total = check_files()
    result = {'publishedFileIntegrity': 'passed', 'files': len(manifest['files']), 'bytes': total,
              'historicalEngineCommit': manifest['engineCommit'], 'publicPhysicsReplayRequested': False}
    role = args.canonical or args.prepare_frames or args.audit_frames
    if role:
        require(not args.audit_frames or args.canonical_report, '--audit-frames needs --canonical-report')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
        output = ROOT / 'testcases/outputs/shared-rehearsal-verification' / stamp
        output.mkdir(parents=True, exist_ok=False)
        branch = 'full-' + role
        native = TASK / 'final/rehearsal.ressectionlab'
        review = TASK / 'review'
        if args.canonical:
            command = [sys.executable, str(review / 'accept_native.py'), str(native), branch,
                       '--report', str(output / (role + '-canonical.json'))]
        elif args.prepare_frames:
            command = [sys.executable, str(review / 'export_observed_telemetry.py'), str(native), branch,
                       str(output / (role + '-telemetry')), '--prepare-frames']
        else:
            command = [sys.executable, str(review / 'verify_every_cached_frame_monitoring.py'), str(native),
                       branch, str(args.canonical_report.resolve()), str(output / (role + '-every-frame.json'))]
        env = dict(os.environ, RESSECTIONLAB_VERIFIED_CACHE_MAX_DISK_BYTES='8589934592',
                   RESSECTIONLAB_OBSERVED_CACHE_MAX_DISK_BYTES='8589934592')
        print(json.dumps({**result, 'operation': parser.prog, 'branch': branch,
                          'output': str(output), 'slowOperationExplicitlyRequested': True}), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        return
    if not args.files_only:
        records = quick_records()
        result['savedSourceRecords'] = 'passed'
        result['branches'] = {name: {'actions': row['actions'], 'stateId': row['stateId'],
                             'removedOwnedTargetCells': row['target']['removedOwnedCells'],
                             'retainedOwnedTargetCells': row['target']['retainedOwnedCells']}
                              for name, row in records.items()}
    result['scope'] = 'Published bytes and saved source records only; no new canonical or GUI verification.'
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
