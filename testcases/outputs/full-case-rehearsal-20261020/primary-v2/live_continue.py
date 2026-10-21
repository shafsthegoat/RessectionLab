"""Keep one canonical patient engine alive while recovering a blocked director.

Run with python -u -i. Only proposal modules may be explicitly reloaded. Patient
state is advanced only by full_case_rehearsal.run_slice/public preview + step;
every saved boundary retains the native source and complete action ancestry.
"""
from pathlib import Path
import hashlib
import importlib
import json
import sys
import time
from datetime import datetime, timezone

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'src/resectionlab').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from resectionlab.core import array_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.workspace_bundle import load_workspace
import resectionlab.full_case_rehearsal as rehearsal
from run_full_case_rehearsal import persist

destination = Path(__file__).resolve().parent / 'primary.ressectionlab'
original = ROOT / 'testcases/cases/meningioma-sub-06-standard-v2.ressectionlab'
source_sha = hashlib.sha256(original.read_bytes()).hexdigest()
assert source_sha == 'ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838'
case, artifacts, workspace = load_workspace(destination)
scene = load_environment(destination)
original_case, _, _ = load_workspace(original)
assert scene.identity == load_environment(original).identity
assert case.semantic_hash == original_case.semantic_hash
assert array_digest(case.mri) == array_digest(original_case.mri)
record = artifacts['surgical_toolbox']
branch_id = record['branchId']
snapshot = record['branches'][branch_id]
parent = record['parents'][branch_id]
director = record['fullCase'][branch_id]
print(json.dumps({'canonicalRestoreStarting': len(snapshot['actions']),
                  'expectedStateId': snapshot['stateId'], 'utc': datetime.now(timezone.utc).isoformat()}), flush=True)
restore_started = time.perf_counter()
# This diagnostic session verifies the full native replay without staging an
# optional disk cache while proposal files are being repaired.
engine = SurgicalEngine.restore(scene, snapshot, use_cache=False)
assert engine.state_id == director['lastStateId']
assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
label_hash, image_hash = array_digest(scene.labels), array_digest(case.mri)
print(json.dumps({'canonicalRestorePassed': len(engine.actions), 'stateId': engine.state_id,
                  'restoreWallSeconds': time.perf_counter() - restore_started}), flush=True)
started = time.perf_counter()
live_slice = 0


def reload_proposals():
    """Reload only reviewed read-only proposal/director code, never physics."""
    global rehearsal
    for name in ('full_case_geometry', 'full_case_debulking', 'full_case_withdrawal', 'full_case_rehearsal'):
        module = importlib.import_module('resectionlab.' + name)
        importlib.reload(module)
    rehearsal = importlib.import_module('resectionlab.full_case_rehearsal')
    print('Reviewed proposal modules reloaded; patient engine unchanged.', flush=True)


def resume():
    global director
    director = rehearsal.resume_director(director)
    print('Director ready at recorded state', engine.state_id, flush=True)


def receipt():
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    value = {'schema': 'resectionlab.full-case-execution.v1', 'originalCase': str(original),
        'originalSha256': source_sha, 'episode': str(destination), 'sceneId': scene.identity,
        'labelArrayHash': label_hash, 'imageHash': image_hash, 'branchId': branch_id,
        'stateId': engine.state_id, 'actions': len(engine.actions), 'metrics': engine.metrics(),
        'accounting': rehearsal.accounting(engine), 'director': director,
        'wallSeconds': time.perf_counter() - started, 'sourceUnchanged': True,
        'executionMode': 'live canonical recovery session; wall includes console idle time'}
    (destination.parent / 'execution-receipt.json').write_text(json.dumps(value, indent=2) + '\n')
    return value


def advance(slices=4000):
    global engine, director, live_slice
    for _ in range(slices):
        before = (len(engine.actions), director['stage'], director['accessIndex'])
        engine, director = rehearsal.run_slice(engine, director, max_actions=2)
        row = {'slice': live_slice, 'executionMode': 'live-recovery', 'position': len(engine.actions),
            'stateId': engine.state_id, 'stage': director['stage'], 'phase': director['currentPhaseId'],
            'status': director['status'], 'metrics': engine.metrics(),
            'wallSeconds': time.perf_counter() - started, 'lastOutcome': director.get('lastOutcome')}
        with (destination.parent / 'execution.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        persist(case, artifacts, workspace, engine, director, destination, branch_id, parent)
        live_slice += 1
        print(json.dumps({key: row[key] for key in ('position', 'stage', 'phase', 'status')} |
            {'targetRemovedMm3': row['metrics']['targetRemovedMm3']}), flush=True)
        if director['status'] in ('complete', 'stopped'):
            receipt()
            print('Saved stop:', director.get('stopReason'), flush=True)
            return
        if before == (len(engine.actions), director['stage'], director['accessIndex']):
            raise ValueError('Director made no physical or scheduling progress')
    receipt()


print('Live canonical recovery ready. Use reload_proposals(); resume(); advance().', flush=True)
