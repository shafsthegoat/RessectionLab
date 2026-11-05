#!/usr/bin/env python3
"""Read-only acquired-source/task preflight. Reuses existing testcases data."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_source_preflight import source_preflight
from resectionlab.surgical_task_admission import source_evidence, bind_admission
from resectionlab.workspace_bundle import load_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--branch')
    parser.add_argument('--native-collection', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_relative_to(ROOT/'testcases') or not output.is_relative_to(ROOT/'testcases') or output.exists():
        parser.error('Reuse a testcases source and choose a fresh testcases JSON output')
    original = sha256(source.read_bytes()).hexdigest()
    started = time.perf_counter()
    case, artifacts, _ = load_workspace(source); scene = load_environment(source)
    record = artifacts['surgical_toolbox']; before = semantic_digest(record)
    result = {'schema': 'resectionlab.surgical-source-preflight-run.v1', 'source': str(source),
              'sourceSha256': original, 'caseHash': case.semantic_hash, 'inspectionCompleted': False}
    try:
        # Refuse undeclared/reserved sources before any native action replay or
        # additional source-collection arrays are read. No role is minted here.
        source_evidence(scene)
        branch = args.branch or record['branchId']
        engine = SurgicalEngine.restore(scene, record['branches'][branch], use_cache=False)
        snapshot = engine.snapshot()
        if record.get('taskAdmission') is not None: bind_admission(engine, record['taskAdmission'])
        result.update(branch=branch, preflight=source_preflight(engine, native_collection=args.native_collection,
            verify_replay=True), inspectionCompleted=True)
        assert engine.snapshot() == snapshot
    except ValueError as error:
        result.update(refusal=str(error))
    assert semantic_digest(record) == before and sha256(source.read_bytes()).hexdigest() == original
    result.update(sourcePreserved=True, nativeArtifactsPreserved=True)
    result.update(elapsedSeconds=time.perf_counter()-started,
        timingScope='Source loading, role and native checkpoint/source preflight; rendering excluded; no candidate search or instrument edits',
        verifierCodeSha256={name:sha256((ROOT/'src/resectionlab'/f'{name}.py').read_bytes()).hexdigest()
            for name in ('surgical_source_roles','surgical_source_decode','surgical_source_preflight','surgical_task_admission')})
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'inspectionCompleted': result['inspectionCompleted'], 'refusal': result.get('refusal'),
        'planningAdmitted': result.get('preflight', {}).get('planningAdmitted', False), 'output': str(output)}))


if __name__ == '__main__': main()
