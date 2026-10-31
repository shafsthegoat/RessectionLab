#!/usr/bin/env python3
"""Reproduce the admitted public-input contract on an existing native baseline.

Uses no new acquisition or exposure. Reference swaps are software controls, not
additional patients or anatomical uncertainty samples. Outputs stay ignored.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_task_admission import bind_admission
from resectionlab.surgical_interactive_planning import (generate_interactive_approaches,
    interactive_planning_availability, materialize_interactive_approach, evaluate_sealed_contact_references)
from resectionlab.workspace_bundle import load_workspace
from verify_interactive_geometry import verify


def check(source, output):
    source, output = source.resolve(), output.resolve()
    if not source.is_relative_to(ROOT/'testcases') or not output.is_relative_to(ROOT/'testcases'):
        raise ValueError('Use ignored testcases paths')
    if output.exists(): raise ValueError('Preserve evidence; choose a fresh output')
    output.mkdir(parents=True)
    result = {'schema': 'resectionlab.public-planning-check.v1', 'source': str(source), 'timings': {}, 'passed': False}
    def run(name, fn):
        start = time.perf_counter()
        try: return fn()
        finally:
            result['timings'][name] = time.perf_counter()-start
            print(name, round(result['timings'][name], 3), flush=True)
    try:
        original = hashlib.sha256(source.read_bytes()).hexdigest()
        _, artifacts, _ = run('loadWorkspace', lambda: load_workspace(source))
        record = artifacts['surgical_toolbox']; branch = record['branchId']
        scene = run('loadScene', lambda: load_environment(source))
        e = run('coldRestore', lambda: SurgicalEngine.restore(scene, record['branches'][branch]))
        run('bindSourceAdmission', lambda: bind_admission(e, record['taskAdmission']))
        before = e.snapshot(); defaults = interactive_planning_availability(e)['defaults']
        plan = run('coldGenerateAndPreview', lambda: generate_interactive_approaches(e, defaults))
        again = run('warmGenerateAndPreview', lambda: generate_interactive_approaches(e, defaults))
        assert plan == again
        (output/'comparison.json').write_text(json.dumps(plan, indent=2)+'\n')
        e._evaluation_truth = np.zeros(scene.shape, bool)
        private_a = run('privateReferenceA', lambda: generate_interactive_approaches(e, defaults))
        e._evaluation_truth = np.ones(scene.shape, bool)
        e._private_manifest = {'patient': 'changed', 'hash': 'private-only', 'file': '/private/control'}
        private_b = run('privateReferenceB', lambda: generate_interactive_approaches(e, defaults))
        assert plan == private_a == private_b and e.snapshot() == before
        # These synthetic full/empty reference masks are loaded only after the
        # plan is sealed. They provide a negative control for evaluator response.
        low = evaluate_sealed_contact_references(plan, lambda: {'software-reference': np.zeros(scene.shape, bool)})
        high = evaluate_sealed_contact_references(plan, lambda: {'software-reference': np.ones(scene.shape, bool)})
        if plan['routes']: assert low['referenceContactCells'] != high['referenceContactCells']
        result['postSealControls'] = [low, high]
        result['nativeReceipts'] = []
        for route in plan['routes']:
            candidate, receipt = run('native-'+route['id'], lambda: materialize_interactive_approach(e, plan, route['id']))
            restored = run('coldReplay-'+route['id'], lambda: SurgicalEngine.restore(scene, candidate.snapshot()))
            assert restored.snapshot() == candidate.snapshot() and receipt['nativeMatchesPublicPrediction']
            result['nativeReceipts'].append(receipt)
        if plan['routes']:
            audit = run('independentGeometry', lambda: verify(plan, source, branch=branch))
            assert audit['passed']
            (output/'independent-geometry.json').write_text(json.dumps(audit, indent=2)+'\n')
        constrained = copy.deepcopy(defaults); constrained['requiredEvidence'].append('functional')
        refusal = run('functionalEvidenceStop', lambda: generate_interactive_approaches(e, constrained))
        assert refusal['decision']['operation'] == 'STOP' and refusal['missingRequiredEvidence'] == ['functional']
        assert refusal['publicObservation'] == plan['publicObservation']
        (output/'functional-stop.json').write_text(json.dumps(refusal, indent=2)+'\n')
        assert e.snapshot() == before and hashlib.sha256(source.read_bytes()).hexdigest() == original
        result.update(passed=True, sourceSha256=original, sourceSceneId=scene.identity,
            stateId=e.state_id, publicObservation=plan['publicObservation'], publicPlanSeal=plan['publicPlanSeal'],
            decision=plan['decision'], status=plan['status'], candidateCount=len(plan['routes']),
            privateReferenceInvariance=True, sourceAndBaselinePreserved=True,
            scope='Annotation-assisted inactive access; software contract and independent geometry, not clinical validity')
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path); parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.source, args.output)['decision']))
