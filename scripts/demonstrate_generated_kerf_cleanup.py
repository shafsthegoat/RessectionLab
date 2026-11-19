"""Compare the post-scalp square control, save the recommendation and replay it cold.

Run with .venv/bin/python scripts/demonstrate_generated_kerf_cleanup.py
--output testcases/outputs/kerf-cleanup. The original unacted control is reused
when installed. A fresh checkout can reconstruct its generated source locally.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from generated_cranial_demo_case import bone_actions, joint_source, pose, preparation_program, request
from resectionlab.core import CaseData, SourceRef, semantic_digest
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.imaging import load_case, read_case_artifacts, save_case
from resectionlab.surgical_cranial_planning import validate_cranial_report
from resectionlab.surgical_cranial_planning_roots import FIELD, candidate_scene, create_registry, source_binding, validate_registry
from resectionlab.surgical_cranial_stages import freeze_cranial_stage_contract
from resectionlab.surgical_craniotomy import binding, contract_hash
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_joint_cranial_exit import joint_exit_complete
from resectionlab.surgical_joint_cranial_search import canonical_endpoint_assessment, endpoint_eligible

INSTALLED = ROOT / 'testcases/decision-system-v1/cranial-generate-desktop-v2/generated-unacted-source.ressectionlab'
CASE_ID = 'sha256:9818506d2f779b8fece10d048e0f21e2357f86851936ad0f5b86ee6f6cfaa1e0'
DIRTY_STATE = 'sha256:50a0af330ac9d1c2d6360f7cf450f8c5dd4b8f45826a58454425b7f02c0b9e8a'


def square_source():
    template, stage, contour, _, _ = joint_source()
    scene = template.scene
    properties = copy.deepcopy(scene.properties)
    opening = properties.pop('cranialAccessV1')
    regions = scene.regions
    case = CaseData(case_id='generated-immutable-opening-roots', mri=scene.labels.astype(np.float32),
        affine=scene.affine, compartments={r['name']: scene.labels == r['id'] for r in regions},
        source_refs=(SourceRef('generated', 'synthetic://immutable-opening-roots', provenance='simulated'),),
        metadata={'generated': True, 'is_synthetic': True, 'surgical_properties': properties,
                  'anatomy_layers': {r['name']: {'kind': r['material']} for r in regions}})
    assert case.semantic_hash == CASE_ID
    base = from_case(case, regions)
    opening.update(binding(base))
    opening['contentHash'] = contract_hash(opening)
    candidate = candidate_scene(base, 'cranialAccessV1', opening)
    contract = freeze_cranial_stage_contract(SurgicalEngine(candidate), membrane_cells=stage.membrane_cells,
        retained_membrane_cells=stage.retained_membrane_cells, membrane_anchor_cells=stage.membrane_anchor_cells,
        work_cells=stage.work_cells, evidence=stage.as_dict()['evidence'])
    engine = SurgicalEngine(candidate, cranial_stages=contract)
    opening_actions = bone_actions(contour)
    preparation = preparation_program(engine, contract, contour)[len(opening_actions):]
    preparation += [request('exchange', instrument='cusa'), request('place', pose=pose())]
    # Rebuild the original source declaration, including the evidence receipt digest.
    evidence = dict(schema='generated-declared-legacy-joint-desktop-source.v1',
        source='tests/test_surgical_joint_cranial_forest.py::test_native_forest_complete_legacy_sequence_and_standardized_inner_stop[declared_legacy]',
        scope='Generated idealized categorical source only; no patient/clinical qualification',
        caseSemanticHash=case.semantic_hash, sourceBinding=source_binding(base),
        openingContractHash=opening['contentHash'], stageContractHash=contract.content_hash)
    digest = hashlib.sha256((json.dumps(evidence, indent=2) + '\n').encode()).hexdigest()
    planning = {
        'limits': {'max_bone_mm3': 285., 'max_preparation_nonbone_mm3': 49., 'max_work_nonbone_mm3': 20.,
            'max_native_actions': 300, 'max_public_calls': 1400, 'max_wall_seconds': 180.,
            'max_bone_contact_replans': 6, 'max_tissue_contact_replans': 8, 'export_replay_reserve_calls': 300},
        'openingSkill': {'kind': 'opening', 'id': 'source-opening', 'actions': opening_actions},
        'preparationActions': preparation, 'outsidePoses': {},
        'workRoutes': [{'id': 'unreached-long-work', 'motion': request('apply', pose=pose(z=68.), duration=.05)},
                       {'id': 'target-work', 'motion': request('apply', pose=pose(z=66.5), duration=30.)}],
        'qualification': {'status': 'reviewed', 'sourceBinding': source_binding(base), 'evidenceSha256': digest,
            'provenance': 'Generated declared-legacy engineering source only; exact native forest control. Not patient anatomy.',
            'blockers': []}}
    registry = create_registry(base, [{'id': 'opening-0', 'label': 'Generated opening, preparation and target work',
        'openingProperty': 'cranialAccessV1', 'openingContract': opening, 'sourceSnapshot': engine.snapshot(),
        'planning': planning}])
    return case, {'mapping': regions, 'branches': {'main': SurgicalEngine(base).snapshot()},
        'branchId': 'main', 'position': 0, 'parents': {'main': None}, 'timeline': {'main': []}, FIELD: registry}


def plan(source, destination):
    with TemporaryDirectory() as temp:
        session = BridgeSession(Path(temp))
        def dispatch(operation, **args):
            return session.execute(operation, args, _Request('generated-kerf-cleanup'), lambda *unused: None)
        try:
            case = dispatch('loadCase', path=str(source))['caseHash']
            assert case == CASE_ID
            def surgical(command, **args):
                return dispatch('surgicalRehearsal', caseHash=case, command=command, **args)
            source_view = surgical('inspect')['view']
            result = surgical('plan', expectedStateId=source_view['stateId'], opponent='anatomy-cranial-strategy',
                              planningHorizon=5, beamWidth=3, acknowledgeResearchUnknowns=True)
            report = result['cranialPlanning']
            assert report['decision'] == 'COMPLETED_FROZEN_TASK' and report['selectedBranchId']
            surgical('selectBranch', expectedStateId=result['view']['stateId'], branchId=report['selectedBranchId'])
            dispatch('saveCase', caseHash=case, path=str(destination))
        finally:
            session.close()


def verify(package):
    case = load_case(package)
    record = read_case_artifacts(package)['surgical_toolbox']
    assert case.semantic_hash == CASE_ID
    base = from_case(case, record['mapping'])
    report = record['cranialPlanning']
    validate_cranial_report(base, record, report)
    selected = next(r for r in report['alternatives'] if r['branchId'] == report['selectedBranchId'])
    assert record['branchId'] == selected['branchId'] and record['position'] == selected['nativeActions']
    root = next(r for r in record[FIELD]['roots'] if r['id'] == selected['rootId'])
    scene = validate_registry(base, record[FIELD])[root['id']]
    candidates = report['selectionEvidence']['candidates']
    dirty = next(c for c in candidates if c['finalStateId'] == DIRTY_STATE)
    source_stop = next(c for c in candidates if c['finalStateId'] == root['sourceSnapshot']['stateId'])
    assert source_stop['eligible'] and source_stop['endpointAssessment']['classification'] == 'valid-stop-with-residual'
    selected_key = selected['candidateKey']
    checks = {}
    for key, name in ((selected_key, 'selected'), (dirty['candidateKey'], 'originalFailure'),
                      (source_stop['candidateKey'], 'beforeAccessStop')):
        proof = next(p for p in report['selectionProofs'] if p['candidateKey'] == key)
        expected = next(c for c in candidates if c['candidateKey'] == key)['endpointAssessment']
        # This preserved source predates the procedure policy required by the
        # RL adapter. Literal native replay and its own canonical STOP contract
        # are authoritative here, with no change to source qualification.
        native = SurgicalEngine.restore(scene, root['sourceSnapshot'], use_cache=False)
        for event in proof['nativeHistory']:
            outcome = native.step(event['action'])
            assert outcome['status'] == event['status'] and native.history[-1] == event
        assert native.snapshot() == proof['snapshot']
        before = native.state_id
        ledger = proof['ledger']
        physical_exit = joint_exit_complete(native)
        outcome = canonical_endpoint_assessment(native, within_limits=ledger['withinLimits'],
            refusal_reasons=ledger['refusalReasons'], stop_checked=physical_exit)
        assert native.state_id == before and outcome == expected
        checks[name] = {'stateId': before, 'primitiveActions': len(proof['nativeHistory']),
            'historyHash': semantic_digest(proof['nativeHistory']), 'logicalStopAccepted': endpoint_eligible(outcome),
            'bothToolsExterior': physical_exit, 'endpoint': outcome, 'debrisRemovedMm3': native.state.get('debrisRemovedMm3', 0.),
            'modeledSeconds': ledger['modeledElapsedSeconds'],
            **{k: ledger[k] for k in ('boneRemovedMm3','preparationNonboneRemovedMm3','workNonTargetRemovedMm3',
                                       'unauthorizedRemovedSourceCells','priorExposureCost')}}
    success, failure, stop = (checks[k] for k in ('selected','originalFailure','beforeAccessStop'))
    assert success['endpoint']['completedTask'] and success['endpoint']['completeOriginalTargetRemoval']
    assert failure['endpoint']['debrisMm3'] == 236.
    assert abs(success['debrisRemovedMm3'] - failure['endpoint']['debrisMm3']) < 1e-8
    assert success['endpoint']['debrisMm3'] == 0 and success['endpoint']['progress']['originalTargetTreatmentMm3'] == 27.
    assert failure['primitiveActions'] == 119 and failure['endpoint']['pending'] == ['debris']
    assert stop['primitiveActions'] == stop['boneRemovedMm3'] == stop['preparationNonboneRemovedMm3'] == 0
    return {'schema': 'resectionlab.generated-square-cleanup.v1', 'caseSemanticHash': case.semantic_hash,
        'sourceStateId': report['sourceStateId'], 'rootSeal': root['rootSeal'], 'branchId': selected['branchId'],
        'comparisonSeal': report['comparisonSeal'], 'configHash': report['endpointContract']['configHash'],
        'decision': report['decision'], 'candidateCount': len(candidates),
        'planSearchAndExport': {k: report['cost'][k] for k in ('physicalCalls','previewCalls','directStepCalls',
                                                              'maxPhysicalCalls','wallSeconds')},
        'coldNativeStatesAndOutcomesMatched': True, 'actual': checks,
        'rlAdapter': 'This legacy source lacks the procedure evaluation contract required for staged RL access.',
        'scope': 'Generated planar post-scalp source. Earlier scalp exposure cost is unknown. Clinical validation is absent.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, help='Previously saved unacted square control')
    args = parser.parse_args()
    # Preserve the lexical testcases path so it can point at the mounted evidence drive.
    output = Path(args.output.absolute())
    if '..' in output.parts or not output.is_relative_to(ROOT / 'testcases'):
        parser.error('Place generated app datasets under testcases/')
    output.mkdir(parents=True, exist_ok=False)
    source = args.source or INSTALLED
    if args.source is None and not source.exists():
        case, record = square_source()
        source = output / 'source.ressectionlab'
        save_case(case, source, artifacts={'surgical_toolbox': record})
    package = output / 'complete.ressectionlab'
    plan(source.resolve(strict=True), package)
    report = verify(package)
    report.update(sourceSha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  packageSha256=hashlib.sha256(package.read_bytes()).hexdigest())
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'decision': report['decision'], 'branchId': report['branchId'], 'package': str(package)}))


if __name__ == '__main__':
    main()
