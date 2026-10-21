#!/usr/bin/env python3
"""Read saved native/receipt evidence only; never replay physics or use a GUI."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT = next(path for path in Path(__file__).resolve().parents if (path / 'src/resectionlab').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.imaging import read_case_artifacts, load_case
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_tools import TOOLS

BASE = ROOT / 'testcases/outputs/full-case-rehearsal-20261020'
SOURCE = ROOT / 'testcases/cases/meningioma-sub-06-standard-v2.ressectionlab'
EXPECTED_SOURCE = 'ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saved(path):
    for _ in range(3):
        before = sha(path)
        record = read_case_artifacts(path)['surgical_toolbox']
        after = sha(path)
        if before == after:
            return record, before
    raise ValueError('Native archive changed repeatedly while reading; no inconsistent inventory written')


def tool_actions(actions):
    tools = {'left': 'cranial_drill', 'right': 'cusa'}
    active = {'left': False, 'right': False}
    per_tool = {tool: {'scheduled': [], 'actuation': [], 'sharedClockExposure': [], 'operations': Counter()} for tool in TOOLS}
    acting_tools = []
    for index, action in enumerate(actions):
        slot, operation = action['slot'], action['operation']
        if operation == 'exchange': tools[slot] = action['instrument']
        tool = tools[slot]
        acting_tools.append(tool)
        row = per_tool[tool]
        row['scheduled'].append(index + 1)
        row['operations'][operation] += 1
        if operation in ('activate', 'apply', 'hold', 'grasp', 'extract') or operation == 'move' and active[slot]:
            row['actuation'].append(index + 1)
        for other in tools:
            if other != slot and active[other]:
                per_tool[tools[other]]['sharedClockExposure'].append(index + 1)
        if operation == 'activate': active[slot] = True
        if operation in ('deactivate', 'release', 'withdraw'): active[slot] = False
    for row in per_tool.values(): row['operations'] = dict(row['operations'])
    return per_tool, acting_tools


def endpoint(scene, state, case):
    removed = np.asarray(state['removed'], dtype=np.int64)
    removed_labels = scene.labels.ravel()[removed]
    materials = {}
    for region in scene.regions:
        materials[region['material']] = materials.get(region['material'], 0.) + float(np.count_nonzero(removed_labels == region['id'])) * scene.voxel_mm3
    target = float(sum(np.count_nonzero(removed_labels == region['id']) for region in scene.regions if region.get('target'))) * scene.voxel_mm3
    initial_target = float(sum(np.count_nonzero(scene.labels == region['id']) for region in scene.regions if region.get('target'))) * scene.voxel_mm3
    raw = {name: int(np.count_nonzero(mask.ravel()[removed])) for name, mask in case.compartments.items()}
    vessels = state.get('vesselsV2', {})
    faces = vessels.get('faces', {})
    present = sum(state['fluid'].values()) + sum(source.get('reservoirMm3', 0.) for source in state['sources'])
    return {'removedOwnedMaterialMm3': materials, 'removedOwnedTargetMm3': target,
        'initialOwnedTargetMm3': initial_target, 'retainedOwnedTargetMm3': initial_target - target,
        'removedSourceCellRawCompartmentMemberships': raw,
        'rawMembershipMeaning': 'Overlapping unreviewed annotation memberships, not clinical healthy tissue truth; do not change owned target denominator.',
        'drilling': {key: value for key, value in state.get('drillingV2', {}).items() if not isinstance(value, dict)},
        'debrisPresentMm3': state['debrisMm3'], 'debrisRemovedMm3': state.get('debrisRemovedMm3', 0.),
        'fluidAddedMm3': state['fluidAddedMm3'], 'fluidRemovedMm3': state['fluidRemovedMm3'],
        'fluidEscapedMm3': state['fluidEscapedMm3'], 'fluidPresentIncludingOutletsMm3': present,
        'fluidConservationResidualMm3': state['fluidAddedMm3'] - state['fluidRemovedMm3'] - state['fluidEscapedMm3'] - present,
        'wounds': {'count': len(vessels.get('wounds', {})), 'damagedFaces': len(faces),
            'faceAreaMm2': sum(face['areaMm2'] for face in faces.values()),
            'patentDamagedFaces': sum(face['patent'] for face in faces.values()),
            'coagulationDoseByFace': {key: face['coagulationDose'] for key, face in faces.items()},
            'clotByFace': {key: face['clot'] for key, face in faces.items()},
            'activeSources': sum(bool(source.get('bleedingActive')) for source in state['sources']),
            'generatedMm3': vessels.get('generatedMm3', 0.),
            'vascularConservationResidualMm3': vessels.get('initialMm3', 0.) + vessels.get('refilledMm3', 0.) - vessels.get('remainingMm3', 0.) - vessels.get('generatedMm3', 0.)}}


def main():
    assert sha(SOURCE) == EXPECTED_SOURCE
    old_inventory = BASE / 'vascular/tool-family-inventory.json'
    old_hash = sha(old_inventory)
    fixture = BASE / 'vascular/tool-family-fixture.xml'
    fixture_tests = ET.parse(fixture).getroot().findall('.//testcase')
    source_case = load_case(SOURCE)
    scene = load_environment(SOURCE)
    episodes = {}
    for name, relative in [('primary', 'primary-v2/primary.ressectionlab'), ('vascular', 'vascular-v2/vascular.ressectionlab')]:
        path = BASE / relative
        record, digest = saved(path)
        branch = record['branchId']; snapshot = record['branches'][branch]
        assert snapshot['sceneId'] == scene.identity
        meta = record.get('fullCase', {}).get(branch, {})
        uses, acting = tool_actions(snapshot['actions'])
        entry = {'native': str(path), 'nativeSha256': digest, 'branch': branch,
            'actionCount': len(snapshot['actions']), 'stateId': snapshot['stateId'],
            'directorStatus': meta.get('status'), 'directorStage': meta.get('stage'),
            'endpointSource': 'Saved native checkpoint; this inventory refresh does not independently replay or readmit it.',
            'endpointEffects': endpoint(scene, snapshot['checkpoint']['state'], source_case),
            'toolUses': uses, 'verifiedTelemetry': None, 'savedUncommittedProposalReasons': dict(Counter(row.get('reason') for row in meta.get('rejections', [])))}
        receipt_path = path.parent / 'execution-receipt.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        entry['matchingExecutionReceipt'] = {'path': str(receipt_path), 'sha256': sha(receipt_path),
            'matchesEndpoint': bool(receipt and receipt.get('stateId') == snapshot['stateId'] and receipt.get('actions') == len(snapshot['actions']))}
        telemetry_receipt_path = path.parent / 'telemetry/export-receipt.json'
        if telemetry_receipt_path.exists():
            telemetry_receipt = json.loads(telemetry_receipt_path.read_text())
            if telemetry_receipt['nativeSha256'] == digest and telemetry_receipt['finalStateId'] == snapshot['stateId']:
                telemetry_path = Path(telemetry_receipt['paths']['json'])
                observed = json.loads(telemetry_path.read_text())['episodes'][0]
                assert observed['finalStateId'] == snapshot['stateId'] and observed['expectedStateVerified']
                assert [row['action'] for row in observed['attempts']] == snapshot['actions']
                evidence = {'receipt': str(telemetry_receipt_path), 'receiptSha256': sha(telemetry_receipt_path),
                    'json': str(telemetry_path), 'jsonSha256': sha(telemetry_path),
                    'scope': 'Existing independently observed public replay matched this exact native endpoint; not rerun by inventory refresh.',
                    'statuses': dict(Counter(row['status'] for row in observed['attempts'])),
                    'partialActions': [{'actionPosition1Based': row['actionIndex'] + 1, 'reason': row.get('reason')} for row in observed['attempts'] if row['status'] == 'partial']}
                for tool, use in uses.items():
                    events = [row for row in observed['events'] if row['timing'] == 'committed_microstep' and acting[row['actionIndex']] == tool]
                    meaningful = [row for row in events if row['event']['kind'] != 'no_material_effect']
                    use['observedEvents'] = {'eventKinds': dict(Counter(row['event']['kind'] for row in meaningful)),
                        'actionPositions1Based': sorted({row['actionIndex'] + 1 for row in meaningful}),
                        'totalsByEventKind': {}}
                    for kind, fields in {'drill_work': ('removedVolumeMm3', 'inputEnergyJ', 'cuttingWorkJ'),
                                         'removal': ('cells', 'volumeMm3'), 'strand_transection': ('edges',),
                                         'local_energy': ('exposure',), 'irrigation': ('volumeMm3',),
                                         'aspiration': ('fluid', 'debris')}.items():
                        matching = [row['event'] for row in meaningful if row['event']['kind'] == kind]
                        if matching:
                            use['observedEvents']['totalsByEventKind'][kind] = {field: sum(row.get(field, 0.) for row in matching) for field in fields}
                entry['verifiedTelemetry'] = evidence
        episodes[name] = entry
    limitations = {
        'periosteal_elevator': 'No declared periosteal attachment properties; no patient separation claim.',
        'dissector': 'No declared soft/tumor attachment properties; no patient dissection/separation claim.',
        'dural_scissors': 'No supplied membrane/dura anatomy; no patient dural opening claim.',
        'craniotome': 'No declared bone anchors establish flap separability; no patient craniotome/flap claim.',
        'brain_retractor': 'Supplied owned soft tissue exists, but no retraction in these selected patient branches.',
        'tumor_forceps': 'No forceps use/free-fragment extraction proof in these branches; attachments unknown.',
        'cortical_probe': 'No supplied functionalEvidence; no patient functional response or localization claim.',
        'subcortical_probe': 'No supplied functionalEvidence; no patient functional response or localization claim.'}
    effects = {
        'cranial_drill': 'Actual saved bone removal plus drilling work/heat ledger; vascular observed drill_work events independently match native endpoint.',
        'cusa': 'Actual saved owned-target removal; vascular observed removal events independently match312mm3 target. Non-target bone removal is listed separately and is not a healthy-injury label.',
        'micro_scissors': 'Vascular observed strand_transection events at action48; final five fully damaged vessel faces share one wound and remain nonpatent.',
        'bipolar_forceps': 'Vascular observed local_energy at55–56; final five faces each actual coagulationDose2.0/clot1.0, no active wound; later idle observations retain control.',
        'irrigation': 'Vascular observed irrigation at62–63 totals6mm3; endpoint fluidAdded also includes4.400156722mm3 modeled wound emission.',
        'frazier_suction': 'Vascular observed aspiration at68–69 removes5.966244726mm3 fluid. Primary endpoint also records177.102564103mm3 debris removal with stationary second-hand suction; per-microstep primary telemetry is not available in this refresh.'}
    tools = []
    for tool, model in TOOLS.items():
        case = next(row for row in fixture_tests if '[' + tool + '-' in row.attrib['name'])
        assert not list(case)
        patient = {name: episode['toolUses'][tool] for name, episode in episodes.items() if episode['toolUses'][tool]['actuation'] or episode['toolUses'][tool]['sharedClockExposure']}
        tools.append({'tool': tool, 'effectFamily': model['effect'],
            'patientEvidenceStatus': 'actual action/effect evidence in selected native patient branches' if tool in effects else 'engineering fixture proof only; no action/effect in selected patient branches',
            'patientProof': effects.get(tool), 'patientActionEvidence': patient,
            'engineeringFixtureEvidence': {'status': 'passed-existing-receipt',
                'test': 'tests/test_surgical_tools.py::' + case.attrib['name'],
                'xml': str(fixture), 'xmlSha256': sha(fixture),
                'qualification': 'Explicit synthetic anatomy/properties; prior saved execution, not rerun during this refresh.'},
            'patientEvidenceLimit': limitations.get(tool, 'Declared uncalibrated research laws; no clinical force, blood-loss, safety, or outcome claim.')})
    created = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report = {'schema': 'resectionlab.tool-family-evidence-inventory.v2', 'createdUtc': created,
        'scope': 'Read-only inventory of current saved primary and completed vascular alternatives; no physics replay, tests, GUI, source edits or treatment performed by this refresh.',
        'source': str(SOURCE), 'sourceSha256': EXPECTED_SOURCE, 'sceneId': scene.identity,
        'supersedesForCurrentEvidenceOnly': {'path': str(old_inventory), 'sha256': old_hash, 'oldFilePreserved': True},
        'indexConvention': 'All action position lists are 1-based saved branch positions; subtract 1 for zero-based actionIndex.',
        'episodes': episodes, 'toolCount': len(tools), 'tools': tools,
        'anatomyEvidence': {'suppliedOwnedMaterials': sorted({row['material'] for row in scene.regions}),
            'membraneSupplied': any(row['material'] == 'membrane' for row in scene.regions),
            'declaredAttachments': len(scene.properties.get('attachments', [])),
            'declaredBoneAnchors': len(scene.properties.get('boneAnchors', [])),
            'declaredFunctionalEvidence': len(scene.properties.get('functionalEvidence', [])),
            'scalpDuraAttachmentAndFunctionalCompleteness': 'UNKNOWN; absent evidence was not fabricated',
            'vascularMeaning': 'Supplied unreviewed vessel mask and five modeled wound faces do not establish a complete anatomical network, perfusion, or calibrated patient physiology.'},
        'guiReview': 'Not performed by this inventory; existing telemetry receipt reports guiVerified=false.',
        'limitations': ['Primary is a captured running endpoint, not complete tumor removal.',
            'Checkpoint endpoint values are recorded simulation state; independent canonical replay status is only claimed where a matching existing receipt is identified.',
            'Saved branch actions omit rejected attempts; existing vascular telemetry has '
            + str(len((episodes['vascular']['verifiedTelemetry'] or {}).get('partialActions', [])))
            + ' partial committed actions. Uncommitted proposal reasons are reported separately.',
            'Raw source brain/target masks overlap heavily. Zero owned non-target soft-tissue removal does not mean zero overlap with the raw unreviewed brain annotation.',
            'Engineering fixtures do not establish patient applicability, calibrated forces, clinical injury, safety or outcomes.']}
    destination = BASE / 'review' / ('tool-family-inventory-current-' + created + '.json')
    with destination.open('x') as stream: stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    assert sha(SOURCE) == EXPECTED_SOURCE and sha(old_inventory) == old_hash
    summary = {'inventory': str(destination), 'tools': len(tools),
        'patientEffectTools': [tool for tool in effects],
        'primary': {key: episodes['primary'][key] for key in ('actionCount', 'stateId', 'directorStatus')},
        'primaryEndpoint': episodes['primary']['endpointEffects'],
        'vascular': {key: episodes['vascular'][key] for key in ('actionCount', 'stateId', 'directorStatus')}}
    print(json.dumps(summary))


if __name__ == '__main__': main()
