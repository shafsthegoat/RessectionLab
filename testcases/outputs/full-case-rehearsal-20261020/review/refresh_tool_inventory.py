#!/usr/bin/env python3
"""Read saved native/receipt evidence only; never replay physics or use a GUI."""
from collections import Counter
import argparse
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
    participants = []
    for index, action in enumerate(actions):
        slot, operation = action['slot'], action['operation']
        if operation == 'exchange': tools[slot] = action['instrument']
        tool = tools[slot]
        effective = operation in ('apply', 'hold', 'grasp', 'extract', 'activate') or operation == 'move' and active[slot] or operation == 'inspect' and TOOLS[tool]['effect'] == 'probe'
        participants.append({hand: candidate for hand, candidate in tools.items()
                             if (hand == slot and effective) or (hand != slot and active[hand])})
        row = per_tool[tool]
        row['scheduled'].append(index + 1)
        row['operations'][operation] += 1
        if effective:
            row['actuation'].append(index + 1)
        for other in tools:
            if other != slot and active[other]:
                per_tool[tools[other]]['sharedClockExposure'].append(index + 1)
        if operation == 'activate': active[slot] = True
        if operation in ('deactivate', 'release', 'withdraw'): active[slot] = False
    for row in per_tool.values(): row['operations'] = dict(row['operations'])
    return per_tool, participants


def endpoint(scene, state, case):
    assert scene.shape == case.mri.shape and np.array_equal(scene.affine, case.affine)
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
        'removedSourceCellOriginalSourceCompartmentMemberships': {
            name: int(np.count_nonzero(mask.ravel()[removed])) for name, mask in (case.source_compartments or {}).items()},
        'removedSourceCellRawBrainMaskMembership': None if case.brain_mask is None else int(np.count_nonzero(case.brain_mask.ravel()[removed])),
        'sourceMembershipGridAndAffineVerified': True,
        'rawMembershipMeaning': 'Overlapping unreviewed annotation memberships, not clinical healthy tissue truth; do not change owned target denominator.',
        'drilling': {key: value for key, value in state.get('drillingV2', {}).items() if not isinstance(value, dict)},
        'debrisPresentMm3': state['debrisMm3'], 'debrisRemovedMm3': state.get('debrisRemovedMm3', 0.),
        'debrisEscapedMm3': state.get('debrisEscapedMm3', 0.),
        'boneDebrisConservationResidualMm3': state.get('drillingV2', {}).get('removedVolumeMm3', 0.) - state['debrisMm3'] - state.get('debrisRemovedMm3', 0.) - state.get('debrisEscapedMm3', 0.),
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, relative in [('primary', 'primary-v2/primary.ressectionlab'), ('vascular', 'vascular-v2/vascular.ressectionlab')]:
        parser.add_argument('--' + name + '-native', type=Path, default=BASE / relative,
                            help='Native path; both roles may select different branches of one file.')
        parser.add_argument('--' + name + '-branch', help='Branch ID; defaults to archive branchId.')
        parser.add_argument('--' + name + '-telemetry-receipt', type=Path,
                            help='Existing receipt; exact scene/actions/endpoint must match even if combined archive hash differs.')
    parser.add_argument('--output', type=Path, help='New JSON output; existing files are never overwritten.')
    return parser.parse_args(argv)


def select_branch(record, branch=None):
    selected = record['branchId'] if branch is None else branch
    if selected not in record['branches']: raise ValueError('Requested branch is absent: ' + selected)
    snapshot = record['branches'][selected]; checkpoint = snapshot['checkpoint']
    if checkpoint['position'] != len(snapshot['actions']) or checkpoint['stateId'] != snapshot['stateId']:
        raise ValueError('Checkpoint boundary differs from selected branch')
    return selected, snapshot


EVENT_EFFECTS = {'drill_work': {'remove'}, 'removal': {'remove'}, 'strand_transection': {'transect'},
    'local_energy': {'coagulate'}, 'irrigation': {'irrigate'}, 'aspiration': {'suction'},
    'incision': {'incise'}, 'sheet_opening': {'sheet_cut'}, 'bone_kerf': {'kerf'},
    'attachment_release': {'elevate', 'dissect'}, 'grasp_created': {'grasp'},
    'detached_extraction': {'grasp'}, 'mapping': {'probe'}}


def attach_telemetry(entry, snapshot, participants, path, *, explicit=False):
    if not path.exists(): return
    receipt = json.loads(path.read_text())
    binding = (receipt.get('status') == 'passed' and receipt.get('expectedEndpointVerified') is True
        and receipt.get('sourceSha256') == EXPECTED_SOURCE and receipt.get('sceneId') == snapshot['sceneId']
        and receipt.get('finalStateId') == snapshot['stateId']
        and (receipt.get('nativeSha256') == entry['nativeSha256'] or explicit))
    if not binding:
        entry['unmatchedTelemetryReceipt'] = {'path': str(path), 'reason': 'Source/scene/endpoint/native binding mismatch.'}
        return
    telemetry = Path(receipt['paths']['json'])
    matches = [episode for episode in json.loads(telemetry.read_text())['episodes']
        if episode.get('finalStateId') == snapshot['stateId'] and episode.get('sceneId') == snapshot['sceneId']
        and episode.get('expectedStateVerified') is True
        and [row['action'] for row in episode['attempts']] == snapshot['actions']]
    if len(matches) != 1:
        entry['unmatchedTelemetryReceipt'] = {'path': str(path), 'reason': 'No unique exact branch-action/endpoint match.'}
        return
    observed = matches[0]
    evidence = {'receipt': str(path), 'receiptSha256': sha(path), 'json': str(telemetry), 'jsonSha256': sha(telemetry),
        'scope': 'Existing observed public replay matches selected source/scene/actions/endpoint; no replay performed by refresh.',
        'archiveBinding': 'same-native-sha256' if receipt.get('nativeSha256') == entry['nativeSha256'] else 'explicit-receipt-identical-branch-content-in-different-native',
        'receiptNativeSha256': receipt.get('nativeSha256'), 'episodeId': observed.get('episodeId'),
        'statuses': dict(Counter(row['status'] for row in observed['attempts'])),
        'partialActions': [{'actionPosition1Based': row['actionIndex'] + 1, 'reason': row.get('reason')}
                           for row in observed['attempts'] if row['status'] == 'partial']}
    assigned = {tool: [] for tool in TOOLS}; unassigned = Counter()
    for event in observed['events']:
        if event['timing'] != 'committed_microstep' or event['event']['kind'] == 'no_material_effect': continue
        kind = event['event']['kind']
        compatible = {tool for tool in participants[event['actionIndex']].values()
                      if TOOLS[tool]['effect'] in EVENT_EFFECTS.get(kind, set())}
        # An event does not consistently name its hand. Retain only unambiguous
        # compatible effective families, including stationary active other hands.
        if len(compatible) != 1:
            unassigned[kind] += 1; continue
        assigned[compatible.pop()].append(event)
    evidence['unattributedOrAmbiguousEventKinds'] = dict(unassigned)
    for tool, use in entry['toolUses'].items():
        events = assigned[tool]
        use['observedEvents'] = {'eventKinds': dict(Counter(row['event']['kind'] for row in events)),
            'actionPositions1Based': sorted({row['actionIndex'] + 1 for row in events}), 'totalsByEventKind': {},
            'attribution': 'Unique compatible effective family among acting and stationary active hands; ambiguous events omitted.'}
        for kind, fields in {'drill_work': ('removedVolumeMm3', 'inputEnergyJ', 'cuttingWorkJ'),
            'removal': ('cells', 'volumeMm3'), 'strand_transection': ('edges',), 'local_energy': ('exposure',),
            'irrigation': ('volumeMm3',), 'aspiration': ('fluid', 'debris')}.items():
            matching = [row['event'] for row in events if row['event']['kind'] == kind]
            if matching:
                use['observedEvents']['totalsByEventKind'][kind] = {field: sum(row.get(field, 0.) for row in matching) for field in fields}
    entry['verifiedTelemetry'] = evidence


def main(argv=None):
    args = parse_args(argv)
    assert sha(SOURCE) == EXPECTED_SOURCE
    old_inventory = BASE / 'vascular/tool-family-inventory.json'; old_hash = sha(old_inventory)
    fixture = BASE / 'vascular/tool-family-fixture.xml'
    fixture_tests = ET.parse(fixture).getroot().findall('.//testcase')
    source_case = load_case(SOURCE); scene = load_environment(SOURCE)
    episodes, archives = {}, {}
    for name in ('primary', 'vascular'):
        path = getattr(args, name + '_native').expanduser().resolve()
        if path not in archives: archives[path] = saved(path)
        record, digest = archives[path]
        branch, snapshot = select_branch(record, getattr(args, name + '_branch'))
        assert snapshot['sceneId'] == scene.identity
        meta = record.get('fullCase', {}).get(branch, {})
        uses, participants = tool_actions(snapshot['actions'])
        entry = {'native': str(path), 'nativeSha256': digest, 'branch': branch,
            'actionCount': len(snapshot['actions']), 'stateId': snapshot['stateId'],
            'directorStatus': meta.get('status'), 'directorStage': meta.get('stage'),
            'directorStopReason': meta.get('stopReason'),
            'endpointSource': 'Saved checkpoint; no independent replay or readmission by this refresh.',
            'endpointEffects': endpoint(scene, snapshot['checkpoint']['state'], source_case),
            'toolUses': uses, 'verifiedTelemetry': None,
            'savedUncommittedProposalReasons': dict(Counter(row.get('reason') for row in meta.get('rejections', [])))}
        receipt_path = path.parent / 'execution-receipt.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        entry['matchingExecutionReceipt'] = {'path': str(receipt_path),
            'sha256': sha(receipt_path) if receipt else None,
            'matchesEndpoint': bool(receipt and receipt.get('stateId') == snapshot['stateId'] and receipt.get('actions') == len(snapshot['actions']))}
        explicit = getattr(args, name + '_telemetry_receipt')
        telemetry_path = explicit.expanduser().resolve() if explicit else path.parent / 'telemetry/export-receipt.json'
        attach_telemetry(entry, snapshot, participants, telemetry_path, explicit=explicit is not None)
        episodes[name] = entry
    limitations = {
        'periosteal_elevator': 'No declared periosteal attachment properties; no patient separation claim.',
        'dissector': 'No declared soft/tumor attachment properties; no patient dissection/separation claim.',
        'dural_scissors': 'No supplied membrane/dura anatomy; no patient dural opening claim.',
        'craniotome': 'No declared bone anchors establish flap separability; no patient craniotome/flap claim.',
        'brain_retractor': 'Supplied owned soft tissue does not establish patient-specific calibrated retraction; inspect selected action/effect evidence.',
        'tumor_forceps': 'Free-fragment extraction requires explicit action/effect evidence; attachments remain unknown.',
        'cortical_probe': 'No supplied functionalEvidence; no patient functional response or localization claim.',
        'subcortical_probe': 'No supplied functionalEvidence; no patient functional response or localization claim.'}
    tools = []
    for tool, model in TOOLS.items():
        case = next(row for row in fixture_tests if '[' + tool + '-' in row.attrib['name'])
        assert not list(case)
        patient = {name: episode['toolUses'][tool] for name, episode in episodes.items() if episode['toolUses'][tool]['actuation'] or episode['toolUses'][tool]['sharedClockExposure']}
        observed_effects = {name: use['observedEvents'] for name, use in patient.items()
            if any(value > 0. for totals in use.get('observedEvents', {}).get('totalsByEventKind', {}).values() for value in totals.values())
            or any(kind not in ('drill_work', 'removal', 'strand_transection', 'local_energy', 'irrigation', 'aspiration')
                   and count > 0 for kind, count in use.get('observedEvents', {}).get('eventKinds', {}).items())}
        status = ('actual patient actions and matching observed effect receipts' if observed_effects else
                  'actual patient actions; endpoint ledgers reported separately without per-tool causal attribution' if patient else
                  'engineering fixture proof only; no actuation in selected patient branches')
        tools.append({'tool': tool, 'effectFamily': model['effect'],
            'patientEvidenceStatus': status, 'patientProof': {
                'observedEffectsBySelectedEpisode': observed_effects,
                'endpointLedgerReferences': {name: 'episodes.' + name + '.endpointEffects' for name in patient},
                'qualification': 'Actions, field ledgers and uniquely attributable observed effects are separate evidence; no clinical outcome inferred.'},
            'patientActionEvidence': patient,
            'engineeringFixtureEvidence': {'status': 'passed-existing-receipt',
                'test': 'tests/test_surgical_tools.py::' + case.attrib['name'],
                'xml': str(fixture), 'xmlSha256': sha(fixture),
                'qualification': 'Explicit synthetic anatomy/properties; prior saved execution, not rerun during this refresh.'},
            'patientEvidenceLimit': limitations.get(tool, 'Declared uncalibrated research laws; no clinical force, blood-loss, safety, or outcome claim.')})
    created = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report = {'schema': 'resectionlab.tool-family-evidence-inventory.v2', 'createdUtc': created,
        'scope': 'Read-only inventory of explicitly selected saved branches; no physics replay, tests, GUI, source edits or treatment performed by this refresh.',
        'source': str(SOURCE), 'sourceSha256': EXPECTED_SOURCE, 'sceneId': scene.identity,
        'generatorScriptSha256': sha(Path(__file__)),
        'sourceCounts': {'ownedTargetCells': sum(int(np.count_nonzero(scene.labels == row['id'])) for row in scene.regions if row.get('target')),
            'rawCompartments': {name: int(np.count_nonzero(mask)) for name, mask in source_case.compartments.items()},
            'rawBrainMaskCells': None if source_case.brain_mask is None else int(np.count_nonzero(source_case.brain_mask)),
            'qualification': 'Overlapping raw annotation memberships are not clinical healthy-tissue truth and do not change the owned-target denominator.'},
        'supersedesForCurrentEvidenceOnly': {'path': str(old_inventory), 'sha256': old_hash, 'oldFilePreserved': True},
        'indexConvention': 'All action position lists are 1-based saved branch positions; subtract 1 for zero-based actionIndex.',
        'episodes': episodes, 'toolCount': len(tools), 'tools': tools,
        'anatomyEvidence': {'suppliedOwnedMaterials': sorted({row['material'] for row in scene.regions}),
            'membraneSupplied': any(row['material'] == 'membrane' for row in scene.regions),
            'declaredAttachments': len(scene.properties.get('attachments', [])),
            'declaredBoneAnchors': len(scene.properties.get('boneAnchors', [])),
            'declaredFunctionalEvidence': len(scene.properties.get('functionalEvidence', [])),
            'scalpDuraAttachmentAndFunctionalCompleteness': 'UNKNOWN; absent evidence was not fabricated',
            'vascularMeaning': 'Supplied unreviewed vessel masks and recorded modeled wound faces do not establish a complete anatomical network, perfusion, or calibrated patient physiology.'},
        'guiReview': 'Not performed by this inventory; no GUI pass inferred from numeric receipts.',
        'limitations': ['Director status/stage/stop reason are copied per selected branch; retained owned target is explicitly measured, without inferring complete tumor removal.',
            'Checkpoint endpoint values are recorded simulation state; independent canonical replay status is only claimed where a matching existing receipt is identified.',
            'Saved branch actions omit rejected attempts; partial counts are reported only from matching observed receipts, and uncommitted proposal reasons remain separate.',
            'Raw source brain/target masks overlap heavily. Zero owned non-target soft-tissue removal does not mean zero overlap with the raw unreviewed brain annotation.',
            'Engineering fixtures do not establish patient applicability, calibrated forces, clinical injury, safety or outcomes.']}
    destination = args.output.expanduser().resolve() if args.output else BASE / 'review' / ('tool-family-inventory-current-' + created + '.json')
    with destination.open('x') as stream: stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    assert sha(SOURCE) == EXPECTED_SOURCE and sha(old_inventory) == old_hash
    summary = {'inventory': str(destination), 'tools': len(tools),
        'patientActionTools': [row['tool'] for row in tools if row['patientActionEvidence']],
        'patientObservedEffectTools': [row['tool'] for row in tools if row['patientProof']['observedEffectsBySelectedEpisode']],
        'primary': {key: episodes['primary'][key] for key in ('actionCount', 'stateId', 'directorStatus')},
        'primaryEndpoint': episodes['primary']['endpointEffects'],
        'vascular': {key: episodes['vascular'][key] for key in ('actionCount', 'stateId', 'directorStatus')}}
    print(json.dumps(summary))


if __name__ == '__main__': main()
