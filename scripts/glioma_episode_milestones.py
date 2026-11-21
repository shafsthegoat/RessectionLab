"""Physical milestones collected from accepted engine boundaries, independent of the demo script."""

from collections import Counter, defaultdict
from hashlib import sha256

import numpy as np
from scipy.spatial.transform import Rotation

from resectionlab.surgical_sequence_outcomes import original_target_residual_mm3


def id_digest(cells):
    return sha256(np.array(sorted(cells), dtype='<u8').tobytes()).hexdigest()


class Milestones:
    def __init__(self, scene):
        self.source = scene.properties['operativeExposureV1']['bodies']
        self.events = {}
        self.last_bodies = {}
        self.last_ports = {}
        self.last_objects = {}
        self.bimanual = []
        self.partials = []
        self.bone = {}
        self.removed_by_tool = defaultdict(set)
        self.port_changes = []
        self.inventory = []
        self.peaks = {'brainMaxDisplacementMm': 0.0}
        self.source_residual = None
        self.initial_bodies = {}

    def mark(self, name, index, **measurement):
        row = self.events.setdefault(name, {'first': index, 'last': index, 'actions': 0})
        if row['last'] != index or not row['actions']:
            row['actions'] += 1
        row['last'] = index
        if measurement:
            row['measurement'] = measurement

    def boundary(self, engine, index):
        if not index:
            self.source_residual = original_target_residual_mm3(engine)
            for name, body in engine.state['operativeExposureV1']['bodies'].items():
                pose = (body['angleRad'], tuple(body['translationMm']), tuple(body['quaternion']))
                self.last_bodies[name] = (body['status'], body['heldBy'], pose)
                self.initial_bodies[name] = {'status':body['status'], 'cutEdges':len(body['cut']),
                                             'sourceCells':len(self.source[name]['cells'])}
            return
        history = engine.history[-1]
        action = history['action']
        state = engine.state
        actor = action['slot']
        slots = state['slots']
        tool = slots[actor]['instrument']
        if history['status'] == 'partial':
            self.partials.append({'action':index, 'hand':actor, 'instrument':tool, 'operation':action['operation'],
                                  'reason':history['reasonCode'], 'fraction':history['completedFraction']})
        self.removed_by_tool[tool].update(history['deltas']['removedCells'])
        body_state = state['operativeExposureV1']
        for name, body in body_state['bodies'].items():
            pose = (body['angleRad'], tuple(body['translationMm']), tuple(body['quaternion']))
            now = (body['status'], body['heldBy'], pose)
            old = self.last_bodies.get(name)
            if old and old[2] != pose:
                if actor not in (body['heldBy'], old[1]):
                    raise ValueError(f'{name} moved without acting holder at action {index}')
                self.mark(name + '-motion', index, hand=actor, tool=tool, angleRad=pose[0], translationMm=pose[1])
            if not old or now[:2] != old[:2]:
                self.mark(
                    name + '-' + body['status'],
                    index,
                    heldBy=body['heldBy'],
                    closedAnchors=len(body['closedAnchors']),
                )
            self.last_bodies[name] = now
        object_ids = {obj['id'] for obj in body_state['objects']}
        if not self.last_objects.keys() <= object_ids or len(object_ids) != len(body_state['objects']):
            raise ValueError(f'Lost or duplicated persistent item at action {index}')
        for item in body_state['objects']:
            identity = item['id']
            current = (item['kind'], item['body'], item['status'], item.get('sourceCell'))
            old = self.last_objects.get(identity)
            if old and old[3] != current[3]:
                raise ValueError(f'{identity} lost its source cell at action {index}')
            if item['status'] == 'carried':
                hand = slots[item['heldBy']]
                point = np.asarray(hand['position']) + Rotation.from_quat(hand['quaternion']).apply(item['offset'])
                if not np.allclose(point, item['positionMm'], atol=1e-8, rtol=0):
                    raise ValueError(f'{identity} left its carrying hand at action {index}')
            if current != self.last_objects.get(identity):
                self.inventory.append(
                    {
                        'action': index,
                        'id': identity,
                        'kind': item['kind'],
                        'body': item['body'],
                        'status': item['status'],
                        'sourceCell': item.get('sourceCell'),
                    }
                )
                self.mark(item['body'] + '-' + item['kind'] + '-' + item['status'], index, objectId=identity)
                self.last_objects[identity] = current
        other = 'left' if actor == 'right' else 'right'
        if slots[actor]['inserted'] and slots[other]['inserted']:
            hold = state['holds'].get(other)
            engaged = [
                event['kind']
                for event in history['events']
                if event['kind'] in ('attachment_release', 'incision', 'cusa_contact', 'strand_transection')
            ]
            if hold and hold.get('cells') and engaged:
                self.bimanual.append(
                    {
                        'action': index,
                        'active': [actor, tool],
                        'holding': [other, slots[other]['instrument']],
                        'heldCells': len(hold['cells']),
                        'events': sorted(set(engaged)),
                        'poses': {s: {k: t[k] for k in ('position', 'quaternion')} for s, t in slots.items()},
                    }
                )
        for event in history['events']:
            kind = event['kind']
            if kind == 'exposure_seam_cut':
                self.mark(event['body'] + '-incision', index, uncutEdges=event['remainingEdges'])
            elif kind == 'drill_work' and event['cuttingWorkJ'] > 0:
                row = self.bone.setdefault(tool, {'inputEnergyJ': 0.0, 'workJ': 0.0, 'cells': set()})
                row['inputEnergyJ'] += event['inputEnergyJ']
                row['workJ'] += event['cuttingWorkJ']
                row['cells'].update(event['removedCells'])
                self.mark(tool + '-bone-work', index)
            elif kind == 'mapping' and event['cells']:
                self.mark('mapping-' + event['result'], index, probe=event['probe'], sourceCells=event['cells'])
            elif kind == 'detached_extraction' and event['cells'] and event.get('checkedFragmentMotion'):
                self.mark(
                    'exterior-sample', index, cells=history['deltas']['removedCells'], volumeMm3=event['volumeMm3']
                )
            elif kind == 'cusa_contact' and event['aspiratedSolidMm3'] > 0:
                self.mark('cusa-solid-removal', index)
            elif kind in ('irrigation', 'aspiration') and any(
                v > 0 for k, v in event.items() if k in ('volumeMm3', 'fluid')
            ):
                self.mark(kind, index)
            elif kind == 'attachment_release' and event['attachments']:
                self.mark('attachment-release', index, attachments=event['attachments'])
        from resectionlab.surgical_vascular_graph import consequence_evidence

        vascular = consequence_evidence(engine)
        wounds = {w['id']: w for edge in vascular['segments'].values() for w in edge['wounds']}
        for key, port in state['vascularGraphV1']['ports'].items():
            current = {
                k: port[k]
                for k in ('cell', 'edge', 'generatedMm3', 'sealFraction', 'coagulationDose', 'compressionFraction')
            }
            current['leakMm3PerSecond'] = wounds[key]['modeledFlowMm3PerSecond']
            old = self.last_ports.get(key, {})
            if (
                not old
                or current['sealFraction'] > old['sealFraction']
                or current['compressionFraction'] != old['compressionFraction']
            ):
                self.port_changes.append({'action': index, 'port': key, 'tool': tool, 'was': old, 'now': current})
                if (
                    old
                    and tool == 'raney_clip_applier'
                    and current['edge'] == 'scalp'
                    and old['leakMm3PerSecond'] > 0
                    and current['leakMm3PerSecond'] == 0
                    and current['compressionFraction'] > old['compressionFraction']
                ):
                    self.mark('raney-compression-stopped-scalp-leak', index, port=key, before=old, after=current)
                if (
                    tool == 'bipolar_forceps'
                    and old
                    and old['generatedMm3'] > 0
                    and current['coagulationDose'] > old['coagulationDose']
                ):
                    self.mark('bipolar-seal-after-source-bleed', index, port=key, **current)
            self.last_ports[key] = current
        self.peaks['brainMaxDisplacementMm'] = max(
            self.peaks['brainMaxDisplacementMm'], engine.metrics().get('brainMaxDisplacementMm', 0.0)
        )
        if action['operation'] == 'withdraw' and not slots[actor]['inserted']:
            self.mark(actor + '-withdrawn', index, tool=tool)

    def finish(self, engine):
        state, metrics = engine.state, engine.metrics()
        body = state['operativeExposureV1']['bodies']
        bone = {}
        port = set(self.source['bone']['portCells'])
        kerf = set(self.source['bone']['kerfCells'])
        flap = set(self.source['bone']['cells'])
        for name, row in self.bone.items():
            cells = row['cells']
            bone[name] = {
                'inputEnergyJ': row['inputEnergyJ'],
                'cuttingWorkJ': row['workJ'],
                'uniqueCells': len(cells),
                'sourceIdSHA256': id_digest(cells),
                'pilotPortCells': len(cells & port),
                'kerfCells': len(cells & kerf),
                'outsideDeclaredPortAndKerf': len(cells - port - kerf),
                'flapCellsRemoved': len(cells & flap),
            }
        remaining = original_target_residual_mm3(engine)
        objects = state['operativeExposureV1']['objects']
        source_removals = {}
        for tool, cells in self.removed_by_tool.items():
            if not cells:
                continue
            materials = Counter(engine.by_label[int(engine.scene.labels.flat[c])]['name'] for c in cells)
            target = sum(bool(engine.by_label[int(engine.scene.labels.flat[c])].get('target')) for c in cells)
            source_removals[tool] = {
                'uniqueCells': len(cells),
                'sourceIdSHA256': id_digest(cells),
                'ownedTargetMm3': target * engine.scene.voxel_mm3,
                'sourceLabels': dict(materials),
            }
        missing = [
            name
            for name in (
                'scalp-incision',
                'scalp-held',
                'scalp-motion',
                'scalp-supported',
                'scalp-raney-clip-placed',
                'raney-compression-stopped-scalp-leak',
                'cranial_drill-bone-work',
                'craniotome-bone-work',
                'bone-free',
                'bone-held',
                'bone-motion',
                'bone-parked',
                'dura-incision',
                'dura-held',
                'dura-motion',
                'dura-supported',
                'mapping-unknown',
                'attachment-release',
                'exterior-sample',
                'cusa-solid-removal',
                'bipolar-seal-after-source-bleed',
                'irrigation',
                'aspiration',
                'field-cottonoid-placed',
                'field-cottonoid-retrieved',
                'field-hemostat-intended-retained',
                'dura-closed',
                'bone-fixed',
                'scalp-closed',
            )
            if name not in self.events
        ]
        if not self.bimanual:
            missing.append('coordinated-working-hand-and-holding-hand')
        if any(t['active'] or t['inserted'] for t in state['slots'].values()):
            missing.append('both-instruments-withdrawn-and-inactive')
        if any(o['status'] not in ('retrieved', 'intended-retained') for o in objects):
            missing.append('object-inventory-closed')
        if any(row['outsideDeclaredPortAndKerf'] or row['flapCellsRemoved'] for row in bone.values()):
            missing.append('bone-work-source-locality')
        if not flap.isdisjoint(state['removed']):
            missing.append('rigid-flap-material-preserved')
        if source_removals.get('cusa', {}).get('ownedTargetMm3', 0.0) <= 0:
            missing.append('cusa-removed-target-material')
        sequence = [
            'scalp-incision',
            'scalp-supported',
            'cranial_drill-bone-work',
            'craniotome-bone-work',
            'bone-parked',
            'dura-incision',
            'dura-supported',
            'mapping-unknown',
            'dura-closed',
            'bone-fixed',
            'scalp-closed',
        ]
        observed = [(key, self.events[key]['first']) for key in sequence if key in self.events]
        if any(b < a for (_, a), (_, b) in zip(observed, observed[1:])):
            missing.append('causal-phase-order')
        return {
            'satisfied': not missing,
            'missing': missing,
            'orderedExposureAndClosure': observed,
            'events': self.events,
            'initialBodies': self.initial_bodies,
            'bimanual': self.bimanual,
            'partialActions': self.partials,
            'sourceBoundBoneWork': bone,
            'sourceRemovalsByActingInstrument': source_removals,
            'intactBoneFlapCells': len(flap - set(state['removed'])),
            'sourceBoneFlapCells': len(flap),
            'initialSourceTargetMm3': self.source_residual,
            'finalSourceTargetMm3': remaining,
            'ownedTargetRemovedMm3': metrics['targetRemovedMm3'],
            'otherRemovedMm3': metrics['nonTargetRemovedMm3'],
            'mappedObservations': state['observations'],
            'peakDeformation': self.peaks,
            'sourceSpecificPortChanges': self.port_changes,
            'finalPorts': self.last_ports,
            'finalVesselEdges': state['vascularGraphV1']['edges'],
            'vascularConservationResidualMm3': metrics['vesselPhysicsV2']['conservationResidualMm3'],
            'placedInventory': self.inventory,
            'finalObjectCounts': dict(Counter(o['kind'] + ' ' + o['status'] for o in objects)),
            'finalBodies': {
                name: {k: v[k] for k in ('status', 'heldBy', 'closedAnchors', 'translationMm', 'angleRad')}
                for name, v in body.items()
            },
            'finalHands': {
                hand: {k: v[k] for k in ('instrument', 'inserted', 'active', 'position', 'quaternion')}
                for hand, v in state['slots'].items()
            },
            'fluid': {
                key: metrics[key]
                for key in (
                    'fluidAddedMm3',
                    'fluidRemovedMm3',
                    'fluidPresentMm3',
                    'fluidEscapedMm3',
                    'debrisMm3',
                    'debrisRemovedMm3',
                )
            },
        }
