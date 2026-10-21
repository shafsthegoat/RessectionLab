#!/usr/bin/env python3
"""Independent native acceptance audit; never supplies or changes patient state.

One canonical public restore, cache reads disabled, observes every real transition
and may seal its independently verified states for later fresh-process opens.
Residual regions use six-neighbor ORIGINAL SOURCE ownership, not displaced
display geometry or an inference that no further legal approach exists.
"""
from __future__ import annotations
import argparse
import copy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'src/resectionlab').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from scipy import ndimage
from resectionlab.core import array_digest, semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_verified_cache import VerifiedSurgicalCache, implementation_fingerprint
from resectionlab.workspace_bundle import load_workspace

SOURCE = ROOT / 'testcases/cases/meningioma-sub-06-standard-v2.ressectionlab'
SOURCE_SHA256 = 'ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838'
SCENE_ID = 'sha256:7a5d368382fe2a66192bdef88ff6af3a55cd533c9ba3787c68a1d1e3cb48f856'
PROTECTION = ROOT / 'testcases/environments/meningioma-sub-06/rl-protection.json'
PROTECTION_MANIFEST_ID = 'sha256:d7abac452a30c7487cd52d7b3f841466bf16b66ad61e0901afcd0a051845d0c5'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def array_hashes(case):
    result = {'mri': array_digest(case.mri), 'affine': array_digest(case.affine)}
    if case.brain_mask is not None:
        result['brain_mask'] = array_digest(case.brain_mask)
    for group in ('compartments', 'source_compartments'):
        result[group] = {k: array_digest(v) for k, v in getattr(case, group).items()}
    return result


def originals(workspace):
    return [{'name': row.name, 'sourceUri': row.source_uri, 'role': row.role,
             'sha256': file_bytes_hash(row.data), 'bytes': len(row.data)}
            for row in (workspace or {}).get('originals', ())]


def file_bytes_hash(data):
    return hashlib.sha256(data).hexdigest()


def protection_audit(scene, snapshot):
    """Compare every protected source cell to the existing patient evidence."""
    from resectionlab.surgical_rl_protection import GuardConfig, load_protection, protection_contract
    document = json.loads(PROTECTION.read_text())
    require(document['contentHash'] == PROTECTION_MANIFEST_ID, 'Patient protection manifest differs')
    protected = load_protection(scene, PROTECTION, strict=True)
    expected = protection_contract(protected, GuardConfig())
    require(len(protected.annotations) == 11 and protected.clearance_mm == 2., 'Patient mask coverage/clearance differs')
    require(snapshot.get('protectionContract') == expected, 'Patient exact protection contract differs')
    return {'manifest': str(PROTECTION), 'manifestSha256': file_hash(PROTECTION),
            'arraysSha256': file_hash(PROTECTION.with_name(document['arrays'])),
            'sourceBinding': document['sourceBinding'], 'contractId': expected['contentHash'],
            'matchingMasks': len(protected.annotations), 'clearanceMm': protected.clearance_mm,
            'criticalMaxDisplacementMm': 0., 'exactCellsAndProvenanceMatch': True}


def vascular_sample(engine, position):
    vessel = engine.state.get('vesselsV2')
    return {'position': position, 'elapsedSeconds': engine.state['elapsedSeconds'],
            'slots': copy.deepcopy(engine.state['slots']),
            'faces': copy.deepcopy(vessel['faces']) if vessel else {},
            'generatedMm3': vessel['generatedMm3'] if vessel else 0.,
            'activeWounds': sum(s['bleedingActive'] for s in engine.state['sources'] if s.get('modelVersion') == 2),
            'presentMm3': math.fsum(engine.state['fluid'].values()),
            'addedMm3': engine.state['fluidAddedMm3'], 'removedMm3': engine.state['fluidRemovedMm3']}


def causal_vascular_audit(samples, actions, dose, *, required=False):
    """Use observed transition states, never director phase names or receipts."""
    injury, idle_bleed, bipolar, sealed_idle, irrigation, suction = [], [], [], [], [], []
    for before, after, action in zip(samples, samples[1:], actions):
        position = after['position']
        instrument = before['slots'][action['slot']]['instrument']
        operation = action['operation']
        a, b = before['faces'], after['faces']
        if not a and b:
            require(operation == 'apply', 'First injury did not arise from an actual active tool path')
            injury.append({'position': position, 'instrument': instrument, 'sourceFaceIds': sorted(b)})
        if a and operation == 'inspect' and not any(s['active'] for s in before['slots'].values()) and after['generatedMm3'] > before['generatedMm3'] + 1e-10:
            idle_bleed.append({'position': position, 'generatedDeltaMm3': after['generatedMm3'] - before['generatedMm3'],
                               'presentDeltaMm3': after['presentMm3'] - before['presentMm3']})
        if instrument == 'bipolar_forceps' and operation == 'apply' and a:
            actual_increase = math.fsum(f['coagulationDose'] for f in b.values()) - math.fsum(f['coagulationDose'] for f in a.values())
            if actual_increase > 1e-10:
                bipolar.append({'position': position, 'actualFaceCoagulationDoseIncrease': actual_increase})
        sealed = bool(b) and all(f['clot'] >= 1. - 1e-12 and f['coagulationDose'] >= dose - 1e-12 for f in b.values())
        if sealed and operation == 'inspect' and not any(s['active'] for s in before['slots'].values()):
            require(math.isclose(after['generatedMm3'], before['generatedMm3'], abs_tol=1e-10, rel_tol=0.) and after['activeWounds'] == 0,
                    'Fully bipolar-sealed wound continued bleeding during idle time')
            sealed_idle.append(position)
        if instrument == 'irrigation' and operation == 'apply' and after['addedMm3'] > before['addedMm3'] + 1e-10:
            irrigation.append(position)
        if instrument == 'frazier_suction' and operation == 'apply' and after['removedMm3'] > before['removedMm3'] + 1e-10:
            require({k: (v['damage'], v['patent']) for k, v in a.items()} ==
                    {k: (v['damage'], v['patent']) for k, v in b.items()}, 'Suction repaired vessel damage/patency')
            require(after['generatedMm3'] >= before['generatedMm3'], 'Suction erased historical blood loss')
            suction.append({'position': position, 'removedDeltaMm3': after['removedMm3'] - before['removedMm3']})
    chain = bool(injury and idle_bleed and bipolar and sealed_idle and irrigation and suction)
    if chain:
        chain = (injury[0]['position'] < idle_bleed[0]['position'] < bipolar[0]['position'] < sealed_idle[0]
                 and bipolar[0]['position'] < min(irrigation[0], suction[0]['position']))
    if required:
        require(chain, 'Required time-ordered patient vascular injury/rescue chain is incomplete')
        require(samples[0]['generatedMm3'] == 0. and not samples[0]['faces'], 'Vascular branch did not start intact')
        require(samples[-1]['activeWounds'] == 0 and samples[-1]['faces'] and
                all(f['clot'] >= 1. - 1e-12 and f['coagulationDose'] >= dose - 1e-12 and not f['patent']
                    for f in samples[-1]['faces'].values()), 'Final wound lacks actual bipolar dose, full clot or nonpatency')
    return {'required': required, 'completeChain': chain, 'injury': injury,
            'passivePersistentBleeding': idle_bleed, 'actualBipolarDose': bipolar,
            'sealedIdlePositions': sealed_idle, 'irrigationPositions': irrigation,
            'suctionWithoutVesselRepair': suction}


def fluid_audit(state, initial_offset=0.):
    amounts = [*state['fluid'].values(), *(s.get('reservoirMm3', 0.) for s in state['sources']),
               state['fluidAddedMm3'], state['fluidRemovedMm3'], state['fluidEscapedMm3']]
    require(all(math.isfinite(value) and value >= -1e-10 for value in amounts),
            'Fluid ledger contains a negative or nonfinite physical volume')
    present = math.fsum(state['fluid'].values())
    trapped = math.fsum(s.get('reservoirMm3', 0.) for s in state['sources'])
    accounted = math.fsum([present, trapped, state['fluidRemovedMm3'], state['fluidEscapedMm3']])
    residual = state['fluidAddedMm3'] - accounted - initial_offset
    require(math.isclose(residual, 0., abs_tol=1e-8, rel_tol=0.), 'Fluid mass ledger differs')
    result = {'addedMm3': state['fluidAddedMm3'], 'presentMm3': present,
              'trappedOutletMm3': trapped, 'removedMm3': state['fluidRemovedMm3'],
              'escapedMm3': state['fluidEscapedMm3'], 'residualMm3': residual}
    vessel = state.get('vesselsV2')
    if vessel is not None:
        require(all(math.isfinite(vessel[name]) and vessel[name] >= -1e-10
                    for name in ('initialMm3', 'refilledMm3', 'remainingMm3', 'generatedMm3')),
                'Vascular reservoir contains a negative or nonfinite physical volume')
        vascular_residual = math.fsum([vessel['initialMm3'], vessel['refilledMm3'],
                                      -vessel['remainingMm3'], -vessel['generatedMm3']])
        require(abs(vascular_residual) < 1e-8, 'Vascular reservoir mass ledger differs')
        sources = [s for s in state['sources'] if s.get('modelVersion') == 2]
        emitted = math.fsum(s['emittedMm3'] for s in sources)
        require(math.isclose(emitted, vessel['generatedMm3'], abs_tol=1e-8, rel_tol=1e-8),
                'Generated blood and field emission differ')
        require(all(math.isclose(s['emittedMm3'], s['expectedEmittedMm3'],
                                 abs_tol=1e-8, rel_tol=1e-8) for s in sources),
                'Vascular-to-field handoff differs')
        result['vascular'] = {'initialMm3': vessel['initialMm3'],
                             'refilledMm3': vessel['refilledMm3'],
                             'remainingMm3': vessel['remainingMm3'],
                             'generatedMm3': vessel['generatedMm3'],
                             'emittedMm3': emitted, 'residualMm3': vascular_residual}
    return result


def drilling_debris_audit(state, scene, generated_debris_mm3=0.):
    profile = scene.properties.get('tissuePhysicsV2', {}).get('bone', {})
    drilling = state.get('drillingV2')
    result = {}
    if drilling is not None:
        require(all(math.isfinite(drilling[name]) and drilling[name] >= -1e-10
                    for name in ('inputEnergyJ', 'cuttingWorkJ', 'unusedEnergyJ', 'heatGeneratedJ',
                                 'coolingLossJ', 'removedVolumeMm3')),
                'Drilling ledger contains a negative or nonfinite physical quantity')
        pending = math.fsum(drilling['workJ'].values())
        threshold = profile['specificEnergyJPerMm3'] * scene.voxel_mm3
        require(all(math.isfinite(value) and 0 <= value < threshold + 1e-10
                    for value in drilling['workJ'].values()), 'Pending drill work is outside voxel threshold')
        require(not set(map(int, drilling['workJ'])).intersection(state['removed']),
                'Removed source cell retains pending drilling work')
        bone_ids = {r['id'] for r in scene.regions if r['material'] == 'bone'}
        require(all(0 <= int(cell) < scene.labels.size and int(scene.labels.flat[int(cell)]) in bone_ids
                    for cell in drilling['workJ']), 'Pending drilling work is not original bone ownership')
        energy = drilling['inputEnergyJ'] - drilling['cuttingWorkJ'] - drilling['unusedEnergyJ']
        work = drilling['cuttingWorkJ'] - drilling['removedVolumeMm3'] * profile['specificEnergyJPerMm3'] - pending
        partition = drilling['heatGeneratedJ'] - drilling['cuttingWorkJ'] * profile['heatFraction']
        stored_heat = math.fsum((value - profile['ambientTemperatureC']) * profile['heatCapacityJPerMm3K'] * scene.voxel_mm3
                               for value in drilling['temperatureC'].values())
        thermal = drilling['heatGeneratedJ'] - stored_heat - drilling['coolingLossJ']
        require(all(abs(value) < 1e-6 for value in (energy, work, partition, thermal)),
                'Independent drilling energy/work/thermal conservation differs')
        result['drilling'] = {'inputEnergyJ': drilling['inputEnergyJ'], 'cuttingWorkJ': drilling['cuttingWorkJ'],
            'unusedEnergyJ': drilling['unusedEnergyJ'], 'pendingWorkJ': pending,
            'removedVolumeMm3': drilling['removedVolumeMm3'], 'storedHeatJ': stored_heat,
            'heatGeneratedJ': drilling['heatGeneratedJ'], 'coolingLossJ': drilling['coolingLossJ'],
            'inputEnergyResidualJ': energy, 'voxelWorkResidualJ': work,
            'heatPartitionResidualJ': partition, 'thermalResidualJ': thermal,
            'absoluteToleranceJ': 1e-6}
    present = math.fsum(state.get('debris', {}).values())
    removed = state.get('debrisRemovedMm3', 0.)
    escaped = state.get('debrisEscapedMm3', 0.)
    require(all(math.isfinite(v) and v >= -1e-10 for v in (*state.get('debris', {}).values(), present, removed, escaped)),
            'Debris contains a negative or nonfinite volume')
    require(math.isclose(present, state['debrisMm3'], abs_tol=1e-8, rel_tol=1e-10), 'Published debris differs from spatial ledger')
    residual = generated_debris_mm3 - present - removed - escaped
    require(abs(residual) < 1e-7, 'Generated drill/kerf debris is not conserved')
    result['debris'] = {'generatedByActualDrillAndKerfEventsMm3': generated_debris_mm3,
        'presentMm3': present, 'removedMm3': removed, 'escapedMm3': escaped, 'residualMm3': residual}
    return result


def wound_audit(state, scene=None):
    vessel = state.get('vesselsV2')
    if vessel is None:
        return {'enabled': False}
    faces, wounds = vessel['faces'], vessel['wounds']
    sources = {s['id']: s for s in state['sources'] if s.get('modelVersion') == 2}
    grouped = [key for wound in wounds.values() for key in wound['faces']]
    require(len(grouped) == len(set(grouped)) and set(grouped) == set(faces),
            'Wound membership duplicates or omits a damaged face')
    require(set(wounds) == set(sources), 'Wounds and blood sources differ')
    severed = []
    face_parents = {key: key for key in faces}
    def root(key):
        while face_parents[key] != key:
            key = face_parents[key]
        return key
    endpoints = {}
    for key, face in faces.items():
        require(0 < face['damage'] <= 1 and 0 <= face['clot'] <= 1,
                'Wound damage/clot is outside physical range')
        require(face['woundId'] in wounds and key in wounds[face['woundId']]['faces'],
                'Face-to-wound link differs')
        if scene is not None:
            a, b = face['cells']
            require(key == f'{min(a, b)}:{max(a, b)}', 'Wound face source identity differs')
            require(all(0 <= cell < scene.labels.size for cell in (a, b)), 'Wound outside source volume')
            vessel_ids = {r['id'] for r in scene.regions if r['material'] == 'vessel'}
            require(all(int(scene.labels.flat[cell]) in vessel_ids for cell in (a, b)),
                    'Wound face is not supplied vessel ownership')
            ijk = [np.array(np.unravel_index(cell, scene.shape)) for cell in (a, b)]
            delta = np.abs(ijk[0] - ijk[1])
            require(int(delta.sum()) == 1 and face['axis'] == int(delta.argmax()), 'Wound is not a source-grid neighbor face')
            other = [axis for axis in range(3) if axis != face['axis']]
            area = float(np.linalg.norm(np.cross(scene.affine[:3, other[0]], scene.affine[:3, other[1]])))
            require(math.isclose(area, face['areaMm2'], abs_tol=1e-10, rel_tol=1e-10),
                    'Wound face area disagrees with source affine')
            for cell, index in zip((a, b), ijk):
                neighbors = [cell]
                for axis in range(3):
                    for step in (-1, 1):
                        adjacent = index.copy(); adjacent[axis] += step
                        if np.all(adjacent >= 0) and np.all(adjacent < scene.shape):
                            neighbors.append(int(np.ravel_multi_index(adjacent, scene.shape)))
                for neighbor in neighbors:
                    for other_key in endpoints.get(neighbor, []):
                        face_parents[root(key)] = root(other_key)
                endpoints.setdefault(cell, []).append(key)
        if face['damage'] >= 1.:
            severed.append(key)
            require(face['patent'] is False, 'Severed vessel became patent')
    for key, wound in wounds.items():
        require(wound['sourceId'] == key, 'Wound source identity differs')
        require(sources[key]['patent'] == all(faces[k]['patent'] for k in wound['faces']),
                'Source patency disagrees with damaged faces')
    if scene is not None:
        clusters = {}
        for key in faces: clusters.setdefault(root(key), set()).add(key)
        require({frozenset(group) for group in clusters.values()} ==
                {frozenset(wound['faces']) for wound in wounds.values()},
                'Independent adjacent-face wound clusters differ')
    if not faces:
        require(vessel['generatedMm3'] == 0., 'Uninjured vessels generated blood')
    return {'enabled': True, 'woundCount': len(wounds), 'damagedFaceCount': len(faces),
            'areaMm2': math.fsum(f['areaMm2'] for f in faces.values()),
            'severedFaceCount': len(severed), 'severedPatentCount': 0,
            'fullyClottedFaces': sum(f['clot'] >= 1. - 1e-12 for f in faces.values()),
            'activeWounds': sum(s['bleedingActive'] for s in sources.values()),
            'wounds': wounds, 'faces': faces, 'sources': list(sources.values())}


def source_accounting(scene, state):
    labels = scene.labels.ravel()
    removed = np.asarray(state['removed'], dtype=np.int64)
    require(len(set(map(int, removed))) == len(removed), 'Duplicate removed source IDs')
    require(not len(removed) or int(removed.min()) >= 0 and int(removed.max()) < labels.size,
            'Removed source ID outside original volume')
    require(not len(removed) or np.all(labels[removed] != 0), 'Removed empty source cells')
    counts = Counter(map(int, labels[removed]))
    materials = {}
    for region in scene.regions:
        row = materials.setdefault(region['material'], {'originalCells': 0, 'removedCells': 0})
        row['originalCells'] += int(np.count_nonzero(labels == region['id']))
        row['removedCells'] += counts[region['id']]
    for row in materials.values():
        row['retainedCells'] = row['originalCells'] - row['removedCells']
        for name in ('original', 'removed', 'retained'):
            row[name + 'Mm3'] = row[name + 'Cells'] * scene.voxel_mm3
    target_ids = [r['id'] for r in scene.regions if r.get('target')]
    residual = np.isin(scene.labels, target_ids)
    original = int(residual.sum())
    if len(removed): residual.ravel()[removed] = False
    components, count = ndimage.label(residual, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    regions = []
    for identifier, box in enumerate(ndimage.find_objects(components), start=1):
        if box is None: continue
        lo = [x.start for x in box]; hi = [x.stop - 1 for x in box]
        regions.append({'id': identifier, 'cells': int(sizes[identifier]),
                        'volumeMm3': int(sizes[identifier]) * scene.voxel_mm3,
                        'sourceIndexBoundsInclusive': [lo, hi]})
    regions.sort(key=lambda row: (-row['cells'], row['id']))
    retained = int(residual.sum())
    healthy_non_target_ids = {r['id'] for r in scene.regions
        if not r.get('target') and r['material'] in ('soft_tissue', 'unknown', 'functional_region', 'membrane')}
    owned_non_target_count = sum(counts[identifier] for identifier in healthy_non_target_ids)
    return {'materials': materials,
            'materialCountingMeaning': 'Exclusive OWNED source material after declared overlap precedence; not raw annotation membership or clinical healthy-tissue truth.',
            'ownedHealthyOrUnknownNonTargetRemoval': {'cells': owned_non_target_count,
                'volumeMm3': owned_non_target_count * scene.voxel_mm3,
                'materialLabels': ['soft_tissue', 'unknown', 'functional_region', 'membrane'],
                'excludesDeclaredTargets': True},
            'target': {'originalOwnedCells': original,
            'removedOwnedCells': original - retained, 'retainedOwnedCells': retained,
            'originalOwnedMm3': original * scene.voxel_mm3,
            'removedOwnedMm3': (original - retained) * scene.voxel_mm3,
            'removedOwnedPercent': 100. * (original - retained) / original if original else None,
            'retainedOwnedMm3': retained * scene.voxel_mm3,
            'residualConnectedRegions': regions, 'connectivity': 6,
            'coordinates': 'original source-grid ownership, independent of display displacement'}}


def protected_target_membership(scene, state, contract):
    """Direct source-cell membership only; never a swept-clearance inference."""
    if contract is None:
        return {'configured': False}
    targets = set(map(int, np.flatnonzero(np.isin(scene.labels,
        [region['id'] for region in scene.regions if region.get('target')]))))
    removed = set(state['removed'])
    union, structures = set(), []
    for annotation in contract['annotations']:
        direct = targets.intersection(annotation['cells'])
        retained = direct - removed
        union.update(direct)
        structures.append({'structureId': annotation['structureId'],
            'originalOwnedTargetDirectCells': len(direct),
            'retainedOwnedTargetDirectCells': len(retained),
            'removedOwnedTargetDirectCells': len(direct & removed),
            'originalOwnedTargetDirectMm3': len(direct) * scene.voxel_mm3,
            'retainedOwnedTargetDirectMm3': len(retained) * scene.voxel_mm3})
    retained = union - removed
    return {'configured': True, 'contractId': contract['contentHash'],
        'originalOwnedTargetDirectCells': len(union), 'retainedOwnedTargetDirectCells': len(retained),
        'removedOwnedTargetDirectCells': len(union & removed),
        'originalOwnedTargetDirectMm3': len(union) * scene.voxel_mm3,
        'retainedOwnedTargetDirectMm3': len(retained) * scene.voxel_mm3,
        'originalOwnedTargetDenominatorCells': len(targets),
        'originalOwnedTargetDirectPercent': 100. * len(union) / len(targets) if targets else None,
        'perStructureNonexclusive': structures,
        'meaning': 'Union of original OWNED target identities directly inside the exact supplied protection masks. Per-structure counts overlap. This does not change the original target denominator or establish swept-tool clearance, reachability, or that all legal approaches were exhausted. The 2 mm guard clearance is separate.'}


def annotation_memberships(case, scene, removed_ids):
    """Count overlapping original mask memberships using verified source IDs.

    These are annotation overlaps. A raw unreviewed brain mask overlapping a
    target is not a clinical assertion that removal reached healthy brain.
    """
    transform = np.asarray(case.affine)
    if case.frame == 'LPS+':
        transform = np.diag([-1., -1., 1., 1.]) @ transform
    require(tuple(case.mri.shape) == tuple(scene.shape) and np.array_equal(transform, scene.affine),
            'Raw annotation membership requires identical source grid and RAS affine')
    removed = np.asarray(removed_ids, dtype=np.int64)
    require(not len(removed) or int(removed.min()) >= 0 and int(removed.max()) < scene.labels.size,
            'Raw membership removed source ID outside verified grid')
    target_names = {r['name'] for r in scene.regions if r.get('target')}
    raw_target = np.zeros(scene.shape, dtype=bool)
    for name in target_names:
        if name in case.compartments:
            raw_target |= case.compartments[name]
    groups = {}
    for group in ('compartments', 'source_compartments'):
        groups[group] = {}
        for name, mask in getattr(case, group).items():
            require(mask.shape == scene.shape, 'Raw source annotation grid differs')
            original = int(np.count_nonzero(mask))
            removed_count = int(np.count_nonzero(mask.ravel()[removed]))
            overlap = mask & raw_target
            groups[group][name] = {'originalAnnotationCells': original,
                'removedSourceCellsWithinAnnotation': removed_count,
                'retainedSourceCellsWithinAnnotation': original - removed_count,
                'originalAnnotationMm3': original * scene.voxel_mm3,
                'removedSourceMembershipMm3': removed_count * scene.voxel_mm3,
                'declaredTargetName': name in target_names,
                'overlapWithRawTargetAnnotationCells': int(np.count_nonzero(overlap)),
                'removedSourceCellsWithinRawTargetOverlap': int(np.count_nonzero(overlap.ravel()[removed]))}
    if case.brain_mask is not None:
        mask = case.brain_mask
        original = int(np.count_nonzero(mask))
        removed_count = int(np.count_nonzero(mask.ravel()[removed]))
        groups['brain_mask'] = {'originalAnnotationCells': original,
            'removedSourceCellsWithinAnnotation': removed_count,
            'retainedSourceCellsWithinAnnotation': original - removed_count,
            'originalAnnotationMm3': original * scene.voxel_mm3,
            'removedSourceMembershipMm3': removed_count * scene.voxel_mm3,
            'overlapWithRawTargetAnnotationCells': int(np.count_nonzero(mask & raw_target)),
            'removedSourceCellsWithinRawTargetOverlap': int(np.count_nonzero((mask & raw_target).ravel()[removed]))}
    raw_total = int(np.count_nonzero(raw_target))
    raw_removed = int(np.count_nonzero(raw_target.ravel()[removed]))
    return {'verifiedSameSourceGridAndRasAffine': True,
            'uniqueRemovedOwnedSourceCells': len(removed),
            'rawTargetAnnotationUnionCells': raw_total,
            'removedSourceCellsWithinRawTargetAnnotation': raw_removed,
            'rawTargetAnnotationMembershipRemovedMm3': raw_removed * scene.voxel_mm3,
            'rawTargetAnnotationMembershipRemovedPercent': 100. * raw_removed / raw_total if raw_total else None,
            'groups': groups,
            'meaning': 'Nonexclusive memberships of removed source cells in original supplied masks. Counts may overlap and must not be summed as removed volume. Raw brain/target overlap is unreviewed annotation overlap, not clinical healthy-tissue truth. These counts do not change the OWNED target denominator.'}


def verify(native, branch, *, source=SOURCE, expected_source_sha256=None,
           engineering=False, report=None, cache_directory=None):
    native, source = Path(native).resolve(), Path(source).resolve()
    require(native != source, 'Acceptance must not use the original as its episode file')
    started = time.perf_counter()
    source_sha, native_sha = file_hash(source), file_hash(native)
    expected = expected_source_sha256 or (SOURCE_SHA256 if source == SOURCE.resolve() else None)
    require(expected is not None or engineering, 'Supply an expected source hash for a different patient source')
    require(expected is None or source_sha == expected, 'Original source file hash differs from expected')
    source_case, _, source_workspace = load_workspace(source)
    case, artifacts, workspace = load_workspace(native)
    scene, original_scene = load_environment(native), load_environment(source)
    require(scene.identity == original_scene.identity, 'Episode scene differs from exact original mapping/physics')
    require(engineering or scene.identity == SCENE_ID, 'Patient audit requires exact standard sub-06 scene')
    require(case.semantic_hash == source_case.semantic_hash, 'Original case semantic identity differs')
    baseline_arrays = array_hashes(source_case)
    require(array_hashes(case) == baseline_arrays, 'MRI/affine/source annotation arrays differ')
    require(case.source_refs == source_case.source_refs, 'Acquisition SourceRefs differ')
    require(originals(workspace) == originals(source_workspace), 'Embedded acquired original bytes differ')
    label_hash = array_digest(original_scene.labels)
    require(array_digest(scene.labels) == label_hash, 'Owned source labels differ')
    record = artifacts['surgical_toolbox']; snapshot = record['branches'][branch]
    meta = record.get('fullCase', {}).get(branch, {})
    protection_evidence = None if engineering else protection_audit(scene, snapshot)
    protected_ids = {cell for annotation in snapshot.get('protectionContract', {}).get('annotations', [])
                     for cell in annotation['cells']}
    output = Path(report) if report else Path(__file__).parent / (native.stem + '-' + branch + '-acceptance.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    trace = output.with_suffix('.actions.jsonl')
    require(output.resolve() not in (native, source) and trace.resolve() not in (native, source),
            'Audit output would overwrite source or native episode')
    fingerprint = implementation_fingerprint()
    require(fingerprint is not None, 'Implementation cannot be fingerprinted consistently')
    boundaries, statuses, physical_events = {}, Counter(), Counter()
    max_fluid_error = 0.
    fluid_offset = 0.
    vascular_samples = []
    generated_debris = 0.
    with trace.open('w') as stream:
        class AuditWriter:
            def __init__(self, delegate):
                self.delegate = delegate
                self.removed_before = set()
                self.position = -1

            def capture(self, observed, check=None):
                nonlocal max_fluid_error, fluid_offset, generated_debris
                index, after = len(observed.actions), observed.state_id
                require(index == self.position + 1, 'Canonical capture omitted a boundary')
                vascular_samples.append(vascular_sample(observed, index))
                if protection_evidence:
                    require(not protected_ids.intersection(observed.state['removed']), 'Protected source cells were removed')
                    for cell, vector in observed.state.get('deformation', {}).items():
                        if int(cell) in protected_ids:
                            require(np.linalg.norm(vector) <= 1e-10, 'Protected source cell displaced despite zero allowance')
                    for hold in observed.state['holds'].values():
                        if np.linalg.norm(hold.get('offset', [0., 0., 0.])) > 1e-10:
                            require(not protected_ids.intersection(hold['cells']), 'Protected source cell translated by hold')
                if index == 0:
                    accounted = math.fsum([math.fsum(observed.state['fluid'].values()),
                        math.fsum(s.get('reservoirMm3', 0.) for s in observed.state['sources']),
                        observed.state['fluidRemovedMm3'], observed.state['fluidEscapedMm3']])
                    fluid_offset = observed.state['fluidAddedMm3'] - accounted
                    boundaries[0] = (after, semantic_digest(observed.state))
                    self.removed_before = set(observed.state['removed'])
                    self.position = 0
                    require(not observed.state.get('debris') and not observed.state.get('debrisRemovedMm3', 0.)
                            and not observed.state.get('debrisEscapedMm3', 0.), 'Acceptance initial state has unexplained debris')
                    if self.delegate is not None: self.delegate.capture(observed, check)
                    return
                before = boundaries[index - 1][0]
                action, outcome = observed.actions[-1], observed.history[-1]
                require(outcome['status'] in ('accepted', 'partial'), 'Saved action is rejected by current physics')
                fraction = outcome.get('completedFraction')
                require(type(fraction) in (int, float) and math.isfinite(fraction) and 0 < fraction <= 1.,
                        'Committed physical completion fraction is invalid')
                require(outcome['elapsedSeconds'] > 0 and math.isfinite(outcome['elapsedSeconds']),
                        'Committed physical elapsed time is invalid')
                if outcome['status'] == 'partial':
                    require(bool(outcome.get('reason')), 'Partial physical result lacks its blocking reason')
                require(len(observed.history) == index and action == snapshot['actions'][index - 1],
                        'Action admission/history length differs')
                require(outcome['action'] == action and outcome['stateBefore'] == before and outcome['stateAfter'] == after,
                        'Per-action before/after canonical history differs')
                delta = sorted(set(observed.state['removed']) - self.removed_before)
                require(outcome['deltas']['removedCells'] == delta, 'Removal delta differs from actual source IDs')
                require(self.removed_before <= set(observed.state['removed']), 'Removed source tissue reappeared')
                mass = fluid_audit(observed.state, fluid_offset)
                for event in outcome.get('events', []):
                    if event.get('kind') == 'drill_work': generated_debris += event['removedVolumeMm3']
                    elif event.get('kind') == 'bone_kerf': generated_debris += event['cells'] * scene.voxel_mm3
                solid_mass = drilling_debris_audit(observed.state, scene, generated_debris)
                wound_audit(observed.state, scene)
                max_fluid_error = max(max_fluid_error, abs(mass['residualMm3']))
                boundaries[index] = (after, semantic_digest(observed.state))
                statuses[outcome['status']] += 1
                for event in outcome.get('events', []):
                    physical_events[event.get('type', event.get('kind', 'unspecified')) if isinstance(event, dict) else str(event)] += 1
                detail = {'position': index, 'action': action, 'stateBefore': before,
                          'stateAfter': after, 'status': outcome['status'], 'reason': outcome.get('reason'),
                          'reasonCode': outcome.get('reasonCode'),
                          'completedFraction': outcome.get('completedFraction'),
                          'elapsedSeconds': outcome.get('elapsedSeconds'),
                          'removedCells': delta, 'events': outcome.get('events'), 'fluid': mass,
                          'vascularState': vascular_samples[-1], 'drillingAndDebris': solid_mass}
                stream.write(json.dumps(detail, allow_nan=False) + '\n'); stream.flush()
                if index % 10 == 0:
                    print(json.dumps({'verifiedPosition': index, 'of': len(snapshot['actions']),
                                      'status': outcome['status'], 'reason': outcome.get('reason')}), flush=True)
                self.position = index
                self.removed_before = set(observed.state['removed'])
                if self.delegate is not None: self.delegate.capture(observed, check)

            def publish(self, observed, record, check=None):
                # The canonical restore calls this only AFTER checkpoint and
                # endpoint verification. Audit failures above propagate and
                # the restore's finally block aborts the unsealed staging.
                require(file_hash(source) == source_sha and file_hash(native) == native_sha,
                        'Source or episode changed before canonical publication')
                if self.delegate is not None: self.delegate.publish(observed, record, check)

            def abort(self):
                if self.delegate is not None: self.delegate.abort()

        class AuditedVerifiedCache(VerifiedSurgicalCache):
            def _begin_replay(self, observed, record, check=None):
                return AuditWriter(super()._begin_replay(observed, record, check))

        cache = AuditedVerifiedCache(cache_directory)
        engine = SurgicalEngine.restore(scene, snapshot, use_cache=True, force_verify=True, verified_cache=cache)
    require(implementation_fingerprint() == fingerprint, 'Implementation changed during canonical verification')
    accounting = source_accounting(scene, engine.state)
    metrics = engine.metrics()
    require(math.isclose(metrics['targetRemovedMm3'], accounting['target']['removedOwnedMm3'], abs_tol=1e-8),
            'Published target metric disagrees with independent source count')
    target_removed_cells = accounting['target']['removedOwnedCells']
    require(math.isclose(metrics['nonTargetRemovedMm3'],
                        (len(engine.state['removed']) - target_removed_cells) * scene.voxel_mm3, abs_tol=1e-8),
            'Published non-target metric disagrees with independent source count')
    removed_labels = scene.labels.ravel()[np.asarray(engine.state['removed'], dtype=np.int64)]
    forbidden = {region['name']: int(np.count_nonzero(removed_labels == region['id'])) for region in scene.regions
                 if region['material'] in ('soft_tissue', 'unknown', 'functional_region', 'membrane')
                 and not region.get('target') and np.any(removed_labels == region['id'])}
    require(not forbidden, 'OWNED non-target healthy/unknown/membrane/functional source removal: ' + repr(forbidden))
    require(engineering or engine.guard_report()['mode'] == 'strict-research', 'Patient guard is not strict')
    if protection_evidence:
        require(protection_evidence == protection_audit(scene, snapshot), 'Protection evidence changed during replay')
    causal = causal_vascular_audit(vascular_samples, engine.actions,
        scene.properties.get('tissuePhysicsV2', {}).get('vessel', {}).get('coagulationDose', math.inf),
        required=meta.get('scenario') == 'vascular')
    prefix_checks = []
    for name, other in record['branches'].items():
        n = len(other['actions'])
        if n <= len(engine.actions) and engine.actions[:n] == other['actions']:
            require(other.get('protectionContract') == snapshot.get('protectionContract'),
                    'Same-action branch prefix protection contract differs: ' + name)
            require(boundaries[n][0] == other['stateId'], 'Same-action branch prefix state differs: ' + name)
            checkpoint = other.get('checkpoint')
            if checkpoint:
                require(checkpoint['position'] == n and checkpoint['stateId'] == boundaries[n][0]
                        and semantic_digest(checkpoint['state']) == boundaries[n][1],
                        'Same-action branch prefix checkpoint differs: ' + name)
            prefix_checks.append({'branch': name, 'position': n, 'stateId': boundaries[n][0]})
    parent = record.get('parents', {}).get(branch)
    if parent:
        other = record['branches'].get(parent['branchId'])
        require(other is not None, 'Declared parent branch is missing')
        n = parent['position']
        require(type(n) is int and 0 <= n <= min(len(other['actions']), len(engine.actions)), 'Invalid branch parent boundary')
        require(engine.actions[:n] == other['actions'][:n], 'Declared branch parent action prefix differs')
    if meta.get('commonPrefixPosition') is not None and 'exposure-prefix' in record['branches']:
        require(meta['commonPrefixPosition'] == len(record['branches']['exposure-prefix']['actions']),
                'Common exposure prefix position differs')
    if meta.get('lastStateId'):
        require(meta['lastStateId'] == engine.state_id, 'Director boundary differs from final snapshot')
    require(file_hash(source) == source_sha and file_hash(native) == native_sha, 'Source or episode file changed during audit')
    require(array_hashes(case) == baseline_arrays and array_hashes(source_case) == baseline_arrays
            and array_digest(scene.labels) == label_hash, 'Source arrays changed during audit')
    partials = [{'position': i + 1, 'action': row['action'], 'reason': row.get('reason'),
                 'completedFraction': row.get('completedFraction'), 'stateBefore': row['stateBefore'],
                 'stateAfter': row['stateAfter']} for i, row in enumerate(engine.history) if row['status'] == 'partial']
    result = {'schema': 'resectionlab.independent-native-acceptance.v1', 'auditStatus': 'passed',
              'episodeCompletionStatus': meta.get('status', 'not-declared'),
              'engineeringFixture': engineering, 'native': str(native), 'nativeSha256': native_sha,
              'source': str(source), 'originalSha256': source_sha, 'sourceUnchanged': True,
              'caseHash': case.semantic_hash, 'sourceArrays': baseline_arrays,
              'originalFiles': originals(workspace), 'sceneId': scene.identity, 'labelArrayHash': label_hash,
              'implementationFingerprint': fingerprint, 'canonicalReplay': {'forced': True, 'cacheUsed': False,
              'verifiedCachePublication': cache.statistics,
              'perActionStateAssertions': len(engine.actions), 'statuses': dict(statuses),
              'rejectedCommittedActions': 0, 'partialActions': partials, 'trace': str(trace.resolve())},
              'branch': branch, 'parent': parent, 'verifiedSavedPrefixStates': prefix_checks,
              'stateId': engine.state_id, 'actions': len(engine.actions), 'accounting': accounting,
              'rawAnnotationMemberships': annotation_memberships(source_case, original_scene, engine.state['removed']),
              'fluid': fluid_audit(engine.state, fluid_offset), 'maxFluidResidualMm3': max_fluid_error,
              'drillingAndDebris': drilling_debris_audit(engine.state, scene, generated_debris),
              'vascular': wound_audit(engine.state, scene), 'protection': engine.guard_report(),
              'sourceProtectionEvidence': protection_evidence, 'causalVascularChain': causal,
              'directProtectionMaskTargetMembership': protected_target_membership(scene, engine.state, snapshot.get('protectionContract')),
              'endpointTools': engine.state['slots'], 'endpointHolds': engine.state['holds'],
              'directorPhases': meta.get('phases', []), 'recordedUncommittedProposals': meta.get('rejections', []),
              'proposalQualification': 'Saved proposal outcomes reported verbatim; these were not committed or re-executed by this audit.',
              'physicalEventCounts': dict(physical_events), 'wallSeconds': time.perf_counter() - started,
              'guiReview': 'pending: this script verifies engine records only; actual complete-episode Electron playback is a separate gate',
              'limitations': ['Uncalibrated supplied engineering laws; no clinical outcome claim.',
                              'Retained source target regions do not prove that every possible legal approach is exhausted.']}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native', type=Path)
    parser.add_argument('branch')
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--expected-source-sha256')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--cache-directory', type=Path)
    parser.add_argument('--engineering', action='store_true', help='Explicitly label a simulated fixture; never a patient acceptance')
    args = parser.parse_args()
    result = verify(args.native, args.branch, source=args.source,
                    expected_source_sha256=args.expected_source_sha256,
                    engineering=args.engineering, report=args.report, cache_directory=args.cache_directory)
    print(json.dumps({key: result[key] for key in ('auditStatus', 'stateId', 'actions', 'guiReview')}))


if __name__ == '__main__':
    main()
