"""Separate source-bound full-aperture milling declarations and bone readiness.

This read-only module creates no opening, task admission or patient evidence.
The flap contract and its replay semantics remain unchanged. Source removal is
not clearance: accepted native cutting, suction, conservation and actual exit
must all be established, without manufacturing a flap evacuation receipt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

import numpy as np
from scipy.ndimage import label

from .core import array_digest, semantic_digest
from . import surgical_craniotomy as flap


PROPERTY = 'cranialMillingV1'
SCHEMA = 'resectionlab.cranial-milling.v1'
BASE_CASE_HASH = 'cranialMillingBaseCaseHash'
_PROOFS = ('window', 'boneScope', 'innerBoundary', 'exit')
_FIELDS = {'schema', 'method', 'sourceBinding', 'targetBinding', 'attachmentsBinding',
           'apertureCells', 'retainedSkullCells', 'boneAnchorCells',
           'entry', 'exit', 'evidence', 'contentHash'}
_EPS = 1e-9


def _invalid(reason):
    raise ValueError('MILLING_CONTRACT_INVALID: '+reason)


def _blocked(reason):
    raise ValueError('MILLING_'+reason)


def default_evidence(scope='patient-source-review'):
    """Explicit unknown coverage; it can be inspected but cannot execute."""
    return {'scope': scope, **{name: {'status': 'unknown',
        'provenance': 'Coverage not supplied; source-bound engineering review required',
        'evidenceSha256': None} for name in _PROOFS}}


@dataclass(frozen=True)
class MillingContract:
    source_binding: str
    target_binding: str
    attachments_binding: str
    aperture_cells: tuple[int, ...]
    retained_skull_cells: tuple[int, ...]
    bone_anchor_cells: tuple[int, ...]
    entry: flap.EntryGeometry
    exit: flap.ExitGeometry
    evidence_scope: str
    proofs: tuple[flap.EvidenceProof, ...]
    content_hash: str
    method: str = field(default='milling', init=False)

    @property
    def window_cells(self):
        return self.aperture_cells

    @property
    def kerf_cells(self):
        return self.aperture_cells

    @property
    def flap_cells(self):
        return ()

    def as_dict(self):
        return {'schema': SCHEMA, 'method': self.method,
            'sourceBinding': self.source_binding, 'targetBinding': self.target_binding,
            'attachmentsBinding': self.attachments_binding,
            'apertureCells': list(self.aperture_cells),
            'retainedSkullCells': list(self.retained_skull_cells),
            'boneAnchorCells': list(self.bone_anchor_cells),
            'entry': {'centerRasMm': list(self.entry.center_ras_mm),
                'normalRasUnit': list(self.entry.normal_ras_unit),
                'uRasUnit': list(self.entry.u_ras_unit),
                'halfExtentsMm': list(self.entry.half_extents_mm),
                'maximumInwardMm': self.entry.maximum_inward_mm,
                'maximumAngleDegrees': self.entry.maximum_angle_degrees},
            'exit': {'axis': self.exit.axis, 'side': self.exit.side,
                'planeVoxelBoundary': self.exit.plane_voxel_boundary,
                'minimumVoxelBoundary': list(self.exit.minimum_voxel_boundary),
                'maximumVoxelBoundary': list(self.exit.maximum_voxel_boundary)},
            'evidence': {'scope': self.evidence_scope, **{name: {
                'status': proof.status, 'provenance': proof.provenance,
                'evidenceSha256': proof.evidence_sha256}
                for name, proof in zip(_PROOFS, self.proofs)}},
            'contentHash': self.content_hash}


def is_milling(contract):
    return isinstance(contract, MillingContract)


def binding(scene):
    """Bind all anatomy/properties except this declaration's own case digest."""
    targets = [row for row in scene.regions if row.get('target')]
    ids = sorted(row['id'] for row in targets)
    source = dict(scene.source)
    if BASE_CASE_HASH in source:
        base = source.pop(BASE_CASE_HASH)
        if (type(source.get('caseHash')) is not str or type(base) is not str or
                not re.fullmatch('sha256:[a-f0-9]{64}', base)):
            _invalid('computed base case binding is invalid')
        source['caseHash'] = base
    return {'sourceBinding': semantic_digest({'labels': array_digest(scene.labels),
        'affine': scene.affine.tolist(), 'regions': scene.regions, 'source': source,
        'properties': {key: value for key, value in scene.properties.items() if key != PROPERTY}}),
        'targetBinding': semantic_digest({'ownedTarget': array_digest(np.isin(scene.labels, ids)),
            'targetRegions': targets, 'originalAnnotationHashes': {
                row['name']: scene.source.get('annotationHashes', {}).get(row['name']) for row in targets}}),
        'attachmentsBinding': semantic_digest({'present': 'attachments' in scene.properties,
                                               'records': scene.properties.get('attachments')})}


def contract_hash(raw):
    return semantic_digest({key: value for key, value in raw.items() if key != 'contentHash'})


def _proofs(raw):
    if type(raw) is not dict or set(raw) != {'scope', *_PROOFS} or raw['scope'] not in (
            'idealized-engineering', 'patient-source-review'):
        _invalid('explicit evidence scope and four milling proofs required')
    proofs = []
    for name in _PROOFS:
        proof = raw[name]
        if (type(proof) is not dict or set(proof) != {'status', 'provenance', 'evidenceSha256'} or
                proof['status'] not in ('unknown', 'unreviewed', 'reviewed') or
                type(proof['provenance']) is not str or not proof['provenance'].strip()):
            _invalid('invalid '+name+' proof')
        digest = proof['evidenceSha256']
        if (digest is not None and (type(digest) is not str or
                not re.fullmatch('[a-f0-9]{64}', digest))) or proof['status'] == 'reviewed' and digest is None:
            _invalid('reviewed '+name+' evidence requires a SHA256')
        proofs.append(flap.EvidenceProof(proof['status'], proof['provenance'], digest))
    return raw['scope'], tuple(proofs)


def validate_contract(scene, contract=None):
    """Validate immutable A/R identity and geometry; unknown evidence is inert."""
    if PROPERTY in scene.properties and flap.PROPERTY in scene.properties:
        _invalid('milling and flap declarations are exclusive')
    raw = scene.properties.get(PROPERTY) if contract is None else contract
    if raw is None:
        if PROPERTY in scene.properties:
            _invalid('present declaration cannot be null')
        return None
    if flap.PROPERTY in scene.properties:
        _invalid('milling and flap declarations are exclusive')
    if is_milling(raw):
        try:
            raw = raw.as_dict()
        except (AttributeError, TypeError):
            _invalid('malformed typed milling contract')
    if (type(raw) is not dict or set(raw) != _FIELDS or
            raw.get('schema') != SCHEMA or raw.get('method') != 'milling'):
        _invalid('exact versioned milling declaration required')
    if raw['contentHash'] != contract_hash(raw) or any(raw[key] != value for key, value in binding(scene).items()):
        _invalid('content or source/target/attachment binding differs')
    try:
        aperture = flap._cells(raw['apertureCells'], scene.labels.size, 'aperture')
        retained = flap._cells(raw['retainedSkullCells'], scene.labels.size, 'retained skull')
        anchors = flap._cells(raw['boneAnchorCells'], scene.labels.size, 'bone anchors', empty=True)
        entry, exit_geometry = flap._entry(raw['entry']), flap._exit(raw['exit'], scene.shape)
        flap._window_geometry(scene, aperture, entry, exit_geometry)
    except ValueError as error:
        _invalid(str(error))
    bone = set(map(int, np.flatnonzero(np.isin(scene.labels,
        [row['id'] for row in scene.regions if row['material'] == 'bone']))))
    a, r = set(aperture), set(retained)
    if a & r or a | r != bone:
        _invalid('A/R must partition all categorical source bone')
    if not set(anchors) <= r or not set(scene.properties.get('boneAnchors', ())) <= set(anchors):
        _invalid('explicit source bone anchors must remain in R and cannot be omitted')
    occupied = np.column_stack(np.nonzero(scene.labels))
    axis, side, plane = exit_geometry.axis, exit_geometry.side, exit_geometry.plane_voxel_boundary
    if (side == -1 and plane > occupied[:, axis].min()-.5+_EPS or
            side == 1 and plane < occupied[:, axis].max()+.5-_EPS):
        _invalid('exit plane must bound the original source exterior')
    scope, proofs = _proofs(raw['evidence'])
    return MillingContract(raw['sourceBinding'], raw['targetBinding'], raw['attachmentsBinding'],
        aperture, retained, anchors, entry, exit_geometry, scope, proofs, raw['contentHash'])


def resolve_contract(scene):
    """Dispatch only the explicit new property; legacy parsing is unchanged."""
    return validate_contract(scene) if PROPERTY in scene.properties else flap.validate_contract(scene)


def require_coverage(contract):
    if not is_milling(contract):
        return flap.require_evidence(contract)
    if contract.method != 'milling' or len(contract.proofs) != len(_PROOFS) or any(
            not isinstance(proof, flap.EvidenceProof) for proof in contract.proofs):
        _blocked('COVERAGE_PROOFS_REQUIRED')
    _proofs(contract.as_dict()['evidence'])
    for name, proof in zip(_PROOFS, contract.proofs):
        if proof.status != 'reviewed':
            _blocked(name.upper()+'_EVIDENCE_REQUIRED')
    if contract.retained_skull_cells and not contract.bone_anchor_cells:
        _blocked('SUPPORT_EVIDENCE_REQUIRED')
    return contract


def _support_verified(scene, contract):
    mask = np.zeros(scene.shape, bool)
    mask.ravel()[list(contract.retained_skull_cells)] = True
    components, count = label(mask)  # Explicitly face/6 connected.
    anchored = set(map(int, components.ravel()[list(contract.bone_anchor_cells)]))-{0}
    return count > 0 and anchored == set(range(1, count+1))


def validate_source_scope(engine, contract=None):
    """Preserve original target/protection and retained skull before effects."""
    from .surgical_engine import SurgicalEngine
    from .surgical_transition_meter import MeteredEngine
    if not isinstance(engine, (SurgicalEngine, MeteredEngine)):
        _blocked('NATIVE_ENGINE_REQUIRED')
    contract = resolve_contract(engine.scene) if contract is None else contract
    if not is_milling(contract):
        return flap.validate_bone_scope(engine, contract)
    engine.scene.require_cache_source()
    contract = require_coverage(validate_contract(engine.scene, contract))
    from .surgical_drilling_v2 import validate
    physical = engine.scene.properties.get('tissuePhysicsV2')
    bone_profile = physical.get('bone') if isinstance(physical, dict) else None
    if bone_profile is None:
        _blocked('V2_BONE_PROFILE_REQUIRED')
    validate(bone_profile)
    intended, retained = set(contract.aperture_cells), set(contract.retained_skull_cells)
    if intended & flap._original_targets(engine.scene):
        _blocked('TARGET_OVERLAP')
    protected = set(getattr(engine, '_protected_ids', ()))
    for row in engine.scene.source.get('protectedAnatomyV1', []):
        protected.update(row['cells'])
    protected.update(map(int, np.flatnonzero(np.isin(engine.scene.labels,
        [row['id'] for row in engine.scene.regions if row.get('protected') is True or row.get('critical') is True]))))
    if intended & protected:
        _blocked('PROTECTED_STRUCTURE')
    removed = engine.state['removed']
    if (any(type(cell) is not int or not 0 <= cell < engine.scene.labels.size for cell in removed) or
            len(removed) != len(set(removed))):
        _blocked('REMOVED_LEDGER_INVALID')
    fragments = [*engine.state['holds'].values(), *engine.state.get('detachedFragments', [])]
    if (retained & set(removed) or any(retained & set(fragment.get('cells', ())) for fragment in fragments) or
            any(int(cell) in retained and any(offset) for cell, offset in engine.state.get('deformation', {}).items())):
        _blocked('RETAINED_SKULL_CHANGED')
    if not _support_verified(engine.scene, contract):
        _blocked('RETAINED_SUPPORT_REQUIRED')
    return contract


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _history_evidence(engine, contract):
    """Check native accepted ancestry/events; never execute an inline replay."""
    from .surgical_engine import SurgicalEngine
    canonical = SurgicalEngine(engine.scene, protection=engine.protection, guard_config=engine.guard_config,
            **({'cranial_stages':engine._cranial_stages} if getattr(engine, '_cranial_stages', None) is not None else {}))
    previous = canonical.state_id
    instruments = {key: slot['instrument'] for key, slot in canonical.state['slots'].items()}
    active = {key: slot['active'] for key, slot in canonical.state['slots'].items()}
    mechanics, clock = canonical.state.get('toolMechanicsRevision'), canonical.state.get('physicalClockRevision')
    removed, generated, suctioned = set(), 0., 0.
    if len(engine.actions) != len(engine.history):
        return None
    for action, event in zip(engine.actions, engine.history):
        fraction = event.get('completedFraction')
        accepted = (event.get('status') == 'accepted' and
                    type(fraction) in (int, float) and fraction == 1.)
        capacity_partial = (event.get('status') == 'partial' and
            event.get('reason') == 'BONE_CUTTING_IN_PROGRESS' and
            type(fraction) in (int, float) and math.isfinite(fraction) and 0 <= fraction <= 1.)
        if (event.get('action') != action or not (accepted or capacity_partial) or
                event.get('stateBefore') != previous or
                not isinstance(event.get('stateAfter'), str) or
                'stateId' in event and event['stateId'] != event['stateAfter']):
            return None
        slot = action.get('slot')
        if slot not in instruments:
            return None
        operation = action.get('operation')
        if operation == 'configure':
            if 'mechanicsRevision' in action or 'clockRevision' not in action:
                mechanics = action.get('mechanicsRevision', 1)
            if 'clockRevision' in action:
                clock = action['clockRevision']
        if operation == 'exchange':
            if active[slot]:
                return None
            instruments[slot] = action.get('instrument')
        if operation == 'activate':
            active[slot] = True
        elif operation in ('deactivate', 'withdraw', 'release'):
            active[slot] = False
        tool = instruments[slot]
        finite_setup = type(mechanics) is int and mechanics == 2 and type(clock) is int and clock == 1
        if capacity_partial and (tool != 'cranial_drill' or operation not in ('apply', 'move') or
                not finite_setup or
                not _number(event.get('elapsedSeconds')) or event['elapsedSeconds'] <= 0 or
                not any(effect.get('kind') == 'drill_work' and
                    (_number(effect.get('cuttingWorkJ')) and effect['cuttingWorkJ'] > 0 or
                     type(effect.get('pendingCells')) is int and effect['pendingCells'] > 0)
                    for effect in event.get('events', ()))):
            return None

        def effect_tool(effect):
            hand = effect.get('sourceSlot')
            if not finite_setup or hand not in instruments or effect.get('instrumentId') != instruments[hand]:
                return None
            expected_operation = operation if hand == slot else 'apply'
            if effect.get('effectOperation') != expected_operation:
                return None
            effective = (operation in ('apply', 'activate', 'hold', 'grasp', 'extract') or active[hand]) if hand == slot else active[hand]
            if not effective or operation in ('configure', 'position'):
                return None
            return instruments[hand]
        cells = event.get('deltas', {}).get('removedCells')
        if (not isinstance(cells, list) or any(type(cell) is not int for cell in cells) or
                cells != sorted(set(cells)) or removed.intersection(cells) or
                set(cells)-set(contract.aperture_cells)):
            return None
        volume, observed, explicit_ids = len(cells)*engine.scene.voxel_mm3, 0., set()
        for effect in event.get('events', ()):
            kind = effect.get('kind')
            if kind in ('removal', 'bone_kerf', 'drill_work'):
                if kind != 'drill_work' or effect_tool(effect) != 'cranial_drill':
                    return None
                value = effect.get('removedVolumeMm3')
                ids = effect.get('removedCells')
                energy, work, pending = effect.get('inputEnergyJ'), effect.get('cuttingWorkJ'), effect.get('pendingCells')
                if (not isinstance(ids, list) or any(type(cell) is not int for cell in ids) or
                        ids != sorted(set(ids)) or not set(ids) <= set(cells) or
                        explicit_ids.intersection(ids) or not _number(value) or
                        abs(value-len(ids)*engine.scene.voxel_mm3) > _EPS or
                        not _number(energy) or not _number(work) or work > energy+_EPS or
                        type(pending) is not int or pending < 0 or ids and work <= 0):
                    return None
                explicit_ids.update(ids)
                observed += value
            elif kind == 'aspiration':
                value = effect.get('debris')
                if effect_tool(effect) != 'frazier_suction' or not _number(value):
                    return None
                suctioned += value
        if abs(observed-volume) > _EPS:
            return None
        removed.update(cells); generated += observed
        previous = event['stateAfter']
    if (previous != engine.state_id or removed != set(engine.state['removed']) or
            any(instruments[key] != slot['instrument'] or active[key] != slot['active']
                for key, slot in engine.state['slots'].items()) or
            mechanics != engine.state.get('toolMechanicsRevision') or clock != engine.state.get('physicalClockRevision')):
        return None
    return generated, suctioned


def _withdrawn(engine, contract):
    from .surgical_motion import envelopes
    from .surgical_task_domain import _posed
    from .surgical_tools import TOOLS
    if engine.state['holds'] or engine.state.get('detachedFragments'):
        return False
    axis, side, plane = contract.exit.axis, contract.exit.side, contract.exit.plane_voxel_boundary
    inverse = engine.scene.inverse
    margin = np.linalg.norm(inverse[axis, :3])
    for key, slot in engine.state['slots'].items():
        if slot['inserted'] or slot['active'] or slot['grip']:
            return False
        if not _posed(engine, key):
            continue
        for a, b, radius, _ in envelopes(TOOLS[slot['instrument']], engine.parameters(slot), slot['position'], slot['quaternion']):
            coordinates = np.asarray([a, b])@inverse[axis, :3]+inverse[axis, 3]
            if side == -1 and coordinates.max()+radius*margin >= plane-_EPS:
                return False
            if side == 1 and coordinates.min()-radius*margin <= plane+_EPS:
                return False
    return True


def milling_readiness(engine):
    """Bone-only clearance descriptor; it never authorizes CUSA/dural entry."""
    contract = resolve_contract(engine.scene)
    if not is_milling(contract):
        return None
    result = {'method': 'milling', 'contractHash': contract.content_hash,
        'phase': 'BOUND', 'apertureRemoved': False, 'retainedSkullIntact': False,
        'retainedSupportVerified': False, 'historyVerified': False, 'debrisConserved': False,
        'apertureCleared': False, 'openingComplete': False,
        'removedBoneMm3': 0., 'milledBoneMm3': 0., 'generatedBoneDebrisMm3': None,
        'debrisRemovedMm3': None, 'debrisPresentMm3': None, 'debrisEscapedMm3': None,
        'reason': None, 'clinicalValidation': 'none', 'treatmentEntrySupported': False}
    try:
        validate_source_scope(engine, contract)
    except ValueError as error:
        result['reason'] = str(error)
        return result
    a, r = set(contract.aperture_cells), set(contract.retained_skull_cells)
    removed = set(engine.state['removed'])
    result.update(apertureRemoved=a <= removed, retainedSkullIntact=not bool(r & removed),
        retainedSupportVerified=True, removedBoneMm3=len(a & removed)*engine.scene.voxel_mm3,
        milledBoneMm3=len(a & removed)*engine.scene.voxel_mm3)
    evidence = _history_evidence(engine, contract)
    if evidence is None:
        result['reason'] = 'MILLING_NATIVE_HISTORY_REQUIRED'
        return result
    generated, suctioned = evidence
    result.update(historyVerified=True, generatedBoneDebrisMm3=generated)
    debris = engine.state.get('debris')
    if (not isinstance(debris, dict) or any(not _number(value) for value in debris.values())):
        result['reason'] = 'MILLING_DEBRIS_EVIDENCE_REQUIRED'
        return result
    present = sum(debris.values())
    removed_debris = engine.state.get('debrisRemovedMm3', 0.)
    escaped = engine.state.get('debrisEscapedMm3', 0.)
    scalar_present = engine.state.get('debrisMm3')
    if (not all(_number(value) for value in (present, removed_debris, escaped, scalar_present)) or
            abs(scalar_present-present) > _EPS or abs(removed_debris-suctioned) > _EPS):
        result['reason'] = 'MILLING_DEBRIS_EVIDENCE_REQUIRED'
        return result
    total = len(a)*engine.scene.voxel_mm3
    conserved = (generated <= total+_EPS and abs(generated-removed_debris-present-escaped) <= _EPS)
    result.update(debrisConserved=conserved, debrisRemovedMm3=removed_debris,
        debrisPresentMm3=present, debrisEscapedMm3=escaped)
    fragments = [*engine.state['holds'].values(), *engine.state.get('detachedFragments', [])]
    no_bone_hold = not any((a | r) & set(fragment.get('cells', ())) for fragment in fragments)
    finite_setup = engine.state.get('toolMechanicsRevision') == 2 and engine.state.get('physicalClockRevision') == 1
    clear = (finite_setup and result['apertureRemoved'] and conserved and no_bone_hold and
        generated+_EPS >= total and removed_debris+_EPS >= total and present == 0. and escaped == 0.)
    complete = clear and _withdrawn(engine, contract)
    result.update(apertureCleared=clear, openingComplete=complete,
        phase='WITHDRAWN' if complete else 'CLEARED' if clear else
              'CLEARANCE_PENDING' if result['apertureRemoved'] else
              'MILLING_IN_PROGRESS' if removed or any(engine.state.get('drillingV2', {}).get('workJ', {}).values()) else 'BOUND',
        reason=None if complete else 'MILLING_WITHDRAWAL_REQUIRED' if clear else
               'MILLING_DEBRIS_CONSERVATION_REQUIRED' if not conserved else
               'MILLING_CLEARANCE_REQUIRED' if result['apertureRemoved'] else 'MILLING_APERTURE_INCOMPLETE')
    return result
