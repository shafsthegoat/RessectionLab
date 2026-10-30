"""Read-only eligibility checks for bounded full-case research rehearsal."""
from __future__ import annotations

import numpy as np

from .surgical_rl_protection import (
    GuardConfig, load_protection, missing_critical, protection_contract,
    restore_protection_contract,
)

SCHEMA = 'resectionlab.full-case-readiness.v1'


class ProtectionBindingError(ValueError):
    """Actionable evidence failure, shared by bridge and readiness responses."""

    def __init__(self, code, message, details=()):
        super().__init__(message)
        self.code = code
        self.details = list(details)

    def blocker(self):
        return {'code': self.code, 'message': str(self), 'details': self.details.copy()}


def checked_case_protection(engine, protection_path=None):
    """Return validated evidence and its prospective strict contract, read-only.

    Only the saved self-contained engine contract or an explicitly selected
    supported protection manifest is evidence. No local-path search is used.
    """
    try:
        engine.scene.require_cache_source()
    except ValueError as exc:
        raise ProtectionBindingError('source-mismatch',
            'The current scene differs from its saved source binding.', [str(exc)]) from exc

    existing = getattr(engine, '_protection_contract', None)
    previous = None
    if existing is not None:
        try:
            previous, _ = restore_protection_contract(engine.scene, existing)
            if (engine.protection is None
                    or protection_contract(engine.protection, engine.guard_config) != existing
                    or frozenset(map(int, engine.protection.cells)) != engine._protected_ids):
                raise ValueError('Active protection differs from the saved contract')
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ProtectionBindingError('invalid-protection-contract',
                'Saved protection no longer matches its integrity or current source grid.', [str(exc)]) from exc
    elif engine.protection is not None or engine.guard_config is not None:
        raise ProtectionBindingError('invalid-protection-contract',
            'Active protection has no saved source-bound contract.')

    if protection_path is not None:
        try:
            protection = load_protection(engine.scene, protection_path, strict=True)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
            raise ProtectionBindingError('invalid-protection-manifest',
                'The selected protection manifest is unavailable or incompatible with this source grid.',
                [str(exc)]) from exc
    elif previous is not None:
        protection = previous
    else:
        raise ProtectionBindingError('missing-protection',
            'Attach source-bound protection evidence before full rehearsal.',
            ['Select a supported protection manifest for this exact source, labels and affine grid.',
             'Brainstem and every declared required structure need nonempty supplied cells.'])

    missing = missing_critical(protection)
    if missing:
        raise ProtectionBindingError('missing-critical-annotations',
            'Protection evidence is missing required critical coverage.',
            ['Brainstem and every declared required structure need nonempty supplied cells.', *missing])

    if previous is not None:
        reductions = []
        if protection.clearance_mm < previous.clearance_mm:
            reductions.append('Clearance cannot decrease from the existing protection contract.')
        removed_required = sorted(set(previous.required) - set(protection.required))
        if removed_required:
            reductions.append('Required structures removed: ' + ', '.join(removed_required))
        cells_by_structure = {a.structure_id: set(a.cells) for a in protection.annotations}
        for annotation in previous.annotations:
            if set(annotation.cells) - cells_by_structure.get(annotation.structure_id, set()):
                reductions.append('Protected source cells removed from ' + annotation.structure_id + '.')
        if reductions:
            raise ProtectionBindingError('protection-relaxation',
                'Attaching protection must preserve existing clearance, cells and requirements.', reductions)

    contract = protection_contract(protection, GuardConfig())
    if engine.actions and contract != existing:
        raise ProtectionBindingError('initial-checkpoint-required',
            'Bind matching strict protection at an intact initial checkpoint before full rehearsal.',
            ['Saved action history can continue only with its identical strict protection contract.'])
    return protection, contract


def full_case_readiness(engine):
    """Describe full-case eligibility without changing evidence, state or history."""
    blockers = []
    protection_id = None
    try:
        _, contract = checked_case_protection(engine)
        protection_id = contract['contentHash']
    except ProtectionBindingError as exc:
        blockers.append(exc.blocker())
    recoverable_protection = any(b['code'] in (
        'missing-protection', 'missing-critical-annotations', 'initial-checkpoint-required') for b in blockers)

    intrinsic_blockers = []
    if not any(r.get('target') and r['material'] == 'tumor'
               and np.any(engine.scene.labels == r['id']) for r in engine.scene.regions):
        intrinsic_blockers.append({'code': 'missing-owned-tumor-target',
                         'message': 'Full rehearsal requires a supplied owned tumor target.',
                         'details': ['Declare a tumor target with occupied source cells in this scene.']})

    linear = engine.scene.affine[:3, :3]
    if not np.allclose(linear, np.diag(np.diag(linear))) or linear[2, 2] <= 0:
        intrinsic_blockers.append({'code': 'unsupported-director-grid',
                         'message': 'Superior candidates require an axis-aligned superior-increasing grid.',
                         'details': ['The current bounded director does not support this scene affine.']})
    blockers = intrinsic_blockers + blockers

    actions = ['inspect', 'replay']
    recovery = None
    if not blockers:
        actions.extend(['primary', 'vascular'])
    elif recoverable_protection:
        if engine.actions:
            actions.append('seekInitial')
            if not intrinsic_blockers:
                recovery = {'action': 'seekInitial', 'label': 'Seek initial checkpoint', 'position': 0}
        else:
            actions.append('attachProtection')
            if not intrinsic_blockers:
                recovery = {'action': 'attachProtection', 'label': 'Attach matching protection'}

    return {'schema': SCHEMA, 'sceneId': engine.scene.identity, 'stateId': engine.state_id,
            'status': 'blocked' if blockers else 'ready', 'availableActions': actions,
            'blockers': blockers, 'recovery': recovery, 'protectionContractId': protection_id}
