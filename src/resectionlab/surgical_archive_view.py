"""Read-only display of an imported endpoint, without causal replay admission.

A matching checkpoint digest establishes internal consistency only. Anyone can
fabricate a different state and recompute its digest. This module returns JSON
display data, never an editable engine, replay admission, or trusted cache entry.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

from .core import freeze_json, thaw_json, semantic_digest
from .surgical_engine import MAX_STEPS, SCHEMA, SurgicalEngine, rotation
from .surgical_rl_protection import restore_protection_contract
from .surgical_scene import SurgicalScene
from .surgical_tools import TOOLS


class _ArchiveDisplayShell(SurgicalEngine):
    """Private view adapter whose transition/persistence entry points fail shut."""

    def _unavailable(self, *args, **kwargs):
        raise RuntimeError('Unverified imported display cannot execute or authorize physics')

    step = preview = clone = fresh = branch = snapshot = _unavailable

    @classmethod
    def restore(cls, *args, **kwargs):
        raise RuntimeError('Unverified imported display cannot restore an engine')


def _sequence(value, label):
    if not isinstance(value, (list, tuple)):
        raise ValueError('Imported ' + label + ' must be an array')
    return value


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Imported ' + label + ' must be finite and nonnegative')


def _state_shape(state, scene, initial):
    """Check display inputs, without asserting physical reachability."""
    if not isinstance(state, dict) or set(initial) - set(state):
        raise ValueError('Imported checkpoint state is incomplete')
    def cells(value, label):
        for cell in _sequence(value, label):
            if type(cell) is not int or not 0 <= cell < scene.labels.size:
                raise ValueError('Imported ' + label + ' has an invalid grid cell')
    cells(state['removed'], 'removed')
    if len(set(state['removed'])) != len(state['removed']) or any(not scene.labels.flat[c] for c in state['removed']):
        raise ValueError('Imported removal ledger has duplicate or unowned cells')
    if not isinstance(state['slots'], dict) or set(state['slots']) != {'left', 'right'}:
        raise ValueError('Imported checkpoint requires both instrument slots')
    for slot in state['slots'].values():
        if not isinstance(slot, dict) or set(initial['slots']['left']) - set(slot):
            raise ValueError('Imported instrument slot is incomplete')
        if not isinstance(slot['instrument'], str) or slot['instrument'] not in TOOLS:
            raise ValueError('Imported instrument is unknown')
        if type(slot['active']) is not bool or type(slot['inserted']) is not bool:
            raise ValueError('Imported instrument switches must be boolean')
        if len(_sequence(slot['position'], 'instrument position')) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in slot['position']):
            raise ValueError('Imported instrument position must be finite RAS millimeters')
        rotation(slot['quaternion'])
        cells(slot['grip'], 'instrument grip')
        if not isinstance(slot['parameters'], dict):
            raise ValueError('Imported instrument parameters must be an object')
        for value in slot['parameters'].values():
            _number(value, 'instrument parameter')
    for name, value in initial.items():
        if isinstance(value, dict) and not isinstance(state[name], dict):
            raise ValueError('Imported ' + name + ' must be an object')
        if isinstance(value, list):
            _sequence(state[name], name)
        if type(value) in (int, float):
            _number(state[name], name)


def inspect_archive_checkpoint(scene, record: Mapping, *, branch_id: str | None = None, check=None) -> dict:
    """Display one saved branch endpoint, explicitly without replay verification.

    ``scene`` must be the exact source/mapping/physics scene selected by the
    caller's bounded native loader. ``record`` is the whole surgical_toolbox
    artifact. The saved cursor is not reconstructed: only an available endpoint
    checkpoint is shown. No timeline or interpolated/recorded frames are made.
    """
    if check is not None:
        check()
    if not isinstance(scene, SurgicalScene):
        raise ValueError('Exact source SurgicalScene required for imported display')
    # Identity is cached on scene construction. Refuse metadata/array rebinding
    # that would otherwise display changed anatomy under the old source digest.
    scene.require_cache_source()
    if not isinstance(record, Mapping) or not isinstance(record.get('branches'), Mapping):
        raise ValueError('Imported surgical archive requires branch records')
    branches = record['branches']
    selected = record.get('branchId') if branch_id is None else branch_id
    if not isinstance(selected, str) or not selected or selected not in branches:
        raise ValueError('Imported branch is unavailable')
    rows = []
    for key, snapshot in branches.items():
        if not isinstance(key, str) or not key or not isinstance(snapshot, Mapping):
            raise ValueError('Imported branch record is malformed')
        actions = _sequence(snapshot.get('actions'), 'actions')
        if len(actions) > MAX_STEPS or any(not isinstance(action, Mapping) for action in actions):
            raise ValueError('Imported action archive is malformed or exceeds the engine limit')
        rows.append({'id': key, 'steps': len(actions)})
    snapshot = branches[selected]
    if snapshot.get('schema') != SCHEMA or snapshot.get('sceneId') != scene.identity:
        raise ValueError('Imported scene or engine version differs')
    checkpoint = snapshot.get('checkpoint')
    if not isinstance(checkpoint, Mapping) or type(checkpoint.get('position')) is not int or checkpoint['position'] != len(snapshot['actions']):
        raise ValueError('Imported endpoint checkpoint position differs from its action count')
    if not isinstance(snapshot.get('stateId'), str) or checkpoint.get('stateId') != snapshot['stateId']:
        raise ValueError('Imported endpoint checkpoint identity differs')
    # Detach/validate JSON before giving any caller-owned state to display code.
    try:
        state = thaw_json(freeze_json(checkpoint.get('state')))
        options = {}
        if 'protectionContract' in snapshot:
            contract = thaw_json(freeze_json(snapshot['protectionContract']))
            if not isinstance(contract, dict):
                raise ValueError('Imported protection contract must be an object')
            protection, config = restore_protection_contract(scene, contract)
            options = {'protection': protection, 'guard_config': config}
        shell = _ArchiveDisplayShell(scene, **options)
        _state_shape(state, scene, shell.state)
        shell.state = state
        if shell.state_id != snapshot['stateId']:
            raise ValueError('Imported checkpoint state hash mismatch')
        if check is not None:
            check()
        view = shell.view(include_history=False)
    except (TypeError, KeyError, IndexError, OverflowError, AttributeError) as error:
        raise ValueError('Imported checkpoint cannot be displayed: ' + str(error)) from error
    # Collision counts require actual outcome receipts; an empty display-shell
    # history cannot establish zero collisions in the imported action prefix.
    view['metrics']['collisions'] = None
    full_cases = record.get('fullCase', {})
    if not isinstance(full_cases, Mapping):
        raise ValueError('Imported full-case metadata must be an object')
    full_case = thaw_json(freeze_json(full_cases.get(selected)))
    if full_case is not None and not isinstance(full_case, dict):
        raise ValueError('Imported branch metadata must be an object')
    verification = {'readOnly': True, 'status': 'unverified-import',
        'sourceSceneMatched': True, 'checkpointHashMatched': True,
        'causalReplayVerified': False, 'canonical': False, 'physicsActionsExecuted': 0,
        'protectionContractId': shell._protection_contract['contentHash'] if shell._protection_contract else None,
        'limitations': ['Imported endpoint self-consistency only; action causality and recorded outcomes are unverified.',
            'No editable state, canonical replay admission, frame cache, or RL state is created.',
            'Historical collision count is unavailable without verified outcome receipts.']}
    if check is not None:
        check()
    return {'view': view, 'outcome': None, 'branches': rows,
        'position': checkpoint['position'], 'branchId': selected, 'fullCase': full_case,
        'verification': verification}
