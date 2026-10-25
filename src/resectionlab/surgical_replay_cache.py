"""Bounded, source-owned checkpoints for the desktop's recorded-state cursor.

Archives on disk keep their existing complete action log and independently
verified endpoint checkpoint. This cache is disposable: an unrecognized or
changed archive takes the ordinary full replay admission path. Checkpoints keep
all engine state, including irreversible effects; seeking never inverts effects.
"""
from collections import OrderedDict
import copy
import sys

import numpy as np

from .surgical_engine import SCHEMA, SurgicalEngine


def _owned_bytes(value, seen=None):
    """Conservative Python-object footprint, excluding shared source anatomy."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_owned_bytes(k, seen) + _owned_bytes(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(_owned_bytes(v, seen) for v in value)
    return size


class SurgicalReplayCache:
    """One session's LRU of demanded checkpoints, bounded by bytes and count.

    Only this owner sees retained state/actions/history. Every returned engine
    receives detached writable copies. Source arrays and the exact-cover cache
    may be shared; occupancy and deformation solver caches are discarded.
    """
    def __init__(self, *, max_bytes=16 * 1024**2, max_checkpoints=8, max_archives=4, verified_cache=False):
        if min(max_bytes, max_checkpoints, max_archives) < 1:
            raise ValueError('Replay cache budgets must be positive')
        self.max_bytes = max_bytes
        self.max_checkpoints = max_checkpoints
        self.max_archives = max_archives
        self._archives = OrderedDict()
        self._serial = 0
        if verified_cache is True:
            from .surgical_verified_cache import VerifiedSurgicalCache
            verified_cache = VerifiedSurgicalCache()
        self.verified_cache = verified_cache or None

    @property
    def statistics(self):
        grids, covers, protections, sources = {}, {}, {}, {}
        for archive in self._archives.values():
            template = archive['template']
            grids[id(template.grid)] = template.grid
            sources[id(template.scene)] = template.scene
            cover = getattr(template, '_cell_cover_cache', None)
            if cover is not None: covers[id(cover)] = cover
            if template.protection is not None: protections[id(template.protection)] = template.protection
        def arrays_bytes(objects):
            arrays = {id(value): value for obj in objects for value in vars(obj).values() if isinstance(value, np.ndarray)}
            return sum(value.nbytes for value in arrays.values())
        # These source resources are shared with live engines, not charged as
        # private checkpoint payload. Report them explicitly rather than hiding
        # a retained patient grid or a pure-cover LRU behind the 16 MiB budget.
        grid_bytes = arrays_bytes(grids.values())
        cover_bytes = sum(cover.stats()['payloadBytes'] for cover in covers.values())
        candidate_bytes = sum(cover.stats()['candidatePayloadBytes'] for cover in covers.values())
        protection_bytes = arrays_bytes(protections.values())
        return {'archives': len(self._archives),
                'checkpoints': sum(len(a['checkpoints']) for a in self._archives.values()),
                'ownedBytes': sum(a['bytes'] for a in self._archives.values()),
                'maxBytes': self.max_bytes, 'maxCheckpoints': self.max_checkpoints,
                'sharedSourceArrayBytes': arrays_bytes(sources.values()),
                'sharedGeometryGrids': len(grids), 'sharedGeometryGridBytes': grid_bytes,
                'sharedPureCoverPayloadBytes': cover_bytes, 'sharedCandidateGridPayloadBytes': candidate_bytes,
                'sharedProtectionArrayBytes': protection_bytes,
                'memoryMeaning': 'ownedBytes caps private state/actions/history Python objects; shared source/grid/cover/protection arrays reported separately; excludes scipy tree native allocation'}

    def _trim(self, keep):
        def excessive():
            stats = self.statistics
            return (stats['ownedBytes'] > self.max_bytes or stats['checkpoints'] > self.max_checkpoints
                    or stats['archives'] > self.max_archives)
        while self._archives and excessive():
            oldest = next(iter(self._archives))
            if oldest != keep or len(self._archives) > 1:
                del self._archives[oldest]
                continue
            archive = self._archives[oldest]
            removable = next((p for p in archive['checkpoints'] if p != len(archive['snapshot']['actions'])), None)
            if removable is None:
                del self._archives[oldest]
            else:
                del archive['checkpoints'][removable]
                archive['bytes'] = _owned_bytes((archive['snapshot'], archive['history'], archive['checkpoints']))

    @staticmethod
    def _shell(engine):
        result = copy.copy(engine)
        result.state, result.actions, result.history = {}, [], []
        result._occupancy_cache = None
        result.__dict__.pop('_deformation_system_v2', None)
        result.__dict__.pop('_drill_step_powers', None)
        return result

    def _template(self, full):
        template = self._shell(full)
        if self._archives:
            shared = next(iter(self._archives.values()))['template']
            # Full archive admission has finished before sharing any runtime.
            # GeometryScene and exact covers contain no mutable patient effects.
            template.grid = shared.grid
            if hasattr(shared, '_cell_cover_cache'):
                template._cell_cover_cache = shared._cell_cover_cache
            if template._protection_contract == shared._protection_contract:
                template.protection = shared.protection
                template.guard_config = shared.guard_config
                template._protected_ids = shared._protected_ids
        return template

    def seek(self, current, snapshot, position, *, check=lambda: None, include_history=False):
        """Admit a complete branch, then replay from its nearest prior checkpoint.

        A live engine may admit its own exact endpoint. Every other new archive
        is fully reconstructed by SurgicalEngine.restore before caching. Exact
        snapshot comparison includes actions, source, guards and endpoint state;
        neither a timeline index nor a caller-supplied state ID alone is trusted.
        """
        check()
        current.scene.require_cache_source()
        # Case replacement drops all old source references, including grids.
        if self._archives and next(iter(self._archives.values()))['template'].scene is not current.scene:
            self._archives.clear()
        if type(position) is not int or not 0 <= position <= len(snapshot['actions']):
            raise ValueError('Invalid history position')
        if self.verified_cache is not None:
            persisted = self.verified_cache.restore(type(current), current.scene, snapshot,
                position=position, check=check, include_history=include_history)
            if persisted is not None:
                return persisted
        found = next(((key, a) for key, a in self._archives.items()
                      if a['template'].scene is current.scene and a['snapshot'] == snapshot), None)
        if found is None:
            checkpoint = snapshot.get('checkpoint')
            live_endpoint = (snapshot.get('schema') == SCHEMA and snapshot.get('sceneId') == current.scene.identity
                and snapshot.get('protectionContract') == current._protection_contract
                and snapshot['actions'] == current.actions and snapshot.get('stateId') == current.state_id
                and (checkpoint is None or (checkpoint.get('position') == len(current.actions)
                     and checkpoint.get('stateId') == snapshot['stateId'] and checkpoint.get('state') == current.state)))
            full = current if live_endpoint else SurgicalEngine.restore(current.scene, snapshot, check=check)
            check()
            # Size before copying: oversized states use the full-replay fallback.
            if _owned_bytes((snapshot, full.history)) > self.max_bytes:
                result = full.branch(position, check=check)
                return (result, copy.deepcopy(full.history)) if include_history else result
            owned_snapshot, history = copy.deepcopy((snapshot, full.history))
            state = owned_snapshot.get('checkpoint', {}).get('state')
            if state is None:
                state = copy.deepcopy(full.state)
            self._serial += 1
            key = self._serial
            archive = {'snapshot': owned_snapshot, 'history': history, 'template': self._template(full),
                       'checkpoints': OrderedDict([(len(full.actions), state)])}
            archive['bytes'] = _owned_bytes((owned_snapshot, history, archive['checkpoints']))
            self._archives[key] = archive
            # Admission itself owns an endpoint checkpoint. Enforce its limits
            # before replay can be interrupted; cancellation must not leave a
            # growing collection of admitted but unfinished branch requests.
            # An oversized entry may be evicted here; this call can still use
            # its detached local archive without retaining it in the cache.
            self._trim(key)
        else:
            key, archive = found
            self._archives.move_to_end(key)
        expected = archive['history'][position - 1]['stateAfter'] if position else (
            archive['history'][0]['stateBefore'] if archive['history'] else snapshot['stateId'])
        previous = max((p for p in archive['checkpoints'] if p <= position), default=None)
        # A newly checked alternative can share the live decision state even
        # when its archive only has an endpoint checkpoint. Match the complete
        # source, guard, action prefix and physical-state hash AFTER full archive
        # admission, then detach it like any other demanded checkpoint. No
        # imported checkpoint or caller-supplied ID grants this shortcut.
        live_prefix = (position not in archive['checkpoints']
            and current.scene is archive['template'].scene
            and current._protection_contract == archive['template']._protection_contract
            and len(current.actions) == position
            and current.actions == archive['snapshot']['actions'][:position]
            and current.state_id == expected)
        if live_prefix:
            result = self._shell(archive['template'])
            result.state = copy.deepcopy(current.state)
            result.actions = copy.deepcopy(archive['snapshot']['actions'][:position])
            result.history = copy.deepcopy(archive['history'][:position])
            start = position
        elif previous is None:
            result = archive['template'].fresh()
            result.grid = archive['template'].grid
            if hasattr(archive['template'], '_cell_cover_cache'):
                result._cell_cover_cache = archive['template']._cell_cover_cache
            start = 0
        else:
            start = previous
            result = self._shell(archive['template'])
            result.state = copy.deepcopy(archive['checkpoints'][start])
            result.actions = copy.deepcopy(archive['snapshot']['actions'][:start])
            result.history = copy.deepcopy(archive['history'][:start])
            archive['checkpoints'].move_to_end(start)
        for action in archive['snapshot']['actions'][start:position]:
            check()
            if result.step(action, check=check)['status'] == 'rejected':
                raise ValueError('Saved action no longer replays')
            check()
        if result.state_id != expected:
            raise ValueError('Replay checkpoint ancestry differs')
        check()
        if position not in archive['checkpoints']:
            archive['checkpoints'][position] = copy.deepcopy(result.state)
            archive['bytes'] = _owned_bytes((archive['snapshot'], archive['history'], archive['checkpoints']))
        self._trim(key)
        return (result, copy.deepcopy(archive['history'])) if include_history else result
