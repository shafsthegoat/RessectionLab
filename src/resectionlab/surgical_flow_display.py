"""Bounded historical field-write indicators, outside physical engine state.

Only observed replay owners install a recorder. Operations are samples, not
particle mass/species transport or routes through an unobserved tool path.
"""
from __future__ import annotations

import json
import math

SCHEMA = 'resectionlab.recorded-field-transfers.v1'
MAX_OPERATIONS = 256
MAX_ENCODED_BYTES = 64 * 1024
_BLOCK_KEYS = ('schema', 'sceneId', 'actionIndex', 'frameIndex', 'beforeStateId',
    'stateId', 'startSeconds', 'endSeconds', 'truncated', 'operations')
_OP_KEYS = ('kind', 'field', 'provenance', 'fromCell', 'toCell',
    'volumeMm3', 'startSeconds', 'endSeconds')


def _need(condition):
    if not condition:
        raise ValueError('Invalid recorded field-transfer display metadata')


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _size(value):
    return len(json.dumps(value, separators=(',', ':'), allow_nan=False, ensure_ascii=True).encode('ascii'))


def validate_flow_display(value, *, scene_id, action_index, frame_index, before_state_id,
                          state_id, start_seconds, end_seconds, shape=None):
    """Validate authenticated context and structure; never reconstruct physics.

    A prepared package's generic disk reader has no source grid; its prefix
    owner repeats this validation with the admitted shape before publishing a
    view. Free occupancy at the original write is enforced by the recorder's
    caller, not inferred from later endpoint anatomy.
    """
    _need(type(value) is dict and set(value) == set(_BLOCK_KEYS))
    _need(type(value['schema']) is str and value['schema'] == SCHEMA)
    for key, expected in (('sceneId', scene_id), ('beforeStateId', before_state_id), ('stateId', state_id)):
        _need(type(value[key]) is str and len(value[key]) == 71 and value[key].startswith('sha256:') and
              all(c in '0123456789abcdef' for c in value[key][7:]) and value[key] == expected)
    for key, expected in (('actionIndex', action_index), ('frameIndex', frame_index)):
        _need(type(value[key]) is int and 0 <= value[key] < 2**63 and value[key] == expected)
    _need(_number(value['startSeconds']) and _number(value['endSeconds']) and
          value['startSeconds'] == start_seconds and value['endSeconds'] == end_seconds and
          0 <= start_seconds <= end_seconds and type(value['truncated']) is bool)
    operations = value['operations']
    _need(type(operations) is list and len(operations) <= MAX_OPERATIONS)
    if start_seconds == end_seconds:
        _need(not operations)
    if frame_index == 0:
        _need(before_state_id == state_id and start_seconds == end_seconds and not operations)
    limit = math.prod(shape) if shape is not None else 2**63
    previous_start, previous_end = start_seconds, start_seconds
    for operation in operations:
        _need(type(operation) is dict and set(operation) == set(_OP_KEYS))
        kind, field, provenance = operation['kind'], operation['field'], operation['provenance']
        _need(type(kind) is str and kind in ('transfer', 'uptake', 'deposit', 'escape'))
        _need(type(field) is str and field in ('fluid', 'debris') and type(provenance) is str)
        _need(provenance in (('debris',) if field == 'debris' else
                            ('mixed', 'irrigation', 'wound', 'declared-source')))
        if kind in ('transfer', 'uptake', 'escape'):
            _need(provenance == ('mixed' if field == 'fluid' else 'debris'))
        left, right = operation['fromCell'], operation['toCell']
        for cell in (left, right):
            _need(cell is None or type(cell) is int and 0 <= cell < limit)
        _need((kind == 'transfer' and left is not None and right is not None and left != right) or
              (kind == 'uptake' and left is not None and right is None) or
              (kind == 'deposit' and left is None and right is not None) or
              (kind == 'escape' and right is None))
        if kind == 'transfer' and shape is not None:
            _, ny, nz = shape
            a = (left // (ny*nz), left // nz % ny, left % nz)
            b = (right // (ny*nz), right // nz % ny, right % nz)
            _need(sum(abs(x-y) for x, y in zip(a, b)) == 1)
        start, end = operation['startSeconds'], operation['endSeconds']
        _need(_number(operation['volumeMm3']) and operation['volumeMm3'] > 0 and
              _number(start) and _number(end) and
              start_seconds <= start <= end <= end_seconds and
              start >= previous_start and end >= previous_end)
        previous_start, previous_end = start, end
    _need(_size(value) <= MAX_ENCODED_BYTES)
    return value


def trim_flow_display(value):
    """Drop optional samples without weakening their immutable context."""
    return {**value, 'truncated': True, 'operations': []}


def trim_frame_flows(frames, *, omit=False):
    """Private staging only; never modify already-published frame graphs."""
    changed = False
    for row in frames:
        if 'flowDisplay' not in row:
            continue
        flow = row['flowDisplay']
        if omit:
            del row['flowDisplay']; changed = True
        elif flow['operations']:
            row['flowDisplay'] = trim_flow_display(flow); changed = True
    return changed


class FlowDisplayTrace:
    """First-256 sampling with separate failed-substep staging and no counters."""
    def __init__(self, scene_id, action_index, start_seconds, shape=None):
        self.scene_id, self.action_index = scene_id, action_index
        self.shape = shape
        self.cell_limit = math.prod(shape) if shape is not None else 2**63
        self.clock = self.previous_time = float(start_seconds)
        self.previous_state = None
        self.frame_index = 0
        self.operations = []; self.pending = []
        self.truncated = self.pending_truncated = False
        self.active = True

    def record(self, kind, field, provenance, left, right, volume):
        if not self.active:
            return
        # Unsupported optional values never introduce conversions/callbacks or
        # change the outcome/error path of the original numerical write.
        if not _number(volume) or volume <= 0:
            self.pending_truncated |= not _number(volume)
            return
        if any(cell is not None and (type(cell) is not int or not 0 <= cell < self.cell_limit)
               for cell in (left, right)):
            self.pending_truncated = True
            return
        if len(self.operations) + len(self.pending) >= MAX_OPERATIONS:
            self.pending_truncated = True
            return
        self.pending.append({'kind': kind, 'field': field, 'provenance': provenance,
            'fromCell': left, 'toCell': right, 'volumeMm3': volume})

    def commit_substep(self, end_seconds):
        end = float(end_seconds)
        if end == self.clock:
            # A positive physical dt can round elapsed+dt back to elapsed.
            # Field relaxation/uptake may still write nonzero quantities, but
            # they have no representable observed interval. Omit the samples,
            # rather than rejecting a historically valid physical action.
            self.truncated |= bool(self.pending) or self.pending_truncated
            self.discard_substep()
            return
        for operation in self.pending:
            operation.update(startSeconds=self.clock, endSeconds=end)
        self.operations.extend(self.pending)
        self.truncated |= self.pending_truncated
        self.pending = []; self.pending_truncated = False
        self.clock = end

    def discard_substep(self):
        self.pending = []; self.pending_truncated = False

    def capture(self, state_id, end_seconds):
        self.discard_substep()  # An unsuccessful final attempted step is absent.
        end = float(end_seconds)
        if end == self.previous_time and self.operations:
            self.operations = []; self.truncated = True
        block = {'schema': SCHEMA, 'sceneId': self.scene_id, 'actionIndex': self.action_index,
            'frameIndex': self.frame_index, 'beforeStateId': self.previous_state or state_id,
            'stateId': state_id, 'startSeconds': self.previous_time, 'endSeconds': end,
            'truncated': self.truncated, 'operations': self.operations}
        while _size(block) > MAX_ENCODED_BYTES and block['operations']:
            block['operations'].pop(); block['truncated'] = True
        self.operations = []; self.truncated = False
        self.previous_state, self.previous_time = state_id, end
        self.frame_index += 1
        return block

    def close(self):
        self.active = False
        self.operations = []; self.discard_substep()
