"""Process-local, IL-only reuse of complete admitted and replayed TRAIN teachers.

The owned collector/replayer is trusted, as in PatientTrainingTrace admission;
this is not an authenticator for arbitrary JSON or hostile Python. No task,
case, source array, replay callback, tensor graph or policy is retained here.
Observations already own immutable byte-backed crops. RL stays on-policy.
"""
from dataclasses import dataclass
import json

from .core import freeze_json, semantic_digest, thaw_json
from .patient_planning_cohort_spec import CACHED_TEACHERS, TRAIN, validate_sequential_protocol
from .patient_planning_learning import PatientTrainingTrace, PatientTrainSession
from .patient_planning_preflight import _history_identity


_ARRAYS = ('image_channels', 'coverage', 'channel_available', 'affine_ras_mm',
           'spacing_mm', 'action_geometry', 'action_mask', 'state_features')


def _payload_bytes(trace):
    """Conservative per-row array bytes plus serialized metadata, not Python RSS."""
    arrays = sum(getattr(row.observation, name).nbytes
                 for row in trace.transitions for name in _ARRAYS)
    record = {'context': trace.context.record(), 'history': trace.history,
        'seal': trace.seal_hash, 'transitions': [
            {'action': r.action_id, 'reward': r.reward, 'terminated': r.terminated,
             'observation_hash': r.observation.fingerprint,
             'ids': r.observation.action_ids, 'tools': r.observation.action_tool_ids,
             'provenance': r.observation.channel_provenance,
             'public_target_context': None if r.observation.public_target_context is None
                 else r.observation.public_target_context.record()}
            for r in trace.transitions]}
    metadata = len(json.dumps(thaw_json(freeze_json(record)), sort_keys=True, separators=(',', ':'),
                              allow_nan=False).encode('utf-8'))
    return int(arrays), metadata


@dataclass(frozen=True)
class _Entry:
    trace: PatientTrainingTrace
    pin: object


class PatientTeacherTraceCache:
    def __init__(self, protocol):
        self.protocol = validate_sequential_protocol(freeze_json(protocol))
        execution = self.protocol['cohort_execution']
        if execution['teacher_observations'] != CACHED_TEACHERS:
            raise ValueError('Teacher cache must be declared before context admission')
        self._protocol_hash = semantic_digest(self.protocol)
        self._entries = ()
        self._pins_hash = semantic_digest(())

    def _require(self):
        if (semantic_digest(self.protocol) != self._protocol_hash
                or semantic_digest(tuple(e.pin for e in self._entries)) != self._pins_hash):
            raise ValueError('Teacher cache metadata changed')
        for entry in self._entries:
            trace = entry.trace.require()
            if (trace.seal_hash != entry.pin['trace_seal']
                    or trace.context.fingerprint != entry.pin['context_hash']
                    or trace.behavior_parameter_hash is not None
                    or tuple(r.observation.fingerprint for r in trace.transitions) != entry.pin['observations']):
                raise ValueError('Cached teacher changed since verified replay')

    def add_replayed(self, trace, sealed):
        """Call inside the original source visit, immediately after full replay.

The caller still performs _SequentialVisits' source weakref-release check.
The sealed replay object is validated and discarded; only compact pins remain.
"""
        self._require()
        if type(trace) is not PatientTrainingTrace:
            raise TypeError('Exact admitted complete teacher trace required')
        trace.require(); context = trace.context.record()
        if (len(self._entries) >= len(TRAIN) or context['subject'] != TRAIN[len(self._entries)]
                or context['learning_protocol_hash'] != self._protocol_hash
                or trace.behavior_parameter_hash is not None):
            raise ValueError('Ordered fixed TRAIN teachers under this storage protocol required')
        expected_plan = {'context_hash': trace.context.fingerprint,
            'source_hash': context['source_hash'], 'decision_model_hash': context['decision_model_hash'],
            'initial_observation_hash': trace.transitions[0].observation.fingerprint,
            'max_steps': trace.context.max_steps, 'actions': [r.action_id for r in trace.transitions],
            'history': trace.history, 'terminal_reason': 'STOP' if trace.transitions[-1].action_id == 'STOP' else 'HORIZON',
            'parameter_hash': None, 'architecture_hash': None, 'learning_updates': 0}
        plan_hash = semantic_digest(expected_plan)
        audit = sealed['independent_geometry']
        replayed = sealed['replayed_history']
        if (sealed['method'] != 'SEARCH' or sealed['plan_seal'] != plan_hash
                or semantic_digest(sealed['plan']) != plan_hash
                or _history_identity(replayed) != _history_identity(trace.history)
                or audit.get('schema') != 'native-spatial-independent-episode-v1'
                or audit.get('accepted') is not True or audit.get('complete_episode') is not True
                or audit.get('source_hash') != context['source_hash']
                or audit.get('decision_model_hash') != context['decision_model_hash']
                or audit.get('committed_history_hash') != semantic_digest(replayed)
                or audit.get('geometry', {}).get('feasible') is not True):
            raise ValueError('Full exact native/independent replay is required before caching')
        arrays, metadata = _payload_bytes(trace)
        pin = freeze_json({'subject': context['subject'], 'context_hash': trace.context.fingerprint,
            'trace_seal': trace.seal_hash, 'plan_seal': plan_hash,
            'independent_replay_hash': semantic_digest(audit),
            'observations': [r.observation.fingerprint for r in trace.transitions],
            'steps': len(trace.transitions), 'stop_steps': sum(r.action_id == 'STOP' for r in trace.transitions),
            'array_bytes': arrays, 'metadata_json_bytes': metadata})
        total = sum(e.pin['array_bytes']+e.pin['metadata_json_bytes'] for e in self._entries)+arrays+metadata
        if total > self.protocol['cohort_execution']['teacher_cache_payload_bytes']:
            raise MemoryError('Declared teacher cache payload budget exceeded; no recollection fallback')
        entries = self._entries + (_Entry(trace, pin),)
        pins_hash = semantic_digest(tuple(e.pin for e in entries))
        self._entries, self._pins_hash = entries, pins_hash
        return thaw_json(pin)

    def trace_for_il(self, session, subject):
        self._require()
        if type(session) is not PatientTrainSession:
            raise TypeError('Exact admitted IL session required')
        session.require('IL')
        if (len(self._entries) != len(TRAIN) or semantic_digest(session.protocol) != self._protocol_hash
                or tuple(c.fingerprint for c in session.contexts) != tuple(e.pin['context_hash'] for e in self._entries)):
            raise ValueError('All four replayed teachers and the exact common session are required')
        entry = self._entries[TRAIN.index(subject)]
        session.require_trace(entry.trace)
        return entry.trace

    def record(self):
        self._require()
        return {'mode': CACHED_TEACHERS, 'complete': len(self._entries) == len(TRAIN),
            'learning_protocol_hash': self._protocol_hash, 'cache_seal': self._pins_hash,
            'traces': [thaw_json(e.pin) for e in self._entries],
            'array_bytes': sum(e.pin['array_bytes'] for e in self._entries),
            'metadata_json_bytes': sum(e.pin['metadata_json_bytes'] for e in self._entries),
            'memory_scope': 'owned crop buffers plus JSON byte estimate; Python overhead and tree RSS remain supervised',
            'RL_reuse': False}
