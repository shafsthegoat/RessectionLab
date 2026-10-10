"""Explicit, prospective sequential TRAIN configuration; no execution authority.

The action/horizon choice is supplied by the caller, never inferred from a
teacher outcome. This stage does not assert that any patient task is reachable.
"""
from dataclasses import asdict

from .core import freeze_json, semantic_digest, thaw_json
from .native_proposals import NominalCavityProposalConfig
from .public_target_context import VERSION as TARGET_CONTEXT

VERSION = 'fixed-four-TRAIN-sequential-cohort-v2'
ACCUMULATION = 'complete-trace-actionwise-shared-gradient-v1'
BALANCED_TEACHER_CE = 'balanced_STOP_motion_CE_v1'
RECOLLECT_TEACHERS = 'recollect_complete_pinned_plan_each_IL_update'
CACHED_TEACHERS = 'cache_complete_replayed_TRAIN_teacher_traces_v1'
TRAIN = ('ReMIND-008', 'ReMIND-010', 'ReMIND-020', 'ReMIND-025')
CLOSED = {'SELECT': ('ReMIND-013', 'ReMIND-037'), 'EVAL': ('ReMIND-067',)}


def sequential_learning_protocol(*, updates, max_steps, search, proposal_config,
        retention_mode='return_plus_opening_depth_v1', il_teacher_weighting=None,
        teacher_observations=RECOLLECT_TEACHERS, teacher_cache_payload_bytes=None):
    from .patient_planning_learning import PREFLIGHT_PROTOCOL
    # The sole longer endpoint is a fixed balanced-teacher optimization contrast.
    # Existing 1..8 protocols retain their exact records and bounds.
    if (type(updates) is not int or not (1 <= updates <= 8
            or updates == 64 and il_teacher_weighting == BALANCED_TEACHER_CE)):
        raise ValueError('Explicit 1..8 updates or balanced-teacher fixed64 endpoint required')
    if type(max_steps) is not int or not 1 <= max_steps <= 24:
        raise ValueError('Explicit horizon in 1..24 required')
    if il_teacher_weighting is not None and il_teacher_weighting != BALANCED_TEACHER_CE:
        raise ValueError('Unknown explicit IL teacher weighting')
    if teacher_observations not in (RECOLLECT_TEACHERS, CACHED_TEACHERS):
        raise ValueError('Unknown explicit teacher observation storage mode')
    if ((teacher_observations == RECOLLECT_TEACHERS and teacher_cache_payload_bytes is not None)
            or (teacher_observations == CACHED_TEACHERS and
                (type(teacher_cache_payload_bytes) is not int
                 or not 1 <= teacher_cache_payload_bytes <= 256*1024**2))):
        raise ValueError('Cached teachers require an explicit bounded payload allowance')
    if (type(proposal_config) is not NominalCavityProposalConfig
            or proposal_config.max_candidates not in (96, 120)
            or proposal_config.intermediate_opening_mm != 1.
            or proposal_config.tool_footprint_opening is not True):
        raise ValueError('Explicit 96 or120 footprint plus intermediate proposal configuration required')
    if (type(retention_mode) is not str or retention_mode not in {
            'return_plus_opening_depth_v1', 'return_plus_opening_depth_volume_v1'}):
        raise ValueError('Explicit reviewed depth or depth-volume two-lane retention required')
    candidates = proposal_config.max_candidates
    search = thaw_json(freeze_json(search))
    if (set(search) != {'max_calls', 'beam_width', 'seconds'}
            or any(type(v) is not int or v <= 0 for v in search.values())
            or search['beam_width'] != 2 or search['seconds'] > 300
            or not max(858, candidates*(2*max_steps-1)) <= search['max_calls'] <= 8192):
        raise ValueError('Explicit bounded two-lane search with a full-layer call allowance required')
    record = thaw_json(PREFLIGHT_PROTOCOL)
    record.update(version=VERSION, updates_per_method=updates, rl_episodes=4,
        scope='fixed_four_TRAIN_shared_population_pilot', population_training=True,
        public_target_context_variant=TARGET_CONTEXT,
        cohort_execution={
            'max_steps': max_steps, 'search': search,
            'retention_mode': retention_mode,
            'retained_prefix_diagnostics': True,
            'proposal_config': proposal_config.to_record(),
            'proposal_rule_hash': proposal_config.fingerprint,
            'accumulation': ACCUMULATION,
            'patient_order': list(TRAIN), 'live_source_patients': 1,
            'teacher_observations': teacher_observations,
            'task_condition': 'PARTIAL_TARGET_PROGRESS',
            'heldout_execution': False})
    if il_teacher_weighting is not None:
        record['cohort_execution']['il_teacher_weighting'] = il_teacher_weighting
    if teacher_observations == CACHED_TEACHERS:
        record['cohort_execution']['teacher_cache_payload_bytes'] = teacher_cache_payload_bytes
    return freeze_json(record)


def validate_sequential_protocol(protocol):
    if protocol.get('version') != VERSION:
        raise ValueError('Exact sequential cohort version required')
    execution = protocol.get('cohort_execution', {})
    config = NominalCavityProposalConfig(**thaw_json(execution.get('proposal_config', {})))
    expected = sequential_learning_protocol(updates=protocol.get('updates_per_method'),
        max_steps=execution.get('max_steps'), search=execution.get('search', {}), proposal_config=config,
        retention_mode=execution.get('retention_mode'),
        il_teacher_weighting=execution.get('il_teacher_weighting'),
        teacher_observations=execution.get('teacher_observations'),
        teacher_cache_payload_bytes=execution.get('teacher_cache_payload_bytes'))
    if semantic_digest(expected) != semantic_digest(protocol):
        raise ValueError('Sequential objective, scheduling or task options changed')
    return expected


def preview_budget_sizing(protocol):
    """Conservative dispatch estimate; actual instrumented counters are authority.

Each visit includes one construction inventory. A complete collection, native
replay and independent replay check each receive a horizon allowance. IL now
recollects/replays its teacher every update, in addition to the original RL work.
Search receives one root inventory plus a child inventory allowance per call.
The opt-in cache retains this conservative recollection envelope; actual cache
fill, reuse, native and forward costs are counted separately by the runner.
This does not prove runtime/RSS feasibility and is not a measured cost.
"""
    protocol = validate_sequential_protocol(protocol)
    execution = protocol['cohort_execution']; updates = protocol['updates_per_method']
    horizon = execution['max_steps']; visits = 1+2*updates+2
    candidates = execution['proposal_config']['max_candidates']
    dispatches = visits*(1+3*horizon)+1+execution['search']['max_calls']
    return {'patients': 4, 'horizon': horizon, 'visits_per_patient': visits,
        'candidate_inventory_ceiling': candidates,
        'teacher_search_calls_per_patient': execution['search']['max_calls'],
        'suggested_preview_cap': 4*candidates*dispatches,
        'suggested_forward_cap': 4*horizon*(updates+2*updates+2),
        'interpretation': 'sizing only; no measured runtime or feasibility claim'}


def sequential_limits(protocol, *, worker_seconds, memory_bytes, output_bytes):
    sizing = preview_budget_sizing(protocol)
    limits = {'max_steps': protocol['cohort_execution']['max_steps'],
        'max_optimizer_updates': 2*protocol['updates_per_method'],
        'max_native_previews': sizing['suggested_preview_cap'],
        'max_policy_forwards': sizing['suggested_forward_cap'],
        'worker_seconds': worker_seconds, 'memory_bytes': memory_bytes, 'threads': 1,
        'search': thaw_json(protocol['cohort_execution']['search']),
        'output_bytes': output_bytes, 'checkpoint_bytes': 8*1024**2}
    validate_limits(protocol, limits)
    return freeze_json(limits)


def validate_limits(protocol, limits):
    protocol = validate_sequential_protocol(protocol)
    keys = {'max_steps', 'max_optimizer_updates', 'max_native_previews', 'max_policy_forwards',
        'worker_seconds', 'memory_bytes', 'threads', 'search', 'output_bytes', 'checkpoint_bytes'}
    if set(limits) != keys or any(type(limits[k]) is not int or limits[k] <= 0 for k in keys-{'search'}):
        raise ValueError('Exact positive bounded sequential limits required')
    sizing = preview_budget_sizing(protocol)
    # This is the conservative two-method context envelope, not permission to
    # execute both methods. The owned IL-only contrast enforces its smaller
    # actual counters separately. Extend only to this protocol's derived size.
    long_balanced = (protocol['updates_per_method'] == 64
        and protocol['cohort_execution'].get('il_teacher_weighting') == BALANCED_TEACHER_CE)
    preview_ceiling = max(4194304, sizing['suggested_preview_cap']) if long_balanced else 4194304
    forward_ceiling = max(4096, sizing['suggested_forward_cap']) if long_balanced else 4096
    if (limits['max_steps'] != protocol['cohort_execution']['max_steps']
            or limits['search'] != protocol['cohort_execution']['search']
            or limits['max_optimizer_updates'] != 2*protocol['updates_per_method']
            or not sizing['suggested_preview_cap'] <= limits['max_native_previews'] <= preview_ceiling
            or not sizing['suggested_forward_cap'] <= limits['max_policy_forwards'] <= forward_ceiling
            or limits['worker_seconds'] > 3600 or limits['memory_bytes'] > 3*1024**3
            or limits['threads'] != 1 or not 16*1024**2 <= limits['output_bytes'] <= 256*1024**2
            or limits['checkpoint_bytes'] != 8*1024**2):
        raise ValueError('Sequential limits differ from declared protocol or bounded staging envelope')


def validate_factories(factories):
    if set(factories) != set(TRAIN) or any(not callable(factories[s]) for s in TRAIN):
        raise ValueError('Exactly four fixed TRAIN visit factories required; SELECT/EVAL closed')
