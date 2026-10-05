"""Independent outcome accounting for a completed geometric spatial episode.

This is an evaluation adapter, not an actor input or patient loader. The native
history checker owns complete-tool geometry; this layer independently counts
unique removed/contacted cells and reconstructs the declared geometric reward.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
import time
from types import SimpleNamespace

import numpy as np

from .core import semantic_digest
from .evaluation import independent_check_native_history
from .native_spatial_task import NativeSpatialTask


def _indices(value, shape):
    array = np.asarray(value)
    if array.size == 0:
        return set()
    if (array.ndim != 2 or array.shape[1] != 3 or array.dtype.kind not in 'iu'
            or np.any(array < 0) or np.any(array >= shape)
            or len(np.unique(array, axis=0)) != len(array)):
        raise ValueError('Complete unique source-cell indices required')
    return {tuple(int(v) for v in row) for row in array}


def _same_number(claim, actual, name):
    if (isinstance(claim, (bool, np.bool_)) or not isinstance(claim, (int, float, np.number))
            or not math.isfinite(float(claim)) or not math.isclose(float(claim), float(actual), rel_tol=1e-7, abs_tol=1e-7)):
        raise ValueError('Independent episode accounting mismatch: ' + name)


def evaluate_native_spatial_episode(task, *, minimum_target_cells=1, metrics=None,
                                    cancelled=None, distance_backend='batch', distance_batch_size=256):
    """Audit a complete committed episode, returning success separately from reward.

    ``minimum_target_cells`` counts fully removed source cells with positive
    supplied/reference target membership. Target-weighted volume and fraction
    remain separate metrics. A passing STOP-only episode is never target access.
    Invalid action attempts are recorded by the caller, not inferred from a
    history containing only accepted transitions. Batch/scalar geometry choice
    is explicit in the returned receipt; neither is a tissue mechanics model.
    """
    started = time.perf_counter()
    if not isinstance(task, NativeSpatialTask) or type(minimum_target_cells) is not int or minimum_target_cells < 1:
        raise ValueError('Typed native task and positive source-cell success threshold required')
    current = task.metrics()
    recorded = deepcopy(current if metrics is None else metrics)
    history = recorded.get('history')
    if (not isinstance(history, list) or not 1 <= len(history) <= task.max_steps
            or recorded.get('terminated') is not True or not task.terminated
            or recorded.get('steps') != len(history)):
        raise ValueError('Only a completed STOP-or-horizon episode can be accepted')
    if semantic_digest(history) != semantic_digest(current['history']):
        raise ValueError('Evaluation history differs from the actually committed episode')
    stops = [i for i, row in enumerate(history) if row.get('action_id') == 'STOP']
    if (stops and stops != [len(history) - 1]) or (not stops and len(history) != task.max_steps):
        raise ValueError('STOP must terminate immediately; a non-STOP prefix is incomplete')
    for name in ('source_hash', 'reference_hash', 'decision_model_hash'):
        if recorded.get(name) != current[name]:
            raise ValueError('Frozen episode source/objective binding changed: ' + name)
    case = task.case
    if recorded.get('planning_estimator_only') is not False:
        raise ValueError('A planning clone cannot substitute for committed evaluator execution')
    source = SimpleNamespace(mri=case.structural_intensity, affine=case._native_affine_ras_mm,
                             frame='RAS+', semantic_hash=case.source_hash)
    audit = independent_check_native_history(
        source, case.tools, history, tissue_mask=case.observed_support, access=case.access,
        hard_exclusion=task._config.hard_exclusion, geometry_frame='RAS+', cancelled=cancelled,
        distance_backend=distance_backend, distance_batch_size=distance_batch_size,
    )
    result = {
        'schema': 'native-spatial-independent-episode-v1',
        'source_hash': case.source_hash, 'reference_hash': case.reference_hash,
        'decision_model_hash': task.decision_model_hash,
        'committed_history_hash': semantic_digest(history),
        'complete_episode': True, 'geometry': asdict(audit),
        'distance_backend': distance_backend, 'distance_batch_size': distance_batch_size,
        'minimum_positive_target_source_cells': minimum_target_cells,
        'invalid_action_attempts': None,
        'invalid_action_attempts_reason': 'caller must count rejected requests; committed history contains accepted transitions only',
        'scope': 'geometric annotation/estimate-conditioned task, not physical surgery or clinical outcome prediction',
    }
    if not audit.feasible:
        return {**result, 'accepted': False, 'target_access_success': None, 'outcomes': None,
                'evaluation_seconds': time.perf_counter() - started}
    shape = case.observed_support.shape
    volume = float(abs(np.linalg.det(case._native_affine_ras_mm[:3, :3])))
    removed, contacts = set(), set()
    target_volume = normal_volume = reward = path_length = 0.
    previous_tool = None
    tool_changes = 0
    weights = task.reward_spec
    for record in history:
        if cancelled is not None and cancelled():
            raise InterruptedError('Independent outcome accounting cancelled')
        if record['action_id'] == 'STOP':
            if record.get('microsteps') or _indices(record.get('removed_indices_native', ()), shape):
                raise ValueError('STOP cannot carry a removal history')
            for name in ('reward', 'target_removed_mm3', 'normal_removed_mm3', 'complete_tool_path_length_mm'):
                _same_number(record.get(name), 0., 'STOP ' + name)
            continue
        cells = _indices(record['removed_indices_native'], shape)
        if removed & cells:
            raise ValueError('Repeated removed tissue cannot earn reward twice')
        contact = _indices(record.get('contact_indices_native', ()), shape)
        micro_contact = set().union(*(_indices(step.get('contact_indices_native', ()), shape)
                                      for step in record['microsteps']))
        if contact != micro_contact or any(not case.observed_support[key] for key in contact):
            raise ValueError('Macro contact accounting differs from source-tissue microsteps')
        # Reconstruct mass independently from individual stored memberships;
        # do not reuse the simulator's reduction or its reported action totals.
        target = math.fsum(float(case.reference_target[cell]) for cell in sorted(cells)) * volume
        normal = len(cells) * volume - target
        distance = float(np.linalg.norm(np.asarray(record['tip_mm']) - np.asarray(record['entry_mm'])))
        change = int(previous_tool is not None and previous_tool != record['tool_id'])
        actual_reward = (weights.target_per_mm3 * target - weights.normal_per_mm3 * normal
                         - weights.action_cost - 2 * weights.motion_per_mm * distance
                         - weights.tool_change_cost * change)
        for name, value in (('target_removed_mm3', target), ('normal_removed_mm3', normal),
                            ('reward', actual_reward), ('insertion_distance_mm', distance),
                            ('complete_tool_path_length_mm', 2 * distance)):
            _same_number(record.get(name), value, name)
        removed.update(cells)
        contacts.update(contact)
        target_volume += target
        normal_volume += normal
        reward += actual_reward
        path_length += 2 * distance
        tool_changes += change
        previous_tool = record['tool_id']
    union = np.asarray(sorted(removed), dtype=int)
    target_cells = int(np.count_nonzero(case.reference_target[tuple(union.T)] > 0)) if len(union) else 0
    target_total = float(case.reference_target.sum(dtype=np.float64)) * volume
    outcomes = {
        'target_removed_mm3': target_volume, 'normal_removed_mm3': normal_volume,
        'simulated_removed_volume_mm3': len(removed) * volume,
        'cumulative_contacted_tissue_upper_bound_mm3': len(contacts) * volume,
        'currently_retained_contacted_tissue_upper_bound_mm3': len(contacts - removed) * volume,
        'total_reward': reward, 'complete_tool_path_length_mm': path_length,
        'tool_changes': tool_changes, 'nonstop_actions': len(history) - len(stops),
        'positive_target_source_cells_removed': target_cells,
        'total_reference_target_mm3': target_total,
        'reference_target_fraction_removed': target_volume / target_total if target_total > 0 else None,
        'partial_contact_reward_weight': 0., 'motor_surrogate': None, 'language_surrogate': None,
        'clinical_deficit_probability': None,
    }
    for name in ('target_removed_mm3', 'normal_removed_mm3', 'simulated_removed_volume_mm3',
                 'cumulative_contacted_tissue_upper_bound_mm3',
                 'currently_retained_contacted_tissue_upper_bound_mm3', 'total_reward'):
        _same_number(recorded.get(name), outcomes[name], name)
    _same_number(audit.contained_source_tissue_volume_mm3, len(removed) * volume, 'independent removed-cell union')
    if semantic_digest(task.metrics()['history']) != result['committed_history_hash']:
        raise RuntimeError('Committed history changed during independent evaluation')
    return {**result, 'accepted': True, 'target_access_success': target_cells >= minimum_target_cells,
            'outcomes': outcomes, 'evaluation_seconds': time.perf_counter() - started}
