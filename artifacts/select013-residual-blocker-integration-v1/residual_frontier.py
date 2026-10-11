"""Prospective catalog on a fixed authenticated native state; no task mutation."""
from dataclasses import asdict, replace

import numpy as np

from resectionlab.core import semantic_digest, thaw_json
from resectionlab.native_proposals import (PreparedNominalCavityProposer,
    RESIDUAL_OBSTRUCTION_RULE, OBSTRUCTION_CELLS_PER_PREVIEW)


def physical(ray):
    return (ray.family, ray.tool_id, tuple(ray.voxel), ray.entry_mm, ray.tip_mm)


def state(engine):
    return {'source_hash': engine.config.source_hash, 'model_hash': engine.config.fingerprint,
        'state_hash': engine.state_hash, 'revision': engine.revision,
        'masks': dict(engine.committed_mask_digests()),
        'history_hash': semantic_digest(engine.history), 'history_length': len(engine.history)}


def preview_record(ray, result):
    sidecar = result.obstruction_diagnostic
    return {'proposal': asdict(ray), 'feasible': result.feasible, 'reason': result.reason,
        'sidecar': None if sidecar is None else thaw_json(sidecar),
        'history_if_feasible': result.to_history_record() if result.feasible else None}


def diagnose(task, *, write, guard=lambda: None, max_branches=3):
    """Verify every offered extra, then <=3 independent one-prep branches.

    These native clones have no policy/task episode authority. The old task
    remains untouched; no action is inserted into its authenticated inventory.
    Public nominal score estimates come only from its nominal planning model.
    """
    if (not task._planning or task.terminated or type(max_branches) is not int
            or not 0 <= max_branches <= 3
            or not np.array_equal(task.case.reference_target, task.case.nominal_target)
            or task.case.proposal_config.obstruction_page_size != 8
            or task.case.proposal_config.obstruction_selection_rule is not None):
        raise ValueError('Unchanged old-rule nominal planning frontier required')
    before = state(task._engine)
    engine = task._engine.clone()
    if engine.config is not task._engine.config or state(engine) != before:
        raise ValueError('Cloned native source/state differs')
    old = task.case._nominal_proposer
    config = replace(task.case.proposal_config, obstruction_selection_rule=RESIDUAL_OBSTRUCTION_RULE)
    provider = PreparedNominalCavityProposer(engine.config, old._nominal,
        nominal_provenance=old._provenance, config=config, index_affine=old._index_affine,
        index_frame_record=old._frame_record, target_semantics=task.case.target_semantics)
    original = old.propose(engine)
    base = provider.propose(engine)
    if [physical(r) for r in original.proposals] != [physical(r) for r in base.proposals]:
        raise ValueError('Prospective rule changed original physical base rays')
    if old.model_hash == provider.model_hash or len(base.proposals) > 120:
        raise ValueError('Declared rule identity or base cap differs')
    evidence = []; original_records = []; blockers = set()
    for i, ray in enumerate(base.proposals):
        guard()
        preview = engine.preview_stroke(ray.tool_id, ray.tip_mm, entry_mm=ray.entry_mm,
            obstruction_diagnostics=True, obstruction_cell_limit=OBSTRUCTION_CELLS_PER_PREVIEW)
        evidence.append(preview.obstruction_diagnostic)
        record = preview_record(ray, preview); original_records.append(record)
        write('base-ray-%03d.json' % i, record)
        if record['sidecar'] is not None:
            blockers.update(tuple(c) for c in record['sidecar']['blocked_indices_native'])
    extended = provider.append_obstruction_openings(base, engine, evidence)
    extras = extended.proposals[len(base.proposals):]
    if len(extras) > 32 or extended.proposals[:len(base.proposals)] != base.proposals:
        raise ValueError('Bounded preserved prefix violated')
    write('prospective-catalog.json', {'source_state': before,
        'old_proposal_model_hash': old.model_hash, 'prospective_proposal_model_hash': provider.model_hash,
        'old_rule': task.case.proposal_config.to_record(), 'prospective_rule': config.to_record(),
        'old_task_authority_unchanged': True, 'new_rule_generated_prefix': False,
        'native_source_and_model_unchanged': True, 'base_physical_prefix_exact': True,
        'base_count': len(base.proposals), 'extra_count': len(extras),
        'ledger': [asdict(r) for r in extended.ledger],
        'accounting': thaw_json(extended.obstruction_accounting)})
    examined = []; eligible = []
    for ordinal, ray in enumerate(extras):
        guard()
        result = engine.preview_stroke(ray.tool_id, ray.tip_mm, entry_mm=ray.entry_mm,
            obstruction_diagnostics=True, obstruction_cell_limit=OBSTRUCTION_CELLS_PER_PREVIEW)
        record = preview_record(ray, result)
        removed = set(map(tuple, result.removed_indices_native.tolist())) if result.feasible else set()
        overlap = sorted(removed & blockers)
        record.update(ordinal=ordinal, sampled_first_blockers_removed=overlap,
            selection_score=len(overlap),
            nominal_increment=None if not result.feasible else task._score_record(result.to_history_record(), task._current_tool))
        write('extra-%02d.json' % ordinal, record); examined.append(record)
        if result.feasible and overlap:
            eligible.append((-len(overlap), ordinal, ray, record))
    eligible.sort(key=lambda row: (row[0], row[1]))
    chosen = eligible[:max_branches]
    write('branch-selection.json', {'rule': 'most_current_sampled_first_blockers_actually_removed_then_emitted_order',
        'feasible_positive_overlap_candidates': len(eligible), 'branch_cap': max_branches,
        'chosen_ordinals': [r[1] for r in chosen], 'unchosen_ordinals': [r[1] for r in eligible[max_branches:]],
        'zero_overlap_candidates_not_tried': sum(r['feasible'] and not r['sampled_first_blockers_removed'] for r in examined),
        'scope': 'bounded one-prep diagnostic; zero-overlap preparations may have other effects; no completeness claim'})
    branches = []; audit_histories = []
    for branch_index, (_, ordinal, ray, examined_record) in enumerate(chosen):
        guard(); branch = engine.clone(); replay = engine.clone()
        if state(branch) != before or state(replay) != before:
            raise ValueError('Independent prep does not begin at the exact frontier')
        prep = branch.preview_stroke(ray.tool_id, ray.tip_mm, entry_mm=ray.entry_mm)
        if not prep.feasible or semantic_digest(prep.to_history_record()) != semantic_digest(examined_record['history_if_feasible']):
            raise ValueError('Fresh preparation certificate differs')
        branch.commit_preview(prep)
        repeated = replay.preview_stroke(ray.tool_id, ray.tip_mm, entry_mm=ray.entry_mm)
        replay.commit_preview(repeated)
        if state(branch) != state(replay) or semantic_digest(branch.history) != semantic_digest(replay.history):
            raise ValueError('Independent one-prep native replay differs')
        prep_score = task._score_record(prep.to_history_record(), task._current_tool)
        changes = []; newly_feasible = []
        for i, original_ray in enumerate(base.proposals):
            guard()
            result = branch.preview_stroke(original_ray.tool_id, original_ray.tip_mm, entry_mm=original_ray.entry_mm,
                obstruction_diagnostics=True, obstruction_cell_limit=OBSTRUCTION_CELLS_PER_PREVIEW)
            record = preview_record(original_ray, result)
            previous = original_records[i]
            record.update(before_feasible=previous['feasible'], before_reason=previous['reason'],
                newly_feasible=not previous['feasible'] and result.feasible,
                nominal_increment=None if not result.feasible else task._score_record(result.to_history_record(), ray.tool_id))
            write('branch-%02d/ray-%03d.json' % (branch_index, i), record)
            if previous['reason'] != result.reason:
                changes.append({'ray': i, 'before': previous['reason'], 'after': result.reason})
            if record['newly_feasible']:
                newly_feasible.append({'ray': i, 'proposal': asdict(original_ray),
                    'nominal_increment': record['nominal_increment']['reward'],
                    'prospective_prefix_prep_ray_return': task._total_reward + prep_score['reward'] + record['nominal_increment']['reward'],
                    'executed': False})
        row = {'index': branch_index, 'extra_ordinal': ordinal, 'prep': prep.to_history_record(),
            'nominal_prep': prep_score, 'prefix_return': task._total_reward,
            'prefix_plus_prep_STOP_return': task._total_reward + prep_score['reward'], 'root_STOP_return': 0.,
            'first_blockers_removed': examined_record['sampled_first_blockers_removed'],
            'remaining_sampled_first_blockers': sorted(blockers-set(map(tuple, prep.removed_indices_native.tolist()))),
            'newly_feasible_original_rays': newly_feasible, 'reason_changes': changes,
            'full_native_history': branch.history, 'native_state': state(branch),
            'deterministic_native_commit_replay_exact': True,
            'independent_geometry_certificate': 'written separately by the existing independent history checker',
            'accepted_plan_claim': False, 'complete_new_rule_task_episode': False}
        write('branch-%02d/result.json' % branch_index, row); branches.append(row)
        audit_histories.append(branch.history)
        del branch, replay
    if state(engine) != before or state(task._engine) != before:
        raise ValueError('Prospective diagnostics mutated the original frontier')
    return {'base_count': len(base.proposals), 'extra_count': len(extras),
        'sampled_first_blockers': sorted(blockers), 'feasible_extras': sum(r['feasible'] for r in examined),
        'branch_count': len(branches), 'branch_results': [{k: v for k, v in row.items()
            if k not in ('full_native_history', 'prep', 'nominal_prep')} for row in branches],
        'original_task_state_unchanged': True, 'native_source_unchanged': True,
        'prospective_rule_hash': config.fingerprint, 'prospective_model_hash': provider.model_hash,
        'omitted_pairs': [asdict(r) for r in extended.ledger[len(base.ledger):]
            if r.reason in ('CANDIDATE_CAP', 'OBSTRUCTION_SELECTION_CAP')],
        'all_extra_pairs_verified': True, 'accepted_plan_claim': False,
        '_audit_histories': audit_histories}
