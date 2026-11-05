"""Pure exact-own-prefix validator; no bridge, replay, or physical actions."""
def validate_prefix(rows, literal_actions, *, expected_count, initial_state_id):
    """Only durable complete native commits may form a replay restoration log."""
    if type(expected_count) is not int or expected_count <= 0 or len(rows) != expected_count:
        raise ValueError('Exact audited committed prefix required')
    prior = initial_state_id
    for index, row in enumerate(rows):
        actual = row.get('actual', {}); view = actual.get('view', {}); outcome = actual.get('outcome', {})
        preview = row.get('preview', {})
        if (row.get('literal_index') != index or type(row.get('literal_index')) is not int
                or row.get('cleanup') is not False or row.get('committed') is not True
                or row.get('step_attempted') is not True or row.get('task_predicate_passed') is not True
                or row.get('task_reason') is not None or row.get('action') != literal_actions[index]
                or row.get('state_before') != prior
                or row.get('state_after') != view.get('stateId')
                or row.get('state_after') != preview.get('stateId')
                or outcome.get('status') not in ('accepted', 'partial')
                or outcome.get('reason') not in (None, 'BONE_CUTTING_IN_PROGRESS', 'TISSUE_ASPIRATION_IN_PROGRESS')
                or outcome.get('deltas') != preview.get('deltas')
                or row.get('actual_metrics_after') != view.get('metrics')
                or row.get('actual_slots_after') != view.get('slots')):
            raise ValueError('Journal prefix contains missing/uncertain/nonliteral commit')
        prior = row['state_after']
    return {'state_id': prior, 'committed_actions': [row['action'] for row in rows],
            'declared_seconds': sum(row['action'].get('duration', 1.) for row in rows),
            'removed_source_ids': sorted({cell for row in rows for cell in row['actual']['outcome']['deltas'].get('removedCells', [])})}
