"""Independent scalar audit of the declared deployment comparison; unrun draft.

Arguments are normalized saved report columns, already authenticated by the
caller. This code imports no policy, task, native geometry, numpy, or torch.
It must not be used as a live selector or to create a training capability.
"""
import math


def audit_choice(action_ids, legal_mask, actual_rewards, q_predictions, chosen,
                 *, method):
    """Verify stable-order actual-positive greedy or guarded-Q selection."""
    assert method in ('greedy', 'guarded_Q')
    assert action_ids and action_ids[0] == 'STOP'
    assert len(set(action_ids)) == len(action_ids)
    assert len(action_ids) == len(legal_mask) == len(actual_rewards)
    assert all(type(value) is bool for value in legal_mask) and legal_mask[0]
    assert actual_rewards[0] == 0
    positive = []
    for i, (legal, reward) in enumerate(zip(legal_mask, actual_rewards)):
        if legal:
            assert type(reward) in (int, float) and math.isfinite(reward)
            if i and reward > 0:
                positive.append(i)
        else:
            assert reward is None
    # max preserves the first occurrence, unlike sorting by opaque action ID.
    expected = max(positive, key=lambda i: actual_rewards[i]) if positive else 0
    reason = 'greedy_actual_positive' if positive else 'no_actual_positive_STOP'
    if method == 'guarded_Q' and q_predictions is not None:
        assert len(q_predictions) == len(action_ids)
        assert q_predictions[0] == 0
        for legal, value in zip(legal_mask, q_predictions):
            assert ((legal and type(value) in (int, float) and math.isfinite(value))
                    or (not legal and value is None))
        if positive:
            predicted_positive = [i for i in positive if q_predictions[i] > 0]
            if predicted_positive:
                expected = max(predicted_positive, key=lambda i: q_predictions[i])
                reason = 'positive_Q_among_actual_positive'
            else:
                reason = 'no_positive_Q_greedy_fallback'
    elif method == 'guarded_Q':
        # A forward can be omitted only when its ranking cannot affect STOP.
        assert not positive
    assert chosen == action_ids[expected], 'Saved choice violates declared guard/ranking/tie rule'
    return {'chosen': chosen, 'reason': reason, 'actual_positive_count': len(positive),
            'actual_selected_reward': actual_rewards[expected]}


def equal_subject_trace_state_weights(trace_lengths, subjects):
    """Unrounded reference weights; STOP states retain denominator mass."""
    assert len(trace_lengths) == len(subjects) == 36
    assert set(subjects) == {'ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045'}
    assert all(subjects.count(s) == 9 for s in set(subjects))
    assert sum(trace_lengths) == 497
    assert all(type(n) is int and 1 <= n <= 24 for n in trace_lengths)
    return [[1. / (4 * 9 * n)] * n for n in trace_lengths]
