"""Independent pure scalar arithmetic; no task, producer, tensor or array imports."""
import math


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Finite scalar required')
    return value


def best_fixed_prefix(rewards, *, prefix_length, horizon):
    """Direct prefix sums oracle, retaining prefix and earliest strict improvement.

    Input contains actual exploration rewards, including any final STOP0. This
    function cannot establish geometry or completeness; those are separate joins.
    """
    if type(prefix_length) is not int or type(horizon) is not int or not 0 < prefix_length < horizon:
        raise ValueError('Absolute horizon with nonempty retained prefix required')
    if not prefix_length <= len(rewards) <= horizon:
        raise ValueError('Exploration length outside fixed full-route horizon')
    for value in rewards:
        _finite(value)
    # Direct independent prefix sums rather than producer's state accumulator.
    totals=[math.fsum(rewards[:n]) for n in range(prefix_length,len(rewards)+1)]
    best=max(range(len(totals)),key=lambda i:totals[i])
    return {'prefix_return':totals[0], 'best_full_return':totals[best],
            'best_absolute_prefix_length':prefix_length+best,
            'best_net_suffix_return':totals[best]-totals[0]}


def useful_gate(baseline, candidate, *, admissible):
    """No outside/contact nonregression silently added to the declared gate.

    Physical/reference denominator and completeness must be checked separately.
    All outcome deltas are returned, including unfavorable preparation costs.
    """
    for result in (baseline,candidate):
        for key in ('total_reward','target_removed_mm3','normal_removed_mm3',
                    'complete_tool_path_length_mm','cumulative_contacted_tissue_upper_bound_mm3'):
            _finite(result[key])
        for key in ('positive_target_source_cells_removed','nonstop_actions','tool_changes'):
            if type(result[key]) is not int or result[key] < 0:
                raise ValueError('Exact nonnegative integer outcome required')
    differences={key:candidate[key]-baseline[key] for key in (
        'total_reward','target_removed_mm3','positive_target_source_cells_removed',
        'normal_removed_mm3','complete_tool_path_length_mm','nonstop_actions',
        'tool_changes','cumulative_contacted_tissue_upper_bound_mm3')}
    additional=differences['positive_target_source_cells_removed']>0
    positive=differences['total_reward']>0
    return {'comparison_admissible':admissible is True,
            'additional_target_cells':additional,'positive_net_suffix_return':positive,
            'useful_gain':admissible is True and additional and positive,
            'tie_is_negative':True,'candidate_minus_original_STOP':differences}
