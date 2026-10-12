"""Independent scalar oracles for generated focused-policy controls.

No task construction, tensor, patient payload, checkpoint or fitted parameters.
Actual context/native fixture tests will call these after the model API freezes.
"""
import math


def choose_positive(action_ids, rewards, raw_tails=None):
    """Exhaustively order eligible rows; actual reward enters exactly once."""
    if (not action_ids or action_ids[0] != 'STOP' or len(set(action_ids)) != len(action_ids)
            or len(rewards) != len(action_ids) or rewards[0] != 0.
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in rewards)):
        raise ValueError('Unique aligned finite candidates with STOP0 required')
    eligible = [i for i in range(1, len(rewards)) if rewards[i] > 0.]
    if not eligible:
        return 'STOP'
    if raw_tails is None:
        return action_ids[sorted(eligible, key=lambda i: (-rewards[i], i))[0]]
    if (len(raw_tails) != len(action_ids) or raw_tails[0] != 0.
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in raw_tails)):
        raise ValueError('Aligned finite raw tails with STOP0 required')
    scores = [rewards[i] + (raw_tails[i] if raw_tails[i] > 0. else 0.) for i in eligible]
    if any(not math.isfinite(v) for v in scores):
        raise ValueError('Combined score overflow')
    best = sorted(zip(eligible, scores), key=lambda pair: (-pair[1], pair[0]))[0][0]
    return action_ids[best]


def union_rows(pages):
    """First-appearance order; repeated base IDs must have identical geometry.

Pages supply public rows only: action_id, tool_id, entry_mm and tip_mm.
This oracle never authorizes a row or produces a certificate/action packet.
"""
    rows = []; owners = {}; seen = {}
    for page_index, page in enumerate(pages):
        for row in page:
            identifier = row['action_id']
            physical = (row['tool_id'], tuple(row['entry_mm']), tuple(row['tip_mm']))
            if identifier in seen:
                if seen[identifier] != physical:
                    raise ValueError('Same current action ID maps to different physical rays')
                continue
            seen[identifier] = physical; owners[identifier] = page_index
            rows.append(row)
    return tuple(rows), owners
