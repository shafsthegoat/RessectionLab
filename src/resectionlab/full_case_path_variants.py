"""New CUSA path proposals; each variant requires its own public preview.

Coalescing changes engine sampling and dt, so effects and partial outcomes may
differ. It never rewrites an action log or claims equivalence to a fine action.
"""
from __future__ import annotations

import copy


def cusa_path_variants(action):
    """Try exact axial coalescing, then retain the untouched fine fallback.

    Only strictly monotone axis-aligned runs with unchanged quaternion qualify.
    First/last poses, corners, reversals and duplicate poses remain. Changing
    quaternion paths and diagonal runs retain their original representation.
    Duration and parameters remain those of the original proposed action.
    """
    path = action.get('path', [])
    if action.get('operation') != 'apply' or len(path) < 3:
        return [action]
    if any(pose['quaternion'] != path[0]['quaternion'] for pose in path):
        return [action]

    def redundant(a, b, c):
        ab = [i for i in range(3) if a['position'][i] != b['position'][i]]
        bc = [i for i in range(3) if b['position'][i] != c['position'][i]]
        if len(ab) != 1 or ab != bc:
            return False
        axis = ab[0]
        return ((b['position'][axis] > a['position'][axis]) ==
                (c['position'][axis] > b['position'][axis]))

    kept = []
    for pose in path:
        while len(kept) >= 2 and redundant(kept[-2], kept[-1], pose):
            kept.pop()
        kept.append(pose)
    if len(kept) == len(path):
        return [action]
    coarse = copy.deepcopy(action)
    coarse['path'] = copy.deepcopy(kept)
    return [coarse, action]
