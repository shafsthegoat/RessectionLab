"""Existing pilot observation-only local greedy policy, without a learning runtime.

No scene, preview, reward or info is accepted. V2 observations declare their
sampling spacing; legacy observations retain the original 1 mm convention.
"""
import numpy as np

LOCAL_SPACING_MM = 1.
STEP_MM = 2.
ACTION_SECONDS = .2
ACTION_IDS = ('x-', 'x+', 'y-', 'y+', 'z-', 'z+', 'apply_here', 'inspect', 'STOP')
OFFSETS = np.array([[-2, 0, 0], [2, 0, 0], [0, -2, 0], [0, 2, 0],
                    [0, 0, -2], [0, 0, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]], float)

def candidates(obs):
    """Candidate actions and scores depend ONLY on the published observation.

    No info, environment, source label maps, occupancy, snapshot, or preview is
    accepted. Samples are an imperfect nominal estimate, not a feasibility oracle.
    """
    local = np.asarray(obs['local'][0], dtype=float)
    spacing_value = np.asarray(obs.get('local_spacing_mm', [LOCAL_SPACING_MM]), dtype=float)
    if spacing_value.shape != (1,) or not np.isfinite(spacing_value).all() or spacing_value[0] <= 0:
        raise ValueError('Observation local_spacing_mm must declare one positive finite millimeter value')
    spacing = float(spacing_value[0])
    points = np.stack(np.meshgrid(*([np.arange(-4, 5) * spacing] * 3),
                                 indexing='ij'), axis=-1).reshape(-1, 3)
    channels = local.reshape(8, -1)
    radius = float(obs['tool_parameters'][0, 0])
    common = channels.mean(axis=1)
    rows, actions = [], []
    for i, delta in enumerate(OFFSETS):
        fraction = np.clip(points @ delta / (delta @ delta), 0, 1) if delta @ delta else np.zeros(len(points))
        distance = np.linalg.norm(points - fraction[:, None] * delta, axis=1)
        selected = distance <= radius + .5 * spacing
        mean = channels[:, selected].mean(axis=1) if selected.any() else np.zeros(8)
        healthy = ((channels[0, selected] > 0) & (channels[1, selected] == 0)).mean() if selected.any() else 0.
        rows.append([*mean, *common, np.linalg.norm(delta) / STEP_MM, float(i == 7), healthy, float(i == 8)])
        action = dict(slot='left', operation='deactivate' if i == 8 else 'inspect' if i == 7 else 'apply',
                      duration=ACTION_SECONDS)
        if i < 7:
            q = np.asarray(obs['slots'][0, 3:7], float); q = q / np.linalg.norm(q)
            action['pose'] = dict(position=(np.asarray(obs['slots'][0, :3], float) + delta).tolist(), quaternion=q.tolist())
        actions.append(action)
    return np.asarray(rows, np.float32), actions


def observed_greedy(features):
    # Existing native observed greedy ranks permitted nominal gain minus burden.
    # Its NativeSpatialTask geometry is incompatible with SurgicalRLEnv. This
    # explicitly labeled local adaptation scores the same nine observed options
    # supplied to the policy, with zero hidden-state transition previews.
    target, protected, healthy = features[:, 1], features[:, 2], features[:, 18]
    scores = target - 2. * healthy - 5. * protected - .01 * features[:, 16] - .0012
    scores[7:] = -.0012
    # Explicit nominal no-positive-progress STOP threshold, not hidden truth.
    selected = 8 if float(scores.max()) <= float(scores[8]) else int(np.argmax(scores))
    return selected, scores.tolist()


