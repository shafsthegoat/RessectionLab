import type {NativeEpisodeGeometry} from './native-episode-replay.ts';
import type {PublicContactEpisode} from './public-contact-types.ts';
import type {ViewerReplay} from './viewer/contracts.ts';
import {contactNeed, contactSame, CONTACT_COSTS} from './public-contact-authority.ts';

export interface ContactPrefix {
  retained: boolean;
  contactedAndRetained: boolean;
  removedVolumeMm3: number;
}
export type NativeContactEpisode = Omit<NativeEpisodeGeometry, 'history'> & {
  publicGoal: {nativeIndex: [number, number, number]; objectiveHash: string};
  history: PublicContactEpisode['history'];
  metrics: Record<string, unknown>;
};
const close = (a: unknown, b: number) => typeof a === 'number' && Number.isFinite(a) &&
  Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));

/** Called only after task-specific admission and exact shared native replay.
 * All-removal/contact accounting is identical for fixed v2 and family v3. */
export function checkedNativeContactOutcome(
  episode: NativeContactEpisode, frames: ViewerReplay[], voxelVolumeMm3: number,
) {
  const goal = episode.publicGoal.nativeIndex;
  const index = (goal[0] * episode.shape[1] + goal[1]) * episode.shape[2] + goal[2];
  const prefixes: ContactPrefix[] = [];
  let touched = false;
  for (const [i, frame] of episode.replayFrames.entries()) {
    if (frame.probeContactIndicesNative.some(cell => contactSame(cell, goal))) touched = true;
    const retained = frames[i].removedMask[index] === 0;
    prefixes.push({retained, contactedAndRetained: retained && touched,
      removedVolumeMm3: frames[i].removedTargetVolumeMm3 + frames[i].removedNormalVolumeMm3});
  }
  let before = 0, contacted = false, retained = true, removed = 0, reward = 0, path = 0, cost = 0;
  let priorTool: string | undefined;
  for (const [actionIndex, action] of episode.history.entries()) {
    const key = (cell: number[]) => cell.join(',');
    const committed = [...new Set(episode.replayFrames
      .filter(frame => frame.actionIndex === actionIndex && frame.phase === 'insertion')
      .flatMap(frame => frame.probeContactIndicesNative.map(key)))].sort();
    contactNeed(contactSame(action.probe_contact_indices_native.map(key).sort(), committed),
      'macro probe contact differs from retained committed insertion contact.');
    const stop = action.interaction_mode === 'stop';
    const volume = action.removed_indices_native.length * voxelVolumeMm3;
    if (action.removed_indices_native.some(cell => contactSame(cell, goal))) retained = false;
    if (action.interaction_mode === 'probe' && action.probe_contact_indices_native.some(cell => contactSame(cell, goal))) contacted = true;
    const after = Number(retained && contacted);
    const distance = stop ? 0 : Math.hypot(...action.tip_mm!.map((n, i) => n - action.entry_mm![i]));
    const change = Number(!stop && priorTool !== undefined && priorTool !== action.tool_id);
    const charge = CONTACT_COSTS.normal_per_mm3 * volume + CONTACT_COSTS.action_cost * Number(!stop) +
      2 * CONTACT_COSTS.motion_per_mm * distance + CONTACT_COSTS.tool_change_cost * change;
    contactNeed(action.goal_potential_before === before && action.goal_potential_after === after &&
      action.public_objective_hash === episode.publicGoal.objectiveHash &&
      action.clinical_deficit_probability === null && action.outcome_scope === 'public_geometric_retained_surface_contact' &&
      close(action.removal_cost_volume_mm3, volume) && close(action.insertion_distance_mm, distance) &&
      close(action.complete_tool_path_length_mm, 2 * distance) && action.tool_change_count === change &&
      close(action.effort_and_removal_cost, charge) && close(action.reward, after - before - charge),
      'goal potential, effort or reward differs from executed geometry.');
    removed += volume; reward += action.reward; path += 2 * distance; cost += charge; before = after;
    if (!stop) priorTool = action.tool_id;
  }
  const final = prefixes.at(-1)!;
  contactNeed(retained === final.retained && Boolean(before) === final.contactedAndRetained &&
    episode.metrics.goal_retained === final.retained && episode.metrics.goal_contacted_and_retained === final.contactedAndRetained &&
    close(episode.metrics.removed_volume_mm3, removed) && close(episode.metrics.total_reward, reward) &&
    close(final.removedVolumeMm3, removed), 'final contact/removal metrics differ from replay.');
  return {prefixes, outcome: {modeledReturn: reward, totalRemovedMm3: removed,
    completePathMm: path, totalCost: cost, retained: final.retained,
    contactedAndRetained: final.contactedAndRetained,
    termination: episode.history.at(-1)!.interaction_mode === 'stop' ? 'STOP' as const : 'horizon' as const}};
}
