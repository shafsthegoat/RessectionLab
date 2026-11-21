import {Quaternion, Vector3} from 'three';
import type {SurgicalInteraction, SurgicalPose, SurgicalResponse} from './surgical-types';

type Jog = {offset: [number, number, number]} | {pitch: number; yaw: number};

export function practiceJogKey(key: string, tilt = false): Jog | null {
  const horizontal = Number(key === 'ArrowRight') - Number(key === 'ArrowLeft');
  const vertical = Number(key === 'ArrowUp') - Number(key === 'ArrowDown');
  const depth = Number(key === 'PageDown') - Number(key === 'PageUp');
  if (tilt && (vertical || horizontal)) return {pitch: -2 * vertical, yaw: 2 * horizontal};
  return horizontal || vertical || depth ? {offset: [horizontal * .5, vertical * .5, depth * .5]} : null;
}

/** Instrument-local +Z advances the tip. A drag proposes at most 4 mm from
 * its start, independent of event count and the number of display frames. */
export function practiceDrag(dx: number, dy: number, depth: boolean): Jog {
  const delta = new Vector3(dx / 20, -dy / 20, 0).clampLength(0, 4);
  return {offset: depth ? [0, 0, delta.y] : [delta.x, delta.y, 0]};
}

export function jogPracticePose(start: SurgicalPose, jog: Jog): SurgicalPose {
  const orientation = new Quaternion(...start.quaternion).normalize();
  if ('offset' in jog) {
    const delta = new Vector3(...jog.offset).applyQuaternion(orientation);
    return {position: delta.add(new Vector3(...start.position)).toArray(), quaternion: [...start.quaternion]};
  }
  const pitch = new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), jog.pitch * Math.PI / 180);
  const yaw = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), jog.yaw * Math.PI / 180);
  return {position: [...start.position], quaternion: orientation.multiply(yaw).multiply(pitch).normalize().toArray()};
}

/** A numeric edit or jog temporarily reveals the transform handles. Explicit
 * viewport choices survive submission, while an acknowledged pose has no
 * overlapping draft. Quaternion sign does not change physical orientation. */
export function practiceViewportMode(mode: SurgicalInteraction['mode'], pose: SurgicalPose,
  accepted: SurgicalPose | undefined, publishing: boolean): SurgicalInteraction['mode'] {
  if (mode !== 'navigate' || !accepted || publishing) return mode;
  const moved = pose.position.some((value,i) => Math.abs(value - accepted.position[i]) > 1e-5);
  const dot = pose.quaternion.reduce((sum,value,i) => sum + value * accepted.quaternion[i], 0);
  const rotated = 1 - Math.abs(dot) > 1e-9;
  return moved || rotated ? 'translate' : 'navigate';
}

export function practiceStartIssue(result: SurgicalResponse | null, studyRecording: boolean, preview: boolean) {
  if (studyRecording) return 'This study authorizes recorded playback only.';
  if (result?.playback) return 'Return to a recorded action boundary to check practice availability.';
  if (result?.verification?.readOnly) return 'Verify the saved replay before starting practice.';
  if (preview) return 'Cancel the preview to use its accepted checkpoint.';
  if (!result?.practiceAdmission?.available || !result.practiceAdmission.checkpointSeal)
    return result?.practiceAdmission?.reason || 'This checkpoint has no verified practice authorization.';
  return null;
}

export function matchesPracticeParent(parent: SurgicalResponse, child: SurgicalResponse) {
  const episode = child.practiceEpisode;
  return !!episode && child.caseHash === parent.caseHash && child.view.scene.sceneId === parent.view.scene.sceneId &&
    episode.schema === 'resectionlab.surgical-practice.v1' && episode.parent.caseHash === parent.caseHash &&
    episode.parent.sceneId === parent.view.scene.sceneId && episode.parent.checkpointSeal === parent.practiceAdmission?.checkpointSeal &&
    child.position === parent.position && child.branchId === episode.episodeId && child.branchId !== parent.branchId && !child.readOnly && !child.verification?.readOnly &&
    !child.playback && child.view.stateId === parent.view.stateId && episode.parent.stateId === parent.view.stateId &&
    episode.parent.branchId === parent.branchId && episode.parent.position === parent.position;
}

/** Only routes a saved child to Rehearse. The engine replays its checkpoint
 * before the controller or HUD receives any practice authority. */
export function savedPracticeEpisode(artifacts: Record<string,unknown> | undefined): boolean {
  const archive = artifacts?.surgical_toolbox as {
    branchId?: string; practiceEpisodeId?: string; practiceEpisodes?: Record<string,{episodeId?: string}>;
  } | undefined;
  const id = archive?.branchId;
  return !!id && id.startsWith('practice-') && (archive?.practiceEpisodeId === id || archive?.practiceEpisodes?.[id]?.episodeId === id);
}

export function checkPracticeReceipt(previous: SurgicalResponse, next: SurgicalResponse) {
  if (!previous.practiceEpisode) return;
  if (next.caseHash !== previous.caseHash || next.branchId !== previous.branchId ||
    next.practiceEpisode?.episodeId !== previous.practiceEpisode.episodeId ||
    next.practiceEpisode?.parent.checkpointSeal !== previous.practiceEpisode.parent.checkpointSeal ||
    !next.practiceMetrics || next.practiceMetrics.originalTargetVolumeMm3 !== previous.practiceMetrics?.originalTargetVolumeMm3)
    throw Error('The engine receipt differs from the displayed practice episode or source target.');
}

/** Latest accepted exterior placement for this hand and this tool exchange.
 * Cursors can inspect earlier actions, so later placements are excluded. */
export function practiceWithdrawalPose(result: SurgicalResponse, hand: string): SurgicalPose | undefined {
  const timeline = result.timeline ?? result.view.history;
  for (let i=Math.min(result.position,timeline.length)-1; i>=0; i--) {
    const {action,status} = timeline[i];
    if (action.slot !== hand || status === 'rejected') continue;
    if (action.operation === 'exchange') return;
    if (action.operation === 'place' && status === 'accepted') {
      const placed = action.pose ?? (action.path?.length === 1 ? action.path[0] : undefined);
      if (placed?.position?.length === 3 && placed.quaternion?.length === 4 &&
        [...placed.position,...placed.quaternion].every(Number.isFinite)) return placed;
      return;
    }
  }
}
