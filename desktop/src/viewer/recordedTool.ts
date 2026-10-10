import type { ViewerVolume, ViewerReplay } from "./contracts.ts";
import type { InstrumentCapsuleDisplay } from "./inspectionTool.ts";
import type { Bounds3 } from "./coordinates.ts";
export interface RecordedEpisodeBounds {
  caseHash: string;
  episodeId: string;
  boundsRasMm: Bounds3;
}
export interface RecordedToolPose {
  scope: "executed-generated-episode";
  caseHash: string;
  episodeId: string;
  frameIndex: number;
  stateId: string;
  toolId: string;
  mode: "aspirate" | "probe";
  phase: "insertion" | "withdrawal";
  tipRasMm: [number, number, number];
  axis: [number, number, number];
  workingLengthMm: number;
  tipLengthMm: number;
  shaftRadiusMm: number;
  tipRadiusMm: number;
}
export interface RecordedToolDisplay extends InstrumentCapsuleDisplay { identity: string; }
/** A recorded pose has its own execution scope; it never enters the preview inspector. */
export function recordedToolDisplay(volume: ViewerVolume, replay: ViewerReplay): RecordedToolDisplay | null {
  const pose = replay.recordedTool;
  if (!pose) return null;
  const ok = pose.scope === "executed-generated-episode" && pose.caseHash === volume.caseHash &&
    pose.frameIndex === replay.step && /^sha256:[a-f0-9]{64}$/.test(pose.episodeId) &&
    /^sha256:[a-f0-9]{64}$/.test(pose.stateId) && typeof pose.toolId === "string" && pose.toolId.length > 0 &&
    ["aspirate", "probe"].includes(pose.mode) && ["insertion", "withdrawal"].includes(pose.phase) &&
    [pose.tipRasMm, pose.axis].every(p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite)) &&
    Math.abs(Math.hypot(...pose.axis) - 1) < 1e-10 &&
    [pose.workingLengthMm, pose.tipLengthMm, pose.shaftRadiusMm, pose.tipRadiusMm].every(n => Number.isFinite(n) && n > 0) &&
    pose.tipLengthMm < pose.workingLengthMm;
  if (!ok) throw new Error("Recorded tool withheld: pose does not match this checked replay frame.");
  const shaftStart = pose.tipRasMm.map((n,i) => n - pose.axis[i] * pose.workingLengthMm);
  const shaftEnd = pose.tipRasMm.map((n,i) => n - pose.axis[i] * pose.tipLengthMm);
  return {identity: `${pose.episodeId}:${pose.frameIndex}:${pose.stateId}`, shaftStart, shaftEnd,
    tip: [...pose.tipRasMm], shaftRadius: pose.shaftRadiusMm, tipRadius: pose.tipRadiusMm,
    color: pose.mode === "probe" ? "#68c7ed" : "#efb566"};
}

/** Display-only envelope of every accepted pose, including shaft/tip radii.
 * It is derived after full replay validation, never supplied by the backend. */
export function recordedEpisodeBounds(volume: ViewerVolume, frames: readonly ViewerReplay[], episodeId: string): RecordedEpisodeBounds | null {
  const bounds: Bounds3 = [[Infinity, Infinity, Infinity], [-Infinity, -Infinity, -Infinity]];
  let any = false;
  for (const frame of frames) {
    const display = recordedToolDisplay(volume, frame);
    if (!display) continue;
    if (frame.recordedTool!.episodeId !== episodeId) throw new Error("Recorded episode camera bounds mix episodes.");
    for (const [point, radius] of [
      [display.shaftStart, display.shaftRadius], [display.shaftEnd, display.shaftRadius],
      [display.shaftEnd, display.tipRadius], [display.tip, display.tipRadius],
    ] as const) for (let axis = 0; axis < 3; axis++) {
      bounds[0][axis] = Math.min(bounds[0][axis], point[axis] - radius);
      bounds[1][axis] = Math.max(bounds[1][axis], point[axis] + radius);
    }
    any = true;
  }
  return any ? {caseHash: volume.caseHash, episodeId, boundsRasMm: bounds} : null;
}

export function checkedRecordedEpisodeBounds(volume: ViewerVolume, replay: ViewerReplay): RecordedEpisodeBounds | null {
  const input = replay.recordedEpisodeBounds;
  if (!input) return null;
  const bounds = input.boundsRasMm;
  if (input.caseHash !== volume.caseHash || !/^sha256:[a-f0-9]{64}$/.test(input.episodeId) ||
      (replay.recordedTool && replay.recordedTool.episodeId !== input.episodeId) ||
      !Array.isArray(bounds) || bounds.length !== 2 ||
      !bounds.every(p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite)) ||
      bounds[0].some((n, axis) => n > bounds[1][axis]))
    throw new Error("Recorded episode camera bounds do not match this replay.");
  return {caseHash: input.caseHash, episodeId: input.episodeId, boundsRasMm: [[...bounds[0]], [...bounds[1]]]};
}
