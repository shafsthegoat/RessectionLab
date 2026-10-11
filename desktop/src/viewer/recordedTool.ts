import type { ViewerVolume, ViewerReplay, ViewerPassiveReplay } from "./contracts.ts";
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

/** A pose from generated passive geometry, not a legacy executed stroke. */
export interface PassiveRecordedToolPose {
  scope: "generated-persistent-passive-episode";
  caseHash: string;
  episodeId: string;
  frameIndex: number;
  toolId: string;
  mode: "probe" | "aspirate";
  tipRasMm: [number, number, number];
  axis: [number, number, number];
  workingLengthMm: number;
  tipLengthMm: number;
  shaftRadiusMm: number;
  tipRadiusMm: number;
  retainedAtDepth: boolean;
  physicalAbsenceClaim: false;
  poseStatus: "held_in_declared_void" | "access_pose_recorded";
  activation: "passive_only" | "off";
}

/** Local display guard; the host has already verified the full fresh replay.
 * This checks binding, source grid, every visible capsule and the distinction
 * between primary pair clearance and independent tissue-only motion. */
export function checkedPassiveRecordedTools(volume: ViewerVolume, replay: ViewerPassiveReplay):
    { displays: RecordedToolDisplay[]; bounds: RecordedEpisodeBounds } {
  const hash = (value: unknown) => typeof value === "string" && /^sha256:[a-f0-9]{64}$/.test(value);
  const phases = ["ready", "held", "inspected", "withdrawn", "both_withdrawn", "stopped"];
  const expectedEvent = replay.frameIndex === 2 ? "generated_inspection" :
    replay.frameIndex === replay.stepCount ? "stop" : null;
  if (replay.scope !== "generated-persistent-passive-episode" || replay.caseHash !== volume.caseHash ||
      !hash(replay.episodeId) || replay.primaryPairClearanceChecked !== true ||
      replay.independentTissueMotionChecked !== true ||
      !Array.isArray(replay.recordedTools) || replay.recordedTools.length !== 2 ||
      !Number.isInteger(replay.frameIndex) || replay.frameIndex < 0 ||
      replay.stepCount !== phases.length - 1 || replay.frameIndex > replay.stepCount ||
      replay.phase !== phases[replay.frameIndex] ||
      !Number.isFinite(replay.elapsedGeneratedSeconds) || replay.elapsedGeneratedSeconds < 0 ||
      replay.elapsedGeneratedSeconds > 60 ||
      !["not_acquired", "verified", "unresolved"].includes(replay.inspectionStatus) ||
      (replay.frameIndex < 2 && replay.inspectionStatus !== "not_acquired") ||
      (replay.frameIndex >= 2 && replay.inspectionStatus === "not_acquired") ||
      (expectedEvent === null ? replay.event !== null : replay.event?.kind !== expectedEvent) ||
      (expectedEvent === "generated_inspection" && replay.event?.status !== replay.inspectionStatus) ||
      (expectedEvent === "stop" && typeof replay.event?.reason !== "string") ||
      !Array.isArray(replay.shape) || replay.shape.length !== 3 ||
      replay.shape.some((n, i) => n !== volume.shape[i]) ||
      !["RAS+", "RAS", "LPS+", "LPS"].includes(volume.frame))
    throw new Error("Generated passive replay withheld: episode, source or clearance proof is invalid.");
  const ras = volume.affine.map((row, axis) => row.map(value =>
    volume.frame.startsWith("LPS") && axis < 2 ? -value : value));
  if (!Array.isArray(replay.affine) || replay.affine.length !== 4 ||
      replay.affine.some((row, axis) => !Array.isArray(row) || row.length !== 4 ||
        row.some((value, column) => !Number.isFinite(value) ||
          Math.abs(value - ras[axis][column]) > 1e-7)))
    throw new Error("Generated passive replay withheld: source RAS grid differs.");
  const displays = replay.recordedTools.map((pose, index) => {
    const held = index === 0 ? replay.frameIndex === 1 || replay.frameIndex === 2 : replay.frameIndex < 4;
    const valid = pose.scope === replay.scope && pose.caseHash === replay.caseHash &&
      pose.episodeId === replay.episodeId && pose.frameIndex === replay.frameIndex &&
      typeof pose.toolId === "string" && pose.toolId.length > 0 &&
      pose.mode === (index === 0 ? "probe" : "aspirate") &&
      [pose.tipRasMm, pose.axis].every(point => Array.isArray(point) &&
        point.length === 3 && point.every(Number.isFinite)) &&
      Math.abs(Math.hypot(...pose.axis) - 1) < 1e-10 &&
      [pose.workingLengthMm, pose.tipLengthMm, pose.shaftRadiusMm, pose.tipRadiusMm]
        .every(n => Number.isFinite(n) && n > 0) &&
      pose.tipLengthMm < pose.workingLengthMm &&
      pose.retainedAtDepth === held && pose.physicalAbsenceClaim === false &&
      pose.poseStatus === (held ? "held_in_declared_void" : "access_pose_recorded") &&
      pose.activation === (index === 0 && held ? "passive_only" : "off");
    if (!valid) throw new Error("Generated passive replay withheld: tool pose is invalid.");
    const shaftStart = pose.tipRasMm.map((n, i) => n - pose.axis[i] * pose.workingLengthMm);
    const shaftEnd = pose.tipRasMm.map((n, i) => n - pose.axis[i] * pose.tipLengthMm);
    return { identity: `${pose.episodeId}:${pose.frameIndex}:${pose.toolId}`,
      shaftStart, shaftEnd, tip: [...pose.tipRasMm], shaftRadius: pose.shaftRadiusMm,
      tipRadius: pose.tipRadiusMm, color: index === 0 ? "#68c7ed" : "#efb566" };
  });
  if (replay.recordedTools[0].toolId === replay.recordedTools[1].toolId)
    throw new Error("Generated passive replay withheld: tool identities coincide.");
  const bounds = replay.recordedEpisodeBounds;
  const low = bounds?.boundsRasMm?.[0], high = bounds?.boundsRasMm?.[1];
  if (bounds?.caseHash !== volume.caseHash || bounds?.episodeId !== replay.episodeId ||
      !Array.isArray(low) || !Array.isArray(high) || low.length !== 3 || high.length !== 3 ||
      [...low, ...high].some(n => !Number.isFinite(n)) || low.some((n, i) => n > high[i]))
    throw new Error("Generated passive replay withheld: derived camera bounds are invalid.");
  for (const display of displays) for (const [point, radius] of [
    [display.shaftStart, display.shaftRadius], [display.shaftEnd, display.shaftRadius],
    [display.shaftEnd, display.tipRadius], [display.tip, display.tipRadius],
  ] as const) for (let axis = 0; axis < 3; axis++)
    if (point[axis] - radius < low[axis] - 1e-7 || point[axis] + radius > high[axis] + 1e-7)
      throw new Error("Generated passive replay withheld: derived bounds omit a tool capsule.");
  return { displays, bounds: {caseHash: bounds.caseHash, episodeId: bounds.episodeId,
    boundsRasMm: [[...low], [...high]]} };
}
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
