import type { ViewerVolume, ViewerReplay } from "./contracts.ts";
import type { InstrumentCapsuleDisplay } from "./inspectionTool.ts";
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
