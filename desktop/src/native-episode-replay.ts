import type {CasePayload,ResectionApi} from './types.ts';
import type {DevelopmentEpisode,NativeCell} from './episode-types.ts';
import type {ViewerReplay} from './viewer/contracts.ts';
import {recordedEpisodeBounds} from './viewer/recordedTool.ts';
import {hydrateCase,validateCaseDescriptor} from './case-data.ts';
import {arrayDigest,sha256Bytes,sourceImageDigest,sourceFrameDigest} from './source-integrity.ts';
export type NativeEpisodeGeometry=Omit<DevelopmentEpisode,'schema'|'selector'|'history'|'metrics'|'planning'> & {schema:string;history:Array<Omit<DevelopmentEpisode['history'][number],'reward'|'target_removed_mm3'|'normal_removed_mm3'>>};
const HASH=/^sha256:[a-f0-9]{64}$/;
const same=(a:unknown,b:unknown)=>JSON.stringify(a)===JSON.stringify(b);
function requireValue(value:unknown,message:string):asserts value{if(!value)throw new Error(`Episode replay withheld: ${message}`)}
const count=(mask:Uint8Array)=>mask.reduce((sum,n)=>sum+n,0);
/** Shared physical replay only. Each task-specific caller must first admit its
 * exact schema, objective and execution provenance. This does not infer them. */
export async function hydrateNativeEpisodeReplay<T extends NativeEpisodeGeometry>(input:{case:CasePayload;episode:T;episodeCanonicalJson:string},api:ResectionApi,maxSteps:2|6,admittedShape:readonly [number,number,number]=[13,13,12]){
  const source = structuredClone(input.case), episode = structuredClone(input.episode);
  requireValue(typeof input.episodeCanonicalJson === "string" && input.episodeCanonicalJson.length <= 2*1024*1024 &&
    `sha256:${await sha256Bytes(new TextEncoder().encode(input.episodeCanonicalJson))}` === episode.episodeId, "episode serialization digest changed.");
  const canonicalEpisode = JSON.parse(input.episodeCanonicalJson);
  const stable = (value: unknown): unknown => Array.isArray(value) ? value.map(stable) : value && typeof value === "object" ? Object.fromEntries(Object.entries(value).sort(([a],[b])=>a<b?-1:a>b?1:0).map(([k,v])=>[k,stable(v)])) : value;
  const {episodeId: _episodeId,...body} = episode;
  requireValue(same(stable(canonicalEpisode),stable(body)), "received episode differs from its identity-bound serialization.");
  requireValue(episode.evidenceKind === "generated_software_fixture" &&
    episode.backendStatus === "generated_executed" && episode.patientAdmission === false && episode.clinicalValidation === false &&
    source.metadata.evidence_kind === "generated_software_fixture" && source.metadata.patient_admission === false, "generated provenance is missing.");
  requireValue([episode.episodeId,episode.caseHash,episode.sourceHash,episode.decisionModelHash].every(h => HASH.test(h)) &&
    episode.caseHash === source.caseHash && episode.sourceBinding.display_case_hash === source.caseHash &&
    episode.sourceBinding.native_source_hash === episode.sourceHash && source.metadata.source_task_hash === episode.sourceHash &&
    episode.sourceBinding.private_reference_published === false, "source identities disagree.");
  requireValue(source.frame === "RAS+" && episode.frame === "RAS+" && episode.physicalUnits === "mm" &&
    same(source.shape,episode.shape) && same(source.affine,episode.affine), "physical source grid changed.");
  const {voxelCount,voxelVolumeMm3} = validateCaseDescriptor(source);
  requireValue(same(source.shape,admittedShape) && voxelCount <= 262144 && episode.history.length > 0 && episode.history.length <= maxSteps &&
    episode.replayFrames.length > 1 && episode.replayFrames.length <= 512, "development replay exceeds its bounded contract.");
  requireValue(episode.geometryAudit.feasible === true && episode.geometryAudit.complete_tool_checked === true &&
    episode.geometryAudit.frontier_checked === true && episode.geometryAudit.source_case_hash === episode.sourceHash,
    "complete-tool and frontier audit is unavailable.");
  requireValue(source.brainMask && source.compartments.length === 1 && source.compartments[0].name === "generated_nominal_target", "missing generated tissue/target arrays.");
  const volume = await hydrateCase(source,api);
  const tissue = new Uint8Array(await api.readAsset(source.brainMask.assetId)).slice();
  requireValue(tissue.length === voxelCount && tissue.every(n => n===0 || n===1) &&
    await sha256Bytes(tissue) === source.brainMask.sha256, "tissue transfer changed.");
  const affineBytes = new Uint8Array(new Float64Array(source.affine.flat()).buffer);
  const affineHeader = new TextEncoder().encode('{"dtype": "<f8", "shape": [4, 4]}');
  const affineFramed = new Uint8Array(affineHeader.length+affineBytes.length); affineFramed.set(affineHeader); affineFramed.set(affineBytes,affineHeader.length);
  requireValue(await sourceImageDigest(volume,source.shape) === episode.sourceBinding.structural_intensity_hash &&
    await sourceFrameDigest(source) === episode.sourceBinding.source_frame_hash &&
    await arrayDigest(tissue,source.shape,"|b1") === episode.sourceBinding.support_hash &&
    await arrayDigest(volume.compartments[0].mask,source.shape,"|b1") === episode.sourceBinding.nominal_target_mask_hash &&
    `sha256:${await sha256Bytes(affineFramed)}` === episode.sourceBinding.affine_hash, "source image, tissue, target or affine digest changed.");
  const tools = new Map(episode.tools.map(t => [t.tool_id,t]));
  requireValue(tools.size === episode.tools.length && tools.size === 2, "ambiguous tool registry.");
  const index = (cell: NativeCell): number => {
    requireValue(Array.isArray(cell) && cell.length===3 && cell.every((n,i) => Number.isInteger(n) && n>=0 && n<source.shape[i]), "invalid native cell.");
    return (cell[0]*source.shape[1]+cell[1])*source.shape[2]+cell[2];
  };
  const indices = (cells: NativeCell[]) => {
    requireValue(Array.isArray(cells) && cells.length<=voxelCount, "invalid cell delta.");
    const values=cells.map(index); requireValue(new Set(values).size===values.length,"duplicate cell within a delta."); return values;
  };
  const sorted = (cells: NativeCell[]) => indices(cells).sort((a,b)=>a-b);
  const union = (lists: NativeCell[][]) => [...new Set(lists.flatMap(indices))].sort((a,b)=>a-b);
  const expected: Array<{actionIndex:number; phase:string; toolId:string|null; mode:string|null; tipRasMm:number[]|null; axis:number[]|null; removed:NativeCell[]; contact:NativeCell[]}> = [
    {actionIndex:-1,phase:"initial",toolId:null,mode:null,tipRasMm:null,axis:null,removed:[],contact:[]}];
  for (const [actionIndex, action] of episode.history.entries()) {
    requireValue([action.source_state_hash,action.result_state_hash].every(h=>HASH.test(h)) &&
      (actionIndex===0 || action.source_state_hash===episode.history[actionIndex-1].result_state_hash), "native engine ancestry changed.");
    if (action.interaction_mode === "stop") {
      requireValue(action.action_id === "STOP" && actionIndex===episode.history.length-1 && action.microsteps.length===0 &&
        action.removed_indices_native.length===0 && action.contact_indices_native.length===0 && action.probe_contact_indices_native.length===0,
        "STOP is not an empty terminal action.");
      expected.push({actionIndex,phase:"stop",toolId:null,mode:"stop",tipRasMm:null,axis:null,removed:[],contact:[]}); continue;
    }
    const tool=tools.get(action.tool_id!);
    requireValue(tool && tool.interactionMode===action.interaction_mode && action.microsteps.length>0 && action.microsteps.length<=128 &&
      action.axis_unit?.length===3 && action.axis_unit.every(Number.isFinite) && Math.abs(Math.hypot(...action.axis_unit)-1)<1e-10,
      "invalid committed tool/microstep geometry.");
    requireValue(same(sorted(action.removed_indices_native),union(action.microsteps.map(m=>m.removed_indices_native))) &&
      same(sorted(action.contact_indices_native),union(action.microsteps.map(m=>m.contact_indices_native))), "macro and microstep deltas disagree.");
    requireValue(action.interaction_mode!=="probe" || action.removed_indices_native.length===0, "probe changed cavity.");
    for (const micro of action.microsteps) expected.push({actionIndex,phase:"insertion",toolId:action.tool_id!,mode:action.interaction_mode,
      tipRasMm:micro.tip_end_mm,axis:action.axis_unit!,removed:micro.removed_indices_native,contact:micro.contact_indices_native});
    for (const micro of [...action.microsteps].reverse()) expected.push({actionIndex,phase:"withdrawal",toolId:action.tool_id!,mode:action.interaction_mode,
      tipRasMm:micro.tip_start_mm,axis:action.axis_unit!,removed:[],contact:[]});
  }
  requireValue(expected.length===episode.replayFrames.length && episode.history.at(-1)!.result_state_hash === episode.nativeEngineFinalStateId,
    "incomplete recorded motion or final engine state.");
  requireValue(episode.history.at(-1)!.interaction_mode==="stop" || episode.history.length===maxSteps, "episode did not terminate.");
  for (const diagnostic of episode.attemptDiagnostics) requireValue(diagnostic.status==="rejected" && diagnostic.stateBefore===diagnostic.stateAfter &&
    diagnostic.removedIndicesNative.length===0 && typeof diagnostic.reason==="string", "rejected attempt changed tissue/state.");
  const removed=new Uint8Array(voxelCount), contact=new Uint8Array(voxelCount), probe=new Uint8Array(voxelCount);
  const frames:ViewerReplay[]=[], contactCounts:number[]=[],probeCounts:number[]=[];
  for (const [i, frame] of episode.replayFrames.entries()) {
    const e=expected[i];
    requireValue(frame.frameIndex===i && frame.actionIndex===e.actionIndex && frame.phase===e.phase && frame.toolId===e.toolId &&
      frame.mode===e.mode && same(frame.tipRasMm,e.tipRasMm) && same(frame.axis,e.axis) &&
      same(sorted(frame.removedIndicesNative),sorted(e.removed)) && same(sorted(frame.contactIndicesNative),sorted(e.contact)),
      "frame is not the exact committed insertion or reversed withdrawal.");
    const removedIndices=indices(frame.removedIndicesNative), contactIndices=indices(frame.contactIndicesNative);
    const expectedProbe = e.mode==="probe" && e.phase==="insertion" ? contactIndices.filter(n=>!removed[n]).sort((a,b)=>a-b) : [];
    requireValue(same(sorted(frame.probeContactIndicesNative),expectedProbe), "probe contact differs from retained tissue contact.");
    for (const n of removedIndices) {requireValue(tissue[n]===1 && removed[n]===0,"non-tissue or duplicate removal.");removed[n]=1;}
    for (const n of contactIndices) {requireValue(tissue[n]===1,"contact outside original tissue.");contact[n]=1;}
    for (const n of expectedProbe) probe[n]=1;
    const remaining=tissue.map((n,j)=>n && !removed[j] ? 1:0);
    const actual=await Promise.all([removed,remaining,contact,probe].map(mask=>arrayDigest(mask,source.shape,"|b1")));
    requireValue(same(actual,[frame.cavityHash,frame.remainingHash,frame.contactHash,frame.probeContactHash]) && HASH.test(frame.stateAfter) &&
      frame.stateBefore === (i===0 ? frame.stateAfter:episode.replayFrames[i-1].stateAfter), "mask digest or replay ancestry mismatch.");
    const target=volume.compartments[0].mask, removedTarget=removed.reduce((s,n,j)=>s+n*target[j],0), totalTarget=count(target);
    const tool=frame.toolId ? tools.get(frame.toolId)!:null;
    frames.push({removedMask:removed.slice(),step:i,stepCount:expected.length-1,scope:"native-source-grid",caseHash:source.caseHash,
      shape:[...source.shape],affine:source.affine.map(row=>[...row]),independentlyAccepted:true,
      removedTargetVolumeMm3:removedTarget*voxelVolumeMm3,removedNormalVolumeMm3:(count(removed)-removedTarget)*voxelVolumeMm3,
      residualTargetVolumeMm3:(totalTarget-removedTarget)*voxelVolumeMm3,
      recordedTool: tool ? {scope:"executed-generated-episode",caseHash:source.caseHash,episodeId:episode.episodeId,
        frameIndex:i,stateId:frame.stateAfter,toolId:tool.tool_id,mode:tool.interactionMode,phase:frame.phase as "insertion"|"withdrawal",
        tipRasMm:[...frame.tipRasMm!],axis:[...frame.axis!],workingLengthMm:tool.working_length_mm,tipLengthMm:tool.tip_length_mm,
        shaftRadiusMm:tool.shaft_radius_mm,tipRadiusMm:tool.tip_radius_mm}:null});
    contactCounts.push(count(contact));probeCounts.push(count(probe));
  }
  requireValue(episode.initialStateId===episode.replayFrames[0].stateAfter && episode.finalStateId===episode.replayFrames.at(-1)!.stateAfter &&
    same(sorted(episode.finalRemovedIndicesNative),Array.from(removed.keys()).filter(n=>removed[n])), "final cavity differs from executed history.");
  const cameraBounds = recordedEpisodeBounds(volume, frames, episode.episodeId);
  for (const frame of frames) frame.recordedEpisodeBounds = cameraBounds;
  return {episode,source,volume,frames,contactCounts,probeCounts};
}
