import type {CasePayload,ResectionApi,ViewerCase} from './types';
import type {PersistentPassiveResult,PassiveToolGeometry} from './persistent-passive-types';
import type {ViewerPassiveReplay} from './viewer/contracts';
import {validateCaseDescriptor,hydrateCase} from './case-data.ts';
import {arrayDigest,sha256Bytes,sourceFrameDigest,sourceImageDigest} from './source-integrity.ts';
import {checkedPairedPassiveEpisode} from './persistent-passive-data.ts';

const HASH=/^sha256:[a-f0-9]{64}$/;
// Exact paired v1 fixture receipts. New generated protocols require a new adapter.
const FIXED={
  episode:'sha256:33cbb8bdea5873acece2461fbf1c992791ff7cae18ba2099c4bc1724cd76717a',
  case:'sha256:e8e99cc9ce8ae0efd30540e35d4043420deeec00818818f9f2057174c582b845',
  source:'sha256:c0f4fb9e6bcde3c263d23596e54f393b985394ce243afb066a685c3a3e9c0195',
  decision:'sha256:7e194ca924330fb826a5e5907b8fae96be9b135459d68335578f58e71626b98e',
  binding:'sha256:a40bf2e8313c69b44af29278c71547ae93a653b2236b83ea2ce4a3d19dbed1e5',
  structuralIntensity:'sha256:94b52d6a03fd2fab18435a84cfccb4b33b81f21c9982c21932b54b4b8c7f6b97',
  support:'sha256:ce676378d1f29047f81313f0d5476b6aa8c7c5d79ec01f44409d16357b9d039e',
  nominalTarget:'sha256:00f0288612fa55d2514e3cd3ef7fa62b5505ce94467e229cf5b2e70652a5b5ff',
  affine:'sha256:0a02006154b1285bd52c99defbda45d9b472a933b18e6bf958530e6173750ab4',
  sourceFrame:'sha256:8cb91602c6df8cdae6c2d385c94572358633255b5b5c0d3d40e97adc88ebf343',
};
const same=(a:unknown,b:unknown)=>JSON.stringify(a)===JSON.stringify(b);
function need(ok:unknown,why:string):asserts ok{if(!ok)throw Error(`Persistent passive replay withheld: ${why}`);}
export interface PersistentPassiveView {
  source:CasePayload;volume:ViewerCase;
  episode:PersistentPassiveResult['episode'];frames:ViewerPassiveReplay[];
  generatedOnly:true;origin:'live'|'reopened';
}
function geometry(tool:PassiveToolGeometry){
  need(typeof tool?.tool_id==='string'&&tool.tool_id.length>0&&
    tool.tip_radius_mm===.15&&tool.shaft_radius_mm===.1&&
    tool.working_length_mm===6&&tool.max_access_angle_deg===30&&
    tool.tip_length_mm===.4&&
    tool.parameter_source==='generic research geometry; not a verified commercial device',
    'source-bound tool geometry is invalid');
}
/** The source and complete native tool catalog are checked before any frame is exposed.
 * A backend fresh replay is additionally required for `origin="reopened"`. */
export async function hydratePersistentPassiveEpisode(
  input:PersistentPassiveResult&{validation?:'authoritative_generated_passive_replay'},
  api:ResectionApi,origin:'live'|'reopened'='live',
):Promise<PersistentPassiveView>{
  need(origin==='live'||input.validation==='authoritative_generated_passive_replay',
    'saved passive replay lacks backend revalidation');
  const source=structuredClone(input.case),binding=structuredClone(input.sourceBinding);
  const episode=await checkedPairedPassiveEpisode(input.episode,input.episodeCanonicalJson);
  const {voxelCount}=validateCaseDescriptor(source);
  need(source.frame==='RAS+'&&same(source.shape,[13,13,12])&&voxelCount===2028&&
    source.metadata?.evidence_kind==='generated_software_fixture'&&source.metadata?.patient_admission===false&&
    source.metadata?.source_task_hash===episode.sourceHash&&source.caseHash===binding.caseHash&&
    binding.sourceHash===episode.sourceHash&&binding.decisionModelHash===episode.decisionModelHash&&
    episode.episodeId===FIXED.episode&&source.caseHash===FIXED.case&&
    episode.sourceHash===FIXED.source&&episode.decisionModelHash===FIXED.decision&&
    binding.bindingHash===FIXED.binding&&
    binding.structuralIntensityHash===FIXED.structuralIntensity&&
    binding.supportHash===FIXED.support&&
    binding.nominalTargetMaskHash===FIXED.nominalTarget&&
    binding.affineHash===FIXED.affine&&binding.sourceFrameHash===FIXED.sourceFrame&&
    [binding.bindingHash,binding.structuralIntensityHash,binding.sourceFrameHash,binding.supportHash,
      binding.nominalTargetMaskHash,binding.affineHash].every(x=>HASH.test(x)),
    'generated case and native source identity disagree');
  need(Array.isArray(binding.tools)&&binding.tools.length===2&&
    new Set(binding.tools.map(t=>t.tool_id)).size===2&&
    binding.tools[0].tool_id===episode.protocol.stationary_tool_id&&
    binding.tools[1].tool_id===episode.protocol.moving_tool_id,
    'source-bound two-tool registry changed');
  binding.tools.forEach(geometry);
  need(source.brainMask&&source.compartments.length===1&&
    source.compartments[0].name==='generated_nominal_target',
    'generated support or nominal source absent');
  const volume=await hydrateCase(source,api);
  const support=new Uint8Array(await api.readAsset(source.brainMask.assetId)).slice();
  need(support.length===voxelCount&&support.every(n=>n===0||n===1)&&
    await sha256Bytes(support)===source.brainMask.sha256,
    'generated support bytes changed');
  const affineBytes=new Uint8Array(new Float64Array(source.affine.flat()).buffer);
  const affineHeader=new TextEncoder().encode('{"dtype": "<f8", "shape": [4, 4]}');
  const affineFramed=new Uint8Array(affineHeader.length+affineBytes.length);
  affineFramed.set(affineHeader);affineFramed.set(affineBytes,affineHeader.length);
  need(await sourceImageDigest(volume,source.shape)===binding.structuralIntensityHash&&
    await sourceFrameDigest(source)===binding.sourceFrameHash&&
    await arrayDigest(support,source.shape,'|b1')===binding.supportHash&&
    await arrayDigest(volume.compartments[0].mask,source.shape,'|b1')===binding.nominalTargetMaskHash&&
    `sha256:${await sha256Bytes(affineFramed)}`===binding.affineHash,
    'source image, support, nominal target or physical frame changed');
  const bounds:[[number,number,number],[number,number,number]]=[
    [Infinity,Infinity,Infinity],[-Infinity,-Infinity,-Infinity]];
  const frames:ViewerPassiveReplay[]=episode.frames.map(({frameIndex,operative})=>{
    const states=[operative.moving_tool,operative.other_tool] as const;
    const ordered=[binding.tools[1],binding.tools[0]] as const;
    const recordedTools=states.map((state,i)=>{
      const tool=ordered[i];need(state.pose&&state.tool_id===tool.tool_id,
        'operative pose no longer matches the source-bound tool');
      const pose=state.pose;
      const poseStatus=state.pose_status;
      need(poseStatus==='held_in_declared_void'||poseStatus==='access_pose_recorded',
        'paired tool cannot be absent from this protocol');
      for(const [point,radius] of [
        [pose.tip_mm.map((n,j)=>n-pose.axis_unit[j]*tool.working_length_mm),tool.shaft_radius_mm],
        [pose.tip_mm.map((n,j)=>n-pose.axis_unit[j]*tool.tip_length_mm),tool.shaft_radius_mm],
        [pose.tip_mm.map((n,j)=>n-pose.axis_unit[j]*tool.tip_length_mm),tool.tip_radius_mm],
        [pose.tip_mm,tool.tip_radius_mm],
      ] as [number[],number][])for(let a=0;a<3;a++){
        bounds[0][a]=Math.min(bounds[0][a],point[a]-radius);
        bounds[1][a]=Math.max(bounds[1][a],point[a]+radius);
      }
      return {scope:'generated-persistent-passive-episode' as const,caseHash:source.caseHash,
        episodeId:episode.episodeId,frameIndex,toolId:tool.tool_id,
        mode:i===0?'probe' as const:'aspirate' as const,
        tipRasMm:[...pose.tip_mm] as [number,number,number],
        axis:[...pose.axis_unit] as [number,number,number],
        workingLengthMm:tool.working_length_mm,tipLengthMm:tool.tip_length_mm,
        shaftRadiusMm:tool.shaft_radius_mm,tipRadiusMm:tool.tip_radius_mm,
        retainedAtDepth:state.retained_at_depth,physicalAbsenceClaim:false as const,
        poseStatus,activation:state.activation};
    });
    const event=frameIndex===2?{kind:'generated_inspection' as const,status:operative.inspection_status as 'verified'|'unresolved'}:
      frameIndex===5?{kind:'stop' as const,reason:episode.history[4].stop_reason??undefined}:null;
    return {scope:'generated-persistent-passive-episode',caseHash:source.caseHash,
      episodeId:episode.episodeId,frameIndex,stepCount:5,shape:[...source.shape],
      affine:source.affine.map(row=>[...row]),phase:operative.phase,
      elapsedGeneratedSeconds:operative.elapsed_generated_seconds,
      inspectionStatus:operative.inspection_status,event,recordedTools,
      recordedEpisodeBounds:{caseHash:source.caseHash,episodeId:episode.episodeId,boundsRasMm:bounds},
      primaryPairClearanceChecked:true,independentTissueMotionChecked:true};
  });
  need(frames.length===6&&bounds.flat().every(Number.isFinite),'paired physical bounds unavailable');
  for(const frame of frames)frame.recordedEpisodeBounds={caseHash:source.caseHash,episodeId:episode.episodeId,
    boundsRasMm:[[...bounds[0]],[...bounds[1]]]};
  return {source,volume,episode,frames,generatedOnly:true,origin};
}
