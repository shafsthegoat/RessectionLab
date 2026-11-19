import type {TissueDisplaySupport} from './viewer/contracts';
import type {ArrayDescriptor} from './types';
import type {Point3} from './viewer/coordinates';
import {inverseAffine,rasAffine} from './viewer/coordinates.ts';
import type {NativeApproach} from './native-approaches';
import {sha256Bytes} from './source-integrity.ts';
import {hasAspirationSeed} from './viewer/operativeSeed.ts';

export type OperativePhase = 'overview' | 'bone' | 'dura' | 'brain' | 'tumor';
type Tissue = Exclude<OperativePhase,'overview'>;
export interface FieldLayer {
  id:string; label:string; role:Tissue|'vessels'|'fluid'|'lesionContext'; sourceSupport?:TissueDisplaySupport; detail?:Detail|null;
  sourceSpacingMm?:Point3; labelSpacingMm?:Point3; inferenceSpacingMm?:Point3;
  adjudication?:{disagreementVoxels:number; preferredLayerIds:string[]}; evidence:string; sourceSeries:string;
  model?:string|null; vascularContextOnly?:boolean; reviewStatus:string; sampledVoxels:number; proxy:boolean; sampledDataSha256?:string; coverageMeaning?:string; coveredVoxels?:number;
}
export const detailBits={grayMatter:32,whiteMatter:64,csf:128,ventricles:256,artery:512,vein:1024,sinus:2048,
  enhancing:4096,nonenhancingCore:8192,perilesionalSignal:16384,tumorCore:32768,wholeAbnormality:65536} as const;
export type Detail=keyof typeof detailBits;
export interface SourceOverview {
  shape:[number,number]; spacingMm:number; rightRas:Point3; upRas:Point3;
  entryPx:[number,number]; targetPx:[number,number]; voxels:Uint8Array;
}
export interface OperativeIntensity {
  overview?:SourceOverview;
  shape:Point3; origin:Point3; spacingMm:Point3; unknownValue:0; voxels:Uint8Array;
  sourceSeries:string; modality:string; sourceSha256:string; dataSha256:string;
  window:[number,number]; windowUnits:'HU'|'source intensity'; windowMethod:string;
  coveredVoxels:number; sourceShape:Point3; sourceVoxelToPatientRasMm:number[][];
}
type IntensityPacket=Omit<OperativeIntensity,'voxels'|'overview'>&{signal:ArrayDescriptor;overview?:Omit<SourceOverview,'voxels'>&{signal:ArrayDescriptor}};
interface FieldPacket {
  snapshotId:string; approachId:string; parameterHash:string; targetLayerId:string;
  shape:Point3; origin:Point3; spacingMm:number; radiusMm:number; routeRadiusMm:number;
  entryDepthMm:number; stations:Record<Tissue,number|null>; labels:ArrayDescriptor;
  intensity?:IntensityPacket;
  stationEvidence:Record<Tissue,'source'|'proxy'|null>;
  units:'mm'; targetRasMm:Point3; basisRas:[Point3,Point3,Point3];
  materialBits:Record<Tissue|'vessels',number>;
  layers:FieldLayer[]; excluded:{id:string;label:string;reason:string}[];
  encounters:{id:string;label:string;status:string}[]; blockers:string[]; unknowns:string[]; stationIssues:string[];
  simulationExecutionAuthorized:false;
}
export type OperativeField = Omit<FieldPacket,'labels'|'intensity'> & {voxels:Uint8Array|Uint16Array|Uint32Array;intensity?:OperativeIntensity};

export const operativeSteps = [
  {id:'bone',label:'Bone flap'}, {id:'dura',label:'Dural opening'},
  {id:'brain',label:'Brain retraction'}, {id:'tumor',label:'Tumor aspiration'},
] as const;

export function operativeRouteIssue(route:NativeApproach|null, activeTarget:string|null|undefined, snapshot:string) {
  if (!route) return 'Generate or select an approach in Plan.';
  if (route.sourceSnapshotId!==snapshot || route.assessmentStatus!=='current')
    return 'Reassess the selected approach against the current study in Plan.';
  if (!route.targetLayerId || route.targetLayerId!==activeTarget)
    return 'Select a lesion and an approach assessed for that lesion in Plan.';
  return null;
}

const triple=(value:unknown):value is Point3=>Array.isArray(value)&&value.length===3&&value.every(Number.isFinite);
const bits = {bone:1,dura:2,brain:4,tumor:8,vessels:16} as const;

async function readIntensity(packet:IntensityPacket, field:Pick<FieldPacket,'shape'|'origin'|'spacingMm'>,
                             readAsset:(id:string)=>Promise<Uint8Array>):Promise<OperativeIntensity> {
  const hash=(s:unknown)=>typeof s==='string'&&/^[0-9a-f]{64}$/.test(s);
  const supported=['T1','T1CE','T2','FLAIR','MRI','CT','CTA'];
  if (!packet || packet.unknownValue!==0 || !supported.includes(packet.modality) ||
      typeof packet.sourceSeries!=='string' || !packet.sourceSeries.trim() ||
      !hash(packet.sourceSha256) || !hash(packet.dataSha256) || !triple(packet.sourceShape) ||
      !packet.sourceShape.every(n=>Number.isInteger(n)&&n>0))
    throw Error('The operative intensity source is invalid.');
  try {
    inverseAffine(rasAffine(packet.sourceVoxelToPatientRasMm,'RAS+'));
  } catch {
    throw Error('The operative intensity source affine is invalid.');
  }
  if (!triple(packet.shape) || !packet.shape.every(n=>Number.isInteger(n)&&n>=2&&n<=256) ||
      packet.shape.reduce((a,b)=>a*b,1)>2_000_000 ||
      !triple(packet.origin) || !triple(packet.spacingMm) || !packet.spacingMm.every((step,i)=>step>0&&
        Math.abs(packet.origin[i]-field.origin[i])<1e-5&&
        Math.abs(step*(packet.shape[i]-1)-field.spacingMm*(field.shape[i]-1))<1e-5))
    throw Error('The operative intensity grid does not span the local field.');
  if (!Array.isArray(packet.window) || packet.window.length!==2 || !packet.window.every(Number.isFinite) ||
      packet.window[1]<=packet.window[0] || typeof packet.windowMethod!=='string' || !packet.windowMethod.trim() ||
      packet.windowUnits!==(['CT','CTA'].includes(packet.modality)?'HU':'source intensity'))
    throw Error('The operative intensity window is invalid.');
  const {signal,overview:inset,...metadata}=packet;
  const size=packet.shape.reduce((a,b)=>a*b,1);
  if(!signal||typeof signal.assetId!=='string'||!signal.assetId||signal.dtype!=='uint8'||signal.byteOrder!=='little'||signal.order!=='C'||!hash(signal.sha256)||
      !triple(signal.shape)||!signal.shape.every((n,i)=>n===packet.shape[i])||signal.byteLength!==size||
      !Number.isInteger(packet.coveredVoxels)||packet.coveredVoxels<1||packet.coveredVoxels>size)
    throw Error('The operative intensity transfer does not match its field.');
  const voxels=await readAsset(signal.assetId);
  if(voxels.length!==size||await sha256Bytes(voxels)!==signal.sha256||
      voxels.reduce((count,value)=>count+Number(value!==0),0)!==packet.coveredVoxels)
    throw Error('Could not verify the operative scan intensity.');
  let overview:SourceOverview|undefined;
  if(inset){
    const pair=(p:unknown):p is [number,number]=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite);
    const {signal:asset,...geometry}=inset;
    if(!pair(inset.shape)||!inset.shape.every(n=>Number.isInteger(n)&&n>=2&&n<=162)||
       !Number.isFinite(inset.spacingMm)||inset.spacingMm<=0||!pair(inset.entryPx)||!pair(inset.targetPx)||
       !triple(inset.rightRas)||!triple(inset.upRas)||!asset||asset.dtype!=='uint8'||asset.order!=='C'||
       asset.byteOrder!=='little'||!pair(asset.shape)||asset.shape.some((n,i)=>n!==inset.shape[i])||
       asset.byteLength!==inset.shape[0]*inset.shape[1])throw Error('Invalid source overview geometry.');
    const pixels=await readAsset(asset.assetId);
    if(pixels.length!==asset.byteLength||await sha256Bytes(pixels)!==asset.sha256)throw Error('Could not verify the source overview.');
    overview={...geometry,voxels:pixels};
  }
  return {...metadata,voxels,overview};
}

export async function readOperativeField(raw:unknown, route:NativeApproach, readAsset:(id:string)=>Promise<Uint8Array>):Promise<OperativeField> {
  const data=raw as FieldPacket;
  if (!data || data.simulationExecutionAuthorized!==false || data.snapshotId!==route.sourceSnapshotId ||
      data.approachId!==route.id || data.parameterHash!==route.parameterHash || data.targetLayerId!==route.targetLayerId)
    throw Error('The operative field belongs to another study revision or approach. Reopen it from Plan.');
  const {labels,intensity:packet,...field}=data;
  if (field.units!=='mm' || !triple(field.targetRasMm) || !Array.isArray(field.basisRas) ||
      field.basisRas.length!==3 || !field.basisRas.every(triple) ||
      !field.materialBits || !Object.entries(bits).every(([role,bit])=>field.materialBits[role as keyof typeof bits]===bit) ||
      !triple(field.shape) || !field.shape.every(n=>Number.isInteger(n)&&n>=1) || !triple(field.origin) ||
      ![field.spacingMm,field.radiusMm,field.routeRadiusMm,field.entryDepthMm].every(n=>Number.isFinite(n)&&n>0) ||
      !field.stations || !field.stationEvidence ||
      ![field.layers,field.excluded,field.encounters,field.blockers,field.unknowns,field.stationIssues].every(Array.isArray))
    throw Error('The operative field geometry is incomplete.');
  const maxDepth=field.origin[2]+(field.shape[2]-1)*field.spacingMm;
  for(const {id} of operativeSteps){
    const station=field.stations[id], evidence=field.stationEvidence[id];
    if(station===null ? evidence!==null : !Number.isFinite(station) || station<field.origin[2]-1.e-6 ||
       station>maxDepth+1.e-6 || !(evidence==='source'||(id==='dura'&&evidence==='proxy')))
      throw Error('The operative station is outside its tissue evidence.');
  }
  const size=field.shape.reduce((a,b)=>a*b,1);
  const wordSize=labels?.dtype==='uint32'?4:labels?.dtype==='uint16'?2:1;
  const extended=wordSize>1;
  if (extended && !Object.entries(detailBits).every(([role,bit])=>(field.materialBits as Record<string,number>)[role]===bit))
    throw Error('Unrecognized operative tissue classes.');
  if (!labels || size>16*1024**2 || labels.byteLength!==size*wordSize || (!extended && labels.dtype!=='uint8') || labels.order!=='C' ||
      labels.byteOrder!=='little' || !triple(labels.shape) || labels.shape.some((v,i)=>v!==field.shape[i]))
    throw Error('The operative tissue grid does not match the field.');
  const bytes=await readAsset(labels.assetId);
  if (bytes.length!==labels.byteLength || await sha256Bytes(bytes)!==labels.sha256 || (!extended && bytes.some(value=>value>31)))
    throw Error('Could not verify the operative tissue labels.');
  const aligned=bytes.byteOffset%wordSize===0?bytes:bytes.slice();
  const voxels=wordSize===4?new Uint32Array(aligned.buffer,aligned.byteOffset,size):
    wordSize===2?new Uint16Array(aligned.buffer,aligned.byteOffset,size):bytes;
  if(wordSize===4&&voxels.some(value=>value>131071))throw Error('Unknown operative label bits.');
  const intensity=packet===undefined?undefined:await readIntensity(packet,field,readAsset);
  return {...field,voxels,...(intensity?{intensity}:{})};
}

export function playableSteps(field:OperativeField, retract:boolean) {
  return operativeSteps.filter(({id})=>field.stations[id]!==null && (id!=='brain'||retract) &&
    (id!=='tumor'||hasAspirationSeed(field)));
}

export function timelinePose(steps:ReturnType<typeof playableSteps>,value:number):{phase:OperativePhase;progress:number} {
  if (!steps.length || value<=0) return {phase:'overview',progress:0};
  const position=Math.min(value,steps.length), index=Math.ceil(position)-1;
  return {phase:steps[index].id, progress:position-index};
}
