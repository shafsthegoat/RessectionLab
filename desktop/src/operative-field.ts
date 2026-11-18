import type {ArrayDescriptor} from './types';
import type {Point3} from './viewer/coordinates';
import type {NativeApproach} from './native-approaches';
import {sha256Bytes} from './source-integrity.ts';
import {hasAspirationSeed} from './viewer/operativeSeed.ts';

export type OperativePhase = 'overview' | 'bone' | 'dura' | 'brain' | 'tumor';
type Tissue = Exclude<OperativePhase,'overview'>;
export interface FieldLayer {
  id:string; label:string; role:Tissue|'vessels'; evidence:string; sourceSeries:string;
  reviewStatus:string; sampledVoxels:number; proxy:boolean;
}
interface FieldPacket {
  snapshotId:string; approachId:string; parameterHash:string; targetLayerId:string;
  shape:Point3; origin:Point3; spacingMm:number; radiusMm:number; routeRadiusMm:number;
  entryDepthMm:number; stations:Record<Tissue,number|null>; labels:ArrayDescriptor;
  stationEvidence:Record<Tissue,'source'|'proxy'|null>;
  units:'mm'; targetRasMm:Point3; basisRas:[Point3,Point3,Point3];
  materialBits:Record<Tissue|'vessels',number>;
  layers:FieldLayer[]; excluded:{id:string;label:string;reason:string}[];
  encounters:{id:string;label:string;status:string}[]; blockers:string[]; unknowns:string[]; stationIssues:string[];
  simulationExecutionAuthorized:false;
}
export type OperativeField = Omit<FieldPacket,'labels'> & {voxels:Uint8Array};

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

export async function readOperativeField(raw:unknown, route:NativeApproach, readAsset:(id:string)=>Promise<Uint8Array>):Promise<OperativeField> {
  const data=raw as FieldPacket;
  if (!data || data.simulationExecutionAuthorized!==false || data.snapshotId!==route.sourceSnapshotId ||
      data.approachId!==route.id || data.parameterHash!==route.parameterHash || data.targetLayerId!==route.targetLayerId)
    throw Error('The operative field belongs to another study revision or approach. Reopen it from Plan.');
  const {labels,...field}=data;
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
  if (!labels || size>16*1024**2 || labels.byteLength!==size || labels.dtype!=='uint8' || labels.order!=='C' ||
      labels.byteOrder!=='little' || !triple(labels.shape) || labels.shape.some((v,i)=>v!==field.shape[i]))
    throw Error('The operative tissue grid does not match the field.');
  const voxels=await readAsset(labels.assetId);
  if (voxels.length!==size || await sha256Bytes(voxels)!==labels.sha256 || voxels.some(value=>value>31))
    throw Error('Could not verify the operative tissue labels.');
  return {...field,voxels};
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
