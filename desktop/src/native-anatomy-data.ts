import type {ViewerVolume,VesselDisplayFilter} from './viewer/contracts';
import type {Point3} from './viewer/coordinates';
import type {ViewerWorkspaceSettings} from './viewer/workspaceSettings';
import type {ArrayDescriptor,ResectionApi,CasePayload} from './types';
export type Layer={id:string;label:string;kind:string;role:string;color:string;review_status:string;source_series:string;probability?:unknown;provenance:unknown;coverage:string};
export type Document={patient_id:string;study_timepoint?:string;snapshot_id:string;reference_series:string;revision:number;layers:Layer[];missing:string[];series:Array<{id:string;label?:string;modality:string;grid_spacing_mm:number[];timepoint:string}>;transforms:Record<string,{status:string;parent_series?:string|null}>};
export type Binding={collectionId:string;document:Document;viewState?:NativeViewState|null};
export type Draft={layerId:string;undo:number;redo:number;changedVoxels:number}|null;
export type Args=Record<string,unknown>;
export interface NativeAnatomyApi {
  openWorkspace?():Promise<Binding|CasePayload|null>;
  startupWorkspace?():Promise<Binding|CasePayload|null>;
  openListedWorkspace?(id:string):Promise<Binding|CasePayload>;
  anatomySavePackage(args:Args):Promise<{path:string}|null>;
  anatomySaveView(args:Args):Promise<{saved:boolean}>;
  anatomyComponents(args:Args):Promise<ComponentReview>;
  anatomyGraph(args:Args):Promise<unknown>;
  anatomyCreate(args:Args):Promise<Binding|null>;
  anatomyReconstruct(args:Args):Promise<{document:Document;result:unknown}>;
  anatomyPlan(args:Args):Promise<unknown>;
  anatomyRemoval(args:Args):Promise<unknown>;
  anatomySection(args:Args):Promise<{pixels:ArrayDescriptor;size:number;frame:string;movingStatus:string|null}>;
  anatomyOpen():Promise<Binding|null>;
  anatomyView(args:Args):Promise<ViewPayload>;
  anatomyEdit(args:Args):Promise<{document:Document;draft:Draft}>;
  anatomyRegister(args:Args):Promise<{document:Document;result:unknown}>;
  anatomyImport(args:Args):Promise<{document:Document}|null>;
  anatomyExport(args:Args):Promise<unknown>;
  anatomySample(args:Args):Promise<unknown>;
}
export type Surface=NonNullable<ViewerVolume['compartments'][number]['surface']>;
export type ViewPayload=Omit<ViewerVolume,'mri'|'compartments'|'tractography'|'labelSets'>&{labelSets?:Array<{id:string;data:ArrayDescriptor}>;tractography?:Array<{id:string;label:string;points:ArrayDescriptor;offsets:ArrayDescriptor}>;mri:ArrayDescriptor;modality:string;registrationStatus:string;alignmentPreviews?:string[];unavailableLayers:string[];compartments:Array<{kind?:string;sourceIdentity?:string;qa?:import('./viewer/contracts').AnatomyQuality;surfaceDiagnostics?:import('./viewer/surfacePolicy').SurfaceDiagnostics;name:string;color:string;volumeMm3:number;displayFilter?:VesselDisplayFilter;mask?:ArrayDescriptor;labelSet?:string;labelValue?:number;surface?:{empty:true}|Record<keyof Surface,ArrayDescriptor>}>};
export async function hydrateAnatomy(payload:ViewPayload,api:ResectionApi,cachedImage?:Float32Array,cached?:Pick<ViewerVolume,'labelSets'|'tractography'>):Promise<ViewerVolume>{
  async function bytes(d:ArrayDescriptor){
    if(!d || d.byteLength>512*1024**2 || d.byteOrder!=='little' || d.order!=='C')throw Error('Unsupported anatomy transfer');
    const b=await api.readAsset(d.assetId);if(b.byteLength!==d.byteLength)throw Error('Incomplete anatomy transfer');
    return (b.byteOffset===0&&b.byteLength===b.buffer.byteLength?b.buffer:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)) as ArrayBuffer;
  }
  const mri=cachedImage??new Float32Array(await bytes(payload.mri));
  if(mri.length!==payload.shape.reduce((a,b)=>a*b,1))throw Error('Native image grid mismatch');
  const compartments:ViewerVolume['compartments']=[];
  for(const layer of payload.compartments){
    const mask=layer.mask?new Uint8Array(await bytes(layer.mask)):new Uint8Array(0);if(!layer.labelSet&&mask.length!==mri.length)throw Error('Overlay grid mismatch');
    let surface:Surface|undefined;
    if(layer.surface&&'empty' in layer.surface){
      if(layer.surface.empty!==true||Object.keys(layer.surface).length!==1)throw Error('Invalid empty anatomy surface');
      surface={positions:new Float32Array(),source:new Float32Array(),normals:new Float32Array(),sourceNormals:new Float32Array(),indices:new Uint32Array()};
    }else if(layer.surface){
      const s=layer.surface;surface={positions:new Float32Array(await bytes(s.positions)),source:new Float32Array(await bytes(s.source)),
        normals:new Float32Array(await bytes(s.normals)),sourceNormals:new Float32Array(await bytes(s.sourceNormals)),indices:new Uint32Array(await bytes(s.indices))};
      if(surface.positions.length%3 || surface.indices.length%3 || surface.source.length!==surface.positions.length || surface.normals.length!==surface.positions.length || surface.sourceNormals.length!==surface.positions.length || surface.indices.some(i=>i>=surface!.positions.length/3))throw Error('Invalid indexed anatomy mesh');
    }
    compartments.push({...layer,mask,surface});
  }
  const tractography=[];for(const t of payload.tractography??[]){const resident=cached?.tractography?.find(r=>r.id===t.id);if(resident){tractography.push(resident);continue;}tractography.push({...t,points:new Float32Array(await bytes(t.points)),offsets:new Uint32Array(await bytes(t.offsets))});}
  const labelSets=[];for(const set of payload.labelSets??[]){const resident=cached?.labelSets?.find(r=>r.id===set.id);if(resident){labelSets.push(resident);continue;}const data=new Uint16Array(await bytes(set.data));if(data.length!==mri.length)throw Error("Categorical grid mismatch");labelSets.push({id:set.id,data});}
  return {...payload,mri,compartments,tractography,labelSets};
}

export interface NativeScanState {cursor:Point3|null;visible:Record<string,boolean>;settings:ViewerWorkspaceSettings}
export interface NativeViewState {schema:'patient-anatomy-view-v1';patientId:string;snapshotId:string;seriesId:string;states:Record<string,NativeScanState>;vesselScope?:VesselScope;vesselDisplayVersion?:2}
export interface ComponentReview {reviewId:string;layerId:string;components:Array<{id:number;voxels:number;volumeMm3:number;center:Point3}>}
export function isNativeBinding(value:unknown):value is Binding {return !!value&&typeof value==='object'&&'collectionId' in value&&'document' in value;}

export type VesselScope='brain'|'all'|'intracranial'|'excluded';
