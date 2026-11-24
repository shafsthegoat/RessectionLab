import {studyLayerDisplayKind} from './native-study-state.ts';
import type {ViewerVolume,VesselDisplayFilter,PatientContext,SurfaceAppearanceSource} from './viewer/contracts';
import {inverseAffine,rasAffine} from './viewer/coordinates.ts';
import type {Point3} from './viewer/coordinates';
import type {ViewerWorkspaceSettings} from './viewer/workspaceSettings';
import type {ArrayDescriptor,ResectionApi,CasePayload} from './types';
export type Layer={
  id:string;label:string;kind:string;role:string;color:string;review_status:string;source_series:string;
  probability?:unknown;provenance:unknown;coverage:string;data?:{shape:number[];sha256:string};label_value?:number;
  affine?:number[][];acquisition_grid_spacing_mm?:number[];output_spacing_mm?:number[];
  acquisition_resolution?:string|number[];transform_dependencies?:string[];
  coverage_fields?:{inference_bounds_voxel_inclusive?:number[][];supported_axial_indices?:number[];outside?:string;zero_meaning?:string};
  availability?:string;uncertainty?:unknown;label_ontology?:unknown;ontology?:unknown;score_meaning?:string;
};
export type Series={
  id:string;label?:string;modality:string;grid_spacing_mm:number[];timepoint:string;affine?:number[][];data?:{shape:number[]};
  acquisition_resolution?:string|number[];coverage?:{status?:string;limitations?:Array<{kind:string;basis?:string}>};
};
export type Document={
  before_acquisition?:{snapshotId:string;revision:number;sources:Array<{id:string;label:string;modality:string;originalSha256:string}>;
    layers:Array<{id:string;label:string;sourceSeries:string;arraySha256:string;labelValue:number;review:string}>};
  display?:{initial_layers?:string[];cursor_ras_mm?:Point3;entry_ras_mm?:Point3};
  vascular_review?:{schema?:string};
  source_assets?:unknown[];patient_id:string;study_timepoint?:string;snapshot_id:string;reference_series:string;
  tractography?:Array<{id:string;label:string;source_series:string}>;
  revision:number;layers:Layer[];missing:string[];series:Series[];
  transforms:Record<string,{status:string;parent_series?:string|null;uncertainty_mm?:number|null;method?:string;
    landmarks?:unknown[]|{reviewer:string;moving:Point3[];reference:Point3[];errors_mm:number[]}}>;
};
export type EnvironmentRole='brain'|'tumor'|'bone'|'vessels';
export type AcquisitionContribution={seriesId:string;label:string;modality:string;anatomicalRoles:string[];registration:string;
  structures?:Array<{layerId:string;anchor:boolean;channel:boolean;staleInputs:string[]}>;
  qc:string|null;warnings:string[];disposition:string;reason:string;layerCount:number;emptyLayers:string[];uses:string[];modelInputs:string[];suppliedTracts:string[];asset:boolean};
export type RegistrationSection={pixels:ArrayDescriptor;size:number;frame:string;movingStatus:string|null;
  center:Point3;u:Point3;v:Point3;spacingMm:number;sourceFromSection?:number[][];voxelFromSection?:number[][];sourceShape?:number[]};
export type NativeVesselBranch={branch_id:string;closed:boolean;length_mm:number;source_cells:number[];
  start_ras_mm:Point3;end_ras_mm:Point3;midpoint_ras_mm:Point3;preview_ras_mm:Point3[];
  unsupported_diagonal_edges:number;connectivity:string};
export type NativeVesselGraph={branches?:NativeVesselBranch[];
  nearest?:{source_cell_id:number;distance_mm:number;radius_mm:number|null;point_ras_mm:Point3;radius_limited_by_image_extent:boolean;radius_limited_by_workspace?:boolean}};
export interface SelectedRouteForecast {
  schema:'resectionlab.native-route-forecast.v1';decision:'STOP';stage:'before-access';reason:string;
  preparationActions:number;withdrawalActions:number;targetRemovedMm3:number;nonTargetRemovedMm3:number;
  elapsedSeconds:number;physicalCandidatesExamined:number;requiredEvidence:string[];
}
export type EnvironmentOptions={selectedRouteForecast?:SelectedRouteForecast;acquisitions?:AcquisitionContribution[];referenceIssue:string|null;layers:Array<{
  role:EnvironmentRole;layerId:string;status:'ready'|'estimated_alignment'|'unavailable';reason?:string|null;
}>};
export type Binding={collectionId:string;document:Document;viewState?:NativeViewState|null;readOnlyStudy?:boolean};
export type Draft={layerId:string;undo:number;redo:number;changedVoxels:number}|null;
export type Args=Record<string,unknown>;
export type IntakeKind='nifti'|'dicom'|'labels';
export type IntakeSelection={
  selectionId:string;
  kind:IntakeKind;
  label:string;
  suggestedRole?:string;
  candidates?:string[];
  patientId?:string;
  studyId?:string;
  timepoint?:string;
  warning?:string;
  supported?:boolean;
  storage?:'series'|'source-asset';
};
export type IntakeItem={selectionId:string;kind:IntakeKind;modality?:string;seriesId?:string;sourceInputIndex?:number;roleMap?:Record<string,string|{kind:string;label:string}>;label?:string};
export type PreparationStage={id:string;label:string;status:string;reason?:string;seriesId?:string;retryable?:boolean;device?:string};
export type PreparationPlan={id:string;stages:PreparationStage[]};
export type PreparationResponse={document:Document;plan:PreparationPlan;outcomes?:PreparationStage[]};
export interface NativeAnatomyApi {
  openWorkspace?():Promise<Binding|CasePayload|null>;
  startupWorkspace?():Promise<Binding|CasePayload|null>;
  openListedWorkspace?(id:string):Promise<Binding|CasePayload>;
  anatomySavePackage(args:Args):Promise<{path:string}|null>;
  anatomySaveView(args:Args):Promise<{saved:boolean}>;
  anatomyComponents(args:Args):Promise<ComponentReview>;
  anatomyGraph(args:Args):Promise<NativeVesselGraph>;
  anatomyCreate(args:Args):Promise<Binding|null>;
  anatomyReconstruct(args:Args):Promise<{document:Document;result:unknown}>;
  anatomyPlan(args:Args):Promise<unknown>;
  anatomyRemoval(args:Args):Promise<unknown>;
  anatomySection(args:Args):Promise<RegistrationSection>;
  anatomyOpen():Promise<Binding|null>;
  anatomyView(args:Args):Promise<ViewPayload>;
  anatomyEdit(args:Args):Promise<{document:Document;draft:Draft}>;
  anatomyRegister(args:Args):Promise<{document:Document;result:unknown;retiredClipping?:RetiredClipping[]}>;
  anatomyImport(args:Args):Promise<{document:Document}|null>;
  anatomyPickIntake(args:Args):Promise<{items:IntakeSelection[]}|null>;
  anatomyImportBatch(args:Args&{confirmedAssociation:true;sourcePatientId?:string}):Promise<{document:Document}>;
  anatomyPrepare(args:Args):Promise<PreparationResponse>;
  anatomyExport(args:Args):Promise<unknown>;
  anatomySample(args:Args):Promise<unknown>;
}
export type Surface=NonNullable<ViewerVolume['compartments'][number]['surface']>;
export type ViewPayload=Omit<ViewerVolume,'mri'|'compartments'|'tractography'|'labelSets'|'patientContext'|'headSupport'>&{appearanceSources?:Array<Pick<SurfaceAppearanceSource,'seriesId'|'modality'|'sourceSha256'|'affine'|'nativeAffine'|'nativeShape'>&{signal:ArrayDescriptor}>;headSupport?:ArrayDescriptor;patientContext?:{sources:PatientContext['sources'];affine?:ViewerVolume['affine'];shape?:ViewerVolume['shape'];signal?:ArrayDescriptor};labelSets?:Array<{id:string;data:ArrayDescriptor}>;tractography?:Array<{id:string;label:string;points:ArrayDescriptor;offsets:ArrayDescriptor}>;mri:ArrayDescriptor;modality:string;registrationStatus:string;patientFrameReady?:boolean;alignmentPreviews?:string[];unavailableLayers:string[];compartments:Array<{sourceSeries?:string;alignment?:string;kind?:string;displayKind?:string;sourceIdentity?:string;qa?:import('./viewer/contracts').AnatomyQuality;surfaceDiagnostics?:import('./viewer/surfacePolicy').SurfaceDiagnostics;name:string;color:string;volumeMm3?:number;displayFilter?:VesselDisplayFilter;sourceSupport?:import('./viewer/contracts').TissueDisplaySupport;mask?:ArrayDescriptor;labelSet?:string;labelValue?:number;surface?:{empty:true}|Record<keyof Surface,ArrayDescriptor>}>};
export async function hydrateAnatomy(payload:ViewPayload,api:ResectionApi,cachedImage?:Float32Array,cached?:Pick<ViewerVolume,'labelSets'|'tractography'|'patientContext'|'headSupport'>,studyLayers?:Layer[],appearanceCache?:Map<string,Float32Array>):Promise<ViewerVolume>{
  async function bytes(d:ArrayDescriptor){
    if(!d || d.byteLength>512*1024**2 || d.byteOrder!=='little' || d.order!=='C')throw Error('Unsupported anatomy transfer');
    const b=await api.readAsset(d.assetId);if(b.byteLength!==d.byteLength)throw Error('Incomplete anatomy transfer');
    return (b.byteOffset===0&&b.byteLength===b.buffer.byteLength?b.buffer:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)) as ArrayBuffer;
  }
  const mri=cachedImage??appearanceCache?.get(payload.mri.sha256)??new Float32Array(await bytes(payload.mri));
  if(mri.length!==payload.shape.reduce((a,b)=>a*b,1))throw Error('Native image grid mismatch');
  const appearanceSources=new Map<string,SurfaceAppearanceSource>();
  for(const {signal,...source} of payload.appearanceSources??[]){
    const shape=signal.shape;
    if(signal.dtype!=='float32'||shape.length!==3||shape.some(n=>!Number.isInteger(n)||n<2)||
       shape.reduce((a,b)=>a*b,1)*4!==signal.byteLength)throw Error('Invalid surface appearance grid');
    const affine=rasAffine(source.affine,'RAS+');
    inverseAffine(affine);
    const data=signal.sha256===payload.mri.sha256?mri:appearanceCache?.get(signal.sha256)??new Float32Array(await bytes(signal));
    if(data.length*4!==signal.byteLength)throw Error('Incomplete surface appearance grid');
    appearanceSources.set(source.seriesId,{...source,affine,signalSha256:signal.sha256,shape:shape as Point3,data});
  }
  const compartments:ViewerVolume['compartments']=[];
  for(const layer of payload.compartments){
    const mask=layer.mask?new Uint8Array(await bytes(layer.mask)):new Uint8Array(0);if(!layer.labelSet&&mask.length!==mri.length)throw Error('Overlay grid mismatch');
    let surface:Surface|undefined;
    if(layer.surface&&'empty' in layer.surface){
      if(layer.surface.empty!==true||Object.keys(layer.surface).length!==1)throw Error('Invalid empty anatomy surface');
      surface={positions:new Float32Array(),source:new Float32Array(),normals:new Float32Array(),sourceNormals:new Float32Array(),indices:new Uint32Array()};
    }else if(layer.surface){
      const s=layer.surface;
      const source=new Float32Array(await bytes(s.source)),sourceNormals=new Float32Array(await bytes(s.sourceNormals));
      surface={positions:s.positions.sha256===s.source.sha256?source:new Float32Array(await bytes(s.positions)),source,
        normals:s.normals.sha256===s.sourceNormals.sha256?sourceNormals:new Float32Array(await bytes(s.normals)),sourceNormals,
        indices:new Uint32Array(await bytes(s.indices))};
      if(surface.positions.length%3 || surface.indices.length%3 || surface.source.length!==surface.positions.length || surface.normals.length!==surface.positions.length || surface.sourceNormals.length!==surface.positions.length || surface.indices.some(i=>i>=surface!.positions.length/3))throw Error('Invalid indexed anatomy mesh');
    }
    const declaration=studyLayers?.find(row=>row.id===layer.name);
    const declaredKind=declaration&&studyLayerDisplayKind(declaration);
    const displayKind=declaredKind!==declaration?.kind?declaredKind:layer.displayKind??declaredKind;
    const appearanceSource=layer.alignment!=='source-grid preview'&&layer.sourceSeries&&['gray-matter','cerebral-cortex','cerebellar-cortex','brain','bone','skull'].includes(displayKind??layer.kind??'')
      ?appearanceSources.get(layer.sourceSeries):undefined;
    compartments.push({...layer,displayKind,mask,surface,appearanceSource});
  }
  const tractography=[];for(const t of payload.tractography??[]){const resident=cached?.tractography?.find(r=>r.id===t.id);if(resident){tractography.push(resident);continue;}tractography.push({...t,points:new Float32Array(await bytes(t.points)),offsets:new Uint32Array(await bytes(t.offsets))});}
  const labelSets=[];for(const set of payload.labelSets??[]){const resident=cached?.labelSets?.find(r=>r.id===set.id);if(resident){labelSets.push(resident);continue;}const data=new Uint16Array(await bytes(set.data));if(data.length!==mri.length)throw Error("Categorical grid mismatch");labelSets.push({id:set.id,data});}
  let patientContext:PatientContext|undefined;
  if(payload.patientContext){
    const {sources,signal,shape,affine}=payload.patientContext;
    patientContext={sources};
    if(signal){
      if(!shape||!affine||shape.some(n=>!Number.isInteger(n)||n<2||n>192))throw Error('Invalid patient context grid');
      const known=cached?.patientContext?.field;
      const data=known?.sha256===signal.sha256?known.signal:new Float32Array(await bytes(signal));
      if(data.length!==shape.reduce((a,b)=>a*b,1))throw Error('Incomplete patient context');
      patientContext.field={shape,affine,signal:data,sha256:signal.sha256};
    }
  }
  let headSupport:ViewerVolume['headSupport'];
  if(payload.headSupport){
    const known=cached?.headSupport;
    headSupport=known?.sha256===payload.headSupport.sha256?known:{sha256:payload.headSupport.sha256,data:new Uint8Array(await bytes(payload.headSupport))};
    if(headSupport.data.length!==mri.length)throw Error('Head support differs from its source grid');
  }
  return {...payload,mri,compartments,tractography,labelSets,patientContext,headSupport};
}

export type CranialView = 'planning'|'inspection';
export interface NativeScanState {cranialView?:CranialView;cursor:Point3|null;visible:Record<string,boolean>;settings:ViewerWorkspaceSettings}
export interface RetiredClipping {seriesId:string|null;cutaway:import('./viewer/workspaceSettings').ViewerCutaway}
export interface NativeViewState {schema:'patient-anatomy-view-v1';patientId:string;snapshotId:string;seriesId:string;states:Record<string,NativeScanState>;patientCutaway?:import('./viewer/workspaceSettings').ViewerCutaway;vesselScope?:VesselScope;vesselDisplayVersion?:2;selectedApproachId?:string|null;selectedPlanningTargetId?:string|null;selectedApproachIds?:Record<string,string|null>;selectedStructure?:string|null}
export interface ComponentReview {reviewId:string;layerId:string;components:Array<{id:number;voxels:number;volumeMm3:number;center:Point3}>}
export function isNativeBinding(value:unknown):value is Binding {return !!value&&typeof value==='object'&&'collectionId' in value&&'document' in value;}

export type VesselScope='brain'|'all'|'intracranial'|'excluded';
