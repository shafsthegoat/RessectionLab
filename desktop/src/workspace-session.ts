import {hydrateCase,validateCaseDescriptor} from './case-data.ts';
import {hydrateDisplaySeries} from './workspace-imaging-data.ts';
import {checkedEpisodeAuthority} from './episode-authority.ts';
import {hydrateDevelopmentEpisode} from './episode-data.ts';
import {sourceFrameDigest} from './source-integrity.ts';
import {rasAffine,transformPoint} from './viewer/coordinates.ts';
import type {CasePayload,ResectionApi,ViewerCase} from './types';
import type {DisplaySeriesView} from './workspace-imaging-types';
import type {EpisodeView} from './episode-data';
import type {ImagingSessionState,ImageViewState,WorkspaceSessionPayload} from './workspace-session-types';
const HASH=/^sha256:[a-f0-9]{64}$/;
function requireValue(value:unknown,message:string):asserts value {if(!value)throw new Error(`Workspace reopen withheld: ${message}`);}
export function defaultImageView(volume:Pick<ViewerCase,'shape'|'affine'|'compartments'>):ImageViewState {
  return {cursor:transformPoint(volume.affine,volume.shape.map(n=>(n-1)/2) as [number,number,number]),
    visibleLayers:Object.fromEntries(volume.compartments.map(layer=>[layer.name,true]))};
}
/** Cursor and layer state are interpreted only in their own source grid. */
export function checkedImageView(source:CasePayload,input:ImageViewState):ImageViewState {
  requireValue(input && typeof input==='object' && !Array.isArray(input),'missing image view state.');
  requireValue(input.visibleLayers && typeof input.visibleLayers==='object' && !Array.isArray(input.visibleLayers),'invalid layer visibility.');
  const labels=source.compartments.map(layer=>layer.name),keys=Object.keys(input.visibleLayers);
  requireValue(keys.length===labels.length && keys.every(key=>labels.includes(key) && typeof input.visibleLayers[key]==='boolean'),'visibility does not match source labels.');
  const affine=rasAffine(source.affine,source.frame);
  let cursor=input.cursor;
  if(cursor===null)cursor=transformPoint(affine,source.shape.map(n=>(n-1)/2) as [number,number,number]);
  requireValue(Array.isArray(cursor)&&cursor.length===3&&cursor.every(Number.isFinite),'invalid RAS cursor.');
  const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
  for(const x of [-.5,source.shape[0]-.5])for(const y of [-.5,source.shape[1]-.5])for(const z of [-.5,source.shape[2]-.5]){
    const corner=transformPoint(affine,[x,y,z]);for(let i=0;i<3;i++){low[i]=Math.min(low[i],corner[i]);high[i]=Math.max(high[i],corner[i]);}
  }
  requireValue(cursor.every((v,i)=>v>=low[i]-1e-5&&v<=high[i]+1e-5),'cursor lies outside its image display bounds.');
  return {cursor:[...cursor],visibleLayers:{...input.visibleLayers}};
}
export function checkedImagingState(primary:CasePayload,series:WorkspaceSessionPayload['displaySeries'],input:ImagingSessionState):ImagingSessionState {
  requireValue(input&&typeof input==='object'&&input.states&&typeof input.states==='object'&&!Array.isArray(input.states),'invalid imaging session state.');
  const sources=new Map([['primary',primary],...series.map(row=>[row.seriesId,row.volume] as [string,CasePayload])]);
  requireValue(sources.size===series.length+1&&series.length<=4,'duplicate or excessive display sources.');
  requireValue(input.selectedSeriesId===null||(typeof input.selectedSeriesId==='string'&&input.selectedSeriesId!=='primary'&&sources.has(input.selectedSeriesId)),'selected image is absent from this workspace.');
  const keys=Object.keys(input.states);
  requireValue(keys.length===sources.size&&keys.every(key=>sources.has(key)),'view state must cover exactly the saved sources.');
  return {selectedSeriesId:input.selectedSeriesId,states:Object.fromEntries([...sources].map(([id,source])=>[id,checkedImageView(source,input.states[id])]))};
}
export interface HydratedWorkspace {
  source:CasePayload;volume:ViewerCase;series:DisplaySeriesView[];imagingState:ImagingSessionState|null;
  episodeView:EpisodeView|null;episodeStep:number;episodeVisible:boolean;session:WorkspaceSessionPayload|null;
}
/** Publish nothing until every source, selection and optional recorded replay is checked.
 * isCurrent rejects a delayed load even when it reopens the same case identity. */
export async function hydrateWorkspace(sourceInput:CasePayload,api:ResectionApi,isCurrent:()=>boolean=()=>true):Promise<HydratedWorkspace> {
  const source=structuredClone(sourceInput),session=source.workspaceSession??null;
  requireValue(source.workspaceSession!==null,'invalid empty session.');
  const current=()=>{if(!isCurrent())throw new DOMException('A newer workspace replaced this load.','AbortError');};
  current();validateCaseDescriptor(source);
  let imagingState:ImagingSessionState|null=null;
  if(session){
    requireValue(session.schema==='integrated-workspace-session-v1'&&HASH.test(session.sessionHash)&&session.referenceCaseHash===source.caseHash,'session identity differs from the primary case.');
    requireValue(Array.isArray(session.displaySeries)&&session.displaySeries.length<=4,'invalid display inventory.');
    for(const row of session.displaySeries){requireValue(row.referenceCaseHash===source.caseHash&&HASH.test(row.seriesId),'stale display source.');validateCaseDescriptor(row.volume);}
    imagingState=checkedImagingState(source,session.displaySeries,session.imagingState);
    requireValue(Array.isArray(session.evidenceInventory)&&session.evidenceInventory.length===session.displaySeries.length+1,'missing evidence inventory.');
    const rows=new Map(session.evidenceInventory.map(row=>[row.seriesId,row]));
    requireValue(rows.size===session.evidenceInventory.length,'duplicate evidence inventory.');
    for(const [id,descriptor] of [['primary',source],...session.displaySeries.map(row=>[row.seriesId,row.volume])] as [string,CasePayload][]){
      const attachment=session.displaySeries.find(item=>item.seriesId===id);
      const kind=attachment?({'none':'source_image','source-provided':'source_annotation','estimated':'estimated_annotation'} as const)[attachment.annotationKind]:'primary_source';
      const modality=attachment?.modality??String(source.metadata.selected_modality??'primary');
      const row=rows.get(id);requireValue(row&&row.sourceKind===kind&&row.modality===modality&&row.sourceCaseHash===descriptor.caseHash&&HASH.test(row.sourceFrameHash)&&row.displayAvailable===true&&
        row.planningInput===(id==='primary')&&row.readiness===(id==='primary'?'primary_case_contract':'display_only_registration_unreviewed')&&
        Array.isArray(row.reasons)&&row.reasons.every(reason=>typeof reason==='string'),'evidence inventory differs from display-only source roles.');
      requireValue(await sourceFrameDigest(descriptor)===row.sourceFrameHash,'evidence frame identity changed.');current();
    }
    requireValue(session.episodeReplay!==undefined,'missing saved replay selection.');
    if(session.episodeReplay){
      const replay=session.episodeReplay;
      checkedEpisodeAuthority(replay.episode,replay.episodeAuthorship,'reopened');
      const qualification=replay.episode.selector==='RL256_ASPIRATION_TRANSFER'?'imported_computational_provenance_unverified':'historical_timings_not_remeasured';
      requireValue(replay.validation==='authoritative_generated_replay'&&replay.accountingQualification===qualification&&replay.episode?.caseHash===source.caseHash&&
        Number.isSafeInteger(replay.frameIndex)&&replay.frameIndex>=0&&replay.frameIndex<replay.episode.replayFrames.length&&typeof replay.visible==='boolean',
        'saved episode has no valid backend replay admission.');
      requireValue(!replay.visible||imagingState.selectedSeriesId===null,'primary replay cannot be displayed on an auxiliary image.');
    }
  }
  current();const volume=await hydrateCase(source,api);current();
  const series:DisplaySeriesView[]=[];
  for(const descriptor of session?.displaySeries??[]){series.push(await hydrateDisplaySeries(source,descriptor,api));current();}
  let episodeView:EpisodeView|null=null;
  if(session?.episodeReplay){episodeView=await hydrateDevelopmentEpisode({case:source,...session.episodeReplay},api,'reopened');current();}
  return {source,volume,series,imagingState,episodeView,episodeStep:session?.episodeReplay?.frameIndex??0,
    episodeVisible:session?.episodeReplay?.visible??false,session};
}
