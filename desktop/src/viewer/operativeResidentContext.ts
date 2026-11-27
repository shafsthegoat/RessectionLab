import {hydrateAnatomy,type NativeAnatomyApi} from '../native-anatomy-data.ts';
import {matchingOperativeSource,operativeSourceIdentity} from './operativeSourceIdentity.ts';
import {mergeResidentLayers} from '../native-layer-residency.ts';
import type {FieldLayer,OperativeField} from '../operative-field';
import type {ResectionApi} from '../types';
import type {ViewerVolume} from './contracts';

// A sparse planning presentation may have loaded neither the cortical nor skull mesh.
export function missingOperativeSurfaces(volume:ViewerVolume|null,field:OperativeField,collectionId:string){
  const identity=operativeSourceIdentity(volume);
  if(!volume||!identity||identity.collectionId!==collectionId||identity.snapshotId!==field.snapshotId)return [];
  const missing:FieldLayer[]=[];
  for(const role of ['bone','grayMatter'] as const){
    const candidates=field.layers.filter(l=>l.sampledVoxels>0&&(role==='bone'?l.role==='bone'&&!l.detail:l.role==='brain'&&l.detail==='grayMatter'));
    if(candidates.length===1&&!candidates[0].adjudication&&!volume.compartments.some(l=>l.name===candidates[0].id))
      missing.push(candidates[0]);
  }
  return missing;
}

export async function loadOperativeSurfaces(api:NativeAnatomyApi&ResectionApi,volume:ViewerVolume,layers:FieldLayer[],signal:AbortSignal){
  const identity=operativeSourceIdentity(volume);
  if(!identity)throw Error('The displayed source has no native scan identity.');
  signal.throwIfAborted();
  const {collectionId,seriesId,imageHash,snapshotId}=identity;
  const result=await api.anatomyView({collectionId,snapshotId,seriesId,layers:layers.map(l=>l.id),original:false,patientModel:true,previewAlignment:true});
  signal.throwIfAborted();
  if(result.mri.sha256!==imageHash)throw Error('The surrounding surfaces and displayed scan have different source identities.');
  const matching=result.compartments.filter(item=>{
    const layer=layers.find(l=>l.id===item.name);
    return layer&&item.surface&&!('empty' in item.surface)&&matchingOperativeSource(layer,item);
  });
  const missing=layers.filter(l=>!matching.some(item=>item.name===l.id));
  const issue=missing.length?'No matching registered surface for '+missing.map(layer=>layer.label??layer.id).join(', ')+'.':undefined;
  if(!matching.length)throw Error(issue);
  const signals=new Map(volume.compartments.flatMap(({appearanceSource:s})=>s?[[s.signalSha256,s.data] as const]:[]));
  const reader={...api,readAsset:async(id:string)=>{
    signal.throwIfAborted();
    const bytes=await api.readAsset(id);
    signal.throwIfAborted();
    return bytes;
  }};
  const incoming=await hydrateAnatomy({...result,compartments:matching},reader,volume.mri,volume,undefined,signals);
  signal.throwIfAborted();
  incoming.displayIdentity=volume.displayIdentity;
  return {volume:mergeResidentLayers(volume,incoming,[...volume.compartments.map(l=>l.name),...matching.map(l=>l.name)]),issue};
}
