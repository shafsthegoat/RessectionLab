import {hydrateAnatomy,type NativeAnatomyApi} from '../native-anatomy-data.ts';
import {matchingOperativeSource,operativeSourceIdentity} from './operativeSourceIdentity.ts';
import {mergeResidentLayers} from '../native-layer-residency.ts';
import type {FieldLayer,OperativeField} from '../operative-field';
import type {ResectionApi} from '../types';
import type {ViewerVolume} from './contracts';

/** A sparse planning presentation may never have loaded the source skull. */
export function missingOperativeSkull(volume:ViewerVolume|null,field:OperativeField,collectionId:string){
  const identity=operativeSourceIdentity(volume);
  if(!volume||!identity||identity.collectionId!==collectionId||identity.snapshotId!==field.snapshotId)return null;
  const bones=field.layers.filter(l=>l.role==='bone'&&l.sampledVoxels>0&&!l.detail);
  if(bones.length!==1||bones[0].adjudication||volume.compartments.some(l=>l.name===bones[0].id))return null;
  return bones[0];
}

export async function loadOperativeSkull(api:NativeAnatomyApi&ResectionApi,volume:ViewerVolume,layer:FieldLayer,signal:AbortSignal){
  const identity=operativeSourceIdentity(volume);
  if(!identity)throw Error('The displayed source has no native scan identity.');
  signal.throwIfAborted();
  const {collectionId,seriesId,imageHash,snapshotId}=identity;
  const result=await api.anatomyView({collectionId,snapshotId,seriesId,layers:[layer.id],original:false,patientModel:true,previewAlignment:true});
  signal.throwIfAborted();
  if(result.mri.sha256!==imageHash)throw Error('The surrounding skull and displayed scan have different source identities.');
  const skull=result.compartments.find(item=>item.name===layer.id);
  if(!skull?.surface||'empty' in skull.surface||!matchingOperativeSource(layer,skull))
    throw Error('The selected scan has no matching registered skull surface.');
  const signals=new Map(volume.compartments.flatMap(({appearanceSource:s})=>s?[[s.signalSha256,s.data] as const]:[]));
  const reader={...api,readAsset:async(id:string)=>{
    signal.throwIfAborted();
    const bytes=await api.readAsset(id);
    signal.throwIfAborted();
    return bytes;
  }};
  const incoming=await hydrateAnatomy(result,reader,volume.mri,volume,undefined,signals);
  signal.throwIfAborted();
  incoming.displayIdentity=volume.displayIdentity;
  return mergeResidentLayers(volume,incoming,[...volume.compartments.map(l=>l.name),layer.id]);
}
