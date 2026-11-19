import type {FieldLayer} from '../operative-field';
import type {TissueDisplaySupport,ViewerVolume,VesselDisplayFilter} from './contracts';

/** Native sources carry a structured display receipt. Other rehearsal renderers
 * have their own identity formats and cannot supply this physical frame. */
export function operativeSourceIdentity(volume:ViewerVolume|null){
  if(!volume?.displayIdentity||volume.sourceFrameIssue)return null;
  try{
    const identity:unknown=JSON.parse(volume.displayIdentity);
    if(!Array.isArray(identity)||identity.length<5||!identity.slice(0,4).every(v=>typeof v==='string'&&v.length>0)||identity[4]!==false)return null;
    return {collectionId:identity[0] as string,seriesId:identity[1] as string,imageHash:identity[2] as string,snapshotId:identity[3] as string};
  }catch{return null;}
}

export function matchingOperativeSource(layer:FieldLayer,resident:{
  sourceSeries?:string;alignment?:string;sourceSupport?:TissueDisplaySupport;displayFilter?:VesselDisplayFilter;
}){
  if(resident.displayFilter||resident.sourceSeries!==layer.sourceSeries||
    !['reference','reviewed','source-provided'].includes(resident.alignment??''))return false;
  const actual=resident.sourceSupport,expected=layer.sourceSupport;
  if(expected&&(!actual||actual.status==='unavailable'||expected.version!==actual.version||
    expected.status!==actual.status||expected.rawDataSha256!==actual.rawDataSha256||expected.sourceSha256!==actual.sourceSha256))return false;
  return !actual||actual.sampledDataSha256===layer.sampledDataSha256;
}
