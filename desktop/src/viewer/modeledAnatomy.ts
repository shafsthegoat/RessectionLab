import type {ViewerVolume} from './contracts';
import type {EngineSurfaceReference} from './engineImplicitSurface';
import type {OperativeVesselSurface} from './operativeVesselSurface';
export interface ModeledAnatomyIdentity {
  readonly source:ViewerVolume;
  readonly mappingKey:string;
  readonly surfaceReference?:EngineSurfaceReference;
  readonly modifiedIndices:readonly number[];
  readonly vesselSurfaces?:ReadonlyMap<string,OperativeVesselSurface>;
}
// Only modeledVolume registers derived source-owned states. An arbitrary
// displayIdentity or timeline index cannot admit a volume to resident updates.
const identities=new WeakMap<ViewerVolume,ModeledAnatomyIdentity>();
export function registerModeledAnatomy(source:ViewerVolume,volume:ViewerVolume,mappingKey:string,modifiedIndices:Iterable<number>,surfaceReference?:EngineSurfaceReference,vesselSurfaces?:ReadonlyMap<string,OperativeVesselSurface>):void {
  if(volume.mri!==source.mri||volume.caseHash!==source.caseHash||JSON.stringify([volume.shape,volume.affine,volume.frame])!==JSON.stringify([source.shape,source.affine,source.frame]))
    throw Error('Modeled anatomy must retain its exact source image and grid');
  const count=source.mri.length,indices=[...new Set(modifiedIndices)].sort((a,b)=>a-b);
  if(indices.some(index=>!Number.isInteger(index)||index<0||index>=count))throw Error('Invalid modeled anatomy change index');
  identities.set(volume,Object.freeze({source,mappingKey,surfaceReference,modifiedIndices:Object.freeze(indices),vesselSurfaces}));
}
export function modeledAnatomy(volume:ViewerVolume):ModeledAnatomyIdentity|undefined {return identities.get(volume);}
export function modeledSourceIdentity(volume:ViewerVolume):ViewerVolume|string {
  const source=identities.get(volume)?.source??volume;
  return source.displayIdentity??source;
}
