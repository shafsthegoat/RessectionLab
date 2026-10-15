import type {ViewerVolume} from './viewer/contracts';

export function mergeResidentLayers(current:ViewerVolume, incoming:ViewerVolume, order:readonly string[]):ViewerVolume {
  if(!current.displayIdentity||current.displayIdentity!==incoming.displayIdentity||current.mri!==incoming.mri||
    current.caseHash!==incoming.caseHash||current.frame!==incoming.frame||JSON.stringify([current.shape,current.affine])!==JSON.stringify([incoming.shape,incoming.affine]))
    throw Error('Incremental anatomy belongs to a different displayed source');
  const layers=new Map(current.compartments.map(layer=>[layer.name,layer]));
  for(const layer of incoming.compartments)if(!layers.has(layer.name))layers.set(layer.name,layer);
  const sets=new Map((current.labelSets??[]).map(set=>[set.id,set]));
  for(const set of incoming.labelSets??[])if(!sets.has(set.id))sets.set(set.id,set);
  return {...current,alignmentQuality:{...current.alignmentQuality,...incoming.alignmentQuality},packedLabels:undefined,compartments:order.filter(id=>layers.has(id)).map(id=>layers.get(id)!),labelSets:[...sets.values()]};
}

/** Bounded copies avoid one large structured-clone task on the UI thread. */
export async function copyPackingArray<T extends Uint8Array|Uint16Array>(source:T,signal?:AbortSignal):Promise<T> {
  if(signal?.aborted)throw new DOMException('Cancelled','AbortError');
  const copy=(source instanceof Uint16Array?new Uint16Array(source.length):new Uint8Array(source.length)) as T;
  const block=4*1024*1024/source.BYTES_PER_ELEMENT;
  let yielded=performance.now();
  for(let offset=0;offset<source.length;offset+=block){
    if(signal?.aborted)throw new DOMException('Cancelled','AbortError');
    copy.set(source.subarray(offset,offset+block),offset);
    if(performance.now()-yielded>8){await new Promise(resolve=>setTimeout(resolve,0));yielded=performance.now();}
  }
  return copy;
}

export async function prepareNativeLabels(volume:ViewerVolume, signal?:AbortSignal):Promise<ViewerVolume> {
  const compartments:ViewerVolume['compartments']=[];
  for(const {name,mask,labelSet,labelValue,color} of volume.compartments)
    compartments.push({name,mask:await copyPackingArray(mask,signal),labelSet,labelValue,color});
  const labelSets:NonNullable<ViewerVolume['labelSets']>=[];
  for(const set of volume.labelSets??[])labelSets.push({id:set.id,data:await copyPackingArray(set.data,signal)});
  return new Promise((resolve,reject)=>{
    if(signal?.aborted){reject(new DOMException('Cancelled','AbortError'));return;}
    const worker=new Worker(new URL('./viewer/labelPacking.worker.ts',import.meta.url),{type:'module'});
    const done=()=>{worker.terminate();signal?.removeEventListener('abort',abort);};
    const abort=()=>{done();reject(new DOMException('Cancelled','AbortError'));};
    signal?.addEventListener('abort',abort,{once:true});
    worker.onmessage=event=>{done();if(event.data.error)reject(Error(event.data.error));else resolve({...volume,packedLabels:event.data});};
    worker.onerror=()=>{done();reject(Error('Anatomy label preparation failed'));};
    // Do not clone MRI, tractography or meshes, and never detach source masks.
    worker.postMessage({shape:volume.shape,labelSets,compartments},{transfer:[...compartments.map(l=>l.mask.buffer),...labelSets.map(s=>s.data.buffer)]});
  });
}
