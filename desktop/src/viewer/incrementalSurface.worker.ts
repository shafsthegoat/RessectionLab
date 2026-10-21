import {IncrementalSurfaceEngine} from './incrementalSurface.ts';
import type {IncrementalSurfaceRequest} from './incrementalSurface.ts';
import {packAnatomyLabels} from './labelPacking.ts';
import type {ViewerVolume} from './contracts';
const engine=new IncrementalSurfaceEngine();
self.onmessage=(event:MessageEvent<IncrementalSurfaceRequest|{task:'pack';jobId:number;volume:Pick<ViewerVolume,'shape'|'compartments'|'labelSets'>}>)=>{
  try{
    if('task'in event.data){const packed=packAnatomyLabels(event.data.volume);self.postMessage({jobId:event.data.jobId,task:'pack',packed},{transfer:[packed.data.buffer]});return;}
    const result=engine.prepare(event.data),transfer=[result.positions.buffer,result.indices.buffer,result.normals.buffer];
    if(result.smoothPositions)transfer.push(result.smoothPositions.buffer,result.smoothNormals!.buffer);
    self.postMessage(result,{transfer});
  }catch(cause){self.postMessage({jobId:event.data.jobId,error:cause instanceof Error?cause.message:String(cause)});}
};
