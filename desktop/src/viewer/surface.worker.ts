import {WorkerSurfacePartition} from './engineImplicitSurface';
import type {EnginePartitionRequest} from './engineImplicitSurface';
import {prepareSurface} from './prepareSurface';
import type { Affine, Shape3 } from "./coordinates";

const partition=new WorkerSurfacePartition();
self.onmessage = (
  event: MessageEvent<{ name: string; mask: Uint8Array; shape: Shape3; kind?:string; displaySmoothing?: boolean;partition?:EnginePartitionRequest; affine?: Affine;requestGeneration?:number }>,
) => {
  const { name, mask, shape, kind, displaySmoothing, affine,requestGeneration } = event.data;
  try {
    if(!affine)throw Error('Surface preparation requires the source affine.');
    const result=prepareSurface(mask,shape,affine,kind,displaySmoothing,partition.resolve(event.data.partition));
    const transfer=[result.positions.buffer,result.indices.buffer,result.normals.buffer];
    if(result.smoothPositions)transfer.push(result.smoothPositions.buffer,result.smoothNormals!.buffer);
    self.postMessage({name,requestGeneration,...result},{transfer});
  } catch (error) {
    self.postMessage({
      name,
      requestGeneration,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
