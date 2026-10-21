import {prepareSurface} from './prepareSurface';
import type { Affine, Shape3 } from "./coordinates";

self.onmessage = (
  event: MessageEvent<{ name: string; mask: Uint8Array; shape: Shape3; kind?:string; displaySmoothing?: boolean; affine?: Affine;requestGeneration?:number }>,
) => {
  const { name, mask, shape, kind, displaySmoothing, affine,requestGeneration } = event.data;
  try {
    if(!affine)throw Error('Surface preparation requires the source affine.');
    const result=prepareSurface(mask,shape,affine,kind,displaySmoothing);
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
