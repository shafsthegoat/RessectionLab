import * as THREE from 'three';
/** Uncompressed buffer/attachment estimates; excludes driver/program overhead,
 * default-framebuffer MSAA storage and allocation granularity. Resource counts
 * are reported separately and are never converted into memory bytes. */
export function geometryBufferBytes(geometries:Iterable<THREE.BufferGeometry>):number {
  // WebGLAttributes keys uploads by BufferAttribute (or InterleavedBuffer),
  // not by the underlying ArrayBuffer. Distinct views have distinct GPU stores.
  const buffers=new Set<THREE.BufferAttribute|THREE.InterleavedBuffer>();
  const add=(attribute:THREE.BufferAttribute|THREE.InterleavedBufferAttribute|null)=>{
    if(!attribute)return;
    buffers.add(attribute instanceof THREE.InterleavedBufferAttribute?attribute.data:attribute);
  };
  for(const geometry of geometries){add(geometry.index);Object.values(geometry.attributes).forEach(add);}
  return [...buffers].reduce((sum,buffer)=>sum+buffer.array.byteLength,0);
}
export function renderTargetBytes(target:THREE.WebGLRenderTarget):number {
  const bytes=(type:THREE.TextureDataType)=>type===THREE.FloatType||type===THREE.UnsignedIntType?4:type===THREE.HalfFloatType||type===THREE.UnsignedShortType?2:1;
  const channels=(format:number)=>format===THREE.RedFormat?1:format===THREE.RGFormat?2:format===THREE.RGBFormat?3:4;
  const color=target.textures.reduce((sum,texture)=>sum+bytes(texture.type)*channels(texture.format),0);
  const depth=target.depthBuffer?(target.depthTexture?bytes(target.depthTexture.type):4):0;
  return target.width*target.height*(color+depth)*Math.max(1,target.samples);
}
