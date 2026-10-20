import {buildSourcePickingBvh,type SourceBvhInput} from './sourcePickingBvhBuild';
self.onmessage=(event:MessageEvent<SourceBvhInput>)=>{
  try{
    const result=buildSourcePickingBvh(event.data);
    const transfer:Transferable[]=[...result.serialized.roots];
    if(result.serialized.indirectBuffer)transfer.push(result.serialized.indirectBuffer.buffer as ArrayBuffer);
    self.postMessage(result,{transfer});
  }catch(cause){self.postMessage({id:event.data.id,error:cause instanceof Error?cause.message:String(cause)});}
};
