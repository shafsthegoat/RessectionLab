import {packAnatomyLabels} from './labelPacking';
self.onmessage=event=>{
  try{
    const packed=packAnatomyLabels(event.data,true);
    self.postMessage(packed,{transfer:[packed.data.buffer]});
  }catch(error){self.postMessage({error:error instanceof Error?error.message:String(error)});}
};
