import assert from 'node:assert/strict';
import {test} from 'node:test';
import {packAnatomyLabels,materializeLayerMask} from './labelPacking.ts';
test('38 layers retain independent visibility codes and exact categorical membership without duplicate masks',()=>{
 const data=Uint16Array.from({length:64},(_,i)=>i%33),volume={shape:[4,4,4],labelSets:[{id:'regions',data}],compartments:[]};
 for(let i=1;i<=32;i++)volume.compartments.push({name:`region-${i}`,labelSet:'regions',labelValue:i,mask:new Uint8Array()});
 for(let i=0;i<6;i++)volume.compartments.push({name:`binary-${i}`,mask:Uint8Array.from({length:64},(_,j)=>j%(i+2)===0?1:0)});
 const before=data.slice(),packed=packAnatomyLabels(volume);
 for(let i=0;i<38;i++){
  const mask=materializeLayerMask(volume,volume.compartments[i]),[channel,code]=packed.codes[i];
  for(let j=0;j<64;j++)assert.equal(code<0?(packed.data[j*4+channel]&-code)!==0:packed.data[j*4+channel]===code,mask[j]===1);
 }
 assert.deepEqual(data,before);assert.equal(packed.data.byteLength,64*4);
});
test('missing category sources and malformed independent masks cannot silently produce anatomy',()=>{
 assert.throws(()=>packAnatomyLabels({shape:[2,2,2],compartments:[{labelSet:'missing',labelValue:1,mask:new Uint8Array()}]}));
 assert.throws(()=>packAnatomyLabels({shape:[2,2,2],compartments:[{mask:new Uint8Array(2)}]}));
});
