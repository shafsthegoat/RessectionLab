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

test('65 source masks compact disjoint tissues while retaining every overlapping tract and target membership',()=>{
 const compartments=[];
 compartments.push({name:'brain',mask:Uint8Array.from({length:128},(_,i)=>i<100?1:0)});
 for(let i=0;i<54;i++)compartments.push({name:`tissue-${i}`,mask:Uint8Array.from({length:128},(_,j)=>j===i||j===i+54?1:0)});
 for(let i=0;i<10;i++)compartments.push({name:`overlap-${i}`,mask:Uint8Array.from({length:128},(_,j)=>j%(i+2)===0?1:0)});
 const volume={shape:[4,4,8],compartments},before=compartments.map(l=>l.mask.slice());
 for(const edit of [false,true]){
  if(edit)compartments[5].mask[4]=0;
  const packed=packAnatomyLabels(volume);assert.equal(packed.data.length,128*4);
  compartments.forEach((layer,i)=>{const [channel,code]=packed.codes[i];for(let j=0;j<128;j++)assert.equal(code<0?(packed.data[j*4+channel]&-code)!==0:packed.data[j*4+channel]===code,!!layer.mask[j]);});
 }
 compartments.forEach((layer,i)=>{if(i!==5)assert.deepEqual(layer.mask,before[i]);});
 assert.throws(()=>packAnatomyLabels({shape:[1,1,1],compartments:Array.from({length:33},()=>({mask:new Uint8Array([1])}))}),/GPU texture budget/);
});
