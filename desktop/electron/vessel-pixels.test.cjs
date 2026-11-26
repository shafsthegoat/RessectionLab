'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {pixelClass,countVesselPixels}=require('./vessel-pixels.cjs');
const rgb=hex=>[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16));
test('vessel captures separate red and blue vessels and the amber overlay from other surfaces',()=>{
  for(const color of ['#aa352b','#a32e22','#ef5679','#42668f','#325988'])assert.equal(pixelClass(...rgb(color)),'vessel',color);
  for(const color of ['#ffad54','#ffb454','#c88842'])assert.equal(pixelClass(...rgb(color)),'amber',color);
  assert.notEqual(pixelClass(255,150,120),'amber','a specular highlight on a red wall');
  for(const color of ['#65bfd3','#ece2ca','#a399ad','#d3dce5','#c69891','#000000','#ffffff'])assert.equal(pixelClass(...rgb(color)),null,color);
  const [r,g,b]=rgb('#42668f'),[ar,ag,ab]=rgb('#ffad54');
  assert.deepEqual(countVesselPixels(Uint8Array.from([b,g,r,255,ab,ag,ar,255,0,0,0,255])),{vessel:1,amber:1});
});
